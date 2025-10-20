import os
import sys
import time
import argparse
import json
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np

try:
    import h5py
except Exception:
    h5py = None

try:
    import requests
except Exception:
    requests = None


# ----------------------------
# Download helpers
# ----------------------------
def http_download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    if requests is None:
        raise RuntimeError("`requests` not available to download: %s" % url)
    print(f"[DL] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for ch in r.iter_content(chunk_size=1024 * 1024):
                if ch:
                    f.write(ch)


def download_annb_or_mirror(filename: str, dest_dir: str) -> str:
    """
    Download an ANN-Benchmarks HDF5 by name (e.g., 'sift-128-euclidean.hdf5').
    Tries ann-benchmarks.com first, then a HuggingFace mirror.
    """
    primary = f"https://ann-benchmarks.com/{filename}"
    mirror = f"https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/{filename}?download=true"
    dest = os.path.join(dest_dir, filename)
    # Try primary
    try:
        http_download(primary, dest)
        return dest
    except Exception as e:
        print(f"[WARN] Primary download failed for {filename}: {e}")
    # Try mirror
    http_download(mirror, dest)
    return dest


# ----------------------------
# HDF5 loader
# ----------------------------
def load_annb_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[str]]:
    """
    Load an ANN-Benchmarks HDF5.
    Returns (xb, xq, neighbors_or_None, metric_str_or_None).
    Known keys: 'train' or 'base', 'test' or 'query', 'neighbors', 'distance' (e.g., 'euclidean' or 'angular').
    """
    if h5py is None:
        raise RuntimeError("h5py is required to load HDF5 datasets")
    with h5py.File(path, "r") as f:
        if "train" in f:
            xb = np.array(f["train"], dtype=np.float32)
        else:
            xb = np.array(f["base"], dtype=np.float32)
        if "test" in f:
            xq = np.array(f["test"], dtype=np.float32)
        else:
            xq = np.array(f["query"], dtype=np.float32)
        gt = np.array(f["neighbors"], dtype=np.int64) if "neighbors" in f else None
        metric = None
        if "distance" in f.attrs:
            metric = f.attrs["distance"]
            if isinstance(metric, bytes):
                metric = metric.decode("utf-8")
        return xb, xq, gt, metric


# ----------------------------
# Metrics & preprocessing
# ----------------------------
def maybe_unit_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def prepare_for_metric(xb: np.ndarray, xq: np.ndarray, metric: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Normalize vectors for angular so we can use L2 on the unit sphere.
    Returns possibly transformed (xb, xq) and the faiss metric name we will use.
    """
    if metric and metric.lower() in ("angular", "cosine", "ip", "inner_product"):
        xb2 = maybe_unit_normalize(xb.astype(np.float32, copy=False))
        xq2 = maybe_unit_normalize(xq.astype(np.float32, copy=False))
        return xb2, xq2, "angular"
    else:
        return xb.astype(np.float32, copy=False), xq.astype(np.float32, copy=False), "euclidean"


# ----------------------------
# Ground truth
# ----------------------------
def compute_gt_top1(xb: np.ndarray, xq: np.ndarray) -> np.ndarray:
    d = xb.shape[1]
    exact = faiss.IndexFlatL2(d)
    exact.add(xb)
    _, I = exact.search(xq, 1)
    return I[:, 0]


def recall_at1(top1: np.ndarray, gt_top1: np.ndarray) -> float:
    return float((top1 == gt_top1).sum()) / float(gt_top1.shape[0])


def time_search(index, xq: np.ndarray, k: int = 1, warmup: int = 10):
    nq = xq.shape[0]
    wq = min(warmup, nq)
    if wq > 0:
        index.search(xq[:wq], k)
    t0 = time.time()
    D, I = index.search(xq, k)
    total = time.time() - t0
    return I, total


# ----------------------------
# Main benchmark
# ----------------------------
def run_part2_annb(out_dir: str,
                   m_list: List[int],
                   ef_search: int,
                   ef_construction: int,
                   datasets: Optional[List[str]] = None):
    """
    datasets: list of dataset names (HDF5 filenames) to use. If None, defaults to four diverse ones:
      - sift-128-euclidean.hdf5
      - gist-960-euclidean.hdf5
      - fashion-mnist-784-euclidean.hdf5
      - glove-25-angular.hdf5  (we unit-normalize and use L2)
    """
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "data_cache")
    os.makedirs(cache, exist_ok=True)

    if datasets is None:
        datasets = [
            "sift-128-euclidean.hdf5",
            "gist-960-euclidean.hdf5",
            "fashion-mnist-784-euclidean.hdf5",
            "glove-25-angular.hdf5",
        ]

    all_rows = []

    for name in datasets:
        print(f"\n[DATASET] {name}")
        local_path = os.path.join(cache, name)
        if not os.path.exists(local_path):
            # Try to download (primary ann-benchmarks, fallback HF)
            local_path = download_annb_or_mirror(name, cache)

        xb, xq, gt, metric = load_annb_hdf5(local_path)
        xb, xq, metric_used = prepare_for_metric(xb, xq, metric or ("angular" if "angular" in name else "euclidean"))
        n, d = xb.shape
        nq = xq.shape[0]
        print(f"  xb: {xb.shape}, xq: {xq.shape}, metric={metric_used}")

        # Ground truth
        gt_top1 = gt[:, 0] if gt is not None and gt.size > 0 else compute_gt_top1(xb, xq)

        # Sweep M
        ds_rows = []
        for M in m_list:
            print(f"  [HNSW] Building M={M}, efConstruction={ef_construction} ...")
            index = faiss.IndexHNSWFlat(d, M)  # L2; angular handled by unit normalization
            index.hnsw.efConstruction = ef_construction

            t0 = time.time()
            index.add(xb)
            build_s = time.time() - t0
            print(f"    Build: {build_s:.2f}s on {n} vectors")

            index.hnsw.efSearch = ef_search
            I, total_s = time_search(index, xq, k=1, warmup=10)
            approx_top1 = I[:, 0]
            rec = recall_at1(approx_top1, gt_top1)
            qps = float(nq) / total_s if total_s > 0 else 0.0
            print(f"    efSearch={ef_search} | recall@1={rec:.4f} | QPS={qps:.2f}")

            row = {
                "dataset": name,
                "M": M,
                "efConstruction": ef_construction,
                "efSearch": ef_search,
                "build_time_s": build_s,
                "recall_at1": rec,
                "qps": qps,
                "n_base": int(n),
                "n_query": int(nq),
                "dim": int(d),
                "metric": metric_used,
            }
            ds_rows.append(row)
            all_rows.append(row)

        # Save per-dataset CSV
        import csv
        csv_path = os.path.join(out_dir, f"part2_{os.path.splitext(name)[0]}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ds_rows[0].keys()))
            w.writeheader()
            for r in ds_rows:
                w.writerow(r)
        print(f"  [SAVE] {csv_path}")

    # Combined CSV
    if all_rows:
        import csv
        csv_all = os.path.join(out_dir, "part2_all.csv")
        with open(csv_all, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"[SAVE] {csv_all}")

    # Plots
    import matplotlib
    import matplotlib.pyplot as plt

    # Group rows by dataset
    from collections import defaultdict
    g = defaultdict(list)
    for r in all_rows:
        g[r["dataset"]].append(r)

    # Plot 1: QPS vs Recall (curves per dataset; annotate M)
    plt.figure()
    for dsname, rows in g.items():
        xs = [float(r["recall_at1"]) for r in rows]
        ys = [float(r["qps"]) for r in rows]
        labels = [f"M={r['M']}" for r in rows]
        plt.plot(xs, ys, marker='o', label=dsname)
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y))
    plt.xlabel("1-Recall@1")
    plt.ylabel("QPS")
    plt.title("Part 2 (ANN-Benchmarks): QPS vs Recall across datasets")
    plt.legend()
    plt.tight_layout()
    fig1 = os.path.join(out_dir, "part2_qps_vs_recall.png")
    plt.savefig(fig1, dpi=150)
    print(f"[PLOT] {fig1}")

    # Plot 2: Build time vs Recall
    plt.figure()
    for dsname, rows in g.items():
        xs = [float(r["recall_at1"]) for r in rows]
        ys = [float(r["build_time_s"]) for r in rows]
        labels = [f"M={r['M']}" for r in rows]
        plt.plot(xs, ys, marker='s', label=dsname)
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y))
    plt.xlabel("1-Recall@1")
    plt.ylabel("Index Build Time (s)")
    plt.title("Part 2 (ANN-Benchmarks): Build Time vs Recall across datasets")
    plt.legend()
    plt.tight_layout()
    fig2 = os.path.join(out_dir, "part2_buildtime_vs_recall.png")
    plt.savefig(fig2, dpi=150)
    print(f"[PLOT] {fig2}")


def main():
    ap = argparse.ArgumentParser(description="MP2 Part 2 (Standalone) — HNSW scalability with ANN-Benchmarks HDF5 datasets")
    ap.add_argument("--out", type=str, default="results_part2_annb", help="Output directory")
    ap.add_argument("--m_list", type=int, nargs="*", default=[4, 8, 12, 24, 48], help="HNSW M values")
    ap.add_argument("--ef_search", type=int, default=100, help="HNSW efSearch")
    ap.add_argument("--ef_construction", type=int, default=200, help="HNSW efConstruction")
    ap.add_argument("--datasets", type=str, nargs="*", default=None,
                    help="Names of ANN-Benchmarks HDF5 files, e.g. sift-128-euclidean.hdf5 gist-960-euclidean.hdf5")
    args = ap.parse_args()

    run_part2_annb(out_dir=args.out,
                   m_list=args.m_list,
                   ef_search=args.ef_search,
                   ef_construction=args.ef_construction,
                   datasets=args.datasets)


if __name__ == "__main__":
    main()
