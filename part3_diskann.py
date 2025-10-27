import os
import sys
import time
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import faiss

# Optional deps
try:
    import h5py
except Exception:
    h5py = None

try:
    import requests
except Exception:
    requests = None

# DiskANN Python bindings (optional; required for DiskANN path)
# pip install diskannpy
try:
    import diskannpy as dap  # type: ignore
except Exception:
    dap = None


# ===========================
# Download helpers
# ===========================
def http_download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    if requests is None:
        raise RuntimeError("`requests` not available to download: %s" % url)
    print(f"[DL] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for ch in r.iter_content(chunk_size=1024 * 1024):
                if ch:
                    f.write(ch)


def download_annb_or_mirror(filename: str, dest_dir: str) -> str:
    primary = f"https://ann-benchmarks.com/{filename}"
    mirror = f"https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/{filename}?download=true"
    dest = os.path.join(dest_dir, filename)
    try:
        http_download(primary, dest)
        return dest
    except Exception as e:
        print(f"[WARN] Primary download failed for {filename}: {e}")
    http_download(mirror, dest)
    return dest


# ===========================
# HDF5 loader & metric prep
# ===========================
def load_annb_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
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
        metric = "euclidean"
        if "distance" in f.attrs:
            m = f.attrs["distance"]
            if isinstance(m, bytes):
                m = m.decode("utf-8")
            metric = m
    return xb, xq, gt, metric


def maybe_unit_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def prepare_for_metric(xb: np.ndarray, xq: np.ndarray, metric: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if metric and metric.lower() in ("angular", "cosine", "ip", "inner_product"):
        xb2 = maybe_unit_normalize(xb.astype(np.float32, copy=False))
        xq2 = maybe_unit_normalize(xq.astype(np.float32, copy=False))
        return xb2, xq2, "angular"
    return xb.astype(np.float32, copy=False), xq.astype(np.float32, copy=False), "euclidean"


# ===========================
# Ground truth / metrics
# ===========================
def compute_gt_top1(xb: np.ndarray, xq: np.ndarray) -> np.ndarray:
    d = xb.shape[1]
    exact = faiss.IndexFlatL2(d)
    exact.add(xb)
    _, I = exact.search(xq, 1)
    return I[:, 0]


def recall_at1(top1: np.ndarray, gt_top1: np.ndarray) -> float:
    return float((top1 == gt_top1).sum()) / float(gt_top1.shape[0])


# ===========================
# HNSW path (FAISS)
# ===========================
def hnsw_latency_recall(
    xb: np.ndarray,
    xq: np.ndarray,
    gt_top1: np.ndarray,
    M_values: List[int],
    ef_construction: int,
    ef_search_list: List[int],
) -> List[Dict]:
    # --- NEW: sanitize inputs ---
    if isinstance(ef_construction, (list, tuple)):
        ef_construction = ef_construction[0]
    try:
        ef_construction = int(ef_construction)
    except Exception:
        raise ValueError(f"ef_construction must be an integer; got {type(ef_construction)}={ef_construction}")

    # normalize ef_search_list to a list of ints
    if not isinstance(ef_search_list, (list, tuple)):
        ef_search_list = [ef_search_list]
    ef_search_list = [int(e) for e in ef_search_list]

    # normalize M_values to a list of ints
    if not isinstance(M_values, (list, tuple)):
        M_values = [M_values]
    M_values = [int(m) for m in M_values]
    # --- end sanitize ---

    d = xb.shape[1]
    results = []
    for M in M_values:
        index = faiss.IndexHNSWFlat(d, int(M))
        # either direct attribute or via ParameterSpace (both are fine)
        try:
            index.hnsw.efConstruction = int(ef_construction)
        except TypeError:
            ps = faiss.ParameterSpace()
            ps.set_index_parameter(index, "efConstruction", int(ef_construction))

        # build
        t0 = time.time()
        index.add(xb)
        build_s = time.time() - t0

        for ef in ef_search_list:
            try:
                index.hnsw.efSearch = int(ef)
            except TypeError:
                ps = faiss.ParameterSpace()
                ps.set_index_parameter(index, "efSearch", int(ef))

            # warmup + timed search
            wq = min(10, xq.shape[0])
            if wq > 0:
                index.search(xq[:wq], 1)
            t1 = time.time()
            D, I = index.search(xq, 1)
            elapsed = time.time() - t1
            latency_ms = (elapsed / max(1, xq.shape[0])) * 1000.0
            rec = recall_at1(I[:, 0], gt_top1)
            results.append({
                "algo": "HNSW",
                "M": int(M),
                "efConstruction": int(ef_construction),
                "efSearch": int(ef),
                "latency_ms": latency_ms,
                "recall_at1": rec,
                "build_time_s": build_s,
            })
            print(f"[HNSW] M={int(M):>2} efC={int(ef_construction):>3} efS={int(ef):>3} | "
                  f"latency={latency_ms:.3f} ms | recall@1={rec:.4f} | build={build_s:.2f}s")
    return results



# ===========================
# DiskANN path (diskannpy) with memory and disk sweeps
# ===========================
def _make_disk_index(distance_metric: str, index_dir: str, memory_budget_gb: float):
    """
    Construct a DiskANN DiskIndex with a memory/cache budget.
    diskannpy APIs have evolved; try multiple constructor signatures gracefully.
    """
    # Common kwarg names seen across versions:
    candidates = [
        dict(distance_metric=distance_metric, index_directory=index_dir, search_cache_budget_gb=memory_budget_gb),
        dict(distance_metric=distance_metric, index_directory=index_dir, ram_budget_gb=memory_budget_gb),
        dict(distance_metric=distance_metric, index_directory=index_dir, mem_cache_size_gb=memory_budget_gb),
        dict(distance_metric=distance_metric, index_directory=index_dir)  # fallback: no budget control
    ]
    last_err = None
    for kwargs in candidates:
        try:
            return dap.DiskIndex(**kwargs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not construct DiskIndex with any known memory-budget kwarg. Last error: {last_err}")


def diskann_latency_recall(
    xb: np.ndarray,
    xq: np.ndarray,
    gt_top1: np.ndarray,
    distance: str,
    index_root: str,
    R_values: List[int],
    L_values: List[int],
    search_L_values: List[int],
    pq_disk_bytes_list: List[int],
    memory_budget_gb_list: List[float],
    num_threads: int = max(1, os.cpu_count() or 1),
) -> List[Dict]:
    if dap is None:
        raise RuntimeError("diskannpy is not installed. Install via: pip install diskannpy (Linux, py310/py311).")
    os.makedirs(index_root, exist_ok=True)

    metric = "l2" if distance == "euclidean" else "cosinesimil"
    results = []

    for pq_bytes in pq_disk_bytes_list:
        for R in R_values:
            for L in L_values:
                idx_dir = os.path.join(index_root, f"R{R}_L{L}_pq{pq_bytes}")
                os.makedirs(idx_dir, exist_ok=True)

                # Build index on disk (controls DISK SIZE via pq_disk_bytes)
                build_t0 = time.time()
                index_obj = dap.build_disk_index(
                    data=xb,
                    distance_metric=metric,
                    index_directory=idx_dir,
                    graph_degree=R,
                    complexity=L,
                    alpha=1.2,
                    num_threads=num_threads,
                    pq_disk_bytes=pq_bytes,
                )
                build_s = time.time() - build_t0

                # Ensure we have a DiskIndex instance for searching
                if hasattr(index_obj, "search"):
                    # Some versions return a ready-to-search object
                    build_index_path = idx_dir
                else:
                    build_index_path = idx_dir

                # Now sweep in-memory budget (controls IN-MEMORY SIZE) and search complexity
                for mem_gb in memory_budget_gb_list:
                    # Reopen the on-disk index with a given cache/memory budget
                    try:
                        disk_index = _make_disk_index(metric, build_index_path, mem_gb)
                    except Exception as e:
                        print(f"[WARN] Could not set memory budget={mem_gb} GB on DiskIndex; proceeding without explicit budget. ({e})")
                        disk_index = dap.DiskIndex(distance_metric=metric, index_directory=build_index_path)

                    # Warm-up with lowest search_L
                    try:
                        _ = disk_index.search(xq[: min(10, xq.shape[0])], 1, complexity=search_L_values[0])
                    except TypeError:
                        # Some versions require 'L' instead of 'complexity'
                        _ = disk_index.search(xq[: min(10, xq.shape[0])], 1, L=search_L_values[0])

                    for search_L in search_L_values:
                        t0 = time.time()
                        try:
                            I = disk_index.search(xq, 1, complexity=search_L)
                        except TypeError:
                            I = disk_index.search(xq, 1, L=search_L)
                        elapsed = time.time() - t0

                        latency_ms = (elapsed / max(1, xq.shape[0])) * 1000.0
                        rec = recall_at1(I.reshape(-1), gt_top1)
                        results.append({
                            "algo": "DiskANN",
                            "R": R,
                            "L_build": L,
                            "L_search": search_L,
                            "pq_disk_bytes": pq_bytes,
                            "memory_budget_gb": mem_gb,
                            "latency_ms": latency_ms,
                            "recall_at1": rec,
                            "build_time_s": build_s,
                        })
                        print(f"[DiskANN] R={R:>2} Lb={L:>3} Ls={search_L:>3} pq={pq_bytes:>2}B mem={mem_gb:>4.1f}GB | "
                              f"lat={latency_ms:.3f} ms | rec@1={rec:.4f} | build={build_s:.2f}s")

                    # close / cleanup between mem budgets if needed
                    try:
                        del disk_index
                    except Exception:
                        pass

                # cleanup between builds
                try:
                    del index_obj
                except Exception:
                    pass

    return results


# ===========================
# Orchestration
# ===========================
def run_part3(
    out_dir: str,
    dataset: str,
    limit_base: int,
    limit_queries: int,
    hnsw_M_list: List[int],
    hnsw_efC: int,
    hnsw_efS_list: List[int],
    diskann_R_list: List[int],
    diskann_L_build_list: List[int],
    diskann_L_search_list: List[int],
    diskann_pq_bytes_list: List[int],
    diskann_mem_gb_list: List[float],
):
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "data_cache")
    os.makedirs(cache, exist_ok=True)

    # 1) Load dataset
    local = os.path.join(cache, dataset)
    if not os.path.exists(local):
        local = download_annb_or_mirror(dataset, cache)
    xb, xq, gt, metric = load_annb_hdf5(local)

    # Optional sub-sampling
    if limit_base and xb.shape[0] > limit_base:
        xb = xb[:limit_base]
        # NOTE: provided GT in ANN-B HDF5 is for the full base; restricting base may invalidate GT indices.
        # For strict correctness when sub-sampling base, recompute exact GT:
        gt = None
    if limit_queries and xq.shape[0] > limit_queries:
        xq = xq[:limit_queries]
        if gt is not None and gt.shape[0] >= limit_queries:
            gt = gt[:limit_queries]

    xb, xq, metric_used = prepare_for_metric(xb, xq, metric or ("angular" if "angular" in dataset else "euclidean"))
    gt_top1 = gt[:, 0] if gt is not None and gt.size > 0 else compute_gt_top1(xb, xq)

    # 2) HNSW curve
    print("\n=== HNSW curve ===")
    hnsw_rows = hnsw_latency_recall(
        xb=xb,
        xq=xq,
        gt_top1=gt_top1,
        M_values=hnsw_M_list,
        ef_construction=hnsw_efC,
        ef_search_list=hnsw_efS_list,
    )

    # 3) DiskANN curve with memory/disk sweeps
    print("\n=== DiskANN curve (with memory/disk sweeps) ===")
    diskann_rows = []
    try:
        diskann_rows = diskann_latency_recall(
            xb=xb,
            xq=xq,
            gt_top1=gt_top1,
            distance=metric_used,
            index_root=os.path.join(out_dir, "diskann_indexes"),
            R_values=diskann_R_list,
            L_values=diskann_L_build_list,
            search_L_values=diskann_L_search_list,
            pq_disk_bytes_list=diskann_pq_bytes_list,
            memory_budget_gb_list=diskann_mem_gb_list,
        )
    except RuntimeError as e:
        print(f"[SKIP DiskANN] {e}")
        print("To run DiskANN on macOS/Python 3.13, use a Linux environment (VM/Colab) with Python 3.10/3.11.")

    # 4) Save CSVs
    import csv
    csv_h = os.path.join(out_dir, "part3_hnsw.csv")
    with open(csv_h, "w", newline="") as f:
        if hnsw_rows:
            keys = list(hnsw_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in hnsw_rows:
                w.writerow(r)
    print(f"[SAVE] {csv_h}")

    if diskann_rows:
        csv_d = os.path.join(out_dir, "part3_diskann.csv")
        with open(csv_d, "w", newline="") as f:
            keys = list(diskann_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in diskann_rows:
                w.writerow(r)
        print(f"[SAVE] {csv_d}")

    # 5) Plot latency (y) vs recall (x) on one chart
    import matplotlib.pyplot as plt

    plt.figure()
    # HNSW points
    if hnsw_rows:
        xs = [r["recall_at1"] for r in hnsw_rows]
        ys = [r["latency_ms"] for r in hnsw_rows]
        labels = [f"M={r['M']},efS={r['efSearch']}" for r in hnsw_rows]
        plt.plot(xs, ys, marker='o', label="HNSW")
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y))
    # DiskANN points
    if diskann_rows:
        xs = [r["recall_at1"] for r in diskann_rows]
        ys = [r["latency_ms"] for r in diskann_rows]
        labels = [f"R={r['R']},Lb={r['L_build']},Ls={r['L_search']},pq={r['pq_disk_bytes']},mem={r['memory_budget_gb']}GB" for r in diskann_rows]
        plt.plot(xs, ys, marker='s', label="DiskANN")
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y))

    plt.xlabel("1-Recall@1")
    plt.ylabel("Latency (ms/query)")
    plt.title(f"Part 3: Latency vs Recall — {dataset}")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "part3_latency_vs_recall.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[PLOT] {fig_path}")


def main():
    ap = argparse.ArgumentParser(description="MP2 Part 3 — DiskANN vs HNSW (Latency–Recall) with memory & disk sweeps")
    ap.add_argument("--out", type=str, default="results_part3", help="Output directory")
    ap.add_argument("--dataset", type=str, default="sift-128-euclidean.hdf5",
                    help="ANN-Benchmarks HDF5 filename (default: SIFT1M)")
    ap.add_argument("--limit_base", type=int, default=0, help="Truncate base vectors for speed (0 = all)")
    ap.add_argument("--limit_queries", type=int, default=0, help="Truncate query vectors for speed (0 = all)")

    # HNSW knobs
    ap.add_argument("--hnsw_M", type=int, nargs="*", default=[16, 32], help="HNSW M values")
    ap.add_argument("--hnsw_efC", type=int, default=[50, 100, 200], help="HNSW efConstruction")
    ap.add_argument("--hnsw_efS", type=int, nargs="*", default=[10, 50, 100, 200], help="HNSW efSearch values")

    # DiskANN knobs
    ap.add_argument("--diskann_R", type=int, nargs="*", default=[16, 32], help="DiskANN graph degree (R)")
    ap.add_argument("--diskann_L_build", type=int, nargs="*", default=[50, 100, 200], help="DiskANN build complexity (L)")
    ap.add_argument("--diskann_L_search", type=int, nargs="*", default=[10, 50, 100, 200], help="DiskANN search complexity (Ls)")
    ap.add_argument("--diskann_pq_bytes", type=int, nargs="*", default=[0, 8, 16, 32], help="Disk bytes per vector for PQ compression")
    ap.add_argument("--diskann_mem_gb", type=float, nargs="*", default=[0.5, 1.0, 2.0], help="Approx search cache / memory budget in GB")

    args = ap.parse_args()

    run_part3(
        out_dir=args.out,
        dataset=args.dataset,
        limit_base=args.limit_base,
        limit_queries=args.limit_queries,
        hnsw_M_list=args.hnsw_M,
        hnsw_efC=args.hnsw_efC,
        hnsw_efS_list=args.hnsw_efS,
        diskann_R_list=args.diskann_R,
        diskann_L_build_list=args.diskann_L_build,
        diskann_L_search_list=args.diskann_L_search,
        diskann_pq_bytes_list=args.diskann_pq_bytes,
        diskann_mem_gb_list=args.diskann_mem_gb,
    )


if __name__ == "__main__":
    main()
