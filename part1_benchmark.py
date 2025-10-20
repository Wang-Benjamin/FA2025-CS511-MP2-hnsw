import os
import sys
import time
import argparse
import faiss
import h5py
import numpy as np
import requests
import matplotlib
import matplotlib.pyplot as plt

DATA_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
DATA_FILENAME = "sift-128-euclidean.hdf5"
HNSW_M = 32
HNSW_EFSEARCH_LIST = [10, 50, 100, 200]
LSH_NBITS_LIST = [32, 64, 512, 768]


def _download_file(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        return
    if requests is None:
        raise RuntimeError(
            "Dataset missing and 'requests' not available. "
            f"Manually place the file at {dest_path}."
        )
    print(f"[INFO] Downloading dataset from {url} -> {dest_path} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk = 1024 * 1024
        with open(dest_path, "wb") as f:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if chunk_bytes:
                    f.write(chunk_bytes)
                    downloaded += len(chunk_bytes)
                    if total > 0:
                        pct = downloaded * 100 // total
                        sys.stdout.write(f"\r[INFO] Downloaded {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct}%)")
                        sys.stdout.flush()
    print("\n[INFO] Download complete.")


def _ensure_dataset_here() -> str:
    data_path = os.path.join(os.path.dirname(__file__), DATA_FILENAME)
    if not os.path.exists(data_path):
        try:
            _download_file(DATA_URL, data_path)
        except Exception as e:
            print(f"[WARN] Could not auto-download dataset: {e}")
            print(f"[HINT] Manually download '{DATA_URL}' and place it at '{data_path}'.")
            raise
    return data_path


def _load_sift_hdf5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if "train" not in f or "test" not in f:
            raise KeyError(
                f"Unexpected HDF5 structure. Found keys: {keys}. Expected 'train' and 'test'."
            )
        xb = np.array(f["train"], dtype=np.float32)
        xq = np.array(f["test"], dtype=np.float32)
        return xb, xq


def _build_exact_flatl2(xb: np.ndarray) -> faiss.IndexFlatL2:
    d = xb.shape[1]
    exact = faiss.IndexFlatL2(d)
    exact.add(xb)
    return exact


def _compute_ground_truth_top1(xb: np.ndarray, xq: np.ndarray) -> np.ndarray:
    exact = _build_exact_flatl2(xb)
    _, I = exact.search(xq, 1)
    return I[:, 0]


def _recall_at1(approx_top1: np.ndarray, gt_top1: np.ndarray) -> float:
    return float((approx_top1 == gt_top1).sum()) / float(gt_top1.shape[0])


def _time_search(index, xq: np.ndarray, topk: int = 1, warmup: int = 5):
    nq = xq.shape[0]
    wq = min(warmup, nq)
    if wq > 0:
        index.search(xq[:wq], topk)
    t0 = time.time()
    D, I = index.search(xq, topk)
    total = time.time() - t0
    return I, total


# ----------------------------
# Part 1 benchmark
# ----------------------------
def run_part1(output_dir: str = None,
              hnsw_m: int = HNSW_M,
              hnsw_ef_list = None,
              lsh_nbits_list = None):
    if hnsw_ef_list is None:
        hnsw_ef_list = HNSW_EFSEARCH_LIST
    if lsh_nbits_list is None:
        lsh_nbits_list = LSH_NBITS_LIST

    data_path = _ensure_dataset_here()
    xb, xq = _load_sift_hdf5(data_path)
    if xb.dtype != np.float32:
        xb = xb.astype(np.float32)
    if xq.dtype != np.float32:
        xq = xq.astype(np.float32)

    d = xb.shape[1]
    nq = xq.shape[0]
    print(f"[INFO] [Part 1] Loaded xb={xb.shape}, xq={xq.shape}, dim={d}")

    print("[INFO] [Part 1] Computing exact ground-truth top-1 ... (one-time)")
    gt_top1 = _compute_ground_truth_top1(xb, xq)

    rows = []

    # HNSW
    print("[INFO] [Part 1] Benchmarking HNSW...")
    hnsw_index = faiss.IndexHNSWFlat(d, hnsw_m)
    hnsw_index.hnsw.efConstruction = 200  # keep fixed; vary efSearch only
    hnsw_index.add(xb)
    for ef in hnsw_ef_list:
        hnsw_index.hnsw.efSearch = ef
        I, total_s = _time_search(hnsw_index, xq, topk=1, warmup=10)
        approx_top1 = I[:, 0]
        recall = _recall_at1(approx_top1, gt_top1)
        qps = float(nq) / total_s if total_s > 0 else 0.0
        print(f"[HNSW] efSearch={ef:>4} | recall@1={recall:.4f} | QPS={qps:.2f}")
        rows.append(["HNSW", f"M={hnsw_m},ef={ef}", recall, qps])

    # LSH
    print("[INFO] [Part 1] Benchmarking LSH...")
    for nbits in lsh_nbits_list:
        lsh = faiss.IndexLSH(d, nbits)
        lsh.add(xb)
        I, total_s = _time_search(lsh, xq, topk=1, warmup=10)
        approx_top1 = I[:, 0]
        recall = _recall_at1(approx_top1, gt_top1)
        qps = float(nq) / total_s if total_s > 0 else 0.0
        print(f"[LSH ] nbits={nbits:>4} | recall@1={recall:.4f} | QPS={qps:.2f}")
        rows.append(["LSH", f"nbits={nbits}", recall, qps])

    # Save CSV
    out_dir = output_dir or os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "part1_results.csv")
    with open(csv_path, "w") as f:
        f.write("algo,params,recall_at1,qps\n")
        for r in rows:
            f.write("{},{},{:.6f},{:.6f}\n".format(*r))
    print(f"[INFO] [Part 1] Wrote results to {csv_path}")

    # Plot: QPS (y) vs 1-Recall@1 (x)
    plt.figure()
    hnsw_pts = [(rec, q, params) for algo, params, rec, q in rows if algo == "HNSW"]
    lsh_pts  = [(rec, q, params) for algo, params, rec, q in rows if algo == "LSH"]
    if hnsw_pts:
        xs = [p[0] for p in hnsw_pts]
        ys = [p[1] for p in hnsw_pts]
        plt.plot(xs, ys, marker='o', label="HNSW")
        for x, y, label in hnsw_pts:
            plt.annotate(label, (x, y))
    if lsh_pts:
        xs = [p[0] for p in lsh_pts]
        ys = [p[1] for p in lsh_pts]
        plt.plot(xs, ys, marker='s', label="LSH")
        for x, y, label in lsh_pts:
            plt.annotate(label, (x, y))
    plt.xlabel("1-Recall@1")
    plt.ylabel("QPS")
    plt.title("Part 1: QPS vs 1-Recall@1 (HNSW vs LSH)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "part1_qps_vs_recall.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[INFO] [Part 1] Wrote plot to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="MP2 Part 1 (Standalone)")
    parser.add_argument("--out", type=str, default=None, help="Output directory for artifacts")
    parser.add_argument("--hnsw_m", type=int, default=HNSW_M, help="HNSW M parameter")
    parser.add_argument("--ef", type=int, nargs="*", default=None, help="HNSW efSearch list")
    parser.add_argument("--nbits", type=int, nargs="*", default=None, help="LSH nbits list")
    args = parser.parse_args()

    hnsw_ef_list = args.ef if args.ef is not None else None
    lsh_nbits_list = args.nbits if args.nbits is not None else None

    run_part1(output_dir=args.out, hnsw_m=args.hnsw_m,
              hnsw_ef_list=hnsw_ef_list, lsh_nbits_list=lsh_nbits_list)


if __name__ == "__main__":
    main()
