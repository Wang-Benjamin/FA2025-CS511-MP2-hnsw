import faiss
import h5py
import numpy as np
import os
import requests
import sys
import time

DATA_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
DATA_FILENAME = "sift-128-euclidean.hdf5"
K = 10  # top-k neighbors to retrieve
M = 16  # HNSW graph degree
EF_CONSTRUCTION = 200
EF_SEARCH = 200

def _download_file(url: str, dest_path: str) -> None:
    """
    Download a file from `url` to `dest_path` if it does not already exist.
    Uses streaming download to handle large files robustly.
    """
    if os.path.exists(dest_path):
        return
    if requests is None:
        raise RuntimeError(
            "The dataset file is missing and the 'requests' package is not available. "
            "Install requests or manually place the HDF5 at '{}'.".format(dest_path)
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

def _load_sift_hdf5(h5_path: str):
    """
    Load ANN-Benchmarks SIFT-128 Euclidean HDF5 file.
    Returns (xb, xq) where:
      - xb: database/base vectors (float32)
      - xq: query vectors (float32)
    Expected datasets in file: 'train' (base) and 'test' (queries).
    """
    with h5py.File(h5_path, "r") as f:
        # Inspect available keys and pick the expected ones
        keys = list(f.keys())
        # Most ANN-Benchmarks HDF5 files use 'train' for base vectors and 'test' for queries.
        if "train" not in f or "test" not in f:
            raise KeyError(
                f"Unexpected HDF5 structure. Found keys: {keys}. "
                "Expected datasets named 'train' and 'test'."
            )
        xb = np.array(f["train"], dtype=np.float32)
        xq = np.array(f["test"], dtype=np.float32)
        return xb, xq

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    # 1) Ensure dataset present (download if needed)
    data_path = os.path.join(os.path.dirname(__file__), DATA_FILENAME)
    if not os.path.exists(data_path):
        try:
            _download_file(DATA_URL, data_path)
        except Exception as e:
            print(f"[WARN] Could not auto-download dataset: {e}")
            print(f"[HINT] Manually download '{DATA_URL}' and place it at '{data_path}'.")
            raise

    # 2) Load database (xb) and query (xq) vectors
    xb, xq = _load_sift_hdf5(data_path)
    if xb.dtype != np.float32:
        xb = xb.astype(np.float32)
    if xq.dtype != np.float32:
        xq = xq.astype(np.float32)

    d = xb.shape[1]
    print(f"[INFO] Loaded database: xb shape={xb.shape}, queries: xq shape={xq.shape}, dim={d}")

    # 3) Build HNSW index (L2)
    index = faiss.IndexHNSWFlat(d, M)  # Flat storage + HNSW graph on top
    # Set construction/search parameters
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH

    # 4) Add all database vectors
    t0 = time.time()
    index.add(xb) 
    build_time = time.time() - t0
    print(f"[INFO] Added {xb.shape[0]} vectors to HNSW in {build_time:.2f}s")

    # 5) Query with the FIRST query vector, retrieve top-10
    if xq.shape[0] < 1:
        raise ValueError("No query vectors found in dataset 'test'.")
    q = xq[:1]  # first query
    t0 = time.time()
    D, I = index.search(q, K)  # D: distances, I: indices
    search_time = time.time() - t0
    print(f"[INFO] Searched top-{K} for first query in {search_time*1e3:.2f} ms")

    # 6) Write the 10 neighbor indices to ./output.txt (one per line)
    out_path = os.path.join(os.path.dirname(__file__), "output.txt")
    with open(out_path, "w") as f:
        for idx in I[0].tolist():
            f.write(f"{int(idx)}\n")
    print(f"[INFO] Wrote neighbor indices to {out_path}")
   
if __name__ == "__main__":
    evaluate_hnsw()
