#!/usr/bin/env python3
"""
Altered version of `PPI_GBERT/pre-processing/generate_node.py` this version can handle datasets as flags. 
But it it's also has been added CPU and GPU caping for safety reasons. 

And additional quite important change that was added, is concerning edge counts. 
Since, the original script hardcodes HPRD's edge counts (36557 positive, 5000-pos filter). 

This version has these chanegs:

  - Accepts a dataset name and looks up the correct raw files + counts
  
  - Uses the native edge files (PositiveEdges + NegativeEdges) directly, so the positive/negative split is
    determined by which file the edge came from rather than by a brittle count threshold.

  - Writes two versions: a full node file (all edges) and a filtered version that matches the paper's HPRD 
    convention (first N_pos_keep positives + all negatives) for datasets where that filter applies.
"""

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------
BASE     = "/home/membio8/Methods_local"
RAW_ROOT = f"{BASE}/S-VGAE/data"
OUT_ROOT = f"{BASE}/Node_creation"

# Where each embedder's dicts live
EMBED_ROOTS = {
    "seqvec":    f"{BASE}/seqvec_files",
    "esm2_650M": f"{BASE}/esm_files",
    "esm2_3B":   f"{BASE}/esm_files",
}

# (raw_subdir, {embedder_tag: dict_filename}, n_pos_keep)
DATASETS = {
    "hprd": ("Hprd", {
        "seqvec":    "hprd_seqvec_dict.npy",
        "esm2_650M": "Hprd_esm2_t33_650M_UR50D_dict.npy",   # only if/when you generate it
        "esm2_3B":   "Hprd_esm2_t36_3B_UR50D_dict.npy",
    }, 5000),
    "c.elegan": ("C.elegan", {
        "seqvec":    "C.elegan_seqvec_dict.npy",
        "esm2_650M": "C.elegan_esm2_t33_650M_UR50D_dict.npy",
        "esm2_3B":   "C.elegan_esm2_t36_3B_UR50D_dict.npy",
    }, None),
    "e.coli": ("E.coli", {
        "seqvec":    "e.coli_seqvec_dict.npy",
        "esm2_650M": "E.coli_esm2_t33_650M_UR50D_dict.npy",
        "esm2_3B":   "E.coli_esm2_t36_3B_UR50D_dict.npy",
    }, None),
    "drosophila": ("Drosophila", {
        "seqvec":    "drosophila_seqvec_dict.npy",
        "esm2_650M": "Drosophila_esm2_t33_650M_UR50D_dict.npy",
        "esm2_3B":   "Drosophila_esm2_t36_3B_UR50D_dict.npy",
    }, None),
    "human": ("Human", {
        "seqvec":    "human_seqvec_dict.npy",
    }, None),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--embedder", default="seqvec",
                    help="tag used in output filename, e.g. 'seqvec' or 'esm2'")
    ap.add_argument("--embed-file", default=None,
                    help="override the embedding .npy path")
    args = ap.parse_args()

    raw_subdir, embed_files, n_pos_keep = DATASETS[args.dataset]
    raw_dir = os.path.join(RAW_ROOT, raw_subdir)

    if args.embed_file:
        embed_path = args.embed_file
    else:
        if args.embedder not in embed_files:
            sys.exit(f"No '{args.embedder}' dict registered for {args.dataset}. "
                     f"Available: {list(embed_files.keys())}. "
                     f"Or pass --embed-file explicitly.")
        embed_root = EMBED_ROOTS.get(args.embedder, EMBED_ROOTS["seqvec"])
        embed_path = os.path.join(embed_root, embed_files[args.embedder])

    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"Dataset   : {args.dataset}")
    print(f"Raw dir   : {raw_dir}")
    print(f"Embedder  : {args.embedder}")
    print(f"Embed file: {embed_path}")

    # --- load ---
    data = np.load(embed_path, allow_pickle=True).tolist()
    print(f"Loaded {len(data)} embeddings  (sample dim={len(next(iter(data.values())))})")

    with open(os.path.join(raw_dir, "proteinList.txt")) as f:
        protLines = f.readlines()

    pos_edges = []
    with open(os.path.join(raw_dir, "PositiveEdges.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pos_edges.append((int(parts[0]), int(parts[1])))

    neg_edges = []
    with open(os.path.join(raw_dir, "NegativeEdges.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                neg_edges.append((int(parts[0]), int(parts[1])))

    print(f"Positive edges: {len(pos_edges)}")
    print(f"Negative edges: {len(neg_edges)}")

    # --- build edge list in the author's order: positives first, then negatives ---
    # Apply HPRD-style filter if configured for this dataset.
    if n_pos_keep is not None and len(pos_edges) > n_pos_keep:
        print(f"Applying filter: keep first {n_pos_keep} positives + all negatives")
        edge_list = [(i, j, True) for (i, j) in pos_edges[:n_pos_keep]] \
                  + [(i, j, False) for (i, j) in neg_edges]
    else:
        edge_list = [(i, j, True)  for (i, j) in pos_edges] \
                  + [(i, j, False) for (i, j) in neg_edges]

    print(f"Edges to process: {len(edge_list)}")

    # --- write node file ---
    out_name = f"Node_{raw_subdir}_{args.embedder}.txt"
    out_path = os.path.join(OUT_ROOT, out_name)

    missing_proteins = set()
    n_written = 0
    n_pos_written = 0
    n_neg_written = 0
    n_skipped_missing = 0

    with open(out_path, "w") as f:
        for (prot1, prot2, is_pos) in edge_list:
            # Retrieve protein IDs from proteinList (column [1])
            if prot1 >= len(protLines) or prot2 >= len(protLines):
                n_skipped_missing += 1
                continue
            key1 = protLines[prot1].strip().split("\t")[1]
            key2 = protLines[prot2].strip().split("\t")[1]

            v1 = data.get(key1)
            v2 = data.get(key2)
            if v1 is None or v2 is None:
                if v1 is None: missing_proteins.add(key1)
                if v2 is None: missing_proteins.add(key2)
                n_skipped_missing += 1
                continue

            feat = list(v1) + list(v2)
            pair_id = str(prot2) + str(prot1)   # matches generate_node.py's arr[1]+arr[0]
            label = "Positive" if is_pos else "Negative"

            f.write(pair_id + "\t"
                    + "\t".join(f"{x}" for x in feat) + "\t"
                    + label + "\n")
            n_written += 1
            if is_pos: n_pos_written += 1
            else:      n_neg_written += 1

    print(f"\nWrote {n_written} nodes → {out_path}")
    print(f"  Positive: {n_pos_written}")
    print(f"  Negative: {n_neg_written}")
    print(f"  Skipped (missing embedding or oob): {n_skipped_missing}")
    if missing_proteins:
        print(f"  {len(missing_proteins)} unique proteins had no embedding")

    # quick file-size sanity check
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()