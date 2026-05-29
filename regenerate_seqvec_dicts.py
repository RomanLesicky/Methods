"""
Rebuild the per-protein SeqVec embedding dicts the PPI Graph-BERT pipeline


expects at seqvec_files/<dataset>_seqvec_dict.npy.

The original .npy files were stored via Git LFS and the remote objects are gone (404 on `git lfs pull`), so we 
rebuild them from the raw proteinList.txt + sequence file pairs. The embedding recipe mirrors the original embedding.py:

    emb = SeqVecEmbedder().embed(sequence)                 # (3, L, 1024)
    per_protein = torch.tensor(emb).sum(dim=0).mean(dim=0) # (1024,)

Needs bio-embeddings (pip install bio-embeddings[seqvec]), first run downloads ~400 MB of SeqVec weights. A fresh venv is recommended since it pins old deps.
"""

import os

_N = "8"
os.environ["OMP_NUM_THREADS"]      = _N
os.environ["MKL_NUM_THREADS"]      = _N
os.environ["OPENBLAS_NUM_THREADS"] = _N
os.environ["NUMEXPR_NUM_THREADS"]  = _N

import sys
import time

import numpy as np


RAW_ROOT = "/home/membio8/Methods_local/S-VGAE/data"
OUT_ROOT = "/home/membio8/Methods_local/seqvec_files"

# (dataset dir, sequence filename, output dict filename), ordered small -> large
DATASETS = [
    ("C.elegan",   "sequenceList.txt",  "C.elegan_seqvec_dict.npy"),
    ("E.coli",     "sequenceList.txt",  "e.coli_seqvec_dict.npy"),
    ("Drosophila", "sequenceList.txt",  "drosophila_seqvec_dict.npy"),
    ("Hprd",       "sequence.txt",      "hprd_seqvec_dict.npy"),
    ("Human",      "sequenceList.txt",  "human_seqvec_dict.npy"),
]


def load_ids_and_sequences(raw_dir, seq_filename):
    """Parallel read of proteinList.txt and the sequence file (line N of one
    matches line N of the other). Some species ship extra trailing proteinList
    entries with no sequence; we truncate to the overlap."""
    plist_path = os.path.join(raw_dir, "proteinList.txt")
    seq_path   = os.path.join(raw_dir, seq_filename)

    ids = []
    with open(plist_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[1] if len(parts) > 1 else parts[0])

    seqs = []
    with open(seq_path) as f:
        for line in f:
            seqs.append(line.strip())

    n = min(len(ids), len(seqs))
    if len(ids) != len(seqs):
        print(f"  note: proteinList has {len(ids)} entries, "
              f"sequence file has {len(seqs)}. Using first {n}.")

    return list(zip(ids[:n], seqs[:n]))


def main():
    try:
        import torch
        from bio_embeddings.embed import SeqVecEmbedder
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install with: pip install bio-embeddings[seqvec]")
        sys.exit(1)

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SeqVec embedder on device='{device}' ...")
    embedder = SeqVecEmbedder(device=device)
    print("  SeqVec loaded.\n")

    os.makedirs(OUT_ROOT, exist_ok=True)

    for (raw_name, seq_filename, out_name) in DATASETS:
        raw_dir  = os.path.join(RAW_ROOT, raw_name)
        out_path = os.path.join(OUT_ROOT, out_name)

        print(f"\n{raw_name}")

        if not os.path.isdir(raw_dir):
            print(f"  skip, raw dir not found: {raw_dir}")
            continue

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            print(f"  skip, output already exists ({os.path.getsize(out_path)/1e6:.1f} MB): {out_path}")
            continue

        try:
            pairs = load_ids_and_sequences(raw_dir, seq_filename)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip: {e}")
            continue

        # quick format check before committing to a long run
        sid, sseq = pairs[0]
        print(f"  {len(pairs)} protein entries to embed")
        print(f"  sample:  id='{sid}'  seq='{sseq[:60]}{'...' if len(sseq) > 60 else ''}'  (len={len(sseq)})")

        if not sseq or not sseq[0].isalpha():
            print(f"  WARNING: first sequence looks wrong. Aborting this dataset.")
            continue

        out_dict = {}
        failed = []
        t_start = time.time()

        for i, (uid, seq) in enumerate(pairs):
            if not seq:
                failed.append((uid, "empty sequence"))
                continue
            try:
                emb = embedder.embed(seq)
                vec = torch.from_numpy(np.asarray(emb)).sum(dim=0).mean(dim=0)
                out_dict[uid] = vec.cpu().numpy().astype(np.float32)
            except Exception as e:
                failed.append((uid, str(e)[:80]))

            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                remaining = (len(pairs) - (i + 1)) / max(rate, 1e-6)
                print(f"    [{i+1}/{len(pairs)}]  {rate:.1f} prot/s  eta {remaining/60:.1f} min")

        np.save(out_path, out_dict, allow_pickle=True)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  saved {len(out_dict)} embeddings ({size_mb:.1f} MB) -> {out_path}")
        if failed:
            print(f"  {len(failed)} proteins failed to embed:")
            for uid, err in failed[:10]:
                print(f"    {uid}: {err}")
            if len(failed) > 10:
                print(f"    ... and {len(failed) - 10} more")

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
