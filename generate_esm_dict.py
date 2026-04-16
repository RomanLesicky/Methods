#!/usr/bin/env python3
"""
generate_esm_dict.py
====================
Generate per-protein ESM-2 embedding dictionaries in the same format as the
SeqVec dicts produced by regenerate_seqvec_dicts.py, so they drop into
generate_node_v2.py with no pipeline changes.

Output schema (per dataset):
    { protein_id: np.ndarray shape (D,) dtype float32 }
where D = 1280 for esm2_t33_650M_UR50D (the default).

Per-protein embedding = mean over real-residue token embeddings of the last
hidden layer, excluding BOS/EOS/PAD. This matches SeqVec's per-protein recipe
(sum over SeqVec's 3 layers then mean over residues).

Safety:
  - CPU threads capped at 8 (set BEFORE importing torch/numpy).
  - GPU VRAM capped at 25 GB via per-process memory fraction.
  - Long sequences (>1022 aa) are truncated; a warning counts how many.
"""

# --- resource caps; MUST precede torch/numpy imports ---
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "8"

# Reduce CUDA fragmentation (helps stay under memory cap for long sequences)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
# ------------------------------------------------------

import sys
import time
import argparse

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT = "/home/membio8/Methods_local/S-VGAE/data"
OUT_ROOT = "/home/membio8/Methods_local/esm_files"

# (raw subdir, sequence filename, output dict filename tag)
DATASETS = [
    ("C.elegan",   "sequenceList.txt"),
    ("E.coli",     "sequenceList.txt"),
    ("Drosophila", "sequenceList.txt"),
    ("Hprd",       "sequence.txt"),
    ("Human",      "sequenceList.txt"),
]

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"   # 650M params, D=1280
# Smaller alternatives if you want a size sweep:
#   "facebook/esm2_t30_150M_UR50D"   D=640
#   "facebook/esm2_t12_35M_UR50D"    D=480

MAX_RESIDUES = 1022   # ESM-2 context limit is 1024 tokens incl. BOS/EOS

# VRAM cap (GiB) — soft cap via set_per_process_memory_fraction
VRAM_CAP_GB = 25.0


# ---------------------------------------------------------------------------
# File loading (mirrors the SeqVec regen script, including the orphan-truncation fix)
# ---------------------------------------------------------------------------
def load_ids_and_sequences(raw_dir: str, seq_filename: str):
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
        print(f"  note: proteinList={len(ids)}, sequences={len(seqs)}. "
              f"Using first {n} as parallel overlap.")
    return list(zip(ids[:n], seqs[:n]))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_one(sequence: str, tokenizer, model, device):
    """Return a 1-D numpy float32 vector of size D = model.config.hidden_size."""
    truncated = False
    if len(sequence) > MAX_RESIDUES:
        sequence = sequence[:MAX_RESIDUES]
        truncated = True

    # ESM tokenizers expect whitespace-separated residues OR a plain string;
    # the transformers ESM tokenizer handles a bare string fine.
    import torch
    with torch.no_grad():
        enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        hidden = out.last_hidden_state                  # (1, L+2, D)
        mask = enc["attention_mask"].unsqueeze(-1)      # (1, L+2, 1)

        # Exclude BOS (position 0) and EOS (last non-pad position)
        mask = mask.clone().float()
        mask[:, 0, :] = 0                                # zero out BOS
        # EOS is at the last real token. Find it by summing original mask.
        real_len = int(enc["attention_mask"].sum().item())
        if real_len > 0:
            mask[:, real_len - 1, :] = 0                 # zero out EOS
        denom = mask.sum(dim=1).clamp(min=1.0)
        vec = (hidden * mask).sum(dim=1) / denom         # (1, D)
        vec = vec[0].float().cpu().numpy()

    return vec, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    help="'all' or one of: c.elegan, e.coli, drosophila, hprd, human")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip datasets whose output dict already exists and is non-trivial")
    ap.add_argument("--fp16", action="store_true",
                    help="Run the model in float16. Halves VRAM and roughly doubles speed; "
                         "recommended for the 3B model.")
    ap.add_argument("--vram-cap", type=float, default=None,
                    help=f"Override VRAM cap in GiB (default: {VRAM_CAP_GB}). Raise to ~28 for 3B.")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    vram_cap = args.vram_cap if args.vram_cap is not None else VRAM_CAP_GB
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        frac = min(0.99, vram_cap / total_gb)
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        print(f"GPU: {torch.cuda.get_device_name(0)}  total={total_gb:.1f} GB  "
              f"cap={vram_cap} GB ({frac*100:.1f}%)")

    from transformers import AutoTokenizer, AutoModel
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()
    hidden_size = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded. hidden_size={hidden_size}  params={n_params/1e6:.0f}M  dtype={dtype}")

    # Tag the model size into the output filename so different sizes don't clash.
    model_tag = args.model.split("/")[-1]     # e.g. 'esm2_t33_650M_UR50D'
    os.makedirs(OUT_ROOT, exist_ok=True)

    chosen = DATASETS if args.dataset == "all" else \
             [d for d in DATASETS if d[0].lower() == args.dataset.lower()]
    if not chosen:
        print(f"No dataset matches '{args.dataset}'. Options: {[d[0] for d in DATASETS]}")
        sys.exit(1)

    for (raw_name, seq_filename) in chosen:
        print(f"\n{'='*72}\n  {raw_name}\n{'='*72}")
        raw_dir = os.path.join(RAW_ROOT, raw_name)
        out_name = f"{raw_name}_{model_tag}_dict.npy"
        out_path = os.path.join(OUT_ROOT, out_name)

        if args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  [skip] output already exists ({size_mb:.1f} MB): {out_path}")
            continue

        if not os.path.isdir(raw_dir):
            print(f"  [skip] raw dir missing: {raw_dir}")
            continue

        pairs = load_ids_and_sequences(raw_dir, seq_filename)
        print(f"  {len(pairs)} proteins to embed")

        out_dict = {}
        failed = []
        n_truncated = 0
        t_start = time.time()

        for i, (uid, seq) in enumerate(pairs):
            if not seq:
                failed.append((uid, "empty sequence"))
                continue
            try:
                vec, truncated = embed_one(seq, tokenizer, model, device)
                out_dict[uid] = vec.astype(np.float32)
                if truncated:
                    n_truncated += 1
            except torch.cuda.OutOfMemoryError as e:
                failed.append((uid, f"OOM (len={len(seq)})"))
                torch.cuda.empty_cache()
            except Exception as e:
                failed.append((uid, str(e)[:100]))
                # don't crash the whole run; carry on

            # Keep fragmentation in check for the big model
            if device == "cuda" and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(pairs) - (i + 1)) / max(rate, 1e-6)
                print(f"    [{i+1}/{len(pairs)}]  {rate:.1f} prot/s  "
                      f"elapsed {elapsed/60:.1f} min  eta {eta/60:.1f} min")

        np.save(out_path, out_dict, allow_pickle=True)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"\n  saved {len(out_dict)} embeddings ({size_mb:.1f} MB) → {out_path}")
        print(f"  truncated (len>{MAX_RESIDUES}): {n_truncated}")
        if failed:
            print(f"  {len(failed)} proteins failed to embed:")
            for uid, err in failed[:5]:
                print(f"    {uid}: {err}")
            if len(failed) > 5:
                print(f"    ... and {len(failed) - 5} more")

    print("\nDone.")


if __name__ == "__main__":
    main()