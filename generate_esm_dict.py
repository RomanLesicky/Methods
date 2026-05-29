"""
Generate per-protein ESM-2 embedding dictionaries.

Each protein gets one mean-pooled vector (over real residues, excluding the
BOS/EOS tokens), saved as {id: np.ndarray (D,)}. 

The output format matches the SeqVec dicts so it drops straight into generate_node_v2.py. D = 1280 for the
default esm2_t33_650M model

"""

# Thread caps have to be set before torch/numpy are imported.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import sys
import time
import argparse

import numpy as np


RAW_ROOT = "/home/membio8/Methods_local/S-VGAE/data"
OUT_ROOT = "/home/membio8/Methods_local/esm_files"

# (dataset dir, sequence filename)
DATASETS = [
    ("C.elegan",   "sequenceList.txt"),
    ("E.coli",     "sequenceList.txt"),
    ("Drosophila", "sequenceList.txt"),
    ("Hprd",       "sequence.txt"),
    ("Human",      "sequenceList.txt"),
]

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"   # D=1280; smaller: t30_150M (D=640), t12_35M (D=480)
MAX_RESIDUES = 1022   # 1024-token context minus BOS/EOS
VRAM_CAP_GB = 25.0


def load_ids_and_sequences(raw_dir, seq_filename):
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
        print(f"  note: proteinList={len(ids)}, sequences={len(seqs)}. Using first {n}.")
    return list(zip(ids[:n], seqs[:n]))


def embed_one(sequence, tokenizer, model, device):
    truncated = False
    if len(sequence) > MAX_RESIDUES:
        sequence = sequence[:MAX_RESIDUES]
        truncated = True

    import torch
    with torch.no_grad():
        enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        mask = mask.clone().float()

        # mean over real residues, excluding BOS (first) and EOS (last real token)
        mask[:, 0, :] = 0
        real_len = int(enc["attention_mask"].sum().item())
        if real_len > 0:
            mask[:, real_len - 1, :] = 0
        denom = mask.sum(dim=1).clamp(min=1.0)
        vec = (hidden * mask).sum(dim=1) / denom
        vec = vec[0].float().cpu().numpy()

    return vec, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    help="'all' or one of: c.elegan, e.coli, drosophila, hprd, human")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip datasets whose output dict already exists")
    ap.add_argument("--fp16", action="store_true",
                    help="Run the model in float16 (halves VRAM, roughly doubles speed)")
    ap.add_argument("--vram-cap", type=float, default=None,
                    help=f"Override VRAM cap in GiB (default: {VRAM_CAP_GB})")
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
        print(f"GPU: {torch.cuda.get_device_name(0)}  total={total_gb:.1f} GB  cap={vram_cap} GB")

    from transformers import AutoTokenizer, AutoModel
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()
    hidden_size = model.config.hidden_size
    print(f"  loaded. hidden_size={hidden_size}  dtype={dtype}")

    model_tag = args.model.split("/")[-1]
    os.makedirs(OUT_ROOT, exist_ok=True)

    chosen = DATASETS if args.dataset == "all" else \
             [d for d in DATASETS if d[0].lower() == args.dataset.lower()]
    if not chosen:
        print(f"No dataset matches '{args.dataset}'. Options: {[d[0] for d in DATASETS]}")
        sys.exit(1)

    for (raw_name, seq_filename) in chosen:
        print(f"\n{raw_name}")
        raw_dir = os.path.join(RAW_ROOT, raw_name)
        out_name = f"{raw_name}_{model_tag}_dict.npy"
        out_path = os.path.join(OUT_ROOT, out_name)

        if args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            print(f"  skip, already exists: {out_path}")
            continue

        if not os.path.isdir(raw_dir):
            print(f"  skip, raw dir missing: {raw_dir}")
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
            except torch.cuda.OutOfMemoryError:
                failed.append((uid, f"OOM (len={len(seq)})"))
                torch.cuda.empty_cache()
            except Exception as e:
                failed.append((uid, str(e)[:100]))

            if device == "cuda" and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(pairs) - (i + 1)) / max(rate, 1e-6)
                print(f"    [{i+1}/{len(pairs)}]  {rate:.1f} prot/s  eta {eta/60:.1f} min")

        np.save(out_path, out_dict, allow_pickle=True)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  saved {len(out_dict)} embeddings ({size_mb:.1f} MB) -> {out_path}")
        if n_truncated:
            print(f"  truncated (len>{MAX_RESIDUES}): {n_truncated}")
        if failed:
            print(f"  {len(failed)} proteins failed:")
            for uid, err in failed[:5]:
                print(f"    {uid}: {err}")
            if len(failed) > 5:
                print(f"    ... and {len(failed) - 5} more")


if __name__ == "__main__":
    main()
