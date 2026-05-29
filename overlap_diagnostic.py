"""
Overlap diagnostic for released vs raw PPI edge sets.

Decodes each released pair_id string back into its (idx1, idx2) tuple by brute-forcing the proteinList range (trying both concat orders), 
then compares edges as unordered frozensets so the result is invariant to concat direction and edge orientation.

Also reports why a pair_id failed to decode, which tells you whether a discrepancy is a format bug or a real data mismatch.
"""

import os


RAW_ROOT       = "/home/membio8/Methods_local/S-VGAE/data"
PROCESSED_ROOT = "/home/membio8/Methods_local/data"
SEQVEC_ROOT    = "/home/membio8/Methods_local/seqvec_files"

DATASETS = [
    ("Hprd",       "ppi",        "hprd_seqvec_dict.npy"),
    ("C.elegan",   "c.elegan",   "C.elegan_seqvec_dict.npy"),
    ("Drosophila", "drosophila", None),
    ("E.coli",     "e.coli",     None),
]


def load_protein_list(path):
    ids = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[1] if len(parts) > 1 else parts[0])
    return ids


def load_edges(pos_path, neg_path):
    edges = []
    for path, is_pos in [(pos_path, True), (neg_path, False)]:
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1]), is_pos))
    return edges


def load_released_pair_ids(node_path):
    ids = []
    with open(node_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split("\t", 1)[0] if "\t" in line else line.split(None, 1)[0]
            ids.append(first)
    return ids


def decode_pair_id(pair_id_str, n_proteins):
    """Return the set of frozenset({a, b}) splits of the digit string where both
    a and b are valid protein indices (may be 0, 1, or several)."""
    candidates = set()
    s = pair_id_str
    for split in range(1, len(s)):
        left, right = s[:split], s[split:]
        # reject leading zeros except for the single digit "0"
        if (len(left) > 1 and left[0] == "0") or (len(right) > 1 and right[0] == "0"):
            continue
        a, b = int(left), int(right)
        if a < n_proteins and b < n_proteins:
            candidates.add(frozenset({a, b}))
    return candidates


def diagnose(raw_name, proc_name, seqvec_name):
    print(f"\nDataset: {raw_name}  (processed dir: '{proc_name}')")

    raw_dir  = os.path.join(RAW_ROOT, raw_name)
    proc_dir = os.path.join(PROCESSED_ROOT, proc_name)

    plist_path = os.path.join(raw_dir, "proteinList.txt")
    pos_path   = os.path.join(raw_dir, "PositiveEdges.txt")
    neg_path   = os.path.join(raw_dir, "NegativeEdges.txt")
    node_path  = os.path.join(proc_dir, "node")

    for p in (plist_path, pos_path, neg_path, node_path):
        if not os.path.exists(p):
            print(f"  [skip] missing: {p}")
            return

    proteins = load_protein_list(plist_path)
    n = len(proteins)
    print(f"  proteins: {n}")

    edges = load_edges(pos_path, neg_path)
    raw_edge_set = set(frozenset({i, j}) for (i, j, _) in edges)
    print(f"  raw edges: {len(edges)} ({len(raw_edge_set)} unique unordered)")

    released_ids = load_released_pair_ids(node_path)
    print(f"  released node file: {len(released_ids)} lines")

    decoded_edges = set()
    ambiguous = 0
    unresolvable = 0
    for pid in released_ids:
        cands = decode_pair_id(pid, n)
        if len(cands) == 0:
            unresolvable += 1
        elif len(cands) == 1:
            decoded_edges.add(next(iter(cands)))
        else:
            # ambiguous: keep every candidate but count it
            ambiguous += 1
            decoded_edges.update(cands)

    print(f"  decoded: {len(decoded_edges)} unique edges from released ids")
    print(f"    ambiguous ids (multiple valid splits): {ambiguous}")
    print(f"    unresolvable ids (no valid split):     {unresolvable}")

    inter = decoded_edges & raw_edge_set
    only_rel = decoded_edges - raw_edge_set
    only_raw = raw_edge_set - decoded_edges
    pct = 100 * len(inter) / max(1, len(decoded_edges))
    print(f"  edge-set overlap:")
    print(f"    released ∩ raw : {len(inter)} ({pct:.1f}% of decoded)")
    print(f"    only in released (can't find in raw): {len(only_rel)}")
    print(f"    only in raw (not in released):        {len(only_raw)}")

    print(f"  first 5 released ids -> decodings:")
    for pid in released_ids[:5]:
        cands = decode_pair_id(pid, n)
        matches = [c for c in cands if c in raw_edge_set]
        print(f"    '{pid}'  candidates={[tuple(sorted(c)) for c in cands]}  in_raw={[tuple(sorted(c)) for c in matches]}")


def main():
    print("Overlap diagnostic (edge-set based)")
    for raw, proc, sv in DATASETS:
        try:
            diagnose(raw, proc, sv)
        except Exception as e:
            print(f"\n[ERROR] {raw}: {type(e).__name__}: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
