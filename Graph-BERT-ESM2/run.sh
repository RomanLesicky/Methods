# Unified launched for the ESM2 variant for all the datastes and embeddings 
# Works via: ./run.sh <how many cores> <which gpu> <which script> <which dataset> <which embeddings to use>   

set -euo pipefail
 
NCORES=${1:-4}
GPU=${2:-0}
SCRIPT=${3:-script_3_fine_tuning.py}
DATASET=${4:-ppi}
EMBEDDER=${5:-seqvec}
shift 5 2>/dev/null || true   # remaining args go to $@
 
# ---- CPU thread caps ----
export OMP_NUM_THREADS=$NCORES
export MKL_NUM_THREADS=$NCORES
export OPENBLAS_NUM_THREADS=$NCORES
export NUMEXPR_NUM_THREADS=$NCORES
 
# ---- GPU selection ----
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU
    echo "GPU: $GPU"
else
    export CUDA_VISIBLE_DEVICES=""
    echo "GPU: none (CPU mode)"
fi
 
# ---- Dataset + embedder env vars (read by the patched Python scripts) ----
export PPI_DATASET="$DATASET"
export PPI_EMBEDDER="$EMBEDDER"
 
echo "Running $SCRIPT | dataset=$DATASET | embedder=$EMBEDDER | cores=$NCORES"
 
# ---- Staging: copy the right Node file into ./data/<dataset>/node ----
# Skip staging for script_1 (doesn't depend on embedder — any node file with
# the right IDs works, and the existing one is fine).
# Also skip for script_4 (reads results, not raw data).
SCRIPTS_NEEDING_STAGE="script_2_pre_train.py script_3_fine_tuning.py"
 
# Resolve paths relative to where this script lives (Graph-Bert/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Node_creation is one level up from Graph-Bert/
BASE_DIR="$(dirname "$SCRIPT_DIR")"
 
# Map dataset names to the raw subdir names used in Node file naming
declare -A DS_TO_RAW=(
    [ppi]=Hprd
    [c.elegan]=C.elegan
    [e.coli]=E.coli
    [drosophila]=Drosophila
    [human]=Human
)
 
if echo "$SCRIPTS_NEEDING_STAGE" | grep -qw "$SCRIPT"; then
    RAW_NAME="${DS_TO_RAW[$DATASET]:-$DATASET}"
    SRC="${BASE_DIR}/Node_creation/Node_${RAW_NAME}_${EMBEDDER}.txt"
    DST="${SCRIPT_DIR}/data/${DATASET}/node"
 
    if [ ! -f "$SRC" ]; then
        echo "ERROR: Node file not found: $SRC"
        echo "       Run generate_node_v2.py --dataset $DATASET --embedder $EMBEDDER first."
        exit 1
    fi
 
    mkdir -p "$(dirname "$DST")"
    cp "$SRC" "$DST"
    SRC_SIZE=$(du -h "$DST" | cut -f1)
    echo "Staged: $SRC → $DST ($SRC_SIZE)"
fi
 
# ---- Run ----
cd "$SCRIPT_DIR"
 
if [ "$SCRIPT" = "script_4_evaluation_plots.py" ]; then
    # script_4 uses CLI args instead of env vars
    python "$SCRIPT" --dataset "$DATASET" --embedder "$EMBEDDER" "$@"
else
    python "$SCRIPT" "$@"
fi