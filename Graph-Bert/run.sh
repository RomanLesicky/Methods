#!/bin/bash
# The way this is meant to be used is ./run.sh <num_cores> <gpu_id> <script_name>
# This simplifies and unifies the entire "choosing" which GPU and how many cores. 

NCORES=${1:-4}
GPU=${2:-0}
SCRIPT=${3:-script_3_fine_tuning.py}

export OMP_NUM_THREADS=$NCORES
export MKL_NUM_THREADS=$NCORES
export OPENBLAS_NUM_THREADS=$NCORES
export NUMEXPR_NUM_THREADS=$NCORES
export CUDA_VISIBLE_DEVICES=$GPU

echo "Running $SCRIPT with $NCORES CPU cores on GPU $GPU"
python "$SCRIPT"