#!/bin/bash
# Script to define forget sets with proper OpenMP configuration for Intel Mac / Colab

# Disable nested OpenMP parallelism to avoid conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Disable OpenMP nested parallelism (silences deprecation warning)
export OMP_MAX_ACTIVE_LEVELS=1

echo "Defining forget sets with single-threaded execution..."
python src/define_forget_sets.py \
    --data_path data/adata_processed.h5ad \
    --output_dir data/forget_sets \
    --seed 42
