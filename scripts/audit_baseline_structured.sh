#!/bin/bash
# Audit baseline VAE with structured forget set

MODEL="outputs/p1/baseline/best_model.pt"
FORGET_SET="data/forget_sets/forget_structured.json"
OUTPUT_DIR="outputs/p1/audit_baseline_structured"

echo "Auditing baseline VAE (structured forget set)..."
echo "Model: $MODEL"
echo "Forget set: $FORGET_SET"
echo "Output: $OUTPUT_DIR"
echo ""

python src/train_attacker.py \
    --data_path data/adata_processed.h5ad \
    --model_path "$MODEL" \
    --forget_set_path "$FORGET_SET" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_dims 512 128 \
    --latent_dim 16 \
    --likelihood nb \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --seed 42
