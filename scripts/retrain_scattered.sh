#!/bin/bash
# Gold-standard retrain on D\F (scattered forget set)

FORGET_SET="data/forget_sets/forget_scattered.json"
OUTPUT_DIR="outputs/p1/retrain_scattered"

echo "Gold-standard retraining (scattered forget set)..."
echo "Forget set: $FORGET_SET"
echo "Output: $OUTPUT_DIR"
echo ""

python src/retrain.py \
    --data_path data/adata_processed.h5ad \
    --forget_set_path "$FORGET_SET" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_dims 512 128 \
    --latent_dim 16 \
    --likelihood nb \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.001 \
    --beta 1.0 \
    --seed 42 \
    --print_every 10
