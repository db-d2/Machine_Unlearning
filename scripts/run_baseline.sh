#!/bin/bash
set -e

OUTPUT_DIR="outputs/p1/baseline"

echo "Starting baseline VAE training"
echo "Output directory: $OUTPUT_DIR"

python src/train.py \
    --config configs/baseline.yaml \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Training complete!"
echo "View TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs"
