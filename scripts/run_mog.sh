#!/bin/bash
set -e

OUTPUT_DIR="outputs/p3/mog"

echo "Starting mixture model simulations"
echo "Output directory: $OUTPUT_DIR"

python src/mog.py \
    --config configs/mog.yaml \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "MoG simulations complete!"
echo "View TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs"
