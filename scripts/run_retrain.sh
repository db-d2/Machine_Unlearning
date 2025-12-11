#!/bin/bash
set -e

FORGET_TYPE=${1:-scattered}
OUTPUT_DIR="outputs/p1/retrain_${FORGET_TYPE}"

echo "Starting full retrain on D\F"
echo "Forget type: $FORGET_TYPE"
echo "Output directory: $OUTPUT_DIR"

python src/retrain.py \
    --config configs/retrain.yaml \
    --forget_type "$FORGET_TYPE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Retrain complete!"
echo "View TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs"
