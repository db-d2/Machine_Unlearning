#!/bin/bash
set -e

FORGET_TYPE=${1:-scattered}
LAMBDAS=${2:-"0.2 0.5 0.8"}
OUTPUT_DIR="outputs/p2/unlearn_${FORGET_TYPE}"

echo "Starting adversarial unlearning"
echo "Forget type: $FORGET_TYPE"
echo "Lambda values: $LAMBDAS"
echo "Output directory: $OUTPUT_DIR"

python src/run_unlearn.py \
    --config configs/unlearn.yaml \
    --forget_type "$FORGET_TYPE" \
    --lambda_list $LAMBDAS \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Unlearning complete!"
echo "View TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs"
