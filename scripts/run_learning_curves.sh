#!/bin/bash
# Run Fisher and retrain with AUC-vs-time tracking to generate learning curves.
#
# This script runs both unlearning methods with periodic AUC evaluation
# and generates a comparison figure.
#
# Usage: bash scripts/run_learning_curves.sh

set -e

# Paths
DATA_PATH="data/adata_processed.h5ad"
SPLIT_PATH="outputs/p1/split_structured.json"
BASELINE_CHECKPOINT="outputs/p1/baseline_v2/best_model.pt"
ATTACKER_PATH="outputs/p2/attackers/attacker_v1_seed42.pt"
MATCHED_NEG_PATH="outputs/p1.5/s1_matched_negatives.json"

OUTPUT_DIR="outputs/p2/learning_curves"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Running Learning Curve Generation"
echo "========================================"

# Run Fisher unlearning with AUC tracking
echo ""
echo "[1/3] Running Fisher unlearning with AUC tracking..."
python src/train_fisher_unlearn.py \
    --data_path "$DATA_PATH" \
    --split_path "$SPLIT_PATH" \
    --baseline_checkpoint "$BASELINE_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/fisher" \
    --scrub_steps 100 \
    --finetune_epochs 10 \
    --track_auc \
    --attacker_path "$ATTACKER_PATH" \
    --matched_negatives_path "$MATCHED_NEG_PATH" \
    --eval_interval_scrub 10 \
    --eval_interval_finetune 2 \
    --feature_variant v1

# Run retrain with AUC tracking
# Use same architecture as baseline (latent_dim=32, hidden_dims=[1024,512,128])
echo ""
echo "[2/3] Running retrain with AUC tracking..."
python src/retrain.py \
    --data_path "$DATA_PATH" \
    --forget_set_path "$SPLIT_PATH" \
    --output_dir "$OUTPUT_DIR/retrain" \
    --epochs 100 \
    --hidden_dims 1024 512 128 \
    --latent_dim 32 \
    --track_auc \
    --attacker_path "$ATTACKER_PATH" \
    --matched_negatives_path "$MATCHED_NEG_PATH" \
    --eval_interval 5 \
    --feature_variant v1

# Generate learning curve figure
echo ""
echo "[3/3] Generating learning curve figure..."
python src/plot_learning_curves.py \
    --fisher_path "$OUTPUT_DIR/fisher/learning_curve.json" \
    --retrain_path "$OUTPUT_DIR/retrain/learning_curve.json" \
    --output_path "$OUTPUT_DIR/learning_curves_comparison.png" \
    --retrain_floor_auc 0.864 \
    --baseline_auc 0.951

# Also generate with inset
python src/plot_learning_curves.py \
    --fisher_path "$OUTPUT_DIR/fisher/learning_curve.json" \
    --retrain_path "$OUTPUT_DIR/retrain/learning_curve.json" \
    --output_path "$OUTPUT_DIR/learning_curves_with_inset.png" \
    --retrain_floor_auc 0.864 \
    --baseline_auc 0.951 \
    --with_inset

echo ""
echo "========================================"
echo "Learning Curve Generation Complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Fisher learning curve: $OUTPUT_DIR/fisher/learning_curve.json"
echo "  Retrain learning curve: $OUTPUT_DIR/retrain/learning_curve.json"
echo "  Comparison figure: $OUTPUT_DIR/learning_curves_comparison.png"
echo "  Figure with inset: $OUTPUT_DIR/learning_curves_with_inset.png"
