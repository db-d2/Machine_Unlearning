#!/bin/bash
# Audit baseline VAE with cluster conditioning

MODEL="outputs/p1/baseline/best_model.pt"
FORGET_SET="data/forget_sets/forget_structured.json"
MATCHED_NEGATIVES="outputs/p1.5/s1_matched_negatives.json"
OUTPUT_DIR="outputs/p1.5/audit_structured_conditioned"

echo "Auditing baseline VAE (cluster-conditioned)..."
echo "Model: $MODEL"
echo "Forget set: $FORGET_SET"
echo "Matched negatives: $MATCHED_NEGATIVES"
echo "Output: $OUTPUT_DIR"
echo ""

python src/train_attacker_conditioned.py \
    --data_path data/adata_processed.h5ad \
    --model_path "$MODEL" \
    --forget_set_path "$FORGET_SET" \
    --matched_negatives_path "$MATCHED_NEGATIVES" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_dims 512 128 \
    --latent_dim 16 \
    --likelihood nb \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --seed 42 \
    --conditioning_method residualize
