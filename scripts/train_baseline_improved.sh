#!/bin/bash
# Improved baseline VAE training
#
# Changes from v1:
# - z: 16 -> 32 (larger latent capacity)
# - Layers: [512, 128] -> [1024, 512, 128] (wider)
# - Added LayerNorm + dropout 0.1
# - KL warm-up: 0->1 over 20 epochs
# - Free-bits: 0.03 nats/dim

pyenv activate stat4243

python src/train.py --config configs/baseline_improved.yaml
