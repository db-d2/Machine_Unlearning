#!/bin/bash
# Adversarial unlearning - lambda=0.8 (high utility)

pyenv activate stat4243

python src/train_unlearn.py --config configs/unlearn_structured_lambda0.8.yaml
