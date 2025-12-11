#!/bin/bash
# Adversarial unlearning - lambda=0.2 (low privacy, high utility)

pyenv activate stat4243

python src/train_unlearn.py --config configs/unlearn_structured_lambda0.2.yaml
