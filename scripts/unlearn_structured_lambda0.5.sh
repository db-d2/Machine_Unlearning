#!/bin/bash
# Adversarial unlearning - lambda=0.5 (balanced)

pyenv activate stat4243

python src/train_unlearn.py --config configs/unlearn_structured_lambda0.5.yaml
