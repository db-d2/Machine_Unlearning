"""Global utilities for seed setting and metadata logging."""

import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import json
import yaml


# Device configuration (CPU for Intel Mac, CUDA if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int = 42):
    """Set global random seed for reproducibility.

    Safe for CPU-only environments (e.g., Intel Mac).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def create_run_metadata(task_id: str, config: dict, seed: int) -> dict:
    """Create metadata dictionary for experiment run."""
    return {
        'timestamp': datetime.now().isoformat(),
        'task_id': task_id,
        'seed': seed,
        'config': config
    }


def save_metadata(metadata: dict, output_dir: Path, filename: str = 'metadata.json'):
    """Save run metadata to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {filepath}")


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Timer:
    """Simple timer context manager for measuring wall-clock time."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        print(f"[{self.name}] Starting at {self.start_time.strftime('%H:%M:%S')}")
        return self

    def __exit__(self, *args):
        end_time = datetime.now()
        self.elapsed = (end_time - self.start_time).total_seconds()
        print(f"[{self.name}] Completed in {self.elapsed:.2f} seconds")


GLOBAL_SEED = 42
