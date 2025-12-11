"""Logging utilities for TensorBoard and experiment tracking."""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import json


class ExperimentLogger:
    """Handles TensorBoard logging and metric tracking."""

    def __init__(self, log_dir: Path, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalar values with a common prefix."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram to TensorBoard."""
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)):
                self.log_scalar(tag, value, step)

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def save_metrics_json(metrics: dict, output_dir: Path, filename: str = "metrics.json"):
    """Save metrics dictionary to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {filepath}")
