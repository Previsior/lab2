import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def create_experiment_dirs(experiment_name: str) -> Tuple[Path, Path]:
    """
    Create result and checkpoint directories for an experiment.
    """
    results_dir = Path("results") / experiment_name
    ckpt_dir = Path("checkpoints") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, ckpt_dir


class CSVLogger:
    """
    Lightweight CSV logger that appends rows with a fixed header.
    """

    def __init__(self, filepath: Path, fieldnames: Iterable[str]):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        if not self.filepath.exists():
            with self.filepath.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict):
        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_metrics(path: Path, metrics: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)

