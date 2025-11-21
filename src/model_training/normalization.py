import torch
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class FeatureNormalizer:
    """
    Computes, saves, and applies Z-score normalization (mean/std) to features.
    """
    def __init__(self, stats_file: Path):
        self.stats_file = Path(stats_file)
        self.mean = None
        self.std = None
        self.loaded = False

    def load(self):
        """Loads stats from disk."""
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Stats file not found: {self.stats_file}")
        
        stats = torch.load(self.stats_file)
        self.mean = stats['mean']
        self.std = stats['std']
        self.std[self.std == 0] = 1.0
        self.loaded = True
        logger.debug(f"Loaded feature stats from {self.stats_file}")

    def fit_and_save(self, data_dir: Path):
        """Computes mean/std from all CSVs in data_dir and saves to stats_file."""
        logger.info(f"Computing stats from {data_dir}...")
        files = list(data_dir.glob("*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No .csv files found in {data_dir}")

        all_data = []
        for f in files:
            df = pd.read_csv(f, header=0, index_col=[0, 1, 2])
            all_data.append(torch.tensor(df.values, dtype=torch.float32))
        
        full_tensor = torch.cat(all_data, dim=0)
        self.mean = torch.mean(full_tensor, dim=0)
        self.std = torch.std(full_tensor, dim=0)
        
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({'mean': self.mean, 'std': self.std}, self.stats_file)
        self.loaded = True
        logger.info(f"Stats computed and saved to {self.stats_file}")
        return self.mean.shape[0]

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            self.load()
        return (features - self.mean) / (self.std + 1e-8)

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            self.load()
        return (features * (self.std + 1e-8)) + self.mean