import sys
import os
from pathlib import Path
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging.logging_config import setup_logging
from src.settings import get_settings
from src.model_training.normalization import FeatureNormalizer

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    settings = get_settings()
    
    data_dir = Path(settings.output_directory) / "features"
    stats_file = Path(settings.output_directory) / "feature_stats.pt"

    normalizer = FeatureNormalizer(stats_file=stats_file)
    
    try:
        n_features = normalizer.fit_and_save(data_dir)
        logger.info(f"Success! Computed stats for {n_features} features.")
    except Exception as e:
        logger.error(f"Failed to compute stats: {e}")
        sys.exit(1)