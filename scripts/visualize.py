import os
import sys
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging.logging_config import setup_logging
from src.settings import get_settings
from src.model_training.data_loader import AcousticFeatureDataset, pad_collate
from src.model_training.inference import EmbeddingGenerator

setup_logging()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def main():
    settings = get_settings()
    
    DATA_DIR = Path(settings.output_directory) / "features"
    STATS_FILE = Path(settings.output_directory) / "feature_stats.pt"
    CHECKPOINT_DIR = Path(settings.training.checkpoint_dir)
    LOG_DIR = Path("tensorboard_logs")

    generator = EmbeddingGenerator(
        checkpoint_dir=CHECKPOINT_DIR,
        stats_file=STATS_FILE
    )
    
    try:
        generator.load_resources()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    dataset = AcousticFeatureDataset(
        data_dir=DATA_DIR, 
        normalizer=generator.normalizer, 
        return_metadata=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=settings.training.batch_size,
        collate_fn=pad_collate,
        num_workers=0
    )
    
    logger.info(f"Found {len(dataset)} files to process.")

    embeddings, metadata = generator.generate(loader)

    if len(embeddings) == 0:
        logger.warning("No embeddings generated.")
        return

    logger.info(f"Embeddings shape: {embeddings.shape}")

    writer = SummaryWriter(LOG_DIR)
    writer.add_embedding(
        embeddings,
        metadata=metadata,
        tag="participant_embeddings"
    )
    writer.close()
    
    logger.info(f"Embeddings saved to {LOG_DIR}")

if __name__ == "__main__":
    main()