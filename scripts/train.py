import os
import sys
import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_training.data_loader import AcousticFeatureDataset, pad_collate
from src.model_training.normalization import FeatureNormalizer
from src.model_training.autoencoder import LitConv1D_AE
from src.logging.logging_config import setup_logging
from src.settings import get_settings

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    settings = get_settings()
    
    DATA_DIR = Path(settings.output_directory) / "features"
    STATS_FILE = Path(settings.output_directory) / "feature_stats.pt"
    CHECKPOINT_DIR = Path(settings.training.checkpoint_dir)

    normalizer = FeatureNormalizer(stats_file=STATS_FILE)
    if not STATS_FILE.exists():
        logger.info("Stats file not found. Computing now...")
        try:
            normalizer.fit_and_save(DATA_DIR)
        except FileNotFoundError:
            logger.error(f"No features found in {DATA_DIR}. Run 'make process' first.")
            sys.exit(1)
    else:
        normalizer.load()

    n_features = normalizer.mean.shape[0]
    logger.info(f"Detected {n_features} input features.")

    dataset = AcousticFeatureDataset(data_dir=DATA_DIR, normalizer=normalizer)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=settings.training.batch_size, 
        shuffle=True, 
        collate_fn=pad_collate,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=settings.training.batch_size, 
        collate_fn=pad_collate,
        num_workers=0
    )

    model = LitConv1D_AE(
        n_features=n_features,
        latent_dim=settings.training.latent_dim,
        learning_rate=settings.training.learning_rate
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=CHECKPOINT_DIR,
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = pl.Trainer(
        max_epochs=settings.training.num_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    logger.info("Starting model training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("Training complete!")
    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")