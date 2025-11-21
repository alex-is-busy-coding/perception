import logging
import torch
import os
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm

from .autoencoder import LitConv1D_AE
from .normalization import FeatureNormalizer

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles loading a trained Autoencoder and generating latent embeddings 
    from a dataloader.
    """
    def __init__(self, checkpoint_dir: Path, stats_file: Path, device: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir
        self.stats_file = stats_file
        
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        
        self.model = None
        self.normalizer = None

    def _get_best_checkpoint(self) -> Optional[Path]:
        """Finds the most recently modified .ckpt file."""
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            return None
        return sorted(checkpoints, key=os.path.getmtime, reverse=True)[0]

    def load_resources(self):
        self.normalizer = FeatureNormalizer(stats_file=self.stats_file)
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Stats file missing at {self.stats_file}")
        self.normalizer.load()

        ckpt_path = self._get_best_checkpoint()
        if not ckpt_path:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        
        logger.info(f"Loading checkpoint: {ckpt_path}")
        self.model = LitConv1D_AE.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def generate(self, data_loader) -> Tuple[torch.Tensor, List[str]]:
        """
        Runs the inference loop on the provided data loader using MASKED pooling.
        Combines Average Pooling and Max Pooling.
        """
        if not self.model:
            self.load_resources()
        
        all_embeddings = []
        all_metadata = []

        for sequences, mask, metadata in tqdm(data_loader, desc="Generating Embeddings"):
            sequences = sequences.to(self.device)
            mask = mask.to(self.device)
            
            with torch.no_grad():
                latent_seq = self.model.model.encoder(sequences)
                mask_float = mask.float().unsqueeze(1)
                mask_downsampled = F.interpolate(
                    mask_float, 
                    size=latent_seq.shape[2], 
                    mode='nearest'
                )
                
                latent_masked = latent_seq * mask_downsampled
                latent_sum = latent_masked.sum(dim=2)
                mask_sum = mask_downsampled.sum(dim=2)
                avg_pool = latent_sum / (mask_sum + 1e-8)

                latent_for_max = latent_seq.clone()
                mask_expanded = mask_downsampled.expand_as(latent_seq)
                latent_for_max[mask_expanded == 0] = -1e9
                max_pool = latent_for_max.max(dim=2)[0]

                combined_embedding = torch.cat([avg_pool, max_pool], dim=1)
                                
                all_embeddings.append(combined_embedding.cpu())
                all_metadata.extend(metadata)

        if not all_embeddings:
            return torch.tensor([]), []

        return torch.cat(all_embeddings, dim=0), all_metadata