import logging
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .normalization import FeatureNormalizer

logger = logging.getLogger(__name__)

class AcousticFeatureDataset(Dataset):
    def __init__(self, data_dir: Path, normalizer: FeatureNormalizer, return_metadata=False):
        self.data_dir = data_dir
        self.normalizer = normalizer
        self.return_metadata = return_metadata
        
        self.file_list = list(self.data_dir.glob("*.csv"))
        if not self.file_list:
            raise FileNotFoundError(f"No .csv files found in {data_dir}")
            
        if not self.normalizer.loaded:
            self.normalizer.load()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        features_df = pd.read_csv(file_path, header=0, index_col=[0, 1, 2])
        features = torch.tensor(features_df.values, dtype=torch.float32)
        
        normalized_features = self.normalizer.normalize(features)
        
        if self.return_metadata:
            return normalized_features, file_path.stem
        else:
            return normalized_features

def pad_collate(batch):
    """
    Pads sequences to the max length in a batch.
    Returns: (padded_seqs, mask, [optional_metadata])
    Shape of padded_seqs: (Batch, Features, Time) for Conv1d
    """
    metadata = None
    if isinstance(batch[0], tuple):
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        sequences = [item[0] for item in batch]
        metadata = [item[1] for item in batch]
    else:
        batch.sort(key=lambda x: x.shape[0], reverse=True)
        sequences = batch

    lengths = torch.tensor([s.shape[0] for s in sequences])
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    mask = torch.arange(padded_sequences.size(1))[None, :] < lengths[:, None]
    
    padded_sequences = padded_sequences.permute(0, 2, 1)

    if metadata:
        return padded_sequences, mask, metadata
    else:
        return padded_sequences, mask