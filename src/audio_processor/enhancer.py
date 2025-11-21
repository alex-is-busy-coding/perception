import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation
import os
import logging

logger = logging.getLogger(__name__)


class AudioEnhancer:
    """
    A class to handle audio denoising using a pre-trained SpeechBrain model.
    Handles long audio files by processing them in overlapping chunks.
    """
    def __init__(
        self, 
        model_source: str = "speechbrain/sepformer-wham16k-enhancement", 
        savedir: str = "pretrained_models/sepformer-wham16k-enhancement",
        device: str = None
    ):
        """
        Initializes the AudioEnhancer with a specified model.
        """
        self.model_source = model_source
        self.savedir = savedir
        self.target_sample_rate = 16000
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"AudioEnhancer using device: {self.device}")

        run_opts = {"device": self.device}
        
        self.model = SepformerSeparation.from_hparams(
            source=self.model_source,
            savedir=self.savedir,
            run_opts=run_opts
        )

    def _load_and_resample(self, input_path: str) -> torch.Tensor:
        """Loads and resamples audio to the model's target sample rate."""
        wav, sr = torchaudio.load(input_path)
        
        wav = wav.to(self.device)

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        if sr != self.target_sample_rate:
            logger.info(f"Resampling audio from {sr} Hz to {self.target_sample_rate} Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, 
                new_freq=self.target_sample_rate
            ).to(self.device)
            wav = resampler(wav)
            
        return wav

    def enhance_audio_file(
        self, 
        input_path: str, 
        output_path: str,
        chunk_len_sec: int = 8,
        overlap_sec: float = 0.5
    ):
        """
        Denoises a single audio file and saves the output.
        Handles long files by processing in overlapping chunks.

        Args:
            input_path (str): Path to the noisy input audio file.
            output_path (str): Path to save the enhanced audio file.
            chunk_len_sec (int): Length of each chunk in seconds.
            overlap_sec (float): Overlap between chunks in seconds.
        """
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Enhancing audio file: {input_path}")
        
        wav = self._load_and_resample(input_path)
        num_samples = wav.shape[1]

        chunk_size = chunk_len_sec * self.target_sample_rate
        overlap = int(overlap_sec * self.target_sample_rate)
        step_size = chunk_size - overlap

        enhanced_chunks = []
        
        for start in range(0, num_samples, step_size):
            end = start + chunk_size
            chunk = wav[:, start:end]
            
            if chunk.shape[1] < chunk_size:
                padding = torch.zeros(
                    (1, chunk_size - chunk.shape[1]), 
                    device=self.device
                )
                chunk = torch.cat([chunk, padding], dim=1)

            est_sources = self.model.separate_batch(chunk)
            
            enhanced_chunk = est_sources[:, :, 0].detach()

            if start == 0:
                enhanced_chunks.append(enhanced_chunk)
            else:
                enhanced_chunks.append(enhanced_chunk[:, overlap:])

        enhanced_signal = torch.cat(enhanced_chunks, dim=1)
        enhanced_signal = enhanced_signal[:, :num_samples]
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        torchaudio.save(
            output_path, 
            enhanced_signal.cpu(), 
            self.target_sample_rate
        )
        logger.info(f"Enhanced audio saved to: {output_path}")
