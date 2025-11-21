import os
import logging
import pandas as pd
from tqdm import tqdm
from typing import Optional

from .enhancer import AudioEnhancer
from .diarizer import SpeakerDiarizer
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    Orchestrates the entire audio processing workflow.
    """
    
    def __init__(
        self, 
        output_directory: str,
        hf_token: Optional[str],
        enhancer_model_source: str,
        enhancer_savedir: str,
        asr_model_name: str,
        diarize_model_name: str,
        batch_size: int,
        device: Optional[str] = None
    ):
        """
        Initializes pipeline components with explicit parameters.
        """
        self.output_dir = output_directory
        
        self.dirs = {
            "enhanced": os.path.join(self.output_dir, "enhanced"),
            "transcripts": os.path.join(self.output_dir, "transcripts"),
            "plots": os.path.join(self.output_dir, "diarization_plots"),
            "features": os.path.join(self.output_dir, "features")
        }

        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)
        
        logger.info("Initializing pipeline components...")
        
        self.enhancer = AudioEnhancer(
            model_source=enhancer_model_source,
            savedir=enhancer_savedir,
            device=device
        )
        
        self.diarizer = SpeakerDiarizer(
            hf_token=hf_token,
            asr_model_name=asr_model_name,
            diarize_model_name=diarize_model_name,
            batch_size=batch_size,
            device=device
        )
        
        self.feature_extractor = FeatureExtractor()
        
        logger.info("Pipeline components initialized.")

    def process_single_file(self, audio_path: str) -> None:
        """
        Runs the full processing pipeline on a single audio file.
        """
        filename = os.path.basename(audio_path)
        base_name = os.path.splitext(filename)[0]

        enhanced_path = os.path.join(self.dirs["enhanced"], f"{base_name}.wav")
        
        self.enhancer.enhance_audio_file(
            input_path=audio_path,
            output_path=enhanced_path
        )

        transcript_path = os.path.join(self.dirs["transcripts"], f"{base_name}.csv")
        plot_path = os.path.join(self.dirs["plots"], f"{base_name}.png")
        
        self.diarizer.process(
            audio_file_path=enhanced_path, 
            output_path=transcript_path,
            plot_output_path=plot_path
        )

        opensmile_features = self.feature_extractor.extract_opensmile_features(
            audio_path=enhanced_path
        )
        
        if not opensmile_features.empty:
            feature_path = os.path.join(self.dirs["features"], f"{base_name}.csv")
            opensmile_features.to_csv(feature_path)
            logger.info(f"Saved features to {feature_path}")
        else:
            logger.warning(f"No features extracted for {filename}")

    def process_directory(self, input_directory: str, max_files: Optional[int] = None):
        """
        Runs the pipeline on files in the specified input directory.
        """
        if not os.path.isdir(input_directory):
            logger.error(f"Input directory not found: {input_directory}")
            return

        audio_files = [
            os.path.join(input_directory, f) 
            for f in os.listdir(input_directory) 
            if f.lower().endswith('.wav')
        ]
        
        if not audio_files:
            logger.warning(f"No .wav files found in {input_directory}")
            return

        if max_files:
            logger.info(f"DEV MODE: Limiting processing to first {max_files} files.")
            audio_files = audio_files[:max_files]

        logger.info(f"Found {len(audio_files)} files to process.")
        
        files_processed = 0
        for audio_file in tqdm(audio_files, desc="Processing Pipeline"):
            try:
                self.process_single_file(audio_file)
                files_processed += 1
            except Exception as e:
                logger.error(f"Failed to process '{audio_file}': {e}", exc_info=True)
        
        logger.info(f"Batch processing complete. {files_processed}/{len(audio_files)} files processed.")