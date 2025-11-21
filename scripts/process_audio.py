import os
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audio_processor.pipeline import AudioProcessingPipeline
from src.logging.logging_config import setup_logging
from src.settings import get_settings

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        settings = get_settings()
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        sys.exit(1)

    setup_logging(log_level=settings.logging.level)

    if not settings.hf_token:
         settings.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    if not settings.hf_token or settings.hf_token == "YOUR_HF_TOKEN_HERE":
        logger.error("Valid Hugging Face token required for Audio Processing.")
        sys.exit(1)

    logger.info(f"Starting Audio Pipeline [ENV: {os.getenv('APP_ENV', 'dev')}]")

    audio_pipeline = AudioProcessingPipeline(
        output_directory=settings.output_directory,
        hf_token=settings.hf_token,
        enhancer_model_source=settings.enhancer.model_source,
        enhancer_savedir=settings.enhancer.savedir,
        asr_model_name=settings.diarizer.asr_model_name,
        diarize_model_name=settings.diarizer.diarize_model_name,
        batch_size=settings.diarizer.batch_size,
        device=settings.enhancer.device
    )
    
    audio_pipeline.process_directory(
        input_directory=settings.input_directory,
        max_files=settings.max_files_to_process
    )