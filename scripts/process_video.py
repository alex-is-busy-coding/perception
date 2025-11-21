import os
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.video_processor.pipeline import VideoProcessingPipeline
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

    logger.info(f"Starting Video Pipeline [ENV: {os.getenv('APP_ENV', 'dev')}]")

    video_pipeline = VideoProcessingPipeline(
        settings=settings.video,
        output_dir=settings.output_directory
    )
    
    video_pipeline.process_directory(
        input_dir=settings.input_directory,
        max_files=settings.max_files_to_process
    )