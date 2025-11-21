import os
import logging
import torch
from tqdm import tqdm
from typing import Optional

from src.settings.video_processor_settings import VideoSettings
from .expression import ExpressionExtractor
from .identity import IdentityExtractor
from .pose import PoseEstimator

logger = logging.getLogger(__name__)

class VideoProcessingPipeline:
    """
    Orchestrates video analysis: Expressions, Identity, and Pose.
    """
    def __init__(self, settings: VideoSettings, output_dir: str):
        self.settings = settings
        self.output_dir = output_dir
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.dirs = {
            "expression": os.path.join(self.output_dir, "video", "expression"),
            "identity": os.path.join(self.output_dir, "video", "identity"),
            "pose": os.path.join(self.output_dir, "video", "pose"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        logger.info("Initializing Video Pipeline...")
        
        self.expression_extractor = None
        if self.settings.enable_expression:
            self.expression_extractor = ExpressionExtractor(device=self.device)
            
        self.identity_extractor = None
        if self.settings.enable_identity:
            self.identity_extractor = IdentityExtractor(
                model_name=self.settings.insightface_model_name,
                device=self.device
            )
            
        self.pose_estimator = None
        if self.settings.enable_pose:
            self.pose_estimator = PoseEstimator(
                checkpoint_path=self.settings.pare_model_checkpoint,
                device=self.device
            )

    def process_video(self, video_path: str):
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        logger.info(f"Processing video: {filename}")

        if self.expression_extractor:
            out_path = os.path.join(self.dirs["expression"], f"{base_name}.csv")
            if not os.path.exists(out_path):
                self.expression_extractor.process_video(video_path, out_path)
            else:
                logger.info(f"Skipping Expression (exists): {out_path}")

        if self.identity_extractor:
            out_path = os.path.join(self.dirs["identity"], f"{base_name}.csv")
            if not os.path.exists(out_path):
                self.identity_extractor.process_video(video_path, out_path)
            else:
                logger.info(f"Skipping Identity (exists): {out_path}")
        
        if self.pose_estimator:
            out_dir = os.path.join(self.dirs["pose"], base_name)
            self.pose_estimator.process_video(video_path, out_dir)

    def process_directory(self, input_dir: str, max_files: Optional[int] = None):
        videos = [
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
        ]
        
        if not videos:
            logger.warning(f"No video files found in {input_dir}")
            return

        if max_files:
            videos = videos[:max_files]
            
        for video in tqdm(videos, desc="Video Pipeline"):
            self.process_video(video)