import logging
import os
from feat import Detector

logger = logging.getLogger(__name__)

class ExpressionExtractor:
    """
    Wrapper for Py-Feat to extract Action Units (AUs) and emotions from video.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info("Loading Py-Feat Detector (RetinaFace, ResNet, GAN)...")
        self.detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            emotion_model="resmasknet",
            facepose_model="img2pose",
            device=self.device
        )
        logger.info("Py-Feat Detector initialized.")

    def process_video(self, video_path: str, output_csv_path: str):
        """
        Runs detection on a video file and saves the CSV.
        """
        logger.info(f"Running Expression Extraction on {video_path}...")
        
        try:
            video_prediction = self.detector.detect_video(
                inputFname=video_path,
                skip_frames=1,
                verbose=False
            )
            
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            video_prediction.to_csv(output_csv_path, index=False)
            logger.info(f"Expression features saved to {output_csv_path}")
            
        except Exception as e:
            logger.error(f"Py-Feat failed on {video_path}: {e}")