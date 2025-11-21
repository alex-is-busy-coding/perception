import logging
import os
import torch
import sys

logger = logging.getLogger(__name__)

class PoseEstimator:
    """
    Wrapper for PARE (Part Attention Regressor) for 3D body pose estimation.
    
    NOTE: PARE requires `pare` to be installed or in PYTHONPATH.
    It also requires SMPL model data in specific folders.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model_loaded = False
        
        try:
            import pare
            self.model_loaded = True
        except ImportError:
            logger.warning("PARE library not found. Pose estimation will be skipped.")
            logger.warning("Please install PARE: https://github.com/mkocabas/PARE")

    def process_video(self, video_path: str, output_dir: str):
        if not self.model_loaded:
            logger.error("PARE not loaded. Skipping.")
            return

        logger.info(f"Running PARE on {video_path}...")
                
        import subprocess
        
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "pare.demo",
            "--vid_file", video_path,
            "--output_folder", output_dir,
            "--ckpt", self.checkpoint_path,
            "--save_obj",
            "--draw_keypoints"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"PARE output saved to {output_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"PARE execution failed: {e}")
            logger.error("Ensure you have downloaded SMPL data and PARE checkpoints.")