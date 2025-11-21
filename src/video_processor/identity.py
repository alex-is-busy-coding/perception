import logging
import cv2
import os
import numpy as np
import pandas as pd
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort
from tqdm import tqdm

logger = logging.getLogger(__name__)

class IdentityExtractor:
    """
    Wrapper for InsightFace to extract identity embeddings from video frames.
    """
    def __init__(self, model_name: str = "buffalo_l", device: str = "cuda"):
        self.device = device
        
        available_providers = ort.get_available_providers()
        
        providers = ['CPUExecutionProvider']
        
        if device == "cuda" and 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("Using CUDA for InsightFace.")
        elif 'CoreMLExecutionProvider' in available_providers:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            logger.info("Using CoreML/CPU for InsightFace.")
        else:
            logger.info("CUDA not found. Falling back to CPU for InsightFace.")

        logger.info(f"Loading InsightFace model: {model_name}...")
        self.app = FaceAnalysis(name=model_name, providers=providers)
        
        ctx_id = 0 if providers[0] == 'CUDAExecutionProvider' else -1
        
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace initialized.")

    def process_video(self, video_path: str, output_csv_path: str):
        """
        Iterates through video frames, extracts faces, and saves embeddings.
        """
        logger.info(f"Running Identity Extraction on {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []

        frame_idx = 0
        with tqdm(total=total_frames, desc="InsightFace Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    faces = self.app.get(frame)
                    
                    for face in faces:
                        results.append({
                            "frame": frame_idx,
                            "bbox": face.bbox.tolist(),
                            "kps": face.kps.tolist(),
                            "det_score": float(face.det_score),
                            "embedding": face.embedding.tolist()
                        })
                except Exception as e:
                    logger.warning(f"Frame {frame_idx} failed: {e}")
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()

        if results:
            df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False) 
            logger.info(f"Identity embeddings saved to {output_csv_path}")
        else:
            logger.warning(f"No faces detected in {video_path}")