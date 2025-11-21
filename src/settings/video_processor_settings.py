from pydantic import BaseModel, Field

class VideoSettings(BaseModel):
    enable_expression: bool = Field(default=True, description="Run PyFeat for facial expressions.")
    enable_identity: bool = Field(default=True, description="Run InsightFace for identity embeddings.")
    enable_pose: bool = Field(default=True, description="Run PARE for 3D body pose.")
    feat_model_list: list = Field(default=["retinaface", "img2emotion", "detection"], description="PyFeat models.")
    insightface_model_name: str = Field(default="buffalo_l", description="InsightFace model pack.")
    pare_model_checkpoint: str = Field(default="data/pare_checkpoints/pare_checkpoint.ckpt", description="Path to PARE model.")
