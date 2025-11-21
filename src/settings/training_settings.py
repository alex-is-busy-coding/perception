from pydantic import BaseModel, Field

class TrainingSettings(BaseModel):
    batch_size: int = Field(default=16, description="Batch size for training.")
    num_epochs: int = Field(default=50, description="Number of training epochs.")
    learning_rate: float = Field(default=1e-3, description="Learning rate for the optimizer.")
    latent_dim: int = Field(default=64, description="Dimension of the bottleneck (latent) layer.")
    checkpoint_dir: str = Field(default="checkpoints", description="Directory to save model checkpoints.")
