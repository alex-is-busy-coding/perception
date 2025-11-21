from typing import Optional, Literal
from pydantic import BaseModel, Field

WhisperModelSize = Literal[
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo"
]

class EnhancerSettings(BaseModel):
    model_source: str = Field(
        default="speechbrain/sepformer-wham16k-enhancement",
        description="HuggingFace model ID for the speech enhancement/separation model."
    )
    savedir: str = Field(
        default="pretrained_models/sepformer-wham16k-enhancement",
        description="Local directory to cache the downloaded enhancement model."
    )
    device: Optional[str] = Field(
        default=None,
        description="Device to run the model on (e.g., 'cuda', 'cpu'). If None, auto-detects."
    )

class DiarizerSettings(BaseModel):
    asr_model_name: WhisperModelSize = Field(
        default="large-v3",
        description="Whisper model size/name for ASR (e.g., 'medium', 'large-v3')."
    )
    diarize_model_name: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="HuggingFace model ID for the speaker diarization pipeline."
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for the ASR transcription process."
    )
