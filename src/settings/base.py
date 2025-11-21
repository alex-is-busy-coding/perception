import os
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

from .logging_settings import LoggingSettings
from .audio_processor_settings import EnhancerSettings, DiarizerSettings
from .video_processor_settings import VideoSettings
from .training_settings import TrainingSettings

class AppSettings(BaseSettings):
    """
    Main application settings.
    Loads from default values, then overrides with {env}.config.yaml,
    then overrides with environment variables.
    """
    
    input_directory: str = Field(
        default="data/raw",
        description="Directory containing raw .wav files to process."
    )
    output_directory: str = Field(
        default="data",
        description="Root directory for saving processed outputs (enhanced audio, transcripts, features)."
    )
    max_files_to_process: Optional[int] = Field(
        default=None,
        description="Limit the number of files to process. Useful for testing/dev. None means all files."
    )
    
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    enhancer: EnhancerSettings = Field(default_factory=EnhancerSettings)
    diarizer: DiarizerSettings = Field(default_factory=DiarizerSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face authentication token. Required for Pyannote diarization."
    )

    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        env_file='.env',
        extra='ignore'
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Custom source priority:
        1. Init args
        2. Env vars
        3. {env}.config.yaml (dev/prod)
        4. default.config.yaml
        """
        app_env = os.getenv("APP_ENV", "dev")
        
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=f"config/{app_env}.config.yaml"),
            YamlConfigSettingsSource(settings_cls, yaml_file="config/default.config.yaml"),
        )

@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()