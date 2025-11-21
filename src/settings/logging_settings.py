from pydantic import BaseModel, Field

class LoggingSettings(BaseModel):
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)."
    )
