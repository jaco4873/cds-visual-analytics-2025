from pydantic import BaseModel


class BasePydanticConfig(BaseModel):
    """Base configuration class using Pydantic BaseModel."""

    class Config:
        validate_default = True
        extra = "ignore"
