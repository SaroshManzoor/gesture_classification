from pydantic import BaseModel
from pydantic.fields import Field


class ModelAccuracy(BaseModel):
    classifier: str = Field(nullable=False)
    accuracy: float = Field(nullable=False)


class Prediction(BaseModel):
    classifier: str = Field(nullable=False)
    predicted_class: str = Field(nullable=False)
