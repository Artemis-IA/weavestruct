# src/schemas/train.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime

class TrainInput(BaseModel):
    artifact_name: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    custom_model_name: str = Field(..., example="")
    s3_bucket: Optional[str] = Field(None, example="model-artifacts")
    train_data: str  # Path to the training data
    split_ratio: float = Field(0.9, example=0.9)
    learning_rate: float = Field(5e-6, example=5e-6)
    weight_decay: float = Field(0.01, example=0.01)
    batch_size: int = Field(8, example=8)
    epochs: int = Field(1, example=1)
    compile_model: bool = Field(False, example=False)

class TrainRequest(BaseModel):
    dataset_id: int = Field(..., example=1)
    epochs: int = Field(10, example=20)
    batch_size: int = Field(32, example=64)

class TrainResponse(BaseModel):
    id: int
    run_id: str
    dataset_id: int
    epochs: int
    batch_size: int
    status: str
    created_at: datetime
    class Config:
        from_attributes = True
