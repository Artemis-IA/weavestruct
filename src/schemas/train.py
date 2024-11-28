# src/schemas/train.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
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
