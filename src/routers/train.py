from fastapi import APIRouter, Depends, HTTPException
from typing import List
from schemas.train import TrainRequest, TrainResponse
from services.train_service import TrainService
from dependencies import get_db
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/", response_model=TrainResponse, tags=["Training"])
def train_endpoint(request: TrainRequest, db: Session = Depends(get_db)):
    """
    Endpoint pour entraîner le modèle NER.
    """
    try:
        train_service = TrainService(db=db)
        training_run = train_service.train_model(request)
        return training_run
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
