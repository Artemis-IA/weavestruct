# routers/train.py

from fastapi import APIRouter, Depends, HTTPException
from schemas.train import TrainRequest, TrainResponse
from services.train_service import TrainService
from dependencies import get_db, get_s3_service
from sqlalchemy.orm import Session
from services.s3_service import S3Service

router = APIRouter()

@router.post("/", response_model=TrainResponse, tags=["Training"])
def train_endpoint(
    request: TrainRequest,
    db: Session = Depends(get_db),
    s3_service: S3Service = Depends(get_s3_service)
):
    """
    Endpoint to train the NER model.
    """
    try:
        train_service = TrainService(db=db, s3_service=s3_service)
        training_run = train_service.train_model(request)
        return training_run
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
