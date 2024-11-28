from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional
from loguru import logger
from sqlalchemy.orm import Session

from dependencies import get_db
from models.dataset import Dataset
from schemas.dataset import DatasetCreate, DatasetResponse
from services.dataset_service import DatasetService

router = APIRouter()


@router.post("/", response_model=DatasetResponse, tags=["Datasets"])
async def create_dataset(
    name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    output_format: str = Form("json-ner"),
    db: Session = Depends(get_db),
):
    """
    Crée un jeu de données NER à partir de fichiers téléchargés.
    """
    logger.info(f"Creating dataset with name: {name}")
    try:
        dataset_service = DatasetService(db=db)
        dataset = await dataset_service.create_dataset(
            name=name, files=files, labels=labels, output_format=output_format
        )
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
