# routers/datasets.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional
from loguru import logger
from sqlalchemy.orm import Session

from dependencies import get_db
from models.dataset import Dataset
from schemas.dataset import DatasetResponse
from services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    output_format: str = Form("json-ner"),
    db: Session = Depends(get_db),
):
    """
    Create a NER dataset from uploaded files.
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


@router.get("/", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    """
    List all datasets.
    """
    datasets = db.query(Dataset).all()
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=str(dataset.created_at),
        )
        for dataset in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get a dataset by ID.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        data=dataset.data,
        created_at=str(dataset.created_at),
    )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Delete a dataset by ID.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()
    return {"message": f"Dataset {dataset_id} deleted successfully"}
