# src/routers/dataset.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form
from sqlalchemy.orm import Session, sessionmaker
from typing import List, Optional
from loguru import logger
import os

from src.schemas.dataset import ExportFormat, ImportFormat, DatasetResponse
from src.models.dataset import Dataset
from src.services.dataset_service import DatasetService
from src.services.annotations_pipeline import AnnotationPipelines
from src.services.s3_service import S3Service
from src.dependencies import get_db


router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    responses={404: {"description": "Not found"}},
)


def session_factory(db: Session = Depends(get_db)):
    return sessionmaker(bind=db.get_bind())

@router.post("/", response_model=DatasetResponse, summary="Créer un nouveau dataset")
async def create_dataset_endpoint(
    name: Optional[str] = Form(None, description="Nom du dataset"),
    files: List[UploadFile] = File(..., description="Fichiers à uploader"),
    labels: Optional[str] = Form(None, description="Labels supplémentaires"),
    output_format: ExportFormat = Form(..., description="Format de sortie"),
    db: Session = Depends(get_db)
):
    """
    Crée un nouveau dataset en téléchargeant des fichiers et en les traitant.

    - **name**: Nom du dataset (optionnel)
    - **files**: Liste de fichiers à uploader
    - **labels**: Labels supplémentaires (optionnel)
    - **output_format**: Format de sortie (voir enum ExportFormat)
    """
    annotation_pipeline = AnnotationPipelines()
    service = DatasetService(db, session_factory=session_factory, annotation_pipeline=annotation_pipeline)

    try:
        dataset = await service.create_dataset(name, files, labels, output_format.value)
        return dataset
    except Exception as e:
        logger.error(f"Erreur dans create_dataset_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[DatasetResponse], summary="Lister tous les datasets")
def list_datasets_endpoint(db: Session = Depends(get_db)):
    """
    Liste tous les datasets disponibles.
    """
    datasets = db.query(Dataset).all()
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse, summary="Obtenir un dataset par ID")
def get_dataset_endpoint(dataset_id: int, db: Session = Depends(get_db)):
    """
    Obtient un dataset spécifique par son ID.

    - **dataset_id**: ID du dataset à obtenir
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset non trouvé")
    return dataset


@router.put("/{dataset_id}", response_model=DatasetResponse, summary="Mettre à jour un dataset par ID")
async def update_dataset_endpoint(
    dataset_id: int,
    name: Optional[str] = Form(None, description="Nouveau nom du dataset"),
    files: Optional[List[UploadFile]] = File(None, description="Nouveaux fichiers à uploader"),
    labels: Optional[str] = Form(None, description="Nouveaux labels"),
    output_format: Optional[ExportFormat] = Form(None, description="Nouveau format de sortie"),
    db: Session = Depends(get_db)
):

    service = DatasetService(db)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset non trouvé")

    try:
        if files:
            # Traiter les nouveaux fichiers et remplacer les données existantes
            new_dataset = await service.create_dataset(
                name or dataset.name,
                files,
                labels,
                output_format.value if output_format else dataset.output_format
            )
            dataset.data = new_dataset.data
            dataset.name = name or dataset.name
            dataset.output_format = output_format.value if output_format else dataset.output_format
            db.commit()
            db.refresh(dataset)
        else:
            # Mettre à jour d'autres champs si nécessaire
            if name:
                dataset.name = name
            if labels:
                # Mettre à jour les labels si applicable
                dataset.data['labels'] = labels
            if output_format:
                dataset.output_format = output_format.value
            db.commit()
            db.refresh(dataset)
        return dataset
    except Exception as e:
        logger.error(f"Erreur dans update_dataset_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_id}", response_model=dict, summary="Supprimer un dataset par ID")
def delete_dataset_endpoint(dataset_id: int, db: Session = Depends(get_db)):
    """
    Supprime un dataset spécifique par son ID.

    - **dataset_id**: ID du dataset à supprimer
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset non trouvé")

    try:
        # Optionnel : supprimer le fichier du dataset sur S3
        s3_url = dataset.data.get("s3_url")
        if s3_url:
            # Implémenter la suppression depuis S3
            s3_bucket, s3_key = S3Service.parse_s3_url(s3_url)
            service = S3Service(
                endpoint_url=os.getenv("MINIO_URL", "http://localhost:9000"),
                access_key=os.getenv("MINIO_ACCESS_KEY", "minio"),
                secret_key=os.getenv("MINIO_SECRET_KEY", "minio123"),
                input_bucket=os.getenv("INPUT_BUCKET", "docs-input"),
                output_bucket=os.getenv("OUTPUT_BUCKET", "docs-output"),
                layouts_bucket=os.getenv("LAYOUTS_BUCKET", "layouts")
            )
            service.client.delete_object(Bucket=s3_bucket, Key=s3_key)
            logger.info(f"Fichier du dataset supprimé de S3 : {s3_url}")

        # Supprimer du database
        db.delete(dataset)
        db.commit()
        logger.info(f"Dataset avec ID {dataset_id} supprimé")
        return {"detail": "Dataset supprimé avec succès"}
    except Exception as e:
        logger.error(f"Erreur dans delete_dataset_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))