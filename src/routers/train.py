# routers/train.py

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query, Body
from enum import Enum
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from pathlib import Path
from dependencies import get_db, get_s3_service, get_mlflow_service, get_model_manager
from config import settings
from services.train_service import TrainService
from services.model_manager import ModelManager, ModelSource, ModelInfoFilter
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from schemas.train import TrainInput, TrainResponse
from loguru import logger
import json
import shutil
import aiofiles
from mlflow.tracking import MlflowClient

router = APIRouter()

@router.post("/", response_model=TrainResponse)
async def train_model(
    artifact_name: str = Form(..., description="Name of the base model"),
    custom_model_name: Optional[str] = Form(None, description="Custom name for the fine-tuned model"),

    # Un seul des trois doit être fourni
    dataset_file: Optional[UploadFile] = File(None, description="Upload dataset file (JSON list)"),
    s3_url: Optional[str] = Form(None, description="S3 URL of the dataset (e.g. s3://bucket/path/file.json)"),
    huggingface_dataset: Optional[str] = Form(None, description="HuggingFace dataset name (e.g. 'knowledgator/GLINER-multi-task-synthetic-data')"),

    split_ratio: float = Form(0.9, description="Train/Eval split ratio"),
    learning_rate: float = Form(1e-5, description="Learning rate"),
    weight_decay: float = Form(0.01, description="Weight decay"),
    batch_size: int = Form(16, description="Batch size"),
    epochs: int = Form(3, description="Number of epochs"),
    compile_model: bool = Form(False, description="Compile the model for faster training"),

    training_params_file: Optional[UploadFile] = File(None, description="Upload training parameters JSON file"),

    db: Session = Depends(get_db),
    s3_service: S3Service = Depends(get_s3_service),
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
    model_manager: ModelManager = Depends(get_model_manager)
):
    # Vérifier exclusivité des sources de dataset
    sources = [dataset_file is not None, s3_url is not None, huggingface_dataset is not None]
    if sum(sources) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one dataset source: either 'dataset_file', 's3_url', or 'huggingface_dataset'."
        )

    # Lecture des paramètres d'entraînement depuis le fichier s'il est fourni
    training_params = {}
    if training_params_file:
        content = await training_params_file.read()
        training_params = json.loads(content)

    # Valeurs par défaut pour les training params
    training_params.setdefault("learning_rate", learning_rate)
    training_params.setdefault("weight_decay", weight_decay)
    training_params.setdefault("batch_size", batch_size)
    training_params.setdefault("epochs", epochs)
    training_params.setdefault("compile_model", compile_model)

    # Récupération du dataset
    dataset_path = None
    if dataset_file:
        dataset_path = f"/tmp/{dataset_file.filename}"
        async with aiofiles.open(dataset_path, 'wb') as out_file:
            content = await dataset_file.read()
            await out_file.write(content)
    elif s3_url:
        s3_info = s3_service.parse_s3_url(s3_url)
        if not s3_info:
            raise HTTPException(status_code=400, detail="Invalid S3 URL.")
        bucket_name, object_key = s3_info
        dataset_path = f"/tmp/{object_key.split('/')[-1]}"
        success = s3_service.download_file(bucket_name, object_key, Path(dataset_path))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to download dataset from S3.")
    elif huggingface_dataset:
        # On utilise datasets pour télécharger le dataset, 
        # on suppose qu'il s'agit d'un dataset splitté (train, validation, etc.)
        from datasets import load_dataset
        hf_data = load_dataset(huggingface_dataset)
        # On prend le split 'train' comme base
        # Convertir en liste de dict si ce n'est pas déjà le cas
        train_split = hf_data['train']
        data_list = [dict(row) for row in train_split]
        dataset_path = f"/tmp/{huggingface_dataset.replace('/', '_')}.json"
        with open(dataset_path, "w") as f:
            json.dump(data_list, f)

    # Validation que le modèle existe
    available_models = model_manager.fetch_available_models()
    if artifact_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{artifact_name}' is not available. Available models: {available_models}"
        )

    # Préparer l'input pour le TrainService
    train_input = TrainInput(
        artifact_name=artifact_name,
        custom_model_name=custom_model_name,
        dataset_path=dataset_path,
        split_ratio=split_ratio,
        learning_rate=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"],
        batch_size=training_params["batch_size"],
        epochs=training_params["epochs"],
        compile_model=training_params["compile_model"],
    )

    train_service = TrainService(
        db=db,
        s3_service=s3_service,
        mlflow_service=mlflow_service,
        model_manager=model_manager
    )

    try:
        response = train_service.train_model(train_input)
        return response
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

