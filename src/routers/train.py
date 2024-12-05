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
from datasets import load_dataset, Features, Sequence, Value
from huggingface_hub import hf_hub_download, HfApi
from schemas.train import TrainInput, TrainResponse
from loguru import logger
import json
import shutil
import aiofiles
from mlflow.tracking import MlflowClient


class DatasetSource(str, Enum):
    file = "file"
    s3 = "s3"
    huggingface = "huggingface"


router = APIRouter()

@router.post("/", response_model=TrainResponse)
async def train_model(
    # Dataset parameters
    # source: DatasetSource = Form(..., description="Select dataset source: 'file', 's3', or 'huggingface'"),
    dataset_file: Optional[UploadFile] = File(None, description="Upload dataset file (JSON list)"),
    s3_url: Optional[str] = Form(None, description="S3 URL of the dataset (e.g., s3://bucket/path/file.json)"),
    huggingface_dataset: Optional[str] = Form(None, description="HuggingFace dataset name (e.g., 'knowledgator/GLINER-multi-task-synthetic-data')"),


    artifact_name: str = Form("knowledgator/gliner-multitask-large-v0.5", description="Name of the base model"),
    custom_model_name: Optional[str] = Form(None, description="Custom name for the fine-tuned model"),
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
    sources = [dataset_file, s3_url, huggingface_dataset]
    if sum(map(bool, sources)) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one dataset source: either 'dataset_file', 's3_url', or 'huggingface_dataset'."
        )
    # Dataset handling
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
        # Use HuggingFace API with token
        api = HfApi(token=settings.HF_API_TOKEN)
        try:
            files = api.list_repo_files(repo_id=huggingface_dataset)
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list files in HuggingFace repo '{huggingface_dataset}'. Ensure the dataset exists and the token is valid."
            )

        # Ensure at least one JSON file exists
        json_files = [f for f in files if f.endswith(".json")]
        if not json_files:
            raise HTTPException(status_code=400, detail=f"No JSON file found in the HuggingFace dataset repository '{huggingface_dataset}'.")

        target_file = json_files[0]
        try:
            local_json_path = hf_hub_download(repo_id=huggingface_dataset, filename=target_file, token=settings.HF_API_TOKEN)
        except Exception as e:
            logger.error(f"Failed to download '{target_file}' from HuggingFace: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to download dataset file '{target_file}' from HuggingFace.")

        try:
            hf_loaded = load_dataset("json", data_files=local_json_path)
        except Exception as e:
            logger.warning(f"Failed to load dataset with schema: {str(e)}. Retrying without features.")
            hf_loaded = load_dataset("json", data_files=local_json_path)

        split_name = "train" if "train" in hf_loaded else list(hf_loaded.keys())[0]
        dataset_path = f"/tmp/{huggingface_dataset.replace('/', '_')}.json"
        with open(dataset_path, "w") as f:
            json.dump([dict(x) for x in hf_loaded[split_name]], f)

    # Validate model availability
    available_models = model_manager.fetch_available_models()
    if artifact_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{artifact_name}' is not available. Available models: {available_models}"
        )


    # Lecture des paramètres d'entraînement depuis le fichier s'il est fourni
    training_params = {}
    if training_params_file and training_params_file.filename:
        content = await training_params_file.read()
        training_params = json.loads(content)

    # Valeurs par défaut pour les training params
    training_params.setdefault("learning_rate", learning_rate)
    training_params.setdefault("weight_decay", weight_decay)
    training_params.setdefault("batch_size", batch_size)
    training_params.setdefault("epochs", epochs)
    training_params.setdefault("compile_model", compile_model)


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
