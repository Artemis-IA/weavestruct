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

router = APIRouter()


@router.get("/available_models", response_model=List[Dict[str, Any]])
async def available_models(
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
):
    try:
        models = mlflow_service.search_registered_models()
        return models
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available models")

@router.get("/get_huggingface_models")
async def get_gliner_models(
    sort_by: Optional[ModelInfoFilter] = Query(ModelInfoFilter.size, description="Sort models by 'size', 'recent', 'name', 'downloads', or 'likes'"),

    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Fetch Hugging Face models with optional sorting.
    """
    try:
        models = model_manager.fetch_hf_models(sort_by=sort_by)
        return models
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to fetch Hugging Face models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.post("/upload_model_from_huggingface")
async def upload_model_huggingface(
    model_name: str = Form(..., description="Name of the Hugging Face model"),
    register_name: str = Form(..., description="Name to register the model in MLflow"),
    artifact_path: str = Form("model_artifacts", description="Path to save artifacts in MLflow"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Upload a Hugging Face model and register it in MLflow.
    """
    try:
        model_manager.fetch_and_register_hf_model(
            model_name=model_name,
            artifact_path=artifact_path,
            register_name=register_name,
        )
        return {"message": f"Model '{model_name}' registered successfully as '{register_name}'."}
    except Exception as e:
        logger.error(f"Error during model upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {e}")

@router.post("/", response_model=TrainResponse)
async def train_model(
    model_name: str = Form(..., description="Name of the base model"),
    custom_model_name: Optional[str] = Form(None, description="Custom name for the fine-tuned model"),
    dataset_file: Optional[UploadFile] = File(None, description="Upload dataset file"),
    s3_url: Optional[str] = Form(None, description="S3 URL of the dataset"),
    training_params_file: Optional[UploadFile] = File(None, description="Upload training parameters JSON file"),
    db: Session = Depends(get_db),
    s3_service: S3Service = Depends(get_s3_service),
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
    model_manager: ModelManager = Depends(get_model_manager)
):
    try:
        # Validate model_name
        available_models = model_manager.fetch_available_models()
        if model_name not in available_models:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not available. Available models: {available_models}")

        # Read training parameters from file if provided
        training_params = {}
        if training_params_file:
            content = await training_params_file.read()
            training_params = json.loads(content)
        else:
            # Default training parameters
            training_params = {
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "batch_size": 16,
                "epochs": 3,
                "compile_model": False,
            }

        # Handle dataset
        dataset_path = None
        if dataset_file:
            dataset_path = f"/tmp/{dataset_file.filename}"
            async with aiofiles.open(dataset_path, 'wb') as out_file:
                content = await dataset_file.read()
                await out_file.write(content)
        elif s3_url:
            # Download dataset from S3
            s3_info = s3_service.parse_s3_url(s3_url)
            if not s3_info:
                raise ValueError("Invalid S3 URL.")
            bucket_name, object_key = s3_info
            dataset_path = f"/tmp/{object_key.split('/')[-1]}"
            success = s3_service.download_file(bucket_name, object_key, Path(dataset_path))
            if not success:
                raise ValueError("Failed to download dataset from S3.")
        else:
            raise HTTPException(status_code=400, detail="No dataset provided. Please upload a dataset file or provide an S3 URL.")

        # Prepare the train input
        train_input = TrainInput(
            model_name=model_name,
            custom_model_name=custom_model_name,
            dataset_path=dataset_path,
            **training_params
        )

        train_service = TrainService(
            db=db,
            s3_service=s3_service,
            mlflow_service=mlflow_service,
            model_manager=model_manager
        )
        response = train_service.train_model(train_input)
        return response
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_model_artifact")
async def upload_model_artifact(
    model_source: ModelSource = Form(..., description="Source of the model: 'huggingface' or 'local'"),
    model_name: str = Form("knowledgator/gliner-multitask-large-v0.5", description="Name of the model to be registered"),
    task: Optional[str] = Form(None, description="Task type for the model (e.g., 'ner', 'text-classification')"),
    local_model_file: Optional[UploadFile] = File(None, description="Local model file (zip) if source is 'local'"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Upload the model artifact to MLflow from Hugging Face or local cache.
    """
    ALLOWED_MODELS = settings.MODELS
    try:
        # Validate that model_name is in the list of allowed models
        if model_name not in ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not allowed. Allowed models: {ALLOWED_MODELS}")

        # Check if model_name already exists in MLflow
        existing_models = model_manager.fetch_available_models()
        if model_name in existing_models:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' already exists in MLflow.")

        if model_source == ModelSource.huggingface:
            # Upload model from Hugging Face
            version = model_manager.upload_model_from_huggingface(
                model_name=model_name,
                task=task
            )
            return {"message": f"Model '{model_name}' uploaded from Hugging Face and registered successfully.", "version": version}

        elif model_source == ModelSource.local:
            if not local_model_file:
                raise HTTPException(status_code=400, detail="Local model file must be provided when source is 'local'.")

            # Save the uploaded local model to a temporary directory
            temp_dir = Path(f"/tmp/{model_name}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            local_model_file_path = temp_dir / local_model_file.filename
            async with aiofiles.open(local_model_file_path, 'wb') as out_file:
                content = await local_model_file.read()
                await out_file.write(content)

            # Assume the uploaded file is a zip file containing the model directory
            shutil.unpack_archive(str(local_model_file_path), extract_dir=str(temp_dir))

            # Upload model from local cache
            version = model_manager.upload_model_from_local(
                model_name=model_name,
                local_model_path=temp_dir,
                task=task
            )
            return {"message": f"Model '{model_name}' uploaded from local cache and registered successfully.", "version": version}
        else:
            raise HTTPException(status_code=400, detail="Invalid model source. Must be 'huggingface' or 'local'.")
    except Exception as e:
        logger.error(f"Failed to upload model artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e))