# routers/train.py

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from pathlib import Path
from src.dependencies import get_db, get_s3_service, get_mlflow_service, get_model_manager
from src.config import settings
from src.services.train_service import TrainService
from src.services.model_manager import ModelManager
from src.enums.loopml import ModelSource, ModelInfoFilter
from src.services.s3_service import S3Service
from src.services.mlflow_service import MLFlowService
from datasets import load_dataset, Features, Sequence, Value
from huggingface_hub import hf_hub_download, HfApi
from src.schemas.train import TrainInput, TrainResponse
from loguru import logger
import json
import shutil
import aiofiles
from mlflow.tracking import MlflowClient

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
async def train_model(
    dataset_file: Optional[bytes] = File(None, description="Upload dataset file (JSON list)"),
    s3_url: Optional[str] = Form(None, description="S3 URL of the dataset (e.g., s3://bucket/path/file.json)"),
    huggingface_dataset: Optional[str] = Form(None, description="HuggingFace dataset name"),
    artifact_name: str = Form("bert-base-uncased", description="Name of the base model from Hugging Face"),
    custom_model_name: Optional[str] = Form(None, description="Custom name for the fine-tuned model"),
    split_ratio: float = Form(0.9, description="Train/Eval split ratio"),
    learning_rate: float = Form(1e-5, description="Learning rate"),
    weight_decay: float = Form(0.01, description="Weight decay"),
    batch_size: int = Form(16, description="Batch size"),
    epochs: int = Form(3, description="Number of epochs"),
    compile_model: bool = Form(False, description="Compile the model for faster training"),
    training_params_file: Optional[bytes] = File(None, description="Upload training parameters JSON file"),
    db=Depends(get_db),
    s3_service: S3Service = Depends(get_s3_service),
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
    model_manager: ModelManager = Depends(get_model_manager)
):

    # Validation: exactement une source de dataset
    sources = [dataset_file, s3_url, huggingface_dataset]
    if sum(map(bool, sources)) != 1:
        raise HTTPException(
            status_code=400,
            detail="Veuillez fournir exactement une source de dataset: 'dataset_file', 's3_url' ou 'huggingface_dataset'."
        )

    # Récupération du dataset local
    dataset_path = None
    if dataset_file:
        dataset_path = f"/tmp/{dataset_file.filename}"
        async with aiofiles.open(dataset_path, 'wb') as out_file:
            content = await dataset_file.read()
            await out_file.write(content)
    elif s3_url:
        s3_info = s3_service.parse_s3_url(s3_url)
        if not s3_info:
            raise HTTPException(status_code=400, detail="URL S3 invalide.")
        bucket_name, object_key = s3_info
        dataset_path = f"/tmp/{object_key.split('/')[-1]}"
        if not s3_service.download_file(bucket_name, object_key, Path(dataset_path)):
            raise HTTPException(status_code=500, detail="Échec du téléchargement du dataset depuis S3.")
    elif huggingface_dataset:
        api = HfApi(token=settings.HF_API_TOKEN)
        try:
            files = api.list_repo_files(repo_id=huggingface_dataset)
            json_files = [f for f in files if f.endswith(".json")]
            if not json_files:
                raise HTTPException(status_code=400, detail="Aucun fichier JSON trouvé dans le dataset HuggingFace.")
            target_file = json_files[0]
            local_json_path = hf_hub_download(repo_id=huggingface_dataset, filename=target_file, token=settings.HF_API_TOKEN)
            dataset = load_dataset("json", data_files=local_json_path)
            split_name = "train" if "train" in dataset else list(dataset.keys())[0]
            dataset_path = f"/tmp/{huggingface_dataset.replace('/', '_')}.json"
            with open(dataset_path, "w") as f:
                json.dump([dict(x) for x in dataset[split_name]], f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Échec du traitement du dataset HuggingFace: {e}")

    # Lecture des paramètres d'entraînement depuis un fichier JSON éventuel
    training_params = {}
    if training_params_file and training_params_file.filename:
        content = await training_params_file.read()
        training_params = json.loads(content)

    # Valeurs par défaut si non présentes
    training_params.setdefault("learning_rate", learning_rate)
    training_params.setdefault("weight_decay", weight_decay)
    training_params.setdefault("batch_size", batch_size)
    training_params.setdefault("epochs", epochs)
    training_params.setdefault("compile_model", compile_model)

    # Nom du modèle à enregistrer
    register_name = custom_model_name or artifact_name

    # Téléchargement du modèle Hugging Face et enregistrement dans MLflow Model Registry
    artifact_path = "hf_model"
    model_manager.fetch_and_register_hf_model(
        artifact_name=artifact_name,
        artifact_path=artifact_path,
        register_name=register_name
    )

    # Préparation de l'entrée pour le TrainService
    train_input = TrainInput(
        artifact_name=artifact_name,
        custom_model_name=register_name,
        train_data=dataset_path,
        split_ratio=split_ratio,
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