from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path
from src.dependencies import get_db, get_s3_service, get_mlflow_service, get_model_manager
from src.services.train_service import TrainService
from src.services.model_manager import ModelManager
from src.services.s3_service import S3Service
from src.services.mlflow_service import MLFlowService
from src.schemas.train import TrainInput, TrainResponse
from huggingface_hub import hf_hub_download, HfApi
from src.config import settings
import aiofiles
import json

router = APIRouter()

@router.post("/train", response_model=TrainResponse, summary="Train a model")
async def train_model(
    dataset_file: Optional[UploadFile] = File(
        None, description="Dataset file in JSON format. Required if `s3_url` or `huggingface_dataset` is not provided."
    ),
    s3_url: Optional[str] = Form(
        None, description="S3 URL pointing to the dataset file. Required if `dataset_file` or `huggingface_dataset` is not provided."
    ),
    huggingface_dataset: Optional[str] = Form(
        None, description="Name of a HuggingFace dataset. Required if `dataset_file` or `s3_url` is not provided."
    ),
    artifact_name: str = Form(
        "bert-base-uncased", description="Name of the pre-trained base model from Hugging Face."
    ),
    custom_model_name: Optional[str] = Form(
        None, description="Optional custom name for the fine-tuned model."
    ),
    split_ratio: float = Form(
        0.9, ge=0.0, le=1.0, description="Train/Eval split ratio (default is 0.9)."
    ),
    learning_rate: float = Form(
        1e-5, gt=0, description="Learning rate for the training process."
    ),
    weight_decay: float = Form(
        0.01, ge=0.0, description="Weight decay for optimizer."
    ),
    batch_size: int = Form(
        16, ge=1, description="Batch size for training."
    ),
    epochs: int = Form(
        3, ge=1, description="Number of training epochs."
    ),
    compile_model: bool = Form(
        False, description="Whether to compile the model for faster training."
    ),
    # training_params_file: Optional[UploadFile] = File(
    #     None, description="Optional JSON file with additional training parameters."
    # ),
    db: Session = Depends(get_db),
    s3_service: S3Service = Depends(get_s3_service),
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
    model_manager: ModelManager = Depends(get_model_manager),
):
    # Validation: ensure exactly one dataset source is provided
    sources = [dataset_file, s3_url, huggingface_dataset]
    if sum(bool(source) for source in sources) != 1:
        raise HTTPException(
            status_code=400,
            detail="You must provide exactly one dataset source: `dataset_file`, `s3_url`, or `huggingface_dataset`."
        )

    # Handle dataset from different sources
    dataset_path = None
    if dataset_file:
        dataset_path = f"/tmp/{dataset_file.filename}"
        async with aiofiles.open(dataset_path, "wb") as out_file:
            content = await dataset_file.read()
            await out_file.write(content)
    elif s3_url:
        s3_info = s3_service.parse_s3_url(s3_url)
        if not s3_info:
            raise HTTPException(status_code=400, detail="Invalid S3 URL provided.")
        bucket_name, object_key = s3_info
        dataset_path = f"/tmp/{Path(object_key).name}"
        if not s3_service.download_file(bucket_name, object_key, Path(dataset_path)):
            raise HTTPException(
                status_code=500,
                detail="Failed to download the dataset from the provided S3 URL."
            )
    elif huggingface_dataset:
        try:
            api = HfApi(token=settings.HF_API_TOKEN)
            files = api.list_repo_files(repo_id=huggingface_dataset)
            json_files = [f for f in files if f.endswith(".json")]
            if not json_files:
                raise HTTPException(
                    status_code=400,
                    detail="No JSON files found in the specified HuggingFace dataset."
                )
            target_file = json_files[0]
            local_json_path = hf_hub_download(
                repo_id=huggingface_dataset,
                filename=target_file,
                token=settings.HF_API_TOKEN
            )
            dataset_path = f"/tmp/{huggingface_dataset.replace('/', '_')}.json"
            with open(dataset_path, "w") as f:
                json.dump([dict(x) for x in local_json_path], f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process HuggingFace dataset: {e}"
            )

    # Handle training parameters file
    training_params = {}
    # if training_params_file:
    #     try:
    #         content = await training_params_file.read()
    #         training_params = json.loads(content)
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=400,
    #             detail=f"Invalid training_params_file: {e}"
    #         )

    # Default training parameters
    training_params.setdefault("learning_rate", learning_rate)
    training_params.setdefault("weight_decay", weight_decay)
    training_params.setdefault("batch_size", batch_size)
    training_params.setdefault("epochs", epochs)
    training_params.setdefault("compile_model", compile_model)

    # Model registration name
    register_name = custom_model_name or artifact_name

    # Fetch and register model
    model_manager.fetch_and_register_hf_model(
        artifact_name=artifact_name,
        artifact_path="hf_model",
        register_name=register_name,
    )

    # Prepare training input
    train_input = TrainInput(
        artifact_name=artifact_name,
        custom_model_name=register_name,
        train_data=dataset_path,
        split_ratio=split_ratio,
        **training_params,
    )

    # Train the model
    train_service = TrainService(
        db=db,
        s3_service=s3_service,
        mlflow_service=mlflow_service,
        model_manager=model_manager,
    )
    response = train_service.train_model(train_input)

    return response
