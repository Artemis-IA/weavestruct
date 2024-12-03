# services/model_manager.py
import torch
from enum import Enum
from typing import Dict, Any, Optional, List, Literal, Dict, Any, List, Tuple, Union
from gliner import GLiNER
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification
from huggingface_hub import HfApi
from config import settings
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from loguru import logger
from pathlib import Path
from datetime import datetime
import os
import shutil
import mlflow

class ModelSource(str, Enum):
    huggingface = 'huggingface'
    local = 'local'
class ModelInfoFilter(str, Enum):
    size = 'size'
    recent = 'recent'
    name = 'name'
    task = 'task'
    downloads = 'downloads'
    likes = 'likes'
    emissions_thresholds = 'emissions_thresholds'
    library = 'library'


class ModelManager:
    def __init__(self, s3_service: S3Service, mlflow_service: MLFlowService):
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mlflow_client = mlflow_service.client
        self.model_cache = {}
        self.models_bucket = 'mlflow-artifacts'

        logger.info(f"Using device: {self.device}")

    def fetch_available_models(self) -> List[str]:
        """
        Fetch the list of available models from MLflow registered models.
        """
        models = self.mlflow_client.search_registered_models()
        available_models = [model.name for model in models]
        logger.info(f"Available models fetched from MLflow: {available_models}")
        return available_models

    def fetch_hf_models(self, sort_by: ModelInfoFilter = ModelInfoFilter.size) -> List[Dict[str, Any]]:
        """
        Fetch all GLiNER models from Hugging Face and sort them.

        Args:
            sort_by (ModelInfoFilter): Criteria for sorting the models.

        Returns:
            List[Dict[str, Any]]: List of models with metadata.
        """
        try:
            api = HfApi(token=settings.HF_API_TOKEN)
            models = api.list_models(filter="gliner", cardData=True, full=True)

            model_data = []
            for model in models:
                size = None
                model_info = api.model_info(repo_id=model.modelId, files_metadata=True)
                if model_info.siblings:
                    for sibling in model_info.siblings:
                        if sibling.rfilename == "pytorch_model.bin":
                            size = sibling.size / (1024 * 1024)  # Convert size to MB
                            break
                # Extract description from cardData
                description = ""
                if model_info.card_data and "description" in model_info.card_data:
                    description = model_info.card_data["description"]

                model_data.append({
                    "modelId": model.modelId,
                    "size": size,
                    "lastModified": model.lastModified,
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "description": description,
                })

            # Handle sorting
            if sort_by == ModelInfoFilter.size:
                model_data = sorted(model_data, key=lambda x: x["size"] or float("inf"))
            elif sort_by == ModelInfoFilter.recent:
                model_data = sorted(model_data, key=lambda x: x["lastModified"], reverse=True)
            elif sort_by == ModelInfoFilter.name:
                model_data = sorted(model_data, key=lambda x: x["modelId"])
            elif sort_by == ModelInfoFilter.downloads:
                model_data = sorted(model_data, key=lambda x: x["downloads"], reverse=True)
            elif sort_by == ModelInfoFilter.likes:
                model_data = sorted(model_data, key=lambda x: x["likes"], reverse=True)
            else:
                raise ValueError("Invalid sort criteria. Use 'size', 'recent', 'name', 'downloads', or 'likes'.")

            return model_data
        except Exception as e:
            logger.error(f"Failed to fetch GLiNER models: {e}")
            raise ValueError(f"Error fetching GLiNER models: {e}")
        
    def load_model(self, model_name: str) -> GLiNER:
        """
        Load a model from MLflow artifacts.
        """
        if model_name in self.model_cache:
            logger.info(f"Model {model_name} loaded from cache.")
            return self.model_cache[model_name]

        # Get the latest version of the model
        versions = self.mlflow_service.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")

        version = versions[0]  # Get the latest version
        model_uri = f"models:/{model_name}/{version.version}"

        # Download the model from MLflow
        local_model_path = Path(f"/tmp/{model_name}")
        local_model_path.mkdir(parents=True, exist_ok=True)
        try:
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(local_model_path)
            )
            logger.info(f"Model {model_name} downloaded successfully from MLflow.")
        except Exception as e:
            logger.error(f"Failed to download model {model_name} from MLflow: {e}")
            raise

        # Load the model
        model = GLiNER.from_pretrained(str(local_model_path)).to(self.device)
        self.model_cache[model_name] = model
        logger.info(f"Model {model_name} loaded and cached.")
        return model

    def upload_model_from_huggingface(self, model_name: str, task: Optional[str] = None, **kwargs):
        """Upload a model from Hugging Face to MLflow."""
        try:
            model = GLiNER.from_pretrained(model_name).to(self.device)
            tokenizer = GLiNER.from_pretrained(model_name)

            self.mlflow_service.start_run(run_name=f"Uploading {model_name} from Hugging Face")
            artifact_path = f"{model_name}_artifact"
            version = self.mlflow_service.log_transformers_model(
                transformers_model=model,
                artifact_path=artifact_path,
                model_name=model_name,
                tokenizer=tokenizer,
                **kwargs
            )

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            mlflow.transformers.persist_pretrained_model(model_uri)
            logger.info(f"Persisted pretrained weights for {model_name}.")

            self.mlflow_service.end_run()
            logger.info(f"Model {model_name} uploaded from Hugging Face and registered in MLflow.")
            return version
        except Exception as e:
            logger.error(f"Failed to upload model {model_name} from Hugging Face: {e}")
            self.mlflow_service.end_run(status="FAILED")
            raise

    def upload_model_from_local(self, model_name: str, local_model_path: Path, task: Optional[str] = None, **kwargs):
        """Upload a model from local cache to MLflow."""
        try:
            model = pipeline(task=task, model=str(local_model_path), device=self.device, **kwargs)
            self.mlflow_service.start_run(run_name=f"Uploading {model_name} from local cache")
            artifact_path = f"{model_name}_artifact"
            version = self.mlflow_service.log_transformers_model(
                transformers_model=model,
                artifact_path=artifact_path,
                model_name=model_name,
                task=task,
                **kwargs
            )

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            mlflow.transformers.persist_pretrained_model(model_uri)
            logger.info(f"Persisted pretrained weights for local model {model_name}.")

            self.mlflow_service.end_run()
            logger.info(f"Model {model_name} uploaded from local cache and registered in MLflow.")
            return version
        except Exception as e:
            logger.error(f"Failed to upload model {model_name} from local cache: {e}")
            self.mlflow_service.end_run(status="FAILED")
            raise

    def register_models_at_startup(self, model_names: list):
        """
        Register and log models at startup if they are not already registered.
        """
        for model_name in model_names:
            existing_models = self.mlflow_client.search_registered_models(filter_string=f"name='{model_name}'")
            if not existing_models:
                # Load the model from HuggingFace or another source
                model = GLiNER.from_pretrained(model_name).to(self.device)
                model_save_path = Path(f"models/{model_name}")
                model.save_pretrained(model_save_path)

                # Start MLflow run
                self.mlflow_service.start_run(run_name=f"Registering {model_name}")

                # Log the model as an artifact
                mlflow.log_artifacts(str(model_save_path), artifact_path="model_artifacts")

                # Register the model
                model_uri = f"{mlflow.get_artifact_uri()}/model_artifacts"
                result = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )
                logger.info(f"Model {model_name} registered with version {result.version}.")

                # End MLflow run
                self.mlflow_service.end_run()

                logger.info(f"Model {model_name} registered and logged to MLflow.")
            else:
                logger.info(f"Model {model_name} is already registered in MLflow.")

    def log_model_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        try:
            self.mlflow_service.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def upload_model(self, model_name: str, model_dir: Path):
        """
        Upload a model directory to MLFLow as an artifact.
        """
        try:
            model_uri = str(model_dir.resolve())
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            logger.info(f"Model {model_name} registered successfully with version {result.version}.")
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
        
    def zip_and_upload_model(self, model_name: str):
        model_path = Path("models") / model_name
        zip_path = model_path.with_suffix(".zip")

        if not model_path.exists():
            raise ValueError(f"Model directory {model_name} does not exist.")

        # Create zip file of the model directory
        try:
            shutil.make_archive(str(model_path), 'zip', str(model_path))
        except Exception as e:
            logger.error(f"Failed to create zip archive for {model_name}: {e}")
            return None

        # Upload to S3 bucket
        s3_url = self.s3_service.upload_file(zip_path, bucket_name=self.models_bucket)
        if s3_url:
            logger.info(f"Model {model_name} uploaded successfully to {s3_url}")
            return s3_url
        else:
            logger.error(f"Failed to upload model {model_name} to S3")
            return None
