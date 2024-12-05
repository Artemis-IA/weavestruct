# services/model_manager.py
import torch
from enum import Enum
from typing import Dict, Any, Optional, List, Literal, Dict, Any, List, Tuple, Union
from gliner import GLiNER
from transformers import AutoTokenizer, pipeline
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from config import settings
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

class ModelManager:
    def __init__(self, mlflow_service: MLFlowService):
        self.mlflow_service = mlflow_service
        self.device = settings.DEVICE
        self.hfapi = HfApi(token=settings.HF_API_TOKEN)
        logger.info(f"Using device: {self.device}")

    def fetch_available_models(self) -> List[str]:
        """
        Fetch the list of available models from MLflow registered models.
        """
        models = self.mlflow_service.search_registered_models()
        available_models = [model.name for model in models]
        logger.info(f"Available models fetched from MLflow: {available_models}")
        return available_models
    
    def fetch_hf_models(
        self,
        sort_by: Optional[str] = "name",
        filter: Optional[Union[str, List[str]]] = None,
        author: Optional[str] = None,
        gated: Optional[bool] = None,
        inference: Optional[Literal["cold", "frozen", "warm"]] = None,
        library: Optional[Union[str, List[str]]] = None,
        language: Optional[Union[str, List[str]]] = None,
        model_name: Optional[str] = None,
        task: Optional[Union[str, List[str]]] = None,
        trained_dataset: Optional[Union[str, List[str]]] = None,
        tags: Optional[Union[str, List[str]]] = None,
        search: Optional[str] = None,
        pipeline_tag: Optional[str] = None,
        emissions_thresholds: Optional[Tuple[float, float]] = None,
        limit: Optional[int] = 100,
    ) -> List[Dict[str, Any]]:
        try:
            valid_sort_keys = {
                "size": "modelSize",
                "recent": "lastModified",
                "name": "modelId",
                "downloads": "downloads",
                "likes": "likes",
            }
            sort_key = valid_sort_keys.get(sort_by, "modelId")  # Default to "name" if invalid

            models = self.hfapi.list_models(
                filter=filter,
                author=author,
                gated=gated,
                inference=inference,
                library=library,
                language=language,
                model_name=model_name,
                task=task,
                trained_dataset=trained_dataset,
                tags=tags,
                search=search,
                pipeline_tag=pipeline_tag,
                emissions_thresholds=emissions_thresholds,
                sort=sort_key,
                limit=limit,
                cardData=True,
                full=True,
            )

            model_data = []
            for model in models:
                size = None
                description = ""
                model_info = self.hfapi.model_info(repo_id=model.modelId, files_metadata=True)
                if model_info.siblings:
                    for sibling in model_info.siblings:
                        if sibling.rfilename == "pytorch_model.bin":
                            size = sibling.size / (1024 * 1024)
                            break

                if isinstance(model_info.card_data, dict):
                    description = model_info.card_data.get("description", "")

                model_data.append({
                    "modelId": model.modelId,
                    "size": size,
                    "lastModified": model.lastModified,
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "description": description.strip(),
                })

            # Return sorted results
            return sorted(
                model_data,
                key=lambda x: x.get(valid_sort_keys.get(sort_by, "modelId")),
                reverse=(sort_by != "name"),
            )
        except Exception as e:
            logger.error(f"Failed to fetch models from Hugging Face: {e}")
            raise ValueError(f"Error fetching models: {e}")

            
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

    def fetch_and_register_hf_model(self, model_name: str, artifact_path: str, register_name: str):
        """Download a model from Hugging Face and register it in MLflow."""
        try:
            logger.info(f"Fetching model '{model_name}' from Hugging Face...")
            model_dir = Path(f"/tmp/{model_name.replace('/', '_')}")
            model_dir.mkdir(parents=True, exist_ok=True)

            # List and download relevant files
            files = list_repo_files(repo_id=model_name)
            logger.info(f"Files available: {files}")

            # Define required file patterns
            file_patterns = [".bin", ".safetensors", "config.json", "tokenizer.json"]

            downloaded_files = []
            for file in files:
                if any(pattern in file for pattern in file_patterns):
                    logger.info(f"Downloading {file}...")
                    downloaded_path = hf_hub_download(repo_id=model_name, filename=file, local_dir=str(model_dir))
                    downloaded_files.append(downloaded_path)
                else:
                    logger.debug(f"Skipping irrelevant file: {file}")

            if not downloaded_files:
                raise ValueError(f"No required model files found for '{model_name}'.")

            logger.info("Files downloaded successfully. Logging to MLflow...")
            self.mlflow_service.log_artifacts_and_register_model(
                model_dir=str(model_dir),
                artifact_path=artifact_path,
                register_name=register_name,
            )
        except Exception as e:
            logger.error(f"Failed to fetch or register model '{model_name}': {e}")
            raise ValueError(f"Error during model processing: {e}")

    def upload_model_from_local(self, model_name: str, local_model_path: Path, task: Optional[str] = None, **kwargs):
        """Upload a model from local cache to MLflow."""
        try:
            model = pipeline(task=task, model=str(local_model_path), device=self.device, **kwargs)
            self.mlflow_service.start_run(run_name=f"Uploading {model_name} from local cache")
            artifact_path = f"{model_name}_artifact"
            version = self.mlflow_service.log_model(
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