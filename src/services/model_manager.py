# services/model_manager.py
import torch
from fastapi import HTTPException
from typing import Dict, Any, Optional, List, Literal, Dict, Any, List, Tuple, Union
from gliner import GLiNER
from transformers import AutoTokenizer, pipeline
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from src.config import settings
from src.enums.loopml import ModelSource, ModelInfoFilter
from src.services.mlflow_service import MLFlowService
from mlflow.pyfunc import load_model as mlflow_load_model
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


from loguru import logger
from pathlib import Path
from datetime import datetime
import os
import shutil
import mlflow

class ModelManager:
    def __init__(self, mlflow_service: MLFlowService, tracking_uri: str):
        self.mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        self.mlflow_service = mlflow_service
        self.device = settings.DEVICE
        self.hfapi = HfApi(token=settings.HF_API_TOKEN)
        self.models_bucket = settings.MLFLOW_ARTIFACT_ROOT
        self.model_cache = {}
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, artifact_name: str, alias: Optional[str] = None):

        if not alias:
            alias = "latest" 
        try:
            if alias:
                model_uri = f"models:/{artifact_name}@{alias}"
            else:
                raise ValueError("An alias must be specified to load the model.")
            
            model = mlflow_load_model(model_uri)
            return model
        except MlflowException as e:
            raise RuntimeError(
                f"Failed to load model '{artifact_name}' with alias '{alias}': {e}"
            )

    def promote_model(self, artifact_name: str, version: int, alias: str):

        try:
            self.mlflow_client.set_registered_model_alias(
                name=artifact_name, alias=alias, version=version
            )
        except MlflowException as e:
            raise RuntimeError(
                f"Failed to promote model '{artifact_name}' version '{version}' with alias '{alias}': {e}"
            )
        
    def fetch_available_models(self) -> List[str]:

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
        artifact_name: Optional[str] = None,
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
                "name": "modelId", 
                "size": "modelSize",
                "recent": "lastModified",
                "downloads": "downloads",
                "likes": "likes",
            }
            sort_key = valid_sort_keys.get(sort_by, "modelId")

            models = self.hfapi.list_models(
                filter=filter,
                author=author,
                gated=gated,
                inference=inference,
                library=library,
                language=language,
                model_name=artifact_name,
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

    def load_or_register_model(self, artifact_name: str, model_dir: Path, alias: Optional[str] = "latest"):

        try:
            logger.info(f"Attempting to load model '{artifact_name}' with alias '{alias}'.")
            return self.mlflow_service.load_model(artifact_name, alias)
        except RuntimeError:
            logger.warning(f"Model '{artifact_name}' not found. Attempting to register...")
            self.mlflow_service.register_model(artifact_name, model_dir)
            return self.mlflow_service.load_model(artifact_name, alias)
        
    def fetch_and_register_hf_model(self, artifact_name: str, artifact_path: str, register_name: str):

        try:
            logger.info(f"Fetching model '{artifact_name}' from Hugging Face...")
            model_dir = Path(f"/tmp/{artifact_name.replace('/', '_')}")
            model_dir.mkdir(parents=True, exist_ok=True)

            # List and download relevant files
            files = list_repo_files(repo_id=artifact_name)
            logger.info(f"Files available: {files}")

            # Define required file patterns
            file_patterns = [".bin", ".safetensors", "config.json", "tokenizer.json"]

            downloaded_files = []
            for file in files:
                if any(pattern in file for pattern in file_patterns):
                    logger.info(f"Downloading {file}...")
                    downloaded_path = hf_hub_download(repo_id=artifact_name, filename=file, local_dir=str(model_dir))
                    downloaded_files.append(downloaded_path)
                else:
                    logger.debug(f"Skipping irrelevant file: {file}")

            if not downloaded_files:
                raise ValueError(f"No required model files found for '{artifact_name}'.")

            logger.info("Files downloaded successfully. Logging to MLflow...")
            self.mlflow_service.log_artifacts_and_register_model(
                model_dir=str(model_dir),
                artifact_path=artifact_path,
                register_name=register_name,
            )
        except Exception as e:
            logger.error(f"Failed to fetch or register model '{artifact_name}': {e}")
            raise ValueError(f"Error during model processing: {e}")

    def upload_model_from_local(self, artifact_name: str, local_model_path: Path, task: Optional[str] = None, **kwargs):
        """Upload a model from local cache to MLflow."""
        try:
            model = pipeline(task=task, model=str(local_model_path), device=self.device, **kwargs)
            self.mlflow_service.start_run(run_name=f"Uploading {artifact_name} from local cache")
            artifact_path = f"{artifact_name}_artifact"
            version = self.mlflow_service.log_model(
                transformers_model=model,
                artifact_path=artifact_path,
                artifact_name=artifact_name,
                task=task,
                **kwargs
            )

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            mlflow.transformers.persist_pretrained_model(model_uri)
            logger.info(f"Persisted pretrained weights for local model {artifact_name}.")

            self.mlflow_service.end_run()
            logger.info(f"Model {artifact_name} uploaded from local cache and registered in MLflow.")
            return version
        except Exception as e:
            logger.error(f"Failed to upload model {artifact_name} from local cache: {e}")
            self.mlflow_service.end_run(status="FAILED")
            raise

    def register_models_at_startup(self, model_names: list):


        for artifact_name in model_names:
            existing_models = self.mlflow_client.search_registered_models(filter_string=f"name='{artifact_name}'")
            if not existing_models:
                # Load the model from HuggingFace or another source
                model = GLiNER.from_pretrained(artifact_name).to(self.device)
                model_save_path = Path(f"models/{artifact_name}")
                model.save_pretrained(model_save_path)

                # Start MLflow run
                self.mlflow_service.start_run(run_name=f"Registering {artifact_name}")

                # Log the model as an artifact
                mlflow.log_artifacts(str(model_save_path), artifact_path="model_artifacts")

                # Register the model
                model_uri = f"{mlflow.get_artifact_uri()}/model_artifacts"
                result = mlflow.register_model(
                    model_uri=model_uri,
                    name=artifact_name
                )
                logger.info(f"Model {artifact_name} registered with version {result.version}.")

                # End MLflow run
                self.mlflow_service.end_run()

                logger.info(f"Model {artifact_name} registered and logged to MLflow.")
            else:
                logger.info(f"Model {artifact_name} is already registered in MLflow.")

    def log_model_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        try:
            self.mlflow_service.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def upload_model(self, artifact_name: str, model_dir: Path):
        """
        Upload a model directory to MLFLow as an artifact.
        """
        try:
            model_uri = str(model_dir.resolve())
            result = mlflow.register_model(
                model_uri=model_uri,
                name=artifact_name
            )
            logger.info(f"Model {artifact_name} registered successfully with version {result.version}.")
        except Exception as e:
            logger.error(f"Failed to register model {artifact_name}: {e}")
        
    def zip_and_upload_model(self, artifact_name: str):
        model_path = Path("models") / artifact_name
        zip_path = model_path.with_suffix(".zip")

        if not model_path.exists():
            raise ValueError(f"Model directory {artifact_name} does not exist.")

        # Create zip file of the model directory
        try:
            shutil.make_archive(str(model_path), 'zip', str(model_path))
        except Exception as e:
            logger.error(f"Failed to create zip archive for {artifact_name}: {e}")
            return None

        # Upload to S3 bucket
        s3_url = self.s3_service.upload_file(zip_path, bucket_name=self.models_bucket)
        if s3_url:
            logger.info(f"Model {artifact_name} uploaded successfully to {s3_url}")
            return s3_url
        else:
            logger.error(f"Failed to upload model {artifact_name} to S3")
            return None