# services/mlflow_service.py
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

class MLFlowService:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.tracking_uri = tracking_uri
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    def start_run(self, run_name: str):
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run_name}")

    def end_run(self, status: str = None):
        mlflow.end_run(status=status)
        logger.info(f"MLflow run ended with status: {status}")

    def log_params(self, params: Dict[str, Any]):
        try:
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params}")
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged metrics: {metrics} at step {step}")
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, file_path: str, artifact_path: str = None):
        try:
            mlflow.log_artifact(file_path, artifact_path)
            logger.info(f"Logged artifact: {file_path} at {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact to MLflow: {e}")

    def log_transformers_model(
        self,
        transformers_model,
        artifact_path: str,
        model_name: str,
        processor=None,
        task: Optional[str] = None,
        **kwargs
    ):
        try:
            # Log the model using mlflow.transformers
            mlflow.transformers.log_model(
                transformers_model=transformers_model,
                artifact_path=artifact_path,
                processor=processor,
                task=task,
                **kwargs
            )
            logger.info(f"Transformers model logged to MLflow at {artifact_path}.")

            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            version = self.register_model(model_name, model_uri)
            return version
        except Exception as e:
            logger.error(f"Failed to log and register transformers model {model_name}: {e}")
            raise

    def register_model(self, model_name: str, model_dir: Path):
        try:
            model_uri = str(model_dir.resolve())
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            logger.info(f"Model {model_name} registered successfully with version {result.version}.")
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")

    def get_registered_model(self, model_name: str):
        try:
            model = self.client.get_registered_model(model_name)
            logger.info(f"Retrieved registered model: {model.name}")
            return model
        except Exception as e:
            logger.error(f"Failed to get registered model {model_name}: {e}")
            return None

    def get_latest_versions(self, model_name: str, stages: Optional[List[str]] = None):
        try:
            versions = self.client.get_latest_versions(name=model_name, stages=stages)
            logger.info(f"Retrieved latest versions for model {model_name}: {[v.version for v in versions]}")
            return versions
        except Exception as e:
            logger.error(f"Failed to get latest versions for model {model_name}: {e}")
            return None

    def download_model(self, model_name: str, version: str, download_dir: str):
        try:
            model_uri = f"models:/{model_name}/{version}"
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_dir)
            logger.info(f"Model {model_name} version {version} downloaded successfully to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download model {model_name} version {version}: {e}")
            return None

    def search_registered_models(self, filter_string: str = '', max_results: int = None, order_by: List[str] = None, page_token: str = None):
        try:
            models = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token
            )
            logger.info(f"Retrieved registered models: {[model.name for model in models]}")
            return models
        except Exception as e:
            logger.error(f"Failed to search registered models: {e}")
            return []

    def set_tracking_uri(self, uri: str):
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI updated to: {uri}")

    def validate_connection(self):
        """Validate connection to MLflow tracking server."""
        try:
            # Use search_experiments as a connectivity check
            experiments = self.client.search_experiments(max_results=1)
            logger.info(
                f"MLflow service is accessible. Found {len(experiments)} experiment(s)."
            )
        except Exception as e:
            raise Exception(f"Failed to connect to MLflow: {e}")
