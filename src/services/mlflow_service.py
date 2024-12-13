# services/mlflow_service.py
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

class MLFlowService:
    def __init__(self, tracking_uri: str, s3_endpoint: str, access_key: str, secret_key: str):
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key

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

    def log_model_artifact(self, model: Any, artifact_name: str, artifact_path: str, **kwargs) -> str:

        try:
            from gliner.model import GLiNER
            artifact_dir = Path(artifact_path)
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Handle different model types
            if isinstance(model, GLiNER):
                logger.info("Detected GLiNER model. Saving using `save_pretrained`.")
                model.save_pretrained(str(artifact_dir))
            elif hasattr(model, "save_pretrained"):
                logger.info("Detected Hugging Face model. Saving using `save_pretrained`.")
                model.save_pretrained(str(artifact_dir))
            else:
                logger.info("Generic model detected. Saving as binary artifact.")
                model_path = artifact_dir / f"{artifact_name}.bin"
                with open(model_path, "wb") as f:
                    f.write(model)

            # Log the model artifacts to MLflow
            mlflow.log_artifacts(str(artifact_dir), artifact_path="model")

            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            registered_model = self.client.create_registered_model(name=artifact_name)
            self.client.create_model_version(
                name=artifact_name, 
                source=model_uri, 
                run_id=mlflow.active_run().info.run_id
            )
            logger.info(f"Model {artifact_name} logged and registered successfully.")
            return model_uri

        except Exception as e:
            logger.error(f"Failed to log and register model {artifact_name}: {e}")
            raise ValueError(f"Error logging model artifact for {artifact_name}: {e}")

    def load_model(self, artifact_name: str, alias: str = "latest") -> Any:
        """
        Charge un modèle MLflow en utilisant un alias spécifique ou une version donnée.
        """
        try:
            model_uri = f"models:/{artifact_name}/{alias}"
            logger.info(f"Loading model: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            return model
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to load model {artifact_name} with alias {alias}: {e}")
            raise RuntimeError(f"Error loading model '{artifact_name}': {e}")
        
    def register_model(self, artifact_name: str, model_dir: Path, artifact_path: str = "model"):
        """
        Enregistre un modèle dans MLflow en créant une version du modèle.
        """
        try:
            with mlflow.start_run(run_name=f"Register {artifact_name}"):
                mlflow.log_artifacts(str(model_dir), artifact_path=artifact_path)
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                self.client.create_registered_model(artifact_name)
                self.client.create_model_version(name=artifact_name, source=model_uri, run_id=mlflow.active_run().info.run_id)
                logger.info(f"Model {artifact_name} registered successfully.")
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to register model {artifact_name}: {e}")
            raise RuntimeError(f"Error during model registration: {e}")

    def get_registered_model(self, artifact_name: str):
        """
        Recherche un modèle enregistré par son nom.
        """
        try:
            models = self.client.search_registered_models(filter_string=f"name='{artifact_name}'")
            if not models:
                logger.warning(f"No registered models found with name '{artifact_name}'.")
                return None
            return models[0]
        except Exception as e:
            logger.error(f"Failed to search for registered model '{artifact_name}': {e}")
            return None
            
    def log_artifacts_and_register_model(self, model_dir: str, artifact_path: str, register_name: str):
        """Logs artifacts and registers the model in MLflow."""
        try:
            with mlflow.start_run(run_name=f"Upload {register_name}"):
                mlflow.log_artifacts(model_dir, artifact_path=artifact_path)
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                mlflow.register_model(model_uri=model_uri, name=register_name)
                logger.info(f"Model '{register_name}' registered successfully.")
        except Exception as e:
            logger.error(f"Failed to log artifacts or register model: {e}")
            raise ValueError(f"Error logging and registering model: {e}")
        
    def get_latest_versions(self, artifact_name: str, stages: Optional[List[str]] = None):
        try:
            versions = self.client.get_latest_versions(name=artifact_name, stages=stages)
            logger.info(f"Retrieved latest versions for model {artifact_name}: {[v.version for v in versions]}")
            return versions
        except Exception as e:
            logger.error(f"Failed to get latest versions for model {artifact_name}: {e}")
            return None

    def download_model(self, artifact_name: str, version: str, download_dir: str):
        try:
            model_uri = f"models:/{artifact_name}/{version}"
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_dir)
            logger.info(f"Model {artifact_name} version {version} downloaded successfully to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download model {artifact_name} version {version}: {e}")
            return None

    def search_registered_models(self) -> List[Dict[str, Any]]:
        try:
            models = self.client.search_registered_models()
            result = []
            for model in models:
                model_info = {
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "run_id": version.run_id,
                        }
                        for version in model.latest_versions
                    ] if model.latest_versions else None,
                }
                result.append(model_info)
            return result
        except Exception as e:
            logger.error(f"Failed to fetch registered models: {e}")
            raise ValueError(f"Error fetching registered models: {e}")

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

def delete_model_version(self, name: str, version: str):
    """
    Delete a specific version of a registered model.
    """
    try:
        self.client.delete_model_version(name=name, version=version)
        logger.info(f"Version '{version}' of model '{name}' deleted successfully.")
    except Exception as e:
        logger.error(f"Failed to delete version '{version}' of model '{name}': {e}")
        raise ValueError(f"Error deleting model version: {e}")

def delete_registered_model(self, name: str):
    """
    Delete an entire registered model along with all its versions.
    """
    try:
        self.client.delete_registered_model(name=name)
        logger.info(f"Model '{name}' deleted successfully from MLflow.")
    except Exception as e:
        logger.error(f"Failed to delete model '{name}': {e}")
        raise ValueError(f"Error deleting registered model: {e}")
