
# utils/logging.py
import os
from loguru import logger
from huggingface_hub import HfApi
import mlflow
from mlflow.tracking import MlflowClient
from codecarbon import EmissionsTracker
from typing import Optional
import time
from src.config import settings

class ModelLoggerService:
    def __init__(self):
        self.hf_api = HfApi()  # Initialize the Hugging Face API client
        self.huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub/")
        self.client = MlflowClient()
        self.emissions_tracker = None

        # Initialize the MLflow tracking URI
        db_url = os.getenv("DATABASE_URL", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(db_url)

        # Initialize static models and CodeCarbon tracker
        self.static_models = {
            "Ollama Embedding Model": ("sentence-transformers/all-MiniLM-L6-v2", os.path.join(self.huggingface_cache, "models--sentence-transformers--all-MiniLM-L6-v2")),
            "GLiNER Extractor Model": ("E3-JSI/gliner-multi-pii-domains-v1", os.path.join(self.huggingface_cache, "models--E3-JSI--gliner-multi-pii-domains-v1")),
            "Gliner Transformer Model": ("knowledgator/gliner-multitask-large-v0.5", os.path.join(self.huggingface_cache, "models--knowledgator--gliner-multitask-large-v0.5")),
            "Tokenizer Model": ("microsoft/deberta-v3-large", os.path.join(self.huggingface_cache, "models--microsoft--deberta-v3-large")),
            "Docling Models": ("ds4sd/docling-models", os.path.join(self.huggingface_cache, "models--ds4sd--docling-models"))
        }
        self.initialize_emissions_tracker()

    def initialize_emissions_tracker(self):
        """
        Initialize CodeCarbon tracker with lock file cleanup.
        """
        lock_file = "/tmp/.codecarbon.lock"
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info("CodeCarbon lock file removed.")
            except Exception as e:
                logger.warning(f"Unable to remove CodeCarbon lock file: {e}")

        self.emissions_tracker = EmissionsTracker(project_name="model_logging", save_to_file=False, save_to_prometheus=True, prometheus_url=f"localhost:${settings.PROMETHEUS_PORT_CARBON}")
        logger.info("CodeCarbon tracker initialized.")

    def log_model_details(self):
        logger.info("Starting model logging process...")
        mlflow.end_run()  # Ensure no active runs are in progress

        try:
            with mlflow.start_run(run_name="Model Logging") as run:
                run_id = run.info.run_id

                for artifact_name, (model_id, model_file_path) in self.static_models.items():
                    logger.info(f"Processing model: {artifact_name}")
                    self._log_model_metadata(artifact_name, model_id, model_file_path, run_id)

            logger.info("Model logging process completed.")
            return {"message": "Model logging completed successfully"}
        except Exception as e:
            logger.error(f"Error in logging model details: {e}")
            return {"error": str(e)}

    def _log_model_metadata(self, artifact_name, model_id, model_file_path, run_id):
        try:
            # Check if the model is registered in MLflow
            registered_models = [rm.name for rm in self.client.search_registered_models()]
            if artifact_name not in registered_models:
                self.client.create_registered_model(artifact_name)

            # Fetch model metadata from Hugging Face
            model_info = self.hf_api.model_info(model_id)
            model_version = model_info.sha  # Unique identifier for version
            model_description = self._fetch_readme(model_id) or "No description available."  # Use README.md content as description
            model_tags = model_info.tags

            # Log metadata to MLflow
            mlflow.set_tag(f"{artifact_name}_description", model_description)
            for tag in model_tags:
                mlflow.set_tag(f"{artifact_name}_tag_{tag}", True)
            mlflow.log_param(f"{artifact_name}_version", model_version)

            # Log model file as artifact if it exists
            if os.path.exists(model_file_path):
                artifact_path = f"artifacts/{artifact_name}"
                mlflow.log_artifact(model_file_path, artifact_path=artifact_path)
                self.client.create_model_version(
                    name=artifact_name,
                    source=f"{mlflow.get_artifact_uri()}/{artifact_path}",
                    run_id=run_id
                )
            else:
                logger.warning(f"Model path not found: {model_file_path}")
        except Exception as e:
            logger.error(f"Error logging metadata for model {artifact_name}: {e}")

    def _fetch_readme(self, model_id: str) -> Optional[str]:
        """
        Fetch the README.md content of a Hugging Face model to use as a description.
        """
        try:
            readme_content = self.hf_api.model_info(model_id).cardData.get("model_card", "")
            return readme_content
        except Exception as e:
            logger.warning(f"Unable to fetch README.md for model {model_id}: {e}")
            return None

    def log_query(self, query: str):
        try:
            with mlflow.start_run(run_name="Query Logging") as run:
                mlflow.log_param("query", query)
                mlflow.log_param("timestamp", time.time())
                logger.info("Query logged successfully.")
                return {"message": "Query logged successfully"}
        except Exception as e:
            logger.error(f"Error logging query: {e}")
            return {"error": str(e)}
