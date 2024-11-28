import torch
from typing import Dict, Any
from gliner import GLiNER
from transformers import AutoTokenizer
from services.s3_service import S3Service
from codecarbon import EmissionsTracker
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

AVAILABLE_MODELS = [
    "knowledgator/gliner-multitask-large-v0.5",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large_bio-v0.1",
    "numind/NuNER_Zero",
    "EmergentMethods/gliner_medium_news-v2.1",
]

class ModelManager:
    def __init__(self, s3_service: S3Service):
        self.s3_service = s3_service
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracker_active = False
        self.emissions_tracker = None
        self.mlflow_client = MlflowClient()
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_name: str):
        model_path = Path("models") / model_name
        if model_path.exists():
            model = GLiNER.from_pretrained(str(model_path)).to(self.device)
            return model
        elif model_name in AVAILABLE_MODELS:
            model = GLiNER.from_pretrained(model_name).to(self.device)
            model.save_pretrained(model_path)
            return model
        else:
            raise ValueError(f"Model {model_name} not found.")

    def log_model_metrics(self, metrics: Dict[str, Any]):
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def process_model(self, model_name: str, inputs: Dict[str, Any]):
        # Ensure any previous MLflow run is ended before starting a new one
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=f"Processing {model_name}"):
            # Initialize CodeCarbon tracker if none is active
            if not self.tracker_active:
                try:
                    self.emissions_tracker = EmissionsTracker(project_name="model_processing")
                    self.emissions_tracker.start()
                    self.tracker_active = True
                except Exception as e:
                    logger.warning(f"Unable to start CodeCarbon: {e}")
                    self.emissions_tracker = None

            # Load the model
            model = self.load_model(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs_tokenized = tokenizer(inputs["text"], return_tensors="pt").to(self.device)

            # Run inference
            output = model(**inputs_tokenized)

            # Capture emissions if CodeCarbon tracker is active
            emissions = None
            if self.emissions_tracker and self.tracker_active:
                try:
                    emissions = self.emissions_tracker.stop()
                except Exception as e:
                    logger.warning(f"Error stopping CodeCarbon tracker: {e}")
                finally:
                    self.tracker_active = False  # Reset for next use

            # Log hardware resource usage
            metrics = {
                "gpu_memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cpu_usage": torch.get_num_threads(),
            }
            if emissions is not None:
                metrics["carbon_emissions"] = emissions

            self.log_model_metrics(metrics)

            return output

    def zip_and_upload_model(self, model_name: str):
        model_path = Path("models") / model_name
        zip_path = model_path.with_suffix(".zip")

        if not model_path.exists():
            raise ValueError(f"Model directory {model_name} does not exist.")

        # Create zip file of the model directory
        try:
            import shutil
            shutil.make_archive(str(model_path), 'zip', str(model_path))
        except Exception as e:
            logger.error(f"Failed to create zip archive for {model_name}: {e}")
            return None

        # Upload to S3 bucket
        s3_url = self.s3_service.upload_file(zip_path, bucket_name=self.s3_service.output_bucket)
        if s3_url:
            logger.info(f"Model {model_name} uploaded successfully to {s3_url}")
            return s3_url
        else:
            logger.error(f"Failed to upload model {model_name} to S3")
            return None
