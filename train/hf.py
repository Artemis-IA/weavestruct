import os
import mlflow
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger
from pathlib import Path

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5002"
MLFLOW_ARTIFACT_ROOT = "s3://mlflow/"
HUGGINGFACE_MODEL_NAME = "gretelai/gretel-gliner-bi-small-v1.0"
ARTIFACT_PATH = "model_artifacts"
MODEL_NAME = "HuggingFace_Gliner_Model"

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    logger.info("MLflow and S3 credentials configured.")

def fetch_and_store_model():
    try:
        logger.info(f"Fetching model '{HUGGINGFACE_MODEL_NAME}' from Hugging Face...")

        # Créer un répertoire temporaire pour le modèle
        model_dir = Path(f"/tmp/{HUGGINGFACE_MODEL_NAME.replace('/', '_')}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Lister les fichiers disponibles dans le dépôt
        logger.info("Listing available files in the Hugging Face repository...")
        files = list_repo_files(repo_id=HUGGINGFACE_MODEL_NAME)
        logger.info(f"Files available in repository: {files}")

        # Télécharger uniquement les fichiers nécessaires
        required_files = ["pytorch_model.bin", "config.json"]
        for file in required_files:
            if file in files:
                logger.info(f"Downloading {file}...")
                hf_hub_download(repo_id=HUGGINGFACE_MODEL_NAME, filename=file, local_dir=str(model_dir))
            else:
                logger.warning(f"{file} not found in repository. Skipping.")

        logger.info("Model fetched successfully. Starting MLflow run...")
        with mlflow.start_run(run_name=f"Upload {HUGGINGFACE_MODEL_NAME}"):
            logger.info("Logging model artifacts to MLflow...")
            mlflow.log_artifacts(str(model_dir), artifact_path=ARTIFACT_PATH)

            logger.info("Registering model in MLflow...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{ARTIFACT_PATH}"
            mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

            logger.success(f"Model '{MODEL_NAME}' uploaded and registered successfully.")
    except Exception as e:
        logger.error(f"Failed to fetch and store model: {e}")
        raise

if __name__ == "__main__":
    setup_mlflow()
    fetch_and_store_model()
