import os
import json
import aiofiles
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
from fastapi import FastAPI, APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from loguru import logger
from huggingface_hub import hf_hub_download, HfApi, list_repo_files
from datasets import load_dataset
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow import pyfunc
import shutil
import uuid
from datetime import datetime
import subprocess  
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging
from mlflow.pyfunc import load_model


# ===========================================================================
# Configuration / Settings
# ===========================================================================

# Pour cet exemple, on définit quelques paramètres en variables d'environnement
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5002/"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "docgraph"
os.environ["DATABASE_URL"] = "postgresql://docgraph:docgraph@postgres:5432/docgraph"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", None)
os.environ["MLFLOW_ARTIFACT_ROOT"] = "artifacts"
artifact_path = "models-artifacts"


app = FastAPI(
    title="MLflow Model Registry CRUD API",
    description="API FastAPI exposant des opérations CRUD sur le MLflow Model Registry",
    version="1.0.0",
    docs_url="/"
)
Base = declarative_base()

# ===========================================================================
# Services et Dépendances Mock
# ===========================================================================

# Dans un vrai projet, get_db retournerait une session SQLAlchemy, etc.
# Dependency to get the SQLAlchemy session
# utils/database.py


# Load database URL from settings
db_url = os.environ["DATABASE_URL"]

# Create the SQLAlchemy engine
engine = create_engine(db_url, echo=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for the models
Base = declarative_base()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency to get the SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

# Context manager for database sessions
@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

# Create all tables
def init_db():
    from sqlalchemy import text  # To execute raw SQL if needed
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")


# ===========================================================================
# Schemas
# ===========================================================================


class TrainInput(BaseModel):
    model_name: str
    version: str = "1"
    custom_model_name: Optional[str] = None
    train_data: Optional[str] = None
    split_ratio: float = 0.9
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 3
    compile_model: bool = False

class DatasetSource(str, Enum):
    file = "file"
    s3 = "s3"
    huggingface = "huggingface"

class TrainRequest(TrainInput):
    dataset_source: DatasetSource
    dataset_file: Optional[UploadFile] = None
    s3_url: Optional[str] = None
    huggingface_dataset: Optional[str] = None
    training_params_file: Optional[UploadFile] = None

class TrainResponse(BaseModel):
    model_name: str
    model_version: str
    metrics: dict


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, default=lambda: str(uuid.uuid4()), unique=True)
    dataset_id = Column(Integer, nullable=False)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=32)
    status = Column(String, default="Started")
    created_at = Column(DateTime, default=datetime.utcnow)
    s3_url = Column(String, nullable=True)
    
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="Unnamed Dataset")
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class S3Service:
    def parse_s3_url(self, url: str):
        # Suppose qu'une URL s3 est du type s3://bucket/path/to/file
        if not url.startswith("s3://"):
            return None
        parts = url[5:].split("/", 1)
        if len(parts) < 2:
            return None
        bucket, key = parts
        return bucket, key

    def download_file(self, bucket_name: str, object_key: str, target_path: Path) -> bool:
        # Ici on simule un download. Dans un vrai monde, on utiliserait boto3
        # On peut simuler l'existence du fichier local en échouant systématiquement.
        # Pour l'exemple, on simule un succès si un fichier local fictif existe
        # Cela dit, on va simplement dire qu'on échoue, à moins de le trouver localement.
        logger.info(f"Simulating S3 download for s3://{bucket_name}/{object_key} to {target_path}")
        # Just for example, if a local file with the same name exists, we copy it
        local_path = f"./{object_key.split('/')[-1]}"
        if os.path.isfile(local_path):
            shutil.copy(local_path, target_path)
            return True
        return False

def get_s3_service():
    return S3Service()

class MLFlowService:
    def __init__(self):
        self.client = MlflowClient()

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

    def log_artifacts_and_register_model(self, model_dir: str, artifact_path: str, register_name: str):
        with mlflow.start_run():
            mlflow.log_artifacts(local_dir=model_dir, artifact_path=artifact_path)
            # On crée une nouvelle version du modèle dans le registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            result = mlflow.register_model(model_uri, name=register_name)
            logger.info(f"Model registered: {result.name}, version: {result.version}")

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    def load_model(self, model_name: str, version_or_alias: str):
        # model_uri: models:/model_name/version ou models:/model_name@alias
        model_uri = f"models:/{model_name}/{version_or_alias}"
        return pyfunc.load_model(model_uri)

def get_mlflow_service():
    return MLFlowService()

# ===========================================================================
# Model Manager
# ===========================================================================

class ModelManager:
    def __init__(self, mlflow_service: MLFlowService):
        self.mlflow_service = mlflow_service

    def fetch_and_register_hf_model(self, artifact_name: str, artifact_path: str, register_name: str):
        """Télécharger un modèle Hugging Face et l'enregistrer dans le Model Registry MLflow."""
        try:
            logger.info(f"Téléchargement du modèle '{artifact_name}' depuis Hugging Face...")
            model_dir = Path(f"/tmp/{artifact_name.replace('/', '_')}")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Lister les fichiers du repo HF
            files = list_repo_files(repo_id=artifact_name, token=HF_API_TOKEN)
            logger.info(f"Fichiers dans le repo HF {artifact_name}: {files}")

            file_patterns = [".bin", ".safetensors", "config.json", "tokenizer.json"]
            downloaded_files = []
            for file in files:
                if any(pattern in file for pattern in file_patterns):
                    logger.info(f"Téléchargement de {file}...")
                    downloaded_path = hf_hub_download(repo_id=artifact_name, filename=file, local_dir=str(model_dir), token=HF_API_TOKEN)
                    downloaded_files.append(downloaded_path)
                else:
                    logger.debug(f"Fichier ignoré: {file}")

            if not downloaded_files:
                raise ValueError(f"Aucun fichier modèle requis trouvé pour '{artifact_name}'.")

            logger.info("Fichiers téléchargés avec succès. Enregistrement dans MLflow...")
            self.mlflow_service.log_artifacts_and_register_model(
                model_dir=str(model_dir),
                artifact_path=artifact_path,
                register_name=register_name,
            )
        except Exception as e:
            logger.error(f"Échec du téléchargement/enregistrement du modèle '{artifact_name}': {e}")
            raise ValueError(f"Erreur lors du traitement du modèle: {e}")
    

    def promote_model(self, artifact_name: str, version: int, alias: str):
        """
        Promotes a specific version of a model by assigning an alias.

        Args:
            artifact_name (str): Name of the registered model.
            version (int): Version number of the model.
            alias (str): Alias to assign (e.g., "champion").
        """
        try:
            self.mlflow_client.set_registered_model_alias(
                name=artifact_name, alias=alias, version=version
            )
        except MlflowException as e:
            raise RuntimeError(
                f"Failed to promote model '{artifact_name}' version '{version}' with alias '{alias}': {e}"
            )
        
def get_model_manager(mlflow_service: MLFlowService = Depends(get_mlflow_service)):
    return ModelManager(mlflow_service=mlflow_service)

# ===========================================================================
# TrainService reeal
# ===========================================================================

class TrainService:
    def __init__(
        self,
        db: Session,
        s3_service: S3Service,
        mlflow_service: MLFlowService,
        model_manager: ModelManager,
    ):
        """
        Initializes the TrainService with dependencies.
        """
        self.db = db
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.model_manager = model_manager

    def split_dataset(self, data, split_ratio=0.9):
        """
        Splits data into training and evaluation sets.
        """
        from random import shuffle

        shuffle(data)
        split_idx = int(len(data) * split_ratio)
        return data[:split_idx], data[split_idx:]

    def load_dataset(self, train_input: TrainInput) -> str:
        """
        Retrieves a dataset from the specified source.
        """
        if train_input.dataset_id:
            # Load from database and S3
            dataset = (
                self.db.query(Dataset)
                .filter(Dataset.id == train_input.dataset_id)
                .first()
            )
            if not dataset:
                raise ValueError("Dataset not found.")
            s3_url = dataset.data.get("s3_url")
        elif train_input.s3_url:
            s3_url = train_input.s3_url
        elif train_input.train_data:
            # Assume local file
            dataset_path = train_input.train_data
            return dataset_path
        else:
            raise ValueError("No dataset provided.")

        # Proceed to download from S3
        s3_info = self.s3_service.parse_s3_url(s3_url)
        if not s3_info:
            raise ValueError("Invalid S3 URL.")
        bucket_name, object_key = s3_info
        local_dataset_path = f"/tmp/{os.path.basename(object_key)}"
        success = self.s3_service.download_file(
            bucket_name, object_key, Path(local_dataset_path)
        )
        if not success:
            raise ValueError("Failed to download dataset from S3.")
        return local_dataset_path

    def train_model(self, train_input: TrainInput) -> TrainResponse:
        try:
            # Load dataset from the given dataset_path
            dataset_path = train_input.train_data
            if not dataset_path or not os.path.exists(dataset_path):
                raise FileNotFoundError(f"The dataset file {dataset_path} was not found.")

            with open(dataset_path, "r") as f:
                data = json.load(f)

            train_data, eval_data = self.split_dataset(data, train_input.split_ratio)

            # Load model using ModelManager
            model = self.mlflow_service.load_model(train_input.model_name, train_input.model_version)

            # Start MLFlow run
            run_name = f"Training: {train_input.custom_model_name or train_input.artifact_name}"
            self.mlflow_service.start_run(run_name=run_name)
            self.mlflow_service.log_params({
                "artifact_name": train_input.artifact_name,
                "custom_model_name": train_input.custom_model_name,
                "split_ratio": train_input.split_ratio,
                "learning_rate": train_input.learning_rate,
                "weight_decay": train_input.weight_decay,
                "batch_size": train_input.batch_size,
                "epochs": train_input.epochs,
                "compile_model": train_input.compile_model,
            })

            # Train model
            model.train(
                train_data=train_data,
                eval_data={"samples": eval_data},
                learning_rate=train_input.learning_rate,
                weight_decay=train_input.weight_decay,
                batch_size=train_input.batch_size,
                epochs=train_input.epochs,
                compile=train_input.compile_model,
            )
            model_save_name = train_input.custom_model_name or train_input.artifact_name
            model_save_path = Path(f"models/{model_save_name}")
            model_save_path.mkdir(parents=True, exist_ok=True)

            # Save the fine-tuned model before logging artifacts
            model.save_pretrained(model_save_path)

            # Log artifacts and register model in MLflow
            mlflow.log_artifacts(str(model_save_path), artifact_path="trained_model")
            self.mlflow_service.register_model(model_save_name, model_save_path)

            # Zip and upload the trained model to S3
            s3_url = self.model_manager.zip_and_upload_model(model_save_name)

            # Fin de run MLflow
            run_id = mlflow.active_run().info.run_id
            self.mlflow_service.end_run()
            logger.info(f"Training {run_id} completed and model saved at {model_save_path}")

            # Log training run to the database
            training_run = TrainingRun(
                dataset_id=train_input.dataset_id,
                epochs=train_input.epochs,
                batch_size=train_input.batch_size,
                status="Completed",
                s3_url=s3_url
            )
            self.db.add(training_run)
            self.db.commit()
            self.db.refresh(training_run)

            return TrainResponse(
                id=training_run.id,
                run_id=run_id,
                dataset_id=training_run.dataset_id,
                epochs=training_run.epochs,
                batch_size=training_run.batch_size,
                status=training_run.status,
                created_at=training_run.created_at,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.mlflow_service.end_run(status="FAILED")
            raise



# ===========================================================================
# Router
# ===========================================================================

router = APIRouter()

@router.post("/upload_model_from_huggingface")
async def upload_model_huggingface(
    artifact_name: str = Form(..., description="Name of the Hugging Face model"),
    register_name: str = Form(..., description="Name to register the model in MLflow"),
    artifact_path: str = Form("model_artifacts", description="Path to save artifacts in MLflow"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Upload a Hugging Face model and register it in MLflow.
    """
    try:
        model_manager.fetch_and_register_hf_model(
            artifact_name=artifact_name,
            artifact_path=artifact_path,
            register_name=register_name,
        )
        return {"message": f"Model '{artifact_name}' registered successfully as '{register_name}'."}
    except Exception as e:
        logger.error(f"Error during model upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {e}")


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    training_params_file: Optional[UploadFile] = File(None, description="Paramètres d'entraînement au format JSON."),
    db=Depends(get_db),
    s3_service: S3Service = Depends(get_s3_service),
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
    model_manager: ModelManager = Depends(get_model_manager)
):
    dataset_path = None

    # Validation et traitement des différentes sources
    if request.source_type == DatasetSource.file:
        if not request.dataset_file:
            raise HTTPException(status_code=400, detail="Fichier dataset requis pour source_type='file'.")
        dataset_path = f"/tmp/{request.dataset_file.filename}"
        async with aiofiles.open(dataset_path, "wb") as out_file:
            await out_file.write(await request.dataset_file.read())
    elif request.source_type == DatasetSource.s3:
        if not request.s3_url:
            raise HTTPException(status_code=400, detail="URL S3 requise pour source_type='s3'.")
        s3_info = s3_service.parse_s3_url(request.s3_url)
        if not s3_info:
            raise HTTPException(status_code=400, detail="URL S3 invalide.")
        bucket_name, object_key = s3_info
        dataset_path = f"/tmp/{object_key.split('/')[-1]}"
        if not s3_service.download_file(bucket_name, object_key, Path(dataset_path)):
            raise HTTPException(status_code=500, detail="Échec du téléchargement du dataset depuis S3.")
    elif request.source_type == DatasetSource.huggingface:
        if not request.huggingface_dataset:
            raise HTTPException(status_code=400, detail="Nom du dataset HuggingFace requis pour source_type='huggingface'.")
        api = HfApi(token=HF_API_TOKEN)
        try:
            files = api.list_repo_files(repo_id=request.huggingface_dataset)
            json_files = [f for f in files if f.endswith(".json")]
            if not json_files:
                raise HTTPException(status_code=400, detail="Aucun fichier JSON trouvé dans le dataset HuggingFace.")
            target_file = json_files[0]
            local_json_path = hf_hub_download(repo_id=request.huggingface_dataset, filename=target_file, token=HF_API_TOKEN)
            dataset = load_dataset("json", data_files=local_json_path)
            split_name = "train" if "train" in dataset else list(dataset.keys())[0]
            dataset_path = f"/tmp/{request.huggingface_dataset.replace('/', '_')}.json"
            with open(dataset_path, "w") as f:
                json.dump([dict(x) for x in dataset[split_name]], f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Échec du traitement du dataset HuggingFace: {e}")

    # Lecture des paramètres d'entraînement supplémentaires
    training_params = {}
    if request.training_params_file:
        try:
            content = await request.training_params_file.read()
            training_params = json.loads(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Fichier de paramètres JSON invalide : {e}")
 
    # Fusion des paramètres par défaut
    training_params.update({
        "learning_rate": request.learning_rate,
        "weight_decay": request.weight_decay,
        "batch_size": request.batch_size,
        "epochs": request.epochs,
        "compile_model": request.compile_model,
    })

    # Nom du modèle à enregistrer
    register_name = request.custom_model_name or f"{request.model_name}_v{request.version}"
    train_input = TrainInput(
        model_name=request.model_name,
        model_version=request.version,
        custom_model_name=register_name,
        train_data=dataset_path,
        split_ratio=request.split_ratio,
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

# ===========================================================================
# Lancement de l'API
# ===========================================================================

app.include_router(router)
