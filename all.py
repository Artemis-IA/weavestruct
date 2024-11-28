
# config.py
#-----
# src/config.py
import os
from pathlib import Path
from typing import ClassVar, Dict
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    # Application settings
    # Constants for Models and Device
    # Constants for Models and Device
    MODELS: ClassVar[Dict[str, str]] = {
        "GLiNER-S": "urchade/gliner_smallv2.1",
        "GLiNER-M": "urchade/gliner_mediumv2.1",
        "GLiNER-L": "urchade/gliner_largev2.1",
        "GLIREL": "jackboyla/glirel-large-v0",
        "GLiNER-Multitask": "knowledgator/gliner-multitask-large-v0.5",
    }

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Proxy settings
    USE_ET_PROXY: bool = False
    HTTP_PROXY: str = ""
    HTTPS_PROXY: str = ""
    NO_PROXY: str = ""

    # Neo4j settings
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "your_password"

    # PostgreSQL settings
    PG_MAJOR: int = 16
    POSTGRE_PORT: int = 5432
    POSTGRE_USER: str = "postgre_user"
    POSTGRE_PASSWORD: str = "postgre_password"
    POSTGRE_DB: str = "postgre_db"
    POSTGRE_HOST: str = "localhost"
    DJANGO_DB: str = "default"

    # Derived PostgreSQL settings
    DATABASE_URL: str = "postgresql://postgre_user:postgre_password@localhost:5432/postgre_db"

    # MLflow settings
    MLFLOW_USER: str = "mlflow_user"
    MLFLOW_PASSWORD: str = "mlflow_password"
    MLFLOW_DB: str = "mlflow_db"
    MLFLOW_PORT: int = 5002
    MLFLOW_TRACKING_URI: str = "http://mlflow:5002"
    MLFLOW_S3_ENDPOINT_URL: str = "http://minio:9000"
    MLFLOW_S3_IGNORE_TLS: bool = True

    # Derived MLflow settings
    MLFLOW_BACKEND_STORE_URI: str = "postgresql+psycopg2://postgre_user:postgre_password@localhost:5432/mlflow_db"
    MLFLOW_ARTIFACT_ROOT: str = "s3://minio:minio123@http://minio:9000/mlflow"

    # MinIO settings
    MINIO_PORT: int = 9000
    MINIO_CONSOLE_PORT: int = 9001
    MINIO_CLIENT_PORT: int = 9002
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "minio123"
    MINIO_ROOT_USER: str = "minio"
    MINIO_ROOT_PASSWORD: str = "minio123"
    MINIO_API_URL: str = "http://minio:9000"
    MINIO_URL: str = "http://minio:9000"

    # AWS settings for MinIO compatibility
    AWS_ACCESS_KEY_ID: str = "minio"
    AWS_SECRET_ACCESS_KEY: str = "minio123"
    AWS_DEFAULT_REGION: str = "eu-west-1"

    # Label Studio settings
    LABEL_STUDIO_USER: str = "labelstudio_user"
    LABEL_STUDIO_PASSWORD: str = "labelstudio_password"
    LABEL_STUDIO_DB: str = "labelstudio_db"
    LABEL_STUDIO_HOST: str = "label-studio"
    LABEL_STUDIO_PORT: int = 8081
    LABEL_STUDIO_USERNAME: str = "admin_user"
    LABEL_STUDIO_EMAIL: str = "admin@example.com"
    LABEL_STUDIO_API_KEY: str = "secure_api_key_123"
    LABEL_STUDIO_BUCKET_NAME: str = "mlflow-source"
    LABEL_STUDIO_BUCKET_PREFIX: str = "source_data/"
    LABEL_STUDIO_BUCKET_ENDPOINT_URL: str = "http://minio:9000"
    LABEL_STUDIO_BUCKET_ACCESS_KEY: str = "minio"
    LABEL_STUDIO_BUCKET_SECRET_KEY: str = "minio123"
    LABEL_STUDIO_TARGET_BUCKET: str = "mlflow-annotations"
    LABEL_STUDIO_TARGET_PREFIX: str = "annotations/"
    LABEL_STUDIO_TARGET_ACCESS_KEY: str = "minio"
    LABEL_STUDIO_TARGET_SECRET_KEY: str = "minio123"
    LABEL_STUDIO_TARGET_ENDPOINT_URL: str = "http://minio:9000"
    LABEL_STUDIO_PROJECT_NAME: str = "proj-1"
    LABEL_STUDIO_PROJECT_TITLE: str = "Machine Learning Annotations Project"
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED: bool = True
    LS_DATABASE_URL: str = "postgresql://labelstudio_user:labelstudio_password@localhost:5432/labelstudio_db"

    # Prometheus settings
    PROMETHEUS_PORT: int = 9090

    # GLiNER settings
    GLINER_BASIC_AUTH_USER: str = "my_user"
    GLINER_BASIC_AUTH_PASS: str = "my_password"
    GLINER_MODEL_NAME: str = "knowledgator/gliner-multitask-large-v0.5"
    LABEL_STUDIO_ML_BACKENDS: str = '[{"url": "http://gliner:9097", "name": "GLiNER"}]'

    GLIREL_MODEL_NAME: str = "jackboyla/glirel-large-v0"
    # Secret Key
    SECRET_KEY: str = "super_secret_key_123"

    # General settings
    WORKERS: int = 4
    THREADS: int = 4
    TEST_ENV: str = "my_test_env"
    LOCIP: str = "192.168.1.106"

    # ML Backend
    MLBACKEND_PORT: int = 9097

    # Default Models
    DEFAULT_MODELS: str = "urchade/gliner_smallv2.1"

    # Training Configuration
    train_config: dict = {
        "num_steps": 10_000,
        "train_batch_size": 2,
        "eval_every": 1_000,
        "save_directory": "checkpoints",
        "warmup_ratio": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr_encoder": 1e-5,
        "lr_others": 5e-5,
        "freeze_token_rep": False,
        "max_types": 25,
        "shuffle_types": True,
        "random_drop": True,
        "max_neg_type_ratio": 1,
        "max_len": 384,
    }

    # Text splitting settings
    TEXT_CHUNK_SIZE: int = 1000
    TEXT_CHUNK_OVERLAP: int = 200
    CONF_FILE: str = "../conf/gli_config.yml"
    

    # Ollama
    OLLAMA_MODEL: str ="nomic-embed-text"

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

# Instanciation de la configuration
settings = Settings()
#-----

# __init__.py
#-----

#-----

# main.py
#-----
import time
from fastapi import FastAPI, Response, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from routers import documents, entities, relationships, search, graph, datasets, train
from utils.metrics import MetricsManager
from config import settings

# Initialize metrics manager
metrics_manager = MetricsManager(prometheus_port=8002)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Document Processing and Graph API",
        version="2.0.0",
        description="API for Document processing, NER/Relation Extraction & Embeddings/Graph Indexing",
        docs_url="/",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middleware for CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics middleware
    Instrumentator(
        excluded_handlers=["^/metrics", "^/redoc", "^/openapi.json"],
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(
        app,
        metric_namespace="metrics"
    ).expose(app)

    # Include Routers
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(entities.router, prefix="/entities", tags=["Entities"])
    app.include_router(relationships.router, prefix="/relationships", tags=["Relationships"])
    app.include_router(search.router, prefix="/search", tags=["Search"])
    app.include_router(graph.router, prefix="/graph", tags=["Graph"])
    app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
    app.include_router(train.router, prefix="/train", tags=["Training"])

    # Metrics Router
    metrics_router = APIRouter(prefix="/metrics", tags=["Metrics"])

    @metrics_router.get("/")
    async def metrics():
        """Expose Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(metrics_router)

    # Middleware to track custom metrics
    @app.middleware("http")
    async def custom_metrics_middleware(request: Request, call_next):
        """Track request metrics and system stats."""
        start_time = time.time()
        metrics_manager.REQUEST_COUNT.inc()
        response = await call_next(request)
        latency = time.time() - start_time
        metrics_manager.PROCESS_TIME.observe(latency)
        metrics_manager.log_system_metrics()
        return response

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Application starting...")
    metrics_manager._remove_codecarbon_lock()
    metrics_manager.start_metrics_server()
    system_metrics = metrics_manager.get_system_metrics()
    device_type = "cuda" if system_metrics.get("cuda") else "CPU"
    logger.info(f"Device Type: {device_type}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Application shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

#-----

# dependencies.py
#-----
# dependencies.py
import yaml
from fastapi import Depends
from sqlalchemy.orm import Session
from typing import Generator

from config import settings
from utils.database import SessionLocal
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.document_processor import DocumentProcessor
from services.pgvector_service import PGVectorService
from services.neo4j_service import Neo4jService
from services.rag_service import RAGChainService
from services.embedding_service import EmbeddingService
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_postgres import PGVector
from langchain_ollama.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase

# Dependency to get the SQLAlchemy session
def get_db() -> Generator[Session, None, None]:
    """Yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get the S3 service
def get_s3_service() -> S3Service:
    """Returns an instance of the S3 service."""
    return S3Service(
        s3_client=None,
        endpoint_url=settings.MINIO_URL,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        input_bucket="docs-input",
        output_bucket="docs-output",
        layouts_bucket="layouts"
    )

# Dependency to get the MLflow service
def get_mlflow_service() -> MLFlowService:
    """Returns an instance of the MLflow service."""
    return MLFlowService(tracking_uri=settings.MLFLOW_TRACKING_URI)

# Dependency for PGVector service
def get_pgvector_service() -> PGVectorService:
    return PGVectorService(
        db_url=settings.DATABASE_URL,
        table_name="document_embeddings"
    )


# Dependency for embedding service
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=settings.OLLAMA_MODEL)


# Dependency for PGVector vector store
def get_pgvector_vector_store() -> PGVector:
    embedding_service = get_embedding_service()
    return PGVector(
        collection_name="document_embeddings",
        connection=settings.DATABASE_URL,
        embeddings=embedding_service.embedding_model.embed_documents
    )

# Dependency to get the Neo4j driver
def get_neo4j_driver() -> GraphDatabase:
    """Returns a Neo4j driver instance."""
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

# Dependency to get the Neo4j service
def get_neo4j_service() -> Neo4jService:
    """Returns an instance of the Neo4j service."""
    return Neo4jService(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )

# Initialize reusable text splitter
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Returns an instance of the text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.TEXT_CHUNK_SIZE,
        chunk_overlap=settings.TEXT_CHUNK_OVERLAP
    )

# Dependency to get the GLiNER extractor

def get_gliner_extractor() -> GLiNERLinkExtractor:
    with open(settings.CONF_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return GLiNERLinkExtractor(
        labels=config["labels"],
        model=settings.GLINER_MODEL_NAME,
    )


def get_graph_transformer() -> GlinerGraphTransformer:
    with open(settings.CONF_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return GlinerGraphTransformer(
        allowed_nodes=config["allowed_nodes"],
        allowed_relationships=config["allowed_relationships"],
        gliner_model=settings.GLINER_MODEL_NAME,
        glirel_model=settings.GLIREL_MODEL_NAME,
        entity_confidence_threshold=0.1,
        relationship_confidence_threshold=0.1,
    )


def get_document_processor(db: Session = Depends(get_db)) -> DocumentProcessor:
    s3_service = get_s3_service()
    mlflow_service = get_mlflow_service()
    pgvector_service = get_pgvector_service()
    text_splitter = get_text_splitter()
    neo4j_service = get_neo4j_service()
    embedding_service = get_embedding_service()
    gliner_extractor = get_gliner_extractor()
    graph_transformer = get_graph_transformer()
    
    return DocumentProcessor(
        s3_service=s3_service,
        mlflow_service=mlflow_service,
        pgvector_service=pgvector_service,
        neo4j_service=neo4j_service,
        embedding_service=embedding_service,
        session=db,
        text_splitter=text_splitter,
        graph_transformer=graph_transformer,
        gliner_extractor=gliner_extractor,
    )

# Dependency to get the RAG service
def get_rag_service() -> RAGChainService:
    """Returns an instance of the RAG Chain Service."""
    vector_store = get_pgvector_vector_store()
    return RAGChainService(retriever=vector_store.as_retriever())

#-----

# schemas/__init__.py
#-----

#-----

# schemas/dataset.py
#-----
# src/schemas/dataset.py
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

class DatasetCreate(BaseModel):
    name: str = Field(..., example="Sample Dataset")
    data: List[Dict] = Field(..., example=[{"text": "Example", "entities": ["Entity1", "Entity2"]}])

class DatasetUpdate(BaseModel):
    name: str = Field(..., example="Updated Dataset")
    data: List[Dict] = Field(..., example=[{"text": "Updated Example", "entities": ["Entity3"]}])

class DatasetResponse(BaseModel):
    id: int
    name: str
    data: List[Dict]
    created_at: datetime

    class Config:
        from_attributes = True


#-----

# schemas/models_dict.py
#-----
# src/schemas/models_dict.py

from enum import Enum

class ModelName(str, Enum):
    GLiNER_S = "GLiNER-S"
    GLiNER_M = "GLiNER-M"
    GLiNER_L = "GLiNER-L"
    GLiNER_News = "GLiNER-News"
    GLiNER_PII = "GLiNER-PII"
    GLiNER_Bio = "GLiNER-Bio"
    GLiNER_Bird = "GLiNER-Bird"
    NuNER_Zero = "NuNER-Zero"
    NuNER_Zero_4K = "NuNER-Zero-4K"
    NuNER_Zero_span = "NuNER-Zero-span"

#-----

# schemas/auth.py
#-----
# schemas/auth.py

from pydantic import BaseModel

class User(BaseModel):
    username: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: str | None = None

#-----

# schemas/inference.py
#-----
# src/schemas/inference.py
from pydantic import BaseModel
from fastapi import Form
from typing import List, Dict
from datetime import datetime

class InferenceRequest(BaseModel):
    labels: List[str]
    threshold: float
    flat_ner: bool
    multi_label: bool
    batch_size: int

    @classmethod
    def as_form(
        cls,
        labels: str = Form("PERSON,PLACE,THING,ORGANIZATION,DATE,TIME", description="Types d'entités à extraire"),
        threshold: float = Form(0.3, description="Seuil de confiance pour l'inférence"),
        flat_ner: bool = Form(True, description="If need to extract parts of complex entities: False"),
        multi_label: bool = Form(False, description="If entities belong to several classes: True"),
        batch_size: int = Form(12, description="Taille du lot d'inférence")
    ) -> "InferenceRequest":
        # Les labels sont séparés par des virgules dans le formulaire, donc nous les convertissons en liste
        return cls(
            labels=labels.split(","),
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            batch_size=batch_size
        )

class InferenceResponse(BaseModel):
    id: int
    file_path: str
    entities: List[Dict]
    created_at: datetime

    class Config:   
        from_attributes = True

#-----

# schemas/train.py
#-----
# src/schemas/train.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
class TrainRequest(BaseModel):
    dataset_id: int = Field(..., example=1)
    epochs: int = Field(10, example=20)
    batch_size: int = Field(32, example=64)

class TrainResponse(BaseModel):
    id: int
    run_id: str
    dataset_id: int
    epochs: int
    batch_size: int
    status: str
    created_at: datetime
    class Config:
        from_attributes = True

#-----

# utils/helpers.py
#-----
import os
import random
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Load a YAML configuration file
def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its content as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed content of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Save data to a YAML file
def save_to_yaml(data: Dict[str, Any], file_path: str):
    """
    Save a dictionary to a YAML file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to the YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

# Load a JSON configuration file
def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file and return its content as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed content of the JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

# Save data to a JSON file
def save_to_json(data: Dict[str, Any], file_path: str):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Create a directory if it doesn't exist
def ensure_directory(directory_path: str):
    """
    Create a directory if it does not already exist.

    Args:
        directory_path (str): Path of the directory to create.
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

# Delete a directory
def delete_directory(directory_path: str):
    """
    Delete a directory and all its contents.

    Args:
        directory_path (str): Path of the directory to delete.
    """
    shutil.rmtree(directory_path, ignore_errors=True)

# Generate a random string
def generate_random_string(length: int = 8) -> str:
    """
    Generate a random alphanumeric string of specified length.

    Args:
        length (int): Length of the generated string (default is 8).

    Returns:
        str: Random alphanumeric string.
    """
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choices(characters, k=length))

# Get an environment variable with a default value
def get_env_variable(key: str, default: Optional[str] = None) -> str:
    """
    Get the value of an environment variable or return a default value if not set.

    Args:
        key (str): The environment variable key.
        default (str, optional): The default value to return if the variable is not set.

    Returns:
        str: The value of the environment variable or the default value.
    """
    return os.getenv(key, default)

#-----

# utils/metrics.py
#-----
import os
import psutil
import GPUtil
import torch
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from codecarbon import EmissionsTracker


class MetricsManager:
    """
    Manage application metrics, system stats logging, and Prometheus metrics server.
    """

    def __init__(self, prometheus_port: int = 8002):
        # Define Prometheus metrics
        self.REQUEST_COUNT = Counter("app_request_count", "Total number of requests")
        self.PROCESS_TIME = Histogram("app_process_time_seconds", "Request processing time")
        self.GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "GPU memory usage")
        self.CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percentage")
        self.MEMORY_USAGE = Gauge("memory_usage_bytes", "Memory usage in bytes")
        self.CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Estimated CO2 emissions")

        # Neo4j metrics
        self.NEO4J_REQUEST_COUNT = Counter("neo4j_request_count", "Number of requests sent to Neo4j")
        self.NEO4J_REQUEST_FAILURES = Counter("neo4j_request_failures", "Number of failed Neo4j requests")
        self.NEO4J_REQUEST_LATENCY = Histogram("neo4j_request_latency_seconds", "Latency of Neo4j requests")

        # PostgreSQL metrics
        self.POSTGRES_QUERY_COUNT = Counter("postgres_query_count", "Number of successful PostgreSQL queries")
        self.POSTGRES_QUERY_FAILURES = Counter("postgres_query_failures", "Number of failed PostgreSQL queries")
        self.POSTGRES_QUERY_LATENCY = Histogram("postgres_query_latency_seconds", "Latency of PostgreSQL queries")

        # Document processing metrics
        self.DOCUMENT_PROCESSING_SUCCESS = Counter(
            "document_processing_success", "Number of successfully processed documents"
        )
        self.DOCUMENT_PROCESSING_FAILURES = Counter(
            "document_processing_failures", "Number of failed document processing attempts"
        )

        # CO2 emissions tracker
        self.emissions_tracker = EmissionsTracker(
            project_name="doc_processing",
            save_to_file=False,
            save_to_prometheus=True,
            prometheus_url=f"localhost:{prometheus_port}",
        )

        self.prometheus_port = prometheus_port

    def start_metrics_server(self):
        """
        Start the Prometheus metrics server.
        """
        start_http_server(self.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {self.prometheus_port}.")

    def log_system_metrics(self):
        """
        Log system metrics: CPU, memory, GPU usage, and CO2 emissions.
        """
        try:
            # Log CPU and memory usage
            self.CPU_USAGE.set(psutil.cpu_percent())
            self.MEMORY_USAGE.set(psutil.virtual_memory().used)

            # Log GPU memory usage
            gpus = GPUtil.getGPUs()
            if gpus:
                self.GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)

            # Log CO2 emissions
            emissions = self.emissions_tracker.stop()
            if emissions is not None:
                self.CARBON_EMISSIONS.set(emissions)
                logger.info(f"CO2 emissions logged: {emissions:.6f} kgCO₂eq")
            else:
                logger.warning("No emissions data available.")
        except Exception as e:
            logger.warning(f"Error logging system metrics: {e}")

    @staticmethod
    def get_system_metrics() -> dict:
        """
        Retrieve and log system metrics.
        """
        metrics = {
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_mb": psutil.virtual_memory().used / (1024 * 1024),
        }

        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics["gpu_memory_usage_mb"] = gpus[0].memoryUsed
        else:
            metrics["gpu_memory_usage_mb"] = None

        # CUDA or CPU check
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            metrics["cuda"] = True
            metrics["gpu_name"] = gpu_name
            logger.info(f"CUDA is available. Using GPU: {gpu_name}")
        else:
            metrics["cuda"] = False
            logger.info("CUDA is not available. Using CPU.")

        logger.info(f"System metrics: {metrics}")
        return metrics

    @staticmethod
    def _remove_codecarbon_lock() -> None:
        """
        Remove CodeCarbon lock file to avoid tracker errors.
        """
        lock_file = "/tmp/.codecarbon.lock"
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info("CodeCarbon lock file removed.")
            except Exception as e:
                logger.warning(f"Error removing CodeCarbon lock file: {e}")

#-----

# utils/logging_utils.py
#-----

# utils/logging.py
import os
from loguru import logger
from huggingface_hub import HfApi
import mlflow
from mlflow.tracking import MlflowClient
from codecarbon import EmissionsTracker
from typing import Optional
import time

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
            "Tokenizer Model": ("microsoft/deberta-v3-large", os.path.join(self.huggingface_cache, "models--microsoft--deberta-v3-large"))
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

        self.emissions_tracker = EmissionsTracker(project_name="model_logging", save_to_file=False, save_to_prometheus=True, prometheus_url="localhost:8002")
        logger.info("CodeCarbon tracker initialized.")

    def log_model_details(self):
        logger.info("Starting model logging process...")
        mlflow.end_run()  # Ensure no active runs are in progress

        try:
            with mlflow.start_run(run_name="Model Logging") as run:
                run_id = run.info.run_id

                for model_name, (model_id, model_file_path) in self.static_models.items():
                    logger.info(f"Processing model: {model_name}")
                    self._log_model_metadata(model_name, model_id, model_file_path, run_id)

            logger.info("Model logging process completed.")
            return {"message": "Model logging completed successfully"}
        except Exception as e:
            logger.error(f"Error in logging model details: {e}")
            return {"error": str(e)}

    def _log_model_metadata(self, model_name, model_id, model_file_path, run_id):
        try:
            # Check if the model is registered in MLflow
            registered_models = [rm.name for rm in self.client.search_registered_models()]
            if model_name not in registered_models:
                self.client.create_registered_model(model_name)

            # Fetch model metadata from Hugging Face
            model_info = self.hf_api.model_info(model_id)
            model_version = model_info.sha  # Unique identifier for version
            model_description = self._fetch_readme(model_id) or "No description available."  # Use README.md content as description
            model_tags = model_info.tags

            # Log metadata to MLflow
            mlflow.set_tag(f"{model_name}_description", model_description)
            for tag in model_tags:
                mlflow.set_tag(f"{model_name}_tag_{tag}", True)
            mlflow.log_param(f"{model_name}_version", model_version)

            # Log model file as artifact if it exists
            if os.path.exists(model_file_path):
                artifact_path = f"artifacts/{model_name}"
                mlflow.log_artifact(model_file_path, artifact_path=artifact_path)
                self.client.create_model_version(
                    name=model_name,
                    source=f"{mlflow.get_artifact_uri()}/{artifact_path}",
                    run_id=run_id
                )
            else:
                logger.warning(f"Model path not found: {model_file_path}")
        except Exception as e:
            logger.error(f"Error logging metadata for model {model_name}: {e}")

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

#-----

# utils/database.py
#-----
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from config import settings
import logging

# Load database URL from settings
db_url = settings.DATABASE_URL

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

#-----

# models/dataset.py
#-----
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="Unnamed Dataset")
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

#-----

# models/document_log.py
#-----
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Callable
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentLog(Base):
    __tablename__ = 'document_logs'

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    s3_url = Column(String, nullable=False)

    def __repr__(self):
        return f"<DocumentLog(id={self.id}, file_name='{self.file_name}', s3_url='{self.s3_url}')>"


class DocumentLogService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def log_document(self, file_name: str, s3_url: str) -> None:
        """Logs a document entry in the database."""
        try:
            with self.session_factory() as session:
                log_entry = DocumentLog(file_name=file_name, s3_url=s3_url)
                session.add(log_entry)
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"Failed to log document: {e}")

#-----

# models/entity.py
#-----
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class EntityBase(BaseModel):
    name: str
    type: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Sample Entity",
                "type": "Organization",
                "properties": {
                    "location": "New York",
                    "employees": 100
                }
            }
        }

class EntityCreate(EntityBase):
    """
    Model for creating a new entity.
    Inherits from EntityBase and can be extended for additional fields required at creation.
    """
    pass

class Entity(EntityBase):
    """
    Model representing an entity with an ID, as returned from the database or API.
    """
    id: str = Field(..., description="The unique identifier of the entity")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "name": "Sample Entity",
                "type": "Organization",
                "properties": {
                    "location": "New York",
                    "employees": 100
                }
            }
        }

#-----

# models/relationship.py
#-----
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class RelationshipBase(BaseModel):
    source_id: str = Field(..., description="The ID of the source entity")
    target_id: str = Field(..., description="The ID of the target entity")
    type: str = Field(..., description="The type of the relationship")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties of the relationship")

    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "123e4567-e89b-12d3-a456-426614174001",
                "target_id": "789e4567-e89b-12d3-a456-426614174002",
                "type": "Partnership",
                "properties": {
                    "since": "2021-01-01",
                    "status": "active"
                }
            }
        }

class RelationshipCreate(RelationshipBase):
    """
    Model for creating a new relationship.
    Inherits from RelationshipBase and can be extended for additional fields required at creation.
    """
    pass

class Relationship(RelationshipBase):
    """
    Model representing a relationship with an ID, as returned from the database or API.
    """
    id: str = Field(..., description="The unique identifier of the relationship")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "456e4567-e89b-12d3-a456-426614174003",
                "source_id": "123e4567-e89b-12d3-a456-426614174001",
                "target_id": "789e4567-e89b-12d3-a456-426614174002",
                "type": "Partnership",
                "properties": {
                    "since": "2021-01-01",
                    "status": "active"
                }
            }
        }

#-----

# models/training_run.py
#-----
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, default=lambda: str(uuid.uuid4()), unique=True)
    dataset_id = Column(Integer, nullable=False)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=32)
    status = Column(String, default="Started")
    created_at = Column(DateTime, default=datetime.utcnow)

#-----

# models/document.py
#-----
from pydantic import BaseModel
from typing import Dict, Any

class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Sample Document",
                "content": "This is the content of the document.",
                "metadata": {
                    "author": "John Doe",
                    "tags": ["example", "sample"]
                },
                "created_at": "2023-11-12T10:00:00Z",
                "updated_at": "2023-11-12T12:00:00Z"
            }
        }

#-----

# middleware/custom_metrics.py
#-----
import time
import psutil
import GPUtil
from prometheus_client import Histogram, Counter, Gauge
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Prometheus metrics
REQUEST_COUNT = Counter("app_request_count", "Total number of requests received")
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests in seconds")
CPU_USAGE = Gauge("app_cpu_usage_percent", "CPU usage in percent")
MEMORY_USAGE = Gauge("app_memory_usage_bytes", "Memory usage in bytes")
GPU_MEMORY_USAGE = Gauge("app_gpu_memory_usage_bytes", "GPU memory usage in bytes")

class CustomMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        REQUEST_COUNT.inc()  # Increment the request count

        # Call the next middleware or actual request handler
        response = await call_next(request)

        # Measure latency
        process_time = time.time() - start_time
        REQUEST_LATENCY.observe(process_time)

        # Log system metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)

        # Log GPU metrics if available
        gpus = GPUtil.getGPUs()
        if gpus:
            GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)  # Only log the first GPU

        return response
#-----

# routers/entities.py
#-----
# routers/entities.py
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from models.entity import EntityCreate, Entity
from dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.post("/entities/", response_model=Entity)
async def create_entity(entity: EntityCreate):
    logger.info(f"Creating entity: {entity.name}")
    try:
        created_entity = neo4j_service.create_entity(entity)
        logger.info(f"Successfully created entity: {created_entity.name}")
        return created_entity
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/", response_model=List[Entity])
async def get_entities():
    logger.info("Retrieving all entities")
    try:
        entities = neo4j_service.get_all_entities()
        logger.info(f"Retrieved {len(entities)} entities")
        return entities
    except Exception as e:
        logger.error(f"Error retrieving entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(entity_id: str):
    logger.info(f"Retrieving entity with ID: {entity_id}")
    try:
        entity = neo4j_service.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        logger.info(f"Successfully retrieved entity: {entity.name}")
        return entity
    except Exception as e:
        logger.error(f"Error retrieving entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/entities/{entity_id}", response_model=dict)
async def delete_entity(entity_id: str):
    logger.info(f"Deleting entity with ID: {entity_id}")
    try:
        success = neo4j_service.delete_entity(entity_id)
        if not success:
            raise HTTPException(status_code=404, detail="Entity not found")
        logger.info(f"Successfully deleted entity with ID: {entity_id}")
        return {"message": f"Entity {entity_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/graph.py
#-----
# routers/graph.py
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List, Dict, Any

from services.neo4j_service import Neo4jService
from dependencies import get_neo4j_service
from models.entity import Entity
from models.relationship import Relationship

router = APIRouter()

neo4j_service: Neo4jService = get_neo4j_service()


@router.get("/entities/", response_model=List[Entity])
async def get_all_entities():
    logger.info("Retrieving all entities from the graph")
    try:
        entities = neo4j_service.get_all_entities()
        logger.info(f"Retrieved {len(entities)} entities from the graph")
        return entities
    except Exception as e:
        logger.error(f"Error retrieving entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/", response_model=List[Relationship])
async def get_all_relationships():
    logger.info("Retrieving all relationships from the graph")
    try:
        relationships = neo4j_service.get_all_relationships()
        logger.info(f"Retrieved {len(relationships)} relationships from the graph")
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/", response_model=Dict[str, List[Dict[str, Any]]])
async def visualize_graph():
    logger.info("Generating graph visualization data")
    try:
        graph_data = neo4j_service.generate_graph_visualization()
        logger.info("Graph visualization data generated successfully")
        return graph_data
    except Exception as e:
        logger.error(f"Error generating graph visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/relationships.py
#-----
# # routers/relationships.py
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from models.relationship import RelationshipCreate, Relationship
from dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.post("/relationships/", response_model=Relationship)
async def create_relationship(relationship: RelationshipCreate):
    logger.info(f"Creating relationship: {relationship.type} between {relationship.source_id} and {relationship.target_id}")
    try:
        created_relationship = neo4j_service.create_relationship(relationship)
        logger.info(f"Successfully created relationship: {created_relationship.type}")
        return created_relationship
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/relationships/", response_model=List[Relationship])
async def get_relationships():
    logger.info("Retrieving all relationships")
    try:
        relationships = neo4j_service.get_all_relationships()
        logger.info(f"Retrieved {len(relationships)} relationships")
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/relationships/{relationship_id}", response_model=Relationship)
async def get_relationship(relationship_id: str):
    logger.info(f"Retrieving relationship with ID: {relationship_id}")
    try:
        relationship = neo4j_service.get_relationship(relationship_id)
        if not relationship:
            raise HTTPException(status_code=404, detail="Relationship not found")
        logger.info(f"Successfully retrieved relationship: {relationship.type}")
        return relationship
    except Exception as e:
        logger.error(f"Error retrieving relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/relationships/{relationship_id}", response_model=dict)
async def delete_relationship(relationship_id: str):
    logger.info(f"Deleting relationship with ID: {relationship_id}")
    try:
        success = neo4j_service.delete_relationship(relationship_id)
        if not success:
            raise HTTPException(status_code=404, detail="Relationship not found")
        logger.info(f"Successfully deleted relationship with ID: {relationship_id}")
        return {"message": f"Relationship {relationship_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/logging_router.py
#-----
# routers/logging.py
from fastapi import APIRouter
from utils.logging_utils import ModelLoggerService

router = APIRouter()

# Initialize the ModelLoggerService
model_logger_service = ModelLoggerService()

@router.post("/log_models/")
def log_models():
    """API endpoint to trigger logging of model details."""
    return model_logger_service.log_model_details()

@router.post("/log_queries/")
def log_queries(query: str):
    """API endpoint to trigger logging of queries."""
    return model_logger_service.log_query(query)
#-----

# routers/documents.py
#-----
# routers/documents.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
from typing import List
from loguru import logger
import aiofiles

from services.document_processor import DocumentProcessor
from services.rag_service import RAGChainService
from dependencies import get_document_processor, get_rag_service

router = APIRouter()

# Dependency injection
document_processor: DocumentProcessor = get_document_processor()
rag_service: RAGChainService = get_rag_service()


@router.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[str] = Form(default=["json"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False),
):
    """
    Endpoint pour télécharger et traiter des fichiers pour l'extraction de données.
    """
    logger.info(f"Received {len(files)} files for upload")
    success_count, failure_count = 0, 0

    for file in files:
        temp_file = Path(f"/tmp/{file.filename}")
        try:
            # Sauvegarde du fichier en utilisant aiofiles pour compatibilité async
            async with aiofiles.open(temp_file, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)

            # Traitement du document
            result = document_processor.process_and_index_document(temp_file)

            # Exportation
            document_processor.export_document(
                result, temp_file.parent, export_formats, export_figures, export_tables
            )

            logger.info(f"File {file.filename} processed successfully.")
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            failure_count += 1
        finally:
            if temp_file.exists():
                temp_file.unlink()  # Suppression du fichier temporaire

    return {
        "message": "Files processed",
        "success_count": success_count,
        "failure_count": failure_count,
    }


@router.post("/upload_path/")
async def upload_path(
    file_path: str = Form("/home/pi/Documents/IF-SRV/4pdfs_subset/"),
    export_formats: List[str] = Form(default=["json"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False),
):
    """
    Endpoint pour traiter un répertoire de fichiers.
    """
    logger.info(f"Processing directory: {file_path}")

    input_dir_path = Path(file_path)
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    input_file_paths = [
        file for file in input_dir_path.glob("*")
        if file.suffix.lower() in [".pdf", ".docx"]
    ]
    logger.info(f"Found {len(input_file_paths)} valid files in directory")

    success_count, failure_count = 0, 0

    for doc_path in input_file_paths:
        try:
            result = document_processor.process_and_index_document(doc_path)

            document_processor.export_document(
                result, doc_path.parent, export_formats, export_figures, export_tables
            )

            logger.info(f"File {doc_path.name} processed successfully.")
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {doc_path.name}: {e}")
            failure_count += 1

    return {
        "message": "Directory processed",
        "success_count": success_count,
        "failure_count": failure_count,
    }


@router.post("/index_document/")
async def index_document(file: UploadFile = File(...)):
    """
    Index a single document by extracting entities and relationships.
    """
    logger.info(f"Indexing document: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")

    # Use aiofiles for asynchronous file writing
    import aiofiles
    async with aiofiles.open(temp_file, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        document_processor.process_and_index_document(temp_file)
        logger.info(f"Successfully indexed document: {file.filename}")
        return {"message": f"Document {file.filename} indexed successfully."}
    except Exception as e:
        logger.error(f"Error indexing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")
    finally:
        temp_file.unlink()



@router.post("/rag_process/")
async def process_rag_document(file: UploadFile = File(...)):
    """
    Process a document for RAG, splitting it, embedding it, and storing it in the vector store.
    """
    logger.info(f"Processing document for RAG: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")

    async with temp_file.open("wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        result = rag_service.process_document_for_rag(temp_file)
        logger.info(f"Document successfully processed for RAG: {file.filename}")
        return {"message": "Document successfully processed for RAG.", "details": result}
    except Exception as e:
        logger.error(f"Error processing document for RAG: {file.filename}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document for RAG: {e}")
    finally:
        temp_file.unlink()

#-----

# routers/train.py
#-----
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from schemas.train import TrainRequest, TrainResponse
from services.train_service import TrainService
from dependencies import get_db
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/", response_model=TrainResponse, tags=["Training"])
def train_endpoint(request: TrainRequest, db: Session = Depends(get_db)):
    """
    Endpoint pour entraîner le modèle NER.
    """
    try:
        train_service = TrainService(db=db)
        training_run = train_service.train_model(request)
        return training_run
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/datasets.py
#-----
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional
from loguru import logger
from sqlalchemy.orm import Session

from dependencies import get_db
from models.dataset import Dataset
from schemas.dataset import DatasetCreate, DatasetResponse
from services.dataset_service import DatasetService

router = APIRouter()


@router.post("/", response_model=DatasetResponse, tags=["Datasets"])
async def create_dataset(
    name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    output_format: str = Form("json-ner"),
    db: Session = Depends(get_db),
):
    """
    Crée un jeu de données NER à partir de fichiers téléchargés.
    """
    logger.info(f"Creating dataset with name: {name}")
    try:
        dataset_service = DatasetService(db=db)
        dataset = await dataset_service.create_dataset(
            name=name, files=files, labels=labels, output_format=output_format
        )
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/search.py
#-----
from fastapi import APIRouter, HTTPException, Query
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from dependencies import get_neo4j_service
from models.entity import Entity
from models.relationship import Relationship

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.get("/search/entities/", response_model=List[Entity])
async def search_entities(keyword: str = Query(..., description="Keyword to search for")):
    logger.info(f"Searching entities with keyword: {keyword}")
    try:
        entities = neo4j_service.search_entities(keyword)
        logger.info(f"Found {len(entities)} entities matching keyword: {keyword}")
        return entities
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/relationships/", response_model=List[Relationship])
async def search_relationships(keyword: str = Query(..., description="Keyword to search for")):
    logger.info(f"Searching relationships with keyword: {keyword}")
    try:
        relationships = neo4j_service.search_relationships(keyword)
        logger.info(f"Found {len(relationships)} relationships matching keyword: {keyword}")
        return relationships
    except Exception as e:
        logger.error(f"Error searching relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# services/neo4j_service.py
#-----
from neo4j import GraphDatabase, Transaction
from loguru import logger
from typing import List, Dict, Any, Optional


class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the connection to the Neo4j database.
        
        Args:
            uri (str): URI of the Neo4j database.
            user (str): Username for authentication.
            password (str): Password for authentication.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the connection to the Neo4j database."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def index_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        """
        Index nodes and edges into the Neo4j database.
        """
        with self.driver.session() as session:
            if nodes:
                session.write_transaction(self._index_nodes, nodes)
            if edges:
                session.write_transaction(self._index_edges, edges)

    @staticmethod
    def _index_nodes(tx: Transaction, nodes: List[Dict[str, Any]]):
        """
        Helper function to index nodes into Neo4j.

        Args:
            tx (Transaction): Neo4j transaction object.
            nodes (List[Dict[str, Any]]): List of nodes to index.
        """
        for node in nodes:
            try:
                query = """
                MERGE (n {id: $id})
                ON CREATE SET n += $properties
                """
                tx.run(query, id=node["id"], properties=node.get("properties", {}))
                logger.info(f"Node {node['id']} indexed successfully.")
            except Exception as e:
                logger.error(f"Failed to index node {node['id']}: {e}")

    @staticmethod
    def _index_edges(tx: Transaction, edges: List[Dict[str, Any]]):
        """
        Helper function to index relationships into Neo4j.

        Args:
            tx (Transaction): Neo4j transaction object.
            edges (List[Dict[str, Any]]): List of relationships to index.
        """
        for edge in edges:
            try:
                query = """
                MATCH (a {id: $source}), (b {id: $target})
                MERGE (a)-[r:$type]->(b)
                SET r += $properties
                """
                tx.run(
                    query,
                    source=edge["source"],
                    target=edge["target"],
                    type=edge["type"],
                    properties=edge.get("properties", {}),
                )
                logger.info(f"Edge from {edge['source']} to {edge['target']} indexed successfully.")
            except Exception as e:
                logger.error(f"Failed to index edge from {edge['source']} to {edge['target']}: {e}")

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes from the Neo4j database.

        Returns:
            List[Dict[str, Any]]: List of all nodes with their properties.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_entities_transaction)

    @staticmethod
    def _get_all_entities_transaction(tx: Transaction) -> List[Dict[str, Any]]:
        """
        Helper function to retrieve all nodes.

        Args:
            tx (Transaction): Neo4j transaction object.

        Returns:
            List[Dict[str, Any]]: List of all nodes.
        """
        query = """
        MATCH (e)
        RETURN id(e) AS id, labels(e) AS labels, properties(e) AS properties
        """
        try:
            result = tx.run(query)
            entities = [{"id": record["id"], "labels": record["labels"], "properties": record["properties"]} for record in result]
            return entities
        except Exception as e:
            logger.error(f"Failed to retrieve entities: {e}")
            return []

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Retrieve all relationships from the Neo4j database.

        Returns:
            List[Dict[str, Any]]: List of all relationships with their properties.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_relationships_transaction)

    @staticmethod
    def _get_all_relationships_transaction(tx: Transaction) -> List[Dict[str, Any]]:
        """
        Helper function to retrieve all relationships.

        Args:
            tx (Transaction): Neo4j transaction object.

        Returns:
            List[Dict[str, Any]]: List of all relationships.
        """
        query = """
        MATCH ()-[r]->()
        RETURN id(r) AS id, type(r) AS type, startNode(r) AS source, endNode(r) AS target, properties(r) AS properties
        """
        try:
            result = tx.run(query)
            relationships = [
                {
                    "id": record["id"],
                    "type": record["type"],
                    "source": record["source"],
                    "target": record["target"],
                    "properties": record["properties"],
                }
                for record in result
            ]
            return relationships
        except Exception as e:
            logger.error(f"Failed to retrieve relationships: {e}")
            return []

    def generate_graph_visualization(self) -> dict:
        """
        Generate a visualization of the graph by retrieving all nodes and relationships.

        Returns:
            dict: Dictionary containing nodes and relationships.
        """
        with self.driver.session() as session:
            nodes = session.read_transaction(self._get_all_entities_transaction)
            relationships = session.read_transaction(self._get_all_relationships_transaction)
            return {"nodes": nodes, "relationships": relationships}

#-----

# services/dataset_service.py
#-----
import os
from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session
from models.dataset import Dataset
from schemas.dataset import DatasetResponse
from loguru import logger
import aiofiles
from pathlib import Path
from gliner import GLiNER
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dependencies import get_s3_service
from config import settings
from services.document_processor import DoclingPDFLoader
from fastapi.encoders import jsonable_encoder


class DatasetService:
    def __init__(self, db: Session):
        self.db = db
        # Chargement du modèle GLiNER
        model_name = settings.GLINER_MODEL_NAME
        self.gliner_model = GLiNER.from_pretrained(model_name).to(settings.DEVICE)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.TEXT_CHUNK_SIZE,
            chunk_overlap=settings.TEXT_CHUNK_OVERLAP,
        )
        self.s3_service = get_s3_service()

    async def create_dataset(
        self,
        name: Optional[str],
        files: List[UploadFile],
        labels: Optional[str],
        output_format: str,
    ) -> DatasetResponse:
        temp_files = []
        for file in files:
            temp_file = Path(f"/tmp/{file.filename}")
            async with aiofiles.open(temp_file, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)
            temp_files.append(temp_file)

        all_annotations = []
        for temp_file in temp_files:
            # Chargement du document
            loader = DoclingPDFLoader(file_path=str(temp_file))
            docs = list(loader.lazy_load())
            chunks = []
            for doc in docs:
                text = doc.page_content
                split_texts = self.text_splitter.split_text(text)
                chunks.extend(split_texts)

            # Traitement des chunks avec GLiNER
            results = self.gliner_model.predict(chunks)
            for text_chunk, entities in zip(chunks, results):
                all_annotations.append(
                    {
                        "text": text_chunk,
                        "entities": [
                            {
                                "start": ent["start"],
                                "end": ent["end"],
                                "label": ent["label"],
                                "text": text_chunk[ent["start"] : ent["end"]],
                            }
                            for ent in entities
                        ],
                    }
                )

        # Formatage des annotations dans le format souhaité
        if output_format.lower() == "json-ner":
            dataset_content = self.format_to_json_ner(all_annotations)
        elif output_format.lower() == "conllu":
            dataset_content = self.format_to_conllu(all_annotations)
        else:
            raise ValueError(f"Format de sortie non supporté: {output_format}")

        # Sauvegarde du fichier de jeu de données dans S3
        dataset_file_name = f"{name or 'dataset'}.{output_format}"
        dataset_file_path = Path(f"/tmp/{dataset_file_name}")
        async with aiofiles.open(dataset_file_path, "w", encoding="utf-8") as dataset_file:
            await dataset_file.write(dataset_content)

        s3_url = self.s3_service.upload_file(
            dataset_file_path, bucket_name=self.s3_service.output_bucket
        )

        # Enregistrement des métadonnées du jeu de données dans la base de données
        dataset = Dataset(
            name=name or "Unnamed Dataset",
            data={"s3_url": s3_url},
            output_format=output_format,
        )
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)

        logger.info(f"Dataset {dataset.name} created with ID {dataset.id}")

        # Nettoyage des fichiers temporaires
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        if dataset_file_path.exists():
            dataset_file_path.unlink()

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=str(dataset.created_at),
        )

    def format_to_json_ner(self, annotations: List[dict]) -> str:
        """
        Formate les annotations au format JSON NER.
        """
        import json

        return json.dumps(annotations, ensure_ascii=False, indent=2)

    def format_to_conllu(self, annotations: List[dict]) -> str:
        """
        Formate les annotations au format CoNLL-U.
        """
        conllu_lines = []
        for item in annotations:
            text = item["text"]
            tokens = text.split()
            entities = item.get("entities", [])
            bio_tags = ["O"] * len(tokens)

            for entity in entities:
                entity_text = entity["text"].split()
                for i in range(len(tokens)):
                    if tokens[i : i + len(entity_text)] == entity_text:
                        bio_tags[i] = f"B-{entity['label']}"
                        for j in range(1, len(entity_text)):
                            bio_tags[i + j] = f"I-{entity['label']}"
                        break

            for idx, (token, tag) in enumerate(zip(tokens, bio_tags), start=1):
                conllu_lines.append(
                    f"{idx}\t{token}\t_\t_\t_\t_\t_\t_\t_\t{tag}"
                )
            conllu_lines.append("")

        return "\n".join(conllu_lines)

#-----

# services/s3_service.py
#-----
# services/s3_service.py
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from loguru import logger
from typing import Optional
from pathlib import Path

class S3Service:
    def __init__(self, s3_client, endpoint_url: str, access_key: str, secret_key: str, region_name: Optional[str] = None, input_bucket: str = "input", output_bucket: str = "output", layouts_bucket: str = "layouts"):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.layouts_bucket = layouts_bucket

        logger.info(f"Connected to S3 at {endpoint_url}")

    def upload_file(self, file_path: Path, bucket_name: str, object_name: Optional[str] = None) -> Optional[str]:
        if object_name is None:
            object_name = file_path.name
        try:
            self.s3_client.upload_file(str(file_path), bucket_name, object_name)
            logger.info(f"File {file_path} uploaded to bucket {bucket_name} as {object_name}")
            return f"s3://{bucket_name}/{object_name}"
        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to upload file {file_path} to S3: {e}")
        return None

    def download_file(self, bucket_name: str, object_name: str, download_path: Path) -> bool:
        try:
            self.s3_client.download_file(bucket_name, object_name, str(download_path))
            logger.info(f"File {object_name} downloaded from bucket {bucket_name} to {download_path}")
            return True
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to download file {object_name} from S3: {e}")
        return False

    def create_bucket(self, bucket_name: str) -> bool:
        try:
            self.s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} created successfully.")
            return True
        except ClientError as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
        return False

    def list_buckets(self) -> Optional[list]:
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            logger.info(f"Buckets retrieved: {buckets}")
            return buckets
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e}")
        return None

    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_name)
            logger.info(f"File {object_name} deleted from bucket {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file {object_name} from bucket {bucket_name}: {e}")
        return False

    def file_exists(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=object_name)
            logger.info(f"File {object_name} exists in bucket {bucket_name}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"File {object_name} does not exist in bucket {bucket_name}")
            else:
                logger.error(f"Error checking existence of file {object_name} in bucket {bucket_name}: {e}")
        return False

#-----

# services/rag_service.py
#-----
# service/rag_service.py
import os
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint

class RAGChainService:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = self._initialize_llm()

        # Define the prompt
        self.prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
        )

    def _initialize_llm(self):
        HF_API_KEY = os.environ.get("HF_API_KEY")
        HF_LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        return HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL_ID,
            huggingfacehub_api_token=HF_API_KEY,
        )

    def format_docs(self, docs: Iterable[LCDocument]):
        """
        Format the documents for RAG.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self):
        """
        Build the RAG chain.
        """
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run_query(self, query: str):
        """
        Run the query through the RAG chain.
        """
        rag_chain = self.build_chain()
        return rag_chain.invoke(query)

#-----

# services/mlflow_service.py
#-----
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
from pathlib import Path
import os
import json
from typing import Dict, Any

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

    def log_params(self, params: Dict[str, Any]):
        try:
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params}")
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, Any]):
        try:
            mlflow.log_metrics(metrics)
            logger.info(f"Logged metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, file_path: str, artifact_path: str = None):
        try:
            mlflow.log_artifact(file_path, artifact_path)
            logger.info(f"Logged artifact: {file_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact to MLflow: {e}")

    def register_model(self, model_name: str, model_dir: Path):
        try:
            model_uri = f"{self.tracking_uri}/{model_dir}"
            self.client.create_registered_model(model_name)
            self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id
            )
            logger.info(f"Model {model_name} registered successfully.")
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")

    def get_model_version(self, model_name: str):
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            logger.info(f"Retrieved versions for model {model_name}: {versions}")
            return versions
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {e}")
            return None

    def download_model(self, model_name: str, version: str, download_dir: str):
        try:
            model_uri = f"models:/{model_name}/{version}"
            local_path = mlflow.pyfunc.load_model(model_uri).save(download_dir)
            logger.info(f"Model {model_name} version {version} downloaded successfully to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download model {model_name} version {version}: {e}")
            return None

    def list_registered_models(self):
        try:
            models = self.client.list_registered_models()
            logger.info(f"Retrieved registered models: {models}")
            return models
        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []

    def set_tracking_uri(self, uri: str):
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI updated to: {uri}")

#-----

# services/pgvector_service.py
#-----
# services/pgector_services.py
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any, List, Optional
from loguru import logger
from langchain.docstore.document import Document


class PGVectorService:
    def __init__(self, db_url: str, table_name: str = "document_vectors"):
        """
        Initialize PGVectorService with a PostgreSQL connection and table name.

        Args:
            db_url (str): Database connection string.
            table_name (str): Name of the table to store and query vectors.
        """
        self.db_url = db_url
        self.table_name = table_name
        self.connection = self._connect_to_db()
        self.cursor = self.connection.cursor()
        self._ensure_table_exists()

    def _connect_to_db(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            connection = psycopg2.connect(self.db_url)
            logger.info("Successfully connected to PostgreSQL database.")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _ensure_table_exists(self):
        """Ensure the required table exists in the database."""
        try:
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding VECTOR,
                    metadata JSONB,
                    content TEXT
                );
            """)
            self.connection.commit()
            logger.info(f"Table '{self.table_name}' ensured in database.")
        except Exception as e:
            logger.error(f"Failed to ensure table exists: {e}")
            raise

    def store_vector(self, embedding: List[float], metadata: Dict[str, Any], content: str) -> Optional[int]:
        """
        Store a vector in the database.

        Args:
            embedding (List[float]): Vector embedding.
            metadata (Dict[str, Any]): Metadata for the document.
            content (str): Document content.

        Returns:
            Optional[int]: Row ID of the stored vector.
        """
        try:
            self.cursor.execute(
                f"""
                INSERT INTO {self.table_name} (embedding, metadata, content)
                VALUES (%s, %s, %s) RETURNING id;
                """,
                (embedding, Json(metadata), content)
            )
            row_id = self.cursor.fetchone()[0]
            self.connection.commit()
            logger.info(f"Vector stored with ID {row_id}.")
            return row_id
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
            self.connection.rollback()
            return None

    def index_documents(self, documents: List[Document]):
        """
        Index multiple document chunks into the PGVector table.

        Args:
            documents (List[Document]): List of document chunks to index.
        """
        try:
            for document in documents:
                # Extract embedding, metadata, and content
                embedding = document.metadata.get("embedding", [])
                metadata = {key: value for key, value in document.metadata.items() if key != "embedding"}
                content = document.page_content

                # Validate embedding
                if not embedding:
                    logger.warning("No embedding found for document chunk; skipping.")
                    continue

                # Store the vector in the database
                self.store_vector(embedding, metadata, content)
            logger.info(f"Indexed {len(documents)} document chunks into PGVector.")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    def search_vector(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the nearest vectors.

        Args:
            query_vector (List[float]): Query vector.
            k (int): Number of nearest neighbors to return.

        Returns:
            List[Dict[str, Any]]: List of results with metadata and distances.
        """
        try:
            self.cursor.execute(
                f"""
                SELECT id, content, metadata, embedding <=> %s AS distance
                FROM {self.table_name}
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (query_vector, k)
            )
            results = self.cursor.fetchall()
            logger.info(f"Found {len(results)} nearest vectors.")
            return [
                {"id": row[0], "content": row[1], "metadata": row[2], "distance": row[3]}
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error searching vector: {e}")
            return []

    def close(self):
        """Close the database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed.")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

#-----

# services/train_service.py
#-----
# services/train_service.py
from sqlalchemy.orm import Session
from schemas.train import TrainRequest, TrainResponse
from models.training_run import TrainingRun
from models.dataset import Dataset
from loguru import logger
from ml_models.ner_model import NERModel

class TrainService:
    def __init__(self, db: Session):
        self.db = db

    def train_model(self, request: TrainRequest) -> TrainResponse:
        # Retrieve dataset
        dataset = self.db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise ValueError("Dataset not found.")

        # Initialize NERModel and train
        ner_model = NERModel()
        ner_model.train(
            train_data=dataset.train_data,
            eval_data=dataset.eval_data,
            epochs=request.epochs,
            batch_size=request.batch_size
        )

        # Log training run
        training_run = TrainingRun(
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status="Completed"
        )
        self.db.add(training_run)
        self.db.commit()
        self.db.refresh(training_run)

        logger.info(f"Training completed for dataset {request.dataset_id}")

        return TrainResponse(
            id=training_run.id,
            run_id=str(training_run.run_id),
            dataset_id=training_run.dataset_id,
            epochs=training_run.epochs,
            batch_size=training_run.batch_size,
            status=training_run.status,
            created_at=str(training_run.created_at)
        )

#-----

# services/model_manager.py
#-----
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

#-----

# services/document_processor.py
#-----
import os
import re
import json
import yaml
from pathlib import Path
from typing import List, Iterator

from sqlalchemy.orm import Session
from models.document_log import DocumentLog

from loguru import logger
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from py2neo import Relationship, Node, Graph

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import PictureItem

from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.pgvector_service import PGVectorService
from services.neo4j_service import Neo4jService
from services.embedding_service import EmbeddingService

class CustomPdfPipelineOptions(PdfPipelineOptions):
    """Custom pipeline options for PDF processing."""
    do_picture_classifier: bool = False


class DoclingPDFLoader(BaseLoader):
    """Loader for converting PDFs to LCDocument format using Docling."""

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)


class DocumentProcessor:
    """Orchestrates the document processing pipeline: splitting, exporting, and indexing."""

    def __init__(
        self,
        s3_service: S3Service,
        mlflow_service: MLFlowService,
        pgvector_service: PGVectorService,
        neo4j_service: Neo4jService,
        embedding_service: EmbeddingService,
        session: Session,
        text_splitter: CharacterTextSplitter,
        graph_transformer: GlinerGraphTransformer,
        gliner_extractor: GLiNERLinkExtractor,
    ):
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.pgvector_service = pgvector_service
        self.neo4j_service = neo4j_service
        self.embedding_service = embedding_service
        self.session = session
        self.text_splitter = text_splitter
        self.graph_transformer = graph_transformer
        self.gliner_extractor = gliner_extractor

    def create_converter(self, use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool) -> DocumentConverter:
        """Create and configure a document converter."""
        options = CustomPdfPipelineOptions()
        options.do_ocr = use_ocr
        options.generate_page_images = True
        options.generate_table_images = export_tables
        options.generate_picture_images = export_figures
        options.do_picture_classifier = enrich_figures

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)}
        )

    def clean_text(self, text: str) -> str:
        """Clean up text by removing unwanted characters and normalizing whitespace."""
        text = text.replace("\n", " ").strip()
        return re.sub(r'\s+', ' ', text)

    def log_document(self, file_name: str, s3_url: str):
        """Log document metadata into the database."""
        try:
            log = DocumentLog(file_name=file_name, s3_url=s3_url)
            self.session.add(log)
            self.session.commit()
            logger.info(f"Document logged: {file_name}")
        except Exception as e:
            logger.error(f"Failed to log document {file_name}: {e}")
            self.session.rollback()
            raise

    def export_document(self, result: ConversionResult, output_dir: Path, export_formats: List[str], export_figures: bool, export_tables: bool):
        """Export document into specified formats and upload to S3."""
        try:
            doc_filename = result.input.file.stem
            if result.status == ConversionStatus.SUCCESS:
                self._export_file(result, output_dir, export_formats, export_figures, export_tables, doc_filename)
                logger.info(f"Document exported successfully: {doc_filename}")
            else:
                logger.warning(f"Document export failed for {doc_filename}: {result.status}")
        except Exception as e:
            logger.error(f"Error exporting document: {e}")
            raise

    def _export_file(self, result, output_dir, export_formats, export_figures, export_tables, doc_filename):
        """Save and upload the exported document files."""
        for ext in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, ext, export_format=ext)

        if export_figures:
            self._export_images(result, output_dir / "figures", doc_filename, self.s3_service.layouts_bucket)
        if export_tables:
            self._export_tables(result, output_dir / "tables", doc_filename, self.s3_service.layouts_bucket)

    def _save_and_upload(self, result, output_dir, doc_filename, ext, export_format="json"):
        """Save a specific document format locally and upload it to S3."""
        file_path = output_dir / f"{doc_filename}.{ext}"
        with file_path.open("w", encoding="utf-8") as file:
            if export_format == "json":
                json.dump(result.document.export_to_dict(), file, ensure_ascii=False, indent=2)
            elif export_format == "yaml":
                yaml.dump(result.document.export_to_dict(), file, allow_unicode=True)
            elif export_format == "md":
                file.write(result.document.export_to_markdown())
        self.s3_service.upload_file(file_path, self.s3_service.output_bucket)

    def _export_images(self, result, figures_dir, doc_filename, bucket):
        """Export and upload document images."""
        figures_dir.mkdir(exist_ok=True)
        for idx, element in enumerate(result.document.iterate_items()):
            if isinstance(element, PictureItem):
                image_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                element.image.pil_image.save(image_path, format="PNG")
                self.s3_service.upload_file(image_path, bucket)

    def _export_tables(self, result, tables_dir, doc_filename, bucket):
        """Export and upload document tables."""
        tables_dir.mkdir(exist_ok=True)
        for idx, table in enumerate(result.document.tables):
            csv_path = tables_dir / f"{doc_filename}_table_{idx + 1}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False, encoding="utf-8")
            self.s3_service.upload_file(csv_path, bucket)

    def process_and_index_document(self, file_path: str):
        """Process a document: extract embeddings, store them in PGVector, and index them in Neo4j."""
        try:
            logger.info(f"Loading document from {file_path}")
            loader = DoclingPDFLoader(file_path=file_path)
            docs = list(loader.lazy_load())
            if not docs:
                raise ValueError("No valid documents found.")

            logger.info(f"Loaded {len(docs)} document(s) from {file_path}")

            logger.info("Splitting document into chunks...")
            split_docs = self.text_splitter.split_documents(docs)
            split_docs = [
                LCDocument(page_content=self.clean_text(chunk.page_content), metadata=chunk.metadata)
                for chunk in split_docs
            ]
            logger.info(f"Document split into {len(split_docs)} chunks.")

            # Step 1: Generate embeddings and store them in PGVector
            for doc in split_docs:
                embedding = self.embedding_service.generate_embedding(doc.page_content)
                if embedding:
                    self.pgvector_service.store_vector(
                        embedding=embedding, metadata=doc.metadata, content=doc.page_content
                    )

            logger.info(f"Indexed {len(split_docs)} chunks into PGVector.")

            # Step 2: Transform chunks into a graph structure
            logger.info("Transforming document chunks into graph structure...")
            graph_docs = self.graph_transformer.convert_to_graph_documents(split_docs)
            doc_links = [self.gliner_extractor.extract_one(chunk) for chunk in split_docs]

            # Step 3: Index nodes, edges, and links into Neo4j
            with self.neo4j_service.driver.session() as session:
                with session.begin_transaction() as tx:
                    for graph_doc, links in zip(graph_docs, doc_links):
                        # Add nodes
                        if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                            for node in graph_doc.nodes:
                                tx.run(
                                    """
                                    MERGE (e:Entity {id: $id, name: $name, type: $type})
                                    ON CREATE SET e.created_at = timestamp()
                                    """,
                                    {
                                        "id": node.id,
                                        "name": node.properties.get("name", ""),
                                        "type": node.type,
                                    },
                                )
                                logger.info(f"Indexed Node: {node.id}, Type: {node.type}")

                        # Add relationships
                        if hasattr(graph_doc, "edges") and graph_doc.edges:
                            self.add_relationships(tx, graph_doc.edges)

                        # Add links
                        for link in links:
                            if not link.tag or not link.kind:
                                logger.warning(f"Skipping invalid link: {link}")
                                continue
                            logger.info(f"Adding Link: {link}")
                            tx.run(
                                """
                                MERGE (e:Entity {name: $name})
                                ON CREATE SET e.created_at = timestamp()
                                RETURN e
                                """,
                                {"name": link.tag},
                            )

            logger.info("Successfully processed and indexed document.")

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise

    def add_relationships(self, tx, relationships: List[Relationship]):
        """Add relationships to Neo4j."""
        for rel in relationships:
            try:
                if not rel.source or not rel.target or not rel.type:
                    logger.warning(f"Skipping invalid relationship: {rel}")
                    continue
                tx.run(
                    """
                    MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
                    MERGE (source)-[r:$type {properties: $properties}]->(target)
                    ON CREATE SET r.created_at = timestamp()
                    """,
                    {
                        "source_id": rel.source.id,
                        "target_id": rel.target.id,
                        "type": rel.type,
                        "properties": rel.properties or {},
                    },
                )
                logger.info(f"Added Relationship: {rel.type} from {rel.source.id} to {rel.target.id}")
            except Exception as e:
                logger.error(f"Failed to add relationship: {rel}. Error: {e}")
#-----

# services/embedding_service.py
#-----
# services/embedding_service.py
from langchain_ollama.embeddings import OllamaEmbeddings
from loguru import logger
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str):
        self.embedding_model = OllamaEmbeddings(model=model_name)
        logger.info(f"Embedding model '{model_name}' initialized.")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

    def generate_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.embedding_model.embed_query(text)
            logger.info(f"Generated embedding for the given text.")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    def embed_documents(self, texts):
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)
#-----

# ml_models/ner_model.py
#-----
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Dict

import GPUtil
import torch
from torch.utils.data import Dataset
from gliner import GLiNER
from loguru import logger
from tqdm import tqdm
from transformers import TrainingArguments, Trainer

from config import settings

try:
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore
    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass


class NERDataset(Dataset):
    """Custom Dataset for Named Entity Recognition."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class NERModel:
    """Named Entity Recognition model."""

    def __init__(
        self,
        name: str = "GLiNER-S",
        local_model_path: Optional[str] = None,
        overwrite: bool = False,
        train_config: dict = settings.train_config,
    ) -> None:
        """Initialize the NERModel."""
        if name not in settings.MODELS:  # Accéder via settings
            raise ValueError(f"Invalid model name: {name}")
        self.model_id: str = settings.MODELS[name]  # Accéder via settings

        # Create a directory for models
        workdir = Path.cwd() / "models"
        workdir.mkdir(parents=True, exist_ok=True)
        if local_model_path is None:
            local_model_path = name
        else:
            local_model_path = (workdir / local_model_path).resolve()
        if Path(local_model_path).exists() and not overwrite:
            raise ValueError(f"Model path already exists: {str(local_model_path)}")

        self.local_model_path: Path = Path(local_model_path)

        # Set device
        self.device: str = train_config.get("device", str(settings.DEVICE))
        logger.info(f"Device: [{self.device}]")

        # Define hyperparameters
        self.train_config: SimpleNamespace = SimpleNamespace(**train_config)

        # Initialize model as None for lazy loading
        self.model: Optional[GLiNER] = None

    def __load_model_remote(self) -> None:
        """Load the model from remote repository."""
        self.model = GLiNER.from_pretrained(self.model_id)

    def __load_model_local(self) -> None:
        """Load the model from a local path."""
        try:
            local_model_path = str(self.local_model_path.resolve())
            self.model = GLiNER.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        except Exception as e:
            logger.exception("Failed to load model from local path.", e)
            raise

    def load(self, mode: Literal["local", "remote", "auto"] = "auto") -> None:
        """Load the model."""
        if self.model is None:
            if mode == "local":
                self.__load_model_local()
            elif mode == "remote":
                self.__load_model_remote()
            elif mode == "auto":
                if self.local_model_path.exists():
                    self.__load_model_local()
                else:
                    self.__load_model_remote()
            else:
                raise ValueError(f"Invalid mode: {mode}")

            GPUtil.showUtilization()
            logger.info(
                f"Loaded model: [{self.model_id}] | N Params: [{self.model_param_count}] | [{self.model_size_in_mb}]"
            )
        else:
            logger.warning("Model already loaded.")

        logger.info(f"Moving model weights to: [{self.device}]")
        self.model = self.model.to(self.device)

    @property
    def model_size_in_bytes(self) -> int:
        """Returns the approximate size of the model parameters in bytes."""
        total_size = sum(param.numel() * param.element_size() for param in self.model.parameters())
        return total_size

    @property
    def model_param_count(self) -> str:
        """Returns the number of model parameters in billions."""
        return f"{sum(p.numel() for p in self.model.parameters()) / 1e9:,.2f} B"

    @property
    def model_size_in_mb(self) -> str:
        """Returns the string repr of the model parameter size in MB."""
        return f"{self.model_size_in_bytes / 1024**2:,.2f} MB"

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """Train the GLiNER model."""
        if self.model is None:
            self.load()

        GPUtil.showUtilization()

        # Prepare datasets
        train_dataset = NERDataset(train_data)
        eval_dataset = NERDataset(eval_data["samples"]) if eval_data else None

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.train_config.save_directory,
            num_train_epochs=self.train_config.num_steps,
            per_device_train_batch_size=self.train_config.train_batch_size,
            per_device_eval_batch_size=self.train_config.train_batch_size,
            learning_rate=self.train_config.lr_others,
            evaluation_strategy="steps",
            eval_steps=self.train_config.eval_every,
            logging_dir=f"{self.train_config.save_directory}/logs",
            logging_steps=10,
            save_steps=self.train_config.eval_every,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

    def batch_predict(
        self,
        targets: List[str],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.3,
        multi_label: bool = False,
        batch_size: int = 12,
    ) -> List[List[str]]:
        """Batch predict."""
        if self.model is None:
            self.load()

        self.model.eval()
        predictions = []
        for i, batch in enumerate(tqdm(self.chunk_list(targets, batch_size), desc="Predicting")):
            if i % 100 == 0:
                logger.debug(f"Predicting Batch [{i:,}]...")
            entities = self.model.batch_predict_entities(
                texts=batch,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
            predictions.extend(entities)
        return predictions

    def save(self, file_name: str) -> None:
        """Save the model to a file."""
        self.model.save_pretrained(file_name)

    @staticmethod
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Utility function to split a list into chunks."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

#-----

# ml_models/__init__.py
#-----

#-----
