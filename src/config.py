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
    PGVECTOR_TABLE_NAME: str = "documents_embeddings"
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"

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
    INPUT_BUCKET: str = "docs-input"
    OUTPUT_BUCKET: str = "docs-output"
    LAYOUTS_BUCKET: str = "layouts"
    
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