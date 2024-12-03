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
        "GLiNER-S": os.getenv("GLINER_S_MODEL", "urchade/gliner_smallv2.1"),
        "GLiNER-M": os.getenv("GLINER_M_MODEL", "urchade/gliner_mediumv2.1"),
        "GLiNER-L": os.getenv("GLINER_L_MODEL", "urchade/gliner_largev2.1"),
        "GLIREL": os.getenv("GLIREL_MODEL", "jackboyla/glirel-large-v0"),
        "GLiNER-Multitask": os.getenv("GLINER_MULTITASK_MODEL", "knowledgator/gliner-multitask-large-v0.5"),
    }

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8008))

    # Neo4j settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "your_password")

    # PostgreSQL settings
    PG_MAJOR: int = int(os.getenv("PG_MAJOR", 16))
    POSTGRE_PORT: int = int(os.getenv("POSTGRE_PORT", 5432))
    POSTGRE_USER: str = os.getenv("POSTGRE_USER", "postgre_user")
    POSTGRE_PASSWORD: str = os.getenv("POSTGRE_PASSWORD", "postgre_password")
    POSTGRE_DB: str = os.getenv("POSTGRE_DB", "postgre_db")
    POSTGRE_HOST: str = os.getenv("POSTGRE_HOST", "localhost")
    DJANGO_DB: str = "default"
    PGVECTOR_TABLE_NAME: str = "documents_embeddings"
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"

    # Derived PostgreSQL settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"postgresql://{POSTGRE_USER}:{POSTGRE_PASSWORD}@{POSTGRE_HOST}:{POSTGRE_PORT}/{POSTGRE_DB}")

    # MLflow settings
    MLFLOW_USER: str = os.getenv("MLFLOW_USER", "mlflow_user")
    MLFLOW_PASSWORD: str = os.getenv("MLFLOW_PASSWORD", "mlflow_password")
    MLFLOW_DB: str = os.getenv("MLFLOW_DB", "mlflow_db")
    MLFLOW_PORT: int = int(os.getenv("MLFLOW_PORT", 5002))
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    MLFLOW_S3_IGNORE_TLS: bool = os.getenv("MLFLOW_S3_IGNORE_TLS", "true").lower() == "true"
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")

    # Derived MLflow settings
    MLFLOW_BACKEND_STORE_URI: str = os.getenv(
        "MLFLOW_BACKEND_STORE_URI", 
        f"postgresql+psycopg2://{POSTGRE_USER}:{POSTGRE_PASSWORD}@{POSTGRE_HOST}:{POSTGRE_PORT}/{MLFLOW_DB}"
    )
    MLFLOW_ARTIFACT_ROOT: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow/")

    # MinIO settings
    MINIO_PORT: int = int(os.getenv("MINIO_PORT", 9000))
    MINIO_CONSOLE_PORT: int = int(os.getenv("MINIO_CONSOLE_PORT", 9001))
    MINIO_CLIENT_PORT: int = int(os.getenv("MINIO_CLIENT_PORT", 9002))
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minio")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minio123")
    MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "minio")
    MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
    MINIO_API_URL: str = os.getenv("MINIO_API_URL", "http://minio:9000")
    MINIO_URL: str = os.getenv("MINIO_URL", "http://localhost:9000")
    INPUT_BUCKET: str = os.getenv("INPUT_BUCKET", "docs-input")
    OUTPUT_BUCKET: str = os.getenv("OUTPUT_BUCKET", "docs-output")
    LAYOUTS_BUCKET: str = os.getenv("LAYOUTS_BUCKET", "layouts")
    
    # AWS settings for MinIO compatibility
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
    AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

    # Label Studio settings
    LABEL_STUDIO_USER: str = os.getenv("LABEL_STUDIO_USER", "labelstudio_user")
    LABEL_STUDIO_PASSWORD: str = os.getenv("LABEL_STUDIO_PASSWORD", "labelstudio_password")
    LABEL_STUDIO_DB: str = os.getenv("LABEL_STUDIO_DB", "labelstudio_db")
    LABEL_STUDIO_HOST: str = os.getenv("LABEL_STUDIO_HOST", "label-studio")
    LABEL_STUDIO_PORT: int = int(os.getenv("LABEL_STUDIO_PORT", 8081))
    LABEL_STUDIO_USERNAME: str = os.getenv("LABEL_STUDIO_USERNAME", "admin_user")
    LABEL_STUDIO_EMAIL: str = os.getenv("LABEL_STUDIO_EMAIL", "admin@example.com")
    LABEL_STUDIO_API_KEY: str = os.getenv("LABEL_STUDIO_API_KEY", "secure_api_key_123")
    LABEL_STUDIO_BUCKET_NAME: str = os.getenv("LABEL_STUDIO_BUCKET_NAME", "mlflow-source")
    LABEL_STUDIO_BUCKET_PREFIX: str = os.getenv("LABEL_STUDIO_BUCKET_PREFIX", "source_data/")
    LABEL_STUDIO_BUCKET_ENDPOINT_URL: str = os.getenv("LABEL_STUDIO_BUCKET_ENDPOINT_URL", "http://minio:9000")
    LABEL_STUDIO_BUCKET_ACCESS_KEY: str = os.getenv("LABEL_STUDIO_BUCKET_ACCESS_KEY", "minio")
    LABEL_STUDIO_BUCKET_SECRET_KEY: str = os.getenv("LABEL_STUDIO_BUCKET_SECRET_KEY", "minio123")
    LABEL_STUDIO_TARGET_BUCKET: str = os.getenv("LABEL_STUDIO_TARGET_BUCKET", "mlflow-annotations")
    LABEL_STUDIO_TARGET_PREFIX: str = os.getenv("LABEL_STUDIO_TARGET_PREFIX", "annotations/")
    LABEL_STUDIO_TARGET_ACCESS_KEY: str = os.getenv("LABEL_STUDIO_TARGET_ACCESS_KEY", "minio")
    LABEL_STUDIO_TARGET_SECRET_KEY: str = os.getenv("LABEL_STUDIO_TARGET_SECRET_KEY", "minio123")
    LABEL_STUDIO_TARGET_ENDPOINT_URL: str = os.getenv("LABEL_STUDIO_TARGET_ENDPOINT_URL", "http://minio:9000")
    LABEL_STUDIO_PROJECT_NAME: str = os.getenv("LABEL_STUDIO_PROJECT_NAME", "proj-1")
    LABEL_STUDIO_PROJECT_TITLE: str = os.getenv("LABEL_STUDIO_PROJECT_TITLE", "Machine Learning Annotations Project")
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED: bool = os.getenv("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", "true").lower() == "true"
    LS_DATABASE_URL: str = os.getenv("LS_DATABASE_URL", f"postgresql://{LABEL_STUDIO_USER}:{LABEL_STUDIO_PASSWORD}@{POSTGRE_HOST}:{POSTGRE_PORT}/{LABEL_STUDIO_DB}")

    # Prometheus settings
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", 9090))

    # GLiNER settings
    GLINER_BASIC_AUTH_USER: str = os.getenv("GLINER_BASIC_AUTH_USER", "my_user")
    GLINER_BASIC_AUTH_PASS: str = os.getenv("GLINER_BASIC_AUTH_PASS", "my_password")
    GLINER_MODEL_NAME: str = os.getenv("GLINER_MODEL_NAME", "knowledgator/gliner-multitask-large-v0.5")
    LABEL_STUDIO_ML_BACKENDS: str = os.getenv("LABEL_STUDIO_ML_BACKENDS", '[{"url": "http://gliner:9097", "name": "GLiNER"}]')
    GLIREL_MODEL_NAME: str = os.getenv("GLIREL_MODEL_NAME", "jackboyla/glirel-large-v0")

    # Secret Key
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super_secret_key_123")

    # General settings
    WORKERS: int = int(os.getenv("WORKERS", 4))
    THREADS: int = int(os.getenv("THREADS", 4))
    TEST_ENV: str = os.getenv("TEST_ENV", "my_test_env")
    LOCIP: str = os.getenv("LOCIP", "192.168.1.106")

    # ML Backend
    MLBACKEND_PORT: int = int(os.getenv("MLBACKEND_PORT", 9097))

    # Default Models
    DEFAULT_MODELS: str = os.getenv("DEFAULT_MODELS", "urchade/gliner_smallv2.1")

    # Training Configuration
    train_config: dict = {
        "num_steps": int(os.getenv("TRAIN_NUM_STEPS", 10_000)),
        "train_batch_size": int(os.getenv("TRAIN_BATCH_SIZE", 2)),
        "eval_every": int(os.getenv("TRAIN_EVAL_EVERY", 1_000)),
        "save_directory": os.getenv("TRAIN_SAVE_DIRECTORY", "checkpoints"),
        "warmup_ratio": float(os.getenv("TRAIN_WARMUP_RATIO", 0.1)),
        "device": os.getenv("TRAIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "lr_encoder": float(os.getenv("TRAIN_LR_ENCODER", 1e-5)),
        "lr_others": float(os.getenv("TRAIN_LR_OTHERS", 5e-5)),
        "freeze_token_rep": os.getenv("TRAIN_FREEZE_TOKEN_REP", "false").lower() == "true",
        "max_types": int(os.getenv("TRAIN_MAX_TYPES", 25)),
        "shuffle_types": os.getenv("TRAIN_SHUFFLE_TYPES", "true").lower() == "true",
        "random_drop": os.getenv("TRAIN_RANDOM_DROP", "true").lower() == "true",
        "max_neg_type_ratio": int(os.getenv("TRAIN_MAX_NEG_TYPE_RATIO", 1)),
        "max_len": int(os.getenv("TRAIN_MAX_LEN", 384)),
    }

    # Text splitting settings
    TEXT_CHUNK_SIZE: int = int(os.getenv("TEXT_CHUNK_SIZE", 1000))
    TEXT_CHUNK_OVERLAP: int = int(os.getenv("TEXT_CHUNK_OVERLAP", 200))
    CONF_FILE: str = os.getenv("CONF_FILE", "../conf/gli_config.yml")

    # Ollama
    OLLAMA_MODEL: str ="nomic-embed-text"

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

# Instanciation de la configuration
settings = Settings()