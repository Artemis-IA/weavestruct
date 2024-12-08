# dependencies.py
import yaml
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Generator
from functools import lru_cache
from loguru import logger

from src.config import settings
from src.utils.database import DatabaseUtils
from src.utils.metrics import MetricsManager
from src.services.s3_service import S3Service
from src.services.mlflow_service import MLFlowService
from src.services.model_manager import ModelManager
from src.services.document_processor import DocumentProcessor
from src.services.pgvector_service import PGVectorService
from src.services.neo4j_service import Neo4jService
from src.services.rag_service import RAGChainService
from src.services.embedding_service import EmbeddingService
from src.services.gliner_service import GLiNERService
from src.services.glirel_service import GLiRELService

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_ollama.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase

# Dependency to get the SQLAlchemy session
def get_db() -> Generator[Session, None, None]:
    """Yields a database session."""
    yield from DatabaseUtils.get_db()

@lru_cache
def get_metrics_manager() -> MetricsManager:
    """Returns a singleton instance of MetricsManager."""
    return MetricsManager(prometheus_port=settings.PROMETHEUS_PORT)

# Dependency to get the S3 service
@lru_cache
def get_s3_service() -> S3Service:
    """Returns an instance of the S3 service and validates connection."""
    s3_service = S3Service(
        endpoint_url=settings.MINIO_URL,
        s3_client=None,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        input_bucket=settings.INPUT_BUCKET,
        output_bucket=settings.OUTPUT_BUCKET,
        layouts_bucket=settings.LAYOUTS_BUCKET
    )

    # Vérification et création des buckets si nécessaire
    required_buckets = [settings.INPUT_BUCKET, settings.OUTPUT_BUCKET, settings.LAYOUTS_BUCKET]
    existing_buckets = s3_service.list_buckets() or []

    for bucket in required_buckets:
        if bucket not in existing_buckets:
            try:
                s3_service.s3_client.create_bucket(Bucket=bucket)
                logger.info(f"Bucket '{bucket}' created successfully.")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create S3 bucket '{bucket}': {str(e)}"
                )

    return s3_service


# Dependency to get the MLflow service
@lru_cache
def get_mlflow_service() -> MLFlowService:
    try:
        mlflow_service = MLFlowService(
            tracking_uri=settings.MLFLOW_TRACKING_URI,
            s3_endpoint=settings.MINIO_API_URL,
            access_key=settings.AWS_ACCESS_KEY_ID,
            secret_key=settings.AWS_SECRET_ACCESS_KEY
        )
        mlflow_service.validate_connection()
        return mlflow_service
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MLflow service is not accessible: {e}"
        )
    
# Dependency to get the ModelManager
@lru_cache
def get_model_manager(
    mlflow_service: MLFlowService = Depends(get_mlflow_service),
) -> ModelManager:
    """Returns an instance of the ModelManager."""
    return ModelManager(mlflow_service=mlflow_service, tracking_uri=settings.MLFLOW_TRACKING_URI)


# Dependency for embedding service
@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Returns an instance of the Embedding service."""
    return EmbeddingService(artifact_name=settings.EMBEDDING_MODEL_NAME)


# Dependency for PGVector vector store
@lru_cache
def get_pgvector_vector_store() -> PGVectorService:
    """Returns an instance of the PGVectorService."""
    try:
        pgvector_service = PGVectorService(
            db_url=settings.DATABASE_URL,  # PostgreSQL connection string
            table_name=settings.PGVECTOR_TABLE_NAME  # Table to store vectors
        )
        pgvector_service.validate_connection()
        return pgvector_service
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PGVector service is not accessible: {str(e)}"
        )


# Dependency to get the Neo4j driver
@lru_cache
def get_neo4j_driver() -> GraphDatabase:
    """Returns a Neo4j driver instance."""
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

@lru_cache
def get_neo4j_service() -> Neo4jService:
    """Returns an instance of the Neo4jService."""
    try:
        neo4j_service = Neo4jService(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )
        neo4j_service.validate_connection()
        return neo4j_service
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Neo4j service is not accessible: {str(e)}"
        )


# Initialize reusable text splitter
@lru_cache
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Returns an instance of the text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.TEXT_CHUNK_SIZE,
        chunk_overlap=settings.TEXT_CHUNK_OVERLAP
    )

# Dependency to get the GLiNER extractor
@lru_cache
def get_gliner_extractor() -> GLiNERLinkExtractor:
    """Returns an instance of the GLiNERLinkExtractor."""
    with open(settings.CONF_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return GLiNERLinkExtractor(
        labels=config.get("labels", []),
        model=settings.GLINER_MODEL_NAME,
    )

# Dependency to get the GLiNER service
@lru_cache
def get_gliner_service() -> GLiNERService:
    """Returns an instance of the GLiNER service."""
    return GLiNERService()


# Dependency to get the GLiREL service
@lru_cache
def get_glirel_service() -> GLiRELService:
    """Returns an instance of the GLiREL service."""
    return GLiRELService()

# Dependency to get the Graph Transformer
@lru_cache
def get_graph_transformer() -> GlinerGraphTransformer:
    """Returns an instance of the GlinerGraphTransformer."""
    with open(settings.CONF_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return GlinerGraphTransformer(
        allowed_nodes=config.get("allowed_nodes", []),
        allowed_relationships=config.get("allowed_relationships", []),
        gliner_model=settings.GLINER_MODEL_NAME,
        glirel_model=settings.GLIREL_MODEL_NAME,
        entity_confidence_threshold=0.1,
        relationship_confidence_threshold=0.1,
    )

# Dependency to get the RAG service
@lru_cache
def get_rag_service() -> RAGChainService:
    """Returns an instance of the RAG Chain Service."""
    vector_store = get_pgvector_vector_store()
    return RAGChainService(retriever=vector_store.as_retriever())


# Updated Dependency to get the DocumentProcessor
@lru_cache
def get_document_processor(
    db: Session = Depends(get_db),
    gliner_service: GLiNERService = Depends(get_gliner_service),
    glirel_service: GLiRELService = Depends(get_glirel_service)
) -> DocumentProcessor:
    """Returns an instance of the DocumentProcessor with all dependencies."""
    s3_service = get_s3_service()
    mlflow_service = get_mlflow_service()
    pgvector_service = get_pgvector_vector_store()
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
        gliner_service=gliner_service,
        glirel_service=glirel_service,
        session=db,
        text_splitter=text_splitter,
        graph_transformer=graph_transformer,
        gliner_extractor=gliner_extractor,
    )
