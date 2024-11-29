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
        input_bucket=settings.INPUT_BUCKET,
        output_bucket=settings.OUTPUT_BUCKET,
        layouts_bucket=settings.LAYOUTS_BUCKET
    )

# Dependency to get the MLflow service
def get_mlflow_service() -> MLFlowService:
    """Returns an instance of the MLflow service."""
    return MLFlowService(tracking_uri=settings.MLFLOW_TRACKING_URI)

# Dependency for PGVector service
def get_pgvector_service() -> PGVectorService:
    return PGVectorService(
        db_url=settings.DATABASE_URL,
        table_name=settings.PGVECTOR_TABLE_NAME
    )

# Dependency for embedding service
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)


# Dependency for PGVector vector store
def get_pgvector_vector_store() -> PGVector:
    embedding_service = get_embedding_service()
    return PGVector(
        collection_name=settings.PGVECTOR_TABLE_NAME,
        connection=settings.DATABASE_URL,
        embeddings=embedding_service.embed_documents,
        use_jsonb=True
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

def get_gliner_link_extractor() -> GLiNERLinkExtractor:
    with open(settings.CONF_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return GLiNERLinkExtractor(
        labels=config.get("labels", []),
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
