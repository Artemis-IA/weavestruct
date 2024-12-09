# routers/graph.py
import boto3, os, re, mlflow, time, yaml
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List, Dict, Any, Optional
from fastapi import Form

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.document_log import DocumentLog, DocumentLogService
from src.services.neo4j_service import Neo4jService
from src.services.s3_service import S3Service
from src.utils.metrics import MetricsManager
from src.dependencies import get_neo4j_service, get_s3_service, get_metrics_manager
from src.models.entity import Entity
from src.models.relationship import Relationship

router = APIRouter()

neo4j_service: Neo4jService = get_neo4j_service()
s3_client : S3Service = get_s3_service()
metrics = get_metrics_manager()

bucket_name = 'docs-input'

# GLiNER extractor and transformer
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
gliner_extractor = GLiNERLinkExtractor(
    labels=config["labels"],
    model="E3-JSI/gliner-multi-pii-domains-v1"
)
graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="knowledgator/gliner-multitask-large-v0.5",
    glirel_model="jackboyla/glirel-large-v0",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
)

def upload_to_s3(file_name, bucket):
    try:
        s3_client.upload_file(file_name, bucket, os.path.basename(file_name))
        return f"s3://{bucket}/{os.path.basename(file_name)}"
    except Exception as e:
        logger.error(f"Failed to upload {file_name}: {e}")
        return None

def add_graph_to_neo4j(graph_docs):
    with neo4j_service.driver.session() as session:
        for doc in graph_docs:
            for node in doc.nodes:
                logger.info(f"Adding node: {node.id}, Type: {node.type}")
                session.run("MERGE (e:Entity {name: $name, type: $type})", {"name": node.id, "type": node.type})

            for edge in doc.relationships:
                logger.info(f"Adding relationship: {edge.type} between Source: {edge.source.id} and Target: {edge.target.id}")
                session.run(
                    "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                    "MERGE (source)-[:RELATED_TO {type: $type}]->(target)",
                    {"source": edge.source.id, "target": edge.target.id, "type": edge.type}
                )



# Endpoint to index documents
@router.post("/index_nerrel/")
@metrics.NEO4J_REQUEST_LATENCY.time()
def index_pdfs(folder_path: Optional[str] = Form("/home/pi/Documents/IF-SRV/1pdf_subset")):
    metrics.NEO4J_REQUEST_COUNT.inc()
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path")

    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    start_time = time.time()
    for doc in documents:
        if not hasattr(doc, "page_content"):
            continue
        split_docs = text_splitter.split_documents([doc])
        graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
        add_graph_to_neo4j(graph_docs)

        file_name = doc.metadata.get("source", "unknown")
        s3_url = upload_to_s3(file_name, bucket_name)
        if s3_url:
            logger.info(f"Document {file_name} uploaded to S3: {s3_url}")

    return {"message": "Documents indexed successfully"}


