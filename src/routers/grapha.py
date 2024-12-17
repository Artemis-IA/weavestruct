# routers/graph.py
import boto3, os, re, mlflow, time, yaml
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List, Dict, Any, Optional, Literal, Union
from fastapi import Form, File, UploadFile

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
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
from src.config import settings

router = APIRouter()

neo4j_service: Neo4jService = get_neo4j_service()
s3_client : S3Service = get_s3_service()
metrics = get_metrics_manager()

bucket_name = settings.INPUT_BUCKET

# Endpoint to index documents
# @router.post("/index_nerrel/")
# @metrics.NEO4J_REQUEST_LATENCY.time()
# def index_pdfs(folder_path: Optional[str] = Form("/home/pi/Documents/IF-SRV/1pdf_subset")):
#     metrics.NEO4J_REQUEST_COUNT.inc()
#     if not os.path.isdir(folder_path):
#         raise HTTPException(status_code=400, detail="Invalid folder path")

#     loader = PyPDFDirectoryLoader(folder_path)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#     start_time = time.time()
#     for doc in documents:
#         if not hasattr(doc, "page_content"):
#             continue
#         split_docs = text_splitter.split_documents([doc])
#         graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
#         add_graph_to_neo4j(graph_docs)

#         file_name = doc.metadata.get("source", "unknown")
#         s3_url = upload_to_s3(file_name, bucket_name)
#         neo4j_url = f"neo4j://{neo4j_service.host}:{neo4j_service.port}"
#         if neo4j_url:
#             logger.info(f"Document {file_name} indexed successfully. S3 URL: {s3_url}, Neo4j URL: {neo4j_url}")
#     return {"message": "Documents indexed successfully"}



# @router.post("/index_nerrel/")
# @metrics.NEO4J_REQUEST_LATENCY.time()
# def index_pdfs(
#     folder_path: Optional[str] = Form(
#         None,
#         description="Path to a local folder containing PDFs. Leave empty if using S3. /home/pi/Documents/IF-SRV/1pdf_subset",
#         openapi_examples="/home/pi/Documents/IF-SRV/1pdf_subset"
#     ),
#     bucket_name: Optional[str] = Form(
#         None,
#         description="Name of the S3 bucket. Leave empty if using a local folder path. s3://docs-input/",
#         openapi_examples="s3://docs-input/"
#     ),
#     prefix: Optional[str] = Form(
#         "",
#         description="Prefix for objects in the S3 bucket (if applicable).",
#         openapi_examples="my/prefix/"
#     ),
# ):
#     """
#     Index PDFs from either a local folder or an S3 bucket into Neo4j.
#     Automatically determines the source type based on the input fields.
#     """
#     metrics.NEO4J_REQUEST_COUNT.inc()

#     # Determine source type
#     folder_path = folder_path.strip() if folder_path else None
#     bucket_name = bucket_name.strip() if bucket_name else None

#     if folder_path and bucket_name:
#         raise HTTPException(status_code=400, detail="Provide either folder_path or bucket_name, not both.")
#     if folder_path:
#         source_type = "local"
#     elif bucket_name:
#         source_type = "s3"
#     else:
#         raise HTTPException(status_code=400, detail="Either folder_path or bucket_name must be provided.")

#     # Process documents based on the source type
#     if source_type == "local":
#         if not os.path.isdir(folder_path):
#             raise HTTPException(status_code=400, detail="Invalid folder path for local source.")
#         loader = PyPDFDirectoryLoader(folder_path)
#         documents = loader.load()
#     elif source_type == "s3":
#         try:
#             objects = s3_client.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", [])
#             file_paths = [
#                 os.path.join("/tmp", obj["Key"].replace("/", "_"))
#                 for obj in objects if s3_client.download_file(bucket_name, obj["Key"], os.path.join("/tmp", obj["Key"].replace("/", "_")))
#             ]
#             loader = PyPDFDirectoryLoader("/tmp")
#             documents = loader.load()
#         except Exception as e:
#             logger.error(f"Failed to load documents from S3: {e}")
#             raise HTTPException(status_code=500, detail=f"Error loading documents from S3: {e}")

#     # Split and transform documents
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     for doc in documents:
#         if not hasattr(doc, "page_content"):
#             continue
#         split_docs = text_splitter.split_documents([doc])
#         graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
#         add_graph_to_neo4j(graph_docs)

#     return {"source_type": source_type, "message": "Documents indexed successfully."}
