
# neo4j_service.py
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


    def validate_connection(self):
        """
        Validate the connection to the Neo4j database by running a test query.
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                if result.single()[0] == 1:
                    logger.info("Neo4j connection validated successfully.")
        except Exception as e:
            logger.error(f"Failed to validate Neo4j connection: {e}")
            raise
#-----

# gliner_service.py
#-----
# services/gliner_service.py

import os
import yaml
import torch
from gliner import GLiNER
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from loguru import logger
from config import settings

class GLiNERService:
    """
    Service class for GLiNER model operations.
    """

    def __init__(self):
        # Load configurations
        config_file = settings.CONF_FILE
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize GLiNER model
        gliner_model_name = settings.GLINER_MODEL_NAME
        try:
            self.gliner_model = GLiNER.from_pretrained(gliner_model_name).to(self.device)
            logger.info(f"GLiNER model '{gliner_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            raise RuntimeError("GLiNER model loading failed.")

        # Initialize GlinerGraphTransformer
        self.graph_transformer = GlinerGraphTransformer(
            allowed_nodes=self.config["allowed_nodes"],
            allowed_relationships=self.config["allowed_relationships"],
            gliner_model=gliner_model_name,
            glirel_model=settings.GLIREL_MODEL_NAME,
            entity_confidence_threshold=0.1,
            relationship_confidence_threshold=0.1,
        )
        logger.info("GlinerGraphTransformer initialized.")

    def extract_entities(self, texts):
        """
        Extract entities from a list of texts using GlinerGraphTransformer.

        Args:
            texts (List[str]): List of texts to process.

        Returns:
            List[List[Dict]]: A list where each element corresponds to the entities extracted from a text.
        """
        try:
            # Convert texts to the format expected by GlinerGraphTransformer
            documents = [{"text": text} for text in texts]

            # Perform entity extraction
            graph_docs = self.graph_transformer.convert_to_graph_documents(documents)

            # Extract entities
            all_entities = []
            for graph_doc in graph_docs:
                entities = []
                if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                    for node in graph_doc.nodes:
                        entity = {
                            "text": node.properties.get("name", ""),
                            "label": node.type,
                            "start": node.properties.get("start", 0),
                            "end": node.properties.get("end", 0),
                        }
                        entities.append(entity)
                all_entities.append(entities)
            return all_entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            raise

    def predict_entities(self, text, labels=None, threshold=0.5, nested_ner=False):
        """
        Predict entities in a single text using GLiNER.

        Args:
            text (str): The text to analyze.
            labels (List[str], optional): List of labels to consider.
            threshold (float, optional): Confidence threshold.
            nested_ner (bool, optional): Whether to perform nested NER.

        Returns:
            Dict: A dictionary containing the text and the list of entities.
        """
        try:
            entities = self.gliner_model.predict_entities(
                text,
                labels=labels,
                flat_ner=not nested_ner,
                threshold=threshold,
            )
            return {
                "text": text,
                "entities": entities,
            }
        except Exception as e:
            logger.error(f"Error in predict_entities: {e}")
            raise

#-----

# dataset_service.py
#-----
# services/dataset_service.py

from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session
from models.dataset import Dataset
from schemas.dataset import DatasetResponse
from loguru import logger
import aiofiles
from pathlib import Path
from dependencies import (
    get_s3_service,
    get_mlflow_service,
    get_pgvector_vector_store,
    get_neo4j_service,
    get_embedding_service,
    get_text_splitter,
    get_graph_transformer,
)
from config import settings
from services.document_processor import DocumentProcessor
from services.gliner_service import GLiNERService
import json
from datetime import datetime

class DatasetService:
    """
    Service class for handling dataset creation and management.
    """

    def __init__(self, db: Session):
        self.db = db
        self.s3_service = get_s3_service()
        self.mlflow_service = get_mlflow_service()
        self.pgvector_service = get_pgvector_vector_store()
        self.neo4j_service = get_neo4j_service()
        self.embedding_service = get_embedding_service()
        self.session = db  # Use the passed-in database session
        self.text_splitter = get_text_splitter()
        self.graph_transformer = get_graph_transformer()
        self.gliner_extractor = GLiNERService()
        self.document_processor = DocumentProcessor(
            s3_service=self.s3_service,
            mlflow_service=self.mlflow_service,
            pgvector_service=self.pgvector_service,
            neo4j_service=self.neo4j_service,
            embedding_service=self.embedding_service,
            session=self.session,
            text_splitter=self.text_splitter,
            graph_transformer=self.graph_transformer,
            gliner_extractor=self.gliner_extractor,
        )
        logger.info("DatasetService initialized.")

    async def create_dataset(
        self,
        name: Optional[str],
        files: List[UploadFile],
        labels: Optional[str],
        output_format: str,
    ) -> DatasetResponse:
        temp_files = []
        try:
            # Save uploaded files temporarily
            for file in files:
                temp_file = Path(f"/tmp/{file.filename}")
                async with aiofiles.open(temp_file, "wb") as out_file:
                    content = await file.read()
                    await out_file.write(content)
                temp_files.append(temp_file)

            all_annotations = []
            for temp_file in temp_files:
                # Process document with DocumentProcessor to extract texts
                texts = self.document_processor.extract_texts(temp_file)
                cleaned_texts = [self.clean_text(text) for text in texts]

                if not cleaned_texts:
                    logger.warning(f"No text extracted from document: {temp_file}")
                    continue

                # Extract entities using GLiNERService
                logger.info(f"Extracting entities from {len(cleaned_texts)} texts...")
                try:
                    entities_list = self.gliner_extractor.extract_entities(cleaned_texts)
                except Exception as e:
                    logger.error(f"Error extracting entities: {e}")
                    raise ValueError(f"Failed to extract entities: {e}")

                # Format annotations
                for text, entities in zip(cleaned_texts, entities_list):
                    # Format entities to be compatible with GLiNER json-ner format
                    formatted_entities = [
                        {
                            "start": entity["start"],
                            "end": entity["end"],
                            "label": entity["label"],
                            "text": entity["text"],
                        }
                        for entity in entities
                    ]
                    all_annotations.append(
                        {
                            "text": text,
                            "entities": formatted_entities,
                        }
                    )

            # Convert to required format
            if output_format.lower() == "json-ner":
                dataset_content = json.dumps(all_annotations, ensure_ascii=False, indent=2)
            elif output_format.lower() == "conllu":
                dataset_content = self.format_to_conllu(all_annotations)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            # Save to S3
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            dataset_file_name = f"{name or 'dataset'}_{timestamp}.{output_format}"
            dataset_file_path = Path(f"/tmp/{dataset_file_name}")
            async with aiofiles.open(dataset_file_path, "w", encoding="utf-8") as dataset_file:
                await dataset_file.write(dataset_content)

            s3_url = self.s3_service.upload_file(
                dataset_file_path, bucket_name=settings.OUTPUT_BUCKET
            )

            # Save to database
            dataset = Dataset(
                name=name or "Unnamed Dataset",
                data={"s3_url": s3_url},
                output_format=output_format,
            )
            self.db.add(dataset)
            self.db.commit()
            self.db.refresh(dataset)

            logger.info(f"Dataset {dataset.name} created with ID {dataset.id}")

            return DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                data=dataset.data,
                created_at=str(dataset.created_at),
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise RuntimeError(f"Dataset creation failed: {e}")
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()


    def clean_text(self, text: str) -> str:
        """
        Cleans text by removing unnecessary whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        return " ".join(text.split())
    
    def forma_to_json_ner(self, annotations: List[dict]) -> str:
        """
        Formats annotations to JSON-NER format.

        Args:
            annotations (List[dict]): List of annotations.

        Returns:
            str: Annotations formatted in JSON-NER format.
        """
        return json.dumps(annotations, ensure_ascii=False, indent=2)
    

    def format_to_conllu(self, annotations: List[dict]) -> str:
        """
        Formats annotations to CoNLL-U format.

        Args:
            annotations (List[dict]): List of annotations.

        Returns:
            str: Annotations formatted in CoNLL-U format.
        """
        conllu_lines = []
        for item in annotations:
            text = item["text"]
            tokens = text.split()
            entities = item.get("entities", [])
            bio_tags = ["O"] * len(tokens)

            for entity in entities:
                entity_tokens = entity["text"].split()
                for i in range(len(tokens)):
                    if tokens[i : i + len(entity_tokens)] == entity_tokens:
                        bio_tags[i] = f"B-{entity['label']}"
                        for j in range(1, len(entity_tokens)):
                            bio_tags[i + j] = f"I-{entity['label']}"
                        break

            for idx, (token, tag) in enumerate(zip(tokens, bio_tags), start=1):
                conllu_lines.append(f"{idx}\t{token}\t_\t_\t_\t_\t_\t_\t_\t{tag}")
            conllu_lines.append("")

        return "\n".join(conllu_lines)

#-----

# s3_service.py
#-----
# services/s3_service.py
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from loguru import logger
from typing import Optional, Union, IO, Tuple
from pathlib import Path
from urllib.parse import urlparse
from config import settings

class S3Service:
    def __init__(self, s3_client, endpoint_url: str, access_key: str, secret_key: str, region_name: Optional[str] = None, input_bucket: str = "input", output_bucket: str = "output", layouts_bucket: str = "layouts"):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        self.input_bucket = settings.INPUT_BUCKET
        self.output_bucket = settings.OUTPUT_BUCKET
        self.layouts_bucket = settings.LAYOUTS_BUCKET

        logger.info(f"Connected to S3 at {endpoint_url}")

    def upload_file(self, file_path: Path, bucket_name: str, object_name: Optional[str] = None) -> Optional[str]:
        if object_name is None:
            object_name = file_path.name
        try:
            self.s3_client.upload_file(str(file_path), bucket_name, object_name)
            logger.info(f"File {file_path} uploaded to bucket {bucket_name} as {object_name}")
            return f"https://{bucket_name}/{object_name}"
        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to upload file {file_path} to S3: {e}")
        return None

    def upload_fileobj(self, file_obj: IO, bucket_name: str, object_name: str) -> Optional[str]:
        """
        Upload a file-like object directly to S3.
        """
        try:
            self.s3_client.upload_fileobj(file_obj, bucket_name, object_name)
            logger.info(f"File object uploaded to bucket {bucket_name} as {object_name}")
            return f"https://{bucket_name}/{object_name}"
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to upload file object to S3: {e}")
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
    
    def parse_s3_url(self, s3_url: str) -> Optional[Tuple[str, str]]:
        """
        Parse an S3 URL into bucket name and object key.
        """
        parsed = urlparse(s3_url)
        if parsed.scheme not in ['http', 'https']:
            logger.error(f"Invalid S3 URL scheme: {s3_url}")
            return None
        path_parts = parsed.path.lstrip('/').split('/', 1)
        if len(path_parts) != 2:
            logger.error(f"Invalid S3 URL path: {s3_url}")
            return None
        bucket_name, object_key = path_parts
        return bucket_name, object_key

    def get_s3_url(self, bucket_name: str, object_name: str) -> str:
        """
        Generate a public S3 URL for the object.
        """
        return f"{self.s3_client.meta.endpoint_url}/{bucket_name}/{object_name}"

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
#-----

# rag_service.py
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

# mlflow_service.py
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

    def validate_connection(self):
        """Validate connection to MLflow tracking server."""
        try:
            # Use search_experiments as a connectivity check
            experiments = self.client.search_experiments(max_results=1)
            logger.info(
                f"MLflow service is accessible. Found {len(experiments)} experiment(s)."
            )
        except AttributeError as e:
            # Provide hints for known issues
            if "search_experiments" in str(e):
                raise Exception(
                    f"MLflowClient object lacks 'search_experiments'. Possible version mismatch. {e}"
                )
            raise
        except Exception as e:
            raise Exception(f"Failed to connect to MLflow: {e}")
#-----

# pgvector_service.py
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


    def validate_connection(self):
        """Validate the connection to the PostgreSQL database."""
        try:
            self.cursor.execute("SELECT 1;")
            self.connection.commit()
            logger.info("PostgreSQL connection validated.")
        except Exception as e:
            logger.error(f"Failed to validate PostgreSQL connection: {e}")
            raise
#-----

# train_service.py
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

# model_manager.py
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

# document_processor.py
#-----
# services/document_processor.py
import os
import re
import json
import yaml
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Any
from enum import Enum

import aiofiles
from sqlalchemy.orm import Session
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
from docling.datamodel.base_models import InputFormat

from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.pgvector_service import PGVectorService
from services.neo4j_service import Neo4jService
from services.embedding_service import EmbeddingService
from services.gliner_service import GLiNERService
from services.glirel_service import GLiRELService

from models.document_log import DocumentLog
from models.document import Document
from utils.database import SessionLocal  # Assurez-vous que ce module est correctement défini

class CustomPdfPipelineOptions(PdfPipelineOptions):
    """Custom pipeline options for PDF processing."""
    do_picture_classifier: bool = False

class DoclingPDFLoader(BaseLoader):
    """Loader for converting PDFs to LCDocument format using Docling."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=CustomPdfPipelineOptions(),
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        try:
            conversion_result = self._converter.convert(self.file_path)
            if conversion_result.status == ConversionStatus.SUCCESS:
                dl_doc = conversion_result.document
                text = dl_doc.export_to_markdown()
                yield LCDocument(page_content=text)
            else:
                logger.warning(f"Conversion failed for {self.file_path} with status {conversion_result.status}")
        except Exception as e:
            logger.error(f"Error loading document {self.file_path}: {e}")

# Enumération des formats d'import et d'export
class ImportFormat(str, Enum):
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    IMAGE = "image"
    PDF = "pdf"
    ASCIIDOC = "asciidoc"
    MD = "md"

class ExportFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    MARKDOWN = "md"
    DOCTAGS = "doctags"

class DocumentProcessor:
    """Orchestrates the document processing pipeline: loading, splitting, embedding, extracting, and indexing."""

    def __init__(
        self,
        s3_service: S3Service,
        mlflow_service: MLFlowService,
        pgvector_service: PGVectorService,
        neo4j_service: Neo4jService,
        embedding_service: EmbeddingService,
        gliner_service: GLiNERService,
        glirel_service: GLiRELService,
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
        self.gliner_service = gliner_service
        self.glirel_service = glirel_service
        self.session = session
        self.text_splitter = text_splitter
        self.graph_transformer = graph_transformer
        self.gliner_extractor = gliner_extractor

    def create_converter(
        self,
        use_ocr: bool,
        export_figures: bool,
        export_tables: bool,
        enrich_figures: bool
    ) -> DocumentConverter:
        """Create and configure a document converter."""
        options = CustomPdfPipelineOptions()
        options.do_ocr = use_ocr
        options.generate_page_images = True
        options.generate_table_images = export_tables
        options.generate_picture_images = export_figures
        options.do_picture_classifier = enrich_figures

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.IMAGE, InputFormat.HTML],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)},
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

    def export_document(
        self,
        result: ConversionResult,
        export_formats: List[ExportFormat],
        export_figures: bool,
        export_tables: bool
    ):
        """Export document into specified formats and upload to S3."""
        try:
            doc_filename = Path(result.input.file).stem
            if result.status == ConversionStatus.SUCCESS:
                output_dir = Path("/tmp/exports")  # Utilisé temporairement pour le stockage local avant l'upload
                output_dir.mkdir(parents=True, exist_ok=True)
                self._export_file(
                    result=result,
                    output_dir=output_dir,
                    export_formats=export_formats,
                    export_figures=export_figures,
                    export_tables=export_tables,
                    doc_filename=doc_filename
                )
                logger.info(f"Document exported successfully: {doc_filename}")

                # Upload exported files to S3 output bucket
                for ext in export_formats:
                    file_path = output_dir / f"{doc_filename}.{ext}"
                    s3_output_key = f"output/{doc_filename}.{ext}"
                    self.s3_service.upload_file(file_path, bucket_name=self.s3_service.output_bucket)
                    logger.info(f"Uploaded {file_path} to {self.s3_service.output_bucket}/{s3_output_key}")

                if export_figures:
                    figures_dir = output_dir / "figures"
                    for figure_file in figures_dir.glob("*.png"):
                        s3_figures_key = f"layouts/figures/{figure_file.name}"
                        self.s3_service.upload_file(figure_file, bucket_name=self.s3_service.layouts_bucket)
                        logger.info(f"Uploaded {figure_file} to {self.s3_service.layouts_bucket}/{s3_figures_key}")

                if export_tables:
                    tables_dir = output_dir / "tables"
                    for table_file in tables_dir.glob("*.csv"):
                        s3_tables_key = f"layouts/tables/{table_file.name}"
                        self.s3_service.upload_file(table_file, bucket_name=self.s3_service.layouts_bucket)
                        logger.info(f"Uploaded {table_file} to {self.s3_service.layouts_bucket}/{s3_tables_key}")

                # Log document metadata
                self.log_document(file_name=result.input.file, s3_url=s3_output_key)

                # Optionnel: Supprimer les fichiers exportés localement après l'upload
                for ext in export_formats:
                    file_path = output_dir / f"{doc_filename}.{ext}"
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Deleted local file: {file_path}")

                if export_figures:
                    for figure_file in figures_dir.glob("*.png"):
                        figure_file.unlink()
                        logger.info(f"Deleted local figure file: {figure_file}")

                if export_tables:
                    for table_file in tables_dir.glob("*.csv"):
                        table_file.unlink()
                        logger.info(f"Deleted local table file: {table_file}")

            else:
                logger.warning(f"Document export failed for {doc_filename}: {result.status}")
        except Exception as e:
            logger.error(f"Error exporting document: {e}")
            raise

    def _export_file(
        self,
        result: ConversionResult,
        output_dir: Path,
        export_formats: List[ExportFormat],
        export_figures: bool,
        export_tables: bool,
        doc_filename: str
    ):
        """Save a specific document format locally."""
        for ext in export_formats:
            file_path = output_dir / f"{doc_filename}.{ext}"
            try:
                with file_path.open("w", encoding="utf-8") as file:
                    if ext == "json":
                        json.dump(result.document.export_to_dict(), file, ensure_ascii=False, indent=2)
                    elif ext == "yaml":
                        yaml.dump(result.document.export_to_dict(), file, allow_unicode=True)
                    elif ext == "md":
                        file.write(result.document.export_to_markdown())
                    else:
                        logger.warning(f"Unsupported export format: {ext}")
                        continue
                logger.info(f"Exported file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to export file {file_path}: {e}")
                raise

    def process_documents(
        self,
        s3_url: str,
        export_formats: List[ExportFormat],
        use_ocr: bool = False,
        export_figures: bool = True,
        export_tables: bool = True,
        enrich_figures: bool = False,
    ) -> List[LCDocument]:
        """
        Process a document: convert it using Docling, generate embeddings,
        and export results to S3.

        Args:
            s3_url (str): S3 URL of the input document.
            export_formats (List[str]): Formats for exporting the processed document.
            use_ocr (bool): Whether to use OCR during processing.
            export_figures (bool): Whether to export figures.
            export_tables (bool): Whether to export tables.
            enrich_figures (bool): Whether to enrich figures.

        Returns:
            List[LCDocument]: List of processed document chunks.
        """
        logger.info(f"Starting processing for document: {s3_url}")

        # Step 1: Parse S3 URL to get bucket and object key
        parsed = self.s3_service.parse_s3_url(s3_url)
        if not parsed:
            logger.error(f"Failed to parse S3 URL: {s3_url}")
            return []
        bucket_name, object_key = parsed

        # Step 2: Download the file from S3
        local_path = Path("/tmp") / Path(object_key).name
        if not self.s3_service.download_file(bucket_name, object_key, local_path):
            logger.error(f"Failed to download file from S3: {s3_url}")
            return []

        logger.info(f"Downloaded document {s3_url} to {local_path}")

        # Step 3: Process the document using Docling
        converter = self.create_converter(use_ocr, export_figures, export_tables, enrich_figures)
        conversion_result = converter.convert(str(local_path))

        if conversion_result.status != ConversionStatus.SUCCESS:
            logger.error(f"Failed to process document {local_path}: {conversion_result.status}")
            return []

        logger.info(f"Document {local_path.name} converted successfully")

        # Step 4: Export processed document results to S3
        output_dir = Path("/tmp/exports")
        output_dir.mkdir(parents=True, exist_ok=True)

        for fmt in export_formats:
            export_path = output_dir / f"{local_path.stem}.{fmt}"
            with export_path.open("w", encoding="utf-8") as f:
                if fmt == "json":
                    json.dump(conversion_result.document.export_to_dict(), f, indent=2)
                elif fmt == "yaml":
                    yaml.dump(conversion_result.document.export_to_dict(), f)
                elif fmt == "md":
                    f.write(conversion_result.document.export_to_markdown())

            # Upload the exported file to the `output` bucket
            s3_output_key = f"output/{local_path.stem}.{fmt}"
            upload_url = self.s3_service.upload_file(export_path, bucket_name=self.s3_service.output_bucket)
            if upload_url:
                logger.info(f"Uploaded {export_path} to {self.s3_service.output_bucket}/{s3_output_key}")
            else:
                logger.error(f"Failed to upload {export_path} to S3")

        # Optionnel: Supprimer les fichiers exportés localement après l'upload
        for fmt in export_formats:
            file_path = output_dir / f"{local_path.stem}.{fmt}"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted local file: {file_path}")

        if export_figures:
            figures_dir = output_dir / "figures"
            if figures_dir.exists():
                for figure_file in figures_dir.glob("*.png"):
                    figure_file.unlink()
                    logger.info(f"Deleted local figure file: {figure_file}")

        if export_tables:
            tables_dir = output_dir / "tables"
            if tables_dir.exists():
                for table_file in tables_dir.glob("*.csv"):
                    table_file.unlink()
                    logger.info(f"Deleted local table file: {table_file}")

        # Clean up the downloaded file
        if local_path.exists():
            local_path.unlink()
            logger.info(f"Deleted local file: {local_path}")

        # Log document metadata
        self.log_document(file_name=local_path.name, s3_url=s3_output_key)

        # Log metrics to MLFlow
        metrics = {
            "documents_loaded": 1,
            "chunks_processed": 1,
            "embeddings_generated": 0,  # À ajuster selon vos besoins
        }
        self.mlflow_service.log_metrics(metrics)
        logger.info("Metrics logged to MLFlow")

        return []  # Retournez la liste des documents traités si nécessaire

    def process_and_index_document(self, s3_url: str):
        """
        Process a document and index the results into Neo4j: load, split, generate embeddings, extract entities and relationships,
        store embeddings, and index graph data in Neo4j.

        Args:
            s3_url (str): S3 URL of the document file.
        """
        try:
            logger.info(f"Starting full processing and indexing for document: {s3_url}")
            self.mlflow_service.start_run(run_name=f"Processing and Indexing {Path(s3_url).stem}")

            # Step 1: Process documents without indexing
            split_docs = self.process_documents(
                s3_url=s3_url,
                export_formats=["json", "yaml", "md"],
                use_ocr=True,  # Ajustez selon vos besoins
                export_figures=True,
                export_tables=True,
                enrich_figures=False  # Ajustez selon vos besoins
            )

            if not split_docs:
                logger.warning(f"No chunks to index for document: {s3_url}")
                return

            # Step 2: Extract entities using GLiNER
            logger.info("Extracting entities using GLiNER...")
            entities = self.gliner_service.extract_entities([doc.page_content for doc in split_docs])
            logger.info("Entities extraction completed")

            # Step 3: Extract relationships using GLiREL
            logger.info("Extracting relationships using GLiREL...")
            relationships = self.glirel_service.extract_relationships([doc.page_content for doc in split_docs])
            logger.info("Relationships extraction completed")

            # Step 4: Transform documents into graph structure
            logger.info("Transforming documents into graph structure...")
            graph_docs = self.graph_transformer.convert_to_graph_documents(split_docs)
            logger.info("Graph transformation completed")

            # Step 5: Prepare nodes and edges for Neo4j
            logger.info("Preparing nodes and edges for Neo4j indexing...")
            nodes = []
            edges = []
            for doc_idx, graph_doc in enumerate(graph_docs):
                # Add entities as nodes
                for entity in entities[doc_idx]:
                    node_id = f"entity_{doc_idx}_{entity['start']}_{entity['end']}"
                    node = {
                        "id": node_id,
                        "properties": {
                            "text": entity["text"],
                            "label": entity["label"],
                            "start": entity["start"],
                            "end": entity["end"],
                        }
                    }
                    nodes.append(node)

                # Add relationships as edges
                for rel in relationships[doc_idx]:
                    edge = {
                        "source": f"entity_{doc_idx}_{rel['source']['start']}_{rel['source']['end']}",
                        "target": f"entity_{doc_idx}_{rel['target']['start']}_{rel['target']['end']}",
                        "type": rel["type"],
                        "properties": rel.get("properties", {})
                    }
                    edges.append(edge)

            logger.info(f"Prepared {len(nodes)} nodes and {len(edges)} edges for Neo4j")

            # Step 6: Index nodes and edges in Neo4j
            logger.info("Indexing nodes and edges in Neo4j...")
            self.neo4j_service.index_graph(nodes=nodes, edges=edges)
            logger.info("Neo4j indexing completed")

            # Step 7: Log metrics to MLFlow
            metrics = {
                "documents_loaded": len(split_docs),
                "chunks_processed": len(split_docs),
                "embeddings_generated": 0,  # Ajustez selon vos besoins
                "entities_extracted": sum(len(e) for e in entities),
                "relationships_extracted": sum(len(r) for r in relationships),
                "nodes_indexed": len(nodes),
                "edges_indexed": len(edges),
            }
            self.mlflow_service.log_metrics(metrics)
            logger.info("Metrics logged to MLFlow")

        except Exception as e:
            logger.error(f"Error processing and indexing document {s3_url}: {e}")
            self.mlflow_service.log_metrics({"error": 1})
            raise
#-----

# embedding_service.py
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

# glirel_service.py
#-----
# services/glirel_service.py

import torch
from glirel import GLiREL
from loguru import logger
from config import settings

class GLiRELService:
    """
    Service class for GLiREL model operations.
    """

    def __init__(self):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize GLiREL model
        glirel_model_name = settings.GLIREL_MODEL_NAME
        try:
            self.glirel_model = GLiREL.from_pretrained(glirel_model_name).to(self.device)
            logger.info(f"GLiREL model '{glirel_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GLiREL model: {e}")
            raise RuntimeError("GLiREL model loading failed.")

    def extract_relationships(self, texts):
        """
        Extract relationships from a list of texts using GLiREL.

        Args:
            texts (List[str]): List of texts to process.

        Returns:
            List[List[Dict]]: A list where each element corresponds to the relationships extracted from a text.
        """
        try:
            all_relationships = []
            for text in texts:
                relationships = self.glirel_model.predict_relationships(text)
                all_relationships.append(relationships)
            return all_relationships
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            raise

#-----
