
# all.py
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
# services/train_service.py# services/train_service.py
from sqlalchemy.orm import Session
from schemas.train import TrainInput, TrainResponse
from models.training_run import TrainingRun
from models.dataset import Dataset
from loguru import logger
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.model_manager import ModelManager
from pathlib import Path
import mlflow
import os
import json
from typing import Optional

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
            # Load dataset
            dataset_path = self.load_dataset(train_input)

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"The dataset file {dataset_path} was not found.")
            with open(dataset_path, "r") as f:
                data = json.load(f)

            train_data, test_data = self.split_dataset(data, train_input.split_ratio)

            # Load model using ModelManager
            model = self.model_manager.load_model(train_input.model_name)

            # Start MLFlow run
            run_name = f"Training: {train_input.custom_model_name or train_input.model_name}"
            self.mlflow_service.start_run(run_name=run_name)
            self.mlflow_service.log_params({
                "model_name": train_input.model_name,
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
                eval_data={"samples": test_data},
                learning_rate=train_input.learning_rate,
                weight_decay=train_input.weight_decay,
                batch_size=train_input.batch_size,
                epochs=train_input.epochs,
                compile=train_input.compile_model,
            )

            # Save the fine-tuned model
            model_save_name = train_input.custom_model_name or train_input.model_name
            model_save_path = Path(f"models/{model_save_name}")
            model.save_pretrained(model_save_path)

            # Zip and upload the trained model to S3
            s3_url = self.model_manager.zip_and_upload_model(model_save_name)

            # Log artifacts and register model
            self.mlflow_service.log_artifact(str(model_save_path), artifact_path="trained_model")
            self.mlflow_service.register_model(model_save_name, model_save_path)

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

            logger.info(f"Training completed and model saved at {model_save_path}")
            self.mlflow_service.end_run()

            return TrainResponse(
                id=training_run.id,
                run_id=mlflow.active_run().info.run_id,
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

#-----

# model_manager.py
#-----
# services/model_manager.py
import torch
from typing import Dict, Any, Optional
from gliner import GLiNER
from transformers import AutoTokenizer
from services.s3_service import S3Service
from codecarbon import EmissionsTracker
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import os

class ModelManager:
    def __init__(self, s3_service: S3Service):
        self.s3_service = s3_service
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mlflow_client = MlflowClient()
        self.model_cache = {}
        self.model_registry_uri = os.environ.get("MLFLOW_TRACKING_URI")
        self.models_bucket = 'mlflow-artifacts'

        logger.info(f"Using device: {self.device}")

    def fetch_available_models(self):
        """
        Fetch the list of available models from MLflow registered models.
        """
        models = self.mlflow_client.list_registered_models()
        available_models = [model.name for model in models]
        logger.info(f"Available models fetched from MLflow: {available_models}")
        return available_models

    def load_model(self, model_name: str) -> GLiNER:
        """
        Load a model from MLflow artifacts stored in S3.
        """
        if model_name in self.model_cache:
            logger.info(f"Model {model_name} loaded from cache.")
            return self.model_cache[model_name]

        # Get the latest version of the model
        versions = self.mlflow_client.get_latest_versions(model_name, stages=['None'])
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")

        version = versions[0]
        model_uri = version.source

        # Parse the S3 URI
        s3_info = self.s3_service.parse_s3_url(model_uri)
        if not s3_info:
            raise ValueError(f"Invalid S3 URI: {model_uri}")
        bucket_name, object_key = s3_info

        # Download the model from S3
        local_model_path = Path(f"/tmp/{model_name}")
        local_model_path.mkdir(parents=True, exist_ok=True)
        success = self.s3_service.download_file(bucket_name, object_key, local_model_path / "model.zip")
        if not success:
            raise ValueError(f"Failed to download model {model_name} from S3.")

        # Unzip the model
        import zipfile
        with zipfile.ZipFile(local_model_path / "model.zip", 'r') as zip_ref:
            zip_ref.extractall(local_model_path)

        # Load the model
        model = GLiNER.from_pretrained(str(local_model_path)).to(self.device)
        self.model_cache[model_name] = model
        logger.info(f"Model {model_name} loaded from S3 artifacts.")
        return model

    def register_models_at_startup(self, model_names: list):
        """
        Register and log models at startup if they are not already registered.
        """
        for model_name in model_names:
            if not any(m.name == model_name for m in self.mlflow_client.list_registered_models()):
                # Load the model from HuggingFace or another source
                model = GLiNER.from_pretrained(model_name).to(self.device)
                model_save_path = Path(f"models/{model_name}")
                model.save_pretrained(model_save_path)

                # Start MLflow run
                mlflow.start_run(run_name=f"Registering {model_name}")

                # Log the model as an artifact
                mlflow.log_artifact(
                    str(model_save_path), artifact_path="model_artifacts"
                )

                # Register the model
                self.mlflow_client.create_registered_model(model_name)
                self.mlflow_client.create_model_version(
                    name=model_name,
                    source=f"{mlflow.get_artifact_uri()}/model_artifacts",
                    run_id=mlflow.active_run().info.run_id
                )

                # End MLflow run
                mlflow.end_run()

                logger.info(f"Model {model_name} registered and logged to MLflow.")
            else:
                logger.info(f"Model {model_name} is already registered in MLflow.")

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
            # Load the model
            model = self.load_model(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs_tokenized = tokenizer(inputs["text"], return_tensors="pt").to(self.device)

            # Run inference
            output = model(**inputs_tokenized)

            # Log hardware resource usage
            metrics = {
                "gpu_memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cpu_usage": torch.get_num_threads(),
            }

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
        s3_url = self.s3_service.upload_file(zip_path, bucket_name=self.models_bucket)
        if s3_url:
            logger.info(f"Model {model_name} uploaded successfully to {s3_url}")
            return s3_url
        else:
            logger.error(f"Failed to upload model {model_name} to S3")
            return None

