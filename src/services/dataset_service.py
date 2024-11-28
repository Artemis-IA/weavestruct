# services/dataset_service.py

from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session
from models.dataset import Dataset
from schemas.dataset import DatasetResponse
from loguru import logger
import aiofiles
from pathlib import Path
from dependencies import get_s3_service, get_mlflow_service, get_pgvector_service, get_neo4j_service, get_embedding_service, get_session
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
        self.pgvector_service = get_pgvector_service()
        self.neo4j_service = get_neo4j_service()
        self.embedding_service = get_embedding_service()
        self.session = get_session()
        self.text_splitter = None  # Initialize as needed
        self.graph_transformer = None  # Initialize as needed
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
                dataset_file_path, bucket_name=settings.S3_OUTPUT_BUCKET
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
