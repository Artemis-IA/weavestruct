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
