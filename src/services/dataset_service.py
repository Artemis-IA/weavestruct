# services/dataset_service.py

from typing import List, Optional, Callable
from fastapi import UploadFile
from sqlalchemy.orm import Session, sessionmaker
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
from loguru import logger
import aiofiles
from pathlib import Path
import unicodedata
from src.models.dataset import Dataset
from src.models.document_log import DocumentLog, DocumentLogService
from src.schemas.dataset import DatasetResponse
from src.dependencies import (
    get_s3_service,
    get_mlflow_service,
    get_pgvector_vector_store,
    get_neo4j_service,
    get_embedding_service,
    get_text_splitter,
    get_graph_transformer,
)
from src.config import settings
import json
from datetime import datetime
from src.models.document_log import DocumentLog, DocumentLogService
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from pathlib import Path
from typing import List, Dict
import json
import yaml
from loguru import logger

from src.services.annotations_pipeline import AnnotationPipelines


class DatasetService:

    def __init__(self, db: Session, session_factory: Callable[[], Session], annotation_pipeline: AnnotationPipelines):
        self.db = db
        self.s3_service = get_s3_service()
        self.mlflow_service = get_mlflow_service()
        self.embedding_service = get_embedding_service()
        self.session = db
        self.annotation_pipeline = annotation_pipeline
        self.dataset_processor = DatasetProcessor(
            s3_service=self.s3_service,
            document_log_service=DocumentLogService(session_factory=session_factory),
            annotation_pipeline=annotation_pipeline
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
                logger.info(f"Uploaded and saved temporary file: {temp_file}")


            all_annotations = []
            for temp_file in temp_files:
                texts = self.dataset_processor.extract_texts(temp_file)
                cleaned_texts = [self.dataset_processor.clean_text(text) for text in texts]
                logger.info(f"Cleaned texts from file '{temp_file.name}': {cleaned_texts}")

                if not cleaned_texts:
                    logger.warning(f"No text extracted from document: {temp_file}")
                    continue

                # Extract entities 
                logger.info(f"Extracting entities from {len(cleaned_texts)} texts...")
                try:
                    entities_list = self.annotation_pipeline.extract_entities(cleaned_texts, labels)
                    logger.info(f"Extracted entities from cleaned texts: {entities_list}")
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
                    logger.info(f"Formatted entities for text: {formatted_entities}")
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
                dataset_content = self.convert_to_conllu(all_annotations)
            elif output_format.lower() == "conll":
                dataset_content = self.convert_to_conll(all_annotations)
            elif output_format.lower() == "json-rel":
                dataset_content = json.dumps(all_annotations, ensure_ascii=False, indent=2)
            elif output_format.lower() == "xml":
                dataset_content = self.convert_to_xml(all_annotations)
            elif output_format.lower() == "rdf":
                dataset_content = self.convert_to_rdf(all_annotations)
            elif output_format.lower() == "uima":
                dataset_content = self.convert_to_uima(all_annotations)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            logger.info(f"Dataset content generated for output format '{output_format}': {dataset_content[:500]}")

            # Save to S3
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            dataset_file_name = f"{name or 'dataset'}_{timestamp}.{output_format}"
            dataset_file_path = Path(f"/tmp/{dataset_file_name}")
            async with aiofiles.open(dataset_file_path, "w", encoding="utf-8") as dataset_file:
                await dataset_file.write(dataset_content)
            logger.info(f"Dataset saved locally at: {dataset_file_path}")

            s3_url = self.s3_service.upload_file(
                dataset_file_path, bucket_name=settings.OUTPUT_BUCKET
            )
            logger.info(f"Dataset uploaded to S3 at: {s3_url}")

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


    def convert_to_conll(text: str, entities: List[Dict]) -> str:
        sentences = sent_tokenize(text)
        conll_data = []
        for sent in sentences:
            tokens = word_tokenize(sent)
            token_entities = defaultdict(str)
            for ent in entities:
                ent_text = ent['word']
                if ent_text in sent:
                    token_entities[ent_text] = ent['label']
            for token in tokens:
                label = token_entities.get(token, 'O')
                if label != 'O':
                    label = f"B-{label}" if label.startswith("B-") else f"I-{label}"
                conll_data.append(f"{token} {label}")
            conll_data.append("")  # Ligne vide pour séparer les phrases
        return "\n".join(conll_data)

    def convert_to_json_rel(text: str, entities: List[Dict], relations: List[Dict]) -> Dict:
        return {
            "text": text,
            "entities": entities,
            "relations": relations
        }

    def convert_to_xml(text: str, entities: List[Dict], relations: List[Dict]) -> str:
        root = ET.Element("document")
        text_element = ET.SubElement(root, "text")
        text_element.text = text
        entities_element = ET.SubElement(root, "entities")
        for ent in entities:
            entity = ET.SubElement(entities_element, "entity", type=ent['label'], start=str(ent['start']), end=str(ent['end']))
            entity.text = ent['word']
        relations_element = ET.SubElement(root, "relations")
        for rel in relations:
            relation = ET.SubElement(relations_element, "relation", type=rel['type'])
            head = ET.SubElement(relation, "head")
            head.text = rel['head']
            tail = ET.SubElement(relation, "tail")
            tail.text = rel['tail']
        return ET.tostring(root, encoding='utf-8').decode('utf-8')

    def convert_to_rdf(text: str, entities: List[Dict], relations: List[Dict]) -> str:
        rdf = "@prefix ex: <http://example.org/> .\n\n"
        for ent in entities:
            ent_id = ent['word'].replace(" ", "_")
            rdf += f"ex:{ent_id} a ex:{ent['label']} ;\n"
            rdf += f"    ex:hasText \"{ent['word']}\" .\n\n"
        for rel in relations:
            head_id = rel['head'].replace(" ", "_")
            tail_id = rel['tail'].replace(" ", "_")
            rdf += f"ex:{head_id} ex:{rel['type']} ex:{tail_id} .\n\n"
        return rdf

    def convert_to_uima(text: str, entities: List[Dict], relations: List[Dict]) -> str:
        root = ET.Element("CAS")
        text_element = ET.SubElement(root, "text")
        text_element.text = text
        annotations = ET.SubElement(root, "annotations")
        for ent in entities:
            annotation = ET.SubElement(annotations, "annotation", type=ent['label'], start=str(ent['start']), end=str(ent['end']))
            annotation.text = ent['word']
        relations_element = ET.SubElement(root, "relations")
        for rel in relations:
            relation = ET.SubElement(relations_element, "relation", type=rel['type'])
            head = ET.SubElement(relation, "head")
            head.text = rel['head']
            tail = ET.SubElement(relation, "tail")
            tail.text = rel['tail']
        return ET.tostring(root, encoding='utf-8').decode('utf-8')

    def convert_to_conllu(self, all_annotations: List[Dict]) -> str:
        conllu_data = []
        for idx, annotation in enumerate(all_annotations, 1):
            text = annotation["text"]
            entities = annotation["entities"]
            conll_data = self.convert_to_conll(text, entities)
            conllu_data.append(f"# Text {idx}")
            conllu_data.append(conll_data)
        return "\n\n".join(conllu_data)




class DatasetProcessor:
    def __init__(self, s3_service, document_log_service, annotation_pipeline: AnnotationPipelines):
        self.s3_service = s3_service
        self.document_log_service = document_log_service
        self.converter = self.create_converter()
        self.annotation_pipeline = annotation_pipeline


    def create_converter(self):
        options = PdfPipelineOptions()
        options.do_ocr = False
        options.generate_page_images = False
        options.generate_table_images = False
        options.generate_picture_images = False
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.IMAGE, InputFormat.HTML],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)},
        )

    def clean_text(self, text: str) -> str:
        # Normalisation Unicode
        text = unicodedata.normalize("NFKC", text)
        # Supprimer les caractères invisibles
        text = re.sub(r"[\u200b\u200c\u200d\u2060\uFEFF]", "", text)
        # Supprimer les balises ou placeholders comme <missing-text>
        text = re.sub(r"<.*?>", "", text)
        # Supprimer les caractères non supportés par SpaCy
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s.,;!?'\-():/]", "", text)
        # Conserver les symboles utiles collés au texte
        text = re.sub(r"\s*([-/])\s*", r"\1", text)
        # Consolidation des espaces multiples
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    
    def extract_texts(self, doc_path: Path) -> List[str]:
        results = list(self.converter.convert_all([doc_path]))
        if not results:
            logger.error(f"No result for document: {doc_path}")
            return []

        return [result.document.export_to_text() for result in results if result.status == ConversionStatus.SUCCESS]
    
    async def process_document(self, doc_path: Path) -> Dict:
        logger.info(f"Traitement du document '{doc_path.name}'")
        self.device_manager.log_device_stats()
        results = list(self.converter.convert_all([doc_path]))
        if not results:
            logger.error(f"Aucun résultat pour le document '{doc_path.name}'")
            return {}

        result = results[0]
        if result.status != ConversionStatus.SUCCESS:
            logger.error(f"Échec de la conversion pour le document '{doc_path.name}'")
            return {}

        text = result.document.export_to_text()
        entities = self.annotate_ner(text)
        relations = self.annotate_relations(text, entities)
        coreferences = self.annotate_coreferences(text)
        sentiments = self.annotate_sentiments(text)

        annotated_data = {
            "text": text,
            "entities": entities,
            "relations": relations,
            "coreferences": coreferences,
            "sentiments": sentiments
        }

        logger.info(f"Annotations complétées pour le document '{doc_path.name}'")
        return annotated_data

    def export_datasets(self, annotated_data: Dict, output_dir: Path, doc_filename: str, export_formats: List[str], destination: str):
        # Conversion des annotations en différents formats
        conll_data = self.convert_to_conll(annotated_data['text'], annotated_data['entities'])
        json_rel_data = self.convert_to_json_rel(annotated_data['text'], annotated_data['entities'], annotated_data['relations'])
        xml_data = self.convert_to_xml(annotated_data['text'], annotated_data['entities'], annotated_data['relations'])
        rdf_data = self.convert_to_rdf(annotated_data['text'], annotated_data['entities'], annotated_data['relations'])
        uima_data = self.convert_to_uima(annotated_data['text'], annotated_data['entities'], annotated_data['relations'])

        # Sauvegarder les fichiers dans les formats demandés
        if "conll" in export_formats:
            conll_path = output_dir / f"{doc_filename}.conll"
            with open(conll_path, "w", encoding="utf-8") as f:
                f.write(conll_data)
            if destination == "s3":
                self.s3_service.upload_file(conll_path, self.s3_service.output_bucket)
            logger.info(f"Exporté en CoNLL: '{conll_path}'")

        if "json" in export_formats:
            json_path = output_dir / f"{doc_filename}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotated_data, f, ensure_ascii=False, indent=2)
            if destination == "s3":
                self.s3_service.upload_file(json_path, self.s3_service.output_bucket)
            logger.info(f"Exporté en JSON: '{json_path}'")

        if "yaml" in export_formats:
            yaml_path = output_dir / f"{doc_filename}.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(annotated_data, f, allow_unicode=True)
            if destination == "s3":
                self.s3_service.upload_file(yaml_path, self.s3_service.output_bucket)
            logger.info(f"Exporté en YAML: '{yaml_path}'")

        if "xml" in export_formats:
            xml_path = output_dir / f"{doc_filename}.xml"
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_data)
            if destination == "s3":
                self.s3_service.upload_file(xml_path, self.s3_service.output_bucket)
            logger.info(f"Exporté en XML: '{xml_path}'")

        if "rdf" in export_formats:
            rdf_path = output_dir / f"{doc_filename}.ttl"
            with open(rdf_path, "w", encoding="utf-8") as f:
                f.write(rdf_data)
            if destination == "s3":
                self.s3_service.upload_file(rdf_path, self.s3_service.output_bucket)
            logger.info(f"Exporté en RDF: '{rdf_path}'")

        if "uima" in export_formats:
            uima_path = output_dir / f"{doc_filename}.xml"
            with open(uima_path, "w", encoding="utf-8") as f:
                f.write(uima_data)
            if destination == "s3":
                self.s3_service.upload_file(uima_path, self.s3_service.output_bucket)
            logger.info(f"Exporté en UIMA: '{uima_path}'")

        # Vous pouvez ajouter d'autres formats selon les besoins

    async def handle_file(self, file: UploadFile, export_formats: List[str], use_s3: bool, output_dir: Path, destination: str):
        temp_file = output_dir / file.filename
        async with aiofiles.open(temp_file, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        if use_s3:
            s3_url = self.s3_service.upload_file(temp_file, self.s3_service.input_bucket)
            self.document_log_service.log_document(file.filename, s3_url)
            logger.info(f"Uploadé '{file.filename}' vers S3.")
        else:
            self.document_log_service.log_document(file.filename)
            logger.info(f"Enregistré '{file.filename}' localement.")

        annotated_data = await self.process_document(temp_file)
        if annotated_data:
            self.export_datasets(annotated_data, output_dir, file.filename.stem, export_formats, destination)
            return True
        return False