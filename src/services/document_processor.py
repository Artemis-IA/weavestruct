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

from models.document_log import DocumentLog, DocumentLogService
from models.document import Document
from utils.database import SessionLocal

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
                output_dir = Path(self.s3_service.output_bucket)  # Utilisé temporairement pour le stockage local avant l'upload
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
        output_dir = Path(self.s3_service.output_bucket)
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

        # Log metrics to Prometheus
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

            metrics = {
                "documents_loaded": 0,
                "chunks_processed": 0,
                "embeddings_generated": 0,
                "entities_extracted": 0,
                "relationships_extracted": 0,
                "nodes_indexed": 0,
                "edges_indexed": 0,
                "errors": 0,
            }
            metrics["documents_loaded"] = 1

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

            # Step 7: Log metrics to Prometheus
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