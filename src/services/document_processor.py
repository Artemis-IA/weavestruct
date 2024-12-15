# services/document_processor.py
import os
import re
import json
import yaml
from pathlib import Path
from typing import List, Iterator

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
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import PictureItem

from src.enums.documents import ImportFormat, ExportFormat, ConversionStatus
from src.services.s3_service import S3Service
from src.services.mlflow_service import MLFlowService


from src.models.document_log import DocumentLog, DocumentLogService
from src.models.document import Document
from src.utils.database import DatabaseUtils

class CustomPdfPipelineOptions(PdfPipelineOptions):
    """Custom pipeline options for PDF processing."""
    do_picture_classifier: bool = False

class DoclingPDFLoader(BaseLoader):
    """Loader for converting PDFs to LCDocument format using Docling."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._converter = DocumentConverter(
            allowed_formats=[ImportFormat.PDF, ImportFormat.DOCX],
            format_options={
                ImportFormat.PDF: PdfFormatOption(
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

class DocumentProcessor:
    """Orchestrates the document processing pipeline: loading, splitting, embedding, extracting, and indexing."""

    def __init__(
        self,
        s3_service: S3Service,
        mlflow_service: MLFlowService,
        # pgvector_service: PGVectorService,
        # embedding_service: EmbeddingService,
        session: Session,
        text_splitter: CharacterTextSplitter,
    ):
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        # self.pgvector_service = pgvector_service
        # self.embedding_service = embedding_service
        self.session = session
        self.text_splitter = text_splitter

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
            allowed_formats=[ImportFormat.PDF, ImportFormat.DOCX, ImportFormat.PPTX, ImportFormat.IMAGE, ImportFormat.HTML],
            format_options={ImportFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)},
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

    def _export_file(
        self,
        result: ConversionResult,
        export_formats: List[ExportFormat],
        export_figures: bool,
        export_tables: bool
    ):
        """Export document into specified formats and upload to S3."""
        try:
            doc_filename = Path(result.input.file).stem
            if result.status != ConversionStatus.SUCCESS:
                logger.warning(f"Document export failed for {doc_filename}: {result.status}")
                return

            output_dir = Path("/tmp/exports")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export des formats demandés
            for fmt in export_formats:
                file_path = output_dir / f"{doc_filename}.{fmt.value.lower()}"
                with file_path.open("w", encoding="utf-8") as file_obj:
                    if fmt == ExportFormat.JSON:
                        json.dump(result.document.export_to_dict(), file_obj, ensure_ascii=False, indent=2)
                    elif fmt == ExportFormat.YAML:
                        yaml.dump(result.document.export_to_dict(), file_obj, allow_unicode=True, default_flow_style=False)
                    elif fmt == ExportFormat.MARKDOWN:
                        file_obj.write(result.document.export_to_markdown())
                    elif fmt == ExportFormat.TEXT:
                        file_obj.write(result.document.export_to_text())

                self.s3_service.upload_to_specific_bucket(file_path, fmt.value.lower())
                logger.info(f"Exported and uploaded {file_path} as {fmt.value}")

            # Export des figures et tableaux
            if export_figures:
                figures_dir = output_dir / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                for figure_file in figures_dir.glob("*.png"):
                    self.s3_service.upload_to_specific_bucket(figure_file, "figures")
                    logger.info(f"Exported and uploaded figure {figure_file}")

            if export_tables:
                tables_dir = output_dir / "tables"
                tables_dir.mkdir(parents=True, exist_ok=True)
                for table_file in tables_dir.glob("*.csv"):
                    self.s3_service.upload_to_specific_bucket(table_file, "tables")
                    logger.info(f"Exported and uploaded table {table_file}")

                # Log document metadata
                self.log_document(file_name=result.input.file, s3_url=f"output/{doc_filename}.{export_formats[0].value.lower()}")

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
        except Exception as e:
            logger.error(f"Error exporting document: {e}")
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
        
        # Step 4: Export results
        self._export_file(conversion_result, export_formats, export_figures, export_tables)
        # Clean up the downloaded file
        if local_path.exists():
            local_path.unlink()
            logger.info(f"Deleted local file: {local_path}")

        # Log document metadata
        s3_output_key = f"output/{local_path.stem}.{export_formats[0].value.lower()}"
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
    
    def process_and_index_document(
        self,
        s3_url: str,
        export_formats: List[ExportFormat],
        use_ocr: bool = False,
        export_figures: bool = True,
        export_tables: bool = True,
        enrich_figures: bool = False,
    ):
        try:
            logger.info(f"Starting full processing and indexing for document: {s3_url}")

            # Start an MLFlow run for tracking
            self.mlflow_service.start_run(run_name=f"Process and Index {Path(s3_url).stem}")

            # Step 1: Process the document
            processed_documents = self.process_documents(
                s3_url=s3_url,
                export_formats=export_formats,
                use_ocr=use_ocr,
                export_figures=export_figures,
                export_tables=export_tables,
                enrich_figures=enrich_figures,
            )
            if not processed_documents:
                logger.error(f"Processing failed for document: {s3_url}")
                return

            logger.info(f"Processing completed for document: {s3_url}")

            # Step 2: Index the document into Neo4j
            for doc in processed_documents:
                self.index_graph(doc)

            logger.info(f"Indexing completed for document: {s3_url}")

        except Exception as e:
            logger.error(f"Error during process and index for document {s3_url}: {e}")
            # Log the error in MLFlow
            self.mlflow_service.log_metrics({"errors": 1})

        finally:
            # End the MLFlow run
            self.mlflow_service.end_run()