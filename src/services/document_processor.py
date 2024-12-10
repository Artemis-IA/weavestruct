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
from src.services.pgvector_service import PGVectorService
from src.services.neo4j_service import Neo4jService
from src.services.embedding_service import EmbeddingService
from src.services.gliner_service import GLiNERService
from src.services.glirel_service import GLiRELService

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
            if result.status == ConversionStatus.SUCCESS:
                output_dir = Path(self.s3_service.output_bucket)  
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
        doc_filename = local_path.stem

        def upload_to_s3(file_path: Path, bucket: str):
            """Helper function to upload a file to S3."""
            s3_key = f"output/{file_path.name}"
            try:
                self.s3_service.upload_file(file_path, bucket_name=bucket)
                logger.info(f"Uploaded {file_path} to {bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload {file_path} to S3: {e}")

        # Export des résultats dans les formats sélectionnés
        if ExportFormat.JSON in export_formats:
            json_path = output_dir / f"{doc_filename}.json"
            with json_path.open("w", encoding="utf-8") as json_file:
                json.dump(conversion_result.document.export_to_dict(), json_file, ensure_ascii=False, indent=2)
            upload_to_s3(json_path, self.s3_service.output_bucket)

        if ExportFormat.YAML in export_formats:
            yaml_path = output_dir / f"{doc_filename}.yaml"
            with yaml_path.open("w", encoding="utf-8") as yaml_file:
                yaml.dump(conversion_result.document.export_to_dict(), yaml_file, allow_unicode=True, default_flow_style=False)
            upload_to_s3(yaml_path, self.s3_service.output_bucket)

        if ExportFormat.MARKDOWN in export_formats:
            md_path = output_dir / f"{doc_filename}.md"
            with md_path.open("w", encoding="utf-8") as md_file:
                md_file.write(conversion_result.document.export_to_markdown())
            upload_to_s3(md_path, self.s3_service.output_bucket)

        if ExportFormat.TEXT in export_formats:
            text_path = output_dir / f"{doc_filename}.txt"
            with text_path.open("w", encoding="utf-8") as text_file:
                text_file.write(conversion_result.document.export_to_text())
            upload_to_s3(text_path, self.s3_service.output_bucket)



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
    
    def index_graph(self, doc: Document):
        """Process a document to extract nodes and relationships, adding them to Neo4j."""
        try:
            split_docs = self.text_splitter.split_documents([doc])
            split_docs = [
                Document(page_content=self.clean_text(chunk.page_content), metadata=chunk.metadata)
                for chunk in split_docs
            ]
            logger.debug(f"Document split into {len(split_docs)} chunks.")

            # Extract graph data and links
            graph_docs = self.graph_transformer.convert_to_graph_documents(split_docs)
            doc_links = [self.gliner_extractor.extract_many(chunk) for chunk in split_docs]
            # Log extracted nodes, edges, and links
            logger.info("Extracted graph data:")
            for graph_doc in graph_docs:
                if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                    for node in graph_doc.nodes:
                        logger.info(f"Node extracted: ID={node.id}, Name={node.properties.get('name', '')}, Type={node.type}")
                if hasattr(graph_doc, "edges") and graph_doc.edges:
                    for edge in graph_doc.edges:
                        logger.info(f"Edge extracted: Start={edge.start}, End={edge.end}, Type={edge.type}")

            logger.info("Extracted links:")
            for links in doc_links:
                for link in links:
                    logger.info(f"Link extracted: Tag={link.tag}, Kind={link.kind}")

            
            driver = self.neo4j_service.driver
            with driver.session() as session:
                with session.begin_transaction() as tx:
                    for graph_doc, links in zip(graph_docs, doc_links):
                        # Add nodes
                        if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                            for node in graph_doc.nodes:
                                tx.run(
                                    """
                                    MERGE (e:Entity {id: $id, name: $name, type: $type})
                                    ON CREATE SET e.created_at = timestamp()
                                    """,
                                    {
                                        "id": node.id,
                                        "name": node.properties.get("name", ""),
                                        "type": node.type,
                                    },
                                )
                                logger.info(f"Indexed Node: {node.id}, Type: {node.type}")

                        # Add relationships
                        if hasattr(graph_doc, "edges") and graph_doc.edges:
                            GLiRELService.add_relationships(tx, graph_doc.edges)

                        # Add links
                        for link in links:
                            if not link.tag or not link.kind:
                                logger.warning(f"Skipping invalid link: {link}")
                                continue
                            logger.info(f"Adding Link: {link}")
                            tx.run(
                                """
                                MERGE (e:Entity {name: $name})
                                ON CREATE SET e.created_at = timestamp()
                                RETURN e
                                """,
                                {"name": link.tag},
                            )
        except Exception as e:
            logger.error(f"Error processing document: {doc.metadata.get('name', 'unknown')} - {e}")

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