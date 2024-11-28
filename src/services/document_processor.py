import os
import re
import json
import yaml
from pathlib import Path
from typing import List, Iterator

from sqlalchemy.orm import Session
from models.document_log import DocumentLog

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

from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.pgvector_service import PGVectorService
from services.neo4j_service import Neo4jService
from services.embedding_service import EmbeddingService

class CustomPdfPipelineOptions(PdfPipelineOptions):
    """Custom pipeline options for PDF processing."""
    do_picture_classifier: bool = False


class DoclingPDFLoader(BaseLoader):
    """Loader for converting PDFs to LCDocument format using Docling."""

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)


class DocumentProcessor:
    """Orchestrates the document processing pipeline: splitting, exporting, and indexing."""

    def __init__(
        self,
        s3_service: S3Service,
        mlflow_service: MLFlowService,
        pgvector_service: PGVectorService,
        neo4j_service: Neo4jService,
        embedding_service: EmbeddingService,
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
        self.session = session
        self.text_splitter = text_splitter
        self.graph_transformer = graph_transformer
        self.gliner_extractor = gliner_extractor

    def create_converter(self, use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool) -> DocumentConverter:
        """Create and configure a document converter."""
        options = CustomPdfPipelineOptions()
        options.do_ocr = use_ocr
        options.generate_page_images = True
        options.generate_table_images = export_tables
        options.generate_picture_images = export_figures
        options.do_picture_classifier = enrich_figures

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)}
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

    def export_document(self, result: ConversionResult, output_dir: Path, export_formats: List[str], export_figures: bool, export_tables: bool):
        """Export document into specified formats and upload to S3."""
        try:
            doc_filename = result.input.file.stem
            if result.status == ConversionStatus.SUCCESS:
                self._export_file(result, output_dir, export_formats, export_figures, export_tables, doc_filename)
                logger.info(f"Document exported successfully: {doc_filename}")
            else:
                logger.warning(f"Document export failed for {doc_filename}: {result.status}")
        except Exception as e:
            logger.error(f"Error exporting document: {e}")
            raise

    def _export_file(self, result, output_dir, export_formats, export_figures, export_tables, doc_filename):
        """Save and upload the exported document files."""
        for ext in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, ext, export_format=ext)

        if export_figures:
            self._export_images(result, output_dir / "figures", doc_filename, self.s3_service.layouts_bucket)
        if export_tables:
            self._export_tables(result, output_dir / "tables", doc_filename, self.s3_service.layouts_bucket)

    def _save_and_upload(self, result, output_dir, doc_filename, ext, export_format="json"):
        """Save a specific document format locally and upload it to S3."""
        file_path = output_dir / f"{doc_filename}.{ext}"
        with file_path.open("w", encoding="utf-8") as file:
            if export_format == "json":
                json.dump(result.document.export_to_dict(), file, ensure_ascii=False, indent=2)
            elif export_format == "yaml":
                yaml.dump(result.document.export_to_dict(), file, allow_unicode=True)
            elif export_format == "md":
                file.write(result.document.export_to_markdown())
        self.s3_service.upload_file(file_path, self.s3_service.output_bucket)

    def _export_images(self, result, figures_dir, doc_filename, bucket):
        """Export and upload document images."""
        figures_dir.mkdir(exist_ok=True)
        for idx, element in enumerate(result.document.iterate_items()):
            if isinstance(element, PictureItem):
                image_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                element.image.pil_image.save(image_path, format="PNG")
                self.s3_service.upload_file(image_path, bucket)

    def _export_tables(self, result, tables_dir, doc_filename, bucket):
        """Export and upload document tables."""
        tables_dir.mkdir(exist_ok=True)
        for idx, table in enumerate(result.document.tables):
            csv_path = tables_dir / f"{doc_filename}_table_{idx + 1}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False, encoding="utf-8")
            self.s3_service.upload_file(csv_path, bucket)

    def process_and_index_document(self, file_path: str):
        """Process a document: extract embeddings, store them in PGVector, and index them in Neo4j."""
        try:
            logger.info(f"Loading document from {file_path}")
            loader = DoclingPDFLoader(file_path=file_path)
            docs = list(loader.lazy_load())
            if not docs:
                raise ValueError("No valid documents found.")

            logger.info(f"Loaded {len(docs)} document(s) from {file_path}")

            logger.info("Splitting document into chunks...")
            split_docs = self.text_splitter.split_documents(docs)
            split_docs = [
                LCDocument(page_content=self.clean_text(chunk.page_content), metadata=chunk.metadata)
                for chunk in split_docs
            ]
            logger.info(f"Document split into {len(split_docs)} chunks.")

            # Step 1: Generate embeddings and store them in PGVector
            for doc in split_docs:
                embedding = self.embedding_service.generate_embedding(doc.page_content)
                if embedding:
                    self.pgvector_service.store_vector(
                        embedding=embedding, metadata=doc.metadata, content=doc.page_content
                    )

            logger.info(f"Indexed {len(split_docs)} chunks into PGVector.")

            # Step 2: Transform chunks into a graph structure
            logger.info("Transforming document chunks into graph structure...")
            graph_docs = self.graph_transformer.convert_to_graph_documents(split_docs)
            doc_links = [self.gliner_extractor.extract_one(chunk) for chunk in split_docs]

            # Step 3: Index nodes, edges, and links into Neo4j
            with self.neo4j_service.driver.session() as session:
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
                            self.add_relationships(tx, graph_doc.edges)

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

            logger.info("Successfully processed and indexed document.")

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise

    def add_relationships(self, tx, relationships: List[Relationship]):
        """Add relationships to Neo4j."""
        for rel in relationships:
            try:
                if not rel.source or not rel.target or not rel.type:
                    logger.warning(f"Skipping invalid relationship: {rel}")
                    continue
                tx.run(
                    """
                    MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
                    MERGE (source)-[r:$type {properties: $properties}]->(target)
                    ON CREATE SET r.created_at = timestamp()
                    """,
                    {
                        "source_id": rel.source.id,
                        "target_id": rel.target.id,
                        "type": rel.type,
                        "properties": rel.properties or {},
                    },
                )
                logger.info(f"Added Relationship: {rel.type} from {rel.source.id} to {rel.target.id}")
            except Exception as e:
                logger.error(f"Failed to add relationship: {rel}. Error: {e}")