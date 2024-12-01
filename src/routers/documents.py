# routers/documents.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, Response
from pathlib import Path
from typing import List, Optional
from loguru import logger
import aiofiles

from dependencies import get_document_processor, get_s3_service
from services.document_processor import DocumentProcessor, ExportFormat, ImportFormat
from services.s3_service import S3Service

router = APIRouter()

# Endpoint pour uploader et traiter des fichiers
@router.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[ExportFormat] = Query(
        default=[ExportFormat.JSON],
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation",
        enum=[ExportFormat.JSON, ExportFormat.YAML, ExportFormat.MARKDOWN]
    ),
    use_ocr: bool = Query(False, title="Utiliser OCR", description="Activez l'OCR lors de la conversion."),
    export_figures: bool = Query(True, title="Exporter les figures", description="Activer ou désactiver l'exportation des figures"),
    export_tables: bool = Query(True, title="Exporter les tableaux", description="Activer ou désactiver l'exportation des tableaux"),
    enrich_figures: bool = Query(False, title="Enrichir les figures", description="Activer ou désactiver l'enrichissement des figures"),
    s3_service: S3Service = Depends(get_s3_service),
    document_processor: DocumentProcessor = Depends(get_document_processor),
):
    """
    Upload files to S3, process them using the DocumentProcessor, and export results in specified formats.
    """
    logger.info(f"Received {len(files)} files for processing.")
    success_count, failure_count = 0, 0

    for file in files:
        try:
            # Upload the file to the input bucket
            s3_input_key = f"input/{file.filename}"
            s3_service.upload_fileobj(file.file, s3_service.input_bucket, s3_input_key)
            logger.info(f"File {file.filename} uploaded to S3 input bucket.")

            # Process the document
            s3_input_url = s3_service.get_s3_url(s3_service.input_bucket, s3_input_key)
            document_processor.process_documents(
                s3_url=s3_input_url,
                export_formats=export_formats,
                use_ocr=use_ocr,
                export_figures=export_figures,
                export_tables=export_tables,
                enrich_figures=enrich_figures,
            )
            logger.info(f"File {file.filename} processed successfully.")
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            failure_count += 1
        finally:
            await file.close()

    return {
        "message": "Files processed",
        "success_count": success_count,
        "failure_count": failure_count,
    }


@router.post("/upload_path/")
async def upload_path(
    directory_path: str = Form(...),
    export_formats: List[ExportFormat] = Form(default=["json", "yaml", "md"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False),
    s3_service: S3Service = Depends(get_s3_service),
    document_processor: DocumentProcessor = Depends(get_document_processor),
):
    """
    Upload all files from a directory to the `input` bucket, process them,
    and store results in the `output` and `layouts` buckets.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    files = [f for f in dir_path.iterdir() if f.is_file()]
    logger.info(f"Found {len(files)} files in directory {directory_path}")

    success_count, failure_count = 0, 0

    for file_path in files:
        try:
            # Upload file to the `input` bucket
            s3_input_key = f"input/{file_path.name}"
            with file_path.open("rb") as file_obj:
                s3_service.upload_fileobj(file_obj, s3_service.input_bucket, s3_input_key)
            logger.info(f"File {file_path.name} uploaded to input bucket")

            # Process the document and store results in the `output` and `layouts` buckets
            s3_input_url = s3_service.get_s3_url(s3_service.input_bucket, s3_input_key)
            document_processor.process_documents(
                s3_url=s3_input_url,
                export_formats=export_formats,
                use_ocr=use_ocr,
                export_figures=export_figures,
                export_tables=export_tables,
                enrich_figures=enrich_figures,
            )
            logger.info(f"File {file_path.name} processed and results exported to S3")
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            failure_count += 1

    return {
        "message": "Directory processed",
        "success_count": success_count,
        "failure_count": failure_count,
    }

# Indexing endpoints (NER, Embeddings, and Graph Indexing)
@router.post("/index_document/")
async def index_document(
    file: Optional[UploadFile] = File(None),
    export_formats: List[ExportFormat] = Query(
        default=[ExportFormat.JSON],
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation",
        enum=[ExportFormat.JSON, ExportFormat.YAML, ExportFormat.MARKDOWN]
    ),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    s3_service: S3Service = Depends(get_s3_service),
):
    """
    Index a document by extracting entities, embeddings, and relationships. 
    Allows indexing either by uploading a document or providing its S3 URL.
    """
    logger.info(f"Indexing document: {file.filename if file else 'from S3 URL'}")

    try:
        if file:
            # Upload the file to the input bucket if provided
            s3_input_key = f"input/{file.filename}"
            s3_service.upload_fileobj(file.file, s3_service.input_bucket, s3_input_key)
            logger.info(f"File {file.filename} uploaded to S3 input bucket.")
            s3_url = s3_service.get_s3_url(s3_service.input_bucket, s3_input_key)

        # Process and index the document using the S3 URL
        document_processor.process_and_index_document(s3_url=s3_url, export_formats=export_formats)

        logger.info(f"Successfully indexed document: {file.filename if file else s3_url}.")
        return {"message": f"Document {file.filename if file else s3_url} indexed successfully."}
    except Exception as e:
        logger.error(f"Error indexing document {file.filename if file else s3_url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")
    finally:
        if file:
            await file.close()


@router.post("/index_path/")
async def index_path(
    directory_path: str = Form(...),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    s3_service: S3Service = Depends(get_s3_service)
):
    """
    Index all valid files in a directory.
    """
    logger.info(f"Indexing directory: {directory_path}")
    input_dir_path = Path(directory_path)

    if not input_dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    input_file_paths = [
        file for file in input_dir_path.glob("*") if file.suffix.lower() in [".pdf", ".png"]
    ]
    logger.info(f"Found {len(input_file_paths)} valid files in directory")

    success_count, failure_count = 0, 0

    for doc_path in input_file_paths:
        try:
            # Upload le fichier dans le bucket S3 'input'
            s3_input_key = f"input/{doc_path.name}"
            with doc_path.open("rb") as f:
                await s3_service.upload_file(f, s3_input_key, bucket_name=s3_service.input_bucket)
            s3_input_url = s3_service.get_s3_url(s3_input_key)

            # Traite et indexe le document depuis S3
            document_processor.process_and_index_document(s3_url=s3_input_url)

            logger.info(f"Indexed file {doc_path.name} successfully.")
            success_count += 1
        except Exception as e:
            logger.error(f"Error indexing file {doc_path.name}: {e}")
            failure_count += 1

    return {
        "message": "Directory indexed",
        "success_count": success_count,
        "failure_count": failure_count,
    }
