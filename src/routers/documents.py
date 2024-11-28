# routers/documents.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
from typing import List
from loguru import logger
import aiofiles

from services.document_processor import DocumentProcessor
from services.rag_service import RAGChainService
from dependencies import get_document_processor, get_rag_service

router = APIRouter()

# Dependency injection
document_processor: DocumentProcessor = get_document_processor()
rag_service: RAGChainService = get_rag_service()


@router.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[str] = Form(default=["json"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False),
):
    """
    Endpoint pour télécharger et traiter des fichiers pour l'extraction de données.
    """
    logger.info(f"Received {len(files)} files for upload")
    success_count, failure_count = 0, 0

    for file in files:
        temp_file = Path(f"/tmp/{file.filename}")
        try:
            # Sauvegarde du fichier en utilisant aiofiles pour compatibilité async
            async with aiofiles.open(temp_file, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)

            # Traitement du document
            result = document_processor.process_and_index_document(temp_file)

            # Exportation
            document_processor.export_document(
                result, temp_file.parent, export_formats, export_figures, export_tables
            )

            logger.info(f"File {file.filename} processed successfully.")
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            failure_count += 1
        finally:
            if temp_file.exists():
                temp_file.unlink()  # Suppression du fichier temporaire

    return {
        "message": "Files processed",
        "success_count": success_count,
        "failure_count": failure_count,
    }


@router.post("/upload_path/")
async def upload_path(
    file_path: str = Form("/home/pi/Documents/IF-SRV/4pdfs_subset/"),
    export_formats: List[str] = Form(default=["json"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False),
):
    """
    Endpoint pour traiter un répertoire de fichiers.
    """
    logger.info(f"Processing directory: {file_path}")

    input_dir_path = Path(file_path)
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    input_file_paths = [
        file for file in input_dir_path.glob("*")
        if file.suffix.lower() in [".pdf", ".docx"]
    ]
    logger.info(f"Found {len(input_file_paths)} valid files in directory")

    success_count, failure_count = 0, 0

    for doc_path in input_file_paths:
        try:
            result = document_processor.process_and_index_document(doc_path)

            document_processor.export_document(
                result, doc_path.parent, export_formats, export_figures, export_tables
            )

            logger.info(f"File {doc_path.name} processed successfully.")
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {doc_path.name}: {e}")
            failure_count += 1

    return {
        "message": "Directory processed",
        "success_count": success_count,
        "failure_count": failure_count,
    }


@router.post("/index_document/")
async def index_document(file: UploadFile = File(...)):
    """
    Index a single document by extracting entities and relationships.
    """
    logger.info(f"Indexing document: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")

    # Use aiofiles for asynchronous file writing
    import aiofiles
    async with aiofiles.open(temp_file, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        document_processor.process_and_index_document(temp_file)
        logger.info(f"Successfully indexed document: {file.filename}")
        return {"message": f"Document {file.filename} indexed successfully."}
    except Exception as e:
        logger.error(f"Error indexing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")
    finally:
        temp_file.unlink()



@router.post("/rag_process/")
async def process_rag_document(file: UploadFile = File(...)):
    """
    Process a document for RAG, splitting it, embedding it, and storing it in the vector store.
    """
    logger.info(f"Processing document for RAG: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")

    async with temp_file.open("wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        result = rag_service.process_document_for_rag(temp_file)
        logger.info(f"Document successfully processed for RAG: {file.filename}")
        return {"message": "Document successfully processed for RAG.", "details": result}
    except Exception as e:
        logger.error(f"Error processing document for RAG: {file.filename}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document for RAG: {e}")
    finally:
        temp_file.unlink()
