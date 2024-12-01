
# entities.py
#-----
# routers/entities.py
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from models.entity import EntityCreate, Entity
from dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.post("/entities/", response_model=Entity)
async def create_entity(entity: EntityCreate):
    logger.info(f"Creating entity: {entity.name}")
    try:
        created_entity = neo4j_service.create_entity(entity)
        logger.info(f"Successfully created entity: {created_entity.name}")
        return created_entity
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/", response_model=List[Entity])
async def get_entities():
    logger.info("Retrieving all entities")
    try:
        entities = neo4j_service.get_all_entities()
        logger.info(f"Retrieved {len(entities)} entities")
        return entities
    except Exception as e:
        logger.error(f"Error retrieving entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(entity_id: str):
    logger.info(f"Retrieving entity with ID: {entity_id}")
    try:
        entity = neo4j_service.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        logger.info(f"Successfully retrieved entity: {entity.name}")
        return entity
    except Exception as e:
        logger.error(f"Error retrieving entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/entities/{entity_id}", response_model=dict)
async def delete_entity(entity_id: str):
    logger.info(f"Deleting entity with ID: {entity_id}")
    try:
        success = neo4j_service.delete_entity(entity_id)
        if not success:
            raise HTTPException(status_code=404, detail="Entity not found")
        logger.info(f"Successfully deleted entity with ID: {entity_id}")
        return {"message": f"Entity {entity_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# all.py
#-----

#-----

# graph.py
#-----
# routers/graph.py
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List, Dict, Any

from services.neo4j_service import Neo4jService
from dependencies import get_neo4j_service
from models.entity import Entity
from models.relationship import Relationship

router = APIRouter()

neo4j_service: Neo4jService = get_neo4j_service()


@router.get("/entities/", response_model=List[Entity])
async def get_all_entities():
    logger.info("Retrieving all entities from the graph")
    try:
        entities = neo4j_service.get_all_entities()
        logger.info(f"Retrieved {len(entities)} entities from the graph")
        return entities
    except Exception as e:
        logger.error(f"Error retrieving entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/", response_model=List[Relationship])
async def get_all_relationships():
    logger.info("Retrieving all relationships from the graph")
    try:
        relationships = neo4j_service.get_all_relationships()
        logger.info(f"Retrieved {len(relationships)} relationships from the graph")
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/", response_model=Dict[str, List[Dict[str, Any]]])
async def visualize_graph():
    logger.info("Generating graph visualization data")
    try:
        graph_data = neo4j_service.generate_graph_visualization()
        logger.info("Graph visualization data generated successfully")
        return graph_data
    except Exception as e:
        logger.error(f"Error generating graph visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# relationships.py
#-----
# # routers/relationships.py
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from models.relationship import RelationshipCreate, Relationship
from dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.post("/relationships/", response_model=Relationship)
async def create_relationship(relationship: RelationshipCreate):
    logger.info(f"Creating relationship: {relationship.type} between {relationship.source_id} and {relationship.target_id}")
    try:
        created_relationship = neo4j_service.create_relationship(relationship)
        logger.info(f"Successfully created relationship: {created_relationship.type}")
        return created_relationship
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/relationships/", response_model=List[Relationship])
async def get_relationships():
    logger.info("Retrieving all relationships")
    try:
        relationships = neo4j_service.get_all_relationships()
        logger.info(f"Retrieved {len(relationships)} relationships")
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/relationships/{relationship_id}", response_model=Relationship)
async def get_relationship(relationship_id: str):
    logger.info(f"Retrieving relationship with ID: {relationship_id}")
    try:
        relationship = neo4j_service.get_relationship(relationship_id)
        if not relationship:
            raise HTTPException(status_code=404, detail="Relationship not found")
        logger.info(f"Successfully retrieved relationship: {relationship.type}")
        return relationship
    except Exception as e:
        logger.error(f"Error retrieving relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/relationships/{relationship_id}", response_model=dict)
async def delete_relationship(relationship_id: str):
    logger.info(f"Deleting relationship with ID: {relationship_id}")
    try:
        success = neo4j_service.delete_relationship(relationship_id)
        if not success:
            raise HTTPException(status_code=404, detail="Relationship not found")
        logger.info(f"Successfully deleted relationship with ID: {relationship_id}")
        return {"message": f"Relationship {relationship_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# logging_router.py
#-----
# routers/logging.py
from fastapi import APIRouter
from utils.logging_utils import ModelLoggerService

router = APIRouter()

# Initialize the ModelLoggerService
model_logger_service = ModelLoggerService()

@router.post("/log_models/")
def log_models():
    """API endpoint to trigger logging of model details."""
    return model_logger_service.log_model_details()

@router.post("/log_queries/")
def log_queries(query: str):
    """API endpoint to trigger logging of queries."""
    return model_logger_service.log_query(query)
#-----

# documents.py
#-----
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
    document_processor: DocumentProcessor = Depends(get_document_processor),
    s3_service: S3Service = Depends(get_s3_service),
):
    """
    Index a document by extracting entities, embeddings, and relationships. 
    Allows indexing either by uploading a document or providing its S3 URL.
    """
    logger.info(f"Indexing document: {file.filename if file else s3_url}")

    if not file and not s3_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 's3_url' must be provided.")

    try:
        if file:
            # Upload the file to the input bucket if provided
            s3_input_key = f"input/{file.filename}"
            s3_service.upload_fileobj(file.file, s3_service.input_bucket, s3_input_key)
            logger.info(f"File {file.filename} uploaded to S3 input bucket.")
            s3_url = s3_service.get_s3_url(s3_service.input_bucket, s3_input_key)

        # Process and index the document using the S3 URL
        document_processor.process_and_index_document(s3_url=s3_url)

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

#-----

# train.py
#-----
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from schemas.train import TrainRequest, TrainResponse
from services.train_service import TrainService
from dependencies import get_db
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/", response_model=TrainResponse, tags=["Training"])
def train_endpoint(request: TrainRequest, db: Session = Depends(get_db)):
    """
    Endpoint pour entraîner le modèle NER.
    """
    try:
        train_service = TrainService(db=db)
        training_run = train_service.train_model(request)
        return training_run
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-----

# datasets.py
#-----
# routers/datasets.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional
from loguru import logger
from sqlalchemy.orm import Session

from dependencies import get_db
from models.dataset import Dataset
from schemas.dataset import DatasetResponse
from services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    output_format: str = Form("json-ner"),
    db: Session = Depends(get_db),
):
    """
    Create a NER dataset from uploaded files.
    """
    logger.info(f"Creating dataset with name: {name}")
    try:
        dataset_service = DatasetService(db=db)
        dataset = await dataset_service.create_dataset(
            name=name, files=files, labels=labels, output_format=output_format
        )
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    """
    List all datasets.
    """
    datasets = db.query(Dataset).all()
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=str(dataset.created_at),
        )
        for dataset in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get a dataset by ID.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        data=dataset.data,
        created_at=str(dataset.created_at),
    )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Delete a dataset by ID.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()
    return {"message": f"Dataset {dataset_id} deleted successfully"}

#-----

# search.py
#-----
from fastapi import APIRouter, HTTPException, Query
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from dependencies import get_neo4j_service
from models.entity import Entity
from models.relationship import Relationship

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.get("/search/entities/", response_model=List[Entity])
async def search_entities(keyword: str = Query(..., description="Keyword to search for")):
    logger.info(f"Searching entities with keyword: {keyword}")
    try:
        entities = neo4j_service.search_entities(keyword)
        logger.info(f"Found {len(entities)} entities matching keyword: {keyword}")
        return entities
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/relationships/", response_model=List[Relationship])
async def search_relationships(keyword: str = Query(..., description="Keyword to search for")):
    logger.info(f"Searching relationships with keyword: {keyword}")
    try:
        relationships = neo4j_service.search_relationships(keyword)
        logger.info(f"Found {len(relationships)} relationships matching keyword: {keyword}")
        return relationships
    except Exception as e:
        logger.error(f"Error searching relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----
