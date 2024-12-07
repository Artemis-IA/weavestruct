# routers/entities.py
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from src.services.neo4j_service import Neo4jService
from src.models.entity import EntityCreate, Entity
from src.dependencies import get_neo4j_service

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
