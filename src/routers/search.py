from fastapi import APIRouter, HTTPException, Query
from typing import List
from loguru import logger
from dependencies import get_neo4j_service
from services.neo4j_service import Neo4jService
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
