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
