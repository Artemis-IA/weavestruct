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
