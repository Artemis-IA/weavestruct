# routers/entities.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from services.neo4j_service import Neo4jService
from schemas.ner_rel import (
    EntityCreate,
    EntityUpdate,
    EntityResponse,
    RelationshipCreate,
    RelationshipResponse,
    EntityRelationshipsResponse,
)

router = APIRouter()
neo4j_service = Neo4jService(uri="bolt://localhost:7687", user="neo4j", password="password")

@router.post("/entities", response_model=EntityResponse)
def create_entity(entity: EntityCreate):
    result = neo4j_service.create_entity(entity.dict())
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create entity")
    return {"id": entity.id, "type": entity.type, "properties": entity.properties}

@router.get("/entities/{entity_id}", response_model=EntityResponse)
def get_entity(entity_id: str):
    entity = neo4j_service.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return {
        "id": entity["e"]["id"],
        "type": entity["e"].get("type", "Unknown"),
        "properties": entity["e"].get("properties", {}),
    }

@router.put("/entities/{entity_id}", response_model=EntityResponse)
def update_entity(entity_id: str, update_data: EntityUpdate):
    result = neo4j_service.update_entity(entity_id, update_data.dict(exclude_none=True))
    if not result:
        raise HTTPException(status_code=400, detail="Failed to update entity")
    return {"id": entity_id, "type": result["e"].get("type", "Unknown"), "properties": result["e"].get("properties", {})}

@router.delete("/entities/{entity_id}")
def delete_entity(entity_id: str):
    neo4j_service.delete_entity(entity_id)
    return {"message": f"Entity {entity_id} deleted successfully"}

@router.post("/relationships", response_model=RelationshipResponse)
def create_relationship(relationship: RelationshipCreate):
    result = neo4j_service.create_relationship(relationship.dict())
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create relationship")
    return relationship

@router.get("/entities/{entity_id}/relationships", response_model=EntityRelationshipsResponse)
def get_relationships(entity_id: str):
    entity = neo4j_service.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    relationships = neo4j_service.get_relationships_by_entity(entity_id)
    return {
        "entity": {
            "id": entity["e"]["id"],
            "type": entity["e"].get("type", "Unknown"),
            "properties": entity["e"].get("properties", {}),
        },
        "relationships": [
            {
                "source": entity_id,
                "target": rel["related"]["id"],
                "type": rel["type"],
                "properties": rel["properties"],
            }
            for rel in relationships
        ],
    }
