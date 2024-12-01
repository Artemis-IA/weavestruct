# # routers/relationships.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from services.neo4j_service import Neo4jService
from schemas.ner_rel import RelationshipCreate, RelationshipResponse

router = APIRouter()
neo4j_service = Neo4jService(uri="bolt://localhost:7687", user="neo4j", password="password")


@router.post("/relationships", response_model=RelationshipResponse)
def create_relationship(relationship: RelationshipCreate):
    """
    Crée une nouvelle relation entre deux entités dans Neo4j.
    """
    result = neo4j_service.create_relationship(relationship.dict())
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create relationship")
    return {
        "source": relationship.source,
        "target": relationship.target,
        "type": relationship.type,
        "properties": relationship.properties,
    }


@router.get("/relationships/{relationship_id}", response_model=RelationshipResponse)
def get_relationship(relationship_id: str):
    """
    Récupère une relation spécifique par son ID.
    """
    relationship = neo4j_service.get_relationship(relationship_id)
    if not relationship:
        raise HTTPException(status_code=404, detail="Relationship not found")
    return {
        "source": relationship["start"]["id"],
        "target": relationship["end"]["id"],
        "type": relationship["type"],
        "properties": relationship.get("properties", {}),
    }


@router.delete("/relationships/{relationship_id}")
def delete_relationship(relationship_id: str):
    """
    Supprime une relation spécifique par son ID.
    """
    success = neo4j_service.delete_relationship(relationship_id)
    if not success:
        raise HTTPException(status_code=404, detail="Relationship not found or could not be deleted")
    return {"message": f"Relationship {relationship_id} deleted successfully"}


@router.get("/relationships", response_model=List[RelationshipResponse])
def get_all_relationships():
    """
    Récupère toutes les relations dans Neo4j.
    """
    relationships = neo4j_service.get_all_relationships()
    if not relationships:
        raise HTTPException(status_code=404, detail="No relationships found")
    return [
        {
            "source": rel["start"]["id"],
            "target": rel["end"]["id"],
            "type": rel["type"],
            "properties": rel.get("properties", {}),
        }
        for rel in relationships
    ]
