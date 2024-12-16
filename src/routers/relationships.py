# # routers/relationships.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from loguru import logger

from src.services.neo4j_service import Neo4jService
from src.models.relationship import RelationshipCreate, Relationship
from src.dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()


@router.get("/relationships", response_model=List[Dict[str, Any]])
def get_all_relationships():
    """ Récupère toutes les relations. """
    with neo4j_service.driver.session() as session:
        result = session.run("MATCH ()-[r:RELATED_TO]->() RETURN r")
        relationships = [record["r"] for record in result]
    return [dict(rel) for rel in relationships]

@router.post("/relationships", response_model=Dict[str, Any])
def create_relationship(source_name: str, target_name: str, type: str):
    """ Crée ou fusionne une relation entre deux nœuds existants. """
    with neo4j_service.driver.session() as session:
        # Vérification de l'existence des deux nœuds
        source_node = session.run("MATCH (e:Entity {name:$name}) RETURN e", {"name": source_name}).single()
        if not source_node:
            raise HTTPException(status_code=404, detail=f"Source node '{source_name}' not found.")
        target_node = session.run("MATCH (e:Entity {name:$name}) RETURN e", {"name": target_name}).single()
        if not target_node:
            raise HTTPException(status_code=404, detail=f"Target node '{target_name}' not found.")

        # Création/fusion de la relation
        result = session.run(
            "MATCH (s:Entity {name:$source}), (t:Entity {name:$target}) "
            "MERGE (s)-[r:RELATED_TO {type:$type}]->(t) RETURN r",
            {"source": source_name, "target": target_name, "type": type}
        ).single()
    return dict(result["r"])

@router.put("/relationships/update")
def update_relationship(source_name: str, target_name: str, new_type: str):
    """ Met à jour le type d'une relation. """
    with neo4j_service.driver.session() as session:
        rel = session.run(
            "MATCH (s:Entity {name:$source})-[r:RELATED_TO]->(t:Entity {name:$target}) RETURN r",
            {"source": source_name, "target": target_name}
        ).single()
        if not rel:
            raise HTTPException(status_code=404, detail="Relationship not found.")

        result = session.run(
            "MATCH (s:Entity {name:$source})-[r:RELATED_TO]->(t:Entity {name:$target}) "
            "SET r.type=$new_type RETURN r",
            {"source": source_name, "target": target_name, "new_type": new_type}
        ).single()
    return dict(result["r"])

@router.delete("/relationships")
def delete_relationship(source_name: str, target_name: str):
    """ Supprime une relation entre deux nœuds. """
    with neo4j_service.driver.session() as session:
        rel = session.run(
            "MATCH (s:Entity {name:$source})-[r:RELATED_TO]->(t:Entity {name:$target}) RETURN r",
            {"source": source_name, "target": target_name}
        ).single()
        if not rel:
            raise HTTPException(status_code=404, detail="Relationship not found.")

        session.run(
            "MATCH (s:Entity {name:$source})-[r:RELATED_TO]->(t:Entity {name:$target}) DELETE r",
            {"source": source_name, "target": target_name}
        )
    return {"message": f"Relationship between '{source_name}' and '{target_name}' deleted successfully."}