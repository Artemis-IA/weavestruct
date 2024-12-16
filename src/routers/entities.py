# routers/entities.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from loguru import logger

from src.services.neo4j_service import Neo4jService
from src.models.entity import EntityCreate, Entity
from src.dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()




#######################################
#          CRUD ENTITÉS (NODES)        #
#######################################

@router.get("/nodes", response_model=List[Dict[str, Any]])
def get_all_nodes():
    """ Récupère tous les nœuds de type Entity. """
    with neo4j_service.driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e")
        nodes = [record["e"] for record in result]
    return [dict(node) for node in nodes]

@router.get("/nodes/{name}", response_model=Dict[str, Any])
def get_node(name: str):
    """ Récupère un nœud par son nom. """
    with neo4j_service.driver.session() as session:
        result = session.run("MATCH (e:Entity {name:$name}) RETURN e", {"name": name})
        record = result.single()
    if record is None:
        raise HTTPException(status_code=404, detail="No node found with that name.")
    return dict(record["e"])

@router.post("/nodes", response_model=Dict[str, Any])
def create_node(name: str, type: str):
    """ Crée ou met à jour un nœud (Entity) en évitant les doublons. 
        Utilise MERGE pour éviter la création de doublons. 
    """
    with neo4j_service.driver.session() as session:
        result = session.run(
            "MERGE (e:Entity {name:$name}) "
            "SET e.type=$type "
            "RETURN e",
            {"name": name, "type": type}
        ).single()
    return dict(result["e"])

@router.put("/nodes/{name}", response_model=Dict[str, Any])
def update_node(name: str, new_type: Optional[str] = None):
    """ Met à jour un nœud. On peut par exemple mettre à jour son type. """
    with neo4j_service.driver.session() as session:
        node = session.run("MATCH (e:Entity {name:$name}) RETURN e", {"name": name}).single()
        if not node:
            raise HTTPException(status_code=404, detail="Node not found.")
        
        result = session.run(
            "MATCH (e:Entity {name:$name}) SET e.type=$type RETURN e",
            {"name": name, "type": new_type}
        ).single()
    return dict(result["e"])

@router.delete("/nodes/{name}")
def delete_node(name: str):
    """ Supprime un nœud par son nom, ainsi que les relations qui y sont connectées. """
    with neo4j_service.driver.session() as session:
        node = session.run("MATCH (e:Entity {name:$name}) RETURN e", {"name": name}).single()
        if not node:
            raise HTTPException(status_code=404, detail="Node not found.")
        
        session.run("MATCH (e:Entity {name:$name}) DETACH DELETE e", {"name": name})
    return {"message": f"Node '{name}' deleted successfully."}