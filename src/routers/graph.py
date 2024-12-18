# routers/graph.py
import boto3, os, re, mlflow, time, yaml
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List, Dict, Any, Optional, Literal, Union
from fastapi import Form, File, UploadFile

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


from src.models.document_log import DocumentLog, DocumentLogService
from src.services.neo4j_service import Neo4jService
from src.services.s3_service import S3Service
from src.utils.metrics import MetricsManager
from src.dependencies import get_neo4j_service, get_s3_service, get_metrics_manager
from src.models.entity import Entity
from src.models.relationship import Relationship
from src.config import settings

router = APIRouter()

neo4j_service: Neo4jService = get_neo4j_service()
s3_client : S3Service = get_s3_service()
metrics = get_metrics_manager()

bucket_name = settings.INPUT_BUCKET

# GLiNER extractor and transformer
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
gliner_extractor = GLiNERLinkExtractor(
    labels=config["labels"],
    model="E3-JSI/gliner-multi-pii-domains-v1"
)
graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="knowledgator/gliner-multitask-large-v0.5",
    glirel_model="jackboyla/glirel-large-v0",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
)


def add_graph_to_neo4j(graph_docs):
    with neo4j_service.driver.session() as session:
        for doc in graph_docs:
            for node in doc.nodes:
                logger.info(f"Adding node: {node.id}, Type: {node.type}")
                session.run("MERGE (e:Entity {name: $name, type: $type})", {"name": node.id, "type": node.type})

            for edge in doc.relationships:
                logger.info(f"Adding relationship: {edge.type} between Source: {edge.source.id} and Target: {edge.target.id}")
                session.run(
                    "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                    "MERGE (source)-[:RELATED_TO {type: $type}]->(target)",
                    {"source": edge.source.id, "target": edge.target.id, "type": edge.type}
                )


@router.post("/index_nerrel/")
@metrics.NEO4J_REQUEST_LATENCY.time()
def index_pdfs(
    folder_path: Optional[str] = Form(
        None,
        description="Chemin local vers un dossier contenant des PDFs.",
        openapi_examples="/home/pi/Documents/IF-SRV/1pdf_subset"
    ),
    bucket_name: Optional[str] = Form(
        None,
        description="Nom du bucket S3. Laisser vide si on utilise un chemin local.",
        openapi_examples="s3://docs-input/"
    ),
    prefix: Optional[str] = Form(
        "",
        description="Préfixe pour filtrer les objets dans le bucket S3.",
        openapi_examples="my/prefix/"
    ),
):
    metrics.NEO4J_REQUEST_COUNT.inc()

    # Détermination de la source
    folder_path = folder_path.strip() if folder_path else None
    bucket_name = bucket_name.strip() if bucket_name else None

    if folder_path and bucket_name:
        raise HTTPException(status_code=400, detail="Fournir soit folder_path, soit bucket_name, pas les deux.")
    if folder_path:
        source_type = "local"
    elif bucket_name:
        source_type = "s3"
    else:
        raise HTTPException(status_code=400, detail="folder_path ou bucket_name est requis.")

    # Chargement des documents
    if source_type == "local":
        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail="Chemin local invalide.")
        loader = PyPDFDirectoryLoader(folder_path)
        documents = loader.load()
    else:
        # Source S3
        try:
            objects = s3_client.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", [])
            file_paths = []
            for obj in objects:
                local_path = os.path.join("/tmp", obj["Key"].replace("/", "_"))
                s3_client.download_file(bucket_name, obj["Key"], local_path)
                file_paths.append(local_path)
            loader = PyPDFDirectoryLoader("/tmp")
            documents = loader.load()
        except Exception as e:
            logger.error(f"Failed to load documents from S3: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des documents S3: {e}")

    # Transformation des documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for doc in documents:
        if not hasattr(doc, "page_content"):
            continue
        split_docs = text_splitter.split_documents([doc])
        graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
        add_graph_to_neo4j(graph_docs)

    return {"source_type": source_type, "message": "Documents indexés avec succès."}



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








@router.post("/upload_pdf")
@metrics.NEO4J_REQUEST_LATENCY.time()
def upload_pdf(file: UploadFile = File(...)):
    """
    Permet de charger un PDF via un POST multipart/form-data.
    Le PDF sera analysé et le graphe (nœuds + relations) sera inséré dans Neo4j.
    """
    metrics.NEO4J_REQUEST_COUNT.inc()
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Veuillez uploader un fichier PDF.")

    # Enregistrement temporaire du fichier PDF
    tmp_file_path = f"/tmp/{file.filename}"
    with open(tmp_file_path, "wb") as f:
        f.write(file.file.read())

    # Chargement du PDF avec le loader
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Transformation des documents (NER/REL)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for doc in documents:
        if not hasattr(doc, "page_content"):
            continue
        split_docs = text_splitter.split_documents([doc])
        graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
        add_graph_to_neo4j(graph_docs)

    return {"message": "PDF analysé et ajouté au graphe avec succès."}


@router.get("/graph", response_model=Dict[str, Any])
def get_full_graph():
    """
    Récupère la totalité du graphe (tous les nœuds et relations) 
    sous forme JSON (nodes et edges).
    Ce format peut être facilement consommé par une appli Streamlit
    pour construire une visualisation du graphe.
    """
    with neo4j_service.driver.session() as session:
        # Récupération de tous les nœuds
        nodes_result = session.run("MATCH (n:Entity) RETURN n")
        nodes = []
        for record in nodes_result:
            node_data = dict(record["n"])
            node_data["id"] = record["n"].id  # On peut ajouter l'ID interne de Neo4j
            nodes.append(node_data)
        
        # Récupération de toutes les relations
        relationships_result = session.run(
            "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) RETURN a, r, b"
        )
        edges = []
        for record in relationships_result:
            rel_data = dict(record["r"])
            rel_data["source"] = record["a"].id
            rel_data["target"] = record["b"].id
            edges.append(rel_data)

    return {"nodes": nodes, "edges": edges}


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




#######################################
#      ROUTES GESTION DES DOUBLONS     #
#######################################

@router.get("/nodes/duplicates", response_model=List[Dict[str, Any]])
def list_duplicate_nodes():
    """
    Liste les noms de nœuds qui apparaissent plus d'une fois, 
    avec la liste des nœuds concernés.
    """
    with neo4j_service.driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WITH e.name AS name, COLLECT(e) AS nodes, COUNT(e) AS c
            WHERE c > 1
            RETURN name, nodes, c
            """
        )
        duplicates = []
        for record in result:
            duplicates.append({
                "name": record["name"],
                "count": record["c"],
                "nodes": [dict(n) for n in record["nodes"]]
            })
    return duplicates

@router.delete("/nodes/duplicates")
def remove_duplicate_nodes():
    """
    Supprime les doublons, c'est-à-dire si plusieurs nœuds ont le même name,
    on n'en garde qu'un seul (le premier trouvé) et on supprime les autres.

    Attention : cette opération est destructive. Les relations des nœuds supprimés
    seront également supprimées (DETACH DELETE).
    """
    with neo4j_service.driver.session() as session:
        # Récupérer les groupes de doublons
        duplicates = session.run(
            """
            MATCH (e:Entity)
            WITH e.name AS name, COLLECT(e) AS nodes, COUNT(e) AS c
            WHERE c > 1
            RETURN name, nodes
            """
        ).values()

        # Pour chaque groupe, on supprime tous sauf le premier
        for name, nodes in duplicates:
            # nodes est une liste de nœuds
            # On garde le premier et on supprime les autres
            nodes_to_delete = nodes[1:]
            for n in nodes_to_delete:
                node_id = n.id  # identifiant interne Neo4j (id du nœud)
                session.run("MATCH (e) WHERE id(e)=$id DETACH DELETE e", {"id": node_id})

    return {"message": "Duplicates removed, only one node per name kept."}