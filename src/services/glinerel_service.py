# src/services/glinerel_service.py
import os
import yaml
import torch
from typing import List, Dict, Any
from loguru import logger

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer

from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_community.graph_vectorstores.links import add_links
from src.config import settings

class GlinerELService:
    """
    Service unifié qui utilise GlinerGraphTransformer pour extraire entités (GLiNER) et relations (GLiREL),
    et GLiNERLinkExtractor pour créer des liens entre documents sur la base des entités reconnues.
    """

    def __init__(self):
        # Chargement de la configuration
        with open(settings.CONF_FILE, 'r') as file:
            self.config = yaml.safe_load(file)

        # Configuration du device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialisation du GlinerGraphTransformer
        self.graph_transformer = GlinerGraphTransformer(
            allowed_nodes=self.config["allowed_nodes"],
            allowed_relationships=self.config["allowed_relationships"],
            gliner_model=settings.GLINER_MODEL_NAME,
            glirel_model=settings.GLIREL_MODEL_NAME,
            entity_confidence_threshold=0.1,
            relationship_confidence_threshold=0.1,
            device=str(self.device)
        )
        logger.info("GlinerGraphTransformer initialized successfully.")

        # Initialisation du GLiNERLinkExtractor
        # On fournit la liste des labels d'entités qu'on veut extraire pour créer des liens entre documents.
        # Ici, on utilise les mêmes `allowed_nodes`.
        self.link_extractor = GLiNERLinkExtractor(labels=self.config["allowed_nodes"], kind="entity")

    def convert_documents_to_graph(self, documents: List[Document]):
        """
        Convertit des documents en GraphDocuments contenant entités + relations.
        """
        try:
            graph_docs = self.graph_transformer.convert_to_graph_documents(documents)
            return graph_docs
        except Exception as e:
            logger.error(f"Error converting documents to graphs: {e}")
            raise

    def extract_entities_and_relations(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extrait entités et relations à partir d'une liste de textes.
        """
        documents = [Document(page_content=text) for text in texts]
        graph_docs = self.convert_documents_to_graph(documents)

        results = []
        for graph_doc in graph_docs:
            entities = []
            if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                for node in graph_doc.nodes:
                    entities.append({"text": node.id, "type": node.type})

            relationships = []
            if hasattr(graph_doc, "relationships") and graph_doc.relationships:
                for rel in graph_doc.relationships:
                    relationships.append({
                        "source": rel.source.id,
                        "target": rel.target.id,
                        "type": rel.type
                    })

            results.append({
                "entities": entities,
                "relationships": relationships
            })
        return results

    def extract_links_for_documents(self, documents: List[Document]):
        """
        Utilise GLiNERLinkExtractor pour extraire des liens (basés sur les entités nommées)
        et les ajoute aux métadonnées de chaque document.
        """
        for doc in documents:
            links = self.link_extractor.extract_one(doc)
            # On ajoute les links aux métadonnées
            add_links(doc, links)
        return documents
