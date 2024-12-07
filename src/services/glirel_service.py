# services/glirel_service.py

import torch
from glirel import GLiREL
from loguru import logger
from src.config import settings
from py2neo import Graph, NodeMatcher, Relationship
from typing import List

class GLiRELService:
    """
    Service class for GLiREL model operations.
    """

    def __init__(self):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize GLiREL model
        glirel_model_name = settings.GLIREL_MODEL_NAME
        try:
            self.glirel_model = GLiREL.from_pretrained(glirel_model_name).to(self.device)
            logger.info(f"GLiREL model '{glirel_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GLiREL model: {e}")
            raise RuntimeError("GLiREL model loading failed.")

    def extract_relationships(self, texts):
        """
        Extract relationships from a list of texts using GLiREL.

        Args:
            texts (List[str]): List of texts to process.

        Returns:
            List[List[Dict]]: A list where each element corresponds to the relationships extracted from a text.
        """
        try:
            all_relationships = []
            for text in texts:
                relationships = self.glirel_model.predict_relationships(text)
                all_relationships.append(relationships)
            return all_relationships
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            raise

    def add_relationships(tx, relationships: List[Relationship]):
        """Ajoute des relations extraites dans la base Neo4j."""
        for rel in relationships:
            try:
                # Validation des données
                if not rel.source or not rel.target or not rel.type:
                    logger.warning(f"Relation invalide, ignorée : {rel}")
                    continue
                
                logger.info(f"Ajout de la relation : {rel.type} ({rel.source.id} -> {rel.target.id})")
                
                # Cypher pour ajouter la relation
                tx.run(
                    """
                    MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATES_TO {type: $type, created_at: timestamp()}]->(target)
                    """,
                    {
                        "source_id": rel.source.id,
                        "target_id": rel.target.id,
                        "type": rel.type.upper(),  # Normalisation du type
                    },
                )
            except Exception as e:
                logger.error(f"Échec de l'ajout de la relation {rel.type}: {e}")
