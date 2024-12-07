# services/gliner_service.py

import os
import yaml
import torch
from gliner import GLiNER
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from loguru import logger
from src.config import settings

class GLiNERService:
    """
    Service class for GLiNER model operations.
    """

    def __init__(self):
        # Load configurations
        config_file = settings.CONF_FILE
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize GLiNER model
        gliner_model_name = settings.GLINER_MODEL_NAME
        try:
            self.gliner_model = GLiNER.from_pretrained(gliner_model_name).to(self.device)
            logger.info(f"GLiNER model '{gliner_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            raise RuntimeError("GLiNER model loading failed.")

        # Initialize GlinerGraphTransformer
        self.graph_transformer = GlinerGraphTransformer(
            allowed_nodes=self.config["allowed_nodes"],
            allowed_relationships=self.config["allowed_relationships"],
            gliner_model=gliner_model_name,
            glirel_model=settings.GLIREL_MODEL_NAME,
            entity_confidence_threshold=0.1,
            relationship_confidence_threshold=0.1,
        )
        logger.info("GlinerGraphTransformer initialized.")

    def extract_entities(self, texts):
        """
        Extract entities from a list of texts using GlinerGraphTransformer.

        Args:
            texts (List[str]): List of texts to process.

        Returns:
            List[List[Dict]]: A list where each element corresponds to the entities extracted from a text.
        """
        try:
            # Convert texts to the format expected by GlinerGraphTransformer
            documents = [{"text": text} for text in texts]

            # Perform entity extraction
            graph_docs = self.graph_transformer.convert_to_graph_documents(documents)

            # Extract entities
            all_entities = []
            for graph_doc in graph_docs:
                entities = []
                if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                    for node in graph_doc.nodes:
                        entity = {
                            "text": node.properties.get("name", ""),
                            "label": node.type,
                            "start": node.properties.get("start", 0),
                            "end": node.properties.get("end", 0),
                        }
                        entities.append(entity)
                all_entities.append(entities)
            return all_entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            raise

    def predict_entities(self, text, labels=None, threshold=0.5, nested_ner=False):
        """
        Predict entities in a single text using GLiNER.

        Args:
            text (str): The text to analyze.
            labels (List[str], optional): List of labels to consider.
            threshold (float, optional): Confidence threshold.
            nested_ner (bool, optional): Whether to perform nested NER.

        Returns:
            Dict: A dictionary containing the text and the list of entities.
        """
        try:
            entities = self.gliner_model.predict_entities(
                text,
                labels=labels,
                flat_ner=not nested_ner,
                threshold=threshold,
            )
            return {
                "text": text,
                "entities": entities,
            }
        except Exception as e:
            logger.error(f"Error in predict_entities: {e}")
            raise
