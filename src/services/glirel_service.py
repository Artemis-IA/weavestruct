# services/glirel_service.py

import torch
from glirel import GLiREL
from loguru import logger
from config import settings

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
