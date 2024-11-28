# services/embedding_service.py
from langchain_ollama.embeddings import OllamaEmbeddings
from loguru import logger
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str):
        self.embedding_model = OllamaEmbeddings(model=model_name)
        logger.info(f"Embedding model '{model_name}' initialized.")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

    def generate_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.embedding_model.embed_query(text)
            logger.info(f"Generated embedding for the given text.")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    def embed_documents(self, texts):
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)