# services/pgvector_service.py
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any, List, Optional
from loguru import logger
from langchain.docstore.document import Document


class PGVectorService:
    def __init__(self, db_url: str, table_name: str = "document_embeddings"):
        """
        Initialize PGVectorService with a PostgreSQL connection and table name.

        Args:
            db_url (str): Database connection string.
            table_name (str): Name of the table to store and query vectors.
        """
        self.db_url = db_url
        self.table_name = table_name
        self.connection = self._connect_to_db()
        self.cursor = self.connection.cursor()
        self._ensure_table_exists()

    def _connect_to_db(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            connection = psycopg2.connect(self.db_url)
            logger.info("Successfully connected to PostgreSQL database.")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _ensure_table_exists(self):
        """
        Ensure the required tables exist in the database.
        First ensure the 'documents' table, then the 'document_embeddings' table
        that references it.
        """
        try:
            # Ensure the documents table exists
            # The documents table should have at least an 'id' column to reference
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Ensure the document_embeddings table exists
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    document_id UUID PRIMARY KEY REFERENCES {self.table_name}(id) ON DELETE CASCADE,
                    embedding vector(768),
                    metadata JSONB,
                    content TEXT
                );
            """)

            self.connection.commit()
            logger.info(f"Tables '{self.table_name}' and '{self.table_name}' ensured in database.")
        except Exception as e:
            logger.error(f"Failed to ensure tables exist: {e}")
            raise

    def store_vector(self, document_id: str, embedding: List[float], metadata: Dict[str, Any], content: str) -> Optional[str]:
        """
        Store a vector in the database.

        Args:
            document_id (str): UUID of the document.
            embedding (List[float]): Vector embedding.
            metadata (Dict[str, Any]): Metadata for the document.
            content (str): Document content.

        Returns:
            Optional[str]: Document ID of the stored vector.
        """
        try:
            self.cursor.execute(
                f"""
                INSERT INTO {self.table_name} (document_id, embedding, metadata, content)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE
                SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata, content = EXCLUDED.content
                RETURNING document_id;
                """,
                (document_id, embedding, Json(metadata), content)
            )
            stored_id = self.cursor.fetchone()[0]
            self.connection.commit()
            logger.info(f"Vector stored for document ID {stored_id}.")
            return stored_id
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
            self.connection.rollback()
            return None

    def index_documents(self, documents: List[Document]):
        """
        Index multiple document chunks into the PGVector table.

        Args:
            documents (List[Document]): List of document chunks to index.
        """
        try:
            for document in documents:
                document_id = document.metadata.get("document_id")
                embedding = document.metadata.get("embedding", [])
                metadata = {key: value for key, value in document.metadata.items() if key not in ["embedding", "document_id"]}
                content = document.page_content

                if not document_id:
                    logger.warning("No document ID found for document chunk; skipping.")
                    continue

                if not embedding:
                    logger.warning("No embedding found for document chunk; skipping.")
                    continue

                self.store_vector(document_id, embedding, metadata, content)
            logger.info(f"Indexed {len(documents)} document chunks into PGVector.")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    def search_vector(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the nearest vectors.

        Args:
            query_vector (List[float]): Query vector.
            k (int): Number of nearest neighbors to return.

        Returns:
            List[Dict[str, Any]]: List of results with metadata and distances.
        """
        try:
            self.cursor.execute(
                f"""
                SELECT document_id, content, metadata, embedding <=> %s AS distance
                FROM {self.table_name}
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (query_vector, k)
            )
            results = self.cursor.fetchall()
            logger.info(f"Found {len(results)} nearest vectors.")
            return [
                {"document_id": row[0], "content": row[1], "metadata": row[2], "distance": row[3]}
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error searching vector: {e}")
            return []

    def close(self):
        """Close the database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed.")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    def validate_connection(self):
        """Validate the connection to the PostgreSQL database."""
        try:
            self.cursor.execute("SELECT 1;")
            self.connection.commit()
            logger.info("PostgreSQL connection validated.")
        except Exception as e:
            logger.error(f"Failed to validate PostgreSQL connection: {e}")
            raise
