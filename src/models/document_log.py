from sqlalchemy import Column, String, Integer, Table, MetaData, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from typing import Callable
from sqlalchemy.ext.declarative import declarative_base
from loguru import logger

Base = declarative_base()

class DocumentLog(Base):
    __tablename__ = 'document_logs'

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    s3_url = Column(String, nullable=False)

    def __repr__(self):
        return f"<DocumentLog(id={self.id}, file_name='{self.file_name}', s3_url='{self.s3_url}')>"


class DocumentLogService:
    def __init__(self, session_factory: Callable[[], Session], engine=None):
        self.session_factory = session_factory
        self.engine = engine

    def ensure_table_exists(self):
        """Ensure that the `document_logs` table exists in the database."""
        try:
            with self.engine.connect() as connection:
                logger.debug("Vérification de l'existence de la table 'document_logs'.")
                connection.execute("SELECT 1 FROM document_logs LIMIT 1")
        except ProgrammingError:
            logger.warning("Table 'document_logs' inexistante. Tentative de création.")
            try:
                # Création de la table si elle n'existe pas
                Base.metadata.create_all(self.engine)
                logger.info("Table 'document_logs' créée avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors de la création de la table 'document_logs' : {e}")
                raise
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la vérification ou création de la table : {e}")
            raise

    def log_document(self, file_name: str, s3_url: str) -> None:
        """Logs a document entry in the database."""
        try:
            self.ensure_table_exists()  # Vérifie ou crée la table avant l'insertion
            with self.session_factory() as session:
                log_entry = DocumentLog(file_name=file_name, s3_url=s3_url)
                session.add(log_entry)
                session.commit()
                logger.info(f"Document enregistré : {file_name}")
        except ProgrammingError as e:
            logger.error(f"Problème SQL lors du logging du document {file_name} : {e}")
            self.recover_from_missing_table()
        except OperationalError as e:
            logger.error(f"Erreur opérationnelle SQL : {e}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Échec de l'enregistrement du document {file_name} : {e}")
        except Exception as e:
            logger.error(f"Erreur critique lors du logging du document {file_name} : {e}")

    def recover_from_missing_table(self):
        """Tente de recréer la table si elle est manquante."""
        logger.warning("Tentative de récupération après l'absence de table.")
        try:
            self.ensure_table_exists()
        except Exception as e:
            logger.error(f"Impossible de récupérer après une erreur : {e}")
