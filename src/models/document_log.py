from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Callable
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentLog(Base):
    __tablename__ = 'document_logs'

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    s3_url = Column(String, nullable=False)

    def __repr__(self):
        return f"<DocumentLog(id={self.id}, file_name='{self.file_name}', s3_url='{self.s3_url}')>"


class DocumentLogService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def log_document(self, file_name: str, s3_url: str) -> None:
        """Logs a document entry in the database."""
        try:
            with self.session_factory() as session:
                log_entry = DocumentLog(file_name=file_name, s3_url=s3_url)
                session.add(log_entry)
                session.commit()
                logger.info(f"Document enregistré : {file_name}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Échec de l'enregistrement du document {file_name} : {e}")
            raise RuntimeError(f"Failed to log document: {e}")