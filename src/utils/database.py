# utils/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from src.config import settings
import logging

class DatabaseUtils:
    # Load database URL from settings
    db_url = settings.DATABASE_URL

    # Create the SQLAlchemy engine
    engine = create_engine(db_url, echo=True)

    # Create a configured "Session" class
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create a base class for the models
    Base = declarative_base()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    @staticmethod
    def get_db():
        db = DatabaseUtils.SessionLocal()
        try:
            yield db
        except Exception as e:
            DatabaseUtils.logger.error(f"Database session error: {e}")
            raise
        finally:
            db.close()

    # Context manager for database sessions
    @staticmethod
    @contextmanager
    def db_session():
        db = DatabaseUtils.SessionLocal()
        try:
            yield db
        except Exception as e:
            DatabaseUtils.logger.error(f"Database session error: {e}")
            raise
        finally:
            db.close()

    # Create all tables
    @staticmethod
    def init_db():
        inspector = inspect(DatabaseUtils.engine)
        if not inspector.has_table("document_logs"):
            DatabaseUtils.Base.metadata.create_all(bind=DatabaseUtils.engine)
            DatabaseUtils.logger.info("Database tables created successfully.")
        else:
            DatabaseUtils.logger.info("Database tables already exist.")
