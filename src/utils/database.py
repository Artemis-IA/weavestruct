# utils/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from src.config import settings
import logging

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

# Dependency to get the SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

# Context manager for database sessions
@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

# Create all tables
def init_db():
    from sqlalchemy import text  # To execute raw SQL if needed
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")
