# utils/database.py
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from alembic.config import Config
from alembic import command
from alembic.migration import MigrationContext
import os
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
        """Yields a database session."""
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
        """Context manager for database sessions."""
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
        """Initializes the database by creating all tables."""
        DatabaseUtils.Base.metadata.create_all(bind=DatabaseUtils.engine)
        DatabaseUtils.logger.info("Database tables created successfully.")

    # Run Alembic migrations
    @staticmethod
    def run_migrations():
        """Runs Alembic migrations automatically."""
        alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "../../alembic.ini"))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
        DatabaseUtils.logger.info("Running Alembic migrations...")
        command.upgrade(alembic_cfg, "head")

    # Generate Alembic migrations
    @staticmethod
    def generate_migrations():
        """Generates new Alembic migration scripts if schema changes are detected."""
        alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "../../alembic.ini"))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

        with DatabaseUtils.engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current_revision = context.get_current_revision()
            if current_revision is None:
                DatabaseUtils.logger.info("No migrations applied yet. Initializing database.")
                command.stamp(alembic_cfg, "base")

        DatabaseUtils.logger.info("Generating Alembic migration scripts...")
        command.revision(alembic_cfg, autogenerate=True, message="Auto-generated migration")
