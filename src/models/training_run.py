from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, default=lambda: str(uuid.uuid4()), unique=True)
    dataset_id = Column(Integer, nullable=False)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=32)
    status = Column(String, default="Started")
    created_at = Column(DateTime, default=datetime.utcnow)
