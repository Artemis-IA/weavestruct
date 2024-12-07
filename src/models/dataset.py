# models/dataset.py
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import declarative_base

from datetime import datetime

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="Unnamed Dataset")
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)




