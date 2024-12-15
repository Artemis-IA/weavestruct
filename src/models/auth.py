# /Models/Auth.py
from sqlalchemy import Column, Integer, String, Boolean
from src.utils.database import DatabaseUtils

class User(DatabaseUtils.Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
