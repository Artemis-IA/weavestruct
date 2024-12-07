# src/schemas/dataset.py
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime
from enum import Enum



from docling.datamodel.base_models import InputFormat

class ImportFormat(str, Enum):
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    IMAGE = "image"
    PDF = "pdf"
    ASCIIDOC = "asciidoc"
    MD = "md"

class ExportFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    MARKDOWN = "md"
    DOCTAGS = "doctags"


class DatasetCreate(BaseModel):
    name: str = Field(..., example="Sample Dataset")
    data: List[Dict] = Field(..., example=[{"text": "Example", "entities": ["Entity1", "Entity2"]}])

class DatasetUpdate(BaseModel):
    name: str = Field(..., example="Updated Dataset")
    data: List[Dict] = Field(..., example=[{"text": "Updated Example", "entities": ["Entity3"]}])

class DatasetResponse(BaseModel):
    id: int
    name: str
    data: Dict 
    created_at: datetime

    class Config:
        from_attributes = True

