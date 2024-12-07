from docling.datamodel.base_models import InputFormat, OutputFormat
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime
from enum import Enum, auto


# Enumération des formats d'import et d'export
class ImportFormat(str, Enum):
    DOCX = InputFormat.DOCX.value
    PPTX = InputFormat.PPTX.value
    HTML = InputFormat.HTML.value
    IMAGE = InputFormat.IMAGE.value
    PDF = InputFormat.PDF.value
    ASCIIDOC = InputFormat.ASCIIDOC.value
    MD = InputFormat.MD.value
    YAML = "yaml"

class ExportFormat(str, Enum):
    JSON = OutputFormat.JSON.value
    YAML = "yaml"
    MARKDOWN = OutputFormat.MARKDOWN.value
    TEXT = OutputFormat.TEXT.value
    DOCTAGS = OutputFormat.DOCTAGS.value


# Conversion status
class ConversionStatus(str, Enum):
    PENDING = auto()
    STARTED = auto()
    FAILURE = auto()
    SUCCESS = auto()
    PARTIAL_SUCCESS = auto()

class DocInputType(str, Enum):
    PATH = auto()
    STREAM = auto()

class DocOutputType(str, Enum):
    PATH = auto()
    STREAM = auto()

class DoclingComponentType(str, Enum):
    DOCUMENT_BACKEND = auto()
    MODEL = auto()
    DOC_ASSEMBLER = auto()