from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional
from io import BytesIO
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc import TableCell, PictureDataType



# Base schemas
class ErrorItem(BaseModel):
    component_type: str
    module_name: str
    error_message: str

class Cell(BaseModel):
    id: int
    text: str
    bbox: BoundingBox

class OcrCell(Cell):
    confidence: float

class Cluster(BaseModel):
    id: int
    label: DocItemLabel
    bbox: BoundingBox
    confidence: float = 1.0
    cells: List[Cell] = []


class DocumentStream(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    stream: BytesIO

# Docliing prediction schemas
class LayoutPrediction(BaseModel):
    clusters: List[Cluster] = []

class Table(BaseModel):
    otsl_seq: List[str]
    num_rows: int = 0
    num_cols: int = 0
    table_cells: List[TableCell]

class TableStructurePrediction(BaseModel):
    table_map: Dict[int, Table] = {}

class FigureElement(BaseModel):
    annotations: List[PictureDataType] = []
    provenance: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None

class FigureClassificationPrediction(BaseModel):
    figure_count: int = 0
    figure_map: Dict[int, FigureElement] = {}

class EquationPrediction(BaseModel):
    equation_count: int = 0
    equation_map: Dict[int, str] = {}

#pages schemas
class BasePageElement(BaseModel):
    label: str
    id: int
    page_no: int
    cluster: Cluster
    text: Optional[str] = None

class PageElement(BasePageElement):
    pass

class AssembledUnit(BaseModel):
    elements: List[PageElement] = []
    body: List[PageElement] = []
    headers: List[PageElement] = []

class PagePredictions(BaseModel):
    layout: Optional[LayoutPrediction] = None
    tablestructure: Optional[TableStructurePrediction] = None
    figures_classification: Optional[FigureClassificationPrediction] = None
    equations_prediction: Optional[EquationPrediction] = None

class Page(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_no: int
    size: Optional[dict] = None
    cells: List[Cell] = []
    predictions: PagePredictions = PagePredictions()
    assembled: Optional[AssembledUnit] = None