# routers/logging.py
from fastapi import APIRouter
from utils.logging_utils import ModelLoggerService

router = APIRouter()

# Initialize the ModelLoggerService
model_logger_service = ModelLoggerService()

@router.post("/log_models/")
def log_models():
    """API endpoint to trigger logging of model details."""
    return model_logger_service.log_model_details()

@router.post("/log_queries/")
def log_queries(query: str):
    """API endpoint to trigger logging of queries."""
    return model_logger_service.log_query(query)