from fastapi import Depends
from .auth import router as auth_router
from .chat import router as chat_router
from .datasets import router as datasets_router
from .documents import router as documents_router
from .entities import router as entities_router
# from .graph import router as graph_router
from .logging_router import router as logging_router
from .loopml import router as loopml_router
from .relationships import router as relationships_router
from .search import router as search_router
from .trainb import router as trainb_router
from .train import router as train_router
from ..services.security import verify_token

# Centralized configuration of all routers
ROUTERS = [
    {"router": auth_router, "prefix": "/auth", "tags": ["Authentication"]},
    {"router": chat_router, "prefix": "/chat", "tags": ["Chat"]},
    {"router": datasets_router, "prefix": "/datasets", "tags": ["Datasets"]},
    {"router": documents_router, "prefix": "/documents", "tags": ["Documents"]},
    # {"router": entities_router, "prefix": "/entities", "tags": ["Entities"]},
    # {"router": graph_router, "prefix": "/graph", "tags": ["Graph"]},
    {"router": logging_router, "prefix": "/logging", "tags": ["Logging"], "dependencies": [Depends(verify_token)]},
    {"router": loopml_router, "prefix": "/loopml", "tags": ["LoopML"], "dependencies": [Depends(verify_token)]},
    {"router": relationships_router, "prefix": "/relationships", "tags": ["Relationships"]},
    # {"router": search_router, "prefix": "/search", "tags": ["Search"]},
    # {"router": trainb_router, "prefix": "/train", "tags": ["Training"]},
    {"router": train_router, "prefix": "/train_operations", "tags": ["Training"]},
]
