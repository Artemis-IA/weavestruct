import time
from fastapi import FastAPI, Response, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from routers import documents, entities, relationships, search, graph, datasets, train, loopml
from utils.metrics import MetricsManager
from config import settings

# Initialize metrics manager
metrics_manager = MetricsManager(prometheus_port=8002)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Document Processing and Graph API",
        version="2.0.0",
        description="API for Document processing, NER/Relation Extraction & Embeddings/Graph Indexing",
        docs_url="/",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middleware for CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics middleware
    Instrumentator(
        excluded_handlers=["^/metrics", "^/redoc", "^/openapi.json"],
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(
        app,
        metric_namespace="metrics"
    ).expose(app)

    # Include Routers
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(entities.router, prefix="/entities", tags=["Entities"])
    app.include_router(relationships.router, prefix="/relationships", tags=["Relationships"])
    app.include_router(search.router, prefix="/search", tags=["Search"])
    app.include_router(graph.router, prefix="/graph", tags=["Graph"])
    app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
    app.include_router(train.router, prefix="/train", tags=["Training"])
    app.include_router(loopml.router, prefix="/loopml", tags=["LoopML link MLflow and Hugging Face"])

    # Metrics Router
    metrics_router = APIRouter(prefix="/metrics", tags=["Metrics"])

    @metrics_router.get("/")
    async def metrics():
        """Expose Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(metrics_router)

    # Middleware to track custom metrics
    @app.middleware("http")
    async def custom_metrics_middleware(request: Request, call_next):
        """Track request metrics and system stats."""
        start_time = time.time()
        metrics_manager.REQUEST_COUNT.inc()
        response = await call_next(request)
        latency = time.time() - start_time
        metrics_manager.PROCESS_TIME.observe(latency)
        metrics_manager.log_system_metrics()
        return response

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Application starting...")
    metrics_manager.start_emissions_tracker()
    metrics_manager.start_metrics_server()
    system_metrics = metrics_manager.get_system_metrics()
    device_type = "cuda" if system_metrics.get("cuda") else "CPU"
    logger.info(f"Device Type: {device_type}")
    metrics_manager.validate_services()


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    metrics_manager.emissions_tracker.stop()
    logger.info("Application shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
