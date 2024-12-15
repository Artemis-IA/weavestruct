import time
from fastapi import FastAPI, Response, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.routers import ROUTERS
from src.utils.database import DatabaseUtils
from src.utils.metrics import MetricsManager
from src.config import settings
from src.utils.swagger_ui import SwaggerUISetup

class AppLauncher:
    def __init__(self):
        self.metrics_manager = MetricsManager(prometheus_port=settings.PROMETHEUS_PORT)
        self.app = FastAPI(
            title="Document Processing and Graph API",
            version="2.0.0",
            description="API for Document processing, NER/Relation Extraction & Embeddings/Graph Indexing",
            docs_url="/",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            swagger_ui_parameters={
                "defaultModelsExpandDepth": -1,  # Fermer les modèles de réponse
                "defaultTagsExpandDepth": 0,    # Fermer toutes les sections des tags
            },
        )

    def setup(self):
        DatabaseUtils.init_db()
        self._setup_swagger_ui()
        self._setup_middleware()
        self._setup_routers()
        self._setup_metrics()
        self._setup_events()
        return self.app

    def _setup_swagger_ui(self):
        SwaggerUISetup(self.app).setup()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        Instrumentator(
            excluded_handlers=["^/metrics", "^/redoc", "^/openapi.json"],
            should_group_status_codes=True,
            should_ignore_untemplated=True,
        ).instrument(self.app, metric_namespace="metrics").expose(self.app)

        @self.app.middleware("http")
        async def custom_metrics_middleware(request: Request, call_next):
            start_time = time.time()
            self.metrics_manager.REQUEST_COUNT.inc()
            response = await call_next(request)
            latency = time.time() - start_time
            self.metrics_manager.PROCESS_TIME.observe(latency)
            self.metrics_manager.log_system_metrics()
            return response

    def _setup_routers(self):
        for route_config in ROUTERS:
            self.app.include_router(
                route_config["router"], prefix=route_config["prefix"], tags=route_config["tags"]
            )

    def _setup_metrics(self):
        metrics_router = APIRouter(prefix="/metrics", tags=["Metrics"])

        @metrics_router.get("/")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        self.app.include_router(metrics_router)

    def _setup_events(self):
        @self.app.on_event("startup")
        async def startup_event():
            logger.info("Application starting...")
            self.metrics_manager.start_emissions_tracker()
            system_metrics = self.metrics_manager.get_system_metrics()
            try:
                logger.info("Generating and applying Alembic migrations...")
                DatabaseUtils.generate_migrations()
                DatabaseUtils.run_migrations()
                logger.info("Migrations applied successfully.")
            except Exception as e:
                logger.error(f"Failed to apply migrations: {e}")
                raise e
        @self.app.on_event("shutdown")
        async def shutdown_event():
            self.metrics_manager.emissions_tracker.stop()
            logger.info("Application shutting down...")

app = AppLauncher().setup()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)