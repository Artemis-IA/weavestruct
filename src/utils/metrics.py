
# utils/metrics.py
import os
import psutil
import GPUtil
import torch
from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, start_http_server, REGISTRY
from codecarbon import EmissionsTracker
from src.config import settings

class MetricsManager:
    """
    Manage application metrics, system stats logging, and Prometheus metrics server.
    """

    def __init__(self, prometheus_port: int = settings.PROMETHEUS_PORT):
        MetricsManager._unregister_existing_metrics()
        self.prometheus_port = prometheus_port
        self.REQUEST_COUNT = Counter("app_request_count", "Total number of requests")
        self.PROCESS_TIME = Histogram("app_process_time_seconds", "Request processing time")
        self.GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "GPU memory usage")
        self.CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percentage")
        self.MEMORY_USAGE = Gauge("memory_usage_bytes", "Memory usage in bytes")
        self.CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Estimated CO2 emissions")

        # Neo4j metrics
        self.NEO4J_REQUEST_COUNT = Counter("neo4j_request_count", "Number of requests sent to Neo4j")
        self.NEO4J_REQUEST_FAILURES = Counter("neo4j_request_failures", "Number of failed Neo4j requests")
        self.NEO4J_REQUEST_LATENCY = Histogram("neo4j_request_latency_seconds", "Latency of Neo4j requests")

        # PostgreSQL metrics
        self.POSTGRES_QUERY_COUNT = Counter("postgres_query_count", "Number of successful PostgreSQL queries")
        self.POSTGRES_QUERY_FAILURES = Counter("postgres_query_failures", "Number of failed PostgreSQL queries")
        self.POSTGRES_QUERY_LATENCY = Histogram("postgres_query_latency_seconds", "Latency of PostgreSQL queries")

        # Document processing metrics
        self.DOCUMENT_PROCESSING_SUCCESS = Counter(
            "document_processing_success", "Number of successfully processed documents"
        )
        self.DOCUMENT_PROCESSING_FAILURES = Counter(
            "document_processing_failures", "Number of failed document processing attempts"
        )

        # CO2 emissions tracker
        self.emissions_tracker = EmissionsTracker(
            project_name="doc_processing",
            save_to_file=False,
            save_to_prometheus=True,
            prometheus_url=f"http://localhost:{settings.PROMETHEUS_PORT_CARBON}",
        )


    @staticmethod
    def _unregister_existing_metrics():
        """Unregister all existing metrics from the default registry."""
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)
            
    # def start_metrics_server(self):

    #     """
    #     Start the Prometheus metrics server.
    #     """
    #     start_http_server(settings.PROMETHEUS_PORT_CARBON)
    #     logger.info(f"Prometheus metrics server started on port (settings.PROMETHEUS_PORT_CARBON).")

    def log_system_metrics(self):
        try:
            # Log CPU and memory usage
            self.CPU_USAGE.set(psutil.cpu_percent())
            self.MEMORY_USAGE.set(psutil.virtual_memory().used)

            gpus = GPUtil.getGPUs()
            if gpus:
                self.GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)
            self.start_emissions_tracker()

            emissions = self.emissions_tracker.flush() if getattr(self.emissions_tracker, "_started", False) else None
            if emissions is not None:
                self.CARBON_EMISSIONS.set(emissions)
                logger.info(f"CO2 emissions logged: {emissions:.6f} kgCO₂eq")
            else:
                logger.info("No emissions data available.")
        except Exception as e:
            logger.warning(f"Error logging system metrics: {e}")


    def start_emissions_tracker(self):
        try:
            if self.emissions_tracker and not getattr(self.emissions_tracker, '_started', False):
                self.emissions_tracker.start()
                logger.info("Emissions tracker started.")
        except Exception as e:
            logger.error(f"Error starting emissions tracker: {e}")
            self._remove_codecarbon_lock()

    def stop_emissions_tracker(self):
        try:
            if self.emissions_tracker and hasattr(self.emissions_tracker, '_started') and self.emissions_tracker._started:
                self.emissions_tracker.stop()
                logger.info("Emissions tracker stopped.")
        except Exception as e:
            logger.error(f"Error stopping emissions tracker: {e}")

    @staticmethod
    def get_system_metrics() -> dict:
        metrics = {
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_mb": psutil.virtual_memory().used / (1024 * 1024),
        }

        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics["gpu_memory_usage_mb"] = gpus[0].memoryUsed
        else:
            metrics["gpu_memory_usage_mb"] = None

        # CUDA or CPU check
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            metrics["cuda"] = True
            metrics["gpu_name"] = gpu_name
            logger.info(f"CUDA is available. Using GPU: {gpu_name}")
        else:
            metrics["cuda"] = False
            logger.info("CUDA is not available. Using CPU.")

        logger.info(f"System metrics: {metrics}")
        return metrics

    @staticmethod
    def _remove_codecarbon_lock() -> None:
        """
        Remove CodeCarbon lock file to avoid tracker errors.
        """
        lock_file = "/tmp/.codecarbon.lock"
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info("CodeCarbon lock file removed.")
            except Exception as e:
                logger.warning(f"Error removing CodeCarbon lock file: {e}")

    def validate_services(self):
        """
        Validate the availability of critical services at startup.
        """
        from src.dependencies import get_s3_service, get_mlflow_service, get_pgvector_vector_store, get_neo4j_service

        try:
            # Check S3 service and required buckets
            s3_service = get_s3_service()
            required_buckets = [settings.INPUT_BUCKET, settings.OUTPUT_BUCKET, settings.LAYOUTS_BUCKET]
            existing_buckets = s3_service.list_buckets()
            for bucket in required_buckets:
                if bucket not in existing_buckets:
                    raise RuntimeError(f"S3 bucket '{bucket}' is missing.")
            logger.info("S3 service and buckets validated successfully.")

            # Check MLflow service
            mlflow_service = get_mlflow_service()
            mlflow_service.validate_connection()
            logger.info("MLflow service validated successfully.")

            # Check PostgreSQL connection (via PGVector)
            pgvector_service = get_pgvector_vector_store()
            pgvector_service.validate_connection()
            logger.info("PostgreSQL service validated successfully.")

            # Check Neo4j service
            neo4j_service = get_neo4j_service()
            neo4j_service.validate_connection()
            logger.info("Neo4j service validated successfully.")

        except Exception as e:
            logger.error(f"Service check failed: {e}")
            raise RuntimeError("Startup failed due to unavailable services.") from e

        logger.info("All services validated successfully.")
