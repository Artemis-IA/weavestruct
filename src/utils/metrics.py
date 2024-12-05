
# utils/metrics.py
import os
import psutil
import GPUtil
import torch
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from codecarbon import EmissionsTracker
from config import settings

class MetricsManager:
    """
    Manage application metrics, system stats logging, and Prometheus metrics server.
    """

    def __init__(self, prometheus_port: int = 8002):
        # Define Prometheus metrics
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
            prometheus_url=f"http://localhost:{prometheus_port}",
        )

        self.prometheus_port = prometheus_port

    def start_metrics_server(self):
        """
        Start the Prometheus metrics server.
        """
        start_http_server(self.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {self.prometheus_port}.")

    def log_system_metrics(self):
        """
        Log system metrics: CPU, memory, GPU usage, and CO2 emissions.
        """
        try:
            # Log CPU and memory usage
            self.CPU_USAGE.set(psutil.cpu_percent())
            self.MEMORY_USAGE.set(psutil.virtual_memory().used)

            # Log GPU memory usage
            gpus = GPUtil.getGPUs()
            if gpus:
                self.GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)

            # Ensure the emissions tracker is running
            self.start_emissions_tracker()

            # Log CO2 emissions
            emissions = self.emissions_tracker.flush() if getattr(self.emissions_tracker, "_started", False) else None
            if emissions is not None:
                self.CARBON_EMISSIONS.set(emissions)
                logger.info(f"CO2 emissions logged: {emissions:.6f} kgCO₂eq")
            else:
                logger.info("No emissions data available.")
        except Exception as e:
            logger.warning(f"Error logging system metrics: {e}")

    def start_emissions_tracker(self):
        """
        Ensure the emissions tracker is running.
        """
        try:
            if self.emissions_tracker and not getattr(self.emissions_tracker._scheduler, '_stopped', True):
                self.emissions_tracker.start()
                logger.info("Emissions tracker started.")
        except Exception as e:
            logger.error(f"Error starting emissions tracker: {e}")
            self._remove_codecarbon_lock()

    def stop_emissions_tracker(self):
        """
        Gracefully stop the emissions tracker.
        """
        try:
            if self.emissions_tracker and not getattr(self.emissions_tracker._scheduler, '_stopped', True):
                self.emissions_tracker.stop()
                logger.info("Emissions tracker stopped.")
        except Exception as e:
            logger.error(f"Error stopping emissions tracker: {e}")

    @staticmethod
    def get_system_metrics() -> dict:
        """
        Retrieve and log system metrics.
        """
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
        from dependencies import get_s3_service, get_mlflow_service, get_pgvector_vector_store, get_neo4j_service

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
