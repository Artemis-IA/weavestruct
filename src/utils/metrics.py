
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
        self.prometheus_port = prometheus_port
        # self._unregister_existing_metrics()
        # Reuse existing metrics or create new ones
        self.REQUEST_COUNT = self._get_or_create_metric(
            Counter, "app_request_count", "Total number of requests"
        )
        self.PROCESS_TIME = self._get_or_create_metric(
            Histogram, "app_process_time_seconds", "Request processing time"
        )
        self.USING_GPU = self._get_or_create_metric(
            Gauge, "using_gpu", "Whether the GPU is being used (1 for GPU, 0 for CPU)"
        )
        self.GPU_MEMORY_USAGE = self._get_or_create_metric(
            Gauge, "gpu_memory_usage_bytes", "GPU memory usage"
        )
        self.GPU_MEMORY_USAGE_GB = self._get_or_create_metric(
            Gauge, "gpu_memory_usage_gb", "GPU memory usage in GB"
        )
        self.GPU_TEMPERATURE = self._get_or_create_metric(
            Gauge, "gpu_temperature_celsius", "GPU temperature in Celsius"
        )
        self.CPU_USAGE = self._get_or_create_metric(
            Gauge, "cpu_usage_percent", "CPU usage percentage"
        )
        self.MEMORY_USAGE = self._get_or_create_metric(
            Gauge, "memory_usage_bytes", "Memory usage in bytes"
        )
        self.CARBON_EMISSIONS = self._get_or_create_metric(
            Gauge, "carbon_emissions_grams", "Estimated CO2 emissions"
        )
        # Neo4j metrics
        self.NEO4J_REQUEST_COUNT = self._get_or_create_metric(
            Counter, "neo4j_request_count", "Number of requests sent to Neo4j"
        )
        self.NEO4J_REQUEST_FAILURES = self._get_or_create_metric(
            Counter, "neo4j_request_failures", "Number of failed Neo4j requests"
        )
        self.NEO4J_REQUEST_LATENCY = self._get_or_create_metric(
            Histogram, "neo4j_request_latency_seconds", "Latency of Neo4j requests"
        )
        # PostgreSQL metrics
        self.POSTGRES_QUERY_COUNT = self._get_or_create_metric(
            Counter, "postgres_query_count", "Number of successful PostgreSQL queries"
        )
        self.POSTGRES_QUERY_FAILURES = self._get_or_create_metric(
            Counter, "postgres_query_failures", "Number of failed PostgreSQL queries"
        )
        self.POSTGRES_QUERY_LATENCY = self._get_or_create_metric(
            Histogram, "postgres_query_latency_seconds", "Latency of PostgreSQL queries"
        )
        # Document processing metrics
        self.DOCUMENT_PROCESSING_SUCCESS = self._get_or_create_metric(
            Counter, "document_processing_success", "Number of successfully processed documents"
        )
        self.DOCUMENT_PROCESSING_FAILURES = self._get_or_create_metric(Counter, "document_processing_failures", "Number of failed document processing attempts")

        # CO2 emissions tracker
        self.emissions_tracker = None
        self.current_task_name = None


    def _get_or_create_metric(self, metric_type, name, description):
        """
        Get an existing metric if it is already registered, otherwise create it.
        """
        for collector in REGISTRY._collector_to_names.keys():
            if name in REGISTRY._collector_to_names[collector]:
                return collector
        return metric_type(name, description)
            
    def log_system_metrics(self):
        try:
            # Log CPU and memory usage
            self.CPU_USAGE.set(psutil.cpu_percent())
            self.MEMORY_USAGE.set(psutil.virtual_memory().used)

            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.GPU_MEMORY_USAGE_GB.set(gpu.memoryUsed / 1024)  # Convert to GB
                    self.GPU_TEMPERATURE.set(gpu.temperature)
                else:
                    logger.warning("No GPUs found.")
            else:
                logger.info("Using CPU. No GPU metrics available.")
        except Exception as e:
            logger.warning(f"Error logging system metrics: {e}")
            def log_hardware_usage(self):
                self.USING_GPU.set(1 if torch.cuda.is_available() else 0)

    def start_emissions_tracker(self):
        """
        Démarre le tracker d'émissions global.
        """
        try:
            if not self.emissions_tracker:
                self.emissions_tracker = EmissionsTracker(
                project_name="doc_processing",
                save_to_file=False,
                save_to_prometheus=True,
                prometheus_url=f"http://0.0.0.0:{settings.PROMETHEUS_PORT}", 
            )
                self.emissions_tracker.start()
                logger.info("Emissions tracker started.")
            else:
                logger.info("Emissions tracker is already running.")
        except Exception as e:
            logger.error(f"Failed to start emissions tracker: {e}")

    def stop_emissions_tracker(self):
        """
        Arrête le tracker d'émissions global.
        """
        try:
            if self.emissions_tracker:
                total_emissions = self.emissions_tracker.stop()
                logger.info(f"Emissions tracker stopped. Total CO2 emissions: {total_emissions:.6f} kg.")
                self.emissions_tracker = None
            else:
                logger.info("Emissions tracker is not running.")
        except Exception as e:
            logger.error(f"Failed to stop emissions tracker: {e}")

    def start_task(self, task_name: str):
        """
        Démarre une tâche spécifique pour le suivi des émissions.
        """
        if not self.emissions_tracker:
            logger.error("Emissions tracker is not running. Start it before starting tasks.")
            return

        try:
            self.emissions_tracker.start_task(task_name)
            self.current_task_name = task_name
            logger.info(f"Task '{task_name}' started.")
        except Exception as e:
            logger.error(f"Failed to start task '{task_name}': {e}")

    def stop_task(self, task_name: str = None):
        """
        Arrête le suivi des émissions pour une tâche spécifique.
        """
        if not self.emissions_tracker:
            logger.error("Emissions tracker is not running. Start it before stopping tasks.")
            return

        try:
            task_name = task_name or self.current_task_name
            if not task_name:
                logger.error("No task name specified.")
                return

            emissions = self.emissions_tracker.stop_task(task_name)
            logger.info(f"Task '{task_name}' stopped. Task emissions: {emissions.emissions:.6f} kg.")
            self.current_task_name = None
        except Exception as e:
            logger.error(f"Failed to stop task '{task_name}': {e}")

    def flush_emissions_data(self):
        """
        Vide les données d'émissions sans arrêter le tracker.
        """
        if not self.emissions_tracker:
            logger.error("Emissions tracker is not running.")
            return

        try:
            emissions = self.emissions_tracker.flush()
            logger.info(f"Emissions data flushed. Current total: {emissions:.6f} kg.")
        except Exception as e:
            logger.error(f"Failed to flush emissions data: {e}")


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

    def validate_services(self):
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
