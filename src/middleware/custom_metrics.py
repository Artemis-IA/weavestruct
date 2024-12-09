# # src/middleware/custom_metrics.py
# import time
# import psutil
# import GPUtil
# from prometheus_client import CollectorRegistry, Histogram, Counter, Gauge
# from fastapi import Request, Response
# from starlette.middleware.base import BaseHTTPMiddleware

# # Prometheus metrics
# REQUEST_COUNT = Counter("app_request_count", "Total number of requests received")
# REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests in seconds")
# CPU_USAGE = Gauge("app_cpu_usage_percent", "CPU usage in percent")
# MEMORY_USAGE = Gauge("app_memory_usage_bytes", "Memory usage in bytes")
# GPU_MEMORY_USAGE = Gauge("app_gpu_memory_usage_bytes", "GPU memory usage in bytes")

# class CustomMetricsMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next) -> Response:
        
#         start_time = time.time()
#         REQUEST_COUNT.inc()  # Increment the request count

#         # Call the next middleware or actual request handler
#         response = await call_next(request)

#         # Measure latency
#         process_time = time.time() - start_time
#         REQUEST_LATENCY.observe(process_time)

#         # Log system metrics
#         CPU_USAGE.set(psutil.cpu_percent())
#         MEMORY_USAGE.set(psutil.virtual_memory().used)

#         # Log GPU metrics if available
#         gpus = GPUtil.getGPUs()
#         if gpus:
#             GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)  # Only log the first GPU

#         return response