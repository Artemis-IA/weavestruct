from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from pathlib import Path
from fastapi import FastAPI


class SwaggerUISetup:
    def __init__(self, app: FastAPI, static_dir: str = "static"):
        self.app = app
        self.project_root = Path(__file__).resolve().parents[2]
        self.static_dir = self.project_root / static_dir

    def setup(self):
        self._mount_static()
        self._add_swagger_ui_route()
        self._customize_openapi()

    def _mount_static(self):
        if not self.static_dir.exists():
            raise RuntimeError(f"Static directory '{self.static_dir}' does not exist.")
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

    def _add_swagger_ui_route(self):
        @self.app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="Document Processing API - Swagger UI",
                swagger_js_url="https://cdn.jsdelivr.net/gh/Artemis-IA/weavestruct@main/static/darkmode.js",
                swagger_css_url="https://cdn.jsdelivr.net/gh/Artemis-IA/weavestruct@main/static/darkmode.css",
            )

    def _customize_openapi(self):
        def custom_openapi():
            if self.app.openapi_schema:
                return self.app.openapi_schema
            openapi_schema = get_openapi(
                title=self.app.title,
                version=self.app.version,
                description=self.app.description,
                routes=self.app.routes,
            )
            # Add security scheme
            openapi_schema["components"]["securitySchemes"] = {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                }
            }
            for path in openapi_schema["paths"]:
                for method in openapi_schema["paths"][path]:
                    openapi_schema["paths"][path][method]["security"] = [{"bearerAuth": []}]
            self.app.openapi_schema = openapi_schema
            return self.app.openapi_schema

        self.app.openapi = custom_openapi
