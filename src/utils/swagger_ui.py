from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI

def setup_swagger_ui(app: FastAPI, static_dir: str = "static"):
    project_root = Path(__file__).resolve().parent.parent.parent
    static_dir = project_root / "static"
    
    if not static_dir.exists():
        raise RuntimeError(f"Static directory '{static_dir}' does not exist.")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    # Override the Swagger UI route to include custom JS and CSS
    @app.get("/", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="Document Processing API - Swagger UI",
            swagger_js_url="/static/darkmode.js",
            swagger_css_url="/static/darkmode.css",
        )
