# src/scripts/generate_openapi.py
from fastapi.openapi.utils import get_openapi
from pathlib import Path
import json
import os
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.main import app

# Génération du schéma OpenAPI
def generate_openapi(output_path="docs/static/swagger.json"):
    # Crée le dossier de destination si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Générer le schéma OpenAPI
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Sauvegarder dans un fichier JSON
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    print(f"✅ OpenAPI schema saved to {output_path}")

if __name__ == "__main__":
    generate_openapi()
