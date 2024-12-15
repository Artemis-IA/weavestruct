# src/scripts/generate_openapi.py
import os
import json
from pathlib import Path
from fastapi.openapi.utils import get_openapi
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models import OllamaModel
from src.config import Settings  # Importez vos configurations Pydantic
from src.main import app  # Importez votre application FastAPI

# Configurer le modèle Ollama
ollama_model = OllamaModel(model_name="llama3.2")
agent = Agent(ollama_model, result_type=None)

def validate_config():
    """Valider les configurations Pydantic."""
    try:
        settings = Settings()
        print("Validation des configurations réussie :", settings.model_dump())
        return True
    except ValidationError as e:
        print("Erreur de validation des configurations :", e.errors())
        return False

def validate_with_ollama(prompt: str):
    """Valider les entrées utilisateur avec Ollama."""
    try:
        result = agent.run_sync(prompt)
        print("Validation avec Ollama réussie :", result.data)
        return True
    except ValidationError as e:
        print("Validation échouée :", e.errors())
        return False

def generate_openapi(output_path="docs/static/swagger.json"):
    """Générer le schéma OpenAPI si les validations passent."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    print(f"✅ Schéma OpenAPI généré : {output_path}")

if __name__ == "__main__":
    if validate_config():
        # Validation supplémentaire avec Ollama si nécessaire
        prompt = "Validate all configurations are correct."
        if validate_with_ollama(prompt):
            generate_openapi()
        else:
            print("Validation avec Ollama échouée.")
    else:
        print("La validation des configurations a échoué.")
