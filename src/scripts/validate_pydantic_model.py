import os
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models import OllamaModel

# Définir un modèle Pydantic pour valider les entrées
class MyModel(BaseModel):
    city: str
    country: str

# Configurer le modèle Ollama
ollama_model = OllamaModel(model_name="llama3.2")
agent = Agent(ollama_model, result_type=MyModel)

def validate_input(prompt: str):
    """Valider une entrée utilisateur avec Ollama et Pydantic AI."""
    try:
        result = agent.run_sync(prompt)
        print("Validation réussie :", result.data)
        return True
    except ValidationError as e:
        print("Validation échouée :", e.errors())
        return False

if __name__ == "__main__":
    # Exemple d’entrée utilisateur
    prompt = "The windy city in the US of A."
    if validate_input(prompt):
        print("Les données sont valides.")
    else:
        print("Validation échouée.")
