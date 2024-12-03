import os
import random
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Load a YAML configuration file
def load_yaml_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Save data to a YAML file
def save_to_yaml(data: Dict[str, Any], file_path: str):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

# Load a JSON configuration file
def load_json_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        return json.load(file)

# Save data to a JSON file
def save_to_json(data: Dict[str, Any], file_path: str):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Create a directory if it doesn't exist
def ensure_directory(directory_path: str):
    Path(directory_path).mkdir(parents=True, exist_ok=True)

# Delete a directory
def delete_directory(directory_path: str):
    shutil.rmtree(directory_path, ignore_errors=True)

# Generate a random string
def generate_random_string(length: int = 8) -> str:
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choices(characters, k=length))

# Get an environment variable with a default value
def get_env_variable(key: str, default: Optional[str] = None) -> str:

    return os.getenv(key, default)
