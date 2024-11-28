import os
import random
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Load a YAML configuration file
def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its content as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed content of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Save data to a YAML file
def save_to_yaml(data: Dict[str, Any], file_path: str):
    """
    Save a dictionary to a YAML file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to the YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

# Load a JSON configuration file
def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file and return its content as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed content of the JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

# Save data to a JSON file
def save_to_json(data: Dict[str, Any], file_path: str):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Create a directory if it doesn't exist
def ensure_directory(directory_path: str):
    """
    Create a directory if it does not already exist.

    Args:
        directory_path (str): Path of the directory to create.
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

# Delete a directory
def delete_directory(directory_path: str):
    """
    Delete a directory and all its contents.

    Args:
        directory_path (str): Path of the directory to delete.
    """
    shutil.rmtree(directory_path, ignore_errors=True)

# Generate a random string
def generate_random_string(length: int = 8) -> str:
    """
    Generate a random alphanumeric string of specified length.

    Args:
        length (int): Length of the generated string (default is 8).

    Returns:
        str: Random alphanumeric string.
    """
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choices(characters, k=length))

# Get an environment variable with a default value
def get_env_variable(key: str, default: Optional[str] = None) -> str:
    """
    Get the value of an environment variable or return a default value if not set.

    Args:
        key (str): The environment variable key.
        default (str, optional): The default value to return if the variable is not set.

    Returns:
        str: The value of the environment variable or the default value.
    """
    return os.getenv(key, default)
