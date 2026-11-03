from pathlib import Path

import yaml
from pydantic import BaseModel

# define the absolute base directory for the project
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


def convert_relative_paths(config_data: dict, base_path: Path) -> dict:
    """
    Recursively converts all relative paths in a dictionary to absolute paths based on base_path.
    Creates parent directories if they don't exist. Files don't need to exist (might be output files).
    """
    for key, value in config_data.items():
        if isinstance(value, dict):
            config_data[key] = convert_relative_paths(value, base_path)
        elif isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                full_path = (base_path / p).resolve()
                # Create parent directory if it doesn't exist
                full_path.parent.mkdir(parents=True, exist_ok=True)
                config_data[key] = str(full_path)
    return config_data


def load_yaml_config(filename: str) -> dict:
    """Loads a YAML file as a Python dictionary and converts relative paths to absolute."""
    path = CONFIG_DIR / filename
    with open(path, "r") as file:
        config_data = yaml.safe_load(file) or {}
    return convert_relative_paths(config_data, base_path=BASE_DIR)


class Config(BaseModel):
    soccer: dict


# load and parse config
config_data = load_yaml_config("params.yml")
config = Config(**config_data)
