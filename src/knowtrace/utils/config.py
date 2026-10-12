"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str | Path) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save the YAML file.
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)