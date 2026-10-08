"""
Configuration Loader and Validation
===================================
Provides deterministic and validated configuration loading from YAML files.
"""

from pathlib import Path
from typing import Any, Union
import yaml


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing parsed configuration parameters.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the file is empty or does not parse into a valid mapping.
        yaml.YAMLError: If parsing encounters invalid YAML syntax.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Configuration file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping/dict, got {type(data).__name__}: {path}")

    return data
