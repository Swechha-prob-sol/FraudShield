"""Configuration management for FraudShield."""

from pathlib import Path
from typing import Any, Dict
import re

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _parse_scalar(val: str) -> Any:
    """Parses a scalar YAML string into appropriate Python types."""
    val = val.strip()
    if val.lower() in ("true", "yes", "on"):
        return True
    if val.lower() in ("false", "no", "off"):
        return False
    if val.lower() in ("null", "none", "~", ""):
        return None
    # Check for integer
    if re.fullmatch(r"[-+]?\d+", val):
        return int(val)
    # Check for float
    if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", val):
        return float(val)
    # Strip quotes if present
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def _simple_yaml_loader(text: str) -> Dict[str, Any]:
    """Lightweight fallback YAML loader for basic nested mappings.
    Used when PyYAML is not installed in the environment.
    """
    root_dict: Dict[str, Any] = {}
    current_section: Dict[str, Any] = root_dict

    for line in text.splitlines():
        # Remove comments and trailing whitespace
        line_clean = line.split("#", 1)[0].rstrip()
        if not line_clean:
            continue

        indent = len(line_clean) - len(line_clean.lstrip(" "))
        content = line_clean.strip()

        if ":" not in content:
            continue

        key, raw_val = content.split(":", 1)
        key = key.strip()
        val = raw_val.strip()

        if indent == 0:
            if not val:
                current_section = {}
                root_dict[key] = current_section
            else:
                root_dict[key] = _parse_scalar(val)
                current_section = root_dict
        else:
            current_section[key] = _parse_scalar(val)

    return root_dict


def get_project_root() -> Path:
    """Returns the absolute root directory of the FraudShield project."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file into a Python dictionary.

    Args:
        config_path: Relative or absolute path to the configuration YAML file.

    Returns:
        Dictionary representation of the YAML configuration.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
    """
    root = get_project_root()
    resolved_path = Path(config_path)

    if not resolved_path.is_absolute():
        resolved_path = root / config_path

    if not resolved_path.exists():
        logger.error(f"Configuration file not found: {resolved_path}")
        raise FileNotFoundError(f"Config file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml
        config = yaml.safe_load(content)
    except ImportError:
        logger.debug("PyYAML not found; using built-in lightweight YAML loader.")
        config = _simple_yaml_loader(content)

    return config
