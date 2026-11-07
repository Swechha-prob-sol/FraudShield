"""Tests for configuration and logging utilities."""

from pathlib import Path
from src.utils.config import get_project_root, load_config
from src.utils.logger import setup_logger


def test_get_project_root():
    root = get_project_root()
    assert root.exists()
    assert (root / "README.md").exists()
    assert (root / ".gitignore").exists()


def test_load_config():
    config = load_config()
    assert isinstance(config, dict)
    assert "project" in config
    assert config["project"]["name"] == "FraudShield"
    assert config["project"]["random_seed"] == 42
    assert "data" in config
    assert config["data"]["target_column"] == "Class"


def test_logger_creation():
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert len(logger.handlers) >= 1

