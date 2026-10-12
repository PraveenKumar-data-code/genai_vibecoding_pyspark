"""Project paths, defaults, and logging configuration."""

import logging
import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
DEFAULT_SEED: int = 42
DEFAULT_CUSTOMER_COUNT: int = 1_000
DEFAULT_PRODUCT_COUNT: int = 100
DEFAULT_ORDER_COUNT: int = 5_000
SPARK_APP_NAME: str = "EcommerceDataPipeline"


def configure_logging() -> None:
    """Configure console logging using the optional LOG_LEVEL environment variable."""
    level_name: str = os.getenv("LOG_LEVEL", "INFO").upper()
    level: int = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )