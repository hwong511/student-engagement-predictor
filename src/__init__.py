"""
Engagement Detection Utilities

Shared helper functions for the engagement detection pipeline.
Contains only code that is reused across multiple scripts.
"""

__version__ = "1.0.0"

from .utils import (
    load_merged_data,
    create_lag_features,
    save_model,
    load_model,
    DATA_DIR,
    OUTPUT_DIR,
    RANDOM_SEED
)

__all__ = [
    'load_merged_data',
    'create_lag_features',
    'save_model',
    'load_model',
    'DATA_DIR',
    'OUTPUT_DIR',
    'RANDOM_SEED',
]