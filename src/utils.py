"""
Shared utility functions for engagement detection project.

This module contains ONLY functions that are used in multiple scripts.
Core methodology and feature engineering remain visible in the main scripts.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configuration constants
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RANDOM_SEED = 42


def load_merged_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load merged BROMP-Caliper dataset.
    
    Args:
        filepath: Path to merged data. Uses default if None.
        
    Returns:
        DataFrame with merged features
    """
    if filepath is None:
        filepath = DATA_DIR / "prediction_features.csv"
    
    df = pd.read_csv(filepath)
    
    # Convert categorical columns
    cat_cols = ['behavior', 'affect', 'class', 'most_recent_event_type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df


def create_lag_features(
    df: pd.DataFrame, 
    features: List[str], 
    groupby_col: str = 'student_id'
) -> pd.DataFrame:
    """
    Create lagged features using proper temporal ordering.
    
    Uses shift(1).expanding() to prevent data leakage by ensuring
    features only use information from prior observations.
    
    Args:
        df: DataFrame with temporal data
        features: List of column names to create lag features for
        groupby_col: Column to group by (default: student_id)
        
    Returns:
        DataFrame with added lag features
    """
    for col in features:
        # Mean (expanding window)
        df[f'student_avg_{col}'] = df.groupby(groupby_col)[col].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        
        # Standard deviation (expanding window)
        df[f'student_std_{col}'] = df.groupby(groupby_col)[col].transform(
            lambda x: x.shift(1).expanding().std()
        )
        
        # Rolling mean (last 5 observations)
        df[f'student_recent_{col}'] = df.groupby(groupby_col)[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    
    # Fill NaN values with 0 (first observation has no prior data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def save_model(
    model: Any, 
    filepath: Path, 
    metadata: Optional[Dict] = None
) -> None:
    """
    Save trained model with optional metadata.
    
    Args:
        model: Trained model object
        filepath: Path to save model
        metadata: Optional dictionary with model metadata (params, metrics, etc.)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model': model,
        'metadata': metadata or {}
    }
    
    joblib.dump(save_dict, filepath)
    print(f"✓ Model saved to: {filepath}")
    
    if metadata:
        print(f"  Metadata: {metadata}")


def load_model(filepath: Path) -> tuple:
    """
    Load trained model and metadata.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Tuple of (model, metadata)
    """
    save_dict = joblib.load(filepath)
    model = save_dict['model']
    metadata = save_dict.get('metadata', {})
    
    print(f"✓ Model loaded from: {filepath}")
    if metadata:
        print(f"  Metadata: {metadata}")
    
    return model, metadata


def validate_data_files() -> bool:
    """
    Check if required data files exist.
    
    Returns:
        True if all required files exist, False otherwise
    """
    required_files = [
        DATA_DIR / 'BROMPclean.csv',
        DATA_DIR / 'caliperclean.csv',
    ]
    
    missing = [f for f in required_files if not f.exists()]
    
    if missing:
        print("❌ Missing required data files:")
        for f in missing:
            print(f"  - {f}")
        print("\nSee data/README.md for instructions.")
        return False
    
    print("✓ All required data files found")
    return True