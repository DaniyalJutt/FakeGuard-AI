"""
VALIDATORS
Data validation functions for input data
"""

import pandas as pd


def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        bool: True if valid, raises error otherwise
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return True


def validate_text_column(df, text_column='text'):
    """
    Validate text column exists and has valid data
    
    Args:
        df: DataFrame to validate
        text_column: Name of text column
        
    Returns:
        bool: True if valid, raises error otherwise
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    
    if df[text_column].isna().all():
        raise ValueError(f"Text column '{text_column}' contains only NaN values")
    
    return True


def validate_file_path(file_path):
    """
    Validate file path exists
    
    Args:
        file_path: Path to file
        
    Returns:
        bool: True if valid, raises error otherwise
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.endswith('.csv'):
        raise ValueError("File must be a CSV file")
    
    return True

