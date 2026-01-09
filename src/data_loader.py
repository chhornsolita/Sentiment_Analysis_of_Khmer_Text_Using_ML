"""
Data Loading and Preparation Module

This module handles data loading, cleaning, and train-test splitting
for the Khmer sentiment analysis project.
"""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        encoding: File encoding (default: 'utf-8')
        
    Returns:
        Pandas DataFrame containing the dataset
    """
    df = pd.read_csv(file_path, encoding=encoding)
    return df


def clean_data(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Clean the dataset by removing invalid entries.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        
    Returns:
        Cleaned DataFrame
    """
    # Remove rows where text equals 'text' (header duplicates)
    df = df[df[text_column].str.lower() != text_column.lower()]
    
    # Remove null values
    df = df.dropna(subset=[text_column])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    text_column: str = 'text_clean',
    target_column: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        target_column: Name of the target column
        test_size: Proportion of test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split based on target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df[text_column]
    y = df[target_column]
    
    stratify_val = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_val
    )
    
    return X_train, X_test, y_train, y_test


def get_class_distribution(y: pd.Series) -> pd.Series:
    """
    Get class distribution counts.
    
    Args:
        y: Target labels series
        
    Returns:
        Series with class counts
    """
    return y.value_counts()
