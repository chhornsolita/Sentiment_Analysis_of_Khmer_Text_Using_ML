"""
Feature Extraction Module

This module provides functions for extracting features from text data
using TF-IDF and other vectorization techniques.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def create_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with specified parameters.
    
    Args:
        max_features: Maximum number of features to extract
        ngram_range: Range of n-grams (e.g., (1, 2) for unigrams and bigrams)
        min_df: Minimum document frequency
        
    Returns:
        Configured TfidfVectorizer instance
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )
    return vectorizer


def compute_class_weights(y_train, classes=None):
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        y_train: Training labels
        classes: List of class names (optional)
        
    Returns:
        Dictionary mapping class labels to weights
    """
    if classes is None:
        classes = np.unique(y_train)
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    
    class_weight_dict = dict(zip(classes, weights))
    return class_weight_dict
