"""
Khmer Sentiment Analysis Package

This package provides a complete pipeline for Khmer text sentiment analysis,
including preprocessing, feature extraction, model training, and evaluation.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main modules for easy access
from .preprocessing import preprocess_khmer, batch_preprocess, KHMER_SLANG
from .data_loader import load_data, clean_data, prepare_train_test_split
from .feature_extraction import create_tfidf_vectorizer, compute_class_weights
from .evaluation import evaluate_model, compare_models, plot_confusion_matrix
from .model_persistence import save_model, load_model, save_comparison_report

__all__ = [
    'preprocess_khmer',
    'batch_preprocess',
    'KHMER_SLANG',
    'load_data',
    'clean_data',
    'prepare_train_test_split',
    'create_tfidf_vectorizer',
    'compute_class_weights',
    'evaluate_model',
    'compare_models',
    'plot_confusion_matrix',
    'save_model',
    'load_model',
    'save_comparison_report',
]
