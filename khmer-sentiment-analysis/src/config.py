"""
Configuration file for Khmer Sentiment Analysis

This file contains all hyperparameters and settings for the project.
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# Data paths
RAW_DATA_PATH = os.path.join(DATA_DIR, 'Data Collection - Sheet1.csv')
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'data_cleaned_all.csv')

# Data preprocessing
TEXT_COLUMN = 'text'
TARGET_COLUMN = 'target'
CLEAN_TEXT_COLUMN = 'text_clean'

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_SPLIT = True

# Feature extraction
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# Model training
CV_FOLDS = 3
N_ITER_SEARCH = 10
SCORING_METRIC = 'f1_macro'
N_JOBS = -1

# Deep learning hyperparameters
MAX_WORDS = 5000
MAX_LEN = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 3

# Class labels
CLASS_LABELS = ["neg", "neu", "pos"]
NUM_CLASSES = len(CLASS_LABELS)

# Khmer slang dictionary
KHMER_SLANG = {
    "មិនចេះ": "មិនដឹង",
    "ចេះតែ": "តែងតែ",
    "អត់": "មិន",
    "ហ៊ាន": "ហ៊ាន",
}
