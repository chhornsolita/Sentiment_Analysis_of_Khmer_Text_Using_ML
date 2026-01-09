"""
Deep Learning Models Module

This module provides functions for training and evaluating deep learning
models (LSTM, BiLSTM) for Khmer sentiment analysis.
"""

from typing import Tuple, Optional
import numpy as np


def create_lstm_model(
    max_words: int,
    max_len: int,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    num_classes: int = 3
):
    """
    Create a Bidirectional LSTM model for text classification.
    
    Args:
        max_words: Maximum vocabulary size
        max_len: Maximum sequence length
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of LSTM units
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras Sequential model, or None if TensorFlow not installed
    """
    try:
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
        
        model = Sequential([
            Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(lstm_units)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except ImportError:
        print("⚠ TensorFlow not installed. Cannot create LSTM model.")
        return None


def prepare_sequences(
    texts,
    max_words: int = 5000,
    max_len: int = 100,
    tokenizer=None
) -> Tuple:
    """
    Prepare text sequences for LSTM training.
    
    Args:
        texts: List of text strings
        max_words: Maximum vocabulary size
        max_len: Maximum sequence length
        tokenizer: Existing tokenizer (if None, creates new one)
        
    Returns:
        Tuple of (sequences, tokenizer) or (None, None) if TensorFlow not installed
    """
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len, padding='post')
        
        return padded, tokenizer
    except ImportError:
        print("⚠ TensorFlow not installed. Cannot prepare sequences.")
        return None, None


def train_lstm_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 3
):
    """
    Train LSTM model with early stopping.
    
    Args:
        model: Keras model
        X_train: Training sequences
        y_train: Training labels (encoded)
        X_val: Validation sequences
        y_val: Validation labels (encoded)
        epochs: Maximum number of epochs
        batch_size: Batch size
        patience: Early stopping patience
        
    Returns:
        Training history or None if TensorFlow not installed
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    except ImportError:
        print("⚠ TensorFlow not installed. Cannot train LSTM model.")
        return None
