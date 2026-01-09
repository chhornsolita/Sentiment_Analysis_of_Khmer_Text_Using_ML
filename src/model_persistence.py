"""
Model Persistence Module

This module provides functions for saving and loading trained models,
along with their metadata and preprocessing objects.
"""

import joblib
import json
import os
from datetime import datetime
from typing import Dict, Optional, Any


def save_model(
    model,
    model_name: str,
    model_type: str,
    performance: Dict[str, float],
    hyperparameters: Dict[str, Any],
    save_dir: str = 'models/saved_models',
    tokenizer=None,
    label_encoder=None
) -> Dict[str, str]:
    """
    Save model with metadata.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        model_type: Type ('traditional_ml' or 'deep_learning')
        performance: Dictionary of performance metrics
        hyperparameters: Dictionary of hyperparameters
        save_dir: Directory to save models
        tokenizer: Tokenizer object (for LSTM)
        label_encoder: Label encoder object
        
    Returns:
        Dictionary with paths to saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    if model_type == 'deep_learning':
        # Save LSTM/deep learning model
        model_path = os.path.join(save_dir, f'best_model_lstm_{timestamp}.keras')
        model.save(model_path)
        saved_files['model'] = model_path
        print(f"✓ LSTM model saved: {model_path}")
        
        # Save tokenizer
        if tokenizer is not None:
            tokenizer_path = os.path.join(save_dir, f'tokenizer_lstm_{timestamp}.pkl')
            joblib.dump(tokenizer, tokenizer_path)
            saved_files['tokenizer'] = tokenizer_path
            print(f"✓ Tokenizer saved: {tokenizer_path}")
        
        # Save label encoder
        if label_encoder is not None:
            le_path = os.path.join(save_dir, f'label_encoder_lstm_{timestamp}.pkl')
            joblib.dump(label_encoder, le_path)
            saved_files['label_encoder'] = le_path
            print(f"✓ Label Encoder saved: {le_path}")
    
    else:
        # Save traditional ML model
        safe_name = model_name.lower().replace(" ", "_")
        model_path = os.path.join(save_dir, f'best_model_{safe_name}_{timestamp}.pkl')
        joblib.dump(model, model_path)
        saved_files['model'] = model_path
        print(f"✓ Model saved: {model_path}")
        
        # Save label encoder if provided
        if label_encoder is not None:
            le_path = os.path.join(save_dir, f'label_encoder_{safe_name}_{timestamp}.pkl')
            joblib.dump(label_encoder, le_path)
            saved_files['label_encoder'] = le_path
            print(f"✓ Label Encoder saved: {le_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'performance': performance,
        'hyperparameters': hyperparameters,
        'files': saved_files
    }
    
    metadata_path = os.path.join(save_dir, f'best_model_metadata_{timestamp}.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Metadata saved: {metadata_path}")
    
    return saved_files


def load_model(model_path: str, model_type: str = 'traditional_ml'):
    """
    Load a saved model.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ('traditional_ml' or 'deep_learning')
        
    Returns:
        Loaded model object
    """
    if model_type == 'deep_learning':
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            return model
        except ImportError:
            print("⚠ TensorFlow not installed. Cannot load LSTM model.")
            return None
    else:
        model = joblib.load(model_path)
        return model


def load_preprocessing_objects(tokenizer_path: Optional[str] = None, 
                               le_path: Optional[str] = None):
    """
    Load preprocessing objects (tokenizer, label encoder).
    
    Args:
        tokenizer_path: Path to tokenizer file
        le_path: Path to label encoder file
        
    Returns:
        Tuple of (tokenizer, label_encoder)
    """
    tokenizer = None
    label_encoder = None
    
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = joblib.load(tokenizer_path)
    
    if le_path and os.path.exists(le_path):
        label_encoder = joblib.load(le_path)
    
    return tokenizer, label_encoder


def save_comparison_report(
    comparison_df,
    save_dir: str = 'results/reports',
    filename: Optional[str] = None
) -> str:
    """
    Save model comparison report to CSV.
    
    Args:
        comparison_df: DataFrame with comparison results
        save_dir: Directory to save report
        filename: Optional custom filename
        
    Returns:
        Path to saved report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_comparison_{timestamp}.csv'
    
    report_path = os.path.join(save_dir, filename)
    comparison_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ Model comparison report saved: {report_path}")
    return report_path
