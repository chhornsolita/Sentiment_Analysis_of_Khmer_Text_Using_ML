"""
Model Evaluation Module

This module provides functions for evaluating model performance,
generating reports, and visualizing results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from typing import Dict, List, Optional


def evaluate_model(y_true, y_pred, target_names: List[str] = None) -> Dict:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if target_names is None:
        target_names = ["neg", "neu", "pos"]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    
    # Print detailed classification report
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    return metrics


def compare_models(models: Dict, X_test, y_test, le=None) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        le: Label encoder (for XGBoost)
        
    Returns:
        DataFrame with comparison results
    """
    comparison_results = []
    
    for name, model in models.items():
        if name == "XGBoost" and le is not None:
            y_pred = model.predict(X_test)
            y_pred_labels = le.inverse_transform(y_pred)
            y_test_labels = y_test
        else:
            y_pred_labels = model.predict(X_test)
            y_test_labels = y_test
        
        result = {
            'Model': name,
            'Accuracy': accuracy_score(y_test_labels, y_pred_labels),
            'F1-Macro': f1_score(y_test_labels, y_pred_labels, average='macro'),
            'Precision-Macro': precision_score(y_test_labels, y_pred_labels, average='macro'),
            'Recall-Macro': recall_score(y_test_labels, y_pred_labels, average='macro'),
            'Best CV Score': model.best_score_ if hasattr(model, 'best_score_') else 0
        }
        comparison_results.append(result)
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.round(4)
    comparison_df = comparison_df.sort_values('F1-Macro', ascending=False)
    
    return comparison_df


def plot_confusion_matrix(y_true, y_pred, labels: List[str] = None, figsize: tuple = (8, 6)):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        figsize: Figure size
    """
    if labels is None:
        labels = ["neg", "neu", "pos"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame, figsize: tuple = (12, 7)):
    """
    Plot model comparison bar chart.
    
    Args:
        comparison_df: DataFrame with model comparison results
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(comparison_df))
    width = 0.2
    
    metrics = ['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, comparison_df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Khmer Sentiment Analysis - Model Performance Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        for j, value in enumerate(comparison_df[metric]):
            ax.text(j + i*width, value + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_lstm_history(history, figsize: tuple = (14, 5)):
    """
    Plot LSTM training history (accuracy and loss).
    
    Args:
        history: Keras training history object
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('BiLSTM Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('BiLSTM Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_errors(df: pd.DataFrame, y_test, y_pred, X_test, num_examples: int = 10):
    """
    Analyze and display misclassified examples.
    
    Args:
        df: Original DataFrame with text
        y_test: True labels
        y_pred: Predicted labels
        X_test: Test indices
        num_examples: Number of examples to display
    """
    misclassified_idx = X_test.index[y_test != y_pred].tolist()
    
    print("="*80)
    print("SAMPLE MISCLASSIFICATIONS - Understanding Khmer Sentiment Challenges")
    print("="*80)
    
    for i, idx in enumerate(misclassified_idx[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"Text: {df.loc[idx, 'text'][:100]}...")
        if 'text_clean' in df.columns:
            print(f"Cleaned: {df.loc[idx, 'text_clean'][:100]}...")
        print(f"True Sentiment: {y_test.loc[idx]}")
        print(f"Predicted: {y_pred[list(X_test.index).index(idx)]}")
        print("-" * 80)
