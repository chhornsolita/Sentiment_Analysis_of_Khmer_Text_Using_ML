"""
ROC Analysis and Threshold Optimization Module

This module provides functions for ROC curve analysis, threshold optimization,
and finding the best decision thresholds for classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


def compute_roc_curves_multiclass(
    y_true,
    y_pred_proba,
    classes: List[str] = None
) -> Dict:
    """
    Compute ROC curves for multi-class classification (One-vs-Rest).
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        classes: List of class names
        
    Returns:
        Dictionary with ROC data for each class
    """
    from sklearn.preprocessing import label_binarize
    
    if classes is None:
        classes = np.unique(y_true)
    
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(
            y_true_bin[:, i], 
            y_pred_proba[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(),
        y_pred_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'thresholds': thresholds,
        'classes': classes
    }


def find_optimal_threshold_youden(fpr, tpr, thresholds) -> Tuple[float, float]:
    """
    Find optimal threshold using Youden's J statistic (maximizes TPR - FPR).
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Decision thresholds
        
    Returns:
        Tuple of (optimal_threshold, youden_index)
    """
    # Youden's J statistic = Sensitivity + Specificity - 1 = TPR - FPR
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, youden_index[optimal_idx]


def find_optimal_threshold_f1(y_true, y_pred_proba, class_idx: int = 1) -> Tuple[float, float]:
    """
    Find optimal threshold by maximizing F1 score.
    
    Args:
        y_true: True labels (binarized for specific class)
        y_pred_proba: Predicted probabilities for the class
        class_idx: Class index (for display)
        
    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    best_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, best_f1


def find_optimal_threshold_cost_sensitive(
    fpr, tpr, thresholds,
    fp_cost: float = 1.0,
    fn_cost: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal threshold for cost-sensitive classification.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Decision thresholds
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative
        
    Returns:
        Tuple of (optimal_threshold, min_cost)
    """
    # Cost = FP_cost * FPR + FN_cost * FNR
    # FNR = 1 - TPR
    fnr = 1 - tpr
    total_cost = fp_cost * fpr + fn_cost * fnr
    
    optimal_idx = np.argmin(total_cost)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = total_cost[optimal_idx]
    
    return optimal_threshold, min_cost


def get_optimal_thresholds_multiclass(
    y_true,
    y_pred_proba,
    classes: List[str] = None,
    method: str = 'youden'
) -> Dict:
    """
    Get optimal thresholds for all classes in multi-class classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: List of class names
        method: Optimization method ('youden', 'f1', 'cost')
        
    Returns:
        Dictionary with optimal thresholds for each class
    """
    from sklearn.preprocessing import label_binarize
    
    if classes is None:
        classes = np.unique(y_true)
    
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    optimal_thresholds = {}
    
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        
        if method == 'youden':
            opt_threshold, _ = find_optimal_threshold_youden(fpr, tpr, thresholds)
        elif method == 'f1':
            opt_threshold, _ = find_optimal_threshold_f1(
                y_true_bin[:, i], 
                y_pred_proba[:, i]
            )
        elif method == 'cost':
            opt_threshold, _ = find_optimal_threshold_cost_sensitive(
                fpr, tpr, thresholds
            )
        else:
            opt_threshold = 0.5  # Default
        
        optimal_thresholds[classes[i]] = opt_threshold
    
    return optimal_thresholds


def predict_with_threshold(
    y_pred_proba,
    thresholds: Dict,
    classes: List[str]
) -> np.ndarray:
    """
    Make predictions using custom thresholds for each class.
    
    Args:
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        thresholds: Dictionary mapping class names to thresholds
        classes: List of class names
        
    Returns:
        Array of predicted class labels
    """
    n_samples = y_pred_proba.shape[0]
    predictions = []
    
    for i in range(n_samples):
        # Apply threshold to each class
        class_scores = []
        for j, cls in enumerate(classes):
            threshold = thresholds.get(cls, 0.5)
            # Score is probability if above threshold, 0 otherwise
            score = y_pred_proba[i, j] if y_pred_proba[i, j] >= threshold else 0
            class_scores.append(score)
        
        # Predict class with highest score above threshold
        if max(class_scores) > 0:
            pred_class_idx = np.argmax(class_scores)
        else:
            # If no class above threshold, use highest probability
            pred_class_idx = np.argmax(y_pred_proba[i])
        
        predictions.append(classes[pred_class_idx])
    
    return np.array(predictions)


def plot_roc_curves_multiclass(
    roc_data: Dict,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        roc_data: Dictionary from compute_roc_curves_multiclass
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    roc_auc = roc_data['roc_auc']
    classes = roc_data['classes']
    
    plt.figure(figsize=figsize)
    
    # Plot ROC curve for each class
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, cls in enumerate(classes):
        color = colors[i % len(colors)]
        plt.plot(
            fpr[i], tpr[i],
            color=color,
            lw=2,
            label=f'{cls} (AUC = {roc_auc[i]:.3f})'
        )
    
    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"], tpr["micro"],
        color='deeppink',
        linestyle='--',
        lw=2,
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})'
    )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved: {save_path}")
    
    plt.show()


def plot_threshold_analysis(
    y_true,
    y_pred_proba,
    class_idx: int,
    class_name: str,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
):
    """
    Plot threshold analysis showing precision, recall, and F1 vs threshold.
    
    Args:
        y_true: True labels (binarized)
        y_pred_proba: Predicted probabilities for the class
        class_idx: Class index
        class_name: Class name
        figsize: Figure size
        save_path: Path to save figure
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores for different thresholds
    thresholds_range = np.linspace(0, 1, 100)
    f1_scores = []
    
    for threshold in thresholds_range:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        f1_scores.append(f1)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_range[optimal_idx]
    best_f1 = f1_scores[optimal_idx]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Precision-Recall vs Threshold
    ax1.plot(thresholds_pr, precision[:-1], 'b-', label='Precision', linewidth=2)
    ax1.plot(thresholds_pr, recall[:-1], 'r-', label='Recall', linewidth=2)
    ax1.axvline(optimal_threshold, color='green', linestyle='--', 
                label=f'Optimal (F1={best_f1:.3f})', linewidth=2)
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Precision-Recall vs Threshold\nClass: {class_name}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: F1 Score vs Threshold
    ax2.plot(thresholds_range, f1_scores, 'g-', linewidth=2)
    ax2.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal = {optimal_threshold:.3f}', linewidth=2)
    ax2.axhline(best_f1, color='blue', linestyle=':', 
                label=f'Max F1 = {best_f1:.3f}', linewidth=1)
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title(f'F1 Score vs Threshold\nClass: {class_name}', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Threshold analysis saved: {save_path}")
    
    plt.show()
    
    return optimal_threshold, best_f1


def generate_threshold_report(
    y_true,
    y_pred_proba,
    classes: List[str],
    methods: List[str] = ['youden', 'f1']
) -> pd.DataFrame:
    """
    Generate comprehensive threshold report with multiple optimization methods.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: List of class names
        methods: List of optimization methods
        
    Returns:
        DataFrame with threshold recommendations
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    
    y_true_bin = label_binarize(y_true, classes=classes)
    
    results = []
    
    for i, cls in enumerate(classes):
        row = {'Class': cls}
        
        # Compute AUC
        try:
            row['AUC'] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        except:
            row['AUC'] = 0.0
        
        # Get optimal thresholds using different methods
        fpr, tpr, thresholds_roc = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        
        if 'youden' in methods:
            opt_thresh_youden, youden_j = find_optimal_threshold_youden(
                fpr, tpr, thresholds_roc
            )
            row['Threshold_Youden'] = opt_thresh_youden
            row['Youden_J'] = youden_j
        
        if 'f1' in methods:
            opt_thresh_f1, best_f1 = find_optimal_threshold_f1(
                y_true_bin[:, i], 
                y_pred_proba[:, i]
            )
            row['Threshold_F1'] = opt_thresh_f1
            row['Best_F1'] = best_f1
        
        row['Default_Threshold'] = 0.5
        
        results.append(row)
    
    df = pd.DataFrame(results)
    return df.round(4)
