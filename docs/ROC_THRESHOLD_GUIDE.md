# ROC Analysis & Threshold Optimization Guide

## 📊 Overview

This guide explains how to use ROC (Receiver Operating Characteristic) analysis and optimal threshold finding in the Khmer Sentiment Analysis project.

## 🎯 What is Threshold Optimization?

In classification, models typically use a default threshold of 0.5 for binary decisions. However, **the optimal threshold depends on your use case**:

- **High Precision**: Use higher threshold (reduce false positives)
- **High Recall**: Use lower threshold (reduce false negatives)
- **Balanced F1**: Find threshold that maximizes F1 score
- **Cost-Sensitive**: Account for different costs of errors

## 🔍 Optimization Methods

### 1. **Youden's J Statistic** (Maximum Separation)
- Maximizes: `TPR - FPR` (True Positive Rate - False Positive Rate)
- Best for: Maximizing separation between classes
- Formula: `J = Sensitivity + Specificity - 1`

### 2. **F1 Score Maximization** (Balanced Performance)
- Maximizes: F1 Score (harmonic mean of precision and recall)
- Best for: Balanced precision and recall
- Recommended for: Most classification tasks

### 3. **Cost-Sensitive** (Business Optimization)
- Minimizes: Total cost of misclassification
- Best for: When false positives and negatives have different costs
- Requires: Domain knowledge of error costs

## 🚀 Usage

### Step 1: Train Model with ROC Analysis

```bash
# Standard training (includes ROC analysis by default)
python train.py
```

**Output:**
- `results/figures/roc_curves_TIMESTAMP.png` - ROC curves for all classes
- `results/figures/threshold_analysis_CLASS_TIMESTAMP.png` - Threshold analysis per class
- `results/reports/threshold_report_TIMESTAMP.csv` - Comprehensive threshold report
- `models/saved_models/optimal_thresholds_TIMESTAMP.json` - Optimal thresholds

### Step 2: View ROC Results

The training script automatically generates:

1. **ROC Curves Plot**: Shows AUC for each class
2. **Threshold Analysis**: Precision, Recall, F1 vs. Threshold
3. **Threshold Report**: Optimal thresholds for each method

Example output:
```
================================================================================
OPTIMAL THRESHOLDS (Youden's Index Method)
================================================================================
        neg: 0.4521
        neu: 0.5234
        pos: 0.3876

================================================================================
OPTIMAL THRESHOLDS (F1 Score Maximization)
================================================================================
        neg: 0.4200
        neu: 0.5500
        pos: 0.3500
```

### Step 3: Make Predictions with Optimal Thresholds

#### Using F1-Optimized Thresholds (Recommended)

```bash
python predict.py \
    --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method f1 \
    --text "អត្ថបទខ្មែរ"
```

#### Using Youden-Optimized Thresholds

```bash
python predict.py \
    --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method youden \
    --text "អត្ថបទខ្មែរ"
```

#### Using Default Thresholds (0.5)

```bash
python predict.py \
    --model_path models/saved_models/best_model_*.pkl \
    --text "អត្ថបទខ្មែរ"
```

### Step 4: Batch Prediction with Probabilities

```bash
python predict.py \
    --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method f1 \
    --input_file data/new_texts.csv \
    --output_file results/predictions_optimized.csv \
    --return_proba
```

This will output:
- `predicted_sentiment`: Final prediction using optimal thresholds
- `prob_neg`, `prob_neu`, `prob_pos`: Raw probabilities for each class

## 📈 Understanding the Plots

### ROC Curve Plot

```
Features:
- One curve per class (One-vs-Rest)
- Diagonal line = random classifier
- Higher curve = better performance
- AUC (Area Under Curve) summarizes performance
```

**Interpretation:**
- AUC = 1.0: Perfect classifier
- AUC = 0.9-1.0: Excellent
- AUC = 0.8-0.9: Good
- AUC = 0.7-0.8: Fair
- AUC = 0.5: Random guessing

### Threshold Analysis Plot

**Left Plot: Precision-Recall vs. Threshold**
- Shows trade-off between precision and recall
- Optimal threshold marked with vertical line

**Right Plot: F1 Score vs. Threshold**
- Shows F1 score at each threshold
- Peak indicates optimal threshold for F1

## 🎓 Practical Examples

### Example 1: Balanced Classification

**Goal:** Maximize overall F1 score

```bash
# Train
python train.py

# Predict with F1 thresholds
python predict.py \
    --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method f1 \
    --input_file data/test.csv \
    --output_file results/predictions_f1.csv
```

### Example 2: High Precision for Positive Class

**Goal:** Minimize false positives for "positive" sentiment

**Solution:**
1. Check threshold report for positive class
2. Increase threshold for positive class
3. Create custom thresholds JSON:

```json
{
  "custom_method": {
    "neg": 0.45,
    "neu": 0.50,
    "pos": 0.70
  }
}
```

### Example 3: Compare Different Thresholds

```bash
# Default thresholds (0.5)
python predict.py --model_path models/saved_models/best_model_*.pkl \
    --input_file data/test.csv --output_file results/pred_default.csv

# F1-optimized
python predict.py --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method f1 \
    --input_file data/test.csv --output_file results/pred_f1.csv

# Youden-optimized
python predict.py --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method youden \
    --input_file data/test.csv --output_file results/pred_youden.csv
```

## 📊 Threshold Report Columns

| Column | Description |
|--------|-------------|
| `Class` | Class name (neg, neu, pos) |
| `AUC` | Area under ROC curve |
| `Threshold_Youden` | Optimal threshold (Youden method) |
| `Youden_J` | Youden's J statistic value |
| `Threshold_F1` | Optimal threshold (F1 method) |
| `Best_F1` | Best F1 score achieved |
| `Default_Threshold` | Default threshold (0.5) |

## 🔧 Programmatic Usage

### Using in Python Code

```python
from src.threshold_optimization import (
    compute_roc_curves_multiclass,
    get_optimal_thresholds_multiclass,
    predict_with_threshold
)
from src.model_persistence import load_model
import json

# Load model
model = load_model('models/saved_models/best_model_*.pkl')

# Get predictions with probabilities
y_pred_proba = model.predict_proba(X_test)

# Compute ROC curves
roc_data = compute_roc_curves_multiclass(y_test, y_pred_proba, ['neg', 'neu', 'pos'])

# Get optimal thresholds
optimal_thresholds = get_optimal_thresholds_multiclass(
    y_test, y_pred_proba, ['neg', 'neu', 'pos'], method='f1'
)

print("Optimal Thresholds:", optimal_thresholds)

# Make predictions with custom thresholds
y_pred_optimized = predict_with_threshold(
    y_pred_proba, 
    optimal_thresholds, 
    ['neg', 'neu', 'pos']
)
```

## 💡 Best Practices

1. **Always visualize ROC curves** before deploying
2. **Use F1-optimized thresholds** as default for balanced tasks
3. **Consider class imbalance** when choosing thresholds
4. **Test multiple thresholds** on validation set
5. **Document threshold choices** in production
6. **Monitor threshold performance** over time
7. **Retrain and re-optimize** when data distribution changes

## ❓ FAQ

**Q: Which threshold method should I use?**
A: For most cases, use F1-optimized thresholds. Use Youden if you want maximum class separation.

**Q: Can I use different thresholds for each class?**
A: Yes! Each class can have its own optimal threshold.

**Q: When should I NOT use custom thresholds?**
A: If your model already performs well (F1 > 0.9) or if you have perfectly balanced classes.

**Q: How do thresholds affect multi-class classification?**
A: Each class gets a threshold. A sample is assigned to the class with the highest score above its threshold.

**Q: Should I optimize thresholds on train or test set?**
A: Always optimize on validation/test set, never on training set to avoid overfitting.

## 📚 References

- Youden, W. J. (1950). "Index for rating diagnostic tests"
- Powers, D. M. (2011). "Evaluation: from precision, recall and F-measure to ROC"
- Fawcett, T. (2006). "An introduction to ROC analysis"

## 🎯 Summary

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **F1** | Balanced classification | Optimal precision-recall trade-off | May not align with business goals |
| **Youden** | Maximum separation | Maximizes TPR-FPR | May favor majority class |
| **Default (0.5)** | Quick baseline | Simple, no optimization needed | Usually suboptimal |
| **Custom** | Specific requirements | Tailored to needs | Requires domain expertise |

**Recommendation:** Start with F1-optimized thresholds, then adjust based on your specific requirements.
