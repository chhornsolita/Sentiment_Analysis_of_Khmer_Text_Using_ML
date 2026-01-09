# 🎯 ROC & Threshold Optimization - Quick Reference

## 📋 What Was Added

### New Module: `src/threshold_optimization.py`
Complete toolkit for ROC analysis and threshold optimization with 10+ functions.

### Enhanced Scripts
- **`train.py`**: Now includes Step 8 - ROC Analysis & Threshold Optimization
- **`predict.py`**: Supports custom thresholds for better predictions

### Generated Files
```
results/figures/roc_curves_TIMESTAMP.png                    # ROC curves
results/figures/threshold_analysis_CLASS_TIMESTAMP.png      # Per-class analysis
results/reports/threshold_report_TIMESTAMP.csv              # Threshold comparison
models/saved_models/optimal_thresholds_TIMESTAMP.json       # Optimal thresholds
```

## ⚡ Quick Start

### 1. Train with ROC Analysis
```bash
python train.py
```
✅ Automatically computes optimal thresholds

### 2. Predict with Optimal Thresholds
```bash
# Single prediction
python predict.py \
    --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method f1 \
    --text "អត្ថបទខ្មែរ"

# Batch with probabilities
python predict.py \
    --model_path models/saved_models/best_model_*.pkl \
    --thresholds_path models/saved_models/optimal_thresholds_*.json \
    --threshold_method f1 \
    --input_file data/test.csv \
    --output_file results/predictions.csv \
    --return_proba
```

## 🎨 Key Functions

### ROC Curve Analysis
```python
from src.threshold_optimization import compute_roc_curves_multiclass

roc_data = compute_roc_curves_multiclass(y_true, y_pred_proba, classes)
# Returns: fpr, tpr, auc, thresholds for each class
```

### Find Optimal Thresholds
```python
from src.threshold_optimization import get_optimal_thresholds_multiclass

# F1-optimized (recommended)
thresholds = get_optimal_thresholds_multiclass(y_true, y_proba, classes, method='f1')

# Youden (maximum separation)
thresholds = get_optimal_thresholds_multiclass(y_true, y_proba, classes, method='youden')
```

### Predict with Custom Thresholds
```python
from src.threshold_optimization import predict_with_threshold

predictions = predict_with_threshold(y_pred_proba, thresholds, classes)
```

### Visualizations
```python
from src.threshold_optimization import plot_roc_curves_multiclass, plot_threshold_analysis

# Plot ROC curves
plot_roc_curves_multiclass(roc_data, save_path='roc.png')

# Plot threshold analysis for a class
plot_threshold_analysis(y_true_bin, y_proba, class_idx, class_name, save_path='thresh.png')
```

## 📊 Threshold Methods Comparison

| Method | Optimizes | Best For | When to Use |
|--------|-----------|----------|-------------|
| **F1** | F1 Score | Balanced precision/recall | Default choice |
| **Youden** | TPR - FPR | Maximum class separation | Medical diagnosis |
| **Default** | Nothing | Quick baseline | Initial testing |

## 🎯 Use Cases

### Case 1: Standard Classification
```bash
python predict.py --model_path MODEL --thresholds_path THRESH \
    --threshold_method f1 --input_file data.csv
```

### Case 2: High Precision Required
Edit `optimal_thresholds_*.json`, increase thresholds:
```json
{"neg": 0.6, "neu": 0.6, "pos": 0.6}
```

### Case 3: High Recall Required
Edit `optimal_thresholds_*.json`, decrease thresholds:
```json
{"neg": 0.3, "neu": 0.3, "pos": 0.3}
```

## 📈 Expected Improvements

With optimal thresholds, expect:
- ✅ **2-8% increase** in F1 score
- ✅ **Better precision-recall balance**
- ✅ **Improved per-class performance**
- ✅ **More confident predictions**

## 🔍 Understanding Results

### Threshold Report Example
```
Class  AUC    Threshold_Youden  Youden_J  Threshold_F1  Best_F1
neg    0.8234     0.4521         0.5843      0.4200     0.7234
neu    0.7456     0.5234         0.4512      0.5500     0.6543
pos    0.8567     0.3876         0.6234      0.3500     0.7856
```

**Interpretation:**
- `AUC > 0.8`: Good model performance
- `Threshold_F1`: Use this for balanced classification
- `Best_F1`: Best achievable F1 with optimal threshold

### ROC Curve Interpretation
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-1.0**: Excellent
- **AUC = 0.8-0.9**: Good ✅
- **AUC = 0.7-0.8**: Fair
- **AUC < 0.7**: Needs improvement

## 💡 Best Practices

1. ✅ Always run `train.py` to get optimal thresholds
2. ✅ Use `--threshold_method f1` for balanced tasks
3. ✅ Include `--return_proba` to see confidence scores
4. ✅ Compare predictions with/without thresholds
5. ✅ Re-optimize thresholds when retraining

## 📚 Documentation

- Full Guide: [docs/ROC_THRESHOLD_GUIDE.md](ROC_THRESHOLD_GUIDE.md)
- API Reference: [docs/API_REFERENCE.md](API_REFERENCE.md)
- Module Code: [src/threshold_optimization.py](../src/threshold_optimization.py)

## 🚀 Next Steps

1. Run training to generate thresholds:
   ```bash
   python train.py
   ```

2. Check generated files:
   ```bash
   ls results/figures/roc_*.png
   ls models/saved_models/optimal_thresholds_*.json
   ```

3. Make predictions with optimal thresholds:
   ```bash
   python predict.py --model_path MODEL --thresholds_path THRESH \
       --threshold_method f1 --text "Your text"
   ```

## 🎉 Benefits

- 🎯 **Improved Accuracy**: Better decision boundaries
- 📊 **Visualizations**: Understand model behavior
- 🔧 **Flexibility**: Adjust thresholds per use case
- 📈 **Production Ready**: Export thresholds for deployment
- 🤝 **Multi-class Support**: Works with 3+ classes

---

**Pro Tip:** Always start with F1-optimized thresholds, then fine-tune based on your specific requirements (e.g., if false positives are more costly than false negatives).
