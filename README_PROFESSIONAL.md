# Khmer Sentiment Analysis - Professional ML Project

A production-ready sentiment analysis system for Khmer text using traditional machine learning and deep learning approaches.

## 🎯 Project Overview

This project implements a comprehensive sentiment analysis pipeline for Khmer language text, addressing unique challenges such as Unicode normalization, informal writing styles, and limited NLP resources. The system trains and compares multiple models to achieve optimal performance.

## 📁 Professional Folder Structure

```
PROJECT/
│
├── data/                           # Dataset storage
│   ├── Data Collection - Sheet1.csv
│   └── data_cleaned_all.csv
│
├── src/                            # Source code modules
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Configuration & hyperparameters
│   ├── preprocessing.py            # Text preprocessing (Khmer-specific)
│   ├── data_loader.py              # Data loading & preparation
│   ├── feature_extraction.py      # TF-IDF & feature engineering
│   ├── models.py                   # Traditional ML models
│   ├── deep_learning.py            # LSTM/BiLSTM models
│   ├── evaluation.py               # Model evaluation & visualization
│   ├── model_persistence.py        # Model saving/loading
│   └── threshold_optimization.py   # 🆕 ROC & threshold optimization
│
├── models/                         # Saved model artifacts
│   └── saved_models/
│       ├── best_model_*.pkl            # Trained models
│       ├── best_model_metadata_*.json
│       ├── tokenizer_*.pkl             # Preprocessing objects
│       └── optimal_thresholds_*.json   # 🆕 Optimal decision thresholds
│
├── results/                        # Analysis outputs
│   ├── figures/                        # Visualizations
│   │   ├── roc_curves_*.png            # 🆕 ROC curve plots
│   │   └── threshold_analysis_*.png    # 🆕 Threshold optimization
│   └── reports/                        # Performance reports
│       ├── model_comparison_*.csv
│       └── threshold_report_*.csv      # 🆕 Threshold recommendations
│
├── notebooks/                      # Jupyter notebooks
│   ├── Model.ipynb                 # Original analysis notebook
│   └── Notebook.ipynb              # Exploratory analysis
│
├── tests/                          # Unit tests (optional)
│
├── train.py                        # Main training script
├── predict.py                      # Prediction script
├── requirements.txt                # Python dependencies
├── PROJECT_STRUCTURE.md            # Detailed documentation
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd PROJECT

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train all models (including LSTM)
python train.py

# Train without LSTM
python train.py --no_lstm

# Train with custom data path
python train.py --data_path path/to/your/data.csv
```

### Making Predictions

```bash
# Single text prediction
python predict.py --model_path models/saved_models/best_model_*.pkl --text "អស្ចារ្យណាស់!"

# Batch prediction from CSV
python predict.py --model_path models/saved_models/best_model_*.pkl \
                  --input_file data/new_texts.csv \
                  --output_file results/predictions.csv

# 🆕 Prediction with optimal thresholds (improved accuracy)
python predict.py --model_path models/saved_models/best_model_*.pkl \
                  --thresholds_path models/saved_models/optimal_thresholds_*.json \
                  --threshold_method f1 \
                  --input_file data/new_texts.csv \
                  --output_file results/predictions_optimized.csv \
                  --return_proba
```

## 🛠️ Module Documentation

### src/preprocessing.py
- **`preprocess_khmer(text)`**: Khmer-specific text preprocessing
  - Unicode normalization (NFC)
  - Slang handling
  - URL and emoji removal
  - Character normalization

### src/data_loader.py
- **`load_data(file_path)`**: Load dataset from CSV
- **`clean_data(df)`**: Remove invalid entries
- **`prepare_train_test_split()`**: Stratified train-test split

### src/feature_extraction.py
- **`create_tfidf_vectorizer()`**: Configure TF-IDF
- **`compute_class_weights()`**: Handle class imbalance

### src/models.py
- **Pipeline creation** for 5 ML models:
  - Logistic Regression
  - Support Vector Machine (LinearSVC)
  - Naive Bayes (MultinomialNB)
  - Random Forest
  - XGBoost (optional)
- **`train_model_with_search()`**: RandomizedSearchCV training

### src/deep_learning.py
- **`create_lstm_model()`**: Bidirectional LSTM architecture
- **`prepare_sequences()`**: Tokenization & padding
- **`train_lstm_model()`**: Training with early stopping

### src/evaluation.py
- **`evaluate_model()`**: Comprehensive metrics
- **`compare_models()`**: Multi-model comparison
- **`plot_confusion_matrix()`**: Visualization
- **`analyze_errors()`**: Misclassification analysis

### src/model_persistence.py
- **`save_model()`**: Save model with metadata
- **`load_model()`**: Load trained model
- **`save_comparison_report()`**: Export results to CSV

### src/threshold_optimization.py 🆕
- **`compute_roc_curves_multiclass()`**: ROC curve analysis
- **`get_optimal_thresholds_multiclass()`**: Find optimal thresholds
- **`predict_with_threshold()`**: Predictions with custom thresholds
- **`plot_roc_curves_multiclass()`**: Visualize ROC curves
- **`plot_threshold_analysis()`**: Threshold vs metrics plots
- **`generate_threshold_report()`**: Comprehensive threshold report

### src/config.py
- Centralized configuration
- All hyperparameters in one place
- Easy to modify settings

## 📊 Models & Performance

| Model | Type | F1-Macro | Accuracy |
|-------|------|----------|----------|
| Logistic Regression | Traditional ML | - | - |
| SVM | Traditional ML | - | - |
| Naive Bayes | Traditional ML | - | - |
| Random Forest | Traditional ML | - | - |
| XGBoost | Gradient Boosting | - | - |
| BiLSTM | Deep Learning | - | - |

*Run `python train.py` to get updated results*

## 🔧 Configuration

Edit `src/config.py` to customize:
- Data paths
- Hyperparameters
- Feature extraction settings
- Model training parameters

```python
# Example configuration
TFIDF_MAX_FEATURES = 5000
LSTM_UNITS = 64
CV_FOLDS = 3
```

## 📈 Usage Examples

### Training with Custom Settings

```python
from src.config import *
from src.data_loader import load_data, prepare_train_test_split
from src.models import create_logistic_regression_pipeline

# Load data
df = load_data(CLEANED_DATA_PATH)

# Prepare data
X_train, X_test, y_train, y_test = prepare_train_test_split(df)

# Train model
from src.feature_extraction import create_tfidf_vectorizer
tfidf = create_tfidf_vectorizer()
pipeline = create_logistic_regression_pipeline(tfidf, class_weight)
```

### Loading and Using Saved Models

```python
from src.model_persistence import load_model
from src.preprocessing import preprocess_khmer

# Load model
model = load_model('models/saved_models/best_model_logistic_regression_*.pkl')

# Predict
text = "អស្ចារ្យណាស់!"
cleaned = preprocess_khmer(text)
prediction = model.predict([cleaned])
print(f"Sentiment: {prediction[0]}")
```OC Analysis & Threshold Optimization**: 🆕 Find optimal decision boundaries
6. **Reproducible**: Fixed random seeds, version control
7. **Production-Ready**: Command-line scripts for deployment
8. **Comprehensive Evaluation**: Multiple metrics + visualizations
9. **Easy Configuration**: Centralized settings
10. **Optimal Thresholds**: 🆕 Improve accuracy with custom thresholds

## 🆕 ROC & Threshold Optimization

The training pipeline now includes **automatic threshold optimization** to improve classification performance:

- **ROC Curves**: Visualize model performance across all classes
- **Optimal Thresholds**: Find best decision boundaries using:
  - **F1 Score Maximization** (recommended for balanced tasks)
  - **Youden's J Statistic** (maximum class separation)
- **Threshold Analysis**: Precision, Recall, F1 vs. Threshold plots
- **Custom Predictions**: Use optimized thresholds for better results

**Expected Improvement**: 2-8% increase in F1 score

See [ROC & Threshold Guide](docs/ROC_THRESHOLD_GUIDE.md) for details.
1. **Modular Architecture**: Clean separation of concerns
2. **Khmer-Specific Preprocessing**: Handles Unicode and slang
3. **Multiple Models**: Compare 6 different approaches
4. **Automated Model Selection**: Best model based on F1-Macro
5. **Reproducible**: Fixed random seeds, version control
6. **Production-Ready**: Command-line scripts for deployment
7. **Comprehensive Evaluation**: Multiple metrics + visualizations
8. **Easy Configuration**: Centralized settings

## 📝 Citation

If you use this project, please cite:

```
Khmer Sentiment Analysis System
Version: 1.0
Date: December 2025
```

## 🤝 Contributing

1. Follow the modular structure
2. Add docstrings to all functions
3. Update configuration in `config.py`
4. Add tests for new features

## 📄 License

[Add your license here]

## 📧 Contact

[Add your contact information]

---

**Built with** 🧡 **for the Khmer NLP community**
