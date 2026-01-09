# Professional Folder Structure Documentation

## 🎯 Overview

This document describes the professional, production-ready folder structure for the Khmer Sentiment Analysis project. The structure follows industry best practices for ML/AI projects.

## 📂 Complete Directory Tree

```
PROJECT/
│
├── data/                           # 📊 Data Directory
│   ├── Data Collection - Sheet1.csv    # Raw collected data
│   └── data_cleaned_all.csv            # Preprocessed dataset
│
├── src/                            # 💻 Source Code (Main Package)
│   ├── __init__.py                     # Package initializer
│   ├── config.py                       # Configuration & hyperparameters
│   ├── preprocessing.py                # Text preprocessing module
│   ├── data_loader.py                  # Data loading & preparation
│   ├── feature_extraction.py          # Feature engineering
│   ├── models.py                       # Traditional ML models
│   ├── deep_learning.py               # Deep learning models (LSTM)
│   ├── evaluation.py                   # Model evaluation & metrics
│   └── model_persistence.py            # Model saving/loading
│
├── models/                         # 🤖 Model Artifacts
│   └── saved_models/                   # Trained model storage
│       ├── best_model_*.pkl                # Serialized ML models
│       ├── best_model_*.keras              # Keras/TensorFlow models
│       ├── tokenizer_*.pkl                 # Text tokenizers
│       ├── label_encoder_*.pkl             # Label encoders
│       └── best_model_metadata_*.json      # Model metadata
│
├── results/                        # 📈 Output & Analysis
│   ├── figures/                        # Plots & visualizations
│   │   ├── confusion_matrix_*.png          # Confusion matrices
│   │   ├── model_comparison_*.png          # Comparison charts
│   │   └── lstm_history_*.png              # Training curves
│   └── reports/                        # Performance reports
│       └── model_comparison_*.csv          # Metrics CSV
│
├── notebooks/                      # 📓 Jupyter Notebooks
│   ├── Model.ipynb                     # Main analysis notebook
│   └── Notebook.ipynb                  # Exploratory notebook
│
├── tests/                          # 🧪 Unit Tests
│   ├── __init__.py                     # Test package init
│   ├── test_preprocessing.py           # Preprocessing tests
│   ├── test_data_loader.py             # Data loader tests
│   ├── test_models.py                  # Model tests
│   └── test_evaluation.py              # Evaluation tests
│
├── docs/                           # 📚 Documentation
│   ├── API_REFERENCE.md                # API documentation
│   ├── USER_GUIDE.md                   # User guide
│   └── DEVELOPMENT.md                  # Development guide
│
├── scripts/                        # 🔧 Utility Scripts (Optional)
│   ├── download_data.py                # Data download script
│   ├── benchmark.py                    # Benchmarking script
│   └── export_model.py                 # Model export utilities
│
├── train.py                        # 🚀 Main Training Script
├── predict.py                      # 🔮 Prediction Script
├── setup.py                        # 📦 Package Setup File
├── requirements.txt                # 📋 Python Dependencies
├── README.md                       # 📖 Project Overview (Original)
├── README_PROFESSIONAL.md          # 📖 Professional README
├── PROJECT_STRUCTURE.md            # 📁 This File
├── .gitignore                      # 🚫 Git Ignore Rules
└── LICENSE                         # ⚖️ License File

```

## 🗂️ Directory Descriptions

### 📊 `/data/`
**Purpose**: Store all datasets (raw and processed)

**Contents**:
- Raw collected data from various sources
- Cleaned and preprocessed datasets
- Train/test splits (if saved separately)
- Data validation reports

**Best Practices**:
- Keep raw data immutable
- Version control for data splits
- Document data sources and preprocessing steps

### 💻 `/src/`
**Purpose**: Main source code package

**Modules**:

1. **`config.py`**: Centralized configuration
   - File paths
   - Hyperparameters
   - Model settings
   - Constants

2. **`preprocessing.py`**: Text preprocessing
   - Khmer Unicode normalization
   - Slang handling
   - Text cleaning functions

3. **`data_loader.py`**: Data operations
   - Load CSV/JSON data
   - Data cleaning
   - Train-test splitting

4. **`feature_extraction.py`**: Feature engineering
   - TF-IDF vectorization
   - Class weight computation
   - Feature transformation

5. **`models.py`**: Traditional ML models
   - Pipeline creation
   - Hyperparameter grids
   - Model training functions

6. **`deep_learning.py`**: Deep learning models
   - LSTM/BiLSTM architectures
   - Sequence preparation
   - Model training

7. **`evaluation.py`**: Model evaluation
   - Metrics calculation
   - Model comparison
   - Visualization functions

8. **`model_persistence.py`**: Persistence layer
   - Model saving/loading
   - Metadata management
   - Report generation

### 🤖 `/models/saved_models/`
**Purpose**: Store trained model artifacts

**Contents**:
- Serialized models (.pkl, .keras)
- Tokenizers and encoders
- Model metadata (JSON)
- Version timestamps

**Naming Convention**:
```
best_model_{model_name}_{timestamp}.{extension}
best_model_metadata_{timestamp}.json
tokenizer_{type}_{timestamp}.pkl
```

### 📈 `/results/`
**Purpose**: Store analysis outputs and reports

**Subdirectories**:
- `figures/`: Plots and visualizations
- `reports/`: CSV reports and metrics

**Best Practices**:
- Timestamp all outputs
- Organize by experiment/run
- Keep figures in publication-ready format

### 📓 `/notebooks/`
**Purpose**: Exploratory analysis and experimentation

**Usage**:
- Interactive data exploration
- Prototyping new features
- Visualization experiments
- Should import from `src/` modules

### 🧪 `/tests/`
**Purpose**: Unit tests for code quality

**Structure**:
- Mirror `src/` structure
- Each module has corresponding test file
- Use pytest framework

**Run Tests**:
```bash
pytest tests/ -v
pytest tests/ --cov=src
```

### 📚 `/docs/`
**Purpose**: Project documentation

**Contents**:
- API reference
- User guides
- Development guidelines
- Architecture documentation

## 🚀 Usage Workflows

### Workflow 1: Training Pipeline
```bash
# Step 1: Configure settings
vim src/config.py

# Step 2: Run training
python train.py

# Step 3: Check results
ls results/reports/
```

### Workflow 2: Making Predictions
```bash
# Single prediction
python predict.py --model_path models/saved_models/best_model_*.pkl \
                  --text "អត្ថបទខ្មែរ"

# Batch prediction
python predict.py --model_path models/saved_models/best_model_*.pkl \
                  --input_file data/new_data.csv \
                  --output_file results/predictions.csv
```

### Workflow 3: Development
```bash
# Step 1: Create new feature branch
git checkout -b feature/new-model

# Step 2: Add new model to src/models.py
vim src/models.py

# Step 3: Write tests
vim tests/test_models.py

# Step 4: Run tests
pytest tests/

# Step 5: Train with new model
python train.py
```

## 📋 File Naming Conventions

### Code Files
- Use snake_case: `feature_extraction.py`
- Descriptive names: `model_persistence.py` not `utils.py`
- Test files: `test_{module_name}.py`

### Data Files
- Raw data: `data_raw_{source}_{date}.csv`
- Cleaned data: `data_cleaned_{version}.csv`
- Versioning: Use dates or semantic versions

### Model Files
- Include timestamp: `model_{type}_{timestamp}.pkl`
- Metadata: `metadata_{timestamp}.json`
- Descriptive names: `best_model_logistic_regression_20241226.pkl`

### Result Files
- Timestamp required: `comparison_20241226_143022.csv`
- Descriptive prefix: `confusion_matrix_lstm_*.png`

## 🔧 Configuration Management

All configurations in `src/config.py`:

```python
# Paths
DATA_DIR = 'data/'
MODELS_DIR = 'models/saved_models/'
RESULTS_DIR = 'results/'

# Hyperparameters
TFIDF_MAX_FEATURES = 5000
LSTM_UNITS = 64
CV_FOLDS = 3

# Model Settings
SCORING_METRIC = 'f1_macro'
RANDOM_STATE = 42
```

## 🎯 Benefits of This Structure

1. **Modularity**: Each component has single responsibility
2. **Reusability**: Functions can be imported and reused
3. **Testability**: Easy to write and run unit tests
4. **Scalability**: Easy to add new models/features
5. **Collaboration**: Clear structure for team development
6. **Reproducibility**: Version control and configuration management
7. **Production-Ready**: Can be deployed as package
8. **Documentation**: Self-documenting code structure

## 📦 Package Installation

Install as editable package for development:

```bash
pip install -e .
```

Install with extras:

```bash
# With deep learning dependencies
pip install -e .[deep_learning]

# With development tools
pip install -e .[dev]
```

## 🔄 Version Control

### What to commit:
- All code in `src/`
- Scripts (`train.py`, `predict.py`)
- Tests
- Documentation
- Configuration files
- `requirements.txt`
- `setup.py`

### What NOT to commit:
- Data files (use `.gitignore`)
- Model artifacts (too large)
- Results/outputs
- `__pycache__/`
- `.ipynb_checkpoints/`
- IDE configuration

### Sample `.gitignore`:
```
# Data
data/*
!data/.gitkeep

# Models
models/saved_models/*
!models/saved_models/.gitkeep

# Results
results/figures/*
results/reports/*
!results/**/.gitkeep

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

## 🎓 Learning Resources

- **Cookiecutter Data Science**: Industry-standard project structure
- **Scikit-learn**: Best practices for ML pipelines
- **TensorFlow**: Model organization guidelines
- **Python Packaging**: Creating installable packages

---

**Maintained by**: [Your Name]  
**Last Updated**: December 26, 2025  
**Version**: 1.0
