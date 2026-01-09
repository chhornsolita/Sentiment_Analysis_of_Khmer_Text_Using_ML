# API Reference - Khmer Sentiment Analysis

Complete API documentation for all modules and functions.

## Table of Contents
1. [preprocessing](#preprocessing)
2. [data_loader](#data_loader)
3. [feature_extraction](#feature_extraction)
4. [models](#models)
5. [deep_learning](#deep_learning)
6. [evaluation](#evaluation)
7. [model_persistence](#model_persistence)
8. [config](#config)

---

## preprocessing

### `preprocess_khmer(text, slang_dict=None)`
Preprocess Khmer text for sentiment analysis.

**Parameters:**
- `text` (str): Raw Khmer text string
- `slang_dict` (dict, optional): Custom slang dictionary. Defaults to `KHMER_SLANG`

**Returns:**
- str: Cleaned and normalized Khmer text

**Example:**
```python
from src.preprocessing import preprocess_khmer

text = "អត់ល្អទេ http://example.com 😊"
cleaned = preprocess_khmer(text)
print(cleaned)  # Output: "មិនល្អទេ"
```

### `batch_preprocess(texts, slang_dict=None)`
Apply preprocessing to multiple texts.

**Parameters:**
- `texts` (list): List of raw text strings
- `slang_dict` (dict, optional): Custom slang dictionary

**Returns:**
- list: List of preprocessed texts

---

## data_loader

### `load_data(file_path, encoding='utf-8')`
Load dataset from CSV file.

**Parameters:**
- `file_path` (str): Path to CSV file
- `encoding` (str): File encoding

**Returns:**
- pd.DataFrame: Loaded dataset

### `clean_data(df, text_column='text')`
Clean dataset by removing invalid entries.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `text_column` (str): Name of text column

**Returns:**
- pd.DataFrame: Cleaned DataFrame

### `prepare_train_test_split(df, text_column='text_clean', target_column='target', test_size=0.2, random_state=42, stratify=True)`
Split data into training and testing sets.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `text_column` (str): Text column name
- `target_column` (str): Target column name
- `test_size` (float): Test set proportion
- `random_state` (int): Random seed
- `stratify` (bool): Stratify by target

**Returns:**
- tuple: (X_train, X_test, y_train, y_test)

---

## feature_extraction

### `create_tfidf_vectorizer(max_features=5000, ngram_range=(1,2), min_df=2)`
Create TF-IDF vectorizer.

**Parameters:**
- `max_features` (int): Maximum number of features
- `ngram_range` (tuple): N-gram range
- `min_df` (int): Minimum document frequency

**Returns:**
- TfidfVectorizer: Configured vectorizer

### `compute_class_weights(y_train, classes=None)`
Compute class weights for imbalanced data.

**Parameters:**
- `y_train`: Training labels
- `classes` (list, optional): Class names

**Returns:**
- dict: Class weight dictionary

---

## models

### Pipeline Creation Functions

#### `create_logistic_regression_pipeline(tfidf, class_weight)`
Create Logistic Regression pipeline.

#### `create_svm_pipeline(tfidf, class_weight)`
Create Linear SVM pipeline.

#### `create_naive_bayes_pipeline(tfidf)`
Create Naive Bayes pipeline.

#### `create_random_forest_pipeline(tfidf, class_weight)`
Create Random Forest pipeline.

#### `create_xgboost_pipeline(tfidf)`
Create XGBoost pipeline.

**Parameters:**
- `tfidf`: TfidfVectorizer instance
- `class_weight` (dict): Class weight dictionary

**Returns:**
- Pipeline: sklearn Pipeline object

### `get_hyperparameter_grids()`
Get hyperparameter search grids for all models.

**Returns:**
- dict: Dictionary mapping model names to parameter grids

### `train_model_with_search(pipeline, param_grid, X_train, y_train, n_iter=10, cv=3, scoring='f1_macro', random_state=42)`
Train model with hyperparameter tuning.

**Parameters:**
- `pipeline`: sklearn Pipeline
- `param_grid` (dict): Parameter search space
- `X_train`: Training features
- `y_train`: Training labels
- `n_iter` (int): Number of iterations
- `cv` (int): Cross-validation folds
- `scoring` (str): Scoring metric
- `random_state` (int): Random seed

**Returns:**
- RandomizedSearchCV: Trained model

---

## deep_learning

### `create_lstm_model(max_words, max_len, embedding_dim=128, lstm_units=64, num_classes=3)`
Create Bidirectional LSTM model.

**Parameters:**
- `max_words` (int): Maximum vocabulary size
- `max_len` (int): Maximum sequence length
- `embedding_dim` (int): Embedding dimension
- `lstm_units` (int): LSTM units
- `num_classes` (int): Number of classes

**Returns:**
- Sequential: Compiled Keras model

### `prepare_sequences(texts, max_words=5000, max_len=100, tokenizer=None)`
Prepare text sequences for LSTM.

**Parameters:**
- `texts`: List of text strings
- `max_words` (int): Max vocabulary size
- `max_len` (int): Max sequence length
- `tokenizer`: Existing tokenizer (optional)

**Returns:**
- tuple: (padded_sequences, tokenizer)

### `train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, patience=3)`
Train LSTM with early stopping.

**Parameters:**
- `model`: Keras model
- `X_train`: Training sequences
- `y_train`: Training labels
- `X_val`: Validation sequences
- `y_val`: Validation labels
- `epochs` (int): Max epochs
- `batch_size` (int): Batch size
- `patience` (int): Early stopping patience

**Returns:**
- History: Training history

---

## evaluation

### `evaluate_model(y_true, y_pred, target_names=['neg','neu','pos'])`
Evaluate model performance.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `target_names` (list): Class names

**Returns:**
- dict: Performance metrics

### `compare_models(models, X_test, y_test, le=None)`
Compare multiple models.

**Parameters:**
- `models` (dict): Dictionary of trained models
- `X_test`: Test features
- `y_test`: Test labels
- `le`: Label encoder (for XGBoost)

**Returns:**
- pd.DataFrame: Comparison results

### `plot_confusion_matrix(y_true, y_pred, labels=['neg','neu','pos'], figsize=(8,6))`
Plot confusion matrix heatmap.

### `plot_model_comparison(comparison_df, figsize=(12,7))`
Plot model comparison bar chart.

### `plot_lstm_history(history, figsize=(14,5))`
Plot LSTM training history.

### `analyze_errors(df, y_test, y_pred, X_test, num_examples=10)`
Analyze misclassified examples.

---

## model_persistence

### `save_model(model, model_name, model_type, performance, hyperparameters, save_dir='models/saved_models', tokenizer=None, label_encoder=None)`
Save model with metadata.

**Parameters:**
- `model`: Trained model
- `model_name` (str): Model name
- `model_type` (str): 'traditional_ml' or 'deep_learning'
- `performance` (dict): Performance metrics
- `hyperparameters` (dict): Hyperparameters
- `save_dir` (str): Save directory
- `tokenizer`: Tokenizer (optional)
- `label_encoder`: Label encoder (optional)

**Returns:**
- dict: Paths to saved files

### `load_model(model_path, model_type='traditional_ml')`
Load saved model.

**Parameters:**
- `model_path` (str): Path to model file
- `model_type` (str): Model type

**Returns:**
- Model object

### `load_preprocessing_objects(tokenizer_path=None, le_path=None)`
Load tokenizer and label encoder.

**Returns:**
- tuple: (tokenizer, label_encoder)

### `save_comparison_report(comparison_df, save_dir='results/reports', filename=None)`
Save comparison report to CSV.

**Returns:**
- str: Path to saved report

---

## config

Configuration constants:

### Paths
- `BASE_DIR`: Project root directory
- `DATA_DIR`: Data directory
- `MODELS_DIR`: Models directory
- `RESULTS_DIR`: Results directory

### Data Settings
- `TEXT_COLUMN`: 'text'
- `TARGET_COLUMN`: 'target'
- `CLEAN_TEXT_COLUMN`: 'text_clean'

### Training Parameters
- `TEST_SIZE`: 0.2
- `RANDOM_STATE`: 42
- `CV_FOLDS`: 3
- `N_ITER_SEARCH`: 10
- `SCORING_METRIC`: 'f1_macro'

### Feature Extraction
- `TFIDF_MAX_FEATURES`: 5000
- `TFIDF_NGRAM_RANGE`: (1, 2)
- `TFIDF_MIN_DF`: 2

### Deep Learning
- `MAX_WORDS`: 5000
- `MAX_LEN`: 100
- `EMBEDDING_DIM`: 128
- `LSTM_UNITS`: 64
- `LSTM_EPOCHS`: 20
- `LSTM_BATCH_SIZE`: 32
- `LSTM_PATIENCE`: 3

### Classes
- `CLASS_LABELS`: ["neg", "neu", "pos"]
- `NUM_CLASSES`: 3

---

## Complete Example

```python
# Import modules
from src import (
    load_data, clean_data, prepare_train_test_split,
    preprocess_khmer, create_tfidf_vectorizer,
    create_logistic_regression_pipeline, train_model_with_search,
    evaluate_model, save_model
)
from src.config import *

# 1. Load and preprocess data
df = load_data(CLEANED_DATA_PATH)
df = clean_data(df)
df[CLEAN_TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_khmer)

# 2. Split data
X_train, X_test, y_train, y_test = prepare_train_test_split(df)

# 3. Create features
tfidf = create_tfidf_vectorizer()
from src.feature_extraction import compute_class_weights
class_weight = compute_class_weights(y_train)

# 4. Train model
pipeline = create_logistic_regression_pipeline(tfidf, class_weight)
param_grid = {"clf__C": [0.1, 1, 10]}
model = train_model_with_search(pipeline, param_grid, X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
metrics = evaluate_model(y_test, y_pred)

# 6. Save model
save_model(model, "Logistic Regression", "traditional_ml", metrics, model.best_params_)
```
