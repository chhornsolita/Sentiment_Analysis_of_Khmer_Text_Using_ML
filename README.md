# Khmer Sentiment Analysis

A machine learning project for sentiment analysis of Khmer (Cambodian) text using various classification algorithms.

## 📋 Project Overview

This project implements sentiment analysis for Khmer text, classifying text into three categories:
- **Positive** (អវិជ្ជមាន)
- **Neutral** (អព្យាក្រឹត)
- **Negative** (វិជ្ជមាន)

The project explores multiple machine learning approaches including traditional ML models (Logistic Regression, SVM, Naive Bayes) and ensemble methods (Random Forest, XGBoost, Voting Classifier).

## 📁 Project Structure

```
PROJECT/
├── data/
│   └── Data Collection - Sheet1.csv    # Dataset
├── notebooks/
│   └── Notebook.ipynb                  # Exploratory analysis and experiments
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── data_preprocessing.py           # Text preprocessing functions
│   ├── feature_extraction.py           # TF-IDF vectorization
│   ├── models.py                       # Model training functions
│   ├── evaluation.py                   # Evaluation and visualization
│   ├── train.py                        # Main training script
│   └── predict.py                      # Prediction script
├── models/                             # Saved trained models
├── results/                            # Results and plots
├── requirements.txt                    # Python dependencies
└── # Khmer Sentiment Analysis Using Machine Learning

## 📋 Project Overview

This project performs comprehensive sentiment analysis on Khmer text data using multiple machine learning approaches, from traditional ML to deep learning models. The goal is to classify Khmer social media posts, reviews, or news comments into three sentiment categories: **positive**, **neutral**, and **negative**.

## 🎯 Key Features

- **Khmer-Specific Preprocessing**: Unicode normalization (NFC), slang handling, special character removal
- **Multiple ML Models Comparison**:
  - Traditional ML: Logistic Regression, SVM, Naive Bayes
  - Deep Learning: Bidirectional LSTM
- **Comprehensive Evaluation**: Confusion matrices, per-class metrics, error analysis
- **Class Imbalance Handling**: Balanced class weights for fair evaluation
- **Visualization**: Performance comparison charts and training curves

## 📁 Project Structure

```
PROJECT/
│
├── Model.ipynb              # Main notebook with all analysis
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── Data Collection - Sheet1.csv  # Original data
│   └── data_cleaned_all.csv          # Cleaned dataset (1057 samples)
│
├── models/                  # Saved models (generated after training)
│   ├── best_lr_model.pkl
│   ├── lstm_model.h5
│   └── lstm_tokenizer.pkl
│
└── src/
    └── clean_space.py       # Utility scripts
```

## 🔧 Installation

1. **Clone or download this project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Required Python Version**: Python 3.8 or higher

## 📊 Dataset

- **Size**: 1,057 Khmer text samples
- **Classes**: 
  - Positive
  - Negative  
  - Neutral
- **Source**: Social media posts and news comments in Khmer language
- **Format**: CSV with columns: `text`, `target`

## 🚀 Usage

1. **Open the Jupyter Notebook**:
```bash
jupyter notebook Model.ipynb
```

2. **Run cells sequentially**:
   - Load and explore data
   - Preprocess Khmer text
   - Train traditional ML models (LR, SVM, NB)
   - Train LSTM model
   - Compare model performance
   - Analyze errors and visualize results

3. **Make predictions on new text**:
```python
# Example prediction
new_text = "ខ្ញុំចូលចិត្តផលិតផលនេះណាស់"
new_text_clean = preprocess_khmer(new_text)
prediction = best_model.predict([new_text_clean])
print(f"Sentiment: {prediction[0]}")
```

## 📈 Model Performance

The notebook includes comprehensive comparison of all models with:
- Accuracy scores
- F1-Macro and F1-Weighted scores
- Confusion matrices
- Per-class precision, recall, F1-score
- Training curves (for LSTM)

## 🔍 Khmer-Specific Challenges Addressed

1. **Unicode Normalization**: Proper handling of Khmer Unicode (NFD → NFC)
2. **Slang Handling**: Dictionary-based normalization of informal Khmer
3. **Special Markers**: Removal of URL artifacts and special symbols
4. **Character Range**: Preservation of Khmer Unicode range (U+1780 to U+17FF)
5. **Class Imbalance**: Balanced class weights for minority classes

## 🎓 Key Findings & Recommendations

### Current Limitations:
- Small dataset (~1000 samples) limits deep learning performance
- Limited slang dictionary coverage
- No BERT-based models yet

### Future Improvements:
1. **Expand Dataset**: Target 5,000-10,000+ samples
2. **Add BERT Models**: Fine-tune mBERT or XLM-RoBERTa
3. **Enhanced Preprocessing**: Comprehensive slang dictionary, negation handling
4. **Feature Engineering**: Character n-grams, text length features
5. **Cross-Validation**: K-fold CV for robust evaluation

## 📚 Requirements

See [requirements.txt](requirements.txt) for full list. Main dependencies:
- pandas, numpy
- scikit-learn (traditional ML)
- tensorflow/keras (LSTM)
- matplotlib, seaborn (visualization)

## 🤝 Contributing

To improve this project:
1. Expand the labeled Khmer dataset
2. Add more slang mappings to the preprocessing function
3. Implement BERT-based models
4. Enhance error analysis with linguistic features

## 📝 License

This project is for educational purposes. Feel free to use and modify.

## 👤 Author

Created as part of the I5-AMS WR project focusing on Khmer NLP and sentiment analysis.

---

**Note**: This project demonstrates practical implementation of sentiment analysis on a low-resource language (Khmer) with limited data. Performance can be significantly improved with more training data and advanced models.                           # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this project

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install XGBoost for advanced models:
```bash
pip install xgboost
```

## 📊 Dataset

The dataset should be in CSV format with the following columns:
- `text`: Khmer text to analyze
- `target`: Sentiment label (positive, neutral, negative)

Place your dataset in the `data/` directory.

## 🔧 Usage

### Training Models

Train all models with basic settings:
```bash
cd src
python train.py
```

Train specific models:
```bash
python train.py --models lr svm nb
```

Train with enhanced preprocessing:
```bash
python train.py --enhanced
```

Train with ensemble voting classifier:
```bash
python train.py --ensemble --save-models
```

Available options:
- `--data-path`: Path to dataset (default: `data/Data Collection - Sheet1.csv`)
- `--models`: Models to train: `all`, `lr`, `svm`, `nb`, `rf`, `xgb`
- `--enhanced`: Use enhanced preprocessing with stopwords removal
- `--ensemble`: Train voting ensemble classifier
- `--save-models`: Save trained models to disk
- `--model-dir`: Directory to save models (default: `models`)
- `--result-dir`: Directory to save results (default: `results`)

### Making Predictions

Predict sentiment for new text:
```bash
python predict.py --text "អរគុណច្រើន" --model models/svm.pkl
```

Options:
- `--text`: Khmer text to analyze (required)
- `--vectorizer`: Path to saved vectorizer (default: `models/vectorizer.pkl`)
- `--model`: Path to saved model (default: `models/svm.pkl`)
- `--label-encoder`: Path to saved label encoder (default: `models/label_encoder.pkl`)

## 📚 Module Documentation

### data_preprocessing.py

Functions for cleaning and preprocessing Khmer text:
- `khmer_preprocess(text)`: Basic text cleaning
- `khmer_preprocess_enhanced(text)`: Enhanced cleaning with stopword removal
- `load_and_clean_data(filepath)`: Load and preprocess dataset
- `encode_labels(df)`: Encode sentiment labels to numeric values

### feature_extraction.py

TF-IDF feature extraction:
- `create_tfidf_vectorizer()`: Create standard vectorizer
- `create_enhanced_vectorizer()`: Create enhanced vectorizer with optimized parameters
- `extract_features(df)`: Extract TF-IDF features from text
- `split_data(X, y)`: Split data into train/test sets

### models.py

Model training functions:
- `train_logistic_regression()`: Train Logistic Regression with grid search
- `train_svm()`: Train Support Vector Machine
- `train_naive_bayes()`: Train Multinomial Naive Bayes
- `train_random_forest()`: Train Random Forest classifier
- `train_xgboost()`: Train XGBoost classifier
- `train_voting_classifier()`: Train ensemble voting classifier
- `save_model()` / `load_model()`: Save/load trained models

### evaluation.py

Model evaluation and visualization:
- `evaluate_model()`: Calculate accuracy, F1 score, and print classification report
- `plot_confusion_matrix()`: Visualize confusion matrix
- `plot_sentiment_distribution()`: Plot label distribution
- `compare_models()`: Compare multiple models and visualize results
- `save_results()`: Save evaluation results to CSV

## 📈 Model Performance

Based on the experimental results:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~46% |
| SVM | ~52% |
| Naive Bayes | ~49% |
| Random Forest | ~50-55% |
| XGBoost | ~50-55% |
| Voting Classifier | ~52-56% |

*Note: Results may vary based on dataset and hyperparameters*

## 🔍 Key Features

### Text Preprocessing
- Unicode normalization for Khmer characters
- Removal of numbers and punctuation
- Khmer stopwords removal (enhanced mode)
- Whitespace normalization

### Feature Engineering
- TF-IDF vectorization with n-grams (unigrams, bigrams, trigrams)
- Configurable feature limits and document frequency filters
- Sublinear TF scaling for better performance

### Model Training
- Automated class weight balancing for imbalanced datasets
- Grid search hyperparameter optimization
- Cross-validation for robust evaluation
- Support for multiple model architectures

### Evaluation
- Comprehensive metrics (accuracy, F1-score, precision, recall)
- Confusion matrix visualization
- Model comparison charts
- Results export to CSV

## 💡 Improvement Strategies

To improve model performance:

1. **Collect More Data**: Sentiment analysis benefits greatly from larger, diverse datasets
2. **Advanced Preprocessing**: Implement Khmer-specific stemming/lemmatization
3. **Deep Learning**: Use pre-trained models like XLM-RoBERTa for multilingual understanding
4. **Feature Engineering**: Add character-level features, text length, sentiment lexicons
5. **Data Augmentation**: Use back-translation, synonym replacement



---

**Note**: This project is designed for Khmer language sentiment analysis. Results may vary based on dataset quality, size, and domain specificity.
