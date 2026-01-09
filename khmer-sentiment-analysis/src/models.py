"""
Machine Learning Models Module

This module provides functions for training and evaluating various
machine learning models for Khmer sentiment analysis.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Optional


def create_logistic_regression_pipeline(tfidf, class_weight: Dict) -> Pipeline:
    """
    Create a Logistic Regression pipeline with TF-IDF.
    
    Args:
        tfidf: TfidfVectorizer instance
        class_weight: Class weight dictionary
        
    Returns:
        sklearn Pipeline with TF-IDF and Logistic Regression
    """
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=2000
        ))
    ])
    return pipeline


def create_svm_pipeline(tfidf, class_weight: Dict) -> Pipeline:
    """
    Create a Linear SVM pipeline with TF-IDF.
    
    Args:
        tfidf: TfidfVectorizer instance
        class_weight: Class weight dictionary
        
    Returns:
        sklearn Pipeline with TF-IDF and LinearSVC
    """
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", LinearSVC(class_weight=class_weight))
    ])
    return pipeline


def create_naive_bayes_pipeline(tfidf) -> Pipeline:
    """
    Create a Naive Bayes pipeline with TF-IDF.
    
    Args:
        tfidf: TfidfVectorizer instance
        
    Returns:
        sklearn Pipeline with TF-IDF and MultinomialNB
    """
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", MultinomialNB())
    ])
    return pipeline


def create_random_forest_pipeline(tfidf, class_weight: Dict) -> Pipeline:
    """
    Create a Random Forest pipeline with TF-IDF.
    
    Args:
        tfidf: TfidfVectorizer instance
        class_weight: Class weight dictionary
        
    Returns:
        sklearn Pipeline with TF-IDF and RandomForestClassifier
    """
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", RandomForestClassifier(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline


def create_xgboost_pipeline(tfidf) -> Optional[Pipeline]:
    """
    Create an XGBoost pipeline with TF-IDF.
    
    Args:
        tfidf: TfidfVectorizer instance
        
    Returns:
        sklearn Pipeline with TF-IDF and XGBClassifier, or None if XGBoost not installed
    """
    try:
        import xgboost as xgb
        pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ))
        ])
        return pipeline
    except ImportError:
        print("⚠ XGBoost not installed. Skipping XGBoost model.")
        return None


def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """
    Get hyperparameter search grids for all models.
    
    Returns:
        Dictionary mapping model names to their hyperparameter grids
    """
    grids = {
        "lr": {
            "clf__C": [0.01, 0.1, 1, 5, 10],
            "clf__solver": ["lbfgs", "saga"]
        },
        "svm": {
            "clf__C": [0.01, 0.1, 1, 5, 10]
        },
        "nb": {
            "clf__alpha": [0.1, 0.5, 1.0, 2.0]
        },
        "rf": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [10, 20, 30, None],
            "clf__min_samples_split": [2, 5]
        },
        "xgb": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.3]
        }
    }
    return grids


def train_model_with_search(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42
) -> RandomizedSearchCV:
    """
    Train a model using RandomizedSearchCV for hyperparameter tuning.
    
    Args:
        pipeline: sklearn Pipeline
        param_grid: Hyperparameter search space
        X_train: Training features
        y_train: Training labels
        n_iter: Number of parameter settings to sample
        cv: Number of cross-validation folds
        scoring: Scoring metric
        random_state: Random seed
        
    Returns:
        Trained RandomizedSearchCV object
    """
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state
    )
    
    search.fit(X_train, y_train)
    return search
