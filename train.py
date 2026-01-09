"""
Main Training Pipeline for Khmer Sentiment Analysis

This script orchestrates the entire training pipeline:
1. Load and preprocess data
2. Extract features
3. Train multiple models
4. Evaluate and compare models
5. Save the best model

Usage:
    python train.py --data_path path/to/data.csv
"""

import argparse
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

# Import custom modules
from src.config import *
from src.data_loader import load_data, clean_data, prepare_train_test_split, get_class_distribution
from src.preprocessing import preprocess_khmer, KHMER_SLANG
from src.feature_extraction import create_tfidf_vectorizer, compute_class_weights
from src.models import (
    create_logistic_regression_pipeline,
    create_svm_pipeline,
    create_naive_bayes_pipeline,
    create_random_forest_pipeline,
    create_xgboost_pipeline,
    get_hyperparameter_grids,
    train_model_with_search
)
from src.deep_learning import create_lstm_model, prepare_sequences, train_lstm_model
from src.evaluation import compare_models, plot_model_comparison, plot_confusion_matrix
from src.model_persistence import save_model, save_comparison_report
from src.threshold_optimization import (
    compute_roc_curves_multiclass,
    get_optimal_thresholds_multiclass,
    plot_roc_curves_multiclass,
    plot_threshold_analysis,
    generate_threshold_report
)

warnings.filterwarnings('ignore')


def main(data_path: str = None, train_lstm: bool = True):
    """
    Main training pipeline.
    
    Args:
        data_path: Path to the dataset CSV file
        train_lstm: Whether to train LSTM model (requires TensorFlow)
    """
    
    # ========== 1. LOAD AND PREPROCESS DATA ==========
    print("="*80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    if data_path is None:
        data_path = CLEANED_DATA_PATH
    
    df = load_data(data_path)
    print(f"✓ Loaded data: {df.shape}")
    
    df = clean_data(df, TEXT_COLUMN)
    print(f"✓ Cleaned data: {df.shape}")
    
    # Apply Khmer preprocessing
    df[CLEAN_TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: preprocess_khmer(x, KHMER_SLANG))
    print(f"✓ Applied Khmer preprocessing")
    
    # Check class distribution
    print(f"\nClass Distribution:")
    print(get_class_distribution(df[TARGET_COLUMN]))
    
    # ========== 2. TRAIN-TEST SPLIT ==========
    print("\n" + "="*80)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*80)
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df,
        text_column=CLEAN_TEXT_COLUMN,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=STRATIFY_SPLIT
    )
    
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # ========== 3. FEATURE EXTRACTION ==========
    print("\n" + "="*80)
    print("STEP 3: FEATURE EXTRACTION")
    print("="*80)
    
    tfidf = create_tfidf_vectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF
    )
    
    class_weight = compute_class_weights(y_train)
    print(f"✓ Class weights: {class_weight}")
    
    # ========== 4. TRAIN TRADITIONAL ML MODELS ==========
    print("\n" + "="*80)
    print("STEP 4: TRAINING TRADITIONAL ML MODELS")
    print("="*80)
    
    param_grids = get_hyperparameter_grids()
    trained_models = {}
    
    # Logistic Regression
    print("\n[1/5] Training Logistic Regression...")
    pipe_lr = create_logistic_regression_pipeline(tfidf, class_weight)
    rs_lr = train_model_with_search(pipe_lr, param_grids['lr'], X_train, y_train)
    trained_models["Logistic Regression"] = rs_lr
    print(f"✓ Best F1-Macro: {rs_lr.best_score_:.4f}")
    
    # SVM
    print("\n[2/5] Training SVM...")
    pipe_svm = create_svm_pipeline(tfidf, class_weight)
    rs_svm = train_model_with_search(pipe_svm, param_grids['svm'], X_train, y_train)
    trained_models["SVM"] = rs_svm
    print(f"✓ Best F1-Macro: {rs_svm.best_score_:.4f}")
    
    # Naive Bayes
    print("\n[3/5] Training Naive Bayes...")
    pipe_nb = create_naive_bayes_pipeline(tfidf)
    rs_nb = train_model_with_search(pipe_nb, param_grids['nb'], X_train, y_train)
    trained_models["Naive Bayes"] = rs_nb
    print(f"✓ Best F1-Macro: {rs_nb.best_score_:.4f}")
    
    # Random Forest
    print("\n[4/5] Training Random Forest...")
    pipe_rf = create_random_forest_pipeline(tfidf, class_weight)
    rs_rf = train_model_with_search(pipe_rf, param_grids['rf'], X_train, y_train)
    trained_models["Random Forest"] = rs_rf
    print(f"✓ Best F1-Macro: {rs_rf.best_score_:.4f}")
    
    # XGBoost (optional)
    print("\n[5/5] Training XGBoost...")
    pipe_xgb = create_xgboost_pipeline(tfidf)
    if pipe_xgb is not None:
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        rs_xgb = train_model_with_search(pipe_xgb, param_grids['xgb'], X_train, y_train_encoded)
        trained_models["XGBoost"] = rs_xgb
        print(f"✓ Best F1-Macro: {rs_xgb.best_score_:.4f}")
    else:
        le = None
    
    # ========== 5. TRAIN DEEP LEARNING MODEL (OPTIONAL) ==========
    if train_lstm:
        print("\n" + "="*80)
        print("STEP 5: TRAINING DEEP LEARNING MODEL (BiLSTM)")
        print("="*80)
        
        # Prepare sequences
        X_train_pad, tokenizer = prepare_sequences(X_train, MAX_WORDS, MAX_LEN)
        X_test_pad, _ = prepare_sequences(X_test, MAX_WORDS, MAX_LEN, tokenizer)
        
        if X_train_pad is not None:
            # Encode labels
            le_lstm = LabelEncoder()
            y_train_lstm = le_lstm.fit_transform(y_train)
            y_test_lstm = le_lstm.transform(y_test)
            
            # Create and train model
            model_lstm = create_lstm_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM, LSTM_UNITS, NUM_CLASSES)
            
            if model_lstm is not None:
                history = train_lstm_model(
                    model_lstm,
                    X_train_pad, y_train_lstm,
                    X_test_pad, y_test_lstm,
                    epochs=LSTM_EPOCHS,
                    batch_size=LSTM_BATCH_SIZE,
                    patience=LSTM_PATIENCE
                )
                print("✓ BiLSTM trained successfully")
            else:
                model_lstm = None
                tokenizer = None
                le_lstm = None
        else:
            model_lstm = None
            tokenizer = None
            le_lstm = None
    else:
        model_lstm = None
        tokenizer = None
        le_lstm = None
    
    # ========== 6. MODEL COMPARISON ==========
    print("\n" + "="*80)
    print("STEP 6: MODEL COMPARISON")
    print("="*80)
    
    comparison_df = compare_models(trained_models, X_test, y_test, le if 'XGBoost' in trained_models else None)
    
    # Add LSTM results if available
    if model_lstm is not None:
        y_pred_lstm = model_lstm.predict(X_test_pad, verbose=0)
        y_pred_lstm_classes = y_pred_lstm.argmax(axis=1)
        y_pred_lstm_labels = le_lstm.inverse_transform(y_pred_lstm_classes)
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        lstm_result = {
            'Model': 'Bidirectional LSTM',
            'Accuracy': accuracy_score(y_test, y_pred_lstm_labels),
            'F1-Macro': f1_score(y_test, y_pred_lstm_labels, average='macro'),
            'Precision-Macro': precision_score(y_test, y_pred_lstm_labels, average='macro'),
            'Recall-Macro': recall_score(y_test, y_pred_lstm_labels, average='macro'),
            'Best CV Score': max(history.history['val_accuracy']) if history else 0
        }
        comparison_df = pd.concat([comparison_df, pd.DataFrame([lstm_result])], ignore_index=True)
        comparison_df = comparison_df.sort_values('F1-Macro', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # ========== 7. SAVE BEST MODEL ==========
    print("\n" + "="*80)
    print("STEP 7: SAVING BEST MODEL")
    print("="*80)
    
    best_model_idx = comparison_df['F1-Macro'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   F1-Macro: {comparison_df.loc[best_model_idx, 'F1-Macro']:.4f}")
    
    # Prepare performance metrics
    performance = {
        'accuracy': float(comparison_df.loc[best_model_idx, 'Accuracy']),
        'f1_macro': float(comparison_df.loc[best_model_idx, 'F1-Macro']),
        'precision_macro': float(comparison_df.loc[best_model_idx, 'Precision-Macro']),
        'recall_macro': float(comparison_df.loc[best_model_idx, 'Recall-Macro']),
        'best_cv_score': float(comparison_df.loc[best_model_idx, 'Best CV Score'])
    }
    
    # Save based on model type
    if best_model_name == "Bidirectional LSTM":
        hyperparameters = {
            'max_words': MAX_WORDS,
            'max_len': MAX_LEN,
            'embedding_dim': EMBEDDING_DIM,
            'lstm_units': LSTM_UNITS
        }
        save_model(
            model_lstm,
            best_model_name,
            'deep_learning',
            performance,
            hyperparameters,
            MODELS_DIR,
            tokenizer=tokenizer,
            label_encoder=le_lstm
        )
    else:
        best_model_obj = trained_models[best_model_name]
        hyperparameters = best_model_obj.best_params_
        
        label_enc = le if best_model_name == "XGBoost" else None
        
        save_model(
            best_model_obj,
            best_model_name,
            'traditional_ml',
            performance,
            hyperparameters,
            MODELS_DIR,
            label_encoder=label_enc
        )
    
    # Save comparison report
    save_comparison_report(comparison_df, REPORTS_DIR)
    
    # ========== 8. ROC ANALYSIS & THRESHOLD OPTIMIZATION ==========
    print("\n" + "="*80)
    print("STEP 8: ROC ANALYSIS & OPTIMAL THRESHOLD FINDING")
    print("="*80)
    
    # Get probability predictions from best model
    if best_model_name == "Bidirectional LSTM" and model_lstm is not None:
        y_pred_proba = model_lstm.predict(X_test_pad, verbose=0)
        classes_used = le_lstm.classes_
    elif best_model_name == "XGBoost" and 'XGBoost' in trained_models:
        # XGBoost returns class predictions, need to get probabilities
        best_model_obj = trained_models[best_model_name]
        # Get decision function or predict_proba
        if hasattr(best_model_obj, 'predict_proba'):
            y_pred_proba = best_model_obj.predict_proba(X_test)
        else:
            print("⚠ XGBoost model doesn't support probability predictions for ROC analysis")
            y_pred_proba = None
        classes_used = le.classes_ if le else CLASS_LABELS
    else:
        best_model_obj = trained_models[best_model_name]
        if hasattr(best_model_obj, 'predict_proba'):
            y_pred_proba = best_model_obj.predict_proba(X_test)
            classes_used = best_model_obj.classes_
        else:
            # SVM might use decision_function
            print("⚠ Model doesn't support probability predictions. Skipping ROC analysis.")
            y_pred_proba = None
            classes_used = CLASS_LABELS
    
    if y_pred_proba is not None:
        print("\n📊 Computing ROC curves for all classes...")
        
        # Compute ROC curves
        roc_data = compute_roc_curves_multiclass(y_test, y_pred_proba, list(classes_used))
        
        # Plot ROC curves
        import os
        os.makedirs(FIGURES_DIR, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        roc_plot_path = os.path.join(FIGURES_DIR, f'roc_curves_{timestamp}.png')
        plot_roc_curves_multiclass(roc_data, save_path=roc_plot_path)
        
        # Find optimal thresholds using multiple methods
        print("\n🎯 Finding optimal thresholds...")
        
        optimal_thresholds_youden = get_optimal_thresholds_multiclass(
            y_test, y_pred_proba, list(classes_used), method='youden'
        )
        
        optimal_thresholds_f1 = get_optimal_thresholds_multiclass(
            y_test, y_pred_proba, list(classes_used), method='f1'
        )
        
        print("\n" + "-"*80)
        print("OPTIMAL THRESHOLDS (Youden's Index Method)")
        print("-"*80)
        for cls, threshold in optimal_thresholds_youden.items():
            print(f"  {cls:>10s}: {threshold:.4f}")
        
        print("\n" + "-"*80)
        print("OPTIMAL THRESHOLDS (F1 Score Maximization)")
        print("-"*80)
        for cls, threshold in optimal_thresholds_f1.items():
            print(f"  {cls:>10s}: {threshold:.4f}")
        
        # Generate comprehensive threshold report
        print("\n📋 Generating threshold optimization report...")
        threshold_report = generate_threshold_report(
            y_test, y_pred_proba, list(classes_used), methods=['youden', 'f1']
        )
        
        print("\n" + "="*80)
        print("THRESHOLD OPTIMIZATION REPORT")
        print("="*80)
        print(threshold_report.to_string(index=False))
        
        # Save threshold report
        threshold_report_path = os.path.join(REPORTS_DIR, f'threshold_report_{timestamp}.csv')
        threshold_report.to_csv(threshold_report_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Threshold report saved: {threshold_report_path}")
        
        # Plot threshold analysis for each class
        print("\n📈 Generating threshold analysis plots...")
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=list(classes_used))
        
        for i, cls in enumerate(classes_used):
            threshold_plot_path = os.path.join(
                FIGURES_DIR, 
                f'threshold_analysis_{cls}_{timestamp}.png'
            )
            opt_thresh, best_f1 = plot_threshold_analysis(
                y_test_bin[:, i],
                y_pred_proba[:, i],
                i,
                cls,
                save_path=threshold_plot_path
            )
            print(f"  ✓ {cls}: Optimal threshold = {opt_thresh:.4f}, Best F1 = {best_f1:.4f}")
        
        # Save optimal thresholds to JSON
        import json
        thresholds_data = {
            'model_name': best_model_name,
            'timestamp': timestamp,
            'youden_method': {str(k): float(v) for k, v in optimal_thresholds_youden.items()},
            'f1_method': {str(k): float(v) for k, v in optimal_thresholds_f1.items()},
            'default': {str(cls): 0.5 for cls in classes_used},
            'recommendation': 'Use F1 method for balanced classification, Youden for maximum separation'
        }
        
        thresholds_json_path = os.path.join(MODELS_DIR, f'optimal_thresholds_{timestamp}.json')
        with open(thresholds_json_path, 'w', encoding='utf-8') as f:
            json.dump(thresholds_data, f, indent=4, ensure_ascii=False)
        print(f"\n✓ Optimal thresholds saved: {thresholds_json_path}")
    
    print("\n" + "="*80)
    print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Khmer Sentiment Analysis models')
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset CSV file')
    parser.add_argument('--no_lstm', action='store_true', help='Skip LSTM training')
    
    args = parser.parse_args()
    
    main(data_path=args.data_path, train_lstm=not args.no_lstm)
