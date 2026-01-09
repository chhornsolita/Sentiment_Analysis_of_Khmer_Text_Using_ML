"""
Prediction Script for Khmer Sentiment Analysis

Load a saved model and make predictions on new text data.
Supports custom threshold optimization for better classification.

Usage:
    python predict.py --model_path path/to/model.pkl --text "Your Khmer text here"
    python predict.py --model_path path/to/model.pkl --input_file path/to/texts.csv
    python predict.py --model_path path/to/model.pkl --text "text" --thresholds_path path/to/thresholds.json
"""

import argparse
import pandas as pd
import json
import numpy as np
from src.preprocessing import preprocess_khmer, KHMER_SLANG
from src.model_persistence import load_model, load_preprocessing_objects
from src.threshold_optimization import predict_with_threshold


def predict_single_text(text: str, model, model_type: str = 'traditional_ml', 
                        tokenizer=None, label_encoder=None, max_len=100,
                        thresholds=None, use_proba=False):
    """
    Predict sentiment for a single text.
    
    Args:
        text: Raw Khmer text
        model: Loaded model
        model_type: Type of model ('traditional_ml' or 'deep_learning')
        tokenizer: Tokenizer for LSTM (if model_type='deep_learning')
        label_encoder: Label encoder
        max_len: Maximum sequence length (for LSTM)
        thresholds: Dictionary of custom thresholds for each class
        use_proba: Return probabilities instead of class label
        
    Returns:
        Predicted sentiment label (or probabilities if use_proba=True)
    """
    # Preprocess text
    cleaned_text = preprocess_khmer(text, KHMER_SLANG)
    
    if model_type == 'deep_learning':
        # LSTM prediction
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        prediction_proba = model.predict(padded, verbose=0)
        
        if use_proba:
            return prediction_proba[0]
        
        # Apply custom thresholds if provided
        if thresholds is not None and label_encoder is not None:
            classes = label_encoder.classes_
            predicted_labels = predict_with_threshold(
                prediction_proba, thresholds, list(classes)
            )
            predicted_label = predicted_labels[0]
        else:
            predicted_class = prediction_proba.argmax(axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    else:
        # Traditional ML prediction
        if hasattr(model, 'predict_proba') and (use_proba or thresholds is not None):
            prediction_proba = model.predict_proba([cleaned_text])
            
            if use_proba:
                return prediction_proba[0]
            
            # Apply custom thresholds if provided
            if thresholds is not None:
                classes = model.classes_
                predicted_labels = predict_with_threshold(
                    prediction_proba, thresholds, list(classes)
                )
                predicted_label = predicted_labels[0]
            else:
                predicted_label = model.classes_[prediction_proba[0].argmax()]
        else:
            predicted_label = model.predict([cleaned_text])[0]
        
        # If label encoder exists, decode
        if label_encoder is not None and not isinstance(predicted_label, str):
            predicted_label = label_encoder.inverse_transform([predicted_label])[0]
    
    return predicted_label


def predict_batch(texts: list, model, model_type: str = 'traditional_ml',
                 tokenizer=None, label_encoder=None, max_len=100,
                 thresholds=None, return_proba=False):
    """
    Predict sentiment for multiple texts.
    
    Args:
        texts: List of raw Khmer texts
        model: Loaded model
        model_type: Type of model
        tokenizer: Tokenizer for LSTM
        label_encoder: Label encoder
        max_len: Maximum sequence length
        thresholds: Dictionary of custom thresholds
        return_proba: Also return prediction probabilities
        
    Returns:
        List of predicted sentiment labels (and probabilities if return_proba=True)
    """
    predictions = []
    probabilities = []
    
    for text in texts:
        if return_proba or thresholds is not None:
            proba = predict_single_text(text, model, model_type, tokenizer, 
                                       label_encoder, max_len, thresholds=None, 
                                       use_proba=True)
            probabilities.append(proba)
        
        pred = predict_single_text(text, model, model_type, tokenizer, 
                                  label_encoder, max_len, thresholds=thresholds)
        predictions.append(pred)
    
    if return_proba:
        return predictions, probabilities
    return predictions
thresholds_path', type=str, help='Path to optimal thresholds JSON')
    parser.add_argument('--threshold_method', type=str, default='f1', 
                       choices=['f1', 'youden', 'default'],
                       help='Threshold optimization method to use')
    parser.add_argument('--return_proba', action='store_true', 
                       help='Return prediction probabilities')
    parser.add_argument('--

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment for Khmer text')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--model_type', type=str, default='traditional_ml', 
                       choices=['traditional_ml', 'deep_learning'],
                       help='Type of model')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--input_file', type=str, help='CSV file with texts')
    parser.add_argument('--text_column', type=str, default='text', help='Column name for text')
    parser.add_argument('--tokenizer_path', type=str, help='Path to tokenizer (for LSTM)')
    parser.add_argument('--label_encoder_path', type=str, help='Path to label encoder')
    parser.add_argument('--output_file', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Load model
    prLoad optimal thresholds if provided
    thresholds = None
    if args.thresholds_path:
        print(f"Loading optimal thresholds from {args.thresholds_path}...")
        with open(args.thresholds_path, 'r', encoding='utf-8') as f:
            thresholds_data = json.load(f)
        
        # Select threshold method
        if args.threshold_method == 'f1':
            thresholds = thresholds_data.get('f1_method', {})
            print(f"✓ Using F1-optimized thresholds")
        elif args.threshold_method == 'youden':
            thresholds = thresholds_data.get('youden_method', {})
            print(f"✓ Using Youden-optimized thresholds")
        else:
            thresholds = thresholds_data.get('default', {})
            print(f"✓ Using default thresholds (0.5)")
        
        print("\nThresholds:")
        for cls, thresh in thresholds.items():
            print(f"  {cls}: {thresh:.4f}")
    
    # Predict
    if args.text:
        # Single text prediction
        if args.return_proba:
            proba = predict_single_text(
                args.text, model, args.model_type, tokenizer, label_encoder,
                thresholds=None, use_proba=True
            )
            prediction = predict_single_text(
                args.text, model, args.model_type, tokenizer, label_encoder,
                thresholds=thresholds
            )
            print(f"\nText: {args.text}")
            print(f"Predicted Sentiment: {prediction}")
            print(f"Probabilities: {proba}")
        else:
            prediction = predict_single_text(
                args.text, model, args.model_type, tokenizer, label_encoder,
                thresholds=thresholds
            )
            print(f"\nText: {args.text}")
            print(f"Predicted Sentiment: {prediction}")
    
    elif args.input_file:
        # Batch prediction
        print(f"Loading texts from {args.input_file}...")
        df = pd.read_csv(args.input_file)
        
        texts = df[args.text_column].tolist()
        
        if args.return_proba:
            predictions, probabilities = predict_batch(
                texts, model, args.model_type, tokenizer, label_encoder,
                thresholds=thresholds, return_proba=True
            )
            df['predicted_sentiment'] = predictions
            
            # Add probability columns
            if len(probabilities) > 0 and len(probabilities[0]) > 0:
                proba_array = np.array(probabilities)
                classes = model.classes_ if hasattr(model, 'classes_') else ['neg', 'neu', 'pos']
                for i, cls in enumerate(classes):
                    df[f'prob_{cls}'] = proba_array[:, i]
        else:
            predictions = predict_batch(
                texts, model, args.model_type, tokenizer, label_encoder,
                thresholds=thresholds
            )
            df['predicted_sentiment'] = predictions
        
        if args.output_file:
            df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
            print(f"✓ Predictions saved to {args.output_file}")
        else:
            cols_to_show = [args.text_column, 'predicted_sentiment']
            if args.return_proba:
                cols_to_show.extend([col for col in df.columns if col.startswith('prob_')])
            print("\nPredictions:")
            print(df[cols_to_show
        if args.output_file:
            df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
            print(f"✓ Predictions saved to {args.output_file}")
        else:
            print("\nPredictions:")
            print(df[[args.text_column, 'predicted_sentiment']])
    
    else:
        print("Error: Please provide either --text or --input_file")


if __name__ == "__main__":
    main()
