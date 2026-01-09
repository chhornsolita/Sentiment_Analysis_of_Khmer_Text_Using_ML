"""
Gradio Interface for Khmer Sentiment Analysis - Hugging Face Spaces
This app provides a user-friendly interface for analyzing Khmer text sentiment.
"""

import gradio as gr
import os
import json
import glob
import numpy as np
from datetime import datetime

# Import custom modules
from src.preprocessing import preprocess_khmer, KHMER_SLANG
from src.model_persistence import load_model, load_preprocessing_objects

# Global variables
MODEL = None
TFIDF = None
LABEL_ENCODER = None
MODEL_INFO = {}

def load_best_model():
    """Load the most recent best model and preprocessing objects."""
    global MODEL, TFIDF, LABEL_ENCODER, MODEL_INFO
    
    try:
        # Find the most recent model file
        model_files = glob.glob('models/saved_models/best_model_*.pkl')
        if not model_files:
            # Try alternative locations
            model_files = glob.glob('notebooks/best_model_*.joblib')
        
        if not model_files:
            raise FileNotFoundError("No saved models found")
        
        # Get the most recent model
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Loading model: {latest_model}")
        
        # Load model
        MODEL = load_model(latest_model)
        
        # Extract TF-IDF vectorizer from the model pipeline
        if hasattr(MODEL, 'best_estimator_'):
            pipeline = MODEL.best_estimator_
        else:
            pipeline = MODEL
        
        # Extract the TF-IDF vectorizer from the pipeline
        if hasattr(pipeline, 'named_steps') and 'tfidf' in pipeline.named_steps:
            TFIDF = pipeline.named_steps['tfidf']
            print("   ✓ TF-IDF vectorizer extracted from pipeline")
        else:
            raise ValueError("Could not extract TF-IDF vectorizer from model")
        
        # Load label encoder
        from sklearn.preprocessing import LabelEncoder
        import joblib
        
        # Try to load from separate file
        le_path = latest_model.replace('best_model_', 'label_encoder_')
        if os.path.exists(le_path):
            LABEL_ENCODER = joblib.load(le_path)
            print("   ✓ Label encoder loaded from file")
        else:
            # Get from metadata
            metadata_file = latest_model.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if 'classes' in metadata:
                        LABEL_ENCODER = LabelEncoder()
                        LABEL_ENCODER.classes_ = np.array(metadata['classes'])
                        print("   ✓ Label encoder created from metadata")
            
            # Default classes for sentiment analysis
            if LABEL_ENCODER is None:
                LABEL_ENCODER = LabelEncoder()
                LABEL_ENCODER.classes_ = np.array(['negative', 'neutral', 'positive'])
                print("   ⚠ Using default label encoder (negative, neutral, positive)")
        
        # Load metadata
        metadata_file = latest_model.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                MODEL_INFO = json.load(f)
        else:
            MODEL_INFO = {
                'model_name': os.path.basename(latest_model),
                'loaded_at': datetime.now().isoformat()
            }
        
        print("✓ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

def predict_sentiment(text):
    """
    Predict sentiment for Khmer text.
    
    Args:
        text: Raw Khmer text input
        
    Returns:
        Dictionary with prediction results and confidence scores
    """
    if not text or text.strip() == "":
        return {
            "error": "Please enter some text",
            "sentiment": None,
            "confidence": 0.0
        }
    
    try:
        # Preprocess text
        cleaned_text = preprocess_khmer(text, KHMER_SLANG)
        
        if not cleaned_text or cleaned_text.strip() == "":
            return {
                "error": "Text preprocessing resulted in empty text",
                "sentiment": None,
                "confidence": 0.0
            }
        
        # Make prediction using the model directly
        if hasattr(MODEL, 'best_estimator_'):
            pipeline = MODEL.best_estimator_
        else:
            pipeline = MODEL
        
        # Get prediction and probabilities
        prediction = pipeline.predict([cleaned_text])[0]
        probabilities = pipeline.predict_proba([cleaned_text])[0]
        
        # Get class labels
        if LABEL_ENCODER is not None:
            classes = LABEL_ENCODER.classes_
            predicted_label = LABEL_ENCODER.inverse_transform([prediction])[0]
        else:
            # Get classes from pipeline
            classes = pipeline.classes_
            predicted_label = classes[prediction]
        
        # Get confidence score
        confidence = float(probabilities[prediction])
        
        # Create confidence breakdown for all classes
        confidence_breakdown = {
            str(label): float(prob) 
            for label, prob in zip(classes, probabilities)
        }
        
        # Determine emoji
        emoji_map = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞'
        }
        emoji = emoji_map.get(str(predicted_label).lower(), '❓')
        
        # Format result
        result_text = f"""
## Prediction Results

### Sentiment: {emoji} **{predicted_label.upper()}**

**Confidence:** {confidence:.2%}

---

### Confidence Breakdown:
"""
        for label, prob in confidence_breakdown.items():
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            result_text += f"\n**{label.capitalize()}:** {bar} {prob:.2%}"
        
        result_text += f"""

---

**Original Text:** {text}

**Cleaned Text:** {cleaned_text}
"""
        
        return result_text
        
    except Exception as e:
        error_msg = f"""
## Error

An error occurred during prediction:

```
{str(e)}
```

Please try again with different text.
"""
        return error_msg

def get_examples():
    """Return example Khmer texts for demonstration."""
    return [
        ["ល្អណាស់! វាពិតជាអស្ចារ្យ"],  # Positive: Very good! It's amazing
        ["គួរឱ្យខកចិត្ត មិនល្អទេ"],  # Negative: Disappointing, not good
        ["ធម្មតាៗ មិនអីទេ"],  # Neutral: Normal, it's okay
        ["ខ្ញុំចូលចិត្តនេះណាស់ សប្បាយរីករាយ"],  # Positive: I like this very much, happy
        ["អាក្រក់ណាស់ មិនគួរឱ្យទុកចិត្ត"],  # Negative: Very bad, untrustworthy
    ]

# Initialize model on startup
print("Initializing Khmer Sentiment Analysis Model...")
model_loaded = load_best_model()

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Khmer Sentiment Analysis",
    css="""
    .gradio-container {
        max-width: 900px !important;
    }
    """
) as demo:
    gr.Markdown(
        """
        # 🇰🇭 Khmer Text Sentiment Analysis
        
        Analyze the sentiment of Khmer (Cambodian) text using machine learning.
        This model can classify text as **Positive**, **Neutral**, or **Negative**.
        
        ### How to use:
        1. Enter Khmer text in the input box
        2. Click "Analyze Sentiment"
        3. View the prediction results and confidence scores
        
        ### Features:
        - ✅ Supports Khmer language text
        - ✅ Real-time sentiment prediction
        - ✅ Confidence scores for all sentiment classes
        - ✅ Handles Khmer slang and common phrases
        """
    )
    
    if not model_loaded:
        gr.Markdown(
            """
            ⚠️ **Warning:** Model failed to load. Please check the model files.
            """
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Enter Khmer Text",
                placeholder="បញ្ចូលអត្ថបទខ្មែរនៅទីនេះ...",
                lines=5,
                max_lines=10
            )
            
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Results")
    
    # Examples
    gr.Markdown("### 📝 Example Texts (Click to try):")
    gr.Examples(
        examples=get_examples(),
        inputs=[input_text],
        label=None
    )
    
    # Set up the event
    analyze_btn.click(
        fn=predict_sentiment,
        inputs=[input_text],
        outputs=[output_text]
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Model Information:**
        - Trained on Khmer sentiment analysis dataset
        - Uses TF-IDF vectorization and machine learning classification
        - Supports common Khmer phrases and slang
        
        **Note:** This is a research project for educational purposes.
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
