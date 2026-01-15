"""
Flask API for Khmer Sentiment Analysis
Provides REST endpoints for sentiment prediction using the trained model.
"""

from flask import Flask, request, jsonify, render_template_string
import os
import json
import glob
from datetime import datetime
import numpy as np

# Import custom modules
from src.preprocessing import preprocess_khmer
from src.model_persistence import load_model, load_preprocessing_objects

app = Flask(__name__)

# Global variables to store loaded model and objects
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
            raise FileNotFoundError("No saved models found in models/saved_models/")
        
        # Get the most recent model
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Loading model: {latest_model}")
        
        # Load model
        MODEL = load_model(latest_model)
        
        # Extract TF-IDF vectorizer from the model pipeline
        # The model is a RandomizedSearchCV object containing a Pipeline
        if hasattr(MODEL, 'best_estimator_'):
            pipeline = MODEL.best_estimator_
        else:
            pipeline = MODEL
        
        # Extract the TF-IDF vectorizer from the pipeline
        if hasattr(pipeline, 'named_steps') and 'tfidf' in pipeline.named_steps:
            TFIDF = pipeline.named_steps['tfidf']
            print("   ✓ TF-IDF vectorizer extracted from pipeline")
        else:
            # Fallback: try to load from separate file
            try:
                TFIDF, LABEL_ENCODER = load_preprocessing_objects(latest_model)
            except:
                raise ValueError("Could not extract TF-IDF vectorizer from model")
        
        # Load label encoder - try multiple methods
        from sklearn.preprocessing import LabelEncoder
        import joblib
        
        # Method 1: Load from separate file
        le_path = latest_model.replace('best_model_', 'label_encoder_')
        if os.path.exists(le_path):
            LABEL_ENCODER = joblib.load(le_path)
            print("   ✓ Label encoder loaded from file")
        else:
            # Method 2: Get from metadata
            metadata_file = latest_model.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if 'classes' in metadata:
                        LABEL_ENCODER = LabelEncoder()
                        LABEL_ENCODER.classes_ = np.array(metadata['classes'])
                        print("   ✓ Label encoder created from metadata")
            
            # Method 3: Try to get from pipeline's classes_ attribute
            if LABEL_ENCODER is None and hasattr(pipeline, 'classes_'):
                LABEL_ENCODER = LabelEncoder()
                LABEL_ENCODER.classes_ = pipeline.classes_
                print("   ✓ Label encoder created from model classes")
            
            # Method 4: Default classes for sentiment analysis
            if LABEL_ENCODER is None:
                LABEL_ENCODER = LabelEncoder()
                LABEL_ENCODER.classes_ = np.array(['negative', 'neutral', 'positive'])
                print("   ⚠ Using default label encoder (negative, neutral, positive)")
        
        # Load metadata
        metadata_file = latest_model.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                MODEL_INFO = json.load(f)
        else:
            MODEL_INFO = {
                'model_name': os.path.basename(latest_model),
                'loaded_at': datetime.now().isoformat()
            }
        
        print("✅ Model loaded successfully!")
        print(f"   Model: {MODEL_INFO.get('model_name', 'Unknown')}")
        print(f"   Classes: {LABEL_ENCODER.classes_.tolist() if LABEL_ENCODER else 'Unknown'}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def predict_sentiment(text, return_proba=False):
    """
    Predict sentiment for a given text.
    
    Args:
        text: Input text in Khmer
        return_proba: If True, return probability scores for all classes
    
    Returns:
        Dictionary with prediction and optionally probabilities
    """
    try:
        # Preprocess text
        processed_text = preprocess_khmer(text)
        
        if not processed_text.strip():
            return {
                'error': 'Text is empty after preprocessing',
                'original_text': text
            }
        
        # The model is a pipeline that includes TF-IDF transformation
        # So we just pass the processed text directly to predict
        prediction_raw = MODEL.predict([processed_text])[0]
        
        # Handle prediction - it might be a string or an index
        if isinstance(prediction_raw, str):
            sentiment = prediction_raw
            prediction_id = list(LABEL_ENCODER.classes_).index(sentiment)
        else:
            # It's a numeric prediction
            prediction_id = int(prediction_raw)
            # Map to sentiment label
            if prediction_id < len(LABEL_ENCODER.classes_):
                sentiment = LABEL_ENCODER.classes_[prediction_id]
            else:
                sentiment = LABEL_ENCODER.classes_[0]  # Fallback
        
        result = {
            'text': text,
            'processed_text': processed_text,
            'sentiment': sentiment,
            'prediction_id': prediction_id
        }
        
        # Add probabilities if requested
        if return_proba and hasattr(MODEL, 'predict_proba'):
            probabilities = MODEL.predict_proba([processed_text])[0]
            result['probabilities'] = {
                class_name: float(prob)
                for class_name, prob in zip(LABEL_ENCODER.classes_, probabilities)
            }
            result['confidence'] = float(max(probabilities))
        
        return result
    
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'text': text
        }

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Khmer Sentiment Analysis API</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .info-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            font-family: 'Khmer OS', Arial, sans-serif;
            resize: vertical;
            box-sizing: border-box;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .checkbox-group {
            margin: 15px 0;
        }
        .checkbox-group label {
            display: inline;
            font-weight: normal;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            font-weight: 600;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 8px;
            display: none;
        }
        .result.success {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        .result.error {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        .sentiment {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .sentiment.positive { color: #28a745; }
        .sentiment.negative { color: #dc3545; }
        .sentiment.neutral { color: #ffc107; }
        .probabilities {
            margin-top: 15px;
        }
        .prob-bar {
            margin: 10px 0;
        }
        .prob-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 14px;
        }
        .prob-track {
            background: #e9ecef;
            height: 25px;
            border-radius: 12px;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 12px;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: 600;
        }
        .prob-fill.positive { background: linear-gradient(90deg, #28a745, #20c997); }
        .prob-fill.negative { background: linear-gradient(90deg, #dc3545, #e74c3c); }
        .prob-fill.neutral { background: linear-gradient(90deg, #ffc107, #ffb300); }
        .endpoints {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .endpoint {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🇰🇭 Khmer Sentiment Analysis</h1>
        <p class="subtitle">AI-Powered Sentiment Detection for Khmer Language</p>
        
        <div class="info-box">
            <strong>📊 Model Info:</strong><br>
            Model: {{ model_name }}<br>
            Classes: {{ classes }}<br>
            Status: <span style="color: #28a745;">✅ Ready</span>
        </div>
        
        <div class="input-group">
            <label for="text">Enter Khmer Text:</label>
            <textarea id="text" rows="5" placeholder="សូមបញ្ចូលអត្ថបទជាភាសាខ្មែរនៅទីនេះ..."></textarea>
        </div>
        
        <div class="checkbox-group">
            <input type="checkbox" id="show_proba" checked>
            <label for="show_proba">Show probability scores</label>
        </div>
        
        <button onclick="analyzeSentiment()">🔍 Analyze Sentiment</button>
        
        <div id="result" class="result">
            <div id="result-content"></div>
        </div>
        
        <div class="endpoints">
            <h3>📡 API Endpoints</h3>
            <div class="endpoint">
                <strong>POST /predict</strong><br>
                <code>{"text": "your_text", "return_proba": true}</code>
            </div>
            <div class="endpoint">
                <strong>POST /predict/batch</strong><br>
                <code>{"texts": ["text1", "text2"], "return_proba": true}</code>
            </div>
            <div class="endpoint">
                <strong>GET /health</strong><br>
                Check API status
            </div>
        </div>
    </div>
    
    <script>
        async function analyzeSentiment() {
            const text = document.getElementById('text').value;
            const showProba = document.getElementById('show_proba').checked;
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('result-content');
            
            if (!text.trim()) {
                resultDiv.className = 'result error';
                resultContent.innerHTML = '❌ Please enter some text';
                resultDiv.style.display = 'block';
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        return_proba: showProba
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.className = 'result error';
                    resultContent.innerHTML = `❌ Error: ${data.error}`;
                } else {
                    resultDiv.className = 'result success';
                    let html = `
                        <div class="sentiment ${data.sentiment}">
                            Sentiment: ${data.sentiment.toUpperCase()}
                        </div>
                        <p><strong>Original:</strong> ${data.text}</p>
                        <p><strong>Processed:</strong> ${data.processed_text}</p>
                    `;
                    
                    if (data.probabilities) {
                        html += '<div class="probabilities"><strong>Confidence Scores:</strong>';
                        for (const [sentiment, prob] of Object.entries(data.probabilities)) {
                            const percentage = (prob * 100).toFixed(1);
                            html += `
                                <div class="prob-bar">
                                    <div class="prob-label">
                                        <span>${sentiment}</span>
                                        <span>${percentage}%</span>
                                    </div>
                                    <div class="prob-track">
                                        <div class="prob-fill ${sentiment}" style="width: ${percentage}%">
                                            ${percentage}%
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                        html += '</div>';
                    }
                    
                    resultContent.innerHTML = html;
                }
                
                resultDiv.style.display = 'block';
            } catch (error) {
                resultDiv.className = 'result error';
                resultContent.innerHTML = `❌ Error: ${error.message}`;
                resultDiv.style.display = 'block';
            }
        }
        
        // Allow Enter key to submit (with Ctrl/Cmd)
        document.getElementById('text').addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                analyzeSentiment();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the web interface."""
    return render_template_string(
        HTML_TEMPLATE,
        model_name=MODEL_INFO.get('model_name', 'Unknown'),
        classes=', '.join(LABEL_ENCODER.classes_.tolist()) if LABEL_ENCODER else 'Unknown'
    )

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_info': MODEL_INFO,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a single text.
    
    Request body:
    {
        "text": "ខ្ញុំពិតជាសប្បាយចិត្តណាស់",
        "return_proba": true  // optional, default false
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        return_proba = data.get('return_proba', False)
        
        result = predict_sentiment(text, return_proba=return_proba)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple texts.
    
    Request body:
    {
        "texts": ["text1", "text2", "text3"],
        "return_proba": true  // optional, default false
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts'
            }), 400
        
        texts = data['texts']
        return_proba = data.get('return_proba', False)
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts must be a list'
            }), 400
        
        results = []
        for text in texts:
            result = predict_sentiment(text, return_proba=return_proba)
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get detailed model information."""
    return jsonify({
        'model_info': MODEL_INFO,
        'classes': LABEL_ENCODER.classes_.tolist() if LABEL_ENCODER else [],
        'model_type': type(MODEL).__name__ if MODEL else None,
        'has_predict_proba': hasattr(MODEL, 'predict_proba') if MODEL else False
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Khmer Sentiment Analysis API")
    print("="*60)
    
    # Load model on startup
    if load_best_model():
        print("\n✅ Starting Flask server...")
        print("   URL: http://localhost:5000")
        print("   Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True
        )
    else:
        print("\n❌ Failed to load model. Please check the models directory.")
        print("   Make sure you have trained and saved a model first.")
