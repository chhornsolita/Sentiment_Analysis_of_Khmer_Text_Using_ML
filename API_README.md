# 🌐 Khmer Sentiment Analysis - Flask API

A production-ready REST API for Khmer sentiment analysis using machine learning models.

## 🚀 Quick Start

### 1. Start the API Server

```bash
python app.py
```

The API will be available at: **http://localhost:5000**

### 2. Open Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

You'll see a beautiful web interface where you can:
- Enter Khmer text for sentiment analysis
- View real-time predictions with confidence scores
- See probability distributions across all sentiment classes

### 3. Test API Endpoints

Run the test script in another terminal:
```bash
python test_api.py
```

## 📡 API Endpoints

### 1. Web Interface
**GET /** 
- Opens the interactive web interface
- Visual sentiment analysis tool
- Real-time predictions

### 2. Health Check
**GET /health**

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_name": "best_model_logistic_regression",
    "accuracy": 0.85
  },
  "timestamp": "2025-12-26T12:00:00"
}
```

### 3. Single Prediction
**POST /predict**

Predict sentiment for a single text.

**Request:**
```json
{
  "text": "ខ្ញុំពិតជាសប្បាយចិត្តណាស់",
  "return_proba": true
}
```

**Response:**
```json
{
  "text": "ខ្ញុំពិតជាសប្បាយចិត្តណាស់",
  "processed_text": "ខ្ញុំ ពិតជា សប្បាយចិត្ត ណាស់",
  "sentiment": "positive",
  "prediction_id": 2,
  "probabilities": {
    "negative": 0.05,
    "neutral": 0.15,
    "positive": 0.80
  },
  "confidence": 0.80
}
```

### 4. Batch Prediction
**POST /predict/batch**

Predict sentiment for multiple texts at once.

**Request:**
```json
{
  "texts": [
    "ល្អណាស់",
    "មិនល្អ",
    "ធម្មតា"
  ],
  "return_proba": true
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "ល្អណាស់",
      "sentiment": "positive",
      "confidence": 0.85
    },
    {
      "text": "មិនល្អ",
      "sentiment": "negative",
      "confidence": 0.78
    },
    {
      "text": "ធម្មតា",
      "sentiment": "neutral",
      "confidence": 0.72
    }
  ],
  "count": 3
}
```

### 5. Model Information
**GET /model/info**

Get detailed information about the loaded model.

**Response:**
```json
{
  "model_info": {
    "model_name": "best_model_logistic_regression",
    "accuracy": 0.85,
    "f1_score": 0.83
  },
  "classes": ["negative", "neutral", "positive"],
  "model_type": "RandomizedSearchCV",
  "has_predict_proba": true
}
```

## 💻 Usage Examples

### Python (requests)

```python
import requests
import json

# Single prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'text': 'ខ្ញុំពិតជាសប្បាយចិត្តណាស់',
        'return_proba': True
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "ខ្ញុំពិតជាសប្បាយចិត្តណាស់", "return_proba": true}'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["ល្អណាស់", "មិនល្អ"], "return_proba": true}'
```

### JavaScript (fetch)

```javascript
// Single prediction
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'ខ្ញុំពិតជាសប្បាយចិត្តណាស់',
    return_proba: true
  })
});

const result = await response.json();
console.log(`Sentiment: ${result.sentiment}`);
console.log(`Confidence: ${result.confidence}`);
```

## 🎨 Web Interface Features

The interactive web interface provides:

- **Text Input**: Large text area for entering Khmer text
- **Real-time Analysis**: Instant sentiment prediction on button click
- **Visual Feedback**: Color-coded sentiment display
  - 🟢 Green for positive
  - 🔴 Red for negative  
  - 🟡 Yellow for neutral
- **Confidence Scores**: Animated progress bars showing probability distribution
- **API Documentation**: Built-in endpoint reference
- **Responsive Design**: Beautiful gradient design with modern UI

## 🔧 Configuration

### Change Port

Edit `app.py`:
```python
app.run(
    host='0.0.0.0',
    port=5000,  # Change this
    debug=True
)
```

### Load Specific Model

The API automatically loads the most recent model from `models/saved_models/`.

To load a specific model, modify the `load_best_model()` function in `app.py`:
```python
latest_model = 'models/saved_models/best_model_logistic_regression_20251226_125017.pkl'
```

### Enable/Disable Debug Mode

For production, set `debug=False`:
```python
app.run(debug=False)
```

## 📊 Performance

- **Single Prediction**: ~10-50ms per request
- **Batch Prediction**: ~100-300ms for 10 texts
- **Model Loading**: ~1-2 seconds on startup
- **Memory Usage**: ~200-500MB depending on model size

## 🛡️ Error Handling

The API handles common errors gracefully:

- **Empty text**: Returns error message
- **Invalid JSON**: Returns 400 Bad Request
- **Model not loaded**: Returns 500 Internal Server Error
- **Missing fields**: Returns 400 with descriptive error

Example error response:
```json
{
  "error": "Text is empty after preprocessing",
  "original_text": "!!!"
}
```

## 🚀 Deployment Tips

### For Production:

1. **Use Gunicorn** (Linux/Mac):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Use Waitress** (Windows):
```bash
pip install waitress
waitress-serve --port=5000 app:app
```

3. **Set Environment Variables**:
```bash
export FLASK_ENV=production
```

4. **Use HTTPS**: Deploy behind a reverse proxy (nginx/Apache)

5. **Add Rate Limiting**: Prevent API abuse

## 📝 Notes

- The API uses the best trained model from the `models/saved_models/` directory
- All text preprocessing is handled automatically
- The API supports Khmer Unicode characters (U+1780 - U+17FF)
- Confidence scores are returned when `return_proba=true`

## 🐛 Troubleshooting

**Issue: "No saved models found"**
- Solution: Run `python train.py` first to train and save a model

**Issue: "Connection refused"**
- Solution: Make sure the Flask server is running on port 5000

**Issue: "Module not found"**
- Solution: Make sure you're in the project root directory and all dependencies are installed

## 📞 Support

For issues or questions, check:
- [README_PROFESSIONAL.md](README_PROFESSIONAL.md) - Main project documentation
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - Detailed API reference
