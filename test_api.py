"""
Test script for the Khmer Sentiment Analysis API
Demonstrates how to interact with the API endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health endpoint."""
    print("\n" + "="*60)
    print("🔍 Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_single_prediction():
    """Test single text prediction."""
    print("\n" + "="*60)
    print("🔍 Testing Single Prediction")
    print("="*60)
    
    # Test cases in Khmer
    test_texts = [
        "ខ្ញុំពិតជាសប្បាយចិត្តណាស់",  # I'm very happy
        "វាអាក្រក់ណាស់",  # It's very bad
        "ធម្មតា",  # Normal
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        payload = {
            "text": text,
            "return_proba": True
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if 'error' not in result:
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            if 'probabilities' in result:
                print("Probabilities:")
                for sentiment, prob in result['probabilities'].items():
                    print(f"  {sentiment}: {prob:.4f}")
        else:
            print(f"Error: {result['error']}")

def test_batch_prediction():
    """Test batch prediction."""
    print("\n" + "="*60)
    print("🔍 Testing Batch Prediction")
    print("="*60)
    
    texts = [
        "ល្អណាស់",  # Very good
        "មិនល្អ",  # Not good
        "ធម្មតា",  # Normal
    ]
    
    payload = {
        "texts": texts,
        "return_proba": True
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"Total predictions: {result['count']}")
    
    for i, prediction in enumerate(result['predictions'], 1):
        print(f"\n--- Prediction {i} ---")
        print(f"Text: {prediction['text']}")
        print(f"Sentiment: {prediction['sentiment']}")
        if 'confidence' in prediction:
            print(f"Confidence: {prediction['confidence']:.4f}")

def test_model_info():
    """Test model info endpoint."""
    print("\n" + "="*60)
    print("🔍 Testing Model Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Khmer Sentiment Analysis API - Test Suite")
    print("="*60)
    print("\nMake sure the Flask API is running on http://localhost:5000")
    print("Run 'python app.py' in another terminal first!\n")
    
    try:
        # Run all tests
        test_health_check()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API.")
        print("   Make sure the Flask server is running:")
        print("   python app.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
