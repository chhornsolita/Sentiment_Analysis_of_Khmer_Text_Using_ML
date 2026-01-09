---
title: Khmer Sentiment Analysis
<<<<<<< HEAD
emoji: 🇰🇭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app_gradio.py
pinned: false
license: mit
---

# 🇰🇭 Khmer Text Sentiment Analysis

An advanced machine learning application for analyzing sentiment in Khmer (Cambodian) text. This project uses TF-IDF vectorization and traditional machine learning models to classify Khmer text into three sentiment categories: Positive, Neutral, and Negative.

## 🌟 Features

- **Khmer Language Support**: Specialized preprocessing for Khmer text
- **Real-time Predictions**: Instant sentiment analysis through an intuitive interface
- **Confidence Scores**: Detailed probability breakdown for all sentiment classes
- **Slang Handling**: Recognizes and processes common Khmer slang and phrases
- **User-Friendly Interface**: Clean and responsive Gradio-based UI

## 🎯 How It Works

1. **Text Input**: Enter Khmer text in the input box
2. **Preprocessing**: Text is cleaned and normalized for Khmer language
3. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
4. **Prediction**: Machine learning model classifies the sentiment
5. **Results**: View sentiment label with confidence scores for all classes

## 📊 Model Information

- **Algorithm**: Logistic Regression / Random Forest (best performing model)
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classes**: Positive, Neutral, Negative
- **Training Data**: Khmer sentiment analysis dataset

## 🚀 Usage

Simply enter your Khmer text in the input box and click "Analyze Sentiment" to get instant results.

### Example Texts:

**Positive:**

```
ល្អណាស់! វាពិតជាអស្ចារ្យ
(Very good! It's amazing)
```

**Negative:**

```
គួរឱ្យខកចិត្ត មិនល្អទេ
(Disappointing, not good)
```

**Neutral:**

```
ធម្មតាៗ មិនអីទេ
(Normal, it's okay)
```

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Scikit-learn**: Machine learning models and preprocessing
- **Gradio**: Interactive web interface
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Hugging Face Spaces**: Deployment platform

## 📁 Project Structure

```
├── app_gradio.py          # Gradio interface (main app)
├── src/
│   ├── preprocessing.py   # Khmer text preprocessing
│   ├── model_persistence.py # Model loading utilities
│   └── ...
├── models/
│   └── saved_models/      # Trained model files
├── data/                  # Training datasets
└── requirements_hf.txt    # Dependencies for Hugging Face
```

## 🎓 Academic Context

This project was developed as part of academic research in Natural Language Processing (NLP) for low-resource languages, specifically focusing on Khmer sentiment analysis.

### Key Challenges Addressed:

- **Limited Resources**: Khmer is a low-resource language with limited NLP tools
- **Complex Script**: Khmer script requires special preprocessing techniques
- **Slang and Informal Text**: Model handles colloquial expressions
- **Multi-class Classification**: Distinguishes between three sentiment categories

## 📈 Performance

The model has been evaluated on various metrics including:

- Accuracy
- Precision, Recall, and F1-Score
- ROC-AUC scores
- Confusion Matrix analysis

For detailed performance metrics, see the documentation in the repository.

## 🤝 Contributing

This is a research project. Feedback and suggestions are welcome!

## 📝 License

MIT License - Feel free to use this project for educational and research purposes.

## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/khmer-sentiment-analysis)
- [Documentation](./docs/)
- [API Reference](./docs/API_REFERENCE.md)

## ⚠️ Disclaimer

This model is designed for research and educational purposes. Results may vary based on text complexity and context. The model performs best on general sentiment expressions in Khmer text.

## 📧 Contact

For questions or collaborations, please open an issue in the GitHub repository.

---

**Built with ❤️ for the Khmer NLP community**
=======
emoji: 💻
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 34550782f58f490e29eca4c518ddf0dfd3c5f966
