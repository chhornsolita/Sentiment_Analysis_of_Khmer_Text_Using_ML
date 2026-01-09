"""
Khmer Text Preprocessing Module

This module provides functions for preprocessing Khmer text data,
including Unicode normalization, slang handling, and text cleaning.
"""

import re
import unicodedata
from typing import Dict, Optional


# Common Khmer slang/informal mapping (expand this with more slangs)
KHMER_SLANG = {
    "មិនចេះ": "មិនដឹង",
    "ចេះតែ": "តែងតែ",
    "អត់": "មិន",
    "ហ៊ាន": "ហ៊ាន",
    # Add more Khmer slang mappings as you discover them
}


def preprocess_khmer(text: str, slang_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Preprocess Khmer text for sentiment analysis.
    
    Steps:
    1. Unicode normalization (NFC for Khmer)
    2. Lowercase conversion
    3. URL removal
    4. Special markers removal
    5. Khmer slang normalization
    6. Emoji and symbol removal (keep Khmer characters, numbers, punctuation)
    7. Extra whitespace removal
    
    Args:
        text: Raw Khmer text string
        slang_dict: Optional custom slang dictionary (defaults to KHMER_SLANG)
        
    Returns:
        Cleaned and normalized Khmer text
        
    Example:
        >>> preprocess_khmer("អត់ល្អទេ http://example.com")
        'មិនល្អទេ'
    """
    if slang_dict is None:
        slang_dict = KHMER_SLANG
    
    # Unicode normalization (NFC for Khmer)
    text = unicodedata.normalize("NFC", text)
    
    # Lowercase (safe for Khmer + English mix)
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove special markers like :://, //, etc.
    text = re.sub(r"[:]+\/\/|\/\/", " ", text)
    
    # Handle Khmer slang (replace with standard form)
    for slang, standard in slang_dict.items():
        text = text.replace(slang, standard)
    
    # Remove emojis & symbols (keep Khmer + numbers + basic punctuation)
    text = re.sub(r"[^\u1780-\u17FF0-9\s\?\!\.]", " ", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def batch_preprocess(texts: list, slang_dict: Optional[Dict[str, str]] = None) -> list:
    """
    Apply preprocessing to a batch of texts.
    
    Args:
        texts: List of raw text strings
        slang_dict: Optional custom slang dictionary
        
    Returns:
        List of preprocessed texts
    """
    return [preprocess_khmer(text, slang_dict) for text in texts]
