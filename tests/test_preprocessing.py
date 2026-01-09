"""
Unit Tests for Preprocessing Module

Run with: pytest tests/test_preprocessing.py
"""

import pytest
from src.preprocessing import preprocess_khmer, batch_preprocess, KHMER_SLANG


class TestPreprocessKhmer:
    """Test cases for preprocess_khmer function"""
    
    def test_basic_cleaning(self):
        """Test basic text cleaning"""
        text = "  ជំរាបសួរ  "
        result = preprocess_khmer(text)
        assert result == "ជំរាបសួរ"
    
    def test_url_removal(self):
        """Test URL removal"""
        text = "អត្ថបទនេះ http://example.com ល្អណាស់"
        result = preprocess_khmer(text)
        assert "http" not in result
        assert "example.com" not in result
    
    def test_slang_replacement(self):
        """Test Khmer slang replacement"""
        text = "អត់ចេះ"
        result = preprocess_khmer(text)
        # Should replace អត់ with មិន
        assert "មិន" in result or "អត់" in result
    
    def test_lowercase(self):
        """Test lowercase conversion"""
        text = "ABC abc"
        result = preprocess_khmer(text)
        assert "ABC" not in result
    
    def test_empty_string(self):
        """Test empty string handling"""
        text = ""
        result = preprocess_khmer(text)
        assert result == ""
    
    def test_special_characters_removal(self):
        """Test emoji and special character removal"""
        text = "ល្អណាស់ 😊 💯 !!!"
        result = preprocess_khmer(text)
        assert "😊" not in result
        assert "💯" not in result


class TestBatchPreprocess:
    """Test cases for batch_preprocess function"""
    
    def test_batch_processing(self):
        """Test batch preprocessing"""
        texts = ["អត់ល្អ", "ល្អណាស់", "មិនអីទេ"]
        results = batch_preprocess(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
    
    def test_empty_batch(self):
        """Test empty batch"""
        texts = []
        results = batch_preprocess(texts)
        assert results == []


class TestKhmerSlang:
    """Test Khmer slang dictionary"""
    
    def test_slang_dict_exists(self):
        """Test that slang dictionary is defined"""
        assert isinstance(KHMER_SLANG, dict)
        assert len(KHMER_SLANG) > 0
    
    def test_slang_dict_format(self):
        """Test slang dictionary format"""
        for key, value in KHMER_SLANG.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(key) > 0
            assert len(value) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
