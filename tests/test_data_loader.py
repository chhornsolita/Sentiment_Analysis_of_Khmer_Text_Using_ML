"""
Unit Tests for Data Loader Module

Run with: pytest tests/test_data_loader.py
"""

import pytest
import pandas as pd
from src.data_loader import clean_data, get_class_distribution, prepare_train_test_split


class TestCleanData:
    """Test cases for clean_data function"""
    
    def test_remove_header_duplicates(self):
        """Test removal of header duplicates"""
        df = pd.DataFrame({
            'text': ['text', 'អត្ថបទ១', 'អត្ថបទ២'],
            'target': ['target', 'pos', 'neg']
        })
        
        result = clean_data(df, 'text')
        assert len(result) == 2
        assert 'text' not in result['text'].values
    
    def test_remove_null_values(self):
        """Test removal of null values"""
        df = pd.DataFrame({
            'text': ['អត្ថបទ១', None, 'អត្ថបទ២'],
            'target': ['pos', 'neg', 'neu']
        })
        
        result = clean_data(df, 'text')
        assert len(result) == 2
        assert result['text'].isnull().sum() == 0
    
    def test_index_reset(self):
        """Test index reset"""
        df = pd.DataFrame({
            'text': ['អត្ថបទ១', 'អត្ថបទ២', 'អត្ថបទ៣'],
            'target': ['pos', 'neg', 'neu']
        })
        
        result = clean_data(df, 'text')
        assert list(result.index) == list(range(len(result)))


class TestGetClassDistribution:
    """Test cases for get_class_distribution function"""
    
    def test_class_counts(self):
        """Test class distribution counting"""
        y = pd.Series(['pos', 'pos', 'neg', 'neu', 'pos'])
        result = get_class_distribution(y)
        
        assert result['pos'] == 3
        assert result['neg'] == 1
        assert result['neu'] == 1


class TestPrepareTrainTestSplit:
    """Test cases for prepare_train_test_split function"""
    
    def test_split_sizes(self):
        """Test train-test split sizes"""
        df = pd.DataFrame({
            'text_clean': [f'text_{i}' for i in range(100)],
            'target': ['pos' if i % 3 == 0 else 'neg' if i % 3 == 1 else 'neu' 
                      for i in range(100)]
        })
        
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_stratification(self):
        """Test stratified splitting"""
        df = pd.DataFrame({
            'text_clean': [f'text_{i}' for i in range(300)],
            'target': ['pos']*100 + ['neg']*100 + ['neu']*100
        })
        
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df, test_size=0.2, random_state=42, stratify=True
        )
        
        # Check approximate class balance
        train_dist = y_train.value_counts()
        test_dist = y_test.value_counts()
        
        assert all(train_dist[c] >= 75 for c in ['pos', 'neg', 'neu'])
        assert all(test_dist[c] >= 15 for c in ['pos', 'neg', 'neu'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
