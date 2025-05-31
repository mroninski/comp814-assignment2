"""
Shared pytest configuration and fixtures for topic_extractor tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add the src directory to the Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def mock_word2vec_model():
    """Create a mock Word2Vec model for testing."""
    mock_model = MagicMock()
    mock_model.vector_size = 300

    # Common test words
    test_vocabulary = {
        "friends": 0,
        "crush": 1,
        "embarrassed": 2,
        "classmate": 3,
        "jealous": 4,
        "like": 5,
        "really": 6,
        "bad": 7,
        "nice": 8,
        "respiratory": 9,
        "cell": 10,
        "turtle": 11,
        "pleisiosaur": 12,
        "environment": 13,
        "body": 14,
        "exchange": 15,
        "gas": 16,
        "lung": 17,
        "oxygen": 18,
        "breath": 19,
        "technology": 20,
        "computers": 21,
        "internet": 22,
        "software": 23,
        "love": 24,
        "relationship": 25,
        "family": 26,
        "marriage": 27,
        "dating": 28,
    }

    mock_model.key_to_index = test_vocabulary

    def mock_getitem(word):
        """Mock __getitem__ method to return embeddings."""
        if word in test_vocabulary:
            # Create somewhat consistent embeddings based on word type
            if word in [
                "friends",
                "crush",
                "jealous",
                "love",
                "relationship",
                "dating",
                "marriage",
            ]:
                # Relationship words get similar embeddings
                base = np.array([0.8, 0.7, 0.6] + [0.1] * 297)
                return base + np.random.normal(0, 0.1, 300)
            elif word in ["respiratory", "cell", "lung", "oxygen", "breath"]:
                # Health words get similar embeddings
                base = np.array([0.1, 0.8, 0.7] + [0.1] * 297)
                return base + np.random.normal(0, 0.1, 300)
            elif word in ["technology", "computers", "internet", "software"]:
                # Tech words get similar embeddings
                base = np.array([0.7, 0.1, 0.8] + [0.1] * 297)
                return base + np.random.normal(0, 0.1, 300)
            else:
                # Other words get random embeddings
                return np.random.rand(300)
        else:
            raise KeyError(f"Word '{word}' not in vocabulary")

    mock_model.__getitem__ = mock_getitem
    return mock_model


@pytest.fixture
def mock_gensim_api():
    """Mock the gensim downloader API."""

    def mock_load(model_name):
        return mock_word2vec_model()

    with patch(
        "src.topic_extractor.topic_simplifying.api.load", side_effect=mock_load
    ) as mock:
        yield mock


# Test data fixtures
@pytest.fixture
def relationship_words():
    """Sample relationship-focused words."""
    return {
        "friends",
        "crush",
        "embarrassed",
        "classmate",
        "jealous",
        "like",
        "really",
        "bad",
        "nice",
        "love",
        "dating",
    }


@pytest.fixture
def scientific_words():
    """Sample scientific words."""
    return {
        "respiratory",
        "cell",
        "lung",
        "oxygen",
        "breath",
        "gas",
        "exchange",
        "turtle",
        "environment",
        "pleisiosaur",
    }


@pytest.fixture
def technology_words():
    """Sample technology words."""
    return {
        "technology",
        "computers",
        "internet",
        "software",
        "programming",
        "apps",
        "digital",
        "innovation",
    }
