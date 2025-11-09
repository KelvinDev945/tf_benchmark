"""
Tests for dataset loaders.

Note: These tests use mock data to avoid downloading large datasets.
Full integration tests can be run separately.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image


class TestImageDatasetLoader:
    """Tests for ImageDatasetLoader."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        from src.dataset import ImageDatasetLoader

        loader = ImageDatasetLoader(
            dataset_name="imagenet-1k",
            split="validation",
            num_samples=100,
            target_size=(224, 224),
        )

        assert loader.dataset_name == "imagenet-1k"
        assert loader.split == "validation"
        assert loader.num_samples == 100
        assert loader.target_size == (224, 224)

    def test_init_invalid_num_samples(self):
        """Test initialization with invalid num_samples."""
        from src.dataset import ImageDatasetLoader

        with pytest.raises(ValueError, match="num_samples must be positive"):
            ImageDatasetLoader(num_samples=-1)

    def test_init_invalid_target_size(self):
        """Test initialization with invalid target_size."""
        from src.dataset import ImageDatasetLoader

        with pytest.raises(ValueError, match="target_size must be"):
            ImageDatasetLoader(target_size=(224,))

        with pytest.raises(ValueError, match="dimensions must be positive"):
            ImageDatasetLoader(target_size=(224, -1))

    def test_preprocess_image(self):
        """Test image preprocessing."""
        from src.dataset import ImageDatasetLoader

        loader = ImageDatasetLoader()

        # Create a dummy image
        image = Image.new("RGB", (256, 256), color="red")

        # Preprocess
        processed = loader.preprocess(image, normalize=False)

        # Check output shape and type
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_preprocess_with_normalization(self):
        """Test image preprocessing with normalization."""
        from src.dataset import ImageDatasetLoader

        loader = ImageDatasetLoader()

        # Create a dummy image
        image = Image.new("RGB", (256, 256), color="white")

        # Preprocess with normalization
        processed = loader.preprocess(image, normalize=True)

        # Check that normalization was applied (values can be negative)
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
        # After ImageNet normalization, values should be different from [0, 1]
        assert not (processed.min() >= 0.0 and processed.max() <= 1.0)

    def test_repr(self):
        """Test string representation."""
        from src.dataset import ImageDatasetLoader

        loader = ImageDatasetLoader(dataset_name="test-dataset", split="train", num_samples=50)

        repr_str = repr(loader)
        assert "ImageDatasetLoader" in repr_str
        assert "test-dataset" in repr_str
        assert "train" in repr_str


class TestTextDatasetLoader:
    """Tests for TextDatasetLoader."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        from src.dataset import TextDatasetLoader

        loader = TextDatasetLoader(
            dataset_name="glue",
            subset="sst2",
            split="validation",
            tokenizer="bert-base-uncased",
            num_samples=100,
            max_length=128,
        )

        assert loader.dataset_name == "glue"
        assert loader.subset == "sst2"
        assert loader.split == "validation"
        assert loader.num_samples == 100
        assert loader.max_length == 128

    def test_init_invalid_num_samples(self):
        """Test initialization with invalid num_samples."""
        from src.dataset import TextDatasetLoader

        with pytest.raises(ValueError, match="num_samples must be positive"):
            TextDatasetLoader(num_samples=-1)

    def test_init_invalid_max_length(self):
        """Test initialization with invalid max_length."""
        from src.dataset import TextDatasetLoader

        with pytest.raises(ValueError, match="max_length must be positive"):
            TextDatasetLoader(max_length=-1)

    def test_tokenize_single_text(self):
        """Test tokenization of single text."""
        from src.dataset import TextDatasetLoader

        loader = TextDatasetLoader()

        text = "This is a test sentence."
        encoding = loader.tokenize(text, max_length=32, padding="max_length")

        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert len(encoding["input_ids"]) == 32
        assert len(encoding["attention_mask"]) == 32

    def test_tokenize_batch_texts(self):
        """Test tokenization of batch of texts."""
        from src.dataset import TextDatasetLoader

        loader = TextDatasetLoader()

        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        encoding = loader.tokenize(texts, max_length=32, padding="max_length")

        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert len(encoding["input_ids"]) == 3
        assert all(len(ids) == 32 for ids in encoding["input_ids"])

    def test_repr(self):
        """Test string representation."""
        from src.dataset import TextDatasetLoader

        loader = TextDatasetLoader(
            dataset_name="glue", subset="sst2", split="train", num_samples=50
        )

        repr_str = repr(loader)
        assert "TextDatasetLoader" in repr_str
        assert "glue/sst2" in repr_str
        assert "train" in repr_str


class TestDatasetIntegration:
    """Integration tests for dataset loaders (can be slow, marked as integration)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_image_dataset_load_small_sample(self):
        """Test loading a small sample of image dataset."""
        from src.dataset import ImageDatasetLoader

        # This test actually downloads data - skip in CI
        pytest.skip("Skipping integration test (requires network)")

        loader = ImageDatasetLoader(num_samples=10)
        loader.load()

        assert len(loader) == 10
        assert loader.dataset is not None

        stats = loader.get_stats()
        assert stats["num_samples"] == 10

    @pytest.mark.integration
    @pytest.mark.slow
    def test_text_dataset_load_small_sample(self):
        """Test loading a small sample of text dataset."""
        from src.dataset import TextDatasetLoader

        # This test actually downloads data - skip in CI
        pytest.skip("Skipping integration test (requires network)")

        loader = TextDatasetLoader(num_samples=10)
        loader.load()

        assert len(loader) == 10
        assert loader.dataset is not None

        stats = loader.get_stats()
        assert stats["num_samples"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
