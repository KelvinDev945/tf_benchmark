"""
Tests for model loaders.

Note: These tests may download pretrained models.
Use mock data where possible to speed up tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf


class TestModelLoader:
    """Tests for ModelLoader."""

    def test_image_models_registry(self):
        """Test that image models registry is properly defined."""
        from src.models import ModelLoader

        assert len(ModelLoader.IMAGE_MODELS) > 0
        assert "mobilenet_v2" in ModelLoader.IMAGE_MODELS
        assert "resnet50" in ModelLoader.IMAGE_MODELS
        assert "efficientnet_b0" in ModelLoader.IMAGE_MODELS

        # Check structure
        for model_name, model_info in ModelLoader.IMAGE_MODELS.items():
            assert "class" in model_info
            assert "input_shape" in model_info
            assert len(model_info["input_shape"]) == 3

    def test_text_models_registry(self):
        """Test that text models registry is properly defined."""
        from src.models import ModelLoader

        assert len(ModelLoader.TEXT_MODELS) > 0
        assert "distilbert-base-uncased" in ModelLoader.TEXT_MODELS
        assert "bert-base-uncased" in ModelLoader.TEXT_MODELS

        # Check structure
        for model_name, model_info in ModelLoader.TEXT_MODELS.items():
            assert "max_length" in model_info
            assert "num_labels" in model_info

    def test_load_image_model_invalid_name(self):
        """Test loading image model with invalid name."""
        from src.models import ModelLoader

        with pytest.raises(ValueError, match="Unsupported image model"):
            ModelLoader.load_image_model("nonexistent_model")

    def test_load_text_model_invalid_name(self):
        """Test loading text model with invalid name."""
        from src.models import ModelLoader

        with pytest.raises(ValueError, match="Unsupported text model"):
            ModelLoader.load_text_model("nonexistent_model")

    def test_get_input_shape_image(self):
        """Test getting input shape for image models."""
        from src.models import ModelLoader

        shape = ModelLoader.get_input_shape("mobilenet_v2", "image")
        assert shape == (224, 224, 3)

        shape = ModelLoader.get_input_shape("resnet50", "image")
        assert shape == (224, 224, 3)

    def test_get_input_shape_text(self):
        """Test getting input shape for text models."""
        from src.models import ModelLoader

        shape = ModelLoader.get_input_shape("bert-base-uncased", "text")
        assert shape == (512,)

    def test_get_input_shape_invalid_type(self):
        """Test getting input shape with invalid model type."""
        from src.models import ModelLoader

        with pytest.raises(ValueError, match="Unsupported model_type"):
            ModelLoader.get_input_shape("mobilenet_v2", "invalid")

    def test_create_dummy_input_image(self):
        """Test creating dummy input for image models."""
        from src.models import ModelLoader

        dummy_input = ModelLoader.create_dummy_input("mobilenet_v2", "image", batch_size=4)

        assert isinstance(dummy_input, np.ndarray)
        assert dummy_input.shape == (4, 224, 224, 3)
        assert dummy_input.dtype == np.float32

    def test_create_dummy_input_text(self):
        """Test creating dummy input for text models."""
        from src.models import ModelLoader

        dummy_input = ModelLoader.create_dummy_input("bert-base-uncased", "text", batch_size=4)

        assert isinstance(dummy_input, dict)
        assert "input_ids" in dummy_input
        assert "attention_mask" in dummy_input
        assert dummy_input["input_ids"].shape == (4, 512)
        assert dummy_input["attention_mask"].shape == (4, 512)

    def test_get_model_info(self):
        """Test getting model information."""
        from src.models import ModelLoader

        # Create a simple test model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        info = ModelLoader.get_model_info(model)

        assert "num_parameters" in info
        assert "num_trainable_params" in info
        assert "num_layers" in info
        assert "input_shape" in info
        assert "output_shape" in info
        assert "model_size_mb" in info

        assert info["num_parameters"] > 0
        assert info["num_layers"] == 4
        assert info["model_size_mb"] > 0

    def test_verify_model_success(self):
        """Test model verification with valid input."""
        from src.models import ModelLoader

        # Create a simple test model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(10,)),
                tf.keras.layers.Dense(5, activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )

        # Create sample input
        sample_input = np.random.rand(1, 10).astype(np.float32)

        # Verify
        result = ModelLoader.verify_model(model, sample_input, expected_output_shape=(1, 2))

        assert result is True

    def test_verify_model_wrong_shape(self):
        """Test model verification with wrong expected shape."""
        from src.models import ModelLoader

        # Create a simple test model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(10,)),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )

        # Create sample input
        sample_input = np.random.rand(1, 10).astype(np.float32)

        # Verify with wrong expected shape
        result = ModelLoader.verify_model(model, sample_input, expected_output_shape=(1, 5))

        assert result is False


class TestModelLoaderIntegration:
    """Integration tests for model loading (can be slow, marked as integration)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_load_mobilenet_v2(self):
        """Test loading MobileNetV2 model."""
        from src.models import ModelLoader

        # This test actually downloads model - skip in CI
        pytest.skip("Skipping integration test (requires network and storage)")

        model = ModelLoader.load_image_model("mobilenet_v2")

        assert model is not None
        assert isinstance(model, tf.keras.Model)

        # Test inference
        dummy_input = ModelLoader.create_dummy_input("mobilenet_v2", "image")
        output = model(dummy_input)

        assert output.shape == (1, 1000)  # ImageNet classes

    @pytest.mark.integration
    @pytest.mark.slow
    def test_load_bert(self):
        """Test loading BERT model."""
        from src.models import ModelLoader

        # This test actually downloads model - skip in CI
        pytest.skip("Skipping integration test (requires network and storage)")

        model = ModelLoader.load_text_model("bert-base-uncased")

        assert model is not None

        # Test inference
        dummy_input = ModelLoader.create_dummy_input("bert-base-uncased", "text")
        output = model(dummy_input)

        assert hasattr(output, "logits")


class TestModelConverter:
    """Tests for ModelConverter (placeholder tests for Phase 3)."""

    def test_to_tflite_not_implemented(self):
        """Test that TFLite conversion raises NotImplementedError."""
        from src.models.model_converter import ModelConverter

        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])

        with pytest.raises(NotImplementedError, match="Phase 3"):
            ModelConverter.to_tflite(model)

    def test_to_onnx_not_implemented(self):
        """Test that ONNX conversion raises NotImplementedError."""
        from src.models.model_converter import ModelConverter

        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])

        with pytest.raises(NotImplementedError, match="Phase 3"):
            ModelConverter.to_onnx(model, "output.onnx")

    def test_to_openvino_not_implemented(self):
        """Test that OpenVINO conversion raises NotImplementedError."""
        from src.models.model_converter import ModelConverter

        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])

        with pytest.raises(NotImplementedError, match="Phase 3"):
            ModelConverter.to_openvino(model, "output")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
