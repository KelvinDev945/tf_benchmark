"""
Tests for model loaders.

Note: These tests may download pretrained models.
Use mock data where possible to speed up tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

# Skip all tests in this module if src.models is not available
try:
    import src.models  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("src.models module not implemented yet", allow_module_level=True)


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

    def test_load_image_model_invalid_name(self):
        """Test loading image model with invalid name."""
        from src.models import ModelLoader

        with pytest.raises(ValueError, match="Unsupported image model"):
            ModelLoader.load_image_model("nonexistent_model")

    def test_text_models_registry(self):
        """Ensure text models registry contains expected entries."""
        from src.models import ModelLoader

        assert "bert-base-uncased" in ModelLoader.TEXT_MODELS
        info = ModelLoader.TEXT_MODELS["bert-base-uncased"]
        assert "hub_url" in info
        assert info["max_length"] == 512

    def test_load_text_model_invalid_name(self):
        """Loading an unsupported text model should raise ValueError."""
        from src.models import ModelLoader

        with pytest.raises(ValueError, match="Unsupported text model"):
            ModelLoader.load_text_model("unknown-text-model")

    def test_get_input_shape_text(self):
        """Text models should report their max sequence length."""
        from src.models import ModelLoader

        shape = ModelLoader.get_input_shape("bert-base-uncased", "text")
        assert shape == (512,)

    def test_create_dummy_input_text(self):
        """Dummy input for text models should be a dict with BERT keys."""
        from src.models import ModelLoader

        dummy_input = ModelLoader.create_dummy_input("bert-base-uncased", "text", batch_size=2)

        assert isinstance(dummy_input, dict)
        assert set(dummy_input.keys()) == {
            "input_word_ids",
            "input_mask",
            "input_type_ids",
        }
        for value in dummy_input.values():
            assert value.shape == (2, 512)
            assert value.dtype == np.int32

    def test_get_input_shape_image(self):
        """Test getting input shape for image models."""
        from src.models import ModelLoader

        shape = ModelLoader.get_input_shape("mobilenet_v2", "image")
        assert shape == (224, 224, 3)

        shape = ModelLoader.get_input_shape("resnet50", "image")
        assert shape == (224, 224, 3)

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

    @patch("src.models.model_loader.hub.load")
    def test_load_text_model_builds_keras_model(self, mock_hub_load):
        """Ensure load_text_model constructs a classifier when TF Hub is available."""
        import src.models.model_loader as model_loader_module
        from src.models import ModelLoader

        @tf.function
        def dummy_signature(  # type: ignore[no-untyped-def]
            input_word_ids, input_mask, input_type_ids
        ):
            batch_size = tf.shape(input_word_ids)[0]
            return {"bert_encoder": tf.ones((batch_size, 768), dtype=tf.float32)}

        mock_module = MagicMock()
        mock_module.signatures = {"serving_default": dummy_signature}
        mock_hub_load.return_value = mock_module
        model_loader_module.HUB_AVAILABLE = True

        model = ModelLoader.load_text_model("bert-base-uncased", num_labels=3)

        assert isinstance(model, tf.keras.Model)
        dummy_input = ModelLoader.create_dummy_input("bert-base-uncased", "text", batch_size=2)

        outputs = model(dummy_input, training=False)
        assert outputs.shape == (2, 3)

    @pytest.mark.skip(reason="TF 2.20 compatibility issue - layer count mismatch")
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


class TestModelConverter:
    """Tests for ModelConverter (placeholder tests for Phase 3)."""

    @pytest.mark.skip(reason="TF 2.20 compatibility - Sequential._get_save_spec missing")
    def test_to_tflite_not_implemented(self):
        """Test that TFLite conversion raises NotImplementedError."""
        from src.models.model_converter import ModelConverter

        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])

        with pytest.raises(NotImplementedError, match="Phase 3"):
            ModelConverter.to_tflite(model)

    @pytest.mark.skip(reason="TF 2.20 compatibility - Sequential missing input_shape")
    def test_to_onnx_not_implemented(self):
        """Test that ONNX conversion raises NotImplementedError."""
        from src.models.model_converter import ModelConverter

        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])

        with pytest.raises(NotImplementedError, match="Phase 3"):
            ModelConverter.to_onnx(model, "output.onnx")

    @pytest.mark.skip(reason="OpenVINO not installed - optional dependency")
    def test_to_openvino_not_implemented(self):
        """Test that OpenVINO conversion raises NotImplementedError."""
        from src.models.model_converter import ModelConverter

        model = tf.keras.Sequential([tf.keras.layers.Dense(10)])

        with pytest.raises(NotImplementedError, match="Phase 3"):
            ModelConverter.to_openvino(model, "output")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
