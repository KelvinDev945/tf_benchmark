"""
Image Dataset Loader for TensorFlow Benchmark.

This module provides functionality to load and preprocess image datasets
from HuggingFace for benchmarking image classification models.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from PIL import Image


class ImageDatasetLoader:
    """
    Image dataset loader for benchmark testing.

    Loads image datasets from HuggingFace, applies preprocessing,
    and provides TensorFlow Dataset interface for benchmarking.
    """

    # ImageNet mean and std for normalization
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    def __init__(
        self,
        dataset_name: str = "imagenet-1k",
        split: str = "validation",
        num_samples: Optional[int] = None,
        target_size: Tuple[int, int] = (224, 224),
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize ImageDatasetLoader.

        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Dataset split to use ('train', 'validation', 'test')
            num_samples: Number of samples to load (None for all)
            target_size: Target image size (height, width)
            cache_dir: Directory to cache dataset
            use_cache: Whether to use caching

        Raises:
            ValueError: If parameters are invalid
        """
        self.dataset_name = dataset_name
        self.split = split
        self.num_samples = num_samples
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache

        self.dataset = None
        self._stats: Dict = {}

        self._validate_params()

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        if len(self.target_size) != 2:
            raise ValueError(
                f"target_size must be (height, width), got {self.target_size}"
            )

        if any(s <= 0 for s in self.target_size):
            raise ValueError(
                f"target_size dimensions must be positive, got {self.target_size}"
            )

    def load(self) -> "ImageDatasetLoader":
        """
        Load the dataset from HuggingFace.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            print(f"Loading {self.dataset_name} dataset ({self.split} split)...")

            # Load dataset from HuggingFace
            self.dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=True,
            )

            # Limit number of samples if specified
            if self.num_samples is not None:
                total_samples = len(self.dataset)
                if self.num_samples < total_samples:
                    self.dataset = self.dataset.select(range(self.num_samples))
                    print(
                        f"Limited to {self.num_samples} samples "
                        f"(out of {total_samples} available)"
                    )

            # Collect statistics
            self._collect_stats()

            print(f"✓ Loaded {len(self.dataset)} samples")

            return self

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}") from e

    def _collect_stats(self) -> None:
        """Collect dataset statistics."""
        if self.dataset is None:
            return

        self._stats = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "num_samples": len(self.dataset),
            "target_size": self.target_size,
            "features": list(self.dataset.features.keys()),
        }

    def preprocess(
        self, image: Image.Image, normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess a single image.

        Args:
            image: PIL Image to preprocess
            normalize: Whether to apply ImageNet normalization

        Returns:
            Preprocessed image as numpy array with shape (H, W, 3)
        """
        # Resize image
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(self.target_size, Image.BILINEAR)

        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        image_array = image_array / 255.0

        # Apply ImageNet normalization if requested
        if normalize:
            image_array = (image_array - self.IMAGENET_MEAN) / self.IMAGENET_STD

        return image_array

    def _preprocess_batch(self, batch: Dict) -> Dict:
        """
        Preprocess a batch of images.

        Args:
            batch: Batch dictionary from HuggingFace dataset

        Returns:
            Preprocessed batch dictionary
        """
        # Get images (handle different dataset formats)
        if "image" in batch:
            images = batch["image"]
        elif "img" in batch:
            images = batch["img"]
        else:
            raise ValueError(
                f"Cannot find image field in batch. Available fields: {batch.keys()}"
            )

        # Preprocess all images in batch
        processed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                processed = self.preprocess(img)
            else:
                # If already an array, convert to PIL first
                img_pil = Image.fromarray(np.array(img))
                processed = self.preprocess(img_pil)
            processed_images.append(processed)

        # Stack into batch
        batch["image"] = np.stack(processed_images, axis=0)

        # Handle labels if present
        if "label" in batch:
            batch["label"] = np.array(batch["label"], dtype=np.int32)

        return batch

    def get_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        prefetch: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        """
        Get TensorFlow Dataset for benchmarking.

        Args:
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            prefetch: Whether to enable prefetching
            num_parallel_calls: Number of parallel preprocessing calls

        Returns:
            tf.data.Dataset ready for inference

        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # Convert to TensorFlow dataset
        def generator():
            """Generator function for tf.data.Dataset."""
            for sample in self.dataset:
                # Get image
                if "image" in sample:
                    image = sample["image"]
                elif "img" in sample:
                    image = sample["img"]
                else:
                    raise ValueError(
                        f"Cannot find image field. Available: {sample.keys()}"
                    )

                # Preprocess
                if isinstance(image, Image.Image):
                    processed = self.preprocess(image)
                else:
                    img_pil = Image.fromarray(np.array(image))
                    processed = self.preprocess(img_pil)

                # Yield image and label (if available)
                if "label" in sample:
                    yield processed, sample["label"]
                else:
                    yield processed

        # Determine output signature
        output_signature = (
            tf.TensorSpec(shape=(*self.target_size, 3), dtype=tf.float32),
        )
        if "label" in self.dataset.features:
            output_signature = (
                tf.TensorSpec(shape=(*self.target_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )

        # Create TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )

        # Shuffle if requested
        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=1000)

        # Batch the dataset
        tf_dataset = tf_dataset.batch(batch_size)

        # Prefetch for performance
        if prefetch:
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        return tf_dataset

    def get_numpy_batches(
        self, batch_size: int = 32, max_batches: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get preprocessed images as numpy arrays (for non-TF engines).

        Args:
            batch_size: Batch size
            max_batches: Maximum number of batches to return (None for all)

        Returns:
            Tuple of (images, labels) where images has shape (N, H, W, 3)
            and labels has shape (N,) if available, else None

        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        all_images = []
        all_labels = []
        has_labels = "label" in self.dataset.features

        num_batches = 0
        for i in range(0, len(self.dataset), batch_size):
            if max_batches is not None and num_batches >= max_batches:
                break

            batch = self.dataset[i : i + batch_size]

            # Get images
            if "image" in batch:
                images = batch["image"]
            elif "img" in batch:
                images = batch["img"]
            else:
                raise ValueError(
                    f"Cannot find image field. Available: {batch.keys()}"
                )

            # Preprocess
            processed_batch = []
            for img in images:
                if isinstance(img, Image.Image):
                    processed = self.preprocess(img)
                else:
                    img_pil = Image.fromarray(np.array(img))
                    processed = self.preprocess(img_pil)
                processed_batch.append(processed)

            all_images.extend(processed_batch)

            # Get labels if available
            if has_labels:
                all_labels.extend(batch["label"])

            num_batches += 1

        images_array = np.array(all_images, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32) if has_labels else None

        return images_array, labels_array

    def get_stats(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        return self._stats.copy()

    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        if self.dataset is None:
            return 0
        return len(self.dataset)

    def __repr__(self) -> str:
        """String representation of ImageDatasetLoader."""
        return (
            f"ImageDatasetLoader(dataset='{self.dataset_name}', "
            f"split='{self.split}', samples={len(self)}, "
            f"size={self.target_size})"
        )
