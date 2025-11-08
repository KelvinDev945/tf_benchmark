"""
Text Dataset Loader for TensorFlow Benchmark.

This module provides functionality to load and preprocess text datasets
from HuggingFace for benchmarking text understanding models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


class TextDatasetLoader:
    """
    Text dataset loader for benchmark testing.

    Loads text datasets from HuggingFace, applies tokenization,
    and provides TensorFlow Dataset interface for benchmarking.
    """

    def __init__(
        self,
        dataset_name: str = "glue",
        subset: Optional[str] = "sst2",
        split: str = "validation",
        tokenizer: Union[str, PreTrainedTokenizer] = "bert-base-uncased",
        num_samples: Optional[int] = None,
        max_length: int = 128,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize TextDatasetLoader.

        Args:
            dataset_name: Name of the dataset on HuggingFace
            subset: Dataset subset/task name (e.g., 'sst2' for GLUE)
            split: Dataset split to use ('train', 'validation', 'test')
            tokenizer: Tokenizer name or instance
            num_samples: Number of samples to load (None for all)
            max_length: Maximum sequence length
            cache_dir: Directory to cache dataset
            use_cache: Whether to use caching

        Raises:
            ValueError: If parameters are invalid
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.num_samples = num_samples
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache

        # Initialize tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.dataset = None
        self._stats: Dict = {}

        self._validate_params()

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")

    def load(self) -> "TextDatasetLoader":
        """
        Load the dataset from HuggingFace.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            print(
                f"Loading {self.dataset_name}"
                f"{f'/{self.subset}' if self.subset else ''} "
                f"dataset ({self.split} split)..."
            )

            # Load dataset from HuggingFace
            if self.subset:
                self.dataset = load_dataset(
                    self.dataset_name,
                    self.subset,
                    split=self.split,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    trust_remote_code=True,
                )
            else:
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
            "subset": self.subset,
            "split": self.split,
            "num_samples": len(self.dataset),
            "max_length": self.max_length,
            "features": list(self.dataset.features.keys()),
            "tokenizer": str(self.tokenizer.__class__.__name__),
        }

    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List, np.ndarray]]:
        """
        Tokenize text using the configured tokenizer.

        Args:
            text: Text or list of texts to tokenize
            max_length: Maximum sequence length (uses self.max_length if None)
            padding: Padding strategy ('max_length', 'longest', or False)
            truncation: Whether to truncate sequences
            return_tensors: Return tensors format ('tf', 'pt', 'np', or None)

        Returns:
            Dictionary containing input_ids, attention_mask, etc.
        """
        if max_length is None:
            max_length = self.max_length

        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

        return encoding

    def _get_text_field(self, sample: Dict) -> str:
        """
        Extract text from a sample (handles different dataset formats).

        Args:
            sample: Dataset sample

        Returns:
            Text string

        Raises:
            ValueError: If text field cannot be found
        """
        # Common text field names
        text_fields = ["sentence", "text", "content", "question", "passage"]

        for field in text_fields:
            if field in sample:
                return sample[field]

        # For paired text (e.g., sentence1 + sentence2 in some GLUE tasks)
        if "sentence1" in sample and "sentence2" in sample:
            # Concatenate with [SEP] token
            return f"{sample['sentence1']} {self.tokenizer.sep_token} {sample['sentence2']}"

        raise ValueError(
            f"Cannot find text field in sample. Available fields: {sample.keys()}"
        )

    def _tokenize_sample(
        self, sample: Dict, max_length: Optional[int] = None
    ) -> Dict:
        """
        Tokenize a single sample.

        Args:
            sample: Dataset sample
            max_length: Maximum sequence length

        Returns:
            Tokenized sample dictionary
        """
        text = self._get_text_field(sample)

        encoding = self.tokenize(
            text, max_length=max_length, padding="max_length", truncation=True
        )

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        # Include token_type_ids if available
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"]

        # Include label if available
        if "label" in sample:
            result["label"] = sample["label"]

        return result

    def get_tf_dataset(
        self,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        shuffle: bool = False,
        prefetch: bool = True,
    ) -> tf.data.Dataset:
        """
        Get TensorFlow Dataset for benchmarking.

        Args:
            batch_size: Batch size for the dataset
            max_length: Maximum sequence length (uses self.max_length if None)
            shuffle: Whether to shuffle the dataset
            prefetch: Whether to enable prefetching

        Returns:
            tf.data.Dataset ready for inference

        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        if max_length is None:
            max_length = self.max_length

        def generator():
            """Generator function for tf.data.Dataset."""
            for sample in self.dataset:
                tokenized = self._tokenize_sample(sample, max_length=max_length)

                # Prepare output
                inputs = {
                    "input_ids": np.array(tokenized["input_ids"], dtype=np.int32),
                    "attention_mask": np.array(
                        tokenized["attention_mask"], dtype=np.int32
                    ),
                }

                if "token_type_ids" in tokenized:
                    inputs["token_type_ids"] = np.array(
                        tokenized["token_type_ids"], dtype=np.int32
                    )

                if "label" in tokenized:
                    yield inputs, tokenized["label"]
                else:
                    yield inputs

        # Determine output signature
        output_signature = {
            "input_ids": tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
        }

        # Add token_type_ids if tokenizer supports it
        if hasattr(self.tokenizer, "token_type_ids") or "bert" in str(
            self.tokenizer.__class__.__name__
        ).lower():
            output_signature["token_type_ids"] = tf.TensorSpec(
                shape=(max_length,), dtype=tf.int32
            )

        # Add label if available
        if "label" in self.dataset.features:
            full_signature = (
                output_signature,
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )
        else:
            full_signature = (output_signature,)

        # Create TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_generator(
            generator, output_signature=full_signature
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
        self,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get preprocessed text as numpy arrays (for non-TF engines).

        Args:
            batch_size: Batch size
            max_length: Maximum sequence length (uses self.max_length if None)
            max_batches: Maximum number of batches to return (None for all)

        Returns:
            Dictionary containing:
                - input_ids: shape (N, max_length)
                - attention_mask: shape (N, max_length)
                - token_type_ids: shape (N, max_length) if available
                - labels: shape (N,) if available

        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        if max_length is None:
            max_length = self.max_length

        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_labels = []

        has_token_type_ids = False
        has_labels = "label" in self.dataset.features

        num_batches = 0
        for i in range(0, len(self.dataset), batch_size):
            if max_batches is not None and num_batches >= max_batches:
                break

            batch = self.dataset[i : i + batch_size]

            # Get texts
            texts = []
            for j in range(len(batch[list(batch.keys())[0]])):
                sample = {key: batch[key][j] for key in batch.keys()}
                texts.append(self._get_text_field(sample))

            # Tokenize batch
            encoding = self.tokenize(
                texts, max_length=max_length, padding="max_length", truncation=True
            )

            all_input_ids.append(encoding["input_ids"])
            all_attention_masks.append(encoding["attention_mask"])

            if "token_type_ids" in encoding:
                all_token_type_ids.append(encoding["token_type_ids"])
                has_token_type_ids = True

            if has_labels:
                all_labels.extend(batch["label"])

            num_batches += 1

        # Convert to numpy arrays
        result = {
            "input_ids": np.concatenate(all_input_ids, axis=0).astype(np.int32),
            "attention_mask": np.concatenate(all_attention_masks, axis=0).astype(
                np.int32
            ),
        }

        if has_token_type_ids:
            result["token_type_ids"] = np.concatenate(
                all_token_type_ids, axis=0
            ).astype(np.int32)

        if has_labels:
            result["labels"] = np.array(all_labels, dtype=np.int32)

        return result

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
        """String representation of TextDatasetLoader."""
        return (
            f"TextDatasetLoader(dataset='{self.dataset_name}"
            f"{f'/{self.subset}' if self.subset else ''}', "
            f"split='{self.split}', samples={len(self)}, "
            f"max_length={self.max_length})"
        )
