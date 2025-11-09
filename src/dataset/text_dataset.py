"""
Lightweight Text Dataset Loader used for optional BERT benchmarks.

This implementation avoids HuggingFace dependencies by providing a tiny
in-memory dataset together with a simple tokenizer that performs
whitespace tokenization and maps tokens to integer IDs. The goal is to
support unit tests and small-scale demos without downloading large
datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class TextSample:
    """Container for a single text example."""

    text: str
    label: Optional[int] = None


class SimpleTokenizer:
    """
    Extremely small tokenizer that performs whitespace splitting.

    The resulting vocabulary is built dynamically when processing texts.
    The tokenizer reserves IDs for special tokens to mimic BERT inputs.
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    def __init__(self) -> None:
        self.vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().lower().split()

    def encode(self, text: str, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
        tokens = self._tokenize(text)
        token_ids: List[int] = []

        for token in tokens[:max_length]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            token_ids.append(self.vocab[token])

        attention_mask = [1] * len(token_ids)

        # Pad sequences
        while len(token_ids) < max_length:
            token_ids.append(self.vocab[self.PAD_TOKEN])
            attention_mask.append(0)

        token_ids_array = np.array(token_ids, dtype=np.int32)
        attention_mask_array = np.array(attention_mask, dtype=np.int32)

        return token_ids_array, attention_mask_array

    def batch_encode(
        self, texts: Sequence[str], max_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        encoded_ids = []
        encoded_masks = []

        for text in texts:
            token_ids, mask = self.encode(text, max_length=max_length)
            encoded_ids.append(token_ids)
            encoded_masks.append(mask)

        return np.stack(encoded_ids, axis=0), np.stack(encoded_masks, axis=0)


class TextDatasetLoader:
    """
    Minimal text dataset loader with built-in samples.

    This loader is intentionally lightweight and is not intended for
    large-scale NLP training. It primarily serves as a utility for unit
    tests and quick demonstrations when full HuggingFace integration is
    unnecessary.
    """

    DEFAULT_DATASET: List[TextSample] = [
        TextSample("this movie was fantastic and full of surprises", label=1),
        TextSample("i really enjoyed the performances and the story", label=1),
        TextSample("the plot was predictable and boring", label=0),
        TextSample("acting felt wooden and uninspired", label=0),
    ]

    def __init__(
        self,
        dataset_name: str = "builtin-sst2",
        subset: str = "validation",
        split: str = "validation",
        num_samples: Optional[int] = None,
        max_length: int = 128,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.num_samples = num_samples
        self.max_length = max_length

        self._tokenizer = SimpleTokenizer()
        self.dataset: List[TextSample] = []

    def load(self) -> "TextDatasetLoader":
        """Load a tiny in-memory dataset."""

        samples = self.DEFAULT_DATASET.copy()

        if self.num_samples is not None:
            if self.num_samples <= 0:
                raise ValueError("num_samples must be positive")
            samples = samples[: self.num_samples]

        self.dataset = samples
        return self

    def tokenize(
        self, texts: Union[str, Sequence[str]], max_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Tokenize text or batch of texts using the simple tokenizer.

        Args:
            texts: Single string or sequence of strings.
            max_length: Optional override for sequence length.

        Returns:
            Dictionary mimicking BERT inputs with `input_ids` and `attention_mask`.
        """

        target_length = max_length or self.max_length
        if target_length <= 0:
            raise ValueError("max_length must be positive")

        if isinstance(texts, str):
            input_ids, attention_mask = self._tokenizer.encode(
                texts, max_length=target_length
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        if not isinstance(texts, Iterable):
            raise TypeError("texts must be a string or an iterable of strings")

        encoded_ids, encoded_masks = self._tokenizer.batch_encode(
            texts, max_length=target_length
        )

        return {
            "input_ids": encoded_ids,
            "attention_mask": encoded_masks,
        }

    def get_stats(self) -> Dict[str, Union[str, int]]:
        """Return basic dataset statistics."""

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "num_samples": len(self.dataset),
            "max_length": self.max_length,
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return (
            f"TextDatasetLoader(dataset='{self.dataset_name}', subset='{self.subset}', "
            f"split='{self.split}', samples={len(self)}, max_length={self.max_length})"
        )

