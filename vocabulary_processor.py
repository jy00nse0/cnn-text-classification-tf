import json
from typing import Iterable, List

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class _VocabularyWrapper:
    """A minimal wrapper to mimic the interface of TensorFlow's VocabularyProcessor."""

    def __init__(self, word_index):
        self._mapping = dict(word_index)

    def __len__(self) -> int:
        # +1 accounts for the reserved 0 index used for padding
        return len(self._mapping) + 1

    def __contains__(self, item: str) -> bool:
        return item in self._mapping


class VocabularyProcessor:
    """Lightweight replacement for tf.contrib.learn.preprocessing.VocabularyProcessor.

    The original VocabularyProcessor handled tokenisation, vocabulary building,
    sequence conversion and persistence. This reimplementation offers the
    subset of that behaviour relied upon by the project so it can run on
    TensorFlow 2.x where tf.contrib is unavailable.
    """

    def __init__(self, max_document_length: int, oov_token: str = "<OOV>") -> None:
        self.max_document_length = max_document_length
        self._tokenizer = Tokenizer(oov_token=oov_token)
        self.vocabulary_ = _VocabularyWrapper({})

    def _pad(self, sequences: List[List[int]]):
        return pad_sequences(
            sequences,
            maxlen=self.max_document_length,
            padding="post",
            truncating="post",
        )

    def fit_transform(self, raw_documents: Iterable[str]):
        texts = list(raw_documents)
        self._tokenizer.fit_on_texts(texts)
        sequences = self._tokenizer.texts_to_sequences(texts)
        self.vocabulary_ = _VocabularyWrapper(self._tokenizer.word_index)
        return self._pad(sequences)

    def transform(self, raw_documents: Iterable[str]):
        texts = list(raw_documents)
        sequences = self._tokenizer.texts_to_sequences(texts)
        return self._pad(sequences)

    def save(self, path: str) -> None:
        payload = {
            "max_document_length": self.max_document_length,
            "word_index": self._tokenizer.word_index,
            "oov_token": self._tokenizer.oov_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def restore(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        processor = cls(payload["max_document_length"], oov_token=payload.get("oov_token"))
        processor._tokenizer.word_index = {str(k): int(v) for k, v in payload["word_index"].items()}
        processor._tokenizer.index_word = {int(v): str(k) for k, v in payload["word_index"].items()}
        processor.vocabulary_ = _VocabularyWrapper(processor._tokenizer.word_index)
        return processor
