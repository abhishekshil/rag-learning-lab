# base_store.py
# Contract every vector store follows (Strategy Pattern).
# Implementations live in this package: PgVectorStore (+ Neo4j / MinIO in production_pipeline).

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VectorRecord:
    """
    One stored item: a vector plus everything we want to keep about it.

    Attributes:
        id:       Stable identifier for the chunk (e.g. "chunk-0").
        vector:   The embedding (1D float32 array).
        text:     The original chunk text (so we can show/return it).
        metadata: Free-form dict: source file, char offsets, chunk index, etc.
    """
    id: str
    vector: np.ndarray
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single hit from a search: the matched record and its score."""
    record: VectorRecord
    score: float

    @property
    def rank_text(self) -> str:
        return self.record.text


class BaseVectorStore(ABC):
    """Abstract base class for all vector stores."""

    @abstractmethod
    def add(self, records: list[VectorRecord]) -> None:
        """Insert records into the store."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        """
        Return the top-k records most similar to query_vector.

        Results are sorted best-first. `score` is higher = more similar,
        regardless of the underlying metric.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
