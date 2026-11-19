from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple, List, Self
import bm25s

from src.models import MinimalSource
from src.interfaces import BaseIndex


class LexicalIndex(BaseIndex):
    """Lexical search index backed by high-performance bm25s."""

    def __init__(self) -> None:
        self.retriever: bm25s.BM25 | None = None
        self.doc_metadata: List[Tuple[str, int, int]] = []

    def ingest(self, chunk: Iterator[Tuple[str, str, int, int]]) -> None:
        ...

