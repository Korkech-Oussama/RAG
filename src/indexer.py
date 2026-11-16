from models import MinimalSource
from __future__ import annotations
from typing import Iterator, Tuple, List, Self
from src.interfaces import BaseIndex


class LexicalIndex(BaseIndex):


    def __init__(self) -> None:
        ...

    def ingest(self, chunk: Iterator[Tuple[str, str, int, int]]) -> None:
        ...

