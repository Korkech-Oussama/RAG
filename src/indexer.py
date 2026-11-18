from models import MinimalSource
from __future__ import annotations
from typing import Iterator, Tuple, List, Self
from src.interfaces import BaseIndex


class LexicalIndex(BaseIndex):


    def __init__(self) -> None:
        ...

    def ingest(self, chunk: Iterator[Tuple[str, str, int, int]]) -> None:
        ...


    def search(self, query: str, k: int) -> List[MinimalSource]:
        ...


    def save(self, save_directory: str) -> None:
        ...


    def load(cls, directory_: str) -> Self:
        ...

