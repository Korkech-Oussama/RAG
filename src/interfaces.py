from abc import ABC, abstractmethod
from typing import Iterator, Tuple, List, Self
from src.models import MinimalSource

class BaseIndex(ABC):
 

    @abstractmethod
    def ingest(self, chunk: Iterator[Tuple[str, str, int, int]]) -> None:
        ...


    @abstractmethod
    def search(self, query: str, k: int) -> List[MinimalSource]:
        ...


    @abstractmethod
    def save(self, save_directory: str) -> None:
        ...


    @classmethod
    @abstractmethod
    def load(cls, directory_) -> Self:
        ...
