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

    def ingest(self, chunks: Iterator[Tuple[str, str, int, int]]) -> None:
        """Tokenize chunks, fit BM25 model, and index source locations."""
        corpus_texts: List[str] = []
        self.doc_metadata.clear()

        for chunk_text, file_path, start_idx, end_idx in chunks:
            corpus_texts.append(chunk_text)
            self.doc_metadata.append((file_path, start_idx, end_idx))

        if not corpus_texts:
            return

        # Simple white-space tokenization or stemmer tokenization via bm25s
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
        
        # Initialize and index
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Search query against index and return top-k MinimalSource models."""
        if self.retriever is None or not self.doc_metadata or k <= 0:
            return []

        query_tokens = bm25s.tokenize([query], stopwords="en")
        
        # bm25s handles top-k retrieval in vectorized C/NumPy execution
        results, scores = self.retriever.retrieve(query_tokens, k=min(k, len(self.doc_metadata)))

        sources: List[MinimalSource] = []
        # results[0] contains the indices of top documents for the first query
        for doc_idx in results[0]:
            file_path, start_idx, end_idx = self.doc_metadata[doc_idx]
            sources.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=start_idx,
                    last_character_index=end_idx
                )
            )

        return sources

    def save(self, save_directory: str) -> None:
        """Save BM25 index artifacts and metadata."""
        save_path = Path(save_directory) / "bm25s_index"
        if self.retriever:
            self.retriever.save(str(save_path), corpus=self.doc_metadata)

    @classmethod
    def load(cls, directory_path: str) -> Self:
        """Load BM25 index artifacts from disk."""
        load_path = Path(directory_path) / "bm25s_index"
        instance = cls()
        
        if load_path.exists():
            instance.retriever = bm25s.BM25.load(str(load_path), load_corpus=True)
            instance.doc_metadata = instance.retriever.corpus
            
        return instance