"""Lexical (BM25) index implementing the BaseIndex interface."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Type, TypeVar

import bm25s
from tqdm import tqdm

from src.interfaces import BaseIndex
from src.models import MinimalSource

T = TypeVar("T", bound="LexicalIndex")

# (index_text, file_path, first_character_index, last_character_index)
Chunk = Tuple[str, str, int, int]

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "from",
    "with", "and", "or", "is", "are", "be", "it", "this", "that", "as",
    "how", "what", "which", "does", "do", "can", "i", "you",
})

META_FILE = "metadata.json"
INDEX_DIR = "bm25s_index"


def tokenize(text: str) -> List[str]:
    """Identifier-aware tokenizer shared by indexing and querying.

    ``max_num_seqs`` -> ``max_num_seqs, max, num, seqs``
    ``LoRARequest``  -> ``lorarequest, lora, request``
    The full identifier is kept so verbatim identifiers in a question
    match exactly, and the parts let a paraphrase match too.
    """
    tokens: List[str] = []
    for word in _WORD_RE.findall(text):
        parts = [p.lower() for p in _PART_RE.findall(word)]
        if len(parts) > 1:
            tokens.append(word.lower())
        tokens.extend(parts)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


class LexicalIndex(BaseIndex):
    """BM25 index over chunks; stores only locations, never chunk text."""

    def __init__(self) -> None:
        """Create an empty index."""
        self.retriever: bm25s.BM25 | None = None
        self.doc_metadata: List[Tuple[str, int, int]] = []

    def ingest(
        self, chunks: Iterable[Chunk], show_progress: bool = True
    ) -> int:
        """Tokenize chunks, fit BM25 and remember each chunk's location.

        Returns the number of chunks indexed (0 leaves the index empty).
        """
        self.doc_metadata = []
        texts: List[str] = []

        # Pulling from `chunks` is what advances the "Chunking" bar.
        for text, file_path, start, end in chunks:
            texts.append(text)
            self.doc_metadata.append((file_path, start, end))

        corpus_tokens: List[List[str]] = []
        kept: List[Tuple[str, int, int]] = []
        for tokens, meta in zip(
            (tokenize(t) for t in tqdm(
                texts, desc="Tokenizing", unit="chunk",
                disable=not show_progress)),
            self.doc_metadata,
        ):
            if tokens:
                corpus_tokens.append(tokens)
                kept.append(meta)
        self.doc_metadata = kept

        if not corpus_tokens:
            self.retriever = None
            return 0

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=show_progress)
        self.retriever = retriever
        return len(self.doc_metadata)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the top-k sources, best first ([] if query or k is empty)."""
        if self.retriever is None or not self.doc_metadata or k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        doc_ids, scores = self.retriever.retrieve(
            [tokens], k=min(k, len(self.doc_metadata)), show_progress=False
        )
        sources: List[MinimalSource] = []
        for doc_id, score in zip(doc_ids[0], scores[0]):
            if score <= 0:  # no query term matched this chunk
                continue
            file_path, start, end = self.doc_metadata[int(doc_id)]
            sources.append(MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end,
                score=float(score),
            ))
        return sources

    def save(self, directory: str) -> None:
        """Persist the BM25 arrays and the chunk locations."""
        if self.retriever is None:
            raise RuntimeError("Nothing to save: the index is empty.")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(root / INDEX_DIR))
        with open(root / META_FILE, "w", encoding="utf-8") as f:
            json.dump(self.doc_metadata, f)

    @classmethod
    def load(cls: Type[T], directory: str) -> T:
        """Load a saved index; raises FileNotFoundError if none exists."""
        root = Path(directory)
        index_path = root / INDEX_DIR
        meta_path = root / META_FILE

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index found in '{directory}'. Run the index command."
            )
        instance = cls()
        instance.retriever = bm25s.BM25.load(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        instance.doc_metadata = [(p, int(s), int(e)) for p, s, e in raw]
        return instance