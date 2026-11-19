"""Corpus chunking: turns .py and .md files into offset-exact chunks.

A chunk is a span (start, end) over the original file content, with end
exclusive, so the raw text is always ``content[start:end]``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

# (index_text, file_path, first_character_index, last_character_index)
Chunk = Tuple[str, str, int, int]
# (start, end, label): a contiguous span plus a name (heading / def / class)
Unit = Tuple[int, int, str]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
EXTENSIONS = {".py", ".md"}


class Corpus:
    """Walks a folder and yields chunks of its .py and .md files."""

    def __init__(
        self,
        path_to_folder: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
    ) -> None:
        """Store the corpus root and the maximum chunk size in characters."""
        self.path_to_folder = Path(path_to_folder)
        self.max_chunk_size = max_chunk_size

    # ------------------------------------------------------------------
    # File level
    # ------------------------------------------------------------------
    def _discover_files(self) -> List[Path]:
        """Return the sorted list of indexable files (sorted = reproducible)."""
        if not self.path_to_folder.exists():
            return []
        return sorted(
            p for p in self.path_to_folder.rglob("*")
            if p.is_file() and p.suffix in EXTENSIONS
        )

    def _read(self, path: Path) -> Optional[str]:
        """Read a file keeping line endings untouched; None on failure."""
        try:
            with open(path, "r", encoding="utf-8",
                      errors="replace", newline="") as f:
                return f.read()
        except OSError as e:
            tqdm.write(f"Warning: cannot read {path}: {e}")
            return None

    def iterate_chunks(self) -> Iterator[Chunk]:
        """Yield (index_text, file_path, start, end) for every chunk."""
        files = self._discover_files()
        for path in tqdm(files, desc="Chunking", unit="file"):
            content = self._read(path)
            if content is None or not content.strip():
                continue

            if path.suffix == ".py":
                units = self._python_units(content)
            else:
                units = self._markdown_units(content)

            for start, end, label in self._pack(units):
                raw = content[start:end]
                if not raw.strip():
                    continue
                yield (
                    self._index_text(str(path), label, raw),
                    str(path), start, end,
                )

    @staticmethod
    def _index_text(path: str, label: str, raw: str) -> str:
        """Text given to BM25: path and label make chunks easier to match."""
        header = f"{path}\n{label}\n" if label else f"{path}\n"
        return header + raw

    # ------------------------------------------------------------------
    # Span level helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _line_starts(content: str) -> List[int]:
        """Offset of the first character of every line ('\\n' only)."""
        starts = [0]
        for m in re.finditer("\n", content):
            starts.append(m.end())
        return starts

    def _hard_split(
        self, content: str, start: int, end: int
    ) -> List[Tuple[int, int]]:
        """Cut an oversized span into pieces of at most max_chunk_size."""
        limit = self.max_chunk_size
        pieces: List[Tuple[int, int]] = []
        while end - start > limit:
            window_end = start + limit
            cut = content.rfind("\n", start, window_end)
            if cut <= start:
                cut = content.rfind(" ", start, window_end)
            stop = cut + 1 if cut > start else window_end
            pieces.append((start, stop))
            start = stop
        pieces.append((start, end))
        return pieces

    def _fit_content(
        self, content: str, units: Sequence[Unit]
    ) -> List[Unit]:
        """Hard split every oversized unit, keep the others untouched."""
        fitted: List[Unit] = []
        for start, end, label in units:
            if end - start <= self.max_chunk_size:
                fitted.append((start, end, label))
            else:
                for s, e in self._hard_split(content, start, end):
                    fitted.append((s, e, label))
        return fitted

    def _pack(self, units: Sequence[Unit]) -> List[Unit]:
        """Greedily merge contiguous units while they fit in one chunk."""
        if not units:
            return []
        packed: List[Unit] = []
        cur_start, cur_end, first = units[0]
        labels: List[str] = [first] if first else []
        for start, end, label in units[1:]:
            if end - cur_start <= self.max_chunk_size:
                cur_end = end
                if label and label not in labels:
                    labels.append(label)
            else:
                packed.append((cur_start, cur_end, " ".join(labels)))
                cur_start, cur_end = start, end
                labels = [label] if label else []
        packed.append((cur_start, cur_end, " ".join(labels)))
        return packed

    def _blank_blocks(
        self, content: str, start: int, end: int, label: str, fenced: bool
    ) -> List[Unit]:
        """Split a span at blank lines, never inside a fenced code block."""
        units: List[Unit] = []
        block_start = start
        pos = start
        in_fence = False
        prev_blank = False
        while pos < end:
            nl = content.find("\n", pos, end)
            line_end = end if nl == -1 else nl + 1
            line = content[pos:line_end]
            was_in_fence = in_fence
            if fenced and FENCE_RE.match(line):
                in_fence = not in_fence
            blank = not line.strip()
            if prev_blank and not blank and not was_in_fence:
                units.append((block_start, pos, label))
                block_start = pos
            prev_blank = blank and not in_fence
            pos = line_end
        units.append((block_start, end, label))
        return units

    # ------------------------------------------------------------------
    # Markdown strategy
    # ------------------------------------------------------------------
    def _markdown_units(self, content: str) -> List[Unit]:
        """Cut at headings (outside code fences), then at blank lines."""
        starts = self._line_starts(content)
        sections: List[Tuple[int, str]] = []
        stack: List[Tuple[int, str]] = []
        in_fence = False

        for i, line_start in enumerate(starts):
            line_end = starts[i + 1] if i + 1 < len(starts) else len(content)
            line = content[line_start:line_end]
            if FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = HEADING_RE.match(line.rstrip("\r\n"))
            if m:
                level = len(m.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, m.group(2)))
                sections.append(
                    (line_start, " > ".join(t for _, t in stack))
                )

        if not sections or sections[0][0] != 0:
            sections.insert(0, (0, ""))

        units: List[Unit] = []
        for j, (s, label) in enumerate(sections):
            e = sections[j + 1][0] if j + 1 < len(sections) else len(content)
            if e <= s:
                continue
            if e - s > self.max_chunk_size:
                units.extend(self._blank_blocks(content, s, e, label, True))
            else:
                units.append((s, e, label))
        return self._fit_content(content, units)

    # ------------------------------------------------------------------
    # Python strategy
    # ------------------------------------------------------------------
    def _python_units(self, content: str) -> List[Unit]:
        """Tile the file by top-level statements, recursing when too big."""
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError, RecursionError):
            blocks = self._blank_blocks(content, 0, len(content), "", False)
            return self._fit_content(content, blocks)
        starts = self._line_starts(content)
        units = self._tile(content, starts, tree.body, 0, len(content), "")
        return self._fit_content(content, units)

    def _tile(
        self,
        content: str,
        starts: List[int],
        nodes: Sequence[ast.stmt],
        start: int,
        end: int,
        prefix: str,
    ) -> List[Unit]:
        """Turn consecutive statements into contiguous units over a span.

        Each unit begins where the previous one ended, so decorators,
        comments and blank lines attach to the code that follows them.
        """
        units: List[Unit] = []
        cursor = start
        for idx, node in enumerate(nodes):
            if idx == len(nodes) - 1:
                unit_end = end
            else:
                last_line = node.end_lineno or node.lineno
                unit_end = starts[last_line] if last_line < len(starts) \
                    else len(content)
                unit_end = min(unit_end, end)
            if unit_end <= cursor:
                continue

            is_def = isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            if is_def:
                name = getattr(node, "name", "")
                label = f"{prefix}.{name}" if prefix else name
            else:
                label = prefix

            if unit_end - cursor > self.max_chunk_size and is_def:
                body = getattr(node, "body", [])
                units.extend(
                    self._tile(content, starts, body, cursor, unit_end, label)
                )
            else:
                units.append((cursor, unit_end, label))
            cursor = unit_end

        if cursor < end:
            units.append((cursor, end, prefix))
        return units