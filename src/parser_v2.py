from pathlib import Path
from typing import Tuple, Iterator, Dict, Optional
import hashlib
import re


class Corpus:

    def __init__(self, path_to_folder: str = 'data/raw/vllm-0.10.1', max_chunk_size: int = 2000):
        self.path_to_folder = Path(path_to_folder)
        self.max_chunk_size = max_chunk_size
        self.metadata: Dict[str, str] = {}

    def hash_full_file(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'rb') as f:
                digest = hashlib.file_digest(f, "sha256")
            return digest.hexdigest()
        except Exception as e:
            print(f"Error hashing {file_path}: {e}")
            return None

    def iterate_chunks(self) -> Iterator[Tuple[str, str, int, int]]:
        if not self.path_to_folder.exists():
            return

        for file_path in self.path_to_folder.rglob('*'):
            if file_path.is_dir() or file_path.suffix not in {'.py', '.md'}:
                continue

            path: str = str(file_path)
            f_hash = self.hash_full_file(path)

            # Skip if hashing failed or file content hasn't changed
            if f_hash is None or self.metadata.get(path) == f_hash:
                continue

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

            if file_path.suffix == '.py':
                yield from self._python_chunk(content, path)
            elif file_path.suffix == '.md':
                yield from self._md_chunk(content, path)

            # Only record metadata after the file was read and yielded cleanly
            self.metadata[path] = f_hash

    def _python_chunk(self, content: str, path: str) -> Iterator[Tuple[str, str, int, int]]:
        # Split before top-level def/async def/class declarations while keeping exact offsets
        blocks = re.split(r'(?=\n(?:def |async def |class ))', content)
        yield from self._assemble_chunks(blocks, path, separator='')

    def _md_chunk(self, content: str, path: str) -> Iterator[Tuple[str, str, int, int]]:
        # Split on blank lines, keeping exact character offsets intact
        blocks = content.split('\n\n')
        yield from self._assemble_chunks(blocks, path, separator='\n\n')

    def _assemble_chunks(
        self, blocks: list[str], file_path: str, separator: str
    ) -> Iterator[Tuple[str, str, int, int]]:
        """Assemble blocks into chunks without exceeding max_chunk_size."""
        current_chunk = ""
        current_start_idx = 0
        current_pointer = 0
        num_blocks = len(blocks)

        for idx, block in enumerate(blocks):
            sep_len = len(separator) if idx < num_blocks - 1 else 0

            # If a single block exceeds max_chunk_size, forcefully slice it
            if len(block) > self.max_chunk_size:
                if current_chunk:
                    yield (current_chunk, file_path, current_start_idx, current_start_idx + len(current_chunk))
                    current_chunk = ""

                for i in range(0, len(block), self.max_chunk_size):
                    slice_text = block[i:i + self.max_chunk_size]
                    yield (slice_text, file_path, current_pointer + i, current_pointer + i + len(slice_text))

                current_pointer += len(block) + sep_len
                current_start_idx = current_pointer
                continue

            projected_len = len(current_chunk) + len(separator) + len(block) if current_chunk else len(block)

            if projected_len > self.max_chunk_size:
                yield (current_chunk, file_path, current_start_idx, current_start_idx + len(current_chunk))
                current_chunk = block
                current_start_idx = current_pointer
            else:
                if current_chunk:
                    current_chunk += separator + block
                else:
                    current_chunk = block
                    current_start_idx = current_pointer

            current_pointer += len(block) + sep_len

        if current_chunk:
            yield (current_chunk, file_path, current_start_idx, current_start_idx + len(current_chunk))
