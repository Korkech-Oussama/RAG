from pathlib import Path
from typing import Tuple, Iterator, List, Dict
import hashlib

class Corpus:


	def __init__(self, path_to_folder: str = 'data/raw/vllm-0.10.1', max_chunk_size: int = 2000):
		self.path_to_folder = Path(path_to_folder)
		self.max_chunk_size = max_chunk_size
		self.metadata: Dict[str, str] = {}


	def hash_full_file(self, file_path: str) -> str:
	
		try:
			with open(file_path, 'rb', encoding='utf-8') as f:
				digest = hashlib.file_digest(f, "sha256")
		except Exception as e:
			print(e)
			return

		return digest.hexdigest()


	def iterate_chunks(self) -> Iterator[Tuple[str, str, int, int]]:

		if not self.path_to_folder.exists():
			return

		for file_path in self.path_to_folder.rglob('*'):
			
			if file_path.is_dir() or file_path.suffix not in {'.py', '.md'}:
				continue
			
			path: str = str(file_path)
			f_hash = self.hash_full_file(path)

			if self.metadata.get(path) == f_hash:
				continue


			if file_path.suffix == '.py':
				yield from self._python_chunk(path)	

			elif file_path.suffix == '.md':
				yield from self._md_chunk(path)

			self.metadata[path] = f_hash


	def _python_chunk(self, path: str) -> Iterator[Tuple[str, str, int, int]]:
		
		try:
			with open(path, 'r', 'utf-8') as f:
				content = f.read()
		except Exception as e:
			print(e)
			return

		blocks = content.split('\n')
		yield from self._assemble_chunks(blocks,path, sep='\n')


	def _md_chunk(self, path: str) -> Iterator[Tuple[str, str, int, int]]:

		try:
			with open(path, 'r', 'utf-8') as f:
				content = f.read()
		except Exception as e:
			print(e)
			return

		blocks = content.split('\n\n')
		yield from self._assemble_chunks(blocks, path, sep='\n\n')


	def _assemble_chunks(
        self, blocks: list[str], file_path: str, sep: str
    ) -> Iterator[Tuple[str, str, int, int]]:
		"""Assemble blocks into chunks without exceeding max_chunk_size.

		Args:
			blocks: The raw text blocks to assemble.
			file_path: The string path of the source file.
			separator: The string used to join blocks back together.

		Yields:
			Tuple[str, str, int, int]: The assembled chunk and its location metadata.
		"""
		current_chunk = ""
		current_start_idx = 0
		current_pointer = 0

		for block in blocks:
			# If a single block exceeds the max_chunk_size, forcefully slice it
			if len(block) > self.max_chunk_size:
				if current_chunk:
					yield (current_chunk, file_path, current_start_idx, current_start_idx + len(current_chunk))
					current_chunk = ""
				
				# Hard-slice the oversized block safely
				for i in range(0, len(block), self.max_chunk_size):
					slice_text = block[i:i + self.max_chunk_size]
					yield (slice_text, file_path, current_pointer + i, current_pointer + i + len(slice_text))
				
				current_pointer += len(block) + len(separator)
				current_start_idx = current_pointer
				continue

			# Calculate if adding this block exceeds the limit
			projected_len = len(current_chunk) + len(separator) + len(block) if current_chunk else len(block)
			
			if projected_len > self.max_chunk_size:
				# Yield the current chunk and reset
				yield (current_chunk, file_path, current_start_idx, current_start_idx + len(current_chunk))
				current_chunk = block
				current_start_idx = current_pointer
			else:
				# Append to current chunk
				if current_chunk:
					current_chunk += separator + block
				else:
					current_chunk = block
					current_start_idx = current_pointer

			# Move pointer forward by block length plus the hidden separator length
			current_pointer += len(block) + len(separator)

        # Yield any remaining text
		if current_chunk:
			yield (current_chunk, file_path, current_start_idx, current_start_idx + len(current_chunk))
