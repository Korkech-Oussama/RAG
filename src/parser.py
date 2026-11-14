from pathlib import Path
from typing import Tuple, Iterator


class Parser:

	def __init__(self, path_to_folder: str = 'data/raw/vllm-0.10.1', max_chunk_size: int = 2000):
		self.path_to_folder = Path(path_to_folder)

	def iterate_chunks(self) -> Iterator[Tuple[str, str, int, int]]:


		if not self.path_to_folder.exists():
			return

		for file_path in self.path_to_folder.rglob('*'):
			
			if file_path.is_dir():
				continue

			path: str = str(file_path)

			if file_path.suffix == '.py':
				yield from self._python_chunk(path)

			if file_path.suffix == '.md':
				yield from self._md_chunk(path)
	
	def _python_chunk(self, path: str) -> Iterator[Tuple[str, str, int, int]]:
		
		with open(path, )
		


	def _md_chunk(self, path: str) -> Iterator[Tuple[str, str, int, int]]:
		...




parser = Parser('')

parser.get_file_path()














