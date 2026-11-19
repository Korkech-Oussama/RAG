from src.indexer import LexicalIndex
from src.parser_v2 import Corpus
import time
from typing import Optional
from pydantic import ValidationError
from src.models import (
    MinimalSearchResults, RagDataset, StudentSearchResults,
)
import json
from tqdm import tqdm
from pathlib import Path
INDEX_DIR = "data/processed"

class RagCLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        start = time.perf_counter()
        corpus = Corpus("data/raw/vllm-0.10.1", max_chunk_size)
        idx = LexicalIndex()
        count = idx.ingest(corpus.iterate_chunks())
        idx.save("data/processed")
        print(f"Ingestion complete! Indexed {count} chunks under data/processed/ "
            f"in {time.perf_counter() - start:.1f}s")
        
    def _load_index(self) -> Optional[LexicalIndex]:
        """Load the saved index, or print a clean error and return None."""
        try:
            return LexicalIndex.load(INDEX_DIR)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: could not load the index: {e}")
        return None

    def search(self, query: str, k: int = 10) -> None:
        """Print the top-k sources for a single query."""
        query = str(query).strip()          # Fire may parse "123" as an int
        if not query:
            print("Error: the query is empty.")
            return
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            print("Error: --k must be a positive integer.")
            return

        idx = self._load_index()
        if idx is None:
            return

        results = idx.search(query, k)
        if not results:
            print("No results found.")
            return
        for r in results:
            print(f"{r.file_path} "
                  f"[{r.first_character_index}:{r.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        """Search every question of a dataset and save the results as JSON."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            print("Error: --k must be a positive integer.")
            return

        path = Path(dataset_path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except FileNotFoundError:
            print(f"Error: dataset file not found: {path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: {path} is not valid JSON: {e}")
            return
        except ValidationError as e:
            print(f"Error: {path} does not match the dataset format:\n{e}")
            return

        idx = self._load_index()           # load once, not per question
        if idx is None:
            return

        results = [
            MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=idx.search(q.question, k),
            )
            for q in tqdm(dataset.rag_questions,
                          desc="Searching", unit="question")
        ]

        out_file = Path(save_directory) / path.name
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            payload = StudentSearchResults(search_results=results, k=k)
            out_file.write_text(payload.model_dump_json(indent=2),
                                encoding="utf-8")
        except OSError as e:
            print(f"Error: could not write {out_file}: {e}")
            return
        print(f"Saved student_search_results to {out_file}")

    def answer(self, query: str, k: int = 10) -> None:
        query = str(query).strip()
        if not query:
            print("Error: the query is empty.")
            return
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            print("Error: --k must be a positive integer.")
            return
        idx = self._load_index()
        if idx is None:
            return
        sources = idx.search(query, k)
        if not sources:
            print("No relevant sources found.")
            return
        from src.generator import AnswerGenerator
        print(AnswerGenerator().generate(query, sources[:5]))
        print("\nSources:")
        for s in sources[:5]:
            print(f"  {s.file_path} "
                  f"[{s.first_character_index}:{s.last_character_index}]")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str, limit: int = 0) -> None:
        path = Path(student_search_results_path)
        try:
            data = StudentSearchResults.model_validate(
                json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error: cannot load {path}: {e}")
            return
        from src.generator import AnswerGenerator
        gen = AnswerGenerator()
        rows = data.search_results[:limit] if limit > 0 \
            else data.search_results
        answered = []
        for r in tqdm(rows, desc="Answering", unit="question"):
            try:
                text = gen.generate(r.question, r.retrieved_sources[:5])
            except Exception as e:            # one bad question must not
                tqdm.write(f"Warning: {r.question_id}: {e}")   # kill the run
                text = "Unable to generate an answer."
            answered.append(MinimalAnswer(
                question_id=r.question_id, question=r.question,
                retrieved_sources=r.retrieved_sources, answer=text))
        out = Path(save_directory) / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(StudentSearchResultsAndAnswer(
            search_results=answered, k=data.k).model_dump_json(indent=2),
            encoding="utf-8")
        print(f"Saved student_search_results_and_answer to {out}")
    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None: ...