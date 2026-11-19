• install: Install project dependencies using pip, uv, pipx, or any other package
manager of your choice.
• run: Execute the main script of your project (e.g., via Python interpreter).
• debug: Run the main script in debug mode using Python’s built-in debugger (e.g.,
pdb).
8
RAG against the machine Will you answer my questions?
• clean: Remove temporary files or caches (e.g., __pycache__, .mypy_cache) to
keep the project environment clean.
• lint: Execute the commands flake8 . and mypy . --warn-return-any
--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs
--check-untyped-defs
• lint-strict (optional): Execute the commands flake8 . and mypy . --strict



install:
	uv sync

run:
	uv run python3 -m src

debug:
	...


clean:
	...

• lint:
	flake8 .\
	mypy . --warn-return-any
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs
	--check-untyped-defs
	