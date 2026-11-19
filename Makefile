GOINFRE_DIR := /goinfre/$(USER)

# Route all uv caches, temp files, and the .venv to goinfre
export UV_CACHE_DIR := $(GOINFRE_DIR)/.uv_cache
export UV_PROJECT_ENVIRONMENT := $(GOINFRE_DIR)/.venv
export TMPDIR := $(GOINFRE_DIR)/.tmp

# Route Hugging Face model and dataset downloads to goinfre
export HF_HOME := $(GOINFRE_DIR)/.huggingface_cache


.PHONY: install run debug clean lint lint-strict


install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf src/pycache data/output .mypy_cache llm_sdk/llm_sdk/pycache +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	uv run mypy src --follow-imports=silent \
        --warn-unused-ignores \
        --ignore-missing-imports \
        --disallow-untyped-defs \
        --check-untyped-defs \
        --warn-return-any
	uv run flake8 src