# Obsidian RAG

Local retrieval over Markdown notes using Python, NumPy, and Ollama.

The repository currently contains the project environment, a Python package
skeleton, and five English notes in `example_notes/`. Retrieval and answer
generation are not implemented yet.

## Requirements

- Python 3.13
- uv
- Ollama, running locally for model operations

## Set up the Python environment

Run the following command from the repository root:

```sh
uv sync --locked
```

This installs the project, its runtime dependencies, and the development
dependencies into `.venv/`. The default development dependency is pytest.
Dependencies are declared in `pyproject.toml`; exact resolved versions are stored
in `uv.lock`. The uv cache is kept in `.uv-cache/`.

Use `uv run` to execute commands in the project environment without activating it
manually:

```sh
uv run --locked python --version
uv run --locked python -c "import obsidian_rag, numpy, ollama; print('Imports OK')"
uv run --locked pytest --version
```

## Local models

Ollama serves its local API at `http://127.0.0.1:11434` by default. Start Ollama if
it is not already running, then inspect the available models:

```sh
ollama list
```

| Role | Model |
| --- | --- |
| Text embeddings | `qwen3-embedding:0.6b` |
| Answer generation | `qwen3.5:4b` |

The generation model is planned for the answer-generation stage. Model files are
managed separately by Ollama; syncing the Python environment does not download
them.

## Repository layout

```text
example_notes/       Markdown documents
src/obsidian_rag/    Python package
pyproject.toml      Project metadata and dependencies
uv.lock             Resolved dependency versions
```

The sample documents can be shared with the repository. Virtual environments,
caches, local `config.toml`, and local `.env` files are excluded from Git. Keep
machine-specific paths and credentials in ignored local files.
