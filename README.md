# Obsidian RAG

Local retrieval over Markdown notes using Python, NumPy, and Ollama.

The repository includes a Markdown note loader, an Ollama embedding client
function, and five English notes in `example_notes/`. Similarity search and answer
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

## Read notes

`obsidian_rag.notes.load_notes` accepts a directory as a `pathlib.Path` and returns
notes with `title`, `content`, and `source` fields. It reads UTF-8 `.md` files
directly inside that directory in filename order.

The first line starting with `# ` becomes the title and is removed from the body.
If there is no such line, the filename without its extension becomes the title.
Other Markdown remains in the body, with leading and trailing whitespace removed.
Source references contain filenames rather than absolute paths. An empty
directory returns an empty list; filesystem and decoding errors are reported to
the caller.

Run the tests with:

```sh
uv run --locked pytest -q
```

## Generate embeddings

`obsidian_rag.embeddings.embed_texts` converts a list of texts into a NumPy matrix
with one vector per input, preserving order. Supply an Ollama client to select the
server and timeout. The default model is `qwen3-embedding:0.6b`.

```python
from ollama import Client
from obsidian_rag.embeddings import embed_texts

client = Client(host="http://127.0.0.1:11434", timeout=60, trust_env=False)
vectors = embed_texts(
    ["Capture a passing thought.", "Connect related ideas."],
    client=client,
)
print(vectors.shape)
```

Use the same embedding model for documents and questions. An empty input list
returns a `(0, 0)` matrix without contacting Ollama. Blank texts, inconsistent
vector shapes, non-finite values, and zero vectors raise `ValueError`. Automatic
truncation is disabled; oversized inputs and Ollama service errors are reported
to the caller.

The automated embedding tests mock the Ollama client and require no running
model. The example above makes a real request to the local Ollama service.

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
tests/              Automated tests
pyproject.toml      Project metadata and dependencies
uv.lock             Resolved dependency versions
```

The sample documents can be shared with the repository. Virtual environments,
caches, local `config.toml`, and local `.env` files are excluded from Git. Keep
machine-specific paths and credentials in ignored local files.
