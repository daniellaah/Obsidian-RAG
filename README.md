# Obsidian RAG

Local retrieval over Markdown notes using Python, NumPy, and Ollama.

The repository includes a command-line interface, a Markdown note loader, an
Ollama embedding client function, cosine similarity search, answer generation,
and five English notes in `example_notes/`.

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

## Ask a question

After syncing the environment and starting Ollama with the models below
available, run this command from the repository root:

```sh
uv run --locked obsidian-rag "When should temporary notes be processed and deleted?"
```

The command reads `example_notes/`, embeds the notes and question, retrieves the
two most similar notes, and prints the generated answer. The embedding query
uses a Qwen retrieval instruction; answer generation receives the original
question. Each invocation reads and embeds the notes again, keeping vectors in
memory for that invocation.

Specify another note directory or result count with:

```sh
uv run --locked obsidian-rag "How do literature and permanent notes differ?" \
  --notes-dir example_notes --top-k 2
```

Relative note paths are resolved from the current working directory. The loader
reads Markdown files directly in that directory without visiting subdirectories.

| Option | Default | Purpose |
| --- | --- | --- |
| `--notes-dir` | `example_notes` | Directory containing Markdown notes |
| `--top-k` | `2` | Maximum number of notes used for the answer |
| `--embedding-model` | `qwen3-embedding:0.6b` | Qwen embedding model |
| `--generation-model` | `qwen3.5:4b` | Model used to generate the answer |
| `--host` | `http://127.0.0.1:11434` | Ollama server URL |
| `--timeout` | `180` | Request timeout in seconds |

`--top-k` must be a positive integer and `--timeout` a positive finite number.
Answers go to standard output; errors go to standard error. Exit codes are `0`
for success, `1` for a runtime failure, and `2` for invalid arguments. An empty
note directory reports an error without contacting Ollama.

Show all options with `uv run --locked obsidian-rag --help`. The same interface
is available through `uv run --locked python -m obsidian_rag.cli`.

CLI tests exercise the note loader, embedding conversion, retrieval, and answer
generation together while mocking only the external Ollama client. They do not
require a running model.

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

## Count tokens

`obsidian_rag.tokenization` provides local token counts for Ollama's
`qwen3-embedding:0.6b`. This is a standalone building block for chunking; the CLI
still embeds and retrieves whole notes.

```python
from obsidian_rag.tokenization import count_tokens, load_tokenizer

tokenizer = load_tokenizer()  # Download the tokenizer if needed, then reuse it.
title = "Permanent Notes"
body = "Develop one idea per note."

body_tokens = count_tokens(body, tokenizer=tokenizer)
input_tokens = count_tokens(
    f"{title}\n\n{body}", tokenizer=tokenizer, add_special_tokens=True
)
print(body_tokens, input_tokens)
```

The loader fetches only `tokenizer.json` (about 11.4 MB) from the public
[Qwen3-Embedding-0.6B repository](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
at revision `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418`. It uses the Hugging Face
Hub cache, or an explicit `cache_dir=Path(...)`. It does not download model
weights or require PyTorch, Transformers, or an Ollama connection. After caching
that snapshot, use `load_tokenizer(local_files_only=True)` to prevent all Hub
network requests. A missing offline snapshot or download failure is reported to
the caller; there is no fallback to character counting.

`count_tokens` excludes automatically added special tokens by default, including
when measuring chunk size or overlap. Empty text has a count of zero. For a full
nonempty embedding request, measure the assembled title and body together with
`add_special_tokens=True`; Ollama appends an end marker even when the text itself
ends with that marker. Concatenated text must be measured as a whole, since its
token count need not equal the sum of the individual counts. The helper does not
split text or enforce a context limit. Keep the returned tokenizer's padding and
truncation disabled.

The tokenizer library is pinned to `0.23.2`. The loader disables the official
file's NFC normalization to match the combining-character behavior verified
with Ollama `0.33.2`. This pairing is validated for `qwen3-embedding:0.6b`; do not
assume it measures arbitrary embedding models correctly. Recheck compatibility
when changing the model or tokenizer configuration.

Regular tokenization tests run offline using a tiny tokenizer in a temporary Hub
cache. They cover loading, counts, Unicode handling, end markers, and failures.
The optional integration tests use the actual cached snapshot and the local
Ollama model, including Chinese, English, Markdown, code, emoji, and token-length
boundaries. Prepare the tokenizer with `load_tokenizer()` and make sure Ollama is
running with `qwen3-embedding:0.6b`, then run:

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 .venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/integration/test_qwen_tokenizer.py
```

## Split notes into chunks

`obsidian_rag.chunking` exposes `chunk_notes` and `whole_note_chunks`. Both return
immutable `Chunk` objects with `content`, `title`, `source`, `chunk_index`,
`start_char`, and `end_char`. Positions are Python character offsets into the
loaded `Note.content`, with an exclusive end; the loader has already removed the
title line and outer whitespace, so these are not raw-file line numbers.

```python
from functools import partial
from pathlib import Path

from obsidian_rag.chunking import chunk_notes, whole_note_chunks
from obsidian_rag.notes import load_notes
from obsidian_rag.tokenization import count_tokens, load_tokenizer

notes = load_notes(Path("example_notes"))
tokenizer = load_tokenizer()
chunks = chunk_notes(notes, count_tokens=partial(count_tokens, tokenizer=tokenizer))
whole_chunks = whole_note_chunks(notes)  # B0: one whole note per chunk.
```

Recursive splitting defaults to a 512-token body budget and up to 64 overlapping
tokens. It prefers paragraphs, lines, sentence punctuation in Chinese/English,
then spaces, falling back to character boundaries for oversized units. Smaller
units are merged by recounting the combined text. Overlap retains whole trailing
units, so it may be below the target or zero. Titles and embedding end markers
are outside the body budget and need to be counted with the final input.

Short notes remain whole. Notes are processed independently, indices restart at
zero, and repeated passages keep their distinct positions. Text, whitespace, and
Markdown markers are preserved verbatim; this baseline does not interpret code
fences or table structure. Empty bodies retain their title and source in one
chunk. Invalid budgets and a fallback character that cannot fit raise `ValueError`.
All chunks remain in memory. CLI integration is a separate step.

The standard chunking tests are offline. To check the real Qwen tokenizer's
512/64 budgets and lossless reconstruction on long multilingual examples, cache
the tokenizer first and run (no Ollama service is required for this test file):

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 .venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/integration/test_qwen_chunking.py
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

## Retrieve notes

`obsidian_rag.retrieval.retrieve` ranks notes by cosine similarity and returns
`SearchResult` objects containing the original `chunk` and a numeric `score`.
Each row of the note matrix must correspond to the note at the same index. Pass
a single query vector with the same dimension, using the same embedding model
for notes and queries.

```python
from pathlib import Path

from ollama import Client

from obsidian_rag.embeddings import embed_texts
from obsidian_rag.notes import load_notes
from obsidian_rag.retrieval import retrieve

client = Client(host="http://127.0.0.1:11434", timeout=60, trust_env=False)
notes = load_notes(Path("example_notes"))
note_vectors = embed_texts(
    [f"{note.title}\n\n{note.content}" for note in notes],
    client=client,
)
question = "When should temporary notes be processed and deleted?"
query = (
    "Instruct: Given a question, retrieve relevant notes that help answer it.\n"
    f"Query:{question}"
)
query_vector = embed_texts([query], client=client)[0]

for result in retrieve(notes, note_vectors, query_vector, top_k=2):
    print(f"{result.score:.4f} {result.chunk.source}: {result.chunk.title}")
```

The query includes a task instruction in the format recommended by the
[Qwen3 Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Note text is embedded without that instruction. `embed_texts` passes its inputs
to the model unchanged, so the caller prepares the query text.

`top_k` defaults to 2 and must be a positive integer. Results are sorted by
score from highest to lowest; ties preserve input order. If fewer notes are
available, all are returned. An empty collection with a zero-row matrix returns
an empty list. Incompatible shapes, non-finite values, zero vectors, and
non-finite vector norms raise `ValueError`. Input vectors are left unchanged.

Retrieval tests use small, fixed vectors and require no model or network access.
The example above calls Ollama. A similarity score ranks relevance; it is not a
probability or proof that a note contains an answer. Answer generation must still
check whether the retrieved text supports a response.

## Generate an answer

`obsidian_rag.generation.generate_answer` takes the original question and the
`SearchResult` list returned by `retrieve`. It sends the question and retrieved chunk titles,
content, and source filenames to Ollama, then returns the answer as a string.
Retrieval order is preserved. The default generation model is `qwen3.5:4b`.

After running the retrieval example above, generate an answer with:

```python
from obsidian_rag.generation import generate_answer

results = retrieve(notes, note_vectors, query_vector, top_k=2)
answer = generate_answer(question, results, client=client)
print(answer)
```

Pass the original question rather than the embedding query with its task
instruction. Supply `model="your-local-model"` to select another generation
model. The supplied Ollama client controls the host and timeout. Requests use
non-streaming output, disable thinking, and set temperature to zero. Only answer
text is returned, with surrounding whitespace removed.

The prompt asks the model to use only the retrieved notes, cite claims with
filenames such as `[fleeting_notes.md]`, and explicitly acknowledge missing
information. Note text is treated as source material. These instructions do not
guarantee factual accuracy; the function does not independently verify generated
claims or citations.

An empty result list returns an insufficient-information message without
contacting Ollama. Blank questions and empty model responses raise `ValueError`.
Ollama and connection errors propagate to the caller.

Generation tests mock the external Ollama client and run without a model. They
verify the request and response contract. Check answer quality separately with
the local model, including questions whose answers are absent from the notes.

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

Model files are managed separately by Ollama; syncing the Python environment
does not download them.

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
