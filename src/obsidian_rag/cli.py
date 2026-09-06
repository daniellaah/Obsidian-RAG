"""Answer questions about Markdown notes from the command line."""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from httpx import HTTPError
from ollama import Client, ResponseError

from obsidian_rag.embeddings import embed_texts
from obsidian_rag.generation import generate_answer
from obsidian_rag.notes import load_notes
from obsidian_rag.retrieval import retrieve


def main(argv: Sequence[str] | None = None) -> int:
    """Print an answer and return zero, or report a runtime error and return one.

    Parse the supplied arguments, or the process arguments when argv is None.
    Argument errors exit with status two; help exits with status zero.
    """
    parser = argparse.ArgumentParser(
        prog="obsidian-rag",
        description="Answer a question using local Markdown notes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", help="Question to answer from the notes.")
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=Path("example_notes"),
        help="Directory containing Markdown notes, relative to the working directory.",
    )
    parser.add_argument(
        "--top-k", type=int, default=2, help="Maximum number of notes to retrieve."
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding:0.6b",
        help="Ollama embedding model; queries use a Qwen retrieval instruction.",
    )
    parser.add_argument(
        "--generation-model",
        default="qwen3.5:4b",
        help="Ollama model for answer generation.",
    )
    parser.add_argument(
        "--host", default="http://127.0.0.1:11434", help="Ollama server URL."
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="Request timeout in seconds."
    )
    args = parser.parse_args(argv)

    if not args.question.strip():
        parser.error("question must not be blank")
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")

    try:
        notes = load_notes(args.notes_dir)
        if not notes:
            raise ValueError(f"No Markdown notes found in {args.notes_dir}.")

        with Client(host=args.host, timeout=args.timeout, trust_env=False) as client:
            note_vectors = embed_texts(
                [f"{note.title}\n\n{note.content}" for note in notes],
                client=client,
                model=args.embedding_model,
            )
            query = (
                "Instruct: Given a question, retrieve relevant notes that help answer it.\n"
                f"Query:{args.question}"
            )
            query_vector = embed_texts(
                [query], client=client, model=args.embedding_model
            )[0]
            results = retrieve(notes, note_vectors, query_vector, top_k=args.top_k)
            answer = generate_answer(
                args.question, results, client=client, model=args.generation_model
            )
    except (OSError, ValueError, ResponseError, HTTPError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
