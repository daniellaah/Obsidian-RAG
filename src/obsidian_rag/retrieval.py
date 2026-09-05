"""Rank Markdown notes by cosine similarity."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from obsidian_rag.notes import Note


@dataclass(frozen=True)
class SearchResult:
    """A note and its cosine similarity to a query."""

    note: Note
    score: float


def retrieve(
    notes: list[Note],
    note_vectors: NDArray[np.float64],
    query_vector: NDArray[np.float64],
    *,
    top_k: int = 2,
) -> list[SearchResult]:
    """Return up to top_k notes ordered by descending cosine similarity.

    Each row of note_vectors must correspond to the note at the same index.
    The query must be one vector with the same dimension as the note vectors.
    Equal scores preserve input order, and input vectors are left unchanged.
    An empty collection with a zero-row matrix returns no results.

    Raise ValueError for an invalid result count, incompatible vector shapes,
    non-finite values, or vector norms that are zero or non-finite.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    matrix = np.asarray(note_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(notes):
        raise ValueError("Expected a matrix with one vector per note.")
    if not notes:
        return []

    query = np.asarray(query_vector, dtype=np.float64)
    if query.ndim != 1 or query.size == 0 or matrix.shape[1] != query.size:
        raise ValueError("Expected a nonempty query vector matching the note dimension.")
    if not np.isfinite(matrix).all() or not np.isfinite(query).all():
        raise ValueError("Vectors must contain only finite values.")

    with np.errstate(over="ignore", under="ignore"):
        note_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
    if not np.isfinite(note_norms).all() or not np.isfinite(query_norm):
        raise ValueError("Vector norms must be finite.")
    if np.any(note_norms == 0) or query_norm == 0:
        raise ValueError("Vectors must have nonzero norms.")

    normalized_notes = matrix / note_norms[:, None]
    normalized_query = query / query_norm
    scores = np.clip(normalized_notes @ normalized_query, -1.0, 1.0)
    indices = np.argsort(-scores, kind="stable")[:top_k]

    return [
        SearchResult(note=notes[index], score=float(scores[index]))
        for index in indices
    ]
