import numpy as np
import pytest
from numpy.typing import NDArray

from obsidian_rag.chunking import Chunk, chunk_notes, whole_note_chunks
from obsidian_rag.notes import Note
from obsidian_rag.retrieval import retrieve


@pytest.fixture
def notes() -> list[Chunk]:
    return whole_note_chunks([
        Note(title="Diagonal", content="A diagonal vector.", source="diagonal.md"),
        Note(title="Opposite", content="An opposite vector.", source="opposite.md"),
        Note(title="Aligned", content="An aligned vector.", source="aligned.md"),
        Note(title="Orthogonal", content="A perpendicular vector.", source="orthogonal.md"),
    ])


@pytest.fixture
def note_vectors() -> NDArray[np.float64]:
    return np.array([[3.0, 4.0], [-4.0, 0.0], [1.0, 0.0], [0.0, 10.0]])


def test_retrieve_ranks_notes_by_cosine_similarity(
    notes: list[Chunk], note_vectors: NDArray[np.float64]
) -> None:
    results = retrieve(notes, note_vectors, np.array([2.0, 0.0]), top_k=4)

    assert [result.chunk for result in results] == [
        notes[2], notes[0], notes[3], notes[1]
    ]
    np.testing.assert_allclose([result.score for result in results], [1.0, 0.6, 0.0, -1.0])


def test_retrieve_returns_two_notes_by_default(
    notes: list[Chunk], note_vectors: NDArray[np.float64]
) -> None:
    results = retrieve(notes, note_vectors, np.array([2.0, 0.0]))

    assert [result.chunk for result in results] == [notes[2], notes[0]]


@pytest.mark.parametrize(
    ("top_k", "expected_sources"),
    [
        (1, ["aligned.md"]),
        (8, ["aligned.md", "diagonal.md", "orthogonal.md", "opposite.md"]),
    ],
)
def test_retrieve_limits_results_to_the_requested_and_available_count(
    notes: list[Chunk],
    note_vectors: NDArray[np.float64],
    top_k: int,
    expected_sources: list[str],
) -> None:
    results = retrieve(notes, note_vectors, np.array([2.0, 0.0]), top_k=top_k)

    assert [result.chunk.source for result in results] == expected_sources


def test_retrieve_preserves_input_order_when_scores_are_equal(notes: list[Chunk]) -> None:
    vectors = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]])

    results = retrieve(notes, vectors, np.array([2.0, 0.0]), top_k=3)

    assert [result.chunk for result in results] == notes[:3]


def test_retrieve_does_not_modify_input_vectors(
    notes: list[Chunk], note_vectors: NDArray[np.float64]
) -> None:
    query_vector = np.array([2.0, 0.0])
    original_notes = note_vectors.copy()
    original_query = query_vector.copy()

    retrieve(notes, note_vectors, query_vector)

    np.testing.assert_array_equal(note_vectors, original_notes)
    np.testing.assert_array_equal(query_vector, original_query)


def test_retrieve_returns_no_results_for_an_empty_collection() -> None:
    assert retrieve([], np.empty((0, 0)), np.array([1.0, 0.0])) == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_retrieve_rejects_invalid_result_counts(
    notes: list[Chunk], note_vectors: NDArray[np.float64], top_k: int | float
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        retrieve(notes, note_vectors, np.array([2.0, 0.0]), top_k=top_k)


@pytest.mark.parametrize(
    ("vectors", "query"),
    [
        ([1.0, 0.0], [1.0, 0.0]),
        ([[1.0, 0.0]], [1.0, 0.0]),
        ([[1.0, 0.0]] * 4, [[1.0, 0.0]]),
        ([[1.0, 0.0]] * 4, [1.0, 0.0, 0.0]),
        ([[], [], [], []], []),
    ],
    ids=[
        "non-matrix-documents",
        "note-count-mismatch",
        "non-vector-query",
        "dimension-mismatch",
        "empty-vectors",
    ],
)
def test_retrieve_rejects_incompatible_vector_shapes(
    notes: list[Chunk], vectors: list, query: list
) -> None:
    with pytest.raises(ValueError):
        retrieve(notes, np.array(vectors), np.array(query))


@pytest.mark.parametrize("target", ["notes", "query"])
def test_retrieve_rejects_zero_vectors(
    notes: list[Chunk], note_vectors: NDArray[np.float64], target: str
) -> None:
    query_vector = np.array([2.0, 0.0])
    if target == "notes":
        note_vectors[0] = 0.0
    else:
        query_vector[:] = 0.0

    with pytest.raises(ValueError, match="nonzero"):
        retrieve(notes, note_vectors, query_vector)


@pytest.mark.parametrize(
    ("target", "value"), [("notes", float("nan")), ("query", float("inf"))]
)
def test_retrieve_rejects_non_finite_values(
    notes: list[Chunk],
    note_vectors: NDArray[np.float64],
    target: str,
    value: float,
) -> None:
    query_vector = np.array([2.0, 0.0])
    if target == "notes":
        note_vectors[0, 0] = value
    else:
        query_vector[0] = value

    with pytest.raises(ValueError, match="finite"):
        retrieve(notes, note_vectors, query_vector)


def test_retrieve_returns_distinct_chunks_from_the_same_note() -> None:
    chunks = chunk_notes(
        [Note(title="One note", content="abcdef", source="same.md")],
        count_tokens=len, chunk_size=3, chunk_overlap=0,
    )
    vectors = np.array([[0.0, 1.0], [1.0, 0.0]])

    results = retrieve(chunks, vectors, np.array([1.0, 0.0]), top_k=2)

    assert [(r.chunk.content, r.chunk.source, r.chunk.start_char) for r in results] == [
        ("def", "same.md", 3), ("abc", "same.md", 0),
    ]
    assert results[0].chunk is chunks[1]
