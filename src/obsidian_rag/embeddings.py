"""Generate text embeddings with Ollama."""

import numpy as np
from numpy.typing import NDArray
from ollama import Client


def embed_texts(
    texts: list[str],
    *,
    client: Client,
    model: str = "qwen3-embedding:0.6b",
) -> NDArray[np.float64]:
    """Return one vector per text, preserving input order.

    The caller supplies the client and controls its host and timeout. Empty
    input returns a (0, 0) matrix without calling Ollama. Nonempty input must
    contain no blank texts. Truncation is disabled so oversized inputs fail
    instead of losing content.

    Raise ValueError for invalid vector shapes, non-finite values, or zero
    vectors. Ollama and connection errors propagate to the caller.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float64)
    if any(not text.strip() for text in texts):
        raise ValueError("Input texts must not be blank.")

    response = client.embed(model=model, input=texts, truncate=False)
    vectors = np.asarray(response.embeddings, dtype=np.float64)

    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(texts)
        or vectors.shape[1] == 0
    ):
        raise ValueError("Expected one nonempty embedding vector per input text.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors must contain only finite values.")
    if np.any(np.all(vectors == 0, axis=1)):
        raise ValueError("Embedding vectors must not be zero vectors.")

    return vectors
