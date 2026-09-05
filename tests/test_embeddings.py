from unittest.mock import Mock

import numpy as np
import pytest
from ollama import Client, EmbedResponse, ResponseError

from obsidian_rag.embeddings import embed_texts


@pytest.fixture
def client() -> Mock:
    return Mock(spec=Client)


def test_embed_texts_returns_vectors_in_input_order(client: Mock) -> None:
    texts = ["Capture a passing thought.", "Connect related ideas."]
    client.embed.return_value = EmbedResponse(
        embeddings=[[0.6, 0.8], [-0.8, 0.6]]
    )

    vectors = embed_texts(texts, client=client)

    assert isinstance(vectors, np.ndarray)
    assert np.issubdtype(vectors.dtype, np.floating)
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [-0.8, 0.6]])
    client.embed.assert_called_once_with(
        model="qwen3-embedding:0.6b", input=texts, truncate=False
    )


def test_embed_texts_supports_a_selected_model_and_its_vector_dimension(
    client: Mock,
) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.0, 0.6, 0.8]])

    vectors = embed_texts(
        ["Preserve the source."], client=client, model="qwen3-embedding:4b"
    )

    assert vectors.shape == (1, 3)
    assert client.embed.call_args.kwargs["model"] == "qwen3-embedding:4b"


def test_embed_texts_returns_an_empty_matrix_without_calling_ollama(
    client: Mock,
) -> None:
    vectors = embed_texts([], client=client)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (0, 0)
    client.embed.assert_not_called()


@pytest.mark.parametrize("blank_text", ["", " \n\t"])
def test_embed_texts_rejects_blank_text_before_calling_ollama(
    client: Mock, blank_text: str
) -> None:
    with pytest.raises(ValueError, match="blank"):
        embed_texts(["A useful idea.", blank_text], client=client)

    client.embed.assert_not_called()


@pytest.mark.parametrize(
    "embeddings",
    [
        [[0.6, 0.8]],
        [[0.6, 0.8], [1.0]],
        [[], []],
    ],
    ids=["missing-vector", "inconsistent-dimensions", "empty-vectors"],
)
def test_embed_texts_rejects_invalid_matrix_shapes(
    client: Mock, embeddings: list[list[float]]
) -> None:
    client.embed.return_value = EmbedResponse(embeddings=embeddings)

    with pytest.raises(ValueError):
        embed_texts(["First idea.", "Second idea."], client=client)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_embed_texts_rejects_non_finite_values(
    client: Mock, invalid_value: float
) -> None:
    client.embed.return_value = EmbedResponse(
        embeddings=[[invalid_value, 0.8]]
    )

    with pytest.raises(ValueError, match="finite"):
        embed_texts(["A useful idea."], client=client)


def test_embed_texts_rejects_zero_vectors(client: Mock) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.0, 0.0]])

    with pytest.raises(ValueError, match="zero"):
        embed_texts(["A useful idea."], client=client)


def test_embed_texts_propagates_model_errors(client: Mock) -> None:
    client.embed.side_effect = ResponseError("Model not found.", status_code=404)

    with pytest.raises(ResponseError) as error:
        embed_texts(["A useful idea."], client=client)

    assert error.value.status_code == 404


def test_embed_texts_propagates_connection_errors(client: Mock) -> None:
    client.embed.side_effect = ConnectionError("Ollama is unavailable.")

    with pytest.raises(ConnectionError, match="Ollama is unavailable"):
        embed_texts(["A useful idea."], client=client)
