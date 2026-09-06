import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from httpx import ReadError, ReadTimeout
from ollama import ChatResponse, Client, EmbedResponse, Message, ResponseError

from obsidian_rag.cli import main


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "example_notes"
    directory.mkdir()
    (directory / "habits.md").write_text(
        "# Habit Stages\n\nA cue starts a habit.\n", encoding="utf-8"
    )
    (directory / "literature.md").write_text(
        "# Literature Notes\n\nPreserve the author's meaning.\n", encoding="utf-8"
    )
    (directory / "permanent.md").write_text(
        "# Permanent Notes\n\nDevelop one idea per note.\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock(spec=Client)
    client.__enter__.return_value = client
    client.embed.side_effect = [
        EmbedResponse(embeddings=[[0.0, 1.0], [3.0, 4.0], [1.0, 0.0]]),
        EmbedResponse(embeddings=[[1.0, 0.0]]),
    ]
    client.chat.return_value = ChatResponse(
        message=Message(
            role="assistant", content="Develop one idea per note. [permanent.md]"
        )
    )
    return client


@pytest.fixture(autouse=True)
def client_factory(client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> Mock:
    factory = Mock(return_value=client)
    monkeypatch.setattr("obsidian_rag.cli.Client", factory)
    return factory


def test_main_answers_using_the_most_relevant_notes(
    client: MagicMock, client_factory: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["How should I write permanent notes?"])

    output = capsys.readouterr()
    assert status == 0
    assert output.out == "Develop one idea per note. [permanent.md]\n"
    assert output.err == ""
    client_factory.assert_called_once_with(
        host="http://127.0.0.1:11434", timeout=180.0, trust_env=False
    )
    assert [call.kwargs for call in client.embed.call_args_list] == [
        {
            "model": "qwen3-embedding:0.6b",
            "input": [
                "Habit Stages\n\nA cue starts a habit.",
                "Literature Notes\n\nPreserve the author's meaning.",
                "Permanent Notes\n\nDevelop one idea per note.",
            ],
            "truncate": False,
        },
        {
            "model": "qwen3-embedding:0.6b",
            "input": [
                "Instruct: Given a question, retrieve relevant notes that help answer it.\n"
                "Query:How should I write permanent notes?"
            ],
            "truncate": False,
        },
    ]
    request = client.chat.call_args.kwargs
    assert request["model"] == "qwen3.5:4b"
    assert json.loads(request["messages"][1]["content"]) == {
        "question": "How should I write permanent notes?",
        "notes": [
            {
                "title": "Permanent Notes",
                "content": "Develop one idea per note.",
                "source": "permanent.md",
            },
            {
                "title": "Literature Notes",
                "content": "Preserve the author's meaning.",
                "source": "literature.md",
            },
        ],
    }


def test_main_accepts_directory_retrieval_model_and_connection_options(
    workspace: Path,
    client: MagicMock,
    client_factory: Mock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (workspace / "example_notes").rename(workspace / "other_notes")

    status = main([
        "How should I write permanent notes?",
        "--notes-dir", "other_notes",
        "--top-k", "1",
        "--embedding-model", "qwen3-embedding:4b",
        "--generation-model", "another-local-model",
        "--host", "http://127.0.0.1:11435",
        "--timeout", "15.5",
    ])

    assert status == 0
    assert capsys.readouterr().err == ""
    client_factory.assert_called_once_with(
        host="http://127.0.0.1:11435", timeout=15.5, trust_env=False
    )
    assert [call.kwargs["model"] for call in client.embed.call_args_list] == [
        "qwen3-embedding:4b", "qwen3-embedding:4b"
    ]
    request = client.chat.call_args.kwargs
    assert request["model"] == "another-local-model"
    assert [
        note["source"] for note in json.loads(request["messages"][1]["content"])["notes"]
    ] == ["permanent.md"]


def test_main_shows_help_without_connecting_to_ollama(
    client_factory: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr()
    assert "usage:" in output.out
    assert "--notes-dir" in output.out
    assert "--top-k" in output.out
    assert output.err == ""
    client_factory.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [""],
        [" \n\t"],
        ["Question?", "--top-k", "0"],
        ["Question?", "--top-k", "-1"],
        ["Question?", "--top-k", "1.5"],
        ["Question?", "--timeout", "0"],
        ["Question?", "--timeout", "-1"],
        ["Question?", "--timeout", "nan"],
        ["Question?", "--timeout", "inf"],
        ["Question?", "--unknown"],
    ],
    ids=[
        "missing-question", "empty-question", "whitespace-question",
        "zero-top-k", "negative-top-k", "fractional-top-k",
        "zero-timeout", "negative-timeout", "nan-timeout", "infinite-timeout",
        "unknown-option",
    ],
)
def test_main_rejects_invalid_arguments_before_connecting_to_ollama(
    arguments: list[str], client_factory: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)

    assert exit_info.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "error:" in output.err
    client_factory.assert_not_called()


@pytest.mark.parametrize("directory", ["missing", "plain.txt", "empty"])
def test_main_reports_unusable_note_directories_without_connecting_to_ollama(
    workspace: Path,
    directory: str,
    client_factory: Mock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (workspace / "plain.txt").write_text("Not a directory.", encoding="utf-8")
    (workspace / "empty").mkdir()

    status = main(["Question?", "--notes-dir", directory])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
    assert "Traceback" not in output.err
    client_factory.assert_not_called()


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        ("embed", ConnectionError("Ollama is unavailable.")),
        ("embed", ReadTimeout("Ollama request timed out.")),
        ("chat", ReadError("Ollama connection was interrupted.")),
        ("chat", ResponseError("Model not found.", status_code=404)),
    ],
    ids=["connection", "timeout", "read-error", "missing-model"],
)
def test_main_reports_service_errors_without_printing_an_answer(
    client: MagicMock,
    operation: str,
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    getattr(client, operation).side_effect = error

    status = main(["Question?"])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
    assert str(error) in output.err
    assert "Traceback" not in output.err


def test_main_reports_invalid_embeddings_before_generating_an_answer(
    client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    client.embed.side_effect = [EmbedResponse(embeddings=[[1.0, 0.0]])]

    status = main(["Question?"])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
    assert "one nonempty embedding vector per input text" in output.err
    client.chat.assert_not_called()


def test_main_reports_an_empty_model_answer(
    client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    client.chat.return_value = ChatResponse(
        message=Message(role="assistant", content=" \n")
    )

    status = main(["Question?"])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "empty answer" in output.err
