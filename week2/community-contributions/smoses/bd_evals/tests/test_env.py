"""Tests for workspace .env loading."""

import os
from pathlib import Path

import pytest

from rag_evals.env import env_file_candidates, load_workspace_env
from rag_evals.workspace import Workspace


def make_workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    (root / "config").mkdir(parents=True)
    (root / "results").mkdir()
    return Workspace.from_path(root)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("BD_API_BASE_URL", "BD_API_KEY", "RAG_EVALS_TEST_VALUE"):
        monkeypatch.delenv(name, raising=False)


def test_candidates_are_workspace_then_parent(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    assert env_file_candidates(workspace) == [
        workspace.root / ".env",
        workspace.root.parent / ".env",
    ]


def test_no_env_file_loads_nothing(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    assert load_workspace_env(workspace) == []
    assert "BD_API_BASE_URL" not in os.environ


def test_workspace_env_file_is_loaded(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    (workspace.root / ".env").write_text(
        "# comment\nBD_API_BASE_URL=https://workspace.example\n", encoding="utf-8"
    )
    loaded = load_workspace_env(workspace)
    assert loaded == [workspace.root / ".env"]
    assert os.environ["BD_API_BASE_URL"] == "https://workspace.example"


def test_parent_env_file_is_loaded(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    (workspace.root.parent / ".env").write_text(
        "BD_API_BASE_URL=https://parent.example\n", encoding="utf-8"
    )
    loaded = load_workspace_env(workspace)
    assert loaded == [workspace.root.parent / ".env"]
    assert os.environ["BD_API_BASE_URL"] == "https://parent.example"


def test_workspace_env_file_wins_over_parent(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    (workspace.root / ".env").write_text(
        "BD_API_BASE_URL=https://workspace.example\n", encoding="utf-8"
    )
    (workspace.root.parent / ".env").write_text(
        "BD_API_BASE_URL=https://parent.example\nBD_API_KEY=parent-key\n", encoding="utf-8"
    )
    loaded = load_workspace_env(workspace)
    assert loaded == [workspace.root / ".env", workspace.root.parent / ".env"]
    assert os.environ["BD_API_BASE_URL"] == "https://workspace.example"
    assert os.environ["BD_API_KEY"] == "parent-key"


def test_existing_shell_environment_is_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    monkeypatch.setenv("BD_API_BASE_URL", "https://shell.example")
    (workspace.root / ".env").write_text("BD_API_BASE_URL=https://file.example\n", encoding="utf-8")
    load_workspace_env(workspace)
    assert os.environ["BD_API_BASE_URL"] == "https://shell.example"


def test_secret_values_are_not_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    workspace = make_workspace(tmp_path)
    (workspace.root / ".env").write_text("BD_API_KEY=super-secret-value\n", encoding="utf-8")
    with caplog.at_level("DEBUG", logger="rag_evals.env"):
        load_workspace_env(workspace)
    assert "super-secret-value" not in caplog.text
    assert "BD_API_KEY" in caplog.text
