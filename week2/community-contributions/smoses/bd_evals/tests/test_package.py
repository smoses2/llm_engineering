"""Tests for package scaffold: import and CLI --help."""

from typer.testing import CliRunner

import rag_evals
from rag_evals.cli import app

runner = CliRunner()


def test_version() -> None:
    assert rag_evals.__version__ == "0.1.0"


def test_import() -> None:
    assert rag_evals is not None
    assert hasattr(rag_evals, "__version__")


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Bedrock RAG evaluation framework" in result.stdout
