"""Tests for CLI command registration and not-implemented behavior."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_evals.cli import app

runner = CliRunner()


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "config").mkdir(parents=True)
    (root / "results").mkdir()
    return root


ALL_COMMANDS = [
    "validate",
    "plan",
    "run-retrieval",
    "judge-retrieval",
    "generate-answers",
    "judge-answers",
    "report",
    "compare",
    "status",
    "resume",
    "seal",
    "clean-dataset",
]


def test_top_level_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Bedrock RAG evaluation framework" in result.stdout


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_command_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert command in result.stdout


@pytest.mark.parametrize(
    "command",
    [
        "judge-retrieval",
        "generate-answers",
        "judge-answers",
        "compare",
        "resume",
    ],
)
def test_definition_commands_are_functional_and_reject_invalid_input(
    command: str, tmp_path: Path
) -> None:
    workspace = _workspace(tmp_path)
    definition = workspace / "config" / "test.yaml"
    definition.write_text("schema_version: 1\n")
    result = runner.invoke(app, ["--workspace", str(workspace), command, "test.yaml"])
    assert result.exit_code != 0
    assert "not yet implemented" not in result.output


def test_validate_is_implemented(tmp_path: Path) -> None:
    """Validate is no longer a stub; it should run and report config errors."""
    workspace = _workspace(tmp_path)
    definition = workspace / "config" / "test.yaml"
    definition.write_text("schema_version: 1\n")
    result = runner.invoke(
        app,
        ["--workspace", str(workspace), "validate", "test.yaml"],
    )
    assert result.exit_code == 3
    assert "test" in result.output.lower()


def test_report_is_implemented() -> None:
    """Report command is implemented (may exit with error but not code 2)."""
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0


def test_status_is_implemented(tmp_path: Path) -> None:
    """Status command is implemented (exits with error for missing dataset)."""
    result = runner.invoke(
        app,
        ["--workspace", str(_workspace(tmp_path)), "status", "retrieval", "nonexistent"],
    )
    assert result.exit_code == 3


def test_seal_is_implemented(tmp_path: Path) -> None:
    """Seal command is implemented (exits with error for missing dataset)."""
    result = runner.invoke(
        app,
        ["--workspace", str(_workspace(tmp_path)), "seal", "retrieval", "nonexistent"],
    )
    assert result.exit_code == 3


def test_clean_dataset_is_implemented(tmp_path: Path) -> None:
    """Clean-dataset command is implemented (dry run succeeds)."""
    result = runner.invoke(
        app,
        [
            "--workspace",
            str(_workspace(tmp_path)),
            "clean-dataset",
            "retrieval",
            "nonexistent",
        ],
    )
    assert result.exit_code == 0


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0
