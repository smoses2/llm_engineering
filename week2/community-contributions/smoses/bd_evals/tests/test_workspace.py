"""Tests for evaluation workspace path resolution."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_evals.artifacts.store import create_dataset
from rag_evals.cli import app
from rag_evals.errors import ConfigError
from rag_evals.workspace import Workspace


def make_workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "config").mkdir(parents=True)
    (root / "results").mkdir()
    return root


def test_workspace_requires_config_and_results(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(ConfigError, match="config"):
        Workspace.from_path(root)
    (root / "config").mkdir()
    with pytest.raises(ConfigError, match="results"):
        Workspace.from_path(root)


def test_workspace_derived_paths(tmp_path: Path) -> None:
    workspace = Workspace.from_path(make_workspace(tmp_path))
    assert workspace.datasets_root == workspace.root / "results" / "datasets"
    assert workspace.reports_root == workspace.root / "results" / "reports"


def test_definition_resolves_relative_to_config(tmp_path: Path) -> None:
    workspace = Workspace.from_path(make_workspace(tmp_path))
    definition = workspace.config_root / "definitions" / "test.yaml"
    definition.parent.mkdir()
    definition.write_text("schema_version: 1\n", encoding="utf-8")
    assert workspace.resolve_definition(Path("definitions/test.yaml")) == definition


@pytest.mark.parametrize("argument", [Path("../outside.yaml"), Path("/tmp/outside.yaml")])
def test_definition_rejects_paths_outside_config(tmp_path: Path, argument: Path) -> None:
    workspace = Workspace.from_path(make_workspace(tmp_path))
    with pytest.raises(ConfigError):
        workspace.resolve_definition(argument)


def test_report_output_resolves_below_results(tmp_path: Path) -> None:
    workspace = Workspace.from_path(make_workspace(tmp_path))
    assert workspace.resolve_report_output(Path("latest")) == (
        workspace.root / "results" / "reports" / "latest"
    )
    with pytest.raises(ConfigError):
        workspace.resolve_report_output(Path("../outside"))


def test_help_exposes_workspace_without_requiring_it() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--workspace" in result.stdout
    assert "RAG_EVALS_WORKSPACE" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["validate", "definition.yaml"],
        ["plan", "definition.yaml"],
        ["run-retrieval", "definition.yaml"],
        ["judge-retrieval", "definition.yaml"],
        ["generate-answers", "definition.yaml"],
        ["judge-answers", "definition.yaml"],
        ["report", "definition.yaml"],
        ["compare", "definition.yaml"],
        ["status", "retrieval", "missing"],
        ["resume", "definition.yaml"],
        ["seal", "retrieval", "missing"],
        ["clean-dataset", "retrieval", "missing"],
    ],
)
def test_commands_fail_closed_without_workspace(arguments: list[str]) -> None:
    result = CliRunner().invoke(
        app,
        arguments,
        env={"RAG_EVALS_WORKSPACE": ""},
    )
    assert result.exit_code == ConfigError.exit_code
    assert "--workspace" in result.output
    assert "RAG_EVALS_WORKSPACE" in result.output


def test_environment_selects_workspace(tmp_path: Path) -> None:
    root = make_workspace(tmp_path)
    create_dataset(root / "results", "retrieval", "dataset", {}, [])
    result = CliRunner().invoke(
        app,
        ["status", "retrieval", "dataset"],
        env={"RAG_EVALS_WORKSPACE": str(root)},
    )
    assert result.exit_code == 0, result.output
    assert "retrieval/dataset" in result.output


def test_explicit_workspace_overrides_environment(tmp_path: Path) -> None:
    environment_root = make_workspace(tmp_path / "environment")
    explicit_root = make_workspace(tmp_path / "explicit")
    create_dataset(environment_root / "results", "retrieval", "dataset", {}, [])
    result = CliRunner().invoke(
        app,
        [
            "--workspace",
            str(explicit_root),
            "status",
            "retrieval",
            "dataset",
        ],
        env={"RAG_EVALS_WORKSPACE": str(environment_root)},
    )
    assert result.exit_code == ConfigError.exit_code
    assert "Dataset not found" in result.output


def test_definition_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = Workspace.from_path(make_workspace(tmp_path))
    outside = tmp_path / "outside.yaml"
    outside.write_text("schema_version: 1\n", encoding="utf-8")
    link = workspace.config_root / "linked.yaml"
    link.symlink_to(outside)
    with pytest.raises(ConfigError, match="escapes workspace boundary"):
        workspace.resolve_definition(Path("linked.yaml"))


def test_two_workspaces_keep_same_dataset_id_isolated(tmp_path: Path) -> None:
    first = Workspace.from_path(make_workspace(tmp_path / "first"))
    second = Workspace.from_path(make_workspace(tmp_path / "second"))
    paths = create_dataset(first.results_root, "retrieval", "same-id", {}, [])
    assert paths.base.exists()
    assert not (second.datasets_root / "retrieval" / "same-id").exists()
