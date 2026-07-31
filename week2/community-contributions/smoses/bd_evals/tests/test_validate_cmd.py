"""Tests for the validate command."""

from pathlib import Path

from typer.testing import CliRunner

from rag_evals.cli import app
from rag_evals.config.validate_cmd import validate_definition

runner = CliRunner()

_EXAMPLE_WORKSPACE = Path(__file__).resolve().parents[2] / "bd_evals_config_example"
_EXAMPLE_DEFINITION = Path("definitions/retrieval-example.yaml")
_EXAMPLE_DEFINITION_PATH = _EXAMPLE_WORKSPACE / "config" / _EXAMPLE_DEFINITION


class TestValidateCommand:
    def test_validate_help(self) -> None:
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout

    def test_validate_missing_file(self) -> None:
        result = runner.invoke(
            app,
            ["--workspace", str(_EXAMPLE_WORKSPACE), "validate", "missing.yaml"],
        )
        assert result.exit_code == 3
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_validate_example_definition(self) -> None:
        result = runner.invoke(
            app,
            ["--workspace", str(_EXAMPLE_WORKSPACE), "validate", str(_EXAMPLE_DEFINITION)],
        )
        assert result.exit_code == 0, result.output
        assert "valid" in result.output.lower()

    def test_validate_json_format(self) -> None:
        import json

        result = runner.invoke(
            app,
            [
                "--workspace",
                str(_EXAMPLE_WORKSPACE),
                "validate",
                str(_EXAMPLE_DEFINITION),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["status"] == "valid"
        assert "definition_hash" in data
        assert len(data["inputs"]) >= 2


class TestValidateDefinition:
    def test_resolves_example(self) -> None:
        result = validate_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        assert result["definition_data"]["schema_version"] == 1
        assert result["definition_hash"]
        assert len(result["inputs"]) >= 2

    def test_questions_catalog_loaded(self) -> None:
        result = validate_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        questions = result["resolved_data"]["questions"]
        assert "questions" in questions
        assert questions["questions"][0]["id"] == "q001"

    def test_kb_catalog_loaded(self) -> None:
        result = validate_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        kbs = result["resolved_data"]["knowledge_bases"]
        assert len(kbs["knowledge_bases"]) == 3
        assert kbs["knowledge_bases"][0]["id"] == "kb_placeholder_1"

    def test_placeholder_warnings(self) -> None:
        result = validate_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        assert any("placeholder" in w.lower() for w in result["warnings"])
