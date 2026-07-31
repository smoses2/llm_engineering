"""Tests for the plan command."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rag_evals.artifacts.store import (
    Lifecycle,
    create_dataset,
    transition_lifecycle,
    write_artifact,
)
from rag_evals.cli import app
from rag_evals.errors import PlanError
from rag_evals.planning.plan_cmd import plan_definition

runner = CliRunner()

_EXAMPLE_WORKSPACE = Path(__file__).resolve().parents[2] / "bd_evals_config_example"
_EXAMPLE_DEFINITION = Path("definitions/retrieval-example.yaml")
_EXAMPLE_DEFINITION_PATH = _EXAMPLE_WORKSPACE / "config" / _EXAMPLE_DEFINITION

JUDGE_MODELS = {
    "schema_version": 1,
    "models": [{"id": "judge_a", "model_id": "real.model.v1", "description": "judge"}],
}

JUDGMENT_DEFINITION = {
    "schema_version": 1,
    "test": {"id": "judgment-plan", "type": "retrieval_judgment"},
    "source_dataset": {"type": "retrieval", "id": "src-ds"},
    "judge_models": {"source": "../judge-models.yaml"},
    "selection": {"judge_models": {"include": ["judge_a"]}},
    "prompt_variants": [
        {
            "id": "v1",
            "template": {
                "system_instructions": "prompts/system.md",
                "user_message": "prompts/user.md",
            },
        }
    ],
    "inference_variants": [{"id": "det", "inference_config": {"maxTokens": 512, "temperature": 0}}],
    "api": {"base_url_env": "BD_API_BASE_URL", "api_key_env": "BD_API_KEY"},
    "output": {"dataset_id": "judgment-plan-001"},
}


@pytest.fixture
def ask_workspace(tmp_path: Path) -> Path:
    """A workspace with a complete two-artifact retrieval dataset and a judgment definition."""
    config = tmp_path / "config"
    (config / "definitions").mkdir(parents=True)
    (config / "prompts").mkdir()
    results = tmp_path / "results"
    results.mkdir()

    (config / "judge-models.yaml").write_text(yaml.safe_dump(JUDGE_MODELS), encoding="utf-8")
    (config / "prompts" / "system.md").write_text("Judge the context.", encoding="utf-8")
    (config / "prompts" / "user.md").write_text(
        "Q: {{ question }}\nC: {{ retrieval_context }}", encoding="utf-8"
    )
    (config / "definitions" / "judgment.yaml").write_text(
        yaml.safe_dump(JUDGMENT_DEFINITION), encoding="utf-8"
    )

    ids = [f"sha256:{i:064d}" for i in range(2)]
    paths = create_dataset(
        results, "retrieval", "src-ds", {"test": "data"}, [{"artifact_id": i} for i in ids]
    )
    for index, artifact_id in enumerate(ids):
        write_artifact(
            paths,
            artifact_id,
            {
                "artifact_id": artifact_id,
                "question_id": f"q{index + 1:03d}",
                "question_text": f"Question {index + 1}?",
                "prepared_context": "context",
            },
        )
    transition_lifecycle(paths, Lifecycle.COMPLETE)
    return tmp_path


class TestPlanCommand:
    def test_plan_help(self) -> None:
        result = runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0

    def test_plan_retrieval_example(self) -> None:
        result = runner.invoke(
            app,
            ["--workspace", str(_EXAMPLE_WORKSPACE), "plan", str(_EXAMPLE_DEFINITION)],
        )
        assert result.exit_code == 0, result.output
        assert "Total calls" in result.output
        assert "--approve-call-count" in result.output

    def test_plan_json_format(self) -> None:
        result = runner.invoke(
            app,
            [
                "--workspace",
                str(_EXAMPLE_WORKSPACE),
                "plan",
                str(_EXAMPLE_DEFINITION),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["definition_type"] == "retrieval"
        assert "total_calls" in data
        assert "rows" in data
        assert len(data["rows"]) == data["total_calls"]

    def test_plan_missing_file(self) -> None:
        result = runner.invoke(
            app,
            ["--workspace", str(_EXAMPLE_WORKSPACE), "plan", "missing.yaml"],
        )
        assert result.exit_code == 3


class TestPlanDefinition:
    def test_retrieval_plan(self) -> None:
        result = plan_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        assert result["definition_type"] == "retrieval"
        assert result["total_calls"] > 0
        assert all("artifact_id" in row for row in result["rows"])

    def test_deterministic_ids(self) -> None:
        r1 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        r2 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        assert [r["artifact_id"] for r in r1["rows"]] == [r["artifact_id"] for r in r2["rows"]]

    def test_call_count_stable(self) -> None:
        r1 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        r2 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        assert r1["total_calls"] == r2["total_calls"]

    def test_retrieval_example_dimensions(self) -> None:
        result = plan_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        # 2 questions * 3 KBs * 2 modes * 1 candidate * 1 result = 12
        assert result["total_calls"] == 12
        assert result["dimensions"]["questions"] == 2
        assert result["dimensions"]["knowledge_bases"] == 3


class TestPlanAskDefinition:
    """Ask-stage definitions carry no question catalog; questions come from the dataset."""

    def _definition(self, workspace: Path) -> Path:
        return workspace / "config" / "definitions" / "judgment.yaml"

    def test_plans_from_source_dataset(self, ask_workspace: Path) -> None:
        result = plan_definition(
            self._definition(ask_workspace),
            project_root=ask_workspace / "config",
            results_root=ask_workspace / "results",
        )
        assert result["definition_type"] == "retrieval_judgment"
        # 2 source artifacts * 1 model * 1 prompt * 1 inference = 2
        assert result["total_calls"] == 2
        assert result["dimensions"]["source_artifacts"] == 2
        assert result["dimensions"]["questions"] == 2
        assert result["selected_ids"]["models"] == ["judge_a"]
        assert result["source_dataset"] == {"type": "retrieval", "id": "src-ds"}
        assert len(result["rows"]) == result["total_calls"]

    def test_requires_results_root(self, ask_workspace: Path) -> None:
        with pytest.raises(PlanError, match="results directory"):
            plan_definition(
                self._definition(ask_workspace),
                project_root=ask_workspace / "config",
            )

    def test_missing_source_dataset_is_actionable(self, ask_workspace: Path) -> None:
        definition = dict(JUDGMENT_DEFINITION)
        definition["source_dataset"] = {"type": "retrieval", "id": "absent-ds"}
        path = ask_workspace / "config" / "definitions" / "absent.yaml"
        path.write_text(yaml.safe_dump(definition), encoding="utf-8")
        with pytest.raises(PlanError, match="Cannot plan against source dataset"):
            plan_definition(
                path,
                project_root=ask_workspace / "config",
                results_root=ask_workspace / "results",
            )

    def test_plan_matches_dry_run_count(self, ask_workspace: Path) -> None:
        planned = plan_definition(
            self._definition(ask_workspace),
            project_root=ask_workspace / "config",
            results_root=ask_workspace / "results",
        )
        result = runner.invoke(
            app,
            [
                "--workspace",
                str(ask_workspace),
                "judge-retrieval",
                "definitions/judgment.yaml",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert f"Planned calls: {planned['total_calls']}" in result.output

    def test_cli_plan_reports_approval_count(self, ask_workspace: Path) -> None:
        result = runner.invoke(
            app,
            ["--workspace", str(ask_workspace), "plan", "definitions/judgment.yaml"],
        )
        assert result.exit_code == 0, result.output
        assert "--approve-call-count 2" in result.output
        assert "source_artifacts" in result.output

    def test_deterministic_artifact_ids(self, ask_workspace: Path) -> None:
        args = (
            self._definition(ask_workspace),
            ask_workspace / "config",
            ask_workspace / "results",
        )
        first = plan_definition(*args)
        second = plan_definition(*args)
        assert [row["artifact_id"] for row in first["rows"]] == [
            row["artifact_id"] for row in second["rows"]
        ]
