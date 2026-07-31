# Tests

## How to Run

### Prerequisites

```bash
cd bd_evals
python3.12 -m pip install -e ".[dev]"
```

### Run All Tests

```bash
python3.12 -m pytest
```

All 360 tests pass with no network access. No live API calls are made.

### Run a Single File

```bash
python3.12 -m pytest tests/test_schemas.py
```

### Run a Single Test Class or Method

```bash
python3.12 -m pytest tests/test_adapter.py::TestRetryExecutor
python3.12 -m pytest tests/test_adapter.py::TestRetryExecutor::test_retry_on_503
```

### Run with Verbose Output

```bash
python3.12 -m pytest tests/ -v
```

### Run with Coverage

```bash
python3.12 -m pytest tests/ --cov=rag_evals --cov-report=term-missing
```

## Async Tests

Async tests use `pytest-asyncio` in auto mode (configured in `pyproject.toml`).
All `async def` test functions are automatically detected; no `@pytest.mark.asyncio`
decorator is needed.

## Test Strategy

- **No live API calls.** Every test uses the `FakeAdapter`
  (`src/rag_evals/api/fake_adapter.py`) which generates deterministic responses
  and can simulate configurable failures.
- **No network access required.** The entire suite runs offline.
- **No credentials.** No test reads real API keys or URLs from the environment.

## Test Files

| File | Tests | What it covers |
|---|---:|---|
| `test_package.py` | 3 | Package import, `__version__`, top-level `--help` |
| `test_cli.py` | 24 | All 12 commands respond to `--help` and execute through a selected workspace |
| `test_errors.py` | 23 | Error hierarchy, stable exit codes 1–7, uniqueness, message preservation |
| `test_logging.py` | 9 | Secret key detection, recursive redaction in flat/nested/list structures, non-mutation, passthrough |
| `test_schemas.py` | 66 | Pydantic models for questions, KBs, models, API config, deployment, execution, output, selection, retrieval tuning, all 5 definition types, `parse_definition` dispatcher |
| `test_resolver.py` | 22 | YAML safe loading, containing-file-relative resolution, cycle detection, path traversal rejection, absolute path rejection, missing sources, snapshot deduplication, content hashes |
| `test_selection.py` | 12 | Include/exclude rules, unknown-ID rejection, zero-result rejection, deterministic catalog ordering |
| `test_templates.py` | 27 | Variable extraction, filter extraction, stage allowlists (retrieval_judge, answer, answer_judge), allowed filters (`to_yaml`, `to_json`, `length`, `default`), rejection of statements/calls/imports/includes/macros, rendering with `StrictUndefined` |
| `test_validate_cmd.py` | 8 | `rag-evals validate` on example definition, JSON output, missing file error, resolved catalog loading, placeholder warnings |
| `test_identity.py` | 25 | Canonical JSON hashing (sorted keys, compact separators, UTF-8), SHA-256, NaN/Infinity rejection, set serialization, artifact ID format, path generation, `identity_hash` excluding `content_hash` |
| `test_matrices.py` | 8 | Retrieval matrix cartesian product, deterministic ordering across input permutations, merge/no-chunk exclusion, `result > candidate` skipping, stable artifact IDs |
| `test_plan_cmd.py` | 8 | `rag-evals plan` on example definition, JSON output, deterministic IDs across repeated runs, stable call counts, dimension verification (2×3×2 = 12 calls) |
| `test_artifacts.py` | 35 | Atomic IO (JSON/YAML/text with fsync), exclusive create, dataset creation/manifest/status, artifact writing (no overwrite of same, reject different), append-only failures, lifecycle transitions (building→complete→sealed, sealed rejects all), resume finds missing and preserves existing, cleanup dry-run/execute/trash/reuse/downstream-reference-block |
| `test_adapter.py` | 17 | Fake adapter retrieve/ask success, call recording, determinism, configurable failures; retry executor (success first try, retry on retryable, no retry on non-retryable, max attempts exhausted, retry on 503, no retry on 400, jitter); concurrency semaphore (default 5, serial at 1, max-concurrent enforcement) |
| `test_retrieval_stage.py` | 14 | Token estimate (`ceil(chars/4)`), full success reaching complete lifecycle, dry run, artifact content verification (context, estimated tokens, objective metrics), partial failure with continue-on-error, retry then success, placeholder KB rejection, resume missing artifacts, resume preserves existing bytes |
| `test_stages_integration.py` | 11 | Retrieval judgment success and raw text storage, answer generation success with exact usage/frozen context hash/no retrieval call, report generation (CSV + Markdown), lineage join with judgments, CSV determinism, cost summary separation, zero adapter calls during reporting |
| `test_cli_functional.py` | 4 | Offline ask-stage planning and workspace report output |
| `test_final_scenario.py` | 3 | Multi-variant pipeline, fail-fast behavior, source verification failures |
| `test_remediation_core.py` | 13 | Transactionality, races, retries, cleanup, adapter and status regressions |
| `test_resolved_bundle.py` | 1 | Resolved input and prompt hashing |
| `test_resolver_cache.py` | 2 | Repeated source reference equality |
| `test_schema_guardrail.py` | 2 | Manifest schema migration guardrails |
| `test_workspace.py` | 12 | Workspace validation, boundaries, CLI/env precedence, and isolation |

## Adding New Tests

1. Create `tests/test_<feature>.py`.
2. Use `FakeAdapter` for any API interaction.
3. For async tests, define `async def test_*` functions; auto mode handles the rest.
4. Use `tmp_path` fixture for filesystem tests (creates isolated temp directories).
5. Run `python3.12 -m pytest tests/test_<feature>.py -v` to verify.
