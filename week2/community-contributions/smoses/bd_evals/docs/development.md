# Development

**Status: Implemented and covered by the offline quality checks below.**

## Environment

The package targets Python `>=3.12,<3.13`. Runtime dependencies are Pydantic 2,
PyYAML 6, Jinja2 3.1, Typer, and HTTPX. Development tools are pytest,
pytest-asyncio, pytest-cov, Ruff, mypy, and build.

`bd_api` is an external local dependency. Do not modify it from this project and
do not replace it with direct HTTP calls.

The wrapper currently lives at `../bd_api` and has no package metadata. Add its
parent directory to the import path for development:

```bash
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
python3.12 -c "import bd_api; print(bd_api.__file__)"
```

Live commands run `preflight_bd_api()` before dataset creation. Offline tests
mock the wrapper and make no network calls.

## Installation

```bash
cd bd_evals
python3.12 -m pip install -e ".[dev]"
```

This installs the `rag-evals` console script and all development dependencies.

## Verified Checks

```bash
python3.12 -m pytest
python3.12 -m ruff check .
python3.12 -m ruff format --check .
python3.12 -m mypy src/rag_evals
python3.12 -m build
```

All commands are verified. Run them from the `bd_evals` root.

## Source Layout

```text
src/rag_evals/
  __init__.py          # version 0.1.0
  cli.py               # Typer app and all advertised command dispatch
  errors.py            # 7 domain errors with stable exit codes 1-7
  logging.py           # structured logging with secret redaction
  workspace.py         # external config/results workspace selection
  py.typed             # PEP 561 marker for mypy
  config/
    __init__.py
    schemas.py         # Pydantic models: catalogs, definitions, config blocks
    resolver.py        # YAML source resolver, cycle detection, hashing
    execution.py       # resolved execution bundle and input snapshots
    selection.py       # include/exclude validation
    validate_cmd.py    # validate command implementation
  planning/
    __init__.py
    identity.py        # canonical_hash, artifact_id, stable_short_id
    matrices.py        # retrieval and ask-stage matrix generation
    plan_cmd.py        # plan command implementation
  artifacts/
    __init__.py
    io.py              # atomic JSON/YAML/text writes with fsync+os.replace
    store.py           # dataset creation, manifest, status, lifecycle, artifacts
    lineage.py         # lineage references and hash verification
    cleanup.py         # safe dataset cleanup with dry-run and trash
  api/
    __init__.py
    protocol.py        # ApiAdapter Protocol, requests, responses
    fake_adapter.py    # deterministic fake with configurable failures
    real_adapter.py    # RealBdApiAdapter importing AsyncBdApiClient
    retry.py           # D004 retry executor, D005 concurrency semaphore
  templates/
    __init__.py        # Jinja2 sandbox, validation, rendering
  stages/
    __init__.py
    retrieval.py       # retrieval artifact schema and runner
    judgment.py        # retrieval and answer judgment stages
    answer.py          # frozen answer generation stage
  reporting/
    __init__.py
    reports.py         # lineage join, CSV/Markdown renderers
```

## Test Organization

| File | Tests | Scope |
|---|---|---|
| `tests/test_package.py` | 3 | Import, version, CLI --help |
| `tests/test_cli.py` | 30 | Command registration, help, implemented/stub behavior |
| `tests/test_errors.py` | 22 | Error hierarchy, exit codes, uniqueness |
| `tests/test_logging.py` | 9 | Secret redaction, nested, non-mutation |
| `tests/test_schemas.py` | 66 | Catalogs, definitions, validation, parse_definition |
| `tests/test_resolver.py` | 22 | YAML loading, cycles, traversal, hashes, resolution |
| `tests/test_selection.py` | 12 | Include/exclude, unknown IDs, ordering |
| `tests/test_templates.py` | 27 | Variables, filters, validation, rendering |
| `tests/test_validate_cmd.py` | 8 | CLI validate, example definition, JSON format |
| `tests/test_identity.py` | 25 | Canonical hash, artifact IDs, paths, NaN rejection |
| `tests/test_matrices.py` | 8 | Cartesian, ordering, merge exclusion, stable IDs |
| `tests/test_plan_cmd.py` | 8 | Help, example, JSON, deterministic, dimensions |
| `tests/test_artifacts.py` | 35 | Atomic IO, dataset creation, lifecycle, resume, cleanup |
| `tests/test_adapter.py` | 17 | Fake retrieve/ask, retry, concurrency |
| `tests/test_retrieval_stage.py` | 14 | Success, failure, retry, resume, placeholder rejection |
| `tests/test_stages_integration.py` | 11 | Judgment, answer, reporting integration |
| `tests/test_workspace.py` | 12 | Workspace validation, boundaries, CLI/env selection, isolation |
| **Current total** | **360** | Includes additional remediation and functional regression files |

## Test Strategy

- Configuration: YAML errors, references, cycles, traversal, IDs, selection,
  optional rubrics, templates.
- Planning: retrieval/model matrices, exclusions, call counts, deterministic
  identity.
- Artifact store: atomic writes, hashing, lifecycle, resume, lineage, stale
  detection, cleanup.
- Adapter: mocked async wrapper behavior, SSE aggregation outputs, error
  classification, retries.
- Stages: fake adapter integration only; no live unit-test calls.
- Reporting: lineage joins, missing judgments, cost partitions, stable
  CSV/Markdown.

Live smoke tests must be explicitly enabled and skipped by default.

## Documentation Workflow

Documentation is part of every task:

1. Update affected files in `docs/`.
2. Change planned statements to implemented only after verification.
3. Copy command syntax from actual `--help`.
4. Keep examples validated in automated tests where practical.
5. Record documentation changes in the change summary.
6. Do not delete known limitations merely because they are inconvenient.

## Adding a Stage or Test Type

Reuse configuration resolution, planner identity, artifact lifecycle, adapter
protocol, and lineage verification. A new stage needs its own definition schema,
plan rows/call count, artifact type or justified reuse, fake integration tests,
CLI entry, and documentation.

## External API Enhancements

The user owns changes to the Bedrock API and `bd_api`. This project maintains a
precise limitations list in `docs/architecture.md` under "Known Wrapper
Limitations". Current requested future enhancement: return token usage from
`/retrieve` and expose it through the wrapper. Potential wrapper enhancements
also include parsed request error metadata, complete ask start metadata, raw SSE
capture, configurable timeout, and streaming timings.

## Coding Conventions

- Python 3.12 with `from __future__ import annotations`
- Strict Pydantic models (`extra="forbid"`) except extensible rubric and
  inference config (`extra="allow"`)
- `StrEnum` for enumerations
- No comments in source code unless explicitly requested
- Ruff for linting and formatting (line length 100)
- Mypy strict mode with no errors
- Atomic writes for all file operations (temp + fsync + os.replace)
- UTC RFC 3339 timestamps ending in `Z`
