# Architecture

**Status: Implemented and verified. All packages, stages, and reporting are
covered by the current offline test suite.**

## Purpose

The project evaluates an existing Bedrock RAG API through its Python `bd_api`
wrapper. It separates expensive operations into reusable, immutable datasets so
retrieval can be evaluated independently from generation and traced into answer
quality.

```text
questions and catalogs
        |
        v
retrieval execution --> retrieval dataset --> retrieval judgments
                              |
                              v
                    frozen-context answers --> answer judgments
                              |
                              v
                     offline reports and joins
```

## Core Boundaries

- **Verified:** `/retrieve` returns raw retrieval output, normalized items,
  metadata, estimated retrieval/reranking cost, and profile-formatted context at
  `preparedPromptInput.promptVariables.context`.
- **Verified:** `/ask` streams SSE and `bd_api` aggregates answer text, sources,
  usage, metrics, costs, stop reason, answer ID, persistence status, and
  retrieval metadata.
- **Verified:** all API access uses `bd_api` through a narrow adapter protocol;
  the framework does not call Bedrock or HTTP directly.
- **Verified:** the filesystem is the only MVP database.
- **Verified:** retrieval, answers, and judgments are separate datasets linked
  by IDs and SHA-256 hashes.

## Workspace Boundary

Installed package code does not contain user configuration or generated results.
The global `--workspace PATH` option, or `RAG_EVALS_WORKSPACE`, selects a
workspace with `config/` and `results/`. Explicit CLI selection takes precedence
and commands fail closed when neither is configured.

Definitions and their sources remain below `<workspace>/config`. Artifacts,
trash, and reports remain below `<workspace>/results`. All 12 commands use this
same boundary, allowing multiple isolated workspaces to share one installed
package.

## Frozen Retrieval

Answer generation loads context from a saved retrieval artifact and renders it
into the test's `user_message.yaml` before calling `/ask`. The final
`user_message` already contains the frozen context.

`/ask` may perform another retrieval internally. Its sources and retrieval
metadata are incidental and are stored separately with:

```text
used_as_frozen_generation_context: false
```

They never replace the saved retrieval artifact as answer provenance.

**Verified:** `AnswerArtifact` stores `frozen_context_hash` separately from
`incidental_retrieval`. See `src/rag_evals/stages/answer.py`.

## Package

**Verified:** All modules are implemented:

```text
src/rag_evals/
  __init__.py          # version 0.1.0
  cli.py               # Typer app and all 12 workspace-aware commands
  errors.py            # 7 domain errors with stable exit codes 1-7
  logging.py           # structured logging with secret redaction
  workspace.py         # config/results workspace and path boundaries
  py.typed             # PEP 561 marker
  config/
    __init__.py
    schemas.py         # Pydantic models: catalogs, definitions, config blocks
    resolver.py        # YAML source resolver, cycle detection, hashing
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

Configuration resolves and validates YAML. Planning creates deterministic
matrices and call counts. The artifact store owns atomic writes, lifecycle,
hashing, lineage, and cleanup. The adapter isolates `bd_api`. Stage runners
perform one pipeline stage each. Reporting reads files only.

## Concurrency and Reliability

**Verified:** `AsyncBdApiClient` implements asynchronous retrieval and ask
calls, including async SSE aggregation.

**Verified:** runners use a configurable async semaphore with default
concurrency 5 (`ConcurrencyExecutor` in `src/rag_evals/api/retry.py`).
Completion order does not affect IDs, manifests, or reports. Each matrix row
owns its retries. The retry policy (D004) allows at most three attempts for
explicitly retryable API failures, selected transport failures, and HTTP
429/503/504, using exponential backoff with full jitter.

## Known Wrapper Limitations

- Request errors expose status and raw body but not parsed API `code` or
  `retryable`; the adapter parses these defensively in
  `src/rag_evals/api/real_adapter.py`.
- Ask aggregation discards some `start` metadata and raw SSE events. These are
  recorded as limitations in each ask-based artifact.
- The wrapper has no first-event or first-token timing.
- Adapter-observed elapsed time and API `metrics.latencyMs` are stored
  separately.
- `/retrieve` returns no token usage. Retrieval context tokens use
  `ceil(character_count / 4)` and are labeled `estimated` until the API and
  wrapper are enhanced.

## Future Reporting Upgrade

MVP reports are deterministic CSV and Markdown suitable for Google Sheets.
Charts and richer visualization are the next logical reporting phase after real
result data establishes useful chart requirements.
