# Artifact Model

**Status: Manifest schema 2. Schema-1 datasets are incompatible and rejected;
clean and regenerate them under a new dataset ID.**

## Dataset Types

```text
<workspace>/results/datasets/
  retrieval/
  retrieval_judgments/
  answers/
  answer_judgments/
  .trash/
```

Each dataset contains `manifest.yaml`, `resolved_definition.yaml`, `status.yaml`,
input snapshots, immutable successful artifacts, append-only failed attempts,
and optional summaries.

**Verified:** `DatasetPath` in `src/rag_evals/artifacts/store.py` resolves all
paths. `create_dataset()` creates the full directory structure with manifest,
resolved definition, and status.

## Lifecycle

| Transition | From | To | Condition |
|---|---|---|---|
| Create | (none) | `building` | New dataset with plan |
| Complete | `building` | `complete` | All planned artifacts have successful results |
| Seal | `complete` | `sealed` | Explicit `seal` command |
| Fail | `building` | `failed` | Dataset-level terminal failure |

- `building`: missing and failed work may resume.
- `complete`: all and only planned rows are valid successes; immutable.
- `sealed`: explicitly immutable; no mutation or cleanup.
- `failed`: dataset-level terminal failure when completion cannot proceed.

Successful artifacts are never silently overwritten. Changed resolved
configuration requires a different dataset ID. The current CLI has no `--force`
option.

**Verified:** `transition_lifecycle()` enforces all transition rules.
Artifact and failure writes are accepted only while building. Dataset reuse
requires matching resolved-definition and ordered plan hashes.

## Identity and Hashing

Canonical serialization is UTF-8 JSON with sorted keys, compact separators, no
NaN, and SHA-256. Timestamps never determine cache identity.

**Verified:** `canonical_hash()` in `src/rag_evals/planning/identity.py`.
Artifact IDs are `sha256:` + full lowercase hex digest. Content hashes cover the
artifact payload excluding the `content_hash` field.

| Identity | Included inputs |
|---|---|
| Retrieval | identity version/stage, question ID/text, KB catalog/actual ID, mode/counts, deployment and merge state |
| Answer | exact source ID/hash, question ID/text/rubric hash, model catalog/actual ID, prompt variant and both template hashes, inference variant/config, identity version/stage |
| Judgment | the complete ask identity above with retrieval-judge or answer-judge stage and exact source artifact |

## Lineage

Downstream artifacts record source dataset type/ID, source artifact ID,
results-relative path, and expected content hash. Loaders verify every field.
Retrieval data is not copied into downstream datasets, although exact frozen
context hashes and rendered messages are retained where required for
auditability.

**Verified:** `LineageRef` and `verify_lineage()` in
`src/rag_evals/artifacts/lineage.py`. Verifies existence, hash match, and
acceptable source lifecycle (complete or sealed).

## Retrieval Artifacts

They preserve identity, question/rubric, deployment, KB metadata, request,
complete raw response, normalized items, exact formatted context, retrieval
metadata, fallback/deduplication, cost, retries, errors, elapsed time, and
objective metrics.

The context token metric is `ceil(context characters / 4)`, labeled as an
estimate. Future API retrieval token usage must be stored separately and must
not silently replace historical estimate semantics.

**Verified:** `RetrievalArtifact` in `src/rag_evals/stages/retrieval.py`.
Fields: `estimated_context_tokens`, `token_estimate_method`,
`token_estimate_label`. Tests in `tests/test_retrieval_stage.py`.

## Answer and Judgment Artifacts

Ask-based artifacts preserve returned answer text, sources, exact usage
dictionaries, metrics including `latencyMs` when present, costs, stop reason,
answer ID, persistence status, client elapsed time, retries, and errors.

Incidental ask retrieval is structurally separated from authoritative frozen
retrieval. Judgments always store raw model text; parsed data is optional.

**Verified:** `AnswerArtifact` stores `usage` (from API `done` event),
`metrics` (including `latencyMs`), `frozen_context_hash`, and
`incidental_retrieval` with `used_as_frozen_generation_context: false`.
`JudgmentArtifact` stores `raw_judge_text` and `parser_mode: "raw_text"`.

## Cost Separation

Reports distinguish original retrieval/reranking cost, tested answer inference
and guardrail costs, incidental answer-call retrieval, retrieval-judge invocation
cost, answer-judge invocation cost, and incidental judge retrieval. Judge cost
is never counted as tested-system cost.

**Verified:** `render_markdown()` in `src/rag_evals/reporting/reports.py`
produces a cost summary separating tested-system from evaluation infrastructure.

## Atomic Writes and Cleanup

Mutable metadata uses unique same-directory temporary files. Immutable artifacts
use no-replace `O_EXCL` creation. Dataset initialization uses a unique temporary
directory and atomic publication.

Cleanup atomically renames an exact eligible dataset into timestamped
project-local trash. It does not permanently delete data or cascade into
downstream datasets.

**Verified:** `atomic_write_bytes()` in `src/rag_evals/artifacts/io.py` uses
temp file, `fsync`, `os.replace`. `execute_cleanup()` in
`src/rag_evals/artifacts/cleanup.py` moves to
`<workspace>/results/datasets/.trash/<type>/<id>-<timestamp>-<uuid>`.

## Restore Procedure

The library provides validated `restore_dataset()` behavior, but there is no
public restore CLI command yet. Do not manually move or rewrite trashed datasets
without an approved recovery procedure. Trash remains under
`<workspace>/results/datasets/.trash/` with all original artifacts intact.

Permanent trash purge is outside MVP.
