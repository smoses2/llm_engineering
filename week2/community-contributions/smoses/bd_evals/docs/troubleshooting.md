# Troubleshooting

**Status: Verified operational guidance with actual error messages and exit
codes from implemented commands.**

## Invalid YAML or Schema

Run validation and use the reported source path. Relative references are
resolved from the containing file. Check indentation, `schema_version`,
duplicate IDs, selected IDs, and `result_count <= candidate_count`.

```bash
rag-evals validate definitions/retrieval-example.yaml
```

**Verified error:** Missing `test` block produces exit code 3 with message:
`Error: Definition must contain a 'test' block`.

Unknown template variables fail before API execution. Missing optional rubric
data must be handled through an approved template default filter (`default`),
not an unknown or misspelled variable.

## Workspace Missing or Invalid

Every command requires `--workspace PATH` or `RAG_EVALS_WORKSPACE`. The selected
directory must contain both `config/` and `results/`.

```bash
export RAG_EVALS_WORKSPACE=/path/to/workspace
```

Definition arguments are relative to `<workspace>/config`. Absolute paths,
`..` traversal, and symlinks escaping `config/` are rejected.

## Placeholder Rejected

Example KB and model IDs are intentionally non-runnable. Replace
`KB_REPLACE_ME_*`, `MODEL_REPLACE_ME_*`, and placeholder catalog entries with
real deployment values before a live run.

**Verified error:** Running `run-retrieval` with placeholder KB IDs produces
exit code 7 with message: `Error: Cannot execute live retrieval with placeholder
KB ID: KB_REPLACE_ME_1 (catalog ID: kb_placeholder_1). Replace with a real
Knowledge Base ID.`

## API Authentication or URL Failure

Verify that `BD_API_BASE_URL` and `BD_API_KEY` are present in the process
environment. Never add their values to YAML. `/ask` and `/retrieve` normally
require the API Gateway key.

**Verified error:** Missing `BD_API_BASE_URL` produces exit code 3 with message:
`Error: BD_API_BASE_URL environment variable not set.`

If `bd_api` fails with `No module named 'httpx'`, reinstall this package in the
same Python environment with `python3.12 -m pip install -e ".[dev]"`. `httpx`
is a declared runtime dependency.

## Rerank Falls Back to Standard

Inspect retrieval metadata for requested/effective mode, `fallback_used`, and
reason. The Knowledge Base service role needs Bedrock invoke/rerank and
Marketplace permissions. Fallback artifacts remain valid but reports classify
them separately.

**Verified:** Retrieval artifacts store `retrieval_metadata.fallback_used` and
`objective_metrics.has_fallback`.

## Throttling

Default concurrency is 5, but API Gateway, Lambda, Bedrock, model, and account
quotas may differ. Lower configured concurrency in the definition's
`execution.concurrency` field. Retryable 429/503/504 and explicit retryable API
errors use the approved bounded retry policy (max 3 attempts, exponential
backoff with full jitter).

**Verified:** `RetryConfig` defaults to `maximum_attempts: 3`. Tests confirm
retry on 503 and no retry on 400.

## Missing Retrieval Token Usage

This is a known API limitation. Retrieval artifacts record
`ceil(context characters / 4)` as an estimate. `/ask` artifacts use exact
API-returned usage when available. A future API and `bd_api` update should add
retrieval token usage.

**Verified:** Retrieval artifacts store `estimated_context_tokens`,
`token_estimate_method: "ceil(context_character_count / 4)"`, and
`token_estimate_label: "estimated"`. Answer artifacts store `usage.inputTokens`,
`usage.outputTokens`, `usage.totalTokens` from the API.

## Invalid Judge Output

Initial judge mode is raw text, so no JSON parse is required. Future strict JSON
mode must preserve raw text and report parse failure; automatic repair is not
allowed unless separately approved.

**Verified:** Judgment artifacts store `raw_judge_text` and
`parser_mode: "raw_text"`. The `parsed_result` field is `null` until a
production score schema is defined.

## Stale or Changed Dataset

A dataset ID cannot be reused with a different resolved-definition hash. Create
a new ID, or clean an eligible unsealed/unreferenced dataset before rerunning it
from scratch.

**Verified error:** Reusing a dataset ID with a different definition produces
exit code 5 with message: `Dataset ... exists with a different resolved-definition
hash. Use a different dataset ID.`

## Cleanup Refused

- Sealed datasets cannot be cleaned.
- A dataset referenced by downstream manifests cannot be cleaned.
- Confirmation must exactly equal the dataset ID.
- Cleanup never accepts wildcard or prefix selection.

**Verified errors:**
- Sealed: `Cannot clean: Dataset is sealed`
- Referenced: `Cannot clean: Dataset is referenced by N downstream dataset(s)`
- Wrong confirmation: `Confirmation ID 'wrong' does not match dataset ID 'my-dataset'`

Reports referencing a cleaned dataset may become stale and are reported as
warnings. Restore from `.trash` before reusing the ID if cleanup was
unintended.

## Interrupted Run

Inspect `status`, then resume only missing or eligible failed rows. Successful
artifact bytes must remain unchanged. If the status index disagrees with actual
files, reconciliation uses verified artifacts as the source of truth.

**Verified commands:**

```bash
# Check status
rag-evals status retrieval my-dataset

# Resume missing artifacts
rag-evals run-retrieval definitions/retrieval/my-test.yaml \
  --only-missing --approve-call-count N
```

## Exit Codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 1 | General error |
| 2 | CLI usage error |
| 3 | Configuration error (invalid YAML, missing file, missing env var) |
| 4 | Planning error |
| 5 | Artifact store error (sealed, stale, hash mismatch, cleanup refused) |
| 6 | API adapter error |
| 7 | Execution error (placeholder KB, execution failure) |
