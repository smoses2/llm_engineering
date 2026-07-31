# Evaluation Workflows

**Status: Implemented and verified. All stages work with fake adapter in tests.
Live execution requires real KB IDs, model IDs, and call-count approval.**

The examples assume `RAG_EVALS_WORKSPACE` points to a directory containing
`config/` and `results/`. Definition paths are relative to its `config/`
directory.

## 1. Validate and Plan

Always validate before execution. Planning resolves all inputs, displays selected
IDs, exclusions, matrix dimensions, deterministic IDs, and exact API call counts.

**Verified commands:**

```bash
# Validate configuration and templates
rag-evals validate definitions/retrieval-example.yaml

# Validate with JSON output
rag-evals validate definitions/retrieval-example.yaml --format json

# Plan (shows dimensions, call counts, approval requirement)
rag-evals plan definitions/retrieval-example.yaml

# Plan with JSON output
rag-evals plan definitions/retrieval-example.yaml --format json
```

Any live run with nonzero calls requires `--approve-call-count N`, where `N`
exactly matches the current plan. This catches changed inputs and accidental
Cartesian expansion.

**Verified output (plan):**

```text
Definition type: retrieval
Test ID: retrieval-example
Dimensions: {'questions': 2, 'knowledge_bases': 3, 'modes': ['rerank', 'standard'], ...}
Total calls: 12
Required approval: --approve-call-count 12
```

## 2. Run Retrieval

For each question, KB, retrieval mode, candidate count, result count, and
declared deployment state, the runner calls `/retrieve` through
`AsyncBdApiClient`.

It saves complete raw JSON, normalized items, formatted context, fallback
metadata, cost, elapsed time, retries, and objective metrics. Retrieval context
token count is a labeled estimate because the endpoint has no usage metadata.

**Verified command (dry run):**

```bash
rag-evals run-retrieval definitions/retrieval-example.yaml --dry-run
```

**Live execution requires:**

```bash
rag-evals run-retrieval definitions/retrieval-example.yaml \
  --approve-call-count 12
```

Placeholder KB IDs (`KB_REPLACE_ME_*`) are rejected before any API call.

Merge-disabled and merge-enabled deployments normally produce separate datasets.
No-chunking KBs participate only in merge-disabled runs.

## 3. Judge Retrieval

The retrieval judge reads an existing complete or sealed retrieval dataset. It
never reruns retrieval. It renders question, frozen context, optional
items/metadata, and optional rubric into example raw-text judge prompts, then
invokes `/ask`.

Judgment output is a separate artifact. Initial parsing mode is `raw_text`; no
score schema or JSON repair is invented.

## 4. Generate Frozen Answers

The answer runner verifies the retrieval artifact hash, extracts the exact saved
context, and renders a final user message. It can generate multiple answers from
one retrieval artifact using explicit model, prompt, and inference variants.

Answer artifacts retain the retrieval dataset ID, artifact ID, hash, relative
reference, and frozen context hash. Exact `/ask` token usage and server latency
are stored when present. Internal `/ask` retrieval is labeled incidental.

## 5. Judge Answers

The answer judge reads existing answer artifacts and their verified retrieval
lineage. It renders question, frozen context, candidate answer, optional rubric,
and optional source data into its own prompt. It does not regenerate answers or
retrieval.

## 6. Report and Compare

Reports perform no API calls. They join:

```text
question -> retrieval -> retrieval judgment -> answer -> answer judgment
```

CSV and Markdown outputs compare KBs, retrieval modes, merge states, models,
prompts, inference settings, costs, latency, fallback, failures, and available
judge results. Tested-system and evaluation-infrastructure costs remain
separate.

**Verified command:**

```bash
rag-evals report definitions/reports/retrieval-evaluation.yaml
```

The report definition identifies retrieval and retrieval-judgment dataset IDs.
Output defaults to `<workspace>/results/reports/latest/report.csv` and
`report.md`.

## Resume

Interrupted building datasets are reconciled from actual artifacts. Successful
artifacts are never rewritten. Missing and eligible failed work can resume.
Completion order under concurrency does not change deterministic output ordering.

**Verified:** `--only-missing` flag on `run-retrieval` resumes only missing
artifacts. Existing artifact bytes are preserved. See
`tests/test_retrieval_stage.py::TestRetrievalStageResume`.

## Clean and Restart a Dataset

Use `clean-dataset` instead of manually deleting output. It requires exact type
and ID, displays a dry run, and requires exact ID confirmation. It rejects sealed
or downstream-referenced datasets and never cascades. Eligible data moves to
`.trash` so it can be restored. After the move, the same dataset ID can be run
from scratch.

**Verified commands:**

```bash
# Dry run (default)
rag-evals clean-dataset retrieval my-dataset

# Execute
rag-evals clean-dataset retrieval my-dataset \
  --confirm-dataset-id my-dataset
```

## Seal a Dataset

```bash
rag-evals seal retrieval my-dataset
```

Transitions a complete dataset to sealed. Sealed datasets reject all mutations
including artifact writes, failure appends, and cleanup.

## Check Status

```bash
rag-evals status retrieval my-dataset
```

Displays lifecycle, planned/successful/failed/missing counts.
