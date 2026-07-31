# CLI Reference

**Status: All advertised commands are wired. Live stage commands require exact
call-count approval and preflight before dataset creation.**

The executable is `rag-evals`.

Every command requires a workspace selected by `--workspace PATH` or
`RAG_EVALS_WORKSPACE`. The global option must appear before the command. An
explicit option overrides the environment value. Commands fail with exit code 3
when neither is configured; help remains available.

## Global Options

| Option | Description |
|---|---|
| `--workspace PATH` | Workspace containing `config/` and `results/`; defaults from `RAG_EVALS_WORKSPACE`. |
| `--verbose`, `-v` | Enable verbose (DEBUG) logging. |
| `--help` | Show help and exit. |

## Commands

| Command | Status | API calls | Description |
|---|---|---:|---|
| `validate DEFINITION` | implemented | 0 | Resolve and validate configuration/templates |
| `plan DEFINITION` | implemented | 0 | Show matrix, exclusions, IDs, and call counts |
| `run-retrieval DEFINITION` | implemented | `/retrieve` | Build/resume retrieval dataset |
| `judge-retrieval DEFINITION` | implemented | `/ask` | Judge saved retrieval artifacts |
| `generate-answers DEFINITION` | implemented | `/ask` | Generate answers from frozen context |
| `judge-answers DEFINITION` | implemented | `/ask` | Judge saved answer artifacts |
| `report DEFINITION` | implemented | 0 | Produce CSV and Markdown reports |
| `compare DEFINITION` | implemented | 0 | Compare selected datasets |
| `status TYPE DATASET_ID` | implemented | 0 | Inspect dataset lifecycle and work counts |
| `resume DEFINITION` | implemented | varies | Resume eligible missing work |
| `seal TYPE DATASET_ID` | implemented | 0 | Make a complete dataset immutable |
| `clean-dataset TYPE DATASET_ID` | implemented | 0 | Plan or trash an eligible dataset |

Malformed definitions use the normal configuration error path; no command uses a
not-implemented sentinel.

`DEFINITION` is a relative path below `<workspace>/config`. Absolute paths and
paths escaping that directory are rejected.

## Implemented Command Details

### validate

```bash
rag-evals validate DEFINITION [--format text|json]
```

Resolves all source references, validates Pydantic schemas, applies selection
rules, and reports placeholder warnings. Outputs definition hash, input list
with content hashes, and validation status.

### plan

```bash
rag-evals plan DEFINITION [--format text|json]
```

Generates the deterministic matrix, prints dimensions, selected IDs, exclusions,
per-row artifact IDs, total call count, and the required `--approve-call-count`
value.

Ask-stage definitions (`retrieval_judgment`, `answer`, `answer_judgment`) have no
question catalog: their questions and rubrics are rebuilt from the source dataset,
which must be `complete` or `sealed`. Their plan reports a `source_artifacts`
dimension, and the call count equals source artifacts x models x prompt variants x
inference variants, matching the corresponding `--dry-run`.

### run-retrieval

```bash
rag-evals run-retrieval DEFINITION \
  [--dry-run] \
  [--only-missing] \
  [--question-id ID] \
  [--approve-call-count N]
```

Builds or resumes a retrieval dataset. `--dry-run` plans without API calls.
`--only-missing` resumes only missing artifacts. `--approve-call-count N` is
required for live execution and must exactly match the plan.

Placeholder KB IDs are rejected before any API call.

### report

```bash
rag-evals report DEFINITION
```

Joins lineage across datasets and generates
`<workspace>/results/reports/latest/report.csv` and `report.md`. Makes zero API
calls.

The report definition lists datasets, an output directory, and an optional
`scoring` block:

```yaml
datasets:
  retrieval: retrieval-smoke-001
  retrieval_judgments: retrieval-judge-smoke-001
output_dir: smoke
scoring:
  retrieval_judgment:
    parser: json
    minimum: 1
    maximum: 5
    criteria: [criterion_one, criterion_two]
    comment_field: comments
    source_assessment_field: source_assessments
```

Scores are parsed from the immutable raw judge text on every run, so criteria can
be changed and the report regenerated without new API calls. Each declared
criterion becomes a `retrieval_score_<name>` CSV column, plus a code-computed
`retrieval_score_total`; any total reported by the judge is ignored. Missing,
non-integer, or out-of-range values set `retrieval_parse_status` to `failed` with
`retrieval_parse_error` explaining why, and never produce a guessed score.
`answer_judgment` scoring works the same way.

CSV text columns are not truncated, and rows include chunking metadata, effective
mode, fallback state, dedup counts, relevance-score statistics, and estimated
context tokens so a spreadsheet pivot can aggregate per configuration.

Optional `source_assessment_field` names a JSON array in which the judge classifies
each retrieved source as `relevant`, `partial`, or `irrelevant`:

```json
"source_assessments": [{"index": 1, "relevance": "relevant"},
                       {"index": 2, "relevance": "irrelevant"}]
```

Counts and the ratio are then computed in code, not by the judge, and reported as
`judged_source_count`, `judged_sources_relevant`, `judged_sources_partial`,
`judged_sources_irrelevant`, and `judged_relevance_ratio`
(`relevant + 0.5 * partial`, divided by the source count). A missing or malformed
enumeration sets `judged_sources_status` to `failed` with `judged_sources_error`,
leaves the ratio empty, and does not affect criterion scores.

#### Markdown output

`report.md` contains one row per judged artifact, a `## Configuration Summary`
table, and a `## Cost Summary` separating tested-system from evaluation cost.

The configuration summary groups rows by knowledge base, chunking strategy,
retrieval mode, and merge state, then averages within each group:

| Column | Meaning |
|---|---|
| `Rows` | judged artifacts in the group |
| `Mean <criterion>` | mean judge score for each declared criterion |
| `Mean total` | mean of the code-computed criterion sum |
| `Mean relevant-source ratio` | mean `judged_relevance_ratio`, so 0.53 means about 53% of retrieved sources were useful with partials counted as half |
| `Mean est. tokens` | mean `estimated_context_tokens`, informational only |
| `Mean rel. score` | mean API relevance score |
| `Fallbacks` | rows where retrieval reported a mode fallback |
| `Parse failures` | rows whose judge output did not yield valid scores |

Rows lacking a value are skipped rather than counted as zero, and a group with no
values shows `-`.

The summary reports trade-offs only. It does not rank configurations or name a
winner, per RD013.

Two comparisons the numbers do not support:

- **API relevance scores are not comparable across retrieval modes.** Standard
  scores come from embedding similarity and rerank scores from a rerank model, so
  they are different scales. They are also not calibrated across differently chunked
  knowledge bases. Compare them only within one mode and one knowledge base.
- **The relevant-source ratio ignores rank order.** Returning the same sources in a
  better order produces an identical ratio, so the ratio under-credits reranking,
  whose main benefit is ordering.

### status

```bash
rag-evals status TYPE DATASET_ID
```

Displays lifecycle (`building`/`complete`/`sealed`/`failed`), planned count,
successful count, failed count, and missing count.
### seal

```bash
rag-evals seal TYPE DATASET_ID
```

Transitions a complete dataset to sealed. Rejects if not complete or already
sealed.

### clean-dataset

```bash
rag-evals clean-dataset TYPE DATASET_ID [--confirm-dataset-id ID]
```

Without `--confirm-dataset-id`: displays a dry-run inventory (lifecycle, file
count, size, downstream references, report references, can-clean status).

With `--confirm-dataset-id`: atomically moves the dataset to
`<workspace>/results/datasets/.trash/<type>/<id>-<timestamp>-<uuid>`. Rejects sealed datasets,
downstream-referenced datasets, and mismatched confirmation IDs.

## Exit Codes

| Code | Meaning | Error Class |
|---:|---|---|
| 0 | Success | - |
| 1 | General error | `RagEvalsError` |
| 2 | CLI usage error | Typer |
| 3 | Configuration error | `ConfigError` |
| 4 | Planning error | `PlanError` |
| 5 | Artifact store error | `ArtifactError` |
| 6 | API adapter error | `AdapterError` |
| 7 | Execution error | `ExecutionError` |

Configuration, artifact, adapter, and execution failures retain their documented
domain exit codes.

## Call Approval

Any live execution with planned API calls requires the exact current count:

```bash
rag-evals run-retrieval definitions/retrieval/test.yaml --approve-call-count 30
```

If filtering or source changes alter the plan, the command fails and prints the
new count. `--yes` is intentionally not provided.

## Cleanup Safety

The initial invocation is a dry run:

```bash
rag-evals clean-dataset retrieval my-dataset
```

Execution requires exact confirmation:

```bash
rag-evals clean-dataset retrieval my-dataset \
  --confirm-dataset-id my-dataset
```

The command refuses globs, prefixes, sealed datasets, and datasets referenced by
downstream manifests. It moves data to `.trash`; it does not permanently delete
it.
