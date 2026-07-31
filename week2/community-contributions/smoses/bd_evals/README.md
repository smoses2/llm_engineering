# Bedrock RAG Evals

Bedrock RAG Evals is a Python 3.12 command-line framework for evaluating an
existing Bedrock RAG service through its local `bd_api` wrapper. It separates
retrieval, retrieval judgment, frozen-context answer generation, and answer
judgment into traceable datasets so expensive results can be reused and compared.

## Pipeline

```text
questions + catalogs + evaluation definition
                    |
              validate and plan
                    |
                    v
retrieval dataset ---> retrieval judgments
        |
        v
frozen-context answers ---> answer judgments
        |
        v
offline CSV and Markdown reports
```

Configuration resolution, deterministic matrix planning, artifact storage, API
adaptation, stage execution, and reporting are separate modules under
`src/rag_evals/`. Datasets are linked by IDs and SHA-256 hashes; reports read those
artifacts without making API calls.

## Install

The package requires Python `>=3.12,<3.13`.

```bash
cd bd_evals
python3.12 -m pip install -e ".[dev]"
rag-evals --help
```

`bd_api` is a local external dependency, not installed by this package. In the
expected sibling layout, expose its parent directory before live execution:

```bash
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
python3.12 -c "import bd_api; print(bd_api.__file__)"
```

This checkout instead uses a shared, git-ignored `smoses/.venv` with a
`site-packages/_bd_api_path.pth` file holding the `smoses` path, so `bd_api`
imports from any working directory without exporting `PYTHONPATH`. Recreate that
`.pth` file if the virtual environment is rebuilt.

Live stage commands run a `bd_api` compatibility preflight before creating a
dataset. Do not modify `bd_api` from this project. Offline validation, planning,
reporting, and tests do not require network access.

## Select a Workspace

Package source is separate from evaluation configuration and results. Every
command requires an external workspace containing `config/` and `results/`:

```bash
export RAG_EVALS_WORKSPACE="$(cd ../bd_evals_config_example && pwd)"
```

Or pass it explicitly before the command:

```bash
rag-evals --workspace ../bd_evals_config_example --help
```

The explicit option overrides `RAG_EVALS_WORKSPACE`. Commands fail closed when
neither is configured. Definition arguments resolve relative to the selected
workspace's `config/` directory. See `docs/workspaces.md`.

## Secrets

Commands load `<workspace>/.env` and then `<workspace>/../.env`, if present,
without overriding variables already exported in the shell. A single untracked
file beside the workspaces therefore serves every workspace and the `bd_api`
helper scripts:

```text
BD_API_BASE_URL=<service base URL>
BD_API_KEY=<service API key>
```

Definitions reference these names, never their values.

## Configure

The sibling `bd_evals_config_example` workspace contains safe examples, not
production configuration:

- `config/questions.yaml`: questions and optional rubrics
- `config/knowledge-bases.yaml`: Knowledge Base catalog
- `config/answer-models.yaml`: answer model catalog
- `config/judge-models.yaml`: judge model catalog
- `config/definitions/retrieval-example.yaml`: retrieval evaluation definition

Catalog and definition files use `schema_version: 1`. Replace values such as
`KB_REPLACE_ME_*` and `MODEL_REPLACE_ME_*` with real identifiers before a live
run. Placeholder values intentionally validate and plan offline but are rejected
for live execution. API configuration stores environment variable names, normally
`BD_API_BASE_URL` and `BD_API_KEY`, rather than secret values.

Ask-based stages require their own definition YAML with the appropriate
`test.type`: `retrieval_judgment`, `answer`, or `answer_judgment`. See
`docs/configuration.md` and `docs/examples.md` for schemas and examples.

## Validate, Plan, Run

Always resolve and validate inputs first, then inspect the deterministic matrix
and exact call count:

```bash
rag-evals validate definitions/retrieval-example.yaml
rag-evals plan definitions/retrieval-example.yaml
rag-evals run-retrieval definitions/retrieval-example.yaml --dry-run
```

`validate` and `plan` make no API calls. Every live run requires an explicit count
that exactly matches the current selected work:

```bash
rag-evals run-retrieval definitions/retrieval-example.yaml \
  --approve-call-count 12
```

The same approval rule applies to `judge-retrieval`, `generate-answers`,
`judge-answers`, and `resume`. If configuration, filters, source datasets, or
missing work change the count, execution stops and reports the required value.
There is no blanket confirmation flag.

## Datasets and Recovery

Successful artifacts are immutable. A dataset moves through `building`,
`complete`, `sealed`, or `failed` lifecycle states, and downstream stages verify
source hashes before use.

```bash
rag-evals status retrieval my-dataset
rag-evals resume path/to/definition.yaml --approve-call-count N
rag-evals seal retrieval my-dataset
```

Resume selects eligible missing or failed work without rewriting successful
artifacts. Sealed datasets reject mutation. Cleanup is a two-step, exact-ID
operation and moves eligible data to `results/datasets/.trash` rather than
deleting it:

```bash
rag-evals clean-dataset retrieval my-dataset
rag-evals clean-dataset retrieval my-dataset \
  --confirm-dataset-id my-dataset
```

Cleanup refuses sealed datasets and datasets referenced by downstream manifests.
References from existing reports are shown as warnings because those reports may
become stale after cleanup.

## Reports

`report` and `compare` are offline operations. Reports join lineage from questions
through retrieval, judgments, and answers, while keeping tested-system and
evaluation-infrastructure costs separate.

```bash
rag-evals report path/to/report-definition.yaml --format both
rag-evals compare path/to/compare-definition.yaml --format markdown
```

Report output defaults to `<workspace>/results/reports/latest/`. See
`docs/artifact-model.md` and `docs/cli-reference.md` for formats and command
details.

## Tests and Documentation

```bash
python3.12 -m pytest
python3.12 -m ruff check .
python3.12 -m ruff format --check .
python3.12 -m mypy src/rag_evals
python3.12 -m build
```

The user and developer manual starts at `docs/README.md`. Tests use fake or mocked
adapters and make no live calls by default.

## Safety and Limitations

- Live calls require real identifiers, environment configuration, successful
  wrapper preflight, and exact call-count approval.
- Secrets must remain in environment variables; they are redacted from structured
  logs and are not included in snapshots.
- `/retrieve` exposes no token usage, so retrieval context tokens are labeled
  estimates based on character count.
- The wrapper does not preserve every raw SSE event or all ask-start metadata and
  does not expose first-event or first-token timing.
- Adapter elapsed time and API-reported latency are recorded separately.
- Filesystem artifacts are the only database; backup and retention are operator
  responsibilities.
- Run live evaluations only against services and data you are authorized to use.
