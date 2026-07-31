# Evaluation Workspaces

**Status: implemented and verified by the M2 offline quality gate.**

`rag-evals` separates installed package code from evaluation configuration and generated results. A workspace has exactly two top-level directories:

```text
workspace/
  config/
  results/
```

Catalogs, definitions, and prompt files live below `config/`. Generated datasets, reports, snapshots, failures, summaries, and trash live below `results/`.

## Selecting a Workspace

The approved interface supports an explicit global option:

```bash
rag-evals --workspace /path/to/workspace validate definitions/retrieval/test.yaml
```

Or an environment default:

```bash
export RAG_EVALS_WORKSPACE=/path/to/workspace
rag-evals validate definitions/retrieval/test.yaml
```

The explicit option takes precedence over `RAG_EVALS_WORKSPACE`. If neither is provided, commands fail with an instruction to configure a workspace. Top-level and command help remain available without one.

Definition arguments are relative to `<workspace>/config`, not the shell working directory. Absolute definition paths and paths escaping `config/` are rejected.

## Results

The framework derives output paths rather than accepting arbitrary destinations:

```text
<workspace>/results/datasets/
  retrieval/
  retrieval_judgments/
  answers/
  answer_judgments/
  .trash/

<workspace>/results/reports/
```

All commands, including `status`, `seal`, and `clean-dataset`, use the same selected workspace.

## Project Workspaces

Jake's real workspace:

```text
bd_evals_config_jake/
  config/
  results/
```

Shipped examples:

```text
bd_evals_config_example/
  config/
    questions.yaml
    knowledge-bases.yaml
    answer-models.yaml
    judge-models.yaml
    prompts/
      retrieval-judge/
        system.md
        user.md
    definitions/
      retrieval-example.yaml
      retrieval-judgment-example.yaml
      report-example.yaml
  results/
    .gitkeep
```

The example workspace uses rejected placeholder identifiers and is safe only for offline validation and planning. It contains no example results.

## Secrets

Workspaces load environment values from `.env` files, searched most specific
first:

1. `<workspace>/.env`
2. `<workspace>/../.env`

Both files are applied without overriding variables already present in the
process environment, so an explicit shell export always wins, and a
workspace-local file wins over one shared between sibling workspaces. No other
location is searched, and a missing file is not an error.

A single shared file in the parent directory therefore serves every sibling
workspace as well as the `bd_api` helper scripts, which load the same file
through `python-dotenv`.

Definitions contain environment-variable names only. Secret values must never
enter configuration snapshots, artifacts, reports, or logs. `.env` files must
stay untracked.
