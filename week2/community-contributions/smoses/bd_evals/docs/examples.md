# Examples

**Status: Verified examples with implemented commands. Placeholder values
cannot be used for live calls.**

## Retrieval Definition

```yaml
schema_version: 1
test:
  id: retrieval-example
  type: retrieval
questions:
  source: ../questions.yaml
knowledge_bases:
  source: ../knowledge-bases.yaml
selection:
  knowledge_bases:
    include: [kb_placeholder_1, kb_placeholder_2, kb_placeholder_3]
retrieval:
  modes: [standard, rerank]
  candidate_counts: [20]
  result_counts: [5]
deployment:
  label: merge-disabled-example
  merge:
    enabled: false
    maximum_chunks_per_document: null
api:
  base_url_env: BD_API_BASE_URL
  api_key_env: BD_API_KEY
output:
  dataset_id: retrieval-example
execution:
  concurrency: 5
  continue_on_error: true
  retries:
    maximum_attempts: 3
```

**Verified:** This definition validates and plans offline:

```bash
export RAG_EVALS_WORKSPACE="$(cd ../bd_evals_config_example && pwd)"
rag-evals validate definitions/retrieval-example.yaml
rag-evals plan definitions/retrieval-example.yaml
```

Plan output: 2 questions x 3 KBs x 2 modes = 12 calls.

## Frozen Answer Template

```yaml
schema_version: 1
template: |
  Answer the question using only the supplied context.

  Question:
  {{ question }}

  Context:
  {{ retrieval_context }}
```

The framework renders this fully before calling `/ask`.

**Verified:** Template validation rejects unknown variables, statement blocks,
function calls, imports, and unknown filters. Allowed filters: `to_yaml`,
`to_json`, `length`, `default`.

## Example Retrieval Judge Prompt

System instructions:

```text
You are reviewing retrieved context for a question. Explain whether the context is relevant, sufficient, and unnecessarily noisy. Identify missing information. Return concise plain text. This is an example prompt and requires domain review before production use.
```

User message:

```yaml
schema_version: 1
template: |
  Question:
  {{ question }}

  Retrieved context:
  {{ retrieval_context }}

  Optional rubric:
  {{ rubric | to_yaml }}

  Assess retrieval relevance, sufficiency, noise, and important omissions in plain text.
```

**Verified:** `to_yaml` is implemented as a safe framework filter in
`src/rag_evals/templates/__init__.py`. The filter serializes data to YAML with
sorted keys.

## Example Answer Judge Prompt

System instructions:

```text
You are reviewing a candidate answer against its question and supplied context. Explain correctness, completeness, grounding, and unsupported claims. Return concise plain text. This is an example prompt and requires domain review before production use.
```

User message:

```yaml
schema_version: 1
template: |
  Question:
  {{ question }}

  Context supplied to the answer model:
  {{ retrieval_context }}

  Candidate answer:
  {{ candidate_answer }}

  Optional rubric:
  {{ rubric | to_yaml }}

  Assess the answer in plain text and identify unsupported claims explicitly.
```

## Merge-Enabled Variant

Set `deployment.merge.enabled: true`, provide the deployed merge limit, and use
a distinct deployment label and dataset ID. Planning excludes `kb_placeholder_3`
because its no-chunking metadata has `supports_merge_eval: false`.

```yaml
deployment:
  label: merge-enabled-example
  merge:
    enabled: true
    maximum_chunks_per_document: 5
```

**Verified:** Matrix generation excludes KBs with `supports_merge_eval: false`
when merge is enabled. Excluded rows are visible in `matrix.excluded`.

## Config Files

The following files ship in `bd_evals_config_example/config/`:

| File | Contents |
|---|---|
| `config/questions.yaml` | 2 example questions (q001, q002) |
| `config/knowledge-bases.yaml` | 3 placeholder KBs (fixed, semantic, none) |
| `config/answer-models.yaml` | 2 placeholder answer models |
| `config/judge-models.yaml` | 3 placeholder judge models |
| `config/prompts/retrieval-judge/system.md` | Retrieval judge criteria, 1-5 anchored, JSON output |
| `config/prompts/retrieval-judge/user.md` | Retrieval judge user message |
| `config/definitions/retrieval-example.yaml` | Example retrieval definition |
| `config/definitions/retrieval-judgment-example.yaml` | Example retrieval judgment definition |
| `config/definitions/report-example.yaml` | Example report definition with scoring block |

All placeholder values (`KB_REPLACE_ME_*`, `MODEL_REPLACE_ME_*`) validate and
plan offline but are rejected by live execution.

## Retrieval Judgment Definition

```yaml
schema_version: 1
test:
  id: retrieval-judgment-example
  type: retrieval_judgment
source_dataset:
  type: retrieval
  id: retrieval-example
judge_models:
  source: ../judge-models.yaml
selection:
  judge_models:
    include: [judge_placeholder_1]
prompt_variants:
  - id: clinical_v1
    template:
      system_instructions: prompts/retrieval-judge/system.md
      user_message: prompts/retrieval-judge/user.md
inference_variants:
  - id: deterministic
    inference_config:
      maxTokens: 2048
      temperature: 0
api:
  base_url_env: BD_API_BASE_URL
  api_key_env: BD_API_KEY
output:
  dataset_id: retrieval-judgment-example
execution:
  concurrency: 5
  continue_on_error: true
  retries:
    maximum_attempts: 3
```

Calls equal source artifacts x judge models x prompt variants x inference
variants. Template paths resolve relative to `<workspace>/config`. The retrieval
judge template allowlist excludes knowledge base identity and retrieval mode, so
standard and rerank artifacts are judged blind and compared afterwards in the
report by grouping on `retrieval_mode`.

**Verified:** `rag-evals validate definitions/retrieval-judgment-example.yaml`
reports `Status: valid` with placeholder model warnings, and both prompt files
resolve from disk and are content-hashed into the execution bundle.

## Report Definition With Scoring

```yaml
datasets:
  retrieval: retrieval-example
  retrieval_judgments: retrieval-judgment-example
output_dir: latest
scoring:
  retrieval_judgment:
    parser: json
    minimum: 1
    maximum: 5
    criteria:
      - medical_safety
      - answerability
      - clinical_completeness
      - context_integrity
      - signal_density
    comment_field: comments
```

Scores are parsed from immutable raw judge text at report time. See
`cli-reference.md` for column behavior and parse-failure semantics.
