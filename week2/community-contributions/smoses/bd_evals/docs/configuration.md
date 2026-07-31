# Configuration

**Status: Implemented and verified. Schemas, resolution, and validation are
covered by the current offline test suite.**

## Workspace Configuration

```text
<workspace>/config/
  questions.yaml
  knowledge-bases.yaml
  answer-models.yaml
  judge-models.yaml
  definitions/
```

YAML is loaded safely with `yaml.safe_load`. Every file has `schema_version: 1`.
IDs must be unique. Unknown selections, invalid schemas, and ambiguous
include/exclude rules fail before API calls.

## Source References

Reusable content is referenced with:

```yaml
questions:
  source: ../questions.yaml
```

The path is resolved relative to the YAML file containing `source`, never the
shell working directory. References must remain within the selected workspace's
`config/` boundary. Missing files, circular references, absolute paths, symlink
escapes, and path traversal fail validation. Inputs are hashed and snapshotted
without secret values.

**Verified:** `SourceResolver` in `src/rag_evals/config/resolver.py` implements
containing-file-relative resolution, cycle detection, boundary checks, and
SHA-256 content hashing.

## Questions

```yaml
schema_version: 1
questions:
  - id: q001
    question: What are the diagnostic criteria for major depressive disorder?
  - id: q002
    question: What is the first-line treatment for anaphylaxis?
    rubric:
      expected_answer: The answer should identify intramuscular epinephrine.
      required_facts:
        - State that epinephrine is first-line.
```

Rubrics are optional and extensible (model uses `extra="allow"`). Known fields
are `expected_answer`, `required_facts`, `expected_sources`,
`unacceptable_claims`, and `notes`. No score fields are predefined.

The provided question JSON source contains 91 questions and is represented in a
workspace YAML catalog as `q001` through `q091`, preserving order and exact text.

**Verified:** `QuestionCatalog` and `Question` in
`src/rag_evals/config/schemas.py`. Tests in `tests/test_schemas.py`.

## Knowledge Bases

Initial examples use non-runnable placeholders:

```yaml
schema_version: 1
knowledge_bases:
  - id: kb_placeholder_1
    knowledge_base_id: KB_REPLACE_ME_1
    description: Example fixed-size Knowledge Base.
    chunking:
      strategy: fixed
      size: 500
      overlap: 100
    supports_merge_eval: true
  - id: kb_placeholder_2
    knowledge_base_id: KB_REPLACE_ME_2
    description: Example semantic Knowledge Base.
    chunking:
      strategy: semantic
    supports_merge_eval: true
  - id: kb_placeholder_3
    knowledge_base_id: KB_REPLACE_ME_3
    description: Example no-chunking Knowledge Base.
    chunking:
      strategy: none
    supports_merge_eval: false
```

Placeholder values may validate and plan offline but are rejected by live
execution.

Chunking fields are descriptive provenance recorded with every artifact; they are
never sent to the API. For fixed-size chunking, `overlap` is a percentage. Semantic
chunking additionally accepts `similarity_percentile_threshold` (1-99), which is
rejected for other strategies. Unknown chunking fields remain a validation error.

**Verified:** `KnowledgeBaseCatalog` and `KnowledgeBase` in
`src/rag_evals/config/schemas.py`. The `is_placeholder` property detects
`KB_REPLACE_ME_*` prefixes. Catalog `id` values must be unique, and two entries
may not declare the same non-placeholder `knowledge_base_id`, because that would
compare a Knowledge Base against itself. Repeated placeholders remain allowed for
offline examples. Tests in `tests/test_schemas.py`.

## Models

Answer and judge catalogs remain separate. Examples use `judge_placeholder_1`,
`judge_placeholder_2`, and `judge_placeholder_3` as obvious non-production
entries. Answer placeholders (`answer_placeholder_1`, `answer_placeholder_2`)
are in a separate catalog.

```yaml
schema_version: 1
models:
  - id: judge_placeholder_1
    model_id: MODEL_REPLACE_ME_1
    description: Non-runnable example judge model.
    default_inference_config:
      maxTokens: 2048
      temperature: 0
```

`InferenceConfig` allows extension fields (`extra="allow"`) for model-specific
parameters beyond `maxTokens` and `temperature`.

Tests may select one or multiple catalog entries and override defaults. Live
calls reject unresolved model placeholders.

**Verified:** `ModelCatalog` and `ModelEntry` in
`src/rag_evals/config/schemas.py`.

## Deployment Metadata

Merge state is declared, not requested from the API:

```yaml
deployment:
  label: merge-disabled-example
  merge:
    enabled: false
    maximum_chunks_per_document: null
```

No-chunking KBs are excluded automatically from merge-enabled plans. API
deployment is never automated.

**Verified:** `DeploymentConfig` and `MergeConfig` enforce that
`maximum_chunks_per_document` is required when `enabled: true` and must be `null`
when `enabled: false`.

## API and Secrets

Definitions store environment variable names only:

```yaml
api:
  base_url_env: BD_API_BASE_URL
  api_key_env: BD_API_KEY
```

Environment variable names must be uppercase letters, digits, and underscores.
Environment values must never enter snapshots, logs, manifests, or artifacts.

Values are read from the process environment, which is populated from
`<workspace>/.env` and `<workspace>/../.env` if present. Shell exports take
precedence. See `workspaces.md`.

**Verified:** `ApiConfig` validates the env var name pattern and rejects extra
fields. `redact()` in `src/rag_evals/logging.py` recursively redacts values
whose keys match `key`, `token`, `secret`, `password`, or `credential`.

## Templates

Every ask-based test owns `system_instructions` and `user_message` template
content. Rendering uses sandboxed Jinja2 with `StrictUndefined`. Unknown
variables and unsafe expressions fail validation.

Supported variables by stage:

- **Retrieval judge:** `question`, `question_id`, `rubric`, `retrieval_context`,
  `retrieved_items`, `retrieval_metadata`
- **Answer generation:** `question`, `question_id`, `rubric`, `retrieval_context`
- **Answer judge:** `question`, `question_id`, `rubric`, `retrieval_context`,
  `retrieved_items`, `retrieval_metadata`, `candidate_answer`

Allowed filters: `to_yaml`, `to_json`, `length`, `default`.

Statement blocks (`{% %}`), function calls, imports, includes, macros, blocks,
and template inheritance are rejected.

**Verified:** `src/rag_evals/templates/__init__.py`. Tests in
`tests/test_templates.py` (27 tests).

## Definition Types

Five definition types are supported, dispatched by `test.type`:

| Type | Schema | Purpose |
|---|---|---|
| `retrieval` | `RetrievalDefinition` | Build retrieval dataset via `/retrieve` |
| `retrieval_judgment` | `RetrievalJudgmentDefinition` | Judge retrieval via `/ask` |
| `answer` | `AnswerDefinition` | Generate frozen-context answers via `/ask` |
| `answer_judgment` | `AnswerJudgmentDefinition` | Judge answers via `/ask` |
| `compare` | `CompareDefinition` | Compare multiple datasets |

All definitions require `schema_version: 1` and a `test` block with `id` and
`type`. Ask-based definitions require named `prompt_variants` and
`inference_variants` (no hidden Cartesian expansion). Dataset IDs must match
`[a-z0-9][a-z0-9-]{0,62}`.

Definition arguments passed to the CLI are relative to `<workspace>/config`.
For example, `definitions/retrieval-example.yaml` resolves to
`<workspace>/config/definitions/retrieval-example.yaml`.

**Verified:** `parse_definition()` in `src/rag_evals/config/schemas.py` dispatches
by `test.type` and raises `ConfigError` on validation failure.
