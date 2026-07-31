Clinician question:
{{ question }}

Retrieved context supplied to the answering model:
{{ retrieval_context }}

Classify each numbered source block above as relevant, partial, or irrelevant to
this question, then score the five criteria in the order given: context_integrity,
signal_density, answerability, clinical_completeness, medical_safety.

Judge only the context above, as if it were the only material available. Score
each criterion independently; a low score on one does not imply a low score on
another. Return the JSON object only, including one source_assessments entry per
source block.
