# Evaluation Matrix

`questions-200.json` is a deterministic regression matrix for the supplied 115-row FAQ corpus. It intentionally mixes answerable paraphrases with questions that require facts absent from the corpus, prompt attacks, and unrelated requests. Strict verdicts check result, allowed reason code, verified source identity, answer/reference overlap, required terms, and real generation-provider execution. A technical error cannot pass as `NO_ANSWER`, and repaired or missing model citations cannot pass as verified grounding.
