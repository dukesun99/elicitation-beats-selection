# Elicitation Beats Selection

Closed-book knowledge base construction for the [LM-KBC 2026 shared
task](https://github.com/lm-kbc/dataset2026): given a subject and a relation,
predict the complete set of object entities from a language model's parameters
alone — no retrieval, no training, no more than 32B parameters.

**Official test macro-F1: 0.6273** · Validation: 0.658 · Task baseline: 0.313

The system pairs a Mistral-Small-3.2-24B backbone (plus a Qwen3-4B agreement
term for `hasArea`, 28B total) with multi-channel elicitation, one-token
verification probes, and a calibrated decision layer. Every decision constant
is inlined in `src/predict.py`.

Paper: *Elicitation Beats Selection: Closed-Book Knowledge Base Construction
from Language Models*.

## How it works

**Elicitation.** Per subject, several prompt channels sample the model with
few-shot self-consistency: direct question-answering, recitation
(recall-then-answer), a presupposition frame for city-of-death, wiki-table
formatting and type disambiguation for numerics, and per-year enumeration
(1950–2026) with exclusion rounds for awards. Each channel targets a different
route to the same fact.

**Verification.** Where the answer space is enumerable, open-ended generation
is replaced by exhaustive one-token Yes/No probing: an alive/deceased gate for
people, bidirectional land-border probes over a model-generated country
universe (~26k probes), and exchange-listing probes corrected for yes-bias
against known-unlisted companies.

**Decision.** String relations fuse sampling frequency with probe scores under
per-relation thresholds. Numerics use 5%-linkage cluster voting with duel and
PMI terms. Deterministic rules finish the job: integral-territory mapping,
an award first-year filter, and never abstaining on numerics or awards.

## Requirements

- 2×32GB GPUs (sharded) or 1×80GB; ~50GB disk for weights
- `pip install torch transformers accelerate` (CUDA build for your GPU)
- Task data: `git clone https://github.com/lm-kbc/dataset2026` into the repo root

## Run

```bash
bash scripts/run_inference.sh    # stages: mistral -> qwen4b -> finalize
bash scripts/make_zip.sh         # outputs/submission.zip
```

The `mistral` stage is the bulk of the compute — sampling channels for all six
relations plus roughly 26k border probes and 10k exchange probes, several hours
on 2×RTX 5090. Each relation checkpoints to `outputs/<relation>.json` and the
stage resumes from checkpoints if interrupted. The `qwen4b` stage takes
minutes. The `finalize` stage is CPU-only and deterministic given the
checkpoints.

To evaluate on validation, point `TEST` at `val.jsonl` in `src/predict.py`,
rerun, and score with the official evaluator:

```bash
python3 dataset2026/evaluate.py -p outputs/predictions.jsonl -g dataset2026/data/val.jsonl
```

## Reproducibility

Sampling channels use temperature sampling, so regenerated predictions are not
bit-identical run to run; macro-F1 reproduces within about one point.
Everything downstream of the sampling checkpoints — fusion, thresholds,
decision rules — is deterministic. Running this code from scratch on fresh
hardware and submitting the result independently produced the 0.6273 official
score reported above.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — use freely with
attribution to the authors.
