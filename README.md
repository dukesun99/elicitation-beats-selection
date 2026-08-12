# Elicitation Beats Selection

Closed-book knowledge base construction for the [LM-KBC 2026 shared
task](https://github.com/lm-kbc/dataset2026): given a subject and a relation,
predict the complete set of object entities from a language model's parameters
alone — no retrieval, no training, no more than 32B parameters.

**Official test macro-F1: 0.6743** · Task baseline: 0.313

The system runs a single backbone,
[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
(29.6B parameters), with multi-channel elicitation, one-token verification
probes, and a per-relation decision layer. Every decision constant is inlined
in the source.

Paper: *Elicitation Beats Selection: Closed-Book Knowledge Base Construction
from Language Models*.

## How it works

**Elicitation.** Per subject, several prompt channels sample the model with
few-shot self-consistency: direct question-answering, recitation
(recall-then-answer), a presupposition frame and a local-language frame for
city-of-death, commonly-cited and wiki-table frames for the numerics, and
list-plus-boost passes for awards and borders. Each channel targets a
different route to the same fact.

Muse answers in two channels, reasoning first and the user-facing answer
second. Sampling keeps the reasoning, which is where the model's factual
advantage lives. Probing appends the answer-channel header to the prompt so
the model answers immediately — without this, one-token Yes/No probing reads
the first token of deliberation instead of the verdict.

**Verification.** Where the answer space is enumerable, open-ended generation
is replaced by exhaustive one-token Yes/No probing: an alive/deceased gate for
people, bidirectional land-border probes over a model-generated country
universe (~26k probes), and exchange-listing probes corrected for yes-bias
against known-unlisted companies.

**Decision.** String relations fuse sampling frequency with probe scores under
per-relation thresholds. City-of-death abstains unless the deceased gate and
the fraction of samples naming a city clear a rule fitted on validation gold.
Numerics use 8%-linkage cluster voting with a capacity-term PMI tiebreak and
an overshoot correction. Deterministic rules finish the job:
integral-territory mapping, an award name-shape filter, and never abstaining
on numerics or awards.

## Requirements

- 1×80GB GPU, or 2×32GB GPUs (explicit layer split, built in);
  ~60GB disk for weights
- `pip install torch transformers accelerate` — transformers ≥ 5.15.0 for the
  `muse_glimmer` architecture (CUDA build for your GPU)
- Task data: `git clone https://github.com/lm-kbc/dataset2026` into the repo
  root

## Run

```bash
bash scripts/run_inference.sh    # elicit -> probe -> fuse, outputs/predictions.jsonl
bash scripts/make_zip.sh         # outputs/submission.zip
```

`outputs/` ships the checkpoints behind the official score, and every stage
resumes from existing checkpoints, so the commands above rebuild the scored
predictions without touching a GPU. For a from-scratch reproduction, delete
`outputs/raw_*.json` first.

The sampling and probe stages are the bulk of the compute — channels for all
six relations plus roughly 26k border probes and 6k exchange probes, several
hours on one A100 80GB. `build_submission.py` is
CPU-only and deterministic given the checkpoints.

To evaluate on validation, set `MUSE_SPLIT=val` for the elicitation stages,
then score with the official evaluator:

```bash
python3 dataset2026/evaluate.py -p outputs/predictions.jsonl -g dataset2026/data/val.jsonl
```

## Reproducibility

Sampling channels use temperature sampling, so regenerated predictions are not
bit-identical run to run; macro-F1 reproduces within about one point.
Everything downstream of the sampling checkpoints — fusion, thresholds,
decision rules — is deterministic: rebuilding from the shipped checkpoints
reproduces the officially scored predictions exactly.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — use freely with
attribution to the authors.
