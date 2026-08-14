# Closed-book knowledge base construction, LM-KBC 2026

Given a subject and a relation, predict the complete set of object entities out
of a language model's parameters alone — no retrieval, no training, no more
than 32B parameters. This is our entry to the [LM-KBC 2026 shared
task](https://github.com/lm-kbc/dataset2026).

| | official macro-F1 |
|---|---|
| shipped artifact (the checkpoints in `outputs/`) | **0.6743** |
| best from-scratch re-elicitation | **0.6553** |

**Read the second number before you use the first.** The shipped artifact is
what the leaderboard scored. 0.6553 is the best score reached by regenerating
every pool from nothing. The 0.0190 difference is not sampling luck, and the
paper says where it went; a short version is under
[Reproduction](#reproduction) below.

Paper: *Fitting the Draw: Closed-Book Knowledge Base Construction and a
Reproduction Study of Our Own System*.

## How it works

One backbone,
[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
(29.6B parameters including a 1.8B vision encoder), fits the budget as a
single model, so there is no second network anywhere in the pipeline. Every
decision constant is inlined in the source.

**Elicitation.** The four relations answered by free generation draw k=6
samples at temperature 1 from a few-shot prompt carrying the relation
definition verbatim, a JSON schema, and eight training exemplars. Auxiliary
channels then approach the same fact from other angles: recitation, a
commonly-cited-figure frame, a wiki-table frame and a greedy pass for the
numerics; an obituary frame, a local-language frame and a presupposition frame
for city-of-death; a completeness-framed list plus per-year enumeration for
awards; two list passes for borders.

Muse writes two channels, reasoning first and the user-facing answer second.
Sampling keeps the reasoning, which is where the model's factual advantage
lives. Probing appends the answer-channel header so the first generated token
is the verdict. Without that header, one-token Yes/No probing on this backbone
returns noise: on the alive-or-dead gate the forced probe gives Albert
Einstein 0.777 and Taylor Swift 0.002, while the unforced port gives 0.547 and
0.665 and ranks the living person as the more likely dead one.

**Verification.** Where the answer space can be enumerated, exhaustive
one-token probing replaces open-ended generation. The model writes the
universe first (the world's sovereign states, the world's stock exchanges),
then borders probe every (subject, country) pair in both directions, about 26k
probes, and company probes every (subject, exchange) pair with the yes-bias
removed in log-odds against placebo companies known to be unlisted.

**Decision.** String relations fuse sampling frequency with probe score under
a per-relation threshold. City-of-death abstains unless twice the
deceased-gate probability plus twice the fraction of samples naming a city
clears 1.45. Numerics cluster the pooled values at 8% linkage and take the
largest cluster's maximum, with a PMI term and an overshoot correction on
capacity. Definition-derived rules finish the job: integral-territory mapping,
an award name-shape filter, and no abstention on numerics or awards.

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

`outputs/` ships the checkpoints behind the 0.6743 submission, and every stage
resumes from existing checkpoints, so the commands above rebuild the scored
predictions byte-for-byte without touching a GPU.

For a genuine from-scratch re-elicitation use `scripts/fresh_run.sh`, which
writes to an output root outside the repo:

```bash
bash scripts/fresh_run.sh                       # ../akbc-fresh-run/outputs/
RUN_ROOT=/scratch/mine bash scripts/fresh_run.sh
```

**Do not** delete `outputs/raw_*.json` and re-run in place instead. Those
files are tracked, so any later `git pull`, `git stash`, or `git checkout`
restores them mid-run, after which every stage logs `cached` and the build
re-emits the original predictions. That failure mode reports success. We hit
it, and what caught it was an internal contradiction rather than any log:
temperature-1 resampling cannot return 67 of 67 identical border rows.

The sampling and probe stages are the bulk of the compute — channels for all
six relations plus roughly 26k border probes and 6k exchange probes, several
hours on one A100 80GB. `build_submission.py` is CPU-only and deterministic
given the checkpoints.

To evaluate on validation, set `MUSE_SPLIT=val` for the elicitation stages,
then score with the official evaluator:

```bash
python3 dataset2026/evaluate.py -p outputs/predictions.jsonl -g dataset2026/data/val.jsonl
```

## Reproduction

Everything downstream of the sampling checkpoints is deterministic, so
rebuilding from the shipped checkpoints reproduces the scored predictions
exactly. Regenerating the pools does not reproduce the score. Four independent
from-scratch runs scored 0.6495 (k=6) and 0.6457 (k=12) with the constants
inlined in this repository, 0.6511 with those constants refitted on
validation gold, and 0.6553 once the city rule had been cross-validated as
well. The code here carries the released constants, so a clean
`scripts/fresh_run.sh` lands near 0.6495; the refit that reaches 0.6553 is
described in the paper and uses validation gold only.

The 0.0190 shortfall is about three times the whole-metric per-draw standard
deviation of 0.0063, so it is not draw luck. It sits mostly in
`hasCapacity`, which carries 0.0147 of it. The reason is that the numeric
decision constants were selected against Wikidata-derived weak labels, and
those labels are exact for `hasCapacity` and `hasArea` and inexact everywhere
else — so for those two relations the tuning signal came from the test
subjects, and it was applied on one specific draw of a stochastic pipeline.
Bootstrapping fresh pools puts `hasArea`, which has no fitted numeric
constants, 1.0 standard deviation above the distribution a fresh run draws
from, and `hasCapacity`, which has two, 2.7 above it. The capacity PMI term on
its own is worth +0.041 on the pool it was fitted to and −0.010 on a fresh
pool.

Refitting on validation gold recovers part of the rest. Borders closes to
within 0.0003 of the shipped score and city matches it exactly, both from
validation data alone, and capacity gains +0.031 by *removing* fitted
structure (PMI weight to zero, channel weights rebalanced). Capacity still
ends 0.0147 short, and since the other five relations hold only 0.0042 of
remaining gap between them, no from-scratch run of this system exceeds 0.6595
without moving capacity.

## Sampling budget

Every channel samples at temperature 1, so a from-scratch re-elicitation draws
different pools than the released checkpoints, and rows whose cluster mass or
frequency sits near a frozen threshold can flip either way. `AKBC_K_SCALE`
draws more samples per channel:

```bash
AKBC_K_SCALE=2 bash scripts/fresh_run.sh   # 2x samples, ~2x GPU time
```

This does not close the gap. The k=12 run scored 0.6457 against 0.6495 for
k=6, because more samples estimate the modal cluster better and for capacity
the mode is usually wrong. The default of 1 reproduces the released
configuration. Greedy (temperature 0) channels are never scaled.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — use freely with
attribution to the authors.
