# Closed-book knowledge base construction, LM-KBC 2026

Given a subject and a relation, predict the complete set of object entities out
of a language model's parameters alone — no retrieval, no training, no more
than 32B parameters. This is our entry to the [LM-KBC 2026 shared
task](https://github.com/lm-kbc/dataset2026).

| | official macro-F1 |
|---|---|
| this system | **0.6553** |
| prompting baseline, same backbone | 0.6415 |

The checkpoints in `outputs/` are the ones behind that score, and
`src/build_submission.py` rebuilds the submitted predictions from them
exactly. `src/build_baseline.py` rebuilds the baseline the same way, from the
same checkpoints. See [What the machinery
buys](#what-the-machinery-buys) before assuming every stage below earns its
place.

Paper: see `paper/` in the accompanying submission.

## How it works

One backbone,
[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
(29.6B parameters including a 1.8B vision encoder), fits the budget as a
single model, so there is no second network anywhere in the pipeline. Every
decision constant is inlined in the source.

**Elicitation.** The four relations answered by free generation draw k=12
samples at temperature 1 from a few-shot prompt carrying the relation
definition verbatim, a JSON schema, and eight training exemplars. Auxiliary
channels then approach the same fact from other angles, at k=8: recitation, a
commonly-cited-figure frame, a wiki-table frame and a greedy pass at
temperature 0 for the numerics; a local-language frame and a presupposition
frame for city-of-death; a list enumeration for awards; two list passes at
different sampling depths for borders.

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
a per-relation threshold. Borders mixes the two at 0.7/0.3 with the boosted
pass at triple weight and a 0.40 threshold. City-of-death votes over the
direct channel and recitation at double weight, and abstains unless the
deceased-gate probability plus three times the fraction of samples naming a
city clears 1.80. Numerics cluster the pooled values and take the largest
cluster's maximum, at 5% linkage for capacity and 8% for area.
Definition-derived rules finish the job: integral-territory mapping, an award
name-shape filter, and no abstention on numerics or awards.

**Where the constants come from.** The borders fusion, the city vote and
abstention rule, and the capacity linkage and channel weights were fitted on
validation gold, and a candidate replaced the default only when it strictly
improved on validation by more than one row. The city setting was additionally
confirmed by two-fold cross-validation of the fitting procedure. Area and
company kept their defaults because no validation candidate beat them. The
award rule was chosen against Wikidata-derived weak labels, not validation
gold, and is the one place where the tuning signal came from the test
subjects.

## What the machinery buys

The baseline is the same backbone with none of the above attached: the k=12
direct samples, aggregated by self-consistency under rules taken from the task
definition instead of fitted. Numerics take the median of the largest cluster
at the metric's own 5% tolerance, city takes a majority over parsed cities
including the null answer, and the set-valued relations admit an object named
in more than half the samples that parsed.

| relation | prompting | this system | delta |
|---|---|---|---|
| countryLandBordersCountry | 0.9346 | 0.9481 | +0.0135 |
| hasArea | 0.8300 | 0.8600 | +0.0300 |
| companyTradesAtStockExchange | 0.7838 | 0.7838 | 0.0000 |
| personHasCityOfDeath | 0.5400 | 0.5700 | +0.0300 |
| hasCapacity | 0.2653 | **0.2449** | **-0.0204** |
| awardWonBy | 0.0696 | 0.2358 | +0.1662 |
| **all 475 rows** | **0.6415** | **0.6553** | **+0.0138** |

Two rows do not favour the system. **On `hasCapacity` plain prompting wins by
0.0204**: the auxiliary channels and the max-representative rule are net
harmful on the relation this system spent the most effort on. That was not
visible when the configuration was chosen. On validation the system's capacity
setting scores 0.3505 against the baseline's 0.3196, so validation preferred
it and the protocol above kept it. The relation ships as the system built it,
and the gap is recorded rather than papered over. `companyTradesAtStockExchange`
is the second row: it gains exactly 0.0000, because the exchange probes and
the placebo correction recover what the direct channel already said.

```bash
python3 src/build_baseline.py     # outputs/predictions_baseline.jsonl
```

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

`outputs/` ships the checkpoints behind the submitted predictions, and every
stage resumes from existing checkpoints, so the commands above rebuild those
predictions byte-for-byte without touching a GPU.

To elicit from scratch instead, use `scripts/fresh_run.sh`, which writes to an
output root outside the repo:

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

## Sampling budget

Every channel samples at temperature 1, so eliciting from scratch draws
different pools than the released checkpoints, and rows whose cluster mass or
frequency sits near a threshold can fall either way. `AKBC_K_SCALE` scales the
per-channel sample count; the default of 2 is the released configuration
(k=12 direct, k=8 auxiliary), which is the budget the decision constants were
fitted against.

```bash
AKBC_K_SCALE=1 bash scripts/fresh_run.sh   # half the samples, half the GPU time
```

Cutting it is not advisable and raising it further does not help capacity,
where more samples estimate the modal cluster better and the model's modal
value is usually wrong. Greedy (temperature 0) channels are never scaled.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — use freely with
attribution to the authors.
