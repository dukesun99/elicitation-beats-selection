#!/bin/bash
# From-scratch re-elicitation. The repo ships the checkpoints behind the
# reported score, and every stage resumes from whatever is in outputs/, so a
# genuine fresh run needs an output root that holds nothing - and one that
# git cannot repopulate. This uses a directory outside the worktree for both
# reasons; pointing AKBC_ROOT at the worktree would let any later git
# operation restore the shipped checkpoints mid-run, after which the stages
# log "cached" and the build re-emits the original predictions.
set -e
RUN_ROOT="${RUN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/../akbc-fresh-run}"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$RUN_ROOT/outputs"
ln -sfn "$REPO/dataset2026" "$RUN_ROOT/dataset2026"
export AKBC_ROOT="$RUN_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 "$SRC/muse_elicit.py"   --stage sample
python3 "$SRC/muse_elicit.py"   --stage probe
python3 "$SRC/muse_elicit.py"   --stage award
python3 "$SRC/muse_channels.py" --stage channels
python3 "$SRC/muse_channels.py" --stage exchange
python3 "$SRC/muse_channels.py" --stage capterms
python3 "$SRC/muse_channels.py" --stage borders_freq
python3 "$SRC/muse_channels.py" --stage boost
python3 "$SRC/build_submission.py"
cd "$RUN_ROOT/outputs" && zip -j -q submission.zip predictions.jsonl
echo "fresh run complete: $RUN_ROOT/outputs/submission.zip"
