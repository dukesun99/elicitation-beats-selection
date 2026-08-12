#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# sampling + probe stages (GPU; each relation checkpoints to outputs/ and
# every stage resumes from its checkpoints if interrupted)
python3 src/muse_elicit.py --stage sample
python3 src/muse_elicit.py --stage probe
python3 src/muse_elicit.py --stage award
python3 src/muse_channels.py --stage channels
python3 src/muse_channels.py --stage exchange
python3 src/muse_channels.py --stage capterms
python3 src/muse_channels.py --stage borders_freq
python3 src/muse_channels.py --stage boost
# deterministic fusion + decision rules (CPU)
python3 src/build_submission.py
