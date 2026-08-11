#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/predict.py --stage mistral
python3 src/predict.py --stage qwen4b
python3 src/predict.py --stage finalize
