#!/bin/bash
set -e
cd "$(dirname "$0")/.."
cd outputs
zip -j -q submission.zip predictions.jsonl
echo "outputs/submission.zip ready ($(wc -l < predictions.jsonl | tr -d ' ') rows)"
