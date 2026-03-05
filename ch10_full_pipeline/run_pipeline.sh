#!/usr/bin/env bash
set -euo pipefail

QUESTION=${1:-"What is the purpose of the SQuAD dataset?"}
PROFILE=${PROFILE:-cpu_demo}
SOURCE=${SOURCE:-squad}
SPLIT=${SPLIT:-train}
SAMPLE_SIZE=${SAMPLE_SIZE:-5000}

python3 start.py run \
  --profile "$PROFILE" \
  --source "$SOURCE" \
  --split "$SPLIT" \
  --sample-size "$SAMPLE_SIZE" \
  --question "$QUESTION"
