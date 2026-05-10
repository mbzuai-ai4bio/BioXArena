#!/bin/bash

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda)"

set -euo pipefail

PREFIX_DIR="xxx"
MODEL="anthropic/claude-sonnet-4"
SOURCE="Custom"
BASE_URL="https://openrouter.ai/api/v1"
ROUND_NAME="round1"
MAX_WORKERS="1"
TEMPERATURE="0"
TIMEOUT_SECONDS="7200"
TASK_WALL_CLOCK_SEC="7200"

TRAINING_DIR="$(pwd)"
BioXArena_DIR="${TRAINING_DIR}/.."
RUNNER="${TRAINING_DIR}/run_biomni_agent.py"

ENV_FILE="${BioXArena_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  API_KEY=$(grep "^api_key=" "${ENV_FILE}" | cut -d'=' -f2)
else
  API_KEY=""
fi

# Biomni + Claude-Sonnet-4.6 — Part 1 (19 tasks)
TASK_NAMES=(
  "chemical-biology/bace1-binding-affinity"
  "chemical-biology/cell-painting-perturbation"
  "chemical-biology/cyp-inhibition-multi-label"
  "chemical-biology/egfr-binding-affinity"
  "chemical-biology/gpcr-binding-multi-class"
  "chemical-biology/herg-binding-affinity"
  "chemical-biology/kinase-selectivity-multi-label"
  "chemical-biology/tox21-sr-are"
  "imaging/amos-organ-segmentation"
  "imaging/drug-moa-prediction"
  "imaging/labelfree-cell-counting"
  "imaging/lung-nodule-malignancy"
  "imaging/mitochondria-counting"
  "imaging/nucleus-type-classification"
  "imaging/skin-lesion-diagnosis"
  "imaging/virtual-staining"
  "network-biology/gene-disease-association"
  "network-biology/go-function-multi-label"
  "network-biology/metabolic-network-kegg"
)

COMMON_ARGS=(
  --prefix-dir "${PREFIX_DIR}"
  --model "${MODEL}"
  --source "${SOURCE}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --round-name "${ROUND_NAME}"
  --temperature "${TEMPERATURE}"
  --timeout-seconds "${TIMEOUT_SECONDS}"
  --task-wall-clock-sec "${TASK_WALL_CLOCK_SEC}"
)

echo "============================================"
echo "Biomni + Claude-Sonnet-4.6 Part 1 (${#TASK_NAMES[@]} tasks, 1 GPU)"
echo "TASK_WALL_CLOCK_SEC=${TASK_WALL_CLOCK_SEC}"
echo "============================================"

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 1
fi

TASK_ARGS=()
for task_name in "${TASK_NAMES[@]}"; do
  TASK_ARGS+=(--task "${task_name}")
done

python "${RUNNER}" "${COMMON_ARGS[@]}" "${TASK_ARGS[@]}" --max-workers "${MAX_WORKERS}"
