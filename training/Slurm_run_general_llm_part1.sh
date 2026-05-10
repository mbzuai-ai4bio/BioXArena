#!/bin/bash

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda)"

set -euo pipefail

PREFIX_DIR="xxx"
MODEL="z-ai/glm-5.1"
ROUND_NAME="round1"
MAX_WORKERS="1"
MAX_ATTEMPTS="3"
REQUEST_TIMEOUT_SEC="180"
RUN_TIMEOUT_SEC="7200"
TASK_TIMEOUT_SEC="7200"
TEMPERATURE="0"

TRAINING_DIR="$(pwd)"
RUNNER="${TRAINING_DIR}/run_general_llm_agents.py"

# General-LLM + GLM-5.1 — Part 1 (19 tasks)
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
  --round-name "${ROUND_NAME}"
  --max-attempts "${MAX_ATTEMPTS}"
  --request-timeout-sec "${REQUEST_TIMEOUT_SEC}"
  --run-timeout-sec "${RUN_TIMEOUT_SEC}"
  --task-timeout-sec "${TASK_TIMEOUT_SEC}"
  --temperature "${TEMPERATURE}"
)

echo "============================================"
echo "General-LLM + GLM-5.1 Part 1 (${#TASK_NAMES[@]} tasks, 1 GPU)"
echo "TASK_TIMEOUT_SEC=${TASK_TIMEOUT_SEC}"
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
