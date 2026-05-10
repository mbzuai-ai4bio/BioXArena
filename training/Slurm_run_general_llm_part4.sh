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

# General-LLM + GLM-5.1 — Part 4 (19 tasks)
TASK_NAMES=(
  "single-cell/gene-expression-denoising"
  "single-cell/label-projection"
  "single-cell/rna-to-protein-prediction"
  "structure/complex-structure-evaluation"
  "structure/enzyme-commission-prediction"
  "structure/protein-binding-site-detection"
  "structure/protein-fold-classification"
  "structure/protein-ligand-binding-affinity"
  "structure/protein-protein-interface"
  "structure/protein-stability-change"
  "structure/protein-structure-prediction"
  "text-integrated/biomedical-figure-vqa"
  "text-integrated/dna-enzyme-function"
  "text-integrated/ecg-signal-qa"
  "text-integrated/gene-expression-classification"
  "text-integrated/medical-vqa"
  "text-integrated/molecule-qa"
  "text-integrated/pathology-vqa"
  "text-integrated/protein-function-matching"
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
echo "General-LLM + GLM-5.1 Part 4 (${#TASK_NAMES[@]} tasks, 1 GPU)"
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
