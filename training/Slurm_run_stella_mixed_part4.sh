#!/bin/bash

export PYTHONUNBUFFERED=1
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda)"

set -euo pipefail

PREFIX_DIR="xxx"  # it is the location of BioXArena-Data-Public
# Dev Agent + Tool Creation Agent
MODEL="anthropic/claude-sonnet-4.6"
# Manager Agent + Critic Agent
MANAGER_MODEL="google/gemini-3.1-pro-preview"
ROUND_NAME="round1"
MAX_WORKERS="1"
TEMPERATURE="0"
TIME_LIMIT="7200"
BASE_URL="https://openrouter.ai/api/v1"

TRAINING_DIR="$(pwd)"
BioXArena_DIR="${TRAINING_DIR}/.."
RUNNER="${TRAINING_DIR}/run_stella_agent.py"

ENV_FILE="${BioXArena_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  API_KEY=$(grep "^api_key=" "${ENV_FILE}" | cut -d'=' -f2)
else
  API_KEY=""
fi

# STELLA mixed-LLM: Dev+Tool=Claude-Sonnet-4.6, Manager+Critic=Gemini-3.1-Pro-Preview
# Part 4 (19 tasks)
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
  --manager-model "${MANAGER_MODEL}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --round-name "${ROUND_NAME}"
  --temperature "${TEMPERATURE}"
  --time-limit "${TIME_LIMIT}"
)

echo "============================================"
echo "STELLA Mixed-LLM Part 4 (${#TASK_NAMES[@]} tasks)"
echo "Dev+Tool: ${MODEL}"
echo "Manager+Critic: ${MANAGER_MODEL}"
echo "TIME_LIMIT=${TIME_LIMIT}"
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
