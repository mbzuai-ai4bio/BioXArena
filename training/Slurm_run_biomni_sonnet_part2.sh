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

# Biomni + Claude-Sonnet-4.6 — Part 2 (19 tasks)
TASK_NAMES=(
  "network-biology/pathway-membership-reactome"
  "network-biology/ppi-prediction-string"
  "network-biology/protein-complex-corum"
  "network-biology/synthetic-lethality-prediction"
  "network-biology/tf-regulatory-prediction"
  "perturbation-dynamics/cancer-drug-sensitivity"
  "perturbation-dynamics/crispr-perturbation-prediction"
  "perturbation-dynamics/drug-transcriptional-response"
  "perturbation-dynamics/eccite-multimodal-perturbation"
  "perturbation-dynamics/gene-regulatory-network-inference"
  "perturbation-dynamics/multi-timepoint-perturbation"
  "perturbation-dynamics/rna-velocity-cell-transition"
  "perturbation-dynamics/spear-atac-perturbation"
  "phenotype-disease/alzheimers-disease-staging"
  "phenotype-disease/autism-diagnosis"
  "phenotype-disease/breast-cancer-subtype"
  "phenotype-disease/covid19-severity-classification"
  "phenotype-disease/diabetes-readmission"
  "phenotype-disease/genotype-to-phenotype"
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
echo "Biomni + Claude-Sonnet-4.6 Part 2 (${#TASK_NAMES[@]} tasks, 1 GPU)"
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
