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

# General-LLM + GLM-5.1 — Part 2 (19 tasks)
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
  --round-name "${ROUND_NAME}"
  --max-attempts "${MAX_ATTEMPTS}"
  --request-timeout-sec "${REQUEST_TIMEOUT_SEC}"
  --run-timeout-sec "${RUN_TIMEOUT_SEC}"
  --task-timeout-sec "${TASK_TIMEOUT_SEC}"
  --temperature "${TEMPERATURE}"
)

echo "============================================"
echo "General-LLM + GLM-5.1 Part 2 (${#TASK_NAMES[@]} tasks, 1 GPU)"
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
