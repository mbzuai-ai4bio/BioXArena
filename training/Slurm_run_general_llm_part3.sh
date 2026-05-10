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

# General-LLM + GLM-5.1 — Part 3 (19 tasks)
TASK_NAMES=(
  "phenotype-disease/pan-cancer-survival-prediction"
  "phenotype-disease/spatial-immune-infiltration"
  "sequence/gene-tissue-expression"
  "sequence/isoform-expression"
  "sequence/multi-tf-binding"
  "sequence/protein-protein-interaction"
  "sequence/regulatory-element-detection"
  "sequence/remote-homology-detection"
  "sequence/rna-protein-binding-affinity"
  "sequence/rna-protein-binding-signal"
  "sequence/rna-reactivity-imputation"
  "sequence/variant-effect-pathogenicity"
  "single-cell/batch-integration"
  "single-cell/cell-type-from-expression"
  "single-cell/chromatin-to-expression"
  "single-cell/cite-seq-protein-prediction"
  "single-cell/cross-modality-cell-matching"
  "single-cell/cross-modality-cell-type"
  "single-cell/developmental-stage-prediction"
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
echo "General-LLM + GLM-5.1 Part 3 (${#TASK_NAMES[@]} tasks, 1 GPU)"
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
