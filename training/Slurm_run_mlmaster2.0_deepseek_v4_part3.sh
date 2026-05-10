#!/bin/bash

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda)"

set -euo pipefail

PREFIX_DIR="xxx"  # it is the location of BioXArena-Data-Public
MODEL="deepseek/deepseek-v4-pro"
ROUND_NAME="round1"
MAX_WORKERS="1"
TEMPERATURE="0"
STEPS="999"
TIME_LIMIT="7200"
SERVER_ID="113"

BASE_URL="https://openrouter.ai/api/v1"

TRAINING_DIR="$(pwd)"
BioXArena_DIR="${TRAINING_DIR}/.."
EVOMASTER_DIR="${BioXArena_DIR}/agents/EvoMaster"
GRADING_LAUNCHER="${BioXArena_DIR}/agents/MLEvolve/launch_server.sh"
RUNNER="${TRAINING_DIR}/run_mlmaster2.0_agent.py"

ENV_FILE="${BioXArena_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  API_KEY=$(grep "^api_key=" "${ENV_FILE}" | cut -d'=' -f2-)
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  API_KEY=""
fi

# MLMaster2.0 + DeepSeek V4 - Part 3 (19 tasks)
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
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --round-name "${ROUND_NAME}"
  --temperature "${TEMPERATURE}"
  --steps "${STEPS}"
  --time-limit "${TIME_LIMIT}"
)

echo "============================================"
echo "MLMaster2.0 + DeepSeek V4 Part 3 (${#TASK_NAMES[@]} tasks, 1 GPU)"
echo "TIME_LIMIT=${TIME_LIMIT}"
echo "============================================"

export DATASET_DIR="${PREFIX_DIR}/BioXArena-Data-Public"
bash "${GRADING_LAUNCHER}" "${SERVER_ID}"

BASE_PORT=5005
GRADING_SERVER_PORT=$((BASE_PORT + SERVER_ID))
export GRADING_SERVER_PORT

echo "Waiting for grading server on port ${GRADING_SERVER_PORT} ..."
MAX_WAIT=30
WAITED=0
while [ "${WAITED}" -lt "${MAX_WAIT}" ]; do
  if curl -s "http://127.0.0.1:${GRADING_SERVER_PORT}/health" > /dev/null 2>&1; then
    echo "Grading server ready (port ${GRADING_SERVER_PORT})."
    break
  fi
  sleep 1
  WAITED=$((WAITED + 1))
done
if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
  echo "Warning: grading server may not be ready yet, proceeding anyway ..."
fi

TASK_ARGS=()
for task_name in "${TASK_NAMES[@]}"; do
  TASK_ARGS+=(--task "${task_name}")
done

python "${RUNNER}" "${COMMON_ARGS[@]}" "${TASK_ARGS[@]}" --max-workers "${MAX_WORKERS}"
