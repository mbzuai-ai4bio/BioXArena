#!/bin/bash

export PYTHONUNBUFFERED=1
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda)"

set -euo pipefail

PREFIX_DIR="xxx"  # it is the location of BioXArena-Data-Public
MODEL="deepseek/deepseek-v4-pro"
ROUND_NAME="round1"
MAX_WORKERS="1"
TEMPERATURE="0"
STEPS="999"
TIME_LIMIT="7200"
SERVER_ID="112"

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

# MLMaster2.0 + DeepSeek V4 - Part 2 (19 tasks)
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
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --round-name "${ROUND_NAME}"
  --temperature "${TEMPERATURE}"
  --steps "${STEPS}"
  --time-limit "${TIME_LIMIT}"
)

echo "============================================"
echo "MLMaster2.0 + DeepSeek V4 Part 2 (${#TASK_NAMES[@]} tasks, 1 GPU)"
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
