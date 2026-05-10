#!/bin/bash

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda)"

set -euo pipefail

PREFIX_DIR="xxx"  # it is the location of BioXArena-Data-Public
MODEL="google/gemini-3.1-pro-preview" # model name on OpenRouter
ROUND_NAME="round1"
MAX_WORKERS="1"
TEMPERATURE="0"
STEPS="999"
TIME_LIMIT="7200"            # 2h hard wall-clock per task (subprocess timeout in runner)
SERVER_ID="112"

BASE_URL="https://openrouter.ai/api/v1"   #api cloud platform url

TRAINING_DIR="$(pwd)" # <dir>/BioXArena/training
BioXArena_DIR="${TRAINING_DIR}/.." #<dir>/BioXArena
MLEVOLVE_DIR="${BioXArena_DIR}/agents/MLEvolve" # <dir>/BioXArena/agents/MLEvolve
RUNNER="${TRAINING_DIR}/run_mlevolve_agent.py"

ENV_FILE="${BioXArena_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  API_KEY=$(grep "^api_key=" "${ENV_FILE}" | cut -d'=' -f2-)
else
  API_KEY=""
fi

# MLEvolve + Gemini 3.1 pro — Part 2 (19 tasks)
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
echo "MLEvolve + Gemini 3.1 pro Part 2 (${#TASK_NAMES[@]} tasks, 1 GPU)"
echo "TIME_LIMIT=${TIME_LIMIT}"
echo "============================================"

export DATASET_DIR="${PREFIX_DIR}/BioXArena-Data-Public"
bash "${MLEVOLVE_DIR}/launch_server.sh" "${SERVER_ID}"

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

