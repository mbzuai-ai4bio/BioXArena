#!/bin/bash


#conda activate bioxbench
# cd BioXArena/evaluation

set -euo pipefail

PREFIX_DIR="xxx"
RUNNER="evaluate_llm_agents.py"

echo "============================================"
echo "Evaluate: z-ai__glm-5.1 / round1 / round1"
echo "============================================"

python "${RUNNER}" \
  --prefix-dir "${PREFIX_DIR}" \
  --model "z-ai/glm-5.1" \
  --round-name "round1" \
  --all-tasks \
  --max-workers 4
