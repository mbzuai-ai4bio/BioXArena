#!/bin/bash
#SBATCH --job-name=bioxarena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#
# Run one part (19 tasks) of the benchmark for one agent/backbone on 1 GPU, as in the paper.
#
# Usage (from BioXArena/training):
#   sbatch Slurm_run_agent.sh <agent> <openrouter-model-id> <part 1-4>
#   sbatch Slurm_run_agent.sh claudecode anthropic/claude-opus-4.6 1
# or reproduce a paper configuration (see `python run_agent.py --list-configs`):
#   sbatch Slurm_run_agent.sh --config codex_g6a 1
#
# The OpenRouter key is read from OPENROUTER_API_KEY or from BioXArena/.env (OPENROUTER_API_KEY=...).

set -euo pipefail
export PYTHONUNBUFFERED=1

PREFIX_DIR="xxx"   # directory that contains the extracted BioXArena-Data-Public/
ROUND_NAME="round1"
TIME_LIMIT="7200"  # seconds per task

if [[ "${1:-}" == "--config" ]]; then
  SELECT=(--config "$2"); PART="$3"
else
  SELECT=(--agent "$1" --model "$2"); PART="$3"
fi

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Version:', torch.version.cuda)"

python run_agent.py "${SELECT[@]}" \
  --data-root "${PREFIX_DIR}/BioXArena-Data-Public" \
  --output-root "${PREFIX_DIR}/BioXArena-Output" \
  --round-name "${ROUND_NAME}" \
  --time-limit "${TIME_LIMIT}" \
  --part "${PART}"
