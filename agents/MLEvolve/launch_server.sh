#!/bin/bash
# Spawn a background grading server that validates submission format.
# Accepts an optional SERVER_ID argument (default 111) to derive the port.
set -x

SERVER_ID=${1:-111}
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

dataset_dir=${DATASET_DIR:?Please set DATASET_DIR env var to the mle-bench data root}

BASE_PORT=5005
PORT=$((BASE_PORT + SERVER_ID))

# Ensure the log directory exists
GRADING_SERVERS_DIR="grading_servers"
mkdir -p "${GRADING_SERVERS_DIR}"

nohup env GRADING_SERVER_PORT="${PORT}" python -u -m engine.validation.format_server \
  dataset_dir="${dataset_dir}" \
  data_dir="none" \
  desc_file="none" > "${GRADING_SERVERS_DIR}/grading_server_${SERVER_ID}.out" 2>&1 &

echo $! > "${GRADING_SERVERS_DIR}/grading_server_${SERVER_ID}.pid"
echo "Grading server started with PID: $! (ID: ${SERVER_ID}, PORT: ${PORT})"
