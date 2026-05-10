#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Browse-Master MCP Services - One-click Startup Script
# ============================================
#
# Usage:
#   ./start_all.sh              # Start all services (default ports)
#   ./start_all.sh stop         # Stop all services
#   ./start_all.sh status       # Check service status
#   ./start_all.sh restart      # Restart all services
#
# Custom Ports:
#   SANDBOX_PORT=8001 SEARCH_PORT=8002 ./start_all.sh
#
# Default Ports:
#   - mcp-sandbox:   8001 (Code execution sandbox)
#   - search-tools:  8002 (Search tools)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${SCRIPT_DIR}/.pids"

# Default ports
SANDBOX_PORT="${SANDBOX_PORT:-8001}"
SEARCH_PORT="${SEARCH_PORT:-8002}"
API_PORT="${API_PORT:-1234}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure PID directory exists
mkdir -p "${PID_DIR}"

start_sandbox() {
    log_info "Starting mcp-sandbox (Port: ${SANDBOX_PORT})..."
    cd "${SCRIPT_DIR}/MCP"
    PORT="${SANDBOX_PORT}" nohup python evomaster_mcp_server.py > "${PID_DIR}/sandbox.log" 2>&1 &
    echo $! > "${PID_DIR}/sandbox.pid"
    log_info "mcp-sandbox started (PID: $(cat ${PID_DIR}/sandbox.pid))"
}

start_search() {
    log_info "Starting search-tools (API Port: ${API_PORT}, MCP Port: ${SEARCH_PORT})..."
    cd "${SCRIPT_DIR}/api_proxy"

    # Start API service
    PORT="${API_PORT}" nohup python api_server.py > "${PID_DIR}/api.log" 2>&1 &
    echo $! > "${PID_DIR}/api.pid"

    # Wait for API service to start
    sleep 2

    # Start MCP adapter
    MCP_PORT="${SEARCH_PORT}" nohup python browse_master_mcp_adapter.py > "${PID_DIR}/search.log" 2>&1 &
    echo $! > "${PID_DIR}/search.pid"

    log_info "search-tools started (API PID: $(cat ${PID_DIR}/api.pid), MCP PID: $(cat ${PID_DIR}/search.pid))"
}

stop_service() {
    local name=$1
    local pid_file="${PID_DIR}/${name}.pid"

    if [[ -f "${pid_file}" ]]; then
        local pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            log_info "Stopped ${name} (PID: ${pid})"
        else
            log_warn "${name} process does not exist"
        fi
        rm -f "${pid_file}"
    else
        log_warn "${name} PID file does not exist"
    fi
}

stop_all() {
    log_info "Stopping all services..."
    stop_service "sandbox"
    stop_service "api"
    stop_service "search"
    log_info "All services stopped"
}

check_service() {
    local name=$1
    local port=$2
    local pid_file="${PID_DIR}/${name}.pid"

    if [[ -f "${pid_file}" ]]; then
        local pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} ${name}: Running (PID: ${pid}, Port: ${port})"
            return 0
        fi
    fi
    echo -e "  ${RED}○${NC} ${name}: Not running"
    return 1
}

status() {
    echo "============================================"
    echo "Browse-Master MCP Services Status"
    echo "============================================"
    check_service "sandbox" "${SANDBOX_PORT}" || true
    check_service "api" "${API_PORT}" || true
    check_service "search" "${SEARCH_PORT}" || true
    echo "============================================"
    echo ""
    echo "MCP Endpoints:"
    echo "  - mcp-sandbox:  http://127.0.0.1:${SANDBOX_PORT}/mcp"
    echo "  - search-tools: http://127.0.0.1:${SEARCH_PORT}/mcp"
    echo ""
    echo "Log Files: ${PID_DIR}/*.log"
}

start_all() {
    log_info "Starting all Browse-Master MCP services..."
    echo ""
    start_sandbox
    start_search
    echo ""
    log_info "All services started!"
    echo ""
    status
}

# Main logic
case "${1:-start}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac