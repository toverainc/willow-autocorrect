#!/bin/bash
set -e

. .env

# Log level - acceptable values are debug, info, warning, error, critical. Suggest info or debug.
LOG_LEVEL=${LOG_LEVEL:-info}

set +a

uvicorn wac:app --host 0.0.0.0 --port 9000 --reload --log-config uvicorn-log-config.json \
    --log-level "$LOG_LEVEL" --loop uvloop --timeout-graceful-shutdown 5 \
    --no-server-header