#!/bin/bash
set -e

if [ -r .env ]; then
    . .env
fi

# Log level - acceptable values are debug, info, warning, error, critical. Suggest info or debug.
LOG_LEVEL=${LOG_LEVEL:-info}

RUN_MODE=${RUN_MODE:-prod}

set -a

if [ "$RUN_MODE" == "dev" ]; then
    echo "Running in dev mode. Make sure you have Typesense started on host!"
    LOG_LEVEL="debug"
    uvicorn wac:app --host 0.0.0.0 --port 9000 --reload --log-config uvicorn-log-config.json \
        --log-level "$LOG_LEVEL" --loop uvloop --timeout-graceful-shutdown 5 \
        --no-server-header
else
    uvicorn wac:app --host 0.0.0.0 --port 9000 --log-config uvicorn-log-config.json \
        --log-level "$LOG_LEVEL" --loop uvloop --timeout-graceful-shutdown 5 \
        --no-server-header
fi