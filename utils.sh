#!/bin/bash
set -e
WAC_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$WAC_DIR"

# Test for local environment file and use any overrides
if [ -r .env ]; then
    echo "Using configuration overrides from .env file"
    . .env
fi

set -a

if [ -z "$HA_URL" ] || [ -z "$HA_TOKEN" ];then
    echo "You need to create .env with at least HA_TOKEN and HA_URL - exiting"
    exit 1
fi

# Set the basics
TAG=${TAG:-latest}
IMAGE=${IMAGE:-wac}

TYPESENSE="typesense/typesense:0.25.1"
TYPESENSE_API_KEY=${TYPESENSE_API_KEY:-testing}

TYPESENSE_DIR=$WAC_DIR/data/ts

WAC_PORT=${WAC_PORT:-9000}
TYPESENSE_PORT=${TYPESENSE_PORT:-8108}

# Just in case
mkdir -p "$TYPESENSE_DIR"

freeze_requirements() {
    if [ ! -f /.dockerenv ]; then
        echo "This script is meant to be run inside the container - exiting"
        exit 1
    fi

    # Freeze
    pip freeze > requirements.txt
}

case $1 in

build|build-docker)
    docker build -t "$IMAGE":"$TAG" .
;;

freeze-requirements|fr)
    freeze_requirements
;;

gen_commands|gc)
    docker run --rm -it -v $WAC_DIR:/app "$IMAGE":"$TAG" python3 generate_commands.py
;;

run|start)
    # Reachable WAC IP for the "default" interface
    WAC_IP=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+')

    # Docker temp workaround for dev
    TYPESENSE_HOST=${TYPESENSE_HOST:-"$WAC_IP"}

    docker run --rm -it -p "$WAC_PORT":9000 -v $WAC_DIR:/app -e TYPESENSE_HOST \
        -e TYPESENSE_API_KEY --name wac "$IMAGE":"$TAG"
;;

shell)
    docker run --rm -it -v $WAC_DIR:/app -e TYPESENSE_HOST \
        -e TYPESENSE_API_KEY "$IMAGE":"$TAG" /bin/bash
;;

wipe)
    echo "Removing $TYPESENSE_DIR - needs sudo"
    sudo rm -rf "$TYPESENSE_DIR"
;;

typesense|ts)
    docker run --rm -it -p "$TYPESENSE_PORT":8108 \
        -v"$TYPESENSE_DIR":/data --name wac-ts "$TYPESENSE" \
        --data-dir /data \
        --api-key="$TYPESENSE_API_KEY"
;;

esac