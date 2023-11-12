#!/bin/bash
set -e
WAC_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$WAC_DIR"

# Test for local environment file and use any overrides
if [ -r .env ]; then
    echo "Using configuration overrides from .env file"
    . .env
else
    echo "Using default configuration values"
    touch .env
fi

#Import source the .env file
set -a
source .env

# Set the basics
TAG=${TAG:-latest}
IMAGE=${IMAGE:-wac}

TYPESENSE="typesense/typesense:0.25.1"
TYPESENSE_API_KEY=${TYPESENSE_API_KEY:-testing}

TS_DIR=$PWD/data/ts

WAC_PORT=${WAC_PORT:-9000}
TYPESENSE_PORT=${TYPESENSE_PORT:-8108}

# Just in case
mkdir -p data

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
    docker run --rm -it -v $PWD:/app "$IMAGE":"$TAG" python3 generate_commands.py
;;

run|start)
    docker run --rm -it -p "$WAC_PORT":9000 -v $PWD:/app -e WAC_DB -e TYPESENSE_API_KEY "$IMAGE":"$TAG"
;;

shell)
    docker run --rm -it -v $PWD:/app -e WAC_DB -e TYPESENSE_API_KEY "$IMAGE":"$TAG" /bin/bash
;;

typesense|ts)
    mkdir -p $TS_DIR

    docker run --rm -it -p "$TYPESENSE_PORT":8108 \
        -v"$TS_DIR":/data "$TYPESENSE" \
        --data-dir /data \
        --api-key="$TYPESENSE_API_KEY"
;;

esac