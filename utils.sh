#!/bin/bash
set -e 
. .env

TAG=${TAG:-latest}
IMAGE=${IMAGE:-wac}

TYPESENSE="typesense/typesense:0.25.1"
TYPESENSE_API_KEY=${TYPESENSE_API_KEY:-testing}

TS_DIR=$PWD/data/ts

# Not really used yet
WAC_DB=$PWD/data/wac.db

set +a

# Just in case
mkdir -p data

case $1 in

build)
    docker build -t "$IMAGE":"$TAG" .
;;

gen_commands|gc)
    docker run --rm -it -v $PWD:/app "$IMAGE":"$TAG" python3 generate_commands.py
;;

run)
    docker run --rm -it -p 9000:9000 -v $PWD:/app -e WAC_DB -e TYPESENSE_API_KEY "$IMAGE":"$TAG"
;;

ts)
    mkdir -p $TS_DIR

    docker run --rm -it -p 9001:8108 \
        -v"$TS_DIR"/typesense-data:/data "$TYPESENSE" \
        --data-dir /data \
        --enable-search-analytics \
        --api-key=$TYPESENSE_API_KEY
;;

web)
    docker run -it --rm \
        -p 9002:8080 \
        -v $PWD:/data \
        coleifer/sqlite-web sqlite_web -H 0.0.0.0 -x $WAC_DB
;;

esac