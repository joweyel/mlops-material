#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="stream-model-duration:${LOCAL_TAG}"
export PREDICTIONS_STREAM_NAME="ride_predictions"

docker build -t ${LOCAL_IMAGE_NAME} ..

docker-compose up -d

sleep 1

aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ${PREDICTIONS_STREAM_NAME} \
    --shard-count 1

pipenv run python3 test_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
fi

# docker-compose down

# exit ${ERROR_CODE}