```bash
docker build -t stream-model-duration:v2 .
```

```bash
export AWS_PROFILE=...

docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e RUN_ID="10f4197008104ad183466cdb19e26c4e" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    -e AWS_PROFILE=${AWS_PROFILE} \
    -v ~/.aws:/root/.aws \
    stream-model-duration:v2
```

```bash
export AWS_PROFILE=...

docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e RUN_ID="Test123" \
    -e MODEL_LOCATION="/app/model" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    -e AWS_PROFILE=${AWS_PROFILE} \
    -v ~/.aws:/root/.aws \
    -v $(pwd)/model:/app/model \
    stream-model-duration:v2
```

### `localstack`

List kinesis-streams in localstack
```bash
aws --endpoint-url=http://localhost:4566 kinesis list-streams
```

Create kinesis-stream in localstack
```bash
aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ride_predictions \
    --shard-count 1
```
```bash
export SHARD='shardId-000000000000'
export PREDICTIONS_STREAM_NAME='ride_predictions'

# Get shard iterator
SHARD_ITERATOR=$(aws --endpoint-url=http://localhost:4566 \
    kinesis get-shard-iterator \
    --shard-id ${SHARD} \
    --shard-iterator-type TRIM_HORIZON \
    --stream-name ${PREDICTIONS_STREAM_NAME} \
    --query 'ShardIterator' \
)

# Extracting records from shard
RESULT=$(aws --endpoint-url=http://localhost:4566 kinesis get-records --shard-iterator $SHARD_ITERATOR)

# Getting the predictions and decode them
echo $RESULT | jq -r '.Records[0].Data' | base64 --decode
```
### Makefiles

Without Makefile:
```bash
isort .
black .
pylint --recursive=y .
pytest tests/
```

To prepare the project run:
```bash
make setup
```

To run the quality check with the commands as seen above, run:
```bash
make quality_checks
```

For all possible options for executing of the [Makefile](Makefile) please refere to the file itsel.
