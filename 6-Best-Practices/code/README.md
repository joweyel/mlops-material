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