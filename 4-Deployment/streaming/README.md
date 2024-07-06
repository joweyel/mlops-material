## Code snippets

### Sending data
```bash
KINESIS_STREAM_INPUT=ride_events
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1
    --data "Hello, this is a test."
```
**Parameters:**
- `--stream-name`: name of kinesis stream on AWS
- `--partition-key`: number of shart to send the data to
- `--data`: event data to send shard in kinesis stream

This should return something like this:
```json
{
    "ShardId": "shardId-000000000000",
    "SequenceNumber": "49653673755737192565750919189021188059313346570486808578"
}
```

Example of record that was obtained from the kinesis stream.
```json
{
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49653673755737192565750919189021188059313346570486808578",
                "data": "Hellothisisatest",
                "approximateArrivalTimestamp": 1720280798.411
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49653673755737192565750919189021188059313346570486808578",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::886638369043:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:886638369043:stream/ride_events"
        }
    ]
}
```