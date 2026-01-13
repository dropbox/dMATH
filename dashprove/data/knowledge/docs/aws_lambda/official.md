# AWS Lambda - Serverless Compute Service

AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the underlying compute resources. You can use Lambda to extend other AWS services or create your own backend services.

## Supported Runtimes

- Python 3.9, 3.10, 3.11, 3.12
- Node.js 18.x, 20.x
- Java 11, 17, 21
- .NET 6, 8
- Ruby 3.2, 3.3
- Go (via provided runtime)
- Rust (via provided runtime)
- Custom runtimes (AL2, AL2023)

## Creating Functions

### AWS Console

1. Navigate to Lambda in AWS Console
2. Click "Create function"
3. Choose runtime and architecture
4. Configure handler and memory

### AWS CLI

```bash
# Create function from ZIP
aws lambda create-function \
    --function-name my-function \
    --runtime python3.12 \
    --handler handler.lambda_handler \
    --role arn:aws:iam::123456789012:role/lambda-role \
    --zip-file fileb://function.zip

# Create function from S3
aws lambda create-function \
    --function-name my-function \
    --runtime python3.12 \
    --handler handler.lambda_handler \
    --role arn:aws:iam::123456789012:role/lambda-role \
    --code S3Bucket=my-bucket,S3Key=function.zip
```

### SAM/CloudFormation

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handler.lambda_handler
      Runtime: python3.12
      CodeUri: src/
      MemorySize: 256
      Timeout: 30
      Events:
        Api:
          Type: Api
          Properties:
            Path: /hello
            Method: get
```

## Handler Functions

### Python

```python
def lambda_handler(event, context):
    """
    event: Input data (dict)
    context: Runtime information
    """
    print(f"Request ID: {context.aws_request_id}")
    print(f"Function: {context.function_name}")
    print(f"Memory: {context.memory_limit_in_mb}MB")
    print(f"Remaining time: {context.get_remaining_time_in_millis()}ms")

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Hello from Lambda!'})
    }
```

### Node.js

```javascript
export const handler = async (event, context) => {
    console.log('Event:', JSON.stringify(event, null, 2));

    return {
        statusCode: 200,
        body: JSON.stringify({ message: 'Hello from Lambda!' })
    };
};
```

### Rust (Custom Runtime)

```rust
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde_json::{json, Value};

async fn handler(event: LambdaEvent<Value>) -> Result<Value, Error> {
    Ok(json!({
        "statusCode": 200,
        "body": "Hello from Rust Lambda!"
    }))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}
```

## Event Sources

### API Gateway

```python
def handler(event, context):
    # HTTP event structure
    http_method = event['httpMethod']
    path = event['path']
    query_params = event.get('queryStringParameters', {})
    body = event.get('body', '')
    headers = event['headers']

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'received': body})
    }
```

### S3 Events

```python
def handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        event_name = record['eventName']  # e.g., 'ObjectCreated:Put'

        print(f"Processing {key} from {bucket}")
```

### SQS Events

```python
def handler(event, context):
    for record in event['Records']:
        body = json.loads(record['body'])
        message_id = record['messageId']

        # Process message
        print(f"Processing message: {message_id}")

    # Return batch item failures for partial batch response
    return {'batchItemFailures': []}
```

### DynamoDB Streams

```python
def handler(event, context):
    for record in event['Records']:
        event_name = record['eventName']  # INSERT, MODIFY, REMOVE
        new_image = record['dynamodb'].get('NewImage', {})
        old_image = record['dynamodb'].get('OldImage', {})
```

### EventBridge (CloudWatch Events)

```python
def handler(event, context):
    detail_type = event['detail-type']
    source = event['source']
    detail = event['detail']
```

## Configuration

### Memory and Timeout

```bash
# Update function configuration
aws lambda update-function-configuration \
    --function-name my-function \
    --memory-size 512 \
    --timeout 30
```

### Environment Variables

```bash
aws lambda update-function-configuration \
    --function-name my-function \
    --environment "Variables={DB_HOST=hostname,DB_PORT=5432}"
```

```python
import os

def handler(event, context):
    db_host = os.environ['DB_HOST']
    db_port = os.environ['DB_PORT']
```

### Layers

```bash
# Create layer
aws lambda publish-layer-version \
    --layer-name my-layer \
    --zip-file fileb://layer.zip \
    --compatible-runtimes python3.12

# Add layer to function
aws lambda update-function-configuration \
    --function-name my-function \
    --layers arn:aws:lambda:us-east-1:123456789012:layer:my-layer:1
```

### VPC Configuration

```bash
aws lambda update-function-configuration \
    --function-name my-function \
    --vpc-config SubnetIds=subnet-1234,subnet-5678,SecurityGroupIds=sg-1234
```

## Invocation

### Synchronous

```bash
aws lambda invoke \
    --function-name my-function \
    --payload '{"key": "value"}' \
    output.json
```

### Asynchronous

```bash
aws lambda invoke \
    --function-name my-function \
    --invocation-type Event \
    --payload '{"key": "value"}' \
    output.json
```

### Python SDK

```python
import boto3
import json

client = boto3.client('lambda')

# Synchronous
response = client.invoke(
    FunctionName='my-function',
    InvocationType='RequestResponse',
    Payload=json.dumps({'key': 'value'})
)
result = json.loads(response['Payload'].read())

# Asynchronous
response = client.invoke(
    FunctionName='my-function',
    InvocationType='Event',
    Payload=json.dumps({'key': 'value'})
)
```

## Permissions

### Execution Role

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

### Resource-Based Policy

```bash
# Allow API Gateway to invoke
aws lambda add-permission \
    --function-name my-function \
    --statement-id apigateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:us-east-1:123456789012:api-id/*"
```

## Best Practices

### Cold Start Optimization

```python
# Initialize outside handler (reused across invocations)
import boto3
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('my-table')

def handler(event, context):
    # Use pre-initialized resources
    table.get_item(Key={'id': '123'})
```

### Provisioned Concurrency

```bash
aws lambda put-provisioned-concurrency-config \
    --function-name my-function \
    --qualifier prod \
    --provisioned-concurrent-executions 10
```

### Error Handling

```python
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    try:
        # Process event
        result = process(event)
        return {'statusCode': 200, 'body': json.dumps(result)}
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {'statusCode': 400, 'body': str(e)}
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise  # Let Lambda retry
```

## Monitoring

### CloudWatch Logs

```bash
# View recent logs
aws logs tail /aws/lambda/my-function --follow
```

### Metrics

- Invocations
- Duration
- Errors
- Throttles
- ConcurrentExecutions
- IteratorAge (for stream sources)

### X-Ray Tracing

```bash
aws lambda update-function-configuration \
    --function-name my-function \
    --tracing-config Mode=Active
```

## Links

- Documentation: https://docs.aws.amazon.com/lambda/
- Pricing: https://aws.amazon.com/lambda/pricing/
- Quotas: https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html
