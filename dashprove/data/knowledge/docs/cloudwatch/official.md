# Amazon CloudWatch - Monitoring and Observability

Amazon CloudWatch is a monitoring and observability service that provides data and actionable insights for AWS, hybrid, and on-premises applications and infrastructure resources.

## Core Components

- **Metrics**: Time-series data points
- **Logs**: Log data from AWS resources
- **Alarms**: Automated actions based on metrics
- **Dashboards**: Visualizations
- **Events/EventBridge**: Event-driven automation

## Metrics

### AWS CLI Commands

```bash
# List metrics
aws cloudwatch list-metrics --namespace AWS/EC2

# Get metric statistics
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-02T00:00:00Z \
    --period 3600 \
    --statistics Average Maximum

# Put custom metric
aws cloudwatch put-metric-data \
    --namespace MyApp \
    --metric-name RequestCount \
    --value 100 \
    --dimensions Environment=Production,Service=API
```

### Python SDK (boto3)

```python
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

# Get metrics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/EC2',
    MetricName='CPUUtilization',
    Dimensions=[{'Name': 'InstanceId', 'Value': 'i-1234567890abcdef0'}],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    EndTime=datetime.utcnow(),
    Period=300,
    Statistics=['Average', 'Maximum']
)

# Put custom metrics
cloudwatch.put_metric_data(
    Namespace='MyApp',
    MetricData=[
        {
            'MetricName': 'RequestCount',
            'Value': 100,
            'Dimensions': [
                {'Name': 'Environment', 'Value': 'Production'},
                {'Name': 'Service', 'Value': 'API'}
            ]
        }
    ]
)
```

### Metric Math

```bash
# Create alarm with metric math
aws cloudwatch put-metric-alarm \
    --alarm-name "ErrorRateAlarm" \
    --metrics '[
        {"Id": "errors", "MetricStat": {"Metric": {"Namespace": "MyApp", "MetricName": "Errors"}, "Period": 300, "Stat": "Sum"}},
        {"Id": "requests", "MetricStat": {"Metric": {"Namespace": "MyApp", "MetricName": "Requests"}, "Period": 300, "Stat": "Sum"}},
        {"Id": "errorRate", "Expression": "(errors/requests)*100", "Label": "Error Rate"}
    ]' \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2
```

## Logs

### Log Groups and Streams

```bash
# Create log group
aws logs create-log-group --log-group-name /my-app/production

# Set retention
aws logs put-retention-policy \
    --log-group-name /my-app/production \
    --retention-in-days 30

# Create log stream
aws logs create-log-stream \
    --log-group-name /my-app/production \
    --log-stream-name app-server-1

# Put log events
aws logs put-log-events \
    --log-group-name /my-app/production \
    --log-stream-name app-server-1 \
    --log-events timestamp=1704067200000,message="Application started"
```

### Querying Logs

```bash
# Filter log events
aws logs filter-log-events \
    --log-group-name /my-app/production \
    --filter-pattern "ERROR" \
    --start-time 1704067200000 \
    --end-time 1704153600000

# Logs Insights query
aws logs start-query \
    --log-group-name /my-app/production \
    --start-time 1704067200 \
    --end-time 1704153600 \
    --query-string 'fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 100'

# Get query results
aws logs get-query-results --query-id "query-id"

# Tail logs (live)
aws logs tail /my-app/production --follow
```

### Log Insights Query Syntax

```sql
-- Filter by pattern
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 100

-- Aggregations
fields @timestamp, @message
| stats count(*) as errorCount by bin(1h)
| filter @message like /ERROR/

-- Parse structured logs
fields @timestamp, @message
| parse @message "user=* action=* status=*" as user, action, status
| filter status = "error"
| stats count(*) by user

-- JSON parsing
fields @timestamp, @message
| filter ispresent(requestId)
| stats avg(duration) as avgDuration by operation
```

### Python Logging

```python
import logging
import watchtower

# Configure CloudWatch logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler = watchtower.CloudWatchLogHandler(
    log_group='/my-app/production',
    stream_name='app-server-1'
)
logger.addHandler(handler)

logger.info('Application started')
logger.error('An error occurred', extra={'error_code': 500})
```

## Alarms

### Create Alarms

```bash
# Create metric alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "HighCPU" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:my-topic

# Create composite alarm
aws cloudwatch put-composite-alarm \
    --alarm-name "CompositeHighCPUAndErrors" \
    --alarm-rule "ALARM(HighCPU) AND ALARM(HighErrors)" \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:my-topic
```

### Alarm States

- **OK**: Metric within threshold
- **ALARM**: Metric breached threshold
- **INSUFFICIENT_DATA**: Not enough data

### Python SDK

```python
cloudwatch.put_metric_alarm(
    AlarmName='HighCPU',
    MetricName='CPUUtilization',
    Namespace='AWS/EC2',
    Statistic='Average',
    Period=300,
    Threshold=80,
    ComparisonOperator='GreaterThanThreshold',
    Dimensions=[{'Name': 'InstanceId', 'Value': 'i-1234567890abcdef0'}],
    EvaluationPeriods=2,
    AlarmActions=['arn:aws:sns:us-east-1:123456789012:my-topic']
)
```

## Dashboards

```bash
# Create dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "MyDashboard" \
    --dashboard-body file://dashboard.json
```

```json
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/EC2", "CPUUtilization", "InstanceId", "i-1234567890abcdef0"]
                ],
                "period": 300,
                "stat": "Average",
                "region": "us-east-1",
                "title": "EC2 CPU Utilization"
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 6,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '/my-app/production' | fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20",
                "region": "us-east-1",
                "title": "Recent Errors"
            }
        }
    ]
}
```

## Container Insights

```bash
# Enable Container Insights for EKS
aws eks update-cluster-config \
    --name my-cluster \
    --logging '{"clusterLogging":[{"types":["api","audit","authenticator","controllerManager","scheduler"],"enabled":true}]}'

# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluent-bit-quickstart.yaml
```

## Lambda Insights

```bash
# Enable Lambda Insights
aws lambda update-function-configuration \
    --function-name my-function \
    --layers arn:aws:lambda:us-east-1:580247275435:layer:LambdaInsightsExtension:38
```

## Synthetics (Canaries)

```bash
# Create canary
aws synthetics create-canary \
    --name my-canary \
    --code Handler=pageLoadBlueprint.handler,S3Bucket=my-bucket,S3Key=canary.zip \
    --artifact-s3-location s3://my-bucket/canary-artifacts/ \
    --execution-role-arn arn:aws:iam::123456789012:role/canary-role \
    --schedule Expression="rate(5 minutes)" \
    --runtime-version syn-nodejs-puppeteer-6.1

# Start canary
aws synthetics start-canary --name my-canary
```

## Common Namespaces

| Namespace | Service |
|-----------|---------|
| AWS/EC2 | EC2 instances |
| AWS/Lambda | Lambda functions |
| AWS/RDS | RDS databases |
| AWS/ECS | ECS services |
| AWS/EKS | EKS clusters |
| AWS/S3 | S3 buckets |
| AWS/DynamoDB | DynamoDB tables |
| AWS/ApiGateway | API Gateway |
| AWS/SQS | SQS queues |
| AWS/SNS | SNS topics |

## Best Practices

1. **Use namespaces** - Organize metrics logically
2. **Set retention** - Control log storage costs
3. **Use Logs Insights** - Efficient log analysis
4. **Create dashboards** - Visualize key metrics
5. **Set up alarms** - Proactive monitoring
6. **Use metric math** - Complex alerting logic

## Links

- Documentation: https://docs.aws.amazon.com/cloudwatch/
- Logs Insights Query Syntax: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax.html
- Pricing: https://aws.amazon.com/cloudwatch/pricing/
