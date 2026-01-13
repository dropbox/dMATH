# Amazon S3 - Simple Storage Service

Amazon S3 is an object storage service offering industry-leading scalability, data availability, security, and performance. It's designed for 99.999999999% (11 9's) durability.

## Core Concepts

- **Bucket**: Container for objects (globally unique name)
- **Object**: File + metadata (up to 5TB)
- **Key**: Unique identifier within a bucket
- **Region**: Geographic location of bucket

## AWS CLI Commands

### Bucket Operations

```bash
# List buckets
aws s3 ls

# Create bucket
aws s3 mb s3://my-bucket

# Create bucket in specific region
aws s3 mb s3://my-bucket --region us-west-2

# Delete empty bucket
aws s3 rb s3://my-bucket

# Delete bucket and all contents
aws s3 rb s3://my-bucket --force
```

### Object Operations

```bash
# List objects
aws s3 ls s3://my-bucket/
aws s3 ls s3://my-bucket/prefix/ --recursive

# Copy files
aws s3 cp file.txt s3://my-bucket/
aws s3 cp s3://my-bucket/file.txt ./
aws s3 cp s3://bucket1/file.txt s3://bucket2/

# Sync directories
aws s3 sync ./local-dir s3://my-bucket/prefix/
aws s3 sync s3://my-bucket/prefix/ ./local-dir

# Move files
aws s3 mv file.txt s3://my-bucket/
aws s3 mv s3://my-bucket/old-key s3://my-bucket/new-key

# Remove files
aws s3 rm s3://my-bucket/file.txt
aws s3 rm s3://my-bucket/prefix/ --recursive
```

### Advanced CLI Options

```bash
# Sync with delete (mirror)
aws s3 sync ./local s3://my-bucket/ --delete

# Exclude/include patterns
aws s3 sync ./local s3://my-bucket/ --exclude "*.log" --include "*.txt"

# Dry run
aws s3 sync ./local s3://my-bucket/ --dryrun

# Set storage class
aws s3 cp file.txt s3://my-bucket/ --storage-class GLACIER

# Set ACL
aws s3 cp file.txt s3://my-bucket/ --acl public-read

# Multipart upload threshold
aws s3 cp large-file.zip s3://my-bucket/ --expected-size 5GB
```

## Python SDK (boto3)

### Basic Operations

```python
import boto3

s3 = boto3.client('s3')

# Upload file
s3.upload_file('local-file.txt', 'my-bucket', 'remote-key.txt')

# Upload with metadata
s3.upload_file(
    'local-file.txt',
    'my-bucket',
    'remote-key.txt',
    ExtraArgs={
        'ContentType': 'text/plain',
        'Metadata': {'author': 'john'}
    }
)

# Download file
s3.download_file('my-bucket', 'remote-key.txt', 'local-file.txt')

# Delete object
s3.delete_object(Bucket='my-bucket', Key='remote-key.txt')

# List objects
response = s3.list_objects_v2(Bucket='my-bucket', Prefix='folder/')
for obj in response.get('Contents', []):
    print(obj['Key'], obj['Size'])
```

### File-like Objects

```python
import boto3
from io import BytesIO

s3 = boto3.client('s3')

# Upload from memory
data = b'Hello, World!'
s3.put_object(Bucket='my-bucket', Key='hello.txt', Body=data)

# Download to memory
response = s3.get_object(Bucket='my-bucket', Key='hello.txt')
content = response['Body'].read()

# Stream large files
s3.upload_fileobj(BytesIO(data), 'my-bucket', 'key.txt')
```

### Presigned URLs

```python
# Generate presigned URL for download
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'file.txt'},
    ExpiresIn=3600  # 1 hour
)

# Generate presigned URL for upload
url = s3.generate_presigned_url(
    'put_object',
    Params={'Bucket': 'my-bucket', 'Key': 'upload.txt'},
    ExpiresIn=3600
)
```

## Storage Classes

| Class | Use Case | Retrieval |
|-------|----------|-----------|
| STANDARD | Frequently accessed | Milliseconds |
| INTELLIGENT_TIERING | Unknown access patterns | Milliseconds |
| STANDARD_IA | Infrequent access | Milliseconds |
| ONEZONE_IA | Infrequent, single AZ | Milliseconds |
| GLACIER_IR | Archive, instant retrieval | Milliseconds |
| GLACIER | Archive | Minutes to hours |
| DEEP_ARCHIVE | Long-term archive | Hours |

```bash
# Set storage class
aws s3 cp file.txt s3://bucket/ --storage-class STANDARD_IA

# Transition via lifecycle rule (see below)
```

## Lifecycle Rules

```json
{
    "Rules": [
        {
            "ID": "MoveToGlacier",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "logs/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ],
            "Expiration": {
                "Days": 365
            }
        }
    ]
}
```

```bash
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json
```

## Versioning

```bash
# Enable versioning
aws s3api put-bucket-versioning \
    --bucket my-bucket \
    --versioning-configuration Status=Enabled

# List versions
aws s3api list-object-versions --bucket my-bucket

# Download specific version
aws s3api get-object \
    --bucket my-bucket \
    --key file.txt \
    --version-id "version-id" \
    local-file.txt

# Delete specific version
aws s3api delete-object \
    --bucket my-bucket \
    --key file.txt \
    --version-id "version-id"
```

## Bucket Policies

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicRead",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-bucket/*"
        },
        {
            "Sid": "AllowUpload",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::123456789012:role/UploadRole"
            },
            "Action": ["s3:PutObject"],
            "Resource": "arn:aws:s3:::my-bucket/uploads/*"
        }
    ]
}
```

```bash
aws s3api put-bucket-policy \
    --bucket my-bucket \
    --policy file://policy.json
```

## CORS Configuration

```json
{
    "CORSRules": [
        {
            "AllowedOrigins": ["https://example.com"],
            "AllowedMethods": ["GET", "PUT", "POST"],
            "AllowedHeaders": ["*"],
            "ExposeHeaders": ["ETag"],
            "MaxAgeSeconds": 3000
        }
    ]
}
```

```bash
aws s3api put-bucket-cors \
    --bucket my-bucket \
    --cors-configuration file://cors.json
```

## Static Website Hosting

```bash
# Enable website hosting
aws s3 website s3://my-bucket/ \
    --index-document index.html \
    --error-document error.html

# Access URL: http://my-bucket.s3-website-us-east-1.amazonaws.com
```

## Server-Side Encryption

```bash
# SSE-S3 (Amazon managed keys)
aws s3 cp file.txt s3://bucket/ --sse AES256

# SSE-KMS (AWS KMS keys)
aws s3 cp file.txt s3://bucket/ --sse aws:kms --sse-kms-key-id alias/my-key

# Default encryption for bucket
aws s3api put-bucket-encryption \
    --bucket my-bucket \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "aws:kms",
                "KMSMasterKeyID": "arn:aws:kms:..."
            }
        }]
    }'
```

## Event Notifications

```bash
# Trigger Lambda on object creation
aws s3api put-bucket-notification-configuration \
    --bucket my-bucket \
    --notification-configuration '{
        "LambdaFunctionConfigurations": [{
            "LambdaFunctionArn": "arn:aws:lambda:...",
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {
                    "FilterRules": [{
                        "Name": "prefix",
                        "Value": "uploads/"
                    }]
                }
            }
        }]
    }'
```

## Best Practices

1. **Use unique bucket names** - Globally unique across AWS
2. **Enable versioning** - Protect against accidental deletion
3. **Use lifecycle policies** - Optimize storage costs
4. **Enable encryption** - Server-side encryption by default
5. **Use presigned URLs** - Time-limited access
6. **Block public access** - Unless explicitly needed

## Links

- Documentation: https://docs.aws.amazon.com/s3/
- Pricing: https://aws.amazon.com/s3/pricing/
- Best Practices: https://docs.aws.amazon.com/AmazonS3/latest/userguide/best-practices.html
