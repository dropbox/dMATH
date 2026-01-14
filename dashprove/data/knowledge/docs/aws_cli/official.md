# AWS CLI - Amazon Web Services Command Line Interface

The AWS Command Line Interface (CLI) is a unified tool to manage AWS services. With just one tool to download and configure, you can control multiple AWS services from the command line and automate them through scripts.

## Installation

```bash
# macOS (Homebrew)
brew install awscli

# macOS (pkg installer)
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Linux (bundled installer)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Windows (MSI)
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi

# Using pip (not recommended)
pip install awscli
```

## Configuration

### Initial Setup

```bash
# Configure default profile
aws configure

# Configure named profile
aws configure --profile production

# SSO configuration
aws configure sso
```

### Configuration Files

```ini
# ~/.aws/config
[default]
region = us-east-1
output = json

[profile production]
region = us-west-2
output = table
role_arn = arn:aws:iam::123456789012:role/AdminRole
source_profile = default

[profile sso-user]
sso_start_url = https://my-company.awsapps.com/start
sso_region = us-east-1
sso_account_id = 123456789012
sso_role_name = PowerUserAccess
region = us-east-1
```

```ini
# ~/.aws/credentials
[default]
aws_access_key_id = AKIAFAKEEXAMPLEKEY00
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

[production]
aws_access_key_id = AKIAFAKEEXAMPLEKEY01
aws_secret_access_key = je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY
```

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID=AKIAFAKEEXAMPLEKEY00
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=production
export AWS_SESSION_TOKEN=token  # For temporary credentials
```

## Basic Usage

### Command Structure

```bash
aws <service> <command> [options]

# Examples
aws s3 ls
aws ec2 describe-instances
aws lambda list-functions
```

### Common Options

```bash
# Specify profile
aws s3 ls --profile production

# Specify region
aws ec2 describe-instances --region us-west-2

# Output format
aws iam list-users --output json
aws iam list-users --output table
aws iam list-users --output text

# Query with JMESPath
aws ec2 describe-instances --query 'Reservations[].Instances[].InstanceId'

# Dry run (test permissions)
aws ec2 run-instances --dry-run ...

# Debug output
aws s3 ls --debug
```

## Common Service Commands

### S3

```bash
# List buckets
aws s3 ls

# List objects in bucket
aws s3 ls s3://bucket-name

# Copy file
aws s3 cp file.txt s3://bucket-name/

# Sync directory
aws s3 sync ./local s3://bucket-name/prefix

# Remove file
aws s3 rm s3://bucket-name/file.txt

# Create bucket
aws s3 mb s3://bucket-name

# Delete bucket
aws s3 rb s3://bucket-name --force
```

### EC2

```bash
# List instances
aws ec2 describe-instances

# Start/stop instances
aws ec2 start-instances --instance-ids i-1234567890abcdef0
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Launch instance
aws ec2 run-instances \
    --image-id ami-12345678 \
    --count 1 \
    --instance-type t2.micro \
    --key-name MyKeyPair

# Terminate instance
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

### Lambda

```bash
# List functions
aws lambda list-functions

# Invoke function
aws lambda invoke --function-name my-function output.json

# Update code
aws lambda update-function-code \
    --function-name my-function \
    --zip-file fileb://function.zip

# Create function
aws lambda create-function \
    --function-name my-function \
    --runtime python3.12 \
    --handler handler.lambda_handler \
    --role arn:aws:iam::123456789012:role/lambda-role \
    --zip-file fileb://function.zip
```

### IAM

```bash
# List users
aws iam list-users

# Create user
aws iam create-user --user-name newuser

# List policies
aws iam list-policies

# Attach policy
aws iam attach-user-policy \
    --user-name newuser \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Get caller identity (whoami)
aws sts get-caller-identity
```

### CloudWatch

```bash
# Get metrics
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-02T00:00:00Z \
    --period 3600 \
    --statistics Average

# List log groups
aws logs describe-log-groups

# Get log events
aws logs get-log-events \
    --log-group-name /aws/lambda/my-function \
    --log-stream-name stream-name
```

## JMESPath Queries

```bash
# Filter instances by state
aws ec2 describe-instances \
    --query 'Reservations[].Instances[?State.Name==`running`].InstanceId'

# Get specific fields
aws ec2 describe-instances \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]'

# First result
aws ec2 describe-instances --query 'Reservations[0].Instances[0]'

# Sort by field
aws ec2 describe-instances \
    --query 'sort_by(Reservations[].Instances[], &LaunchTime)'
```

## Pagination

```bash
# Auto-pagination (default)
aws s3api list-objects-v2 --bucket my-bucket

# Manual pagination
aws s3api list-objects-v2 --bucket my-bucket --max-items 100

# Get next page
aws s3api list-objects-v2 --bucket my-bucket --starting-token <token>
```

## Wait Commands

```bash
# Wait for instance running
aws ec2 wait instance-running --instance-ids i-1234567890abcdef0

# Wait for stack creation
aws cloudformation wait stack-create-complete --stack-name my-stack

# Wait for deployment
aws ecs wait services-stable --cluster my-cluster --services my-service
```

## SSO Login

```bash
# Configure SSO
aws configure sso

# Login
aws sso login --profile sso-user

# Logout
aws sso logout
```

## Assume Role

```bash
# Assume role
aws sts assume-role \
    --role-arn arn:aws:iam::123456789012:role/MyRole \
    --role-session-name MySession

# Use assumed role credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

## CLI Aliases

```ini
# ~/.aws/cli/alias
[toplevel]
whoami = sts get-caller-identity
running = ec2 describe-instances --query 'Reservations[].Instances[?State.Name==`running`]'
```

## Completion

```bash
# Bash
complete -C '/usr/local/bin/aws_completer' aws

# Zsh
autoload bashcompinit && bashcompinit
complete -C '/usr/local/bin/aws_completer' aws
```

## Best Practices

1. **Use named profiles** - Separate dev/staging/prod
2. **Use IAM roles** - Avoid long-term credentials
3. **Enable MFA** - For sensitive operations
4. **Use SSO** - Centralized access management
5. **Use --query** - Filter results efficiently
6. **Use wait commands** - For scripting

## Links

- Documentation: https://docs.aws.amazon.com/cli/
- GitHub: https://github.com/aws/aws-cli
- User Guide: https://docs.aws.amazon.com/cli/latest/userguide/
