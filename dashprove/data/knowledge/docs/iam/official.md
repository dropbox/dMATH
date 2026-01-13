# AWS IAM - Identity and Access Management

AWS Identity and Access Management (IAM) enables you to manage access to AWS services and resources securely. Using IAM, you can create and manage AWS users and groups, and use permissions to allow and deny their access to AWS resources.

## Core Concepts

- **Users**: Individual identities with long-term credentials
- **Groups**: Collections of users with shared permissions
- **Roles**: Temporary credentials for AWS services or users
- **Policies**: JSON documents defining permissions
- **Identity Providers**: External authentication systems

## AWS CLI Commands

### Users

```bash
# Create user
aws iam create-user --user-name developer

# List users
aws iam list-users

# Delete user
aws iam delete-user --user-name developer

# Create access key
aws iam create-access-key --user-name developer

# List access keys
aws iam list-access-keys --user-name developer

# Delete access key
aws iam delete-access-key --user-name developer --access-key-id AKIAIOSFODNN7EXAMPLE
```

### Groups

```bash
# Create group
aws iam create-group --group-name Developers

# Add user to group
aws iam add-user-to-group --user-name developer --group-name Developers

# List groups
aws iam list-groups

# List users in group
aws iam get-group --group-name Developers

# Remove user from group
aws iam remove-user-from-group --user-name developer --group-name Developers
```

### Policies

```bash
# Create policy
aws iam create-policy \
    --policy-name MyPolicy \
    --policy-document file://policy.json

# List policies
aws iam list-policies --scope Local

# Get policy version
aws iam get-policy-version \
    --policy-arn arn:aws:iam::123456789012:policy/MyPolicy \
    --version-id v1

# Attach policy to user
aws iam attach-user-policy \
    --user-name developer \
    --policy-arn arn:aws:iam::123456789012:policy/MyPolicy

# Attach policy to group
aws iam attach-group-policy \
    --group-name Developers \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Attach policy to role
aws iam attach-role-policy \
    --role-name MyRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### Roles

```bash
# Create role
aws iam create-role \
    --role-name MyRole \
    --assume-role-policy-document file://trust-policy.json

# List roles
aws iam list-roles

# Delete role
aws iam delete-role --role-name MyRole

# Assume role (STS)
aws sts assume-role \
    --role-arn arn:aws:iam::123456789012:role/MyRole \
    --role-session-name MySession
```

## Policy Documents

### Basic Policy Structure

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3Read",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ]
        }
    ]
}
```

### Policy Elements

| Element | Description |
|---------|-------------|
| Version | Policy language version (use "2012-10-17") |
| Statement | Array of permission statements |
| Sid | Optional statement identifier |
| Effect | Allow or Deny |
| Action | List of actions (service:action) |
| Resource | ARNs the policy applies to |
| Condition | Optional conditions |
| Principal | Who the policy applies to (for resource policies) |

### Trust Policy (for Roles)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

### Cross-Account Access

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::999999999999:root"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "sts:ExternalId": "unique-id"
                }
            }
        }
    ]
}
```

## Conditions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": "*",
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": ["192.168.1.0/24", "10.0.0.0/8"]
                },
                "Bool": {
                    "aws:MultiFactorAuthPresent": "true"
                },
                "StringEquals": {
                    "s3:x-amz-acl": "bucket-owner-full-control"
                }
            }
        }
    ]
}
```

### Common Condition Keys

| Key | Description |
|-----|-------------|
| aws:SourceIp | Requester's IP address |
| aws:MultiFactorAuthPresent | MFA used |
| aws:PrincipalOrgID | Organization ID |
| aws:RequestedRegion | Target region |
| aws:CurrentTime | Current time |
| aws:SecureTransport | HTTPS used |

## Common Policies

### Administrator Access

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "*",
            "Resource": "*"
        }
    ]
}
```

### Read-Only Access

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:Get*",
                "s3:List*",
                "ec2:Describe*"
            ],
            "Resource": "*"
        }
    ]
}
```

### Deny All Except Specific Region

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "aws:RequestedRegion": "us-east-1"
                }
            }
        }
    ]
}
```

### Require MFA

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "BoolIfExists": {
                    "aws:MultiFactorAuthPresent": "false"
                }
            }
        }
    ]
}
```

## Service-Linked Roles

```bash
# Create service-linked role
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com

# List service-linked roles
aws iam list-roles --path-prefix /aws-service-role/
```

## Identity Providers

### OIDC Provider (for EKS IRSA)

```bash
# Create OIDC provider
aws iam create-open-id-connect-provider \
    --url https://oidc.eks.us-west-2.amazonaws.com/id/EXAMPLED539D4633E53DE1B716D3041E \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 9e99a48a9960b14926bb7f3b02e22da0e3c8d8c8
```

### SAML Provider

```bash
aws iam create-saml-provider \
    --saml-metadata-document file://metadata.xml \
    --name MyIdP
```

## Password Policy

```bash
# Set password policy
aws iam update-account-password-policy \
    --minimum-password-length 14 \
    --require-symbols \
    --require-numbers \
    --require-uppercase-characters \
    --require-lowercase-characters \
    --max-password-age 90 \
    --password-reuse-prevention 12
```

## MFA

```bash
# Enable virtual MFA
aws iam create-virtual-mfa-device \
    --virtual-mfa-device-name developer-mfa \
    --outfile /tmp/qrcode.png \
    --bootstrap-method QRCodePNG

# Enable MFA for user
aws iam enable-mfa-device \
    --user-name developer \
    --serial-number arn:aws:iam::123456789012:mfa/developer-mfa \
    --authentication-code1 123456 \
    --authentication-code2 789012
```

## Credential Report

```bash
# Generate credential report
aws iam generate-credential-report

# Get credential report
aws iam get-credential-report --output text --query Content | base64 --decode > report.csv
```

## Policy Simulator

```bash
# Simulate policy
aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::123456789012:user/developer \
    --action-names s3:GetObject \
    --resource-arns arn:aws:s3:::my-bucket/*
```

## Best Practices

1. **Use roles instead of users** - For applications and services
2. **Enable MFA** - Especially for privileged users
3. **Use groups** - Manage permissions at group level
4. **Least privilege** - Grant only needed permissions
5. **Regular rotation** - Rotate credentials regularly
6. **Use conditions** - Add IP, MFA, time restrictions
7. **Monitor with CloudTrail** - Track API calls
8. **Use AWS Organizations SCPs** - Governance guardrails

## Policy Evaluation Logic

1. **Explicit Deny** - Always wins
2. **Organization SCPs** - Must allow
3. **Resource Policies** - Cross-account
4. **IAM Permissions Boundary** - Maximum permissions
5. **Session Policies** - Temporary restrictions
6. **Identity Policies** - User/Role policies

## Links

- Documentation: https://docs.aws.amazon.com/iam/
- Policy Reference: https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies.html
- Best Practices: https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html
