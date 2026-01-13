# Amazon EKS - Elastic Kubernetes Service

Amazon EKS is a managed Kubernetes service that makes it easy to run Kubernetes on AWS without needing to install, operate, and maintain your own Kubernetes control plane.

## Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"

# Install eksctl
brew tap weaveworks/tap
brew install weaveworks/tap/eksctl

# Configure AWS CLI
aws configure
```

## Cluster Creation

### Using eksctl (Recommended)

```bash
# Create cluster with defaults
eksctl create cluster --name my-cluster --region us-west-2

# Create with specific configuration
eksctl create cluster \
    --name my-cluster \
    --region us-west-2 \
    --version 1.30 \
    --nodegroup-name standard-workers \
    --node-type t3.medium \
    --nodes 3 \
    --nodes-min 1 \
    --nodes-max 5 \
    --managed

# Create with config file
eksctl create cluster -f cluster.yaml
```

### Cluster Configuration File

```yaml
# cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: my-cluster
  region: us-west-2
  version: "1.30"

managedNodeGroups:
  - name: ng-1
    instanceType: t3.medium
    desiredCapacity: 3
    minSize: 1
    maxSize: 5
    volumeSize: 80
    ssh:
      allow: true
      publicKeyName: my-key
    iam:
      withAddonPolicies:
        autoScaler: true
        albIngress: true
        cloudWatch: true

  - name: ng-spot
    instanceTypes: ["t3.medium", "t3.large"]
    spot: true
    desiredCapacity: 2

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
```

### Using AWS CLI

```bash
# Create cluster
aws eks create-cluster \
    --name my-cluster \
    --role-arn arn:aws:iam::123456789012:role/eks-cluster-role \
    --resources-vpc-config subnetIds=subnet-1,subnet-2,securityGroupIds=sg-1

# Wait for cluster to be active
aws eks wait cluster-active --name my-cluster

# Create node group
aws eks create-nodegroup \
    --cluster-name my-cluster \
    --nodegroup-name my-nodes \
    --scaling-config minSize=1,maxSize=5,desiredSize=3 \
    --subnets subnet-1 subnet-2 \
    --instance-types t3.medium \
    --node-role arn:aws:iam::123456789012:role/eks-node-role
```

## kubectl Configuration

```bash
# Update kubeconfig
aws eks update-kubeconfig --name my-cluster --region us-west-2

# Verify connection
kubectl get nodes
kubectl get pods -A
```

## Node Groups

### Managed Node Groups

```bash
# Create node group
eksctl create nodegroup \
    --cluster my-cluster \
    --name new-nodes \
    --node-type t3.large \
    --nodes 3

# Scale node group
eksctl scale nodegroup \
    --cluster my-cluster \
    --name new-nodes \
    --nodes 5

# Delete node group
eksctl delete nodegroup \
    --cluster my-cluster \
    --name old-nodes
```

### Fargate Profiles

```bash
# Create Fargate profile
eksctl create fargateprofile \
    --cluster my-cluster \
    --name fp-default \
    --namespace default

# With selectors
eksctl create fargateprofile \
    --cluster my-cluster \
    --name fp-app \
    --namespace app \
    --labels app=web
```

```yaml
# Fargate profile config
fargateProfiles:
  - name: fp-default
    selectors:
      - namespace: default
      - namespace: kube-system
  - name: fp-app
    selectors:
      - namespace: app
        labels:
          workload: fargate
```

## IAM Integration

### IRSA (IAM Roles for Service Accounts)

```bash
# Enable OIDC provider
eksctl utils associate-iam-oidc-provider \
    --cluster my-cluster \
    --approve

# Create IAM service account
eksctl create iamserviceaccount \
    --name my-service-account \
    --namespace default \
    --cluster my-cluster \
    --role-name my-role \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
    --approve
```

```yaml
# Pod using service account
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  serviceAccountName: my-service-account
  containers:
    - name: app
      image: my-app
```

### aws-auth ConfigMap

```bash
# View current auth config
kubectl describe configmap aws-auth -n kube-system

# Edit to add users/roles
kubectl edit configmap aws-auth -n kube-system
```

```yaml
# aws-auth ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-auth
  namespace: kube-system
data:
  mapRoles: |
    - rolearn: arn:aws:iam::123456789012:role/eks-node-role
      username: system:node:{{EC2PrivateDNSName}}
      groups:
        - system:bootstrappers
        - system:nodes
    - rolearn: arn:aws:iam::123456789012:role/AdminRole
      username: admin
      groups:
        - system:masters
  mapUsers: |
    - userarn: arn:aws:iam::123456789012:user/developer
      username: developer
      groups:
        - dev-group
```

## Add-ons

### Install Add-ons

```bash
# List available add-ons
aws eks describe-addon-versions --kubernetes-version 1.30

# Install add-on
aws eks create-addon \
    --cluster-name my-cluster \
    --addon-name vpc-cni \
    --addon-version v1.18.0-eksbuild.1

# Or with eksctl
eksctl create addon \
    --cluster my-cluster \
    --name aws-ebs-csi-driver
```

### Common Add-ons

| Add-on | Purpose |
|--------|---------|
| vpc-cni | VPC networking |
| coredns | DNS |
| kube-proxy | Network proxy |
| aws-ebs-csi-driver | EBS volumes |
| aws-efs-csi-driver | EFS volumes |

## Load Balancers

### AWS Load Balancer Controller

```bash
# Install AWS Load Balancer Controller
eksctl create iamserviceaccount \
    --cluster my-cluster \
    --namespace kube-system \
    --name aws-load-balancer-controller \
    --attach-policy-arn arn:aws:iam::123456789012:policy/AWSLoadBalancerControllerIAMPolicy \
    --approve

helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName=my-cluster \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller
```

### Ingress with ALB

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
spec:
  rules:
    - http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-service
                port:
                  number: 80
```

## Storage

### EBS Volumes

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  fsType: ext4
volumeBindingMode: WaitForFirstConsumer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp3
  resources:
    requests:
      storage: 10Gi
```

## Cluster Autoscaler

```bash
# Deploy Cluster Autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Edit deployment
kubectl -n kube-system edit deployment cluster-autoscaler
# Add: --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/my-cluster
```

## Monitoring

### CloudWatch Container Insights

```bash
# Enable Container Insights
eksctl utils update-cluster-logging \
    --cluster my-cluster \
    --enable-types all \
    --approve

# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluent-bit-quickstart.yaml
```

## Cluster Upgrades

```bash
# Upgrade control plane
eksctl upgrade cluster --name my-cluster --version 1.30

# Upgrade node groups
eksctl upgrade nodegroup \
    --cluster my-cluster \
    --name my-nodes \
    --kubernetes-version 1.30
```

## Common Commands

```bash
# Cluster management
eksctl get cluster
eksctl delete cluster --name my-cluster

# Node groups
eksctl get nodegroup --cluster my-cluster
eksctl scale nodegroup --cluster my-cluster --name ng-1 --nodes 5

# Logs
eksctl utils update-cluster-logging --cluster my-cluster --enable-types all

# Debugging
kubectl get events -A
kubectl describe node <node-name>
kubectl logs -n kube-system <pod-name>
```

## Best Practices

1. **Use managed node groups** - Easier upgrades and management
2. **Enable IRSA** - Least privilege for pods
3. **Use multiple AZs** - High availability
4. **Enable cluster logging** - Audit and troubleshooting
5. **Use Cluster Autoscaler** - Cost optimization
6. **Regular upgrades** - Security and features

## Links

- Documentation: https://docs.aws.amazon.com/eks/
- eksctl: https://eksctl.io/
- Best Practices: https://aws.github.io/aws-eks-best-practices/
