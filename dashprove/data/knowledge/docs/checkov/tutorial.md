* [Docs][1]
* 2.basics
* Installing Checkov
[ Edit on GitHub ][2]

Installing Checkov is quick and straightforward—just install, configure input, and scan.

### Install From PyPi Using Pip

`pip install checkov
`

or

`pip3 install checkov
`

### Install on Alpine

In general, it is not recommended to use Alpine with larger Python projects, like Checkov, because
of incompatible C extensions. Currently, Checkov can only be installed on Alpine with Python 3.11+,
but it is not officially tested or supported.

`pip3 install --upgrade pip && pip3 install --upgrade setuptools
pip3 install checkov
`

### Install with Homebrew

`brew install checkov
`

### Install in a virtual environment

For environments like Debian 12, it’s recommended to use a Python virtual environment:

**Create and Activate Virtual Environment**:

`python3 -m venv /path/to/venv/checkov
cd /path/to/venv/checkov
source ./bin/activate
`

**Install Checkov**:

`pip install checkov
`

**Optional: Create Symlink for Easy Access**:

`sudo ln -s /path/to/venv/checkov/bin/checkov /usr/local/bin/checkov
`

## Upgrading Checkov

If you installed Checkov with pip3, use the following command to upgrade:

`pip3 install -U checkov
`

or with Homebrew

`brew upgrade checkov
`

## Configure an input folder or file

### Configure a folder

`checkov --directory /user/path/to/iac/code
`

### Configure a specific file

`checkov --file /user/tf/example.tf
`

### Configure Multiple Specific Files

`checkov -f /user/cloudformation/example1.yml -f /user/cloudformation/example2.yml
`

### Configure a Terraform Plan file in JSON

`terraform init
terraform plan -out tf.plan
terraform show -json tf.plan  > tf.json 
checkov -f tf.json
`

Note: The Terraform show output file `tf.json` will be a single line. For that reason Checkov will
report all findings as line number 0.

`check: CKV_AWS_21: "Ensure all data stored in the S3 bucket have versioning enabled"
        FAILED for resource: aws_s3_bucket.customer
        File: /tf/tf.json:0-0
        Guide: https://docs.prismacloud.io/en/enterprise-edition/policy-reference/aws-policies/s3-po
licies/s3-16-enable-versioning
`

If you have installed jq, you can convert a JSON file into multiple lines with the command
`terraform show -json tf.plan | jq '.' > tf.json`, making it easier to read the scan result. NOTE:
`jq` is required to show the code block as seen below.

`checkov -f tf.json
Check: CKV_AWS_21: "Ensure all data stored in the S3 bucket have versioning enabled"
        FAILED for resource: aws_s3_bucket.customer
        File: /tf/tf1.json:224-268
        Guide: https://docs.prismacloud.io/en/enterprise-edition/policy-reference/aws-policies/s3-po
licies/s3-16-enable-versioning

                225 |               "values": {
                226 |                 "acceleration_status": "",
                227 |                 "acl": "private",
                228 |                 "arn": "arn:aws:s3:::mybucket",
`

[1]: https://www.checkov.io
[2]: https://github.com/bridgecrewio/checkov//blob/master/docs/2.Basics/Installing Checkov.md
