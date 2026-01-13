v1.14.x (latest)

* Terraform
* [v1.15.x (alpha)][1]
* [v1.13.x][2]
* [v1.12.x][3]
* [v1.11.x][4]
* [v1.10.x][5]
* [v1.9.x][6]
* [v1.8.x][7]
* [v1.7.x][8]
* [v1.6.x][9]
* [v1.5.x][10]
* [v1.4.x][11]
* [v1.3.x][12]
* [v1.2.x][13]
* [v1.1.x][14]

# Terraform Language Documentation

Use the Terraform configuration language to describe the infrastructure that Terraform manages.

This is the documentation for Terraform's configuration language. It is relevant to users of
[Terraform CLI][15], [HCP Terraform][16], and [Terraform Enterprise][17]. Terraform's language is
its primary user interface. Configuration files you write in Terraform language tell Terraform what
plugins to install, what infrastructure to create, and what data to fetch. Terraform language also
lets you define dependencies between resources and create multiple similar resources from a single
configuration block.

> **Hands-on:** Try the [Write Terraform Configuration][18] tutorials.

## About the Terraform Language

The main purpose of the Terraform language is declaring [resources][19], which represent
infrastructure objects. All other language features exist only to make the definition of resources
more flexible and convenient.

A *Terraform configuration* is a complete document in the Terraform language that tells Terraform
how to manage a given collection of infrastructure. A configuration can consist of multiple files
and directories.

The syntax of the Terraform language consists of only a few basic elements:

`resource "aws_vpc" "main" {
  cidr_block = var.base_cidr_block
}

<BLOCK TYPE> "<BLOCK LABEL>" "<BLOCK LABEL>" {
  # Block body
  <IDENTIFIER> = <EXPRESSION> # Argument
}
`

* *Blocks* are containers for other content and usually represent the configuration of some kind of
  object, like a resource. Blocks have a *block type,* can have zero or more *labels,* and have a
  *body* that contains any number of arguments and nested blocks. Most of Terraform's features are
  controlled by top-level blocks in a configuration file.
* *Arguments* assign a value to a name. They appear within blocks.
* *Expressions* represent a value, either literally or by referencing and combining other values.
  They appear as values for arguments, or within other expressions.

The Terraform language is declarative, describing an intended goal rather than the steps to reach
that goal. The ordering of blocks and the files they are organized into are generally not
significant; Terraform only considers implicit and explicit relationships between resources when
determining an order of operations.

### Example

The following example describes a simple network topology for Amazon Web Services, just to give a
sense of the overall structure and syntax of the Terraform language. Similar configurations can be
created for other virtual network services, using resource types defined by other providers, and a
practical network configuration will often contain additional elements not shown here.

`terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 1.0.4"
    }
  }
}

variable "aws_region" {}

variable "base_cidr_block" {
  description = "A /16 CIDR range definition, such as 10.1.0.0/16, that the VPC will use"
  default = "10.1.0.0/16"
}

variable "availability_zones" {
  description = "A list of availability zones in which to create subnets"
  type = list(string)
}

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "main" {
  # Referencing the base_cidr_block variable allows the network address
  # to be changed without modifying the configuration.
  cidr_block = var.base_cidr_block
}

resource "aws_subnet" "az" {
  # Create one subnet for each given availability zone.
  count = length(var.availability_zones)

  # For each subnet, use one of the specified availability zones.
  availability_zone = var.availability_zones[count.index]

  # By referencing the aws_vpc.main object, Terraform knows that the subnet
  # must be created only after the VPC is created.
  vpc_id = aws_vpc.main.id

  # Built-in functions and operators can be used for simple transformations of
  # values, such as computing a subnet address. Here we create a /20 prefix for
  # each subnet, using consecutive addresses for each availability zone,
  # such as 10.1.16.0/20 .
  cidr_block = cidrsubnet(aws_vpc.main.cidr_block, 4, count.index+1)
}
`
[Edit this page on GitHub][20]

[1]: /terraform/language/v1.15.x
[2]: /terraform/language/v1.13.x
[3]: /terraform/language/v1.12.x
[4]: /terraform/language/v1.11.x
[5]: /terraform/language/v1.10.x
[6]: /terraform/language/v1.9.x
[7]: /terraform/language/v1.8.x
[8]: /terraform/language/v1.7.x
[9]: /terraform/language/v1.6.x
[10]: /terraform/language/v1.5.x
[11]: /terraform/language/v1.4.x
[12]: /terraform/language/v1.3.x
[13]: /terraform/language/v1.2.x
[14]: /terraform/language/v1.1.x
[15]: /terraform/cli
[16]: https://cloud.hashicorp.com/products/terraform
[17]: /terraform/enterprise
[18]: /terraform/tutorials/configuration-language
[19]: /terraform/language/resources
[20]: https://github.com/hashicorp/web-unified-docs/blob/main/content/terraform/v1.14.x/docs/languag
e/index.mdx
