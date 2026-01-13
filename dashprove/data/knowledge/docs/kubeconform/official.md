[[Kubeconform-GitHub-Hero]][1]

[[Build status]][2] [[Homebrew]][3] [[Go Report card]][4] [[PkgGoDev]][5]

`Kubeconform` is a Kubernetes manifest validation tool. Incorporate it into your CI, or use it
locally to validate your Kubernetes configuration!

It is inspired by, contains code from and is designed to stay close to [Kubeval][6], but with the
following improvements:

* **high performance**: will validate & download manifests over multiple routines, caching
  downloaded files in memory
* configurable list of **remote, or local schemas locations**, enabling validating Kubernetes custom
  resources (CRDs) and offline validation capabilities
* uses by default a [self-updating fork][7] of the schemas registry maintained by the
  kubernetes-json-schema project - which guarantees up-to-date **schemas for all recent versions of
  Kubernetes**.

#### Speed comparison with Kubeval

Running on a pretty large kubeconfigs setup, on a laptop with 4 cores:

$ time kubeconform -ignore-missing-schemas -n 8 -summary  preview staging production
Summary: 50714 resources found in 35139 files - Valid: 27334, Invalid: 0, Errors: 0 Skipped: 23380
real    0m6,710s
user    0m38,701s
sys     0m1,161s
$ time kubeval -d preview,staging,production --ignore-missing-schemas --quiet
[... Skipping output]
real    0m35,336s
user    0m0,717s
sys     0m1,069s

## Table of contents

* [A small overview of Kubernetes manifest validation][8]
  
  * [Limits of Kubeconform validation][9]
* [Installation][10]
* [Usage][11]
  
  * [Usage examples][12]
  * [Proxy support][13]
* [Overriding schemas location][14]
  
  * [CustomResourceDefinition (CRD) Support][15]
  * [OpenShift schema Support][16]
* [Integrating Kubeconform in the CI][17]
  
  * [Github Workflow][18]
  * [Gitlab-CI][19]
* [Helm charts][20]
* [Using kubeconform as a Go Module][21]
* [Credits][22]

## A small overview of Kubernetes manifest validation

Kubernetes's API is described using the [OpenAPI (formerly swagger) specification][23], in a
[file][24] checked into the main Kubernetes repository.

Because of the state of the tooling to perform validation against OpenAPI schemas, projects usually
convert the OpenAPI schemas to [JSON schemas][25] first. Kubeval relies on
[instrumenta/OpenApi2JsonSchema][26] to convert Kubernetes' Swagger file and break it down into
multiple JSON schemas, stored in github at [instrumenta/kubernetes-json-schema][27] and published on
[kubernetesjsonschema.dev][28].

`Kubeconform` relies on [a fork of kubernetes-json-schema][29] that is more meticulously kept
up-to-date, and contains schemas for all recent versions of Kubernetes.

### Limits of Kubeconform validation

`Kubeconform`, similar to `kubeval`, only validates manifests using the official Kubernetes OpenAPI
specifications. The Kubernetes controllers still perform additional server-side validations that are
not part of the OpenAPI specifications. Those server-side validations are not covered by
`Kubeconform` (examples: [#65][30], [#122][31], [#142][32]). You can use a 3rd-party tool or the
`kubectl --dry-run=server` command to fill the missing (validation) gap.

## Installation

If you are a [Homebrew][33] user, you can install by running:

$ brew install kubeconform

If you are a Windows user, you can install with [winget][34] by running:

winget install YannHamon.kubeconform

You can also download the latest version from the [release page][35].

Another way of installation is via Golang's package manager:

# With a specific version tag
$ go install github.com/yannh/kubeconform/cmd/kubeconform@v0.4.13

# Latest version
$ go install github.com/yannh/kubeconform/cmd/kubeconform@latest

## Usage

`$ kubeconform -h
Usage: kubeconform [OPTION]... [FILE OR FOLDER]...
  -cache string
        cache schemas downloaded via HTTP to this folder
  -debug
        print debug information
  -exit-on-error
        immediately stop execution when the first error is encountered
  -h    show help information
  -ignore-filename-pattern value
        regular expression specifying paths to ignore (can be specified multiple times)
  -ignore-missing-schemas
        skip files with missing schemas instead of failing
  -insecure-skip-tls-verify
        disable verification of the server's SSL certificate. This will make your HTTPS connections 
insecure
  -kubernetes-version string
        version of Kubernetes to validate against, e.g.: 1.18.0 (default "master")
  -n int
        number of goroutines to run concurrently (default 4)
  -output string
        output format - json, junit, pretty, tap, text (default "text")
  -reject string
        comma-separated list of kinds or GVKs to reject
  -schema-location value
        override schemas location search path (can be specified multiple times)
  -skip string
        comma-separated list of kinds or GVKs to ignore
  -strict
        disallow additional properties not in schema or duplicated keys
  -summary
        print a summary at the end (ignored for junit output)
  -v    show version information
  -verbose
        print results for all resources (ignored for tap and junit output)
`

### Usage examples

* Validating a single, valid file
$ kubeconform fixtures/valid.yaml
$ echo $?
0

* Validating a single invalid file, setting output to json, and printing a summary
$ kubeconform -summary -output json fixtures/invalid.yaml
{
  "resources": [
    {
      "filename": "fixtures/invalid.yaml",
      "kind": "ReplicationController",
      "version": "v1",
      "status": "INVALID",
      "msg": "Additional property templates is not allowed - Invalid type. Expected: [integer,null],
 given: string"
    }
  ],
  "summary": {
    "valid": 0,
    "invalid": 1,
    "errors": 0,
    "skipped": 0
  }
}
$ echo $?
1

* Passing manifests via Stdin
cat fixtures/valid.yaml  | ./bin/kubeconform -summary
Summary: 1 resource found parsing stdin - Valid: 1, Invalid: 0, Errors: 0 Skipped: 0

* Validating a file, ignoring its resource using both Kind, and GVK (Group, Version, Kind) notations
`# This will ignore ReplicationController for all apiVersions
$ kubeconform -summary -skip ReplicationController fixtures/valid.yaml
Summary: 1 resource found in 1 file - Valid: 0, Invalid: 0, Errors: 0, Skipped: 1

# This will ignore ReplicationController only for apiVersion v1
$ kubeconform -summary -skip v1/ReplicationController fixtures/valid.yaml
Summary: 1 resource found in 1 file - Valid: 0, Invalid: 0, Errors: 0, Skipped: 1
`

* Validating a folder, increasing the number of parallel workers
`$ kubeconform -summary -n 16 fixtures
fixtures/crd_schema.yaml - CustomResourceDefinition trainingjobs.sagemaker.aws.amazon.com failed val
idation: could not find schema for CustomResourceDefinition
fixtures/invalid.yaml - ReplicationController bob is invalid: Invalid type. Expected: [integer,null]
, given: string
[...]
Summary: 65 resources found in 34 files - Valid: 55, Invalid: 2, Errors: 8 Skipped: 0
`

### Proxy support

`Kubeconform` will respect the **HTTPS_PROXY** variable when downloading schema files.

$ HTTPS_PROXY=proxy.local bin/kubeconform fixtures/valid.yaml

## Overriding schemas location

When the `-schema-location` parameter is not used, or set to `default`, kubeconform will default to
downloading schemas from [https://github.com/yannh/kubernetes-json-schema][36]. Kubeconform however
supports passing one, or multiple, schemas locations - HTTP(s) URLs, or local filesystem paths, in
which case it will lookup for schema definitions in each of them, in order, stopping as soon as a
matching file is found.

* If the `-schema-location` value does not end with `.json`, Kubeconform will assume filenames / a
  file structure identical to that of [kubernetesjsonschema.dev][37] or
  [yannh/kubernetes-json-schema][38].
* if the `-schema-location` value ends with `.json` - Kubeconform assumes the value is a **Go
  templated string** that indicates how to search for JSON schemas.
* the `-schema-location` value of `default` is an alias for
  `https://raw.githubusercontent.com/yannh/kubernetes-json-schema/master/{{.NormalizedKubernetesVers
  ion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json`.

**The following command lines are equivalent:**

$ kubeconform fixtures/valid.yaml
$ kubeconform -schema-location default fixtures/valid.yaml
$ kubeconform -schema-location 'https://raw.githubusercontent.com/yannh/kubernetes-json-schema/maste
r/{{.NormalizedKubernetesVersion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json
' fixtures/valid.yaml

Here are the variables you can use in -schema-location:

* *NormalizedKubernetesVersion* - Kubernetes Version, prefixed by v
* *StrictSuffix* - "-strict" or "" depending on whether validation is running in strict mode or not
* *ResourceKind* - Kind of the Kubernetes Resource
* *ResourceAPIVersion* - Version of API used for the resource - "v1" in "apiVersion:
  monitoring.coreos.com/v1"
* *Group* - the group name as stated in this resource's definition - "monitoring.coreos.com" in
  "apiVersion: monitoring.coreos.com/v1"
* *KindSuffix* - suffix computed from apiVersion - for compatibility with `Kubeval` schema
  registries

### CustomResourceDefinition (CRD) Support

Because Custom Resources (CR) are not native Kubernetes objects, they are not included in the
default schema.
If your CRs are present in [Datree's CRDs-catalog][39], you can specify this project as an
additional registry to lookup:

# Look in the CRDs-catalog for the desired schema/s
$ kubeconform -schema-location default -schema-location 'https://raw.githubusercontent.com/datreeio/
CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' [MANIFEST]

If your CRs are not present in the CRDs-catalog, you will need to manually pull the CRDs manifests
from your cluster and convert the `OpenAPI.spec` to JSON schema format.

Converting an OpenAPI file to a JSON Schema


`Kubeconform` uses JSON schemas to validate Kubernetes resources. For Custom Resource, the
CustomResourceDefinition first needs to be converted to JSON Schema. A script is provided to convert
these CustomResourceDefinitions to JSON schema. Here is an example how to use it:

$ python ./scripts/openapi2jsonschema.py https://raw.githubusercontent.com/aws/amazon-sagemaker-oper
ator-for-k8s/master/config/crd/bases/sagemaker.aws.amazon.com_trainingjobs.yaml
JSON schema written to trainingjob_v1.json

By default, the file name output format is `{kind}_{version}`. The `FILENAME_FORMAT` environment
variable can be used to change the output file name (Available variables: `kind`, `group`,
`fullgroup`, `version`):

`$ export FILENAME_FORMAT='{kind}-{group}-{version}'
$ ./scripts/openapi2jsonschema.py https://raw.githubusercontent.com/aws/amazon-sagemaker-operator-fo
r-k8s/master/config/crd/bases/sagemaker.aws.amazon.com_trainingjobs.yaml
JSON schema written to trainingjob-sagemaker-v1.json

$ export FILENAME_FORMAT='{kind}-{fullgroup}-{version}'
$ ./scripts/openapi2jsonschema.py https://raw.githubusercontent.com/aws/amazon-sagemaker-operator-fo
r-k8s/master/config/crd/bases/sagemaker.aws.amazon.com_trainingjobs.yaml
JSON schema written to trainingjob-sagemaker.aws.amazon.com-v1.json
`

After converting your CRDs to JSON schema files, you can use `kubeconform` to validate your CRs
against them:

`# If the resource Kind is not found in default, also lookup in the schemas/ folder for a matching f
ile
$ kubeconform -schema-location default -schema-location 'schemas/{{ .ResourceKind }}{{ .KindSuffix }
}.json' fixtures/custom-resource.yaml
`

ℹ️ Datree's [CRD Extractor][40] is a utility that can be used instead of this manual process.

### OpenShift schema Support

You can validate Openshift manifests using a custom schema location. Set the OpenShift version
(v3.10.0-4.1.0) to validate against using `-kubernetes-version`.

`kubeconform -kubernetes-version 3.8.0  -schema-location 'https://raw.githubusercontent.com/garethr/
openshift-json-schema/master/{{ .NormalizedKubernetesVersion }}-standalone{{ .StrictSuffix }}/{{ .Re
sourceKind }}.json'  -summary fixtures/valid.yaml
Summary: 1 resource found in 1 file - Valid: 1, Invalid: 0, Errors: 0 Skipped: 0
`

## Integrating Kubeconform in the CI

`Kubeconform` publishes Docker Images to Github's new Container Registry (ghcr.io). These images can
be used directly in a Github Action, once logged in using a [*Github Token*][41].

### Github Workflow

Example:

name: kubeconform
on: push
jobs:
  kubeconform:
    runs-on: ubuntu-latest
    steps:
      - name: login to Github Packages
        run: echo "${{ github.token }}" | docker login https://ghcr.io -u ${GITHUB_ACTOR} --password
-stdin
      - uses: actions/checkout@v2
      - uses: docker://ghcr.io/yannh/kubeconform:latest
        with:
          entrypoint: '/kubeconform'
          args: "-summary -output json kubeconfigs/"

*Note on pricing*: Kubeconform relies on Github Container Registry which is currently in Beta.
During that period, [bandwidth is free][42]. After that period, bandwidth costs might be applicable.
Since bandwidth from Github Packages within Github Actions is free, I expect Github Container
Registry to also be usable for free within Github Actions in the future. If that were not to be the
case, I might publish the Docker image to a different platform.

### Gitlab-CI

The Kubeconform Docker image can be used in Gitlab-CI. Here is an example of a Gitlab-CI job:

lint-kubeconform:
  stage: validate
  image:
    name: ghcr.io/yannh/kubeconform:latest-alpine
    entrypoint: [""]
  script:
  - /kubeconform -summary -output json kubeconfigs/

See [issue 106][43] for more details.

## Helm charts

There is a 3rd party [repository][44] that allows to use `kubeconform` to test [Helm charts][45] in
the form of a [Helm plugin][46] and [`pre-commit` hook][47].

## Using kubeconform as a Go Module

**Warning**: This is a work-in-progress, the interface is not yet considered stable. Feedback is
encouraged.

`Kubeconform` contains a package that can be used as a library. An example of usage can be found in
[examples/main.go][48]

Additional documentation on [pkg.go.dev][49]

## Credits

* @garethr for the [Kubeval][50] and [kubernetes-json-schema][51] projects ❤️

[1]: https://user-images.githubusercontent.com/19731161/142411871-f695e40c-bfa8-43ca-97c0-94c2567497
32.png
[2]: https://github.com/yannh/kubeconform/actions?query=branch%3Amaster
[3]: https://formulae.brew.sh/formula/kubeconform
[4]: https://goreportcard.com/report/github.com/yannh/kubeconform
[5]: https://pkg.go.dev/github.com/yannh/kubeconform/pkg/validator
[6]: https://github.com/instrumenta/kubeval
[7]: https://github.com/yannh/kubernetes-json-schema
[8]: #a-small-overview-of-kubernetes-manifest-validation
[9]: #Limits-of-Kubeconform-validation
[10]: #Installation
[11]: #Usage
[12]: #Usage-examples
[13]: #Proxy-support
[14]: #Overriding-schemas-location
[15]: #CustomResourceDefinition-CRD-Support
[16]: #OpenShift-schema-Support
[17]: #Integrating-Kubeconform-in-the-CI
[18]: #Github-Workflow
[19]: #Gitlab-CI
[20]: #helm-charts
[21]: #Using-kubeconform-as-a-Go-Module
[22]: #Credits
[23]: https://www.openapis.org
[24]: https://github.com/kubernetes/kubernetes/blob/master/api/openapi-spec/swagger.json
[25]: https://json-schema.org/
[26]: https://github.com/instrumenta/openapi2jsonschema
[27]: https://github.com/instrumenta/kubernetes-json-schema
[28]: https://kubernetesjsonschema.dev/
[29]: https://github.com/yannh/kubernetes-json-schema/
[30]: https://github.com/yannh/kubeconform/issues/65
[31]: https://github.com/yannh/kubeconform/issues/122
[32]: https://github.com/yannh/kubeconform/issues/142
[33]: https://brew.sh/
[34]: https://learn.microsoft.com/en-us/windows/package-manager/winget/
[35]: https://github.com/yannh/kubeconform/releases
[36]: https://github.com/yannh/kubernetes-json-schema
[37]: https://kubernetesjsonschema.dev/
[38]: https://github.com/yannh/kubernetes-json-schema
[39]: https://github.com/datreeio/CRDs-catalog
[40]: https://github.com/datreeio/CRDs-catalog#crd-extractor
[41]: https://github.blog/changelog/2021-03-24-packages-container-registry-now-supports-github_token
/
[42]: https://docs.github.com/en/packages/guides/about-github-container-registry
[43]: https://github.com/yannh/kubeconform/issues/106
[44]: https://github.com/jtyr/kubeconform-helm
[45]: https://helm.sh
[46]: https://helm.sh/docs/topics/plugins/
[47]: https://pre-commit.com/
[48]: /yannh/kubeconform/blob/master/examples/main.go
[49]: https://pkg.go.dev/github.com/yannh/kubeconform/pkg/validator
[50]: https://github.com/instrumenta/kubeval
[51]: https://github.com/instrumenta/kubernetes-json-schema
