# Kubeval

**NOTE: This project is [no longer maintained][1], a good replacement is [kubeconform][2]**

`kubeval` is a tool for validating a Kubernetes YAML or JSON configuration file. It does so using
schemas generated from the Kubernetes OpenAPI specification, and therefore can validate schemas for
multiple versions of Kubernetes.

[[CircleCI]][3] [[Go Report Card]][4] [[GoDoc]][5]

`$ kubeval my-invalid-rc.yaml
WARN - fixtures/my-invalid-rc.yaml contains an invalid ReplicationController - spec.replicas: Invali
d type. Expected: [integer,null], given: string
$ echo $?
1
`

For full usage and installation instructions see [kubeval.com][6].

[1]: https://github.com/instrumenta/kubeval/issues/268#issuecomment-902128481
[2]: https://github.com/yannh/kubeconform
[3]: https://circleci.com/gh/instrumenta/kubeval
[4]: https://goreportcard.com/report/github.com/instrumenta/kubeval
[5]: https://godoc.org/github.com/instrumenta/kubeval
[6]: https://kubeval.com/
