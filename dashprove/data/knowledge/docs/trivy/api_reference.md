# Overview

## trivy[¶][1]

Unified security scanner

### Synopsis[¶][2]

Scanner for vulnerabilities in container images, file systems, and Git repositories, as well as for
configuration issues and hard-coded secrets

`trivy [global flags] command [flags] target
`

### Examples[¶][3]

`  # Scan a container image
  $ trivy image python:3.4-alpine

  # Scan a container image from a tar archive
  $ trivy image --input ruby-3.1.tar

  # Scan local filesystem
  $ trivy fs .

  # Run in server mode
  $ trivy server
`

### Options[¶][4]

`      --cache-dir string          cache directory (default "/path/to/cache")
  -c, --config string             config path (default "trivy.yaml")
  -d, --debug                     debug mode
  -f, --format string             version format (json)
      --generate-default-config   write the default config to trivy-default.yaml
  -h, --help                      help for trivy
      --insecure                  allow insecure server connections
  -q, --quiet                     suppress progress bar and log output
      --timeout duration          timeout (default 5m0s)
  -v, --version                   show version
`

### SEE ALSO[¶][5]

* [trivy clean][6] - Remove cached files
* [trivy config][7] - Scan config files for misconfigurations
* [trivy convert][8] - Convert Trivy JSON report into a different format
* [trivy filesystem][9] - Scan local filesystem
* [trivy image][10] - Scan a container image
* [trivy kubernetes][11] - [EXPERIMENTAL] Scan kubernetes cluster
* [trivy module][12] - Manage modules
* [trivy plugin][13] - Manage plugins
* [trivy registry][14] - Manage registry authentication
* [trivy repository][15] - Scan a repository
* [trivy rootfs][16] - Scan rootfs
* [trivy sbom][17] - Scan SBOM for vulnerabilities and licenses
* [trivy server][18] - Server mode
* [trivy version][19] - Print the version
* [trivy vex][20] - [EXPERIMENTAL] VEX utilities
* [trivy vm][21] - [EXPERIMENTAL] Scan a virtual machine image

[1]: #trivy
[2]: #synopsis
[3]: #examples
[4]: #options
[5]: #see-also
[6]: ../trivy_clean/
[7]: ../trivy_config/
[8]: ../trivy_convert/
[9]: ../trivy_filesystem/
[10]: ../trivy_image/
[11]: ../trivy_kubernetes/
[12]: ../trivy_module/
[13]: ../trivy_plugin/
[14]: ../trivy_registry/
[15]: ../trivy_repository/
[16]: ../trivy_rootfs/
[17]: ../trivy_sbom/
[18]: ../trivy_server/
[19]: ../trivy_version/
[20]: ../trivy_vex/
[21]: ../trivy_vm/
