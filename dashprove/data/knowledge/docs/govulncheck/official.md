## Documentation [¶][1]

### Overview [¶][2]

* [Usage][3]
* [Integrations][4]
* [Exit codes][5]
* [Limitations][6]
* [Feedback][7]

Govulncheck reports known vulnerabilities that affect Go code. It uses static analysis of source
code or a binary's symbol table to narrow down reports to only those that could affect the
application.

By default, govulncheck makes requests to the Go vulnerability database at [https://vuln.go.dev][8].
Requests to the vulnerability database contain only module paths with vulnerabilities already known
to the database, not code or other properties of your program. See
[https://vuln.go.dev/privacy.html][9] for more. Use the -db flag to specify a different database,
which must implement the specification at [https://go.dev/security/vuln/database][10].

Govulncheck looks for vulnerabilities in Go programs using a specific build configuration. For
analyzing source code, that configuration is the Go version specified by the “go” command found on
the PATH. For binaries, the build configuration is the one used to build the binary. Note that
different build configurations may have different known vulnerabilities.

#### Usage [¶][11]

To analyze source code, run govulncheck from the module directory, using the same package path
syntax that the go command uses:

$ cd my-module
$ govulncheck ./...

If no vulnerabilities are found, govulncheck will display a short message. If there are
vulnerabilities, each is displayed briefly, with a summary of a call stack. The summary shows in
brief how the package calls a vulnerable function. For example, it might say

main.go:[line]:[column]: mypackage.main calls golang.org/x/text/language.Parse

To control which files are processed, use the -tags flag to provide a comma-separated list of build
tags, and the -test flag to indicate that test files should be included.

To include more detailed stack traces, pass '-show traces', this will cause it to print the full
call stack for each entry.

To include progress messages and more details on findings, pass '-show verbose'.

To run govulncheck on a compiled binary, pass it the path to the binary file with the '-mode binary'
flag:

$ govulncheck -mode binary $HOME/go/bin/my-go-program

Govulncheck uses the binary's symbol information to find mentions of vulnerable functions. These
functions can belong to binary's transitive dependencies and also the main module of the binary. The
latter functions are checked for only when the precise version of the binary module is known.
Govulncheck output on binaries omits call stacks, which require source code analysis.

Govulncheck also supports '-mode extract' on a Go binary for extraction of minimal information
needed to analyze the binary. This will produce a blob, typically much smaller than the binary, that
can also be passed to govulncheck as an argument with '-mode binary'. The users should not rely on
the contents or representation of the blob.

#### Integrations [¶][12]

Govulncheck supports streaming JSON. For more details, please see
[golang.org/x/vuln/internal/govulncheck][13].

Govulncheck also supports Static Analysis Results Interchange Format (SARIF) output format,
following the specification at
[https://www.oasis-open.org/committees/tc_home.php?wg_abbrev=sarif][14]. For more details, please
see [golang.org/x/vuln/internal/sarif][15].

Govulncheck supports the Vulnerability EXchange (VEX) output format, following the specification at
[https://github.com/openvex/spec][16]. For more details, please see
[golang.org/x/vuln/internal/openvex][17].

#### Exit codes [¶][18]

Govulncheck exits successfully (exit code 0) if there are no vulnerabilities, and exits
unsuccessfully if there are. It also exits successfully if the 'format -json' ('-json'), '-format
sarif', or '-format openvex' is provided, regardless of the number of detected vulnerabilities.

#### Limitations [¶][19]

Govulncheck has these limitations:

* Govulncheck analyzes function pointer and interface calls conservatively, which may result in
  false positives or inaccurate call stacks in some cases.
* Calls to functions made using package reflect are not visible to static analysis. Vulnerable code
  reachable only through those calls will not be reported in source scan mode. Similarly, use of the
  unsafe package may result in false negatives.
* Because Go binaries do not contain detailed call information, govulncheck cannot show the call
  graphs for detected vulnerabilities. It may also report false positives for code that is in the
  binary but unreachable.
* There is no support for silencing vulnerability findings. See [https://go.dev/issue/61211][20] for
  updates.
* Govulncheck reports only standard library vulnerabilities for binaries built with Go versions
  prior to Go 1.18.
* For binaries where the symbol information cannot be extracted, govulncheck reports vulnerabilities
  for all modules on which the binary depends.

#### Feedback [¶][21]

To share feedback, see [https://go.dev/security/vuln#feedback][22].

## Source Files [¶][23]

[View all Source files][24]

* [doc.go][25]
* [gotypesalias.go][26]
* [main.go][27]
* [test_utils.go][28]

## Directories [¶][29]

Show internal Expand all

────────────────────────────────────┬────────
Path                                │Synopsis
────────────────────────────────────┼────────
integration                         │        
────────────────────────────────────┼────────
[internal/integration][30]          │        
────────────────────────────────────┼────────
[k8s][31] command                   │        
────────────────────────────────────┼────────
[stackrox-scanner][32] command      │        
────────────────────────────────────┴────────
Click to show internal directories.
Click to hide internal directories.

[1]: #section-documentation
[2]: #pkg-overview
[3]: #hdr-Usage
[4]: #hdr-Integrations
[5]: #hdr-Exit_codes
[6]: #hdr-Limitations
[7]: #hdr-Feedback
[8]: https://vuln.go.dev
[9]: https://vuln.go.dev/privacy.html
[10]: https://go.dev/security/vuln/database
[11]: #hdr-Usage
[12]: #hdr-Integrations
[13]: /golang.org/x/vuln@v1.1.4/internal/govulncheck
[14]: https://www.oasis-open.org/committees/tc_home.php?wg_abbrev=sarif
[15]: /golang.org/x/vuln@v1.1.4/internal/sarif
[16]: https://github.com/openvex/spec
[17]: /golang.org/x/vuln@v1.1.4/internal/openvex
[18]: #hdr-Exit_codes
[19]: #hdr-Limitations
[20]: https://go.dev/issue/61211
[21]: #hdr-Feedback
[22]: https://go.dev/security/vuln#feedback
[23]: #section-sourcefiles
[24]: https://go.googlesource.com/vuln/+/v1.1.4/cmd/govulncheck
[25]: https://go.googlesource.com/vuln/+/v1.1.4/cmd/govulncheck/doc.go
[26]: https://go.googlesource.com/vuln/+/v1.1.4/cmd/govulncheck/gotypesalias.go
[27]: https://go.googlesource.com/vuln/+/v1.1.4/cmd/govulncheck/main.go
[28]: https://go.googlesource.com/vuln/+/v1.1.4/cmd/govulncheck/test_utils.go
[29]: #section-directories
[30]: /golang.org/x/vuln@v1.1.4/cmd/govulncheck/integration/internal/integration
[31]: /golang.org/x/vuln@v1.1.4/cmd/govulncheck/integration/k8s
[32]: /golang.org/x/vuln@v1.1.4/cmd/govulncheck/integration/stackrox-scanner
