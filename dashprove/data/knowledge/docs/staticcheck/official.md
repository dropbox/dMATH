1. [Documentation][1]

# Welcome to Staticcheck

Staticcheck is a state of the art linter for the [Go programming language][2]. Using static
analysis, it finds bugs and performance issues, offers simplifications, and enforces style rules.

Each of the [150+][3] checks has been designed to be fast, precise and useful. When Staticcheck
flags code, you can be sure that it isn’t wasting your time with unactionable warnings. Unlike many
other linters, Staticcheck focuses on checks that produce few to no false positives. It’s the ideal
candidate for running in CI without risking spurious failures.

Staticcheck aims to be trivial to adopt. It behaves just like the official `go` tool and requires no
learning to get started with. Just run `staticcheck ./...` on your code in addition to `go vet
./...`.

While checks have been designed to be useful out of the box, they still provide [configuration][4]
where necessary, to fine-tune to your needs, without overwhelming you with hundreds of options.

Staticcheck can be used from the command line, in CI, and even [directly from your editor][5].

Staticcheck is open source and offered completely free of charge. [Sponsors][6] guarantee its
continued development.

[Getting started][7]

Quickly get started using Staticcheck

[Running Staticcheck][8]
[Configuration][9]

Tweak Staticcheck to your requirements

[Checks][10]

Explanations for all checks in Staticcheck

[Frequently Asked Questions][11]
[Release notes][12]

[1]: https://staticcheck.dev/docs/
[2]: https://go.dev/
[3]: /docs/checks/
[4]: /docs/configuration/
[5]: https://github.com/golang/tools/blob/master/gopls/doc/settings.md#staticcheck-bool
[6]: /sponsors/
[7]: /docs/getting-started/
[8]: /docs/running-staticcheck/
[9]: /docs/configuration/
[10]: /docs/checks/
[11]: /docs/faq/
[12]: /changes/
