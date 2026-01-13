# cpplint - static code checker for C++

Cpplint is a command-line tool to check C/C++ files for style issues according to [Google's C++
style guide][1].

Cpplint used to be developed and maintained by Google Inc. at [google/styleguide][2]. Nowadays,
[Google is no longer maintaining the public version of cpplint][3], and pretty much everything in
their repo's PRs and issues about cpplint have gone unimplemented.

This fork aims to update cpplint to modern specifications, and be (somewhat) more open to adding
fixes and features to make cpplint usable in wider contexts.

## Installation

Use [pipx]([https://pipx.pypa.io][4]) to install cpplint from PyPI, run:

$ pipx install cpplint

### Usage

$ cpplint [OPTIONS] files

For full usage instructions, run:

$ cpplint --help

cpplint can also be run as a pre-commit hook by adding to .pre-commit-config.yaml:

- repo: https://github.com/cpplint/cpplint
  rev: 2.0.0
  hooks:
    - id: cpplint
      args:
        - --filter=-whitespace/line_length,-whitespace/parens

## Changes

* python 3 compatibility
* more default file extensions
* customizable file extensions with the --extensions argument
* continuous integration on github
* support for recursive file discovery via the --recursive argument
* support for excluding files via --exclude
* JUnit XML output format
* Overriding repository root auto-detection via --repository
* Support `#pragma once` as an alternative to header include guards
* ... and [quite a bit][5] more

## Acknowledgements

Thanks to Google Inc. for open-sourcing their in-house tool.

Thanks to [our contributors][6].

### Maintainers

* [@aaronliu0130][7]
* [@jayvdb][8]

#### Former

* [@tkruse][9]
* [@mattyclarkson][10]
* [@theandrewdavis][11]

[1]: http://google.github.io/styleguide/cppguide.html
[2]: https://github.com/google/styleguide
[3]: https://github.com/google/styleguide/pull/528#issuecomment-592315430
[4]: https://pipx.pypa.io
[5]: https://github.com/cpplint/cpplint/blob/master/CHANGELOG.rst
[6]: https://github.com/cpplint/cpplint/graphs/contributors
[7]: https://github.com/aaronliu0130
[8]: https://github.com/jayvdb
[9]: https://github.com/tkruse
[10]: https://github.com/mattyclarkson
[11]: https://github.com/theandrewdavis
