# [Pylint][1][¶][2]

[[https://github.com/pylint-dev/pylint/actions/workflows/tests.yaml/badge.svg?branch=main] ][3]
[[https://codecov.io/gh/pylint-dev/pylint/branch/main/graph/badge.svg?token=ZETEzayrfk] ][4] [[PyPI
Package version] ][5] [[Documentation Status] ][6]
[[https://img.shields.io/badge/code%20style-black-000000.svg] ][7]
[[https://img.shields.io/badge/linting-pylint-yellowgreen] ][8] [[pre-commit.ci status] ][9] [[CII
Best Practices] ][10] [[OpenSSF Scorecard] ][11] [[Discord] ][12]

## What is Pylint?[¶][13]

Pylint is a [static code analyser][14] for Python 2 or 3. The latest version supports Python 3.10.0
and above.

Pylint analyses your code without actually running it. It checks for errors, enforces a coding
standard, looks for [code smells][15], and can make suggestions about how the code could be
refactored.

## Install[¶][16]

For command line use, pylint is installed with:

pip install pylint

Or if you want to also check spelling with `enchant` (you might need to [install the enchant C
library][17]):

pip install pylint[spelling]

It can also be integrated in most editors or IDEs. More information can be found [in the
documentation][18].

## What differentiates Pylint?[¶][19]

Pylint is not trusting your typing and is inferring the actual values of nodes (for a start because
there was no typing when pylint started off) using its internal code representation (astroid). If
your code is `import logging as argparse`, Pylint can check and know that `argparse.error(...)` is
in fact a logging call and not an argparse call. This makes pylint slower, but it also lets pylint
find more issues if your code is not fully typed.

> [inference] is the killer feature that keeps us using [pylint] in our project despite how
> painfully slow it is. - [Realist pylint user][20], 2022

pylint, not afraid of being a little slower than it already is, is also a lot more thorough than
other linters. There are more checks, including some opinionated ones that are deactivated by
default but can be enabled using configuration.

## How to use pylint[¶][21]

Pylint isn't smarter than you: it may warn you about things that you have conscientiously done or
check for some things that you don't care about. During adoption, especially in a legacy project
where pylint was never enforced, it's best to start with the `--errors-only` flag, then disable
convention and refactor messages with `--disable=C,R` and progressively re-evaluate and re-enable
messages as your priorities evolve.

Pylint is highly configurable and permits to write plugins in order to add your own checks (for
example, for internal libraries or an internal rule). Pylint also has an ecosystem of existing
plugins for popular frameworks and third-party libraries.

Note

Pylint supports the Python standard library out of the box. Third-party libraries are not always
supported, so a plugin might be needed. A good place to start is `PyPI` which often returns a plugin
by searching for `pylint <library>`. [pylint-pydantic][22], [pylint-django][23] and
[pylint-sonarjson][24] are examples of such plugins. More information about plugins and how to load
them can be found at [plugins][25].

## Advised linters alongside pylint[¶][26]

Projects that you might want to use alongside pylint include [ruff][27] (**really** fast, with
builtin auto-fix and a large number of checks taken from popular linters, but implemented in `rust`)
or [flake8][28] (a framework to implement your own checks in python using `ast` directly),
[mypy][29], [pyright][30] / pylance or [pyre][31] (typing checks), [bandit][32] (security oriented
checks), [black][33] and [isort][34] (auto-formatting), [autoflake][35] (automated removal of unused
imports or variables), [pyupgrade][36] (automated upgrade to newer python syntax) and
[pydocstringformatter][37] (automated pep257).

## Additional tools included in pylint[¶][38]

Pylint ships with two additional tools:

* [pyreverse][39] (standalone tool that generates package and class diagrams.)
* [symilar][40] (duplicate code finder that is also integrated in pylint)

## Contributing[¶][41]

We welcome all forms of contributions such as updates for documentation, new code, checking issues
for duplicates or telling us that we can close them, confirming that issues still exist, [creating
issues because you found a bug or want a feature][42], etc. Everything is much appreciated!

Please follow the [code of conduct][43] and check [the Contributor Guides][44] if you want to make a
code contribution.

## Show your usage[¶][45]

You can place this badge in your README to let others know your project uses pylint.

> [[https://img.shields.io/badge/linting-pylint-yellowgreen] ][46]

Learn how to add a badge to your documentation in [the badge documentation][47].

## License[¶][48]

pylint is, with a few exceptions listed below, [GPLv2][49].

The icon files are licensed under the [CC BY-SA 4.0][50] license:

* [doc/logo.png][51]
* [doc/logo.svg][52]

## Support[¶][53]

Please check [the contact information][54].

─────┬──────────────────────────────────────────────────────────────────────────────────────────────
[[Tid│Professional support for pylint is available as part of the [Tidelift Subscription][56].      
elift│Tidelift gives software development teams a single source for purchasing and maintaining their
]][55│software, with professional grade assurances from the experts who know it best, while         
]    │seamlessly integrating with existing tools.                                                   
─────┴──────────────────────────────────────────────────────────────────────────────────────────────

[1]: https://pylint.readthedocs.io/
[2]: #pylint
[3]: https://github.com/pylint-dev/pylint/actions
[4]: https://codecov.io/gh/pylint-dev/pylint
[5]: https://pypi.python.org/pypi/pylint
[6]: https://pylint.readthedocs.io/en/latest/?badge=latest
[7]: https://github.com/ambv/black
[8]: https://github.com/pylint-dev/pylint
[9]: https://results.pre-commit.ci/latest/github/pylint-dev/pylint/main
[10]: https://bestpractices.coreinfrastructure.org/projects/6328
[11]: https://api.securityscorecards.dev/projects/github.com/PyCQA/pylint
[12]: https://discord.gg/qYxpadCgkx
[13]: #what-is-pylint
[14]: https://en.wikipedia.org/wiki/Static_code_analysis
[15]: https://martinfowler.com/bliki/CodeSmell.html
[16]: #install
[17]: https://pyenchant.github.io/pyenchant/install.html#installing-the-enchant-c-library
[18]: https://pylint.readthedocs.io/en/latest/user_guide/installation/index.html
[19]: #what-differentiates-pylint
[20]: https://github.com/charliermarsh/ruff/issues/970#issuecomment-1381067064
[21]: #how-to-use-pylint
[22]: https://pypi.org/project/pylint-pydantic
[23]: https://github.com/pylint-dev/pylint-django
[24]: https://github.com/cnescatlab/pylint-sonarjson-catlab
[25]: https://pylint.readthedocs.io/en/latest/development_guide/how_tos/plugins.html#plugins
[26]: #advised-linters-alongside-pylint
[27]: https://github.com/astral-sh/ruff
[28]: https://github.com/PyCQA/flake8
[29]: https://github.com/python/mypy
[30]: https://github.com/microsoft/pyright
[31]: https://github.com/facebook/pyre-check
[32]: https://github.com/PyCQA/bandit
[33]: https://github.com/psf/black
[34]: https://pycqa.github.io/isort/
[35]: https://github.com/myint/autoflake
[36]: https://github.com/asottile/pyupgrade
[37]: https://github.com/DanielNoord/pydocstringformatter
[38]: #additional-tools-included-in-pylint
[39]: https://pylint.readthedocs.io/en/latest/additional_tools/pyreverse/index.html
[40]: https://pylint.readthedocs.io/en/latest/additional_tools/symilar/index.html
[41]: #contributing
[42]: https://pylint.readthedocs.io/en/latest/contact.html#bug-reports-feedback
[43]: https://github.com/pylint-dev/pylint/blob/main/CODE_OF_CONDUCT.md
[44]: https://pylint.readthedocs.io/en/latest/development_guide/contribute.html
[45]: #show-your-usage
[46]: https://github.com/pylint-dev/pylint
[47]: https://pylint.readthedocs.io/en/latest/user_guide/installation/badge.html
[48]: #license
[49]: https://github.com/pylint-dev/pylint/blob/main/LICENSE
[50]: https://creativecommons.org/licenses/by-sa/4.0/
[51]: https://raw.githubusercontent.com/pylint-dev/pylint/main/doc/logo.png
[52]: https://raw.githubusercontent.com/pylint-dev/pylint/main/doc/logo.svg
[53]: #support
[54]: https://pylint.readthedocs.io/en/latest/contact.html
[55]: https://raw.githubusercontent.com/pylint-dev/pylint/main/doc/media/Tidelift_Logos_RGB_Tidelift
_Shorthand_On-White.png
[56]: https://tidelift.com/subscription/pkg/pypi-pylint?utm_source=pypi-pylint&utm_medium=referral&u
tm_campaign=readme
