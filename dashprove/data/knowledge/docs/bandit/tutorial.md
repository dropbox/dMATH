* Getting Started
* [ View page source][1]

# Getting Started[][2]

## Installation[][3]

Bandit is distributed on PyPI. The best way to install it is with pip.

Create a virtual environment and activate it using virtualenv (optional):

virtualenv bandit-env
source bandit-env/bin/activate

Alternatively, use venv instead of virtualenv (optional):

python3 -m venv bandit-env
source bandit-env/bin/activate

Install Bandit:

pip install bandit

If you want to include TOML support, install it with the toml extras:

pip install bandit[toml]

If you want to use the bandit-baseline CLI, install it with the baseline extras:

pip install bandit[baseline]

If you want to include SARIF output formatter support, install it with the sarif extras:

pip install bandit[sarif]

Run Bandit:

bandit -r path/to/your/code

Bandit can also be installed from source. To do so, either clone the repository or download the
source tarball from PyPI, then install it:

python setup.py install

Alternatively, let pip do the downloading for you, like this:

pip install git+https://github.com/PyCQA/bandit#egg=bandit

## Usage[][4]

Example usage across a code tree:

bandit -r ~/your_repos/project

Two examples of usage across the `examples/` directory, showing three lines of context and only
reporting on the high-severity issues:

bandit examples/*.py -n 3 --severity-level=high
bandit examples/*.py -n 3 -lll

Bandit can be run with profiles. To run Bandit against the examples directory using only the plugins
listed in the `ShellInjection` profile:

bandit examples/*.py -p ShellInjection

Bandit also supports passing lines of code to scan using standard input. To run Bandit with standard
input:

cat examples/imports.py | bandit -

For more usage information:

bandit -h

## Baseline[][5]

Bandit allows specifying the path of a baseline report to compare against using the base line
argument (i.e. `-b BASELINE` or `--baseline BASELINE`).

bandit -b BASELINE

This is useful for ignoring known vulnerabilities that you believe are non-issues (e.g. a cleartext
password in a unit test). To generate a baseline report simply run Bandit with the output format set
to `json` (only JSON-formatted files are accepted as a baseline) and output file path specified:

bandit -f json -o PATH_TO_OUTPUT_FILE

## Version control integration[][6]

Use [pre-commit][7]. Once you [have it installed][8], add this to the `.pre-commit-config.yaml` in
your repository (be sure to update rev to point to a [real git tag/revision][9]!):

repos:
- repo: https://github.com/PyCQA/bandit
  rev: '' # Update me!
  hooks:
  - id: bandit

Then run `pre-commit install` and you’re ready to go.

[ Previous][10] [Next ][11]

© Copyright 2025, Bandit Developers.

Built with [Sphinx][12] using a [theme][13] provided by [Read the Docs][14].

[1]: _sources/start.rst.txt
[2]: #getting-started
[3]: #installation
[4]: #usage
[5]: #baseline
[6]: #version-control-integration
[7]: https://pre-commit.com/
[8]: https://pre-commit.com/#install
[9]: https://github.com/PyCQA/bandit/releases
[10]: index.html
[11]: config.html
[12]: https://www.sphinx-doc.org/
[13]: https://github.com/readthedocs/sphinx_rtd_theme
[14]: https://readthedocs.org
