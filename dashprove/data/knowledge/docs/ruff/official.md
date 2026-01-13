# [Ruff][1]

[[Ruff]][2] [[image]][3] [[image]][4] [[image]][5] [[Actions status]][6] [[Discord]][7]

[**Docs**][8] | [**Playground**][9]

An extremely fast Python linter and code formatter, written in Rust.

[Shows a bar chart with benchmark results.]

[Shows a bar chart with benchmark results.]

*Linting the CPython codebase from scratch.*

* ⚡️ 10-100x faster than existing linters (like Flake8) and formatters (like Black)
* 🐍 Installable via `pip`
* 🛠️ `pyproject.toml` support
* 🤝 Python 3.14 compatibility
* ⚖️ Drop-in parity with [Flake8][10], isort, and [Black][11]
* 📦 Built-in caching, to avoid re-analyzing unchanged files
* 🔧 Fix support, for automatic error correction (e.g., automatically remove unused imports)
* 📏 Over [800 built-in rules][12], with native re-implementations of popular Flake8 plugins, like
  flake8-bugbear
* ⌨️ First-party [editor integrations][13] for [VS Code][14] and [more][15]
* 🌎 Monorepo-friendly, with [hierarchical and cascading configuration][16]

Ruff aims to be orders of magnitude faster than alternative tools while integrating more
functionality behind a single, common interface.

Ruff can be used to replace [Flake8][17] (plus dozens of plugins), [Black][18], [isort][19],
[pydocstyle][20], [pyupgrade][21], [autoflake][22], and more, all while executing tens or hundreds
of times faster than any individual tool.

Ruff is extremely actively developed and used in major open-source projects like:

* [Apache Airflow][23]
* [Apache Superset][24]
* [FastAPI][25]
* [Hugging Face][26]
* [Pandas][27]
* [SciPy][28]

...and [many more][29].

Ruff is backed by [Astral][30], the creators of [uv][31] and [ty][32].

Read the [launch post][33], or the original [project announcement][34].

## [Testimonials][35]

[**Sebastián Ramírez**][36], creator of [FastAPI][37]:

> Ruff is so fast that sometimes I add an intentional bug in the code just to confirm it's actually
> running and checking the code.

[**Nick Schrock**][38], founder of [Elementl][39], co-creator of [GraphQL][40]:

> Why is Ruff a gamechanger? Primarily because it is nearly 1000x faster. Literally. Not a typo. On
> our largest module (dagster itself, 250k LOC) pylint takes about 2.5 minutes, parallelized across
> 4 cores on my M1. Running ruff against our *entire* codebase takes .4 seconds.

[**Bryan Van de Ven**][41], co-creator of [Bokeh][42], original author of [Conda][43]:

> Ruff is ~150-200x faster than flake8 on my machine, scanning the whole repo takes ~0.2s instead of
> ~20s. This is an enormous quality of life improvement for local dev. It's fast enough that I added
> it as an actual commit hook, which is terrific.

[**Timothy Crosley**][44], creator of [isort][45]:

> Just switched my first project to Ruff. Only one downside so far: it's so fast I couldn't believe
> it was working till I intentionally introduced some errors.

[**Tim Abbott**][46], lead developer of [Zulip][47] (also [here][48]):

> This is just ridiculously fast... `ruff` is amazing.

[1]: #ruff
[2]: https://github.com/astral-sh/ruff
[3]: https://pypi.python.org/pypi/ruff
[4]: https://github.com/astral-sh/ruff/blob/main/LICENSE
[5]: https://pypi.python.org/pypi/ruff
[6]: https://github.com/astral-sh/ruff/actions
[7]: https://discord.com/invite/astral-sh
[8]: ./
[9]: https://play.ruff.rs/
[10]: faq/#how-does-ruffs-linter-compare-to-flake8
[11]: faq/#how-does-ruffs-formatter-compare-to-black
[12]: rules/
[13]: editors/
[14]: https://github.com/astral-sh/ruff-vscode
[15]: editors/setup/
[16]: configuration/#config-file-discovery
[17]: https://pypi.org/project/flake8/
[18]: https://github.com/psf/black
[19]: https://pypi.org/project/isort/
[20]: https://pypi.org/project/pydocstyle/
[21]: https://pypi.org/project/pyupgrade/
[22]: https://pypi.org/project/autoflake/
[23]: https://github.com/apache/airflow
[24]: https://github.com/apache/superset
[25]: https://github.com/tiangolo/fastapi
[26]: https://github.com/huggingface/transformers
[27]: https://github.com/pandas-dev/pandas
[28]: https://github.com/scipy/scipy
[29]: https://github.com/astral-sh/ruff#whos-using-ruff
[30]: https://astral.sh
[31]: https://github.com/astral-sh/uv
[32]: https://github.com/astral-sh/ty
[33]: https://astral.sh/blog/announcing-astral-the-company-behind-ruff
[34]: https://notes.crmarsh.com/python-tooling-could-be-much-much-faster
[35]: #testimonials
[36]: https://twitter.com/tiangolo/status/1591912354882764802
[37]: https://github.com/tiangolo/fastapi
[38]: https://twitter.com/schrockn/status/1612615862904827904
[39]: https://www.elementl.com/
[40]: https://graphql.org/
[41]: https://github.com/bokeh/bokeh/pull/12605
[42]: https://github.com/bokeh/bokeh/
[43]: https://docs.conda.io/en/latest/
[44]: https://twitter.com/timothycrosley/status/1606420868514877440
[45]: https://github.com/PyCQA/isort
[46]: https://github.com/zulip/zulip/pull/23431#issuecomment-1302557034
[47]: https://github.com/zulip/zulip
[48]: https://github.com/astral-sh/ruff/issues/465#issuecomment-1317400028
