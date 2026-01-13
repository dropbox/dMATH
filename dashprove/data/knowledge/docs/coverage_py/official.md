# Coverage.py - Code Coverage Measurement for Python

Coverage.py is a tool for measuring code coverage of Python programs. It monitors your program, noting which parts of the code have been executed, then analyzes the source to identify code that could have been executed but was not.

## Installation

```bash
# Using pip
pip install coverage

# With TOML support (for pyproject.toml)
pip install "coverage[toml]"

# Using uv
uv add coverage
```

## Basic Usage

### Running with Coverage

```bash
# Run a script
coverage run script.py

# Run a module
coverage run -m pytest

# Run with branch coverage
coverage run --branch -m pytest

# Specify source directories
coverage run --source=src -m pytest
```

### Viewing Reports

```bash
# Terminal report
coverage report

# HTML report
coverage html
# Opens htmlcov/index.html

# XML report (for CI tools)
coverage xml

# JSON report
coverage json

# LCOV report
coverage lcov
```

### Combining Multiple Runs

```bash
# Run multiple times with append
coverage run --append test1.py
coverage run --append test2.py

# Or run parallel and combine
coverage run --parallel-mode -m pytest tests/unit/
coverage run --parallel-mode -m pytest tests/integration/
coverage combine
coverage report
```

## Configuration

### pyproject.toml (Recommended)

```toml
[tool.coverage.run]
branch = true
source = ["src"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/migrations/*",
]
parallel = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true
skip_covered = true

[tool.coverage.html]
directory = "htmlcov"

[tool.coverage.xml]
output = "coverage.xml"
```

### .coveragerc (Alternative)

```ini
[run]
branch = True
source = src
omit =
    */tests/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    raise NotImplementedError

[html]
directory = htmlcov
```

## pytest Integration

### pytest-cov Plugin

```bash
# Install
pip install pytest-cov

# Run with coverage
pytest --cov=src tests/

# With HTML report
pytest --cov=src --cov-report=html tests/

# With branch coverage
pytest --cov=src --cov-branch tests/

# Fail under threshold
pytest --cov=src --cov-fail-under=80 tests/

# Multiple reports
pytest --cov=src --cov-report=html --cov-report=xml tests/
```

### pyproject.toml for pytest-cov

```toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=term-missing"
```

## Excluding Code

### Pragma Comments

```python
# Exclude a single line
x = 1  # pragma: no cover

# Exclude a block
if DEBUG:  # pragma: no cover
    print("debug mode")
    do_debug_stuff()

# Exclude function
def deprecated():  # pragma: no cover
    pass
```

### Pattern Exclusion

```toml
[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "def __str__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]

exclude_also = [
    "if self.debug:",
    "if settings.DEBUG",
]
```

## Branch Coverage

```bash
# Enable branch coverage
coverage run --branch -m pytest

# Report shows missing branches
coverage report --show-missing
```

Output format:
```
Name                 Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------
src/module.py           50      5     20      3    85%   10->12, 25
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run tests with coverage
  run: |
    coverage run -m pytest
    coverage xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    files: coverage.xml
    fail_ci_if_error: true

# Or using Coveralls
- name: Upload to Coveralls
  uses: coverallsapp/github-action@v2
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

### Fail Under Threshold

```yaml
- name: Check coverage
  run: coverage report --fail-under=80
```

## Advanced Usage

### Context Support

```bash
# Track coverage per test
coverage run --context=test -m pytest

# View by context in HTML report
coverage html --show-contexts
```

### Dynamic Context

```toml
[tool.coverage.run]
dynamic_context = "test_function"
```

### Coverage for Subprocesses

```python
# In conftest.py
import coverage

coverage.process_startup()
```

```bash
# Set environment variable
export COVERAGE_PROCESS_START=.coveragerc
```

### Measuring Import Time

```bash
coverage run --timid script.py
```

## Common Patterns

### Measuring Package Coverage

```bash
# Install package in editable mode
pip install -e .

# Run with source
coverage run --source=mypackage -m pytest
```

### Coverage Badges

```bash
# Generate badge (with genbadge)
pip install genbadge[coverage]
genbadge coverage -i coverage.xml -o coverage-badge.svg
```

### Diff Coverage

```bash
# Show coverage only for changed lines
pip install diff-cover
diff-cover coverage.xml --compare-branch=main
```

## Troubleshooting

### Missing Coverage

```bash
# Debug mode
coverage debug sys
coverage debug data

# Verbose
coverage run --debug=trace script.py
```

### C Extension Coverage

```bash
# Use timid tracer for C extensions
coverage run --timid -m pytest
```

## API Usage

```python
import coverage

cov = coverage.Coverage(branch=True, source=['src'])
cov.start()

# Run code here
import mymodule
mymodule.main()

cov.stop()
cov.save()

# Generate reports
cov.report()
cov.html_report(directory='htmlcov')
```

## Best Practices

1. **Enable branch coverage** - More thorough than line coverage
2. **Set fail_under threshold** - Prevent coverage regression
3. **Exclude generated code** - Focus on meaningful coverage
4. **Use pytest-cov** - Better integration with pytest
5. **Track coverage trends** - Use Codecov or Coveralls
6. **Don't aim for 100%** - Some code shouldn't be covered

## Links

- Documentation: https://coverage.readthedocs.io/
- GitHub: https://github.com/nedbat/coveragepy
- PyPI: https://pypi.org/project/coverage/
