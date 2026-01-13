# Hatch - Modern Python Project Manager

Hatch is a modern, extensible Python project manager that handles environments, versioning, building, and publishing. It's designed for the modern Python packaging ecosystem and follows PEP standards.

## Installation

```bash
# Using pipx (recommended)
pipx install hatch

# Using pip
pip install hatch

# Using homebrew (macOS)
brew install hatch

# Using uv
uv tool install hatch
```

## Basic Usage

### Create New Project

```bash
# Create new project
hatch new my-project

# Create library project
hatch new --init my-library

# Initialize in existing directory
hatch new --init
```

### Environment Management

```bash
# Enter default environment
hatch shell

# Run command in environment
hatch run python script.py

# Run in specific environment
hatch run test:pytest

# List environments
hatch env show
```

### Building and Publishing

```bash
# Build package
hatch build

# Build wheel only
hatch build -t wheel

# Build sdist only
hatch build -t sdist

# Publish to PyPI
hatch publish

# Publish to test PyPI
hatch publish -r test
```

## Configuration

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-project"
version = "0.1.0"
description = "My project description"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["sample", "project"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "mypy>=1.0",
]

[project.scripts]
my-cli = "my_project.cli:main"

[project.urls]
Documentation = "https://example.com/docs"
Source = "https://github.com/user/my-project"
```

### Hatch Configuration

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/my_project"]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
]

[tool.hatch.version]
path = "src/my_project/__about__.py"

[tool.hatch.envs.default]
dependencies = [
    "pytest",
    "pytest-cov",
]

[tool.hatch.envs.default.scripts]
test = "pytest {args:tests}"
cov = "pytest --cov=src {args:tests}"
```

## Environments

### Defining Environments

```toml
[tool.hatch.envs.test]
dependencies = [
    "pytest",
    "pytest-cov",
]

[tool.hatch.envs.test.scripts]
run = "pytest {args}"
cov = "pytest --cov {args}"

[tool.hatch.envs.lint]
detached = true
dependencies = [
    "ruff",
    "mypy",
]

[tool.hatch.envs.lint.scripts]
check = [
    "ruff check .",
    "mypy src/",
]
fmt = "ruff format ."
```

### Matrix Testing

```toml
[[tool.hatch.envs.test.matrix]]
python = ["3.11", "3.12", "3.13"]

[tool.hatch.envs.test.scripts]
run = "pytest {args:tests}"
```

```bash
# Run tests in all Python versions
hatch run test:run
```

### Environment Variables

```toml
[tool.hatch.envs.default.env-vars]
MY_VAR = "value"
DEBUG = "true"
```

## Version Management

### Static Version

```toml
[project]
version = "1.0.0"
```

### Dynamic Version from File

```toml
[project]
dynamic = ["version"]

[tool.hatch.version]
path = "src/my_project/__about__.py"
```

```python
# src/my_project/__about__.py
__version__ = "1.0.0"
```

### Version Commands

```bash
# Show current version
hatch version

# Bump version
hatch version minor
hatch version patch
hatch version major

# Set specific version
hatch version 2.0.0
```

## Scripts

### Define Scripts

```toml
[tool.hatch.envs.default.scripts]
test = "pytest {args:tests}"
lint = "ruff check {args:.}"
format = "ruff format {args:.}"
typecheck = "mypy src/"
all = ["format", "lint", "typecheck", "test"]
```

### Run Scripts

```bash
# Run default script
hatch run test

# Run with arguments
hatch run test -- -v -x

# Run multiple scripts
hatch run all
```

## Build Configuration

### Source Distributions

```toml
[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/*.md",
]
exclude = [
    "/.github",
    "/docs",
]
```

### Wheel Configuration

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/my_project"]

# For src layout
[tool.hatch.build.targets.wheel.sources]
"src" = ""
```

### Build Hooks

```toml
[tool.hatch.build.hooks.custom]
path = "build_hook.py"
```

## Project Structure

### Recommended Structure

```
my-project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       └── __about__.py
├── tests/
│   └── test_main.py
├── pyproject.toml
└── README.md
```

### Source Layout Configuration

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/my_project"]
```

## CI/CD Integration

### GitHub Actions

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install Hatch
        run: pipx install hatch

      - name: Run tests
        run: hatch run +py=${{ matrix.python }} test:run

  publish:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Hatch
        run: pipx install hatch

      - name: Build
        run: hatch build

      - name: Publish
        run: hatch publish
        env:
          HATCH_INDEX_USER: __token__
          HATCH_INDEX_AUTH: ${{ secrets.PYPI_TOKEN }}
```

## Common Commands

```bash
# Project management
hatch new project-name    # Create new project
hatch new --init          # Initialize in current directory

# Environment
hatch shell               # Enter environment
hatch env show            # List environments
hatch env create          # Create environment
hatch env remove          # Remove environment

# Running
hatch run script          # Run script
hatch run env:script      # Run in specific environment

# Building
hatch build               # Build sdist and wheel
hatch clean               # Remove build artifacts

# Publishing
hatch publish             # Publish to PyPI

# Version
hatch version             # Show version
hatch version minor       # Bump minor version
```

## Best Practices

1. **Use src layout** - Prevents import confusion
2. **Define multiple environments** - Separate test, lint, docs
3. **Use matrix testing** - Test multiple Python versions
4. **Store version in __about__.py** - Single source of truth
5. **Use hatchling** as build backend - Modern, fast

## Links

- Documentation: https://hatch.pypa.io/
- GitHub: https://github.com/pypa/hatch
- PyPI: https://pypi.org/project/hatch/
