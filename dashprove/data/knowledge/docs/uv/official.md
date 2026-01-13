# uv - Extremely Fast Python Package Manager

uv is an extremely fast Python package installer and resolver, written in Rust. It's designed as a drop-in replacement for pip and pip-tools with 10-100x faster performance.

## Installation

```bash
# Using curl (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using pip
pip install uv

# Using homebrew (macOS)
brew install uv

# Using cargo
cargo install uv
```

## Basic Usage

### Package Installation

```bash
# Install packages
uv pip install requests

# Install from requirements.txt
uv pip install -r requirements.txt

# Install with extras
uv pip install "fastapi[all]"

# Install editable package
uv pip install -e .

# Install specific version
uv pip install "requests==2.31.0"

# Uninstall
uv pip uninstall requests
```

### Virtual Environments

```bash
# Create virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.12

# Create at specific path
uv venv .venv

# Activate (Unix)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Lock Files (pip-compile replacement)

```bash
# Compile requirements.in to requirements.txt
uv pip compile requirements.in -o requirements.txt

# Compile pyproject.toml
uv pip compile pyproject.toml -o requirements.txt

# Sync environment to lock file
uv pip sync requirements.txt

# Upgrade all packages in lock file
uv pip compile requirements.in --upgrade

# Upgrade specific package
uv pip compile requirements.in --upgrade-package requests
```

## Project Management (uv 0.4+)

### Create New Project

```bash
# Create new project
uv init my-project

# Create library project
uv init --lib my-library

# Create application project
uv init --app my-app
```

### Dependency Management

```bash
# Add dependency
uv add requests

# Add dev dependency
uv add --dev pytest

# Add optional dependency
uv add --optional ml pandas

# Remove dependency
uv remove requests

# Sync dependencies
uv sync

# Update dependencies
uv lock --upgrade
```

### Running Commands

```bash
# Run Python script
uv run python script.py

# Run installed command
uv run pytest

# Run with inline dependencies
uv run --with requests python -c "import requests; print(requests.__version__)"
```

## Tool Management

```bash
# Install tool globally
uv tool install ruff

# Run tool without installing
uvx ruff check .

# Upgrade tool
uv tool upgrade ruff

# List installed tools
uv tool list

# Uninstall tool
uv tool uninstall ruff
```

## Configuration

### pyproject.toml

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My project"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
]

[tool.uv.pip]
index-url = "https://pypi.org/simple"
```

### Environment Variables

```bash
# Custom index
UV_INDEX_URL=https://private.pypi.org/simple

# Extra index
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Cache directory
UV_CACHE_DIR=/path/to/cache

# Disable cache
UV_NO_CACHE=1

# Python preference
UV_PYTHON_PREFERENCE=only-managed
```

## Python Version Management

```bash
# Install Python version
uv python install 3.12

# List available Python versions
uv python list

# Pin Python version for project
uv python pin 3.12

# Use specific Python
uv venv --python 3.12
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v4
  with:
    version: "latest"

- name: Install dependencies
  run: uv sync

- name: Run tests
  run: uv run pytest
```

### Caching

```yaml
- name: Cache uv
  uses: actions/cache@v4
  with:
    path: ~/.cache/uv
    key: uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
```

## Performance Comparison

uv is 10-100x faster than pip:
- Cold install: ~10x faster
- Warm cache: ~100x faster
- Resolution: Near-instant

## Migration from pip

```bash
# Replace pip install
pip install requests  →  uv pip install requests

# Replace pip freeze
pip freeze  →  uv pip freeze

# Replace pip-compile
pip-compile  →  uv pip compile

# Replace pipx
pipx install ruff  →  uv tool install ruff
```

## Migration from Poetry

```bash
# Export poetry lock to requirements
uv pip compile pyproject.toml -o requirements.txt

# Use uv for dependency management
uv add requests  # instead of poetry add requests
```

## Common Commands Reference

```bash
# Show installed packages
uv pip list

# Show package info
uv pip show requests

# Check for outdated packages
uv pip list --outdated

# Generate lock file
uv lock

# Clean cache
uv cache clean

# Self update
uv self update
```

## Best Practices

1. **Use uv.lock** for reproducible builds
2. **Use uv tool** for CLI tools instead of pip
3. **Enable caching in CI** for faster builds
4. **Use uv run** to execute in project environment
5. **Pin Python version** with `uv python pin`

## Links

- Documentation: https://docs.astral.sh/uv/
- GitHub: https://github.com/astral-sh/uv
- Blog: https://astral.sh/blog
