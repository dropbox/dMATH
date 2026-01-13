# Poetry - Python Dependency Management

Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and manages them for you.

## Installation

```bash
# Official installer (recommended)
curl -sSL https://install.python-poetry.org | python3 -

# Using pipx
pipx install poetry

# Using homebrew (macOS)
brew install poetry

# Upgrade poetry
poetry self update
```

## Basic Usage

### Create New Project

```bash
# Create new project with structure
poetry new my-project

# Initialize in existing directory
poetry init
```

### Dependency Management

```bash
# Add dependency
poetry add requests

# Add dev dependency
poetry add --group dev pytest

# Add with version constraints
poetry add "requests>=2.25.0,<3.0.0"

# Add from git
poetry add git+https://github.com/user/repo.git

# Add optional dependency
poetry add --optional pandas

# Remove dependency
poetry remove requests

# Update dependencies
poetry update

# Update specific package
poetry update requests
```

### Virtual Environments

```bash
# Activate virtual environment
poetry shell

# Run command in virtual environment
poetry run python script.py
poetry run pytest

# Show environment info
poetry env info

# List environments
poetry env list

# Remove environment
poetry env remove python3.11
```

### Lock File

```bash
# Install from lock file (exact versions)
poetry install

# Update lock file without installing
poetry lock

# Install without dev dependencies
poetry install --without dev

# Install only specific groups
poetry install --only main
```

## Configuration

### pyproject.toml

```toml
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "A sample project"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "my_project", from = "src"}]

[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.31.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
black = "^24.0"
mypy = "^1.0"

[tool.poetry.group.docs.dependencies]
sphinx = "^7.0"

[tool.poetry.scripts]
my-cli = "my_project.cli:main"

[tool.poetry.extras]
pandas = ["pandas"]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Version Constraints

```toml
# Caret (^) - Compatible releases
requests = "^2.31.0"  # >=2.31.0 <3.0.0

# Tilde (~) - Minor version
requests = "~2.31.0"  # >=2.31.0 <2.32.0

# Exact version
requests = "2.31.0"

# Wildcard
requests = "2.31.*"

# Inequality
requests = ">=2.31.0,<3.0.0"
```

### Poetry Configuration

```bash
# Configure virtual environment location
poetry config virtualenvs.in-project true

# Use specific Python version
poetry env use python3.11

# Configure PyPI token
poetry config pypi-token.pypi <token>

# Show all config
poetry config --list
```

## Publishing

```bash
# Build package
poetry build

# Publish to PyPI
poetry publish

# Build and publish
poetry publish --build

# Publish to custom repository
poetry publish -r private
```

## Common Commands

```bash
# Show dependencies
poetry show

# Show dependency tree
poetry show --tree

# Show outdated packages
poetry show --outdated

# Export to requirements.txt
poetry export -f requirements.txt -o requirements.txt

# Check pyproject.toml validity
poetry check

# Search packages
poetry search requests
```

## Monorepo Support

```toml
# Path dependencies
[tool.poetry.dependencies]
my-shared-lib = {path = "../shared-lib", develop = true}
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install Poetry
  uses: snok/install-poetry@v1
  with:
    version: 1.8.0
    virtualenvs-create: true
    virtualenvs-in-project: true

- name: Install dependencies
  run: poetry install --no-interaction

- name: Run tests
  run: poetry run pytest
```

### Caching

```yaml
- name: Cache Poetry virtualenv
  uses: actions/cache@v4
  with:
    path: .venv
    key: venv-${{ runner.os }}-${{ hashFiles('poetry.lock') }}
```

## Best Practices

1. **Commit poetry.lock** to version control for reproducible builds
2. **Use groups** for organizing dev, test, docs dependencies
3. **Set virtualenvs.in-project = true** for IDE compatibility
4. **Use caret constraints** (^) for most dependencies
5. **Run poetry lock** before committing changes

## Links

- Documentation: https://python-poetry.org/docs/
- GitHub: https://github.com/python-poetry/poetry
- PyPI: https://pypi.org/project/poetry/
