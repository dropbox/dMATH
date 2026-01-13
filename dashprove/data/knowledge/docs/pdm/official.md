# PDM - Python Development Master

PDM is a modern Python package and dependency manager supporting the latest PEP standards. It features PEP 582 (Python local packages directory) support, lockfile generation, and a plugin system.

## Installation

```bash
# Using pipx (recommended)
pipx install pdm

# Using pip
pip install pdm

# Using homebrew (macOS)
brew install pdm

# Using script installer
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

## Basic Usage

### Create New Project

```bash
# Create new project interactively
pdm init

# Create with defaults
pdm init --non-interactive

# Create library project
pdm init --lib

# Create application project
pdm init --python 3.12
```

### Dependency Management

```bash
# Add dependency
pdm add requests

# Add dev dependency
pdm add -dG dev pytest

# Add to specific group
pdm add -G docs sphinx

# Add with version constraint
pdm add "requests>=2.31.0,<3.0"

# Add from git
pdm add "git+https://github.com/user/repo.git"

# Remove dependency
pdm remove requests

# Update all dependencies
pdm update

# Update specific package
pdm update requests
```

### Environment Management

```bash
# Install dependencies
pdm install

# Install with specific group
pdm install -G dev

# Install production only
pdm install --prod

# Sync environment (remove unused)
pdm sync

# List packages
pdm list

# Show dependency tree
pdm list --tree
```

### Running Commands

```bash
# Run Python script
pdm run python script.py

# Run installed command
pdm run pytest

# Start shell in environment
pdm venv activate
```

## Configuration

### pyproject.toml

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My project"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0",
]
requires-python = ">=3.11"
readme = "README.md"
license = {text = "MIT"}

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "mypy>=1.0",
]
docs = [
    "sphinx>=7.0",
]

[project.scripts]
my-cli = "my_project.cli:main"

[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"

[tool.pdm]
distribution = true

[tool.pdm.dev-dependencies]
test = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
]
lint = [
    "ruff>=0.5",
]

[tool.pdm.scripts]
test = "pytest tests/"
lint = "ruff check ."
format = "ruff format ."
```

### PDM Configuration

```bash
# Set virtual environment location
pdm config venv.in_project true

# Use specific Python
pdm use 3.12

# Configure PyPI
pdm config pypi.url https://pypi.org/simple

# Show all config
pdm config
```

## Scripts

### Define Scripts

```toml
[tool.pdm.scripts]
test = "pytest tests/"
lint = "ruff check ."
format = "ruff format ."

# With arguments
test-cov = "pytest --cov=src tests/"

# Composite scripts
check = {composite = ["lint", "test"]}

# Shell scripts
setup = {shell = "pip install -e . && pre-commit install"}

# With environment variables
dev = {cmd = "flask run", env = {FLASK_ENV = "development"}}
```

### Run Scripts

```bash
# Run defined script
pdm run test

# Run with extra arguments
pdm run test -- -v -x

# Run composite script
pdm run check
```

## Lock Files

```bash
# Generate/update lock file
pdm lock

# Lock without updating
pdm lock --no-update

# Lock for specific platforms
pdm lock --platform linux --platform windows

# Export to requirements.txt
pdm export -o requirements.txt

# Export with hashes
pdm export --without-hashes -o requirements.txt
```

## Virtual Environment Management

```bash
# Create virtual environment
pdm venv create

# Create with specific Python
pdm venv create 3.12

# List virtual environments
pdm venv list

# Remove virtual environment
pdm venv remove my-venv

# Activate environment
eval $(pdm venv activate)

# Or use shell integration
pdm venv activate
```

## PEP 582 Mode (Experimental)

```bash
# Enable PEP 582
pdm config python.use_venv false

# Packages stored in __pypackages__/
```

## Building and Publishing

```bash
# Build package
pdm build

# Build wheel only
pdm build --no-sdist

# Publish to PyPI
pdm publish

# Publish to test PyPI
pdm publish -r testpypi

# Configure credentials
pdm config pypi.username __token__
pdm config pypi.password <token>
```

## Plugin System

```bash
# Install plugin
pdm plugin add pdm-bump

# List plugins
pdm plugin list

# Remove plugin
pdm plugin remove pdm-bump
```

### Popular Plugins

- `pdm-bump` - Version bumping
- `pdm-vscode` - VS Code integration
- `pdm-autoexport` - Auto-export requirements

## CI/CD Integration

### GitHub Actions

```yaml
- name: Setup PDM
  uses: pdm-project/setup-pdm@v4
  with:
    python-version: "3.12"
    cache: true

- name: Install dependencies
  run: pdm install

- name: Run tests
  run: pdm run test
```

### Caching

```yaml
- name: Cache PDM
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pdm
      .venv
    key: pdm-${{ hashFiles('pdm.lock') }}
```

## Monorepo Support

```toml
# Root pyproject.toml
[tool.pdm.overrides]
requests = ">=2.31.0"

# Package reference
[project]
dependencies = [
    "shared-lib @ file:///${PROJECT_ROOT}/packages/shared",
]
```

## Common Commands

```bash
# Project setup
pdm init                  # Initialize project
pdm install              # Install dependencies
pdm sync                 # Sync environment

# Dependencies
pdm add package          # Add dependency
pdm remove package       # Remove dependency
pdm update              # Update dependencies
pdm list --tree         # Show dependency tree

# Running
pdm run command         # Run command in environment
pdm run script          # Run defined script

# Building
pdm build               # Build package
pdm publish            # Publish to PyPI

# Lock file
pdm lock                # Generate lock file
pdm export             # Export to requirements.txt

# Environment
pdm venv create        # Create virtual environment
pdm use 3.12           # Use specific Python
```

## Best Practices

1. **Always commit pdm.lock** - For reproducible builds
2. **Use dependency groups** - Separate dev, test, docs
3. **Enable venv.in_project** - For IDE compatibility
4. **Define scripts** - Standardize common commands
5. **Use pdm export** - For Docker builds

## Links

- Documentation: https://pdm-project.org/
- GitHub: https://github.com/pdm-project/pdm
- PyPI: https://pypi.org/project/pdm/
