# Rye - Python Project Management

Rye is a comprehensive project and package management solution for Python. It aims to be a one-stop-shop for Python development, handling Python installations, dependencies, virtual environments, and more.

## Installation

```bash
# Using curl (Unix/macOS)
curl -sSf https://rye.astral.sh/get | bash

# Using PowerShell (Windows)
irm https://rye.astral.sh/get | iex

# Using homebrew
brew install rye
```

After installation, add to your shell:
```bash
# Add to ~/.bashrc or ~/.zshrc
source "$HOME/.rye/env"
```

## Basic Usage

### Create New Project

```bash
# Create new project
rye init my-project

# Create in current directory
rye init

# Create library project
rye init --lib

# Create virtual project (no package)
rye init --virtual
```

### Dependency Management

```bash
# Add dependency
rye add requests

# Add dev dependency
rye add --dev pytest

# Add with version constraint
rye add "requests>=2.31.0"

# Add from git
rye add "requests @ git+https://github.com/psf/requests"

# Remove dependency
rye remove requests

# Sync dependencies
rye sync
```

### Running Commands

```bash
# Run Python script
rye run python script.py

# Run installed command
rye run pytest

# Enter shell with activated environment
rye shell
```

## Configuration

### pyproject.toml

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My project"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0",
]
requires-python = ">= 3.11"
readme = "README.md"
license = { text = "MIT" }

[project.scripts]
my-cli = "my_project.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.rye]
managed = true
dev-dependencies = [
    "pytest>=8.0",
    "mypy>=1.0",
]

[tool.rye.scripts]
test = "pytest tests/"
lint = "ruff check ."
```

### Global Configuration

```bash
# Show config
rye config --show

# Set Python preference
rye config --set behavior.use-uv=true

# Set default toolchain
rye config --set default.toolchain=cpython@3.12
```

## Python Management

```bash
# Install Python version
rye fetch 3.12

# List available Python versions
rye toolchain list --include-downloadable

# List installed versions
rye toolchain list

# Pin project Python version
rye pin 3.12

# Remove Python version
rye toolchain remove cpython@3.11
```

## Scripts

### Define Scripts

```toml
[tool.rye.scripts]
test = "pytest tests/"
lint = "ruff check ."
format = "ruff format ."

# Chain commands
check = { chain = ["lint", "test"] }

# With environment variables
dev = { cmd = "flask run", env = { FLASK_DEBUG = "1" } }

# Call Python function
serve = { call = "my_project.server:run" }
```

### Run Scripts

```bash
# Run defined script
rye run test

# Run with arguments
rye run test -- -v -x
```

## Lock Files

```bash
# Sync and update lock file
rye sync

# Lock without installing
rye lock

# Update all dependencies
rye lock --update-all

# Update specific package
rye sync --update requests
```

## Virtual Environments

Rye automatically manages virtual environments:

```bash
# Environment is created in .venv
ls .venv/

# Activate manually if needed
source .venv/bin/activate

# Or use rye shell
rye shell
```

## Building and Publishing

```bash
# Build package
rye build

# Build wheel only
rye build --wheel

# Publish to PyPI
rye publish

# Publish to test PyPI
rye publish --repository testpypi
```

## Global Tools

```bash
# Install tool globally
rye install ruff

# Run global tool
ruff check .

# List global tools
rye tools list

# Uninstall global tool
rye tools uninstall ruff
```

## Workspaces (Monorepos)

```toml
# Root pyproject.toml
[tool.rye.workspace]
members = ["packages/*"]

# Package structure:
# packages/
#   package-a/
#     pyproject.toml
#   package-b/
#     pyproject.toml
```

```bash
# Sync all workspace packages
rye sync

# Add dependency to specific package
cd packages/package-a && rye add requests
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install Rye
  uses: eifinger/setup-rye@v4
  with:
    version: 'latest'

- name: Sync dependencies
  run: rye sync

- name: Run tests
  run: rye run test
```

### Caching

```yaml
- name: Cache Rye
  uses: actions/cache@v4
  with:
    path: |
      ~/.rye
      .venv
    key: rye-${{ hashFiles('requirements.lock') }}
```

## uv Integration

Rye uses uv by default for fast package installation:

```bash
# Check if uv is enabled
rye config --show | grep uv

# Disable uv (use pip instead)
rye config --set behavior.use-uv=false
```

## Common Commands

```bash
# Project setup
rye init                    # Initialize project
rye sync                    # Sync dependencies
rye shell                   # Enter environment

# Dependencies
rye add package             # Add dependency
rye add --dev package       # Add dev dependency
rye remove package          # Remove dependency
rye lock                    # Update lock file

# Python management
rye fetch 3.12              # Install Python
rye pin 3.12                # Pin version
rye toolchain list          # List versions

# Running
rye run command             # Run command
rye run script              # Run script

# Building
rye build                   # Build package
rye publish                 # Publish package

# Global tools
rye install tool            # Install global tool
rye tools list              # List global tools

# Self management
rye self update             # Update Rye
```

## Self Update

```bash
# Update Rye
rye self update

# Check version
rye --version
```

## Best Practices

1. **Let Rye manage Python** - Use `rye fetch` instead of pyenv
2. **Commit requirements.lock** - For reproducible builds
3. **Use workspaces** - For monorepo projects
4. **Define scripts** - Standardize commands
5. **Use global tools** - For CLI utilities

## Migration from Poetry

```bash
# Rye can import from poetry
rye init  # Converts pyproject.toml
rye sync  # Creates new lock file
```

## Links

- Documentation: https://rye.astral.sh/
- GitHub: https://github.com/astral-sh/rye
