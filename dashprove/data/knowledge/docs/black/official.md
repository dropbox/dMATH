# Black - The Uncompromising Python Code Formatter

Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. Black makes code review faster by producing the smallest diffs possible.

## Installation

```bash
# Using pip
pip install black

# Using pipx (recommended for CLI tools)
pipx install black

# Using uv
uv tool install black

# With Jupyter notebook support
pip install 'black[jupyter]'
```

## Basic Usage

```bash
# Format a single file
black script.py

# Format a directory
black src/

# Check without modifying (exit code 1 if changes needed)
black --check src/

# Show diff without modifying
black --diff src/

# Format stdin
echo "x  =  1" | black -

# Format specific Python version target
black --target-version py311 src/
```

## Configuration

### pyproject.toml (Recommended)

```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | migrations
)/
'''
```

### Command Line Options

```bash
# Line length (default: 88)
black --line-length 100 src/

# Skip string normalization (keep single/double quotes as-is)
black --skip-string-normalization src/

# Skip magic trailing comma
black --skip-magic-trailing-comma src/

# Preview mode (enable upcoming style changes)
black --preview src/

# Quiet mode
black --quiet src/

# Verbose mode
black --verbose src/
```

## Editor Integration

### VS Code
```json
{
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
```

### Pre-commit Hook
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        language_version: python3.11
```

## Key Formatting Rules

1. **Line Length**: Default 88 characters (10% over PEP 8's 79)
2. **Strings**: Prefers double quotes, converts single quotes
3. **Trailing Commas**: Adds trailing commas for multi-line structures
4. **Blank Lines**: Two blank lines around top-level definitions
5. **Imports**: Does NOT sort imports (use isort for that)

## Common Patterns

### Format Check in CI
```yaml
# GitHub Actions
- name: Check formatting with Black
  run: black --check --diff .
```

### Integration with isort
```toml
# pyproject.toml
[tool.isort]
profile = "black"  # Makes isort compatible with Black
```

### Excluding Files
```bash
# Via command line
black --exclude '/migrations/' src/

# Via pyproject.toml (see configuration above)
```

## Jupyter Notebook Support

```bash
# Install with Jupyter support
pip install 'black[jupyter]'

# Format notebooks
black notebook.ipynb

# Format only code cells (skip markdown)
black --ipynb notebook.ipynb
```

## Magic Comments

```python
# fmt: off
# Disables formatting for subsequent code

# fmt: on
# Re-enables formatting

# fmt: skip
# Skip formatting for a single statement
x=1  # fmt: skip
```

## Best Practices

1. **Run Black before other linters** - Formatting first, then lint
2. **Use with isort** - Black doesn't sort imports
3. **CI Integration** - Add `black --check` to CI pipeline
4. **Pre-commit hooks** - Automate formatting before commits
5. **Team agreement** - Use default settings when possible

## Links

- Documentation: https://black.readthedocs.io/
- GitHub: https://github.com/psf/black
- Playground: https://black.vercel.app/
