# isort - Python Import Sorter

isort is a Python utility to sort imports alphabetically and automatically separated into sections and by type. It provides a command line utility, Python library, and plugins for various editors.

## Installation

```bash
# Using pip
pip install isort

# Using pipx
pipx install isort

# Using uv
uv tool install isort

# With specific plugins
pip install isort[colors]
pip install isort[requirements_deprecated_finder]
```

## Basic Usage

```bash
# Sort imports in a file
isort script.py

# Sort imports in a directory
isort src/

# Check without modifying (exit code 1 if changes needed)
isort --check-only src/

# Show diff without modifying
isort --diff src/

# Sort imports in stdin
echo "import sys\nimport os" | isort -

# Show what would change
isort --diff --check-only src/
```

## Configuration

### pyproject.toml (Recommended)

```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
skip_gitignore = true
known_first_party = ["myproject"]
known_third_party = ["django", "requests"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
```

### setup.cfg

```ini
[isort]
profile = black
line_length = 88
multi_line_output = 3
include_trailing_comma = True
```

## Import Section Order

Default sections (in order):
1. **FUTURE** - `from __future__ import ...`
2. **STDLIB** - Standard library imports
3. **THIRDPARTY** - Third-party packages
4. **FIRSTPARTY** - Your project's packages
5. **LOCALFOLDER** - Relative imports

## Multi-line Output Modes

```python
# Mode 0 - Hanging indent
from package import (a,
    b, c, d)

# Mode 1 - Vertical hanging indent
from package import (a,
    b,
    c,
    d)

# Mode 2 - Vertical hanging indent (bracket on separate line)
from package import (
    a,
    b,
    c,
    d
)

# Mode 3 - Vertical hanging indent with trailing comma (Black compatible)
from package import (
    a,
    b,
    c,
    d,
)

# Mode 4 - NOQA
from package import a, b, c, d  # NOQA

# Mode 5 - Wrap at line length
from package import (
    a, b, c, d)
```

## Profiles

```bash
# Use Black-compatible profile
isort --profile black src/

# Use Django profile
isort --profile django src/

# Use PEP 8 profile
isort --profile pep8 src/
```

Available profiles: `black`, `django`, `pycharm`, `google`, `open_stack`, `plone`, `attrs`, `hug`, `wemake`, `appnexus`

## Pre-commit Integration

```yaml
repos:
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        name: isort (python)
```

## Editor Integration

### VS Code
```json
{
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.isort"
    }
}
```

## Magic Comments

```python
# isort: skip_file
# Skip entire file from sorting

import unsorted_import  # isort: skip
# Skip this specific line

# isort: off
import manually_ordered
import another_manual
# isort: on

# isort: split
# Force a new section here
```

## Common Patterns

### Integration with Black

```toml
# pyproject.toml - Black-compatible settings
[tool.isort]
profile = "black"
```

### Custom Section Order

```toml
[tool.isort]
sections = ["FUTURE", "STDLIB", "DJANGO", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
known_django = ["django"]
```

### Force Single Line Imports

```toml
[tool.isort]
force_single_line = true
```

### Sort by Length

```toml
[tool.isort]
length_sort = true
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Check import sorting
  run: isort --check-only --diff .
```

### Combining with Black

```bash
# Run isort before Black
isort . && black .

# Or in one command
isort . --profile black && black .
```

## Best Practices

1. **Always use profile = "black"** when using Black formatter
2. **Define known_first_party** for your project packages
3. **Use pre-commit hooks** for automatic sorting
4. **Run in CI** with `--check-only --diff`
5. **Configure in pyproject.toml** for single source of truth

## Links

- Documentation: https://pycqa.github.io/isort/
- GitHub: https://github.com/PyCQA/isort
- Configuration Options: https://pycqa.github.io/isort/docs/configuration/options.html
