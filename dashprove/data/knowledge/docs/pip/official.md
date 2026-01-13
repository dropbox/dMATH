# pip - Python Package Installer

pip is the package installer for Python. It is the standard tool for installing Python packages from the Python Package Index (PyPI) and other indexes.

## Installation

```bash
# Usually pre-installed with Python
# To ensure latest version:
python -m pip install --upgrade pip

# Bootstrap pip if missing
python -m ensurepip --upgrade

# Install specific version
python -m pip install pip==24.0
```

## Basic Usage

### Installing Packages

```bash
# Install a package
pip install requests

# Install specific version
pip install requests==2.31.0

# Install with version constraints
pip install "requests>=2.25.0,<3.0.0"

# Install from requirements.txt
pip install -r requirements.txt

# Install editable (development mode)
pip install -e .

# Install with extras
pip install "fastapi[all]"

# Install from git
pip install git+https://github.com/user/repo.git

# Install from local archive
pip install ./package.tar.gz
```

### Uninstalling Packages

```bash
# Uninstall a package
pip uninstall requests

# Uninstall without confirmation
pip uninstall -y requests

# Uninstall multiple packages
pip uninstall requests urllib3 certifi
```

### Listing Packages

```bash
# List installed packages
pip list

# List outdated packages
pip list --outdated

# Show package details
pip show requests

# Freeze installed packages
pip freeze > requirements.txt
```

### Upgrading Packages

```bash
# Upgrade a package
pip install --upgrade requests

# Upgrade pip itself
pip install --upgrade pip

# Upgrade all packages (not recommended)
pip list --outdated --format=freeze | cut -d= -f1 | xargs -n1 pip install -U
```

## Requirements Files

### requirements.txt Format

```text
# Exact versions (recommended for production)
requests==2.31.0
numpy==1.26.0

# Version ranges
django>=4.0,<5.0
flask~=3.0.0

# Git dependencies
git+https://github.com/user/repo.git@v1.0.0

# Local packages
-e ./my-local-package

# Comments
# This is a comment

# Environment markers
pywin32; sys_platform == 'win32'

# Extra index
--extra-index-url https://download.pytorch.org/whl/cpu
```

### Constraints Files

```bash
# Install with constraints
pip install -c constraints.txt -r requirements.txt
```

```text
# constraints.txt - pin transitive dependencies
urllib3==2.0.0
certifi==2023.7.22
```

## Configuration

### pip.conf / pip.ini

```ini
# Unix: ~/.config/pip/pip.conf
# macOS: ~/Library/Application Support/pip/pip.conf
# Windows: %APPDATA%\pip\pip.ini

[global]
timeout = 60
index-url = https://pypi.org/simple
trusted-host = pypi.org
               files.pythonhosted.org

[install]
no-cache-dir = false

[freeze]
timeout = 10
```

### Environment Variables

```bash
# Custom index
PIP_INDEX_URL=https://private.pypi.org/simple

# Extra index
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Disable cache
PIP_NO_CACHE_DIR=1

# Trusted host
PIP_TRUSTED_HOST=private.pypi.org
```

## Advanced Usage

### Hash Checking

```bash
# Generate requirements with hashes
pip hash ./package.tar.gz

# requirements.txt with hashes
pip install --require-hashes -r requirements.txt
```

```text
# requirements.txt with hashes
requests==2.31.0 \
    --hash=sha256:abc123... \
    --hash=sha256:def456...
```

### Wheel Files

```bash
# Download wheel without installing
pip download requests

# Build wheel from source
pip wheel .

# Install only from wheels
pip install --only-binary :all: requests
```

### Cache Management

```bash
# Show cache info
pip cache info

# List cached packages
pip cache list

# Clear cache
pip cache purge

# Remove specific package from cache
pip cache remove requests
```

### Dependency Resolution

```bash
# Check for dependency conflicts
pip check

# Show dependency tree (with pipdeptree)
pip install pipdeptree
pipdeptree

# Dry run install
pip install --dry-run requests
```

## Virtual Environments

```bash
# Create virtual environment
python -m venv .venv

# Activate (Unix)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install in venv
pip install requests

# Deactivate
deactivate
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt

- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: pip-${{ hashFiles('requirements.txt') }}
```

### Docker

```dockerfile
# Best practice: use --no-cache-dir in containers
RUN pip install --no-cache-dir -r requirements.txt

# Or: multi-stage build
FROM python:3.12-slim as builder
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
```

## Common Patterns

### Development vs Production

```text
# requirements.txt (production)
django==4.2.0
gunicorn==21.0.0

# requirements-dev.txt (development)
-r requirements.txt
pytest==8.0.0
black==24.0.0
```

### Pin All Dependencies

```bash
# Generate fully pinned requirements
pip freeze > requirements-lock.txt
```

## Security

```bash
# Check for vulnerabilities (use pip-audit)
pip install pip-audit
pip-audit

# Install from trusted sources only
pip install --trusted-host pypi.org requests
```

## Troubleshooting

```bash
# Verbose output
pip install -v requests

# Very verbose
pip install -vvv requests

# Debug dependency resolution
pip install --report install-report.json requests
```

## Best Practices

1. **Always use virtual environments**
2. **Pin exact versions** in production
3. **Use hash checking** for security
4. **Cache in CI/CD** for faster builds
5. **Consider uv** for faster installs
6. **Run pip check** to verify dependencies

## Links

- Documentation: https://pip.pypa.io/
- GitHub: https://github.com/pypa/pip
- PyPI: https://pypi.org/project/pip/
