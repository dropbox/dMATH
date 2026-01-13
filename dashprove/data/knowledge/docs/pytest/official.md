# pytest - Python Testing Framework

pytest is a mature full-featured Python testing tool that helps you write better programs. It supports simple unit tests to complex functional testing.

## Installation

```bash
pip install pytest
pip install pytest-cov pytest-xdist pytest-mock  # Common plugins
```

## Basic Usage

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest test_module.py

# Run specific test
pytest test_module.py::test_function
pytest test_module.py::TestClass::test_method

# Run tests matching pattern
pytest -k "test_login"
pytest -k "test_login or test_logout"
pytest -k "not slow"

# Verbose output
pytest -v
pytest -vv  # More verbose

# Stop on first failure
pytest -x
pytest --maxfail=3
```

### Test Discovery

pytest finds tests by:
- Files matching `test_*.py` or `*_test.py`
- Classes starting with `Test`
- Functions starting with `test_`

## Writing Tests

### Simple Test

```python
def test_addition():
    assert 1 + 1 == 2

def test_list():
    items = [1, 2, 3]
    assert len(items) == 3
    assert 2 in items
```

### Test Classes

```python
class TestCalculator:
    def test_add(self):
        assert add(2, 3) == 5

    def test_subtract(self):
        assert subtract(5, 3) == 2
```

### Assertions

```python
# Simple assertion
assert x == y
assert x != y
assert x is None
assert x is not None
assert x in collection

# With message
assert x == y, f"Expected {y}, got {x}"

# Approximate equality
assert x == pytest.approx(0.1, rel=1e-3)
assert x == pytest.approx(0.1, abs=1e-5)
```

### Testing Exceptions

```python
import pytest

def test_raises():
    with pytest.raises(ValueError):
        raise ValueError("error")

def test_raises_with_match():
    with pytest.raises(ValueError, match="invalid"):
        raise ValueError("invalid input")

def test_raises_capture():
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("error message")
    assert "error" in str(exc_info.value)
```

## Fixtures

### Basic Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_data(sample_data):
    assert sample_data["key"] == "value"
```

### Fixture Scopes

```python
@pytest.fixture(scope="function")  # Default, per test
def func_fixture():
    return "per-test"

@pytest.fixture(scope="class")
def class_fixture():
    return "per-class"

@pytest.fixture(scope="module")
def module_fixture():
    return "per-module"

@pytest.fixture(scope="session")
def session_fixture():
    return "per-session"
```

### Setup and Teardown

```python
@pytest.fixture
def database():
    # Setup
    db = connect_database()
    yield db
    # Teardown
    db.close()

def test_query(database):
    result = database.query("SELECT 1")
    assert result is not None
```

### Autouse Fixtures

```python
@pytest.fixture(autouse=True)
def reset_state():
    # Runs before every test automatically
    yield
    # Cleanup after
```

### Parametrized Fixtures

```python
@pytest.fixture(params=["mysql", "postgres", "sqlite"])
def database(request):
    return connect(request.param)

def test_db(database):  # Runs 3 times
    assert database.is_connected()
```

## Parametrization

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert input * 2 == expected

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
], ids=["positive", "zero", "mixed"])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

## Markers

### Built-in Markers

```python
@pytest.mark.skip(reason="Not implemented")
def test_unimplemented():
    pass

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_new_feature():
    pass

@pytest.mark.xfail(reason="Known bug")
def test_known_failure():
    assert False

@pytest.mark.slow
def test_slow():
    time.sleep(10)
```

### Custom Markers

```python
# pytest.ini
[pytest]
markers =
    slow: marks tests as slow
    integration: marks as integration test

# Usage
@pytest.mark.slow
def test_slow():
    pass

# Run only marked tests
# pytest -m slow
# pytest -m "not slow"
```

## Plugins

### pytest-cov (Coverage)

```bash
pytest --cov=mypackage
pytest --cov=mypackage --cov-report=html
pytest --cov=mypackage --cov-report=xml
pytest --cov=mypackage --cov-fail-under=80
```

### pytest-xdist (Parallel)

```bash
pytest -n auto          # Auto-detect CPUs
pytest -n 4             # 4 workers
pytest -n auto --dist=loadfile  # Group by file
```

### pytest-mock

```python
def test_with_mock(mocker):
    mock_func = mocker.patch("module.function")
    mock_func.return_value = "mocked"

    result = code_that_uses_function()

    mock_func.assert_called_once()
    assert result == "mocked"
```

## Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
markers =
    slow: slow tests
    integration: integration tests
filterwarnings =
    error
    ignore::DeprecationWarning
```

### pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --strict-markers"
markers = [
    "slow: slow tests",
    "integration: integration tests",
]
```

### conftest.py

```python
# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def app():
    """Create application for testing."""
    return create_app(testing=True)

def pytest_configure(config):
    """Custom pytest configuration."""
    pass

def pytest_collection_modifyitems(items):
    """Modify test collection."""
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
```

## Output and Reporting

```bash
# Show print statements
pytest -s
pytest --capture=no

# Show local variables on failure
pytest -l
pytest --showlocals

# Detailed traceback
pytest --tb=short   # Short traceback
pytest --tb=long    # Full traceback
pytest --tb=no      # No traceback

# Duration of slow tests
pytest --durations=10

# JUnit XML output
pytest --junitxml=report.xml
```

## Documentation

- Official: https://docs.pytest.org/
- Plugins: https://docs.pytest.org/en/latest/reference/plugin_list.html
- Fixtures: https://docs.pytest.org/en/latest/explanation/fixtures.html
