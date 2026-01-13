"""
TLC Comparison Test Framework

Compares TLA2 output against TLC (Java baseline) for semantic equivalence.
"""

import os
import re
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pytest


@dataclass
class CheckResult:
    """Result of running a model checker."""
    states: int
    distinct_states: int
    has_error: bool
    error_type: Optional[str]  # "invariant", "deadlock", "liveness", etc.
    raw_output: str
    exit_code: int


def parse_tlc_output(output: str) -> CheckResult:
    """Parse TLC output to extract state count and error info."""
    # Find the FINAL summary line: "N states generated, M distinct states found, K states left on queue."
    # This is the definitive line (not intermediate progress lines which have commas in numbers)
    # Match pattern: "12345 states generated, 6789 distinct states found, 0 states left"
    match = re.search(r'^(\d+) states generated, (\d+) distinct states found, (\d+) states left',
                      output, re.MULTILINE)
    if match:
        generated = int(match.group(1))
        distinct = int(match.group(2))
    else:
        # Fallback: try to find just distinct states (may be wrong if matches progress line)
        match = re.search(r'(\d+) distinct states found', output)
        distinct = int(match.group(1)) if match else 0
        match = re.search(r'(\d+) states generated', output)
        generated = int(match.group(1)) if match else 0

    # Check for errors
    has_error = False
    error_type = None

    if re.search(r'Error:', output):
        has_error = True
        if 'Invariant' in output and 'violated' in output:
            error_type = 'invariant'
        elif 'Deadlock' in output:
            error_type = 'deadlock'
        elif 'Parsing or semantic analysis failed' in output:
            error_type = 'parse'
        else:
            error_type = 'unknown'

    return CheckResult(
        states=generated,
        distinct_states=distinct,
        has_error=has_error,
        error_type=error_type,
        raw_output=output,
        exit_code=0  # Will be set by caller
    )


def parse_tla2_output(output: str) -> CheckResult:
    """Parse TLA2 output to extract state count and error info."""
    # Find states: "States found: N"
    match = re.search(r'States found:\s*(\d+)', output)
    states = int(match.group(1)) if match else 0

    # Check for errors
    has_error = False
    error_type = None

    if re.search(r'Error:', output, re.IGNORECASE):
        has_error = True
        if 'Invariant' in output and 'violated' in output.lower():
            error_type = 'invariant'
        elif 'Deadlock' in output:
            error_type = 'deadlock'
        elif 'Liveness property' in output:
            # TLA2 outputs "Liveness property 'X' violated!"
            error_type = 'liveness'
        elif re.search(r"Property '[^']+' violated", output):
            # TLA2 outputs "Property 'X' violated!" for safety properties
            # (non-liveness temporal properties like []P or <>P without fairness)
            error_type = 'safety'
        else:
            error_type = 'unknown'

    return CheckResult(
        states=states,
        distinct_states=states,  # TLA2 reports distinct by default
        has_error=has_error,
        error_type=error_type,
        raw_output=output,
        exit_code=0
    )


def run_tlc(spec: Path, config: Optional[Path] = None, timeout: int = 120) -> CheckResult:
    """Run TLC on a spec."""
    tlc_jar = Path.home() / "tlaplus" / "tla2tools.jar"
    if not tlc_jar.exists():
        pytest.skip(f"TLC not found at {tlc_jar}")

    # Path to TLAPS library modules (TLAPS.tla, FiniteSetTheorems.tla, etc.)
    repo_root = Path(__file__).parent.parent.parent
    tla_library = repo_root / "test_specs" / "tla_library"

    cmd = [
        "java", "-XX:+UseParallelGC", "-Xmx4g",
        f"-DTLA-Library={tla_library}",
        "-jar", str(tlc_jar),
        "-workers", "1"
    ]

    if config and config.exists():
        cmd.extend(["-config", str(config)])

    cmd.append(str(spec))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=spec.parent
        )
        output = result.stdout + result.stderr
        parsed = parse_tlc_output(output)
        parsed.exit_code = result.returncode
        return parsed
    except subprocess.TimeoutExpired:
        return CheckResult(
            states=0, distinct_states=0, has_error=True,
            error_type='timeout', raw_output=f"Timeout after {timeout}s",
            exit_code=-1
        )


def run_tla2(
    spec: Path,
    config: Optional[Path] = None,
    timeout: int = 120,
    extra_env: Optional[dict[str, str]] = None,
) -> CheckResult:
    """Run TLA2 on a spec."""
    repo_root = Path(__file__).parent.parent.parent
    tla2_bin = repo_root / "target" / "release" / "tla"

    if not tla2_bin.exists():
        # Try to build
        subprocess.run(["cargo", "build", "--release", "-p", "tla-cli"],
                      cwd=repo_root, capture_output=True)

    if not tla2_bin.exists():
        pytest.skip("TLA2 binary not found")

    cmd = [str(tla2_bin), "check", "--workers", "1"]

    if config and config.exists():
        cmd.extend(["--config", str(config)])

    cmd.append(str(spec))

    env = os.environ.copy()
    # Add TLA_PATH for TLAPS library modules
    tla_library = repo_root / "test_specs" / "tla_library"
    env["TLA_PATH"] = str(tla_library)
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = result.stdout + result.stderr
        parsed = parse_tla2_output(output)
        parsed.exit_code = result.returncode
        return parsed
    except subprocess.TimeoutExpired:
        return CheckResult(
            states=0, distinct_states=0, has_error=True,
            error_type='timeout', raw_output=f"Timeout after {timeout}s",
            exit_code=-1
        )


@pytest.fixture(scope="session")
def examples_dir():
    """Path to tlaplus-examples repo."""
    path = Path.home() / "tlaplus-examples" / "specifications"
    if not path.exists():
        pytest.skip("tlaplus-examples not found")
    return path


@pytest.fixture(scope="session")
def test_specs_dir():
    """Path to TLA2 test specs."""
    return Path(__file__).parent.parent.parent / "test_specs"


def pytest_addoption(parser):
    """Add --live-tlc option for running live TLC comparisons."""
    parser.addoption(
        "--live-tlc",
        action="store_true",
        default=False,
        help="Run live TLC comparison instead of using cached results"
    )


@pytest.fixture
def use_live_tlc(request):
    """Check if live TLC mode is enabled."""
    return request.config.getoption("--live-tlc", default=False)
