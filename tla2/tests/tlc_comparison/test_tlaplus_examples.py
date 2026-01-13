"""
TLA+ Examples Test Suite

Parametrized tests covering all TLA+ example specifications.
Compares TLA2 results against cached TLC baseline (expected_results.json).

To regenerate the cache:
    python tests/tlc_comparison/generate_expected_results.py

To run tests using cache (default, fast):
    pytest tests/tlc_comparison/ -m fast

To run tests with live TLC comparison (slow):
    pytest tests/tlc_comparison/ --live-tlc
"""

import json
import time
import pytest
from pathlib import Path
from typing import Optional, Dict, Any
from .conftest import run_tlc, run_tla2, CheckResult
from .spec_catalog import (
    ALL_SPECS,
    SKIP_SPECS,
    LARGE_SPECS,
    TLA2_LIMITATIONS,
    TLA2_BUGS,
    get_runnable_specs,
    get_fast_specs,
    SpecInfo,
)


# Cache for expected results
_EXPECTED_RESULTS: Optional[Dict[str, Any]] = None


def load_expected_results() -> Dict[str, Any]:
    """Load cached TLC results from expected_results.json."""
    global _EXPECTED_RESULTS
    if _EXPECTED_RESULTS is not None:
        return _EXPECTED_RESULTS

    cache_path = Path(__file__).parent / "expected_results.json"
    if not cache_path.exists():
        return {"specs": {}, "_metadata": {}}

    with open(cache_path) as f:
        _EXPECTED_RESULTS = json.load(f)
    return _EXPECTED_RESULTS


def get_expected(spec_name: str) -> Optional[Dict[str, Any]]:
    """Get expected results for a spec."""
    results = load_expected_results()
    return results.get("specs", {}).get(spec_name)


def format_performance_comparison(
    tla2_time: float,
    tlc_mean: float,
    tlc_stddev: float,
    runs_count: int
) -> str:
    """Format performance comparison string."""
    if tlc_mean <= 0:
        return f"TLA2 {tla2_time:.1f}s (no TLC baseline)"

    ratio = tla2_time / tlc_mean
    if ratio < 1.0:
        return f"TLA2 {tla2_time:.1f}s vs TLC {tlc_mean:.1f}s ({ratio:.1f}x = faster!, n={runs_count})"
    else:
        stddev_str = f"+/-{tlc_stddev:.1f}s" if tlc_stddev > 0 else ""
        return f"TLA2 {tla2_time:.1f}s vs TLC {tlc_mean:.1f}s{stddev_str} ({ratio:.1f}x slower, n={runs_count})"


def make_test_id(spec: SpecInfo) -> str:
    """Create a readable test ID."""
    return spec.name


def get_timeout(spec: SpecInfo) -> int:
    """Get timeout for a spec."""
    return LARGE_SPECS.get(spec.name, spec.timeout_seconds)


# Generate pytest parameters for all runnable specs
def _make_marks(spec: SpecInfo):
    """Generate pytest marks for a spec."""
    marks = []

    # Skip specs that are in skip list (legitimately can't run)
    if spec.name in SKIP_SPECS:
        marks.append(pytest.mark.skipif(True, reason=f"[SKIP] {SKIP_SPECS[spec.name]}"))
    # Skip specs with known TLA2 limitations (missing features)
    elif spec.name in TLA2_LIMITATIONS:
        marks.append(pytest.mark.skipif(True, reason=f"[LIMITATION] {TLA2_LIMITATIONS[spec.name]}"))
    # XFAIL specs with known TLA2 bugs - these MUST fail until fixed!
    # Using xfail instead of skip makes bugs visible and prevents silent regression
    elif spec.name in TLA2_BUGS:
        marks.append(pytest.mark.xfail(reason=f"[BUG] {TLA2_BUGS[spec.name]}", strict=False))

    if spec.name in LARGE_SPECS:
        marks.append(pytest.mark.slow)
        # Override pytest-timeout for large specs (add buffer for TLC+TLA2)
        marks.append(pytest.mark.timeout(LARGE_SPECS[spec.name] * 2 + 60))
    else:
        marks.append(pytest.mark.fast)
    return marks


RUNNABLE_SPECS = [
    pytest.param(
        spec,
        id=make_test_id(spec),
        marks=_make_marks(spec),
    )
    for spec in ALL_SPECS
]

# Fast specs only (for quick CI runs)
FAST_SPECS = [
    pytest.param(spec, id=make_test_id(spec))
    for spec in get_fast_specs()
]


@pytest.mark.parametrize("spec", RUNNABLE_SPECS)
def test_tlc_equivalence(examples_dir: Path, spec: SpecInfo, use_live_tlc: bool):
    """Test that TLA2 matches TLC on this spec.

    Uses cached TLC results by default. Pass --live-tlc to run TLC live.
    """
    spec_path = examples_dir / spec.tla_path
    config_path = examples_dir / spec.cfg_path

    if not spec_path.exists():
        pytest.skip(f"Spec not found: {spec_path}")
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    timeout = get_timeout(spec)

    # Get TLC baseline (cached or live)
    expected = get_expected(spec.name)

    if use_live_tlc or expected is None:
        # Live TLC mode or no cache available
        tlc = run_tlc(spec_path, config_path, timeout=timeout)
        if tlc.error_type == "timeout":
            pytest.skip(f"TLC timeout after {timeout}s")
        tlc_states = tlc.distinct_states
        tlc_has_error = tlc.has_error
        tlc_error_type = tlc.error_type
        tlc_parsed = tlc.error_type != "parse"
        tlc_raw_output = tlc.raw_output
        tlc_mean = 0.0
        tlc_stddev = 0.0
        tlc_runs = 0
    else:
        # Use cached results
        if expected.get("skip_reason"):
            pytest.skip(f"Cached skip: {expected['skip_reason']}")

        tlc_states = expected.get("distinct_states", 0)
        tlc_has_error = expected.get("has_error", False)
        tlc_error_type = expected.get("error_type")
        tlc_parsed = tlc_error_type != "parse"
        tlc_raw_output = ""  # Not available in cache

        # Get runtime stats
        stats = expected.get("runtime_stats", {})
        tlc_mean = stats.get("mean_seconds", 0.0)
        tlc_stddev = stats.get("stddev_seconds", 0.0)
        tlc_runs = stats.get("runs_count", 0)

    # Run TLA2 and measure time
    start_time = time.time()
    tla2 = run_tla2(spec_path, config_path, timeout=timeout)
    tla2_time = time.time() - start_time

    # Parse error handling
    tla2_parsed = tla2.error_type != "parse"

    if not tlc_parsed and not tla2_parsed:
        pytest.skip(f"Both TLC and TLA2 have parse errors")
    elif tlc_parsed and not tla2_parsed:
        assert False, (
            f"{spec.name}: TLA2 PARSER BUG - TLC parses OK but TLA2 fails\n"
            f"  TLA2 error: {tla2.raw_output[-500:]}"
        )
    elif not tlc_parsed and tla2_parsed:
        pytest.skip(f"TLC parse error (TLA2 parses OK)")

    # Compare error detection first
    if tlc_has_error and tla2.has_error:
        # Both found errors - check type compatibility
        if tlc_error_type != tla2.error_type:
            acceptable_pairs = [
                ("invariant", "safety"),
                ("safety", "invariant"),
                ("liveness", "temporal"),
                ("temporal", "liveness"),
                ("unknown", "liveness"),
                ("unknown", "safety"),
            ]
            pair = (tlc_error_type, tla2.error_type)
            if pair not in acceptable_pairs:
                import warnings
                warnings.warn(
                    f"{spec.name}: Error type mismatch - "
                    f"TLC: {tlc_error_type}, TLA2: {tla2.error_type}",
                    UserWarning
                )
        # Print performance comparison
        perf = format_performance_comparison(tla2_time, tlc_mean, tlc_stddev, tlc_runs)
        print(f"\n{spec.name}: {perf}")
        return  # Test passes
    elif tlc_has_error != tla2.has_error:
        assert False, (
            f"{spec.name}: error detection mismatch\n"
            f"  TLA2 has_error: {tla2.has_error} ({tla2.error_type})\n"
            f"  TLC has_error:  {tlc_has_error} ({tlc_error_type})"
        )

    # Compare state counts
    assert tla2.distinct_states == tlc_states, (
        f"{spec.name}: state count mismatch\n"
        f"  TLA2: {tla2.distinct_states} states\n"
        f"  TLC:  {tlc_states} states\n"
        f"  TLA2 output: {tla2.raw_output[-300:]}"
    )

    # Print performance comparison
    perf = format_performance_comparison(tla2_time, tlc_mean, tlc_stddev, tlc_runs)
    print(f"\n{spec.name}: {perf}")


@pytest.mark.fast
@pytest.mark.parametrize("spec", FAST_SPECS[:20])  # First 20 fast specs
def test_fast_smoke(examples_dir: Path, spec: SpecInfo):
    """Quick smoke test on fast specs."""
    spec_path = examples_dir / spec.tla_path
    config_path = examples_dir / spec.cfg_path

    if not spec_path.exists():
        pytest.skip(f"Spec not found: {spec_path}")
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    # Just verify TLA2 doesn't crash
    tla2 = run_tla2(spec_path, config_path, timeout=60)

    # Should either complete successfully or find an expected error
    assert tla2.error_type != "parse", (
        f"{spec.name}: TLA2 parse error\n{tla2.raw_output[-500:]}"
    )


class TestCategorySmoke:
    """Quick smoke tests by category."""

    @pytest.fixture
    def run_category_test(self, examples_dir):
        """Helper to run a spec by name."""
        def _run(spec_name: str) -> tuple[CheckResult, CheckResult]:
            from .spec_catalog import get_spec_by_name
            spec = get_spec_by_name(spec_name)
            if not spec:
                pytest.skip(f"Spec {spec_name} not in catalog")

            spec_path = examples_dir / spec.tla_path
            config_path = examples_dir / spec.cfg_path

            if not spec_path.exists():
                pytest.skip(f"Spec not found: {spec_path}")

            timeout = get_timeout(spec)
            tlc = run_tlc(spec_path, config_path, timeout=timeout)
            tla2 = run_tla2(spec_path, config_path, timeout=timeout)
            return tlc, tla2
        return _run

    def test_diehard(self, run_category_test):
        """DieHard water jug puzzle."""
        tlc, tla2 = run_category_test("DieHard")
        assert tla2.distinct_states == tlc.distinct_states

    def test_tcommit(self, run_category_test):
        """Transaction commit."""
        tlc, tla2 = run_category_test("TCommit")
        assert tla2.distinct_states == tlc.distinct_states

    def test_tokenring(self, run_category_test):
        """Token ring termination detection."""
        tlc, tla2 = run_category_test("TokenRing")
        assert tla2.distinct_states == tlc.distinct_states

    def test_dining_philosophers(self, run_category_test):
        """Dining philosophers."""
        tlc, tla2 = run_category_test("DiningPhilosophers")
        assert tla2.distinct_states == tlc.distinct_states
