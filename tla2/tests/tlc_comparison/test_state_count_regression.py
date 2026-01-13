"""
State Count Regression Tests

Verifies that key specs produce exact state counts matching TLC.
These are regression tests for bugs that have been fixed in TLA2.

Each entry documents:
- The spec name and path
- Expected state count (verified against TLC)
- The bug/fix reference
- Whether an error is expected

This is verification tooling - state counts MUST match TLC exactly.

Part of #83 - regression test coverage for state counts.
"""

import pytest
from pathlib import Path
from typing import NamedTuple, Optional
from .conftest import run_tla2, run_tlc, CheckResult


class RegressionSpec(NamedTuple):
    """Specification for regression testing."""
    name: str
    tla_path: str
    cfg_path: str
    expected_states: int
    expects_error: bool
    error_type: Optional[str]
    timeout: int
    bug_ref: str  # Reference to bug/fix (issue number or commit)
    known_bug: Optional[str] = None  # Description of known TLA2 bug (for xfail)


# Regression test table: specs where bugs were fixed
# Each entry documents the TLC-verified state count
REGRESSION_SPECS = [
    # === Paxos Family ===
    # FastPaxos: Init enumeration fix for CONSTANT Ballot handling
    RegressionSpec(
        name="FastPaxos",
        tla_path="SimplifiedFastPaxos/FastPaxos.tla",
        cfg_path="SimplifiedFastPaxos/FastPaxos.cfg",
        expected_states=25617,  # TLC: 25617 distinct states
        expects_error=False,
        error_type=None,
        timeout=300,
        bug_ref="W#94",
    ),
    # PaxosCommit: Large spec, 1.3M states - #86: symmetry canonicalization bug
    RegressionSpec(
        name="PaxosCommit",
        tla_path="transaction_commit/PaxosCommit.tla",
        cfg_path="transaction_commit/PaxosCommit.cfg",
        expected_states=1321761,  # TLC: 1321761 distinct states (verified 2026-01-12)
        expects_error=False,
        error_type=None,
        timeout=600,  # ~107s TLC runtime
        bug_ref="#86",
        known_bug="#86: TLA2 over-explores by 19.5% (1,579,313 vs TLC 1,321,761) - symmetry canonicalization",
    ),
    # === Allocator Family ===
    # AllocatorImplementation: FIXED in 34b73b1 (#87)
    RegressionSpec(
        name="AllocatorImplementation",
        tla_path="allocator/AllocatorImplementation.tla",
        cfg_path="allocator/AllocatorImplementation.cfg",
        expected_states=17701,  # TLC: 17701 distinct states, no errors
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="34b73b1/#87",
    ),
    # AllocatorRefinement: Config operator replacement fix (W#45)
    RegressionSpec(
        name="AllocatorRefinement",
        tla_path="allocator/AllocatorRefinement.tla",
        cfg_path="allocator/AllocatorRefinement.cfg",
        expected_states=1690,  # TLC: 1690 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="W#45/14da02d",
    ),
    # === YoYo (W#45 fixes) ===
    # MCYoYoPruning: Config operator replacement fix
    RegressionSpec(
        name="MCYoYoPruning",
        tla_path="YoYo/MCYoYoPruning.tla",
        cfg_path="YoYo/MCYoYoPruning.cfg",
        expected_states=102,  # TLC: 102 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="W#45/14da02d",
    ),
    # MCYoYoNoPruning: Working spec
    RegressionSpec(
        name="MCYoYoNoPruning",
        tla_path="YoYo/MCYoYoNoPruning.tla",
        cfg_path="YoYo/MCYoYoNoPruning.cfg",
        expected_states=60,  # TLC: 60 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="verified 2026-01-05",
    ),
    # === Mutex (W#93 fixes) ===
    # dijkstra-mutex: Stuttering edge fix for liveness
    RegressionSpec(
        name="dijkstra-mutex_LSpec-model",
        tla_path="dijkstra-mutex/DijkstraMutex.toolbox/LSpec-model/MC.tla",
        cfg_path="dijkstra-mutex/DijkstraMutex.toolbox/LSpec-model/MC.cfg",
        expected_states=90882,  # TLC: 90882 distinct states
        expects_error=False,
        error_type=None,
        timeout=300,  # ~13s TLC, allow buffer
        bug_ref="W#93/b2b2179",
    ),
    # === Cache specs ===
    # MCWriteThroughCache: TLA2 reports false positive property violation
    RegressionSpec(
        name="MCWriteThroughCache",
        tla_path="SpecifyingSystems/CachingMemory/MCWriteThroughCache.tla",
        cfg_path="SpecifyingSystems/CachingMemory/MCWriteThroughCache.cfg",
        expected_states=5196,  # TLC: 5196 distinct states, no errors
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="need fix",
        known_bug="TLA2 reports false positive 'Property LM_Inner_ISpec violated' - TLC finds no errors",
    ),
    # MCLiveInternalMemory: Working correctly
    RegressionSpec(
        name="MCLiveInternalMemory",
        tla_path="SpecifyingSystems/Liveness/MCLiveInternalMemory.tla",
        cfg_path="SpecifyingSystems/Liveness/MCLiveInternalMemory.cfg",
        expected_states=4408,  # TLC: 4408 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="W#45/14da02d",
    ),
    # MCLiveWriteThroughCache: TLA2 reports false positive property violation
    RegressionSpec(
        name="MCLiveWriteThroughCache",
        tla_path="SpecifyingSystems/Liveness/MCLiveWriteThroughCache.tla",
        cfg_path="SpecifyingSystems/Liveness/MCLiveWriteThroughCache.cfg",
        expected_states=5196,  # TLC: 5196 distinct states, no errors
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="need fix",
        known_bug="TLA2 reports false positive property violation - TLC finds no errors",
    ),
    # === Cat Puzzle (f7e6163 fix) ===
    # CatEvenBoxes: Stuttering edge fix
    RegressionSpec(
        name="CatEvenBoxes",
        tla_path="Moving_Cat_Puzzle/Cat.tla",
        cfg_path="Moving_Cat_Puzzle/CatEvenBoxes.cfg",
        expected_states=48,  # TLC: 48 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="f7e6163",
    ),
    # CatOddBoxes: Stuttering edge fix
    RegressionSpec(
        name="CatOddBoxes",
        tla_path="Moving_Cat_Puzzle/Cat.tla",
        cfg_path="Moving_Cat_Puzzle/CatOddBoxes.cfg",
        expected_states=30,  # TLC: 30 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="f7e6163",
    ),
    # === Classic specs (baseline) ===
    # DieHard: Invariant violation expected (NotSolved violated when solved)
    RegressionSpec(
        name="DieHard",
        tla_path="DieHard/DieHard.tla",
        cfg_path="DieHard/DieHard.cfg",
        expected_states=14,  # TLC: 14 distinct states
        expects_error=True,
        error_type="invariant",
        timeout=60,
        bug_ref="baseline",
    ),
    # TCommit: Transaction commit baseline
    RegressionSpec(
        name="TCommit",
        tla_path="transaction_commit/TCommit.tla",
        cfg_path="transaction_commit/TCommit.cfg",
        expected_states=34,  # TLC: 34 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="baseline",
    ),
    # TwoPhase: Two-phase commit baseline
    RegressionSpec(
        name="TwoPhase",
        tla_path="transaction_commit/TwoPhase.tla",
        cfg_path="transaction_commit/TwoPhase.cfg",
        expected_states=288,  # TLC: 288 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="baseline",
    ),
    # DiningPhilosophers: Classic baseline
    RegressionSpec(
        name="DiningPhilosophers",
        tla_path="DiningPhilosophers/DiningPhilosophers.tla",
        cfg_path="DiningPhilosophers/DiningPhilosophers.cfg",
        expected_states=67,  # TLC: 67 distinct states
        expects_error=False,
        error_type=None,
        timeout=60,
        bug_ref="baseline",
    ),
]


def _make_test_id(spec: RegressionSpec) -> str:
    """Create readable test ID."""
    return spec.name


def _make_marks(spec: RegressionSpec):
    """Generate pytest marks for a spec."""
    marks = []
    if spec.known_bug:
        marks.append(pytest.mark.xfail(reason=f"[BUG] {spec.known_bug}", strict=False))
    if spec.timeout >= 300:
        marks.append(pytest.mark.slow)
    return marks


# Build parametrized list with marks
REGRESSION_PARAMS = [
    pytest.param(spec, id=_make_test_id(spec), marks=_make_marks(spec))
    for spec in REGRESSION_SPECS
]


@pytest.mark.parametrize("spec", REGRESSION_PARAMS)
def test_state_count_regression(examples_dir: Path, spec: RegressionSpec):
    """Verify TLA2 state count matches expected TLC baseline.

    This is a regression test - if this fails, a previously fixed bug
    may have regressed, or the expected count was wrong.
    """
    spec_path = examples_dir / spec.tla_path
    config_path = examples_dir / spec.cfg_path

    if not spec_path.exists():
        pytest.skip(f"Spec not found: {spec_path}")
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    # Run TLA2
    result = run_tla2(spec_path, config_path, timeout=spec.timeout)

    # Check for timeout
    if result.error_type == "timeout":
        pytest.fail(
            f"{spec.name}: TLA2 timed out after {spec.timeout}s "
            f"(bug ref: {spec.bug_ref})"
        )

    # Verify error detection matches expected
    if spec.expects_error:
        assert result.has_error, (
            f"{spec.name}: Expected error ({spec.error_type}) but TLA2 found none "
            f"(bug ref: {spec.bug_ref})"
        )
    else:
        assert not result.has_error, (
            f"{spec.name}: TLA2 found unexpected error ({result.error_type}) "
            f"(bug ref: {spec.bug_ref})\n"
            f"Output: {result.raw_output[-500:]}"
        )

    # Verify exact state count match
    assert result.distinct_states == spec.expected_states, (
        f"{spec.name}: State count mismatch (bug ref: {spec.bug_ref})\n"
        f"  Expected (TLC): {spec.expected_states}\n"
        f"  Got (TLA2):     {result.distinct_states}\n"
        f"  Delta:          {result.distinct_states - spec.expected_states}"
    )


# Fast-only specs for quick CI runs (excludes specs with timeout >= 300s)
FAST_REGRESSION_PARAMS = [
    pytest.param(spec, id=_make_test_id(spec), marks=_make_marks(spec))
    for spec in REGRESSION_SPECS
    if spec.timeout < 300
]


@pytest.mark.fast
@pytest.mark.parametrize("spec", FAST_REGRESSION_PARAMS)
def test_fast_state_count_regression(examples_dir: Path, spec: RegressionSpec):
    """Fast regression tests (< 300s timeout).

    These run in CI on every commit. Use -m fast to run only these.
    """
    spec_path = examples_dir / spec.tla_path
    config_path = examples_dir / spec.cfg_path

    if not spec_path.exists():
        pytest.skip(f"Spec not found: {spec_path}")
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    result = run_tla2(spec_path, config_path, timeout=spec.timeout)

    if result.error_type == "timeout":
        pytest.fail(f"{spec.name}: TLA2 timed out after {spec.timeout}s")

    if spec.expects_error:
        assert result.has_error, f"{spec.name}: Expected error but TLA2 found none"
    else:
        assert not result.has_error, (
            f"{spec.name}: TLA2 found unexpected error ({result.error_type})"
        )

    assert result.distinct_states == spec.expected_states, (
        f"{spec.name}: State count mismatch\n"
        f"  Expected: {spec.expected_states}, Got: {result.distinct_states}"
    )


class TestLiveComparison:
    """Live TLC comparison tests for validation.

    These tests run both TLA2 and TLC to verify the expected values
    are still correct. Run with --live-tlc flag.
    """

    @pytest.mark.parametrize(
        "spec",
        REGRESSION_SPECS[:5],  # Just first 5 for quick validation
        ids=[_make_test_id(s) for s in REGRESSION_SPECS[:5]]
    )
    def test_verify_expected_values(
        self, examples_dir: Path, spec: RegressionSpec, use_live_tlc: bool
    ):
        """Verify expected values match current TLC output."""
        if not use_live_tlc:
            pytest.skip("Use --live-tlc to run live TLC comparison")

        spec_path = examples_dir / spec.tla_path
        config_path = examples_dir / spec.cfg_path

        if not spec_path.exists():
            pytest.skip(f"Spec not found: {spec_path}")

        tlc = run_tlc(spec_path, config_path, timeout=spec.timeout)

        if tlc.error_type == "timeout":
            pytest.skip(f"TLC timed out after {spec.timeout}s")

        # Verify our expected values are correct
        assert tlc.distinct_states == spec.expected_states, (
            f"{spec.name}: Expected value may be wrong!\n"
            f"  Documented: {spec.expected_states}\n"
            f"  TLC says:   {tlc.distinct_states}\n"
            f"  Please update REGRESSION_SPECS"
        )
