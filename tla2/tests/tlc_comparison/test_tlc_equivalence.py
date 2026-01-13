"""
TLC Equivalence Tests

Verifies TLA2 produces semantically identical results to TLC.
These tests are the source of truth for correctness.
"""

import pytest
from pathlib import Path
from .conftest import run_tlc, run_tla2


class TestCoreAlgorithms:
    """Core algorithm specs - must match TLC exactly."""

    def test_diehard(self, examples_dir):
        """DieHard water jug puzzle - basic BFS."""
        spec = examples_dir / "DieHard" / "DieHard.tla"
        config = examples_dir / "DieHard" / "DieHard.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states, \
            f"State count mismatch: TLC={tlc.distinct_states}, TLA2={tla2.distinct_states}"
        assert tlc.has_error == tla2.has_error, \
            f"Error detection mismatch: TLC={tlc.has_error}, TLA2={tla2.has_error}"

    def test_tcommit(self, examples_dir):
        """Transaction commit - distributed commit protocol."""
        spec = examples_dir / "transaction_commit" / "TCommit.tla"
        config = examples_dir / "transaction_commit" / "TCommit.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states
        assert tlc.has_error == tla2.has_error

    def test_twophase(self, examples_dir):
        """Two-phase commit protocol."""
        spec = examples_dir / "transaction_commit" / "TwoPhase.tla"
        config = examples_dir / "transaction_commit" / "TwoPhase.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states
        assert tlc.has_error == tla2.has_error


class TestMutualExclusion:
    """Mutual exclusion algorithms."""

    def test_peterson(self, examples_dir):
        """Peterson's algorithm with TLAPS proofs."""
        spec = examples_dir / "locks_auxiliary_vars" / "Peterson.tla"
        config = examples_dir / "locks_auxiliary_vars" / "Peterson.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        # Skip if TLC can't parse (missing TLAPS module)
        if tlc.error_type == 'parse':
            pytest.skip("TLC missing TLAPS module")

        assert tlc.distinct_states == tla2.distinct_states
        assert tlc.has_error == tla2.has_error

    def test_lock(self, examples_dir):
        """Simple lock with auxiliary variables."""
        spec = examples_dir / "locks_auxiliary_vars" / "Lock.tla"
        config = examples_dir / "locks_auxiliary_vars" / "Lock.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        if tlc.error_type == 'parse':
            pytest.skip("TLC missing TLAPS module")

        assert tlc.distinct_states == tla2.distinct_states
        assert tlc.has_error == tla2.has_error

    def test_mcbakery(self, examples_dir, test_specs_dir):
        """Lamport's Bakery algorithm with TLAPS."""
        spec = examples_dir / "Bakery-Boulangerie" / "MCBakery.tla"
        config = test_specs_dir / "MCBakery.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        if tlc.error_type == 'parse':
            pytest.skip("TLC missing TLAPS module")

        assert tlc.distinct_states == tla2.distinct_states, \
            f"MCBakery: TLC={tlc.distinct_states}, TLA2={tla2.distinct_states}"
        assert tlc.has_error == tla2.has_error

    def test_mcboulanger_small(self, examples_dir, test_specs_dir):
        """Boulangerie algorithm (small config) - exercises prime-dependent IF extraction."""
        spec = examples_dir / "Bakery-Boulangerie" / "MCBoulanger.tla"
        config = test_specs_dir / "MCBoulanger_small.cfg"

        tlc = run_tlc(spec, config, timeout=60)
        tla2 = run_tla2(spec, config, timeout=60)

        if tlc.error_type == 'parse':
            pytest.skip("TLC parse error on MCBoulanger")

        assert tlc.distinct_states == tla2.distinct_states, \
            f"MCBoulanger_small: TLC={tlc.distinct_states}, TLA2={tla2.distinct_states}"
        assert tlc.has_error == tla2.has_error


class TestDistributedSystems:
    """Distributed algorithm specs."""

    def test_tokenring(self, examples_dir):
        """EWD426 Token Ring termination detection."""
        spec = examples_dir / "ewd426" / "TokenRing.tla"
        config = examples_dir / "ewd426" / "TokenRing.cfg"

        tlc = run_tlc(spec, config, timeout=180)
        tla2 = run_tla2(spec, config, timeout=180)

        assert tlc.distinct_states == tla2.distinct_states, \
            f"TokenRing: TLC={tlc.distinct_states}, TLA2={tla2.distinct_states}"

    def test_mcchangroberts(self, examples_dir):
        """Chang-Roberts leader election."""
        spec = examples_dir / "chang_roberts" / "MCChangRoberts.tla"
        config = examples_dir / "chang_roberts" / "MCChangRoberts.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states

    def test_huang(self, examples_dir):
        """Huang termination detection."""
        spec = examples_dir / "Huang" / "Huang.tla"
        config = examples_dir / "Huang" / "Huang.cfg"

        tlc = run_tlc(spec, config, timeout=180)
        tla2 = run_tla2(spec, config, timeout=180)

        assert tlc.distinct_states == tla2.distinct_states, \
            f"Huang: TLC={tlc.distinct_states}, TLA2={tla2.distinct_states}"


class TestResourceAllocation:
    """Resource allocation specs."""

    def test_simple_allocator(self, examples_dir):
        """Simple resource allocator."""
        spec = examples_dir / "allocator" / "SimpleAllocator.tla"
        config = examples_dir / "allocator" / "SimpleAllocator.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states

    def test_scheduling_allocator(self, examples_dir):
        """Scheduling allocator with liveness."""
        spec = examples_dir / "allocator" / "SchedulingAllocator.tla"
        config = examples_dir / "allocator" / "SchedulingAllocator.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states, \
            f"SchedulingAllocator: TLC={tlc.distinct_states}, TLA2={tla2.distinct_states}"


class TestClassicProblems:
    """Classic CS problems."""

    def test_dining_philosophers(self, examples_dir):
        """Dining philosophers deadlock detection."""
        spec = examples_dir / "DiningPhilosophers" / "DiningPhilosophers.tla"
        config = examples_dir / "DiningPhilosophers" / "DiningPhilosophers.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states

    def test_missionaries_and_cannibals(self, examples_dir):
        """River crossing puzzle."""
        spec = examples_dir / "MissionariesAndCannibals" / "MissionariesAndCannibals.tla"
        config = examples_dir / "MissionariesAndCannibals" / "MissionariesAndCannibals.cfg"

        tlc = run_tlc(spec, config)
        tla2 = run_tla2(spec, config)

        assert tlc.distinct_states == tla2.distinct_states


# Parametrized test for quick smoke testing
# Format: (name, spec_path, config_path, expected_states, expects_error)
SMOKE_TEST_SPECS = [
    # DieHard: NotSolved invariant is violated when puzzle is solved (expected behavior)
    ("DieHard", "DieHard/DieHard.tla", "DieHard/DieHard.cfg", 14, True),
    ("TCommit", "transaction_commit/TCommit.tla", "transaction_commit/TCommit.cfg", 34, False),
    ("DiningPhilosophers", "DiningPhilosophers/DiningPhilosophers.tla",
     "DiningPhilosophers/DiningPhilosophers.cfg", 67, False),
]


@pytest.mark.parametrize("name,spec_path,config_path,expected_states,expects_error", SMOKE_TEST_SPECS)
def test_smoke(examples_dir, name, spec_path, config_path, expected_states, expects_error):
    """Quick smoke test with known expected values."""
    spec = examples_dir / spec_path
    config = examples_dir / config_path

    tla2 = run_tla2(spec, config)

    assert tla2.distinct_states == expected_states, \
        f"{name}: expected {expected_states} states, got {tla2.distinct_states}"
    assert tla2.has_error == expects_error, \
        f"{name}: expected has_error={expects_error}, got {tla2.has_error}"
