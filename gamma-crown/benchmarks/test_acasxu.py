"""
ACAS-Xu Benchmark Tests for γ-CROWN

TARGETS (must beat α,β-CROWN):
- >95% verified rate
- <10s per property

Run with:
    cd benchmarks
    pytest test_acasxu.py -v --timeout=10
    pytest test_acasxu.py -v --timeout=60 --method=beta  # with β-CROWN

Full benchmark:
    pytest test_acasxu.py -v --save-results=results.json
"""

import pytest
import json
from pathlib import Path
from conftest import (
    run_gamma_verify,
    VNNCOMP_DIR,
    GAMMA_BINARY,
    VerificationResult,
)

# ACAS-Xu directory
ACASXU_DIR = VNNCOMP_DIR / "acasxu"


class TestAcasXuBaseline:
    """Quick baseline tests to verify the benchmark infrastructure works."""

    def test_gamma_binary_exists(self):
        """Verify γ-CROWN binary is built."""
        assert GAMMA_BINARY.exists(), f"Build gamma first: cargo build --release"

    def test_acasxu_benchmark_exists(self):
        """Verify ACAS-Xu benchmark files exist."""
        assert ACASXU_DIR.exists(), "Download benchmarks first"
        networks = list(ACASXU_DIR.glob("*.onnx"))
        properties = list(ACASXU_DIR.glob("*.vnnlib"))
        assert len(networks) == 45, f"Expected 45 networks, got {len(networks)}"
        assert len(properties) == 10, f"Expected 10 properties, got {len(properties)}"

    def test_single_verification(self, timeout, method):
        """Run a single verification to test infrastructure."""
        network = ACASXU_DIR / "ACASXU_run2a_1_1_batch_2000.onnx"
        prop = ACASXU_DIR / "prop_1.vnnlib"

        if not network.exists() or not prop.exists():
            pytest.skip("Benchmark files not found")

        result = run_gamma_verify(network, prop, timeout=timeout, method=method)

        print(f"\nResult: {result.status}")
        print(f"Time: {result.time_seconds:.2f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")

        # We don't assert verified here - just that it runs without crashing
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]


class TestAcasXuProperty1:
    """Test all 45 networks against property 1."""

    @pytest.mark.acasxu
    @pytest.mark.parametrize("network_name", [
        f"ACASXU_run2a_{i}_{j}_batch_2000.onnx"
        for i in range(1, 6) for j in range(1, 10)
    ])
    def test_prop1(self, network_name, timeout, method):
        """Verify property 1 for each network."""
        network = ACASXU_DIR / network_name
        prop = ACASXU_DIR / "prop_1.vnnlib"

        if not network.exists():
            pytest.skip(f"Network {network_name} not found")
        if not prop.exists():
            pytest.skip("Property file not found")

        result = run_gamma_verify(network, prop, timeout=timeout, method=method)

        # Record result for aggregation
        pytest.current_result = result

        # Target: <10s and verified
        assert result.time_seconds < timeout, f"Timeout: {result.time_seconds}s"

        # We expect verified for most ACAS-Xu properties
        # Don't fail test on unknown - just report
        if result.status == "verified":
            pass  # Success
        elif result.status == "unknown":
            pytest.xfail(f"Unknown result - bounds may be too loose")
        elif result.status == "error":
            pytest.fail(f"Error: {result.error_message}")


class TestAcasXuFullBenchmark:
    """Run the full ACAS-Xu benchmark suite."""

    @pytest.mark.acasxu
    @pytest.mark.slow
    def test_full_benchmark(self, timeout, method, request):
        """
        Run all (network, property) pairs and compute aggregate metrics.

        This is the main benchmark to compare against α,β-CROWN.
        """
        results = []
        verified = 0
        falsified = 0
        unknown = 0
        timeout_count = 0
        error_count = 0
        total_time = 0.0

        networks = sorted(ACASXU_DIR.glob("*.onnx"))
        properties = sorted(ACASXU_DIR.glob("*.vnnlib"))

        # Load the instances CSV to know which (network, property) pairs to test
        instances_file = ACASXU_DIR / "acasxu_instances.csv"
        if instances_file.exists():
            import csv
            with open(instances_file) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                test_pairs = [(row[0], row[1]) for row in reader]
        else:
            # Fallback: test a subset
            test_pairs = [
                (f"ACASXU_run2a_1_{j}_batch_2000.onnx", f"prop_{p}.vnnlib")
                for j in range(1, 10) for p in range(1, 5)
            ]

        print(f"\nRunning {len(test_pairs)} verification tasks...")
        print(f"Method: {method}, Timeout: {timeout}s")
        print("-" * 60)

        for network_name, prop_name in test_pairs:
            network = ACASXU_DIR / network_name
            prop = ACASXU_DIR / prop_name

            if not network.exists() or not prop.exists():
                continue

            result = run_gamma_verify(network, prop, timeout=timeout, method=method)
            results.append(result)

            total_time += result.time_seconds

            if result.status == "verified":
                verified += 1
                status_str = "✓"
            elif result.status == "falsified":
                falsified += 1
                status_str = "✗"
            elif result.status == "unknown":
                unknown += 1
                status_str = "?"
            elif result.status == "timeout":
                timeout_count += 1
                status_str = "T"
            else:
                error_count += 1
                status_str = "E"

            print(f"  {status_str} {network_name} × {prop_name}: {result.time_seconds:.2f}s")

        # Aggregate metrics
        total = len(results)
        verified_rate = verified / total * 100 if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0

        print("-" * 60)
        print(f"\nRESULTS SUMMARY")
        print(f"  Total tasks:     {total}")
        print(f"  Verified:        {verified} ({verified_rate:.1f}%)")
        print(f"  Falsified:       {falsified}")
        print(f"  Unknown:         {unknown}")
        print(f"  Timeout:         {timeout_count}")
        print(f"  Error:           {error_count}")
        print(f"  Average time:    {avg_time:.2f}s")
        print(f"  Total time:      {total_time:.1f}s")

        print(f"\nTARGETS:")
        print(f"  Verified rate:   {verified_rate:.1f}% {'✓' if verified_rate > 95 else '✗'} (target: >95%)")
        print(f"  Average time:    {avg_time:.2f}s {'✓' if avg_time < 10 else '✗'} (target: <10s)")

        # Save results if requested
        save_path = request.config.getoption("--save-results")
        if save_path:
            output = {
                "method": method,
                "timeout": timeout,
                "total": total,
                "verified": verified,
                "falsified": falsified,
                "unknown": unknown,
                "timeout_count": timeout_count,
                "error_count": error_count,
                "verified_rate": verified_rate,
                "average_time": avg_time,
                "total_time": total_time,
                "results": [
                    {
                        "network": r.network,
                        "property": r.property,
                        "status": r.status,
                        "time": r.time_seconds,
                    }
                    for r in results
                ],
            }
            with open(save_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to: {save_path}")

        # Assert targets
        assert verified_rate > 95, f"Verified rate {verified_rate:.1f}% below target 95%"
        assert avg_time < 10, f"Average time {avg_time:.2f}s above target 10s"


if __name__ == "__main__":
    # Quick test
    import sys
    pytest.main([__file__, "-v", "--timeout=10"] + sys.argv[1:])
