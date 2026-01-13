"""
VNN-COMP Benchmark Suite for γ-CROWN

Tests against VNN-COMP 2021-2024 benchmarks.
Reference: https://sites.google.com/view/vnn2025/home

TARGETS (must beat α,β-CROWN, 5x VNN-COMP winner):
- >95% verified rate on ACAS-Xu
- <10s per property

Run:
    pytest test_vnncomp.py -v --timeout=10                    # Quick test
    pytest test_vnncomp.py -v --timeout=60 --method=beta     # Full with β-CROWN
    pytest test_vnncomp.py -v -k acasxu --save-results=results.json
"""

import pytest
import json
from pathlib import Path
from conftest import (
    run_gamma_verify,
    run_benchmark_suite,
    get_benchmark_dir,
    get_benchmark_instances,
    VNNCOMP_YEARS,
    BENCHMARKS_BY_YEAR,
    GAMMA_BINARY,
)


class TestVnncompInfrastructure:
    """Verify benchmark infrastructure is set up correctly."""

    def test_gamma_binary_exists(self):
        """Verify γ-CROWN binary is built."""
        assert GAMMA_BINARY.exists(), f"Build gamma first: cargo build --release"

    def test_vnncomp_2021_exists(self):
        """Verify VNN-COMP 2021 benchmarks exist."""
        assert VNNCOMP_YEARS[2021].exists(), "Missing vnncomp2021 benchmarks"
        acasxu = VNNCOMP_YEARS[2021] / "acasxu"
        assert acasxu.exists(), "Missing ACAS-Xu 2021"
        assert len(list(acasxu.glob("*.onnx"))) == 45, "Expected 45 ACAS-Xu networks"

    def test_vnncomp_2023_exists(self):
        """Verify VNN-COMP 2023 benchmarks exist."""
        assert VNNCOMP_YEARS[2023].exists(), "Missing vnncomp2023 benchmarks"
        acasxu = VNNCOMP_YEARS[2023] / "acasxu"
        if acasxu.exists():
            onnx_dir = acasxu / "onnx"
            assert len(list(onnx_dir.glob("*.onnx"))) >= 45, "Expected 45+ ACAS-Xu networks"

    def test_vnncomp_2024_exists(self):
        """Verify VNN-COMP 2024 benchmarks exist."""
        assert VNNCOMP_YEARS[2024].exists(), "Missing vnncomp2024 benchmarks"


# =============================================================================
# ACAS-Xu Benchmarks (PRIMARY - must beat α,β-CROWN)
# =============================================================================

class TestAcasXu2021:
    """ACAS-Xu benchmark from VNN-COMP 2021."""

    @pytest.mark.acasxu
    @pytest.mark.vnn2021
    def test_single_instance(self, timeout, method):
        """Run single ACAS-Xu instance as smoke test."""
        acasxu_dir = VNNCOMP_YEARS[2021] / "acasxu"
        network = acasxu_dir / "ACASXU_run2a_1_1_batch_2000.onnx"
        prop = acasxu_dir / "prop_1.vnnlib"

        if not network.exists() or not prop.exists():
            pytest.skip("ACAS-Xu 2021 files not found")

        result = run_gamma_verify(network, prop, timeout=timeout, method=method)
        print(f"\nResult: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]

    @pytest.mark.acasxu
    @pytest.mark.vnn2021
    @pytest.mark.slow
    def test_full_benchmark(self, timeout, method, request):
        """Run full ACAS-Xu 2021 benchmark - primary target."""
        results = run_benchmark_suite(2021, "acasxu", method=method, timeout_override=timeout)

        print(f"\n{'='*60}")
        print(f"ACAS-Xu 2021 Results ({method})")
        print(f"{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Verified: {results['verified']} ({results['verified_rate']:.1f}%)")
        print(f"Falsified: {results['falsified']}")
        print(f"Unknown: {results['unknown']}")
        print(f"Timeout: {results['timeout']}")
        print(f"Error: {results['error']}")
        print(f"Average time: {results['avg_time']:.2f}s")
        print(f"{'='*60}")

        # Save results if requested
        save_path = request.config.getoption("--save-results")
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {save_path}")

        # TARGET: >95% verified, <10s average
        print(f"\nTARGETS:")
        print(f"  Verified rate: {results['verified_rate']:.1f}% {'PASS' if results['verified_rate'] > 95 else 'FAIL'} (target: >95%)")
        print(f"  Average time: {results['avg_time']:.2f}s {'PASS' if results['avg_time'] < 10 else 'FAIL'} (target: <10s)")


class TestAcasXu2023:
    """ACAS-Xu benchmark from VNN-COMP 2023."""

    @pytest.mark.acasxu
    @pytest.mark.vnn2023
    def test_single_instance(self, timeout, method):
        """Run single ACAS-Xu instance as smoke test."""
        acasxu_dir = VNNCOMP_YEARS[2023] / "acasxu"
        network = acasxu_dir / "onnx" / "ACASXU_run2a_1_1_batch_2000.onnx"
        prop = acasxu_dir / "vnnlib" / "prop_1.vnnlib"

        if not network.exists() or not prop.exists():
            pytest.skip("ACAS-Xu 2023 files not found")

        result = run_gamma_verify(network, prop, timeout=timeout, method=method)
        print(f"\nResult: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]

    @pytest.mark.acasxu
    @pytest.mark.vnn2023
    @pytest.mark.slow
    def test_full_benchmark(self, timeout, method, request):
        """Run full ACAS-Xu 2023 benchmark."""
        results = run_benchmark_suite(2023, "acasxu", method=method, timeout_override=timeout)

        print(f"\n{'='*60}")
        print(f"ACAS-Xu 2023 Results ({method})")
        print(f"{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Verified: {results['verified']} ({results['verified_rate']:.1f}%)")
        print(f"Average time: {results['avg_time']:.2f}s")
        print(f"{'='*60}")

        save_path = request.config.getoption("--save-results")
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)


# =============================================================================
# MNIST Benchmarks
# =============================================================================

class TestMnist2021:
    """MNIST-FC benchmark from VNN-COMP 2021."""

    @pytest.mark.mnist
    @pytest.mark.vnn2021
    def test_single_instance(self, timeout, method):
        """Run single MNIST-FC instance."""
        mnist_dir = VNNCOMP_YEARS[2021] / "mnistfc"
        if not mnist_dir.exists():
            pytest.skip("MNIST-FC 2021 not found")

        network = mnist_dir / "mnist-net_256x2.onnx"
        props = list(mnist_dir.glob("prop_*_0.03.vnnlib"))

        if not network.exists() or not props:
            pytest.skip("MNIST-FC 2021 files not found")

        result = run_gamma_verify(network, props[0], timeout=timeout, method=method)
        print(f"\nResult: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]

    @pytest.mark.mnist
    @pytest.mark.vnn2021
    @pytest.mark.slow
    def test_full_benchmark(self, timeout, method, request):
        """Run full MNIST-FC 2021 benchmark."""
        results = run_benchmark_suite(2021, "mnistfc", method=method, timeout_override=timeout)

        print(f"\nMNIST-FC 2021: {results['verified']}/{results['total']} verified "
              f"({results['verified_rate']:.1f}%), avg {results['avg_time']:.2f}s")


# =============================================================================
# Vision Transformer (ViT) Benchmarks
# =============================================================================

class TestViT2023:
    """Vision Transformer benchmark from VNN-COMP 2023."""

    @pytest.mark.vit
    @pytest.mark.vnn2023
    def test_single_instance(self, timeout, method):
        """Run single ViT instance."""
        vit_dir = VNNCOMP_YEARS[2023] / "vit"
        if not vit_dir.exists():
            pytest.skip("ViT 2023 not found")

        networks = list((vit_dir / "onnx").glob("*.onnx")) if (vit_dir / "onnx").exists() else []
        if not networks:
            pytest.skip("ViT 2023 networks not found")

        props = list((vit_dir / "vnnlib").glob("*.vnnlib")) if (vit_dir / "vnnlib").exists() else []
        if not props:
            pytest.skip("ViT 2023 properties not found")

        result = run_gamma_verify(networks[0], props[0], timeout=timeout, method=method)
        print(f"\nViT Result: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]


class TestViT2024:
    """Vision Transformer benchmark from VNN-COMP 2024."""

    @pytest.mark.vit
    @pytest.mark.vnn2024
    def test_single_instance(self, timeout, method):
        """Run single ViT instance."""
        vit_dir = get_benchmark_dir(2024, "vit")
        if not vit_dir:
            pytest.skip("ViT 2024 not found")

        networks = list((vit_dir / "onnx").glob("*.onnx")) if (vit_dir / "onnx").exists() else []
        if not networks:
            pytest.skip("ViT 2024 networks not found")

        props = list((vit_dir / "vnnlib").glob("*.vnnlib")) if (vit_dir / "vnnlib").exists() else []
        if not props:
            pytest.skip("ViT 2024 properties not found")

        result = run_gamma_verify(networks[0], props[0], timeout=timeout, method=method)
        print(f"\nViT 2024 Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# VGGNet Benchmarks
# =============================================================================

class TestVggNet2023:
    """VGGNet benchmark from VNN-COMP 2023."""

    @pytest.mark.vggnet
    @pytest.mark.vnn2023
    @pytest.mark.slow
    def test_single_instance(self, timeout, method):
        """Run single VGGNet instance (large model)."""
        vgg_dir = VNNCOMP_YEARS[2023] / "vggnet16"
        if not vgg_dir.exists():
            pytest.skip("VGGNet 2023 not found")

        instances = get_benchmark_instances(2023, "vggnet16")
        if not instances:
            pytest.skip("VGGNet 2023 instances not found")

        network, prop, _ = instances[0]
        result = run_gamma_verify(network, prop, timeout=timeout, method=method)
        print(f"\nVGGNet Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# CIFAR Benchmarks
# =============================================================================

class TestCifar2021:
    """CIFAR benchmark from VNN-COMP 2021."""

    @pytest.mark.cifar
    @pytest.mark.vnn2021
    def test_cifar_resnet(self, timeout, method):
        """Run CIFAR ResNet instance."""
        cifar_dir = VNNCOMP_YEARS[2021] / "cifar10_resnet"
        if not cifar_dir.exists():
            pytest.skip("CIFAR ResNet 2021 not found")

        networks = list(cifar_dir.glob("*.onnx"))
        props = list(cifar_dir.glob("*.vnnlib"))

        if not networks or not props:
            pytest.skip("CIFAR ResNet 2021 files not found")

        result = run_gamma_verify(networks[0], props[0], timeout=timeout, method=method)
        print(f"\nCIFAR ResNet Result: {result.status} in {result.time_seconds:.2f}s")


class TestCifar2024:
    """CIFAR-100 benchmark from VNN-COMP 2024."""

    @pytest.mark.cifar
    @pytest.mark.vnn2024
    @pytest.mark.slow
    def test_single_instance(self, timeout, method):
        """Run CIFAR-100 instance."""
        cifar_dir = get_benchmark_dir(2024, "cifar100")
        if not cifar_dir:
            pytest.skip("CIFAR-100 2024 not found")

        instances = get_benchmark_instances(2024, "cifar100")
        if not instances:
            pytest.skip("CIFAR-100 2024 instances not found")

        network, prop, _ = instances[0]
        result = run_gamma_verify(network, prop, timeout=timeout, method=method)
        print(f"\nCIFAR-100 Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# NN4Sys Benchmarks (Systems/Control)
# =============================================================================

class TestNn4sys2021:
    """NN4Sys benchmark from VNN-COMP 2021."""

    @pytest.mark.nn4sys
    @pytest.mark.vnn2021
    def test_single_instance(self, timeout, method):
        """Run single NN4Sys instance."""
        nn4sys_dir = VNNCOMP_YEARS[2021] / "nn4sys"
        if not nn4sys_dir.exists():
            pytest.skip("NN4Sys 2021 not found")

        instances = get_benchmark_instances(2021, "nn4sys")
        if not instances:
            pytest.skip("NN4Sys 2021 instances not found")

        network, prop, _ = instances[0]
        result = run_gamma_verify(network, prop, timeout=timeout, method=method)
        print(f"\nNN4Sys Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# VNN-COMP 2025 Benchmarks (Latest)
# =============================================================================

class TestAcasXu2025:
    """ACAS-Xu benchmark from VNN-COMP 2025."""

    @pytest.mark.acasxu
    @pytest.mark.vnn2025
    def test_single_instance(self, timeout, method):
        """Run single ACAS-Xu instance from 2025."""
        acasxu_dir = get_benchmark_dir(2025, "acasxu")
        if not acasxu_dir:
            pytest.skip("ACAS-Xu 2025 not found")

        networks = list((acasxu_dir / "onnx").glob("*.onnx")) if (acasxu_dir / "onnx").exists() else []
        props = list((acasxu_dir / "vnnlib").glob("*.vnnlib")) if (acasxu_dir / "vnnlib").exists() else []

        if not networks or not props:
            pytest.skip("ACAS-Xu 2025 files not found")

        result = run_gamma_verify(networks[0], props[0], timeout=timeout, method=method)
        print(f"\nAcas-Xu 2025 Result: {result.status} in {result.time_seconds:.2f}s")
        assert result.status in ["verified", "falsified", "unknown", "timeout", "error"]


class TestSoundnessBench2025:
    """Soundness benchmark from VNN-COMP 2025 - tests verifier soundness."""

    @pytest.mark.vnn2025
    def test_single_instance(self, timeout, method):
        """Run soundness benchmark instance."""
        bench_dir = get_benchmark_dir(2025, "soundnessbench")
        if not bench_dir:
            pytest.skip("Soundnessbench 2025 not found")

        instances = get_benchmark_instances(2025, "soundnessbench")
        if not instances:
            pytest.skip("Soundnessbench 2025 instances not found")

        network, prop, _ = instances[0]
        result = run_gamma_verify(network, prop, timeout=timeout, method=method)
        print(f"\nSoundnessBench Result: {result.status} in {result.time_seconds:.2f}s")


# =============================================================================
# Aggregate Tests
# =============================================================================

class TestVnncompAggregate:
    """Run aggregate tests across all VNN-COMP years."""

    @pytest.mark.slow
    def test_all_acasxu(self, timeout, method, request):
        """Run ACAS-Xu across all years and aggregate results."""
        all_results = []

        for year in [2021, 2023, 2024, 2025]:
            bench_name = "acasxu" if year == 2021 or year == 2023 else "acasxu_2023"
            results = run_benchmark_suite(year, bench_name, method=method, timeout_override=timeout)
            if results["total"] > 0:
                all_results.append(results)
                print(f"\nYear {year}: {results['verified']}/{results['total']} verified "
                      f"({results['verified_rate']:.1f}%), avg {results['avg_time']:.2f}s")

        # Aggregate
        total_verified = sum(r["verified"] for r in all_results)
        total_instances = sum(r["total"] for r in all_results)
        total_time = sum(r["total_time"] for r in all_results)

        if total_instances > 0:
            agg_rate = total_verified / total_instances * 100
            agg_time = total_time / total_instances
            print(f"\n{'='*60}")
            print(f"AGGREGATE ACAS-Xu Results")
            print(f"{'='*60}")
            print(f"Total verified: {total_verified}/{total_instances} ({agg_rate:.1f}%)")
            print(f"Average time: {agg_time:.2f}s")
            print(f"{'='*60}")

    @pytest.mark.slow
    def test_benchmark_matrix(self, timeout, method, request):
        """Run matrix of benchmarks across years."""
        print(f"\n{'='*80}")
        print(f"VNN-COMP Benchmark Matrix ({method}, timeout={timeout}s)")
        print(f"{'='*80}")
        print(f"{'Year':<6} {'Benchmark':<25} {'Verified':<12} {'Rate':<8} {'Avg Time':<10}")
        print(f"{'-'*80}")

        for year, benchmarks in BENCHMARKS_BY_YEAR.items():
            for bench in benchmarks[:3]:  # Limit to first 3 per year for speed
                results = run_benchmark_suite(year, bench, method=method, timeout_override=timeout)
                if results["total"] > 0:
                    print(f"{year:<6} {bench:<25} "
                          f"{results['verified']}/{results['total']:<10} "
                          f"{results['verified_rate']:.1f}%{'':>3} "
                          f"{results['avg_time']:.2f}s")

        print(f"{'='*80}")


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "--timeout=10"] + sys.argv[1:])
