"""
pytest configuration for γ-CROWN benchmarks.

VNN-COMP 2021-2025 benchmark suite.
https://github.com/VNN-COMP/vnncomp2025_benchmarks

Targets to beat α,β-CROWN:
- >95% verified rate on ACAS-Xu
- <10s per property

Run:
    cd benchmarks
    pytest -v --timeout=10
    pytest test_acasxu.py -v --timeout=10 --method=crown
    pytest test_vnncomp.py -v --timeout=60 --method=beta
"""

import pytest
import subprocess
import json
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# Benchmark directories
BENCHMARK_DIR = Path(__file__).parent
GAMMA_BINARY = BENCHMARK_DIR.parent / "target" / "release" / "gamma"

# VNN-COMP directories by year
# https://github.com/VNN-COMP/vnncomp2025_benchmarks
VNNCOMP_YEARS = {
    2021: BENCHMARK_DIR / "vnncomp2021" / "benchmarks",
    2023: BENCHMARK_DIR / "vnncomp2023" / "benchmarks",
    2024: BENCHMARK_DIR / "vnncomp2024" / "benchmarks",
    2025: BENCHMARK_DIR / "vnncomp2025" / "benchmarks",
}

# Legacy alias for backwards compatibility
VNNCOMP_DIR = VNNCOMP_YEARS[2021]


@dataclass
class VerificationResult:
    """Result of a single verification task."""
    network: str
    property: str
    status: str  # "verified", "falsified", "unknown", "timeout", "error"
    time_seconds: float
    bounds: Optional[list] = None
    error_message: Optional[str] = None


def pytest_configure(config):
    """Register custom markers."""
    # Benchmark categories
    config.addinivalue_line("markers", "acasxu: ACAS-Xu collision avoidance benchmark")
    config.addinivalue_line("markers", "mnist: MNIST digit classification benchmark")
    config.addinivalue_line("markers", "cifar: CIFAR image classification benchmark")
    config.addinivalue_line("markers", "vit: Vision Transformer benchmark")
    config.addinivalue_line("markers", "vggnet: VGGNet benchmark")
    config.addinivalue_line("markers", "nn4sys: NN4Sys systems benchmark")
    config.addinivalue_line("markers", "cgan: CGAN benchmark")
    config.addinivalue_line("markers", "yolo: YOLO object detection benchmark")

    # Year markers
    config.addinivalue_line("markers", "vnn2021: VNN-COMP 2021 benchmarks")
    config.addinivalue_line("markers", "vnn2023: VNN-COMP 2023 benchmarks")
    config.addinivalue_line("markers", "vnn2024: VNN-COMP 2024 benchmarks")
    config.addinivalue_line("markers", "vnn2025: VNN-COMP 2025 benchmarks")

    # Speed markers
    config.addinivalue_line("markers", "slow: marks tests as slow (>60s)")
    config.addinivalue_line("markers", "fast: marks tests as fast (<10s)")


def pytest_addoption(parser):
    """Add command-line options."""
    parser.addoption(
        "--timeout",
        action="store",
        default=10,
        type=int,
        help="Timeout per verification task in seconds (default: 10)",
    )
    parser.addoption(
        "--method",
        action="store",
        default="crown",
        help="Verification method: ibp, crown, beta (default: crown)",
    )
    parser.addoption(
        "--save-results",
        action="store",
        default=None,
        help="Save results to JSON file",
    )


@pytest.fixture
def timeout(request):
    """Get timeout from command line."""
    return request.config.getoption("--timeout")


@pytest.fixture
def method(request):
    """Get verification method from command line."""
    return request.config.getoption("--method")


def run_gamma_verify(network_path: Path, vnnlib_path: Path, timeout: int = 10, method: str = "crown") -> VerificationResult:
    """
    Run γ-CROWN verification on a single (network, property) pair.

    Returns VerificationResult with status, time, and bounds.
    """
    start = time.time()

    try:
        # For beta method, use beta-crown subcommand which has branch-and-bound
        if method == "beta":
            cmd = [
                str(GAMMA_BINARY),
                "beta-crown",
                str(network_path),
                "--property", str(vnnlib_path),
                "--timeout", str(timeout),
                "--branching", "input",   # Input splitting for ACAS-Xu (5D input)
                "--pgd-attack",           # Enable PGD attack for counterexample finding
                "--pgd-restarts", "5000", # High restarts for hard cases
                "--max-domains", "50000", # More domains for input splitting
                "--json",
            ]
        else:
            cmd = [
                str(GAMMA_BINARY),
                "verify",
                str(network_path),
                "--property", str(vnnlib_path),
                "--method", method,
                "--timeout", str(timeout),  # Timeout in seconds
                "--json",
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # Allow some buffer
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                # Use property_status for actual verification result
                # (status = process status, property_status = verification result)
                status = output.get("property_status", output.get("status", "unknown")).lower()
                # Normalize status values: "safe" -> "verified", "violated" -> "falsified"
                if status == "safe":
                    status = "verified"
                elif status == "violated":
                    status = "falsified"
                bounds = output.get("output_bounds", output.get("bounds"))
                return VerificationResult(
                    network=network_path.name,
                    property=vnnlib_path.name,
                    status=status,
                    time_seconds=elapsed,
                    bounds=bounds,
                )
            except json.JSONDecodeError:
                # Try to parse non-JSON output
                stdout = result.stdout.lower()
                if "verified" in stdout:
                    status = "verified"
                elif "falsified" in stdout or "violated" in stdout:
                    status = "falsified"
                else:
                    status = "unknown"
                return VerificationResult(
                    network=network_path.name,
                    property=vnnlib_path.name,
                    status=status,
                    time_seconds=elapsed,
                )
        else:
            return VerificationResult(
                network=network_path.name,
                property=vnnlib_path.name,
                status="error",
                time_seconds=elapsed,
                error_message=result.stderr[:500],
            )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            network=network_path.name,
            property=vnnlib_path.name,
            status="timeout",
            time_seconds=timeout,
        )
    except Exception as e:
        return VerificationResult(
            network=network_path.name,
            property=vnnlib_path.name,
            status="error",
            time_seconds=time.time() - start,
            error_message=str(e),
        )


# Collect ACAS-Xu test cases
def get_acasxu_test_cases():
    """Generate (network, property) pairs for ACAS-Xu benchmark."""
    acasxu_dir = VNNCOMP_DIR / "acasxu"
    if not acasxu_dir.exists():
        return []

    networks = sorted(acasxu_dir.glob("*.onnx"))
    properties = sorted(acasxu_dir.glob("*.vnnlib"))

    # The CSV file maps which properties apply to which networks
    # For simplicity, we'll test all combinations (some may be N/A)
    test_cases = []
    for network in networks:
        for prop in properties:
            test_cases.append((network, prop))

    return test_cases


ACASXU_TEST_CASES = get_acasxu_test_cases()


def get_benchmark_dir(year: int, benchmark: str) -> Optional[Path]:
    """Get benchmark directory for a given year and benchmark name."""
    if year not in VNNCOMP_YEARS:
        return None

    base = VNNCOMP_YEARS[year]
    if not base.exists():
        return None

    # Handle year suffixes (e.g., acasxu_2023 in 2024)
    candidates = [
        base / benchmark,
        base / f"{benchmark}_{year}",
        base / f"{benchmark}_{year - 1}",  # Sometimes uses previous year
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_benchmark_instances(year: int, benchmark: str) -> List[tuple]:
    """
    Get (network, property, timeout) tuples from instances.csv.

    Returns list of (network_path, property_path, timeout_seconds).
    """
    bench_dir = get_benchmark_dir(year, benchmark)
    if not bench_dir:
        return []

    instances_file = bench_dir / "instances.csv"
    if not instances_file.exists():
        # Fallback: look for acasxu_instances.csv etc.
        instances_file = bench_dir / f"{benchmark}_instances.csv"

    if not instances_file.exists():
        return []

    instances = []
    with open(instances_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].startswith("#") or row[0] == "network":
                continue  # Skip comments and headers

            network_name = row[0]
            prop_name = row[1]
            try:
                timeout = int(float(row[2])) if len(row) > 2 else 60
            except (ValueError, IndexError):
                timeout = 60

            # Find files - may be in onnx/ subdir or root
            # Also handle .gz files by trying without .gz extension
            network_candidates = [
                bench_dir / network_name,
                bench_dir / "onnx" / network_name,
            ]
            # If network ends with .gz, also try without .gz
            if network_name.endswith('.gz'):
                network_nogz = network_name[:-3]
                network_candidates.extend([
                    bench_dir / network_nogz,
                    bench_dir / "onnx" / network_nogz,
                ])

            prop_candidates = [
                bench_dir / prop_name,
                bench_dir / "vnnlib" / prop_name,
            ]

            network_path = None
            for p in network_candidates:
                if p.exists():
                    network_path = p
                    break

            prop_path = None
            for p in prop_candidates:
                if p.exists():
                    prop_path = p
                    break

            if network_path and prop_path:
                instances.append((network_path, prop_path, timeout))

    return instances


def run_benchmark_suite(
    year: int,
    benchmark: str,
    method: str = "crown",
    timeout_override: Optional[int] = None,
) -> Dict:
    """
    Run all instances in a benchmark and return aggregate statistics.

    Returns dict with verified/falsified/unknown/timeout/error counts and times.
    """
    instances = get_benchmark_instances(year, benchmark)

    results = {
        "year": year,
        "benchmark": benchmark,
        "method": method,
        "total": 0,
        "verified": 0,
        "falsified": 0,
        "unknown": 0,
        "timeout": 0,
        "error": 0,
        "total_time": 0.0,
        "instances": [],
    }

    for network_path, prop_path, default_timeout in instances:
        timeout = timeout_override or default_timeout
        result = run_gamma_verify(network_path, prop_path, timeout=timeout, method=method)

        results["total"] += 1
        results["total_time"] += result.time_seconds
        results[result.status] += 1
        results["instances"].append({
            "network": result.network,
            "property": result.property,
            "status": result.status,
            "time": result.time_seconds,
        })

    if results["total"] > 0:
        results["verified_rate"] = results["verified"] / results["total"] * 100
        results["avg_time"] = results["total_time"] / results["total"]
    else:
        results["verified_rate"] = 0.0
        results["avg_time"] = 0.0

    return results


# Available benchmarks by year
BENCHMARKS_BY_YEAR = {
    2021: ["acasxu", "mnistfc", "cifar10_resnet", "cifar2020", "nn4sys", "oval21"],
    2023: ["acasxu", "vit", "vggnet16", "nn4sys", "cgan", "yolo", "traffic_signs_recognition"],
    2024: ["acasxu_2023", "vit_2023", "vggnet16_2023", "cifar100", "cora", "safenlp", "tinyimagenet"],
    2025: ["acasxu_2023", "vit_2023", "nn4sys", "soundnessbench", "malbeware", "sat_relu", "cersyve"],
}
