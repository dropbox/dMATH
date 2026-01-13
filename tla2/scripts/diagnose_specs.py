#!/usr/bin/env python3
"""Quick diagnostic of TLA+ Examples spec status."""

import subprocess
import sys
import os
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "tlc_comparison"))
from spec_catalog import ALL_SPECS, SKIP_SPECS

TLA2_BIN = Path(__file__).parent.parent / "target" / "release" / "tla"
EXAMPLES_DIR = Path.home() / "tlaplus-examples" / "specifications"
TIMEOUT = 60

def run_tla2(spec_path, cfg_path):
    """Run TLA2 and return (states, error_type)."""
    try:
        result = subprocess.run(
            [str(TLA2_BIN), "check", str(spec_path), "--config", str(cfg_path), "--workers", "1"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        output = result.stdout + result.stderr

        # Parse state count
        for line in output.split("\n"):
            if "States found:" in line:
                try:
                    states = int(line.split(":")[-1].strip())
                    return states, None
                except:
                    pass

        # Check for errors
        if "error" in output.lower() or result.returncode != 0:
            if "Module" in output and "not found" in output:
                return None, "missing_module"
            if "Unsupported" in output:
                return None, "unsupported"
            if "parse" in output.lower() or "Parse" in output:
                return None, "parse"
            return None, "other"

        return None, "unknown"
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, f"exception: {e}"

def run_tlc(spec_path, cfg_path):
    """Run TLC and return state count."""
    try:
        result = subprocess.run(
            ["java", "-jar", str(Path.home() / "tlaplus" / "tla2tools.jar"),
             "-config", str(cfg_path), str(spec_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=str(Path.home() / "tlaplus"),
        )
        output = result.stdout + result.stderr

        # Parse state count
        for line in output.split("\n"):
            if "distinct states found" in line:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "distinct":
                            return int(parts[i-1].replace(",", ""))
                except:
                    pass

        if "Cannot find source file" in output:
            return None

        return None
    except:
        return None

def main():
    results = {"pass": [], "fail_mismatch": [], "fail_tla2": [], "fail_tlc": [], "skip": []}

    print(f"Testing {len(ALL_SPECS)} specs...")

    for i, spec in enumerate(ALL_SPECS):
        if spec.name in SKIP_SPECS:
            results["skip"].append((spec.name, SKIP_SPECS[spec.name]))
            continue

        spec_path = EXAMPLES_DIR / spec.tla_path
        cfg_path = EXAMPLES_DIR / spec.cfg_path

        if not spec_path.exists() or not cfg_path.exists():
            results["skip"].append((spec.name, "file not found"))
            continue

        tla2_states, tla2_err = run_tla2(spec_path, cfg_path)

        if tla2_err:
            # TLA2 failed - check if TLC also fails
            tlc_states = run_tlc(spec_path, cfg_path)
            if tlc_states is None:
                results["skip"].append((spec.name, f"both fail: {tla2_err}"))
            else:
                results["fail_tla2"].append((spec.name, tla2_err, tlc_states))
        else:
            # TLA2 succeeded - compare with TLC
            tlc_states = run_tlc(spec_path, cfg_path)
            if tlc_states is None:
                results["fail_tlc"].append((spec.name, tla2_states))
            elif tla2_states == tlc_states:
                results["pass"].append((spec.name, tla2_states))
            else:
                results["fail_mismatch"].append((spec.name, tla2_states, tlc_states))

        # Progress
        done = len(results["pass"]) + len(results["fail_mismatch"]) + len(results["fail_tla2"]) + len(results["fail_tlc"]) + len(results["skip"])
        print(f"\r[{done}/{len(ALL_SPECS)}] PASS:{len(results['pass'])} MISMATCH:{len(results['fail_mismatch'])} TLA2_FAIL:{len(results['fail_tla2'])} TLC_FAIL:{len(results['fail_tlc'])} SKIP:{len(results['skip'])}", end="", flush=True)

    print("\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PASS (exact match):     {len(results['pass'])}")
    print(f"STATE COUNT MISMATCH:   {len(results['fail_mismatch'])}")
    print(f"TLA2 FAILS (TLC works): {len(results['fail_tla2'])}")
    print(f"TLC FAILS (TLA2 works): {len(results['fail_tlc'])}")
    print(f"SKIPPED:                {len(results['skip'])}")
    print()

    if results["fail_mismatch"]:
        print("STATE COUNT MISMATCHES:")
        for name, tla2, tlc in results["fail_mismatch"]:
            print(f"  {name}: TLA2={tla2} TLC={tlc}")
        print()

    if results["fail_tla2"]:
        print("TLA2 FAILURES (TLC works):")
        for name, err, tlc in results["fail_tla2"]:
            print(f"  {name}: {err} (TLC: {tlc} states)")
        print()

    if results["fail_tlc"]:
        print("TLC FAILURES (TLA2 works):")
        for name, tla2 in results["fail_tlc"]:
            print(f"  {name}: TLA2={tla2} states")
        print()

if __name__ == "__main__":
    main()
