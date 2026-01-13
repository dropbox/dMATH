# Z4 Test Suite Extension Plan

**Date**: 2026-01-02
**Goal**: Comprehensive testing against all competitors with automated CI

---

## Current Test Coverage

### Z4 Existing Tests
```
benchmarks/
├── chc/           # CHC-COMP benchmarks
├── dimacs/        # DIMACS CNF files
├── qbf/           # QBF benchmarks
├── samples/       # Sample files
├── satlib/        # SATLIB benchmarks
├── smt/           # SMT-LIB files
└── smtcomp/       # SMT-COMP files
```

**Estimated total**: ~500 benchmarks

---

## Test Sources to Incorporate

### 1. CaDiCaL Test Suite

**Location**: `reference/cadical/test/`

| Directory | Contents | Count | Priority |
|-----------|----------|-------|----------|
| `cnf/` | DIMACS test files | 88 | High |
| `trace/` | Solver trace tests | ~50 | Medium |
| `api/` | API tests | ~20 | Low |
| `contrib/` | Contributed tests | ~10 | Medium |

**Tasks**:
```bash
# Copy CNF tests
cp -r reference/cadical/test/cnf/* benchmarks/dimacs/cadical/

# Create test runner for trace tests
scripts/run_cadical_traces.py
```

### 2. CryptoMiniSat Test Suite

**Location**: `reference/cryptominisat/tests/`

| File | Purpose | Priority |
|------|---------|----------|
| `cnf-files/` | CNF benchmarks | High |
| `gauss_test.cpp` | Gaussian elimination | High |
| `gate_test.cpp` | Gate detection | Medium |
| `distiller_test.cpp` | Distillation | Medium |
| `cardfinder_test.cpp` | Cardinality | Medium |

**XOR Test Generation**:
```bash
# Generate XOR-heavy benchmarks
python3 scripts/generate_xor_benchmarks.py --count 100 --vars 50-500
```

### 3. CVC5 Regression Suite

**Location**: `reference/cvc5/test/regress/`

| Directory | Contents | Count | Priority |
|-----------|----------|-------|----------|
| `regress0/` | Fast tests (<1s) | 2,500+ | High |
| `regress1/` | Medium tests (<10s) | 1,000+ | High |
| `regress2/` | Long tests (<60s) | 400+ | Medium |
| `regress3/` | Very long tests | 100+ | Low |

**Key Subdirectories**:
```
regress0/
├── arith/          # Arithmetic (QF_LIA, QF_LRA, QF_NIA)
├── bv/             # Bitvectors (QF_BV, QF_ABV)
├── strings/        # Strings (QF_S, QF_SLIA)
├── quantifiers/    # Quantified formulas
├── arrays/         # Arrays (QF_AX, QF_AUFLIA)
├── uf/             # Uninterpreted functions
├── datatypes/      # Algebraic datatypes
└── sets/           # Set theory
```

**Import Script**:
```bash
# Import CVC5 regression tests by category
python3 scripts/import_cvc5_tests.py \
  --source reference/cvc5/test/regress/ \
  --dest benchmarks/smt/cvc5-regress/ \
  --categories arith,bv,strings,arrays,uf
```

---

## External Benchmark Sources

### 4. SAT Competition Benchmarks

**Source**: https://satcompetition.github.io/

| Year | Main Track | Incremental | Cloud |
|------|------------|-------------|-------|
| 2023 | 400 | 100 | 200 |
| 2022 | 400 | 100 | 200 |
| 2021 | 400 | 100 | 200 |

**Download Script**:
```bash
#!/bin/bash
# scripts/download_satcomp.sh
YEAR=2023
wget -r -np -nd -A "*.cnf.xz" \
  https://satcompetition.github.io/${YEAR}/downloads/
xz -d *.cnf.xz
mv *.cnf benchmarks/sat/satcomp${YEAR}/
```

### 5. SMT-LIB Benchmarks

**Source**: https://smtlib.cs.uiowa.edu/benchmarks.shtml

| Logic | Count | Size | Priority |
|-------|-------|------|----------|
| QF_LIA | 10,000+ | 2GB | High |
| QF_LRA | 3,000+ | 500MB | High |
| QF_BV | 40,000+ | 10GB | High |
| QF_UF | 2,000+ | 200MB | High |
| QF_AUFLIA | 5,000+ | 1GB | Medium |
| QF_S | 20,000+ | 3GB | High |
| QF_SLIA | 5,000+ | 1GB | High |
| LIA | 5,000+ | 500MB | Medium |
| LRA | 3,000+ | 300MB | Medium |

**Download Script**:
```bash
#!/bin/bash
# scripts/download_smtlib.sh
for logic in QF_LIA QF_LRA QF_BV QF_UF QF_S QF_SLIA; do
  wget -r -np -nd -A "*.smt2.bz2" \
    https://clc-gitlab.cs.uiowa.edu:2443/SMT-LIB-benchmarks/${logic}/
  bunzip2 *.smt2.bz2
  mv *.smt2 benchmarks/smt/${logic}/
done
```

### 6. CHC Competition Benchmarks

**Source**: https://chc-comp.github.io/

| Category | Count | Priority |
|----------|-------|----------|
| LIA-Lin | 200+ | High |
| LIA-Nonlin | 100+ | High |
| LIA-Arrays | 100+ | Medium |
| ADT | 50+ | Medium |

**Download Script**:
```bash
#!/bin/bash
# scripts/download_chccomp.sh
git clone https://github.com/chc-comp/benchmarks \
  benchmarks/chc/chccomp-benchmarks
```

---

## Test Infrastructure

### 7. Benchmark Runner

**File**: `scripts/benchmark_runner.py`

```python
#!/usr/bin/env python3
"""
Unified benchmark runner for Z4 vs competitors.
"""

import subprocess
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

SOLVERS = {
    'z4': './target/release/z4',
    'z3': 'z3',
    'cvc5': 'cvc5',
    'cadical': 'reference/cadical/build/cadical',
}

def run_benchmark(solver: str, file: Path, timeout: int = 60):
    """Run single benchmark and return result."""
    start = time.time()
    try:
        result = subprocess.run(
            [SOLVERS[solver], str(file)],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        elapsed = time.time() - start
        return {
            'solver': solver,
            'file': str(file),
            'result': parse_result(result.stdout),
            'time': elapsed,
            'status': 'ok'
        }
    except subprocess.TimeoutExpired:
        return {
            'solver': solver,
            'file': str(file),
            'result': None,
            'time': timeout,
            'status': 'timeout'
        }

def run_suite(suite_dir: Path, solvers: list, timeout: int = 60):
    """Run benchmark suite and return comparison."""
    results = []
    files = list(suite_dir.glob('**/*.smt2')) + list(suite_dir.glob('**/*.cnf'))

    with ProcessPoolExecutor(max_workers=8) as executor:
        for file in files:
            for solver in solvers:
                future = executor.submit(run_benchmark, solver, file, timeout)
                results.append(future)

    return [r.result() for r in results]
```

### 8. CI Configuration

**File**: `.github/workflows/benchmarks.yml`

```yaml
name: Benchmark Comparison

on:
  push:
    branches: [main]
  pull_request:

jobs:
  compare-solvers:
    runs-on: ubuntu-latest
    timeout-minutes: 120

    steps:
    - uses: actions/checkout@v4

    - name: Install solvers
      run: |
        # Install Z3
        sudo apt-get install -y z3

        # Build CaDiCaL
        cd reference/cadical && ./configure && make

        # Build Z4
        cargo build --release

    - name: Run SAT benchmarks
      run: |
        python3 scripts/benchmark_runner.py \
          --suite benchmarks/sat/quick/ \
          --solvers z4,cadical \
          --timeout 10 \
          --output results/sat.json

    - name: Run SMT benchmarks
      run: |
        python3 scripts/benchmark_runner.py \
          --suite benchmarks/smt/quick/ \
          --solvers z4,z3 \
          --timeout 10 \
          --output results/smt.json

    - name: Check no regressions
      run: |
        python3 scripts/check_regressions.py \
          --baseline results/baseline.json \
          --current results/sat.json results/smt.json \
          --tolerance 0.1
```

### 9. Proof Verification Pipeline

**File**: `scripts/verify_proofs.py`

```python
#!/usr/bin/env python3
"""
Verify all UNSAT results with proof checker.
"""

import subprocess
from pathlib import Path

def verify_drat(cnf_file: Path, proof_file: Path) -> bool:
    """Verify DRAT proof with drat-trim."""
    result = subprocess.run(
        ['drat-trim', str(cnf_file), str(proof_file)],
        capture_output=True,
        timeout=300
    )
    return b'VERIFIED' in result.stdout

def verify_lrat(cnf_file: Path, proof_file: Path) -> bool:
    """Verify LRAT proof with lrat-check."""
    result = subprocess.run(
        ['lrat-check', str(cnf_file), str(proof_file)],
        capture_output=True,
        timeout=300
    )
    return result.returncode == 0

def run_and_verify(solver: str, cnf_file: Path):
    """Run solver with proof output and verify."""
    proof_file = cnf_file.with_suffix('.proof')

    # Run Z4 with proof output
    subprocess.run([
        solver, str(cnf_file),
        '--proof', str(proof_file)
    ])

    # Verify if UNSAT
    if proof_file.exists():
        if verify_drat(cnf_file, proof_file):
            print(f"VERIFIED: {cnf_file}")
        else:
            print(f"FAILED: {cnf_file}")
            return False
    return True
```

---

## Test Categories

### 10. Quick Tests (CI, <1 min total)

For every PR, run quick sanity tests:

```
benchmarks/quick/
├── sat/          # 20 SAT instances (<1s each)
├── unsat/        # 20 UNSAT instances (<1s each)
├── qf_lia/       # 20 QF_LIA instances
├── qf_bv/        # 20 QF_BV instances
└── qf_uf/        # 20 QF_UF instances
```

### 11. Nightly Tests (Full suite, ~2 hours)

Comprehensive comparison against competitors:

```
benchmarks/nightly/
├── sat/          # 500 SAT instances
├── smt/          # 1000 SMT instances
├── chc/          # 100 CHC instances
└── proofs/       # All UNSAT verified
```

### 12. Weekly Tests (Competition, ~8 hours)

Full competition benchmark suites:

```
benchmarks/weekly/
├── satcomp2023/  # Full SAT-COMP
├── smtcomp2023/  # Full SMT-COMP (main track)
└── chccomp2023/  # Full CHC-COMP
```

---

## Differential Testing

### 13. Fuzzing with DDSmt

**Tool**: https://github.com/ddsmt/ddsmt

```bash
# Generate random SMT formulas and test
for i in $(seq 1 1000); do
  python3 scripts/fuzz_smt.py --logic QF_LIA > /tmp/test.smt2

  Z4_RESULT=$(z4 /tmp/test.smt2)
  Z3_RESULT=$(z3 /tmp/test.smt2)

  if [ "$Z4_RESULT" != "$Z3_RESULT" ]; then
    echo "DISAGREEMENT on /tmp/test.smt2"
    # Delta-debug to minimize
    ddsmt /tmp/test.smt2 --cmd "z4 {} != z3 {}"
  fi
done
```

### 14. Cross-Solver Validation

**File**: `scripts/cross_validate.py`

```python
def cross_validate(file: Path, solvers: list = ['z4', 'z3', 'cvc5']):
    """Check all solvers agree on result."""
    results = {}
    for solver in solvers:
        results[solver] = run_solver(solver, file)

    sat_solvers = [s for s, r in results.items() if r == 'sat']
    unsat_solvers = [s for s, r in results.items() if r == 'unsat']

    if sat_solvers and unsat_solvers:
        print(f"DISAGREEMENT: {file}")
        print(f"  SAT: {sat_solvers}")
        print(f"  UNSAT: {unsat_solvers}")
        return False
    return True
```

---

## Reporting

### 15. Benchmark Dashboard

Generate HTML report with:
- Solve rate comparison (virtual best solver)
- Scatter plots (Z4 vs competitor time)
- Cactus plots (cumulative solved over time)
- Category breakdown

**File**: `scripts/generate_report.py`

```python
def generate_cactus_plot(results: list, output: Path):
    """Generate cactus plot comparing solvers."""
    import matplotlib.pyplot as plt

    for solver in ['z4', 'z3', 'cvc5']:
        times = sorted([r['time'] for r in results
                       if r['solver'] == solver and r['status'] == 'ok'])
        plt.plot(range(len(times)), times, label=solver)

    plt.xlabel('Instances solved')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig(output)
```

---

## Implementation Checklist

### Phase 1: Infrastructure
- [ ] Create `scripts/benchmark_runner.py`
- [ ] Create `scripts/verify_proofs.py`
- [ ] Set up `.github/workflows/benchmarks.yml`
- [ ] Create quick test suite (100 files)

### Phase 2: Import Tests
- [ ] Import CaDiCaL CNF tests (88 files)
- [ ] Import CVC5 regress0 arithmetic (500+ files)
- [ ] Import CVC5 regress0 strings (500+ files)
- [ ] Download SAT-COMP 2023 subset (100 files)

### Phase 3: Full Suites
- [ ] Download full SMT-LIB QF_* benchmarks
- [ ] Download full SAT-COMP benchmarks
- [ ] Download CHC-COMP benchmarks
- [ ] Set up nightly benchmark runs

### Phase 4: Verification
- [ ] Enable DRAT proof verification in CI
- [ ] Add cross-solver validation
- [ ] Set up fuzzing pipeline
- [ ] Create regression tracking

---

## Storage Estimates

| Category | Files | Size | Location |
|----------|-------|------|----------|
| Quick tests | 100 | 10MB | Git |
| Nightly tests | 2,000 | 500MB | Git LFS |
| Weekly tests | 20,000 | 20GB | External |
| SMT-LIB full | 100,000+ | 50GB | External |

---

## Success Criteria

1. **Quick CI**: All 100 quick tests pass in <1 min
2. **Nightly**: Z4 solve rate within 5% of competitors
3. **Weekly**: Track trends, no regressions >10%
4. **Proofs**: 100% of UNSAT verified
5. **Agreement**: 100% agreement with Z3 on deterministic benchmarks
