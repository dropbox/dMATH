#!/usr/bin/env python3
"""Generate SMT-LIB benchmark files for testing Z4 vs Z3."""

import os
import random

BENCHMARK_DIR = "benchmarks/smt"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def generate_qf_bv_benchmarks():
    """Generate QF_BV (quantifier-free bitvector) benchmarks."""
    bv_dir = f"{BENCHMARK_DIR}/QF_BV"
    ensure_dir(bv_dir)

    # 1. Simple satisfiable
    for i in range(20):
        content = f"""; QF_BV benchmark: simple_sat_{i:02d}
(set-logic QF_BV)
(declare-fun x () (_ BitVec 32))
(declare-fun y () (_ BitVec 32))
(assert (= (bvadd x y) #x{random.randint(0, 0xFFFF):08x}))
(assert (bvugt x #x{random.randint(0, 0xFF):08x}))
(check-sat)
(exit)
"""
        with open(f"{bv_dir}/simple_sat_{i:02d}.smt2", "w") as f:
            f.write(content)

    # 2. Harder bitvector puzzles
    for i in range(20):
        width = random.choice([8, 16, 32])
        num_vars = random.randint(2, 4)
        vars = [f"v{j}" for j in range(num_vars)]

        content = f"""; QF_BV benchmark: puzzle_{i:02d}
(set-logic QF_BV)
"""
        for v in vars:
            content += f"(declare-fun {v} () (_ BitVec {width}))\n"

        # Add some constraints
        for _ in range(random.randint(3, 6)):
            v1, v2 = random.sample(vars, 2)
            op = random.choice(["bvadd", "bvand", "bvor", "bvxor"])
            target = random.randint(0, 2**width - 1)
            content += f"(assert (= ({op} {v1} {v2}) #x{target:0{width//4}x}))\n"

        content += "(check-sat)\n(exit)\n"
        with open(f"{bv_dir}/puzzle_{i:02d}.smt2", "w") as f:
            f.write(content)

    # 3. Unsatisfiable cases
    for i in range(10):
        content = f"""; QF_BV benchmark: unsat_{i:02d}
(set-logic QF_BV)
(declare-fun x () (_ BitVec 16))
(assert (= x #x{random.randint(0, 0xFFFF):04x}))
(assert (not (= x #x{random.randint(0, 0xFFFF):04x})))
(assert (= x #x{random.randint(0, 0xFFFF):04x}))
(check-sat)
(exit)
"""
        with open(f"{bv_dir}/unsat_{i:02d}.smt2", "w") as f:
            f.write(content)

def generate_qf_lia_benchmarks():
    """Generate QF_LIA (quantifier-free linear integer arithmetic) benchmarks."""
    lia_dir = f"{BENCHMARK_DIR}/QF_LIA"
    ensure_dir(lia_dir)

    # 1. Simple linear equations
    for i in range(20):
        content = f"""; QF_LIA benchmark: linear_{i:02d}
(set-logic QF_LIA)
(declare-fun x () Int)
(declare-fun y () Int)
(assert (= (+ (* {random.randint(1, 10)} x) (* {random.randint(1, 10)} y)) {random.randint(10, 100)}))
(assert (>= x 0))
(assert (>= y 0))
(assert (<= x {random.randint(20, 50)}))
(check-sat)
(exit)
"""
        with open(f"{lia_dir}/linear_{i:02d}.smt2", "w") as f:
            f.write(content)

    # 2. System of inequalities
    for i in range(20):
        num_vars = random.randint(2, 4)
        vars = [f"v{j}" for j in range(num_vars)]

        content = f"""; QF_LIA benchmark: system_{i:02d}
(set-logic QF_LIA)
"""
        for v in vars:
            content += f"(declare-fun {v} () Int)\n"

        # Add constraints
        for _ in range(random.randint(4, 8)):
            coeffs = [(v, random.randint(-5, 5)) for v in vars]
            terms = " ".join(f"(* {c} {v})" for v, c in coeffs if c != 0)
            if not terms:
                terms = "0"
            rhs = random.randint(-50, 50)
            op = random.choice([">=", "<=", "="])
            content += f"(assert ({op} (+ {terms}) {rhs}))\n"

        content += "(check-sat)\n(exit)\n"
        with open(f"{lia_dir}/system_{i:02d}.smt2", "w") as f:
            f.write(content)

    # 3. Obvious unsat
    for i in range(10):
        content = f"""; QF_LIA benchmark: unsat_{i:02d}
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (> x 10))
(assert (< x 5))
(check-sat)
(exit)
"""
        with open(f"{lia_dir}/unsat_{i:02d}.smt2", "w") as f:
            f.write(content)

def generate_qf_uf_benchmarks():
    """Generate QF_UF (uninterpreted functions) benchmarks."""
    uf_dir = f"{BENCHMARK_DIR}/QF_UF"
    ensure_dir(uf_dir)

    # 1. Simple equality chains
    for i in range(20):
        n = random.randint(3, 6)
        vars = [f"x{j}" for j in range(n)]

        content = f"""; QF_UF benchmark: chain_{i:02d}
(set-logic QF_UF)
(declare-sort U 0)
"""
        for v in vars:
            content += f"(declare-fun {v} () U)\n"

        # Equality chain
        for j in range(n-1):
            content += f"(assert (= {vars[j]} {vars[j+1]}))\n"

        # Inequality at end (sat or unsat)
        if random.random() < 0.3:
            # Make unsat
            content += f"(assert (not (= {vars[0]} {vars[-1]})))\n"

        content += "(check-sat)\n(exit)\n"
        with open(f"{uf_dir}/chain_{i:02d}.smt2", "w") as f:
            f.write(content)

if __name__ == "__main__":
    random.seed(42)  # Reproducible
    print("Generating QF_BV benchmarks...")
    generate_qf_bv_benchmarks()
    print("Generating QF_LIA benchmarks...")
    generate_qf_lia_benchmarks()
    print("Generating QF_UF benchmarks...")
    generate_qf_uf_benchmarks()
    print("Done!")
