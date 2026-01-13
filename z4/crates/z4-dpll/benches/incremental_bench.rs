//! Criterion benchmarks for incremental BV solving
//!
//! Compares performance of:
//! 1. Incremental solving with clause retention (assumption-based)
//! 2. Non-incremental solving with fresh solver per check-sat

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use z4_dpll::Executor;

/// Generate an SMT-LIB script with n push/check-sat/pop cycles
fn generate_incremental_script(num_checks: usize, bv_width: u32) -> String {
    let mut script = String::new();

    // Header
    script.push_str("(set-logic QF_BV)\n");
    script.push_str(&format!("(declare-const x (_ BitVec {}))\n", bv_width));
    script.push_str(&format!("(declare-const y (_ BitVec {}))\n", bv_width));

    // Global constraint
    let max_val = if bv_width <= 16 {
        format!(
            "#x{:0width$x}",
            (1u64 << bv_width) - 1,
            width = (bv_width / 4) as usize
        )
    } else {
        format!("#x{:08x}", 0xFFFFFFFF_u32)
    };
    script.push_str(&format!("(assert (bvult x {}))\n", max_val));
    script.push_str(&format!("(assert (bvult y {}))\n", max_val));

    // Multiple push/check-sat/pop cycles
    for i in 0..num_checks {
        script.push_str("(push 1)\n");
        // Add a constraint that depends on iteration (ensures different problems)
        let val = (i % 256) as u8;
        let hex_val = if bv_width == 8 {
            format!("#x{:02x}", val)
        } else if bv_width == 16 {
            format!("#x{:04x}", val as u16)
        } else {
            format!("#x{:08x}", val as u32)
        };
        script.push_str(&format!("(assert (bvugt (bvadd x y) {}))\n", hex_val));
        script.push_str("(check-sat)\n");
        script.push_str("(pop 1)\n");
    }

    script
}

/// Generate SMT-LIB scripts that simulate fresh solver approach (reset between checks)
/// Returns N separate scripts, each doing one check-sat
fn generate_fresh_solver_scripts(num_checks: usize, bv_width: u32) -> Vec<String> {
    let max_val = if bv_width <= 16 {
        format!(
            "#x{:0width$x}",
            (1u64 << bv_width) - 1,
            width = (bv_width / 4) as usize
        )
    } else {
        format!("#x{:08x}", 0xFFFFFFFF_u32)
    };

    (0..num_checks)
        .map(|i| {
            let mut script = String::new();
            script.push_str("(set-logic QF_BV)\n");
            script.push_str(&format!("(declare-const x (_ BitVec {}))\n", bv_width));
            script.push_str(&format!("(declare-const y (_ BitVec {}))\n", bv_width));
            script.push_str(&format!("(assert (bvult x {}))\n", max_val));
            script.push_str(&format!("(assert (bvult y {}))\n", max_val));
            let val = (i % 256) as u8;
            let hex_val = if bv_width == 8 {
                format!("#x{:02x}", val)
            } else if bv_width == 16 {
                format!("#x{:04x}", val as u16)
            } else {
                format!("#x{:08x}", val as u32)
            };
            script.push_str(&format!("(assert (bvugt (bvadd x y) {}))\n", hex_val));
            script.push_str("(check-sat)\n");
            script
        })
        .collect()
}

/// Generate Kani-like pattern: global memory model + verification conditions
fn generate_kani_pattern_script(num_verif_conditions: usize) -> String {
    let mut script = String::new();

    // Header
    script.push_str("(set-logic QF_BV)\n");

    // Memory model variables (32-bit addresses)
    script.push_str("(declare-const mem_base (_ BitVec 32))\n");
    script.push_str("(declare-const mem_size (_ BitVec 32))\n");
    script.push_str("(declare-const ptr (_ BitVec 32))\n");

    // Global memory model constraints
    script.push_str("(assert (bvuge mem_size #x00001000))\n"); // At least 4KB
    script.push_str("(assert (bvult mem_size #x10000000))\n"); // At most 256MB
    script.push_str("(assert (bvuge ptr mem_base))\n");
    script.push_str("(assert (bvult ptr (bvadd mem_base mem_size)))\n");

    // Verification conditions (each in its own scope)
    for i in 0..num_verif_conditions {
        script.push_str("(push 1)\n");
        // Verification condition: check that pointer arithmetic is in bounds
        let offset = i as u32 * 4; // 4-byte aligned accesses
        script.push_str(&format!(
            "(assert (bvult (bvadd ptr #x{:08x}) (bvadd mem_base mem_size)))\n",
            offset
        ));
        script.push_str("(check-sat)\n");
        script.push_str("(pop 1)\n");
    }

    script
}

fn run_script(script: &str) -> usize {
    let mut executor = Executor::new();
    let commands = z4_frontend::parse(script).expect("Failed to parse script");
    let mut check_count = 0;
    for cmd in commands {
        if let z4_frontend::Command::CheckSat = cmd {
            check_count += 1;
        }
        let _ = executor.execute(&cmd);
    }
    check_count
}

/// Benchmark incremental solving with varying number of check-sat calls
fn bench_incremental_checks(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_bv");

    // Test different numbers of check-sat calls
    for num_checks in [5, 10, 20, 50] {
        let script = generate_incremental_script(num_checks, 8);
        let label = format!("{}checks_8bit", num_checks);

        group.bench_with_input(BenchmarkId::new("incremental", &label), &script, |b, s| {
            b.iter(|| run_script(black_box(s)))
        });
    }

    group.finish();
}

/// Benchmark different bitvector widths
fn bench_bv_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("bv_widths");

    let num_checks = 10;

    for bv_width in [8, 16, 32] {
        let script = generate_incremental_script(num_checks, bv_width);
        let label = format!("{}bit_{}checks", bv_width, num_checks);

        group.bench_with_input(BenchmarkId::new("width", &label), &script, |b, s| {
            b.iter(|| run_script(black_box(s)))
        });
    }

    group.finish();
}

/// Benchmark Kani-like verification pattern
fn bench_kani_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("kani_pattern");

    for num_verif in [5, 10, 20] {
        let script = generate_kani_pattern_script(num_verif);
        let label = format!("{}conditions", num_verif);

        group.bench_with_input(BenchmarkId::new("kani", &label), &script, |b, s| {
            b.iter(|| run_script(black_box(s)))
        });
    }

    group.finish();
}

/// Run multiple scripts with fresh executors (simulates no clause retention)
fn run_scripts_fresh(scripts: &[String]) -> usize {
    let mut count = 0;
    for script in scripts {
        let mut executor = Executor::new();
        let commands = z4_frontend::parse(script).expect("Failed to parse script");
        for cmd in commands {
            if let z4_frontend::Command::CheckSat = cmd {
                count += 1;
            }
            let _ = executor.execute(&cmd);
        }
    }
    count
}

/// Compare incremental (with clause retention) vs fresh solver
fn bench_incremental_vs_fresh(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_vs_fresh");

    // For fair comparison, both approaches do N check-sat calls
    // Incremental: single executor with push/check/pop cycles (clause retention)
    // Fresh: N separate executors, each doing one check-sat (no clause retention)

    for num_checks in [5, 10, 20] {
        // Incremental with push/pop (uses clause retention)
        let incr_script = generate_incremental_script(num_checks, 8);
        let incr_label = format!("{}checks_incremental", num_checks);

        group.bench_with_input(
            BenchmarkId::new("mode", &incr_label),
            &incr_script,
            |b, s| b.iter(|| run_script(black_box(s))),
        );

        // Fresh solver for each check (no clause retention benefit)
        // This simulates the "old" approach where we don't reuse learned clauses
        let fresh_scripts = generate_fresh_solver_scripts(num_checks, 8);
        let fresh_label = format!("{}checks_fresh", num_checks);

        group.bench_with_input(
            BenchmarkId::new("mode", &fresh_label),
            &fresh_scripts,
            |b, scripts| b.iter(|| run_scripts_fresh(black_box(scripts))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_incremental_checks,
    bench_bv_widths,
    bench_kani_pattern,
    bench_incremental_vs_fresh,
);

criterion_main!(benches);
