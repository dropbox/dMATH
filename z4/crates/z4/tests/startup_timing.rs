//! Startup timing benchmarks and regression tests for Kani Fast Priority 2.2
//!
//! Target: Cold start to first check_sat in <10ms (in-process)
//!
//! These tests ensure Z4 maintains fast startup for incremental verification
//! workflows where many solver instances may be created.

use std::time::{Duration, Instant};
use z4_dpll::Executor;
use z4_frontend::parse;

/// Regression test: in-process startup must be under 10ms
/// This is a hard requirement for Kani Fast (Priority 2.2)
#[test]
fn startup_under_10ms() {
    let smt =
        "(set-logic QF_BV)(declare-const x (_ BitVec 32))(assert (bvugt x #x00000000))(check-sat)";

    // Warm up - ensure all code paths are loaded
    {
        let cmds = parse(smt).unwrap();
        let mut exec = Executor::new();
        for cmd in &cmds {
            let _ = exec.execute(cmd);
        }
    }

    // Measure 10 iterations and take the median
    let iterations = 10;
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let commands = parse(smt).unwrap();
        let mut executor = Executor::new();
        for cmd in &commands {
            let _ = executor.execute(cmd);
        }
        times.push(start.elapsed());
    }

    times.sort();
    let median = times[iterations / 2];
    let max_allowed = Duration::from_millis(10);

    assert!(
        median < max_allowed,
        "Startup time regression: median {}us exceeds 10ms target",
        median.as_micros()
    );
}

/// Regression test: startup for SAT logic should be even faster (no theory overhead)
#[test]
fn sat_startup_under_1ms() {
    // Pure SAT via DIMACS-style input
    let smt = "(set-logic SAT)(declare-const p1 Bool)(declare-const p2 Bool)(assert (or p1 p2))(check-sat)";

    // Warm up
    {
        let cmds = parse(smt).unwrap();
        let mut exec = Executor::new();
        for cmd in &cmds {
            let _ = exec.execute(cmd);
        }
    }

    let iterations = 10;
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let commands = parse(smt).unwrap();
        let mut executor = Executor::new();
        for cmd in &commands {
            let _ = executor.execute(cmd);
        }
        times.push(start.elapsed());
    }

    times.sort();
    let median = times[iterations / 2];
    let max_allowed = Duration::from_millis(1);

    assert!(
        median < max_allowed,
        "SAT startup time regression: median {}us exceeds 1ms target",
        median.as_micros()
    );
}

/// Detailed startup timing breakdown (run with --nocapture)
#[test]
#[ignore] // Run with: cargo test -p z4 --release -- --ignored startup_timing_breakdown
fn startup_timing_breakdown() {
    let smt =
        "(set-logic QF_BV)(declare-const x (_ BitVec 32))(assert (bvugt x #x00000000))(check-sat)";

    // Warm up
    {
        let cmds = parse(smt).unwrap();
        let mut exec = Executor::new();
        for cmd in &cmds {
            let _ = exec.execute(cmd);
        }
    }

    let iterations = 10;
    let mut parse_times = Vec::with_capacity(iterations);
    let mut executor_times = Vec::with_capacity(iterations);
    let mut execute_times = Vec::with_capacity(iterations);
    let mut total_times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let total_start = Instant::now();

        let parse_start = Instant::now();
        let commands = parse(smt).unwrap();
        let parse_elapsed = parse_start.elapsed();

        let executor_start = Instant::now();
        let mut executor = Executor::new();
        let executor_elapsed = executor_start.elapsed();

        let execute_start = Instant::now();
        for cmd in &commands {
            let _ = executor.execute(cmd);
        }
        let execute_elapsed = execute_start.elapsed();

        let total_elapsed = total_start.elapsed();

        parse_times.push(parse_elapsed.as_micros());
        executor_times.push(executor_elapsed.as_micros());
        execute_times.push(execute_elapsed.as_micros());
        total_times.push(total_elapsed.as_micros());
    }

    let avg_parse = parse_times.iter().sum::<u128>() as f64 / iterations as f64;
    let avg_executor = executor_times.iter().sum::<u128>() as f64 / iterations as f64;
    let avg_execute = execute_times.iter().sum::<u128>() as f64 / iterations as f64;
    let avg_total = total_times.iter().sum::<u128>() as f64 / iterations as f64;

    println!();
    println!(
        "=== Startup Timing Breakdown (average of {} runs) ===",
        iterations
    );
    println!(
        "Parse:         {:7.1} us ({:5.1}%)",
        avg_parse,
        100.0 * avg_parse / avg_total
    );
    println!(
        "Executor::new: {:7.1} us ({:5.1}%)",
        avg_executor,
        100.0 * avg_executor / avg_total
    );
    println!(
        "Execute:       {:7.1} us ({:5.1}%)",
        avg_execute,
        100.0 * avg_execute / avg_total
    );
    println!("Total:         {:7.1} us", avg_total);
    println!();
    println!("Target: <10,000 us (<10ms) for Kani Fast Priority 2.2");
    println!(
        "Status: {}",
        if avg_total < 10000.0 {
            "PASS"
        } else {
            "FAIL - need optimization"
        }
    );
}

/// Test incremental startup - creating multiple executors in sequence
/// This simulates Kani Fast creating many solver instances
#[test]
fn incremental_startup_under_100ms_for_10_instances() {
    let smt =
        "(set-logic QF_BV)(declare-const x (_ BitVec 32))(assert (bvugt x #x00000000))(check-sat)";

    // Warm up
    {
        let cmds = parse(smt).unwrap();
        let mut exec = Executor::new();
        for cmd in &cmds {
            let _ = exec.execute(cmd);
        }
    }

    let num_instances = 10;
    let start = Instant::now();

    for _ in 0..num_instances {
        let commands = parse(smt).unwrap();
        let mut executor = Executor::new();
        for cmd in &commands {
            let _ = executor.execute(cmd);
        }
    }

    let total = start.elapsed();
    let max_allowed = Duration::from_millis(100);

    assert!(
        total < max_allowed,
        "Creating {} solver instances took {}ms, exceeds 100ms target",
        num_instances,
        total.as_millis()
    );
}
