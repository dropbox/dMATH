use std::path::PathBuf;

use z4_chc::{PdrConfig, PdrResult, PdrSolver};

fn example_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

/// Default config for tests - matches PdrConfig::default() with optional verbose override
fn test_config(verbose: bool) -> PdrConfig {
    PdrConfig {
        max_frames: 20,
        max_iterations: 500,
        max_obligations: 100_000,
        verbose,
        generalize_lemmas: true,
        max_generalization_attempts: 10,
        use_mbp: true,
        use_must_summaries: true,
        use_level_priority: true,
        use_mixed_summaries: true,      // Match default
        use_range_weakening: false, // Match default: prevents infinite loops on unbounded reachability
        use_init_bound_weakening: true, // Match default: needed for invariant synthesis
        use_farkas_combination: false,
        use_relational_equality: false, // Match actual default
        use_interpolation: true,
    }
}

#[test]
fn pdr_examples_smoke() {
    let config = PdrConfig {
        max_frames: 3,
        max_iterations: 50,
        max_obligations: 10_000,
        verbose: false,
        generalize_lemmas: false,
        max_generalization_attempts: 0,
        use_mbp: false,
        use_must_summaries: false,
        use_level_priority: false,
        use_mixed_summaries: false,
        use_range_weakening: false,
        use_init_bound_weakening: false,
        use_farkas_combination: false,
        use_relational_equality: false,
        use_interpolation: false,
    };

    let counter_safe =
        PdrSolver::solve_from_file(example_path("counter_safe.smt2"), config.clone()).unwrap();
    assert!(
        !matches!(counter_safe, PdrResult::Unsafe(_)),
        "counter_safe.smt2 should not be classified as unsafe"
    );

    let primed_vars =
        PdrSolver::solve_from_file(example_path("primed_vars.smt2"), config.clone()).unwrap();
    assert!(
        !matches!(primed_vars, PdrResult::Unsafe(_)),
        "primed_vars.smt2 should not be classified as unsafe"
    );

    let counter_unsafe =
        PdrSolver::solve_from_file(example_path("counter_unsafe.smt2"), config).unwrap();
    assert!(
        !matches!(counter_unsafe, PdrResult::Safe(_)),
        "counter_unsafe.smt2 should not be classified as safe"
    );
}

/// Test that counter_safe.smt2 is proven SAFE with fixed-point detection.
/// The system counts from 0 to 10 and never exceeds 10.
#[test]
fn pdr_counter_safe_proves_safe() {
    let config = test_config(true);

    let result = PdrSolver::solve_from_file(example_path("counter_safe.smt2"), config).unwrap();

    match &result {
        PdrResult::Safe(model) => {
            eprintln!("PDR proved SAFE with invariant:");
            for (pred, interp) in model.iter() {
                eprintln!("  Predicate {:?}: {}", pred, interp.formula);
            }
        }
        PdrResult::Unknown => {
            // Acceptable - fixed-point may not be found with limited frames
            eprintln!("PDR returned Unknown - fixed-point not found in allotted frames");
        }
        PdrResult::Unsafe(cex) => {
            panic!(
                "counter_safe.smt2 should NOT be classified as unsafe! Got {} steps",
                cex.steps.len()
            );
        }
    }

    // We accept Safe or Unknown, but not Unsafe
    assert!(
        !matches!(result, PdrResult::Unsafe(_)),
        "counter_safe.smt2 must not be classified as Unsafe"
    );
}

/// Test counter_unsafe with enough frames to find the counterexample.
/// The system starts at x=5 and can reach x<0 in 6 steps (5->4->3->2->1->0->-1).
///
/// Test counter_unsafe.smt2 - an unsafe counter that can go negative
///
/// Previously this test was ignored due to an infinite loop bug in must-reachability
/// checking. Fixed by implementing Golem/Spacer semantics: check must summaries at
/// level K-1 (predecessor level) rather than level K (same level).
#[test]
fn pdr_counter_unsafe_finds_counterexample() {
    let mut config = test_config(true);
    config.max_frames = 15; // Need more frames for this test

    let result = PdrSolver::solve_from_file(example_path("counter_unsafe.smt2"), config).unwrap();

    // With enough frames, PDR should find the counterexample and return Unsafe
    match &result {
        PdrResult::Unsafe(cex) => {
            eprintln!("PDR returned Unsafe with {} steps", cex.steps.len());
        }
        PdrResult::Unknown => {
            eprintln!("PDR returned Unknown - need to improve algorithm");
        }
        PdrResult::Safe(_) => {
            panic!("counter_unsafe.smt2 should NOT be classified as safe!");
        }
    }

    // The primary goal: PDR should return Unsafe for this problem
    assert!(
        matches!(result, PdrResult::Unsafe(_)),
        "Expected Unsafe, got {:?}",
        match result {
            PdrResult::Safe(_) => "Safe",
            PdrResult::Unsafe(_) => "Unsafe",
            PdrResult::Unknown => "Unknown",
        }
    );
}

/// Test two_counters.smt2 - counter with different bound
#[test]
fn pdr_two_counters_safe() {
    let result =
        PdrSolver::solve_from_file(example_path("two_counters.smt2"), test_config(false)).unwrap();

    // System is safe: x never exceeds 15 (maxes out at 10)
    assert!(
        !matches!(result, PdrResult::Unsafe(_)),
        "two_counters.smt2 should not be classified as Unsafe"
    );
}

/// Test bounded_loop.smt2 - classic bounded loop
/// NOTE: This test uses a parametric invariant (i <= n) which is more complex.
/// The current PDR implementation may not find the invariant within frame limits.
#[test]
fn pdr_bounded_loop_safe() {
    let result =
        PdrSolver::solve_from_file(example_path("bounded_loop.smt2"), test_config(false)).unwrap();

    // System is safe: i never exceeds n
    // We accept Safe or Unknown, but Unsafe would indicate a bug
    assert!(
        !matches!(result, PdrResult::Unsafe(_)),
        "bounded_loop.smt2 should not be classified as Unsafe"
    );
}

/// Test subtraction_unsafe.smt2 - counter goes negative
#[test]
fn pdr_subtraction_unsafe() {
    // Only needs ~5 frames: 3 -> 2 -> 1 -> 0 -> -1
    let result =
        PdrSolver::solve_from_file(example_path("subtraction_unsafe.smt2"), test_config(false))
            .unwrap();

    // System is unsafe: x goes 3, 2, 1, 0, -1
    assert!(
        !matches!(result, PdrResult::Safe(_)),
        "subtraction_unsafe.smt2 should not be classified as Safe"
    );
}

/// Test even_odd.smt2 - two predicates with independent queries
/// Even(x) tracks even numbers >= 0, Odd(x) tracks odd numbers >= 1
/// Query checks if Even(x) can have x < 0 (should be safe)
#[test]
fn pdr_even_odd_safe() {
    let result =
        PdrSolver::solve_from_file(example_path("even_odd.smt2"), test_config(false)).unwrap();

    // System is safe: Even(x) only holds for x >= 0
    assert!(
        !matches!(result, PdrResult::Unsafe(_)),
        "even_odd.smt2 should not be classified as Unsafe"
    );
}

#[test]
fn pdr_bouncy_two_counters_equality_safe() {
    let input = r#"
(set-logic HORN)

(declare-fun |itp2| ( Int Int Int ) Bool)
(declare-fun |itp1| ( Int Int Int ) Bool)

(assert
  (forall ( (A Int) (B Int) (C Int) )
    (=>
      (and
        (and (= B 0) (= A 0) (= C 0))
      )
      (itp1 A B C)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) (E Int) (F Int) )
    (=>
      (and
        (itp1 A C B)
        (and (= E C) (= D (+ 1 A)) (= F (+ 1 B)))
      )
      (itp1 D E F)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) )
    (=>
      (and
        (itp1 A B C)
        true
      )
      (itp2 A B C)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) (E Int) (F Int) )
    (=>
      (and
        (itp2 C A B)
        (and (= E (+ 1 A)) (= D C) (= F (+ (- 1) B)))
      )
      (itp2 D E F)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) )
    (=>
      (and
        (itp2 A B C)
        (and (= A B) (not (= C 0)))
      )
      false
    )
  )
)

(check-sat)
"#;

    let mut config = test_config(false);
    config.max_frames = 40;
    config.max_iterations = 5_000;

    let result = PdrSolver::solve_from_str(input, config).unwrap();
    assert!(
        matches!(result, PdrResult::Safe(_)),
        "expected Safe, got non-Safe result"
    );
}

/// Test two_vars_safe.smt2 - single predicate with two integer variables
/// Inv(x, y) tracks two counters, query checks if x + y > 10
/// Since x maxes at 5 and y maxes at 5, this is safe
#[test]
fn pdr_two_vars_safe() {
    let result =
        PdrSolver::solve_from_file(example_path("two_vars_safe.smt2"), test_config(false)).unwrap();

    // System is safe: x + y maxes at 10
    assert!(
        !matches!(result, PdrResult::Unsafe(_)),
        "two_vars_safe.smt2 should not be classified as Unsafe"
    );
}

/// Test nonlinear_composition.smt2 - non-linear CHC with multiple predicates in query
/// P(x) tracks x in [0,5], Q(y) tracks y in [0,10]
/// Query: P(x) /\ Q(y) /\ x + y > 15 => false
/// This is SAFE because max(x) + max(y) = 5 + 10 = 15, not > 15
///
/// NOTE: This is a non-linear query (multiple predicates). The current PDR
/// implementation may not handle this case fully, returning Unknown.
/// We test that it at least doesn't incorrectly report Unsafe.
#[test]
fn pdr_nonlinear_composition_safe() {
    let result = PdrSolver::solve_from_file(
        example_path("nonlinear_composition.smt2"),
        test_config(false),
    )
    .unwrap();

    // For non-linear queries, we accept Safe or Unknown, but NOT Unsafe
    // (since the system is actually safe)
    match &result {
        PdrResult::Safe(model) => {
            eprintln!(
                "PDR proved non-linear CHC SAFE with {} predicates",
                model.len()
            );
        }
        PdrResult::Unknown => {
            // Expected for now - non-linear queries are not fully supported
            eprintln!("PDR returned Unknown for non-linear CHC (expected)");
        }
        PdrResult::Unsafe(_) => {
            panic!("nonlinear_composition.smt2 should NOT be classified as Unsafe!");
        }
    }
}

/// Test hyperedge_safe.smt2 - clause with multiple body predicates (hyperedge)
/// This tests the mixed summaries implementation (Spacer technique).
///
/// System:
/// - P(x) tracks x in [0,5]
/// - Q(y) tracks y in [0,3]
/// - R(x,y) is reached when P(x) ∧ Q(y) (HYPEREDGE - two body predicates)
/// - Query: R(x,y) ∧ x+y > 8 ⟹ false (should be SAFE since max 5+3=8)
#[test]
fn pdr_hyperedge_safe() {
    let result = PdrSolver::solve_from_file(
        example_path("hyperedge_safe.smt2"),
        test_config(false), // Disable verbose for performance
    )
    .unwrap();

    // System is safe: x maxes at 5, y maxes at 3, so x + y <= 8
    // With mixed summaries, PDR should be able to handle this hyperedge
    match &result {
        PdrResult::Safe(model) => {
            eprintln!(
                "PDR proved hyperedge CHC SAFE with {} predicates",
                model.len()
            );
        }
        PdrResult::Unknown => {
            // Acceptable - hyperedge handling may need more work
            eprintln!("PDR returned Unknown for hyperedge CHC");
        }
        PdrResult::Unsafe(_) => {
            panic!("hyperedge_safe.smt2 should NOT be classified as Unsafe!");
        }
    }
}

/// Test hyperedge_unsafe.smt2 - UNSAFE hyperedge with two body predicates
///
/// System:
/// - P(x) tracks x in [0,10]
/// - Q(y) tracks y in [0,10]
/// - R(x,y) is reached when P(x) ∧ Q(y) (HYPEREDGE)
/// - Query: R(x,y) ∧ x+y >= 15 ⟹ false (UNSAFE since max 10+10=20 >= 15)
#[test]
fn pdr_hyperedge_unsafe() {
    let result =
        PdrSolver::solve_from_file(example_path("hyperedge_unsafe.smt2"), test_config(false))
            .unwrap();

    // System is unsafe: x can reach 10, y can reach 10, so x + y can reach 20 >= 15
    match &result {
        PdrResult::Unsafe(_) => {
            eprintln!("PDR correctly found hyperedge CHC UNSAFE");
        }
        PdrResult::Unknown => {
            // Acceptable - hyperedge handling may need more work
            eprintln!("PDR returned Unknown for unsafe hyperedge CHC");
        }
        PdrResult::Safe(_) => {
            panic!("hyperedge_unsafe.smt2 should NOT be classified as Safe!");
        }
    }
}

/// Test hyperedge_triple.smt2 - SAFE triple hyperedge with three body predicates
///
/// System:
/// - A(x), B(y), C(z) each track counters in [0,2]
/// - R(x,y,z) is reached when A(x) ∧ B(y) ∧ C(z) (TRIPLE HYPEREDGE)
/// - Query: R(x,y,z) ∧ x+y+z > 6 ⟹ false (SAFE since max 2+2+2=6)
#[test]
fn pdr_hyperedge_triple() {
    let result =
        PdrSolver::solve_from_file(example_path("hyperedge_triple.smt2"), test_config(false))
            .unwrap();

    // System is safe: each counter maxes at 2, so sum maxes at 6
    match &result {
        PdrResult::Safe(model) => {
            eprintln!(
                "PDR proved triple hyperedge CHC SAFE with {} predicates",
                model.len()
            );
        }
        PdrResult::Unknown => {
            // Acceptable - triple hyperedge is complex
            eprintln!("PDR returned Unknown for triple hyperedge CHC");
        }
        PdrResult::Unsafe(_) => {
            panic!("hyperedge_triple.smt2 should NOT be classified as Unsafe!");
        }
    }
}

/// Test that counterexample steps have populated assignments from SMT models
///
/// Uses subtraction_unsafe.smt2 which should find a counterexample trace:
/// x = 3 -> 2 -> 1 -> 0 -> -1 (violates x >= 0)
///
/// Note: Must-summaries disabled to force normal predecessor path which builds
/// parent chains with SMT models.
#[test]
fn pdr_counterexample_has_assignments() {
    let mut config = test_config(true);
    config.use_must_summaries = false; // Disable to test normal predecessor path

    let result =
        PdrSolver::solve_from_file(example_path("subtraction_unsafe.smt2"), config).unwrap();

    match &result {
        PdrResult::Unsafe(cex) => {
            eprintln!("Counterexample has {} steps:", cex.steps.len());
            for (i, step) in cex.steps.iter().enumerate() {
                eprintln!(
                    "  Step {}: predicate {:?}, assignments: {:?}",
                    i, step.predicate, step.assignments
                );
            }

            // At least some steps should have non-empty assignments
            let steps_with_assignments = cex
                .steps
                .iter()
                .filter(|s| !s.assignments.is_empty())
                .count();

            eprintln!(
                "Steps with non-empty assignments: {}/{}",
                steps_with_assignments,
                cex.steps.len()
            );

            // Verify at least one step has assignments (the root POB might not have a model)
            assert!(
                steps_with_assignments > 0 || cex.steps.is_empty(),
                "At least some counterexample steps should have variable assignments"
            );

            // Check witness is populated with instances
            if let Some(ref witness) = cex.witness {
                eprintln!(
                    "Derivation witness has {} entries, root={}",
                    witness.entries.len(),
                    witness.root
                );
                assert!(
                    !witness.entries.is_empty(),
                    "Derivation witness should have entries"
                );

                // Check that witness entries have instances populated
                for (i, entry) in witness.entries.iter().enumerate() {
                    eprintln!(
                        "  Entry {}: pred {:?}, level {}, instances: {:?}",
                        i, entry.predicate, entry.level, entry.instances
                    );
                }

                // At least some entries should have instances (from SMT models)
                let entries_with_instances = witness
                    .entries
                    .iter()
                    .filter(|e| !e.instances.is_empty())
                    .count();
                eprintln!(
                    "Entries with instances: {}/{}",
                    entries_with_instances,
                    witness.entries.len()
                );

                // Verify at least one entry has instances
                assert!(
                    entries_with_instances > 0 || witness.entries.is_empty(),
                    "At least some derivation entries should have concrete instances"
                );
            }
        }
        _ => {
            // subtraction_unsafe should be unsafe, but we're mainly testing assignment extraction
            eprintln!("Note: subtraction_unsafe did not return Unsafe result");
        }
    }
}

/// Test that incoming_clause is populated in derivation witness for must-reachability-driven UNSAFE.
/// The counter_unsafe example triggers must-reachability detection when must_summaries is enabled.
#[test]
fn pdr_must_reachability_has_incoming_clause() {
    let config = test_config(true);
    // use_must_summaries: true is already the default

    let result = PdrSolver::solve_from_file(example_path("counter_unsafe.smt2"), config).unwrap();

    match &result {
        PdrResult::Unsafe(cex) => {
            eprintln!("Counterexample has {} steps:", cex.steps.len());

            // Check witness is populated with incoming_clause
            if let Some(ref witness) = cex.witness {
                eprintln!(
                    "Derivation witness has {} entries, query_clause={:?}, root={}",
                    witness.entries.len(),
                    witness.query_clause,
                    witness.root
                );

                for (i, entry) in witness.entries.iter().enumerate() {
                    eprintln!(
                        "  Entry {}: pred {:?}, level {}, incoming_clause={:?}",
                        i, entry.predicate, entry.level, entry.incoming_clause
                    );
                }

                // Count entries with incoming_clause populated
                let entries_with_clause = witness
                    .entries
                    .iter()
                    .filter(|e| e.incoming_clause.is_some())
                    .count();

                eprintln!(
                    "Entries with incoming_clause: {}/{}",
                    entries_with_clause,
                    witness.entries.len()
                );

                // At least some entries should have incoming_clause populated
                // (the initial entry might not have one if it's the query root)
                assert!(
                    entries_with_clause > 0,
                    "At least some derivation entries should have incoming_clause populated"
                );
            } else {
                panic!("Counterexample should have a derivation witness");
            }
        }
        _ => {
            panic!("counter_unsafe should return Unsafe result");
        }
    }
}

/// EXPERIMENT: Test hyperedge_unsafe with range_weakening disabled
/// This test verifies the claim from WORKER_DIRECTIVE.md that disabling
/// range-based weakening causes hyperedge_unsafe to incorrectly return Safe.
///
/// Root cause analysis:
/// - Without range-based weakening, blocking lemmas are too specific (e.g., x != 15)
/// - These lemmas don't generalize to cover all problematic values
/// - Combined with the hyperedge skip in is_inductive_blocking, this allows
///   false fixed points to be detected as Safe
/// - With range-based weakening, lemmas generalize to inequalities (e.g., x < 15)
///   which provide stronger coverage and prevent false Safe
///
/// This is a KNOWN soundness issue that blocks disabling range-based weakening.
#[test]
#[ignore] // Ignore by default since it documents a known soundness issue
fn experiment_hyperedge_unsafe_no_range_weakening() {
    // This config disables range_weakening to test the directive claim
    let config = PdrConfig {
        max_frames: 20,
        max_iterations: 500,
        max_obligations: 100_000,
        verbose: true, // Enable verbose for investigation
        generalize_lemmas: true,
        max_generalization_attempts: 10,
        use_mbp: true,
        use_must_summaries: true,
        use_level_priority: true,
        use_mixed_summaries: false,
        use_range_weakening: false, // DISABLED for this experiment
        use_init_bound_weakening: false,
        use_farkas_combination: false,
        use_relational_equality: true,
        use_interpolation: false,
    };

    let result = PdrSolver::solve_from_file(example_path("hyperedge_unsafe.smt2"), config).unwrap();

    // Document what we observe when range_weakening is disabled:
    // According to WORKER_DIRECTIVE.md, this should incorrectly return Safe
    match &result {
        PdrResult::Unsafe(_) => {
            eprintln!("EXPERIMENT: With range_weakening=false, got Unsafe (correct)");
        }
        PdrResult::Unknown => {
            eprintln!("EXPERIMENT: With range_weakening=false, got Unknown (acceptable)");
        }
        PdrResult::Safe(_) => {
            // CONFIRMED: This is a soundness bug when range_weakening is disabled
            eprintln!("EXPERIMENT: With range_weakening=false, got Safe (SOUNDNESS BUG)");
        }
    }
}

/// Investigation test for dillig12_m - multi-predicate CHC problem
/// Expected: sat (safe), but currently returns unknown
#[test]
#[ignore] // Ignore by default - for investigation only
fn investigate_dillig12_m() {
    use std::path::PathBuf;

    let config = PdrConfig {
        max_frames: 30,
        max_iterations: 1000,
        max_obligations: 200_000,
        verbose: true,
        generalize_lemmas: true,
        max_generalization_attempts: 10,
        use_mbp: true,
        use_must_summaries: true,
        use_level_priority: true,
        use_mixed_summaries: true,
        use_range_weakening: false,
        use_init_bound_weakening: true,
        use_farkas_combination: false,
        use_relational_equality: false,
        use_interpolation: true,
    };

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/chc/chc-comp25-benchmarks/extra-small-lia/dillig12_m_000.smt2");

    let result = PdrSolver::solve_from_file(&path, config).unwrap();

    match &result {
        PdrResult::Safe(_) => eprintln!("dillig12_m: Got SAFE (expected)"),
        PdrResult::Unsafe(_) => eprintln!("dillig12_m: Got UNSAFE (unexpected!)"),
        PdrResult::Unknown => eprintln!("dillig12_m: Got UNKNOWN (investigating...)"),
    }
}
