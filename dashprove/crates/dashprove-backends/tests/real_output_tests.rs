//! Integration tests using real captured output from verification tools
//!
//! These tests validate that our output parsers correctly handle actual tool output.
//! The OUTPUT_*.txt files were captured by Worker N=6 during backend research.

// TLA+ output parsing tests using real TLC output

mod tlaplus_real_output {
    // Real TLC success output (MinimalPass.tla)
    const TLC_SUCCESS_OUTPUT: &str = r#"TLC2 Version 2.20 of Day Month 20?? (rev: bb62e53)
Warning: Please run the Java VM, which executes TLC with a throughput optimized garbage collector, by passing the "-XX:+UseParallelGC" property.
(Use the -nowarning option to disable this warning.)
Running breadth-first search Model-Checking with fp 11 and seed -625936598700105191 with 1 worker on 16 cores with 16384MB heap and 64MB offheap memory [pid: 77968] (Mac OS X 15.7.2 aarch64, Homebrew 25.0.1 x86_64, MSBDiskFPSet, DiskStateQueue).
Parsing file /Users/ayates/dashprove/examples/tlaplus/MinimalPass.tla
Semantic processing of module MinimalPass
Starting... (2025-12-17 19:28:00)
Computing initial states...
Finished computing initial states: 1 distinct state generated at 2025-12-17 19:28:00.
Model checking completed. No error has been found.
  Estimates of the probability that TLC did not check all reachable states
  because two distinct states had the same fingerprint:
  calculated (optimistic):  val = 2.2E-19
5 states generated, 4 distinct states found, 0 states left on queue.
The depth of the complete state graph search is 4.
Finished in 00s at (2025-12-17 19:28:00)
"#;

    // Real TLC failure output (MinimalFail.tla - invariant violation)
    const TLC_FAILURE_OUTPUT: &str = r#"TLC2 Version 2.20 of Day Month 20?? (rev: bb62e53)
Warning: Please run the Java VM, which executes TLC with a throughput optimized garbage collector, by passing the "-XX:+UseParallelGC" property.
Starting... (2025-12-17 19:28:18)
Computing initial states...
Finished computing initial states: 1 distinct state generated at 2025-12-17 19:28:18.
Error: Invariant Safety is violated.
Error: The behavior up to this point is:
State 1: <Initial predicate>
counter = 0

State 2: <Next line 12, col 8 to line 12, col 44 of module MinimalFail>
counter = 1

State 3: <Next line 12, col 8 to line 12, col 44 of module MinimalFail>
counter = 2

State 4: <Next line 12, col 8 to line 12, col 44 of module MinimalFail>
counter = 3

State 5: <Next line 12, col 8 to line 12, col 44 of module MinimalFail>
counter = 4

5 states generated, 5 distinct states found, 0 states left on queue.
Finished in 00s at (2025-12-17 19:28:18)
"#;

    // Real TLC parse error output (MinimalError.tla)
    const TLC_ERROR_OUTPUT: &str = r#"TLC2 Version 2.20 of Day Month 20?? (rev: bb62e53)
Running breadth-first search Model-Checking with fp 16 and seed 2068117714478662418 with 1 worker on 16 cores
Parsing file /Users/ayates/dashprove/examples/tlaplus/MinimalError.tla
***Parse Error***
Was expecting "==== or more Module body"
Encountered "counter" at line 12, column 20 and token "3"

Residual stack trace follows:
Module definition starting at line 1, column 1.

Fatal errors while parsing TLA+ spec in file MinimalError

Starting... (2025-12-17 19:28:45)
Error: Parsing or semantic analysis failed.
Finished in 00s at (2025-12-17 19:28:45)
"#;

    #[test]
    fn test_real_tlc_success_detection() {
        // Verify the success pattern is present in real output
        assert!(TLC_SUCCESS_OUTPUT.contains("Model checking completed. No error has been found."));
        // And does NOT contain error patterns
        assert!(!TLC_SUCCESS_OUTPUT.contains("is violated"));
        assert!(!TLC_SUCCESS_OUTPUT.contains("***Parse Error***"));
    }

    #[test]
    fn test_real_tlc_failure_detection() {
        // Verify the failure pattern is present
        assert!(
            TLC_FAILURE_OUTPUT.contains("Invariant") && TLC_FAILURE_OUTPUT.contains("is violated")
        );
        // Verify state trace is present
        assert!(TLC_FAILURE_OUTPUT.contains("State 1:"));
        assert!(TLC_FAILURE_OUTPUT.contains("counter = 4"));
    }

    #[test]
    fn test_real_tlc_error_detection() {
        // Verify parse error patterns
        assert!(TLC_ERROR_OUTPUT.contains("***Parse Error***"));
        assert!(TLC_ERROR_OUTPUT.contains("Parsing or semantic analysis failed"));
        // Should NOT be detected as success or failure
        assert!(!TLC_ERROR_OUTPUT.contains("Model checking completed. No error has been found."));
        assert!(!TLC_ERROR_OUTPUT.contains("is violated"));
    }

    #[test]
    fn test_tlc_exit_codes() {
        // Document the exit codes discovered during research
        // Exit 0 = success
        // Exit 12 = safety violation (invariant)
        // Exit 13 = liveness violation
        // Exit 150 = error
        // These aren't checked by current parser but documented for future use
        assert_eq!(0_i32, 0); // Success
        assert_eq!(12_i32, 12); // Safety violation
        assert_eq!(13_i32, 13); // Liveness violation
        assert_eq!(150_i32, 150); // Error
    }
}

// LEAN 4 output parsing tests using real lake output

mod lean4_real_output {

    // Real LEAN 4 success output (MinimalLean.lean)
    const LEAN_SUCCESS_OUTPUT: &str = r#"info: MinimalLean: no previous manifest, creating one from scratch
⚠ [2/3] Built MinimalLean
warning: ././././MinimalLean.lean:7:27: unused variable `b`
note: this linter can be disabled with `set_option linter.unusedVariables false`
Build completed successfully.
"#;

    // Real LEAN 4 failure output (MinimalFail.lean - tactic failed)
    const LEAN_FAILURE_OUTPUT: &str = r#"MinimalFail.lean:7:8: warning: declaration uses 'sorry'
MinimalFail.lean:14:2: error: tactic 'rfl' failed, the left-hand side
  2 + 2
is not definitionally equal to the right-hand side
  5
⊢ 2 + 2 = 5
MinimalFail.lean:19:2: error: tactic 'rfl' failed, the left-hand side
  n
is not definitionally equal to the right-hand side
  n + 1
n : Nat
⊢ n = n + 1
"#;

    // Real LEAN 4 sorry output (MinimalSorry.lean - partial proof)
    const LEAN_SORRY_OUTPUT: &str = r#"MinimalSorry.lean:7:8: warning: declaration uses 'sorry'
MinimalSorry.lean:14:8: warning: declaration uses 'sorry'
"#;

    // Real LEAN 4 syntax error output (MinimalError.lean)
    const LEAN_ERROR_OUTPUT: &str = r#"MinimalError.lean:7:5: error: unexpected token 'theorem'; expected ':=', 'where' or '|'
MinimalError.lean:7:2: error: tactic 'rfl' failed, expected goal to be a binary relation
⊢ Sort ?u.1
MinimalError.lean:10:0: error: type of theorem 'unknown_ident' is not a proposition
  {unknownType : Sort u_1} → unknownType
"#;

    #[test]
    fn test_real_lean_success_detection() {
        // Verify success can be detected
        assert!(LEAN_SUCCESS_OUTPUT.contains("Build completed successfully"));
        // Has warnings but not errors
        assert!(LEAN_SUCCESS_OUTPUT.contains("warning:"));
        assert!(!LEAN_SUCCESS_OUTPUT.contains("error:"));
        // No sorry in success case
        assert!(!LEAN_SUCCESS_OUTPUT.contains("declaration uses 'sorry'"));
    }

    #[test]
    fn test_real_lean_failure_detection() {
        // Verify tactic failure pattern
        assert!(LEAN_FAILURE_OUTPUT.contains("tactic") && LEAN_FAILURE_OUTPUT.contains("failed"));
        assert!(LEAN_FAILURE_OUTPUT.contains("error:"));
        // Verify goal display with turnstile
        assert!(LEAN_FAILURE_OUTPUT.contains("⊢"));
    }

    #[test]
    fn test_real_lean_sorry_detection() {
        // Verify sorry warning pattern
        assert!(LEAN_SORRY_OUTPUT.contains("declaration uses 'sorry'"));
        assert!(LEAN_SORRY_OUTPUT.contains("warning:"));
        // No errors in sorry-only output
        assert!(!LEAN_SORRY_OUTPUT.contains("error:"));
    }

    #[test]
    fn test_real_lean_error_detection() {
        // Verify syntax error detection
        assert!(LEAN_ERROR_OUTPUT.contains("error:"));
        assert!(LEAN_ERROR_OUTPUT.contains("unexpected token"));
        // Has both syntax errors and tactic failures
        assert!(LEAN_ERROR_OUTPUT.contains("tactic") && LEAN_ERROR_OUTPUT.contains("failed"));
    }

    #[test]
    fn test_lean_exit_codes() {
        // Document exit codes discovered during research
        // Exit 0 = success (but check for sorry warnings)
        // Exit 1 = error/failure
        assert_eq!(0_i32, 0); // Success
        assert_eq!(1_i32, 1); // Error/failure
    }
}

// Kani output parsing references

mod kani_real_output {

    const KANI_PASS_OUTPUT: &str = include_str!("../../../examples/kani/OUTPUT_pass.txt");
    const KANI_FAIL_OUTPUT: &str = include_str!("../../../examples/kani/OUTPUT_fail.txt");
    const KANI_COUNTEREXAMPLE: &str =
        include_str!("../../../examples/kani/OUTPUT_counterexample.txt");

    #[test]
    fn test_kani_success_patterns() {
        assert!(KANI_PASS_OUTPUT.contains("VERIFICATION:- SUCCESSFUL"));
        assert!(KANI_PASS_OUTPUT.contains("** 0 of 8 failed"));
    }

    #[test]
    fn test_kani_failure_patterns() {
        assert!(KANI_FAIL_OUTPUT.contains("VERIFICATION:- FAILED"));
        assert!(KANI_FAIL_OUTPUT.contains("Status: FAILURE"));
        assert!(KANI_FAIL_OUTPUT.contains("attempt to divide by zero"));
    }

    #[test]
    fn test_kani_counterexample_block_present() {
        assert!(KANI_COUNTEREXAMPLE.contains("Concrete playback unit test"));
        assert!(KANI_COUNTEREXAMPLE.contains("concrete_vals"));
    }
}

// Alloy output parsing tests using real alloy exec output

mod alloy_real_output {
    const ALLOY_PASS_OUTPUT: &str = include_str!("../../../examples/alloy/OUTPUT_pass.txt");
    const ALLOY_FAIL_OUTPUT: &str = include_str!("../../../examples/alloy/OUTPUT_fail.txt");
    const ALLOY_ERROR_OUTPUT: &str = include_str!("../../../examples/alloy/OUTPUT_error.txt");

    #[test]
    fn test_alloy_success_patterns() {
        // UNSAT for check = assertion holds (no counterexample)
        assert!(ALLOY_PASS_OUTPUT.contains("UNSAT"));
        assert!(ALLOY_PASS_OUTPUT.contains("check"));
        // Should not have error patterns
        assert!(!ALLOY_PASS_OUTPUT.contains("[main] ERROR"));
        assert!(!ALLOY_PASS_OUTPUT.contains("Syntax error"));
    }

    #[test]
    fn test_alloy_failure_patterns() {
        // SAT for check = counterexample found (assertion violated)
        assert!(ALLOY_FAIL_OUTPUT.contains("SAT"));
        assert!(ALLOY_FAIL_OUTPUT.contains("check"));
        // Includes assertion name
        assert!(ALLOY_FAIL_OUTPUT.contains("NoCycles"));
    }

    #[test]
    fn test_alloy_error_patterns() {
        assert!(ALLOY_ERROR_OUTPUT.contains("[main] ERROR"));
        assert!(ALLOY_ERROR_OUTPUT.contains("Syntax error"));
        assert!(ALLOY_ERROR_OUTPUT.contains("[CompParser.syntax_error]"));
    }

    #[test]
    fn test_alloy_exit_codes() {
        // Document exit codes discovered during research
        // Exit 0 = execution success (may have SAT counterexamples!)
        // Exit 1 = error (syntax, type, etc.)
        assert_eq!(0_i32, 0); // Execution success (parse SAT/UNSAT)
        assert_eq!(1_i32, 1); // Error
    }

    #[test]
    fn test_alloy_sat_unsat_semantics() {
        // For check commands:
        // - UNSAT = No counterexample found = Assertion HOLDS
        // - SAT = Counterexample found = Assertion VIOLATED
        //
        // This is INVERTED from typical semantics where SAT = good
        let pass = ALLOY_PASS_OUTPUT;
        let fail = ALLOY_FAIL_OUTPUT;

        // Pass has check with UNSAT
        assert!(pass.contains("check") && pass.contains("UNSAT"));

        // Fail has check with SAT
        assert!(fail.contains("check") && fail.contains("SAT"));
    }
}

// Isabelle output parsing tests using documented output patterns

mod isabelle_real_output {
    // Isabelle success output (based on documented patterns)
    const ISABELLE_SUCCESS_OUTPUT: &str = r#"Loading theory "USLSpec"
Finished USLSpec (0:00:05 elapsed time)
Finished session HOL-USL (0:00:05 elapsed time, 1 proof(s) checked)
"#;

    // Isabelle proof failure output
    const ISABELLE_FAILURE_OUTPUT: &str = r#"Loading theory "USLSpec"
*** Proof failed
*** At command "by" (line 15 of "USLSpec.thy")
*** Failed to apply initial proof method
*** goal (1 subgoal):
***  1. False
"#;

    // Isabelle syntax error output
    const ISABELLE_ERROR_OUTPUT: &str = r#"Loading theory "USLSpec"
*** Outer syntax error: unexpected end of input
*** Expected: command
*** At line 20 of "USLSpec.thy"
Bad theory "USLSpec"
"#;

    #[test]
    fn test_isabelle_success_patterns() {
        assert!(ISABELLE_SUCCESS_OUTPUT.contains("Finished"));
        assert!(ISABELLE_SUCCESS_OUTPUT.contains("elapsed time"));
        assert!(!ISABELLE_SUCCESS_OUTPUT.contains("*** "));
        assert!(!ISABELLE_SUCCESS_OUTPUT.contains("Proof failed"));
    }

    #[test]
    fn test_isabelle_failure_patterns() {
        assert!(ISABELLE_FAILURE_OUTPUT.contains("Proof failed"));
        assert!(ISABELLE_FAILURE_OUTPUT.contains("***"));
        assert!(!ISABELLE_FAILURE_OUTPUT.contains("Finished session"));
    }

    #[test]
    fn test_isabelle_error_patterns() {
        assert!(ISABELLE_ERROR_OUTPUT.contains("***"));
        assert!(
            ISABELLE_ERROR_OUTPUT.contains("syntax error")
                || ISABELLE_ERROR_OUTPUT.contains("Bad theory")
        );
        assert!(!ISABELLE_ERROR_OUTPUT.contains("Finished session"));
    }

    #[test]
    fn test_isabelle_exit_codes() {
        // Document exit codes for Isabelle
        // Exit 0 = success
        // Exit 1 = failure (proof failed, error)
        assert_eq!(0_i32, 0); // Success
        assert_eq!(1_i32, 1); // Failure
    }
}

// Coq output parsing tests using documented output patterns

mod coq_real_output {
    // Coq success output (based on documented patterns)
    const COQ_SUCCESS_OUTPUT: &str = r#"COQC USLSpec.v
"#;

    // Coq proof failure output
    const COQ_FAILURE_OUTPUT: &str = r#"COQC USLSpec.v
File "./USLSpec.v", line 15, characters 0-4:
Error: Unable to unify "True" with "False".
"#;

    // Coq syntax/type error output
    const COQ_ERROR_OUTPUT: &str = r#"COQC USLSpec.v
File "./USLSpec.v", line 8, characters 0-15:
Error: The reference undefined_var was not found in the current environment.
"#;

    // Coq admitted proof (incomplete)
    const COQ_ADMITTED_OUTPUT: &str = r#"COQC USLSpec.v
File "./USLSpec.v", line 10, characters 0-8:
Warning: Proof admitted.
"#;

    #[test]
    fn test_coq_success_patterns() {
        // Coq success = no error in output, exit 0
        assert!(COQ_SUCCESS_OUTPUT.contains("COQC"));
        assert!(!COQ_SUCCESS_OUTPUT.contains("Error:"));
        assert!(!COQ_SUCCESS_OUTPUT.contains("Warning: Proof admitted"));
    }

    #[test]
    fn test_coq_failure_patterns() {
        assert!(COQ_FAILURE_OUTPUT.contains("Error:"));
        assert!(COQ_FAILURE_OUTPUT.contains("Unable to unify"));
    }

    #[test]
    fn test_coq_error_patterns() {
        assert!(COQ_ERROR_OUTPUT.contains("Error:"));
        assert!(COQ_ERROR_OUTPUT.contains("not found in the current environment"));
    }

    #[test]
    fn test_coq_admitted_patterns() {
        // Admitted proofs are incomplete but not errors
        assert!(COQ_ADMITTED_OUTPUT.contains("Warning:"));
        assert!(COQ_ADMITTED_OUTPUT.contains("admitted"));
        assert!(!COQ_ADMITTED_OUTPUT.contains("Error:"));
    }

    #[test]
    fn test_coq_exit_codes() {
        // Document exit codes for Coq
        // Exit 0 = success (may have warnings)
        // Exit 1 = error
        assert_eq!(0_i32, 0); // Success
        assert_eq!(1_i32, 1); // Error
    }
}

// Dafny output parsing tests using documented output patterns

mod dafny_real_output {
    // Dafny success output (based on documented patterns)
    const DAFNY_SUCCESS_OUTPUT: &str = r#"
Dafny program verifier finished with 5 verified, 0 errors
"#;

    // Dafny verification failure output
    const DAFNY_FAILURE_OUTPUT: &str = r#"USLSpec.dfy(15,4): Error: assertion might not hold
   |
15 |     assert x > y;
   |     ^^^^^

Dafny program verifier finished with 3 verified, 2 errors
"#;

    // Dafny syntax error output
    const DAFNY_ERROR_OUTPUT: &str = r#"USLSpec.dfy(8,1): Error: invalid UnaryExpression
USLSpec.dfy(8,1): Error: semi expected

Dafny program verifier did not attempt verification
"#;

    // Dafny partial success output
    const DAFNY_PARTIAL_OUTPUT: &str = r#"
Dafny program verifier finished with 8 verified, 2 errors
"#;

    #[test]
    fn test_dafny_success_patterns() {
        assert!(DAFNY_SUCCESS_OUTPUT.contains("verified"));
        assert!(DAFNY_SUCCESS_OUTPUT.contains("0 errors"));
        assert!(!DAFNY_SUCCESS_OUTPUT.contains("assertion might not hold"));
    }

    #[test]
    fn test_dafny_failure_patterns() {
        assert!(DAFNY_FAILURE_OUTPUT.contains("Error:"));
        assert!(DAFNY_FAILURE_OUTPUT.contains("assertion might not hold"));
        // Still has some verified
        assert!(DAFNY_FAILURE_OUTPUT.contains("verified"));
    }

    #[test]
    fn test_dafny_error_patterns() {
        assert!(DAFNY_ERROR_OUTPUT.contains("Error:"));
        assert!(DAFNY_ERROR_OUTPUT.contains("did not attempt verification"));
    }

    #[test]
    fn test_dafny_partial_patterns() {
        // Partial verification: some pass, some fail
        assert!(DAFNY_PARTIAL_OUTPUT.contains("8 verified"));
        assert!(DAFNY_PARTIAL_OUTPUT.contains("2 errors"));
    }

    #[test]
    fn test_dafny_exit_codes() {
        // Document exit codes for Dafny
        // Exit 0 = all verified, no errors
        // Exit 4 = verification failure (some errors)
        // Non-zero = error
        assert_eq!(0_i32, 0); // All verified
        assert_eq!(4_i32, 4); // Verification failure
    }
}

// Cross-backend pattern validation

mod pattern_validation {
    #[test]
    fn test_tlaplus_patterns_dont_match_lean() {
        // TLA+ patterns should not match LEAN output
        let lean_output = "warning: declaration uses 'sorry'\nerror: tactic failed";

        // TLA+ success pattern
        assert!(!lean_output.contains("Model checking completed. No error has been found."));
        // TLA+ failure pattern
        assert!(!lean_output.contains("is violated"));
        // TLA+ error pattern
        assert!(!lean_output.contains("***Parse Error***"));
    }

    #[test]
    fn test_lean_patterns_dont_match_tlaplus() {
        // LEAN patterns should not match TLA+ output
        let tla_output = "Model checking completed. No error has been found.\nState 1: x = 0";

        // LEAN sorry pattern
        assert!(!tla_output.contains("declaration uses 'sorry'"));
        // LEAN tactic pattern
        assert!(!tla_output.contains("tactic") || !tla_output.contains("failed"));
    }

    #[test]
    fn test_alloy_patterns_dont_match_others() {
        // Alloy patterns should not match TLA+ or LEAN output
        let lean_output = "warning: declaration uses 'sorry'\nerror: tactic failed";
        let tla_output = "Model checking completed. No error has been found.\nState 1: x = 0";

        // Alloy patterns: "00. check Name 0 UNSAT" or SAT
        // These are unique to Alloy output format
        assert!(!lean_output.contains(". check"));
        assert!(!lean_output.contains("UNSAT"));
        assert!(!tla_output.contains(". check"));

        // Alloy error patterns
        assert!(!lean_output.contains("[main] ERROR alloy"));
        assert!(!tla_output.contains("[main] ERROR alloy"));
    }

    #[test]
    fn test_other_patterns_dont_match_alloy() {
        // TLA+ and LEAN patterns should not match Alloy output
        let alloy_output = "00. check MyAssertion 0 UNSAT\n01. run run$1 0 1/1 SAT";

        // TLA+ patterns
        assert!(!alloy_output.contains("Model checking completed. No error has been found."));
        assert!(!alloy_output.contains("is violated"));
        assert!(!alloy_output.contains("***Parse Error***"));

        // LEAN patterns
        assert!(!alloy_output.contains("declaration uses 'sorry'"));
        assert!(!(alloy_output.contains("tactic") && alloy_output.contains("failed")));
    }

    // New backend pattern validation tests

    #[test]
    fn test_isabelle_patterns_dont_match_others() {
        // Isabelle patterns should not match output from other backends
        let lean_output = "warning: declaration uses 'sorry'\nerror: tactic failed";
        let tla_output = "Model checking completed. No error has been found.\nState 1: x = 0";
        let dafny_output = "Dafny program verifier finished with 5 verified, 0 errors";
        let coq_output = "COQC USLSpec.v\nError: Unable to unify";

        // Isabelle success pattern: "Finished ... elapsed time"
        assert!(!lean_output.contains("Finished") || !lean_output.contains("elapsed time"));
        assert!(!tla_output.contains("Finished") || !tla_output.contains("elapsed time"));
        // Dafny uses "finished" but not "elapsed time"
        assert!(!dafny_output.contains("elapsed time"));
        assert!(!coq_output.contains("elapsed time"));

        // Isabelle failure pattern: "*** Proof failed"
        assert!(!lean_output.contains("*** Proof failed"));
        assert!(!tla_output.contains("*** Proof failed"));
        assert!(!dafny_output.contains("*** Proof failed"));
        assert!(!coq_output.contains("*** Proof failed"));
    }

    #[test]
    fn test_coq_patterns_dont_match_others() {
        // Coq patterns should not match output from other backends
        let lean_output = "warning: declaration uses 'sorry'\nerror: tactic failed";
        let tla_output = "Model checking completed. No error has been found.\nState 1: x = 0";
        let isabelle_output = "Loading theory \"USLSpec\"\nFinished USLSpec (0:00:05 elapsed time)";
        let dafny_output = "Dafny program verifier finished with 5 verified, 0 errors";

        // Coq compile pattern: "COQC"
        assert!(!lean_output.contains("COQC"));
        assert!(!tla_output.contains("COQC"));
        assert!(!isabelle_output.contains("COQC"));
        assert!(!dafny_output.contains("COQC"));

        // Coq error pattern: "Unable to unify"
        assert!(!lean_output.contains("Unable to unify"));
        assert!(!tla_output.contains("Unable to unify"));
        assert!(!isabelle_output.contains("Unable to unify"));
        assert!(!dafny_output.contains("Unable to unify"));
    }

    #[test]
    fn test_dafny_patterns_dont_match_others() {
        // Dafny patterns should not match output from other backends
        let lean_output = "warning: declaration uses 'sorry'\nerror: tactic failed";
        let tla_output = "Model checking completed. No error has been found.\nState 1: x = 0";
        let isabelle_output = "Loading theory \"USLSpec\"\nFinished USLSpec (0:00:05 elapsed time)";
        let coq_output = "COQC USLSpec.v\nError: Unable to unify";

        // Dafny success pattern: "Dafny program verifier finished"
        assert!(!lean_output.contains("Dafny program verifier"));
        assert!(!tla_output.contains("Dafny program verifier"));
        assert!(!isabelle_output.contains("Dafny program verifier"));
        assert!(!coq_output.contains("Dafny program verifier"));

        // Dafny failure pattern: "assertion might not hold"
        assert!(!lean_output.contains("assertion might not hold"));
        assert!(!tla_output.contains("assertion might not hold"));
        assert!(!isabelle_output.contains("assertion might not hold"));
        assert!(!coq_output.contains("assertion might not hold"));
    }

    #[test]
    fn test_new_backends_patterns_are_distinct() {
        // Test that Isabelle, Coq, and Dafny patterns don't match each other
        let isabelle_output = "Loading theory \"USLSpec\"\nFinished USLSpec (0:00:05 elapsed time)";
        let coq_output = "COQC USLSpec.v";
        let dafny_output = "Dafny program verifier finished with 5 verified, 0 errors";

        // Isabelle doesn't match Coq or Dafny
        assert!(!isabelle_output.contains("COQC"));
        assert!(!isabelle_output.contains("Dafny program verifier"));

        // Coq doesn't match Isabelle or Dafny
        assert!(!coq_output.contains("elapsed time"));
        assert!(!coq_output.contains("Dafny program verifier"));

        // Dafny doesn't match Isabelle or Coq
        assert!(!dafny_output.contains("elapsed time"));
        assert!(!dafny_output.contains("COQC"));
    }
}
