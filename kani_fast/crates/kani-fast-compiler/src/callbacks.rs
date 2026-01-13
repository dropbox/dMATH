//! Rustc callbacks for Kani Fast
//!
//! This module implements the rustc callbacks that intercept compilation
//! to extract MIR for verification.
//!
//! # Architecture
//!
//! The callbacks integrate with kani_middle (copied from upstream Kani) for
//! MIR analysis and ChcCodegenCtx for CHC generation:
//!
//! ```text
//! rustc → Callbacks::after_analysis() → ChcCodegenCtx → CHC → Z4
//!                                            ↓
//!                                    kani_middle (analysis)
//! ```

use rustc_driver::{Callbacks, Compilation};
use rustc_interface::Config;
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use rustc_public::rustc_internal;
use rustc_span::def_id::DefId;

use crate::codegen_chc::ChcCodegenCtx;
use crate::mir_collector::MirCollector;
use kani_fast_chc::{ChcResult, ChcSolver, ChcSolverConfig};
use tokio::runtime::Runtime;
use tracing::warn;

/// Result of verifying a single harness
#[derive(Debug, Clone)]
pub struct HarnessResult {
    /// Name of the harness function
    pub name: String,
    /// Whether verification succeeded
    pub verified: bool,
    /// Human-readable result message
    pub message: String,
    /// SMT-LIB2 encoding (for debugging)
    pub smt2: Option<String>,
}

/// Verification results for the entire crate
#[derive(Debug, Clone, Default)]
pub struct VerificationResults {
    /// Results for each harness
    pub harnesses: Vec<HarnessResult>,
    /// Total number of harnesses found
    pub total_harnesses: usize,
    /// Number of verified harnesses
    pub verified_count: usize,
    /// Number of failed harnesses
    pub failed_count: usize,
}

impl VerificationResults {
    /// Create empty results
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a harness result
    pub fn add(&mut self, result: HarnessResult) {
        if result.verified {
            self.verified_count += 1;
        } else {
            self.failed_count += 1;
        }
        self.total_harnesses += 1;
        self.harnesses.push(result);
    }

    /// Check if all harnesses passed
    pub fn all_passed(&self) -> bool {
        self.failed_count == 0 && self.total_harnesses > 0
    }

    /// Print summary to stdout
    pub fn print_summary(&self) {
        println!("\n=== Kani Fast Verification Summary ===");
        println!("Total harnesses: {}", self.total_harnesses);
        println!("  Verified: {}", self.verified_count);
        println!("  Failed:   {}", self.failed_count);

        if !self.harnesses.is_empty() {
            println!("\nResults:");
            for result in &self.harnesses {
                let status = if result.verified { "✓" } else { "✗" };
                println!("  {} {}: {}", status, result.name, result.message);
            }
        }

        if self.all_passed() {
            println!("\n✓ All harnesses verified successfully!");
        } else if self.total_harnesses == 0 {
            println!("\n⚠ No harnesses found. Add #[kani::proof] to functions to verify.");
        } else {
            println!("\n✗ Some harnesses failed verification.");
        }
    }
}

/// Callbacks for the Kani Fast compiler driver
pub struct KaniFastCallbacks {
    /// Collected MIR programs ready for verification
    pub mir_collector: MirCollector,
    /// Verification results
    pub results: VerificationResults,
    /// Whether to output SMT-LIB2 for debugging
    pub debug_smt: bool,
    /// Whether to skip actual verification (just generate CHC)
    pub dry_run: bool,
}

impl KaniFastCallbacks {
    /// Create a new callbacks instance
    pub fn new() -> Self {
        // Check environment for debug flags
        let debug_smt = std::env::var("KANI_FAST_DEBUG").is_ok();
        let dry_run = std::env::var("KANI_FAST_DRY_RUN").is_ok();

        Self {
            mir_collector: MirCollector::new(),
            results: VerificationResults::new(),
            debug_smt,
            dry_run,
        }
    }

    /// Enable SMT-LIB2 debug output
    pub fn with_debug_smt(mut self, enabled: bool) -> Self {
        self.debug_smt = enabled;
        self
    }

    /// Enable dry-run mode (generate CHC but don't verify)
    pub fn with_dry_run(mut self, enabled: bool) -> Self {
        self.dry_run = enabled;
        self
    }
}

impl Default for KaniFastCallbacks {
    fn default() -> Self {
        Self::new()
    }
}

impl Callbacks for KaniFastCallbacks {
    /// Called before parsing - we don't intercept here
    fn config(&mut self, _config: &mut Config) {
        // We could modify compiler configuration here if needed
        // For now, we use default settings
        tracing::debug!("Kani Fast compiler driver initialized");
    }

    /// Called after analysis is complete
    ///
    /// This is where we extract MIR from the compiled crate.
    fn after_analysis(&mut self, _compiler: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        self.process_crate(tcx);
        // Continue compilation (or stop if we only want verification)
        Compilation::Continue
    }
}

impl KaniFastCallbacks {
    /// Process the entire crate, finding and verifying all harnesses
    ///
    /// Uses ChcCodegenCtx to manage the MIR-to-CHC pipeline with kani_middle
    /// for comprehensive MIR analysis.
    ///
    /// Note: We wrap the processing in `rustc_internal::run()` to set up the
    /// thread-local context required by Kani's stable MIR transforms. This
    /// allows `rustc_internal::stable()` calls in the transform passes to work.
    fn process_crate(&mut self, tcx: TyCtxt<'_>) {
        println!("Kani Fast: Processing crate for verification...");

        // Set up the rustc_public TLV context for stable MIR conversions.
        // This is required because Kani transforms use rustc_internal::stable().
        let _ = rustc_internal::run(tcx, || {
            // Create codegen context (integrates with kani_middle)
            let mut codegen = ChcCodegenCtx::new(tcx).with_debug(self.debug_smt);

            // Find all proof harnesses using kani_middle attribute parsing
            let harnesses = codegen.find_harnesses();
            println!("Found {} proof harness(es)", harnesses.len());

            // Verify each harness - collect results in a vec first to avoid closure issues
            let harness_results: Vec<_> = harnesses
                .iter()
                .map(|&def_id| self.verify_harness_internal(&mut codegen, def_id))
                .collect();

            // Store results
            for result in harness_results {
                self.results.add(result);
            }
        });

        // Print summary
        self.results.print_summary();
    }

    /// Internal harness verification that returns a result instead of storing directly.
    /// This is used within the rustc_internal::run closure context.
    fn verify_harness_internal(
        &self,
        codegen: &mut ChcCodegenCtx<'_>,
        def_id: DefId,
    ) -> HarnessResult {
        let name = codegen.function_name(def_id);
        println!("\nVerifying: {}", name);

        // Generate CHC using codegen context
        let chc = match codegen.codegen_function(def_id) {
            Ok(chc) => chc,
            Err(e) => {
                return HarnessResult {
                    name,
                    verified: false,
                    message: format!("Codegen error: {}", e),
                    smt2: None,
                };
            }
        };

        let smt2 = chc.to_smt2();

        if self.debug_smt {
            println!("\n--- SMT-LIB2 for {} ---\n{}\n---", name, smt2);
        }

        if self.dry_run {
            return HarnessResult {
                name,
                verified: true,
                message: "CHC generated (dry run)".to_string(),
                smt2: Some(smt2),
            };
        }

        // Solve using Z4 PDR (synchronous call)
        let result = run_chc_verification_sync(&smt2);

        HarnessResult {
            name,
            verified: result.0,
            message: result.1,
            smt2: self.debug_smt.then_some(smt2),
        }
    }
}

// Note: find_proof_harnesses has been moved to ChcCodegenCtx::find_harnesses()
// for better integration with kani_middle attribute parsing

/// Run CHC verification using Z4 native Rust solver
///
/// This uses the z4-chc crate directly for maximum performance.
/// Z4 is the sole SMT backend - if it returns unknown, that's the answer.
fn run_chc_verification_sync(smt2: &str) -> (bool, String) {
    use z4_chc::{ChcParser, PdrConfig, PdrResult, PdrSolver};

    // Parse the SMT-LIB2 CHC input
    let mut problem = match ChcParser::parse(smt2) {
        Ok(p) => p,
        Err(e) => return (false, format!("Z4 failed to parse CHC: {}", e)),
    };

    // Scalarize constant array selects for better solving
    problem.try_scalarize_const_array_selects();

    // Configure PDR solver with reasonable limits
    let config = PdrConfig {
        max_frames: 100,
        max_iterations: 10_000,
        ..Default::default()
    };

    // Create solver and solve
    let mut solver = PdrSolver::new(problem, config);
    let result = solver.solve();

    match result {
        PdrResult::Safe(_model) => {
            // Found inductive invariant - property holds (cross-check to avoid false positives)
            if crosscheck_enabled() {
                match cross_check_with_external_solver(smt2) {
                    CrossCheckOutcome::Override(result) => return result,
                    CrossCheckOutcome::Skipped(reason) => {
                        return (
                            false,
                            format!(
                                "Property unknown: Z4 PDR reported Safe but cross-check skipped ({reason}). Install z4 CLI or set KANI_FAST_CHC_CROSSCHECK=0 to trust PDR result (may be unsound)"
                            ),
                        );
                    }
                    CrossCheckOutcome::ConfirmedSafe => { /* fall through */ }
                }
            }
            (true, "Property verified (Z4 found invariant)".to_string())
        }
        PdrResult::Unsafe(_cex) => {
            // Counterexample found - property violated
            (
                false,
                "Property violated (Z4 found counterexample)".to_string(),
            )
        }
        PdrResult::Unknown => {
            let base_msg = "Z4 returned unknown (reached iteration/frame limit)";
            if std::env::var("KANI_FAST_AI_HINT").is_ok() {
                (
                    false,
                    format!(
                        "{} (Hint: For complex loops, use VerificationEngine with ai_synthesis=true)",
                        base_msg
                    ),
                )
            } else {
                (false, base_msg.to_string())
            }
        }
    }
}

/// Result of attempting a cross-check run.
#[derive(Debug)]
enum CrossCheckOutcome {
    /// Cross-check agreed with the PDR result (safe)
    ConfirmedSafe,
    /// Cross-check produced a different verdict (override the PDR result)
    Override((bool, String)),
    /// Cross-check could not run (solver missing/failed)
    Skipped(String),
}

/// Optional cross-check using the CLI CHC solver to guard against solver regressions.
fn cross_check_with_external_solver(smt2: &str) -> CrossCheckOutcome {
    // Build solver with Z4 backend
    let solver = match ChcSolver::new(ChcSolverConfig::new()) {
        Ok(s) => s,
        Err(e) => {
            warn!("CHC cross-check skipped (solver unavailable): {}", e);
            return CrossCheckOutcome::Skipped(e.to_string());
        }
    };

    // Run the async solver in a lightweight runtime
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            warn!("CHC cross-check skipped (runtime init failed): {}", e);
            return CrossCheckOutcome::Skipped(e.to_string());
        }
    };

    match rt.block_on(solver.solve_smt2(smt2)) {
        Ok(ChcResult::Unsat { .. }) => CrossCheckOutcome::Override((
            false,
            "Property violated (cross-check solver found counterexample)".to_string(),
        )),
        Ok(ChcResult::Unknown { reason, .. }) => CrossCheckOutcome::Override((
            false,
            format!("Property unknown after cross-check: {}", reason),
        )),
        Ok(ChcResult::Sat { .. }) => CrossCheckOutcome::ConfirmedSafe,
        Err(e) => {
            warn!("CHC cross-check failed: {}", e);
            CrossCheckOutcome::Skipped(e.to_string())
        }
    }
}

/// Cross-checks are enabled by default to preserve soundness.
/// Disable with `KANI_FAST_CHC_CROSSCHECK=0` if a secondary solver is unavailable.
fn crosscheck_enabled() -> bool {
    std::env::var("KANI_FAST_CHC_CROSSCHECK")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
        std::sync::OnceLock::new();

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap()
    }

    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &'static str) -> Self {
            let original = std::env::var_os(key);
            // Nightly marks env mutation helpers as unsafe.
            unsafe { std::env::set_var(key, value) };
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(val) = &self.original {
                unsafe { std::env::set_var(self.key, val) };
            } else {
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    #[test]
    fn negative_switch_cross_check_detects_violation() {
        let _env_lock = env_lock();
        // This CHC models: _1 = 20; assert!(_1 == 9999);
        let smt = r#"
(set-logic HORN)

(declare-fun Inv (Int Int) Bool)

; initial_state
(assert (forall ((pc Int) (_1 Int)) (=> (= pc 0) (Inv pc _1))))
; transition
(assert (forall ((pc Int) (_1 Int) (pc_next Int) (_1_next Int))
    (=> (and (Inv pc _1)
             (or (and (= pc 0) (= _1_next 20) (= _1_next 9999) (= pc_next 1))
                 (and (= pc 0) (= _1_next 20) (and (not (= _1_next 9999)) (= pc 0)) (= pc_next 2))
                 (and (= pc 1) (= _1_next _1) (= pc_next -1))
                 (and (= pc 2) (= _1_next _1) (= pc_next 999999))))
        (Inv pc_next _1_next))))
; property_assertions
(assert (forall ((pc Int) (_1 Int)) (=> (and (Inv pc _1) (not (and (not (= pc -2)) (not (= pc 999999))))) false)))

(check-sat)
"#;

        // Ensure cross-checks remain enabled for the test
        // SAFETY: We're in a single-threaded test context and the env var is not being
        // read concurrently by other code. This is a test-only pattern to reset state.
        unsafe { std::env::remove_var("KANI_FAST_CHC_CROSSCHECK") };
        let (verified, message) = run_chc_verification_sync(smt);
        assert!(
            !verified,
            "expected violation to be reported, got verified: {message}"
        );
    }

    #[test]
    fn cross_check_skips_when_solver_missing() {
        let _env_lock = env_lock();
        // Empty PATH ensures the external z4 binary cannot be found, so the guard
        // should downgrade the result to unknown instead of trusting PDR alone.
        let _path_guard = EnvVarGuard::set("PATH", "");
        let outcome = cross_check_with_external_solver("(set-logic HORN)\n(check-sat)\n");
        match outcome {
            CrossCheckOutcome::Skipped(reason) => {
                assert!(
                    !reason.is_empty(),
                    "expected a skip reason when cross-check is unavailable"
                );
            }
            other => panic!("expected cross-check to be skipped, got {:?}", other),
        }
    }
}
