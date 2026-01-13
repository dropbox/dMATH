//! CHC Codegen - Convert MIR to Constrained Horn Clauses
//!
//! This module replaces Kani's GOTO codegen with CHC-based verification.
//! It uses kani_middle for MIR analysis and transforms, then converts
//! the processed MIR to CHC constraints for Z4 PDR verification.
//!
//! # Architecture
//!
//! ```text
//! Rust Source → rustc → MIR → kani_middle (analysis) → codegen_chc → Z4 PDR
//!                                    ↓
//!                         [reachability, transforms, checks]
//! ```
//!
//! # Comparison to Kani's GOTO Codegen
//!
//! | Kani (GOTO) | Kani Fast (CHC) |
//! |-------------|-----------------|
//! | codegen_cprover_gotoc/ | codegen_chc/ |
//! | GotoProgram | ChcSystem |
//! | CBMC | Z4 PDR |
//! | Bounded model checking | Unbounded CHC solving |
//!
//! # Function Inlining
//!
//! By default, calls to local user-defined functions are inlined into the
//! caller's MIR before CHC encoding. This enables the CHC solver to reason
//! about the semantics of helper functions rather than treating them as
//! uninterpreted functions.

mod inliner;

use crate::args::Arguments;
use crate::kani_middle::attributes::is_proof_harness_from_def_id;
use crate::kani_middle::codegen_units::CodegenUnit;
use crate::kani_middle::transform::{BodyTransformation, RustcInternalMir};
use crate::kani_queries::QueryDb;
use crate::mir_to_chc::{MirConvertConfig, convert_body_to_mir_program_with_config};
use inliner::{InlinerConfig, inline_functions};
use kani_fast_chc::mir::MirProgram;
use kani_fast_chc::{
    ChcSystem, DelegationReason, VerificationPath, VerificationResult, encode_mir_to_chc_bitvec,
    encode_mir_to_chc_with_overflow_checks, encode_mir_to_chc_with_strategy,
    program_needs_bitvec_encoding,
};
use rustc_hir::def::DefKind;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_public::rustc_internal;
use rustc_span::Symbol;
use rustc_span::def_id::DefId;
use std::collections::HashMap;
use std::rc::Rc;

/// CHC code generation context
///
/// This struct manages the conversion of MIR to CHC, using kani_middle
/// for analysis and our CHC encoder for output generation.
pub struct ChcCodegenCtx<'tcx> {
    /// The type context for the current compilation
    tcx: TyCtxt<'tcx>,
    /// Cache of converted functions (CHC output)
    function_cache: HashMap<DefId, ChcSystem>,
    /// Cache of MIR programs for functions (used for inlining, shared via Rc to avoid clones)
    mir_cache: HashMap<String, Rc<MirProgram>>,
    /// Whether to output debug information
    debug: bool,
    /// Whether to enable function inlining
    enable_inlining: bool,
    /// Inliner configuration
    inliner_config: InlinerConfig,
    /// Query database for Kani transforms (intrinsics, contracts)
    queries: QueryDb,
    /// Whether to use bitvector encoding for precise bitwise reasoning
    use_bitvec: bool,
    /// Bit width for bitvector encoding (default: 32 for i32)
    bitvec_width: u32,
    /// Whether to enable explicit overflow checking for checked operations
    enable_overflow_checks: bool,
}

impl<'tcx> ChcCodegenCtx<'tcx> {
    /// Create a new CHC codegen context
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        let mut queries = QueryDb::default();
        queries.set_args(Arguments::default());
        Self {
            tcx,
            function_cache: HashMap::new(),
            mir_cache: HashMap::new(),
            debug: std::env::var("KANI_FAST_DEBUG").is_ok(),
            enable_inlining: std::env::var("KANI_FAST_NO_INLINE").is_err(),
            inliner_config: InlinerConfig::default(),
            queries,
            use_bitvec: std::env::var("KANI_FAST_BITVEC").is_ok(),
            bitvec_width: std::env::var("KANI_FAST_BITVEC_WIDTH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32),
            enable_overflow_checks: std::env::var("KANI_FAST_OVERFLOW_CHECKS").is_ok(),
        }
    }

    /// Enable debug output
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Enable or disable function inlining
    pub fn with_inlining(mut self, enable: bool) -> Self {
        self.enable_inlining = enable;
        self
    }

    /// Enable bitvector encoding for precise bitwise reasoning
    ///
    /// When enabled, integers are encoded as fixed-width bitvectors instead of
    /// unbounded integers. This allows Z4 to reason precisely about bitwise
    /// operations like `12 & 10 = 8`.
    pub fn with_bitvec(mut self, enable: bool, width: u32) -> Self {
        self.use_bitvec = enable;
        self.bitvec_width = width;
        self
    }

    /// Enable explicit overflow checking for checked arithmetic operations
    ///
    /// When enabled, checked operations like `i32::checked_add()` will have their
    /// overflow flag properly computed rather than hardcoded to `false`. This catches
    /// integer overflow bugs but significantly increases CHC solving time.
    ///
    /// For most verification, using bitvector mode (`with_bitvec`) is preferred as
    /// it naturally models overflow semantics without the performance penalty.
    ///
    /// # Environment Variable
    /// Can also be enabled via `KANI_FAST_OVERFLOW_CHECKS=1`
    pub fn with_overflow_checks(mut self, enable: bool) -> Self {
        self.enable_overflow_checks = enable;
        self
    }

    /// Create configuration for MIR to CHC conversion based on current context settings
    fn mir_convert_config(&self) -> MirConvertConfig {
        MirConvertConfig::new().with_overflow_checks(self.enable_overflow_checks)
    }

    /// Apply Kani's MIR transformation passes before CHC encoding.
    ///
    /// The transforms rely on the Kani library models being linked so that intrinsic
    /// replacements (e.g., `kani::any`, `kani::assume`) are available.
    fn apply_kani_transforms(&mut self, def_id: DefId, body: &mut rustc_middle::mir::Body<'tcx>) {
        let kani_functions = self.queries.kani_functions();
        if kani_functions.is_empty() {
            // CRITICAL: Gracefully skip transforms when Kani library is not linked.
            //
            // DO NOT add `assert!(!kani_functions.is_empty())` or similar enforcement!
            // This bug has been re-introduced 5 times (#204, #216, #226, #308) and
            // breaks the entire test suite (678 tests) each time.
            //
            // This graceful handling allows verification of simple Rust code without
            // requiring the full Kani library. The BodyTransformation::new() call
            // would panic trying to lookup Kani intrinsic functions that don't exist.
            //
            // Note: The `--extern kani=...` flag doesn't make the crate discoverable
            // via rustc_public::find_crates() - the library must be explicitly
            // `extern crate kani;` declared in the source code being compiled.
            if self.debug {
                println!("Skipping Kani transforms (Kani library not present)");
            }
            return;
        }

        // Run Kani's MIR instrumentation pipeline using the stable-MIR transformer,
        // then convert back to rustc_middle MIR for CHC encoding.
        let instance = Instance::mono(self.tcx, def_id);
        let stable_instance = rustc_internal::stable(instance);
        let mut transformer =
            BodyTransformation::new(&self.queries, self.tcx, &CodegenUnit::default());
        let transformed_body = transformer.body(self.tcx, stable_instance);
        *body = transformed_body.internal_mir(self.tcx);

        if self.debug {
            println!(
                "Applied Kani transforms for {} ({} kani functions)",
                self.tcx.def_path_str(def_id),
                kani_functions.len()
            );
        }
    }

    /// Find all proof harnesses in the crate
    ///
    /// This uses kani_middle's attribute parsing to find functions marked with
    /// `#[kani::proof]` or related attributes.
    pub fn find_harnesses(&self) -> Vec<DefId> {
        let mut harnesses = Vec::new();

        // Symbols for legacy/fallback attribute detection
        let kani_proof_sym = Symbol::intern("kani_proof");
        let proof_sym = Symbol::intern("proof");

        for item_id in self.tcx.hir_free_items() {
            let def_id = item_id.owner_id.to_def_id();

            // Only consider functions
            if !matches!(self.tcx.def_kind(def_id), DefKind::Fn) {
                continue;
            }

            let name = self.tcx.def_path_str(def_id);

            // Prefer canonical kani_middle attribute parsing to catch #[kani::proof] and
            // #[kani::proof_for_contract] reliably.
            // Use the internal DefId version to avoid TLV context issues with rustc_internal.
            let has_kani_proof = is_proof_harness_from_def_id(self.tcx, def_id)
                || self.tcx.get_attrs(def_id, kani_proof_sym).next().is_some()
                || self.tcx.get_attrs(def_id, proof_sym).next().is_some();

            // Also check naming conventions as fallback
            let has_proof_name = name.ends_with("_proof")
                || name.ends_with("::proof")
                || (name.contains("::test_") && !name.contains("::tests::"));

            if has_kani_proof || has_proof_name {
                if self.debug {
                    tracing::debug!(
                        "Found harness: {} (attr={}, name={})",
                        name,
                        has_kani_proof,
                        has_proof_name
                    );
                }
                harnesses.push(def_id);
            }
        }

        harnesses
    }

    /// Collect all available function bodies for inlining
    ///
    /// This populates the mir_cache with MirPrograms for all functions, methods,
    /// and closures in the crate.
    fn collect_function_bodies(&mut self) {
        if !self.mir_cache.is_empty() {
            return; // Already collected
        }

        // Collect all items with MIR - this includes functions, associated functions
        // (methods in impl blocks), and closures
        for local_def_id in self.tcx.mir_keys(()) {
            let def_id = local_def_id.to_def_id();
            let def_kind = self.tcx.def_kind(def_id);

            // Collect functions, associated functions (methods), and closures
            if !matches!(def_kind, DefKind::Fn | DefKind::AssocFn | DefKind::Closure) {
                continue;
            }

            // Skip if already have MIR or not available
            if !self.tcx.is_mir_available(def_id) {
                continue;
            }

            let name = self.tcx.def_path_str(def_id);

            // Skip if already cached (shouldn't happen)
            if self.mir_cache.contains_key(&name) {
                continue;
            }

            // Get the optimized MIR body and convert to MirProgram
            let mut body = self.tcx.optimized_mir(def_id).clone();
            self.apply_kani_transforms(def_id, &mut body);
            let config = self.mir_convert_config();
            let mir_program = Rc::new(convert_body_to_mir_program_with_config(
                self.tcx, &body, &config,
            ));

            self.mir_cache.insert(name.clone(), mir_program.clone());

            // For closures, also add with the call method names used in actual calls
            // Closures implement Fn/FnMut/FnOnce traits
            if def_kind == DefKind::Closure {
                let call_name = format!("<{} as std::ops::Fn<_>>::call", name);
                let call_once_name = format!("<{} as std::ops::FnOnce<_>>::call_once", name);
                let call_mut_name = format!("<{} as std::ops::FnMut<_>>::call_mut", name);

                self.mir_cache.insert(call_name, mir_program.clone());
                self.mir_cache.insert(call_once_name, mir_program.clone());
                self.mir_cache.insert(call_mut_name, mir_program);
            }

            if self.debug {
                let kind_str = match def_kind {
                    DefKind::Fn => "function",
                    DefKind::AssocFn => "method",
                    DefKind::Closure => "closure",
                    _ => "other",
                };
                println!("Collected {}: {}", kind_str, name);
            }
        }

        if self.debug {
            println!(
                "Collected {} function/closure bodies for inlining",
                self.mir_cache.len()
            );
        }
    }

    /// Generate CHC for a single function
    ///
    /// This method:
    /// 1. Gets the MIR body from rustc
    /// 2. Collects function bodies for inlining (if enabled)
    /// 3. Performs function inlining (if enabled)
    /// 4. Converts to our MirProgram representation
    /// 5. Encodes to CHC
    pub fn codegen_function(&mut self, def_id: DefId) -> Result<ChcSystem, CodegenError> {
        // Check cache first
        if let Some(cached) = self.function_cache.get(&def_id) {
            return Ok(cached.clone());
        }

        let name = self.tcx.def_path_str(def_id);

        // Check if MIR is available
        if !self.tcx.is_mir_available(def_id) {
            return Err(CodegenError::MirNotAvailable(name));
        }

        // Collect all function bodies for inlining (lazy initialization)
        if self.enable_inlining {
            self.collect_function_bodies();
        }

        // Get the optimized MIR body and run Kani's MIR transforms before encoding
        let mut body = self.tcx.optimized_mir(def_id).clone();
        self.apply_kani_transforms(def_id, &mut body);

        // Convert MIR to our simplified representation
        let config = self.mir_convert_config();
        let mut mir_program = convert_body_to_mir_program_with_config(self.tcx, &body, &config);

        // Perform function inlining if enabled
        if self.enable_inlining && !self.mir_cache.is_empty() {
            if self.debug {
                println!("\n--- Before inlining ---");
                self.debug_print_mir(&name, &mir_program);
            }

            mir_program =
                inline_functions(&mir_program, &self.mir_cache, &self.inliner_config).into_owned();

            if self.debug {
                println!("\n--- After inlining ---");
                self.debug_print_mir(&name, &mir_program);
            }
        } else if self.debug {
            self.debug_print_mir(&name, &mir_program);
        }

        // Check if bitvector encoding should be used
        // Use bitvec if: (1) explicitly enabled via env var, OR
        //                (2) auto-detect bitwise ops that would benefit from BV theory
        let use_bitvec = self.use_bitvec || self.should_use_bitvec(&mir_program);

        if use_bitvec {
            // Use BitVec encoding for precise bitwise reasoning
            if self.debug {
                println!(
                    "  Strategy: bitvec (QF_BV theory, {} bits)",
                    self.bitvec_width
                );
            }
            let chc = encode_mir_to_chc_bitvec(&mir_program, self.bitvec_width);
            self.function_cache.insert(def_id, chc.clone());
            return Ok(chc);
        }

        // When overflow checks are enabled, use direct CHC encoding that preserves overflow flags
        // (strategy selection uses optimizations that eliminate overflow checks)
        if self.enable_overflow_checks {
            if self.debug {
                println!("  Strategy: overflow-checks (preserving overflow flags)");
            }
            let chc = encode_mir_to_chc_with_overflow_checks(&mir_program);
            self.function_cache.insert(def_id, chc.clone());
            return Ok(chc);
        }

        // Encode to CHC with strategy selection
        // This uses algebraic rewriting for bitwise patterns when possible,
        // and may delegate to Kani for complex cases
        let result = encode_mir_to_chc_with_strategy(&mir_program);

        match result {
            VerificationResult::ChcResult {
                chc,
                strategy,
                rewrites_applied,
            } => {
                if self.debug {
                    let strategy_str = match &strategy {
                        VerificationPath::ChcFast => "fast (no bitwise)".to_string(),
                        VerificationPath::ChcRewritten { rewritten_ops } => {
                            format!("rewritten (algebraic, {} ops)", rewritten_ops.len())
                        }
                        VerificationPath::DelegateKani { .. } => {
                            "delegate (should not happen)".to_string()
                        }
                    };
                    println!(
                        "  Strategy: {}, rewrites applied: {}",
                        strategy_str, rewrites_applied
                    );
                }

                // Cache the result
                self.function_cache.insert(def_id, chc.clone());
                Ok(chc)
            }
            VerificationResult::Delegated { reason } => {
                // For delegation cases, try bitvec encoding as fallback before
                // completely giving up (bitvec can handle complex bitwise)
                if self.debug {
                    let reason_str = match &reason {
                        DelegationReason::ComplexBitwise { operations } => {
                            format!("complex bitwise ({} operations)", operations.len())
                        }
                        DelegationReason::RecursiveFunctions => "recursive functions".to_string(),
                        DelegationReason::UnsupportedFeature { feature } => {
                            format!("unsupported feature: {}", feature)
                        }
                        DelegationReason::UserRequested => "user requested".to_string(),
                    };
                    println!(
                        "  Note: Delegation suggested ({}) - trying bitvec encoding",
                        reason_str
                    );
                }

                // Use bitvec encoding for complex bitwise cases
                if matches!(reason, DelegationReason::ComplexBitwise { .. }) {
                    let chc = encode_mir_to_chc_bitvec(&mir_program, self.bitvec_width);
                    self.function_cache.insert(def_id, chc.clone());
                    return Ok(chc);
                }

                // Fall back to standard encoding (may produce unknown result)
                let chc = kani_fast_chc::encode_mir_to_chc(&mir_program);
                self.function_cache.insert(def_id, chc.clone());
                Ok(chc)
            }
        }
    }

    /// Check if bitvector encoding should be used for this MIR program
    ///
    /// Returns true if the program contains bitwise operations that benefit
    /// from native BitVec theory (e.g., `12 & 10` concrete values).
    fn should_use_bitvec(&self, program: &MirProgram) -> bool {
        program_needs_bitvec_encoding(program)
    }

    /// Generate CHC for all harnesses
    pub fn codegen_harnesses(&mut self) -> Vec<(DefId, Result<ChcSystem, CodegenError>)> {
        let harnesses = self.find_harnesses();
        harnesses
            .into_iter()
            .map(|def_id| {
                let result = self.codegen_function(def_id);
                (def_id, result)
            })
            .collect()
    }

    /// Debug print the MIR program structure
    fn debug_print_mir(&self, name: &str, program: &kani_fast_chc::mir::MirProgram) {
        println!("\n--- MIR Program for {} ---", name);
        println!(
            "Locals: {:?}",
            program.locals.iter().map(|l| &l.name).collect::<Vec<_>>()
        );
        for block in &program.basic_blocks {
            println!("Block {}: {:?}", block.id, block.terminator);
            for stmt in &block.statements {
                println!("  {:?}", stmt);
            }
        }
    }

    /// Get function name for a DefId
    pub fn function_name(&self, def_id: DefId) -> String {
        self.tcx.def_path_str(def_id)
    }
}

/// Errors that can occur during CHC codegen
#[derive(Debug, Clone)]
pub enum CodegenError {
    /// MIR is not available for the function
    MirNotAvailable(String),
    /// Type not supported for CHC encoding
    UnsupportedType(String),
    /// Operation not supported
    UnsupportedOperation(String),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::MirNotAvailable(name) => {
                write!(f, "MIR not available for function: {}", name)
            }
            CodegenError::UnsupportedType(ty) => {
                write!(f, "Unsupported type for CHC encoding: {}", ty)
            }
            CodegenError::UnsupportedOperation(op) => {
                write!(f, "Unsupported operation: {}", op)
            }
        }
    }
}

impl std::error::Error for CodegenError {}

#[cfg(test)]
mod tests {
    // Tests will use the rustc test framework
    // For now, integration tests are in the test binaries
}
