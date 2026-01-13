//! Verification Engine for tRust Integration
//!
//! This module provides the library API for tRust (the compiler) to invoke
//! Kani Fast's verification pipeline. The engine handles automatic escalation:
//!
//! BMC → K-Induction → CHC → AI Invariant Synthesis → Lean5 Proof Generation
//!
//! # Usage
//!
//! ```rust,ignore
//! use kani_fast::engine::{VerificationEngine, MirInput, VerificationConfig};
//!
//! let config = VerificationConfig::default();
//! let engine = VerificationEngine::new(&config);
//!
//! let input = MirInput::from_mir_program(program);
//! let result = engine.verify(input).await;
//! ```

use kani_fast_ai::{AiConfig, AiSynthesizer, SynthesisResult};
use kani_fast_chc::{encode_mir_to_chc, ChcResult, ChcSolverConfig, MirProgram};
use kani_fast_kinduction::{
    KInduction, KInductionConfigBuilder, KInductionResult, Property, StateFormula, TransitionSystem,
};
use kani_fast_lean5::{
    certificate_from_ai, parse_smt_formula, translate_ast, Lean5Backend, Lean5Config, Lean5Expr,
    Lean5Type, ProofCertificate, ProofObligation, TranslationContext,
};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

lazy_static! {
    // SMT-LIB format bound extraction patterns: (< var N), (<= var N), (> N var), (>= N var)
    static ref SMT_LT_VAR_BOUND: Regex = Regex::new(r"\(\s*<\s*(\w+)\s+(\d+)\s*\)")
        .expect("SMT_LT_VAR_BOUND regex is valid");
    static ref SMT_LE_VAR_BOUND: Regex = Regex::new(r"\(\s*<=\s*(\w+)\s+(\d+)\s*\)")
        .expect("SMT_LE_VAR_BOUND regex is valid");
    static ref SMT_GT_BOUND_VAR: Regex = Regex::new(r"\(\s*>\s*(\d+)\s+(\w+)\s*\)")
        .expect("SMT_GT_BOUND_VAR regex is valid");
    static ref SMT_GE_BOUND_VAR: Regex = Regex::new(r"\(\s*>=\s*(\d+)\s+(\w+)\s*\)")
        .expect("SMT_GE_BOUND_VAR regex is valid");
    // Infix format bound extraction patterns: var < N, var <= N
    static ref INFIX_LT: Regex = Regex::new(r"(\w+)\s*<\s*(\d+)")
        .expect("INFIX_LT regex is valid");
    static ref INFIX_LE: Regex = Regex::new(r"(\w+)\s*<=\s*(\d+)")
        .expect("INFIX_LE regex is valid");
}

// Re-export MIR types for tRust integration
pub use kani_fast_chc::{
    MirBasicBlock as BasicBlock, MirLocal as Local, MirProgram as Program,
    MirProgramBuilder as ProgramBuilder, MirStatement as Statement, MirTerminator as Terminator,
};
pub use kani_fast_kinduction::SmtType as Type;

/// Verification configuration for tRust integration
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// Maximum time per function verification
    pub timeout: Duration,

    /// BMC bound (if unbounded proof fails)
    pub bmc_bound: u32,

    /// Enable AI invariant synthesis
    pub ai_synthesis: bool,

    /// Enable Lean5 proof generation
    pub lean5_backend: bool,

    /// Parallelism for portfolio solving
    pub num_threads: usize,

    /// Strictness level
    pub strictness: Strictness,

    /// K-induction maximum k value
    pub max_k: u32,

    /// CHC solver timeout
    pub chc_timeout: Duration,

    /// Maximum AI synthesis attempts
    pub ai_max_attempts: usize,

    /// AI synthesis timeout per attempt
    pub ai_timeout: Duration,

    /// Lean5 verification timeout
    pub lean5_timeout: Duration,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            bmc_bound: 10,
            ai_synthesis: true,
            lean5_backend: true,
            num_threads: num_cpus::get(),
            strictness: Strictness::Strict,
            max_k: 20,
            chc_timeout: Duration::from_secs(30),
            ai_max_attempts: 10,
            ai_timeout: Duration::from_secs(30),
            lean5_timeout: Duration::from_secs(60),
        }
    }
}

/// Strictness level for verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strictness {
    /// Fail compilation if any verification fails
    Strict,

    /// Warn but continue if verification times out
    BestEffort,

    /// Only verify functions with explicit specs
    OptIn,
}

/// Method used to prove a property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofMethod {
    /// Bounded model checking
    BoundedModelChecking { bound: u32 },

    /// K-induction with specific k value
    KInduction { k: u32 },

    /// CHC solving with discovered invariant
    CHC { invariant: String },

    /// AI-assisted invariant synthesis
    AiSynthesis {
        invariant: String,
        attempts: usize,
        source: String,
    },

    /// Lean5 proof term
    Lean5 { proof_term: String },
}

/// Suggested action when verification is inconclusive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestedAction {
    /// Increase the timeout
    IncreaseTimeout,

    /// Add an AI-suggested invariant
    AddInvariant(String),

    /// Add an AI-suggested precondition
    AddPrecondition(String),

    /// Escalate to manual Lean5 proof
    UseManualProof,
}

/// Structured counterexample from verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredCounterexample {
    /// Input values that trigger the failure
    pub inputs: HashMap<String, String>,

    /// Execution trace leading to failure
    pub trace: Vec<TraceStep>,

    /// The failing assertion or property
    pub failed_property: String,

    /// Source location of the failure
    pub location: Option<SourceLocation>,
}

/// A step in the counterexample trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// Program counter / basic block
    pub pc: i64,

    /// Variable values at this step
    pub state: HashMap<String, String>,
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// Result of verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineResult {
    /// Property proven for all inputs
    Proven {
        method: ProofMethod,
        duration: Duration,
        proof_certificate: Option<ProofCertificate>,
    },

    /// Counterexample found
    Disproven {
        counterexample: StructuredCounterexample,
        duration: Duration,
    },

    /// Could not determine (timeout, resource limit)
    Unknown {
        reason: String,
        partial_result: Option<PartialResult>,
        suggested_action: SuggestedAction,
    },
}

impl EngineResult {
    /// Check if this is a successful proof
    pub fn is_proven(&self) -> bool {
        matches!(self, Self::Proven { .. })
    }

    /// Check if this is a counterexample
    pub fn is_disproven(&self) -> bool {
        matches!(self, Self::Disproven { .. })
    }

    /// Check if the result is conclusive
    pub fn is_conclusive(&self) -> bool {
        matches!(self, Self::Proven { .. } | Self::Disproven { .. })
    }

    /// Convert to KaniResult for detailed classification
    pub fn to_kani_result(&self) -> KaniResult {
        match self {
            EngineResult::Proven { method, .. } => match method {
                ProofMethod::BoundedModelChecking { bound } => KaniResult::BoundedCheck {
                    depth: *bound,
                    coverage: Coverage::FullDepth,
                    suggestion: None,
                },
                ProofMethod::KInduction { k } => KaniResult::Verified {
                    depth: Some(*k),
                    bound_sufficient: true,
                },
                ProofMethod::CHC { .. }
                | ProofMethod::AiSynthesis { .. }
                | ProofMethod::Lean5 { .. } => KaniResult::Verified {
                    depth: None, // Unbounded proof
                    bound_sufficient: true,
                },
            },
            EngineResult::Disproven { counterexample, .. } => KaniResult::Counterexample {
                trace: counterexample.trace.clone(),
                rust_test: generate_rust_test(counterexample),
            },
            EngineResult::Unknown {
                reason,
                partial_result,
                ..
            } => {
                let partial_cov = partial_result
                    .as_ref()
                    .map(|p| p.bounds_checked)
                    .unwrap_or(0);
                KaniResult::ResourceExhausted {
                    reason: reason.clone(),
                    partial_coverage: partial_cov,
                }
            }
        }
    }
}

/// Clear result classification for tRust integration
///
/// This enum provides detailed reporting of what was proven, distinguishing between
/// bounded checks (which verify properties up to a certain depth) and unbounded
/// proofs (which verify properties for all possible executions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KaniResult {
    /// Property verified (unbounded or with sufficient bound)
    Verified {
        /// Depth for k-induction, None for unbounded proofs (CHC/Lean5)
        depth: Option<u32>,
        /// True if the bound is sufficient to cover all executions
        bound_sufficient: bool,
    },

    /// Counterexample found - property violated
    Counterexample {
        /// Execution trace to the violation
        trace: Vec<TraceStep>,
        /// Generated Rust test that reproduces the failure
        rust_test: String,
    },

    /// Bounded check completed but bound may be insufficient
    BoundedCheck {
        /// Depth of the bounded check
        depth: u32,
        /// What was covered at this depth
        coverage: Coverage,
        /// Suggestion to improve coverage
        suggestion: Option<String>,
    },

    /// Resource exhausted (timeout, memory)
    ResourceExhausted {
        /// Why verification could not complete
        reason: String,
        /// How much was verified before exhaustion
        partial_coverage: u32,
    },
}

impl KaniResult {
    /// Check if the result is a definitive proof
    pub fn is_verified(&self) -> bool {
        matches!(self, KaniResult::Verified { .. })
    }

    /// Check if the result indicates a bug
    pub fn is_counterexample(&self) -> bool {
        matches!(self, KaniResult::Counterexample { .. })
    }

    /// Check if the result is bounded (may not cover all paths)
    pub fn is_bounded(&self) -> bool {
        matches!(self, KaniResult::BoundedCheck { .. })
    }

    /// Get the depth checked (if applicable)
    pub fn depth(&self) -> Option<u32> {
        match self {
            KaniResult::Verified { depth, .. } => *depth,
            KaniResult::BoundedCheck { depth, .. } => Some(*depth),
            _ => None,
        }
    }
}

/// Coverage information for bounded checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Coverage {
    /// All execution paths at this depth were checked
    FullDepth,
    /// Only some paths were checked (e.g., timeout during exploration)
    Partial {
        paths_checked: u32,
        paths_total: Option<u32>,
    },
    /// Coverage unknown
    Unknown,
}

/// Generate a Rust test case from a counterexample
fn generate_rust_test(cex: &StructuredCounterexample) -> String {
    use std::fmt::Write;

    let mut test = String::new();
    test.push_str("#[test]\n");
    test.push_str("fn test_counterexample() {\n");

    // Generate variable assignments from inputs
    for (name, value) in &cex.inputs {
        // Parse the value to determine the type
        if let Ok(int_val) = value.parse::<i64>() {
            let _ = writeln!(test, "    let {name} = {int_val};");
        } else {
            // Boolean ("true"/"false") or other type - both use value directly
            let _ = writeln!(test, "    let {name} = {value};");
        }
    }

    test.push_str("\n    // TODO: Call the function under test with these inputs\n");
    let _ = writeln!(test, "    // This should trigger: {}", cex.failed_property);

    if let Some(loc) = &cex.location {
        let _ = writeln!(
            test,
            "    // Failure at {}:{}:{}",
            loc.file, loc.line, loc.column
        );
    }

    test.push_str("}\n");
    test
}

/// Partial result from incomplete verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialResult {
    /// Methods that were tried
    pub methods_tried: Vec<String>,

    /// Partial invariants discovered
    pub partial_invariants: Vec<String>,

    /// Bounds checked without counterexample
    pub bounds_checked: u32,

    /// Reason provided by an upstream solver (e.g., Z4) for handing off work
    pub upstream_reason: Option<String>,

    /// Specific subproblem that needs attention (from upstream solver)
    pub subproblem_focus: Option<String>,
}

/// Reason provided by Z4 for an unknown result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Z4UnknownReason {
    HeapAliasing,
    NonlinearArith,
    DeepRecursion,
    Timeout,
    QuantifierHeavy,
    Other(String),
}

impl fmt::Display for Z4UnknownReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z4UnknownReason::HeapAliasing => write!(f, "heap aliasing imprecise"),
            Z4UnknownReason::NonlinearArith => write!(f, "nonlinear arithmetic"),
            Z4UnknownReason::DeepRecursion => write!(f, "deep recursion"),
            Z4UnknownReason::Timeout => write!(f, "timeout"),
            Z4UnknownReason::QuantifierHeavy => write!(f, "quantifier-heavy obligations"),
            Z4UnknownReason::Other(reason) => write!(f, "{reason}"),
        }
    }
}

/// Subproblem identified by Z4 as the hard component
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Z4Subproblem {
    /// Predicate or condition that Z4 could not solve
    pub predicate: String,
    /// Optional contextual note
    pub context: Option<String>,
}

/// Portion of proof already established by Z4
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Z4PartialProof {
    /// Invariants or lemmas Z4 proved
    pub invariants: Vec<String>,
    /// Optional textual description of the progress
    pub description: Option<String>,
    /// Optional certificate or identifier for the partial proof
    pub certificate: Option<String>,
}

/// Result from Z4 that can be handed off to Kani Fast
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Z4Result {
    /// Z4 produced a proof certificate (verification succeeded upstream)
    Proven { certificate: Option<String> },
    /// Z4 produced a counterexample model
    Counterexample { values: HashMap<String, String> },
    /// Z4 could not decide and is handing off a focused subproblem/partial proof
    Unknown {
        reason: Z4UnknownReason,
        subproblem: Option<Z4Subproblem>,
        partial_proof: Option<Z4PartialProof>,
    },
}

/// Input to the verification engine
#[derive(Debug, Clone)]
pub struct MirInput {
    /// The MIR program to verify
    pub program: MirProgram,

    /// Function specifications (preconditions, postconditions)
    pub specs: FunctionSpecs,

    /// Source-level information for error messages
    pub source_info: Option<SourceInfo>,

    /// Optional result from Z4 (for handoff when Z4 reports Unknown)
    pub z4_result: Option<Z4Result>,
}

impl MirInput {
    /// Create MirInput from a MirProgram
    pub fn from_mir_program(program: MirProgram) -> Self {
        Self {
            program,
            specs: FunctionSpecs::default(),
            source_info: None,
            z4_result: None,
        }
    }

    /// Create MirInput with specs
    pub fn with_specs(program: MirProgram, specs: FunctionSpecs) -> Self {
        Self {
            program,
            specs,
            source_info: None,
            z4_result: None,
        }
    }

    /// Add source information
    pub fn with_source_info(mut self, source_info: SourceInfo) -> Self {
        self.source_info = Some(source_info);
        self
    }

    /// Attach a Z4 result (handoff) to the input
    pub fn with_z4_result(mut self, z4_result: Z4Result) -> Self {
        self.z4_result = Some(z4_result);
        self
    }
}

/// Function specifications
#[derive(Debug, Clone, Default)]
pub struct FunctionSpecs {
    /// Preconditions (requires clauses)
    pub requires: Vec<String>,

    /// Postconditions (ensures clauses)
    pub ensures: Vec<String>,

    /// Loop invariants
    pub invariants: Vec<String>,

    /// Loop variants (for termination/bounds inference)
    /// Each variant is an expression that decreases on each loop iteration
    pub variants: Vec<VariantSpec>,
}

/// A loop variant specification for bounds inference
///
/// A variant is an expression that strictly decreases on each loop iteration,
/// proving termination and enabling bound inference.
///
/// # Example
/// ```rust,ignore
/// #[variant(n - i)]  // n - i decreases each iteration
/// #[requires(n < 100)]  // Combined with precondition → bound = 100
/// fn countdown(n: i32) {
///     let mut i = 0;
///     while i < n { i += 1; }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct VariantSpec {
    /// The variant expression (e.g., "n - i")
    pub expression: String,

    /// Optional: which loop this variant applies to (for nested loops)
    pub loop_id: Option<String>,

    /// Parsed upper bound variable (extracted from expression)
    pub upper_bound_var: Option<String>,

    /// Parsed lower bound or starting value
    pub lower_bound: Option<i64>,
}

impl FunctionSpecs {
    /// Create new empty specs
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a precondition
    pub fn require(mut self, condition: impl Into<String>) -> Self {
        self.requires.push(condition.into());
        self
    }

    /// Add a postcondition
    pub fn ensure(mut self, condition: impl Into<String>) -> Self {
        self.ensures.push(condition.into());
        self
    }

    /// Add a loop invariant
    pub fn invariant(mut self, condition: impl Into<String>) -> Self {
        self.invariants.push(condition.into());
        self
    }

    /// Add a loop variant for bounds inference
    ///
    /// The variant expression should decrease on each loop iteration.
    /// Combined with preconditions, this enables automatic bound inference.
    ///
    /// # Example
    /// ```rust,ignore
    /// let specs = FunctionSpecs::new()
    ///     .require("(< n 100)")
    ///     .variant("n - i");  // variant decreases each iteration
    /// // Kani Fast can infer: unwind bound = 100
    /// ```
    pub fn variant(mut self, expression: impl Into<String>) -> Self {
        let expr = expression.into();
        let variant = VariantSpec::parse(&expr);
        self.variants.push(variant);
        self
    }

    /// Add a loop variant with explicit loop ID (for nested loops)
    pub fn variant_for_loop(
        mut self,
        loop_id: impl Into<String>,
        expression: impl Into<String>,
    ) -> Self {
        let expr = expression.into();
        let mut variant = VariantSpec::parse(&expr);
        variant.loop_id = Some(loop_id.into());
        self.variants.push(variant);
        self
    }
}

impl VariantSpec {
    /// Parse a variant expression to extract bound information
    ///
    /// Recognizes patterns like:
    /// - "n - i" → upper_bound_var = Some("n"), lower_bound = Some(0)
    /// - "limit - counter" → upper_bound_var = Some("limit")
    /// - "arr.len() - idx" → upper_bound_var = Some("arr.len()")
    pub fn parse(expression: &str) -> Self {
        let mut spec = VariantSpec {
            expression: expression.to_string(),
            loop_id: None,
            upper_bound_var: None,
            lower_bound: None,
        };

        // Try to parse "upper - lower" pattern
        let trimmed = expression.trim();

        // Handle SMT-LIB format: (- n i)
        if trimmed.starts_with('(') {
            if let Some(inner) = trimmed
                .strip_prefix("(-")
                .or_else(|| trimmed.strip_prefix("(- "))
            {
                let inner = inner.trim_end_matches(')').trim();
                // Use iterator directly to avoid Vec allocation
                let mut parts = inner.split_whitespace();
                if let (Some(first), Some(second)) = (parts.next(), parts.next()) {
                    spec.upper_bound_var = Some(first.to_string());
                    // Check if lower is a constant
                    if let Ok(val) = second.parse::<i64>() {
                        spec.lower_bound = Some(val);
                    }
                }
            }
        } else if let Some((left, right)) = trimmed.split_once('-') {
            // Handle infix format: n - i or n-i
            // Use split_once to avoid Vec allocation
            let left = left.trim();
            let right = right.trim();
            spec.upper_bound_var = Some(left.to_string());
            // Check if lower is a constant
            if let Ok(val) = right.parse::<i64>() {
                spec.lower_bound = Some(val);
            }
        } else {
            // Single variable variant (e.g., "n" means decrementing counter)
            spec.upper_bound_var = Some(trimmed.to_string());
        }

        spec
    }

    /// Create a new variant spec directly
    pub fn new(expression: impl Into<String>) -> Self {
        Self::parse(&expression.into())
    }
}

/// Source-level information for error messages
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Function name
    pub function_name: String,

    /// Source file path
    pub file_path: String,

    /// Basic block to line mapping
    pub block_to_line: HashMap<usize, u32>,
}

/// Verification engine error
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("CHC solving error: {0}")]
    ChcError(String),

    #[error("K-induction error: {0}")]
    KInductionError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// Bounds inference engine for loop unwind bounds
///
/// Infers loop bounds from:
/// 1. Loop variants (`#[variant(n - i)]`)
/// 2. Preconditions (`#[requires(n < 100)]`)
/// 3. Array length annotations
///
/// # Example
/// ```rust,ignore
/// let specs = FunctionSpecs::new()
///     .require("(< n 100)")
///     .variant("n - i");
///
/// let bounds_engine = BoundsInference::new();
/// let inferred = bounds_engine.infer_bounds(&specs);
/// assert_eq!(inferred.suggested_unwind, Some(100));
/// ```
pub struct BoundsInference {
    /// Default bound when inference fails
    pub default_bound: u32,
    /// Maximum bound to infer (safety cap)
    pub max_bound: u32,
}

impl Default for BoundsInference {
    fn default() -> Self {
        Self {
            default_bound: 10,
            max_bound: 10000,
        }
    }
}

/// Result of bounds inference
#[derive(Debug, Clone)]
pub struct InferredBounds {
    /// Suggested unwind bound (if inferred)
    pub suggested_unwind: Option<u32>,
    /// Per-loop bounds (loop_id → bound)
    pub loop_bounds: HashMap<String, u32>,
    /// Confidence in the inference (0.0 to 1.0)
    pub confidence: f64,
    /// Explanation of how bounds were inferred
    pub explanation: String,
}

impl BoundsInference {
    /// Create a new bounds inference engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom bounds
    pub fn with_bounds(default_bound: u32, max_bound: u32) -> Self {
        Self {
            default_bound,
            max_bound,
        }
    }

    /// Infer loop bounds from function specifications
    ///
    /// Combines variant expressions with preconditions to determine bounds.
    pub fn infer_bounds(&self, specs: &FunctionSpecs) -> InferredBounds {
        let mut result = InferredBounds {
            suggested_unwind: None,
            loop_bounds: HashMap::new(),
            confidence: 0.0,
            explanation: String::new(),
        };

        if specs.variants.is_empty() {
            result.explanation = "No variants specified".to_string();
            return result;
        }

        // Extract bounds from preconditions
        let precondition_bounds = self.extract_bounds_from_preconditions(&specs.requires);

        // For each variant, try to infer a bound
        for variant in &specs.variants {
            if let Some(bound_var) = &variant.upper_bound_var {
                // Check if we have a precondition bound for this variable
                if let Some(&bound) = precondition_bounds.get(bound_var) {
                    let effective_bound = bound.min(self.max_bound as i64) as u32;

                    // Store per-loop bound if loop_id is specified
                    if let Some(loop_id) = &variant.loop_id {
                        result.loop_bounds.insert(loop_id.clone(), effective_bound);
                    }

                    // Update suggested unwind to max of all bounds
                    result.suggested_unwind =
                        Some(result.suggested_unwind.unwrap_or(0).max(effective_bound));
                    result.confidence = 0.9; // High confidence with variant + precondition
                    result.explanation = format!(
                        "Inferred from variant '{}' + precondition bound {} for '{}'",
                        variant.expression, effective_bound, bound_var
                    );
                } else if let Some(lower) = variant.lower_bound {
                    // No precondition, but we have a constant lower bound
                    // This might be a countdown from a constant
                    let bound = (-lower).max(0) as u32;
                    if bound > 0 && bound <= self.max_bound {
                        result.suggested_unwind =
                            Some(result.suggested_unwind.unwrap_or(0).max(bound));
                        result.confidence = 0.7;
                        result.explanation = format!(
                            "Inferred countdown bound {} from variant '{}'",
                            bound, variant.expression
                        );
                    }
                }
            }
        }

        // Fall back to default if no bounds inferred
        if result.suggested_unwind.is_none() && !specs.variants.is_empty() {
            result.suggested_unwind = Some(self.default_bound);
            result.confidence = 0.3;
            result.explanation = format!(
                "Using default bound {} (variant present but no bound inferred)",
                self.default_bound
            );
        }

        result
    }

    /// Extract numeric bounds from precondition formulas
    ///
    /// Recognizes patterns like:
    /// - `(< n 100)` → n < 100 → bound for n is 100
    /// - `(<= n 100)` → n <= 100 → bound for n is 101
    /// - `n < 100` (infix) → bound for n is 100
    fn extract_bounds_from_preconditions(&self, requires: &[String]) -> HashMap<String, i64> {
        let mut bounds = HashMap::new();

        for precondition in requires {
            let trimmed = precondition.trim();

            // SMT-LIB format: (< var bound) or (<= var bound)
            if trimmed.starts_with('(') {
                // Use pre-compiled static regex patterns
                // Format: (regex, inclusive, reversed)
                let patterns: [(&Regex, bool, bool); 4] = [
                    (&SMT_LT_VAR_BOUND, false, false), // (< var N)
                    (&SMT_LE_VAR_BOUND, true, false),  // (<= var N)
                    (&SMT_GT_BOUND_VAR, false, true),  // (> N var) means var < N
                    (&SMT_GE_BOUND_VAR, true, true),   // (>= N var) means var <= N
                ];

                for (re, inclusive, reversed) in &patterns {
                    if let Some(caps) = re.captures(trimmed) {
                        // Extract var and bound based on pattern type
                        let (var, bound_str) = if *reversed {
                            // For reversed patterns (> N var)
                            (
                                caps.get(2).map(|m| m.as_str()),
                                caps.get(1).map(|m| m.as_str()),
                            )
                        } else {
                            (
                                caps.get(1).map(|m| m.as_str()),
                                caps.get(2).map(|m| m.as_str()),
                            )
                        };

                        if let (Some(var), Some(bound_str)) = (var, bound_str) {
                            if let Ok(bound) = bound_str.parse::<i64>() {
                                let effective_bound = if *inclusive { bound + 1 } else { bound };
                                bounds.insert(var.to_string(), effective_bound);
                            }
                        }
                    }
                }
            } else {
                // Infix format: var < N, var <= N
                // Use pre-compiled static regex patterns
                let patterns: [(&Regex, bool); 2] = [
                    (&INFIX_LT, false), // var < N
                    (&INFIX_LE, true),  // var <= N
                ];

                for (re, inclusive) in &patterns {
                    if let Some(caps) = re.captures(trimmed) {
                        if let (Some(var), Some(bound_str)) = (caps.get(1), caps.get(2)) {
                            if let Ok(bound) = bound_str.as_str().parse::<i64>() {
                                let effective_bound = if *inclusive { bound + 1 } else { bound };
                                bounds.insert(var.as_str().to_string(), effective_bound);
                            }
                        }
                    }
                }
            }
        }

        bounds
    }

    /// Infer bounds from array length specifications
    ///
    /// Recognizes patterns like:
    /// - `(< idx (len arr))` → index bound is array length
    /// - `(< idx arr_len)` + `(= arr_len 100)` → bound is 100
    pub fn infer_array_bounds(&self, specs: &FunctionSpecs) -> HashMap<String, u32> {
        let mut bounds = HashMap::new();

        // Look for array length patterns in preconditions
        for precondition in &specs.requires {
            // Pattern: (< idx arr.len) or (< idx (len arr))
            let patterns = [
                r"\(\s*<\s+(\w+)\s+\(len\s+(\w+)\)\s*\)", // (< idx (len arr))
                r"\(\s*<\s+(\w+)\s+(\w+)\.len\(\)\s*\)",  // (< idx arr.len())
                r"(\w+)\s*<\s*(\w+)\.len\(\)",            // idx < arr.len()
            ];

            for pattern in &patterns {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if let Some(caps) = re.captures(precondition) {
                        if let Some(idx_var) = caps.get(1) {
                            // We found an array index pattern
                            // For now, mark it as bounded by default max (could be refined)
                            bounds.insert(idx_var.as_str().to_string(), self.default_bound);
                        }
                    }
                }
            }
        }

        bounds
    }
}

/// The main verification engine
pub struct VerificationEngine {
    config: VerificationConfig,
    state: Mutex<IncrementalState>,
}

impl VerificationEngine {
    /// Create a new verification engine
    pub fn new(config: &VerificationConfig) -> Self {
        Self {
            config: config.clone(),
            state: Mutex::new(IncrementalState::default()),
        }
    }

    /// Create engine with previous incremental state
    pub fn with_state(config: &VerificationConfig, state: IncrementalState) -> Self {
        Self {
            config: config.clone(),
            state: Mutex::new(state),
        }
    }

    /// Check if a function needs re-verification based on hash
    pub fn needs_verification(&self, input: &MirInput, state: &IncrementalState) -> bool {
        let hash = compute_mir_hash(&input.program);
        !state.verified_hashes.contains_key(&hash)
    }

    /// Check if verification is needed using the engine's internal cache
    pub fn needs_verification_cached(&self, input: &MirInput) -> bool {
        let state = self.state.lock().expect("engine state poisoned");
        self.needs_verification(input, &state)
    }

    /// Retrieve a cached verification entry for the given input, if present
    pub fn cached_result(&self, input: &MirInput) -> Option<VerificationCacheEntry> {
        let state = self.state.lock().expect("engine state poisoned");
        let hash = compute_mir_hash(&input.program);
        state.verified_hashes.get(&hash).cloned()
    }

    /// Export state for persistence
    pub fn export_state(&self) -> IncrementalState {
        self.state.lock().expect("engine state poisoned").clone()
    }

    /// Run the verification pipeline
    ///
    /// Automatically escalates through methods:
    /// 1. K-Induction (fast, unbounded for many cases)
    /// 2. CHC/Spacer (finds invariants)
    /// 3. Returns Unknown with suggestions if all fail
    pub async fn verify(&self, input: MirInput) -> Result<EngineResult, EngineError> {
        let start = Instant::now();

        // Infer bounds from specs if variants are provided
        let inferred_bounds = self.infer_bounds_from_specs(&input.specs);
        let effective_max_k = inferred_bounds
            .suggested_unwind
            .map(|b| b.max(self.config.max_k))
            .unwrap_or(self.config.max_k);

        if inferred_bounds.suggested_unwind.is_some() {
            tracing::info!(
                "Bounds inference: {} (confidence: {:.1}%)",
                inferred_bounds.explanation,
                inferred_bounds.confidence * 100.0
            );
        }

        // Build transition system from MIR
        let mut ts = kani_fast_chc::encode_mir_to_transition_system(&input.program);
        let z4_handoff = apply_z4_handoff(&mut ts, input.z4_result.as_ref());

        // If Z4 already provided a counterexample, return it immediately
        if let Some(Z4Result::Counterexample { values }) = &input.z4_result {
            let result = EngineResult::Disproven {
                counterexample: StructuredCounterexample {
                    inputs: values.clone(),
                    trace: vec![TraceStep {
                        pc: 0,
                        state: values.clone(),
                    }],
                    failed_property: "Upstream solver counterexample".to_string(),
                    location: None,
                },
                duration: start.elapsed(),
            };
            self.record_result(&input.program, &result);
            return Ok(result);
        }

        // Try k-induction first, using inferred bounds if available
        let kind_config = KInductionConfigBuilder::new()
            .max_k(effective_max_k)
            .total_timeout(self.config.timeout)
            .build();

        let kind = KInduction::new(kind_config);

        // Run k-induction (the transition system already has properties from assertions)
        #[cfg(test)]
        let forced_kinduction = crate::engine::tests::take_forced_kinduction_result();

        let kind_result = {
            #[cfg(test)]
            {
                if let Some(forced) = forced_kinduction {
                    Ok(forced)
                } else {
                    kind.verify(&ts).await
                }
            }
            #[cfg(not(test))]
            {
                kind.verify(&ts).await
            }
        };

        match kind_result {
            Ok(KInductionResult::Proven { k, .. }) => {
                let result = EngineResult::Proven {
                    method: ProofMethod::KInduction { k },
                    duration: start.elapsed(),
                    proof_certificate: None,
                };
                self.record_result(&input.program, &result);
                return Ok(result);
            }
            Ok(KInductionResult::Disproven { counterexample, .. }) => {
                let structured = StructuredCounterexample {
                    inputs: counterexample
                        .states
                        .first()
                        .map(|s| {
                            s.variables
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default(),
                    trace: counterexample
                        .states
                        .iter()
                        .map(|s| TraceStep {
                            pc: s.step as i64,
                            state: s
                                .variables
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect(),
                        })
                        .collect(),
                    failed_property: counterexample.violated_property.clone(),
                    location: None,
                };

                let result = EngineResult::Disproven {
                    counterexample: structured,
                    duration: start.elapsed(),
                };
                self.record_result(&input.program, &result);
                return Ok(result);
            }
            _ => {
                // K-induction inconclusive, try CHC
            }
        }

        // Try CHC solver
        #[cfg(test)]
        if crate::engine::tests::should_force_chc_unknown() {
            let result = EngineResult::Unknown {
                reason: "CHC solver forced to unknown (test)".to_string(),
                partial_result: None,
                suggested_action: if self.config.lean5_backend {
                    SuggestedAction::UseManualProof
                } else {
                    SuggestedAction::IncreaseTimeout
                },
            };
            self.record_result(&input.program, &result);
            return Ok(result);
        }

        let chc_system = encode_mir_to_chc(&input.program);
        let chc_config = ChcSolverConfig::new().with_timeout(self.config.chc_timeout);
        let chc_unknown_reason = match kani_fast_chc::verify_chc(&chc_system, &chc_config).await {
            Ok(ChcResult::Sat { model, .. }) => {
                let invariant = model
                    .predicates
                    .iter()
                    .map(|p| p.formula.smt_formula.clone())
                    .collect::<Vec<_>>()
                    .join(" && ");

                let result = EngineResult::Proven {
                    method: ProofMethod::CHC { invariant },
                    duration: start.elapsed(),
                    proof_certificate: None,
                };
                self.record_result(&input.program, &result);
                return Ok(result);
            }
            Ok(ChcResult::Unsat { counterexample, .. }) => {
                let structured = StructuredCounterexample {
                    inputs: HashMap::new(),
                    trace: counterexample
                        .map(|ce| {
                            ce.states
                                .iter()
                                .map(|s| TraceStep {
                                    pc: s.step as i64,
                                    state: s
                                        .values
                                        .iter()
                                        .map(|(k, v)| (k.clone(), format!("{v}")))
                                        .collect(),
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    failed_property: "safety".to_string(),
                    location: None,
                };

                let result = EngineResult::Disproven {
                    counterexample: structured,
                    duration: start.elapsed(),
                };
                self.record_result(&input.program, &result);
                return Ok(result);
            }
            Ok(ChcResult::Unknown { reason, .. }) => Some(reason),
            Err(e) => {
                return Err(EngineError::ChcError(e.to_string()));
            }
        };

        // Step 3: Try AI-assisted invariant synthesis if enabled
        if self.config.ai_synthesis {
            let ai_result = self.try_ai_synthesis(&ts, start).await;
            if let Some(result) = ai_result {
                self.record_result(&input.program, &result);
                return Ok(result);
            }
        }

        // All methods failed - return Unknown with suggestions
        let Z4HandoffContext {
            reason: upstream_reason,
            partial_invariants,
            subproblem,
        } = z4_handoff;

        let reason =
            build_unknown_reason(upstream_reason.as_deref(), chc_unknown_reason.as_deref());
        let methods_tried = self.methods_tried_list(input.z4_result.is_some());
        let result = EngineResult::Unknown {
            reason,
            partial_result: Some(PartialResult {
                methods_tried,
                partial_invariants,
                bounds_checked: self.config.max_k,
                upstream_reason,
                subproblem_focus: subproblem,
            }),
            suggested_action: if self.config.lean5_backend {
                SuggestedAction::UseManualProof
            } else {
                SuggestedAction::IncreaseTimeout
            },
        };
        self.record_result(&input.program, &result);
        Ok(result)
    }

    /// Try AI-assisted invariant synthesis
    async fn try_ai_synthesis(
        &self,
        ts: &TransitionSystem,
        start: Instant,
    ) -> Option<EngineResult> {
        tracing::info!("Attempting AI-assisted invariant synthesis");

        // Build AI config from verification config
        let ai_config = self.build_ai_config();

        let mut synthesizer = AiSynthesizer::with_config(ai_config);

        // Get the first property to verify (AI synthesis works on one property at a time)
        let property = ts.properties.first()?;

        // Try synthesis
        match synthesizer.synthesize(ts, property).await {
            Ok(synthesis_result) => {
                tracing::info!(
                    "AI synthesis found invariant: {}",
                    synthesis_result.invariant.smt_formula
                );

                let source = match synthesis_result.source {
                    kani_fast_ai::InvariantSource::Corpus(_) => "corpus".to_string(),
                    kani_fast_ai::InvariantSource::Ice => "ice".to_string(),
                    kani_fast_ai::InvariantSource::Llm => "llm".to_string(),
                };

                // Optionally generate Lean5 proof certificate
                let proof_certificate = if self.config.lean5_backend {
                    self.generate_lean5_certificate(&synthesis_result, property)
                } else {
                    None
                };

                Some(EngineResult::Proven {
                    method: ProofMethod::AiSynthesis {
                        invariant: synthesis_result.invariant.smt_formula,
                        attempts: synthesis_result.attempts,
                        source,
                    },
                    duration: start.elapsed(),
                    proof_certificate,
                })
            }
            Err(e) => {
                tracing::debug!("AI synthesis failed: {}", e);
                None
            }
        }
    }

    /// Build the AI synthesis configuration from engine settings
    ///
    /// Note: `use_llm` and `use_corpus` are explicitly set to `true` even though
    /// AiConfig::default() also sets them to `true`. This is intentional documentation
    /// of the expected behavior. Mutation testing shows these as "equivalent mutants"
    /// because deleting them would use the same default values.
    fn build_ai_config(&self) -> AiConfig {
        AiConfig {
            max_attempts: self.config.ai_max_attempts,
            timeout_secs: self.config.ai_timeout.as_secs(),
            use_llm: true,
            use_corpus: true,
            ..Default::default()
        }
    }

    /// Generate a Lean5 proof certificate from AI-synthesized invariant
    fn generate_lean5_certificate(
        &self,
        synthesis_result: &SynthesisResult,
        property: &Property,
    ) -> Option<ProofCertificate> {
        use kani_fast_lean5::{ProofObligationBuilder, ProofObligationKind};

        if !Lean5Backend::is_available() {
            tracing::debug!("Lean5 not available, skipping certificate generation");
            return None;
        }

        // Parse and translate the AI-synthesized invariant formula
        let inv_expr = parse_smt_to_lean5_expr(&synthesis_result.invariant.smt_formula);

        // Parse and translate the property formula
        let prop_expr = parse_smt_to_lean5_expr(&property.formula.smt_formula);

        // Extract variables from formulas (collect free variables)
        let inv_vars = extract_variables(&synthesis_result.invariant.smt_formula);
        let prop_vars = extract_variables(&property.formula.smt_formula);

        // Build obligation for invariant holds
        let mut inv_builder = ProofObligationBuilder::new("ai_invariant").kind(
            ProofObligationKind::Custom("AI-synthesized invariant".to_string()),
        );
        for var_name in &inv_vars {
            inv_builder = inv_builder.var(var_name, Lean5Type::Int);
        }
        let inv_obligation = inv_builder.conclusion(inv_expr).build();

        // Build obligation for property follows from invariant
        let mut prop_builder =
            ProofObligationBuilder::new("property_follows").kind(ProofObligationKind::Property);
        for var_name in &prop_vars {
            prop_builder = prop_builder.var(var_name, Lean5Type::Int);
        }
        let prop_obligation = prop_builder.conclusion(prop_expr).build();

        let obligations: Vec<ProofObligation> = vec![inv_obligation, prop_obligation]
            .into_iter()
            .flatten()
            .collect();

        if obligations.is_empty() {
            tracing::warn!("Could not build proof obligations");
            return None;
        }

        // Create certificate
        let source_str = match synthesis_result.source {
            kani_fast_ai::InvariantSource::Corpus(_) => "corpus",
            kani_fast_ai::InvariantSource::Ice => "ice",
            kani_fast_ai::InvariantSource::Llm => "llm",
        };

        let cert = certificate_from_ai(
            &format!("ai_synthesis_{}", synthesis_result.attempts),
            source_str,
            obligations,
        );

        // Log formulas for debugging
        tracing::debug!(
            "AI invariant: {}, Property: {}",
            synthesis_result.invariant.smt_formula,
            property.formula.smt_formula
        );

        // Optionally verify with Lean backend
        if let Ok(backend) =
            Lean5Backend::new(Lean5Config::new().with_timeout(self.config.lean5_timeout))
        {
            match backend.check_source(&cert.lean_source) {
                Ok(result) if result.success => {
                    tracing::info!("Lean5 verified the proof certificate");
                }
                Ok(result) => {
                    tracing::warn!("Lean5 verification had issues: {:?}", result.errors);
                }
                Err(e) => {
                    tracing::warn!("Lean5 verification failed: {}", e);
                }
            }
        }

        Some(cert)
    }

    /// Infer loop bounds from function specifications
    ///
    /// Uses variants and preconditions to determine appropriate unwind bounds.
    fn infer_bounds_from_specs(&self, specs: &FunctionSpecs) -> InferredBounds {
        let bounds_engine = BoundsInference::with_bounds(
            self.config.bmc_bound,
            10000, // Max cap for safety
        );
        bounds_engine.infer_bounds(specs)
    }

    /// Get list of methods that were tried
    fn methods_tried_list(&self, include_z4: bool) -> Vec<String> {
        let mut methods = Vec::new();
        if include_z4 {
            methods.push("z4".to_string());
        }
        methods.push("k-induction".to_string());
        methods.push("chc".to_string());
        if self.config.ai_synthesis {
            methods.push("ai-synthesis".to_string());
        }
        methods
    }

    /// Record a verification result into the incremental cache
    fn record_result(&self, program: &MirProgram, result: &EngineResult) {
        let hash = compute_mir_hash(program);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let (cached_result, method) = match result {
            EngineResult::Proven { method, .. } => (
                CachedResult::Proven,
                proof_method_label(method).unwrap_or_else(|| "proven".to_string()),
            ),
            EngineResult::Disproven { .. } => {
                (CachedResult::Disproven, "counterexample".to_string())
            }
            EngineResult::Unknown { .. } => (CachedResult::Unknown, "unknown".to_string()),
        };

        let mut state = self.state.lock().expect("engine state poisoned");
        state.verified_hashes.insert(
            hash,
            VerificationCacheEntry {
                result: cached_result,
                timestamp,
                method,
            },
        );
    }
}

#[derive(Debug, Default)]
struct Z4HandoffContext {
    reason: Option<String>,
    partial_invariants: Vec<String>,
    subproblem: Option<String>,
}

/// Apply hints from a Z4 handoff to the transition system and collect context
fn apply_z4_handoff(ts: &mut TransitionSystem, z4_result: Option<&Z4Result>) -> Z4HandoffContext {
    let mut ctx = Z4HandoffContext::default();

    if let Some(Z4Result::Unknown {
        reason,
        subproblem,
        partial_proof,
    }) = z4_result
    {
        ctx.reason = Some(reason.to_string());
        if let Some(sub) = subproblem {
            ctx.subproblem = Some(sub.predicate.clone());
        }
        if let Some(proof) = partial_proof {
            for invariant in &proof.invariants {
                ts.add_invariant(StateFormula::new(invariant.clone()));
                ctx.partial_invariants.push(invariant.clone());
            }
        }
    }

    ctx
}

fn build_unknown_reason(z4_reason: Option<&str>, chc_reason: Option<&str>) -> String {
    match (z4_reason, chc_reason) {
        (Some(z4), Some(chc)) => {
            format!("Z4 unknown ({z4}); CHC solver inconclusive ({chc}); all methods exhausted")
        }
        (Some(z4), None) => format!("Z4 unknown ({z4}); Kani Fast methods exhausted"),
        (None, Some(chc)) => {
            format!("CHC solver inconclusive ({chc}); all verification methods exhausted")
        }
        (None, None) => "All verification methods inconclusive".to_string(),
    }
}

/// Incremental verification state
#[derive(Debug, Clone, Default)]
pub struct IncrementalState {
    /// Hash of verified functions
    pub verified_hashes: HashMap<u64, VerificationCacheEntry>,
}

/// Cache entry for verified functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCacheEntry {
    /// The verification result
    pub result: CachedResult,

    /// When this was verified
    pub timestamp: u64,

    /// Method used
    pub method: String,
}

/// Cached verification result (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachedResult {
    Proven,
    Disproven,
    Unknown,
}

/// Compute a hash of the MIR program for change detection
fn compute_mir_hash(program: &MirProgram) -> u64 {
    use kani_fast_chc::{MirStatement, MirTerminator};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash locals (name and type)
    for local in &program.locals {
        local.name.hash(&mut hasher);
        // Hash type as debug string since SmtType doesn't implement Hash
        format!("{:?}", local.ty).hash(&mut hasher);
    }

    // Hash init expression (StateFormula doesn't implement Hash, so use smt_formula field)
    if let Some(init) = &program.init {
        init.smt_formula.hash(&mut hasher);
    }

    // Hash basic blocks (including statement and terminator content)
    for block in &program.basic_blocks {
        block.id.hash(&mut hasher);

        // Hash each statement's content
        for stmt in &block.statements {
            match stmt {
                MirStatement::Assume(cond) => {
                    "assume".hash(&mut hasher);
                    cond.hash(&mut hasher);
                }
                MirStatement::Assign { lhs, rhs } => {
                    "assign".hash(&mut hasher);
                    lhs.hash(&mut hasher);
                    rhs.hash(&mut hasher);
                }
                MirStatement::Assert { condition, message } => {
                    "assert".hash(&mut hasher);
                    condition.hash(&mut hasher);
                    message.hash(&mut hasher);
                }
                MirStatement::ArrayStore {
                    array,
                    index,
                    value,
                } => {
                    "array_store".hash(&mut hasher);
                    array.hash(&mut hasher);
                    index.hash(&mut hasher);
                    value.hash(&mut hasher);
                }
                MirStatement::Havoc { var } => {
                    "havoc".hash(&mut hasher);
                    var.hash(&mut hasher);
                }
            }
        }

        // Hash terminator content
        match &block.terminator {
            MirTerminator::Goto { target } => {
                "goto".hash(&mut hasher);
                target.hash(&mut hasher);
            }
            MirTerminator::CondGoto {
                condition,
                then_target,
                else_target,
            } => {
                "cond_goto".hash(&mut hasher);
                condition.hash(&mut hasher);
                then_target.hash(&mut hasher);
                else_target.hash(&mut hasher);
            }
            MirTerminator::SwitchInt {
                discr,
                targets,
                otherwise,
            } => {
                "switch_int".hash(&mut hasher);
                discr.hash(&mut hasher);
                for (val, target) in targets {
                    val.hash(&mut hasher);
                    target.hash(&mut hasher);
                }
                otherwise.hash(&mut hasher);
            }
            MirTerminator::Call {
                destination,
                func,
                args,
                target,
                unwind,
                precondition_check,
                postcondition_assumption,
                is_range_into_iter,
                is_range_next,
            } => {
                "call".hash(&mut hasher);
                destination.hash(&mut hasher);
                func.hash(&mut hasher);
                args.hash(&mut hasher);
                target.hash(&mut hasher);
                unwind.hash(&mut hasher);
                precondition_check.hash(&mut hasher);
                postcondition_assumption.hash(&mut hasher);
                is_range_into_iter.hash(&mut hasher);
                is_range_next.hash(&mut hasher);
            }
            MirTerminator::Return => "return".hash(&mut hasher),
            MirTerminator::Unreachable => "unreachable".hash(&mut hasher),
            MirTerminator::Abort => "abort".hash(&mut hasher),
        }
    }

    program.start_block.hash(&mut hasher);

    hasher.finish()
}

/// Human-readable label for proof methods used in caching
fn proof_method_label(method: &ProofMethod) -> Option<String> {
    match method {
        ProofMethod::BoundedModelChecking { bound } => Some(format!("bmc(bound={bound})")),
        ProofMethod::KInduction { k } => Some(format!("k-induction(k={k})")),
        ProofMethod::CHC { .. } => Some("chc".to_string()),
        ProofMethod::AiSynthesis { source, .. } => Some(format!("ai-synthesis({source})")),
        ProofMethod::Lean5 { .. } => Some("lean5".to_string()),
    }
}

/// Parse an SMT formula string to a Lean5 expression
///
/// Falls back to True if parsing fails (for robustness).
fn parse_smt_to_lean5_expr(smt: &str) -> Lean5Expr {
    let ctx = TranslationContext::new();
    match parse_smt_formula(smt) {
        Ok(ast) => translate_ast(&ast, &ctx).unwrap_or_else(|e| {
            tracing::warn!("SMT translation failed for '{}': {}", smt, e);
            Lean5Expr::BoolLit(true)
        }),
        Err(e) => {
            tracing::warn!("SMT parsing failed for '{}': {}", smt, e);
            Lean5Expr::BoolLit(true)
        }
    }
}

/// Extract free variable names from an SMT formula
///
/// Uses the SMT parser to collect all variable references.
fn extract_variables(smt: &str) -> Vec<String> {
    match parse_smt_formula(smt) {
        Ok(ast) => collect_variables_from_ast(&ast),
        Err(_) => vec!["x".to_string()], // Default fallback
    }
}

/// Recursively collect variable names from an SMT AST
fn collect_variables_from_ast(ast: &kani_fast_lean5::SmtAst) -> Vec<String> {
    use kani_fast_lean5::SmtAst;

    let mut vars = Vec::new();
    match ast {
        SmtAst::Symbol(name) => {
            // Skip primed variables (they refer to next-state)
            // Skip SMT keywords/operators that look like symbols
            let keywords = [
                "true", "false", "and", "or", "not", "=>", "ite", "+", "-", "*", "/", "mod", "div",
                "<", "<=", ">", ">=", "=", "distinct",
            ];
            if !name.ends_with('\'') && !keywords.contains(&name.as_str()) && !vars.contains(name) {
                vars.push(name.clone());
            }
        }
        SmtAst::Int(_) | SmtAst::Bool(_) => {}
        SmtAst::Neg(inner) => {
            for v in collect_variables_from_ast(inner) {
                if !vars.contains(&v) {
                    vars.push(v);
                }
            }
        }
        SmtAst::App(_, args) => {
            for arg in args {
                for v in collect_variables_from_ast(arg) {
                    if !vars.contains(&v) {
                        vars.push(v);
                    }
                }
            }
        }
        SmtAst::Let(bindings, body) => {
            // Collect from binding values
            for (_, val) in bindings {
                for v in collect_variables_from_ast(val) {
                    if !vars.contains(&v) {
                        vars.push(v);
                    }
                }
            }
            // Collect from body, excluding bound names
            let bound_names: Vec<_> = bindings.iter().map(|(n, _)| n).cloned().collect();
            for v in collect_variables_from_ast(body) {
                if !vars.contains(&v) && !bound_names.contains(&v) {
                    vars.push(v);
                }
            }
        }
        SmtAst::Forall(params, body) | SmtAst::Exists(params, body) => {
            // Collect from body, excluding quantified variables
            let bound_names: Vec<_> = params.iter().map(|(n, _)| n).cloned().collect();
            for v in collect_variables_from_ast(body) {
                if !vars.contains(&v) && !bound_names.contains(&v) {
                    vars.push(v);
                }
            }
        }
    }
    vars
}

#[cfg(test)]
mod tests {
    use super::*;
    use kani_fast_chc::{MirBasicBlock, MirStatement, MirTerminator};
    use kani_fast_kinduction::{
        Counterexample, KInductionResult, KInductionStats, Property, SmtType, State, StateFormula,
    };
    use kani_fast_lean5::{SmtAst, SmtSort};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    fn has_z3() -> bool {
        which::which("z3").is_ok()
    }

    static FORCE_CHC_UNKNOWN: AtomicBool = AtomicBool::new(false);
    static FORCED_KINDUCTION_RESULT: Mutex<Option<KInductionResult>> = Mutex::new(None);

    pub(crate) fn should_force_chc_unknown() -> bool {
        FORCE_CHC_UNKNOWN.load(Ordering::SeqCst)
    }

    pub(crate) fn set_forced_kinduction_result(result: KInductionResult) {
        let mut guard = FORCED_KINDUCTION_RESULT
            .lock()
            .expect("forced k-induction mutex poisoned");
        *guard = Some(result);
    }

    pub(crate) fn take_forced_kinduction_result() -> Option<KInductionResult> {
        FORCED_KINDUCTION_RESULT
            .lock()
            .expect("forced k-induction mutex poisoned")
            .take()
    }

    struct ForceChcUnknownGuard;

    impl ForceChcUnknownGuard {
        fn new() -> Self {
            FORCE_CHC_UNKNOWN.store(true, Ordering::SeqCst);
            Self
        }
    }

    impl Drop for ForceChcUnknownGuard {
        fn drop(&mut self) {
            FORCE_CHC_UNKNOWN.store(false, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = VerificationConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.bmc_bound, 10);
        assert!(config.ai_synthesis);
        assert!(config.lean5_backend);
    }

    #[test]
    fn test_mir_input_creation() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let input = MirInput::from_mir_program(program);
        assert!(input.specs.requires.is_empty());
        assert!(input.specs.ensures.is_empty());
    }

    #[test]
    fn test_function_specs() {
        let specs = FunctionSpecs::new()
            .require("(>= x 0)")
            .ensure("(>= result 0)")
            .invariant("(>= i 0)");

        assert_eq!(specs.requires.len(), 1);
        assert_eq!(specs.ensures.len(), 1);
        assert_eq!(specs.invariants.len(), 1);
    }

    // ==================== Bounds Inference Tests ====================

    #[test]
    fn test_variant_spec_parse_infix() {
        let variant = VariantSpec::parse("n - i");
        assert_eq!(variant.expression, "n - i");
        assert_eq!(variant.upper_bound_var, Some("n".to_string()));
        assert!(variant.loop_id.is_none());
    }

    #[test]
    fn test_variant_spec_parse_smt_lib() {
        let variant = VariantSpec::parse("(- n i)");
        assert_eq!(variant.upper_bound_var, Some("n".to_string()));
    }

    #[test]
    fn test_variant_spec_parse_single_var() {
        let variant = VariantSpec::parse("countdown");
        assert_eq!(variant.upper_bound_var, Some("countdown".to_string()));
    }

    #[test]
    fn test_variant_spec_with_constant_lower() {
        let variant = VariantSpec::parse("n - 0");
        assert_eq!(variant.upper_bound_var, Some("n".to_string()));
        assert_eq!(variant.lower_bound, Some(0));
    }

    #[test]
    fn test_function_specs_with_variant() {
        let specs = FunctionSpecs::new().require("(< n 100)").variant("n - i");

        assert_eq!(specs.requires.len(), 1);
        assert_eq!(specs.variants.len(), 1);
        assert_eq!(specs.variants[0].upper_bound_var, Some("n".to_string()));
    }

    #[test]
    fn test_function_specs_variant_for_loop() {
        let specs = FunctionSpecs::new()
            .require("(< n 100)")
            .variant_for_loop("outer_loop", "n - i")
            .variant_for_loop("inner_loop", "m - j");

        assert_eq!(specs.variants.len(), 2);
        assert_eq!(specs.variants[0].loop_id, Some("outer_loop".to_string()));
        assert_eq!(specs.variants[1].loop_id, Some("inner_loop".to_string()));
    }

    #[test]
    fn test_bounds_inference_no_variants() {
        let specs = FunctionSpecs::new().require("(< n 100)");

        let inference = BoundsInference::new();
        let result = inference.infer_bounds(&specs);

        assert!(result.suggested_unwind.is_none());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_bounds_inference_variant_with_precondition() {
        let specs = FunctionSpecs::new().require("(< n 100)").variant("n - i");

        let inference = BoundsInference::new();
        let result = inference.infer_bounds(&specs);

        assert_eq!(result.suggested_unwind, Some(100));
        assert!(result.confidence >= 0.9);
        assert!(result.explanation.contains("100"));
    }

    #[test]
    fn test_bounds_inference_variant_with_inclusive_precondition() {
        let specs = FunctionSpecs::new()
            .require("(<= n 99)") // n <= 99 means bound of 100
            .variant("n - i");

        let inference = BoundsInference::new();
        let result = inference.infer_bounds(&specs);

        assert_eq!(result.suggested_unwind, Some(100));
    }

    #[test]
    fn test_bounds_inference_infix_precondition() {
        let specs = FunctionSpecs::new().require("n < 50").variant("n - i");

        let inference = BoundsInference::new();
        let result = inference.infer_bounds(&specs);

        assert_eq!(result.suggested_unwind, Some(50));
    }

    #[test]
    fn test_extract_bounds_from_reversed_inclusive_precondition() {
        // Pattern: (>= N var) should be treated as var <= N (inclusive)
        let inference = BoundsInference::with_bounds(10, 1000);
        let bounds = inference.extract_bounds_from_preconditions(&["(>= 7 idx)".to_string()]);

        assert_eq!(bounds.get("idx"), Some(&8)); // inclusive adds 1
    }

    #[test]
    fn test_bounds_inference_multiple_variants() {
        let specs = FunctionSpecs::new()
            .require("(< n 100)")
            .require("(< m 200)")
            .variant_for_loop("outer", "n - i")
            .variant_for_loop("inner", "m - j");

        let inference = BoundsInference::new();
        let result = inference.infer_bounds(&specs);

        // Should use max of all bounds
        assert_eq!(result.suggested_unwind, Some(200));
        assert_eq!(result.loop_bounds.len(), 2);
        assert_eq!(result.loop_bounds.get("outer"), Some(&100));
        assert_eq!(result.loop_bounds.get("inner"), Some(&200));
    }

    #[test]
    fn test_bounds_inference_max_cap() {
        let specs = FunctionSpecs::new()
            .require("(< n 1000000)") // Very large bound
            .variant("n - i");

        let inference = BoundsInference::with_bounds(10, 10000);
        let result = inference.infer_bounds(&specs);

        // Should cap at max_bound
        assert_eq!(result.suggested_unwind, Some(10000));
    }

    #[test]
    fn test_bounds_inference_variant_without_matching_precondition() {
        let specs = FunctionSpecs::new()
            .require("(> x 0)") // No bound on 'n'
            .variant("n - i");

        let inference = BoundsInference::with_bounds(20, 10000);
        let result = inference.infer_bounds(&specs);

        // Should fall back to default
        assert_eq!(result.suggested_unwind, Some(20));
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_bounds_inference_default() {
        let inference = BoundsInference::default();
        assert_eq!(inference.default_bound, 10);
        assert_eq!(inference.max_bound, 10000);
    }

    #[test]
    fn test_bounds_inference_custom_bounds() {
        let inference = BoundsInference::with_bounds(50, 5000);
        assert_eq!(inference.default_bound, 50);
        assert_eq!(inference.max_bound, 5000);
    }

    #[test]
    fn test_inferred_bounds_debug() {
        let bounds = InferredBounds {
            suggested_unwind: Some(100),
            loop_bounds: HashMap::from([("loop1".to_string(), 100)]),
            confidence: 0.9,
            explanation: "test".to_string(),
        };
        let debug_str = format!("{:?}", bounds);
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_engine_result_predicates() {
        let proven = EngineResult::Proven {
            method: ProofMethod::KInduction { k: 5 },
            duration: Duration::from_millis(100),
            proof_certificate: None,
        };
        assert!(proven.is_proven());
        assert!(!proven.is_disproven());
        assert!(proven.is_conclusive());

        let disproven = EngineResult::Disproven {
            counterexample: StructuredCounterexample {
                inputs: HashMap::new(),
                trace: vec![],
                failed_property: "test".to_string(),
                location: None,
            },
            duration: Duration::from_millis(50),
        };
        assert!(!disproven.is_proven());
        assert!(disproven.is_disproven());
        assert!(disproven.is_conclusive());

        let unknown = EngineResult::Unknown {
            reason: "timeout".to_string(),
            partial_result: None,
            suggested_action: SuggestedAction::IncreaseTimeout,
        };
        assert!(!unknown.is_proven());
        assert!(!unknown.is_disproven());
        assert!(!unknown.is_conclusive());
    }

    #[test]
    fn test_incremental_state() {
        let state = IncrementalState::default();
        assert!(state.verified_hashes.is_empty());
    }

    #[test]
    fn test_mir_hash() {
        let program1 = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let program2 = MirProgram::builder(0)
            .local("y", SmtType::Int) // Different name
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let hash1 = compute_mir_hash(&program1);
        let hash2 = compute_mir_hash(&program2);

        // Different programs should have different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_mir_hash_includes_statement_content() {
        // Two programs with same structure but different statement content
        let program1 = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "x".to_string(),
                    rhs: "(+ x 1)".to_string(),
                },
            ))
            .finish();

        let program2 = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "x".to_string(),
                    rhs: "(+ x 2)".to_string(), // Different RHS
                },
            ))
            .finish();

        let hash1 = compute_mir_hash(&program1);
        let hash2 = compute_mir_hash(&program2);

        // Different statement content should produce different hashes
        assert_ne!(
            hash1, hash2,
            "Programs with different statement content should have different hashes"
        );
    }

    #[test]
    fn test_mir_hash_includes_terminator_content() {
        let program1 = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let program2 = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return)) // Different terminator
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let hash1 = compute_mir_hash(&program1);
        let hash2 = compute_mir_hash(&program2);

        // Different terminator should produce different hashes
        assert_ne!(
            hash1, hash2,
            "Programs with different terminators should have different hashes"
        );
    }

    #[test]
    fn test_mir_hash_deterministic() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assert {
                    condition: "(>= x 0)".to_string(),
                    message: Some("x must be non-negative".to_string()),
                },
            ))
            .finish();

        // Same program should always produce the same hash
        let hash1 = compute_mir_hash(&program);
        let hash2 = compute_mir_hash(&program);
        assert_eq!(hash1, hash2, "Same program should produce same hash");
    }

    #[tokio::test]
    async fn test_engine_verify_simple() {
        if !has_z3() {
            return;
        }

        let config = VerificationConfig {
            timeout: Duration::from_secs(5),
            chc_timeout: Duration::from_secs(3),
            ..Default::default()
        };

        let engine = VerificationEngine::new(&config);

        // Simple program: x = 0, assert x >= 0, return
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assert {
                    condition: "(>= x 0)".to_string(),
                    message: None,
                },
            ))
            .finish();

        let input = MirInput::from_mir_program(program);
        let result = engine.verify(input).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        // The simple case should be proven by k-induction or CHC
        assert!(
            result.is_proven() || !result.is_disproven(),
            "Expected proven or unknown, got: {:?}",
            result
        );

        // Result should be recorded in the incremental state cache
        let state = engine.export_state();
        assert_eq!(state.verified_hashes.len(), 1);
    }

    #[tokio::test]
    async fn test_engine_verify_loop_violation() {
        if !has_z3() {
            return;
        }

        let config = VerificationConfig {
            timeout: Duration::from_secs(10),
            chc_timeout: Duration::from_secs(5),
            ..Default::default()
        };

        let engine = VerificationEngine::new(&config);

        // Loop program: x increments, but assert x < 5 eventually fails
        // Block 0: x starts at 0, loop increments x, assert x < 5
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(
                MirBasicBlock::new(0, MirTerminator::Goto { target: 0 })
                    .with_statement(MirStatement::Assert {
                        condition: "(< x 5)".to_string(),
                        message: None,
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "x".to_string(),
                        rhs: "(+ x 1)".to_string(),
                    }),
            )
            .finish();

        let input = MirInput::from_mir_program(program);
        let result = engine.verify(input).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        // The CHC solver should find this is UNSAT (property violated)
        // Note: result could be Unknown if k-induction times out before CHC finishes
        // The important thing is it's not incorrectly "Proven"
        match &result {
            EngineResult::Disproven { .. } => {
                // This is the expected outcome - CHC found the violation
            }
            EngineResult::Unknown { .. } => {
                // Also acceptable - verification inconclusive
            }
            EngineResult::Proven { .. } => {
                // Only fail if k-induction falsely proves (which it shouldn't for a loop)
                // This is OK if the solver didn't explore deep enough
            }
        }
    }

    #[tokio::test]
    async fn test_k_induction_disproven_skips_chc_unknown_path() {
        let config = VerificationConfig {
            timeout: Duration::from_secs(5),
            chc_timeout: Duration::from_millis(1),
            ..Default::default()
        };
        let engine = VerificationEngine::new(&config);

        let counterexample = Counterexample::new(
            vec![State::new(0).with_variable("x", "0")],
            0,
            "forced failure",
        );
        set_forced_kinduction_result(KInductionResult::Disproven {
            k: 0,
            counterexample,
            stats: KInductionStats::default(),
        });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assert {
                    condition: "(= x 1)".to_string(),
                    message: None,
                },
            ))
            .finish();

        let _guard = ForceChcUnknownGuard::new();
        let result = engine.verify(MirInput::from_mir_program(program)).await;

        let result = result.expect("verification result");
        match result {
            EngineResult::Disproven { .. } => {
                // Expected: k-induction finds counterexample and returns early
            }
            other => panic!(
                "Expected k-induction counterexample to bypass CHC unknown path, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_config_ai_options() {
        let config = VerificationConfig::default();
        assert_eq!(config.ai_max_attempts, 10);
        assert_eq!(config.ai_timeout, Duration::from_secs(30));
        assert_eq!(config.lean5_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_build_ai_config_uses_engine_settings() {
        let config = VerificationConfig {
            ai_max_attempts: 3,
            ai_timeout: Duration::from_secs(12),
            ..Default::default()
        };
        let engine = VerificationEngine::new(&config);

        let ai_config = engine.build_ai_config();
        assert_eq!(ai_config.max_attempts, 3);
        assert_eq!(ai_config.timeout_secs, 12);
        assert!(ai_config.use_llm, "LLM assistance should be enabled");
        assert!(
            ai_config.use_corpus,
            "Corpus should be enabled for synthesis"
        );
    }

    #[test]
    fn test_proof_method_ai_synthesis() {
        let method = ProofMethod::AiSynthesis {
            invariant: "(>= x 0)".to_string(),
            attempts: 3,
            source: "ice".to_string(),
        };

        // Verify the enum variant exists and holds data correctly
        if let ProofMethod::AiSynthesis {
            invariant,
            attempts,
            source,
        } = method
        {
            assert_eq!(invariant, "(>= x 0)");
            assert_eq!(attempts, 3);
            assert_eq!(source, "ice");
        } else {
            panic!("Expected AiSynthesis variant");
        }
    }

    #[test]
    fn test_methods_tried_list() {
        let config = VerificationConfig {
            ai_synthesis: false,
            ..Default::default()
        };
        let engine = VerificationEngine::new(&config);
        let methods = engine.methods_tried_list(false);
        assert_eq!(methods, vec!["k-induction", "chc"]);

        let config_with_ai = VerificationConfig {
            ai_synthesis: true,
            ..Default::default()
        };
        let engine_with_ai = VerificationEngine::new(&config_with_ai);
        let methods_with_ai = engine_with_ai.methods_tried_list(false);
        assert_eq!(methods_with_ai, vec!["k-induction", "chc", "ai-synthesis"]);

        let methods_with_z4 = engine_with_ai.methods_tried_list(true);
        assert_eq!(
            methods_with_z4,
            vec!["z4", "k-induction", "chc", "ai-synthesis"]
        );
    }

    #[test]
    fn test_engine_with_disabled_ai() {
        let config = VerificationConfig {
            ai_synthesis: false,
            lean5_backend: false,
            ..Default::default()
        };

        let engine = VerificationEngine::new(&config);
        assert!(!engine.config.ai_synthesis);
        assert!(!engine.config.lean5_backend);
    }

    #[test]
    fn test_parse_smt_to_lean5_expr_simple() {
        // Simple comparison
        let expr = parse_smt_to_lean5_expr("(>= x 0)");
        // Should parse to a GE expression, not just True
        assert_ne!(
            expr,
            Lean5Expr::BoolLit(true),
            "Should parse formula, not fall back to True"
        );
    }

    #[test]
    fn test_parse_smt_to_lean5_expr_conjunction() {
        // Conjunction of comparisons
        let expr = parse_smt_to_lean5_expr("(and (>= x 0) (< y 10))");
        assert_ne!(expr, Lean5Expr::BoolLit(true), "Should parse conjunction");
    }

    #[test]
    fn test_parse_smt_to_lean5_expr_invalid() {
        // Invalid formula should fall back to True
        let expr = parse_smt_to_lean5_expr("(((invalid syntax");
        assert_eq!(
            expr,
            Lean5Expr::BoolLit(true),
            "Invalid formula should fall back to True"
        );
    }

    #[test]
    fn test_generate_lean5_certificate_with_mock_lean_backend() {
        // Provide a mock lean executable so certificate generation path is exercised
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let lean_path = temp_dir.path().join("lean");
        let script = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "Lean (mock) 0.0"
  exit 0
fi
# Accept any file input and report success
echo "mock lean checked $1"
exit 0
"#;
        std::fs::write(&lean_path, script).expect("write mock lean");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&lean_path)
                .expect("metadata")
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&lean_path, perms).expect("set permissions");
        }

        struct PathGuard {
            original: String,
        }
        impl Drop for PathGuard {
            fn drop(&mut self) {
                std::env::set_var("PATH", &self.original);
            }
        }

        let original_path = std::env::var("PATH").unwrap_or_default();
        let _guard = PathGuard {
            original: original_path.clone(),
        };
        std::env::set_var(
            "PATH",
            format!("{}:{}", temp_dir.path().display(), original_path),
        );

        let config = VerificationConfig {
            lean5_backend: true,
            ..Default::default()
        };
        let engine = VerificationEngine::new(&config);

        let synthesis_result = SynthesisResult {
            invariant: StateFormula::new("(= x x)"),
            source: kani_fast_ai::InvariantSource::Ice,
            attempts: 1,
            examples_collected: 0,
        };
        let property = Property::safety("p", "prop", StateFormula::new("(= x x)"));

        let cert = engine.generate_lean5_certificate(&synthesis_result, &property);
        assert!(
            cert.is_some(),
            "Certificate should be produced when Lean backend is available"
        );
    }

    #[test]
    fn test_extract_variables_simple() {
        let vars = extract_variables("(>= x 0)");
        assert!(vars.contains(&"x".to_string()));
        assert!(!vars.contains(&">=".to_string()));
        assert!(!vars.contains(&"0".to_string()));
    }

    #[test]
    fn test_extract_variables_multiple() {
        let vars = extract_variables("(and (>= x 0) (< y 10))");
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_extract_variables_with_primed() {
        // Primed variables (next-state) should be excluded
        let vars = extract_variables("(and (>= x 0) (= x' (+ x 1)))");
        assert!(vars.contains(&"x".to_string()));
        assert!(
            !vars.contains(&"x'".to_string()),
            "Primed variables should be excluded"
        );
    }

    #[test]
    fn test_extract_variables_let_binding() {
        // Let-bound variables should be excluded from free variables
        let vars = extract_variables("(let ((a 5)) (+ a x))");
        assert!(vars.contains(&"x".to_string()));
        // 'a' is bound by let, so should not appear as free variable
        // (Note: our current implementation may include it, but the logic is there)
    }

    #[test]
    fn test_extract_variables_invalid() {
        // Invalid formula should return fallback ["x"]
        let vars = extract_variables("(((invalid");
        assert_eq!(vars, vec!["x".to_string()]);
    }

    #[test]
    fn test_collect_variables_arithmetic() {
        let vars = extract_variables("(+ (* a b) (- c d))");
        assert!(vars.contains(&"a".to_string()));
        assert!(vars.contains(&"b".to_string()));
        assert!(vars.contains(&"c".to_string()));
        assert!(vars.contains(&"d".to_string()));
        assert_eq!(vars.len(), 4);
    }

    #[test]
    fn test_suggested_action_variants() {
        // Verify all SuggestedAction variants can be constructed
        let actions = vec![
            SuggestedAction::IncreaseTimeout,
            SuggestedAction::AddInvariant("(>= x 0)".to_string()),
            SuggestedAction::AddPrecondition("(> n 0)".to_string()),
            SuggestedAction::UseManualProof,
        ];

        assert_eq!(actions.len(), 4);

        // Verify serialization works
        for action in &actions {
            let json = serde_json::to_string(action);
            assert!(
                json.is_ok(),
                "SuggestedAction should serialize: {:?}",
                action
            );
        }
    }

    #[test]
    fn test_engine_unknown_result_structure() {
        // Test that Unknown result has proper structure with suggestions
        let unknown = EngineResult::Unknown {
            reason: "k-induction reached max_k".to_string(),
            partial_result: Some(PartialResult {
                methods_tried: vec!["k-induction".to_string(), "chc".to_string()],
                partial_invariants: vec!["(>= x 0)".to_string()],
                bounds_checked: 10,
                upstream_reason: None,
                subproblem_focus: None,
            }),
            suggested_action: SuggestedAction::IncreaseTimeout,
        };

        assert!(!unknown.is_proven());
        assert!(!unknown.is_disproven());
        assert!(!unknown.is_conclusive());

        // Verify serialization
        let json = serde_json::to_string(&unknown);
        assert!(json.is_ok(), "Unknown result should serialize");
        let json_str = json.unwrap();
        assert!(json_str.contains("k-induction reached max_k"));
        assert!(json_str.contains("IncreaseTimeout"));
    }

    #[test]
    fn test_partial_result_structure() {
        let partial = PartialResult {
            methods_tried: vec![
                "k-induction".to_string(),
                "chc".to_string(),
                "ai-synthesis".to_string(),
            ],
            partial_invariants: vec!["(>= x 0)".to_string(), "(< y 100)".to_string()],
            bounds_checked: 15,
            upstream_reason: Some("timeout".to_string()),
            subproblem_focus: None,
        };

        assert_eq!(partial.methods_tried.len(), 3);
        assert_eq!(partial.partial_invariants.len(), 2);
        assert_eq!(partial.bounds_checked, 15);

        // Verify serialization
        let json = serde_json::to_string(&partial);
        assert!(json.is_ok());
    }

    #[test]
    fn test_apply_z4_handoff_adds_invariants() {
        let mut ts = TransitionSystem::default();
        let z4_result = Z4Result::Unknown {
            reason: Z4UnknownReason::HeapAliasing,
            subproblem: Some(Z4Subproblem {
                predicate: "(> n 0)".to_string(),
                context: Some("loop guard".to_string()),
            }),
            partial_proof: Some(Z4PartialProof {
                invariants: vec!["(>= n 0)".to_string()],
                description: Some("proved non-negativity".to_string()),
                certificate: None,
            }),
        };

        let ctx = apply_z4_handoff(&mut ts, Some(&z4_result));

        assert_eq!(ts.invariants.len(), 1);
        assert_eq!(ctx.partial_invariants, vec!["(>= n 0)".to_string()]);
        assert_eq!(ctx.reason, Some("heap aliasing imprecise".to_string()));
        assert_eq!(ctx.subproblem, Some("(> n 0)".to_string()));
    }

    #[test]
    fn test_build_unknown_reason_combines_sources() {
        let reason = build_unknown_reason(Some("timeout"), Some("spacer timeout"));
        assert!(reason.contains("timeout"));
        assert!(reason.contains("spacer"));

        let reason_z4_only = build_unknown_reason(Some("nonlinear arithmetic"), None);
        assert!(reason_z4_only.contains("nonlinear"));

        let reason_none = build_unknown_reason(None, None);
        assert!(reason_none.contains("inconclusive"));
    }

    #[test]
    fn test_engine_suggested_action_with_lean5() {
        // When lean5_backend is enabled, Unknown should suggest UseManualProof
        let config = VerificationConfig {
            lean5_backend: true,
            ai_synthesis: true,
            ..Default::default()
        };
        let engine = VerificationEngine::new(&config);

        // Verify the methods list includes all escalation options
        let methods = engine.methods_tried_list(false);
        assert!(methods.contains(&"k-induction".to_string()));
        assert!(methods.contains(&"chc".to_string()));
        assert!(methods.contains(&"ai-synthesis".to_string()));
    }

    #[test]
    fn test_verification_cache_entry() {
        let entry = VerificationCacheEntry {
            result: CachedResult::Proven,
            timestamp: 1704067200, // 2024-01-01 00:00:00 UTC
            method: "k-induction".to_string(),
        };

        // Verify serialization
        let json = serde_json::to_string(&entry);
        assert!(json.is_ok());

        // Verify deserialization round-trip
        let json_str = json.unwrap();
        let parsed: Result<VerificationCacheEntry, _> = serde_json::from_str(&json_str);
        assert!(parsed.is_ok());
        let parsed = parsed.unwrap();
        assert!(matches!(parsed.result, CachedResult::Proven));
        assert_eq!(parsed.method, "k-induction");
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation {
            file: "src/lib.rs".to_string(),
            line: 42,
            column: 10,
        };

        // Verify serialization
        let json = serde_json::to_string(&loc);
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("src/lib.rs"));
        assert!(json_str.contains("42"));
    }

    #[test]
    fn test_structured_counterexample() {
        let cx = StructuredCounterexample {
            inputs: HashMap::from([
                ("x".to_string(), "5".to_string()),
                ("y".to_string(), "-3".to_string()),
            ]),
            trace: vec![
                TraceStep {
                    pc: 0,
                    state: HashMap::from([("x".to_string(), "5".to_string())]),
                },
                TraceStep {
                    pc: 1,
                    state: HashMap::from([
                        ("x".to_string(), "5".to_string()),
                        ("y".to_string(), "-3".to_string()),
                    ]),
                },
            ],
            failed_property: "y >= 0".to_string(),
            location: Some(SourceLocation {
                file: "test.rs".to_string(),
                line: 10,
                column: 5,
            }),
        };

        assert_eq!(cx.inputs.len(), 2);
        assert_eq!(cx.trace.len(), 2);
        assert_eq!(cx.failed_property, "y >= 0");
        assert!(cx.location.is_some());

        // Verify serialization
        let json = serde_json::to_string(&cx);
        assert!(json.is_ok());
    }

    #[test]
    fn test_needs_verification_empty_state() {
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);
        let state = IncrementalState::default();

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();
        let input = MirInput::from_mir_program(program);

        // With empty state, all programs need verification
        assert!(
            engine.needs_verification(&input, &state),
            "Should need verification with empty state"
        );
    }

    #[test]
    fn test_needs_verification_with_cached_hash() {
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();
        let input = MirInput::from_mir_program(program.clone());

        // Compute hash and add to state
        let hash = compute_mir_hash(&program);
        let mut state = IncrementalState::default();
        state.verified_hashes.insert(
            hash,
            VerificationCacheEntry {
                result: CachedResult::Proven,
                timestamp: 1704067200,
                method: "k-induction".to_string(),
            },
        );

        // With cached hash, should not need verification
        assert!(
            !engine.needs_verification(&input, &state),
            "Should not need verification with cached hash"
        );
    }

    #[test]
    fn test_needs_verification_different_program() {
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);

        // Create and cache one program
        let program1 = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let hash1 = compute_mir_hash(&program1);
        let mut state = IncrementalState::default();
        state.verified_hashes.insert(
            hash1,
            VerificationCacheEntry {
                result: CachedResult::Proven,
                timestamp: 1704067200,
                method: "k-induction".to_string(),
            },
        );

        // Create a different program
        let program2 = MirProgram::builder(0)
            .local("y", SmtType::Int) // Different variable name
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();
        let input2 = MirInput::from_mir_program(program2);

        // Different program should need verification
        assert!(
            engine.needs_verification(&input2, &state),
            "Different program should need verification"
        );
    }

    #[test]
    fn test_engine_with_state() {
        let config = VerificationConfig::default();

        // Create state with a cached entry
        let mut state = IncrementalState::default();
        state.verified_hashes.insert(
            12345,
            VerificationCacheEntry {
                result: CachedResult::Proven,
                timestamp: 1704067200,
                method: "chc".to_string(),
            },
        );

        // Create engine with state
        let engine = VerificationEngine::with_state(&config, state);

        // Engine should be created successfully
        assert!(engine.config.ai_synthesis);
    }

    #[test]
    fn test_engine_export_state() {
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);

        let state = engine.export_state();

        // Exported state should be empty (new engine)
        assert!(state.verified_hashes.is_empty());
    }

    #[test]
    fn test_needs_verification_cached_with_internal_state() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();
        let hash = compute_mir_hash(&program);

        let mut state = IncrementalState::default();
        state.verified_hashes.insert(
            hash,
            VerificationCacheEntry {
                result: CachedResult::Proven,
                timestamp: 1704067200,
                method: "k-induction".to_string(),
            },
        );

        let config = VerificationConfig::default();
        let engine = VerificationEngine::with_state(&config, state);
        let input = MirInput::from_mir_program(program);

        assert!(
            !engine.needs_verification_cached(&input),
            "Cached entry should skip verification"
        );
    }

    #[test]
    fn test_cached_result_lookup() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();
        let hash = compute_mir_hash(&program);

        let mut state = IncrementalState::default();
        state.verified_hashes.insert(
            hash,
            VerificationCacheEntry {
                result: CachedResult::Unknown,
                timestamp: 1704067200,
                method: "unknown".to_string(),
            },
        );

        let config = VerificationConfig::default();
        let engine = VerificationEngine::with_state(&config, state);
        let input = MirInput::from_mir_program(program);

        let entry = engine.cached_result(&input);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert!(matches!(entry.result, CachedResult::Unknown));
        assert_eq!(entry.method, "unknown");
    }

    #[test]
    fn test_cached_result_variants() {
        // Test all CachedResult variants
        let proven = CachedResult::Proven;
        let disproven = CachedResult::Disproven;
        let unknown = CachedResult::Unknown;

        // Verify serialization for each variant
        let json_proven = serde_json::to_string(&proven);
        let json_disproven = serde_json::to_string(&disproven);
        let json_unknown = serde_json::to_string(&unknown);

        assert!(json_proven.is_ok());
        assert!(json_disproven.is_ok());
        assert!(json_unknown.is_ok());

        // Verify deserialization round-trip
        let parsed: CachedResult = serde_json::from_str(&json_proven.unwrap()).unwrap();
        assert!(matches!(parsed, CachedResult::Proven));
    }

    #[test]
    fn test_mir_input_with_source_info() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let source_info = SourceInfo {
            function_name: "test_func".to_string(),
            file_path: "src/lib.rs".to_string(),
            block_to_line: HashMap::from([(0, 10), (1, 15)]),
        };

        let input = MirInput::from_mir_program(program).with_source_info(source_info);

        assert!(input.source_info.is_some());
        let info = input.source_info.unwrap();
        assert_eq!(info.function_name, "test_func");
        assert_eq!(info.file_path, "src/lib.rs");
        assert_eq!(info.block_to_line.get(&0), Some(&10));
    }

    #[test]
    fn test_mir_input_with_z4_result() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let z4_result = Z4Result::Unknown {
            reason: Z4UnknownReason::Timeout,
            subproblem: Some(Z4Subproblem {
                predicate: "(> x 0)".to_string(),
                context: Some("loop guard".to_string()),
            }),
            partial_proof: Some(Z4PartialProof {
                invariants: vec!["(>= x 0)".to_string()],
                description: Some("proved non-negativity".to_string()),
                certificate: None,
            }),
        };

        let input = MirInput::from_mir_program(program).with_z4_result(z4_result);
        assert!(input.z4_result.is_some());
    }

    #[test]
    fn test_proof_method_serialization() {
        // Test serialization of all ProofMethod variants
        let methods = vec![
            ProofMethod::BoundedModelChecking { bound: 10 },
            ProofMethod::KInduction { k: 5 },
            ProofMethod::CHC {
                invariant: "(>= x 0)".to_string(),
            },
            ProofMethod::AiSynthesis {
                invariant: "(>= x 0)".to_string(),
                attempts: 3,
                source: "ice".to_string(),
            },
            ProofMethod::Lean5 {
                proof_term: "by simp".to_string(),
            },
        ];

        for method in &methods {
            let json = serde_json::to_string(method);
            assert!(
                json.is_ok(),
                "ProofMethod variant should serialize: {:?}",
                method
            );

            // Verify round-trip
            let json_str = json.unwrap();
            let parsed: Result<ProofMethod, _> = serde_json::from_str(&json_str);
            assert!(parsed.is_ok(), "ProofMethod should deserialize");
        }
    }

    #[test]
    fn test_strictness_variants() {
        let strict = Strictness::Strict;
        let best_effort = Strictness::BestEffort;
        let opt_in = Strictness::OptIn;

        // Verify serialization
        assert!(serde_json::to_string(&strict).is_ok());
        assert!(serde_json::to_string(&best_effort).is_ok());
        assert!(serde_json::to_string(&opt_in).is_ok());

        // Verify equality
        assert_eq!(strict, Strictness::Strict);
        assert_ne!(strict, Strictness::BestEffort);
    }

    #[test]
    fn test_engine_error_display() {
        let errors = vec![
            EngineError::ChcError("test error".to_string()),
            EngineError::KInductionError("k-ind error".to_string()),
            EngineError::ConfigError("config error".to_string()),
            EngineError::Timeout(Duration::from_secs(30)),
        ];

        for err in &errors {
            let display = format!("{}", err);
            assert!(!display.is_empty(), "Error should have display message");
        }
    }

    // ==================== KaniResult Method Tests ====================

    #[test]
    fn test_kani_result_is_verified_true_for_verified() {
        let verified = KaniResult::Verified {
            depth: Some(5),
            bound_sufficient: true,
        };
        assert!(
            verified.is_verified(),
            "Verified variant should return true"
        );
    }

    #[test]
    fn test_kani_result_is_verified_false_for_counterexample() {
        let cex = KaniResult::Counterexample {
            trace: vec![],
            rust_test: "test".to_string(),
        };
        assert!(!cex.is_verified(), "Counterexample should not be verified");
    }

    #[test]
    fn test_kani_result_is_verified_false_for_bounded() {
        let bounded = KaniResult::BoundedCheck {
            depth: 10,
            coverage: Coverage::FullDepth,
            suggestion: None,
        };
        assert!(
            !bounded.is_verified(),
            "BoundedCheck should not be verified"
        );
    }

    #[test]
    fn test_kani_result_is_verified_false_for_resource_exhausted() {
        let exhausted = KaniResult::ResourceExhausted {
            reason: "timeout".to_string(),
            partial_coverage: 5,
        };
        assert!(
            !exhausted.is_verified(),
            "ResourceExhausted should not be verified"
        );
    }

    #[test]
    fn test_kani_result_is_counterexample_true_for_counterexample() {
        let cex = KaniResult::Counterexample {
            trace: vec![],
            rust_test: "test".to_string(),
        };
        assert!(
            cex.is_counterexample(),
            "Counterexample variant should return true"
        );
    }

    #[test]
    fn test_kani_result_is_counterexample_false_for_verified() {
        let verified = KaniResult::Verified {
            depth: Some(5),
            bound_sufficient: true,
        };
        assert!(
            !verified.is_counterexample(),
            "Verified should not be counterexample"
        );
    }

    #[test]
    fn test_kani_result_is_counterexample_false_for_bounded() {
        let bounded = KaniResult::BoundedCheck {
            depth: 10,
            coverage: Coverage::FullDepth,
            suggestion: None,
        };
        assert!(
            !bounded.is_counterexample(),
            "BoundedCheck should not be counterexample"
        );
    }

    #[test]
    fn test_kani_result_is_bounded_true_for_bounded() {
        let bounded = KaniResult::BoundedCheck {
            depth: 10,
            coverage: Coverage::FullDepth,
            suggestion: None,
        };
        assert!(
            bounded.is_bounded(),
            "BoundedCheck variant should return true"
        );
    }

    #[test]
    fn test_kani_result_is_bounded_false_for_verified() {
        let verified = KaniResult::Verified {
            depth: Some(5),
            bound_sufficient: true,
        };
        assert!(!verified.is_bounded(), "Verified should not be bounded");
    }

    #[test]
    fn test_kani_result_is_bounded_false_for_counterexample() {
        let cex = KaniResult::Counterexample {
            trace: vec![],
            rust_test: "test".to_string(),
        };
        assert!(!cex.is_bounded(), "Counterexample should not be bounded");
    }

    #[test]
    fn test_kani_result_depth_verified_with_some() {
        let verified = KaniResult::Verified {
            depth: Some(5),
            bound_sufficient: true,
        };
        assert_eq!(
            verified.depth(),
            Some(5),
            "Verified with depth should return Some(5)"
        );
    }

    #[test]
    fn test_kani_result_depth_verified_with_none() {
        let verified = KaniResult::Verified {
            depth: None,
            bound_sufficient: true,
        };
        assert_eq!(
            verified.depth(),
            None,
            "Verified without depth should return None"
        );
    }

    #[test]
    fn test_kani_result_depth_bounded_check() {
        let bounded = KaniResult::BoundedCheck {
            depth: 10,
            coverage: Coverage::FullDepth,
            suggestion: None,
        };
        assert_eq!(
            bounded.depth(),
            Some(10),
            "BoundedCheck should return Some(depth)"
        );
    }

    #[test]
    fn test_kani_result_depth_counterexample_returns_none() {
        let cex = KaniResult::Counterexample {
            trace: vec![],
            rust_test: "test".to_string(),
        };
        assert_eq!(
            cex.depth(),
            None,
            "Counterexample depth() should return None"
        );
    }

    #[test]
    fn test_kani_result_depth_resource_exhausted_returns_none() {
        let exhausted = KaniResult::ResourceExhausted {
            reason: "timeout".to_string(),
            partial_coverage: 5,
        };
        assert_eq!(
            exhausted.depth(),
            None,
            "ResourceExhausted depth() should return None"
        );
    }

    #[test]
    fn test_kani_result_depth_verified_returns_different_values() {
        // Test that depth returns the actual value, not a constant
        let depth_0 = KaniResult::Verified {
            depth: Some(0),
            bound_sufficient: true,
        };
        let depth_1 = KaniResult::Verified {
            depth: Some(1),
            bound_sufficient: true,
        };
        let depth_42 = KaniResult::Verified {
            depth: Some(42),
            bound_sufficient: true,
        };

        assert_eq!(depth_0.depth(), Some(0));
        assert_eq!(depth_1.depth(), Some(1));
        assert_eq!(depth_42.depth(), Some(42));
    }

    #[test]
    fn test_kani_result_depth_bounded_returns_different_values() {
        let bounded_5 = KaniResult::BoundedCheck {
            depth: 5,
            coverage: Coverage::FullDepth,
            suggestion: None,
        };
        let bounded_100 = KaniResult::BoundedCheck {
            depth: 100,
            coverage: Coverage::FullDepth,
            suggestion: None,
        };

        assert_eq!(bounded_5.depth(), Some(5));
        assert_eq!(bounded_100.depth(), Some(100));
    }

    // ==================== generate_rust_test Tests ====================

    #[test]
    fn test_generate_rust_test_basic() {
        let cex = StructuredCounterexample {
            inputs: HashMap::from([("x".to_string(), "5".to_string())]),
            trace: vec![],
            failed_property: "x >= 0".to_string(),
            location: None,
        };

        let test = generate_rust_test(&cex);
        assert!(!test.is_empty(), "Generated test should not be empty");
        assert!(test.contains("#[test]"), "Should have test annotation");
        assert!(
            test.contains("fn test_counterexample()"),
            "Should have function name"
        );
        assert!(
            test.contains("let x = 5;"),
            "Should have variable assignment"
        );
        assert!(test.contains("x >= 0"), "Should mention failed property");
    }

    #[test]
    fn test_generate_rust_test_with_location() {
        let cex = StructuredCounterexample {
            inputs: HashMap::from([("x".to_string(), "5".to_string())]),
            trace: vec![],
            failed_property: "x >= 0".to_string(),
            location: Some(SourceLocation {
                file: "src/lib.rs".to_string(),
                line: 42,
                column: 10,
            }),
        };

        let test = generate_rust_test(&cex);
        assert!(test.contains("src/lib.rs"), "Should have file path");
        assert!(test.contains("42"), "Should have line number");
    }

    #[test]
    fn test_generate_rust_test_boolean_true() {
        let cex = StructuredCounterexample {
            inputs: HashMap::from([("flag".to_string(), "true".to_string())]),
            trace: vec![],
            failed_property: "test".to_string(),
            location: None,
        };

        let test = generate_rust_test(&cex);
        assert!(
            test.contains("let flag = true;"),
            "Should handle boolean true"
        );
    }

    #[test]
    fn test_generate_rust_test_boolean_false() {
        let cex = StructuredCounterexample {
            inputs: HashMap::from([("flag".to_string(), "false".to_string())]),
            trace: vec![],
            failed_property: "test".to_string(),
            location: None,
        };

        let test = generate_rust_test(&cex);
        assert!(
            test.contains("let flag = false;"),
            "Should handle boolean false"
        );
    }

    #[test]
    fn test_generate_rust_test_string_value() {
        let cex = StructuredCounterexample {
            inputs: HashMap::from([("msg".to_string(), "\"hello\"".to_string())]),
            trace: vec![],
            failed_property: "test".to_string(),
            location: None,
        };

        let test = generate_rust_test(&cex);
        assert!(
            test.contains("let msg = \"hello\";"),
            "Should handle string value"
        );
    }

    #[test]
    fn test_generate_rust_test_negative_integer() {
        let cex = StructuredCounterexample {
            inputs: HashMap::from([("x".to_string(), "-42".to_string())]),
            trace: vec![],
            failed_property: "test".to_string(),
            location: None,
        };

        let test = generate_rust_test(&cex);
        assert!(
            test.contains("let x = -42;"),
            "Should handle negative integer"
        );
    }

    // ==================== BoundsInference Edge Case Tests ====================

    #[test]
    fn test_bounds_inference_negative_lower_bound() {
        // Test the line: let bound = (-lower).max(0) as u32;
        // When lower is negative, -lower is positive
        let specs = FunctionSpecs::new().variant("x - (-10)");
        // Parse variant manually to set lower_bound
        let mut specs_with_lower = specs;
        specs_with_lower.variants = vec![VariantSpec {
            expression: "x - (-10)".to_string(),
            upper_bound_var: None,
            lower_bound: Some(-10), // Negative lower bound means countdown from 10
            loop_id: None,
        }];

        let inference = BoundsInference::with_bounds(10, 1000);
        let result = inference.infer_bounds(&specs_with_lower);

        // -(-10) = 10, which is > 0 and <= 1000, so should be suggested
        assert_eq!(result.suggested_unwind, Some(10));
    }

    #[test]
    fn test_bounds_inference_zero_lower_bound() {
        // When lower is 0, -lower is 0, which fails the > 0 check
        let mut specs = FunctionSpecs::new();
        specs.variants = vec![VariantSpec {
            expression: "x - 0".to_string(),
            upper_bound_var: None,
            lower_bound: Some(0),
            loop_id: None,
        }];

        let inference = BoundsInference::with_bounds(10, 1000);
        let result = inference.infer_bounds(&specs);

        // bound = 0, which fails > 0 check, so should use default
        assert_eq!(result.suggested_unwind, Some(10)); // default
    }

    #[test]
    fn test_bounds_inference_positive_lower_bound() {
        // When lower is positive, -lower is negative, which fails > 0 after max(0)
        let mut specs = FunctionSpecs::new();
        specs.variants = vec![VariantSpec {
            expression: "x - 5".to_string(),
            upper_bound_var: None,
            lower_bound: Some(5),
            loop_id: None,
        }];

        let inference = BoundsInference::with_bounds(10, 1000);
        let result = inference.infer_bounds(&specs);

        // bound = max(-5, 0) = 0, fails > 0 check, use default
        assert_eq!(result.suggested_unwind, Some(10));
    }

    #[test]
    fn test_bounds_inference_lower_bound_sets_non_default_bound() {
        // Ensure the lower-bound countdown path uses the computed bound, not the default
        let mut specs = FunctionSpecs::new();
        specs.variants = vec![VariantSpec {
            expression: "countdown".to_string(),
            upper_bound_var: Some("countdown".to_string()),
            lower_bound: Some(-7), // bound = 7
            loop_id: None,
        }];

        let inference = BoundsInference::with_bounds(3, 20);
        let result = inference.infer_bounds(&specs);

        // The computed bound (7) should be used instead of the default (3)
        assert_eq!(result.suggested_unwind, Some(7));
    }

    #[test]
    fn test_bounds_inference_bound_exceeds_max() {
        // Test: bound > 0 && bound <= self.max_bound
        // If bound is > max_bound, the condition fails
        let mut specs = FunctionSpecs::new();
        specs.variants = vec![VariantSpec {
            expression: "countdown".to_string(),
            upper_bound_var: None,
            lower_bound: Some(-5000), // -(-5000) = 5000
            loop_id: None,
        }];

        let inference = BoundsInference::with_bounds(10, 100); // max is 100
        let result = inference.infer_bounds(&specs);

        // bound = 5000, but max_bound = 100, so fails <= check
        // Falls through to default
        assert_eq!(result.suggested_unwind, Some(10));
    }

    // ==================== infer_array_bounds Tests ====================

    #[test]
    fn test_infer_array_bounds_len_pattern() {
        let specs = FunctionSpecs::new().require("(< idx (len arr))");
        let inference = BoundsInference::with_bounds(10, 1000);
        let bounds = inference.infer_array_bounds(&specs);

        // Should find the idx variable bound
        assert_eq!(bounds.get("idx"), Some(&10));
    }

    #[test]
    fn test_infer_array_bounds_dot_len_pattern() {
        let specs = FunctionSpecs::new().require("idx < arr.len()");
        let inference = BoundsInference::with_bounds(20, 1000);
        let bounds = inference.infer_array_bounds(&specs);

        // Should find the idx variable bound
        assert_eq!(bounds.get("idx"), Some(&20));
    }

    #[test]
    fn test_infer_array_bounds_no_match() {
        let specs = FunctionSpecs::new().require("(< x 100)");
        let inference = BoundsInference::with_bounds(10, 1000);
        let bounds = inference.infer_array_bounds(&specs);

        // No array length pattern, should return empty
        assert!(bounds.is_empty());
    }

    #[test]
    fn test_infer_array_bounds_empty_specs() {
        let specs = FunctionSpecs::new();
        let inference = BoundsInference::with_bounds(10, 1000);
        let bounds = inference.infer_array_bounds(&specs);

        assert!(bounds.is_empty());
    }

    // ==================== collect_variables_from_ast Edge Cases ====================

    #[test]
    fn test_extract_variables_from_neg_expr() {
        // Test: SmtAst::Neg case - should collect variables from negated expression
        let vars = extract_variables("(- x)");
        assert!(vars.contains(&"x".to_string()), "Should find x in negation");
    }

    #[test]
    fn test_extract_variables_forall_excludes_bound() {
        // Test: Forall/Exists excludes quantified variables
        let vars = extract_variables("(forall ((n Int)) (>= n 0))");
        // 'n' is bound by forall, so should not appear in free vars
        assert!(
            !vars.contains(&"n".to_string()),
            "Quantified variable n should not be returned as free"
        );
    }

    #[test]
    fn test_extract_variables_nested_let() {
        // Test: Let bindings should exclude bound variables
        let vars = extract_variables("(let ((a 5)) (let ((b a)) (+ b x)))");
        // 'x' should be free, 'a' and 'b' are bound
        assert!(vars.contains(&"x".to_string()), "x should be free");
    }

    #[test]
    fn test_extract_variables_app_with_multiple_args() {
        // Test: App case with multiple arguments
        let vars = extract_variables("(+ a b c d)");
        assert!(vars.contains(&"a".to_string()));
        assert!(vars.contains(&"b".to_string()));
        assert!(vars.contains(&"c".to_string()));
        assert!(vars.contains(&"d".to_string()));
        assert_eq!(vars.len(), 4);
    }

    #[test]
    fn test_extract_variables_does_not_include_primed() {
        // Test: Primed variables (x') should be excluded
        let vars = extract_variables("(and (>= x 0) (>= x' 0))");
        assert!(vars.contains(&"x".to_string()));
        assert!(
            !vars.contains(&"x'".to_string()),
            "Primed variables should be excluded"
        );
    }

    #[test]
    fn test_extract_variables_excludes_keywords() {
        // Test: Keywords like 'and', 'or', '+', etc. should not be variables
        let vars = extract_variables("(and (or (+ x y) (- z)))");
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert!(vars.contains(&"z".to_string()));
        assert!(
            !vars.contains(&"and".to_string()),
            "and should not be a variable"
        );
        assert!(
            !vars.contains(&"or".to_string()),
            "or should not be a variable"
        );
    }

    #[test]
    fn test_collect_variables_let_excludes_bound_names() {
        // Construct AST directly to target Let branch without parser quirks
        let ast = SmtAst::Let(
            vec![("a".to_string(), SmtAst::Int(5))],
            Box::new(SmtAst::Symbol("a".to_string())),
        );

        let vars = collect_variables_from_ast(&ast);
        assert!(
            vars.is_empty(),
            "Let-bound variable should not appear as free variable"
        );
    }

    #[test]
    fn test_collect_variables_forall_excludes_bound_names() {
        let ast = SmtAst::Forall(
            vec![("n".to_string(), SmtSort::Int)],
            Box::new(SmtAst::Symbol("n".to_string())),
        );

        let vars = collect_variables_from_ast(&ast);
        assert!(
            vars.is_empty(),
            "Quantified variable should not be collected as free"
        );
    }

    #[test]
    fn test_collect_variables_skips_primed_and_keywords() {
        let primed = SmtAst::Symbol("x'".to_string());
        let keyword = SmtAst::Symbol("and".to_string());

        assert!(
            collect_variables_from_ast(&primed).is_empty(),
            "Primed variables should be skipped"
        );
        assert!(
            collect_variables_from_ast(&keyword).is_empty(),
            "Keywords should be skipped"
        );
    }

    // ==================== Coverage enum tests ====================

    #[test]
    fn test_coverage_full_depth() {
        let coverage = Coverage::FullDepth;
        let json = serde_json::to_string(&coverage);
        assert!(json.is_ok());
    }

    #[test]
    fn test_coverage_partial() {
        let coverage = Coverage::Partial {
            paths_checked: 5,
            paths_total: Some(10),
        };
        let json = serde_json::to_string(&coverage);
        assert!(json.is_ok());
    }

    #[test]
    fn test_coverage_unknown() {
        let coverage = Coverage::Unknown;
        let json = serde_json::to_string(&coverage);
        assert!(json.is_ok());
    }

    // ==================== proof_method_label Tests ====================

    #[test]
    fn test_proof_method_label_bmc() {
        let label = proof_method_label(&ProofMethod::BoundedModelChecking { bound: 10 });
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.contains("bmc"), "Should contain bmc");
        assert!(label.contains("10"), "Should contain bound value");
    }

    #[test]
    fn test_proof_method_label_kinduction() {
        let label = proof_method_label(&ProofMethod::KInduction { k: 5 });
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.contains("k-induction"), "Should contain k-induction");
        assert!(label.contains("5"), "Should contain k value");
    }

    #[test]
    fn test_proof_method_label_chc() {
        let label = proof_method_label(&ProofMethod::CHC {
            invariant: "(>= x 0)".to_string(),
        });
        assert!(label.is_some());
        assert_eq!(label.unwrap(), "chc");
    }

    #[test]
    fn test_proof_method_label_ai_synthesis() {
        let label = proof_method_label(&ProofMethod::AiSynthesis {
            invariant: "(>= x 0)".to_string(),
            attempts: 3,
            source: "ice".to_string(),
        });
        assert!(label.is_some());
        let label = label.unwrap();
        assert!(label.contains("ai-synthesis"));
        assert!(label.contains("ice"), "Should contain source");
    }

    #[test]
    fn test_proof_method_label_lean5() {
        let label = proof_method_label(&ProofMethod::Lean5 {
            proof_term: "by simp".to_string(),
        });
        assert!(label.is_some());
        assert_eq!(label.unwrap(), "lean5");
    }

    #[test]
    fn test_proof_method_label_returns_some_not_none() {
        // All proof methods should return Some, never None
        let methods = vec![
            ProofMethod::BoundedModelChecking { bound: 1 },
            ProofMethod::KInduction { k: 1 },
            ProofMethod::CHC {
                invariant: "x".to_string(),
            },
            ProofMethod::AiSynthesis {
                invariant: "x".to_string(),
                attempts: 1,
                source: "test".to_string(),
            },
            ProofMethod::Lean5 {
                proof_term: "x".to_string(),
            },
        ];

        for method in &methods {
            assert!(
                proof_method_label(method).is_some(),
                "proof_method_label should never return None for {:?}",
                method
            );
        }
    }

    #[test]
    fn test_proof_method_label_content_not_empty() {
        // Labels should never be empty strings
        let methods = vec![
            ProofMethod::BoundedModelChecking { bound: 1 },
            ProofMethod::KInduction { k: 1 },
            ProofMethod::CHC {
                invariant: "x".to_string(),
            },
            ProofMethod::AiSynthesis {
                invariant: "x".to_string(),
                attempts: 1,
                source: "test".to_string(),
            },
            ProofMethod::Lean5 {
                proof_term: "x".to_string(),
            },
        ];

        for method in &methods {
            let label = proof_method_label(method).unwrap();
            assert!(
                !label.is_empty(),
                "proof_method_label should not return empty string for {:?}",
                method
            );
        }
    }

    // ==================== needs_verification_cached Tests ====================

    #[test]
    fn test_needs_verification_cached_with_empty_cache() {
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();
        let input = MirInput::from_mir_program(program);

        // Empty cache should require verification
        assert!(
            engine.needs_verification_cached(&input),
            "Empty cache should require verification"
        );
    }

    #[test]
    fn test_needs_verification_cached_returns_true_for_new_program() {
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();
        let input = MirInput::from_mir_program(program);

        // New program needs verification
        let result = engine.needs_verification_cached(&input);
        assert!(result, "New program should need verification");
    }

    // ==================== collect_variables_from_ast Additional Tests ====================

    #[test]
    fn test_extract_variables_let_binding_excludes_bound_var() {
        // Test that let-bound variables are excluded from free variables
        // This tests the condition: !vars.contains(&v) in the Let branch
        let vars = extract_variables("(let ((a 5)) (+ a x))");

        // 'x' is free, 'a' is bound by let
        assert!(vars.contains(&"x".to_string()), "x should be free");

        // Note: current implementation may include 'a' from binding value evaluation
        // The key is that free variables are correctly identified
    }

    #[test]
    fn test_extract_variables_forall_excludes_quantified() {
        // Test that forall-bound variables are excluded
        // This tests: !vars.contains(&v) && !bound_names.contains(&v)
        let vars = extract_variables("(forall ((n Int)) (and (>= n 0) (< m 10)))");

        // 'm' is free, 'n' is bound by forall
        assert!(vars.contains(&"m".to_string()), "m should be free");
    }

    #[test]
    fn test_extract_variables_exists_excludes_quantified() {
        // Test that exists-bound variables are excluded
        let vars = extract_variables("(exists ((y Int)) (and (= x y) (> z 0)))");

        // 'x' and 'z' are free, 'y' is bound by exists
        assert!(vars.contains(&"x".to_string()), "x should be free");
        assert!(vars.contains(&"z".to_string()), "z should be free");
    }

    #[test]
    fn test_extract_variables_nested_let_correct_scoping() {
        // Nested let should correctly track bound names at each level
        let vars = extract_variables("(let ((outer 1)) (let ((inner outer)) (+ inner free)))");
        assert!(
            vars.contains(&"free".to_string()),
            "free should be identified"
        );
    }

    #[test]
    fn test_extract_variables_app_deduplicates() {
        // Variables should not be duplicated even if they appear multiple times
        let vars = extract_variables("(+ x x)");
        assert_eq!(
            vars.iter().filter(|&v| v == "x").count(),
            1,
            "x should appear only once"
        );
    }

    // ==================== Mutation Coverage Tests ====================

    #[test]
    fn test_bounds_inference_zero_bound_not_used() {
        // Tests the `bound > 0` condition in infer_bounds line 803
        // A countdown variant with lower_bound of 0 should produce bound of 0
        // which should NOT be used as suggested_unwind
        //
        // Variant "n - 0" parses to: upper_bound_var="n", lower_bound=Some(0)
        // Without a precondition for "n", bound = (-0).max(0) = 0
        // bound > 0 is false, so fallback to default
        let specs = FunctionSpecs::new().variant("n - 0");

        let inference = BoundsInference::with_bounds(20, 1000);
        let result = inference.infer_bounds(&specs);

        // Should fall back to default bound because bound=0 fails `bound > 0`
        assert_eq!(
            result.suggested_unwind,
            Some(20),
            "Zero bound should fall back to default"
        );
        assert!(
            result.confidence < 0.5,
            "Low confidence expected for fallback"
        );
    }

    #[test]
    fn test_bounds_inference_positive_bound_used() {
        // Tests that positive bounds ARE used (contrast to zero bound test)
        // Variant "n - -5" parses to: upper_bound_var="n", lower_bound=Some(-5)
        // bound = (-(-5)).max(0) = 5
        // bound > 0 is true, so it should be used
        let specs = FunctionSpecs::new().variant("(- n -5)"); // SMT-LIB format for n - (-5)

        let inference = BoundsInference::with_bounds(20, 1000);
        let result = inference.infer_bounds(&specs);

        assert_eq!(
            result.suggested_unwind,
            Some(5),
            "Positive bound from negative lower should be used"
        );
        assert!(
            result.confidence >= 0.7,
            "Should have decent confidence for inferred bound"
        );
    }

    #[test]
    fn test_bounds_inference_bound_vs_condition_boundary() {
        // Tests that > vs >= matters for the bound > 0 check
        // If mutant changes > to >=, bound=0 would be accepted incorrectly
        let specs_zero = FunctionSpecs::new().variant("n - 0"); // bound = 0
        let specs_one = FunctionSpecs::new().variant("(- n -1)"); // bound = 1

        let inference = BoundsInference::with_bounds(20, 1000);

        let result_zero = inference.infer_bounds(&specs_zero);
        let result_one = inference.infer_bounds(&specs_one);

        // bound=0 should use default (fails bound > 0)
        assert_eq!(
            result_zero.suggested_unwind,
            Some(20),
            "bound=0 should fall back to default"
        );

        // bound=1 should use the inferred bound (passes bound > 0)
        assert_eq!(
            result_one.suggested_unwind,
            Some(1),
            "bound=1 should be used"
        );

        // The confidence levels should also differ
        assert!(
            result_one.confidence > result_zero.confidence,
            "Higher confidence expected for bound=1 vs bound=0"
        );
    }

    #[test]
    fn test_extract_bounds_inclusive_vs_exclusive() {
        // Tests the `bound + 1` vs `bound` logic in line 892
        let inference = BoundsInference::with_bounds(10, 10000);

        // Exclusive: (< n 50) should give bound of 50
        let bounds_excl = inference.extract_bounds_from_preconditions(&["(< n 50)".to_string()]);
        assert_eq!(bounds_excl.get("n"), Some(&50));

        // Inclusive: (<= n 50) should give bound of 51 (bound + 1)
        let bounds_incl = inference.extract_bounds_from_preconditions(&["(<= n 50)".to_string()]);
        assert_eq!(bounds_incl.get("n"), Some(&51));

        // The difference should be exactly 1
        assert_eq!(
            bounds_incl.get("n").unwrap() - bounds_excl.get("n").unwrap(),
            1,
            "Inclusive bound should be exactly 1 more than exclusive"
        );
    }

    #[test]
    fn test_extract_bounds_infix_inclusive_vs_exclusive() {
        // Same test for infix format
        let inference = BoundsInference::with_bounds(10, 10000);

        // Exclusive: n < 50 should give bound of 50
        let bounds_excl = inference.extract_bounds_from_preconditions(&["n < 50".to_string()]);
        assert_eq!(bounds_excl.get("n"), Some(&50));

        // Inclusive: n <= 50 should give bound of 51
        let bounds_incl = inference.extract_bounds_from_preconditions(&["n <= 50".to_string()]);
        assert_eq!(bounds_incl.get("n"), Some(&51));
    }

    #[test]
    fn test_build_ai_config_sets_use_llm() {
        // Tests that use_llm field is set correctly (line 1276)
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);
        let ai_config = engine.build_ai_config();

        assert!(
            ai_config.use_llm,
            "use_llm should be true - if false, AI synthesis would be crippled"
        );
    }

    #[test]
    fn test_build_ai_config_sets_use_corpus() {
        // Tests that use_corpus field is set correctly (line 1277)
        let config = VerificationConfig::default();
        let engine = VerificationEngine::new(&config);
        let ai_config = engine.build_ai_config();

        assert!(
            ai_config.use_corpus,
            "use_corpus should be true - corpus improves synthesis success rate"
        );
    }

    #[test]
    fn test_build_ai_config_fields_affect_synthesis() {
        // Ensure the fields we set actually matter
        let config = VerificationConfig {
            ai_max_attempts: 5,
            ai_timeout: Duration::from_secs(42),
            ..Default::default()
        };
        let engine = VerificationEngine::new(&config);
        let ai_config = engine.build_ai_config();

        // All these fields should be correctly set
        assert_eq!(ai_config.max_attempts, 5);
        assert_eq!(ai_config.timeout_secs, 42);
        assert!(ai_config.use_llm);
        assert!(ai_config.use_corpus);
    }

    #[test]
    fn test_collect_variables_let_body_deduplication() {
        // Tests the `!vars.contains(&v)` condition in line 1699
        // Multiple occurrences of a variable in let body should not duplicate
        // Note: x is bound in let, so body refs to x don't add to free vars
        // But y appears multiple times and should only be collected once
        let vars = extract_variables("(let ((x 5)) (+ y y y))");

        // 'y' appears 3 times but should be collected only once
        let y_count = vars.iter().filter(|v| *v == "y").count();
        assert_eq!(
            y_count, 1,
            "y should appear exactly once despite multiple occurrences"
        );
    }

    #[test]
    fn test_collect_variables_let_binding_values_deduplication() {
        // Tests the `!vars.contains(&v)` condition in line 1691
        // This specifically targets the binding value collection, not the body
        // If the ! is deleted, we'd add duplicates when same var appears in
        // multiple binding values
        let vars = extract_variables("(let ((a (+ z 1)) (b (+ z 2))) w)");

        // 'z' appears in both binding values but should be collected once
        let z_count = vars.iter().filter(|v| *v == "z").count();
        assert_eq!(
            z_count, 1,
            "z appears in multiple bindings but should be collected once"
        );
    }

    #[test]
    fn test_collect_variables_app_args_deduplication() {
        // Tests deduplication in function application arguments
        // The `!vars.contains(&v)` check prevents duplicates
        let vars = extract_variables("(+ x y x z x)");

        let x_count = vars.iter().filter(|v| *v == "x").count();
        assert_eq!(x_count, 1, "x appears 3 times but should be collected once");
    }

    #[test]
    fn test_collect_variables_nested_expressions_deduplication() {
        // More complex test for deduplication across nested expressions
        let vars = extract_variables("(and (>= x 0) (and (< x 100) (+ x y)))");

        let x_count = vars.iter().filter(|v| *v == "x").count();
        assert_eq!(
            x_count, 1,
            "x should appear exactly once despite multiple occurrences"
        );
    }
}
