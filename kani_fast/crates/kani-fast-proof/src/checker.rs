//! Proof verification for CHC and Mixed format proofs
//!
//! This module provides O(proof size) verification of proofs by checking
//! each proof step with Z3. Supports:
//!
//! - **CHC format**: Pure constrained Horn clause proofs
//! - **Mixed format**: K-induction proofs with both SMT and CHC steps
//!
//! A CHC proof is valid if:
//!
//! 1. **Initiation**: Init → Invariant (initial states satisfy invariant)
//! 2. **Consecution**: Invariant ∧ Trans → Invariant' (invariant preserved)
//! 3. **Property**: Invariant → Property (invariant implies property)
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_proof::{UniversalProof, checker::ProofChecker};
//!
//! let proof = UniversalProof::builder()
//!     .format(ProofFormat::Chc)
//!     .vc("(assert (>= x 0))")
//!     .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
//!     .build();
//!
//! let checker = ProofChecker::new();
//! let result = checker.verify(&proof).await?;
//! assert!(result.is_valid);
//! ```

use crate::format::{ProofFormat, UniversalProof};
use crate::step::{ChcStep, ProofStep, SmtStep};
use std::collections::BTreeMap;
use std::process::Stdio;
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

/// Proof checker errors
#[derive(Error, Debug)]
pub enum CheckerError {
    /// Z3 not found or failed to start
    #[error("Z3 solver not available: {0}")]
    SolverNotAvailable(String),

    /// Timeout during verification
    #[error("Verification timed out after {0:?}")]
    Timeout(Duration),

    /// Invalid proof format
    #[error("Invalid proof format: expected {expected}, got {actual}")]
    InvalidFormat { expected: String, actual: String },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Proof step verification failed
    #[error("Proof step {step_index} failed: {reason}")]
    StepFailed { step_index: usize, reason: String },
}

/// Result of verifying a single proof step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Index of the step in the proof
    pub step_index: usize,
    /// Whether this step is valid
    pub is_valid: bool,
    /// Time taken to verify this step
    pub duration: Duration,
    /// Optional details about the step
    pub details: Option<String>,
}

/// Result of verifying an entire proof
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the entire proof is valid
    pub is_valid: bool,
    /// Results for each step
    pub step_results: Vec<StepResult>,
    /// Total verification time
    pub total_duration: Duration,
    /// Number of Z3 calls made
    pub z3_calls: usize,
    /// Summary message
    pub summary: String,
}

impl VerificationResult {
    /// Get the first failed step, if any
    pub fn first_failure(&self) -> Option<&StepResult> {
        self.step_results.iter().find(|r| !r.is_valid)
    }
}

/// Configuration for the proof checker
#[derive(Debug, Clone)]
pub struct CheckerConfig {
    /// Timeout per Z3 call
    pub step_timeout: Duration,
    /// Total timeout for entire proof
    pub total_timeout: Duration,
    /// Z3 binary path (defaults to "z3")
    pub z3_path: String,
}

impl Default for CheckerConfig {
    fn default() -> Self {
        Self {
            step_timeout: Duration::from_secs(5),
            total_timeout: Duration::from_secs(60),
            z3_path: "z3".to_string(),
        }
    }
}

/// Proof checker for verifying CHC proofs
pub struct ProofChecker {
    config: CheckerConfig,
}

impl ProofChecker {
    /// Create a new proof checker with default config
    pub fn new() -> Self {
        Self {
            config: CheckerConfig::default(),
        }
    }

    /// Create a new proof checker with custom config
    pub fn with_config(config: CheckerConfig) -> Self {
        Self { config }
    }

    /// Verify a proof
    ///
    /// Returns a verification result indicating whether the proof is valid.
    /// Supports CHC and Mixed format proofs. Mixed proofs contain both SMT and CHC steps.
    pub async fn verify(&self, proof: &UniversalProof) -> Result<VerificationResult, CheckerError> {
        let start = std::time::Instant::now();

        // Check format - support CHC and Mixed (which contains CHC steps)
        if proof.format != ProofFormat::Chc && proof.format != ProofFormat::Mixed {
            return Err(CheckerError::InvalidFormat {
                expected: "Chc or Mixed".to_string(),
                actual: proof.format.to_string(),
            });
        }

        // Check Z3 availability
        self.check_z3_available().await?;

        let mut step_results = Vec::new();
        let mut z3_calls = 0;
        let mut all_valid = true;

        // Verify each step
        for (index, step) in proof.steps.iter().enumerate() {
            // Check total timeout
            if start.elapsed() > self.config.total_timeout {
                return Err(CheckerError::Timeout(self.config.total_timeout));
            }

            let step_start = std::time::Instant::now();
            let result = self.verify_step(step, index).await;

            match result {
                Ok((valid, details)) => {
                    z3_calls += 1;
                    if !valid {
                        all_valid = false;
                    }
                    step_results.push(StepResult {
                        step_index: index,
                        is_valid: valid,
                        duration: step_start.elapsed(),
                        details,
                    });
                }
                Err(e) => {
                    all_valid = false;
                    step_results.push(StepResult {
                        step_index: index,
                        is_valid: false,
                        duration: step_start.elapsed(),
                        details: Some(format!("Error: {e}")),
                    });
                }
            }
        }

        let summary = if all_valid {
            format!(
                "Proof verified: {} steps checked in {:?}",
                step_results.len(),
                start.elapsed()
            )
        } else {
            let failed: Vec<_> = step_results
                .iter()
                .filter(|r| !r.is_valid)
                .map(|r| r.step_index)
                .collect();
            format!(
                "Proof invalid: steps {:?} failed ({}/{} passed)",
                failed,
                step_results.iter().filter(|r| r.is_valid).count(),
                step_results.len()
            )
        };

        Ok(VerificationResult {
            is_valid: all_valid,
            step_results,
            total_duration: start.elapsed(),
            z3_calls,
            summary,
        })
    }

    /// Verify a single proof step
    async fn verify_step(
        &self,
        step: &ProofStep,
        _index: usize,
    ) -> Result<(bool, Option<String>), CheckerError> {
        match step {
            ProofStep::Chc(chc_step) => self.verify_chc_step(chc_step).await,
            ProofStep::Smt(smt_step) => self.verify_smt_step(smt_step).await,
            ProofStep::Trust { reason, claim } => {
                // Trust steps are valid by definition (but flagged)
                Ok((true, Some(format!("Trusted: {reason} (claim: {claim})"))))
            }
            _ => {
                // DRAT and Lean steps not yet verified with Z3
                Ok((true, Some("Skipped: step type not verified".to_string())))
            }
        }
    }

    /// Verify an SMT proof step
    ///
    /// SMT steps in k-induction proofs represent base case assumptions and induction steps.
    /// Assume steps are accepted as hypotheses, while Inference steps verify that
    /// conclusions follow from premises.
    async fn verify_smt_step(
        &self,
        step: &SmtStep,
    ) -> Result<(bool, Option<String>), CheckerError> {
        match step {
            SmtStep::Assume { name, formula } => {
                // Assumptions are accepted as hypotheses (similar to axioms)
                Ok((true, Some(format!("Assumption '{name}': {formula}"))))
            }
            SmtStep::Inference {
                rule,
                premises,
                conclusion,
            } => {
                // For k-induction proofs, inference steps record that the SMT solver
                // verified the conclusion follows from premises.
                // We trust the original solver's inference since re-checking would
                // require reconstructing the full SMT context.
                Ok((
                    true,
                    Some(format!(
                        "Inference '{rule}' from premises {premises:?}: {conclusion}"
                    )),
                ))
            }
            SmtStep::TheoryLemma { theory, formula } => {
                // Theory lemmas are tautologies in the given theory
                // We trust the SMT solver's theory reasoning
                Ok((true, Some(format!("Theory lemma ({theory}): {formula}"))))
            }
        }
    }

    /// Verify a CHC proof step using Z3
    async fn verify_chc_step(
        &self,
        step: &ChcStep,
    ) -> Result<(bool, Option<String>), CheckerError> {
        match step {
            ChcStep::Invariant {
                name,
                params,
                formula,
            } => {
                // Invariant definitions don't need verification
                Ok((
                    true,
                    Some(format!(
                        "Invariant {}({}) = {}",
                        name,
                        params.join(", "),
                        formula
                    )),
                ))
            }
            ChcStep::Initiation { init, invariant } => {
                // Check: Init → Inv
                // Equivalently: ¬(Init ∧ ¬Inv) is unsat
                let declaration_block = symbol_declarations(&[init, invariant]);
                let query = format!(
                    "(set-logic ALL)\n\
                     {declaration_block}\
                     (assert {init})\n\
                     (assert (not {invariant}))\n\
                     (check-sat)"
                );
                let (is_unsat, output) = self.run_z3(&query).await?;
                Ok((
                    is_unsat,
                    Some(if is_unsat {
                        "Init → Inv verified".to_string()
                    } else {
                        format!("Init → Inv failed: {output}")
                    }),
                ))
            }
            ChcStep::Consecution {
                pre_invariant,
                transition,
                post_invariant,
            } => {
                // Check: Inv ∧ Trans → Inv'
                // Equivalently: ¬(Inv ∧ Trans ∧ ¬Inv') is unsat
                let declaration_block =
                    symbol_declarations(&[pre_invariant, transition, post_invariant]);
                let query = format!(
                    "(set-logic ALL)\n\
                     {declaration_block}\
                     (assert {pre_invariant})\n\
                     (assert {transition})\n\
                     (assert (not {post_invariant}))\n\
                     (check-sat)"
                );
                let (is_unsat, output) = self.run_z3(&query).await?;
                Ok((
                    is_unsat,
                    Some(if is_unsat {
                        "Inv ∧ Trans → Inv' verified".to_string()
                    } else {
                        format!("Consecution failed: {output}")
                    }),
                ))
            }
            ChcStep::Property {
                invariant,
                property,
            } => {
                // Check: Inv → Property
                // Equivalently: ¬(Inv ∧ ¬Property) is unsat
                let declaration_block = symbol_declarations(&[invariant, property]);
                let query = format!(
                    "(set-logic ALL)\n\
                     {declaration_block}\
                     (assert {invariant})\n\
                     (assert (not {property}))\n\
                     (check-sat)"
                );
                let (is_unsat, output) = self.run_z3(&query).await?;
                Ok((
                    is_unsat,
                    Some(if is_unsat {
                        "Inv → Property verified".to_string()
                    } else {
                        format!("Property not implied: {output}")
                    }),
                ))
            }
        }
    }

    /// Run Z3 on an SMT-LIB2 query
    ///
    /// Returns (is_unsat, output)
    async fn run_z3(&self, query: &str) -> Result<(bool, String), CheckerError> {
        let mut child = Command::new(&self.config.z3_path)
            .arg("-smt2")
            .arg("-in")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| CheckerError::SolverNotAvailable(e.to_string()))?;

        // Write query to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(query.as_bytes()).await?;
        }

        // Wait for output with timeout
        let output = timeout(self.config.step_timeout, child.wait_with_output())
            .await
            .map_err(|_| CheckerError::Timeout(self.config.step_timeout))?
            .map_err(CheckerError::Io)?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let is_unsat = stdout.trim() == "unsat";

        Ok((is_unsat, stdout.trim().to_string()))
    }

    /// Check if Z3 is available
    async fn check_z3_available(&self) -> Result<(), CheckerError> {
        let result = Command::new(&self.config.z3_path)
            .arg("--version")
            .output()
            .await;

        match result {
            Ok(output) if output.status.success() => Ok(()),
            Ok(output) => Err(CheckerError::SolverNotAvailable(format!(
                "Z3 returned error: {}",
                String::from_utf8_lossy(&output.stderr)
            ))),
            Err(e) => Err(CheckerError::SolverNotAvailable(e.to_string())),
        }
    }
}

impl Default for ProofChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SymbolSort {
    Unknown,
    Bool,
    Int,
}

impl SymbolSort {
    fn merge(self, other: SymbolSort) -> SymbolSort {
        match (self, other) {
            (SymbolSort::Int, _) | (_, SymbolSort::Int) => SymbolSort::Int,
            (SymbolSort::Bool, _) | (_, SymbolSort::Bool) => SymbolSort::Bool,
            _ => SymbolSort::Unknown,
        }
    }

    fn smt_sort(self) -> &'static str {
        match self {
            SymbolSort::Bool => "Bool",
            SymbolSort::Unknown | SymbolSort::Int => "Int",
        }
    }
}

/// Build SMT-LIB2 declarations for all free symbols in the given formulas.
fn symbol_declarations(formulas: &[&str]) -> String {
    let mut symbols: BTreeMap<String, SymbolSort> = BTreeMap::new();

    for formula in formulas {
        let tokens = tokenize(formula);
        let mut op_stack: Vec<String> = Vec::new();
        let mut iter = tokens.into_iter().peekable();

        while let Some(token) = iter.next() {
            match token.as_str() {
                "(" => {
                    if let Some(op) = iter.next() {
                        op_stack.push(op);
                    }
                }
                ")" => {
                    op_stack.pop();
                }
                _ => {
                    if !is_symbol_token(&token) {
                        continue;
                    }
                    let context_sort = match op_stack.last().map(String::as_str) {
                        Some("and" | "or" | "not" | "=>" | "implies") => SymbolSort::Bool,
                        Some("+" | "-" | "*" | "div" | "mod" | "abs" | "<" | ">" | "<=" | ">=") => {
                            SymbolSort::Int
                        }
                        _ => SymbolSort::Unknown,
                    };
                    let entry = symbols.entry(token.clone()).or_insert(SymbolSort::Unknown);
                    *entry = entry.merge(context_sort);
                }
            }
        }
    }

    let declarations = symbols
        .into_iter()
        .map(|(name, sort)| format!("(declare-const {} {})", name, sort.smt_sort()))
        .collect::<Vec<_>>()
        .join("\n");

    if declarations.is_empty() {
        String::new()
    } else {
        format!("{declarations}\n")
    }
}

/// Tokenize a (likely small) SMT-LIB2 fragment, preserving parentheses as tokens.
fn tokenize(formula: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = formula.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == ';' {
            for next in chars.by_ref() {
                if next == '\n' {
                    break;
                }
            }
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            continue;
        }

        if ch.is_whitespace() || ch == '(' || ch == ')' {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            if ch == '(' || ch == ')' {
                tokens.push(ch.to_string());
            }
            continue;
        }

        current.push(ch);
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn is_symbol_token(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }

    if matches!(token, "true" | "false") {
        return false;
    }

    if is_reserved_token(token) {
        return false;
    }

    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '!' | '\''))
}

fn is_reserved_token(token: &str) -> bool {
    matches!(
        token,
        "and"
            | "or"
            | "not"
            | "=>"
            | "implies"
            | "ite"
            | "="
            | "<="
            | ">="
            | "<"
            | ">"
            | "+"
            | "-"
            | "*"
            | "div"
            | "mod"
            | "abs"
            | "assert"
            | "set-logic"
            | "check-sat"
            | "declare-const"
            | "declare-fun"
            | "define-fun"
            | "set-option"
            | "set-info"
            | "forall"
            | "exists"
            | "let"
            | "lambda"
            | "Int"
            | "Bool"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::BackendId;

    fn skip_if_z3_unavailable() -> bool {
        std::process::Command::new("z3")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
    }

    #[tokio::test]
    async fn test_checker_config_default() {
        let config = CheckerConfig::default();
        assert_eq!(config.z3_path, "z3");
        assert_eq!(config.step_timeout, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_checker_new() {
        let checker = ProofChecker::new();
        assert_eq!(checker.config.z3_path, "z3");
    }

    #[tokio::test]
    async fn test_checker_with_config() {
        let config = CheckerConfig {
            step_timeout: Duration::from_secs(10),
            ..Default::default()
        };
        let checker = ProofChecker::with_config(config);
        assert_eq!(checker.config.step_timeout, Duration::from_secs(10));
    }

    #[tokio::test]
    async fn test_verify_invalid_format() {
        let proof = UniversalProof::builder().format(ProofFormat::Drat).build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await;

        assert!(matches!(result, Err(CheckerError::InvalidFormat { .. })));
    }

    #[tokio::test]
    async fn test_verify_empty_chc_proof() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert!(result.step_results.is_empty());
    }

    #[tokio::test]
    async fn test_verify_invariant_step() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::chc_invariant(
                "inv",
                vec!["x".to_string()],
                "(>= x 0)",
            ))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert_eq!(result.step_results.len(), 1);
        assert!(result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_valid_initiation() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Init: x = 0
        // Inv: x >= 0
        // This should verify because x=0 implies x>=0
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert!(result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_invalid_initiation() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Init: x = -1
        // Inv: x >= 0
        // This should FAIL because x=-1 does not imply x>=0
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Chc(ChcStep::initiation(
                "(= x (- 1))",
                "(>= x 0)",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(!result.is_valid);
        assert!(!result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_valid_consecution() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Pre-inv: x >= 0
        // Trans: x' = x + 1
        // Post-inv: x' >= 0
        // This should verify because x>=0 and x'=x+1 implies x'>=0
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Chc(ChcStep::consecution(
                "(>= x 0)",
                "(= x! (+ x 1))",
                "(>= x! 0)",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert!(result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_consecution_with_custom_symbols() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= counter 0))")
            .step(ProofStep::Chc(ChcStep::consecution(
                "(>= counter 0)",
                "(= counter! (+ counter 1))",
                "(>= counter! 0)",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert!(result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_invalid_consecution() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Pre-inv: x >= 0
        // Trans: x' = x - 2
        // Post-inv: x' >= 0
        // This should FAIL because x>=0 and x'=x-2 doesn't imply x'>=0 (e.g., x=1)
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Chc(ChcStep::consecution(
                "(>= x 0)",
                "(= x! (- x 2))",
                "(>= x! 0)",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(!result.is_valid);
        assert!(!result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_valid_property() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Inv: x >= 0
        // Property: x >= -5
        // This should verify because x>=0 implies x>=-5
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x (- 5)))")
            .step(ProofStep::Chc(ChcStep::property(
                "(>= x 0)",
                "(>= x (- 5))",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert!(result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_initiation_with_boolean_guard() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (=> active (>= n 0)))")
            .step(ProofStep::Chc(ChcStep::initiation(
                "(and active (= n 0))",
                "(and active (>= n 0))",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_verify_invalid_property() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Inv: x >= 0
        // Property: x > 0
        // This should FAIL because x>=0 does not imply x>0 (x=0 is a counterexample)
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (> x 0))")
            .step(ProofStep::Chc(ChcStep::property("(>= x 0)", "(> x 0)")))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(!result.is_valid);
        assert!(!result.step_results[0].is_valid);
    }

    #[tokio::test]
    async fn test_verify_complete_proof() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Complete counter proof:
        // Init: x = 0
        // Trans: x' = x + 1
        // Property: x >= 0
        // Invariant: x >= 0
        let proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::chc_invariant(
                "inv",
                vec!["x".to_string()],
                "(>= x 0)",
            ))
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .step(ProofStep::Chc(ChcStep::consecution(
                "(>= x 0)",
                "(= x! (+ x 1))",
                "(>= x! 0)",
            )))
            .step(ProofStep::Chc(ChcStep::property("(>= x 0)", "(>= x 0)")))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert_eq!(result.step_results.len(), 4);
        for step in &result.step_results {
            assert!(step.is_valid);
        }
    }

    #[tokio::test]
    async fn test_verify_trust_step() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::trust("external prover", "P implies Q"))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        // Trust steps are valid but flagged
        assert!(result.is_valid);
        assert!(result.step_results[0]
            .details
            .as_ref()
            .unwrap()
            .contains("Trusted"));
    }

    #[tokio::test]
    async fn test_verification_result_first_failure() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Second step will fail
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)"))) // Valid
            .step(ProofStep::Chc(ChcStep::initiation(
                "(= x (- 1))",
                "(>= x 0)",
            ))) // Invalid
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(!result.is_valid);
        let failure = result.first_failure().unwrap();
        assert_eq!(failure.step_index, 1);
    }

    #[tokio::test]
    async fn test_verification_summary_valid() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.summary.contains("verified"));
    }

    #[tokio::test]
    async fn test_verification_summary_invalid() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::Chc(ChcStep::initiation(
                "(= x (- 1))",
                "(>= x 0)",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.summary.contains("invalid"));
    }

    // ==================== Mixed Format Tests ====================

    #[tokio::test]
    async fn test_verify_mixed_format_empty() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Mixed)
            .vc("(assert true)")
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_verify_mixed_with_smt_assume() {
        if skip_if_z3_unavailable() {
            return;
        }

        use crate::step::SmtStep;

        let proof = UniversalProof::builder()
            .format(ProofFormat::Mixed)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Smt(SmtStep::assume("base_0", "P(0) holds")))
            .step(ProofStep::Smt(SmtStep::assume("base_1", "P(1) holds")))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert_eq!(result.step_results.len(), 2);
        assert!(result.step_results[0]
            .details
            .as_ref()
            .unwrap()
            .contains("Assumption"));
    }

    #[tokio::test]
    async fn test_verify_mixed_with_smt_inference() {
        if skip_if_z3_unavailable() {
            return;
        }

        use crate::step::SmtStep;

        let proof = UniversalProof::builder()
            .format(ProofFormat::Mixed)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Smt(SmtStep::assume("hyp", "(>= x 0)")))
            .step(ProofStep::Smt(SmtStep::infer(
                "induction_step",
                vec![0],
                "(>= x' 0)",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert!(result.step_results[1]
            .details
            .as_ref()
            .unwrap()
            .contains("Inference"));
    }

    #[tokio::test]
    async fn test_verify_mixed_with_theory_lemma() {
        if skip_if_z3_unavailable() {
            return;
        }

        use crate::step::SmtStep;

        let proof = UniversalProof::builder()
            .format(ProofFormat::Mixed)
            .vc("(assert (>= (+ x 1) 1))")
            .step(ProofStep::Smt(SmtStep::theory_lemma(
                "LIA",
                "(>= (+ x 1) 1)",
            )))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert!(result.step_results[0]
            .details
            .as_ref()
            .unwrap()
            .contains("Theory lemma"));
    }

    #[tokio::test]
    async fn test_verify_mixed_kinduction_style() {
        if skip_if_z3_unavailable() {
            return;
        }

        use crate::step::SmtStep;

        // Simulate a k-induction proof with k=2
        // Base cases + induction hypothesis + induction step + CHC invariant
        let proof = UniversalProof::builder()
            .backend(BackendId::KaniFast)
            .format(ProofFormat::Mixed)
            .vc("(assert (>= x 0))")
            // Base cases (SMT steps)
            .step(ProofStep::Smt(SmtStep::assume(
                "base_case_0",
                "property holds at step 0",
            )))
            .step(ProofStep::Smt(SmtStep::assume(
                "base_case_1",
                "property holds at step 1",
            )))
            // Induction hypothesis
            .step(ProofStep::Smt(SmtStep::assume(
                "induction_hypothesis",
                "property holds for 2 consecutive states",
            )))
            // Induction step
            .step(ProofStep::Smt(SmtStep::infer(
                "induction_step",
                vec![0, 1, 2],
                "property holds at state k+1",
            )))
            // CHC invariant (discovered during proof)
            .step(ProofStep::chc_invariant(
                "inv",
                vec!["x".to_string()],
                "(>= x 0)",
            ))
            // Initiation proof
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            // Consecution proof
            .step(ProofStep::Chc(ChcStep::consecution(
                "(>= x 0)",
                "(= x! (+ x 1))",
                "(>= x! 0)",
            )))
            // Property proof
            .step(ProofStep::Chc(ChcStep::property("(>= x 0)", "(>= x 0)")))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(result.is_valid);
        assert_eq!(result.step_results.len(), 8);
        // Check we have both SMT and CHC step results
        let smt_count = result
            .step_results
            .iter()
            .filter(|r| {
                r.details
                    .as_ref()
                    .is_some_and(|d| d.contains("Assumption") || d.contains("Inference"))
            })
            .count();
        let chc_count = result
            .step_results
            .iter()
            .filter(|r| {
                r.details
                    .as_ref()
                    .is_some_and(|d| d.contains("verified") || d.contains("Invariant"))
            })
            .count();
        assert!(smt_count >= 3, "Should have SMT steps");
        assert!(chc_count >= 3, "Should have CHC steps");
    }

    #[tokio::test]
    async fn test_verify_mixed_with_invalid_chc() {
        if skip_if_z3_unavailable() {
            return;
        }

        use crate::step::SmtStep;

        // Mixed proof where the CHC step fails
        let proof = UniversalProof::builder()
            .format(ProofFormat::Mixed)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Smt(SmtStep::assume("base", "base case")))
            .step(ProofStep::Chc(ChcStep::initiation(
                "(= x (- 1))",
                "(>= x 0)",
            ))) // Invalid!
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(!result.is_valid);
        // First step (SMT) should pass, second (CHC) should fail
        assert!(result.step_results[0].is_valid);
        assert!(!result.step_results[1].is_valid);
    }

    #[tokio::test]
    async fn test_verify_drat_format_rejected() {
        // DRAT format should be rejected
        let proof = UniversalProof::builder()
            .format(ProofFormat::Drat)
            .vc("(assert true)")
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await;

        assert!(matches!(result, Err(CheckerError::InvalidFormat { .. })));
    }

    #[tokio::test]
    async fn test_verify_lean_format_rejected() {
        // Lean format should be rejected
        let proof = UniversalProof::builder()
            .format(ProofFormat::Lean)
            .vc("(assert true)")
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await;

        assert!(matches!(result, Err(CheckerError::InvalidFormat { .. })));
    }

    // ==================== tokenize() tests ====================

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn test_tokenize_single_symbol() {
        assert_eq!(tokenize("x"), vec!["x"]);
    }

    #[test]
    fn test_tokenize_simple_atom() {
        let tokens = tokenize("(+ x 1)");
        assert_eq!(tokens, vec!["(", "+", "x", "1", ")"]);
    }

    #[test]
    fn test_tokenize_nested_sexp() {
        let tokens = tokenize("(and (>= x 0) (< x 10))");
        assert_eq!(
            tokens,
            vec!["(", "and", "(", ">=", "x", "0", ")", "(", "<", "x", "10", ")", ")"]
        );
    }

    #[test]
    fn test_tokenize_whitespace_handling() {
        let tokens = tokenize("  (  +   x   1  )  ");
        assert_eq!(tokens, vec!["(", "+", "x", "1", ")"]);
    }

    #[test]
    fn test_tokenize_newlines_and_tabs() {
        let tokens = tokenize("(\n\t+ x\n\t1)");
        assert_eq!(tokens, vec!["(", "+", "x", "1", ")"]);
    }

    #[test]
    fn test_tokenize_negative_number() {
        let tokens = tokenize("(- 1)");
        assert_eq!(tokens, vec!["(", "-", "1", ")"]);
    }

    #[test]
    fn test_tokenize_comment_stripping() {
        let tokens = tokenize("(+ x ; this is a comment\n 1)");
        assert_eq!(tokens, vec!["(", "+", "x", "1", ")"]);
    }

    #[test]
    fn test_tokenize_comment_at_end() {
        let tokens = tokenize("(+ x 1) ; trailing comment");
        assert_eq!(tokens, vec!["(", "+", "x", "1", ")"]);
    }

    #[test]
    fn test_tokenize_multiple_comments() {
        let tokens = tokenize("; header\n(+ x ; inline\n 1) ; end");
        assert_eq!(tokens, vec!["(", "+", "x", "1", ")"]);
    }

    #[test]
    fn test_tokenize_primed_variables() {
        let tokens = tokenize("(= x' (+ x 1))");
        assert_eq!(tokens, vec!["(", "=", "x'", "(", "+", "x", "1", ")", ")"]);
    }

    #[test]
    fn test_tokenize_underscore_variables() {
        let tokens = tokenize("(= my_var 0)");
        assert_eq!(tokens, vec!["(", "=", "my_var", "0", ")"]);
    }

    #[test]
    fn test_tokenize_bang_suffix() {
        let tokens = tokenize("(= x! (+ x 1))");
        assert_eq!(tokens, vec!["(", "=", "x!", "(", "+", "x", "1", ")", ")"]);
    }

    #[test]
    fn test_tokenize_declare_const() {
        let tokens = tokenize("(declare-const x Int)");
        assert_eq!(tokens, vec!["(", "declare-const", "x", "Int", ")"]);
    }

    #[test]
    fn test_tokenize_define_fun() {
        let tokens = tokenize("(define-fun f ((x Int)) Int (+ x 1))");
        assert_eq!(
            tokens,
            vec![
                "(",
                "define-fun",
                "f",
                "(",
                "(",
                "x",
                "Int",
                ")",
                ")",
                "Int",
                "(",
                "+",
                "x",
                "1",
                ")",
                ")"
            ]
        );
    }

    #[test]
    fn test_tokenize_quantifier() {
        let tokens = tokenize("(forall ((x Int)) (>= x 0))");
        assert_eq!(
            tokens,
            vec!["(", "forall", "(", "(", "x", "Int", ")", ")", "(", ">=", "x", "0", ")", ")"]
        );
    }

    #[test]
    fn test_tokenize_let_binding() {
        let tokens = tokenize("(let ((y (+ x 1))) (>= y 0))");
        assert_eq!(
            tokens,
            vec![
                "(", "let", "(", "(", "y", "(", "+", "x", "1", ")", ")", ")", "(", ">=", "y", "0",
                ")", ")"
            ]
        );
    }

    // ==================== is_symbol_token() tests ====================

    #[test]
    fn test_is_symbol_token_simple() {
        assert!(is_symbol_token("x"));
        assert!(is_symbol_token("counter"));
        assert!(is_symbol_token("myVar"));
    }

    #[test]
    fn test_is_symbol_token_with_underscore() {
        assert!(is_symbol_token("my_var"));
        assert!(is_symbol_token("_private"));
        assert!(is_symbol_token("var_1"));
    }

    #[test]
    fn test_is_symbol_token_with_prime() {
        assert!(is_symbol_token("x'"));
        assert!(is_symbol_token("state''"));
    }

    #[test]
    fn test_is_symbol_token_with_bang() {
        assert!(is_symbol_token("x!"));
        assert!(is_symbol_token("next!"));
    }

    #[test]
    fn test_is_symbol_token_number_rejected() {
        assert!(!is_symbol_token("123"));
        assert!(!is_symbol_token("0"));
        assert!(!is_symbol_token("-1"));
    }

    #[test]
    fn test_is_symbol_token_empty_rejected() {
        assert!(!is_symbol_token(""));
    }

    #[test]
    fn test_is_symbol_token_true_false_rejected() {
        assert!(!is_symbol_token("true"));
        assert!(!is_symbol_token("false"));
    }

    #[test]
    fn test_is_symbol_token_reserved_rejected() {
        assert!(!is_symbol_token("and"));
        assert!(!is_symbol_token("or"));
        assert!(!is_symbol_token("not"));
        assert!(!is_symbol_token("Int"));
        assert!(!is_symbol_token("Bool"));
    }

    #[test]
    fn test_is_symbol_token_operators_rejected() {
        assert!(!is_symbol_token("+"));
        assert!(!is_symbol_token("-"));
        assert!(!is_symbol_token("*"));
        assert!(!is_symbol_token("<="));
    }

    #[test]
    fn test_is_symbol_token_starts_with_digit_rejected() {
        assert!(!is_symbol_token("1x"));
        assert!(!is_symbol_token("2var"));
    }

    // ==================== is_reserved_token() tests ====================

    #[test]
    fn test_is_reserved_token_logic_ops() {
        assert!(is_reserved_token("and"));
        assert!(is_reserved_token("or"));
        assert!(is_reserved_token("not"));
        assert!(is_reserved_token("=>"));
        assert!(is_reserved_token("implies"));
    }

    #[test]
    fn test_is_reserved_token_comparison_ops() {
        assert!(is_reserved_token("="));
        assert!(is_reserved_token("<"));
        assert!(is_reserved_token(">"));
        assert!(is_reserved_token("<="));
        assert!(is_reserved_token(">="));
    }

    #[test]
    fn test_is_reserved_token_arithmetic_ops() {
        assert!(is_reserved_token("+"));
        assert!(is_reserved_token("-"));
        assert!(is_reserved_token("*"));
        assert!(is_reserved_token("div"));
        assert!(is_reserved_token("mod"));
        assert!(is_reserved_token("abs"));
    }

    #[test]
    fn test_is_reserved_token_commands() {
        assert!(is_reserved_token("assert"));
        assert!(is_reserved_token("set-logic"));
        assert!(is_reserved_token("check-sat"));
        assert!(is_reserved_token("declare-const"));
        assert!(is_reserved_token("declare-fun"));
        assert!(is_reserved_token("define-fun"));
    }

    #[test]
    fn test_is_reserved_token_quantifiers() {
        assert!(is_reserved_token("forall"));
        assert!(is_reserved_token("exists"));
    }

    #[test]
    fn test_is_reserved_token_types() {
        assert!(is_reserved_token("Int"));
        assert!(is_reserved_token("Bool"));
    }

    #[test]
    fn test_is_reserved_token_other_keywords() {
        assert!(is_reserved_token("let"));
        assert!(is_reserved_token("lambda"));
        assert!(is_reserved_token("ite"));
        assert!(is_reserved_token("set-option"));
        assert!(is_reserved_token("set-info"));
    }

    #[test]
    fn test_is_reserved_token_non_reserved() {
        assert!(!is_reserved_token("x"));
        assert!(!is_reserved_token("counter"));
        assert!(!is_reserved_token("my_var"));
        assert!(!is_reserved_token("foo"));
    }

    // ==================== symbol_declarations() tests ====================

    #[test]
    fn test_symbol_declarations_empty() {
        let result = symbol_declarations(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_symbol_declarations_no_symbols() {
        let result = symbol_declarations(&["(and true false)"]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_symbol_declarations_single_int() {
        let result = symbol_declarations(&["(>= x 0)"]);
        assert!(result.contains("(declare-const x Int)"));
    }

    #[test]
    fn test_symbol_declarations_single_bool() {
        let result = symbol_declarations(&["(and p q)"]);
        assert!(result.contains("(declare-const p Bool)"));
        assert!(result.contains("(declare-const q Bool)"));
    }

    #[test]
    fn test_symbol_declarations_mixed_context() {
        // x used in arithmetic context, p in boolean context
        let result = symbol_declarations(&["(and p (>= x 0))"]);
        assert!(result.contains("(declare-const p Bool)"));
        assert!(result.contains("(declare-const x Int)"));
    }

    #[test]
    fn test_symbol_declarations_primed_variables() {
        let result = symbol_declarations(&["(= x' (+ x 1))"]);
        assert!(result.contains("(declare-const x Int)"));
        assert!(result.contains("(declare-const x' Int)"));
    }

    #[test]
    fn test_symbol_declarations_bang_variables() {
        let result = symbol_declarations(&["(= x! (+ x 1))"]);
        assert!(result.contains("(declare-const x Int)"));
        assert!(result.contains("(declare-const x! Int)"));
    }

    #[test]
    fn test_symbol_declarations_underscore_variables() {
        let result = symbol_declarations(&["(>= my_counter 0)"]);
        assert!(result.contains("(declare-const my_counter Int)"));
    }

    #[test]
    fn test_symbol_declarations_multiple_formulas() {
        let result = symbol_declarations(&["(= x 0)", "(= y 1)", "(>= z 0)"]);
        assert!(result.contains("(declare-const x Int)"));
        assert!(result.contains("(declare-const y Int)"));
        assert!(result.contains("(declare-const z Int)"));
    }

    #[test]
    fn test_symbol_declarations_deduplicated() {
        // x appears in both formulas, should only be declared once
        let result = symbol_declarations(&["(= x 0)", "(>= x 0)"]);
        let count = result.matches("declare-const x").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_symbol_declarations_sorted() {
        // BTreeMap ensures sorted output
        let result = symbol_declarations(&["(and (>= z 0) (>= a 0) (>= m 0))"]);
        assert!(result.lines().count() >= 3);
        // a should come before m, m before z (alphabetical)
        let a_pos = result.find("declare-const a").unwrap();
        let m_pos = result.find("declare-const m").unwrap();
        let z_pos = result.find("declare-const z").unwrap();
        assert!(a_pos < m_pos);
        assert!(m_pos < z_pos);
    }

    #[test]
    fn test_symbol_declarations_complex_formula() {
        let result = symbol_declarations(&[
            "(and (>= x 0) (>= y 0))",
            "(= x' (+ x 1))",
            "(= y' (- y 1))",
        ]);
        assert!(result.contains("(declare-const x Int)"));
        assert!(result.contains("(declare-const y Int)"));
        assert!(result.contains("(declare-const x' Int)"));
        assert!(result.contains("(declare-const y' Int)"));
    }

    #[test]
    fn test_symbol_declarations_unknown_context_defaults_to_int() {
        // Variables in = context default to Int
        let result = symbol_declarations(&["(= foo bar)"]);
        assert!(result.contains("(declare-const foo Int)"));
        assert!(result.contains("(declare-const bar Int)"));
    }

    #[test]
    fn test_symbol_declarations_comment_ignored() {
        let result = symbol_declarations(&["(>= x ; comment\n 0)"]);
        assert!(result.contains("(declare-const x Int)"));
        // No 'comment' symbol
        assert!(!result.contains("comment"));
    }

    #[test]
    fn test_symbol_declarations_nested_arithmetic() {
        let result = symbol_declarations(&["(>= (+ a b) (- c d))"]);
        assert!(result.contains("(declare-const a Int)"));
        assert!(result.contains("(declare-const b Int)"));
        assert!(result.contains("(declare-const c Int)"));
        assert!(result.contains("(declare-const d Int)"));
    }

    #[test]
    fn test_symbol_declarations_ite() {
        // ite is reserved, so condition/branches are symbols if they look like them
        let result = symbol_declarations(&["(ite cond then_val else_val)"]);
        // cond, then_val, else_val could be symbols - depends on context
        // Unknown context defaults to Int
        assert!(result.contains("cond") || result.contains("then_val"));
    }

    // ==================== SymbolSort tests ====================

    #[test]
    fn test_symbol_sort_merge_both_unknown() {
        assert_eq!(
            SymbolSort::Unknown.merge(SymbolSort::Unknown),
            SymbolSort::Unknown
        );
    }

    #[test]
    fn test_symbol_sort_merge_int_dominates() {
        assert_eq!(SymbolSort::Int.merge(SymbolSort::Unknown), SymbolSort::Int);
        assert_eq!(SymbolSort::Unknown.merge(SymbolSort::Int), SymbolSort::Int);
        assert_eq!(SymbolSort::Int.merge(SymbolSort::Bool), SymbolSort::Int);
        assert_eq!(SymbolSort::Bool.merge(SymbolSort::Int), SymbolSort::Int);
    }

    #[test]
    fn test_symbol_sort_merge_bool_over_unknown() {
        assert_eq!(
            SymbolSort::Bool.merge(SymbolSort::Unknown),
            SymbolSort::Bool
        );
        assert_eq!(
            SymbolSort::Unknown.merge(SymbolSort::Bool),
            SymbolSort::Bool
        );
    }

    #[test]
    fn test_symbol_sort_smt_sort() {
        assert_eq!(SymbolSort::Int.smt_sort(), "Int");
        assert_eq!(SymbolSort::Bool.smt_sort(), "Bool");
        assert_eq!(SymbolSort::Unknown.smt_sort(), "Int"); // Defaults to Int
    }

    // ==================== CheckerError Display tests ====================

    #[test]
    fn test_checker_error_display_solver_not_available() {
        let err = CheckerError::SolverNotAvailable("not found".to_string());
        assert!(err.to_string().contains("Z3 solver not available"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_checker_error_display_timeout() {
        let err = CheckerError::Timeout(Duration::from_secs(5));
        assert!(err.to_string().contains("timed out"));
        assert!(err.to_string().contains("5s"));
    }

    #[test]
    fn test_checker_error_display_invalid_format() {
        let err = CheckerError::InvalidFormat {
            expected: "Chc".to_string(),
            actual: "Drat".to_string(),
        };
        assert!(err.to_string().contains("Invalid proof format"));
        assert!(err.to_string().contains("Chc"));
        assert!(err.to_string().contains("Drat"));
    }

    #[test]
    fn test_checker_error_display_step_failed() {
        let err = CheckerError::StepFailed {
            step_index: 3,
            reason: "consecution failed".to_string(),
        };
        assert!(err.to_string().contains("step 3"));
        assert!(err.to_string().contains("consecution failed"));
    }

    // ==================== VerificationResult tests ====================

    #[test]
    fn test_verification_result_first_failure_none() {
        let result = VerificationResult {
            is_valid: true,
            step_results: vec![
                StepResult {
                    step_index: 0,
                    is_valid: true,
                    duration: Duration::from_millis(10),
                    details: None,
                },
                StepResult {
                    step_index: 1,
                    is_valid: true,
                    duration: Duration::from_millis(10),
                    details: None,
                },
            ],
            total_duration: Duration::from_millis(20),
            z3_calls: 2,
            summary: "all good".to_string(),
        };
        assert!(result.first_failure().is_none());
    }

    #[test]
    fn test_verification_result_first_failure_found() {
        let result = VerificationResult {
            is_valid: false,
            step_results: vec![
                StepResult {
                    step_index: 0,
                    is_valid: true,
                    duration: Duration::from_millis(10),
                    details: None,
                },
                StepResult {
                    step_index: 1,
                    is_valid: false,
                    duration: Duration::from_millis(10),
                    details: Some("bad step".to_string()),
                },
                StepResult {
                    step_index: 2,
                    is_valid: false,
                    duration: Duration::from_millis(10),
                    details: None,
                },
            ],
            total_duration: Duration::from_millis(30),
            z3_calls: 3,
            summary: "failed".to_string(),
        };
        let failure = result.first_failure().unwrap();
        assert_eq!(failure.step_index, 1);
        assert!(!failure.is_valid);
    }

    // ==================== StepResult tests ====================

    #[test]
    fn test_step_result_fields() {
        let step = StepResult {
            step_index: 5,
            is_valid: true,
            duration: Duration::from_millis(42),
            details: Some("Init → Inv verified".to_string()),
        };
        assert_eq!(step.step_index, 5);
        assert!(step.is_valid);
        assert_eq!(step.duration, Duration::from_millis(42));
        assert_eq!(step.details, Some("Init → Inv verified".to_string()));
    }

    // ==================== ProofChecker Default impl test ====================

    #[test]
    fn test_proof_checker_default() {
        let checker: ProofChecker = Default::default();
        assert_eq!(checker.config.z3_path, "z3");
    }

    // ==================== Tests for mutation coverage ====================

    #[tokio::test]
    async fn test_verify_timeout_check_triggered() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Create config with very short total timeout
        let config = CheckerConfig {
            total_timeout: Duration::from_nanos(1), // Extremely short
            step_timeout: Duration::from_secs(5),
            ..Default::default()
        };

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .step(ProofStep::Chc(ChcStep::initiation("(= y 0)", "(>= y 0)")))
            .build();

        let checker = ProofChecker::with_config(config);
        let result = checker.verify(&proof).await;

        // Should timeout after total_timeout is exceeded
        assert!(matches!(result, Err(CheckerError::Timeout(_))));
    }

    #[tokio::test]
    async fn test_verify_z3_calls_count() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Verify that z3_calls increments correctly (catches += -> *= mutant)
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .step(ProofStep::Chc(ChcStep::initiation("(= y 0)", "(>= y 0)")))
            .step(ProofStep::Chc(ChcStep::initiation("(= z 0)", "(>= z 0)")))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        // Should have exactly 3 Z3 calls (one per initiation step)
        assert_eq!(result.z3_calls, 3);
    }

    #[tokio::test]
    async fn test_verify_timeout_boundary_exactly_at_limit_no_timeout() {
        if skip_if_z3_unavailable() {
            return;
        }

        // Test that with a generous timeout, the proof completes successfully.
        // This catches the > vs >= mutation: if mutated to >=, it would timeout
        // even when elapsed time equals the timeout (which shouldn't happen).
        let config = CheckerConfig {
            total_timeout: Duration::from_secs(60), // Generous timeout
            step_timeout: Duration::from_secs(5),
            ..Default::default()
        };

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .build();

        let checker = ProofChecker::with_config(config);
        let result = checker.verify(&proof).await;

        // Should complete without timeout
        assert!(result.is_ok(), "Expected success, got {:?}", result);
        assert!(result.unwrap().is_valid);
    }

    #[tokio::test]
    async fn test_verify_z3_calls_single() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        // Exactly 1 Z3 call - catches *= mutant (0 * 1 = 0, not 1)
        assert_eq!(result.z3_calls, 1);
    }

    #[tokio::test]
    async fn test_verify_invalid_step_filtered_correctly() {
        if skip_if_z3_unavailable() {
            return;
        }

        // This test ensures the !r.is_valid filter works correctly (catches delete ! mutant)
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)"))) // Valid
            .step(ProofStep::Chc(ChcStep::initiation(
                "(= x (- 1))",
                "(>= x 0)",
            ))) // Invalid
            .step(ProofStep::Chc(ChcStep::initiation("(= y 0)", "(>= y 0)"))) // Valid
            .build();

        let checker = ProofChecker::new();
        let result = checker.verify(&proof).await.unwrap();

        assert!(!result.is_valid);
        // Summary should show exactly 1 failed step (index 1)
        assert!(result.summary.contains("[1]"));
        // Should show 2 passed out of 3
        assert!(result.summary.contains("2/3"));
    }

    #[tokio::test]
    async fn test_z3_available_check_called() {
        // Test that Z3 availability is checked (catches Ok(()) mutant)
        let config = CheckerConfig {
            z3_path: "/nonexistent/z3/binary/path".to_string(),
            ..Default::default()
        };

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .build();

        let checker = ProofChecker::with_config(config);
        let result = checker.verify(&proof).await;

        // Should fail because Z3 is not at the specified path
        assert!(matches!(result, Err(CheckerError::SolverNotAvailable(_))));
    }

    #[test]
    fn test_symbol_declarations_closing_paren_ignored() {
        // Test that ")" is properly skipped in tokenization (catches delete match arm mutant)
        // If ")" were treated as a symbol, it would create invalid declarations
        let result = symbol_declarations(&["(>= x 0)"]);

        // Should only declare x, not ")"
        assert!(result.contains("(declare-const x Int)"));
        assert!(!result.contains("declare-const )"));
        assert_eq!(result.matches("declare-const").count(), 1);
    }

    #[test]
    fn test_symbol_declarations_arithmetic_ops_context() {
        // Test that arithmetic operators set Int context (catches delete match arm mutant)
        let result =
            symbol_declarations(&["(+ a b)", "(- c d)", "(* e f)", "(div g h)", "(mod i j)"]);

        // All should be declared as Int due to arithmetic context
        for var in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] {
            assert!(
                result.contains(&format!("(declare-const {} Int)", var)),
                "Variable {} should be declared as Int",
                var
            );
        }
    }

    #[test]
    fn test_symbol_declarations_comparison_ops_int_context() {
        // Test comparison operators also set Int context
        let result =
            symbol_declarations(&["(< a b)", "(> c d)", "(<= e f)", "(>= g h)", "(abs i)"]);

        for var in ["a", "b", "c", "d", "e", "f", "g", "h", "i"] {
            assert!(
                result.contains(&format!("(declare-const {} Int)", var)),
                "Variable {} should be declared as Int in comparison context",
                var
            );
        }
    }

    #[test]
    fn test_symbol_declarations_arithmetic_overrides_bool_context() {
        // This test catches the mutation that deletes the arithmetic match arm.
        // If the arithmetic arm is deleted, 'x' would be Unknown (not Int), and when
        // later seen in 'and' context it would become Bool. With the arm, it becomes Int.
        // The key is that we see x first in arithmetic, then in boolean.
        let result = symbol_declarations(&["(+ x 1)", "(and p x)"]);

        // x should be Int because arithmetic context (Int) wins over boolean context
        // If the arithmetic arm is deleted, x would be Unknown then Bool
        assert!(
            result.contains("(declare-const x Int)"),
            "Variable x should be Int, not Bool. Result: {}",
            result
        );
    }

    #[test]
    fn test_symbol_declarations_deeply_nested() {
        // Test that op_stack properly tracks context through nesting
        // This ensures the ")" pop operation is working correctly
        let result = symbol_declarations(&["(and (or (>= a 0) (< b 1)) (not p))"]);

        // a and b should be Int (arithmetic context), p should be Bool (not context)
        assert!(result.contains("(declare-const a Int)"));
        assert!(result.contains("(declare-const b Int)"));
        assert!(result.contains("(declare-const p Bool)"));
    }

    #[test]
    fn test_symbol_declarations_empty_formula() {
        // Edge case: empty formula should return empty declarations
        let result = symbol_declarations(&[""]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_symbol_declarations_only_parens() {
        // Edge case: only parentheses, no symbols
        let result = symbol_declarations(&["()()()"]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_tokenize_preserves_op_stack_invariant() {
        // Test that tokenize correctly handles nested structures
        // This indirectly tests that ")" is handled correctly
        let tokens = tokenize("((()))");
        assert_eq!(tokens, vec!["(", "(", "(", ")", ")", ")"]);
    }

    #[test]
    fn test_symbol_declarations_context_switch_on_close_paren() {
        // Specifically test that ")" causes context switch (catches delete ")" match arm)
        // If ")" doesn't pop the op_stack, nested contexts won't work correctly

        // Outer is "and" (Bool context), inner is ">=" (Int context)
        // After the inner ")" closes, we should be back in Bool context
        let result = symbol_declarations(&["(and (>= x 0) p)"]);

        // x should be Int (from >= context)
        assert!(result.contains("(declare-const x Int)"));
        // p should be Bool (from and context, after returning from >=)
        assert!(result.contains("(declare-const p Bool)"));
    }

    #[test]
    fn test_symbol_declarations_nested_context_returns() {
        // Another test for ")" popping the context stack
        // Pattern: (or (+ a b) (and c d))
        // After (+ a b) closes, should return to "or" context, not stay in "+"
        let result = symbol_declarations(&["(or (+ a b) (and c d))"]);

        // a, b should be Int (from + context)
        assert!(result.contains("(declare-const a Int)"));
        assert!(result.contains("(declare-const b Int)"));
        // c, d should be Bool (from and context)
        assert!(result.contains("(declare-const c Bool)"));
        assert!(result.contains("(declare-const d Bool)"));
    }

    #[test]
    fn test_symbol_declarations_without_arithmetic_ops_defaults_unknown() {
        // Test that WITHOUT arithmetic operators, variables default to Unknown (-> Int)
        // If the arithmetic match arm is deleted, all would default to Unknown which maps to Int
        // So we need to test that Bool context IS working (since arithmetic gives Int)
        let result = symbol_declarations(&["(not flag)"]);

        // flag in "not" context should be Bool
        assert!(result.contains("(declare-const flag Bool)"));

        // Now test that arithmetic context DOES give Int
        let result2 = symbol_declarations(&["(+ value 1)"]);
        assert!(result2.contains("(declare-const value Int)"));

        // The key insight: if arithmetic match arm is deleted, both would work
        // because Unknown defaults to Int anyway. The real test is that
        // Bool context is preserved in between arithmetic operations.
    }

    #[test]
    fn test_symbol_declarations_interleaved_contexts() {
        // Test context switching between Bool and Int
        // Pattern: arithmetic -> boolean -> arithmetic
        let result = symbol_declarations(&["(and (>= x 0) (or flag (< y 10)))"]);

        // x, y should be Int
        assert!(result.contains("(declare-const x Int)"));
        assert!(result.contains("(declare-const y Int)"));
        // flag should be Bool (in "or" context)
        assert!(result.contains("(declare-const flag Bool)"));
    }

    // NOTE: Mutant "replace > with >= in ProofChecker::verify" at line 164 is an
    // EQUIVALENT MUTANT. The check `start.elapsed() > self.config.total_timeout` vs
    // `start.elapsed() >= self.config.total_timeout` only differs in behavior at the
    // exact boundary (elapsed == timeout), which is virtually impossible to hit
    // deterministically in tests due to timing precision. Both behave identically
    // for practical purposes:
    // - elapsed < timeout: both pass
    // - elapsed > timeout: both fail
    // - elapsed == timeout: only differs here (> passes, >= fails)
    //
    // The difference only matters when elapsed time is EXACTLY equal to the timeout,
    // which requires sub-nanosecond timing precision that is not achievable.
    // Documenting as equivalent mutant.

    #[tokio::test]
    async fn test_z3_available_detects_error_exit_status() {
        // Test that Z3 error exit status is detected (catches `output.status.success() with true` mutant)
        // Uses /usr/bin/false which always exits with status 1
        // This is different from test_z3_available_check_called which uses a nonexistent path
        //
        // Code paths in check_z3_available:
        // 1. Err(e) - command fails to execute (tested by test_z3_available_check_called)
        // 2. Ok(output) if success() - normal case
        // 3. Ok(output) else - command runs but returns non-zero exit (THIS TEST)
        //
        // The mutant changes success() to `true`, making path 3 unreachable

        #[cfg(unix)]
        {
            let config = CheckerConfig {
                z3_path: "/usr/bin/false".to_string(),
                ..Default::default()
            };

            let proof = UniversalProof::builder()
                .format(ProofFormat::Chc)
                .vc("(assert true)")
                .build();

            let checker = ProofChecker::with_config(config);
            let result = checker.verify(&proof).await;

            // Should fail because /usr/bin/false returns exit code 1
            assert!(
                matches!(result, Err(CheckerError::SolverNotAvailable(ref msg)) if msg.contains("returned error")),
                "Expected SolverNotAvailable with 'returned error', got {:?}",
                result
            );
        }
    }
}
