//! Spacer solver integration for CHC
//!
//! This module provides integration with Spacer backends (Z3/Z4) for solving
//! Constrained Horn Clauses and extracting inductive invariants.

use crate::clause::ChcSystem;
use crate::result::{
    parse_spacer_proof, parse_z3_statistics, ChcResult, ChcSolverStats, CounterexampleTrace,
    InvariantModel, SolvedPredicate,
};
use kani_fast_kinduction::{SmtType, StateFormula};
use std::fmt;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

use crate::find_executable;

/// Errors from CHC solving
#[derive(Debug, Error)]
pub enum ChcSolverError {
    #[error("{0}")]
    SolverNotFound(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// CHC solving backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChcBackend {
    /// Automatically select (prefers Z4, falls back to Z3)
    Auto,
    /// Use Z3 Spacer
    Z3,
    /// Use Z4 Spacer
    Z4,
}

impl fmt::Display for ChcBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChcBackend::Auto => write!(f, "auto"),
            ChcBackend::Z3 => write!(f, "Z3"),
            ChcBackend::Z4 => write!(f, "Z4"),
        }
    }
}

/// Configuration for CHC solving
#[derive(Debug, Clone)]
pub struct ChcSolverConfig {
    /// Backend to use (auto prefers Z4, then Z3)
    pub backend: ChcBackend,
    /// Maximum time for solving
    pub timeout: Duration,
    /// Use Spacer engine (vs auto or datalog)
    pub use_spacer: bool,
    /// Print statistics
    pub print_stats: bool,
    /// Verbosity level (0-10)
    pub verbosity: u32,
    /// Extract counterexample traces on UNSAT (requires proof generation)
    pub extract_counterexample: bool,
    /// Additional backend options (forwarded to solver)
    pub options: Vec<(String, String)>,
}

impl Default for ChcSolverConfig {
    fn default() -> Self {
        // Allow backend override via environment variable for benchmarking
        let backend = match std::env::var("KANI_FAST_CHC_BACKEND").ok().as_deref() {
            Some("z3" | "Z3") => ChcBackend::Z3,
            Some("z4" | "Z4") => ChcBackend::Z4,
            _ => ChcBackend::Auto,
        };

        Self {
            backend,
            timeout: Duration::from_secs(60),
            use_spacer: true,
            print_stats: false,
            verbosity: 0,
            extract_counterexample: true,
            options: vec![],
        }
    }
}

impl ChcSolverConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_backend(mut self, backend: ChcBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_spacer(mut self, use_spacer: bool) -> Self {
        self.use_spacer = use_spacer;
        self
    }

    pub fn with_stats(mut self, print_stats: bool) -> Self {
        self.print_stats = print_stats;
        self
    }

    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.push((key.into(), value.into()));
        self
    }

    pub fn with_counterexample(mut self, extract: bool) -> Self {
        self.extract_counterexample = extract;
        self
    }
}

struct SolverBinary {
    backend: ChcBackend,
    path: String,
}

/// CHC solver using Spacer backends (Z3/Z4)
pub struct ChcSolver {
    /// Selected backend binary
    binary: SolverBinary,
    /// Solver configuration
    config: ChcSolverConfig,
}

impl ChcSolver {
    /// Create a new CHC solver
    pub fn new(config: ChcSolverConfig) -> Result<Self, ChcSolverError> {
        let binary = Self::find_backend(config.backend)?;
        Ok(Self { binary, config })
    }

    /// Create with default configuration
    pub fn default_solver() -> Result<Self, ChcSolverError> {
        Self::new(ChcSolverConfig::default())
    }

    /// Discover the backend binary to use
    fn find_backend(preference: ChcBackend) -> Result<SolverBinary, ChcSolverError> {
        match preference {
            ChcBackend::Z3 => Self::locate_backend(ChcBackend::Z3),
            ChcBackend::Z4 => Self::locate_backend(ChcBackend::Z4),
            ChcBackend::Auto => Self::locate_backend(ChcBackend::Z4)
                .or_else(|_| Self::locate_backend(ChcBackend::Z3))
                .map_err(|_| {
                    ChcSolverError::SolverNotFound(
                        "No CHC solver found (looked for Z4, Z3)".to_string(),
                    )
                }),
        }
    }

    fn locate_backend(backend: ChcBackend) -> Result<SolverBinary, ChcSolverError> {
        let names = match backend {
            ChcBackend::Z3 => ["z3", "Z3"],
            ChcBackend::Z4 => ["z4", "Z4"],
            ChcBackend::Auto => unreachable!(),
        };

        for name in names {
            if let Some(path) = find_executable(name) {
                return Ok(SolverBinary {
                    backend,
                    path: path.to_string_lossy().to_string(),
                });
            }
        }

        Err(ChcSolverError::SolverNotFound(format!(
            "{} not found in PATH",
            backend
        )))
    }

    /// Solve a CHC system
    pub async fn solve(&self, system: &ChcSystem) -> Result<ChcResult, ChcSolverError> {
        let smt2 = system.to_smt2();
        self.solve_smt2(&smt2).await
    }

    /// Return the active backend
    pub fn backend(&self) -> ChcBackend {
        self.binary.backend
    }

    /// Solve from SMT-LIB2 string
    pub async fn solve_smt2(&self, smt2: &str) -> Result<ChcResult, ChcSolverError> {
        let start = Instant::now();

        let mut cmd = Command::new(&self.binary.path);

        // Z3 and Z4 have different CLI interfaces
        match self.binary.backend {
            ChcBackend::Z3 | ChcBackend::Auto => {
                // Z3 requires explicit SMT-LIB2 and Spacer options
                cmd.arg("-smt2").arg("-in");

                // Set engine
                if self.config.use_spacer {
                    cmd.arg("fp.engine=spacer");
                }

                // Timeout in milliseconds
                let timeout_ms = self.config.timeout.as_millis();
                if timeout_ms > 0 {
                    cmd.arg(format!("-t:{}", timeout_ms));
                }

                // Statistics
                if self.config.print_stats {
                    cmd.arg("-st");
                }

                // Verbosity
                if self.config.verbosity > 0 {
                    cmd.arg(format!("-v:{}", self.config.verbosity));
                }

                // Additional options
                for (key, value) in &self.config.options {
                    cmd.arg(format!("{}={}", key, value));
                }
            }
            ChcBackend::Z4 => {
                // Z4 supports stdin - no special flags needed, reads SMT-LIB2 from stdin
                if self.config.verbosity > 0 {
                    cmd.arg("--verbose");
                }
                // Z4 doesn't support timeout flag yet - we use tokio timeout wrapper
            }
        }

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!(
            "Running {:?} CHC solver binary `{}`: {:?}",
            self.binary.backend, self.binary.path, cmd
        );

        let mut child = cmd.spawn().map_err(|e| {
            ChcSolverError::ExecutionFailed(format!(
                "{:?} failed to start: {}",
                self.binary.backend, e
            ))
        })?;

        // Prepare input - add proof generation if counterexample extraction is enabled
        let input = if self.config.extract_counterexample {
            // Insert (set-option :produce-proofs true) after set-logic
            // and add (get-proof) for unsat cases
            self.prepare_input_with_proof(smt2)
        } else {
            smt2.to_string()
        };

        // Write SMT2 to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(input.as_bytes()).await.map_err(|e| {
                ChcSolverError::ExecutionFailed(format!(
                    "{:?} failed to receive input: {}",
                    self.binary.backend, e
                ))
            })?;
        }

        // Wait for result with timeout
        let result = timeout(self.config.timeout, child.wait_with_output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                debug!("{:?} stdout: {}", self.binary.backend, stdout);
                if !stderr.is_empty() {
                    debug!("{:?} stderr: {}", self.binary.backend, stderr);
                }

                self.parse_result(&stdout, &stderr, duration)
            }
            Ok(Err(e)) => Err(ChcSolverError::ExecutionFailed(format!(
                "{:?} failed while waiting for output: {}",
                self.binary.backend, e
            ))),
            Err(_) => {
                warn!(
                    "{:?} CHC solver timed out after {:?}",
                    self.binary.backend, self.config.timeout
                );
                Err(ChcSolverError::Timeout(self.config.timeout))
            }
        }
    }

    /// Prepare input with proof generation enabled
    fn prepare_input_with_proof(&self, smt2: &str) -> String {
        let mut result = String::new();

        // Check if set-logic is present
        if let Some(pos) = smt2.find("(set-logic") {
            // Find end of set-logic command
            let after_set_logic = &smt2[pos..];
            if let Some(end) = after_set_logic.find(')') {
                let end_pos = pos + end + 1;
                result.push_str(&smt2[..end_pos]);
                result.push_str("\n(set-option :produce-proofs true)\n");
                result.push_str(&smt2[end_pos..]);
            } else {
                result.push_str(smt2);
            }
        } else {
            // No set-logic, add option at start
            result.push_str("(set-option :produce-proofs true)\n");
            result.push_str(smt2);
        }

        // Replace (get-model) with (get-proof) for unsat, or add both
        // Actually, we want get-model for sat and get-proof for unsat
        // Since we don't know ahead of time, we'll request both via check-sat-using
        // For now, we add (get-proof) if it's not there
        if !result.contains("(get-proof)") {
            // Add get-proof after check-sat if present
            if let Some(check_sat_pos) = result.rfind("(check-sat)") {
                let after_check = check_sat_pos + "(check-sat)".len();
                let mut new_result = result[..after_check].to_string();
                new_result.push_str("\n(get-proof)");
                // Keep get-model if it was there
                if let Some(rest_start) = result[after_check..].find('(') {
                    new_result.push_str(&result[after_check + rest_start..]);
                }
                result = new_result;
            }
        }

        result
    }

    /// Solve from file
    pub async fn solve_file(&self, path: &Path) -> Result<ChcResult, ChcSolverError> {
        let smt2 = tokio::fs::read_to_string(path).await?;
        self.solve_smt2(&smt2).await
    }

    /// Parse solver output into ChcResult
    fn parse_result(
        &self,
        stdout: &str,
        stderr: &str,
        duration: Duration,
    ) -> Result<ChcResult, ChcSolverError> {
        // Parse Spacer statistics if enabled
        let mut stats = if self.config.print_stats {
            parse_z3_statistics(stdout)
        } else {
            ChcSolverStats::default()
        };
        stats.solve_time = duration;

        // Check for sat/unsat
        let stdout_trimmed = stdout.trim();

        if stdout_trimmed.starts_with("sat") || stdout_trimmed.contains("\nsat") {
            // SAT means property holds - extract invariant model
            let model = self.parse_model(stdout)?;
            Ok(ChcResult::Sat {
                model,
                stats,
                raw_output: Some(stdout.to_string()),
            })
        } else if stdout_trimmed.starts_with("unsat") || stdout_trimmed.contains("\nunsat") {
            // UNSAT means property is violated - extract counterexample if available
            let counterexample = self.parse_counterexample(stdout);
            Ok(ChcResult::Unsat {
                counterexample,
                stats,
                raw_output: Some(stdout.to_string()),
            })
        } else if stdout.contains("unknown") {
            let reason = self.extract_reason(stdout, stderr);
            Ok(ChcResult::Unknown {
                reason,
                stats,
                raw_output: Some(stdout.to_string()),
            })
        } else if stderr.contains("error") || stdout.contains("error") {
            Err(ChcSolverError::ParseError(format!(
                "{:?} error: {}{}",
                self.binary.backend, stdout, stderr
            )))
        } else {
            Ok(ChcResult::Unknown {
                reason: format!(
                    "Unexpected output: {}",
                    stdout.lines().next().unwrap_or("empty")
                ),
                stats,
                raw_output: Some(stdout.to_string()),
            })
        }
    }

    /// Parse invariant model from solver output
    fn parse_model(&self, output: &str) -> Result<InvariantModel, ChcSolverError> {
        let mut predicates = Vec::new();

        // Find model section
        let Some(model_start) = output.find("(model") else {
            // Try to parse define-fun directly (some Z3 versions output without model wrapper)
            return self.parse_define_funs(output);
        };

        let model_section = &output[model_start..];

        // Parse each define-fun in the model
        let mut depth = 0;
        let mut current_def_start = None;

        for (i, c) in model_section.char_indices() {
            match c {
                '(' => {
                    if depth == 1 && model_section[i..].starts_with("(define-fun") {
                        current_def_start = Some(i);
                    }
                    depth += 1;
                }
                ')' => {
                    depth -= 1;
                    if let Some(start) = current_def_start.filter(|_| depth == 1) {
                        let def = &model_section[start..=i];
                        if let Some(pred) = self.parse_define_fun(def) {
                            predicates.push(pred);
                        }
                        current_def_start = None;
                    }
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }

        Ok(InvariantModel { predicates })
    }

    /// Parse define-fun statements directly
    fn parse_define_funs(&self, output: &str) -> Result<InvariantModel, ChcSolverError> {
        let mut predicates = Vec::new();

        let mut depth = 0;
        let mut def_start = None;
        let mut in_outer_model = false;

        for (i, c) in output.char_indices() {
            match c {
                '(' => {
                    // Check if this starts a define-fun
                    if output[i..].starts_with("(define-fun") && def_start.is_none() {
                        def_start = Some(i);
                    } else if depth == 0 && !output[i..].starts_with("(define-fun") {
                        // This might be the outer model wrapper
                        in_outer_model = true;
                    }
                    depth += 1;
                }
                ')' => {
                    depth -= 1;
                    if let Some(start) = def_start {
                        // Count parens within the define-fun
                        let def_str = &output[start..=i];
                        let def_depth: i32 = def_str
                            .chars()
                            .map(|c| match c {
                                '(' => 1,
                                ')' => -1,
                                _ => 0,
                            })
                            .sum();

                        if def_depth == 0 {
                            if let Some(pred) = self.parse_define_fun(def_str) {
                                predicates.push(pred);
                            }
                            def_start = None;
                        }
                    }
                    if depth == 0 && in_outer_model {
                        in_outer_model = false;
                    }
                }
                _ => {}
            }
        }

        Ok(InvariantModel { predicates })
    }

    /// Parse a single define-fun into a SolvedPredicate
    fn parse_define_fun(&self, def: &str) -> Option<SolvedPredicate> {
        // Format: (define-fun Name ((arg1 Type1) ...) Bool body)
        let def = def.trim();
        if !def.starts_with("(define-fun ") {
            return None;
        }

        // Find predicate name
        let after_define = &def[12..]; // Skip "(define-fun "
        let name_end = after_define.find(|c: char| c.is_whitespace() || c == '(')?;
        let name = after_define[..name_end].trim().to_string();

        // Find parameter list - it starts with (( and ends with ))
        let params_list_start = after_define.find("((")?;
        let params_section = &after_define[params_list_start..];

        // Find the end of the parameter list (matching closing parens)
        let mut depth = 0;
        let mut params_list_end = 0;
        let mut params = Vec::new();
        let mut param_start = None;

        for (i, c) in params_section.char_indices() {
            match c {
                '(' => {
                    if depth == 1 {
                        param_start = Some(i + 1);
                    }
                    depth += 1;
                }
                ')' => {
                    depth -= 1;
                    if let Some(start) = param_start.filter(|_| depth == 1) {
                        let param_str = params_section[start..i].trim();
                        if let Some((pname, ptype)) = self.parse_param(param_str) {
                            params.push((pname, ptype));
                        }
                        param_start = None;
                    }
                    if depth == 0 {
                        params_list_end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        // After params list, we expect whitespace, return type (Bool), whitespace, then body
        let after_params = &params_section[params_list_end..];
        let after_params_trimmed = after_params.trim_start();

        // Skip "Bool" return type
        if !after_params_trimmed.starts_with("Bool") {
            return None;
        }
        let after_bool = &after_params_trimmed[4..].trim_start();

        // The body is a balanced S-expression; extract it
        let body = extract_balanced_sexp(after_bool)?;

        // The body is the invariant formula
        let formula = StateFormula::new(body);

        Some(SolvedPredicate {
            name,
            params,
            formula,
        })
    }

    /// Parse a parameter binding like "x Int" or "bits (_ BitVec 32)"
    fn parse_param(&self, s: &str) -> Option<(String, SmtType)> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        // Find the first whitespace to separate name from type
        let first_space = s.find(char::is_whitespace)?;
        let name = s[..first_space].to_string();
        let type_str = s[first_space..].trim();

        if name.is_empty() || type_str.is_empty() {
            return None;
        }

        let smt_type = match type_str {
            "Bool" => SmtType::Bool,
            "Int" => SmtType::Int,
            "Real" => SmtType::Real,
            s if s.starts_with("(_ BitVec") && s.ends_with(')') => {
                // Parse "(_ BitVec N)" format
                let inner = s
                    .trim_start_matches("(_ BitVec")
                    .trim_end_matches(')')
                    .trim();
                let width: u32 = inner.parse().ok()?;
                SmtType::BitVec(width)
            }
            _ => return None,
        };

        Some((name, smt_type))
    }

    /// Parse counterexample from unsat result (proof output)
    fn parse_counterexample(&self, output: &str) -> Option<CounterexampleTrace> {
        // Parse the Spacer proof output to extract the counterexample trace
        // The proof shows the derivation of the violation through predicate applications
        if self.config.extract_counterexample {
            parse_spacer_proof(output)
        } else {
            None
        }
    }
}

/// Extract a balanced S-expression from the start of a string
fn extract_balanced_sexp(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // If it doesn't start with '(', it's an atom - find its end
    if !s.starts_with('(') {
        let end = s
            .find(|c: char| c.is_whitespace() || c == ')')
            .unwrap_or(s.len());
        return Some(s[..end].to_string());
    }

    // Find matching closing paren
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[..=i].to_string());
                }
            }
            _ => {}
        }
    }

    // Unbalanced - return what we have
    Some(s.to_string())
}

impl ChcSolver {
    /// Extract reason for unknown result
    fn extract_reason(&self, stdout: &str, stderr: &str) -> String {
        if let Some(start) = stdout.find("(:reason-unknown") {
            let rest = &stdout[start..];
            if let Some(end) = rest.find(')') {
                return rest[..=end].to_string();
            }
        }

        if stderr.contains("timeout") {
            return "timeout".to_string();
        }

        if stderr.contains("canceled") {
            return "canceled".to_string();
        }

        "unknown".to_string()
    }
}

/// Verify a property using CHC solving
pub async fn verify_chc(
    system: &ChcSystem,
    config: &ChcSolverConfig,
) -> Result<ChcResult, ChcSolverError> {
    let solver = ChcSolver::new(config.clone())?;
    solver.solve(system).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::encode_simple_loop;

    fn has_solver() -> bool {
        find_executable("z4").is_some() || find_executable("z3").is_some()
    }

    fn dummy_solver() -> ChcSolver {
        ChcSolver {
            binary: super::SolverBinary {
                backend: ChcBackend::Z3,
                path: "z3".to_string(),
            },
            config: ChcSolverConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_solver_creation() {
        if !has_solver() {
            return;
        }
        let solver = ChcSolver::default_solver();
        assert!(solver.is_ok());
    }

    #[tokio::test]
    async fn test_solve_simple_counter() {
        if !has_solver() {
            return;
        }

        let system = encode_simple_loop("x", SmtType::Int, "(= x 0)", "(= x' (+ x 1))", "(>= x 0)");

        let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(10));
        let result = verify_chc(&system, &config).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(
            result.is_sat(),
            "Expected SAT (property holds), got: {:?}",
            result
        );

        // Check that we got an invariant
        if let ChcResult::Sat {
            model, raw_output, ..
        } = result
        {
            assert!(
                !model.predicates.is_empty(),
                "Expected invariant predicates, raw output: {:?}",
                raw_output
            );
        }
    }

    #[tokio::test]
    async fn test_solve_violated_property() {
        if !has_solver() {
            return;
        }

        // Counter starting at 5, decrementing - will eventually be negative
        let system = encode_simple_loop("x", SmtType::Int, "(= x 5)", "(= x' (- x 1))", "(>= x 0)");

        let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(10));
        let result = verify_chc(&system, &config).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(
            result.is_unsat(),
            "Expected UNSAT (property violated), got: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_solve_with_spacer() {
        if !has_solver() {
            return;
        }

        let smt2 = r"
(set-logic HORN)
(declare-fun Inv (Int Int) Bool)
(assert (forall ((x Int) (y Int)) (=> (and (= x 0) (= y 0)) (Inv x y))))
(assert (forall ((x Int) (y Int) (x1 Int) (y1 Int))
  (=> (and (Inv x y) (= x1 (+ x 1)) (or (= y1 (+ y 1)) (= y1 y))) (Inv x1 y1))))
(assert (forall ((x Int) (y Int)) (=> (and (Inv x y) (not (>= x y))) false)))
(check-sat)
(get-model)
";

        let config = ChcSolverConfig::new()
            .with_spacer(true)
            .with_timeout(Duration::from_secs(10));

        let solver = ChcSolver::new(config).unwrap();
        let result = solver.solve_smt2(smt2).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_sat(), "Expected SAT, got: {:?}", result);
    }

    #[test]
    fn test_parse_model() {
        let solver = dummy_solver();

        let output = r"sat
(model
  (define-fun Inv ((x!0 Int)) Bool
    (not (<= x!0 (- 1))))
)
";

        let model = solver.parse_model(output);
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.predicates.len(), 1);
        assert_eq!(model.predicates[0].name, "Inv");
    }

    #[tokio::test]
    async fn test_counterexample_extraction() {
        if !has_solver() {
            return;
        }

        // Counter starting at 5, decrementing - will eventually be negative
        // This should produce a counterexample trace: 5 -> 4 -> 3 -> 2 -> 1 -> 0 -> -1
        let system = encode_simple_loop("x", SmtType::Int, "(= x 5)", "(= x' (- x 1))", "(>= x 0)");

        let config = ChcSolverConfig::new()
            .with_timeout(Duration::from_secs(10))
            .with_counterexample(true);
        let result = verify_chc(&system, &config).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_unsat(), "Expected UNSAT, got: {:?}", result);

        // Check counterexample was extracted
        if let Some(cex) = result.counterexample() {
            assert!(!cex.is_empty(), "Counterexample trace should not be empty");
            assert!(
                !cex.states.is_empty(),
                "Expected at least one state in trace"
            );

            // Check violation message is set
            assert!(cex.violation.is_some());
        }
    }

    #[test]
    fn test_prepare_input_with_proof() {
        let solver = ChcSolver {
            binary: super::SolverBinary {
                backend: ChcBackend::Z3,
                path: "z3".to_string(),
            },
            config: ChcSolverConfig::default().with_counterexample(true),
        };

        let input = r"(set-logic HORN)
(declare-fun Inv (Int) Bool)
(check-sat)
(get-model)
";

        let prepared = solver.prepare_input_with_proof(input);

        // Should contain produce-proofs option
        assert!(prepared.contains("(set-option :produce-proofs true)"));

        // Should contain get-proof
        assert!(prepared.contains("(get-proof)"));

        // set-logic should come before produce-proofs
        let set_logic_pos = prepared.find("(set-logic").unwrap();
        let produce_proofs_pos = prepared.find(":produce-proofs").unwrap();
        assert!(set_logic_pos < produce_proofs_pos);
    }

    #[tokio::test]
    async fn test_statistics_extraction() {
        if !has_solver() {
            return;
        }

        let system = encode_simple_loop("x", SmtType::Int, "(= x 0)", "(= x' (+ x 1))", "(>= x 0)");

        // Enable statistics output
        let config = ChcSolverConfig::new()
            .with_timeout(Duration::from_secs(10))
            .with_stats(true);
        let solver = ChcSolver::new(config).unwrap();
        let result = solver.solve(&system).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_sat(), "Expected SAT, got: {:?}", result);

        let stats = result.stats();

        // Solve time should always be set
        assert!(stats.solve_time > Duration::ZERO);

        // Z3 provides detailed statistics, Z4 doesn't (yet)
        // With Z3 we expect iterations and max_depth
        // With Z4 we just check that solve_time is reasonable
        if matches!(solver.backend(), ChcBackend::Z3) {
            assert!(
                stats.iterations.is_some() || stats.max_depth.is_some(),
                "Expected Z3 Spacer statistics to be extracted, got: {:?}",
                stats
            );
        }
    }

    // ==================== Unit tests for ChcBackend ====================

    #[test]
    fn test_chc_backend_display_auto() {
        assert_eq!(ChcBackend::Auto.to_string(), "auto");
    }

    #[test]
    fn test_chc_backend_display_z3() {
        assert_eq!(ChcBackend::Z3.to_string(), "Z3");
    }

    #[test]
    fn test_chc_backend_display_z4() {
        assert_eq!(ChcBackend::Z4.to_string(), "Z4");
    }

    #[test]
    fn test_chc_backend_equality() {
        assert_eq!(ChcBackend::Auto, ChcBackend::Auto);
        assert_eq!(ChcBackend::Z3, ChcBackend::Z3);
        assert_eq!(ChcBackend::Z4, ChcBackend::Z4);
        assert_ne!(ChcBackend::Z3, ChcBackend::Z4);
    }

    // ==================== Unit tests for ChcSolverConfig ====================

    #[test]
    fn test_chc_solver_config_default() {
        let config = ChcSolverConfig::default();
        // Backend depends on KANI_FAST_CHC_BACKEND env var (Auto if not set)
        let expected_backend = match std::env::var("KANI_FAST_CHC_BACKEND").ok().as_deref() {
            Some("z3" | "Z3") => ChcBackend::Z3,
            Some("z4" | "Z4") => ChcBackend::Z4,
            _ => ChcBackend::Auto,
        };
        assert!(
            matches!(config.backend, b if b == expected_backend),
            "Expected {:?}, got {:?}",
            expected_backend,
            config.backend
        );
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(config.use_spacer);
        assert!(!config.print_stats);
        assert_eq!(config.verbosity, 0);
        assert!(config.extract_counterexample);
        assert!(config.options.is_empty());
    }

    #[test]
    fn test_chc_solver_config_new() {
        let config = ChcSolverConfig::new();
        // Backend depends on KANI_FAST_CHC_BACKEND env var (Auto if not set)
        let expected_backend = match std::env::var("KANI_FAST_CHC_BACKEND").ok().as_deref() {
            Some("z3" | "Z3") => ChcBackend::Z3,
            Some("z4" | "Z4") => ChcBackend::Z4,
            _ => ChcBackend::Auto,
        };
        assert!(
            matches!(config.backend, b if b == expected_backend),
            "Expected {:?}, got {:?}",
            expected_backend,
            config.backend
        );
    }

    #[test]
    fn test_chc_solver_config_with_backend() {
        let config = ChcSolverConfig::new().with_backend(ChcBackend::Z3);
        assert!(matches!(config.backend, ChcBackend::Z3));

        let config = ChcSolverConfig::new().with_backend(ChcBackend::Z4);
        assert!(matches!(config.backend, ChcBackend::Z4));
    }

    #[test]
    fn test_chc_solver_config_with_timeout() {
        let config = ChcSolverConfig::new().with_timeout(Duration::from_secs(120));
        assert_eq!(config.timeout, Duration::from_secs(120));

        let config = ChcSolverConfig::new().with_timeout(Duration::from_millis(500));
        assert_eq!(config.timeout, Duration::from_millis(500));
    }

    #[test]
    fn test_chc_solver_config_with_spacer() {
        let config = ChcSolverConfig::new().with_spacer(false);
        assert!(!config.use_spacer);

        let config = ChcSolverConfig::new().with_spacer(true);
        assert!(config.use_spacer);
    }

    #[test]
    fn test_chc_solver_config_with_stats() {
        let config = ChcSolverConfig::new().with_stats(true);
        assert!(config.print_stats);

        let config = ChcSolverConfig::new().with_stats(false);
        assert!(!config.print_stats);
    }

    #[test]
    fn test_chc_solver_config_with_option() {
        let config = ChcSolverConfig::new().with_option("fp.engine", "spacer");
        assert_eq!(config.options.len(), 1);
        assert_eq!(
            config.options[0],
            ("fp.engine".to_string(), "spacer".to_string())
        );
    }

    #[test]
    fn test_chc_solver_config_with_multiple_options() {
        let config = ChcSolverConfig::new()
            .with_option("fp.engine", "spacer")
            .with_option("fp.xform.inline_eager", "false");
        assert_eq!(config.options.len(), 2);
    }

    #[test]
    fn test_chc_solver_config_with_counterexample() {
        let config = ChcSolverConfig::new().with_counterexample(false);
        assert!(!config.extract_counterexample);

        let config = ChcSolverConfig::new().with_counterexample(true);
        assert!(config.extract_counterexample);
    }

    #[test]
    fn test_chc_solver_config_builder_chain() {
        let config = ChcSolverConfig::new()
            .with_backend(ChcBackend::Z3)
            .with_timeout(Duration::from_secs(30))
            .with_spacer(true)
            .with_stats(true)
            .with_counterexample(true)
            .with_option("key", "value");

        assert!(matches!(config.backend, ChcBackend::Z3));
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.use_spacer);
        assert!(config.print_stats);
        assert!(config.extract_counterexample);
        assert_eq!(config.options.len(), 1);
    }

    // ==================== Unit tests for extract_balanced_sexp ====================

    #[test]
    fn test_extract_balanced_sexp_atom() {
        assert_eq!(extract_balanced_sexp("true"), Some("true".to_string()));
        assert_eq!(extract_balanced_sexp("false"), Some("false".to_string()));
        assert_eq!(extract_balanced_sexp("42"), Some("42".to_string()));
        assert_eq!(extract_balanced_sexp("x"), Some("x".to_string()));
    }

    #[test]
    fn test_extract_balanced_sexp_atom_with_trailing() {
        assert_eq!(extract_balanced_sexp("true)"), Some("true".to_string()));
        assert_eq!(extract_balanced_sexp("x y z"), Some("x".to_string()));
    }

    #[test]
    fn test_extract_balanced_sexp_simple() {
        assert_eq!(
            extract_balanced_sexp("(+ 1 2)"),
            Some("(+ 1 2)".to_string())
        );
    }

    #[test]
    fn test_extract_balanced_sexp_nested() {
        assert_eq!(
            extract_balanced_sexp("(+ (- x 1) (* y 2))"),
            Some("(+ (- x 1) (* y 2))".to_string())
        );
    }

    #[test]
    fn test_extract_balanced_sexp_deeply_nested() {
        assert_eq!(
            extract_balanced_sexp("(and (or (= x 0) (> x 1)) (< x 10))"),
            Some("(and (or (= x 0) (> x 1)) (< x 10))".to_string())
        );
    }

    #[test]
    fn test_extract_balanced_sexp_with_whitespace() {
        assert_eq!(
            extract_balanced_sexp("  (+ 1 2)  "),
            Some("(+ 1 2)".to_string())
        );
    }

    #[test]
    fn test_extract_balanced_sexp_empty() {
        assert_eq!(extract_balanced_sexp(""), None);
        assert_eq!(extract_balanced_sexp("   "), None);
    }

    #[test]
    fn test_extract_balanced_sexp_unbalanced() {
        // Unbalanced - returns what we have
        let result = extract_balanced_sexp("(+ 1 2");
        assert!(result.is_some());
        assert!(result.unwrap().starts_with("(+ 1 2"));
    }

    // ==================== Unit tests for ChcSolverError ====================

    #[test]
    fn test_chc_solver_error_solver_not_found() {
        let error = ChcSolverError::SolverNotFound("z3 not found".to_string());
        assert_eq!(error.to_string(), "z3 not found");
    }

    #[test]
    fn test_chc_solver_error_execution_failed() {
        let error = ChcSolverError::ExecutionFailed("process crashed".to_string());
        assert_eq!(error.to_string(), "Execution failed: process crashed");
    }

    #[test]
    fn test_chc_solver_error_timeout() {
        let error = ChcSolverError::Timeout(Duration::from_secs(60));
        assert_eq!(error.to_string(), "Timeout after 60s");
    }

    #[test]
    fn test_chc_solver_error_parse_error() {
        let error = ChcSolverError::ParseError("unexpected token".to_string());
        assert_eq!(error.to_string(), "Parse error: unexpected token");
    }

    #[test]
    fn test_chc_solver_error_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error = ChcSolverError::IoError(io_error);
        assert!(error.to_string().contains("file not found"));
    }

    #[test]
    fn test_chc_solver_error_debug() {
        let error = ChcSolverError::Timeout(Duration::from_secs(30));
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("Timeout"));
    }

    // ==================== Unit tests for parse_param ====================

    #[test]
    fn test_parse_param_int() {
        let solver = dummy_solver();
        let result = solver.parse_param("x Int");
        assert!(result.is_some());
        let (name, ty) = result.unwrap();
        assert_eq!(name, "x");
        assert!(matches!(ty, SmtType::Int));
    }

    #[test]
    fn test_parse_param_bool() {
        let solver = dummy_solver();
        let result = solver.parse_param("flag Bool");
        assert!(result.is_some());
        let (name, ty) = result.unwrap();
        assert_eq!(name, "flag");
        assert!(matches!(ty, SmtType::Bool));
    }

    #[test]
    fn test_parse_param_real() {
        let solver = dummy_solver();
        let result = solver.parse_param("value Real");
        assert!(result.is_some());
        let (name, ty) = result.unwrap();
        assert_eq!(name, "value");
        assert!(matches!(ty, SmtType::Real));
    }

    #[test]
    fn test_parse_param_bitvec_32() {
        let solver = dummy_solver();
        let result = solver.parse_param("bits (_ BitVec 32)");
        assert!(result.is_some());
        let (name, ty) = result.unwrap();
        assert_eq!(name, "bits");
        assert!(matches!(ty, SmtType::BitVec(32)));
    }

    #[test]
    fn test_parse_param_bitvec_64() {
        let solver = dummy_solver();
        let result = solver.parse_param("addr (_ BitVec 64)");
        assert!(result.is_some());
        let (name, ty) = result.unwrap();
        assert_eq!(name, "addr");
        assert!(matches!(ty, SmtType::BitVec(64)));
    }

    #[test]
    fn test_parse_param_bitvec_8() {
        let solver = dummy_solver();
        let result = solver.parse_param("byte (_ BitVec 8)");
        assert!(result.is_some());
        let (name, ty) = result.unwrap();
        assert_eq!(name, "byte");
        assert!(matches!(ty, SmtType::BitVec(8)));
    }

    #[test]
    fn test_parse_param_bitvec_1() {
        let solver = dummy_solver();
        let result = solver.parse_param("flag (_ BitVec 1)");
        assert!(result.is_some());
        let (name, ty) = result.unwrap();
        assert_eq!(name, "flag");
        assert!(matches!(ty, SmtType::BitVec(1)));
    }

    #[test]
    fn test_parse_param_invalid_single_token() {
        let solver = dummy_solver();
        let result = solver.parse_param("x");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_param_invalid_too_many_tokens() {
        let solver = dummy_solver();
        let result = solver.parse_param("x Int extra");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_param_invalid_unknown_type() {
        let solver = dummy_solver();
        let result = solver.parse_param("x UnknownType");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_param_empty() {
        let solver = dummy_solver();
        let result = solver.parse_param("");
        assert!(result.is_none());
    }

    // ==================== Unit tests for extract_reason ====================

    #[test]
    fn test_extract_reason_unknown_with_reason() {
        let solver = dummy_solver();
        let stdout = "unknown\n(:reason-unknown \"incomplete\")";
        let reason = solver.extract_reason(stdout, "");
        assert_eq!(reason, "(:reason-unknown \"incomplete\")");
    }

    #[test]
    fn test_extract_reason_timeout_in_stderr() {
        let solver = dummy_solver();
        let reason = solver.extract_reason("unknown", "timeout occurred");
        assert_eq!(reason, "timeout");
    }

    #[test]
    fn test_extract_reason_canceled_in_stderr() {
        let solver = dummy_solver();
        let reason = solver.extract_reason("unknown", "operation canceled by user");
        assert_eq!(reason, "canceled");
    }

    #[test]
    fn test_extract_reason_no_reason() {
        let solver = dummy_solver();
        let reason = solver.extract_reason("unknown", "");
        assert_eq!(reason, "unknown");
    }

    #[test]
    fn test_extract_reason_empty_output() {
        let solver = dummy_solver();
        let reason = solver.extract_reason("", "");
        assert_eq!(reason, "unknown");
    }

    // ==================== Unit tests for parse_define_fun ====================

    #[test]
    fn test_parse_define_fun_simple() {
        let solver = dummy_solver();
        let def = "(define-fun Inv ((x Int)) Bool true)";
        let result = solver.parse_define_fun(def);
        assert!(result.is_some());
        let pred = result.unwrap();
        assert_eq!(pred.name, "Inv");
        assert_eq!(pred.params.len(), 1);
        assert_eq!(pred.params[0].0, "x");
        assert!(matches!(pred.params[0].1, SmtType::Int));
        assert_eq!(pred.formula.smt_formula, "true");
    }

    #[test]
    fn test_parse_define_fun_with_body() {
        let solver = dummy_solver();
        let def = "(define-fun Inv ((x Int)) Bool (>= x 0))";
        let result = solver.parse_define_fun(def);
        assert!(result.is_some());
        let pred = result.unwrap();
        assert_eq!(pred.name, "Inv");
        assert_eq!(pred.formula.smt_formula, "(>= x 0)");
    }

    #[test]
    fn test_parse_define_fun_multiple_params() {
        let solver = dummy_solver();
        let def = "(define-fun Inv ((x Int) (y Int)) Bool (and (>= x 0) (<= y x)))";
        let result = solver.parse_define_fun(def);
        assert!(result.is_some());
        let pred = result.unwrap();
        assert_eq!(pred.name, "Inv");
        assert_eq!(pred.params.len(), 2);
        assert_eq!(pred.params[0].0, "x");
        assert_eq!(pred.params[1].0, "y");
    }

    #[test]
    fn test_parse_define_fun_complex_body() {
        let solver = dummy_solver();
        let def = "(define-fun Inv ((x!0 Int)) Bool (not (<= x!0 (- 1))))";
        let result = solver.parse_define_fun(def);
        assert!(result.is_some());
        let pred = result.unwrap();
        assert_eq!(pred.name, "Inv");
        assert_eq!(pred.params[0].0, "x!0");
        assert!(pred.formula.smt_formula.contains("not"));
    }

    #[test]
    fn test_parse_define_fun_invalid_not_define_fun() {
        let solver = dummy_solver();
        let def = "(declare-fun Inv (Int) Bool)";
        let result = solver.parse_define_fun(def);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_define_fun_invalid_not_bool_return() {
        let solver = dummy_solver();
        let def = "(define-fun foo ((x Int)) Int x)";
        let result = solver.parse_define_fun(def);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_define_fun_empty_params() {
        let solver = dummy_solver();
        let def = "(define-fun Const (()) Bool true)";
        let result = solver.parse_define_fun(def);
        assert!(result.is_some());
        let pred = result.unwrap();
        assert_eq!(pred.name, "Const");
        assert!(pred.params.is_empty());
    }

    // ==================== Unit tests for parse_define_funs ====================

    #[test]
    fn test_parse_define_funs_single() {
        let solver = dummy_solver();
        let output = "(define-fun Inv ((x Int)) Bool (>= x 0))";
        let result = solver.parse_define_funs(output);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.predicates.len(), 1);
    }

    #[test]
    fn test_parse_define_funs_multiple() {
        let solver = dummy_solver();
        let output = r"
(define-fun Inv1 ((x Int)) Bool (>= x 0))
(define-fun Inv2 ((y Int)) Bool (<= y 10))
";
        let result = solver.parse_define_funs(output);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.predicates.len(), 2);
    }

    #[test]
    fn test_parse_define_funs_with_noise() {
        let solver = dummy_solver();
        let output = r"sat
(define-fun Inv ((x Int)) Bool (>= x 0))
some other text
";
        let result = solver.parse_define_funs(output);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.predicates.len(), 1);
    }

    #[test]
    fn test_parse_define_funs_empty() {
        let solver = dummy_solver();
        let output = "sat\n";
        let result = solver.parse_define_funs(output);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert!(model.predicates.is_empty());
    }

    // ==================== Unit tests for parse_model ====================

    #[test]
    fn test_parse_model_with_wrapper() {
        let solver = dummy_solver();
        let output = r"sat
(model
  (define-fun Inv ((x Int)) Bool (>= x 0))
)
";
        let result = solver.parse_model(output);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.predicates.len(), 1);
        assert_eq!(model.predicates[0].name, "Inv");
    }

    #[test]
    fn test_parse_model_without_wrapper() {
        let solver = dummy_solver();
        let output = r"sat
(define-fun Inv ((x Int)) Bool (>= x 0))
";
        let result = solver.parse_model(output);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.predicates.len(), 1);
    }

    #[test]
    fn test_parse_model_multiple_predicates() {
        let solver = dummy_solver();
        let output = r"sat
(model
  (define-fun Inv1 ((x Int)) Bool (>= x 0))
  (define-fun Inv2 ((y Int)) Bool (<= y 100))
)
";
        let result = solver.parse_model(output);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.predicates.len(), 2);
    }

    // ==================== Unit tests for prepare_input_with_proof ====================

    #[test]
    fn test_prepare_input_no_set_logic() {
        let solver = ChcSolver {
            binary: SolverBinary {
                backend: ChcBackend::Z3,
                path: "z3".to_string(),
            },
            config: ChcSolverConfig::default().with_counterexample(true),
        };

        let input = "(declare-fun x () Int)\n(check-sat)";
        let prepared = solver.prepare_input_with_proof(input);
        assert!(prepared.starts_with("(set-option :produce-proofs true)"));
    }

    #[test]
    fn test_prepare_input_already_has_get_proof() {
        let solver = ChcSolver {
            binary: SolverBinary {
                backend: ChcBackend::Z3,
                path: "z3".to_string(),
            },
            config: ChcSolverConfig::default().with_counterexample(true),
        };

        let input = "(set-logic HORN)\n(check-sat)\n(get-proof)";
        let prepared = solver.prepare_input_with_proof(input);
        // Should not add duplicate get-proof
        let count = prepared.matches("(get-proof)").count();
        assert_eq!(count, 1);
    }

    // ==================== Unit tests for parse_result ====================

    #[test]
    fn test_parse_result_sat() {
        let solver = dummy_solver();
        let result = solver.parse_result(
            "sat\n(model\n(define-fun Inv ((x Int)) Bool true)\n)",
            "",
            Duration::from_millis(100),
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_sat());
    }

    #[test]
    fn test_parse_result_unsat() {
        let solver = dummy_solver();
        let result = solver.parse_result("unsat\n", "", Duration::from_millis(100));
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_parse_result_unknown() {
        let solver = dummy_solver();
        let result = solver.parse_result("unknown\n", "", Duration::from_millis(100));
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(matches!(result, ChcResult::Unknown { .. }));
    }

    #[test]
    fn test_parse_result_error_in_output() {
        let solver = dummy_solver();
        let result = solver.parse_result("", "error: invalid syntax", Duration::from_millis(100));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_result_unexpected_output() {
        let solver = dummy_solver();
        let result = solver.parse_result("something unexpected", "", Duration::from_millis(100));
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(matches!(result, ChcResult::Unknown { .. }));
    }

    #[test]
    fn test_parse_result_sat_with_newline_prefix() {
        let solver = dummy_solver();
        let result = solver.parse_result(
            "some warnings\nsat\n(model)",
            "",
            Duration::from_millis(100),
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_sat());
    }

    #[test]
    fn test_parse_result_unsat_with_newline_prefix() {
        let solver = dummy_solver();
        let result = solver.parse_result("some output\nunsat", "", Duration::from_millis(100));
        assert!(result.is_ok());
        assert!(result.unwrap().is_unsat());
    }
}
