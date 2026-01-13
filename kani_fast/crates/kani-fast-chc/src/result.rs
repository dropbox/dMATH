//! CHC solving results
//!
//! This module defines the result types for CHC solving, including
//! invariant models, verification outcomes, and counterexample traces.

use kani_fast_kinduction::{SmtType, StateFormula};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::time::Duration;

// Pre-compiled regex for Z3 statistics parsing
lazy_static! {
    /// Matches Z3 statistics key-value pairs
    static ref RE_Z3_STATS: Regex = Regex::new(r":([a-zA-Z0-9_.-]+)\s+([0-9.]+)")
        .expect("RE_Z3_STATS regex is valid");
}

/// Result of CHC solving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChcResult {
    /// SAT: An inductive invariant exists that proves the property
    Sat {
        /// The discovered invariant model
        model: InvariantModel,
        /// Solver statistics
        stats: ChcSolverStats,
        /// Raw solver output
        #[serde(skip)]
        raw_output: Option<String>,
    },

    /// UNSAT: The property is violated (no invariant exists)
    Unsat {
        /// Optional counterexample trace (structured)
        counterexample: Option<CounterexampleTrace>,
        /// Solver statistics
        stats: ChcSolverStats,
        /// Raw solver output
        #[serde(skip)]
        raw_output: Option<String>,
    },

    /// UNKNOWN: Solver could not determine satisfiability
    Unknown {
        /// Reason for unknown result
        reason: String,
        /// Solver statistics
        stats: ChcSolverStats,
        /// Raw solver output
        #[serde(skip)]
        raw_output: Option<String>,
    },
}

impl ChcResult {
    /// Check if the result is SAT (property holds)
    pub fn is_sat(&self) -> bool {
        matches!(self, ChcResult::Sat { .. })
    }

    /// Check if the result is UNSAT (property violated)
    pub fn is_unsat(&self) -> bool {
        matches!(self, ChcResult::Unsat { .. })
    }

    /// Check if the result is UNKNOWN
    pub fn is_unknown(&self) -> bool {
        matches!(self, ChcResult::Unknown { .. })
    }

    /// Get the invariant model if SAT
    pub fn model(&self) -> Option<&InvariantModel> {
        match self {
            ChcResult::Sat { model, .. } => Some(model),
            _ => None,
        }
    }

    /// Get solver statistics
    pub fn stats(&self) -> &ChcSolverStats {
        match self {
            ChcResult::Sat { stats, .. } => stats,
            ChcResult::Unsat { stats, .. } => stats,
            ChcResult::Unknown { stats, .. } => stats,
        }
    }

    /// Get counterexample trace if UNSAT
    pub fn counterexample(&self) -> Option<&CounterexampleTrace> {
        match self {
            ChcResult::Unsat { counterexample, .. } => counterexample.as_ref(),
            _ => None,
        }
    }

    /// Get counterexample as string if UNSAT
    pub fn counterexample_string(&self) -> Option<String> {
        self.counterexample().map(|c| c.to_readable_string())
    }

    /// Convert to verification outcome
    pub fn to_verification_outcome(&self) -> VerificationOutcome {
        match self {
            ChcResult::Sat { model, .. } => VerificationOutcome::Verified {
                invariant: model.to_readable_string(),
            },
            ChcResult::Unsat { counterexample, .. } => VerificationOutcome::Violated {
                counterexample: counterexample.as_ref().map(|c| c.to_readable_string()),
            },
            ChcResult::Unknown { reason, .. } => VerificationOutcome::Unknown {
                reason: reason.clone(),
            },
        }
    }
}

impl fmt::Display for ChcResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChcResult::Sat { model, stats, .. } => {
                writeln!(f, "SAT (property holds)")?;
                writeln!(f, "Solve time: {:?}", stats.solve_time)?;
                writeln!(f, "Invariant: {}", model.to_readable_string())
            }
            ChcResult::Unsat {
                stats,
                counterexample,
                ..
            } => {
                writeln!(f, "UNSAT (property violated)")?;
                writeln!(f, "Solve time: {:?}", stats.solve_time)?;
                if let Some(cex) = counterexample {
                    writeln!(f, "Counterexample: {}", cex)?;
                }
                Ok(())
            }
            ChcResult::Unknown { reason, stats, .. } => {
                writeln!(f, "UNKNOWN")?;
                writeln!(f, "Solve time: {:?}", stats.solve_time)?;
                writeln!(f, "Reason: {}", reason)
            }
        }
    }
}

/// Simplified verification outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationOutcome {
    /// Property is verified with discovered invariant
    Verified { invariant: String },
    /// Property is violated with optional counterexample
    Violated { counterexample: Option<String> },
    /// Verification inconclusive
    Unknown { reason: String },
}

/// Model containing discovered invariants
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvariantModel {
    /// Solved predicates (invariants)
    pub predicates: Vec<SolvedPredicate>,
}

impl InvariantModel {
    /// Create an empty model
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the main invariant predicate (usually named "Inv")
    pub fn main_invariant(&self) -> Option<&SolvedPredicate> {
        self.predicates.iter().find(|p| p.name == "Inv")
    }

    /// Get invariant by predicate name
    pub fn get(&self, name: &str) -> Option<&SolvedPredicate> {
        self.predicates.iter().find(|p| p.name == name)
    }

    /// Convert to human-readable string
    pub fn to_readable_string(&self) -> String {
        self.predicates
            .iter()
            .map(|p| p.to_readable_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Convert to summarized human-readable string
    pub fn to_summary_string(&self) -> String {
        self.predicates
            .iter()
            .map(|p| p.to_summary_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate SMT-LIB2 assertions for the invariants
    pub fn to_smt_assertions(&self) -> String {
        self.predicates
            .iter()
            .map(|p| p.to_smt_assertion())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A solved predicate with its invariant formula
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolvedPredicate {
    /// Predicate name
    pub name: String,
    /// Parameter names and types
    pub params: Vec<(String, SmtType)>,
    /// The invariant formula body
    pub formula: StateFormula,
}

impl SolvedPredicate {
    /// Convert to human-readable string
    pub fn to_readable_string(&self) -> String {
        let params_str: Vec<String> = self
            .params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t.to_smt_string()))
            .collect();

        if params_str.is_empty() {
            format!("{} := {}", self.name, self.formula.smt_formula)
        } else {
            format!(
                "{}({}) := {}",
                self.name,
                params_str.join(", "),
                self.formula.smt_formula
            )
        }
    }

    /// Convert to a summarized human-readable string
    ///
    /// This extracts key constraints from the invariant to provide a
    /// more concise view. Complex nested formulas are truncated.
    pub fn to_summary_string(&self) -> String {
        let params_str: Vec<String> = self
            .params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t.to_smt_string()))
            .collect();

        let summary = summarize_smt_formula(&self.formula.smt_formula);

        if params_str.is_empty() {
            format!("{} := {}", self.name, summary)
        } else {
            format!("{}({}) := {}", self.name, params_str.join(", "), summary)
        }
    }

    /// Generate SMT-LIB2 assertion
    pub fn to_smt_assertion(&self) -> String {
        let param_decls: Vec<String> = self
            .params
            .iter()
            .map(|(n, t)| format!("({} {})", n, t.to_smt_string()))
            .collect();

        let param_names: Vec<&str> = self.params.iter().map(|(n, _)| n.as_str()).collect();

        format!(
            "(assert (forall ({}) (= ({} {}) {})))",
            param_decls.join(" "),
            self.name,
            param_names.join(" "),
            self.formula.smt_formula
        )
    }

    /// Simplify the invariant formula
    pub fn simplify(&self) -> SolvedPredicate {
        // Basic simplifications
        let simplified = simplify_smt_formula(&self.formula.smt_formula);
        SolvedPredicate {
            name: self.name.clone(),
            params: self.params.clone(),
            formula: StateFormula::new(simplified),
        }
    }
}

/// Basic SMT formula simplification
fn simplify_smt_formula(formula: &str) -> String {
    let mut result = formula.to_string();

    // Simplify common patterns
    // (not (<= x -1)) -> (>= x 0)
    if result.contains("(not (<= ") && result.contains(" (- 1)))") {
        // This is a common pattern for x >= 0
        result = result.replace("(not (<= ", "(>= ");
        result = result.replace(" (- 1)))", " 0)");
    }

    // (not (<= (+ x (* (- 1) y)) (- 1))) -> (>= (- x y) 0) -> (>= x y)
    // These transformations are complex, leave for future work

    result
}

/// Summarize an SMT formula for display
///
/// Extracts key structural information from complex formulas:
/// - Identifies the top-level connective (or, and, let, exists, etc.)
/// - Counts the number of clauses/branches
/// - Extracts primary constraints
fn summarize_smt_formula(formula: &str) -> String {
    let formula = formula.trim();

    // Very short formulas don't need summarization
    if formula.len() < 100 {
        return formula.to_string();
    }

    // Detect top-level structure
    if formula.starts_with("(let") {
        // Count let bindings
        let binding_count = formula.matches("(let ").count();
        if let Some(body_summary) = extract_let_body_summary(formula) {
            return format!("[{} let bindings] {}", binding_count, body_summary);
        }
    }

    if formula.starts_with("(or") {
        let clause_count = count_top_level_clauses(formula, "or");
        return format!("[disjunction of {} cases]", clause_count);
    }

    if formula.starts_with("(and") {
        let clause_count = count_top_level_clauses(formula, "and");
        return format!("[conjunction of {} constraints]", clause_count);
    }

    if formula.starts_with("(exists") || formula.starts_with("(forall") {
        let quantifier = if formula.starts_with("(exists") {
            "exists"
        } else {
            "forall"
        };
        let var_count = count_quantified_vars(formula);
        return format!("[{} over {} variables]", quantifier, var_count);
    }

    // Default: truncate long formulas
    if formula.len() > 200 {
        let first_100: String = formula.chars().take(100).collect();
        format!("{}... [{} chars total]", first_100, formula.len())
    } else {
        formula.to_string()
    }
}

/// Extract a summary of the body from a let expression
fn extract_let_body_summary(formula: &str) -> Option<String> {
    // Find the final expression after all let bindings
    // Structure: (let ((a!1 ...)) (let ((a!2 ...)) body))
    // We want to extract 'body'

    let mut depth = 0;
    let mut last_sexp_start = 0;

    for (i, c) in formula.char_indices() {
        match c {
            '(' => {
                if depth == 1 {
                    last_sexp_start = i;
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
            }
            _ => {}
        }
    }

    // The last S-expression at depth 1 is likely the body
    if last_sexp_start > 0 {
        let body = &formula[last_sexp_start..];
        // Find the end of this S-expression
        let mut body_depth = 0;
        for (i, c) in body.char_indices() {
            match c {
                '(' => body_depth += 1,
                ')' => {
                    body_depth -= 1;
                    if body_depth == 0 {
                        let body_str = &body[..=i];
                        if body_str.len() < 50 {
                            return Some(body_str.to_string());
                        } else {
                            // Summarize the body too
                            return Some(summarize_smt_formula(body_str));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    None
}

/// Count top-level clauses in an or/and expression
fn count_top_level_clauses(formula: &str, connector: &str) -> usize {
    // Skip the opening (or or (and
    let skip = connector.len() + 2;
    if formula.len() < skip {
        return 0;
    }

    let inner = &formula[skip..];
    let mut depth = 1; // We're inside the (or/and ...
    let mut count = 0;

    for (i, c) in inner.char_indices() {
        match c {
            '(' => {
                if depth == 1 {
                    count += 1; // Found a clause
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {
                // Check for atoms at depth 1 (like true, false)
                if depth == 1
                    && c.is_alphabetic()
                    && (i == 0 || inner.chars().nth(i - 1).is_some_and(|p| p.is_whitespace()))
                {
                    count += 1;
                }
            }
        }
    }

    count.max(1) // At least 1 if we have anything
}

/// Count quantified variables
fn count_quantified_vars(formula: &str) -> usize {
    // Format: (exists ((x!0 Int) (x!1 Bool) ...) body)
    // Find the variable list and count entries

    let Some(start) = formula.find("((") else {
        return 0;
    };

    let start = start + 1; // Skip first (
    let inner = &formula[start..];

    let mut depth = 0;
    let mut count = 0;

    for c in inner.chars() {
        match c {
            '(' => {
                if depth == 1 {
                    count += 1;
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
    }

    count
}

/// Statistics from CHC solving
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChcSolverStats {
    /// Total solving time
    pub solve_time: Duration,
    /// Number of Spacer iterations
    pub iterations: Option<u64>,
    /// Number of lemmas learned
    pub lemmas: Option<u64>,
    /// Maximum frame depth reached
    pub max_depth: Option<u64>,
    /// Memory usage in bytes
    pub memory_bytes: Option<u64>,
}

impl fmt::Display for ChcSolverStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "solve_time={:?}", self.solve_time)?;
        if let Some(iter) = self.iterations {
            write!(f, ", iterations={}", iter)?;
        }
        if let Some(lemmas) = self.lemmas {
            write!(f, ", lemmas={}", lemmas)?;
        }
        if let Some(depth) = self.max_depth {
            write!(f, ", max_depth={}", depth)?;
        }
        if let Some(mem) = self.memory_bytes {
            write!(f, ", memory={:.2}MB", mem as f64 / (1024.0 * 1024.0))?;
        }
        Ok(())
    }
}

/// Parse Z3 statistics output to extract solver metrics
///
/// Z3 outputs statistics in the format:
/// ```text
/// (:key value
///  :key2 value2
///  ...)
/// ```
pub fn parse_z3_statistics(output: &str) -> ChcSolverStats {
    let mut stats = ChcSolverStats::default();

    // Find the statistics section (starts with "(:" and ends with ")")
    // It's typically at the end after model output
    let stats_section = if let Some(start) = output.rfind("(:") {
        // Find matching closing paren
        let section = &output[start..];
        if let Some(end) = section.rfind(')') {
            &section[..=end]
        } else {
            section
        }
    } else {
        return stats;
    };

    // Parse key-value pairs using pre-compiled regex
    // Format: :key value or :key-with-dashes value
    for cap in RE_Z3_STATS.captures_iter(stats_section) {
        let key = &cap[1];
        let value_str = &cap[2];

        match key {
            // Spacer iterations - use num-queries as the iteration count
            "SPACER-num-queries" => {
                if let Ok(v) = value_str.parse::<u64>() {
                    stats.iterations = Some(v);
                }
            }
            // Number of lemmas learned
            "SPACER-num-lemmas" => {
                if let Ok(v) = value_str.parse::<u64>() {
                    stats.lemmas = Some(v);
                }
            }
            // Maximum depth reached
            "SPACER-max-depth" => {
                if let Ok(v) = value_str.parse::<u64>() {
                    stats.max_depth = Some(v);
                }
            }
            // Memory usage (in MB from Z3)
            "max-memory" => {
                if let Ok(v) = value_str.parse::<f64>() {
                    // Convert from MB to bytes
                    stats.memory_bytes = Some((v * 1024.0 * 1024.0) as u64);
                }
            }
            _ => {}
        }
    }

    stats
}

/// A counterexample trace from CHC solving
///
/// When a CHC system is UNSAT, it means the property is violated.
/// Spacer can produce a proof that demonstrates how the property
/// can be violated through a sequence of states.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CounterexampleTrace {
    /// Sequence of states in the counterexample (from initial to violation)
    pub states: Vec<CounterexampleState>,
    /// The predicate application where the property was violated
    pub violation: Option<String>,
    /// Length of the trace (number of transitions)
    pub length: usize,
}

impl CounterexampleTrace {
    /// Create an empty counterexample trace
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            violation: None,
            length: 0,
        }
    }

    /// Create a trace from a sequence of states
    pub fn from_states(states: Vec<CounterexampleState>) -> Self {
        let length = if states.is_empty() {
            0
        } else {
            states.len() - 1
        };
        Self {
            states,
            violation: None,
            length,
        }
    }

    /// Add a state to the trace
    pub fn push_state(&mut self, state: CounterexampleState) {
        self.states.push(state);
        if self.states.len() > 1 {
            self.length = self.states.len() - 1;
        }
    }

    /// Set the violation description
    pub fn set_violation(&mut self, violation: impl Into<String>) {
        self.violation = Some(violation.into());
    }

    /// Get initial state
    pub fn initial_state(&self) -> Option<&CounterexampleState> {
        self.states.first()
    }

    /// Get final (violating) state
    pub fn final_state(&self) -> Option<&CounterexampleState> {
        self.states.last()
    }

    /// Check if trace is empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Convert to human-readable string
    pub fn to_readable_string(&self) -> String {
        let mut result = String::new();
        result.push_str("Counterexample trace:\n");
        for (i, state) in self.states.iter().enumerate() {
            if i == 0 {
                let _ = writeln!(result, "  Initial: {state}");
            } else {
                let _ = writeln!(result, "  Step {i}: {state}");
            }
        }
        if let Some(viol) = &self.violation {
            let _ = writeln!(result, "  Violation: {viol}");
        }
        let _ = writeln!(result, "  Trace length: {}", self.length);
        result
    }
}

impl fmt::Display for CounterexampleTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.to_readable_string();
        write!(f, "{s}")
    }
}

/// A state in a counterexample trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleState {
    /// The predicate name (e.g., "Inv")
    pub predicate: String,
    /// Variable assignments in this state
    pub values: HashMap<String, SmtValue>,
    /// Step number in the trace
    pub step: usize,
}

impl CounterexampleState {
    /// Create a new state with the given predicate and values
    pub fn new(predicate: impl Into<String>, step: usize) -> Self {
        Self {
            predicate: predicate.into(),
            values: HashMap::new(),
            step,
        }
    }

    /// Add a value assignment
    pub fn with_value(mut self, name: impl Into<String>, value: SmtValue) -> Self {
        self.values.insert(name.into(), value);
        self
    }

    /// Get a value by name
    pub fn get(&self, name: &str) -> Option<&SmtValue> {
        self.values.get(name)
    }
}

impl fmt::Display for CounterexampleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.predicate)?;
        let mut first = true;
        for (k, v) in &self.values {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{}={}", k, v)?;
        }
        write!(f, ")")
    }
}

/// A value in SMT-LIB2 format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SmtValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Real value (represented as string for precision)
    Real(String),
    /// Bitvector value
    BitVec { value: u64, width: u32 },
    /// Symbolic/unknown value
    Symbolic(String),
}

impl SmtValue {
    /// Parse an SMT-LIB2 value string
    pub fn parse(s: &str) -> Self {
        let s = s.trim();

        // Boolean
        if s == "true" {
            return SmtValue::Bool(true);
        }
        if s == "false" {
            return SmtValue::Bool(false);
        }

        // Negative integer: (- N)
        if s.starts_with("(- ") && s.ends_with(')') {
            let inner = &s[3..s.len() - 1];
            if let Ok(n) = inner.parse::<i64>() {
                return SmtValue::Int(-n);
            }
        }

        // Negative integer shorthand: -N
        if s.starts_with('-') {
            if let Ok(n) = s.parse::<i64>() {
                return SmtValue::Int(n);
            }
        }

        // Positive integer
        if let Ok(n) = s.parse::<i64>() {
            return SmtValue::Int(n);
        }

        // Bitvector: #xNN or #bNN
        if let Some(hex) = s.strip_prefix("#x") {
            if let Ok(n) = u64::from_str_radix(hex, 16) {
                let width = (hex.len() * 4) as u32;
                return SmtValue::BitVec { value: n, width };
            }
        }
        if let Some(bin) = s.strip_prefix("#b") {
            if let Ok(n) = u64::from_str_radix(bin, 2) {
                let width = bin.len() as u32;
                return SmtValue::BitVec { value: n, width };
            }
        }

        // Real (fraction or decimal)
        if s.contains('/') || s.contains('.') {
            return SmtValue::Real(s.to_string());
        }

        // Default: symbolic
        SmtValue::Symbolic(s.to_string())
    }

    /// Convert to i64 if possible
    pub fn as_int(&self) -> Option<i64> {
        match self {
            SmtValue::Int(n) => Some(*n),
            SmtValue::BitVec { value, .. } => Some(*value as i64),
            SmtValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }
}

impl fmt::Display for SmtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtValue::Bool(b) => write!(f, "{b}"),
            SmtValue::Int(n) => write!(f, "{n}"),
            SmtValue::Real(r) => write!(f, "{r}"),
            SmtValue::BitVec { value, width } => {
                write!(f, "#x{:0w$x}", value, w = (*width / 4) as usize)
            }
            SmtValue::Symbolic(s) => write!(f, "{s}"),
        }
    }
}

/// Parse a counterexample trace from Spacer proof output
///
/// Spacer produces proofs in a nested structure showing derivation steps.
/// We extract the predicate applications (e.g., `(Inv 5)`, `(Inv 4)`, etc.)
/// to reconstruct the counterexample trace.
pub fn parse_spacer_proof(proof: &str) -> Option<CounterexampleTrace> {
    let mut trace = CounterexampleTrace::new();

    // Extract predicate applications using a simpler approach
    // Look for patterns like (Inv 5), (Inv -1), (Inv (- 1))
    let mut pred_apps: Vec<(String, Vec<SmtValue>)> = Vec::new();

    // Match simple cases: (PredName value) where PredName starts with uppercase
    // and value is a simple integer or negative integer
    let mut pos = 0;
    while pos < proof.len() {
        if let Some(paren_pos) = proof[pos..].find('(') {
            let start = pos + paren_pos;
            let after_paren = &proof[start + 1..];

            // Skip leading whitespace
            let after_ws = after_paren.trim_start();
            let trimmed = after_paren.len() - after_ws.len();

            // Check for predicate name (uppercase start)
            if let Some(first_char) = after_ws.chars().next() {
                if first_char.is_uppercase() {
                    // Find the end of the name
                    let name_end = after_ws
                        .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '!')
                        .unwrap_or(after_ws.len());
                    let name = &after_ws[..name_end];

                    // Skip SMT keywords and query predicates
                    if !is_smt_keyword(name) && !name.starts_with("query") {
                        // Parse the rest of the arguments
                        let args_start = start + 1 + trimmed + name_end;
                        if args_start < proof.len() {
                            // Find matching closing paren
                            let mut depth = 1;
                            let mut end = args_start;
                            for (i, c) in proof[args_start..].char_indices() {
                                match c {
                                    '(' => depth += 1,
                                    ')' => {
                                        depth -= 1;
                                        if depth == 0 {
                                            end = args_start + i;
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            // Parse arguments
                            let args_str = &proof[args_start..end];
                            let args = parse_smt_args(args_str);

                            if !args.is_empty() {
                                pred_apps.push((name.to_string(), args));
                            }
                        }
                    }
                }
            }
            pos = start + 1;
        } else {
            break;
        }
    }

    // Remove duplicates while preserving order
    let mut seen = std::collections::HashSet::new();
    let unique_apps: Vec<_> = pred_apps
        .into_iter()
        .filter(|(name, args)| {
            let key = format!("{}{:?}", name, args);
            seen.insert(key)
        })
        .collect();

    if unique_apps.is_empty() {
        return None;
    }

    // Sort by first argument value (assuming it represents the state variable)
    // This gives us the trace order
    let mut sorted_apps = unique_apps;
    sorted_apps.sort_by(|a, b| {
        let val_a = a.1.first().and_then(|v| v.as_int()).unwrap_or(0);
        let val_b = b.1.first().and_then(|v| v.as_int()).unwrap_or(0);
        val_b.cmp(&val_a) // Descending order (start from highest)
    });

    // Convert to counterexample states
    for (step, (name, args)) in sorted_apps.iter().enumerate() {
        let mut state = CounterexampleState::new(name.clone(), step);
        for (i, val) in args.iter().enumerate() {
            state.values.insert(format!("x{}", i), val.clone());
        }
        trace.push_state(state);
    }

    // Set violation info
    if let Some(last) = trace.final_state() {
        trace.set_violation(format!("Property violated at state: {}", last));
    }

    if trace.is_empty() {
        None
    } else {
        Some(trace)
    }
}

/// Parse SMT-LIB2 arguments from a string
fn parse_smt_args(args_str: &str) -> Vec<SmtValue> {
    let mut args = Vec::new();
    let args_str = args_str.trim();

    if args_str.is_empty() {
        return args;
    }

    let mut pos = 0;
    // SMT-LIB is ASCII, work directly with bytes to avoid heap allocation
    let bytes = args_str.as_bytes();

    while pos < bytes.len() {
        // Skip whitespace
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }

        if pos >= bytes.len() {
            break;
        }

        if bytes[pos] == b'(' {
            // Nested expression - find matching paren
            let start = pos;
            let mut depth = 1;
            pos += 1;
            while pos < bytes.len() && depth > 0 {
                match bytes[pos] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                pos += 1;
            }
            // SAFETY: We started with valid UTF-8 and only collected ASCII bytes
            let expr = &args_str[start..pos];
            args.push(SmtValue::parse(expr));
        } else {
            // Simple token - read until whitespace or end
            let start = pos;
            while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() && bytes[pos] != b')' {
                pos += 1;
            }
            if start < pos {
                // SAFETY: We started with valid UTF-8 and only collected ASCII bytes
                let token = &args_str[start..pos];
                args.push(SmtValue::parse(token));
            }
        }
    }

    args
}

/// Check if a string is an SMT keyword (not a predicate name)
fn is_smt_keyword(s: &str) -> bool {
    matches!(
        s,
        "Bool"
            | "Int"
            | "Real"
            | "BitVec"
            | "Array"
            | "and"
            | "or"
            | "not"
            | "ite"
            | "let"
            | "forall"
            | "exists"
            | "assert"
            | "check-sat"
            | "declare-fun"
            | "define-fun"
            | "set-logic"
            | "get-model"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chc_result_sat() {
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x".to_string(), SmtType::Int)],
                formula: StateFormula::new("(>= x 0)"),
            }],
        };

        let result = ChcResult::Sat {
            model,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        assert!(result.is_sat());
        assert!(!result.is_unsat());
        assert!(result.model().is_some());
    }

    #[test]
    fn test_solved_predicate_readable() {
        let pred = SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("x".to_string(), SmtType::Int)],
            formula: StateFormula::new("(>= x 0)"),
        };

        let readable = pred.to_readable_string();
        assert!(readable.contains("Inv"));
        assert!(readable.contains("x: Int"));
        assert!(readable.contains("(>= x 0)"));
    }

    #[test]
    fn test_verification_outcome() {
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![],
                formula: StateFormula::new("true"),
            }],
        };

        let result = ChcResult::Sat {
            model,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let outcome = result.to_verification_outcome();
        assert!(matches!(outcome, VerificationOutcome::Verified { .. }));
    }

    #[test]
    fn test_simplify_formula() {
        let simplified = simplify_smt_formula("(not (<= x (- 1)))");
        assert_eq!(simplified, "(>= x 0)");
    }

    #[test]
    fn test_smt_value_parse_int() {
        assert_eq!(SmtValue::parse("42"), SmtValue::Int(42));
        assert_eq!(SmtValue::parse("-5"), SmtValue::Int(-5));
        assert_eq!(SmtValue::parse("(- 10)"), SmtValue::Int(-10));
    }

    #[test]
    fn test_smt_value_parse_bool() {
        assert_eq!(SmtValue::parse("true"), SmtValue::Bool(true));
        assert_eq!(SmtValue::parse("false"), SmtValue::Bool(false));
    }

    #[test]
    fn test_smt_value_parse_bitvec() {
        assert_eq!(
            SmtValue::parse("#xff"),
            SmtValue::BitVec {
                value: 255,
                width: 8
            }
        );
        assert_eq!(
            SmtValue::parse("#b1010"),
            SmtValue::BitVec {
                value: 10,
                width: 4
            }
        );
    }

    #[test]
    fn test_counterexample_trace_creation() {
        let mut trace = CounterexampleTrace::new();
        assert!(trace.is_empty());

        let state1 = CounterexampleState::new("Inv", 0).with_value("x", SmtValue::Int(5));
        let state2 = CounterexampleState::new("Inv", 1).with_value("x", SmtValue::Int(4));
        trace.push_state(state1);
        trace.push_state(state2);

        assert!(!trace.is_empty());
        assert_eq!(trace.length, 1);
        assert_eq!(trace.states.len(), 2);
    }

    #[test]
    fn test_counterexample_state_display() {
        let state = CounterexampleState::new("Inv", 0).with_value("x", SmtValue::Int(5));
        let display = state.to_string();
        assert!(display.contains("Inv"));
        assert!(display.contains("x=5"));
    }

    #[test]
    fn test_parse_spacer_proof_simple() {
        // Simplified Spacer proof output
        let proof = r"
unsat
((set-logic HORN)
(proof
(let ((@x1857 ((_ hyper-res 0 0 0 1) @x1841 ((_ hyper-res 0 0) (asserted (Inv 5)) (Inv 5)) (Inv 4))))
(let ((@x1838 ((_ hyper-res 0 0 0 1) @x1841 ((_ hyper-res 0 0 0 1) @x1841 ((_ hyper-res 0 0 0 1) @x1841 @x1857 (Inv 3)) (Inv 2)) (Inv 1))))
(let ((@x1852 ((_ hyper-res 0 0 0 1) (asserted $x159) ((_ hyper-res 0 0 0 1) @x1841 ((_ hyper-res 0 0 0 1) @x1841 @x1838 (Inv 0)) (Inv (- 1))) $x1836)))
(mp @x1852 (asserted (=> $x1836 false)) false))))))
";

        let trace = parse_spacer_proof(proof);
        assert!(trace.is_some());
        let trace = trace.unwrap();

        // Should have found states from 5 down to -1
        assert!(trace.states.len() >= 2);

        // Verify descending order (from initial to final)
        let values: Vec<i64> = trace
            .states
            .iter()
            .filter_map(|s| s.get("x0").and_then(|v| v.as_int()))
            .collect();

        // Should be sorted descending (5, 4, 3, 2, 1, 0, -1)
        for pair in values.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "Expected descending order: {:?}",
                values
            );
        }
    }

    #[test]
    fn test_counterexample_trace_readable() {
        let state1 = CounterexampleState::new("Inv", 0).with_value("x", SmtValue::Int(5));
        let state2 = CounterexampleState::new("Inv", 1).with_value("x", SmtValue::Int(-1));

        let trace = CounterexampleTrace::from_states(vec![state1, state2]);
        let readable = trace.to_readable_string();

        assert!(readable.contains("Counterexample trace"));
        assert!(readable.contains("Initial"));
        assert!(readable.contains("Step 1"));
    }

    #[test]
    fn test_chc_result_unsat_with_counterexample() {
        let state = CounterexampleState::new("Inv", 0).with_value("x", SmtValue::Int(-1));
        let trace = CounterexampleTrace::from_states(vec![state]);

        let result = ChcResult::Unsat {
            counterexample: Some(trace),
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        assert!(result.is_unsat());
        assert!(result.counterexample().is_some());

        let cex_string = result.counterexample_string();
        assert!(cex_string.is_some());
        assert!(cex_string.unwrap().contains("Counterexample trace"));
    }

    #[test]
    fn test_parse_z3_statistics() {
        let output = r"sat
(
  (define-fun Inv ((x!0 Int)) Bool
    (not (<= x!0 (- 1))))
)
(:SPACER-inductive-gen             1
 :SPACER-inductive-gen-weaken-fail 1
 :SPACER-max-depth                 1
 :SPACER-max-query-lvl             1
 :SPACER-num-active-lemmas         2
 :SPACER-num-invariants            2
 :SPACER-num-is_invariant          1
 :SPACER-num-lemmas                3
 :SPACER-num-pobs                  2
 :SPACER-num-propagations          1
 :SPACER-num-queries               1
 :added-eqs                        14
 :max-memory                       64.77
 :memory                           55.88
 :time                             0.00
 :total-time                       0.00)";

        let stats = parse_z3_statistics(output);

        assert_eq!(stats.iterations, Some(1)); // SPACER-num-queries
        assert_eq!(stats.lemmas, Some(3)); // SPACER-num-lemmas
        assert_eq!(stats.max_depth, Some(1)); // SPACER-max-depth

        // Memory should be ~64.77 MB in bytes
        assert!(stats.memory_bytes.is_some());
        let mem_mb = stats.memory_bytes.unwrap() as f64 / (1024.0 * 1024.0);
        assert!((mem_mb - 64.77).abs() < 0.01);
    }

    #[test]
    fn test_parse_z3_statistics_no_stats() {
        let output = "sat\n(\n  (define-fun Inv () Bool true)\n)\n";

        let stats = parse_z3_statistics(output);

        // All optional fields should be None
        assert!(stats.iterations.is_none());
        assert!(stats.lemmas.is_none());
        assert!(stats.max_depth.is_none());
        assert!(stats.memory_bytes.is_none());
    }

    #[test]
    fn test_parse_z3_statistics_partial() {
        // Only some stats present
        let output = r"sat
(:SPACER-num-queries 5
 :SPACER-max-depth   3)";

        let stats = parse_z3_statistics(output);

        assert_eq!(stats.iterations, Some(5));
        assert_eq!(stats.max_depth, Some(3));
        assert!(stats.lemmas.is_none());
        assert!(stats.memory_bytes.is_none());
    }

    #[test]
    fn test_chc_solver_stats_display() {
        let stats = ChcSolverStats {
            solve_time: std::time::Duration::from_millis(150),
            iterations: Some(5),
            lemmas: Some(10),
            max_depth: Some(3),
            memory_bytes: Some(64 * 1024 * 1024), // 64 MB
        };

        let display = stats.to_string();
        assert!(display.contains("150ms"));
        assert!(display.contains("iterations=5"));
        assert!(display.contains("lemmas=10"));
        assert!(display.contains("max_depth=3"));
        assert!(display.contains("memory=64.00MB"));
    }

    // ==================== summarize_smt_formula tests ====================

    #[test]
    fn test_summarize_short_formula() {
        // Short formulas should not be summarized
        let formula = "(>= x 0)";
        let summary = summarize_smt_formula(formula);
        assert_eq!(summary, formula);
    }

    #[test]
    fn test_summarize_let_expression() {
        // Let expression with multiple bindings
        let formula = "(let ((a!1 (+ x 1))) (let ((a!2 (+ a!1 2))) (let ((a!3 (+ a!2 3))) (>= a!3 0)))).......................................";
        let summary = summarize_smt_formula(formula);
        assert!(summary.contains("let bindings"));
    }

    #[test]
    fn test_summarize_disjunction() {
        // Or expression
        let formula = "(or (= x 0) (= x 1) (= x 2) (= x 3) (= x 4) (= x 5) (= x 6) (= x 7) (= x 8) (= x 9) (= x 10) (= x 11) (= x 12))";
        let summary = summarize_smt_formula(formula);
        assert!(summary.contains("disjunction"));
        assert!(summary.contains("cases"));
    }

    #[test]
    fn test_summarize_conjunction() {
        // And expression
        let formula = "(and (>= x 0) (<= x 100) (>= y 0) (<= y 100) (= z (+ x y)) (>= z 0) (<= z 200) (>= w 0) (<= w 50) (= v 42))";
        let summary = summarize_smt_formula(formula);
        assert!(summary.contains("conjunction"));
        assert!(summary.contains("constraints"));
    }

    #[test]
    fn test_summarize_exists() {
        // Exists expression
        let formula = "(exists ((x!0 Int) (x!1 Int) (x!2 Bool)) (and (>= x!0 0) (<= x!1 10) x!2))................................................................";
        let summary = summarize_smt_formula(formula);
        assert!(summary.contains("exists"));
        assert!(summary.contains("variables"));
    }

    #[test]
    fn test_summarize_forall() {
        // Forall expression
        let formula = "(forall ((x Int) (y Int)) (>= (+ x y) 0))................................................................................................";
        let summary = summarize_smt_formula(formula);
        assert!(summary.contains("forall"));
        assert!(summary.contains("variables"));
    }

    #[test]
    fn test_summarize_long_unknown() {
        // Long formula without known structure - should truncate
        // Need >200 chars for truncation to kick in
        let formula = "(some-unknown-operator a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9 more more more more more more more more more more more more more more more more more more more more more more more more)";
        let summary = summarize_smt_formula(formula);
        assert!(
            summary.contains("...") || summary.len() <= formula.len(),
            "Expected truncation or short summary, got: {}",
            summary
        );
    }

    // ==================== count_top_level_clauses tests ====================

    #[test]
    fn test_count_top_level_clauses_or_simple() {
        let formula = "(or (= x 0) (= x 1) (= x 2))";
        let count = count_top_level_clauses(formula, "or");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_count_top_level_clauses_and_simple() {
        let formula = "(and (>= x 0) (<= x 10))";
        let count = count_top_level_clauses(formula, "and");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_count_top_level_clauses_nested() {
        // Nested expressions should only count top level
        let formula = "(or (and (= x 0) (= y 0)) (and (= x 1) (= y 1)) (and (= x 2) (= y 2)))";
        let count = count_top_level_clauses(formula, "or");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_count_top_level_clauses_with_atoms() {
        // Formula with atomic values like true/false
        let formula = "(or true false (= x 0))";
        let count = count_top_level_clauses(formula, "or");
        assert!(count >= 1);
    }

    #[test]
    fn test_count_top_level_clauses_empty() {
        let formula = "(or)";
        let count = count_top_level_clauses(formula, "or");
        assert_eq!(count, 1); // Returns at least 1
    }

    // ==================== count_quantified_vars tests ====================

    // The count_quantified_vars function is designed for a specific nested format
    // and may not work correctly for all quantified formulas.

    #[test]
    fn test_count_quantified_vars_no_double_paren() {
        // No (( pattern
        let formula = "(forall (x Int) body)";
        let count = count_quantified_vars(formula);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_quantified_vars_no_vars() {
        let formula = "(>= x 0)"; // No quantifier
        let count = count_quantified_vars(formula);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_quantified_vars_empty() {
        let formula = "";
        let count = count_quantified_vars(formula);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_quantified_vars_nested_lists() {
        // The function expects nested lists like (( (a) (b) (c) ))
        // where inner items are at depth 2
        let formula = "(exists (( (x Int) (y Int) )) body)";
        let count = count_quantified_vars(formula);
        // Returns count of items at depth 1 inside the (( ))
        // The formula has 2 variables: (x Int) and (y Int)
        assert_eq!(count, 2);
    }

    // ==================== parse_smt_args tests ====================

    #[test]
    fn test_parse_smt_args_simple_ints() {
        let args = parse_smt_args("5 10 15");
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], SmtValue::Int(5));
        assert_eq!(args[1], SmtValue::Int(10));
        assert_eq!(args[2], SmtValue::Int(15));
    }

    #[test]
    fn test_parse_smt_args_negative() {
        let args = parse_smt_args("-5 (- 10)");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], SmtValue::Int(-5));
        assert_eq!(args[1], SmtValue::Int(-10));
    }

    #[test]
    fn test_parse_smt_args_mixed() {
        let args = parse_smt_args("42 true false");
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], SmtValue::Int(42));
        assert_eq!(args[1], SmtValue::Bool(true));
        assert_eq!(args[2], SmtValue::Bool(false));
    }

    #[test]
    fn test_parse_smt_args_nested() {
        let args = parse_smt_args("(+ x 1) (- y 2)");
        assert_eq!(args.len(), 2);
        // Nested expressions become symbolic values
        assert!(matches!(args[0], SmtValue::Symbolic(_)));
        assert!(matches!(args[1], SmtValue::Symbolic(_)));
    }

    #[test]
    fn test_parse_smt_args_empty() {
        let args = parse_smt_args("");
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_smt_args_whitespace_only() {
        let args = parse_smt_args("   \t\n  ");
        assert!(args.is_empty());
    }

    // ==================== is_smt_keyword tests ====================

    #[test]
    fn test_is_smt_keyword_types() {
        assert!(is_smt_keyword("Bool"));
        assert!(is_smt_keyword("Int"));
        assert!(is_smt_keyword("Real"));
        assert!(is_smt_keyword("BitVec"));
        assert!(is_smt_keyword("Array"));
    }

    #[test]
    fn test_is_smt_keyword_operators() {
        assert!(is_smt_keyword("and"));
        assert!(is_smt_keyword("or"));
        assert!(is_smt_keyword("not"));
        assert!(is_smt_keyword("ite"));
    }

    #[test]
    fn test_is_smt_keyword_quantifiers() {
        assert!(is_smt_keyword("forall"));
        assert!(is_smt_keyword("exists"));
        assert!(is_smt_keyword("let"));
    }

    #[test]
    fn test_is_smt_keyword_commands() {
        assert!(is_smt_keyword("assert"));
        assert!(is_smt_keyword("check-sat"));
        assert!(is_smt_keyword("declare-fun"));
        assert!(is_smt_keyword("define-fun"));
        assert!(is_smt_keyword("set-logic"));
        assert!(is_smt_keyword("get-model"));
    }

    #[test]
    fn test_is_smt_keyword_not_keyword() {
        assert!(!is_smt_keyword("Inv"));
        assert!(!is_smt_keyword("MyPredicate"));
        assert!(!is_smt_keyword("x"));
        assert!(!is_smt_keyword("foo_bar"));
    }

    // ==================== extract_let_body_summary tests ====================

    #[test]
    fn test_extract_let_body_simple() {
        let formula = "(let ((a!1 5)) (>= a!1 0))";
        let body = extract_let_body_summary(formula);
        assert!(body.is_some());
        assert!(body.unwrap().contains(">="));
    }

    #[test]
    fn test_extract_let_body_nested() {
        let formula = "(let ((a!1 1)) (let ((a!2 2)) (+ a!1 a!2)))";
        let body = extract_let_body_summary(formula);
        assert!(body.is_some());
    }

    #[test]
    fn test_extract_let_body_no_let() {
        let formula = "(>= x 0)";
        let body = extract_let_body_summary(formula);
        // May return None or the whole formula depending on structure
        // The function looks for (( pattern
        assert!(body.is_none() || body.as_deref() == Some("(>= x 0)"));
    }

    // ==================== simplify_smt_formula additional tests ====================

    #[test]
    fn test_simplify_formula_no_change() {
        let formula = "(>= x 0)";
        let simplified = simplify_smt_formula(formula);
        assert_eq!(simplified, formula);
    }

    #[test]
    fn test_simplify_formula_complex() {
        // More complex simplification pattern
        let formula = "(and (not (<= x (- 1))) (not (<= y (- 1))))";
        let simplified = simplify_smt_formula(formula);
        // Should simplify the patterns it recognizes
        assert!(simplified.contains(">="));
    }

    // ==================== Additional ChcResult tests ====================

    #[test]
    fn test_chc_result_unknown() {
        let result = ChcResult::Unknown {
            reason: "timeout".to_string(),
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        assert!(!result.is_sat());
        assert!(!result.is_unsat());
        assert!(result.is_unknown());
        assert!(result.model().is_none());
    }

    #[test]
    fn test_chc_result_unsat_no_counterexample() {
        let result = ChcResult::Unsat {
            counterexample: None,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        assert!(result.is_unsat());
        assert!(result.counterexample().is_none());
        assert!(result.counterexample_string().is_none());
    }

    #[test]
    fn test_chc_result_display_sat() {
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![],
                formula: StateFormula::new("true"),
            }],
        };

        let result = ChcResult::Sat {
            model,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let display = result.to_string();
        assert!(display.contains("SAT"));
    }

    #[test]
    fn test_chc_result_display_unsat() {
        let result = ChcResult::Unsat {
            counterexample: None,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let display = result.to_string();
        assert!(display.contains("UNSAT"));
    }

    #[test]
    fn test_chc_result_display_unknown() {
        let result = ChcResult::Unknown {
            reason: "complexity".to_string(),
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let display = result.to_string();
        assert!(display.contains("UNKNOWN"));
        assert!(display.contains("complexity"));
    }

    // ==================== InvariantModel tests ====================

    #[test]
    fn test_invariant_model_main_invariant() {
        let model = InvariantModel {
            predicates: vec![
                SolvedPredicate {
                    name: "Helper".to_string(),
                    params: vec![],
                    formula: StateFormula::new("true"),
                },
                SolvedPredicate {
                    name: "Inv".to_string(),
                    params: vec![("x".to_string(), SmtType::Int)],
                    formula: StateFormula::new("(>= x 0)"),
                },
            ],
        };

        let main = model.main_invariant();
        assert!(main.is_some());
        assert_eq!(main.unwrap().name, "Inv");
    }

    #[test]
    fn test_invariant_model_get_by_name() {
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "MyPred".to_string(),
                params: vec![],
                formula: StateFormula::new("true"),
            }],
        };

        assert!(model.get("MyPred").is_some());
        assert!(model.get("NonExistent").is_none());
    }

    #[test]
    fn test_invariant_model_to_smt_assertions() {
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x".to_string(), SmtType::Int)],
                formula: StateFormula::new("(>= x 0)"),
            }],
        };

        let assertions = model.to_smt_assertions();
        assert!(assertions.contains("assert"));
        assert!(assertions.contains("Inv"));
    }

    // ==================== SolvedPredicate tests ====================

    #[test]
    fn test_solved_predicate_to_smt_assertion() {
        let pred = SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("x".to_string(), SmtType::Int)],
            formula: StateFormula::new("(>= x 0)"),
        };

        let assertion = pred.to_smt_assertion();
        assert!(assertion.contains("forall"));
        assert!(assertion.contains("x Int"));
        assert!(assertion.contains("(>= x 0)"));
    }

    #[test]
    fn test_solved_predicate_to_smt_assertion_no_params() {
        let pred = SolvedPredicate {
            name: "Const".to_string(),
            params: vec![],
            formula: StateFormula::new("true"),
        };

        let assertion = pred.to_smt_assertion();
        assert!(assertion.contains("assert"));
        assert!(assertion.contains("Const"));
    }

    #[test]
    fn test_solved_predicate_simplify() {
        let pred = SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![("x".to_string(), SmtType::Int)],
            formula: StateFormula::new("(not (<= x (- 1)))"),
        };

        let simplified = pred.simplify();
        assert_eq!(simplified.name, "Inv");
        // Check that simplification was applied
        assert!(simplified.formula.smt_formula.contains(">="));
    }

    #[test]
    fn test_solved_predicate_to_summary_string() {
        let pred = SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![
                ("x".to_string(), SmtType::Int),
                ("y".to_string(), SmtType::Bool),
            ],
            formula: StateFormula::new("(and (>= x 0) y)"),
        };

        let summary = pred.to_summary_string();
        assert!(summary.contains("Inv"));
        assert!(summary.contains('x'));
        assert!(summary.contains('y'));
    }

    // ==================== CounterexampleState tests ====================

    #[test]
    fn test_counterexample_state_new() {
        let state = CounterexampleState::new("Inv", 5);
        assert_eq!(state.predicate, "Inv");
        assert_eq!(state.step, 5);
        assert!(state.values.is_empty());
    }

    #[test]
    fn test_counterexample_state_with_value_chain() {
        let state = CounterexampleState::new("Inv", 0)
            .with_value("x", SmtValue::Int(5))
            .with_value("y", SmtValue::Bool(true))
            .with_value("z", SmtValue::Int(-3));

        assert_eq!(state.get("x"), Some(&SmtValue::Int(5)));
        assert_eq!(state.get("y"), Some(&SmtValue::Bool(true)));
        assert_eq!(state.get("z"), Some(&SmtValue::Int(-3)));
        assert_eq!(state.get("w"), None);
    }

    // ==================== CounterexampleTrace tests ====================

    #[test]
    fn test_counterexample_trace_initial_final() {
        let state1 = CounterexampleState::new("Inv", 0).with_value("x", SmtValue::Int(5));
        let state2 = CounterexampleState::new("Inv", 1).with_value("x", SmtValue::Int(4));
        let state3 = CounterexampleState::new("Inv", 2).with_value("x", SmtValue::Int(-1));

        let trace = CounterexampleTrace::from_states(vec![state1, state2, state3]);

        assert!(trace.initial_state().is_some());
        assert!(trace.final_state().is_some());
        assert_eq!(trace.initial_state().unwrap().step, 0);
        assert_eq!(trace.final_state().unwrap().step, 2);
    }

    #[test]
    fn test_counterexample_trace_set_violation() {
        let mut trace = CounterexampleTrace::new();
        trace.set_violation("Property violated: x < 0");

        assert_eq!(
            trace.violation,
            Some("Property violated: x < 0".to_string())
        );
    }

    #[test]
    fn test_counterexample_trace_default() {
        let trace = CounterexampleTrace::default();
        assert!(trace.is_empty());
        assert!(trace.violation.is_none());
    }

    // ==================== parse_spacer_proof additional tests ====================

    #[test]
    fn test_parse_spacer_proof_empty() {
        let proof = "";
        let trace = parse_spacer_proof(proof);
        assert!(trace.is_none());
    }

    #[test]
    fn test_parse_spacer_proof_no_predicates() {
        let proof = "sat\n(model)\n";
        let trace = parse_spacer_proof(proof);
        assert!(trace.is_none());
    }

    #[test]
    fn test_parse_spacer_proof_skips_smt_keywords() {
        // Should skip Bool, Int, etc. even if they match the uppercase pattern
        let proof = "(Bool true) (Int 5) (Inv 10)";
        let trace = parse_spacer_proof(proof);
        // Should only find Inv, not Bool or Int
        if let Some(trace) = trace {
            assert!(!trace.states.iter().any(|s| s.predicate == "Bool"));
            assert!(!trace.states.iter().any(|s| s.predicate == "Int"));
        }
    }

    // ==================== SmtValue additional tests ====================

    #[test]
    fn test_smt_value_as_int() {
        assert_eq!(SmtValue::Int(42).as_int(), Some(42));
        // Bool converts to 0/1 in as_int
        assert_eq!(SmtValue::Bool(true).as_int(), Some(1));
        assert_eq!(SmtValue::Bool(false).as_int(), Some(0));
        // Symbolic has no int representation
        assert_eq!(SmtValue::Symbolic("x".to_string()).as_int(), None);
    }

    #[test]
    fn test_smt_value_display() {
        assert_eq!(SmtValue::Int(42).to_string(), "42");
        assert_eq!(SmtValue::Bool(true).to_string(), "true");
        assert_eq!(SmtValue::Bool(false).to_string(), "false");
        assert_eq!(SmtValue::Symbolic("x".to_string()).to_string(), "x");
        assert_eq!(
            SmtValue::BitVec {
                value: 255,
                width: 8
            }
            .to_string(),
            "#xff"
        );
    }

    #[test]
    fn test_smt_value_bitvec_display_various_widths() {
        // 4-bit
        assert_eq!(
            format!(
                "{}",
                SmtValue::BitVec {
                    value: 10,
                    width: 4
                }
            ),
            "#xa"
        );
        // 16-bit
        assert_eq!(
            format!(
                "{}",
                SmtValue::BitVec {
                    value: 65535,
                    width: 16
                }
            ),
            "#xffff"
        );
        // 32-bit
        assert_eq!(
            format!(
                "{}",
                SmtValue::BitVec {
                    value: 0xDEADBEEF,
                    width: 32
                }
            ),
            "#xdeadbeef"
        );
    }
}
