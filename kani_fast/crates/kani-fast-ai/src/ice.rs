//! ICE Learning - Iterative Counterexample-guided Inductive synthesis
//!
//! ICE learning uses three types of examples:
//! - **Implication**: Pairs (s, s') where s -> s' is a valid transition
//! - **Counterexample**: States that violate the property
//! - **Example**: States that must satisfy the invariant
//!
//! The algorithm iteratively refines invariant candidates based on
//! examples that show where current candidates fail.

use kani_fast_kinduction::{Property, SmtType, StateFormula, StateVariable, TransitionSystem};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;
use std::process::Command;

use crate::AiError;

// Pre-compiled regexes for predicate evaluation (compiled once, reused)
lazy_static! {
    /// Matches: (>= var N) where N is an integer
    static ref RE_GE: Regex = Regex::new(r"^\(>=\s+(\w+)\s+(-?\d+)\)$").unwrap();
    /// Matches: (<= var N) where N is an integer
    static ref RE_LE: Regex = Regex::new(r"^\(<=\s+(\w+)\s+(-?\d+)\)$").unwrap();
    /// Matches: (>= var (- N)) for negative bounds
    static ref RE_GE_NEG: Regex = Regex::new(r"^\(>=\s+(\w+)\s+\(-\s*(\d+)\)\)$").unwrap();
    /// Matches: (<= var1 var2) two-variable comparison
    static ref RE_LE_VAR: Regex = Regex::new(r"^\(<=\s+(\w+)\s+(\w+)\)$").unwrap();
    /// Matches: (>= var1 var2) two-variable comparison
    static ref RE_GE_VAR: Regex = Regex::new(r"^\(>=\s+(\w+)\s+(\w+)\)$").unwrap();
    /// Matches: (= var N) for extracting variable assignments
    static ref RE_EQ: Regex = Regex::new(r"\(=\s+(\w+)\s+(-?\d+)\)").unwrap();
}

/// Type alias for induction counterexample: (pre-state, post-state)
type InductionCounterexample = (HashMap<String, i64>, HashMap<String, i64>);

/// Configuration for ICE learning
#[derive(Debug, Clone)]
pub struct IceConfig {
    /// Maximum learning iterations
    pub max_iterations: usize,
    /// Timeout per SMT query (seconds)
    pub smt_timeout_secs: u64,
    /// Use decision tree learning
    pub use_decision_tree: bool,
    /// Maximum clause size for conjunctive invariants
    pub max_clause_size: usize,
}

impl Default for IceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            smt_timeout_secs: 5,
            use_decision_tree: true,
            max_clause_size: 5,
        }
    }
}

/// An example for ICE learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Variable assignments
    pub values: HashMap<String, i64>,
    /// Kind of example
    pub kind: ExampleKind,
}

impl Example {
    /// Create a positive example (must satisfy invariant)
    pub fn positive(values: HashMap<String, i64>) -> Self {
        Self {
            values,
            kind: ExampleKind::Positive,
        }
    }

    /// Create a negative example (must not satisfy invariant)
    pub fn negative(values: HashMap<String, i64>) -> Self {
        Self {
            values,
            kind: ExampleKind::Negative,
        }
    }

    /// Create an implication example
    pub fn implication(pre: HashMap<String, i64>, post: HashMap<String, i64>) -> Self {
        Self {
            values: pre,
            kind: ExampleKind::Implication { post },
        }
    }

    /// Check if this example is positive
    pub fn is_positive(&self) -> bool {
        matches!(self.kind, ExampleKind::Positive)
    }

    /// Check if this example is negative
    pub fn is_negative(&self) -> bool {
        matches!(self.kind, ExampleKind::Negative)
    }
}

/// Kind of ICE example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExampleKind {
    /// Must satisfy the invariant (reachable state)
    Positive,
    /// Must not satisfy the invariant (violating state)
    Negative,
    /// Implication: if pre satisfies, post must satisfy
    Implication { post: HashMap<String, i64> },
}

/// Result of ICE learning
#[derive(Debug, Clone)]
pub struct IceResult {
    /// Discovered invariant (if found)
    pub invariant: Option<StateFormula>,
    /// Examples collected during learning
    pub examples: Vec<Example>,
    /// Number of iterations
    pub iterations: usize,
    /// Whether learning converged
    pub converged: bool,
}

/// ICE learning engine
pub struct IceEngine {
    config: IceConfig,
    /// Collected examples
    examples: Vec<Example>,
    /// Candidate predicates
    predicates: Vec<CandidatePredicate>,
}

impl IceEngine {
    /// Create a new ICE engine
    pub fn new(config: IceConfig) -> Self {
        Self {
            config,
            examples: Vec::new(),
            predicates: Vec::new(),
        }
    }

    /// Add an example to the learning set
    pub fn add_example(&mut self, example: Example) {
        self.examples.push(example);
    }

    /// Run ICE learning to find an invariant
    pub fn learn(
        &mut self,
        system: &TransitionSystem,
        property: &Property,
    ) -> Result<IceResult, AiError> {
        self.examples.clear();
        self.generate_candidate_predicates(system);

        // Initialize with initial state as positive example
        if let Some(init_example) = self.extract_initial_example(system) {
            self.examples.push(init_example);
        }

        for iteration in 0..self.config.max_iterations {
            // Try to find a consistent invariant
            if let Some(invariant) = self.find_consistent_invariant(system)? {
                // Check if it's inductive
                match self.check_inductiveness(&invariant, system, property)? {
                    InductivenessResult::Inductive => {
                        return Ok(IceResult {
                            invariant: Some(invariant),
                            examples: self.examples.clone(),
                            iterations: iteration + 1,
                            converged: true,
                        });
                    }
                    InductivenessResult::InitFails(cex) => {
                        // Add as positive example (reachable from init)
                        self.examples.push(Example::positive(cex));
                    }
                    InductivenessResult::InductionFails { pre, post } => {
                        // Add as implication example
                        self.examples.push(Example::implication(pre, post));
                    }
                    InductivenessResult::PropertyFails(cex) => {
                        // Add as negative example
                        self.examples.push(Example::negative(cex));
                    }
                }
            } else {
                // No consistent invariant found with current predicates
                // Could add more predicates here
                break;
            }
        }

        Ok(IceResult {
            invariant: None,
            examples: self.examples.clone(),
            iterations: self.config.max_iterations,
            converged: false,
        })
    }

    /// Verify that an invariant is correct
    pub fn verify_invariant(
        &self,
        invariant: &StateFormula,
        system: &TransitionSystem,
        property: &Property,
    ) -> Result<bool, AiError> {
        match self.check_inductiveness(invariant, system, property)? {
            InductivenessResult::Inductive => Ok(true),
            _ => Ok(false),
        }
    }

    /// Get a counterexample for a candidate invariant
    pub fn get_counterexample(
        &self,
        invariant: &StateFormula,
        system: &TransitionSystem,
        property: &Property,
    ) -> Result<Option<Example>, AiError> {
        match self.check_inductiveness(invariant, system, property)? {
            InductivenessResult::Inductive => Ok(None),
            InductivenessResult::InitFails(cex) => Ok(Some(Example::positive(cex))),
            InductivenessResult::InductionFails { pre, post } => {
                Ok(Some(Example::implication(pre, post)))
            }
            InductivenessResult::PropertyFails(cex) => Ok(Some(Example::negative(cex))),
        }
    }

    /// Generate candidate predicates from the system
    fn generate_candidate_predicates(&mut self, system: &TransitionSystem) {
        self.predicates.clear();

        // Generate predicates for each integer variable
        for var in &system.variables {
            if var.smt_type == SmtType::Int {
                // Non-negativity
                self.predicates.push(CandidatePredicate {
                    formula: format!("(>= {} 0)", var.name),
                });

                // Non-positivity
                self.predicates.push(CandidatePredicate {
                    formula: format!("(<= {} 0)", var.name),
                });

                // Bounds with common constants
                for bound in [1, 10, 100, 1000] {
                    self.predicates.push(CandidatePredicate {
                        formula: format!("(<= {} {})", var.name, bound),
                    });
                    self.predicates.push(CandidatePredicate {
                        formula: format!("(>= {} (- {}))", var.name, bound),
                    });
                }
            }
        }

        // Generate ordering predicates for pairs of integer variables
        let int_vars: Vec<&StateVariable> = system
            .variables
            .iter()
            .filter(|v| v.smt_type == SmtType::Int)
            .collect();

        for i in 0..int_vars.len() {
            for j in (i + 1)..int_vars.len() {
                let v1 = &int_vars[i].name;
                let v2 = &int_vars[j].name;

                self.predicates.push(CandidatePredicate {
                    formula: format!("(<= {v1} {v2})"),
                });

                self.predicates.push(CandidatePredicate {
                    formula: format!("(>= {v1} {v2})"),
                });
            }
        }
    }

    /// Extract initial state as example
    fn extract_initial_example(&self, system: &TransitionSystem) -> Option<Example> {
        // Parse init formula to extract variable values
        let init = &system.init.smt_formula;
        let mut values = HashMap::new();

        // Simple pattern matching for (= var value) using pre-compiled regex
        for cap in RE_EQ.captures_iter(init) {
            let name = cap.get(1)?.as_str().to_string();
            let value: i64 = cap.get(2)?.as_str().parse().ok()?;
            values.insert(name, value);
        }

        if values.is_empty() {
            None
        } else {
            Some(Example::positive(values))
        }
    }

    /// Find an invariant consistent with current examples
    fn find_consistent_invariant(
        &self,
        _system: &TransitionSystem,
    ) -> Result<Option<StateFormula>, AiError> {
        // Try each predicate
        for pred in &self.predicates {
            if self.is_consistent_with_examples(&pred.formula) {
                return Ok(Some(StateFormula::with_description(
                    pred.formula.clone(),
                    "ICE candidate".to_string(),
                )));
            }
        }

        // Try conjunctions of predicates
        if self.config.use_decision_tree {
            if let Some(conj) = self.find_conjunctive_invariant()? {
                return Ok(Some(conj));
            }
        }

        Ok(None)
    }

    /// Check if predicate is consistent with examples
    fn is_consistent_with_examples(&self, predicate: &str) -> bool {
        for example in &self.examples {
            let result = self.evaluate_predicate(predicate, &example.values);

            match &example.kind {
                ExampleKind::Positive => {
                    if !result {
                        return false;
                    }
                }
                ExampleKind::Negative => {
                    if result {
                        return false;
                    }
                }
                ExampleKind::Implication { post } => {
                    // If pre satisfies, post must satisfy
                    if result && !self.evaluate_predicate(predicate, post) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Evaluate predicate with concrete values (simple cases).
    ///
    /// Uses pre-compiled regexes from lazy_static for better performance.
    /// Returns true for unknown predicates (conservative assumption).
    fn evaluate_predicate(&self, predicate: &str, values: &HashMap<String, i64>) -> bool {
        // Handle simple comparison predicates using cached regexes
        // Pattern: (>= var N)
        if let Some(cap) = RE_GE.captures(predicate) {
            if let (Some(var_match), Some(bound_match)) = (cap.get(1), cap.get(2)) {
                let var = var_match.as_str();
                let bound: i64 = bound_match.as_str().parse().unwrap_or(0);
                if let Some(&val) = values.get(var) {
                    return val >= bound;
                }
            }
        }

        // Pattern: (<= var N)
        if let Some(cap) = RE_LE.captures(predicate) {
            if let (Some(var_match), Some(bound_match)) = (cap.get(1), cap.get(2)) {
                let var = var_match.as_str();
                let bound: i64 = bound_match.as_str().parse().unwrap_or(0);
                if let Some(&val) = values.get(var) {
                    return val <= bound;
                }
            }
        }

        // Pattern: (>= var (- N)) for negative bounds
        if let Some(cap) = RE_GE_NEG.captures(predicate) {
            if let (Some(var_match), Some(bound_match)) = (cap.get(1), cap.get(2)) {
                let var = var_match.as_str();
                let bound: i64 = bound_match.as_str().parse().unwrap_or(0);
                if let Some(&val) = values.get(var) {
                    return val >= -bound;
                }
            }
        }

        // Pattern: (<= var1 var2) two-variable comparison
        if let Some(cap) = RE_LE_VAR.captures(predicate) {
            if let (Some(var1_match), Some(var2_match)) = (cap.get(1), cap.get(2)) {
                let var1 = var1_match.as_str();
                let var2 = var2_match.as_str();
                if let (Some(&val1), Some(&val2)) = (values.get(var1), values.get(var2)) {
                    return val1 <= val2;
                }
            }
        }

        // Pattern: (>= var1 var2) two-variable comparison
        if let Some(cap) = RE_GE_VAR.captures(predicate) {
            if let (Some(var1_match), Some(var2_match)) = (cap.get(1), cap.get(2)) {
                let var1 = var1_match.as_str();
                let var2 = var2_match.as_str();
                if let (Some(&val1), Some(&val2)) = (values.get(var1), values.get(var2)) {
                    return val1 >= val2;
                }
            }
        }

        // Default: assume true for unknown predicates (conservative)
        true
    }

    /// Find conjunctive invariant
    fn find_conjunctive_invariant(&self) -> Result<Option<StateFormula>, AiError> {
        // Find predicates that are individually consistent with positive examples
        let mut consistent_preds: Vec<&CandidatePredicate> = Vec::new();

        for pred in &self.predicates {
            let mut ok = true;
            for example in &self.examples {
                if example.is_positive() && !self.evaluate_predicate(&pred.formula, &example.values)
                {
                    ok = false;
                    break;
                }
            }
            if ok {
                consistent_preds.push(pred);
            }
        }

        // Try to find a conjunction that excludes all negative examples
        for size in 1..=self.config.max_clause_size.min(consistent_preds.len()) {
            for combo in combinations(&consistent_preds, size) {
                let conj = format!(
                    "(and {})",
                    combo
                        .iter()
                        .map(|p| p.formula.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                );

                if self.is_consistent_with_examples(&conj) {
                    return Ok(Some(StateFormula::with_description(
                        conj,
                        "ICE conjunctive".to_string(),
                    )));
                }
            }
        }

        Ok(None)
    }

    /// Check if invariant is inductive
    fn check_inductiveness(
        &self,
        invariant: &StateFormula,
        system: &TransitionSystem,
        property: &Property,
    ) -> Result<InductivenessResult, AiError> {
        // Check 1: Init => Inv
        if let Some(cex) = self.check_init_implies_inv(invariant, system)? {
            return Ok(InductivenessResult::InitFails(cex));
        }

        // Check 2: Inv & Trans => Inv'
        if let Some((pre, post)) = self.check_induction_step(invariant, system)? {
            return Ok(InductivenessResult::InductionFails { pre, post });
        }

        // Check 3: Inv => Property
        if let Some(cex) = self.check_inv_implies_property(invariant, property, system)? {
            return Ok(InductivenessResult::PropertyFails(cex));
        }

        Ok(InductivenessResult::Inductive)
    }

    /// Check Init => Inv
    fn check_init_implies_inv(
        &self,
        invariant: &StateFormula,
        system: &TransitionSystem,
    ) -> Result<Option<HashMap<String, i64>>, AiError> {
        let smt = self.generate_init_check_smt(invariant, system);
        self.solve_and_extract_model(&smt, system)
    }

    /// Check Inv & Trans => Inv'
    fn check_induction_step(
        &self,
        invariant: &StateFormula,
        system: &TransitionSystem,
    ) -> Result<Option<InductionCounterexample>, AiError> {
        let smt = self.generate_induction_check_smt(invariant, system);

        // Try to solve
        let result = self.run_z3(&smt)?;

        if result.contains("sat") && !result.contains("unsat") {
            // Extract pre and post state
            let pre = self.extract_model_values(&result, &system.variables, false);
            let post = self.extract_model_values(&result, &system.variables, true);

            if !pre.is_empty() && !post.is_empty() {
                return Ok(Some((pre, post)));
            }
        }

        Ok(None)
    }

    /// Check Inv => Property
    fn check_inv_implies_property(
        &self,
        invariant: &StateFormula,
        property: &Property,
        system: &TransitionSystem,
    ) -> Result<Option<HashMap<String, i64>>, AiError> {
        let smt = self.generate_property_check_smt(invariant, property, system);
        self.solve_and_extract_model(&smt, system)
    }

    /// Generate SMT for init check
    fn generate_init_check_smt(
        &self,
        invariant: &StateFormula,
        system: &TransitionSystem,
    ) -> String {
        let mut smt = String::new();
        smt.push_str("(set-logic ALL)\n");

        // Declare variables
        for var in &system.variables {
            let _ = writeln!(
                smt,
                "(declare-const {} {})",
                var.name,
                var.smt_type.to_smt_string()
            );
        }

        // Assert init AND NOT invariant
        let _ = writeln!(smt, "(assert {})", system.init.smt_formula);
        let _ = writeln!(smt, "(assert (not {}))", invariant.smt_formula);

        smt.push_str("(check-sat)\n");
        smt.push_str("(get-model)\n");

        smt
    }

    /// Generate SMT for induction step check
    fn generate_induction_check_smt(
        &self,
        invariant: &StateFormula,
        system: &TransitionSystem,
    ) -> String {
        let mut smt = String::new();
        smt.push_str("(set-logic ALL)\n");

        // Declare current state variables
        for var in &system.variables {
            let _ = writeln!(
                smt,
                "(declare-const {} {})",
                var.name,
                var.smt_type.to_smt_string()
            );
        }

        // Declare next state variables
        for var in &system.variables {
            let _ = writeln!(
                smt,
                "(declare-const {}_next {})",
                var.name,
                var.smt_type.to_smt_string()
            );
        }

        // Assert Inv(current)
        let _ = writeln!(smt, "(assert {})", invariant.smt_formula);

        // Assert transition
        let _ = writeln!(smt, "(assert {})", system.transition.smt_formula);

        // Assert NOT Inv(next)
        let inv_next = self.substitute_next(&invariant.smt_formula, system);
        let _ = writeln!(smt, "(assert (not {inv_next}))");

        smt.push_str("(check-sat)\n");
        smt.push_str("(get-model)\n");

        smt
    }

    /// Generate SMT for property check
    fn generate_property_check_smt(
        &self,
        invariant: &StateFormula,
        property: &Property,
        system: &TransitionSystem,
    ) -> String {
        let mut smt = String::new();
        smt.push_str("(set-logic ALL)\n");

        // Declare variables
        for var in &system.variables {
            let _ = writeln!(
                smt,
                "(declare-const {} {})",
                var.name,
                var.smt_type.to_smt_string()
            );
        }

        // Assert invariant AND NOT property
        let _ = writeln!(smt, "(assert {})", invariant.smt_formula);
        let _ = writeln!(smt, "(assert (not {}))", property.formula.smt_formula);

        smt.push_str("(check-sat)\n");
        smt.push_str("(get-model)\n");

        smt
    }

    /// Substitute variables with next state versions
    fn substitute_next(&self, formula: &str, system: &TransitionSystem) -> String {
        let mut result = formula.to_string();
        for var in &system.variables {
            // Replace var with var_next (careful with word boundaries)
            let pattern = format!(r"\b{}\b", regex::escape(&var.name));
            if let Ok(re) = regex::Regex::new(&pattern) {
                result = re
                    .replace_all(&result, format!("{}_next", var.name))
                    .to_string();
            }
        }
        result
    }

    /// Run Z3 and get output
    fn run_z3(&self, smt: &str) -> Result<String, AiError> {
        let mut child = Command::new("z3")
            .args(["-in", "-t:5000"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| AiError::IceLearning(format!("Failed to spawn Z3: {e}")))?;

        use std::io::Write;
        if let Some(ref mut stdin) = child.stdin {
            stdin
                .write_all(smt.as_bytes())
                .map_err(|e| AiError::IceLearning(format!("Failed to write to Z3: {e}")))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| AiError::IceLearning(format!("Z3 error: {e}")))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Solve and extract model
    fn solve_and_extract_model(
        &self,
        smt: &str,
        system: &TransitionSystem,
    ) -> Result<Option<HashMap<String, i64>>, AiError> {
        let result = self.run_z3(smt)?;

        if result.contains("sat") && !result.contains("unsat") {
            let values = self.extract_model_values(&result, &system.variables, false);
            if !values.is_empty() {
                return Ok(Some(values));
            }
        }

        Ok(None)
    }

    /// Extract variable values from model
    fn extract_model_values(
        &self,
        model: &str,
        variables: &[StateVariable],
        next: bool,
    ) -> HashMap<String, i64> {
        let mut values = HashMap::new();

        for var in variables {
            let var_name = if next {
                format!("{}_next", var.name)
            } else {
                var.name.clone()
            };

            // Pattern: (define-fun var_name () Int value)
            let pattern = format!(
                r"\(define-fun\s+{}\s*\(\)\s*Int\s+(-?\d+|\(-\s*\d+\))\)",
                regex::escape(&var_name)
            );

            if let Ok(re) = regex::Regex::new(&pattern) {
                if let Some(cap) = re.captures(model) {
                    if let Some(val_match) = cap.get(1) {
                        let val_str = val_match.as_str();
                        // Handle (- N) format
                        let value: i64 = if val_str.starts_with("(-") {
                            let inner = val_str
                                .trim_start_matches("(-")
                                .trim_end_matches(')')
                                .trim();
                            -inner.parse::<i64>().unwrap_or(0)
                        } else {
                            val_str.parse().unwrap_or(0)
                        };

                        let store_name = if next {
                            var.name.clone()
                        } else {
                            var_name.clone()
                        };
                        values.insert(store_name, value);
                    }
                }
            }
        }

        values
    }
}

/// Result of inductiveness check
#[derive(Debug)]
enum InductivenessResult {
    /// Invariant is inductive
    Inductive,
    /// Init does not imply invariant
    InitFails(HashMap<String, i64>),
    /// Induction step fails
    InductionFails {
        pre: HashMap<String, i64>,
        post: HashMap<String, i64>,
    },
    /// Invariant does not imply property
    PropertyFails(HashMap<String, i64>),
}

/// A candidate predicate for learning
#[derive(Debug, Clone)]
struct CandidatePredicate {
    formula: String,
}

/// Generate combinations of items
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();

    // Include first element
    for mut combo in combinations(&items[1..], k - 1) {
        combo.insert(0, items[0].clone());
        result.push(combo);
    }

    // Exclude first element
    result.extend(combinations(&items[1..], k));

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use kani_fast_kinduction::TransitionSystemBuilder;
    #[cfg(unix)]
    use std::env;
    #[cfg(unix)]
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use tempfile::tempdir;

    #[test]
    fn test_ice_config_default() {
        let config = IceConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!(config.use_decision_tree);
    }

    #[test]
    fn test_example_creation() {
        let mut values = HashMap::new();
        values.insert("x".to_string(), 5);

        let pos = Example::positive(values.clone());
        assert!(pos.is_positive());

        let neg = Example::negative(values.clone());
        assert!(neg.is_negative());
    }

    #[test]
    fn test_predicate_evaluation() {
        let engine = IceEngine::new(IceConfig::default());

        let mut values = HashMap::new();
        values.insert("x".to_string(), 5);

        assert!(engine.evaluate_predicate("(>= x 0)", &values));
        assert!(!engine.evaluate_predicate("(>= x 10)", &values));
        assert!(engine.evaluate_predicate("(<= x 10)", &values));
        assert!(!engine.evaluate_predicate("(<= x 2)", &values));
    }

    #[test]
    fn test_combinations() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 2);
        assert_eq!(combos.len(), 3);
        assert!(combos.contains(&vec![1, 2]));
        assert!(combos.contains(&vec![1, 3]));
        assert!(combos.contains(&vec![2, 3]));
    }

    #[test]
    fn test_ice_engine_creation() {
        let engine = IceEngine::new(IceConfig::default());
        assert!(engine.examples.is_empty());
        assert!(engine.predicates.is_empty());
    }

    #[test]
    fn test_consistency_check() {
        let mut engine = IceEngine::new(IceConfig::default());

        let mut pos_values = HashMap::new();
        pos_values.insert("x".to_string(), 5);
        engine.add_example(Example::positive(pos_values));

        let mut neg_values = HashMap::new();
        neg_values.insert("x".to_string(), -1);
        engine.add_example(Example::negative(neg_values));

        // (>= x 0) should be consistent: includes 5, excludes -1
        assert!(engine.is_consistent_with_examples("(>= x 0)"));

        // (>= x 10) should not be consistent: excludes 5
        assert!(!engine.is_consistent_with_examples("(>= x 10)"));

        // (<= x 0) should not be consistent: includes -1
        assert!(!engine.is_consistent_with_examples("(<= x 0)"));
    }

    #[test]
    fn test_ice_config_custom() {
        let config = IceConfig {
            max_iterations: 50,
            smt_timeout_secs: 10,
            use_decision_tree: false,
            max_clause_size: 3,
        };
        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.smt_timeout_secs, 10);
        assert!(!config.use_decision_tree);
        assert_eq!(config.max_clause_size, 3);
    }

    #[test]
    fn test_example_kind_positive() {
        let values = HashMap::from([("x".to_string(), 10)]);
        let example = Example::positive(values.clone());
        assert!(matches!(example.kind, ExampleKind::Positive));
        assert_eq!(example.values.get("x"), Some(&10));
        assert!(example.is_positive());
        assert!(!example.is_negative());
    }

    #[test]
    fn test_example_kind_negative() {
        let values = HashMap::from([("x".to_string(), -5)]);
        let example = Example::negative(values.clone());
        assert!(matches!(example.kind, ExampleKind::Negative));
        assert_eq!(example.values.get("x"), Some(&-5));
        assert!(!example.is_positive());
        assert!(example.is_negative());
    }

    #[test]
    fn test_example_kind_implication() {
        let pre = HashMap::from([("x".to_string(), 5)]);
        let post = HashMap::from([("x".to_string(), 6)]);
        let example = Example::implication(pre.clone(), post.clone());
        assert!(matches!(example.kind, ExampleKind::Implication { .. }));
        assert_eq!(example.values.get("x"), Some(&5));
        if let ExampleKind::Implication { post: p } = &example.kind {
            assert_eq!(p.get("x"), Some(&6));
        }
        assert!(!example.is_positive());
        assert!(!example.is_negative());
    }

    #[test]
    fn test_ice_result_structure() {
        let result = IceResult {
            invariant: Some(StateFormula::new("(>= x 0)")),
            examples: vec![Example::positive(HashMap::from([("x".to_string(), 0)]))],
            iterations: 5,
            converged: true,
        };
        assert!(result.invariant.is_some());
        assert_eq!(result.examples.len(), 1);
        assert_eq!(result.iterations, 5);
        assert!(result.converged);
    }

    #[test]
    fn test_ice_result_no_invariant() {
        let result = IceResult {
            invariant: None,
            examples: vec![],
            iterations: 100,
            converged: false,
        };
        assert!(result.invariant.is_none());
        assert!(result.examples.is_empty());
        assert_eq!(result.iterations, 100);
        assert!(!result.converged);
    }

    #[test]
    fn test_add_example() {
        let mut engine = IceEngine::new(IceConfig::default());
        assert!(engine.examples.is_empty());

        let ex1 = Example::positive(HashMap::from([("x".to_string(), 1)]));
        engine.add_example(ex1);
        assert_eq!(engine.examples.len(), 1);

        let ex2 = Example::negative(HashMap::from([("x".to_string(), -1)]));
        engine.add_example(ex2);
        assert_eq!(engine.examples.len(), 2);
    }

    #[test]
    fn test_predicate_evaluation_negative_bounds() {
        let engine = IceEngine::new(IceConfig::default());
        let mut values = HashMap::new();
        values.insert("x".to_string(), -5);

        // Test (>= x (- N)) pattern
        assert!(engine.evaluate_predicate("(>= x (- 10))", &values));
        assert!(!engine.evaluate_predicate("(>= x (- 3))", &values));
    }

    #[test]
    fn test_predicate_evaluation_two_variables() {
        let engine = IceEngine::new(IceConfig::default());
        let mut values = HashMap::new();
        values.insert("x".to_string(), 5);
        values.insert("y".to_string(), 10);

        assert!(engine.evaluate_predicate("(<= x y)", &values));
        assert!(!engine.evaluate_predicate("(>= x y)", &values));
    }

    #[test]
    fn test_predicate_evaluation_equal_values() {
        let engine = IceEngine::new(IceConfig::default());
        let mut values = HashMap::new();
        values.insert("x".to_string(), 7);
        values.insert("y".to_string(), 7);

        assert!(engine.evaluate_predicate("(<= x y)", &values));
        assert!(engine.evaluate_predicate("(>= x y)", &values));
    }

    #[test]
    fn test_predicate_evaluation_unknown_var() {
        let engine = IceEngine::new(IceConfig::default());
        let values = HashMap::from([("x".to_string(), 5)]);
        // Unknown variable should default to true
        assert!(engine.evaluate_predicate("(>= z 0)", &values));
    }

    #[test]
    fn test_predicate_evaluation_unknown_pattern() {
        let engine = IceEngine::new(IceConfig::default());
        let values = HashMap::from([("x".to_string(), 5)]);
        // Unknown pattern should default to true
        assert!(engine.evaluate_predicate("(some_unknown_pattern)", &values));
    }

    #[test]
    fn test_combinations_empty() {
        let items: Vec<i32> = vec![];
        let combos = combinations(&items, 2);
        assert!(combos.is_empty());
    }

    #[test]
    fn test_combinations_zero_k() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 0);
        assert_eq!(combos.len(), 1);
        assert!(combos[0].is_empty());
    }

    #[test]
    fn test_combinations_k_equals_len() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 3);
        assert_eq!(combos.len(), 1);
        assert_eq!(combos[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_combinations_k_greater_than_len() {
        let items = vec![1, 2];
        let combos = combinations(&items, 3);
        assert!(combos.is_empty());
    }

    #[test]
    fn test_combinations_single_item() {
        let items = vec![42];
        let combos = combinations(&items, 1);
        assert_eq!(combos.len(), 1);
        assert_eq!(combos[0], vec![42]);
    }

    #[test]
    fn test_implication_consistency() {
        let mut engine = IceEngine::new(IceConfig::default());

        let pre = HashMap::from([("x".to_string(), 5)]);
        let post = HashMap::from([("x".to_string(), 6)]);
        engine.add_example(Example::implication(pre, post));

        // (>= x 0) should be consistent: if pre (5) satisfies, post (6) must satisfy
        assert!(engine.is_consistent_with_examples("(>= x 0)"));

        // (<= x 5) would fail: pre satisfies but post (6) doesn't
        assert!(!engine.is_consistent_with_examples("(<= x 5)"));
    }

    #[test]
    fn test_implication_pre_does_not_satisfy() {
        let mut engine = IceEngine::new(IceConfig::default());

        let pre = HashMap::from([("x".to_string(), -1)]);
        let post = HashMap::from([("x".to_string(), 0)]);
        engine.add_example(Example::implication(pre, post));

        // (>= x 0): pre (-1) doesn't satisfy, so implication is vacuously true
        assert!(engine.is_consistent_with_examples("(>= x 0)"));
    }

    #[test]
    fn test_generate_candidate_predicates() {
        let mut engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        engine.generate_candidate_predicates(&system);

        // Should have predicates for both variables
        assert!(!engine.predicates.is_empty());

        // Check some expected predicates exist
        let formulas: Vec<&str> = engine
            .predicates
            .iter()
            .map(|p| p.formula.as_str())
            .collect();
        assert!(formulas.contains(&"(>= x 0)"));
        assert!(formulas.contains(&"(>= y 0)"));
        assert!(formulas.contains(&"(<= x y)"));
        assert!(formulas.contains(&"(>= x y)"));
    }

    #[test]
    fn test_generate_candidate_predicates_single_var() {
        let mut engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("n", SmtType::Int)
            .init("(= n 0)")
            .transition("(= n' (+ n 1))")
            .build();

        engine.generate_candidate_predicates(&system);

        let formulas: Vec<&str> = engine
            .predicates
            .iter()
            .map(|p| p.formula.as_str())
            .collect();
        assert!(formulas.contains(&"(>= n 0)"));
        assert!(formulas.contains(&"(<= n 0)"));
        assert!(formulas.contains(&"(<= n 1)"));
        assert!(formulas.contains(&"(<= n 10)"));
        assert!(formulas.contains(&"(<= n 100)"));
    }

    #[test]
    fn test_generate_candidate_predicates_non_int() {
        let mut engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("b", SmtType::Bool)
            .init("(= b true)")
            .transition("(= b' (not b))")
            .build();

        engine.generate_candidate_predicates(&system);

        // Bool variables shouldn't generate predicates with our current implementation
        assert!(engine.predicates.is_empty());
    }

    #[test]
    fn test_extract_initial_example_simple() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let example = engine.extract_initial_example(&system);
        assert!(example.is_some());
        let ex = example.unwrap();
        assert!(ex.is_positive());
        assert_eq!(ex.values.get("x"), Some(&0));
    }

    #[test]
    fn test_extract_initial_example_multiple_vars() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .init("(and (= x 5) (= y 10))")
            .transition("true")
            .build();

        let example = engine.extract_initial_example(&system);
        assert!(example.is_some());
        let ex = example.unwrap();
        assert_eq!(ex.values.get("x"), Some(&5));
        assert_eq!(ex.values.get("y"), Some(&10));
    }

    #[test]
    fn test_extract_initial_example_negative_value() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x -42)")
            .transition("true")
            .build();

        let example = engine.extract_initial_example(&system);
        assert!(example.is_some());
        let ex = example.unwrap();
        assert_eq!(ex.values.get("x"), Some(&-42));
    }

    #[test]
    fn test_extract_initial_example_no_values() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(>= x 0)") // Not an equality, so no concrete value
            .transition("true")
            .build();

        let example = engine.extract_initial_example(&system);
        // Pattern doesn't match (= var value), should return None
        assert!(example.is_none());
    }

    #[test]
    fn test_substitute_next_single_var() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        let formula = "(>= x 0)";
        let result = engine.substitute_next(formula, &system);
        assert_eq!(result, "(>= x_next 0)");
    }

    #[test]
    fn test_substitute_next_multiple_vars() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .init("true")
            .transition("true")
            .build();

        let formula = "(and (>= x 0) (<= x y))";
        let result = engine.substitute_next(formula, &system);
        assert_eq!(result, "(and (>= x_next 0) (<= x_next y_next))");
    }

    #[test]
    fn test_generate_init_check_smt() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        let invariant = StateFormula::new("(>= x 0)");
        let smt = engine.generate_init_check_smt(&invariant, &system);

        assert!(smt.contains("(set-logic ALL)"));
        assert!(smt.contains("(declare-const x Int)"));
        assert!(smt.contains("(assert (= x 0))"));
        assert!(smt.contains("(assert (not (>= x 0)))"));
        assert!(smt.contains("(check-sat)"));
    }

    #[test]
    fn test_generate_induction_check_smt() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x_next (+ x 1))")
            .build();

        let invariant = StateFormula::new("(>= x 0)");
        let smt = engine.generate_induction_check_smt(&invariant, &system);

        assert!(smt.contains("(declare-const x Int)"));
        assert!(smt.contains("(declare-const x_next Int)"));
        assert!(smt.contains("(assert (>= x 0))"));
        assert!(smt.contains("(assert (= x_next (+ x 1)))"));
        assert!(smt.contains("(assert (not (>= x_next 0)))"));
    }

    #[test]
    fn test_generate_property_check_smt() {
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        let invariant = StateFormula::new("(>= x 0)");
        let property = Property::safety("p1", "positive", StateFormula::new("(> x -1)"));
        let smt = engine.generate_property_check_smt(&invariant, &property, &system);

        assert!(smt.contains("(declare-const x Int)"));
        assert!(smt.contains("(assert (>= x 0))"));
        assert!(smt.contains("(assert (not (> x -1)))"));
    }

    #[test]
    fn test_find_consistent_invariant_single_predicate() {
        let mut engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x_next (+ x 1))")
            .build();

        engine.generate_candidate_predicates(&system);

        // Add examples that should lead to (>= x 0)
        engine.add_example(Example::positive(HashMap::from([("x".to_string(), 0)])));
        engine.add_example(Example::positive(HashMap::from([("x".to_string(), 5)])));
        engine.add_example(Example::negative(HashMap::from([("x".to_string(), -1)])));

        let result = engine.find_consistent_invariant(&system).unwrap();
        assert!(result.is_some());
        let inv = result.unwrap();
        // Should find (>= x 0) or similar
        assert!(inv.smt_formula.contains(">=") || inv.smt_formula.contains("x"));
    }

    #[test]
    fn test_example_serialization() {
        let values = HashMap::from([("x".to_string(), 42)]);
        let example = Example::positive(values);

        let json = serde_json::to_string(&example).unwrap();
        assert!(json.contains("42"));
        assert!(json.contains("Positive"));

        let parsed: Example = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_positive());
        assert_eq!(parsed.values.get("x"), Some(&42));
    }

    #[test]
    fn test_example_kind_serialization() {
        let kind = ExampleKind::Positive;
        let json = serde_json::to_string(&kind).unwrap();
        let parsed: ExampleKind = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ExampleKind::Positive));

        let kind = ExampleKind::Negative;
        let json = serde_json::to_string(&kind).unwrap();
        let parsed: ExampleKind = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ExampleKind::Negative));

        let kind = ExampleKind::Implication {
            post: HashMap::from([("x".to_string(), 10)]),
        };
        let json = serde_json::to_string(&kind).unwrap();
        let parsed: ExampleKind = serde_json::from_str(&json).unwrap();
        if let ExampleKind::Implication { post } = parsed {
            assert_eq!(post.get("x"), Some(&10));
        } else {
            panic!("Expected Implication");
        }
    }

    #[test]
    fn test_ice_config_clone() {
        let config = IceConfig {
            max_iterations: 200,
            smt_timeout_secs: 15,
            use_decision_tree: true,
            max_clause_size: 7,
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_iterations, 200);
        assert_eq!(cloned.smt_timeout_secs, 15);
        assert!(cloned.use_decision_tree);
        assert_eq!(cloned.max_clause_size, 7);
    }

    #[test]
    fn test_ice_result_clone() {
        let result = IceResult {
            invariant: Some(StateFormula::new("(>= x 0)")),
            examples: vec![Example::positive(HashMap::from([("x".to_string(), 1)]))],
            iterations: 3,
            converged: true,
        };
        let cloned = result.clone();
        assert_eq!(cloned.iterations, 3);
        assert!(cloned.converged);
        assert!(cloned.invariant.is_some());
    }

    // --- Mutation coverage tests ---

    #[test]
    fn test_predicate_ge_boundary_exact() {
        // Tests line 378: replace >= with < - must catch this by testing exact boundary
        let engine = IceEngine::new(IceConfig::default());
        let values = HashMap::from([("x".to_string(), 5)]);

        // x=5, bound=5: >= should be true, < should be false
        assert!(engine.evaluate_predicate("(>= x 5)", &values));
        // x=5, bound=6: >= should be false
        assert!(!engine.evaluate_predicate("(>= x 6)", &values));
    }

    #[test]
    fn test_predicate_le_boundary_exact() {
        // Tests line 389: replace <= with > - must catch this by testing exact boundary
        let engine = IceEngine::new(IceConfig::default());
        let values = HashMap::from([("x".to_string(), 5)]);

        // x=5, bound=5: <= should be true, > should be false
        assert!(engine.evaluate_predicate("(<= x 5)", &values));
        // x=5, bound=4: <= should be false
        assert!(!engine.evaluate_predicate("(<= x 4)", &values));
    }

    #[test]
    fn test_predicate_ge_negative_boundary_exact() {
        // Tests line 400: replace >= with < in negative bounds
        let engine = IceEngine::new(IceConfig::default());
        let values = HashMap::from([("x".to_string(), -5)]);

        // x=-5, (>= x (- 5)) means x >= -5, should be true
        assert!(engine.evaluate_predicate("(>= x (- 5))", &values));
        // x=-5, (>= x (- 4)) means x >= -4, should be false (-5 < -4)
        assert!(!engine.evaluate_predicate("(>= x (- 4))", &values));
    }

    #[test]
    fn test_predicate_ge_negative_sign_handling() {
        // Tests line 400: delete - in evaluate_predicate (negation of bound)
        let engine = IceEngine::new(IceConfig::default());
        let values = HashMap::from([("x".to_string(), 5)]);

        // x=5, (>= x (- 10)) means x >= -10, should be true
        assert!(engine.evaluate_predicate("(>= x (- 10))", &values));
        // x=5, (>= x (- 3)) means x >= -3, should be true
        assert!(engine.evaluate_predicate("(>= x (- 3))", &values));

        // Now test with negative x to distinguish >= vs < and - vs +
        let neg_values = HashMap::from([("x".to_string(), -7)]);
        // x=-7, (>= x (- 10)) means x >= -10, -7 >= -10 is true
        assert!(engine.evaluate_predicate("(>= x (- 10))", &neg_values));
        // x=-7, (>= x (- 5)) means x >= -5, -7 >= -5 is false
        assert!(!engine.evaluate_predicate("(>= x (- 5))", &neg_values));
    }

    #[test]
    fn test_predicate_le_var_boundary_exact() {
        // Tests line 411: replace <= with > in two-variable comparison
        let engine = IceEngine::new(IceConfig::default());

        // x=5, y=5: x <= y should be true
        let eq_values = HashMap::from([("x".to_string(), 5), ("y".to_string(), 5)]);
        assert!(engine.evaluate_predicate("(<= x y)", &eq_values));

        // x=6, y=5: x <= y should be false
        let gt_values = HashMap::from([("x".to_string(), 6), ("y".to_string(), 5)]);
        assert!(!engine.evaluate_predicate("(<= x y)", &gt_values));
    }

    #[test]
    fn test_predicate_ge_var_boundary_exact() {
        // Tests line 422: replace >= with < in two-variable comparison
        let engine = IceEngine::new(IceConfig::default());

        // x=5, y=5: x >= y should be true
        let eq_values = HashMap::from([("x".to_string(), 5), ("y".to_string(), 5)]);
        assert!(engine.evaluate_predicate("(>= x y)", &eq_values));

        // x=4, y=5: x >= y should be false
        let lt_values = HashMap::from([("x".to_string(), 4), ("y".to_string(), 5)]);
        assert!(!engine.evaluate_predicate("(>= x y)", &lt_values));
    }

    #[test]
    fn test_is_consistent_positive_false_case() {
        // Tests line 346: delete ! in is_consistent_with_examples
        let mut engine = IceEngine::new(IceConfig::default());

        // Add positive example that does NOT satisfy the predicate
        engine.add_example(Example::positive(HashMap::from([("x".to_string(), -5)])));

        // (>= x 0) is not consistent because positive example has x=-5
        // The ! on line 346 is needed to return false when result is false
        assert!(!engine.is_consistent_with_examples("(>= x 0)"));
    }

    #[test]
    fn test_is_consistent_negative_true_case() {
        // Tests line 346: the negative example path
        let mut engine = IceEngine::new(IceConfig::default());

        // Add negative example that DOES satisfy the predicate (should fail consistency)
        engine.add_example(Example::negative(HashMap::from([("x".to_string(), 5)])));

        // (>= x 0) is not consistent because negative example satisfies it
        // A negative example must NOT satisfy the predicate
        assert!(!engine.is_consistent_with_examples("(>= x 0)"));
    }

    #[test]
    fn test_is_consistent_implication_both_conditions() {
        // Tests line 357: replace && with || and delete !
        // The condition is: result && !self.evaluate_predicate(predicate, post)
        let mut engine = IceEngine::new(IceConfig::default());

        // Implication: pre satisfies, post does NOT satisfy -> should fail
        let pre = HashMap::from([("x".to_string(), 10)]);
        let post = HashMap::from([("x".to_string(), -5)]);
        engine.add_example(Example::implication(pre, post));

        // (>= x 0): pre=10 satisfies, post=-5 does not -> INCONSISTENT
        // If we replace && with ||: result(true) || !eval(true) = true || false = true -> wrong
        // If we delete !: result(true) && eval(true) = true && true = true -> wrong
        assert!(!engine.is_consistent_with_examples("(>= x 0)"));
    }

    #[test]
    fn test_is_consistent_implication_pre_not_satisfying() {
        // When pre doesn't satisfy, the implication is vacuously true
        let mut engine = IceEngine::new(IceConfig::default());

        let pre = HashMap::from([("x".to_string(), -10)]);
        let post = HashMap::from([("x".to_string(), -5)]);
        engine.add_example(Example::implication(pre, post));

        // (>= x 0): pre=-10 doesn't satisfy (result=false)
        // Implication: false => anything is true (vacuous truth)
        // So this should be consistent
        assert!(engine.is_consistent_with_examples("(>= x 0)"));
    }

    #[test]
    fn test_is_consistent_implication_both_satisfy() {
        // When both pre and post satisfy, should be consistent
        let mut engine = IceEngine::new(IceConfig::default());

        let pre = HashMap::from([("x".to_string(), 5)]);
        let post = HashMap::from([("x".to_string(), 10)]);
        engine.add_example(Example::implication(pre, post));

        // (>= x 0): pre=5 satisfies, post=10 satisfies -> consistent
        assert!(engine.is_consistent_with_examples("(>= x 0)"));
    }

    #[test]
    fn test_example_positive_returns_correct_values() {
        // Tests Example::positive constructor
        let values = HashMap::from([("a".to_string(), 100), ("b".to_string(), -50)]);
        let ex = Example::positive(values.clone());

        assert_eq!(ex.values, values);
        assert!(matches!(ex.kind, ExampleKind::Positive));
    }

    #[test]
    fn test_example_negative_returns_correct_values() {
        // Tests Example::negative constructor
        let values = HashMap::from([("a".to_string(), -100), ("b".to_string(), 50)]);
        let ex = Example::negative(values.clone());

        assert_eq!(ex.values, values);
        assert!(matches!(ex.kind, ExampleKind::Negative));
    }

    #[test]
    fn test_example_implication_returns_correct_values() {
        // Tests Example::implication constructor
        let pre = HashMap::from([("x".to_string(), 1)]);
        let post = HashMap::from([("x".to_string(), 2)]);
        let ex = Example::implication(pre.clone(), post.clone());

        assert_eq!(ex.values, pre);
        if let ExampleKind::Implication { post: p } = &ex.kind {
            assert_eq!(*p, post);
        } else {
            panic!("Expected Implication kind");
        }
    }

    #[test]
    fn test_is_positive_distinguishes_from_negative() {
        // Tests line 100: replace is_positive -> bool with true/false
        let pos = Example::positive(HashMap::new());
        let neg = Example::negative(HashMap::new());
        let imp = Example::implication(HashMap::new(), HashMap::new());

        assert!(pos.is_positive());
        assert!(!neg.is_positive());
        assert!(!imp.is_positive());
    }

    #[test]
    fn test_is_negative_distinguishes_from_positive() {
        // Tests line 105: replace is_negative -> bool with true/false
        let pos = Example::positive(HashMap::new());
        let neg = Example::negative(HashMap::new());
        let imp = Example::implication(HashMap::new(), HashMap::new());

        assert!(!pos.is_negative());
        assert!(neg.is_negative());
        assert!(!imp.is_negative());
    }

    #[test]
    fn test_add_example_increments_count() {
        // Tests line 154: replace add_example with ()
        let mut engine = IceEngine::new(IceConfig::default());

        assert!(engine.examples.is_empty());
        engine.add_example(Example::positive(HashMap::new()));
        assert_eq!(engine.examples.len(), 1);
        engine.add_example(Example::negative(HashMap::new()));
        assert_eq!(engine.examples.len(), 2);
        engine.add_example(Example::implication(HashMap::new(), HashMap::new()));
        assert_eq!(engine.examples.len(), 3);
    }

    #[test]
    fn test_generate_candidate_predicates_includes_bounds() {
        // Tests line 248: replace == with != (Int type check)
        let mut engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        engine.generate_candidate_predicates(&system);

        // Should have predicates for Int variable
        assert!(!engine.predicates.is_empty());

        // Check bounds predicates exist
        let formulas: Vec<&str> = engine
            .predicates
            .iter()
            .map(|p| p.formula.as_str())
            .collect();
        assert!(formulas.contains(&"(<= x 1)"));
        assert!(formulas.contains(&"(<= x 10)"));
        assert!(formulas.contains(&"(<= x 100)"));
        assert!(formulas.contains(&"(<= x 1000)"));
    }

    #[test]
    fn test_generate_candidate_predicates_var_ordering() {
        // Tests line 279: replace + with - or * in loop index calculation
        let mut engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("a", SmtType::Int)
            .variable("b", SmtType::Int)
            .variable("c", SmtType::Int)
            .init("true")
            .transition("true")
            .build();

        engine.generate_candidate_predicates(&system);

        let formulas: Vec<&str> = engine
            .predicates
            .iter()
            .map(|p| p.formula.as_str())
            .collect();

        // Check ordering predicates exist between different pairs
        assert!(formulas.contains(&"(<= a b)"));
        assert!(formulas.contains(&"(>= a b)"));
        assert!(formulas.contains(&"(<= a c)"));
        assert!(formulas.contains(&"(>= a c)"));
        assert!(formulas.contains(&"(<= b c)"));
        assert!(formulas.contains(&"(>= b c)"));
    }

    #[test]
    fn test_find_consistent_invariant_no_predicates() {
        // Tests line 320: find_consistent_invariant returns None when no predicates
        let mut engine = IceEngine::new(IceConfig {
            use_decision_tree: false, // Disable conjunctive search
            ..Default::default()
        });

        // Don't generate any predicates
        engine.predicates.clear();

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        let result = engine.find_consistent_invariant(&system).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_find_consistent_invariant_returns_some() {
        // Tests line 320: find_consistent_invariant returns Some when predicate found
        let mut engine = IceEngine::new(IceConfig {
            use_decision_tree: false,
            ..Default::default()
        });

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        engine.generate_candidate_predicates(&system);
        // No examples = all predicates consistent
        let result = engine.find_consistent_invariant(&system).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_find_conjunctive_invariant_single_predicate() {
        // Tests line 434: find_conjunctive_invariant returns result
        let mut engine = IceEngine::new(IceConfig::default());

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        engine.generate_candidate_predicates(&system);

        // Add conflicting examples that require conjunction
        engine.add_example(Example::positive(HashMap::from([("x".to_string(), 5)])));
        engine.add_example(Example::negative(HashMap::from([("x".to_string(), -1)])));

        let result = engine.find_conjunctive_invariant().unwrap();
        // Should find something or None, but not crash
        // The exact result depends on predicate ordering
        let _ = result;
    }

    #[test]
    fn test_is_consistent_returns_true_with_no_examples() {
        // Tests line 341: is_consistent_with_examples returns true/false
        let engine = IceEngine::new(IceConfig::default());

        // No examples = always consistent
        assert!(engine.is_consistent_with_examples("(>= x 0)"));
        assert!(engine.is_consistent_with_examples("(<= x 0)"));
        assert!(engine.is_consistent_with_examples("anything"));
    }

    #[test]
    fn test_is_consistent_returns_false_with_conflict() {
        // Tests line 341: is_consistent_with_examples must return false on conflict
        let mut engine = IceEngine::new(IceConfig::default());

        // Positive example that doesn't satisfy the predicate
        engine.add_example(Example::positive(HashMap::from([("x".to_string(), -1)])));

        // (>= x 0) is NOT consistent because -1 < 0
        assert!(!engine.is_consistent_with_examples("(>= x 0)"));
    }

    #[test]
    fn test_extract_initial_example_returns_none_for_empty() {
        // Tests line 297: extract_initial_example returns None
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("true") // No equality constraints
            .transition("true")
            .build();

        let result = engine.extract_initial_example(&system);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_initial_example_returns_some() {
        // Tests line 297: extract_initial_example returns Some
        let engine = IceEngine::new(IceConfig::default());
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 42)")
            .transition("true")
            .build();

        let result = engine.extract_initial_example(&system);
        assert!(result.is_some());
        let ex = result.unwrap();
        assert_eq!(ex.values.get("x"), Some(&42));
    }

    #[test]
    fn test_candidate_predicate_debug() {
        // Test CandidatePredicate Debug derive
        let pred = CandidatePredicate {
            formula: "(>= x 0)".to_string(),
        };
        let debug = format!("{:?}", pred);
        assert!(debug.contains("CandidatePredicate"));
        assert!(debug.contains("(>= x 0)"));
    }

    #[test]
    fn test_candidate_predicate_clone() {
        // Test CandidatePredicate Clone derive
        let pred = CandidatePredicate {
            formula: "(<= y 10)".to_string(),
        };
        let cloned = pred.clone();
        assert_eq!(cloned.formula, "(<= y 10)");
    }

    #[test]
    fn test_inductiveness_result_debug() {
        // Test InductivenessResult Debug derive
        let result = InductivenessResult::Inductive;
        let debug = format!("{:?}", result);
        assert!(debug.contains("Inductive"));

        let result = InductivenessResult::InitFails(HashMap::new());
        let debug = format!("{:?}", result);
        assert!(debug.contains("InitFails"));

        let result = InductivenessResult::InductionFails {
            pre: HashMap::new(),
            post: HashMap::new(),
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("InductionFails"));

        let result = InductivenessResult::PropertyFails(HashMap::new());
        let debug = format!("{:?}", result);
        assert!(debug.contains("PropertyFails"));
    }

    // --- Fake Z3 helper for mutation coverage tests that need solver output ---

    // Mutex to serialize tests that modify the PATH environment variable.
    // Without this, parallel tests can interfere with each other's PATH changes.
    #[cfg(unix)]
    static PATH_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(unix)]
    fn with_fake_z3_output(output: &str, test: impl FnOnce()) {
        // Acquire lock to prevent concurrent PATH modifications
        let _lock = PATH_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        let dir = tempdir().expect("tempdir");
        let script = dir.path().join("z3");
        let script_content = format!("#!/bin/sh\ncat <<'EOF'\n{}\nEOF\n", output);
        fs::write(&script, script_content).expect("write fake z3 script");

        // Make executable
        let mut perms = fs::metadata(&script).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script, perms).expect("set permissions");

        // Prepend to PATH and run test
        let original_path = env::var("PATH").unwrap_or_default();
        let new_path = format!("{}:{}", dir.path().display(), original_path);
        env::set_var("PATH", &new_path);

        struct PathGuard {
            original: String,
        }
        impl Drop for PathGuard {
            fn drop(&mut self) {
                env::set_var("PATH", &self.original);
            }
        }

        let _guard = PathGuard {
            original: original_path,
        };

        // Keep tempdir alive for the duration of the test closure
        let _keep_dir = dir;
        test();
    }

    #[cfg(unix)]
    #[test]
    fn test_run_z3_uses_fake_binary_output() {
        let fake_output = "sat\n(model\n  (define-fun x () Int 1)\n)";
        with_fake_z3_output(fake_output, || {
            let engine = IceEngine::new(IceConfig::default());
            let output = engine.run_z3("(check-sat)").expect("run_z3 should succeed");
            assert!(
                output.contains("sat"),
                "Output should include sat from fake solver"
            );
            assert!(
                output.contains("define-fun x"),
                "Should propagate solver stdout"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_solve_and_extract_model_returns_values_on_sat() {
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        let fake_output = "sat\n(model\n  (define-fun x () Int 5)\n)";
        with_fake_z3_output(fake_output, || {
            let engine = IceEngine::new(IceConfig::default());
            let result = engine
                .solve_and_extract_model("(check-sat)", &system)
                .expect("run should succeed");
            let values = result.expect("expected model values");
            assert_eq!(values.get("x"), Some(&5));
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_solve_and_extract_model_returns_none_on_unsat_output() {
        // Output intentionally contains both "unsat" and a model to ensure the unsat guard is used
        let fake_output = "unsat\n(model\n  (define-fun x () Int 7)\n)";
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        with_fake_z3_output(fake_output, || {
            let engine = IceEngine::new(IceConfig::default());
            let result = engine
                .solve_and_extract_model("(check-sat)", &system)
                .expect("run should succeed");
            assert!(
                result.is_none(),
                "Unsat outputs should not produce model values"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_check_induction_step_extracts_pre_and_post_states() {
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x_next (+ x 1))")
            .build();
        let invariant = StateFormula::new("(>= x 0)");

        let fake_output =
            "sat\n(model\n  (define-fun x () Int 2)\n  (define-fun x_next () Int 3)\n)";
        with_fake_z3_output(fake_output, || {
            let engine = IceEngine::new(IceConfig::default());
            let result = engine
                .check_induction_step(&invariant, &system)
                .expect("run should succeed");
            let (pre, post) = result.expect("expected induction counterexample");
            assert_eq!(pre.get("x"), Some(&2));
            assert_eq!(post.get("x"), Some(&3));
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_check_induction_step_ignores_unsat_output() {
        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x_next (+ x 1))")
            .build();
        let invariant = StateFormula::new("(>= x 0)");

        let fake_output =
            "unsat\n(model\n  (define-fun x () Int 4)\n  (define-fun x_next () Int 5)\n)";
        with_fake_z3_output(fake_output, || {
            let engine = IceEngine::new(IceConfig::default());
            let result = engine
                .check_induction_step(&invariant, &system)
                .expect("run should succeed");
            assert!(
                result.is_none(),
                "Unsat outputs should not produce counterexamples"
            );
        });
    }
}
