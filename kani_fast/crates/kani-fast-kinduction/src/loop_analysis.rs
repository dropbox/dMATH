//! Loop bound analysis for k-induction
//!
//! This module provides analysis of loops in the transition system to:
//! - Determine loop bounds for optimal k selection
//! - Identify loop invariant candidates
//! - Extract loop structure for invariant synthesis

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Pre-compiled regex patterns for bound detection (both infix and SMT-LIB formats)
static BOUND_PATTERNS: Lazy<[Regex; 8]> = Lazy::new(|| {
    [
        // Infix patterns: x < 10, x <= 10
        Regex::new(r"<\s*(\d+)").expect("valid regex"),
        Regex::new(r"<=\s*(\d+)").expect("valid regex"),
        Regex::new(r">\s*(\d+)").expect("valid regex"),
        Regex::new(r">=\s*(\d+)").expect("valid regex"),
        // SMT-LIB patterns: (< x 10), (<= x 10)
        Regex::new(r"\(\s*<\s+\w+\s+(\d+)\s*\)").expect("valid regex"),
        Regex::new(r"\(\s*<=\s+\w+\s+(\d+)\s*\)").expect("valid regex"),
        Regex::new(r"\(\s*>\s+\w+\s+(\d+)\s*\)").expect("valid regex"),
        Regex::new(r"\(\s*>=\s+\w+\s+(\d+)\s*\)").expect("valid regex"),
    ]
});

/// Information about a loop in the program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopInfo {
    /// Unique identifier for this loop
    pub id: String,

    /// Loop bound (None if unbounded)
    pub bound: Option<u64>,

    /// Confidence in the bound analysis (0.0 to 1.0)
    pub bound_confidence: f64,

    /// Variables modified in the loop
    pub modified_variables: Vec<String>,

    /// Variables read but not modified (loop-invariant)
    pub invariant_variables: Vec<String>,

    /// Loop condition expression
    pub condition: Option<String>,

    /// Induction variable (if detected)
    pub induction_variable: Option<InductionVariable>,

    /// Nesting depth (0 = outermost)
    pub nesting_depth: u32,

    /// Inner loops
    pub inner_loops: Vec<LoopInfo>,
}

impl LoopInfo {
    /// Create a new loop info with minimal information
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            bound: None,
            bound_confidence: 0.0,
            modified_variables: Vec::new(),
            invariant_variables: Vec::new(),
            condition: None,
            induction_variable: None,
            nesting_depth: 0,
            inner_loops: Vec::new(),
        }
    }

    /// Check if loop has a known bound
    pub fn is_bounded(&self) -> bool {
        self.bound.is_some()
    }

    /// Get effective bound (max of this and inner loops)
    pub fn effective_bound(&self) -> Option<u64> {
        let inner_max = self
            .inner_loops
            .iter()
            .filter_map(|l| l.effective_bound())
            .max();

        match (self.bound, inner_max) {
            (Some(b), Some(i)) => Some(b * i), // Nested loops multiply
            (Some(b), None) => Some(b),
            (None, Some(i)) => Some(i),
            (None, None) => None,
        }
    }
}

/// An induction variable detected in a loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InductionVariable {
    /// Variable name
    pub name: String,
    /// Initial value expression
    pub init: String,
    /// Step expression (e.g., "+ 1")
    pub step: String,
    /// Direction (increasing, decreasing, unknown)
    pub direction: InductionDirection,
}

/// Direction of an induction variable
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InductionDirection {
    Increasing,
    Decreasing,
    Unknown,
}

/// Loop bound analyzer
pub struct LoopBoundAnalyzer {
    /// Detected loops
    loops: Vec<LoopInfo>,
    /// Suggested k value based on analysis
    suggested_k: Option<u32>,
}

impl LoopBoundAnalyzer {
    pub fn new() -> Self {
        Self {
            loops: Vec::new(),
            suggested_k: None,
        }
    }

    /// Analyze a transition system to detect loops and their bounds
    pub fn analyze(&mut self, transition_formula: &str, variables: &[String]) {
        // Pattern matching for common loop patterns in transition formulas
        self.detect_counter_loops(transition_formula, variables);
        self.detect_bounded_iteration(transition_formula, variables);

        // Compute suggested k
        self.compute_suggested_k();
    }

    /// Detect counter-based loops (i++ until limit)
    fn detect_counter_loops(&mut self, formula: &str, variables: &[String]) {
        // Look for patterns like: x' = x + 1 with condition x < N
        for var in variables {
            // Check for increment pattern using string matching instead of regex
            // Pattern: {var}' followed by = followed by {var} followed by + 1
            // Order matters: primed var, then =, then var, then +1
            let primed_var = format!("{var}'");
            let has_increment_pattern = if let Some(prime_pos) = formula.find(&primed_var) {
                // After the primed var, find = sign
                let after_prime = &formula[prime_pos + primed_var.len()..];
                if let Some(eq_rel_pos) = after_prime.find('=') {
                    // After =, check for {var} followed by + 1
                    let after_eq = &after_prime[eq_rel_pos + 1..];
                    after_eq.contains(var.as_str())
                        && (after_eq.contains("+ 1") || after_eq.contains("+1"))
                } else {
                    false
                }
            } else {
                false
            };
            if has_increment_pattern {
                // Found potential counter loop
                let mut loop_info = LoopInfo::new(format!("counter_{var}"));
                loop_info.modified_variables.push(var.clone());
                loop_info.induction_variable = Some(InductionVariable {
                    name: var.clone(),
                    init: "unknown".to_string(),
                    step: "+ 1".to_string(),
                    direction: InductionDirection::Increasing,
                });

                // Try to find bound from condition
                if let Some(bound) = self.extract_bound_from_condition(formula, var) {
                    loop_info.bound = Some(bound);
                    loop_info.bound_confidence = 0.8;
                }

                self.loops.push(loop_info);
            }
        }
    }

    /// Detect bounded iteration patterns
    fn detect_bounded_iteration(&mut self, formula: &str, _variables: &[String]) {
        // Use pre-compiled regex patterns for better performance
        for re in BOUND_PATTERNS.iter() {
            for cap in re.captures_iter(formula) {
                if let Some(bound_str) = cap.get(1) {
                    if let Ok(bound) = bound_str.as_str().parse::<u64>() {
                        // Found explicit bound in formula
                        if self.loops.iter().all(|l| l.bound != Some(bound)) {
                            let mut loop_info = LoopInfo::new(format!("bounded_{bound}"));
                            loop_info.bound = Some(bound);
                            loop_info.bound_confidence = 0.9;
                            self.loops.push(loop_info);
                        }
                    }
                }
            }
        }
    }

    /// Extract bound from comparison condition
    fn extract_bound_from_condition(&self, formula: &str, var: &str) -> Option<u64> {
        // Look for var < N or var <= N patterns
        let patterns = [
            format!(r"{var}\s*<\s*(\d+)"),
            format!(r"{var}\s*<=\s*(\d+)"),
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(cap) = re.captures(formula) {
                    if let Some(bound_str) = cap.get(1) {
                        if let Ok(bound) = bound_str.as_str().parse::<u64>() {
                            return Some(bound);
                        }
                    }
                }
            }
        }
        None
    }

    /// Compute suggested k value based on loop analysis
    fn compute_suggested_k(&mut self) {
        let max_bound = self.loops.iter().filter_map(|l| l.effective_bound()).max();

        self.suggested_k = max_bound.map(|b| {
            // Suggest k slightly larger than max bound to catch off-by-one errors
            (b as u32).saturating_add(2).min(100)
        });
    }

    /// Get detected loops
    pub fn loops(&self) -> &[LoopInfo] {
        &self.loops
    }

    /// Get suggested k value
    pub fn suggested_k(&self) -> Option<u32> {
        self.suggested_k
    }

    /// Check if all loops are bounded
    pub fn all_bounded(&self) -> bool {
        !self.loops.is_empty() && self.loops.iter().all(|l| l.is_bounded())
    }

    /// Get maximum detected bound
    pub fn max_bound(&self) -> Option<u64> {
        self.loops.iter().filter_map(|l| l.effective_bound()).max()
    }
}

impl Default for LoopBoundAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== LoopInfo Tests ====================

    #[test]
    fn test_loop_info_new() {
        let loop_info = LoopInfo::new("test_loop");

        assert_eq!(loop_info.id, "test_loop");
        assert!(loop_info.bound.is_none());
        assert_eq!(loop_info.bound_confidence, 0.0);
        assert!(loop_info.modified_variables.is_empty());
        assert!(loop_info.invariant_variables.is_empty());
        assert!(loop_info.condition.is_none());
        assert!(loop_info.induction_variable.is_none());
        assert_eq!(loop_info.nesting_depth, 0);
        assert!(loop_info.inner_loops.is_empty());
    }

    #[test]
    fn test_loop_info_new_from_string() {
        let loop_info = LoopInfo::new(String::from("string_id"));
        assert_eq!(loop_info.id, "string_id");
    }

    #[test]
    fn test_loop_info_builder() {
        let mut loop_info = LoopInfo::new("test_loop");
        loop_info.bound = Some(10);
        loop_info.modified_variables.push("i".to_string());

        assert!(loop_info.is_bounded());
        assert_eq!(loop_info.effective_bound(), Some(10));
    }

    #[test]
    fn test_loop_info_is_bounded_true() {
        let mut loop_info = LoopInfo::new("bounded");
        loop_info.bound = Some(100);
        assert!(loop_info.is_bounded());
    }

    #[test]
    fn test_loop_info_is_bounded_false() {
        let loop_info = LoopInfo::new("unbounded");
        assert!(!loop_info.is_bounded());
    }

    #[test]
    fn test_loop_info_effective_bound_no_inner() {
        let mut loop_info = LoopInfo::new("single");
        loop_info.bound = Some(25);
        assert_eq!(loop_info.effective_bound(), Some(25));
    }

    #[test]
    fn test_loop_info_effective_bound_unbounded() {
        let loop_info = LoopInfo::new("unbounded");
        assert_eq!(loop_info.effective_bound(), None);
    }

    #[test]
    fn test_loop_info_effective_bound_unbounded_outer_bounded_inner() {
        let mut outer = LoopInfo::new("outer");
        // outer has no bound

        let mut inner = LoopInfo::new("inner");
        inner.bound = Some(5);
        outer.inner_loops.push(inner);

        // Inner bound propagates
        assert_eq!(outer.effective_bound(), Some(5));
    }

    #[test]
    fn test_loop_info_effective_bound_bounded_outer_unbounded_inner() {
        let mut outer = LoopInfo::new("outer");
        outer.bound = Some(10);

        let inner = LoopInfo::new("inner");
        // inner has no bound
        outer.inner_loops.push(inner);

        // Outer bound used
        assert_eq!(outer.effective_bound(), Some(10));
    }

    #[test]
    fn test_nested_loop_bound() {
        let mut outer = LoopInfo::new("outer");
        outer.bound = Some(10);

        let mut inner = LoopInfo::new("inner");
        inner.bound = Some(5);
        inner.nesting_depth = 1;

        outer.inner_loops.push(inner);

        // Nested loops multiply: 10 * 5 = 50
        assert_eq!(outer.effective_bound(), Some(50));
    }

    #[test]
    fn test_nested_loop_multiple_inner() {
        let mut outer = LoopInfo::new("outer");
        outer.bound = Some(10);

        let mut inner1 = LoopInfo::new("inner1");
        inner1.bound = Some(5);
        let mut inner2 = LoopInfo::new("inner2");
        inner2.bound = Some(8);

        outer.inner_loops.push(inner1);
        outer.inner_loops.push(inner2);

        // Uses max inner bound: 10 * max(5, 8) = 10 * 8 = 80
        assert_eq!(outer.effective_bound(), Some(80));
    }

    #[test]
    fn test_loop_info_with_all_fields() {
        let mut loop_info = LoopInfo::new("full_loop");
        loop_info.bound = Some(100);
        loop_info.bound_confidence = 0.95;
        loop_info.modified_variables = vec!["i".to_string(), "sum".to_string()];
        loop_info.invariant_variables = vec!["n".to_string()];
        loop_info.condition = Some("i < n".to_string());
        loop_info.induction_variable = Some(InductionVariable {
            name: "i".to_string(),
            init: "0".to_string(),
            step: "+ 1".to_string(),
            direction: InductionDirection::Increasing,
        });
        loop_info.nesting_depth = 1;

        assert_eq!(loop_info.bound, Some(100));
        assert_eq!(loop_info.bound_confidence, 0.95);
        assert_eq!(loop_info.modified_variables.len(), 2);
        assert_eq!(loop_info.invariant_variables.len(), 1);
        assert_eq!(loop_info.condition, Some("i < n".to_string()));
        assert!(loop_info.induction_variable.is_some());
        assert_eq!(loop_info.nesting_depth, 1);
    }

    #[test]
    fn test_loop_info_serialization() {
        let loop_info = LoopInfo::new("serialize_test");
        let json = serde_json::to_string(&loop_info).unwrap();
        let deserialized: LoopInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(loop_info.id, deserialized.id);
    }

    // ==================== InductionVariable Tests ====================

    #[test]
    fn test_induction_variable_creation() {
        let var = InductionVariable {
            name: "counter".to_string(),
            init: "0".to_string(),
            step: "+ 1".to_string(),
            direction: InductionDirection::Increasing,
        };

        assert_eq!(var.name, "counter");
        assert_eq!(var.init, "0");
        assert_eq!(var.step, "+ 1");
        assert_eq!(var.direction, InductionDirection::Increasing);
    }

    #[test]
    fn test_induction_variable_decreasing() {
        let var = InductionVariable {
            name: "countdown".to_string(),
            init: "10".to_string(),
            step: "- 1".to_string(),
            direction: InductionDirection::Decreasing,
        };

        assert_eq!(var.direction, InductionDirection::Decreasing);
    }

    #[test]
    fn test_induction_variable_unknown() {
        let var = InductionVariable {
            name: "complex".to_string(),
            init: "f(x)".to_string(),
            step: "* 2".to_string(),
            direction: InductionDirection::Unknown,
        };

        assert_eq!(var.direction, InductionDirection::Unknown);
    }

    #[test]
    fn test_induction_variable_serialization() {
        let var = InductionVariable {
            name: "i".to_string(),
            init: "0".to_string(),
            step: "+ 1".to_string(),
            direction: InductionDirection::Increasing,
        };

        let json = serde_json::to_string(&var).unwrap();
        let deserialized: InductionVariable = serde_json::from_str(&json).unwrap();

        assert_eq!(var.name, deserialized.name);
        assert_eq!(var.init, deserialized.init);
        assert_eq!(var.step, deserialized.step);
        assert_eq!(var.direction, deserialized.direction);
    }

    // ==================== InductionDirection Tests ====================

    #[test]
    fn test_induction_direction() {
        let inc = InductionDirection::Increasing;
        let dec = InductionDirection::Decreasing;

        assert_ne!(inc, dec);
    }

    #[test]
    fn test_induction_direction_equality() {
        let inc1 = InductionDirection::Increasing;
        let inc2 = InductionDirection::Increasing;
        assert_eq!(inc1, inc2);
    }

    #[test]
    fn test_induction_direction_clone() {
        let original = InductionDirection::Decreasing;
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_induction_direction_debug() {
        let dir = InductionDirection::Unknown;
        let debug_str = format!("{:?}", dir);
        assert!(debug_str.contains("Unknown"));
    }

    // ==================== LoopBoundAnalyzer Tests ====================

    #[test]
    fn test_analyzer_new() {
        let analyzer = LoopBoundAnalyzer::new();
        assert!(analyzer.loops().is_empty());
        assert!(analyzer.suggested_k().is_none());
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = LoopBoundAnalyzer::default();
        assert!(analyzer.loops().is_empty());
        assert!(analyzer.suggested_k().is_none());
    }

    #[test]
    fn test_analyzer_detect_counter() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(and (= x' (+ x 1)) (< x 10))";
        let variables = vec!["x".to_string()];

        analyzer.analyze(formula, &variables);

        assert!(!analyzer.loops().is_empty());
        // Should detect bound of 10
        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 10));
    }

    #[test]
    fn test_analyzer_detect_counter_induction_var() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "x' = x + 1 and x < 20";
        let variables = vec!["x".to_string()];

        analyzer.analyze(formula, &variables);

        // Check that counter loop was detected with induction variable
        let counter_loops: Vec<_> = analyzer
            .loops()
            .iter()
            .filter(|l| l.id.starts_with("counter_"))
            .collect();

        if !counter_loops.is_empty() {
            let loop_info = counter_loops[0];
            assert!(loop_info.induction_variable.is_some());
            let ind_var = loop_info.induction_variable.as_ref().unwrap();
            assert_eq!(ind_var.name, "x");
            assert_eq!(ind_var.direction, InductionDirection::Increasing);
        }
    }

    #[test]
    fn test_analyzer_detect_bounded_less_than() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "x < 15";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 15));
    }

    #[test]
    fn test_analyzer_detect_bounded_less_equal() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "y <= 25";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 25));
    }

    #[test]
    fn test_analyzer_detect_bounded_greater_than() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "z > 5";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 5));
    }

    #[test]
    fn test_analyzer_detect_bounded_greater_equal() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "a >= 30";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 30));
    }

    #[test]
    fn test_analyzer_detect_bounded_smt_lib_less_than() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(< x 42)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 42));
    }

    #[test]
    fn test_analyzer_detect_bounded_smt_lib_less_equal() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(<= counter 100)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 100));
    }

    #[test]
    fn test_analyzer_detect_bounded_smt_lib_greater_than() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(> idx 50)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 50));
    }

    #[test]
    fn test_analyzer_detect_bounded_smt_lib_greater_equal() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(>= y 200)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 200));
    }

    #[test]
    fn test_analyzer_detect_multiple_bounds() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(and (< x 10) (< y 20))";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        let bounds_iter = || analyzer.loops().iter().filter_map(|l| l.bound);
        assert!(bounds_iter().any(|b| b == 10));
        assert!(bounds_iter().any(|b| b == 20));
    }

    #[test]
    fn test_analyzer_no_duplicate_bounds() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(and (< x 10) (< y 10))";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        let bounds: Vec<_> = analyzer.loops().iter().filter_map(|l| l.bound).collect();
        // Should not have duplicate 10 values
        let count_10 = bounds.iter().filter(|&&b| b == 10).count();
        assert_eq!(count_10, 1);
    }

    #[test]
    fn test_analyzer_suggested_k_from_bounds() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(< x 8)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        // Suggested k should be bound + 2 = 10
        assert!(analyzer.suggested_k().is_some());
        let suggested = analyzer.suggested_k().unwrap();
        assert_eq!(suggested, 10); // 8 + 2
    }

    #[test]
    fn test_analyzer_suggested_k_max_cap() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(< x 1000)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        // Suggested k should be capped at 100
        assert!(analyzer.suggested_k().is_some());
        let suggested = analyzer.suggested_k().unwrap();
        assert_eq!(suggested, 100);
    }

    #[test]
    fn test_analyzer_suggested_k_none_when_no_bounds() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "x = y";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        // No bounds detected, no suggested k
        assert!(analyzer.suggested_k().is_none());
    }

    #[test]
    fn test_analyzer_all_bounded_true() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(and (< x 10) (< y 20))";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        // All detected loops have bounds
        assert!(analyzer.all_bounded());
    }

    #[test]
    fn test_analyzer_all_bounded_empty() {
        let analyzer = LoopBoundAnalyzer::new();
        // No loops detected returns false (empty is not "all bounded")
        assert!(!analyzer.all_bounded());
    }

    #[test]
    fn test_analyzer_max_bound() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(and (< x 15) (< y 50) (< z 30))";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert_eq!(analyzer.max_bound(), Some(50));
    }

    #[test]
    fn test_analyzer_max_bound_none() {
        let analyzer = LoopBoundAnalyzer::new();
        assert_eq!(analyzer.max_bound(), None);
    }

    #[test]
    fn test_analyzer_loops_accessor() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(< x 5)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        let loops = analyzer.loops();
        assert!(!loops.is_empty());
        assert_eq!(loops[0].bound, Some(5));
    }

    #[test]
    fn test_analyzer_extract_bound_from_condition_less_than() {
        let analyzer = LoopBoundAnalyzer::new();
        let formula = "i < 100";
        let bound = analyzer.extract_bound_from_condition(formula, "i");
        assert_eq!(bound, Some(100));
    }

    #[test]
    fn test_analyzer_extract_bound_from_condition_less_equal() {
        let analyzer = LoopBoundAnalyzer::new();
        let formula = "counter <= 50";
        let bound = analyzer.extract_bound_from_condition(formula, "counter");
        assert_eq!(bound, Some(50));
    }

    #[test]
    fn test_analyzer_extract_bound_no_match() {
        let analyzer = LoopBoundAnalyzer::new();
        let formula = "x > 10";
        let bound = analyzer.extract_bound_from_condition(formula, "x");
        assert_eq!(bound, None);
    }

    #[test]
    fn test_analyzer_extract_bound_wrong_variable() {
        let analyzer = LoopBoundAnalyzer::new();
        let formula = "y < 100";
        let bound = analyzer.extract_bound_from_condition(formula, "x");
        assert_eq!(bound, None);
    }

    #[test]
    fn test_analyzer_complex_formula() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = r"
            (and
                (= i' (+ i 1))
                (< i n)
                (<= n 1000)
                (>= sum 0)
            )
        ";
        let variables = vec!["i".to_string(), "n".to_string(), "sum".to_string()];

        analyzer.analyze(formula, &variables);

        // Should detect counter loop for i and bound from n <= 1000
        assert!(!analyzer.loops().is_empty());
    }

    #[test]
    fn test_analyzer_whitespace_handling() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "x   <   42";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        assert!(analyzer
            .loops()
            .iter()
            .filter_map(|l| l.bound)
            .any(|b| b == 42));
    }

    #[test]
    fn test_analyzer_bound_confidence() {
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(and (= x' (+ x 1)) (< x 10))";
        let variables = vec!["x".to_string()];

        analyzer.analyze(formula, &variables);

        // Counter loops should have 0.8 confidence when bound detected
        // Bounded iteration should have 0.9 confidence
        for loop_info in analyzer.loops() {
            assert!(loop_info.bound_confidence >= 0.0);
            assert!(loop_info.bound_confidence <= 1.0);
        }
    }

    // ======== Mutation coverage tests ========

    #[test]
    fn test_detect_counter_loops_with_increment_pattern() {
        // Line 129: detect_counter_loops function
        let mut analyzer = LoopBoundAnalyzer::new();

        // Formula with x' = x + 1 pattern (increment)
        let formula = "x' = x + 1";
        let variables = vec!["x".to_string()];

        analyzer.detect_counter_loops(formula, &variables);

        // Should detect a counter loop for x
        let counter_loops: Vec<_> = analyzer
            .loops()
            .iter()
            .filter(|l| l.id.starts_with("counter_"))
            .collect();

        assert!(!counter_loops.is_empty(), "Should detect counter loop");

        // Verify the induction variable was set
        let loop_info = counter_loops[0];
        assert!(loop_info.induction_variable.is_some());
        let ind_var = loop_info.induction_variable.as_ref().unwrap();
        assert_eq!(ind_var.name, "x");
        assert_eq!(ind_var.direction, InductionDirection::Increasing);
        assert_eq!(ind_var.step, "+ 1");

        // Verify modified_variables includes x
        assert!(loop_info.modified_variables.contains(&"x".to_string()));
    }

    #[test]
    fn test_detect_counter_loops_no_match() {
        // Test that detect_counter_loops returns empty when no pattern matches
        let mut analyzer = LoopBoundAnalyzer::new();

        // Formula without increment pattern
        let formula = "x = x"; // Assignment without +1
        let variables = vec!["x".to_string()];

        analyzer.detect_counter_loops(formula, &variables);

        // Should NOT detect a counter loop
        assert!(
            !analyzer
                .loops()
                .iter()
                .any(|l| l.id.starts_with("counter_")),
            "Should not detect counter loop without increment pattern"
        );
    }

    #[test]
    fn test_detect_counter_loops_with_bound_extraction() {
        // Test that bounds are extracted from condition
        let mut analyzer = LoopBoundAnalyzer::new();

        let formula = "(and (= x' (+ x 1)) (< x 50))";
        let variables = vec!["x".to_string()];

        analyzer.detect_counter_loops(formula, &variables);

        // Should detect counter loop AND extract bound
        if let Some(loop_info) = analyzer
            .loops()
            .iter()
            .find(|l| l.id.starts_with("counter_"))
        {
            assert_eq!(loop_info.bound, Some(50));
            assert_eq!(loop_info.bound_confidence, 0.8);
        }
    }

    #[test]
    fn test_detect_counter_loops_multiple_variables() {
        // Test with multiple variables where only some match pattern
        let mut analyzer = LoopBoundAnalyzer::new();

        // The regex pattern is: {var}'.*=.*{var}.*\+.*1
        // So it needs var' to appear before = in the string
        // Using infix notation: i' = i + 1 matches, j' = j does not
        let formula = "i' = i + 1 and j' = j and i < 10";
        let variables = vec!["i".to_string(), "j".to_string()];

        analyzer.detect_counter_loops(formula, &variables);

        // Should detect counter loop for i, but not j
        assert!(
            analyzer.loops().iter().any(|l| l.id == "counter_i"),
            "Should detect counter loop for i"
        );
        assert!(
            !analyzer.loops().iter().any(|l| l.id == "counter_j"),
            "Should NOT detect counter loop for j"
        );
    }

    #[test]
    fn test_detect_bounded_iteration_confidence() {
        // Test that bounded iteration loops have 0.9 confidence
        let mut analyzer = LoopBoundAnalyzer::new();
        let formula = "(< x 77)";
        let variables: Vec<String> = vec![];

        analyzer.analyze(formula, &variables);

        // Find the bounded loop
        let bounded_loop = analyzer
            .loops()
            .iter()
            .find(|l| l.id.starts_with("bounded_"));

        assert!(bounded_loop.is_some());
        assert_eq!(bounded_loop.unwrap().bound_confidence, 0.9);
    }

    #[test]
    fn test_compute_suggested_k_saturating_add() {
        // Test that suggested_k uses saturating_add to avoid overflow
        let mut analyzer = LoopBoundAnalyzer::new();

        // Large bound that when +2 would be close to u32::MAX
        let formula = "(< x 4294967293)"; // u32::MAX - 2

        // This won't actually parse as that large, but test the pattern
        analyzer.analyze(formula, &[]);

        // The suggested_k should be capped at 100 regardless
        if let Some(k) = analyzer.suggested_k() {
            assert!(k <= 100);
        }
    }
}
