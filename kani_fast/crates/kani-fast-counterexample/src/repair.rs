//! Repair suggestions for counterexamples
//!
//! This module generates actionable repair suggestions based on the type
//! of verification failure. Suggestions include code patterns, defensive
//! programming techniques, and links to documentation.

use crate::explanation::FailureCategory;
use crate::types::{CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample};
use std::collections::HashMap;
use std::fmt::Write;

/// A repair suggestion for fixing a verification failure
#[derive(Debug, Clone)]
pub struct RepairSuggestion {
    /// Brief title for the suggestion
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Code snippet showing the fix pattern
    pub code_snippet: Option<String>,
    /// Original problematic code (if available)
    pub original_code: Option<String>,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Where the fix should be applied
    pub location: Option<SourceLocation>,
    /// Related Rust documentation links
    pub doc_links: Vec<String>,
    /// Type of repair strategy
    pub strategy: RepairStrategy,
}

/// Types of repair strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Add input validation
    InputValidation,
    /// Use checked arithmetic
    CheckedArithmetic,
    /// Use saturating arithmetic
    SaturatingArithmetic,
    /// Use wider types
    WiderTypes,
    /// Add bounds checking
    BoundsCheck,
    /// Use Option/Result
    OptionResult,
    /// Add precondition
    AddPrecondition,
    /// Strengthen invariant
    StrengthenInvariant,
    /// Use safe API alternatives
    SafeApi,
    /// Add defensive checks
    DefensiveCheck,
    /// Refactor logic
    RefactorLogic,
}

impl RepairStrategy {
    /// Get a human-readable name
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InputValidation => "Input Validation",
            Self::CheckedArithmetic => "Checked Arithmetic",
            Self::SaturatingArithmetic => "Saturating Arithmetic",
            Self::WiderTypes => "Use Wider Types",
            Self::BoundsCheck => "Bounds Checking",
            Self::OptionResult => "Use Option/Result",
            Self::AddPrecondition => "Add Precondition",
            Self::StrengthenInvariant => "Strengthen Invariant",
            Self::SafeApi => "Use Safe APIs",
            Self::DefensiveCheck => "Defensive Programming",
            Self::RefactorLogic => "Refactor Logic",
        }
    }
}

impl RepairSuggestion {
    /// Create a new repair suggestion
    #[must_use]
    pub fn new(
        title: String,
        description: String,
        strategy: RepairStrategy,
        confidence: f64,
    ) -> Self {
        Self {
            title,
            description,
            code_snippet: None,
            original_code: None,
            confidence,
            location: None,
            doc_links: Vec::new(),
            strategy,
        }
    }

    /// Add a code snippet
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code_snippet = Some(code.into());
        self
    }

    /// Add documentation links
    #[must_use]
    pub fn with_docs(mut self, links: Vec<String>) -> Self {
        self.doc_links = links;
        self
    }

    /// Add source location
    #[must_use]
    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Format the suggestion as a string
    #[must_use]
    pub fn format(&self) -> String {
        let mut output = String::new();

        let _ = writeln!(
            output,
            "### {} ({:.0}% confidence)\n",
            self.title,
            self.confidence * 100.0
        );
        let _ = writeln!(output, "**Strategy:** {}\n", self.strategy.as_str());
        let _ = writeln!(output, "{}\n", self.description);

        if let Some(snippet) = &self.code_snippet {
            output.push_str("**Suggested fix:**\n```rust\n");
            output.push_str(snippet);
            output.push_str("\n```\n\n");
        }

        if let Some(loc) = &self.location {
            let _ = writeln!(output, "**Location:** {loc}\n");
        }

        if !self.doc_links.is_empty() {
            output.push_str("**Documentation:**\n");
            for link in &self.doc_links {
                let _ = writeln!(output, "- {link}");
            }
            output.push('\n');
        }

        output
    }
}

/// Engine for generating repair suggestions
pub struct RepairEngine {
    /// Include alternative suggestions
    pub include_alternatives: bool,
    /// Maximum number of suggestions to generate
    pub max_suggestions: usize,
}

impl Default for RepairEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RepairEngine {
    /// Create a new repair engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_alternatives: true,
            max_suggestions: 3,
        }
    }

    /// Generate repair suggestions for a counterexample
    #[must_use]
    pub fn suggest(&self, counterexample: &StructuredCounterexample) -> Vec<RepairSuggestion> {
        let mut suggestions = Vec::new();

        for check in &counterexample.failed_checks {
            let category = FailureCategory::from_description(&check.description);
            let mut check_suggestions =
                self.suggest_for_category(category, check, &counterexample.witness);
            suggestions.append(&mut check_suggestions);
        }

        // Sort by confidence and limit
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(self.max_suggestions);

        suggestions
    }

    /// Generate suggestions for a specific failure category
    fn suggest_for_category(
        &self,
        category: FailureCategory,
        check: &FailedCheck,
        witness: &HashMap<String, CounterexampleValue>,
    ) -> Vec<RepairSuggestion> {
        match category {
            FailureCategory::DivisionByZero => self.suggest_division_fix(check, witness),
            FailureCategory::Overflow => self.suggest_overflow_fix(check, witness),
            FailureCategory::IndexOutOfBounds => self.suggest_bounds_fix(check, witness),
            FailureCategory::NullDereference => self.suggest_null_fix(check),
            FailureCategory::AssertionFailure => self.suggest_assertion_fix(check, witness),
            FailureCategory::PreconditionViolation => self.suggest_precondition_fix(check),
            FailureCategory::PostconditionViolation => self.suggest_postcondition_fix(check),
            FailureCategory::InvariantViolation => self.suggest_invariant_fix(check),
            _ => self.suggest_generic_fix(check),
        }
    }

    /// Generate suggestions for division by zero
    fn suggest_division_fix(
        &self,
        check: &FailedCheck,
        witness: &HashMap<String, CounterexampleValue>,
    ) -> Vec<RepairSuggestion> {
        let mut suggestions = Vec::new();

        // Find the likely divisor variable
        let divisor_var = find_zero_variable(witness);

        // Primary suggestion: input validation
        let mut validation = RepairSuggestion::new(
            "Add divisor validation".to_string(),
            "Check that the divisor is not zero before performing division.".to_string(),
            RepairStrategy::InputValidation,
            0.95,
        );

        let var_name = divisor_var.unwrap_or("divisor");
        validation.code_snippet = Some(format!(
            "// Before the division:\nif {var_name} == 0 {{\n    return Err(\"Division by zero\");\n}}\nlet result = numerator / {var_name};"
        ));
        validation.doc_links =
            vec!["https://doc.rust-lang.org/std/primitive.i32.html#method.checked_div".to_string()];

        if let Some(loc) = &check.location {
            validation = validation.with_location(loc.clone());
        }

        suggestions.push(validation);

        // Alternative: use checked_div
        if self.include_alternatives {
            let checked = RepairSuggestion::new(
                "Use checked_div()".to_string(),
                "Use checked arithmetic that returns None on division by zero.".to_string(),
                RepairStrategy::CheckedArithmetic,
                0.85,
            )
            .with_code(format!(
                "// Returns None if {var_name} is 0\nlet result = numerator.checked_div({var_name}).ok_or(\"Division by zero\")?;"
            ))
            .with_docs(vec![
                "https://doc.rust-lang.org/std/primitive.i32.html#method.checked_div".to_string(),
            ]);

            suggestions.push(checked);
        }

        suggestions
    }

    /// Generate suggestions for arithmetic overflow
    fn suggest_overflow_fix(
        &self,
        check: &FailedCheck,
        witness: &HashMap<String, CounterexampleValue>,
    ) -> Vec<RepairSuggestion> {
        let mut suggestions = Vec::new();

        // Identify the type from witness
        let inferred_type = infer_type_from_witness(witness);

        // Primary: checked arithmetic
        let mut checked = RepairSuggestion::new(
            "Use checked arithmetic".to_string(),
            "Replace standard arithmetic with checked operations that return None on overflow."
                .to_string(),
            RepairStrategy::CheckedArithmetic,
            0.90,
        )
        .with_code(
            "// Instead of: let result = a + b;\nlet result = a.checked_add(b).expect(\"Overflow\");",
        )
        .with_docs(vec![
            "https://doc.rust-lang.org/std/primitive.i32.html#method.checked_add".to_string(),
            "https://doc.rust-lang.org/std/primitive.i32.html#method.checked_mul".to_string(),
        ]);

        if let Some(loc) = &check.location {
            checked.location = Some(loc.clone());
        }
        suggestions.push(checked);

        // Alternative: saturating arithmetic
        if self.include_alternatives {
            let saturating = RepairSuggestion::new(
                "Use saturating arithmetic".to_string(),
                "Clamp results to type boundaries instead of overflowing.".to_string(),
                RepairStrategy::SaturatingArithmetic,
                0.80,
            )
            .with_code(
                "// Clamps to type max/min instead of overflowing\nlet result = a.saturating_add(b);",
            )
            .with_docs(vec![
                "https://doc.rust-lang.org/std/primitive.i32.html#method.saturating_add".to_string(),
            ]);

            suggestions.push(saturating);

            // Alternative: wider types
            let wider_type = suggest_wider_type(&inferred_type);
            if let Some(wider) = wider_type {
                let widen = RepairSuggestion::new(
                    format!("Use {wider} instead"),
                    format!(
                        "Use a wider integer type ({wider}) that can represent larger values."
                    ),
                    RepairStrategy::WiderTypes,
                    0.70,
                )
                .with_code(format!(
                    "// Change variable type\nlet a: {wider} = value as {wider};\nlet result = a + b;"
                ));

                suggestions.push(widen);
            }
        }

        suggestions
    }

    /// Generate suggestions for index out of bounds
    fn suggest_bounds_fix(
        &self,
        check: &FailedCheck,
        _witness: &HashMap<String, CounterexampleValue>,
    ) -> Vec<RepairSuggestion> {
        let mut suggestions = Vec::new();

        // Primary: use .get()
        let safe_access = RepairSuggestion::new(
            "Use .get() for safe access".to_string(),
            "Use .get() which returns Option<&T> instead of panicking on invalid index."
                .to_string(),
            RepairStrategy::SafeApi,
            0.90,
        )
        .with_code(
            "// Instead of: let value = array[index];\nlet value = array.get(index).ok_or(\"Index out of bounds\")?;",
        )
        .with_docs(vec![
            "https://doc.rust-lang.org/std/vec/struct.Vec.html#method.get".to_string(),
        ]);

        if let Some(loc) = &check.location {
            suggestions.push(safe_access.with_location(loc.clone()));
        } else {
            suggestions.push(safe_access);
        }

        // Alternative: validate bounds
        if self.include_alternatives {
            let bounds_check = RepairSuggestion::new(
                "Add bounds validation".to_string(),
                "Validate the index against the array length before access.".to_string(),
                RepairStrategy::BoundsCheck,
                0.85,
            )
            .with_code(
                "// Before array access:\nif index >= array.len() {\n    return Err(\"Index out of bounds\");\n}\nlet value = array[index];",
            );

            suggestions.push(bounds_check);
        }

        suggestions
    }

    /// Generate suggestions for null dereference
    fn suggest_null_fix(&self, check: &FailedCheck) -> Vec<RepairSuggestion> {
        let mut suggestions = Vec::new();

        let mut option_use = RepairSuggestion::new(
            "Use Option<T> pattern".to_string(),
            "Rust's Option type explicitly handles the absence of a value.".to_string(),
            RepairStrategy::OptionResult,
            0.90,
        )
        .with_code(
            "// Use Option instead of nullable pointer\nlet value: Option<T> = get_value();\nif let Some(v) = value {\n    // Use v safely\n}",
        )
        .with_docs(vec![
            "https://doc.rust-lang.org/std/option/enum.Option.html".to_string(),
        ]);

        if let Some(loc) = &check.location {
            option_use.location = Some(loc.clone());
        }
        suggestions.push(option_use);

        suggestions
    }

    /// Generate suggestions for assertion failures
    fn suggest_assertion_fix(
        &self,
        check: &FailedCheck,
        _witness: &HashMap<String, CounterexampleValue>,
    ) -> Vec<RepairSuggestion> {
        let mut suggestions = Vec::new();

        let strengthen = RepairSuggestion::new(
            "Strengthen preconditions".to_string(),
            "Add input validation to ensure the assertion's assumptions hold.".to_string(),
            RepairStrategy::AddPrecondition,
            0.75,
        )
        .with_code(
            "// Add validation before the operation:\nfn process(input: i32) -> Result<i32, Error> {\n    // Precondition check\n    if !valid_input(input) {\n        return Err(Error::InvalidInput);\n    }\n    // Now the assertion should hold\n    assert!(input > 0);\n    Ok(compute(input))\n}",
        );

        if let Some(loc) = &check.location {
            suggestions.push(strengthen.with_location(loc.clone()));
        } else {
            suggestions.push(strengthen);
        }

        // Alternative: refactor logic
        if self.include_alternatives {
            let refactor = RepairSuggestion::new(
                "Review algorithm logic".to_string(),
                format!(
                    "The assertion `{}` may indicate a logic error that needs deeper analysis.",
                    check.description
                ),
                RepairStrategy::RefactorLogic,
                0.60,
            );

            suggestions.push(refactor);
        }

        suggestions
    }

    /// Generate suggestions for precondition violations
    fn suggest_precondition_fix(&self, check: &FailedCheck) -> Vec<RepairSuggestion> {
        let suggestion = RepairSuggestion::new(
            "Validate inputs at call site".to_string(),
            "Ensure callers satisfy the function's preconditions.".to_string(),
            RepairStrategy::InputValidation,
            0.85,
        )
        .with_code(
            "// Before calling the function:\nif !precondition_holds(args) {\n    return Err(\"Precondition not satisfied\");\n}\nlet result = function(args);",
        );

        if let Some(loc) = &check.location {
            vec![suggestion.with_location(loc.clone())]
        } else {
            vec![suggestion]
        }
    }

    /// Generate suggestions for postcondition violations
    fn suggest_postcondition_fix(&self, check: &FailedCheck) -> Vec<RepairSuggestion> {
        let suggestion = RepairSuggestion::new(
            "Fix implementation to satisfy postcondition".to_string(),
            "The function's implementation does not guarantee the specified postcondition."
                .to_string(),
            RepairStrategy::RefactorLogic,
            0.80,
        )
        .with_code(
            "// Ensure the postcondition is established:\nfn compute(x: i32) -> i32 {\n    let result = /* computation */;\n    debug_assert!(postcondition(result)); // Verify\n    result\n}",
        );

        if let Some(loc) = &check.location {
            vec![suggestion.with_location(loc.clone())]
        } else {
            vec![suggestion]
        }
    }

    /// Generate suggestions for invariant violations
    fn suggest_invariant_fix(&self, check: &FailedCheck) -> Vec<RepairSuggestion> {
        let suggestion = RepairSuggestion::new(
            "Strengthen loop invariant".to_string(),
            "The loop invariant is not maintained across iterations.".to_string(),
            RepairStrategy::StrengthenInvariant,
            0.75,
        )
        .with_code(
            "// Ensure invariant is preserved:\nwhile condition {\n    // Assert invariant at loop entry\n    debug_assert!(invariant_holds());\n    \n    // Loop body that maintains invariant\n    \n    // Assert invariant at loop exit\n    debug_assert!(invariant_holds());\n}",
        );

        if let Some(loc) = &check.location {
            vec![suggestion.with_location(loc.clone())]
        } else {
            vec![suggestion]
        }
    }

    /// Generate generic suggestions
    fn suggest_generic_fix(&self, check: &FailedCheck) -> Vec<RepairSuggestion> {
        let suggestion = RepairSuggestion::new(
            "Add defensive checks".to_string(),
            format!(
                "Add input validation and error handling around: {}",
                check.description
            ),
            RepairStrategy::DefensiveCheck,
            0.50,
        );

        if let Some(loc) = &check.location {
            vec![suggestion.with_location(loc.clone())]
        } else {
            vec![suggestion]
        }
    }
}

/// Find a variable with zero value in the witness
fn find_zero_variable(witness: &HashMap<String, CounterexampleValue>) -> Option<&str> {
    for (name, value) in witness {
        match value {
            CounterexampleValue::Int { value: 0, .. }
            | CounterexampleValue::UInt { value: 0, .. } => {
                return Some(name.as_str());
            }
            _ => {}
        }
    }
    None
}

/// Infer the likely type from witness values
fn infer_type_from_witness(witness: &HashMap<String, CounterexampleValue>) -> Option<String> {
    for value in witness.values() {
        match value {
            CounterexampleValue::Int { type_hint, .. }
            | CounterexampleValue::UInt { type_hint, .. } => {
                if let Some(hint) = type_hint {
                    return Some(hint.clone());
                }
            }
            _ => {}
        }
    }
    None
}

/// Suggest a wider type for the given type
fn suggest_wider_type(current: &Option<String>) -> Option<&'static str> {
    match current.as_deref() {
        Some("i8") => Some("i16"),
        Some("i16") => Some("i32"),
        Some("i32") => Some("i64"),
        Some("i64") => Some("i128"),
        Some("u8") => Some("u16"),
        Some("u16") => Some("u32"),
        Some("u32") => Some("u64"),
        Some("u64") => Some("u128"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== RepairStrategy Tests ====================

    #[test]
    fn test_repair_strategy_as_str() {
        assert_eq!(RepairStrategy::InputValidation.as_str(), "Input Validation");
        assert_eq!(
            RepairStrategy::CheckedArithmetic.as_str(),
            "Checked Arithmetic"
        );
    }

    #[test]
    fn test_repair_strategy_as_str_all_variants() {
        assert_eq!(RepairStrategy::InputValidation.as_str(), "Input Validation");
        assert_eq!(
            RepairStrategy::CheckedArithmetic.as_str(),
            "Checked Arithmetic"
        );
        assert_eq!(
            RepairStrategy::SaturatingArithmetic.as_str(),
            "Saturating Arithmetic"
        );
        assert_eq!(RepairStrategy::WiderTypes.as_str(), "Use Wider Types");
        assert_eq!(RepairStrategy::BoundsCheck.as_str(), "Bounds Checking");
        assert_eq!(RepairStrategy::OptionResult.as_str(), "Use Option/Result");
        assert_eq!(RepairStrategy::AddPrecondition.as_str(), "Add Precondition");
        assert_eq!(
            RepairStrategy::StrengthenInvariant.as_str(),
            "Strengthen Invariant"
        );
        assert_eq!(RepairStrategy::SafeApi.as_str(), "Use Safe APIs");
        assert_eq!(
            RepairStrategy::DefensiveCheck.as_str(),
            "Defensive Programming"
        );
        assert_eq!(RepairStrategy::RefactorLogic.as_str(), "Refactor Logic");
    }

    #[test]
    fn test_repair_strategy_debug() {
        let strategy = RepairStrategy::InputValidation;
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("InputValidation"));
    }

    #[test]
    fn test_repair_strategy_clone() {
        let strategy = RepairStrategy::CheckedArithmetic;
        // Use Clone::clone to explicitly test Clone trait (strategy is Copy, so .clone() triggers clippy)
        let cloned = Clone::clone(&strategy);
        assert_eq!(strategy, cloned);
    }

    #[test]
    fn test_repair_strategy_copy() {
        let strategy = RepairStrategy::BoundsCheck;
        let copied: RepairStrategy = strategy;
        assert_eq!(strategy, copied);
    }

    #[test]
    fn test_repair_strategy_eq() {
        assert_eq!(RepairStrategy::SafeApi, RepairStrategy::SafeApi);
        assert_ne!(RepairStrategy::SafeApi, RepairStrategy::WiderTypes);
    }

    // ==================== RepairSuggestion Construction Tests ====================

    #[test]
    fn test_repair_suggestion_new() {
        let suggestion = RepairSuggestion::new(
            "Test".to_string(),
            "Description".to_string(),
            RepairStrategy::InputValidation,
            0.9,
        );
        assert_eq!(suggestion.title, "Test");
        assert_eq!(suggestion.confidence, 0.9);
    }

    #[test]
    fn test_repair_suggestion_new_defaults() {
        let suggestion = RepairSuggestion::new(
            "Title".to_string(),
            "Desc".to_string(),
            RepairStrategy::SafeApi,
            0.5,
        );
        assert!(suggestion.code_snippet.is_none());
        assert!(suggestion.original_code.is_none());
        assert!(suggestion.location.is_none());
        assert!(suggestion.doc_links.is_empty());
    }

    #[test]
    fn test_repair_suggestion_with_code() {
        let suggestion = RepairSuggestion::new(
            "Test".to_string(),
            "Description".to_string(),
            RepairStrategy::InputValidation,
            0.9,
        )
        .with_code("let x = 5;");

        assert_eq!(suggestion.code_snippet, Some("let x = 5;".to_string()));
    }

    #[test]
    fn test_repair_suggestion_with_code_from_string() {
        let code = String::from("fn test() {}");
        let suggestion = RepairSuggestion::new(
            "Test".to_string(),
            "Desc".to_string(),
            RepairStrategy::RefactorLogic,
            0.8,
        )
        .with_code(code);

        assert_eq!(suggestion.code_snippet, Some("fn test() {}".to_string()));
    }

    #[test]
    fn test_repair_suggestion_with_docs() {
        let suggestion = RepairSuggestion::new(
            "Test".to_string(),
            "Desc".to_string(),
            RepairStrategy::CheckedArithmetic,
            0.85,
        )
        .with_docs(vec![
            "https://doc.rust-lang.org/test1".to_string(),
            "https://doc.rust-lang.org/test2".to_string(),
        ]);

        assert_eq!(suggestion.doc_links.len(), 2);
        assert!(suggestion.doc_links[0].contains("test1"));
    }

    #[test]
    fn test_repair_suggestion_with_location() {
        let loc = SourceLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: Some(10),
        };

        let suggestion = RepairSuggestion::new(
            "Test".to_string(),
            "Desc".to_string(),
            RepairStrategy::BoundsCheck,
            0.9,
        )
        .with_location(loc.clone());

        assert!(suggestion.location.is_some());
        assert_eq!(suggestion.location.as_ref().unwrap().line, 42);
    }

    #[test]
    fn test_repair_suggestion_chain_builders() {
        let loc = SourceLocation {
            file: "src/lib.rs".to_string(),
            line: 100,
            column: None,
        };

        let suggestion = RepairSuggestion::new(
            "Chained".to_string(),
            "Chained description".to_string(),
            RepairStrategy::OptionResult,
            0.75,
        )
        .with_code("Option::Some(x)")
        .with_docs(vec!["https://example.com".to_string()])
        .with_location(loc);

        assert!(suggestion.code_snippet.is_some());
        assert!(!suggestion.doc_links.is_empty());
        assert!(suggestion.location.is_some());
    }

    #[test]
    fn test_repair_suggestion_debug() {
        let suggestion = RepairSuggestion::new(
            "Debug Test".to_string(),
            "Testing debug".to_string(),
            RepairStrategy::DefensiveCheck,
            0.6,
        );
        let debug = format!("{:?}", suggestion);
        assert!(debug.contains("Debug Test"));
        assert!(debug.contains("DefensiveCheck"));
    }

    #[test]
    fn test_repair_suggestion_clone() {
        let suggestion = RepairSuggestion::new(
            "Clone Test".to_string(),
            "Testing clone".to_string(),
            RepairStrategy::WiderTypes,
            0.7,
        )
        .with_code("u64 instead of u32");

        let cloned = suggestion.clone();
        assert_eq!(suggestion.title, cloned.title);
        assert_eq!(suggestion.code_snippet, cloned.code_snippet);
        assert_eq!(suggestion.confidence, cloned.confidence);
    }

    // ==================== RepairSuggestion::format() Tests ====================

    #[test]
    fn test_repair_suggestion_format_basic() {
        let suggestion = RepairSuggestion::new(
            "Test Suggestion".to_string(),
            "A test description.".to_string(),
            RepairStrategy::InputValidation,
            0.95,
        );

        let formatted = suggestion.format();
        assert!(formatted.contains("Test Suggestion"));
        assert!(formatted.contains("95%"));
        assert!(formatted.contains("Input Validation"));
        assert!(formatted.contains("A test description."));
    }

    #[test]
    fn test_repair_suggestion_format_with_code() {
        let suggestion = RepairSuggestion::new(
            "Code Suggestion".to_string(),
            "With code.".to_string(),
            RepairStrategy::CheckedArithmetic,
            0.8,
        )
        .with_code("let x = a.checked_add(b);");

        let formatted = suggestion.format();
        assert!(formatted.contains("Suggested fix:"));
        assert!(formatted.contains("```rust"));
        assert!(formatted.contains("checked_add"));
    }

    #[test]
    fn test_repair_suggestion_format_with_location() {
        let loc = SourceLocation {
            file: "src/main.rs".to_string(),
            line: 50,
            column: Some(5),
        };

        let suggestion = RepairSuggestion::new(
            "Located Fix".to_string(),
            "Has location.".to_string(),
            RepairStrategy::SafeApi,
            0.85,
        )
        .with_location(loc);

        let formatted = suggestion.format();
        assert!(formatted.contains("Location:"));
        assert!(formatted.contains("src/main.rs"));
    }

    #[test]
    fn test_repair_suggestion_format_with_docs() {
        let suggestion = RepairSuggestion::new(
            "Doc Links".to_string(),
            "With documentation.".to_string(),
            RepairStrategy::OptionResult,
            0.9,
        )
        .with_docs(vec![
            "https://doc.rust-lang.org/option".to_string(),
            "https://doc.rust-lang.org/result".to_string(),
        ]);

        let formatted = suggestion.format();
        assert!(formatted.contains("Documentation:"));
        assert!(formatted.contains("option"));
        assert!(formatted.contains("result"));
    }

    #[test]
    fn test_repair_suggestion_format_complete() {
        let loc = SourceLocation {
            file: "lib.rs".to_string(),
            line: 99,
            column: None,
        };

        let suggestion = RepairSuggestion::new(
            "Complete Example".to_string(),
            "All fields set.".to_string(),
            RepairStrategy::BoundsCheck,
            0.88,
        )
        .with_code("array.get(i)")
        .with_docs(vec!["https://example.com".to_string()])
        .with_location(loc);

        let formatted = suggestion.format();
        assert!(formatted.contains("Complete Example"));
        assert!(formatted.contains("88%"));
        assert!(formatted.contains("Bounds Checking"));
        assert!(formatted.contains("Suggested fix:"));
        assert!(formatted.contains("Location:"));
        assert!(formatted.contains("Documentation:"));
    }

    #[test]
    fn test_repair_suggestion_format_zero_confidence() {
        let suggestion = RepairSuggestion::new(
            "Zero Conf".to_string(),
            "Desc".to_string(),
            RepairStrategy::RefactorLogic,
            0.0,
        );

        let formatted = suggestion.format();
        assert!(formatted.contains("0%"));
    }

    #[test]
    fn test_repair_suggestion_format_full_confidence() {
        let suggestion = RepairSuggestion::new(
            "Full Conf".to_string(),
            "Desc".to_string(),
            RepairStrategy::AddPrecondition,
            1.0,
        );

        let formatted = suggestion.format();
        assert!(formatted.contains("100%"));
    }

    // ==================== RepairEngine Tests ====================

    #[test]
    fn test_repair_engine_creation() {
        let engine = RepairEngine::new();
        assert!(engine.include_alternatives);
        assert_eq!(engine.max_suggestions, 3);
    }

    #[test]
    fn test_repair_engine_default() {
        let engine = RepairEngine::default();
        assert!(engine.include_alternatives);
        assert_eq!(engine.max_suggestions, 3);
    }

    #[test]
    fn test_repair_engine_custom_config() {
        let mut engine = RepairEngine::new();
        engine.include_alternatives = false;
        engine.max_suggestions = 5;

        assert!(!engine.include_alternatives);
        assert_eq!(engine.max_suggestions, 5);
    }

    #[test]
    fn test_suggest_empty_counterexample() {
        let engine = RepairEngine::new();
        let ce = StructuredCounterexample::new();
        let suggestions = engine.suggest(&ce);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_division_by_zero() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "attempt to divide by zero".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: Some("i32".to_string()),
            },
        );

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(
            suggestions[0].title.contains("divisor") || suggestions[0].title.contains("checked")
        );
    }

    #[test]
    fn test_suggest_division_with_location() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        let loc = SourceLocation {
            file: "div.rs".to_string(),
            line: 10,
            column: None,
        };
        ce.failed_checks.push(FailedCheck {
            check_id: "div1".to_string(),
            description: "division by zero".to_string(),
            location: Some(loc),
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(suggestions.iter().any(|s| s.location.is_some()));
    }

    #[test]
    fn test_suggest_division_no_alternatives() {
        let mut engine = RepairEngine::new();
        engine.include_alternatives = false;

        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "div".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        // Should get only primary suggestion
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_suggest_overflow() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "arithmetic overflow".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::CheckedArithmetic));
    }

    #[test]
    fn test_suggest_overflow_with_type_hint() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "of".to_string(),
            description: "overflow occurred".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i64::MAX as i128,
                type_hint: Some("i32".to_string()),
            },
        );

        let suggestions = engine.suggest(&ce);
        // Should include wider type suggestion
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::WiderTypes
                || s.strategy == RepairStrategy::CheckedArithmetic
                || s.strategy == RepairStrategy::SaturatingArithmetic));
    }

    #[test]
    fn test_suggest_overflow_no_alternatives() {
        let mut engine = RepairEngine::new();
        engine.include_alternatives = false;

        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "of".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        // Only primary suggestion
        assert!(suggestions
            .iter()
            .all(|s| s.strategy == RepairStrategy::CheckedArithmetic));
    }

    #[test]
    fn test_suggest_bounds() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::SafeApi
                || s.strategy == RepairStrategy::BoundsCheck));
    }

    #[test]
    fn test_suggest_bounds_with_location() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        let loc = SourceLocation {
            file: "array.rs".to_string(),
            line: 25,
            column: Some(8),
        };
        ce.failed_checks.push(FailedCheck {
            check_id: "bounds1".to_string(),
            description: "index out of bounds".to_string(),
            location: Some(loc),
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(suggestions.iter().any(|s| s.location.is_some()));
    }

    #[test]
    fn test_suggest_null_dereference() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "null1".to_string(),
            description: "null pointer dereference".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::OptionResult));
    }

    #[test]
    fn test_suggest_assertion_failure() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "assert1".to_string(),
            description: "assertion `x > 0` failed".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::AddPrecondition
                || s.strategy == RepairStrategy::RefactorLogic));
    }

    #[test]
    fn test_suggest_precondition_violation() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "pre1".to_string(),
            description: "precondition violated".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::InputValidation));
    }

    #[test]
    fn test_suggest_postcondition_violation() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "post1".to_string(),
            description: "postcondition not satisfied".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::RefactorLogic));
    }

    #[test]
    fn test_suggest_invariant_violation() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "inv1".to_string(),
            description: "invariant violated".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::StrengthenInvariant));
    }

    #[test]
    fn test_suggest_generic_failure() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "gen1".to_string(),
            description: "some unknown failure type".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.strategy == RepairStrategy::DefensiveCheck));
    }

    #[test]
    fn test_suggest_multiple_checks() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "c1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });
        ce.failed_checks.push(FailedCheck {
            check_id: "c2".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        // Should get suggestions for both types
        assert!(!suggestions.is_empty());
        // Limited by max_suggestions
        assert!(suggestions.len() <= engine.max_suggestions);
    }

    #[test]
    fn test_suggest_sorted_by_confidence() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "c1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        for window in suggestions.windows(2) {
            assert!(window[0].confidence >= window[1].confidence);
        }
    }

    #[test]
    fn test_suggest_max_suggestions_limit() {
        let mut engine = RepairEngine::new();
        engine.max_suggestions = 1;

        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "c1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(suggestions.len() <= 1);
    }

    // ==================== Helper Function Tests ====================

    #[test]
    fn test_find_zero_variable() {
        let mut witness = HashMap::new();
        witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        );
        witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );

        assert_eq!(find_zero_variable(&witness), Some("y"));
    }

    #[test]
    fn test_find_zero_variable_uint() {
        let mut witness = HashMap::new();
        witness.insert(
            "index".to_string(),
            CounterexampleValue::UInt {
                value: 0,
                type_hint: Some("usize".to_string()),
            },
        );

        assert_eq!(find_zero_variable(&witness), Some("index"));
    }

    #[test]
    fn test_find_zero_variable_none() {
        let mut witness = HashMap::new();
        witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 5,
                type_hint: None,
            },
        );
        witness.insert(
            "y".to_string(),
            CounterexampleValue::Int {
                value: 10,
                type_hint: None,
            },
        );

        assert!(find_zero_variable(&witness).is_none());
    }

    #[test]
    fn test_find_zero_variable_empty() {
        let witness = HashMap::new();
        assert!(find_zero_variable(&witness).is_none());
    }

    #[test]
    fn test_find_zero_variable_non_numeric() {
        let mut witness = HashMap::new();
        witness.insert("flag".to_string(), CounterexampleValue::Bool(true));
        witness.insert(
            "name".to_string(),
            CounterexampleValue::String("test".to_string()),
        );

        assert!(find_zero_variable(&witness).is_none());
    }

    #[test]
    fn test_infer_type_from_witness_int() {
        let mut witness = HashMap::new();
        witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: Some("i32".to_string()),
            },
        );

        assert_eq!(infer_type_from_witness(&witness), Some("i32".to_string()));
    }

    #[test]
    fn test_infer_type_from_witness_uint() {
        let mut witness = HashMap::new();
        witness.insert(
            "len".to_string(),
            CounterexampleValue::UInt {
                value: 100,
                type_hint: Some("usize".to_string()),
            },
        );

        assert_eq!(infer_type_from_witness(&witness), Some("usize".to_string()));
    }

    #[test]
    fn test_infer_type_from_witness_no_hint() {
        let mut witness = HashMap::new();
        witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );

        assert!(infer_type_from_witness(&witness).is_none());
    }

    #[test]
    fn test_infer_type_from_witness_empty() {
        let witness = HashMap::new();
        assert!(infer_type_from_witness(&witness).is_none());
    }

    #[test]
    fn test_infer_type_from_witness_non_numeric() {
        let mut witness = HashMap::new();
        witness.insert("flag".to_string(), CounterexampleValue::Bool(false));

        assert!(infer_type_from_witness(&witness).is_none());
    }

    #[test]
    fn test_suggest_wider_type() {
        assert_eq!(suggest_wider_type(&Some("i32".to_string())), Some("i64"));
        assert_eq!(suggest_wider_type(&Some("u8".to_string())), Some("u16"));
        assert_eq!(suggest_wider_type(&None), None);
    }

    #[test]
    fn test_suggest_wider_type_all_signed() {
        assert_eq!(suggest_wider_type(&Some("i8".to_string())), Some("i16"));
        assert_eq!(suggest_wider_type(&Some("i16".to_string())), Some("i32"));
        assert_eq!(suggest_wider_type(&Some("i32".to_string())), Some("i64"));
        assert_eq!(suggest_wider_type(&Some("i64".to_string())), Some("i128"));
        assert_eq!(suggest_wider_type(&Some("i128".to_string())), None);
    }

    #[test]
    fn test_suggest_wider_type_all_unsigned() {
        assert_eq!(suggest_wider_type(&Some("u8".to_string())), Some("u16"));
        assert_eq!(suggest_wider_type(&Some("u16".to_string())), Some("u32"));
        assert_eq!(suggest_wider_type(&Some("u32".to_string())), Some("u64"));
        assert_eq!(suggest_wider_type(&Some("u64".to_string())), Some("u128"));
        assert_eq!(suggest_wider_type(&Some("u128".to_string())), None);
    }

    #[test]
    fn test_suggest_wider_type_unknown() {
        assert_eq!(suggest_wider_type(&Some("isize".to_string())), None);
        assert_eq!(suggest_wider_type(&Some("f32".to_string())), None);
        assert_eq!(suggest_wider_type(&Some("char".to_string())), None);
    }

    // ==================== Edge Case and Integration Tests ====================

    #[test]
    fn test_suggestion_code_with_multiline() {
        let suggestion = RepairSuggestion::new(
            "Multiline".to_string(),
            "Desc".to_string(),
            RepairStrategy::InputValidation,
            0.9,
        )
        .with_code("if x == 0 {\n    return Err(\"zero\");\n}");

        let formatted = suggestion.format();
        assert!(formatted.contains("```rust"));
        assert!(formatted.contains("return Err"));
    }

    #[test]
    fn test_suggestion_empty_description() {
        let suggestion = RepairSuggestion::new(
            "Title Only".to_string(),
            String::new(),
            RepairStrategy::SafeApi,
            0.5,
        );

        let formatted = suggestion.format();
        assert!(formatted.contains("Title Only"));
    }

    #[test]
    fn test_counterexample_with_all_value_types() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();

        ce.witness.insert(
            "i".to_string(),
            CounterexampleValue::Int {
                value: -5,
                type_hint: Some("i32".to_string()),
            },
        );
        ce.witness.insert(
            "u".to_string(),
            CounterexampleValue::UInt {
                value: 10,
                type_hint: Some("u64".to_string()),
            },
        );
        ce.witness
            .insert("b".to_string(), CounterexampleValue::Bool(true));
        ce.witness.insert(
            "s".to_string(),
            CounterexampleValue::String("test".to_string()),
        );

        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "generic check".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_confidence_bounds() {
        // Test that suggestions have valid confidence values
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "c1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });
        ce.failed_checks.push(FailedCheck {
            check_id: "c2".to_string(),
            description: "bounds".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        for s in &suggestions {
            assert!(s.confidence >= 0.0);
            assert!(s.confidence <= 1.0);
        }
    }

    #[test]
    fn test_doc_links_are_valid_urls() {
        let engine = RepairEngine::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "c1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });

        let suggestions = engine.suggest(&ce);
        for s in &suggestions {
            for link in &s.doc_links {
                assert!(link.starts_with("https://"));
            }
        }
    }
}
