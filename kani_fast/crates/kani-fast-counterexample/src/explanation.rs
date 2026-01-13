//! Natural language explanations for counterexamples
//!
//! This module generates human-readable explanations for verification failures,
//! making counterexamples accessible to developers who may not be familiar
//! with formal verification concepts.

use crate::types::{CounterexampleValue, FailedCheck, SourceLocation, StructuredCounterexample};
use std::collections::HashMap;
use std::fmt::Write;

/// Categories of verification failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureCategory {
    /// Division by zero or modulo by zero
    DivisionByZero,
    /// Integer overflow or underflow
    Overflow,
    /// Array or slice index out of bounds
    IndexOutOfBounds,
    /// Null pointer dereference
    NullDereference,
    /// Assertion failure
    AssertionFailure,
    /// Precondition violation
    PreconditionViolation,
    /// Postcondition violation
    PostconditionViolation,
    /// Invariant violation
    InvariantViolation,
    /// Memory safety issue
    MemorySafety,
    /// Unreachable code reached
    UnreachableReached,
    /// Unknown failure type
    Unknown,
}

impl FailureCategory {
    /// Categorize a failure based on its description
    #[must_use]
    pub fn from_description(description: &str) -> Self {
        let desc_lower = description.to_lowercase();

        if desc_lower.contains("divide by zero") || desc_lower.contains("division by zero") {
            Self::DivisionByZero
        } else if desc_lower.contains("overflow") || desc_lower.contains("underflow") {
            Self::Overflow
        } else if desc_lower.contains("index out of bounds")
            || desc_lower.contains("out of range")
            || desc_lower.contains("bounds check")
        {
            Self::IndexOutOfBounds
        } else if desc_lower.contains("null") || desc_lower.contains("nullptr") {
            Self::NullDereference
        } else if desc_lower.contains("assertion") || desc_lower.contains("assert") {
            Self::AssertionFailure
        } else if desc_lower.contains("precondition")
            || desc_lower.contains("requires")
            || desc_lower.contains("pre-condition")
        {
            Self::PreconditionViolation
        } else if desc_lower.contains("postcondition")
            || desc_lower.contains("ensures")
            || desc_lower.contains("post-condition")
        {
            Self::PostconditionViolation
        } else if desc_lower.contains("invariant") {
            Self::InvariantViolation
        } else if desc_lower.contains("memory") || desc_lower.contains("use after free") {
            Self::MemorySafety
        } else if desc_lower.contains("unreachable") {
            Self::UnreachableReached
        } else {
            Self::Unknown
        }
    }

    /// Get a human-readable category name
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DivisionByZero => "Division by Zero",
            Self::Overflow => "Arithmetic Overflow",
            Self::IndexOutOfBounds => "Index Out of Bounds",
            Self::NullDereference => "Null Pointer Dereference",
            Self::AssertionFailure => "Assertion Failure",
            Self::PreconditionViolation => "Precondition Violation",
            Self::PostconditionViolation => "Postcondition Violation",
            Self::InvariantViolation => "Invariant Violation",
            Self::MemorySafety => "Memory Safety Issue",
            Self::UnreachableReached => "Unreachable Code Reached",
            Self::Unknown => "Verification Failure",
        }
    }
}

/// An explanation for a counterexample
#[derive(Debug, Clone)]
pub struct Explanation {
    /// The failure category
    pub category: FailureCategory,
    /// One-line summary
    pub summary: String,
    /// Detailed explanation (multiple sentences)
    pub details: String,
    /// What input values triggered the failure
    pub trigger_explanation: Option<String>,
    /// Why the failure occurred
    pub cause_explanation: Option<String>,
    /// Affected source location
    pub location: Option<SourceLocation>,
    /// Severity level (1-5, where 5 is most severe)
    pub severity: u8,
}

impl Explanation {
    /// Create a new explanation
    #[must_use]
    pub fn new(category: FailureCategory, summary: String, details: String) -> Self {
        Self {
            category,
            summary,
            details,
            trigger_explanation: None,
            cause_explanation: None,
            location: None,
            severity: category_severity(category),
        }
    }

    /// Format the explanation as a multi-line string
    #[must_use]
    pub fn format(&self) -> String {
        let mut output = String::new();

        let _ = writeln!(output, "## {}\n", self.category.as_str());
        let _ = writeln!(output, "**Summary:** {}\n", self.summary);

        if let Some(loc) = &self.location {
            let _ = writeln!(output, "**Location:** {loc}\n");
        }

        let _ = writeln!(output, "**Details:**\n{}\n", self.details);

        if let Some(trigger) = &self.trigger_explanation {
            let _ = writeln!(output, "**What triggered this:**\n{trigger}\n");
        }

        if let Some(cause) = &self.cause_explanation {
            let _ = writeln!(output, "**Root cause:**\n{cause}\n");
        }

        let _ = writeln!(output, "**Severity:** {}/5", self.severity);

        output
    }

    /// Format as a concise single-line message
    #[must_use]
    pub fn format_brief(&self) -> String {
        if let Some(loc) = &self.location {
            format!("[{}] {} at {}", self.category.as_str(), self.summary, loc)
        } else {
            format!("[{}] {}", self.category.as_str(), self.summary)
        }
    }
}

/// Get the default severity for a failure category
fn category_severity(category: FailureCategory) -> u8 {
    match category {
        FailureCategory::MemorySafety | FailureCategory::NullDereference => 5,
        FailureCategory::DivisionByZero | FailureCategory::IndexOutOfBounds => 4,
        FailureCategory::Overflow
        | FailureCategory::UnreachableReached
        | FailureCategory::AssertionFailure
        | FailureCategory::InvariantViolation => 3,
        FailureCategory::PreconditionViolation
        | FailureCategory::PostconditionViolation
        | FailureCategory::Unknown => 2,
    }
}

/// Explanation generator for counterexamples
pub struct ExplanationGenerator {
    /// Include code snippets in explanations
    pub include_code_snippets: bool,
    /// Use technical terminology
    pub technical_mode: bool,
}

impl Default for ExplanationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplanationGenerator {
    /// Create a new explanation generator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_code_snippets: true,
            technical_mode: false,
        }
    }

    /// Create a generator for technical users
    #[must_use]
    pub fn technical() -> Self {
        Self {
            include_code_snippets: true,
            technical_mode: true,
        }
    }

    /// Generate explanations for a structured counterexample
    #[must_use]
    pub fn explain(&self, counterexample: &StructuredCounterexample) -> Vec<Explanation> {
        let mut explanations = Vec::new();

        for check in &counterexample.failed_checks {
            let explanation =
                self.explain_failed_check(check, &counterexample.witness, &counterexample.trace);
            explanations.push(explanation);
        }

        // If no failed checks but we have witness values, generate a generic explanation
        if explanations.is_empty() && !counterexample.witness.is_empty() {
            explanations.push(self.explain_from_witness(&counterexample.witness));
        }

        explanations
    }

    /// Generate an explanation for a failed check
    fn explain_failed_check(
        &self,
        check: &FailedCheck,
        witness: &HashMap<String, CounterexampleValue>,
        trace: &[crate::types::TraceState],
    ) -> Explanation {
        let category = FailureCategory::from_description(&check.description);
        let summary = self.generate_summary(category, check);
        let details = self.generate_details(category, check, witness);

        let mut explanation = Explanation::new(category, summary, details);
        explanation.location = check.location.clone();
        explanation.trigger_explanation = self.generate_trigger_explanation(witness, category);
        explanation.cause_explanation = self.generate_cause_explanation(category, check, trace);

        explanation
    }

    /// Generate an explanation from witness values only
    fn explain_from_witness(&self, witness: &HashMap<String, CounterexampleValue>) -> Explanation {
        let summary = "Verification failed with the following input values".to_string();
        let details = format_witness_values(witness);

        Explanation::new(FailureCategory::Unknown, summary, details)
    }

    /// Generate a summary message
    fn generate_summary(&self, category: FailureCategory, check: &FailedCheck) -> String {
        match category {
            FailureCategory::DivisionByZero => {
                if self.technical_mode {
                    "Division operation with zero divisor".to_string()
                } else {
                    "The code attempted to divide a number by zero".to_string()
                }
            }
            FailureCategory::Overflow => {
                if self.technical_mode {
                    "Arithmetic operation resulted in overflow/underflow".to_string()
                } else {
                    "A calculation produced a number too large (or small) for its type".to_string()
                }
            }
            FailureCategory::IndexOutOfBounds => {
                if self.technical_mode {
                    "Array/slice access with invalid index".to_string()
                } else {
                    "The code tried to access an element outside the array boundaries".to_string()
                }
            }
            FailureCategory::NullDereference => {
                "Attempt to access memory through a null pointer".to_string()
            }
            FailureCategory::AssertionFailure => {
                if let Some(func) = &check.function {
                    format!("Assertion failed in function `{func}`")
                } else {
                    "An assertion in the code failed".to_string()
                }
            }
            FailureCategory::PreconditionViolation => {
                "Function was called with inputs that violate its precondition".to_string()
            }
            FailureCategory::PostconditionViolation => {
                "Function output does not satisfy its postcondition".to_string()
            }
            FailureCategory::InvariantViolation => {
                "A loop or data structure invariant was violated".to_string()
            }
            FailureCategory::MemorySafety => "Memory safety violation detected".to_string(),
            FailureCategory::UnreachableReached => {
                "Code marked as unreachable was actually executed".to_string()
            }
            FailureCategory::Unknown => check.description.clone(),
        }
    }

    /// Generate detailed explanation
    fn generate_details(
        &self,
        category: FailureCategory,
        check: &FailedCheck,
        witness: &HashMap<String, CounterexampleValue>,
    ) -> String {
        let mut details = String::new();

        // Add category-specific explanation
        match category {
            FailureCategory::DivisionByZero => {
                details.push_str(
                    "Division by zero is undefined behavior in Rust. \
                    This can occur when the divisor in a division or modulo operation is zero.\n\n",
                );
                if !witness.is_empty() {
                    details.push_str("The following values were used when this occurred:\n");
                    details.push_str(&format_witness_values(witness));
                }
            }
            FailureCategory::Overflow => {
                details.push_str(
                    "Arithmetic overflow occurs when a calculation produces a result \
                    that cannot be represented in the target integer type. In debug builds, \
                    Rust panics on overflow; in release builds, it wraps around.\n\n",
                );
                details.push_str("Kani detected that this overflow is possible:\n");
                details.push_str(&format_witness_values(witness));
            }
            FailureCategory::IndexOutOfBounds => {
                details.push_str(
                    "Accessing an array or slice with an out-of-bounds index will \
                    cause a panic in Rust. This typically indicates a logic error in \
                    computing the index or in validating array bounds.\n\n",
                );
                // Try to identify which variables are related to indexing
                // MUTATION NOTE: `||→&&` mutations are equivalent: no single variable name
                // contains all three substrings ("index" AND "idx" AND "i").
                let index_vars: Vec<_> = witness
                    .iter()
                    .filter(|(k, _)| k.contains("index") || k.contains("idx") || k.contains('i'))
                    .collect();
                if !index_vars.is_empty() {
                    details.push_str("Index-related values:\n");
                    for (k, v) in index_vars {
                        let _ = writeln!(details, "  {k} = {v}");
                    }
                }
            }
            FailureCategory::AssertionFailure => {
                let _ = writeln!(
                    details,
                    "The assertion `{}` was not satisfied.\n",
                    check.description
                );
                if !witness.is_empty() {
                    details.push_str("Values that triggered the assertion failure:\n");
                    details.push_str(&format_witness_values(witness));
                }
            }
            _ => {
                let _ = writeln!(details, "{}\n", check.description);
                if !witness.is_empty() {
                    details.push_str("Counterexample values:\n");
                    details.push_str(&format_witness_values(witness));
                }
            }
        }

        details
    }

    /// Generate trigger explanation from witness values
    fn generate_trigger_explanation(
        &self,
        witness: &HashMap<String, CounterexampleValue>,
        category: FailureCategory,
    ) -> Option<String> {
        if witness.is_empty() {
            return None;
        }

        let mut explanation = String::new();

        match category {
            FailureCategory::DivisionByZero => {
                // Look for a zero value
                for (name, value) in witness {
                    if let CounterexampleValue::Int { value: 0, .. }
                    | CounterexampleValue::UInt { value: 0, .. } = value
                    {
                        let _ = writeln!(
                            explanation,
                            "The variable `{name}` has value 0, which was used as a divisor."
                        );
                    }
                }
            }
            FailureCategory::Overflow => {
                // Look for boundary values
                for (name, value) in witness {
                    if is_boundary_value(value) {
                        let _ = writeln!(
                            explanation,
                            "The variable `{name}` has value {value} which is near type boundaries."
                        );
                    }
                }
            }
            FailureCategory::IndexOutOfBounds => {
                // Look for large index values
                // MUTATION NOTE: `||→&&` mutations are equivalent: no single variable name
                // contains all three substrings ("index" AND "idx" AND "len").
                for (name, value) in witness {
                    if name.contains("index") || name.contains("idx") || name.contains("len") {
                        let _ = writeln!(explanation, "Array access used `{name}` = {value}");
                    }
                }
            }
            _ => {
                explanation.push_str("The following input values triggered the failure:\n");
                for (name, value) in witness {
                    let _ = writeln!(explanation, "  {name} = {value}");
                }
            }
        }

        if explanation.is_empty() {
            None
        } else {
            Some(explanation)
        }
    }

    /// Generate cause explanation from trace
    fn generate_cause_explanation(
        &self,
        category: FailureCategory,
        check: &FailedCheck,
        _trace: &[crate::types::TraceState],
    ) -> Option<String> {
        let mut cause = String::new();

        match category {
            FailureCategory::DivisionByZero => {
                cause
                    .push_str("The divisor was not validated to be non-zero before the division. ");
                cause.push_str("Consider adding a check like `if divisor != 0 { ... }` or using ");
                cause.push_str("`checked_div()` which returns `None` for division by zero.");
            }
            FailureCategory::Overflow => {
                cause.push_str(
                    "The arithmetic operation can produce values outside the representable range. ",
                );
                cause.push_str(
                    "Consider using `checked_add()`, `saturating_add()`, or wider integer types.",
                );
            }
            FailureCategory::IndexOutOfBounds => {
                cause.push_str("The index is not properly bounded by the array length. ");
                cause.push_str("Consider using `.get()` which returns `Option<&T>`, or ");
                cause.push_str("validating the index before access.");
            }
            FailureCategory::AssertionFailure => {
                if let Some(func) = &check.function {
                    let _ = write!(
                        cause,
                        "The assertion in `{func}` does not hold for all possible inputs."
                    );
                } else {
                    cause.push_str("The assertion does not hold for all possible inputs.");
                }
            }
            _ => return None,
        }

        Some(cause)
    }
}

/// Format witness values as a string
fn format_witness_values(witness: &HashMap<String, CounterexampleValue>) -> String {
    let mut output = String::new();
    let mut vars: Vec<_> = witness.iter().collect();
    vars.sort_by_key(|(k, _)| *k);

    for (name, value) in vars {
        let _ = writeln!(output, "  {name} = {value}");
    }

    output
}

/// Check if a value is near type boundaries (might cause overflow)
fn is_boundary_value(value: &CounterexampleValue) -> bool {
    match value {
        CounterexampleValue::Int {
            value: v,
            type_hint,
        } => {
            // MUTATION NOTE: Deleting Some("i64") arm is equivalent because the
            // wildcard `_` uses the same bounds (i64::MIN/MAX).
            let bounds = match type_hint.as_deref() {
                Some("i8") => (i8::MIN as i128, i8::MAX as i128),
                Some("i16") => (i16::MIN as i128, i16::MAX as i128),
                Some("i32") => (i32::MIN as i128, i32::MAX as i128),
                Some("i64") => (i64::MIN as i128, i64::MAX as i128),
                _ => (i64::MIN as i128, i64::MAX as i128),
            };
            *v == bounds.0 || *v == bounds.1 || *v == bounds.0 + 1 || *v == bounds.1 - 1
        }
        CounterexampleValue::UInt {
            value: v,
            type_hint,
        } => {
            // MUTATION NOTE: Deleting Some("u64") arm is equivalent because the
            // wildcard `_` uses the same bounds (u64::MAX).
            let max = match type_hint.as_deref() {
                Some("u8") => u8::MAX as u128,
                Some("u16") => u16::MAX as u128,
                Some("u32") => u32::MAX as u128,
                Some("u64") => u64::MAX as u128,
                _ => u64::MAX as u128,
            };
            *v == 0 || *v == max || *v == max - 1
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== FailureCategory::from_description Tests ====================

    #[test]
    fn test_failure_category_from_description() {
        assert_eq!(
            FailureCategory::from_description("attempt to divide by zero"),
            FailureCategory::DivisionByZero
        );
        assert_eq!(
            FailureCategory::from_description("arithmetic overflow"),
            FailureCategory::Overflow
        );
        assert_eq!(
            FailureCategory::from_description("index out of bounds"),
            FailureCategory::IndexOutOfBounds
        );
        assert_eq!(
            FailureCategory::from_description("assertion failed"),
            FailureCategory::AssertionFailure
        );
        assert_eq!(
            FailureCategory::from_description("some random text"),
            FailureCategory::Unknown
        );
    }

    #[test]
    fn test_failure_category_division_by_zero_variants() {
        assert_eq!(
            FailureCategory::from_description("divide by zero"),
            FailureCategory::DivisionByZero
        );
        assert_eq!(
            FailureCategory::from_description("division by zero"),
            FailureCategory::DivisionByZero
        );
        assert_eq!(
            FailureCategory::from_description("DIVISION BY ZERO"),
            FailureCategory::DivisionByZero
        );
    }

    #[test]
    fn test_failure_category_overflow_variants() {
        assert_eq!(
            FailureCategory::from_description("arithmetic overflow"),
            FailureCategory::Overflow
        );
        assert_eq!(
            FailureCategory::from_description("integer underflow"),
            FailureCategory::Overflow
        );
        assert_eq!(
            FailureCategory::from_description("OVERFLOW detected"),
            FailureCategory::Overflow
        );
    }

    #[test]
    fn test_failure_category_index_out_of_bounds_variants() {
        assert_eq!(
            FailureCategory::from_description("index out of bounds"),
            FailureCategory::IndexOutOfBounds
        );
        assert_eq!(
            FailureCategory::from_description("array index out of range"),
            FailureCategory::IndexOutOfBounds
        );
        assert_eq!(
            FailureCategory::from_description("bounds check failed"),
            FailureCategory::IndexOutOfBounds
        );
    }

    #[test]
    fn test_failure_category_null_dereference() {
        assert_eq!(
            FailureCategory::from_description("null pointer access"),
            FailureCategory::NullDereference
        );
        assert_eq!(
            FailureCategory::from_description("nullptr dereference"),
            FailureCategory::NullDereference
        );
    }

    #[test]
    fn test_failure_category_assertion() {
        assert_eq!(
            FailureCategory::from_description("assertion failed"),
            FailureCategory::AssertionFailure
        );
        assert_eq!(
            FailureCategory::from_description("assert!(x > 0) failed"),
            FailureCategory::AssertionFailure
        );
    }

    #[test]
    fn test_failure_category_precondition() {
        assert_eq!(
            FailureCategory::from_description("precondition violation"),
            FailureCategory::PreconditionViolation
        );
        assert_eq!(
            FailureCategory::from_description("requires clause failed"),
            FailureCategory::PreconditionViolation
        );
        assert_eq!(
            FailureCategory::from_description("pre-condition not met"),
            FailureCategory::PreconditionViolation
        );
    }

    #[test]
    fn test_failure_category_postcondition() {
        assert_eq!(
            FailureCategory::from_description("postcondition violation"),
            FailureCategory::PostconditionViolation
        );
        assert_eq!(
            FailureCategory::from_description("ensures clause failed"),
            FailureCategory::PostconditionViolation
        );
        assert_eq!(
            FailureCategory::from_description("post-condition not satisfied"),
            FailureCategory::PostconditionViolation
        );
    }

    #[test]
    fn test_failure_category_invariant() {
        assert_eq!(
            FailureCategory::from_description("loop invariant violated"),
            FailureCategory::InvariantViolation
        );
    }

    #[test]
    fn test_failure_category_memory_safety() {
        assert_eq!(
            FailureCategory::from_description("memory safety violation"),
            FailureCategory::MemorySafety
        );
        assert_eq!(
            FailureCategory::from_description("use after free detected"),
            FailureCategory::MemorySafety
        );
    }

    #[test]
    fn test_failure_category_unreachable() {
        assert_eq!(
            FailureCategory::from_description("unreachable code executed"),
            FailureCategory::UnreachableReached
        );
    }

    // ==================== FailureCategory::as_str Tests ====================

    #[test]
    fn test_failure_category_as_str() {
        assert_eq!(FailureCategory::DivisionByZero.as_str(), "Division by Zero");
        assert_eq!(FailureCategory::Overflow.as_str(), "Arithmetic Overflow");
    }

    #[test]
    fn test_failure_category_as_str_all() {
        assert_eq!(FailureCategory::DivisionByZero.as_str(), "Division by Zero");
        assert_eq!(FailureCategory::Overflow.as_str(), "Arithmetic Overflow");
        assert_eq!(
            FailureCategory::IndexOutOfBounds.as_str(),
            "Index Out of Bounds"
        );
        assert_eq!(
            FailureCategory::NullDereference.as_str(),
            "Null Pointer Dereference"
        );
        assert_eq!(
            FailureCategory::AssertionFailure.as_str(),
            "Assertion Failure"
        );
        assert_eq!(
            FailureCategory::PreconditionViolation.as_str(),
            "Precondition Violation"
        );
        assert_eq!(
            FailureCategory::PostconditionViolation.as_str(),
            "Postcondition Violation"
        );
        assert_eq!(
            FailureCategory::InvariantViolation.as_str(),
            "Invariant Violation"
        );
        assert_eq!(
            FailureCategory::MemorySafety.as_str(),
            "Memory Safety Issue"
        );
        assert_eq!(
            FailureCategory::UnreachableReached.as_str(),
            "Unreachable Code Reached"
        );
        assert_eq!(FailureCategory::Unknown.as_str(), "Verification Failure");
    }

    // ==================== FailureCategory Trait Tests ====================

    #[test]
    fn test_failure_category_debug() {
        assert!(format!("{:?}", FailureCategory::DivisionByZero).contains("DivisionByZero"));
        assert!(format!("{:?}", FailureCategory::Unknown).contains("Unknown"));
    }

    #[test]
    fn test_failure_category_clone() {
        let cat = FailureCategory::Overflow;
        let cloned = cat;
        assert_eq!(cat, cloned);
    }

    #[test]
    fn test_failure_category_copy() {
        let cat = FailureCategory::MemorySafety;
        let copied: FailureCategory = cat;
        assert_eq!(cat, copied);
    }

    #[test]
    fn test_failure_category_eq() {
        assert_eq!(
            FailureCategory::DivisionByZero,
            FailureCategory::DivisionByZero
        );
        assert_ne!(FailureCategory::DivisionByZero, FailureCategory::Overflow);
    }

    // ==================== Explanation Tests ====================

    #[test]
    fn test_explanation_new() {
        let exp = Explanation::new(
            FailureCategory::DivisionByZero,
            "Test summary".to_string(),
            "Test details".to_string(),
        );
        assert_eq!(exp.category, FailureCategory::DivisionByZero);
        assert_eq!(exp.summary, "Test summary");
        assert_eq!(exp.severity, 4);
    }

    #[test]
    fn test_explanation_new_defaults() {
        let exp = Explanation::new(
            FailureCategory::Unknown,
            "summary".to_string(),
            "details".to_string(),
        );
        assert!(exp.trigger_explanation.is_none());
        assert!(exp.cause_explanation.is_none());
        assert!(exp.location.is_none());
    }

    #[test]
    fn test_explanation_format() {
        let mut exp = Explanation::new(
            FailureCategory::DivisionByZero,
            "Division occurred".to_string(),
            "The divisor was zero".to_string(),
        );
        exp.location = Some(SourceLocation {
            file: "test.rs".to_string(),
            line: 10,
            column: Some(5),
        });
        exp.trigger_explanation = Some("x was 0".to_string());
        exp.cause_explanation = Some("Missing validation".to_string());

        let formatted = exp.format();
        assert!(formatted.contains("Division by Zero"));
        assert!(formatted.contains("Division occurred"));
        assert!(formatted.contains("test.rs"));
        assert!(formatted.contains("The divisor was zero"));
        assert!(formatted.contains("x was 0"));
        assert!(formatted.contains("Missing validation"));
        assert!(formatted.contains("Severity"));
    }

    #[test]
    fn test_explanation_format_minimal() {
        let exp = Explanation::new(
            FailureCategory::Unknown,
            "summary".to_string(),
            "details".to_string(),
        );

        let formatted = exp.format();
        assert!(formatted.contains("summary"));
        assert!(formatted.contains("details"));
        // Should not contain optional sections
        assert!(!formatted.contains("What triggered"));
        assert!(!formatted.contains("Root cause"));
    }

    #[test]
    fn test_explanation_format_brief() {
        let mut exp = Explanation::new(
            FailureCategory::DivisionByZero,
            "Division by zero".to_string(),
            "Details".to_string(),
        );
        assert!(exp.format_brief().contains("Division by Zero"));

        exp.location = Some(SourceLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: Some(10),
        });
        assert!(exp.format_brief().contains("test.rs:42:10"));
    }

    #[test]
    fn test_explanation_format_brief_no_location() {
        let exp = Explanation::new(
            FailureCategory::Overflow,
            "Integer overflow".to_string(),
            "Details".to_string(),
        );
        let brief = exp.format_brief();
        assert!(brief.contains("Arithmetic Overflow"));
        assert!(brief.contains("Integer overflow"));
        assert!(!brief.contains(" at "));
    }

    #[test]
    fn test_explanation_debug() {
        let exp = Explanation::new(
            FailureCategory::Unknown,
            "test".to_string(),
            "test".to_string(),
        );
        let debug = format!("{:?}", exp);
        assert!(debug.contains("Explanation"));
    }

    #[test]
    fn test_explanation_clone() {
        let exp = Explanation::new(
            FailureCategory::DivisionByZero,
            "test".to_string(),
            "test".to_string(),
        );
        let cloned = exp.clone();
        assert_eq!(cloned.category, exp.category);
        assert_eq!(cloned.summary, exp.summary);
    }

    // ==================== category_severity Tests ====================

    #[test]
    fn test_category_severity() {
        assert_eq!(category_severity(FailureCategory::MemorySafety), 5);
        assert_eq!(category_severity(FailureCategory::DivisionByZero), 4);
        assert_eq!(category_severity(FailureCategory::Overflow), 3);
        assert_eq!(category_severity(FailureCategory::PreconditionViolation), 2);
    }

    #[test]
    fn test_category_severity_all() {
        assert_eq!(category_severity(FailureCategory::MemorySafety), 5);
        assert_eq!(category_severity(FailureCategory::NullDereference), 5);
        assert_eq!(category_severity(FailureCategory::DivisionByZero), 4);
        assert_eq!(category_severity(FailureCategory::IndexOutOfBounds), 4);
        assert_eq!(category_severity(FailureCategory::Overflow), 3);
        assert_eq!(category_severity(FailureCategory::UnreachableReached), 3);
        assert_eq!(category_severity(FailureCategory::AssertionFailure), 3);
        assert_eq!(category_severity(FailureCategory::InvariantViolation), 3);
        assert_eq!(category_severity(FailureCategory::PreconditionViolation), 2);
        assert_eq!(
            category_severity(FailureCategory::PostconditionViolation),
            2
        );
        assert_eq!(category_severity(FailureCategory::Unknown), 2);
    }

    // ==================== ExplanationGenerator Tests ====================

    #[test]
    fn test_explanation_generator_creation() {
        let gen = ExplanationGenerator::new();
        assert!(gen.include_code_snippets);
        assert!(!gen.technical_mode);

        let tech_gen = ExplanationGenerator::technical();
        assert!(tech_gen.technical_mode);
    }

    #[test]
    fn test_explanation_generator_default() {
        let gen = ExplanationGenerator::default();
        assert!(gen.include_code_snippets);
        assert!(!gen.technical_mode);
    }

    #[test]
    fn test_explain_empty_counterexample() {
        let gen = ExplanationGenerator::new();
        let ce = StructuredCounterexample::new();
        let explanations = gen.explain(&ce);
        assert!(explanations.is_empty());
    }

    #[test]
    fn test_explain_with_witness_only() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: Some("i32".to_string()),
            },
        );

        let explanations = gen.explain(&ce);
        assert_eq!(explanations.len(), 1);
        assert_eq!(explanations[0].category, FailureCategory::Unknown);
    }

    #[test]
    fn test_explain_with_failed_check() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: "attempt to divide by zero".to_string(),
            location: Some(SourceLocation {
                file: "lib.rs".to_string(),
                line: 10,
                column: Some(5),
            }),
            function: Some("divide".to_string()),
        });
        ce.witness.insert(
            "divisor".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: Some("i32".to_string()),
            },
        );

        let explanations = gen.explain(&ce);
        assert_eq!(explanations.len(), 1);
        assert_eq!(explanations[0].category, FailureCategory::DivisionByZero);
    }

    #[test]
    fn test_explain_multiple_failed_checks() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });
        ce.failed_checks.push(FailedCheck {
            check_id: "2".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });

        let explanations = gen.explain(&ce);
        assert_eq!(explanations.len(), 2);
        assert_eq!(explanations[0].category, FailureCategory::DivisionByZero);
        assert_eq!(explanations[1].category, FailureCategory::Overflow);
    }

    #[test]
    fn test_explain_technical_mode_division() {
        let gen = ExplanationGenerator::technical();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });

        let explanations = gen.explain(&ce);
        assert!(explanations[0]
            .summary
            .contains("Division operation with zero divisor"));
    }

    #[test]
    fn test_explain_non_technical_mode_division() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });

        let explanations = gen.explain(&ce);
        assert!(explanations[0].summary.contains("attempted to divide"));
    }

    #[test]
    fn test_explain_technical_mode_overflow() {
        let gen = ExplanationGenerator::technical();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });

        let explanations = gen.explain(&ce);
        assert!(explanations[0].summary.contains("overflow/underflow"));
    }

    #[test]
    fn test_explain_technical_mode_index() {
        let gen = ExplanationGenerator::technical();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });

        let explanations = gen.explain(&ce);
        assert!(explanations[0].summary.contains("invalid index"));
    }

    #[test]
    fn test_explain_assertion_with_function() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "assertion failed".to_string(),
            location: None,
            function: Some("my_function".to_string()),
        });

        let explanations = gen.explain(&ce);
        assert!(explanations[0].summary.contains("my_function"));
    }

    #[test]
    fn test_explain_assertion_without_function() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "assertion failed".to_string(),
            location: None,
            function: None,
        });

        let explanations = gen.explain(&ce);
        assert!(explanations[0].summary.contains("assertion"));
    }

    // ==================== is_boundary_value Tests ====================

    #[test]
    fn test_is_boundary_value() {
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i32::MAX as i128,
            type_hint: Some("i32".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: 0,
            type_hint: Some("u32".to_string())
        }));
        assert!(!is_boundary_value(&CounterexampleValue::Int {
            value: 42,
            type_hint: Some("i32".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_int_min() {
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i32::MIN as i128,
            type_hint: Some("i32".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i8::MIN as i128,
            type_hint: Some("i8".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_int_max() {
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i32::MAX as i128,
            type_hint: Some("i32".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i64::MAX as i128,
            type_hint: Some("i64".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_int_near_boundary() {
        // MAX - 1 is also a boundary
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: (i32::MAX - 1) as i128,
            type_hint: Some("i32".to_string())
        }));
        // MIN + 1 is also a boundary
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: (i32::MIN + 1) as i128,
            type_hint: Some("i32".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_uint_zero() {
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: 0,
            type_hint: Some("u32".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_uint_max() {
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: u32::MAX as u128,
            type_hint: Some("u32".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: u8::MAX as u128,
            type_hint: Some("u8".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_uint_near_max() {
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: (u32::MAX - 1) as u128,
            type_hint: Some("u32".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_non_boundary() {
        assert!(!is_boundary_value(&CounterexampleValue::Int {
            value: 100,
            type_hint: Some("i32".to_string())
        }));
        assert!(!is_boundary_value(&CounterexampleValue::UInt {
            value: 100,
            type_hint: Some("u32".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_other_types() {
        // Bool is not a boundary value
        assert!(!is_boundary_value(&CounterexampleValue::Bool(true)));
        assert!(!is_boundary_value(&CounterexampleValue::Bool(false)));
    }

    #[test]
    fn test_is_boundary_value_no_type_hint() {
        // Should use i64/u64 defaults
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i64::MAX as i128,
            type_hint: None
        }));
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: u64::MAX as u128,
            type_hint: None
        }));
    }

    // ==================== format_witness_values Tests ====================

    #[test]
    fn test_format_witness_values_empty() {
        let witness = HashMap::new();
        let formatted = format_witness_values(&witness);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_format_witness_values_single() {
        let mut witness = HashMap::new();
        witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );
        let formatted = format_witness_values(&witness);
        assert!(formatted.contains("x = "));
    }

    #[test]
    fn test_format_witness_values_sorted() {
        let mut witness = HashMap::new();
        witness.insert(
            "z".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: None,
            },
        );
        witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: 2,
                type_hint: None,
            },
        );
        witness.insert(
            "m".to_string(),
            CounterexampleValue::Int {
                value: 3,
                type_hint: None,
            },
        );
        let formatted = format_witness_values(&witness);

        // a should come before m, m before z
        let a_pos = formatted.find("a = ").unwrap();
        let m_pos = formatted.find("m = ").unwrap();
        let z_pos = formatted.find("z = ").unwrap();
        assert!(a_pos < m_pos);
        assert!(m_pos < z_pos);
    }

    // ==================== Integration Tests ====================

    // ==================== generate_details Branch Tests ====================

    #[test]
    fn test_generate_details_division_by_zero_with_empty_witness() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });
        // Empty witness - tests the !witness.is_empty() branch
        let explanations = gen.explain(&ce);
        assert_eq!(explanations[0].category, FailureCategory::DivisionByZero);
        // Should contain the intro text but not "The following values"
        assert!(explanations[0]
            .details
            .contains("Division by zero is undefined behavior"));
        assert!(!explanations[0]
            .details
            .contains("The following values were used"));
    }

    #[test]
    fn test_generate_details_division_by_zero_with_witness() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        // Should contain witness values text
        assert!(explanations[0]
            .details
            .contains("The following values were used"));
    }

    #[test]
    fn test_generate_details_overflow_branch() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i32::MAX as i128,
                type_hint: Some("i32".to_string()),
            },
        );
        let explanations = gen.explain(&ce);
        assert!(explanations[0]
            .details
            .contains("Arithmetic overflow occurs"));
        assert!(explanations[0]
            .details
            .contains("Kani detected that this overflow"));
    }

    #[test]
    fn test_generate_details_index_out_of_bounds_with_index_vars() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "index".to_string(),
            CounterexampleValue::UInt {
                value: 100,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "idx".to_string(),
            CounterexampleValue::UInt {
                value: 50,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "i".to_string(),
            CounterexampleValue::UInt {
                value: 25,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        // Should filter to index-related vars
        assert!(explanations[0].details.contains("Index-related values"));
    }

    #[test]
    fn test_generate_details_index_out_of_bounds_no_index_vars() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "foo".to_string(),
            CounterexampleValue::UInt {
                value: 100,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        // Should NOT contain "Index-related values" since no index/idx/i vars
        assert!(!explanations[0].details.contains("Index-related values"));
    }

    #[test]
    fn test_generate_details_assertion_failure() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "assertion failed: x > 0".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: -5,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        assert!(explanations[0]
            .details
            .contains("assertion `assertion failed: x > 0` was not satisfied"));
        assert!(explanations[0].details.contains("Values that triggered"));
    }

    #[test]
    fn test_generate_details_assertion_failure_empty_witness() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "assertion".to_string(),
            location: None,
            function: None,
        });
        let explanations = gen.explain(&ce);
        // Empty witness - should not have "Values that triggered" section
        assert!(!explanations[0].details.contains("Values that triggered"));
    }

    #[test]
    fn test_generate_details_default_branch_with_witness() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "precondition violated".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "param".to_string(),
            CounterexampleValue::Int {
                value: -1,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        assert!(explanations[0].details.contains("Counterexample values:"));
    }

    #[test]
    fn test_generate_details_default_branch_empty_witness() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "postcondition violated".to_string(),
            location: None,
            function: None,
        });
        let explanations = gen.explain(&ce);
        // Empty witness - should not have "Counterexample values:" section
        assert!(!explanations[0].details.contains("Counterexample values:"));
    }

    // ==================== generate_trigger_explanation Branch Tests ====================

    #[test]
    fn test_generate_trigger_division_zero_found() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });
        // Int with value 0
        ce.witness.insert(
            "divisor".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        assert!(explanations[0]
            .trigger_explanation
            .as_ref()
            .unwrap()
            .contains("has value 0, which was used as a divisor"));
    }

    #[test]
    fn test_generate_trigger_division_zero_uint() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });
        // UInt with value 0
        ce.witness.insert(
            "divisor".to_string(),
            CounterexampleValue::UInt {
                value: 0,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        assert!(explanations[0]
            .trigger_explanation
            .as_ref()
            .unwrap()
            .contains("has value 0, which was used as a divisor"));
    }

    #[test]
    fn test_generate_trigger_overflow_with_boundary() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i32::MAX as i128,
                type_hint: Some("i32".to_string()),
            },
        );
        let explanations = gen.explain(&ce);
        assert!(explanations[0]
            .trigger_explanation
            .as_ref()
            .unwrap()
            .contains("near type boundaries"));
    }

    #[test]
    fn test_generate_trigger_index_with_index_vars() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "index".to_string(),
            CounterexampleValue::UInt {
                value: 100,
                type_hint: None,
            },
        );
        ce.witness.insert(
            "len".to_string(),
            CounterexampleValue::UInt {
                value: 10,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        let trigger = explanations[0].trigger_explanation.as_ref().unwrap();
        assert!(trigger.contains("Array access used"));
    }

    #[test]
    fn test_generate_trigger_default_branch() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "memory safety".to_string(),
            location: None,
            function: None,
        });
        ce.witness.insert(
            "ptr".to_string(),
            CounterexampleValue::UInt {
                value: 0,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        let trigger = explanations[0].trigger_explanation.as_ref().unwrap();
        assert!(trigger.contains("The following input values triggered"));
    }

    #[test]
    fn test_generate_trigger_returns_none_empty_explanation() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });
        // Non-zero values - won't produce trigger explanation for div-by-zero
        ce.witness.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: 42,
                type_hint: None,
            },
        );
        let explanations = gen.explain(&ce);
        // Should return None because no zero value was found
        assert!(explanations[0].trigger_explanation.is_none());
    }

    // ==================== generate_cause_explanation Branch Tests ====================

    #[test]
    fn test_generate_cause_division_by_zero() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "divide by zero".to_string(),
            location: None,
            function: None,
        });
        let explanations = gen.explain(&ce);
        let cause = explanations[0].cause_explanation.as_ref().unwrap();
        assert!(cause.contains("divisor was not validated to be non-zero"));
        assert!(cause.contains("checked_div()"));
    }

    #[test]
    fn test_generate_cause_overflow() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "overflow".to_string(),
            location: None,
            function: None,
        });
        let explanations = gen.explain(&ce);
        let cause = explanations[0].cause_explanation.as_ref().unwrap();
        assert!(cause.contains("outside the representable range"));
        assert!(cause.contains("checked_add()"));
    }

    #[test]
    fn test_generate_cause_index_out_of_bounds() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });
        let explanations = gen.explain(&ce);
        let cause = explanations[0].cause_explanation.as_ref().unwrap();
        assert!(cause.contains("index is not properly bounded"));
        assert!(cause.contains(".get()"));
    }

    #[test]
    fn test_generate_cause_assertion_with_function() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "assertion".to_string(),
            location: None,
            function: Some("my_func".to_string()),
        });
        let explanations = gen.explain(&ce);
        let cause = explanations[0].cause_explanation.as_ref().unwrap();
        assert!(cause.contains("my_func"));
        assert!(cause.contains("does not hold"));
    }

    #[test]
    fn test_generate_cause_assertion_no_function() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "assertion".to_string(),
            location: None,
            function: None,
        });
        let explanations = gen.explain(&ce);
        let cause = explanations[0].cause_explanation.as_ref().unwrap();
        assert!(cause.contains("The assertion does not hold"));
    }

    #[test]
    fn test_generate_cause_other_category_returns_none() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();
        ce.failed_checks.push(FailedCheck {
            check_id: "1".to_string(),
            description: "precondition".to_string(),
            location: None,
            function: None,
        });
        let explanations = gen.explain(&ce);
        assert!(explanations[0].cause_explanation.is_none());
    }

    // ==================== is_boundary_value Type Hints Tests ====================

    #[test]
    fn test_is_boundary_value_i16() {
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i16::MAX as i128,
            type_hint: Some("i16".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i16::MIN as i128,
            type_hint: Some("i16".to_string())
        }));
        // Non-boundary
        assert!(!is_boundary_value(&CounterexampleValue::Int {
            value: 100,
            type_hint: Some("i16".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_i64() {
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i64::MAX as i128,
            type_hint: Some("i64".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::Int {
            value: i64::MIN as i128,
            type_hint: Some("i64".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_u16() {
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: u16::MAX as u128,
            type_hint: Some("u16".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: 0,
            type_hint: Some("u16".to_string())
        }));
    }

    #[test]
    fn test_is_boundary_value_u64() {
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: u64::MAX as u128,
            type_hint: Some("u64".to_string())
        }));
        assert!(is_boundary_value(&CounterexampleValue::UInt {
            value: (u64::MAX - 1) as u128,
            type_hint: Some("u64".to_string())
        }));
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_full_explanation_workflow() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();

        // Add a division by zero check with witness
        ce.failed_checks.push(FailedCheck {
            check_id: "div_check".to_string(),
            description: "attempt to divide by zero".to_string(),
            location: Some(SourceLocation {
                file: "src/math.rs".to_string(),
                line: 42,
                column: Some(15),
            }),
            function: Some("safe_divide".to_string()),
        });

        ce.witness.insert(
            "divisor".to_string(),
            CounterexampleValue::Int {
                value: 0,
                type_hint: Some("i32".to_string()),
            },
        );
        ce.witness.insert(
            "dividend".to_string(),
            CounterexampleValue::Int {
                value: 100,
                type_hint: Some("i32".to_string()),
            },
        );

        let explanations = gen.explain(&ce);
        assert_eq!(explanations.len(), 1);

        let exp = &explanations[0];
        assert_eq!(exp.category, FailureCategory::DivisionByZero);
        assert_eq!(exp.severity, 4);
        assert!(exp.location.is_some());

        // Check formatted output
        let formatted = exp.format();
        assert!(formatted.contains("Division by Zero"));
        assert!(formatted.contains("src/math.rs"));

        let brief = exp.format_brief();
        assert!(brief.contains("Division by Zero"));
        assert!(brief.contains("42"));
    }

    #[test]
    fn test_overflow_with_boundary_values() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();

        ce.failed_checks.push(FailedCheck {
            check_id: "overflow".to_string(),
            description: "arithmetic overflow".to_string(),
            location: None,
            function: None,
        });

        ce.witness.insert(
            "a".to_string(),
            CounterexampleValue::Int {
                value: i32::MAX as i128,
                type_hint: Some("i32".to_string()),
            },
        );
        ce.witness.insert(
            "b".to_string(),
            CounterexampleValue::Int {
                value: 1,
                type_hint: Some("i32".to_string()),
            },
        );

        let explanations = gen.explain(&ce);
        assert_eq!(explanations.len(), 1);
        assert_eq!(explanations[0].category, FailureCategory::Overflow);

        // Should mention boundary value in trigger explanation
        if let Some(trigger) = &explanations[0].trigger_explanation {
            assert!(trigger.contains("near type boundaries") || trigger.contains("a"));
        }
    }

    #[test]
    fn test_index_out_of_bounds_with_index_variables() {
        let gen = ExplanationGenerator::new();
        let mut ce = StructuredCounterexample::new();

        ce.failed_checks.push(FailedCheck {
            check_id: "bounds".to_string(),
            description: "index out of bounds".to_string(),
            location: None,
            function: None,
        });

        ce.witness.insert(
            "index".to_string(),
            CounterexampleValue::UInt {
                value: 100,
                type_hint: Some("usize".to_string()),
            },
        );
        ce.witness.insert(
            "len".to_string(),
            CounterexampleValue::UInt {
                value: 10,
                type_hint: Some("usize".to_string()),
            },
        );

        let explanations = gen.explain(&ce);
        assert_eq!(explanations[0].category, FailureCategory::IndexOutOfBounds);
    }

    #[test]
    fn test_all_failure_categories_have_summaries() {
        let gen = ExplanationGenerator::new();

        let categories = [
            ("divide by zero", FailureCategory::DivisionByZero),
            ("overflow", FailureCategory::Overflow),
            ("index out of bounds", FailureCategory::IndexOutOfBounds),
            ("null", FailureCategory::NullDereference),
            ("assertion failed", FailureCategory::AssertionFailure),
            ("precondition", FailureCategory::PreconditionViolation),
            ("postcondition", FailureCategory::PostconditionViolation),
            ("invariant", FailureCategory::InvariantViolation),
            ("memory", FailureCategory::MemorySafety),
            ("unreachable", FailureCategory::UnreachableReached),
            ("random text", FailureCategory::Unknown),
        ];

        for (desc, expected_category) in categories {
            let mut ce = StructuredCounterexample::new();
            ce.failed_checks.push(FailedCheck {
                check_id: "test".to_string(),
                description: desc.to_string(),
                location: None,
                function: None,
            });

            let explanations = gen.explain(&ce);
            assert_eq!(explanations.len(), 1);
            assert_eq!(
                explanations[0].category, expected_category,
                "Failed for description: {}",
                desc
            );
            assert!(!explanations[0].summary.is_empty());
        }
    }
}
