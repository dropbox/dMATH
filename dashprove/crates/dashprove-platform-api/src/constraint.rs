//! API constraint definitions

use serde::{Deserialize, Serialize};

/// Severity level for constraint violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum Severity {
    /// Will cause crash/undefined behavior
    Critical,
    /// May cause incorrect results
    Error,
    /// May cause performance issues or deprecated usage
    Warning,
    /// Informational/style issue
    Info,
}

impl Severity {
    /// Convert to string for display
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Critical => "critical",
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        }
    }
}

/// Temporal relationship between API calls
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum TemporalRelation {
    /// method_a must be called before method_b
    Before,
    /// method_a must be called after method_b
    After,
    /// method_a must not be called between method_b calls
    NotBetween,
    /// method_a must be called immediately before method_b (no other calls between)
    ImmediatelyBefore,
    /// method_a must be called immediately after method_b
    ImmediatelyAfter,
}

/// Kind of constraint
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Temporal ordering constraint between two methods
    Temporal(TemporalRelation),
    /// Method is forbidden from a particular state
    Forbidden { state: String },
    /// Method is required when in a particular state before transitioning
    Required { state: String },
    /// Methods must be called in pairs (e.g., lock/unlock)
    Paired,
    /// Method must be called at most once per object lifetime
    AtMostOnce,
    /// Method must be called exactly once per object lifetime
    ExactlyOnce,
    /// Custom constraint with predicate expression
    Custom { predicate: String },
}

/// A constraint on API usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConstraint {
    /// The kind of constraint
    pub kind: ConstraintKind,
    /// The primary method this constraint applies to
    pub method_a: String,
    /// The secondary method (for temporal/paired constraints)
    pub method_b: Option<String>,
    /// Human-readable explanation of the constraint
    pub message: String,
    /// Severity of violating this constraint
    pub severity: Severity,
}

impl ApiConstraint {
    /// Create a new "must call before" constraint
    pub fn must_call_before(
        first: impl Into<String>,
        second: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind: ConstraintKind::Temporal(TemporalRelation::Before),
            method_a: first.into(),
            method_b: Some(second.into()),
            message: message.into(),
            severity: Severity::Critical,
        }
    }

    /// Create a new "forbidden from state" constraint
    pub fn forbidden_from_state(
        method: impl Into<String>,
        state: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind: ConstraintKind::Forbidden {
                state: state.into(),
            },
            method_a: method.into(),
            method_b: None,
            message: message.into(),
            severity: Severity::Critical,
        }
    }

    /// Create a paired constraint (e.g., lock/unlock)
    pub fn paired(
        method_a: impl Into<String>,
        method_b: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind: ConstraintKind::Paired,
            method_a: method_a.into(),
            method_b: Some(method_b.into()),
            message: message.into(),
            severity: Severity::Error,
        }
    }

    /// Create an "at most once" constraint
    pub fn at_most_once(method: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: ConstraintKind::AtMostOnce,
            method_a: method.into(),
            method_b: None,
            message: message.into(),
            severity: Severity::Error,
        }
    }

    /// Create an "exactly once" constraint
    pub fn exactly_once(method: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: ConstraintKind::ExactlyOnce,
            method_a: method.into(),
            method_b: None,
            message: message.into(),
            severity: Severity::Critical,
        }
    }

    /// Set severity
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Check if this is a temporal constraint
    pub fn is_temporal(&self) -> bool {
        matches!(self.kind, ConstraintKind::Temporal(_))
    }

    /// Get a short description of the constraint type
    pub fn constraint_type(&self) -> &'static str {
        match &self.kind {
            ConstraintKind::Temporal(TemporalRelation::Before) => "must-call-before",
            ConstraintKind::Temporal(TemporalRelation::After) => "must-call-after",
            ConstraintKind::Temporal(TemporalRelation::NotBetween) => "not-between",
            ConstraintKind::Temporal(TemporalRelation::ImmediatelyBefore) => "immediately-before",
            ConstraintKind::Temporal(TemporalRelation::ImmediatelyAfter) => "immediately-after",
            ConstraintKind::Forbidden { .. } => "forbidden",
            ConstraintKind::Required { .. } => "required",
            ConstraintKind::Paired => "paired",
            ConstraintKind::AtMostOnce => "at-most-once",
            ConstraintKind::ExactlyOnce => "exactly-once",
            ConstraintKind::Custom { .. } => "custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_constraints() {
        let before = ApiConstraint::must_call_before(
            "addCompletedHandler",
            "commit",
            "Handler must be added before commit",
        );
        assert!(before.is_temporal());
        assert_eq!(before.method_a, "addCompletedHandler");
        assert_eq!(before.method_b, Some("commit".to_string()));
        assert_eq!(before.severity, Severity::Critical);

        let forbidden = ApiConstraint::forbidden_from_state(
            "encode",
            "Committed",
            "Cannot encode after commit",
        );
        assert!(!forbidden.is_temporal());

        let paired = ApiConstraint::paired("lock", "unlock", "Every lock must have an unlock");
        assert_eq!(paired.constraint_type(), "paired");
    }

    #[test]
    fn test_severity() {
        assert_eq!(Severity::Critical.as_str(), "critical");
        assert_eq!(Severity::Error.as_str(), "error");
        assert_eq!(Severity::Warning.as_str(), "warning");
        assert_eq!(Severity::Info.as_str(), "info");
    }
}
