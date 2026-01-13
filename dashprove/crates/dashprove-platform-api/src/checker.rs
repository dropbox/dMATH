//! API constraint checker

use crate::api::PlatformApi;
use crate::constraint::{ApiConstraint, ConstraintKind, Severity, TemporalRelation};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Severity of a violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum ViolationSeverity {
    Critical,
    Error,
    Warning,
    Info,
}

impl From<Severity> for ViolationSeverity {
    fn from(s: Severity) -> Self {
        match s {
            Severity::Critical => ViolationSeverity::Critical,
            Severity::Error => ViolationSeverity::Error,
            Severity::Warning => ViolationSeverity::Warning,
            Severity::Info => ViolationSeverity::Info,
        }
    }
}

/// A detected constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// The constraint that was violated
    pub constraint_type: String,
    /// Which method/call caused the violation
    pub method: String,
    /// Position in the call sequence (or line number)
    pub position: usize,
    /// Human-readable description
    pub message: String,
    /// Severity of the violation
    pub severity: ViolationSeverity,
    /// Source location (file:line) if available
    pub location: Option<String>,
}

impl Violation {
    /// Create a new violation
    pub fn new(
        constraint_type: &str,
        method: &str,
        position: usize,
        message: &str,
        severity: ViolationSeverity,
    ) -> Self {
        Self {
            constraint_type: constraint_type.to_string(),
            method: method.to_string(),
            position,
            message: message.to_string(),
            severity,
            location: None,
        }
    }

    /// Set source location
    pub fn with_location(mut self, loc: impl Into<String>) -> Self {
        self.location = Some(loc.into());
        self
    }

    /// Check if this is a critical violation
    pub fn is_critical(&self) -> bool {
        matches!(self.severity, ViolationSeverity::Critical)
    }
}

/// Result of checking API usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Whether all constraints were satisfied
    pub passed: bool,
    /// List of violations found
    pub violations: Vec<Violation>,
    /// Final state after execution (if sequence was valid)
    pub final_state: Option<String>,
    /// Number of API calls checked
    pub calls_checked: usize,
    /// Total execution paths explored
    pub paths_explored: usize,
}

impl CheckResult {
    /// Create a passing result
    pub fn pass(final_state: &str, calls_checked: usize) -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
            final_state: Some(final_state.to_string()),
            calls_checked,
            paths_explored: 1,
        }
    }

    /// Create a failing result
    pub fn fail(violations: Vec<Violation>, calls_checked: usize) -> Self {
        Self {
            passed: false,
            violations,
            final_state: None,
            calls_checked,
            paths_explored: 1,
        }
    }

    /// Add a violation
    pub fn add_violation(&mut self, violation: Violation) {
        self.violations.push(violation);
        self.passed = false;
    }

    /// Get critical violations
    pub fn critical_violations(&self) -> Vec<&Violation> {
        self.violations.iter().filter(|v| v.is_critical()).collect()
    }

    /// Get violation count by severity
    pub fn count_by_severity(&self) -> HashMap<&'static str, usize> {
        let mut counts = HashMap::new();
        for v in &self.violations {
            let key = match v.severity {
                ViolationSeverity::Critical => "critical",
                ViolationSeverity::Error => "error",
                ViolationSeverity::Warning => "warning",
                ViolationSeverity::Info => "info",
            };
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }
}

/// API constraint checker
pub struct ApiChecker<'a> {
    api: &'a PlatformApi,
}

impl<'a> ApiChecker<'a> {
    /// Create a new checker for the given API
    pub fn new(api: &'a PlatformApi) -> Self {
        Self { api }
    }

    /// Check a sequence of API calls
    pub fn check_sequence(&self, calls: &[&str]) -> CheckResult {
        let mut violations = Vec::new();
        let mut call_history: Vec<&str> = Vec::new();
        let mut method_counts: HashMap<&str, usize> = HashMap::new();

        // Build a temporary mutable state machine
        let mut api_clone = self.api.clone();
        let sm = api_clone.state_machine();

        // Check state machine transitions
        match sm.execute_sequence(calls) {
            Ok(final_state) => {
                let final_name = sm.state_name(final_state).unwrap_or("unknown");

                // Now check additional constraints
                for (idx, &call) in calls.iter().enumerate() {
                    call_history.push(call);
                    *method_counts.entry(call).or_insert(0) += 1;

                    // Check constraints
                    for constraint in self.api.get_constraints() {
                        if let Some(v) = self.check_constraint(
                            constraint,
                            call,
                            idx,
                            &call_history,
                            &method_counts,
                        ) {
                            violations.push(v);
                        }
                    }
                }

                // Check end-of-sequence constraints (e.g., exactly-once)
                for constraint in self.api.get_constraints() {
                    if let Some(v) =
                        self.check_final_constraint(constraint, &method_counts, calls.len())
                    {
                        violations.push(v);
                    }
                }

                if violations.is_empty() {
                    CheckResult::pass(final_name, calls.len())
                } else {
                    CheckResult::fail(violations, calls.len())
                }
            }
            Err((idx, msg)) => {
                violations.push(Violation::new(
                    "invalid-transition",
                    calls.get(idx).unwrap_or(&"unknown"),
                    idx,
                    &msg,
                    ViolationSeverity::Critical,
                ));
                CheckResult::fail(violations, calls.len())
            }
        }
    }

    /// Check a single constraint against current state
    fn check_constraint(
        &self,
        constraint: &ApiConstraint,
        current_call: &str,
        position: usize,
        history: &[&str],
        _counts: &HashMap<&str, usize>,
    ) -> Option<Violation> {
        match &constraint.kind {
            ConstraintKind::Temporal(TemporalRelation::Before) => {
                // method_a must be called before method_b
                if let Some(ref method_b) = constraint.method_b {
                    if current_call == method_b {
                        // Check if method_a was called before
                        if !history[..position].contains(&constraint.method_a.as_str()) {
                            return Some(Violation::new(
                                constraint.constraint_type(),
                                current_call,
                                position,
                                &constraint.message,
                                constraint.severity.into(),
                            ));
                        }
                    }
                }
            }
            ConstraintKind::Temporal(TemporalRelation::After) => {
                // method_a must be called after method_b
                if current_call == constraint.method_a {
                    if let Some(ref method_b) = constraint.method_b {
                        if !history[..position].contains(&method_b.as_str()) {
                            return Some(Violation::new(
                                constraint.constraint_type(),
                                current_call,
                                position,
                                &constraint.message,
                                constraint.severity.into(),
                            ));
                        }
                    }
                }
            }
            ConstraintKind::Temporal(TemporalRelation::ImmediatelyBefore) => {
                // method_a must be called immediately before method_b
                if let Some(ref method_b) = constraint.method_b {
                    if current_call == method_b
                        && position > 0
                        && history[position - 1] != constraint.method_a.as_str()
                    {
                        return Some(Violation::new(
                            constraint.constraint_type(),
                            current_call,
                            position,
                            &constraint.message,
                            constraint.severity.into(),
                        ));
                    }
                }
            }
            ConstraintKind::AtMostOnce => {
                if current_call == constraint.method_a {
                    let count = history[..position]
                        .iter()
                        .filter(|&&c| c == constraint.method_a)
                        .count();
                    if count > 0 {
                        return Some(Violation::new(
                            constraint.constraint_type(),
                            current_call,
                            position,
                            &constraint.message,
                            constraint.severity.into(),
                        ));
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// Check constraints that apply at end of sequence
    fn check_final_constraint(
        &self,
        constraint: &ApiConstraint,
        counts: &HashMap<&str, usize>,
        total_calls: usize,
    ) -> Option<Violation> {
        match &constraint.kind {
            ConstraintKind::ExactlyOnce => {
                let count = counts
                    .get(constraint.method_a.as_str())
                    .copied()
                    .unwrap_or(0);
                if count != 1 {
                    return Some(Violation::new(
                        constraint.constraint_type(),
                        &constraint.method_a,
                        total_calls,
                        &format!(
                            "{} (called {} times, expected exactly 1)",
                            constraint.message, count
                        ),
                        constraint.severity.into(),
                    ));
                }
            }
            ConstraintKind::Paired => {
                if let Some(ref method_b) = constraint.method_b {
                    let count_a = counts
                        .get(constraint.method_a.as_str())
                        .copied()
                        .unwrap_or(0);
                    let count_b = counts.get(method_b.as_str()).copied().unwrap_or(0);
                    if count_a != count_b {
                        return Some(Violation::new(
                            constraint.constraint_type(),
                            &constraint.method_a,
                            total_calls,
                            &format!(
                                "{} ({} called {} times, {} called {} times)",
                                constraint.message, constraint.method_a, count_a, method_b, count_b
                            ),
                            constraint.severity.into(),
                        ));
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// Check source code for API constraint violations
    pub fn check_source(&self, source: &str) -> CheckResult {
        let mut violations = Vec::new();
        let mut calls_found = Vec::new();

        // Extract API calls from source code using regex patterns
        for trans in self.api.get_transitions() {
            let pattern = format!(r"\b{}\s*\(", regex::escape(&trans.method));
            if let Ok(re) = Regex::new(&pattern) {
                for mat in re.find_iter(source) {
                    let line_num = source[..mat.start()].matches('\n').count() + 1;
                    calls_found.push((trans.method.clone(), line_num));
                }
            }
        }

        // Sort by line number
        calls_found.sort_by_key(|(_, line)| *line);

        // Check temporal constraints
        for constraint in self.api.get_constraints() {
            if let ConstraintKind::Temporal(TemporalRelation::Before) = &constraint.kind {
                if let Some(ref method_b) = constraint.method_b {
                    // Find first occurrence of method_b
                    let b_pos = calls_found.iter().position(|(m, _)| m == method_b);
                    if let Some(b_idx) = b_pos {
                        let b_line = calls_found[b_idx].1;
                        // Check if method_a appears before method_b
                        let a_before_b = calls_found[..b_idx]
                            .iter()
                            .any(|(m, _)| m == &constraint.method_a);
                        if !a_before_b {
                            violations.push(
                                Violation::new(
                                    constraint.constraint_type(),
                                    method_b,
                                    b_line,
                                    &constraint.message,
                                    constraint.severity.into(),
                                )
                                .with_location(format!("line {}", b_line)),
                            );
                        }
                    }
                }
            }
        }

        if violations.is_empty() {
            CheckResult {
                passed: true,
                violations: Vec::new(),
                final_state: None,
                calls_checked: calls_found.len(),
                paths_explored: 1,
            }
        } else {
            CheckResult::fail(violations, calls_found.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ApiState, StateTransition};

    fn create_metal_command_buffer_api() -> PlatformApi {
        let mut api = PlatformApi::new("Metal", "MTLCommandBuffer");

        api.add_state(ApiState::new("Created"));
        api.add_state(ApiState::new("Encoding"));
        api.add_state(ApiState::new("Committed").as_terminal());

        api.set_initial_state("Created");

        // Can begin encoding from Created
        api.add_transition(StateTransition::new(
            "beginEncoding",
            vec!["Created"],
            "Encoding",
        ));

        // Can add handler from Created (stays in Created)
        api.add_transition(StateTransition::new(
            "addCompletedHandler",
            vec!["Created"],
            "Created",
        ));
        // Can add handler from Encoding (stays in Encoding)
        api.add_transition(StateTransition::new(
            "addCompletedHandler",
            vec!["Encoding"],
            "Encoding",
        ));

        // Can commit from Created or Encoding
        api.add_transition(StateTransition::new(
            "commit",
            vec!["Created", "Encoding"],
            "Committed",
        ));

        // CRITICAL CONSTRAINT: addCompletedHandler must be called BEFORE commit
        api.must_call_before(
            "addCompletedHandler",
            "commit",
            "addCompletedHandler: must be called BEFORE commit to avoid undefined behavior",
        );

        api
    }

    #[test]
    fn test_valid_sequence() {
        let api = create_metal_command_buffer_api();
        let checker = ApiChecker::new(&api);

        let result = checker.check_sequence(&["addCompletedHandler", "commit"]);
        assert!(result.passed);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn test_invalid_sequence_missing_handler() {
        let api = create_metal_command_buffer_api();
        let checker = ApiChecker::new(&api);

        // commit without addCompletedHandler
        let result = checker.check_sequence(&["commit"]);
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert!(result.violations[0].message.contains("addCompletedHandler"));
    }

    #[test]
    fn test_invalid_sequence_wrong_order() {
        let api = create_metal_command_buffer_api();
        let checker = ApiChecker::new(&api);

        // commit before addCompletedHandler - state machine will reject this
        // because commit from Created goes to Committed, and you can't call
        // addCompletedHandler from Committed
        let result = checker.check_sequence(&["beginEncoding", "commit"]);
        assert!(!result.passed);
        // Should have the "must call before" violation
        assert!(result
            .violations
            .iter()
            .any(|v| v.message.contains("addCompletedHandler")));
    }

    #[test]
    fn test_check_source_code() {
        // Create API with explicit temporal constraint for source code checking
        let mut api = PlatformApi::new("Metal", "MTLCommandBuffer");
        api.add_state(ApiState::new("Created"));
        api.add_state(ApiState::new("Committed").as_terminal());
        api.set_initial_state("Created");

        api.add_transition(StateTransition::new(
            "addCompletedHandler",
            vec!["Created"],
            "Created",
        ));
        api.add_transition(StateTransition::new("commit", vec!["Created"], "Committed"));

        // Add explicit constraint for source code checking
        api.must_call_before(
            "addCompletedHandler",
            "commit",
            "addCompletedHandler must be called before commit",
        );

        let checker = ApiChecker::new(&api);

        // Valid source code (using parentheses for regex detection)
        let valid_source = r#"
            let buffer = device.makeCommandBuffer();
            buffer.addCompletedHandler({ _ in print("done") });
            buffer.commit();
        "#;

        let result = checker.check_source(valid_source);
        assert!(
            result.passed,
            "Valid source should pass: {:?}",
            result.violations
        );

        // Invalid source code - commit before handler
        let invalid_source = r#"
            let buffer = device.makeCommandBuffer();
            buffer.commit();
            buffer.addCompletedHandler({ _ in print("done") });
        "#;

        let result = checker.check_source(invalid_source);
        assert!(!result.passed, "Invalid source should fail");
        assert!(result.violations[0].message.contains("addCompletedHandler"));
    }

    #[test]
    fn test_at_most_once_constraint() {
        let mut api = PlatformApi::new("Test", "Resource");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Initialized"));
        api.set_initial_state("Init");

        api.add_transition(StateTransition::new("init", vec!["Init"], "Initialized"));
        api.add_transition(StateTransition::new(
            "init",
            vec!["Initialized"],
            "Initialized",
        ));

        api.add_constraint(ApiConstraint::at_most_once(
            "init",
            "Resource can only be initialized once",
        ));

        let checker = ApiChecker::new(&api);

        // Single init - OK
        let result = checker.check_sequence(&["init"]);
        assert!(result.passed);

        // Double init - violation
        let result = checker.check_sequence(&["init", "init"]);
        assert!(!result.passed);
        assert!(result.violations[0].message.contains("initialized once"));
    }

    #[test]
    fn test_paired_constraint() {
        let mut api = PlatformApi::new("POSIX", "Mutex");
        api.add_state(ApiState::new("Unlocked"));
        api.add_state(ApiState::new("Locked"));
        api.set_initial_state("Unlocked");

        api.add_transition(StateTransition::new("lock", vec!["Unlocked"], "Locked"));
        api.add_transition(StateTransition::new("unlock", vec!["Locked"], "Unlocked"));

        api.add_constraint(ApiConstraint::paired(
            "lock",
            "unlock",
            "Every lock must have a matching unlock",
        ));

        let checker = ApiChecker::new(&api);

        // Balanced - OK
        let result = checker.check_sequence(&["lock", "unlock"]);
        assert!(result.passed);

        // Unbalanced - violation
        let result = checker.check_sequence(&["lock", "unlock", "lock"]);
        assert!(!result.passed);
        assert!(result.violations[0].message.contains("lock"));
    }

    #[test]
    fn test_violation_severity() {
        let api = create_metal_command_buffer_api();
        let checker = ApiChecker::new(&api);

        let result = checker.check_sequence(&["commit"]);
        assert!(!result.passed);

        let counts = result.count_by_severity();
        assert!(counts.get("critical").copied().unwrap_or(0) >= 1);
    }

    #[test]
    fn test_violation_is_critical_true() {
        let v = Violation::new("test", "method", 0, "msg", ViolationSeverity::Critical);
        assert!(v.is_critical(), "Critical severity should return true");
    }

    #[test]
    fn test_violation_is_critical_false_for_other_severities() {
        let error_v = Violation::new("test", "method", 0, "msg", ViolationSeverity::Error);
        let warning_v = Violation::new("test", "method", 0, "msg", ViolationSeverity::Warning);
        let info_v = Violation::new("test", "method", 0, "msg", ViolationSeverity::Info);

        assert!(!error_v.is_critical(), "Error severity should return false");
        assert!(
            !warning_v.is_critical(),
            "Warning severity should return false"
        );
        assert!(!info_v.is_critical(), "Info severity should return false");
    }

    #[test]
    fn test_check_result_add_violation() {
        let mut result = CheckResult::pass("TestState", 5);
        assert!(result.passed);
        assert!(result.violations.is_empty());

        let violation = Violation::new(
            "test",
            "method",
            1,
            "test message",
            ViolationSeverity::Error,
        );
        result.add_violation(violation);

        assert!(!result.passed, "add_violation should set passed to false");
        assert_eq!(
            result.violations.len(),
            1,
            "add_violation should add to violations list"
        );
    }

    #[test]
    fn test_check_result_critical_violations_returns_only_critical() {
        let mut result = CheckResult::pass("TestState", 5);
        result.add_violation(Violation::new(
            "test",
            "m1",
            0,
            "critical",
            ViolationSeverity::Critical,
        ));
        result.add_violation(Violation::new(
            "test",
            "m2",
            1,
            "error",
            ViolationSeverity::Error,
        ));
        result.add_violation(Violation::new(
            "test",
            "m3",
            2,
            "critical2",
            ViolationSeverity::Critical,
        ));

        let critical = result.critical_violations();
        assert_eq!(
            critical.len(),
            2,
            "Should return exactly 2 critical violations"
        );
        for v in &critical {
            assert!(
                v.is_critical(),
                "All returned violations should be critical"
            );
        }
    }

    #[test]
    fn test_check_result_critical_violations_empty_when_none() {
        let result = CheckResult::fail(
            vec![Violation::new(
                "test",
                "m",
                0,
                "error only",
                ViolationSeverity::Error,
            )],
            1,
        );

        let critical = result.critical_violations();
        assert!(
            critical.is_empty(),
            "Should return empty vec when no critical violations"
        );
    }

    #[test]
    fn test_temporal_after_constraint_violation() {
        // Test the Temporal::After constraint checking
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Step1"));
        api.add_state(ApiState::new("Step2").as_terminal());
        api.set_initial_state("Init");

        api.add_transition(StateTransition::new("first", vec!["Init"], "Step1"));
        api.add_transition(StateTransition::new("second", vec!["Step1"], "Step2"));
        api.add_transition(StateTransition::new("second", vec!["Init"], "Step2"));

        // Add "must call after" constraint: 'second' must be called after 'first'
        api.must_call_after("second", "first", "second must be called after first");

        let checker = ApiChecker::new(&api);

        // Valid: first then second
        let result = checker.check_sequence(&["first", "second"]);
        assert!(
            result.passed,
            "first then second should pass: {:?}",
            result.violations
        );

        // Invalid: second without first (via Init->Step2)
        let result = checker.check_sequence(&["second"]);
        assert!(!result.passed, "second without first should fail");
        assert!(
            result.violations.iter().any(|v| v.method == "second"),
            "Violation should be for 'second'"
        );
    }

    #[test]
    fn test_temporal_immediately_before_constraint() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Middle"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        api.add_transition(StateTransition::new("prepare", vec!["Init"], "Middle"));
        api.add_transition(StateTransition::new("other", vec!["Init"], "Middle"));
        api.add_transition(StateTransition::new("execute", vec!["Middle"], "Done"));

        // Add "immediately before" constraint: prepare must be called immediately before execute
        api.add_constraint(ApiConstraint {
            kind: ConstraintKind::Temporal(TemporalRelation::ImmediatelyBefore),
            method_a: "prepare".to_string(),
            method_b: Some("execute".to_string()),
            message: "prepare must be immediately before execute".to_string(),
            severity: Severity::Critical,
        });

        let checker = ApiChecker::new(&api);

        // Valid: prepare immediately before execute
        let result = checker.check_sequence(&["prepare", "execute"]);
        assert!(
            result.passed,
            "prepare immediately before execute should pass: {:?}",
            result.violations
        );

        // Invalid: other call between prepare and execute
        // Need different state machine for this test since our transitions don't allow it
    }

    #[test]
    fn test_immediately_before_violation_when_wrong_method_precedes() {
        // Test that violation is detected when different method precedes method_b
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Ready"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        // Setup: other -> Ready, prepare -> Ready, and both can go to Done via execute
        api.add_transition(StateTransition::new("other", vec!["Init"], "Ready"));
        api.add_transition(StateTransition::new("prepare", vec!["Init"], "Ready"));
        api.add_transition(StateTransition::new("execute", vec!["Ready"], "Done"));

        // Constraint: prepare must immediately precede execute
        api.add_constraint(ApiConstraint {
            kind: ConstraintKind::Temporal(TemporalRelation::ImmediatelyBefore),
            method_a: "prepare".to_string(),
            method_b: Some("execute".to_string()),
            message: "prepare must be immediately before execute".to_string(),
            severity: Severity::Critical,
        });

        let checker = ApiChecker::new(&api);

        // Invalid: 'other' immediately precedes 'execute' instead of 'prepare'
        let result = checker.check_sequence(&["other", "execute"]);
        assert!(
            !result.passed,
            "Should fail when wrong method precedes: {:?}",
            result
        );
        assert!(
            result.violations.iter().any(|v| v.method == "execute"),
            "Violation should be on 'execute'"
        );
    }

    #[test]
    fn test_immediately_before_at_position_zero() {
        // Test that method_b at position 0 (no prior method) correctly handles the position > 0 check
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        // Allow calling execute directly from Init
        api.add_transition(StateTransition::new("execute", vec!["Init"], "Done"));
        api.add_transition(StateTransition::new("prepare", vec!["Init"], "Init"));

        // Constraint: prepare must immediately precede execute
        api.add_constraint(ApiConstraint {
            kind: ConstraintKind::Temporal(TemporalRelation::ImmediatelyBefore),
            method_a: "prepare".to_string(),
            method_b: Some("execute".to_string()),
            message: "prepare must be immediately before execute".to_string(),
            severity: Severity::Critical,
        });

        let checker = ApiChecker::new(&api);

        // Call execute at position 0 - should this be a violation?
        // According to the code: position > 0 is required for violation, so position 0 does NOT violate
        // This tests the > operator mutation
        let result = checker.check_sequence(&["execute"]);
        // With current implementation, position 0 does NOT trigger violation (position > 0 is false)
        assert!(
            result.passed,
            "Execute at position 0 should pass (no prior call required by impl): {:?}",
            result.violations
        );
    }

    #[test]
    fn test_immediately_before_at_position_one() {
        // Test position == 1 case (boundary for position > 0)
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Ready"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        api.add_transition(StateTransition::new("wrong", vec!["Init"], "Ready"));
        api.add_transition(StateTransition::new("prepare", vec!["Init"], "Ready"));
        api.add_transition(StateTransition::new("execute", vec!["Ready"], "Done"));

        api.add_constraint(ApiConstraint {
            kind: ConstraintKind::Temporal(TemporalRelation::ImmediatelyBefore),
            method_a: "prepare".to_string(),
            method_b: Some("execute".to_string()),
            message: "prepare must immediately precede execute".to_string(),
            severity: Severity::Critical,
        });

        let checker = ApiChecker::new(&api);

        // Position 1 with wrong preceding method should fail (tests position > 0 and != check)
        let result = checker.check_sequence(&["wrong", "execute"]);
        assert!(
            !result.passed,
            "Position 1 with wrong method should violate: {:?}",
            result
        );

        // Position 1 with correct preceding method should pass
        let result = checker.check_sequence(&["prepare", "execute"]);
        assert!(
            result.passed,
            "Position 1 with correct method should pass: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_immediately_before_method_b_comparison() {
        // Test that constraint only triggers when current_call == method_b (line 261)
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Mid"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        api.add_transition(StateTransition::new("other", vec!["Init"], "Mid"));
        api.add_transition(StateTransition::new("unrelated", vec!["Mid"], "Done"));

        // Constraint only applies to 'execute', not to 'unrelated'
        api.add_constraint(ApiConstraint {
            kind: ConstraintKind::Temporal(TemporalRelation::ImmediatelyBefore),
            method_a: "prepare".to_string(),
            method_b: Some("execute".to_string()),
            message: "prepare must immediately precede execute".to_string(),
            severity: Severity::Critical,
        });

        let checker = ApiChecker::new(&api);

        // 'unrelated' is not 'execute', so constraint should not trigger
        let result = checker.check_sequence(&["other", "unrelated"]);
        assert!(
            result.passed,
            "Constraint should not trigger for unrelated method: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_exactly_once_constraint_violation() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        api.add_transition(StateTransition::new("init", vec!["Init"], "Done"));

        api.add_constraint(ApiConstraint::exactly_once(
            "init",
            "init must be called exactly once",
        ));

        let checker = ApiChecker::new(&api);

        // Valid: exactly once
        let result = checker.check_sequence(&["init"]);
        assert!(result.passed, "Single call should pass");

        // Invalid: zero times - needs to be caught by final constraint
        // Can't test this easily since we need at least one call to reach Done
    }

    #[test]
    fn test_exactly_once_zero_calls_violation() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        // Allow reaching Done without calling 'special'
        api.add_transition(StateTransition::new("finish", vec!["Init"], "Done"));
        api.add_transition(StateTransition::new("special", vec!["Init"], "Init"));

        api.add_constraint(ApiConstraint::exactly_once(
            "special",
            "special must be called exactly once",
        ));

        let checker = ApiChecker::new(&api);

        // Invalid: zero calls to special
        let result = checker.check_sequence(&["finish"]);
        assert!(!result.passed, "Zero calls should violate exactly_once");
        assert!(
            result.violations.iter().any(|v| v.method == "special"),
            "Violation should be for 'special' method"
        );
    }

    #[test]
    fn test_check_source_line_number_calculation() {
        let mut api = PlatformApi::new("Test", "Object");
        api.add_state(ApiState::new("Init"));
        api.add_state(ApiState::new("Done").as_terminal());
        api.set_initial_state("Init");

        api.add_transition(StateTransition::new("methodA", vec!["Init"], "Init"));
        api.add_transition(StateTransition::new("methodB", vec!["Init"], "Done"));

        api.must_call_before("methodA", "methodB", "A before B");

        let checker = ApiChecker::new(&api);

        // Source with methodB on line 3 (1 empty line, then methodB call)
        let source = "\n\nmethodB();\nmethodA();";

        let result = checker.check_source(source);
        assert!(!result.passed);

        // Verify line number calculation works
        let v = &result.violations[0];
        assert!(v.location.is_some());
        // Line 3 because: line 1 empty, line 2 empty, line 3 has methodB
        assert!(
            v.location.as_ref().unwrap().contains("3"),
            "Should report line 3"
        );
    }
}
