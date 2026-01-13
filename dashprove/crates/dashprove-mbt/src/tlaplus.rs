//! TLA+ specification parser for model-based testing
//!
//! This module parses TLA+ specifications into state machine models that can
//! be used for test generation. It extracts:
//! - Variables and their types/domains
//! - Initial state predicates
//! - Actions (Next state relations)
//! - Invariants

use std::collections::HashMap;

use indexmap::IndexMap;

use crate::error::MbtResult;
use crate::model::{
    ActionSpec, Invariant, ModelState, ModelValue, StateMachineModel, VariableDomain,
};

/// Parser for TLA+ specifications
pub struct TlaPlusParser {
    /// Source text
    source: String,
    /// Current position in source (reserved for future use)
    #[allow(dead_code)]
    pos: usize,
    /// Parsed variables
    variables: IndexMap<String, VariableDomain>,
    /// Parsed actions
    actions: Vec<ActionSpec>,
    /// Parsed invariants
    invariants: Vec<Invariant>,
    /// Initial state expressions
    init_exprs: Vec<String>,
    /// Constants defined in the spec (reserved for future use)
    #[allow(dead_code)]
    constants: HashMap<String, ModelValue>,
}

impl TlaPlusParser {
    /// Create a new parser for the given TLA+ source
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            pos: 0,
            variables: IndexMap::new(),
            actions: Vec::new(),
            invariants: Vec::new(),
            init_exprs: Vec::new(),
            constants: HashMap::new(),
        }
    }

    /// Parse the specification into a state machine model
    pub fn parse(&mut self) -> MbtResult<StateMachineModel> {
        let source = self.source.clone();
        let lines: Vec<&str> = source.lines().collect();

        let mut module_name = "Unknown".to_string();
        let mut in_variables = false;
        let mut current_def_name: Option<String> = None;
        let mut current_def_body = String::new();

        for line in lines {
            let trimmed = line.trim();

            // Skip comments
            if trimmed.starts_with("\\*") || trimmed.starts_with("(*") {
                continue;
            }

            // Module declaration
            if let Some(name) = trimmed.strip_prefix("---- MODULE ") {
                module_name = name
                    .trim_end_matches('-')
                    .trim()
                    .trim_end_matches('-')
                    .trim()
                    .to_string();
                continue;
            }

            // Alternative module declaration
            if let Some(rest) = trimmed.strip_prefix("MODULE ") {
                module_name = rest.trim().to_string();
                continue;
            }

            // VARIABLES section
            if trimmed.starts_with("VARIABLES") || trimmed.starts_with("VARIABLE") {
                in_variables = true;
                let rest = trimmed
                    .strip_prefix("VARIABLES")
                    .or_else(|| trimmed.strip_prefix("VARIABLE"))
                    .unwrap_or("");
                self.parse_variable_list(rest);
                continue;
            }

            // CONSTANTS section (just reset in_variables flag)
            if trimmed.starts_with("CONSTANTS") || trimmed.starts_with("CONSTANT") {
                in_variables = false;
                continue;
            }

            // Continue variables across lines
            if in_variables && !trimmed.is_empty() && !trimmed.contains("==") {
                self.parse_variable_list(trimmed);
                continue;
            }

            // Definition (Name == ...)
            if let Some(idx) = trimmed.find("==") {
                in_variables = false;

                // Save previous definition
                if let Some(name) = current_def_name.take() {
                    self.process_definition(&name, &current_def_body);
                }

                let name = trimmed[..idx].trim().to_string();
                let body = trimmed[idx + 2..].trim().to_string();
                current_def_name = Some(name);
                current_def_body = body;
                continue;
            }

            // Continue definition body
            if current_def_name.is_some() && !trimmed.is_empty() {
                current_def_body.push(' ');
                current_def_body.push_str(trimmed);
            }
        }

        // Process last definition
        if let Some(name) = current_def_name.take() {
            self.process_definition(&name, &current_def_body);
        }

        // Build the model
        let mut model = StateMachineModel::new(module_name);
        model.variables = self.variables.clone();
        model.actions = self.actions.clone();
        model.invariants = self.invariants.clone();

        // Generate initial states from Init predicate
        if let Some(initial) = self.generate_initial_state() {
            model.initial_states.push(initial);
        }

        Ok(model)
    }

    /// Parse a comma-separated list of variable names
    fn parse_variable_list(&mut self, text: &str) {
        for part in text.split(',') {
            let var_name = part.trim();
            if !var_name.is_empty() {
                // Default to unknown domain until we see type constraints
                self.variables
                    .insert(var_name.to_string(), VariableDomain::Boolean);
            }
        }
    }

    /// Process a definition (Init, Next, actions, invariants)
    fn process_definition(&mut self, name: &str, body: &str) {
        let name_lower = name.to_lowercase();

        if name_lower == "init" {
            self.parse_init(body);
        } else if name_lower == "next" {
            self.parse_next(body);
        } else if name_lower.contains("inv") || name_lower.contains("invariant") {
            self.invariants.push(Invariant::new(name, body));
        } else if name_lower.contains("type") || name_lower.contains("typeinv") {
            self.parse_type_invariant(body);
        } else if body.contains("'") {
            // This is likely an action (contains primed variables)
            self.parse_action(name, body);
        }
    }

    /// Parse Init predicate to extract initial state constraints
    fn parse_init(&mut self, body: &str) {
        self.init_exprs.push(body.to_string());

        // Parse simple assignments: var = value
        // First collect updates, then apply them to avoid borrow issues
        let updates: Vec<(String, VariableDomain)> = body
            .split('/')
            .filter_map(|part| {
                let part = part.trim().trim_start_matches('\\');
                if let Some(idx) = part.find('=') {
                    let var = part[..idx].trim();
                    let val = part[idx + 1..].trim();
                    if self.variables.contains_key(var) {
                        self.parse_value_to_domain(val)
                            .map(|domain| (var.to_string(), domain))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        for (var, domain) in updates {
            self.variables.insert(var, domain);
        }
    }

    /// Parse Next predicate to extract action disjunctions
    fn parse_next(&mut self, body: &str) {
        // Next is typically: Action1 \/ Action2 \/ ...
        for part in body.split("\\/") {
            let action_name = part.trim();
            if !action_name.is_empty() && !action_name.starts_with('(') {
                // This is an action reference
                let spec = ActionSpec::new(action_name).with_effect("See action definition");
                self.actions.push(spec);
            }
        }
    }

    /// Parse an action definition
    fn parse_action(&mut self, name: &str, body: &str) {
        let mut enabled = String::new();
        let mut effect = String::new();

        // Split on /\ to separate conjuncts
        let parts: Vec<&str> = body.split("/\\").map(|s| s.trim()).collect();

        for part in parts {
            if part.contains("'") {
                // This is an effect (next state)
                effect.push_str(part);
                effect.push_str(" /\\ ");
            } else if !part.is_empty() {
                // This is an enabling condition
                enabled.push_str(part);
                enabled.push_str(" /\\ ");
            }
        }

        let spec = ActionSpec::new(name)
            .with_enabled(enabled.trim_end_matches(" /\\ "))
            .with_effect(effect.trim_end_matches(" /\\ "));

        // Check if this action already exists in the list
        if !self.actions.iter().any(|a| a.name == name) {
            self.actions.push(spec);
        } else {
            // Update existing action
            if let Some(existing) = self.actions.iter_mut().find(|a| a.name == name) {
                existing.enabled_description = enabled.trim_end_matches(" /\\ ").to_string();
                existing.effect_description = effect.trim_end_matches(" /\\ ").to_string();
            }
        }
    }

    /// Parse type invariant to infer variable domains
    fn parse_type_invariant(&mut self, body: &str) {
        // TypeInvariant == /\ var \in Domain /\ ...
        for part in body.split("/\\") {
            let part = part.trim();
            if let Some(idx) = part.find("\\in") {
                let var = part[..idx].trim();
                let domain_str = part[idx + 3..].trim();

                if let Some(domain) = self.parse_domain(domain_str) {
                    if self.variables.contains_key(var) {
                        self.variables.insert(var.to_string(), domain);
                    }
                }
            }
        }
    }

    /// Parse a domain expression
    fn parse_domain(&self, expr: &str) -> Option<VariableDomain> {
        let expr = expr.trim();

        // BOOLEAN
        if expr == "BOOLEAN" {
            return Some(VariableDomain::Boolean);
        }

        // Integer range: a..b
        if let Some(idx) = expr.find("..") {
            let min_str = expr[..idx].trim();
            let max_str = expr[idx + 2..].trim();
            if let (Ok(min), Ok(max)) = (min_str.parse::<i64>(), max_str.parse::<i64>()) {
                return Some(VariableDomain::IntRange { min, max });
            }
        }

        // Nat (natural numbers) - use reasonable default
        if expr == "Nat" {
            return Some(VariableDomain::IntRange { min: 0, max: 100 });
        }

        // Int - use reasonable default
        if expr == "Int" {
            return Some(VariableDomain::IntRange {
                min: -100,
                max: 100,
            });
        }

        // Set: {a, b, c}
        if expr.starts_with('{') && expr.ends_with('}') {
            let inner = &expr[1..expr.len() - 1];
            let values: Vec<ModelValue> = inner
                .split(',')
                .filter_map(|s| self.parse_simple_value(s.trim()))
                .collect();
            if !values.is_empty() {
                return Some(VariableDomain::Enumeration(values));
            }
        }

        // SUBSET of something
        if let Some(rest) = expr.strip_prefix("SUBSET") {
            if let Some(inner_domain) = self.parse_domain(rest.trim()) {
                return Some(VariableDomain::SetOf(Box::new(inner_domain)));
            }
        }

        // Seq(Domain)
        if let Some(rest) = expr.strip_prefix("Seq(") {
            if let Some(inner) = rest.strip_suffix(')') {
                if let Some(inner_domain) = self.parse_domain(inner) {
                    return Some(VariableDomain::SequenceOf {
                        element: Box::new(inner_domain),
                        max_length: 10,
                    });
                }
            }
        }

        None
    }

    /// Parse a simple value from TLA+ syntax
    fn parse_simple_value(&self, s: &str) -> Option<ModelValue> {
        let s = s.trim();

        // Boolean
        if s == "TRUE" {
            return Some(ModelValue::Bool(true));
        }
        if s == "FALSE" {
            return Some(ModelValue::Bool(false));
        }

        // Integer
        if let Ok(i) = s.parse::<i64>() {
            return Some(ModelValue::Int(i));
        }

        // String (quoted)
        if s.starts_with('"') && s.ends_with('"') {
            return Some(ModelValue::String(s[1..s.len() - 1].to_string()));
        }

        // Identifier (treat as string)
        if s.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Some(ModelValue::String(s.to_string()));
        }

        None
    }

    /// Parse a value expression to infer domain
    fn parse_value_to_domain(&self, val: &str) -> Option<VariableDomain> {
        let val = val.trim();

        // Boolean
        if val == "TRUE" || val == "FALSE" {
            return Some(VariableDomain::Boolean);
        }

        // Integer
        if val.parse::<i64>().is_ok() {
            return Some(VariableDomain::IntRange { min: 0, max: 100 });
        }

        // Set literal
        if val.starts_with('{') && val.ends_with('}') {
            return Some(VariableDomain::SetOf(Box::new(VariableDomain::Boolean)));
        }

        // Sequence literal
        if val.starts_with("<<") && val.ends_with(">>") {
            return Some(VariableDomain::SequenceOf {
                element: Box::new(VariableDomain::Boolean),
                max_length: 10,
            });
        }

        None
    }

    /// Generate initial state from parsed Init predicate
    fn generate_initial_state(&self) -> Option<ModelState> {
        let mut state = ModelState::new();

        // Set default values for all variables
        for (name, domain) in &self.variables {
            if let Some(min_val) = domain.min_value() {
                state.set(name, min_val);
            }
        }

        // Parse Init expressions to override defaults
        for init_expr in &self.init_exprs {
            for part in init_expr.split("/\\") {
                let part = part.trim();
                // Match: var = value
                if let Some(idx) = part.find('=') {
                    if !part.contains("\\in") {
                        let var = part[..idx].trim();
                        let val_str = part[idx + 1..].trim();
                        if state.contains(var) {
                            if let Some(val) = self.parse_simple_value(val_str) {
                                state.set(var, val);
                            }
                        }
                    }
                }
            }
        }

        if state.variables.is_empty() {
            None
        } else {
            Some(state)
        }
    }
}

/// Parse a TLA+ specification file into a state machine model
pub fn parse_tlaplus_spec(source: &str) -> MbtResult<StateMachineModel> {
    let mut parser = TlaPlusParser::new(source);
    parser.parse()
}

/// Parse a TLA+ file from path
pub fn parse_tlaplus_file(path: &std::path::Path) -> MbtResult<StateMachineModel> {
    let source = std::fs::read_to_string(path)?;
    parse_tlaplus_spec(&source)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COUNTER_SPEC: &str = r#"
---- MODULE Counter ----
EXTENDS Naturals

VARIABLE count

TypeInvariant == count \in 0..10

Init == count = 0

Increment == count' = count + 1

Decrement ==
    /\ count > 0
    /\ count' = count - 1

Next == Increment \/ Decrement

Spec == Init /\ [][Next]_count

====
"#;

    #[test]
    fn test_parse_counter_spec() {
        let model = parse_tlaplus_spec(COUNTER_SPEC).unwrap();

        assert_eq!(model.name, "Counter");
        assert!(model.variables.contains_key("count"));
        assert!(!model.actions.is_empty());
    }

    #[test]
    fn test_parse_domain() {
        let parser = TlaPlusParser::new("");

        // Integer range
        let domain = parser.parse_domain("0..10").unwrap();
        assert!(matches!(
            domain,
            VariableDomain::IntRange { min: 0, max: 10 }
        ));

        // Boolean
        let domain = parser.parse_domain("BOOLEAN").unwrap();
        assert!(matches!(domain, VariableDomain::Boolean));

        // Enumeration
        let domain = parser.parse_domain("{1, 2, 3}").unwrap();
        assert!(matches!(domain, VariableDomain::Enumeration(_)));
    }

    #[test]
    fn test_parse_value() {
        let parser = TlaPlusParser::new("");

        assert_eq!(
            parser.parse_simple_value("TRUE"),
            Some(ModelValue::Bool(true))
        );
        assert_eq!(parser.parse_simple_value("42"), Some(ModelValue::Int(42)));
        assert_eq!(
            parser.parse_simple_value("\"hello\""),
            Some(ModelValue::String("hello".into()))
        );
    }

    const TWO_PHASE_COMMIT: &str = r#"
---- MODULE TwoPhaseCommit ----
EXTENDS Naturals

VARIABLES rmState, tmState, tmPrepared

RMs == {"rm1", "rm2", "rm3"}

TypeOK ==
    /\ rmState \in [RMs -> {"working", "prepared", "committed", "aborted"}]
    /\ tmState \in {"init", "committed", "aborted"}
    /\ tmPrepared \in SUBSET RMs

Init ==
    /\ rmState = [rm \in RMs |-> "working"]
    /\ tmState = "init"
    /\ tmPrepared = {}

TMRcvPrepared(rm) ==
    /\ tmState = "init"
    /\ rmState[rm] = "prepared"
    /\ tmPrepared' = tmPrepared \union {rm}
    /\ UNCHANGED <<rmState, tmState>>

TMCommit ==
    /\ tmState = "init"
    /\ tmPrepared = RMs
    /\ tmState' = "committed"
    /\ UNCHANGED <<rmState, tmPrepared>>

TMAbort ==
    /\ tmState = "init"
    /\ tmState' = "aborted"
    /\ UNCHANGED <<rmState, tmPrepared>>

RMPrepare(rm) ==
    /\ rmState[rm] = "working"
    /\ rmState' = [rmState EXCEPT ![rm] = "prepared"]
    /\ UNCHANGED <<tmState, tmPrepared>>

Next ==
    \/ TMCommit
    \/ TMAbort
    \/ \E rm \in RMs : TMRcvPrepared(rm)
    \/ \E rm \in RMs : RMPrepare(rm)

====
"#;

    #[test]
    fn test_parse_two_phase_commit() {
        let model = parse_tlaplus_spec(TWO_PHASE_COMMIT).unwrap();

        assert_eq!(model.name, "TwoPhaseCommit");
        assert!(model.variables.contains_key("rmState"));
        assert!(model.variables.contains_key("tmState"));
        assert!(model.variables.contains_key("tmPrepared"));
    }
}
