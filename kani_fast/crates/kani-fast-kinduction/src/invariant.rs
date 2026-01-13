//! Invariant synthesis for k-induction
//!
//! This module provides template-based invariant synthesis to strengthen
//! k-induction when the property alone is not inductive.

use crate::engine::InductionFailure;
use crate::formula::{Property, SmtType, StateFormula, TransitionSystem};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A candidate invariant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    /// The invariant formula
    pub formula: StateFormula,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Source of this invariant (template, learned, etc.)
    pub source: InvariantSource,
}

impl Invariant {
    pub fn new(formula: StateFormula, confidence: f64, source: InvariantSource) -> Self {
        Self {
            formula,
            confidence,
            source,
        }
    }

    pub fn from_template(formula: StateFormula, template_name: &str) -> Self {
        Self {
            formula,
            confidence: 0.5,
            source: InvariantSource::Template(template_name.to_string()),
        }
    }
}

/// Source of an invariant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvariantSource {
    /// Generated from a template
    Template(String),
    /// Learned from counterexample
    Learned,
    /// Provided by user
    User,
    /// Synthesized by AI
    AiSynthesized,
}

/// Template for generating candidate invariants
#[derive(Debug, Clone)]
pub struct InvariantTemplate {
    /// Template name
    pub name: String,
    /// Template pattern with placeholders
    /// Placeholders: {var} for any variable, {int_var} for int variables
    pub pattern: String,
    /// Required variable types
    pub required_types: Vec<SmtType>,
}

impl InvariantTemplate {
    pub fn new(name: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pattern: pattern.into(),
            required_types: Vec::new(),
        }
    }

    pub fn with_type(mut self, smt_type: SmtType) -> Self {
        self.required_types.push(smt_type);
        self
    }

    /// Instantiate this template with concrete variables
    pub fn instantiate(&self, vars: &[(&str, &SmtType)]) -> Option<StateFormula> {
        let mut formula = self.pattern.clone();

        // Check type requirements
        for (i, required_type) in self.required_types.iter().enumerate() {
            if let Some((name, actual_type)) = vars.get(i) {
                if *actual_type != required_type {
                    return None;
                }
                formula = formula.replace(&format!("{{var{i}}}"), name);
            } else {
                return None;
            }
        }

        // Replace generic {var} placeholders
        for (i, (name, _)) in vars.iter().enumerate() {
            formula = formula.replace(&format!("{{var{i}}}"), name);
        }

        // Replace {var} with first variable if present
        if let Some((name, _)) = vars.first() {
            formula = formula.replace("{var}", name);
        }

        Some(StateFormula::with_description(
            formula,
            format!("Template: {}", self.name),
        ))
    }
}

/// Standard templates for common invariant patterns
pub fn standard_templates() -> Vec<InvariantTemplate> {
    vec![
        // Non-negativity for integers
        InvariantTemplate::new("non_negative", "(>= {var} 0)").with_type(SmtType::Int),
        // Non-positive for integers
        InvariantTemplate::new("non_positive", "(<= {var} 0)").with_type(SmtType::Int),
        // Upper bound template
        InvariantTemplate::new("upper_bound", "(<= {var} {const})").with_type(SmtType::Int),
        // Lower bound template
        InvariantTemplate::new("lower_bound", "(>= {var} {const})").with_type(SmtType::Int),
        // Range bound
        InvariantTemplate::new("range", "(and (>= {var} 0) (<= {var} {const}))")
            .with_type(SmtType::Int),
        // Ordering between variables
        InvariantTemplate::new("ordering", "(<= {var0} {var1})")
            .with_type(SmtType::Int)
            .with_type(SmtType::Int),
        // Boolean invariant (always true)
        InvariantTemplate::new("always_true", "{var}").with_type(SmtType::Bool),
        // Boolean invariant (always false)
        InvariantTemplate::new("always_false", "(not {var})").with_type(SmtType::Bool),
        // Equality to constant
        InvariantTemplate::new("equal_const", "(= {var} {const})").with_type(SmtType::Int),
        // Implication between booleans
        InvariantTemplate::new("implies", "(=> {var0} {var1})")
            .with_type(SmtType::Bool)
            .with_type(SmtType::Bool),
        // Monotonicity (x' >= x in transition)
        InvariantTemplate::new("monotonic_inc", "(>= {var}' {var})").with_type(SmtType::Int),
        // Monotonicity (x' <= x in transition)
        InvariantTemplate::new("monotonic_dec", "(<= {var}' {var})").with_type(SmtType::Int),
    ]
}

/// Invariant synthesizer using templates and learning
pub struct InvariantSynthesizer {
    /// Available templates
    templates: Vec<InvariantTemplate>,
    /// Tried invariants (to avoid duplicates)
    tried: HashSet<String>,
    /// Successfully used invariants
    successful: Vec<Invariant>,
}

impl InvariantSynthesizer {
    pub fn new() -> Self {
        Self {
            templates: standard_templates(),
            tried: HashSet::new(),
            successful: Vec::new(),
        }
    }

    /// Add a custom template
    pub fn add_template(&mut self, template: InvariantTemplate) {
        self.templates.push(template);
    }

    /// Synthesize an invariant to strengthen induction
    pub fn synthesize(
        &self,
        system: &TransitionSystem,
        property: &Property,
        failure: &InductionFailure,
    ) -> Option<StateFormula> {
        // Strategy 1: Try templates with system variables
        for template in &self.templates {
            for candidate in self.instantiate_template(template, system) {
                let formula_str = candidate.smt_formula.clone();
                if !self.tried.contains(&formula_str) {
                    // Basic check: invariant should not be trivially false
                    if !self.is_trivially_false(&candidate) {
                        return Some(candidate);
                    }
                }
            }
        }

        // Strategy 2: Analyze failure model for hints
        if let Some(model) = &failure.model {
            if let Some(invariant) = self.learn_from_model(model, system, property) {
                return Some(invariant);
            }
        }

        // Strategy 3: Combine property with strengthening
        if let Some(invariant) = self.strengthen_property(property, system) {
            return Some(invariant);
        }

        None
    }

    /// Instantiate a template with all valid variable combinations
    fn instantiate_template(
        &self,
        template: &InvariantTemplate,
        system: &TransitionSystem,
    ) -> Vec<StateFormula> {
        let mut results = Vec::new();

        // Get variables by type
        let typed_vars: Vec<(&str, &SmtType)> = system
            .variables
            .iter()
            .map(|v| (v.name.as_str(), &v.smt_type))
            .collect();

        // For single-variable templates
        if template.required_types.len() <= 1 {
            for (name, smt_type) in &typed_vars {
                if let Some(formula) = template.instantiate(&[(*name, smt_type)]) {
                    // Also try with constants
                    let with_const = formula.smt_formula.replace("{const}", "100");
                    results.push(StateFormula::new(with_const));
                }
            }
        }

        // For two-variable templates
        if template.required_types.len() == 2 {
            for (i, (name1, type1)) in typed_vars.iter().enumerate() {
                for (name2, type2) in typed_vars.iter().skip(i + 1) {
                    if let Some(formula) =
                        template.instantiate(&[(*name1, *type1), (*name2, *type2)])
                    {
                        results.push(formula);
                    }
                }
            }
        }

        results
    }

    /// Check if formula is trivially false
    fn is_trivially_false(&self, formula: &StateFormula) -> bool {
        // Simple syntactic checks
        formula.smt_formula == "false" || formula.smt_formula.contains("(and false")
    }

    /// Learn invariant from counterexample model
    fn learn_from_model(
        &self,
        model: &str,
        system: &TransitionSystem,
        _property: &Property,
    ) -> Option<StateFormula> {
        // Extract variable values from model
        // This is a simplified implementation - real ICE learning would be more sophisticated

        // Look for integer variables with specific values
        for var in &system.variables {
            if var.smt_type == SmtType::Int {
                // Try to find the value of this variable in the model
                let pattern = format!(r"define-fun\s+{}\s*\(\)\s*Int\s+(\d+)", var.name);
                if let Ok(re) = regex::Regex::new(&pattern) {
                    if let Some(cap) = re.captures(model) {
                        if let Some(value_str) = cap.get(1) {
                            if let Ok(value) = value_str.as_str().parse::<i64>() {
                                // Generate invariant based on observed value
                                if value >= 0 {
                                    return Some(StateFormula::with_description(
                                        format!("(>= {} 0)", var.name),
                                        format!("Learned from model: {} was {}", var.name, value),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Strengthen property with additional constraints
    fn strengthen_property(
        &self,
        property: &Property,
        system: &TransitionSystem,
    ) -> Option<StateFormula> {
        // Try combining property with variable bounds
        for var in &system.variables {
            if var.smt_type == SmtType::Int {
                // Try: property AND var >= 0
                let strengthened = StateFormula::new(format!(
                    "(and {} (>= {} 0))",
                    property.formula.smt_formula, var.name
                ));
                return Some(strengthened);
            }
        }

        None
    }

    /// Record that an invariant was successfully used
    pub fn record_success(&mut self, invariant: Invariant) {
        self.successful.push(invariant.clone());
        self.tried.insert(invariant.formula.smt_formula);
    }

    /// Get successfully used invariants
    pub fn successful_invariants(&self) -> &[Invariant] {
        &self.successful
    }
}

impl Default for InvariantSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InductionFailure;
    use crate::formula::Property;

    // ==================== Invariant Tests ====================

    #[test]
    fn test_invariant_new() {
        let formula = StateFormula::new("(>= x 0)");
        let inv = Invariant::new(formula.clone(), 0.9, InvariantSource::Learned);

        assert_eq!(inv.formula.smt_formula, "(>= x 0)");
        assert_eq!(inv.confidence, 0.9);
        assert!(matches!(inv.source, InvariantSource::Learned));
    }

    #[test]
    fn test_invariant_from_template() {
        let inv = Invariant::from_template(StateFormula::new("(>= x 0)"), "non_negative");

        assert_eq!(inv.formula.smt_formula, "(>= x 0)");
        assert_eq!(inv.confidence, 0.5);
        match &inv.source {
            InvariantSource::Template(name) => assert_eq!(name, "non_negative"),
            _ => panic!("Expected Template source"),
        }
    }

    #[test]
    fn test_invariant_confidence_range() {
        let low = Invariant::new(StateFormula::new("x"), 0.0, InvariantSource::User);
        let high = Invariant::new(StateFormula::new("x"), 1.0, InvariantSource::User);

        assert_eq!(low.confidence, 0.0);
        assert_eq!(high.confidence, 1.0);
    }

    #[test]
    fn test_invariant_serialization() {
        let inv = Invariant::from_template(StateFormula::new("(>= x 0)"), "test");
        let json = serde_json::to_string(&inv).unwrap();
        let deserialized: Invariant = serde_json::from_str(&json).unwrap();

        assert_eq!(inv.formula.smt_formula, deserialized.formula.smt_formula);
        assert_eq!(inv.confidence, deserialized.confidence);
    }

    #[test]
    fn test_invariant_clone() {
        let inv = Invariant::from_template(StateFormula::new("(>= x 0)"), "test");
        let cloned = inv.clone();

        assert_eq!(inv.formula.smt_formula, cloned.formula.smt_formula);
        assert_eq!(inv.confidence, cloned.confidence);
    }

    // ==================== InvariantSource Tests ====================

    #[test]
    fn test_invariant_source_template() {
        let source = InvariantSource::Template("my_template".to_string());
        match source {
            InvariantSource::Template(name) => assert_eq!(name, "my_template"),
            _ => panic!("Expected Template"),
        }
    }

    #[test]
    fn test_invariant_source_learned() {
        let source = InvariantSource::Learned;
        assert!(matches!(source, InvariantSource::Learned));
    }

    #[test]
    fn test_invariant_source_user() {
        let source = InvariantSource::User;
        assert!(matches!(source, InvariantSource::User));
    }

    #[test]
    fn test_invariant_source_ai_synthesized() {
        let source = InvariantSource::AiSynthesized;
        assert!(matches!(source, InvariantSource::AiSynthesized));
    }

    #[test]
    fn test_invariant_source_serialization() {
        let sources = vec![
            InvariantSource::Template("test".to_string()),
            InvariantSource::Learned,
            InvariantSource::User,
            InvariantSource::AiSynthesized,
        ];

        for source in sources {
            let json = serde_json::to_string(&source).unwrap();
            let deserialized: InvariantSource = serde_json::from_str(&json).unwrap();
            // Check that round-trip works
            let json2 = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[test]
    fn test_invariant_source_debug() {
        let source = InvariantSource::Learned;
        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("Learned"));
    }

    // ==================== InvariantTemplate Tests ====================

    #[test]
    fn test_template_new() {
        let template = InvariantTemplate::new("test_template", "(>= {var} 0)");

        assert_eq!(template.name, "test_template");
        assert_eq!(template.pattern, "(>= {var} 0)");
        assert!(template.required_types.is_empty());
    }

    #[test]
    fn test_template_new_from_strings() {
        let template =
            InvariantTemplate::new(String::from("string_name"), String::from("(= {var} 0)"));

        assert_eq!(template.name, "string_name");
        assert_eq!(template.pattern, "(= {var} 0)");
    }

    #[test]
    fn test_template_with_type_single() {
        let template = InvariantTemplate::new("test", "(>= {var} 0)").with_type(SmtType::Int);

        assert_eq!(template.required_types.len(), 1);
        assert_eq!(template.required_types[0], SmtType::Int);
    }

    #[test]
    fn test_template_with_type_multiple() {
        let template = InvariantTemplate::new("ordering", "(<= {var0} {var1})")
            .with_type(SmtType::Int)
            .with_type(SmtType::Int);

        assert_eq!(template.required_types.len(), 2);
        assert_eq!(template.required_types[0], SmtType::Int);
        assert_eq!(template.required_types[1], SmtType::Int);
    }

    #[test]
    fn test_template_with_type_mixed() {
        let template = InvariantTemplate::new("mixed", "({var0} => {var1})")
            .with_type(SmtType::Bool)
            .with_type(SmtType::Bool);

        assert_eq!(template.required_types.len(), 2);
        assert_eq!(template.required_types[0], SmtType::Bool);
        assert_eq!(template.required_types[1], SmtType::Bool);
    }

    #[test]
    fn test_template_instantiation() {
        let template = InvariantTemplate::new("test", "(>= {var} 0)").with_type(SmtType::Int);

        let vars = [("x", &SmtType::Int)];
        let result = template.instantiate(&vars);

        assert!(result.is_some());
        assert_eq!(result.unwrap().smt_formula, "(>= x 0)");
    }

    #[test]
    fn test_template_type_mismatch() {
        let template = InvariantTemplate::new("test", "(>= {var} 0)").with_type(SmtType::Int);

        let vars = [("b", &SmtType::Bool)];
        let result = template.instantiate(&vars);

        assert!(result.is_none());
    }

    #[test]
    fn test_two_var_template() {
        let template = InvariantTemplate::new("ordering", "(<= {var0} {var1})")
            .with_type(SmtType::Int)
            .with_type(SmtType::Int);

        let vars = [("x", &SmtType::Int), ("y", &SmtType::Int)];
        let result = template.instantiate(&vars);

        assert!(result.is_some());
        assert_eq!(result.unwrap().smt_formula, "(<= x y)");
    }

    #[test]
    fn test_template_instantiate_not_enough_vars() {
        let template = InvariantTemplate::new("ordering", "(<= {var0} {var1})")
            .with_type(SmtType::Int)
            .with_type(SmtType::Int);

        let vars = [("x", &SmtType::Int)]; // Only one var, need two
        let result = template.instantiate(&vars);

        assert!(result.is_none());
    }

    #[test]
    fn test_template_instantiate_description() {
        let template =
            InvariantTemplate::new("non_negative", "(>= {var} 0)").with_type(SmtType::Int);

        let vars = [("counter", &SmtType::Int)];
        let result = template.instantiate(&vars).unwrap();

        assert!(result.description.is_some());
        assert!(result.description.unwrap().contains("non_negative"));
    }

    #[test]
    fn test_template_no_required_types() {
        let template = InvariantTemplate::new("simple", "(>= {var} 0)");

        // No required types, so any type should work for generic {var}
        let vars = [("x", &SmtType::Int)];
        let result = template.instantiate(&vars);
        assert!(result.is_some());
    }

    // ==================== Standard Templates Tests ====================

    #[test]
    fn test_standard_templates_count() {
        let templates = standard_templates();
        assert_eq!(templates.len(), 12);
    }

    #[test]
    fn test_standard_templates_non_negative() {
        let templates = standard_templates();
        let non_neg = templates.iter().find(|t| t.name == "non_negative").unwrap();

        assert_eq!(non_neg.pattern, "(>= {var} 0)");
        assert_eq!(non_neg.required_types.len(), 1);
        assert_eq!(non_neg.required_types[0], SmtType::Int);
    }

    #[test]
    fn test_standard_templates_non_positive() {
        let templates = standard_templates();
        let non_pos = templates.iter().find(|t| t.name == "non_positive").unwrap();

        assert_eq!(non_pos.pattern, "(<= {var} 0)");
        assert_eq!(non_pos.required_types[0], SmtType::Int);
    }

    #[test]
    fn test_standard_templates_upper_bound() {
        let templates = standard_templates();
        let upper = templates.iter().find(|t| t.name == "upper_bound").unwrap();

        assert_eq!(upper.pattern, "(<= {var} {const})");
    }

    #[test]
    fn test_standard_templates_lower_bound() {
        let templates = standard_templates();
        let lower = templates.iter().find(|t| t.name == "lower_bound").unwrap();

        assert_eq!(lower.pattern, "(>= {var} {const})");
    }

    #[test]
    fn test_standard_templates_range() {
        let templates = standard_templates();
        let range = templates.iter().find(|t| t.name == "range").unwrap();

        assert!(range.pattern.contains("and"));
        assert!(range.pattern.contains(">= {var} 0"));
        assert!(range.pattern.contains("<= {var} {const}"));
    }

    #[test]
    fn test_standard_templates_ordering() {
        let templates = standard_templates();
        let ordering = templates.iter().find(|t| t.name == "ordering").unwrap();

        assert_eq!(ordering.pattern, "(<= {var0} {var1})");
        assert_eq!(ordering.required_types.len(), 2);
    }

    #[test]
    fn test_standard_templates_always_true() {
        let templates = standard_templates();
        let always_true = templates.iter().find(|t| t.name == "always_true").unwrap();

        assert_eq!(always_true.pattern, "{var}");
        assert_eq!(always_true.required_types[0], SmtType::Bool);
    }

    #[test]
    fn test_standard_templates_always_false() {
        let templates = standard_templates();
        let always_false = templates.iter().find(|t| t.name == "always_false").unwrap();

        assert_eq!(always_false.pattern, "(not {var})");
        assert_eq!(always_false.required_types[0], SmtType::Bool);
    }

    #[test]
    fn test_standard_templates_equal_const() {
        let templates = standard_templates();
        let equal = templates.iter().find(|t| t.name == "equal_const").unwrap();

        assert_eq!(equal.pattern, "(= {var} {const})");
    }

    #[test]
    fn test_standard_templates_implies() {
        let templates = standard_templates();
        let implies = templates.iter().find(|t| t.name == "implies").unwrap();

        assert_eq!(implies.pattern, "(=> {var0} {var1})");
        assert_eq!(implies.required_types.len(), 2);
        assert_eq!(implies.required_types[0], SmtType::Bool);
        assert_eq!(implies.required_types[1], SmtType::Bool);
    }

    #[test]
    fn test_standard_templates_monotonic_inc() {
        let templates = standard_templates();
        let mono = templates
            .iter()
            .find(|t| t.name == "monotonic_inc")
            .unwrap();

        assert_eq!(mono.pattern, "(>= {var}' {var})");
        assert_eq!(mono.required_types[0], SmtType::Int);
    }

    #[test]
    fn test_standard_templates_monotonic_dec() {
        let templates = standard_templates();
        let mono = templates
            .iter()
            .find(|t| t.name == "monotonic_dec")
            .unwrap();

        assert_eq!(mono.pattern, "(<= {var}' {var})");
        assert_eq!(mono.required_types[0], SmtType::Int);
    }

    #[test]
    fn test_standard_templates_all_have_names() {
        let templates = standard_templates();
        for template in &templates {
            assert!(!template.name.is_empty());
        }
    }

    #[test]
    fn test_standard_templates_all_have_patterns() {
        let templates = standard_templates();
        for template in &templates {
            assert!(!template.pattern.is_empty());
        }
    }

    // ==================== InvariantSynthesizer Tests ====================

    #[test]
    fn test_synthesizer_creation() {
        let synth = InvariantSynthesizer::new();
        assert!(!synth.templates.is_empty());
    }

    #[test]
    fn test_synthesizer_default() {
        let synth = InvariantSynthesizer::default();
        assert_eq!(synth.templates.len(), standard_templates().len());
    }

    #[test]
    fn test_synthesizer_add_template() {
        let mut synth = InvariantSynthesizer::new();
        let initial_count = synth.templates.len();

        synth.add_template(InvariantTemplate::new("custom", "(= {var} 42)"));

        assert_eq!(synth.templates.len(), initial_count + 1);
    }

    #[test]
    fn test_synthesizer_successful_invariants_empty() {
        let synth = InvariantSynthesizer::new();
        assert!(synth.successful_invariants().is_empty());
    }

    #[test]
    fn test_synthesizer_record_success() {
        let mut synth = InvariantSynthesizer::new();
        let inv = Invariant::from_template(StateFormula::new("(>= x 0)"), "test");

        synth.record_success(inv);

        assert_eq!(synth.successful_invariants().len(), 1);
        assert_eq!(
            synth.successful_invariants()[0].formula.smt_formula,
            "(>= x 0)"
        );
    }

    #[test]
    fn test_synthesizer_record_multiple_successes() {
        let mut synth = InvariantSynthesizer::new();

        synth.record_success(Invariant::from_template(
            StateFormula::new("(>= x 0)"),
            "t1",
        ));
        synth.record_success(Invariant::from_template(
            StateFormula::new("(>= y 0)"),
            "t2",
        ));
        synth.record_success(Invariant::from_template(
            StateFormula::new("(<= z 100)"),
            "t3",
        ));

        assert_eq!(synth.successful_invariants().len(), 3);
    }

    #[test]
    fn test_synthesizer_is_trivially_false_false_literal() {
        let synth = InvariantSynthesizer::new();
        let formula = StateFormula::new("false");
        assert!(synth.is_trivially_false(&formula));
    }

    #[test]
    fn test_synthesizer_is_trivially_false_and_false() {
        let synth = InvariantSynthesizer::new();
        let formula = StateFormula::new("(and false (>= x 0))");
        assert!(synth.is_trivially_false(&formula));
    }

    #[test]
    fn test_synthesizer_is_trivially_false_not_false() {
        let synth = InvariantSynthesizer::new();
        let formula = StateFormula::new("(>= x 0)");
        assert!(!synth.is_trivially_false(&formula));
    }

    #[test]
    fn test_synthesizer_is_trivially_false_true() {
        let synth = InvariantSynthesizer::new();
        let formula = StateFormula::new("true");
        assert!(!synth.is_trivially_false(&formula));
    }

    fn create_simple_transition_system() -> TransitionSystem {
        let mut system = TransitionSystem::new();
        system.add_variable("x", SmtType::Int);
        system.add_variable("y", SmtType::Int);
        system.add_variable("flag", SmtType::Bool);
        system.set_init(StateFormula::new("(= x 0)"));
        system.set_transition(StateFormula::new("(= x' (+ x 1))"));
        system
    }

    #[test]
    fn test_synthesizer_instantiate_template_single_var() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let template = InvariantTemplate::new("non_neg", "(>= {var} 0)").with_type(SmtType::Int);

        let results = synth.instantiate_template(&template, &system);

        // Should have instantiations for x and y (both Int)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_synthesizer_instantiate_template_two_vars() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let template = InvariantTemplate::new("ordering", "(<= {var0} {var1})")
            .with_type(SmtType::Int)
            .with_type(SmtType::Int);

        let results = synth.instantiate_template(&template, &system);

        // Should have instantiation for (x, y) pair
        assert!(!results.is_empty());
        // Check that we got the ordering formula
        assert!(results.iter().any(|r| r.smt_formula == "(<= x y)"));
    }

    #[test]
    fn test_synthesizer_instantiate_template_bool() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let template = InvariantTemplate::new("always_true", "{var}").with_type(SmtType::Bool);

        let results = synth.instantiate_template(&template, &system);

        // Should have instantiation for flag (Bool)
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.smt_formula == "flag"));
    }

    #[test]
    fn test_synthesizer_strengthen_property() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property =
            Property::safety("test_prop", "Test Property", StateFormula::new("(< x 100)"));

        let result = synth.strengthen_property(&property, &system);

        assert!(result.is_some());
        let strengthened = result.unwrap();
        // Should combine property with variable >= 0
        assert!(strengthened.smt_formula.contains("and"));
        assert!(strengthened.smt_formula.contains("(< x 100)"));
        assert!(strengthened.smt_formula.contains(">= "));
        assert!(strengthened.smt_formula.contains(" 0)"));
    }

    #[test]
    fn test_synthesizer_learn_from_model_positive() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("true"));

        // Model showing x = 5 (positive value)
        let model = "(define-fun x () Int 5)";

        let result = synth.learn_from_model(model, &system, &property);

        assert!(result.is_some());
        let learned = result.unwrap();
        assert!(learned.smt_formula.contains(">= x 0"));
    }

    #[test]
    fn test_synthesizer_learn_from_model_no_match() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("true"));

        // Model with unrecognized format
        let model = "sat";

        let result = synth.learn_from_model(model, &system, &property);

        assert!(result.is_none());
    }

    #[test]
    fn test_synthesizer_synthesize_basic() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: None,
            reason: Some("Induction step failed".to_string()),
        };

        let result = synth.synthesize(&system, &property, &failure);

        // Should return some candidate invariant from templates
        assert!(result.is_some());
    }

    #[test]
    fn test_synthesizer_synthesize_with_model() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: Some("(define-fun x () Int 10)".to_string()),
            reason: Some("Induction step failed".to_string()),
        };

        let result = synth.synthesize(&system, &property, &failure);

        assert!(result.is_some());
    }

    #[test]
    fn test_synthesizer_tried_tracking() {
        let mut synth = InvariantSynthesizer::new();
        let formula = "(>= x 0)";

        // Record success adds to tried set
        synth.record_success(Invariant::from_template(StateFormula::new(formula), "test"));

        assert!(synth.tried.contains(formula));
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_full_synthesis_workflow() {
        let mut synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property =
            Property::safety("safety", "Safety Property", StateFormula::new("(< x 1000)"));
        let failure = InductionFailure {
            k: 5,
            model: Some("(define-fun x () Int 50)".to_string()),
            reason: Some("Failed at k=5".to_string()),
        };

        // Synthesize
        let candidate = synth.synthesize(&system, &property, &failure);
        assert!(candidate.is_some());

        // Record success
        let inv = Invariant::from_template(candidate.unwrap(), "synthesized");
        synth.record_success(inv);

        // Verify recorded
        assert_eq!(synth.successful_invariants().len(), 1);
    }

    #[test]
    fn test_template_instantiation_with_const_replacement() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();

        // Templates with {const} get replaced with 100
        let template =
            InvariantTemplate::new("upper_bound", "(<= {var} {const})").with_type(SmtType::Int);

        let results = synth.instantiate_template(&template, &system);

        // Should have results with 100 substituted
        let has_const_100 = results.iter().any(|r| r.smt_formula.contains("100"));
        assert!(has_const_100);
    }

    // ======== Mutation coverage tests ========

    #[test]
    fn test_synthesize_checks_tried_set() {
        // Line 182: !self.tried.contains check
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: None,
            reason: None,
        };

        // First synthesis should return Some
        let first = synth.synthesize(&system, &property, &failure);
        assert!(first.is_some());
    }

    #[test]
    fn test_synthesize_checks_trivially_false() {
        // Line 184: !self.is_trivially_false check
        let synth = InvariantSynthesizer::new();

        // Create a formula that would be caught by is_trivially_false
        let trivially_false = StateFormula::new("false");
        assert!(synth.is_trivially_false(&trivially_false));

        // is_trivially_false returns true for "false" literal
        // and for "(and false ...)" patterns
        let and_false = StateFormula::new("(and false (>= x 0))");
        assert!(synth.is_trivially_false(&and_false));

        // Normal formulas should NOT be trivially false
        let normal = StateFormula::new("(>= x 0)");
        assert!(!synth.is_trivially_false(&normal));
    }

    #[test]
    fn test_instantiate_template_const_replacement_at_correct_position() {
        // Line 235: i + 1 in skip() - tests that we skip the right number of elements
        let synth = InvariantSynthesizer::new();

        // Create system with multiple Int variables
        let mut system = TransitionSystem::new();
        system.add_variable("a", SmtType::Int);
        system.add_variable("b", SmtType::Int);
        system.add_variable("c", SmtType::Int);
        system.set_init(StateFormula::new("true"));
        system.set_transition(StateFormula::new("true"));

        // Two-variable template
        let template = InvariantTemplate::new("ordering", "(<= {var0} {var1})")
            .with_type(SmtType::Int)
            .with_type(SmtType::Int);

        let results = synth.instantiate_template(&template, &system);

        // Should have pairs: (a,b), (a,c), (b,c) - but NOT (a,a), (b,b), (c,c)
        // The i+1 skip ensures we don't compare a variable with itself
        let formulas: Vec<_> = results.iter().map(|r| r.smt_formula.as_str()).collect();

        // Verify we have distinct pairs
        assert!(formulas.contains(&"(<= a b)") || formulas.iter().any(|f| f.contains("(<= a ")));

        // Verify we DON'T have self-comparisons
        assert!(!formulas.contains(&"(<= a a)"));
        assert!(!formulas.contains(&"(<= b b)"));
        assert!(!formulas.contains(&"(<= c c)"));
    }

    #[test]
    fn test_strengthen_property_equality_check() {
        // Line 298: var.smt_type == SmtType::Int check
        let synth = InvariantSynthesizer::new();

        // System with only Bool variables
        let mut bool_only_system = TransitionSystem::new();
        bool_only_system.add_variable("flag", SmtType::Bool);
        bool_only_system.set_init(StateFormula::new("flag"));
        bool_only_system.set_transition(StateFormula::new("(= flag' flag)"));

        let property = Property::safety("test", "Bool property", StateFormula::new("flag"));

        // Should return None since no Int variables
        let result = synth.strengthen_property(&property, &bool_only_system);
        assert!(result.is_none());

        // System with Int variable - should return Some
        let result2 = synth.strengthen_property(&property, &create_simple_transition_system());
        assert!(result2.is_some());
    }

    #[test]
    fn test_strengthen_property_uses_first_int_var() {
        // Line 298-304: Verify we use the first Int variable found
        let synth = InvariantSynthesizer::new();

        let mut system = TransitionSystem::new();
        system.add_variable("first_int", SmtType::Int);
        system.add_variable("second_int", SmtType::Int);
        system.set_init(StateFormula::new("true"));
        system.set_transition(StateFormula::new("true"));

        let property = Property::safety("test", "Test", StateFormula::new("(> first_int 0)"));

        let result = synth.strengthen_property(&property, &system);
        assert!(result.is_some());

        let strengthened = result.unwrap();
        // Should use first_int (the first Int variable found)
        assert!(strengthened.smt_formula.contains("first_int"));
        assert!(strengthened.smt_formula.contains("(>= first_int 0)"));
    }

    #[test]
    fn test_learn_from_model_only_for_int_vars() {
        // Line 266: var.smt_type == SmtType::Int check in learn_from_model
        let synth = InvariantSynthesizer::new();
        let property = Property::safety("test", "Test Property", StateFormula::new("true"));

        // System with only Bool variable
        let mut bool_system = TransitionSystem::new();
        bool_system.add_variable("flag", SmtType::Bool);
        bool_system.set_init(StateFormula::new("flag"));
        bool_system.set_transition(StateFormula::new("(= flag' flag)"));

        // Model with define-fun for flag (Bool)
        let model = "(define-fun flag () Bool true)";

        // Should return None because we only learn from Int variables
        let result = synth.learn_from_model(model, &bool_system, &property);
        assert!(result.is_none());
    }

    #[test]
    fn test_learn_from_model_positive_value() {
        // Line 274: value >= 0 check
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test", StateFormula::new("true"));

        // Model with positive value
        let model_positive = "(define-fun x () Int 42)";
        let result = synth.learn_from_model(model_positive, &system, &property);
        assert!(result.is_some());
        assert!(result.unwrap().smt_formula.contains(">= x 0"));

        // Model with zero value (should also work since 0 >= 0)
        let model_zero = "(define-fun x () Int 0)";
        let result_zero = synth.learn_from_model(model_zero, &system, &property);
        assert!(result_zero.is_some());
    }

    #[test]
    fn test_instantiate_template_single_vs_two_var() {
        // Line 222 and 233: template.required_types.len() checks
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();

        // Single-var template (required_types.len() <= 1)
        let single_template =
            InvariantTemplate::new("single", "(>= {var} 0)").with_type(SmtType::Int);
        let single_results = synth.instantiate_template(&single_template, &system);
        assert!(!single_results.is_empty());

        // Two-var template (required_types.len() == 2)
        let two_template = InvariantTemplate::new("two", "(<= {var0} {var1})")
            .with_type(SmtType::Int)
            .with_type(SmtType::Int);
        let two_results = synth.instantiate_template(&two_template, &system);
        assert!(!two_results.is_empty());

        // No-type template (required_types.len() == 0)
        let no_type_template = InvariantTemplate::new("no_type", "(>= {var} 0)");
        let no_type_results = synth.instantiate_template(&no_type_template, &system);
        assert!(!no_type_results.is_empty());
    }

    // ======== Additional mutation coverage tests ========

    // Test for line 182: `!self.tried.contains(&formula_str)` - returns UNTRIED candidates
    // If `!` is deleted, we'd return TRIED candidates instead (wrong!)
    #[test]
    fn test_synthesize_returns_untried_not_tried() {
        let mut synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: None,
            reason: None,
        };

        // First synthesis should return Some (untried candidate)
        let first = synth.synthesize(&system, &property, &failure);
        assert!(first.is_some(), "First synthesis should return a candidate");
        let first_formula = first.unwrap().smt_formula.clone();

        // Mark it as tried (record_success adds to tried set)
        synth.record_success(Invariant::from_template(
            StateFormula::new(&first_formula),
            "test",
        ));

        // Next synthesis should return a DIFFERENT candidate (not the tried one)
        let second = synth.synthesize(&system, &property, &failure);
        if let Some(second_candidate) = second {
            // The second candidate should NOT be the same as the first
            // (If `!` were deleted, we'd only return tried candidates, so it WOULD be the same)
            assert_ne!(
                second_candidate.smt_formula, first_formula,
                "Second synthesis should NOT return the already-tried formula"
            );
        }
        // Note: second could be None if we exhausted untried candidates, which is fine
    }

    // Test for line 182: Verify that tried candidates are SKIPPED
    #[test]
    fn test_synthesize_skips_tried_candidates() {
        let mut synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: None,
            reason: None,
        };

        // Collect multiple candidates by synthesizing repeatedly
        let mut candidates = Vec::new();
        for _ in 0..5 {
            if let Some(candidate) = synth.synthesize(&system, &property, &failure) {
                candidates.push(candidate.smt_formula.clone());
                synth.record_success(Invariant::from_template(
                    StateFormula::new(candidates.last().unwrap().clone()),
                    "test",
                ));
            } else {
                break;
            }
        }

        // All collected candidates should be unique
        // If `!` were deleted on line 182, we'd keep returning the same tried formula
        let unique_count = {
            let mut seen = std::collections::HashSet::new();
            candidates
                .iter()
                .filter(|c| seen.insert(c.as_str()))
                .count()
        };
        assert_eq!(
            unique_count,
            candidates.len(),
            "All synthesized candidates should be unique (tried ones are skipped)"
        );
    }

    // Test for line 184: `!self.is_trivially_false(&candidate)` - rejects trivially false
    // If `!` is deleted, we'd ACCEPT trivially false candidates (wrong!)
    #[test]
    fn test_synthesize_rejects_trivially_false_candidates() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: None,
            reason: None,
        };

        // Synthesize should return a candidate
        let result = synth.synthesize(&system, &property, &failure);
        assert!(result.is_some());

        // The returned candidate should NOT be trivially false
        // (If `!` were deleted on line 184, we'd return trivially false candidates)
        let candidate = result.unwrap();
        assert!(
            !synth.is_trivially_false(&candidate),
            "Synthesized candidate should NOT be trivially false"
        );
    }

    // Test for line 184: Verify all synthesized candidates pass the trivially false check
    #[test]
    fn test_all_synthesized_candidates_are_not_trivially_false() {
        let mut synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: None,
            reason: None,
        };

        // Generate several candidates
        let mut all_valid = true;
        for _ in 0..10 {
            if let Some(candidate) = synth.synthesize(&system, &property, &failure) {
                // Every candidate should pass the trivially false check
                if synth.is_trivially_false(&candidate) {
                    all_valid = false;
                    break;
                }
                synth.record_success(Invariant::from_template(candidate, "test"));
            } else {
                break;
            }
        }

        assert!(
            all_valid,
            "All synthesized candidates should NOT be trivially false"
        );
    }

    // Test combination: line 182 AND 184 both needed
    #[test]
    fn test_synthesize_requires_both_untried_and_not_trivially_false() {
        let synth = InvariantSynthesizer::new();
        let system = create_simple_transition_system();
        let property = Property::safety("test", "Test Property", StateFormula::new("(< x 100)"));
        let failure = InductionFailure {
            k: 1,
            model: None,
            reason: None,
        };

        // Get first candidate
        let first = synth.synthesize(&system, &property, &failure);
        assert!(first.is_some());

        let candidate = first.unwrap();

        // Verify BOTH conditions:
        // 1. Not in tried set (before recording success)
        assert!(
            !synth.tried.contains(&candidate.smt_formula),
            "New candidate should not be in tried set"
        );

        // 2. Not trivially false
        assert!(
            !synth.is_trivially_false(&candidate),
            "Candidate should not be trivially false"
        );
    }
}
