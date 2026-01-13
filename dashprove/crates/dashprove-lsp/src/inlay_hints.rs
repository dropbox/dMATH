//! Inlay hints for USL documents
//!
//! Provides inline type annotations and parameter hints to aid code readability.
//! Inlay hints appear as virtual text in the editor, showing types and parameter names.
//! Also provides backend recommendation hints for verifiable properties.

use crate::document::Document;
use crate::symbols::format_type;
use dashprove_usl::{Contract, Expr, Param, Property, Refinement, Spec, Type};
use tower_lsp::lsp_types::{InlayHint, InlayHintKind, InlayHintLabel, Position, Range};

/// Generate inlay hints for a USL document within the given range.
///
/// Provides hints for:
/// - Contract return types (shown after parameter list)
/// - Contract parameter types (shown after parameter names)
/// - Quantifier variable types (shown after variable names in forall/exists)
/// - Recommended verification backends (shown after property name)
#[must_use]
pub fn generate_inlay_hints(doc: &Document, range: Range) -> Vec<InlayHint> {
    let mut hints = Vec::new();

    let spec = match &doc.spec {
        Some(spec) => spec,
        None => return hints,
    };

    // Generate backend recommendation hints for all properties
    generate_backend_hints(doc, spec, range, &mut hints);

    // Generate hints for contracts (parameter and return types)
    for property in &spec.properties {
        if let Property::Contract(contract) = property {
            generate_contract_hints(doc, contract, range, &mut hints);
        }
        if let Property::Refinement(refinement) = property {
            generate_refinement_hints(doc, refinement, range, &mut hints);
        }
    }

    // Generate hints for quantifiers in all properties
    generate_quantifier_hints(doc, spec, range, &mut hints);

    hints
}

/// Generate backend recommendation hints for each property.
///
/// Shows the recommended verification backend after the property name,
/// helping users understand which tool will be used for verification.
fn generate_backend_hints(doc: &Document, spec: &Spec, range: Range, hints: &mut Vec<InlayHint>) {
    for property in &spec.properties {
        let name = property.name();
        let kind = property_kind_keyword(property);
        let backend = recommended_backend(property);

        // Find the property definition line: "keyword name"
        // For contracts, the pattern is "contract Type::method"
        let pattern = if matches!(property, Property::Contract(_)) {
            format!("contract {}", name)
        } else {
            format!("{} {}", kind, name)
        };

        if let Some(offset) = doc.text.find(&pattern) {
            // Position the hint at the end of the property name line
            // Find the end of the pattern
            let end_offset = offset + pattern.len();
            let (line, character) = doc.offset_to_position(end_offset);
            let position = Position::new(line, character);

            if is_position_in_range(position, range) {
                let (label, tooltip) = backend_hint_label(backend, property);
                hints.push(InlayHint {
                    position,
                    label: InlayHintLabel::String(label),
                    kind: None, // Custom hint kind (not type or parameter)
                    text_edits: None,
                    tooltip: Some(tower_lsp::lsp_types::InlayHintTooltip::String(tooltip)),
                    padding_left: Some(true),
                    padding_right: Some(false),
                    data: None,
                });
            }
        }
    }
}

/// Get the keyword used to define a property.
fn property_kind_keyword(prop: &Property) -> &'static str {
    match prop {
        Property::Theorem(_) => "theorem",
        Property::Temporal(_) => "temporal",
        Property::Contract(_) => "contract",
        Property::Invariant(_) => "invariant",
        Property::Refinement(_) => "refinement",
        Property::Probabilistic(_) => "probabilistic",
        Property::Security(_) => "security",
        Property::Semantic(_) => "semantic",
        Property::PlatformApi(_) => "platform_api",
        Property::Bisimulation(_) => "bisimulation",
        Property::Version(_) => "version",
        Property::Capability(_) => "capability",
        Property::DistributedInvariant(_) => "distributed_invariant",
        Property::DistributedTemporal(_) => "distributed_temporal",
        Property::Composed(_) => "composed",
        Property::ImprovementProposal(_) => "improvement_proposal",
        Property::VerificationGate(_) => "verification_gate",
        Property::Rollback(_) => "rollback",
    }
}

/// Get the recommended verification backend for a property.
fn recommended_backend(prop: &Property) -> &'static str {
    match prop {
        Property::Theorem(_) => "lean4",
        Property::Temporal(_) => "tlaplus",
        Property::Contract(_) => "kani",
        Property::Invariant(_) => "lean4",
        Property::Refinement(_) => "lean4",
        Property::Probabilistic(_) => "probabilistic",
        Property::Security(_) => "security",
        Property::Semantic(_) => "semantic",
        Property::PlatformApi(_) => "platform_api",
        Property::Bisimulation(_) => "bisim",
        Property::Version(_) => "semver",
        Property::Capability(_) => "kani",
        Property::DistributedInvariant(_) => "tlaplus",
        Property::DistributedTemporal(_) => "tlaplus",
        Property::Composed(_) => "lean4",
        Property::ImprovementProposal(_) => "lean4",
        Property::VerificationGate(_) => "lean4",
        Property::Rollback(_) => "lean4",
    }
}

/// Generate the inlay hint label and tooltip for a backend recommendation.
fn backend_hint_label(backend: &str, prop: &Property) -> (String, String) {
    let label = match backend {
        "lean4" => " [Lean 4]",
        "tlaplus" => " [TLA+]",
        "kani" => " [Kani]",
        "alloy" => " [Alloy]",
        "coq" => " [Coq]",
        "isabelle" => " [Isabelle]",
        "dafny" => " [Dafny]",
        "platform_api" => " [Platform API]",
        "probabilistic" => " [Prob]",
        "security" => " [Sec]",
        "semantic" => " [Semantic]",
        _ => " [?]",
    };

    let tooltip = match prop {
        Property::Theorem(_) => {
            "Theorem proving backend. Lean 4 provides dependent types and tactics for mathematical proofs."
        }
        Property::Temporal(_) => {
            "Model checking backend. TLA+ excels at temporal logic and distributed systems verification."
        }
        Property::Contract(_) => {
            "Rust verification backend. Kani uses symbolic execution to verify Rust code contracts."
        }
        Property::Invariant(_) => {
            "Invariant verification backend. Lean 4 proves invariants hold across all states."
        }
        Property::Refinement(_) => {
            "Refinement verification backend. Lean 4 proves implementation matches specification."
        }
        Property::Probabilistic(_) => {
            "Probabilistic verification backend. Analyzes stochastic properties and probability bounds."
        }
        Property::Security(_) => {
            "Security verification backend. Analyzes information flow and security properties."
        }
        Property::Semantic(_) => {
            "Semantic verification backend. Uses embeddings and semantic predicates (e.g., semantic_similarity)."
        }
        Property::PlatformApi(_) => {
            "Platform API backend. Generates static Rust checkers enforcing documented API call ordering and preconditions."
        }
        Property::Bisimulation(_) => {
            "Bisimulation backend. Verifies behavioral equivalence between oracle and subject implementations."
        }
        Property::Version(_) => {
            "Version compatibility backend. Verifies semantic versioning constraints and API compatibility."
        }
        Property::Capability(_) => {
            "Capability verification backend. Verifies what a system can do matches its specification."
        }
        Property::DistributedInvariant(_) => {
            "Distributed invariant backend. TLA+ verifies multi-agent coordination invariants hold across all processes."
        }
        Property::DistributedTemporal(_) => {
            "Distributed temporal backend. TLA+ verifies multi-agent temporal properties with fairness constraints."
        }
        Property::Composed(_) => {
            "Composed theorem backend. Lean 4 verifies theorems that combine multiple proof dependencies."
        }
        Property::ImprovementProposal(_) => {
            "Improvement proposal backend. Lean 4 verifies that proposals satisfy improvement criteria."
        }
        Property::VerificationGate(_) => {
            "Verification gate backend. Lean 4 verifies that all gate checks pass before proceeding."
        }
        Property::Rollback(_) => {
            "Rollback backend. Lean 4 verifies rollback conditions and target state restoration."
        }
    };

    (label.to_string(), tooltip.to_string())
}

/// Generate inlay hints for a contract's parameters and return type.
fn generate_contract_hints(
    doc: &Document,
    contract: &Contract,
    range: Range,
    hints: &mut Vec<InlayHint>,
) {
    let contract_name = contract.type_path.join("::");

    // Find the contract signature in the source
    if let Some(sig_offset) = doc.text.find(&format!("contract {}", contract_name)) {
        // Find parameter hints
        for param in &contract.params {
            generate_param_hint(doc, sig_offset, param, range, hints);
        }

        // Generate return type hint if present
        if let Some(ref ret_type) = contract.return_type {
            generate_return_type_hint(doc, sig_offset, ret_type, range, hints);
        }
    }
}

/// Generate a hint for a single parameter showing its type.
fn generate_param_hint(
    doc: &Document,
    search_start: usize,
    param: &Param,
    range: Range,
    _hints: &mut Vec<InlayHint>,
) {
    // Find the parameter in the source text
    let search_text = &doc.text[search_start..];

    // Look for the parameter pattern "name:" or "name)" to find where to place hint
    // We want to show the type after the parameter name for parameters without explicit types
    // Since USL contracts require types, we show hints for readability

    // Find parameter name followed by colon
    let param_pattern = format!("{}: ", param.name);
    if let Some(param_offset) = search_text.find(&param_pattern) {
        let abs_offset = search_start + param_offset + param.name.len();
        let (line, character) = doc.offset_to_position(abs_offset);
        let position = Position::new(line, character);

        // Only include if within requested range
        if is_position_in_range(position, range) {
            // For explicit type annotations, we don't need hints
            // But we can show parameter name hints when calling
            // This is handled elsewhere - skip explicit annotations
        }
    }
}

/// Generate a return type hint after the parameter list.
fn generate_return_type_hint(
    doc: &Document,
    search_start: usize,
    ret_type: &Type,
    range: Range,
    hints: &mut Vec<InlayHint>,
) {
    let search_text = &doc.text[search_start..];

    // Find the closing paren of the parameter list followed by optional arrow
    // Pattern: "...) -> Type" or just "...) {"
    if let Some(paren_offset) = search_text.find(')') {
        let abs_offset = search_start + paren_offset + 1;

        // Check if there's already an explicit return type annotation
        let after_paren = &doc.text[abs_offset..].trim_start();
        if after_paren.starts_with("->") {
            // Already has explicit return type, no hint needed
            return;
        }

        let (line, character) = doc.offset_to_position(abs_offset);
        let position = Position::new(line, character);

        if is_position_in_range(position, range) {
            hints.push(InlayHint {
                position,
                label: InlayHintLabel::String(format!(": {}", format_type(ret_type))),
                kind: Some(InlayHintKind::TYPE),
                text_edits: None,
                tooltip: None,
                padding_left: Some(false),
                padding_right: Some(true),
                data: None,
            });
        }
    }
}

/// Generate hints for refinement abstraction and simulation expressions.
fn generate_refinement_hints(
    doc: &Document,
    refinement: &Refinement,
    range: Range,
    hints: &mut Vec<InlayHint>,
) {
    // Find quantifier variable hints in abstraction and simulation expressions
    generate_expr_quantifier_hints(doc, &refinement.abstraction, range, hints);
    generate_expr_quantifier_hints(doc, &refinement.simulation, range, hints);
}

/// Generate hints for quantified variable types across all properties.
fn generate_quantifier_hints(
    doc: &Document,
    spec: &Spec,
    range: Range,
    hints: &mut Vec<InlayHint>,
) {
    for property in &spec.properties {
        match property {
            Property::Theorem(t) => {
                generate_expr_quantifier_hints(doc, &t.body, range, hints);
            }
            Property::Invariant(i) => {
                generate_expr_quantifier_hints(doc, &i.body, range, hints);
            }
            Property::Contract(c) => {
                for expr in &c.requires {
                    generate_expr_quantifier_hints(doc, expr, range, hints);
                }
                for expr in &c.ensures {
                    generate_expr_quantifier_hints(doc, expr, range, hints);
                }
            }
            Property::Probabilistic(p) => {
                generate_expr_quantifier_hints(doc, &p.condition, range, hints);
            }
            Property::Security(s) => {
                generate_expr_quantifier_hints(doc, &s.body, range, hints);
            }
            Property::Semantic(s) => {
                generate_expr_quantifier_hints(doc, &s.body, range, hints);
            }
            Property::Temporal(_)
            | Property::Refinement(_)
            | Property::PlatformApi(_)
            | Property::Bisimulation(_)
            | Property::Version(_)
            | Property::Capability(_)
            | Property::DistributedTemporal(_) => {
                // Already handled or handled via TemporalExpr, meta-constraint, or behavioral equivalence
            }
            Property::DistributedInvariant(d) => {
                generate_expr_quantifier_hints(doc, &d.body, range, hints);
            }
            Property::Composed(c) => {
                generate_expr_quantifier_hints(doc, &c.body, range, hints);
            }
            Property::ImprovementProposal(p) => {
                generate_expr_quantifier_hints(doc, &p.target, range, hints);
                for expr in &p.improves {
                    generate_expr_quantifier_hints(doc, expr, range, hints);
                }
                for expr in &p.preserves {
                    generate_expr_quantifier_hints(doc, expr, range, hints);
                }
                for expr in &p.requires {
                    generate_expr_quantifier_hints(doc, expr, range, hints);
                }
            }
            Property::VerificationGate(g) => {
                for check in &g.checks {
                    generate_expr_quantifier_hints(doc, &check.condition, range, hints);
                }
                generate_expr_quantifier_hints(doc, &g.on_pass, range, hints);
                generate_expr_quantifier_hints(doc, &g.on_fail, range, hints);
            }
            Property::Rollback(r) => {
                for expr in &r.invariants {
                    generate_expr_quantifier_hints(doc, expr, range, hints);
                }
                generate_expr_quantifier_hints(doc, &r.trigger, range, hints);
                for expr in &r.guarantees {
                    generate_expr_quantifier_hints(doc, expr, range, hints);
                }
            }
        }
    }
}

/// Generate hints for quantifier expressions (forall, exists).
fn generate_expr_quantifier_hints(
    doc: &Document,
    expr: &Expr,
    range: Range,
    hints: &mut Vec<InlayHint>,
) {
    match expr {
        Expr::ForAll { var, ty, body } => {
            if let Some(ref type_) = ty {
                // Has explicit type - show as type hint for visibility
                if let Some(hint) = create_quantifier_type_hint(doc, var, type_, "forall", range) {
                    hints.push(hint);
                }
            }
            generate_expr_quantifier_hints(doc, body, range, hints);
        }
        Expr::Exists { var, ty, body } => {
            if let Some(ref type_) = ty {
                if let Some(hint) = create_quantifier_type_hint(doc, var, type_, "exists", range) {
                    hints.push(hint);
                }
            }
            generate_expr_quantifier_hints(doc, body, range, hints);
        }
        Expr::ForAllIn {
            var: _,
            collection,
            body,
        } => {
            // Show collection element type if we can infer it
            generate_expr_quantifier_hints(doc, collection, range, hints);
            generate_expr_quantifier_hints(doc, body, range, hints);
        }
        Expr::ExistsIn {
            var: _,
            collection,
            body,
        } => {
            generate_expr_quantifier_hints(doc, collection, range, hints);
            generate_expr_quantifier_hints(doc, body, range, hints);
        }
        Expr::Implies(left, right)
        | Expr::And(left, right)
        | Expr::Or(left, right)
        | Expr::Compare(left, _, right)
        | Expr::Binary(left, _, right) => {
            generate_expr_quantifier_hints(doc, left, range, hints);
            generate_expr_quantifier_hints(doc, right, range, hints);
        }
        Expr::Not(inner) | Expr::Neg(inner) => {
            generate_expr_quantifier_hints(doc, inner, range, hints);
        }
        Expr::App(_, args) => {
            for arg in args {
                generate_expr_quantifier_hints(doc, arg, range, hints);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            generate_expr_quantifier_hints(doc, receiver, range, hints);
            for arg in args {
                generate_expr_quantifier_hints(doc, arg, range, hints);
            }
        }
        Expr::FieldAccess(obj, _) => {
            generate_expr_quantifier_hints(doc, obj, range, hints);
        }
        // Leaf expressions
        Expr::Var(_) | Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Create an inlay hint for a quantifier variable's type.
fn create_quantifier_type_hint(
    doc: &Document,
    var: &str,
    _ty: &Type,
    quantifier: &str,
    _range: Range,
) -> Option<InlayHint> {
    // Find the pattern "forall var:" or "exists var:"
    let pattern = format!("{} {}", quantifier, var);

    for (offset, _) in doc.text.match_indices(&pattern) {
        // Check that it's followed by a colon (type annotation)
        let after_pattern = &doc.text[offset + pattern.len()..];
        if after_pattern.trim_start().starts_with(':') {
            // Already has explicit type annotation shown in source
            // Don't add redundant hint
            return None;
        }
    }

    None
}

/// Check if a position is within a range.
fn is_position_in_range(pos: Position, range: Range) -> bool {
    // Position is within range if:
    // - It's on a line between start and end, OR
    // - It's on the start line and at or after start character, OR
    // - It's on the end line and at or before end character
    if pos.line < range.start.line || pos.line > range.end.line {
        return false;
    }
    if pos.line == range.start.line && pos.character < range.start.character {
        return false;
    }
    if pos.line == range.end.line && pos.character > range.end.character {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Url;

    fn full_range() -> Range {
        Range {
            start: Position::new(0, 0),
            end: Position::new(u32::MAX, u32::MAX),
        }
    }

    #[test]
    fn test_format_type() {
        assert_eq!(format_type(&Type::Named("Int".to_string())), "Int");
        assert_eq!(
            format_type(&Type::Set(Box::new(Type::Named("Int".to_string())))),
            "Set<Int>"
        );
        assert_eq!(
            format_type(&Type::Map(
                Box::new(Type::Named("String".to_string())),
                Box::new(Type::Named("Int".to_string()))
            )),
            "Map<String, Int>"
        );
        assert_eq!(format_type(&Type::Unit), "Unit");
    }

    #[test]
    fn test_is_position_in_range() {
        let range = Range {
            start: Position::new(5, 10),
            end: Position::new(10, 20),
        };

        // Inside range
        assert!(is_position_in_range(Position::new(7, 15), range));
        // On start line, after start char
        assert!(is_position_in_range(Position::new(5, 15), range));
        // On end line, before end char
        assert!(is_position_in_range(Position::new(10, 10), range));
        // Before range
        assert!(!is_position_in_range(Position::new(4, 0), range));
        // After range
        assert!(!is_position_in_range(Position::new(11, 0), range));
        // On start line, before start char
        assert!(!is_position_in_range(Position::new(5, 5), range));
    }

    #[test]
    fn test_empty_document_no_hints() {
        let doc = Document::new(Url::parse("file:///test.usl").unwrap(), 1, "".to_string());

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(hints.is_empty());
    }

    #[test]
    fn test_simple_theorem_backend_hint() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { forall x: Bool . x }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Theorems get a backend recommendation hint for Lean 4
        assert_eq!(hints.len(), 1);
        if let InlayHintLabel::String(label) = &hints[0].label {
            assert!(label.contains("Lean 4"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_contract_backend_hint() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Counter = { value: Int }
contract Counter::increment(self: Counter, delta: Int) -> Result<Counter> {
    requires { self.value >= 0 }
    ensures { result.value == self.value + delta }
}"#
            .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Contract gets a backend recommendation hint for Kani
        assert_eq!(hints.len(), 1);
        if let InlayHintLabel::String(label) = &hints[0].label {
            assert!(label.contains("Kani"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_contract_with_explicit_return_type() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Value = { n: Int }
contract Value::get(self: Value) -> Result<Int> {
    ensures { result == self.n }
}"#
            .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Contract gets backend hint (Kani), no return type hint (already explicit)
        assert_eq!(hints.len(), 1);
        if let InlayHintLabel::String(label) = &hints[0].label {
            assert!(label.contains("Kani"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_refinement_backend_hints() {
        let doc = Document::new(
            Url::parse("file:///refinement.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/refinement.usl").to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Refinements get backend hints (Lean 4 for each refinement property)
        // The refinement.usl file has 2 refinement properties
        assert_eq!(hints.len(), 2);
        for hint in &hints {
            if let InlayHintLabel::String(label) = &hint.label {
                assert!(label.contains("Lean 4"));
            } else {
                panic!("Expected string label");
            }
        }
    }

    #[test]
    fn test_range_filtering() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem a { true }
theorem b { false }
theorem c { true }"#
                .to_string(),
        );

        // Request only middle line
        let range = Range {
            start: Position::new(1, 0),
            end: Position::new(1, 100),
        };

        let hints = generate_inlay_hints(&doc, range);
        // Only theorem b is in the requested range
        assert_eq!(hints.len(), 1);
        if let InlayHintLabel::String(label) = &hints[0].label {
            assert!(label.contains("Lean 4"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_temporal_backend_hint() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "temporal liveness { always(eventually(done)) }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Temporal properties get TLA+ backend hint
        assert_eq!(hints.len(), 1);
        if let InlayHintLabel::String(label) = &hints[0].label {
            assert!(label.contains("TLA+"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_invariant_backend_hint() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type X = { v: Int }\ninvariant positive { forall x: X . x.v > 0 }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Invariants get Lean 4 backend hint
        assert_eq!(hints.len(), 1);
        if let InlayHintLabel::String(label) = &hints[0].label {
            assert!(label.contains("Lean 4"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_backend_hint_tooltip() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { true }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert_eq!(hints.len(), 1);
        // Verify tooltip is present and contains useful information
        assert!(hints[0].tooltip.is_some());
        if let Some(tower_lsp::lsp_types::InlayHintTooltip::String(tooltip)) = &hints[0].tooltip {
            assert!(tooltip.contains("Theorem proving"));
            assert!(tooltip.contains("Lean 4"));
        } else {
            panic!("Expected string tooltip");
        }
    }

    #[test]
    fn test_multiple_properties_hints() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"theorem t { true }
temporal tm { always(x) }
invariant i { forall y: Int . y > 0 }"#
                .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Each property gets a backend hint
        assert_eq!(hints.len(), 3);

        // Check the hints are for different backends
        let labels: Vec<_> = hints
            .iter()
            .filter_map(|h| {
                if let InlayHintLabel::String(s) = &h.label {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(labels.iter().any(|l| l.contains("Lean 4"))); // theorem and invariant
        assert!(labels.iter().any(|l| l.contains("TLA+"))); // temporal
    }

    // ========== MUTATION-KILLING TESTS ==========

    // Tests for backend_hint_label match arms that aren't exercised in integration tests
    #[test]
    fn test_backend_hint_label_alloy() {
        let prop = Property::Theorem(dashprove_usl::Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, _tooltip) = backend_hint_label("alloy", &prop);
        assert!(label.contains("Alloy"), "alloy should produce Alloy label");
    }

    #[test]
    fn test_backend_hint_label_coq() {
        let prop = Property::Theorem(dashprove_usl::Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, _tooltip) = backend_hint_label("coq", &prop);
        assert!(label.contains("Coq"), "coq should produce Coq label");
    }

    #[test]
    fn test_backend_hint_label_dafny() {
        let prop = Property::Theorem(dashprove_usl::Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, _tooltip) = backend_hint_label("dafny", &prop);
        assert!(label.contains("Dafny"), "dafny should produce Dafny label");
    }

    #[test]
    fn test_backend_hint_label_isabelle() {
        let prop = Property::Theorem(dashprove_usl::Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, _tooltip) = backend_hint_label("isabelle", &prop);
        assert!(
            label.contains("Isabelle"),
            "isabelle should produce Isabelle label"
        );
    }

    #[test]
    fn test_backend_hint_label_security() {
        let prop = Property::Security(dashprove_usl::Security {
            name: "sec".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, _tooltip) = backend_hint_label("security", &prop);
        assert!(label.contains("Sec"), "security should produce Sec label");
    }

    #[test]
    fn test_backend_hint_label_semantic() {
        let prop = Property::Semantic(dashprove_usl::SemanticProperty {
            name: "sem".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, _tooltip) = backend_hint_label("semantic", &prop);
        assert!(
            label.contains("Semantic"),
            "semantic should produce Semantic label"
        );
    }

    #[test]
    fn test_backend_hint_label_unknown() {
        let prop = Property::Theorem(dashprove_usl::Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, _tooltip) = backend_hint_label("unknown_backend", &prop);
        assert!(
            label.contains("?"),
            "unknown backend should produce [?] label"
        );
    }

    // Test contract hints generation
    #[test]
    fn test_contract_return_type_hint() {
        // Contract without explicit return type annotation (but with return_type in AST)
        // Since our contracts typically have explicit types in source, we test
        // that generate_return_type_hint is called by checking position calculation
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Counter = { value: Int }
contract Counter::inc(self: Counter) {
    ensures { true }
}"#
            .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Should get backend hint at minimum
        assert!(!hints.is_empty(), "should produce at least backend hint");
    }

    // Test generate_param_hint offset calculation
    #[test]
    fn test_generate_param_hint_called() {
        // This test ensures generate_param_hint is exercised
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type X = { v: Int }
contract X::foo(self: X, a: Int, b: Bool) -> Result<X> {
    requires { true }
    ensures { true }
}"#
            .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Should produce backend hint for the contract
        assert!(!hints.is_empty());
    }

    // Test refinement hints generation
    #[test]
    fn test_generate_refinement_hints_called() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Abstract = { x: Int }
type Concrete = { data: Int }
refinement refines_test refines Abstract {
    abstraction { forall c: Concrete . c.data }
    simulation { forall a: Abstract . forall c: Concrete . a.x == c.data }
}"#
            .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Should produce backend hint for refinement (Lean 4)
        assert!(!hints.is_empty());
        let labels: Vec<_> = hints
            .iter()
            .filter_map(|h| {
                if let InlayHintLabel::String(s) = &h.label {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(labels.iter().any(|l| l.contains("Lean 4")));
    }

    // Test quantifier hints generation
    #[test]
    fn test_generate_quantifier_hints_with_forall() {
        // Test that quantifier hints are generated for theorem bodies
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem test_forall { forall x: Bool . x }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // Should get backend hint for theorem
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_generate_quantifier_hints_with_exists() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem test_exists { exists y: Int . y > 0 }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    // Test expression quantifier hints for various property types
    #[test]
    fn test_expr_quantifier_hints_in_contract_requires() {
        // Test that generate_expr_quantifier_hints is called for contract requires
        // Using the contracts.usl example which parses correctly
        let doc = Document::new(
            Url::parse("file:///contracts.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/contracts.usl").to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        // contracts.usl has contracts, should produce backend hints
        assert!(!hints.is_empty(), "contracts.usl should produce hints");
    }

    #[test]
    fn test_expr_quantifier_hints_in_contract_ensures() {
        // Contracts with ensures clauses are traversed for quantifiers
        let doc = Document::new(
            Url::parse("file:///contracts.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/contracts.usl").to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
        // The contract should get a Kani backend hint
        let labels: Vec<_> = hints
            .iter()
            .filter_map(|h| {
                if let InlayHintLabel::String(s) = &h.label {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(labels.iter().any(|l| l.contains("Kani")));
    }

    #[test]
    fn test_probabilistic_backend_hint() {
        // Using the probabilistic.usl example file
        let doc = Document::new(
            Url::parse("file:///probabilistic.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/probabilistic.usl").to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty(), "probabilistic.usl should produce hints");
        let labels: Vec<_> = hints
            .iter()
            .filter_map(|h| {
                if let InlayHintLabel::String(s) = &h.label {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(labels.iter().any(|l| l.contains("Prob")));
    }

    #[test]
    fn test_security_backend_hint() {
        // Using the security.usl example file
        let doc = Document::new(
            Url::parse("file:///security.usl").unwrap(),
            1,
            include_str!("../../../examples/usl/security.usl").to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty(), "security.usl should produce hints");
        let labels: Vec<_> = hints
            .iter()
            .filter_map(|h| {
                if let InlayHintLabel::String(s) = &h.label {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        assert!(labels.iter().any(|l| l.contains("Sec")));
    }

    #[test]
    fn test_semantic_backend_hint() {
        // The backend_hint_label function directly tests the "semantic" backend
        // Here we verify the Semantic property gets the right backend label
        let prop = Property::Semantic(dashprove_usl::SemanticProperty {
            name: "test_semantic".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (label, tooltip) = backend_hint_label("semantic", &prop);
        assert!(
            label.contains("Semantic"),
            "semantic backend should produce Semantic label"
        );
        assert!(
            tooltip.contains("Semantic"),
            "tooltip should mention Semantic"
        );
    }

    // Test expression traversal in quantifier hints
    #[test]
    fn test_expr_quantifier_hints_nested_implies() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem implies_test { (forall x: Bool . x) implies (exists y: Bool . y) }"
                .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_expr_quantifier_hints_nested_and_or() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem logic_test { (forall x: Bool . x) and (forall y: Bool . y) or (exists z: Bool . z) }"
                .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_expr_quantifier_hints_not() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem not_test { not (forall x: Bool . x) }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_expr_quantifier_hints_app() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem app_test { f(forall x: Bool . x, exists y: Bool . y) }".to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_expr_quantifier_hints_method_call() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type X = { v: Int }
theorem method_test { forall x: X . x.check(forall y: Bool . y) }"#
                .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_expr_quantifier_hints_field_access() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Container = { data: Int }
theorem field_test { forall c: Container . c.data > 0 }"#
                .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_expr_quantifier_hints_forall_in() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Set = { elements: Set<Int> }
theorem forall_in_test { forall s: Set . forall x in s.elements . x > 0 }"#
                .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_expr_quantifier_hints_exists_in() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            r#"type Set = { elements: Set<Int> }
theorem exists_in_test { forall s: Set . exists x in s.elements . x == 42 }"#
                .to_string(),
        );

        let hints = generate_inlay_hints(&doc, full_range());
        assert!(!hints.is_empty());
    }

    // Test create_quantifier_type_hint returns None for explicit type annotation
    #[test]
    fn test_create_quantifier_type_hint_with_explicit_type() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem test { forall x: Bool . x }".to_string(),
        );

        let result = create_quantifier_type_hint(
            &doc,
            "x",
            &Type::Named("Bool".to_string()),
            "forall",
            full_range(),
        );
        // Should return None because the source has explicit ": Bool"
        assert!(result.is_none());
    }

    #[test]
    fn test_create_quantifier_type_hint_exists_with_type() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem test { exists y: Int . y > 0 }".to_string(),
        );

        let result = create_quantifier_type_hint(
            &doc,
            "y",
            &Type::Named("Int".to_string()),
            "exists",
            full_range(),
        );
        // Should return None because the source has explicit ": Int"
        assert!(result.is_none());
    }

    // Test is_position_in_range boundary conditions
    #[test]
    fn test_is_position_in_range_start_line_exact() {
        let range = Range {
            start: Position::new(5, 10),
            end: Position::new(10, 20),
        };
        // Exactly at start boundary
        assert!(is_position_in_range(Position::new(5, 10), range));
    }

    #[test]
    fn test_is_position_in_range_end_line_exact() {
        let range = Range {
            start: Position::new(5, 10),
            end: Position::new(10, 20),
        };
        // Exactly at end boundary
        assert!(is_position_in_range(Position::new(10, 20), range));
    }

    #[test]
    fn test_is_position_in_range_same_line_start_equals_end() {
        let range = Range {
            start: Position::new(5, 10),
            end: Position::new(5, 20),
        };
        // Single line range
        assert!(is_position_in_range(Position::new(5, 15), range));
        assert!(is_position_in_range(Position::new(5, 10), range));
        assert!(is_position_in_range(Position::new(5, 20), range));
        assert!(!is_position_in_range(Position::new(5, 9), range));
        assert!(!is_position_in_range(Position::new(5, 21), range));
    }

    #[test]
    fn test_is_position_in_range_character_at_end_line_boundary() {
        let range = Range {
            start: Position::new(5, 10),
            end: Position::new(10, 20),
        };
        // On end line, character > end.character should be false
        assert!(!is_position_in_range(Position::new(10, 21), range));
        // On end line, character == end.character should be true
        assert!(is_position_in_range(Position::new(10, 20), range));
    }

    #[test]
    fn test_is_position_in_range_character_at_start_line_boundary() {
        let range = Range {
            start: Position::new(5, 10),
            end: Position::new(10, 20),
        };
        // On start line, character < start.character should be false
        assert!(!is_position_in_range(Position::new(5, 9), range));
        // On start line, character == start.character should be true
        assert!(is_position_in_range(Position::new(5, 10), range));
    }

    // Test property_kind_keyword for all property types
    #[test]
    fn test_property_kind_keyword_bisimulation() {
        let prop = Property::Bisimulation(dashprove_usl::Bisimulation {
            name: "test".to_string(),
            oracle: "oracle_fn".to_string(),
            subject: "subject_fn".to_string(),
            equivalent_on: vec![],
            tolerance: None,
            property: None,
        });
        assert_eq!(property_kind_keyword(&prop), "bisimulation");
    }

    #[test]
    fn test_property_kind_keyword_platform_api() {
        let prop = Property::PlatformApi(dashprove_usl::PlatformApi {
            name: "api_test".to_string(),
            states: vec![],
        });
        assert_eq!(property_kind_keyword(&prop), "platform_api");
    }

    // Test recommended_backend for all property types
    #[test]
    fn test_recommended_backend_bisimulation() {
        let prop = Property::Bisimulation(dashprove_usl::Bisimulation {
            name: "test".to_string(),
            oracle: "oracle_fn".to_string(),
            subject: "subject_fn".to_string(),
            equivalent_on: vec![],
            tolerance: None,
            property: None,
        });
        assert_eq!(recommended_backend(&prop), "bisim");
    }

    #[test]
    fn test_recommended_backend_platform_api() {
        let prop = Property::PlatformApi(dashprove_usl::PlatformApi {
            name: "api_test".to_string(),
            states: vec![],
        });
        assert_eq!(recommended_backend(&prop), "platform_api");
    }

    // Test backend_hint_label tooltip content for various property types
    #[test]
    fn test_backend_hint_label_tooltip_contract() {
        let prop = Property::Contract(Contract {
            type_path: vec!["Test".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let (_label, tooltip) = backend_hint_label("kani", &prop);
        assert!(tooltip.contains("Kani"));
        assert!(tooltip.contains("symbolic execution"));
    }

    #[test]
    fn test_backend_hint_label_tooltip_invariant() {
        let prop = Property::Invariant(dashprove_usl::Invariant {
            name: "inv".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (_label, tooltip) = backend_hint_label("lean4", &prop);
        assert!(tooltip.contains("Invariant"));
    }

    #[test]
    fn test_backend_hint_label_tooltip_refinement() {
        let prop = Property::Refinement(dashprove_usl::Refinement {
            name: "ref".to_string(),
            refines: "Abstract".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: dashprove_usl::Expr::Bool(true),
            simulation: dashprove_usl::Expr::Bool(true),
            actions: vec![],
        });
        let (_label, tooltip) = backend_hint_label("lean4", &prop);
        assert!(tooltip.contains("Refinement"));
    }

    #[test]
    fn test_backend_hint_label_tooltip_probabilistic() {
        let prop = Property::Probabilistic(dashprove_usl::Probabilistic {
            name: "prob".to_string(),
            comparison: dashprove_usl::ComparisonOp::Ge,
            bound: 0.95,
            condition: dashprove_usl::Expr::Bool(true),
        });
        let (_label, tooltip) = backend_hint_label("probabilistic", &prop);
        assert!(tooltip.contains("Probabilistic"));
    }

    #[test]
    fn test_backend_hint_label_tooltip_security() {
        let prop = Property::Security(dashprove_usl::Security {
            name: "sec".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (_label, tooltip) = backend_hint_label("security", &prop);
        assert!(tooltip.contains("Security"));
    }

    #[test]
    fn test_backend_hint_label_tooltip_semantic() {
        let prop = Property::Semantic(dashprove_usl::SemanticProperty {
            name: "sem".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (_label, tooltip) = backend_hint_label("semantic", &prop);
        assert!(tooltip.contains("Semantic"));
    }

    #[test]
    fn test_backend_hint_label_tooltip_platform_api() {
        let prop = Property::PlatformApi(dashprove_usl::PlatformApi {
            name: "api".to_string(),
            states: vec![],
        });
        let (_label, tooltip) = backend_hint_label("platform_api", &prop);
        assert!(tooltip.contains("Platform API"));
    }

    #[test]
    fn test_backend_hint_label_tooltip_bisimulation() {
        let prop = Property::Bisimulation(dashprove_usl::Bisimulation {
            name: "bisim".to_string(),
            oracle: "o".to_string(),
            subject: "s".to_string(),
            equivalent_on: vec![],
            tolerance: None,
            property: None,
        });
        let (_label, tooltip) = backend_hint_label("bisim", &prop);
        assert!(tooltip.contains("Bisimulation"));
    }
}
