//! Diagnostic generation for USL documents.
//!
//! This module provides functions for generating LSP diagnostics,
//! including expert hints for properties and error position extraction.

use crate::commands::recommended_backend;
use crate::document::Document;
use dashprove_usl::Property;
use tower_lsp::lsp_types::*;

/// Generate an informational diagnostic with expert hints for a property.
pub fn generate_property_hint_diagnostic(
    doc: &Document,
    property: &Property,
) -> Option<Diagnostic> {
    let name = property.name();
    let kind = property_kind_keyword(property);
    let backend = recommended_backend(property);

    // Find the property definition line
    let pattern = if matches!(property, Property::Contract(_)) {
        format!("contract {}", name)
    } else {
        format!("{} {}", kind, name)
    };

    let offset = doc.text.find(&pattern)?;
    let (line, _) = doc.offset_to_position(offset);

    // Generate hint based on property type
    let (message, code) = property_expert_hint(property, backend);

    Some(Diagnostic {
        range: Range {
            start: Position::new(line, 0),
            end: Position::new(line, pattern.len() as u32),
        },
        severity: Some(DiagnosticSeverity::HINT),
        code: Some(NumberOrString::String(code.to_string())),
        code_description: None,
        source: Some("dashprove-expert".to_string()),
        message,
        related_information: None,
        tags: None,
        data: Some(serde_json::json!({
            "property_name": name,
            "property_kind": kind,
            "backend": backend,
        })),
    })
}

/// Get the keyword used to define a property (for diagnostic generation).
pub fn property_kind_keyword(prop: &Property) -> &'static str {
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

/// Generate expert hint message and code for a property.
pub fn property_expert_hint(property: &Property, backend: &str) -> (String, &'static str) {
    match property {
        Property::Theorem(t) => {
            let hint = if t.body.contains_quantifier() {
                format!(
                    "Theorem '{}' will be verified with {}. Consider using 'intro' followed by case analysis for quantified goals.",
                    t.name, backend_display_name(backend)
                )
            } else {
                format!(
                    "Theorem '{}' will be verified with {}. Use 'simp' or 'decide' for simple boolean expressions.",
                    t.name, backend_display_name(backend)
                )
            };
            (hint, "dashprove.theorem-hint")
        }
        Property::Temporal(t) => {
            let hint = format!(
                "Temporal property '{}' will be model-checked with {}. Ensure state space is finite for exhaustive checking.",
                t.name, backend_display_name(backend)
            );
            (hint, "dashprove.temporal-hint")
        }
        Property::Contract(c) => {
            let name = c.type_path.join("::");
            let hint = if c.ensures_err.is_empty() {
                format!(
                    "Contract '{}' will be verified with {}. Consider adding 'ensures_err' clauses for error handling.",
                    name, backend_display_name(backend)
                )
            } else {
                format!(
                    "Contract '{}' with error handling will be verified with {}. Good practice: both success and error paths are specified.",
                    name, backend_display_name(backend)
                )
            };
            (hint, "dashprove.contract-hint")
        }
        Property::Invariant(i) => {
            let hint = format!(
                "Invariant '{}' will be verified with {}. For bounded checking, consider Alloy as an alternative.",
                i.name, backend_display_name(backend)
            );
            (hint, "dashprove.invariant-hint")
        }
        Property::Refinement(r) => {
            let hint = format!(
                "Refinement '{}' proves implementation matches specification using {}. Abstraction function must be injective.",
                r.name, backend_display_name(backend)
            );
            (hint, "dashprove.refinement-hint")
        }
        Property::Probabilistic(p) => {
            let hint = format!(
                "Probabilistic property '{}' will analyze probability bounds. Ensure sample space is well-defined.",
                p.name
            );
            (hint, "dashprove.probabilistic-hint")
        }
        Property::Security(s) => {
            let hint = format!(
                "Security property '{}' will analyze information flow. Consider non-interference and confidentiality policies.",
                s.name
            );
            (hint, "dashprove.security-hint")
        }
        Property::Semantic(s) => {
            let hint = format!(
                "Semantic property '{}' will be checked with embedding-based similarity and semantic predicates. Configure thresholds like semantic_similarity >= 0.8.",
                s.name
            );
            (hint, "dashprove.semantic-hint")
        }
        Property::PlatformApi(p) => {
            let hint = format!(
                "Platform API '{}' defines external platform constraints. These inform verification assumptions.",
                p.name
            );
            (hint, "dashprove.platform-api-hint")
        }
        Property::Bisimulation(b) => {
            let hint = format!(
                "Bisimulation '{}' verifies behavioral equivalence between oracle and subject implementations.",
                b.name
            );
            (hint, "dashprove.bisimulation-hint")
        }
        Property::Version(v) => {
            let hint = format!(
                "Version '{}' specifies improvement over '{}'. Capabilities must be preserved or enhanced.",
                v.name, v.improves
            );
            (hint, "dashprove.version-hint")
        }
        Property::Capability(c) => {
            let hint = format!(
                "Capability '{}' defines {} abilities with {} requirements.",
                c.name,
                c.abilities.len(),
                c.requires.len()
            );
            (hint, "dashprove.capability-hint")
        }
        Property::DistributedInvariant(d) => {
            let hint = format!(
                "Distributed invariant '{}' will be verified with {} for multi-agent coordination.",
                d.name,
                backend_display_name(backend)
            );
            (hint, "dashprove.distributed-invariant-hint")
        }
        Property::DistributedTemporal(d) => {
            let hint = format!(
                "Distributed temporal '{}' will be model-checked with {} with {} fairness constraints.",
                d.name, backend_display_name(backend), d.fairness.len()
            );
            (hint, "dashprove.distributed-temporal-hint")
        }
        Property::Composed(c) => {
            let hint = format!(
                "Composed theorem '{}' combines {} dependencies and will be verified with {}.",
                c.name,
                c.uses.len(),
                backend_display_name(backend)
            );
            (hint, "dashprove.composed-hint")
        }
        Property::ImprovementProposal(p) => {
            let hint = format!(
                "Improvement proposal '{}' improves {} properties and will be verified with {}.",
                p.name,
                p.improves.len(),
                backend_display_name(backend)
            );
            (hint, "dashprove.improvement-proposal-hint")
        }
        Property::VerificationGate(g) => {
            let hint = format!(
                "Verification gate '{}' has {} checks and will be verified with {}.",
                g.name,
                g.checks.len(),
                backend_display_name(backend)
            );
            (hint, "dashprove.verification-gate-hint")
        }
        Property::Rollback(r) => {
            let hint = format!(
                "Rollback '{}' has {} invariants and will be verified with {}.",
                r.name,
                r.invariants.len(),
                backend_display_name(backend)
            );
            (hint, "dashprove.rollback-hint")
        }
    }
}

/// Get display name for a backend.
pub fn backend_display_name(backend: &str) -> &'static str {
    match backend {
        "lean4" => "Lean 4",
        "tlaplus" => "TLA+",
        "kani" => "Kani",
        "alloy" => "Alloy",
        "coq" => "Coq",
        "isabelle" => "Isabelle",
        "dafny" => "Dafny",
        "probabilistic" => "Probabilistic Checker",
        "security" => "Security Analyzer",
        "semantic" => "Semantic Checker",
        _ => "Unknown Backend",
    }
}

/// Extract line/column from pest error message.
pub fn extract_position_from_error(msg: &str) -> (u32, u32) {
    // Pest errors look like: "... --> 1:5 ..."
    if let Some(pos) = msg.find(" --> ") {
        let rest = &msg[pos + 5..];
        if let Some(colon) = rest.find(':') {
            let line_str = &rest[..colon];
            let col_start = colon + 1;
            let col_end = rest[col_start..]
                .find(|c: char| !c.is_ascii_digit())
                .map(|i| col_start + i)
                .unwrap_or(rest.len());
            let col_str = &rest[col_start..col_end];

            if let (Ok(line), Ok(col)) = (line_str.parse::<u32>(), col_str.parse::<u32>()) {
                return (line.saturating_sub(1), col.saturating_sub(1));
            }
        }
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::{Contract, Invariant, Theorem};
    use tower_lsp::lsp_types::Url;

    #[test]
    fn test_extract_position_from_error() {
        let msg = "Parse error:  --> 5:10\n  |\n5 | theorem { }";
        assert_eq!(extract_position_from_error(msg), (4, 9));

        let msg = "Parse error:  --> 1:1\n  |";
        assert_eq!(extract_position_from_error(msg), (0, 0));

        let msg = "Some other error without position";
        assert_eq!(extract_position_from_error(msg), (0, 0));
    }

    #[test]
    fn test_property_kind_keyword() {
        let theorem = Property::Theorem(Theorem {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        assert_eq!(property_kind_keyword(&theorem), "theorem");

        let invariant = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        assert_eq!(property_kind_keyword(&invariant), "invariant");
    }

    #[test]
    fn test_backend_display_name() {
        assert_eq!(backend_display_name("lean4"), "Lean 4");
        assert_eq!(backend_display_name("tlaplus"), "TLA+");
        assert_eq!(backend_display_name("kani"), "Kani");
        assert_eq!(backend_display_name("alloy"), "Alloy");
        assert_eq!(backend_display_name("coq"), "Coq");
        assert_eq!(backend_display_name("isabelle"), "Isabelle");
        assert_eq!(backend_display_name("dafny"), "Dafny");
        assert_eq!(
            backend_display_name("probabilistic"),
            "Probabilistic Checker"
        );
        assert_eq!(backend_display_name("security"), "Security Analyzer");
        assert_eq!(backend_display_name("unknown"), "Unknown Backend");
    }

    #[test]
    fn test_backend_display_name_semantic() {
        // Mutation test: ensure "semantic" arm is covered
        assert_eq!(backend_display_name("semantic"), "Semantic Checker");
        // Verify it's different from other backends
        assert_ne!(backend_display_name("semantic"), "Unknown Backend");
        assert_ne!(backend_display_name("semantic"), "Security Analyzer");
    }

    #[test]
    fn test_extract_position_offset_arithmetic() {
        // Mutation test: verify the +5 offset in " --> " parsing
        // The pattern " --> " is 5 chars, so content starts at pos+5
        let msg = "error --> 3:7\nrest";
        let (line, col) = extract_position_from_error(msg);
        assert_eq!(line, 2); // 3-1 = 2 (0-indexed)
        assert_eq!(col, 6); // 7-1 = 6 (0-indexed)

        // Test with content right after --> to verify offset calculation
        let msg2 = "x --> 10:20 y";
        let (line2, col2) = extract_position_from_error(msg2);
        assert_eq!(line2, 9);
        assert_eq!(col2, 19);

        // Test edge case: pos=0 means " --> " at start, content at index 5
        let msg3 = " --> 1:1";
        let (line3, col3) = extract_position_from_error(msg3);
        assert_eq!(line3, 0);
        assert_eq!(col3, 0);
    }

    #[test]
    fn test_property_expert_hint_theorem() {
        // Simple theorem without quantifiers
        let simple_theorem = Property::Theorem(Theorem {
            name: "simple".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (hint, code) = property_expert_hint(&simple_theorem, "lean4");
        assert!(hint.contains("simple"));
        assert!(hint.contains("Lean 4"));
        assert!(hint.contains("simp") || hint.contains("decide"));
        assert_eq!(code, "dashprove.theorem-hint");

        // Theorem with quantifiers
        let quantified_theorem = Property::Theorem(Theorem {
            name: "quantified".to_string(),
            body: dashprove_usl::Expr::ForAll {
                var: "x".to_string(),
                ty: None,
                body: Box::new(dashprove_usl::Expr::Bool(true)),
            },
        });
        let (hint, _) = property_expert_hint(&quantified_theorem, "lean4");
        assert!(hint.contains("intro"));
    }

    #[test]
    fn test_property_expert_hint_contract() {
        // Contract without ensures_err
        let no_err_contract = Property::Contract(Contract {
            type_path: vec!["Stack".to_string(), "push".to_string()],
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
        let (hint, code) = property_expert_hint(&no_err_contract, "kani");
        assert!(hint.contains("Stack::push"));
        assert!(hint.contains("Kani"));
        assert!(hint.contains("ensures_err"));
        assert_eq!(code, "dashprove.contract-hint");

        // Contract with ensures_err (good practice)
        let with_err_contract = Property::Contract(Contract {
            type_path: vec!["Stack".to_string(), "pop".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![],
            ensures: vec![],
            ensures_err: vec![dashprove_usl::Expr::Bool(true)],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        });
        let (hint, _) = property_expert_hint(&with_err_contract, "kani");
        assert!(hint.contains("Good practice"));
    }

    #[test]
    fn test_property_expert_hint_temporal() {
        let temporal = Property::Temporal(dashprove_usl::Temporal {
            name: "test_temporal".to_string(),
            body: dashprove_usl::TemporalExpr::Atom(dashprove_usl::Expr::Bool(true)),
            fairness: vec![],
        });
        let (hint, code) = property_expert_hint(&temporal, "tlaplus");
        assert!(hint.contains("test_temporal"));
        assert!(hint.contains("TLA+"));
        assert!(hint.contains("model-checked"));
        assert_eq!(code, "dashprove.temporal-hint");
    }

    #[test]
    fn test_property_expert_hint_invariant() {
        let invariant = Property::Invariant(Invariant {
            name: "test_inv".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (hint, code) = property_expert_hint(&invariant, "lean4");
        assert!(hint.contains("test_inv"));
        assert!(hint.contains("Lean 4"));
        assert!(hint.contains("Alloy"));
        assert_eq!(code, "dashprove.invariant-hint");
    }

    #[test]
    fn test_property_expert_hint_refinement() {
        let refinement = Property::Refinement(dashprove_usl::Refinement {
            name: "test_ref".to_string(),
            refines: "AbstractStack".to_string(),
            mappings: vec![],
            invariants: vec![],
            abstraction: dashprove_usl::Expr::Bool(true),
            simulation: dashprove_usl::Expr::Bool(true),
            actions: vec![],
        });
        let (hint, code) = property_expert_hint(&refinement, "lean4");
        assert!(hint.contains("test_ref"));
        assert!(hint.contains("injective"));
        assert_eq!(code, "dashprove.refinement-hint");
    }

    #[test]
    fn test_property_expert_hint_probabilistic() {
        let prob = Property::Probabilistic(dashprove_usl::Probabilistic {
            name: "test_prob".to_string(),
            condition: dashprove_usl::Expr::Var("success".to_string()),
            comparison: dashprove_usl::ComparisonOp::Eq,
            bound: 0.95,
        });
        let (hint, code) = property_expert_hint(&prob, "probabilistic");
        assert!(hint.contains("test_prob"));
        assert!(hint.contains("sample space"));
        assert_eq!(code, "dashprove.probabilistic-hint");
    }

    #[test]
    fn test_property_expert_hint_security() {
        let security = Property::Security(dashprove_usl::Security {
            name: "test_sec".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });
        let (hint, code) = property_expert_hint(&security, "security");
        assert!(hint.contains("test_sec"));
        assert!(hint.contains("information flow"));
        assert_eq!(code, "dashprove.security-hint");
    }

    #[test]
    fn test_generate_property_hint_diagnostic() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem my_test { true }".to_string(),
        );

        let theorem = Property::Theorem(Theorem {
            name: "my_test".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });

        let diag = generate_property_hint_diagnostic(&doc, &theorem);
        assert!(diag.is_some());

        let diag = diag.unwrap();
        assert_eq!(diag.severity, Some(DiagnosticSeverity::HINT));
        assert_eq!(diag.source, Some("dashprove-expert".to_string()));
        assert!(diag.message.contains("my_test"));
        assert!(diag.message.contains("Lean 4"));
        assert!(diag.code.is_some());
        assert!(diag.data.is_some());
    }

    #[test]
    fn test_generate_property_hint_diagnostic_contract() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "contract Foo::bar(self: Int) -> Int {\n  requires { true }\n  ensures { true }\n}"
                .to_string(),
        );

        let contract = Property::Contract(Contract {
            type_path: vec!["Foo".to_string(), "bar".to_string()],
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

        let diag = generate_property_hint_diagnostic(&doc, &contract);
        assert!(diag.is_some());

        let diag = diag.unwrap();
        assert_eq!(diag.source, Some("dashprove-expert".to_string()));
        assert!(diag.message.contains("Foo::bar"));
    }

    #[test]
    fn test_generate_property_hint_diagnostic_not_found() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem other { true }".to_string(),
        );

        // Property name doesn't match anything in doc
        let theorem = Property::Theorem(Theorem {
            name: "nonexistent".to_string(),
            body: dashprove_usl::Expr::Bool(true),
        });

        let diag = generate_property_hint_diagnostic(&doc, &theorem);
        assert!(diag.is_none());
    }
}

// ========== Kani proof harnesses ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify backend_display_name returns non-empty for "lean4"
    #[kani::proof]
    fn verify_backend_display_name_lean4() {
        let name = backend_display_name("lean4");
        kani::assert(!name.is_empty(), "lean4 display name should not be empty");
    }

    /// Verify backend_display_name returns non-empty for "tlaplus"
    #[kani::proof]
    fn verify_backend_display_name_tlaplus() {
        let name = backend_display_name("tlaplus");
        kani::assert(!name.is_empty(), "tlaplus display name should not be empty");
    }

    /// Verify backend_display_name returns non-empty for "kani"
    #[kani::proof]
    fn verify_backend_display_name_kani() {
        let name = backend_display_name("kani");
        kani::assert(!name.is_empty(), "kani display name should not be empty");
    }

    /// Verify backend_display_name returns non-empty for "alloy"
    #[kani::proof]
    fn verify_backend_display_name_alloy() {
        let name = backend_display_name("alloy");
        kani::assert(!name.is_empty(), "alloy display name should not be empty");
    }

    /// Verify backend_display_name returns non-empty for "coq"
    #[kani::proof]
    fn verify_backend_display_name_coq() {
        let name = backend_display_name("coq");
        kani::assert(!name.is_empty(), "coq display name should not be empty");
    }

    /// Verify backend_display_name returns non-empty for "isabelle"
    #[kani::proof]
    fn verify_backend_display_name_isabelle() {
        let name = backend_display_name("isabelle");
        kani::assert(
            !name.is_empty(),
            "isabelle display name should not be empty",
        );
    }

    /// Verify backend_display_name returns non-empty for "dafny"
    #[kani::proof]
    fn verify_backend_display_name_dafny() {
        let name = backend_display_name("dafny");
        kani::assert(!name.is_empty(), "dafny display name should not be empty");
    }

    /// Verify backend_display_name returns non-empty for "semantic"
    #[kani::proof]
    fn verify_backend_display_name_semantic() {
        let name = backend_display_name("semantic");
        kani::assert(
            !name.is_empty(),
            "semantic display name should not be empty",
        );
    }

    /// Verify backend_display_name returns non-empty for unknown
    #[kani::proof]
    fn verify_backend_display_name_unknown() {
        let name = backend_display_name("unknown_backend");
        kani::assert(!name.is_empty(), "unknown display name should not be empty");
    }

    /// Verify extract_position_from_error returns (0, 0) for no match
    #[kani::proof]
    fn verify_extract_position_no_match() {
        let (line, col) = extract_position_from_error("no position marker here");
        kani::assert(line == 0, "line should be 0 for no match");
        kani::assert(col == 0, "col should be 0 for no match");
    }

    /// Verify extract_position_from_error parses valid format
    #[kani::proof]
    fn verify_extract_position_valid_format() {
        let (line, col) = extract_position_from_error(" --> 1:1");
        kani::assert(line == 0, "line 1 should become 0");
        kani::assert(col == 0, "col 1 should become 0");
    }

    /// Verify extract_position_from_error subtracts 1 from line
    #[kani::proof]
    fn verify_extract_position_line_subtract() {
        let (line, _) = extract_position_from_error(" --> 5:1");
        kani::assert(line == 4, "line 5 should become 4");
    }

    /// Verify extract_position_from_error subtracts 1 from col
    #[kani::proof]
    fn verify_extract_position_col_subtract() {
        let (_, col) = extract_position_from_error(" --> 1:10");
        kani::assert(col == 9, "col 10 should become 9");
    }

    /// Verify property_kind_keyword returns "theorem" for Theorem
    #[kani::proof]
    fn verify_property_kind_keyword_theorem() {
        let kw = property_kind_keyword(&Property::Theorem(dashprove_usl::Theorem {
            name: String::new(),
            body: dashprove_usl::Expr::Bool(true),
        }));
        kani::assert(kw == "theorem", "should return theorem");
    }

    /// Verify property_kind_keyword returns "invariant" for Invariant
    #[kani::proof]
    fn verify_property_kind_keyword_invariant() {
        let kw = property_kind_keyword(&Property::Invariant(dashprove_usl::Invariant {
            name: String::new(),
            body: dashprove_usl::Expr::Bool(true),
        }));
        kani::assert(kw == "invariant", "should return invariant");
    }

    /// Verify backend_display_name returns different values for different backends
    #[kani::proof]
    fn verify_backend_display_names_differ() {
        let lean = backend_display_name("lean4");
        let kani = backend_display_name("kani");
        kani::assert(
            lean != kani,
            "lean4 and kani should have different display names",
        );
    }
}
