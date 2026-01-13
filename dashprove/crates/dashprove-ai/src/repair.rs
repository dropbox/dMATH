//! Proof repair system
//!
//! When code changes break existing proofs, this module helps identify
//! the cause and suggests repairs.

use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};

/// A diff describing what changed in a proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofDiff {
    /// Lines removed from old proof
    pub removed: Vec<String>,
    /// Lines added in new proof
    pub added: Vec<String>,
    /// Lines that changed (old, new)
    pub changed: Vec<(String, String)>,
}

/// Type of proof repair
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairKind {
    /// Add missing tactics
    AddTactics,
    /// Replace outdated tactics
    ReplaceTactic,
    /// Update type annotations
    UpdateTypes,
    /// Add import or library reference
    AddImport,
    /// Strengthen hypothesis
    StrengthenHypothesis,
    /// Weaken conclusion
    WeakenConclusion,
    /// General refactoring needed
    Refactor,
}

/// A suggestion for repairing a broken proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairSuggestion {
    /// Type of repair
    pub kind: RepairKind,
    /// Human-readable description
    pub description: String,
    /// Suggested fix (code or tactic)
    pub fix: String,
    /// Confidence in this suggestion (0.0 to 1.0)
    pub confidence: f64,
    /// Location hint (line number if known)
    pub location: Option<usize>,
}

/// Detect potential breaks in a proof based on error message
pub fn detect_proof_breaks(error: &str, backend: &BackendId) -> Vec<String> {
    let mut breaks = Vec::new();

    match backend {
        BackendId::Lean4 => {
            if error.contains("unknown identifier") {
                breaks.push("Reference to undefined identifier".to_string());
            }
            if error.contains("type mismatch") {
                breaks.push("Type mismatch - signature may have changed".to_string());
            }
            if error.contains("unsolved goals") {
                breaks.push("Incomplete proof - additional tactics needed".to_string());
            }
            if error.contains("application type mismatch") {
                breaks.push("Function argument types changed".to_string());
            }
            if error.contains("unknown tactic") {
                breaks.push("Tactic not available - check imports".to_string());
            }
            if error.contains("invalid 'simp'") {
                breaks.push("Simp lemma reference is invalid".to_string());
            }
        }
        BackendId::TlaPlus => {
            if error.contains("Unknown operator") {
                breaks.push("Operator definition missing or renamed".to_string());
            }
            if error.contains("Invariant") && error.contains("violated") {
                breaks.push("Invariant no longer holds after code change".to_string());
            }
            if error.contains("deadlock") {
                breaks.push("Specification can now reach deadlock".to_string());
            }
        }
        BackendId::Kani => {
            if error.contains("assertion failed") {
                breaks.push("Assertion no longer holds".to_string());
            }
            if error.contains("undefined behavior") {
                breaks.push("Code may exhibit undefined behavior".to_string());
            }
            if error.contains("overflow") {
                breaks.push("Arithmetic overflow possible".to_string());
            }
        }
        BackendId::Alloy => {
            if error.contains("Counterexample found") {
                breaks.push("Property no longer holds within bounds".to_string());
            }
            if error.contains("Unsatisfiable") {
                breaks.push("Constraints became contradictory".to_string());
            }
        }
        _ => {
            if !error.is_empty() {
                breaks.push("Verification failed".to_string());
            }
        }
    }

    breaks
}

/// Suggest repairs for a failed proof
pub fn suggest_repairs(
    property: &Property,
    old_proof: Option<&str>,
    error: &str,
    backend: &BackendId,
) -> Vec<RepairSuggestion> {
    let mut suggestions = Vec::new();

    match backend {
        BackendId::Lean4 => {
            suggestions.extend(suggest_lean_repairs(property, old_proof, error));
        }
        BackendId::TlaPlus => {
            suggestions.extend(suggest_tlaplus_repairs(property, error));
        }
        BackendId::Kani => {
            suggestions.extend(suggest_kani_repairs(property, error));
        }
        BackendId::Alloy => {
            suggestions.extend(suggest_alloy_repairs(property, error));
        }
        _ => {}
    }

    // Sort by confidence
    suggestions.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    suggestions
}

/// LEAN 4 specific repair suggestions
fn suggest_lean_repairs(
    _property: &Property,
    old_proof: Option<&str>,
    error: &str,
) -> Vec<RepairSuggestion> {
    let mut suggestions = Vec::new();

    // Unknown identifier
    if error.contains("unknown identifier") {
        // Extract the identifier name
        let ident = extract_quoted(error, '\'');
        if let Some(name) = ident {
            suggestions.push(RepairSuggestion {
                kind: RepairKind::AddImport,
                description: format!("'{}' may need to be imported or defined", name),
                fix: format!("import Mathlib.Tactic -- or define {}", name),
                confidence: 0.6,
                location: None,
            });

            // Check if it looks like a renamed identifier
            if let Some(proof) = old_proof {
                if proof.contains(&name) {
                    suggestions.push(RepairSuggestion {
                        kind: RepairKind::ReplaceTactic,
                        description: format!(
                            "'{}' was used in old proof but is now undefined",
                            name
                        ),
                        fix: format!("Replace '{}' with updated identifier", name),
                        confidence: 0.7,
                        location: None,
                    });
                }
            }
        }
    }

    // Type mismatch
    if error.contains("type mismatch") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::UpdateTypes,
            description: "Type mismatch detected - signatures may have changed".to_string(),
            fix: "Check function signatures and update type annotations".to_string(),
            confidence: 0.6,
            location: None,
        });

        if error.contains("expected") && error.contains("got") {
            suggestions.push(RepairSuggestion {
                kind: RepairKind::ReplaceTactic,
                description: "Use explicit type casting or conversion".to_string(),
                fix: "exact <expr> (with explicit type)".to_string(),
                confidence: 0.5,
                location: None,
            });
        }
    }

    // Unsolved goals
    if error.contains("unsolved goals") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::AddTactics,
            description: "Proof is incomplete - additional tactics needed".to_string(),
            fix: "simp [*]\n-- or --\ndecide".to_string(),
            confidence: 0.5,
            location: None,
        });

        // If omega might help
        if error.contains("Nat")
            || error.contains("Int")
            || error.contains("≤")
            || error.contains("≥")
        {
            suggestions.push(RepairSuggestion {
                kind: RepairKind::AddTactics,
                description: "Goal involves arithmetic - try omega".to_string(),
                fix: "omega".to_string(),
                confidence: 0.7,
                location: None,
            });
        }
    }

    // Unknown tactic
    if error.contains("unknown tactic") {
        let tactic = extract_quoted(error, '\'');
        if let Some(name) = tactic {
            suggestions.push(RepairSuggestion {
                kind: RepairKind::AddImport,
                description: format!("Tactic '{}' requires import", name),
                fix: format!("import Mathlib.Tactic.{}", capitalize(&name)),
                confidence: 0.6,
                location: None,
            });
        }
    }

    suggestions
}

/// TLA+ specific repair suggestions
fn suggest_tlaplus_repairs(_property: &Property, error: &str) -> Vec<RepairSuggestion> {
    let mut suggestions = Vec::new();

    if error.contains("Unknown operator") {
        let op = extract_quoted(error, '"');
        if let Some(name) = op {
            suggestions.push(RepairSuggestion {
                kind: RepairKind::AddImport,
                description: format!("Operator '{}' is not defined", name),
                fix: format!("EXTENDS {} \\* or define {} locally", name, name),
                confidence: 0.6,
                location: None,
            });
        }
    }

    if error.contains("Invariant") && error.contains("violated") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::StrengthenHypothesis,
            description: "Invariant no longer holds - strengthen preconditions".to_string(),
            fix: "Add constraining conditions to Init or Next".to_string(),
            confidence: 0.5,
            location: None,
        });

        suggestions.push(RepairSuggestion {
            kind: RepairKind::WeakenConclusion,
            description: "Consider weakening the invariant".to_string(),
            fix: "Modify invariant to allow new valid states".to_string(),
            confidence: 0.4,
            location: None,
        });
    }

    if error.contains("deadlock") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::AddTactics,
            description: "Deadlock detected - add enabled conditions".to_string(),
            fix: "Add ENABLED Next condition to Spec".to_string(),
            confidence: 0.6,
            location: None,
        });
    }

    suggestions
}

/// Kani specific repair suggestions
fn suggest_kani_repairs(_property: &Property, error: &str) -> Vec<RepairSuggestion> {
    let mut suggestions = Vec::new();

    if error.contains("assertion failed") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::StrengthenHypothesis,
            description: "Assertion failed - add preconditions".to_string(),
            fix: "kani::assume(additional_constraint);".to_string(),
            confidence: 0.6,
            location: None,
        });

        suggestions.push(RepairSuggestion {
            kind: RepairKind::WeakenConclusion,
            description: "Assertion may be too strong".to_string(),
            fix: "Weaken the kani::assert condition".to_string(),
            confidence: 0.4,
            location: None,
        });
    }

    if error.contains("overflow") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::AddTactics,
            description: "Arithmetic overflow - add bounds check".to_string(),
            fix: "kani::assume(x < MAX_VALUE / 2);".to_string(),
            confidence: 0.7,
            location: None,
        });
    }

    if error.contains("undefined behavior") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::Refactor,
            description: "Undefined behavior detected".to_string(),
            fix: "Review code for null dereference, out-of-bounds access, etc.".to_string(),
            confidence: 0.5,
            location: None,
        });
    }

    suggestions
}

/// Alloy specific repair suggestions
fn suggest_alloy_repairs(_property: &Property, error: &str) -> Vec<RepairSuggestion> {
    let mut suggestions = Vec::new();

    if error.contains("Counterexample found") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::StrengthenHypothesis,
            description: "Add constraints to exclude counterexample".to_string(),
            fix: "fact { additional_constraint }".to_string(),
            confidence: 0.6,
            location: None,
        });

        suggestions.push(RepairSuggestion {
            kind: RepairKind::WeakenConclusion,
            description: "Property may be too strong".to_string(),
            fix: "Weaken the assertion to allow valid cases".to_string(),
            confidence: 0.4,
            location: None,
        });
    }

    if error.contains("Unsatisfiable") {
        suggestions.push(RepairSuggestion {
            kind: RepairKind::Refactor,
            description: "Constraints are contradictory".to_string(),
            fix: "Review facts and predicates for inconsistencies".to_string(),
            confidence: 0.7,
            location: None,
        });
    }

    suggestions
}

/// Extract text between delimiters (e.g., 'name' or "name")
fn extract_quoted(text: &str, delim: char) -> Option<String> {
    let start = text.find(delim)?;
    let rest = &text[start + 1..];
    let end = rest.find(delim)?;
    Some(rest[..end].to_string())
}

/// Capitalize first letter
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(c).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Invariant, Theorem};

    fn make_theorem() -> Property {
        Property::Theorem(Theorem {
            name: "test".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        })
    }

    fn make_invariant() -> Property {
        Property::Invariant(Invariant {
            name: "test_inv".to_string(),
            body: dashprove_usl::ast::Expr::Bool(true),
        })
    }

    #[test]
    fn test_detect_lean_unknown_identifier() {
        let error = "unknown identifier 'foo'";
        let breaks = detect_proof_breaks(error, &BackendId::Lean4);

        assert!(breaks.iter().any(|b| b.contains("undefined identifier")));
    }

    #[test]
    fn test_detect_lean_type_mismatch() {
        let error = "type mismatch";
        let breaks = detect_proof_breaks(error, &BackendId::Lean4);

        assert!(breaks.iter().any(|b| b.contains("Type mismatch")));
    }

    #[test]
    fn test_detect_lean_unsolved_goals() {
        let error = "unsolved goals\n⊢ P";
        let breaks = detect_proof_breaks(error, &BackendId::Lean4);

        assert!(breaks.iter().any(|b| b.contains("Incomplete proof")));
    }

    #[test]
    fn test_suggest_lean_import_repair() {
        let prop = make_theorem();
        let error = "unknown identifier 'myLemma'";
        let suggestions = suggest_repairs(&prop, None, error, &BackendId::Lean4);

        assert!(suggestions.iter().any(|s| s.kind == RepairKind::AddImport));
    }

    #[test]
    fn test_suggest_lean_omega_for_arithmetic() {
        let prop = make_theorem();
        let error = "unsolved goals\n⊢ n ≤ m";
        let suggestions = suggest_repairs(&prop, None, error, &BackendId::Lean4);

        assert!(suggestions.iter().any(|s| s.fix.contains("omega")));
    }

    #[test]
    fn test_suggest_tlaplus_invariant_repair() {
        let prop = make_invariant();
        let error = "Invariant Safety violated";
        let suggestions = suggest_repairs(&prop, None, error, &BackendId::TlaPlus);

        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.kind == RepairKind::StrengthenHypothesis));
    }

    #[test]
    fn test_suggest_kani_overflow_repair() {
        let prop = make_theorem();
        let error = "arithmetic overflow";
        let suggestions = suggest_repairs(&prop, None, error, &BackendId::Kani);

        assert!(suggestions.iter().any(|s| s.fix.contains("assume")));
    }

    #[test]
    fn test_suggest_alloy_counterexample_repair() {
        let prop = make_invariant();
        let error = "Counterexample found";
        let suggestions = suggest_repairs(&prop, None, error, &BackendId::Alloy);

        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_extract_quoted() {
        assert_eq!(
            extract_quoted("unknown 'foo' bar", '\''),
            Some("foo".to_string())
        );
        assert_eq!(
            extract_quoted("error \"msg\" here", '"'),
            Some("msg".to_string())
        );
        assert_eq!(extract_quoted("no quotes", '\''), None);
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("A"), "A");
    }

    #[test]
    fn test_suggestions_sorted_by_confidence() {
        let prop = make_theorem();
        let error = "unsolved goals\n⊢ n ≤ m"; // Should suggest omega with higher confidence
        let suggestions = suggest_repairs(&prop, None, error, &BackendId::Lean4);

        // Verify suggestions are sorted by confidence (descending)
        for i in 1..suggestions.len() {
            assert!(suggestions[i - 1].confidence >= suggestions[i].confidence);
        }
    }
}
