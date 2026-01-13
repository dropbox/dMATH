//! Tactic suggestion system
//!
//! Provides suggestions for proof tactics based on:
//! - Property structure (compiler analysis)
//! - Learning from past proofs
//! - Backend-specific heuristics

use crate::Confidence;
use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::{Expr, Property};
use serde::{Deserialize, Serialize};

/// Source of a tactic suggestion
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionSource {
    /// From compiler/structural analysis
    Compiler,
    /// From learning system (past successful proofs)
    Learning,
    /// From heuristic rules
    Heuristic,
}

/// A suggested tactic with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TacticSuggestion {
    /// The tactic name/code
    pub tactic: String,
    /// Confidence level
    pub confidence: Confidence,
    /// Where this suggestion came from
    pub source: SuggestionSource,
    /// Explanation of why this tactic might work
    pub rationale: String,
}

/// Get compiler-based suggestions for a property
pub fn compiler_suggestions(property: &Property, backend: &BackendId) -> Vec<TacticSuggestion> {
    match backend {
        BackendId::Lean4 => lean_suggestions(property),
        BackendId::TlaPlus => tlaplus_suggestions(property),
        BackendId::Kani => kani_suggestions(property),
        BackendId::Alloy => alloy_suggestions(property),
        _ => vec![],
    }
}

/// Suggest tactics for a property (main entry point)
pub fn suggest_tactics(
    property: &Property,
    backend: &BackendId,
    learning: Option<&dashprove_learning::ProofLearningSystem>,
) -> Vec<TacticSuggestion> {
    let mut suggestions = Vec::new();

    // Learning-based suggestions first (higher confidence)
    if let Some(ls) = learning {
        let learned = ls.suggest_tactics(property, 5);
        for (tactic, score) in learned {
            suggestions.push(TacticSuggestion {
                tactic,
                confidence: Confidence::from_score(score),
                source: SuggestionSource::Learning,
                rationale: "Based on similar successful proofs".to_string(),
            });
        }
    }

    // Compiler suggestions
    let compiler = compiler_suggestions(property, backend);
    for cs in compiler {
        if !suggestions.iter().any(|s| s.tactic == cs.tactic) {
            suggestions.push(cs);
        }
    }

    suggestions
}

/// LEAN 4 specific tactic suggestions
fn lean_suggestions(property: &Property) -> Vec<TacticSuggestion> {
    let mut suggestions = Vec::new();

    let expr = match property {
        Property::Theorem(t) => Some(&t.body),
        Property::Invariant(i) => Some(&i.body),
        Property::Refinement(r) => Some(&r.abstraction),
        _ => None,
    };

    if let Some(expr) = expr {
        suggestions.extend(suggest_for_expr(expr));
    }

    // Always suggest basic tactics as fallbacks
    if suggestions.is_empty() {
        suggestions.push(TacticSuggestion {
            tactic: "decide".to_string(),
            confidence: Confidence::Low,
            source: SuggestionSource::Heuristic,
            rationale: "Default tactic for decidable propositions".to_string(),
        });
    }

    suggestions
}

/// Analyze expression structure to suggest tactics
fn suggest_for_expr(expr: &Expr) -> Vec<TacticSuggestion> {
    let mut suggestions = Vec::new();

    match expr {
        Expr::ForAll { body, .. } => {
            suggestions.push(TacticSuggestion {
                tactic: "intro".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Introduce universally quantified variable".to_string(),
            });
            suggestions.extend(suggest_for_expr(body));
        }
        Expr::Exists { .. } => {
            suggestions.push(TacticSuggestion {
                tactic: "use".to_string(),
                confidence: Confidence::Medium,
                source: SuggestionSource::Compiler,
                rationale: "Provide witness for existential".to_string(),
            });
        }
        Expr::Implies(_, rhs) => {
            suggestions.push(TacticSuggestion {
                tactic: "intro".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Introduce hypothesis of implication".to_string(),
            });
            suggestions.extend(suggest_for_expr(rhs));
        }
        Expr::And(lhs, rhs) => {
            suggestions.push(TacticSuggestion {
                tactic: "constructor".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Split conjunction into two goals".to_string(),
            });
            suggestions.extend(suggest_for_expr(lhs));
            suggestions.extend(suggest_for_expr(rhs));
        }
        Expr::Or(_, _) => {
            suggestions.push(TacticSuggestion {
                tactic: "left".to_string(),
                confidence: Confidence::Low,
                source: SuggestionSource::Compiler,
                rationale: "Prove left side of disjunction".to_string(),
            });
            suggestions.push(TacticSuggestion {
                tactic: "right".to_string(),
                confidence: Confidence::Low,
                source: SuggestionSource::Compiler,
                rationale: "Prove right side of disjunction".to_string(),
            });
        }
        Expr::Not(inner) => {
            suggestions.push(TacticSuggestion {
                tactic: "intro".to_string(),
                confidence: Confidence::Medium,
                source: SuggestionSource::Compiler,
                rationale: "Introduce hypothesis to derive contradiction".to_string(),
            });
            suggestions.extend(suggest_for_expr(inner));
        }
        Expr::Compare(_, _, _) => {
            suggestions.push(TacticSuggestion {
                tactic: "omega".to_string(),
                confidence: Confidence::Medium,
                source: SuggestionSource::Compiler,
                rationale: "Solve linear arithmetic".to_string(),
            });
            suggestions.push(TacticSuggestion {
                tactic: "decide".to_string(),
                confidence: Confidence::Low,
                source: SuggestionSource::Compiler,
                rationale: "Decide equality for decidable types".to_string(),
            });
        }
        Expr::Binary(_, _, _) => {
            suggestions.push(TacticSuggestion {
                tactic: "ring".to_string(),
                confidence: Confidence::Medium,
                source: SuggestionSource::Compiler,
                rationale: "Solve ring arithmetic".to_string(),
            });
            suggestions.push(TacticSuggestion {
                tactic: "linarith".to_string(),
                confidence: Confidence::Medium,
                source: SuggestionSource::Compiler,
                rationale: "Solve linear arithmetic with hypotheses".to_string(),
            });
        }
        Expr::Bool(true) => {
            suggestions.push(TacticSuggestion {
                tactic: "trivial".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Goal is trivially true".to_string(),
            });
        }
        Expr::Bool(false) => {
            suggestions.push(TacticSuggestion {
                tactic: "contradiction".to_string(),
                confidence: Confidence::Medium,
                source: SuggestionSource::Compiler,
                rationale: "Derive contradiction from hypotheses".to_string(),
            });
        }
        Expr::App(name, _) => {
            // Suggest simp for function applications
            suggestions.push(TacticSuggestion {
                tactic: format!("simp [{}]", name),
                confidence: Confidence::Low,
                source: SuggestionSource::Compiler,
                rationale: format!("Simplify using definition of {}", name),
            });
        }
        Expr::MethodCall { method, .. } => {
            suggestions.push(TacticSuggestion {
                tactic: format!("simp [{}]", method),
                confidence: Confidence::Low,
                source: SuggestionSource::Compiler,
                rationale: format!("Simplify using definition of {}", method),
            });
        }
        _ => {}
    }

    // Deduplicate by tactic name, keeping highest confidence
    let mut seen = std::collections::HashSet::new();
    suggestions.retain(|s| seen.insert(s.tactic.clone()));

    suggestions
}

/// TLA+ specific suggestions
fn tlaplus_suggestions(property: &Property) -> Vec<TacticSuggestion> {
    let mut suggestions = Vec::new();

    if let Property::Temporal(t) = property {
        match &t.body {
            dashprove_usl::ast::TemporalExpr::Always(_) => {
                suggestions.push(TacticSuggestion {
                    tactic: "InvariantProof".to_string(),
                    confidence: Confidence::Medium,
                    source: SuggestionSource::Compiler,
                    rationale: "Prove invariant by induction on steps".to_string(),
                });
            }
            dashprove_usl::ast::TemporalExpr::Eventually(_) => {
                suggestions.push(TacticSuggestion {
                    tactic: "LivenessProof".to_string(),
                    confidence: Confidence::Medium,
                    source: SuggestionSource::Compiler,
                    rationale: "Prove liveness using fairness assumptions".to_string(),
                });
            }
            dashprove_usl::ast::TemporalExpr::LeadsTo(_, _) => {
                suggestions.push(TacticSuggestion {
                    tactic: "LeadsToProof".to_string(),
                    confidence: Confidence::Medium,
                    source: SuggestionSource::Compiler,
                    rationale: "Prove leads-to using ranking function".to_string(),
                });
            }
            _ => {}
        }
    }

    suggestions
}

/// Kani specific suggestions
fn kani_suggestions(property: &Property) -> Vec<TacticSuggestion> {
    let mut suggestions = Vec::new();

    if let Property::Contract(c) = property {
        if !c.requires.is_empty() {
            suggestions.push(TacticSuggestion {
                tactic: "kani::assume".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Encode preconditions as assumptions".to_string(),
            });
        }
        if !c.ensures.is_empty() {
            suggestions.push(TacticSuggestion {
                tactic: "kani::assert".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Check postconditions with assertions".to_string(),
            });
        }
    }

    suggestions
}

/// Alloy specific suggestions
fn alloy_suggestions(property: &Property) -> Vec<TacticSuggestion> {
    let mut suggestions = Vec::new();

    match property {
        Property::Invariant(_) => {
            suggestions.push(TacticSuggestion {
                tactic: "check".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Run bounded model checking".to_string(),
            });
        }
        Property::Theorem(_) => {
            suggestions.push(TacticSuggestion {
                tactic: "assert".to_string(),
                confidence: Confidence::High,
                source: SuggestionSource::Compiler,
                rationale: "Check assertion with counterexample search".to_string(),
            });
        }
        _ => {}
    }

    suggestions
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Invariant, Temporal, TemporalExpr, Theorem, Type};

    fn make_forall_theorem() -> Property {
        Property::Theorem(Theorem {
            name: "forall_test".to_string(),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: Some(Type::Named("Nat".to_string())),
                body: Box::new(Expr::Compare(
                    Box::new(Expr::Var("x".to_string())),
                    dashprove_usl::ast::ComparisonOp::Ge,
                    Box::new(Expr::Int(0)),
                )),
            },
        })
    }

    fn make_implies_theorem() -> Property {
        Property::Theorem(Theorem {
            name: "implies_test".to_string(),
            body: Expr::Implies(
                Box::new(Expr::Var("P".to_string())),
                Box::new(Expr::Var("P".to_string())),
            ),
        })
    }

    fn make_temporal_always() -> Property {
        Property::Temporal(Temporal {
            name: "always_test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        })
    }

    #[test]
    fn test_forall_suggests_intro() {
        let prop = make_forall_theorem();
        let suggestions = compiler_suggestions(&prop, &BackendId::Lean4);

        assert!(suggestions.iter().any(|s| s.tactic == "intro"));
    }

    #[test]
    fn test_implies_suggests_intro() {
        let prop = make_implies_theorem();
        let suggestions = compiler_suggestions(&prop, &BackendId::Lean4);

        assert!(suggestions.iter().any(|s| s.tactic == "intro"));
    }

    #[test]
    fn test_temporal_always_suggests_invariant() {
        let prop = make_temporal_always();
        let suggestions = compiler_suggestions(&prop, &BackendId::TlaPlus);

        assert!(suggestions.iter().any(|s| s.tactic == "InvariantProof"));
    }

    #[test]
    fn test_comparison_suggests_omega() {
        let prop = Property::Theorem(Theorem {
            name: "compare_test".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Int(1)),
                dashprove_usl::ast::ComparisonOp::Lt,
                Box::new(Expr::Int(2)),
            ),
        });
        let suggestions = compiler_suggestions(&prop, &BackendId::Lean4);

        assert!(suggestions.iter().any(|s| s.tactic == "omega"));
    }

    #[test]
    fn test_suggestion_deduplication() {
        let prop = Property::Theorem(Theorem {
            name: "nested_forall".to_string(),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: None,
                body: Box::new(Expr::ForAll {
                    var: "y".to_string(),
                    ty: None,
                    body: Box::new(Expr::Bool(true)),
                }),
            },
        });
        let suggestions = compiler_suggestions(&prop, &BackendId::Lean4);

        // "intro" should only appear once
        let intro_count = suggestions.iter().filter(|s| s.tactic == "intro").count();
        assert_eq!(intro_count, 1);
    }

    #[test]
    fn test_invariant_suggests_check_for_alloy() {
        let prop = Property::Invariant(Invariant {
            name: "test_inv".to_string(),
            body: Expr::Bool(true),
        });
        let suggestions = compiler_suggestions(&prop, &BackendId::Alloy);

        assert!(suggestions.iter().any(|s| s.tactic == "check"));
    }
}
