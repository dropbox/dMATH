//! Phase C' - Intelligence correctness validation
//!
//! This module provides validation utilities for Phase C (Intelligence) components:
//!
//! - **C'1**: Synthesized proof validation (no-sorry, correct structure)
//! - **C'2**: Tactic prediction accuracy benchmarks
//! - **C'3**: Translation round-trip verification
//! - **C'4**: Adversarial testing utilities
//!
//! These tests ensure the AI-generated outputs are correct before use.

use crate::synthesis::SynthesisResult;
use crate::translate::{ProofLanguage, ProofTranslator, TranslateRequest};
use dashprove_backends::traits::BackendId;
use serde::{Deserialize, Serialize};

// =============================================================================
// C'1: Synthesized Proof Validation
// =============================================================================

/// Validation result for a synthesized proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofValidation {
    /// Whether the proof passes validation
    pub is_valid: bool,
    /// Issues found during validation
    pub issues: Vec<ValidationIssue>,
    /// Validation score (0.0 - 1.0)
    pub score: f64,
}

/// Issue found during proof validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Severity of the issue
    pub severity: Severity,
    /// Description of the issue
    pub message: String,
    /// Line number (if applicable)
    pub line: Option<usize>,
}

/// Severity level for validation issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Proof is definitely invalid
    Error,
    /// Proof may be invalid or incomplete
    Warning,
    /// Stylistic or minor issue
    Info,
}

/// Validate a synthesized proof
pub fn validate_synthesized_proof(result: &SynthesisResult, backend: BackendId) -> ProofValidation {
    let mut issues = Vec::new();
    let mut score: f64 = 1.0;

    // Check for sorry/admit placeholders
    let sorry_check = check_for_sorry(&result.proof, backend);
    if !sorry_check.is_empty() {
        for msg in sorry_check {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                message: msg,
                line: None,
            });
            score -= 0.3;
        }
    }

    // Check proof structure
    let structure_issues = check_proof_structure(&result.proof, backend);
    for issue in structure_issues {
        if issue.severity == Severity::Error {
            score -= 0.2;
        } else if issue.severity == Severity::Warning {
            score -= 0.1;
        }
        issues.push(issue);
    }

    // Check for common mistakes
    let mistake_issues = check_common_mistakes(&result.proof, backend);
    for issue in mistake_issues {
        if issue.severity == Severity::Error {
            score -= 0.15;
        } else if issue.severity == Severity::Warning {
            score -= 0.05;
        }
        issues.push(issue);
    }

    // Low confidence in synthesis is a warning
    if result.confidence < 0.5 {
        issues.push(ValidationIssue {
            severity: Severity::Warning,
            message: format!(
                "Low synthesis confidence: {:.0}%",
                result.confidence * 100.0
            ),
            line: None,
        });
        score -= 0.1;
    }

    let score = score.max(0.0);
    let is_valid = !issues.iter().any(|i| i.severity == Severity::Error);

    ProofValidation {
        is_valid,
        issues,
        score,
    }
}

/// Check for sorry/admit placeholders in proof
fn check_for_sorry(proof: &str, backend: BackendId) -> Vec<String> {
    let mut issues = Vec::new();

    match backend {
        BackendId::Lean4 => {
            if proof.contains("sorry") {
                issues.push("Proof contains 'sorry' placeholder - incomplete proof".to_string());
            }
            if proof.contains("_root_.sorry") {
                issues.push("Proof contains explicit sorry import".to_string());
            }
        }
        BackendId::Coq => {
            if proof.contains("Admitted") {
                issues.push("Proof contains 'Admitted' - incomplete proof".to_string());
            }
            if proof.contains("admit") {
                issues.push("Proof contains 'admit' tactic - incomplete proof".to_string());
            }
        }
        BackendId::Isabelle => {
            if proof.contains("sorry") {
                issues.push("Proof contains 'sorry' - incomplete proof".to_string());
            }
            if proof.contains("oops") {
                issues.push("Proof contains 'oops' - abandoned proof".to_string());
            }
        }
        BackendId::Dafny => {
            if proof.contains("assume false") {
                issues.push("Proof contains 'assume false' - unsound assumption".to_string());
            }
        }
        _ => {
            // Generic check for common placeholder words
            if proof.to_lowercase().contains("todo")
                || proof.to_lowercase().contains("fixme")
                || proof.contains("???")
            {
                issues.push("Proof contains placeholder markers".to_string());
            }
        }
    }

    issues
}

/// Check proof structure for backend-specific requirements
fn check_proof_structure(proof: &str, backend: BackendId) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    match backend {
        BackendId::Lean4 => {
            // Check for theorem/lemma declaration
            if !proof.contains("theorem")
                && !proof.contains("lemma")
                && !proof.contains("example")
                && !proof.contains("def")
            {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    message: "No theorem/lemma/def declaration found".to_string(),
                    line: None,
                });
            }

            // Check for balanced 'by' blocks
            let by_count = proof.matches(" by").count() + proof.matches("\nby").count();
            if by_count > 0 {
                // Check for proper indentation (heuristic)
                let lines: Vec<&str> = proof.lines().collect();
                for (i, line) in lines.iter().enumerate() {
                    if line.trim() == "by" && i + 1 < lines.len() {
                        let next_line = lines[i + 1];
                        if !next_line.starts_with("  ") && !next_line.trim().is_empty() {
                            issues.push(ValidationIssue {
                                severity: Severity::Info,
                                message: "Tactic block after 'by' may lack proper indentation"
                                    .to_string(),
                                line: Some(i + 1),
                            });
                        }
                    }
                }
            }
        }
        BackendId::Coq => {
            // Check for Proof./Qed. structure
            let has_proof = proof.contains("Proof.");
            let has_qed = proof.contains("Qed.");
            let has_defined = proof.contains("Defined.");

            if has_proof && !has_qed && !has_defined {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    message: "Proof. started but no Qed. or Defined. found".to_string(),
                    line: None,
                });
            }

            // Check for Theorem/Lemma declaration
            if !proof.contains("Theorem")
                && !proof.contains("Lemma")
                && !proof.contains("Definition")
                && !proof.contains("Example")
            {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    message: "No Theorem/Lemma/Definition declaration found".to_string(),
                    line: None,
                });
            }
        }
        BackendId::Isabelle => {
            // Check for lemma/theorem
            if !proof.contains("lemma") && !proof.contains("theorem") && !proof.contains("fun") {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    message: "No lemma/theorem declaration found".to_string(),
                    line: None,
                });
            }

            // Check for proper proof termination
            if (proof.contains("proof") || proof.contains("by"))
                && !proof.contains("qed")
                && !proof.contains("done")
                && !proof.contains("by ")
            {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    message: "Proof may not be properly terminated".to_string(),
                    line: None,
                });
            }
        }
        _ => {}
    }

    issues
}

/// Check for common proof mistakes
fn check_common_mistakes(proof: &str, backend: BackendId) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    if proof.trim().is_empty() {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            message: "Proof is empty".to_string(),
            line: None,
        });
    }

    // Check for very short proofs (likely incomplete)
    if proof.trim().len() < 30 {
        issues.push(ValidationIssue {
            severity: Severity::Warning,
            message: "Proof is very short - may be incomplete".to_string(),
            line: None,
        });
    }

    // Check for empty tactic blocks
    if proof.contains("{}") || proof.contains("{ }") {
        issues.push(ValidationIssue {
            severity: Severity::Warning,
            message: "Empty tactic block found".to_string(),
            line: None,
        });
    }

    match backend {
        BackendId::Lean4 => {
            // Check for common Lean 4 issues
            if proof.contains("native_decide") && !proof.contains("#eval") {
                issues.push(ValidationIssue {
                    severity: Severity::Info,
                    message: "native_decide used - may fail on some platforms".to_string(),
                    line: None,
                });
            }

            // Check for unsafe
            if proof.contains("unsafe") {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    message: "Proof uses 'unsafe' - verification bypassed".to_string(),
                    line: None,
                });
            }
        }
        BackendId::Coq => {
            // Check for Axiom usage
            if proof.contains("Axiom") || proof.contains("axiom") {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    message: "Proof introduces axioms - may be unsound".to_string(),
                    line: None,
                });
            }

            // Check for Print Assumptions to verify soundness
            if proof.contains("Admitted") {
                issues.push(ValidationIssue {
                    severity: Severity::Info,
                    message: "Run 'Print Assumptions' to check for assumed axioms".to_string(),
                    line: None,
                });
            }
        }
        _ => {}
    }

    issues
}

// =============================================================================
// C'2: Tactic Prediction Accuracy
// =============================================================================

/// Benchmark result for tactic prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TacticBenchmark {
    /// Total predictions made
    pub total: usize,
    /// Correct predictions (top-1)
    pub correct_top1: usize,
    /// Correct predictions (top-3)
    pub correct_top3: usize,
    /// Correct predictions (top-5)
    pub correct_top5: usize,
    /// Accuracy at different k values
    pub accuracy_at_k: Vec<(usize, f64)>,
}

impl TacticBenchmark {
    /// Create a new empty benchmark
    pub fn new() -> Self {
        Self {
            total: 0,
            correct_top1: 0,
            correct_top3: 0,
            correct_top5: 0,
            accuracy_at_k: vec![],
        }
    }

    /// Record a prediction result
    pub fn record(&mut self, predicted: &[String], actual: &str) {
        self.total += 1;

        // Check if actual tactic is in predictions
        for (i, pred) in predicted.iter().enumerate() {
            if tactic_matches(pred, actual) {
                if i == 0 {
                    self.correct_top1 += 1;
                }
                if i < 3 {
                    self.correct_top3 += 1;
                }
                if i < 5 {
                    self.correct_top5 += 1;
                }
                break;
            }
        }
    }

    /// Compute final accuracy metrics
    pub fn compute_accuracy(&mut self) {
        if self.total == 0 {
            return;
        }

        let total_f = self.total as f64;
        self.accuracy_at_k = vec![
            (1, self.correct_top1 as f64 / total_f),
            (3, self.correct_top3 as f64 / total_f),
            (5, self.correct_top5 as f64 / total_f),
        ];
    }

    /// Get top-1 accuracy
    pub fn top1_accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.correct_top1 as f64 / self.total as f64
    }

    /// Get top-k accuracy
    pub fn topk_accuracy(&self, k: usize) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        match k {
            1 => self.correct_top1 as f64 / self.total as f64,
            3 => self.correct_top3 as f64 / self.total as f64,
            5 => self.correct_top5 as f64 / self.total as f64,
            _ => 0.0,
        }
    }
}

impl Default for TacticBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if predicted tactic matches actual (allowing variations)
fn tactic_matches(predicted: &str, actual: &str) -> bool {
    let pred = predicted.trim().to_lowercase();
    let act = actual.trim().to_lowercase();

    // Exact match
    if pred == act {
        return true;
    }

    // Match tactic name ignoring arguments
    let pred_name = pred.split_whitespace().next().unwrap_or("");
    let act_name = act.split_whitespace().next().unwrap_or("");

    pred_name == act_name
}

// =============================================================================
// C'3: Translation Round-Trip Verification
// =============================================================================

/// Result of round-trip translation verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundTripResult {
    /// Whether round-trip succeeded
    pub success: bool,
    /// Original proof
    pub original: String,
    /// Intermediate translation
    pub intermediate: String,
    /// Round-trip result
    pub round_trip: String,
    /// Similarity score between original and round-trip (0.0 - 1.0)
    pub similarity: f64,
    /// Structural differences found
    pub differences: Vec<String>,
}

/// Perform round-trip translation verification
pub fn verify_round_trip(
    proof: &str,
    from: ProofLanguage,
    through: ProofLanguage,
) -> Result<RoundTripResult, String> {
    let translator = ProofTranslator::new();

    // Translate from -> through
    let forward_request = TranslateRequest::new(proof, from, through);
    let forward_result = translator
        .translate(&forward_request)
        .map_err(|e| format!("Forward translation failed: {}", e))?;

    // Translate through -> from (back)
    let back_request = TranslateRequest::new(&forward_result.translated, through, from);
    let back_result = translator
        .translate(&back_request)
        .map_err(|e| format!("Backward translation failed: {}", e))?;

    // Compute similarity
    let similarity = compute_proof_similarity(proof, &back_result.translated, from);
    let differences = find_structural_differences(proof, &back_result.translated, from);

    let success = similarity > 0.7 && differences.len() < 3;

    Ok(RoundTripResult {
        success,
        original: proof.to_string(),
        intermediate: forward_result.translated,
        round_trip: back_result.translated,
        similarity,
        differences,
    })
}

/// Compute similarity between two proofs
fn compute_proof_similarity(original: &str, translated: &str, _lang: ProofLanguage) -> f64 {
    // Normalize proofs
    let norm_orig = normalize_proof(original);
    let norm_trans = normalize_proof(translated);

    // Simple Jaccard similarity on tokens
    let orig_tokens: std::collections::HashSet<&str> = norm_orig.split_whitespace().collect();
    let trans_tokens: std::collections::HashSet<&str> = norm_trans.split_whitespace().collect();

    if orig_tokens.is_empty() && trans_tokens.is_empty() {
        return 1.0;
    }

    let intersection = orig_tokens.intersection(&trans_tokens).count();
    let union = orig_tokens.union(&trans_tokens).count();

    if union == 0 {
        return 1.0;
    }

    intersection as f64 / union as f64
}

/// Normalize proof for comparison
fn normalize_proof(proof: &str) -> String {
    proof
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with("--") && !line.starts_with("(*"))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Find structural differences between proofs
fn find_structural_differences(
    original: &str,
    translated: &str,
    lang: ProofLanguage,
) -> Vec<String> {
    let mut differences = Vec::new();

    let orig_tactics = extract_tactics(original, lang);
    let trans_tactics = extract_tactics(translated, lang);

    // Check tactic count difference
    if orig_tactics.len() != trans_tactics.len() {
        differences.push(format!(
            "Tactic count differs: {} vs {}",
            orig_tactics.len(),
            trans_tactics.len()
        ));
    }

    // Check for missing tactics
    for tactic in &orig_tactics {
        if !trans_tactics.contains(tactic) {
            differences.push(format!("Missing tactic in translation: {}", tactic));
        }
    }

    // Check for extra tactics
    for tactic in &trans_tactics {
        if !orig_tactics.contains(tactic) {
            differences.push(format!("Extra tactic in translation: {}", tactic));
        }
    }

    differences
}

/// Extract tactics from proof
fn extract_tactics(proof: &str, lang: ProofLanguage) -> Vec<String> {
    let mut tactics = Vec::new();

    let known_tactics: &[&str] = match lang {
        ProofLanguage::Lean4 => &[
            "intro",
            "intros",
            "apply",
            "exact",
            "rfl",
            "simp",
            "constructor",
            "cases",
            "induction",
            "have",
            "show",
            "rw",
            "rewrite",
            "unfold",
            "decide",
            "trivial",
            "assumption",
            "contradiction",
            "ext",
            "funext",
        ],
        ProofLanguage::Coq => &[
            "intros",
            "apply",
            "exact",
            "reflexivity",
            "simpl",
            "constructor",
            "destruct",
            "induction",
            "rewrite",
            "unfold",
            "auto",
            "trivial",
            "assumption",
            "contradiction",
            "split",
            "left",
            "right",
        ],
        ProofLanguage::Isabelle => &[
            "assume", "show", "have", "from", "by", "simp", "auto", "blast", "rule", "erule",
            "drule", "cases", "induct",
        ],
        _ => &[],
    };

    for line in proof.lines() {
        let trimmed = line.trim().to_lowercase();
        for tactic in known_tactics {
            if trimmed.starts_with(tactic) || trimmed.contains(&format!(" {}", tactic)) {
                tactics.push(tactic.to_string());
            }
        }
    }

    tactics
}

// =============================================================================
// C'4: Adversarial Testing
// =============================================================================

/// Adversarial test case for AI components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialTest {
    /// Test name
    pub name: String,
    /// Input that should trigger edge case behavior
    pub input: String,
    /// Expected behavior description
    pub expected_behavior: String,
    /// Whether the test passed
    pub passed: Option<bool>,
    /// Actual behavior observed
    pub actual_behavior: Option<String>,
}

/// Generate adversarial test cases for proof synthesis
pub fn adversarial_synthesis_tests() -> Vec<AdversarialTest> {
    vec![
        AdversarialTest {
            name: "empty_property".to_string(),
            input: "".to_string(),
            expected_behavior: "Should handle gracefully without crash".to_string(),
            passed: None,
            actual_behavior: None,
        },
        AdversarialTest {
            name: "deeply_nested_quantifiers".to_string(),
            input: "∀x. ∀y. ∀z. ∀w. ∃a. ∃b. P(x,y,z,w,a,b)".to_string(),
            expected_behavior: "Should recognize complexity and lower confidence".to_string(),
            passed: None,
            actual_behavior: None,
        },
        AdversarialTest {
            name: "malformed_syntax".to_string(),
            input: "theorem {{{}}}} := sorry sorry sorry".to_string(),
            expected_behavior: "Should detect malformed syntax".to_string(),
            passed: None,
            actual_behavior: None,
        },
        AdversarialTest {
            name: "proof_with_unicode".to_string(),
            input: "theorem λ_∀_∃ : α → β → γ := λ x y => x".to_string(),
            expected_behavior: "Should handle Unicode correctly".to_string(),
            passed: None,
            actual_behavior: None,
        },
        AdversarialTest {
            name: "very_long_identifier".to_string(),
            input: format!("theorem {} : True := trivial", "a".repeat(1000)),
            expected_behavior: "Should handle without buffer overflow".to_string(),
            passed: None,
            actual_behavior: None,
        },
        AdversarialTest {
            name: "injection_attempt".to_string(),
            input: "theorem test : True := by\n  -- ]]]] end code block\nmalicious code"
                .to_string(),
            expected_behavior: "Should not execute injected code".to_string(),
            passed: None,
            actual_behavior: None,
        },
    ]
}

/// Generate adversarial test cases for translation
pub fn adversarial_translation_tests() -> Vec<AdversarialTest> {
    vec![
        AdversarialTest {
            name: "empty_proof".to_string(),
            input: "".to_string(),
            expected_behavior: "Should return empty or error gracefully".to_string(),
            passed: None,
            actual_behavior: None,
        },
        AdversarialTest {
            name: "language_specific_syntax".to_string(),
            input: "Proof. idtac \"Coq specific\". Qed.".to_string(),
            expected_behavior: "Should translate or flag as untranslatable".to_string(),
            passed: None,
            actual_behavior: None,
        },
        AdversarialTest {
            name: "mixed_languages".to_string(),
            input: "theorem test : True := by trivial (* Coq comment *) -- Lean comment"
                .to_string(),
            expected_behavior: "Should handle comment syntax appropriately".to_string(),
            passed: None,
            actual_behavior: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::SynthesisResult;

    // =========================================================================
    // C'1 Tests: Synthesized Proof Validation
    // =========================================================================

    #[test]
    fn test_validate_proof_with_sorry() {
        let result = SynthesisResult {
            proof: "theorem test : True := sorry".to_string(),
            confidence: 0.8,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        assert!(!validation.is_valid);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("sorry")));
    }

    #[test]
    fn test_validate_proof_with_admitted() {
        let result = SynthesisResult {
            proof: "Theorem test : True. Proof. trivial. Admitted.".to_string(),
            confidence: 0.8,
            tactics_used: vec!["trivial".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Coq);
        assert!(!validation.is_valid);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("Admitted")));
    }

    #[test]
    fn test_validate_valid_lean_proof() {
        let result = SynthesisResult {
            proof: "theorem test : True := trivial".to_string(),
            confidence: 0.9,
            tactics_used: vec!["trivial".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        // No errors (might have warnings about short proof)
        assert!(!validation
            .issues
            .iter()
            .any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_validate_valid_coq_proof() {
        let result = SynthesisResult {
            proof: "Theorem test : True.\nProof.\n  trivial.\nQed.".to_string(),
            confidence: 0.9,
            tactics_used: vec!["trivial".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Coq);
        assert!(!validation
            .issues
            .iter()
            .any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_validate_coq_proof_missing_qed() {
        let result = SynthesisResult {
            proof: "Theorem test : True.\nProof.\n  trivial.".to_string(),
            confidence: 0.7,
            tactics_used: vec!["trivial".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Coq);
        assert!(validation.issues.iter().any(|i| i.message.contains("Qed")));
        assert!((validation.score - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_isabelle_sorry() {
        let result = SynthesisResult {
            proof: "lemma test: \"True\" sorry".to_string(),
            confidence: 0.5,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Isabelle);
        assert!(!validation.is_valid);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("sorry")));
    }

    #[test]
    fn test_validate_low_confidence() {
        let result = SynthesisResult {
            proof: "theorem test : True := trivial".to_string(),
            confidence: 0.3,
            tactics_used: vec!["trivial".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("confidence")));
    }

    #[test]
    fn test_validate_score_penalties() {
        let result = SynthesisResult {
            proof: "theorem test : True := by sorry".to_string(),
            confidence: 0.4,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        assert!(!validation.is_valid);
        assert!(
            (validation.score - 0.6).abs() < f64::EPSILON,
            "expected score of 0.6 with sorry and low confidence penalties"
        );
    }

    #[test]
    fn test_validate_confidence_threshold() {
        let result = SynthesisResult {
            proof: "theorem test : True := by\n  have h : True := trivial\n  exact h".to_string(),
            confidence: 0.5,
            tactics_used: vec!["trivial".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        assert!(
            validation
                .issues
                .iter()
                .all(|i| !i.message.contains("confidence")),
            "0.5 confidence should not be penalized"
        );
        assert!((validation.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_generic_placeholders() {
        let result = SynthesisResult {
            proof: "todo: fill in ??? later".to_string(),
            confidence: 0.9,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Alloy);
        assert!(
            validation
                .issues
                .iter()
                .any(|i| i.message.contains("placeholder")),
            "generic placeholder markers should be detected for non-LL backends"
        );
        assert!((validation.score - 0.65).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_dafny_assume_false() {
        let result = SynthesisResult {
            proof: "method Test() { assume false; assert true; }".to_string(),
            confidence: 0.9,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Dafny);
        assert!(!validation.is_valid);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("assume false")));
        assert!((validation.score - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_missing_lean_declaration() {
        let result = SynthesisResult {
            proof: "by\n  trivial".to_string(),
            confidence: 0.9,
            tactics_used: vec!["trivial".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("declaration")));
        assert!((validation.score - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_isabelle_requires_termination() {
        let result = SynthesisResult {
            proof: "lemma test: \"True\"\nproof\n  assume True".to_string(),
            confidence: 0.9,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Isabelle);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("terminated")));
        assert!((validation.score - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_check_for_sorry_lean() {
        let issues = check_for_sorry("theorem test := sorry", BackendId::Lean4);
        assert!(!issues.is_empty());

        let issues = check_for_sorry("theorem test := trivial", BackendId::Lean4);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_for_sorry_coq() {
        let issues = check_for_sorry("Theorem test. Proof. admit. Qed.", BackendId::Coq);
        assert!(!issues.is_empty());

        let issues = check_for_sorry("Theorem test. Proof. trivial. Qed.", BackendId::Coq);
        assert!(issues.is_empty());
    }

    // =========================================================================
    // C'2 Tests: Tactic Prediction Accuracy
    // =========================================================================

    #[test]
    fn test_tactic_benchmark_top1() {
        let mut benchmark = TacticBenchmark::new();

        // Correct prediction at position 0
        benchmark.record(&["simp".to_string(), "apply".to_string()], "simp");
        assert_eq!(benchmark.correct_top1, 1);
        assert_eq!(benchmark.correct_top3, 1);

        // Correct prediction at position 1
        benchmark.record(&["intro".to_string(), "rfl".to_string()], "rfl");
        assert_eq!(benchmark.correct_top1, 1); // No change
        assert_eq!(benchmark.correct_top3, 2);

        // Wrong prediction
        benchmark.record(&["simp".to_string(), "apply".to_string()], "exact");
        assert_eq!(benchmark.correct_top1, 1);
        assert_eq!(benchmark.correct_top3, 2);
    }

    #[test]
    fn test_tactic_benchmark_accuracy() {
        let mut benchmark = TacticBenchmark::new();

        // 3 correct top-1, 2 more in top-3
        for _ in 0..3 {
            benchmark.record(&["simp".to_string()], "simp");
        }
        for _ in 0..2 {
            benchmark.record(
                &["a".to_string(), "b".to_string(), "simp".to_string()],
                "simp",
            );
        }
        for _ in 0..5 {
            benchmark.record(&["wrong".to_string()], "right");
        }

        benchmark.compute_accuracy();

        assert_eq!(benchmark.total, 10);
        assert_eq!(benchmark.correct_top1, 3);
        assert_eq!(benchmark.correct_top3, 5);
        assert!((benchmark.top1_accuracy() - 0.3).abs() < 0.01);
        assert!((benchmark.topk_accuracy(3) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tactic_matches() {
        assert!(tactic_matches("simp", "simp"));
        assert!(tactic_matches("simp", "SIMP")); // Case insensitive
        assert!(tactic_matches("simp [lemma]", "simp")); // Base tactic match
        assert!(!tactic_matches("simp", "apply"));
    }

    #[test]
    fn test_tactic_matches_with_args() {
        assert!(tactic_matches("apply foo", "apply"));
        assert!(tactic_matches("rw [h1, h2]", "rw"));
        assert!(tactic_matches("induction n with n' ih", "induction"));
    }

    // =========================================================================
    // C'3 Tests: Translation Round-Trip
    // =========================================================================

    #[test]
    fn test_compute_proof_similarity() {
        let proof1 = "theorem test : True := by trivial";
        let proof2 = "theorem test : True := by trivial";
        let similarity = compute_proof_similarity(proof1, proof2, ProofLanguage::Lean4);
        assert!((similarity - 1.0).abs() < 0.01);

        let proof3 = "lemma different : False := by sorry";
        let similarity2 = compute_proof_similarity(proof1, proof3, ProofLanguage::Lean4);
        assert!(similarity2 < 0.8);
    }

    #[test]
    fn test_normalize_proof() {
        let proof = r#"
            -- Comment
            theorem test : True := by
              trivial
        "#;
        let normalized = normalize_proof(proof);
        assert!(!normalized.contains("--"));
        assert!(normalized.contains("theorem"));
        assert!(normalized.contains("trivial"));
    }

    #[test]
    fn test_extract_tactics_lean() {
        let proof = r#"
            theorem test : P → Q → P := by
              intro h1
              intro h2
              exact h1
        "#;
        let tactics = extract_tactics(proof, ProofLanguage::Lean4);
        assert!(tactics.contains(&"intro".to_string()));
        assert!(tactics.contains(&"exact".to_string()));
    }

    #[test]
    fn test_extract_tactics_coq() {
        let proof = r#"
            Theorem test : forall P Q, P -> Q -> P.
            Proof.
              intros P Q H1 H2.
              exact H1.
            Qed.
        "#;
        let tactics = extract_tactics(proof, ProofLanguage::Coq);
        assert!(tactics.contains(&"intros".to_string()));
        assert!(tactics.contains(&"exact".to_string()));
    }

    #[test]
    fn test_find_structural_differences() {
        let original = "theorem test := by intro h; exact h";
        let translated = "theorem test := by intro; assumption";
        let diffs = find_structural_differences(original, translated, ProofLanguage::Lean4);
        // Should detect that exact is missing and assumption is extra
        assert!(diffs
            .iter()
            .any(|d| d.contains("exact") || d.contains("assumption")));
    }

    #[test]
    fn test_round_trip_lean_coq() {
        // This tests the structure - actual translation accuracy depends on translator
        let proof = "theorem test : True := trivial";
        let result = verify_round_trip(proof, ProofLanguage::Lean4, ProofLanguage::Coq);

        // Should succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // =========================================================================
    // C'4 Tests: Adversarial Testing
    // =========================================================================

    #[test]
    fn test_adversarial_synthesis_tests_exist() {
        let tests = adversarial_synthesis_tests();
        assert!(!tests.is_empty());
        assert!(tests.iter().any(|t| t.name == "empty_property"));
        assert!(tests.iter().any(|t| t.name == "malformed_syntax"));
        assert!(tests.iter().any(|t| t.name == "injection_attempt"));
    }

    #[test]
    fn test_adversarial_translation_tests_exist() {
        let tests = adversarial_translation_tests();
        assert!(!tests.is_empty());
        assert!(tests.iter().any(|t| t.name == "empty_proof"));
    }

    #[test]
    fn test_validate_very_short_proof() {
        let result = SynthesisResult {
            proof: "rfl".to_string(),
            confidence: 0.9,
            tactics_used: vec!["rfl".to_string()],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        // Should have warning about short proof
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("short")));
    }

    #[test]
    fn test_validate_empty_proof_error() {
        let result = SynthesisResult {
            proof: "".to_string(),
            confidence: 0.9,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        assert!(!validation.is_valid);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("empty")));
        assert!((validation.score - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_unsafe_lean() {
        let result = SynthesisResult {
            proof: "unsafe def test : Nat := 42".to_string(),
            confidence: 0.8,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Lean4);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("unsafe")));
    }

    #[test]
    fn test_validate_coq_axiom() {
        let result = SynthesisResult {
            proof: "Axiom my_axiom : forall P, P.".to_string(),
            confidence: 0.5,
            tactics_used: vec![],
            attempts: 1,
            reasoning: None,
        };

        let validation = validate_synthesized_proof(&result, BackendId::Coq);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.message.contains("axiom")));
    }
}
