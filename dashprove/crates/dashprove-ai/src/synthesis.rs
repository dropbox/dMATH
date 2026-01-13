//! AI-powered proof synthesis
//!
//! This module provides LLM-based proof synthesis capabilities:
//!
//! - Generate complete proofs from specifications
//! - Elaborate proof sketches into full proofs
//! - Fill in proof holes with appropriate tactics
//!
//! ## Usage
//!
//! ```rust,ignore
//! use dashprove_ai::synthesis::{ProofSynthesizer, SynthesisRequest};
//!
//! let synthesizer = ProofSynthesizer::new(llm_client);
//! let proof = synthesizer.synthesize(&property).await?;
//! ```

use crate::llm::{LlmClient, LlmError, LlmMessage};
use crate::sketch::ProofSketch;
use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};

/// Request for proof synthesis
#[derive(Debug, Clone)]
pub struct SynthesisRequest<'a> {
    /// Property to prove
    pub property: &'a Property,
    /// Target backend
    pub backend: BackendId,
    /// Optional sketch to elaborate
    pub sketch: Option<&'a ProofSketch>,
    /// Optional hints from user
    pub hints: Vec<String>,
    /// Maximum attempts
    pub max_attempts: u32,
}

impl<'a> SynthesisRequest<'a> {
    /// Create a new synthesis request
    pub fn new(property: &'a Property, backend: BackendId) -> Self {
        Self {
            property,
            backend,
            sketch: None,
            hints: vec![],
            max_attempts: 3,
        }
    }

    /// Add a sketch to elaborate
    pub fn with_sketch(mut self, sketch: &'a ProofSketch) -> Self {
        self.sketch = Some(sketch);
        self
    }

    /// Add hints
    pub fn with_hints(mut self, hints: Vec<String>) -> Self {
        self.hints = hints;
        self
    }
}

/// Result of proof synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    /// Generated proof code
    pub proof: String,
    /// Confidence in the proof (0.0 - 1.0)
    pub confidence: f64,
    /// Tactics used in the proof
    pub tactics_used: Vec<String>,
    /// Number of attempts made
    pub attempts: u32,
    /// Reasoning chain from the LLM
    pub reasoning: Option<String>,
}

/// Error during synthesis
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    /// LLM error
    #[error("LLM error: {0}")]
    LlmError(#[from] LlmError),

    /// Failed to generate valid proof
    #[error("Failed to generate valid proof after {attempts} attempts")]
    SynthesisFailed { attempts: u32 },

    /// Parse error in generated proof
    #[error("Failed to parse generated proof: {0}")]
    ParseError(String),

    /// Unsupported backend
    #[error("Proof synthesis not supported for backend: {0:?}")]
    UnsupportedBackend(BackendId),
}

/// AI-powered proof synthesizer
pub struct ProofSynthesizer {
    client: Box<dyn LlmClient>,
}

impl ProofSynthesizer {
    /// Create a new synthesizer with an LLM client
    pub fn new(client: Box<dyn LlmClient>) -> Self {
        Self { client }
    }

    /// Synthesize a proof for a property
    pub async fn synthesize(
        &self,
        request: &SynthesisRequest<'_>,
    ) -> Result<SynthesisResult, SynthesisError> {
        let system_prompt = self.build_system_prompt(request.backend);
        let user_prompt = self.build_user_prompt(request);

        let mut attempts = 0;
        let mut last_response = None;

        while attempts < request.max_attempts {
            attempts += 1;

            let messages = vec![
                LlmMessage::system(&system_prompt),
                LlmMessage::user(&user_prompt),
            ];

            let response = self.client.complete_messages(&messages).await?;
            last_response = Some(response.content.clone());

            // Try to extract the proof from the response
            if let Some(proof) = self.extract_proof(&response.content, request.backend) {
                let tactics = self.extract_tactics(&proof, request.backend);
                let confidence = self.estimate_confidence(&proof, &tactics);

                return Ok(SynthesisResult {
                    proof,
                    confidence,
                    tactics_used: tactics,
                    attempts,
                    reasoning: Some(response.content),
                });
            }
        }

        // If we exhausted attempts, return the last response as a low-confidence result
        if let Some(content) = last_response {
            Ok(SynthesisResult {
                proof: content.clone(),
                confidence: 0.1,
                tactics_used: vec![],
                attempts,
                reasoning: Some(content),
            })
        } else {
            Err(SynthesisError::SynthesisFailed { attempts })
        }
    }

    /// Elaborate a proof sketch into a complete proof
    pub async fn elaborate_sketch(
        &self,
        sketch: &ProofSketch,
    ) -> Result<SynthesisResult, SynthesisError> {
        let system_prompt = self.build_system_prompt(sketch.target_backend);
        let sketch_code = sketch.to_lean();

        let user_prompt = format!(
            r#"Elaborate the following proof sketch into a complete proof.

The sketch contains `sorry` placeholders that need to be filled with actual tactics.
Replace each `sorry` with appropriate tactics to complete the proof.

Proof sketch:
```
{}
```

{}

Provide the complete proof with all `sorry` placeholders replaced with working tactics.
Enclose the final proof in a code block."#,
            sketch_code,
            if !sketch.hints_used.is_empty() {
                format!("Hints: {}", sketch.hints_used.join(", "))
            } else {
                String::new()
            }
        );

        let messages = vec![
            LlmMessage::system(&system_prompt),
            LlmMessage::user(&user_prompt),
        ];

        let response = self.client.complete_messages(&messages).await?;

        if let Some(proof) = self.extract_proof(&response.content, sketch.target_backend) {
            let tactics = self.extract_tactics(&proof, sketch.target_backend);
            let confidence = self.estimate_confidence(&proof, &tactics);

            Ok(SynthesisResult {
                proof,
                confidence,
                tactics_used: tactics,
                attempts: 1,
                reasoning: Some(response.content),
            })
        } else {
            Ok(SynthesisResult {
                proof: response.content.clone(),
                confidence: 0.2,
                tactics_used: vec![],
                attempts: 1,
                reasoning: Some(response.content),
            })
        }
    }

    /// Build system prompt for the given backend
    fn build_system_prompt(&self, backend: BackendId) -> String {
        match backend {
            BackendId::Lean4 => LEAN4_SYSTEM_PROMPT.to_string(),
            BackendId::Coq => COQ_SYSTEM_PROMPT.to_string(),
            BackendId::Isabelle => ISABELLE_SYSTEM_PROMPT.to_string(),
            BackendId::Dafny => DAFNY_SYSTEM_PROMPT.to_string(),
            BackendId::Kani => KANI_SYSTEM_PROMPT.to_string(),
            BackendId::TlaPlus => TLAPLUS_SYSTEM_PROMPT.to_string(),
            _ => GENERIC_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Build user prompt from synthesis request
    fn build_user_prompt(&self, request: &SynthesisRequest<'_>) -> String {
        let property_str = format!("{:?}", request.property); // TODO: Pretty print

        let mut prompt = format!(
            "Generate a proof for the following property:\n\n{}\n",
            property_str
        );

        if let Some(sketch) = request.sketch {
            prompt.push_str(&format!(
                "\nUse this proof sketch as a guide:\n```\n{}\n```\n",
                sketch.to_lean()
            ));
        }

        if !request.hints.is_empty() {
            prompt.push_str(&format!("\nHints: {}\n", request.hints.join(", ")));
        }

        prompt.push_str("\nProvide the complete proof enclosed in a code block.");

        prompt
    }

    /// Extract proof code from LLM response
    fn extract_proof(&self, response: &str, backend: BackendId) -> Option<String> {
        // Look for code blocks with appropriate language tags
        let lang_tag = match backend {
            BackendId::Lean4 => "lean",
            BackendId::Coq => "coq",
            BackendId::Isabelle => "isabelle",
            BackendId::Dafny => "dafny",
            BackendId::Kani => "rust",
            BackendId::TlaPlus => "tla",
            _ => "",
        };

        // Try language-specific code block first
        if !lang_tag.is_empty() {
            let pattern = format!("```{}", lang_tag);
            if let Some(start) = response.find(&pattern) {
                if let Some(end) = response[start + pattern.len()..].find("```") {
                    let code = response[start + pattern.len()..start + pattern.len() + end].trim();
                    if !code.is_empty() {
                        return Some(code.to_string());
                    }
                }
            }
        }

        // Fall back to generic code block
        if let Some(start) = response.find("```") {
            let after_backticks = &response[start + 3..];
            // Skip language identifier if present
            let code_start = after_backticks.find('\n').map(|i| i + 1).unwrap_or(0);
            if let Some(end) = after_backticks[code_start..].find("```") {
                let code = after_backticks[code_start..code_start + end].trim();
                if !code.is_empty() {
                    return Some(code.to_string());
                }
            }
        }

        None
    }

    /// Extract tactics used in a proof
    fn extract_tactics(&self, proof: &str, backend: BackendId) -> Vec<String> {
        match backend {
            BackendId::Lean4 => extract_lean_tactics(proof),
            BackendId::Coq => extract_coq_tactics(proof),
            _ => vec![],
        }
    }

    /// Estimate confidence based on proof structure
    fn estimate_confidence(&self, proof: &str, tactics: &[String]) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // Boost if no sorry/admit
        if !proof.contains("sorry") && !proof.contains("admit") && !proof.contains("Admitted") {
            confidence += 0.2;
        }

        // Boost if uses standard tactics
        let standard_tactics = [
            "rfl",
            "simp",
            "intro",
            "apply",
            "exact",
            "constructor",
            "trivial",
        ];
        let standard_count = tactics
            .iter()
            .filter(|t| standard_tactics.iter().any(|st| t.contains(st)))
            .count();

        if standard_count > 0 {
            confidence += 0.1 * (standard_count as f64 / tactics.len().max(1) as f64);
        }

        // Penalize very short proofs (might be incomplete)
        if proof.len() < 20 {
            confidence -= 0.2;
        }

        confidence.clamp(0.0, 1.0)
    }
}

/// Extract Lean 4 tactics from proof
fn extract_lean_tactics(proof: &str) -> Vec<String> {
    let mut tactics = Vec::new();
    let known_tactics = [
        "intro",
        "apply",
        "exact",
        "rfl",
        "simp",
        "constructor",
        "cases",
        "induction",
        "have",
        "show",
        "calc",
        "rw",
        "rewrite",
        "unfold",
        "decide",
        "trivial",
        "omega",
        "ring",
        "norm_num",
        "linarith",
        "ext",
        "funext",
        "congr",
        "assumption",
        "contradiction",
        "exfalso",
    ];

    for line in proof.lines() {
        // Strip common prefixes (bullet points, whitespace, etc.)
        let trimmed = line.trim();
        let cleaned = trimmed.trim_start_matches(|c: char| {
            c == '·' || c == '-' || c == '+' || c == '*' || c.is_whitespace()
        });

        for tactic in &known_tactics {
            // Check if line starts with tactic or contains it with word boundaries
            let matches = cleaned.starts_with(tactic)
                || trimmed.contains(&format!(" {} ", tactic))
                || trimmed.contains(&format!(" {}", tactic)) && trimmed.ends_with(tactic);
            if matches && !tactics.contains(&tactic.to_string()) {
                tactics.push(tactic.to_string());
            }
        }
    }

    tactics
}

/// Extract Coq tactics from proof
fn extract_coq_tactics(proof: &str) -> Vec<String> {
    let mut tactics = Vec::new();
    let known_tactics = [
        "intros",
        "apply",
        "exact",
        "reflexivity",
        "simpl",
        "constructor",
        "destruct",
        "induction",
        "assert",
        "rewrite",
        "unfold",
        "auto",
        "trivial",
        "omega",
        "ring",
        "lia",
        "discriminate",
        "injection",
        "assumption",
        "contradiction",
        "exfalso",
        "split",
        "left",
        "right",
    ];

    for line in proof.lines() {
        let trimmed = line.trim();
        for tactic in &known_tactics {
            let matches = trimmed.starts_with(tactic) || trimmed.contains(&format!(" {} ", tactic));
            if matches && !tactics.contains(&tactic.to_string()) {
                tactics.push(tactic.to_string());
            }
        }
    }

    tactics
}

// System prompts for different backends

const LEAN4_SYSTEM_PROMPT: &str = r#"You are an expert Lean 4 proof assistant. You generate formal proofs in Lean 4 syntax.

Key Lean 4 tactics:
- intro/intros: Introduce hypotheses
- apply: Apply a theorem or hypothesis
- exact: Provide an exact proof term
- rfl: Reflexivity (equality proofs)
- simp: Simplification
- constructor: Build inductive types
- cases/induction: Case analysis and induction
- rw/rewrite: Rewriting with equalities
- decide: Decidable propositions
- omega: Linear arithmetic
- ring: Ring equations

Guidelines:
1. Use structured proofs with clear tactic applications
2. Prefer simple, readable tactics over complex proof terms
3. Use simp [lemmas] to provide simplification hints
4. For arithmetic, try omega or ring
5. Never leave sorry in the final proof

Always enclose your proof in a ```lean code block."#;

const COQ_SYSTEM_PROMPT: &str = r#"You are an expert Coq proof assistant. You generate formal proofs in Coq/Gallina syntax.

Key Coq tactics:
- intros: Introduce hypotheses
- apply: Apply a theorem
- exact: Provide exact proof term
- reflexivity: Reflexive equality
- simpl: Simplification
- constructor: Build inductive types
- destruct/induction: Case analysis and induction
- rewrite: Rewriting
- auto: Automatic proof search
- omega/lia: Linear arithmetic

Guidelines:
1. Start proofs with Proof. and end with Qed.
2. Use bullet points (-, +, *) for subgoals
3. Prefer simple tactics over complex terms
4. Never leave Admitted in the final proof

Always enclose your proof in a ```coq code block."#;

const ISABELLE_SYSTEM_PROMPT: &str = r#"You are an expert Isabelle proof assistant. You generate proofs in Isabelle/HOL.

Key Isabelle methods:
- rule: Apply inference rules
- simp: Simplification
- auto: Automatic proof search
- blast: Tableau prover
- induct: Induction
- cases: Case analysis
- sledgehammer: External ATPs

Guidelines:
1. Use structured proofs with have/show
2. Prefer apply-style or Isar style
3. Use sledgehammer hints when available
4. End with done or qed

Always enclose your proof in a ```isabelle code block."#;

const DAFNY_SYSTEM_PROMPT: &str = r#"You are an expert Dafny verification engineer. You write Dafny code with specifications and proofs.

Key Dafny features:
- requires/ensures: Pre/postconditions
- invariant: Loop invariants
- assert: Intermediate assertions
- calc: Calculation proofs
- forall/exists: Quantifiers
- decreases: Termination metrics

Guidelines:
1. Write clear requires and ensures clauses
2. Add loop invariants for all loops
3. Use assert for intermediate steps
4. Use calc blocks for equation chains

Always enclose your code in a ```dafny code block."#;

const KANI_SYSTEM_PROMPT: &str = r#"You are an expert Rust verification engineer using Kani for formal verification.

Key Kani features:
- #[kani::proof]: Mark proof harness
- kani::any(): Symbolic values
- kani::assume(): Preconditions
- kani::assert(): Assertions to verify
- #[kani::unwind(N)]: Loop bounds
- #[kani::stub]: Function stubs

Guidelines:
1. Use kani::any() for symbolic inputs
2. Add kani::assume() for preconditions
3. Use kani::assert() for properties
4. Bound loops with #[kani::unwind]
5. Handle panic cases explicitly

Always enclose your code in a ```rust code block."#;

const TLAPLUS_SYSTEM_PROMPT: &str = r#"You are an expert TLA+ specification and verification engineer.

Key TLA+ features:
- VARIABLE: State variables
- Init: Initial state predicate
- Next: Transition relation
- Spec: Complete specification (Init /\ [][Next]_vars)
- THEOREM: Properties to verify
- TypeInvariant: Type correctness
- SafetyProperty: Safety invariants
- LivenessProperty: Liveness with WF/SF

Guidelines:
1. Define clear state variables
2. Write Init and Next predicates
3. Express safety as invariants
4. Use temporal operators for liveness
5. Add type invariants

Always enclose your specification in a ```tla code block."#;

const GENERIC_SYSTEM_PROMPT: &str = r#"You are an expert formal verification assistant. Generate proofs and specifications for formal verification tools.

Guidelines:
1. Use clear, structured proof notation
2. Explain your proof strategy
3. Handle edge cases explicitly
4. Prefer simple proofs over complex ones

Enclose your proof in an appropriate code block."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_lean_tactics() {
        let proof = r#"
theorem test : True := by
  intro h
  apply And.intro
  · exact h
  · simp
        "#;

        let tactics = extract_lean_tactics(proof);
        assert!(tactics.contains(&"intro".to_string()));
        assert!(tactics.contains(&"apply".to_string()));
        assert!(tactics.contains(&"exact".to_string()));
        assert!(tactics.contains(&"simp".to_string()));
    }

    #[test]
    fn test_extract_coq_tactics() {
        let proof = r#"
Proof.
  intros x y.
  apply eq_refl.
  simpl.
Qed.
        "#;

        let tactics = extract_coq_tactics(proof);
        assert!(tactics.contains(&"intros".to_string()));
        assert!(tactics.contains(&"apply".to_string()));
        assert!(tactics.contains(&"simpl".to_string()));
    }

    #[test]
    fn test_synthesis_request() {
        use dashprove_usl::ast::{Expr, Theorem};

        let prop = Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });

        let request = SynthesisRequest::new(&prop, BackendId::Lean4)
            .with_hints(vec!["use trivial".to_string()]);

        assert_eq!(request.backend, BackendId::Lean4);
        assert_eq!(request.hints.len(), 1);
    }
}
