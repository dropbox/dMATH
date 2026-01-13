//! Cross-tool proof translation
//!
//! This module provides translation between different proof assistant languages:
//!
//! - **Lean 4 ↔ Coq**: Syntax-directed translation between Lean 4 and Coq/Gallina
//! - **Lean 4 ↔ Isabelle**: Translation between Lean 4 and Isabelle/HOL
//! - **Coq ↔ Isabelle**: Translation between Coq and Isabelle
//!
//! Translation is based on syntax patterns and tactic correspondence, with optional
//! LLM assistance for complex cases.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use dashprove_ai::translate::{ProofTranslator, TranslateRequest, ProofLanguage};
//!
//! let translator = ProofTranslator::new();
//! let coq_proof = translator.translate(
//!     lean_proof,
//!     ProofLanguage::Lean4,
//!     ProofLanguage::Coq,
//! );
//! ```

use crate::llm::{LlmClient, LlmError, LlmMessage};
use serde::{Deserialize, Serialize};

/// Supported proof languages for translation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofLanguage {
    /// Lean 4 theorem prover
    Lean4,
    /// Coq/Gallina
    Coq,
    /// Isabelle/HOL
    Isabelle,
    /// Dafny
    Dafny,
    /// Agda
    Agda,
    /// F*
    FStar,
}

impl ProofLanguage {
    /// Get the file extension for this language
    pub fn extension(self) -> &'static str {
        match self {
            Self::Lean4 => "lean",
            Self::Coq => "v",
            Self::Isabelle => "thy",
            Self::Dafny => "dfy",
            Self::Agda => "agda",
            Self::FStar => "fst",
        }
    }

    /// Get the name for display
    pub fn name(self) -> &'static str {
        match self {
            Self::Lean4 => "Lean 4",
            Self::Coq => "Coq",
            Self::Isabelle => "Isabelle",
            Self::Dafny => "Dafny",
            Self::Agda => "Agda",
            Self::FStar => "F*",
        }
    }

    /// Check if translation to target is supported
    pub fn can_translate_to(self, target: Self) -> bool {
        // Core supported translations
        matches!(
            (self, target),
            (Self::Lean4, Self::Coq)
                | (Self::Coq, Self::Lean4)
                | (Self::Lean4, Self::Isabelle)
                | (Self::Isabelle, Self::Lean4)
                | (Self::Coq, Self::Isabelle)
                | (Self::Isabelle, Self::Coq)
        )
    }
}

/// Request for proof translation
#[derive(Debug, Clone)]
pub struct TranslateRequest<'a> {
    /// Source proof code
    pub source: &'a str,
    /// Source language
    pub from: ProofLanguage,
    /// Target language
    pub to: ProofLanguage,
    /// Optional context (e.g., imports, definitions)
    pub context: Option<&'a str>,
    /// Whether to use LLM assistance for complex cases
    pub use_llm: bool,
}

impl<'a> TranslateRequest<'a> {
    /// Create a new translation request
    pub fn new(source: &'a str, from: ProofLanguage, to: ProofLanguage) -> Self {
        Self {
            source,
            from,
            to,
            context: None,
            use_llm: false,
        }
    }

    /// Add context (imports, definitions)
    pub fn with_context(mut self, context: &'a str) -> Self {
        self.context = Some(context);
        self
    }

    /// Enable LLM assistance
    pub fn with_llm(mut self) -> Self {
        self.use_llm = true;
        self
    }
}

/// Result of proof translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResult {
    /// Translated proof code
    pub translated: String,
    /// Confidence in the translation (0.0 - 1.0)
    pub confidence: f64,
    /// Warnings about potential issues
    pub warnings: Vec<String>,
    /// Whether LLM was used
    pub used_llm: bool,
    /// Translation notes
    pub notes: Vec<String>,
}

/// Error during translation
#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    /// Unsupported language pair
    #[error("Translation from {from:?} to {to:?} is not supported")]
    UnsupportedPair {
        from: ProofLanguage,
        to: ProofLanguage,
    },

    /// Parse error in source
    #[error("Failed to parse source: {0}")]
    ParseError(String),

    /// LLM error
    #[error("LLM error: {0}")]
    LlmError(#[from] LlmError),

    /// Translation incomplete
    #[error("Translation incomplete: {0}")]
    Incomplete(String),
}

/// Proof translator
pub struct ProofTranslator {
    client: Option<Box<dyn LlmClient>>,
}

impl Default for ProofTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofTranslator {
    /// Create a translator without LLM
    pub fn new() -> Self {
        Self { client: None }
    }

    /// Create a translator with LLM client
    pub fn with_llm(client: Box<dyn LlmClient>) -> Self {
        Self {
            client: Some(client),
        }
    }

    /// Translate a proof between languages
    pub fn translate(
        &self,
        request: &TranslateRequest<'_>,
    ) -> Result<TranslateResult, TranslateError> {
        // Verify translation is supported
        if !request.from.can_translate_to(request.to) {
            return Err(TranslateError::UnsupportedPair {
                from: request.from,
                to: request.to,
            });
        }

        // Perform syntax-directed translation
        let result = match (request.from, request.to) {
            (ProofLanguage::Lean4, ProofLanguage::Coq) => self.lean4_to_coq(request.source),
            (ProofLanguage::Coq, ProofLanguage::Lean4) => self.coq_to_lean4(request.source),
            (ProofLanguage::Lean4, ProofLanguage::Isabelle) => {
                self.lean4_to_isabelle(request.source)
            }
            (ProofLanguage::Isabelle, ProofLanguage::Lean4) => {
                self.isabelle_to_lean4(request.source)
            }
            (ProofLanguage::Coq, ProofLanguage::Isabelle) => self.coq_to_isabelle(request.source),
            (ProofLanguage::Isabelle, ProofLanguage::Coq) => self.isabelle_to_coq(request.source),
            _ => {
                return Err(TranslateError::UnsupportedPair {
                    from: request.from,
                    to: request.to,
                })
            }
        };

        Ok(result)
    }

    /// Translate with LLM assistance
    pub async fn translate_with_llm(
        &self,
        request: &TranslateRequest<'_>,
    ) -> Result<TranslateResult, TranslateError> {
        // First try syntax-directed translation
        let mut result = self.translate(request)?;

        // If LLM is requested and available, refine the translation
        if request.use_llm {
            if let Some(ref client) = self.client {
                if let Some(refined) = self
                    .refine_with_llm(client.as_ref(), request, &result)
                    .await?
                {
                    result = refined;
                }
            }
        }

        Ok(result)
    }

    /// Lean 4 -> Coq translation
    fn lean4_to_coq(&self, source: &str) -> TranslateResult {
        let mut translated = String::new();
        let mut warnings = Vec::new();
        let mut notes = Vec::new();

        for line in source.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() {
                translated.push('\n');
                continue;
            }
            if trimmed.starts_with("--") {
                // Convert -- comment to (* *)
                translated.push_str(&format!(
                    "(* {} *)\n",
                    trimmed.trim_start_matches("--").trim()
                ));
                continue;
            }

            // Translate theorem/lemma declarations
            if let Some(rest) = trimmed.strip_prefix("theorem ") {
                let converted = convert_lean4_theorem_to_coq(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("lemma ") {
                let converted = convert_lean4_theorem_to_coq(rest);
                translated.push_str(&format!(
                    "Lemma {}",
                    converted.trim_start_matches("Theorem ")
                ));
                translated.push('\n');
                continue;
            }

            // Translate definitions
            if let Some(rest) = trimmed.strip_prefix("def ") {
                let converted = convert_lean4_def_to_coq(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Translate proof mode
            if trimmed == "by" {
                translated.push_str("Proof.\n");
                continue;
            }

            // Translate tactics
            let tactic = translate_lean4_tactic_to_coq(trimmed);
            if !tactic.is_empty() {
                translated.push_str(&format!("  {}.\n", tactic));
            } else if trimmed == "done" || trimmed.starts_with("Qed") {
                translated.push_str("Qed.\n");
            } else {
                // Pass through unknown lines with warning
                translated.push_str(&format!("  (* TODO: {} *)\n", trimmed));
                warnings.push(format!("Could not translate: {}", trimmed));
            }
        }

        notes.push("Syntax-directed Lean 4 to Coq translation".to_string());

        let confidence = calculate_confidence(&warnings, source.lines().count());

        TranslateResult {
            translated: translated.trim().to_string(),
            confidence,
            warnings,
            used_llm: false,
            notes,
        }
    }

    /// Coq -> Lean 4 translation
    fn coq_to_lean4(&self, source: &str) -> TranslateResult {
        let mut translated = String::new();
        let mut warnings = Vec::new();
        let mut notes = Vec::new();

        for line in source.lines() {
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                translated.push('\n');
                continue;
            }

            // Convert (* *) comments to --
            if trimmed.starts_with("(*") && trimmed.ends_with("*)") {
                let comment = trimmed
                    .trim_start_matches("(*")
                    .trim_end_matches("*)")
                    .trim();
                translated.push_str(&format!("-- {}\n", comment));
                continue;
            }

            // Translate theorem/lemma declarations
            if let Some(rest) = trimmed.strip_prefix("Theorem ") {
                let converted = convert_coq_theorem_to_lean4(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("Lemma ") {
                let converted = convert_coq_theorem_to_lean4(rest);
                translated.push_str(&format!(
                    "lemma {}",
                    converted.trim_start_matches("theorem ")
                ));
                translated.push('\n');
                continue;
            }

            // Translate definitions
            if let Some(rest) = trimmed.strip_prefix("Definition ") {
                let converted = convert_coq_def_to_lean4(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Translate proof keywords
            if trimmed == "Proof." {
                translated.push_str("  by\n");
                continue;
            }
            if trimmed == "Qed." || trimmed == "Defined." {
                // Lean 4 doesn't need explicit Qed
                continue;
            }

            // Translate tactics
            let tactic_line = trimmed.trim_end_matches('.');
            let tactic = translate_coq_tactic_to_lean4(tactic_line);
            if !tactic.is_empty() {
                translated.push_str(&format!("    {}\n", tactic));
            } else {
                // Pass through unknown lines with warning
                translated.push_str(&format!("    -- TODO: {}\n", trimmed));
                warnings.push(format!("Could not translate: {}", trimmed));
            }
        }

        notes.push("Syntax-directed Coq to Lean 4 translation".to_string());

        let confidence = calculate_confidence(&warnings, source.lines().count());

        TranslateResult {
            translated: translated.trim().to_string(),
            confidence,
            warnings,
            used_llm: false,
            notes,
        }
    }

    /// Lean 4 -> Isabelle translation
    fn lean4_to_isabelle(&self, source: &str) -> TranslateResult {
        let mut translated = String::new();
        let mut warnings = Vec::new();
        let mut notes = Vec::new();

        // Isabelle theories need a header
        translated.push_str("theory Translated\nimports Main\nbegin\n\n");

        for line in source.lines() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                translated.push('\n');
                continue;
            }
            if trimmed.starts_with("--") {
                // Convert -- comment to (* *)
                translated.push_str(&format!(
                    "(* {} *)\n",
                    trimmed.trim_start_matches("--").trim()
                ));
                continue;
            }

            // Translate theorem/lemma
            if let Some(rest) = trimmed.strip_prefix("theorem ") {
                let converted = convert_lean4_theorem_to_isabelle(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("lemma ") {
                let converted = convert_lean4_theorem_to_isabelle(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Translate definitions
            if let Some(rest) = trimmed.strip_prefix("def ") {
                let converted = convert_lean4_def_to_isabelle(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Translate proof mode
            if trimmed == "by" {
                translated.push_str("proof -\n");
                continue;
            }

            // Translate tactics
            let tactic = translate_lean4_tactic_to_isabelle(trimmed);
            if !tactic.is_empty() {
                translated.push_str(&format!("  {}\n", tactic));
            } else if trimmed == "done" {
                translated.push_str("qed\n");
            } else {
                translated.push_str(&format!("  (* TODO: {} *)\n", trimmed));
                warnings.push(format!("Could not translate: {}", trimmed));
            }
        }

        translated.push_str("\nend\n");

        notes.push("Syntax-directed Lean 4 to Isabelle translation".to_string());

        let confidence = calculate_confidence(&warnings, source.lines().count());

        TranslateResult {
            translated: translated.trim().to_string(),
            confidence,
            warnings,
            used_llm: false,
            notes,
        }
    }

    /// Isabelle -> Lean 4 translation
    fn isabelle_to_lean4(&self, source: &str) -> TranslateResult {
        let mut translated = String::new();
        let mut warnings = Vec::new();
        let mut notes = Vec::new();
        let mut in_proof = false;

        for line in source.lines() {
            let trimmed = line.trim();

            // Skip theory header lines
            if trimmed.starts_with("theory ")
                || trimmed.starts_with("imports ")
                || trimmed == "begin"
                || trimmed == "end"
            {
                continue;
            }

            if trimmed.is_empty() {
                translated.push('\n');
                continue;
            }

            // Convert (* *) comments to --
            if trimmed.starts_with("(*") && trimmed.ends_with("*)") {
                let comment = trimmed
                    .trim_start_matches("(*")
                    .trim_end_matches("*)")
                    .trim();
                translated.push_str(&format!("-- {}\n", comment));
                continue;
            }

            // Translate lemma/theorem
            if let Some(rest) = trimmed.strip_prefix("lemma ") {
                let converted = convert_isabelle_theorem_to_lean4(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("theorem ") {
                let converted = convert_isabelle_theorem_to_lean4(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Translate definitions
            if let Some(rest) = trimmed.strip_prefix("definition ") {
                let converted = convert_isabelle_def_to_lean4(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Handle proof blocks
            if trimmed.starts_with("proof") {
                translated.push_str("  by\n");
                in_proof = true;
                continue;
            }
            if trimmed == "qed" || trimmed == "done" {
                in_proof = false;
                continue;
            }

            // Translate methods/tactics
            if in_proof {
                let tactic = translate_isabelle_tactic_to_lean4(trimmed);
                if !tactic.is_empty() {
                    translated.push_str(&format!("    {}\n", tactic));
                } else {
                    translated.push_str(&format!("    -- TODO: {}\n", trimmed));
                    warnings.push(format!("Could not translate: {}", trimmed));
                }
            } else {
                // Try to translate as a statement
                translated.push_str(&format!("-- {}\n", trimmed));
            }
        }

        notes.push("Syntax-directed Isabelle to Lean 4 translation".to_string());

        let confidence = calculate_confidence(&warnings, source.lines().count());

        TranslateResult {
            translated: translated.trim().to_string(),
            confidence,
            warnings,
            used_llm: false,
            notes,
        }
    }

    /// Coq -> Isabelle translation
    fn coq_to_isabelle(&self, source: &str) -> TranslateResult {
        let mut translated = String::new();
        let mut warnings = Vec::new();
        let mut notes = Vec::new();

        // Isabelle theories need a header
        translated.push_str("theory Translated\nimports Main\nbegin\n\n");

        for line in source.lines() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                translated.push('\n');
                continue;
            }

            // Comments are same style
            if trimmed.starts_with("(*") {
                translated.push_str(trimmed);
                translated.push('\n');
                continue;
            }

            // Translate theorem/lemma
            if let Some(rest) = trimmed.strip_prefix("Theorem ") {
                let converted = convert_coq_theorem_to_isabelle(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("Lemma ") {
                let converted = convert_coq_theorem_to_isabelle(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Translate definitions
            if let Some(rest) = trimmed.strip_prefix("Definition ") {
                let converted = convert_coq_def_to_isabelle(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Translate proof keywords
            if trimmed == "Proof." {
                translated.push_str("proof -\n");
                continue;
            }
            if trimmed == "Qed." {
                translated.push_str("qed\n");
                continue;
            }

            // Translate tactics
            let tactic_line = trimmed.trim_end_matches('.');
            let tactic = translate_coq_tactic_to_isabelle(tactic_line);
            if !tactic.is_empty() {
                translated.push_str(&format!("  {}\n", tactic));
            } else {
                translated.push_str(&format!("  (* TODO: {} *)\n", trimmed));
                warnings.push(format!("Could not translate: {}", trimmed));
            }
        }

        translated.push_str("\nend\n");

        notes.push("Syntax-directed Coq to Isabelle translation".to_string());

        let confidence = calculate_confidence(&warnings, source.lines().count());

        TranslateResult {
            translated: translated.trim().to_string(),
            confidence,
            warnings,
            used_llm: false,
            notes,
        }
    }

    /// Isabelle -> Coq translation
    fn isabelle_to_coq(&self, source: &str) -> TranslateResult {
        let mut translated = String::new();
        let mut warnings = Vec::new();
        let mut notes = Vec::new();
        let mut in_proof = false;

        for line in source.lines() {
            let trimmed = line.trim();

            // Skip theory header lines
            if trimmed.starts_with("theory ")
                || trimmed.starts_with("imports ")
                || trimmed == "begin"
                || trimmed == "end"
            {
                continue;
            }

            if trimmed.is_empty() {
                translated.push('\n');
                continue;
            }

            // Comments are same style
            if trimmed.starts_with("(*") {
                translated.push_str(trimmed);
                translated.push('\n');
                continue;
            }

            // Translate lemma/theorem
            if let Some(rest) = trimmed.strip_prefix("lemma ") {
                let converted = convert_isabelle_theorem_to_coq(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("theorem ") {
                let converted = convert_isabelle_theorem_to_coq(rest);
                translated.push_str(&format!(
                    "Theorem {}",
                    converted.trim_start_matches("Lemma ")
                ));
                translated.push('\n');
                continue;
            }

            // Translate definitions
            if let Some(rest) = trimmed.strip_prefix("definition ") {
                let converted = convert_isabelle_def_to_coq(rest);
                translated.push_str(&converted);
                translated.push('\n');
                continue;
            }

            // Handle proof blocks
            if trimmed.starts_with("proof") {
                translated.push_str("Proof.\n");
                in_proof = true;
                continue;
            }
            if trimmed == "qed" {
                translated.push_str("Qed.\n");
                in_proof = false;
                continue;
            }

            // Translate methods/tactics
            if in_proof {
                let tactic = translate_isabelle_tactic_to_coq(trimmed);
                if !tactic.is_empty() {
                    translated.push_str(&format!("  {}.\n", tactic));
                } else {
                    translated.push_str(&format!("  (* TODO: {} *)\n", trimmed));
                    warnings.push(format!("Could not translate: {}", trimmed));
                }
            }
        }

        notes.push("Syntax-directed Isabelle to Coq translation".to_string());

        let confidence = calculate_confidence(&warnings, source.lines().count());

        TranslateResult {
            translated: translated.trim().to_string(),
            confidence,
            warnings,
            used_llm: false,
            notes,
        }
    }

    /// Refine translation with LLM
    async fn refine_with_llm(
        &self,
        client: &dyn LlmClient,
        request: &TranslateRequest<'_>,
        baseline: &TranslateResult,
    ) -> Result<Option<TranslateResult>, TranslateError> {
        if baseline.warnings.is_empty() && baseline.confidence > 0.8 {
            // High confidence, no need to refine
            return Ok(None);
        }

        let system_prompt = format!(
            "You are an expert in formal proof translation. \
             Translate proofs from {} to {}. \
             Preserve the proof structure and semantics.",
            request.from.name(),
            request.to.name()
        );

        let user_prompt = format!(
            "Translate this {} proof to {}:\n\n```\n{}\n```\n\n\
             Initial translation attempt (may have issues):\n```\n{}\n```\n\n\
             Issues found: {}\n\n\
             Please provide a corrected translation in a code block.",
            request.from.name(),
            request.to.name(),
            request.source,
            baseline.translated,
            if baseline.warnings.is_empty() {
                "None".to_string()
            } else {
                baseline.warnings.join("; ")
            }
        );

        let messages = vec![
            LlmMessage::system(&system_prompt),
            LlmMessage::user(&user_prompt),
        ];

        let response = client.complete_messages(&messages).await?;

        // Extract code block from response
        if let Some(code) = extract_code_block(&response.content) {
            let mut result = baseline.clone();
            result.translated = code;
            result.confidence = (baseline.confidence + 0.2).min(0.95);
            result.used_llm = true;
            result.notes.push("LLM-refined translation".to_string());
            result.warnings.clear(); // LLM should have fixed issues
            return Ok(Some(result));
        }

        Ok(None)
    }
}

// =============================================================================
// Tactic translation tables
// =============================================================================

/// Translate Lean 4 tactic to Coq
fn translate_lean4_tactic_to_coq(tactic: &str) -> String {
    let trimmed = tactic.trim().trim_start_matches(|c: char| {
        c == '·' || c == '-' || c == '+' || c == '*' || c.is_whitespace()
    });

    // Direct mappings
    let result = match trimmed {
        "rfl" => "reflexivity",
        "trivial" => "trivial",
        "assumption" => "assumption",
        "contradiction" => "contradiction",
        "exfalso" => "exfalso",
        "constructor" => "constructor",
        "left" => "left",
        "right" => "right",
        "split" => "split",
        "ext" => "extensionality",
        "funext" => "functional_extensionality",
        "decide" => "auto",
        _ => "",
    };

    if !result.is_empty() {
        return result.to_string();
    }

    // Pattern-based translations
    if let Some(rest) = trimmed.strip_prefix("intro ") {
        return format!("intros {}", rest);
    }
    if trimmed == "intro" {
        return "intro".to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("intros ") {
        return format!("intros {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("apply ") {
        return format!("apply {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("exact ") {
        return format!("exact {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("have ") {
        // Lean: have h : T := ... -> Coq: assert (h : T) by ...
        return format!("assert {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("show ") {
        return format!("show {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("simp") {
        if rest.is_empty() || rest.starts_with(' ') {
            return format!("simpl{}", rest);
        }
    }
    if let Some(rest) = trimmed.strip_prefix("rw ") {
        return format!("rewrite {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("rewrite ") {
        return format!("rewrite {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("cases ") {
        return format!("destruct {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("induction ") {
        return format!("induction {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("unfold ") {
        return format!("unfold {}", rest);
    }
    if trimmed == "omega" {
        return "lia".to_string();
    }
    if trimmed == "ring" {
        return "ring".to_string();
    }
    if trimmed == "linarith" {
        return "lia".to_string();
    }
    if trimmed == "norm_num" {
        return "auto".to_string();
    }

    String::new()
}

/// Translate Coq tactic to Lean 4
fn translate_coq_tactic_to_lean4(tactic: &str) -> String {
    let trimmed = tactic.trim();

    // Direct mappings
    let result = match trimmed {
        "reflexivity" => "rfl",
        "trivial" => "trivial",
        "assumption" => "assumption",
        "contradiction" => "contradiction",
        "exfalso" => "exfalso",
        "constructor" => "constructor",
        "left" => "left",
        "right" => "right",
        "split" => "constructor",
        "auto" => "decide",
        _ => "",
    };

    if !result.is_empty() {
        return result.to_string();
    }

    // Pattern-based translations
    if let Some(rest) = trimmed.strip_prefix("intros ") {
        return format!("intro {}", rest);
    }
    if trimmed == "intros" || trimmed == "intro" {
        return "intro".to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("apply ") {
        return format!("apply {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("exact ") {
        return format!("exact {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("assert ") {
        return format!("have {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("simpl") {
        if rest.is_empty() || rest.starts_with(' ') {
            return format!("simp{}", rest);
        }
    }
    if let Some(rest) = trimmed.strip_prefix("rewrite ") {
        return format!("rw {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("destruct ") {
        return format!("cases {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("induction ") {
        return format!("induction {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("unfold ") {
        return format!("unfold {}", rest);
    }
    if trimmed == "lia" || trimmed == "omega" {
        return "omega".to_string();
    }
    if trimmed == "ring" {
        return "ring".to_string();
    }

    String::new()
}

/// Translate Lean 4 tactic to Isabelle
fn translate_lean4_tactic_to_isabelle(tactic: &str) -> String {
    let trimmed = tactic.trim().trim_start_matches(|c: char| {
        c == '·' || c == '-' || c == '+' || c == '*' || c.is_whitespace()
    });

    match trimmed {
        "rfl" => "by (rule refl)".to_string(),
        "trivial" => "by auto".to_string(),
        "assumption" => "by assumption".to_string(),
        "contradiction" => "by contradiction".to_string(),
        "constructor" => "by (rule conjI)".to_string(),
        "left" => "by (rule disjI1)".to_string(),
        "right" => "by (rule disjI2)".to_string(),
        "simp" | "simp_all" => "by simp".to_string(),
        "decide" => "by auto".to_string(),
        "omega" | "linarith" => "by arith".to_string(),
        _ => {
            if let Some(rest) = trimmed.strip_prefix("intro ") {
                format!("assume {}", rest)
            } else if let Some(rest) = trimmed.strip_prefix("apply ") {
                format!("by (rule {})", rest)
            } else if let Some(rest) = trimmed.strip_prefix("exact ") {
                format!("by (rule {})", rest)
            } else if let Some(rest) = trimmed.strip_prefix("have ") {
                format!("have {}", rest)
            } else if let Some(rest) = trimmed.strip_prefix("cases ") {
                format!("by (cases {})", rest)
            } else if let Some(rest) = trimmed.strip_prefix("induction ") {
                format!("by (induction {})", rest)
            } else {
                String::new()
            }
        }
    }
}

/// Translate Isabelle tactic to Lean 4
fn translate_isabelle_tactic_to_lean4(tactic: &str) -> String {
    let trimmed = tactic.trim();

    if trimmed == "by auto" || trimmed == "auto" {
        return "decide".to_string();
    }
    if trimmed == "by simp" || trimmed == "simp" {
        return "simp".to_string();
    }
    if trimmed == "by blast" || trimmed == "blast" {
        return "decide".to_string();
    }
    if trimmed == "by arith" || trimmed == "arith" {
        return "omega".to_string();
    }
    if trimmed.contains("rule refl") {
        return "rfl".to_string();
    }
    if trimmed.contains("rule conjI") {
        return "constructor".to_string();
    }
    if trimmed.contains("rule disjI1") {
        return "left".to_string();
    }
    if trimmed.contains("rule disjI2") {
        return "right".to_string();
    }
    if trimmed == "assumption" || trimmed == "by assumption" {
        return "assumption".to_string();
    }

    if let Some(rest) = trimmed.strip_prefix("assume ") {
        return format!("intro {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("have ") {
        return format!("have {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("show ") {
        return format!("show {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("by (rule ") {
        let rule = rest.trim_end_matches(')');
        return format!("apply {}", rule);
    }
    if let Some(rest) = trimmed.strip_prefix("by (cases ") {
        let var = rest.trim_end_matches(')');
        return format!("cases {}", var);
    }
    if let Some(rest) = trimmed.strip_prefix("by (induction ") {
        let var = rest.trim_end_matches(')');
        return format!("induction {}", var);
    }

    String::new()
}

/// Translate Coq tactic to Isabelle
fn translate_coq_tactic_to_isabelle(tactic: &str) -> String {
    let trimmed = tactic.trim();

    match trimmed {
        "reflexivity" => "by (rule refl)".to_string(),
        "trivial" => "by auto".to_string(),
        "assumption" => "by assumption".to_string(),
        "auto" => "by auto".to_string(),
        "simpl" => "by simp".to_string(),
        "lia" | "omega" => "by arith".to_string(),
        "constructor" => "by (rule conjI)".to_string(),
        "left" => "by (rule disjI1)".to_string(),
        "right" => "by (rule disjI2)".to_string(),
        _ => {
            if let Some(rest) = trimmed.strip_prefix("intros ") {
                format!("assume {}", rest)
            } else if trimmed == "intros" || trimmed == "intro" {
                "assume".to_string()
            } else if let Some(rest) = trimmed.strip_prefix("apply ") {
                format!("by (rule {})", rest)
            } else if let Some(rest) = trimmed.strip_prefix("exact ") {
                format!("by (rule {})", rest)
            } else if let Some(rest) = trimmed.strip_prefix("assert ") {
                format!("have {}", rest)
            } else if let Some(rest) = trimmed.strip_prefix("destruct ") {
                format!("by (cases {})", rest)
            } else if let Some(rest) = trimmed.strip_prefix("induction ") {
                format!("by (induction {})", rest)
            } else {
                String::new()
            }
        }
    }
}

/// Translate Isabelle tactic to Coq
fn translate_isabelle_tactic_to_coq(tactic: &str) -> String {
    let trimmed = tactic.trim();

    if trimmed == "by auto" || trimmed == "auto" {
        return "auto".to_string();
    }
    if trimmed == "by simp" || trimmed == "simp" {
        return "simpl".to_string();
    }
    if trimmed == "by blast" || trimmed == "blast" {
        return "auto".to_string();
    }
    if trimmed == "by arith" || trimmed == "arith" {
        return "lia".to_string();
    }
    if trimmed.contains("rule refl") {
        return "reflexivity".to_string();
    }
    if trimmed.contains("rule conjI") {
        return "constructor".to_string();
    }
    if trimmed == "assumption" || trimmed == "by assumption" {
        return "assumption".to_string();
    }

    if let Some(rest) = trimmed.strip_prefix("assume ") {
        return format!("intros {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("have ") {
        return format!("assert {}", rest);
    }
    if let Some(rest) = trimmed.strip_prefix("by (rule ") {
        let rule = rest.trim_end_matches(')');
        return format!("apply {}", rule);
    }
    if let Some(rest) = trimmed.strip_prefix("by (cases ") {
        let var = rest.trim_end_matches(')');
        return format!("destruct {}", var);
    }
    if let Some(rest) = trimmed.strip_prefix("by (induction ") {
        let var = rest.trim_end_matches(')');
        return format!("induction {}", var);
    }

    String::new()
}

// =============================================================================
// Declaration conversion helpers
// =============================================================================

/// Convert Lean 4 theorem declaration to Coq
fn convert_lean4_theorem_to_coq(decl: &str) -> String {
    // Lean: theorem name (params) : type := ...
    // Coq: Theorem name : forall params, type.
    let parts: Vec<&str> = decl.splitn(2, ':').collect();
    if parts.len() < 2 {
        return format!("Theorem {}.", decl);
    }

    let name_params = parts[0].trim();
    let type_body = parts[1].trim().trim_start_matches('=').trim();

    // Extract name and params
    let (name, params) = if let Some(paren_start) = name_params.find('(') {
        let name = name_params[..paren_start].trim();
        let params = &name_params[paren_start..];
        (name, params.to_string())
    } else {
        (name_params, String::new())
    };

    if params.is_empty() {
        format!("Theorem {} : {}.", name, type_body)
    } else {
        // Convert (x : T) -> forall (x : T),
        let coq_params = convert_lean4_params_to_coq(&params);
        format!("Theorem {} : forall {}, {}.", name, coq_params, type_body)
    }
}

/// Convert Lean 4 def to Coq Definition
fn convert_lean4_def_to_coq(decl: &str) -> String {
    let parts: Vec<&str> = decl.splitn(2, ":=").collect();
    if parts.len() < 2 {
        return format!("Definition {}.", decl);
    }

    let signature = parts[0].trim();
    let body = parts[1].trim();

    format!("Definition {} := {}.", signature, body)
}

/// Convert Lean 4 params to Coq forall format
fn convert_lean4_params_to_coq(params: &str) -> String {
    // Simple conversion - just replace outer parens
    params.replace('{', "(").replace('}', ")")
}

/// Convert Coq theorem to Lean 4
fn convert_coq_theorem_to_lean4(decl: &str) -> String {
    // Coq: name : type.
    // Lean: theorem name : type := by ...
    let decl = decl.trim_end_matches('.');
    let parts: Vec<&str> = decl.splitn(2, ':').collect();

    if parts.len() < 2 {
        return format!("theorem {}", decl);
    }

    let name = parts[0].trim();
    let ty = parts[1].trim();

    // Remove forall if present at the start
    let ty = ty.strip_prefix("forall ").unwrap_or(ty);

    format!("theorem {} : {} := ", name, ty)
}

/// Convert Coq Definition to Lean 4 def
fn convert_coq_def_to_lean4(decl: &str) -> String {
    let decl = decl.trim_end_matches('.');
    let parts: Vec<&str> = decl.splitn(2, ":=").collect();

    if parts.len() < 2 {
        return format!("def {}", decl);
    }

    let signature = parts[0].trim();
    let body = parts[1].trim();

    format!("def {} := {}", signature, body)
}

/// Convert Lean 4 theorem to Isabelle lemma
fn convert_lean4_theorem_to_isabelle(decl: &str) -> String {
    let parts: Vec<&str> = decl.splitn(2, ':').collect();
    if parts.len() < 2 {
        return format!("lemma \"{}\"", decl);
    }

    let name = parts[0].split_whitespace().next().unwrap_or("unnamed");
    let ty = parts[1].trim().trim_start_matches('=').trim();

    format!("lemma {}: \"{}\"", name, ty)
}

/// Convert Lean 4 def to Isabelle definition
fn convert_lean4_def_to_isabelle(decl: &str) -> String {
    let parts: Vec<&str> = decl.splitn(2, ":=").collect();
    if parts.len() < 2 {
        return format!("definition \"{}\"", decl);
    }

    let signature = parts[0].trim();
    let body = parts[1].trim();

    // Extract just the name
    let name = signature.split_whitespace().next().unwrap_or("unnamed");

    format!("definition {} where \"{} = {}\"", name, name, body)
}

/// Convert Isabelle lemma to Lean 4 theorem
fn convert_isabelle_theorem_to_lean4(decl: &str) -> String {
    // Isabelle: name: "prop"
    let parts: Vec<&str> = decl.splitn(2, ':').collect();
    if parts.len() < 2 {
        return format!("theorem {}", decl);
    }

    let name = parts[0].trim();
    let prop = parts[1].trim().trim_matches('"');

    format!("theorem {} : {} := ", name, prop)
}

/// Convert Isabelle definition to Lean 4 def
fn convert_isabelle_def_to_lean4(decl: &str) -> String {
    // Isabelle: name where "name = body"
    if let Some(where_idx) = decl.find("where") {
        let name = decl[..where_idx].trim();
        let body_part = &decl[where_idx + 5..];
        let body = body_part.trim().trim_matches('"');

        // Extract just the RHS of the equation
        if let Some(eq_idx) = body.find('=') {
            let rhs = body[eq_idx + 1..].trim();
            return format!("def {} := {}", name, rhs);
        }
    }

    format!("def {}", decl)
}

/// Convert Coq theorem to Isabelle lemma
fn convert_coq_theorem_to_isabelle(decl: &str) -> String {
    let decl = decl.trim_end_matches('.');
    let parts: Vec<&str> = decl.splitn(2, ':').collect();

    if parts.len() < 2 {
        return format!("lemma \"{}\"", decl);
    }

    let name = parts[0].trim();
    let ty = parts[1].trim();

    format!("lemma {}: \"{}\"", name, ty)
}

/// Convert Coq Definition to Isabelle definition
fn convert_coq_def_to_isabelle(decl: &str) -> String {
    let decl = decl.trim_end_matches('.');
    let parts: Vec<&str> = decl.splitn(2, ":=").collect();

    if parts.len() < 2 {
        return format!("definition \"{}\"", decl);
    }

    let signature = parts[0].trim();
    let body = parts[1].trim();

    let name = signature.split_whitespace().next().unwrap_or("unnamed");

    format!("definition {} where \"{} = {}\"", name, name, body)
}

/// Convert Isabelle lemma to Coq
fn convert_isabelle_theorem_to_coq(decl: &str) -> String {
    let parts: Vec<&str> = decl.splitn(2, ':').collect();
    if parts.len() < 2 {
        return format!("Lemma {}.", decl);
    }

    let name = parts[0].trim();
    let prop = parts[1].trim().trim_matches('"');

    format!("Lemma {} : {}.", name, prop)
}

/// Convert Isabelle definition to Coq
fn convert_isabelle_def_to_coq(decl: &str) -> String {
    if let Some(where_idx) = decl.find("where") {
        let name = decl[..where_idx].trim();
        let body_part = &decl[where_idx + 5..];
        let body = body_part.trim().trim_matches('"');

        if let Some(eq_idx) = body.find('=') {
            let rhs = body[eq_idx + 1..].trim();
            return format!("Definition {} := {}.", name, rhs);
        }
    }

    format!("Definition {}.", decl)
}

// =============================================================================
// Utility functions
// =============================================================================

/// Calculate confidence based on warning count
fn calculate_confidence(warnings: &[String], total_lines: usize) -> f64 {
    if total_lines == 0 {
        return 0.5;
    }

    let warning_ratio = warnings.len() as f64 / total_lines as f64;
    let base_confidence = 0.7;

    (base_confidence - warning_ratio * 2.0).clamp(0.1, 0.95)
}

/// Extract code block from LLM response
fn extract_code_block(response: &str) -> Option<String> {
    if let Some(start) = response.find("```") {
        let after_backticks = &response[start + 3..];
        // Skip language identifier
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lean4_to_coq_simple() {
        let translator = ProofTranslator::new();
        let lean_proof = r#"
theorem test : True := by
  trivial
"#;

        let result = translator
            .translate(&TranslateRequest::new(
                lean_proof,
                ProofLanguage::Lean4,
                ProofLanguage::Coq,
            ))
            .unwrap();

        assert!(result.translated.contains("Theorem"));
        assert!(result.translated.contains("trivial"));
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_coq_to_lean4_simple() {
        let translator = ProofTranslator::new();
        let coq_proof = r#"
Theorem test : True.
Proof.
  trivial.
Qed.
"#;

        let result = translator
            .translate(&TranslateRequest::new(
                coq_proof,
                ProofLanguage::Coq,
                ProofLanguage::Lean4,
            ))
            .unwrap();

        assert!(result.translated.contains("theorem"));
        assert!(result.translated.contains("trivial"));
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_lean4_to_isabelle() {
        let translator = ProofTranslator::new();
        let lean_proof = r#"
theorem test : True := by
  trivial
"#;

        let result = translator
            .translate(&TranslateRequest::new(
                lean_proof,
                ProofLanguage::Lean4,
                ProofLanguage::Isabelle,
            ))
            .unwrap();

        assert!(result.translated.contains("lemma"));
        assert!(result.translated.contains("theory"));
        assert!(result.translated.contains("end"));
    }

    #[test]
    fn test_tactic_translation_lean4_to_coq() {
        assert_eq!(translate_lean4_tactic_to_coq("rfl"), "reflexivity");
        assert_eq!(translate_lean4_tactic_to_coq("intro x"), "intros x");
        assert_eq!(translate_lean4_tactic_to_coq("apply h"), "apply h");
        assert_eq!(translate_lean4_tactic_to_coq("simp"), "simpl");
        assert_eq!(translate_lean4_tactic_to_coq("omega"), "lia");
        assert_eq!(translate_lean4_tactic_to_coq("cases x"), "destruct x");
    }

    // Additional tactic tests to catch mutation testing gaps
    #[test]
    fn test_lean4_to_coq_direct_tactics() {
        // Test each direct mapping in translate_lean4_tactic_to_coq
        assert_eq!(translate_lean4_tactic_to_coq("trivial"), "trivial");
        assert_eq!(translate_lean4_tactic_to_coq("assumption"), "assumption");
        assert_eq!(
            translate_lean4_tactic_to_coq("contradiction"),
            "contradiction"
        );
        assert_eq!(translate_lean4_tactic_to_coq("exfalso"), "exfalso");
        assert_eq!(translate_lean4_tactic_to_coq("constructor"), "constructor");
        assert_eq!(translate_lean4_tactic_to_coq("left"), "left");
        assert_eq!(translate_lean4_tactic_to_coq("right"), "right");
        assert_eq!(translate_lean4_tactic_to_coq("split"), "split");
        assert_eq!(translate_lean4_tactic_to_coq("ext"), "extensionality");
        assert_eq!(
            translate_lean4_tactic_to_coq("funext"),
            "functional_extensionality"
        );
        assert_eq!(translate_lean4_tactic_to_coq("decide"), "auto");
    }

    #[test]
    fn test_lean4_to_coq_pattern_tactics() {
        // Test pattern-based translations
        assert_eq!(translate_lean4_tactic_to_coq("intro"), "intro");
        assert_eq!(translate_lean4_tactic_to_coq("intros a b"), "intros a b");
        assert_eq!(translate_lean4_tactic_to_coq("exact h"), "exact h");
        assert_eq!(translate_lean4_tactic_to_coq("have h : T"), "assert h : T");
        assert_eq!(translate_lean4_tactic_to_coq("show P"), "show P");
        assert_eq!(translate_lean4_tactic_to_coq("simp [h]"), "simpl [h]");
        assert_eq!(translate_lean4_tactic_to_coq("rw h"), "rewrite h");
        assert_eq!(translate_lean4_tactic_to_coq("rewrite h"), "rewrite h");
        assert_eq!(translate_lean4_tactic_to_coq("induction n"), "induction n");
        assert_eq!(translate_lean4_tactic_to_coq("unfold f"), "unfold f");
        assert_eq!(translate_lean4_tactic_to_coq("ring"), "ring");
        assert_eq!(translate_lean4_tactic_to_coq("linarith"), "lia");
        assert_eq!(translate_lean4_tactic_to_coq("norm_num"), "auto");
    }

    #[test]
    fn test_tactic_translation_coq_to_lean4() {
        assert_eq!(translate_coq_tactic_to_lean4("reflexivity"), "rfl");
        assert_eq!(translate_coq_tactic_to_lean4("intros x"), "intro x");
        assert_eq!(translate_coq_tactic_to_lean4("apply h"), "apply h");
        assert_eq!(translate_coq_tactic_to_lean4("simpl"), "simp");
        assert_eq!(translate_coq_tactic_to_lean4("lia"), "omega");
        assert_eq!(translate_coq_tactic_to_lean4("destruct x"), "cases x");
    }

    #[test]
    fn test_coq_to_lean4_direct_tactics() {
        // Test all direct mappings in translate_coq_tactic_to_lean4
        assert_eq!(translate_coq_tactic_to_lean4("trivial"), "trivial");
        assert_eq!(translate_coq_tactic_to_lean4("assumption"), "assumption");
        assert_eq!(
            translate_coq_tactic_to_lean4("contradiction"),
            "contradiction"
        );
        assert_eq!(translate_coq_tactic_to_lean4("exfalso"), "exfalso");
        assert_eq!(translate_coq_tactic_to_lean4("constructor"), "constructor");
        assert_eq!(translate_coq_tactic_to_lean4("left"), "left");
        assert_eq!(translate_coq_tactic_to_lean4("right"), "right");
        assert_eq!(translate_coq_tactic_to_lean4("split"), "constructor");
        assert_eq!(translate_coq_tactic_to_lean4("auto"), "decide");
    }

    #[test]
    fn test_coq_to_lean4_pattern_tactics() {
        // Test pattern-based translations
        assert_eq!(translate_coq_tactic_to_lean4("intros"), "intro");
        assert_eq!(translate_coq_tactic_to_lean4("intro"), "intro");
        assert_eq!(translate_coq_tactic_to_lean4("exact h"), "exact h");
        assert_eq!(translate_coq_tactic_to_lean4("assert h : T"), "have h : T");
        assert_eq!(
            translate_coq_tactic_to_lean4("simpl with hints"),
            "simp with hints"
        );
        assert_eq!(translate_coq_tactic_to_lean4("rewrite h"), "rw h");
        assert_eq!(translate_coq_tactic_to_lean4("induction n"), "induction n");
        assert_eq!(translate_coq_tactic_to_lean4("unfold f"), "unfold f");
        assert_eq!(translate_coq_tactic_to_lean4("omega"), "omega");
        assert_eq!(translate_coq_tactic_to_lean4("ring"), "ring");
    }

    #[test]
    fn test_tactic_translation_lean4_to_isabelle() {
        assert_eq!(translate_lean4_tactic_to_isabelle("rfl"), "by (rule refl)");
        assert_eq!(translate_lean4_tactic_to_isabelle("simp"), "by simp");
        assert_eq!(translate_lean4_tactic_to_isabelle("omega"), "by arith");
        assert_eq!(translate_lean4_tactic_to_isabelle("intro x"), "assume x");
    }

    #[test]
    fn test_lean4_to_isabelle_direct_tactics() {
        // Test all direct mappings in translate_lean4_tactic_to_isabelle
        assert_eq!(translate_lean4_tactic_to_isabelle("trivial"), "by auto");
        assert_eq!(
            translate_lean4_tactic_to_isabelle("assumption"),
            "by assumption"
        );
        assert_eq!(
            translate_lean4_tactic_to_isabelle("contradiction"),
            "by contradiction"
        );
        assert_eq!(
            translate_lean4_tactic_to_isabelle("constructor"),
            "by (rule conjI)"
        );
        assert_eq!(
            translate_lean4_tactic_to_isabelle("left"),
            "by (rule disjI1)"
        );
        assert_eq!(
            translate_lean4_tactic_to_isabelle("right"),
            "by (rule disjI2)"
        );
        assert_eq!(translate_lean4_tactic_to_isabelle("simp_all"), "by simp");
        assert_eq!(translate_lean4_tactic_to_isabelle("decide"), "by auto");
        assert_eq!(translate_lean4_tactic_to_isabelle("linarith"), "by arith");
    }

    #[test]
    fn test_lean4_to_isabelle_pattern_tactics() {
        assert_eq!(translate_lean4_tactic_to_isabelle("apply h"), "by (rule h)");
        assert_eq!(translate_lean4_tactic_to_isabelle("exact h"), "by (rule h)");
        assert_eq!(
            translate_lean4_tactic_to_isabelle("have h : T"),
            "have h : T"
        );
        assert_eq!(
            translate_lean4_tactic_to_isabelle("cases x"),
            "by (cases x)"
        );
        assert_eq!(
            translate_lean4_tactic_to_isabelle("induction n"),
            "by (induction n)"
        );
    }

    #[test]
    fn test_tactic_translation_isabelle_to_lean4() {
        assert_eq!(translate_isabelle_tactic_to_lean4("by auto"), "decide");
        assert_eq!(translate_isabelle_tactic_to_lean4("by simp"), "simp");
        assert_eq!(translate_isabelle_tactic_to_lean4("by arith"), "omega");
        assert_eq!(translate_isabelle_tactic_to_lean4("assume x"), "intro x");
    }

    #[test]
    fn test_isabelle_to_lean4_direct_tactics() {
        // Test all direct mappings in translate_isabelle_tactic_to_lean4
        assert_eq!(translate_isabelle_tactic_to_lean4("auto"), "decide");
        assert_eq!(translate_isabelle_tactic_to_lean4("simp"), "simp");
        assert_eq!(translate_isabelle_tactic_to_lean4("blast"), "decide");
        assert_eq!(translate_isabelle_tactic_to_lean4("by blast"), "decide");
        assert_eq!(translate_isabelle_tactic_to_lean4("arith"), "omega");
        assert_eq!(
            translate_isabelle_tactic_to_lean4("assumption"),
            "assumption"
        );
        assert_eq!(
            translate_isabelle_tactic_to_lean4("by assumption"),
            "assumption"
        );
        assert_eq!(translate_isabelle_tactic_to_lean4("by (rule refl)"), "rfl");
        assert_eq!(
            translate_isabelle_tactic_to_lean4("by (rule conjI)"),
            "constructor"
        );
        assert_eq!(
            translate_isabelle_tactic_to_lean4("by (rule disjI1)"),
            "left"
        );
        assert_eq!(
            translate_isabelle_tactic_to_lean4("by (rule disjI2)"),
            "right"
        );
    }

    #[test]
    fn test_isabelle_to_lean4_pattern_tactics() {
        assert_eq!(
            translate_isabelle_tactic_to_lean4("have h : T"),
            "have h : T"
        );
        assert_eq!(translate_isabelle_tactic_to_lean4("show P"), "show P");
        assert_eq!(translate_isabelle_tactic_to_lean4("by (rule h)"), "apply h");
        assert_eq!(
            translate_isabelle_tactic_to_lean4("by (cases x)"),
            "cases x"
        );
        assert_eq!(
            translate_isabelle_tactic_to_lean4("by (induction n)"),
            "induction n"
        );
    }

    #[test]
    fn test_coq_to_isabelle_tactics() {
        // Test translate_coq_tactic_to_isabelle
        assert_eq!(
            translate_coq_tactic_to_isabelle("reflexivity"),
            "by (rule refl)"
        );
        assert_eq!(translate_coq_tactic_to_isabelle("trivial"), "by auto");
        assert_eq!(
            translate_coq_tactic_to_isabelle("assumption"),
            "by assumption"
        );
        assert_eq!(translate_coq_tactic_to_isabelle("auto"), "by auto");
        assert_eq!(translate_coq_tactic_to_isabelle("simpl"), "by simp");
        assert_eq!(translate_coq_tactic_to_isabelle("lia"), "by arith");
        assert_eq!(translate_coq_tactic_to_isabelle("omega"), "by arith");
        assert_eq!(
            translate_coq_tactic_to_isabelle("constructor"),
            "by (rule conjI)"
        );
        assert_eq!(translate_coq_tactic_to_isabelle("left"), "by (rule disjI1)");
        assert_eq!(
            translate_coq_tactic_to_isabelle("right"),
            "by (rule disjI2)"
        );
        // Pattern-based
        assert_eq!(translate_coq_tactic_to_isabelle("intros x"), "assume x");
        assert_eq!(translate_coq_tactic_to_isabelle("intros"), "assume");
        assert_eq!(translate_coq_tactic_to_isabelle("intro"), "assume");
        assert_eq!(translate_coq_tactic_to_isabelle("apply h"), "by (rule h)");
        assert_eq!(translate_coq_tactic_to_isabelle("exact h"), "by (rule h)");
        assert_eq!(
            translate_coq_tactic_to_isabelle("assert h : T"),
            "have h : T"
        );
        assert_eq!(
            translate_coq_tactic_to_isabelle("destruct x"),
            "by (cases x)"
        );
        assert_eq!(
            translate_coq_tactic_to_isabelle("induction n"),
            "by (induction n)"
        );
    }

    #[test]
    fn test_isabelle_to_coq_tactics() {
        // Test translate_isabelle_tactic_to_coq
        assert_eq!(translate_isabelle_tactic_to_coq("by auto"), "auto");
        assert_eq!(translate_isabelle_tactic_to_coq("auto"), "auto");
        assert_eq!(translate_isabelle_tactic_to_coq("by simp"), "simpl");
        assert_eq!(translate_isabelle_tactic_to_coq("simp"), "simpl");
        assert_eq!(translate_isabelle_tactic_to_coq("by blast"), "auto");
        assert_eq!(translate_isabelle_tactic_to_coq("blast"), "auto");
        assert_eq!(translate_isabelle_tactic_to_coq("by arith"), "lia");
        assert_eq!(translate_isabelle_tactic_to_coq("arith"), "lia");
        assert_eq!(
            translate_isabelle_tactic_to_coq("by (rule refl)"),
            "reflexivity"
        );
        assert_eq!(
            translate_isabelle_tactic_to_coq("by (rule conjI)"),
            "constructor"
        );
        assert_eq!(translate_isabelle_tactic_to_coq("assumption"), "assumption");
        assert_eq!(
            translate_isabelle_tactic_to_coq("by assumption"),
            "assumption"
        );
        // Pattern-based
        assert_eq!(translate_isabelle_tactic_to_coq("assume x"), "intros x");
        assert_eq!(
            translate_isabelle_tactic_to_coq("have h : T"),
            "assert h : T"
        );
        assert_eq!(translate_isabelle_tactic_to_coq("by (rule h)"), "apply h");
        assert_eq!(
            translate_isabelle_tactic_to_coq("by (cases x)"),
            "destruct x"
        );
        assert_eq!(
            translate_isabelle_tactic_to_coq("by (induction n)"),
            "induction n"
        );
    }

    #[test]
    fn test_unsupported_pair() {
        let translator = ProofTranslator::new();
        let result = translator.translate(&TranslateRequest::new(
            "test",
            ProofLanguage::Dafny,
            ProofLanguage::Agda,
        ));

        assert!(matches!(
            result,
            Err(TranslateError::UnsupportedPair { .. })
        ));
    }

    #[test]
    fn test_language_can_translate_to() {
        assert!(ProofLanguage::Lean4.can_translate_to(ProofLanguage::Coq));
        assert!(ProofLanguage::Coq.can_translate_to(ProofLanguage::Lean4));
        assert!(ProofLanguage::Lean4.can_translate_to(ProofLanguage::Isabelle));
        assert!(!ProofLanguage::Dafny.can_translate_to(ProofLanguage::Agda));
    }

    #[test]
    fn test_proof_language_extension() {
        assert_eq!(ProofLanguage::Lean4.extension(), "lean");
        assert_eq!(ProofLanguage::Coq.extension(), "v");
        assert_eq!(ProofLanguage::Isabelle.extension(), "thy");
        assert_eq!(ProofLanguage::Dafny.extension(), "dfy");
        assert_eq!(ProofLanguage::Agda.extension(), "agda");
        assert_eq!(ProofLanguage::FStar.extension(), "fst");
    }

    #[test]
    fn test_proof_language_name() {
        // Test all ProofLanguage::name variants
        assert_eq!(ProofLanguage::Lean4.name(), "Lean 4");
        assert_eq!(ProofLanguage::Coq.name(), "Coq");
        assert_eq!(ProofLanguage::Isabelle.name(), "Isabelle");
        assert_eq!(ProofLanguage::Dafny.name(), "Dafny");
        assert_eq!(ProofLanguage::Agda.name(), "Agda");
        assert_eq!(ProofLanguage::FStar.name(), "F*");
    }

    #[test]
    fn test_convert_lean4_theorem_to_coq() {
        let result = convert_lean4_theorem_to_coq("test : True");
        assert!(result.contains("Theorem test"));
        assert!(result.contains("True"));
    }

    #[test]
    fn test_convert_lean4_theorem_to_coq_with_params() {
        // Test with parameters
        let result = convert_lean4_theorem_to_coq("add_comm (a b : Nat) : a + b = b + a");
        assert!(result.contains("Theorem add_comm"));
        assert!(result.contains("forall"));
    }

    #[test]
    fn test_convert_lean4_theorem_to_coq_no_type() {
        // Test edge case with no colon
        let result = convert_lean4_theorem_to_coq("simple_name");
        assert!(result.contains("simple_name"));
    }

    #[test]
    fn test_convert_lean4_def_to_coq() {
        // Test definition conversion
        let result = convert_lean4_def_to_coq("f := 42");
        assert!(result.contains("Definition f"));
        assert!(result.contains("42"));

        // Test edge case with no :=
        let result2 = convert_lean4_def_to_coq("simple_def");
        assert!(result2.contains("Definition simple_def"));
    }

    #[test]
    fn test_convert_lean4_params_to_coq() {
        // Test param conversion
        let result = convert_lean4_params_to_coq("{x : T}");
        assert_eq!(result, "(x : T)");
    }

    #[test]
    fn test_convert_coq_theorem_to_lean4() {
        let result = convert_coq_theorem_to_lean4("test : True.");
        assert!(result.contains("theorem test"));
        assert!(result.contains("True"));
    }

    #[test]
    fn test_convert_coq_theorem_to_lean4_edge_cases() {
        // Test with no colon
        let result = convert_coq_theorem_to_lean4("simple.");
        assert!(result.contains("theorem simple"));

        // Test with forall
        let result2 = convert_coq_theorem_to_lean4("test : forall x, P x.");
        assert!(result2.contains("theorem test"));
        assert!(result2.contains("P x"));
    }

    #[test]
    fn test_convert_coq_def_to_lean4() {
        // Test definition conversion
        let result = convert_coq_def_to_lean4("f := 42.");
        assert!(result.contains("def f"));
        assert!(result.contains("42"));

        // Test edge case with no :=
        let result2 = convert_coq_def_to_lean4("simple.");
        assert!(result2.contains("def simple"));
    }

    #[test]
    fn test_convert_lean4_theorem_to_isabelle() {
        let result = convert_lean4_theorem_to_isabelle("test : True");
        assert!(result.contains("lemma test"));
        assert!(result.contains("True"));

        // Test edge case with no colon
        let result2 = convert_lean4_theorem_to_isabelle("simple_name");
        assert!(result2.contains("simple_name"));
    }

    #[test]
    fn test_convert_lean4_def_to_isabelle() {
        // Test definition conversion
        let result = convert_lean4_def_to_isabelle("f := 42");
        assert!(result.contains("definition f"));
        assert!(result.contains("42"));

        // Test edge case with no :=
        let result2 = convert_lean4_def_to_isabelle("simple");
        assert!(result2.contains("simple"));
    }

    #[test]
    fn test_convert_isabelle_theorem_to_lean4() {
        let result = convert_isabelle_theorem_to_lean4("test: \"True\"");
        assert!(result.contains("theorem test"));
        assert!(result.contains("True"));

        // Test edge case with no colon
        let result2 = convert_isabelle_theorem_to_lean4("simple");
        assert!(result2.contains("theorem simple"));
    }

    #[test]
    fn test_convert_isabelle_def_to_lean4() {
        // Test definition conversion
        let result = convert_isabelle_def_to_lean4("f where \"f = 42\"");
        assert!(result.contains("def"));
        assert!(result.contains("42"));
    }

    #[test]
    fn test_convert_coq_theorem_to_isabelle() {
        let result = convert_coq_theorem_to_isabelle("test : True.");
        assert!(result.contains("lemma test"));
        assert!(result.contains("True"));

        // Test edge case with no colon
        let result2 = convert_coq_theorem_to_isabelle("simple.");
        assert!(result2.contains("simple"));
    }

    #[test]
    fn test_convert_coq_def_to_isabelle() {
        // Test definition conversion
        let result = convert_coq_def_to_isabelle("f := 42.");
        assert!(result.contains("definition f"));
        assert!(result.contains("42"));

        // Test edge case with no :=
        let result2 = convert_coq_def_to_isabelle("simple.");
        assert!(result2.contains("simple"));
    }

    #[test]
    fn test_convert_isabelle_theorem_to_coq() {
        let result = convert_isabelle_theorem_to_coq("test: \"True\"");
        assert!(result.contains("Lemma test"));
        assert!(result.contains("True"));

        // Test edge case with no colon
        let result2 = convert_isabelle_theorem_to_coq("simple");
        assert!(result2.contains("simple"));
    }

    #[test]
    fn test_convert_isabelle_def_to_coq() {
        // Test definition conversion
        let result = convert_isabelle_def_to_coq("f where \"f = 42\"");
        assert!(result.contains("Definition"));
        assert!(result.contains("42"));
    }

    #[test]
    fn test_confidence_calculation() {
        // No warnings = high confidence
        let conf1 = calculate_confidence(&[], 10);
        assert!(conf1 > 0.6);

        // Many warnings = low confidence
        let warnings: Vec<String> = (0..5).map(|i| format!("warning {}", i)).collect();
        let conf2 = calculate_confidence(&warnings, 10);
        assert!(conf2 < conf1);
    }

    #[test]
    fn test_confidence_calculation_edge_cases() {
        // Test with zero warnings and different line counts
        let conf_short = calculate_confidence(&[], 5);
        let conf_long = calculate_confidence(&[], 100);
        // Both should be high confidence when no warnings
        assert!(conf_short > 0.5);
        assert!(conf_long > 0.5);

        // Test that confidence changes based on number of warnings
        let one_warning = vec!["warning".to_string()];
        let conf_one = calculate_confidence(&one_warning, 10);
        let two_warnings = vec!["warning1".to_string(), "warning2".to_string()];
        let conf_two = calculate_confidence(&two_warnings, 10);
        assert!(conf_one > conf_two);
    }

    #[test]
    fn test_extract_code_block() {
        let response = "Here is the code:\n```lean\ntheorem test : True := by trivial\n```\nDone.";
        let code = extract_code_block(response);
        assert!(code.is_some());
        assert!(code.unwrap().contains("theorem test"));
    }

    #[test]
    fn test_extract_code_block_variants() {
        // Test with different language markers
        let response_coq = "Here is the code:\n```coq\nTheorem test : True.\n```";
        let code_coq = extract_code_block(response_coq);
        assert!(code_coq.is_some());
        assert!(code_coq.unwrap().contains("Theorem"));

        // Test with no language marker
        let response_bare = "Here:\n```\ndef f := 42\n```";
        let code_bare = extract_code_block(response_bare);
        assert!(code_bare.is_some());

        // Test with no code block
        let response_none = "No code here";
        let code_none = extract_code_block(response_none);
        assert!(code_none.is_none());
    }

    #[test]
    fn test_bullet_point_handling() {
        // Lean 4 uses bullet points like · - + *
        assert_eq!(translate_lean4_tactic_to_coq("· rfl"), "reflexivity");
        assert_eq!(translate_lean4_tactic_to_coq("  - apply h"), "apply h");
        assert_eq!(translate_lean4_tactic_to_coq("    + simp"), "simpl");
    }

    #[test]
    fn test_coq_to_isabelle() {
        let translator = ProofTranslator::new();
        let coq_proof = r#"
Theorem test : True.
Proof.
  trivial.
Qed.
"#;

        let result = translator
            .translate(&TranslateRequest::new(
                coq_proof,
                ProofLanguage::Coq,
                ProofLanguage::Isabelle,
            ))
            .unwrap();

        assert!(result.translated.contains("theory"));
        assert!(result.translated.contains("lemma") || result.translated.contains("Theorem"));
        assert!(result.confidence > 0.4);
    }

    #[test]
    fn test_isabelle_to_coq() {
        let translator = ProofTranslator::new();
        let isabelle_proof = r#"
theory Test
imports Main
begin

lemma test: "True"
proof -
  by auto
qed

end
"#;

        let result = translator
            .translate(&TranslateRequest::new(
                isabelle_proof,
                ProofLanguage::Isabelle,
                ProofLanguage::Coq,
            ))
            .unwrap();

        assert!(result.translated.contains("Lemma"));
        assert!(result.confidence > 0.4);
    }

    #[test]
    fn test_isabelle_to_lean4() {
        let translator = ProofTranslator::new();
        let isabelle_proof = r#"
theory Test
imports Main
begin

lemma test: "True"
  by auto

end
"#;

        let result = translator
            .translate(&TranslateRequest::new(
                isabelle_proof,
                ProofLanguage::Isabelle,
                ProofLanguage::Lean4,
            ))
            .unwrap();

        assert!(result.translated.contains("theorem") || result.translated.contains("lemma"));
        assert!(result.confidence > 0.4);
    }
}
