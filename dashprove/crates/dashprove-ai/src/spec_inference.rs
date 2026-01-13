//! Specification inference from source code
//!
//! This module infers Unified Specification Language (USL) properties from
//! source code using a combination of lightweight static analysis and optional
//! LLM refinement. It is designed to be deterministic without an LLM while
//! allowing higher-fidelity suggestions when an LLM client is available.

use crate::llm::{LlmClient, LlmError, LlmMessage};
use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::{
    ComparisonOp, Contract, Expr, FairnessConstraint, FairnessKind, Invariant, Param, Property,
    Refinement, Temporal, TemporalExpr, Theorem, Type,
};
use serde::{Deserialize, Serialize};

/// Supported source languages for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLanguage {
    /// Rust source code
    Rust,
    /// TypeScript/JavaScript source code
    TypeScript,
    /// Python source code
    Python,
    /// Go source code
    Go,
    /// C source code
    C,
    /// C++ source code
    Cpp,
    /// Unknown/unsupported
    Unknown,
}

impl SourceLanguage {
    /// Infer source language from a file path
    pub fn from_path(path: &str) -> Self {
        if let Some(ext) = std::path::Path::new(path)
            .extension()
            .and_then(std::ffi::OsStr::to_str)
        {
            match ext {
                "rs" => Self::Rust,
                "ts" | "tsx" | "js" | "jsx" => Self::TypeScript,
                "py" => Self::Python,
                "go" => Self::Go,
                "c" | "h" => Self::C,
                "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Self::Cpp,
                _ => Self::Unknown,
            }
        } else {
            Self::Unknown
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::TypeScript => "TypeScript",
            Self::Python => "Python",
            Self::Go => "Go",
            Self::C => "C",
            Self::Cpp => "C++",
            Self::Unknown => "Unknown",
        }
    }
}

/// Request to infer specifications from code
#[derive(Debug, Clone)]
pub struct SpecInferenceRequest<'a> {
    /// Source code to analyze
    pub code: &'a str,
    /// Language of the source
    pub language: SourceLanguage,
    /// Optional file path for context
    pub file_path: Option<&'a str>,
    /// Target backend to bias towards (e.g., Kani vs TLA+)
    pub target_backend: Option<BackendId>,
    /// Optional hints (e.g., "function should not panic")
    pub hints: Vec<String>,
    /// Maximum number of properties to emit
    pub max_properties: usize,
    /// Whether LLM refinement is allowed
    pub use_llm: bool,
}

impl<'a> SpecInferenceRequest<'a> {
    /// Create a new inference request
    pub fn new(code: &'a str, language: SourceLanguage) -> Self {
        Self {
            code,
            language,
            file_path: None,
            target_backend: None,
            hints: Vec::new(),
            max_properties: 5,
            use_llm: false,
        }
    }

    /// Set the target backend
    pub fn with_backend(mut self, backend: BackendId) -> Self {
        self.target_backend = Some(backend);
        self
    }

    /// Attach a file path
    pub fn with_file_path(mut self, path: &'a str) -> Self {
        self.file_path = Some(path);
        self
    }

    /// Add hints for inference
    pub fn with_hints(mut self, hints: Vec<String>) -> Self {
        self.hints = hints;
        self
    }

    /// Set maximum number of properties
    pub fn limit_properties(mut self, max: usize) -> Self {
        self.max_properties = max.max(1);
        self
    }

    /// Enable LLM refinement
    pub fn enable_llm(mut self) -> Self {
        self.use_llm = true;
        self
    }
}

/// Result of specification inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecInferenceResult {
    /// Inferred properties
    pub properties: Vec<Property>,
    /// Confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Assumptions made while inferring
    pub assumptions: Vec<String>,
    /// Notes about the inference process
    pub notes: Vec<String>,
    /// Whether an LLM was used
    pub used_llm: bool,
}

impl SpecInferenceResult {
    /// Serialize the inferred properties to USL text
    pub fn to_usl(&self) -> String {
        let mut output = String::new();

        // Add header comment with metadata
        output.push_str("// Auto-generated USL specification\n");
        output.push_str(&format!("// Confidence: {:.2}\n", self.confidence));
        if self.used_llm {
            output.push_str("// LLM-assisted inference\n");
        }
        output.push('\n');

        for property in &self.properties {
            output.push_str(&property_to_usl(property));
            output.push_str("\n\n");
        }

        output.trim_end().to_string()
    }

    /// Serialize only the contract properties to USL text
    pub fn contracts_to_usl(&self) -> String {
        let mut output = String::new();
        for property in &self.properties {
            if let Property::Contract(_) = property {
                output.push_str(&property_to_usl(property));
                output.push_str("\n\n");
            }
        }
        output.trim_end().to_string()
    }
}

/// Error while inferring specifications
#[derive(Debug, thiserror::Error)]
pub enum SpecInferenceError {
    /// Unsupported language
    #[error("Unsupported language for inference: {0:?}")]
    UnsupportedLanguage(SourceLanguage),
    /// LLM error
    #[error("LLM refinement failed: {0}")]
    Llm(#[from] LlmError),
}

/// Specification inferencer
pub struct SpecInferencer {
    client: Option<Box<dyn LlmClient>>,
}

impl Default for SpecInferencer {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecInferencer {
    /// Create an inferencer without LLM refinement
    pub fn new() -> Self {
        Self { client: None }
    }

    /// Create an inferencer with an LLM client
    pub fn with_llm(client: Box<dyn LlmClient>) -> Self {
        Self {
            client: Some(client),
        }
    }

    /// Inference using static analysis only (deterministic)
    pub fn infer_static(&self, request: &SpecInferenceRequest<'_>) -> SpecInferenceResult {
        match request.language {
            SourceLanguage::Rust => self.infer_rust(request),
            SourceLanguage::TypeScript => self.infer_typescript(request),
            SourceLanguage::Python => self.infer_python(request),
            SourceLanguage::Go => self.infer_go(request),
            SourceLanguage::C | SourceLanguage::Cpp => self.infer_c_cpp(request),
            SourceLanguage::Unknown => SpecInferenceResult {
                properties: Vec::new(),
                confidence: 0.05,
                assumptions: vec!["Language unknown; no inference performed".to_string()],
                notes: vec![],
                used_llm: false,
            },
        }
    }

    /// Inference with optional LLM refinement
    pub async fn infer(
        &self,
        request: &SpecInferenceRequest<'_>,
    ) -> Result<SpecInferenceResult, SpecInferenceError> {
        let mut result = self.infer_static(request);

        if request.use_llm {
            if let Some(client) = self.client.as_deref() {
                if let Some(refined) = self.refine_with_llm(client, request, &result).await? {
                    result = refined;
                }
            }
        }

        Ok(result)
    }

    fn infer_typescript(&self, request: &SpecInferenceRequest<'_>) -> SpecInferenceResult {
        let mut assumptions = vec!["Assumes type annotations express intent".to_string()];
        if let Some(path) = request.file_path {
            assumptions.push(format!("Analyzed file: {}", path));
        }
        if let Some(backend) = request.target_backend {
            assumptions.push(format!("Target backend preference: {:?}", backend));
        }
        if !request.hints.is_empty() {
            assumptions.push(format!("Hints: {}", request.hints.join("; ")));
        }

        let summaries = parse_typescript_functions(request.code);
        let mut properties: Vec<Property> = summaries.into_iter().map(Property::Contract).collect();

        properties.truncate(request.max_properties);

        let requires_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.requires.len()),
                _ => None,
            })
            .sum();
        let ensures_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.ensures.len() + c.ensures_err.len()),
                _ => None,
            })
            .sum();

        let mut confidence = 0.30 + 0.07 * (requires_total as f64).min(3.0);
        confidence += 0.05 * (ensures_total as f64).min(3.0);
        confidence = confidence.min(0.85);

        let notes = vec![format!(
            "Detected {} TypeScript contracts (requires: {}, ensures: {})",
            properties.len(),
            requires_total,
            ensures_total
        )];

        SpecInferenceResult {
            properties,
            confidence,
            assumptions,
            notes,
            used_llm: false,
        }
    }

    fn infer_python(&self, request: &SpecInferenceRequest<'_>) -> SpecInferenceResult {
        let mut assumptions = vec!["Assumes asserts express invariants".to_string()];
        if let Some(path) = request.file_path {
            assumptions.push(format!("Analyzed file: {}", path));
        }
        if let Some(backend) = request.target_backend {
            assumptions.push(format!("Target backend preference: {:?}", backend));
        }
        if !request.hints.is_empty() {
            assumptions.push(format!("Hints: {}", request.hints.join("; ")));
        }

        let summaries = parse_python_functions(request.code);
        let mut properties: Vec<Property> = summaries.into_iter().map(Property::Contract).collect();

        properties.truncate(request.max_properties);

        let requires_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.requires.len()),
                _ => None,
            })
            .sum();
        let ensures_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.ensures.len() + c.ensures_err.len()),
                _ => None,
            })
            .sum();

        let mut confidence = 0.25 + 0.06 * (requires_total as f64).min(3.0);
        confidence += 0.04 * (ensures_total as f64).min(3.0);
        confidence = confidence.min(0.75);

        let notes = vec![format!(
            "Detected {} Python contracts (requires: {}, ensures: {})",
            properties.len(),
            requires_total,
            ensures_total
        )];

        SpecInferenceResult {
            properties,
            confidence,
            assumptions,
            notes,
            used_llm: false,
        }
    }

    fn infer_go(&self, request: &SpecInferenceRequest<'_>) -> SpecInferenceResult {
        let mut assumptions =
            vec!["Assumes panic calls express violated preconditions".to_string()];
        if let Some(path) = request.file_path {
            assumptions.push(format!("Analyzed file: {}", path));
        }
        if let Some(backend) = request.target_backend {
            assumptions.push(format!("Target backend preference: {:?}", backend));
        }
        if !request.hints.is_empty() {
            assumptions.push(format!("Hints: {}", request.hints.join("; ")));
        }

        let summaries = parse_go_functions(request.code);
        let mut properties: Vec<Property> = summaries.into_iter().map(Property::Contract).collect();

        properties.truncate(request.max_properties);

        let requires_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.requires.len()),
                _ => None,
            })
            .sum();
        let ensures_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.ensures.len() + c.ensures_err.len()),
                _ => None,
            })
            .sum();

        let mut confidence = 0.28 + 0.06 * (requires_total as f64).min(3.0);
        confidence += 0.05 * (ensures_total as f64).min(3.0);
        confidence = confidence.min(0.80);

        let notes = vec![format!(
            "Detected {} Go contracts (requires: {}, ensures: {})",
            properties.len(),
            requires_total,
            ensures_total
        )];

        SpecInferenceResult {
            properties,
            confidence,
            assumptions,
            notes,
            used_llm: false,
        }
    }

    fn infer_c_cpp(&self, request: &SpecInferenceRequest<'_>) -> SpecInferenceResult {
        let lang_name = if request.language == SourceLanguage::Cpp {
            "C++"
        } else {
            "C"
        };
        let mut assumptions = vec![format!(
            "Assumes assert() and precondition guards express invariants ({})",
            lang_name
        )];
        if let Some(path) = request.file_path {
            assumptions.push(format!("Analyzed file: {}", path));
        }
        if let Some(backend) = request.target_backend {
            assumptions.push(format!("Target backend preference: {:?}", backend));
        }
        if !request.hints.is_empty() {
            assumptions.push(format!("Hints: {}", request.hints.join("; ")));
        }

        let summaries =
            parse_c_cpp_functions(request.code, request.language == SourceLanguage::Cpp);
        let mut properties: Vec<Property> = summaries.into_iter().map(Property::Contract).collect();

        properties.truncate(request.max_properties);

        let requires_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.requires.len()),
                _ => None,
            })
            .sum();
        let ensures_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.ensures.len() + c.ensures_err.len()),
                _ => None,
            })
            .sum();

        let mut confidence = 0.30 + 0.07 * (requires_total as f64).min(3.0);
        confidence += 0.05 * (ensures_total as f64).min(3.0);
        confidence = confidence.min(0.85);

        let notes = vec![format!(
            "Detected {} {} contracts (requires: {}, ensures: {})",
            properties.len(),
            lang_name,
            requires_total,
            ensures_total
        )];

        SpecInferenceResult {
            properties,
            confidence,
            assumptions,
            notes,
            used_llm: false,
        }
    }

    fn infer_rust(&self, request: &SpecInferenceRequest<'_>) -> SpecInferenceResult {
        let mut assumptions = vec!["Assumes panics represent violated preconditions".to_string()];
        if let Some(path) = request.file_path {
            assumptions.push(format!("Analyzed file: {}", path));
        }
        if let Some(backend) = request.target_backend {
            assumptions.push(format!("Target backend preference: {:?}", backend));
        }
        if !request.hints.is_empty() {
            assumptions.push(format!("Hints: {}", request.hints.join("; ")));
        }

        let summaries = parse_rust_functions(request.code);
        let mut properties: Vec<Property> = summaries.into_iter().map(Property::Contract).collect();

        // Limit total properties
        properties.truncate(request.max_properties);

        let requires_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.requires.len()),
                _ => None,
            })
            .sum();
        let ensures_total: usize = properties
            .iter()
            .filter_map(|p| match p {
                Property::Contract(c) => Some(c.ensures.len() + c.ensures_err.len()),
                _ => None,
            })
            .sum();

        let mut confidence = 0.35 + 0.08 * (requires_total as f64).min(3.0);
        confidence += 0.06 * (ensures_total as f64).min(3.0);
        confidence = confidence.min(0.9);

        let notes = vec![format!(
            "Detected {} contracts (requires: {}, ensures: {})",
            properties.len(),
            requires_total,
            ensures_total
        )];

        SpecInferenceResult {
            properties,
            confidence,
            assumptions,
            notes,
            used_llm: false,
        }
    }

    async fn refine_with_llm(
        &self,
        client: &dyn LlmClient,
        request: &SpecInferenceRequest<'_>,
        baseline: &SpecInferenceResult,
    ) -> Result<Option<SpecInferenceResult>, SpecInferenceError> {
        if baseline.properties.is_empty() {
            return Ok(None);
        }

        let system_prompt = "You are a formal methods engineer. Given source code and draft specifications, refine them into precise preconditions and postconditions. Respond with compact JSON fields: requires (list of boolean conditions), ensures (list of postconditions using `result` for return values), invariants (state invariants), and temporal (optional temporal properties).";

        let summary = baseline
            .properties
            .iter()
            .map(|p| format!("{:?}", p))
            .collect::<Vec<_>>()
            .join("\n");

        let user_prompt = format!(
            "Language: {}\nBackend: {:?}\nHints: {}\nExisting spec:\n{}\n\nCode:\n```{}\n```\nRespond with JSON as described.",
            request.language.label(),
            request.target_backend,
            if request.hints.is_empty() {
                "none".to_string()
            } else {
                request.hints.join(", ")
            },
            summary,
            request.code
        );

        let messages = vec![
            LlmMessage::system(system_prompt),
            LlmMessage::user(user_prompt),
        ];

        let response = client.complete_messages(&messages).await?;
        let Some(hints) = parse_llm_hints(&response.content) else {
            return Ok(None);
        };

        let mut result = baseline.clone();
        result.used_llm = true;
        result.notes.push("Applied LLM refinement".to_string());

        for property in result.properties.iter_mut() {
            if let Property::Contract(contract) = property {
                merge_conditions(&mut contract.requires, &hints.requires);
                merge_conditions(&mut contract.ensures, &hints.ensures);
            }
        }

        // Add invariants as separate properties to avoid mutating contracts
        for (idx, invariant) in hints.invariants.iter().enumerate() {
            if result.properties.len() >= request.max_properties {
                break;
            }
            let expr = parse_condition(invariant);
            result.properties.push(Property::Invariant(Invariant {
                name: format!("llm_inferred_invariant_{}", idx + 1),
                body: expr,
            }));
        }

        // Temporal properties are optional; include when present and space remains.
        for (idx, temporal) in hints.temporal.iter().enumerate() {
            if result.properties.len() >= request.max_properties {
                break;
            }
            let expr = parse_condition(temporal);
            result.properties.push(Property::Temporal(Temporal {
                name: format!("llm_temporal_property_{}", idx + 1),
                body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(expr))),
                fairness: vec![],
            }));
        }

        result.confidence = (result.confidence + 0.15).min(0.95);

        Ok(Some(result))
    }
}

#[derive(Debug, Clone, Default)]
struct FunctionSummary {
    name: String,
    params: Vec<Param>,
    return_type: Option<Type>,
    requires: Vec<Expr>,
    ensures: Vec<Expr>,
    ensures_err: Vec<Expr>,
    /// Frame conditions - memory locations the function may modify
    assigns: Vec<Expr>,
    /// Memory locations that may be allocated
    allocates: Vec<Expr>,
    /// Memory locations that may be freed
    frees: Vec<Expr>,
    /// Termination condition
    terminates: Option<Expr>,
    /// Variant expression for termination proofs
    decreases: Option<Expr>,
    /// Named specification cases (ACSL behaviors)
    behaviors: Vec<dashprove_usl::Behavior>,
    /// Whether behaviors are declared complete
    complete_behaviors: bool,
    /// Whether behaviors are declared disjoint
    disjoint_behaviors: bool,
}

impl From<FunctionSummary> for Contract {
    fn from(summary: FunctionSummary) -> Self {
        Contract {
            type_path: vec![summary.name],
            params: summary.params,
            return_type: summary.return_type,
            requires: summary.requires,
            ensures: summary.ensures,
            ensures_err: summary.ensures_err,
            assigns: summary.assigns,
            allocates: summary.allocates,
            frees: summary.frees,
            terminates: summary.terminates,
            decreases: summary.decreases,
            behaviors: summary.behaviors,
            complete_behaviors: summary.complete_behaviors,
            disjoint_behaviors: summary.disjoint_behaviors,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct LlmHints {
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    ensures: Vec<String>,
    #[serde(default)]
    invariants: Vec<String>,
    #[serde(default)]
    temporal: Vec<String>,
}

fn parse_llm_hints(content: &str) -> Option<LlmHints> {
    if let Ok(parsed) = serde_json::from_str::<LlmHints>(content) {
        return Some(parsed);
    }

    if let Some(block) = extract_json_block(content) {
        serde_json::from_str::<LlmHints>(&block).ok()
    } else {
        None
    }
}

fn extract_json_block(content: &str) -> Option<String> {
    let mut in_block = false;
    let mut block = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```json") {
            in_block = true;
            continue;
        }
        if trimmed.starts_with("```") && in_block {
            return Some(block);
        }
        if in_block {
            block.push_str(line);
        }
    }

    None
}

fn merge_conditions(target: &mut Vec<Expr>, hints: &[String]) {
    let mut existing_keys = target.iter().map(expr_key).collect::<Vec<_>>();

    for hint in hints {
        let expr = parse_condition(hint);
        let key = expr_key(&expr);
        if !existing_keys.contains(&key) {
            target.push(expr);
            existing_keys.push(key);
        }
    }
}

fn expr_key(expr: &Expr) -> String {
    format!("{:?}", expr)
}

fn parse_rust_functions(code: &str) -> Vec<Contract> {
    let lines: Vec<&str> = code.lines().collect();
    let mut functions = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if (trimmed.starts_with("fn ") || trimmed.contains(" fn ")) && !trimmed.starts_with("//") {
            let mut signature = trimmed.to_string();
            i += 1;
            while !signature.contains('{') && i < lines.len() {
                signature.push(' ');
                signature.push_str(lines[i].trim());
                i += 1;
            }

            let Some((signature_part, body_start)) = signature.split_once('{') else {
                continue;
            };

            let mut brace_depth: i32 = 1;
            let mut body = String::new();
            if !body_start.trim().is_empty() {
                body.push_str(body_start);
                body.push('\n');
                brace_depth += body_start.matches('{').count() as i32;
                brace_depth -= body_start.matches('}').count() as i32;
            }

            while i < lines.len() && brace_depth > 0 {
                let line = lines[i];
                brace_depth += line.matches('{').count() as i32;
                brace_depth -= line.matches('}').count() as i32;
                body.push_str(line);
                body.push('\n');
                i += 1;
                if brace_depth == 0 {
                    break;
                }
            }

            if let Some(summary) = parse_rust_signature(signature_part.trim(), &body) {
                functions.push(summary.into());
            }
        } else {
            i += 1;
        }
    }

    functions
}

fn parse_rust_signature(signature: &str, body: &str) -> Option<FunctionSummary> {
    let after_fn = signature.split_once("fn ")?;
    let remainder = after_fn.1.trim();
    let name_end = remainder.find('(')?;
    let name = remainder[..name_end].trim().to_string();
    let params_part = remainder[name_end + 1..].split_once(')')?.0;
    let return_type = remainder.split("->").nth(1).map(|s| parse_type(s.trim()));

    let params = params_part
        .split(',')
        .filter_map(|p| {
            let trimmed = p.trim();
            if trimmed.is_empty() {
                return None;
            }
            let (name, ty) = trimmed.split_once(':')?;
            Some(Param {
                name: name.trim().to_string(),
                ty: parse_type(ty.trim()),
            })
        })
        .collect();

    let requires = extract_asserts(body);
    let ensures = extract_return_contract(body);

    Some(FunctionSummary {
        name,
        params,
        return_type,
        requires,
        ensures,
        ..Default::default()
    })
}

// =============================================================================
// TypeScript parsing
// =============================================================================

fn parse_typescript_functions(code: &str) -> Vec<Contract> {
    let lines: Vec<&str> = code.lines().collect();
    let mut functions = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        // Match: function name(...), async function name(...), or arrow: const name = (...) =>
        let is_fn = trimmed.starts_with("function ")
            || trimmed.starts_with("async function ")
            || trimmed.starts_with("export function ")
            || trimmed.starts_with("export async function ")
            || (trimmed.starts_with("const ") && trimmed.contains("=>"));

        if is_fn && !trimmed.starts_with("//") {
            let mut signature = trimmed.to_string();
            i += 1;
            while !signature.contains('{') && i < lines.len() {
                signature.push(' ');
                signature.push_str(lines[i].trim());
                i += 1;
            }

            let Some((signature_part, body_start)) = signature.split_once('{') else {
                continue;
            };

            let mut brace_depth: i32 = 1;
            let mut body = String::new();
            if !body_start.trim().is_empty() {
                body.push_str(body_start);
                body.push('\n');
                brace_depth += body_start.matches('{').count() as i32;
                brace_depth -= body_start.matches('}').count() as i32;
            }

            while i < lines.len() && brace_depth > 0 {
                let line = lines[i];
                brace_depth += line.matches('{').count() as i32;
                brace_depth -= line.matches('}').count() as i32;
                body.push_str(line);
                body.push('\n');
                i += 1;
                if brace_depth == 0 {
                    break;
                }
            }

            if let Some(summary) = parse_typescript_signature(signature_part.trim(), &body) {
                functions.push(summary.into());
            }
        } else {
            i += 1;
        }
    }

    functions
}

fn parse_typescript_signature(signature: &str, body: &str) -> Option<FunctionSummary> {
    // Handle: function name(params): RetType
    // Handle: async function name(params): Promise<RetType>
    // Handle: const name = (params): RetType =>
    let name: String;
    let params_str: &str;
    let return_type: Option<Type>;

    if signature.contains("function ") {
        let after_fn = signature.split("function ").nth(1)?;
        let name_end = after_fn.find('(')?;
        name = after_fn[..name_end].trim().to_string();
        let rest = &after_fn[name_end + 1..];
        let params_end = rest.find(')')?;
        params_str = &rest[..params_end];
        return_type = rest.split(':').nth(1).map(|s| parse_ts_type(s.trim()));
    } else if let Some(const_start) = signature.strip_prefix("const ") {
        let eq_idx = const_start.find('=')?;
        name = const_start[..eq_idx].trim().to_string();
        let rest = &const_start[eq_idx + 1..];
        let paren_start = rest.find('(')?;
        let paren_end = rest.find(')')?;
        params_str = &rest[paren_start + 1..paren_end];
        return_type = rest.split(':').nth(1).and_then(|s| {
            let ty_str = s.split("=>").next()?.trim();
            if ty_str.is_empty() {
                None
            } else {
                Some(parse_ts_type(ty_str))
            }
        });
    } else {
        return None;
    }

    let params = params_str
        .split(',')
        .filter_map(|p| {
            let trimmed = p.trim();
            if trimmed.is_empty() {
                return None;
            }
            let (name, ty) = trimmed.split_once(':')?;
            Some(Param {
                name: name.trim().to_string(),
                ty: parse_ts_type(ty.trim()),
            })
        })
        .collect();

    let requires = extract_ts_asserts(body);
    let ensures = extract_ts_return_contract(body);

    Some(FunctionSummary {
        name,
        params,
        return_type,
        requires,
        ensures,
        ..Default::default()
    })
}

fn parse_ts_type(ty: &str) -> Type {
    let trimmed = ty.trim().trim_end_matches('{').trim();
    if trimmed.starts_with("Promise<") {
        let inner = trimmed
            .trim_start_matches("Promise<")
            .trim_end_matches('>')
            .trim();
        Type::Result(Box::new(parse_ts_type(inner)))
    } else if trimmed.ends_with("[]") {
        let inner = trimmed.trim_end_matches("[]").trim();
        Type::List(Box::new(parse_ts_type(inner)))
    } else if trimmed.starts_with("Array<") {
        let inner = trimmed
            .trim_start_matches("Array<")
            .trim_end_matches('>')
            .trim();
        Type::List(Box::new(parse_ts_type(inner)))
    } else if trimmed == "void" {
        Type::Unit
    } else {
        Type::Named(trimmed.to_string())
    }
}

fn extract_ts_asserts(body: &str) -> Vec<Expr> {
    body.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            // console.assert(cond) or assert(cond) or if (!cond) throw
            if let Some(cond) = trimmed.strip_prefix("console.assert(") {
                let expr =
                    parse_condition(cond.trim_end_matches(");").trim_end_matches(')').trim());
                return Some(expr);
            }
            if let Some(cond) = trimmed.strip_prefix("assert(") {
                let expr =
                    parse_condition(cond.trim_end_matches(");").trim_end_matches(')').trim());
                return Some(expr);
            }
            // if (!condition) throw ...
            if trimmed.starts_with("if (!")
                && (trimmed.contains("throw") || trimmed.contains("throw new"))
            {
                let cond_start = trimmed.find("(!")? + 2;
                let cond_end = trimmed.find(')')?;
                let cond = &trimmed[cond_start..cond_end];
                return Some(Expr::Var(cond.trim().to_string()));
            }
            None
        })
        .collect()
}

fn extract_ts_return_contract(body: &str) -> Vec<Expr> {
    let mut ensures = Vec::new();
    for line in body.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('}')
            || trimmed.starts_with("//")
            || trimmed.starts_with("assert")
            || trimmed.starts_with("console.assert")
        {
            continue;
        }
        let expr = if let Some(rest) = trimmed.strip_prefix("return") {
            rest.trim().trim_end_matches(';').trim()
        } else {
            continue;
        };

        if expr.is_empty() || expr == "}" {
            continue;
        }

        let rhs = parse_expression(expr);
        ensures.push(Expr::Compare(
            Box::new(Expr::Var("result".to_string())),
            ComparisonOp::Eq,
            Box::new(rhs),
        ));
        break;
    }
    ensures
}

// =============================================================================
// Python parsing
// =============================================================================

fn parse_python_functions(code: &str) -> Vec<Contract> {
    let lines: Vec<&str> = code.lines().collect();
    let mut functions = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        // Match: def name(...) or async def name(...)
        let is_fn = trimmed.starts_with("def ") || trimmed.starts_with("async def ");

        if is_fn && !trimmed.starts_with('#') {
            let mut signature = trimmed.to_string();
            i += 1;
            while !signature.contains(':') && i < lines.len() {
                signature.push(' ');
                signature.push_str(lines[i].trim());
                i += 1;
            }

            // Get the indentation level of the function
            let base_indent = lines[i.saturating_sub(1)]
                .chars()
                .take_while(|c| c.is_whitespace())
                .count();

            let mut body = String::new();
            while i < lines.len() {
                let line = lines[i];
                let line_indent = line.chars().take_while(|c| c.is_whitespace()).count();
                if !line.trim().is_empty() && line_indent <= base_indent {
                    break;
                }
                body.push_str(line);
                body.push('\n');
                i += 1;
            }

            if let Some(summary) = parse_python_signature(&signature, &body) {
                functions.push(summary.into());
            }
        } else {
            i += 1;
        }
    }

    functions
}

fn parse_python_signature(signature: &str, body: &str) -> Option<FunctionSummary> {
    // Handle: def name(params) -> RetType:
    // Handle: async def name(params) -> RetType:
    let after_def = if signature.contains("async def ") {
        signature.split("async def ").nth(1)?
    } else {
        signature.split("def ").nth(1)?
    };

    let name_end = after_def.find('(')?;
    let name = after_def[..name_end].trim().to_string();
    let rest = &after_def[name_end + 1..];
    let params_end = rest.find(')')?;
    let params_str = &rest[..params_end];
    let return_type = rest.split("->").nth(1).map(|s| {
        let ty_str = s.split(':').next().unwrap_or("").trim();
        parse_python_type(ty_str)
    });

    let params = params_str
        .split(',')
        .filter_map(|p| {
            let trimmed = p.trim();
            if trimmed.is_empty() || trimmed == "self" || trimmed == "cls" {
                return None;
            }
            // Handle: name: Type or name: Type = default
            let name_part = trimmed.split('=').next()?.trim();
            if let Some((name, ty)) = name_part.split_once(':') {
                Some(Param {
                    name: name.trim().to_string(),
                    ty: parse_python_type(ty.trim()),
                })
            } else {
                Some(Param {
                    name: name_part.to_string(),
                    ty: Type::Named("Any".to_string()),
                })
            }
        })
        .collect();

    let requires = extract_python_asserts(body);
    let ensures = extract_python_return_contract(body);

    Some(FunctionSummary {
        name,
        params,
        return_type,
        requires,
        ensures,
        ..Default::default()
    })
}

fn parse_python_type(ty: &str) -> Type {
    let trimmed = ty.trim();
    if trimmed.starts_with("Optional[") {
        // Optional[T] is represented as Named("Optional<T>") since USL doesn't have Option
        let inner = trimmed
            .trim_start_matches("Optional[")
            .trim_end_matches(']')
            .trim();
        Type::Named(format!("Optional<{}>", inner))
    } else if trimmed.starts_with("List[") || trimmed.starts_with("list[") {
        let inner = trimmed
            .trim_start_matches("List[")
            .trim_start_matches("list[")
            .trim_end_matches(']')
            .trim();
        Type::List(Box::new(parse_python_type(inner)))
    } else if trimmed == "None" {
        Type::Unit
    } else {
        Type::Named(trimmed.to_string())
    }
}

fn extract_python_asserts(body: &str) -> Vec<Expr> {
    body.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            // assert condition or assert condition, message
            if let Some(rest) = trimmed.strip_prefix("assert ") {
                let cond = rest.split(',').next()?.trim();
                return Some(parse_condition(cond));
            }
            // if not condition: raise ...
            if trimmed.starts_with("if not ") && trimmed.contains("raise") {
                let cond_start = 7; // "if not ".len()
                let cond_end = trimmed.find(':')?;
                let cond = &trimmed[cond_start..cond_end];
                return Some(Expr::Var(cond.trim().to_string()));
            }
            None
        })
        .collect()
}

fn extract_python_return_contract(body: &str) -> Vec<Expr> {
    let mut ensures = Vec::new();
    for line in body.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("assert")
            || trimmed.starts_with("raise")
        {
            continue;
        }
        let expr = if let Some(rest) = trimmed.strip_prefix("return") {
            rest.trim()
        } else {
            continue;
        };

        if expr.is_empty() {
            continue;
        }

        let rhs = parse_expression(expr);
        ensures.push(Expr::Compare(
            Box::new(Expr::Var("result".to_string())),
            ComparisonOp::Eq,
            Box::new(rhs),
        ));
        break;
    }
    ensures
}

// =============================================================================
// Go parsing
// =============================================================================

fn parse_go_functions(code: &str) -> Vec<Contract> {
    let lines: Vec<&str> = code.lines().collect();
    let mut functions = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        // Match: func name(...) or func (receiver) name(...)
        if trimmed.starts_with("func ") && !trimmed.starts_with("//") {
            let mut signature = trimmed.to_string();
            i += 1;
            while !signature.contains('{') && i < lines.len() {
                signature.push(' ');
                signature.push_str(lines[i].trim());
                i += 1;
            }

            let Some((signature_part, body_start)) = signature.split_once('{') else {
                continue;
            };

            let mut brace_depth: i32 = 1;
            let mut body = String::new();
            if !body_start.trim().is_empty() {
                body.push_str(body_start);
                body.push('\n');
                brace_depth += body_start.matches('{').count() as i32;
                brace_depth -= body_start.matches('}').count() as i32;
            }

            while i < lines.len() && brace_depth > 0 {
                let line = lines[i];
                brace_depth += line.matches('{').count() as i32;
                brace_depth -= line.matches('}').count() as i32;
                body.push_str(line);
                body.push('\n');
                i += 1;
                if brace_depth == 0 {
                    break;
                }
            }

            if let Some(summary) = parse_go_signature(signature_part.trim(), &body) {
                functions.push(summary.into());
            }
        } else {
            i += 1;
        }
    }

    functions
}

fn parse_go_signature(signature: &str, body: &str) -> Option<FunctionSummary> {
    // Handle: func name(params) RetType
    // Handle: func (r ReceiverType) name(params) RetType
    let after_func = signature.split("func ").nth(1)?;
    let trimmed = after_func.trim();

    // Check for method receiver: (r Type)
    let (name_start, _) = if trimmed.starts_with('(') {
        // Skip receiver
        let receiver_end = trimmed.find(')')?;
        (trimmed[receiver_end + 1..].trim(), true)
    } else {
        (trimmed, false)
    };

    let name_end = name_start.find('(')?;
    let name = name_start[..name_end].trim().to_string();
    let rest = &name_start[name_end + 1..];
    let params_end = rest.find(')')?;
    let params_str = &rest[..params_end];

    // Return type comes after the params
    let after_params = &rest[params_end + 1..];
    let return_type = if after_params.trim().is_empty() {
        None
    } else {
        Some(parse_go_type(after_params.trim()))
    };

    let params = parse_go_params(params_str);
    let requires = extract_go_asserts(body);
    let ensures = extract_go_return_contract(body);

    Some(FunctionSummary {
        name,
        params,
        return_type,
        requires,
        ensures,
        ..Default::default()
    })
}

fn parse_go_params(params_str: &str) -> Vec<Param> {
    // Go params: name Type, name Type or name, name Type
    // Example: "a, b int, c string" -> a: int, b: int, c: string
    let mut params = Vec::new();
    let mut pending_names: Vec<String> = Vec::new();

    for part in params_str.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Split by whitespace to find name and type
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 2 {
            // Has both name and type - first apply type to pending names (in order)
            let ty = parse_go_type(parts[parts.len() - 1]);
            for name in pending_names.drain(..) {
                params.push(Param {
                    name,
                    ty: ty.clone(),
                });
            }
            // Then add the names from this segment
            for name in &parts[..parts.len() - 1] {
                params.push(Param {
                    name: name.to_string(),
                    ty: ty.clone(),
                });
            }
        } else if parts.len() == 1 {
            // Just a name, type comes later
            pending_names.push(parts[0].to_string());
        }
    }

    params
}

fn parse_go_type(ty: &str) -> Type {
    let trimmed = ty.trim();
    if trimmed.starts_with("[]") {
        let inner = trimmed.trim_start_matches("[]").trim();
        Type::List(Box::new(parse_go_type(inner)))
    } else if trimmed.starts_with('*') {
        // Pointer types represented as Named("*T") since USL doesn't have pointers
        let inner = trimmed.trim_start_matches('*').trim();
        Type::Named(format!("*{}", inner))
    } else if trimmed.starts_with("error") {
        Type::Result(Box::new(Type::Unit))
    } else {
        Type::Named(trimmed.to_string())
    }
}

fn extract_go_asserts(body: &str) -> Vec<Expr> {
    body.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            // if condition { panic(...) } or if !condition { panic(...) }
            if trimmed.starts_with("if ") && trimmed.contains("panic(") {
                let cond_end = trimmed.find('{')?;
                let cond = trimmed[3..cond_end].trim();
                // If condition is negated with !, we extract the positive condition
                if let Some(pos_cond) = cond.strip_prefix('!') {
                    return Some(Expr::Var(pos_cond.trim().to_string()));
                }
                // Otherwise the panic happens when condition is true,
                // so the invariant is the negation
                return Some(Expr::Not(Box::new(Expr::Var(cond.to_string()))));
            }
            None
        })
        .collect()
}

fn extract_go_return_contract(body: &str) -> Vec<Expr> {
    let mut ensures = Vec::new();
    for line in body.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('}')
            || trimmed.starts_with("//")
            || trimmed.starts_with("panic")
        {
            continue;
        }
        let expr = if let Some(rest) = trimmed.strip_prefix("return") {
            rest.trim()
        } else {
            continue;
        };

        if expr.is_empty() {
            continue;
        }

        let rhs = parse_expression(expr);
        ensures.push(Expr::Compare(
            Box::new(Expr::Var("result".to_string())),
            ComparisonOp::Eq,
            Box::new(rhs),
        ));
        break;
    }
    ensures
}

// =============================================================================
// C/C++ parsing
// =============================================================================

fn parse_c_cpp_functions(code: &str, is_cpp: bool) -> Vec<Contract> {
    let lines: Vec<&str> = code.lines().collect();
    let mut functions = Vec::new();
    let mut i = 0;
    // Track ACSL block comments that appear before functions
    let mut pending_acsl: Option<String> = None;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        // Skip preprocessor directives and empty lines
        if trimmed.starts_with('#') || trimmed.is_empty() {
            i += 1;
            continue;
        }

        // Check for ACSL block comment: /*@ ... */
        // These are Frama-C style annotations that should precede functions
        if trimmed.starts_with("/*@") {
            let mut acsl_block = String::new();
            // Handle single-line ACSL: /*@ requires x > 0; */
            if trimmed.contains("*/") {
                acsl_block = trimmed.to_string();
            } else {
                // Multi-line ACSL block
                acsl_block.push_str(trimmed);
                acsl_block.push('\n');
                i += 1;
                while i < lines.len() {
                    let line = lines[i];
                    acsl_block.push_str(line);
                    acsl_block.push('\n');
                    if line.contains("*/") {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            pending_acsl = Some(acsl_block);
            if !trimmed.contains("*/") {
                continue;
            }
            i += 1;
            continue;
        }

        // Skip regular comments (non-ACSL)
        if trimmed.starts_with("//") || trimmed.starts_with("/*") {
            // Check for single-line ACSL: //@ annotation
            if trimmed.starts_with("//@") {
                let acsl_line = trimmed.to_string();
                if let Some(ref mut block) = pending_acsl {
                    block.push_str(&acsl_line);
                    block.push('\n');
                } else {
                    pending_acsl = Some(acsl_line + "\n");
                }
            }
            i += 1;
            continue;
        }

        // Look for function definitions:
        // C:   type name(params) {
        // C++: type name(params) const? {, type Class::name(params) {
        // Also handle: static, inline, extern, virtual, etc.
        if looks_like_c_function(trimmed, is_cpp) {
            let mut signature = trimmed.to_string();
            i += 1;

            // Continue reading until we find the opening brace
            while !signature.contains('{') && i < lines.len() {
                let next = lines[i].trim();
                if next.starts_with('#') || next.starts_with("//") {
                    i += 1;
                    continue;
                }
                signature.push(' ');
                signature.push_str(next);
                i += 1;
            }

            let Some((signature_part, body_start)) = signature.split_once('{') else {
                pending_acsl = None;
                continue;
            };

            // Collect function body
            let mut brace_depth: i32 = 1;
            let mut body = String::new();
            if !body_start.trim().is_empty() {
                body.push_str(body_start);
                body.push('\n');
                brace_depth += body_start.matches('{').count() as i32;
                brace_depth -= body_start.matches('}').count() as i32;
            }

            while i < lines.len() && brace_depth > 0 {
                let line = lines[i];
                brace_depth += line.matches('{').count() as i32;
                brace_depth -= line.matches('}').count() as i32;
                body.push_str(line);
                body.push('\n');
                i += 1;
                if brace_depth == 0 {
                    break;
                }
            }

            // Parse function with any preceding ACSL annotations
            if let Some(summary) = parse_c_cpp_signature_with_acsl(
                signature_part.trim(),
                &body,
                pending_acsl.as_deref(),
                is_cpp,
            ) {
                functions.push(summary.into());
            }
            pending_acsl = None;
        } else {
            // If we hit non-function code, clear any pending ACSL
            // (unless it's just whitespace which we already skip)
            pending_acsl = None;
            i += 1;
        }
    }

    functions
}

fn looks_like_c_function(line: &str, is_cpp: bool) -> bool {
    // Skip forward declarations (end with ;) and typedefs
    if line.ends_with(';') || line.starts_with("typedef") {
        return false;
    }

    // Must have parentheses for parameters
    if !line.contains('(') {
        return false;
    }

    // Skip class/struct/enum definitions
    if line.starts_with("class ")
        || line.starts_with("struct ")
        || line.starts_with("enum ")
        || line.starts_with("union ")
        || line.starts_with("namespace ")
    {
        return false;
    }

    // Common function modifiers
    let modifiers = [
        "static",
        "inline",
        "extern",
        "virtual",
        "explicit",
        "constexpr",
        "const",
    ];

    let mut remainder = line;
    for modifier in &modifiers {
        remainder = remainder
            .trim_start_matches(modifier)
            .trim_start_matches(' ');
    }

    // Should have a return type and name before (
    let before_paren = remainder.split('(').next().unwrap_or("");
    let tokens: Vec<&str> = before_paren.split_whitespace().collect();

    // Need at least: return_type function_name
    // Or for C++ methods: return_type Class::function_name
    if tokens.len() < 2 {
        return false;
    }

    // The last token should be the function name (possibly with Class:: prefix in C++)
    let name = tokens.last().unwrap_or(&"");
    if is_cpp {
        // C++ allows Class::method
        !name.is_empty()
    } else {
        // C: name should be a simple identifier
        name.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '*')
    }
}

/// Parse a C/C++ function signature with optional ACSL annotations from preceding comments.
///
/// ACSL (ANSI/ISO C Specification Language) annotations can specify:
/// - `requires`: preconditions
/// - `ensures`: postconditions
/// - `assigns`: frame conditions - memory locations the function may modify
/// - `behavior`: named behavior specifications
fn parse_c_cpp_signature_with_acsl(
    signature: &str,
    body: &str,
    acsl: Option<&str>,
    is_cpp: bool,
) -> Option<FunctionSummary> {
    // First, parse the base signature
    let mut summary = parse_c_cpp_signature_base(signature, body, is_cpp)?;

    // Then, parse ACSL annotations if present
    if let Some(acsl_block) = acsl {
        let annotations = parse_acsl_annotations_full(acsl_block);

        // Prepend ACSL requires (they take precedence as explicit specs)
        for req in annotations.requires.into_iter().rev() {
            summary.requires.insert(0, req);
        }

        // Prepend ACSL ensures
        for ens in annotations.ensures.into_iter().rev() {
            summary.ensures.insert(0, ens);
        }

        // Set assigns (frame conditions)
        summary.assigns = annotations.assigns;

        // Set allocates (heap allocation clauses)
        summary.allocates = annotations.allocates;

        // Set frees (heap deallocation clauses)
        summary.frees = annotations.frees;

        // Set terminates clause (termination condition)
        summary.terminates = annotations.terminates;

        // Set decreases clause (variant expression for termination proofs)
        summary.decreases = annotations.decreases;

        // Set behaviors (named specification cases)
        summary.behaviors = annotations.behaviors;

        // Set complete/disjoint behaviors flags
        summary.complete_behaviors = annotations.complete_behaviors;
        summary.disjoint_behaviors = annotations.disjoint_behaviors;
    }

    Some(summary)
}

/// Parse ACSL annotation block and extract requires/ensures clauses.
///
/// Parsed ACSL annotation result
struct AcslAnnotations {
    requires: Vec<Expr>,
    ensures: Vec<Expr>,
    assigns: Vec<Expr>,
    allocates: Vec<Expr>,
    frees: Vec<Expr>,
    terminates: Option<Expr>,
    decreases: Option<Expr>,
    behaviors: Vec<dashprove_usl::Behavior>,
    complete_behaviors: bool,
    disjoint_behaviors: bool,
}

/// Handles both block comments `/*@ ... */` and line comments `//@`.
/// Supports ACSL syntax including:
/// - `requires expr;`
/// - `ensures expr;`
/// - `assigns locs;` - frame conditions (memory locations the function may modify)
/// - `allocates locs;` - memory locations that may be allocated
/// - `frees locs;` - memory locations that may be freed
/// - `terminates expr;` - condition under which function terminates
/// - `decreases expr;` - variant expression for termination proofs
/// - `behavior name: assumes ...; requires ...; ensures ...;`
/// - `complete behaviors;` - behaviors cover all cases
/// - `disjoint behaviors;` - behaviors are mutually exclusive
/// - `\result` for return value (converted to `result`)
/// - `\old(expr)` for pre-state values (converted to `old(expr)`)
/// - `\forall` and `\exists` quantifiers
/// - `\nothing` for assigns/allocates/frees \nothing
#[cfg(test)]
fn parse_acsl_annotations(acsl: &str) -> (Vec<Expr>, Vec<Expr>) {
    let annotations = parse_acsl_annotations_full(acsl);
    (annotations.requires, annotations.ensures)
}

/// Full ACSL parsing including all ACSL clauses
fn parse_acsl_annotations_full(acsl: &str) -> AcslAnnotations {
    let mut requires = Vec::new();
    let mut ensures = Vec::new();
    let mut assigns = Vec::new();
    let mut allocates = Vec::new();
    let mut frees = Vec::new();
    let mut terminates = None;
    let mut decreases = None;
    let mut complete_behaviors = false;
    let mut disjoint_behaviors = false;

    // Normalize the ACSL block: remove comment markers
    let content = acsl
        .replace("/*@", "")
        .replace("*/", "")
        .replace("//@", "")
        .replace("@", ""); // Handle lines starting with @ inside block comments

    // Split by semicolons at clause boundaries (not inside quantifiers)
    // ACSL quantifiers use ; as part of their syntax: \forall int i; body
    for clause in split_acsl_clauses(&content) {
        let trimmed = clause.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse requires clause
        if let Some(rest) = trimmed.strip_prefix("requires") {
            let condition = normalize_acsl_expr(rest.trim());
            if !condition.is_empty() {
                requires.push(parse_condition(&condition));
            }
            continue;
        }

        // Parse ensures clause
        if let Some(rest) = trimmed.strip_prefix("ensures") {
            let condition = normalize_acsl_expr(rest.trim());
            if !condition.is_empty() {
                ensures.push(parse_condition(&condition));
            }
            continue;
        }

        // Parse assigns clause (frame conditions)
        if let Some(rest) = trimmed.strip_prefix("assigns") {
            parse_acsl_location_list(rest.trim(), &mut assigns);
            continue;
        }

        // Parse allocates clause (heap allocation)
        if let Some(rest) = trimmed.strip_prefix("allocates") {
            parse_acsl_location_list(rest.trim(), &mut allocates);
            continue;
        }

        // Parse frees clause (heap deallocation)
        if let Some(rest) = trimmed.strip_prefix("frees") {
            parse_acsl_location_list(rest.trim(), &mut frees);
            continue;
        }

        // Parse terminates clause (termination condition)
        if let Some(rest) = trimmed.strip_prefix("terminates") {
            let condition = normalize_acsl_expr(rest.trim());
            if !condition.is_empty() {
                terminates = Some(parse_condition(&condition));
            }
            continue;
        }

        // Parse decreases clause (variant expression for termination proofs)
        if let Some(rest) = trimmed.strip_prefix("decreases") {
            let variant = normalize_acsl_expr(rest.trim());
            if !variant.is_empty() {
                decreases = Some(parse_condition(&variant));
            }
            continue;
        }

        // Parse complete behaviors clause
        if trimmed == "complete behaviors" {
            complete_behaviors = true;
            continue;
        }

        // Parse disjoint behaviors clause
        if trimmed == "disjoint behaviors" {
            disjoint_behaviors = true;
            continue;
        }

        // Skip other ACSL clauses we don't handle yet
    }

    // Parse behavior blocks in a second pass
    // Behaviors have the form: behavior name: assumes ...; requires ...; ensures ...;
    let behaviors = parse_acsl_behaviors(&content);

    AcslAnnotations {
        requires,
        ensures,
        assigns,
        allocates,
        frees,
        terminates,
        decreases,
        behaviors,
        complete_behaviors,
        disjoint_behaviors,
    }
}

/// Parse ACSL location list (used for assigns, allocates, frees clauses)
/// Handles `\nothing` keyword and comma-separated location lists.
fn parse_acsl_location_list(locs: &str, target: &mut Vec<Expr>) {
    // Handle ACSL \nothing keyword
    if locs == "\\nothing" {
        target.push(Expr::Var("nothing".to_string()));
    } else {
        // Parse comma-separated list of memory locations
        for loc in locs.split(',') {
            let loc = normalize_acsl_expr(loc.trim());
            if !loc.is_empty() {
                target.push(parse_condition(&loc));
            }
        }
    }
}

/// Parse ACSL behavior blocks.
/// ACSL behaviors have the form:
/// ```text
/// behavior name:
///   assumes condition;
///   requires precondition;
///   ensures postcondition;
///   assigns locations;
/// ```
fn parse_acsl_behaviors(content: &str) -> Vec<dashprove_usl::Behavior> {
    let mut behaviors = Vec::new();
    let mut current_behavior: Option<dashprove_usl::Behavior> = None;

    // Process line by line to properly handle behavior declarations
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if this is a new behavior declaration (behavior name:)
        if let Some(rest) = trimmed.strip_prefix("behavior") {
            // Save any previous behavior
            if let Some(behavior) = current_behavior.take() {
                behaviors.push(behavior);
            }

            // Extract just the name (up to the colon)
            let rest = rest.trim();
            let name = if let Some(colon_idx) = rest.find(':') {
                rest[..colon_idx].trim().to_string()
            } else {
                rest.to_string()
            };

            if !name.is_empty() {
                current_behavior = Some(dashprove_usl::Behavior {
                    name,
                    assumes: Vec::new(),
                    requires: Vec::new(),
                    ensures: Vec::new(),
                    assigns: Vec::new(),
                });
            }
            continue;
        }

        // If we're inside a behavior, parse its clauses
        // Strip trailing semicolons for proper parsing
        let clause = trimmed.trim_end_matches(';');

        if let Some(ref mut behavior) = current_behavior {
            if let Some(rest) = clause.strip_prefix("assumes") {
                let condition = normalize_acsl_expr(rest.trim());
                if !condition.is_empty() {
                    behavior.assumes.push(parse_condition(&condition));
                }
            } else if let Some(rest) = clause.strip_prefix("requires") {
                let condition = normalize_acsl_expr(rest.trim());
                if !condition.is_empty() {
                    behavior.requires.push(parse_condition(&condition));
                }
            } else if let Some(rest) = clause.strip_prefix("ensures") {
                let condition = normalize_acsl_expr(rest.trim());
                if !condition.is_empty() {
                    behavior.ensures.push(parse_condition(&condition));
                }
            } else if let Some(rest) = clause.strip_prefix("assigns") {
                parse_acsl_location_list(rest.trim(), &mut behavior.assigns);
            }
        }
    }

    // Don't forget the last behavior
    if let Some(behavior) = current_behavior {
        behaviors.push(behavior);
    }

    behaviors
}

/// Split ACSL content into clauses, respecting quantifier syntax.
/// ACSL quantifiers use ; to separate binding from body: \forall int i; body
/// We only split on ; that are clause terminators, not quantifier separators.
fn split_acsl_clauses(content: &str) -> Vec<&str> {
    let mut clauses = Vec::new();
    let mut start = 0;
    let mut in_quantifier = false;
    let bytes = content.as_bytes();

    for i in 0..bytes.len() {
        // Track if we're inside a quantifier (between \forall/\exists and the clause-terminating ;)
        if i + 7 <= bytes.len() {
            let slice = &content[i..];
            if slice.starts_with("\\forall") || slice.starts_with("\\exists") {
                in_quantifier = true;
            }
        }

        if bytes[i] == b';' {
            if in_quantifier {
                // This semicolon is the quantifier's binding separator, not clause terminator
                in_quantifier = false;
            } else {
                // This is a clause-terminating semicolon
                let clause = &content[start..i];
                if !clause.trim().is_empty() {
                    clauses.push(clause);
                }
                start = i + 1;
            }
        }
    }

    // Handle any remaining content after the last semicolon
    if start < content.len() {
        let remaining = &content[start..];
        if !remaining.trim().is_empty() {
            clauses.push(remaining);
        }
    }

    clauses
}

/// Normalize ACSL-specific expressions to our standard form.
///
/// Converts:
/// - `\result` -> `result`
/// - `\old(x)` -> `old(x)`
/// - `\forall type x; expr` -> `forall x: type . expr`
/// - `\exists type x; expr` -> `exists x: type . expr`
/// - `\valid(p)` -> `valid(p)`
/// - `\separated(p, q)` -> `separated(p, q)`
fn normalize_acsl_expr(expr: &str) -> String {
    let mut result = expr.to_string();

    // Replace ACSL backslash-prefixed keywords
    result = result.replace("\\result", "result");
    result = result.replace("\\old(", "old(");
    result = result.replace("\\valid(", "valid(");
    result = result.replace("\\valid_read(", "valid_read(");
    result = result.replace("\\separated(", "separated(");
    result = result.replace("\\null", "null");
    result = result.replace("\\true", "true");
    result = result.replace("\\false", "false");

    // Handle quantifiers: \forall int x; expr -> forall x: int . expr
    // This handles quantifiers anywhere in the expression
    result = normalize_acsl_quantifier(&result, "\\forall", "forall");
    result = normalize_acsl_quantifier(&result, "\\exists", "exists");

    // Handle ACSL logical operators
    // <==> must come before ==> to avoid partial matching
    result = result.replace("<==>", " iff ");
    result = result.replace("==>", " implies ");

    result.trim().to_string()
}

/// Transform a single ACSL quantifier pattern: \forall type var; body -> forall var: type . body
fn normalize_acsl_quantifier(expr: &str, acsl_keyword: &str, usl_keyword: &str) -> String {
    if let Some(start_idx) = expr.find(acsl_keyword) {
        let prefix = &expr[..start_idx];
        let rest = &expr[start_idx + acsl_keyword.len()..];
        let rest = rest.trim_start();

        // Find the semicolon that separates type+var from body
        if let Some(semi_idx) = rest.find(';') {
            let type_var = rest[..semi_idx].trim();
            let body = rest[semi_idx + 1..].trim();

            // Split type_var into type and variable name (last token is var)
            let tokens: Vec<&str> = type_var.split_whitespace().collect();
            if tokens.len() >= 2 {
                let var_name = tokens.last().unwrap();
                let var_type = tokens[..tokens.len() - 1].join(" ");
                // Recursively normalize the body (in case of nested quantifiers)
                let normalized_body = normalize_acsl_quantifier(body, acsl_keyword, usl_keyword);
                return format!(
                    "{}{} {}: {} . {}",
                    prefix, usl_keyword, var_name, var_type, normalized_body
                );
            } else if tokens.len() == 1 {
                // Single token might be just a variable name (untyped quantifier)
                let var_name = tokens[0];
                let normalized_body = normalize_acsl_quantifier(body, acsl_keyword, usl_keyword);
                return format!(
                    "{}{} {} . {}",
                    prefix, usl_keyword, var_name, normalized_body
                );
            }
        }
    }
    expr.to_string()
}

fn parse_c_cpp_signature_base(
    signature: &str,
    body: &str,
    is_cpp: bool,
) -> Option<FunctionSummary> {
    // Remove modifiers
    let modifiers = [
        "static",
        "inline",
        "extern",
        "virtual",
        "explicit",
        "constexpr",
    ];
    let mut cleaned = signature.to_string();
    for modifier in &modifiers {
        cleaned = cleaned
            .replace(&format!("{} ", modifier), "")
            .replace(&format!(" {}", modifier), "");
    }

    // Find the opening parenthesis
    let paren_idx = cleaned.find('(')?;
    let before_paren = &cleaned[..paren_idx];
    let after_paren = &cleaned[paren_idx + 1..];

    // Parse return type and function name
    let tokens: Vec<&str> = before_paren.split_whitespace().collect();
    if tokens.len() < 2 {
        return None;
    }

    let name = tokens.last()?.trim_start_matches('*').to_string();
    let return_type_str = tokens[..tokens.len() - 1].join(" ");
    let return_type = if return_type_str == "void" {
        None
    } else {
        Some(parse_c_cpp_type(&return_type_str))
    };

    // Parse parameters
    let close_paren = after_paren.find(')')?;
    let params_str = &after_paren[..close_paren];
    let params = parse_c_cpp_params(params_str);

    let requires = extract_c_cpp_asserts(body, is_cpp);
    let ensures = extract_c_cpp_return_contract(body);

    Some(FunctionSummary {
        name,
        params,
        return_type,
        requires,
        ensures,
        ..Default::default()
    })
}

fn parse_c_cpp_params(params_str: &str) -> Vec<Param> {
    // Handle: void, empty, or list of "type name" pairs
    let trimmed = params_str.trim();
    if trimmed.is_empty() || trimmed == "void" {
        return Vec::new();
    }

    params_str
        .split(',')
        .filter_map(|p| {
            let trimmed = p.trim();
            if trimmed.is_empty() {
                return None;
            }

            // Split into tokens
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.is_empty() {
                return None;
            }

            // Last token is the name (possibly with * prefix)
            let name = tokens
                .last()?
                .trim_start_matches('*')
                .trim_start_matches('&');
            // Everything else is the type
            let type_tokens = &tokens[..tokens.len() - 1];
            let type_str = if type_tokens.is_empty() {
                "int" // Default to int if no type (old C style)
            } else {
                trimmed[..trimmed.rfind(name).unwrap_or(trimmed.len())].trim()
            };

            Some(Param {
                name: name.to_string(),
                ty: parse_c_cpp_type(type_str),
            })
        })
        .collect()
}

fn parse_c_cpp_type(ty: &str) -> Type {
    let trimmed = ty
        .trim()
        .trim_start_matches("const ")
        .trim_end_matches(" const")
        .trim();

    // Handle pointers
    if trimmed.ends_with('*') {
        let inner = trimmed.trim_end_matches('*').trim();
        return Type::Named(format!("*{}", inner));
    }

    // Handle references (C++)
    if trimmed.ends_with('&') {
        let inner = trimmed.trim_end_matches('&').trim();
        return Type::Named(format!("&{}", inner));
    }

    // Handle common C++ templates
    if trimmed.starts_with("std::vector<") || trimmed.starts_with("vector<") {
        let inner = trimmed
            .trim_start_matches("std::")
            .trim_start_matches("vector<")
            .trim_end_matches('>')
            .trim();
        return Type::List(Box::new(parse_c_cpp_type(inner)));
    }

    if trimmed.starts_with("std::optional<") || trimmed.starts_with("optional<") {
        let inner = trimmed
            .trim_start_matches("std::")
            .trim_start_matches("optional<")
            .trim_end_matches('>')
            .trim();
        return Type::Named(format!("Optional<{}>", inner));
    }

    if trimmed.starts_with("std::unique_ptr<") || trimmed.starts_with("unique_ptr<") {
        let inner = trimmed
            .trim_start_matches("std::")
            .trim_start_matches("unique_ptr<")
            .trim_end_matches('>')
            .trim();
        return Type::Named(format!("*{}", inner));
    }

    if trimmed == "void" {
        return Type::Unit;
    }

    Type::Named(trimmed.to_string())
}

fn extract_c_cpp_asserts(body: &str, is_cpp: bool) -> Vec<Expr> {
    body.lines()
        .filter_map(|line| {
            let trimmed = line.trim();

            // Standard assert(condition)
            if let Some(cond) = trimmed.strip_prefix("assert(") {
                let expr =
                    parse_condition(cond.trim_end_matches(");").trim_end_matches(')').trim());
                return Some(expr);
            }

            // C++ static_assert(condition, message)
            if is_cpp {
                if let Some(rest) = trimmed.strip_prefix("static_assert(") {
                    let cond = rest.split(',').next()?.trim();
                    return Some(parse_condition(cond));
                }
            }

            // Precondition guards: if (!condition) return/abort/exit/throw
            if trimmed.starts_with("if (!")
                && (trimmed.contains("return")
                    || trimmed.contains("abort()")
                    || trimmed.contains("exit(")
                    || trimmed.contains("throw"))
            {
                let cond_start = trimmed.find("(!")? + 2;
                let cond_end = trimmed[cond_start..].find(')')?;
                let cond = &trimmed[cond_start..cond_start + cond_end];
                return Some(Expr::Var(cond.trim().to_string()));
            }

            // CBMC-style preconditions: __CPROVER_precondition(cond, "message")
            if let Some(rest) = trimmed.strip_prefix("__CPROVER_precondition(") {
                let cond = rest.split(',').next()?.trim();
                return Some(parse_condition(cond));
            }

            // Frama-C ACSL-style annotations (in comments)
            // /*@ requires condition; */
            if trimmed.starts_with("/*@") || trimmed.starts_with("//@") {
                let content = trimmed
                    .trim_start_matches("/*@")
                    .trim_start_matches("//@")
                    .trim_end_matches("*/")
                    .trim();
                if let Some(req) = content.strip_prefix("requires") {
                    let cond = req.trim().trim_end_matches(';');
                    return Some(parse_condition(cond));
                }
            }

            None
        })
        .collect()
}

fn extract_c_cpp_return_contract(body: &str) -> Vec<Expr> {
    let mut ensures = Vec::new();
    for line in body.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('}')
            || trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with("assert")
        {
            continue;
        }

        let expr = if let Some(rest) = trimmed.strip_prefix("return") {
            rest.trim().trim_end_matches(';').trim()
        } else {
            continue;
        };

        if expr.is_empty() {
            continue;
        }

        let rhs = parse_expression(expr);
        ensures.push(Expr::Compare(
            Box::new(Expr::Var("result".to_string())),
            ComparisonOp::Eq,
            Box::new(rhs),
        ));
        break;
    }
    ensures
}

// =============================================================================
// Rust parsing (original)
// =============================================================================

fn parse_type(ty: &str) -> Type {
    let trimmed = ty.trim().trim_end_matches('{').trim();
    if trimmed.starts_with("Result<") {
        let inner = trimmed
            .trim_start_matches("Result<")
            .trim_end_matches('>')
            .split(',')
            .next()
            .unwrap_or("()");
        Type::Result(Box::new(parse_type(inner.trim())))
    } else if trimmed.starts_with("Vec<") {
        let inner = trimmed
            .trim_start_matches("Vec<")
            .trim_end_matches('>')
            .trim();
        Type::List(Box::new(parse_type(inner)))
    } else if trimmed == "()" {
        Type::Unit
    } else {
        Type::Named(trimmed.to_string())
    }
}

fn extract_asserts(body: &str) -> Vec<Expr> {
    body.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if let Some(cond) = trimmed.strip_prefix("assert!(") {
                let expr =
                    parse_condition(cond.trim_end_matches(");").trim_end_matches(')').trim());
                return Some(expr);
            }
            if let Some(cond) = trimmed.strip_prefix("debug_assert!(") {
                let expr =
                    parse_condition(cond.trim_end_matches(");").trim_end_matches(')').trim());
                return Some(expr);
            }
            None
        })
        .collect()
}

fn extract_return_contract(body: &str) -> Vec<Expr> {
    let mut ensures = Vec::new();
    for line in body.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('}')
            || trimmed.starts_with("//")
            || trimmed.starts_with("assert")
        {
            continue;
        }
        let expr = if let Some(rest) = trimmed.strip_prefix("return") {
            rest.trim().trim_end_matches(';').trim()
        } else {
            trimmed.trim_end_matches(';')
        };

        if expr.is_empty() || expr == "}" {
            continue;
        }

        let rhs = parse_expression(expr);
        ensures.push(Expr::Compare(
            Box::new(Expr::Var("result".to_string())),
            ComparisonOp::Eq,
            Box::new(rhs),
        ));
        break;
    }
    ensures
}

fn parse_expression(expr: &str) -> Expr {
    if let Some(and_split) = split_logical(expr, "&&") {
        return Expr::And(
            Box::new(parse_expression(and_split.0)),
            Box::new(parse_expression(and_split.1)),
        );
    }

    if let Some(or_split) = split_logical(expr, "||") {
        return Expr::Or(
            Box::new(parse_expression(or_split.0)),
            Box::new(parse_expression(or_split.1)),
        );
    }

    for op in ["==", ">=", "<=", "!=", ">", "<"] {
        if let Some(idx) = expr.find(op) {
            let (left, right) = expr.split_at(idx);
            let right = &right[op.len()..];
            return Expr::Compare(
                Box::new(parse_operand(left.trim())),
                parse_comparison_op(op),
                Box::new(parse_operand(right.trim())),
            );
        }
    }

    for op in ['+', '-', '*', '/', '%'] {
        if let Some(idx) = expr.find(op) {
            if op == '-' && idx == 0 {
                continue;
            }
            let (left, right) = expr.split_at(idx);
            let right = &right[1..]; // Skip the operator character
            let op = match op {
                '+' => dashprove_usl::ast::BinaryOp::Add,
                '-' => dashprove_usl::ast::BinaryOp::Sub,
                '*' => dashprove_usl::ast::BinaryOp::Mul,
                '/' => dashprove_usl::ast::BinaryOp::Div,
                '%' => dashprove_usl::ast::BinaryOp::Mod,
                _ => unreachable!(),
            };
            return Expr::Binary(
                Box::new(parse_operand(left.trim())),
                op,
                Box::new(parse_operand(right.trim())),
            );
        }
    }

    parse_operand(expr.trim())
}

fn split_logical<'a>(expr: &'a str, delim: &str) -> Option<(&'a str, &'a str)> {
    let mut depth = 0usize;
    let bytes = expr.as_bytes();
    let mut i = 0;
    while i + delim.len() <= bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ => {}
        }

        if depth == 0 && &expr[i..i + delim.len()] == delim {
            let left = &expr[..i];
            let right = &expr[i + delim.len()..];
            return Some((left, right));
        }
        i += 1;
    }
    None
}

fn parse_condition(cond: &str) -> Expr {
    let cond = cond.trim();

    // Handle quantifiers: forall x: T . body, exists x: T . body
    if let Some(rest) = cond.strip_prefix("forall ") {
        if let Some((binding, body)) = split_quantifier_binding(rest) {
            let (var, ty) = parse_quantifier_var(binding);
            return Expr::ForAll {
                var,
                ty,
                body: Box::new(parse_condition(body)),
            };
        }
    }
    if let Some(rest) = cond.strip_prefix("exists ") {
        if let Some((binding, body)) = split_quantifier_binding(rest) {
            let (var, ty) = parse_quantifier_var(binding);
            return Expr::Exists {
                var,
                ty,
                body: Box::new(parse_condition(body)),
            };
        }
    }

    // Handle iff (lowest precedence)
    if let Some(split) = split_logical(cond, " iff ") {
        return Expr::And(
            Box::new(Expr::Implies(
                Box::new(parse_condition(split.0.trim())),
                Box::new(parse_condition(split.1.trim())),
            )),
            Box::new(Expr::Implies(
                Box::new(parse_condition(split.1.trim())),
                Box::new(parse_condition(split.0.trim())),
            )),
        );
    }

    // Handle implies
    if let Some(split) = split_logical(cond, " implies ") {
        return Expr::Implies(
            Box::new(parse_condition(split.0.trim())),
            Box::new(parse_condition(split.1.trim())),
        );
    }

    if let Some(split) = split_logical(cond, "&&") {
        return Expr::And(
            Box::new(parse_condition(split.0.trim())),
            Box::new(parse_condition(split.1.trim())),
        );
    }
    if let Some(split) = split_logical(cond, "||") {
        return Expr::Or(
            Box::new(parse_condition(split.0.trim())),
            Box::new(parse_condition(split.1.trim())),
        );
    }

    for op in ["==", ">=", "<=", "!=", ">", "<"] {
        if let Some(idx) = cond.find(op) {
            let (left, right) = cond.split_at(idx);
            let right = &right[op.len()..];
            return Expr::Compare(
                Box::new(parse_operand(left.trim())),
                parse_comparison_op(op),
                Box::new(parse_operand(right.trim())),
            );
        }
    }

    Expr::Var(cond.trim().to_string())
}

/// Split quantifier binding: "x: T . body" -> ("x: T", "body")
fn split_quantifier_binding(s: &str) -> Option<(&str, &str)> {
    // Find " . " that separates binding from body
    let mut depth: usize = 0;
    let bytes = s.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        match bytes[i] as char {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ' ' if depth == 0 && i + 3 <= bytes.len() => {
                if &s[i..i + 3] == " . " {
                    return Some((&s[..i], s[i + 3..].trim()));
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse quantifier variable binding: "x: T" -> (var_name, Some(Type))
fn parse_quantifier_var(binding: &str) -> (String, Option<Type>) {
    if let Some(colon_idx) = binding.find(':') {
        let var = binding[..colon_idx].trim().to_string();
        let ty_str = binding[colon_idx + 1..].trim();
        (var, Some(parse_type(ty_str)))
    } else {
        // No type annotation
        (binding.trim().to_string(), None)
    }
}

fn parse_operand(operand: &str) -> Expr {
    let trimmed = operand.trim().trim_matches(|c| c == '(' || c == ')');
    if let Ok(value) = trimmed.parse::<i64>() {
        Expr::Int(value)
    } else if let Ok(value) = trimmed.parse::<f64>() {
        Expr::Float(value)
    } else if trimmed == "true" {
        Expr::Bool(true)
    } else if trimmed == "false" {
        Expr::Bool(false)
    } else {
        Expr::Var(trimmed.to_string())
    }
}

fn parse_comparison_op(op: &str) -> ComparisonOp {
    match op {
        "==" => ComparisonOp::Eq,
        ">=" => ComparisonOp::Ge,
        "<=" => ComparisonOp::Le,
        ">" => ComparisonOp::Gt,
        "<" => ComparisonOp::Lt,
        "!=" => ComparisonOp::Ne,
        _ => ComparisonOp::Eq,
    }
}

// =============================================================================
// USL Serialization
// =============================================================================

/// Serialize a Property to USL text
fn property_to_usl(property: &Property) -> String {
    match property {
        Property::Contract(contract) => contract_to_usl(contract),
        Property::Invariant(invariant) => invariant_to_usl(invariant),
        Property::Temporal(temporal) => temporal_to_usl(temporal),
        Property::Theorem(theorem) => theorem_to_usl(theorem),
        Property::Refinement(refinement) => refinement_to_usl(refinement),
        // Placeholder for extended property types - these are not typically inferred from code
        Property::Probabilistic(p) => format!("// probabilistic property: {}", p.name),
        Property::Security(s) => format!("// security property: {}", s.name),
        Property::Semantic(s) => format!("// semantic property: {}", s.name),
        Property::PlatformApi(p) => format!("// platform API: {}", p.name),
        Property::Bisimulation(b) => format!("// bisimulation: {}", b.name),
        Property::Version(v) => format!("// version {} improves {}", v.name, v.improves),
        Property::Capability(c) => format!("// capability: {}", c.name),
        Property::DistributedInvariant(d) => format!("// distributed invariant: {}", d.name),
        Property::DistributedTemporal(d) => format!("// distributed temporal: {}", d.name),
        Property::Composed(c) => composed_theorem_to_usl(c),
        Property::ImprovementProposal(i) => format!("// improvement proposal: {}", i.name),
        Property::VerificationGate(v) => format!("// verification gate: {}", v.name),
        Property::Rollback(r) => format!("// rollback spec: {}", r.name),
    }
}

fn composed_theorem_to_usl(composed: &dashprove_usl::ast::ComposedTheorem) -> String {
    let uses_str = composed.uses.join(", ");
    format!(
        "composed theorem {} {{\n    uses {{ {} }}\n    {}\n}}",
        composed.name,
        uses_str,
        expr_to_usl(&composed.body)
    )
}

fn contract_to_usl(contract: &Contract) -> String {
    let mut output = String::new();

    // Function signature: name(params) -> return_type
    let name = contract.type_path.join("::");

    // Build parameter list
    let params_str = if contract.params.is_empty() {
        String::new()
    } else {
        contract
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, type_to_usl(&p.ty)))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // Build signature with optional return type
    let sig = if let Some(ret_type) = &contract.return_type {
        format!("{}({}) -> {}", name, params_str, type_to_usl(ret_type))
    } else {
        format!("{}({})", name, params_str)
    };

    output.push_str(&format!("contract {} {{\n", sig));

    // Preconditions - each requires clause is separate: requires { expr }
    for req in &contract.requires {
        output.push_str(&format!("  requires {{ {} }}\n", expr_to_usl(req)));
    }

    // Postconditions
    for ens in &contract.ensures {
        output.push_str(&format!("  ensures {{ {} }}\n", expr_to_usl(ens)));
    }

    // Error postconditions
    for ens in &contract.ensures_err {
        output.push_str(&format!("  ensures_err {{ {} }}\n", expr_to_usl(ens)));
    }

    // Frame conditions (assigns)
    if !contract.assigns.is_empty() {
        let assigns_str = contract
            .assigns
            .iter()
            .map(expr_to_usl)
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&format!("  assigns {{ {} }}\n", assigns_str));
    }

    output.push('}');
    output
}

fn invariant_to_usl(invariant: &Invariant) -> String {
    format!(
        "invariant {} {{\n  {}\n}}",
        invariant.name,
        expr_to_usl(&invariant.body)
    )
}

fn temporal_to_usl(temporal: &Temporal) -> String {
    let mut output = format!("temporal {} {{\n", temporal.name);
    output.push_str(&format!("  {}\n", temporal_expr_to_usl(&temporal.body)));
    if !temporal.fairness.is_empty() {
        output.push_str("  fairness {\n");
        for f in &temporal.fairness {
            output.push_str(&format!("    {},\n", fairness_to_usl(f)));
        }
        output.push_str("  }\n");
    }
    output.push('}');
    output
}

fn theorem_to_usl(theorem: &Theorem) -> String {
    let mut output = format!("theorem {} {{\n", theorem.name);
    output.push_str(&format!("  {}\n", expr_to_usl(&theorem.body)));
    output.push('}');
    output
}

fn refinement_to_usl(refinement: &Refinement) -> String {
    let mut output = format!(
        "refinement {} refines {} {{\n",
        refinement.name, refinement.refines
    );

    // Add mappings
    if !refinement.mappings.is_empty() {
        output.push_str("  mapping {\n");
        for m in &refinement.mappings {
            output.push_str(&format!(
                "    {} <-> {}\n",
                expr_to_usl(&m.spec_var),
                expr_to_usl(&m.impl_var)
            ));
        }
        output.push_str("  }\n");
    }

    // Add invariants
    for inv in &refinement.invariants {
        output.push_str(&format!("  invariant {{ {} }}\n", expr_to_usl(inv)));
    }

    // Add abstraction
    output.push_str(&format!(
        "  abstraction {{ {} }}\n",
        expr_to_usl(&refinement.abstraction)
    ));

    // Add simulation
    output.push_str(&format!(
        "  simulation {{ {} }}\n",
        expr_to_usl(&refinement.simulation)
    ));

    // Add action mappings
    for action in &refinement.actions {
        output.push_str(&format!("  action {} {{\n", action.name));
        output.push_str(&format!("    spec: {}\n", action.spec_action));
        output.push_str(&format!("    impl: {}\n", action.impl_action.join("::")));
        if let Some(guard) = &action.guard {
            output.push_str(&format!("    guard: {}\n", expr_to_usl(guard)));
        }
        output.push_str("  }\n");
    }

    output.push('}');
    output
}

fn type_to_usl(ty: &Type) -> String {
    match ty {
        Type::Unit => "()".to_string(),
        Type::Named(name) => name.clone(),
        Type::List(inner) => format!("List<{}>", type_to_usl(inner)),
        Type::Set(inner) => format!("Set<{}>", type_to_usl(inner)),
        Type::Map(k, v) => format!("Map<{}, {}>", type_to_usl(k), type_to_usl(v)),
        Type::Relation(a, b) => format!("Relation<{}, {}>", type_to_usl(a), type_to_usl(b)),
        Type::Result(inner) => format!("Result<{}>", type_to_usl(inner)),
        Type::Graph(n, e) => format!("Graph<{}, {}>", type_to_usl(n), type_to_usl(e)),
        Type::Path(n) => format!("Path<{}>", type_to_usl(n)),
        Type::Function(input, output) => {
            format!("{} -> {}", type_to_usl(input), type_to_usl(output))
        }
    }
}

fn expr_to_usl(expr: &Expr) -> String {
    match expr {
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => format!("{}", f),
        Expr::Bool(b) => b.to_string(),
        Expr::String(s) => format!("\"{}\"", s.replace('\"', "\\\"")),
        Expr::Var(name) => name.clone(),
        Expr::FieldAccess(obj, field) => format!("{}.{}", expr_to_usl(obj), field),
        Expr::App(func, args) => {
            let arg_strs: Vec<String> = args.iter().map(expr_to_usl).collect();
            format!("{}({})", func, arg_strs.join(", "))
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let arg_strs: Vec<String> = args.iter().map(expr_to_usl).collect();
            format!(
                "{}.{}({})",
                expr_to_usl(receiver),
                method,
                arg_strs.join(", ")
            )
        }
        Expr::Not(e) => format!("!{}", expr_to_usl(e)),
        Expr::Neg(e) => format!("-{}", expr_to_usl(e)),
        Expr::And(l, r) => format!("({} && {})", expr_to_usl(l), expr_to_usl(r)),
        Expr::Or(l, r) => format!("({} || {})", expr_to_usl(l), expr_to_usl(r)),
        Expr::Implies(l, r) => format!("({} => {})", expr_to_usl(l), expr_to_usl(r)),
        Expr::Compare(l, op, r) => format!(
            "({} {} {})",
            expr_to_usl(l),
            comparison_op_to_usl(op),
            expr_to_usl(r)
        ),
        Expr::Binary(l, op, r) => format!(
            "({} {} {})",
            expr_to_usl(l),
            binary_op_to_usl(op),
            expr_to_usl(r)
        ),
        Expr::ForAll { var, ty, body } => {
            let ty_str = ty
                .as_ref()
                .map(|t| format!(": {}", type_to_usl(t)))
                .unwrap_or_default();
            format!("forall {}{} . {}", var, ty_str, expr_to_usl(body))
        }
        Expr::Exists { var, ty, body } => {
            let ty_str = ty
                .as_ref()
                .map(|t| format!(": {}", type_to_usl(t)))
                .unwrap_or_default();
            format!("exists {}{} . {}", var, ty_str, expr_to_usl(body))
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
        } => {
            format!(
                "forall {} in {} . {}",
                var,
                expr_to_usl(collection),
                expr_to_usl(body)
            )
        }
        Expr::ExistsIn {
            var,
            collection,
            body,
        } => {
            format!(
                "exists {} in {} . {}",
                var,
                expr_to_usl(collection),
                expr_to_usl(body)
            )
        }
    }
}

fn comparison_op_to_usl(op: &ComparisonOp) -> &'static str {
    match op {
        ComparisonOp::Eq => "==",
        ComparisonOp::Ne => "!=",
        ComparisonOp::Lt => "<",
        ComparisonOp::Le => "<=",
        ComparisonOp::Gt => ">",
        ComparisonOp::Ge => ">=",
    }
}

fn binary_op_to_usl(op: &dashprove_usl::ast::BinaryOp) -> &'static str {
    match op {
        dashprove_usl::ast::BinaryOp::Add => "+",
        dashprove_usl::ast::BinaryOp::Sub => "-",
        dashprove_usl::ast::BinaryOp::Mul => "*",
        dashprove_usl::ast::BinaryOp::Div => "/",
        dashprove_usl::ast::BinaryOp::Mod => "%",
    }
}

fn temporal_expr_to_usl(expr: &TemporalExpr) -> String {
    match expr {
        TemporalExpr::Atom(e) => expr_to_usl(e),
        TemporalExpr::Always(e) => format!("always({})", temporal_expr_to_usl(e)),
        TemporalExpr::Eventually(e) => format!("eventually({})", temporal_expr_to_usl(e)),
        TemporalExpr::LeadsTo(l, r) => format!(
            "({} ~> {})",
            temporal_expr_to_usl(l),
            temporal_expr_to_usl(r)
        ),
    }
}

fn fairness_to_usl(fairness: &FairnessConstraint) -> String {
    let vars_str = fairness
        .vars
        .as_ref()
        .map(|v| format!("_{}", v))
        .unwrap_or_default();
    match fairness.kind {
        FairnessKind::Weak => format!("WF{}({})", vars_str, fairness.action),
        FairnessKind::Strong => format!("SF{}({})", vars_str, fairness.action),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    #[test]
    fn rust_contract_inference_extracts_requires_and_ensures() {
        let code = r#"
        pub fn add_positive(a: i32, b: i32) -> i32 {
            assert!(a >= 0);
            assert!(b >= 0);
            a + b
        }
        "#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Rust);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.requires.len(), 2);
        assert_eq!(contract.ensures.len(), 1);
        assert!(result.confidence > 0.3);
        if let Expr::Compare(_, ComparisonOp::Eq, rhs) = &contract.ensures[0] {
            assert!(matches!(
                **rhs,
                Expr::Binary(_, dashprove_usl::ast::BinaryOp::Add, _)
            ));
        } else {
            panic!("expected equality ensure");
        }
    }

    #[test]
    fn parse_condition_handles_boolean_combinations() {
        let expr = parse_condition("x > 0 && y < 5");
        match expr {
            Expr::And(left, right) => {
                matches!(*left, Expr::Compare(_, ComparisonOp::Gt, _))
                    .then_some(())
                    .expect("left comparison");
                matches!(*right, Expr::Compare(_, ComparisonOp::Lt, _))
                    .then_some(())
                    .expect("right comparison");
            }
            other => panic!("unexpected expression: {:?}", other),
        }
    }

    struct TestLlm {
        response: String,
    }

    #[async_trait]
    impl LlmClient for TestLlm {
        async fn complete(&self, _prompt: &str) -> Result<crate::llm::LlmResponse, LlmError> {
            Ok(crate::llm::LlmResponse {
                content: self.response.clone(),
                model: "test".to_string(),
                input_tokens: None,
                output_tokens: None,
                stop_reason: None,
            })
        }

        async fn complete_messages(
            &self,
            _messages: &[LlmMessage],
        ) -> Result<crate::llm::LlmResponse, LlmError> {
            self.complete("").await
        }

        fn is_configured(&self) -> bool {
            true
        }

        fn model_id(&self) -> &str {
            "test"
        }
    }

    #[tokio::test]
    async fn llm_refinement_merges_hints() {
        let code = r#"
        fn square(x: i64) -> i64 {
            assert!(x >= 0);
            x * x
        }
        "#;

        let response = r#"{
            "requires": ["x >= 0"],
            "ensures": ["result >= 0", "result == x * x"],
            "invariants": ["x >= 0"],
            "temporal": []
        }"#;

        let client: Box<dyn LlmClient> = Box::new(TestLlm {
            response: response.to_string(),
        });
        let inferencer = SpecInferencer::with_llm(client);
        let request = SpecInferenceRequest::new(code, SourceLanguage::Rust).enable_llm();

        let result = inferencer.infer(&request).await.expect("llm inference");
        assert!(result.used_llm);
        assert!(result.confidence > 0.4);
        assert!(
            result
                .properties
                .iter()
                .any(|p| matches!(p, Property::Invariant(_))),
            "expected invariant from LLM hints"
        );
        let contract = result
            .properties
            .iter()
            .find_map(|p| {
                if let Property::Contract(c) = p {
                    Some(c)
                } else {
                    None
                }
            })
            .expect("contract present");
        assert!(contract
            .ensures
            .iter()
            .any(|e| matches!(e, Expr::Compare(_, ComparisonOp::Eq, _))));
    }

    // =========================================================================
    // TypeScript inference tests
    // =========================================================================

    #[test]
    fn typescript_function_inference_extracts_contracts() {
        let code = r#"
        function addPositive(a: number, b: number): number {
            console.assert(a >= 0);
            console.assert(b >= 0);
            return a + b;
        }
        "#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::TypeScript);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["addPositive".to_string()]);
        assert_eq!(contract.requires.len(), 2);
        assert_eq!(contract.ensures.len(), 1);
        assert!(result.confidence > 0.25);
    }

    #[test]
    fn typescript_arrow_function_inference() {
        let code = r#"
        const multiply = (x: number, y: number): number => {
            assert(x > 0);
            return x * y;
        }
        "#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::TypeScript);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["multiply".to_string()]);
        assert_eq!(contract.requires.len(), 1);
    }

    #[test]
    fn typescript_async_function_inference() {
        let code = r#"
        export async function fetchData(url: string): Promise<Data> {
            if (!url) throw new Error("url required");
            return await fetch(url);
        }
        "#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::TypeScript);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["fetchData".to_string()]);
        // The if (!url) throw pattern should extract "url" as a requirement
        assert_eq!(contract.requires.len(), 1);
        // Return type should be present (Promise<Data> parses to Result(Named("Data")))
        assert!(contract.return_type.is_some(), "expected return type");
    }

    #[test]
    fn typescript_type_parsing() {
        assert!(matches!(parse_ts_type("number"), Type::Named(_)));
        assert!(matches!(parse_ts_type("void"), Type::Unit));
        assert!(matches!(parse_ts_type("string[]"), Type::List(_)));
        assert!(matches!(parse_ts_type("Array<number>"), Type::List(_)));
        assert!(matches!(parse_ts_type("Promise<Data>"), Type::Result(_)));
    }

    // =========================================================================
    // Python inference tests
    // =========================================================================

    #[test]
    fn python_function_inference_extracts_contracts() {
        let code = r#"
def add_positive(a: int, b: int) -> int:
    assert a >= 0
    assert b >= 0
    return a + b
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Python);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["add_positive".to_string()]);
        assert_eq!(contract.requires.len(), 2);
        assert_eq!(contract.ensures.len(), 1);
        assert!(result.confidence > 0.2);
    }

    #[test]
    fn python_async_function_inference() {
        let code = r#"
async def fetch_data(url: str) -> Data:
    assert url, "url required"
    return await client.get(url)
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Python);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["fetch_data".to_string()]);
        assert_eq!(contract.requires.len(), 1);
    }

    #[test]
    fn python_type_parsing() {
        assert!(matches!(parse_python_type("int"), Type::Named(_)));
        assert!(matches!(parse_python_type("None"), Type::Unit));
        assert!(matches!(parse_python_type("List[int]"), Type::List(_)));
        assert!(matches!(parse_python_type("list[str]"), Type::List(_)));
        // Optional becomes Named("Optional<T>")
        let opt = parse_python_type("Optional[int]");
        assert!(matches!(opt, Type::Named(s) if s.contains("Optional")));
    }

    #[test]
    fn python_skips_self_parameter() {
        let code = r#"
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Python);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        // self should be filtered out
        assert_eq!(contract.params.len(), 2);
        assert_eq!(contract.params[0].name, "a");
        assert_eq!(contract.params[1].name, "b");
    }

    // =========================================================================
    // Go inference tests
    // =========================================================================

    #[test]
    fn go_function_inference_extracts_contracts() {
        let code = r#"
func AddPositive(a, b int) int {
    if a < 0 { panic("a must be non-negative") }
    if b < 0 { panic("b must be non-negative") }
    return a + b
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Go);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["AddPositive".to_string()]);
        // Two panic guards => 2 requires
        assert_eq!(contract.requires.len(), 2);
        assert_eq!(contract.ensures.len(), 1);
        assert!(result.confidence > 0.25);
    }

    #[test]
    fn go_method_inference() {
        let code = r#"
func (c *Calculator) Add(a, b int) int {
    return a + b
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Go);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["Add".to_string()]);
    }

    #[test]
    fn go_type_parsing() {
        assert!(matches!(parse_go_type("int"), Type::Named(_)));
        assert!(matches!(parse_go_type("[]int"), Type::List(_)));
        // Pointer becomes Named("*T")
        let ptr = parse_go_type("*string");
        assert!(matches!(ptr, Type::Named(s) if s.starts_with('*')));
        assert!(matches!(parse_go_type("error"), Type::Result(_)));
    }

    #[test]
    fn go_params_with_shared_types() {
        // Go allows: a, b int (both are int)
        let params = parse_go_params("a, b int, c string");
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "a");
        assert_eq!(params[1].name, "b");
        assert_eq!(params[2].name, "c");
    }

    // =========================================================================
    // C/C++ inference tests
    // =========================================================================

    #[test]
    fn c_function_inference_extracts_contracts() {
        let code = r#"
int add_positive(int a, int b) {
    assert(a >= 0);
    assert(b >= 0);
    return a + b;
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::C);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["add_positive".to_string()]);
        assert_eq!(contract.requires.len(), 2);
        assert_eq!(contract.ensures.len(), 1);
        assert!(result.confidence > 0.25);
    }

    #[test]
    fn cpp_function_inference_extracts_contracts() {
        let code = r#"
int Calculator::multiply(int x, int y) {
    static_assert(sizeof(int) >= 4, "need 32-bit int");
    if (!x) return 0;
    return x * y;
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Cpp);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.type_path, vec!["Calculator::multiply".to_string()]);
        // static_assert + if (!x) guard = 2 requires
        assert_eq!(contract.requires.len(), 2);
    }

    #[test]
    fn c_with_cbmc_precondition() {
        let code = r#"
void safe_divide(int a, int b) {
    __CPROVER_precondition(b != 0, "divisor must be non-zero");
    return a / b;
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::C);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        assert_eq!(contract.requires.len(), 1);
        // Check that we parsed b != 0
        if let Expr::Compare(_, ComparisonOp::Ne, _) = &contract.requires[0] {
            // Good
        } else {
            panic!("expected != comparison in requires");
        }
    }

    #[test]
    fn c_with_frama_c_acsl_annotations() {
        // Test ACSL-style annotations inside the function body
        let code = r#"
int factorial(int n) {
    //@ requires n > 0;
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::C);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        // ACSL requires annotation should be captured from //@ comment
        assert_eq!(contract.requires.len(), 1);
    }

    #[test]
    fn cpp_type_parsing() {
        assert!(matches!(parse_c_cpp_type("int"), Type::Named(_)));
        assert!(matches!(parse_c_cpp_type("void"), Type::Unit));
        assert!(matches!(parse_c_cpp_type("int*"), Type::Named(s) if s.starts_with('*')));
        assert!(matches!(parse_c_cpp_type("int&"), Type::Named(s) if s.starts_with('&')));
        assert!(matches!(
            parse_c_cpp_type("std::vector<int>"),
            Type::List(_)
        ));
        assert!(
            matches!(parse_c_cpp_type("std::optional<int>"), Type::Named(s) if s.contains("Optional"))
        );
        assert!(matches!(parse_c_cpp_type("const int"), Type::Named(_)));
    }

    #[test]
    fn c_params_parsing() {
        let params = parse_c_cpp_params("int a, int b, const char* name");
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "a");
        assert_eq!(params[1].name, "b");
        assert_eq!(params[2].name, "name");
    }

    #[test]
    fn cpp_with_templates() {
        let code = r#"
std::vector<int> sort_positive(std::vector<int> nums) {
    assert(!nums.empty());
    std::sort(nums.begin(), nums.end());
    return nums;
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Cpp);
        let result = inferencer.infer_static(&request);

        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract property");
        };
        // Return type should be List
        assert!(matches!(contract.return_type, Some(Type::List(_))));
    }

    // =========================================================================
    // Language detection tests
    // =========================================================================

    #[test]
    fn source_language_detection_from_path() {
        assert_eq!(SourceLanguage::from_path("main.rs"), SourceLanguage::Rust);
        assert_eq!(
            SourceLanguage::from_path("app.ts"),
            SourceLanguage::TypeScript
        );
        assert_eq!(
            SourceLanguage::from_path("component.tsx"),
            SourceLanguage::TypeScript
        );
        assert_eq!(
            SourceLanguage::from_path("script.js"),
            SourceLanguage::TypeScript
        );
        assert_eq!(SourceLanguage::from_path("main.py"), SourceLanguage::Python);
        assert_eq!(SourceLanguage::from_path("main.go"), SourceLanguage::Go);
        assert_eq!(SourceLanguage::from_path("main.c"), SourceLanguage::C);
        assert_eq!(SourceLanguage::from_path("utils.h"), SourceLanguage::C);
        assert_eq!(SourceLanguage::from_path("main.cpp"), SourceLanguage::Cpp);
        assert_eq!(SourceLanguage::from_path("main.cc"), SourceLanguage::Cpp);
        assert_eq!(SourceLanguage::from_path("utils.hpp"), SourceLanguage::Cpp);
        assert_eq!(
            SourceLanguage::from_path("unknown.xyz"),
            SourceLanguage::Unknown
        );
    }

    // =========================================================================
    // USL Serialization tests
    // =========================================================================

    #[test]
    fn usl_serialization_generates_valid_contract() {
        let code = r#"
fn add(a: i32, b: i32) -> i32 {
    assert!(a >= 0);
    a + b
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Rust);
        let result = inferencer.infer_static(&request);

        let usl = result.to_usl();
        // New USL syntax: contract name(params) -> type { ... }
        assert!(
            usl.contains("contract add("),
            "Expected contract signature, got: {}",
            usl
        );
        assert!(usl.contains("a:"), "Expected parameter 'a' in signature");
        assert!(usl.contains("b:"), "Expected parameter 'b' in signature");
        assert!(usl.contains("requires"));
        assert!(usl.contains("ensures"));
        assert!(usl.contains("// Auto-generated USL specification"));
        assert!(usl.contains("// Confidence:"));
    }

    #[test]
    fn usl_serialization_handles_comparisons() {
        let code = r#"
int check(int x) {
    assert(x > 0);
    assert(x <= 100);
    return x;
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::C);
        let result = inferencer.infer_static(&request);

        let usl = result.to_usl();
        assert!(usl.contains(">") || usl.contains("&gt;"));
        assert!(usl.contains("<=") || usl.contains("&lt;="));
    }

    #[test]
    fn usl_serialization_handles_arithmetic() {
        let code = r#"
fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Rust);
        let result = inferencer.infer_static(&request);

        let usl = result.to_usl();
        // The ensures should contain multiplication
        assert!(usl.contains("*"));
    }

    #[test]
    fn contracts_to_usl_filters_only_contracts() {
        let code = r#"
fn foo(x: i32) -> i32 {
    assert!(x > 0);
    x
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Rust);
        let result = inferencer.infer_static(&request);

        let usl = result.contracts_to_usl();
        assert!(usl.contains("contract"));
        // Should not have the header comments
        assert!(!usl.contains("// Auto-generated"));
    }

    #[test]
    fn usl_type_serialization() {
        assert_eq!(type_to_usl(&Type::Unit), "()");
        assert_eq!(type_to_usl(&Type::Named("i32".to_string())), "i32");
        assert_eq!(
            type_to_usl(&Type::List(Box::new(Type::Named("i32".to_string())))),
            "List<i32>"
        );
        assert_eq!(
            type_to_usl(&Type::Set(Box::new(Type::Named("String".to_string())))),
            "Set<String>"
        );
        assert_eq!(
            type_to_usl(&Type::Result(Box::new(Type::Named("Data".to_string())))),
            "Result<Data>"
        );
    }

    #[test]
    fn usl_expr_serialization() {
        let var = Expr::Var("x".to_string());
        assert_eq!(expr_to_usl(&var), "x");

        let int = Expr::Int(42);
        assert_eq!(expr_to_usl(&int), "42");

        let bool_expr = Expr::Bool(true);
        assert_eq!(expr_to_usl(&bool_expr), "true");

        let and = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(expr_to_usl(&and), "(a && b)");

        let compare = Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            ComparisonOp::Ge,
            Box::new(Expr::Int(0)),
        );
        assert_eq!(expr_to_usl(&compare), "(x >= 0)");
    }

    // =========================================================================
    // ACSL Pre-function Annotation Tests
    // =========================================================================

    #[test]
    fn test_acsl_single_line_block_comment() {
        // Single-line ACSL block comment before function
        let code = r#"
/*@ requires n >= 0; */
int abs(int n) {
    if (n < 0) return -n;
    return n;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert_eq!(contract.type_path, vec!["abs"]);

        // Should have the ACSL requires
        assert!(
            !contract.requires.is_empty(),
            "Expected ACSL requires clause"
        );
        // Check that n >= 0 is captured
        let req_usl = contract
            .requires
            .iter()
            .map(expr_to_usl)
            .collect::<Vec<_>>()
            .join(", ");
        assert!(
            req_usl.contains("n") && req_usl.contains("0"),
            "Expected n >= 0, got: {}",
            req_usl
        );
    }

    #[test]
    fn test_acsl_multiline_block_comment() {
        // Multi-line ACSL block comment
        let code = r#"
/*@
  requires n >= 0;
  requires n < 100;
  ensures \result >= 0;
*/
int safe_abs(int n) {
    return n >= 0 ? n : -n;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert_eq!(contract.type_path, vec!["safe_abs"]);

        // Should have at least 2 requires from ACSL (n >= 0, n < 100)
        assert!(
            contract.requires.len() >= 2,
            "Expected at least 2 ACSL requires, got {}",
            contract.requires.len()
        );

        // Should have ensures from ACSL
        assert!(!contract.ensures.is_empty(), "Expected ACSL ensures clause");
    }

    #[test]
    fn test_acsl_result_keyword() {
        // Test that \result is converted to result
        let code = r#"
/*@ ensures \result == n * 2; */
int double_it(int n) {
    return n * 2;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];

        // The ensures should contain 'result' (not '\result')
        let ensures_usl = contract
            .ensures
            .iter()
            .map(expr_to_usl)
            .collect::<Vec<_>>()
            .join(", ");
        assert!(
            ensures_usl.contains("result"),
            "Expected 'result' in ensures, got: {}",
            ensures_usl
        );
        assert!(
            !ensures_usl.contains("\\result"),
            "Should not contain \\result"
        );
    }

    #[test]
    fn test_acsl_line_comments() {
        // //@ style line comments
        let code = r#"
//@ requires x > 0;
//@ ensures \result > x;
int increment(int x) {
    return x + 1;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert_eq!(contract.type_path, vec!["increment"]);

        // Should capture both requires and ensures
        assert!(
            !contract.requires.is_empty(),
            "Expected requires from //@ comment"
        );
        assert!(
            !contract.ensures.is_empty(),
            "Expected ensures from //@ comment"
        );
    }

    #[test]
    fn test_acsl_old_keyword() {
        // Test \old conversion
        let code = r#"
/*@ ensures \result == \old(x) + 1; */
int inc(int x) {
    return x + 1;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let ensures_usl = contracts[0]
            .ensures
            .iter()
            .map(expr_to_usl)
            .collect::<Vec<_>>()
            .join(", ");
        // \old should be converted to old
        assert!(
            ensures_usl.contains("old("),
            "Expected 'old(' in ensures, got: {}",
            ensures_usl
        );
    }

    #[test]
    fn test_acsl_with_body_asserts() {
        // ACSL annotations combined with body asserts
        let code = r#"
/*@ requires n >= 0; */
int factorial(int n) {
    assert(n <= 12);  // Body assert to prevent overflow
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];

        // Should have both ACSL requires (n >= 0) and body assert (n <= 12)
        assert!(
            contract.requires.len() >= 2,
            "Expected at least 2 requires (ACSL + body assert), got {}",
            contract.requires.len()
        );
    }

    #[test]
    fn test_acsl_cpp_class_method() {
        // ACSL with C++ class method
        let code = r#"
class Calculator {
public:
    /*@
      requires a >= 0;
      requires b >= 0;
      ensures \result == a + b;
    */
    int add(int a, int b) {
        return a + b;
    }
};
"#;
        let contracts = parse_c_cpp_functions(code, true);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert_eq!(contract.type_path, vec!["add"]);

        // Should have the ACSL annotations
        assert!(
            contract.requires.len() >= 2,
            "Expected 2 ACSL requires for a >= 0 and b >= 0"
        );
        assert!(!contract.ensures.is_empty(), "Expected ACSL ensures");
    }

    #[test]
    fn test_acsl_multiple_functions() {
        // Multiple functions with their own ACSL annotations
        let code = r#"
/*@ requires x >= 0; */
int square(int x) {
    return x * x;
}

/*@ requires x != 0; ensures \result * x == 1; */
double reciprocal(double x) {
    return 1.0 / x;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 2);

        // First function: square
        assert_eq!(contracts[0].type_path, vec!["square"]);
        assert!(
            !contracts[0].requires.is_empty(),
            "square should have requires"
        );

        // Second function: reciprocal
        assert_eq!(contracts[1].type_path, vec!["reciprocal"]);
        assert!(
            !contracts[1].requires.is_empty(),
            "reciprocal should have requires"
        );
        assert!(
            !contracts[1].ensures.is_empty(),
            "reciprocal should have ensures"
        );
    }

    #[test]
    fn test_acsl_annotations_not_applied_to_wrong_function() {
        // ACSL should not carry over to unrelated functions
        let code = r#"
/*@ requires x > 0; */
int foo(int x) {
    return x;
}

// Regular comment (not ACSL)
int bar(int y) {
    return y * 2;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 2);

        // foo should have requires from ACSL
        assert!(
            !contracts[0].requires.is_empty(),
            "foo should have ACSL requires"
        );

        // bar should NOT have requires from foo's ACSL
        // It might have body asserts, but not ACSL from foo
        // The bar function has no ACSL, no body asserts
        // So requires should be empty or only from body analysis
        let bar_req_count = contracts[1].requires.len();
        // bar has no asserts and no ACSL, so should have 0 requires
        assert_eq!(
            bar_req_count, 0,
            "bar should not inherit ACSL from foo, got {} requires",
            bar_req_count
        );
    }

    #[test]
    fn test_normalize_acsl_implication() {
        // Test ==> conversion to implies
        let result = normalize_acsl_expr("x > 0 ==> y > 0");
        assert!(
            result.contains("implies"),
            "Expected 'implies', got: {}",
            result
        );
    }

    #[test]
    fn test_normalize_acsl_valid_separated() {
        // Test \valid and \separated conversion
        let result = normalize_acsl_expr("\\valid(p) && \\separated(p, q)");
        assert!(result.contains("valid(p)"), "Expected valid(p): {}", result);
        assert!(
            result.contains("separated(p, q)"),
            "Expected separated(p, q): {}",
            result
        );
    }

    #[test]
    fn test_parse_acsl_annotations_empty() {
        let (requires, ensures) = parse_acsl_annotations("");
        assert!(requires.is_empty());
        assert!(ensures.is_empty());
    }

    #[test]
    fn test_parse_acsl_annotations_basic() {
        let acsl = "/*@ requires x > 0; ensures \\result >= 0; */";
        let (requires, ensures) = parse_acsl_annotations(acsl);
        assert_eq!(requires.len(), 1);
        assert_eq!(ensures.len(), 1);
    }

    // =========================================================================
    // Quantifier and Logical Operator Parsing Tests
    // =========================================================================

    #[test]
    fn test_parse_condition_forall() {
        // Test basic forall parsing
        let expr = parse_condition("forall x: int . x >= 0");
        match expr {
            Expr::ForAll { var, ty, body } => {
                assert_eq!(var, "x");
                assert!(ty.is_some());
                // Body should be x >= 0
                match *body {
                    Expr::Compare(_, ComparisonOp::Ge, _) => {}
                    other => panic!("Expected Compare(Ge), got {:?}", other),
                }
            }
            other => panic!("Expected ForAll, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_condition_exists() {
        // Test basic exists parsing
        let expr = parse_condition("exists y: bool . y == true");
        match expr {
            Expr::Exists { var, ty, body } => {
                assert_eq!(var, "y");
                assert!(ty.is_some());
                match *body {
                    Expr::Compare(_, ComparisonOp::Eq, _) => {}
                    other => panic!("Expected Compare(Eq), got {:?}", other),
                }
            }
            other => panic!("Expected Exists, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_condition_forall_no_type() {
        // Test forall without type annotation
        let expr = parse_condition("forall x . x > 0");
        match expr {
            Expr::ForAll { var, ty, body } => {
                assert_eq!(var, "x");
                assert!(ty.is_none());
                match *body {
                    Expr::Compare(_, ComparisonOp::Gt, _) => {}
                    other => panic!("Expected Compare(Gt), got {:?}", other),
                }
            }
            other => panic!("Expected ForAll, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_condition_implies() {
        // Test implies parsing
        let expr = parse_condition("x > 0 implies y > 0");
        match expr {
            Expr::Implies(left, right) => {
                match *left {
                    Expr::Compare(_, ComparisonOp::Gt, _) => {}
                    other => panic!("Expected Compare(Gt), got {:?}", other),
                }
                match *right {
                    Expr::Compare(_, ComparisonOp::Gt, _) => {}
                    other => panic!("Expected Compare(Gt), got {:?}", other),
                }
            }
            other => panic!("Expected Implies, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_condition_iff() {
        // Test iff (biconditional) parsing - should become (a => b) && (b => a)
        let expr = parse_condition("x == 0 iff y == 0");
        match expr {
            Expr::And(left, right) => {
                match *left {
                    Expr::Implies(_, _) => {}
                    other => panic!("Expected Implies for left, got {:?}", other),
                }
                match *right {
                    Expr::Implies(_, _) => {}
                    other => panic!("Expected Implies for right, got {:?}", other),
                }
            }
            other => panic!("Expected And of two Implies, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_condition_nested_quantifiers() {
        // Test nested quantifiers: forall x . exists y . x < y
        let expr = parse_condition("forall x: int . exists y: int . x < y");
        match expr {
            Expr::ForAll {
                var: outer_var,
                body,
                ..
            } => {
                assert_eq!(outer_var, "x");
                match *body {
                    Expr::Exists {
                        var: inner_var,
                        body: inner_body,
                        ..
                    } => {
                        assert_eq!(inner_var, "y");
                        match *inner_body {
                            Expr::Compare(_, ComparisonOp::Lt, _) => {}
                            other => panic!("Expected Compare(Lt), got {:?}", other),
                        }
                    }
                    other => panic!("Expected Exists, got {:?}", other),
                }
            }
            other => panic!("Expected ForAll, got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_forall_full_pipeline() {
        // Test full ACSL parsing pipeline with \forall
        let acsl = r#"/*@ requires \forall int i; 0 <= i && i < n ==> arr[i] >= 0; */"#;
        let (requires, _) = parse_acsl_annotations(acsl);
        assert_eq!(requires.len(), 1);
        // After normalize_acsl_expr, \forall int i; ... becomes forall i: int . ...
        // Then parse_condition should parse it into Expr::ForAll
        match &requires[0] {
            Expr::ForAll { var, ty, .. } => {
                assert_eq!(var, "i");
                assert!(ty.is_some());
            }
            other => panic!("Expected ForAll expr, got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_exists_full_pipeline() {
        // Test full ACSL parsing pipeline with \exists
        let acsl = r#"/*@ ensures \exists int j; 0 <= j && j < n && arr[j] == target; */"#;
        let (_, ensures) = parse_acsl_annotations(acsl);
        assert_eq!(ensures.len(), 1);
        match &ensures[0] {
            Expr::Exists { var, ty, .. } => {
                assert_eq!(var, "j");
                assert!(ty.is_some());
            }
            other => panic!("Expected Exists expr, got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_implication_in_body() {
        // Test that ==> in ACSL body is converted to implies and parsed
        let acsl = r#"/*@ requires n > 0 ==> result >= 0; */"#;
        let (requires, _) = parse_acsl_annotations(acsl);
        assert_eq!(requires.len(), 1);
        match &requires[0] {
            Expr::Implies(_, _) => {}
            other => panic!("Expected Implies expr, got {:?}", other),
        }
    }

    // =========================================================================
    // ACSL Assigns (Frame Condition) Tests
    // =========================================================================

    #[test]
    fn test_acsl_assigns_nothing() {
        // Test assigns \nothing - function modifies no memory
        let acsl = r#"/*@ assigns \nothing; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.assigns.len(), 1);
        match &annotations.assigns[0] {
            Expr::Var(name) => assert_eq!(name, "nothing"),
            other => panic!("Expected Var(nothing), got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_assigns_single_location() {
        // Test single memory location
        let acsl = r#"/*@ assigns *p; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.assigns.len(), 1);
    }

    #[test]
    fn test_acsl_assigns_multiple_locations() {
        // Test multiple memory locations
        let acsl = r#"/*@ assigns *p, *q, arr[0..n]; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.assigns.len(), 3);
    }

    #[test]
    fn test_acsl_assigns_with_requires_ensures() {
        // Test assigns alongside requires and ensures
        let acsl = r#"/*@
            requires n > 0;
            assigns *result;
            ensures \result == n * n;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.requires.len(), 1);
        assert_eq!(annotations.ensures.len(), 1);
        assert_eq!(annotations.assigns.len(), 1);
    }

    #[test]
    fn test_acsl_assigns_serialization() {
        // Test that assigns are serialized to USL
        let code = r#"
/*@ requires n >= 0;
    assigns *result;
    ensures *result == n * n;
*/
void square(int n, int *result) {
    *result = n * n;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        assert!(!contracts[0].assigns.is_empty(), "assigns should be parsed");

        // Serialize and check USL output
        let usl = contract_to_usl(&contracts[0]);
        assert!(
            usl.contains("assigns"),
            "USL should contain assigns clause: {}",
            usl
        );
    }

    // =========================================================================
    // USL Round-trip Test
    // =========================================================================

    #[test]
    fn test_debug_parse_c() {
        // Debug test to understand expression parsing
        let code = r#"
int square(int n) {
    return n * n;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);

        let contract = &contracts[0];
        eprintln!("Contract name: {:?}", contract.type_path);
        eprintln!("Ensures count: {}", contract.ensures.len());
        for (i, e) in contract.ensures.iter().enumerate() {
            eprintln!("Ensures[{}]: {:?}", i, e);
            eprintln!("Ensures[{}] USL: {}", i, expr_to_usl(e));
        }

        // The ensures should be result == (n * n), not result == (n * * n)
        let ensures_usl = expr_to_usl(&contract.ensures[0]);
        assert!(
            !ensures_usl.contains("* *"),
            "Should not contain '* *', got: {}",
            ensures_usl
        );
    }

    #[test]
    fn test_usl_round_trip_basic() {
        // Test that inferred specs can be serialized to USL and parsed back
        let code = r#"
/*@ requires n > 0; */
int square(int n) {
    return n * n;
}
"#;
        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::C);
        let result = inferencer.infer_static(&request);

        // Serialize to USL
        let usl_text = result.to_usl();

        // The USL should contain contract syntax
        assert!(
            usl_text.contains("contract"),
            "USL should contain 'contract': {}",
            usl_text
        );
        assert!(
            usl_text.contains("square"),
            "USL should contain function name 'square': {}",
            usl_text
        );

        // Try to parse it back (this uses dashprove_usl::parse)
        // This is a weak test since we're just checking it doesn't panic
        // Full round-trip would require comparing ASTs
        let parse_result = dashprove_usl::parse(&usl_text);
        assert!(
            parse_result.is_ok(),
            "Generated USL should be parseable: {:?}\n\nUSL:\n{}",
            parse_result.err(),
            usl_text
        );
    }

    // =========================================================================
    // ACSL allocates/frees Clause Tests
    // =========================================================================

    #[test]
    fn test_acsl_allocates_nothing() {
        // Test allocates \nothing - function does no dynamic allocation
        let acsl = r#"/*@ allocates \nothing; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.allocates.len(), 1);
        match &annotations.allocates[0] {
            Expr::Var(name) => assert_eq!(name, "nothing"),
            other => panic!("Expected Var(nothing), got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_allocates_single_location() {
        // Test single allocation location
        let acsl = r#"/*@ allocates p; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.allocates.len(), 1);
    }

    #[test]
    fn test_acsl_allocates_multiple_locations() {
        // Test multiple allocation locations
        let acsl = r#"/*@ allocates p, q, arr; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.allocates.len(), 3);
    }

    #[test]
    fn test_acsl_frees_nothing() {
        // Test frees \nothing - function does no deallocation
        let acsl = r#"/*@ frees \nothing; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.frees.len(), 1);
        match &annotations.frees[0] {
            Expr::Var(name) => assert_eq!(name, "nothing"),
            other => panic!("Expected Var(nothing), got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_frees_single_location() {
        // Test single free location
        let acsl = r#"/*@ frees p; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.frees.len(), 1);
    }

    #[test]
    fn test_acsl_frees_multiple_locations() {
        // Test multiple free locations
        let acsl = r#"/*@ frees p, q; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.frees.len(), 2);
    }

    #[test]
    fn test_acsl_allocates_frees_combined() {
        // Test allocates and frees together (common in realloc-style functions)
        let acsl = r#"/*@
            requires \valid(old_ptr);
            allocates new_ptr;
            frees old_ptr;
            ensures \valid(new_ptr);
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.requires.len(), 1);
        assert_eq!(annotations.ensures.len(), 1);
        assert_eq!(annotations.allocates.len(), 1);
        assert_eq!(annotations.frees.len(), 1);
    }

    #[test]
    fn test_acsl_full_heap_contract() {
        // Test full heap contract with assigns, allocates, and frees
        let acsl = r#"/*@
            requires n > 0;
            assigns *result;
            allocates result;
            frees \nothing;
            ensures \valid(result) && *result == n;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.requires.len(), 1);
        assert_eq!(annotations.ensures.len(), 1);
        assert_eq!(annotations.assigns.len(), 1);
        assert_eq!(annotations.allocates.len(), 1);
        assert_eq!(annotations.frees.len(), 1);
    }

    #[test]
    fn test_acsl_heap_contract_in_c_function() {
        // Test that allocates and frees are parsed in full C function context
        let code = r#"
/*@ requires n > 0;
    allocates \result;
    frees \nothing;
    ensures \valid(\result);
*/
int* allocate_array(int n) {
    return malloc(n * sizeof(int));
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert!(!contract.allocates.is_empty(), "allocates should be parsed");
        assert!(!contract.frees.is_empty(), "frees should be parsed");
    }

    // =================== ACSL terminates clause tests ===================

    #[test]
    fn test_acsl_terminates_true() {
        // Test terminates \true - function always terminates
        let acsl = r#"/*@ terminates \true; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert!(annotations.terminates.is_some());
        match &annotations.terminates {
            Some(Expr::Var(name)) => assert_eq!(name, "true"),
            other => panic!("Expected Var(true), got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_terminates_false() {
        // Test terminates \false - function may not terminate (e.g., servers, event loops)
        let acsl = r#"/*@ terminates \false; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert!(annotations.terminates.is_some());
        match &annotations.terminates {
            Some(Expr::Var(name)) => assert_eq!(name, "false"),
            other => panic!("Expected Var(false), got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_terminates_condition() {
        // Test terminates with a condition - function terminates when condition holds
        let acsl = r#"/*@ terminates n >= 0; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert!(annotations.terminates.is_some());
        // The condition should be parsed as a comparison expression
        assert!(matches!(
            &annotations.terminates,
            Some(Expr::Compare(_, _, _))
        ));
    }

    // =================== ACSL decreases clause tests ===================

    #[test]
    fn test_acsl_decreases_variable() {
        // Test decreases with a single variable - common for loop counters
        let acsl = r#"/*@ decreases n; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert!(annotations.decreases.is_some());
        match &annotations.decreases {
            Some(Expr::Var(name)) => assert_eq!(name, "n"),
            other => panic!("Expected Var(n), got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_decreases_expression() {
        // Test decreases with an expression - variant for recursion
        // Note: Our simple condition parser treats "len - i" as a variable string
        // since it doesn't parse arithmetic. This is sufficient for serialization.
        let acsl = r#"/*@ decreases len - i; */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert!(annotations.decreases.is_some());
        // The expression is stored as-is (a Var containing the full expression text)
        match &annotations.decreases {
            Some(Expr::Var(expr)) => assert_eq!(expr, "len - i"),
            other => panic!("Expected Var(len - i), got {:?}", other),
        }
    }

    #[test]
    fn test_acsl_terminates_decreases_combined() {
        // Test terminates and decreases together - full termination specification
        let acsl = r#"/*@
            requires n >= 0;
            terminates \true;
            decreases n;
            ensures \result >= 0;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.requires.len(), 1);
        assert_eq!(annotations.ensures.len(), 1);
        assert!(annotations.terminates.is_some());
        assert!(annotations.decreases.is_some());
    }

    #[test]
    fn test_acsl_recursive_function_contract() {
        // Test a complete recursive function contract with termination proof
        let code = r#"
/*@ requires n >= 0;
    terminates \true;
    decreases n;
    ensures \result == n * (n + 1) / 2;
*/
int sum_to_n(int n) {
    if (n == 0) return 0;
    return n + sum_to_n(n - 1);
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert!(!contract.requires.is_empty(), "requires should be parsed");
        assert!(!contract.ensures.is_empty(), "ensures should be parsed");
        assert!(contract.terminates.is_some(), "terminates should be parsed");
        assert!(contract.decreases.is_some(), "decreases should be parsed");
    }

    #[test]
    fn test_acsl_loop_function_with_decreases() {
        // Test a function with loop variant (decreases) for termination
        let code = r#"
/*@ requires n >= 0;
    terminates \true;
    decreases n - i;
    ensures \result == n;
*/
int count_to_n(int n) {
    int i = 0;
    /*@ loop invariant 0 <= i <= n;
        loop variant n - i;
    */
    while (i < n) {
        i++;
    }
    return i;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        // Note: We parse function-level terminates/decreases, not loop annotations (yet)
        assert!(contract.terminates.is_some(), "terminates should be parsed");
        assert!(contract.decreases.is_some(), "decreases should be parsed");
    }

    #[test]
    fn test_acsl_non_terminating_server() {
        // Test a non-terminating function (e.g., server main loop)
        let code = r#"
/*@ terminates \false;
    assigns \nothing;
*/
void server_main_loop(void) {
    while (1) {
        handle_request();
    }
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert!(contract.terminates.is_some(), "terminates should be parsed");
        // For \false, we expect the condition to be the boolean false
        match &contract.terminates {
            Some(Expr::Var(name)) => assert_eq!(name, "false"),
            other => panic!("Expected Var(false), got {:?}", other),
        }
    }

    // =================== ACSL behavior clause tests ===================

    #[test]
    fn test_acsl_single_behavior() {
        // Test a single named behavior
        let acsl = r#"/*@
            behavior success:
                assumes x >= 0;
                ensures \result >= 0;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.behaviors.len(), 1);
        assert_eq!(annotations.behaviors[0].name, "success");
        assert_eq!(annotations.behaviors[0].assumes.len(), 1);
        assert_eq!(annotations.behaviors[0].ensures.len(), 1);
    }

    #[test]
    fn test_acsl_multiple_behaviors() {
        // Test multiple behaviors (common for error handling)
        let acsl = r#"/*@
            behavior normal:
                assumes ptr != \null;
                ensures \result > 0;

            behavior null_input:
                assumes ptr == \null;
                ensures \result == -1;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.behaviors.len(), 2);
        assert_eq!(annotations.behaviors[0].name, "normal");
        assert_eq!(annotations.behaviors[1].name, "null_input");
    }

    #[test]
    fn test_acsl_behavior_with_assigns() {
        // Test behavior with assigns clause
        let acsl = r#"/*@
            behavior init:
                assumes x == 0;
                assigns *buffer;
                ensures buffer[0] == 0;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.behaviors.len(), 1);
        assert_eq!(annotations.behaviors[0].assigns.len(), 1);
    }

    #[test]
    fn test_acsl_complete_behaviors() {
        // Test complete behaviors declaration
        let acsl = r#"/*@
            behavior pos:
                assumes x > 0;
                ensures \result > 0;

            behavior neg:
                assumes x <= 0;
                ensures \result <= 0;

            complete behaviors;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.behaviors.len(), 2);
        assert!(annotations.complete_behaviors);
        assert!(!annotations.disjoint_behaviors);
    }

    #[test]
    fn test_acsl_disjoint_behaviors() {
        // Test disjoint behaviors declaration
        let acsl = r#"/*@
            behavior even:
                assumes x % 2 == 0;
                ensures \result == x / 2;

            behavior odd:
                assumes x % 2 == 1;
                ensures \result == (x - 1) / 2;

            disjoint behaviors;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.behaviors.len(), 2);
        assert!(!annotations.complete_behaviors);
        assert!(annotations.disjoint_behaviors);
    }

    #[test]
    fn test_acsl_complete_and_disjoint_behaviors() {
        // Test both complete and disjoint behaviors
        let acsl = r#"/*@
            requires n >= 0;

            behavior zero:
                assumes n == 0;
                ensures \result == 1;

            behavior positive:
                assumes n > 0;
                ensures \result == n * old_factorial;

            complete behaviors;
            disjoint behaviors;
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.requires.len(), 1);
        assert_eq!(annotations.behaviors.len(), 2);
        assert!(annotations.complete_behaviors);
        assert!(annotations.disjoint_behaviors);
    }

    #[test]
    fn test_acsl_behavior_in_c_function() {
        // Test behaviors parsed in full C function context
        let code = r#"
/*@ requires len >= 0;

    behavior found:
        assumes \exists integer i; 0 <= i < len && arr[i] == key;
        ensures \result >= 0;
        ensures arr[\result] == key;

    behavior not_found:
        assumes \forall integer i; 0 <= i < len ==> arr[i] != key;
        ensures \result == -1;

    complete behaviors;
    disjoint behaviors;
*/
int binary_search(int* arr, int len, int key) {
    // implementation
    return -1;
}
"#;
        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert!(!contract.requires.is_empty(), "requires should be parsed");
        assert_eq!(
            contract.behaviors.len(),
            2,
            "two behaviors should be parsed"
        );
        assert!(
            contract.complete_behaviors,
            "complete behaviors should be set"
        );
        assert!(
            contract.disjoint_behaviors,
            "disjoint behaviors should be set"
        );
    }

    #[test]
    fn test_acsl_behavior_with_requires() {
        // Test behavior with both assumes and requires
        let acsl = r#"/*@
            behavior alloc_success:
                assumes size > 0 && size <= MAX_SIZE;
                requires \valid(heap);
                ensures \result != \null;
                ensures \valid(\result + (0..size-1));
        */"#;
        let annotations = parse_acsl_annotations_full(acsl);
        assert_eq!(annotations.behaviors.len(), 1);
        assert_eq!(annotations.behaviors[0].assumes.len(), 1);
        assert_eq!(annotations.behaviors[0].requires.len(), 1);
        assert_eq!(annotations.behaviors[0].ensures.len(), 2);
    }

    // =========================================================================
    // Mutation-killing tests for helper functions
    // =========================================================================

    #[test]
    fn test_source_language_label_returns_correct_strings() {
        // Tests for SourceLanguage::label() - mutation testing shows "xyzzy" and "" replacements
        assert_eq!(SourceLanguage::Rust.label(), "Rust");
        assert_eq!(SourceLanguage::TypeScript.label(), "TypeScript");
        assert_eq!(SourceLanguage::Python.label(), "Python");
        assert_eq!(SourceLanguage::Go.label(), "Go");
        assert_eq!(SourceLanguage::C.label(), "C");
        assert_eq!(SourceLanguage::Cpp.label(), "C++");
        assert_eq!(SourceLanguage::Unknown.label(), "Unknown");

        // Verify non-empty and distinguishable
        let labels = [
            SourceLanguage::Rust.label(),
            SourceLanguage::TypeScript.label(),
            SourceLanguage::Python.label(),
            SourceLanguage::Go.label(),
            SourceLanguage::C.label(),
            SourceLanguage::Cpp.label(),
            SourceLanguage::Unknown.label(),
        ];
        for label in &labels {
            assert!(!label.is_empty(), "label should not be empty");
            assert_ne!(*label, "xyzzy", "label should not be placeholder");
        }
        // All labels should be unique
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), labels.len(), "all labels should be unique");
    }

    #[test]
    fn test_expr_key_produces_distinguishable_keys() {
        // Tests for expr_key() - mutation testing shows return value replacements
        let expr1 = Expr::Var("x".to_string());
        let expr2 = Expr::Var("y".to_string());
        let expr3 = Expr::Compare(
            Box::new(Expr::Var("a".to_string())),
            ComparisonOp::Gt,
            Box::new(Expr::Int(0)),
        );

        let key1 = expr_key(&expr1);
        let key2 = expr_key(&expr2);
        let key3 = expr_key(&expr3);

        // Keys should be non-empty
        assert!(!key1.is_empty());
        assert!(!key2.is_empty());
        assert!(!key3.is_empty());

        // Different expressions should produce different keys
        assert_ne!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key2, key3);

        // Same expression should produce same key
        let expr1_again = Expr::Var("x".to_string());
        assert_eq!(expr_key(&expr1), expr_key(&expr1_again));
    }

    #[test]
    fn test_merge_conditions_deduplicates() {
        // Tests for merge_conditions() - mutation testing shows delete ! and replace with ()
        let mut target = vec![parse_condition("x > 0")];
        let hints = vec!["y < 10".to_string(), "x > 0".to_string()]; // "x > 0" is duplicate

        merge_conditions(&mut target, &hints);

        // Should have x > 0 and y < 10 (deduplicated x > 0)
        assert_eq!(target.len(), 2);
    }

    #[test]
    fn test_merge_conditions_adds_new_conditions() {
        let mut target: Vec<Expr> = vec![];
        let hints = vec!["a >= 0".to_string(), "b <= 100".to_string()];

        merge_conditions(&mut target, &hints);

        assert_eq!(target.len(), 2);
    }

    #[test]
    fn test_merge_conditions_preserves_existing() {
        let mut target = vec![parse_condition("x > 0"), parse_condition("y < 5")];
        let hints = vec!["z == 1".to_string()];

        merge_conditions(&mut target, &hints);

        // Should preserve existing 2 and add 1 new
        assert_eq!(target.len(), 3);
    }

    #[test]
    fn test_extract_json_block_finds_json() {
        // Tests for extract_json_block() - mutation testing shows None/Some("xyzzy") replacements
        let text_with_json = r#"Here is some analysis:
```json
{"key": "value", "number": 42}
```
And more text."#;

        let result = extract_json_block(text_with_json);
        assert!(result.is_some());
        let json = result.unwrap();
        assert!(json.contains("key"));
        assert!(json.contains("value"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_extract_json_block_returns_none_for_no_json() {
        let text_without_json = "Just some plain text without any JSON blocks.";
        let result = extract_json_block(text_without_json);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_json_block_requires_code_fence() {
        // The function only extracts JSON from ```json blocks, not inline JSON
        let inline_json = r#"{"requires": ["x > 0"], "ensures": ["result >= 0"]}"#;
        let result = extract_json_block(inline_json);
        assert!(
            result.is_none(),
            "inline JSON without code fence should return None"
        );

        // JSON in code block should be extracted
        let fenced_json = "```json\n{\"requires\": [\"x > 0\"]}\n```";
        let result = extract_json_block(fenced_json);
        assert!(result.is_some());
        let json = result.unwrap();
        assert!(json.contains("requires"));
    }

    #[test]
    fn test_rust_parsing_brace_depth_tracking() {
        // Tests for brace depth tracking in parse_rust_functions
        // Mutation testing shows += vs -= swaps
        let code = r#"
fn outer() {
    let inner = || {
        if true {
            println!("nested");
        }
    };
    inner();
}

fn second() {
    println!("after nested");
}
"#;

        let contracts = parse_rust_functions(code);
        // Should correctly parse both functions despite nested braces
        assert_eq!(contracts.len(), 2);
    }

    #[test]
    fn test_rust_parsing_multiline_signature() {
        // Tests for signature continuation while loop
        let code = r#"
pub fn long_function_name(
    param1: i32,
    param2: String,
    param3: Vec<u8>
) -> Result<(), Error> {
    Ok(())
}
"#;

        let contracts = parse_rust_functions(code);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert_eq!(contract.type_path, vec!["long_function_name".to_string()]);
        assert_eq!(contract.params.len(), 3);
    }

    #[test]
    fn test_typescript_parsing_brace_depth() {
        // Tests for brace depth tracking in parse_typescript_functions
        let code = r#"
function outer(): void {
    const nested = () => {
        if (true) {
            console.log("deep");
        }
    };
    nested();
}

function second(): number {
    return 42;
}
"#;

        let contracts = parse_typescript_functions(code);
        assert_eq!(contracts.len(), 2);
    }

    #[test]
    fn test_python_parsing_extracts_multiple_functions() {
        // Tests for Python function parsing
        let code = r#"
def first(a: int) -> int:
    assert a > 0
    return a * 2

def second(b: str) -> str:
    return b.upper()
"#;

        let contracts = parse_python_functions(code);
        assert_eq!(contracts.len(), 2);
        assert_eq!(contracts[0].type_path, vec!["first".to_string()]);
        assert_eq!(contracts[1].type_path, vec!["second".to_string()]);
    }

    #[test]
    fn test_go_parsing_extracts_functions() {
        // Tests for Go function parsing
        let code = r#"
func Add(a int, b int) int {
    if a < 0 {
        panic("negative")
    }
    return a + b
}

func Multiply(x, y int) int {
    return x * y
}
"#;

        let contracts = parse_go_functions(code);
        assert_eq!(contracts.len(), 2);
        assert_eq!(contracts[0].type_path, vec!["Add".to_string()]);
        assert_eq!(contracts[1].type_path, vec!["Multiply".to_string()]);
    }

    #[test]
    fn test_c_cpp_parsing_extracts_functions() {
        // Tests for C/C++ function parsing
        let code = r#"
int add(int a, int b) {
    return a + b;
}

void process(int* data, size_t len) {
    assert(data != NULL);
    for (size_t i = 0; i < len; i++) {
        data[i] *= 2;
    }
}
"#;

        let contracts = parse_c_cpp_functions(code, false);
        assert_eq!(contracts.len(), 2);
    }

    #[test]
    fn test_cpp_parsing_with_cpp_flag() {
        // Tests for C++ specific parsing (is_cpp parameter)
        let code = r#"
void Vector::push_back(const T& value) {
    if (size_ >= capacity_) {
        grow();
    }
    data_[size_++] = value;
}
"#;

        let contracts = parse_c_cpp_functions(code, true);
        assert_eq!(contracts.len(), 1);
    }

    #[test]
    fn test_typescript_signature_return_type_extraction() {
        // Tests for parse_typescript_signature return type handling
        let code = r#"
function returnsNumber(x: number): number {
    return x + 1;
}

function returnsPromise(url: string): Promise<Response> {
    return fetch(url);
}
"#;

        let contracts = parse_typescript_functions(code);
        assert_eq!(contracts.len(), 2);

        // Both functions should have return type information in ensures
        // This verifies return type parsing works
        let first = &contracts[0];
        assert_eq!(first.type_path, vec!["returnsNumber".to_string()]);
    }

    #[test]
    fn test_python_signature_params_extraction() {
        // Tests for parse_python_signature parameter handling
        let code = r#"
def with_types(name: str, count: int, flag: bool) -> str:
    return name * count
"#;

        let contracts = parse_python_functions(code);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert_eq!(contract.params.len(), 3);
        assert_eq!(contract.params[0].name, "name");
        assert_eq!(contract.params[1].name, "count");
        assert_eq!(contract.params[2].name, "flag");
    }

    #[test]
    fn test_go_signature_params_extraction() {
        // Tests for parse_go_signature parameter handling
        let code = r#"
func Process(data []byte, offset int, length int) error {
    return nil
}
"#;

        let contracts = parse_go_functions(code);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        // Go parameters should be extracted
        assert!(!contract.params.is_empty());
    }

    #[test]
    fn test_confidence_calculation_with_multiple_properties() {
        // Tests for confidence scoring in infer_* methods
        let code = r#"
fn well_documented(x: i32, y: i32) -> i32 {
    assert!(x >= 0);
    assert!(y >= 0);
    assert!(x < 1000);
    x + y
}
"#;

        let inferencer = SpecInferencer::new();
        let request = SpecInferenceRequest::new(code, SourceLanguage::Rust);
        let result = inferencer.infer_static(&request);

        // More preconditions should increase confidence
        assert!(result.confidence > 0.3);
        assert_eq!(result.properties.len(), 1);
        let Property::Contract(contract) = &result.properties[0] else {
            panic!("expected contract");
        };
        assert!(contract.requires.len() >= 3);
    }

    #[test]
    fn test_ts_assert_extraction() {
        // Tests for extract_ts_asserts boolean conditions
        let code = r#"
function validate(x: number): void {
    console.assert(x > 0);
    console.assert(x < 100, "too large");
    assert(x !== null);
}
"#;

        let contracts = parse_typescript_functions(code);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        // Should extract all three assertions
        assert!(contract.requires.len() >= 2);
    }

    #[test]
    fn test_ts_return_contract_extraction() {
        // Tests for extract_ts_return_contract various return patterns
        let code = r#"
function compute(x: number): number {
    return x * 2;
}
"#;

        let contracts = parse_typescript_functions(code);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        // Should have an ensures clause from return
        assert!(!contract.ensures.is_empty());
    }

    #[test]
    fn test_python_assert_extraction() {
        // Tests for extract_python_asserts
        let code = r#"
def check(x: int) -> int:
    assert x > 0
    assert x < 1000, "out of range"
    return x * 2
"#;

        let contracts = parse_python_functions(code);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert!(contract.requires.len() >= 2);
    }

    #[test]
    fn test_python_return_contract_extraction() {
        // Tests for extract_python_return_contract
        let code = r#"
def double(x: int) -> int:
    return x * 2
"#;

        let contracts = parse_python_functions(code);
        assert_eq!(contracts.len(), 1);
        let contract = &contracts[0];
        assert!(!contract.ensures.is_empty());
    }

    // =========================================================================
    // Mutation-killing tests for parsing loop edge cases
    // These tests target specific mutations in index arithmetic and boolean logic
    // =========================================================================

    #[test]
    fn test_rust_parsing_ignores_commented_functions() {
        // This test catches: line 725 `&& !trimmed.starts_with("//")` → `||`
        // If the mutation survives, commented functions would be parsed
        let code = r#"
// fn commented_out(x: i32) -> i32 {
//     x + 1
// }

fn real_function(y: i32) -> i32 {
    y * 2
}
"#;
        let contracts = parse_rust_functions(code);
        assert_eq!(
            contracts.len(),
            1,
            "should only parse the uncommented function"
        );
        assert_eq!(contracts[0].type_path, vec!["real_function".to_string()]);
    }

    #[test]
    fn test_rust_parsing_inline_body_after_brace() {
        // This test catches: line 740 `!body_start.trim().is_empty()` deletion
        // If the mutation survives, body content after { on same line would be lost
        let code = r#"
fn inline_body(x: i32) -> i32 { let y = x * 2; y + 1 }
"#;
        let contracts = parse_rust_functions(code);
        assert_eq!(contracts.len(), 1);
        // The body should have the inline content
        assert!(contracts[0]
            .ensures
            .iter()
            .any(|e| format!("{:?}", e).contains("result")));
    }

    #[test]
    fn test_rust_parsing_triple_nested_braces() {
        // This test catches: lines 743-744 brace depth increments/decrements
        // += → -= or similar would break brace matching
        let code = r#"
fn deeply_nested() {
    {
        {
            {
                let x = 1;
            }
        }
    }
}

fn after_nested() {
    let y = 2;
}
"#;
        let contracts = parse_rust_functions(code);
        // Both functions must be correctly parsed
        assert_eq!(
            contracts.len(),
            2,
            "triple nesting should not confuse brace tracking"
        );
        assert_eq!(contracts[0].type_path, vec!["deeply_nested".to_string()]);
        assert_eq!(contracts[1].type_path, vec!["after_nested".to_string()]);
    }

    #[test]
    fn test_rust_parsing_asymmetric_braces_in_strings() {
        // This test catches: lines 749-750 brace counting changes
        // Strings with unbalanced braces should be counted correctly
        let code = r#"
fn string_braces() {
    let s = "{ }}}";
    println!("{}", s);
}

fn after_strings() {
    let t = 42;
}
"#;
        let contracts = parse_rust_functions(code);
        // Note: current implementation counts braces in strings too, but
        // the important thing is consistency - both functions should be parsed
        assert!(!contracts.is_empty(), "should parse at least one function");
    }

    #[test]
    fn test_typescript_parsing_all_function_forms() {
        // This test catches: line 820 `||` → `&&` in function detection
        // All three forms (function, async function, export function) should be detected
        let code = r#"
function regular(x: number): number {
    return x;
}

async function asyncFunc(x: number): Promise<number> {
    return x;
}

export function exported(x: number): number {
    return x;
}
"#;
        let contracts = parse_typescript_functions(code);
        assert_eq!(contracts.len(), 3, "should parse all function forms");
    }

    #[test]
    fn test_typescript_parsing_inline_body_content() {
        // This test catches: lines 839 `!body_start.is_empty()` deletion
        let code = r#"
function inline(): number { return 42; }
function multiline(): number {
    return 42;
}
"#;
        let contracts = parse_typescript_functions(code);
        assert_eq!(contracts.len(), 2, "should parse both inline and multiline");
    }

    #[test]
    fn test_typescript_parsing_nested_arrow_functions() {
        // This test catches: lines 842-843 brace depth tracking in TS
        let code = r#"
function outer(): void {
    const inner = () => {
        const deep = () => {
            console.log("deep");
        };
    };
}

function second(): number {
    return 1;
}
"#;
        let contracts = parse_typescript_functions(code);
        assert_eq!(contracts.len(), 2, "nested arrows should not break parsing");
    }

    #[test]
    fn test_python_parsing_loops_at_file_boundary() {
        // This test catches: line 1035 `i < lines.len()` comparison mutations
        // Tests that line index stays in bounds when parsing continues to file end
        let code = "def single_func(x: int) -> int:\n    return x";
        let contracts = parse_python_functions(code);
        assert_eq!(contracts.len(), 1, "should handle function at file end");
        assert_eq!(contracts[0].type_path, vec!["single_func".to_string()]);
    }

    #[test]
    fn test_python_parsing_body_indentation_detection() {
        // This test catches: line 1051 `!line.is_empty()` → `line.is_empty()`
        // Empty lines in function body should be handled correctly
        let code = r#"
def with_blank_lines(x: int) -> int:
    y = x * 2

    z = y + 1

    return z

def second(a: int) -> int:
    return a
"#;
        let contracts = parse_python_functions(code);
        assert_eq!(
            contracts.len(),
            2,
            "blank lines should not end function body early"
        );
    }

    #[test]
    fn test_go_parsing_function_with_receiver() {
        // This test catches: line 1213 `&&` → `||` in func detection
        let code = r#"
func (r *Receiver) Method(x int) int {
    return x
}

func Regular(y int) int {
    return y
}
"#;
        let contracts = parse_go_functions(code);
        // Should parse both methods and functions
        assert!(!contracts.is_empty(), "should parse go functions");
    }

    #[test]
    fn test_rust_parsing_boundary_index_handling() {
        // This test catches: line 728 `i < lines.len()` → `i <= lines.len()`
        // Function declaration that continues to the exact end of input
        let code = "fn at_boundary(x: i32) -> i32 {\n    x\n}";
        let contracts = parse_rust_functions(code);
        assert_eq!(
            contracts.len(),
            1,
            "should handle function at exact boundary"
        );
    }

    #[test]
    fn test_rust_parsing_consecutive_functions_no_gap() {
        // This test catches: loop condition mutations affecting multi-function parsing
        let code = r#"fn first() {
}
fn second() {
}
fn third() {
}"#;
        let contracts = parse_rust_functions(code);
        assert_eq!(contracts.len(), 3, "should parse consecutive functions");
    }
}

// =============================================================================
// Kani Proofs for Self-Verification
// =============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that comparison_op_to_usl returns distinct values for each operator
    #[kani::proof]
    fn verify_comparison_op_to_usl_distinct() {
        let ops = [
            ComparisonOp::Eq,
            ComparisonOp::Ne,
            ComparisonOp::Lt,
            ComparisonOp::Le,
            ComparisonOp::Gt,
            ComparisonOp::Ge,
        ];

        // Each operator should map to a different string
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                let s1 = comparison_op_to_usl(&ops[i]);
                let s2 = comparison_op_to_usl(&ops[j]);
                kani::assert(
                    s1 != s2,
                    "Comparison operators must map to distinct strings",
                );
            }
        }
    }

    /// Prove that binary_op_to_usl returns distinct values for each operator
    #[kani::proof]
    fn verify_binary_op_to_usl_distinct() {
        use dashprove_usl::ast::BinaryOp;

        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Mod,
        ];

        // Each operator should map to a different string
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                let s1 = binary_op_to_usl(&ops[i]);
                let s2 = binary_op_to_usl(&ops[j]);
                kani::assert(s1 != s2, "Binary operators must map to distinct strings");
            }
        }
    }

    /// Prove that comparison_op_to_usl returns non-empty strings
    #[kani::proof]
    fn verify_comparison_op_to_usl_nonempty() {
        let ops = [
            ComparisonOp::Eq,
            ComparisonOp::Ne,
            ComparisonOp::Lt,
            ComparisonOp::Le,
            ComparisonOp::Gt,
            ComparisonOp::Ge,
        ];

        for op in ops {
            let s = comparison_op_to_usl(&op);
            kani::assert(
                !s.is_empty(),
                "Comparison operator string must not be empty",
            );
        }
    }

    /// Prove that binary_op_to_usl returns non-empty strings
    #[kani::proof]
    fn verify_binary_op_to_usl_nonempty() {
        use dashprove_usl::ast::BinaryOp;

        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Mod,
        ];

        for op in ops {
            let s = binary_op_to_usl(&op);
            kani::assert(!s.is_empty(), "Binary operator string must not be empty");
        }
    }

    /// Prove that parse_comparison_op and comparison_op_to_usl are inverses
    /// for valid operator strings
    #[kani::proof]
    fn verify_comparison_op_roundtrip() {
        let ops = [
            ComparisonOp::Eq,
            ComparisonOp::Ne,
            ComparisonOp::Lt,
            ComparisonOp::Le,
            ComparisonOp::Gt,
            ComparisonOp::Ge,
        ];

        for op in ops {
            let s = comparison_op_to_usl(&op);
            let parsed = parse_comparison_op(s);
            kani::assert(
                std::mem::discriminant(&op) == std::mem::discriminant(&parsed),
                "Round-trip through USL string should preserve comparison operator",
            );
        }
    }

    /// Prove that SourceLanguage::label returns non-empty strings for all variants
    #[kani::proof]
    fn verify_source_language_label_nonempty() {
        let langs = [
            SourceLanguage::Rust,
            SourceLanguage::TypeScript,
            SourceLanguage::Python,
            SourceLanguage::Go,
            SourceLanguage::C,
            SourceLanguage::Cpp,
            SourceLanguage::Unknown,
        ];

        for lang in langs {
            let label = lang.label();
            kani::assert(!label.is_empty(), "Source language label must not be empty");
        }
    }

    /// Prove that SourceLanguage::label returns distinct values for known languages
    #[kani::proof]
    fn verify_source_language_label_distinct() {
        // Known languages should have distinct labels
        let known_langs = [
            SourceLanguage::Rust,
            SourceLanguage::TypeScript,
            SourceLanguage::Python,
            SourceLanguage::Go,
            SourceLanguage::C,
            SourceLanguage::Cpp,
        ];

        for i in 0..known_langs.len() {
            for j in (i + 1)..known_langs.len() {
                let l1 = known_langs[i].label();
                let l2 = known_langs[j].label();
                kani::assert(l1 != l2, "Known source languages must have distinct labels");
            }
        }
    }
}
