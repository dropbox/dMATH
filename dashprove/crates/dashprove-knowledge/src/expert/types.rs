//! Types for expert recommendations

use dashprove_backends::BackendId;
use serde::{Deserialize, Serialize};

/// Context for expert queries
#[derive(Debug, Clone, Default)]
pub struct ExpertContext {
    /// The user's specification or code
    pub specification: Option<String>,
    /// The current backend (if any)
    pub current_backend: Option<BackendId>,
    /// Any error messages
    pub error_messages: Vec<String>,
    /// Property types in the specification
    pub property_types: Vec<PropertyType>,
    /// Code language (for code verification)
    pub code_language: Option<String>,
    /// Additional context tags
    pub tags: Vec<String>,
}

/// Types of properties that can be verified
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyType {
    /// Safety properties (invariants, memory safety)
    Safety,
    /// Liveness properties (something eventually happens)
    Liveness,
    /// Temporal logic properties
    Temporal,
    /// Functional correctness
    Correctness,
    /// Probabilistic properties
    Probabilistic,
    /// Neural network properties (robustness, etc.)
    NeuralNetwork,
    /// Security protocol properties
    SecurityProtocol,
    /// Refinement/simulation relations
    Refinement,
    /// General SMT-level properties
    Smt,
    /// Platform API constraints (external API state machines)
    PlatformApi,
}

impl PropertyType {
    /// Get the relevant backends for this property type
    pub fn relevant_backends(&self) -> Vec<BackendId> {
        match self {
            Self::Safety => vec![
                BackendId::Lean4,
                BackendId::Kani,
                BackendId::Verus,
                BackendId::Creusot,
                BackendId::Prusti,
                BackendId::Dafny,
                BackendId::CBMC,
                BackendId::CPAchecker,
                BackendId::SeaHorn,
                BackendId::FramaC,
                BackendId::Infer,
                BackendId::KLEE,
                BackendId::SPIN,
                BackendId::NuSMV,
            ],
            Self::Liveness => vec![
                BackendId::TlaPlus,
                BackendId::Alloy,
                BackendId::Storm,
                BackendId::Prism,
            ],
            Self::Temporal => vec![
                BackendId::TlaPlus,
                BackendId::Apalache,
                BackendId::Storm,
                BackendId::Prism,
                BackendId::SPIN,
                BackendId::NuSMV,
            ],
            Self::Correctness => vec![
                BackendId::Lean4,
                BackendId::Coq,
                BackendId::Isabelle,
                BackendId::Dafny,
                BackendId::Verus,
            ],
            Self::Probabilistic => vec![BackendId::Storm, BackendId::Prism],
            Self::NeuralNetwork => vec![
                BackendId::Marabou,
                BackendId::AlphaBetaCrown,
                BackendId::Eran,
            ],
            Self::SecurityProtocol => {
                vec![BackendId::Tamarin, BackendId::ProVerif, BackendId::Verifpal]
            }
            Self::Refinement => vec![BackendId::TlaPlus, BackendId::Alloy, BackendId::Isabelle],
            Self::Smt => vec![
                BackendId::Z3,
                BackendId::Cvc5,
                BackendId::Yices,
                BackendId::Boolector,
                BackendId::MathSAT,
            ],
            Self::PlatformApi => vec![BackendId::PlatformApi],
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Safety => "Safety properties: invariants, memory safety, resource safety",
            Self::Liveness => "Liveness properties: something eventually happens",
            Self::Temporal => "Temporal logic: sequences of states over time",
            Self::Correctness => "Functional correctness: input/output specifications",
            Self::Probabilistic => "Probabilistic properties: expected values, bounds",
            Self::NeuralNetwork => "Neural network: robustness, reachability",
            Self::SecurityProtocol => "Security protocols: authentication, secrecy",
            Self::Refinement => "Refinement: simulation and abstraction relations",
            Self::Smt => "SMT: satisfiability, theory reasoning",
            Self::PlatformApi => {
                "Platform API: external API state machine constraints (static checkers)"
            }
        }
    }
}

/// A recommendation from an expert
#[derive(Debug, Clone)]
pub struct ExpertRecommendation {
    /// The recommended item (backend, tactic, etc.)
    pub recommendation: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Human-readable explanation
    pub explanation: String,
    /// Supporting evidence from knowledge base
    pub evidence: Vec<Evidence>,
    /// Alternative recommendations
    pub alternatives: Vec<Alternative>,
}

/// Evidence supporting a recommendation
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Source document/chunk
    pub source: String,
    /// Relevant excerpt
    pub excerpt: String,
    /// Relevance score
    pub relevance: f32,
}

/// An alternative recommendation
#[derive(Debug, Clone)]
pub struct Alternative {
    /// The alternative recommendation
    pub recommendation: String,
    /// Why this is an alternative
    pub reason: String,
    /// Confidence relative to primary
    pub relative_confidence: f32,
}

/// Backend recommendation with detailed rationale
#[derive(Debug, Clone)]
pub struct BackendRecommendation {
    /// Primary recommended backend
    pub backend: BackendId,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Why this backend was chosen
    pub rationale: String,
    /// Capabilities of this backend relevant to the query
    pub relevant_capabilities: Vec<String>,
    /// Known limitations for the use case
    pub limitations: Vec<String>,
    /// Alternative backends to consider
    pub alternatives: Vec<BackendAlternative>,
    /// Supporting evidence from documentation
    pub evidence: Vec<Evidence>,
}

/// An alternative backend suggestion
#[derive(Debug, Clone)]
pub struct BackendAlternative {
    /// The alternative backend
    pub backend: BackendId,
    /// Why this is a good alternative
    pub rationale: String,
    /// When to prefer this over the primary
    pub prefer_when: String,
    /// Confidence score
    pub confidence: f32,
}

/// Error explanation from the expert
#[derive(Debug, Clone)]
pub struct ErrorExplanation {
    /// The original error message
    pub original_error: String,
    /// Human-readable explanation
    pub explanation: String,
    /// Root cause analysis
    pub root_cause: String,
    /// Suggested fixes
    pub suggested_fixes: Vec<SuggestedFix>,
    /// Related documentation
    pub related_docs: Vec<Evidence>,
    /// Similar known issues
    pub similar_issues: Vec<String>,
}

/// A suggested fix for an error
#[derive(Debug, Clone)]
pub struct SuggestedFix {
    /// Description of the fix
    pub description: String,
    /// Code example (if applicable)
    pub code_example: Option<String>,
    /// Confidence that this fix will work
    pub confidence: f32,
}

/// Tactic suggestion from the expert
#[derive(Debug, Clone)]
pub struct TacticSuggestion {
    /// Suggested tactic/strategy
    pub tactic: String,
    /// Backend this tactic applies to
    pub backend: BackendId,
    /// When to use this tactic
    pub when_to_use: String,
    /// Expected effect
    pub expected_effect: String,
    /// Usage example
    pub example: Option<String>,
    /// Confidence score
    pub confidence: f32,
    /// Alternative tactics
    pub alternatives: Vec<String>,
}

/// Compilation guidance from the expert
#[derive(Debug, Clone)]
pub struct CompilationGuidance {
    /// The specification or code being compiled
    pub input_summary: String,
    /// Target backend
    pub target_backend: BackendId,
    /// Step-by-step guidance
    pub steps: Vec<CompilationStep>,
    /// Common pitfalls to avoid
    pub pitfalls: Vec<String>,
    /// Best practices for this backend
    pub best_practices: Vec<String>,
    /// Related documentation
    pub related_docs: Vec<Evidence>,
}

/// A step in compilation guidance
#[derive(Debug, Clone)]
pub struct CompilationStep {
    /// Step number
    pub step_number: usize,
    /// Step description
    pub description: String,
    /// Code example (if applicable)
    pub code_example: Option<String>,
    /// What to check after this step
    pub verification: Option<String>,
}

/// Research recommendation from academic papers
#[derive(Debug, Clone)]
pub struct ResearchRecommendation {
    /// The technique or approach being recommended
    pub technique: String,
    /// ArXiv paper IDs supporting this recommendation
    pub paper_ids: Vec<String>,
    /// Paper titles for display
    pub paper_titles: Vec<String>,
    /// How this technique applies to the user's problem
    pub application: String,
    /// Key insights from the papers
    pub key_insights: Vec<String>,
    /// Implementation considerations
    pub implementation_notes: Vec<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Related techniques to also consider
    pub related_techniques: Vec<RelatedTechnique>,
}

/// A related technique mentioned in research
#[derive(Debug, Clone)]
pub struct RelatedTechnique {
    /// Name of the technique
    pub name: String,
    /// Brief description
    pub description: String,
    /// Paper IDs where this is discussed
    pub paper_ids: Vec<String>,
    /// How it relates to the primary recommendation
    pub relationship: String,
}

/// Paper citation for evidence
#[derive(Debug, Clone)]
pub struct PaperCitation {
    /// ArXiv ID
    pub arxiv_id: String,
    /// Paper title
    pub title: String,
    /// Authors
    pub authors: Vec<String>,
    /// Year published
    pub year: u16,
    /// Relevant excerpt from abstract
    pub relevant_excerpt: String,
    /// Relevance score
    pub relevance: f32,
}

/// Research query context
#[derive(Debug, Clone, Default)]
pub struct ResearchContext {
    /// The verification problem or question
    pub problem_description: String,
    /// Property types of interest
    pub property_types: Vec<PropertyType>,
    /// Target backends (if any)
    pub target_backends: Vec<BackendId>,
    /// Keywords to include in search
    pub keywords: Vec<String>,
    /// Maximum number of papers to consider
    pub max_papers: usize,
    /// Minimum recency (year)
    pub min_year: Option<u16>,
}
