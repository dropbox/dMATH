//! Expert behaviors for RAG-powered recommendations
//!
//! This module provides expert systems that leverage the knowledge base
//! to provide intelligent recommendations for:
//! - Backend selection
//! - Error explanation
//! - Tactic suggestion
//! - Compilation guidance
//! - Research-backed technique recommendations
//!
//! # ExpertFactory
//!
//! The recommended way to create experts is through the [`ExpertFactory`]:
//!
//! ```ignore
//! use dashprove_knowledge::expert::ExpertFactory;
//!
//! let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
//! let backend_expert = factory.backend_selection();
//! let error_expert = factory.error_explanation();
//! let research_expert = factory.research_recommendation();
//! ```

mod backend_selection;
mod compilation_guidance;
mod error_explanation;
mod factory;
mod research_recommendation;
mod tactic_suggestion;
mod types;
mod util;

#[cfg(test)]
mod tests;

pub use backend_selection::BackendSelectionExpert;
pub use compilation_guidance::CompilationGuidanceExpert;
pub use error_explanation::ErrorExplanationExpert;
pub use factory::ExpertFactory;
pub use research_recommendation::ResearchRecommendationExpert;
pub use tactic_suggestion::TacticSuggestionExpert;
pub use types::*;
pub use util::{
    all_backends, backend_id_to_tool_id, backend_tactic_domain, extract_tactic_from_chunk,
};
