//! Research-aware recommendation expert
//!
//! Provides technique recommendations backed by ArXiv papers and academic research.
//! This expert searches the paper corpus to find relevant verification techniques,
//! proof strategies, and tool recommendations supported by recent research.

use crate::embedding::Embedder;
use crate::store::KnowledgeStore;
use crate::tool_knowledge::ToolKnowledgeStore;
use crate::types::{ArxivPaper, ContentType, KnowledgeQuery};
use crate::Result;
use dashprove_backends::BackendId;

use super::types::{
    PaperCitation, PropertyType, RelatedTechnique, ResearchContext, ResearchRecommendation,
};

/// Expert for research-backed technique recommendations
///
/// This expert uses:
/// 1. The vector-based KnowledgeStore for semantic search over papers
/// 2. ArXiv paper abstracts and metadata for technique extraction
/// 3. Property type to paper category mapping for relevant recommendations
pub struct ResearchRecommendationExpert<'a> {
    store: &'a KnowledgeStore,
    embedder: &'a Embedder,
    tool_store: Option<&'a ToolKnowledgeStore>,
}

impl<'a> ResearchRecommendationExpert<'a> {
    /// Create a new research recommendation expert
    pub fn new(store: &'a KnowledgeStore, embedder: &'a Embedder) -> Self {
        Self {
            store,
            embedder,
            tool_store: None,
        }
    }

    /// Create a new research recommendation expert with tool knowledge store
    pub fn with_tool_store(
        store: &'a KnowledgeStore,
        embedder: &'a Embedder,
        tool_store: &'a ToolKnowledgeStore,
    ) -> Self {
        Self {
            store,
            embedder,
            tool_store: Some(tool_store),
        }
    }

    /// Returns whether this expert has a ToolKnowledgeStore attached
    pub fn has_tool_store(&self) -> bool {
        self.tool_store.is_some()
    }

    /// Recommend techniques for a verification problem based on research
    pub async fn recommend(
        &self,
        context: &ResearchContext,
    ) -> Result<Vec<ResearchRecommendation>> {
        // Build query from context
        let query_text = self.build_query_text(context);
        let query_embedding = self.embedder.embed_text(&query_text).await?;

        let query = KnowledgeQuery {
            text: query_text.clone(),
            backend: context.target_backends.first().copied(),
            content_type: Some(ContentType::Paper),
            tags: context.keywords.clone(),
            limit: context.max_papers.max(10),
            include_papers: true,
            include_repos: false,
        };

        let results = self.store.search(&query, &query_embedding);

        // Extract techniques from papers
        let mut recommendations = Vec::new();

        // Group papers by technique
        let technique_papers = self.group_papers_by_technique(&results.papers);

        for (technique, papers) in technique_papers {
            if papers.is_empty() {
                continue;
            }

            let recommendation = self.build_recommendation(&technique, &papers, context);
            recommendations.push(recommendation);
        }

        // Add recommendations based on property types
        for prop_type in &context.property_types {
            if let Some(rec) = self.recommend_for_property_type(*prop_type, &results.papers) {
                // Avoid duplicates
                if !recommendations.iter().any(|r| r.technique == rec.technique) {
                    recommendations.push(rec);
                }
            }
        }

        // Sort by confidence
        recommendations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top recommendations
        recommendations.truncate(5);

        Ok(recommendations)
    }

    /// Find papers relevant to a specific verification technique
    pub async fn find_papers_for_technique(
        &self,
        technique: &str,
        max_results: usize,
    ) -> Result<Vec<PaperCitation>> {
        let query_embedding = self.embedder.embed_text(technique).await?;

        let query = KnowledgeQuery {
            text: technique.to_string(),
            backend: None,
            content_type: Some(ContentType::Paper),
            tags: vec![],
            limit: max_results,
            include_papers: true,
            include_repos: false,
        };

        let results = self.store.search(&query, &query_embedding);

        Ok(results
            .papers
            .into_iter()
            .map(|sp| self.paper_to_citation(&sp.paper, sp.score))
            .collect())
    }

    /// Get technique recommendations for a specific backend
    pub async fn recommend_for_backend(
        &self,
        backend: BackendId,
        problem: &str,
    ) -> Result<Vec<ResearchRecommendation>> {
        let context = ResearchContext {
            problem_description: problem.to_string(),
            property_types: vec![],
            target_backends: vec![backend],
            keywords: vec![],
            max_papers: 10,
            min_year: None,
        };
        self.recommend(&context).await
    }

    // Private helper methods

    fn build_query_text(&self, context: &ResearchContext) -> String {
        let mut parts = vec![context.problem_description.clone()];

        // Add property type keywords
        for prop_type in &context.property_types {
            parts.push(property_type_keywords(*prop_type).to_string());
        }

        // Add backend keywords
        for backend in &context.target_backends {
            parts.push(backend_to_research_keywords(*backend).to_string());
        }

        // Add user keywords
        parts.extend(context.keywords.iter().cloned());

        parts.join(" ")
    }

    fn group_papers_by_technique<'b>(
        &self,
        papers: &'b [crate::types::ScoredPaper],
    ) -> Vec<(String, Vec<&'b crate::types::ScoredPaper>)> {
        // Extract techniques from paper abstracts and titles
        let mut technique_map: std::collections::HashMap<
            String,
            Vec<&'b crate::types::ScoredPaper>,
        > = std::collections::HashMap::new();

        for sp in papers {
            let techniques = self.extract_techniques_from_paper(&sp.paper);
            for technique in techniques {
                technique_map.entry(technique).or_default().push(sp);
            }
        }

        // Convert to vec and sort by paper count
        let mut result: Vec<_> = technique_map.into_iter().collect();
        result.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        result
    }

    fn extract_techniques_from_paper(&self, paper: &ArxivPaper) -> Vec<String> {
        let mut techniques = Vec::new();
        let text = format!("{} {}", paper.title, paper.abstract_text).to_lowercase();

        // Common verification technique keywords
        let technique_patterns = [
            ("bounded model checking", "Bounded Model Checking"),
            ("symbolic execution", "Symbolic Execution"),
            ("abstract interpretation", "Abstract Interpretation"),
            ("separation logic", "Separation Logic"),
            ("hoare logic", "Hoare Logic"),
            ("refinement type", "Refinement Types"),
            ("liquid type", "Liquid Types"),
            ("smt", "SMT Solving"),
            ("sat", "SAT Solving"),
            ("theorem proving", "Theorem Proving"),
            ("model checking", "Model Checking"),
            ("static analysis", "Static Analysis"),
            ("program synthesis", "Program Synthesis"),
            ("counterexample", "Counterexample-Guided"),
            ("inductive invariant", "Inductive Invariants"),
            ("interpolation", "Craig Interpolation"),
            ("predicate abstraction", "Predicate Abstraction"),
            ("k-induction", "K-Induction"),
            ("property directed", "Property-Directed Reachability"),
            ("ic3", "IC3/PDR"),
            ("neural network verification", "Neural Network Verification"),
            ("robustness verification", "Robustness Verification"),
            ("lipschitz", "Lipschitz Analysis"),
            ("interval analysis", "Interval Analysis"),
            ("zonotope", "Zonotope Abstraction"),
            (
                "probabilistic model checking",
                "Probabilistic Model Checking",
            ),
            ("markov chain", "Markov Chain Analysis"),
            ("tla+", "TLA+ Specification"),
            ("alloy", "Alloy Modeling"),
            ("lean", "Lean 4 Proof"),
            ("coq", "Coq Proof"),
            ("isabelle", "Isabelle/HOL"),
            ("dafny", "Dafny Verification"),
            ("rust verification", "Rust Verification"),
            ("unsafe code", "Unsafe Code Verification"),
            ("memory safety", "Memory Safety"),
            ("concurrency", "Concurrency Verification"),
            ("data race", "Data Race Detection"),
            ("deadlock", "Deadlock Detection"),
            ("termination", "Termination Analysis"),
            ("liveness", "Liveness Verification"),
            ("fairness", "Fairness Checking"),
            ("bisimulation", "Bisimulation"),
            ("simulation relation", "Simulation Relations"),
            ("llm", "LLM-Assisted Proving"),
            ("neural theorem", "Neural Theorem Proving"),
            ("autoformalization", "Autoformalization"),
        ];

        for (pattern, name) in technique_patterns {
            if text.contains(pattern) {
                techniques.push(name.to_string());
            }
        }

        // Also use paper categories for technique inference
        for cat in &paper.categories {
            if let Some(tech) = category_to_technique(cat) {
                if !techniques.contains(&tech.to_string()) {
                    techniques.push(tech.to_string());
                }
            }
        }

        if techniques.is_empty() {
            // Default based on primary category
            if let Some(tech) = category_to_technique(&paper.primary_category) {
                techniques.push(tech.to_string());
            }
        }

        techniques
    }

    fn build_recommendation(
        &self,
        technique: &str,
        papers: &[&crate::types::ScoredPaper],
        context: &ResearchContext,
    ) -> ResearchRecommendation {
        let paper_ids: Vec<String> = papers.iter().map(|p| p.paper.arxiv_id.clone()).collect();
        let paper_titles: Vec<String> = papers.iter().map(|p| p.paper.title.clone()).collect();

        // Extract key insights from abstracts
        let key_insights: Vec<String> = papers
            .iter()
            .take(3)
            .filter_map(|p| self.extract_key_insight(&p.paper.abstract_text))
            .collect();

        // Generate application note
        let application = self.generate_application_note(technique, context);

        // Generate implementation notes
        let implementation_notes = self.generate_implementation_notes(technique);

        // Calculate confidence based on paper count and relevance
        let avg_score: f32 = papers.iter().map(|p| p.score).sum::<f32>() / papers.len() as f32;
        let confidence = (avg_score * 0.6 + (papers.len().min(5) as f32 / 5.0) * 0.4).min(0.95);

        // Find related techniques
        let related_techniques = self.find_related_techniques(technique, papers);

        ResearchRecommendation {
            technique: technique.to_string(),
            paper_ids,
            paper_titles,
            application,
            key_insights,
            implementation_notes,
            confidence,
            related_techniques,
        }
    }

    fn extract_key_insight(&self, abstract_text: &str) -> Option<String> {
        // Extract the first sentence that contains a result or contribution
        let indicators = [
            "we show",
            "we prove",
            "we present",
            "we demonstrate",
            "we propose",
        ];

        for sentence in abstract_text.split('.') {
            let lower = sentence.to_lowercase();
            for indicator in indicators {
                if lower.contains(indicator) {
                    let trimmed = sentence.trim();
                    if trimmed.len() > 20 && trimmed.len() < 300 {
                        return Some(format!("{}.", trimmed));
                    }
                }
            }
        }

        // Fall back to first substantive sentence
        abstract_text
            .split('.')
            .find(|s| s.len() > 50)
            .map(|s| format!("{}.", s.trim()))
    }

    fn generate_application_note(&self, technique: &str, context: &ResearchContext) -> String {
        // Generate a note on how this technique applies to the user's problem
        let problem = if context.problem_description.is_empty() {
            "your verification problem"
        } else {
            &context.problem_description
        };

        format!(
            "{} can be applied to {} by leveraging its capabilities for automated reasoning and proof generation.",
            technique, problem
        )
    }

    fn generate_implementation_notes(&self, technique: &str) -> Vec<String> {
        // Generate implementation guidance based on technique
        let mut notes = Vec::new();

        match technique {
            "Bounded Model Checking" => {
                notes.push("Set appropriate bounds for loop unrolling and array sizes".to_string());
                notes.push("Consider using incremental BMC for better scalability".to_string());
                notes.push("Use counterexamples to guide debugging".to_string());
            }
            "Symbolic Execution" => {
                notes.push("Manage path explosion with path merging or prioritization".to_string());
                notes
                    .push("Use concolic execution for better coverage of complex code".to_string());
                notes.push("Consider memory models carefully for pointer analysis".to_string());
            }
            "SMT Solving" => {
                notes.push("Choose appropriate theories (LIA, LRA, BV, etc.)".to_string());
                notes.push("Use incremental solving for related queries".to_string());
                notes.push("Consider theory combination and quantifier handling".to_string());
            }
            "Theorem Proving" => {
                notes.push("Start with automation (simp, auto) before manual proof".to_string());
                notes.push("Use lemmas to decompose complex proofs".to_string());
                notes.push("Leverage proof libraries for common patterns".to_string());
            }
            "Neural Network Verification" => {
                notes.push("Consider network architecture for verifier selection".to_string());
                notes.push("Start with local robustness before global properties".to_string());
                notes.push("Use abstraction techniques for scalability".to_string());
            }
            "Model Checking" => {
                notes.push("Abstract the system to manage state space".to_string());
                notes.push("Use partial order reduction for concurrent systems".to_string());
                notes.push("Start with safety properties before liveness".to_string());
            }
            _ => {
                notes.push(format!(
                    "Consult tool documentation for {} setup",
                    technique
                ));
                notes.push("Start with simple examples to validate the approach".to_string());
                notes.push("Consider combining with complementary techniques".to_string());
            }
        }

        notes
    }

    fn find_related_techniques(
        &self,
        technique: &str,
        papers: &[&crate::types::ScoredPaper],
    ) -> Vec<RelatedTechnique> {
        let mut related = Vec::new();
        let technique_relations = get_technique_relations(technique);

        for (related_name, description, relationship) in technique_relations {
            // Check if any paper mentions this related technique
            let related_paper_ids: Vec<String> = papers
                .iter()
                .filter(|p| {
                    let text =
                        format!("{} {}", p.paper.title, p.paper.abstract_text).to_lowercase();
                    text.contains(&related_name.to_lowercase())
                })
                .map(|p| p.paper.arxiv_id.clone())
                .collect();

            related.push(RelatedTechnique {
                name: related_name.to_string(),
                description: description.to_string(),
                paper_ids: related_paper_ids,
                relationship: relationship.to_string(),
            });
        }

        related
    }

    fn recommend_for_property_type(
        &self,
        prop_type: PropertyType,
        papers: &[crate::types::ScoredPaper],
    ) -> Option<ResearchRecommendation> {
        let (technique, description) = match prop_type {
            PropertyType::Safety => (
                "Bounded Model Checking",
                "Effective for finding safety violations in bounded executions",
            ),
            PropertyType::Liveness => (
                "LTL Model Checking",
                "Handles eventually and always-eventually properties",
            ),
            PropertyType::Temporal => (
                "Temporal Logic Model Checking",
                "Verifies properties over execution traces",
            ),
            PropertyType::Correctness => (
                "Deductive Verification",
                "Proves functional correctness using contracts",
            ),
            PropertyType::Probabilistic => (
                "Probabilistic Model Checking",
                "Computes probability bounds and expected values",
            ),
            PropertyType::NeuralNetwork => (
                "Neural Network Verification",
                "Verifies robustness and reachability of neural networks",
            ),
            PropertyType::SecurityProtocol => (
                "Protocol Verification",
                "Proves security properties of cryptographic protocols",
            ),
            PropertyType::Refinement => (
                "Refinement Checking",
                "Verifies simulation and refinement relations",
            ),
            PropertyType::Smt => (
                "SMT Solving",
                "Satisfiability modulo theories for constraint solving",
            ),
            PropertyType::PlatformApi => (
                "Static Analysis",
                "Checks API usage patterns and state machine conformance",
            ),
        };

        // Find relevant papers for this technique
        let relevant_papers: Vec<_> = papers
            .iter()
            .filter(|p| {
                let text = format!("{} {}", p.paper.title, p.paper.abstract_text).to_lowercase();
                text.contains(&technique.to_lowercase())
                    || text.contains(&prop_type.description().to_lowercase())
            })
            .collect();

        if relevant_papers.is_empty() {
            // Still return the recommendation without paper backing
            return Some(ResearchRecommendation {
                technique: technique.to_string(),
                paper_ids: vec![],
                paper_titles: vec![],
                application: description.to_string(),
                key_insights: vec![format!(
                    "Recommended approach for {:?} properties",
                    prop_type
                )],
                implementation_notes: self.generate_implementation_notes(technique),
                confidence: 0.5,
                related_techniques: vec![],
            });
        }

        let paper_ids: Vec<String> = relevant_papers
            .iter()
            .take(3)
            .map(|p| p.paper.arxiv_id.clone())
            .collect();
        let paper_titles: Vec<String> = relevant_papers
            .iter()
            .take(3)
            .map(|p| p.paper.title.clone())
            .collect();

        Some(ResearchRecommendation {
            technique: technique.to_string(),
            paper_ids,
            paper_titles,
            application: description.to_string(),
            key_insights: relevant_papers
                .iter()
                .take(2)
                .filter_map(|p| self.extract_key_insight(&p.paper.abstract_text))
                .collect(),
            implementation_notes: self.generate_implementation_notes(technique),
            confidence: 0.7,
            related_techniques: vec![],
        })
    }

    fn paper_to_citation(&self, paper: &ArxivPaper, relevance: f32) -> PaperCitation {
        // Extract year from published date
        let year = paper
            .published
            .format("%Y")
            .to_string()
            .parse()
            .unwrap_or(2024);

        // Get first relevant sentence from abstract
        let relevant_excerpt = self
            .extract_key_insight(&paper.abstract_text)
            .unwrap_or_else(|| paper.abstract_text.chars().take(200).collect::<String>() + "...");

        PaperCitation {
            arxiv_id: paper.arxiv_id.clone(),
            title: paper.title.clone(),
            authors: paper.authors.clone(),
            year,
            relevant_excerpt,
            relevance,
        }
    }
}

/// Map property type to search keywords
fn property_type_keywords(prop_type: PropertyType) -> &'static str {
    match prop_type {
        PropertyType::Safety => "safety invariant assertion verification",
        PropertyType::Liveness => "liveness eventually fairness progress",
        PropertyType::Temporal => "temporal logic LTL CTL model checking",
        PropertyType::Correctness => "functional correctness specification",
        PropertyType::Probabilistic => "probabilistic model checking PCTL",
        PropertyType::NeuralNetwork => "neural network verification robustness",
        PropertyType::SecurityProtocol => "security protocol verification",
        PropertyType::Refinement => "refinement simulation abstraction",
        PropertyType::Smt => "SMT satisfiability modulo theories",
        PropertyType::PlatformApi => "API verification static analysis",
    }
}

/// Map backend to research keywords
fn backend_to_research_keywords(backend: BackendId) -> &'static str {
    match backend {
        BackendId::Lean4 => "Lean theorem prover dependent types",
        BackendId::Coq => "Coq proof assistant type theory",
        BackendId::Isabelle => "Isabelle HOL theorem proving",
        BackendId::TlaPlus => "TLA+ temporal logic specification",
        BackendId::Alloy => "Alloy relational modeling",
        BackendId::Kani => "Rust verification model checking",
        BackendId::Verus => "Verus Rust linear types verification",
        BackendId::Dafny => "Dafny program verification",
        BackendId::Z3 => "Z3 SMT solver",
        BackendId::Cvc5 => "CVC5 SMT solver",
        BackendId::Marabou => "Marabou neural network verification",
        BackendId::Storm => "Storm probabilistic model checker",
        BackendId::Prism => "PRISM probabilistic model checker",
        BackendId::Tamarin => "Tamarin security protocol",
        BackendId::ProVerif => "ProVerif security protocol",
        _ => "formal verification",
    }
}

/// Map ArXiv category to technique
fn category_to_technique(category: &str) -> Option<&'static str> {
    match category {
        "cs.LO" => Some("Logic and Verification"),
        "cs.PL" => Some("Programming Languages"),
        "cs.SE" => Some("Software Engineering"),
        "cs.FL" => Some("Formal Languages"),
        "cs.LG" => Some("Machine Learning"),
        "cs.AI" => Some("Artificial Intelligence"),
        "cs.CR" => Some("Cryptography and Security"),
        "cs.DC" => Some("Distributed Computing"),
        "cs.SY" => Some("Systems and Control"),
        "stat.ML" => Some("Statistical Machine Learning"),
        _ => None,
    }
}

/// Get related techniques for a given technique
fn get_technique_relations(technique: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    match technique {
        "Bounded Model Checking" => vec![
            (
                "K-Induction",
                "Extends BMC with inductive reasoning",
                "complementary",
            ),
            (
                "IC3/PDR",
                "Property-directed reachability",
                "alternative for invariants",
            ),
            (
                "CEGAR",
                "Counterexample-guided abstraction refinement",
                "scalability enhancement",
            ),
        ],
        "Symbolic Execution" => vec![
            (
                "Concolic Testing",
                "Combines concrete and symbolic execution",
                "practical variant",
            ),
            (
                "Path Merging",
                "Reduces path explosion",
                "scalability technique",
            ),
            (
                "Selective Symbolic Execution",
                "Focuses on specific code paths",
                "efficiency improvement",
            ),
        ],
        "Theorem Proving" => vec![
            (
                "SMT Solving",
                "Automated theory reasoning",
                "automation backend",
            ),
            (
                "Proof Automation",
                "Automated tactic application",
                "productivity tool",
            ),
            (
                "Proof Repair",
                "Automatic proof maintenance",
                "maintenance aid",
            ),
        ],
        "Neural Network Verification" => vec![
            (
                "Abstract Interpretation",
                "Over-approximation for scalability",
                "scalability technique",
            ),
            (
                "Lipschitz Analysis",
                "Bound propagation through layers",
                "robustness analysis",
            ),
            (
                "MILP Verification",
                "Mixed-integer programming formulation",
                "exact method",
            ),
        ],
        "Model Checking" => vec![
            (
                "Partial Order Reduction",
                "Reduces state space for concurrency",
                "efficiency technique",
            ),
            (
                "Symmetry Reduction",
                "Exploits structural symmetry",
                "efficiency technique",
            ),
            (
                "Abstraction",
                "Reduces state space complexity",
                "scalability technique",
            ),
        ],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{Embedder, EmbeddingModel};
    use crate::store::KnowledgeStore;
    use std::path::PathBuf;

    fn create_test_store() -> KnowledgeStore {
        // Use default embedding dimensions for SentenceTransformers (384)
        KnowledgeStore::new(PathBuf::from("/tmp/test_store"), 384)
    }

    fn create_test_embedder() -> Embedder {
        // Use SentenceTransformers as the test model (local, no API key needed)
        Embedder::new(EmbeddingModel::SentenceTransformers)
    }

    #[tokio::test]
    async fn test_research_expert_creation() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let expert = ResearchRecommendationExpert::new(&store, &embedder);
        assert!(!expert.has_tool_store());
    }

    #[tokio::test]
    async fn test_research_context_default() {
        let context = ResearchContext::default();
        assert!(context.problem_description.is_empty());
        assert!(context.property_types.is_empty());
        assert!(context.target_backends.is_empty());
        assert!(context.keywords.is_empty());
        assert_eq!(context.max_papers, 0);
        assert!(context.min_year.is_none());
    }

    #[tokio::test]
    async fn test_property_type_keywords() {
        assert!(property_type_keywords(PropertyType::Safety).contains("safety"));
        assert!(property_type_keywords(PropertyType::Liveness).contains("liveness"));
        assert!(property_type_keywords(PropertyType::NeuralNetwork).contains("neural"));
    }

    #[tokio::test]
    async fn test_backend_to_research_keywords() {
        assert!(backend_to_research_keywords(BackendId::Lean4).contains("Lean"));
        assert!(backend_to_research_keywords(BackendId::Kani).contains("Rust"));
        assert!(backend_to_research_keywords(BackendId::TlaPlus).contains("TLA"));
    }

    #[tokio::test]
    async fn test_category_to_technique() {
        assert_eq!(
            category_to_technique("cs.LO"),
            Some("Logic and Verification")
        );
        assert_eq!(
            category_to_technique("cs.PL"),
            Some("Programming Languages")
        );
        assert!(category_to_technique("unknown").is_none());
    }

    #[tokio::test]
    async fn test_get_technique_relations() {
        let relations = get_technique_relations("Bounded Model Checking");
        assert!(!relations.is_empty());
        assert!(relations.iter().any(|(name, _, _)| *name == "K-Induction"));
    }

    #[tokio::test]
    async fn test_recommend_empty_context() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let expert = ResearchRecommendationExpert::new(&store, &embedder);

        let context = ResearchContext {
            problem_description: "test problem".to_string(),
            property_types: vec![PropertyType::Safety],
            target_backends: vec![],
            keywords: vec![],
            max_papers: 5,
            min_year: None,
        };

        let recommendations = expert.recommend(&context).await.unwrap();
        // Should return at least property-type-based recommendations
        assert!(recommendations.iter().any(|r| r.confidence > 0.0));
    }

    #[tokio::test]
    async fn test_find_papers_for_technique() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let expert = ResearchRecommendationExpert::new(&store, &embedder);

        let citations = expert
            .find_papers_for_technique("bounded model checking", 5)
            .await
            .unwrap();
        // Empty store returns empty results
        assert!(citations.is_empty());
    }

    #[tokio::test]
    async fn test_recommend_for_backend() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let expert = ResearchRecommendationExpert::new(&store, &embedder);

        let recommendations = expert
            .recommend_for_backend(BackendId::Kani, "verify memory safety")
            .await
            .unwrap();
        // Returns recommendations (may be property-based fallbacks)
        assert!(recommendations.is_empty() || recommendations[0].confidence > 0.0);
    }

    #[test]
    fn test_research_recommendation_structure() {
        let rec = ResearchRecommendation {
            technique: "Test Technique".to_string(),
            paper_ids: vec!["2401.12345".to_string()],
            paper_titles: vec!["Test Paper".to_string()],
            application: "Test application".to_string(),
            key_insights: vec!["Insight 1".to_string()],
            implementation_notes: vec!["Note 1".to_string()],
            confidence: 0.8,
            related_techniques: vec![],
        };
        assert_eq!(rec.technique, "Test Technique");
        assert_eq!(rec.confidence, 0.8);
    }

    #[test]
    fn test_related_technique_structure() {
        let related = RelatedTechnique {
            name: "Related".to_string(),
            description: "A related technique".to_string(),
            paper_ids: vec![],
            relationship: "complementary".to_string(),
        };
        assert_eq!(related.name, "Related");
    }

    #[test]
    fn test_paper_citation_structure() {
        let citation = PaperCitation {
            arxiv_id: "2401.12345".to_string(),
            title: "Test Paper".to_string(),
            authors: vec!["Author One".to_string()],
            year: 2024,
            relevant_excerpt: "This paper presents...".to_string(),
            relevance: 0.9,
        };
        assert_eq!(citation.year, 2024);
        assert_eq!(citation.relevance, 0.9);
    }
}
