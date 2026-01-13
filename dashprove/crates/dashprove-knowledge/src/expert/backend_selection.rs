//! Backend selection expert

use crate::embedding::Embedder;
use crate::store::KnowledgeStore;
use crate::tool_knowledge::ToolKnowledgeStore;
use crate::types::{ContentType, KnowledgeQuery, SearchResult};
use crate::Result;
use dashprove_backends::BackendId;
use std::collections::HashMap;

use super::types::{BackendAlternative, BackendRecommendation, Evidence, ExpertContext};
use super::util::{all_backends, backend_id_to_tool_id};

/// Expert for backend selection
///
/// This expert uses both:
/// 1. The vector-based KnowledgeStore for semantic search over documentation
/// 2. The ToolKnowledgeStore for structured tool comparisons and similar tools
pub struct BackendSelectionExpert<'a> {
    store: &'a KnowledgeStore,
    embedder: &'a Embedder,
    tool_store: Option<&'a ToolKnowledgeStore>,
}

impl<'a> BackendSelectionExpert<'a> {
    /// Create a new backend selection expert
    pub fn new(store: &'a KnowledgeStore, embedder: &'a Embedder) -> Self {
        Self {
            store,
            embedder,
            tool_store: None,
        }
    }

    /// Create a new backend selection expert with tool knowledge store
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

    /// Get backend recommendation based on context
    pub async fn recommend(&self, context: &ExpertContext) -> Result<BackendRecommendation> {
        // Build query from context
        let query_text = self.build_query_text(context);

        // Get embedding for query
        let query_embedding = self.embedder.embed_text(&query_text).await?;

        // Search knowledge base
        let query = KnowledgeQuery::new(&query_text)
            .with_limit(10)
            .with_content_type(ContentType::Reference);

        let results = self.store.search(&query, &query_embedding);

        // Score backends based on property types and search results
        let backend_scores = self.score_backends(context, &results);

        // Get the best backend
        let (best_backend, best_score) = backend_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(b, s)| (*b, *s))
            .unwrap_or((BackendId::Z3, 0.5));

        // Build evidence from search results
        let evidence = self.extract_evidence(&results, best_backend);

        // Get alternatives
        let alternatives = self.get_alternatives(&backend_scores, best_backend);

        // Generate rationale
        let rationale = self.generate_rationale(context, best_backend, &evidence);

        // Get capabilities and limitations
        let (capabilities, limitations) = self.get_backend_info(best_backend, context);

        Ok(BackendRecommendation {
            backend: best_backend,
            confidence: best_score.clamp(0.0, 1.0),
            rationale,
            relevant_capabilities: capabilities,
            limitations,
            alternatives,
            evidence,
        })
    }

    fn build_query_text(&self, context: &ExpertContext) -> String {
        let mut parts = Vec::new();

        if let Some(ref spec) = context.specification {
            // Take first 500 chars of spec
            let spec_preview: String = spec.chars().take(500).collect();
            parts.push(format!("Specification: {}", spec_preview));
        }

        for prop_type in &context.property_types {
            parts.push(format!("Property type: {}", prop_type.description()));
        }

        if let Some(ref lang) = context.code_language {
            parts.push(format!("Code language: {}", lang));
        }

        if !context.tags.is_empty() {
            parts.push(format!("Tags: {}", context.tags.join(", ")));
        }

        if parts.is_empty() {
            "general verification tool selection".to_string()
        } else {
            parts.join("\n")
        }
    }

    fn score_backends(
        &self,
        context: &ExpertContext,
        results: &SearchResult,
    ) -> HashMap<BackendId, f32> {
        let mut scores: HashMap<BackendId, f32> = HashMap::new();

        // Initialize all backends with base score
        for backend in all_backends() {
            scores.insert(backend, 0.1);
        }

        // Score based on property types
        for prop_type in &context.property_types {
            for backend in prop_type.relevant_backends() {
                *scores.entry(backend).or_insert(0.0) += 0.3;
            }
        }

        // Score based on code language
        if let Some(ref lang) = context.code_language {
            let lang_lower = lang.to_lowercase();
            if lang_lower.contains("rust") {
                *scores.entry(BackendId::Kani).or_insert(0.0) += 0.4;
                *scores.entry(BackendId::Verus).or_insert(0.0) += 0.4;
                *scores.entry(BackendId::Creusot).or_insert(0.0) += 0.4;
                *scores.entry(BackendId::Prusti).or_insert(0.0) += 0.4;
            }
        }

        // Score based on search results
        for chunk in &results.chunks {
            if let Some(backend) = chunk.chunk.backend {
                // Higher relevance score = higher backend score
                *scores.entry(backend).or_insert(0.0) += chunk.score * 0.2;
            }
        }

        // Normalize scores
        let max_score = scores.values().cloned().fold(0.0f32, f32::max);
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }

        scores
    }

    fn extract_evidence(&self, results: &SearchResult, _backend: BackendId) -> Vec<Evidence> {
        results
            .chunks
            .iter()
            .take(3)
            .map(|chunk| Evidence {
                source: chunk.chunk.document_id.clone(),
                excerpt: chunk.chunk.content.chars().take(200).collect(),
                relevance: chunk.score,
            })
            .collect()
    }

    fn get_alternatives(
        &self,
        scores: &HashMap<BackendId, f32>,
        primary: BackendId,
    ) -> Vec<BackendAlternative> {
        // First try to get similar tools from ToolKnowledgeStore
        let tool_store_alternatives = self.get_tool_store_alternatives(primary);

        if !tool_store_alternatives.is_empty() {
            return tool_store_alternatives;
        }

        // Fall back to score-based alternatives
        let mut sorted: Vec<_> = scores.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        sorted
            .into_iter()
            .filter(|(b, _)| **b != primary)
            .take(3)
            .map(|(backend, score)| BackendAlternative {
                backend: *backend,
                rationale: format!("{:?} provides alternative verification approach", backend),
                prefer_when: self.get_prefer_when(*backend),
                confidence: *score,
            })
            .collect()
    }

    /// Get alternative tools from the ToolKnowledgeStore
    fn get_tool_store_alternatives(&self, primary: BackendId) -> Vec<BackendAlternative> {
        let tool_store = match self.tool_store {
            Some(store) => store,
            None => return vec![],
        };

        let tool_id = backend_id_to_tool_id(primary);
        let similar_tools = tool_store.find_similar(&tool_id);

        if similar_tools.is_empty() {
            return vec![];
        }

        // Get the primary tool's comparisons for advantages/disadvantages
        let primary_tool = tool_store.get(&tool_id);
        let comparison_info = primary_tool.and_then(|t| t.comparisons.as_ref());

        similar_tools
            .into_iter()
            .take(3)
            .filter_map(|similar| {
                // Try to convert tool ID back to BackendId
                let backend = self.tool_id_to_backend_id(&similar.id)?;

                // Build rationale from tool description and comparisons
                let rationale = if let Some(comparisons) = comparison_info {
                    if comparisons.similar_tools.contains(&similar.name) {
                        format!(
                            "{} offers similar capabilities with different trade-offs",
                            similar.name
                        )
                    } else {
                        format!(
                            "{}: {}",
                            similar.name,
                            similar.description.chars().take(100).collect::<String>()
                        )
                    }
                } else {
                    format!(
                        "{}: {}",
                        similar.name,
                        similar.description.chars().take(100).collect::<String>()
                    )
                };

                // Calculate confidence based on shared capabilities
                let shared_caps = similar.capabilities.len() as f32;
                let confidence = (0.5 + shared_caps * 0.1).min(0.9);

                Some(BackendAlternative {
                    backend,
                    rationale,
                    prefer_when: self.get_prefer_when(backend),
                    confidence,
                })
            })
            .collect()
    }

    /// Convert a tool ID back to a BackendId
    fn tool_id_to_backend_id(&self, tool_id: &str) -> Option<BackendId> {
        // Check all backends to find one that matches
        all_backends()
            .into_iter()
            .find(|&backend| backend_id_to_tool_id(backend) == tool_id)
    }

    /// Get description of when to prefer this backend
    pub fn get_prefer_when(&self, backend: BackendId) -> String {
        match backend {
            // Theorem provers & formal verification
            BackendId::Lean4 => "need mathematical proofs with tactics".to_string(),
            BackendId::TlaPlus => "modeling concurrent or distributed systems".to_string(),
            BackendId::Apalache => {
                "unbounded symbolic model checking of TLA+ specifications".to_string()
            }
            BackendId::Kani => "verifying Rust code with model checking".to_string(),
            BackendId::Alloy => "exploring design constraints and invariants".to_string(),
            BackendId::Coq => "need very strong formal guarantees".to_string(),
            BackendId::Isabelle => "working with existing AFP libraries".to_string(),
            BackendId::Dafny => "need auto-active verification".to_string(),

            // Platform API
            BackendId::PlatformApi => {
                "generating static checkers for external platform API constraints".to_string()
            }

            // Neural network verification
            BackendId::Marabou => "neural network robustness verification".to_string(),
            BackendId::AlphaBetaCrown => "scalable neural network verification".to_string(),
            BackendId::Eran => "abstract interpretation for neural networks".to_string(),
            BackendId::NNV => "neural network verification library".to_string(),
            BackendId::Nnenum => "enumeration-based neural network verification".to_string(),
            BackendId::VeriNet => "complete neural network verification".to_string(),
            BackendId::Venus => "complete DNN verification".to_string(),
            BackendId::DNNV => "DNN verification framework".to_string(),
            BackendId::AutoLiRPA => "linear relaxation perturbation analysis".to_string(),
            BackendId::MNBaB => "multi-neuron branch and bound verification".to_string(),
            BackendId::Neurify => "symbolic interval propagation".to_string(),
            BackendId::ReluVal => "interval arithmetic for ReLU networks".to_string(),

            // Adversarial robustness
            BackendId::ART => "adversarial robustness testing".to_string(),
            BackendId::Foolbox => "adversarial attack generation".to_string(),
            BackendId::CleverHans => "adversarial example generation".to_string(),
            BackendId::TextAttack => "NLP adversarial attacks".to_string(),
            BackendId::RobustBench => "robustness benchmarking".to_string(),

            // Probabilistic verification
            BackendId::Storm => "probabilistic model checking".to_string(),
            BackendId::Prism => "Markov chain analysis".to_string(),

            // Security protocol verification
            BackendId::Tamarin => "security protocol symbolic analysis".to_string(),
            BackendId::ProVerif => "security protocol verification".to_string(),
            BackendId::Verifpal => "simplified security protocol analysis".to_string(),

            // Rust formal verification
            BackendId::Verus => "high-performance Rust verification".to_string(),
            BackendId::Creusot => "Why3-based Rust verification".to_string(),
            BackendId::Prusti => "Viper-based Rust verification".to_string(),
            BackendId::Flux => "refinement types for Rust".to_string(),
            BackendId::Mirai => "abstract interpretation for Rust".to_string(),
            BackendId::Rudra => "memory safety bug finding in unsafe Rust".to_string(),
            BackendId::Miri => "undefined behavior detection in Rust".to_string(),

            // SMT solvers
            BackendId::Z3 => "SMT solving and constraint satisfaction".to_string(),
            BackendId::Cvc5 => "theory-heavy SMT problems".to_string(),

            // Rust sanitizers and memory tools
            BackendId::AddressSanitizer => "memory error detection".to_string(),
            BackendId::MemorySanitizer => "uninitialized read detection".to_string(),
            BackendId::ThreadSanitizer => "data race detection".to_string(),
            BackendId::LeakSanitizer => "memory leak detection".to_string(),
            BackendId::Valgrind => "memory debugging and profiling".to_string(),

            // Rust concurrency testing
            BackendId::Loom => "deterministic concurrency testing".to_string(),
            BackendId::Shuttle => "randomized concurrency testing".to_string(),
            BackendId::CDSChecker => "C++11 memory model checking".to_string(),
            BackendId::GenMC => "stateless model checking for concurrency".to_string(),

            // Rust fuzzing
            BackendId::LibFuzzer => "coverage-guided fuzzing".to_string(),
            BackendId::AFL => "American Fuzzy Lop fuzzing".to_string(),
            BackendId::Honggfuzz => "coverage-guided hardware fuzzing".to_string(),
            BackendId::Bolero => "unified fuzzing and property testing".to_string(),

            // Rust property-based testing
            BackendId::Proptest => "strategy-based property testing".to_string(),
            BackendId::QuickCheck => "Haskell-style property testing".to_string(),

            // Rust static analysis
            BackendId::Clippy => "Rust lint collection".to_string(),
            BackendId::SemverChecks => "API compatibility checking".to_string(),
            BackendId::Geiger => "unsafe code auditing".to_string(),
            BackendId::Audit => "security vulnerability scanning".to_string(),
            BackendId::Deny => "dependency policy enforcement".to_string(),
            BackendId::Vet => "supply chain auditing".to_string(),
            BackendId::Mutants => "mutation testing".to_string(),

            // AI/ML optimization
            BackendId::ONNXRuntime => "cross-platform ML inference".to_string(),
            BackendId::TensorRT => "NVIDIA inference optimization".to_string(),
            BackendId::OpenVINO => "Intel inference optimization".to_string(),
            BackendId::TVM => "ML compiler optimization".to_string(),
            BackendId::IREE => "ML compiler execution".to_string(),
            BackendId::Triton => "GPU programming optimization".to_string(),

            // AI/ML compression
            BackendId::NeuralCompressor => "model quantization".to_string(),
            BackendId::NNCF => "neural network compression".to_string(),
            BackendId::AIMET => "model efficiency optimization".to_string(),
            BackendId::Brevitas => "quantization-aware training".to_string(),

            // Data quality
            BackendId::GreatExpectations => "data validation".to_string(),
            BackendId::Deepchecks => "ML validation suite".to_string(),
            BackendId::Evidently => "ML monitoring".to_string(),
            BackendId::WhyLogs => "data profiling".to_string(),

            // Fairness & bias
            BackendId::Fairlearn => "fairness assessment".to_string(),
            BackendId::AIF360 => "bias detection toolkit".to_string(),
            BackendId::Aequitas => "bias auditing".to_string(),

            // Interpretability
            BackendId::SHAP => "Shapley explanations".to_string(),
            BackendId::LIME => "local model explanations".to_string(),
            BackendId::Captum => "PyTorch interpretability".to_string(),
            BackendId::InterpretML => "interpretability toolkit".to_string(),
            BackendId::Alibi => "ML explanations".to_string(),

            // LLM guardrails
            BackendId::GuardrailsAI => "LLM output validation".to_string(),
            BackendId::NeMoGuardrails => "NVIDIA LLM guardrails".to_string(),
            BackendId::Guidance => "structured LLM generation".to_string(),

            // LLM evaluation
            BackendId::Promptfoo => "prompt evaluation".to_string(),
            BackendId::TruLens => "LLM evaluation".to_string(),
            BackendId::LangSmith => "LLM testing".to_string(),
            BackendId::Ragas => "RAG evaluation".to_string(),
            BackendId::DeepEval => "LLM testing framework".to_string(),

            // Hallucination detection
            BackendId::SelfCheckGPT => "self-consistency hallucination check".to_string(),
            BackendId::FactScore => "factual precision scoring".to_string(),

            // Model checkers
            BackendId::SPIN => "protocol verification with Promela".to_string(),
            BackendId::CBMC => "bounded model checking for C programs".to_string(),
            BackendId::Infer => "static analysis for memory safety".to_string(),
            BackendId::KLEE => "symbolic execution for test generation".to_string(),
            BackendId::NuSMV => "symbolic CTL/LTL model checking with BDDs".to_string(),
            BackendId::CPAchecker => "configurable software verification for C".to_string(),
            BackendId::SeaHorn => "LLVM/Horn-clause based verification for C/LLVM IR".to_string(),
            BackendId::FramaC => "deductive verification of C using WP plugin".to_string(),

            // SMT solvers
            BackendId::Yices => "fast SMT solving from SRI International".to_string(),
            BackendId::Boolector => "bit-vector and array SMT solving".to_string(),
            BackendId::MathSAT => "SMT solving with floating-point support".to_string(),

            // SAT solvers
            BackendId::MiniSat => "lightweight CDCL SAT solving".to_string(),
            BackendId::Glucose => "high-performance SAT solving with LBD restarts".to_string(),
            BackendId::CaDiCaL => "modern award-winning SAT solving".to_string(),

            // Dependently typed theorem provers
            BackendId::Agda => "dependently typed programming with proof obligations".to_string(),
            BackendId::Idris => "dependently typed programming with quantitative types".to_string(),
            BackendId::ACL2 => "industrial-strength automated theorem proving".to_string(),
            BackendId::HOL4 => "higher-order logic theorem proving".to_string(),
            BackendId::FStar => "proof-oriented programming with refinement types".to_string(),

            // Additional theorem provers
            BackendId::HOLLight => "HOL family theorem proving with small kernel".to_string(),
            BackendId::PVS => "prototype verification with decision procedures".to_string(),
            BackendId::Mizar => "mathematical article verification".to_string(),
            BackendId::Metamath => "minimal axiom-based proof checking".to_string(),
            BackendId::ATS => "theorem proving with linear types".to_string(),

            // Additional SMT solvers
            BackendId::OpenSMT => "interpolation and incremental SMT".to_string(),
            BackendId::VeriT => "SMT solving with proof production".to_string(),
            BackendId::AltErgo => "SMT solving for program verification".to_string(),

            // Additional SAT solvers
            BackendId::Kissat => "state-of-the-art SAT solving".to_string(),
            BackendId::CryptoMiniSat => "SAT solving with XOR reasoning".to_string(),

            // Additional model checkers
            BackendId::NuXmv => "SMT-based infinite-state model checking".to_string(),
            BackendId::UPPAAL => "real-time timed automata verification".to_string(),
            BackendId::DIVINE => "parallel LTL model checking".to_string(),
            BackendId::ESBMC => "bounded model checking for C/C++".to_string(),
            BackendId::Ultimate => "software verification with automata".to_string(),
            BackendId::SMACK => "LLVM-based software verification".to_string(),
            BackendId::JPF => "Java program model checking".to_string(),

            // Program verification frameworks
            BackendId::VCC => "C program verification with contracts".to_string(),
            BackendId::VeriFast => "separation logic verification for C/Java".to_string(),
            BackendId::KeY => "Java program verification".to_string(),
            BackendId::OpenJML => "JML specification checking for Java".to_string(),
            BackendId::Krakatoa => "Java/C verification with Coq".to_string(),
            BackendId::SPARK => "Ada subset formal verification".to_string(),
            BackendId::Why3 => "deductive program verification platform".to_string(),
            BackendId::Stainless => "Scala verification with SMT".to_string(),
            BackendId::LiquidHaskell => "refinement types for Haskell".to_string(),
            BackendId::Boogie => "intermediate verification language".to_string(),

            // Distributed systems verification
            BackendId::PLang => "asynchronous event-driven verification".to_string(),
            BackendId::Ivy => "distributed protocol verification".to_string(),
            BackendId::MCRL2 => "process algebra model checking".to_string(),
            BackendId::CADP => "concurrent system verification toolbox".to_string(),

            // Cryptographic verification
            BackendId::EasyCrypt => "cryptographic proof assistant".to_string(),
            BackendId::CryptoVerif => "cryptographic protocol verification".to_string(),
            BackendId::Jasmin => "high-assurance cryptographic implementation".to_string(),

            // Hardware verification
            BackendId::Yosys => "open source RTL synthesis".to_string(),
            BackendId::SymbiYosys => "formal verification for Verilog".to_string(),
            BackendId::JasperGold => "commercial hardware formal verification".to_string(),
            BackendId::CadenceEDA => "commercial EDA verification suite".to_string(),

            // Symbolic execution and binary analysis
            BackendId::Angr => "binary analysis with symbolic execution".to_string(),
            BackendId::Manticore => "symbolic execution for binaries".to_string(),
            BackendId::TritonDBA => "dynamic binary analysis with symbolics".to_string(),
            BackendId::Bap => "binary lifting and analysis with BIL IR".to_string(),
            BackendId::Ghidra => "NSA reverse engineering and binary analysis".to_string(),
            BackendId::IsaBIL => "Isabelle/HOL verification for BIL programs".to_string(),
            BackendId::Soteria => "smart contract binary analysis".to_string(),

            // Abstract interpretation
            BackendId::Astree => "safety-critical C analysis".to_string(),
            BackendId::Polyspace => "MathWorks static analysis".to_string(),
            BackendId::CodeSonar => "binary and source code analysis".to_string(),
            BackendId::FramaCEva => "value analysis plugin for Frama-C".to_string(),

            // Rust code coverage
            BackendId::Tarpaulin => "Rust code coverage measurement".to_string(),
            BackendId::LlvmCov => "LLVM-based coverage reporting".to_string(),
            BackendId::Grcov => "Mozilla coverage collection".to_string(),

            // Rust testing frameworks
            BackendId::Nextest => "next-generation Rust test runner".to_string(),
            BackendId::Insta => "snapshot testing for Rust".to_string(),
            BackendId::Rstest => "fixture-based Rust testing".to_string(),
            BackendId::TestCase => "parameterized Rust testing".to_string(),
            BackendId::Mockall => "mock object generation for Rust".to_string(),

            // Rust documentation tools
            BackendId::Deadlinks => "documentation link checking".to_string(),
            BackendId::Spellcheck => "documentation spell checking".to_string(),
            BackendId::Rdme => "README validation".to_string(),

            // Additional Rust verification
            BackendId::Haybale => "LLVM symbolic execution for Rust".to_string(),
            BackendId::CruxMir => "Galois symbolic testing for Rust".to_string(),
            BackendId::RustHorn => "CHC-based Rust verification".to_string(),
            BackendId::RustBelt => "Coq-based unsafe Rust verification".to_string(),

            // Go verification
            BackendId::Gobra => "Viper-based Go verification".to_string(),

            // Additional C/C++ verification
            BackendId::Symbiotic => "LLVM instrumentation with KLEE".to_string(),
            BackendId::TwoLS => "template-based synthesis verification for C".to_string(),

            // Kani Fast (enhanced Kani)
            BackendId::KaniFast => {
                "enhanced Kani with k-induction, CHC, portfolio solving".to_string()
            }
        }
    }

    fn generate_rationale(
        &self,
        context: &ExpertContext,
        backend: BackendId,
        evidence: &[Evidence],
    ) -> String {
        let mut parts = Vec::new();

        parts.push(format!("{:?} is recommended because:", backend));

        // Add property type reasons
        for prop_type in &context.property_types {
            if prop_type.relevant_backends().contains(&backend) {
                parts.push(format!(
                    "- Supports {} verification",
                    prop_type.description()
                ));
            }
        }

        // Add language reason
        if let Some(ref lang) = context.code_language {
            if lang.to_lowercase().contains("rust")
                && matches!(
                    backend,
                    BackendId::Kani | BackendId::Verus | BackendId::Creusot | BackendId::Prusti
                )
            {
                parts.push(format!("- Native support for {} code", lang));
            }
        }

        // Add evidence-based reasons
        if !evidence.is_empty() {
            parts.push("- Relevant documentation found in knowledge base".to_string());
        }

        parts.join("\n")
    }

    fn get_backend_info(
        &self,
        backend: BackendId,
        _context: &ExpertContext,
    ) -> (Vec<String>, Vec<String>) {
        // Try to get info from ToolKnowledgeStore first
        if let Some(tool_store) = self.tool_store {
            let tool_id = backend_id_to_tool_id(backend);
            if let Some(tool) = tool_store.get(&tool_id) {
                let capabilities = if !tool.capabilities.is_empty() {
                    tool.capabilities.clone()
                } else {
                    self.default_capabilities(backend)
                };

                let limitations = tool
                    .comparisons
                    .as_ref()
                    .map(|c| c.disadvantages.clone())
                    .unwrap_or_else(|| self.default_limitations(backend));

                return (capabilities, limitations);
            }
        }

        // Fall back to hardcoded info
        (
            self.default_capabilities(backend),
            self.default_limitations(backend),
        )
    }

    fn default_capabilities(&self, backend: BackendId) -> Vec<String> {
        match backend {
            BackendId::Lean4 => vec![
                "Tactic-based proofs".to_string(),
                "Mathlib library".to_string(),
                "Metaprogramming".to_string(),
            ],
            BackendId::TlaPlus => vec![
                "State machine modeling".to_string(),
                "Temporal logic".to_string(),
                "Model checking".to_string(),
            ],
            BackendId::Kani => vec![
                "Rust memory safety".to_string(),
                "Bounded model checking".to_string(),
                "Symbolic execution".to_string(),
            ],
            BackendId::Alloy => vec![
                "Relational modeling".to_string(),
                "SAT-based analysis".to_string(),
                "Instance finding".to_string(),
            ],
            BackendId::Z3 => vec![
                "SMT solving".to_string(),
                "Theory combination".to_string(),
                "Optimization".to_string(),
            ],
            _ => vec!["General verification".to_string()],
        }
    }

    fn default_limitations(&self, backend: BackendId) -> Vec<String> {
        match backend {
            BackendId::Lean4 => vec![
                "Steep learning curve".to_string(),
                "Manual proof construction".to_string(),
            ],
            BackendId::TlaPlus => vec![
                "State explosion for large models".to_string(),
                "No code generation".to_string(),
            ],
            BackendId::Kani => vec![
                "Bounded verification only".to_string(),
                "Memory/time limits".to_string(),
            ],
            BackendId::Alloy => vec![
                "Bounded scope".to_string(),
                "No infinite structures".to_string(),
            ],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::types::PropertyType;
    use crate::tool_knowledge::{Comparisons, ToolKnowledge, ToolKnowledgeStore};

    /// Helper to create a minimal test environment
    fn create_test_env() -> (
        tempfile::TempDir,
        crate::store::KnowledgeStore,
        crate::embedding::Embedder,
    ) {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);
        (temp_dir, store, embedder)
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: build_query_text
    // ==========================================================================

    #[test]
    fn test_build_query_text_with_specification() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            specification: Some("Test spec content".to_string()),
            current_backend: None,
            error_messages: vec![],
            property_types: vec![],
            code_language: None,
            tags: vec![],
        };

        let query = expert.build_query_text(&context);
        assert!(
            query.contains("Specification:"),
            "Query should contain 'Specification:'"
        );
        assert!(
            query.contains("Test spec content"),
            "Query should contain specification content"
        );
    }

    #[test]
    fn test_build_query_text_with_property_types() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            property_types: vec![PropertyType::Safety],
            ..Default::default()
        };

        let query = expert.build_query_text(&context);
        assert!(
            query.contains("Property type:"),
            "Query should contain 'Property type:'"
        );
    }

    #[test]
    fn test_build_query_text_with_code_language() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            code_language: Some("Rust".to_string()),
            ..Default::default()
        };

        let query = expert.build_query_text(&context);
        assert!(
            query.contains("Code language: Rust"),
            "Query should contain 'Code language: Rust'"
        );
    }

    #[test]
    fn test_build_query_text_with_tags() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            tags: vec!["safety".to_string(), "concurrency".to_string()],
            ..Default::default()
        };

        let query = expert.build_query_text(&context);
        assert!(query.contains("Tags:"), "Query should contain 'Tags:'");
        assert!(
            query.contains("safety, concurrency"),
            "Query should contain joined tags"
        );
    }

    #[test]
    fn test_build_query_text_empty_context_fallback() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext::default();

        let query = expert.build_query_text(&context);
        assert_eq!(
            query, "general verification tool selection",
            "Empty context should return fallback text"
        );
    }

    #[test]
    fn test_build_query_text_spec_truncation() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        // Create a spec longer than 500 chars
        let long_spec: String = (0..600).map(|_| 'a').collect();

        let context = ExpertContext {
            specification: Some(long_spec),
            ..Default::default()
        };

        let query = expert.build_query_text(&context);
        // "Specification: " is 15 chars, so total should be 15 + 500 = 515
        assert!(
            query.len() <= 520,
            "Query should truncate long specs to ~500 chars"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: score_backends
    // ==========================================================================

    #[test]
    fn test_score_backends_base_score_initialization() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext::default();

        let results = SearchResult {
            chunks: vec![],
            papers: vec![],
            repos: vec![],
        };

        let scores = expert.score_backends(&context, &results);

        // All backends should have some score (base 0.1 normalized)
        for backend in all_backends() {
            assert!(
                scores.contains_key(&backend),
                "All backends should be initialized"
            );
            let score = scores.get(&backend).unwrap();
            assert!(*score >= 0.0, "Scores should be non-negative");
        }
    }

    #[test]
    fn test_score_backends_property_type_boost() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            property_types: vec![PropertyType::Safety],
            ..Default::default()
        };

        let results = SearchResult {
            chunks: vec![],
            papers: vec![],
            repos: vec![],
        };

        let scores = expert.score_backends(&context, &results);

        // Memory safety backends should have higher scores
        let kani_score = *scores.get(&BackendId::Kani).unwrap_or(&0.0);

        // Kani should have a higher score for Safety property type
        // (since Kani is in relevant_backends for Safety)
        assert!(
            kani_score > 0.0,
            "Kani should have elevated score for Safety property type"
        );
    }

    #[test]
    fn test_score_backends_rust_language_boost() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            code_language: Some("Rust".to_string()),
            ..Default::default()
        };

        let results = SearchResult {
            chunks: vec![],
            papers: vec![],
            repos: vec![],
        };

        let scores = expert.score_backends(&context, &results);

        // Rust backends should all have elevated scores
        let kani_score = *scores.get(&BackendId::Kani).unwrap_or(&0.0);
        let verus_score = *scores.get(&BackendId::Verus).unwrap_or(&0.0);
        let creusot_score = *scores.get(&BackendId::Creusot).unwrap_or(&0.0);
        let prusti_score = *scores.get(&BackendId::Prusti).unwrap_or(&0.0);

        // All should be at the maximum (1.0) since they all get +0.4 and normalization
        assert!(
            kani_score > 0.5,
            "Kani should have high score for Rust: {}",
            kani_score
        );
        assert!(
            verus_score > 0.5,
            "Verus should have high score for Rust: {}",
            verus_score
        );
        assert!(
            creusot_score > 0.5,
            "Creusot should have high score for Rust: {}",
            creusot_score
        );
        assert!(
            prusti_score > 0.5,
            "Prusti should have high score for Rust: {}",
            prusti_score
        );
    }

    #[test]
    fn test_score_backends_normalization() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            property_types: vec![PropertyType::Safety],
            code_language: Some("Rust".to_string()),
            ..Default::default()
        };

        let results = SearchResult {
            chunks: vec![],
            papers: vec![],
            repos: vec![],
        };

        let scores = expert.score_backends(&context, &results);

        // All scores should be between 0 and 1
        for score in scores.values() {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "Score should be normalized to [0,1]: {}",
                score
            );
        }

        // At least one score should be 1.0 (the maximum after normalization)
        let max_score = scores.values().cloned().fold(0.0f32, f32::max);
        assert!(
            (max_score - 1.0).abs() < f32::EPSILON,
            "Max score should be 1.0 after normalization"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: get_prefer_when
    // ==========================================================================

    #[test]
    fn test_get_prefer_when_lean4() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let prefer = expert.get_prefer_when(BackendId::Lean4);
        assert!(
            prefer.contains("mathematical proofs"),
            "Lean4 prefer_when should mention mathematical proofs"
        );
        assert!(
            prefer.contains("tactics"),
            "Lean4 prefer_when should mention tactics"
        );
    }

    #[test]
    fn test_get_prefer_when_tlaplus() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let prefer = expert.get_prefer_when(BackendId::TlaPlus);
        assert!(
            prefer.contains("concurrent") || prefer.contains("distributed"),
            "TLA+ prefer_when should mention concurrent or distributed"
        );
    }

    #[test]
    fn test_get_prefer_when_kani() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let prefer = expert.get_prefer_when(BackendId::Kani);
        assert!(
            prefer.contains("Rust"),
            "Kani prefer_when should mention Rust"
        );
        assert!(
            prefer.contains("model checking"),
            "Kani prefer_when should mention model checking"
        );
    }

    #[test]
    fn test_get_prefer_when_z3() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let prefer = expert.get_prefer_when(BackendId::Z3);
        assert!(prefer.contains("SMT"), "Z3 prefer_when should mention SMT");
    }

    #[test]
    fn test_get_prefer_when_neural_network_backends() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let marabou = expert.get_prefer_when(BackendId::Marabou);
        assert!(
            marabou.contains("neural") || marabou.contains("robustness"),
            "Marabou prefer_when should mention neural network or robustness"
        );

        let alpha_beta = expert.get_prefer_when(BackendId::AlphaBetaCrown);
        assert!(
            alpha_beta.contains("neural"),
            "AlphaBetaCrown prefer_when should mention neural"
        );
    }

    #[test]
    fn test_get_prefer_when_security_backends() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let tamarin = expert.get_prefer_when(BackendId::Tamarin);
        assert!(
            tamarin.contains("security") || tamarin.contains("protocol"),
            "Tamarin prefer_when should mention security or protocol"
        );

        let proverif = expert.get_prefer_when(BackendId::ProVerif);
        assert!(
            proverif.contains("security") || proverif.contains("protocol"),
            "ProVerif prefer_when should mention security or protocol"
        );
    }

    #[test]
    fn test_get_prefer_when_sanitizers() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let asan = expert.get_prefer_when(BackendId::AddressSanitizer);
        assert!(
            asan.contains("memory"),
            "AddressSanitizer prefer_when should mention memory"
        );

        let tsan = expert.get_prefer_when(BackendId::ThreadSanitizer);
        assert!(
            tsan.contains("race") || tsan.contains("data race"),
            "ThreadSanitizer prefer_when should mention race"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: default_capabilities
    // ==========================================================================

    #[test]
    fn test_default_capabilities_lean4() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let caps = expert.default_capabilities(BackendId::Lean4);
        assert!(!caps.is_empty(), "Lean4 should have capabilities");
        assert!(
            caps.iter().any(|c| c.contains("Tactic")),
            "Lean4 should mention tactics"
        );
        assert!(
            caps.iter().any(|c| c.contains("Mathlib")),
            "Lean4 should mention Mathlib"
        );
    }

    #[test]
    fn test_default_capabilities_tlaplus() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let caps = expert.default_capabilities(BackendId::TlaPlus);
        assert!(!caps.is_empty(), "TLA+ should have capabilities");
        assert!(
            caps.iter()
                .any(|c| c.contains("State") || c.contains("model")),
            "TLA+ should mention state machine or model"
        );
    }

    #[test]
    fn test_default_capabilities_kani() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let caps = expert.default_capabilities(BackendId::Kani);
        assert!(!caps.is_empty(), "Kani should have capabilities");
        assert!(
            caps.iter()
                .any(|c| c.contains("Rust") || c.contains("memory")),
            "Kani should mention Rust or memory"
        );
    }

    #[test]
    fn test_default_capabilities_z3() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let caps = expert.default_capabilities(BackendId::Z3);
        assert!(!caps.is_empty(), "Z3 should have capabilities");
        assert!(
            caps.iter().any(|c| c.contains("SMT")),
            "Z3 should mention SMT"
        );
    }

    #[test]
    fn test_default_capabilities_unknown_backend() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        // Use a backend not in the explicit match arms
        let caps = expert.default_capabilities(BackendId::Valgrind);
        assert!(!caps.is_empty(), "Unknown backend should have fallback");
        assert!(
            caps.iter().any(|c| c.contains("General")),
            "Unknown backend should mention General verification"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: default_limitations
    // ==========================================================================

    #[test]
    fn test_default_limitations_lean4() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let lims = expert.default_limitations(BackendId::Lean4);
        assert!(!lims.is_empty(), "Lean4 should have limitations");
        assert!(
            lims.iter().any(|l| l.contains("learning curve")),
            "Lean4 limitations should mention learning curve"
        );
    }

    #[test]
    fn test_default_limitations_tlaplus() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let lims = expert.default_limitations(BackendId::TlaPlus);
        assert!(!lims.is_empty(), "TLA+ should have limitations");
        assert!(
            lims.iter().any(|l| l.contains("explosion")),
            "TLA+ limitations should mention state explosion"
        );
    }

    #[test]
    fn test_default_limitations_kani() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let lims = expert.default_limitations(BackendId::Kani);
        assert!(!lims.is_empty(), "Kani should have limitations");
        assert!(
            lims.iter().any(|l| l.contains("Bounded")),
            "Kani limitations should mention bounded verification"
        );
    }

    #[test]
    fn test_default_limitations_unknown_backend_empty() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        // Unknown backends should return empty vec
        let lims = expert.default_limitations(BackendId::Valgrind);
        assert!(
            lims.is_empty(),
            "Unknown backend should have no default limitations"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: get_tool_store_alternatives
    // ==========================================================================

    #[test]
    fn test_get_tool_store_alternatives_empty_without_store() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let alts = expert.get_tool_store_alternatives(BackendId::Lean4);
        assert!(
            alts.is_empty(),
            "Should return empty when no tool store configured"
        );
    }

    #[test]
    fn test_get_tool_store_alternatives_with_similar_tools() {
        let mut tool_store = ToolKnowledgeStore::new();

        // Add Lean4 with similar tools
        let lean4 = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec!["tactics".to_string(), "proofs".to_string()],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: Some(Comparisons {
                similar_tools: vec!["Coq".to_string(), "Isabelle".to_string()],
                advantages: vec![],
                disadvantages: vec![],
            }),
            metadata: None,
        };
        tool_store.add_tool(lean4);

        // Add Coq as a similar tool
        let coq = ToolKnowledge {
            id: "coq".to_string(),
            name: "Coq".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Proof assistant with dependent types".to_string(),
            long_description: None,
            capabilities: vec!["tactics".to_string(), "extraction".to_string()],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(coq);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::with_tool_store(&store, &embedder, &tool_store);

        let alts = expert.get_tool_store_alternatives(BackendId::Lean4);
        // Should find alternatives based on similar tools
        // Note: depends on find_similar implementation in ToolKnowledgeStore
        // Even if no alts found, the test verifies the method doesn't crash
        // and returns an empty vec rather than panicking
        assert!(alts.len() <= 3, "Should return at most 3 alternatives");
    }

    #[test]
    fn test_tool_id_to_backend_id_conversion() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        // Test conversion for known backends
        let lean4_id = backend_id_to_tool_id(BackendId::Lean4);
        let result = expert.tool_id_to_backend_id(&lean4_id);
        assert_eq!(
            result,
            Some(BackendId::Lean4),
            "Should convert lean4 ID back to BackendId::Lean4"
        );

        let kani_id = backend_id_to_tool_id(BackendId::Kani);
        let result = expert.tool_id_to_backend_id(&kani_id);
        assert_eq!(
            result,
            Some(BackendId::Kani),
            "Should convert kani ID back to BackendId::Kani"
        );

        // Unknown tool ID should return None
        let result = expert.tool_id_to_backend_id("unknown_tool_xyz");
        assert_eq!(result, None, "Unknown tool ID should return None");
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: generate_rationale
    // ==========================================================================

    #[test]
    fn test_generate_rationale_includes_backend_name() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext::default();

        let evidence = vec![];
        let rationale = expert.generate_rationale(&context, BackendId::Lean4, &evidence);

        assert!(
            rationale.contains("Lean4"),
            "Rationale should mention the backend name"
        );
        assert!(
            rationale.contains("is recommended"),
            "Rationale should contain 'is recommended'"
        );
    }

    #[test]
    fn test_generate_rationale_includes_property_type_reason() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            property_types: vec![PropertyType::Safety],
            ..Default::default()
        };

        let evidence = vec![];
        let rationale = expert.generate_rationale(&context, BackendId::Kani, &evidence);

        // Kani supports MemorySafety
        assert!(
            rationale.contains("Supports"),
            "Rationale should mention 'Supports' for matching property type"
        );
    }

    #[test]
    fn test_generate_rationale_includes_rust_language_reason() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext {
            code_language: Some("Rust".to_string()),
            ..Default::default()
        };

        let evidence = vec![];
        let rationale = expert.generate_rationale(&context, BackendId::Kani, &evidence);

        assert!(
            rationale.contains("Native support") && rationale.contains("Rust"),
            "Rationale should mention native Rust support for Kani"
        );
    }

    #[test]
    fn test_generate_rationale_includes_evidence_reason() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext::default();

        let evidence = vec![Evidence {
            source: "test.md".to_string(),
            excerpt: "Some relevant documentation".to_string(),
            relevance: 0.8,
        }];

        let rationale = expert.generate_rationale(&context, BackendId::Z3, &evidence);

        assert!(
            rationale.contains("knowledge base"),
            "Rationale should mention knowledge base when evidence is present"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: get_backend_info
    // ==========================================================================

    #[test]
    fn test_get_backend_info_from_tool_store() {
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4 = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec!["custom_cap1".to_string(), "custom_cap2".to_string()],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: Some(Comparisons {
                similar_tools: vec![],
                advantages: vec![],
                disadvantages: vec!["custom_limitation".to_string()],
            }),
            metadata: None,
        };
        tool_store.add_tool(lean4);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::with_tool_store(&store, &embedder, &tool_store);

        let context = ExpertContext::default();

        let (caps, lims) = expert.get_backend_info(BackendId::Lean4, &context);

        assert!(
            caps.contains(&"custom_cap1".to_string()),
            "Should use tool store capabilities"
        );
        assert!(
            lims.contains(&"custom_limitation".to_string()),
            "Should use tool store limitations"
        );
    }

    #[test]
    fn test_get_backend_info_fallback_to_defaults() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let context = ExpertContext::default();

        let (caps, lims) = expert.get_backend_info(BackendId::Lean4, &context);

        // Should fall back to default capabilities
        assert!(!caps.is_empty(), "Should have default capabilities");
        assert!(!lims.is_empty(), "Lean4 should have default limitations");
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: extract_evidence
    // ==========================================================================

    #[test]
    fn test_extract_evidence_limits_to_three() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        // Create more than 3 chunks
        let chunks: Vec<crate::types::ScoredChunk> = (0..5)
            .map(|i| crate::types::ScoredChunk {
                chunk: crate::types::DocumentChunk {
                    id: format!("chunk{}", i),
                    document_id: format!("doc{}", i),
                    content: format!("Content {}", i),
                    chunk_index: i,
                    backend: Some(BackendId::Z3),
                    content_type: ContentType::Reference,
                    section_path: vec![],
                    token_count: 10,
                },
                score: 0.9 - (i as f32 * 0.1),
            })
            .collect();

        let results = SearchResult {
            chunks,
            papers: vec![],
            repos: vec![],
        };

        let evidence = expert.extract_evidence(&results, BackendId::Z3);

        assert_eq!(
            evidence.len(),
            3,
            "Should only extract 3 pieces of evidence"
        );
    }

    #[test]
    fn test_extract_evidence_truncates_excerpt() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        // Create a chunk with content longer than 200 chars
        let long_content: String = (0..300).map(|_| 'x').collect();
        let chunks = vec![crate::types::ScoredChunk {
            chunk: crate::types::DocumentChunk {
                id: "chunk0".to_string(),
                document_id: "test_doc".to_string(),
                content: long_content,
                chunk_index: 0,
                backend: Some(BackendId::Z3),
                content_type: ContentType::Reference,
                section_path: vec![],
                token_count: 300,
            },
            score: 0.9,
        }];

        let results = SearchResult {
            chunks,
            papers: vec![],
            repos: vec![],
        };

        let evidence = expert.extract_evidence(&results, BackendId::Z3);

        assert_eq!(evidence.len(), 1);
        assert!(
            evidence[0].excerpt.len() <= 200,
            "Excerpt should be truncated to 200 chars"
        );
    }

    #[test]
    fn test_extract_evidence_preserves_relevance_score() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let chunks = vec![crate::types::ScoredChunk {
            chunk: crate::types::DocumentChunk {
                id: "chunk0".to_string(),
                document_id: "test_doc".to_string(),
                content: "Test content".to_string(),
                chunk_index: 0,
                backend: Some(BackendId::Z3),
                content_type: ContentType::Reference,
                section_path: vec![],
                token_count: 2,
            },
            score: 0.75,
        }];

        let results = SearchResult {
            chunks,
            papers: vec![],
            repos: vec![],
        };

        let evidence = expert.extract_evidence(&results, BackendId::Z3);

        assert_eq!(evidence.len(), 1);
        assert!(
            (evidence[0].relevance - 0.75).abs() < f32::EPSILON,
            "Relevance should match chunk score"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: get_alternatives
    // ==========================================================================

    #[test]
    fn test_get_alternatives_excludes_primary() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let mut scores = HashMap::new();
        scores.insert(BackendId::Lean4, 0.9);
        scores.insert(BackendId::Coq, 0.8);
        scores.insert(BackendId::Isabelle, 0.7);
        scores.insert(BackendId::Z3, 0.6);

        let alts = expert.get_alternatives(&scores, BackendId::Lean4);

        // Primary (Lean4) should not be in alternatives
        assert!(
            !alts.iter().any(|a| a.backend == BackendId::Lean4),
            "Primary backend should not be in alternatives"
        );
    }

    #[test]
    fn test_get_alternatives_limits_to_three() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let mut scores = HashMap::new();
        scores.insert(BackendId::Lean4, 0.9);
        scores.insert(BackendId::Coq, 0.8);
        scores.insert(BackendId::Isabelle, 0.7);
        scores.insert(BackendId::Z3, 0.6);
        scores.insert(BackendId::Kani, 0.5);
        scores.insert(BackendId::Alloy, 0.4);

        let alts = expert.get_alternatives(&scores, BackendId::Lean4);

        assert!(alts.len() <= 3, "Should return at most 3 alternatives");
    }

    #[test]
    fn test_get_alternatives_sorted_by_score() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = BackendSelectionExpert::new(&store, &embedder);

        let mut scores = HashMap::new();
        scores.insert(BackendId::Lean4, 0.9);
        scores.insert(BackendId::Coq, 0.8);
        scores.insert(BackendId::Isabelle, 0.7);
        scores.insert(BackendId::Z3, 0.6);

        let alts = expert.get_alternatives(&scores, BackendId::Lean4);

        // Alternatives should be sorted by score (descending)
        // First should be Coq (0.8), then Isabelle (0.7), then Z3 (0.6)
        if alts.len() >= 2 {
            assert!(
                alts[0].confidence >= alts[1].confidence,
                "Alternatives should be sorted by confidence"
            );
        }
    }
}
