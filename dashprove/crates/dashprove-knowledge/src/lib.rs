//! DashProve Knowledge Base
//!
//! This crate provides:
//! - Documentation collection for all 20 verification backends
//! - ArXiv paper fetching for research library
//! - GitHub trending analysis
//! - Chunking and embedding for RAG pipeline
//! - Vector store for semantic search
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Knowledge Base                           │
//! ├─────────────────┬─────────────────┬────────────────────────┤
//! │  Documentation  │  Research       │  Structured            │
//! │  Collector      │  Library        │  Knowledge             │
//! ├─────────────────┼─────────────────┼────────────────────────┤
//! │  - HTML fetch   │  - ArXiv API    │  - Grammars            │
//! │  - PDF extract  │  - GitHub API   │  - Output schemas      │
//! │  - Git clone    │  - Paper parse  │  - USL mappings        │
//! └─────────────────┴─────────────────┴────────────────────────┘
//!                           │
//!                           ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Processing Pipeline                       │
//! ├─────────────────┬─────────────────┬────────────────────────┤
//! │  Chunking       │  Embedding      │  Indexing              │
//! │  - Semantic     │  - OpenAI       │  - Vector store        │
//! │  - Section      │  - Local        │  - Metadata            │
//! └─────────────────┴─────────────────┴────────────────────────┘
//! ```

pub mod chunking;
pub mod collector;
pub mod doc_fetcher;
pub mod embedding;
pub mod expert;
pub mod pdf;
pub mod research;
pub mod store;
pub mod tool_knowledge;
pub mod types;

pub use chunking::{Chunker, ChunkingConfig};
pub use collector::DocumentationCollector;
pub use doc_fetcher::{fetch_priority_docs, DocFetcher, DocMetadata, FetchResult};
pub use embedding::{Embedder, EmbeddingModel};
pub use expert::{
    backend_id_to_tool_id, BackendAlternative, BackendRecommendation, BackendSelectionExpert,
    CompilationGuidance, CompilationGuidanceExpert, CompilationStep, ErrorExplanation,
    ErrorExplanationExpert, Evidence, ExpertContext, ExpertFactory, ExpertRecommendation,
    PropertyType, SuggestedFix, TacticSuggestion, TacticSuggestionExpert,
};
pub use pdf::{clean_pdf_text, extract_sections, PaperSection, PdfProcessor, ProcessingStats};
pub use research::{ArxivFetcher, GithubSearcher};
pub use store::{EmbeddedPaper, KnowledgeStore, PaperSearchEngine, VectorStore};
pub use tool_knowledge::{
    load_default_tool_knowledge, Comparisons, DocumentationUrls, EntryMetadata, ErrorMatch,
    ErrorPattern, InstallMethod, InstallationInfo, IntegrationInfo, PerformanceInfo, Tactic,
    ToolKnowledge, ToolKnowledgeStore,
};
pub use types::*;

use dashprove_backends::BackendId;
use std::path::PathBuf;
use thiserror::Error;

/// Errors from knowledge base operations
#[derive(Error, Debug)]
pub enum KnowledgeError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("API rate limited: retry after {0}s")]
    RateLimited(u64),

    #[error("Backend {0:?} documentation not found")]
    BackendNotFound(BackendId),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Store error: {0}")]
    StoreError(String),
}

pub type Result<T> = std::result::Result<T, KnowledgeError>;

/// Configuration for the knowledge base
#[derive(Debug, Clone)]
pub struct KnowledgeConfig {
    /// Base directory for storing knowledge
    pub base_dir: PathBuf,
    /// Maximum concurrent HTTP requests
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds
    pub request_timeout: u64,
    /// Embedding model to use
    pub embedding_model: EmbeddingModel,
    /// ArXiv search parameters
    pub arxiv_config: ArxivConfig,
    /// GitHub search parameters
    pub github_config: GithubConfig,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            // Store within project directory for self-contained system
            base_dir: PathBuf::from("data/knowledge"),
            max_concurrent_requests: 5,
            request_timeout: 30,
            embedding_model: EmbeddingModel::default(),
            arxiv_config: ArxivConfig::default(),
            github_config: GithubConfig::default(),
        }
    }
}

/// ArXiv search configuration
#[derive(Debug, Clone)]
pub struct ArxivConfig {
    /// Search categories
    pub categories: Vec<String>,
    /// Start date for papers (YYYY-MM-DD)
    pub start_date: String,
    /// Maximum papers per category
    pub max_results_per_category: usize,
}

impl Default for ArxivConfig {
    fn default() -> Self {
        Self {
            categories: vec![
                "cs.LO".to_string(), // Logic in Computer Science
                "cs.PL".to_string(), // Programming Languages
                "cs.SE".to_string(), // Software Engineering
                "cs.AI".to_string(), // Artificial Intelligence
                "cs.CR".to_string(), // Cryptography and Security
                "cs.LG".to_string(), // Machine Learning
            ],
            start_date: "2024-01-01".to_string(),
            max_results_per_category: 100,
        }
    }
}

/// GitHub search configuration
#[derive(Debug, Clone)]
pub struct GithubConfig {
    /// GitHub API token (optional, increases rate limit)
    pub api_token: Option<String>,
    /// Minimum stars for repositories
    pub min_stars: usize,
    /// Search queries
    pub queries: Vec<String>,
}

impl Default for GithubConfig {
    fn default() -> Self {
        Self {
            api_token: std::env::var("GITHUB_TOKEN").ok(),
            min_stars: 20,
            queries: vec![
                "formal verification".to_string(),
                "theorem prover".to_string(),
                "proof assistant".to_string(),
                "SMT solver".to_string(),
                "rust verification".to_string(),
            ],
        }
    }
}

/// Backend documentation sources
pub struct BackendDocSource {
    pub backend: BackendId,
    pub name: &'static str,
    pub docs_url: &'static str,
    pub tutorial_url: Option<&'static str>,
    pub api_url: Option<&'static str>,
    pub examples_repo: Option<&'static str>,
}

/// Get documentation sources for all backends
pub fn get_all_backend_sources() -> Vec<BackendDocSource> {
    vec![
        // Core theorem provers
        BackendDocSource {
            backend: BackendId::Lean4,
            name: "Lean 4",
            docs_url: "https://lean-lang.org/lean4/doc/",
            tutorial_url: Some("https://lean-lang.org/theorem_proving_in_lean4/"),
            api_url: Some("https://leanprover-community.github.io/mathlib4_docs/"),
            examples_repo: Some("https://github.com/leanprover-community/mathlib4"),
        },
        BackendDocSource {
            backend: BackendId::TlaPlus,
            name: "TLA+",
            docs_url: "https://lamport.azurewebsites.net/tla/tla.html",
            tutorial_url: Some("https://learntla.com/"),
            api_url: None,
            examples_repo: Some("https://github.com/tlaplus/Examples"),
        },
        BackendDocSource {
            backend: BackendId::Kani,
            name: "Kani",
            docs_url: "https://model-checking.github.io/kani/",
            tutorial_url: Some("https://model-checking.github.io/kani/getting-started.html"),
            api_url: Some("https://model-checking.github.io/kani/reference/attributes.html"),
            examples_repo: Some("https://github.com/model-checking/kani/tree/main/tests"),
        },
        BackendDocSource {
            backend: BackendId::Alloy,
            name: "Alloy",
            docs_url: "https://alloytools.org/documentation.html",
            tutorial_url: Some("https://alloytools.org/tutorials/online/"),
            api_url: None,
            examples_repo: None,
        },
        BackendDocSource {
            backend: BackendId::Isabelle,
            name: "Isabelle",
            docs_url: "https://isabelle.in.tum.de/documentation.html",
            tutorial_url: Some("https://isabelle.in.tum.de/doc/tutorial.pdf"),
            api_url: Some("https://isabelle.in.tum.de/doc/isar-ref.pdf"),
            examples_repo: Some("https://www.isa-afp.org/"),
        },
        BackendDocSource {
            backend: BackendId::Coq,
            name: "Coq",
            docs_url: "https://coq.inria.fr/documentation",
            tutorial_url: Some("https://softwarefoundations.cis.upenn.edu/"),
            api_url: Some("https://coq.inria.fr/refman/"),
            examples_repo: None,
        },
        BackendDocSource {
            backend: BackendId::Dafny,
            name: "Dafny",
            docs_url: "https://dafny.org/dafny/DafnyRef/DafnyRef",
            tutorial_url: Some("https://dafny.org/latest/OnlineTutorial/guide"),
            api_url: None,
            examples_repo: Some("https://github.com/dafny-lang/dafny/tree/master/Test"),
        },
        // Neural network verification
        BackendDocSource {
            backend: BackendId::Marabou,
            name: "Marabou",
            docs_url: "https://github.com/NeuralNetworkVerification/Marabou/wiki",
            tutorial_url: None,
            api_url: None,
            examples_repo: Some("https://github.com/NeuralNetworkVerification/Marabou"),
        },
        BackendDocSource {
            backend: BackendId::AlphaBetaCrown,
            name: "alpha-beta-CROWN",
            docs_url: "https://github.com/Verified-Intelligence/alpha-beta-CROWN",
            tutorial_url: None,
            api_url: None,
            examples_repo: Some("https://github.com/Verified-Intelligence/alpha-beta-CROWN"),
        },
        BackendDocSource {
            backend: BackendId::Eran,
            name: "ERAN",
            docs_url: "https://github.com/eth-sri/eran",
            tutorial_url: None,
            api_url: None,
            examples_repo: Some("https://github.com/eth-sri/eran"),
        },
        // Probabilistic verification
        BackendDocSource {
            backend: BackendId::Storm,
            name: "Storm",
            docs_url: "https://www.stormchecker.org/documentation.html",
            tutorial_url: Some("https://www.stormchecker.org/getting-started.html"),
            api_url: Some("https://www.stormchecker.org/api/"),
            examples_repo: Some("https://github.com/moves-rwth/storm"),
        },
        BackendDocSource {
            backend: BackendId::Prism,
            name: "PRISM",
            docs_url: "https://www.prismmodelchecker.org/manual/",
            tutorial_url: Some("https://www.prismmodelchecker.org/tutorial/"),
            api_url: None,
            examples_repo: Some(
                "https://github.com/prismmodelchecker/prism/tree/master/prism-examples",
            ),
        },
        // Security protocol verification
        BackendDocSource {
            backend: BackendId::Tamarin,
            name: "Tamarin",
            docs_url: "https://tamarin-prover.github.io/manual/",
            tutorial_url: Some("https://tamarin-prover.github.io/manual/book/"),
            api_url: None,
            examples_repo: Some(
                "https://github.com/tamarin-prover/tamarin-prover/tree/develop/examples",
            ),
        },
        BackendDocSource {
            backend: BackendId::ProVerif,
            name: "ProVerif",
            docs_url: "https://bblanche.gitlabpages.inria.fr/proverif/",
            tutorial_url: None,
            api_url: None,
            examples_repo: None,
        },
        BackendDocSource {
            backend: BackendId::Verifpal,
            name: "Verifpal",
            docs_url: "https://verifpal.com/",
            tutorial_url: None,
            api_url: None,
            examples_repo: Some("https://github.com/symbolicsoft/verifpal"),
        },
        // Rust verification
        BackendDocSource {
            backend: BackendId::Verus,
            name: "Verus",
            docs_url: "https://verus-lang.github.io/verus/guide/",
            tutorial_url: None,
            api_url: Some("https://verus-lang.github.io/verus/reference/"),
            examples_repo: Some(
                "https://github.com/verus-lang/verus/tree/main/source/rust_verify/example",
            ),
        },
        BackendDocSource {
            backend: BackendId::Creusot,
            name: "Creusot",
            docs_url: "https://creusot-rs.github.io/creusot/guide/",
            tutorial_url: None,
            api_url: None,
            examples_repo: Some("https://github.com/creusot-rs/creusot/tree/master/creusot/tests"),
        },
        BackendDocSource {
            backend: BackendId::Prusti,
            name: "Prusti",
            docs_url: "https://viperproject.github.io/prusti-dev/user-guide/",
            tutorial_url: None,
            api_url: None,
            examples_repo: Some(
                "https://github.com/viperproject/prusti-dev/tree/master/prusti-tests",
            ),
        },
        // SMT solvers
        BackendDocSource {
            backend: BackendId::Z3,
            name: "Z3",
            docs_url: "https://microsoft.github.io/z3guide/",
            tutorial_url: Some(
                "https://microsoft.github.io/z3guide/programming/Z3%20JavaScript%20Examples",
            ),
            api_url: Some("https://z3prover.github.io/api/html/"),
            examples_repo: Some("https://github.com/Z3Prover/z3/tree/master/examples"),
        },
        BackendDocSource {
            backend: BackendId::Cvc5,
            name: "CVC5",
            docs_url: "https://cvc5.github.io/docs/",
            tutorial_url: None,
            api_url: Some("https://cvc5.github.io/docs/api/python/pythonic/"),
            examples_repo: Some("https://github.com/cvc5/cvc5/tree/main/examples"),
        },
    ]
}
