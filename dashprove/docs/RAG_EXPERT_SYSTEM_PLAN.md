# DashProve RAG Expert System Architecture

## Overview

This document describes a Retrieval-Augmented Generation (RAG) expert system that provides deep knowledge of all 20 verification backends integrated into DashProve. The system enables intelligent backend selection, proof guidance, error explanation, and tactic suggestions.

## Current Backend Inventory (20 backends)

### Core Theorem Provers & Model Checkers (7)
| Backend | Type | Input Format | Documentation Source |
|---------|------|--------------|---------------------|
| **Lean 4** | Theorem Prover | `.lean` | https://lean-lang.org/lean4/doc/ |
| **TLA+** | Model Checker | `.tla` | https://lamport.azurewebsites.net/tla/tla.html |
| **Kani** | Rust Model Checker | Rust + attributes | https://model-checking.github.io/kani/ |
| **Alloy** | Relational Model Checker | `.als` | https://alloytools.org/documentation.html |
| **Isabelle** | Theorem Prover | `.thy` | https://isabelle.in.tum.de/documentation.html |
| **Coq** | Proof Assistant | `.v` | https://coq.inria.fr/documentation |
| **Dafny** | Program Verifier | `.dfy` | https://dafny.org/dafny/DafnyRef/DafnyRef |

### Neural Network Verification (3)
| Backend | Approach | Input Format | Documentation Source |
|---------|----------|--------------|---------------------|
| **Marabou** | SMT-based | ONNX + VNNLIB | https://github.com/NeuralNetworkVerification/Marabou |
| **alpha-beta-CROWN** | Bound Propagation | ONNX + VNNLIB | https://github.com/Verified-Intelligence/alpha-beta-CROWN |
| **ERAN** | Abstract Interpretation | Various | https://github.com/eth-sri/eran |

### Probabilistic Verification (2)
| Backend | Approach | Input Format | Documentation Source |
|---------|----------|--------------|---------------------|
| **Storm** | Probabilistic MC | PRISM/JANI | https://www.stormchecker.org/documentation.html |
| **PRISM** | Symbolic Probabilistic | `.pm` | https://www.prismmodelchecker.org/manual/ |

### Security Protocol Verification (3)
| Backend | Approach | Input Format | Documentation Source |
|---------|----------|--------------|---------------------|
| **Tamarin** | Multiset Rewriting | `.spthy` | https://tamarin-prover.github.io/manual/ |
| **ProVerif** | Applied Pi-Calculus | `.pv` | https://bblanche.gitlabpages.inria.fr/proverif/manual.pdf |
| **Verifpal** | Symbolic Analysis | `.vp` | https://verifpal.com/res/pdf/manual.pdf |

### Rust Verification (3)
| Backend | Approach | Input Format | Documentation Source |
|---------|----------|--------------|---------------------|
| **Verus** | SMT via Z3 | Rust + macros | https://verus-lang.github.io/verus/guide/ |
| **Creusot** | Deductive via Why3 | Rust + attributes | https://creusot-rs.github.io/creusot/guide/ |
| **Prusti** | Viper-based | Rust + specs | https://viperproject.github.io/prusti-dev/user-guide/ |

### SMT Solvers (2)
| Backend | Type | Input Format | Documentation Source |
|---------|------|--------------|---------------------|
| **Z3** | SMT Solver | SMT-LIB2 | https://microsoft.github.io/z3guide/ |
| **CVC5** | SMT Solver | SMT-LIB2 | https://cvc5.github.io/docs/cvc5-1.0.0/ |

---

## Phase 1: Documentation Collection

### 1.1 Documentation Sources Per Backend

```yaml
lean4:
  official:
    - url: https://lean-lang.org/lean4/doc/
      type: reference
    - url: https://lean-lang.org/theorem_proving_in_lean4/
      type: tutorial
    - url: https://leanprover-community.github.io/mathlib4_docs/
      type: library_api
  papers:
    - "The Lean 4 Theorem Prover and Programming Language" (CADE 2021)
  examples:
    - mathlib4 test files
    - Lean 4 sample projects

tlaplus:
  official:
    - url: https://lamport.azurewebsites.net/tla/book.html
      type: book (Specifying Systems)
    - url: https://lamport.azurewebsites.net/tla/summary-standalone.pdf
      type: quick_reference
    - url: https://learntla.com/
      type: tutorial
  papers:
    - "The TLA+ Toolbox" (F-IDE 2019)
  examples:
    - TLA+ examples repository

kani:
  official:
    - url: https://model-checking.github.io/kani/
      type: guide
    - url: https://model-checking.github.io/kani/reference/attributes.html
      type: reference
  papers:
    - "Kani: A Rust Verification Tool" (ICSE 2023)
  backend_docs:
    - CBMC documentation (underlying engine)

alloy:
  official:
    - url: https://alloytools.org/documentation.html
      type: reference
    - url: https://alloytools.org/tutorials/online/
      type: tutorial
  books:
    - "Software Abstractions" (Daniel Jackson)

isabelle:
  official:
    - url: https://isabelle.in.tum.de/doc/tutorial.pdf
      type: tutorial
    - url: https://isabelle.in.tum.de/doc/isar-ref.pdf
      type: reference
    - url: https://isabelle.in.tum.de/doc/system.pdf
      type: system_manual
  papers:
    - "Isabelle/HOL: A Proof Assistant for Higher-Order Logic"

coq:
  official:
    - url: https://coq.inria.fr/refman/
      type: reference_manual
    - url: https://softwarefoundations.cis.upenn.edu/
      type: tutorial
  papers:
    - "The Coq Proof Assistant Reference Manual"

dafny:
  official:
    - url: https://dafny.org/dafny/DafnyRef/DafnyRef
      type: reference
    - url: https://dafny.org/latest/OnlineTutorial/guide
      type: tutorial
  papers:
    - "Dafny: An Automatic Program Verifier" (LPAR 2010)

marabou:
  official:
    - url: https://github.com/NeuralNetworkVerification/Marabou/wiki
      type: wiki
  papers:
    - "The Marabou Framework for Verification and Analysis of DNNs" (CAV 2019)
  formats:
    - VNNLIB specification
    - ONNX model format

alphabetacrown:
  official:
    - url: https://github.com/Verified-Intelligence/alpha-beta-CROWN
      type: readme
  papers:
    - "Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers" (ICLR 2021)
  competition:
    - VNN-COMP benchmarks and results

eran:
  official:
    - url: https://github.com/eth-sri/eran
      type: readme
  papers:
    - "An Abstract Domain for Certifying Neural Networks" (POPL 2019)
  domains:
    - DeepZ, DeepPoly, RefineZono, RefinePoly

storm:
  official:
    - url: https://www.stormchecker.org/documentation.html
      type: documentation
    - url: https://www.stormchecker.org/getting-started.html
      type: tutorial
  formats:
    - PRISM language
    - JANI format
  papers:
    - "Storm: A Modern Probabilistic Model Checker" (TACAS 2017)

prism:
  official:
    - url: https://www.prismmodelchecker.org/manual/
      type: manual
    - url: https://www.prismmodelchecker.org/tutorial/
      type: tutorial
  papers:
    - "PRISM 4.0: Verification of Probabilistic Real-Time Systems" (CAV 2011)
  logics:
    - PCTL, CSL, LTL, reward properties

tamarin:
  official:
    - url: https://tamarin-prover.github.io/manual/
      type: manual
    - url: https://tamarin-prover.github.io/manual/book/
      type: book
  papers:
    - "The TAMARIN Prover for the Symbolic Analysis of Security Protocols" (CAV 2013)

proverif:
  official:
    - url: https://bblanche.gitlabpages.inria.fr/proverif/
      type: homepage
    - url: https://bblanche.gitlabpages.inria.fr/proverif/manual.pdf
      type: manual
  papers:
    - "ProVerif: Cryptographic Protocol Verifier in the Formal Model"

verifpal:
  official:
    - url: https://verifpal.com/
      type: homepage
    - url: https://verifpal.com/res/pdf/manual.pdf
      type: manual
  papers:
    - "Verifpal: Cryptographic Protocol Analysis for the Real World" (2020)

verus:
  official:
    - url: https://verus-lang.github.io/verus/guide/
      type: guide
    - url: https://verus-lang.github.io/verus/reference/
      type: reference
  papers:
    - "Verus: Verifying Rust Programs using Linear Ghost Types" (OOPSLA 2023)

creusot:
  official:
    - url: https://creusot-rs.github.io/creusot/guide/
      type: guide
  backend_docs:
    - Why3 documentation
  papers:
    - "Creusot: A Foundational Verifier for Rust" (2022)

prusti:
  official:
    - url: https://viperproject.github.io/prusti-dev/user-guide/
      type: guide
  backend_docs:
    - Viper documentation
  papers:
    - "Prusti: A Practical Verification Infrastructure for Rust" (2022)

z3:
  official:
    - url: https://microsoft.github.io/z3guide/
      type: guide
    - url: https://z3prover.github.io/api/html/
      type: api
  standard:
    - SMT-LIB 2.6 specification
  papers:
    - "Z3: An Efficient SMT Solver" (TACAS 2008)

cvc5:
  official:
    - url: https://cvc5.github.io/docs/
      type: documentation
  standard:
    - SMT-LIB 2.6 specification
  papers:
    - "cvc5: A Versatile and Industrial-Strength SMT Solver" (TACAS 2022)
```

### 1.2 Collection Script Structure

```rust
// crates/dashprove-knowledge/src/collector.rs

pub struct DocumentationCollector {
    http_client: reqwest::Client,
    rate_limiter: RateLimiter,
    output_dir: PathBuf,
}

impl DocumentationCollector {
    /// Collect documentation for a backend
    pub async fn collect_backend(&self, backend: BackendId) -> Result<CollectionResult>;

    /// Download and convert HTML to markdown
    pub async fn fetch_html_to_markdown(&self, url: &str) -> Result<String>;

    /// Download PDF and extract text
    pub async fn fetch_pdf_to_text(&self, url: &str) -> Result<String>;

    /// Clone and process example repositories
    pub async fn clone_examples(&self, repo: &str) -> Result<Vec<Example>>;
}
```

---

## Phase 2: Knowledge Base Construction

### 2.1 Directory Structure

```
knowledge-base/
├── raw/                          # Raw downloaded content
│   ├── lean4/
│   │   ├── reference.html
│   │   ├── tutorial.html
│   │   └── ...
│   └── ...
├── processed/                    # Cleaned markdown
│   ├── lean4/
│   │   ├── syntax.md
│   │   ├── tactics.md
│   │   ├── type_system.md
│   │   └── ...
│   └── ...
├── structured/                   # Machine-readable specs
│   ├── grammars/                 # Input format grammars
│   │   ├── lean4.pest
│   │   ├── tlaplus.pest
│   │   └── ...
│   ├── schemas/                  # Output format schemas
│   │   ├── lean4_output.json
│   │   └── ...
│   └── mappings/                 # USL -> backend mappings
│       ├── lean4_mapping.yaml
│       └── ...
├── examples/                     # Curated examples
│   ├── lean4/
│   │   ├── basic_proofs/
│   │   ├── advanced_tactics/
│   │   └── mathlib_patterns/
│   └── ...
├── embeddings/                   # Vector embeddings
│   ├── index.faiss              # FAISS index
│   └── metadata.json            # Chunk metadata
└── cross_cutting/               # Domain knowledge
    ├── verification_theory.md
    ├── proof_strategies.md
    ├── common_patterns.md
    └── error_taxonomy.md
```

### 2.2 Chunking Strategy

```rust
pub struct ChunkingConfig {
    /// Target chunk size in tokens
    target_size: usize,  // ~512 tokens
    /// Overlap between chunks
    overlap: usize,      // ~50 tokens
    /// Respect semantic boundaries
    semantic_chunking: bool,
}

pub struct DocumentChunk {
    /// Unique chunk ID
    id: String,
    /// Source file path
    source: PathBuf,
    /// Section hierarchy (e.g., ["Tactics", "Induction", "Basic"])
    section_path: Vec<String>,
    /// Chunk content
    content: String,
    /// Metadata
    metadata: ChunkMetadata,
}

pub struct ChunkMetadata {
    /// Associated backend
    backend: Option<BackendId>,
    /// Content type
    content_type: ContentType,
    /// Tags for filtering
    tags: Vec<String>,
    /// Code blocks in this chunk
    code_blocks: Vec<CodeBlock>,
}

pub enum ContentType {
    Syntax,           // Language syntax reference
    Tactics,          // Proof tactics
    Theory,           // Theoretical background
    Example,          // Code examples
    ErrorMessage,     // Error explanations
    Tutorial,         // Step-by-step guides
    ApiReference,     // API documentation
}
```

### 2.3 Embedding Model Selection

```rust
pub enum EmbeddingModel {
    /// OpenAI text-embedding-3-small (1536 dims, fast, cheap)
    OpenAISmall,
    /// OpenAI text-embedding-3-large (3072 dims, best quality)
    OpenAILarge,
    /// Local sentence-transformers (384 dims, free, private)
    SentenceTransformers,
    /// Cohere embed-english-v3 (1024 dims, good for code)
    CohereV3,
    /// Voyage code-2 (1536 dims, optimized for code)
    VoyageCode,
}

// Recommendation: Use VoyageCode or OpenAISmall for production
// Use SentenceTransformers for local development
```

---

## Phase 3: RAG Pipeline Architecture

### 3.1 Core Types

```rust
// crates/dashprove-rag/src/lib.rs

pub struct RagPipeline {
    /// Vector store for semantic search
    vector_store: VectorStore,
    /// Structured knowledge lookup
    structured_kb: StructuredKnowledge,
    /// Example database
    examples: ExampleStore,
    /// LLM client for reasoning
    llm: LlmClient,
    /// Reranker model (optional)
    reranker: Option<Reranker>,
}

pub struct Query {
    /// User's question or context
    text: String,
    /// Current backend (if known)
    backend: Option<BackendId>,
    /// Property being verified
    property: Option<Property>,
    /// Error being debugged
    error: Option<BackendError>,
    /// Maximum results per source
    top_k: usize,
}

pub struct RetrievalResult {
    /// Semantic search results
    semantic_chunks: Vec<ScoredChunk>,
    /// Structured knowledge matches
    structured_matches: Vec<StructuredMatch>,
    /// Similar examples
    examples: Vec<ScoredExample>,
    /// Combined relevance score
    overall_score: f32,
}

pub struct ScoredChunk {
    pub chunk: DocumentChunk,
    pub score: f32,
    pub rerank_score: Option<f32>,
}
```

### 3.2 Retrieval Strategy

```rust
impl RagPipeline {
    pub async fn retrieve(&self, query: &Query) -> RetrievalResult {
        // 1. Embed the query
        let query_embedding = self.embed_query(&query.text).await?;

        // 2. Filter by backend if specified
        let filter = query.backend.map(|b| Filter::backend(b));

        // 3. Semantic search
        let semantic_results = self.vector_store
            .search(&query_embedding, query.top_k * 3, filter)
            .await?;

        // 4. Structured lookup (exact matches for syntax, errors)
        let structured_results = self.structured_kb
            .lookup(&query.text, query.backend)
            .await?;

        // 5. Example search
        let example_results = self.examples
            .find_similar(&query.text, query.backend, query.top_k)
            .await?;

        // 6. Rerank if reranker available
        let semantic_results = if let Some(ref reranker) = self.reranker {
            reranker.rerank(&query.text, semantic_results).await?
        } else {
            semantic_results
        };

        // 7. Combine and deduplicate
        self.combine_results(semantic_results, structured_results, example_results)
    }
}
```

### 3.3 Augmented Prompt Construction

```rust
pub struct PromptBuilder {
    system_prompt: String,
    retrieved_context: Vec<String>,
    user_query: String,
    output_format: OutputFormat,
}

impl PromptBuilder {
    pub fn build_expert_prompt(&self, query: &Query, retrieval: &RetrievalResult) -> String {
        let mut prompt = String::new();

        // System context
        prompt.push_str(&format!(
            "You are a DashProve verification expert with deep knowledge of {}.\n\n",
            query.backend.map(|b| backend_name(b)).unwrap_or("all verification tools")
        ));

        // Retrieved documentation
        prompt.push_str("## Relevant Documentation\n\n");
        for chunk in &retrieval.semantic_chunks {
            prompt.push_str(&format!(
                "### From: {}\n{}\n\n",
                chunk.chunk.source.display(),
                chunk.chunk.content
            ));
        }

        // Structured knowledge
        if !retrieval.structured_matches.is_empty() {
            prompt.push_str("## Reference Information\n\n");
            for m in &retrieval.structured_matches {
                prompt.push_str(&format!("{}\n", m.to_markdown()));
            }
        }

        // Examples
        if !retrieval.examples.is_empty() {
            prompt.push_str("## Similar Examples\n\n");
            for ex in &retrieval.examples {
                prompt.push_str(&format!(
                    "```{}\n{}\n```\n\n",
                    ex.language,
                    ex.code
                ));
            }
        }

        // User query
        prompt.push_str(&format!("## Task\n\n{}\n", query.text));

        prompt
    }
}
```

---

## Phase 4: Expert Behaviors

### 4.1 Backend Selection Expert

```rust
pub struct BackendSelector {
    rag: RagPipeline,
    ml_predictor: StrategyModel,
}

impl BackendSelector {
    pub async fn select(&self, spec: &TypedSpec) -> BackendRecommendation {
        // 1. Extract property characteristics
        let features = PropertyFeatureVector::from_spec(spec);

        // 2. Get ML prediction
        let ml_prediction = self.ml_predictor.predict(&features);

        // 3. RAG query for similar verifications
        let query = Query {
            text: format!("Which backend is best for: {}", spec.summary()),
            property: Some(spec.primary_property()),
            ..Default::default()
        };
        let retrieval = self.rag.retrieve(&query).await?;

        // 4. Combine evidence
        let recommendation = self.combine_predictions(
            ml_prediction,
            retrieval,
            &features
        );

        recommendation
    }
}
```

### 4.2 Error Explanation Expert

```rust
pub struct ErrorExplainer {
    rag: RagPipeline,
}

impl ErrorExplainer {
    pub async fn explain(&self, error: &BackendError, context: &Context) -> Explanation {
        // 1. Parse error message
        let parsed = self.parse_error(error);

        // 2. Query for similar errors and solutions
        let query = Query {
            text: format!("Error: {}", error.message()),
            backend: Some(context.backend),
            error: Some(error.clone()),
            ..Default::default()
        };
        let retrieval = self.rag.retrieve(&query).await?;

        // 3. Generate explanation with LLM
        let prompt = self.build_explanation_prompt(error, context, &retrieval);
        let explanation = self.rag.llm.complete(&prompt).await?;

        Explanation {
            error_type: parsed.error_type,
            human_readable: explanation,
            suggested_fixes: self.extract_fixes(&explanation),
            related_docs: retrieval.semantic_chunks.iter()
                .map(|c| c.chunk.source.clone())
                .collect(),
        }
    }
}
```

### 4.3 Tactic Suggestion Expert

```rust
pub struct TacticSuggester {
    rag: RagPipeline,
    proof_db: ProofDatabase,
}

impl TacticSuggester {
    pub async fn suggest(&self, goal: &ProofGoal, context: &ProofContext) -> Vec<TacticSuggestion> {
        // 1. Encode the current proof state
        let state_description = self.describe_state(goal, context);

        // 2. Query for similar proof states
        let query = Query {
            text: state_description,
            backend: Some(context.backend),
            ..Default::default()
        };
        let retrieval = self.rag.retrieve(&query).await?;

        // 3. Look up successful tactics from proof database
        let historical = self.proof_db.find_similar_goals(goal, context.backend);

        // 4. Generate suggestions with LLM
        let prompt = self.build_tactic_prompt(goal, context, &retrieval, &historical);
        let suggestions = self.rag.llm.complete(&prompt).await?;

        self.parse_suggestions(&suggestions)
    }
}
```

### 4.4 Compilation Expert

```rust
pub struct CompilationExpert {
    rag: RagPipeline,
    compilers: HashMap<BackendId, Box<dyn BackendCompiler>>,
}

impl CompilationExpert {
    pub async fn compile_with_guidance(
        &self,
        spec: &TypedSpec,
        backend: BackendId,
    ) -> CompilationResult {
        // 1. Get backend-specific syntax documentation
        let syntax_query = Query {
            text: format!("Syntax for {} property in {}",
                spec.primary_property().kind(),
                backend_name(backend)
            ),
            backend: Some(backend),
            ..Default::default()
        };
        let syntax_docs = self.rag.retrieve(&syntax_query).await?;

        // 2. Find similar compilations
        let example_query = Query {
            text: format!("Example {} specification", backend_name(backend)),
            backend: Some(backend),
            property: Some(spec.primary_property()),
            ..Default::default()
        };
        let examples = self.rag.retrieve(&example_query).await?;

        // 3. Standard compilation
        let compiler = self.compilers.get(&backend).ok_or(Error::NoCompiler)?;
        let initial = compiler.compile(spec)?;

        // 4. Expert review and enhancement
        let prompt = self.build_review_prompt(&initial, &syntax_docs, &examples);
        let review = self.rag.llm.complete(&prompt).await?;

        // 5. Apply suggested improvements
        self.apply_improvements(initial, review)
    }
}
```

---

## Phase 5: Integration Points

### 5.1 CLI Integration

```rust
// crates/dashprove-cli/src/commands/expert.rs

#[derive(Subcommand)]
pub enum ExpertCommand {
    /// Ask the expert system a question
    Ask {
        #[arg(short, long)]
        backend: Option<BackendId>,
        question: String,
    },

    /// Explain an error
    Explain {
        #[arg(short, long)]
        backend: BackendId,
        error_file: PathBuf,
    },

    /// Get tactic suggestions
    Suggest {
        #[arg(short, long)]
        backend: BackendId,
        proof_state: PathBuf,
    },

    /// Recommend backend for spec
    Recommend {
        spec_file: PathBuf,
    },
}
```

### 5.2 LSP Integration

```rust
// crates/dashprove-lsp/src/expert.rs

impl LanguageServer for DashProveLsp {
    async fn code_action(&self, params: CodeActionParams) -> Vec<CodeAction> {
        // Get expert suggestions for errors
        if let Some(diagnostics) = params.context.diagnostics {
            for diag in diagnostics {
                let suggestions = self.expert.suggest_fixes(&diag).await?;
                // Convert to code actions
            }
        }
    }

    async fn completion(&self, params: CompletionParams) -> CompletionList {
        // Get expert suggestions for proof tactics
        if self.is_in_proof_context(&params) {
            let tactics = self.expert.suggest_tactics(&params).await?;
            // Convert to completions
        }
    }
}
```

### 5.3 API Integration

```rust
// crates/dashprove-server/src/routes/expert.rs

pub async fn expert_ask(
    State(state): State<AppState>,
    Json(request): Json<AskRequest>,
) -> Result<Json<AskResponse>, AppError> {
    let answer = state.expert.answer(&request.question, &request.context).await?;
    Ok(Json(AskResponse { answer }))
}

pub async fn expert_explain(
    State(state): State<AppState>,
    Json(request): Json<ExplainRequest>,
) -> Result<Json<ExplainResponse>, AppError> {
    let explanation = state.expert.explain_error(&request.error, &request.context).await?;
    Ok(Json(ExplainResponse { explanation }))
}
```

---

## Implementation Roadmap

### Phase 1: Documentation Collection (Est: 3-5 commits)
- [ ] Create dashprove-knowledge crate
- [ ] Implement documentation fetcher (HTML, PDF, Git)
- [ ] Download docs for all 20 backends
- [ ] Convert to markdown format
- [ ] Extract and organize examples

### Phase 2: Knowledge Base (Est: 3-5 commits)
- [ ] Implement chunking system
- [ ] Set up embedding infrastructure
- [ ] Create vector store (FAISS or similar)
- [ ] Build structured knowledge index
- [ ] Create example database

### Phase 3: RAG Pipeline (Est: 3-5 commits)
- [ ] Implement retrieval system
- [ ] Add query understanding
- [ ] Integrate LLM client (Claude API)
- [ ] Build prompt templates
- [ ] Add reranking

### Phase 4: Expert Behaviors (Est: 5-7 commits)
- [ ] Backend selection expert
- [ ] Error explanation expert
- [ ] Tactic suggestion expert
- [ ] Compilation guidance expert
- [ ] Question answering expert

### Phase 5: Integration (Est: 3-5 commits)
- [ ] CLI expert commands
- [ ] LSP code actions
- [ ] Server API endpoints
- [ ] Documentation and tests

**Total: 17-27 commits estimated**

---

## Worker Directives

### Immediate Next Steps

1. **Create `dashprove-knowledge` crate** - New crate for documentation collection and knowledge base management

2. **Implement documentation collectors** - For each backend, create a collector that:
   - Fetches official documentation
   - Converts HTML/PDF to markdown
   - Extracts code examples
   - Preserves section structure

3. **Start with high-priority backends first**:
   - Lean 4 (most active community)
   - Z3 (foundational SMT solver)
   - Verus (hot Rust verification tool)

4. **Output format**: Store collected documentation in `knowledge-base/processed/{backend}/` as markdown files

### Code Structure

```
crates/
└── dashprove-knowledge/
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── collector/
        │   ├── mod.rs
        │   ├── html.rs
        │   ├── pdf.rs
        │   └── git.rs
        ├── chunking/
        │   ├── mod.rs
        │   └── semantic.rs
        ├── embedding/
        │   ├── mod.rs
        │   └── models.rs
        ├── store/
        │   ├── mod.rs
        │   ├── vector.rs
        │   └── structured.rs
        └── expert/
            ├── mod.rs
            ├── backend_selector.rs
            ├── error_explainer.rs
            └── tactic_suggester.rs
```

### Dependencies to Add

```toml
[dependencies]
# HTTP client
reqwest = { version = "0.11", features = ["json", "stream"] }

# HTML parsing
scraper = "0.18"
html2md = "0.2"

# PDF extraction
pdf-extract = "0.7"

# Vector operations
ndarray = "0.15"
faiss = { version = "0.12", optional = true }  # Or use Qdrant client

# Embeddings (choose one)
tokenizers = "0.15"  # For local models
async-openai = "0.18"  # For OpenAI embeddings

# LLM client
anthropic = "0.1"  # Claude API

# Serialization
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
```
