# API Reference

DashProve exposes three primary interfaces: the Rust crate (`dashprove`), the CLI (`dashprove` / `dashprove-server` binaries), and the USL tooling (`dashprove-usl`). This document summarizes the callable surfaces and configuration knobs.

## Rust Crate (`dashprove`)
- Entry point: `DashProve` in `crates/dashprove/src/client.rs`.
- Construction:
  - `DashProve::new(DashProveConfig)` or `DashProve::default_client()`.
  - `DashProveConfig` fields: `backends: Vec<BackendId>`, `dispatcher: DispatcherConfig`, `learning_enabled: bool`, `api_url: Option<String>`.
  - Helpers: `DashProveConfig::all_backends()`, `with_backend(BackendId)`, `with_learning()`, `remote(url)`.
- Operations:
  - `verify(&mut self, spec_source: &str)` → parse, type-check, dispatch.
  - `verify_typed(&mut self, typed: &TypedSpec)` → dispatch only.
  - `verify_with_backend(&mut self, spec_source: &str, backend: BackendId)` → force a backend.
  - `verify_code(&mut self, code: &str, spec_source: &str)` → verify Rust code against USL contracts using Kani. Creates a self-contained project with the code and generated proof harnesses, runs `cargo kani`, and returns structured results.
  - `backends()` → registered backends, `check_health()` → `(BackendId, healthy)` pairs.
- Results: `VerificationResult` holds `status`, per-property `PropertyResult` (name, status, backends_used, proof/counterexample), `suggestions`, and `confidence`.
- Runtime monitors: `RuntimeMonitor::from_spec(&TypedSpec, &MonitorConfig)` produces Rust/TypeScript/Python monitor code; see `crates/dashprove/src/monitor.rs`.

## USL Helpers (`dashprove-usl`)
- Parse: `parse(&str) -> Spec` with grammar in `crates/dashprove-usl/src/usl.pest`.
- Type-check: `typecheck(Spec) -> TypedSpec` enforcing bool bodies, iterable bindings, and numeric/boolean operator rules.
- Compile: `compile_to_lean`, `compile_to_tlaplus`, `compile_to_kani`, `compile_to_alloy` for backend-specific code.
- Suggest tactics: `suggest_tactics_for_property(&Property)` returns compiler hints surfaced by the CLI `--suggest` flag.

## CLI (`dashprove-cli`)
Binary: `cargo run -p dashprove-cli -- <command>`

- `verify <path>`: run verification.
  - Flags: `--backends lean,tla+,kani,alloy`, `--timeout <secs>` (default 120), `--skip-health-check`, `--learn`, `--data-dir <dir>`, `--suggest`, `--incremental`, `--since <git-ref>` (requires `--incremental`).
  - ML flags: `--ml` (enable ML-based backend selection), `--ml-model <path>` (path to strategy model or ensemble), `--ml-confidence <0.0-1.0>` (minimum confidence threshold, default 0.5).
- `verify-code [code-file] --spec <spec-file>`: verify Rust code against USL contracts using Kani.
  - `--spec <path>`: path to USL specification file containing contracts (required).
  - `--timeout <secs>`: verification timeout (default: 300).
  - `-v, --verbose`: show verbose output.
  - If no code file is specified, reads Rust code from stdin.
- `export <path> --target <lean|tla+|kani|alloy> [-o FILE]`: emit compiled backend code.
- `backends`: list registered backends and health status.
- `explain <counterexample.json> [--backend <name>]`: human-readable explanation for a stored counterexample.
- `corpus <subcommand>`: proof corpus operations (when learning data is available).
  - `stats`: show corpus statistics.
  - `search <usl-file>`: search for similar proofs.
  - `cx-search <json-file>`: search for similar counterexamples.
  - `cx-add <json-file>`: add a counterexample to the corpus.
  - `cx-classify <json-file>`: classify a counterexample against stored clusters.
  - `cx-record-clusters <json-file>`: record cluster patterns from clustering results.
  - `history`: show corpus history over time.
    - `--corpus <proofs|counterexamples>`: corpus to visualize (default: counterexamples).
    - `--period <day|week|month>`: time period granularity (default: day).
    - `--format <text|json|html>`: output format (default: text).
    - `--from <YYYY-MM-DD>`, `--to <YYYY-MM-DD>`: filter by date range.
    - `-o <file>`: output file (stdout if not specified).
  - `compare`: compare two time periods in the corpus.
    - `--baseline-from/--baseline-to <YYYY-MM-DD>`: baseline period dates (required).
    - `--compare-from/--compare-to <YYYY-MM-DD>`: comparison period dates (required).
    - `--format <text|json|html>`: output format (default: text).
    - Outputs count deltas, percent changes, backend-level changes, growth rate projections, and period suggestions (HTML includes suggestion cards with ready-to-run CLI snippets).
  - `suggest-compare`: suggest comparison periods based on available data.
    - `--corpus <proofs|counterexamples>`: corpus to analyze (default: counterexamples).
    - `--format <text|json>`: output format (default: text).
    - Generates week-over-week, month-over-month, rolling period, and other comparison suggestions with CLI command snippets.
- `search "<text query>" [-n <limit>] [--data-dir <dir>]`: text-based proof search.
- `prove <path> [--hints]`: interactive proof mode stub (prints hints when available).
- `topics [topic]`: display help on DashProve concepts.
  - Available topics: `usl`, `backends`, `counterexamples`, `learning`, `properties`.
  - With no topic, lists all available help topics.

### Runtime Monitor Generation

- `monitor <path>`: Generate runtime monitor code from USL specifications.
  - `--target <rust|typescript|python>`: target language (default: rust).
  - `-o, --output <file>`: output file (prints to stdout if not specified).
  - `--assertions`: generate assertions that panic on property violation.
  - `--logging`: generate logging calls for property checks.
  - `--metrics`: generate metrics/counters for property checks.

**Example:**
```bash
# Generate Rust monitor with assertions
dashprove monitor spec.usl --target rust --assertions -o monitor.rs

# Generate TypeScript monitor with logging
dashprove monitor spec.usl --target typescript --logging -o monitor.ts

# Generate Python monitor with all features
dashprove monitor spec.usl --target python --assertions --logging --metrics
```

### Counterexample Analysis & Visualization

- `visualize <path>`: Export counterexample trace to visualization formats.
  - `--format <mermaid|dot|html>`: output format (default: html).
  - `-o, --output <file>`: output file (prints to stdout if not specified).
  - `--title <text>`: title for HTML output.

- `analyze <path> <action>`: Analyze counterexample traces for patterns and insights.
  - Actions:
    - `suggest [--format <text|json>]`: find patterns and suggest fixes.
    - `compress [--format <text|json>] [-o FILE]`: compress trace by removing redundant states.
    - `interleavings [--format <text|json>] [-o FILE]`: extract actor interleavings from concurrent traces.
    - `minimize [--max-states N] [-o FILE]`: minimize trace while preserving violation.
    - `abstract [--min-group-size N] [--format <text|json>] [-o FILE]`: create high-level trace summary.
    - `diff <other.json> [--format <text|json|html>] [-o FILE]`: compare two counterexamples.

- `cluster <paths...>`: Cluster multiple counterexamples to identify common failure patterns.
  - `--threshold <0.0-1.0>`: similarity threshold for clustering (default: 0.7).
  - `--format <text|json|html>`: output format (default: text).
  - `-o, --output <file>`: output file.
  - `--title <text>`: title for HTML output.

**Example:**
```bash
# Visualize counterexample as HTML
dashprove visualize counterexample.json --format html -o trace.html

# Generate DOT graph for GraphViz
dashprove visualize counterexample.json --format dot -o trace.dot
dot -Tpng trace.dot -o trace.png

# Analyze trace for patterns
dashprove analyze counterexample.json suggest

# Compress and minimize trace
dashprove analyze counterexample.json compress -o compressed.json
dashprove analyze counterexample.json minimize --max-states 10 -o minimal.json

# Compare two counterexamples
dashprove analyze cx1.json diff cx2.json --format html -o diff.html

# Cluster counterexamples
dashprove cluster cx1.json cx2.json cx3.json --threshold 0.8 --format html
```

### ML Training Commands

- `train`: Train an ML model for verification strategy prediction from the proof corpus.
  - `--data-dir <dir>`: learning data directory (default: `~/.dashprove`).
  - `-o, --output <path>`: output path for trained model (default: `<data-dir>/strategy_model.json`).
  - `--learning-rate <rate>`: learning rate 0.001-0.5 (default: 0.01).
  - `--epochs <n>`: number of training epochs (default: 20).
  - `--early-stopping`: enable early stopping with `--patience` and `--min-delta`.
  - `--validation-split <frac>`: validation fraction 0.1-0.5 (default: 0.2).
  - `--lr-scheduler <type>`: learning rate scheduler (constant, step, exp, cosine, plateau, warmup).
  - `--checkpoint`: enable model checkpointing with `--checkpoint-dir`, `--checkpoint-interval`, `--keep-best`.
  - `--resume <path>`: resume training from a checkpoint file.
  - `-v, --verbose`: show training progress.

- `tune`: Automatically tune hyperparameters for ML model training.
  - `--method <grid|random|bayesian>`: search method (default: bayesian).
  - `--iterations <n>`: iterations for random/bayesian search (default: 25).
  - `--lr-values <csv>`: learning rates for grid search.
  - `--lr-min/--lr-max <rate>`: learning rate range for random/bayesian.
  - `--epochs-min/--epochs-max <n>`: epoch range for random/bayesian.
  - `--initial-samples <n>`: GP warm-up samples for bayesian (default: 5).
  - `--kappa <k>`: exploration-exploitation trade-off for bayesian (default: 2.576).
  - `--cv-folds <n>`: k-fold cross-validation (0 = simple validation split).
  - `-o, --output <path>`: output path for tuned model.
  - `-v, --verbose`: show search progress.

- `ensemble`: Combine multiple trained models into an ensemble.
  - `--models <paths>`: comma-separated model file paths (required).
  - `--weights <csv>`: optional comma-separated weights matching model order.
  - `--method <soft|weighted>`: aggregation method (default: soft).
  - `-o, --output <path>`: output path for ensemble model.
  - `--data-dir <dir>`: learning data directory.
  - `--verbose`: show verbose output.

**ML Workflow Example:**
```bash
# 1. Build corpus by running verifications with --learn
dashprove verify specs/*.usl --learn

# 2. Train initial model
dashprove train --epochs 50 --early-stopping

# 3. Tune hyperparameters with Bayesian optimization
dashprove tune --method bayesian --iterations 25 --cv-folds 5 -o tuned_model.json

# 4. Train multiple specialized models
dashprove train --epochs 100 -o model_a.json
dashprove train --learning-rate 0.05 --epochs 100 -o model_b.json

# 5. Combine into ensemble
dashprove ensemble --models model_a.json,model_b.json --weights 0.6,0.4 -o ensemble.json

# 6. Use ML-guided verification
dashprove verify spec.usl --ml --ml-model ensemble.json
```

## REST API (`dashprove-server`)
Binary: `cargo run -p dashprove-server -- [FLAGS]`

- Flags/env:
  - `--port` / `DASHPROVE_PORT` (default 3000), `--host` / `DASHPROVE_HOST`.
  - `--require-auth` to enforce API keys; keys via `--api-key KEY:Name[:Rate]`, `DASHPROVE_API_KEYS`, admin keys via `--admin-key`/`DASHPROVE_ADMIN_KEYS`.
  - `--anonymous-rate-limit` (per minute, default 10), `--data-dir` for loading learning corpus, `--keys-file` for persistence.
- HTTP endpoints (see `crates/dashprove-server/src/routes.rs`):
  - `GET /health` (implicit via middleware), `GET /version`.
  - `POST /verify` `{ spec, backend? }` → parses and compiles to all registered backends, returning code snippets and errors.
  - `POST /verify/incremental` `{ base_spec, current_spec, changes, backend? }` → incremental verification comparing two specs. Uses dependency graph analysis to identify affected properties and caches unchanged properties. Returns `{ valid, cached_count, verified_count, affected_properties, unchanged_properties, compilations, errors }`. Changes array contains `{ kind, target, details? }` where kind is one of: `file_modified`, `file_added`, `file_deleted`, `function_modified`, `type_added`, `type_modified`, `dependency_changed`.
  - `GET /corpus/search?query=...&k=10` → similar proofs if corpus loaded.
  - `GET /corpus/stats` → corpus statistics including proof counts, counterexample counts, cluster patterns, and tactic statistics with top tactics by Wilson score.
  - `GET /corpus/history?corpus=<proofs|counterexamples>&period=<day|week|month>&from=YYYY-MM-DD&to=YYYY-MM-DD&format=<json|html>` → corpus history over time with period stats and cumulative counts. Optional `format=html` returns interactive Chart.js visualization (default: JSON).
  - `GET /corpus/compare?corpus=<proofs|counterexamples>&baseline_from=YYYY-MM-DD&baseline_to=YYYY-MM-DD&compare_from=YYYY-MM-DD&compare_to=YYYY-MM-DD&format=<json|html>` → compare two time periods with count deltas, percent changes, backend-level changes, and growth rate projections. Optional `format=html` returns interactive Chart.js visualization with suggestions (default: JSON).
  - `GET /corpus/suggest?corpus=<proofs|counterexamples>` → suggest meaningful comparison periods (week-over-week, month-over-month, rolling periods, etc.) with API query strings for easy follow-up.
  - `GET /corpus/counterexamples?limit=50&offset=0&backend=<backend>&property_name=<substring>&from=YYYY-MM-DD&to=YYYY-MM-DD` → list counterexamples with pagination and optional filters for backend, property name substring, and date range. Returns counterexample entries, total count, and pagination info.
  - `GET /corpus/counterexamples/:id` → get a single counterexample by ID with full details (witness, failed checks, trace, property name, backend, timestamps).
  - `POST /corpus/counterexamples/search` `{ counterexample: {...}, k: 5 }` → search for similar counterexamples based on feature similarity (witness variables, trace patterns, failed checks).
  - `GET /corpus/counterexamples/text-search?query=...&k=10` → search counterexamples by text keywords; matches witness variable names, trace variables, failed check descriptions, and action names.
  - `POST /corpus/counterexamples` `{ counterexample: {...}, property_name, backend, cluster_label? }` → add a counterexample to the corpus for future similarity searches.
  - `POST /corpus/counterexamples/classify` `{ counterexample: {...} }` → classify a counterexample against stored cluster patterns, returns best matching cluster label and similarity score.
  - `POST /corpus/counterexamples/clusters` `{ patterns: [...], similarity_threshold }` → record cluster patterns from clustering results for future classification.
  - `POST /tactics/suggest` `{ goal }` and `POST /sketch/elaborate` `{ property, sketch }` via `dashprove-ai`.
  - `POST /explain` `{ counterexample, backend? }` → natural-language explanation.
  - `GET /backends` → backend IDs and health.
- WebSocket: `GET /ws/verify` for streaming verification progress (`accepted`, `progress`, `backend_started`, `backend_completed`, `completed`, `error` events).
  - Query param `correlation_id` (max 128 chars): optional client-provided ID for distributed tracing; echoed in `Connected` message.
- Authentication: `X-API-Key` or `Authorization: Bearer <key>`; anonymous requests honor rate limits when auth is disabled.

### Observability Endpoints

- `GET /metrics` → Prometheus-compatible metrics in text format for scraping.
- `GET /health` → server health status including drain state for graceful shutdown.

### Prometheus Metrics

The `/metrics` endpoint exposes the following metrics:

**HTTP Request Metrics**
- `dashprove_http_requests_total{method,path}` - Counter of HTTP requests
- `dashprove_http_request_duration_seconds{method,path}` - Histogram of request durations
- `dashprove_http_errors_total{method,path,status}` - Counter of HTTP errors (4xx/5xx)
- `dashprove_active_http_requests` - Gauge of in-flight HTTP requests

**WebSocket Metrics**
- `dashprove_active_websocket_sessions` - Gauge of active WebSocket sessions

**Verification Metrics**
- `dashprove_verifications_total` - Counter of verification requests
- `dashprove_verifications_success` - Counter of successful verifications
- `dashprove_verifications_failed` - Counter of failed verifications
- `dashprove_verification_duration_seconds` - Histogram of overall verification durations

**Per-Backend Metrics**
- `dashprove_backend_verification_duration_seconds{backend}` - Histogram of verification durations by backend (lean4, tlaplus, kani, alloy, etc.)
- `dashprove_backend_verifications_success{backend}` - Counter of successful verifications per backend
- `dashprove_backend_verifications_failed{backend}` - Counter of failed verifications per backend

**Cache Metrics**
- `dashprove_cache_entries{state}` - Gauge of cache entries (state: total, valid, expired)

**Server Health**
- `dashprove_uptime_seconds` - Gauge of server uptime

### Request Tracing

All requests include `X-Request-ID` header support for distributed tracing:
- Client-provided `X-Request-ID` headers are propagated through and returned in responses
- If not provided, a UUIDv4 is generated and returned
- Request IDs are validated (non-empty, max 128 characters)

## Language Server Protocol (`dashprove-lsp`)

The LSP server provides full IDE integration for USL files. Binary: `cargo run -p dashprove-lsp`

**Core Features:**

- **Document Synchronization**: Full text sync on open/change/close
- **Diagnostics**: Real-time syntax and type error reporting
- **Hover**: Documentation for keywords, types, and user definitions
- **Go to Definition**: Navigate to type definitions
- **Find References**: Find all usages of types and properties
- **Completion**: Auto-complete keywords, builtin types, and user-defined types
- **Signature Help**: Parameter hints for contracts (triggers: `(`, `,`)
- **Document Symbols**: Outline of types and properties for IDE symbol panels
- **Workspace Symbols**: Search types and properties across all files
- **Rename**: Rename types and properties with validation

**Advanced Features:**

- **Semantic Tokens**: Context-aware syntax highlighting (types, properties, keywords, quantifiers, operators)
- **Code Actions**: Quick fixes and refactoring suggestions (QuickFix, Refactor.Extract)
- **Code Lenses**: Inline verification actions (Verify, Show Backend Info)
- **Document Formatting**: Format entire document or selection
- **Folding Ranges**: Collapse type definitions, properties, and blocks
- **Inlay Hints**: Inline type annotations
- **Selection Range**: Smart selection expansion based on syntax
- **Document Highlight**: Highlight all occurrences of symbol under cursor
- **Call Hierarchy**: Navigate incoming/outgoing property references
- **Linked Editing**: Simultaneously edit related identifiers
- **Moniker**: Stable symbol identifiers for cross-workspace navigation

**Commands:**

- `dashprove.verify`: Verify the current file
- `dashprove.showBackendInfo`: Display verification backend information

**Editor Integration:**

For **VS Code**, install the DashProve extension from `editors/vscode`:
```bash
cd editors/vscode
npm install
npm run compile
# Then install via VS Code "Install from VSIX" or symlink to ~/.vscode/extensions
```

Configuration:
```json
{
  "dashprove.server.path": "path/to/dashprove-lsp",
  "dashprove.verification.autoVerify": true,
  "dashprove.inlayHints.enabled": true,
  "dashprove.codeLens.enabled": true
}
```

For **Neovim** with `nvim-lspconfig`:
```lua
require('lspconfig').dashprove_lsp.setup{
  cmd = { "dashprove-lsp" },
  filetypes = { "usl" },
}
```

For **Emacs** with `lsp-mode`:
```elisp
(lsp-register-client
  (make-lsp-client
    :new-connection (lsp-stdio-connection '("dashprove-lsp"))
    :major-modes '(usl-mode)
    :server-id 'dashprove-lsp))

**Programmatic Usage:**

```rust
use dashprove_lsp::run_server;

#[tokio::main]
async fn main() {
    run_server().await.unwrap();
}
```

**Document Store API:**

```rust
use dashprove_lsp::{Document, DocumentStore};
use tower_lsp::lsp_types::Url;

let store = DocumentStore::new();
let uri = Url::parse("file:///test.usl").unwrap();

// Open document
store.open(uri.clone(), 1, "theorem foo { true }".to_string());

// Access document
store.with_document(&uri, |doc| {
    // doc.spec - parsed AST
    // doc.typed_spec - type-checked spec
    // doc.parse_error - parse error if any
    // doc.type_errors - type errors if any
});

// Update document
store.update(&uri, 2, "theorem bar { false }".to_string());

// Close document
store.close(&uri);
```

## Learning System (`dashprove-learning`)
- Loaded when `--learn` (CLI) or `--data-dir` (server) is provided.
- Provides `ProofLearningSystem` (corpus, embeddings, tactic statistics) used for `corpus search` and tactic suggestions.
- Stored under `~/.dashprove` by default; corpus size reported in server logs during startup.
- History and comparison features:
  - `HistoryReport`: aggregated counts over time periods, available for proofs and counterexamples.
  - `HistoryComparison`: compares two time periods with deltas, growth rates, and projections.
  - `PeriodSuggestion`: auto-suggests meaningful comparison periods (week-over-week, month-over-month, etc.).
  - HTML output includes interactive Chart.js visualizations.

### Vector Embeddings

The learning system provides vector embeddings for semantic similarity search:

```rust
use dashprove_learning::{Embedding, PropertyEmbedder, ProofCorpus, EmbeddingIndex};

// Create embedder and compute embeddings for corpus
let mut embedder = PropertyEmbedder::new();
let mut corpus = ProofCorpus::new();

// Insert proofs (embeddings are optional)
corpus.insert(&learnable_result);

// Compute embeddings for all proofs that don't have one
let count = corpus.compute_embeddings(&mut embedder);

// Query using embeddings
let query_emb = embedder.embed_query(&query_property);
let similar = corpus.find_similar_embedding(&query_emb, k);
```

**Key types:**
- `Embedding`: Dense vector (96 dimensions) for similarity computation
- `PropertyEmbedder`: Converts properties to embeddings using structural+keyword features
- `EmbeddingIndex`: Standalone index for nearest-neighbor search
- `EmbeddingIndexBuilder`: Builder pattern for creating embedding indices

**Similarity methods:**
- `Embedding::cosine_similarity(&other)`: Returns [-1, 1] cosine similarity
- `Embedding::normalized_similarity(&other)`: Returns [0, 1] normalized similarity
- `Embedding::l2_distance(&other)`: Returns Euclidean distance

**Corpus methods:**
- `corpus.compute_embeddings(&mut embedder)`: Populate embeddings for stored proofs
- `corpus.find_similar_embedding(&query, k)`: Vector similarity search
- `corpus.embedding_count()`: Count proofs with embeddings
- `corpus.insert_with_embedding(result, embedding)`: Insert with pre-computed embedding

## Strategy Prediction (`dashprove-ai`)

The AI module provides ML-based strategy prediction for selecting optimal verification backends and tactics.

### Core Types

- `StrategyPredictor`: Neural network model for predicting backend success likelihood, tactics, and verification time.
- `StrategyModel`: Wrapper that supports both single models and ensembles; automatically detected when loading.
- `EnsembleStrategyPredictor`: Combines multiple models with configurable aggregation (soft voting or weighted).
- `StrategyPrediction`: Prediction result containing backend scores, tactic suggestions, and estimated time.

### Single Model Usage

```rust
use dashprove_ai::{StrategyPredictor, PropertyFeatureVector};

// Load trained model
let predictor = StrategyPredictor::load("strategy_model.json")?;

// Extract features from a property
let features = PropertyFeatureVector::from_property(&property);

// Get prediction
let prediction = predictor.predict(&features);
println!("Backend scores: {:?}", prediction.backend_probabilities);
println!("Suggested tactics: {:?}", prediction.tactics);
println!("Estimated time: {:?}", prediction.estimated_time_secs);
```

### Ensemble Usage

```rust
use dashprove_ai::{StrategyModel, EnsembleAggregation};

// Load model (works for both single and ensemble)
let model = StrategyModel::load("model.json")?;

// Predict using the model
let prediction = model.predict(&features);

// Build ensemble programmatically
use dashprove_ai::EnsembleStrategyPredictor;
let ensemble = EnsembleStrategyPredictor::new(vec![
    (predictor1, 0.6),
    (predictor2, 0.4),
], EnsembleAggregation::Weighted);
let prediction = ensemble.predict(&features);
```

### Training

```rust
use dashprove_ai::{StrategyPredictor, TrainingDataGenerator, EarlyStoppingConfig};

// Generate training data from corpus
let generator = TrainingDataGenerator::new(&corpus);
let (train_data, val_data) = generator.split(0.8);

// Train with early stopping
let config = EarlyStoppingConfig {
    patience: 5,
    min_delta: 0.001,
    restore_best_weights: true,
};
let (predictor, history) = StrategyPredictor::train_with_early_stopping(
    &train_data,
    &val_data,
    50,   // max epochs
    0.01, // learning rate
    &config,
)?;

// Save trained model
predictor.save("strategy_model.json")?;
```

### Hyperparameter Tuning

```rust
use dashprove_ai::{BayesianOptimizer, GridSearchSpace, RandomSearchConfig};

// Bayesian optimization (most efficient)
let optimizer = BayesianOptimizer::new(5, 2.576); // initial_samples, kappa
let result = optimizer.optimize(&data, 25)?; // 25 iterations

// Grid search
let space = GridSearchSpace {
    learning_rates: vec![0.001, 0.01, 0.05],
    epochs: vec![50, 100],
};
let result = space.search(&data)?;

// Random search
let config = RandomSearchConfig {
    lr_range: (0.001, 0.1),
    epochs_range: (20, 200),
    iterations: 20,
    seed: 42,
};
let result = config.search(&data)?;
```

### Cross-Validation

```rust
use dashprove_ai::StrategyPredictor;

// K-fold cross-validation
let cv_result = StrategyPredictor::cross_validate(&data, 5, 50, 0.01)?;
println!("Mean loss: {}, Std: {}", cv_result.mean_loss, cv_result.std_loss);
```

### Model Checkpointing

```rust
use dashprove_ai::{StrategyPredictor, CheckpointConfig};

let config = CheckpointConfig {
    checkpoint_dir: "checkpoints".into(),
    save_interval: 10,  // every 10 epochs
    keep_best: 3,       // keep top 3 by validation loss
};

let result = StrategyPredictor::train_with_checkpointing(
    &train_data, &val_data, 100, 0.01, &config,
)?;

// Resume from checkpoint
let (predictor, history) = StrategyPredictor::resume_from_checkpoint(
    "checkpoints/best.json", &train_data, &val_data, 50, 0.01,
)?;
```
