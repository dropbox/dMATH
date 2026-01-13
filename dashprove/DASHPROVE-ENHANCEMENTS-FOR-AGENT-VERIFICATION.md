# DashProve Enhancement Requirements for Agent System Verification

**Purpose:** Define new DashProve capabilities needed to verify AI agent systems like claude_code_rs
**Date:** December 2025
**Status:** Requirements Draft
**Priority:** P0 (Blocking for claude_code_rs verification)

---

## Executive Summary

While DashProve has 20 verification backends and comprehensive USL support, verifying AI agent systems like claude_code_rs requires capabilities not yet implemented:

1. **Behavioral Equivalence Testing** - Compare two implementations
2. **Trace-Based Verification** - Verify execution traces against TLA+ specs
3. **Runtime Monitor Synthesis** - Generate monitors from specs
4. **Async/Concurrent Verification** - Verify tokio async state machines
5. **Fuzzy/Semantic Properties** - Handle LLM non-determinism
6. **Stateful Property Testing** - Model-based test generation
7. **CI/CD Integration** - Incremental verification in pipelines

---

## Gap Analysis: DashProve vs. claude_code_rs Needs

| Capability | DashProve Status | claude_code_rs Need | Gap |
|------------|------------------|---------------------|-----|
| TLA+ backend | DONE | Agent loop spec | None |
| Kani backend | DONE | Memory safety proofs | None |
| USL contracts | DONE | Tool contracts | None |
| Bisimulation checking | NOT IMPLEMENTED | Parity verification | HIGH |
| Differential testing | NOT IMPLEMENTED | Compare vs Claude Code | HIGH |
| Trace verification | NOT IMPLEMENTED | Verify execution traces | MEDIUM |
| Runtime monitors | MENTIONED, NOT DONE | Runtime invariant checking | MEDIUM |
| Async verification | NOT IMPLEMENTED | tokio state machines | HIGH |
| Semantic similarity | NOT IMPLEMENTED | LLM output comparison | MEDIUM |
| Model-based testing | NOT IMPLEMENTED | State machine exploration | MEDIUM |
| CI/CD integration | NOT IMPLEMENTED | PR verification | LOW |
| MIRI integration | NOT IMPLEMENTED | UB detection | LOW |

---

## Enhancement 1: Behavioral Equivalence / Bisimulation Checking

### Problem

claude_code_rs must behave **exactly** like Claude Code. We need to verify that given the same inputs, both produce equivalent outputs, tool calls, and API requests.

### Proposed Solution

Add a new backend or module: `dashprove-bisim`

```rust
// crates/dashprove-bisim/src/lib.rs

/// Configuration for bisimulation checking
pub struct BisimulationConfig {
    /// Reference implementation (oracle)
    pub oracle: OracleConfig,
    /// Implementation under test
    pub test_subject: TestSubjectConfig,
    /// What aspects to compare
    pub equivalence_criteria: EquivalenceCriteria,
    /// How to handle non-determinism
    pub nondeterminism_strategy: NondeterminismStrategy,
}

/// Oracle configuration (e.g., Claude Code binary)
pub enum OracleConfig {
    /// Run a binary and capture I/O
    Binary { path: PathBuf, args: Vec<String> },
    /// Use recorded traces
    RecordedTraces { trace_dir: PathBuf },
    /// HTTP API endpoint
    HttpEndpoint { url: String },
}

/// What aspects must be equivalent
pub struct EquivalenceCriteria {
    /// API request bodies must match
    pub api_requests: bool,
    /// Tool calls must match (name, arguments)
    pub tool_calls: bool,
    /// Final output must match
    pub output: bool,
    /// Timing bounds (within X% of oracle)
    pub timing_tolerance: Option<f64>,
    /// Allow semantic equivalence vs byte-exact
    pub semantic_comparison: bool,
}

/// Handle non-determinism in LLM responses
pub enum NondeterminismStrategy {
    /// Require exact match (for deterministic components)
    ExactMatch,
    /// Allow any response that satisfies spec
    SpecSatisfaction,
    /// Use semantic similarity threshold
    SemanticSimilarity { threshold: f64 },
    /// Record multiple runs, check distribution
    DistributionMatch { samples: usize },
}

/// Result of bisimulation check
pub struct BisimulationResult {
    pub equivalent: bool,
    pub differences: Vec<Difference>,
    pub traces: Option<(Trace, Trace)>,
    pub confidence: f64,
}

pub enum Difference {
    ApiRequestMismatch {
        index: usize,
        oracle: ApiRequest,
        subject: ApiRequest,
        diff: JsonDiff,
    },
    ToolCallMismatch {
        index: usize,
        oracle: ToolCall,
        subject: ToolCall,
    },
    OutputMismatch {
        oracle: String,
        subject: String,
        semantic_similarity: f64,
    },
    TimingViolation {
        oracle_ms: u64,
        subject_ms: u64,
        tolerance: f64,
    },
}

#[async_trait]
pub trait BisimulationChecker: Send + Sync {
    /// Run a single comparison
    async fn check(&self, input: &TestInput) -> Result<BisimulationResult>;

    /// Run batch comparison
    async fn check_batch(&self, inputs: &[TestInput]) -> Result<Vec<BisimulationResult>>;

    /// Generate inputs that maximize coverage
    async fn generate_inputs(&self, count: usize) -> Vec<TestInput>;
}
```

### USL Extension

```usl
// New USL construct for bisimulation
bisimulation agent_parity {
    oracle: "claude-code"
    subject: "claude-code-rs"

    equivalent on {
        api_request_format,
        tool_call_sequence,
        final_output
    }

    tolerance {
        timing: 10%,
        semantic_similarity: 0.95
    }

    forall input: UserMessage .
        let oracle_trace = execute(oracle, input) in
        let subject_trace = execute(subject, input) in
        traces_equivalent(oracle_trace, subject_trace)
}
```

### API Integration

```rust
// In DashProve main API
impl DashProve {
    /// Check behavioral equivalence between implementations
    pub async fn check_bisimulation(
        &self,
        config: BisimulationConfig,
    ) -> Result<BisimulationResult> {
        let checker = BisimulationChecker::new(config);
        checker.check_all().await
    }
}
```

### CLI Integration

```bash
# Check bisimulation
dashprove bisim \
    --oracle ./claude-code \
    --subject ./claude-code-rs \
    --inputs tests/parity/*.json \
    --criteria api,tools,output

# Generate parity test inputs
dashprove bisim generate \
    --count 100 \
    --coverage-guided \
    --output parity_tests/
```

---

## Enhancement 2: Trace-Based Verification

### Problem

We have TLA+ specs for the agent loop, but need to verify actual execution traces against them.

### Proposed Solution

Add trace verification to TLA+ backend:

```rust
// crates/dashprove-backends/src/tlaplus.rs (extension)

/// Execution trace to verify
pub struct ExecutionTrace {
    pub states: Vec<TraceState>,
    pub actions: Vec<TraceAction>,
}

pub struct TraceState {
    pub variables: HashMap<String, Value>,
    pub timestamp: Option<Instant>,
}

pub struct TraceAction {
    pub name: String,
    pub params: HashMap<String, Value>,
    pub from_state: usize,
    pub to_state: usize,
}

impl TlaPlusBackend {
    /// Verify an execution trace satisfies the TLA+ spec
    pub async fn verify_trace(
        &self,
        spec: &CompiledSpec,
        trace: &ExecutionTrace,
    ) -> Result<TraceVerificationResult> {
        // 1. Convert trace to TLA+ format
        let tla_trace = self.convert_trace(trace)?;

        // 2. Check trace is a valid behavior of spec
        let is_valid = self.check_trace_validity(spec, &tla_trace).await?;

        // 3. Check invariants hold at each state
        let invariant_violations = self.check_invariants(spec, &tla_trace).await?;

        // 4. Check liveness properties (if trace is complete)
        let liveness_check = if trace.is_complete() {
            Some(self.check_liveness(spec, &tla_trace).await?)
        } else {
            None
        };

        Ok(TraceVerificationResult {
            is_valid_behavior: is_valid,
            invariant_violations,
            liveness_result: liveness_check,
        })
    }
}
```

### Trace Recording Integration

```rust
// crates/dashprove/src/trace.rs

/// Trait for components that can record traces
pub trait Traceable {
    fn record_state(&self) -> TraceState;
    fn record_action(&self, action: &str, params: HashMap<String, Value>) -> TraceAction;
}

/// Macro to instrument code for trace recording
#[macro_export]
macro_rules! trace_state {
    ($tracer:expr, $($var:ident),*) => {
        $tracer.record_state(hashmap! {
            $(stringify!($var) => $var.clone().into()),*
        })
    };
}

#[macro_export]
macro_rules! trace_action {
    ($tracer:expr, $name:expr, $($param:ident),*) => {
        $tracer.record_action($name, hashmap! {
            $(stringify!($param) => $param.clone().into()),*
        })
    };
}
```

### Usage in claude_code_rs

```rust
// Example instrumentation
impl Agent {
    pub async fn step(&mut self) -> Result<()> {
        // Record state before action
        trace_state!(self.tracer, state, messages, iterations, input_tokens);

        match self.state {
            AgentState::Processing => {
                trace_action!(self.tracer, "APICall", model, temperature);
                let response = self.api_client.send(&self.messages).await?;
                trace_action!(self.tracer, "APIResponse", stop_reason);
            }
            // ...
        }

        // Record state after action
        trace_state!(self.tracer, state, messages, iterations, input_tokens);

        Ok(())
    }
}
```

### CLI Integration

```bash
# Record trace during execution
DASHPROVE_TRACE=agent_loop.trace cargo run -- "hello"

# Verify trace against spec
dashprove verify-trace \
    --spec agent_executor.tla \
    --trace agent_loop.trace
```

---

## Enhancement 3: Runtime Monitor Synthesis

### Problem

We want to check invariants at runtime, not just statically. Generate runtime monitors from USL specs.

### Proposed Solution

```rust
// crates/dashprove/src/monitor.rs

/// Runtime monitor generated from spec
pub struct RuntimeMonitor {
    /// Invariants to check
    invariants: Vec<CompiledInvariant>,
    /// State to track
    state: MonitorState,
    /// Violation handler
    on_violation: Box<dyn Fn(&Violation) + Send + Sync>,
}

/// Compiled invariant that can be checked at runtime
pub struct CompiledInvariant {
    pub name: String,
    pub check: Box<dyn Fn(&MonitorState) -> bool + Send + Sync>,
    pub description: String,
}

impl RuntimeMonitor {
    /// Check all invariants against current state
    pub fn check(&self, state: &impl Monitorable) -> Vec<Violation> {
        let monitor_state = state.to_monitor_state();
        self.invariants
            .iter()
            .filter_map(|inv| {
                if !(inv.check)(&monitor_state) {
                    Some(Violation {
                        invariant: inv.name.clone(),
                        state: monitor_state.clone(),
                        timestamp: Instant::now(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Compile USL spec to runtime monitor
impl DashProve {
    pub fn compile_monitor(&self, spec: &Spec) -> Result<RuntimeMonitor> {
        let invariants = spec.invariants()
            .map(|inv| self.compile_invariant(inv))
            .collect::<Result<Vec<_>>>()?;

        Ok(RuntimeMonitor {
            invariants,
            state: MonitorState::new(),
            on_violation: Box::new(|v| eprintln!("Invariant violated: {:?}", v)),
        })
    }

    fn compile_invariant(&self, inv: &Invariant) -> Result<CompiledInvariant> {
        // Generate Rust closure from USL expression
        let check_fn = self.compile_to_rust_closure(&inv.body)?;

        Ok(CompiledInvariant {
            name: inv.name.clone(),
            check: check_fn,
            description: inv.to_string(),
        })
    }
}
```

### Proc Macro for Automatic Instrumentation

```rust
// dashprove-macros crate

/// Automatically check invariants on state changes
#[dashprove::monitored(spec = "agent_executor.usl")]
pub struct Agent {
    state: AgentState,
    messages: Vec<Message>,
    iterations: u32,
    // ...
}

// Generates:
impl Agent {
    fn __check_invariants(&self) {
        static MONITOR: Lazy<RuntimeMonitor> = Lazy::new(|| {
            DashProve::new().compile_monitor(
                &Spec::from_file("agent_executor.usl").unwrap()
            ).unwrap()
        });

        let violations = MONITOR.check(self);
        for v in violations {
            // Log, panic, or handle based on config
            tracing::error!("Invariant violated: {}", v.invariant);
        }
    }
}
```

---

## Enhancement 4: Async/Concurrent State Machine Verification

### Problem

claude_code_rs uses tokio for async operations. Need to verify:
- State machines behave correctly under async scheduling
- No race conditions in streaming parser
- Timeout handling is correct

### Proposed Solution

Integration with async verification tools:

```rust
// crates/dashprove-async/src/lib.rs

/// Async state machine verifier
pub struct AsyncVerifier {
    /// Loom model checker integration
    loom_enabled: bool,
    /// Tokio test utilities
    tokio_test: bool,
}

impl AsyncVerifier {
    /// Explore all interleavings of async operations
    pub fn explore_interleavings<F, Fut>(&self, test: F) -> InterleavingResult
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send,
    {
        if self.loom_enabled {
            loom::model(|| {
                loom::future::block_on(test());
            });
        }
        // ...
    }

    /// Verify async state machine against spec
    pub async fn verify_async_state_machine<S: AsyncStateMachine>(
        &self,
        machine: &S,
        spec: &TlaSpec,
    ) -> Result<AsyncVerificationResult> {
        // Generate all possible event orderings
        let orderings = self.generate_orderings(machine);

        // Check each ordering satisfies spec
        for ordering in orderings {
            let trace = machine.execute_with_ordering(&ordering).await;
            self.verify_trace_against_spec(&trace, spec)?;
        }

        Ok(AsyncVerificationResult::AllOrderingsValid)
    }
}

/// Trait for async state machines
pub trait AsyncStateMachine: Send + Sync {
    type State: Clone + Send;
    type Event: Clone + Send;

    fn current_state(&self) -> Self::State;
    fn possible_events(&self) -> Vec<Self::Event>;
    async fn process_event(&mut self, event: Self::Event) -> Result<()>;
}
```

### Loom Integration for Race Condition Detection

```rust
// tests/async_verification.rs

#[test]
fn streaming_parser_no_races() {
    loom::model(|| {
        let parser = Arc::new(Mutex::new(StreamingResponse::new()));

        // Simulate concurrent SSE event processing
        let p1 = parser.clone();
        let h1 = loom::thread::spawn(move || {
            p1.lock().unwrap().append_text("hello");
        });

        let p2 = parser.clone();
        let h2 = loom::thread::spawn(move || {
            p2.lock().unwrap().start_tool_use("id", "name");
        });

        h1.join().unwrap();
        h2.join().unwrap();

        // Verify final state is valid
        let final_state = parser.lock().unwrap();
        assert!(final_state.is_valid());
    });
}
```

---

## Enhancement 5: Semantic / Fuzzy Property Verification

### Problem

LLM outputs are non-deterministic. Need to verify properties like "output is semantically equivalent" or "response addresses user's question".

### Proposed Solution

```rust
// crates/dashprove-semantic/src/lib.rs

/// Semantic similarity checker
pub struct SemanticChecker {
    /// Embedding model for text comparison
    embedder: Box<dyn TextEmbedder>,
    /// Threshold for "equivalent"
    similarity_threshold: f64,
}

impl SemanticChecker {
    /// Check if two outputs are semantically equivalent
    pub async fn are_equivalent(&self, a: &str, b: &str) -> SemanticResult {
        let embed_a = self.embedder.embed(a).await?;
        let embed_b = self.embedder.embed(b).await?;

        let similarity = cosine_similarity(&embed_a, &embed_b);

        SemanticResult {
            equivalent: similarity >= self.similarity_threshold,
            similarity,
            explanation: self.generate_explanation(a, b, similarity).await?,
        }
    }

    /// Check if output satisfies natural language property
    pub async fn satisfies_property(
        &self,
        output: &str,
        property: &str,
    ) -> PropertyResult {
        // Use LLM to check property
        let prompt = format!(
            "Does the following output satisfy the property '{}'?\n\nOutput: {}\n\nAnswer YES or NO and explain.",
            property, output
        );

        let response = self.llm.complete(&prompt).await?;

        PropertyResult {
            satisfied: response.starts_with("YES"),
            confidence: self.extract_confidence(&response),
            explanation: response,
        }
    }
}

/// USL extension for semantic properties
// semantic_property output_helpful {
//     forall response: Response, query: Query .
//         addresses_question(response, query) and
//         is_factually_accurate(response) and
//         semantic_similarity(response, expected_template) >= 0.8
// }
```

### Statistical Property Testing

```rust
/// For non-deterministic properties, use statistical testing
pub struct StatisticalVerifier {
    sample_size: usize,
    confidence_level: f64,
}

impl StatisticalVerifier {
    /// Verify property holds with statistical confidence
    pub async fn verify_statistically<P: Property>(
        &self,
        system: &impl System,
        property: &P,
    ) -> StatisticalResult {
        let mut successes = 0;

        for _ in 0..self.sample_size {
            let output = system.run().await;
            if property.check(&output) {
                successes += 1;
            }
        }

        let success_rate = successes as f64 / self.sample_size as f64;
        let ci = self.confidence_interval(successes, self.sample_size);

        StatisticalResult {
            success_rate,
            confidence_interval: ci,
            passes: ci.lower >= property.required_rate(),
        }
    }
}
```

---

## Enhancement 6: Model-Based Testing / State Machine Exploration

### Problem

Want to generate test cases that exercise all state machine paths, not just random inputs.

### Proposed Solution

```rust
// crates/dashprove-mbt/src/lib.rs

/// Model-based test generator
pub struct ModelBasedTestGenerator {
    /// State machine model (from TLA+ or USL)
    model: StateMachineModel,
    /// Coverage goals
    coverage: CoverageGoal,
}

pub enum CoverageGoal {
    /// Visit all states
    StateCoverage,
    /// Execute all transitions
    TransitionCoverage,
    /// Execute all paths up to length N
    PathCoverage { max_length: usize },
    /// Cover all boundary conditions
    BoundaryCoverage,
    /// Mutation-based coverage
    MutationCoverage,
}

impl ModelBasedTestGenerator {
    /// Generate test cases from model
    pub fn generate(&self) -> Vec<TestCase> {
        match self.coverage {
            CoverageGoal::StateCoverage => self.generate_state_covering_tests(),
            CoverageGoal::TransitionCoverage => self.generate_transition_covering_tests(),
            CoverageGoal::PathCoverage { max_length } => {
                self.generate_path_covering_tests(max_length)
            }
            // ...
        }
    }

    /// Generate tests that reach specific states
    fn generate_state_covering_tests(&self) -> Vec<TestCase> {
        let states = self.model.all_states();
        states.iter()
            .map(|state| self.find_path_to_state(state))
            .filter_map(|path| path.map(|p| self.path_to_test_case(p)))
            .collect()
    }

    /// Generate tests for boundary conditions
    fn generate_boundary_tests(&self) -> Vec<TestCase> {
        // Find variables with bounds
        // Generate inputs at boundaries
        // MAX_ITERATIONS - 1, MAX_ITERATIONS, MAX_ITERATIONS + 1
        // etc.
    }
}

/// Convert model to executable tests
impl TestCase {
    pub fn to_rust_test(&self) -> String {
        format!(
            r#"
            #[test]
            fn {}() {{
                let mut agent = Agent::new(AgentConfig::default());
                {}
                assert_eq!(agent.state, {:?});
            }}
            "#,
            self.name,
            self.steps.iter().map(|s| s.to_rust()).collect::<Vec<_>>().join("\n"),
            self.expected_final_state
        )
    }
}
```

### CLI Integration

```bash
# Generate tests from TLA+ model
dashprove mbt generate \
    --model agent_executor.tla \
    --coverage transition \
    --output tests/generated/

# Generate boundary tests
dashprove mbt generate \
    --model agent_executor.tla \
    --coverage boundary \
    --output tests/boundary/
```

---

## Enhancement 7: CI/CD Integration

### Problem

Want to run verification on every PR, with incremental verification for efficiency.

### Proposed Solution

```rust
// crates/dashprove-ci/src/lib.rs

/// CI integration module
pub struct CIIntegration {
    config: CIConfig,
    cache: VerificationCache,
}

pub struct CIConfig {
    /// Git integration
    pub git_config: GitConfig,
    /// What to verify on PR
    pub pr_verification: PrVerificationConfig,
    /// Cache settings
    pub cache: CacheConfig,
}

pub struct PrVerificationConfig {
    /// Verify specs affected by changed files
    pub incremental: bool,
    /// Maximum verification time per PR
    pub timeout: Duration,
    /// Fail PR if verification fails
    pub blocking: bool,
    /// Comment results on PR
    pub comment_results: bool,
}

impl CIIntegration {
    /// Determine what needs verification based on git diff
    pub fn analyze_changes(&self, base: &str, head: &str) -> ChangeAnalysis {
        let changed_files = self.git.diff_files(base, head);

        // Map files to specs
        let affected_specs = changed_files.iter()
            .flat_map(|f| self.spec_mapping.specs_for_file(f))
            .collect();

        // Determine verification strategy
        ChangeAnalysis {
            changed_files,
            affected_specs,
            requires_full_verification: self.requires_full_check(&changed_files),
            cached_results: self.cache.get_valid(&affected_specs),
        }
    }

    /// Run verification for PR
    pub async fn verify_pr(&self, pr: &PullRequest) -> PrVerificationResult {
        let analysis = self.analyze_changes(&pr.base, &pr.head);

        let mut results = Vec::new();

        for spec in &analysis.affected_specs {
            if let Some(cached) = analysis.cached_results.get(spec) {
                results.push(cached.clone());
            } else {
                let result = self.verify_spec(spec).await;
                self.cache.store(spec, &result);
                results.push(result);
            }
        }

        PrVerificationResult {
            passed: results.iter().all(|r| r.passed()),
            results,
            cached_count: analysis.cached_results.len(),
            verified_count: analysis.affected_specs.len() - analysis.cached_results.len(),
        }
    }
}
```

### GitHub Actions Integration

```yaml
# .github/workflows/verify.yml
name: DashProve Verification

on:
  pull_request:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for incremental

      - name: Install DashProve
        run: cargo install dashprove-cli

      - name: Install verification backends
        run: |
          # TLA+
          brew install tlaplus
          # Kani
          cargo install kani-verifier && kani setup

      - name: Run incremental verification
        run: |
          dashprove ci verify \
            --base origin/main \
            --head HEAD \
            --incremental \
            --timeout 300

      - name: Comment results
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const results = require('./dashprove-results.json');
            // Post comment with verification results
```

---

## Enhancement 8: MIRI Integration

### Problem

Need to detect undefined behavior in unsafe Rust code (streaming parser, FFI).

### Proposed Solution

```rust
// crates/dashprove-miri/src/lib.rs

/// MIRI integration for UB detection
pub struct MiriIntegration {
    config: MiriConfig,
}

pub struct MiriConfig {
    /// Track raw pointer provenance
    pub track_raw_pointers: bool,
    /// Check for data races
    pub check_data_races: bool,
    /// Isolation mode
    pub isolation: IsolationMode,
}

impl MiriIntegration {
    /// Run MIRI on test suite
    pub async fn run_miri_tests(&self, test_filter: &str) -> MiriResult {
        let output = Command::new("cargo")
            .args(["+nightly", "miri", "test", test_filter])
            .env("MIRIFLAGS", self.miri_flags())
            .output()
            .await?;

        self.parse_miri_output(&output)
    }

    /// Generate harnesses for specific functions
    pub fn generate_miri_harnesses(&self, functions: &[&str]) -> String {
        functions.iter()
            .map(|f| format!(
                r#"
                #[test]
                fn miri_test_{}() {{
                    // MIRI will check for UB here
                    let input: [u8; 256] = [0; 256];
                    {}(&input);
                }}
                "#,
                f.replace("::", "_"),
                f
            ))
            .collect()
    }
}
```

### CLI Integration

```bash
# Run MIRI checks
dashprove miri \
    --test-filter streaming \
    --check-races \
    --isolation strict

# Generate MIRI harnesses
dashprove miri generate \
    --functions "StreamingResponse::append_text,StreamingResponse::finalize" \
    --output tests/miri_harnesses.rs
```

---

## Implementation Roadmap

### Phase 1: Core Parity Verification (HIGH PRIORITY)

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Bisimulation checker design | `crates/dashprove-bisim/` skeleton |
| 2 | Oracle integration (binary runner) | Run Claude Code, capture I/O |
| 3 | Differential comparison engine | Compare API requests, tool calls |
| 4 | USL bisimulation syntax | `bisimulation` keyword support |

**Success Criteria:** Can verify claude_code_rs API requests match Claude Code exactly.

### Phase 2: Runtime Verification (MEDIUM PRIORITY)

| Week | Task | Deliverable |
|------|------|-------------|
| 5 | Trace recording infrastructure | `Traceable` trait, macros |
| 6 | Trace verification with TLA+ | Verify traces against specs |
| 7 | Runtime monitor synthesis | USL → Rust closure compilation |
| 8 | Proc macro instrumentation | `#[dashprove::monitored]` |

**Success Criteria:** Can instrument claude_code_rs and verify traces against TLA+ specs.

### Phase 3: Async & Semantic (MEDIUM PRIORITY)

| Week | Task | Deliverable |
|------|------|-------------|
| 9 | Loom integration | Race condition detection |
| 10 | Async state machine verifier | Interleaving exploration |
| 11 | Semantic similarity checker | Embedding-based comparison |
| 12 | Statistical property testing | Non-determinism handling |

**Success Criteria:** Can verify streaming parser has no races, outputs are semantically correct.

### Phase 4: Developer Experience (LOWER PRIORITY)

| Week | Task | Deliverable |
|------|------|-------------|
| 13 | Model-based test generation | Generate tests from TLA+ |
| 14 | CI/CD integration | GitHub Actions workflow |
| 15 | MIRI integration | UB detection in CI |
| 16 | Documentation & examples | Comprehensive guides |

**Success Criteria:** Full verification pipeline running in CI for claude_code_rs.

---

## API Summary

```rust
// New public API surface

// Bisimulation
pub use dashprove_bisim::{BisimulationChecker, BisimulationConfig, BisimulationResult};

// Trace verification
pub use dashprove::trace::{ExecutionTrace, TraceRecorder, Traceable};

// Runtime monitors
pub use dashprove::monitor::{RuntimeMonitor, CompiledInvariant, Violation};

// Async verification
pub use dashprove_async::{AsyncVerifier, AsyncStateMachine, InterleavingResult};

// Semantic verification
pub use dashprove_semantic::{SemanticChecker, SemanticResult, StatisticalVerifier};

// Model-based testing
pub use dashprove_mbt::{ModelBasedTestGenerator, CoverageGoal, TestCase};

// CI integration
pub use dashprove_ci::{CIIntegration, PrVerificationConfig, ChangeAnalysis};
```

---

## USL Grammar Extensions

```pest
// New USL constructs

bisimulation_decl = { "bisimulation" ~ ident ~ "{" ~ bisim_body ~ "}" }
bisim_body = {
    "oracle:" ~ string_lit ~
    "subject:" ~ string_lit ~
    "equivalent on" ~ "{" ~ (ident ~ ",")* ~ ident ~ "}" ~
    "tolerance" ~ "{" ~ tolerance_spec ~ "}" ~
    bisim_property
}
tolerance_spec = { (ident ~ ":" ~ (number | percentage) ~ ","?)* }
bisim_property = { "forall" ~ typed_var ~ "." ~ bisim_expr }
bisim_expr = { "traces_equivalent" ~ "(" ~ expr ~ "," ~ expr ~ ")" }

semantic_property = { "semantic_property" ~ ident ~ "{" ~ semantic_body ~ "}" }
semantic_body = { "forall" ~ typed_var ~ "." ~ semantic_expr }
semantic_expr = {
    "addresses_question" ~ "(" ~ expr ~ "," ~ expr ~ ")" |
    "is_factually_accurate" ~ "(" ~ expr ~ ")" |
    "semantic_similarity" ~ "(" ~ expr ~ "," ~ expr ~ ")" ~ cmp_op ~ number
}
```

---

## Estimated Effort

| Enhancement | New Crates | Estimated AI Commits |
|-------------|------------|---------------------|
| Bisimulation checking | dashprove-bisim | 15-20 |
| Trace verification | (in existing) | 8-10 |
| Runtime monitors | dashprove-monitor | 10-12 |
| Async verification | dashprove-async | 12-15 |
| Semantic verification | dashprove-semantic | 8-10 |
| Model-based testing | dashprove-mbt | 10-12 |
| CI integration | dashprove-ci | 6-8 |
| MIRI integration | dashprove-miri | 4-6 |
| **Total** | **5-6 new crates** | **73-93 commits** |

---

## Dependencies

New external dependencies required:

```toml
# Cargo.toml additions

[dependencies]
# For bisimulation
tokio-process = "0.2"  # Running oracle binaries

# For semantic verification
rust-bert = "0.21"  # Text embeddings (optional, can use API)
candle = "0.3"      # Local embedding models

# For async verification
loom = "0.7"        # Concurrency testing

# For CI integration
octocrab = "0.32"   # GitHub API
```

---

## References

- [Loom - Concurrency Permutation Testing](https://github.com/tokio-rs/loom)
- [Model-Based Testing](https://en.wikipedia.org/wiki/Model-based_testing)
- [Bisimulation](https://en.wikipedia.org/wiki/Bisimulation)
- [MIRI - Rust Undefined Behavior Detector](https://github.com/rust-lang/miri)
- [TLA+ Trace Validation](https://learntla.com/topics/trace-validation.html)
