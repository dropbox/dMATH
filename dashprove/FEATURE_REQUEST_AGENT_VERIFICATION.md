# Feature Request: Agent System Verification Capabilities

**Requested By:** claude_code_rs project
**Date:** 2025-12-20
**Priority:** P0 - Required for claude_code_rs formal verification
**Target:** DashProve verification platform

---

## Summary

The claude_code_rs project requires formal verification of an AI agent system. DashProve's existing 20 backends cover static verification well, but agent systems require additional capabilities for:

1. **Behavioral equivalence** - Proving two implementations behave identically
2. **Runtime verification** - Checking invariants during execution
3. **Async safety** - Verifying concurrent/async state machines
4. **Non-determinism handling** - Verifying properties of LLM-based systems

This document specifies 7 new capabilities as feature requests for DashProve.

---

## Feature 1: Bisimulation / Behavioral Equivalence Checker

**Priority:** P0 (Blocking)
**Estimated Effort:** 15-20 AI commits
**New Crate:** `dashprove-bisim`

### Problem

claude_code_rs must behave **exactly** like Claude Code. Need to verify that given identical inputs, both implementations produce equivalent:
- API requests (headers, body structure, tool definitions)
- Tool call sequences
- Final outputs

### Requirements

```rust
/// Configuration for bisimulation checking
pub struct BisimulationConfig {
    pub oracle: OracleConfig,           // Reference implementation
    pub test_subject: TestSubjectConfig, // Implementation under test
    pub equivalence_criteria: EquivalenceCriteria,
    pub nondeterminism_strategy: NondeterminismStrategy,
}

pub enum OracleConfig {
    Binary { path: PathBuf, args: Vec<String> },
    RecordedTraces { trace_dir: PathBuf },
    HttpEndpoint { url: String },
}

pub struct EquivalenceCriteria {
    pub api_requests: bool,
    pub tool_calls: bool,
    pub output: bool,
    pub timing_tolerance: Option<f64>,
    pub semantic_comparison: bool,
}

pub enum NondeterminismStrategy {
    ExactMatch,
    SpecSatisfaction,
    SemanticSimilarity { threshold: f64 },
    DistributionMatch { samples: usize },
}

pub struct BisimulationResult {
    pub equivalent: bool,
    pub differences: Vec<Difference>,
    pub traces: Option<(Trace, Trace)>,
    pub confidence: f64,
}

#[async_trait]
pub trait BisimulationChecker: Send + Sync {
    async fn check(&self, input: &TestInput) -> Result<BisimulationResult>;
    async fn check_batch(&self, inputs: &[TestInput]) -> Result<Vec<BisimulationResult>>;
    async fn generate_inputs(&self, count: usize) -> Vec<TestInput>;
}
```

### USL Extension

```usl
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

### CLI Commands

```bash
dashprove bisim \
    --oracle ./claude-code \
    --subject ./claude-code-rs \
    --inputs tests/parity/*.json \
    --criteria api,tools,output

dashprove bisim generate \
    --count 100 \
    --coverage-guided \
    --output parity_tests/
```

### Acceptance Criteria

- [ ] Can run binary oracle and capture stdin/stdout/stderr
- [ ] Can compare API request JSON structures with configurable tolerance
- [ ] Can compare tool call sequences (name, arguments, order)
- [ ] Can handle non-determinism via semantic similarity or statistical testing
- [ ] USL `bisimulation` keyword parses and type-checks
- [ ] CLI `bisim` command works end-to-end

---

## Feature 2: Trace-Based Verification

**Priority:** P1
**Estimated Effort:** 8-10 AI commits
**Location:** Extend `dashprove-backends` TLA+ backend

### Problem

We have TLA+ specifications for state machines but need to verify that actual execution traces conform to them.

### Requirements

```rust
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
    ) -> Result<TraceVerificationResult>;
}

pub struct TraceVerificationResult {
    pub is_valid_behavior: bool,
    pub invariant_violations: Vec<InvariantViolation>,
    pub liveness_result: Option<LivenessResult>,
}
```

### Trace Recording Macros

```rust
pub trait Traceable {
    fn record_state(&self) -> TraceState;
    fn record_action(&self, action: &str, params: HashMap<String, Value>) -> TraceAction;
}

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

### CLI Commands

```bash
dashprove verify-trace \
    --spec agent_executor.tla \
    --trace agent_loop.trace
```

### Acceptance Criteria

- [ ] Can parse execution traces from JSON format
- [ ] Can verify traces are valid behaviors of TLA+ spec
- [ ] Can check invariants hold at each trace state
- [ ] Can check liveness properties on complete traces
- [ ] Trace recording macros work for Rust instrumentation

---

## Feature 3: Runtime Monitor Synthesis

**Priority:** P1
**Estimated Effort:** 10-12 AI commits
**New Crate:** `dashprove-monitor`

### Problem

Want to check invariants at runtime, not just statically. Generate executable monitors from USL specifications.

### Requirements

```rust
pub struct RuntimeMonitor {
    invariants: Vec<CompiledInvariant>,
    state: MonitorState,
    on_violation: Box<dyn Fn(&Violation) + Send + Sync>,
}

pub struct CompiledInvariant {
    pub name: String,
    pub check: Box<dyn Fn(&MonitorState) -> bool + Send + Sync>,
    pub description: String,
}

impl RuntimeMonitor {
    pub fn check(&self, state: &impl Monitorable) -> Vec<Violation>;
}

impl DashProve {
    pub fn compile_monitor(&self, spec: &Spec) -> Result<RuntimeMonitor>;
}
```

### Proc Macro for Instrumentation

```rust
#[dashprove::monitored(spec = "agent_executor.usl")]
pub struct Agent {
    state: AgentState,
    messages: Vec<Message>,
    iterations: u32,
}

// Auto-generates invariant checking on state changes
```

### Acceptance Criteria

- [ ] Can compile USL invariants to Rust closures
- [ ] RuntimeMonitor can check invariants at runtime
- [ ] Violations include state snapshot and invariant name
- [ ] Proc macro `#[monitored]` generates instrumentation code
- [ ] Performance overhead < 5% for typical workloads

---

## Feature 4: Async/Concurrent State Machine Verification

**Priority:** P0 (Blocking)
**Estimated Effort:** 12-15 AI commits
**New Crate:** `dashprove-async`

### Problem

claude_code_rs uses tokio for async operations. Need to verify:
- State machines behave correctly under async scheduling
- No race conditions in streaming parser
- Timeout handling is correct

### Requirements

```rust
pub struct AsyncVerifier {
    loom_enabled: bool,
    tokio_test: bool,
}

impl AsyncVerifier {
    /// Explore all interleavings of async operations
    pub fn explore_interleavings<F, Fut>(&self, test: F) -> InterleavingResult
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send;

    /// Verify async state machine against spec
    pub async fn verify_async_state_machine<S: AsyncStateMachine>(
        &self,
        machine: &S,
        spec: &TlaSpec,
    ) -> Result<AsyncVerificationResult>;
}

pub trait AsyncStateMachine: Send + Sync {
    type State: Clone + Send;
    type Event: Clone + Send;

    fn current_state(&self) -> Self::State;
    fn possible_events(&self) -> Vec<Self::Event>;
    async fn process_event(&mut self, event: Self::Event) -> Result<()>;
}

pub struct InterleavingResult {
    pub total_interleavings: usize,
    pub violations: Vec<InterleavingViolation>,
    pub coverage: InterleavingCoverage,
}
```

### Loom Integration

```rust
// Automatic Loom test generation
pub fn generate_loom_test(
    state_machine: &impl AsyncStateMachine,
    invariants: &[Invariant],
) -> String;
```

### Acceptance Criteria

- [ ] Integration with Loom for race condition detection
- [ ] Can explore async interleavings systematically
- [ ] Can verify async state machines against TLA+ specs
- [ ] Can detect data races in concurrent code
- [ ] Generates Loom test harnesses automatically

---

## Feature 5: Semantic / Fuzzy Property Verification

**Priority:** P2
**Estimated Effort:** 8-10 AI commits
**New Crate:** `dashprove-semantic`

### Problem

LLM outputs are non-deterministic. Need to verify properties like "output is semantically equivalent" rather than byte-exact matching.

### Requirements

```rust
pub struct SemanticChecker {
    embedder: Box<dyn TextEmbedder>,
    similarity_threshold: f64,
}

impl SemanticChecker {
    pub async fn are_equivalent(&self, a: &str, b: &str) -> SemanticResult;
    pub async fn satisfies_property(&self, output: &str, property: &str) -> PropertyResult;
}

pub struct SemanticResult {
    pub equivalent: bool,
    pub similarity: f64,
    pub explanation: String,
}

pub struct StatisticalVerifier {
    sample_size: usize,
    confidence_level: f64,
}

impl StatisticalVerifier {
    pub async fn verify_statistically<P: Property>(
        &self,
        system: &impl System,
        property: &P,
    ) -> StatisticalResult;
}
```

### USL Extension

```usl
semantic_property output_helpful {
    forall response: Response, query: Query .
        addresses_question(response, query) and
        is_factually_accurate(response) and
        semantic_similarity(response, expected) >= 0.8
}
```

### Acceptance Criteria

- [ ] Can compute semantic similarity between texts
- [ ] Can verify statistical properties over multiple runs
- [x] USL `semantic_property` keyword parses
- [ ] Built-in predicates: `semantic_similarity`, `addresses_question`
- [ ] Support for local embedding models (candle) or API-based

---

## Feature 6: Model-Based Test Generation

**Priority:** P2
**Estimated Effort:** 10-12 AI commits
**New Crate:** `dashprove-mbt`

### Problem

Want to generate test cases that exercise all state machine paths systematically, not just random inputs.

### Requirements

```rust
pub struct ModelBasedTestGenerator {
    model: StateMachineModel,
    coverage: CoverageGoal,
}

pub enum CoverageGoal {
    StateCoverage,
    TransitionCoverage,
    PathCoverage { max_length: usize },
    BoundaryCoverage,
    MutationCoverage,
}

impl ModelBasedTestGenerator {
    pub fn generate(&self) -> Vec<TestCase>;
    pub fn generate_boundary_tests(&self) -> Vec<TestCase>;
}

pub struct TestCase {
    pub name: String,
    pub steps: Vec<TestStep>,
    pub expected_final_state: State,
}

impl TestCase {
    pub fn to_rust_test(&self) -> String;
    pub fn to_proptest(&self) -> String;
}
```

### CLI Commands

```bash
dashprove mbt generate \
    --model agent_executor.tla \
    --coverage transition \
    --output tests/generated/

dashprove mbt generate \
    --model agent_executor.tla \
    --coverage boundary \
    --output tests/boundary/
```

### Acceptance Criteria

- [ ] Can parse TLA+ spec into state machine model
- [ ] Can generate tests for state coverage
- [ ] Can generate tests for transition coverage
- [ ] Can generate boundary condition tests
- [ ] Output as Rust `#[test]` functions or proptest strategies

---

## Feature 7: MIRI Integration

**Priority:** P3
**Estimated Effort:** 4-6 AI commits
**New Crate:** `dashprove-miri`

### Problem

Need to detect undefined behavior in unsafe Rust code (streaming parser, FFI boundaries).

### Requirements

```rust
pub struct MiriIntegration {
    config: MiriConfig,
}

pub struct MiriConfig {
    pub track_raw_pointers: bool,
    pub check_data_races: bool,
    pub isolation: IsolationMode,
}

impl MiriIntegration {
    pub async fn run_miri_tests(&self, test_filter: &str) -> MiriResult;
    pub fn generate_miri_harnesses(&self, functions: &[&str]) -> String;
}
```

### CLI Commands

```bash
dashprove miri \
    --test-filter streaming \
    --check-races \
    --isolation strict

dashprove miri generate \
    --functions "StreamingResponse::append_text" \
    --output tests/miri_harnesses.rs
```

### Acceptance Criteria

- [ ] Can invoke MIRI on test suite
- [ ] Can parse MIRI output into structured results
- [ ] Can generate MIRI test harnesses for specific functions
- [ ] Integration with DashProve verification results

---

## New Crate Summary

| Crate | Purpose | Priority | Effort |
|-------|---------|----------|--------|
| `dashprove-bisim` | Behavioral equivalence checking | P0 | 15-20 |
| `dashprove-async` | Async/Loom verification | P0 | 12-15 |
| `dashprove-monitor` | Runtime monitor synthesis | P1 | 10-12 |
| `dashprove-semantic` | Semantic/fuzzy verification | P2 | 8-10 |
| `dashprove-mbt` | Model-based test generation | P2 | 10-12 |
| `dashprove-miri` | MIRI integration | P3 | 4-6 |

**Total:** 6 new crates, 59-75 AI commits

---

## USL Grammar Extensions Required

```pest
// Bisimulation
bisimulation_decl = { "bisimulation" ~ ident ~ "{" ~ bisim_body ~ "}" }
bisim_body = {
    "oracle:" ~ string_lit ~
    "subject:" ~ string_lit ~
    "equivalent on" ~ "{" ~ (ident ~ ",")* ~ ident ~ "}" ~
    ("tolerance" ~ "{" ~ tolerance_spec ~ "}")? ~
    bisim_property?
}
tolerance_spec = { (ident ~ ":" ~ (number | percentage) ~ ","?)* }
bisim_property = { "forall" ~ typed_var ~ "." ~ bisim_expr }
bisim_expr = { "traces_equivalent" ~ "(" ~ expr ~ "," ~ expr ~ ")" }

// Semantic properties
semantic_property = { "semantic_property" ~ ident ~ "{" ~ semantic_body ~ "}" }
semantic_body = { "forall" ~ typed_var ~ "." ~ semantic_expr }
semantic_expr = {
    "addresses_question" ~ "(" ~ expr ~ "," ~ expr ~ ")" |
    "is_factually_accurate" ~ "(" ~ expr ~ ")" |
    "semantic_similarity" ~ "(" ~ expr ~ "," ~ expr ~ ")" ~ cmp_op ~ number
}
```

---

## Dependencies Required

```toml
# New dependencies for dashprove workspace

[workspace.dependencies]
# Bisimulation - process execution
tokio-process = "1.0"

# Async verification - race detection
loom = "0.7"

# Semantic verification - embeddings
candle-core = "0.3"
candle-transformers = "0.3"
# Or API-based: reqwest already present

# MIRI - subprocess
# (uses cargo miri, no new deps)
```

---

## Implementation Order Recommendation

### Phase 8: Core Agent Verification (Weeks 1-6)

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | `dashprove-bisim` skeleton | Oracle runner, trace capture |
| 3-4 | Bisim comparison engine | API/tool/output comparison |
| 5-6 | `dashprove-async` with Loom | Race detection, interleaving |

### Phase 9: Runtime Verification (Weeks 7-10)

| Week | Task | Deliverable |
|------|------|-------------|
| 7-8 | Trace verification in TLA+ backend | verify_trace() |
| 9-10 | `dashprove-monitor` | USL to Rust monitor compilation |

### Phase 10: Advanced Features (Weeks 11-14)

| Week | Task | Deliverable |
|------|------|-------------|
| 11-12 | `dashprove-semantic` | Embedding similarity |
| 13-14 | `dashprove-mbt` | Test generation from TLA+ |

### Phase 11: Polish (Weeks 15-16)

| Week | Task | Deliverable |
|------|------|-------------|
| 15 | `dashprove-miri` | MIRI integration |
| 16 | Documentation, examples | Guides for agent verification |

---

## Success Criteria

1. **Bisimulation:** Can verify claude_code_rs API requests match Claude Code
2. **Async:** Can prove streaming parser has no race conditions via Loom
3. **Trace:** Can verify execution traces against TLA+ specs
4. **Monitor:** Can compile USL invariants to runtime monitors
5. **Semantic:** Can verify LLM outputs with semantic similarity
6. **MBT:** Can generate tests covering all TLA+ states/transitions
7. **MIRI:** Can detect UB in unsafe code paths

---

## Contact

**Requesting Project:** claude_code_rs (https://github.com/dropbox/claude_code_rs)
**Target Project:** DashProve (https://github.com/dropbox/dMATH/dashprove)
