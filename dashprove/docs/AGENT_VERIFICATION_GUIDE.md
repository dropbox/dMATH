# DashProve Agent Verification Guide

This guide covers DashProve's capabilities for verifying AI agent systems, including behavioral equivalence checking, runtime verification, async safety, and non-determinism handling.

---

## Overview

DashProve provides six specialized crates for agent system verification:

| Crate | Purpose | Priority |
|-------|---------|----------|
| `dashprove-bisim` | Behavioral equivalence checking | P0 |
| `dashprove-async` | Async/Loom verification | P0 |
| `dashprove-monitor` | Runtime monitor synthesis | P1 |
| `dashprove-semantic` | Semantic/fuzzy verification | P2 |
| `dashprove-mbt` | Model-based test generation | P2 |
| `dashprove-miri` | MIRI undefined behavior detection | P3 |

---

## 1. Behavioral Equivalence (dashprove-bisim)

Verify that two implementations behave identically given the same inputs.

### Use Cases

- Verifying a Rust reimplementation matches original behavior
- Comparing API request formats between implementations
- Ensuring tool call sequences are equivalent
- Validating output equivalence (exact or semantic)

### Core Types

```rust
use dashprove_bisim::{
    OracleRunner, TraceRecorder, TraceDiffer,
    BisimulationConfig, EquivalenceCriteria, Trace,
};

// Configure bisimulation check
let config = BisimulationConfig {
    oracle_path: PathBuf::from("./claude-code"),
    subject_path: PathBuf::from("./claude-code-rs"),
    criteria: EquivalenceCriteria {
        api_requests: true,
        tool_calls: true,
        output: true,
        timing_tolerance: Some(0.1),  // 10% timing variance allowed
    },
};

// Run oracle and capture trace
let oracle = OracleRunner::new(&config.oracle_path);
let oracle_trace = oracle.run_with_input(input).await?;

// Run subject and capture trace
let subject = OracleRunner::new(&config.subject_path);
let subject_trace = subject.run_with_input(input).await?;

// Compare traces
let differ = TraceDiffer::new(&config.criteria);
let result = differ.compare(&oracle_trace, &subject_trace);

if result.equivalent {
    println!("Implementations are equivalent");
} else {
    for diff in &result.differences {
        println!("Difference: {:?}", diff);
    }
}
```

### USL Specification

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
}
```

### CLI Commands

```bash
# Basic bisimulation check
dashprove bisim \
    --oracle ./claude-code \
    --subject ./claude-code-rs \
    --input test_input.json

# Compare with recorded traces
dashprove bisim \
    --oracle-traces traces/oracle/ \
    --subject ./claude-code-rs \
    --criteria api,tools,output

# Generate test inputs for coverage
dashprove bisim generate \
    --count 100 \
    --output parity_tests/
```

---

## 2. Async/Concurrent Verification (dashprove-async)

Verify async state machines and detect race conditions using Loom integration.

### Use Cases

- Detecting race conditions in streaming parsers
- Verifying concurrent state machine correctness
- Exploring all possible async interleavings
- Validating timeout handling behavior

### Core Types

```rust
use dashprove_async::{
    AsyncVerifier, AsyncStateMachine, InterleavingExplorer,
    ExecutionTrace, TraceState, TraceAction,
};

// Define async state machine
struct StreamingParser {
    buffer: String,
    state: ParserState,
}

impl AsyncStateMachine for StreamingParser {
    type State = ParserState;
    type Event = StreamEvent;

    fn current_state(&self) -> Self::State {
        self.state.clone()
    }

    fn possible_events(&self) -> Vec<Self::Event> {
        vec![StreamEvent::Chunk, StreamEvent::End, StreamEvent::Error]
    }

    async fn process_event(&mut self, event: Self::Event) -> Result<()> {
        // State transition logic
    }
}

// Explore interleavings with Loom
let verifier = AsyncVerifier::new();
let result = verifier.explore_interleavings(|| async {
    let parser = StreamingParser::new();
    // Concurrent operations to test
}).await;

println!("Explored {} interleavings", result.total_interleavings);
for violation in &result.violations {
    println!("Race condition: {:?}", violation);
}
```

### Loom Integration

```rust
use dashprove_async::loom_harness::LoomHarnessGenerator;

// Generate Loom test harnesses automatically
let generator = LoomHarnessGenerator::new();
let harness = generator.generate_for_state_machine::<StreamingParser>(&invariants);

// The generated test runs under Loom's scheduler
loom::model(|| {
    // Test code with race detection
});
```

### TLA+ Trace Verification

```rust
use dashprove_async::TlaTraceVerifier;

// Verify execution trace against TLA+ spec
let verifier = TlaTraceVerifier::new();
let result = verifier.verify_trace(
    &tla_spec,
    &execution_trace,
).await?;

if !result.is_valid_behavior {
    for violation in &result.invariant_violations {
        println!("Invariant {} violated at state {}",
            violation.invariant_name, violation.state_index);
    }
}
```

### CLI Commands

```bash
# Verify trace against TLA+ spec
dashprove verify-trace \
    --spec agent_executor.tla \
    --trace execution.json

# Run Loom-based race detection
dashprove async \
    --loom \
    --test-filter streaming
```

---

## 3. Runtime Monitoring (dashprove-monitor)

Compile USL specifications to runtime monitors that check invariants during execution.

### Use Cases

- Runtime invariant checking in production
- Detecting specification violations early
- Monitoring state machine transitions
- Capturing violation context for debugging

### Core Types

```rust
use dashprove_monitor::{
    RuntimeMonitor, CompiledInvariant, MonitorState,
    RuntimeViolation, ScopedMonitor,
};

// Create runtime monitor from USL spec
let monitor = RuntimeMonitor::from_usl_spec(&spec)?;

// Check invariants manually
let state = MonitorState::from_value(&current_state);
let violations = monitor.check(&state);

for violation in &violations {
    println!("Violation: {} at {:?}",
        violation.invariant_name, violation.location);
    println!("State snapshot: {:?}", violation.state_snapshot);
}

// Scoped monitoring (checks on scope exit)
{
    let _guard = ScopedMonitor::new(&monitor, &state);
    // Operations that modify state
} // Invariants checked here
```

### Proc Macro for Automatic Monitoring

```rust
use dashprove_monitor_macros::Monitored;

#[derive(Monitored)]
#[monitored(spec = "agent.usl")]
pub struct AgentState {
    pub iteration: u32,
    pub messages: Vec<Message>,
    pub tool_calls: Vec<ToolCall>,
}

// State changes automatically checked against spec
let mut agent = AgentState::new();
agent.iteration += 1;  // Triggers invariant check
```

### USL Invariants to Rust Closures

```rust
use dashprove_usl::RustClosureCompiler;

// Compile USL invariant to Rust closure
let compiler = RustClosureCompiler::new();
let closure = compiler.compile_invariant(&usl_invariant)?;

// Use closure for runtime checking
let is_valid = closure(&current_state);
```

### Trace Recording Macros

```rust
use dashprove_monitor::{trace_state, trace_action, TraceContext};

let mut ctx = TraceContext::new();

// Record state
trace_state!(ctx, iteration, messages, tool_calls);

// Record action
trace_action!(ctx, "process_message", message_id, content);

// Export trace for analysis
ctx.export_json("execution_trace.json")?;
```

### CLI Commands

```bash
# Verify trace with invariants
dashprove verify-trace \
    --spec agent.tla \
    --trace run.json \
    --invariants "Inv_BoundedIterations,Inv_ValidState"

# Check liveness properties
dashprove verify-trace \
    --spec agent.tla \
    --trace run.json \
    --liveness "Eventually_Terminates"
```

---

## 4. Semantic Verification (dashprove-semantic)

Verify properties of non-deterministic outputs using semantic similarity and statistical testing.

### Use Cases

- Verifying LLM outputs are semantically equivalent
- Statistical property verification over multiple runs
- Fuzzy matching for natural language outputs
- Threshold-based similarity checking

### Core Types

```rust
use dashprove_semantic::{
    SemanticChecker, TextEmbedder, SimilarityMetric,
    StatisticalVerifier, WilsonConfidenceInterval,
};

// Create semantic checker
let embedder = TextEmbedder::tfidf();  // Or local model
let checker = SemanticChecker::new(embedder);

// Check semantic similarity
let result = checker.compare(
    "The function returns an integer",
    "This function outputs a whole number",
).await?;

println!("Similarity: {:.2}", result.similarity);
println!("Equivalent: {}", result.equivalent);

// Statistical verification
let verifier = StatisticalVerifier::new(StatisticalConfig {
    sample_size: 100,
    confidence_level: 0.95,
    early_stopping: true,
});

let result = verifier.verify(
    &system,
    |output| output.contains_expected_content(),
).await?;

println!("Property holds with probability: {:.2}", result.probability);
println!("Confidence interval: [{:.2}, {:.2}]",
    result.confidence_interval.lower,
    result.confidence_interval.upper);
```

### USL Semantic Properties

```usl
semantic_property helpful_response {
    forall response: Response, query: Query .
        addresses_question(response, query) and
        semantic_similarity(response, expected) >= 0.8
}
```

### Built-in Predicates

| Predicate | Description |
|-----------|-------------|
| `semantic_similarity(a, b)` | Cosine similarity between embeddings |
| `addresses_question(response, query)` | Response is relevant to query |
| `contains_concepts(text, concepts)` | Text mentions required concepts |

### CLI Commands

```bash
# Check semantic similarity
dashprove semantic \
    --expected expected_output.txt \
    --actual actual_output.txt \
    --threshold 0.85

# Statistical verification
dashprove semantic \
    --samples 100 \
    --property "contains_tool_call" \
    --confidence 0.95
```

---

## 5. Model-Based Test Generation (dashprove-mbt)

Generate test cases from TLA+ specifications to achieve coverage goals.

### Use Cases

- Generating tests covering all states
- Generating tests covering all transitions
- Boundary value testing from model constraints
- Systematic exploration of state space

### Core Types

```rust
use dashprove_mbt::{
    StateMachineModel, TestGenerator, CoverageGoal,
    TestCase, TestStep,
};

// Parse TLA+ spec into model
let model = StateMachineModel::from_tla(&tla_spec)?;

// Generate tests for transition coverage
let generator = TestGenerator::new(model, CoverageGoal::TransitionCoverage);
let tests = generator.generate();

for test in &tests {
    println!("Test: {}", test.name);
    for step in &test.steps {
        println!("  {} -> {}", step.action, step.expected_state);
    }
}

// Generate boundary tests
let boundary_gen = TestGenerator::new(model, CoverageGoal::BoundaryCoverage);
let boundary_tests = boundary_gen.generate();
```

### Coverage Goals

| Goal | Description |
|------|-------------|
| `StateCoverage` | Visit every reachable state |
| `TransitionCoverage` | Execute every transition |
| `PathCoverage` | Cover paths up to max length |
| `BoundaryCoverage` | Test at variable boundaries |

### Export Formats

```rust
// Export as Rust tests
let rust_tests = test.to_rust_test();

// Export as proptest strategies
let proptest = test.to_proptest();

// Export as JSON test suite
tests.export_json("generated_tests.json")?;
```

### CLI Commands

```bash
# Generate state coverage tests
dashprove mbt generate \
    --model agent_executor.tla \
    --coverage state \
    --output tests/state_coverage/

# Generate transition coverage tests
dashprove mbt generate \
    --model agent_executor.tla \
    --coverage transition \
    --format rust \
    --output tests/generated/

# Generate boundary tests
dashprove mbt generate \
    --model agent_executor.tla \
    --coverage boundary \
    --output tests/boundary/
```

---

## 6. MIRI Integration (dashprove-miri)

Detect undefined behavior in unsafe Rust code using MIRI (Mid-level IR Interpreter).

### Use Cases

- Detecting use-after-free bugs
- Finding data races in unsafe code
- Validating stacked borrows rules
- Testing FFI boundaries

### Core Types

```rust
use dashprove_miri::{
    MiriConfig, MiriRunner, MiriResult,
    UndefinedBehavior, UbKind, HarnessGenerator,
};

// Configure MIRI
let config = MiriConfig::default()
    .with_stacked_borrows(true)
    .with_data_race_detection(true)
    .with_isolation(false);

// Run MIRI on project
let runner = MiriRunner::new(config);
let result = runner.run_project(&project_path).await?;

// Check for undefined behavior
for ub in &result.undefined_behaviors {
    println!("UB detected: {:?}", ub.kind);
    println!("Location: {}:{}", ub.file, ub.line);
    println!("Message: {}", ub.message);
}

// Parse test results
let summary = result.summary();
println!("Tests: {} passed, {} failed", summary.passed, summary.failed);
println!("UB instances: {}", summary.ub_count);
```

### UB Kinds Detected

| Kind | Description |
|------|-------------|
| `UseAfterFree` | Accessing deallocated memory |
| `DataRace` | Concurrent unsynchronized access |
| `StackedBorrowsViolation` | Invalid borrow state |
| `UninitializedMemory` | Reading uninitialized data |
| `InvalidAlignment` | Misaligned memory access |
| `NullPointerDeref` | Null pointer dereference |
| `OutOfBoundsAccess` | Buffer overflow/underflow |
| `InvalidBool` | Non-0/1 bool value |
| `InvalidChar` | Invalid Unicode scalar |
| `DeadlockDetected` | Thread deadlock |
| `MemoryLeak` | Unreleased memory |
| `InvalidFunctionPointer` | Bad function pointer |
| `IntegerOverflow` | Arithmetic overflow |
| `UnknownUb` | Other undefined behavior |

### Harness Generation

```rust
use dashprove_miri::HarnessGenerator;

// Generate test harness for function
let generator = HarnessGenerator::new();
let harness = generator.generate_for_function(
    "StreamingResponse::append_text",
    &function_signature,
)?;

// Write harness to file
std::fs::write("tests/miri_harness.rs", harness)?;
```

### CLI Commands

```bash
# Check availability
dashprove miri --setup

# Run MIRI on project
dashprove miri --project ./my-crate

# Run with specific test filter
dashprove miri --project ./my-crate --filter streaming

# Enable strict checking
dashprove miri --project ./my-crate \
    --stacked-borrows \
    --data-races \
    --isolation

# Generate harness for function
dashprove miri harness \
    --function "StreamingResponse::append_text" \
    --file src/streaming.rs \
    --output tests/miri_harness.rs

# JSON output
dashprove miri --project ./my-crate --output json
```

---

## End-to-End Verification Workflow

### 1. Define Specifications

Create TLA+ spec for state machine behavior:

```tla+
---- MODULE AgentExecutor ----
VARIABLES state, iteration, messages

Init ==
    /\ state = "idle"
    /\ iteration = 0
    /\ messages = <<>>

ProcessMessage ==
    /\ state = "processing"
    /\ iteration' = iteration + 1
    /\ state' = IF iteration' >= 10 THEN "done" ELSE "processing"

Inv_BoundedIterations == iteration <= 10
====
```

Create USL contracts:

```usl
contract agent_executor {
    state: AgentState

    invariant bounded_iterations {
        state.iteration <= 10
    }

    invariant valid_state {
        state.status in {Idle, Processing, Done, Error}
    }
}
```

### 2. Generate Tests

```bash
# Generate tests from model
dashprove mbt generate \
    --model specs/agent_executor.tla \
    --coverage transition \
    --output tests/generated/
```

### 3. Instrument Code

```rust
use dashprove_monitor_macros::Monitored;

#[derive(Monitored)]
#[monitored(spec = "specs/agent.usl")]
pub struct AgentState {
    pub iteration: u32,
    pub status: Status,
    pub messages: Vec<Message>,
}
```

### 4. Run Verification

```bash
# Verify trace against spec
dashprove verify-trace \
    --spec specs/agent_executor.tla \
    --trace logs/execution.json

# Run bisimulation check
dashprove bisim \
    --oracle ./reference-impl \
    --subject ./rust-impl \
    --input tests/inputs/

# Check for UB with MIRI
dashprove miri --project .
```

### 5. Continuous Verification

Add to CI pipeline:

```yaml
verify:
  steps:
    - run: dashprove verify specs/*.usl
    - run: dashprove miri --project .
    - run: dashprove verify-trace --spec agent.tla --trace test_trace.json
```

---

## Best Practices

### 1. Start with Specifications

Write TLA+ or USL specifications before implementation. This clarifies behavior and enables model-based testing.

### 2. Use Layered Verification

- **Static**: USL contracts, type checking
- **Model**: TLA+ model checking
- **Runtime**: Monitor instrumentation
- **Dynamic**: MIRI, Loom testing

### 3. Automate Trace Collection

Use the `Traceable` trait and macros to automatically record execution traces for verification.

### 4. Handle Non-Determinism

For LLM-based systems:
- Use semantic similarity instead of exact matching
- Apply statistical verification with confidence intervals
- Define tolerance thresholds in specifications

### 5. Test Async Code Systematically

Use Loom integration to explore interleavings. Don't rely on random testing for concurrent code.

---

## Troubleshooting

### MIRI Not Found

```bash
# Install MIRI component
rustup +nightly component add miri

# Initialize MIRI
dashprove miri --setup
```

### Trace Format Errors

Ensure traces follow the expected JSON format:

```json
{
  "states": [
    {"variables": {"iteration": 0, "state": "idle"}, "timestamp": "..."}
  ],
  "actions": [
    {"name": "ProcessMessage", "params": {}, "from_state": 0, "to_state": 1}
  ]
}
```

### Loom Timeout

Loom explores many interleavings. For complex tests:
- Reduce shared state
- Limit concurrent operations
- Use `loom::MAX_THREADS` configuration

---

## Further Reading

- [DashProve Design Document](DESIGN.md)
- [USL Specification Reference](USL_SPECIFICATION.md)
- [API Reference](API_REFERENCE.md)
- [TLA+ Examples](tlaplus/)
- [MIRI Documentation](https://github.com/rust-lang/miri)
- [Loom Documentation](https://docs.rs/loom)
