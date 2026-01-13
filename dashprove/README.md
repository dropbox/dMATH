# DashProve

| Director | Status |
|:--------:|:------:|
| MATH | ACTIVE |

Unified AI-native verification across 180 backends including LEAN 4, TLA+, Kani, Alloy, Coq, Z3, CVC5, neural network verifiers, Rust sanitizers, fuzzers, and more — with one specification language and structured results for agents.

## What it does

- Compiles Unified Specification Language (USL) properties to multiple backends
- Dispatches to the best backend per property and merges structured results
- Generates proofs, counterexamples, and tactic suggestions with confidence scores
- Learns from successful proofs to improve future runs
- Exposes a Rust crate, CLI, and REST API, plus runtime monitor generation

## Quickstart

Prerequisites: Rust stable toolchain and standard build tools.

- Run the full test suite: `cargo test --workspace`
- Check lints: `cargo clippy --workspace -- -D warnings`

### Verify with the CLI

```bash
cargo run -p dashprove-cli -- verify examples/usl/basic.usl
# Optional flags:
#   --backends lean   # force a backend
#   --learn           # store tactics that succeeded
#   --suggest         # print tactic suggestions on failure
```

Export a compiled specification:

```bash
cargo run -p dashprove-cli -- export --target tla examples/usl/basic.usl
```

Generate runtime monitors (Rust, TypeScript, or Python):

```bash
cargo run -p dashprove-cli -- monitor examples/usl/basic.usl
cargo run -p dashprove-cli -- monitor examples/usl/basic.usl --target typescript
cargo run -p dashprove-cli -- monitor examples/usl/basic.usl --target python --assertions --logging
```

### Run the REST API server

```bash
cargo run -p dashprove-server -- --port 3000
```

**Authentication and Rate Limiting:**

```bash
# Require API key authentication
cargo run -p dashprove-server -- --require-auth --api-key "my-secret-key:MyApp"

# Multiple keys with custom rate limits (requests per minute)
cargo run -p dashprove-server -- \
  --api-key "key1:App1:100" \
  --api-key "key2:App2:200" \
  --anonymous-rate-limit 5

# Configure via environment variables
export DASHPROVE_PORT=8080
export DASHPROVE_REQUIRE_AUTH=true
export DASHPROVE_API_KEYS="key1:App1:100,key2:App2:200"
export DASHPROVE_ANONYMOUS_RATE_LIMIT=5
cargo run -p dashprove-server
```

Sample requests:

```bash
# Basic verification (with API key if required)
curl -X POST http://localhost:3000/verify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key" \
  -d '{"spec": "theorem t { forall x: Bool . x or not x }"}'

curl http://localhost:3000/backends
curl http://localhost:3000/version
curl http://localhost:3000/health  # Returns JSON with shutdown state
curl http://localhost:3000/metrics # Prometheus-compatible metrics

# Admin endpoints for key management
curl http://localhost:3000/admin/keys  # List all keys
curl -X POST http://localhost:3000/admin/keys \
  -H "Content-Type: application/json" \
  -d '{"key": "new-key-12345678", "name": "NewApp", "rate_limit": 150}'
curl -X DELETE http://localhost:3000/admin/keys/new-key-12345678  # Revoke key
```

### WebSocket streaming

For long-running verifications, connect to `/ws/verify` for real-time progress updates:

```javascript
const ws = new WebSocket('ws://localhost:3000/ws/verify');

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case 'accepted': console.log('Request accepted:', msg.request_id); break;
    case 'progress': console.log(`${msg.phase}: ${msg.message} (${msg.percentage}%)`); break;
    case 'backend_started': console.log('Started:', msg.backend); break;
    case 'backend_completed': console.log('Completed:', msg.backend); break;
    case 'completed': console.log('Done!', msg.compilations.length, 'backends'); break;
    case 'error': console.error(msg.error, msg.details); break;
  }
};

ws.onopen = () => {
  ws.send(JSON.stringify({
    spec: 'theorem test { forall x: Bool . x or not x }',
    request_id: 'my-request-1'  // optional
  }));
};
```

### Use the Rust library

```rust,no_run
use dashprove::{DashProve, DashProveConfig};

# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut client = DashProve::new(DashProveConfig::default());
let result = client.verify("theorem t { forall x: Bool . x or not x }").await?;

for property in &result.properties {
    println!("{} -> {:?}", property.name, property.status);
}

// Remote mode (reuse the REST server)
let mut remote = DashProve::new(DashProveConfig::remote("http://localhost:3000"));
let result = remote.verify("theorem remote { true }").await?;
assert!(result.is_proven());
# Ok(())
# }
```

### Generate runtime monitors

```rust
use dashprove::{MonitorConfig, RuntimeMonitor};
use dashprove_usl::{parse, typecheck};

let spec = parse("theorem safe { true }")?;
let typed = typecheck(spec)?;
let monitor = RuntimeMonitor::from_spec(&typed, &MonitorConfig::default());

println!("{}", monitor.code); // Rust, TypeScript, or Python code
```

### Agent verification

Verify AI agent behavioral equivalence, check async safety, and detect undefined behavior:

```bash
# Bisimulation: verify two implementations behave identically
cargo run -p dashprove-cli -- bisim \
  --oracle ./claude-code \
  --subject ./claude-code-rs \
  --trace-dir ./traces

# Model-based test generation from TLA+ specs
cargo run -p dashprove-cli -- mbt generate \
  --model agent.tla \
  --coverage transition \
  --output tests/

# MIRI undefined behavior detection
cargo run -p dashprove-cli -- miri \
  --test-filter unsafe_code \
  --output-format json

# Verify async execution traces against TLA+ specifications
cargo run -p dashprove-cli -- verify-trace \
  --spec agent.tla \
  --trace run.json \
  --invariants safety.usl
```

See [docs/AGENT_VERIFICATION_GUIDE.md](docs/AGENT_VERIFICATION_GUIDE.md) for full documentation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DASHPROVE API                                      │
│  REST API | Rust crate | CLI | Language Server Protocol                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               UNIFIED SPECIFICATION LANGUAGE (USL)                            │
│  Theorems | Temporal | Contracts | Invariants | Refinements | Probabilistic   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENT DISPATCHER                                   │
│  Analyzes property type → Selects optimal backend(s) → Merges results         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
   ┌──────────┬──────────┬───────────┼───────────┬──────────┬──────────┐
   ▼          ▼          ▼           ▼           ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌─────────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
│LEAN 4│  │ TLA+ │  │  Kani   │  │ Alloy │  │  Coq  │  │  Z3   │  │ CVC5  │
└──────┘  └──────┘  └─────────┘  └───────┘  └───────┘  └───────┘  └───────┘
┌──────┐  ┌──────┐  ┌─────────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
│Dafny │  │Verus │  │ Creusot │  │Prusti │  │ Storm │  │ PRISM │  │Isabelle│
└──────┘  └──────┘  └─────────┘  └───────┘  └───────┘  └───────┘  └───────┘
┌────────────┐  ┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐
│α,β-CROWN   │  │ ERAN   │  │Marabou │  │ Tamarin │  │ ProVerif │  │Verifpal│
└────────────┘  └────────┘  └────────┘  └─────────┘  └──────────┘  └────────┘
```

## Project Structure

```
dashprove/
├── crates/
│   ├── dashprove/             # Main library (client + monitors)
│   ├── dashprove-ai/          # AI proof assistance and LLM integration
│   ├── dashprove-async/       # Async/concurrency verification (Loom)
│   ├── dashprove-backends/    # 180 backends (LEAN 4, TLA+, Kani, NN verifiers, etc.)
│   ├── dashprove-bisim/       # Behavioral equivalence checking
│   ├── dashprove-cli/         # Command-line interface
│   ├── dashprove-dispatcher/  # Intelligent backend selection
│   ├── dashprove-fuzz/        # Fuzz testing integration
│   ├── dashprove-knowledge/   # RAG knowledge base for verification
│   ├── dashprove-learning/    # Proof corpus, embeddings, and learning
│   ├── dashprove-lsp/         # Language Server Protocol for IDEs
│   ├── dashprove-mbt/         # Model-based test generation
│   ├── dashprove-mcp/         # Model Context Protocol integration
│   ├── dashprove-miri/        # MIRI UB detection integration
│   ├── dashprove-monitor/     # Runtime monitor synthesis
│   ├── dashprove-monitor-macros/ # Proc macros for monitor generation
│   ├── dashprove-pbt/         # Property-based testing
│   ├── dashprove-platform-api/ # Platform API verification
│   ├── dashprove-sanitizers/  # Address/thread sanitizer integration
│   ├── dashprove-selfimp/     # Self-improving verification cache
│   ├── dashprove-semantic/    # Semantic/fuzzy verification
│   ├── dashprove-server/      # REST API server
│   ├── dashprove-static/      # Static analysis integration
│   └── dashprove-usl/         # Unified Specification Language
├── usl/                       # USL standard library
└── docs/
    ├── DESIGN.md              # Full design document
    └── AGENT_VERIFICATION_GUIDE.md  # Agent verification guide
```

## Performance

Run benchmarks with `cargo bench --workspace`. Individual crate benchmarks:

```bash
cargo bench -p dashprove           # Core parsing, type checking, compilation
cargo bench -p dashprove-backends  # Counterexample parsing
cargo bench -p dashprove-ai        # ML strategy prediction
cargo bench -p dashprove-learning  # Embeddings, corpus, tactics
cargo bench -p dashprove-usl       # Dependency analysis
cargo bench -p dashprove-dispatcher # Backend selection, result merging
```

Representative results on Apple M-series:

| Operation | Simple (1 property) | Medium (4 properties) | Complex (9 properties) |
|-----------|---------------------|----------------------|------------------------|
| Parsing | 3.8 µs | 24 µs | 59 µs |
| Type checking | 1.1 µs | 3.9 µs | 8.0 µs |
| End-to-end (parse→typecheck→compile) | 6.2 µs | 33 µs | 79 µs |

Backend compilation times (medium spec, 4 properties):

| Backend | Time | Category |
|---------|------|----------|
| Kani | 198 ns | Verifying compiler |
| Alloy | 2.8 µs | Model checker |
| TLA+ | 2.7 µs | Model checker |
| SMT-LIB2 | 2.7 µs | SMT solver (Z3/CVC5) |
| Coq | 2.7 µs | Theorem prover |
| Isabelle | 2.9 µs | Theorem prover |
| Dafny | 3.6 µs | Verifying compiler |
| LEAN 4 | 5.0 µs | Theorem prover |

Runtime monitor generation (medium spec):

| Target | Time |
|--------|------|
| Python | 3.0 µs |
| Rust | 3.0 µs |
| TypeScript | 3.9 µs |

These times are for code generation only. Actual verification time depends on the external backend (LEAN 4, TLA+, etc.) and property complexity.

## Status

Implementation in active use with 7,742 tests passing, 3,894 Kani proofs, and no clippy warnings.

Docs:
- `docs/DESIGN.md` — architecture, roadmap, grammar, and backend traits
- `docs/USL_SPECIFICATION.md` — language grammar, operators, and examples
- `docs/BACKEND_GUIDE.md` — backend trait contract and runtime details
- `docs/API_REFERENCE.md` — Rust API, CLI, and REST endpoints

## License

MIT
