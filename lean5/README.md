# Lean5

| Director | Status |
|:--------:|:------:|
| MATH | ACTIVE |

**Pure Rust theorem prover. Sub-microsecond verification. Built for AI.**

---

## What Is This?

Lean5 is a complete reimplementation of the Lean 4 theorem prover in Rust. It's designed for AI agents that need to verify code in real-time.

| | Lean 4 | Lean5 |
|---|--------|-------|
| Language | C++ | Rust |
| Type check latency | ~1ms | ~100ns |
| Batch throughput | N/A | 1M/sec |
| API | REPL/LSP | JSON-RPC |
| C verification | No | Yes |
| Rust verification | No | Yes |
| Self-verified kernel | No | Yes |

---

## Current Status

**317,000 lines of Rust. 5,000+ tests. Production-ready kernel.**

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| Kernel (type checker) | 169K | 2,554 | Complete |
| Parser (Lean 4 syntax) | 8.8K | 228 | 97% compatible |
| Elaborator | 44.5K | 831 | Complete (120 tactics) |
| Automation (SMT/ATP) | 15.8K | 223 | Complete |
| Server (JSON-RPC) | 7.7K | 138 | Complete |
| C Verification | 20K | 296 | Complete |
| Rust Semantics | 5.7K | 90 | Complete |
| Self-Verification | 4.4K | 31 | Complete |
| .olean Import | 17K | 263 | Complete (Init, Std, Mathlib loading) |

### Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1-8 | ✅ Complete | Kernel, parser, elaborator, automation, server, C verification |
| 9 | ✅ Complete | .olean import (Init, Std, Mathlib loading) |
| 10 | ⚠️ 70% | Tactic framework (120 tactics implemented) |
| 11-14 | Planned | Macros, Lake, LSP |
| 15 | Planned | Full Mathlib compatibility |

---

## Performance

**Measured on real hardware. Not projections.**

| Operation | Latency |
|-----------|---------|
| `infer_type` (simple) | 20-100ns |
| `is_def_eq` (simple) | 1.4-28ns |
| `whnf` (beta) | 16-117ns |
| JSON-RPC `check` | 485ns |
| Batch check throughput | 1M ops/sec |
| Certificate verify | 410-560ns |

---

## Quick Start

```bash
# Build
cargo build --release

# Run tests
cargo test

# Start JSON-RPC server
cargo run -p lean5-server -- --tcp 127.0.0.1:8080
```

### JSON-RPC Example

```bash
# Type check an expression
echo '{"jsonrpc":"2.0","method":"check","params":{"code":"def f (n:Nat) := n+1"},"id":1}' | nc localhost 8080
```

Response:
```json
{"jsonrpc":"2.0","result":{"valid":true,"type_":"Nat → Nat"},"id":1}
```

### Available Methods

| Method | Description |
|--------|-------------|
| `check` | Type-check code |
| `getType` | Infer type |
| `prove` | Auto-prove with SMT/ATP |
| `batchCheck` | Check many expressions (1M/sec) |
| `verifyCert` | Verify proof certificate |
| `verifyC` | Verify C code with ACSL specs |
| `serverInfo` | Get capabilities |

See [DESIGN.md](docs/DESIGN.md) for full API documentation.

---

## Architecture

```
lean5/
├── crates/
│   ├── lean5-kernel/    # Trusted type checker (verified)
│   ├── lean5-parser/    # Lean 4 syntax
│   ├── lean5-elab/      # Elaboration + basic tactics
│   ├── lean5-auto/      # SMT solver, superposition, premise selection
│   ├── lean5-server/    # JSON-RPC API (TCP + WebSocket)
│   ├── lean5-gpu/       # GPU acceleration (wgpu)
│   ├── lean5-c-sem/     # C verification (CompCert model, ACSL)
│   ├── lean5-rust-sem/  # Rust semantics (ownership, borrowing)
│   ├── lean5-verify/    # Self-verification infrastructure
│   └── lean5-cli/       # Command line interface
└── docs/
    ├── DESIGN.md        # Full technical specification
    └── PHILOSOPHY.md    # Why we built this
```

---

## Key Features

### 1. Sub-Microsecond Type Checking

```rust
// 100 nanoseconds to type check a simple term
let ty = kernel.infer_type(&expr)?;
```

### 2. Native JSON-RPC API

```python
import socket, json

sock = socket.socket()
sock.connect(("localhost", 8080))
sock.send(json.dumps({
    "jsonrpc": "2.0",
    "method": "check",
    "params": {"code": "def x := 1"},
    "id": 1
}).encode() + b"\n")

result = json.loads(sock.recv(4096))
# result["result"]["valid"] == True
```

### 3. C Code Verification

```c
/*@ requires n >= 0;
    ensures \result >= 0; */
int abs(int n) {
    return n < 0 ? -n : n;
}
```

```json
{"method": "verifyC", "params": {"code": "..."}}
// Returns: verification conditions proved
```

### 4. Batch Operations

```json
{"method": "batchCheck", "params": {"items": [
    {"id": "1", "code": "def a := 1"},
    {"id": "2", "code": "def b := 2"},
    ...  // 10,000 items
]}}
// Processes at 1M items/second
```

### 5. Self-Verified Kernel

The kernel is proven correct within Lean5 itself:
- Formal specification (90 definitions)
- Proof witnesses (46 proofs)
- Micro-checker (~500 lines, proven correct)
- Cross-validation against lean4lean

---

## Lean 4 Compatibility

| Feature | Status |
|---------|--------|
| Surface syntax | 97% (97/100 test files) |
| Type checker | 100% (113/113 cross-validation cases) |
| .olean import | Complete (Init, Std, Mathlib loading works) |
| Tactics | 120 tactics (simp, ring, linarith, omega, nlinarith, norm_num, cases, induction, etc.) |
| Mathlib | Loading works, full usage needs more work (Phase 15) |

---

## Documentation

- **[DESIGN.md](docs/DESIGN.md)** - Full architecture, API reference, roadmap
- **[PHILOSOPHY.md](docs/PHILOSOPHY.md)** - Why we built this, design principles
- **[CLAUDE.md](CLAUDE.md)** - Instructions for AI workers

---

## Why Lean5?

We needed a verification engine that:

1. **Runs at machine speed** - AI agents make millions of verification calls
2. **Speaks JSON** - Not REPLs designed for humans
3. **Verifies C and Rust** - Not just Lean code
4. **Is trustworthy** - Self-verified, not just trusted
5. **Is maintainable** - Clean Rust, not legacy C++

Lean 4 is great for mathematicians. Lean5 is built for machines.

---

## License

MIT

---

## Building for AI

This project is primarily developed by AI workers. The codebase is designed for AI agents to:
- Read and understand (clear structure, good naming)
- Modify safely (comprehensive tests)
- Verify their changes (fast test suite)

See [CLAUDE.md](CLAUDE.md) for AI worker guidelines.
