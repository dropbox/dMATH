# DashProve Quickstart Guide

DashProve is a unified AI-native verification platform that abstracts over 200+ formal verification tools (Lean 4, TLA+, Kani, Alloy, Coq, Z3, and many more). Write specifications once in USL (Unified Specification Language) and verify against multiple backends automatically.

## Installation

### Prerequisites

- **Rust toolchain** (1.70+): Install via [rustup](https://rustup.rs/)
- **Optional backends** (DashProve works without them, but verification requires at least one):
  - Lean 4: `elan install lean4`
  - TLA+/TLC: Requires Java 11+ (`java -version`)
  - Kani: `cargo install --locked kani-verifier && kani setup`
  - Alloy: Download from [alloytools.org](https://alloytools.org/)

### Build DashProve

```bash
git clone https://github.com/dropbox/dMATH/dashprove.git
cd dashprove
cargo build --release

# Verify installation
cargo run -p dashprove-cli -- --version
cargo run -p dashprove-cli -- backends
```

## Your First Verification

### 1. Create a USL Specification

Create a file `counter.usl`:

```usl
// Define a simple counter type
type Counter = { value: Int }

// Theorem: counter value squared is non-negative
theorem non_negative_square {
    forall c: Counter . c.value * c.value >= 0
}

// Contract: increment operation
contract Counter::inc(self: Counter) -> Counter {
    requires { true }  // No precondition
    ensures  { self'.value == self.value + 1 }
}
```

### 2. Verify the Specification

```bash
# Verify against all available backends
cargo run -p dashprove-cli -- verify counter.usl

# Verify against a specific backend
cargo run -p dashprove-cli -- verify counter.usl --backends lean

# Get suggestions if verification fails
cargo run -p dashprove-cli -- verify counter.usl --suggest
```

### 3. Export to Backend Format

```bash
# Export to Lean 4
cargo run -p dashprove-cli -- export counter.usl --target lean -o counter.lean

# Export to TLA+
cargo run -p dashprove-cli -- export counter.usl --target tla+ -o counter.tla
```

## USL Language Basics

### Types

```usl
// Primitive types
type MyBool = Bool
type MyInt = Int
type MyFloat = Float
type MyString = String

// Composite types
type Point = { x: Int, y: Int }
type Stack = { elements: List<Int>, capacity: Int }
type Cache = Map<String, Int>
```

### Theorems and Invariants

```usl
// Theorem: a provable statement
theorem excluded_middle {
    forall x: Bool . x or not x
}

// Invariant: a property that must always hold
invariant positive_capacity {
    forall s: Stack . s.capacity >= 0
}
```

### Contracts (Pre/Post Conditions)

```usl
contract Stack::push(self: Stack, value: Int) -> Result<Stack> {
    requires {
        self.elements.len() < self.capacity
    }
    ensures {
        result.elements.len() == self.elements.len() + 1
    }
    ensures_err {
        self.elements.len() >= self.capacity
    }
}
```

### Temporal Properties

```usl
// Eventually reaches goal
temporal progress {
    eventually(at_goal)
}

// Always eventually responds
temporal liveness {
    always(eventually(response))
}

// Request leads to response
temporal response_guarantee {
    request ~> response
}
```

## Common Commands

### Verification

```bash
# Basic verification
dashprove verify spec.usl

# With specific backends
dashprove verify spec.usl --backends lean,tla+,kani

# With ML-based backend selection
dashprove verify spec.usl --ml

# Learn from results (adds to proof corpus)
dashprove verify spec.usl --learn

# Get suggestions on failure
dashprove verify spec.usl --suggest
```

### Verify Rust Code Against Contracts

```bash
# Verify Rust code using Kani
dashprove verify-code --code src/lib.rs --spec contracts.usl
```

### Check Backend Availability

```bash
# Check all backends
dashprove check-tools

# Check specific backends
dashprove backends
```

### Counterexample Analysis

```bash
# Explain a counterexample
dashprove explain counterexample.json

# Visualize as HTML
dashprove visualize counterexample.json --format html

# Analyze for patterns
dashprove analyze counterexample.json suggest

# Cluster multiple counterexamples
dashprove cluster cx1.json cx2.json cx3.json
```

### Proof Corpus Operations

```bash
# View corpus statistics
dashprove corpus stats

# Search for similar proofs
dashprove search "termination recursive"

# View corpus history
dashprove corpus history --days 7
```

### Runtime Monitoring

```bash
# Generate runtime monitors from spec
dashprove monitor generate spec.usl --output monitors.rs

# Verify an execution trace
dashprove verify-trace --spec agent.tla --trace run.json
```

### Model-Based Testing

```bash
# Generate tests from TLA+ model
dashprove mbt generate --model system.tla --coverage state

# Generate transition coverage tests
dashprove mbt generate --model system.tla --coverage transition
```

### Expert Knowledge System

```bash
# Get backend recommendations
dashprove expert backend --spec spec.usl

# Get help with a specific tool
dashprove expert tool kani
```

## Backend Selection Guide

| Property Type | Recommended Backends | Why |
|--------------|---------------------|-----|
| Theorems/Invariants | Lean 4, Coq, Isabelle | Theorem provers with rich type theory |
| Contracts (Rust) | Kani, Verus, Creusot | Rust-specific verification |
| Temporal Properties | TLA+, SPIN | Model checking with temporal logic |
| Memory Safety | Kani, MIRI, Valgrind | Runtime and static memory analysis |
| Distributed Systems | TLA+, Alloy | State exploration for protocols |
| Neural Networks | Marabou, ERAN, alpha-beta-CROWN | NN verification |
| Probabilistic | Storm, PRISM | Probabilistic model checking |
| Security Protocols | Tamarin, ProVerif | Security protocol analysis |

## Example Workflows

### 1. Verify a Distributed Protocol

```usl
// distributed_lock.usl
type Lock = { held_by: Option<NodeId>, waiting: Set<NodeId> }

invariant mutual_exclusion {
    forall l: Lock, n1 n2: NodeId .
        (l.held_by == Some(n1) and l.held_by == Some(n2)) implies n1 == n2
}

temporal fairness {
    forall n: NodeId .
        always(n in waiting ~> eventually(held_by == Some(n)))
}
```

```bash
dashprove verify distributed_lock.usl --backends tla+
```

### 2. Verify Rust Code Memory Safety

```usl
// buffer_contract.usl
contract Buffer::write(self: Buffer, data: List<u8>) -> Result<()> {
    requires { data.len() <= self.capacity - self.len }
    ensures  { self'.len == self.len + data.len() }
    ensures_err { data.len() > self.capacity - self.len }
}
```

```bash
dashprove verify-code --code src/buffer.rs --spec buffer_contract.usl
```

### 3. Train ML Model for Backend Selection

```bash
# Train from proof corpus
dashprove train --epochs 50 --learning-rate 0.01 --verbose

# Use trained model for verification
dashprove verify spec.usl --ml --ml-model ~/.dashprove/strategy_model.json
```

## Troubleshooting

### "Backend not available"

```bash
# Check which backends are installed
dashprove check-tools

# Install missing backend (example: Kani)
cargo install --locked kani-verifier
kani setup
```

### "Verification timeout"

```bash
# Increase timeout
dashprove verify spec.usl --timeout 300

# Try a different backend
dashprove verify spec.usl --backends alloy  # Bounded checking is faster
```

### "Type error in specification"

The USL type checker will report line and column numbers:

```
Error: TypeMismatch at line 5, column 10
  Expected: Bool
  Found: Int
```

Check that:
- Comparison operands have the same type
- Boolean operators (`and`, `or`, `implies`) have boolean operands
- Arithmetic operators have numeric operands

### "No proof suggestions"

```bash
# Enable learning mode to build corpus
dashprove verify spec.usl --learn

# Search corpus for similar proofs
dashprove search "your property description"
```

## Next Steps

- **[USL Specification](USL_SPECIFICATION.md)**: Complete language reference
- **[Design Document](DESIGN.md)**: Architecture and internals
- **[API Reference](API_REFERENCE.md)**: Server API documentation
- **[Backend Guide](BACKEND_GUIDE.md)**: Adding new backends
- **[Agent Verification Guide](AGENT_VERIFICATION_GUIDE.md)**: Verify AI agent systems

## Getting Help

- Run `dashprove help <command>` for command-specific help
- Run `dashprove topics <topic>` for detailed topic guides
- File issues at [github.com/dropbox/dMATH/dashprove/issues](https://github.com/dropbox/dMATH/dashprove/issues)
