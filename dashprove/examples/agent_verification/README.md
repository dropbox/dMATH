# Agent Verification Examples

This directory contains examples demonstrating DashProve's agent verification capabilities.

## Examples

| Example | Description | Crate |
|---------|-------------|-------|
| `bisimulation/` | Behavioral equivalence checking | dashprove-bisim |
| `runtime_monitor/` | Runtime invariant monitoring | dashprove-monitor |
| `async_verification/` | Async/concurrent verification | dashprove-async |
| `semantic_check/` | Semantic similarity verification | dashprove-semantic |
| `mbt_generation/` | Model-based test generation | dashprove-mbt |
| `miri_check/` | MIRI undefined behavior detection | dashprove-miri |

## Quick Start

```bash
# Run all agent verification examples
cargo run --example bisim_basic
cargo run --example monitor_basic
cargo run --example async_trace
cargo run --example semantic_basic
cargo run --example mbt_basic
cargo run --example miri_basic

# Or via CLI
dashprove bisim --help
dashprove verify-trace --help
dashprove semantic --help
dashprove mbt --help
dashprove miri --help
```

## Prerequisites

- Rust nightly (for MIRI examples)
- TLC (for TLA+ trace verification)

## File Structure

```
agent_verification/
├── README.md
├── bisimulation/
│   ├── oracle_trace.json      # Reference implementation trace
│   ├── subject_trace.json     # Implementation under test trace
│   └── bisim_spec.usl         # Bisimulation specification
├── runtime_monitor/
│   ├── agent_spec.usl         # Agent specification with invariants
│   └── execution_trace.json   # Recorded execution trace
├── async_verification/
│   ├── state_machine.tla      # TLA+ async state machine spec
│   └── async_trace.json       # Async execution trace
├── semantic_check/
│   ├── expected_output.txt    # Expected LLM output
│   └── actual_output.txt      # Actual LLM output
├── mbt_generation/
│   └── agent_model.tla        # TLA+ model for test generation
└── miri_check/
    └── unsafe_code.rs         # Code to check with MIRI
```
