# Introducing DashProve: Building an AI-Native Verification Platform

*By Andrew Yates*

Formal verification is one of the most powerful techniques in software engineering. It can mathematically prove that code is correct, that protocols are secure, that concurrent systems are free from race conditions. But it's also one of the least accessible. Each verification tool has its own language, its own paradigm, its own learning curve measured in years.

I'm building DashProve to change that.

## The Problem We're Solving

Right now, if you want to formally verify software, you face a fragmented landscape:

- **LEAN 4** for mathematical proofs and algorithm correctness
- **TLA+** for distributed systems and temporal properties
- **Kani** for Rust memory safety and undefined behavior
- **Alloy** for data structure invariants
- **Z3** and **CVC5** for satisfiability and constraint solving
- **Tamarin** and **ProVerif** for security protocols
- **Marabou** and **alpha-beta-CROWN** for neural network verification

Each tool is excellent at what it does. But learning one doesn't help you learn another. The concepts transfer, but the syntax, tooling, and mental models don't. This creates a paradox: the people who would benefit most from verification (those building complex, safety-critical systems) are the ones least likely to have time to become experts in multiple verification paradigms.

## The DashProve Approach

DashProve takes a different approach. Instead of choosing one tool, we abstract over all of them.

**One language, many backends.** You write specifications in USL (Unified Specification Language), a declarative language designed to express properties naturally. DashProve compiles your spec to the appropriate backend and returns structured results.

```
// USL: One specification
theorem list_append_length {
    forall xs: List<Int>, ys: List<Int> .
        length(append(xs, ys)) == length(xs) + length(ys)
}

// Compiles to LEAN 4 for proof
// Or Kani for bounded verification in Rust
// Or Alloy for bounded model checking
```

**Intelligent dispatch.** Different properties are better suited to different tools. A temporal property like "eventually every request gets a response" belongs in TLA+. A memory safety property like "this pointer is never null" belongs in Kani. DashProve's dispatcher analyzes your property and routes it to the best backend automatically.

**AI-native output.** This is the part I'm most excited about. Traditional verification tools output human-readable text that needs to be parsed and interpreted. DashProve returns structured data: proof objects, counterexamples, tactic suggestions, confidence scores. This isn't just about clean APIs—it's about enabling AI agents to interact with verification tools programmatically.

## Why AI-Native Matters

We're entering an era where AI writes significant amounts of code. But AI-generated code has a trust problem. How do you know the code is correct? How do you know it handles edge cases? How do you know it won't introduce security vulnerabilities?

Traditional testing helps, but it can only show the presence of bugs, not their absence. Formal verification can prove correctness, but it's too specialized for most workflows.

DashProve bridges this gap. An AI coding assistant can:

1. Generate code
2. Generate a USL specification of what the code should do
3. Call DashProve to verify the code matches the spec
4. If verification fails, use the structured counterexample to fix the code
5. Iterate until the proof succeeds

The structured output is critical here. When DashProve returns a counterexample, it's not a wall of text—it's a JSON object with the exact failing input, the trace of execution, and suggestions for fixing the issue. An AI can consume this directly and act on it.

## What We're Building

DashProve is currently at ~480,000 lines of Rust across 27 crates:

- **dashprove-usl**: Parser and type checker for the specification language
- **dashprove-backends**: Integrations with 180+ verification backends
- **dashprove-dispatcher**: Intelligent routing to the best backend per property
- **dashprove-learning**: Proof corpus that learns from successful verifications
- **dashprove-ai**: LLM integration for proof sketch elaboration
- **dashprove-monitor**: Runtime monitor synthesis from specifications

The test suite has 7,700+ tests passing, including 3,894 Kani proofs that verify our own implementation.

### Backend Coverage

We're not just wrapping command-line tools. Each backend integration parses output into structured types, handles error cases, and provides diagnostics. Current backends span:

**Theorem Provers**: LEAN 4, Coq, Isabelle, Agda, Idris, F*, HOL4, ACL2

**Model Checkers**: TLA+ (TLC), Apalache, Alloy, NuSMV, SPIN, CBMC

**Rust Verification**: Kani, Verus, Creusot, Prusti, MIRI, Loom

**Neural Network Verification**: Marabou, alpha-beta-CROWN, ERAN, DNNV, NNEnum

**Security**: Tamarin, ProVerif, Verifpal

**Fuzzers & Sanitizers**: AFL, libFuzzer, AddressSanitizer, ThreadSanitizer

**Static Analysis**: Infer, Frama-C, CPAchecker, MIRAI

## What We're Learning

Building DashProve has been an education in the state of formal verification tooling. Some observations:

**The interfaces are the hard part.** The verification algorithms are well-understood. What's hard is parsing 47 different output formats, handling tool-specific error messages, and translating between incompatible type systems.

**Proof reuse is underexplored.** When you prove something in LEAN, that proof is a first-class object you can inspect, store, and reference. We're building a proof corpus that indexes successful proofs and uses them to guide future verification attempts. Early results suggest this significantly improves success rates.

**Compilation is cheap, verification is expensive.** Compiling USL to LEAN 4 takes microseconds. Running LEAN 4 takes milliseconds to hours depending on the proof. This asymmetry suggests a strategy: compile to multiple backends cheaply, run whichever returns first.

**Bounded and unbounded verification complement each other.** Bounded model checkers (Alloy, TLC) find bugs fast. Unbounded provers (LEAN, Coq) prove correctness forever. Running both in parallel gives you quick feedback on likely bugs while working toward full proofs.

## What's Next

This is very much a work in progress. Some areas we're actively developing:

1. **Proof sketch elaboration**: Given a high-level proof strategy, automatically fill in the details
2. **Counterexample-guided refinement**: When verification fails, automatically suggest specification or code changes
3. **Multi-backend consensus**: Run the same property on multiple backends and combine results for higher confidence
4. **Learning from failures**: When a proof attempt fails, record what didn't work to avoid repeating mistakes

## Try It

DashProve is open source. You can:

```bash
# Clone and build
git clone https://github.com/dropbox/dMATH/dashprove
cd dashprove
cargo build --release

# Verify a specification
cargo run -p dashprove-cli -- verify examples/usl/basic.usl

# Start the REST API server
cargo run -p dashprove-server -- --port 3000

# Use from Rust
use dashprove::{DashProve, DashProveConfig};
let client = DashProve::new(DashProveConfig::default());
let result = client.verify("theorem t { forall x: Bool . x or not x }").await?;
```

If you're working on AI coding assistants, formal methods, or just want to make verification more accessible, I'd love to hear from you. This is early days, and we're learning as we go.

---

*DashProve is part of a larger effort to build AI systems that can reason about and verify their own output. If you're interested in the intersection of AI and formal methods, follow along as we figure this out.*
