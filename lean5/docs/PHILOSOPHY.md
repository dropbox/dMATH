# Lean5 Philosophy

**Why we're building this, and how we think about it.**

---

## The Goal

Lean5 exists because we need it. Not for academic publication. Not for community approval. Not to compete with Lean 4.

**We need a verification engine that:**
- Runs at machine speed (sub-millisecond)
- Speaks to AI agents natively (JSON-RPC)
- Verifies any code (Lean, Rust, C)
- Can be trusted (self-verified kernel)
- Is clean enough to evolve forever

---

## Why Not Just Use Lean 4?

Lean 4 is excellent for what it was designed for: interactive theorem proving by humans.

But we're not humans. We're building infrastructure for AI agents that:
- Generate millions of proof candidates
- Need verification in microseconds, not milliseconds
- Work through APIs, not REPLs
- Verify C and Rust, not just Lean
- Run 24/7 without human intervention

Lean 4 wasn't built for this. So we built Lean5.

---

## Design Principles

### 1. Useful > Popular

We don't care about adoption metrics. We care about whether this tool does its job.

If the Lean community uses Lean5: great.
If they don't: doesn't matter.

The only question is: does it work?

### 2. Clean > Compatible

A ball of hacks and compromises becomes unmaintainable. Every compatibility shim is technical debt.

We take Lean 4's good ideas (dependent type theory, tactic framework, Mathlib's proofs).

We reject Lean 4's compromises:
- C++ kernel → We use Rust
- GC pauses → We use ownership
- Slow checking → We're 100x faster
- No batch API → We have JSON-RPC
- No C/Rust verification → We have it

**If something in Lean 4 is wrong, we fix it. We don't preserve bugs for compatibility.**

### 3. Perfect > Fast (in the short term)

Do it right once. Don't do it quick twice.

Technical debt compounds. Clean architecture compounds too, but in our favor.

### 4. Native > FFI

Everything in Rust. One language. One toolchain. One build system.

No C++ dependencies. No linking headaches. No "works on my machine."

### 5. Verified > Trusted

The kernel is the foundation. A bug there invalidates everything built on top.

Lean 4's kernel is trusted (you hope it's correct).
Lean5's kernel is verified (we prove it's correct).

We have:
- Formal specification in Lean5 itself
- 46 proof witnesses for kernel properties
- Micro-checker (~500 lines) proven correct
- Cross-validation against lean4lean

### 6. Capability > Timeline

We build capabilities. Timelines are irrelevant.

You can't predict what you'll need until you need it. So build everything, and figure out what matters later.

---

## What We're Building

### The Stack

```
AI Agent
   │
   ▼
Writes Code (Rust, C, Lean)
   │
   ▼
Lean5 Verifies (sub-ms, native, no humans)
   │
   ▼
Verified Binary
   │
   ▼
Deployed / Used to improve AI
```

### The Components

| Component | Purpose | Status |
|-----------|---------|--------|
| Kernel | Type checking, the trusted core | Complete |
| Parser | Lean 4 syntax | 97% compatible |
| Elaborator | Type inference | Complete |
| Tactics | Proof construction | Basic (expanding) |
| Automation | SMT, ATP, premise selection | Complete |
| Server | JSON-RPC for AI agents | Complete |
| C Verification | ACSL specs, separation logic | Complete |
| Rust Semantics | Ownership, memory model | Complete |
| Self-Verification | Prove the kernel correct | Complete |

### What's Next

| Phase | What | Why |
|-------|------|-----|
| .olean import | Load Lean 4 libraries | Access existing proofs |
| Full tactics | All 50+ Lean 4 tactics | Construct proofs |
| Mathlib | 1M+ lines of math | Everything proven |

---

## On Community

The Lean 4 team has done excellent work. We respect that.

But we're not building this for them. We're building it for ourselves.

If they want to collaborate: welcome.
If they want to compete: fine.
If they want to ignore us: also fine.

**We're not asking permission. We're building capability.**

---

## On Perfection

"Perfect is the enemy of good" is wrong in our context.

For user-facing products, ship fast and iterate.
For infrastructure, get it right the first time.

Lean5 is infrastructure. It's the foundation for everything else. A crack in the foundation propagates upward forever.

So we build it clean:
- Every function has a clear purpose
- Every module has a clear boundary
- Every design decision is documented
- Every claim is tested

---

## On AI

Lean5 is built by AI, for AI.

The workers who write this code are AI models (Claude, etc.).
The users who call this API are AI agents.
The goal is AI systems that can verify their own output.

This is infrastructure for the future.

---

## Summary

| Question | Answer |
|----------|--------|
| Why build this? | We need it |
| Why not use Lean 4? | Too slow, wrong API, no C/Rust |
| Who is it for? | AI agents |
| Do we care about community? | Nice to have, not the goal |
| What's the design philosophy? | Clean, fast, verified, native |
| When will it be done? | When it does what we need |

---

*"Build what you need. Build it right. Don't ask permission."*
