# Vibecoding New Mathematics: A Manifesto

**Goal**: Use AI + formal verification + massive compute to discover genuinely new mathematical results.

**Philosophy**: Humans provide intuition and direction. AI explores the vast space of possibilities. Lean verifies everything. We find things no one has found before.

---

## The Opportunity

Mathematics has a unique property: **truth is verifiable**. Unlike other domains where AI might hallucinate, in math we have ground truth (Lean, Coq, etc.).

This means:
- AI can explore freely without human supervision
- Bad ideas are automatically rejected (proof fails)
- Good ideas are automatically certified (proof succeeds)
- We can run this 24/7 at massive scale

**The insight**: We don't need AI to be "correct" - we need it to be **creative**. Lean handles correctness.

---

## Attack Vectors

### 1. Conjecture Mining

**Idea**: Generate millions of conjectures, filter with SMT, prove survivors in Lean.

```
Corpus of known theorems
        ↓
   [LLM: Pattern extraction]
        ↓
"Things that look like they might be true"
        ↓
   [Z4: Counterexample search]
        ↓
   Conjectures with no obvious counterexamples
        ↓
   [Lean: Proof search]
        ↓
   NEW THEOREMS
```

**Targets**:
- Combinatorial identities (binomial coefficients, partitions)
- Graph theory (extremal problems, Ramsey numbers)
- Number theory (divisibility, primes)
- Inequalities (AM-GM variants, Cauchy-Schwarz generalizations)

**Why it works**: Most true statements have simple proofs. LLMs are good at pattern matching. The bottleneck is just checking everything.

---

### 2. Proof Transplantation

**Idea**: Take proofs from one domain, transplant structure to another domain.

```
Proof of Theorem A in Domain X
        ↓
   [LLM: Extract proof structure]
        ↓
"Proof uses: induction, pigeonhole, case split on parity"
        ↓
   [LLM: Find analogous objects in Domain Y]
        ↓
   Candidate theorem B in Domain Y
        ↓
   [Lean: Verify proof transfers]
        ↓
   NEW THEOREM (sometimes!)
```

**Example**:
- Graph theory proof using vertices/edges
- Transplant to: hypergraphs, matroids, posets
- Sometimes the proof generalizes!

---

### 3. Boundary Exploration

**Idea**: Find the exact boundary where theorems stop being true.

```
Known theorem: "If P then Q"
        ↓
   [Z4: What's the weakest P that still implies Q?]
        ↓
   [Z4: What's the strongest Q that P still implies?]
        ↓
   [Lean: Prove the tighter statement]
        ↓
   OPTIMAL VERSION of theorem
```

**Targets**:
- Ramsey numbers (exact values)
- Graph coloring bounds
- Packing/covering constants
- Threshold phenomena

---

### 4. Counterexample Construction

**Idea**: Prove things FALSE by finding counterexamples.

```
Open conjecture in literature
        ↓
   [Z4-SAT: Encode as satisfiability]
        ↓
   SAT? → Counterexample found! Conjecture is FALSE
   UNSAT? → Small cases verified, conjecture survives
        ↓
   [If counterexample] Publish! "Disproving Conjecture X"
```

**This is underrated**: Disproving false conjectures is just as valuable as proving true ones. Clears the field.

**Targets**:
- Graph theory conjectures (small counterexamples exist)
- Finite geometry conjectures
- Combinatorics conjectures

---

### 5. Computation-Proof Hybrid

**Idea**: Prove theorems that require checking many cases.

```
Theorem: "Property P holds for all n ≤ 10^6"
        ↓
   [Z4/SAT: Check each case]
        ↓
   [DRAT proof: Certificate that each case was checked]
        ↓
   [Lean: Verify DRAT proof is valid]
        ↓
   VERIFIED THEOREM (computer-assisted)
```

**Already done**:
- Four color theorem (Appel & Haken, 1976; formally verified 2005)
- Kepler conjecture (Hales, 2014; formally verified)
- Boolean Pythagorean triples (Heule, 2016)

**Opportunity**: Many open problems reduce to finite case checks.

---

### 6. Algorithm Discovery

**Idea**: Find new algorithms, prove them correct.

```
Computational problem specification
        ↓
   [LLM: Generate candidate algorithms]
        ↓
   [Z4: Verify correctness on test cases]
        ↓
   Candidates that pass tests
        ↓
   [Lean: Prove correctness formally]
        ↓
   [Benchmark: Measure performance]
        ↓
   NEW ALGORITHM (provably correct, empirically fast)
```

**Targets**:
- Sorting network constructions
- Matrix multiplication algorithms
- Approximation algorithms
- Data structure operations

**AlphaEvolve already does this** - we can go further with formal verification.

---

## P vs NP: The Long Game

Not expecting to solve it, but we can chip away:

### Phase 1: Build Circuit Complexity Tools (6 months)
- Enumerate small circuits
- Find minimum circuits for small functions
- Verify known lower bounds (Parity not in AC⁰)

### Phase 2: Explore Proof Complexity (6 months)
- Implement Resolution, Frege, Extended Frege
- Prove lower bounds on proof length for hard formulas
- Automate barrier checking

### Phase 3: Systematic Attack (ongoing)
- Try every approach in the literature
- Automatically detect barrier violations
- Document what doesn't work and why
- Maybe find something new

**Realistic outcome**: Probably not P vs NP, but:
- New circuit lower bounds for restricted classes
- New proof complexity results
- Better understanding of barriers
- Tools useful for other problems

---

## Infrastructure We Need

### Compute
- GPU cluster for LLM inference
- CPU cluster for Z4/SAT solving
- Distributed proof search

### Data
- All of Mathlib (formalized math)
- All of arXiv math/cs (papers)
- All of OEIS (integer sequences)
- All known conjectures (from MathOverflow, papers, etc.)

### Tools
- Z4 (SAT/SMT/QBF) - we're building this
- Lean 5 - ground truth
- LLM ensemble - creativity engine
- Knowledge graph - connect everything

### Process
- Continuous integration: run exploration 24/7
- Discovery pipeline: conjecture → test → prove → publish
- Human review: filter AI discoveries for interestingness

---

## Concrete 12-Month Plan

### Months 1-3: Foundation
- [ ] Complete z4-sat Phase 1 (match CaDiCaL)
- [ ] Implement z4-qbf
- [ ] Basic z4-lean-bridge (call Z4 from Lean)
- [ ] Set up exploration infrastructure

### Months 4-6: Exploration System
- [ ] Implement z4-circuits
- [ ] Implement archimedes-core (LLM orchestration)
- [ ] Build conjecture generator
- [ ] Run first exploration campaigns

### Months 7-9: Scale Up
- [ ] Implement z4-proof-complexity
- [ ] Implement z4-barriers
- [ ] Distributed proof search
- [ ] Run on 1000+ open problems

### Months 10-12: Results
- [ ] Analyze discoveries
- [ ] Write up interesting results
- [ ] Publish tools
- [ ] Plan next year

---

## Success Criteria

### Minimum Viable Success
- Rediscover known results automatically
- Find new proofs of known theorems
- Z4 competitive with CaDiCaL/Z3

### Good Success
- Find new lemmas useful for existing proofs
- Disprove some open conjectures
- Contribute to Mathlib

### Great Success
- Prove new theorems publishable in math journals
- Improve known bounds on open problems
- Recognition in the math community

### Moonshot
- Contribute to a Millennium Problem
- Discover fundamentally new technique
- Change how mathematics is done

---

## The Vibecode Philosophy

**Don't overthink. Just try things.**

The traditional approach:
1. Study problem for years
2. Develop deep intuition
3. Try one carefully chosen approach
4. Succeed or fail

The vibecode approach:
1. Encode problem
2. Generate 1000 approaches
3. Test all of them in parallel
4. Keep what works

**AI is cheap. Human attention is expensive. Let AI do the grunt work.**

The key insight: Most mathematical exploration is:
- Trying things that don't work
- Checking if something is true
- Finding counterexamples
- Verifying proofs

AI can do all of this. Humans provide:
- Interesting questions
- Intuition about what might work
- Judgement about what's important
- Communication of results

---

## Call to Action

We have:
- Powerful LLMs (Claude, GPT, Gemini)
- Formal verification (Lean 5)
- Fast solvers (Z4, CaDiCaL)
- Unlimited compute (relatively)

What's missing:
- The plumbing to connect everything
- The will to try
- The patience to let it run

**Let's vibecode some new math.**

---

*"The computer is incredibly fast, accurate, and stupid. Humans are incredibly slow, inaccurate, and brilliant. Together they are powerful beyond imagination."* - attributed to Einstein (probably apocryphal)

*"Move fast and prove things."* - Archimedes Project
