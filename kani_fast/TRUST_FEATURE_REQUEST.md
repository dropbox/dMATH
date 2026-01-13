# Kani Fast Feature Request from tRust

**From:** tRust (rustc fork with integrated verification)
**To:** Kani Fast team (git@github.com:dropbox/dMATH/kani_fast.git)
**Date:** 2024-12-31

---

## What is tRust?

tRust is a fork of rustc that integrates formal verification into the compiler. Verification happens during compilation—not as a separate tool. If it compiles, it's proven correct.

**Our thesis:** Proofs are context compression. Verified code becomes a trusted black box. AI loads specs instead of implementations.

---

## Why Kani Fast Matters to Us

tRust's primary backend is Z4 (SMT solver). Z4 is fast and handles most verification conditions. But Z4 has limits:

- Complex heap structures
- Deep recursion
- Nonlinear arithmetic
- Bit-level operations

When Z4 says "unknown," we need a fallback. Kani's bounded model checking explores concrete executions—it finds bugs Z4 misses.

**The problem with current Kani:** Too slow for interactive use. Minutes per function. Users won't wait.

**What we need:** Kani Fast. Same power, 10x speed.

---

## Feature Requests (Priority Order)

### 1. 10x Faster (CRITICAL)

**Current Kani:**
```
$ kani my_function.rs
Checking harness...
[==============================] 100%
Time: 47 seconds

VERIFICATION SUCCESSFUL
```

**Target:**
```
$ tRust verify my_function.rs
Z4: unknown (heap aliasing)
Kani Fast: checking to depth 100...
Time: 4.2 seconds

VERIFIED (bounded to depth 100)
```

**How to get there:**
- Better symbolic execution engine (not CBMC?)
- Aggressive path merging
- Function summaries (don't re-analyze called functions)
- Parallelization

---

### 2. Incremental Analysis (CRITICAL)

**Current:** Every run re-analyzes everything

**Requested:** Cache and reuse

```
First run:
  Analyzing foo... 3.2s
  Analyzing bar... 2.1s (calls foo)
  Analyzing baz... 4.5s (calls bar)
  Total: 9.8s

Second run (only baz changed):
  foo: cached
  bar: cached (depends on foo, unchanged)
  Analyzing baz... 4.5s (changed)
  Total: 4.5s
```

**What to cache:**
- Symbolic execution summaries per function
- Path condition results
- Memory layouts

**Invalidation:**
- Function body changed → invalidate
- Function spec changed → invalidate
- Dependency changed → invalidate transitively

---

### 3. Z4 Handoff Protocol (HIGH)

When Z4 fails, it should tell Kani Fast *why*:

```rust
enum Z4Result {
    Proven { certificate },
    Counterexample { values },
    Unknown {
        reason: UnknownReason,
        subproblem: Option<Predicate>,  // The hard part
        partial_proof: Option<PartialProof>,  // What Z4 already proved
    }
}

enum UnknownReason {
    HeapAliasing,      // Pointer reasoning too complex
    NonlinearArith,    // x * y > z type stuff
    DeepRecursion,     // Unrolling explodes
    Timeout,           // Ran out of time
    QuantifierHeavy,   // Too many forall/exists
}
```

Kani Fast focuses on the `subproblem`, not the whole VC.

**Benefit:** Z4 proves the easy parts, Kani Fast handles the hard parts.

---

### 4. Parallel Exploration (HIGH)

**Current Kani:** Single-threaded CBMC

**Requested:** Use all available cores

```
$ tRust verify --jobs 8 large_module.rs

Kani Fast: 8 workers
  Worker 1: foo (checking)
  Worker 2: bar (checking)
  Worker 3: baz (cached)
  ...

Completed: 47 functions, 12.3 seconds (8 cores)
Sequential estimate: 89 seconds
Speedup: 7.2x
```

**Parallelization strategies:**
- Function-level (embarrassingly parallel)
- Path-level within function (work stealing)
- Distributed (across machines for large codebases)

---

### 5. Bounds Inference (MEDIUM)

**Current:** User specifies unwind bounds

```rust
#[kani::unwind(10)]  // Magic number, hope it's enough
fn my_loop() { ... }
```

**Requested:** Infer from specs

```rust
#[requires(n < 100)]
#[invariant(i <= n)]
#[variant(n - i)]
fn countdown(n: i32) {
    let mut i = 0;
    while i < n { i += 1; }
}
// Kani Fast infers: unwind = 100 (from requires + variant)
```

**Inference rules:**
- Loop variant + precondition → bound
- Array length in precondition → index bound
- Recursive depth from decreases measure

---

### 6. Result Classification (MEDIUM)

Kani Fast results should be clear about what was proven:

```rust
enum KaniResult {
    // Actually proven (found no bugs up to bound, and bound is sufficient)
    Verified {
        depth: u64,
        bound_sufficient: bool,  // true if we hit fixpoint
    },

    // Bug found
    Counterexample {
        trace: ExecutionTrace,
        depth: u64,
        rust_test: String,  // Generated test to reproduce
    },

    // Checked but might miss bugs beyond bound
    BoundedCheck {
        depth: u64,
        coverage: f64,  // Estimated state coverage
        suggestion: Option<String>,  // "Try increasing bound"
    },

    // Gave up
    ResourceExhausted {
        reason: String,
        partial_coverage: f64,
    },
}
```

tRust reports this clearly to users:
```
foo: VERIFIED (proven correct)
bar: CHECKED to depth 100 (no bugs found, 94% coverage)
baz: BUG FOUND at depth 7 (see generated test)
```

---

## Integration Protocol

### Input from tRust
```rust
struct KaniQuery {
    function: MIR,  // Already lowered
    spec: FunctionSpec,

    // Context from Z4
    z4_result: Option<Z4Unknown>,
    already_proven: Vec<Predicate>,  // Z4's partial work

    // Bounds (inferred or user-specified)
    max_depth: Option<u64>,
    timeout: Duration,
}
```

### Output to tRust
```rust
struct KaniResponse {
    result: KaniResult,
    time: Duration,
    memory_used: usize,

    // For caching
    cache_key: Hash,
    dependencies: Vec<FunctionId>,
}
```

---

## Performance Targets

| Scenario | Current Kani | Kani Fast Target |
|----------|--------------|------------------|
| Simple function, no loops | 5-10s | <1s |
| Loop with bound 10 | 30-60s | <5s |
| Loop with bound 100 | 5-10min | <30s |
| Incremental re-check | N/A | <1s |
| Parallel (8 cores) | N/A | 6-8x speedup |

---

## Questions for Kani Fast Team

1. Are you keeping CBMC or replacing the backend?
2. What's the main bottleneck in current Kani? (SAT solving? Symbolic execution? Translation?)
3. Can function summaries work across crate boundaries?
4. Incremental: file-level or function-level granularity?
5. How do you handle `unsafe` blocks?

---

## Example Integration

What we want:

```rust
#[requires(arr.len() > 0)]
#[ensures(result < arr.len())]
#[ensures(forall |i| arr[result] <= arr[i])]
fn find_min_index(arr: &[i32]) -> usize {
    let mut min_idx = 0;
    for i in 1..arr.len() {
        if arr[i] < arr[min_idx] {
            min_idx = i;
        }
    }
    min_idx
}

// tRust compilation:
// 1. Generate VC from spec
// 2. Z4 attempts: "unknown - array reasoning complex"
// 3. Kani Fast attempts: depth=arr.len(), max 1000
// 4. Kani Fast: "verified to depth 1000, bound inferred sufficient"
// 5. Compilation succeeds
```

---

## Contact

tRust issues: https://github.com/dropbox/tRust/issues
This request: `reports/main/feature_requests/KANI_FAST_FEATURE_REQUEST.md`
