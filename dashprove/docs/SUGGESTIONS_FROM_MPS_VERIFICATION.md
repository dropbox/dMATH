# DashProve Enhancement Suggestions from MPS Verification Project

**Date:** 2025-12-19
**Source:** metal_mps_parallel formal verification effort (N=1305)
**Author:** Worker AI

---

## Executive Summary

After building a comprehensive formal verification platform for PyTorch's MPS backend (14.7M+ TLA+ states, 3,856 CBMC assertions, 6 Iris/Coq modules, 14 structural checks), we identified capabilities that would significantly enhance DashProve.

**Key Finding:** The most impactful gap was **platform API constraints** - external API state machines that formal tools couldn't verify. A critical SIGABRT crash (N=1305) was caused by violating Metal's `addCompletedHandler` constraint, which no formal tool caught because it's an external API contract.

---

## Suggested Enhancements

### 1. Platform API Constraint Module (HIGH PRIORITY)

**Problem:** Formal verification tools (TLA+, CBMC, Coq) cannot verify external API contracts. They assume the environment behaves correctly.

**Real Example:** Metal's `addCompletedHandler:` must be called BEFORE `commit`. Our TLA+ spec and CBMC harnesses couldn't encode this because it's Apple's API contract.

**Suggestion:** Add a `dashprove-platform-apis` crate that:

```rust
// Platform API definition in USL
platform_api Metal {
    state MTLCommandBuffer {
        enum Status { Created, Encoding, Committed, Completed }

        transition commit() {
            requires { status in {Created, Encoding} }
            ensures { status == Committed }
        }

        transition addCompletedHandler(block) {
            requires { status in {Created, Encoding} }  // CRITICAL
            ensures { block will be called after completion }
        }
    }
}
```

**Implementation:**
1. Pre-defined platform API catalogs (Metal, CUDA, Vulkan, POSIX, etc.)
2. USL syntax for platform API state machines
3. Code generator for static checkers
4. Integration with main dispatcher for combined verification

---

### 2. Liveness Property Support (HIGH PRIORITY)

**Problem:** All our TLA+ specs verified safety properties only. We had no proof of:
- Eventual progress (requests complete)
- Starvation freedom (no thread waits forever)
- Fairness (bounded waiting)

**Suggestion:** Enhance TLA+ backend with explicit liveness support:

```
// USL temporal properties
temporal bounded_wait {
    forall thread in threads .
        (thread.waiting implies eventually(thread.running))
}

temporal starvation_free {
    always(forall slot in slots . eventually(slot.free))
}
```

**Implementation:**
1. Add temporal operators to USL grammar
2. Generate TLA+ temporal properties with fairness assumptions
3. Add liveness checking mode to TLC runner
4. Report counterexample traces for liveness violations

---

### 3. Unbounded Verification via Apalache (MEDIUM PRIORITY)

**Problem:** TLC explores bounded state space (N=3 threads, N=4 buffers). We couldn't prove properties hold for arbitrary N.

**Suggestion:** Add Apalache backend for symbolic TLA+ verification:

```rust
// In dispatcher
if property.requires_unbounded() {
    dispatch_to(Backend::Apalache);
}
```

**Implementation:**
1. Add Apalache type annotation generator to TLA+ backend
2. Automatic type inference for common patterns
3. Hybrid approach: TLC for quick checks, Apalache for unbounded proofs

---

### 4. Refinement Mapping Verification (MEDIUM PRIORITY)

**Problem:** We have TLA+ specs and C++ implementations, but no proof they correspond.

**Suggestion:** Add refinement mapping support:

```
// USL refinement
refinement MPSStreamPoolImpl refines MPSStreamPoolSpec {
    mapping {
        spec.streams <-> impl.m_streams
        spec.bindings <-> impl.thread_local_slots

        invariant: |impl.m_streams| == MAX_STREAMS
    }

    action acquire_stream {
        spec: AcquireStream
        impl: MPSStreamPool::acquireStream()
    }
}
```

**Implementation:**
1. USL syntax for refinement mappings
2. Generate CBMC assertions from refinement predicates
3. Automated correspondence checking

---

### 5. Cross-Component Composition (MEDIUM PRIORITY)

**Problem:** We verified MPSStream, MPSAllocator, MPSEvent separately. No proof they compose safely.

**Suggestion:** Add composition verification:

```
// USL composition
composed_system MPSSystem {
    components: [MPSStreamPool, MPSAllocator, MPSEvent]

    shared_state {
        device: MTLDevice
        command_queue: MTLCommandQueue
    }

    verify {
        no_distributed_deadlock
        global_memory_safety
    }
}
```

**Implementation:**
1. Generate composed TLA+ specification from components
2. Track shared state across components
3. Verify global properties (deadlock freedom, liveness)

---

### 6. Memory Ordering Verification (MEDIUM PRIORITY)

**Problem:** TSA verifies lock discipline but not memory ordering (acquire/release semantics).

**Suggestion:** Add memory model verification:

```
// USL memory model annotations
atomic_operation release_stream {
    memory_order: release
    synchronizes_with: acquire_stream
}

verify memory_model {
    all atomics have correct ordering
    no data races under C++11 memory model
}
```

**Implementation:**
1. Integrate CDSChecker or GenMC backend
2. USL syntax for memory ordering annotations
3. Automatic verification of happens-before relationships

---

### 7. Callback Ordering Model (LOW PRIORITY)

**Problem:** Callback lifetime is verified, but ordering guarantees are not modeled.

**Suggestion:**
```
// USL callback model
callback_model MTLCompletedHandler {
    invocation_order: unspecified  // or: fifo, priority
    thread_affinity: specified_queue

    verify {
        all_callbacks_invoked_before_dealloc
    }
}
```

---

### 8. Pre-commit Hook Generator (LOW PRIORITY)

**Problem:** We manually created pre-commit hooks. Should be automated.

**Suggestion:** Add hook generator to CLI:
```bash
dashprove generate-hooks --type git-precommit --checks "AF.*, ST.*"
```

---

### 9. Proof Learning Corpus Priorities

Based on our experience, the proof learning system should prioritize:

1. **Concurrent data structure proofs** - Most valuable, hardest to write
2. **Lock-free algorithm proofs** - High reuse potential
3. **API contract proofs** - Platform-specific but transferable patterns
4. **Callback safety proofs** - Common pattern in async code

---

### 10. AI-Native Features We Needed

Things we wished the tools could do:

1. **"Fix this crash" mode** - Given a stack trace, identify likely constraints violated
2. **"Verify this diff" mode** - Check if patch maintains verified properties
3. **"What's not proven" mode** - List gaps in current verification
4. **"Generate test from counterexample"** - Convert TLC counterexample to executable test

---

## Concrete Code Contributions

We've created reusable artifacts that could be integrated:

1. **Metal API Constraint Catalog** (`metal_api_catalog.py`)
   - 20+ constraints with patterns and severity
   - Could be ported to USL platform API format

2. **API State Machine Checker** (`platform_api_checker.py`)
   - State tracking through code flow
   - Pattern-based detection
   - Could inform static analysis backend

3. **Structural Check Framework** (`structural_checks.sh`)
   - 14 pattern-based conformance checks
   - Could inform pre-commit generator

---

## Summary Table

| Enhancement | Priority | Effort | Impact |
|-------------|----------|--------|--------|
| Platform API Constraints | HIGH | 3-5 weeks | Prevents critical bugs |
| Liveness Properties | HIGH | 2-3 weeks | Proves termination |
| Apalache Backend | MEDIUM | 2 weeks | Unbounded proofs |
| Refinement Mapping | MEDIUM | 3-4 weeks | Spec-code correspondence |
| Composition Verification | MEDIUM | 4-5 weeks | System-level proofs |
| Memory Ordering | MEDIUM | 4-6 weeks | Prevents subtle races |
| Callback Ordering | LOW | 1-2 weeks | Documents guarantees |
| Hook Generator | LOW | 1 week | Developer experience |

---

## Contact

These suggestions come from the `metal_mps_parallel` project verification effort. The full analysis is available at:
- `reports/main/DASHPROOF_GAP_ANALYSIS.md`
- `reports/main/BENEFITS_AND_LIMITS_OF_STATIC_ANALYSIS.md`
- `mps-verify/checks/metal_api_catalog.py`

Happy to discuss any of these in detail.
