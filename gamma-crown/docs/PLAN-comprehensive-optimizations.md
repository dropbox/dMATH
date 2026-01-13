# PLAN: Comprehensive Optimization Audit

**Status:** IN PROGRESS (Phase 1 Complete)
**Created:** 2026-01-02
**Updated:** 2026-01-01
**Auditor:** Manager
**Goal:** Identify and prioritize all general software optimizations

## Progress (2026-01-01)

| Item | Status | Notes |
|------|--------|-------|
| P1: Inline hot paths | ✅ COMPLETE | 32→124 #[inline] annotations |
| M2: Pre-allocate vectors | ✅ COMPLETE | Added capacity to HashMaps/Vecs in network.rs |
| P2: Vec allocation in hot loop | ✅ COMPLETE | Removed Vec alloc in MatMul IBP (15% improvement) |
| P2: Parallel activation IBP | ✅ COMPLETE | par_for_each for GELU/Tanh/Sigmoid/Sin/Cos (2-3x speedup for large tensors) |
| P2: domain_clip parallelization | ✅ COMPLETE | Added parallel paths but with 1M threshold - operations are memory-bound, see benchmark |
| M1: Clone reduction (partial) | ✅ PARTIAL | collect_ibp_bounds last-iter + sampling loop reuse (1067→1065 static, ~100 allocs/call saved) |
| D1: High-risk unwrap fixes | ✅ PARTIAL | 8 HashMap lookups converted to expect() with safety comments (Worker #269) |
| M1: get_bounds_ref optimization | ✅ COMPLETE | Eliminated ~N BoundedTensor clones per propagation call (Worker #270) |
| M1: Cow<LinearBounds> propagate | ✅ COMPLETE | Return Cow for pass-through layers, avoiding clones (Worker #271) |
| M1: Arc<BoundedTensor> BabDomain | ✅ COMPLETE | Branch-and-bound domain cloning now O(N) Arc pointers vs O(N*M) tensor data (Worker #272) |

### M1 Clone Reduction Analysis (Worker #268)

**Analyzed:** 1067 clone calls across codebase

**Optimizations Applied:**
1. `collect_ibp_bounds`: Avoid clone on last iteration (saves 1 BoundedTensor per call)
2. Softmax/LayerNorm sampling loops: Reuse buffer via `.assign()` instead of clone (saves ~99 allocs per call)

### M1 get_bounds_ref Optimization (Worker #270)

**Problem:** `get_bounds()` method was cloning BoundedTensor on every lookup, but all callers only needed references.

**Solution:** Created `get_bounds_ref()` that returns `&BoundedTensor` instead of owned `BoundedTensor`.

**Impact:**
- Eliminates ~N BoundedTensor clones per propagation call (N = number of graph nodes)
- For Whisper-tiny (100+ nodes), this saves ~200MB of allocation per IBP pass
- Each BoundedTensor = 2 ArrayD<f32>, potentially megabytes each
- 17 call sites updated to use references instead of clones

**Files Changed:** `crates/gamma-propagate/src/network.rs`

**Major Opportunities Identified (require API changes):**
1. **Cow<LinearBounds> for propagate_linear** ✅ COMPLETE (Worker #271)
   - 16+ pass-through `Ok(bounds.clone())` cases in layers/mod.rs
   - LinearBounds contains 4 arrays, each potentially megabytes
   - Fix: Return `Cow<'_, LinearBounds>` - borrow for pass-through, own for modifications
   - Effort: 60+ function signature changes

2. **Arc<BoundedTensor> for BabDomain.layer_bounds** ✅ COMPLETE (Worker #272)
   - Branch-and-bound clones entire domain including all layer bounds
   - BoundedTensor = 2 ArrayD<f32>, potentially MB each
   - Fix: Use `Arc<BoundedTensor>` for shared ownership
   - **Implementation**: Changed `BabDomain.layer_bounds` from `Vec<BoundedTensor>` to `Vec<Arc<BoundedTensor>>`
   - When creating child domains, only clone Arc pointers (cheap!)
   - Only modified layers get new Arc allocations
   - Updated internal functions to use `&[Arc<BoundedTensor>]` and deref coercion
   - Replaced clones with references where possible (`as_ref()` pattern)

3. **String interning for node names** (MEDIUM IMPACT)
   - 30+ string clones per graph for node names
   - Fix: Use `Rc<str>` or string interner
   - Effort: Medium (GraphNode changes + 251 constructor usages)

4. **Cow<BoundedTensor> for pass-through input.clone()** (MEDIUM IMPACT)
   - 9 `return Ok(input.clone())` in network.rs
   - Fix: Return `Cow<'_, BoundedTensor>`
   - Effort: Medium (API change)

---

## Audit Summary

| Category | Issues Found | Priority Items |
|----------|-------------|----------------|
| Processing | 12 | 7 high-impact |
| Memory | 9 | 6 high-impact |
| Debug/Error/UX | 11 | 6 high-impact |
| **Total** | **32** | **19 high-impact** |

---

## PROCESSING OPTIMIZATIONS (12 items)

### P1: Inline Hot Path Functions (HIGH)
**Finding:** Only 32 `#[inline]` annotations across 84K lines of code. Hot paths in layers/mod.rs (9,559 lines) and network.rs (5,818 lines) have minimal inlining.

**Fix:** Add `#[inline]` or `#[inline(always)]` to:
- All `propagate_*` methods
- All bound arithmetic functions
- Small getter/setter methods

**Expected:** 5-15% speedup on hot paths

### P2: Parallel ndarray Operations (HIGH)
**Finding:** Only 5 `par_iter` uses across entire codebase. 164 `zip` operations could be parallelized.

**Fix:** Replace sequential operations with parallel equivalents:
```rust
// Before
azip!((a in &mut arr_a, b in &arr_b) *a += b);

// After
par_azip!((a in &mut arr_a, b in &arr_b) *a += b);
```

**Expected:** 2-8x speedup on large tensor operations

### P3: Loop Fusion (MEDIUM)
**Finding:** 124 iterator chains (`.iter().map().filter()`) often iterate multiple times over same data.

**Fix:** Fuse loops manually or use itertools:
```rust
// Before
let a = x.iter().map(|v| v * 2).collect::<Vec<_>>();
let b = a.iter().filter(|v| **v > 0).collect::<Vec<_>>();

// After
let b: Vec<_> = x.iter().map(|v| v * 2).filter(|v| *v > 0).collect();
```

**Expected:** 10-30% speedup on iterator-heavy code

### P4: Cache-Friendly Memory Access (MEDIUM)
**Finding:** 384 transpose operations. Transposed iteration is cache-unfriendly.

**Fix:**
- Pre-transpose weights at load time when access pattern is known
- Use blocked/tiled algorithms for matmul
- Consider memory layout (row-major vs col-major) for each operation

**Expected:** 10-50% speedup for memory-bound operations

### P5: Batch Size Auto-Tuning (MEDIUM)
**Finding:** Fixed batch sizes in parallel operations.

**Fix:** Auto-tune batch size based on:
- Available cores
- L2/L3 cache size
- Tensor dimensions

```rust
fn optimal_batch_size(tensor_size: usize, element_size: usize) -> usize {
    let l2_cache = 256 * 1024; // Typical L2
    let elements_per_batch = l2_cache / element_size;
    elements_per_batch.min(tensor_size / num_cpus::get())
}
```

**Expected:** 10-20% speedup from better cache utilization

### P6: SIMD for More Operations (MEDIUM)
**Finding:** SIMD only implemented for interval_mul and pos_neg_split.

**Fix:** Add SIMD for:
- `bounds_add`, `bounds_sub`
- `matmul` inner loops
- `softmax` exp/sum
- `layernorm` mean/variance

**Expected:** 2-4x speedup for these operations

### P7: Lazy Computation / Deferred Execution (LOW)
**Finding:** All bounds computed eagerly even when not needed.

**Fix:** Implement lazy evaluation for bounds that may not be used:
```rust
enum LazyBound {
    Computed(f32),
    Deferred(Box<dyn Fn() -> f32>),
}
```

**Expected:** Variable - avoids unnecessary computation

### P8: Prefetch Hints (LOW)
**Finding:** No software prefetching for sequential tensor access.

**Fix:** Add prefetch intrinsics for large tensor operations:
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_prefetch;
```

**Expected:** 5-10% speedup for memory-bound operations

### P9: Branch Prediction Hints (LOW)
**Finding:** No likely/unlikely hints on error paths.

**Fix:** Add hints for predictable branches:
```rust
if unlikely(x.is_nan()) { ... }
```

**Expected:** Minor improvement on error checking paths

### P10: Compile-Time Computation (LOW)
**Finding:** Constants computed at runtime.

**Fix:** Use `const fn` and compile-time evaluation:
```rust
const GELU_COEFF: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = const_sqrt(2.0 / PI);
```

**Expected:** Minor improvement

### P11: Work Stealing for Unbalanced Loads (LOW)
**Finding:** Equal work distribution assumed across threads.

**Fix:** Use rayon's work-stealing properly for variable-cost operations:
```rust
layers.par_iter()
    .with_min_len(1)  // Enable work stealing
    .for_each(|l| ...)
```

**Expected:** Better load balancing for uneven workloads

### P12: Async I/O for Model Loading (LOW)
**Finding:** Synchronous model loading blocks computation.

**Fix:** Async model loading with tokio:
```rust
let model = tokio::fs::read(path).await?;
```

**Expected:** Better UX for large models

---

## MEMORY OPTIMIZATIONS (9 items)

### M1: Reduce Clone Calls (HIGH)
**Finding:** 1,065 `.clone()` calls. Many are unnecessary defensive copies.

**Fix:** Audit and replace with:
- References (`&T`) where ownership not needed
- `Cow<T>` for conditionally-owned data
- `Arc<T>` for shared ownership

**Expected:** 20-40% memory reduction in some paths

### M2: Pre-allocate Vectors (HIGH)
**Finding:** 1,148 `Vec::new()` or `vec![]` without capacity. 525 `push/extend` calls grow vectors.

**Fix:** Use `Vec::with_capacity()`:
```rust
// Before
let mut results = Vec::new();
for item in items { results.push(process(item)); }

// After
let mut results = Vec::with_capacity(items.len());
for item in items { results.push(process(item)); }
```

**Expected:** 30% fewer allocations, faster vector operations

### M3: Arena Allocator for Graph Nodes (HIGH)
**Finding:** Graph nodes allocated individually. No arena allocator.

**Fix:** Use `bumpalo` or `typed-arena` for graph construction:
```rust
use bumpalo::Bump;
let arena = Bump::new();
let node = arena.alloc(GraphNode { ... });
```

**Expected:** 50%+ faster graph construction, better cache locality

### M4: String Interning (MEDIUM)
**Finding:** 749 string allocations, many for layer names that repeat.

**Fix:** Intern common strings:
```rust
use string_interner::{StringInterner, DefaultSymbol};
let mut interner = StringInterner::default();
let sym = interner.get_or_intern("layer_norm");
```

**Expected:** 50%+ reduction in string memory

### M5: Lazy Error Messages (MEDIUM)
**Finding:** Error messages allocated even on success paths.

**Fix:** Use lazy formatting:
```rust
// Before
Err(format!("Shape mismatch: {:?} vs {:?}", a, b))

// After
Err(GammaError::ShapeMismatch { expected: a, got: b })
```

**Expected:** Zero allocation on success paths

### M6: Reuse Intermediate Buffers (MEDIUM)
**Finding:** Each layer allocates new output buffers.

**Fix:** Pass reusable output buffer through propagation:
```rust
fn propagate_ibp_into(&self, input: &BoundedTensor, output: &mut BoundedTensor) -> Result<()>
```

**Expected:** 50% fewer allocations during propagation

### M7: Memory-Mapped Model Files (LOW)
**Finding:** Models fully loaded into RAM.

**Fix:** Use mmap for large models:
```rust
use memmap2::Mmap;
let file = File::open(path)?;
let mmap = unsafe { Mmap::map(&file)? };
```

**Expected:** Faster model loading, OS-managed memory

### M8: Compact Tensor Representation (LOW)
**Finding:** BoundedTensor stores separate lower/upper arrays.

**Fix:** Interleaved storage for better cache locality:
```rust
struct CompactBounds {
    data: Vec<[f32; 2]>,  // [lower, upper] pairs
}
```

**Expected:** Better cache performance for bound access

### M9: Reference-Counted Layer Parameters (LOW)
**Finding:** Layer weights cloned when network is cloned.

**Fix:** Use `Arc` for large weight matrices:
```rust
pub struct LinearLayer {
    weight: Arc<Array2<f32>>,
    bias: Option<Arc<Array1<f32>>>,
}
```

**Expected:** Faster network cloning for branch-and-bound

---

## DEBUGGING / ERROR REPORTING / UX (11 items)

### D1: Replace unwrap/expect with Proper Errors (HIGH)
**Finding:** 1,799 `unwrap()` or `expect()` calls. Panics instead of recoverable errors.

**Fix:** Replace with `?` operator and Result types:
```rust
// Before
let value = map.get("key").unwrap();

// After
let value = map.get("key").ok_or(GammaError::MissingKey("key"))?;
```

**Expected:** Graceful error handling, no panics

### D2: Structured Logging with tracing (HIGH)
**Finding:** 1,121 `println!/eprintln!` calls. Only 25 tracing uses.

**Fix:** Replace print statements with structured logging:
```rust
// Before
println!("Processing layer {}", i);

// After
tracing::info!(layer = i, "Processing layer");
```

**Expected:** Filterable, structured logs; JSON output support

### D3: Error Codes and Categories (HIGH)
**Finding:** No structured error codes. Hard to programmatically handle errors.

**Fix:** Add error codes to GammaError:
```rust
pub enum GammaError {
    #[error("E001: Shape mismatch")]
    ShapeMismatch { code: &'static str, expected: Vec<usize>, got: Vec<usize> },
}
```

**Expected:** Machine-readable errors for AI-to-AI workflows

### D4: Progress Bar with ETA (MEDIUM)
**Finding:** Progress reporting exists but basic.

**Fix:** Add rich progress bars with indicatif:
```rust
use indicatif::{ProgressBar, ProgressStyle};
let pb = ProgressBar::new(total_layers);
pb.set_style(ProgressStyle::default_bar()
    .template("{msg} [{bar:40}] {pos}/{len} ETA: {eta}"));
```

**Expected:** Better UX for long-running verifications

### D5: Memory Usage Reporting (MEDIUM)
**Finding:** No visibility into memory consumption.

**Fix:** Track and report memory usage:
```rust
pub struct MemoryStats {
    peak_bytes: usize,
    current_bytes: usize,
    allocations: usize,
}
```

**Expected:** Ability to diagnose OOM issues

### D6: Verification Telemetry (MEDIUM)
**Finding:** No metrics collection for performance analysis.

**Fix:** Add metrics with metrics crate:
```rust
metrics::histogram!("layer_propagation_time_ms", duration.as_millis() as f64);
metrics::counter!("bounds_computed", 1);
```

**Expected:** Performance insights, regression detection

### D7: Debug Visualization of Bounds (LOW)
**Finding:** No way to visualize bound propagation.

**Fix:** Add SVG/HTML visualization output:
```rust
fn visualize_bounds(bounds: &BoundedTensor, path: &Path) -> Result<()>
```

**Expected:** Easier debugging of bound explosion

### D8: Verbose Mode Levels (LOW)
**Finding:** Single verbose flag. No granularity.

**Fix:** Multiple verbosity levels:
```
-v    : Basic progress
-vv   : Layer-by-layer info
-vvv  : Per-operation debug
-vvvv : Full trace
```

**Expected:** Better debugging control

### D9: Config File Support (LOW)
**Finding:** All options via CLI args.

**Fix:** Support TOML/YAML config files:
```toml
[verification]
method = "crown"
epsilon = 0.001
timeout = 60
```

**Expected:** Reproducible verification configs

### D10: Shell Completions (LOW)
**Finding:** No shell completion support.

**Fix:** Generate completions with clap:
```rust
#[derive(Parser)]
#[command(name = "gamma")]
struct Cli { ... }

// In build.rs
clap_complete::generate_to(Shell::Bash, &mut Cli::command(), "gamma", out_dir)?;
```

**Expected:** Better CLI UX

### D11: Machine-Readable Output Formats (LOW)
**Finding:** Human-readable output only for some commands.

**Fix:** Add `--json` output for all commands:
```rust
if args.json {
    println!("{}", serde_json::to_string(&result)?);
} else {
    println!("{}", result);
}
```

**Expected:** Better integration with automation

---

## Implementation Priority Matrix

### Phase 1: High-Impact, Low-Effort (Do First)
| ID | Optimization | Effort | Impact |
|----|-------------|--------|--------|
| P1 | Inline hot paths | Low | High |
| M2 | Pre-allocate vectors | Low | High |
| D2 | Structured logging | Medium | High |
| M5 | Lazy error messages | Low | Medium |

### Phase 2: High-Impact, Medium-Effort
| ID | Optimization | Effort | Impact |
|----|-------------|--------|--------|
| P2 | Parallel ndarray ops | Medium | High |
| M1 | Reduce clone calls | Medium | High |
| D1 | Replace unwrap/expect | Medium | High |
| M3 | Arena allocator | Medium | High |

### Phase 3: Medium-Impact
| ID | Optimization | Effort | Impact |
|----|-------------|--------|--------|
| P3 | Loop fusion | Medium | Medium |
| P4 | Cache-friendly access | Medium | Medium |
| P6 | More SIMD | Medium | Medium |
| D3 | Error codes | Medium | Medium |
| D4 | Progress bars | Low | Medium |
| D5 | Memory reporting | Medium | Medium |

### Phase 4: Low-Impact / Nice-to-Have
| ID | Optimization | Effort | Impact |
|----|-------------|--------|--------|
| P5 | Batch auto-tuning | Medium | Medium |
| P7-P12 | Various | Low-Medium | Low |
| M4, M7-M9 | Various | Low-Medium | Low |
| D6-D11 | Various | Low-Medium | Low |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Qwen3-0.6B time | ~15s | < 5s |
| Peak memory | ~8GB | < 4GB |
| unwrap calls | 1,799 | < 100 |
| println calls | 1,121 | 0 |
| clone calls | 1,065 | < 300 |

---

## Commands for Workers

```bash
# Find unwrap calls
grep -rn "unwrap()" crates --include="*.rs" | wc -l

# Find clone calls
grep -rn "\.clone()" crates --include="*.rs" | wc -l

# Find print statements
grep -rn "println!\|eprintln!" crates --include="*.rs"

# Profile memory
heaptrack ./target/release/gamma verify ...

# Profile CPU
perf record -g ./target/release/gamma verify ...
```
