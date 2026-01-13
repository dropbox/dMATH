# PLAN: General Performance Optimizations

**Status:** ✅ COMPLETE
**Created:** 2026-01-01
**Completed:** 2026-01-01
**Goal:** 2-10x speedup and 50% memory reduction via general software optimizations

---

## Context

These are algorithm-agnostic optimizations that improve all verification methods.

**Current baseline:**
- Qwen3-0.6B verification: ~10-30 seconds
- Memory usage: scales linearly with model size
- CPU utilization: partially parallelized via rayon

**Findings from codebase analysis:**
- 226 allocation sites in gamma-propagate alone
- ndarray with rayon threading enabled
- No explicit SIMD vectorization
- No tensor memory pooling

---

## OPTIMIZATION 1: Tensor Memory Pool ✅ IMPLEMENTED

**Problem:** Each layer propagation allocates new tensors, causing GC pressure and fragmentation.

**Solution:** Thread-local arena allocator that reuses tensor memory across layers.

**Status:** Implemented in iteration #255

**Implementation:**
1. ✅ `TensorPool` in `gamma-tensor/src/pool.rs` - thread-local pool with size-class bucketing
2. ✅ `PooledBuffer` - auto-returning buffer handle with RAII drop
3. ✅ `pooled` module in `gamma-propagate/src/pooled.rs` - helper functions
4. ✅ Benchmarks in `benches/propagation.rs` - Pool group

**API:**
```rust
use gamma_tensor::{TensorPool, PooledBuffer};

// Acquire buffer (zeros by default)
let mut buffer = TensorPool::acquire(10000);
// Use as slice
buffer.as_mut_slice()[0] = 42.0;
// Auto-returns to pool on drop
drop(buffer);

// Or convert to ArrayD (consumes buffer, won't return to pool)
let array = buffer.into_arrayd(&[100, 100]);
```

**Benchmark results (1M elements = 4MB buffer):**
| Operation | Time | vs Vec::new |
|-----------|------|-------------|
| Vec::new + zero | 143.7 µs | baseline |
| Pool acquire+drop | 17.2 µs | **8.4x faster** |
| Pool reuse (warm) | 17.2 µs | **8.4x faster** |

**Key findings:**
- Pool is highly effective for temporary scratch buffers (8x speedup)
- Pool is NOT beneficial when converting to ArrayD (similar or slower)
- Zeroing dominates: ~17µs to zero 4MB regardless of allocation source

**Usage guidance:**
- Use `PooledBuffer` directly when possible (tight loops, temporary computation)
- Use `into_arrayd()` only for final results that need to persist
- Pool is thread-local, no locking overhead with rayon

---

## OPTIMIZATION 2: SIMD Vectorization for Bound Arithmetic ✅ IMPLEMENTED

**Problem:** Interval arithmetic (min/max/add/mul on bounds) is element-wise but not vectorized.

**Solution:** Use explicit SIMD intrinsics for bound operations.

**Status:** Implemented in iteration #256

**Implementation:**
1. ✅ `simd` module in `gamma-tensor/src/simd.rs` - NEON (aarch64) and AVX2 (x86_64) implementations
2. ✅ `interval_mul` - SIMD interval multiplication with 4-way min/max
3. ✅ `pos_neg_split` - Single-pass positive/negative coefficient split
4. ✅ `dot`, `sum`, `safe_mul_add` - SIMD helper functions
5. ✅ Integrated into `BoundedTensor::mul()` with automatic SIMD dispatch
6. ✅ Benchmarks in `benches/simd.rs`

**API:**
```rust
use gamma_tensor::simd;

// SIMD interval multiplication
simd::interval_mul(a_lower, a_upper, b_lower, b_upper, &mut out_lower, &mut out_upper);

// Single-pass pos/neg split (avoids two separate mapv calls)
simd::pos_neg_split(&x, &mut pos, &mut neg);

// SIMD dot product
let dot = simd::dot(&a, &b);
```

**Benchmark results (Apple Silicon M3):**
| Operation | Size | Throughput | vs Baseline |
|-----------|------|------------|-------------|
| interval_mul | 1M | 4.0 Gelem/s | - |
| pos_neg_split (SIMD) | 1M | 7.4 Gelem/s | **2.9x** faster than mapv |
| dot (SIMD) | 1M | 4.0 Gelem/s | ~1.1x faster |

**Key findings:**
- SIMD interval_mul achieves 4-6 Gelem/s throughput
- pos_neg_split single-pass is **2.9x faster** than two separate mapv calls
- BoundedTensor::mul uses SIMD automatically when arrays are contiguous
- Automatic fallback to scalar for non-contiguous arrays

---

## OPTIMIZATION 3: Streaming Layer Computation ✅ IMPLEMENTED

**Problem:** Full model graph is materialized in memory before verification.

**Solution:** Gradient checkpointing - store bounds at intervals, recompute from checkpoints during backward pass.

**Status:** Implemented in iteration #258

**Implementation:**
1. ✅ `streaming` module in `gamma-propagate/src/streaming.rs`
2. ✅ `StreamingConfig` - configurable checkpoint intervals, memory limits
3. ✅ `StreamingVerifier` with `propagate_crown_streaming()` and `propagate_crown_batched_streaming()`
4. ✅ `CheckpointedBounds` - checkpoint storage with recomputation support
5. ✅ `estimate_memory_savings()` - calculate expected memory reduction
6. ✅ Criterion benchmarks in `benches/propagation.rs` (Streaming group)
7. ✅ 10 unit tests verifying equivalence with regular CROWN

**API:**
```rust
use gamma_propagate::streaming::{StreamingConfig, StreamingVerifier, estimate_memory_savings};

// Configure streaming (checkpoint every 10 layers)
let config = StreamingConfig {
    checkpoint_interval: 10,
    enable_crown_streaming: true,
    max_memory_bytes: 0, // Unlimited (or set limit for auto-adjust)
};

// Or use presets
let config = StreamingConfig::min_memory();   // interval=50
let config = StreamingConfig::balanced();      // interval=10

let verifier = StreamingVerifier::new(config);

// Memory-efficient CROWN propagation
let bounds = verifier.propagate_crown_streaming(&network, &input)?;

// Or batched version for transformers
let bounds = verifier.propagate_crown_batched_streaming(&network, &input)?;

// Estimate memory savings
let (original, streaming, savings_pct) = estimate_memory_savings(100, 1000, 10);
// 100 layers, 1000 elements/tensor, interval=10 → ~89% savings
```

**Memory-Compute Trade-off:**
| Checkpoint Interval | Memory | Compute | Use Case |
|---------------------|--------|---------|----------|
| 1 (no streaming) | O(L*N) | O(L) | Small models |
| 5 | ~80% reduction | ~5x recompute | Medium models |
| 10 | ~90% reduction | ~10x recompute | Large models |
| 50 | ~98% reduction | ~50x recompute | Memory-constrained |

**Key findings:**
- Memory savings scale with checkpoint interval: ~(1 - 1/K) reduction
- Recomputation overhead is bounded by interval (worst case: K layers recomputed per backward step)
- For deep networks (50+ layers), interval=10 provides good balance
- Streaming enables verification of models that would OOM with regular CROWN

---

## OPTIMIZATION 4: Batch Parallelization Across Positions ✅ IMPLEMENTED

**Problem:** Sequence positions are verified serially; each position is independent.

**Solution:** Parallelize across sequence positions using rayon.

**Status:** Implemented in iteration #257

**Implementation:**
1. ✅ `ParallelVerifier` in `gamma-propagate/src/parallel.rs` - parallel position verification
2. ✅ `ParallelConfig` - configurable parallelization settings
3. ✅ `verify_parallel()` and `verify_parallel_with_method()` convenience functions
4. ✅ `slice_axis()`, `stack()`, `concat()` methods in `gamma-tensor` BoundedTensor
5. ✅ Criterion benchmark in `benches/propagation.rs` (Parallel/Positions group)
6. ✅ Unit tests for parallel vs serial equivalence

**API:**
```rust
use gamma_propagate::parallel::{ParallelConfig, ParallelVerifier};
use gamma_propagate::PropagationMethod;

let config = ParallelConfig {
    method: PropagationMethod::Ibp,
    min_positions_for_parallel: 4, // Threshold for parallel execution
    max_threads: None,             // Use rayon default (all cores)
    report_progress: false,
};
let verifier = ParallelVerifier::new(config);

// Input shape: [batch, seq_len, hidden]
let result = verifier.verify_positions_parallel(&graph, &input, 1)?; // axis=1 for seq
// result.output_bounds: [batch, seq_len, output_dim]
// result.num_positions: seq_len
// result.parallel_positions: number actually parallelized
// result.total_time_ms: verification time

// Or use convenience functions:
let output = verify_parallel(&graph, &input, 1)?;
let output = verify_parallel_with_method(&graph, &input, 1, PropagationMethod::Crown)?;
```

**Key findings:**
- Near-linear speedup with cores for position-independent verification
- Automatic serial fallback for small position counts (below threshold)
- Thread-local tensor pools ensure no lock contention
- Results identical to serial verification (verified by tests)

---

## OPTIMIZATION 5: f16 Intermediate Storage ✅ IMPLEMENTED

**Problem:** All intermediate bounds stored as f32, doubling memory vs necessary precision.

**Solution:** Store intermediate bounds as f16, convert to f32 only for computation.

**Status:** Implemented in iteration #259

**Implementation:**
1. ✅ `CompressedBounds` in `gamma-tensor/src/compressed.rs` - f16 storage with serde support
2. ✅ `from_bounded_tensor()` and `to_bounded_tensor()` - conversion methods
3. ✅ `widen_for_soundness()` - conservative bound widening for precision loss
4. ✅ `CompressionStats` - compression quality metrics
5. ✅ Integration with `StreamingConfig` - `use_f16_checkpoints` and `f16_widening_epsilon` options
6. ✅ `CheckpointedBounds` supports both f32 and f16 storage modes
7. ✅ Criterion benchmarks in `benches/compressed.rs`

**API:**
```rust
use gamma_tensor::{BoundedTensor, CompressedBounds, CompressionStats};

// Create bounds
let bounds = BoundedTensor::new(lower, upper).unwrap();

// Compress for storage (50% memory reduction)
let compressed = CompressedBounds::from_bounded_tensor(&bounds);

// Optional: widen for soundness (compensate f16 precision loss)
compressed.widen_for_soundness(0.001);  // 0.1% widening

// Decompress when needed for computation
let restored = compressed.to_bounded_tensor().unwrap();

// Get compression stats
let stats = CompressionStats::from_compression(&bounds, &compressed);
println!("Memory savings: {:.1}%", stats.memory_savings_percent());
```

**Streaming Integration:**
```rust
use gamma_propagate::streaming::{StreamingConfig, StreamingVerifier};

// Enable f16 for streaming checkpoints
let config = StreamingConfig {
    checkpoint_interval: 10,
    use_f16_checkpoints: true,
    f16_widening_epsilon: 0.001,
    ..Default::default()
};
// Or use preset
let config = StreamingConfig::min_memory();  // f16 + interval=50

let verifier = StreamingVerifier::new(config);
let result = verifier.propagate_crown_streaming(&network, &input)?;
```

**Memory savings (measured):**
| Storage | Bytes/element | vs f32 |
|---------|---------------|--------|
| f32 (default) | 8 (4 lower + 4 upper) | baseline |
| f16 (compressed) | 4 (2 lower + 2 upper) | **50% reduction** |

**Combined with checkpointing:**
| Configuration | Memory vs baseline |
|---------------|-------------------|
| Full f32 storage | 100% |
| Checkpointing (K=10) | ~10% |
| Checkpointing + f16 | ~5% |

**Precision considerations:**
- f16 has ~3 decimal digits precision (vs ~7 for f32)
- f16 max value is ~65504 (bounds overflow to infinity above this)
- Widening compensates for precision loss (sound but looser bounds)
- Typical precision loss: < 0.1% for normal bound ranges

---

## Implementation Priority

| Optimization | Impact | Effort | Priority | Status |
|-------------|--------|--------|----------|--------|
| Tensor Memory Pool | HIGH | Medium | 1 | ✅ Done (#255) |
| SIMD Vectorization | HIGH | Medium | 2 | ✅ Done (#256) |
| Batch Parallel Positions | MEDIUM | Low | 3 | ✅ Done (#257) |
| Streaming Computation | MEDIUM | High | 4 | ✅ Done (#258) |
| f16 Intermediate Storage | LOW-MEDIUM | Low | 5 | ✅ Done (#259) |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Qwen3-0.6B verification time | ~15s | < 5s |
| Peak memory (Qwen3-0.6B) | ~8GB | < 4GB |
| CPU utilization | ~40% | > 80% |
| Qwen3-32B feasibility | OOM | Completes |

---

## Commands

```bash
# Benchmark current performance
hyperfine './target/release/gamma verify models/whisper-tiny/model.safetensors --native --method crown --epsilon 0.001'

# Profile memory
heaptrack ./target/release/gamma verify ~/Models/Qwen3-0.6B.Q6_K.gguf --native --method crown --epsilon 0.001

# Profile CPU
perf record ./target/release/gamma verify ...
perf report
```
