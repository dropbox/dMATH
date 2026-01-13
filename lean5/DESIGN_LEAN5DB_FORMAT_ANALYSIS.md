# Critical Analysis: Lean5DB Design

**Date:** 2026-01-06
**Purpose:** Identify flaws, gaps, and improvements before implementation

---

## Part 1: Identified Flaws and Gaps (10 found)

### Flaw 1: Version Coupling to Lean Toolchain

**Problem:** The design stores a `lean_version_hash` but doesn't address what happens when:
- Lean 4.x changes internal representation of expressions
- A new constructor is added to `ConstantKind`
- Universe polymorphism semantics change

**Impact:** Every Lean minor version bump could invalidate all .lean5db files.

**Severity:** HIGH

---

### Flaw 2: No Incremental Update Strategy

**Problem:** The document mentions "delta updates" as a future extension but the core format doesn't support it. If one module in Mathlib changes:
- Must rebuild entire 1GB file
- No way to patch in place
- Build time scales with total size, not change size

**Impact:** Mathlib updates daily. Rebuilding 4GB → 1GB file takes ~10 minutes each time.

**Severity:** HIGH

---

### Flaw 3: String Table Size Limitation

**Problem:** `StringId = u32` limits to 4 billion strings, but more critically:
- `LengthPrefixedString.len: u16` limits strings to 65KB
- Mathlib has some auto-generated names exceeding this
- Proof terms can have enormous string literals

**Impact:** Silent truncation or crash on edge cases.

**Severity:** MEDIUM

---

### Flaw 4: Expression Pool Doesn't Handle DAG Properly

**Problem:** The `ExprPool` assumes expressions form a DAG, but:
- `ExprId = u32` limits to 4 billion unique expressions
- No cycle detection during building
- Hash collisions in content-addressing not handled

**Impact:** Mathlib may have 100M+ unique subexpressions. Could overflow.

**Severity:** MEDIUM

---

### Flaw 5: Lazy Loading Cache Unbounded

**Problem:** `LazyDatabase.chunk_cache: RwLock<HashMap<u32, DecompressedChunk>>` grows without bound:
- No eviction policy
- Eventually holds entire file in memory
- Defeats purpose of lazy loading

**Impact:** Memory usage grows to full-load levels over time.

**Severity:** MEDIUM

---

### Flaw 6: No Error Recovery for Corrupted Chunks

**Problem:** Per-chunk CRC32 detects corruption but:
- No redundancy to recover from it
- Single bit flip = entire chunk lost
- All constants in chunk become inaccessible

**Impact:** Disk errors could lose thousands of constants.

**Severity:** LOW (rare, but catastrophic when it happens)

---

### Flaw 7: Module Dependency Ordering Not Preserved

**Problem:** Streaming load yields constants in chunk order, but:
- Kernel expects topological order (inductives before constructors before recursors)
- Current design doesn't guarantee this
- Could cause registration failures

**Impact:** Streaming mode may not work without post-sorting.

**Severity:** MEDIUM

---

### Flaw 8: No Support for Mutual Inductives Grouping

**Problem:** Mutual inductives must be registered together, but:
- Constants serialized independently
- No grouping metadata in index
- Loader must reconstruct mutual groups

**Impact:** Complex reconstruction logic needed, potential ordering bugs.

**Severity:** MEDIUM

---

### Flaw 9: Dictionary Not Versioned Separately

**Problem:** Zstd dictionary embedded in file, but:
- Can't share dictionary across multiple .lean5db files
- Can't update dictionary without rebuilding
- Dictionary quality depends on training data

**Impact:** Suboptimal compression for small/specialized libraries.

**Severity:** LOW

---

### Flaw 10: No Partial Build Support

**Problem:** Builder requires all .olean files upfront, but:
- Lake builds modules incrementally
- Can't add new modules to existing .lean5db
- Must wait for full Mathlib build to start conversion

**Impact:** Can't integrate into incremental build pipelines.

**Severity:** MEDIUM

---

## Part 2: Potential Optimizations and Improvements (10 found)

### Improvement 1: Tiered Expression Storage

**Observation:** Most constants reference a small "hot" set of expressions repeatedly.

**Proposal:**
```rust
struct TieredExprPool {
    // Tier 0: Ultra-common (Nat, Prop, Type, Bool) - inline in 1 byte
    tier0: [Expr; 256],

    // Tier 1: Common (List, Option, basic types) - 2 bytes
    tier1: [Expr; 65536],

    // Tier 2: Everything else - 4 bytes
    tier2: Vec<Expr>,
}
```

**Expected benefit:** 30-40% reduction in expression reference size.

---

### Improvement 2: Columnar Storage for Index

**Observation:** Index entries have fixed fields that compress better separately.

**Proposal:**
```rust
struct ColumnarIndex {
    names: [StringId; N],           // Compress together
    kinds: [ConstantKind; N],       // 1 byte each, great RLE
    type_expr_ids: [ExprId; N],     // Compress together
    data_offsets: [u32; N],         // Delta encoding
}
```

**Expected benefit:** 20-30% smaller index, faster scans.

---

### Improvement 3: SIMD-Accelerated Name Lookup

**Observation:** Binary search on sorted names is cache-unfriendly.

**Proposal:**
- Store name hash prefixes in contiguous array
- Use SIMD to search 8-16 entries at once
- Fall back to full comparison on hash match

```rust
struct SIMDIndex {
    hash_prefixes: [u32; N],  // First 4 bytes of name hash
    entries: [IndexEntry; N],
}
```

**Expected benefit:** 2-4x faster single-constant lookup.

---

### Improvement 4: Proof Elision with Commitment

**Observation:** Proofs are huge but rarely needed.

**Proposal:** Store proof commitment instead of full proof:
```rust
enum ProofStorage {
    // Full proof term
    Full(ExprId),

    // Hash commitment - can verify if proof provided externally
    Committed {
        type_hash: [u8; 32],
        proof_hash: [u8; 32],
    },

    // Trusted - no verification possible
    Trusted,
}
```

**Expected benefit:** 60-80% file size reduction if proofs elided.

---

### Improvement 5: Expression Tree Flattening

**Observation:** Deep expression trees require many pointer chases.

**Proposal:** Flatten common patterns:
```rust
enum FlatExpr {
    // Normal cases
    BVar(u32),
    Sort(LevelId),

    // Flattened: App chain stored as single node
    AppChain {
        func: ExprId,
        args: SmallVec<[ExprId; 8]>,
    },

    // Flattened: Nested Pi stored as telescope
    PiTelescope {
        binders: SmallVec<[(BinderInfo, ExprId); 8]>,
        body: ExprId,
    },
}
```

**Expected benefit:** 30% fewer expression nodes, faster traversal.

---

### Improvement 6: Parallel Chunk Decompression with Prefetch

**Observation:** Chunk decompression is CPU-bound, loading is I/O-bound.

**Proposal:**
```rust
impl LazyDatabase {
    fn get_constant_with_prefetch(&self, name: &Name) -> Option<ConstantInfo> {
        let entry = self.index.lookup(name)?;

        // Prefetch likely-needed chunks (same module, adjacent in index)
        self.prefetch_chunks(&[
            entry.data_chunk,
            entry.data_chunk + 1,
        ]);

        // Decompress primary chunk
        let chunk = self.ensure_chunk_loaded(entry.data_chunk);
        chunk.deserialize_constant(entry.data_offset)
    }
}
```

**Expected benefit:** Hide I/O latency, 50% faster batch lookups.

---

### Improvement 7: Name Trie Instead of Sorted Array

**Observation:** Module-hierarchical names share prefixes heavily.

**Proposal:**
```rust
struct NameTrie {
    // "Mathlib" → children
    //   "Data" → children
    //     "Nat" → children
    //       "Basic" → [constant indices]
    nodes: Vec<TrieNode>,
}

struct TrieNode {
    component: StringId,
    children: SmallVec<[u32; 4]>,  // Child node indices
    constants: SmallVec<[u32; 8]>, // Constant indices at this path
}
```

**Expected benefit:** O(name_depth) lookup instead of O(log N), better prefix queries.

---

### Improvement 8: Memory-Mapped Expression Pool

**Observation:** Expression pool is read-only after load.

**Proposal:**
- Keep expression pool memory-mapped, don't copy to heap
- Expressions accessed via pointer into mmap region
- OS manages page caching automatically

```rust
struct MmapExprPool<'a> {
    data: &'a [u8],  // Points into mmap
}

impl<'a> MmapExprPool<'a> {
    fn get(&self, id: ExprId) -> &'a SerializedExpr {
        // Zero-copy access
        &self.data[offset_of(id)..]
    }
}
```

**Expected benefit:** 40% lower memory for expression pool.

---

### Improvement 9: Bloom Filter for Negative Lookups

**Observation:** Many lookups are for names that don't exist.

**Proposal:**
```rust
struct Lean5DbHeader {
    // ... existing fields ...

    // Bloom filter for quick negative lookup
    bloom_filter_offset: u64,
    bloom_filter_size: u32,  // ~1MB for 100K entries, <0.1% false positive
}

impl LazyDatabase {
    fn has_constant(&self, name: &Name) -> bool {
        // Fast path: bloom filter says no
        if !self.bloom_filter.maybe_contains(name) {
            return false;
        }
        // Slow path: actual lookup
        self.index.lookup(name).is_some()
    }
}
```

**Expected benefit:** 10x faster "not found" responses.

---

### Improvement 10: Separate Proof Archive

**Observation:** Proofs needed only for re-verification, not normal use.

**Proposal:** Split into two files:
```
mathlib.lean5db        # Types, definitions, theorem statements (~400 MB)
mathlib.proofs.lean5db # Proof terms only (~3 GB)
```

```rust
struct ProofReference {
    theorem_name: StringId,
    proof_file_offset: u64,
    proof_size: u32,
    proof_hash: [u8; 32],
}
```

**Expected benefit:**
- Normal loading: 400 MB instead of 4 GB
- Verification: Load proofs on demand
- Distribution: Ship proofs separately or not at all

---

## Part 3: Prioritized Recommendations

### Must Fix Before v1.0 (Blockers)

| Issue | Solution | Effort |
|-------|----------|--------|
| Flaw 1: Version coupling | Add format migration layer, version negotiation | 3 days |
| Flaw 7: Ordering not preserved | Add topological sort metadata, chunk ordering constraints | 2 days |
| Flaw 8: Mutual inductives | Add `mutual_group_id` to index entries | 1 day |
| Flaw 5: Unbounded cache | Add LRU eviction policy | 1 day |

### Should Include in v1.0 (High Value)

| Improvement | Benefit | Effort |
|-------------|---------|--------|
| Improvement 10: Separate proofs | 90% size reduction for normal use | 3 days |
| Improvement 4: Proof commitment | Enable trustless proof elision | 2 days |
| Improvement 9: Bloom filter | Fast negative lookups | 1 day |
| Improvement 1: Tiered expressions | 30% smaller references | 2 days |

### Can Defer to v1.1 (Nice to Have)

| Item | Reason to Defer |
|------|-----------------|
| Flaw 2: Incremental updates | Complex, needs separate delta format |
| Improvement 2: Columnar storage | Optimization, measure first |
| Improvement 3: SIMD lookup | Micro-optimization |
| Improvement 5: Tree flattening | Significant format change |

### Won't Fix (Acceptable)

| Item | Reason |
|------|--------|
| Flaw 3: String size limit | Increase to u32 if needed, unlikely to hit |
| Flaw 6: No error recovery | Use ECC storage, backup files |
| Flaw 9: Dictionary versioning | Rebuild is fast enough |

---

## Part 4: Revised Design Decisions

Based on this analysis, the following design changes are recommended:

### 4.1 File Structure (Revised)

```
┌────────────────────────────────────────┐
│          LEAN5DB FILE v1.1             │
├────────────────────────────────────────┤
│  Header (192 bytes, expanded)          │
│    + bloom_filter_offset               │
│    + proof_archive_path (optional)     │
│    + format_version (semantic)         │
├────────────────────────────────────────┤
│  Bloom Filter (1 MB typical)           │  ← NEW
├────────────────────────────────────────┤
│  String Table (compressed)             │
├────────────────────────────────────────┤
│  Tiered Expression Pool (compressed)   │  ← CHANGED
│    Tier 0: inline common (256)         │
│    Tier 1: short refs (64K)            │
│    Tier 2: full pool                   │
├────────────────────────────────────────┤
│  Constant Index (columnar, compressed) │  ← CHANGED
│    + mutual_group_id column            │
│    + topological_order column          │
├────────────────────────────────────────┤
│  Constant Data Chunks (compressed)     │
│    Ordered by topological sort         │  ← CHANGED
├────────────────────────────────────────┤
│  Module Metadata (compressed)          │
├────────────────────────────────────────┤
│  Proof Commitments (if proofs elided)  │  ← NEW
├────────────────────────────────────────┤
│  Footer (checksum, 64 bytes)           │
└────────────────────────────────────────┘

Optional companion file:
┌────────────────────────────────────────┐
│      LEAN5DB PROOF ARCHIVE             │
│  (mathlib.proofs.lean5db)              │
├────────────────────────────────────────┤
│  Header (references main file)         │
├────────────────────────────────────────┤
│  Proof Terms (compressed chunks)       │
└────────────────────────────────────────┘
```

### 4.2 LazyDatabase (Revised)

```rust
pub struct LazyDatabase {
    mmap: Mmap,
    header: Lean5DbHeader,

    // Always memory-mapped (zero-copy)
    bloom_filter: BloomFilter<'static>,
    strings: MmapStringTable<'static>,
    expr_pool: MmapExprPool<'static>,
    index: ColumnarIndex,

    // LRU cache with size limit
    chunk_cache: LruCache<u32, Arc<DecompressedChunk>>,
    cache_size_limit: usize,

    // Optional proof archive
    proof_archive: Option<ProofArchive>,
}
```

### 4.3 Builder (Revised)

```rust
pub struct BuildOptions {
    // Existing
    pub compression_level: i32,
    pub train_dictionary: bool,
    pub chunk_size: usize,
    pub parallel: bool,

    // New
    pub proof_mode: ProofMode,
    pub include_bloom_filter: bool,
    pub topological_chunk_ordering: bool,  // Always true in v1.1
}

pub enum ProofMode {
    /// Include full proof terms in main file
    Full,

    /// Store commitment only, proofs in separate archive
    Separate { archive_path: PathBuf },

    /// Elide proofs entirely (trusted mode)
    Elided,
}
```

---

## Part 5: Updated Implementation Phases

### Phase 1: Core Format v1.1 (2.5 weeks)
- Header with new fields
- Bloom filter
- LRU chunk cache
- Topological ordering

### Phase 2: Tiered Expression Pool (2 weeks)
- Tier analysis pass
- Tiered serialization
- Memory-mapped access

### Phase 3: Builder with Proof Separation (2.5 weeks)
- Proof commitment generation
- Separate proof archive format
- Build modes (full/separate/elided)

### Phase 4: Loader with Proofs-on-Demand (1.5 weeks)
- Lazy proof loading
- Proof verification

### Phase 5: Columnar Index (1 week)
- Column-oriented storage
- Delta encoding
- Mutual group tracking

### Phase 6: CLI, Testing, Benchmarks (1.5 weeks)
- Full Mathlib conversion test
- Performance regression tests
- Documentation

**Revised Total: ~11 weeks** (was 10 weeks)

---

## Conclusion

The original design is sound but needs refinements for production use. The most critical additions are:

1. **Proof separation** - Transforms the economics of the format
2. **Topological ordering** - Required for correct loading
3. **LRU caching** - Makes lazy loading actually lazy
4. **Bloom filter** - Essential for interactive use

With these changes, the format will achieve its performance targets while being robust and maintainable.

---

## Summary of Changes Applied to Main Design

The following changes have been integrated into `DESIGN_LEAN5DB_FORMAT.md` v1.1:

### Format Changes
- Header expanded from 128 to 192 bytes
- Added bloom filter section
- Added proof commitments section
- Added proof archive companion file spec

### New Flags
- `FLAG_HAS_BLOOM` (0x0040)
- `FLAG_TIERED_EXPRS` (0x0080)
- `FLAG_TOPO_ORDERED` (0x0100)
- `FLAG_PROOFS_SEPARATE` (0x0200)
- `FLAG_PROOFS_COMMITTED` (0x0400)

### LazyDatabase Changes
- Added bloom filter for O(1) negative lookup
- Added LRU eviction policy with size limit
- Added memory-mapped string/expr pools
- Added optional proof archive reference

### BuildOptions Changes
- Added `ProofMode` enum with 4 modes
- Added `include_bloom_filter` option
- Added `topological_chunk_ordering` option

### Implementation Timeline
- Extended from 10 weeks to 11.5 weeks
- Reorganized phases to address flaws early
- Added risk mitigation table

### Estimated Impact

| Metric | v1.0 Target | v1.1 Target (with proof separation) |
|--------|-------------|-------------------------------------|
| Mathlib file size | <1 GB | **<400 MB** (main) + 3.5 GB (proofs) |
| Load time (types only) | <6s | **<2s** |
| Memory (lazy cold) | <200 MB | **<100 MB** |
| Negative lookup time | O(log N) | **O(1)** |
