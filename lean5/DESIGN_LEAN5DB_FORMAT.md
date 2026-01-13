# DESIGN: Lean5DB - High-Performance Kernel Object Storage

**Status:** Draft v1.1 (revised after critical analysis)
**Author:** AI Worker
**Date:** 2026-01-06
**Project:** lean5-fastload (parallel to lean5-olean)
**Analysis:** See DESIGN_LEAN5DB_FORMAT_ANALYSIS.md for detailed flaw/improvement analysis

---

## Executive Summary

This document specifies **Lean5DB**, a custom binary format optimized for fast loading of Lean 4 kernel objects into the Lean5 kernel. The primary target is Mathlib (~4GB, 120K+ constants) with a goal of **10x faster loading** and **4x lower memory usage** compared to native .olean parsing.

---

## 1. Problem Statement

### 1.1 Current State

The .olean format is optimized for Lean 4's runtime, not external tools:

| Metric | .olean (Mathlib est.) | Problem |
|--------|----------------------|---------|
| Total size | ~4 GB | No compression |
| Load time | ~60-120s | Pointer traversal + conversion |
| Memory | ~8 GB peak | Full parse before use |
| Format | Pointer-based | Requires expensive translation |

### 1.2 Root Causes

1. **Pointer-based serialization**: Objects stored as memory dumps with absolute pointers requiring fixup
2. **No compression**: Trading space for mmap() compatibility
3. **Full proof terms**: Every theorem stores complete proof (often megabytes)
4. **No deduplication**: `Nat`, `Prop`, `List α` repeated millions of times
5. **Per-module granularity**: Can't load partial libraries efficiently

### 1.3 Goals

| Goal | Target |
|------|--------|
| Mathlib load time | < 6 seconds |
| Memory usage | < 2 GB |
| Constants/sec throughput | > 500,000 |
| File size (Mathlib) | < 1 GB |
| Incremental load support | Yes |
| Lazy/on-demand loading | Yes |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        lean5-fastload                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Builder    │    │    Loader    │    │    Index     │      │
│  │              │    │              │    │              │      │
│  │ .olean files │───▶│  .lean5db   │───▶│ Environment  │      │
│  │ (Mathlib)    │    │   file(s)   │    │   (kernel)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                   Shared Components                   │      │
│  ├──────────────────────────────────────────────────────┤      │
│  │  StringTable │ ExprPool │ Compression │ Checksums    │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Crate Structure

```
lean5-fastload/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── format.rs        # File format definitions, magic bytes, versioning
│   ├── header.rs        # Header parsing/writing
│   ├── strings.rs       # String table with interning
│   ├── expr_pool.rs     # Deduplicated expression storage
│   ├── index.rs         # B-tree index for constant lookup
│   ├── compress.rs      # Zstd compression with dictionary
│   ├── builder.rs       # .olean → .lean5db conversion
│   ├── loader.rs        # .lean5db → Environment loading
│   ├── lazy.rs          # On-demand loading support
│   ├── mmap.rs          # Memory-mapped file access
│   └── verify.rs        # Checksum and integrity verification
├── benches/
│   └── loading.rs       # Performance benchmarks
└── tests/
    ├── roundtrip.rs     # Conversion correctness
    └── mathlib.rs       # Full Mathlib tests
```

---

## 3. File Format Specification

### 3.1 Overview

```
┌────────────────────────────────────────┐
│          LEAN5DB FILE v1.1             │
├────────────────────────────────────────┤
│  Header (192 bytes, expanded)          │
├────────────────────────────────────────┤
│  Bloom Filter (~1MB, for fast lookup)  │  ← NEW in v1.1
├────────────────────────────────────────┤
│  String Table (compressed)             │
├────────────────────────────────────────┤
│  Tiered Expression Pool (compressed)   │  ← IMPROVED in v1.1
├────────────────────────────────────────┤
│  Constant Index (columnar, compressed) │  ← IMPROVED in v1.1
├────────────────────────────────────────┤
│  Constant Data Chunks (topo-ordered)   │  ← IMPROVED in v1.1
├────────────────────────────────────────┤
│  Module Metadata (compressed)          │
├────────────────────────────────────────┤
│  Proof Commitments (if proofs elided)  │  ← NEW in v1.1
├────────────────────────────────────────┤
│  Footer (checksum, 64 bytes)           │
└────────────────────────────────────────┘

Optional companion file (when proof_mode=Separate):
┌────────────────────────────────────────┐
│      LEAN5DB PROOF ARCHIVE             │
│  (mathlib.proofs.lean5db)              │
├────────────────────────────────────────┤
│  Header (references main file hash)    │
├────────────────────────────────────────┤
│  Proof Terms (compressed chunks)       │
└────────────────────────────────────────┘
```

**Key v1.1 Changes (from critical analysis):**
1. **Bloom filter** for O(1) negative lookups (Improvement #9)
2. **Tiered expression pool** for smaller references (Improvement #1)
3. **Topological chunk ordering** for correct streaming load (Flaw #7 fix)
4. **Proof separation** for 90% size reduction (Improvement #10)
5. **Mutual group tracking** in index (Flaw #8 fix)

### 3.2 Header (192 bytes) — REVISED in v1.1

```rust
#[repr(C)]
struct Lean5DbHeader {
    // Identification (16 bytes)
    magic: [u8; 8],           // "LEAN5DB\0"
    version_major: u16,       // Format major version (1)
    version_minor: u16,       // Format minor version (1)
    flags: u32,               // Feature flags

    // Section offsets (80 bytes) — expanded in v1.1
    bloom_filter_offset: u64,           // NEW: Bloom filter for fast negative lookup
    bloom_filter_size: u32,
    _padding1: u32,

    string_table_offset: u64,
    string_table_compressed_size: u64,
    string_table_uncompressed_size: u64,

    expr_pool_offset: u64,
    expr_pool_compressed_size: u64,
    expr_pool_uncompressed_size: u64,

    index_offset: u64,
    index_compressed_size: u64,

    data_offset: u64,
    data_compressed_size: u64,

    proof_commitments_offset: u64,      // NEW: For proof elision mode
    proof_commitments_size: u64,

    // Statistics (40 bytes) — expanded in v1.1
    total_constants: u64,
    total_modules: u32,
    total_strings: u32,
    total_expressions: u64,
    total_mutual_groups: u32,           // NEW: For correct ordering
    _padding2: u32,
    lean_version_hash: [u8; 8],         // Lean toolchain identifier

    // Proof archive reference (32 bytes) — NEW in v1.1
    proof_archive_hash: [u8; 32],       // SHA-256 of companion .proofs file (or zeros)

    // Reserved (24 bytes)
    reserved: [u8; 24],
}
```

**Flags:**
```rust
const FLAG_HAS_PROOFS: u32        = 0x0001;  // Proof terms included in main file
const FLAG_HAS_VALUES: u32        = 0x0002;  // Definition values included
const FLAG_LAZY_COMPATIBLE: u32   = 0x0004;  // Supports lazy loading
const FLAG_ZSTD_DICT: u32         = 0x0008;  // Uses trained dictionary
const FLAG_EXPR_DEDUPE: u32       = 0x0010;  // Expression deduplication enabled
const FLAG_CHECKSUMMED: u32       = 0x0020;  // Per-chunk checksums
// NEW in v1.1:
const FLAG_HAS_BLOOM: u32         = 0x0040;  // Bloom filter present
const FLAG_TIERED_EXPRS: u32      = 0x0080;  // Tiered expression pool
const FLAG_TOPO_ORDERED: u32      = 0x0100;  // Chunks in topological order
const FLAG_PROOFS_SEPARATE: u32   = 0x0200;  // Proofs in companion archive
const FLAG_PROOFS_COMMITTED: u32  = 0x0400;  // Proof commitments (hashes) stored
```

### 3.3 String Table

All strings (names, level parameters) stored once and referenced by index.

```rust
struct StringTable {
    // Header
    count: u32,
    total_bytes: u32,

    // Offset array for O(1) lookup
    offsets: [u32; count],  // Byte offset of each string

    // String data (length-prefixed, UTF-8)
    data: [LengthPrefixedString; count],
}

struct LengthPrefixedString {
    len: u16,           // Max 65535 bytes per string
    bytes: [u8; len],   // UTF-8 data (no null terminator)
}
```

**Interning strategy:**
- Common names ("Nat", "Bool", "Prop", "List", etc.) get low indices
- Names sorted by frequency for better compression
- Hierarchical names stored as components: `["Nat", "add", "comm"]` → indices

### 3.4 Expression Pool

Deduplicated expression storage using content-addressing.

```rust
struct ExprPool {
    count: u64,

    // Each expression has a unique ID (index into this pool)
    expressions: [SerializedExpr; count],
}

// Compact expression encoding
enum SerializedExpr {
    BVar(u32),                              // De Bruijn index
    Sort(LevelId),                          // Universe level (pool reference)
    Const(StringId, [LevelId]),             // Name + level args
    App(ExprId, ExprId),                    // Function, argument
    Lam(BinderInfo, ExprId, ExprId),        // Type, body
    Pi(BinderInfo, ExprId, ExprId),         // Domain, codomain
    Let(ExprId, ExprId, ExprId),            // Type, value, body
    Lit(Literal),                           // Nat or String literal
    Proj(StringId, u32, ExprId),            // Struct, index, expr
}

// All IDs are u32 indices
type ExprId = u32;
type StringId = u32;
type LevelId = u32;
```

**Deduplication:**
- Hash each expression by structure
- Store unique expressions only once
- Common subexpressions (`Nat`, `Type`, `∀ x, ...`) shared globally
- Expected deduplication ratio: 5-10x for Mathlib

### 3.5 Level Pool

Universe levels also deduplicated.

```rust
enum SerializedLevel {
    Zero,
    Succ(LevelId),
    Max(LevelId, LevelId),
    IMax(LevelId, LevelId),
    Param(StringId),
}
```

### 3.6 Constant Index

B-tree structure for fast name lookup.

```rust
struct ConstantIndex {
    // Sorted array of entries (enables binary search)
    entries: [IndexEntry; count],
}

struct IndexEntry {
    name_id: StringId,          // Full qualified name
    kind: ConstantKind,         // 1 byte: axiom/def/thm/ind/ctor/rec
    module_id: u16,             // Which module defined this
    data_chunk: u32,            // Which data chunk contains this constant
    data_offset: u32,           // Offset within chunk
    type_expr_id: ExprId,       // Direct reference to type (for quick access)
    flags: u8,                  // is_reducible, has_value, etc.
}

#[repr(u8)]
enum ConstantKind {
    Axiom = 0,
    Definition = 1,
    Theorem = 2,
    Opaque = 3,
    Inductive = 4,
    Constructor = 5,
    Recursor = 6,
    Quotient = 7,
}
```

### 3.7 Constant Data Chunks

Constants grouped into compressed chunks (~64KB each) for:
- Parallel decompression
- Partial loading
- Better cache locality

```rust
struct DataChunk {
    chunk_id: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    constant_count: u16,
    checksum: u32,              // CRC32 for integrity
    data: [u8; compressed_size],
}

// Within each chunk, constants serialized with postcard
struct SerializedConstant {
    // Basic info (index has name, kind, type already)
    level_params: [StringId],

    // Value (if definition/theorem/opaque)
    value: Option<ExprId>,

    // Kind-specific data
    extra: ConstantExtra,
}

enum ConstantExtra {
    None,
    Inductive(InductiveData),
    Constructor(ConstructorData),
    Recursor(RecursorData),
}
```

### 3.8 Module Metadata

Track which constants belong to which modules.

```rust
struct ModuleMetadata {
    modules: [ModuleInfo; count],
}

struct ModuleInfo {
    name_id: StringId,          // "Mathlib.Data.Nat.Basic"
    import_ids: [u16],          // Indices of imported modules
    constant_range: (u32, u32), // Start/end in constant index
    source_hash: [u8; 8],       // For staleness detection
}
```

### 3.9 Footer

```rust
struct Lean5DbFooter {
    content_hash: [u8; 32],     // SHA-256 of header + all sections
    magic_end: [u8; 8],         // "5BDNAEL\0" (reverse of header)
}
```

---

## 4. Compression Strategy

### 4.1 Algorithm Selection

**Zstandard (zstd)** chosen for:
- Fast decompression: ~1.5 GB/s on modern CPUs
- Good compression ratio: 3-5x typical
- Dictionary support: +20-30% better compression with training
- Streaming support: Can decompress chunks independently

### 4.2 Dictionary Training

Train a compression dictionary on representative Lean expressions:

```rust
// Build dictionary from Init + sample Mathlib modules
fn train_dictionary(samples: &[Vec<u8>]) -> Vec<u8> {
    zstd::dict::from_samples(samples, 112 * 1024)  // 112KB dictionary
}
```

Dictionary embedded in file header when `FLAG_ZSTD_DICT` set.

### 4.3 Chunk Sizing

- Target chunk size: 64 KB uncompressed
- Allows parallel decompression
- Good balance of compression ratio vs. random access

---

## 5. Loading Modes

### 5.1 Full Load

Load entire database into memory:

```rust
pub fn load_full(path: &Path) -> Result<Environment, Error> {
    let file = File::open(path)?;
    let header = read_header(&file)?;

    // Decompress all sections in parallel
    let (strings, exprs, index, chunks) = rayon::join4(
        || decompress_strings(&file, &header),
        || decompress_expr_pool(&file, &header),
        || decompress_index(&file, &header),
        || decompress_all_chunks(&file, &header),
    );

    // Build environment from deserialized data
    build_environment(strings?, exprs?, index?, chunks?)
}
```

### 5.2 Lazy Load — REVISED in v1.1

Load index only, deserialize constants on demand with bounded memory:

```rust
pub struct LazyDatabase {
    mmap: Mmap,
    header: Lean5DbHeader,

    // Always memory-mapped (zero-copy) — Improvement #8
    bloom_filter: BloomFilter<'static>,   // NEW: Fast negative lookup
    strings: MmapStringTable<'static>,
    expr_pool: MmapExprPool<'static>,
    index: ColumnarIndex,

    // LRU cache with size limit — Flaw #5 fix
    chunk_cache: Mutex<LruCache<u32, Arc<DecompressedChunk>>>,
    cache_size_limit: usize,  // Default: 64 chunks (~4MB)

    // Optional proof archive — Improvement #10
    proof_archive: Option<ProofArchive>,
}

impl LazyDatabase {
    pub fn get_constant(&self, name: &Name) -> Option<ConstantInfo> {
        // Fast path: bloom filter says definitely not present
        if !self.bloom_filter.maybe_contains(name) {
            return None;  // O(1) rejection
        }

        // Binary search in index
        let entry = self.index.lookup(name)?;

        // Load chunk if not cached (LRU eviction if at limit)
        let chunk = self.ensure_chunk_loaded(entry.data_chunk);

        // Deserialize constant from chunk
        chunk.deserialize_constant(entry.data_offset)
    }

    pub fn get_proof(&self, name: &Name) -> Option<Expr> {
        // Load from separate archive if proofs not in main file
        if let Some(archive) = &self.proof_archive {
            archive.get_proof(name)
        } else {
            // Proof in main file
            self.get_constant(name)?.value
        }
    }
}
```

### 5.3 Streaming Load

Process constants without holding all in memory:

```rust
pub fn stream_constants(path: &Path) -> impl Iterator<Item = ConstantInfo> {
    // Returns iterator that decompresses and yields one constant at a time
    // Memory usage: O(chunk_size) instead of O(total_size)
}
```

---

## 6. Builder Pipeline

### 6.1 Conversion Process

```
.olean files (Mathlib)
        │
        ▼
┌───────────────────┐
│  1. Parse all     │  Parallel parsing with lean5-olean
│     .olean files  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  2. Build string  │  Collect all names, sort by frequency
│     table         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  3. Build expr    │  Hash-consing for deduplication
│     pool          │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  4. Serialize     │  Group into chunks, compress
│     constants     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  5. Build index   │  Sort, create B-tree
│                   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  6. Write file    │  Header, sections, footer
│                   │
└───────────────────┘
        │
        ▼
    mathlib.lean5db
```

### 6.2 Builder API

```rust
pub struct Lean5DbBuilder {
    strings: StringTableBuilder,
    expr_pool: ExprPoolBuilder,
    constants: Vec<ParsedConstant>,
    modules: Vec<ModuleInfo>,
}

impl Lean5DbBuilder {
    pub fn new() -> Self;

    /// Add constants from a parsed .olean module
    pub fn add_module(&mut self, module: &ParsedModule);

    /// Finalize and write to file
    pub fn build(self, path: &Path, options: BuildOptions) -> Result<(), Error>;
}

pub struct BuildOptions {
    pub compression_level: i32,      // 1-22, default 3
    pub train_dictionary: bool,      // Train zstd dictionary
    pub include_values: bool,        // Include definition bodies
    pub chunk_size: usize,           // Target chunk size (default 64KB)
    pub parallel: bool,              // Use rayon for parallel processing

    // NEW in v1.1: Proof handling modes
    pub proof_mode: ProofMode,
    pub include_bloom_filter: bool,          // Strongly recommended
    pub topological_chunk_ordering: bool,    // Required for streaming (default true)
}

/// How to handle proof terms — NEW in v1.1
pub enum ProofMode {
    /// Include full proof terms in main file (largest, most complete)
    Full,

    /// Store proof commitments in main file, full proofs in separate archive
    /// Enables verification without loading full proofs
    Separate { archive_path: PathBuf },

    /// Elide proofs entirely - store hash commitment only (smallest, trusted mode)
    Committed,

    /// No proofs at all - theorems become axioms (smallest, fully trusted)
    Elided,
}
```

---

## 7. Integration with lean5-kernel

### 7.1 Environment Population

```rust
impl Environment {
    /// Load from Lean5DB file
    pub fn from_lean5db(path: &Path) -> Result<Self, Error> {
        let db = Lean5DbLoader::open(path)?;
        let mut env = Environment::default();

        // Reserve capacity upfront
        env.reserve_capacity(db.constant_count());

        // Bulk load (no type checking, trusted format)
        for constant in db.stream_constants() {
            env.register_constant_unchecked(constant);
        }

        Ok(env)
    }

    /// Lazy loading wrapper
    pub fn from_lean5db_lazy(path: &Path) -> Result<LazyEnvironment, Error>;
}
```

### 7.2 Hybrid Loading

Support loading base library from .lean5db, user code from .olean:

```rust
// Load Mathlib from pre-built .lean5db (fast)
let mut env = Environment::from_lean5db("mathlib.lean5db")?;

// Load user's project from .olean (incremental)
load_module_with_deps(&mut env, "MyProject.Main", &search_paths)?;
```

---

## 8. CLI Tool

```bash
# Convert Mathlib to .lean5db
lean5-fastload build \
    --input ~/.elan/toolchains/.../lib/lean \
    --input ~/.lake/packages/mathlib/.lake/build/lib \
    --output mathlib.lean5db \
    --train-dict \
    --compression 6

# Inspect .lean5db file
lean5-fastload info mathlib.lean5db

# Verify integrity
lean5-fastload verify mathlib.lean5db

# Benchmark loading
lean5-fastload bench mathlib.lean5db

# Extract single module (for debugging)
lean5-fastload extract mathlib.lean5db Mathlib.Data.Nat.Basic
```

---

## 9. Performance Targets

### 9.1 Benchmarks

| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Full Mathlib load | < 6s | `time lean5-fastload bench` |
| Single constant lookup | < 1µs | Lazy load benchmark |
| Memory (full load) | < 2 GB | Peak RSS |
| Memory (lazy, cold) | < 200 MB | Index + pools only |
| Build time (Mathlib) | < 10 min | One-time conversion |
| File size (Mathlib) | < 1 GB | `ls -lh` |

### 9.2 Comparison

| Metric | .olean (current) | .lean5db (target) | Improvement |
|--------|-----------------|-------------------|-------------|
| Load time | ~60s | <6s | 10x |
| Memory | ~8 GB | <2 GB | 4x |
| Constants/sec | 83K | 500K+ | 6x |
| File size | ~4 GB | <1 GB | 4x |

---

## 10. Security Considerations

### 10.1 Trust Model

- .lean5db files are **trusted** - no type checking on load
- Only load .lean5db from trusted sources (official builds)
- Checksum verification catches corruption, not malice

### 10.2 Integrity Verification

```rust
pub fn verify_integrity(path: &Path) -> Result<bool, Error> {
    let file = File::open(path)?;
    let header = read_header(&file)?;
    let footer = read_footer(&file)?;

    // Verify content hash
    let computed_hash = compute_content_hash(&file, &header)?;
    Ok(computed_hash == footer.content_hash)
}
```

### 10.3 Version Compatibility

- Major version bump = incompatible format change
- Minor version bump = backward compatible additions
- Lean version hash in header for toolchain matching

---

## 11. Future Extensions

### 11.1 Network Loading

```rust
// Load from HTTP with range requests
let env = Environment::from_lean5db_url(
    "https://releases.mathlib.org/mathlib-v4.5.0.lean5db"
)?;
```

### 11.2 Differential Updates

```rust
// Apply delta to existing .lean5db
lean5-fastload delta \
    --base mathlib-v4.4.0.lean5db \
    --patch mathlib-v4.4.0-to-v4.5.0.delta \
    --output mathlib-v4.5.0.lean5db
```

### 11.3 Proof-on-Demand

```rust
// Store proofs separately
mathlib.lean5db       # Types + definitions only (~400 MB)
mathlib.proofs.lean5db  # Proof terms (~3 GB)

// Load proofs only when needed for verification
```

---

## 12. Implementation Phases — REVISED in v1.1

Based on critical analysis, phases reorganized to address identified flaws early.

### Phase 1: Core Format v1.1 (2.5 weeks)
- Header with expanded fields (192 bytes)
- Bloom filter implementation
- LRU chunk cache (Flaw #5 fix)
- Topological ordering metadata (Flaw #7 fix)
- Basic compression with zstd

### Phase 2: Tiered Expression Pool (2 weeks)
- Tier analysis pass (frequency counting)
- Tier 0/1/2 serialization (Improvement #1)
- Memory-mapped access (Improvement #8)
- Hash-consing deduplication
- Level pool

### Phase 3: Builder with Proof Separation (2.5 weeks)
- .olean parsing integration
- Proof commitment generation (Improvement #4)
- Separate proof archive format (Improvement #10)
- Build modes: Full / Separate / Elided
- Mutual group tracking (Flaw #8 fix)
- Chunk generation with topo ordering

### Phase 4: Loader with Proofs-on-Demand (1.5 weeks)
- Full load implementation
- Lazy proof loading from archive
- Proof verification against commitments
- Environment integration

### Phase 5: Columnar Index & Optimizations (1.5 weeks)
- Column-oriented index storage (Improvement #2)
- Delta encoding for offsets
- SIMD name lookup preparation (Improvement #3)
- Bloom filter integration

### Phase 6: CLI, Testing, Benchmarks (1.5 weeks)
- CLI tool with all build modes
- Full Mathlib conversion test
- Performance regression tests
- Correctness tests (roundtrip verification)
- Documentation

**Revised Total: ~11.5 weeks** (was 10 weeks)

### Critical Path

```
Week 1-2.5:   Core Format ─────────────────┐
                                           │
Week 2.5-4.5: Expression Pool ─────────────┼───┐
                                           │   │
Week 4.5-7:   Builder + Proofs ────────────┘   │
                                               │
Week 7-8.5:   Loader ──────────────────────────┤
                                               │
Week 8.5-10:  Index Optimizations ─────────────┤
                                               │
Week 10-11.5: CLI + Testing ───────────────────┘
```

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Mathlib too large for testing | Use Init stdlib first, scale up |
| Expression dedup ratio lower than expected | Measure early in Phase 2 |
| Proof archive complexity | Start with elided mode, add archive later |
| Performance regression | Continuous benchmarking from Phase 1 |

---

## 13. Dependencies

```toml
[dependencies]
zstd = "0.13"           # Compression
memmap2 = "0.9"         # Memory-mapped files
postcard = "1.0"        # Fast serialization
rayon = "1.8"           # Parallel processing
sha2 = "0.10"           # Checksums
crc32fast = "1.3"       # Chunk checksums
lean5-kernel = { path = "../lean5-kernel" }
lean5-olean = { path = "../lean5-olean" }
```

---

## 14. Open Questions

1. **Proof elision default?** Should we include proofs by default or make them optional?
2. **Multiple files vs single?** One big file or per-module files?
3. **Compression level tradeoff?** Higher compression = slower build, faster transfer
4. **Cache invalidation?** How to detect when .lean5db is stale?
5. **IDE integration?** How do language servers use this?

---

## Appendix A: Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| SQLite | Mature, queryable | Overhead, not cache-friendly |
| FlatBuffers | Zero-copy | Complex schema evolution |
| Cap'n Proto | Fast, zero-copy | C++ dependency |
| Custom (this) | Optimized for use case | Development cost |
| Parquet | Columnar, compressed | Wrong data model |

Custom format chosen for maximum performance and minimal dependencies.

---

## Appendix B: Expression Frequency Analysis

Preliminary analysis of Init standard library:

| Expression Pattern | Frequency | Dedup Potential |
|-------------------|-----------|-----------------|
| `Nat` | 50,000+ | High |
| `Prop` | 30,000+ | High |
| `Type u` | 25,000+ | High |
| `∀ x, ...` | 100,000+ | Medium (structure varies) |
| `List α` | 10,000+ | High |
| `α → β` | 80,000+ | Medium |

Expected overall deduplication: 5-10x reduction in expression storage.

---

*End of Design Document*
