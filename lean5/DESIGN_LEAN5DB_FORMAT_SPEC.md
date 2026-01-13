# Lean5DB Format Specification: Serialization Framework Analysis

**Status:** Draft v1.0
**Date:** 2026-01-06
**Decision Required:** Custom binary vs. Protobuf vs. FlatBuffers vs. Cap'n Proto

---

## 1. Requirements Recap

| Requirement | Priority | Notes |
|-------------|----------|-------|
| Fast random access | HIGH | Lazy loading requires O(1) lookups |
| Memory-mapped access | HIGH | Zero-copy for large files |
| Compression support | HIGH | 4GB → <1GB |
| Schema evolution | MEDIUM | Format updates without breaking |
| Cross-language support | MEDIUM | Python, C++ tooling desired |
| Small file size | MEDIUM | Compression handles most of this |
| Simplicity | MEDIUM | Less code = fewer bugs |
| Stable specification | HIGH | Files must be readable for years |

---

## 2. Framework Comparison

### 2.1 Custom Binary (Current Proposal)

```
[Header: 192 bytes, fixed layout]
[Bloom filter: ~1MB, raw bytes]
[String table: varint lengths + UTF-8]
[Expression pool: tagged union encoding]
[Index: sorted array of fixed-size entries]
[Data chunks: zstd-compressed]
[Footer: checksums]
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Random access | ⭐⭐⭐⭐⭐ | Full control over layout |
| Mmap support | ⭐⭐⭐⭐⭐ | Design for it from start |
| Compression | ⭐⭐⭐⭐⭐ | Integrate zstd directly |
| Schema evolution | ⭐⭐ | Manual versioning, error-prone |
| Cross-language | ⭐⭐ | Must write parser for each language |
| Specification | ⭐⭐ | Must document everything ourselves |
| Maintenance | ⭐⭐ | Parser bugs are our problem |

**Risk:** Format drift, undocumented edge cases, parser bugs.

---

### 2.2 Protocol Buffers (protobuf)

```protobuf
syntax = "proto3";

message Lean5DbFile {
  Header header = 1;
  StringTable string_table = 2;
  ExprPool expr_pool = 3;
  repeated Constant constants = 4;
  repeated ModuleInfo modules = 5;
}

message Constant {
  string name = 1;
  ConstantKind kind = 2;
  repeated string level_params = 3;
  Expr type = 4;
  optional Expr value = 5;
  bytes hash = 6;
}

message Expr {
  oneof expr {
    uint32 bvar = 1;
    Level sort = 2;
    ConstRef const_ref = 3;
    App app = 4;
    Lam lam = 5;
    Pi pi = 6;
    // ...
  }
}
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Random access | ⭐⭐ | Sequential format, must parse to find data |
| Mmap support | ⭐ | Not designed for random access |
| Compression | ⭐⭐⭐ | External compression required |
| Schema evolution | ⭐⭐⭐⭐⭐ | Field numbers, optional fields, excellent |
| Cross-language | ⭐⭐⭐⭐⭐ | Official support for 10+ languages |
| Specification | ⭐⭐⭐⭐⭐ | .proto files ARE the spec |
| Maintenance | ⭐⭐⭐⭐⭐ | Google maintains protoc |

**Problem:** Not suitable for lazy loading—must parse entire file to access one constant.

---

### 2.3 FlatBuffers

```flatbuffers
namespace lean5db;

table Lean5DbFile {
  header: Header;
  bloom_filter: [ubyte];
  string_table: StringTable;
  expr_pool: ExprPool;
  constant_index: [ConstantEntry];
  modules: [ModuleInfo];
  content_hash: [ubyte:32];
}

table ConstantEntry {
  name_id: uint32;
  kind: ConstantKind;
  type_expr_id: uint32;
  value_expr_id: uint32;
  chunk_id: uint32;
  chunk_offset: uint32;
}

table Expr {
  kind: ExprKind;
  data: ExprData;
}

union ExprData {
  BVar,
  Sort,
  Const,
  App,
  Lam,
  Pi,
  Let,
  Lit,
  Proj,
}
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Random access | ⭐⭐⭐⭐⭐ | vtables enable O(1) field access |
| Mmap support | ⭐⭐⭐⭐⭐ | Designed for zero-copy access |
| Compression | ⭐⭐⭐ | Must compress externally, loses random access |
| Schema evolution | ⭐⭐⭐⭐ | Add fields, deprecate, good support |
| Cross-language | ⭐⭐⭐⭐ | Rust, C++, Python, JS, Go, etc. |
| Specification | ⭐⭐⭐⭐⭐ | .fbs files ARE the spec |
| Maintenance | ⭐⭐⭐⭐ | Google maintains, active community |

**Best fit for:** Index, metadata, header—things that need random access.
**Problem:** Compression breaks zero-copy. Large expressions still need chunking.

---

### 2.4 Cap'n Proto

```capnp
@0xabcdef1234567890;

struct Lean5DbFile {
  header @0 :Header;
  bloomFilter @1 :Data;
  stringTable @2 :List(Text);
  exprPool @3 :List(Expr);
  constants @4 :List(ConstantEntry);
  modules @5 :List(ModuleInfo);
}

struct Expr {
  union {
    bvar @0 :UInt32;
    sort @1 :Level;
    const @2 :ConstRef;
    app @3 :App;
    lam @4 :Lam;
    pi @5 :Pi;
  }
}
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Random access | ⭐⭐⭐⭐⭐ | Pointer-based, excellent random access |
| Mmap support | ⭐⭐⭐⭐⭐ | Core design goal |
| Compression | ⭐⭐⭐ | "Packed" mode, but loses random access |
| Schema evolution | ⭐⭐⭐⭐⭐ | Excellent, annotation-based |
| Cross-language | ⭐⭐⭐ | C++ primary, Rust via capnp crate |
| Specification | ⭐⭐⭐⭐⭐ | .capnp files ARE the spec |
| Maintenance | ⭐⭐⭐⭐ | Active, but smaller community than protobuf |

**Concern:** Rust crate (`capnp`) requires C++ capnp library for code generation.

---

### 2.5 Apache Thrift

| Aspect | Rating | Notes |
|--------|--------|-------|
| Random access | ⭐⭐ | Stream-oriented |
| Mmap support | ⭐ | Not designed for it |
| Compression | ⭐⭐⭐ | Multiple protocols available |
| Schema evolution | ⭐⭐⭐⭐ | Field IDs, optional fields |
| Cross-language | ⭐⭐⭐⭐⭐ | Excellent multi-language support |
| Specification | ⭐⭐⭐⭐ | .thrift files |
| Maintenance | ⭐⭐⭐⭐ | Apache foundation |

**Problem:** Same as protobuf—not suitable for random access / lazy loading.

---

### 2.6 MessagePack / CBOR / postcard

Compact binary serialization without schemas:

| Aspect | Rating | Notes |
|--------|--------|-------|
| Random access | ⭐⭐ | Must parse sequentially |
| Mmap support | ⭐⭐ | Not designed for it |
| Compression | ⭐⭐⭐⭐ | Already compact |
| Schema evolution | ⭐⭐ | Ad-hoc, no schema |
| Cross-language | ⭐⭐⭐⭐ | Wide support |
| Specification | ⭐⭐⭐ | Format spec exists, schema is code |
| Maintenance | ⭐⭐⭐⭐ | Stable, simple |

**Best fit for:** Serializing data within compressed chunks.

---

## 3. Recommendation: Hybrid Approach

**Use the right tool for each layer:**

```
┌─────────────────────────────────────────────────────────────┐
│                    LEAN5DB FILE                             │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: FILE CONTAINER                                    │
│  Format: Custom header (192 bytes) + section offsets        │
│  Rationale: Maximum control, simple to parse                │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: INDEX & METADATA                                  │
│  Format: FlatBuffers                                        │
│  Rationale: Random access, mmap, schema evolution           │
│  Includes: String table, constant index, module info        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: EXPRESSION POOL                                   │
│  Format: FlatBuffers (uncompressed)                         │
│  Rationale: Random access to expressions by ID              │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: CONSTANT DATA CHUNKS                              │
│  Format: postcard (inside zstd-compressed chunks)           │
│  Rationale: Maximum compression, sequential access within   │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: PROOF ARCHIVE (optional)                          │
│  Format: postcard + zstd                                    │
│  Rationale: Rarely accessed, compression more important     │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Why Hybrid?

| Layer | Access Pattern | Best Format |
|-------|----------------|-------------|
| Header | Fixed offset | Custom binary |
| Index | Random lookup by name | FlatBuffers |
| Expressions | Random access by ID | FlatBuffers |
| Constant data | Sequential within chunk | postcard + zstd |
| Proofs | Rarely accessed | postcard + zstd |

### 3.2 Schema Files

**`lean5db_index.fbs`** (FlatBuffers schema for index):
```flatbuffers
namespace lean5db;

// File format version
file_identifier "L5DB";
file_extension "lean5db";

// Root table
table Lean5DbIndex {
  // Format identification
  version_major: uint16;
  version_minor: uint16;
  flags: uint32;

  // Content hash (equivalence proof)
  library_hash: [ubyte:32];

  // String interning table
  strings: [String];

  // Constant index (sorted by name for binary search)
  constants: [ConstantEntry];

  // Module metadata
  modules: [ModuleInfo];

  // Verification
  module_hashes: [ModuleHash];
}

table String {
  value: string;
}

table ConstantEntry {
  name_id: uint32;             // Index into strings
  kind: ConstantKind;
  level_param_ids: [uint32];   // Indices into strings
  type_expr_id: uint32;        // Index into expression pool
  value_expr_id: uint32;       // 0 if none
  module_id: uint16;
  chunk_id: uint32;
  chunk_offset: uint32;
  content_hash: [ubyte:32];
}

enum ConstantKind: ubyte {
  Axiom = 0,
  Definition = 1,
  Theorem = 2,
  Opaque = 3,
  Inductive = 4,
  Constructor = 5,
  Recursor = 6,
  Quotient = 7,
}

table ModuleInfo {
  name_id: uint32;
  import_ids: [uint16];
  constant_start: uint32;
  constant_count: uint32;
}

table ModuleHash {
  module_id: uint16;
  hash: [ubyte:32];
}

root_type Lean5DbIndex;
```

**`lean5db_expr.fbs`** (FlatBuffers schema for expressions):
```flatbuffers
namespace lean5db;

table ExprPool {
  expressions: [Expr];
  levels: [Level];
}

table Expr {
  data: ExprData;
}

union ExprData {
  BVar,
  Sort,
  Const,
  App,
  Lam,
  Pi,
  Let,
  Lit,
  Proj,
  MData,
}

table BVar { index: uint32; }
table Sort { level_id: uint32; }
table Const { name_id: uint32; level_ids: [uint32]; }
table App { func_id: uint32; arg_id: uint32; }
table Lam { binder_info: BinderInfo; type_id: uint32; body_id: uint32; }
table Pi { binder_info: BinderInfo; domain_id: uint32; codomain_id: uint32; }
table Let { type_id: uint32; value_id: uint32; body_id: uint32; }
table Proj { struct_name_id: uint32; index: uint32; expr_id: uint32; }
table MData { inner_id: uint32; }

table Lit {
  data: LitData;
}

union LitData {
  NatLit,
  StringLit,
}

table NatLit { value: uint64; }  // For small nats
table StringLit { value: string; }

enum BinderInfo: ubyte {
  Default = 0,
  Implicit = 1,
  StrictImplicit = 2,
  InstImplicit = 3,
}

table Level {
  data: LevelData;
}

union LevelData {
  Zero,
  Succ,
  Max,
  IMax,
  Param,
}

table Zero {}
table Succ { inner_id: uint32; }
table Max { left_id: uint32; right_id: uint32; }
table IMax { left_id: uint32; right_id: uint32; }
table Param { name_id: uint32; }
```

### 3.3 File Layout (Final)

```
┌────────────────────────────────────────────────────────────┐
│  LEAN5DB FILE v2.0 (Hybrid Format)                        │
├────────────────────────────────────────────────────────────┤
│  [0x0000] Custom Header (192 bytes)                       │
│    - Magic: "LEAN5DB2"                                    │
│    - Version, flags                                       │
│    - Section offsets and sizes                            │
│    - Library content hash                                 │
├────────────────────────────────────────────────────────────┤
│  [0x00C0] Bloom Filter (raw bytes, ~1MB)                  │
├────────────────────────────────────────────────────────────┤
│  [varies] FlatBuffers Index Section                       │
│    - Root: Lean5DbIndex                                   │
│    - Strings, constants, modules                          │
│    - NOT compressed (for mmap random access)              │
├────────────────────────────────────────────────────────────┤
│  [varies] FlatBuffers Expression Pool                     │
│    - Root: ExprPool                                       │
│    - All expressions and levels                           │
│    - NOT compressed (for mmap random access)              │
├────────────────────────────────────────────────────────────┤
│  [varies] Chunk Directory (FlatBuffers)                   │
│    - Chunk offsets, sizes, hashes                         │
├────────────────────────────────────────────────────────────┤
│  [varies] Data Chunks (zstd compressed)                   │
│    - Each chunk: postcard-serialized constants            │
│    - Contains values, inductive data, recursor rules      │
├────────────────────────────────────────────────────────────┤
│  [varies] Footer (64 bytes)                               │
│    - File checksum                                        │
│    - Magic end marker                                     │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Benefits of Hybrid Approach

### 4.1 Stable Specification

The `.fbs` files ARE the specification:
- Machine-readable
- Language-agnostic
- Version-controlled
- Generate parsers automatically

### 4.2 Cross-Language Support

FlatBuffers generates code for:
- Rust (`flatbuffers` crate)
- C++ (native)
- Python (`flatbuffers` pip package)
- TypeScript/JavaScript
- Go, Java, C#, Swift, etc.

### 4.3 Schema Evolution

FlatBuffers supports:
- Adding new fields (with defaults)
- Deprecating fields
- Reordering fields (field IDs are stable)

```flatbuffers
table ConstantEntry {
  name_id: uint32;
  kind: ConstantKind;
  // ... existing fields ...

  // Added in v2.1 - old readers ignore, new readers get default
  source_location: SourceLoc;
  doc_string_id: uint32;
}
```

### 4.4 Tooling

```bash
# Generate Rust code from schema
flatc --rust -o src/generated/ lean5db_index.fbs lean5db_expr.fbs

# Validate a file against schema
flatc --binary --schema lean5db_index.fbs

# Convert to JSON for debugging
flatc --json --strict-json lean5db_index.fbs -- index_section.bin
```

---

## 5. Trade-offs Accepted

| Trade-off | Mitigation |
|-----------|------------|
| FlatBuffers adds ~200KB to binary | Acceptable for tooling |
| Expression pool can't be compressed | Deduplication already shrinks it |
| Two serialization formats (FlatBuffers + postcard) | Clear boundary between layers |
| Build dependency on `flatc` | Only needed for schema changes |

---

## 6. Alternative: Pure FlatBuffers

If we want maximum simplicity, we could use FlatBuffers for everything:

```flatbuffers
table Lean5DbFile {
  header: Header;
  bloom_filter: [ubyte];
  index: Lean5DbIndex;
  expr_pool: ExprPool;

  // Constant data inline (not chunked)
  constant_values: [ConstantValue];
}
```

**Pros:**
- Single format
- Simpler mental model
- No chunk management

**Cons:**
- No compression (or lose random access)
- ~4GB file instead of ~1GB
- Slower to load (more data to mmap)

**Verdict:** Worth considering for v1, but hybrid is better for Mathlib scale.

---

## 7. Alternative: Pure Custom + postcard

Skip FlatBuffers entirely, use postcard for everything with custom index:

**Pros:**
- No external schema tools
- Simpler build
- Smaller binary

**Cons:**
- Must maintain our own random-access index format
- No schema evolution guarantees
- Must document format ourselves
- Cross-language requires manual work

**Verdict:** Higher risk, more maintenance burden.

---

## 8. Recommendation Summary

| Component | Format | Rationale |
|-----------|--------|-----------|
| File container | Custom 192-byte header | Simple, fixed layout |
| Bloom filter | Raw bytes | No structure needed |
| Index | FlatBuffers | Random access, schema evolution |
| Expression pool | FlatBuffers | Random access by ID |
| Constant data | postcard + zstd | Compression, sequential access |
| Proofs | postcard + zstd | Rarely accessed, max compression |

**Primary specification:** FlatBuffers `.fbs` files
**Secondary specification:** This document + postcard struct definitions

---

## 9. Migration Path

### v1.0 (Current proposal)
- Custom binary throughout
- Works, but undocumented edge cases

### v2.0 (This proposal)
- Hybrid: FlatBuffers index + postcard chunks
- Stable, documented, cross-language

### Migration
```bash
# Convert v1 to v2
lean5-fastload migrate mathlib-v1.lean5db -o mathlib-v2.lean5db

# Verify equivalence
lean5-fastload check-equivalent mathlib-v1.lean5db mathlib-v2.lean5db
```

---

## 10. Final Dependencies

```toml
[dependencies]
flatbuffers = "24.3"       # FlatBuffers runtime
postcard = "1.0"           # Compact serialization for chunks
zstd = "0.13"              # Compression
blake3 = "1.5"             # Hashing
memmap2 = "0.9"            # Memory mapping
```

Build-time only:
```bash
# Install FlatBuffers compiler
brew install flatbuffers  # macOS
apt install flatbuffers-compiler  # Linux
```

---

*This document specifies the stable format. The `.fbs` files are the authoritative schema.*
