# Lean5DB Verification and Equivalence Proofs

**Status:** Draft v1.0
**Date:** 2026-01-06
**Purpose:** Ensure .lean5db files are provably equivalent to source .olean files

---

## 1. The Verification Problem

### 1.1 Threat Model

What can go wrong during .olean → .lean5db conversion?

| Threat | Likelihood | Impact | Detection |
|--------|------------|--------|-----------|
| Bit flip during I/O | Low | Single constant wrong | Checksums |
| Parser bug | Medium | Systematic errors | Hard to detect |
| Serializer bug | Medium | Data loss/corruption | Hard to detect |
| Expression dedup collision | Low | Wrong sharing | Hash collision |
| Truncated write | Medium | Missing constants | Size check |
| Compression corruption | Low | Chunk unreadable | Decompression fails |
| Version mismatch | Medium | Wrong interpretation | Silent corruption |
| Malicious tampering | Low | Arbitrary | Signatures |

### 1.2 Key Insight

**The .lean5db file is a lossy transformation** - we may:
- Reorder constants
- Deduplicate expressions
- Elide proofs
- Change representation

But the **semantic content must be equivalent**. We need a way to prove this.

---

## 2. Canonical Content Hashing

### 2.1 Design: Semantic Hash Tree

```
                    ┌─────────────────┐
                    │  Library Hash   │  ← Root: proves entire library
                    │   (32 bytes)    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │ Module Hash │   │ Module Hash │   │ Module Hash │
    │  Init.Core  │   │ Init.Data   │   │  Mathlib.*  │
    └──────┬──────┘   └─────────────┘   └─────────────┘
           │
    ┌──────┴──────┬─────────────┐
    │             │             │
┌───▼───┐    ┌────▼────┐   ┌────▼────┐
│Const H│    │Const H  │   │Const H  │
│ Nat   │    │Nat.add  │   │Nat.rec  │
└───────┘    └─────────┘   └─────────┘
```

### 2.2 Constant Hash Definition

Each constant has a **canonical hash** computed from its semantic content:

```rust
/// Compute canonical hash for a constant.
/// This hash is identical whether computed from .olean or .lean5db.
fn canonical_constant_hash(c: &ConstantInfo) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();

    // 1. Hash the fully qualified name (canonical string)
    hasher.update(b"NAME:");
    hasher.update(c.name.to_canonical_string().as_bytes());

    // 2. Hash the kind
    hasher.update(b"KIND:");
    hasher.update(&[constant_kind_tag(&c.kind)]);

    // 3. Hash level parameters (sorted for consistency)
    hasher.update(b"LEVELS:");
    let mut level_names: Vec<_> = c.level_params.iter()
        .map(|n| n.to_canonical_string())
        .collect();
    level_names.sort();  // Canonical ordering
    for name in &level_names {
        hasher.update(name.as_bytes());
        hasher.update(b"\0");
    }

    // 4. Hash the type (canonicalized expression)
    hasher.update(b"TYPE:");
    hasher.update(&canonical_expr_hash(&c.type_));

    // 5. Hash the value if present (for definitions/theorems)
    if let Some(value) = &c.value {
        hasher.update(b"VALUE:");
        hasher.update(&canonical_expr_hash(value));
    }

    // 6. Kind-specific data
    hash_kind_specific(&mut hasher, c);

    hasher.finalize().into()
}
```

### 2.3 Canonical Expression Hash

Expressions must be hashed in a **canonical form** that is:
- Independent of internal pointer addresses
- Independent of expression pool IDs
- Stable across serialization formats

```rust
/// Hash an expression in canonical form.
/// De Bruijn indices make this well-defined.
fn canonical_expr_hash(expr: &Expr) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hash_expr_recursive(&mut hasher, expr);
    hasher.finalize().into()
}

fn hash_expr_recursive(hasher: &mut blake3::Hasher, expr: &Expr) {
    match expr {
        Expr::BVar(idx) => {
            hasher.update(&[0x01]);  // Tag
            hasher.update(&idx.to_le_bytes());
        }
        Expr::Sort(level) => {
            hasher.update(&[0x02]);
            hash_level_recursive(hasher, level);
        }
        Expr::Const(name, levels) => {
            hasher.update(&[0x03]);
            hasher.update(name.to_canonical_string().as_bytes());
            hasher.update(b"\0");
            hasher.update(&(levels.len() as u32).to_le_bytes());
            for level in levels {
                hash_level_recursive(hasher, level);
            }
        }
        Expr::App(f, a) => {
            hasher.update(&[0x04]);
            hash_expr_recursive(hasher, f);
            hash_expr_recursive(hasher, a);
        }
        Expr::Lam(info, ty, body) => {
            hasher.update(&[0x05]);
            hasher.update(&[binder_info_tag(*info)]);
            hash_expr_recursive(hasher, ty);
            hash_expr_recursive(hasher, body);
        }
        Expr::Pi(info, domain, codomain) => {
            hasher.update(&[0x06]);
            hasher.update(&[binder_info_tag(*info)]);
            hash_expr_recursive(hasher, domain);
            hash_expr_recursive(hasher, codomain);
        }
        Expr::Let(ty, val, body) => {
            hasher.update(&[0x07]);
            hash_expr_recursive(hasher, ty);
            hash_expr_recursive(hasher, val);
            hash_expr_recursive(hasher, body);
        }
        Expr::Lit(lit) => {
            hasher.update(&[0x08]);
            match lit {
                Literal::Nat(n) => {
                    hasher.update(&[0x00]);
                    // Hash big integer in canonical form
                    hasher.update(&n.to_le_bytes_canonical());
                }
                Literal::String(s) => {
                    hasher.update(&[0x01]);
                    hasher.update(s.as_bytes());
                }
            }
        }
        Expr::Proj(struct_name, idx, inner) => {
            hasher.update(&[0x09]);
            hasher.update(struct_name.to_canonical_string().as_bytes());
            hasher.update(b"\0");
            hasher.update(&idx.to_le_bytes());
            hash_expr_recursive(hasher, inner);
        }
        // ... other cases
    }
}
```

### 2.4 Module Hash

```rust
/// Hash for an entire module - combines all constant hashes.
/// Order-independent: same constants in any order = same hash.
fn canonical_module_hash(constants: &[ConstantInfo]) -> [u8; 32] {
    // Collect all constant hashes
    let mut const_hashes: Vec<[u8; 32]> = constants
        .iter()
        .map(canonical_constant_hash)
        .collect();

    // Sort for order independence
    const_hashes.sort();

    // Combine into module hash
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"MODULE:");
    hasher.update(&(const_hashes.len() as u64).to_le_bytes());
    for h in &const_hashes {
        hasher.update(h);
    }
    hasher.finalize().into()
}
```

### 2.5 Library Hash

```rust
/// Root hash for entire library - combines all module hashes.
fn canonical_library_hash(modules: &[(String, [u8; 32])]) -> [u8; 32] {
    let mut sorted: Vec<_> = modules.iter().collect();
    sorted.sort_by_key(|(name, _)| name);

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"LIBRARY:");
    hasher.update(&(sorted.len() as u64).to_le_bytes());
    for (name, hash) in &sorted {
        hasher.update(name.as_bytes());
        hasher.update(b"\0");
        hasher.update(*hash);
    }
    hasher.finalize().into()
}
```

---

## 3. Verification Protocol

### 3.1 Build-Time Verification

```
┌─────────────────────────────────────────────────────────────────┐
│                     BUILD PROCESS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Parse .olean files                                          │
│          │                                                      │
│          ▼                                                      │
│  2. Compute canonical hashes (per-constant, per-module)         │
│          │                                                      │
│          ▼                                                      │
│  3. Build .lean5db with expression dedup, compression, etc.     │
│          │                                                      │
│          ▼                                                      │
│  4. Re-load .lean5db and compute hashes again                   │
│          │                                                      │
│          ▼                                                      │
│  5. VERIFY: source_hash == roundtrip_hash                       │
│          │                                                      │
│          ├─── Match ───▶ Store hash in .lean5db footer          │
│          │                                                      │
│          └─── Mismatch ──▶ BUILD FAILURE (bug detected!)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Load-Time Verification

```rust
pub enum VerificationLevel {
    /// No verification - trust the file completely
    None,

    /// Verify file checksums only (fast)
    Checksums,

    /// Verify stored content hash matches recomputed hash (slower)
    ContentHash,

    /// Verify against original .olean files (slowest, most thorough)
    FullRoundtrip,
}

impl Lean5DbLoader {
    pub fn load_verified(
        path: &Path,
        level: VerificationLevel,
    ) -> Result<(Environment, VerificationReport), Error> {
        let db = Self::open(path)?;

        let report = match level {
            VerificationLevel::None => VerificationReport::skipped(),

            VerificationLevel::Checksums => {
                db.verify_all_checksums()?
            }

            VerificationLevel::ContentHash => {
                let stored_hash = db.header.content_hash;
                let computed_hash = db.compute_library_hash()?;
                if stored_hash != computed_hash {
                    return Err(Error::HashMismatch { stored_hash, computed_hash });
                }
                VerificationReport::hash_verified(computed_hash)
            }

            VerificationLevel::FullRoundtrip => {
                // Load original .olean files and compare
                db.verify_against_olean_sources()?
            }
        };

        let env = db.load_all()?;
        Ok((env, report))
    }
}
```

### 3.3 Incremental Verification

For large libraries, full verification is expensive. Use Merkle tree properties:

```rust
/// Verify only specific modules have correct hashes.
pub fn verify_modules(
    db: &Lean5DbLoader,
    module_names: &[&str],
) -> Result<(), VerificationError> {
    for name in module_names {
        let stored_hash = db.get_module_hash(name)?;
        let computed_hash = db.compute_module_hash(name)?;

        if stored_hash != computed_hash {
            return Err(VerificationError::ModuleMismatch {
                module: name.to_string(),
                stored: stored_hash,
                computed: computed_hash,
            });
        }
    }
    Ok(())
}
```

---

## 4. Extended File Format for Verification

### 4.1 Hash Section

Add a new section to store the hash tree:

```rust
struct HashSection {
    // Root hash (proves everything)
    library_hash: [u8; 32],

    // Module hashes (for incremental verification)
    module_count: u32,
    module_hashes: [ModuleHashEntry; module_count],

    // Optional: per-constant hashes (for fine-grained debugging)
    flags: u32,  // FLAG_HAS_CONSTANT_HASHES
    constant_hashes: Option<[u8; 32 * constant_count]>,
}

struct ModuleHashEntry {
    name_id: StringId,
    hash: [u8; 32],
    constant_range: (u32, u32),  // Which constants belong to this module
}
```

### 4.2 Header Extension

```rust
struct Lean5DbHeader {
    // ... existing fields ...

    // Verification section (NEW)
    hash_section_offset: u64,
    hash_section_size: u32,
    verification_flags: u32,
}

const VFLAG_HAS_MODULE_HASHES: u32    = 0x0001;
const VFLAG_HAS_CONSTANT_HASHES: u32  = 0x0002;
const VFLAG_BUILD_VERIFIED: u32       = 0x0004;  // Build did roundtrip verify
const VFLAG_SIGNED: u32               = 0x0008;  // Has cryptographic signature
```

---

## 5. Additional Flaws Found (11-20)

### Flaw 11: No Format Migration Path

**Problem:** If we need to change the format, how do we migrate existing files?

**Fix:** Add format version negotiation:
```rust
struct FormatVersion {
    major: u16,  // Breaking changes
    minor: u16,  // Additive changes

    // Minimum version that can read this file
    min_reader_major: u16,
    min_reader_minor: u16,
}
```

### Flaw 12: Single Point of Failure (Header)

**Problem:** Corrupted header = entire file unusable.

**Fix:** Redundant headers:
```
[Header at offset 0]
[... data ...]
[Header copy at offset (file_size - 256)]
[Footer]
```

### Flaw 13: No Streaming Write Support

**Problem:** Must know all content before writing header.

**Fix:** Write placeholder header, append data, seek back and update header.

### Flaw 14: Chunk Boundaries Split Mutual Groups

**Problem:** Mutual inductives must be registered together but may span chunks.

**Fix:** Add `mutual_group_id` to constants, ensure all constants in a group are in same chunk or consecutive chunks.

### Flaw 15: No Partial Library Support

**Problem:** Can't easily create "Mathlib minus module X".

**Fix:** Module dependency graph in metadata, lazy loading naturally supports this.

### Flaw 16: Expression Pool Size Unknown Upfront

**Problem:** Expression deduplication requires seeing all expressions first.

**Fix:** Two-pass build: first pass collects expressions, second pass writes.

### Flaw 17: No Custom Attributes/Extensions

**Problem:** Lean has extensible attributes, we don't preserve them.

**Fix:** Add extension data section with opaque blobs per constant.

### Flaw 18: TOCTOU for Memory-Mapped Files

**Problem:** File could change between mmap and access.

**Fix:** Lock file during load, or verify checksums after loading.

### Flaw 19: No Provenance/Signatures

**Problem:** Can't verify who built the file or if it's official.

**Fix:** Optional Ed25519 signature section:
```rust
struct SignatureSection {
    signer_public_key: [u8; 32],
    signature: [u8; 64],
    signed_hash: [u8; 32],  // What was signed (library_hash)
    timestamp: u64,
}
```

### Flaw 20: Dictionary Not Shareable

**Problem:** Each file has its own compression dictionary.

**Fix:** Support external dictionary files:
```rust
enum DictionarySource {
    Embedded,                          // In this file
    External { hash: [u8; 32] },       // Load from lean5db-dict-{hash}.dict
    WellKnown { id: u32 },             // Standard dictionaries (mathlib-v4, etc.)
}
```

---

## 6. Additional Improvements (11-20)

### Improvement 11: Merkle Tree for Chunk Integrity

**Benefit:** Verify any subset of chunks without reading entire file.

```rust
struct ChunkMerkleTree {
    // Leaf hashes (one per chunk)
    leaves: Vec<[u8; 32]>,

    // Internal nodes (binary tree)
    nodes: Vec<[u8; 32]>,

    // Root
    root: [u8; 32],
}

impl ChunkMerkleTree {
    /// Generate proof that chunk N has correct hash
    fn proof_for_chunk(&self, n: usize) -> MerkleProof;

    /// Verify chunk against proof
    fn verify_chunk(&self, n: usize, chunk_hash: [u8; 32], proof: &MerkleProof) -> bool;
}
```

### Improvement 12: Content-Addressable Expression Storage

**Benefit:** Automatic deduplication, integrity verification built-in.

```rust
struct ContentAddressedExprPool {
    // Expressions keyed by their hash
    exprs: HashMap<[u8; 32], SerializedExpr>,
}

// Reference an expression by its hash
type ExprRef = [u8; 32];
```

### Improvement 13: Semantic Versioning for Format

```rust
// Example version compatibility matrix
//
// File v1.0 readable by: reader >= v1.0
// File v1.1 readable by: reader >= v1.0 (additive)
// File v2.0 readable by: reader >= v2.0 (breaking)

fn can_read(reader_version: Version, file_version: Version) -> bool {
    reader_version.major == file_version.major &&
    reader_version.minor >= file_version.minor
}
```

### Improvement 14: Redundant Headers for Recovery

```rust
struct FileLayout {
    primary_header: Header,      // Offset 0
    // ... sections ...
    backup_header: Header,       // Offset (size - 256)
    footer: Footer,              // Offset (size - 64)
}

fn read_header(file: &File) -> Result<Header, Error> {
    match Header::read_at(file, 0) {
        Ok(h) => Ok(h),
        Err(_) => {
            // Try backup header
            let size = file.metadata()?.len();
            Header::read_at(file, size - 256)
        }
    }
}
```

### Improvement 15: Expression Normalization

**Benefit:** Equivalent expressions get same hash even with syntactic differences.

```rust
/// Normalize expression to canonical form before hashing.
fn normalize_expr(expr: &Expr) -> Expr {
    match expr {
        // η-reduce: λ x. f x → f (when x not free in f)
        Expr::Lam(_, _, body) if is_eta_reducible(body) => {
            normalize_expr(&eta_reduce(body))
        }

        // β-reduce obvious redexes
        Expr::App(Expr::Lam(_, _, body), arg) => {
            normalize_expr(&substitute(body, arg))
        }

        // Recursively normalize
        Expr::App(f, a) => {
            Expr::App(
                Arc::new(normalize_expr(f)),
                Arc::new(normalize_expr(a)),
            )
        }

        // ... etc
        _ => expr.clone()
    }
}
```

### Improvement 16: Parallel Verification

```rust
/// Verify all chunks in parallel using rayon.
fn verify_all_chunks_parallel(db: &Lean5DbLoader) -> Result<(), Error> {
    let chunk_count = db.header.chunk_count;

    (0..chunk_count)
        .into_par_iter()
        .try_for_each(|i| {
            let stored_hash = db.get_chunk_hash(i)?;
            let computed_hash = db.compute_chunk_hash(i)?;

            if stored_hash != computed_hash {
                Err(Error::ChunkHashMismatch { chunk: i })
            } else {
                Ok(())
            }
        })
}
```

### Improvement 17: Differential Loading

**Benefit:** When updating Mathlib, only load changed constants.

```rust
struct DiffLoader {
    base: Lean5DbLoader,      // Previous version
    updated: Lean5DbLoader,   // New version
}

impl DiffLoader {
    fn load_diff(&self) -> DiffResult {
        let base_hashes = self.base.constant_hashes();
        let new_hashes = self.updated.constant_hashes();

        DiffResult {
            added: new_hashes.keys()
                .filter(|k| !base_hashes.contains_key(*k))
                .collect(),
            removed: base_hashes.keys()
                .filter(|k| !new_hashes.contains_key(*k))
                .collect(),
            modified: new_hashes.iter()
                .filter(|(k, v)| base_hashes.get(*k) != Some(*v))
                .map(|(k, _)| k)
                .collect(),
        }
    }
}
```

### Improvement 18: Type-Only Mode

**Benefit:** For many use cases (autocomplete, type checking), values aren't needed.

```rust
pub struct LoadOptions {
    pub load_types: bool,       // Always true
    pub load_values: bool,      // Set false for type-only mode
    pub load_proofs: bool,      // Separate from values
}

// Type-only mode: ~60% less data to decompress
```

### Improvement 19: Front-Coded Name Index

**Benefit:** Names like `Mathlib.Data.Nat.Basic.*` share long prefixes.

```rust
struct FrontCodedIndex {
    // First name stored fully
    first: String,

    // Subsequent names: (prefix_len, suffix)
    // "Mathlib.Data.Nat.Basic.add" → (22, "add")
    // "Mathlib.Data.Nat.Basic.mul" → (22, "mul")
    entries: Vec<(u16, String)>,
}

// Compression benefit: ~40% smaller index for hierarchical names
```

### Improvement 20: Adaptive Chunk Sizing

**Benefit:** Optimize for access patterns.

```rust
struct AdaptiveChunking {
    // Frequently accessed constants (Init, core types) in small chunks
    hot_chunk_size: usize,      // 16 KB

    // Rarely accessed constants (obscure theorems) in large chunks
    cold_chunk_size: usize,     // 256 KB

    // Classification based on import depth, name patterns
    classify: fn(&ConstantInfo) -> ChunkTemperature,
}

enum ChunkTemperature {
    Hot,    // Small chunks, likely cached
    Warm,   // Medium chunks
    Cold,   // Large chunks, rarely accessed
}
```

---

## 7. Verification CLI Commands

```bash
# Compute and display library hash from .olean files
lean5-fastload hash-olean ~/.elan/.../lib/lean

# Verify .lean5db against stored hash
lean5-fastload verify mathlib.lean5db

# Verify .lean5db against original .olean files
lean5-fastload verify mathlib.lean5db --against ~/.elan/.../lib/lean

# Show hash tree (modules and their hashes)
lean5-fastload hash-tree mathlib.lean5db

# Verify specific modules only
lean5-fastload verify mathlib.lean5db --modules Init.Core,Init.Data.Nat

# Diff two .lean5db files
lean5-fastload diff mathlib-v4.4.lean5db mathlib-v4.5.lean5db

# Sign a .lean5db file
lean5-fastload sign mathlib.lean5db --key private.key

# Verify signature
lean5-fastload verify-signature mathlib.lean5db --pubkey mathlib-release.pub
```

---

## 8. Equivalence Proof Summary

To prove `.olean` ≡ `.lean5db`:

1. **During build:** Compute canonical hash of all constants from .olean
2. **Store hash:** Embed library hash in .lean5db footer
3. **After build:** Reload .lean5db, recompute hash, verify match
4. **At load time:** Optionally re-verify hash matches content
5. **For debugging:** Store per-module and per-constant hashes for fine-grained comparison

**Hash algorithm:** BLAKE3 (fast, secure, parallelizable)

**Canonical form ensures:**
- Same constants in different order → same hash
- Same expressions with different IDs → same hash
- Same names with different interning → same hash

**The equivalence proof is:**
```
H_olean = canonical_hash(parse_olean(files))
H_lean5db = canonical_hash(load_lean5db(file))

EQUIVALENT ⟺ H_olean = H_lean5db
```

This is verifiable by anyone with the original .olean files.

---

## 9. Updated Format with Verification

Final header structure (v1.2):

```rust
struct Lean5DbHeader {
    // ... all v1.1 fields ...

    // Verification (NEW in v1.2)
    hash_section_offset: u64,
    hash_section_size: u32,
    verification_flags: u32,

    // Library root hash (32 bytes) - the equivalence proof
    library_content_hash: [u8; 32],
}
```

This hash is the **cryptographic proof** that the .lean5db contains exactly the same mathematical content as the source .olean files.
