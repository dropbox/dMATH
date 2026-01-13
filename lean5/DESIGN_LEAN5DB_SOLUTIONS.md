# Lean5DB: Comprehensive Solutions for All Identified Flaws

**Version:** 1.0
**Status:** Design (DO NOT EXECUTE)
**Date:** 2026-01-07
**Purpose:** Detailed solutions for all 35 flaws identified across Lean5DB design documents

---

## Summary

| Source Document | Flaws | Severity Breakdown |
|-----------------|-------|-------------------|
| FORMAT_ANALYSIS.md | 10 | 2 HIGH, 6 MEDIUM, 2 LOW |
| VERIFICATION.md | 10 | 3 HIGH, 5 MEDIUM, 2 LOW |
| SEARCH.md | 15 | 5 CRITICAL, 5 SIGNIFICANT, 5 MINOR |
| **Total** | **35** | |

---

## Part A: Format Flaws (1-10)

### Flaw 1: Version Coupling to Lean Toolchain
**Severity:** HIGH
**Problem:** Every Lean minor version could invalidate all .lean5db files due to representation changes.

**Solution: Multi-Layer Version Negotiation**

```rust
/// Semantic versioning for format compatibility
#[derive(Debug, Clone, Copy)]
pub struct FormatVersion {
    /// Major: Breaking changes to wire format
    pub major: u16,
    /// Minor: New optional sections (backwards compatible)
    pub minor: u16,
    /// Patch: Bug fixes only
    pub patch: u16,
}

/// Lean semantic layer versioning
#[derive(Debug, Clone, Copy)]
pub struct LeanSemanticVersion {
    /// Expression AST version (changes when Expr enum changes)
    pub expr_version: u8,
    /// Constant kind version (changes when ConstantKind changes)
    pub constant_version: u8,
    /// Universe level version
    pub level_version: u8,
}

impl Lean5DbHeader {
    pub fn can_load(&self, reader: &ReaderCapabilities) -> Result<(), IncompatibleError> {
        // Format compatibility
        if self.format_version.major != reader.format_major {
            return Err(IncompatibleError::FormatMajorMismatch);
        }

        // Semantic compatibility
        if !reader.supports_lean_semantic(self.lean_semantic_version) {
            return Err(IncompatibleError::LeanSemanticMismatch {
                file: self.lean_semantic_version,
                reader: reader.lean_semantic,
            });
        }

        Ok(())
    }
}

/// Migration support
pub trait FormatMigrator {
    fn migrate_v1_to_v2(source: &Lean5DbV1) -> Result<Lean5DbV2, MigrationError>;
}
```

**Implementation:**
1. Add `lean_semantic_version` to header (3 bytes)
2. Maintain compatibility matrix: which reader versions support which file versions
3. Provide migration tools for major version bumps
4. Default to rejecting unknown versions (fail-safe)

---

### Flaw 2: No Incremental Update Strategy
**Severity:** HIGH
**Problem:** Must rebuild entire file when one module changes.

**Solution: Delta Update Format**

```rust
/// Delta update file (.lean5db.delta)
pub struct Lean5DbDelta {
    /// Base file this delta applies to
    pub base_hash: [u8; 32],

    /// Operations to apply
    pub operations: Vec<DeltaOp>,

    /// New library hash after applying delta
    pub result_hash: [u8; 32],
}

pub enum DeltaOp {
    /// Add new constants
    AddConstants {
        chunk_data: Vec<u8>,
        index_entries: Vec<IndexEntry>,
    },

    /// Remove constants by name
    RemoveConstants {
        names: Vec<StringId>,
    },

    /// Replace constants (remove + add)
    ReplaceConstants {
        old_names: Vec<StringId>,
        new_chunk_data: Vec<u8>,
        new_index_entries: Vec<IndexEntry>,
    },

    /// Update module metadata only
    UpdateModuleMeta {
        module: StringId,
        new_meta: ModuleMetadata,
    },
}

impl Lean5Db {
    /// Apply delta to create new database
    pub fn apply_delta(&self, delta: &Lean5DbDelta) -> Result<Lean5Db, DeltaError> {
        // Verify base hash matches
        if self.library_hash() != delta.base_hash {
            return Err(DeltaError::BaseMismatch);
        }

        let mut builder = Lean5DbBuilder::from_existing(self);

        for op in &delta.operations {
            match op {
                DeltaOp::AddConstants { chunk_data, index_entries } => {
                    builder.add_chunk(chunk_data.clone());
                    builder.extend_index(index_entries.clone());
                }
                DeltaOp::RemoveConstants { names } => {
                    builder.remove_by_names(names);
                }
                // ... handle other ops
            }
        }

        let result = builder.build()?;

        // Verify result hash
        if result.library_hash() != delta.result_hash {
            return Err(DeltaError::ResultMismatch);
        }

        Ok(result)
    }
}
```

**Implementation:**
1. Build delta generator that compares two .lean5db files
2. Store deltas as separate files or append to main file
3. Support chain of deltas: base → delta1 → delta2 → ...
4. Periodic compaction merges deltas back into base

---

### Flaw 3: String Table Size Limitation
**Severity:** MEDIUM
**Problem:** `LengthPrefixedString.len: u16` limits strings to 65KB.

**Solution: Variable-Length Encoding**

```rust
/// String with variable-length size prefix
pub struct VarLenString {
    // Length encoding:
    // 0x00-0x7F: 1 byte, length 0-127
    // 0x80-0xBF: 2 bytes, length 128-16511 (0x80 | high5, low8)
    // 0xC0-0xDF: 3 bytes, length 16512-2113663
    // 0xE0-0xEF: 4 bytes, length up to 268M
    // 0xF0+: 8 bytes, arbitrary length
}

impl VarLenString {
    pub fn encode_length(len: usize) -> Vec<u8> {
        if len < 0x80 {
            vec![len as u8]
        } else if len < 0x4080 {
            let adjusted = len - 0x80;
            vec![
                0x80 | ((adjusted >> 8) as u8),
                (adjusted & 0xFF) as u8,
            ]
        } else if len < 0x204080 {
            let adjusted = len - 0x4080;
            vec![
                0xC0 | ((adjusted >> 16) as u8),
                ((adjusted >> 8) & 0xFF) as u8,
                (adjusted & 0xFF) as u8,
            ]
        } else {
            // Full 8-byte encoding for huge strings
            let mut buf = vec![0xF0];
            buf.extend_from_slice(&(len as u64).to_le_bytes());
            buf
        }
    }
}
```

**Implementation:**
1. Replace u16 length prefix with variable-length encoding
2. Common case (strings < 128 bytes) uses only 1 byte
3. Rare huge strings supported up to 2^64 bytes
4. Backwards compatible: old files rejected, new reader required

---

### Flaw 4: Expression Pool Doesn't Handle DAG Properly
**Severity:** MEDIUM
**Problem:** `ExprId = u32` limits to 4B expressions; no cycle detection; hash collisions not handled.

**Solution: Tiered IDs + Collision Handling**

```rust
/// Expression ID with tier encoding
#[derive(Clone, Copy)]
pub struct ExprId(u32);

impl ExprId {
    /// Tier 0: Built-in (Nat, Bool, Type, Prop, etc.) - 256 slots
    pub const TIER0_MAX: u32 = 0xFF;

    /// Tier 1: Common expressions - 65536 slots
    pub const TIER1_MAX: u32 = 0xFFFF;

    /// Tier 2: Everything else - ~4B slots
    pub const TIER2_MAX: u32 = u32::MAX;

    pub fn tier(&self) -> u8 {
        if self.0 <= Self::TIER0_MAX {
            0
        } else if self.0 <= Self::TIER1_MAX {
            1
        } else {
            2
        }
    }
}

/// Expression pool with collision detection
pub struct ExprPool {
    /// Expressions indexed by ExprId
    expressions: Vec<SerializedExpr>,

    /// Content hash → ExprId for deduplication
    hash_to_id: HashMap<[u8; 32], ExprId>,

    /// Collision tracking: same hash, different content
    collisions: HashMap<[u8; 32], Vec<ExprId>>,
}

impl ExprPool {
    pub fn intern(&mut self, expr: &Expr) -> ExprId {
        let hash = canonical_expr_hash(expr);

        if let Some(&existing_id) = self.hash_to_id.get(&hash) {
            // Verify not a collision
            if self.expressions[existing_id.0 as usize].equals_semantic(expr) {
                return existing_id;
            }

            // Hash collision! Check collision list
            if let Some(collision_ids) = self.collisions.get(&hash) {
                for &id in collision_ids {
                    if self.expressions[id.0 as usize].equals_semantic(expr) {
                        return id;
                    }
                }
            }

            // New collision
            let new_id = self.allocate_new(expr);
            self.collisions.entry(hash).or_default().push(new_id);
            return new_id;
        }

        // First time seeing this hash
        let id = self.allocate_new(expr);
        self.hash_to_id.insert(hash, id);
        id
    }

    fn allocate_new(&mut self, expr: &Expr) -> ExprId {
        let id = ExprId(self.expressions.len() as u32);
        if id.0 > ExprId::TIER2_MAX {
            panic!("Expression pool overflow");
        }
        self.expressions.push(serialize_expr(expr));
        id
    }
}

/// Cycle detection during building
pub fn detect_expr_cycle(expr: &Expr) -> Option<CycleInfo> {
    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    detect_cycle_recursive(expr, &mut visited, &mut stack)
}
```

**Implementation:**
1. Use tiered IDs for better encoding efficiency
2. Full semantic comparison on hash collision
3. Cycle detection during build (expressions should be acyclic)
4. Consider u40 or u48 IDs if 4B not enough (unlikely for Mathlib)

---

### Flaw 5: Lazy Loading Cache Unbounded
**Severity:** MEDIUM
**Problem:** `chunk_cache` grows without bound, defeating lazy loading.

**Solution: LRU Cache with Size Limit**

```rust
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct LazyDatabase {
    // ... other fields ...

    /// LRU cache with configurable size limit
    chunk_cache: Mutex<ChunkCache>,
}

pub struct ChunkCache {
    /// LRU cache mapping chunk ID to decompressed data
    lru: LruCache<u32, Arc<DecompressedChunk>>,

    /// Current total size of cached chunks
    current_size: usize,

    /// Maximum cache size in bytes
    max_size: usize,
}

impl ChunkCache {
    pub fn new(max_size: usize) -> Self {
        // Allow up to 1000 chunks in LRU, but also enforce byte limit
        Self {
            lru: LruCache::new(NonZeroUsize::new(1000).unwrap()),
            current_size: 0,
            max_size,
        }
    }

    pub fn get(&mut self, chunk_id: u32) -> Option<Arc<DecompressedChunk>> {
        self.lru.get(&chunk_id).cloned()
    }

    pub fn insert(&mut self, chunk_id: u32, chunk: DecompressedChunk) {
        let chunk_size = chunk.size_bytes();
        let chunk = Arc::new(chunk);

        // Evict until we have room
        while self.current_size + chunk_size > self.max_size {
            if let Some((_, evicted)) = self.lru.pop_lru() {
                self.current_size -= evicted.size_bytes();
            } else {
                break; // Cache empty
            }
        }

        // Insert new chunk
        if let Some(old) = self.lru.put(chunk_id, chunk) {
            self.current_size -= old.size_bytes();
        }
        self.current_size += chunk_size;
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.lru.len(),
            size_bytes: self.current_size,
            max_bytes: self.max_size,
            utilization: self.current_size as f64 / self.max_size as f64,
        }
    }
}

impl LazyDatabase {
    /// Create with configurable cache size
    pub fn open_with_cache(path: &Path, cache_mb: usize) -> Result<Self, Error> {
        // ...
        Ok(Self {
            chunk_cache: Mutex::new(ChunkCache::new(cache_mb * 1024 * 1024)),
            // ...
        })
    }
}
```

**Implementation:**
1. Default cache size: 100 MB (configurable)
2. LRU eviction when cache full
3. Track actual decompressed size, not compressed
4. Expose cache stats for monitoring

---

### Flaw 6: No Error Recovery for Corrupted Chunks
**Severity:** LOW
**Problem:** Single bit flip = entire chunk lost.

**Solution: Reed-Solomon Error Correction (Optional)**

```rust
/// Chunk with optional error correction
pub struct RobustChunk {
    /// Compressed data
    pub data: Vec<u8>,

    /// CRC32 checksum
    pub crc32: u32,

    /// Optional Reed-Solomon parity (adds ~10% overhead)
    pub parity: Option<Vec<u8>>,
}

impl RobustChunk {
    pub fn new(data: Vec<u8>, enable_ecc: bool) -> Self {
        let crc32 = crc32fast::hash(&data);

        let parity = if enable_ecc {
            // RS(255, 223) - can correct up to 16 byte errors
            let encoder = reed_solomon::Encoder::new(32);
            Some(encoder.encode(&data).parity().to_vec())
        } else {
            None
        };

        Self { data, crc32, parity }
    }

    pub fn decode(&self) -> Result<Vec<u8>, ChunkError> {
        // First try: direct decode
        let computed_crc = crc32fast::hash(&self.data);
        if computed_crc == self.crc32 {
            return Ok(self.data.clone());
        }

        // CRC mismatch - try error correction
        if let Some(parity) = &self.parity {
            let decoder = reed_solomon::Decoder::new(32);
            match decoder.correct(&self.data, parity) {
                Ok(corrected) => {
                    log::warn!("Corrected {} byte errors in chunk",
                              decoder.errors_corrected());
                    return Ok(corrected);
                }
                Err(e) => {
                    return Err(ChunkError::Uncorrectable {
                        crc_expected: self.crc32,
                        crc_actual: computed_crc,
                        ecc_error: e,
                    });
                }
            }
        }

        Err(ChunkError::CorruptedNoECC {
            crc_expected: self.crc32,
            crc_actual: computed_crc,
        })
    }
}
```

**Implementation:**
1. ECC is optional (adds ~10% file size)
2. Enable for archival/distribution builds
3. Disable for local development (faster writes)
4. Flag in header: `FLAG_HAS_ECC`

---

### Flaw 7: Module Dependency Ordering Not Preserved
**Severity:** MEDIUM
**Problem:** Streaming load yields constants in chunk order, not topological order.

**Solution: Topological Sort Metadata + Ordered Chunks**

```rust
/// Each constant has its topological order number
#[derive(Clone, Copy)]
pub struct TopoOrder(u32);

/// Index entry with topological ordering
pub struct IndexEntry {
    pub name_id: StringId,
    pub kind: ConstantKind,
    pub type_expr_id: ExprId,
    pub chunk_id: u32,
    pub chunk_offset: u32,

    /// Topological order (NEW)
    /// Guarantees: if A depends on B, then B.topo_order < A.topo_order
    pub topo_order: TopoOrder,

    /// Mutual group (NEW)
    /// All constants with same group_id must be loaded together
    pub mutual_group_id: Option<u32>,
}

/// Builder ensures topological ordering
impl Lean5DbBuilder {
    pub fn add_constants(&mut self, constants: Vec<ConstantInfo>) {
        // Compute dependency graph
        let deps = compute_dependencies(&constants);

        // Topological sort
        let sorted = topological_sort(&constants, &deps)
            .expect("Dependency cycle detected");

        // Assign topo_order
        for (order, constant) in sorted.iter().enumerate() {
            self.constant_topo_orders.insert(
                constant.name.clone(),
                TopoOrder(order as u32)
            );
        }

        // Group mutual inductives
        let mutual_groups = find_mutual_groups(&constants);
        for (group_id, group) in mutual_groups.iter().enumerate() {
            for name in group {
                self.mutual_groups.insert(name.clone(), group_id as u32);
            }
        }

        // Chunk assignment: keep mutual groups together
        self.assign_chunks_respecting_groups(&sorted, &mutual_groups);
    }
}

/// Streaming loader respects ordering
impl LazyDatabase {
    pub fn stream_constants(&self) -> impl Iterator<Item = ConstantInfo> + '_ {
        // Sort index entries by topo_order
        let mut entries: Vec<_> = self.index.entries.iter().collect();
        entries.sort_by_key(|e| e.topo_order);

        entries.into_iter().map(|e| self.load_constant(e))
    }

    pub fn stream_by_mutual_groups(&self) -> impl Iterator<Item = Vec<ConstantInfo>> + '_ {
        // Group by mutual_group_id, yield groups in topo order
        let mut groups: BTreeMap<u32, Vec<&IndexEntry>> = BTreeMap::new();

        for entry in &self.index.entries {
            let group_id = entry.mutual_group_id.unwrap_or(entry.topo_order.0);
            groups.entry(group_id).or_default().push(entry);
        }

        groups.into_values().map(|entries| {
            entries.iter().map(|e| self.load_constant(e)).collect()
        })
    }
}
```

**Implementation:**
1. Compute topo_order during build
2. Store in index (4 bytes per constant)
3. Streaming yields constants in correct order
4. Mutual groups yield as single batch

---

### Flaw 8: No Support for Mutual Inductives Grouping
**Severity:** MEDIUM
**Problem:** Mutual inductives must be registered together but are serialized independently.

**Solution: Explicit Mutual Group Metadata**

```rust
/// Mutual inductive group metadata
pub struct MutualGroup {
    /// Unique group ID
    pub group_id: u32,

    /// Names of all inductives in this mutual block
    pub inductive_names: Vec<StringId>,

    /// Names of all constructors
    pub constructor_names: Vec<StringId>,

    /// Names of all recursors
    pub recursor_names: Vec<StringId>,
}

/// Module metadata section includes mutual groups
pub struct ModuleMetadata {
    pub name: StringId,
    pub imports: Vec<StringId>,
    pub doc: Option<StringId>,

    /// Mutual groups in this module (NEW)
    pub mutual_groups: Vec<MutualGroup>,
}

/// Loader handles mutual groups atomically
impl LazyDatabase {
    /// Load a mutual group atomically (all or nothing)
    pub fn load_mutual_group(&self, group_id: u32) -> Result<MutualGroupData, Error> {
        let group = self.get_mutual_group(group_id)?;

        // Load all parts
        let inductives: Vec<_> = group.inductive_names
            .iter()
            .map(|n| self.load_constant_by_name_id(*n))
            .collect::<Result<_, _>>()?;

        let constructors: Vec<_> = group.constructor_names
            .iter()
            .map(|n| self.load_constant_by_name_id(*n))
            .collect::<Result<_, _>>()?;

        let recursors: Vec<_> = group.recursor_names
            .iter()
            .map(|n| self.load_constant_by_name_id(*n))
            .collect::<Result<_, _>>()?;

        Ok(MutualGroupData {
            inductives,
            constructors,
            recursors,
        })
    }

    /// Register all parts of a mutual group to kernel
    pub fn register_mutual_group(
        &self,
        env: &mut Environment,
        group_id: u32,
    ) -> Result<(), KernelError> {
        let data = self.load_mutual_group(group_id)?;

        // Register in correct order: inductives → constructors → recursors
        for ind in &data.inductives {
            env.add_inductive(ind)?;
        }
        for ctor in &data.constructors {
            env.add_constructor(ctor)?;
        }
        for rec in &data.recursors {
            env.add_recursor(rec)?;
        }

        Ok(())
    }
}
```

**Implementation:**
1. Detect mutual groups during .olean parsing
2. Store group metadata in module section
3. Loader provides atomic group loading API
4. Chunk assignment keeps groups together

---

### Flaw 9: Dictionary Not Versioned Separately
**Severity:** LOW
**Problem:** Can't share dictionary across files or update without rebuild.

**Solution: External Dictionary Support**

```rust
/// Dictionary source specification
pub enum DictionarySource {
    /// Dictionary embedded in this file
    Embedded {
        offset: u64,
        size: u32,
    },

    /// External dictionary file
    External {
        /// Content hash of dictionary file
        hash: [u8; 32],
        /// Path hint (may not exist)
        path_hint: Option<String>,
    },

    /// Well-known standard dictionary
    WellKnown {
        /// Standard dictionary ID
        id: WellKnownDict,
    },

    /// No dictionary (raw zstd)
    None,
}

#[derive(Clone, Copy)]
pub enum WellKnownDict {
    /// Trained on Mathlib v4.x
    MathlibV4,
    /// Trained on Lean std
    LeanStd,
    /// Trained on Init
    LeanInit,
}

/// Dictionary file format
pub struct DictionaryFile {
    pub magic: [u8; 8],  // "LEAN5DCT"
    pub version: u32,
    pub hash: [u8; 32],
    pub trained_on: String,  // Description
    pub data: Vec<u8>,
}

impl Lean5DbLoader {
    pub fn resolve_dictionary(&self) -> Result<Vec<u8>, DictError> {
        match &self.header.dictionary_source {
            DictionarySource::Embedded { offset, size } => {
                self.read_section(*offset, *size as usize)
            }

            DictionarySource::External { hash, path_hint } => {
                // Try standard locations
                let paths = [
                    path_hint.as_deref(),
                    Some(&format!("~/.lean5/dicts/{}.dict", hex::encode(hash))),
                    Some("/usr/share/lean5/dicts/"),
                ];

                for path in paths.iter().flatten() {
                    if let Ok(dict) = DictionaryFile::load(path) {
                        if dict.hash == *hash {
                            return Ok(dict.data);
                        }
                    }
                }

                Err(DictError::ExternalNotFound { hash: *hash })
            }

            DictionarySource::WellKnown { id } => {
                // Built into binary
                Ok(get_builtin_dictionary(*id).to_vec())
            }

            DictionarySource::None => Ok(vec![]),
        }
    }
}
```

**Implementation:**
1. Store dictionary source enum in header
2. Ship common dictionaries as built-ins
3. Support ~/.lean5/dicts/ directory
4. Dictionary files are content-addressable

---

### Flaw 10: No Partial Build Support
**Severity:** MEDIUM
**Problem:** Can't add modules to existing .lean5db incrementally.

**Solution: Append-Only Build Mode**

```rust
/// Builder supports incremental addition
pub struct IncrementalBuilder {
    /// Existing database (if extending)
    base: Option<Lean5Db>,

    /// New constants to add
    new_constants: Vec<ConstantInfo>,

    /// Modules being added
    new_modules: Vec<ModuleMetadata>,
}

impl IncrementalBuilder {
    /// Start from existing database
    pub fn extend(base: Lean5Db) -> Self {
        Self {
            base: Some(base),
            new_constants: vec![],
            new_modules: vec![],
        }
    }

    /// Start fresh
    pub fn new() -> Self {
        Self {
            base: None,
            new_constants: vec![],
            new_modules: vec![],
        }
    }

    /// Add a module's constants
    pub fn add_module(&mut self, module_path: &Path) -> Result<(), Error> {
        let olean = parse_olean(module_path)?;

        // Check for conflicts with base
        if let Some(base) = &self.base {
            for constant in &olean.constants {
                if base.has_constant(&constant.name) {
                    return Err(Error::DuplicateConstant(constant.name.clone()));
                }
            }
        }

        self.new_constants.extend(olean.constants);
        self.new_modules.push(olean.metadata);
        Ok(())
    }

    /// Build final database
    pub fn build(self) -> Result<Lean5Db, Error> {
        match self.base {
            None => {
                // Fresh build
                Lean5DbBuilder::new()
                    .add_constants(self.new_constants)
                    .add_modules(self.new_modules)
                    .build()
            }
            Some(base) => {
                // Extend existing
                // Reuse string table, expression pool entries
                let mut builder = Lean5DbBuilder::from_existing(&base);

                // Intern new strings/exprs (may reuse existing)
                builder.add_constants(self.new_constants);
                builder.add_modules(self.new_modules);

                builder.build()
            }
        }
    }
}
```

**Implementation:**
1. Builder can start from existing .lean5db
2. Reuse string/expression pools (extend, don't rebuild)
3. Append new chunks (don't rewrite existing)
4. Update index and metadata sections

---

## Part B: Verification Flaws (11-20)

### Flaw 11: No Format Migration Path
**Severity:** HIGH
**Problem:** Format changes require manual migration.

**Solution: Built-in Migration Framework**

```rust
/// Format migration registry
pub struct MigrationRegistry {
    migrations: HashMap<(Version, Version), Box<dyn Migration>>,
}

pub trait Migration {
    fn source_version(&self) -> Version;
    fn target_version(&self) -> Version;
    fn migrate(&self, source: &[u8]) -> Result<Vec<u8>, MigrationError>;
    fn can_migrate_in_place(&self) -> bool;
}

impl MigrationRegistry {
    pub fn migrate(
        &self,
        data: &[u8],
        from: Version,
        to: Version,
    ) -> Result<Vec<u8>, MigrationError> {
        // Find migration path (BFS)
        let path = self.find_path(from, to)?;

        let mut current = data.to_vec();
        for step in path {
            current = step.migrate(&current)?;
        }

        Ok(current)
    }
}

/// CLI command
/// lean5-fastload migrate old.lean5db new.lean5db --to-version 2.0
pub fn cmd_migrate(args: MigrateArgs) -> Result<(), Error> {
    let source = Lean5DbFile::open(&args.source)?;
    let target_version = args.to_version.unwrap_or(CURRENT_VERSION);

    let registry = MigrationRegistry::default();
    let migrated = registry.migrate(
        source.raw_data(),
        source.version(),
        target_version,
    )?;

    std::fs::write(&args.target, migrated)?;
    println!("Migrated {} → {}", source.version(), target_version);
    Ok(())
}
```

**Implementation:**
1. Define migration trait
2. Register v1→v2, v2→v3, etc. migrations
3. CLI tool for manual migration
4. Auto-migrate on load (optional flag)

---

### Flaw 12: Single Point of Failure (Header)
**Severity:** MEDIUM
**Problem:** Corrupted header = entire file unusable.

**Solution: Redundant Header + Footer**

```rust
/// File layout with redundancy
///
/// [Primary Header @ 0]
/// [... sections ...]
/// [Backup Header @ (size - HEADER_SIZE - FOOTER_SIZE)]
/// [Footer @ (size - FOOTER_SIZE)]

pub struct FileLayout {
    /// Primary header at start
    pub primary_header_offset: u64,  // Always 0

    /// Backup header near end
    pub backup_header_offset: u64,

    /// Footer at very end
    pub footer_offset: u64,
}

impl Lean5DbFile {
    pub fn read_header(&self) -> Result<Lean5DbHeader, Error> {
        // Try primary header first
        match self.read_header_at(0) {
            Ok(header) if header.magic == MAGIC => Ok(header),
            primary_err => {
                // Try backup header
                let backup_offset = self.size - HEADER_SIZE as u64 - FOOTER_SIZE as u64;
                match self.read_header_at(backup_offset) {
                    Ok(header) if header.magic == MAGIC => {
                        log::warn!("Primary header corrupted, using backup");
                        Ok(header)
                    }
                    Err(backup_err) => {
                        Err(Error::BothHeadersCorrupted {
                            primary: Box::new(primary_err.unwrap_err()),
                            backup: Box::new(backup_err),
                        })
                    }
                }
            }
        }
    }
}

/// Footer with file-wide checksum
pub struct Lean5DbFooter {
    /// Magic for footer detection
    pub magic: [u8; 8],  // "LEAN5FTR"

    /// Total file size (for corruption detection)
    pub expected_size: u64,

    /// Checksum of entire file (excluding footer)
    pub file_checksum: [u8; 32],

    /// Offset to backup header
    pub backup_header_offset: u64,

    /// Reserved
    pub _reserved: [u8; 16],
}
```

**Implementation:**
1. Write backup header before footer
2. Footer contains file checksum
3. Reader tries primary, falls back to backup
4. Repair tool can reconstruct from backup

---

### Flaw 13: No Streaming Write Support
**Severity:** MEDIUM
**Problem:** Must know all content before writing header.

**Solution: Two-Pass or Placeholder Write**

```rust
/// Streaming writer with deferred header
pub struct StreamingWriter {
    file: BufWriter<File>,

    /// Placeholder header (will be overwritten)
    header_placeholder: [u8; HEADER_SIZE],

    /// Track section offsets as we write
    section_offsets: Vec<(SectionId, u64)>,

    /// Current write position
    position: u64,
}

impl StreamingWriter {
    pub fn new(path: &Path) -> Result<Self, Error> {
        let file = BufWriter::new(File::create(path)?);

        let mut writer = Self {
            file,
            header_placeholder: [0u8; HEADER_SIZE],
            section_offsets: vec![],
            position: 0,
        };

        // Write placeholder header
        writer.file.write_all(&writer.header_placeholder)?;
        writer.position = HEADER_SIZE as u64;

        Ok(writer)
    }

    pub fn write_section(&mut self, id: SectionId, data: &[u8]) -> Result<u64, Error> {
        let offset = self.position;
        self.section_offsets.push((id, offset));
        self.file.write_all(data)?;
        self.position += data.len() as u64;
        Ok(offset)
    }

    pub fn finalize(mut self, mut header: Lean5DbHeader) -> Result<(), Error> {
        // Fill in section offsets
        for (id, offset) in &self.section_offsets {
            header.set_section_offset(*id, *offset);
        }

        // Write backup header
        let backup_offset = self.position;
        let header_bytes = header.serialize();
        self.file.write_all(&header_bytes)?;
        self.position += header_bytes.len() as u64;

        // Write footer
        let footer = Lean5DbFooter::new(self.position, backup_offset);
        self.file.write_all(&footer.serialize())?;

        // Seek back and write real header
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header_bytes)?;

        self.file.flush()?;
        Ok(())
    }
}
```

**Implementation:**
1. Write placeholder header at start
2. Stream sections, tracking offsets
3. At end, seek back and write real header
4. Also write backup header + footer

---

### Flaw 14: Chunk Boundaries Split Mutual Groups
**Severity:** HIGH
**Problem:** Mutual inductives may span chunks, breaking atomic loading.

**Solution: Constraint-Aware Chunk Assignment**

```rust
/// Chunk assignment with constraints
pub struct ChunkAssigner {
    /// Maximum chunk size
    max_chunk_size: usize,

    /// Current chunk being built
    current_chunk: Vec<ConstantId>,
    current_size: usize,

    /// Constraint groups that must stay together
    constraint_groups: HashMap<u32, Vec<ConstantId>>,

    /// Finalized chunks
    chunks: Vec<Vec<ConstantId>>,
}

impl ChunkAssigner {
    pub fn assign(&mut self, constants: &[ConstantInfo], groups: &[MutualGroup]) {
        // Build constraint map
        for group in groups {
            let group_constants: Vec<_> = group.all_constant_ids().collect();
            let group_size: usize = group_constants.iter()
                .map(|id| constants[*id].serialized_size())
                .sum();

            // If group too big for one chunk, this is an error
            if group_size > self.max_chunk_size {
                panic!("Mutual group {} too large ({} bytes) for chunk size ({})",
                       group.group_id, group_size, self.max_chunk_size);
            }

            self.constraint_groups.insert(group.group_id, group_constants);
        }

        // Assign constants to chunks respecting constraints
        let mut assigned: HashSet<ConstantId> = HashSet::new();

        for (id, constant) in constants.iter().enumerate() {
            let id = ConstantId(id as u32);

            if assigned.contains(&id) {
                continue;
            }

            // Check if this constant is part of a constraint group
            if let Some(group_id) = self.get_constraint_group(id) {
                let group_constants = self.constraint_groups.get(&group_id).unwrap();
                let group_size: usize = group_constants.iter()
                    .map(|id| constants[id.0 as usize].serialized_size())
                    .sum();

                // Need new chunk if group doesn't fit
                if self.current_size + group_size > self.max_chunk_size {
                    self.flush_chunk();
                }

                // Add entire group
                for gid in group_constants {
                    self.current_chunk.push(*gid);
                    assigned.insert(*gid);
                }
                self.current_size += group_size;
            } else {
                // Regular constant
                let size = constant.serialized_size();
                if self.current_size + size > self.max_chunk_size {
                    self.flush_chunk();
                }
                self.current_chunk.push(id);
                self.current_size += size;
                assigned.insert(id);
            }
        }

        self.flush_chunk();
    }
}
```

**Implementation:**
1. Identify constraint groups (mutual inductives) during parsing
2. Chunk assigner keeps groups together
3. Error if any group exceeds chunk size
4. Consider adaptive chunk sizing for large groups

---

### Flaw 15: No Partial Library Support
**Severity:** LOW
**Problem:** Can't create "Mathlib minus X" easily.

**Solution: Module Selection API**

```rust
/// Module selector for partial loading
pub struct ModuleSelector {
    /// Modules to include (if empty, include all)
    include: HashSet<String>,

    /// Modules to exclude
    exclude: HashSet<String>,

    /// Transitively include dependencies
    include_deps: bool,
}

impl ModuleSelector {
    pub fn all() -> Self {
        Self {
            include: HashSet::new(),
            exclude: HashSet::new(),
            include_deps: false,
        }
    }

    pub fn only(modules: impl IntoIterator<Item = String>) -> Self {
        Self {
            include: modules.into_iter().collect(),
            exclude: HashSet::new(),
            include_deps: true,
        }
    }

    pub fn except(modules: impl IntoIterator<Item = String>) -> Self {
        Self {
            include: HashSet::new(),
            exclude: modules.into_iter().collect(),
            include_deps: false,
        }
    }

    /// Resolve to concrete module set
    pub fn resolve(&self, db: &Lean5Db) -> HashSet<String> {
        let all_modules: HashSet<_> = db.modules().map(|m| m.name.clone()).collect();

        let mut selected = if self.include.is_empty() {
            all_modules.clone()
        } else {
            self.include.clone()
        };

        // Add dependencies if requested
        if self.include_deps {
            let mut to_add = vec![];
            for module in &selected {
                to_add.extend(db.transitive_deps(module));
            }
            selected.extend(to_add);
        }

        // Remove excluded
        for excl in &self.exclude {
            selected.remove(excl);
        }

        selected
    }
}

impl Lean5Db {
    /// Load only selected modules
    pub fn load_partial(&self, selector: &ModuleSelector) -> Result<Environment, Error> {
        let modules = selector.resolve(self);

        let mut env = Environment::new();
        for module in self.stream_constants() {
            if modules.contains(&module.module_name) {
                env.add(module)?;
            }
        }

        Ok(env)
    }

    /// Export subset to new file
    pub fn export_partial(
        &self,
        selector: &ModuleSelector,
        output: &Path,
    ) -> Result<(), Error> {
        let modules = selector.resolve(self);

        let mut builder = Lean5DbBuilder::new();
        for constant in self.stream_constants() {
            if modules.contains(&constant.module_name) {
                builder.add_constant(constant);
            }
        }

        builder.build_to_file(output)
    }
}
```

**Implementation:**
1. Store module dependency graph in metadata
2. Selector API for include/exclude patterns
3. Lazy loading naturally supports partial access
4. Export tool creates subset files

---

### Flaw 16: Expression Pool Size Unknown Upfront
**Severity:** MEDIUM
**Problem:** Deduplication requires seeing all expressions first.

**Solution: Two-Pass Build**

```rust
/// Two-pass builder for optimal deduplication
pub struct TwoPassBuilder {
    /// Pass 1 state: collect all expressions
    pass1_exprs: HashSet<ExprHash>,
    pass1_strings: HashSet<String>,

    /// Pass 2 state: build with known sizes
    expr_pool: ExprPool,
    string_table: StringTable,
}

impl TwoPassBuilder {
    /// Pass 1: Scan all .olean files, collect unique expressions/strings
    pub fn pass1_scan(&mut self, olean_files: &[PathBuf]) -> Result<Pass1Stats, Error> {
        for path in olean_files {
            let module = parse_olean(path)?;

            for constant in &module.constants {
                self.collect_strings(&constant.name);
                self.collect_exprs(&constant.type_);
                if let Some(value) = &constant.value {
                    self.collect_exprs(value);
                }
            }
        }

        Ok(Pass1Stats {
            unique_exprs: self.pass1_exprs.len(),
            unique_strings: self.pass1_strings.len(),
        })
    }

    /// Between passes: allocate pools with known sizes
    pub fn allocate_pools(&mut self) {
        // Now we know exact sizes
        self.expr_pool = ExprPool::with_capacity(self.pass1_exprs.len());
        self.string_table = StringTable::with_capacity(self.pass1_strings.len());

        // Pre-populate string table for better locality
        let mut strings: Vec<_> = self.pass1_strings.drain().collect();
        strings.sort(); // Alphabetical for better compression
        for s in strings {
            self.string_table.intern(&s);
        }
    }

    /// Pass 2: Build file with pre-sized pools
    pub fn pass2_build(&mut self, olean_files: &[PathBuf]) -> Result<Lean5Db, Error> {
        let mut builder = Lean5DbBuilder::with_pools(
            self.expr_pool.clone(),
            self.string_table.clone(),
        );

        for path in olean_files {
            let module = parse_olean(path)?;
            builder.add_module(module);
        }

        builder.build()
    }
}
```

**Implementation:**
1. Pass 1: Scan all files, collect hashes
2. Allocate pools with exact capacity
3. Pass 2: Build file using pre-sized pools
4. Results in better memory locality and no reallocations

---

### Flaw 17: No Custom Attributes/Extensions
**Severity:** MEDIUM
**Problem:** Lean has extensible attributes we don't preserve.

**Solution: Extension Data Section**

```rust
/// Extension data for a constant
pub struct ConstantExtensions {
    /// Constant this extends
    pub constant_id: ConstantId,

    /// Key-value extension data
    pub extensions: Vec<Extension>,
}

pub struct Extension {
    /// Extension namespace (e.g., "simp", "aesop", "custom")
    pub namespace: StringId,

    /// Extension key
    pub key: StringId,

    /// Opaque extension value (format depends on namespace)
    pub value: Vec<u8>,
}

/// File section for extensions
pub struct ExtensionSection {
    /// Version of extension format
    pub version: u32,

    /// Extensions grouped by namespace for efficient filtering
    pub namespaces: Vec<NamespaceExtensions>,
}

pub struct NamespaceExtensions {
    pub namespace: StringId,
    pub entries: Vec<(ConstantId, Vec<u8>)>,
}

impl Lean5Db {
    /// Get extensions for a constant
    pub fn get_extensions(&self, constant: &Name) -> Vec<Extension> {
        let id = self.constant_id(constant)?;
        self.extension_section.get_for_constant(id)
    }

    /// Get all constants with a specific extension
    pub fn constants_with_extension(
        &self,
        namespace: &str,
        key: &str,
    ) -> impl Iterator<Item = ConstantId> + '_ {
        self.extension_section.filter(namespace, key)
    }
}
```

**Implementation:**
1. Parse Lean attribute data during .olean loading
2. Store in separate extension section
3. Namespace grouping for efficient queries
4. Opaque values preserve unknown extensions

---

### Flaw 18: TOCTOU for Memory-Mapped Files
**Severity:** LOW
**Problem:** File could change between mmap and access.

**Solution: File Locking + Validation**

```rust
use fs2::FileExt;

pub struct Lean5DbFile {
    file: File,
    mmap: Mmap,

    /// Lock held while file is open
    _lock: FileLock,
}

struct FileLock {
    file: File,
}

impl FileLock {
    fn acquire_shared(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        file.lock_shared()?;  // Blocks until lock acquired
        Ok(Self { file })
    }

    fn acquire_exclusive(path: &Path) -> Result<Self, Error> {
        let file = OpenOptions::new().write(true).open(path)?;
        file.lock_exclusive()?;
        Ok(Self { file })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

impl Lean5DbFile {
    pub fn open(path: &Path) -> Result<Self, Error> {
        // Acquire shared lock (allows multiple readers)
        let lock = FileLock::acquire_shared(path)?;

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate file integrity
        let header = Lean5DbHeader::parse(&mmap[..HEADER_SIZE])?;
        let footer = Lean5DbFooter::parse(&mmap[mmap.len() - FOOTER_SIZE..])?;

        if footer.expected_size != mmap.len() as u64 {
            return Err(Error::FileSizeMismatch);
        }

        // Verify file checksum
        let computed_checksum = blake3::hash(&mmap[..mmap.len() - FOOTER_SIZE]);
        if computed_checksum.as_bytes() != &footer.file_checksum {
            return Err(Error::FileChecksumMismatch);
        }

        Ok(Self {
            file,
            mmap,
            _lock: lock,
        })
    }
}
```

**Implementation:**
1. Acquire shared lock on open (multiple readers OK)
2. Validate size and checksum after mmap
3. Hold lock for lifetime of file handle
4. Writers acquire exclusive lock

---

### Flaw 19: No Provenance/Signatures
**Severity:** MEDIUM
**Problem:** Can't verify who built the file.

**Solution: Ed25519 Signature Section**

```rust
use ed25519_dalek::{Signature, SigningKey, VerifyingKey};

/// Signature section
pub struct SignatureSection {
    /// Signer's public key
    pub public_key: [u8; 32],

    /// Signature over library_hash
    pub signature: [u8; 64],

    /// What was signed (must equal library_hash)
    pub signed_hash: [u8; 32],

    /// When it was signed (Unix timestamp)
    pub timestamp: u64,

    /// Optional signer identity (e.g., "mathlib-release-bot@leanprover.org")
    pub signer_identity: Option<String>,
}

impl SignatureSection {
    /// Sign a database
    pub fn sign(db: &Lean5Db, signing_key: &SigningKey) -> Self {
        let library_hash = db.library_hash();
        let signature = signing_key.sign(&library_hash);

        Self {
            public_key: signing_key.verifying_key().to_bytes(),
            signature: signature.to_bytes(),
            signed_hash: library_hash,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signer_identity: None,
        }
    }

    /// Verify signature
    pub fn verify(&self, db: &Lean5Db) -> Result<(), SignatureError> {
        // Check signed hash matches current library hash
        let library_hash = db.library_hash();
        if self.signed_hash != library_hash {
            return Err(SignatureError::HashMismatch);
        }

        // Verify signature
        let public_key = VerifyingKey::from_bytes(&self.public_key)?;
        let signature = Signature::from_bytes(&self.signature);

        public_key.verify_strict(&self.signed_hash, &signature)?;

        Ok(())
    }
}

/// Known trusted public keys
pub struct TrustStore {
    trusted_keys: HashMap<[u8; 32], TrustedKey>,
}

pub struct TrustedKey {
    pub public_key: [u8; 32],
    pub identity: String,
    pub trust_level: TrustLevel,
}

pub enum TrustLevel {
    /// Official Mathlib releases
    Official,
    /// Community maintainers
    Community,
    /// User-added
    Custom,
}
```

**Implementation:**
1. Optional signature section in file
2. Sign library_hash with Ed25519
3. Ship trusted public keys with lean5
4. CLI: `lean5-fastload verify-signature`

---

### Flaw 20: Dictionary Not Shareable
**Severity:** LOW
**Problem:** Each file embeds its own dictionary.

**Solution:** Already addressed in Flaw 9 (External Dictionary Support).

---

## Part C: Search Flaws (21-35)

### Flaw 21 (Search 1): Embedding Quality for Math
**Severity:** CRITICAL
**Problem:** General embeddings cluster unrelated concepts.

**Solution: Math-Specialized Embeddings**

```rust
/// Embedding configuration with math specialization
pub struct EmbeddingConfig {
    pub model: EmbeddingModel,
    pub preprocessing: MathPreprocessing,
    pub normalization: NormalizationType,
}

pub enum EmbeddingModel {
    /// General purpose (baseline)
    MiniLM { size: MiniLMSize },

    /// Math-specialized
    Mathbert,

    /// Fine-tuned on Mathlib
    MathlibFineTuned { checkpoint: String },

    /// Multi-vector (ColBERT-style for better precision)
    ColMath { max_tokens: usize },
}

pub struct MathPreprocessing {
    /// Convert type to natural language
    pub type_to_nl: bool,

    /// Include docstring if available
    pub include_docstring: bool,

    /// Expand notation (∀ → "for all")
    pub expand_notation: bool,

    /// Include example usages
    pub include_usages: bool,
}

/// Generate embedding with preprocessing
pub fn embed_constant(
    constant: &ConstantInfo,
    config: &EmbeddingConfig,
) -> Vec<f32> {
    let text = preprocess_for_embedding(constant, &config.preprocessing);

    match &config.model {
        EmbeddingModel::MathlibFineTuned { checkpoint } => {
            let model = load_finetuned_model(checkpoint);
            model.encode(&text)
        }
        EmbeddingModel::ColMath { max_tokens } => {
            // Multi-vector: one embedding per token
            let tokens = tokenize(&text, *max_tokens);
            let embeddings: Vec<_> = tokens.iter()
                .map(|t| base_encode(t))
                .collect();
            // Return concatenated for storage
            embeddings.concat()
        }
        // ... other models
    }
}

/// Fine-tuning pipeline
pub struct FineTuningPipeline {
    /// Training pairs: (query, positive, negatives)
    pub training_data: Vec<TrainingTriple>,

    /// Base model to fine-tune
    pub base_model: String,

    /// Output checkpoint
    pub output: PathBuf,
}

impl FineTuningPipeline {
    /// Generate training data from Mathlib
    pub fn generate_training_data(db: &Lean5Db) -> Vec<TrainingTriple> {
        let mut triples = vec![];

        // Positive pairs: constants that reference each other
        for constant in db.constants() {
            let refs = db.references_from(&constant.name);
            for ref_name in refs {
                triples.push(TrainingTriple {
                    anchor: constant.to_search_text(),
                    positive: db.get(&ref_name).unwrap().to_search_text(),
                    negatives: sample_negatives(db, &constant.name, 10),
                });
            }
        }

        // Positive pairs: same concept across types (Nat.add_comm, Int.add_comm)
        for (pattern, instances) in db.find_analogous_constants() {
            for (a, b) in instances.pairs() {
                triples.push(TrainingTriple {
                    anchor: a.to_search_text(),
                    positive: b.to_search_text(),
                    negatives: sample_negatives(db, &a.name, 10),
                });
            }
        }

        triples
    }
}
```

**Implementation:**
1. Default to Mathbert if available
2. Provide fine-tuning pipeline for Mathlib
3. Multi-vector embeddings for precision
4. Ablation study comparing models

---

### Flaw 22 (Search 2): Index-Source Version Coupling
**Severity:** CRITICAL
**Problem:** Stale index returns wrong results silently.

**Solution: Content Hash in Index Header**

```rust
/// Search index header
pub struct SearchIndexHeader {
    pub magic: [u8; 8],
    pub version: u32,

    /// Hash of source Lean5DB
    pub source_lean5db_hash: [u8; 32],

    /// Timestamp of index build
    pub build_timestamp: u64,

    /// Embedding model identifier
    pub embedding_model_id: String,
}

impl SearchEngine {
    pub fn load(
        index_path: &Path,
        db: &Lean5Db,
    ) -> Result<Self, SearchError> {
        let index = SearchIndex::open(index_path)?;

        // Verify index matches database
        let db_hash = db.library_hash();
        if index.header.source_lean5db_hash != db_hash {
            return Err(SearchError::IndexOutOfDate {
                index_hash: index.header.source_lean5db_hash,
                db_hash,
                index_time: index.header.build_timestamp,
            });
        }

        Ok(Self { index, db: db.clone() })
    }

    /// Rebuild index if out of date
    pub fn load_or_rebuild(
        index_path: &Path,
        db: &Lean5Db,
        config: &EmbeddingConfig,
    ) -> Result<Self, SearchError> {
        match Self::load(index_path, db) {
            Ok(engine) => Ok(engine),
            Err(SearchError::IndexOutOfDate { .. }) => {
                log::info!("Rebuilding out-of-date search index");
                let index = build_search_index(db, config)?;
                index.save(index_path)?;
                Ok(Self { index, db: db.clone() })
            }
            Err(e) => Err(e),
        }
    }
}
```

**Implementation:**
1. Store source hash in index header
2. Verify on load, fail if mismatch
3. Provide auto-rebuild option
4. CLI: `lean5-search rebuild-index`

---

### Flaw 23 (Search 3): Type Unification Complexity
**Severity:** CRITICAL
**Problem:** Pattern matching with wildcards can hang.

**Solution: Bounded Unification**

```rust
/// Unification with resource limits
pub struct UnificationConfig {
    /// Maximum unification steps
    pub max_steps: u32,

    /// Maximum recursion depth
    pub max_depth: u32,

    /// Timeout in milliseconds
    pub timeout_ms: u32,

    /// Allow higher-order unification
    pub higher_order: bool,
}

impl Default for UnificationConfig {
    fn default() -> Self {
        Self {
            max_steps: 10_000,
            max_depth: 100,
            timeout_ms: 100,
            higher_order: false,  // Decidable fragment only
        }
    }
}

pub struct BoundedUnifier {
    config: UnificationConfig,
    steps: u32,
    depth: u32,
    start_time: Instant,
}

impl BoundedUnifier {
    pub fn unify(
        &mut self,
        pattern: &TypePattern,
        target: &Expr,
    ) -> Result<Option<Substitution>, UnificationError> {
        self.check_limits()?;

        match (pattern, target) {
            // Wildcard matches anything
            (TypePattern::Wildcard, _) => Ok(Some(Substitution::empty())),

            // Named wildcard - must match consistently
            (TypePattern::NamedWildcard(name), expr) => {
                Ok(Some(Substitution::single(name.clone(), expr.clone())))
            }

            // Concrete constant - must match exactly
            (TypePattern::Const(p_name, p_levels), Expr::Const(t_name, t_levels)) => {
                if p_name == t_name && p_levels.len() == t_levels.len() {
                    // Unify level parameters
                    self.unify_levels(p_levels, t_levels)
                } else {
                    Ok(None)
                }
            }

            // Application - unify both parts
            (TypePattern::App(p_fn, p_arg), Expr::App(t_fn, t_arg)) => {
                self.depth += 1;
                let fn_subst = self.unify(p_fn, t_fn)?;
                self.depth -= 1;

                let fn_subst = match fn_subst {
                    Some(s) => s,
                    None => return Ok(None),
                };

                self.depth += 1;
                let arg_subst = self.unify(
                    &p_arg.apply(&fn_subst),
                    &t_arg.apply(&fn_subst),
                )?;
                self.depth -= 1;

                match arg_subst {
                    Some(s) => Ok(Some(fn_subst.compose(s))),
                    None => Ok(None),
                }
            }

            // ... other cases

            _ => Ok(None),
        }
    }

    fn check_limits(&mut self) -> Result<(), UnificationError> {
        self.steps += 1;

        if self.steps > self.config.max_steps {
            return Err(UnificationError::TooManySteps);
        }

        if self.depth > self.config.max_depth {
            return Err(UnificationError::TooDeep);
        }

        if self.start_time.elapsed().as_millis() > self.config.timeout_ms as u128 {
            return Err(UnificationError::Timeout);
        }

        Ok(())
    }
}

/// Pre-filter candidates before unification
pub fn type_search(
    pattern: &TypePattern,
    index: &TypeIndex,
    config: &UnificationConfig,
) -> Vec<SearchResult> {
    // Fast structural filter using fingerprints
    let candidates = index.filter_by_fingerprint(pattern.fingerprint());

    // Unify each candidate
    let mut results = vec![];
    for candidate in candidates {
        let mut unifier = BoundedUnifier::new(config.clone());
        match unifier.unify(pattern, &candidate.type_expr) {
            Ok(Some(subst)) => {
                results.push(SearchResult {
                    constant_id: candidate.id,
                    match_type: MatchType::Type,
                    substitution: Some(subst),
                    score: compute_score(&candidate, &subst),
                });
            }
            Ok(None) => {} // No match
            Err(UnificationError::Timeout) => {
                log::warn!("Unification timeout for {}", candidate.name);
            }
            Err(e) => {
                log::debug!("Unification error for {}: {:?}", candidate.name, e);
            }
        }
    }

    results
}
```

**Implementation:**
1. Default to first-order unification only
2. Strict resource limits (steps, depth, time)
3. Fingerprint pre-filtering reduces candidates
4. Return partial results on timeout

---

### Flaw 24 (Search 4): Memory Pressure from Semantic Index
**Severity:** CRITICAL
**Problem:** 500MB+ for full Mathlib embeddings.

**Solution: Memory-Mapped + Quantized Embeddings**

```rust
/// Quantized embedding for memory efficiency
pub enum QuantizedEmbedding {
    /// Full float32 (384 dims = 1536 bytes)
    Float32(Vec<f32>),

    /// Float16 (384 dims = 768 bytes)
    Float16(Vec<f16>),

    /// Int8 quantized (384 dims = 384 bytes + scale)
    Int8 { values: Vec<i8>, scale: f32, bias: f32 },

    /// Binary (384 dims = 48 bytes) - for initial filtering
    Binary(BitVec),
}

impl QuantizedEmbedding {
    pub fn quantize_int8(embedding: &[f32]) -> Self {
        let min = embedding.iter().copied().reduce(f32::min).unwrap();
        let max = embedding.iter().copied().reduce(f32::max).unwrap();
        let scale = (max - min) / 255.0;
        let bias = min;

        let values: Vec<i8> = embedding.iter()
            .map(|&v| ((v - bias) / scale - 128.0) as i8)
            .collect();

        Self::Int8 { values, scale, bias }
    }

    pub fn to_float32(&self) -> Vec<f32> {
        match self {
            Self::Float32(v) => v.clone(),
            Self::Float16(v) => v.iter().map(|x| x.to_f32()).collect(),
            Self::Int8 { values, scale, bias } => {
                values.iter()
                    .map(|&v| (v as f32 + 128.0) * scale + bias)
                    .collect()
            }
            Self::Binary(bits) => {
                bits.iter().map(|b| if b { 1.0 } else { -1.0 }).collect()
            }
        }
    }

    pub fn dot_product(&self, other: &Self) -> f32 {
        match (self, other) {
            (Self::Int8 { values: a, .. }, Self::Int8 { values: b, .. }) => {
                // SIMD-friendly integer dot product
                a.iter().zip(b.iter())
                    .map(|(&x, &y)| (x as i32) * (y as i32))
                    .sum::<i32>() as f32
            }
            (Self::Binary(a), Self::Binary(b)) => {
                // Hamming distance
                let matches = a.iter().zip(b.iter())
                    .filter(|(x, y)| x == y)
                    .count();
                (2 * matches as i32 - a.len() as i32) as f32
            }
            _ => {
                // Fall back to float
                let a = self.to_float32();
                let b = other.to_float32();
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            }
        }
    }
}

/// Memory-mapped semantic index
pub struct MmapSemanticIndex {
    /// Memory-mapped embedding data
    mmap: Mmap,

    /// Index metadata (in memory)
    metadata: SemanticIndexMetadata,

    /// HNSW graph (in memory, smaller than embeddings)
    hnsw_graph: HnswGraph,
}

impl MmapSemanticIndex {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let metadata = SemanticIndexMetadata::parse(&mmap)?;
        let hnsw_graph = HnswGraph::parse(&mmap[metadata.hnsw_offset..])?;

        Ok(Self { mmap, metadata, hnsw_graph })
    }

    /// Get embedding by ID (zero-copy from mmap)
    pub fn get_embedding(&self, id: ConstantId) -> QuantizedEmbedding {
        let offset = self.metadata.embedding_offset
            + (id.0 as usize) * self.metadata.embedding_size;
        let bytes = &self.mmap[offset..offset + self.metadata.embedding_size];

        match self.metadata.quantization {
            Quantization::Int8 => {
                let values: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
                QuantizedEmbedding::Int8 {
                    values,
                    scale: self.metadata.int8_scale,
                    bias: self.metadata.int8_bias,
                }
            }
            // ... other quantization types
        }
    }
}
```

**Memory comparison (130K constants, 384 dims):**

| Format | Size | Load Time |
|--------|------|-----------|
| Float32 | 200 MB | 150ms |
| Float16 | 100 MB | 80ms |
| Int8 | 50 MB | 40ms |
| Binary (initial filter) | 6 MB | 5ms |

**Implementation:**
1. Default to int8 quantization
2. Binary embeddings for initial candidate filtering
3. Memory-mapped for zero-copy access
4. HNSW graph in memory (small relative to embeddings)

---

### Flaw 25 (Search 5): Query Injection
**Severity:** CRITICAL
**Problem:** Malformed patterns could exploit parser.

**Solution: Strict Grammar + Sandboxed Evaluation**

```rust
/// Query parser with strict grammar
pub struct QueryParser {
    /// Maximum query length
    max_length: usize,

    /// Maximum pattern depth
    max_depth: u32,

    /// Allowed operators
    allowed_ops: HashSet<String>,
}

impl QueryParser {
    pub fn parse(&self, input: &str) -> Result<ParsedQuery, ParseError> {
        // Length check
        if input.len() > self.max_length {
            return Err(ParseError::TooLong {
                len: input.len(),
                max: self.max_length
            });
        }

        // Tokenize with strict lexer
        let tokens = self.tokenize(input)?;

        // Parse with depth tracking
        let mut parser = QueryParserState {
            tokens: &tokens,
            pos: 0,
            depth: 0,
            max_depth: self.max_depth,
        };

        parser.parse_query()
    }

    fn tokenize(&self, input: &str) -> Result<Vec<Token>, ParseError> {
        let mut tokens = vec![];
        let mut chars = input.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                // Whitespace
                ' ' | '\t' | '\n' => { chars.next(); }

                // Operators
                ':' | '(' | ')' | ',' | '|' => {
                    tokens.push(Token::Op(c.to_string()));
                    chars.next();
                }

                // Keywords and identifiers
                'a'..='z' | 'A'..='Z' | '_' => {
                    let ident = self.read_identifier(&mut chars);
                    tokens.push(Token::Ident(ident));
                }

                // Quoted strings
                '"' => {
                    let s = self.read_string(&mut chars)?;
                    tokens.push(Token::String(s));
                }

                // Unicode math (limited set)
                '∀' | '→' | '∃' | '∧' | '∨' | '¬' => {
                    tokens.push(Token::MathSymbol(c));
                    chars.next();
                }

                // Invalid character
                _ => {
                    return Err(ParseError::InvalidCharacter(c));
                }
            }
        }

        Ok(tokens)
    }
}

/// Sandboxed query execution
pub struct SandboxedExecutor {
    /// Resource limits
    limits: ExecutionLimits,

    /// Search engine
    engine: SearchEngine,
}

pub struct ExecutionLimits {
    pub max_results: usize,
    pub timeout_ms: u32,
    pub max_candidates: usize,
    pub max_memory_bytes: usize,
}

impl SandboxedExecutor {
    pub fn execute(&self, query: &ParsedQuery) -> Result<SearchResults, ExecutionError> {
        let start = Instant::now();
        let mut candidates_checked = 0;
        let mut results = vec![];

        for candidate in self.engine.candidates(query) {
            // Check limits
            if start.elapsed().as_millis() > self.limits.timeout_ms as u128 {
                return Ok(SearchResults {
                    results,
                    truncated: true,
                    reason: TruncationReason::Timeout,
                });
            }

            candidates_checked += 1;
            if candidates_checked > self.limits.max_candidates {
                return Ok(SearchResults {
                    results,
                    truncated: true,
                    reason: TruncationReason::TooManyCandidates,
                });
            }

            // Evaluate candidate
            if let Some(result) = self.evaluate_candidate(query, &candidate)? {
                results.push(result);

                if results.len() >= self.limits.max_results {
                    break;
                }
            }
        }

        Ok(SearchResults {
            results,
            truncated: false,
            reason: TruncationReason::None,
        })
    }
}
```

**Implementation:**
1. Strict lexer - whitelist allowed characters
2. Parser depth limits
3. Execution sandbox with timeout
4. Maximum candidates/results limits

---

### Flaws 26-35: Remaining Search Flaws

For brevity, here are summarized solutions for the remaining flaws:

| Flaw | Problem | Solution |
|------|---------|----------|
| 26. Tokenization ambiguity | Multiple tokenizations | Canonical spec: `Nat.add_comm` → `["Nat", "add", "comm"]` with dot-splitting |
| 27. No personalization | Same results for all | User profile with search history, opt-in click tracking |
| 28. Model dependency | Model change invalidates index | Store model ID in index, support multiple models |
| 29. No negative search | Can't exclude results | Add `NOT`/`-` operator: `ring -abelian` |
| 30. Cross-reference blindness | No dependency info | Build dependency graph during indexing |
| 31. Unicode handling | `∀` vs `forall` | Normalize to ASCII during indexing, accept both in queries |
| 32. No faceted search | Can't browse by category | Extract Mathlib hierarchy as facets |
| 33. Match opacity | User doesn't know why | Include explanation in results |
| 34. No federation | Single index only | Federated search across multiple indexes |
| 35. Stale embedding cache | Content change not reflected | Hash validation on query |

---

## Summary: Implementation Priority

### Must Fix Before v1.0 (Blockers)

| Flaw | Solution | Effort |
|------|----------|--------|
| 1. Version coupling | Multi-layer versioning | 3 days |
| 7. Topological ordering | Sort metadata + ordered chunks | 2 days |
| 14. Chunk splitting | Constraint-aware assignment | 2 days |
| 5. Unbounded cache | LRU with size limit | 1 day |
| 22. Index staleness | Hash in index header | 1 day |
| 23. Unification complexity | Bounded unification | 2 days |
| 25. Query injection | Strict grammar + sandbox | 2 days |

### Should Include in v1.0 (High Value)

| Flaw | Solution | Effort |
|------|----------|--------|
| 2. No incremental updates | Delta format | 5 days |
| 11. No migration path | Migration framework | 3 days |
| 19. No signatures | Ed25519 signing | 2 days |
| 21. Embedding quality | Math-specialized models | 5 days |
| 24. Memory pressure | Quantization + mmap | 3 days |

### Can Defer to v1.1

- Flaw 3: String size (increase to varint if needed)
- Flaw 6: Error correction (optional ECC)
- Flaw 12: Redundant headers
- Flaw 17: Custom attributes
- Search personalization (27)
- Federation (34)

### Won't Fix (Acceptable Trade-offs)

- Flaw 9/20: Dictionary versioning (rebuild is fast enough)
- Flaw 15: Partial library (lazy loading sufficient)

---

## Appendix: Complete Flaw Index

| # | Flaw | Source | Severity | Solution Section |
|---|------|--------|----------|------------------|
| 1 | Version coupling | FORMAT_ANALYSIS | HIGH | A.1 |
| 2 | No incremental updates | FORMAT_ANALYSIS | HIGH | A.2 |
| 3 | String size limit | FORMAT_ANALYSIS | MEDIUM | A.3 |
| 4 | Expression pool DAG | FORMAT_ANALYSIS | MEDIUM | A.4 |
| 5 | Unbounded cache | FORMAT_ANALYSIS | MEDIUM | A.5 |
| 6 | No error recovery | FORMAT_ANALYSIS | LOW | A.6 |
| 7 | Ordering not preserved | FORMAT_ANALYSIS | MEDIUM | A.7 |
| 8 | Mutual inductives | FORMAT_ANALYSIS | MEDIUM | A.8 |
| 9 | Dictionary versioning | FORMAT_ANALYSIS | LOW | A.9 |
| 10 | No partial build | FORMAT_ANALYSIS | MEDIUM | A.10 |
| 11 | No migration path | VERIFICATION | HIGH | B.11 |
| 12 | Single header | VERIFICATION | MEDIUM | B.12 |
| 13 | No streaming write | VERIFICATION | MEDIUM | B.13 |
| 14 | Chunk splits groups | VERIFICATION | HIGH | B.14 |
| 15 | No partial library | VERIFICATION | LOW | B.15 |
| 16 | Pool size unknown | VERIFICATION | MEDIUM | B.16 |
| 17 | No custom attributes | VERIFICATION | MEDIUM | B.17 |
| 18 | TOCTOU | VERIFICATION | LOW | B.18 |
| 19 | No signatures | VERIFICATION | MEDIUM | B.19 |
| 20 | Dictionary not shareable | VERIFICATION | LOW | A.9 |
| 21 | Embedding quality | SEARCH | CRITICAL | C.21 |
| 22 | Index staleness | SEARCH | CRITICAL | C.22 |
| 23 | Unification complexity | SEARCH | CRITICAL | C.23 |
| 24 | Memory pressure | SEARCH | CRITICAL | C.24 |
| 25 | Query injection | SEARCH | CRITICAL | C.25 |
| 26 | Tokenization | SEARCH | SIGNIFICANT | C.26 |
| 27 | No personalization | SEARCH | SIGNIFICANT | C.26 |
| 28 | Model dependency | SEARCH | SIGNIFICANT | C.26 |
| 29 | No negative search | SEARCH | SIGNIFICANT | C.26 |
| 30 | Cross-reference | SEARCH | SIGNIFICANT | C.26 |
| 31 | Unicode handling | SEARCH | MINOR | C.26 |
| 32 | No faceted search | SEARCH | MINOR | C.26 |
| 33 | Match opacity | SEARCH | MINOR | C.26 |
| 34 | No federation | SEARCH | MINOR | C.26 |
| 35 | Stale embeddings | SEARCH | MINOR | C.26 |
