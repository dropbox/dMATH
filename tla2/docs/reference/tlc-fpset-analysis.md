# TLC FPSet Architecture Analysis

**Date:** 2026-01-10
**Purpose:** Document TLC's fingerprint storage to inform TLA2 improvements for #49

## TLC FPSet Hierarchy

```
FPSet (abstract)
├── MemFPSet          - Pure in-memory HashSet (small specs)
├── DiskFPSet         - Abstract disk-backed storage
│   ├── HeapBasedDiskFPSet    - On-heap memory + disk
│   │   ├── MSBDiskFPSet      - MSB-indexed (default)
│   │   └── LSBDiskFPSet      - LSB-indexed variant
│   ├── OffHeapDiskFPSet      - Off-heap mmap + disk (large specs)
│   └── NonCheckpointableDiskFPSet
└── MultiFPSet        - Sharded wrapper for parallel workers
```

## Key Design: Two-Tier Storage

TLC's DiskFPSet uses a two-tier architecture:

1. **Primary (In-Memory)**: Hash table or open-addressing array
   - Fixed maximum size (`maxTblCnt`)
   - Fast O(1) insert and lookup
   - Uses MSB bit to mark "flushed" entries

2. **Secondary (Disk)**: Sorted file with index
   - Grows unboundedly (billions of fingerprints)
   - Sorted 8-byte fingerprints
   - Index array stores first FP per page (8KB pages)
   - Interpolated binary search: ~1.05 seeks per lookup

### Eviction Flow

When primary fills up (`tblCnt >= maxTblCnt`):

1. **Signal eviction**: First thread to detect sets `flusherChosen` flag
2. **Synchronize workers**: All workers pause via Phaser/CyclicBarrier
3. **Flush to disk**: Sort in-memory FPs, merge with existing disk file
4. **Mark flushed**: Set MSB on flushed entries (keeps them in memory for contains())
5. **Reset**: Clear flusherChosen, resume workers

### Put Operation (OffHeapDiskFPSet)

```java
public boolean put(long fp) {
    if (checkEvictPending()) return put(fp);  // Wait for eviction

    long fp0 = fp & FLUSHED_MASK;  // Clear MSB

    if (index != null) {  // Disk file exists
        if (memLookup(fp0)) return true;    // In memory
        if (diskLookup(fp0)) return true;   // On disk
    }

    return memInsert(fp0);  // Try to insert
}
```

### Contains Operation

```java
public boolean contains(long fp) {
    if (checkEvictPending()) return contains(fp);

    long fp0 = fp & FLUSHED_MASK;
    if (memLookup(fp0)) return true;
    if (diskLookup(fp0)) return true;
    return false;
}
```

### Disk Lookup (Interpolated Binary Search)

```java
boolean diskLookup(long fp) {
    // Use index array to find disk page
    int lo = 0, hi = index.length - 1;
    while (lo < hi) {
        // Interpolate position based on FP value distribution
        int mid = interpolate(lo, hi, fp);
        if (index[mid] <= fp && fp < index[mid+1]) {
            // Found page - read and search
            return searchPage(mid, fp);
        }
        // Binary search narrowing
    }
}
```

## TLA2 Current State (storage.rs)

TLA2 has:
- `MmapFingerprintSet`: Open addressing with linear probing (like TLC's OffHeapDiskFPSet)
- `ShardedFingerprintSet`: Sharded in-memory (like TLC's MultiFPSet)
- Fixed capacity - **fails when full**

## Gap: No Disk Overflow

TLA2's critical limitation: when `MmapFingerprintSet` reaches capacity, it returns an error. TLC continues by flushing to disk.

## Implementation Plan for TLA2

### Phase 1: Disk-Backed Storage

Add `DiskFingerprintSet` that:
1. Uses existing `MmapFingerprintSet` as primary
2. Adds sorted disk file as secondary
3. Implements eviction when primary fills (sort + merge)
4. Implements disk lookup (interpolated binary search)

### Key Components

```rust
pub struct DiskFingerprintSet {
    /// Primary: in-memory storage
    primary: MmapFingerprintSet,

    /// Secondary: sorted disk file
    disk_file: Option<File>,
    disk_count: u64,

    /// Index: first FP per page for binary search
    page_index: Vec<u64>,

    /// Synchronization for eviction
    evict_flag: AtomicBool,

    /// Metadata directory
    metadir: PathBuf,
}
```

### Eviction Algorithm

```rust
fn evict(&mut self) -> io::Result<()> {
    // 1. Collect all fingerprints from primary
    let mut fps: Vec<u64> = self.collect_primary();

    // 2. Sort them
    fps.sort_unstable();

    // 3. Merge with existing disk file (if any)
    let merged = self.merge_with_disk(&fps)?;

    // 4. Write new sorted file
    self.write_disk_file(&merged)?;

    // 5. Build page index
    self.build_index(&merged);

    // 6. Clear primary (or mark as flushed)
    self.primary.clear();

    Ok(())
}
```

### Priority

This is P1 work (#49) because:
- Enables billion-state specs
- Current limitation blocks large specifications
- TLC achieves this with similar architecture

## References

- TLC source: `~/tlaplus/tlatools/org.lamport.tlatools/src/tlc2/tool/fp/`
- Key files:
  - `DiskFPSet.java` (35K lines) - Base implementation
  - `OffHeapDiskFPSet.java` (52K lines) - Large-scale variant
  - `MultiFPSet.java` (7K lines) - Parallel sharding
