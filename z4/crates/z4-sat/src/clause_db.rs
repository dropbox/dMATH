//! Clause database with arena-allocated literals
//!
//! Stores all clause literals in a single contiguous allocation to reduce
//! memory overhead from per-clause heap allocations.
//!
//! Memory savings vs Box<[Literal]> per clause:
//! - Current: 32 bytes header + 4*N bytes literals (separate heap alloc each)
//! - Arena: 16 bytes header + 4*N bytes literals (single contiguous allocation)

use crate::clause::{ClauseTier, CORE_LBD, TIER1_LBD};
use crate::literal::Literal;

/// Start index into the literal arena
type LitStart = u32;
/// Number of literals in a clause
type LitLen = u16;

/// Compact clause header (16 bytes vs 32 bytes for Box<[Literal]> version)
#[derive(Debug, Clone)]
pub struct ClauseHeader {
    /// Start index in the literal arena
    lit_start: LitStart,
    /// Number of literals (max 65535)
    lit_len: LitLen,
    /// Literal Block Distance (for learned clauses)
    lbd: u16,
    /// Activity (for clause deletion)
    activity: f32,
    /// Flags: bit 0 = learned, bits 1-2 = used counter
    flags: u8,
}

impl ClauseHeader {
    /// Create a new clause header
    #[inline]
    fn new(lit_start: u32, lit_len: u16, learned: bool) -> Self {
        ClauseHeader {
            lit_start,
            lit_len,
            lbd: 0,
            activity: 0.0,
            flags: if learned { 1 } else { 0 },
        }
    }

    /// Check if this is a learned clause
    #[inline]
    pub fn is_learned(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Get the used counter (0-3)
    #[inline]
    pub fn used(&self) -> u8 {
        (self.flags >> 1) & 0x3
    }

    /// Set the used counter
    #[inline]
    pub fn set_used(&mut self, val: u8) {
        self.flags = (self.flags & 1) | ((val.min(3)) << 1);
    }

    /// Increment the used counter (saturating at 2)
    #[inline]
    pub fn mark_used(&mut self) {
        let current = self.used();
        self.set_used(current.saturating_add(1).min(2));
    }

    /// Decrement the used counter (saturating at 0)
    #[inline]
    pub fn decay_used(&mut self) {
        let current = self.used();
        self.set_used(current.saturating_sub(1));
    }

    /// Get the LBD
    #[inline]
    pub fn lbd(&self) -> u32 {
        self.lbd as u32
    }

    /// Set the LBD
    #[inline]
    pub fn set_lbd(&mut self, lbd: u32) {
        self.lbd = lbd.min(u16::MAX as u32) as u16;
    }

    /// Get the activity
    #[inline]
    pub fn activity(&self) -> f32 {
        self.activity
    }

    /// Set the activity
    #[inline]
    pub fn set_activity(&mut self, activity: f32) {
        self.activity = activity;
    }

    /// Get the tier based on LBD
    #[inline]
    pub fn tier(&self) -> ClauseTier {
        let lbd = self.lbd as u32;
        if lbd <= CORE_LBD {
            ClauseTier::Core
        } else if lbd <= TIER1_LBD {
            ClauseTier::Tier1
        } else {
            ClauseTier::Tier2
        }
    }

    /// Get the number of literals
    #[inline]
    pub fn len(&self) -> usize {
        self.lit_len as usize
    }

    /// Check if clause is empty (deleted)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lit_len == 0
    }
}

/// Clause database with arena-allocated literals
pub struct ClauseDB {
    /// Clause headers
    headers: Vec<ClauseHeader>,
    /// All literals stored contiguously
    literals: Vec<Literal>,
}

impl ClauseDB {
    /// Create a new empty clause database
    pub fn new() -> Self {
        ClauseDB {
            headers: Vec::new(),
            literals: Vec::new(),
        }
    }

    /// Create a clause database with pre-allocated capacity
    pub fn with_capacity(clauses: usize, literals: usize) -> Self {
        ClauseDB {
            headers: Vec::with_capacity(clauses),
            literals: Vec::with_capacity(literals),
        }
    }

    /// Number of clauses (including deleted ones)
    #[inline]
    pub fn len(&self) -> usize {
        self.headers.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    /// Add a new clause, returns its index
    pub fn add(&mut self, lits: &[Literal], learned: bool) -> usize {
        let idx = self.headers.len();
        let start = self.literals.len() as u32;
        let len = lits.len().min(u16::MAX as usize) as u16;

        self.literals.extend_from_slice(lits);
        self.headers.push(ClauseHeader::new(start, len, learned));

        idx
    }

    /// Get clause header by index
    #[inline]
    pub fn header(&self, idx: usize) -> &ClauseHeader {
        &self.headers[idx]
    }

    /// Get mutable clause header by index
    #[inline]
    pub fn header_mut(&mut self, idx: usize) -> &mut ClauseHeader {
        &mut self.headers[idx]
    }

    /// Get literals for a clause
    #[inline]
    pub fn literals(&self, idx: usize) -> &[Literal] {
        let h = &self.headers[idx];
        let start = h.lit_start as usize;
        let end = start + h.lit_len as usize;
        &self.literals[start..end]
    }

    /// Get mutable literals for a clause
    #[inline]
    pub fn literals_mut(&mut self, idx: usize) -> &mut [Literal] {
        let h = &self.headers[idx];
        let start = h.lit_start as usize;
        let end = start + h.lit_len as usize;
        &mut self.literals[start..end]
    }

    /// Swap two literals within a clause
    #[inline]
    pub fn swap_literals(&mut self, idx: usize, i: usize, j: usize) {
        let lits = self.literals_mut(idx);
        lits.swap(i, j);
    }

    /// Get a specific literal from a clause
    #[inline]
    pub fn literal(&self, clause_idx: usize, lit_idx: usize) -> Literal {
        let h = &self.headers[clause_idx];
        self.literals[h.lit_start as usize + lit_idx]
    }

    /// Mark a clause as deleted by setting its length to 0
    #[inline]
    pub fn delete(&mut self, idx: usize) {
        self.headers[idx].lit_len = 0;
    }

    /// Replace a clause's literals (for vivification/strengthening)
    ///
    /// Note: The old literals become garbage in the arena.
    /// Periodic compaction is needed to reclaim space.
    pub fn replace(&mut self, idx: usize, new_lits: &[Literal]) {
        let start = self.literals.len() as u32;
        let len = new_lits.len().min(u16::MAX as usize) as u16;

        self.literals.extend_from_slice(new_lits);
        self.headers[idx].lit_start = start;
        self.headers[idx].lit_len = len;
    }

    /// Total number of literals stored
    #[inline]
    pub fn total_literals(&self) -> usize {
        self.literals.len()
    }

    /// Number of active (non-deleted) literals
    pub fn active_literals(&self) -> usize {
        self.headers.iter().map(|h| h.lit_len as usize).sum()
    }

    /// Memory usage estimate in bytes
    pub fn memory_bytes(&self) -> usize {
        // Headers: Vec overhead + data
        let headers_mem = 24 + self.headers.capacity() * std::mem::size_of::<ClauseHeader>();
        // Literals: Vec overhead + data
        let literals_mem = 24 + self.literals.capacity() * std::mem::size_of::<Literal>();
        headers_mem + literals_mem
    }

    /// Header capacity
    #[inline]
    pub fn headers_capacity(&self) -> usize {
        self.headers.capacity()
    }

    /// Literals capacity
    #[inline]
    pub fn literals_capacity(&self) -> usize {
        self.literals.capacity()
    }

    /// Compact the literal arena by removing garbage from deleted/replaced clauses
    ///
    /// This rewrites all clause headers to point to new positions.
    pub fn compact(&mut self) {
        let mut new_literals = Vec::with_capacity(self.active_literals());

        for header in &mut self.headers {
            if header.lit_len == 0 {
                continue; // Skip deleted clauses
            }

            let old_start = header.lit_start as usize;
            let len = header.lit_len as usize;
            let new_start = new_literals.len() as u32;

            new_literals.extend_from_slice(&self.literals[old_start..old_start + len]);
            header.lit_start = new_start;
        }

        self.literals = new_literals;
    }

    /// Iterator over clause indices
    pub fn indices(&self) -> impl Iterator<Item = usize> {
        0..self.headers.len()
    }
}

impl Default for ClauseDB {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Variable;

    /// Create a literal from DIMACS-style signed integer
    fn lit(v: i32) -> Literal {
        let var = Variable(v.unsigned_abs() - 1);
        if v > 0 {
            Literal::positive(var)
        } else {
            Literal::negative(var)
        }
    }

    #[test]
    fn test_clause_header_size() {
        // Verify our compact header is actually 16 bytes
        assert_eq!(std::mem::size_of::<ClauseHeader>(), 16);
    }

    #[test]
    fn test_add_and_get() {
        let mut db = ClauseDB::new();

        let idx0 = db.add(&[lit(1), lit(2), lit(3)], false);
        let idx1 = db.add(&[lit(-1), lit(4)], true);

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(db.len(), 2);

        assert_eq!(db.literals(0), &[lit(1), lit(2), lit(3)]);
        assert_eq!(db.literals(1), &[lit(-1), lit(4)]);

        assert!(!db.header(0).is_learned());
        assert!(db.header(1).is_learned());
    }

    #[test]
    fn test_swap_literals() {
        let mut db = ClauseDB::new();
        db.add(&[lit(1), lit(2), lit(3)], false);

        db.swap_literals(0, 0, 2);
        assert_eq!(db.literals(0), &[lit(3), lit(2), lit(1)]);
    }

    #[test]
    fn test_delete_and_compact() {
        let mut db = ClauseDB::new();
        db.add(&[lit(1), lit(2)], false);
        db.add(&[lit(3), lit(4)], false);
        db.add(&[lit(5), lit(6)], false);

        // Delete middle clause
        db.delete(1);
        assert!(db.header(1).is_empty());

        // Before compact: 6 literals in arena
        assert_eq!(db.total_literals(), 6);

        // Compact
        db.compact();

        // After compact: only 4 active literals
        assert_eq!(db.total_literals(), 4);

        // Clauses still accessible
        assert_eq!(db.literals(0), &[lit(1), lit(2)]);
        assert_eq!(db.literals(2), &[lit(5), lit(6)]);
    }

    #[test]
    fn test_replace() {
        let mut db = ClauseDB::new();
        db.add(&[lit(1), lit(2), lit(3)], false);

        // Replace with shorter clause
        db.replace(0, &[lit(1), lit(2)]);

        assert_eq!(db.literals(0), &[lit(1), lit(2)]);
        assert_eq!(db.header(0).len(), 2);

        // Old literals are garbage, compact to reclaim
        db.compact();
        assert_eq!(db.total_literals(), 2);
    }

    #[test]
    fn test_header_flags() {
        let mut h = ClauseHeader::new(0, 3, true);

        assert!(h.is_learned());
        assert_eq!(h.used(), 0);

        h.mark_used();
        assert_eq!(h.used(), 1);

        h.mark_used();
        assert_eq!(h.used(), 2);

        h.mark_used(); // Should saturate at 2
        assert_eq!(h.used(), 2);

        h.decay_used();
        assert_eq!(h.used(), 1);

        // Learned flag should be preserved
        assert!(h.is_learned());
    }

    #[test]
    fn test_lbd_and_tier() {
        let mut h = ClauseHeader::new(0, 3, true);

        h.set_lbd(2);
        assert_eq!(h.tier(), ClauseTier::Core);

        h.set_lbd(5);
        assert_eq!(h.tier(), ClauseTier::Tier1);

        h.set_lbd(10);
        assert_eq!(h.tier(), ClauseTier::Tier2);
    }
}
