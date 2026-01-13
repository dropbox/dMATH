//! Clause representation
//!
//! Implements tier-based clause management with three tiers:
//! - CORE (LBD <= 2): "Glue" clauses, never deleted
//! - TIER1 (LBD <= 6): Important clauses, kept if recently used
//! - TIER2 (LBD > 6): Less important, deleted based on activity

use crate::literal::Literal;

/// Clause tier based on LBD (Literal Block Distance)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseTier {
    /// Core/Glue clauses (LBD <= 2) - never deleted
    Core,
    /// Tier 1 clauses (2 < LBD <= 6) - kept if recently used
    Tier1,
    /// Tier 2 clauses (LBD > 6) - deleted based on activity
    Tier2,
}

/// LBD threshold for core/glue clauses
pub const CORE_LBD: u32 = 2;
/// LBD threshold for tier 1 clauses
pub const TIER1_LBD: u32 = 6;

/// A clause (disjunction of literals)
#[derive(Debug, Clone)]
pub struct Clause {
    /// The literals in this clause
    pub literals: Box<[Literal]>,
    /// Literal Block Distance (for learned clauses)
    pub lbd: u32,
    /// Activity (for clause deletion)
    pub activity: f32,
    /// Is this a learned clause?
    pub learned: bool,
    /// Usage counter - decremented each reduce, clause survives if > 0
    /// Used for tier-based protection
    pub used: u8,
}

impl Clause {
    /// Create a new clause
    pub fn new(literals: Vec<Literal>, learned: bool) -> Self {
        Clause {
            literals: literals.into_boxed_slice(),
            lbd: 0,
            activity: 0.0,
            learned,
            used: 0,
        }
    }

    /// Replace the clause's literals, shrinking storage to fit exactly.
    #[inline]
    pub fn set_literals(&mut self, literals: Vec<Literal>) {
        self.literals = literals.into_boxed_slice();
    }

    /// Mark this clause as deleted by clearing its literals.
    #[inline]
    pub fn clear_literals(&mut self) {
        self.literals = Box::new([]);
    }

    /// Get the tier of this clause based on LBD
    #[inline]
    pub fn tier(&self) -> ClauseTier {
        if self.lbd <= CORE_LBD {
            ClauseTier::Core
        } else if self.lbd <= TIER1_LBD {
            ClauseTier::Tier1
        } else {
            ClauseTier::Tier2
        }
    }

    /// Mark this clause as recently used (for tier-based protection)
    #[inline]
    pub fn mark_used(&mut self) {
        // Cap at 2 to avoid overflow, represents "used recently"
        self.used = self.used.saturating_add(1).min(2);
    }

    /// Decay the used counter (called during reduce)
    #[inline]
    pub fn decay_used(&mut self) {
        self.used = self.used.saturating_sub(1);
    }

    /// Get the number of literals
    #[inline]
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if unit clause
    #[inline]
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }
}
