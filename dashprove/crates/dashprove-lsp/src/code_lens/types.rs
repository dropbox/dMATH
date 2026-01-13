//! Core types for code lens statistics.
//!
//! Provides shared property counting structures used by both document
//! and workspace statistics.

use crate::symbols::plural;
use dashprove_usl::Property;

/// Counts of properties by type.
///
/// Shared between `DocumentStats` and `WorkspaceStats` to eliminate duplication.
#[derive(Debug, Clone, Default)]
pub struct PropertyCounts {
    /// Total number of properties
    pub total: usize,
    /// Count of theorem properties
    pub theorems: usize,
    /// Count of contract properties
    pub contracts: usize,
    /// Count of temporal properties
    pub temporal: usize,
    /// Count of invariant properties
    pub invariants: usize,
    /// Count of refinement properties
    pub refinements: usize,
    /// Count of probabilistic properties
    pub probabilistic: usize,
    /// Count of security properties
    pub security: usize,
}

impl PropertyCounts {
    /// Add counts from a slice of properties.
    pub fn add_properties(&mut self, properties: &[Property]) {
        for prop in properties {
            self.total += 1;
            match prop {
                Property::Theorem(_) => self.theorems += 1,
                Property::Contract(_) => self.contracts += 1,
                Property::Temporal(_) => self.temporal += 1,
                Property::Invariant(_) => self.invariants += 1,
                Property::Refinement(_) => self.refinements += 1,
                Property::Probabilistic(_) => self.probabilistic += 1,
                Property::Security(_) => self.security += 1,
                Property::Semantic(_) => {}
                Property::PlatformApi(_) => {} // Not counted separately
                Property::Bisimulation(_) => {} // Not counted separately
                Property::Version(_) => {}     // Not counted separately
                Property::Capability(_) => {}  // Not counted separately
                Property::DistributedInvariant(_) => {} // Not counted separately
                Property::DistributedTemporal(_) => {} // Not counted separately
                Property::Composed(_) => self.theorems += 1, // Composed theorems count as theorems
                Property::ImprovementProposal(_) => {} // Not counted separately
                Property::VerificationGate(_) => {} // Not counted separately
                Property::Rollback(_) => {}    // Not counted separately
            }
        }
    }

    /// Merge another PropertyCounts into this one.
    pub fn merge(&mut self, other: &PropertyCounts) {
        self.total += other.total;
        self.theorems += other.theorems;
        self.contracts += other.contracts;
        self.temporal += other.temporal;
        self.invariants += other.invariants;
        self.refinements += other.refinements;
        self.probabilistic += other.probabilistic;
        self.security += other.security;
    }

    /// Format detailed breakdown by property type.
    pub fn detailed_breakdown(&self) -> Option<String> {
        let mut parts = Vec::new();

        if self.theorems > 0 {
            parts.push(format!("{} thm", self.theorems));
        }
        if self.contracts > 0 {
            parts.push(format!("{} ctr", self.contracts));
        }
        if self.temporal > 0 {
            parts.push(format!("{} tmp", self.temporal));
        }
        if self.invariants > 0 {
            parts.push(format!("{} inv", self.invariants));
        }
        if self.refinements > 0 {
            parts.push(format!("{} ref", self.refinements));
        }
        if self.probabilistic > 0 {
            parts.push(format!("{} prob", self.probabilistic));
        }
        if self.security > 0 {
            parts.push(format!("{} sec", self.security));
        }

        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" | "))
        }
    }

    /// Count the number of distinct property types present.
    pub fn type_variety(&self) -> usize {
        [
            self.theorems > 0,
            self.contracts > 0,
            self.temporal > 0,
            self.invariants > 0,
            self.refinements > 0,
            self.probabilistic > 0,
            self.security > 0,
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }
}

/// Format property/properties string with count.
pub fn format_properties_count(count: usize) -> String {
    format!("{} propert{}", count, if count == 1 { "y" } else { "ies" })
}

/// Format file/files string with count.
pub fn format_files_count(count: usize) -> String {
    format!("{} file{}", count, plural(count))
}

/// Format type/types string with count.
pub fn format_types_count(count: usize) -> String {
    format!("{} type{}", count, plural(count))
}
