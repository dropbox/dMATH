//! Assigns clause checking for verification condition generation
//!
//! This module contains methods for checking that all modified locations
//! in a function body are covered by the assigns clause in its specification.

use crate::expr::BinOp;
use crate::spec::{Location, Spec};

use super::{ModifiedLocation, VCGen, VCKind, VC};

impl VCGen {
    /// Generate VCs for assigns clause checking
    ///
    /// Verifies that all modified locations are in the assigns clause.
    /// For each modified location not covered by assigns, generates a VC.
    pub fn check_assigns(
        &mut self,
        assigns: &[Location],
        modified: &[ModifiedLocation],
    ) -> Vec<VC> {
        let mut vcs = Vec::new();

        // If assigns is empty or contains Everything, no checking needed
        if assigns.iter().any(|l| matches!(l, Location::Everything)) {
            return vcs;
        }

        // For each modified location, check if it's covered by assigns clause
        for modified_loc in modified {
            let covered = self.is_location_covered(&modified_loc.location, assigns);

            if !covered {
                // Generate VC: location must be in assigns
                let vc_obligation = self.location_in_assigns_vc(&modified_loc.location, assigns);

                vcs.push(VC {
                    description: format!(
                        "Modified location '{}' must be in assigns clause",
                        modified_loc.description
                    ),
                    obligation: vc_obligation,
                    location: modified_loc.source_line,
                    kind: VCKind::AssignsClause,
                });
            }
        }

        vcs
    }

    /// Check if a location is syntactically covered by the assigns clause
    fn is_location_covered(&self, loc: &Location, assigns: &[Location]) -> bool {
        // Simple syntactic check - conservative (may return false for semantically covered locations)
        for assign_loc in assigns {
            match (loc, assign_loc) {
                (_, Location::Everything) | (Location::Nothing, _) => return true,
                (Location::Deref(e1), Location::Deref(e2)) if e1 == e2 => return true,
                (
                    Location::Deref(Spec::Var(v1)),
                    Location::Range {
                        base: Spec::Var(v2),
                        lo,
                        hi: _,
                    },
                ) => {
                    // *p might be covered by base[lo..hi] if p == base + offset where lo <= offset < hi
                    // This is a complex semantic check - for now, check exact match
                    // Simple case: *x covered by x[0..n]
                    if v1 == v2 {
                        // Check if lo = 0, which covers *x
                        if *lo == Spec::int(0) {
                            return true;
                        }
                    }
                }
                (Location::Deref(ptr_expr), Location::Range { base, lo, hi }) => {
                    // Handle simple pointer arithmetic: *(base + k) within base[lo..hi]
                    if let Some((ptr_base, offset)) = Self::extract_base_offset(ptr_expr) {
                        if ptr_base == *base {
                            if let (Some(lo_val), Some(hi_val)) =
                                (Self::extract_int(lo), Self::extract_int(hi))
                            {
                                if offset >= lo_val && offset < hi_val {
                                    return true;
                                }
                            }
                        }
                    }
                }
                (
                    Location::Range {
                        base: b1,
                        lo: l1,
                        hi: h1,
                    },
                    Location::Range {
                        base: b2,
                        lo: l2,
                        hi: h2,
                    },
                ) => {
                    // Range1 subset of Range2?
                    if b1 == b2 && l1 == l2 && h1 == h2 {
                        return true;
                    }

                    // If both ranges have constant bounds, check interval inclusion
                    if b1 == b2 {
                        if let (Some(l1_val), Some(h1_val), Some(l2_val), Some(h2_val)) = (
                            Self::extract_int(l1),
                            Self::extract_int(h1),
                            Self::extract_int(l2),
                            Self::extract_int(h2),
                        ) {
                            if l1_val >= l2_val && h1_val <= h2_val {
                                return true;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Extract integer literal if present
    pub(crate) fn extract_int(spec: &Spec) -> Option<i64> {
        match spec {
            Spec::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// Extract a base pointer and constant offset from a Spec, if syntactically simple
    ///
    /// Supported patterns:
    /// - `base` -> (base, 0)
    /// - `base + k` or `k + base` where k is integer
    /// - `base - k` where k is integer
    pub(crate) fn extract_base_offset(ptr: &Spec) -> Option<(Spec, i64)> {
        match ptr {
            Spec::Var(_) => Some((ptr.clone(), 0)),
            Spec::BinOp {
                op: BinOp::Add,
                left,
                right,
            } => {
                if let Some(offset) = Self::extract_int(right) {
                    if matches!(left.as_ref(), Spec::Var(_)) {
                        return Some((*left.clone(), offset));
                    }
                }
                if let Some(offset) = Self::extract_int(left) {
                    if matches!(right.as_ref(), Spec::Var(_)) {
                        return Some((*right.clone(), offset));
                    }
                }
                None
            }
            Spec::BinOp {
                op: BinOp::Sub,
                left,
                right,
            } => {
                if let Some(offset) = Self::extract_int(right) {
                    if matches!(left.as_ref(), Spec::Var(_)) {
                        return Some((*left.clone(), -offset));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Generate a VC proposition that a location is in the assigns clause
    fn location_in_assigns_vc(&self, loc: &Location, assigns: &[Location]) -> Spec {
        if assigns.is_empty() {
            // No assigns clause means nothing can be modified
            // VC: false (the modification is not allowed)
            return Spec::False;
        }

        // Build disjunction: loc in assign1 OR loc in assign2 OR ...
        let disjuncts: Vec<Spec> = assigns
            .iter()
            .map(|assign_loc| self.location_subset_of(loc, assign_loc))
            .collect();

        if disjuncts.len() == 1 {
            disjuncts.into_iter().next().unwrap()
        } else {
            Spec::or(disjuncts)
        }
    }

    /// Generate a proposition that loc1 is a subset of loc2
    pub(crate) fn location_subset_of(&self, loc1: &Location, loc2: &Location) -> Spec {
        match (loc1, loc2) {
            (_, Location::Everything) | (Location::Nothing, _) => Spec::True,
            (Location::Deref(e1), Location::Deref(e2)) => {
                // *e1 ⊆ *e2 iff e1 == e2
                Spec::eq(e1.clone(), e2.clone())
            }
            (Location::Deref(e), Location::Range { base, lo, hi }) => {
                // *e ⊆ base[lo..hi] iff base + lo <= e < base + hi
                let offset = Spec::binop(BinOp::Sub, e.clone(), base.clone());
                Spec::and(vec![
                    Spec::le(lo.clone(), offset.clone()),
                    Spec::lt(offset, hi.clone()),
                ])
            }
            (
                Location::Range {
                    base: b1,
                    lo: l1,
                    hi: h1,
                },
                Location::Range {
                    base: b2,
                    lo: l2,
                    hi: h2,
                },
            ) => {
                // b1[l1..h1] ⊆ b2[l2..h2] iff b1 == b2 and l2 <= l1 and h1 <= h2
                Spec::and(vec![
                    Spec::eq(b1.clone(), b2.clone()),
                    Spec::le(l2.clone(), l1.clone()),
                    Spec::le(h1.clone(), h2.clone()),
                ])
            }
            (Location::Reachable(e1), Location::Reachable(e2)) => {
                // reachable(e1) ⊆ reachable(e2) if e1 == e2
                // This is a simplification; full analysis would need shape analysis
                Spec::eq(e1.clone(), e2.clone())
            }
            _ => {
                // Conservative: cannot prove subset
                Spec::False
            }
        }
    }
}
