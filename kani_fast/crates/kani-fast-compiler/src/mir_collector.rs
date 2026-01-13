//! MIR collection from rustc
//!
//! This module extracts MIR (Mid-level IR) from the Rust compiler
//! and prepares it for conversion to our internal representation.

use crate::kani_middle::attributes::is_proof_harness_from_def_id;
use rustc_hir::def::DefKind;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use std::collections::HashMap;

/// Collects MIR from the Rust compiler
pub struct MirCollector {
    /// Functions we've collected (name -> MIR exists flag)
    collected: HashMap<String, bool>,
}

impl MirCollector {
    /// Create a new MIR collector
    pub fn new() -> Self {
        Self {
            collected: HashMap::new(),
        }
    }

    /// Collect MIR for all functions in the crate
    ///
    /// Returns a map of function names to whether they have MIR.
    /// The actual MIR is not stored (lifetime issues) - we process it inline.
    pub fn collect_all_mir(&mut self, tcx: TyCtxt<'_>) -> Vec<(String, bool)> {
        let mut result = Vec::new();

        // Iterate over all items in the crate
        for item_id in tcx.hir_free_items() {
            let def_id = item_id.owner_id.to_def_id();

            // Check if this is a function
            if let DefKind::Fn = tcx.def_kind(def_id) {
                let name = tcx.def_path_str(def_id);

                // Check if we can get optimized MIR for this function
                if tcx.is_mir_available(def_id) {
                    let _body = tcx.optimized_mir(def_id);
                    self.collected.insert(name.clone(), true);
                    result.push((name, true));
                } else {
                    self.collected.insert(name.clone(), false);
                    result.push((name, false));
                }
            }
        }

        result
    }

    /// Get the names of all collected functions
    pub fn function_names(&self) -> impl Iterator<Item = &str> {
        self.collected.keys().map(|s| s.as_str())
    }

    /// Check if a function was collected
    pub fn has_function(&self, name: &str) -> bool {
        self.collected.get(name).is_some_and(|&v| v)
    }
}

impl Default for MirCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Find all functions with `#[kani::proof]` attribute
///
/// Note: This is a placeholder. Full implementation would need to check
/// for the kani::proof attribute specifically.
pub fn find_proof_harnesses(tcx: TyCtxt<'_>) -> Vec<DefId> {
    let mut harnesses = Vec::new();

    for item_id in tcx.hir_free_items() {
        let def_id = item_id.owner_id.to_def_id();

        if let DefKind::Fn = tcx.def_kind(def_id) {
            // Use the internal DefId version to avoid TLV context issues with rustc_internal
            if is_proof_harness_from_def_id(tcx, def_id) {
                harnesses.push(def_id);
                continue;
            }

            // Fallback for legacy naming when attributes are absent
            let name = tcx.def_path_str(def_id);
            if name.ends_with("_proof") || name.contains("::test_") {
                harnesses.push(def_id);
            }
        }
    }

    harnesses
}
