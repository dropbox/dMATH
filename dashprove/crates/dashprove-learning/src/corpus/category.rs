//! Category-based indexing for fast filtered searches
//!
//! This module provides a secondary index that groups proof entries by their
//! PropertyCategory (hierarchical property type). This enables O(1) category
//! lookup instead of O(n) full corpus scan when filtering by category.
//!
//! # Performance
//!
//! For a corpus with n entries and m entries in category c:
//! - `by_category(c)`: O(1) lookup + O(m) iteration
//! - `find_similar_in_category(c, k)`: O(m log k) instead of O(n log k)
//!
//! When m << n (queries filtered to a specific domain), this provides
//! significant speedup for similarity searches.

use super::types::ProofId;
use crate::embedder::PropertyCategory;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Index mapping property categories to proof IDs
///
/// Maintains two levels of indexing:
/// 1. By coarse category (0-7) - for domain-level filtering
/// 2. By (category, subtype) pair - for fine-grained filtering
///
/// The index is not serialized - it's rebuilt from proof entries on load.
/// This avoids JSON key serialization issues with tuple keys.
#[derive(Debug, Clone, Default)]
pub struct CategoryIndex {
    /// Coarse category index: category -> proof IDs
    by_category: HashMap<usize, HashSet<ProofId>>,
    /// Fine-grained index: (category, subtype) -> proof IDs
    by_subtype: HashMap<(usize, usize), HashSet<ProofId>>,
}

// Manual Serialize/Deserialize that always produces empty index
// (we rebuild from proofs on load anyway)
impl Serialize for CategoryIndex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as empty object - index is rebuilt on load
        use serde::ser::SerializeMap;
        let map = serializer.serialize_map(Some(0))?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for CategoryIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Consume any value (we rebuild from proofs on load)
        // This handles both empty {} and any previously serialized data
        let _: serde::de::IgnoredAny = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self::default())
    }
}

impl CategoryIndex {
    /// Create a new empty category index
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a proof ID with its category
    pub fn insert(&mut self, id: ProofId, category: PropertyCategory) {
        self.by_category
            .entry(category.category)
            .or_default()
            .insert(id.clone());

        self.by_subtype
            .entry((category.category, category.subtype))
            .or_default()
            .insert(id);
    }

    /// Remove a proof ID from the index
    ///
    /// Requires the category to efficiently locate and remove the entry.
    /// Returns true if the ID was found and removed.
    pub fn remove(&mut self, id: &ProofId, category: PropertyCategory) -> bool {
        let removed_from_category = self
            .by_category
            .get_mut(&category.category)
            .map(|set| set.remove(id))
            .unwrap_or(false);

        let removed_from_subtype = self
            .by_subtype
            .get_mut(&(category.category, category.subtype))
            .map(|set| set.remove(id))
            .unwrap_or(false);

        removed_from_category || removed_from_subtype
    }

    /// Get all proof IDs in a category (coarse)
    pub fn by_category(&self, category: usize) -> impl Iterator<Item = &ProofId> {
        self.by_category
            .get(&category)
            .into_iter()
            .flat_map(|set| set.iter())
    }

    /// Get all proof IDs with a specific (category, subtype) pair
    pub fn by_subtype(&self, category: usize, subtype: usize) -> impl Iterator<Item = &ProofId> {
        self.by_subtype
            .get(&(category, subtype))
            .into_iter()
            .flat_map(|set| set.iter())
    }

    /// Count entries in a category
    pub fn category_count(&self, category: usize) -> usize {
        self.by_category
            .get(&category)
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Count entries with a specific (category, subtype) pair
    pub fn subtype_count(&self, category: usize, subtype: usize) -> usize {
        self.by_subtype
            .get(&(category, subtype))
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Total number of indexed entries
    ///
    /// Note: This iterates all categories. For frequent access, cache externally.
    pub fn total_count(&self) -> usize {
        self.by_category.values().map(|s| s.len()).sum()
    }

    /// Get all categories that have at least one entry
    pub fn nonempty_categories(&self) -> impl Iterator<Item = usize> + '_ {
        self.by_category
            .iter()
            .filter(|(_, set)| !set.is_empty())
            .map(|(cat, _)| *cat)
    }

    /// Get all (category, subtype) pairs that have at least one entry
    pub fn nonempty_subtypes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.by_subtype
            .iter()
            .filter(|(_, set)| !set.is_empty())
            .map(|(key, _)| *key)
    }

    /// Clear all entries from the index
    pub fn clear(&mut self) {
        self.by_category.clear();
        self.by_subtype.clear();
    }

    /// Rebuild the index from a collection of (id, category) pairs
    pub fn rebuild<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (ProofId, PropertyCategory)>,
    {
        self.clear();
        for (id, category) in entries {
            self.insert(id, category);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_category(cat: usize, sub: usize) -> PropertyCategory {
        PropertyCategory {
            category: cat,
            subtype: sub,
        }
    }

    #[test]
    fn test_empty_index() {
        let index = CategoryIndex::new();
        assert_eq!(index.total_count(), 0);
        assert_eq!(index.category_count(0), 0);
        assert_eq!(index.subtype_count(0, 0), 0);
        assert_eq!(index.by_category(0).count(), 0);
        assert_eq!(index.by_subtype(0, 0).count(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut index = CategoryIndex::new();

        let id1 = ProofId("proof_1".to_string());
        let id2 = ProofId("proof_2".to_string());
        let id3 = ProofId("proof_3".to_string());

        // Insert proofs in different categories
        index.insert(id1.clone(), make_category(0, 0)); // theorem
        index.insert(id2.clone(), make_category(0, 1)); // contract
        index.insert(id3.clone(), make_category(1, 0)); // temporal

        // Check category counts
        assert_eq!(index.category_count(0), 2);
        assert_eq!(index.category_count(1), 1);
        assert_eq!(index.category_count(2), 0);

        // Check subtype counts
        assert_eq!(index.subtype_count(0, 0), 1);
        assert_eq!(index.subtype_count(0, 1), 1);
        assert_eq!(index.subtype_count(1, 0), 1);

        // Check by_category iteration
        let cat0_ids: HashSet<_> = index.by_category(0).cloned().collect();
        assert!(cat0_ids.contains(&id1));
        assert!(cat0_ids.contains(&id2));
        assert!(!cat0_ids.contains(&id3));

        // Check by_subtype iteration
        let subtype_00_ids: Vec<_> = index.by_subtype(0, 0).collect();
        assert_eq!(subtype_00_ids.len(), 1);
        assert_eq!(subtype_00_ids[0], &id1);
    }

    #[test]
    fn test_remove() {
        let mut index = CategoryIndex::new();

        let id1 = ProofId("proof_1".to_string());
        let id2 = ProofId("proof_2".to_string());

        let cat = make_category(0, 0);
        index.insert(id1.clone(), cat);
        index.insert(id2.clone(), cat);

        assert_eq!(index.category_count(0), 2);

        // Remove one entry
        let removed = index.remove(&id1, cat);
        assert!(removed);
        assert_eq!(index.category_count(0), 1);

        // Verify only id2 remains
        let remaining: Vec<_> = index.by_category(0).collect();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0], &id2);

        // Try to remove non-existent
        let removed_again = index.remove(&id1, cat);
        assert!(!removed_again);
    }

    #[test]
    fn test_nonempty_categories() {
        let mut index = CategoryIndex::new();

        index.insert(ProofId("p1".to_string()), make_category(0, 0));
        index.insert(ProofId("p2".to_string()), make_category(2, 1));
        index.insert(ProofId("p3".to_string()), make_category(5, 2));

        let nonempty: HashSet<_> = index.nonempty_categories().collect();
        assert_eq!(nonempty.len(), 3);
        assert!(nonempty.contains(&0));
        assert!(nonempty.contains(&2));
        assert!(nonempty.contains(&5));
        assert!(!nonempty.contains(&1));
    }

    #[test]
    fn test_nonempty_subtypes() {
        let mut index = CategoryIndex::new();

        index.insert(ProofId("p1".to_string()), make_category(0, 0));
        index.insert(ProofId("p2".to_string()), make_category(0, 2));
        index.insert(ProofId("p3".to_string()), make_category(1, 0));

        let nonempty: HashSet<_> = index.nonempty_subtypes().collect();
        assert_eq!(nonempty.len(), 3);
        assert!(nonempty.contains(&(0, 0)));
        assert!(nonempty.contains(&(0, 2)));
        assert!(nonempty.contains(&(1, 0)));
    }

    #[test]
    fn test_rebuild() {
        let mut index = CategoryIndex::new();

        index.insert(ProofId("old".to_string()), make_category(7, 5));
        assert_eq!(index.category_count(7), 1);

        // Rebuild with new entries
        let new_entries = vec![
            (ProofId("new1".to_string()), make_category(0, 0)),
            (ProofId("new2".to_string()), make_category(0, 1)),
            (ProofId("new3".to_string()), make_category(1, 0)),
        ];
        index.rebuild(new_entries);

        // Old entry should be gone
        assert_eq!(index.category_count(7), 0);

        // New entries should be present
        assert_eq!(index.category_count(0), 2);
        assert_eq!(index.category_count(1), 1);
        assert_eq!(index.total_count(), 3);
    }

    #[test]
    fn test_clear() {
        let mut index = CategoryIndex::new();

        index.insert(ProofId("p1".to_string()), make_category(0, 0));
        index.insert(ProofId("p2".to_string()), make_category(1, 1));

        assert_eq!(index.total_count(), 2);

        index.clear();

        assert_eq!(index.total_count(), 0);
        assert_eq!(index.category_count(0), 0);
        assert_eq!(index.category_count(1), 0);
    }

    #[test]
    fn test_duplicate_insert() {
        let mut index = CategoryIndex::new();

        let id = ProofId("proof_1".to_string());
        let cat = make_category(0, 0);

        // Insert same ID twice
        index.insert(id.clone(), cat);
        index.insert(id.clone(), cat);

        // HashSet deduplicates, so count should still be 1
        assert_eq!(index.category_count(0), 1);
        assert_eq!(index.subtype_count(0, 0), 1);
    }

    #[test]
    fn test_total_count() {
        let mut index = CategoryIndex::new();

        // Insert entries across multiple categories
        for i in 0..5 {
            for j in 0..3 {
                index.insert(ProofId(format!("p_{}_{}", i, j)), make_category(i, j));
            }
        }

        assert_eq!(index.total_count(), 15);
    }
}
