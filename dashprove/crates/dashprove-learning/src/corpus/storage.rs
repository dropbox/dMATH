//! ProofCorpus - Database of successful proofs

use super::category::CategoryIndex;
use super::history;
use super::search;
use super::types::{ProofEntry, ProofHistory, ProofId};
use crate::counterexamples::TimePeriod;
use crate::embedder::PropertyCategory;
use crate::similarity::{self, SimilarProof};
use crate::LearnableResult;
use chrono::{DateTime, Utc};
use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::Property;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Database of successful proofs
///
/// Maintains both a primary index by ProofId and a secondary index by
/// PropertyCategory for fast filtered searches.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProofCorpus {
    /// All stored proofs by ID
    pub(super) proofs: HashMap<ProofId, ProofEntry>,
    /// Secondary index by property category (rebuilt on load if missing)
    #[serde(default, skip_serializing_if = "is_index_empty")]
    pub(super) category_index: CategoryIndex,
}

fn is_index_empty(index: &CategoryIndex) -> bool {
    index.total_count() == 0
}

impl ProofCorpus {
    /// Create a new empty corpus
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a learnable result as a proof entry
    pub fn insert(&mut self, result: &LearnableResult) -> ProofId {
        let name = result.property.name();
        let id = ProofId::generate(&name);
        let features = similarity::extract_features(&result.property);

        // Index by category before storing
        let category = PropertyCategory::from_property_type(&features.property_type);
        self.category_index.insert(id.clone(), category);

        let entry = ProofEntry {
            id: id.clone(),
            property: result.property.clone(),
            backend: result.backend,
            tactics: result.tactics.clone(),
            time_taken: result.time_taken,
            proof_output: result.proof_output.clone(),
            features,
            recorded_at: Utc::now(),
            embedding: None,
        };

        self.proofs.insert(id.clone(), entry);
        id
    }

    /// Insert a learnable result with a pre-computed embedding
    pub fn insert_with_embedding(
        &mut self,
        result: &LearnableResult,
        embedding: crate::embedder::Embedding,
    ) -> ProofId {
        let name = result.property.name();
        let id = ProofId::generate(&name);
        let features = similarity::extract_features(&result.property);

        // Index by category before storing
        let category = PropertyCategory::from_property_type(&features.property_type);
        self.category_index.insert(id.clone(), category);

        let entry = ProofEntry {
            id: id.clone(),
            property: result.property.clone(),
            backend: result.backend,
            tactics: result.tactics.clone(),
            time_taken: result.time_taken,
            proof_output: result.proof_output.clone(),
            features,
            recorded_at: Utc::now(),
            embedding: Some(embedding),
        };

        self.proofs.insert(id.clone(), entry);
        id
    }

    /// Get a proof by ID
    pub fn get(&self, id: &ProofId) -> Option<&ProofEntry> {
        self.proofs.get(id)
    }

    /// Number of proofs in corpus
    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    /// Check if corpus is empty
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }

    /// Find proofs similar to the given property
    ///
    /// Returns up to `k` most similar proofs, sorted by similarity score (descending).
    pub fn find_similar(&self, property: &Property, k: usize) -> Vec<SimilarProof> {
        search::find_similar_features(&self.proofs, property, k)
    }

    /// Find proofs similar to the given property using vector embeddings
    ///
    /// This method uses the embedding-based similarity search, which can be more
    /// accurate for semantic similarity than the feature-based `find_similar()`.
    ///
    /// Only proofs with embeddings are considered. Use `compute_embeddings()`
    /// to populate embeddings for the corpus.
    ///
    /// Returns up to `k` most similar proofs, sorted by similarity score (descending).
    pub fn find_similar_embedding(
        &self,
        query_embedding: &crate::embedder::Embedding,
        k: usize,
    ) -> Vec<SimilarProof> {
        search::find_similar_embedding(&self.proofs, query_embedding, k)
    }

    /// Compute and store embeddings for all proofs in the corpus
    ///
    /// Uses the provided embedder to generate embeddings for any proofs
    /// that don't already have one. Returns the number of embeddings computed.
    pub fn compute_embeddings(
        &mut self,
        embedder: &mut crate::embedder::PropertyEmbedder,
    ) -> usize {
        let mut count = 0;
        for entry in self.proofs.values_mut() {
            if entry.embedding.is_none() {
                let embedding = embedder.embed_features(&entry.features);
                entry.embedding = Some(embedding);
                count += 1;
            }
        }
        count
    }

    /// Count how many proofs have embeddings computed
    pub fn embedding_count(&self) -> usize {
        self.proofs
            .values()
            .filter(|e| e.embedding.is_some())
            .count()
    }

    /// Search proofs by text keywords
    ///
    /// Searches the corpus for proofs matching any of the query terms.
    /// Returns up to `k` proofs, sorted by match score (descending).
    /// The match score is based on keyword overlap.
    pub fn search_by_keywords(&self, query: &str, k: usize) -> Vec<SimilarProof> {
        search::search_by_keywords(&self.proofs, query, k)
    }

    /// Get all proofs for a specific backend
    pub fn by_backend(&self, backend: BackendId) -> Vec<&ProofEntry> {
        self.proofs
            .values()
            .filter(|e| e.backend == backend)
            .collect()
    }

    // ========================================================================
    // Category-filtered search methods
    // ========================================================================

    /// Get all proofs in a specific category (coarse)
    ///
    /// Categories are defined in `PropertyCategory`:
    /// - 0: Theorem Proving (theorem, contract, invariant, refinement)
    /// - 1: Model Checking (temporal, probabilistic)
    /// - 2: Neural Networks (neural_robustness, neural_reachability, adversarial_robustness)
    /// - 3: Security (security_protocol, platform_api)
    /// - 4: Memory Safety (memory_safety, undefined_behavior, data_race, memory_leak)
    /// - 5: Testing (fuzzing, property_based, mutation_testing)
    /// - 6: Static Analysis (lint, api_compatibility, security_vulnerability, etc.)
    /// - 7: AI/ML (model_optimization, model_compression, data_quality, etc.)
    pub fn by_category(&self, category: usize) -> Vec<&ProofEntry> {
        self.category_index
            .by_category(category)
            .filter_map(|id| self.proofs.get(id))
            .collect()
    }

    /// Get all proofs with a specific (category, subtype) pair
    ///
    /// For example, to get all DataRace properties (category 4, subtype 2):
    /// ```ignore
    /// let data_race_proofs = corpus.by_subtype(4, 2);
    /// ```
    pub fn by_subtype(&self, category: usize, subtype: usize) -> Vec<&ProofEntry> {
        self.category_index
            .by_subtype(category, subtype)
            .filter_map(|id| self.proofs.get(id))
            .collect()
    }

    /// Count proofs in a specific category
    pub fn category_count(&self, category: usize) -> usize {
        self.category_index.category_count(category)
    }

    /// Count proofs with a specific (category, subtype) pair
    pub fn subtype_count(&self, category: usize, subtype: usize) -> usize {
        self.category_index.subtype_count(category, subtype)
    }

    /// Find proofs similar to the given property, filtered by category
    ///
    /// This is more efficient than `find_similar()` when querying a specific
    /// domain, as it only searches proofs in the matching category.
    ///
    /// # Performance
    ///
    /// For corpus with n total proofs and m proofs in the category:
    /// - Time: O(m log k) instead of O(n log k)
    /// - When m << n, this provides significant speedup
    pub fn find_similar_in_category(
        &self,
        property: &Property,
        category: usize,
        k: usize,
    ) -> Vec<SimilarProof> {
        if k == 0 {
            return vec![];
        }

        // Build a filtered HashMap of only entries in this category
        let filtered: HashMap<ProofId, ProofEntry> = self
            .category_index
            .by_category(category)
            .filter_map(|id| self.proofs.get(id).map(|e| (id.clone(), e.clone())))
            .collect();

        search::find_similar_features(&filtered, property, k)
    }

    /// Find proofs similar to the given property, filtered by (category, subtype)
    ///
    /// This is more efficient than `find_similar_in_category()` when querying
    /// a specific property subtype within a domain.
    pub fn find_similar_in_subtype(
        &self,
        property: &Property,
        category: usize,
        subtype: usize,
        k: usize,
    ) -> Vec<SimilarProof> {
        if k == 0 {
            return vec![];
        }

        // Build a filtered HashMap of only entries with this (category, subtype)
        let filtered: HashMap<ProofId, ProofEntry> = self
            .category_index
            .by_subtype(category, subtype)
            .filter_map(|id| self.proofs.get(id).map(|e| (id.clone(), e.clone())))
            .collect();

        search::find_similar_features(&filtered, property, k)
    }

    /// Search proofs by keywords, filtered by category
    pub fn search_by_keywords_in_category(
        &self,
        query: &str,
        category: usize,
        k: usize,
    ) -> Vec<SimilarProof> {
        if k == 0 || query.is_empty() {
            return vec![];
        }

        let filtered: HashMap<ProofId, ProofEntry> = self
            .category_index
            .by_category(category)
            .filter_map(|id| self.proofs.get(id).map(|e| (id.clone(), e.clone())))
            .collect();

        search::search_by_keywords(&filtered, query, k)
    }

    /// Rebuild the category index from all stored proofs
    ///
    /// This is useful after loading a corpus that was saved without an index
    /// (e.g., from an older version) or if the index becomes inconsistent.
    pub fn rebuild_category_index(&mut self) {
        let entries = self
            .proofs
            .iter()
            .map(|(id, entry)| {
                let category = PropertyCategory::from_property_type(&entry.features.property_type);
                (id.clone(), category)
            })
            .collect::<Vec<_>>();

        self.category_index.rebuild(entries);
    }

    /// Get a reference to the category index for advanced queries
    pub fn category_index(&self) -> &CategoryIndex {
        &self.category_index
    }

    /// Get all categories that have at least one proof
    pub fn nonempty_categories(&self) -> Vec<usize> {
        self.category_index.nonempty_categories().collect()
    }

    /// Get all (category, subtype) pairs that have at least one proof
    pub fn nonempty_subtypes(&self) -> Vec<(usize, usize)> {
        self.category_index.nonempty_subtypes().collect()
    }

    /// Get all proof IDs
    pub fn ids(&self) -> impl Iterator<Item = &ProofId> {
        self.proofs.keys()
    }

    /// Get all proof entries (for history/stats)
    pub fn entries(&self) -> impl Iterator<Item = &ProofEntry> {
        self.proofs.values()
    }

    /// Get proofs recorded within a time range (inclusive)
    pub fn in_time_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&ProofEntry> {
        self.proofs
            .values()
            .filter(|e| e.recorded_at >= start && e.recorded_at <= end)
            .collect()
    }

    /// Get corpus history statistics
    ///
    /// Groups proofs by time periods (day, week, or month) and returns counts
    /// per period and per backend.
    pub fn history(&self, period: TimePeriod) -> ProofHistory {
        history::compute_history(&self.proofs, period)
    }

    /// Get corpus history statistics within an optional time range
    ///
    /// If `from` is Some, only include proofs recorded on or after that date.
    /// If `to` is Some, only include proofs recorded on or before that date.
    pub fn history_in_range(
        &self,
        period: TimePeriod,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> ProofHistory {
        history::compute_history_in_range(&self.proofs, period, from, to)
    }

    /// Persist the corpus to a JSON file (atomic write)
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load a corpus from disk
    ///
    /// If the loaded corpus has an empty category index (e.g., from an older
    /// version without indexing), the index is automatically rebuilt.
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        let mut corpus: Self = crate::io::read_json(path)?;

        // Rebuild index if empty (backwards compatibility)
        if corpus.category_index.total_count() == 0 && !corpus.proofs.is_empty() {
            corpus.rebuild_category_index();
        }

        Ok(corpus)
    }

    /// Load a corpus if it exists, otherwise return empty
    ///
    /// If the loaded corpus has an empty category index, it is automatically rebuilt.
    pub fn load_or_default<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Self::load_from_file(path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }
}
