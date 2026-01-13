//! CounterexampleCorpus - storage and retrieval of counterexamples

use super::history::{CorpusHistory, PeriodStats, TimePeriod};
use super::similarity::{compute_feature_similarity, compute_keyword_score};
use super::types::{
    ClusterPattern, CounterexampleEntry, CounterexampleFeatures, CounterexampleId,
    SimilarCounterexample,
};
use chrono::{DateTime, Utc};
use dashprove_backends::traits::{BackendId, CounterexampleClusters, StructuredCounterexample};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Database of counterexamples for learning and pattern recognition
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CounterexampleCorpus {
    /// All stored counterexamples by ID
    counterexamples: HashMap<CounterexampleId, CounterexampleEntry>,
    /// Stored cluster patterns
    cluster_patterns: Vec<ClusterPattern>,
}

impl CounterexampleCorpus {
    /// Create a new empty corpus
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a counterexample into the corpus
    pub fn insert(
        &mut self,
        property_name: &str,
        backend: BackendId,
        counterexample: StructuredCounterexample,
        cluster_label: Option<String>,
    ) -> CounterexampleId {
        let id = CounterexampleId::generate(property_name);
        let features = CounterexampleFeatures::extract(&counterexample);

        let entry = CounterexampleEntry {
            id: id.clone(),
            property_name: property_name.to_string(),
            backend,
            counterexample,
            features,
            cluster_label,
            recorded_at: Utc::now(),
        };

        self.counterexamples.insert(id.clone(), entry);
        id
    }

    /// Record cluster patterns for future classification
    pub fn record_clusters(&mut self, clusters: &CounterexampleClusters) {
        for cluster in &clusters.clusters {
            let pattern = ClusterPattern {
                label: cluster.label.clone(),
                representative_features: CounterexampleFeatures::extract(&cluster.representative),
                original_count: cluster.size(),
                similarity_threshold: cluster.similarity_threshold,
            };
            self.cluster_patterns.push(pattern);
        }
    }

    /// Get a counterexample by ID
    pub fn get(&self, id: &CounterexampleId) -> Option<&CounterexampleEntry> {
        self.counterexamples.get(id)
    }

    /// Number of counterexamples in corpus
    pub fn len(&self) -> usize {
        self.counterexamples.len()
    }

    /// Check if corpus is empty
    pub fn is_empty(&self) -> bool {
        self.counterexamples.is_empty()
    }

    /// Number of stored cluster patterns
    pub fn pattern_count(&self) -> usize {
        self.cluster_patterns.len()
    }

    /// Find counterexamples similar to the given one
    ///
    /// Returns up to `k` most similar counterexamples, sorted by similarity score (descending).
    pub fn find_similar(
        &self,
        cx: &StructuredCounterexample,
        k: usize,
    ) -> Vec<SimilarCounterexample> {
        if self.counterexamples.is_empty() {
            return vec![];
        }

        let query_features = CounterexampleFeatures::extract(cx);

        // Compute similarity to all stored counterexamples
        let mut similarities: Vec<_> = self
            .counterexamples
            .values()
            .map(|entry| {
                let sim = compute_feature_similarity(&query_features, &entry.features);
                (entry, sim)
            })
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        similarities
            .into_iter()
            .take(k)
            .map(|(entry, sim)| SimilarCounterexample {
                id: entry.id.clone(),
                property_name: entry.property_name.clone(),
                backend: entry.backend,
                cluster_label: entry.cluster_label.clone(),
                similarity: sim,
            })
            .collect()
    }

    /// Classify a counterexample against stored cluster patterns
    ///
    /// Returns the best matching cluster label and its similarity score,
    /// or None if no pattern matches above threshold.
    pub fn classify(&self, cx: &StructuredCounterexample) -> Option<(String, f64)> {
        if self.cluster_patterns.is_empty() {
            return None;
        }

        let query_features = CounterexampleFeatures::extract(cx);

        let mut best_match: Option<(String, f64)> = None;

        for pattern in &self.cluster_patterns {
            let sim = compute_feature_similarity(&query_features, &pattern.representative_features);
            if sim >= pattern.similarity_threshold
                && best_match
                    .as_ref()
                    .is_none_or(|(_, best_sim)| sim > *best_sim)
            {
                best_match = Some((pattern.label.clone(), sim));
            }
        }

        best_match
    }

    /// Search counterexamples by text keywords
    ///
    /// Searches the corpus for counterexamples matching any of the query terms.
    /// Returns up to `k` counterexamples, sorted by match score (descending).
    pub fn search_by_keywords(&self, query: &str, k: usize) -> Vec<SimilarCounterexample> {
        if self.counterexamples.is_empty() || query.is_empty() {
            return vec![];
        }

        // Tokenize query into lowercase terms
        let query_terms: Vec<String> = query
            .split(|c: char| c.is_whitespace() || c == '_' || c == '-')
            .filter(|s| s.len() > 1)
            .map(|s| s.to_lowercase())
            .collect();

        if query_terms.is_empty() {
            return vec![];
        }

        // Score each counterexample by keyword matches
        let mut scored: Vec<_> = self
            .counterexamples
            .values()
            .filter_map(|entry| {
                let score = compute_keyword_score(&query_terms, &entry.features.keywords);
                if score > 0.0 {
                    Some((entry, score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        scored
            .into_iter()
            .take(k)
            .map(|(entry, score)| SimilarCounterexample {
                id: entry.id.clone(),
                property_name: entry.property_name.clone(),
                backend: entry.backend,
                cluster_label: entry.cluster_label.clone(),
                similarity: score,
            })
            .collect()
    }

    /// Get all counterexamples for a specific backend
    pub fn by_backend(&self, backend: BackendId) -> Vec<&CounterexampleEntry> {
        self.counterexamples
            .values()
            .filter(|e| e.backend == backend)
            .collect()
    }

    /// Get all counterexample IDs
    pub fn all_ids(&self) -> Vec<CounterexampleId> {
        self.counterexamples.keys().cloned().collect()
    }

    /// Get all stored cluster patterns
    pub fn get_patterns(&self) -> &[ClusterPattern] {
        &self.cluster_patterns
    }

    /// Get all counterexample entries (for history/stats)
    pub fn all_entries(&self) -> impl Iterator<Item = &CounterexampleEntry> {
        self.counterexamples.values()
    }

    /// Get counterexamples in a time range (inclusive)
    pub fn in_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&CounterexampleEntry> {
        self.counterexamples
            .values()
            .filter(|e| e.recorded_at >= start && e.recorded_at <= end)
            .collect()
    }

    /// Get corpus history statistics
    ///
    /// Groups counterexamples by time periods (day, week, or month) and returns
    /// counts per period and per backend.
    pub fn history(&self, period: TimePeriod) -> CorpusHistory {
        if self.counterexamples.is_empty() {
            return CorpusHistory::default();
        }

        // Find time range
        let (min_time, max_time) = self.counterexamples.values().fold(
            (DateTime::<Utc>::MAX_UTC, DateTime::<Utc>::MIN_UTC),
            |acc, e| (acc.0.min(e.recorded_at), acc.1.max(e.recorded_at)),
        );

        // Group by period
        let mut periods: HashMap<String, PeriodStats> = HashMap::new();
        let mut by_backend: HashMap<BackendId, usize> = HashMap::new();
        let mut by_property: HashMap<String, usize> = HashMap::new();
        let mut by_cluster: HashMap<String, usize> = HashMap::new();

        for entry in self.counterexamples.values() {
            let period_key = period.key_for(entry.recorded_at);
            let stats = periods
                .entry(period_key.clone())
                .or_insert_with(|| PeriodStats {
                    period: period_key,
                    start: period.start_for(entry.recorded_at),
                    count: 0,
                    by_backend: HashMap::new(),
                });
            stats.count += 1;
            *stats.by_backend.entry(entry.backend).or_insert(0) += 1;
            *by_backend.entry(entry.backend).or_insert(0) += 1;
            *by_property.entry(entry.property_name.clone()).or_insert(0) += 1;
            if let Some(cluster) = &entry.cluster_label {
                *by_cluster.entry(cluster.clone()).or_insert(0) += 1;
            }
        }

        // Sort periods chronologically
        let mut period_list: Vec<PeriodStats> = periods.into_values().collect();
        period_list.sort_by_key(|p| p.start);

        // Calculate cumulative totals
        let mut cumulative = 0;
        let cumulative_counts: Vec<usize> = period_list
            .iter()
            .map(|p| {
                cumulative += p.count;
                cumulative
            })
            .collect();

        CorpusHistory {
            total_count: self.counterexamples.len(),
            first_recorded: Some(min_time),
            last_recorded: Some(max_time),
            period_type: period,
            periods: period_list,
            cumulative_counts,
            by_backend,
            by_property,
            by_cluster,
        }
    }

    /// Get corpus history statistics within an optional time range
    ///
    /// If `from` is Some, only include counterexamples recorded on or after that date.
    /// If `to` is Some, only include counterexamples recorded on or before that date.
    pub fn history_in_range(
        &self,
        period: TimePeriod,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    ) -> CorpusHistory {
        // Filter counterexamples by date range
        let filtered: Vec<&CounterexampleEntry> = self
            .counterexamples
            .values()
            .filter(|e| {
                let after_start = from.is_none_or(|f| e.recorded_at >= f);
                let before_end = to.is_none_or(|t| e.recorded_at <= t);
                after_start && before_end
            })
            .collect();

        if filtered.is_empty() {
            return CorpusHistory::default();
        }

        let (min_time, max_time) = filtered.iter().fold(
            (DateTime::<Utc>::MAX_UTC, DateTime::<Utc>::MIN_UTC),
            |acc, e| (acc.0.min(e.recorded_at), acc.1.max(e.recorded_at)),
        );

        let mut periods: HashMap<String, PeriodStats> = HashMap::new();
        let mut by_backend: HashMap<BackendId, usize> = HashMap::new();
        let mut by_property: HashMap<String, usize> = HashMap::new();
        let mut by_cluster: HashMap<String, usize> = HashMap::new();

        for entry in &filtered {
            let period_key = period.key_for(entry.recorded_at);
            let stats = periods
                .entry(period_key.clone())
                .or_insert_with(|| PeriodStats {
                    period: period_key,
                    start: period.start_for(entry.recorded_at),
                    count: 0,
                    by_backend: HashMap::new(),
                });
            stats.count += 1;
            *stats.by_backend.entry(entry.backend).or_insert(0) += 1;
            *by_backend.entry(entry.backend).or_insert(0) += 1;
            *by_property.entry(entry.property_name.clone()).or_insert(0) += 1;
            if let Some(cluster) = &entry.cluster_label {
                *by_cluster.entry(cluster.clone()).or_insert(0) += 1;
            }
        }

        let mut period_list: Vec<PeriodStats> = periods.into_values().collect();
        period_list.sort_by_key(|p| p.start);

        let mut cumulative = 0;
        let cumulative_counts: Vec<usize> = period_list
            .iter()
            .map(|p| {
                cumulative += p.count;
                cumulative
            })
            .collect();

        CorpusHistory {
            total_count: filtered.len(),
            first_recorded: Some(min_time),
            last_recorded: Some(max_time),
            period_type: period,
            periods: period_list,
            cumulative_counts,
            by_backend,
            by_property,
            by_cluster,
        }
    }

    /// Save corpus to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load corpus from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        let corpus = serde_json::from_str(&json)?;
        Ok(corpus)
    }

    /// Load from file or return default (empty) corpus if file doesn't exist
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        match Self::load_from_file(&path) {
            Ok(corpus) => Ok(corpus),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }
}
