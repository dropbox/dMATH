//! Search functionality for proof corpus
//!
//! This module provides efficient similarity search over proof corpora.
//!
//! # Performance
//!
//! For finding top-k similar proofs from n entries:
//! - Uses a min-heap to maintain only k candidates during iteration
//! - Time complexity: O(n log k) instead of O(n log n) for full sort
//! - Space complexity: O(k) for the heap

use super::types::{ProofEntry, ProofId};
use crate::ordered_float::OrderedF64;
use crate::similarity::{self, SimilarProof};
use dashprove_usl::ast::Property;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Wrapper for min-heap ordering (lowest similarity at top for efficient top-k)
///
/// BinaryHeap is a max-heap by default, so we reverse the ordering to get a min-heap.
/// This allows us to efficiently maintain only the k highest-scoring items:
/// - If heap.len() < k, push the item
/// - If heap.len() == k and new item > heap.peek(), pop minimum and push new item
#[derive(Debug)]
struct ScoredEntry<'a> {
    entry: &'a ProofEntry,
    score: OrderedF64,
}

impl PartialEq for ScoredEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.entry.id == other.entry.id
    }
}

impl Eq for ScoredEntry<'_> {}

impl PartialOrd for ScoredEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap: lower scores should come first (be "greater").
        // Use ID tie-breaker for deterministic ordering with equal scores.
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.entry.id.0.cmp(&other.entry.id.0))
    }
}

/// Find top-k elements using a min-heap
///
/// For n items and k results, this is O(n log k) instead of O(n log n) for full sort.
/// When k << n, this provides significant speedup.
fn top_k_by_score<'a>(
    entries: impl Iterator<Item = (&'a ProofEntry, f64)>,
    k: usize,
) -> Vec<(&'a ProofEntry, f64)> {
    if k == 0 {
        return vec![];
    }

    let mut heap: BinaryHeap<ScoredEntry<'a>> = BinaryHeap::with_capacity(k + 1);

    for (entry, score) in entries {
        let ordered_score = OrderedF64(score);

        if heap.len() < k {
            heap.push(ScoredEntry {
                entry,
                score: ordered_score,
            });
        } else if let Some(min) = heap.peek() {
            // heap is a min-heap, so peek() gives the smallest score
            if ordered_score > min.score
                || (ordered_score == min.score && entry.id.0 < min.entry.id.0)
            {
                heap.pop();
                heap.push(ScoredEntry {
                    entry,
                    score: ordered_score,
                });
            }
        }
    }

    // Extract results and sort by descending score
    let mut results: Vec<_> = heap
        .into_iter()
        .map(|se| (se.entry, se.score.into_inner()))
        .collect();

    results.sort_by(|a, b| {
        let a_score = OrderedF64(a.1);
        let b_score = OrderedF64(b.1);
        b_score.cmp(&a_score).then_with(|| a.0.id.0.cmp(&b.0.id.0))
    });
    results
}

/// Find proofs similar to the given property using feature-based similarity
///
/// Uses a min-heap to efficiently find top-k results in O(n log k) time.
pub fn find_similar_features(
    proofs: &HashMap<ProofId, ProofEntry>,
    property: &Property,
    k: usize,
) -> Vec<SimilarProof> {
    if proofs.is_empty() || k == 0 {
        return vec![];
    }

    let query_features = similarity::extract_features(property);

    // Use min-heap for efficient top-k selection
    let entries = proofs.values().map(|entry| {
        let sim = similarity::compute_similarity(&query_features, &entry.features);
        (entry, sim)
    });

    top_k_by_score(entries, k)
        .into_iter()
        .map(|(entry, sim)| SimilarProof {
            id: entry.id.clone(),
            property: entry.property.clone(),
            backend: entry.backend,
            tactics: entry.tactics.clone(),
            similarity: sim,
        })
        .collect()
}

/// Find proofs similar using vector embeddings
///
/// Uses a min-heap to efficiently find top-k results in O(n log k) time.
pub fn find_similar_embedding(
    proofs: &HashMap<ProofId, ProofEntry>,
    query_embedding: &crate::embedder::Embedding,
    k: usize,
) -> Vec<SimilarProof> {
    if proofs.is_empty() || k == 0 {
        return vec![];
    }

    // Use min-heap for efficient top-k selection
    let entries = proofs.values().filter_map(|entry| {
        entry.embedding.as_ref().map(|emb| {
            let sim = query_embedding.normalized_similarity(emb);
            (entry, sim)
        })
    });

    top_k_by_score(entries, k)
        .into_iter()
        .map(|(entry, sim)| SimilarProof {
            id: entry.id.clone(),
            property: entry.property.clone(),
            backend: entry.backend,
            tactics: entry.tactics.clone(),
            similarity: sim,
        })
        .collect()
}

/// Search proofs by text keywords
///
/// Uses a min-heap to efficiently find top-k results in O(n log k) time.
pub fn search_by_keywords(
    proofs: &HashMap<ProofId, ProofEntry>,
    query: &str,
    k: usize,
) -> Vec<SimilarProof> {
    if proofs.is_empty() || query.is_empty() || k == 0 {
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

    // Use min-heap for efficient top-k selection
    let entries = proofs.values().filter_map(|entry| {
        let score = compute_keyword_score(&query_terms, &entry.features.keywords);
        if score > 0.0 {
            Some((entry, score))
        } else {
            None
        }
    });

    top_k_by_score(entries, k)
        .into_iter()
        .map(|(entry, score)| SimilarProof {
            id: entry.id.clone(),
            property: entry.property.clone(),
            backend: entry.backend,
            tactics: entry.tactics.clone(),
            similarity: score,
        })
        .collect()
}

/// Compute keyword match score between query terms and proof keywords
///
/// Returns a score between 0.0 and 1.0 based on:
/// - Exact matches (weighted higher)
/// - Prefix/substring matches
pub fn compute_keyword_score(query_terms: &[String], proof_keywords: &[String]) -> f64 {
    if query_terms.is_empty() || proof_keywords.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0;
    let mut matches = 0;

    for query in query_terms {
        let mut best_match: f64 = 0.0;
        for keyword in proof_keywords {
            if keyword == query {
                // Exact match
                best_match = 1.0;
                break;
            } else if keyword.starts_with(query) || query.starts_with(keyword) {
                // Prefix match
                best_match = best_match.max(0.7);
            } else if keyword.contains(query) || query.contains(keyword) {
                // Substring match
                best_match = best_match.max(0.4);
            }
        }
        if best_match > 0.0 {
            matches += 1;
            total_score += best_match;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Normalize by number of query terms and add bonus for multiple matches
    let base_score = total_score / query_terms.len() as f64;
    let coverage_bonus = (matches as f64 / query_terms.len() as f64) * 0.3;

    (base_score + coverage_bonus).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::PropertyFeatures;
    use chrono::Utc;
    use dashprove_backends::traits::BackendId;
    use dashprove_usl::ast::{Expr, Invariant, Property};
    use std::time::Duration;

    fn make_test_entry(id: &str, name: &str, score_hint: usize) -> ProofEntry {
        ProofEntry {
            id: ProofId(id.to_string()),
            property: Property::Invariant(Invariant {
                name: name.to_string(),
                body: Expr::Bool(true),
            }),
            backend: BackendId::Lean4,
            tactics: vec!["simp".to_string()],
            time_taken: Duration::from_millis(100),
            proof_output: None,
            embedding: None,
            recorded_at: Utc::now(),
            features: PropertyFeatures {
                property_type: "invariant".to_string(),
                depth: score_hint, // Use depth as a proxy for deterministic ordering
                quantifier_depth: 0,
                implication_count: 0,
                arithmetic_ops: 0,
                function_calls: 0,
                variable_count: 0,
                has_temporal: false,
                type_refs: vec![],
                keywords: vec![name.to_lowercase()],
            },
        }
    }

    #[test]
    fn test_top_k_by_score_basic() {
        let entries: Vec<ProofEntry> = (0..10)
            .map(|i| make_test_entry(&format!("id_{}", i), &format!("prop_{}", i), i))
            .collect();

        let map: HashMap<ProofId, ProofEntry> =
            entries.into_iter().map(|e| (e.id.clone(), e)).collect();

        // Create scored iterator with known scores
        let scored = map.values().map(|e| {
            let score = e.features.depth as f64 / 10.0; // 0.0, 0.1, ..., 0.9
            (e, score)
        });

        let top3 = top_k_by_score(scored, 3);

        // Should get top 3 highest scores (0.9, 0.8, 0.7 -> depths 9, 8, 7)
        assert_eq!(top3.len(), 3);

        // Results should be sorted descending
        assert!(top3[0].1 >= top3[1].1);
        assert!(top3[1].1 >= top3[2].1);

        // Should have the highest scoring items
        assert!((top3[0].1 - 0.9).abs() < 0.001);
        assert!((top3[1].1 - 0.8).abs() < 0.001);
        assert!((top3[2].1 - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_top_k_by_score_k_larger_than_n() {
        let entries: Vec<ProofEntry> = (0..3)
            .map(|i| make_test_entry(&format!("id_{}", i), &format!("prop_{}", i), i))
            .collect();

        let map: HashMap<ProofId, ProofEntry> =
            entries.into_iter().map(|e| (e.id.clone(), e)).collect();

        let scored = map.values().map(|e| {
            let score = e.features.depth as f64;
            (e, score)
        });

        // k=10 but only 3 entries
        let top10 = top_k_by_score(scored, 10);
        assert_eq!(top10.len(), 3);
    }

    #[test]
    fn test_top_k_by_score_k_zero() {
        let entries: Vec<ProofEntry> = (0..5)
            .map(|i| make_test_entry(&format!("id_{}", i), &format!("prop_{}", i), i))
            .collect();

        let map: HashMap<ProofId, ProofEntry> =
            entries.into_iter().map(|e| (e.id.clone(), e)).collect();

        let scored = map.values().map(|e| (e, 1.0));
        let top0 = top_k_by_score(scored, 0);
        assert!(top0.is_empty());
    }

    #[test]
    fn test_top_k_by_score_empty_input() {
        let map: HashMap<ProofId, ProofEntry> = HashMap::new();
        let scored = map.values().map(|e| (e, 1.0));
        let result = top_k_by_score(scored, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_by_score_equal_scores() {
        let entries: Vec<ProofEntry> = (0..5)
            .map(|i| make_test_entry(&format!("id_{}", i), &format!("prop_{}", i), 5))
            .collect();

        let map: HashMap<ProofId, ProofEntry> =
            entries.into_iter().map(|e| (e.id.clone(), e)).collect();

        let scored = map.values().map(|e| (e, 0.5)); // All same score
        let top3 = top_k_by_score(scored, 3);

        // Should still return 3 items
        assert_eq!(top3.len(), 3);
        // All should have same score
        for (_, score) in &top3 {
            assert!((score - 0.5).abs() < 0.001);
        }
        // Deterministic ordering picks lowest lexicographic IDs first
        let ids: Vec<_> = top3.iter().map(|(entry, _)| entry.id.0.as_str()).collect();
        assert_eq!(ids, vec!["id_0", "id_1", "id_2"]);
    }

    #[test]
    fn test_scored_entry_ordering() {
        // Test that ScoredEntry implements min-heap ordering correctly
        let entry1 = make_test_entry("1", "prop1", 1);
        let entry2 = make_test_entry("2", "prop2", 2);

        let se1 = ScoredEntry {
            entry: &entry1,
            score: OrderedF64(0.3),
        };
        let se2 = ScoredEntry {
            entry: &entry2,
            score: OrderedF64(0.7),
        };

        // In min-heap ordering, lower scores should be "greater"
        assert!(
            se1 > se2,
            "Lower score should be 'greater' in min-heap ordering"
        );
    }

    #[test]
    fn test_find_similar_features_empty() {
        let proofs: HashMap<ProofId, ProofEntry> = HashMap::new();
        let query = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });

        let results = find_similar_features(&proofs, &query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_similar_features_k_zero() {
        let entry = make_test_entry("1", "test", 1);
        let mut proofs: HashMap<ProofId, ProofEntry> = HashMap::new();
        proofs.insert(entry.id.clone(), entry);

        let query = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });

        let results = find_similar_features(&proofs, &query, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_by_keywords_k_zero() {
        let mut entry = make_test_entry("1", "buffer_overflow", 1);
        entry.features.keywords = vec!["buffer".to_string(), "overflow".to_string()];
        let mut proofs: HashMap<ProofId, ProofEntry> = HashMap::new();
        proofs.insert(entry.id.clone(), entry);

        let results = search_by_keywords(&proofs, "overflow", 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_compute_keyword_score_empty() {
        assert_eq!(compute_keyword_score(&[], &["test".to_string()]), 0.0);
        assert_eq!(compute_keyword_score(&["test".to_string()], &[]), 0.0);
    }

    #[test]
    fn test_compute_keyword_score_exact_match() {
        let query = vec!["overflow".to_string()];
        let keywords = vec!["overflow".to_string(), "buffer".to_string()];
        let score = compute_keyword_score(&query, &keywords);
        // Exact match (1.0) + coverage bonus (1.0 * 0.3) = 1.3, capped at 1.0
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_keyword_score_prefix_match() {
        let query = vec!["over".to_string()];
        let keywords = vec!["overflow".to_string()];
        let score = compute_keyword_score(&query, &keywords);
        // Prefix match (0.7) + coverage bonus (1.0 * 0.3) = 1.0
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_keyword_score_substring_match() {
        let query = vec!["flow".to_string()];
        let keywords = vec!["overflow".to_string()];
        let score = compute_keyword_score(&query, &keywords);
        // Substring match (0.4) + coverage bonus (1.0 * 0.3) = 0.7
        assert!((score - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_compute_keyword_score_no_match() {
        let query = vec!["xyz".to_string()];
        let keywords = vec!["abc".to_string()];
        let score = compute_keyword_score(&query, &keywords);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_top_k_by_score_replaces_lower_id_on_tie() {
        let entries = vec![
            make_test_entry("id_c", "prop_c", 0),
            make_test_entry("id_b", "prop_b", 0),
            make_test_entry("id_a", "prop_a", 0),
        ];

        let map: HashMap<ProofId, ProofEntry> =
            entries.into_iter().map(|e| (e.id.clone(), e)).collect();

        let scored = map.values().map(|e| (e, 0.8));
        let top2 = top_k_by_score(scored, 2);

        let ids: Vec<_> = top2.iter().map(|(entry, _)| entry.id.0.as_str()).collect();

        assert_eq!(ids, vec!["id_a", "id_b"]);
    }

    #[test]
    fn test_top_k_by_score_handles_nan() {
        let good = make_test_entry("good", "good", 1);
        let nan_entry = make_test_entry("nan", "nan", 2);
        let low = make_test_entry("low", "low", 0);

        let mut map: HashMap<ProofId, ProofEntry> = HashMap::new();
        map.insert(good.id.clone(), good);
        map.insert(nan_entry.id.clone(), nan_entry);
        map.insert(low.id.clone(), low);

        let scored = map.values().map(|e| {
            let score = match e.id.0.as_str() {
                "good" => 0.9,
                "nan" => f64::NAN,
                "low" => 0.2,
                _ => 0.0,
            };
            (e, score)
        });

        let results = top_k_by_score(scored, 2);
        let ids: Vec<_> = results
            .iter()
            .map(|(entry, _)| entry.id.0.as_str())
            .collect();

        assert_eq!(ids, vec!["good", "low"]);
        assert!(results.iter().all(|(_, score)| score.is_finite()));
    }
}
