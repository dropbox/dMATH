//! Tests for proof corpus

use super::*;
use crate::counterexamples::TimePeriod;
use crate::LearnableResult;
use chrono::{Duration as ChronoDuration, TimeZone, Utc};
use dashprove_backends::traits::{BackendId, VerificationStatus};
use dashprove_usl::ast::{Expr, Invariant, Property, Theorem};
use std::time::Duration;

fn make_learnable(_name: &str, property: Property) -> LearnableResult {
    LearnableResult {
        property,
        backend: BackendId::Lean4,
        status: VerificationStatus::Proven,
        tactics: vec!["decide".to_string()],
        time_taken: Duration::from_millis(50),
        proof_output: None,
    }
}

#[test]
fn test_corpus_insert_and_get() {
    let mut corpus = ProofCorpus::new();

    let prop = Property::Invariant(Invariant {
        name: "test_inv".to_string(),
        body: Expr::Bool(true),
    });
    let result = make_learnable("test_inv", prop);

    let id = corpus.insert(&result);
    assert_eq!(corpus.len(), 1);

    let retrieved = corpus.get(&id).unwrap();
    assert_eq!(retrieved.tactics, vec!["decide".to_string()]);
}

#[test]
fn test_corpus_find_similar_structural() {
    let mut corpus = ProofCorpus::new();

    // Insert invariants (similar structure)
    for i in 0..3 {
        let prop = Property::Invariant(Invariant {
            name: format!("inv_{}", i),
            body: Expr::Bool(true),
        });
        corpus.insert(&make_learnable(&format!("inv_{}", i), prop));
    }

    // Insert theorems (different structure)
    for i in 0..2 {
        let prop = Property::Theorem(Theorem {
            name: format!("thm_{}", i),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: None,
                body: Box::new(Expr::Bool(true)),
            },
        });
        corpus.insert(&make_learnable(&format!("thm_{}", i), prop));
    }

    // Query with an invariant - should find invariants as more similar
    let query = Property::Invariant(Invariant {
        name: "query_inv".to_string(),
        body: Expr::Bool(true),
    });

    let similar = corpus.find_similar(&query, 3);
    assert_eq!(similar.len(), 3);

    // The most similar should be invariants (same property type)
    for s in &similar {
        assert!(s.similarity > 0.5);
    }
}

#[test]
fn test_corpus_by_backend() {
    let mut corpus = ProofCorpus::new();

    // LEAN proofs
    for i in 0..3 {
        let prop = Property::Invariant(Invariant {
            name: format!("lean_{}", i),
            body: Expr::Bool(true),
        });
        corpus.insert(&LearnableResult {
            property: prop,
            backend: BackendId::Lean4,
            status: VerificationStatus::Proven,
            tactics: vec![],
            time_taken: Duration::from_millis(10),
            proof_output: None,
        });
    }

    // TLA+ proofs
    for i in 0..2 {
        let prop = Property::Invariant(Invariant {
            name: format!("tla_{}", i),
            body: Expr::Bool(true),
        });
        corpus.insert(&LearnableResult {
            property: prop,
            backend: BackendId::TlaPlus,
            status: VerificationStatus::Proven,
            tactics: vec![],
            time_taken: Duration::from_millis(10),
            proof_output: None,
        });
    }

    assert_eq!(corpus.by_backend(BackendId::Lean4).len(), 3);
    assert_eq!(corpus.by_backend(BackendId::TlaPlus).len(), 2);
    assert_eq!(corpus.by_backend(BackendId::Kani).len(), 0);
}

#[test]
fn test_keyword_search_exact_match() {
    let mut corpus = ProofCorpus::new();

    // Insert proofs with distinctive names
    corpus.insert(&make_learnable(
        "loop_termination",
        Property::Theorem(Theorem {
            name: "loop_termination".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "array_bounds",
        Property::Invariant(Invariant {
            name: "array_bounds_check".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "safety_property",
        Property::Invariant(Invariant {
            name: "memory_safety".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Search for "termination" - should find loop_termination
    let results = corpus.search_by_keywords("termination", 5);
    assert!(!results.is_empty());
    assert!(results[0].property.name().contains("termination"));

    // Search for "bounds" - should find array_bounds_check
    let results = corpus.search_by_keywords("bounds", 5);
    assert!(!results.is_empty());
    assert!(results[0].property.name().contains("bounds"));

    // Search for "safety" - should find memory_safety
    let results = corpus.search_by_keywords("safety", 5);
    assert!(!results.is_empty());
    assert!(results[0].property.name().contains("safety"));
}

#[test]
fn test_keyword_search_multiple_terms() {
    let mut corpus = ProofCorpus::new();

    corpus.insert(&make_learnable(
        "loop",
        Property::Theorem(Theorem {
            name: "loop_invariant_preservation".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "array",
        Property::Invariant(Invariant {
            name: "array_index_valid".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Search for "loop invariant" - should find loop_invariant_preservation
    let results = corpus.search_by_keywords("loop invariant", 5);
    assert!(!results.is_empty());
    assert!(results[0].property.name().contains("loop"));
    assert!(results[0].property.name().contains("invariant"));
}

#[test]
fn test_keyword_search_no_matches() {
    let mut corpus = ProofCorpus::new();

    corpus.insert(&make_learnable(
        "test",
        Property::Invariant(Invariant {
            name: "test_property".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Search for unrelated terms
    let results = corpus.search_by_keywords("xyz123nonexistent", 5);
    assert!(results.is_empty());
}

#[test]
fn test_keyword_search_empty_corpus() {
    let corpus = ProofCorpus::new();
    let results = corpus.search_by_keywords("anything", 5);
    assert!(results.is_empty());
}

#[test]
fn test_keyword_search_empty_query() {
    let mut corpus = ProofCorpus::new();
    corpus.insert(&make_learnable(
        "test",
        Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    let results = corpus.search_by_keywords("", 5);
    assert!(results.is_empty());
}

#[test]
fn test_keyword_score_function() {
    // Exact match
    let score = search::compute_keyword_score(
        &["test".to_string()],
        &["test".to_string(), "other".to_string()],
    );
    assert!(score > 0.9, "Exact match should score high: {}", score);

    // Prefix match
    let score = search::compute_keyword_score(&["term".to_string()], &["termination".to_string()]);
    assert!(
        score > 0.5,
        "Prefix match should score moderately: {}",
        score
    );

    // No match
    let score = search::compute_keyword_score(
        &["xyz".to_string()],
        &["abc".to_string(), "def".to_string()],
    );
    assert_eq!(score, 0.0, "No match should score zero");

    // Multiple matches boost score
    let single_match = search::compute_keyword_score(
        &["loop".to_string(), "other".to_string()],
        &["loop".to_string()],
    );
    let double_match = search::compute_keyword_score(
        &["loop".to_string(), "inv".to_string()],
        &["loop".to_string(), "invariant".to_string()],
    );
    assert!(
        double_match > single_match,
        "Multiple matches should score higher"
    );
}

#[test]
fn test_recorded_at_timestamp_set_on_insert() {
    let mut corpus = ProofCorpus::new();
    let prop = Property::Invariant(Invariant {
        name: "timestamp".to_string(),
        body: Expr::Bool(true),
    });
    let result = make_learnable("timestamp", prop);

    let before = Utc::now();
    corpus.insert(&result);
    let after = Utc::now();

    let entry = corpus.proofs.values().next().unwrap();
    assert!(entry.recorded_at >= before);
    assert!(entry.recorded_at <= after);
}

#[test]
fn test_history_empty_corpus() {
    let corpus = ProofCorpus::new();
    let history = corpus.history(TimePeriod::Day);

    assert_eq!(history.total_count, 0);
    assert!(history.first_recorded.is_none());
    assert!(history.last_recorded.is_none());
    assert!(history.periods.is_empty());
    assert!(history.summary().contains("empty"));
}

#[test]
fn test_history_across_periods() {
    let mut corpus = ProofCorpus::new();

    let base = Utc.with_ymd_and_hms(2024, 5, 1, 12, 0, 0).unwrap();

    // First proof (LEAN) day 0
    let id1 = corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "p1".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Second proof (TLA+) day 1
    let id2 = corpus.insert(&make_learnable(
        "p2",
        Property::Invariant(Invariant {
            name: "p2".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Third proof (LEAN) day 2 with extra tactic
    let mut result = make_learnable(
        "p3",
        Property::Theorem(Theorem {
            name: "p3".to_string(),
            body: Expr::Bool(true),
        }),
    );
    result.tactics = vec!["simp".to_string(), "decide".to_string()];
    result.backend = BackendId::TlaPlus;
    let id3 = corpus.insert(&result);

    // Normalize timestamps for determinism
    let mut entries: Vec<_> = corpus.proofs.iter_mut().collect();
    entries.sort_by_key(|(id, _)| id.0.clone());
    entries[0].1.recorded_at = base;
    entries[1].1.recorded_at = base + ChronoDuration::days(1);
    entries[1].1.backend = BackendId::TlaPlus;
    entries[2].1.recorded_at = base + ChronoDuration::days(2);

    let history = corpus.history(TimePeriod::Day);
    assert_eq!(history.total_count, 3);
    assert_eq!(history.periods.len(), 3);
    assert_eq!(history.cumulative_counts, vec![1, 2, 3]);
    assert_eq!(history.by_backend.get(&BackendId::Lean4), Some(&1));
    assert_eq!(history.by_backend.get(&BackendId::TlaPlus), Some(&2));
    assert_eq!(history.by_property.get("p1"), Some(&1));
    assert!(history.by_tactic.contains_key("simp"));
    assert!(history.summary().contains("Total proofs: 3"));

    let html = history.to_html("Proof Corpus History");
    assert!(html.contains("<html>"));
    assert!(html.contains("Proof Corpus History"));
    assert!(html.contains("chart.js"));

    // Ensure IDs still present to avoid unused warnings
    assert!(corpus.get(&id1).is_some());
    assert!(corpus.get(&id2).is_some());
    assert!(corpus.get(&id3).is_some());
}

#[test]
fn test_in_time_range_filters_proofs() {
    let mut corpus = ProofCorpus::new();
    let prop = Property::Invariant(Invariant {
        name: "range_prop".to_string(),
        body: Expr::Bool(true),
    });
    let id = corpus.insert(&make_learnable("range", prop));

    let now = Utc::now();
    let early = now - ChronoDuration::hours(1);
    let later = now + ChronoDuration::hours(1);

    let entries = corpus.in_time_range(early, later);
    assert_eq!(entries.len(), 1);

    let far_past = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let far_future = Utc.with_ymd_and_hms(2020, 1, 2, 0, 0, 0).unwrap();
    assert!(corpus.in_time_range(far_past, far_future).is_empty());

    // Keep the compiler happy about unused ID
    assert!(corpus.get(&id).is_some());
}

#[test]
fn test_history_in_range_filters_by_from_date() {
    let mut corpus = ProofCorpus::new();

    let day1 = Utc.with_ymd_and_hms(2024, 6, 1, 12, 0, 0).unwrap();
    let day2 = Utc.with_ymd_and_hms(2024, 6, 2, 12, 0, 0).unwrap();
    let day3 = Utc.with_ymd_and_hms(2024, 6, 3, 12, 0, 0).unwrap();

    // Insert 3 proofs on different days
    let id1 = corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "p1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    let id2 = corpus.insert(&make_learnable(
        "p2",
        Property::Invariant(Invariant {
            name: "p2".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    let id3 = corpus.insert(&make_learnable(
        "p3",
        Property::Invariant(Invariant {
            name: "p3".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Set timestamps manually
    corpus.proofs.get_mut(&id1).unwrap().recorded_at = day1;
    corpus.proofs.get_mut(&id2).unwrap().recorded_at = day2;
    corpus.proofs.get_mut(&id3).unwrap().recorded_at = day3;

    // Filter from day2 onwards - should get 2 proofs
    let history = corpus.history_in_range(TimePeriod::Day, Some(day2), None);
    assert_eq!(history.total_count, 2);
    assert_eq!(history.periods.len(), 2);
    assert_eq!(history.by_property.get("p2"), Some(&1));
    assert_eq!(history.by_property.get("p3"), Some(&1));
    assert!(!history.by_property.contains_key("p1"));
}

#[test]
fn test_history_in_range_filters_by_to_date() {
    let mut corpus = ProofCorpus::new();

    let day1 = Utc.with_ymd_and_hms(2024, 6, 1, 12, 0, 0).unwrap();
    let day2 = Utc.with_ymd_and_hms(2024, 6, 2, 12, 0, 0).unwrap();
    let day3 = Utc.with_ymd_and_hms(2024, 6, 3, 12, 0, 0).unwrap();

    let id1 = corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "p1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    let id2 = corpus.insert(&make_learnable(
        "p2",
        Property::Invariant(Invariant {
            name: "p2".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    let id3 = corpus.insert(&make_learnable(
        "p3",
        Property::Invariant(Invariant {
            name: "p3".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    corpus.proofs.get_mut(&id1).unwrap().recorded_at = day1;
    corpus.proofs.get_mut(&id2).unwrap().recorded_at = day2;
    corpus.proofs.get_mut(&id3).unwrap().recorded_at = day3;

    // Filter up to day2 - should get 2 proofs
    let history = corpus.history_in_range(TimePeriod::Day, None, Some(day2));
    assert_eq!(history.total_count, 2);
    assert_eq!(history.by_property.get("p1"), Some(&1));
    assert_eq!(history.by_property.get("p2"), Some(&1));
    assert!(!history.by_property.contains_key("p3"));
}

#[test]
fn test_history_in_range_filters_by_both_dates() {
    let mut corpus = ProofCorpus::new();

    let day1 = Utc.with_ymd_and_hms(2024, 6, 1, 12, 0, 0).unwrap();
    let day2 = Utc.with_ymd_and_hms(2024, 6, 2, 12, 0, 0).unwrap();
    let day3 = Utc.with_ymd_and_hms(2024, 6, 3, 12, 0, 0).unwrap();

    let id1 = corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "p1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    let id2 = corpus.insert(&make_learnable(
        "p2",
        Property::Invariant(Invariant {
            name: "p2".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    let id3 = corpus.insert(&make_learnable(
        "p3",
        Property::Invariant(Invariant {
            name: "p3".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    corpus.proofs.get_mut(&id1).unwrap().recorded_at = day1;
    corpus.proofs.get_mut(&id2).unwrap().recorded_at = day2;
    corpus.proofs.get_mut(&id3).unwrap().recorded_at = day3;

    // Filter just day2 - should get 1 proof
    let history = corpus.history_in_range(TimePeriod::Day, Some(day2), Some(day2));
    assert_eq!(history.total_count, 1);
    assert_eq!(history.by_property.get("p2"), Some(&1));
}

#[test]
fn test_history_in_range_empty_result() {
    let mut corpus = ProofCorpus::new();

    let day1 = Utc.with_ymd_and_hms(2024, 6, 1, 12, 0, 0).unwrap();
    let far_future = Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap();

    let id1 = corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "p1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.proofs.get_mut(&id1).unwrap().recorded_at = day1;

    // Filter in future - should get empty result
    let history = corpus.history_in_range(TimePeriod::Day, Some(far_future), None);
    assert_eq!(history.total_count, 0);
    assert!(history.periods.is_empty());
    assert!(history.first_recorded.is_none());
}

#[test]
fn test_history_in_range_no_filter_same_as_history() {
    let mut corpus = ProofCorpus::new();

    let day1 = Utc.with_ymd_and_hms(2024, 6, 1, 12, 0, 0).unwrap();
    let day2 = Utc.with_ymd_and_hms(2024, 6, 2, 12, 0, 0).unwrap();

    let id1 = corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "p1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    let id2 = corpus.insert(&make_learnable(
        "p2",
        Property::Invariant(Invariant {
            name: "p2".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    corpus.proofs.get_mut(&id1).unwrap().recorded_at = day1;
    corpus.proofs.get_mut(&id2).unwrap().recorded_at = day2;

    // No filter - should be same as history()
    let history_all = corpus.history(TimePeriod::Day);
    let history_range = corpus.history_in_range(TimePeriod::Day, None, None);

    assert_eq!(history_all.total_count, history_range.total_count);
    assert_eq!(history_all.periods.len(), history_range.periods.len());
}

#[test]
fn test_compute_embeddings() {
    let mut corpus = ProofCorpus::new();

    corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "array_bounds".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "p2",
        Property::Theorem(Theorem {
            name: "loop_termination".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Initially no embeddings
    assert_eq!(corpus.embedding_count(), 0);

    // Compute embeddings
    let mut embedder = crate::embedder::PropertyEmbedder::new();
    let computed = corpus.compute_embeddings(&mut embedder);

    assert_eq!(computed, 2);
    assert_eq!(corpus.embedding_count(), 2);

    // Running again should not recompute
    let recomputed = corpus.compute_embeddings(&mut embedder);
    assert_eq!(recomputed, 0);
}

#[test]
fn test_find_similar_embedding_empty() {
    let corpus = ProofCorpus::new();
    let query = crate::embedder::Embedding::zeros(crate::embedder::EMBEDDING_DIM);
    let results = corpus.find_similar_embedding(&query, 5);
    assert!(results.is_empty());
}

#[test]
fn test_find_similar_embedding_basic() {
    let mut corpus = ProofCorpus::new();
    let mut embedder = crate::embedder::PropertyEmbedder::new();

    // Insert some proofs
    let inv1 = Property::Invariant(Invariant {
        name: "array_bounds_check".to_string(),
        body: Expr::Bool(true),
    });
    let inv2 = Property::Invariant(Invariant {
        name: "bounds_validation".to_string(),
        body: Expr::Bool(true),
    });
    let thm1 = Property::Theorem(Theorem {
        name: "loop_termination".to_string(),
        body: Expr::Bool(true),
    });

    corpus.insert(&make_learnable("inv1", inv1.clone()));
    corpus.insert(&make_learnable("inv2", inv2.clone()));
    corpus.insert(&make_learnable("thm1", thm1.clone()));

    // Compute embeddings
    corpus.compute_embeddings(&mut embedder);

    // Query with something similar to invariants
    let query_prop = Property::Invariant(Invariant {
        name: "array_check".to_string(),
        body: Expr::Bool(true),
    });
    let query_emb = embedder.embed_query(&query_prop);

    let results = corpus.find_similar_embedding(&query_emb, 2);

    assert_eq!(results.len(), 2);
    // All invariants should rank higher than theorem for invariant query
    // (both results should be invariants)
    for result in &results {
        assert!(
            matches!(result.property, Property::Invariant(_)),
            "Expected invariant, got {:?}",
            result.property
        );
    }
}

#[test]
fn test_find_similar_embedding_no_embeddings() {
    let mut corpus = ProofCorpus::new();

    // Insert proofs WITHOUT computing embeddings
    corpus.insert(&make_learnable(
        "p1",
        Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    let query = crate::embedder::Embedding::zeros(crate::embedder::EMBEDDING_DIM);
    let results = corpus.find_similar_embedding(&query, 5);

    // Should return empty since no proofs have embeddings
    assert!(results.is_empty());
}

#[test]
fn test_insert_with_embedding() {
    let mut corpus = ProofCorpus::new();
    let mut embedder = crate::embedder::PropertyEmbedder::new();

    let prop = Property::Invariant(Invariant {
        name: "test_property".to_string(),
        body: Expr::Bool(true),
    });
    let result = make_learnable("test", prop.clone());
    let embedding = embedder.embed(&prop);

    let id = corpus.insert_with_embedding(&result, embedding);

    // Should have embedding immediately
    assert_eq!(corpus.embedding_count(), 1);
    assert!(corpus.get(&id).unwrap().embedding.is_some());
}

// ========== Category-Filtered Search Tests ==========

use super::category::CategoryIndex;
use dashprove_usl::ast::{Contract, Param, Security, Temporal, TemporalExpr, Type};

fn make_contract(name: &str) -> Property {
    Property::Contract(Contract {
        type_path: vec![name.to_string()],
        params: vec![Param {
            name: "x".to_string(),
            ty: Type::Named("int".to_string()),
        }],
        return_type: None,
        requires: vec![Expr::Bool(true)],
        ensures: vec![Expr::Bool(true)],
        ensures_err: vec![],
        assigns: vec![],
        allocates: vec![],
        frees: vec![],
        terminates: None,
        behaviors: vec![],
        complete_behaviors: false,
        disjoint_behaviors: false,
        decreases: None,
    })
}

fn make_temporal(name: &str) -> Property {
    Property::Temporal(Temporal {
        name: name.to_string(),
        body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
        fairness: vec![],
    })
}

fn make_security(name: &str) -> Property {
    Property::Security(Security {
        name: name.to_string(),
        body: Expr::Bool(true),
    })
}

#[test]
fn test_category_index_populated_on_insert() {
    let mut corpus = ProofCorpus::new();

    // Insert invariants (category 0, subtype 2)
    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Insert contracts (category 0, subtype 1)
    corpus.insert(&make_learnable("contract1", make_contract("contract1")));

    // Insert temporal (category 1, subtype 0)
    corpus.insert(&make_learnable("temporal1", make_temporal("temporal1")));

    // Check category counts
    assert_eq!(corpus.category_count(0), 2); // invariant + contract
    assert_eq!(corpus.category_count(1), 1); // temporal
    assert_eq!(corpus.category_count(2), 0); // neural (empty)

    // Check subtype counts
    assert_eq!(corpus.subtype_count(0, 2), 1); // invariant
    assert_eq!(corpus.subtype_count(0, 1), 1); // contract
    assert_eq!(corpus.subtype_count(1, 0), 1); // temporal
}

#[test]
fn test_by_category() {
    let mut corpus = ProofCorpus::new();

    // Insert different property types
    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "thm1",
        Property::Theorem(Theorem {
            name: "thm1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable("temp1", make_temporal("temp1")));
    corpus.insert(&make_learnable("temp2", make_temporal("temp2")));
    corpus.insert(&make_learnable("sec1", make_security("sec1")));

    // Category 0: Theorem Proving (theorem + invariant)
    let cat0 = corpus.by_category(0);
    assert_eq!(cat0.len(), 2);

    // Category 1: Model Checking (temporal)
    let cat1 = corpus.by_category(1);
    assert_eq!(cat1.len(), 2);

    // Category 3: Security
    let cat3 = corpus.by_category(3);
    assert_eq!(cat3.len(), 1);

    // Empty category
    let cat2 = corpus.by_category(2);
    assert!(cat2.is_empty());
}

#[test]
fn test_by_subtype() {
    let mut corpus = ProofCorpus::new();

    // Insert invariants and theorems (both category 0, different subtypes)
    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "inv2",
        Property::Invariant(Invariant {
            name: "inv2".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "thm1",
        Property::Theorem(Theorem {
            name: "thm1".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Invariants: category 0, subtype 2
    let invariants = corpus.by_subtype(0, 2);
    assert_eq!(invariants.len(), 2);

    // Theorems: category 0, subtype 0
    let theorems = corpus.by_subtype(0, 0);
    assert_eq!(theorems.len(), 1);

    // Contracts: category 0, subtype 1 (empty)
    let contracts = corpus.by_subtype(0, 1);
    assert!(contracts.is_empty());
}

#[test]
fn test_find_similar_in_category() {
    let mut corpus = ProofCorpus::new();

    // Insert invariants (category 0)
    for i in 0..5 {
        corpus.insert(&make_learnable(
            &format!("inv_{}", i),
            Property::Invariant(Invariant {
                name: format!("invariant_{}", i),
                body: Expr::Bool(true),
            }),
        ));
    }

    // Insert temporal properties (category 1)
    for i in 0..3 {
        corpus.insert(&make_learnable(
            &format!("temp_{}", i),
            make_temporal(&format!("temporal_{}", i)),
        ));
    }

    // Query for similar within category 0 only
    let query = Property::Invariant(Invariant {
        name: "query_inv".to_string(),
        body: Expr::Bool(true),
    });

    let similar_cat0 = corpus.find_similar_in_category(&query, 0, 3);
    assert_eq!(similar_cat0.len(), 3);

    // All results should be invariants (from category 0)
    for result in &similar_cat0 {
        assert!(
            matches!(result.property, Property::Invariant(_)),
            "Expected invariant from category 0"
        );
    }

    // Query category 1 with same query should find no invariants
    let similar_cat1 = corpus.find_similar_in_category(&query, 1, 5);
    assert_eq!(similar_cat1.len(), 3); // Only 3 temporal properties exist

    for result in &similar_cat1 {
        assert!(
            matches!(result.property, Property::Temporal(_)),
            "Expected temporal from category 1"
        );
    }
}

#[test]
fn test_find_similar_in_subtype() {
    let mut corpus = ProofCorpus::new();

    // Insert invariants (category 0, subtype 2)
    for i in 0..4 {
        corpus.insert(&make_learnable(
            &format!("inv_{}", i),
            Property::Invariant(Invariant {
                name: format!("invariant_{}", i),
                body: Expr::Bool(true),
            }),
        ));
    }

    // Insert theorems (category 0, subtype 0)
    for i in 0..3 {
        corpus.insert(&make_learnable(
            &format!("thm_{}", i),
            Property::Theorem(Theorem {
                name: format!("theorem_{}", i),
                body: Expr::Bool(true),
            }),
        ));
    }

    let query = Property::Invariant(Invariant {
        name: "query".to_string(),
        body: Expr::Bool(true),
    });

    // Search only invariants (category 0, subtype 2)
    let similar_inv = corpus.find_similar_in_subtype(&query, 0, 2, 10);
    assert_eq!(similar_inv.len(), 4);

    for result in &similar_inv {
        assert!(matches!(result.property, Property::Invariant(_)));
    }

    // Search only theorems (category 0, subtype 0)
    let similar_thm = corpus.find_similar_in_subtype(&query, 0, 0, 10);
    assert_eq!(similar_thm.len(), 3);

    for result in &similar_thm {
        assert!(matches!(result.property, Property::Theorem(_)));
    }
}

#[test]
fn test_search_by_keywords_in_category() {
    let mut corpus = ProofCorpus::new();

    // Insert invariants with array-related names
    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "array_bounds".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "inv2",
        Property::Invariant(Invariant {
            name: "buffer_overflow".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    // Insert temporal with array-related name
    corpus.insert(&make_learnable("temp1", make_temporal("array_liveness")));

    // Search "array" in category 0 only
    let results = corpus.search_by_keywords_in_category("array", 0, 10);
    assert_eq!(results.len(), 1); // Only array_bounds is category 0

    // Search "array" in category 1
    let results_cat1 = corpus.search_by_keywords_in_category("array", 1, 10);
    assert_eq!(results_cat1.len(), 1); // Only array_liveness is category 1
}

#[test]
fn test_nonempty_categories() {
    let mut corpus = ProofCorpus::new();

    // Insert in specific categories
    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable("temp1", make_temporal("temp1")));
    corpus.insert(&make_learnable("sec1", make_security("sec1")));

    let nonempty = corpus.nonempty_categories();
    assert!(nonempty.contains(&0)); // Theorem Proving
    assert!(nonempty.contains(&1)); // Model Checking
    assert!(nonempty.contains(&3)); // Security
    assert!(!nonempty.contains(&2)); // Neural (empty)
    assert!(!nonempty.contains(&4)); // Memory Safety (empty)
}

#[test]
fn test_nonempty_subtypes() {
    let mut corpus = ProofCorpus::new();

    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable(
        "thm1",
        Property::Theorem(Theorem {
            name: "thm1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable("contract1", make_contract("contract1")));

    let nonempty = corpus.nonempty_subtypes();

    // Should contain (0, 0) for theorem, (0, 1) for contract, (0, 2) for invariant
    assert!(nonempty.contains(&(0, 0))); // theorem
    assert!(nonempty.contains(&(0, 1))); // contract
    assert!(nonempty.contains(&(0, 2))); // invariant
    assert!(!nonempty.contains(&(0, 3))); // refinement (empty)
}

#[test]
fn test_rebuild_category_index() {
    let mut corpus = ProofCorpus::new();

    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));
    corpus.insert(&make_learnable("temp1", make_temporal("temp1")));

    assert_eq!(corpus.category_count(0), 1);
    assert_eq!(corpus.category_count(1), 1);

    // Manually clear the index (simulating loading old corpus)
    corpus.category_index = CategoryIndex::new();
    assert_eq!(corpus.category_count(0), 0);
    assert_eq!(corpus.category_count(1), 0);

    // Rebuild
    corpus.rebuild_category_index();

    // Index should be restored
    assert_eq!(corpus.category_count(0), 1);
    assert_eq!(corpus.category_count(1), 1);
}

#[test]
fn test_find_similar_in_category_k_zero() {
    let mut corpus = ProofCorpus::new();
    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    let query = Property::Invariant(Invariant {
        name: "q".to_string(),
        body: Expr::Bool(true),
    });

    let results = corpus.find_similar_in_category(&query, 0, 0);
    assert!(results.is_empty());
}

#[test]
fn test_find_similar_in_category_empty_category() {
    let mut corpus = ProofCorpus::new();
    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    let query = Property::Invariant(Invariant {
        name: "q".to_string(),
        body: Expr::Bool(true),
    });

    // Category 7 (AI/ML) should be empty
    let results = corpus.find_similar_in_category(&query, 7, 10);
    assert!(results.is_empty());
}

#[test]
fn test_category_index_accessor() {
    let mut corpus = ProofCorpus::new();

    corpus.insert(&make_learnable(
        "inv1",
        Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        }),
    ));

    let index = corpus.category_index();
    assert_eq!(index.category_count(0), 1);
    assert_eq!(index.total_count(), 1);
}
