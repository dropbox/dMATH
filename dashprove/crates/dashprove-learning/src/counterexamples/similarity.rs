//! Similarity computation functions for counterexamples

use super::types::CounterexampleFeatures;

/// Compute similarity between two counterexample feature sets
pub(crate) fn compute_feature_similarity(
    a: &CounterexampleFeatures,
    b: &CounterexampleFeatures,
) -> f64 {
    // Weighted combination of different similarity metrics

    // Variable overlap (Jaccard similarity)
    let witness_sim = jaccard_similarity(&a.witness_vars, &b.witness_vars);
    let trace_vars_sim = jaccard_similarity(&a.trace_vars, &b.trace_vars);

    // Failed checks similarity
    let checks_sim = jaccard_similarity(&a.failed_check_ids, &b.failed_check_ids);
    let keywords_sim = jaccard_similarity(&a.failed_check_keywords, &b.failed_check_keywords);

    // Action similarity
    let action_sim = jaccard_similarity(&a.action_names, &b.action_names);

    // Trace length similarity (exponential decay)
    let len_diff = (a.trace_length as i32 - b.trace_length as i32).abs() as f64;
    let length_sim = (-len_diff / 5.0).exp();

    // Weighted average
    let weights = [0.15, 0.20, 0.25, 0.15, 0.10, 0.15];
    let scores = [
        witness_sim,
        trace_vars_sim,
        checks_sim,
        keywords_sim,
        action_sim,
        length_sim,
    ];

    weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum()
}

/// Jaccard similarity between two sets
pub(crate) fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let set_a: std::collections::HashSet<_> = a.iter().collect();
    let set_b: std::collections::HashSet<_> = b.iter().collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    intersection as f64 / union as f64
}

/// Compute keyword match score
pub(crate) fn compute_keyword_score(query_terms: &[String], keywords: &[String]) -> f64 {
    if query_terms.is_empty() || keywords.is_empty() {
        return 0.0;
    }

    let keyword_set: std::collections::HashSet<_> = keywords.iter().map(|s| s.as_str()).collect();

    let matches = query_terms
        .iter()
        .filter(|term| {
            keyword_set.contains(term.as_str())
                || keywords.iter().any(|kw| kw.contains(term.as_str()))
        })
        .count();

    matches as f64 / query_terms.len() as f64
}
