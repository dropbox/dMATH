//! History and statistics for proof corpus

use super::types::{ProofEntry, ProofHistory, ProofId};
use crate::counterexamples::{PeriodStats, TimePeriod};
use chrono::{DateTime, Utc};
use dashprove_backends::traits::BackendId;
use std::collections::HashMap;

/// Get corpus history statistics
///
/// Groups proofs by time periods (day, week, or month) and returns counts
/// per period and per backend.
pub fn compute_history(proofs: &HashMap<ProofId, ProofEntry>, period: TimePeriod) -> ProofHistory {
    if proofs.is_empty() {
        return ProofHistory::default();
    }

    let (min_time, max_time) = proofs.values().fold(
        (DateTime::<Utc>::MAX_UTC, DateTime::<Utc>::MIN_UTC),
        |acc, e| (acc.0.min(e.recorded_at), acc.1.max(e.recorded_at)),
    );

    let mut periods: HashMap<String, PeriodStats> = HashMap::new();
    let mut by_backend: HashMap<BackendId, usize> = HashMap::new();
    let mut by_property: HashMap<String, usize> = HashMap::new();
    let mut by_tactic: HashMap<String, usize> = HashMap::new();

    for entry in proofs.values() {
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
        *by_property.entry(entry.property.name()).or_insert(0) += 1;
        for tactic in &entry.tactics {
            *by_tactic.entry(tactic.clone()).or_insert(0) += 1;
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

    ProofHistory {
        total_count: proofs.len(),
        first_recorded: Some(min_time),
        last_recorded: Some(max_time),
        period_type: period,
        periods: period_list,
        cumulative_counts,
        by_backend,
        by_property,
        by_tactic,
    }
}

/// Get corpus history statistics within an optional time range
///
/// If `from` is Some, only include proofs recorded on or after that date.
/// If `to` is Some, only include proofs recorded on or before that date.
pub fn compute_history_in_range(
    proofs: &HashMap<ProofId, ProofEntry>,
    period: TimePeriod,
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
) -> ProofHistory {
    // Filter proofs by date range
    let filtered: Vec<&ProofEntry> = proofs
        .values()
        .filter(|e| {
            let after_start = from.is_none_or(|f| e.recorded_at >= f);
            let before_end = to.is_none_or(|t| e.recorded_at <= t);
            after_start && before_end
        })
        .collect();

    if filtered.is_empty() {
        return ProofHistory::default();
    }

    let (min_time, max_time) = filtered.iter().fold(
        (DateTime::<Utc>::MAX_UTC, DateTime::<Utc>::MIN_UTC),
        |acc, e| (acc.0.min(e.recorded_at), acc.1.max(e.recorded_at)),
    );

    let mut periods: HashMap<String, PeriodStats> = HashMap::new();
    let mut by_backend: HashMap<BackendId, usize> = HashMap::new();
    let mut by_property: HashMap<String, usize> = HashMap::new();
    let mut by_tactic: HashMap<String, usize> = HashMap::new();

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
        *by_property.entry(entry.property.name()).or_insert(0) += 1;
        for tactic in &entry.tactics {
            *by_tactic.entry(tactic.clone()).or_insert(0) += 1;
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

    ProofHistory {
        total_count: filtered.len(),
        first_recorded: Some(min_time),
        last_recorded: Some(max_time),
        period_type: period,
        periods: period_list,
        cumulative_counts,
        by_backend,
        by_property,
        by_tactic,
    }
}
