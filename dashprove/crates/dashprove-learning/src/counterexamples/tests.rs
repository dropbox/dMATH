//! Tests for counterexample types and operations

use super::comparison::{GrowthProjections, HistoryComparison};
use super::corpus::CounterexampleCorpus;
use super::history::{CorpusHistory, PeriodStats, TimePeriod};
use super::similarity::jaccard_similarity;
use super::suggestions::{format_suggestions, suggest_comparison_periods, SuggestionType};
use super::types::CounterexampleFeatures;
use chrono::Utc;
use dashprove_backends::traits::{
    BackendId, CounterexampleClusters, CounterexampleValue, FailedCheck, StructuredCounterexample,
    TraceState,
};
use std::collections::HashMap;

fn make_counterexample(
    witness_vars: &[(&str, i128)],
    check_desc: &str,
    trace_len: usize,
) -> StructuredCounterexample {
    let mut cx = StructuredCounterexample::new();

    for (name, value) in witness_vars {
        cx.witness.insert(
            name.to_string(),
            CounterexampleValue::Int {
                value: *value,
                type_hint: None,
            },
        );
    }

    if !check_desc.is_empty() {
        cx.failed_checks.push(FailedCheck {
            check_id: "test_check".to_string(),
            description: check_desc.to_string(),
            location: None,
            function: None,
        });
    }

    for i in 0..trace_len {
        let mut state = TraceState::new(i as u32 + 1);
        state.action = Some(format!("Action{}", i));
        state.variables.insert(
            "x".to_string(),
            CounterexampleValue::Int {
                value: i as i128,
                type_hint: None,
            },
        );
        cx.trace.push(state);
    }

    cx
}

#[test]
fn test_corpus_creation() {
    let corpus = CounterexampleCorpus::new();
    assert_eq!(corpus.len(), 0);
    assert!(corpus.is_empty());
}

#[test]
fn test_insert_counterexample() {
    let mut corpus = CounterexampleCorpus::new();

    let cx = make_counterexample(&[("n", 5)], "division by zero", 3);
    let id = corpus.insert("test_prop", BackendId::TlaPlus, cx, None);

    assert_eq!(corpus.len(), 1);
    assert!(corpus.get(&id).is_some());
}

#[test]
fn test_find_similar() {
    let mut corpus = CounterexampleCorpus::new();

    // Add several counterexamples
    for i in 0..5 {
        let cx = make_counterexample(&[("n", i)], "division by zero", 3);
        corpus.insert(&format!("prop_{}", i), BackendId::TlaPlus, cx, None);
    }

    // Find similar to a new counterexample
    let query = make_counterexample(&[("n", 10)], "division by zero", 3);
    let similar = corpus.find_similar(&query, 3);

    assert!(similar.len() <= 3);
    assert!(similar.iter().all(|s| s.similarity > 0.0));
}

#[test]
fn test_classify_with_patterns() {
    let mut corpus = CounterexampleCorpus::new();

    // Create a cluster pattern
    let cx1 = make_counterexample(&[("x", 1)], "overflow", 5);
    let cx2 = make_counterexample(&[("x", 2)], "overflow", 5);
    let clusters = CounterexampleClusters::from_counterexamples(vec![cx1, cx2], 0.5);

    corpus.record_clusters(&clusters);

    // Classify a similar counterexample
    let query = make_counterexample(&[("x", 3)], "overflow", 5);
    let classification = corpus.classify(&query);

    assert!(classification.is_some());
}

#[test]
fn test_search_by_keywords() {
    let mut corpus = CounterexampleCorpus::new();

    let cx1 = make_counterexample(&[("counter", 0)], "buffer overflow", 3);
    corpus.insert("prop1", BackendId::Kani, cx1, None);

    let cx2 = make_counterexample(&[("index", 0)], "null pointer", 3);
    corpus.insert("prop2", BackendId::Kani, cx2, None);

    let results = corpus.search_by_keywords("overflow", 5);
    assert!(!results.is_empty());
    assert!(results[0].property_name == "prop1");
}

#[test]
fn test_feature_extraction() {
    let cx = make_counterexample(&[("x", 1), ("y", 2)], "test error", 4);
    let features = CounterexampleFeatures::extract(&cx);

    assert_eq!(features.witness_vars.len(), 2);
    assert_eq!(features.trace_length, 4);
    assert!(!features.failed_check_ids.is_empty());
    assert!(!features.keywords.is_empty());
}

#[test]
fn test_jaccard_similarity() {
    let a = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let b = vec!["x".to_string(), "y".to_string(), "w".to_string()];

    let sim = jaccard_similarity(&a, &b);
    assert!((sim - 0.5).abs() < 0.01); // 2 shared out of 4 total

    let c: Vec<String> = vec![];
    assert_eq!(jaccard_similarity(&a, &c), 0.0);
    assert_eq!(jaccard_similarity(&c, &c), 1.0);
}

#[test]
fn test_persistence_roundtrip() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut corpus = CounterexampleCorpus::new();
    let cx = make_counterexample(&[("n", 42)], "test_failure", 3);
    corpus.insert(
        "persist_test",
        BackendId::TlaPlus,
        cx,
        Some("cluster_1".to_string()),
    );

    // Create temp file
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("cx_corpus_test_{}.json", ts));

    corpus.save_to_file(&path).unwrap();
    let loaded = CounterexampleCorpus::load_from_file(&path).unwrap();

    assert_eq!(loaded.len(), 1);

    std::fs::remove_file(path).ok();
}

// ========== Corpus History Tests ==========

#[test]
fn test_time_period_key_for() {
    use chrono::TimeZone;
    let dt = Utc.with_ymd_and_hms(2024, 6, 15, 10, 30, 0).unwrap();

    assert_eq!(TimePeriod::Day.key_for(dt), "2024-06-15");
    assert_eq!(TimePeriod::Month.key_for(dt), "2024-06");
    // Week number depends on the year
    assert!(TimePeriod::Week.key_for(dt).starts_with("2024-W"));
}

#[test]
fn test_time_period_parse() {
    assert_eq!("day".parse::<TimePeriod>().unwrap(), TimePeriod::Day);
    assert_eq!("weekly".parse::<TimePeriod>().unwrap(), TimePeriod::Week);
    assert_eq!("month".parse::<TimePeriod>().unwrap(), TimePeriod::Month);
    assert!("invalid".parse::<TimePeriod>().is_err());
}

#[test]
fn test_empty_corpus_history() {
    let corpus = CounterexampleCorpus::new();
    let history = corpus.history(TimePeriod::Day);

    assert_eq!(history.total_count, 0);
    assert!(history.first_recorded.is_none());
    assert!(history.last_recorded.is_none());
    assert!(history.periods.is_empty());
}

#[test]
fn test_corpus_history_single_entry() {
    let mut corpus = CounterexampleCorpus::new();
    let cx = make_counterexample(&[("x", 1)], "test error", 3);
    corpus.insert("test_prop", BackendId::TlaPlus, cx, None);

    let history = corpus.history(TimePeriod::Day);

    assert_eq!(history.total_count, 1);
    assert!(history.first_recorded.is_some());
    assert!(history.last_recorded.is_some());
    assert_eq!(history.periods.len(), 1);
    assert_eq!(history.periods[0].count, 1);
    assert_eq!(history.cumulative_counts, vec![1]);
    assert_eq!(history.by_backend.get(&BackendId::TlaPlus), Some(&1));
}

#[test]
fn test_corpus_history_multiple_backends() {
    let mut corpus = CounterexampleCorpus::new();

    // Add counterexamples from different backends
    let cx1 = make_counterexample(&[("x", 1)], "error 1", 3);
    corpus.insert("prop1", BackendId::TlaPlus, cx1, None);

    let cx2 = make_counterexample(&[("y", 2)], "error 2", 3);
    corpus.insert("prop2", BackendId::Kani, cx2, None);

    let cx3 = make_counterexample(&[("z", 3)], "error 3", 3);
    corpus.insert("prop3", BackendId::TlaPlus, cx3, None);

    let history = corpus.history(TimePeriod::Day);

    assert_eq!(history.total_count, 3);
    assert_eq!(history.by_backend.get(&BackendId::TlaPlus), Some(&2));
    assert_eq!(history.by_backend.get(&BackendId::Kani), Some(&1));
}

#[test]
fn test_corpus_history_summary() {
    let mut corpus = CounterexampleCorpus::new();
    let cx = make_counterexample(&[("x", 1)], "test error", 3);
    corpus.insert(
        "test_prop",
        BackendId::TlaPlus,
        cx,
        Some("cluster_A".to_string()),
    );

    let history = corpus.history(TimePeriod::Day);
    let summary = history.summary();

    assert!(summary.contains("Total counterexamples: 1"));
    assert!(summary.contains("TlaPlus"));
    assert!(summary.contains("cluster_A"));
}

#[test]
fn test_corpus_history_html_generation() {
    let mut corpus = CounterexampleCorpus::new();
    let cx = make_counterexample(&[("x", 1)], "test error", 3);
    corpus.insert("test_prop", BackendId::TlaPlus, cx, None);

    let history = corpus.history(TimePeriod::Day);
    let html = history.to_html("Test Corpus");

    assert!(html.contains("<html>"));
    assert!(html.contains("Test Corpus"));
    assert!(html.contains("chart.js")); // chart.js is lowercase in the CDN URL
    assert!(html.contains("Total Counterexamples"));
}

#[test]
fn test_recorded_at_timestamp() {
    let mut corpus = CounterexampleCorpus::new();
    let cx = make_counterexample(&[("x", 1)], "test error", 3);

    let before = Utc::now();
    corpus.insert("test_prop", BackendId::TlaPlus, cx, None);
    let after = Utc::now();

    let entry = corpus.all_entries().next().unwrap();
    assert!(entry.recorded_at >= before);
    assert!(entry.recorded_at <= after);
}

#[test]
fn test_in_time_range() {
    use chrono::{Duration, TimeZone};

    let mut corpus = CounterexampleCorpus::new();
    let cx = make_counterexample(&[("x", 1)], "test error", 3);
    corpus.insert("test_prop", BackendId::TlaPlus, cx, None);

    let now = Utc::now();
    let one_hour_ago = now - Duration::hours(1);
    let one_hour_later = now + Duration::hours(1);

    // Should find the entry
    let results = corpus.in_time_range(one_hour_ago, one_hour_later);
    assert_eq!(results.len(), 1);

    // Should not find it outside range
    let way_back = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let also_back = Utc.with_ymd_and_hms(2020, 1, 2, 0, 0, 0).unwrap();
    let results = corpus.in_time_range(way_back, also_back);
    assert_eq!(results.len(), 0);
}

// ========== HistoryComparison Tests ==========

#[test]
fn test_history_comparison_basic() {
    let mut corpus = CounterexampleCorpus::new();

    // Create data (3 counterexamples)
    for i in 0..3 {
        let cx = make_counterexample(&[("x", i)], "error", 3);
        corpus.insert(&format!("prop_{}", i), BackendId::TlaPlus, cx, None);
    }

    // Get history for the corpus
    let history = corpus.history(TimePeriod::Day);

    // Create a basic comparison with same data
    let cmp = HistoryComparison::from_corpus_histories(&history, &history, "Week 1", "Week 2");

    assert_eq!(cmp.baseline_count, cmp.comparison_count);
    assert_eq!(cmp.count_delta, 0);
}

#[test]
fn test_history_comparison_from_zero_baseline() {
    let baseline = CorpusHistory::default();
    let comparison = CorpusHistory {
        total_count: 5,
        periods: vec![PeriodStats {
            period: "2024-06-01".to_string(),
            start: Utc::now(),
            count: 5,
            by_backend: HashMap::new(),
        }],
        ..Default::default()
    };

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "Empty", "Has Data");

    assert_eq!(cmp.baseline_count, 0);
    assert_eq!(cmp.comparison_count, 5);
    assert_eq!(cmp.count_delta, 5);
    assert!(cmp.count_percent_change.unwrap().is_infinite());
}

#[test]
fn test_history_comparison_both_empty() {
    let baseline = CorpusHistory::default();
    let comparison = CorpusHistory::default();

    let cmp =
        HistoryComparison::from_corpus_histories(&baseline, &comparison, "Empty 1", "Empty 2");

    assert_eq!(cmp.baseline_count, 0);
    assert_eq!(cmp.comparison_count, 0);
    assert_eq!(cmp.count_delta, 0);
    assert!(cmp.count_percent_change.is_none());
}

#[test]
fn test_history_comparison_to_html_basic() {
    let baseline = CorpusHistory {
        total_count: 10,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 6), (BackendId::Lean4, 4)]),
        periods: vec![],
        cumulative_counts: vec![],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = CorpusHistory {
        total_count: 15,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 8), (BackendId::Lean4, 7)]),
        periods: vec![],
        cumulative_counts: vec![],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "Week 1", "Week 2");

    let html = cmp.to_html();

    // Verify HTML structure
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<html>"));
    assert!(html.contains("Period Comparison"));
    assert!(html.contains("Week 1"));
    assert!(html.contains("Week 2"));
    assert!(html.contains("chart.js"));

    // Verify count values
    assert!(html.contains("10")); // baseline count
    assert!(html.contains("15")); // comparison count
    assert!(html.contains("+5")); // delta
    assert!(html.contains("+50.0%")); // percent change

    // Verify backend chart data
    assert!(html.contains("backendLabels"));
    assert!(html.contains("backendDeltas"));
}

#[test]
fn test_history_comparison_to_html_with_new_backends() {
    let baseline = CorpusHistory {
        total_count: 5,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 5)]),
        periods: vec![],
        cumulative_counts: vec![],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = CorpusHistory {
        total_count: 10,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 6), (BackendId::Lean4, 4)]),
        periods: vec![],
        cumulative_counts: vec![],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "Before", "After");

    let html = cmp.to_html();

    // Should contain new backend alert
    assert!(html.contains("New backends:"));
    assert!(html.contains("Lean4"));
    assert!(html.contains("alert new"));
}

#[test]
fn test_history_comparison_to_html_negative_delta() {
    let baseline = CorpusHistory {
        total_count: 20,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 20)]),
        periods: vec![],
        cumulative_counts: vec![],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = CorpusHistory {
        total_count: 15,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 15)]),
        periods: vec![],
        cumulative_counts: vec![],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "Before", "After");

    let html = cmp.to_html();

    // Should show negative change with proper class
    assert!(html.contains("-5")); // negative delta
    assert!(html.contains("-25.0%")); // percent change
    assert!(html.contains("negative")); // delta class
}

#[test]
fn test_history_comparison_to_html_download_button() {
    let baseline = CorpusHistory {
        total_count: 5,
        period_type: TimePeriod::Day,
        by_backend: HashMap::new(),
        periods: vec![],
        cumulative_counts: vec![],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = baseline.clone();

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "A", "B");

    let html = cmp.to_html();

    // Should have download functionality
    assert!(html.contains("download-btn"));
    assert!(html.contains("downloadData"));
    assert!(html.contains("period_comparison.json"));
}

// ========== Growth Rate and Projection Tests ==========

#[test]
fn test_growth_rate_calculation_positive_growth() {
    // Baseline: 1 period with avg 10
    // Comparison: 1 period with avg 15
    // Growth rate should be 1.5x (50% increase)
    let baseline = CorpusHistory {
        total_count: 10,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 10)]),
        periods: vec![PeriodStats {
            period: "2024-06-01".to_string(),
            start: Utc::now(),
            count: 10,
            by_backend: HashMap::from([(BackendId::TlaPlus, 10)]),
        }],
        cumulative_counts: vec![10],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = CorpusHistory {
        total_count: 15,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 15)]),
        periods: vec![PeriodStats {
            period: "2024-06-02".to_string(),
            start: Utc::now(),
            count: 15,
            by_backend: HashMap::from([(BackendId::TlaPlus, 15)]),
        }],
        cumulative_counts: vec![15],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "Day 1", "Day 2");

    // Growth rate should be 1.5 (15/10)
    assert!(cmp.growth_rate_per_period.is_some());
    let rate = cmp.growth_rate_per_period.unwrap();
    assert!((rate - 1.5).abs() < 0.001);

    // Compound growth rate should be 0.5 (50%)
    assert!(cmp.compound_growth_rate.is_some());
    let compound = cmp.compound_growth_rate.unwrap();
    assert!((compound - 0.5).abs() < 0.001);
}

#[test]
fn test_projection_count_positive_growth() {
    let baseline = CorpusHistory {
        total_count: 10,
        period_type: TimePeriod::Day,
        by_backend: HashMap::new(),
        periods: vec![PeriodStats {
            period: "2024-06-01".to_string(),
            start: Utc::now(),
            count: 10,
            by_backend: HashMap::new(),
        }],
        cumulative_counts: vec![10],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = CorpusHistory {
        total_count: 20,
        period_type: TimePeriod::Day,
        by_backend: HashMap::new(),
        periods: vec![PeriodStats {
            period: "2024-06-02".to_string(),
            start: Utc::now(),
            count: 20,
            by_backend: HashMap::new(),
        }],
        cumulative_counts: vec![20],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "A", "B");

    // 100% growth rate, base = 20
    // Period 1: 20 * 2^1 = 40
    // Period 2: 20 * 2^2 = 80
    // Period 3: 20 * 2^3 = 160
    // Total for 3 periods: 40 + 80 + 160 = 280
    let proj_3 = cmp.project_count(3);
    assert!(proj_3.is_some());
    assert!((proj_3.unwrap() - 280.0).abs() < 1.0);

    // Project avg at 3 periods: 20 * 2^3 = 160
    let avg_at_3 = cmp.project_avg_at(3);
    assert!(avg_at_3.is_some());
    assert!((avg_at_3.unwrap() - 160.0).abs() < 1.0);
}

#[test]
fn test_projection_no_data() {
    let cmp = HistoryComparison::from_corpus_histories(
        &CorpusHistory::default(),
        &CorpusHistory::default(),
        "A",
        "B",
    );

    assert!(cmp.project_count(3).is_none());
    assert!(cmp.project_avg_at(3).is_none());
}

#[test]
fn test_projections_struct() {
    let baseline = CorpusHistory {
        total_count: 10,
        period_type: TimePeriod::Day,
        by_backend: HashMap::new(),
        periods: vec![PeriodStats {
            period: "2024-06-01".to_string(),
            start: Utc::now(),
            count: 10,
            by_backend: HashMap::new(),
        }],
        cumulative_counts: vec![10],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = CorpusHistory {
        total_count: 15,
        period_type: TimePeriod::Day,
        by_backend: HashMap::new(),
        periods: vec![PeriodStats {
            period: "2024-06-02".to_string(),
            start: Utc::now(),
            count: 15,
            by_backend: HashMap::new(),
        }],
        cumulative_counts: vec![15],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "A", "B");
    let projections = cmp.projections();

    // All projections should be available
    assert!(projections.periods_3_total.is_some());
    assert!(projections.periods_6_total.is_some());
    assert!(projections.periods_12_total.is_some());
    assert!(projections.periods_3_avg.is_some());
    assert!(projections.periods_6_avg.is_some());
    assert!(projections.periods_12_avg.is_some());
    assert!(projections.growth_rate_per_period.is_some());
    assert!(projections.compound_growth_rate.is_some());
}

#[test]
fn test_growth_projections_summary_empty() {
    let projections = GrowthProjections::default();
    let summary = projections.summary();

    assert!(summary.contains("No growth projections available"));
}

// ========== Period Suggestion Tests ==========

#[test]
fn test_suggest_comparison_periods_with_no_data() {
    use chrono::TimeZone;
    // With no data range provided, should use defaults (365 days before reference)
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let suggestions = suggest_comparison_periods(None, None, Some(reference));

    // Should get all suggestion types with default 365-day span
    assert!(!suggestions.is_empty());

    // Week over week should be present
    assert!(suggestions
        .iter()
        .any(|s| s.suggestion_type == SuggestionType::WeekOverWeek));
}

#[test]
fn test_suggest_comparison_periods_week_over_week() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(30); // 30 days of data
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));

    // Should include week over week
    let wow = suggestions
        .iter()
        .find(|s| s.suggestion_type == SuggestionType::WeekOverWeek);
    assert!(wow.is_some());

    let wow = wow.unwrap();
    assert!(wow.has_sufficient_data);
    assert_eq!(wow.recommended_period, TimePeriod::Day);
    assert_eq!(wow.expected_period_count, 7);

    // Verify dates are correct
    // This week: June 9-15, Last week: June 2-8
    assert_eq!(
        wow.comparison_end.format("%Y-%m-%d").to_string(),
        "2024-06-15"
    );
}

#[test]
fn test_suggest_comparison_periods_month_over_month() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(90); // 90 days of data
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));

    // Should include month over month
    let mom = suggestions
        .iter()
        .find(|s| s.suggestion_type == SuggestionType::MonthOverMonth);
    assert!(mom.is_some());

    let mom = mom.unwrap();
    assert!(mom.has_sufficient_data);
    assert_eq!(mom.recommended_period, TimePeriod::Day);

    // This month starts June 1
    assert_eq!(
        mom.comparison_start.format("%Y-%m-%d").to_string(),
        "2024-06-01"
    );
    // Last month is May
    assert_eq!(
        mom.baseline_start.format("%Y-%m-%d").to_string(),
        "2024-05-01"
    );
}

#[test]
fn test_suggest_comparison_periods_rolling_days() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(90);
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));

    // Should have rolling 7 and 30 day comparisons
    let rolling_7 = suggestions
        .iter()
        .find(|s| s.suggestion_type == SuggestionType::RollingDays(7));
    assert!(rolling_7.is_some());

    let rolling_30 = suggestions
        .iter()
        .find(|s| s.suggestion_type == SuggestionType::RollingDays(30));
    assert!(rolling_30.is_some());

    let r30 = rolling_30.unwrap();
    assert_eq!(r30.recommended_period, TimePeriod::Week);
    assert_eq!(r30.expected_period_count, 4);
}

#[test]
fn test_suggest_comparison_periods_insufficient_data() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(5); // Only 5 days
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));

    // With only 5 days, no suggestions should be available
    assert!(suggestions.is_empty());
}

#[test]
fn test_period_suggestion_cli_args() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(30);
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));
    let wow = suggestions
        .iter()
        .find(|s| s.suggestion_type == SuggestionType::WeekOverWeek)
        .unwrap();

    let args = wow.cli_args();
    assert!(!args.baseline_from.is_empty());
    assert!(!args.baseline_to.is_empty());
    assert!(!args.compare_from.is_empty());
    assert!(!args.compare_to.is_empty());
    assert_eq!(args.period, "day");

    // CLI command should be properly formatted
    let cmd = args.to_cli_command();
    assert!(cmd.contains("--baseline-from"));
    assert!(cmd.contains("--baseline-to"));
    assert!(cmd.contains("--compare-from"));
    assert!(cmd.contains("--compare-to"));
    assert!(cmd.contains("--period day"));
}

#[test]
fn test_format_suggestions() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(400);
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));
    let formatted = format_suggestions(&suggestions);

    assert!(formatted.contains("Available period comparisons:"));
    assert!(formatted.contains("Week over Week"));
    assert!(formatted.contains("Month over Month"));
    assert!(formatted.contains("CLI:"));
}

#[test]
fn test_format_suggestions_empty() {
    let formatted = format_suggestions(&[]);
    assert!(formatted.contains("No comparison suggestions available"));
}

#[test]
fn test_suggestion_type_display() {
    assert_eq!(
        format!("{}", SuggestionType::WeekOverWeek),
        "Week over Week"
    );
    assert_eq!(
        format!("{}", SuggestionType::MonthOverMonth),
        "Month over Month"
    );
    assert_eq!(
        format!("{}", SuggestionType::RollingDays(7)),
        "Last 7 days vs previous 7 days"
    );
    assert_eq!(
        format!("{}", SuggestionType::QuarterOverQuarter),
        "Quarter over Quarter"
    );
    assert_eq!(
        format!("{}", SuggestionType::YearOverYear),
        "Year over Year"
    );
}

#[test]
fn test_period_suggestion_baseline_label() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(30);
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));
    let wow = suggestions
        .iter()
        .find(|s| s.suggestion_type == SuggestionType::WeekOverWeek)
        .unwrap();

    let baseline_label = wow.baseline_label();
    let comparison_label = wow.comparison_label();

    // Labels should be formatted as "YYYY-MM-DD to YYYY-MM-DD"
    assert!(baseline_label.contains(" to "));
    assert!(comparison_label.contains(" to "));
}

#[test]
fn test_period_suggestion_ranges() {
    use chrono::TimeZone;
    let reference = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
    let first = reference - chrono::Duration::days(30);
    let last = reference;

    let suggestions = suggest_comparison_periods(Some(first), Some(last), Some(reference));
    let wow = suggestions
        .iter()
        .find(|s| s.suggestion_type == SuggestionType::WeekOverWeek)
        .unwrap();

    let (b_start, b_end) = wow.baseline_range();
    let (c_start, c_end) = wow.comparison_range();

    // Baseline should come before comparison
    assert!(b_end < c_start);
    // Each range should be about 7 days (for week-over-week)
    let baseline_days = (b_end - b_start).num_days();
    let comparison_days = (c_end - c_start).num_days();
    assert_eq!(baseline_days, 6); // 7 days = 6 day difference
    assert_eq!(comparison_days, 6);
}

// ========== Side-by-Side Period Charts Tests ==========

#[test]
fn test_to_html_with_periods_includes_period_charts() {
    let baseline = CorpusHistory {
        total_count: 10,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 10)]),
        periods: vec![
            PeriodStats {
                period: "2024-06-01".to_string(),
                start: Utc::now(),
                count: 5,
                by_backend: HashMap::from([(BackendId::TlaPlus, 5)]),
            },
            PeriodStats {
                period: "2024-06-02".to_string(),
                start: Utc::now(),
                count: 5,
                by_backend: HashMap::from([(BackendId::TlaPlus, 5)]),
            },
        ],
        cumulative_counts: vec![5, 10],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };
    let comparison = CorpusHistory {
        total_count: 15,
        period_type: TimePeriod::Day,
        by_backend: HashMap::from([(BackendId::TlaPlus, 15)]),
        periods: vec![
            PeriodStats {
                period: "2024-06-03".to_string(),
                start: Utc::now(),
                count: 7,
                by_backend: HashMap::from([(BackendId::TlaPlus, 7)]),
            },
            PeriodStats {
                period: "2024-06-04".to_string(),
                start: Utc::now(),
                count: 8,
                by_backend: HashMap::from([(BackendId::TlaPlus, 8)]),
            },
        ],
        cumulative_counts: vec![7, 15],
        by_property: HashMap::new(),
        by_cluster: HashMap::new(),
        first_recorded: None,
        last_recorded: None,
    };

    let cmp =
        HistoryComparison::from_corpus_histories(&baseline, &comparison, "Jun 1-2", "Jun 3-4");
    let html = cmp.to_html_with_periods(&baseline.periods, &comparison.periods);

    // Should include side-by-side period charts
    assert!(html.contains("Side-by-Side Period Comparison"));
    assert!(html.contains("baselineChart"));
    assert!(html.contains("comparisonChart"));
    assert!(html.contains("cumulativeComparisonChart"));

    // Should include the period labels in JSON
    assert!(html.contains("2024-06-01"));
    assert!(html.contains("2024-06-03"));

    // Should include the CSS for period charts
    assert!(html.contains(".period-charts"));
    assert!(html.contains("grid-template-columns: 1fr 1fr"));
}

#[test]
fn test_to_html_with_periods_handles_empty_periods() {
    let baseline = CorpusHistory::default();
    let comparison = CorpusHistory::default();

    let cmp = HistoryComparison::from_corpus_histories(&baseline, &comparison, "A", "B");
    let html = cmp.to_html_with_periods(&baseline.periods, &comparison.periods);

    // Should still render (empty arrays in JSON)
    assert!(html.contains("Side-by-Side Period Comparison"));
    assert!(html.contains("baselineChart"));

    // Should have empty arrays in JSON
    assert!(html.contains("[]"));
}

// ============================================================================
// Mutation-killing tests for similarity functions
// ============================================================================

mod similarity_mutation_tests {
    use super::super::similarity::{compute_feature_similarity, compute_keyword_score};
    use super::CounterexampleFeatures;

    #[test]
    fn test_compute_feature_similarity_returns_one_for_identical() {
        let f = CounterexampleFeatures {
            witness_vars: vec!["x".to_string(), "y".to_string()],
            trace_vars: vec!["state".to_string()],
            trace_length: 5,
            failed_check_ids: vec!["check1".to_string()],
            failed_check_keywords: vec!["overflow".to_string()],
            action_names: vec!["step".to_string()],
            keywords: vec![],
        };

        let sim = compute_feature_similarity(&f, &f);
        // Identical features should have very high similarity
        assert!(
            sim > 0.99,
            "Identical features should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_compute_feature_similarity_respects_trace_length_difference() {
        let f1 = CounterexampleFeatures {
            witness_vars: vec!["x".to_string()],
            trace_vars: vec!["s".to_string()],
            trace_length: 10,
            failed_check_ids: vec!["c".to_string()],
            failed_check_keywords: vec!["kw".to_string()],
            action_names: vec!["a".to_string()],
            keywords: vec![],
        };

        let mut f2 = f1.clone();
        f2.trace_length = 30; // 20 different -> exp(-20/5) = exp(-4) ≈ 0.018

        let sim = compute_feature_similarity(&f1, &f2);
        let self_sim = compute_feature_similarity(&f1, &f1);

        // Larger trace difference should reduce similarity
        assert!(
            sim < self_sim,
            "Different trace lengths should reduce similarity"
        );

        // The length component has weight 0.15
        // If subtraction became addition: |10+30|/5 = 8, exp(-8) ≈ 0 vs exp(-4) ≈ 0.018
        // This would change the result measurably
    }

    #[test]
    fn test_compute_feature_similarity_trace_length_exp_decay() {
        // Test the exponential decay formula: exp(-len_diff / 5.0)
        let f1 = CounterexampleFeatures {
            witness_vars: vec![],
            trace_vars: vec![],
            trace_length: 0,
            failed_check_ids: vec![],
            failed_check_keywords: vec![],
            action_names: vec![],
            keywords: vec![],
        };

        let mut f2 = f1.clone();
        f2.trace_length = 5; // diff = 5, exp(-1) ≈ 0.368

        let mut f3 = f1.clone();
        f3.trace_length = 10; // diff = 10, exp(-2) ≈ 0.135

        let sim1 = compute_feature_similarity(&f1, &f2);
        let sim2 = compute_feature_similarity(&f1, &f3);

        // sim1 should be higher than sim2 (smaller diff)
        assert!(
            sim1 > sim2,
            "Smaller trace diff should yield higher similarity"
        );

        // If division were replaced with multiplication: exp(-5*5) vs exp(-10*5) - both basically 0
        // This distinguishes between /5.0 and *5.0
    }

    #[test]
    fn test_compute_feature_similarity_weighted_sum() {
        // Weights: [0.15, 0.20, 0.25, 0.15, 0.10, 0.15]
        // Test that weights sum to 1.0 indirectly via result bounds
        let f = CounterexampleFeatures {
            witness_vars: vec!["a".to_string()],
            trace_vars: vec!["b".to_string()],
            trace_length: 3,
            failed_check_ids: vec!["c".to_string()],
            failed_check_keywords: vec!["d".to_string()],
            action_names: vec!["e".to_string()],
            keywords: vec![],
        };

        let sim = compute_feature_similarity(&f, &f);
        // Result should be close to 1.0 for identical
        assert!(sim > 0.9, "Identical features should yield high similarity");
        assert!(sim <= 1.0, "Similarity should not exceed 1.0");
    }

    #[test]
    fn test_compute_feature_similarity_respects_witness_vars() {
        let f1 = CounterexampleFeatures {
            witness_vars: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            trace_vars: vec![],
            trace_length: 1,
            failed_check_ids: vec![],
            failed_check_keywords: vec![],
            action_names: vec![],
            keywords: vec![],
        };

        let mut f2 = f1.clone();
        f2.witness_vars = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let sim = compute_feature_similarity(&f1, &f2);
        let self_sim = compute_feature_similarity(&f1, &f1);

        assert!(
            sim < self_sim,
            "Different witness vars should reduce similarity"
        );
    }

    #[test]
    fn test_jaccard_similarity_both_empty() {
        use super::super::similarity::jaccard_similarity;

        let empty: Vec<String> = vec![];
        let sim = jaccard_similarity(&empty, &empty);
        assert_eq!(sim, 1.0, "Both empty sets should have similarity 1.0");
    }

    #[test]
    fn test_jaccard_similarity_one_empty() {
        use super::super::similarity::jaccard_similarity;

        let filled = vec!["a".to_string(), "b".to_string()];
        let empty: Vec<String> = vec![];

        // If || became &&: both would need to be empty to return 0.0
        // With ||: either empty → return 0.0 immediately
        // With &&: must have both empty to return 0.0, otherwise continues
        let sim1 = jaccard_similarity(&filled, &empty);
        let sim2 = jaccard_similarity(&empty, &filled);

        assert_eq!(sim1, 0.0, "One empty set should yield 0.0");
        assert_eq!(sim2, 0.0, "One empty set should yield 0.0");

        // Additional test case to specifically catch || -> && mutation:
        // If we have: a=[x], b=[]
        // With ||: a.is_empty() || b.is_empty() => false || true => true => return 0.0
        // With &&: a.is_empty() && b.is_empty() => false && true => false => continue to compute
        // When it continues, it would try to compute intersection/union but union would be non-zero
        // and intersection would be 0, so result would be 0/n = 0.0 (same result in this case)

        // To really distinguish, we need a case where continuing gives different behavior
        // Actually the issue is that with &&, it continues past the guard, but then
        // intersection = 0, union = |filled| = 2, so 0/2 = 0.0 - same result!
        // So this mutation is semantically equivalent for the return value.
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        use super::super::similarity::jaccard_similarity;

        let set = vec!["x".to_string(), "y".to_string()];
        let sim = jaccard_similarity(&set, &set);
        assert_eq!(sim, 1.0, "Identical sets should have similarity 1.0");
    }

    #[test]
    fn test_jaccard_similarity_partial_overlap() {
        use super::super::similarity::jaccard_similarity;

        let a = vec!["x".to_string(), "y".to_string()];
        let b = vec!["y".to_string(), "z".to_string()];

        // intersection = {y}, union = {x, y, z}
        // Jaccard = 1/3 ≈ 0.333
        let sim = jaccard_similarity(&a, &b);
        assert!(
            (sim - 0.333).abs() < 0.01,
            "Partial overlap should yield ~0.333"
        );
    }

    #[test]
    fn test_jaccard_similarity_no_overlap() {
        use super::super::similarity::jaccard_similarity;

        let a = vec!["x".to_string()];
        let b = vec!["y".to_string()];

        // intersection = {}, union = {x, y}
        // Jaccard = 0/2 = 0.0
        let sim = jaccard_similarity(&a, &b);
        assert_eq!(sim, 0.0, "No overlap should yield 0.0");
    }

    #[test]
    fn test_jaccard_similarity_division_order() {
        use super::super::similarity::jaccard_similarity;

        // Test that it's intersection/union, not union/intersection
        let a = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let b = vec!["x".to_string()];

        // intersection = 1, union = 3
        // Correct: 1/3 ≈ 0.333
        // Wrong (union/intersection): 3/1 = 3.0
        let sim = jaccard_similarity(&a, &b);
        assert!(sim < 1.0, "Similarity should be in [0, 1]");
        assert!((sim - 0.333).abs() < 0.01, "Should be intersection/union");
    }

    #[test]
    fn test_compute_keyword_score_both_empty() {
        let empty: Vec<String> = vec![];
        let score = compute_keyword_score(&empty, &empty);
        assert_eq!(score, 0.0, "Both empty should yield 0.0");
    }

    #[test]
    fn test_compute_keyword_score_query_empty() {
        let empty: Vec<String> = vec![];
        let keywords = vec!["test".to_string()];
        let score = compute_keyword_score(&empty, &keywords);
        assert_eq!(score, 0.0, "Empty query should yield 0.0");
    }

    #[test]
    fn test_compute_keyword_score_keywords_empty() {
        let query = vec!["test".to_string()];
        let empty: Vec<String> = vec![];
        let score = compute_keyword_score(&query, &empty);
        assert_eq!(score, 0.0, "Empty keywords should yield 0.0");
    }

    #[test]
    fn test_compute_keyword_score_full_match() {
        let query = vec!["a".to_string(), "b".to_string()];
        let keywords = vec!["a".to_string(), "b".to_string()];

        let score = compute_keyword_score(&query, &keywords);
        assert_eq!(score, 1.0, "All query terms matching should yield 1.0");
    }

    #[test]
    fn test_compute_keyword_score_partial_match() {
        let query = vec!["a".to_string(), "b".to_string()];
        let keywords = vec!["a".to_string(), "c".to_string()];

        // 1 out of 2 match
        let score = compute_keyword_score(&query, &keywords);
        assert_eq!(score, 0.5, "Half matching should yield 0.5");
    }

    #[test]
    fn test_compute_keyword_score_substring_match() {
        let query = vec!["over".to_string()];
        let keywords = vec!["overflow".to_string()];

        // "over" is contained in "overflow"
        let score = compute_keyword_score(&query, &keywords);
        assert_eq!(score, 1.0, "Substring match should count");
    }

    #[test]
    fn test_compute_keyword_score_division_by_query_len() {
        let query = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let keywords = vec!["a".to_string()];

        // 1 match out of 4 query terms
        let score = compute_keyword_score(&query, &keywords);
        assert_eq!(score, 0.25, "Should divide by query length");

        // If division were * instead of /, result would be 4.0
        assert!(score <= 1.0, "Score should not exceed 1.0");
    }
}
