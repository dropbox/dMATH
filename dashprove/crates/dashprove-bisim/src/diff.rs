//! Difference types and comparison utilities for bisimulation checking

use crate::{ApiRequest, ToolCall};
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};

/// A difference found between oracle and subject traces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Difference {
    /// API request mismatch
    ApiRequestMismatch {
        /// Index in the request sequence
        index: usize,
        /// Oracle's request
        oracle: ApiRequest,
        /// Subject's request
        subject: ApiRequest,
        /// Detailed JSON diff
        diff: JsonDiff,
    },
    /// Tool call mismatch
    ToolCallMismatch {
        /// Index in the tool call sequence
        index: usize,
        /// Oracle's tool call
        oracle: ToolCall,
        /// Subject's tool call
        subject: ToolCall,
        /// Detailed JSON diff
        diff: JsonDiff,
    },
    /// Output text mismatch
    OutputMismatch {
        /// Oracle's output
        oracle: String,
        /// Subject's output
        subject: String,
        /// Similarity score (0.0-1.0)
        similarity: f64,
    },
    /// Timing violation
    TimingViolation {
        /// Oracle's time in milliseconds
        oracle_ms: u64,
        /// Subject's time in milliseconds
        subject_ms: u64,
        /// Allowed tolerance
        tolerance: f64,
    },
    /// Sequence length mismatch
    SequenceLengthMismatch {
        /// Type of sequence (api_requests, tool_calls)
        sequence_type: String,
        /// Oracle's count
        oracle_count: usize,
        /// Subject's count
        subject_count: usize,
    },
    /// Missing event in subject
    MissingEvent {
        /// Index where event was expected
        index: usize,
        /// Type of event
        event_type: String,
        /// Description of missing event
        description: String,
    },
    /// Extra event in subject
    ExtraEvent {
        /// Index of extra event
        index: usize,
        /// Type of event
        event_type: String,
        /// Description of extra event
        description: String,
    },
}

impl Difference {
    /// Get a human-readable description of the difference
    pub fn description(&self) -> String {
        match self {
            Self::ApiRequestMismatch { index, .. } => {
                format!("API request #{index} differs")
            }
            Self::ToolCallMismatch {
                index,
                oracle,
                subject,
                diff,
            } => {
                if oracle.name != subject.name {
                    format!(
                        "Tool call #{index}: expected '{}', got '{}'",
                        oracle.name, subject.name
                    )
                } else if oracle.success != subject.success {
                    format!(
                        "Tool call #{index} '{}': success status differs (oracle: {}, subject: {})",
                        oracle.name, oracle.success, subject.success
                    )
                } else if !diff.is_empty() {
                    // Summarize which fields differ
                    let diff_fields: Vec<&str> = diff
                        .differences
                        .iter()
                        .filter_map(|d| {
                            if d.path.contains("arguments") {
                                Some("arguments")
                            } else if d.path.contains("result") {
                                Some("result")
                            } else if d.path.contains("error") {
                                Some("error")
                            } else {
                                None
                            }
                        })
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    if diff_fields.is_empty() {
                        format!("Tool call #{index} '{}': values differ", oracle.name)
                    } else {
                        format!(
                            "Tool call #{index} '{}': {} differ",
                            oracle.name,
                            diff_fields.join(", ")
                        )
                    }
                } else {
                    format!("Tool call #{index} '{}': values differ", oracle.name)
                }
            }
            Self::OutputMismatch { similarity, .. } => {
                format!("Output differs (similarity: {:.1}%)", similarity * 100.0)
            }
            Self::TimingViolation {
                oracle_ms,
                subject_ms,
                tolerance,
            } => {
                let diff_pct = (((*subject_ms as f64) - (*oracle_ms as f64)).abs()
                    / (*oracle_ms as f64))
                    * 100.0;
                format!(
                    "Timing violation: {oracle_ms}ms vs {subject_ms}ms ({diff_pct:.1}% difference, tolerance: {:.1}%)",
                    tolerance * 100.0
                )
            }
            Self::SequenceLengthMismatch {
                sequence_type,
                oracle_count,
                subject_count,
            } => {
                format!(
                    "{sequence_type} count mismatch: oracle has {oracle_count}, subject has {subject_count}"
                )
            }
            Self::MissingEvent {
                index,
                event_type,
                description,
            } => {
                format!("Missing {event_type} at index {index}: {description}")
            }
            Self::ExtraEvent {
                index,
                event_type,
                description,
            } => {
                format!("Extra {event_type} at index {index}: {description}")
            }
        }
    }

    /// Get the severity of this difference (0.0-1.0, higher is more severe)
    pub fn severity(&self) -> f64 {
        match self {
            Self::ApiRequestMismatch { .. } => 1.0,
            Self::ToolCallMismatch { .. } => 0.9,
            Self::OutputMismatch { similarity, .. } => 1.0 - similarity,
            Self::TimingViolation { .. } => 0.3,
            Self::SequenceLengthMismatch { .. } => 0.8,
            Self::MissingEvent { .. } => 0.9,
            Self::ExtraEvent { .. } => 0.7,
        }
    }
}

/// JSON diff result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonDiff {
    /// Paths that differ
    pub differences: Vec<JsonPathDiff>,
}

/// A single path difference in JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonPathDiff {
    /// JSON path (e.g., `$.body.messages[0].content`)
    pub path: String,
    /// Oracle value (as JSON string)
    pub oracle_value: String,
    /// Subject value (as JSON string)
    pub subject_value: String,
}

impl JsonDiff {
    /// Create an empty diff (values are equal)
    pub fn empty() -> Self {
        Self {
            differences: vec![],
        }
    }

    /// Check if the diff is empty (values are equal)
    pub fn is_empty(&self) -> bool {
        self.differences.is_empty()
    }

    /// Compare two JSON values and return differences
    pub fn compare(oracle: &serde_json::Value, subject: &serde_json::Value) -> Self {
        let mut differences = vec![];
        Self::compare_recursive(oracle, subject, "$".to_string(), &mut differences, None);
        Self { differences }
    }

    /// Compare two JSON values with semantic tolerance options
    pub fn compare_with_config(
        oracle: &serde_json::Value,
        subject: &serde_json::Value,
        config: &crate::PayloadComparisonConfig,
    ) -> Self {
        let mut differences = vec![];
        Self::compare_recursive(
            oracle,
            subject,
            "$".to_string(),
            &mut differences,
            Some(config),
        );
        Self { differences }
    }

    fn compare_recursive(
        oracle: &serde_json::Value,
        subject: &serde_json::Value,
        path: String,
        diffs: &mut Vec<JsonPathDiff>,
        config: Option<&crate::PayloadComparisonConfig>,
    ) {
        use serde_json::Value;

        // Check if this path should be ignored
        if let Some(cfg) = config {
            if cfg.should_ignore(&path) {
                return;
            }
        }

        match (oracle, subject) {
            (Value::Object(o_map), Value::Object(s_map)) => {
                // Check all oracle keys
                for (key, o_val) in o_map {
                    let new_path = format!("{path}.{key}");
                    // Check if path should be ignored before recursing
                    if let Some(cfg) = config {
                        if cfg.should_ignore(&new_path) {
                            continue;
                        }
                    }
                    if let Some(s_val) = s_map.get(key) {
                        Self::compare_recursive(o_val, s_val, new_path, diffs, config);
                    } else {
                        diffs.push(JsonPathDiff {
                            path: new_path,
                            oracle_value: o_val.to_string(),
                            subject_value: "undefined".to_string(),
                        });
                    }
                }
                // Check for extra keys in subject
                for (key, s_val) in s_map {
                    if !o_map.contains_key(key) {
                        let new_path = format!("{path}.{key}");
                        if let Some(cfg) = config {
                            if cfg.should_ignore(&new_path) {
                                continue;
                            }
                        }
                        diffs.push(JsonPathDiff {
                            path: new_path,
                            oracle_value: "undefined".to_string(),
                            subject_value: s_val.to_string(),
                        });
                    }
                }
            }
            (Value::Array(o_arr), Value::Array(s_arr)) => {
                let max_len = o_arr.len().max(s_arr.len());
                for i in 0..max_len {
                    let new_path = format!("{path}[{i}]");
                    match (o_arr.get(i), s_arr.get(i)) {
                        (Some(o_val), Some(s_val)) => {
                            Self::compare_recursive(o_val, s_val, new_path, diffs, config);
                        }
                        (Some(o_val), None) => {
                            diffs.push(JsonPathDiff {
                                path: new_path,
                                oracle_value: o_val.to_string(),
                                subject_value: "undefined".to_string(),
                            });
                        }
                        (None, Some(s_val)) => {
                            diffs.push(JsonPathDiff {
                                path: new_path,
                                oracle_value: "undefined".to_string(),
                                subject_value: s_val.to_string(),
                            });
                        }
                        (None, None) => unreachable!(),
                    }
                }
            }
            // Handle numeric comparison with tolerance
            (Value::Number(o_num), Value::Number(s_num)) => {
                if let Some(cfg) = config {
                    if let Some(tolerance) = cfg.numeric_tolerance {
                        if let (Some(o_f), Some(s_f)) = (o_num.as_f64(), s_num.as_f64()) {
                            // Check if within tolerance (relative or absolute)
                            let diff = (o_f - s_f).abs();
                            let relative_diff = if o_f.abs() > f64::EPSILON {
                                diff / o_f.abs()
                            } else {
                                diff
                            };
                            if relative_diff <= tolerance || diff <= tolerance {
                                return; // Within tolerance, no diff
                            }
                        }
                    }
                }
                // Exact comparison
                if oracle != subject {
                    diffs.push(JsonPathDiff {
                        path,
                        oracle_value: oracle.to_string(),
                        subject_value: subject.to_string(),
                    });
                }
            }
            // Handle string comparison with similarity threshold
            (Value::String(o_str), Value::String(s_str)) => {
                if o_str == s_str {
                    return; // Exact match
                }
                if let Some(cfg) = config {
                    if let Some(threshold) = cfg.text_similarity_threshold {
                        let similarity = text_similarity(o_str, s_str);
                        if similarity >= threshold {
                            return; // Within similarity threshold, no diff
                        }
                    }
                }
                diffs.push(JsonPathDiff {
                    path,
                    oracle_value: oracle.to_string(),
                    subject_value: subject.to_string(),
                });
            }
            _ => {
                if oracle != subject {
                    diffs.push(JsonPathDiff {
                        path,
                        oracle_value: oracle.to_string(),
                        subject_value: subject.to_string(),
                    });
                }
            }
        }
    }
}

/// Calculate text similarity using character-based diff ratio
pub fn text_similarity(oracle: &str, subject: &str) -> f64 {
    if oracle == subject {
        return 1.0;
    }
    if oracle.is_empty() && subject.is_empty() {
        return 1.0;
    }
    if oracle.is_empty() || subject.is_empty() {
        return 0.0;
    }

    // Use character-based diff for accurate similarity
    let diff = TextDiff::from_chars(oracle, subject);
    let mut same_chars = 0usize;

    for change in diff.iter_all_changes() {
        if change.tag() == ChangeTag::Equal {
            same_chars += change.value().len();
        }
    }

    // Sørensen–Dice coefficient: 2 * |intersection| / (|A| + |B|)
    (2.0 * same_chars as f64) / (oracle.len() + subject.len()) as f64
}

/// Generate a unified diff string for display
pub fn unified_diff(oracle: &str, subject: &str, context_lines: usize) -> String {
    let diff = TextDiff::from_lines(oracle, subject);
    diff.unified_diff()
        .context_radius(context_lines)
        .header("oracle", "subject")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Property tests
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// text_similarity should always return values in [0.0, 1.0]
            #[test]
            fn text_similarity_bounds(a in ".*", b in ".*") {
                let sim = text_similarity(&a, &b);
                prop_assert!((0.0..=1.0).contains(&sim), "Similarity {} out of bounds", sim);
            }

            /// text_similarity should be reflexive (s, s) == 1.0
            #[test]
            fn text_similarity_reflexive(s in ".*") {
                let sim = text_similarity(&s, &s);
                prop_assert!((sim - 1.0).abs() < f64::EPSILON, "Self-similarity {} != 1.0", sim);
            }

            /// Note: text_similarity is NOT symmetric due to how diff algorithms work.
            /// This is expected behavior - the order affects how edits are calculated.
            /// We test that asymmetry is bounded (both values still in valid range).
            #[test]
            fn text_similarity_asymmetry_bounded(a in ".*", b in ".*") {
                let sim_ab = text_similarity(&a, &b);
                let sim_ba = text_similarity(&b, &a);
                // Both should still be in valid range
                prop_assert!((0.0..=1.0).contains(&sim_ab));
                prop_assert!((0.0..=1.0).contains(&sim_ba));
                // Asymmetry should not be extreme (within 0.5 of each other typically)
                // This is a soft bound based on typical diff behavior
                prop_assert!((sim_ab - sim_ba).abs() <= 0.5,
                    "Extreme asymmetry: ({}, {}) = {} vs ({}, {}) = {}", a, b, sim_ab, b, a, sim_ba);
            }

            /// JsonDiff should be reflexive - comparing a value to itself yields empty diff
            #[test]
            fn json_diff_reflexive(
                s in "[a-z]{0,10}",
                n in any::<i64>(),
                b in any::<bool>()
            ) {
                let val = serde_json::json!({
                    "string": s,
                    "number": n,
                    "bool": b
                });
                let diff = JsonDiff::compare(&val, &val);
                prop_assert!(diff.is_empty(), "Self-diff not empty: {:?}", diff.differences);
            }

            /// JsonDiff should detect primitive differences
            #[test]
            fn json_diff_detects_primitive_change(
                key in "[a-z]{1,5}",
                val1 in any::<i64>(),
                val2 in any::<i64>().prop_filter("Different values", |v| *v != 0)
            ) {
                let a = serde_json::json!({ key.clone(): val1 });
                let b = serde_json::json!({ key.clone(): val1.wrapping_add(val2) });

                if val2 != 0 {
                    let diff = JsonDiff::compare(&a, &b);
                    prop_assert!(!diff.is_empty(), "Should detect difference");
                }
            }

            /// JsonDiff with nested objects should have correct path
            #[test]
            fn json_diff_nested_path_format(
                outer_key in "[a-z]{1,5}",
                inner_key in "[a-z]{1,5}",
                val1 in any::<i32>(),
                val2 in any::<i32>().prop_filter("Different", |v| *v != 0)
            ) {
                if val2 != 0 {
                    let a = serde_json::json!({ outer_key.clone(): { inner_key.clone(): val1 } });
                    let b = serde_json::json!({ outer_key.clone(): { inner_key.clone(): val1.wrapping_add(val2) } });
                    let diff = JsonDiff::compare(&a, &b);
                    if !diff.is_empty() {
                        let path = &diff.differences[0].path;
                        prop_assert!(path.starts_with("$."), "Path should start with $. Got: {}", path);
                        prop_assert!(path.contains(&outer_key), "Path should contain outer key");
                        prop_assert!(path.contains(&inner_key), "Path should contain inner key");
                    }
                }
            }

            /// JsonDiff with arrays should include index in path
            #[test]
            fn json_diff_array_path_format(
                items1 in prop::collection::vec(any::<i32>(), 1..5),
                idx in 0usize..5
            ) {
                if idx < items1.len() {
                    let mut items2 = items1.clone();
                    items2[idx] = items2[idx].wrapping_add(1);
                    if items1[idx] != items2[idx] {
                        let a = serde_json::json!({ "arr": items1 });
                        let b = serde_json::json!({ "arr": items2 });
                        let diff = JsonDiff::compare(&a, &b);
                        if !diff.is_empty() {
                            let path = &diff.differences[0].path;
                            let idx_str = format!("[{}]", idx);
                            prop_assert!(path.contains(&idx_str),
                                "Path {} should contain index {}", path, idx_str);
                        }
                    }
                }
            }

            /// Difference severity should always be in [0.0, 1.0]
            #[test]
            fn difference_severity_bounds(similarity in 0.0f64..=1.0) {
                let diff = Difference::OutputMismatch {
                    oracle: "a".to_string(),
                    subject: "b".to_string(),
                    similarity,
                };
                let sev = diff.severity();
                prop_assert!((0.0..=1.0).contains(&sev), "Severity {} out of bounds", sev);
            }

            /// Difference description should never be empty
            #[test]
            fn difference_description_non_empty(idx in any::<usize>()) {
                let diffs = vec![
                    Difference::SequenceLengthMismatch {
                        sequence_type: "test".to_string(),
                        oracle_count: idx,
                        subject_count: idx.saturating_add(1),
                    },
                    Difference::MissingEvent {
                        index: idx,
                        event_type: "tool".to_string(),
                        description: "desc".to_string(),
                    },
                    Difference::ExtraEvent {
                        index: idx,
                        event_type: "api".to_string(),
                        description: "desc".to_string(),
                    },
                ];

                for diff in diffs {
                    prop_assert!(!diff.description().is_empty(), "Empty description");
                }
            }
        }
    }

    #[test]
    fn test_json_diff_equal() {
        let a = serde_json::json!({"name": "test", "value": 42});
        let b = serde_json::json!({"name": "test", "value": 42});

        let diff = JsonDiff::compare(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_json_diff_different() {
        let a = serde_json::json!({"name": "test", "value": 42});
        let b = serde_json::json!({"name": "test", "value": 43});

        let diff = JsonDiff::compare(&a, &b);
        assert!(!diff.is_empty());
        assert_eq!(diff.differences.len(), 1);
        assert_eq!(diff.differences[0].path, "$.value");
    }

    #[test]
    fn test_json_diff_nested() {
        let a = serde_json::json!({"outer": {"inner": {"deep": "value1"}}});
        let b = serde_json::json!({"outer": {"inner": {"deep": "value2"}}});

        let diff = JsonDiff::compare(&a, &b);
        assert_eq!(diff.differences.len(), 1);
        assert_eq!(diff.differences[0].path, "$.outer.inner.deep");
    }

    #[test]
    fn test_json_diff_array() {
        let a = serde_json::json!({"items": [1, 2, 3]});
        let b = serde_json::json!({"items": [1, 2, 4]});

        let diff = JsonDiff::compare(&a, &b);
        assert_eq!(diff.differences.len(), 1);
        assert_eq!(diff.differences[0].path, "$.items[2]");
    }

    #[test]
    fn test_json_diff_missing_key() {
        let a = serde_json::json!({"name": "test", "extra": "value"});
        let b = serde_json::json!({"name": "test"});

        let diff = JsonDiff::compare(&a, &b);
        assert_eq!(diff.differences.len(), 1);
        assert!(diff.differences[0].subject_value.contains("undefined"));
    }

    // =====================================================
    // Tests for JsonDiff::compare_with_config
    // =====================================================

    #[test]
    fn test_json_diff_ignore_path() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({"name": "test", "timestamp": 1000});
        let b = serde_json::json!({"name": "test", "timestamp": 2000});

        // Without config, should find difference
        let diff = JsonDiff::compare(&a, &b);
        assert!(!diff.is_empty(), "Should detect timestamp difference");

        // With config ignoring timestamp, should be empty
        let config = PayloadComparisonConfig::ignoring(vec!["$.timestamp".to_string()]);
        let diff_with_config = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(diff_with_config.is_empty(), "Should ignore timestamp path");
    }

    #[test]
    fn test_json_diff_ignore_nested_path() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({"body": {"data": "same", "id": "uuid-1"}});
        let b = serde_json::json!({"body": {"data": "same", "id": "uuid-2"}});

        let config = PayloadComparisonConfig::ignoring(vec!["$.body.id".to_string()]);
        let diff = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(diff.is_empty(), "Should ignore nested id path");
    }

    #[test]
    fn test_json_diff_ignore_wildcard() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({
            "headers": {"Auth": "token1", "X-Request-Id": "id1"},
            "body": "same"
        });
        let b = serde_json::json!({
            "headers": {"Auth": "token2", "X-Request-Id": "id2"},
            "body": "same"
        });

        let config = PayloadComparisonConfig::ignoring(vec!["$.headers.*".to_string()]);
        let diff = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(
            diff.is_empty(),
            "Should ignore all header paths with wildcard"
        );
    }

    #[test]
    fn test_json_diff_numeric_tolerance() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({"value": 100.0});
        let b = serde_json::json!({"value": 100.05});

        // Without tolerance, should find difference
        let diff = JsonDiff::compare(&a, &b);
        assert!(
            !diff.is_empty(),
            "Should detect numeric difference without tolerance"
        );

        // With 1% tolerance, should be acceptable (0.05/100 = 0.0005 < 0.01)
        let config = PayloadComparisonConfig::default().with_numeric_tolerance(0.01);
        let diff_with_config = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(
            diff_with_config.is_empty(),
            "Should accept within numeric tolerance"
        );
    }

    #[test]
    fn test_json_diff_numeric_tolerance_exceeded() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({"value": 100.0});
        let b = serde_json::json!({"value": 110.0}); // 10% difference

        // With 1% tolerance, should still find difference (10% > 1%)
        let config = PayloadComparisonConfig::default().with_numeric_tolerance(0.01);
        let diff = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(
            !diff.is_empty(),
            "Should detect numeric difference exceeding tolerance"
        );
    }

    #[test]
    fn test_json_diff_text_similarity_threshold() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({"message": "hello world"});
        let b = serde_json::json!({"message": "hello there"}); // Similar but not identical

        // Without threshold, should find difference
        let diff = JsonDiff::compare(&a, &b);
        assert!(
            !diff.is_empty(),
            "Should detect text difference without threshold"
        );

        // With 0.5 similarity threshold, should be acceptable (~70% similar)
        let config = PayloadComparisonConfig::default().with_text_threshold(0.5);
        let diff_with_config = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(
            diff_with_config.is_empty(),
            "Should accept within text similarity threshold"
        );
    }

    #[test]
    fn test_json_diff_text_similarity_threshold_exceeded() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({"message": "hello"});
        let b = serde_json::json!({"message": "completely different"}); // Very different

        // With 0.9 similarity threshold, should still find difference
        let config = PayloadComparisonConfig::default().with_text_threshold(0.9);
        let diff = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(
            !diff.is_empty(),
            "Should detect text difference exceeding threshold"
        );
    }

    #[test]
    fn test_json_diff_combined_config() {
        use crate::PayloadComparisonConfig;

        let a = serde_json::json!({
            "timestamp": 1000,
            "score": 95.0,
            "message": "test completed successfully",
            "important": "must match"
        });
        let b = serde_json::json!({
            "timestamp": 2000,  // Different but ignored
            "score": 95.5,       // Within 1% tolerance
            "message": "test completed with success", // Similar enough (~80% similar)
            "important": "must match"
        });

        let config = PayloadComparisonConfig::ignoring(vec!["$.timestamp".to_string()])
            .with_numeric_tolerance(0.01)
            .with_text_threshold(0.6);

        let diff = JsonDiff::compare_with_config(&a, &b, &config);
        assert!(
            diff.is_empty(),
            "Combined config should accept all differences"
        );
    }

    #[test]
    fn test_text_similarity_identical() {
        assert_eq!(text_similarity("hello world", "hello world"), 1.0);
    }

    #[test]
    fn test_text_similarity_different() {
        let sim = text_similarity("hello world", "hello there");
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_text_similarity_empty() {
        assert_eq!(text_similarity("", ""), 1.0);
        assert_eq!(text_similarity("hello", ""), 0.0);
        assert_eq!(text_similarity("", "hello"), 0.0);
    }

    #[test]
    fn test_difference_description() {
        let diff = Difference::OutputMismatch {
            oracle: "hello".to_string(),
            subject: "world".to_string(),
            similarity: 0.5,
        };
        assert!(diff.description().contains("50.0%"));

        let diff = Difference::SequenceLengthMismatch {
            sequence_type: "tool_calls".to_string(),
            oracle_count: 5,
            subject_count: 3,
        };
        assert!(diff.description().contains("5"));
        assert!(diff.description().contains("3"));
    }

    #[test]
    fn test_unified_diff() {
        let oracle = "line1\nline2\nline3";
        let subject = "line1\nmodified\nline3";

        let diff = unified_diff(oracle, subject, 1);
        assert!(diff.contains("line2"));
        assert!(diff.contains("modified"));
    }

    // =====================================================
    // Mutation-killing tests for Difference::description
    // =====================================================

    /// Test ToolCallMismatch description distinguishes name mismatch vs arg mismatch
    /// (catches mutation: replace != with == in diff.rs:89)
    #[test]
    fn test_tool_call_mismatch_name_different() {
        use crate::ToolCall;
        let oracle = ToolCall::new("read_file");
        let subject = ToolCall::new("write_file");
        let diff = Difference::ToolCallMismatch {
            index: 0,
            oracle: oracle.clone(),
            subject: subject.clone(),
            diff: JsonDiff::empty(), // Names differ, diff content doesn't matter
        };
        let desc = diff.description();
        // When names are different, description should include both names
        assert!(desc.contains("read_file"), "Should mention oracle name");
        assert!(desc.contains("write_file"), "Should mention subject name");
    }

    /// Test ToolCallMismatch description when names are same but args differ
    #[test]
    fn test_tool_call_mismatch_same_name_args_differ() {
        use crate::ToolCall;
        let oracle =
            ToolCall::new("read_file").with_arguments(serde_json::json!({"path": "/oracle"}));
        let subject =
            ToolCall::new("read_file").with_arguments(serde_json::json!({"path": "/subject"}));
        // Create a diff that shows arguments differing
        let json_diff = JsonDiff {
            differences: vec![JsonPathDiff {
                path: "$.arguments.path".to_string(),
                oracle_value: "\"/oracle\"".to_string(),
                subject_value: "\"/subject\"".to_string(),
            }],
        };
        let diff = Difference::ToolCallMismatch {
            index: 0,
            oracle: oracle.clone(),
            subject: subject.clone(),
            diff: json_diff,
        };
        let desc = diff.description();
        // When names are same but args differ, should mention arguments
        assert!(
            desc.contains("arguments"),
            "Should indicate arguments differ: {}",
            desc
        );
    }

    /// Test ToolCallMismatch description when success status differs
    #[test]
    fn test_tool_call_mismatch_success_differs() {
        use crate::ToolCall;
        let oracle = ToolCall::new("read_file");
        let subject = ToolCall::new("read_file").with_error("file not found");
        let diff = Difference::ToolCallMismatch {
            index: 0,
            oracle: oracle.clone(),
            subject: subject.clone(),
            diff: JsonDiff::empty(), // success differs, will be caught first
        };
        let desc = diff.description();
        // When success status differs, should mention it
        assert!(
            desc.contains("success status differs"),
            "Should indicate success differs: {}",
            desc
        );
    }

    /// Test ToolCallMismatch description when result differs
    #[test]
    fn test_tool_call_mismatch_result_differs() {
        use crate::ToolCall;
        let oracle =
            ToolCall::new("read_file").with_result(serde_json::json!({"content": "oracle data"}));
        let subject =
            ToolCall::new("read_file").with_result(serde_json::json!({"content": "subject data"}));
        let json_diff = JsonDiff {
            differences: vec![JsonPathDiff {
                path: "$.result.content".to_string(),
                oracle_value: "\"oracle data\"".to_string(),
                subject_value: "\"subject data\"".to_string(),
            }],
        };
        let diff = Difference::ToolCallMismatch {
            index: 0,
            oracle: oracle.clone(),
            subject: subject.clone(),
            diff: json_diff,
        };
        let desc = diff.description();
        // When result differs, should mention result
        assert!(
            desc.contains("result"),
            "Should indicate result differs: {}",
            desc
        );
    }

    /// Test TimingViolation description arithmetic precision
    /// (catches mutations: replace arithmetic operators in lines 109-111)
    #[test]
    fn test_timing_violation_description_arithmetic() {
        // oracle_ms = 100, subject_ms = 200
        // diff_pct should be |200-100| / 100 * 100 = 100%
        let diff = Difference::TimingViolation {
            oracle_ms: 100,
            subject_ms: 200,
            tolerance: 0.1,
        };
        let desc = diff.description();
        // Should contain "100.0% difference"
        assert!(
            desc.contains("100.0%"),
            "Expected 100.0% difference, got: {}",
            desc
        );
        // Should contain tolerance as percentage (10.0%)
        assert!(
            desc.contains("10.0%"),
            "Expected tolerance 10.0%, got: {}",
            desc
        );
    }

    /// Test TimingViolation with different values to verify subtraction/division
    #[test]
    fn test_timing_violation_calculation_correctness() {
        // oracle_ms = 200, subject_ms = 250
        // diff_pct should be |250-200| / 200 * 100 = 25%
        let diff = Difference::TimingViolation {
            oracle_ms: 200,
            subject_ms: 250,
            tolerance: 0.5,
        };
        let desc = diff.description();
        assert!(
            desc.contains("25.0%"),
            "Expected 25.0% difference, got: {}",
            desc
        );
    }

    // =====================================================
    // Mutation-killing tests for Difference::severity
    // =====================================================

    /// Test severity returns correct values for different variants
    /// (catches mutation: replace severity -> f64 with 0.0 or 1.0)
    #[test]
    fn test_severity_values_by_variant() {
        // ApiRequestMismatch should be 1.0
        let api_diff = Difference::ApiRequestMismatch {
            index: 0,
            oracle: crate::ApiRequest::new("GET", "http://test"),
            subject: crate::ApiRequest::new("GET", "http://test2"),
            diff: crate::JsonDiff::empty(),
        };
        assert_eq!(api_diff.severity(), 1.0);

        // TimingViolation should be 0.3
        let timing_diff = Difference::TimingViolation {
            oracle_ms: 100,
            subject_ms: 200,
            tolerance: 0.1,
        };
        assert!((timing_diff.severity() - 0.3).abs() < f64::EPSILON);

        // OutputMismatch severity should be 1.0 - similarity
        let output_diff = Difference::OutputMismatch {
            oracle: "a".to_string(),
            subject: "b".to_string(),
            similarity: 0.7,
        };
        assert!((output_diff.severity() - 0.3).abs() < f64::EPSILON);

        // ExtraEvent should be 0.7
        let extra_diff = Difference::ExtraEvent {
            index: 0,
            event_type: "test".to_string(),
            description: "test".to_string(),
        };
        assert!((extra_diff.severity() - 0.7).abs() < f64::EPSILON);
    }

    // =====================================================
    // Mutation-killing tests for text_similarity
    // =====================================================

    /// Test text_similarity handles empty string edge cases
    /// (catches mutation: replace || with && in line 276)
    #[test]
    fn test_text_similarity_one_empty() {
        // If only one is empty, should return 0.0
        // This catches the || -> && mutation because:
        // - With ||: "hello".is_empty() || "".is_empty() = false || true = true -> return 0.0
        // - With &&: "hello".is_empty() && "".is_empty() = false && true = false -> falls through
        let sim1 = text_similarity("hello", "");
        assert!(
            (sim1 - 0.0).abs() < f64::EPSILON,
            "Non-empty vs empty should be exactly 0.0, got {}",
            sim1
        );

        let sim2 = text_similarity("", "world");
        assert!(
            (sim2 - 0.0).abs() < f64::EPSILON,
            "Empty vs non-empty should be exactly 0.0, got {}",
            sim2
        );
    }

    /// Test text_similarity arithmetic formula
    /// (catches mutations: replace * with + in line 291, replace + with * in line 291)
    #[test]
    fn test_text_similarity_formula_correctness() {
        // "ab" vs "ab" -> same_chars = 2, total = 2+2 = 4
        // Sørensen–Dice: 2 * 2 / 4 = 1.0
        assert_eq!(text_similarity("ab", "ab"), 1.0);

        // "abc" vs "abd" -> same_chars = 2 (a,b), total = 3+3 = 6
        // Sørensen–Dice: 2 * 2 / 6 = 0.666...
        let sim = text_similarity("abc", "abd");
        // With character diff, 'a' and 'b' match, 'c' vs 'd' differ
        // Similarity should be between 0 and 1
        assert!(
            sim > 0.5 && sim < 1.0,
            "Expected similarity ~0.67, got {}",
            sim
        );
    }

    /// Test that text_similarity produces non-trivial values
    #[test]
    fn test_text_similarity_partial_match() {
        // "hello" vs "hallo" - most chars match
        let sim = text_similarity("hello", "hallo");
        assert!(sim > 0.7, "Expected high similarity, got {}", sim);
        assert!(sim < 1.0, "Should not be perfect match");
    }
}

// Kani formal verification proofs
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify text_similarity returns exactly 1.0 for identical strings
    #[kani::proof]
    fn verify_text_similarity_identical_is_one() {
        let s = "test string";
        let sim = text_similarity(s, s);
        kani::assert(sim == 1.0, "Identical strings must have similarity 1.0");
    }

    /// Verify text_similarity returns 1.0 for both strings empty
    #[kani::proof]
    fn verify_text_similarity_both_empty_is_one() {
        let sim = text_similarity("", "");
        kani::assert(sim == 1.0, "Both empty strings must have similarity 1.0");
    }

    /// Verify text_similarity returns 0.0 when one string is empty
    #[kani::proof]
    fn verify_text_similarity_one_empty_is_zero() {
        let sim1 = text_similarity("hello", "");
        kani::assert(sim1 == 0.0, "Non-empty vs empty must be 0.0");

        let sim2 = text_similarity("", "world");
        kani::assert(sim2 == 0.0, "Empty vs non-empty must be 0.0");
    }

    /// Verify JsonDiff::empty() creates empty diff
    #[kani::proof]
    fn verify_json_diff_empty_is_empty() {
        let diff = JsonDiff::empty();
        kani::assert(diff.is_empty(), "JsonDiff::empty() must return empty diff");
        kani::assert(
            diff.differences.len() == 0,
            "Empty diff must have no differences",
        );
    }

    /// Verify JsonDiff::is_empty equals differences being empty
    #[kani::proof]
    fn verify_json_diff_is_empty_consistency() {
        let empty_diff = JsonDiff {
            differences: vec![],
        };
        kani::assert(
            empty_diff.is_empty(),
            "Empty differences means is_empty() true",
        );

        let non_empty_diff = JsonDiff {
            differences: vec![JsonPathDiff {
                path: "$".to_string(),
                oracle_value: "1".to_string(),
                subject_value: "2".to_string(),
            }],
        };
        kani::assert(
            !non_empty_diff.is_empty(),
            "Non-empty differences means is_empty() false",
        );
    }

    /// Verify Difference::severity() returns values in [0.0, 1.0]
    #[kani::proof]
    fn verify_severity_api_request_mismatch() {
        let diff = Difference::ApiRequestMismatch {
            index: 0,
            oracle: ApiRequest::new("GET", "http://a"),
            subject: ApiRequest::new("GET", "http://b"),
            diff: JsonDiff::empty(),
        };
        let sev = diff.severity();
        kani::assert(sev >= 0.0 && sev <= 1.0, "Severity must be in [0.0, 1.0]");
        kani::assert(sev == 1.0, "ApiRequestMismatch severity must be 1.0");
    }

    /// Verify Difference::severity() for TimingViolation
    #[kani::proof]
    fn verify_severity_timing_violation() {
        let diff = Difference::TimingViolation {
            oracle_ms: 100,
            subject_ms: 200,
            tolerance: 0.1,
        };
        let sev = diff.severity();
        kani::assert(sev >= 0.0 && sev <= 1.0, "Severity must be in [0.0, 1.0]");
        kani::assert(
            (sev - 0.3).abs() < 0.001,
            "TimingViolation severity must be 0.3",
        );
    }

    /// Verify Difference::severity() for OutputMismatch
    #[kani::proof]
    fn verify_severity_output_mismatch() {
        // Test with similarity 0.0 -> severity should be 1.0
        let diff_zero = Difference::OutputMismatch {
            oracle: "a".to_string(),
            subject: "b".to_string(),
            similarity: 0.0,
        };
        kani::assert(
            diff_zero.severity() == 1.0,
            "similarity=0.0 -> severity=1.0",
        );

        // Test with similarity 1.0 -> severity should be 0.0
        let diff_one = Difference::OutputMismatch {
            oracle: "a".to_string(),
            subject: "a".to_string(),
            similarity: 1.0,
        };
        kani::assert(diff_one.severity() == 0.0, "similarity=1.0 -> severity=0.0");

        // Test with similarity 0.5 -> severity should be 0.5
        let diff_half = Difference::OutputMismatch {
            oracle: "a".to_string(),
            subject: "b".to_string(),
            similarity: 0.5,
        };
        kani::assert(
            (diff_half.severity() - 0.5).abs() < 0.001,
            "similarity=0.5 -> severity=0.5",
        );
    }

    /// Verify Difference::severity() for all other variants
    #[kani::proof]
    fn verify_severity_other_variants() {
        let tool_diff = Difference::ToolCallMismatch {
            index: 0,
            oracle: ToolCall::new("a"),
            subject: ToolCall::new("b"),
            diff: JsonDiff::empty(),
        };
        kani::assert(
            (tool_diff.severity() - 0.9).abs() < 0.001,
            "ToolCallMismatch severity=0.9",
        );

        let seq_diff = Difference::SequenceLengthMismatch {
            sequence_type: "test".to_string(),
            oracle_count: 1,
            subject_count: 2,
        };
        kani::assert(
            (seq_diff.severity() - 0.8).abs() < 0.001,
            "SequenceLengthMismatch severity=0.8",
        );

        let missing_diff = Difference::MissingEvent {
            index: 0,
            event_type: "test".to_string(),
            description: "desc".to_string(),
        };
        kani::assert(
            (missing_diff.severity() - 0.9).abs() < 0.001,
            "MissingEvent severity=0.9",
        );

        let extra_diff = Difference::ExtraEvent {
            index: 0,
            event_type: "test".to_string(),
            description: "desc".to_string(),
        };
        kani::assert(
            (extra_diff.severity() - 0.7).abs() < 0.001,
            "ExtraEvent severity=0.7",
        );
    }

    /// Verify Difference::description() is never empty
    #[kani::proof]
    fn verify_description_non_empty() {
        let diff = Difference::OutputMismatch {
            oracle: "a".to_string(),
            subject: "b".to_string(),
            similarity: 0.5,
        };
        kani::assert(
            !diff.description().is_empty(),
            "Description must not be empty",
        );
    }
}
