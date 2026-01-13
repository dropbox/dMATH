//! Configuration types for bisimulation checking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration for bisimulation checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisimulationConfig {
    /// Oracle (reference) implementation configuration
    pub oracle: OracleConfig,
    /// Subject (test) implementation configuration
    pub subject: TestSubjectConfig,
    /// What aspects must be equivalent
    pub equivalence_criteria: EquivalenceCriteria,
    /// How to handle non-determinism
    pub nondeterminism_strategy: NondeterminismStrategy,
}

impl Default for BisimulationConfig {
    fn default() -> Self {
        Self {
            oracle: OracleConfig::RecordedTraces {
                trace_dir: PathBuf::from("traces"),
            },
            subject: TestSubjectConfig::Binary {
                path: PathBuf::new(),
                args: vec![],
                env: HashMap::new(),
            },
            equivalence_criteria: EquivalenceCriteria::default(),
            nondeterminism_strategy: NondeterminismStrategy::ExactMatch,
        }
    }
}

impl BisimulationConfig {
    /// Create a config for comparing two binaries
    pub fn binary_comparison(oracle_path: PathBuf, subject_path: PathBuf) -> Self {
        Self {
            oracle: OracleConfig::Binary {
                path: oracle_path,
                args: vec![],
                env: HashMap::new(),
            },
            subject: TestSubjectConfig::Binary {
                path: subject_path,
                args: vec![],
                env: HashMap::new(),
            },
            equivalence_criteria: EquivalenceCriteria::default(),
            nondeterminism_strategy: NondeterminismStrategy::ExactMatch,
        }
    }

    /// Create a config for comparing against recorded traces
    pub fn trace_comparison(trace_dir: PathBuf, subject_path: PathBuf) -> Self {
        Self {
            oracle: OracleConfig::RecordedTraces { trace_dir },
            subject: TestSubjectConfig::Binary {
                path: subject_path,
                args: vec![],
                env: HashMap::new(),
            },
            equivalence_criteria: EquivalenceCriteria::default(),
            nondeterminism_strategy: NondeterminismStrategy::ExactMatch,
        }
    }

    /// Set equivalence criteria
    pub fn with_criteria(mut self, criteria: EquivalenceCriteria) -> Self {
        self.equivalence_criteria = criteria;
        self
    }

    /// Set nondeterminism strategy
    pub fn with_strategy(mut self, strategy: NondeterminismStrategy) -> Self {
        self.nondeterminism_strategy = strategy;
        self
    }
}

/// Oracle configuration (reference implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OracleConfig {
    /// Run a binary and capture I/O
    Binary {
        path: PathBuf,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// Use pre-recorded traces
    RecordedTraces { trace_dir: PathBuf },
}

impl OracleConfig {
    /// Create a binary oracle config
    pub fn binary(path: impl Into<PathBuf>) -> Self {
        Self::Binary {
            path: path.into(),
            args: vec![],
            env: HashMap::new(),
        }
    }

    /// Create a recorded traces oracle config
    pub fn traces(dir: impl Into<PathBuf>) -> Self {
        Self::RecordedTraces {
            trace_dir: dir.into(),
        }
    }

    /// Add arguments (only for Binary)
    pub fn with_args(self, args: Vec<String>) -> Self {
        match self {
            Self::Binary { path, env, .. } => Self::Binary { path, args, env },
            other => other,
        }
    }

    /// Add environment variables (only for Binary)
    pub fn with_env(self, env: HashMap<String, String>) -> Self {
        match self {
            Self::Binary { path, args, .. } => Self::Binary { path, args, env },
            other => other,
        }
    }
}

/// Test subject configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestSubjectConfig {
    /// Run a binary and capture I/O
    Binary {
        path: PathBuf,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// In-process subject (not serializable)
    #[serde(skip)]
    InProcess,
}

impl TestSubjectConfig {
    /// Create a binary subject config
    pub fn binary(path: impl Into<PathBuf>) -> Self {
        Self::Binary {
            path: path.into(),
            args: vec![],
            env: HashMap::new(),
        }
    }

    /// Add arguments (only for Binary)
    pub fn with_args(self, args: Vec<String>) -> Self {
        match self {
            Self::Binary { path, env, .. } => Self::Binary { path, args, env },
            other => other,
        }
    }

    /// Add environment variables (only for Binary)
    pub fn with_env(self, env: HashMap<String, String>) -> Self {
        match self {
            Self::Binary { path, args, .. } => Self::Binary { path, args, env },
            other => other,
        }
    }
}

/// What aspects must be equivalent between oracle and subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceCriteria {
    /// Compare API requests
    pub api_requests: bool,
    /// Compare tool calls
    pub tool_calls: bool,
    /// Compare final output
    pub output: bool,
    /// Timing tolerance as a percentage (e.g., 0.1 for 10%)
    pub timing_tolerance: Option<f64>,
    /// Use semantic comparison instead of exact match
    pub semantic_comparison: bool,
    /// Configuration for payload comparison (API bodies, tool arguments/results)
    #[serde(default)]
    pub payload_config: PayloadComparisonConfig,
}

/// Configuration for comparing JSON payloads with tolerance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PayloadComparisonConfig {
    /// JSON paths to ignore during comparison (e.g., `$.timestamp`, `$.request_id`)
    #[serde(default)]
    pub ignored_paths: Vec<String>,
    /// Tolerance for numeric comparisons (e.g., 0.001 for 0.1% tolerance)
    #[serde(default)]
    pub numeric_tolerance: Option<f64>,
    /// Minimum text similarity threshold for string fields (0.0-1.0)
    #[serde(default)]
    pub text_similarity_threshold: Option<f64>,
}

impl PayloadComparisonConfig {
    /// Create a config that ignores specific paths
    pub fn ignoring(paths: Vec<String>) -> Self {
        Self {
            ignored_paths: paths,
            numeric_tolerance: None,
            text_similarity_threshold: None,
        }
    }

    /// Set numeric tolerance
    pub fn with_numeric_tolerance(mut self, tolerance: f64) -> Self {
        self.numeric_tolerance = Some(tolerance);
        self
    }

    /// Set text similarity threshold
    pub fn with_text_threshold(mut self, threshold: f64) -> Self {
        self.text_similarity_threshold = Some(threshold);
        self
    }

    /// Check if a JSON path should be ignored
    ///
    /// Supports the following wildcard patterns:
    /// - `$.foo.*` - matches any direct child of foo (e.g., `$.foo.bar`, `$.foo.baz`)
    /// - `$.**.name` - matches `name` at any depth (e.g., `$.name`, `$.foo.bar.name`)
    /// - `$.**` - matches any path
    /// - `$.items[*].id` - matches id in any array element (e.g., `$.items[0].id`, `$.items[99].id`)
    /// - `$.data[*]` - matches any array element (e.g., `$.data[0]`, `$.data[42]`)
    /// - Exact match when no wildcards are present
    pub fn should_ignore(&self, path: &str) -> bool {
        self.ignored_paths
            .iter()
            .any(|pattern| Self::pattern_matches(pattern, path))
    }

    /// Check if a pattern matches a path
    fn pattern_matches(pattern: &str, path: &str) -> bool {
        // Exact match
        if pattern == path {
            return true;
        }

        // Handle ** (deep recursive matching)
        if pattern.contains("**") {
            return Self::matches_double_star(pattern, path);
        }

        // Handle [*] (array index wildcard)
        if pattern.contains("[*]") {
            return Self::matches_array_wildcard(pattern, path);
        }

        // Handle .* suffix (single level wildcard) - backward compatibility
        if let Some(prefix) = pattern.strip_suffix(".*") {
            return path.starts_with(prefix)
                && path[prefix.len()..].starts_with('.')
                && !path[prefix.len() + 1..].contains('.');
        }

        false
    }

    /// Match patterns containing ** for deep recursive matching
    fn matches_double_star(pattern: &str, path: &str) -> bool {
        // Pattern "$.**" matches everything
        if pattern == "$.**" {
            return path.starts_with('$');
        }

        // Pattern "$.**.suffix" matches any path ending with suffix
        if let Some(suffix) = pattern.strip_prefix("$.**.") {
            // Match if path ends with .suffix or is exactly $.suffix
            return path.ends_with(&format!(".{}", suffix)) || path == format!("$.{}", suffix);
        }

        // Pattern "prefix.**" matches any path starting with prefix
        if let Some(prefix) = pattern.strip_suffix(".**") {
            return path == prefix || path.starts_with(&format!("{}.", prefix));
        }

        // Pattern "prefix.**.suffix" matches prefix, any middle, then suffix
        if let Some((prefix, rest)) = pattern.split_once(".**.") {
            // Path must start with prefix and end with suffix
            let suffix = rest;
            if !path.starts_with(prefix) {
                return false;
            }
            let after_prefix = &path[prefix.len()..];
            // Must have at least one segment then the suffix
            after_prefix.ends_with(&format!(".{}", suffix))
                || after_prefix == format!(".{}", suffix)
        } else {
            false
        }
    }

    /// Match patterns containing [*] for array index wildcards
    fn matches_array_wildcard(pattern: &str, path: &str) -> bool {
        // Split pattern and path into segments at [*] boundaries
        let pattern_parts: Vec<&str> = pattern.split("[*]").collect();
        if pattern_parts.is_empty() {
            return false;
        }

        // Build a regex-like check: each [*] in pattern should match [\d+] in path
        let mut remaining_path = path;

        for (i, pattern_part) in pattern_parts.iter().enumerate() {
            if !remaining_path.starts_with(pattern_part) {
                return false;
            }
            remaining_path = &remaining_path[pattern_part.len()..];

            // After each pattern part (except the last), expect [\d+]
            if i < pattern_parts.len() - 1 {
                // Must start with [
                if !remaining_path.starts_with('[') {
                    return false;
                }
                // Find closing ]
                if let Some(close_bracket) = remaining_path.find(']') {
                    let index_part = &remaining_path[1..close_bracket];
                    // Must be a valid number
                    if index_part.parse::<usize>().is_err() {
                        return false;
                    }
                    remaining_path = &remaining_path[close_bracket + 1..];
                } else {
                    return false;
                }
            }
        }

        remaining_path.is_empty()
    }
}

impl Default for EquivalenceCriteria {
    fn default() -> Self {
        Self {
            api_requests: true,
            tool_calls: true,
            output: true,
            timing_tolerance: None,
            semantic_comparison: false,
            payload_config: PayloadComparisonConfig::default(),
        }
    }
}

impl EquivalenceCriteria {
    /// Create criteria for API-only comparison
    pub fn api_only() -> Self {
        Self {
            api_requests: true,
            tool_calls: false,
            output: false,
            timing_tolerance: None,
            semantic_comparison: false,
            payload_config: PayloadComparisonConfig::default(),
        }
    }

    /// Create criteria for output-only comparison
    pub fn output_only() -> Self {
        Self {
            api_requests: false,
            tool_calls: false,
            output: true,
            timing_tolerance: None,
            semantic_comparison: false,
            payload_config: PayloadComparisonConfig::default(),
        }
    }

    /// Enable semantic comparison
    pub fn with_semantic(mut self) -> Self {
        self.semantic_comparison = true;
        self
    }

    /// Set timing tolerance
    pub fn with_timing_tolerance(mut self, tolerance: f64) -> Self {
        self.timing_tolerance = Some(tolerance);
        self
    }

    /// Set payload comparison configuration
    pub fn with_payload_config(mut self, config: PayloadComparisonConfig) -> Self {
        self.payload_config = config;
        self
    }
}

/// How to handle non-determinism in traces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NondeterminismStrategy {
    /// Require exact match
    ExactMatch,
    /// Use semantic similarity with a threshold
    SemanticSimilarity { threshold: f64 },
    /// Match distribution over multiple samples
    DistributionMatch { samples: usize },
}

impl NondeterminismStrategy {
    /// Create a semantic similarity strategy
    pub fn semantic(threshold: f64) -> Self {
        Self::SemanticSimilarity { threshold }
    }

    /// Create a distribution match strategy
    pub fn distribution(samples: usize) -> Self {
        Self::DistributionMatch { samples }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Property tests
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// EquivalenceCriteria::api_only should have api_requests=true and others=false
            #[test]
            fn api_only_has_correct_flags(_dummy in any::<bool>()) {
                let criteria = EquivalenceCriteria::api_only();
                prop_assert!(criteria.api_requests);
                prop_assert!(!criteria.tool_calls);
                prop_assert!(!criteria.output);
            }

            /// EquivalenceCriteria::output_only should have output=true and others=false
            #[test]
            fn output_only_has_correct_flags(_dummy in any::<bool>()) {
                let criteria = EquivalenceCriteria::output_only();
                prop_assert!(!criteria.api_requests);
                prop_assert!(!criteria.tool_calls);
                prop_assert!(criteria.output);
            }

            /// EquivalenceCriteria::default should have api_requests, tool_calls, and output=true
            #[test]
            fn default_criteria_has_all_enabled(_dummy in any::<bool>()) {
                let criteria = EquivalenceCriteria::default();
                prop_assert!(criteria.api_requests);
                prop_assert!(criteria.tool_calls);
                prop_assert!(criteria.output);
            }

            /// EquivalenceCriteria with_timing_tolerance should set tolerance
            #[test]
            fn criteria_with_timing_tolerance(tolerance in 0.0f64..1.0) {
                let criteria = EquivalenceCriteria::default()
                    .with_timing_tolerance(tolerance);
                prop_assert_eq!(criteria.timing_tolerance, Some(tolerance));
            }

            /// EquivalenceCriteria with_semantic should enable semantic comparison
            #[test]
            fn criteria_with_semantic(_dummy in any::<bool>()) {
                let criteria = EquivalenceCriteria::default().with_semantic();
                prop_assert!(criteria.semantic_comparison);
            }

            /// NondeterminismStrategy::semantic should set threshold
            #[test]
            fn strategy_semantic_threshold(threshold in 0.0f64..=1.0) {
                let strategy = NondeterminismStrategy::semantic(threshold);
                match strategy {
                    NondeterminismStrategy::SemanticSimilarity { threshold: t } => {
                        prop_assert_eq!(t, threshold);
                    }
                    _ => prop_assert!(false, "Wrong variant"),
                }
            }

            /// NondeterminismStrategy::distribution should set samples
            #[test]
            fn strategy_distribution_samples(samples in 1usize..1000) {
                let strategy = NondeterminismStrategy::distribution(samples);
                match strategy {
                    NondeterminismStrategy::DistributionMatch { samples: s } => {
                        prop_assert_eq!(s, samples);
                    }
                    _ => prop_assert!(false, "Wrong variant"),
                }
            }

            /// OracleConfig::binary should preserve path
            #[test]
            fn oracle_binary_preserves_path(path in "[a-z/]+") {
                let config = OracleConfig::binary(&path);
                match config {
                    OracleConfig::Binary { path: p, .. } => {
                        prop_assert_eq!(p, PathBuf::from(&path));
                    }
                    _ => prop_assert!(false, "Wrong variant"),
                }
            }

            /// OracleConfig::traces should preserve directory
            #[test]
            fn oracle_traces_preserves_dir(dir in "[a-z/]+") {
                let config = OracleConfig::traces(&dir);
                match config {
                    OracleConfig::RecordedTraces { trace_dir } => {
                        prop_assert_eq!(trace_dir, PathBuf::from(&dir));
                    }
                    _ => prop_assert!(false, "Wrong variant"),
                }
            }

            /// OracleConfig with_args should set args for Binary variant
            #[test]
            fn oracle_with_args(args in prop::collection::vec("[a-z]+", 0..5)) {
                let config = OracleConfig::binary("/bin/test")
                    .with_args(args.clone());
                match config {
                    OracleConfig::Binary { args: a, .. } => {
                        prop_assert_eq!(a, args);
                    }
                    _ => prop_assert!(false, "Wrong variant"),
                }
            }

            /// OracleConfig with_args on RecordedTraces should be no-op
            #[test]
            fn oracle_traces_with_args_noop(dir in "[a-z]+") {
                let config = OracleConfig::traces(&dir)
                    .with_args(vec!["--flag".to_string()]);
                match config {
                    OracleConfig::RecordedTraces { trace_dir } => {
                        prop_assert_eq!(trace_dir, PathBuf::from(&dir));
                    }
                    _ => prop_assert!(false, "Should remain RecordedTraces"),
                }
            }

            /// BisimulationConfig::binary_comparison should create correct oracle/subject
            #[test]
            fn binary_comparison_config(
                oracle_path in "[a-z/]+",
                subject_path in "[a-z/]+"
            ) {
                let config = BisimulationConfig::binary_comparison(
                    PathBuf::from(&oracle_path),
                    PathBuf::from(&subject_path),
                );
                match (&config.oracle, &config.subject) {
                    (
                        OracleConfig::Binary { path: op, .. },
                        TestSubjectConfig::Binary { path: sp, .. }
                    ) => {
                        prop_assert_eq!(op, &PathBuf::from(&oracle_path));
                        prop_assert_eq!(sp, &PathBuf::from(&subject_path));
                    }
                    _ => prop_assert!(false, "Wrong variants"),
                }
            }

            /// BisimulationConfig with_criteria should preserve criteria
            #[test]
            fn config_with_criteria(tolerance in 0.0f64..1.0) {
                let criteria = EquivalenceCriteria::default()
                    .with_timing_tolerance(tolerance);
                let config = BisimulationConfig::default()
                    .with_criteria(criteria.clone());
                prop_assert_eq!(
                    config.equivalence_criteria.timing_tolerance,
                    Some(tolerance)
                );
            }

            /// BisimulationConfig with_strategy should preserve strategy
            #[test]
            fn config_with_strategy(threshold in 0.0f64..=1.0) {
                let strategy = NondeterminismStrategy::semantic(threshold);
                let config = BisimulationConfig::default()
                    .with_strategy(strategy);
                match config.nondeterminism_strategy {
                    NondeterminismStrategy::SemanticSimilarity { threshold: t } => {
                        prop_assert_eq!(t, threshold);
                    }
                    _ => prop_assert!(false, "Wrong variant"),
                }
            }
        }
    }

    #[test]
    fn test_config_binary_comparison() {
        let config = BisimulationConfig::binary_comparison(
            PathBuf::from("/usr/bin/oracle"),
            PathBuf::from("/usr/bin/subject"),
        );

        match config.oracle {
            OracleConfig::Binary { ref path, .. } => {
                assert_eq!(path, &PathBuf::from("/usr/bin/oracle"));
            }
            _ => panic!("Expected Binary oracle"),
        }

        match config.subject {
            TestSubjectConfig::Binary { ref path, .. } => {
                assert_eq!(path, &PathBuf::from("/usr/bin/subject"));
            }
            _ => panic!("Expected Binary subject"),
        }
    }

    #[test]
    fn test_config_serialization() {
        let config = BisimulationConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: BisimulationConfig = serde_json::from_str(&json).unwrap();

        assert!(parsed.equivalence_criteria.api_requests);
        assert!(parsed.equivalence_criteria.tool_calls);
    }

    #[test]
    fn test_equivalence_criteria_presets() {
        let api = EquivalenceCriteria::api_only();
        assert!(api.api_requests);
        assert!(!api.tool_calls);
        assert!(!api.output);

        let output = EquivalenceCriteria::output_only();
        assert!(!output.api_requests);
        assert!(output.output);
    }

    #[test]
    fn test_nondeterminism_strategies() {
        let semantic = NondeterminismStrategy::semantic(0.9);
        match semantic {
            NondeterminismStrategy::SemanticSimilarity { threshold } => {
                assert_eq!(threshold, 0.9);
            }
            _ => panic!("Expected SemanticSimilarity"),
        }

        let dist = NondeterminismStrategy::distribution(100);
        match dist {
            NondeterminismStrategy::DistributionMatch { samples } => {
                assert_eq!(samples, 100);
            }
            _ => panic!("Expected DistributionMatch"),
        }
    }

    /// Test that trace_comparison produces correct non-default values
    /// (catches mutation: replace trace_comparison -> Self with Default::default())
    #[test]
    fn test_trace_comparison_not_default() {
        let trace_dir = PathBuf::from("/custom/traces");
        let subject_path = PathBuf::from("/custom/subject");
        let config = BisimulationConfig::trace_comparison(trace_dir.clone(), subject_path.clone());

        // The oracle should be RecordedTraces with the specific trace_dir
        match &config.oracle {
            OracleConfig::RecordedTraces {
                trace_dir: actual_dir,
            } => {
                assert_eq!(actual_dir, &trace_dir, "trace_dir should match");
            }
            _ => panic!("Expected RecordedTraces oracle"),
        }

        // The subject should be Binary with the specific path
        match &config.subject {
            TestSubjectConfig::Binary { path, .. } => {
                assert_eq!(path, &subject_path, "subject path should match");
            }
            _ => panic!("Expected Binary subject"),
        }

        // Verify these are NOT the default values
        let default = BisimulationConfig::default();
        if let OracleConfig::RecordedTraces {
            trace_dir: default_dir,
        } = &default.oracle
        {
            assert_ne!(default_dir, &trace_dir, "Should differ from default");
        }
        // If default is a different variant, that's also fine
    }

    // =====================================================
    // Tests for PayloadComparisonConfig
    // =====================================================

    #[test]
    fn test_payload_config_default() {
        let config = PayloadComparisonConfig::default();
        assert!(config.ignored_paths.is_empty());
        assert!(config.numeric_tolerance.is_none());
        assert!(config.text_similarity_threshold.is_none());
    }

    #[test]
    fn test_payload_config_ignoring() {
        let config = PayloadComparisonConfig::ignoring(vec![
            "$.timestamp".to_string(),
            "$.request_id".to_string(),
        ]);
        assert_eq!(config.ignored_paths.len(), 2);
        assert!(config.ignored_paths.contains(&"$.timestamp".to_string()));
    }

    #[test]
    fn test_payload_config_builders() {
        let config = PayloadComparisonConfig::default()
            .with_numeric_tolerance(0.001)
            .with_text_threshold(0.9);

        assert_eq!(config.numeric_tolerance, Some(0.001));
        assert_eq!(config.text_similarity_threshold, Some(0.9));
    }

    #[test]
    fn test_payload_config_should_ignore_exact() {
        let config = PayloadComparisonConfig::ignoring(vec!["$.body.timestamp".to_string()]);

        assert!(config.should_ignore("$.body.timestamp"));
        assert!(!config.should_ignore("$.body.other_field"));
        assert!(!config.should_ignore("$.body.timestamp.nested"));
    }

    #[test]
    fn test_payload_config_should_ignore_wildcard() {
        let config = PayloadComparisonConfig::ignoring(vec!["$.headers.*".to_string()]);

        assert!(config.should_ignore("$.headers.Authorization"));
        assert!(config.should_ignore("$.headers.Content-Type"));
        assert!(config.should_ignore("$.headers.X-Custom-Header"));
        assert!(!config.should_ignore("$.body.field"));
    }

    #[test]
    fn test_equivalence_criteria_with_payload_config() {
        let payload_config = PayloadComparisonConfig::ignoring(vec!["$.timestamp".to_string()])
            .with_numeric_tolerance(0.01);

        let criteria = EquivalenceCriteria::default().with_payload_config(payload_config);

        assert!(criteria.payload_config.should_ignore("$.timestamp"));
        assert_eq!(criteria.payload_config.numeric_tolerance, Some(0.01));
    }

    // =====================================================
    // Tests for enhanced wildcard patterns
    // =====================================================

    #[test]
    fn test_wildcard_single_level_does_not_match_nested() {
        // .* should only match one level deep, not nested paths
        let config = PayloadComparisonConfig::ignoring(vec!["$.headers.*".to_string()]);

        assert!(config.should_ignore("$.headers.Authorization"));
        assert!(config.should_ignore("$.headers.X-Custom"));
        // Should NOT match nested paths
        assert!(
            !config.should_ignore("$.headers.nested.field"),
            "Single level wildcard should not match nested paths"
        );
    }

    #[test]
    fn test_wildcard_double_star_match_all() {
        // $.** should match any path
        let config = PayloadComparisonConfig::ignoring(vec!["$.**".to_string()]);

        assert!(config.should_ignore("$.field"));
        assert!(config.should_ignore("$.body.nested"));
        assert!(config.should_ignore("$.a.b.c.d.e"));
    }

    #[test]
    fn test_wildcard_double_star_suffix() {
        // $.**.timestamp should match timestamp at any depth
        let config = PayloadComparisonConfig::ignoring(vec!["$.**.timestamp".to_string()]);

        assert!(config.should_ignore("$.timestamp"));
        assert!(config.should_ignore("$.body.timestamp"));
        assert!(config.should_ignore("$.response.data.timestamp"));
        assert!(config.should_ignore("$.a.b.c.d.timestamp"));
        // Should NOT match fields that aren't exactly "timestamp"
        assert!(!config.should_ignore("$.timestamp_ms"));
        assert!(!config.should_ignore("$.body.timestamp_value"));
    }

    #[test]
    fn test_wildcard_double_star_prefix() {
        // $.body.** should match anything under body
        let config = PayloadComparisonConfig::ignoring(vec!["$.body.**".to_string()]);

        assert!(config.should_ignore("$.body"));
        assert!(config.should_ignore("$.body.field"));
        assert!(config.should_ignore("$.body.nested.deep.field"));
        // Should NOT match siblings of body
        assert!(!config.should_ignore("$.headers.field"));
        assert!(!config.should_ignore("$.other"));
    }

    #[test]
    fn test_wildcard_double_star_middle() {
        // $.data.**.id should match id under any nested path in data
        let config = PayloadComparisonConfig::ignoring(vec!["$.data.**.id".to_string()]);

        assert!(config.should_ignore("$.data.id"));
        assert!(config.should_ignore("$.data.user.id"));
        assert!(config.should_ignore("$.data.nested.deep.id"));
        // Should NOT match id elsewhere
        assert!(!config.should_ignore("$.other.id"));
        assert!(!config.should_ignore("$.id"));
    }

    #[test]
    fn test_wildcard_array_index() {
        // $.items[*].id should match id in any array element
        let config = PayloadComparisonConfig::ignoring(vec!["$.items[*].id".to_string()]);

        assert!(config.should_ignore("$.items[0].id"));
        assert!(config.should_ignore("$.items[1].id"));
        assert!(config.should_ignore("$.items[99].id"));
        assert!(config.should_ignore("$.items[12345].id"));
        // Should NOT match non-array paths
        assert!(!config.should_ignore("$.items.id"));
        assert!(!config.should_ignore("$.items[0].name"));
        assert!(!config.should_ignore("$.other[0].id"));
    }

    #[test]
    fn test_wildcard_array_index_only() {
        // $.data[*] should match any array element directly
        let config = PayloadComparisonConfig::ignoring(vec!["$.data[*]".to_string()]);

        assert!(config.should_ignore("$.data[0]"));
        assert!(config.should_ignore("$.data[42]"));
        assert!(config.should_ignore("$.data[999]"));
        // Should NOT match nested paths or non-array access
        assert!(!config.should_ignore("$.data[0].field"));
        assert!(!config.should_ignore("$.data.field"));
    }

    #[test]
    fn test_wildcard_multiple_array_indices() {
        // $.matrix[*][*] should match nested arrays
        let config = PayloadComparisonConfig::ignoring(vec!["$.matrix[*][*]".to_string()]);

        assert!(config.should_ignore("$.matrix[0][0]"));
        assert!(config.should_ignore("$.matrix[1][2]"));
        assert!(config.should_ignore("$.matrix[99][42]"));
        // Should NOT match single index
        assert!(!config.should_ignore("$.matrix[0]"));
        // Should NOT match non-numeric indices
        assert!(!config.should_ignore("$.matrix[a][b]"));
    }

    #[test]
    fn test_wildcard_array_with_nested_field() {
        // $.users[*].profile.avatar should match deeply nested paths
        let config =
            PayloadComparisonConfig::ignoring(vec!["$.users[*].profile.avatar".to_string()]);

        assert!(config.should_ignore("$.users[0].profile.avatar"));
        assert!(config.should_ignore("$.users[123].profile.avatar"));
        // Should NOT match different nested paths
        assert!(!config.should_ignore("$.users[0].profile.name"));
        assert!(!config.should_ignore("$.users[0].avatar"));
    }

    #[test]
    fn test_wildcard_combined_patterns() {
        // Multiple different patterns combined
        let config = PayloadComparisonConfig::ignoring(vec![
            "$.**.timestamp".to_string(),
            "$.headers.*".to_string(),
            "$.items[*].id".to_string(),
            "$.exact.path".to_string(),
        ]);

        // Test ** pattern
        assert!(config.should_ignore("$.body.nested.timestamp"));
        // Test .* pattern
        assert!(config.should_ignore("$.headers.Authorization"));
        // Test [*] pattern
        assert!(config.should_ignore("$.items[5].id"));
        // Test exact match
        assert!(config.should_ignore("$.exact.path"));
        // None of these should match
        assert!(!config.should_ignore("$.other.field"));
    }
}

// Kani formal verification proofs
// NOTE: Proofs that use PathBuf or HashMap are excluded because Kani
// doesn't support the CCRandomGenerateBytes C function used by HashMap's hasher.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify PayloadComparisonConfig::default has all fields unset
    #[kani::proof]
    fn verify_payload_config_default_fields() {
        let config = PayloadComparisonConfig::default();

        kani::assert(
            config.ignored_paths.is_empty(),
            "Default PayloadComparisonConfig must have empty ignored_paths",
        );
        kani::assert(
            config.numeric_tolerance.is_none(),
            "Default PayloadComparisonConfig must have numeric_tolerance=None",
        );
        kani::assert(
            config.text_similarity_threshold.is_none(),
            "Default PayloadComparisonConfig must have text_similarity_threshold=None",
        );
    }

    /// Verify with_numeric_tolerance sets the tolerance
    #[kani::proof]
    fn verify_payload_config_numeric_tolerance() {
        let tolerance: f64 = kani::any();
        kani::assume(tolerance >= 0.0 && tolerance <= 1.0);

        let config = PayloadComparisonConfig::default().with_numeric_tolerance(tolerance);

        kani::assert(
            config.numeric_tolerance.is_some(),
            "with_numeric_tolerance must set Some",
        );
        kani::assert(
            config.numeric_tolerance.unwrap() == tolerance,
            "with_numeric_tolerance must preserve tolerance value",
        );
    }

    /// Verify with_text_threshold sets the threshold
    #[kani::proof]
    fn verify_payload_config_text_threshold() {
        let threshold: f64 = kani::any();
        kani::assume(threshold >= 0.0 && threshold <= 1.0);

        let config = PayloadComparisonConfig::default().with_text_threshold(threshold);

        kani::assert(
            config.text_similarity_threshold.is_some(),
            "with_text_threshold must set Some",
        );
        kani::assert(
            config.text_similarity_threshold.unwrap() == threshold,
            "with_text_threshold must preserve threshold value",
        );
    }

    /// Verify EquivalenceCriteria::default has correct boolean flags
    #[kani::proof]
    fn verify_equivalence_criteria_default_flags() {
        let criteria = EquivalenceCriteria::default();

        kani::assert(
            criteria.api_requests,
            "Default EquivalenceCriteria must have api_requests=true",
        );
        kani::assert(
            criteria.tool_calls,
            "Default EquivalenceCriteria must have tool_calls=true",
        );
        kani::assert(
            criteria.output,
            "Default EquivalenceCriteria must have output=true",
        );
        kani::assert(
            criteria.timing_tolerance.is_none(),
            "Default EquivalenceCriteria must have timing_tolerance=None",
        );
        kani::assert(
            !criteria.semantic_comparison,
            "Default EquivalenceCriteria must have semantic_comparison=false",
        );
    }

    /// Verify EquivalenceCriteria::api_only has correct flags
    #[kani::proof]
    fn verify_equivalence_criteria_api_only() {
        let criteria = EquivalenceCriteria::api_only();

        kani::assert(
            criteria.api_requests,
            "api_only must have api_requests=true",
        );
        kani::assert(!criteria.tool_calls, "api_only must have tool_calls=false");
        kani::assert(!criteria.output, "api_only must have output=false");
    }

    /// Verify EquivalenceCriteria::output_only has correct flags
    #[kani::proof]
    fn verify_equivalence_criteria_output_only() {
        let criteria = EquivalenceCriteria::output_only();

        kani::assert(
            !criteria.api_requests,
            "output_only must have api_requests=false",
        );
        kani::assert(
            !criteria.tool_calls,
            "output_only must have tool_calls=false",
        );
        kani::assert(criteria.output, "output_only must have output=true");
    }

    /// Verify with_semantic enables semantic comparison
    #[kani::proof]
    fn verify_equivalence_criteria_with_semantic() {
        let criteria = EquivalenceCriteria::default().with_semantic();

        kani::assert(
            criteria.semantic_comparison,
            "with_semantic must enable semantic_comparison",
        );
    }

    /// Verify with_timing_tolerance sets the tolerance
    #[kani::proof]
    fn verify_equivalence_criteria_with_timing_tolerance() {
        let tolerance: f64 = kani::any();
        kani::assume(tolerance >= 0.0 && tolerance <= 1.0);

        let criteria = EquivalenceCriteria::default().with_timing_tolerance(tolerance);

        kani::assert(
            criteria.timing_tolerance.is_some(),
            "with_timing_tolerance must set Some",
        );
        kani::assert(
            criteria.timing_tolerance.unwrap() == tolerance,
            "with_timing_tolerance must preserve tolerance value",
        );
    }

    /// Verify NondeterminismStrategy::semantic preserves threshold
    #[kani::proof]
    fn verify_strategy_semantic_threshold() {
        let threshold: f64 = kani::any();
        kani::assume(threshold >= 0.0 && threshold <= 1.0);

        let strategy = NondeterminismStrategy::semantic(threshold);

        match strategy {
            NondeterminismStrategy::SemanticSimilarity { threshold: t } => {
                kani::assert(t == threshold, "semantic must preserve threshold");
            }
            _ => kani::assert(false, "semantic must return SemanticSimilarity variant"),
        }
    }

    /// Verify NondeterminismStrategy::distribution preserves samples
    #[kani::proof]
    fn verify_strategy_distribution_samples() {
        let samples: usize = kani::any();
        kani::assume(samples > 0 && samples <= 1000);

        let strategy = NondeterminismStrategy::distribution(samples);

        match strategy {
            NondeterminismStrategy::DistributionMatch { samples: s } => {
                kani::assert(s == samples, "distribution must preserve samples");
            }
            _ => kani::assert(false, "distribution must return DistributionMatch variant"),
        }
    }

    // =====================================================
    // Kani proofs for enhanced wildcard matching
    // =====================================================

    /// Verify exact match always works for pattern_matches
    #[kani::proof]
    fn verify_pattern_exact_match() {
        // Test with fixed strings since Kani struggles with arbitrary strings
        let pattern = "$.body.field";
        let path = "$.body.field";

        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, path),
            "Identical pattern and path must match",
        );
    }

    /// Verify exact match fails for different strings
    #[kani::proof]
    fn verify_pattern_exact_mismatch() {
        let pattern = "$.body.field";
        let path = "$.body.other";

        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, path),
            "Different pattern and path must not match",
        );
    }

    /// Verify $.** matches any path starting with $
    #[kani::proof]
    fn verify_double_star_all_match() {
        let pattern = "$.**";

        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.field"),
            "$. ** must match $.field",
        );
        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.body.nested"),
            "$. ** must match $.body.nested",
        );
        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.a.b.c.d"),
            "$. ** must match deeply nested paths",
        );
    }

    /// Verify $.**.suffix matches paths ending with suffix
    #[kani::proof]
    fn verify_double_star_suffix_match() {
        let pattern = "$.**.timestamp";

        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.timestamp"),
            "$.**.timestamp must match $.timestamp",
        );
        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.body.timestamp"),
            "$.**.timestamp must match $.body.timestamp",
        );
        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.a.b.c.timestamp"),
            "$.**.timestamp must match deeply nested timestamp",
        );
    }

    /// Verify $.**.suffix does not match paths with different suffix
    #[kani::proof]
    fn verify_double_star_suffix_no_false_positive() {
        let pattern = "$.**.timestamp";

        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, "$.timestamp_ms"),
            "$.**.timestamp must not match $.timestamp_ms",
        );
        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, "$.body.other"),
            "$.**.timestamp must not match $.body.other",
        );
    }

    /// Verify prefix.** matches paths starting with prefix
    #[kani::proof]
    fn verify_double_star_prefix_match() {
        let pattern = "$.body.**";

        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.body"),
            "$.body.** must match $.body",
        );
        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.body.field"),
            "$.body.** must match $.body.field",
        );
        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.body.nested.deep"),
            "$.body.** must match $.body.nested.deep",
        );
    }

    /// Verify prefix.** does not match sibling paths
    #[kani::proof]
    fn verify_double_star_prefix_no_sibling() {
        let pattern = "$.body.**";

        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, "$.headers.field"),
            "$.body.** must not match $.headers.field",
        );
        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, "$.other"),
            "$.body.** must not match $.other",
        );
    }

    /// Verify [*] matches numeric array indices
    /// NOTE: Uses unwind(20) because array parsing involves loops over string characters
    #[kani::proof]
    #[kani::unwind(20)]
    fn verify_array_wildcard_numeric() {
        let pattern = "$.items[*].id";

        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.items[0].id"),
            "$.items[*].id must match $.items[0].id",
        );
        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.items[99].id"),
            "$.items[*].id must match $.items[99].id",
        );
    }

    /// Verify [*] does not match non-array paths
    /// NOTE: Uses unwind(20) because array parsing involves loops over string characters
    #[kani::proof]
    #[kani::unwind(20)]
    fn verify_array_wildcard_no_non_array() {
        let pattern = "$.items[*].id";

        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, "$.items.id"),
            "$.items[*].id must not match $.items.id",
        );
        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, "$.items[0].name"),
            "$.items[*].id must not match $.items[0].name",
        );
    }

    /// Verify single level .* does not match nested paths
    #[kani::proof]
    fn verify_single_star_no_nested() {
        let pattern = "$.headers.*";

        kani::assert(
            PayloadComparisonConfig::pattern_matches(pattern, "$.headers.Auth"),
            "$.headers.* must match $.headers.Auth",
        );
        kani::assert(
            !PayloadComparisonConfig::pattern_matches(pattern, "$.headers.nested.field"),
            "$.headers.* must not match $.headers.nested.field",
        );
    }

    // NOTE: The verify_nested_array_wildcards proof was removed because it
    // exceeds reasonable Kani verification time (>5 minutes) due to the complexity
    // of nested array index parsing. The functionality is covered by:
    // - Unit test: test_wildcard_multiple_array_indices
    // - Simpler Kani proof: verify_array_wildcard_numeric (single [*])
}
