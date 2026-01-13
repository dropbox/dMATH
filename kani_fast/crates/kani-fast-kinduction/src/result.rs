//! Result types for k-induction verification

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Result of k-induction verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KInductionResult {
    /// Property proven for all executions
    Proven {
        /// The k value at which induction succeeded
        k: u32,
        /// Optional invariant that was used/discovered
        invariant: Option<String>,
        /// Statistics from the verification
        stats: KInductionStats,
    },

    /// Property disproven with counterexample
    Disproven {
        /// The k value where counterexample was found
        k: u32,
        /// The counterexample trace
        counterexample: Counterexample,
        /// Statistics from the verification
        stats: KInductionStats,
    },

    /// Could not determine result
    Unknown {
        /// Reason verification was inconclusive
        reason: String,
        /// Last k value attempted
        last_k: u32,
        /// Statistics from the verification
        stats: KInductionStats,
    },
}

impl KInductionResult {
    /// Create a proven result
    pub fn proven(k: u32, stats: KInductionStats) -> Self {
        Self::Proven {
            k,
            invariant: None,
            stats,
        }
    }

    /// Create a proven result with invariant
    pub fn proven_with_invariant(k: u32, invariant: String, stats: KInductionStats) -> Self {
        Self::Proven {
            k,
            invariant: Some(invariant),
            stats,
        }
    }

    /// Create a disproven result
    pub fn disproven(k: u32, counterexample: Counterexample, stats: KInductionStats) -> Self {
        Self::Disproven {
            k,
            counterexample,
            stats,
        }
    }

    /// Create an unknown result
    pub fn unknown(reason: impl Into<String>, last_k: u32, stats: KInductionStats) -> Self {
        Self::Unknown {
            reason: reason.into(),
            last_k,
            stats,
        }
    }

    /// Check if property was proven
    pub fn is_proven(&self) -> bool {
        matches!(self, Self::Proven { .. })
    }

    /// Check if property was disproven
    pub fn is_disproven(&self) -> bool {
        matches!(self, Self::Disproven { .. })
    }

    /// Check if result is definitive (proven or disproven)
    pub fn is_definitive(&self) -> bool {
        !matches!(self, Self::Unknown { .. })
    }

    /// Get statistics
    pub fn stats(&self) -> &KInductionStats {
        match self {
            Self::Proven { stats, .. } => stats,
            Self::Disproven { stats, .. } => stats,
            Self::Unknown { stats, .. } => stats,
        }
    }
}

/// A counterexample trace from k-induction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterexample {
    /// States in the execution trace
    pub states: Vec<State>,
    /// The step at which the property was violated
    pub violation_step: u32,
    /// Description of the violated property
    pub violated_property: String,
}

impl Counterexample {
    pub fn new(
        states: Vec<State>,
        violation_step: u32,
        violated_property: impl Into<String>,
    ) -> Self {
        Self {
            states,
            violation_step,
            violated_property: violated_property.into(),
        }
    }
}

/// A state in an execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Step number (0-indexed)
    pub step: u32,
    /// Variable assignments in this state
    pub variables: Vec<(String, String)>,
}

impl State {
    pub fn new(step: u32) -> Self {
        Self {
            step,
            variables: Vec::new(),
        }
    }

    pub fn with_variable(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.push((name.into(), value.into()));
        self
    }

    pub fn add_variable(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.variables.push((name.into(), value.into()));
    }
}

/// Statistics from k-induction verification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KInductionStats {
    /// Total wall-clock time
    pub total_time: Duration,

    /// Time spent on base case checks
    pub base_case_time: Duration,

    /// Time spent on induction step checks
    pub induction_time: Duration,

    /// Time spent on invariant synthesis
    pub invariant_synthesis_time: Duration,

    /// Number of base case queries
    pub base_case_queries: u32,

    /// Number of induction step queries
    pub induction_queries: u32,

    /// Number of invariant synthesis attempts
    pub invariant_attempts: u32,

    /// Number of invariants discovered
    pub invariants_discovered: u32,

    /// Maximum k value reached
    pub max_k_reached: u32,

    /// Whether simple path constraint was used
    pub used_simple_path: bool,
}

impl KInductionStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add time for a base case check
    pub fn add_base_case(&mut self, time: Duration) {
        self.base_case_time += time;
        self.base_case_queries += 1;
    }

    /// Add time for an induction step check
    pub fn add_induction(&mut self, time: Duration) {
        self.induction_time += time;
        self.induction_queries += 1;
    }

    /// Add time for invariant synthesis
    pub fn add_invariant_attempt(&mut self, time: Duration, success: bool) {
        self.invariant_synthesis_time += time;
        self.invariant_attempts += 1;
        if success {
            self.invariants_discovered += 1;
        }
    }

    /// Update max k reached
    pub fn update_max_k(&mut self, k: u32) {
        self.max_k_reached = self.max_k_reached.max(k);
    }

    /// Finalize total time
    pub fn finalize(&mut self, total_time: Duration) {
        self.total_time = total_time;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======== KInductionResult tests ========

    #[test]
    fn test_result_predicates() {
        let proven = KInductionResult::proven(5, KInductionStats::new());
        assert!(proven.is_proven());
        assert!(!proven.is_disproven());
        assert!(proven.is_definitive());

        let disproven = KInductionResult::disproven(
            3,
            Counterexample::new(vec![], 3, "assertion failed"),
            KInductionStats::new(),
        );
        assert!(!disproven.is_proven());
        assert!(disproven.is_disproven());
        assert!(disproven.is_definitive());

        let unknown = KInductionResult::unknown("timeout", 10, KInductionStats::new());
        assert!(!unknown.is_proven());
        assert!(!unknown.is_disproven());
        assert!(!unknown.is_definitive());
    }

    #[test]
    fn test_result_proven() {
        let stats = KInductionStats::new();
        let result = KInductionResult::proven(10, stats);

        if let KInductionResult::Proven { k, invariant, .. } = result {
            assert_eq!(k, 10);
            assert!(invariant.is_none());
        } else {
            panic!("Expected Proven variant");
        }
    }

    #[test]
    fn test_result_proven_with_invariant() {
        let stats = KInductionStats::new();
        let result =
            KInductionResult::proven_with_invariant(5, "(>= x 0)".to_string(), stats.clone());

        if let KInductionResult::Proven { k, invariant, .. } = result {
            assert_eq!(k, 5);
            assert_eq!(invariant, Some("(>= x 0)".to_string()));
        } else {
            panic!("Expected Proven variant");
        }
    }

    #[test]
    fn test_result_disproven() {
        let cex = Counterexample::new(vec![], 3, "test_prop");
        let stats = KInductionStats::new();
        let result = KInductionResult::disproven(3, cex, stats);

        if let KInductionResult::Disproven {
            k, counterexample, ..
        } = result
        {
            assert_eq!(k, 3);
            assert_eq!(counterexample.violation_step, 3);
        } else {
            panic!("Expected Disproven variant");
        }
    }

    #[test]
    fn test_result_unknown() {
        let stats = KInductionStats::new();
        let result = KInductionResult::unknown("solver timeout", 20, stats);

        if let KInductionResult::Unknown { reason, last_k, .. } = result {
            assert_eq!(reason, "solver timeout");
            assert_eq!(last_k, 20);
        } else {
            panic!("Expected Unknown variant");
        }
    }

    #[test]
    fn test_result_stats_accessor() {
        let mut stats = KInductionStats::new();
        stats.max_k_reached = 15;

        let proven = KInductionResult::proven(5, stats.clone());
        assert_eq!(proven.stats().max_k_reached, 15);

        let cex = Counterexample::new(vec![], 3, "test");
        let disproven = KInductionResult::disproven(3, cex, stats.clone());
        assert_eq!(disproven.stats().max_k_reached, 15);

        let unknown = KInductionResult::unknown("test", 10, stats.clone());
        assert_eq!(unknown.stats().max_k_reached, 15);
    }

    #[test]
    fn test_result_clone() {
        let stats = KInductionStats::new();
        let result = KInductionResult::proven(5, stats);
        let cloned = result.clone();
        assert!(cloned.is_proven());
    }

    #[test]
    fn test_result_serialization() {
        let stats = KInductionStats::new();
        let result = KInductionResult::proven(10, stats);
        let json = serde_json::to_string(&result).unwrap();
        let parsed: KInductionResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_proven());
    }

    // ======== Counterexample tests ========

    #[test]
    fn test_counterexample_new() {
        let states = vec![
            State::new(0).with_variable("x", "0"),
            State::new(1).with_variable("x", "1"),
        ];
        let cex = Counterexample::new(states, 1, "x < 0");

        assert_eq!(cex.states.len(), 2);
        assert_eq!(cex.violation_step, 1);
        assert_eq!(cex.violated_property, "x < 0");
    }

    #[test]
    fn test_counterexample_empty_states() {
        let cex = Counterexample::new(vec![], 0, "trivial");
        assert!(cex.states.is_empty());
        assert_eq!(cex.violation_step, 0);
    }

    #[test]
    fn test_counterexample_clone() {
        let cex = Counterexample::new(vec![State::new(0)], 0, "test");
        let cloned = cex.clone();
        assert_eq!(cloned.states.len(), 1);
        assert_eq!(cloned.violated_property, "test");
    }

    #[test]
    fn test_counterexample_serialization() {
        let cex = Counterexample::new(vec![State::new(0).with_variable("x", "42")], 0, "prop");
        let json = serde_json::to_string(&cex).unwrap();
        let parsed: Counterexample = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.violation_step, 0);
        assert_eq!(parsed.states.len(), 1);
    }

    // ======== State tests ========

    #[test]
    fn test_state_builder() {
        let state = State::new(5)
            .with_variable("x", "42")
            .with_variable("y", "true");

        assert_eq!(state.step, 5);
        assert_eq!(state.variables.len(), 2);
        assert_eq!(state.variables[0], ("x".to_string(), "42".to_string()));
    }

    #[test]
    fn test_state_new() {
        let state = State::new(10);
        assert_eq!(state.step, 10);
        assert!(state.variables.is_empty());
    }

    #[test]
    fn test_state_add_variable() {
        let mut state = State::new(0);
        state.add_variable("a", "1");
        state.add_variable("b", "2");

        assert_eq!(state.variables.len(), 2);
        assert_eq!(state.variables[0], ("a".to_string(), "1".to_string()));
        assert_eq!(state.variables[1], ("b".to_string(), "2".to_string()));
    }

    #[test]
    fn test_state_chained_with_variable() {
        let state = State::new(0)
            .with_variable("x", "10")
            .with_variable("y", "20")
            .with_variable("z", "30");

        assert_eq!(state.variables.len(), 3);
    }

    #[test]
    fn test_state_clone() {
        let state = State::new(5).with_variable("test", "value");
        let cloned = state.clone();
        assert_eq!(cloned.step, 5);
        assert_eq!(cloned.variables.len(), 1);
    }

    #[test]
    fn test_state_serialization() {
        let state = State::new(3).with_variable("x", "42");
        let json = serde_json::to_string(&state).unwrap();
        let parsed: State = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.step, 3);
        assert_eq!(parsed.variables[0].0, "x");
    }

    // ======== KInductionStats tests ========

    #[test]
    fn test_stats_accumulation() {
        let mut stats = KInductionStats::new();
        stats.add_base_case(Duration::from_millis(100));
        stats.add_base_case(Duration::from_millis(150));
        stats.add_induction(Duration::from_millis(200));

        assert_eq!(stats.base_case_queries, 2);
        assert_eq!(stats.base_case_time, Duration::from_millis(250));
        assert_eq!(stats.induction_queries, 1);
        assert_eq!(stats.induction_time, Duration::from_millis(200));
    }

    #[test]
    fn test_stats_new() {
        let stats = KInductionStats::new();
        assert_eq!(stats.total_time, Duration::ZERO);
        assert_eq!(stats.base_case_time, Duration::ZERO);
        assert_eq!(stats.induction_time, Duration::ZERO);
        assert_eq!(stats.base_case_queries, 0);
        assert_eq!(stats.induction_queries, 0);
        assert!(!stats.used_simple_path);
    }

    #[test]
    fn test_stats_default() {
        let stats = KInductionStats::default();
        assert_eq!(stats.invariant_attempts, 0);
        assert_eq!(stats.invariants_discovered, 0);
    }

    #[test]
    fn test_stats_add_base_case() {
        let mut stats = KInductionStats::new();
        stats.add_base_case(Duration::from_secs(1));
        stats.add_base_case(Duration::from_secs(2));

        assert_eq!(stats.base_case_queries, 2);
        assert_eq!(stats.base_case_time, Duration::from_secs(3));
    }

    #[test]
    fn test_stats_add_induction() {
        let mut stats = KInductionStats::new();
        stats.add_induction(Duration::from_millis(500));
        stats.add_induction(Duration::from_millis(300));

        assert_eq!(stats.induction_queries, 2);
        assert_eq!(stats.induction_time, Duration::from_millis(800));
    }

    #[test]
    fn test_stats_add_invariant_attempt_success() {
        let mut stats = KInductionStats::new();
        stats.add_invariant_attempt(Duration::from_millis(100), true);

        assert_eq!(stats.invariant_attempts, 1);
        assert_eq!(stats.invariants_discovered, 1);
        assert_eq!(stats.invariant_synthesis_time, Duration::from_millis(100));
    }

    #[test]
    fn test_stats_add_invariant_attempt_failure() {
        let mut stats = KInductionStats::new();
        stats.add_invariant_attempt(Duration::from_millis(100), false);

        assert_eq!(stats.invariant_attempts, 1);
        assert_eq!(stats.invariants_discovered, 0);
    }

    #[test]
    fn test_stats_add_invariant_multiple() {
        let mut stats = KInductionStats::new();
        stats.add_invariant_attempt(Duration::from_millis(100), true);
        stats.add_invariant_attempt(Duration::from_millis(100), false);
        stats.add_invariant_attempt(Duration::from_millis(100), true);

        assert_eq!(stats.invariant_attempts, 3);
        assert_eq!(stats.invariants_discovered, 2);
        assert_eq!(stats.invariant_synthesis_time, Duration::from_millis(300));
    }

    #[test]
    fn test_stats_update_max_k() {
        let mut stats = KInductionStats::new();
        assert_eq!(stats.max_k_reached, 0);

        stats.update_max_k(5);
        assert_eq!(stats.max_k_reached, 5);

        stats.update_max_k(3);
        assert_eq!(stats.max_k_reached, 5); // Should stay at 5

        stats.update_max_k(10);
        assert_eq!(stats.max_k_reached, 10);
    }

    #[test]
    fn test_stats_finalize() {
        let mut stats = KInductionStats::new();
        stats.add_base_case(Duration::from_millis(100));
        stats.add_induction(Duration::from_millis(200));

        stats.finalize(Duration::from_millis(500));
        assert_eq!(stats.total_time, Duration::from_millis(500));
    }

    #[test]
    fn test_stats_used_simple_path() {
        let mut stats = KInductionStats::new();
        assert!(!stats.used_simple_path);

        stats.used_simple_path = true;
        assert!(stats.used_simple_path);
    }

    #[test]
    fn test_stats_clone() {
        let mut stats = KInductionStats::new();
        stats.max_k_reached = 15;
        stats.base_case_queries = 10;

        let cloned = stats.clone();
        assert_eq!(cloned.max_k_reached, 15);
        assert_eq!(cloned.base_case_queries, 10);
    }

    #[test]
    fn test_stats_serialization() {
        let mut stats = KInductionStats::new();
        stats.max_k_reached = 20;
        stats.used_simple_path = true;

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: KInductionStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.max_k_reached, 20);
        assert!(parsed.used_simple_path);
    }

    #[test]
    fn test_stats_debug() {
        let stats = KInductionStats::new();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("KInductionStats"));
    }

    // ======== Integration tests ========

    #[test]
    fn test_full_counterexample_trace() {
        let states = vec![
            State::new(0)
                .with_variable("x", "0")
                .with_variable("y", "true"),
            State::new(1)
                .with_variable("x", "1")
                .with_variable("y", "true"),
            State::new(2)
                .with_variable("x", "-1")
                .with_variable("y", "false"),
        ];

        let cex = Counterexample::new(states, 2, "x >= 0");

        assert_eq!(cex.states.len(), 3);
        assert_eq!(cex.violation_step, 2);
        assert_eq!(cex.states[2].variables.len(), 2);
    }

    #[test]
    fn test_result_with_full_stats() {
        let mut stats = KInductionStats::new();
        stats.add_base_case(Duration::from_millis(100));
        stats.add_base_case(Duration::from_millis(150));
        stats.add_induction(Duration::from_millis(200));
        stats.add_induction(Duration::from_millis(180));
        stats.add_invariant_attempt(Duration::from_millis(50), true);
        stats.update_max_k(5);
        stats.used_simple_path = true;
        stats.finalize(Duration::from_millis(700));

        let result = KInductionResult::proven(5, stats);

        let s = result.stats();
        assert_eq!(s.base_case_queries, 2);
        assert_eq!(s.induction_queries, 2);
        assert_eq!(s.invariant_attempts, 1);
        assert_eq!(s.invariants_discovered, 1);
        assert_eq!(s.max_k_reached, 5);
        assert!(s.used_simple_path);
        assert_eq!(s.total_time, Duration::from_millis(700));
    }
}
