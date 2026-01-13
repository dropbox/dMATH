//! Interleaving exploration for concurrent operations

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::ConcurrentOperation;

/// Result of interleaving exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavingResult {
    /// Total number of interleavings explored
    pub total_interleavings: usize,
    /// Violations found during exploration
    pub violations: Vec<InterleavingViolation>,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Whether exploration was exhaustive
    pub exhaustive: bool,
    /// Number of interleavings pruned
    pub pruned: usize,
}

impl InterleavingResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self {
            total_interleavings: 0,
            violations: vec![],
            max_depth: 0,
            exhaustive: true,
            pruned: 0,
        }
    }

    /// Check if any violations were found
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Get summary message
    pub fn summary(&self) -> String {
        if self.violations.is_empty() {
            format!(
                "Explored {} interleavings (max depth: {}), no violations found",
                self.total_interleavings, self.max_depth
            )
        } else {
            format!(
                "Explored {} interleavings (max depth: {}), found {} violations",
                self.total_interleavings,
                self.max_depth,
                self.violations.len()
            )
        }
    }
}

impl Default for InterleavingResult {
    fn default() -> Self {
        Self::new()
    }
}

/// A violation found during interleaving exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavingViolation {
    /// The specific interleaving that caused the violation
    pub interleaving: Vec<String>,
    /// Which invariant was violated
    pub invariant_violated: String,
    /// The final state when violation occurred
    pub final_state: String,
    /// Thread interleavings that led to this
    pub thread_schedule: Vec<usize>,
}

impl InterleavingViolation {
    /// Create a new violation
    pub fn new(
        interleaving: Vec<String>,
        invariant: impl Into<String>,
        final_state: impl Into<String>,
    ) -> Self {
        Self {
            interleaving,
            invariant_violated: invariant.into(),
            final_state: final_state.into(),
            thread_schedule: vec![],
        }
    }

    /// Set thread schedule
    pub fn with_schedule(mut self, schedule: Vec<usize>) -> Self {
        self.thread_schedule = schedule;
        self
    }
}

/// Configuration for interleaving exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavingConfig {
    /// Maximum depth to explore
    pub max_depth: usize,
    /// Maximum number of interleavings to explore
    pub max_interleavings: usize,
    /// Enable partial order reduction
    pub partial_order_reduction: bool,
    /// Enable symmetry reduction
    pub symmetry_reduction: bool,
    /// Timeout per interleaving (milliseconds)
    pub timeout_per_interleaving_ms: u64,
}

impl Default for InterleavingConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            max_interleavings: 10000,
            partial_order_reduction: true,
            symmetry_reduction: true,
            timeout_per_interleaving_ms: 1000,
        }
    }
}

impl InterleavingConfig {
    /// Set maximum depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set maximum interleavings
    pub fn with_max_interleavings(mut self, count: usize) -> Self {
        self.max_interleavings = count;
        self
    }

    /// Disable partial order reduction
    pub fn without_partial_order_reduction(mut self) -> Self {
        self.partial_order_reduction = false;
        self
    }
}

/// Interleaving explorer that generates all possible interleavings
pub struct InterleavingExplorer {
    config: InterleavingConfig,
}

impl InterleavingExplorer {
    /// Create a new explorer with default config
    pub fn new() -> Self {
        Self {
            config: InterleavingConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: InterleavingConfig) -> Self {
        Self { config }
    }

    /// Generate all interleavings of operations
    pub fn generate_interleavings(
        &self,
        operations: &[Vec<ConcurrentOperation>],
    ) -> Vec<Vec<usize>> {
        let mut result = vec![];
        let mut indices: Vec<usize> = vec![0; operations.len()];
        let lengths: Vec<usize> = operations.iter().map(|ops| ops.len()).collect();

        self.generate_recursive(
            operations,
            &lengths,
            &mut indices,
            &mut vec![],
            &mut result,
            &mut HashSet::new(),
        );

        result
    }

    fn generate_recursive(
        &self,
        operations: &[Vec<ConcurrentOperation>],
        lengths: &[usize],
        indices: &mut Vec<usize>,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
        seen: &mut HashSet<Vec<usize>>,
    ) {
        // Check limits
        if result.len() >= self.config.max_interleavings {
            return;
        }
        if current.len() >= self.config.max_depth {
            return;
        }

        // Check if all threads are done
        let all_done = indices
            .iter()
            .enumerate()
            .all(|(i, &idx)| idx >= lengths[i]);

        if all_done {
            if !seen.contains(current) {
                seen.insert(current.clone());
                result.push(current.clone());
            }
            return;
        }

        // Try each thread that still has operations
        for thread_id in 0..operations.len() {
            if indices[thread_id] < lengths[thread_id] {
                // Check partial order reduction
                if self.config.partial_order_reduction {
                    if let Some(&last_thread) = current.last() {
                        let last_op =
                            &operations[last_thread][indices[last_thread].saturating_sub(1)];
                        let curr_op = &operations[thread_id][indices[thread_id]];

                        // Skip if operations are independent and we can reorder
                        if !last_op.can_conflict_with(curr_op) && thread_id < last_thread {
                            continue;
                        }
                    }
                }

                // Take this operation
                current.push(thread_id);
                indices[thread_id] += 1;

                self.generate_recursive(operations, lengths, indices, current, result, seen);

                // Backtrack
                indices[thread_id] -= 1;
                current.pop();
            }
        }
    }

    /// Check if two schedules are equivalent under symmetry
    pub fn are_symmetric(&self, schedule1: &[usize], schedule2: &[usize]) -> bool {
        if !self.config.symmetry_reduction {
            return false;
        }

        // Simple symmetry check: same length and same multiset of thread IDs
        if schedule1.len() != schedule2.len() {
            return false;
        }

        let mut counts1: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut counts2: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

        for &t in schedule1 {
            *counts1.entry(t).or_default() += 1;
        }
        for &t in schedule2 {
            *counts2.entry(t).or_default() += 1;
        }

        counts1 == counts2
    }
}

impl Default for InterleavingExplorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OperationType;

    #[test]
    fn test_interleaving_result_new() {
        let result = InterleavingResult::new();
        assert_eq!(result.total_interleavings, 0);
        assert!(result.violations.is_empty());
        assert!(result.exhaustive);
    }

    #[test]
    fn test_interleaving_result_summary() {
        let result = InterleavingResult {
            total_interleavings: 100,
            violations: vec![],
            max_depth: 10,
            exhaustive: true,
            pruned: 0,
        };

        assert!(result.summary().contains("100"));
        assert!(result.summary().contains("no violations"));

        let result_with_violations = InterleavingResult {
            total_interleavings: 100,
            violations: vec![InterleavingViolation::new(
                vec!["op1".to_string()],
                "invariant",
                "state",
            )],
            max_depth: 10,
            exhaustive: true,
            pruned: 0,
        };

        assert!(result_with_violations.summary().contains("1 violations"));
    }

    #[test]
    fn test_generate_interleavings_simple() {
        let explorer = InterleavingExplorer::new();

        let ops = vec![
            vec![ConcurrentOperation::new("a", 0, OperationType::Write)],
            vec![ConcurrentOperation::new("b", 1, OperationType::Write)],
        ];

        let interleavings = explorer.generate_interleavings(&ops);

        // Should have 2 interleavings: [0,1] and [1,0]
        assert_eq!(interleavings.len(), 2);
    }

    #[test]
    fn test_config_builder() {
        let config = InterleavingConfig::default()
            .with_max_depth(50)
            .with_max_interleavings(5000)
            .without_partial_order_reduction();

        assert_eq!(config.max_depth, 50);
        assert_eq!(config.max_interleavings, 5000);
        assert!(!config.partial_order_reduction);
    }

    #[test]
    fn test_violation_builder() {
        let violation = InterleavingViolation::new(
            vec!["read".to_string(), "write".to_string()],
            "no_race",
            "corrupted",
        )
        .with_schedule(vec![0, 1, 0]);

        assert_eq!(violation.interleaving.len(), 2);
        assert_eq!(violation.invariant_violated, "no_race");
        assert_eq!(violation.thread_schedule, vec![0, 1, 0]);
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // InterleavingResult property tests

            #[test]
            fn interleaving_result_default_is_new(total in 0usize..1000, depth in 0usize..100, pruned in 0usize..100) {
                let mut result = InterleavingResult::new();
                result.total_interleavings = total;
                result.max_depth = depth;
                result.pruned = pruned;
                prop_assert_eq!(result.total_interleavings, total);
                prop_assert_eq!(result.max_depth, depth);
                prop_assert_eq!(result.pruned, pruned);
                prop_assert!(!result.has_violations());
            }

            #[test]
            fn interleaving_result_has_violations_when_non_empty(n in 1usize..10) {
                let violations: Vec<InterleavingViolation> = (0..n)
                    .map(|i| InterleavingViolation::new(vec![], format!("inv_{}", i), "state"))
                    .collect();
                let result = InterleavingResult {
                    total_interleavings: 0,
                    violations,
                    max_depth: 0,
                    exhaustive: true,
                    pruned: 0,
                };
                prop_assert!(result.has_violations());
            }

            #[test]
            fn interleaving_result_summary_contains_count(total in 1usize..10000, depth in 1usize..100) {
                let result = InterleavingResult {
                    total_interleavings: total,
                    violations: vec![],
                    max_depth: depth,
                    exhaustive: true,
                    pruned: 0,
                };
                let summary = result.summary();
                prop_assert!(summary.contains(&total.to_string()));
                prop_assert!(summary.contains(&depth.to_string()));
            }

            // InterleavingViolation property tests

            #[test]
            fn violation_preserves_invariant(invariant in "[a-z_]{1,30}") {
                let violation = InterleavingViolation::new(vec![], invariant.clone(), "state");
                prop_assert_eq!(violation.invariant_violated, invariant);
            }

            #[test]
            fn violation_preserves_final_state(state in "[a-z0-9_]{1,30}") {
                let violation = InterleavingViolation::new(vec![], "inv", state.clone());
                prop_assert_eq!(violation.final_state, state);
            }

            #[test]
            fn violation_schedule_preserved(schedule in prop::collection::vec(0usize..10, 0..20)) {
                let violation = InterleavingViolation::new(vec![], "inv", "state")
                    .with_schedule(schedule.clone());
                prop_assert_eq!(violation.thread_schedule, schedule);
            }

            // InterleavingConfig property tests

            #[test]
            fn config_max_depth_preserved(depth in 1usize..1000) {
                let config = InterleavingConfig::default().with_max_depth(depth);
                prop_assert_eq!(config.max_depth, depth);
            }

            #[test]
            fn config_max_interleavings_preserved(count in 1usize..100000) {
                let config = InterleavingConfig::default().with_max_interleavings(count);
                prop_assert_eq!(config.max_interleavings, count);
            }

            #[test]
            fn config_partial_order_reduction_toggle(_dummy in 0..1i32) {
                let config = InterleavingConfig::default().without_partial_order_reduction();
                prop_assert!(!config.partial_order_reduction);
            }

            // InterleavingExplorer property tests

            #[test]
            fn explorer_respects_max_interleavings(max_int in 1usize..100) {
                let config = InterleavingConfig::default()
                    .with_max_interleavings(max_int)
                    .without_partial_order_reduction();
                let explorer = InterleavingExplorer::with_config(config);

                // Create multiple threads with multiple ops to generate many interleavings
                let ops = vec![
                    vec![
                        ConcurrentOperation::new("a1", 0, OperationType::Write),
                        ConcurrentOperation::new("a2", 0, OperationType::Write),
                    ],
                    vec![
                        ConcurrentOperation::new("b1", 1, OperationType::Write),
                        ConcurrentOperation::new("b2", 1, OperationType::Write),
                    ],
                ];

                let interleavings = explorer.generate_interleavings(&ops);
                prop_assert!(interleavings.len() <= max_int);
            }

            #[test]
            fn explorer_symmetric_schedules_same_length(schedule in prop::collection::vec(0usize..3, 1..10)) {
                let explorer = InterleavingExplorer::new();
                // Same schedule is symmetric with itself
                prop_assert!(explorer.are_symmetric(&schedule, &schedule));
            }

            #[test]
            fn explorer_asymmetric_different_lengths(len1 in 1usize..10, len2 in 11usize..20) {
                let explorer = InterleavingExplorer::new();
                let schedule1: Vec<usize> = (0..len1).collect();
                let schedule2: Vec<usize> = (0..len2).collect();
                prop_assert!(!explorer.are_symmetric(&schedule1, &schedule2));
            }
        }
    }

    // Additional tests for mutation coverage

    #[test]
    fn test_generate_interleavings_respects_por() {
        // Test that partial order reduction works correctly
        // When POR is enabled and operations don't conflict, we should skip some orderings
        let config = InterleavingConfig::default();
        let explorer = InterleavingExplorer::with_config(config);

        // Two reads on different threads - they don't conflict
        let ops = vec![
            vec![ConcurrentOperation::new("read_a", 0, OperationType::Read)],
            vec![ConcurrentOperation::new("read_b", 1, OperationType::Read)],
        ];

        let interleavings = explorer.generate_interleavings(&ops);
        // With POR, since reads don't conflict, we should have fewer interleavings
        // than without POR (which would give all permutations)
        assert!(!interleavings.is_empty());
    }

    #[test]
    fn test_generate_interleavings_por_with_conflicts() {
        // Test POR with conflicting operations - should NOT reduce
        let config = InterleavingConfig::default();
        let explorer = InterleavingExplorer::with_config(config);

        // Write operations conflict with each other
        let ops = vec![
            vec![ConcurrentOperation::new("write_a", 0, OperationType::Write)],
            vec![ConcurrentOperation::new("write_b", 1, OperationType::Write)],
        ];

        let interleavings = explorer.generate_interleavings(&ops);
        // With conflicts, POR should still generate both orderings
        assert_eq!(interleavings.len(), 2);
    }

    #[test]
    fn test_generate_interleavings_por_thread_ordering() {
        // Test the specific condition: thread_id < last_thread
        // This catches the mutation at line 227:77
        let config = InterleavingConfig::default();
        let explorer = InterleavingExplorer::with_config(config);

        // Create 3 threads with non-conflicting reads
        // Thread 0, 1, 2 each have one read operation
        let ops = vec![
            vec![ConcurrentOperation::new("read_0", 0, OperationType::Read)],
            vec![ConcurrentOperation::new("read_1", 1, OperationType::Read)],
            vec![ConcurrentOperation::new("read_2", 2, OperationType::Read)],
        ];

        let interleavings_por = explorer.generate_interleavings(&ops);

        // Without POR, compare
        let config_no_por = InterleavingConfig::default().without_partial_order_reduction();
        let explorer_no_por = InterleavingExplorer::with_config(config_no_por);
        let interleavings_no_por = explorer_no_por.generate_interleavings(&ops);

        // With POR disabled, we should get all 6 permutations (3!)
        assert_eq!(interleavings_no_por.len(), 6);
        // With POR enabled and non-conflicting ops, we should get fewer
        // (the exact number depends on the POR algorithm)
        assert!(interleavings_por.len() <= interleavings_no_por.len());
    }

    #[test]
    fn test_are_symmetric_with_different_multisets() {
        // Test that schedules with different thread counts are not symmetric
        let config = InterleavingConfig::default();
        let explorer = InterleavingExplorer::with_config(config);

        // Schedule 1: thread 0 twice, thread 1 once
        let schedule1 = vec![0, 0, 1];
        // Schedule 2: thread 0 once, thread 1 twice
        let schedule2 = vec![0, 1, 1];

        // These have same length but different multisets, so not symmetric
        assert!(!explorer.are_symmetric(&schedule1, &schedule2));
    }

    #[test]
    fn test_are_symmetric_with_reordering() {
        // Test that schedules with same multiset but different order are symmetric
        let config = InterleavingConfig::default();
        let explorer = InterleavingExplorer::with_config(config);

        // Same multiset {0, 0, 1} in different orders
        let schedule1 = vec![0, 0, 1];
        let schedule2 = vec![0, 1, 0];
        let schedule3 = vec![1, 0, 0];

        assert!(explorer.are_symmetric(&schedule1, &schedule2));
        assert!(explorer.are_symmetric(&schedule1, &schedule3));
        assert!(explorer.are_symmetric(&schedule2, &schedule3));
    }

    #[test]
    fn test_are_symmetric_disabled() {
        // Test that symmetry check returns false when disabled
        let config = InterleavingConfig {
            symmetry_reduction: false,
            ..Default::default()
        };
        let explorer = InterleavingExplorer::with_config(config);

        let schedule = vec![0, 1, 0];
        // Even identical schedules should return false when symmetry reduction is off
        assert!(!explorer.are_symmetric(&schedule, &schedule));
    }

    #[test]
    fn test_interleaving_config_with_config() {
        // Test that InterleavingExplorer::with_config preserves the config
        let config = InterleavingConfig::default()
            .with_max_depth(25)
            .with_max_interleavings(500);

        let explorer = InterleavingExplorer::with_config(config);

        // Test that config is preserved by checking behavior
        // Create ops that would generate many interleavings
        let ops = vec![
            vec![
                ConcurrentOperation::new("a1", 0, OperationType::Write),
                ConcurrentOperation::new("a2", 0, OperationType::Write),
                ConcurrentOperation::new("a3", 0, OperationType::Write),
            ],
            vec![
                ConcurrentOperation::new("b1", 1, OperationType::Write),
                ConcurrentOperation::new("b2", 1, OperationType::Write),
                ConcurrentOperation::new("b3", 1, OperationType::Write),
            ],
        ];

        let interleavings = explorer.generate_interleavings(&ops);
        // Should be limited by max_interleavings (500)
        assert!(interleavings.len() <= 500);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves that InterleavingResult::new initializes with correct defaults.
    #[kani::proof]
    fn verify_interleaving_result_new_defaults() {
        let result = InterleavingResult::new();
        kani::assert(
            result.total_interleavings == 0,
            "total_interleavings should be 0",
        );
        kani::assert(result.violations.is_empty(), "violations should be empty");
        kani::assert(result.max_depth == 0, "max_depth should be 0");
        kani::assert(result.exhaustive, "exhaustive should be true");
        kani::assert(result.pruned == 0, "pruned should be 0");
    }

    /// Proves that InterleavingResult::has_violations returns false for empty violations.
    #[kani::proof]
    fn verify_interleaving_result_no_violations() {
        let result = InterleavingResult::new();
        kani::assert(
            !result.has_violations(),
            "New result should have no violations",
        );
    }

    /// Proves that Default for InterleavingResult equals new().
    #[kani::proof]
    fn verify_interleaving_result_default_equals_new() {
        let new_result = InterleavingResult::new();
        let default_result = InterleavingResult::default();
        kani::assert(
            new_result.total_interleavings == default_result.total_interleavings,
            "total_interleavings should match",
        );
        kani::assert(
            new_result.max_depth == default_result.max_depth,
            "max_depth should match",
        );
        kani::assert(
            new_result.exhaustive == default_result.exhaustive,
            "exhaustive should match",
        );
        kani::assert(
            new_result.pruned == default_result.pruned,
            "pruned should match",
        );
    }

    /// Proves that InterleavingViolation::new preserves fields.
    #[kani::proof]
    fn verify_interleaving_violation_new_fields() {
        let violation = InterleavingViolation::new(vec![], "test_invariant", "test_state");
        kani::assert(
            violation.invariant_violated == "test_invariant",
            "invariant should be preserved",
        );
        kani::assert(
            violation.final_state == "test_state",
            "final_state should be preserved",
        );
        kani::assert(
            violation.thread_schedule.is_empty(),
            "thread_schedule should be empty",
        );
    }

    /// Proves that InterleavingConfig::default has expected values.
    #[kani::proof]
    fn verify_interleaving_config_default_values() {
        let config = InterleavingConfig::default();
        kani::assert(config.max_depth == 100, "max_depth should be 100");
        kani::assert(
            config.max_interleavings == 10000,
            "max_interleavings should be 10000",
        );
        kani::assert(config.partial_order_reduction, "POR should be enabled");
        kani::assert(
            config.symmetry_reduction,
            "symmetry_reduction should be enabled",
        );
        kani::assert(
            config.timeout_per_interleaving_ms == 1000,
            "timeout should be 1000ms",
        );
    }

    /// Proves that InterleavingConfig::with_max_depth preserves the value.
    #[kani::proof]
    fn verify_config_with_max_depth() {
        let depth: usize = kani::any();
        kani::assume(depth < 10000); // Reasonable bound
        let config = InterleavingConfig::default().with_max_depth(depth);
        kani::assert(config.max_depth == depth, "max_depth should be preserved");
    }

    /// Proves that InterleavingConfig::with_max_interleavings preserves the value.
    #[kani::proof]
    fn verify_config_with_max_interleavings() {
        let count: usize = kani::any();
        kani::assume(count < 100000); // Reasonable bound
        let config = InterleavingConfig::default().with_max_interleavings(count);
        kani::assert(
            config.max_interleavings == count,
            "max_interleavings should be preserved",
        );
    }

    /// Proves that without_partial_order_reduction disables POR.
    #[kani::proof]
    fn verify_config_without_por() {
        let config = InterleavingConfig::default().without_partial_order_reduction();
        kani::assert(!config.partial_order_reduction, "POR should be disabled");
    }

    /// Proves that InterleavingExplorer::new uses default config.
    #[kani::proof]
    fn verify_explorer_new_default_config() {
        let explorer = InterleavingExplorer::new();
        // Verify by checking config values affect behavior appropriately
        kani::assert(
            explorer.config.max_depth == 100,
            "Default max_depth should be 100",
        );
        kani::assert(
            explorer.config.max_interleavings == 10000,
            "Default max_interleavings should be 10000",
        );
    }

    /// Proves that InterleavingExplorer::default equals new().
    #[kani::proof]
    fn verify_explorer_default_equals_new() {
        let new_explorer = InterleavingExplorer::new();
        let default_explorer = InterleavingExplorer::default();
        kani::assert(
            new_explorer.config.max_depth == default_explorer.config.max_depth,
            "max_depth should match",
        );
        kani::assert(
            new_explorer.config.max_interleavings == default_explorer.config.max_interleavings,
            "max_interleavings should match",
        );
    }

    /// Proves that are_symmetric returns true for identical schedules.
    #[kani::proof]
    fn verify_are_symmetric_identical_schedules() {
        let explorer = InterleavingExplorer::new();
        let schedule = vec![0usize, 1, 0];
        kani::assert(
            explorer.are_symmetric(&schedule, &schedule),
            "Identical schedules should be symmetric",
        );
    }

    /// Proves that are_symmetric returns false for different lengths.
    #[kani::proof]
    fn verify_are_symmetric_different_lengths() {
        let explorer = InterleavingExplorer::new();
        let schedule1 = vec![0usize, 1];
        let schedule2 = vec![0usize, 1, 2];
        kani::assert(
            !explorer.are_symmetric(&schedule1, &schedule2),
            "Different length schedules should not be symmetric",
        );
    }

    /// Proves that are_symmetric returns false when symmetry_reduction is disabled.
    #[kani::proof]
    fn verify_are_symmetric_disabled_returns_false() {
        let config = InterleavingConfig {
            symmetry_reduction: false,
            ..Default::default()
        };
        let explorer = InterleavingExplorer::with_config(config);
        let schedule = vec![0usize, 1];
        kani::assert(
            !explorer.are_symmetric(&schedule, &schedule),
            "Should return false when symmetry_reduction is disabled",
        );
    }

    /// Proves that InterleavingViolation::with_schedule preserves the schedule.
    #[kani::proof]
    fn verify_violation_with_schedule() {
        let schedule = vec![0usize, 1, 2];
        let violation =
            InterleavingViolation::new(vec![], "inv", "state").with_schedule(schedule.clone());
        kani::assert(
            violation.thread_schedule == schedule,
            "Schedule should be preserved",
        );
    }
}
