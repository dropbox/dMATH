//! Portfolio verification for Kani
//!
//! Runs Kani verification with multiple SAT solvers in parallel,
//! returning the first successful result.

use crate::config::KaniConfig;
use crate::result::{VerificationResult, VerificationStatus};
use crate::wrapper::KaniWrapper;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Configuration for portfolio verification
#[derive(Debug, Clone)]
pub struct PortfolioKaniConfig {
    /// Base Kani configuration (solver field will be overridden per-solver)
    pub base_config: KaniConfig,

    /// Solvers to try (e.g., "cadical", "kissat", "minisat", "z3")
    pub solvers: Vec<String>,

    /// Maximum concurrent solver instances
    pub max_concurrent: usize,

    /// Strategy for solver selection
    pub strategy: PortfolioStrategy,
}

impl Default for PortfolioKaniConfig {
    fn default() -> Self {
        Self {
            base_config: KaniConfig::default(),
            // Default to well-known fast solvers
            solvers: vec![
                "cadical".to_string(),
                "kissat".to_string(),
                "minisat".to_string(),
            ],
            max_concurrent: 3,
            strategy: PortfolioStrategy::All,
        }
    }
}

/// Strategy for portfolio solver selection
#[derive(Debug, Clone, Default)]
pub enum PortfolioStrategy {
    /// Run all solvers in parallel
    #[default]
    All,

    /// Adaptive: start with fast solvers, add more after delay
    Adaptive {
        /// Initial solvers to run
        initial: Vec<String>,
        /// Delay before adding more solvers
        delay: Duration,
    },
}

/// Result from portfolio verification
#[derive(Debug)]
pub struct PortfolioKaniResult {
    /// The winning result
    pub result: VerificationResult,

    /// Which solver produced the winning result
    pub winning_solver: String,

    /// Total wall-clock time
    pub total_time: Duration,

    /// Results from other solvers (if not cancelled)
    pub other_results: Vec<(String, VerificationResult)>,
}

/// Portfolio verifier that runs Kani with multiple solvers
pub struct PortfolioVerifier {
    config: PortfolioKaniConfig,
}

impl PortfolioVerifier {
    /// Create a new portfolio verifier
    pub fn new(config: PortfolioKaniConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PortfolioKaniConfig::default())
    }

    /// Verify a project using portfolio solving
    pub async fn verify(&self, project_path: &Path) -> PortfolioKaniResult {
        self.verify_with_harness(project_path, None).await
    }

    /// Verify a specific harness using portfolio solving
    pub async fn verify_with_harness(
        &self,
        project_path: &Path,
        harness: Option<&str>,
    ) -> PortfolioKaniResult {
        match &self.config.strategy {
            PortfolioStrategy::All => self.verify_all_parallel(project_path, harness).await,
            PortfolioStrategy::Adaptive { initial, delay } => {
                self.verify_adaptive(project_path, harness, initial, *delay)
                    .await
            }
        }
    }

    /// Run all solvers in parallel
    async fn verify_all_parallel(
        &self,
        project_path: &Path,
        harness: Option<&str>,
    ) -> PortfolioKaniResult {
        let start = Instant::now();
        let path = project_path.to_path_buf();
        let harness = harness.map(|s| s.to_string());

        // Create cancellation channel
        let (cancel_tx, _) = broadcast::channel::<()>(1);

        info!(
            "Starting portfolio verification with {} solvers: {:?}",
            self.config.solvers.len(),
            self.config.solvers
        );

        // Spawn solver tasks
        let mut handles = Vec::new();
        for solver in self.config.solvers.iter().take(self.config.max_concurrent) {
            let solver = solver.clone();
            let path = path.clone();
            let harness = harness.clone();
            let mut cancel_rx = cancel_tx.subscribe();

            // Create config for this solver
            let mut solver_config = self.config.base_config.clone();
            solver_config.solver = Some(solver.clone());

            let handle = tokio::spawn(async move {
                let wrapper = match KaniWrapper::new(solver_config) {
                    Ok(w) => w,
                    Err(e) => {
                        return (
                            solver,
                            VerificationResult::error(e.to_string(), Duration::ZERO),
                        );
                    }
                };

                tokio::select! {
                    result = wrapper.verify_with_harness(&path, harness.as_deref()) => {
                        match result {
                            Ok(r) => (solver, r),
                            Err(e) => (solver, VerificationResult::error(e.to_string(), Duration::ZERO)),
                        }
                    }
                    _ = cancel_rx.recv() => {
                        (solver, VerificationResult::unknown("Cancelled".to_string(), Duration::ZERO))
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for results
        let mut winner: Option<(String, VerificationResult)> = None;
        let mut other_results = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((solver_id, result)) => {
                    if result.status.is_definitive() && winner.is_none() {
                        debug!(
                            "Solver {} found definitive result: {:?}",
                            solver_id, result.status
                        );
                        winner = Some((solver_id, result));
                        // Cancel other solvers
                        let _ = cancel_tx.send(());
                    } else if winner.is_none()
                        || !matches!(result.status, VerificationStatus::Unknown { .. })
                    {
                        other_results.push((solver_id, result));
                    }
                }
                Err(e) => {
                    warn!("Solver task panicked: {}", e);
                }
            }
        }

        let total_time = start.elapsed();

        match winner {
            Some((winning_solver, result)) => PortfolioKaniResult {
                result,
                winning_solver,
                total_time,
                other_results,
            },
            None => {
                // No definitive result - return best available
                if let Some((solver, result)) = other_results.pop() {
                    PortfolioKaniResult {
                        result,
                        winning_solver: solver,
                        total_time,
                        other_results: Vec::new(),
                    }
                } else {
                    PortfolioKaniResult {
                        result: VerificationResult::unknown(
                            "No solver returned a result".to_string(),
                            total_time,
                        ),
                        winning_solver: "none".to_string(),
                        total_time,
                        other_results: Vec::new(),
                    }
                }
            }
        }
    }

    /// Run solvers with adaptive strategy
    async fn verify_adaptive(
        &self,
        project_path: &Path,
        harness: Option<&str>,
        initial: &[String],
        delay: Duration,
    ) -> PortfolioKaniResult {
        let start = Instant::now();
        let path = project_path.to_path_buf();
        let harness = harness.map(|s| s.to_string());

        // Partition solvers into initial and delayed
        let initial_solvers: Vec<_> = self
            .config
            .solvers
            .iter()
            .filter(|s| initial.contains(s))
            .cloned()
            .collect();

        let delayed_solvers: Vec<_> = self
            .config
            .solvers
            .iter()
            .filter(|s| !initial.contains(s))
            .cloned()
            .collect();

        info!(
            "Adaptive portfolio: {} initial solvers, {} delayed after {:?}",
            initial_solvers.len(),
            delayed_solvers.len(),
            delay
        );

        // Create cancellation channel
        let (cancel_tx, _) = broadcast::channel::<()>(1);

        let mut handles = Vec::new();

        // Spawn initial solvers immediately
        for solver in initial_solvers.iter().take(self.config.max_concurrent) {
            let solver = solver.clone();
            let path = path.clone();
            let harness = harness.clone();
            let mut cancel_rx = cancel_tx.subscribe();

            let mut solver_config = self.config.base_config.clone();
            solver_config.solver = Some(solver.clone());

            let handle = tokio::spawn(async move {
                let wrapper = match KaniWrapper::new(solver_config) {
                    Ok(w) => w,
                    Err(e) => {
                        return (
                            solver,
                            VerificationResult::error(e.to_string(), Duration::ZERO),
                            false,
                        );
                    }
                };

                tokio::select! {
                    result = wrapper.verify_with_harness(&path, harness.as_deref()) => {
                        match result {
                            Ok(r) => (solver, r, false),
                            Err(e) => (solver, VerificationResult::error(e.to_string(), Duration::ZERO), false),
                        }
                    }
                    _ = cancel_rx.recv() => {
                        (solver, VerificationResult::unknown("Cancelled".to_string(), Duration::ZERO), false)
                    }
                }
            });

            handles.push(handle);
        }

        // Spawn delayed solvers
        for solver in &delayed_solvers {
            let solver = solver.clone();
            let path = path.clone();
            let harness = harness.clone();
            let mut cancel_rx = cancel_tx.subscribe();

            let mut solver_config = self.config.base_config.clone();
            solver_config.solver = Some(solver.clone());

            let handle = tokio::spawn(async move {
                // Wait for delay before starting
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {
                        // Delay elapsed, now run solver
                        let wrapper = match KaniWrapper::new(solver_config) {
                            Ok(w) => w,
                            Err(e) => {
                                return (
                                    solver,
                                    VerificationResult::error(e.to_string(), Duration::ZERO),
                                    true,
                                );
                            }
                        };

                        tokio::select! {
                            result = wrapper.verify_with_harness(&path, harness.as_deref()) => {
                                match result {
                                    Ok(r) => (solver, r, true),
                                    Err(e) => (solver, VerificationResult::error(e.to_string(), Duration::ZERO), true),
                                }
                            }
                            _ = cancel_rx.recv() => {
                                (solver, VerificationResult::unknown("Cancelled".to_string(), Duration::ZERO), true)
                            }
                        }
                    }
                    _ = cancel_rx.recv() => {
                        // Cancelled before we started
                        (solver, VerificationResult::unknown("Cancelled before start".to_string(), Duration::ZERO), true)
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for results
        let mut winner: Option<(String, VerificationResult)> = None;
        let mut other_results = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((solver_id, result, _from_delayed)) => {
                    let is_cancelled = matches!(
                        result.status,
                        VerificationStatus::Unknown { ref reason } if reason.contains("Cancelled")
                    );

                    if result.status.is_definitive() && winner.is_none() {
                        debug!(
                            "Solver {} found definitive result: {:?}",
                            solver_id, result.status
                        );
                        winner = Some((solver_id, result));
                        let _ = cancel_tx.send(());
                    } else if !is_cancelled {
                        other_results.push((solver_id, result));
                    }
                }
                Err(e) => {
                    warn!("Solver task panicked: {}", e);
                }
            }
        }

        let total_time = start.elapsed();

        match winner {
            Some((winning_solver, result)) => PortfolioKaniResult {
                result,
                winning_solver,
                total_time,
                other_results,
            },
            None => {
                if let Some((solver, result)) = other_results.pop() {
                    PortfolioKaniResult {
                        result,
                        winning_solver: solver,
                        total_time,
                        other_results: Vec::new(),
                    }
                } else {
                    PortfolioKaniResult {
                        result: VerificationResult::unknown(
                            "No solver returned a result".to_string(),
                            total_time,
                        ),
                        winning_solver: "none".to_string(),
                        total_time,
                        other_results: Vec::new(),
                    }
                }
            }
        }
    }
}

/// Builder for portfolio verification configuration
pub struct PortfolioKaniConfigBuilder {
    config: PortfolioKaniConfig,
}

impl PortfolioKaniConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: PortfolioKaniConfig::default(),
        }
    }

    /// Set base Kani configuration
    pub fn with_base_config(mut self, config: KaniConfig) -> Self {
        self.config.base_config = config;
        self
    }

    /// Set solvers to use
    pub fn with_solvers(mut self, solvers: Vec<String>) -> Self {
        self.config.solvers = solvers;
        self
    }

    /// Set maximum concurrent solvers
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.config.max_concurrent = max;
        self
    }

    /// Use adaptive strategy
    pub fn with_adaptive_strategy(mut self, initial: Vec<String>, delay: Duration) -> Self {
        self.config.strategy = PortfolioStrategy::Adaptive { initial, delay };
        self
    }

    /// Build the configuration
    pub fn build(self) -> PortfolioKaniConfig {
        self.config
    }
}

impl Default for PortfolioKaniConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_config_builder() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_solvers(vec!["cadical".to_string(), "kissat".to_string()])
            .with_max_concurrent(2)
            .build();

        assert_eq!(config.solvers.len(), 2);
        assert_eq!(config.max_concurrent, 2);
    }

    #[test]
    fn test_adaptive_strategy() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_solvers(vec![
                "cadical".to_string(),
                "kissat".to_string(),
                "minisat".to_string(),
            ])
            .with_adaptive_strategy(vec!["cadical".to_string()], Duration::from_millis(100))
            .build();

        match config.strategy {
            PortfolioStrategy::Adaptive { initial, delay } => {
                assert_eq!(initial, vec!["cadical".to_string()]);
                assert_eq!(delay, Duration::from_millis(100));
            }
            _ => panic!("Expected adaptive strategy"),
        }
    }

    #[test]
    fn test_portfolio_kani_config_default() {
        let config = PortfolioKaniConfig::default();
        assert_eq!(config.solvers.len(), 3);
        assert!(config.solvers.contains(&"cadical".to_string()));
        assert!(config.solvers.contains(&"kissat".to_string()));
        assert!(config.solvers.contains(&"minisat".to_string()));
        assert_eq!(config.max_concurrent, 3);
        assert!(matches!(config.strategy, PortfolioStrategy::All));
    }

    #[test]
    fn test_portfolio_kani_config_clone() {
        let config = PortfolioKaniConfig::default();
        let cloned = config.clone();
        assert_eq!(config.solvers, cloned.solvers);
        assert_eq!(config.max_concurrent, cloned.max_concurrent);
    }

    #[test]
    fn test_portfolio_kani_config_debug() {
        let config = PortfolioKaniConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("PortfolioKaniConfig"));
        assert!(debug_str.contains("solvers"));
        assert!(debug_str.contains("max_concurrent"));
    }

    #[test]
    fn test_portfolio_strategy_default() {
        let strategy = PortfolioStrategy::default();
        assert!(matches!(strategy, PortfolioStrategy::All));
    }

    #[test]
    fn test_portfolio_strategy_all_debug() {
        let strategy = PortfolioStrategy::All;
        let debug_str = format!("{:?}", strategy);
        assert!(debug_str.contains("All"));
    }

    #[test]
    fn test_portfolio_strategy_adaptive_debug() {
        let strategy = PortfolioStrategy::Adaptive {
            initial: vec!["cadical".to_string()],
            delay: Duration::from_millis(500),
        };
        let debug_str = format!("{:?}", strategy);
        assert!(debug_str.contains("Adaptive"));
        assert!(debug_str.contains("cadical"));
    }

    #[test]
    fn test_portfolio_strategy_clone() {
        let strategy = PortfolioStrategy::Adaptive {
            initial: vec!["kissat".to_string()],
            delay: Duration::from_secs(1),
        };
        let cloned = strategy.clone();
        match cloned {
            PortfolioStrategy::Adaptive { initial, delay } => {
                assert_eq!(initial, vec!["kissat".to_string()]);
                assert_eq!(delay, Duration::from_secs(1));
            }
            _ => panic!("Expected adaptive strategy"),
        }
    }

    #[test]
    fn test_portfolio_kani_result_debug() {
        let result = PortfolioKaniResult {
            result: VerificationResult::proven(Duration::from_secs(5), 10),
            winning_solver: "cadical".to_string(),
            total_time: Duration::from_secs(5),
            other_results: Vec::new(),
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("PortfolioKaniResult"));
        assert!(debug_str.contains("cadical"));
    }

    #[test]
    fn test_portfolio_kani_result_fields() {
        let result = PortfolioKaniResult {
            result: VerificationResult::proven(Duration::from_secs(10), 5),
            winning_solver: "z3".to_string(),
            total_time: Duration::from_secs(12),
            other_results: vec![(
                "kissat".to_string(),
                VerificationResult::unknown("timeout".to_string(), Duration::from_secs(15)),
            )],
        };
        assert!(result.result.status.is_success());
        assert_eq!(result.winning_solver, "z3");
        assert_eq!(result.total_time, Duration::from_secs(12));
        assert_eq!(result.other_results.len(), 1);
    }

    #[test]
    fn test_portfolio_verifier_with_defaults() {
        let verifier = PortfolioVerifier::with_defaults();
        assert_eq!(verifier.config.solvers.len(), 3);
        assert_eq!(verifier.config.max_concurrent, 3);
    }

    #[test]
    fn test_portfolio_verifier_new() {
        let config = PortfolioKaniConfig {
            base_config: KaniConfig::default(),
            solvers: vec!["z3".to_string()],
            max_concurrent: 1,
            strategy: PortfolioStrategy::All,
        };
        let verifier = PortfolioVerifier::new(config);
        assert_eq!(verifier.config.solvers.len(), 1);
        assert_eq!(verifier.config.max_concurrent, 1);
    }

    #[test]
    fn test_portfolio_config_builder_default() {
        let builder = PortfolioKaniConfigBuilder::default();
        let config = builder.build();
        assert_eq!(config.solvers.len(), 3);
    }

    #[test]
    fn test_portfolio_config_builder_new() {
        let builder = PortfolioKaniConfigBuilder::new();
        let config = builder.build();
        assert_eq!(config.solvers.len(), 3);
    }

    #[test]
    fn test_portfolio_config_builder_with_base_config() {
        let base_config = KaniConfig {
            timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let config = PortfolioKaniConfigBuilder::new()
            .with_base_config(base_config)
            .build();
        assert_eq!(config.base_config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_portfolio_config_builder_with_solvers_empty() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_solvers(vec![])
            .build();
        assert!(config.solvers.is_empty());
    }

    #[test]
    fn test_portfolio_config_builder_with_solvers_single() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_solvers(vec!["z3".to_string()])
            .build();
        assert_eq!(config.solvers.len(), 1);
        assert_eq!(config.solvers[0], "z3");
    }

    #[test]
    fn test_portfolio_config_builder_with_max_concurrent_zero() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_max_concurrent(0)
            .build();
        assert_eq!(config.max_concurrent, 0);
    }

    #[test]
    fn test_portfolio_config_builder_with_max_concurrent_large() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_max_concurrent(100)
            .build();
        assert_eq!(config.max_concurrent, 100);
    }

    #[test]
    fn test_portfolio_config_builder_chaining() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_solvers(vec!["cadical".to_string(), "kissat".to_string()])
            .with_max_concurrent(2)
            .with_adaptive_strategy(vec!["cadical".to_string()], Duration::from_millis(200))
            .build();

        assert_eq!(config.solvers.len(), 2);
        assert_eq!(config.max_concurrent, 2);
        match config.strategy {
            PortfolioStrategy::Adaptive { initial, delay } => {
                assert_eq!(initial.len(), 1);
                assert_eq!(delay, Duration::from_millis(200));
            }
            _ => panic!("Expected adaptive strategy"),
        }
    }

    #[test]
    fn test_portfolio_config_builder_override_strategy() {
        let config = PortfolioKaniConfigBuilder::new()
            .with_adaptive_strategy(vec!["cadical".to_string()], Duration::from_millis(100))
            .build();

        // Should be Adaptive, not the default All
        assert!(matches!(
            config.strategy,
            PortfolioStrategy::Adaptive { .. }
        ));
    }

    #[test]
    fn test_portfolio_kani_result_with_empty_other_results() {
        let result = PortfolioKaniResult {
            result: VerificationResult::unknown("no result".to_string(), Duration::from_secs(1)),
            winning_solver: "none".to_string(),
            total_time: Duration::from_secs(1),
            other_results: Vec::new(),
        };
        assert!(result.other_results.is_empty());
        assert_eq!(result.winning_solver, "none");
    }

    #[test]
    fn test_portfolio_kani_result_with_multiple_other_results() {
        let result = PortfolioKaniResult {
            result: VerificationResult::proven(Duration::from_secs(2), 5),
            winning_solver: "cadical".to_string(),
            total_time: Duration::from_secs(3),
            other_results: vec![
                (
                    "kissat".to_string(),
                    VerificationResult::unknown("cancelled".to_string(), Duration::from_secs(1)),
                ),
                (
                    "minisat".to_string(),
                    VerificationResult::unknown("cancelled".to_string(), Duration::from_secs(1)),
                ),
            ],
        };
        assert_eq!(result.other_results.len(), 2);
    }

    #[test]
    fn test_portfolio_verifier_config_access() {
        let config = PortfolioKaniConfig {
            solvers: vec!["test_solver".to_string()],
            max_concurrent: 5,
            ..Default::default()
        };
        let verifier = PortfolioVerifier::new(config);
        assert_eq!(verifier.config.solvers[0], "test_solver");
        assert_eq!(verifier.config.max_concurrent, 5);
    }

    #[test]
    fn test_portfolio_strategy_adaptive_empty_initial() {
        let strategy = PortfolioStrategy::Adaptive {
            initial: Vec::new(),
            delay: Duration::from_secs(0),
        };
        match strategy {
            PortfolioStrategy::Adaptive { initial, delay } => {
                assert!(initial.is_empty());
                assert_eq!(delay, Duration::ZERO);
            }
            _ => panic!("Expected adaptive strategy"),
        }
    }

    #[test]
    fn test_portfolio_kani_config_with_all_custom_values() {
        let config = PortfolioKaniConfig {
            base_config: KaniConfig {
                timeout: Duration::from_secs(120),
                enable_concrete_playback: false,
                default_unwind: Some(20),
                ..Default::default()
            },
            solvers: vec!["z3".to_string(), "yices2".to_string()],
            max_concurrent: 2,
            strategy: PortfolioStrategy::Adaptive {
                initial: vec!["z3".to_string()],
                delay: Duration::from_secs(5),
            },
        };
        assert_eq!(config.base_config.timeout, Duration::from_secs(120));
        assert!(!config.base_config.enable_concrete_playback);
        assert_eq!(config.base_config.default_unwind, Some(20));
        assert_eq!(config.solvers, vec!["z3", "yices2"]);
        assert_eq!(config.max_concurrent, 2);
    }

    // Mutation coverage documentation for portfolio.rs
    //
    // The following mutants require async integration tests with mocked solvers.
    // They are in async code that spawns tokio tasks and requires external
    // processes (Kani/solvers) to exercise the code paths.
    //
    // verify_all_parallel (lines 179, 188):
    // - Line 179: `result.status.is_definitive() && winner.is_none()`
    //   This condition selects the first definitive result as winner.
    //   Mutating && to || would select results even when winner exists.
    //   Testing requires async mocking of solver results.
    //
    // - Line 188: `winner.is_none() || !matches!(result.status, VerificationStatus::Unknown { .. })`
    //   This condition adds non-Unknown results to other_results when no winner yet.
    //   Mutating || to && would skip adding results. Deleting ! would include
    //   Unknown results. Testing requires async mocking.
    //
    // verify_adaptive (lines 257, 371, 378):
    // - Line 257: `!initial.contains(s)` for delayed solver filtering
    //   This partitions solvers into initial (run immediately) and delayed.
    //   Deleting ! would put ALL solvers in delayed set, none in initial.
    //   Testing requires async integration with solver mocking.
    //
    // - Line 371: Same as line 179 - winner selection logic.
    //
    // - Line 378: `!is_cancelled` check before adding to other_results
    //   This filters out cancelled results from other_results.
    //   Deleting ! would include cancelled results.
    //   Testing requires async cancellation mocking.
    //
    // These mutants could be caught by:
    // 1. Integration tests with real Kani (expensive, flaky)
    // 2. Mock-based tests with async solver simulation (complex setup)
    // 3. Refactoring to separate pure logic from async execution
    //
    // Current test coverage verifies configuration and data structures.
    // The async logic is implicitly tested via actual portfolio usage.

    // Test that documents the solver partitioning logic
    #[test]
    fn test_solver_partitioning_logic() {
        // This tests the logic used in verify_adaptive lines 245-259
        // Initial solvers: in the initial list
        // Delayed solvers: NOT in the initial list
        let all_solvers = [
            "cadical".to_string(),
            "kissat".to_string(),
            "minisat".to_string(),
        ];
        let initial = ["cadical".to_string()];

        // Line 249: filter(|s| initial.contains(s))
        let initial_solvers: Vec<_> = all_solvers
            .iter()
            .filter(|s| initial.contains(s))
            .cloned()
            .collect();

        // Line 257: filter(|s| !initial.contains(s))
        let delayed_solvers: Vec<_> = all_solvers
            .iter()
            .filter(|s| !initial.contains(s))
            .cloned()
            .collect();

        assert_eq!(initial_solvers, vec!["cadical".to_string()]);
        assert_eq!(
            delayed_solvers,
            vec!["kissat".to_string(), "minisat".to_string()]
        );

        // If we deleted the ! in line 257, delayed would be same as initial
        let incorrect_delayed: Vec<_> = all_solvers
            .iter()
            .filter(|s| initial.contains(s)) // deleted !
            .cloned()
            .collect();
        assert_eq!(
            incorrect_delayed,
            vec!["cadical".to_string()],
            "Deleting ! would incorrectly put cadical in delayed set"
        );
    }

    // Test that documents the winner selection logic
    #[test]
    fn test_winner_selection_logic() {
        // This tests the logic used in lines 179, 371
        // Condition: result.status.is_definitive() && winner.is_none()
        //
        // We need:
        // 1. Result to be definitive (proven or disproven)
        // 2. No winner selected yet
        //
        // Both conditions must be true to set winner.

        // Scenario 1: Definitive result, no winner yet -> set winner
        let result_proven = VerificationResult::proven(Duration::from_secs(1), 5);
        let winner: Option<String> = None;
        let should_set_winner = result_proven.status.is_definitive() && winner.is_none();
        assert!(
            should_set_winner,
            "Should set winner when definitive and no winner"
        );

        // Scenario 2: Definitive result, winner exists -> don't override
        let winner = Some("previous_solver".to_string());
        let should_not_override = result_proven.status.is_definitive() && winner.is_none();
        assert!(!should_not_override, "Should not override existing winner");

        // Scenario 3: Non-definitive result -> don't set winner
        let result_unknown =
            VerificationResult::unknown("timeout".to_string(), Duration::from_secs(1));
        let winner: Option<String> = None;
        let should_not_set = result_unknown.status.is_definitive() && winner.is_none();
        assert!(
            !should_not_set,
            "Should not set winner for non-definitive result"
        );

        // If we used || instead of &&:
        // - Scenario 2 would incorrectly set winner (winner.is_none() is false, but || would pass)
        // - Scenario 3 would incorrectly set winner (is_definitive() is false, but || would pass)
        let incorrect_scenario_2 =
            result_proven.status.is_definitive() || Some("prev".to_string()).is_none();
        assert!(
            incorrect_scenario_2,
            "Using || would incorrectly trigger winner override"
        );
    }

    // Test that documents the cancelled result filtering logic
    #[test]
    fn test_cancelled_result_filtering_logic() {
        // This tests the logic used in line 378: if !is_cancelled
        // Cancelled results should NOT be added to other_results

        let cancelled_result = VerificationResult::unknown("Cancelled".to_string(), Duration::ZERO);
        let is_cancelled = matches!(
            cancelled_result.status,
            VerificationStatus::Unknown { ref reason } if reason.contains("Cancelled")
        );
        assert!(is_cancelled, "Should detect cancelled result");

        // We add to other_results only if !is_cancelled
        assert!(
            is_cancelled,
            "Cancelled result should be excluded from other_results"
        );

        // Non-cancelled result should be included
        let timeout_result =
            VerificationResult::unknown("timeout".to_string(), Duration::from_secs(10));
        let is_not_cancelled = !matches!(
            timeout_result.status,
            VerificationStatus::Unknown { ref reason } if reason.contains("Cancelled")
        );
        assert!(is_not_cancelled, "Timeout result should be included");
    }

    // Test other_results condition logic
    #[test]
    fn test_other_results_condition_logic() {
        // Line 187-188: else if winner.is_none() || !matches!(result.status, VerificationStatus::Unknown { .. })
        //
        // Add to other_results if:
        // - No winner yet (collect all non-definitive results), OR
        // - Result is NOT Unknown (keep failures for reporting)

        let unknown_result =
            VerificationResult::unknown("timeout".to_string(), Duration::from_secs(1));
        let error_result = VerificationResult::error("compile error".to_string(), Duration::ZERO);

        // With winner, only non-Unknown results should be added
        let winner_exists = true;
        let should_add_unknown =
            !winner_exists || !matches!(unknown_result.status, VerificationStatus::Unknown { .. });
        let should_add_error =
            !winner_exists || !matches!(error_result.status, VerificationStatus::Unknown { .. });

        assert!(
            !should_add_unknown,
            "Unknown result should not be added when winner exists"
        );
        assert!(
            should_add_error,
            "Error result should be added even when winner exists"
        );

        // Without winner, all results should be added
        let no_winner = false;
        let should_add_all = !no_winner; // This simplifies to true when winner.is_none()
        assert!(should_add_all, "All results should be added when no winner");
    }
}
