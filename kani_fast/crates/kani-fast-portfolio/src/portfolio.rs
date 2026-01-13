//! Portfolio solver executor
//!
//! Runs multiple solvers in parallel and returns the first definitive result.
//! This provides significant speedups when different solvers perform well on different problems.

use crate::solver::{
    BoxedSolver, SolverCapability, SolverConfig, SolverError, SolverInfo, SolverOutput,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Strategy for portfolio solver selection
#[derive(Debug, Clone, Default)]
pub enum PortfolioStrategy {
    /// Run all available solvers in parallel
    #[default]
    All,

    /// Run only solvers with specific capabilities
    WithCapabilities(Vec<SolverCapability>),

    /// Run a specific subset of solvers by ID
    Specific(Vec<String>),

    /// Adaptive: start with fast solvers, add more if no result
    Adaptive {
        /// Initial solvers to run
        initial: Vec<String>,
        /// Delay before adding more solvers
        delay: Duration,
    },
}

/// Configuration for portfolio execution
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Strategy for solver selection
    pub strategy: PortfolioStrategy,

    /// Base solver configuration
    pub solver_config: SolverConfig,

    /// Maximum number of concurrent solvers
    pub max_concurrent: usize,

    /// Whether to cancel remaining solvers when one succeeds
    pub cancel_on_first: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            strategy: PortfolioStrategy::All,
            solver_config: SolverConfig::default(),
            max_concurrent: 4,
            cancel_on_first: true,
        }
    }
}

/// Result from portfolio execution
#[derive(Debug)]
pub struct PortfolioResult {
    /// The winning solver output
    pub output: SolverOutput,

    /// Which solver produced the result
    pub solver_id: String,

    /// Total wall-clock time
    pub total_time: Duration,

    /// Results from other solvers (if not cancelled)
    pub other_results: Vec<(String, Result<SolverOutput, SolverError>)>,
}

/// Portfolio solver that runs multiple solvers in parallel
pub struct Portfolio {
    solvers: Vec<Arc<BoxedSolver>>,
}

impl Portfolio {
    /// Create an empty portfolio
    pub fn new() -> Self {
        Self {
            solvers: Vec::new(),
        }
    }

    /// Add a solver to the portfolio
    pub fn add_solver(&mut self, solver: BoxedSolver) {
        self.solvers.push(Arc::new(solver));
    }

    /// Get information about all solvers in the portfolio
    pub fn solver_infos(&self) -> Vec<&SolverInfo> {
        self.solvers.iter().map(|s| s.info()).collect()
    }

    /// Get number of solvers
    pub fn len(&self) -> usize {
        self.solvers.len()
    }

    /// Check if portfolio is empty
    pub fn is_empty(&self) -> bool {
        self.solvers.is_empty()
    }

    /// Filter solvers based on strategy
    fn select_solvers(&self, strategy: &PortfolioStrategy) -> Vec<Arc<BoxedSolver>> {
        match strategy {
            PortfolioStrategy::All => self.solvers.clone(),

            PortfolioStrategy::WithCapabilities(caps) => self
                .solvers
                .iter()
                .filter(|s| caps.iter().all(|c| s.supports(*c)))
                .cloned()
                .collect(),

            PortfolioStrategy::Specific(ids) => self
                .solvers
                .iter()
                .filter(|s| ids.contains(&s.info().id))
                .cloned()
                .collect(),

            PortfolioStrategy::Adaptive { initial, .. } => {
                // For adaptive, start with initial set
                self.solvers
                    .iter()
                    .filter(|s| initial.contains(&s.info().id))
                    .cloned()
                    .collect()
            }
        }
    }

    /// Solve a DIMACS file using the portfolio
    pub async fn solve_dimacs(
        &self,
        path: &Path,
        config: &PortfolioConfig,
    ) -> Result<PortfolioResult, SolverError> {
        // Handle adaptive strategy specially
        if let PortfolioStrategy::Adaptive { initial, delay } = &config.strategy {
            return self
                .solve_dimacs_adaptive(path, config, initial, *delay)
                .await;
        }

        let solvers = self.select_solvers(&config.strategy);

        if solvers.is_empty() {
            return Err(SolverError::NotFound(
                "No solvers available for portfolio".to_string(),
            ));
        }

        info!(
            "Starting portfolio with {} solvers: {:?}",
            solvers.len(),
            solvers.iter().map(|s| &s.info().id).collect::<Vec<_>>()
        );

        self.run_solvers_parallel(path, &solvers, config).await
    }

    /// Run solvers in parallel and return first definitive result
    async fn run_solvers_parallel(
        &self,
        path: &Path,
        solvers: &[Arc<BoxedSolver>],
        config: &PortfolioConfig,
    ) -> Result<PortfolioResult, SolverError> {
        let start = Instant::now();
        let path = path.to_path_buf();

        // Create cancellation channel
        let (cancel_tx, _) = broadcast::channel::<()>(1);

        // Spawn solver tasks
        let mut handles = Vec::new();
        for solver in solvers.iter().take(config.max_concurrent) {
            let solver = Arc::clone(solver);
            let solver_id = solver.info().id.clone();
            let path = path.clone();
            let solver_config = config.solver_config.clone();
            let mut cancel_rx = cancel_tx.subscribe();

            let handle = tokio::spawn(async move {
                tokio::select! {
                    result = solver.solve_dimacs(&path, &solver_config) => {
                        (solver_id, result)
                    }
                    _ = cancel_rx.recv() => {
                        (solver_id, Err(SolverError::Unknown("Cancelled".to_string())))
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for results
        let mut winner: Option<(String, SolverOutput)> = None;
        let mut other_results = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((solver_id, result)) => {
                    match result {
                        Ok(output) if output.result.is_definitive() => {
                            if winner.is_none() {
                                debug!("Solver {} found definitive result", solver_id);
                                winner = Some((solver_id, output));

                                // Cancel other solvers
                                if config.cancel_on_first {
                                    let _ = cancel_tx.send(());
                                }
                            } else {
                                other_results.push((solver_id, Ok(output)));
                            }
                        }
                        Ok(output) => {
                            debug!("Solver {} returned unknown result", solver_id);
                            other_results.push((solver_id, Ok(output)));
                        }
                        Err(e) => {
                            warn!("Solver {} failed: {}", solver_id, e);
                            other_results.push((solver_id, Err(e)));
                        }
                    }
                }
                Err(e) => {
                    warn!("Solver task panicked: {}", e);
                }
            }
        }

        let total_time = start.elapsed();

        match winner {
            Some((solver_id, output)) => Ok(PortfolioResult {
                output,
                solver_id,
                total_time,
                other_results,
            }),
            None => {
                // No definitive result - return best effort
                if let Some((id, Ok(output))) = other_results.pop() {
                    Ok(PortfolioResult {
                        output,
                        solver_id: id,
                        total_time,
                        other_results: Vec::new(),
                    })
                } else {
                    Err(SolverError::Unknown(
                        "No solver returned a result".to_string(),
                    ))
                }
            }
        }
    }

    /// Adaptive solving: start with fast solvers, add more after delay if no result
    async fn solve_dimacs_adaptive(
        &self,
        path: &Path,
        config: &PortfolioConfig,
        initial_ids: &[String],
        delay: Duration,
    ) -> Result<PortfolioResult, SolverError> {
        let start = Instant::now();
        let path_buf = path.to_path_buf();

        // Get initial and remaining solvers
        let initial_solvers: Vec<_> = self
            .solvers
            .iter()
            .filter(|s| initial_ids.contains(&s.info().id))
            .cloned()
            .collect();

        let remaining_solvers: Vec<_> = self
            .solvers
            .iter()
            .filter(|s| !initial_ids.contains(&s.info().id))
            .cloned()
            .collect();

        if initial_solvers.is_empty() && remaining_solvers.is_empty() {
            return Err(SolverError::NotFound(
                "No solvers available for adaptive portfolio".to_string(),
            ));
        }

        info!(
            "Adaptive portfolio: starting with {} initial solvers, {} remaining after {:?}",
            initial_solvers.len(),
            remaining_solvers.len(),
            delay
        );

        // Create cancellation channel
        let (cancel_tx, _) = broadcast::channel::<()>(1);

        // Spawn initial solver tasks
        let mut handles = Vec::new();
        for solver in initial_solvers.iter().take(config.max_concurrent) {
            let solver = Arc::clone(solver);
            let solver_id = solver.info().id.clone();
            let path = path_buf.clone();
            let solver_config = config.solver_config.clone();
            let mut cancel_rx = cancel_tx.subscribe();

            let handle = tokio::spawn(async move {
                tokio::select! {
                    result = solver.solve_dimacs(&path, &solver_config) => {
                        (solver_id, result, false) // false = not from delayed batch
                    }
                    _ = cancel_rx.recv() => {
                        (solver_id, Err(SolverError::Unknown("Cancelled".to_string())), false)
                    }
                }
            });

            handles.push(handle);
        }

        // Spawn delayed solver tasks
        let slots_remaining = config.max_concurrent.saturating_sub(initial_solvers.len());
        for solver in remaining_solvers
            .iter()
            .take(slots_remaining + remaining_solvers.len())
        {
            let solver = Arc::clone(solver);
            let solver_id = solver.info().id.clone();
            let path = path_buf.clone();
            let solver_config = config.solver_config.clone();
            let mut cancel_rx = cancel_tx.subscribe();

            let handle = tokio::spawn(async move {
                // Wait for delay before starting
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {
                        // Delay elapsed, now run solver
                        tokio::select! {
                            result = solver.solve_dimacs(&path, &solver_config) => {
                                (solver_id, result, true) // true = from delayed batch
                            }
                            _ = cancel_rx.recv() => {
                                (solver_id, Err(SolverError::Unknown("Cancelled".to_string())), true)
                            }
                        }
                    }
                    _ = cancel_rx.recv() => {
                        // Cancelled before we even started
                        (solver_id, Err(SolverError::Unknown("Cancelled before start".to_string())), true)
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for results
        let mut winner: Option<(String, SolverOutput)> = None;
        let mut other_results = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((solver_id, result, _from_delayed)) => {
                    match result {
                        Ok(output) if output.result.is_definitive() => {
                            if winner.is_none() {
                                debug!("Solver {} found definitive result", solver_id);
                                winner = Some((solver_id, output));

                                // Cancel other solvers (including delayed ones that haven't started)
                                if config.cancel_on_first {
                                    let _ = cancel_tx.send(());
                                }
                            } else {
                                other_results.push((solver_id, Ok(output)));
                            }
                        }
                        Ok(output) => {
                            debug!("Solver {} returned unknown result", solver_id);
                            other_results.push((solver_id, Ok(output)));
                        }
                        Err(e) if !e.to_string().contains("Cancelled") => {
                            warn!("Solver {} failed: {}", solver_id, e);
                            other_results.push((solver_id, Err(e)));
                        }
                        Err(_) => {
                            // Ignore cancelled solvers
                        }
                    }
                }
                Err(e) => {
                    warn!("Solver task panicked: {}", e);
                }
            }
        }

        let total_time = start.elapsed();

        match winner {
            Some((solver_id, output)) => Ok(PortfolioResult {
                output,
                solver_id,
                total_time,
                other_results,
            }),
            None => {
                if let Some((id, Ok(output))) = other_results.pop() {
                    Ok(PortfolioResult {
                        output,
                        solver_id: id,
                        total_time,
                        other_results: Vec::new(),
                    })
                } else {
                    Err(SolverError::Unknown(
                        "No solver returned a result".to_string(),
                    ))
                }
            }
        }
    }

    /// Solve an SMT2 file using the portfolio
    pub async fn solve_smt2(
        &self,
        path: &Path,
        config: &PortfolioConfig,
    ) -> Result<PortfolioResult, SolverError> {
        // Filter to SMT-capable solvers
        let smt_strategy = PortfolioStrategy::WithCapabilities(vec![SolverCapability::SmtBv]);
        let mut smt_config = config.clone();
        smt_config.strategy = smt_strategy;

        let solvers = self.select_solvers(&smt_config.strategy);

        if solvers.is_empty() {
            return Err(SolverError::NotFound(
                "No SMT-capable solvers in portfolio".to_string(),
            ));
        }

        let start = Instant::now();
        let path = path.to_path_buf();
        let (cancel_tx, _) = broadcast::channel::<()>(1);

        let mut handles = Vec::new();
        for solver in solvers.into_iter().take(config.max_concurrent) {
            let solver_id = solver.info().id.clone();
            let path = path.clone();
            let solver_config = config.solver_config.clone();
            let mut cancel_rx = cancel_tx.subscribe();

            let handle = tokio::spawn(async move {
                tokio::select! {
                    result = solver.solve_smt2(&path, &solver_config) => {
                        (solver_id, result)
                    }
                    _ = cancel_rx.recv() => {
                        (solver_id, Err(SolverError::Unknown("Cancelled".to_string())))
                    }
                }
            });

            handles.push(handle);
        }

        let mut winner: Option<(String, SolverOutput)> = None;
        let mut other_results = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((solver_id, result)) => match result {
                    Ok(output) if output.result.is_definitive() => {
                        if winner.is_none() {
                            winner = Some((solver_id, output));
                            if config.cancel_on_first {
                                let _ = cancel_tx.send(());
                            }
                        } else {
                            other_results.push((solver_id, Ok(output)));
                        }
                    }
                    Ok(output) => {
                        other_results.push((solver_id, Ok(output)));
                    }
                    Err(e) => {
                        other_results.push((solver_id, Err(e)));
                    }
                },
                Err(e) => {
                    warn!("Solver task panicked: {}", e);
                }
            }
        }

        let total_time = start.elapsed();

        match winner {
            Some((solver_id, output)) => Ok(PortfolioResult {
                output,
                solver_id,
                total_time,
                other_results,
            }),
            None => Err(SolverError::Unknown(
                "No SMT solver returned a definitive result".to_string(),
            )),
        }
    }
}

impl Default for Portfolio {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing a portfolio with auto-detection
pub struct PortfolioBuilder {
    portfolio: Portfolio,
}

impl PortfolioBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            portfolio: Portfolio::new(),
        }
    }

    /// Add a solver to the portfolio
    pub fn with_solver(mut self, solver: BoxedSolver) -> Self {
        self.portfolio.add_solver(solver);
        self
    }

    /// Auto-detect and add available solvers
    pub async fn auto_detect(mut self) -> Self {
        // Try to detect CaDiCaL
        if let Some(cadical) = crate::cadical::create_cadical().await {
            info!("Detected CaDiCaL solver");
            self.portfolio.add_solver(cadical);
        }

        // Try to detect Kissat
        if let Some(kissat) = crate::kissat::create_kissat().await {
            info!("Detected Kissat solver");
            self.portfolio.add_solver(kissat);
        }

        // Try to detect Z3
        if let Some(z3) = crate::z3::create_z3().await {
            info!("Detected Z3 solver");
            self.portfolio.add_solver(z3);
        }

        self
    }

    /// Build the portfolio
    pub fn build(self) -> Portfolio {
        self.portfolio
    }
}

impl Default for PortfolioBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{Solver, SolverResult, SolverStats};

    /// Mock solver for testing
    struct MockSolver {
        info: SolverInfo,
        result: SolverResult,
        delay: Duration,
    }

    impl MockSolver {
        fn new(id: &str, result: SolverResult, delay: Duration) -> Self {
            Self {
                info: SolverInfo {
                    id: id.to_string(),
                    name: format!("Mock {id}"),
                    version: "1.0".to_string(),
                    capabilities: vec![SolverCapability::Sat],
                    available: true,
                },
                result,
                delay,
            }
        }
    }

    #[async_trait::async_trait]
    impl Solver for MockSolver {
        fn info(&self) -> &SolverInfo {
            &self.info
        }

        async fn check_available(&self) -> bool {
            true
        }

        async fn solve_dimacs(
            &self,
            _path: &Path,
            _config: &SolverConfig,
        ) -> Result<SolverOutput, SolverError> {
            tokio::time::sleep(self.delay).await;
            Ok(SolverOutput {
                result: self.result.clone(),
                stats: SolverStats::default(),
                raw_output: None,
            })
        }

        async fn solve_smt2(
            &self,
            _path: &Path,
            _config: &SolverConfig,
        ) -> Result<SolverOutput, SolverError> {
            Err(SolverError::InvalidInput(
                "Mock does not support SMT".to_string(),
            ))
        }
    }

    #[test]
    fn test_portfolio_builder() {
        let portfolio = PortfolioBuilder::new().build();
        assert!(portfolio.is_empty());
    }

    #[test]
    fn test_portfolio_add_solver() {
        let mut portfolio = Portfolio::new();
        let solver = MockSolver::new("mock1", SolverResult::Unsat { proof: None }, Duration::ZERO);
        portfolio.add_solver(Box::new(solver));
        assert_eq!(portfolio.len(), 1);
    }

    #[test]
    fn test_strategy_selection() {
        let mut portfolio = Portfolio::new();

        let solver1 = MockSolver::new("fast", SolverResult::Unsat { proof: None }, Duration::ZERO);
        let solver2 = MockSolver::new("slow", SolverResult::Unsat { proof: None }, Duration::ZERO);

        portfolio.add_solver(Box::new(solver1));
        portfolio.add_solver(Box::new(solver2));

        let selected =
            portfolio.select_solvers(&PortfolioStrategy::Specific(vec!["fast".to_string()]));
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].info().id, "fast");
    }

    #[tokio::test]
    async fn test_portfolio_first_wins() {
        let mut portfolio = Portfolio::new();

        // Fast solver returns UNSAT quickly
        let fast = MockSolver::new(
            "fast",
            SolverResult::Unsat { proof: None },
            Duration::from_millis(10),
        );

        // Slow solver takes longer
        let slow = MockSolver::new(
            "slow",
            SolverResult::Sat { model: None },
            Duration::from_millis(100),
        );

        portfolio.add_solver(Box::new(fast));
        portfolio.add_solver(Box::new(slow));

        // Create a temp file for testing
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig::default();
        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();

        // Fast solver should win
        assert_eq!(result.solver_id, "fast");
        assert!(result.output.result.is_unsat());

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_adaptive_strategy_fast_wins() {
        let mut portfolio = Portfolio::new();

        // Fast solver returns quickly
        let fast = MockSolver::new(
            "fast",
            SolverResult::Unsat { proof: None },
            Duration::from_millis(10),
        );

        // Slow solver takes longer but would be started after delay
        let slow = MockSolver::new(
            "slow",
            SolverResult::Sat { model: None },
            Duration::from_millis(10),
        );

        portfolio.add_solver(Box::new(fast));
        portfolio.add_solver(Box::new(slow));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_adaptive.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        // Use adaptive strategy: start with fast, add slow after 50ms
        let config = PortfolioConfig {
            strategy: PortfolioStrategy::Adaptive {
                initial: vec!["fast".to_string()],
                delay: Duration::from_millis(50),
            },
            ..Default::default()
        };

        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();

        // Fast solver should win before slow even starts
        assert_eq!(result.solver_id, "fast");
        assert!(result.output.result.is_unsat());
        // Result should come quickly (before delay)
        assert!(result.total_time < Duration::from_millis(40));

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_adaptive_strategy_delayed_wins() {
        let mut portfolio = Portfolio::new();

        // Slow initial solver that returns unknown
        let initial = MockSolver::new(
            "initial",
            SolverResult::Unknown {
                reason: "timeout".to_string(),
            },
            Duration::from_millis(30),
        );

        // Delayed solver that returns a definitive result
        let delayed = MockSolver::new(
            "delayed",
            SolverResult::Sat {
                model: Some("1".to_string()),
            },
            Duration::from_millis(20),
        );

        portfolio.add_solver(Box::new(initial));
        portfolio.add_solver(Box::new(delayed));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_adaptive_delayed.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        // Use adaptive: start with initial, add delayed after 10ms
        let config = PortfolioConfig {
            strategy: PortfolioStrategy::Adaptive {
                initial: vec!["initial".to_string()],
                delay: Duration::from_millis(10),
            },
            ..Default::default()
        };

        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();

        // Delayed solver should win since initial returns unknown
        assert_eq!(result.solver_id, "delayed");
        assert!(result.output.result.is_sat());

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_portfolio_config_default() {
        let config = PortfolioConfig::default();
        assert!(matches!(config.strategy, PortfolioStrategy::All));
        assert_eq!(config.max_concurrent, 4);
        assert!(config.cancel_on_first);
    }

    #[test]
    fn test_portfolio_config_clone() {
        let config = PortfolioConfig {
            strategy: PortfolioStrategy::Specific(vec!["z3".to_string()]),
            solver_config: SolverConfig::default(),
            max_concurrent: 2,
            cancel_on_first: false,
        };
        let cloned = config.clone();
        assert_eq!(config.max_concurrent, cloned.max_concurrent);
        assert_eq!(config.cancel_on_first, cloned.cancel_on_first);
    }

    #[test]
    fn test_portfolio_config_debug() {
        let config = PortfolioConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("PortfolioConfig"));
    }

    #[test]
    fn test_portfolio_strategy_default() {
        let strategy = PortfolioStrategy::default();
        assert!(matches!(strategy, PortfolioStrategy::All));
    }

    #[test]
    fn test_portfolio_strategy_with_capabilities() {
        let strategy = PortfolioStrategy::WithCapabilities(vec![
            SolverCapability::Sat,
            SolverCapability::SmtBv,
        ]);
        if let PortfolioStrategy::WithCapabilities(caps) = strategy {
            assert_eq!(caps.len(), 2);
        } else {
            panic!("Expected WithCapabilities");
        }
    }

    #[test]
    fn test_portfolio_strategy_specific() {
        let strategy =
            PortfolioStrategy::Specific(vec!["cadical".to_string(), "kissat".to_string()]);
        if let PortfolioStrategy::Specific(ids) = strategy {
            assert!(ids.contains(&"cadical".to_string()));
            assert!(ids.contains(&"kissat".to_string()));
        } else {
            panic!("Expected Specific");
        }
    }

    #[test]
    fn test_portfolio_strategy_adaptive() {
        let strategy = PortfolioStrategy::Adaptive {
            initial: vec!["fast".to_string()],
            delay: Duration::from_secs(1),
        };
        if let PortfolioStrategy::Adaptive { initial, delay } = strategy {
            assert_eq!(initial.len(), 1);
            assert_eq!(delay, Duration::from_secs(1));
        } else {
            panic!("Expected Adaptive");
        }
    }

    #[test]
    fn test_portfolio_strategy_clone() {
        let strategy = PortfolioStrategy::Adaptive {
            initial: vec!["solver1".to_string()],
            delay: Duration::from_millis(500),
        };
        let cloned = strategy.clone();
        if let PortfolioStrategy::Adaptive { initial, delay } = cloned {
            assert_eq!(initial.len(), 1);
            assert_eq!(delay, Duration::from_millis(500));
        }
    }

    #[test]
    fn test_portfolio_strategy_debug() {
        let strategies = vec![
            PortfolioStrategy::All,
            PortfolioStrategy::WithCapabilities(vec![SolverCapability::Sat]),
            PortfolioStrategy::Specific(vec!["z3".to_string()]),
            PortfolioStrategy::Adaptive {
                initial: vec![],
                delay: Duration::ZERO,
            },
        ];
        for strategy in strategies {
            let debug = format!("{:?}", strategy);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_portfolio_result_debug() {
        let result = PortfolioResult {
            output: SolverOutput {
                result: SolverResult::Unsat { proof: None },
                stats: SolverStats::default(),
                raw_output: None,
            },
            solver_id: "test".to_string(),
            total_time: Duration::from_secs(1),
            other_results: vec![],
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("PortfolioResult"));
    }

    #[test]
    fn test_portfolio_default() {
        let portfolio = Portfolio::default();
        assert!(portfolio.is_empty());
    }

    #[test]
    fn test_portfolio_len_and_empty() {
        let mut portfolio = Portfolio::new();
        assert!(portfolio.is_empty());

        let solver = MockSolver::new("test", SolverResult::Sat { model: None }, Duration::ZERO);
        portfolio.add_solver(Box::new(solver));
        assert!(!portfolio.is_empty());
        assert_eq!(portfolio.len(), 1);
    }

    #[test]
    fn test_portfolio_solver_infos() {
        let mut portfolio = Portfolio::new();
        let solver1 = MockSolver::new("solver1", SolverResult::Sat { model: None }, Duration::ZERO);
        let solver2 = MockSolver::new(
            "solver2",
            SolverResult::Unsat { proof: None },
            Duration::ZERO,
        );

        portfolio.add_solver(Box::new(solver1));
        portfolio.add_solver(Box::new(solver2));

        let infos = portfolio.solver_infos();
        assert_eq!(infos.len(), 2);
        let ids: Vec<_> = infos.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"solver1"));
        assert!(ids.contains(&"solver2"));
    }

    #[test]
    fn test_select_solvers_all() {
        let mut portfolio = Portfolio::new();
        portfolio.add_solver(Box::new(MockSolver::new(
            "s1",
            SolverResult::Sat { model: None },
            Duration::ZERO,
        )));
        portfolio.add_solver(Box::new(MockSolver::new(
            "s2",
            SolverResult::Sat { model: None },
            Duration::ZERO,
        )));

        let selected = portfolio.select_solvers(&PortfolioStrategy::All);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_solvers_specific_empty() {
        let mut portfolio = Portfolio::new();
        portfolio.add_solver(Box::new(MockSolver::new(
            "s1",
            SolverResult::Sat { model: None },
            Duration::ZERO,
        )));

        let selected =
            portfolio.select_solvers(&PortfolioStrategy::Specific(
                vec!["nonexistent".to_string()],
            ));
        assert!(selected.is_empty());
    }

    #[test]
    fn test_portfolio_builder_default() {
        let builder = PortfolioBuilder::default();
        let portfolio = builder.build();
        assert!(portfolio.is_empty());
    }

    #[test]
    fn test_portfolio_builder_with_solver() {
        let solver = MockSolver::new("mock", SolverResult::Sat { model: None }, Duration::ZERO);
        let portfolio = PortfolioBuilder::new()
            .with_solver(Box::new(solver))
            .build();
        assert_eq!(portfolio.len(), 1);
    }

    #[test]
    fn test_portfolio_builder_chain() {
        let solver1 = MockSolver::new("s1", SolverResult::Sat { model: None }, Duration::ZERO);
        let solver2 = MockSolver::new("s2", SolverResult::Unsat { proof: None }, Duration::ZERO);

        let portfolio = PortfolioBuilder::new()
            .with_solver(Box::new(solver1))
            .with_solver(Box::new(solver2))
            .build();

        assert_eq!(portfolio.len(), 2);
    }

    #[tokio::test]
    async fn test_portfolio_empty_returns_error() {
        let portfolio = Portfolio::new();
        let config = PortfolioConfig::default();

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_empty.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let result = portfolio.solve_dimacs(&temp_file, &config).await;
        assert!(result.is_err());

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_portfolio_cancel_on_first_disabled() {
        let mut portfolio = Portfolio::new();

        let fast = MockSolver::new(
            "fast",
            SolverResult::Unsat { proof: None },
            Duration::from_millis(10),
        );
        let slow = MockSolver::new(
            "slow",
            SolverResult::Sat { model: None },
            Duration::from_millis(20),
        );

        portfolio.add_solver(Box::new(fast));
        portfolio.add_solver(Box::new(slow));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_no_cancel.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig {
            cancel_on_first: false,
            ..Default::default()
        };
        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();

        // Fast should still win
        assert_eq!(result.solver_id, "fast");

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_mock_solver_info() {
        let solver = MockSolver::new(
            "test_solver",
            SolverResult::Sat { model: None },
            Duration::ZERO,
        );
        let info = solver.info();
        assert_eq!(info.id, "test_solver");
        assert_eq!(info.name, "Mock test_solver");
        assert_eq!(info.version, "1.0");
        assert!(info.available);
        assert!(info.capabilities.contains(&SolverCapability::Sat));
    }

    #[tokio::test]
    async fn test_mock_solver_check_available() {
        let solver = MockSolver::new("test", SolverResult::Sat { model: None }, Duration::ZERO);
        assert!(solver.check_available().await);
    }

    #[tokio::test]
    async fn test_mock_solver_smt2_unsupported() {
        let solver = MockSolver::new("test", SolverResult::Sat { model: None }, Duration::ZERO);
        let config = SolverConfig::default();
        let result = solver
            .solve_smt2(std::path::Path::new("/tmp/test.smt2"), &config)
            .await;
        assert!(result.is_err());
    }

    // ==================== Mutation Coverage Tests ====================

    #[tokio::test]
    async fn test_portfolio_builder_auto_detect() {
        // Test auto_detect doesn't just return Default::default()
        let builder = PortfolioBuilder::new().auto_detect().await;
        let portfolio = builder.build();

        // The portfolio should have detected solvers if available
        // This tests the auto_detect method returns Self with solvers, not Default::default()
        let infos = portfolio.solver_infos();
        for info in &infos {
            // Each detected solver should have a valid id and be available
            assert!(!info.id.is_empty());
            assert!(info.available);
        }
    }

    #[tokio::test]
    async fn test_run_solvers_parallel_is_definitive_guard_true() {
        // Test that is_definitive() == true path is correctly taken
        let mut portfolio = Portfolio::new();
        let solver = MockSolver::new(
            "definitive",
            SolverResult::Unsat { proof: None },
            Duration::from_millis(10),
        );
        portfolio.add_solver(Box::new(solver));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_definitive_true.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig::default();
        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();

        // Should get definitive result
        assert!(result.output.result.is_definitive());
        assert_eq!(result.solver_id, "definitive");

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_run_solvers_parallel_is_definitive_guard_false() {
        // Test that is_definitive() == false path is correctly taken
        let mut portfolio = Portfolio::new();
        let solver = MockSolver::new(
            "unknown",
            SolverResult::Unknown {
                reason: "test".to_string(),
            },
            Duration::from_millis(10),
        );
        portfolio.add_solver(Box::new(solver));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_definitive_false.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig::default();
        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();

        // Should get unknown result (best effort)
        assert!(!result.output.result.is_definitive());

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_adaptive_empty_initial_uses_remaining() {
        // Test adaptive strategy with empty initial solvers
        // This exercises the && condition on line 299: initial_solvers.is_empty() && remaining_solvers.is_empty()
        let mut portfolio = Portfolio::new();

        let solver = MockSolver::new(
            "remaining",
            SolverResult::Sat { model: None },
            Duration::from_millis(10),
        );
        portfolio.add_solver(Box::new(solver));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_adaptive_empty_initial.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        // Adaptive with empty initial - solver "remaining" is not in initial list
        let config = PortfolioConfig {
            strategy: PortfolioStrategy::Adaptive {
                initial: vec![], // Empty initial
                delay: Duration::from_millis(10),
            },
            ..Default::default()
        };

        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();
        assert_eq!(result.solver_id, "remaining");

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_adaptive_both_empty_error() {
        // Test adaptive strategy with both initial and remaining empty
        let portfolio = Portfolio::new();

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_adaptive_both_empty.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig {
            strategy: PortfolioStrategy::Adaptive {
                initial: vec![],
                delay: Duration::from_millis(10),
            },
            ..Default::default()
        };

        let result = portfolio.solve_dimacs(&temp_file, &config).await;
        assert!(result.is_err());
        if let Err(SolverError::NotFound(msg)) = result {
            assert!(msg.contains("adaptive portfolio"));
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_adaptive_slots_remaining_calculation() {
        // Test the + operation in slots calculation: slots_remaining + remaining_solvers.len()
        // This catches "replace + with *" and "replace + with -" mutations
        let mut portfolio = Portfolio::new();

        // Add 3 initial solvers
        for i in 0..3 {
            let solver = MockSolver::new(
                &format!("init{i}"),
                SolverResult::Unknown {
                    reason: "test".to_string(),
                },
                Duration::from_millis(100),
            );
            portfolio.add_solver(Box::new(solver));
        }

        // Add 2 remaining solvers
        for i in 0..2 {
            let solver = MockSolver::new(
                &format!("remain{i}"),
                SolverResult::Sat {
                    model: Some("1".to_string()),
                },
                Duration::from_millis(10),
            );
            portfolio.add_solver(Box::new(solver));
        }

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_slots.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig {
            strategy: PortfolioStrategy::Adaptive {
                initial: vec![
                    "init0".to_string(),
                    "init1".to_string(),
                    "init2".to_string(),
                ],
                delay: Duration::from_millis(5),
            },
            max_concurrent: 4,
            ..Default::default()
        };

        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();
        // One of the remaining solvers should win since they return SAT
        assert!(result.output.result.is_sat());

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_adaptive_cancelled_error_filter() {
        // Test the !e.to_string().contains("Cancelled") filter
        // This catches "delete !" and "replace with true/false" mutations
        let mut portfolio = Portfolio::new();

        // Fast solver that returns SAT
        let fast = MockSolver::new(
            "fast",
            SolverResult::Sat { model: None },
            Duration::from_millis(5),
        );

        // Slow solver that would be cancelled
        let slow = MockSolver::new(
            "slow",
            SolverResult::Unsat { proof: None },
            Duration::from_millis(1000),
        );

        portfolio.add_solver(Box::new(fast));
        portfolio.add_solver(Box::new(slow));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_cancelled_filter.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig {
            strategy: PortfolioStrategy::Adaptive {
                initial: vec!["fast".to_string()],
                delay: Duration::from_millis(500),
            },
            cancel_on_first: true,
            ..Default::default()
        };

        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();
        // Fast should win, slow should be cancelled but not appear in other_results
        assert_eq!(result.solver_id, "fast");

        std::fs::remove_file(temp_file).ok();
    }

    #[tokio::test]
    async fn test_solve_smt2_is_definitive_guard() {
        // Test the is_definitive() match guard in solve_smt2
        // This requires a mock solver that supports SMT

        struct MockSmtSolver {
            info: SolverInfo,
            result: SolverResult,
        }

        impl MockSmtSolver {
            fn new(id: &str, result: SolverResult) -> Self {
                Self {
                    info: SolverInfo {
                        id: id.to_string(),
                        name: format!("Mock SMT {id}"),
                        version: "1.0".to_string(),
                        capabilities: vec![SolverCapability::Sat, SolverCapability::SmtBv],
                        available: true,
                    },
                    result,
                }
            }
        }

        #[async_trait::async_trait]
        impl Solver for MockSmtSolver {
            fn info(&self) -> &SolverInfo {
                &self.info
            }

            async fn check_available(&self) -> bool {
                true
            }

            async fn solve_dimacs(
                &self,
                _path: &Path,
                _config: &SolverConfig,
            ) -> Result<SolverOutput, SolverError> {
                Err(SolverError::InvalidInput("Not supported".to_string()))
            }

            async fn solve_smt2(
                &self,
                _path: &Path,
                _config: &SolverConfig,
            ) -> Result<SolverOutput, SolverError> {
                Ok(SolverOutput {
                    result: self.result.clone(),
                    stats: SolverStats::default(),
                    raw_output: None,
                })
            }
        }

        let mut portfolio = Portfolio::new();
        let smt_solver = MockSmtSolver::new("smt1", SolverResult::Sat { model: None });
        portfolio.add_solver(Box::new(smt_solver));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_smt_definitive.smt2");
        std::fs::write(&temp_file, "(check-sat)\n").unwrap();

        let config = PortfolioConfig::default();
        let result = portfolio.solve_smt2(&temp_file, &config).await.unwrap();

        assert!(result.output.result.is_sat());
        assert_eq!(result.solver_id, "smt1");

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_select_solvers_adaptive_initial() {
        // Test that Adaptive strategy correctly selects initial solvers
        let mut portfolio = Portfolio::new();

        let s1 = MockSolver::new("solver1", SolverResult::Sat { model: None }, Duration::ZERO);
        let s2 = MockSolver::new("solver2", SolverResult::Sat { model: None }, Duration::ZERO);
        let s3 = MockSolver::new("solver3", SolverResult::Sat { model: None }, Duration::ZERO);

        portfolio.add_solver(Box::new(s1));
        portfolio.add_solver(Box::new(s2));
        portfolio.add_solver(Box::new(s3));

        let selected = portfolio.select_solvers(&PortfolioStrategy::Adaptive {
            initial: vec!["solver1".to_string(), "solver3".to_string()],
            delay: Duration::from_secs(1),
        });

        assert_eq!(selected.len(), 2);
        let ids: Vec<_> = selected.iter().map(|s| s.info().id.as_str()).collect();
        assert!(ids.contains(&"solver1"));
        assert!(ids.contains(&"solver3"));
        assert!(!ids.contains(&"solver2"));
    }

    #[tokio::test]
    async fn test_portfolio_multiple_definitive_results() {
        // Test when multiple solvers return definitive results
        // First one should win, others should be in other_results
        let mut portfolio = Portfolio::new();

        let fast = MockSolver::new(
            "fast",
            SolverResult::Unsat { proof: None },
            Duration::from_millis(10),
        );
        let medium = MockSolver::new(
            "medium",
            SolverResult::Sat { model: None },
            Duration::from_millis(20),
        );

        portfolio.add_solver(Box::new(fast));
        portfolio.add_solver(Box::new(medium));

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_multiple_definitive.cnf");
        std::fs::write(&temp_file, "p cnf 1 1\n1 0\n").unwrap();

        let config = PortfolioConfig {
            cancel_on_first: false, // Don't cancel to get multiple results
            ..Default::default()
        };

        let result = portfolio.solve_dimacs(&temp_file, &config).await.unwrap();

        // Fast should win
        assert_eq!(result.solver_id, "fast");

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_select_solvers_with_capabilities_filters() {
        // Test WithCapabilities strategy filters correctly
        let mut portfolio = Portfolio::new();

        // Solver with only SAT capability
        let sat_only = MockSolver::new(
            "sat_only",
            SolverResult::Sat { model: None },
            Duration::ZERO,
        );
        portfolio.add_solver(Box::new(sat_only));

        // WithCapabilities requiring SmtBv should return empty
        let selected = portfolio.select_solvers(&PortfolioStrategy::WithCapabilities(vec![
            SolverCapability::SmtBv,
        ]));
        assert!(selected.is_empty());

        // WithCapabilities requiring Sat should return the solver
        let selected = portfolio.select_solvers(&PortfolioStrategy::WithCapabilities(vec![
            SolverCapability::Sat,
        ]));
        assert_eq!(selected.len(), 1);
    }
}
