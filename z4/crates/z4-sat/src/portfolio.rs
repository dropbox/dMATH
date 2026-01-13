//! Parallel portfolio solver
//!
//! Runs multiple solver configurations in parallel and returns the first result.
//! This is the standard approach for robust SAT solving - different heuristics
//! work better on different problem classes.
//!
//! ## Strategies
//!
//! The portfolio includes diverse strategies:
//! - VSIDS with Luby restarts (classic MiniSat-style)
//! - VSIDS with Glucose restarts (LBD-based)
//! - Aggressive inprocessing (vivification, BVE, BCE)
//! - Conservative (minimal preprocessing, stable search)
//!
//! ## Usage
//!
//! ```ignore
//! use z4_sat::portfolio::{PortfolioSolver, Strategy};
//!
//! let formula = z4_sat::parse_dimacs(cnf_string)?;
//! let result = PortfolioSolver::new(4) // 4 threads
//!     .strategies(vec![
//!         Strategy::default_vsids_luby(),
//!         Strategy::default_vsids_glucose(),
//!         Strategy::aggressive_inprocessing(),
//!         Strategy::conservative(),
//!     ])
//!     .solve(&formula);
//! ```

use crate::dimacs::DimacsFormula;
use crate::literal::Literal;
use crate::solver::{AssumeResult, SolveResult, Solver};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// Configuration for a solver instance
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Name of this configuration (for logging)
    pub name: String,
    /// Whether to use Glucose-style EMA restarts (vs Luby)
    pub glucose_restarts: bool,
    /// Whether to enable chronological backtracking
    pub chrono_enabled: bool,
    /// Whether to enable vivification
    pub vivify_enabled: bool,
    /// Whether to enable subsumption
    pub subsume_enabled: bool,
    /// Whether to enable failed literal probing
    pub probe_enabled: bool,
    /// Whether to enable bounded variable elimination
    pub bve_enabled: bool,
    /// Whether to enable blocked clause elimination
    pub bce_enabled: bool,
    /// Whether to enable hyper-ternary resolution
    pub htr_enabled: bool,
    /// Whether to enable gate extraction
    pub gate_enabled: bool,
    /// Whether to enable SAT sweeping
    pub sweep_enabled: bool,
    /// Initial phase for variables (true = positive, false = negative, None = phase saving)
    pub initial_phase: Option<bool>,
    /// Random seed for tie-breaking in variable selection
    pub seed: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            name: "default".to_string(),
            glucose_restarts: true,
            chrono_enabled: true,
            vivify_enabled: true,
            subsume_enabled: true,
            probe_enabled: true,
            bve_enabled: true,
            bce_enabled: true,
            htr_enabled: true,
            gate_enabled: true,
            sweep_enabled: true,
            initial_phase: None,
            seed: 0,
        }
    }
}

/// Predefined solver strategies for portfolio solving
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Classic VSIDS with Luby restarts (MiniSat-style)
    VsidsLuby,
    /// VSIDS with Glucose-style EMA restarts
    VsidsGlucose,
    /// Aggressive inprocessing (all techniques enabled)
    AggressiveInprocessing,
    /// Conservative search (minimal preprocessing)
    Conservative,
    /// Focus on failed literal probing
    ProbeFocused,
    /// Focus on variable elimination
    BveFocused,
}

impl Strategy {
    /// Convert strategy to solver configuration
    pub fn to_config(self) -> SolverConfig {
        match self {
            Strategy::VsidsLuby => SolverConfig {
                name: "vsids-luby".to_string(),
                glucose_restarts: false,
                chrono_enabled: true,
                vivify_enabled: true,
                subsume_enabled: true,
                probe_enabled: true,
                bve_enabled: true,
                bce_enabled: true,
                htr_enabled: true,
                gate_enabled: true,
                sweep_enabled: true,
                initial_phase: None,
                seed: 0,
            },
            Strategy::VsidsGlucose => SolverConfig {
                name: "vsids-glucose".to_string(),
                glucose_restarts: true,
                chrono_enabled: true,
                vivify_enabled: true,
                subsume_enabled: true,
                probe_enabled: true,
                bve_enabled: true,
                bce_enabled: true,
                htr_enabled: true,
                gate_enabled: true,
                sweep_enabled: true,
                initial_phase: None,
                seed: 1,
            },
            Strategy::AggressiveInprocessing => SolverConfig {
                name: "aggressive".to_string(),
                glucose_restarts: true,
                chrono_enabled: true,
                vivify_enabled: true,
                subsume_enabled: true,
                probe_enabled: true,
                bve_enabled: true,
                bce_enabled: true,
                htr_enabled: true,
                gate_enabled: true,
                sweep_enabled: true,
                initial_phase: None,
                seed: 2,
            },
            Strategy::Conservative => SolverConfig {
                name: "conservative".to_string(),
                glucose_restarts: false,
                chrono_enabled: false,
                vivify_enabled: false,
                subsume_enabled: false,
                probe_enabled: false,
                bve_enabled: false,
                bce_enabled: false,
                htr_enabled: false,
                gate_enabled: false,
                sweep_enabled: false,
                initial_phase: None,
                seed: 3,
            },
            Strategy::ProbeFocused => SolverConfig {
                name: "probe-focused".to_string(),
                glucose_restarts: true,
                chrono_enabled: true,
                vivify_enabled: false,
                subsume_enabled: true,
                probe_enabled: true,
                bve_enabled: false,
                bce_enabled: false,
                htr_enabled: false,
                gate_enabled: false,
                sweep_enabled: false,
                initial_phase: Some(false), // Start with negative phase
                seed: 4,
            },
            Strategy::BveFocused => SolverConfig {
                name: "bve-focused".to_string(),
                glucose_restarts: true,
                chrono_enabled: true,
                vivify_enabled: false,
                subsume_enabled: true,
                probe_enabled: false,
                bve_enabled: true,
                bce_enabled: true,
                htr_enabled: false,
                gate_enabled: true,
                sweep_enabled: false,
                initial_phase: Some(true), // Start with positive phase
                seed: 5,
            },
        }
    }

    /// Get all predefined strategies
    pub fn all() -> Vec<Strategy> {
        vec![
            Strategy::VsidsLuby,
            Strategy::VsidsGlucose,
            Strategy::AggressiveInprocessing,
            Strategy::Conservative,
            Strategy::ProbeFocused,
            Strategy::BveFocused,
        ]
    }

    /// Get the recommended subset of strategies for a given thread count
    pub fn recommended(num_threads: usize) -> Vec<Strategy> {
        match num_threads {
            1 => vec![Strategy::VsidsGlucose],
            2 => vec![Strategy::VsidsGlucose, Strategy::VsidsLuby],
            3 => vec![
                Strategy::VsidsGlucose,
                Strategy::VsidsLuby,
                Strategy::AggressiveInprocessing,
            ],
            4 => vec![
                Strategy::VsidsGlucose,
                Strategy::VsidsLuby,
                Strategy::AggressiveInprocessing,
                Strategy::Conservative,
            ],
            _ => {
                let mut strategies = Strategy::all();
                // If we have more threads than strategies, duplicate with different seeds
                while strategies.len() < num_threads {
                    let base = strategies[strategies.len() % 6];
                    strategies.push(base);
                }
                strategies.truncate(num_threads);
                strategies
            }
        }
    }
}

/// Result from a portfolio solver thread
#[derive(Debug)]
struct ThreadResult {
    /// Which strategy produced this result
    #[allow(dead_code)]
    strategy_name: String,
    /// The solve result
    result: SolveResult,
}

/// Parallel portfolio SAT solver
///
/// Runs multiple solver configurations in parallel and returns the first result.
pub struct PortfolioSolver {
    /// Number of threads to use
    num_threads: usize,
    /// Solver configurations (one per thread)
    configs: Vec<SolverConfig>,
}

impl PortfolioSolver {
    /// Create a new portfolio solver with the specified number of threads
    ///
    /// Uses recommended strategies for the thread count.
    pub fn new(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);
        let strategies = Strategy::recommended(num_threads);
        let configs: Vec<SolverConfig> = strategies
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let mut config = s.to_config();
                config.seed = i as u64;
                config
            })
            .collect();

        PortfolioSolver {
            num_threads,
            configs,
        }
    }

    /// Set custom strategies for the portfolio
    pub fn with_strategies(mut self, strategies: Vec<Strategy>) -> Self {
        self.configs = strategies
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let mut config = s.to_config();
                config.seed = i as u64;
                config
            })
            .collect();
        self.num_threads = self.configs.len().max(1);
        self
    }

    /// Set custom configurations for the portfolio
    pub fn with_configs(mut self, configs: Vec<SolverConfig>) -> Self {
        self.configs = configs;
        self.num_threads = self.configs.len().max(1);
        self
    }

    /// Solve a CNF formula in parallel
    ///
    /// Returns the first result found by any thread.
    pub fn solve(&self, formula: &DimacsFormula) -> SolveResult {
        if self.num_threads == 1 || self.configs.len() == 1 {
            // Single-threaded: just run normally
            let config = self.configs.first().cloned().unwrap_or_default();
            let mut solver = create_solver_from_config(formula.num_vars, &config);
            for clause in &formula.clauses {
                solver.add_clause(clause.clone());
            }
            return solver.solve();
        }

        // Multi-threaded portfolio
        let terminate = Arc::new(AtomicBool::new(false));
        let result: Arc<Mutex<Option<ThreadResult>>> = Arc::new(Mutex::new(None));

        let handles: Vec<_> = self
            .configs
            .iter()
            .cloned()
            .map(|config| {
                let formula_clauses = formula.clauses.clone();
                let num_vars = formula.num_vars;
                let terminate = Arc::clone(&terminate);
                let result = Arc::clone(&result);
                let strategy_name = config.name.clone();

                thread::spawn(move || {
                    // Create solver with this configuration
                    let mut solver = create_solver_from_config(num_vars, &config);

                    // Add clauses
                    for clause in &formula_clauses {
                        solver.add_clause(clause.clone());
                    }

                    // Solve with termination check
                    let solve_result =
                        solver.solve_interruptible(|| terminate.load(Ordering::Relaxed));

                    // If we got a result and haven't been terminated, store it
                    if !terminate.load(Ordering::Relaxed) {
                        let mut guard = result.lock();
                        if guard.is_none() {
                            *guard = Some(ThreadResult {
                                strategy_name,
                                result: solve_result,
                            });
                            // Signal other threads to stop
                            terminate.store(true, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        // Wait for all threads to finish
        for handle in handles {
            let _ = handle.join();
        }

        // Extract result
        let guard = result.lock();
        match guard.as_ref() {
            Some(r) => r.result.clone(),
            None => SolveResult::Unknown, // All threads were interrupted
        }
    }

    /// Solve with assumptions in parallel
    ///
    /// Returns the first result found by any thread.
    pub fn solve_with_assumptions(
        &self,
        formula: &DimacsFormula,
        assumptions: &[Literal],
    ) -> AssumeResult {
        if self.num_threads == 1 || self.configs.len() == 1 {
            // Single-threaded: just run normally
            let config = self.configs.first().cloned().unwrap_or_default();
            let mut solver = create_solver_from_config(formula.num_vars, &config);
            for clause in &formula.clauses {
                solver.add_clause(clause.clone());
            }
            return solver.solve_with_assumptions(assumptions);
        }

        // Multi-threaded portfolio
        let terminate = Arc::new(AtomicBool::new(false));
        let result: Arc<Mutex<Option<AssumeResult>>> = Arc::new(Mutex::new(None));

        let handles: Vec<_> = self
            .configs
            .iter()
            .cloned()
            .map(|config| {
                let formula_clauses = formula.clauses.clone();
                let num_vars = formula.num_vars;
                let assumptions = assumptions.to_vec();
                let terminate = Arc::clone(&terminate);
                let result = Arc::clone(&result);

                thread::spawn(move || {
                    // Create solver with this configuration
                    let mut solver = create_solver_from_config(num_vars, &config);

                    // Add clauses
                    for clause in &formula_clauses {
                        solver.add_clause(clause.clone());
                    }

                    // Solve with assumptions (no interruptible version yet)
                    let solve_result = solver.solve_with_assumptions(&assumptions);

                    // If we haven't been terminated, store result
                    if !terminate.load(Ordering::Relaxed) {
                        let mut guard = result.lock();
                        if guard.is_none() {
                            *guard = Some(solve_result);
                            terminate.store(true, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        // Wait for all threads to finish
        for handle in handles {
            let _ = handle.join();
        }

        // Extract result
        let guard = result.lock();
        match guard.as_ref() {
            Some(r) => r.clone(),
            None => AssumeResult::Unknown,
        }
    }

    /// Get the number of threads this portfolio will use
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Get the configurations being used
    pub fn configs(&self) -> &[SolverConfig] {
        &self.configs
    }
}

/// Create a solver instance from a configuration
fn create_solver_from_config(num_vars: usize, config: &SolverConfig) -> Solver {
    let mut solver = Solver::new(num_vars);

    // Apply configuration
    solver.set_glucose_restarts(config.glucose_restarts);
    solver.set_chrono_enabled(config.chrono_enabled);
    solver.set_vivify_enabled(config.vivify_enabled);
    solver.set_subsume_enabled(config.subsume_enabled);
    solver.set_probe_enabled(config.probe_enabled);
    solver.set_bve_enabled(config.bve_enabled);
    solver.set_bce_enabled(config.bce_enabled);
    solver.set_htr_enabled(config.htr_enabled);
    solver.set_gate_enabled(config.gate_enabled);
    solver.set_sweep_enabled(config.sweep_enabled);

    // Set initial phase if specified
    if let Some(phase) = config.initial_phase {
        solver.set_initial_phase(phase);
    }

    // Set random seed for variable selection tie-breaking
    solver.set_random_seed(config.seed);

    solver
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimacs::parse_str;

    #[test]
    fn test_portfolio_sat_simple() {
        let cnf = "p cnf 3 2\n1 2 0\n-1 3 0\n";
        let formula = parse_str(cnf).unwrap();

        let portfolio = PortfolioSolver::new(2);
        let result = portfolio.solve(&formula);

        assert!(matches!(result, SolveResult::Sat(_)));
    }

    #[test]
    fn test_portfolio_unsat_simple() {
        let cnf = "p cnf 1 2\n1 0\n-1 0\n";
        let formula = parse_str(cnf).unwrap();

        let portfolio = PortfolioSolver::new(2);
        let result = portfolio.solve(&formula);

        assert_eq!(result, SolveResult::Unsat);
    }

    #[test]
    fn test_portfolio_strategies() {
        let strategies = Strategy::all();
        assert_eq!(strategies.len(), 6);

        for strategy in strategies {
            let config = strategy.to_config();
            assert!(!config.name.is_empty());
        }
    }

    #[test]
    fn test_portfolio_recommended_threads() {
        // 1 thread should give 1 strategy
        let s1 = Strategy::recommended(1);
        assert_eq!(s1.len(), 1);

        // 4 threads should give 4 strategies
        let s4 = Strategy::recommended(4);
        assert_eq!(s4.len(), 4);

        // 8 threads should give 8 strategies (with duplicates)
        let s8 = Strategy::recommended(8);
        assert_eq!(s8.len(), 8);
    }

    #[test]
    fn test_portfolio_single_thread_fallback() {
        let cnf = "p cnf 3 3\n1 2 0\n-1 2 0\n-2 3 0\n";
        let formula = parse_str(cnf).unwrap();

        // Single thread should work
        let portfolio = PortfolioSolver::new(1);
        let result = portfolio.solve(&formula);

        assert!(matches!(result, SolveResult::Sat(_)));
    }

    #[test]
    fn test_portfolio_with_custom_config() {
        let cnf = "p cnf 2 2\n1 2 0\n-1 -2 0\n";
        let formula = parse_str(cnf).unwrap();

        let config = SolverConfig {
            name: "custom".to_string(),
            glucose_restarts: false,
            chrono_enabled: false,
            vivify_enabled: false,
            subsume_enabled: false,
            probe_enabled: false,
            bve_enabled: false,
            bce_enabled: false,
            htr_enabled: false,
            gate_enabled: false,
            sweep_enabled: false,
            initial_phase: Some(true),
            seed: 42,
        };

        let portfolio = PortfolioSolver::new(1).with_configs(vec![config]);
        let result = portfolio.solve(&formula);

        assert!(matches!(result, SolveResult::Sat(_)));
    }
}
