//! Portfolio solver management for Kani Fast
//!
//! This crate provides parallel solver execution for SAT/SMT problems.
//! By running multiple solvers simultaneously, we can leverage the fact
//! that different solvers excel on different problem types.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Portfolio                               │
//! │                                                             │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
//! │  │ CaDiCaL  │  │ Kissat   │  │   Z3     │  │  ...     │   │
//! │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
//! │       │             │             │             │         │
//! │       └─────────────┴──────┬──────┴─────────────┘         │
//! │                            │                               │
//! │                     First Result                           │
//! └────────────────────────────┴───────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_portfolio::{Portfolio, PortfolioBuilder, PortfolioConfig};
//!
//! // Auto-detect available solvers
//! let portfolio = PortfolioBuilder::new()
//!     .auto_detect()
//!     .await
//!     .build();
//!
//! // Solve a DIMACS file
//! let config = PortfolioConfig::default();
//! let result = portfolio.solve_dimacs(&path, &config).await?;
//!
//! println!("Solved by: {}", result.solver_id);
//! ```

mod cadical;
mod drat;
mod kissat;
mod portfolio;
mod solver;
mod z3;

use std::env;
use std::path::PathBuf;

/// Find an executable in PATH (used instead of the which crate)
///
/// This utility function searches the system PATH for the given executable name.
/// On Windows, it also tries appending `.exe` to the name.
///
/// # Arguments
/// * `name` - The name of the executable to find (e.g., "z3", "cargo")
///
/// # Returns
/// * `Some(PathBuf)` - The full path to the executable if found
/// * `None` - If the executable is not found in PATH
///
/// # Example
/// ```
/// use kani_fast_portfolio::find_executable;
///
/// if let Some(path) = find_executable("cargo") {
///     println!("Found cargo at: {}", path.display());
/// }
/// ```
pub fn find_executable(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths).find_map(|dir| {
            let full_path = dir.join(name);
            if full_path.is_file() {
                Some(full_path)
            } else {
                // Try with .exe on Windows
                #[cfg(windows)]
                {
                    let with_exe = dir.join(format!("{}.exe", name));
                    if with_exe.is_file() {
                        return Some(with_exe);
                    }
                }
                None
            }
        })
    })
}

pub use solver::{
    util, BoxedSolver, Solver, SolverCapability, SolverConfig, SolverError, SolverInfo,
    SolverOutput, SolverResult, SolverStats,
};

pub use cadical::{create_cadical, CaDiCaL, LearnConfig, LearnedClausesResult};
pub use kissat::{create_kissat, Kissat};
pub use z3::{create_z3, Z3};

pub use portfolio::{
    Portfolio, PortfolioBuilder, PortfolioConfig, PortfolioResult, PortfolioStrategy,
};

pub use drat::{
    extract_learned_clauses, filter_learned_clauses, DratClause, DratError, DratParser, DratStats,
};
