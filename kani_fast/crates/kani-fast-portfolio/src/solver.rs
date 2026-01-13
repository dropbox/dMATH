//! Solver trait abstraction for portfolio solving
//!
//! This module defines the unified interface that all SAT/SMT solvers must implement.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during solving
#[derive(Debug, Error, Clone)]
pub enum SolverError {
    #[error("Solver not found: {0}")]
    NotFound(String),

    #[error("Solver execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Solver timed out after {0:?}")]
    Timeout(Duration),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Solver returned unknown result: {0}")]
    Unknown(String),
}

/// Result of a SAT/SMT solving attempt
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SolverResult {
    /// Formula is satisfiable (property violated, counterexample exists)
    Sat {
        /// Assignment to variables (solver-specific format)
        model: Option<String>,
    },

    /// Formula is unsatisfiable (property holds)
    Unsat {
        /// Proof or unsat core if available
        proof: Option<String>,
    },

    /// Solver could not determine satisfiability
    Unknown { reason: String },
}

impl SolverResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, Self::Sat { .. })
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, Self::Unsat { .. })
    }

    pub fn is_definitive(&self) -> bool {
        matches!(self, Self::Sat { .. } | Self::Unsat { .. })
    }
}

/// Statistics from a solver run
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStats {
    /// Wall-clock time spent solving
    pub solve_time: Duration,

    /// Number of conflicts (SAT solvers)
    pub conflicts: Option<u64>,

    /// Number of decisions (SAT solvers)
    pub decisions: Option<u64>,

    /// Number of propagations (SAT solvers)
    pub propagations: Option<u64>,

    /// Peak memory usage in bytes
    pub memory_bytes: Option<u64>,

    /// Solver-specific additional stats
    pub extra: std::collections::HashMap<String, String>,
}

/// Configuration for a solver run
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum time for solving
    pub timeout: Duration,

    /// Random seed for solver
    pub seed: Option<u64>,

    /// Number of threads (for parallel solvers)
    pub threads: Option<usize>,

    /// Solver-specific options
    pub options: std::collections::HashMap<String, String>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300), // 5 minutes default
            seed: None,
            threads: None,
            options: std::collections::HashMap::new(),
        }
    }
}

/// Output from a completed solver run
#[derive(Debug, Clone)]
pub struct SolverOutput {
    /// The solving result
    pub result: SolverResult,

    /// Statistics from the run
    pub stats: SolverStats,

    /// Raw solver output (for debugging)
    pub raw_output: Option<String>,
}

/// Solver capabilities for portfolio selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverCapability {
    /// Pure SAT solving (CNF)
    Sat,
    /// SMT with bitvectors
    SmtBv,
    /// SMT with arrays
    SmtArrays,
    /// SMT with uninterpreted functions
    SmtUf,
    /// SMT with linear integer arithmetic
    SmtLia,
    /// SMT with linear real arithmetic
    SmtLra,
    /// SMT with nonlinear arithmetic
    SmtNla,
    /// Quantifier support
    Quantifiers,
    /// Incremental solving
    Incremental,
    /// Proof production
    Proofs,
    /// Unsat core extraction
    UnsatCores,
}

/// Solver identity and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverInfo {
    /// Unique identifier for this solver
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Version string
    pub version: String,

    /// Capabilities this solver supports
    pub capabilities: Vec<SolverCapability>,

    /// Whether this solver is currently available
    pub available: bool,
}

/// Trait that all solvers must implement
#[async_trait]
pub trait Solver: Send + Sync {
    /// Get information about this solver
    fn info(&self) -> &SolverInfo;

    /// Check if this solver is available on the system
    async fn check_available(&self) -> bool;

    /// Solve a CNF formula from a DIMACS file
    async fn solve_dimacs(
        &self,
        path: &Path,
        config: &SolverConfig,
    ) -> Result<SolverOutput, SolverError>;

    /// Solve an SMT formula from an SMT-LIB2 file
    async fn solve_smt2(
        &self,
        path: &Path,
        config: &SolverConfig,
    ) -> Result<SolverOutput, SolverError>;

    /// Check if this solver supports a capability
    fn supports(&self, capability: SolverCapability) -> bool {
        self.info().capabilities.contains(&capability)
    }
}

/// A boxed solver for dynamic dispatch
pub type BoxedSolver = Box<dyn Solver>;

/// Utility functions shared across solver implementations
pub mod util {
    /// Extract a number from a statistics line.
    ///
    /// Parses lines like "c conflicts: 12345" to extract 12345.
    pub fn extract_number(line: &str) -> Option<u64> {
        line.split_whitespace().find_map(|word| {
            word.trim_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .ok()
        })
    }

    /// Extract model (variable assignments) from SAT output.
    ///
    /// Parses lines prefixed with "v " and joins them into a single model string.
    pub fn extract_model(output: &str) -> Option<String> {
        let mut model_lines = Vec::new();
        for line in output.lines() {
            if let Some(stripped) = line.strip_prefix("v ") {
                model_lines.push(stripped.to_string());
            }
        }
        if model_lines.is_empty() {
            None
        } else {
            Some(model_lines.join(" "))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_extract_number_basic() {
            assert_eq!(extract_number("c conflicts: 12345"), Some(12345));
            assert_eq!(extract_number("c decisions 5000"), Some(5000));
        }

        #[test]
        fn test_extract_number_with_punctuation() {
            // "memory: 1024MB" - "memory:" has no digits, "1024MB" trims "MB" to get "1024"
            assert_eq!(extract_number("memory: 1024MB"), Some(1024));
            // Bracketed numbers work: "[123]" trims brackets
            assert_eq!(extract_number("stat: [123]"), Some(123));
        }

        #[test]
        fn test_extract_number_no_number() {
            assert_eq!(extract_number("no numbers here"), None);
            assert_eq!(extract_number(""), None);
        }

        #[test]
        fn test_extract_model_basic() {
            let output = "c comment\nv 1 -2 3\nv 4 -5 0\n";
            let model = extract_model(output);
            assert_eq!(model, Some("1 -2 3 4 -5 0".to_string()));
        }

        #[test]
        fn test_extract_model_empty() {
            assert_eq!(extract_model("c only comments\ns UNSATISFIABLE"), None);
            assert_eq!(extract_model(""), None);
        }

        #[test]
        fn test_extract_model_single_line() {
            let output = "v 1 2 3 0";
            let model = extract_model(output);
            assert_eq!(model, Some("1 2 3 0".to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_result_predicates() {
        let sat = SolverResult::Sat { model: None };
        assert!(sat.is_sat());
        assert!(!sat.is_unsat());
        assert!(sat.is_definitive());

        let unsat = SolverResult::Unsat { proof: None };
        assert!(!unsat.is_sat());
        assert!(unsat.is_unsat());
        assert!(unsat.is_definitive());

        let unknown = SolverResult::Unknown {
            reason: "timeout".to_string(),
        };
        assert!(!unknown.is_sat());
        assert!(!unknown.is_unsat());
        assert!(!unknown.is_definitive());
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.seed.is_none());
        assert!(config.threads.is_none());
    }

    #[test]
    fn test_solver_error_display() {
        let errors = vec![
            SolverError::NotFound("cadical".to_string()),
            SolverError::ExecutionFailed("process crashed".to_string()),
            SolverError::Timeout(Duration::from_secs(60)),
            SolverError::InvalidInput("bad dimacs".to_string()),
            SolverError::Unknown("memory limit".to_string()),
        ];

        for err in errors {
            let display = err.to_string();
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_solver_error_not_found() {
        let err = SolverError::NotFound("z3".to_string());
        let display = err.to_string();
        assert!(display.contains("not found"));
        assert!(display.contains("z3"));
    }

    #[test]
    fn test_solver_error_timeout() {
        let err = SolverError::Timeout(Duration::from_secs(120));
        let display = err.to_string();
        assert!(display.contains("timed out"));
    }

    #[test]
    fn test_solver_error_clone() {
        let err = SolverError::InvalidInput("test".to_string());
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }

    #[test]
    fn test_solver_result_sat_with_model() {
        let result = SolverResult::Sat {
            model: Some("1 -2 3".to_string()),
        };
        assert!(result.is_sat());
        assert!(result.is_definitive());
        if let SolverResult::Sat { model } = result {
            assert_eq!(model, Some("1 -2 3".to_string()));
        }
    }

    #[test]
    fn test_solver_result_unsat_with_proof() {
        let result = SolverResult::Unsat {
            proof: Some("(proof ...)".to_string()),
        };
        assert!(result.is_unsat());
        if let SolverResult::Unsat { proof } = result {
            assert!(proof.is_some());
        }
    }

    #[test]
    fn test_solver_result_serialization() {
        let results = vec![
            SolverResult::Sat { model: None },
            SolverResult::Sat {
                model: Some("1 2".to_string()),
            },
            SolverResult::Unsat { proof: None },
            SolverResult::Unknown {
                reason: "test".to_string(),
            },
        ];

        for result in results {
            let json = serde_json::to_string(&result).unwrap();
            let deserialized: SolverResult = serde_json::from_str(&json).unwrap();
            assert_eq!(result, deserialized);
        }
    }

    #[test]
    fn test_solver_stats_default() {
        let stats = SolverStats::default();
        assert_eq!(stats.solve_time, Duration::ZERO);
        assert!(stats.conflicts.is_none());
        assert!(stats.decisions.is_none());
        assert!(stats.propagations.is_none());
        assert!(stats.memory_bytes.is_none());
        assert!(stats.extra.is_empty());
    }

    #[test]
    fn test_solver_stats_clone() {
        let mut stats = SolverStats {
            solve_time: Duration::from_secs(5),
            conflicts: Some(1000),
            ..Default::default()
        };
        stats
            .extra
            .insert("custom".to_string(), "value".to_string());

        let cloned = stats.clone();
        assert_eq!(stats.solve_time, cloned.solve_time);
        assert_eq!(stats.conflicts, cloned.conflicts);
        assert_eq!(stats.extra.get("custom"), cloned.extra.get("custom"));
    }

    #[test]
    fn test_solver_stats_debug() {
        let stats = SolverStats {
            solve_time: Duration::from_secs(10),
            conflicts: Some(500),
            decisions: Some(1000),
            propagations: Some(5000),
            memory_bytes: Some(1024 * 1024),
            extra: std::collections::HashMap::new(),
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("SolverStats"));
        assert!(debug.contains("solve_time"));
    }

    #[test]
    fn test_solver_stats_serialization() {
        let stats = SolverStats {
            solve_time: Duration::from_millis(500),
            conflicts: Some(100),
            decisions: None,
            propagations: Some(200),
            memory_bytes: None,
            extra: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: SolverStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats.solve_time, deserialized.solve_time);
        assert_eq!(stats.conflicts, deserialized.conflicts);
    }

    #[test]
    fn test_solver_config_with_all_fields() {
        let mut options = std::collections::HashMap::new();
        options.insert("verbose".to_string(), "true".to_string());

        let config = SolverConfig {
            timeout: Duration::from_secs(60),
            seed: Some(42),
            threads: Some(4),
            options,
        };

        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.threads, Some(4));
        assert_eq!(config.options.get("verbose"), Some(&"true".to_string()));
    }

    #[test]
    fn test_solver_config_clone() {
        let config = SolverConfig {
            timeout: Duration::from_secs(30),
            seed: Some(123),
            threads: Some(2),
            options: std::collections::HashMap::new(),
        };
        let cloned = config.clone();
        assert_eq!(config.timeout, cloned.timeout);
        assert_eq!(config.seed, cloned.seed);
        assert_eq!(config.threads, cloned.threads);
    }

    #[test]
    fn test_solver_config_debug() {
        let config = SolverConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("SolverConfig"));
        assert!(debug.contains("timeout"));
    }

    #[test]
    fn test_solver_output_debug() {
        let output = SolverOutput {
            result: SolverResult::Sat { model: None },
            stats: SolverStats::default(),
            raw_output: Some("s SATISFIABLE".to_string()),
        };
        let debug = format!("{:?}", output);
        assert!(debug.contains("SolverOutput"));
    }

    #[test]
    fn test_solver_output_clone() {
        let output = SolverOutput {
            result: SolverResult::Unsat { proof: None },
            stats: SolverStats {
                solve_time: Duration::from_secs(1),
                ..Default::default()
            },
            raw_output: Some("s UNSATISFIABLE".to_string()),
        };
        let cloned = output.clone();
        assert_eq!(output.result, cloned.result);
        assert_eq!(output.stats.solve_time, cloned.stats.solve_time);
        assert_eq!(output.raw_output, cloned.raw_output);
    }

    #[test]
    fn test_solver_capability_variants() {
        let capabilities = vec![
            SolverCapability::Sat,
            SolverCapability::SmtBv,
            SolverCapability::SmtArrays,
            SolverCapability::SmtUf,
            SolverCapability::SmtLia,
            SolverCapability::SmtLra,
            SolverCapability::SmtNla,
            SolverCapability::Quantifiers,
            SolverCapability::Incremental,
            SolverCapability::Proofs,
            SolverCapability::UnsatCores,
        ];

        for cap in capabilities {
            let debug = format!("{:?}", cap);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_solver_capability_serialization() {
        let capabilities = vec![
            SolverCapability::Sat,
            SolverCapability::SmtBv,
            SolverCapability::Quantifiers,
        ];

        for cap in capabilities {
            let json = serde_json::to_string(&cap).unwrap();
            let deserialized: SolverCapability = serde_json::from_str(&json).unwrap();
            assert_eq!(cap, deserialized);
        }
    }

    #[test]
    fn test_solver_capability_equality() {
        assert_eq!(SolverCapability::Sat, SolverCapability::Sat);
        assert_ne!(SolverCapability::Sat, SolverCapability::SmtBv);
    }

    #[test]
    fn test_solver_capability_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SolverCapability::Sat);
        set.insert(SolverCapability::SmtBv);
        set.insert(SolverCapability::Sat); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_solver_info_debug() {
        let info = SolverInfo {
            id: "test".to_string(),
            name: "Test Solver".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec![SolverCapability::Sat],
            available: true,
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("SolverInfo"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_solver_info_clone() {
        let info = SolverInfo {
            id: "solver1".to_string(),
            name: "Solver One".to_string(),
            version: "2.0".to_string(),
            capabilities: vec![SolverCapability::Sat, SolverCapability::SmtBv],
            available: false,
        };
        let cloned = info.clone();
        assert_eq!(info.id, cloned.id);
        assert_eq!(info.capabilities.len(), cloned.capabilities.len());
    }

    #[test]
    fn test_solver_info_serialization() {
        let info = SolverInfo {
            id: "z3".to_string(),
            name: "Z3".to_string(),
            version: "4.12.0".to_string(),
            capabilities: vec![SolverCapability::Sat, SolverCapability::SmtBv],
            available: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: SolverInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.id, deserialized.id);
        assert_eq!(info.capabilities, deserialized.capabilities);
    }

    #[test]
    fn test_solver_error_is_error() {
        fn is_error<E: std::error::Error>(_: &E) {}
        let err = SolverError::NotFound("test".to_string());
        is_error(&err);
    }
}
