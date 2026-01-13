//! Configuration types for Apalache backend

use std::path::PathBuf;
use std::time::Duration;

/// Apalache checker mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApalacheMode {
    /// Symbolic bounded model checking (default)
    #[default]
    Check,
    /// Parse and typecheck only
    Parse,
    /// Simulate random execution traces
    Simulate,
}

impl ApalacheMode {
    /// Get the Apalache command for this mode
    pub fn command(&self) -> &'static str {
        match self {
            ApalacheMode::Check => "check",
            ApalacheMode::Parse => "parse",
            ApalacheMode::Simulate => "simulate",
        }
    }
}

/// Configuration for Apalache backend
#[derive(Debug, Clone)]
pub struct ApalacheConfig {
    /// Path to Apalache JAR or executable
    pub apalache_path: Option<PathBuf>,
    /// Path to Java executable (if using JAR)
    pub java_path: PathBuf,
    /// JVM memory limit (e.g., "-Xmx4G")
    pub jvm_memory: Option<String>,
    /// Checker mode
    pub mode: ApalacheMode,
    /// Maximum trace length for bounded checking
    pub length: Option<u32>,
    /// Initialize state with Init predicate
    pub init: Option<String>,
    /// Next-state relation (defaults to "Next")
    pub next: Option<String>,
    /// Invariants to check
    pub inv: Vec<String>,
    /// Temporal properties to check
    pub temporal: Vec<String>,
    /// Timeout for verification
    pub timeout: Duration,
    /// SMT solver (z3 or cvc4)
    pub smt_solver: String,
    /// Enable debugging output
    pub debug: bool,
    /// Output directory for intermediate files
    pub out_dir: Option<PathBuf>,
}

impl Default for ApalacheConfig {
    fn default() -> Self {
        Self {
            apalache_path: None,
            java_path: PathBuf::from("java"),
            jvm_memory: Some("-Xmx4G".to_string()),
            mode: ApalacheMode::Check,
            length: Some(10), // Default bounded length
            init: None,
            next: None,
            inv: Vec::new(),
            temporal: Vec::new(),
            timeout: Duration::from_secs(600), // 10 minutes (symbolic checking is slower)
            smt_solver: "z3".to_string(),
            debug: false,
            out_dir: None,
        }
    }
}
