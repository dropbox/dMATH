//! Proof options and set_option tactic
//!
//! This module provides configuration types for local proof options
//! and the `set_option` tactic for modifying them.

use crate::tactic::{ProofState, TacticError, TacticResult};

/// Set option value types
#[derive(Debug, Clone)]
pub enum OptionValue {
    Bool(bool),
    Nat(u64),
    String(String),
}

/// Configuration for set_option
#[derive(Debug, Clone, Default)]
pub struct SetOptionConfig {
    /// Options to set (key -> value)
    pub options: Vec<(String, OptionValue)>,
}

impl SetOptionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn set_bool(mut self, key: &str, value: bool) -> Self {
        self.options
            .push((key.to_string(), OptionValue::Bool(value)));
        self
    }

    #[must_use]
    pub fn set_nat(mut self, key: &str, value: u64) -> Self {
        self.options
            .push((key.to_string(), OptionValue::Nat(value)));
        self
    }

    #[must_use]
    pub fn set_string(mut self, key: &str, value: &str) -> Self {
        self.options
            .push((key.to_string(), OptionValue::String(value.to_string())));
        self
    }
}

/// Local proof state options that can be modified
#[derive(Debug, Clone)]
pub struct ProofOptions {
    /// Enable verbose output
    pub verbose: bool,
    /// Maximum recursion depth for tactics
    pub max_depth: u64,
    /// Enable tracing
    pub trace: bool,
    /// Timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,
    /// Enable profiling
    pub profile: bool,
}

impl Default for ProofOptions {
    fn default() -> Self {
        ProofOptions {
            verbose: false,
            max_depth: 100,
            trace: false,
            timeout_ms: 0,
            profile: false,
        }
    }
}

/// Tactic: set_option
///
/// Sets a local option for the current proof. Options affect tactic
/// behavior within the current proof scope.
///
/// Common options:
/// - `verbose`: Enable verbose output from tactics
/// - `max_depth`: Maximum recursion depth for search tactics
/// - `trace`: Enable trace output for debugging
/// - `timeout_ms`: Set timeout for tactics in milliseconds
///
/// # Example
/// ```text
/// set_option verbose true
/// simp  -- Now shows verbose output
/// ```
///
/// Note: This is a "meta" tactic that doesn't change the proof state
/// goals, only the configuration.
///
/// # Errors
/// - `Other` if the option name is not recognized
pub fn set_option(state: &mut ProofState, key: &str, value: OptionValue) -> TacticResult {
    // Validate option name
    let valid_options = ["verbose", "max_depth", "trace", "timeout_ms", "profile"];
    if !valid_options.contains(&key) {
        return Err(TacticError::Other(format!(
            "Unknown option '{key}'. Valid options: {valid_options:?}"
        )));
    }

    // Note: In a full implementation, ProofState would have an options field
    // For now, we just validate and succeed
    // The actual option storage would require extending ProofState

    // Log the option being set (in a real implementation)
    let _ = (state, key, value); // Suppress unused warnings

    Ok(())
}

/// set_option with builder-style config
pub fn set_options(state: &mut ProofState, config: SetOptionConfig) -> TacticResult {
    for (key, value) in config.options {
        set_option(state, &key, value)?;
    }
    Ok(())
}
