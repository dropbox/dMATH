//! NeMo Guardrails backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Rail type for NeMo Guardrails
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RailType {
    /// Input rails - filter/modify user input
    #[default]
    Input,
    /// Output rails - filter/modify LLM output
    Output,
    /// Dialog rails - manage conversation flow
    Dialog,
    /// Retrieval rails - filter retrieved content
    Retrieval,
}

impl RailType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RailType::Input => "input",
            RailType::Output => "output",
            RailType::Dialog => "dialog",
            RailType::Retrieval => "retrieval",
        }
    }
}

/// Colang version for rails definition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColangVersion {
    /// Colang 1.0 syntax
    V1,
    /// Colang 2.0 syntax (recommended)
    #[default]
    V2,
}

impl ColangVersion {
    pub fn as_str(&self) -> &'static str {
        match self {
            ColangVersion::V1 => "1.0",
            ColangVersion::V2 => "2.0",
        }
    }
}

/// NeMo Guardrails backend configuration
#[derive(Debug, Clone)]
pub struct NeMoGuardrailsConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Type of rail to verify
    pub rail_type: RailType,
    /// Colang version
    pub colang_version: ColangVersion,
    /// Enable jailbreak detection
    pub jailbreak_detection: bool,
    /// Enable topical rail
    pub topical_rail: bool,
    /// Enable fact-checking
    pub fact_checking: bool,
    /// Minimum pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for NeMoGuardrailsConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            rail_type: RailType::Output,
            colang_version: ColangVersion::V2,
            jailbreak_detection: true,
            topical_rail: false,
            fact_checking: false,
            pass_rate_threshold: 0.85,
            timeout: Duration::from_secs(180),
        }
    }
}

impl NeMoGuardrailsConfig {
    /// Configure for input filtering
    pub fn input_filter() -> Self {
        Self {
            rail_type: RailType::Input,
            jailbreak_detection: true,
            pass_rate_threshold: 0.9,
            ..Default::default()
        }
    }

    /// Configure for dialog management
    pub fn dialog() -> Self {
        Self {
            rail_type: RailType::Dialog,
            topical_rail: true,
            pass_rate_threshold: 0.8,
            ..Default::default()
        }
    }

    /// Configure with fact-checking enabled
    pub fn with_fact_checking() -> Self {
        Self {
            fact_checking: true,
            pass_rate_threshold: 0.9,
            ..Default::default()
        }
    }
}
