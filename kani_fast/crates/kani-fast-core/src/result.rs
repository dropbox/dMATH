//! Verification result types

use kani_fast_counterexample::StructuredCounterexample;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Status of a verification attempt
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Property proven to hold
    Proven,

    /// Property disproven (counterexample found)
    Disproven,

    /// Verification inconclusive
    Unknown { reason: String },

    /// Verification timed out
    Timeout,

    /// Error during verification
    Error { message: String },
}

impl VerificationStatus {
    /// Check if this is a successful verification
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Proven)
    }

    /// Check if this is a definitive result (proven or disproven)
    pub fn is_definitive(&self) -> bool {
        matches!(self, Self::Proven | Self::Disproven)
    }
}

/// Result of a verification run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Verification status
    pub status: VerificationStatus,

    /// Counterexample if property was disproven
    pub counterexample: Option<StructuredCounterexample>,

    /// Proof certificate or summary if property was proven
    pub proof: Option<String>,

    /// Diagnostic messages
    pub diagnostics: Vec<String>,

    /// Time taken for verification
    pub duration: Duration,

    /// Number of checks verified
    pub checks_passed: usize,

    /// Number of checks failed
    pub checks_failed: usize,

    /// Total number of checks
    pub checks_total: usize,
}

impl VerificationResult {
    /// Create a successful (proven) result
    pub fn proven(duration: Duration, checks_total: usize) -> Self {
        Self {
            status: VerificationStatus::Proven,
            counterexample: None,
            proof: Some("All Kani checks passed".to_string()),
            diagnostics: Vec::new(),
            duration,
            checks_passed: checks_total,
            checks_failed: 0,
            checks_total,
        }
    }

    /// Create a disproven result with counterexample
    pub fn disproven(
        counterexample: StructuredCounterexample,
        duration: Duration,
        checks_failed: usize,
        checks_total: usize,
    ) -> Self {
        Self {
            status: VerificationStatus::Disproven,
            counterexample: Some(counterexample),
            proof: None,
            diagnostics: Vec::new(),
            duration,
            checks_passed: checks_total.saturating_sub(checks_failed),
            checks_failed,
            checks_total,
        }
    }

    /// Create an unknown result
    pub fn unknown(reason: String, duration: Duration) -> Self {
        Self {
            status: VerificationStatus::Unknown { reason },
            counterexample: None,
            proof: None,
            diagnostics: Vec::new(),
            duration,
            checks_passed: 0,
            checks_failed: 0,
            checks_total: 0,
        }
    }

    /// Create a timeout result
    pub fn timeout(duration: Duration) -> Self {
        Self {
            status: VerificationStatus::Timeout,
            counterexample: None,
            proof: None,
            diagnostics: vec!["Verification timed out".to_string()],
            duration,
            checks_passed: 0,
            checks_failed: 0,
            checks_total: 0,
        }
    }

    /// Create an error result
    pub fn error(message: String, duration: Duration) -> Self {
        Self {
            status: VerificationStatus::Error {
                message: message.clone(),
            },
            counterexample: None,
            proof: None,
            diagnostics: vec![message],
            duration,
            checks_passed: 0,
            checks_failed: 0,
            checks_total: 0,
        }
    }

    /// Add a diagnostic message
    pub fn with_diagnostic(mut self, diagnostic: String) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }

    /// Add multiple diagnostic messages
    pub fn with_diagnostics(mut self, diagnostics: Vec<String>) -> Self {
        self.diagnostics.extend(diagnostics);
        self
    }
}

impl std::fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.status {
            VerificationStatus::Proven => {
                write!(f, "VERIFIED: {} checks passed", self.checks_passed)?;
            }
            VerificationStatus::Disproven => {
                write!(
                    f,
                    "FAILED: {} of {} checks failed",
                    self.checks_failed, self.checks_total
                )?;
                if let Some(ce) = &self.counterexample {
                    write!(f, "\n{}", ce.summary())?;
                }
            }
            VerificationStatus::Unknown { reason } => {
                write!(f, "UNKNOWN: {reason}")?;
            }
            VerificationStatus::Timeout => {
                write!(f, "TIMEOUT after {:?}", self.duration)?;
            }
            VerificationStatus::Error { message } => {
                write!(f, "ERROR: {message}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_status_is_success() {
        assert!(VerificationStatus::Proven.is_success());
        assert!(!VerificationStatus::Disproven.is_success());
        assert!(!VerificationStatus::Unknown {
            reason: "test".to_string()
        }
        .is_success());
        assert!(!VerificationStatus::Timeout.is_success());
        assert!(!VerificationStatus::Error {
            message: "test".to_string()
        }
        .is_success());
    }

    #[test]
    fn test_verification_status_is_definitive() {
        assert!(VerificationStatus::Proven.is_definitive());
        assert!(VerificationStatus::Disproven.is_definitive());
        assert!(!VerificationStatus::Unknown {
            reason: "test".to_string()
        }
        .is_definitive());
        assert!(!VerificationStatus::Timeout.is_definitive());
        assert!(!VerificationStatus::Error {
            message: "test".to_string()
        }
        .is_definitive());
    }

    #[test]
    fn test_verification_result_proven() {
        let result = VerificationResult::proven(Duration::from_secs(5), 10);
        assert!(result.status.is_success());
        assert_eq!(result.checks_passed, 10);
        assert_eq!(result.checks_failed, 0);
        assert_eq!(result.checks_total, 10);
        assert!(result.counterexample.is_none());
        assert!(result.proof.is_some());
    }

    #[test]
    fn test_verification_result_unknown() {
        let result =
            VerificationResult::unknown("solver gave up".to_string(), Duration::from_secs(30));
        assert!(!result.status.is_success());
        assert!(!result.status.is_definitive());
        assert!(matches!(result.status, VerificationStatus::Unknown { .. }));
    }

    #[test]
    fn test_verification_result_timeout() {
        let result = VerificationResult::timeout(Duration::from_secs(300));
        assert!(matches!(result.status, VerificationStatus::Timeout));
        assert_eq!(result.duration, Duration::from_secs(300));
        assert!(!result.diagnostics.is_empty());
    }

    #[test]
    fn test_verification_result_error() {
        let result =
            VerificationResult::error("compilation failed".to_string(), Duration::from_secs(1));
        assert!(matches!(result.status, VerificationStatus::Error { .. }));
        assert!(result
            .diagnostics
            .contains(&"compilation failed".to_string()));
    }

    #[test]
    fn test_verification_result_with_diagnostic() {
        let result = VerificationResult::proven(Duration::from_secs(1), 1)
            .with_diagnostic("debug info".to_string());
        assert_eq!(result.diagnostics.len(), 1);
        assert!(result.diagnostics.contains(&"debug info".to_string()));
    }

    #[test]
    fn test_verification_result_with_diagnostics() {
        let result = VerificationResult::proven(Duration::from_secs(1), 1)
            .with_diagnostics(vec!["info 1".to_string(), "info 2".to_string()]);
        assert_eq!(result.diagnostics.len(), 2);
    }

    #[test]
    fn test_verification_result_display_proven() {
        let result = VerificationResult::proven(Duration::from_secs(5), 10);
        let display = result.to_string();
        assert!(display.contains("VERIFIED"));
        assert!(display.contains("10"));
    }

    #[test]
    fn test_verification_result_display_unknown() {
        let result =
            VerificationResult::unknown("resource limit".to_string(), Duration::from_secs(10));
        let display = result.to_string();
        assert!(display.contains("UNKNOWN"));
        assert!(display.contains("resource limit"));
    }

    #[test]
    fn test_verification_result_display_timeout() {
        let result = VerificationResult::timeout(Duration::from_secs(60));
        let display = result.to_string();
        assert!(display.contains("TIMEOUT"));
    }

    #[test]
    fn test_verification_result_display_error() {
        let result = VerificationResult::error("fatal error".to_string(), Duration::from_secs(1));
        let display = result.to_string();
        assert!(display.contains("ERROR"));
        assert!(display.contains("fatal error"));
    }

    #[test]
    fn test_verification_status_serialization() {
        let statuses = [
            VerificationStatus::Proven,
            VerificationStatus::Disproven,
            VerificationStatus::Unknown {
                reason: "test".to_string(),
            },
            VerificationStatus::Timeout,
            VerificationStatus::Error {
                message: "test".to_string(),
            },
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: VerificationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, deserialized);
        }
    }
}
