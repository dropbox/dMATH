//! Rollback mechanism for failed improvements
//!
//! This module provides automatic rollback when verification fails or
//! when the system detects issues with a new version.

use crate::error::{SelfImpError, SelfImpResult};
use crate::version::{Version, VersionId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for the rollback manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    /// Maximum number of rollback attempts before giving up
    pub max_rollback_attempts: usize,

    /// Whether to automatically rollback on verification failure
    pub auto_rollback_on_failure: bool,

    /// Whether to keep failed versions in history (for debugging)
    pub keep_failed_versions: bool,
}

impl Default for RollbackConfig {
    fn default() -> Self {
        Self {
            max_rollback_attempts: 3,
            auto_rollback_on_failure: true,
            keep_failed_versions: true,
        }
    }
}

/// Triggers that can initiate a rollback
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RollbackTrigger {
    /// Verification gate rejected improvement
    VerificationFailure,

    /// Soundness violation detected post-deployment
    SoundnessViolation,

    /// Capability regression detected
    CapabilityRegression,

    /// Manual rollback request
    Manual { reason: String },

    /// Health check failure
    HealthCheckFailure { check_name: String },

    /// Timeout during operation
    Timeout,

    /// Internal error
    InternalError { message: String },
}

impl RollbackTrigger {
    /// Get a human-readable description of the trigger
    pub fn description(&self) -> String {
        match self {
            RollbackTrigger::VerificationFailure => {
                "Verification gate rejected improvement".to_string()
            }
            RollbackTrigger::SoundnessViolation => "Soundness violation detected".to_string(),
            RollbackTrigger::CapabilityRegression => "Capability regression detected".to_string(),
            RollbackTrigger::Manual { reason } => format!("Manual rollback: {}", reason),
            RollbackTrigger::HealthCheckFailure { check_name } => {
                format!("Health check failed: {}", check_name)
            }
            RollbackTrigger::Timeout => "Operation timed out".to_string(),
            RollbackTrigger::InternalError { message } => format!("Internal error: {}", message),
        }
    }

    /// Check if this trigger indicates a critical failure
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            RollbackTrigger::SoundnessViolation | RollbackTrigger::InternalError { .. }
        )
    }
}

/// Record of a rollback action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackAction {
    /// Unique ID for this rollback action
    pub id: String,

    /// What triggered the rollback
    pub trigger: RollbackTrigger,

    /// Version we rolled back from
    pub from_version: VersionId,

    /// Version we rolled back to
    pub to_version: VersionId,

    /// When the rollback occurred
    pub timestamp: DateTime<Utc>,

    /// Whether the rollback was successful
    pub success: bool,

    /// Additional details about the rollback
    pub details: Option<String>,
}

impl RollbackAction {
    /// Create a new rollback action
    pub fn new(trigger: RollbackTrigger, from_version: VersionId, to_version: VersionId) -> Self {
        let id = format!(
            "rollback-{}",
            chrono::Utc::now().timestamp_millis() % 1_000_000
        );

        Self {
            id,
            trigger,
            from_version,
            to_version,
            timestamp: Utc::now(),
            success: false, // Set to true after successful rollback
            details: None,
        }
    }

    /// Mark the rollback as successful
    pub fn mark_successful(mut self) -> Self {
        self.success = true;
        self
    }

    /// Add details to the rollback
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// Manager for handling rollbacks
pub struct RollbackManager {
    /// Configuration
    config: RollbackConfig,

    /// History of rollback actions
    history: Vec<RollbackAction>,
}

impl RollbackManager {
    /// Create a new rollback manager
    pub fn new() -> Self {
        Self {
            config: RollbackConfig::default(),
            history: Vec::new(),
        }
    }

    /// Create a rollback manager with custom configuration
    pub fn with_config(config: RollbackConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &RollbackConfig {
        &self.config
    }

    /// Initiate a rollback
    ///
    /// This method handles the rollback process:
    /// 1. Validates that rollback is possible
    /// 2. Records the rollback action
    /// 3. Returns the version to rollback to
    pub fn initiate_rollback(
        &mut self,
        trigger: RollbackTrigger,
        current_version: &Version,
        previous_version: &Version,
    ) -> SelfImpResult<RollbackAction> {
        // Check if we have a version to rollback to
        if current_version.previous_version.is_none() {
            return Err(SelfImpError::NoPreviousVersion);
        }

        // Check rollback attempt count
        let recent_rollbacks = self.recent_rollback_count(&current_version.id);
        if recent_rollbacks >= self.config.max_rollback_attempts {
            return Err(SelfImpError::RollbackFailed(format!(
                "Maximum rollback attempts ({}) exceeded",
                self.config.max_rollback_attempts
            )));
        }

        // Create rollback action
        let action = RollbackAction::new(
            trigger,
            current_version.id.clone(),
            previous_version.id.clone(),
        );

        Ok(action)
    }

    /// Record a completed rollback
    pub fn record_rollback(&mut self, action: RollbackAction) {
        self.history.push(action);
    }

    /// Get rollback history
    pub fn history(&self) -> &[RollbackAction] {
        &self.history
    }

    /// Get recent rollback count for a version
    fn recent_rollback_count(&self, version_id: &VersionId) -> usize {
        let cutoff = Utc::now() - chrono::Duration::hours(1);
        self.history
            .iter()
            .filter(|a| a.from_version == *version_id && a.timestamp > cutoff)
            .count()
    }

    /// Check if auto-rollback is enabled
    pub fn auto_rollback_enabled(&self) -> bool {
        self.config.auto_rollback_on_failure
    }

    /// Find the most recent successful rollback
    pub fn last_successful_rollback(&self) -> Option<&RollbackAction> {
        self.history.iter().rev().find(|a| a.success)
    }

    /// Clear rollback history (for testing)
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Check if the system is in a rollback loop
    ///
    /// A rollback loop is detected when there have been too many
    /// rollbacks in a short period, suggesting a fundamental issue.
    pub fn is_in_rollback_loop(&self) -> bool {
        let cutoff = Utc::now() - chrono::Duration::minutes(30);
        let recent_count = self.history.iter().filter(|a| a.timestamp > cutoff).count();

        recent_count >= self.config.max_rollback_attempts
    }
}

impl Default for RollbackManager {
    fn default() -> Self {
        Self::new()
    }
}
