//! Improvement types and proposals

use crate::certificate::ProofCertificate;
use crate::version::{CapabilitySet, Version};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A proposed improvement to the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    /// Unique ID for this improvement proposal
    pub id: String,

    /// Human-readable description of the improvement
    pub description: String,

    /// What kind of improvement this is
    pub kind: ImprovementKind,

    /// What the improvement targets
    pub target: ImprovementTarget,

    /// Expected capability changes
    pub expected_capabilities: CapabilitySet,

    /// When the improvement was proposed
    pub proposed_at: DateTime<Utc>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,

    /// The actual changes (implementation-specific)
    pub changes: ImprovementChanges,
}

impl Improvement {
    /// Create a new improvement proposal
    pub fn new(
        description: impl Into<String>,
        kind: ImprovementKind,
        target: ImprovementTarget,
    ) -> Self {
        let id = format!("imp-{}", chrono::Utc::now().timestamp_millis() % 1_000_000);

        Self {
            id,
            description: description.into(),
            kind,
            target,
            expected_capabilities: CapabilitySet::new(),
            proposed_at: Utc::now(),
            metadata: HashMap::new(),
            changes: ImprovementChanges::default(),
        }
    }

    /// Set expected capabilities after improvement
    pub fn with_expected_capabilities(mut self, caps: CapabilitySet) -> Self {
        self.expected_capabilities = caps;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the changes
    pub fn with_changes(mut self, changes: ImprovementChanges) -> Self {
        self.changes = changes;
        self
    }

    /// Check if this improvement is valid (basic structural checks)
    pub fn is_valid(&self) -> bool {
        !self.description.is_empty() && !matches!(self.target, ImprovementTarget::Unknown)
    }
}

/// Kind of improvement being proposed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImprovementKind {
    /// Bug fix - should not change capabilities
    BugFix,

    /// Performance optimization
    Optimization,

    /// New feature addition
    Feature,

    /// Security improvement
    Security,

    /// Refactoring - no functional change
    Refactoring,

    /// Configuration change
    Configuration,

    /// Dependency update
    DependencyUpdate,

    /// Custom improvement kind
    Custom(String),
}

impl ImprovementKind {
    /// Check if this kind typically requires capability verification
    pub fn requires_capability_verification(&self) -> bool {
        matches!(
            self,
            ImprovementKind::Feature
                | ImprovementKind::Security
                | ImprovementKind::DependencyUpdate
        )
    }
}

/// Target of the improvement
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImprovementTarget {
    /// Entire system
    System,

    /// Specific module
    Module(String),

    /// Specific function
    Function(String),

    /// Configuration
    Config(String),

    /// Dependencies
    Dependencies,

    /// Unknown target (invalid)
    Unknown,
}

/// The actual changes in an improvement
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImprovementChanges {
    /// Files that were modified
    pub modified_files: Vec<FileChange>,

    /// Configuration changes
    pub config_changes: Vec<ConfigChange>,

    /// Dependency changes
    pub dependency_changes: Vec<DependencyChange>,

    /// Raw patch data (optional)
    pub patch: Option<String>,
}

/// A file change in an improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    /// Path to the file
    pub path: String,

    /// Type of change
    pub change_type: FileChangeType,

    /// Hash of the new content
    pub new_content_hash: Option<String>,

    /// Lines added
    pub lines_added: usize,

    /// Lines removed
    pub lines_removed: usize,
}

/// Type of file change
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FileChangeType {
    /// File was added
    Added,
    /// File was modified
    Modified,
    /// File was deleted
    Deleted,
    /// File was renamed
    Renamed { from: String },
}

/// A configuration change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChange {
    /// Configuration key
    pub key: String,

    /// Old value (None if new key)
    pub old_value: Option<String>,

    /// New value (None if deleted)
    pub new_value: Option<String>,
}

/// A dependency change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyChange {
    /// Dependency name
    pub name: String,

    /// Change type
    pub change_type: DependencyChangeType,

    /// Old version (if upgrade/downgrade)
    pub old_version: Option<String>,

    /// New version (if upgrade/add)
    pub new_version: Option<String>,
}

/// Type of dependency change
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DependencyChangeType {
    /// New dependency added
    Added,
    /// Dependency removed
    Removed,
    /// Dependency upgraded
    Upgraded,
    /// Dependency downgraded
    Downgraded,
}

/// Result of attempting to apply an improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementResult {
    /// Improvement was accepted and applied
    Accepted {
        /// The new version created (boxed to reduce enum size)
        new_version: Box<Version>,
        /// The proof certificate for the new version (boxed to reduce enum size)
        certificate: Box<ProofCertificate>,
    },

    /// Improvement was rejected
    Rejected {
        /// Why the improvement was rejected
        reason: RejectionReason,
        /// Detailed error message
        details: String,
        /// Which checks failed
        failed_checks: Vec<String>,
    },
}

impl ImprovementResult {
    /// Check if the improvement was accepted
    pub fn is_accepted(&self) -> bool {
        matches!(self, ImprovementResult::Accepted { .. })
    }

    /// Get the new version if accepted
    pub fn new_version(&self) -> Option<&Version> {
        match self {
            ImprovementResult::Accepted { new_version, .. } => Some(new_version),
            ImprovementResult::Rejected { .. } => None,
        }
    }

    /// Get the certificate if accepted
    pub fn certificate(&self) -> Option<&ProofCertificate> {
        match self {
            ImprovementResult::Accepted { certificate, .. } => Some(certificate),
            ImprovementResult::Rejected { .. } => None,
        }
    }

    /// Create an accepted result
    pub fn accepted(new_version: Version, certificate: ProofCertificate) -> Self {
        ImprovementResult::Accepted {
            new_version: Box::new(new_version),
            certificate: Box::new(certificate),
        }
    }

    /// Create a rejected result
    pub fn rejected(
        reason: RejectionReason,
        details: impl Into<String>,
        failed_checks: Vec<String>,
    ) -> Self {
        ImprovementResult::Rejected {
            reason,
            details: details.into(),
            failed_checks,
        }
    }
}

/// Reason why an improvement was rejected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RejectionReason {
    /// Improvement would break soundness
    SoundnessViolation,

    /// Improvement would cause capability regression
    CapabilityRegression,

    /// Improvement failed formal verification
    VerificationFailed,

    /// Improvement is malformed or invalid
    InvalidProposal,

    /// Verification timed out
    VerificationTimeout,

    /// System is busy with another improvement
    SystemBusy,

    /// Custom rejection reason
    Custom(String),
}
