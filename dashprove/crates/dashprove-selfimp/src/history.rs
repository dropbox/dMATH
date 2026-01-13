//! Version history management
//!
//! This module maintains the complete history of all verified versions,
//! including their proof certificates. The history is tamper-evident
//! through the certificate chain.

use crate::certificate::{CertificateChain, ProofCertificate};
use crate::error::{SelfImpError, SelfImpResult};
use crate::version::{Version, VersionId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single entry in the version history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The version
    pub version: Version,

    /// The proof certificate for this version
    pub certificate: ProofCertificate,

    /// When this entry was added to history
    pub added_at: DateTime<Utc>,

    /// Status of this version
    pub status: VersionStatus,
}

impl HistoryEntry {
    /// Create a new history entry
    pub fn new(version: Version, certificate: ProofCertificate) -> Self {
        Self {
            version,
            certificate,
            added_at: Utc::now(),
            status: VersionStatus::Active,
        }
    }

    /// Check if this version is verified
    pub fn is_verified(&self) -> bool {
        self.certificate.is_verified()
    }

    /// Mark this version as superseded
    pub fn mark_superseded(&mut self, by: VersionId) {
        self.status = VersionStatus::Superseded { by };
    }

    /// Mark this version as rolled back
    pub fn mark_rolled_back(&mut self, reason: String) {
        self.status = VersionStatus::RolledBack { reason };
    }
}

/// Status of a version in history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionStatus {
    /// This is the current active version
    Active,

    /// This version has been superseded by a newer version
    Superseded { by: VersionId },

    /// This version was rolled back
    RolledBack { reason: String },
}

/// Query parameters for searching history
#[derive(Debug, Clone, Default)]
pub struct HistoryQuery {
    /// Only return versions after this time
    pub after: Option<DateTime<Utc>>,

    /// Only return versions before this time
    pub before: Option<DateTime<Utc>>,

    /// Maximum number of results
    pub limit: Option<usize>,

    /// Only return versions with specific status
    pub status: Option<VersionStatusFilter>,
}

/// Filter for version status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionStatusFilter {
    Active,
    Superseded,
    RolledBack,
}

impl HistoryQuery {
    /// Create a query for active versions only
    pub fn active_only() -> Self {
        Self {
            status: Some(VersionStatusFilter::Active),
            ..Default::default()
        }
    }

    /// Limit results
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Filter by time range
    pub fn in_range(mut self, after: DateTime<Utc>, before: DateTime<Utc>) -> Self {
        self.after = Some(after);
        self.before = Some(before);
        self
    }
}

/// The complete version history
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionHistory {
    /// All versions, keyed by version ID
    versions: HashMap<String, HistoryEntry>,

    /// Certificate chain
    certificates: CertificateChain,

    /// ID of the current (active) version
    current_version_id: Option<String>,

    /// ID of the genesis version
    genesis_version_id: Option<String>,
}

impl VersionHistory {
    /// Create a new empty history
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new version in history
    ///
    /// The version must have a valid certificate. If this is not the
    /// genesis version, the certificate must chain to the previous one.
    pub fn register(
        &mut self,
        version: Version,
        certificate: ProofCertificate,
    ) -> SelfImpResult<()> {
        // Verify certificate matches version
        if certificate.version_id != version.id {
            return Err(SelfImpError::InvalidCertificate(
                "Certificate version ID does not match".to_string(),
            ));
        }

        // Verify certificate is valid
        if !certificate.is_verified() {
            return Err(SelfImpError::InvalidCertificate(
                "Certificate does not show successful verification".to_string(),
            ));
        }

        // Add certificate to chain (validates chain integrity)
        self.certificates.add(certificate.clone())?;

        // Mark previous version as superseded
        if let Some(current_id) = &self.current_version_id {
            if let Some(entry) = self.versions.get_mut(current_id) {
                entry.mark_superseded(version.id.clone());
            }
        }

        // Add version to history
        let version_id = version.id.to_string();
        let entry = HistoryEntry::new(version, certificate);
        self.versions.insert(version_id.clone(), entry);

        // Update current version
        if self.genesis_version_id.is_none() {
            self.genesis_version_id = Some(version_id.clone());
        }
        self.current_version_id = Some(version_id);

        Ok(())
    }

    /// Get the current (active) version
    pub fn current(&self) -> Option<&Version> {
        self.current_version_id
            .as_ref()
            .and_then(|id| self.versions.get(id))
            .map(|e| &e.version)
    }

    /// Get the current version entry (with certificate)
    pub fn current_entry(&self) -> Option<&HistoryEntry> {
        self.current_version_id
            .as_ref()
            .and_then(|id| self.versions.get(id))
    }

    /// Get the genesis version
    pub fn genesis(&self) -> Option<&Version> {
        self.genesis_version_id
            .as_ref()
            .and_then(|id| self.versions.get(id))
            .map(|e| &e.version)
    }

    /// Get a version by ID
    pub fn get(&self, id: &str) -> Option<&HistoryEntry> {
        self.versions.get(id)
    }

    /// Get the previous version (before current)
    pub fn previous(&self) -> Option<&Version> {
        self.current()
            .and_then(|v| v.previous_version.as_ref())
            .and_then(|id| self.versions.get(&id.to_string()))
            .map(|e| &e.version)
    }

    /// Get the previous version entry
    pub fn previous_entry(&self) -> Option<&HistoryEntry> {
        self.current()
            .and_then(|v| v.previous_version.as_ref())
            .and_then(|id| self.versions.get(&id.to_string()))
    }

    /// Query the history
    pub fn query(&self, query: HistoryQuery) -> Vec<&HistoryEntry> {
        let mut results: Vec<_> = self
            .versions
            .values()
            .filter(|e| {
                // Filter by time
                if let Some(after) = &query.after {
                    if e.added_at <= *after {
                        return false;
                    }
                }
                if let Some(before) = &query.before {
                    if e.added_at >= *before {
                        return false;
                    }
                }

                // Filter by status
                if let Some(status_filter) = &query.status {
                    match (&e.status, status_filter) {
                        (VersionStatus::Active, VersionStatusFilter::Active) => {}
                        (VersionStatus::Superseded { .. }, VersionStatusFilter::Superseded) => {}
                        (VersionStatus::RolledBack { .. }, VersionStatusFilter::RolledBack) => {}
                        _ => return false,
                    }
                }

                true
            })
            .collect();

        // Sort by time (most recent first)
        results.sort_by(|a, b| b.added_at.cmp(&a.added_at));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        results
    }

    /// Get all versions in chronological order
    pub fn all_versions(&self) -> Vec<&Version> {
        let mut entries: Vec<_> = self.versions.values().collect();
        entries.sort_by(|a, b| a.added_at.cmp(&b.added_at));
        entries.iter().map(|e| &e.version).collect()
    }

    /// Get the certificate chain
    pub fn certificate_chain(&self) -> &CertificateChain {
        &self.certificates
    }

    /// Verify the integrity of the entire history
    pub fn verify_integrity(&self) -> bool {
        let chain_valid = self.certificates.verify_chain().valid;
        if !chain_valid {
            return false;
        }

        // Verify each version's certificate matches
        for entry in self.versions.values() {
            if entry.certificate.version_id != entry.version.id {
                return false;
            }
        }

        true
    }

    /// Get the count of versions in history
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Mark a version as rolled back
    pub fn mark_rolled_back(&mut self, id: &str, reason: String) -> SelfImpResult<()> {
        let entry = self
            .versions
            .get_mut(id)
            .ok_or_else(|| SelfImpError::VersionNotFound(id.to_string()))?;

        entry.mark_rolled_back(reason);

        Ok(())
    }

    /// Set the current version (for rollback)
    pub fn set_current(&mut self, id: &str) -> SelfImpResult<()> {
        if !self.versions.contains_key(id) {
            return Err(SelfImpError::VersionNotFound(id.to_string()));
        }

        self.current_version_id = Some(id.to_string());
        Ok(())
    }

    /// Get next certificate hash for chaining
    pub fn next_certificate_hash(&self) -> Option<String> {
        self.certificates.next_previous_hash()
    }
}
