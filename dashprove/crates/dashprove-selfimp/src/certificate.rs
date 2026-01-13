//! Proof certificates for verified versions

use crate::error::{SelfImpError, SelfImpResult};
use crate::version::{Version, VersionId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// A cryptographically verifiable proof certificate
///
/// Each certificate attests that a version passed all verification checks.
/// Certificates are chained to form a tamper-evident history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    /// Unique ID for this certificate
    pub id: String,

    /// Version this certificate is for
    pub version_id: VersionId,

    /// Hash of the previous certificate (None for genesis)
    pub previous_certificate_hash: Option<String>,

    /// When the verification was performed
    pub timestamp: DateTime<Utc>,

    /// Results of each verification check
    pub checks: Vec<CertificateCheck>,

    /// Overall verification result
    pub result: CertificateResult,

    /// Hash of this certificate's contents (for chain integrity)
    pub content_hash: String,

    /// Cryptographic signature (placeholder for future HSM integration)
    pub signature: Option<String>,
}

impl ProofCertificate {
    /// Create a new proof certificate
    pub fn new(
        version: &Version,
        checks: Vec<CertificateCheck>,
        previous_cert_hash: Option<String>,
    ) -> Self {
        let timestamp = Utc::now();
        let all_passed = checks.iter().all(|c| c.passed);

        let result = if all_passed {
            CertificateResult::Verified
        } else {
            let failed_checks: Vec<String> = checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.name.clone())
                .collect();
            CertificateResult::Failed { failed_checks }
        };

        // Calculate content hash
        let content_to_hash = format!(
            "{}:{}:{}:{}",
            version.id,
            timestamp.timestamp(),
            previous_cert_hash.as_deref().unwrap_or("genesis"),
            checks.len()
        );
        let mut hasher = Sha256::new();
        hasher.update(content_to_hash.as_bytes());
        let content_hash = hex::encode(hasher.finalize());

        // Generate certificate ID
        let id = format!("cert-{}", &content_hash[..16]);

        Self {
            id,
            version_id: version.id.clone(),
            previous_certificate_hash: previous_cert_hash,
            timestamp,
            checks,
            result,
            content_hash,
            signature: None,
        }
    }

    /// Check if the certificate represents a successful verification
    pub fn is_verified(&self) -> bool {
        matches!(self.result, CertificateResult::Verified)
    }

    /// Get the names of failed checks
    pub fn failed_checks(&self) -> Vec<&str> {
        match &self.result {
            CertificateResult::Verified => vec![],
            CertificateResult::Failed { failed_checks } => {
                failed_checks.iter().map(|s| s.as_str()).collect()
            }
        }
    }

    /// Verify the certificate's integrity (content hash matches)
    pub fn verify_integrity(&self, version: &Version) -> bool {
        let content_to_hash = format!(
            "{}:{}:{}:{}",
            version.id,
            self.timestamp.timestamp(),
            self.previous_certificate_hash
                .as_deref()
                .unwrap_or("genesis"),
            self.checks.len()
        );
        let mut hasher = Sha256::new();
        hasher.update(content_to_hash.as_bytes());
        let expected_hash = hex::encode(hasher.finalize());

        self.content_hash == expected_hash
    }

    /// Sign the certificate (placeholder for HSM integration)
    pub fn sign(&mut self, _key: &str) {
        // In production, this would use proper cryptographic signing
        // For now, we just create a placeholder signature
        let sig_content = format!("{}:{}", self.content_hash, self.timestamp.timestamp());
        let mut hasher = Sha256::new();
        hasher.update(sig_content.as_bytes());
        self.signature = Some(hex::encode(hasher.finalize()));
    }
}

/// Result of a single verification check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateCheck {
    /// Name of the check
    pub name: String,

    /// Whether the check passed
    pub passed: bool,

    /// Details about the check result
    pub details: Option<String>,

    /// Duration of the check in milliseconds
    pub duration_ms: Option<u64>,

    /// Which backend performed the check
    pub backend: Option<String>,
}

impl CertificateCheck {
    /// Create a passed check
    pub fn passed(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            details: None,
            duration_ms: None,
            backend: None,
        }
    }

    /// Create a failed check
    pub fn failed(name: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            details: Some(details.into()),
            duration_ms: None,
            backend: None,
        }
    }

    /// Add details to this check
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Add duration to this check
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    /// Add backend info to this check
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }
}

/// Overall result of certificate verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificateResult {
    /// All checks passed
    Verified,
    /// Some checks failed
    Failed {
        /// Names of checks that failed
        failed_checks: Vec<String>,
    },
}

/// A chain of proof certificates
///
/// The chain provides tamper-evident history of all verified versions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CertificateChain {
    /// All certificates in the chain, keyed by version ID
    pub certificates: HashMap<String, ProofCertificate>,

    /// The current head of the chain
    pub head: Option<String>,
}

impl CertificateChain {
    /// Create a new empty chain
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a certificate to the chain
    pub fn add(&mut self, cert: ProofCertificate) -> SelfImpResult<()> {
        // Verify chain linkage
        if let Some(head_id) = &self.head {
            let head = self.certificates.get(head_id).ok_or_else(|| {
                SelfImpError::HistoryCorruption("Head certificate not found".to_string())
            })?;

            if cert.previous_certificate_hash.as_ref() != Some(&head.content_hash) {
                return Err(SelfImpError::BrokenCertificateChain {
                    version: cert.version_id.to_string(),
                    reason: "Previous hash does not match head".to_string(),
                });
            }
        } else if cert.previous_certificate_hash.is_some() {
            return Err(SelfImpError::BrokenCertificateChain {
                version: cert.version_id.to_string(),
                reason: "First certificate should have no previous hash".to_string(),
            });
        }

        let version_id = cert.version_id.to_string();
        self.certificates.insert(version_id.clone(), cert);
        self.head = Some(version_id);

        Ok(())
    }

    /// Get a certificate by version ID
    pub fn get(&self, version_id: &str) -> Option<&ProofCertificate> {
        self.certificates.get(version_id)
    }

    /// Get the current head certificate
    pub fn current(&self) -> Option<&ProofCertificate> {
        self.head.as_ref().and_then(|id| self.certificates.get(id))
    }

    /// Verify the entire chain's integrity
    pub fn verify_chain(&self) -> CertificateVerification {
        if self.certificates.is_empty() {
            return CertificateVerification {
                valid: true,
                verified_count: 0,
                error: None,
            };
        }

        let mut verified_count = 0;
        let mut current_hash: Option<String> = None;

        // Walk chain from genesis to head
        let mut certs: Vec<_> = self.certificates.values().collect();
        certs.sort_by_key(|c| c.timestamp);

        for cert in certs {
            // Check previous hash linkage
            if cert.previous_certificate_hash != current_hash {
                return CertificateVerification {
                    valid: false,
                    verified_count,
                    error: Some(format!(
                        "Chain broken at {}: expected {:?}, got {:?}",
                        cert.version_id, current_hash, cert.previous_certificate_hash
                    )),
                };
            }

            current_hash = Some(cert.content_hash.clone());
            verified_count += 1;
        }

        CertificateVerification {
            valid: true,
            verified_count,
            error: None,
        }
    }

    /// Get all certificates as a vector (in chronological order)
    pub fn all_certificates(&self) -> Vec<&ProofCertificate> {
        let mut certs: Vec<_> = self.certificates.values().collect();
        certs.sort_by_key(|c| c.timestamp);
        certs
    }

    /// Get the certificate hash that should be used as previous for a new certificate
    pub fn next_previous_hash(&self) -> Option<String> {
        self.current().map(|c| c.content_hash.clone())
    }
}

/// Result of verifying a certificate chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateVerification {
    /// Whether the chain is valid
    pub valid: bool,

    /// Number of certificates verified
    pub verified_count: usize,

    /// Error message if invalid
    pub error: Option<String>,
}
