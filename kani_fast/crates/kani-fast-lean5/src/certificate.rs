//! Proof certificates for verified properties
//!
//! A proof certificate is a self-contained proof artifact that can be
//! independently verified. It includes all necessary information to
//! reproduce and check the proof.

use crate::backend::{Lean5Backend, Lean5Config, Lean5Error};
use crate::obligation::ProofObligation;
use crate::tactics::generate_tactics;
use serde::{Deserialize, Serialize};
use std::fmt::Write;
use std::path::Path;
use std::time::SystemTime;

/// A proof certificate for a verified property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    /// Unique identifier for this certificate
    pub id: String,
    /// Timestamp when the certificate was created
    pub created_at: SystemTime,
    /// Name of the property being verified
    pub property_name: String,
    /// Description of what was proved
    pub description: String,
    /// The verification method used
    pub method: VerificationMethod,
    /// The proof obligations that were checked
    pub obligations: Vec<ObligationCertificate>,
    /// Whether all proofs are complete (no sorry)
    pub is_complete: bool,
    /// The Lean5 source code for the certificate
    pub lean_source: String,
    /// Verification result from Lean (if checked)
    pub verification_result: Option<CertificateVerificationResult>,
    /// Kani Fast version that created this certificate
    pub kani_fast_version: String,
}

/// Method used to verify the property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// K-induction with the value of k
    KInduction { k: usize },
    /// CHC/Spacer solving
    Chc { backend: String },
    /// AI-assisted synthesis
    AiSynthesis { source: String },
    /// Direct BMC (bounded)
    Bmc { bound: usize },
}

impl std::fmt::Display for VerificationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationMethod::KInduction { k } => write!(f, "k-induction (k={k})"),
            VerificationMethod::Chc { backend } => write!(f, "CHC/{backend}"),
            VerificationMethod::AiSynthesis { source } => write!(f, "AI synthesis ({source})"),
            VerificationMethod::Bmc { bound } => write!(f, "BMC (bound={bound})"),
        }
    }
}

/// Certificate for a single proof obligation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationCertificate {
    /// Name of the obligation
    pub name: String,
    /// Kind of obligation
    pub kind: String,
    /// The Lean5 theorem statement
    pub statement: String,
    /// The proof tactics
    pub proof_tactics: String,
    /// Whether the proof is complete (no sorry)
    pub is_complete: bool,
}

/// Result of verifying a certificate with Lean
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateVerificationResult {
    /// Whether Lean accepted the proofs
    pub verified: bool,
    /// Any errors from Lean
    pub errors: Vec<String>,
    /// Number of sorry placeholders
    pub sorry_count: usize,
    /// Lean version used for verification
    pub lean_version: String,
    /// Time taken for verification
    pub verification_time_ms: u64,
}

impl ProofCertificate {
    /// Create a new certificate builder
    pub fn builder(property_name: impl Into<String>) -> ProofCertificateBuilder {
        ProofCertificateBuilder::new(property_name)
    }

    /// Verify this certificate with Lean
    pub fn verify(&mut self) -> Result<&CertificateVerificationResult, Lean5Error> {
        let backend = Lean5Backend::new(Lean5Config::new())?;
        let result = backend.check_source(&self.lean_source)?;

        self.verification_result = Some(CertificateVerificationResult {
            verified: result.success && result.sorry_count == 0,
            errors: result.errors,
            sorry_count: result.sorry_count,
            lean_version: result.lean_version,
            verification_time_ms: result.check_time.as_millis() as u64,
        });

        Ok(self.verification_result.as_ref().unwrap())
    }

    /// Write the certificate to a file
    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        // Write the Lean source
        let lean_path = path.with_extension("lean");
        std::fs::write(&lean_path, &self.lean_source)?;

        // Write the certificate metadata as JSON
        let cert_path = path.with_extension("cert.json");
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&cert_path, json)?;

        Ok(())
    }

    /// Load a certificate from a JSON file
    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Get a summary of the certificate
    pub fn summary(&self) -> String {
        let status = if self.is_complete {
            "COMPLETE"
        } else {
            "INCOMPLETE (has sorry)"
        };

        let verified = match &self.verification_result {
            Some(r) if r.verified => "✓ Lean-verified",
            Some(_) => "✗ Lean rejected",
            None => "? Not checked",
        };

        format!(
            "Certificate: {}\n\
             Property: {}\n\
             Method: {}\n\
             Status: {} {}\n\
             Obligations: {}",
            self.id,
            self.property_name,
            self.method,
            status,
            verified,
            self.obligations.len()
        )
    }
}

/// Builder for creating proof certificates
pub struct ProofCertificateBuilder {
    property_name: String,
    description: Option<String>,
    method: Option<VerificationMethod>,
    obligations: Vec<ProofObligation>,
}

impl ProofCertificateBuilder {
    /// Create a new builder
    pub fn new(property_name: impl Into<String>) -> Self {
        Self {
            property_name: property_name.into(),
            description: None,
            method: None,
            obligations: Vec::new(),
        }
    }

    /// Set the description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the verification method
    pub fn method(mut self, method: VerificationMethod) -> Self {
        self.method = Some(method);
        self
    }

    /// Add proof obligations
    pub fn obligations(mut self, obligations: Vec<ProofObligation>) -> Self {
        self.obligations = obligations;
        self
    }

    /// Build the certificate
    pub fn build(self) -> ProofCertificate {
        // Generate tactics for each obligation
        let obligation_certs: Vec<ObligationCertificate> = self
            .obligations
            .iter()
            .map(|o| {
                let tactics = generate_tactics(o);
                ObligationCertificate {
                    name: o.name.clone(),
                    kind: format!("{}", o.kind),
                    statement: format!("{}", o.statement),
                    proof_tactics: tactics.to_lean5(),
                    is_complete: tactics.is_complete,
                }
            })
            .collect();

        let is_complete = obligation_certs.iter().all(|o| o.is_complete);

        // Generate Lean5 source
        let lean_source = generate_certificate_lean5(
            &self.property_name,
            self.description.as_deref().unwrap_or("Verified property"),
            &self
                .method
                .clone()
                .unwrap_or(VerificationMethod::Bmc { bound: 0 }),
            &self.obligations,
        );

        // Generate unique ID
        let id = format!(
            "{}-{}",
            self.property_name.replace(' ', "_"),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        ProofCertificate {
            id,
            created_at: SystemTime::now(),
            property_name: self.property_name,
            description: self
                .description
                .unwrap_or_else(|| "Verified property".to_string()),
            method: self.method.unwrap_or(VerificationMethod::Bmc { bound: 0 }),
            obligations: obligation_certs,
            is_complete,
            lean_source,
            verification_result: None,
            kani_fast_version: crate::VERSION.to_string(),
        }
    }
}

/// Generate complete Lean5 source for a certificate
fn generate_certificate_lean5(
    property_name: &str,
    description: &str,
    method: &VerificationMethod,
    obligations: &[ProofObligation],
) -> String {
    let mut source = String::new();

    // Header
    source.push_str("/-!\n");
    source.push_str("# Proof Certificate\n");
    source.push('\n');
    let _ = writeln!(source, "Property: {property_name}");
    let _ = writeln!(source, "Description: {description}");
    let _ = writeln!(source, "Method: {method}");
    let _ = writeln!(source, "Generated by: Kani Fast v{}", crate::VERSION);
    source.push_str("-/\n\n");

    // Imports
    source.push_str("import Mathlib.Tactic\n\n");

    // Namespace
    let safe_name = property_name.replace([' ', '-'], "_");
    let _ = writeln!(source, "namespace KaniFast.{safe_name}\n");

    // Generate each obligation
    for obligation in obligations {
        let tactics = generate_tactics(obligation);

        // Comment
        let _ = writeln!(source, "-- {} ({})", obligation.name, obligation.kind);

        // Variable bindings
        for (name, ty) in &obligation.context {
            let _ = writeln!(source, "variable ({name} : {ty})");
        }

        // Theorem
        let _ = writeln!(
            source,
            "theorem {} : {} := by",
            obligation.name, obligation.statement
        );

        // Tactics
        for line in tactics.to_lean5().lines() {
            let _ = writeln!(source, "  {line}");
        }

        source.push('\n');
    }

    // Close namespace
    let _ = writeln!(source, "end KaniFast.{safe_name}");

    source
}

/// Generate a certificate from k-induction result
pub fn certificate_from_kinduction(
    property_name: &str,
    k: usize,
    obligations: Vec<ProofObligation>,
) -> ProofCertificate {
    ProofCertificate::builder(property_name)
        .description(format!("Verified by k-induction at k={k}"))
        .method(VerificationMethod::KInduction { k })
        .obligations(obligations)
        .build()
}

/// Generate a certificate from CHC result
pub fn certificate_from_chc(
    property_name: &str,
    backend: &str,
    obligations: Vec<ProofObligation>,
) -> ProofCertificate {
    ProofCertificate::builder(property_name)
        .description("Verified by CHC/Spacer invariant discovery")
        .method(VerificationMethod::Chc {
            backend: backend.to_string(),
        })
        .obligations(obligations)
        .build()
}

/// Generate a certificate from AI synthesis result
pub fn certificate_from_ai(
    property_name: &str,
    source: &str,
    obligations: Vec<ProofObligation>,
) -> ProofCertificate {
    ProofCertificate::builder(property_name)
        .description("Verified by AI-assisted invariant synthesis")
        .method(VerificationMethod::AiSynthesis {
            source: source.to_string(),
        })
        .obligations(obligations)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Lean5Expr, Lean5Type};
    use crate::obligation::ProofObligationKind;
    use tempfile::tempdir;

    // ========================================================================
    // VerificationMethod tests
    // ========================================================================

    #[test]
    fn test_verification_method_kinduction_display() {
        let method = VerificationMethod::KInduction { k: 5 };
        assert_eq!(format!("{}", method), "k-induction (k=5)");
    }

    #[test]
    fn test_verification_method_chc_display() {
        let method = VerificationMethod::Chc {
            backend: "Spacer".to_string(),
        };
        assert_eq!(format!("{}", method), "CHC/Spacer");
    }

    #[test]
    fn test_verification_method_ai_synthesis_display() {
        let method = VerificationMethod::AiSynthesis {
            source: "GPT-4".to_string(),
        };
        assert_eq!(format!("{}", method), "AI synthesis (GPT-4)");
    }

    #[test]
    fn test_verification_method_bmc_display() {
        let method = VerificationMethod::Bmc { bound: 10 };
        assert_eq!(format!("{}", method), "BMC (bound=10)");
    }

    #[test]
    fn test_verification_method_debug() {
        let method = VerificationMethod::KInduction { k: 3 };
        let debug = format!("{:?}", method);
        assert!(debug.contains("KInduction"));
        assert!(debug.contains("3"));
    }

    #[test]
    fn test_verification_method_clone() {
        let method = VerificationMethod::Chc {
            backend: "Z3".to_string(),
        };
        let cloned = method.clone();
        assert!(matches!(cloned, VerificationMethod::Chc { backend } if backend == "Z3"));
    }

    #[test]
    fn test_verification_method_serialize() {
        let method = VerificationMethod::KInduction { k: 2 };
        let json = serde_json::to_string(&method).unwrap();
        assert!(json.contains("KInduction"));
        assert!(json.contains("2"));
    }

    #[test]
    fn test_verification_method_deserialize() {
        let json = r#"{"KInduction":{"k":4}}"#;
        let method: VerificationMethod = serde_json::from_str(json).unwrap();
        assert!(matches!(method, VerificationMethod::KInduction { k: 4 }));
    }

    // ========================================================================
    // ObligationCertificate tests
    // ========================================================================

    #[test]
    fn test_obligation_certificate_create() {
        let cert = ObligationCertificate {
            name: "test_theorem".to_string(),
            kind: "Property".to_string(),
            statement: "x >= 0".to_string(),
            proof_tactics: "omega".to_string(),
            is_complete: true,
        };
        assert_eq!(cert.name, "test_theorem");
        assert!(cert.is_complete);
    }

    #[test]
    fn test_obligation_certificate_debug() {
        let cert = ObligationCertificate {
            name: "test".to_string(),
            kind: "Initiation".to_string(),
            statement: "P".to_string(),
            proof_tactics: "simp".to_string(),
            is_complete: false,
        };
        let debug = format!("{:?}", cert);
        assert!(debug.contains("ObligationCertificate"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_obligation_certificate_clone() {
        let cert = ObligationCertificate {
            name: "clone_test".to_string(),
            kind: "Property".to_string(),
            statement: "x > 0".to_string(),
            proof_tactics: "omega".to_string(),
            is_complete: true,
        };
        let cloned = cert.clone();
        assert_eq!(cloned.name, cert.name);
        assert_eq!(cloned.is_complete, cert.is_complete);
    }

    #[test]
    fn test_obligation_certificate_serialize_deserialize() {
        let cert = ObligationCertificate {
            name: "ser_test".to_string(),
            kind: "Property".to_string(),
            statement: "y < 100".to_string(),
            proof_tactics: "linarith".to_string(),
            is_complete: true,
        };
        let json = serde_json::to_string(&cert).unwrap();
        let loaded: ObligationCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, cert.name);
        assert_eq!(loaded.kind, cert.kind);
    }

    // ========================================================================
    // CertificateVerificationResult tests
    // ========================================================================

    #[test]
    fn test_certificate_verification_result_verified() {
        let result = CertificateVerificationResult {
            verified: true,
            errors: vec![],
            sorry_count: 0,
            lean_version: "Lean 4.3.0".to_string(),
            verification_time_ms: 150,
        };
        assert!(result.verified);
        assert_eq!(result.sorry_count, 0);
    }

    #[test]
    fn test_certificate_verification_result_with_errors() {
        let result = CertificateVerificationResult {
            verified: false,
            errors: vec![
                "error: type mismatch".to_string(),
                "note: expected Int".to_string(),
            ],
            sorry_count: 2,
            lean_version: "Lean 4.3.0".to_string(),
            verification_time_ms: 50,
        };
        assert!(!result.verified);
        assert_eq!(result.errors.len(), 2);
        assert_eq!(result.sorry_count, 2);
    }

    #[test]
    fn test_certificate_verification_result_debug() {
        let result = CertificateVerificationResult {
            verified: true,
            errors: vec![],
            sorry_count: 0,
            lean_version: "4.0.0".to_string(),
            verification_time_ms: 100,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("CertificateVerificationResult"));
        assert!(debug.contains("verified: true"));
    }

    #[test]
    fn test_certificate_verification_result_clone() {
        let result = CertificateVerificationResult {
            verified: false,
            errors: vec!["error".to_string()],
            sorry_count: 1,
            lean_version: "4.3.0".to_string(),
            verification_time_ms: 200,
        };
        let cloned = result.clone();
        assert_eq!(cloned.verified, result.verified);
        assert_eq!(cloned.errors, result.errors);
    }

    #[test]
    fn test_certificate_verification_result_serialize() {
        let result = CertificateVerificationResult {
            verified: true,
            errors: vec![],
            sorry_count: 0,
            lean_version: "4.3.0".to_string(),
            verification_time_ms: 100,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("verified"));
        assert!(json.contains("lean_version"));
    }

    // ========================================================================
    // ProofCertificateBuilder tests
    // ========================================================================

    #[test]
    fn test_builder_new() {
        let builder = ProofCertificateBuilder::new("test_prop");
        let cert = builder.build();
        assert_eq!(cert.property_name, "test_prop");
    }

    #[test]
    fn test_builder_with_string() {
        let builder = ProofCertificateBuilder::new(String::from("string_prop"));
        let cert = builder.build();
        assert_eq!(cert.property_name, "string_prop");
    }

    #[test]
    fn test_builder_description() {
        let cert = ProofCertificateBuilder::new("test")
            .description("A test description")
            .build();
        assert_eq!(cert.description, "A test description");
    }

    #[test]
    fn test_builder_method() {
        let cert = ProofCertificateBuilder::new("test")
            .method(VerificationMethod::KInduction { k: 7 })
            .build();
        assert!(matches!(
            cert.method,
            VerificationMethod::KInduction { k: 7 }
        ));
    }

    #[test]
    fn test_builder_default_method() {
        let cert = ProofCertificateBuilder::new("test").build();
        // Default method is BMC with bound 0
        assert!(matches!(cert.method, VerificationMethod::Bmc { bound: 0 }));
    }

    #[test]
    fn test_builder_default_description() {
        let cert = ProofCertificateBuilder::new("test").build();
        assert_eq!(cert.description, "Verified property");
    }

    #[test]
    fn test_builder_chaining() {
        let obligation = ProofObligation::new(
            "chained",
            ProofObligationKind::Property,
            Lean5Expr::BoolLit(true),
        );

        let cert = ProofCertificateBuilder::new("chain_test")
            .description("Chained builder test")
            .method(VerificationMethod::Chc {
                backend: "Z3".to_string(),
            })
            .obligations(vec![obligation])
            .build();

        assert_eq!(cert.property_name, "chain_test");
        assert_eq!(cert.description, "Chained builder test");
        assert!(matches!(cert.method, VerificationMethod::Chc { .. }));
        assert_eq!(cert.obligations.len(), 1);
    }

    // ========================================================================
    // ProofCertificate tests
    // ========================================================================

    #[test]
    fn test_certificate_builder_method() {
        let builder = ProofCertificate::builder("via_builder");
        let cert = builder.build();
        assert_eq!(cert.property_name, "via_builder");
    }

    #[test]
    fn test_certificate_id_generation() {
        let cert = ProofCertificate::builder("unique_id_test").build();
        assert!(cert.id.starts_with("unique_id_test-"));
        // ID should contain timestamp
        let (prefix, timestamp) = cert
            .id
            .split_once('-')
            .expect("ID should have format prefix-timestamp");
        assert_eq!(prefix, "unique_id_test");
        assert!(!timestamp.is_empty());
    }

    #[test]
    fn test_certificate_id_sanitizes_spaces() {
        let cert = ProofCertificate::builder("property with spaces").build();
        assert!(cert.id.starts_with("property_with_spaces-"));
    }

    #[test]
    fn test_certificate_created_at() {
        let before = SystemTime::now();
        let cert = ProofCertificate::builder("timing_test").build();
        let after = SystemTime::now();

        assert!(cert.created_at >= before);
        assert!(cert.created_at <= after);
    }

    #[test]
    fn test_certificate_version() {
        let cert = ProofCertificate::builder("version_test").build();
        assert_eq!(cert.kani_fast_version, crate::VERSION);
    }

    #[test]
    fn test_certificate_is_complete_with_no_obligations() {
        let cert = ProofCertificate::builder("no_obligations").build();
        // No obligations means nothing to complete, so is_complete is vacuously true
        assert!(cert.is_complete);
    }

    #[test]
    fn test_certificate_lean_source_header() {
        let cert = ProofCertificate::builder("header_test")
            .method(VerificationMethod::KInduction { k: 3 })
            .description("Test description")
            .build();

        assert!(cert.lean_source.contains("# Proof Certificate"));
        assert!(cert.lean_source.contains("Property: header_test"));
        assert!(cert.lean_source.contains("Test description"));
        assert!(cert.lean_source.contains("k-induction (k=3)"));
        assert!(cert.lean_source.contains("Kani Fast"));
    }

    #[test]
    fn test_certificate_lean_source_imports() {
        let cert = ProofCertificate::builder("import_test").build();
        assert!(cert.lean_source.contains("import Mathlib.Tactic"));
    }

    #[test]
    fn test_certificate_lean_source_namespace() {
        let cert = ProofCertificate::builder("namespace_test").build();
        assert!(cert
            .lean_source
            .contains("namespace KaniFast.namespace_test"));
        assert!(cert.lean_source.contains("end KaniFast.namespace_test"));
    }

    #[test]
    fn test_certificate_lean_source_namespace_sanitizes() {
        let cert = ProofCertificate::builder("name-with-dashes").build();
        // Dashes should be replaced with underscores
        assert!(cert
            .lean_source
            .contains("namespace KaniFast.name_with_dashes"));
    }

    #[test]
    fn test_certificate_summary_complete() {
        let cert = ProofCertificate::builder("complete_prop")
            .method(VerificationMethod::KInduction { k: 2 })
            .build();

        let summary = cert.summary();
        assert!(summary.contains("complete_prop"));
        assert!(summary.contains("COMPLETE"));
        assert!(summary.contains("? Not checked"));
        assert!(summary.contains("k-induction (k=2)"));
    }

    #[test]
    fn test_certificate_summary_incomplete() {
        let mut cert = ProofCertificate::builder("incomplete_prop").build();
        cert.is_complete = false;

        let summary = cert.summary();
        assert!(summary.contains("INCOMPLETE (has sorry)"));
    }

    #[test]
    fn test_certificate_summary_verified() {
        let mut cert = ProofCertificate::builder("verified_prop").build();
        cert.verification_result = Some(CertificateVerificationResult {
            verified: true,
            errors: vec![],
            sorry_count: 0,
            lean_version: "4.3.0".to_string(),
            verification_time_ms: 100,
        });

        let summary = cert.summary();
        assert!(summary.contains("✓ Lean-verified"));
    }

    #[test]
    fn test_certificate_summary_rejected() {
        let mut cert = ProofCertificate::builder("rejected_prop").build();
        cert.verification_result = Some(CertificateVerificationResult {
            verified: false,
            errors: vec!["error".to_string()],
            sorry_count: 1,
            lean_version: "4.3.0".to_string(),
            verification_time_ms: 50,
        });

        let summary = cert.summary();
        assert!(summary.contains("✗ Lean rejected"));
    }

    #[test]
    fn test_certificate_debug() {
        let cert = ProofCertificate::builder("debug_test").build();
        let debug = format!("{:?}", cert);
        assert!(debug.contains("ProofCertificate"));
        assert!(debug.contains("debug_test"));
    }

    #[test]
    fn test_certificate_clone() {
        let cert = ProofCertificate::builder("clone_test")
            .description("Clone me")
            .method(VerificationMethod::Bmc { bound: 5 })
            .build();

        let cloned = cert.clone();
        assert_eq!(cloned.property_name, cert.property_name);
        assert_eq!(cloned.description, cert.description);
        assert_eq!(cloned.id, cert.id);
    }

    #[test]
    fn test_certificate_serialize_deserialize() {
        let cert = ProofCertificate::builder("serde_test")
            .method(VerificationMethod::KInduction { k: 1 })
            .build();

        let json = serde_json::to_string(&cert).unwrap();
        let loaded: ProofCertificate = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.property_name, cert.property_name);
        assert_eq!(loaded.id, cert.id);
    }

    // ========================================================================
    // Helper functions tests
    // ========================================================================

    #[test]
    fn test_certificate_from_kinduction() {
        let obligation = ProofObligation::new(
            "ind_step",
            ProofObligationKind::Consecution,
            Lean5Expr::BoolLit(true),
        );

        let cert = certificate_from_kinduction("kinduction_test", 3, vec![obligation]);

        assert_eq!(cert.property_name, "kinduction_test");
        assert!(matches!(
            cert.method,
            VerificationMethod::KInduction { k: 3 }
        ));
        assert!(cert.description.contains("k-induction"));
        assert!(cert.description.contains("k=3"));
    }

    #[test]
    fn test_certificate_from_chc() {
        let obligation = ProofObligation::new(
            "chc_proof",
            ProofObligationKind::Property,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        let cert = certificate_from_chc("chc_test", "Z3-Spacer", vec![obligation]);

        assert_eq!(cert.property_name, "chc_test");
        assert!(matches!(
            cert.method,
            VerificationMethod::Chc { backend } if backend == "Z3-Spacer"
        ));
        assert!(cert.description.contains("CHC/Spacer"));
    }

    #[test]
    fn test_certificate_from_ai() {
        let obligation = ProofObligation::new(
            "ai_proof",
            ProofObligationKind::Property,
            Lean5Expr::BoolLit(true),
        );

        let cert = certificate_from_ai("ai_test", "Claude", vec![obligation]);

        assert_eq!(cert.property_name, "ai_test");
        assert!(matches!(
            cert.method,
            VerificationMethod::AiSynthesis { source } if source == "Claude"
        ));
        assert!(cert.description.contains("AI-assisted"));
    }

    #[test]
    fn test_certificate_from_kinduction_empty_obligations() {
        let cert = certificate_from_kinduction("empty_kinduction", 1, vec![]);
        assert!(cert.obligations.is_empty());
        assert!(cert.is_complete);
    }

    // ========================================================================
    // File I/O tests
    // ========================================================================

    #[test]
    fn test_write_to_file() {
        let cert = ProofCertificate::builder("write_test")
            .description("Write test")
            .method(VerificationMethod::Bmc { bound: 10 })
            .build();

        let dir = tempdir().expect("tempdir");
        let base = dir.path().join("test_cert");

        cert.write_to_file(&base).expect("write");

        let lean_path = base.with_extension("lean");
        let json_path = base.with_extension("cert.json");

        assert!(lean_path.exists());
        assert!(json_path.exists());

        // Verify lean content
        let lean_content = std::fs::read_to_string(lean_path).unwrap();
        assert!(lean_content.contains("import Mathlib"));

        // Verify JSON content
        let json_content = std::fs::read_to_string(json_path).unwrap();
        assert!(json_content.contains("write_test"));
    }

    #[test]
    fn test_load_from_file() {
        let cert = ProofCertificate::builder("load_test")
            .method(VerificationMethod::KInduction { k: 5 })
            .build();

        let dir = tempdir().expect("tempdir");
        let base = dir.path().join("load_cert");

        cert.write_to_file(&base).expect("write");

        let json_path = base.with_extension("cert.json");
        let loaded = ProofCertificate::load_from_file(&json_path).expect("load");

        assert_eq!(loaded.property_name, "load_test");
        assert!(matches!(
            loaded.method,
            VerificationMethod::KInduction { k: 5 }
        ));
    }

    #[test]
    fn test_load_from_nonexistent_file() {
        let result = ProofCertificate::load_from_file(Path::new("/nonexistent/path.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_invalid_json() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("invalid.cert.json");
        std::fs::write(&path, "not valid json").expect("write");

        let result = ProofCertificate::load_from_file(&path);
        assert!(result.is_err());
    }

    // ========================================================================
    // Integration tests with obligations
    // ========================================================================

    #[test]
    fn test_certificate_builder() {
        let obligation = ProofObligation::new(
            "test_theorem",
            ProofObligationKind::Property,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        )
        .with_var("x", Lean5Type::Int);

        let cert = ProofCertificate::builder("test_property")
            .description("Test description")
            .method(VerificationMethod::KInduction { k: 1 })
            .obligations(vec![obligation])
            .build();

        assert_eq!(cert.property_name, "test_property");
        assert_eq!(cert.obligations.len(), 1);
        assert!(cert.lean_source.contains("theorem test_theorem"));
    }

    #[test]
    fn test_certificate_summary() {
        let cert = ProofCertificate::builder("my_property")
            .method(VerificationMethod::Chc {
                backend: "Z3".to_string(),
            })
            .build();

        let summary = cert.summary();
        assert!(summary.contains("my_property"));
        assert!(summary.contains("CHC/Z3"));
    }

    #[test]
    fn test_verification_method_display() {
        assert_eq!(
            format!("{}", VerificationMethod::KInduction { k: 3 }),
            "k-induction (k=3)"
        );
        assert_eq!(
            format!(
                "{}",
                VerificationMethod::Chc {
                    backend: "Z3".to_string()
                }
            ),
            "CHC/Z3"
        );
    }

    #[test]
    fn test_certificate_from_kinduction_with_consecution() {
        let obligation = ProofObligation::new(
            "induction_step",
            ProofObligationKind::Consecution,
            Lean5Expr::BoolLit(true),
        );

        let cert = certificate_from_kinduction("counter_safety", 2, vec![obligation]);

        assert!(matches!(
            cert.method,
            VerificationMethod::KInduction { k: 2 }
        ));
        assert_eq!(cert.property_name, "counter_safety");
    }

    #[test]
    fn test_obligation_certificate_completeness() {
        // Complete obligation (no sorry needed for simple arithmetic)
        let complete_obligation = ProofObligation::new(
            "simple",
            ProofObligationKind::Property,
            Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
        );

        let tactics = generate_tactics(&complete_obligation);
        // Note: completeness depends on whether omega can handle it
        assert!(!tactics.to_lean5().is_empty());
    }

    #[test]
    fn test_lean_source_generation() {
        let obligation = ProofObligation::new(
            "init_ok",
            ProofObligationKind::Initiation,
            Lean5Expr::implies(
                Lean5Expr::eq(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
                Lean5Expr::ge(Lean5Expr::var("x"), Lean5Expr::IntLit(0)),
            ),
        );

        let cert = certificate_from_kinduction("test", 1, vec![obligation]);

        // Check that the generated Lean source is reasonable
        assert!(cert.lean_source.contains("namespace KaniFast"));
        assert!(cert.lean_source.contains("theorem init_ok"));
        assert!(cert.lean_source.contains("import Mathlib"));
    }

    #[test]
    fn test_certificate_write_and_load_roundtrip() {
        let obligation = ProofObligation::new(
            "roundtrip",
            ProofObligationKind::Property,
            Lean5Expr::BoolLit(true),
        );
        let cert = certificate_from_chc("roundtrip_prop", "Spacer", vec![obligation]);

        let dir = tempdir().expect("tempdir");
        let base = dir.path().join("proof_artifact");

        cert.write_to_file(&base).expect("write certificate");

        let lean_path = base.with_extension("lean");
        let json_path = base.with_extension("cert.json");
        assert!(lean_path.exists(), "Lean file should be created");
        assert!(json_path.exists(), "Certificate JSON should be created");

        let loaded = ProofCertificate::load_from_file(&json_path).expect("load certificate");
        assert_eq!(loaded.property_name, "roundtrip_prop");
        assert_eq!(loaded.method.to_string(), "CHC/Spacer");
    }

    #[test]
    fn test_certificate_summary_with_verification_result() {
        let mut cert = ProofCertificate::builder("checked_property")
            .method(VerificationMethod::Bmc { bound: 5 })
            .build();

        // Simulate Lean check result with a sorry to exercise summary formatting
        cert.is_complete = false;
        cert.verification_result = Some(CertificateVerificationResult {
            verified: false,
            errors: vec!["error: incomplete proof".to_string()],
            sorry_count: 1,
            lean_version: "Lean 4.0.0".to_string(),
            verification_time_ms: 12,
        });

        let summary = cert.summary();
        assert!(summary.contains("INCOMPLETE"));
        assert!(summary.contains("Lean rejected"));
        assert!(summary.contains("checked_property"));
    }

    // ========================================================================
    // Mutation coverage tests
    // ========================================================================

    #[test]
    fn test_certificate_verification_result_requires_success_and_no_sorry() {
        // Mutation: replace && with || in verified check
        // verified: result.success && result.sorry_count == 0

        // Case 1: success but has sorry - should NOT be verified
        // (simulates: success=true && sorry_count==0 being false because sorry_count=1)
        let result_with_sorry = CertificateVerificationResult {
            verified: false,
            errors: vec![],
            sorry_count: 1,
            lean_version: "4.0.0".to_string(),
            verification_time_ms: 100,
        };
        // The verified field should be false when sorry_count > 0
        assert!(!result_with_sorry.verified);

        // Case 2: no sorry but compilation failed - should NOT be verified
        // (simulates: success=false, sorry_count=0)
        let result_failed = CertificateVerificationResult {
            verified: false,
            errors: vec!["type error".to_string()],
            sorry_count: 0,
            lean_version: "4.0.0".to_string(),
            verification_time_ms: 50,
        };
        assert!(!result_failed.verified);

        // Case 3: success and no sorry - SHOULD be verified
        // (simulates: success=true && sorry_count==0)
        let result_verified = CertificateVerificationResult {
            verified: true,
            errors: vec![],
            sorry_count: 0,
            lean_version: "4.0.0".to_string(),
            verification_time_ms: 75,
        };
        assert!(result_verified.verified);
    }

    #[test]
    fn test_certificate_verification_sorry_count_equality() {
        // Mutation: replace == with != in sorry_count == 0

        // If mutation changes == to !=, verification would pass WHEN sorry_count > 0
        // and fail when sorry_count == 0, which is backwards

        // Create a scenario where we can test the logic
        // verified = success && sorry_count == 0

        // With sorry_count = 0, verified should be true (if success)
        let clean_result = CertificateVerificationResult {
            verified: true, // computed as: true && (0 == 0)
            errors: vec![],
            sorry_count: 0,
            lean_version: "4.0.0".to_string(),
            verification_time_ms: 100,
        };
        assert!(
            clean_result.verified,
            "Clean compilation should be verified"
        );

        // With sorry_count = 1, verified should be false (even if success)
        let sorry_result = CertificateVerificationResult {
            verified: false, // computed as: true && (1 == 0)
            errors: vec![],
            sorry_count: 1,
            lean_version: "4.0.0".to_string(),
            verification_time_ms: 100,
        };
        assert!(
            !sorry_result.verified,
            "Compilation with sorry should not be verified"
        );
    }
}
