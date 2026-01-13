//! Universal proof format definition
//!
//! The `UniversalProof` type is the core data structure for proof sharing across backends.

use crate::hash::{ContentAddressable, ContentHash};
use crate::step::ProofStep;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Verification backend identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendId {
    /// Kani Fast native verification
    KaniFast,
    /// Z3 SMT solver
    Z3,
    /// Z4 unified solver (future)
    Z4,
    /// Lean 5 theorem prover
    Lean5,
    /// TLA2 model checker (future)
    Tla2,
    /// CaDiCaL SAT solver
    CaDiCaL,
    /// Kissat SAT solver
    Kissat,
    /// Unknown/custom backend
    Custom(u32),
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KaniFast => write!(f, "KaniFast"),
            Self::Z3 => write!(f, "Z3"),
            Self::Z4 => write!(f, "Z4"),
            Self::Lean5 => write!(f, "Lean5"),
            Self::Tla2 => write!(f, "TLA2"),
            Self::CaDiCaL => write!(f, "CaDiCaL"),
            Self::Kissat => write!(f, "Kissat"),
            Self::Custom(id) => write!(f, "Custom({id})"),
        }
    }
}

/// Proof format type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofFormat {
    /// DRAT (SAT solver proofs)
    Drat,
    /// SMT-LIB2 proofs (Alethe-style)
    Smt,
    /// CHC (Spacer invariants)
    Chc,
    /// Lean proof terms
    Lean,
    /// Mixed format (multiple step types)
    Mixed,
}

impl std::fmt::Display for ProofFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Drat => write!(f, "DRAT"),
            Self::Smt => write!(f, "SMT"),
            Self::Chc => write!(f, "CHC"),
            Self::Lean => write!(f, "Lean"),
            Self::Mixed => write!(f, "Mixed"),
        }
    }
}

/// Metadata about the proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// When the proof was generated
    pub created_at: SystemTime,
    /// Time taken to generate the proof
    pub generation_time: Duration,
    /// Backend version string
    pub backend_version: String,
    /// Kani Fast version
    pub kani_fast_version: String,
    /// Number of proof steps
    pub step_count: usize,
    /// Total size in bytes (serialized)
    pub size_bytes: usize,
    /// Whether the proof is complete (no trust steps)
    pub is_complete: bool,
    /// Optional property name
    pub property_name: Option<String>,
    /// Optional description
    pub description: Option<String>,
}

impl Default for ProofMetadata {
    fn default() -> Self {
        Self {
            created_at: SystemTime::now(),
            generation_time: Duration::ZERO,
            backend_version: String::new(),
            kani_fast_version: crate::VERSION.to_string(),
            step_count: 0,
            size_bytes: 0,
            is_complete: true,
            property_name: None,
            description: None,
        }
    }
}

/// A universal proof that can be shared across backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalProof {
    /// Content-addressable identifier (hash of VC + proof + backend)
    pub id: ContentHash,
    /// Hash of the verification condition
    pub vc_hash: ContentHash,
    /// Backend that generated this proof
    pub backend: BackendId,
    /// Proof format
    pub format: ProofFormat,
    /// Verification condition (SMT-LIB2 format)
    pub vc: String,
    /// Proof steps
    pub steps: Vec<ProofStep>,
    /// Dependencies on other proofs (for composition)
    pub dependencies: Vec<ContentHash>,
    /// Proof metadata
    pub metadata: ProofMetadata,
}

impl UniversalProof {
    /// Create a new proof builder
    pub fn builder() -> UniversalProofBuilder {
        UniversalProofBuilder::new()
    }

    /// Check if this proof has any trust steps
    pub fn has_trust_steps(&self) -> bool {
        self.steps.iter().any(|s| s.is_trust())
    }

    /// Get the number of trust steps
    pub fn trust_step_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_trust()).count()
    }

    /// Check if this proof depends on others
    pub fn has_dependencies(&self) -> bool {
        !self.dependencies.is_empty()
    }

    /// Compute the hash for this proof (without the id field)
    fn compute_hash(&self) -> ContentHash {
        // Serialize components for hashing
        let backend_str = self.backend.to_string();
        let format_str = self.format.to_string();
        let dependencies_bytes: Vec<u8> = self
            .dependencies
            .iter()
            .flat_map(|d| d.as_bytes())
            .copied()
            .collect();
        let steps_str: String = self
            .steps
            .iter()
            .map(|s| serde_json::to_string(s).unwrap_or_default())
            .collect();

        ContentHash::from_components(&[
            self.vc.as_bytes(),
            backend_str.as_bytes(),
            format_str.as_bytes(),
            self.metadata.backend_version.as_bytes(),
            dependencies_bytes.as_slice(),
            steps_str.as_bytes(),
        ])
    }

    /// Verify the integrity of this proof (check that id matches content)
    pub fn verify_integrity(&self) -> bool {
        self.id == self.compute_hash()
    }

    /// Get a summary of this proof
    pub fn summary(&self) -> String {
        format!(
            "Proof {} ({} via {}): {} steps, {} dependencies",
            self.id.short_hex(),
            self.format,
            self.backend,
            self.steps.len(),
            self.dependencies.len(),
        )
    }

    /// Verify this proof using the proof checker
    ///
    /// This is a convenience method that creates a ProofChecker with default
    /// config and verifies the proof. For more control, use ProofChecker directly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let proof = UniversalProof::builder()
    ///     .format(ProofFormat::Chc)
    ///     .vc("(assert (>= x 0))")
    ///     .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
    ///     .build();
    ///
    /// let result = proof.verify().await?;
    /// assert!(result.is_valid);
    /// ```
    pub async fn verify(
        &self,
    ) -> Result<crate::checker::VerificationResult, crate::checker::CheckerError> {
        let checker = crate::checker::ProofChecker::new();
        checker.verify(self).await
    }

    /// Verify this proof with custom configuration
    pub async fn verify_with_config(
        &self,
        config: crate::checker::CheckerConfig,
    ) -> Result<crate::checker::VerificationResult, crate::checker::CheckerError> {
        let checker = crate::checker::ProofChecker::with_config(config);
        checker.verify(self).await
    }
}

impl ContentAddressable for UniversalProof {
    fn content_hash(&self) -> ContentHash {
        self.id
    }
}

/// Builder for creating universal proofs
#[derive(Debug, Default)]
pub struct UniversalProofBuilder {
    backend: Option<BackendId>,
    format: Option<ProofFormat>,
    vc: Option<String>,
    steps: Vec<ProofStep>,
    dependencies: Vec<ContentHash>,
    property_name: Option<String>,
    description: Option<String>,
    backend_version: Option<String>,
    generation_time: Option<Duration>,
}

impl UniversalProofBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the backend
    pub fn backend(mut self, backend: BackendId) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set the proof format
    pub fn format(mut self, format: ProofFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the verification condition
    pub fn vc(mut self, vc: impl Into<String>) -> Self {
        self.vc = Some(vc.into());
        self
    }

    /// Add a proof step
    pub fn step(mut self, step: ProofStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add multiple proof steps
    pub fn steps(mut self, steps: Vec<ProofStep>) -> Self {
        self.steps.extend(steps);
        self
    }

    /// Add a dependency on another proof
    pub fn dependency(mut self, dep: ContentHash) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Add multiple dependencies
    pub fn dependencies(mut self, deps: Vec<ContentHash>) -> Self {
        self.dependencies.extend(deps);
        self
    }

    /// Set the property name
    pub fn property_name(mut self, name: impl Into<String>) -> Self {
        self.property_name = Some(name.into());
        self
    }

    /// Set the description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the backend version
    pub fn backend_version(mut self, version: impl Into<String>) -> Self {
        self.backend_version = Some(version.into());
        self
    }

    /// Set the generation time
    pub fn generation_time(mut self, time: Duration) -> Self {
        self.generation_time = Some(time);
        self
    }

    /// Build the proof
    pub fn build(self) -> UniversalProof {
        let backend = self.backend.unwrap_or(BackendId::KaniFast);
        let format = self.format.unwrap_or(ProofFormat::Mixed);
        let vc = self.vc.unwrap_or_default();
        let is_complete = !self.steps.iter().any(|s| s.is_trust());

        let metadata = ProofMetadata {
            created_at: SystemTime::now(),
            generation_time: self.generation_time.unwrap_or(Duration::ZERO),
            backend_version: self.backend_version.unwrap_or_default(),
            kani_fast_version: crate::VERSION.to_string(),
            step_count: self.steps.len(),
            size_bytes: 0, // Will be updated after serialization
            is_complete,
            property_name: self.property_name,
            description: self.description,
        };

        let vc_hash = ContentHash::hash_str(&vc);

        // Create proof without id first
        let mut proof = UniversalProof {
            id: ContentHash::zero(),
            vc_hash,
            backend,
            format,
            vc,
            steps: self.steps,
            dependencies: self.dependencies,
            metadata,
        };

        // Compute and set the content hash
        proof.id = proof.compute_hash();

        // Update size estimate
        if let Ok(json) = serde_json::to_string(&proof) {
            proof.metadata.size_bytes = json.len();
        }

        proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::ChcStep;

    // ==================== BackendId Tests ====================

    #[test]
    fn test_backend_id_display() {
        assert_eq!(format!("{}", BackendId::KaniFast), "KaniFast");
        assert_eq!(format!("{}", BackendId::Z3), "Z3");
        assert_eq!(format!("{}", BackendId::Z4), "Z4");
        assert_eq!(format!("{}", BackendId::Lean5), "Lean5");
        assert_eq!(format!("{}", BackendId::Tla2), "TLA2");
        assert_eq!(format!("{}", BackendId::CaDiCaL), "CaDiCaL");
        assert_eq!(format!("{}", BackendId::Kissat), "Kissat");
        assert_eq!(format!("{}", BackendId::Custom(42)), "Custom(42)");
    }

    #[test]
    fn test_backend_id_serialize() {
        let backend = BackendId::KaniFast;
        let json = serde_json::to_string(&backend).unwrap();
        let parsed: BackendId = serde_json::from_str(&json).unwrap();
        assert_eq!(backend, parsed);
    }

    #[test]
    fn test_backend_id_eq() {
        assert_eq!(BackendId::Z3, BackendId::Z3);
        assert_ne!(BackendId::Z3, BackendId::Z4);
    }

    #[test]
    fn test_backend_id_clone() {
        let backend = BackendId::Lean5;
        let cloned = backend;
        assert_eq!(backend, cloned);
    }

    #[test]
    fn test_backend_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BackendId::Z3);
        assert!(set.contains(&BackendId::Z3));
        assert!(!set.contains(&BackendId::Z4));
    }

    // ==================== ProofFormat Tests ====================

    #[test]
    fn test_proof_format_display() {
        assert_eq!(format!("{}", ProofFormat::Drat), "DRAT");
        assert_eq!(format!("{}", ProofFormat::Smt), "SMT");
        assert_eq!(format!("{}", ProofFormat::Chc), "CHC");
        assert_eq!(format!("{}", ProofFormat::Lean), "Lean");
        assert_eq!(format!("{}", ProofFormat::Mixed), "Mixed");
    }

    #[test]
    fn test_proof_format_serialize() {
        let format = ProofFormat::Chc;
        let json = serde_json::to_string(&format).unwrap();
        let parsed: ProofFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(format, parsed);
    }

    #[test]
    fn test_proof_format_eq() {
        assert_eq!(ProofFormat::Chc, ProofFormat::Chc);
        assert_ne!(ProofFormat::Chc, ProofFormat::Drat);
    }

    // ==================== ProofMetadata Tests ====================

    #[test]
    fn test_proof_metadata_default() {
        let metadata = ProofMetadata::default();
        assert_eq!(metadata.step_count, 0);
        assert_eq!(metadata.size_bytes, 0);
        assert!(metadata.is_complete);
        assert!(metadata.property_name.is_none());
    }

    #[test]
    fn test_proof_metadata_serialize() {
        let metadata = ProofMetadata {
            property_name: Some("test_prop".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: ProofMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.property_name, Some("test_prop".to_string()));
    }

    #[test]
    fn test_proof_metadata_clone() {
        let metadata = ProofMetadata {
            step_count: 5,
            ..Default::default()
        };
        let cloned = metadata.clone();
        assert_eq!(cloned.step_count, 5);
    }

    // ==================== UniversalProofBuilder Tests ====================

    #[test]
    fn test_builder_minimal() {
        let proof = UniversalProof::builder().build();
        assert_eq!(proof.backend, BackendId::KaniFast);
        assert_eq!(proof.format, ProofFormat::Mixed);
        assert!(proof.vc.is_empty());
        assert!(proof.steps.is_empty());
    }

    #[test]
    fn test_builder_with_backend() {
        let proof = UniversalProof::builder().backend(BackendId::Z3).build();
        assert_eq!(proof.backend, BackendId::Z3);
    }

    #[test]
    fn test_builder_with_format() {
        let proof = UniversalProof::builder().format(ProofFormat::Chc).build();
        assert_eq!(proof.format, ProofFormat::Chc);
    }

    #[test]
    fn test_builder_with_vc() {
        let proof = UniversalProof::builder().vc("(assert (>= x 0))").build();
        assert_eq!(proof.vc, "(assert (>= x 0))");
    }

    #[test]
    fn test_builder_with_step() {
        let proof = UniversalProof::builder()
            .step(ProofStep::chc_invariant(
                "inv",
                vec!["x".to_string()],
                "(>= x 0)",
            ))
            .build();
        assert_eq!(proof.steps.len(), 1);
    }

    #[test]
    fn test_builder_with_steps() {
        let proof = UniversalProof::builder()
            .steps(vec![
                ProofStep::chc_invariant("inv1", vec![], "true"),
                ProofStep::chc_invariant("inv2", vec![], "true"),
            ])
            .build();
        assert_eq!(proof.steps.len(), 2);
    }

    #[test]
    fn test_builder_with_dependency() {
        let dep = ContentHash::hash_str("dep1");
        let proof = UniversalProof::builder().dependency(dep).build();
        assert_eq!(proof.dependencies.len(), 1);
        assert_eq!(proof.dependencies[0], dep);
    }

    #[test]
    fn test_builder_with_dependencies() {
        let dep1 = ContentHash::hash_str("dep1");
        let dep2 = ContentHash::hash_str("dep2");
        let proof = UniversalProof::builder()
            .dependencies(vec![dep1, dep2])
            .build();
        assert_eq!(proof.dependencies.len(), 2);
    }

    #[test]
    fn test_builder_with_property_name() {
        let proof = UniversalProof::builder().property_name("safety").build();
        assert_eq!(proof.metadata.property_name, Some("safety".to_string()));
    }

    #[test]
    fn test_builder_with_description() {
        let proof = UniversalProof::builder()
            .description("Proves x >= 0")
            .build();
        assert_eq!(
            proof.metadata.description,
            Some("Proves x >= 0".to_string())
        );
    }

    #[test]
    fn test_builder_with_backend_version() {
        let proof = UniversalProof::builder().backend_version("4.12.0").build();
        assert_eq!(proof.metadata.backend_version, "4.12.0");
    }

    #[test]
    fn test_builder_with_generation_time() {
        let proof = UniversalProof::builder()
            .generation_time(Duration::from_millis(100))
            .build();
        assert_eq!(proof.metadata.generation_time, Duration::from_millis(100));
    }

    #[test]
    fn test_builder_chaining() {
        let proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("(assert P)")
            .property_name("test")
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();

        assert_eq!(proof.backend, BackendId::Z3);
        assert_eq!(proof.format, ProofFormat::Chc);
        assert_eq!(proof.vc, "(assert P)");
        assert_eq!(proof.metadata.property_name, Some("test".to_string()));
        assert_eq!(proof.steps.len(), 1);
    }

    // ==================== UniversalProof Tests ====================

    #[test]
    fn test_proof_has_trust_steps() {
        let proof_without_trust = UniversalProof::builder()
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();
        assert!(!proof_without_trust.has_trust_steps());

        let proof_with_trust = UniversalProof::builder()
            .step(ProofStep::trust("external", "claim"))
            .build();
        assert!(proof_with_trust.has_trust_steps());
    }

    #[test]
    fn test_proof_trust_step_count() {
        let proof = UniversalProof::builder()
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .step(ProofStep::trust("reason1", "claim1"))
            .step(ProofStep::trust("reason2", "claim2"))
            .build();
        assert_eq!(proof.trust_step_count(), 2);
    }

    #[test]
    fn test_proof_has_dependencies() {
        let proof_no_deps = UniversalProof::builder().build();
        assert!(!proof_no_deps.has_dependencies());

        let proof_with_deps = UniversalProof::builder()
            .dependency(ContentHash::hash_str("dep"))
            .build();
        assert!(proof_with_deps.has_dependencies());
    }

    #[test]
    fn test_proof_hash_changes_with_dependencies() {
        let vc = "(assert true)";

        let proof_no_deps = UniversalProof::builder().vc(vc).build();
        let proof_with_dep = UniversalProof::builder()
            .vc(vc)
            .dependency(ContentHash::hash_str("dep"))
            .build();

        assert_ne!(proof_no_deps.id, proof_with_dep.id);
    }

    #[test]
    fn test_proof_is_complete() {
        let complete_proof = UniversalProof::builder()
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();
        assert!(complete_proof.metadata.is_complete);

        let incomplete_proof = UniversalProof::builder()
            .step(ProofStep::trust("reason", "claim"))
            .build();
        assert!(!incomplete_proof.metadata.is_complete);
    }

    #[test]
    fn test_proof_verify_integrity() {
        let proof = UniversalProof::builder()
            .vc("(assert true)")
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();
        assert!(proof.verify_integrity());
    }

    #[test]
    fn test_proof_verify_integrity_detects_dependency_change() {
        let dep = ContentHash::hash_str("dep");
        let mut proof = UniversalProof::builder()
            .vc("(assert true)")
            .dependency(dep)
            .build();

        assert!(proof.verify_integrity());

        proof.dependencies.push(ContentHash::hash_str("other"));
        assert!(!proof.verify_integrity());
    }

    #[test]
    fn test_proof_verify_integrity_tampered() {
        let mut proof = UniversalProof::builder().vc("(assert true)").build();
        // Tamper with the VC
        proof.vc = "(assert false)".to_string();
        assert!(!proof.verify_integrity());
    }

    #[test]
    fn test_proof_summary() {
        let proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .dependency(ContentHash::hash_str("dep"))
            .build();

        let summary = proof.summary();
        assert!(summary.contains("CHC"));
        assert!(summary.contains("Z3"));
        assert!(summary.contains("1 steps"));
        assert!(summary.contains("1 dependencies"));
    }

    #[test]
    fn test_proof_content_hash() {
        let proof = UniversalProof::builder().vc("(assert true)").build();
        assert_eq!(proof.content_hash(), proof.id);
    }

    #[test]
    fn test_proof_vc_hash() {
        let proof = UniversalProof::builder().vc("(assert true)").build();
        assert_eq!(proof.vc_hash, ContentHash::hash_str("(assert true)"));
    }

    #[test]
    fn test_proof_serialize_deserialize() {
        let proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::chc_invariant(
                "inv",
                vec!["x".to_string()],
                "(>= x 0)",
            ))
            .property_name("safety")
            .build();

        let json = serde_json::to_string(&proof).unwrap();
        let parsed: UniversalProof = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, proof.id);
        assert_eq!(parsed.backend, proof.backend);
        assert_eq!(parsed.format, proof.format);
        assert_eq!(parsed.vc, proof.vc);
        assert_eq!(parsed.steps.len(), proof.steps.len());
    }

    #[test]
    fn test_proof_clone() {
        let proof = UniversalProof::builder()
            .vc("(assert true)")
            .step(ProofStep::drat_add(vec![1, 2, 3]))
            .build();
        let cloned = proof.clone();
        assert_eq!(cloned.id, proof.id);
    }

    #[test]
    fn test_proof_id_deterministic() {
        // Same inputs should produce same ID
        let proof1 = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .backend_version("4.12.0")
            .build();

        let proof2 = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .backend_version("4.12.0")
            .build();

        assert_eq!(proof1.id, proof2.id);
    }

    #[test]
    fn test_proof_id_different_vc() {
        let proof1 = UniversalProof::builder().vc("(assert true)").build();

        let proof2 = UniversalProof::builder().vc("(assert false)").build();

        assert_ne!(proof1.id, proof2.id);
    }

    #[test]
    fn test_proof_metadata_size() {
        let proof = UniversalProof::builder()
            .vc("(assert true)")
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();

        // Size should be non-zero after building
        assert!(proof.metadata.size_bytes > 0);
    }

    #[test]
    fn test_proof_metadata_step_count() {
        let proof = UniversalProof::builder()
            .step(ProofStep::drat_add(vec![1]))
            .step(ProofStep::drat_add(vec![2]))
            .step(ProofStep::drat_add(vec![3]))
            .build();

        assert_eq!(proof.metadata.step_count, 3);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_chc_proof_workflow() {
        // Simulate a CHC proof workflow
        let proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("(assert (forall ((x Int)) (=> (= x 0) (>= x 0))))")
            .backend_version("4.12.0")
            .property_name("counter_safety")
            .description("Proves that counter is always non-negative")
            .step(ProofStep::Chc(ChcStep::invariant(
                "inv",
                vec!["x".to_string()],
                "(>= x 0)",
            )))
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .step(ProofStep::Chc(ChcStep::consecution(
                "(>= x 0)",
                "(= x' (+ x 1))",
                "(>= x' 0)",
            )))
            .step(ProofStep::Chc(ChcStep::property(
                "(>= x 0)",
                "(not (< x 0))",
            )))
            .generation_time(Duration::from_millis(50))
            .build();

        assert_eq!(proof.backend, BackendId::Z3);
        assert_eq!(proof.format, ProofFormat::Chc);
        assert_eq!(proof.steps.len(), 4);
        assert!(proof.metadata.is_complete);
        assert!(proof.verify_integrity());
    }

    #[test]
    fn test_composed_proof() {
        // Create a base proof
        let base_proof = UniversalProof::builder()
            .property_name("lemma1")
            .vc("(assert P)")
            .step(ProofStep::chc_invariant("inv", vec![], "P"))
            .build();

        // Create a proof that depends on the base
        let composed_proof = UniversalProof::builder()
            .property_name("theorem1")
            .vc("(assert Q)")
            .dependency(base_proof.id)
            .step(ProofStep::trust("uses lemma1", "P implies Q"))
            .build();

        assert!(composed_proof.has_dependencies());
        assert_eq!(composed_proof.dependencies[0], base_proof.id);
    }

    // ==================== Verify Method Tests ====================

    fn skip_if_z3_unavailable() -> bool {
        std::process::Command::new("z3")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
    }

    #[tokio::test]
    async fn test_verify_convenience_method() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .build();

        let result = proof.verify().await.unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_verify_with_config() {
        if skip_if_z3_unavailable() {
            return;
        }

        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Chc(ChcStep::initiation("(= x 0)", "(>= x 0)")))
            .build();

        let config = crate::checker::CheckerConfig {
            step_timeout: std::time::Duration::from_secs(10),
            ..Default::default()
        };

        let result = proof.verify_with_config(config).await.unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_verify_invalid_proof() {
        if skip_if_z3_unavailable() {
            return;
        }

        // This proof should fail because x=-1 doesn't satisfy x>=0
        let proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert (>= x 0))")
            .step(ProofStep::Chc(ChcStep::initiation(
                "(= x (- 1))",
                "(>= x 0)",
            )))
            .build();

        let result = proof.verify().await.unwrap();
        assert!(!result.is_valid);
    }

    // ==================== Proof Dependency/Composition Tests ====================

    #[test]
    fn test_multi_level_dependencies() {
        // Create a chain: proof_c depends on proof_b depends on proof_a
        let proof_a = UniversalProof::builder()
            .property_name("axiom_a")
            .vc("(assert (forall ((x Int)) (>= (* x x) 0)))")
            .step(ProofStep::chc_invariant(
                "square_nonneg",
                vec!["x".to_string()],
                "(>= (* x x) 0)",
            ))
            .build();

        let proof_b = UniversalProof::builder()
            .property_name("lemma_b")
            .vc("(assert (forall ((x Int)) (>= (+ (* x x) 1) 1)))")
            .dependency(proof_a.id)
            .step(ProofStep::trust(
                "uses axiom_a",
                "square >= 0 implies square + 1 >= 1",
            ))
            .build();

        let proof_c = UniversalProof::builder()
            .property_name("theorem_c")
            .vc("(assert (forall ((x Int)) (> (+ (* x x) 1) 0)))")
            .dependency(proof_b.id)
            .step(ProofStep::trust(
                "uses lemma_b",
                "square + 1 >= 1 implies square + 1 > 0",
            ))
            .build();

        // Verify dependency chain
        assert!(!proof_a.has_dependencies());
        assert!(proof_b.has_dependencies());
        assert!(proof_c.has_dependencies());
        assert_eq!(proof_b.dependencies[0], proof_a.id);
        assert_eq!(proof_c.dependencies[0], proof_b.id);

        // Each proof has unique hash
        assert_ne!(proof_a.id, proof_b.id);
        assert_ne!(proof_b.id, proof_c.id);
        assert_ne!(proof_a.id, proof_c.id);
    }

    #[test]
    fn test_diamond_dependencies() {
        // Create diamond: D depends on both A and B, both depend on C
        //       D
        //      / \
        //     A   B
        //      \ /
        //       C
        let proof_c = UniversalProof::builder()
            .property_name("base_c")
            .vc("(assert C)")
            .step(ProofStep::chc_invariant("inv_c", vec![], "C"))
            .build();

        let proof_a = UniversalProof::builder()
            .property_name("lemma_a")
            .vc("(assert A)")
            .dependency(proof_c.id)
            .step(ProofStep::trust("uses C", "C implies A"))
            .build();

        let proof_b = UniversalProof::builder()
            .property_name("lemma_b")
            .vc("(assert B)")
            .dependency(proof_c.id)
            .step(ProofStep::trust("uses C", "C implies B"))
            .build();

        let proof_d = UniversalProof::builder()
            .property_name("theorem_d")
            .vc("(assert D)")
            .dependency(proof_a.id)
            .dependency(proof_b.id)
            .step(ProofStep::trust("uses A and B", "A and B implies D"))
            .build();

        // Verify diamond structure
        assert_eq!(proof_a.dependencies.len(), 1);
        assert_eq!(proof_b.dependencies.len(), 1);
        assert_eq!(proof_d.dependencies.len(), 2);

        // Both A and B depend on C
        assert_eq!(proof_a.dependencies[0], proof_c.id);
        assert_eq!(proof_b.dependencies[0], proof_c.id);

        // D depends on both A and B
        assert!(proof_d.dependencies.contains(&proof_a.id));
        assert!(proof_d.dependencies.contains(&proof_b.id));
    }

    #[test]
    fn test_dependency_order_preserved() {
        let dep1 = ContentHash::hash_str("dep1");
        let dep2 = ContentHash::hash_str("dep2");
        let dep3 = ContentHash::hash_str("dep3");

        let proof = UniversalProof::builder()
            .vc("(assert true)")
            .dependency(dep1)
            .dependency(dep2)
            .dependency(dep3)
            .build();

        assert_eq!(proof.dependencies.len(), 3);
        assert_eq!(proof.dependencies[0], dep1);
        assert_eq!(proof.dependencies[1], dep2);
        assert_eq!(proof.dependencies[2], dep3);
    }

    #[test]
    fn test_dependency_affects_proof_hash() {
        // Same VC but different dependencies should produce different hashes
        let base_proof = UniversalProof::builder()
            .vc("(assert P)")
            .step(ProofStep::chc_invariant("inv", vec![], "P"))
            .build();

        let proof_with_dep = UniversalProof::builder()
            .vc("(assert P)")
            .step(ProofStep::chc_invariant("inv", vec![], "P"))
            .dependency(ContentHash::hash_str("some_dependency"))
            .build();

        assert_ne!(base_proof.id, proof_with_dep.id);
        assert!(base_proof.verify_integrity());
        assert!(proof_with_dep.verify_integrity());
    }

    #[test]
    fn test_dependency_order_affects_hash() {
        let dep1 = ContentHash::hash_str("dep1");
        let dep2 = ContentHash::hash_str("dep2");

        let proof_a = UniversalProof::builder()
            .vc("(assert true)")
            .dependencies(vec![dep1, dep2])
            .build();

        let proof_b = UniversalProof::builder()
            .vc("(assert true)")
            .dependencies(vec![dep2, dep1])
            .build();

        // Different dependency order should produce different hash
        assert_ne!(proof_a.id, proof_b.id);
    }

    #[test]
    fn test_composed_proof_with_chc_steps() {
        // Create a real CHC proof that uses dependencies
        let lemma_proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .property_name("bound_lemma")
            .vc("(assert (forall ((n Int)) (=> (> n 0) (>= n 1))))")
            .step(ProofStep::Chc(ChcStep::invariant(
                "bound",
                vec!["n".to_string()],
                "(=> (> n 0) (>= n 1))",
            )))
            .step(ProofStep::Chc(ChcStep::property("(> n 0)", "(>= n 1)")))
            .build();

        // Main proof that uses the lemma
        let main_proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .property_name("loop_safety")
            .vc("(assert (forall ((i Int) (n Int)) (=> (and (= i 0) (> n 0)) (< i n))))")
            .dependency(lemma_proof.id)
            .step(ProofStep::Chc(ChcStep::invariant(
                "loop_inv",
                vec!["i".to_string(), "n".to_string()],
                "(and (<= 0 i) (<= i n))",
            )))
            .step(ProofStep::Chc(ChcStep::initiation(
                "(and (= i 0) (> n 0))",
                "(and (<= 0 0) (<= 0 n))",
            )))
            .step(ProofStep::Chc(ChcStep::consecution(
                "(and (<= 0 i) (<= i n) (< i n))",
                "(= i' (+ i 1))",
                "(and (<= 0 i') (<= i' n))",
            )))
            .build();

        assert_eq!(main_proof.dependencies.len(), 1);
        assert_eq!(main_proof.dependencies[0], lemma_proof.id);
        assert!(main_proof.verify_integrity());
        assert!(!main_proof.has_trust_steps()); // All real CHC steps
    }

    #[test]
    fn test_integrity_check_detects_step_modification() {
        let mut proof = UniversalProof::builder()
            .vc("(assert true)")
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();

        assert!(proof.verify_integrity());

        // Tamper with steps
        proof
            .steps
            .push(ProofStep::chc_invariant("fake", vec![], "false"));
        assert!(!proof.verify_integrity());
    }

    #[test]
    fn test_integrity_check_detects_backend_modification() {
        let mut proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .vc("(assert true)")
            .build();

        assert!(proof.verify_integrity());

        // Tamper with backend
        proof.backend = BackendId::Lean5;
        assert!(!proof.verify_integrity());
    }

    #[test]
    fn test_integrity_check_detects_format_modification() {
        let mut proof = UniversalProof::builder()
            .format(ProofFormat::Chc)
            .vc("(assert true)")
            .build();

        assert!(proof.verify_integrity());

        // Tamper with format
        proof.format = ProofFormat::Drat;
        assert!(!proof.verify_integrity());
    }

    #[test]
    fn test_proof_serialization_preserves_dependencies() {
        let dep1 = ContentHash::hash_str("base_proof_1");
        let dep2 = ContentHash::hash_str("base_proof_2");

        let proof = UniversalProof::builder()
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("(assert P)")
            .dependencies(vec![dep1, dep2])
            .step(ProofStep::trust("combined", "uses both dependencies"))
            .build();

        let json = serde_json::to_string(&proof).unwrap();
        let parsed: UniversalProof = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.dependencies.len(), 2);
        assert_eq!(parsed.dependencies[0], dep1);
        assert_eq!(parsed.dependencies[1], dep2);
        assert!(parsed.verify_integrity());
    }

    #[test]
    fn test_empty_proof_deterministic() {
        // Two empty proofs with same config should have same hash
        let proof1 = UniversalProof::builder()
            .backend(BackendId::KaniFast)
            .format(ProofFormat::Mixed)
            .vc("")
            .backend_version("")
            .build();

        let proof2 = UniversalProof::builder()
            .backend(BackendId::KaniFast)
            .format(ProofFormat::Mixed)
            .vc("")
            .backend_version("")
            .build();

        assert_eq!(proof1.id, proof2.id);
    }

    #[test]
    fn test_proof_with_many_dependencies() {
        // Test proof with many dependencies (stress test)
        let deps: Vec<ContentHash> = (0..100)
            .map(|i| ContentHash::hash_str(&format!("dep_{}", i)))
            .collect();

        let proof = UniversalProof::builder()
            .vc("(assert main)")
            .dependencies(deps.clone())
            .build();

        assert_eq!(proof.dependencies.len(), 100);
        assert!(proof.verify_integrity());

        // Serialization should work
        let json = serde_json::to_string(&proof).unwrap();
        let parsed: UniversalProof = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.dependencies.len(), 100);
    }

    #[test]
    fn test_proof_with_deep_dependency_chain_hashes() {
        // Create a chain of 10 proofs
        let mut prev_id = None;
        let mut proofs = Vec::new();

        for i in 0..10 {
            let mut builder = UniversalProof::builder()
                .property_name(format!("proof_{}", i))
                .vc(format!("(assert P{})", i))
                .step(ProofStep::chc_invariant(
                    format!("inv_{}", i),
                    vec![],
                    format!("P{}", i),
                ));

            if let Some(dep) = prev_id {
                builder = builder.dependency(dep);
            }

            let proof = builder.build();
            prev_id = Some(proof.id);
            proofs.push(proof);
        }

        // Verify chain
        for i in 1..10 {
            assert_eq!(proofs[i].dependencies.len(), 1);
            assert_eq!(proofs[i].dependencies[0], proofs[i - 1].id);
        }

        // All proofs should have unique hashes
        let unique_hashes: std::collections::HashSet<_> = proofs.iter().map(|p| p.id).collect();
        assert_eq!(unique_hashes.len(), 10);
    }

    #[test]
    fn test_proof_trust_step_with_dependency() {
        // A trust step that explicitly references a dependency
        let helper_proof = UniversalProof::builder()
            .property_name("helper")
            .vc("(assert (forall ((x Int)) (=> (> x 0) (not (= x 0)))))")
            .step(ProofStep::chc_invariant(
                "pos_nonzero",
                vec!["x".to_string()],
                "(=> (> x 0) (not (= x 0)))",
            ))
            .build();

        let main_proof = UniversalProof::builder()
            .property_name("main")
            .vc("(assert safe_div)")
            .dependency(helper_proof.id)
            .step(ProofStep::trust(
                format!("helper proof {}", helper_proof.id.short_hex()),
                "positive divisor is non-zero",
            ))
            .build();

        assert!(main_proof.has_trust_steps());
        assert!(main_proof.has_dependencies());
        assert!(!main_proof.metadata.is_complete); // Has trust step
        assert!(main_proof.verify_integrity());
    }

    // ==================== Mutation coverage tests ====================

    #[test]
    fn test_builder_returns_functional_builder() {
        // Test that builder() returns a usable builder (catches Default::default() mutant)
        // The key is verifying that the builder can be chained and produces valid proofs
        let builder = UniversalProof::builder();

        // These operations should work on a proper builder
        let proof = builder
            .backend(BackendId::Z3)
            .format(ProofFormat::Chc)
            .vc("test")
            .step(ProofStep::chc_invariant("inv", vec![], "true"))
            .build();

        // Verify the builder actually set the values
        assert_eq!(proof.backend, BackendId::Z3);
        assert_eq!(proof.format, ProofFormat::Chc);
        assert_eq!(proof.vc, "test");
        assert_eq!(proof.steps.len(), 1);
    }

    #[test]
    fn test_builder_new_equals_builder() {
        // Verify that builder() and UniversalProofBuilder::new() behave identically
        let proof1 = UniversalProof::builder()
            .backend(BackendId::Z3)
            .vc("same")
            .backend_version("1.0")
            .build();

        let proof2 = UniversalProofBuilder::new()
            .backend(BackendId::Z3)
            .vc("same")
            .backend_version("1.0")
            .build();

        // Same inputs should produce same hash
        assert_eq!(proof1.id, proof2.id);
    }

    #[test]
    fn test_builder_default_impl() {
        // Test UniversalProofBuilder::default() directly
        let builder: UniversalProofBuilder = Default::default();
        let proof = builder.build();

        // Should have default values
        assert_eq!(proof.backend, BackendId::KaniFast);
        assert_eq!(proof.format, ProofFormat::Mixed);
    }

    // NOTE: Mutant "replace UniversalProof::builder -> UniversalProofBuilder with
    // Default::default()" at line 136 is an EQUIVALENT MUTANT.
    // The `builder()` method calls `UniversalProofBuilder::new()` which itself
    // calls `Self::default()`. Replacing with `Default::default()` produces
    // identical behavior. The chain is:
    //   builder() -> UniversalProofBuilder::new() -> Self::default()
    // So replacing with Default::default() is functionally equivalent.
}
