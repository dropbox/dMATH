//! Barrier types representing the three known obstacles to P vs NP separation.

use crate::oracle::{AlgebraicOracle, Oracle};

/// The three main barriers to complexity class separations.
///
/// These barriers explain why certain proof techniques fundamentally cannot
/// separate P from NP (or other pairs of complexity classes).
#[derive(Debug, Clone)]
pub enum Barrier {
    /// The proof technique relativizes.
    ///
    /// A proof technique "relativizes" if it would still work when all machines
    /// have access to an oracle. Baker-Gill-Solovay (1975) showed:
    /// - There exists oracle A such that P^A = NP^A
    /// - There exists oracle B such that P^B ≠ NP^B
    ///
    /// Therefore, any proof that relativizes cannot separate P from NP.
    Relativization(RelativizationBarrier),

    /// The proof uses "natural" properties in the sense of Razborov-Rudich.
    ///
    /// A lower bound proof is "natural" if it uses a property that is:
    /// 1. **Large**: Holds for at least 1/poly(n) fraction of n-input Boolean functions
    /// 2. **Constructive**: Can be tested in poly(2^n) time
    ///
    /// Razborov-Rudich (1997) showed: If secure pseudorandom generators exist,
    /// natural proofs cannot prove super-polynomial lower bounds against P/poly.
    NaturalProof(NaturalProofBarrier),

    /// The proof algebrizes.
    ///
    /// Aaronson-Wigderson (2009) extended relativization to algebraic settings.
    /// A proof "algebrizes" if it still works when the oracle is replaced by
    /// a low-degree polynomial extension over a finite field.
    ///
    /// Known algebrizing results: IP = PSPACE, MIP = NEXP, etc.
    /// No known technique avoids both relativization and algebrization.
    Algebrization(AlgebrizationBarrier),
}

impl Barrier {
    /// Check if this is a relativization barrier.
    pub fn is_relativization(&self) -> bool {
        matches!(self, Barrier::Relativization(_))
    }

    /// Check if this is a natural proof barrier.
    pub fn is_natural_proof(&self) -> bool {
        matches!(self, Barrier::NaturalProof(_))
    }

    /// Check if this is an algebrization barrier.
    pub fn is_algebrization(&self) -> bool {
        matches!(self, Barrier::Algebrization(_))
    }

    /// Get a human-readable description of the barrier.
    pub fn description(&self) -> &'static str {
        match self {
            Barrier::Relativization(_) => {
                "Proof relativizes: works with oracles that make P=NP and P≠NP"
            }
            Barrier::NaturalProof(_) => "Proof is natural: uses large, constructive properties",
            Barrier::Algebrization(_) => "Proof algebrizes: works with algebraic oracle extensions",
        }
    }

    /// Get a reference to recommended techniques that avoid this barrier.
    pub fn workarounds(&self) -> &'static [&'static str] {
        match self {
            Barrier::Relativization(_) => &[
                "Interactive proofs (IP=PSPACE is non-relativizing)",
                "Arithmetization",
                "Algebraic techniques",
                "Circuit lower bounds (may still hit natural proofs)",
            ],
            Barrier::NaturalProof(_) => &[
                "Use non-constructive properties",
                "Use properties not large over random functions",
                "Focus on specific hard functions (not generic separations)",
                "Ryan Williams's approach (algorithmic -> lower bounds)",
            ],
            Barrier::Algebrization(_) => &[
                "No known general technique avoids algebrization",
                "May need fundamentally new ideas",
                "Geometric Complexity Theory (GCT) is conjectured non-algebrizing",
            ],
        }
    }
}

/// Detailed information about a relativization barrier.
#[derive(Debug, Clone)]
pub struct RelativizationBarrier {
    /// Oracle that makes the separation true (e.g., PSPACE-complete oracle for P≠NP).
    pub separating_oracle: Oracle,
    /// Oracle that makes the classes collapse (e.g., oracle where P=NP).
    pub collapsing_oracle: Oracle,
    /// Which techniques in the proof are oracle-independent.
    pub relativizing_techniques: Vec<String>,
}

impl RelativizationBarrier {
    /// Create a new relativization barrier.
    pub fn new(
        separating_oracle: Oracle,
        collapsing_oracle: Oracle,
        relativizing_techniques: Vec<String>,
    ) -> Self {
        Self {
            separating_oracle,
            collapsing_oracle,
            relativizing_techniques,
        }
    }
}

/// Detailed information about a natural proof barrier.
#[derive(Debug, Clone)]
pub struct NaturalProofBarrier {
    /// Whether the proof uses a "large" property.
    pub uses_largeness: bool,
    /// Whether the property is efficiently constructive.
    pub uses_constructivity: bool,
    /// Description of the natural property being used.
    pub property_description: String,
    /// The fraction of functions satisfying the property (if known).
    pub largeness_fraction: Option<f64>,
}

impl NaturalProofBarrier {
    /// Create a new natural proof barrier.
    pub fn new(
        uses_largeness: bool,
        uses_constructivity: bool,
        property_description: String,
    ) -> Self {
        Self {
            uses_largeness,
            uses_constructivity,
            property_description,
            largeness_fraction: None,
        }
    }

    /// Set the largeness fraction.
    pub fn with_largeness_fraction(mut self, fraction: f64) -> Self {
        self.largeness_fraction = Some(fraction);
        self
    }

    /// Check if this is a "fully natural" proof (both large and constructive).
    pub fn is_fully_natural(&self) -> bool {
        self.uses_largeness && self.uses_constructivity
    }
}

/// Detailed information about an algebrization barrier.
#[derive(Debug, Clone)]
pub struct AlgebrizationBarrier {
    /// The algebraic oracle that breaks the separation.
    pub algebraic_oracle: AlgebraicOracle,
    /// Which techniques in the proof algebrize.
    pub algebrizing_techniques: Vec<String>,
}

impl AlgebrizationBarrier {
    /// Create a new algebrization barrier.
    pub fn new(algebraic_oracle: AlgebraicOracle, algebrizing_techniques: Vec<String>) -> Self {
        Self {
            algebraic_oracle,
            algebrizing_techniques,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OracleType;

    #[test]
    fn test_barrier_type_checks() {
        let rel = Barrier::Relativization(RelativizationBarrier::new(
            Oracle::new(OracleType::PSPACE),
            Oracle::new(OracleType::TallyNP),
            vec!["diagonalization".into()],
        ));
        assert!(rel.is_relativization());
        assert!(!rel.is_natural_proof());
        assert!(!rel.is_algebrization());

        let nat = Barrier::NaturalProof(NaturalProofBarrier::new(
            true,
            true,
            "random restriction".into(),
        ));
        assert!(!nat.is_relativization());
        assert!(nat.is_natural_proof());
        assert!(!nat.is_algebrization());
    }

    #[test]
    fn test_natural_proof_fully_natural() {
        let np1 = NaturalProofBarrier::new(true, true, "test".into());
        assert!(np1.is_fully_natural());

        let np2 = NaturalProofBarrier::new(true, false, "test".into());
        assert!(!np2.is_fully_natural());

        let np3 = NaturalProofBarrier::new(false, true, "test".into());
        assert!(!np3.is_fully_natural());
    }
}
