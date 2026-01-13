//! Proof sketch representation for barrier analysis.
//!
//! A proof sketch captures the essential characteristics of a proof attempt
//! that are relevant for detecting barriers.

use std::collections::HashSet;

/// A sketch of a proof attempt for barrier analysis.
///
/// This captures the key characteristics of a proof strategy that determine
/// which barriers it might hit.
#[derive(Debug, Clone)]
pub struct ProofSketch {
    /// The lower complexity class being separated.
    pub lower_class: ComplexityClass,
    /// The upper complexity class being separated.
    pub upper_class: ComplexityClass,
    /// Techniques used in the proof.
    pub techniques: HashSet<ProofTechnique>,
    /// Properties of the hard function being used.
    pub function_properties: HashSet<FunctionProperty>,
    /// Whether the proof uses diagonalization.
    pub uses_diagonalization: bool,
    /// Whether the proof aims to show circuit lower bounds.
    pub proves_circuit_bound: bool,
    /// Description of the proof strategy.
    pub description: String,
}

impl ProofSketch {
    /// Create a new proof sketch attempting to separate two complexity classes.
    pub fn new(lower: ComplexityClass, upper: ComplexityClass) -> Self {
        Self {
            lower_class: lower,
            upper_class: upper,
            techniques: HashSet::new(),
            function_properties: HashSet::new(),
            uses_diagonalization: false,
            proves_circuit_bound: false,
            description: String::new(),
        }
    }

    /// Add a proof technique.
    pub fn with_technique(mut self, technique: ProofTechnique) -> Self {
        // Automatically detect diagonalization
        if matches!(technique, ProofTechnique::Diagonalization) {
            self.uses_diagonalization = true;
        }
        self.techniques.insert(technique);
        self
    }

    /// Add a function property.
    pub fn with_property(mut self, property: FunctionProperty) -> Self {
        self.function_properties.insert(property);
        self
    }

    /// Mark that this proof shows a circuit lower bound.
    pub fn with_circuit_bound(mut self) -> Self {
        self.proves_circuit_bound = true;
        self
    }

    /// Set a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Check if the proof uses any relativizing techniques.
    pub fn uses_relativizing_techniques(&self) -> bool {
        self.techniques.iter().any(|t| t.relativizes())
    }

    /// Check if the proof uses techniques known to not relativize.
    pub fn uses_non_relativizing_techniques(&self) -> bool {
        self.techniques.iter().any(|t| !t.relativizes())
    }

    /// Check if the proof uses natural properties.
    pub fn uses_natural_properties(&self) -> bool {
        self.function_properties
            .iter()
            .any(|p| p.is_large() && p.is_constructive())
    }

    /// Get the list of relativizing techniques used.
    pub fn relativizing_techniques(&self) -> Vec<&ProofTechnique> {
        self.techniques.iter().filter(|t| t.relativizes()).collect()
    }

    /// Get the list of natural properties used.
    pub fn natural_properties(&self) -> Vec<&FunctionProperty> {
        self.function_properties
            .iter()
            .filter(|p| p.is_large() && p.is_constructive())
            .collect()
    }
}

/// Complexity classes for separation results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplexityClass {
    /// Polynomial time.
    P,
    /// Nondeterministic polynomial time.
    NP,
    /// co-NP.
    CoNP,
    /// Polynomial space.
    PSPACE,
    /// Exponential time.
    EXP,
    /// Nondeterministic exponential time.
    NEXP,
    /// Bounded-error probabilistic polynomial time.
    BPP,
    /// Polynomial hierarchy.
    PH,
    /// NC (efficient parallel computation).
    NC,
    /// L (logarithmic space).
    L,
    /// NL (nondeterministic logarithmic space).
    NL,
    /// AC0 (constant-depth circuits).
    AC0,
    /// TC0 (constant-depth circuits with threshold gates).
    TC0,
    /// P/poly (polynomial-size circuits).
    PPoly,
}

impl ComplexityClass {
    /// Check if this class is contained in another.
    pub fn contained_in(&self, other: &ComplexityClass) -> bool {
        use ComplexityClass::*;
        match (self, other) {
            // Same class
            (a, b) if a == b => true,
            // Everything is in PSPACE and EXP
            (_, PSPACE | EXP) => true,
            // P ⊆ NP, coNP, BPP, PH, P/poly
            (P, NP | CoNP | BPP | PH | PPoly) => true,
            // NP, coNP ⊆ PH
            (NP | CoNP, PH) => true,
            // L ⊆ NL ⊆ P
            (L, NL | P | NP | CoNP | BPP | PH | PPoly) => true,
            (NL, P | NP | CoNP | BPP | PH | PPoly) => true,
            // AC0 ⊆ TC0 ⊆ NC ⊆ P
            (AC0, TC0 | NC | P | NP | CoNP | BPP | PH | PPoly) => true,
            (TC0, NC | P | NP | CoNP | BPP | PH | PPoly) => true,
            (NC, P | NP | CoNP | BPP | PH | PPoly) => true,
            _ => false,
        }
    }

    /// Get the oracle version of this class (for relativization).
    pub fn oracle_version(&self) -> String {
        format!("{}^A", self.name())
    }

    /// Get the name of this class.
    pub fn name(&self) -> &'static str {
        match self {
            ComplexityClass::P => "P",
            ComplexityClass::NP => "NP",
            ComplexityClass::CoNP => "coNP",
            ComplexityClass::PSPACE => "PSPACE",
            ComplexityClass::EXP => "EXP",
            ComplexityClass::NEXP => "NEXP",
            ComplexityClass::BPP => "BPP",
            ComplexityClass::PH => "PH",
            ComplexityClass::NC => "NC",
            ComplexityClass::L => "L",
            ComplexityClass::NL => "NL",
            ComplexityClass::AC0 => "AC0",
            ComplexityClass::TC0 => "TC0",
            ComplexityClass::PPoly => "P/poly",
        }
    }
}

/// Proof techniques used in complexity separations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofTechnique {
    /// Diagonalization (a la Cantor/Turing/Time hierarchy).
    Diagonalization,
    /// Simulation argument (showing one class can simulate another).
    Simulation,
    /// Padding argument (reducing from one problem to another by padding).
    Padding,
    /// Counting argument (pigeonhole, probabilistic method).
    Counting,
    /// Random restriction (Furst-Saxe-Sipser, Hastad).
    RandomRestriction,
    /// Switching lemma.
    SwitchingLemma,
    /// Polynomial method (e.g., for communication complexity).
    PolynomialMethod,
    /// Approximation method (Razborov).
    ApproximationMethod,
    /// Adversary argument (information-theoretic).
    AdversaryArgument,
    /// Interactive proof techniques.
    InteractiveProof,
    /// Arithmetization (used in IP=PSPACE).
    Arithmetization,
    /// Communication complexity reduction.
    CommunicationComplexity,
    /// Game-theoretic argument (prover-delayer games).
    GameTheoretic,
    /// Algebraic techniques (GCT, etc.).
    Algebraic,
}

impl ProofTechnique {
    /// Check if this technique relativizes.
    ///
    /// A technique relativizes if adding an oracle to all machines
    /// doesn't affect whether the proof works.
    pub fn relativizes(&self) -> bool {
        match self {
            // These techniques all relativize
            ProofTechnique::Diagonalization => true,
            ProofTechnique::Simulation => true,
            ProofTechnique::Padding => true,
            ProofTechnique::Counting => true,
            ProofTechnique::AdversaryArgument => true,
            ProofTechnique::GameTheoretic => true,
            // These techniques may not relativize
            ProofTechnique::RandomRestriction => true, // Still relativizes (circuit lower bounds)
            ProofTechnique::SwitchingLemma => true,
            ProofTechnique::PolynomialMethod => false, // Can be non-relativizing
            ProofTechnique::ApproximationMethod => true,
            ProofTechnique::InteractiveProof => false, // IP=PSPACE doesn't relativize
            ProofTechnique::Arithmetization => false,  // Core of IP=PSPACE
            ProofTechnique::CommunicationComplexity => true,
            ProofTechnique::Algebraic => false, // May avoid barriers
        }
    }

    /// Check if this technique is known to be non-naturalizable.
    pub fn avoids_natural_proofs(&self) -> bool {
        match self {
            // Algebraic techniques like GCT may avoid natural proofs
            ProofTechnique::Algebraic => true,
            // Interactive proofs use different structure
            ProofTechnique::InteractiveProof => true,
            // Most circuit lower bound techniques are natural
            _ => false,
        }
    }

    /// Get a description of this technique.
    pub fn description(&self) -> &'static str {
        match self {
            ProofTechnique::Diagonalization => {
                "Construct a function that differs from every function in the smaller class"
            }
            ProofTechnique::Simulation => {
                "Show one machine can simulate another with bounded overhead"
            }
            ProofTechnique::Padding => "Add padding to reduce between problems",
            ProofTechnique::Counting => "Use counting/pigeonhole arguments",
            ProofTechnique::RandomRestriction => "Randomly set variables to simplify circuits",
            ProofTechnique::SwitchingLemma => {
                "Show DNF/CNF depth reduces under random restrictions"
            }
            ProofTechnique::PolynomialMethod => "Use polynomial representations of functions",
            ProofTechnique::ApproximationMethod => "Approximate circuits by polynomials",
            ProofTechnique::AdversaryArgument => "Information-theoretic lower bounds",
            ProofTechnique::InteractiveProof => "Use interactive verification protocols",
            ProofTechnique::Arithmetization => "Encode computation as polynomial evaluation",
            ProofTechnique::CommunicationComplexity => "Lower bounds via communication protocols",
            ProofTechnique::GameTheoretic => "Prover-delayer games for proof complexity",
            ProofTechnique::Algebraic => "Algebraic geometry / representation theory",
        }
    }
}

/// Properties of functions used in lower bound proofs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionProperty {
    /// Function has high circuit complexity.
    HighCircuitComplexity,
    /// Function has low correlation with small circuits.
    LowCorrelation,
    /// Function is hard for random restrictions.
    HardForRestrictions,
    /// Function has high sensitivity.
    HighSensitivity,
    /// Function has high block sensitivity.
    HighBlockSensitivity,
    /// Function has high degree (as polynomial).
    HighDegree,
    /// Function is a specific function (not generic).
    Specific(String),
    /// Custom property with description.
    Custom {
        name: String,
        large: bool,
        constructive: bool,
    },
}

impl FunctionProperty {
    /// Check if this property is "large" (holds for many functions).
    ///
    /// A property is large if at least 1/poly(n) of all n-input Boolean
    /// functions satisfy it.
    pub fn is_large(&self) -> bool {
        match self {
            // Most generic hardness properties are large
            FunctionProperty::HighCircuitComplexity => true,
            FunctionProperty::LowCorrelation => true,
            FunctionProperty::HardForRestrictions => true,
            FunctionProperty::HighSensitivity => true,
            FunctionProperty::HighBlockSensitivity => true,
            FunctionProperty::HighDegree => true,
            // Specific functions are not large (only one function)
            FunctionProperty::Specific(_) => false,
            FunctionProperty::Custom { large, .. } => *large,
        }
    }

    /// Check if this property is constructive (efficiently testable).
    ///
    /// A property is constructive if it can be tested in poly(2^n) time.
    pub fn is_constructive(&self) -> bool {
        match self {
            // Circuit complexity is constructive (enumerate circuits)
            FunctionProperty::HighCircuitComplexity => true,
            // Correlation can be computed
            FunctionProperty::LowCorrelation => true,
            // Restriction hardness is constructive
            FunctionProperty::HardForRestrictions => true,
            // Sensitivity is polynomial time
            FunctionProperty::HighSensitivity => true,
            FunctionProperty::HighBlockSensitivity => true,
            // Degree is polynomial time
            FunctionProperty::HighDegree => true,
            // Specific functions: checking is efficient
            FunctionProperty::Specific(_) => true,
            FunctionProperty::Custom { constructive, .. } => *constructive,
        }
    }

    /// Get the name of this property.
    pub fn name(&self) -> String {
        match self {
            FunctionProperty::HighCircuitComplexity => "high circuit complexity".into(),
            FunctionProperty::LowCorrelation => "low correlation with small circuits".into(),
            FunctionProperty::HardForRestrictions => "hardness under restrictions".into(),
            FunctionProperty::HighSensitivity => "high sensitivity".into(),
            FunctionProperty::HighBlockSensitivity => "high block sensitivity".into(),
            FunctionProperty::HighDegree => "high polynomial degree".into(),
            FunctionProperty::Specific(name) => format!("specific function: {}", name),
            FunctionProperty::Custom { name, .. } => name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_sketch_builder() {
        let proof = ProofSketch::new(ComplexityClass::P, ComplexityClass::NP)
            .with_technique(ProofTechnique::Diagonalization)
            .with_technique(ProofTechnique::Simulation)
            .with_property(FunctionProperty::HighCircuitComplexity)
            .with_description("Attempt to separate P from NP");

        assert!(proof.uses_diagonalization);
        assert!(proof.techniques.contains(&ProofTechnique::Diagonalization));
        assert!(proof.techniques.contains(&ProofTechnique::Simulation));
        assert!(proof
            .function_properties
            .contains(&FunctionProperty::HighCircuitComplexity));
    }

    #[test]
    fn test_relativizing_techniques() {
        let proof = ProofSketch::new(ComplexityClass::P, ComplexityClass::NP)
            .with_technique(ProofTechnique::Diagonalization)
            .with_technique(ProofTechnique::Simulation);

        assert!(proof.uses_relativizing_techniques());
        assert!(!proof.uses_non_relativizing_techniques());

        let proof2 = ProofSketch::new(ComplexityClass::P, ComplexityClass::PSPACE)
            .with_technique(ProofTechnique::InteractiveProof)
            .with_technique(ProofTechnique::Arithmetization);

        assert!(proof2.uses_non_relativizing_techniques());
    }

    #[test]
    fn test_natural_properties() {
        let proof = ProofSketch::new(ComplexityClass::AC0, ComplexityClass::P)
            .with_property(FunctionProperty::HighCircuitComplexity)
            .with_property(FunctionProperty::LowCorrelation);

        assert!(proof.uses_natural_properties());

        let proof2 = ProofSketch::new(ComplexityClass::P, ComplexityClass::NP)
            .with_property(FunctionProperty::Specific("PARITY".into()));

        assert!(!proof2.uses_natural_properties());
    }

    #[test]
    fn test_complexity_class_containment() {
        assert!(ComplexityClass::P.contained_in(&ComplexityClass::NP));
        assert!(ComplexityClass::P.contained_in(&ComplexityClass::PSPACE));
        assert!(ComplexityClass::NP.contained_in(&ComplexityClass::PH));
        assert!(ComplexityClass::AC0.contained_in(&ComplexityClass::TC0));
        assert!(ComplexityClass::TC0.contained_in(&ComplexityClass::NC));
        assert!(!ComplexityClass::NP.contained_in(&ComplexityClass::P));
    }

    #[test]
    fn test_technique_relativizes() {
        assert!(ProofTechnique::Diagonalization.relativizes());
        assert!(ProofTechnique::Simulation.relativizes());
        assert!(!ProofTechnique::InteractiveProof.relativizes());
        assert!(!ProofTechnique::Arithmetization.relativizes());
    }
}
