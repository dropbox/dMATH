//! Lean5 Mode System
//!
//! The mode system enables Lean5 to support multiple mathematical traditions
//! with proven-safe combinations. Each mode activates different axioms and
//! type-theoretic features.
//!
//! # Mode Compatibility
//!
//! Proofs from one mode can be imported into another if the source mode's
//! axioms are provable in (or axiomatized by) the target mode:
//!
//! - Constructive → All modes (most restrictive, works everywhere)
//! - Impredicative → Classical, SetTheoretic (proof irrelevance compatible)
//! - Cubical → Only itself (different equality/computation rules; no translation into non-cubical modes)
//! - Classical → SetTheoretic (SetTheoretic extends Classical)
//! - SetTheoretic → Only itself (strongest axioms)

use serde::{Deserialize, Serialize};

/// Logical mode controlling which axioms and features are available.
///
/// Different mathematical traditions have different logical foundations.
/// Rather than pick one, Lean5 supports multiple modes with proven-safe
/// combinations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Lean5Mode {
    /// Pure Martin-Löf Type Theory - no axioms, decidable type checking.
    /// Compatible with: All modes (most restrictive)
    ///
    /// This is the default mode and corresponds to Lean 4's core type theory.
    #[default]
    Constructive,

    /// Calculus of Inductive Constructions - impredicative Prop, restricted large elimination.
    /// Compatible with: Constructive, Classical, SetTheoretic
    ///
    /// This mode adds:
    /// - Impredicative Prop (quantification over Prop stays in Prop)
    /// - SProp (strict propositions, always proof-irrelevant)
    /// - Restricted large elimination from Prop
    Impredicative,

    /// Cubical Type Theory - Path types, hcomp, transp, Glue, univalence provable.
    /// Compatible with: Constructive only (NOT with Classical or Impredicative)
    ///
    /// This mode adds:
    /// - Interval type I with endpoints 0 and 1
    /// - Path types as primitive equality
    /// - Homogeneous composition (hcomp)
    /// - Transport along paths (transp)
    /// - Glue types for univalence
    ///
    /// WARNING: Cubical mode is ISOLATED - cannot import from or to Classical/Impredicative
    /// because it uses different equality/computation rules (Path/Glue/hcomp/transp) that are
    /// not available in the other modes. Note: univalence itself is compatible with classical
    /// axioms like LEM; the isolation here is a kernel/translation boundary.
    Cubical,

    /// Classical logic - LEM, Choice as axioms.
    /// Compatible with: Constructive, Impredicative
    ///
    /// This mode adds:
    /// - Law of Excluded Middle (LEM): ∀ P, P ∨ ¬P
    /// - Axiom of Choice
    /// - Function extensionality
    /// - Propositional extensionality
    Classical,

    /// ZFC set theory - sets as first-class, no dependent types required.
    /// Compatible with: Classical (inherits all classical axioms)
    ///
    /// This mode adds:
    /// - ZFC axioms (Extensionality, Pairing, Union, PowerSet, etc.)
    /// - Set membership as primitive
    /// - Set comprehension
    SetTheoretic,
}

impl Lean5Mode {
    /// Check if proofs from `source` mode can be used in `target` mode.
    ///
    /// The import relation is transitive but not symmetric:
    /// - Constructive → Any (most general proofs)
    /// - Cubical is isolated (different kernel rules; no translation provided)
    /// - Classical hierarchy: Impredicative → Classical → SetTheoretic
    ///
    /// # Examples
    ///
    /// ```
    /// use lean5_kernel::mode::Lean5Mode;
    ///
    /// // Constructive proofs work everywhere
    /// assert!(Lean5Mode::can_import(Lean5Mode::Constructive, Lean5Mode::Classical));
    /// assert!(Lean5Mode::can_import(Lean5Mode::Constructive, Lean5Mode::Cubical));
    ///
    /// // Classical doesn't work in Constructive
    /// assert!(!Lean5Mode::can_import(Lean5Mode::Classical, Lean5Mode::Constructive));
    ///
    /// // Cubical is isolated
    /// assert!(!Lean5Mode::can_import(Lean5Mode::Cubical, Lean5Mode::Classical));
    /// assert!(!Lean5Mode::can_import(Lean5Mode::Classical, Lean5Mode::Cubical));
    /// ```
    #[must_use]
    pub fn can_import(source: Lean5Mode, target: Lean5Mode) -> bool {
        use Lean5Mode::*;
        match (source, target) {
            // Constructive proofs work everywhere
            (Constructive, _) => true,

            // Same mode always works
            (m1, m2) if m1 == m2 => true,

            // Impredicative works in Classical (both accept proof irrelevance)
            (Impredicative, Classical) => true,

            // Classical works in SetTheoretic (SetTheoretic extends Classical with ZFC axioms)
            (Classical, SetTheoretic) => true,
            (Impredicative, SetTheoretic) => true,

            // Cubical is isolated: different equality/computation rules (needs translation to cross)
            (Cubical, _) | (_, Cubical) => false,

            // SetTheoretic only imports from Classical hierarchy
            (SetTheoretic, _) => false,

            // Default: not compatible
            _ => false,
        }
    }

    /// Get the default mode for a source system.
    ///
    /// When importing proofs from an external system, this determines
    /// which Lean5 mode they will be checked in.
    #[must_use]
    pub fn from_source_system(system: SourceSystem) -> Self {
        use SourceSystem::*;
        match system {
            Lean4 => Lean5Mode::Constructive,
            Coq => Lean5Mode::Impredicative,
            Agda => Lean5Mode::Constructive,
            CubicalAgda => Lean5Mode::Cubical,
            IsabelleHOL | HOLLight | HOL4 => Lean5Mode::Classical,
            Mizar | MetamathZFC => Lean5Mode::SetTheoretic,
            MetamathSet | ACL2 => Lean5Mode::Classical,
            PVS => Lean5Mode::Classical,
        }
    }

    /// Get the axioms available in this mode.
    #[must_use]
    pub fn available_axioms(&self) -> Vec<AxiomId> {
        use Lean5Mode::*;
        match self {
            Constructive => vec![
                // No logical axioms - pure MLTT
            ],

            Impredicative => vec![
                AxiomId::PropExt,    // Propositional extensionality
                AxiomId::ProofIrrel, // Proof irrelevance for Prop
            ],

            Cubical => vec![
                // Univalence is PROVABLE, not an axiom
                // But we expose it as a theorem
            ],

            Classical => vec![
                AxiomId::PropExt,
                AxiomId::ProofIrrel,
                AxiomId::LEM,    // Law of excluded middle
                AxiomId::Choice, // Axiom of choice
                AxiomId::FunExt, // Function extensionality
            ],

            SetTheoretic => vec![
                // All classical axioms plus ZFC
                AxiomId::PropExt,
                AxiomId::ProofIrrel,
                AxiomId::LEM,
                AxiomId::Choice,
                AxiomId::FunExt,
                AxiomId::ZFCExtensionality,
                AxiomId::ZFCPairing,
                AxiomId::ZFCUnion,
                AxiomId::ZFCPowerSet,
                AxiomId::ZFCInfinity,
                AxiomId::ZFCSeparation,
                AxiomId::ZFCReplacement,
                AxiomId::ZFCFoundation,
            ],
        }
    }

    /// Check if this mode allows large elimination from the given sort.
    ///
    /// Large elimination means eliminating from Prop to produce data (Type).
    /// This is restricted in Impredicative/Classical modes to prevent
    /// inconsistency.
    ///
    /// # Rules
    ///
    /// - Constructive: Always allowed
    /// - Impredicative/Classical: Only for singletons (Empty, Unit, Eq)
    /// - Cubical: Always allowed
    /// - SetTheoretic: Sets can eliminate freely
    #[must_use]
    pub fn allows_large_elimination(&self, from_prop: bool) -> bool {
        match self {
            Lean5Mode::Constructive => true,
            Lean5Mode::Impredicative | Lean5Mode::Classical => {
                // Only small elimination from Prop
                // Large elim only for singletons (handled separately)
                !from_prop
            }
            Lean5Mode::Cubical => true,
            Lean5Mode::SetTheoretic => true,
        }
    }

    /// Get a human-readable name for this mode.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Lean5Mode::Constructive => "Constructive",
            Lean5Mode::Impredicative => "Impredicative",
            Lean5Mode::Cubical => "Cubical",
            Lean5Mode::Classical => "Classical",
            Lean5Mode::SetTheoretic => "SetTheoretic",
        }
    }
}

/// Source proof system for imported declarations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceSystem {
    /// Lean 4
    Lean4,
    /// Coq proof assistant
    Coq,
    /// Agda (standard)
    Agda,
    /// Cubical Agda
    CubicalAgda,
    /// Isabelle/HOL
    IsabelleHOL,
    /// HOL Light
    HOLLight,
    /// HOL4
    HOL4,
    /// Mizar
    Mizar,
    /// Metamath with ZFC axioms
    MetamathZFC,
    /// Metamath set.mm (classical logic)
    MetamathSet,
    /// PVS
    PVS,
    /// ACL2
    ACL2,
}

impl SourceSystem {
    /// Get a human-readable name for this system.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            SourceSystem::Lean4 => "Lean 4",
            SourceSystem::Coq => "Coq",
            SourceSystem::Agda => "Agda",
            SourceSystem::CubicalAgda => "Cubical Agda",
            SourceSystem::IsabelleHOL => "Isabelle/HOL",
            SourceSystem::HOLLight => "HOL Light",
            SourceSystem::HOL4 => "HOL4",
            SourceSystem::Mizar => "Mizar",
            SourceSystem::MetamathZFC => "Metamath/ZFC",
            SourceSystem::MetamathSet => "Metamath/set.mm",
            SourceSystem::PVS => "PVS",
            SourceSystem::ACL2 => "ACL2",
        }
    }
}

/// Axiom identifiers for logical axioms in each mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AxiomId {
    // ════════════════════════════════════════════════════════════════════
    // Logical axioms
    // ════════════════════════════════════════════════════════════════════
    /// Propositional extensionality: (P ↔ Q) → P = Q
    PropExt,
    /// Proof irrelevance: any two proofs of the same Prop are equal
    ProofIrrel,
    /// Law of Excluded Middle: ∀ P, P ∨ ¬P
    LEM,
    /// Axiom of Choice
    Choice,
    /// Function extensionality: (∀ x, f x = g x) → f = g
    FunExt,

    // ════════════════════════════════════════════════════════════════════
    // ZFC axioms
    // ════════════════════════════════════════════════════════════════════
    /// Extensionality: sets with same elements are equal
    ZFCExtensionality,
    /// Pairing: {a, b} exists
    ZFCPairing,
    /// Union: ⋃A exists
    ZFCUnion,
    /// Power Set: P(A) exists
    ZFCPowerSet,
    /// Infinity: ω exists
    ZFCInfinity,
    /// Separation: {x ∈ A | φ(x)} exists
    ZFCSeparation,
    /// Replacement: {F(x) | x ∈ A} exists
    ZFCReplacement,
    /// Foundation: every non-empty set has a ∈-minimal element
    ZFCFoundation,
    /// Choice (ZFC version): every family of non-empty sets has a choice function
    ZFCChoice,
}

impl AxiomId {
    /// Get a human-readable name for this axiom.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            AxiomId::PropExt => "Propositional Extensionality",
            AxiomId::ProofIrrel => "Proof Irrelevance",
            AxiomId::LEM => "Law of Excluded Middle",
            AxiomId::Choice => "Axiom of Choice",
            AxiomId::FunExt => "Function Extensionality",
            AxiomId::ZFCExtensionality => "ZFC Extensionality",
            AxiomId::ZFCPairing => "ZFC Pairing",
            AxiomId::ZFCUnion => "ZFC Union",
            AxiomId::ZFCPowerSet => "ZFC Power Set",
            AxiomId::ZFCInfinity => "ZFC Infinity",
            AxiomId::ZFCSeparation => "ZFC Separation",
            AxiomId::ZFCReplacement => "ZFC Replacement",
            AxiomId::ZFCFoundation => "ZFC Foundation",
            AxiomId::ZFCChoice => "ZFC Choice",
        }
    }
}

/// Error when mode consistency is violated.
#[derive(Debug, Clone)]
pub enum ModeError {
    /// Attempted to use a feature not available in the current mode.
    FeatureNotAvailable {
        /// The current mode
        current: Lean5Mode,
        /// The feature that was attempted
        feature: String,
    },

    /// Attempted to import from an incompatible mode.
    IncompatibleImport {
        /// The source mode of the import
        source: Lean5Mode,
        /// The target mode attempting to use the import
        target: Lean5Mode,
    },

    /// Attempted to use an axiom not available in the current mode.
    AxiomNotAvailable {
        /// The axiom that was attempted
        axiom: AxiomId,
        /// The current mode
        mode: Lean5Mode,
    },
}

impl std::fmt::Display for ModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModeError::FeatureNotAvailable { current, feature } => {
                write!(
                    f,
                    "Feature not available in {} mode: {}",
                    current.name(),
                    feature
                )
            }
            ModeError::IncompatibleImport { source, target } => {
                write!(
                    f,
                    "Cannot import from {} mode into {} mode",
                    source.name(),
                    target.name()
                )
            }
            ModeError::AxiomNotAvailable { axiom, mode } => {
                write!(
                    f,
                    "Axiom {} not available in {} mode",
                    axiom.name(),
                    mode.name()
                )
            }
        }
    }
}

impl std::error::Error for ModeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructive_imports_everywhere() {
        // Constructive proofs should work in all modes
        assert!(Lean5Mode::can_import(
            Lean5Mode::Constructive,
            Lean5Mode::Constructive
        ));
        assert!(Lean5Mode::can_import(
            Lean5Mode::Constructive,
            Lean5Mode::Impredicative
        ));
        assert!(Lean5Mode::can_import(
            Lean5Mode::Constructive,
            Lean5Mode::Cubical
        ));
        assert!(Lean5Mode::can_import(
            Lean5Mode::Constructive,
            Lean5Mode::Classical
        ));
        assert!(Lean5Mode::can_import(
            Lean5Mode::Constructive,
            Lean5Mode::SetTheoretic
        ));
    }

    #[test]
    fn test_cubical_isolated() {
        // Cubical mode is isolated - can't import from or to other non-constructive modes
        assert!(!Lean5Mode::can_import(
            Lean5Mode::Cubical,
            Lean5Mode::Impredicative
        ));
        assert!(!Lean5Mode::can_import(
            Lean5Mode::Cubical,
            Lean5Mode::Classical
        ));
        assert!(!Lean5Mode::can_import(
            Lean5Mode::Cubical,
            Lean5Mode::SetTheoretic
        ));

        assert!(!Lean5Mode::can_import(
            Lean5Mode::Impredicative,
            Lean5Mode::Cubical
        ));
        assert!(!Lean5Mode::can_import(
            Lean5Mode::Classical,
            Lean5Mode::Cubical
        ));
        assert!(!Lean5Mode::can_import(
            Lean5Mode::SetTheoretic,
            Lean5Mode::Cubical
        ));

        // But Constructive can import to Cubical
        assert!(Lean5Mode::can_import(
            Lean5Mode::Constructive,
            Lean5Mode::Cubical
        ));

        // And Cubical can import to itself
        assert!(Lean5Mode::can_import(
            Lean5Mode::Cubical,
            Lean5Mode::Cubical
        ));
    }

    #[test]
    fn test_classical_hierarchy() {
        // Impredicative → Classical → SetTheoretic
        assert!(Lean5Mode::can_import(
            Lean5Mode::Impredicative,
            Lean5Mode::Classical
        ));
        assert!(Lean5Mode::can_import(
            Lean5Mode::Classical,
            Lean5Mode::SetTheoretic
        ));
        assert!(Lean5Mode::can_import(
            Lean5Mode::Impredicative,
            Lean5Mode::SetTheoretic
        ));

        // But not the other way
        assert!(!Lean5Mode::can_import(
            Lean5Mode::Classical,
            Lean5Mode::Impredicative
        ));
        assert!(!Lean5Mode::can_import(
            Lean5Mode::SetTheoretic,
            Lean5Mode::Classical
        ));
        assert!(!Lean5Mode::can_import(
            Lean5Mode::SetTheoretic,
            Lean5Mode::Impredicative
        ));
    }

    #[test]
    fn test_classical_not_in_constructive() {
        // Classical axioms can't be used in constructive mode
        assert!(!Lean5Mode::can_import(
            Lean5Mode::Classical,
            Lean5Mode::Constructive
        ));
        assert!(!Lean5Mode::can_import(
            Lean5Mode::Impredicative,
            Lean5Mode::Constructive
        ));
    }

    #[test]
    fn test_source_system_modes() {
        // Test that source systems map to expected modes
        assert_eq!(
            Lean5Mode::from_source_system(SourceSystem::Lean4),
            Lean5Mode::Constructive
        );
        assert_eq!(
            Lean5Mode::from_source_system(SourceSystem::Coq),
            Lean5Mode::Impredicative
        );
        assert_eq!(
            Lean5Mode::from_source_system(SourceSystem::CubicalAgda),
            Lean5Mode::Cubical
        );
        assert_eq!(
            Lean5Mode::from_source_system(SourceSystem::IsabelleHOL),
            Lean5Mode::Classical
        );
        assert_eq!(
            Lean5Mode::from_source_system(SourceSystem::MetamathZFC),
            Lean5Mode::SetTheoretic
        );
    }

    #[test]
    fn test_available_axioms() {
        // Constructive has no axioms
        assert!(Lean5Mode::Constructive.available_axioms().is_empty());

        // Impredicative has PropExt and ProofIrrel
        let imp_axioms = Lean5Mode::Impredicative.available_axioms();
        assert!(imp_axioms.contains(&AxiomId::PropExt));
        assert!(imp_axioms.contains(&AxiomId::ProofIrrel));
        assert!(!imp_axioms.contains(&AxiomId::LEM));

        // Classical has LEM and Choice
        let class_axioms = Lean5Mode::Classical.available_axioms();
        assert!(class_axioms.contains(&AxiomId::LEM));
        assert!(class_axioms.contains(&AxiomId::Choice));

        // SetTheoretic has ZFC axioms
        let set_axioms = Lean5Mode::SetTheoretic.available_axioms();
        assert!(set_axioms.contains(&AxiomId::ZFCExtensionality));
        assert!(set_axioms.contains(&AxiomId::ZFCInfinity));
    }

    #[test]
    fn test_large_elimination() {
        // Constructive always allows large elimination
        assert!(Lean5Mode::Constructive.allows_large_elimination(false));
        assert!(Lean5Mode::Constructive.allows_large_elimination(true));

        // Impredicative restricts from Prop
        assert!(Lean5Mode::Impredicative.allows_large_elimination(false));
        assert!(!Lean5Mode::Impredicative.allows_large_elimination(true));

        // Same for Classical
        assert!(Lean5Mode::Classical.allows_large_elimination(false));
        assert!(!Lean5Mode::Classical.allows_large_elimination(true));

        // Cubical allows both
        assert!(Lean5Mode::Cubical.allows_large_elimination(false));
        assert!(Lean5Mode::Cubical.allows_large_elimination(true));

        // SetTheoretic allows both
        assert!(Lean5Mode::SetTheoretic.allows_large_elimination(false));
        assert!(Lean5Mode::SetTheoretic.allows_large_elimination(true));
    }

    #[test]
    fn test_reflexive_imports() {
        // Every mode can import from itself
        for mode in [
            Lean5Mode::Constructive,
            Lean5Mode::Impredicative,
            Lean5Mode::Cubical,
            Lean5Mode::Classical,
            Lean5Mode::SetTheoretic,
        ] {
            assert!(
                Lean5Mode::can_import(mode, mode),
                "{mode:?} should be able to import from itself"
            );
        }
    }

    #[test]
    fn test_default_mode() {
        // Default mode should be Constructive
        assert_eq!(Lean5Mode::default(), Lean5Mode::Constructive);
    }
}
