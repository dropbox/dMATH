//! K-induction engine for unbounded verification
//!
//! This crate provides k-induction based verification for Rust programs.
//! K-induction extends BMC (bounded model checking) to prove properties
//! for unbounded executions by combining:
//!
//! 1. **Base Case**: Prove property holds for first k steps
//! 2. **Induction Step**: Assume property holds for k consecutive states,
//!    prove it holds for state k+1
//!
//! # Algorithm
//!
//! ```text
//! for k = 1, 2, 3, ...
//!   // Base case: Check if property fails in first k steps
//!   if BMC(k) finds counterexample:
//!     return Disproven(counterexample)
//!
//!   // Induction step: Assume P holds for k steps, check step k+1
//!   if INDUCTION(k) succeeds:
//!     return Proven
//!
//!   // Induction failed, need more steps
//!   k += 1
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_kinduction::{KInduction, KInductionConfig};
//!
//! let config = KInductionConfig::default();
//! let engine = KInduction::new(config);
//!
//! let result = engine.verify(&smt_formula).await?;
//! match result {
//!     KInductionResult::Proven { k } => println!("Proven with k={}", k),
//!     KInductionResult::Disproven { counterexample, k } => println!("Counterexample at k={}", k),
//!     KInductionResult::Unknown { reason } => println!("Unknown: {}", reason),
//! }
//! ```

mod config;
mod engine;
mod formula;
mod invariant;
mod loop_analysis;
pub mod proof;
mod result;

pub use config::{KInductionConfig, KInductionConfigBuilder};
pub use engine::KInduction;
pub use formula::{
    Property, PropertyType, SmtType, StateFormula, StateVariable, TransitionSystem,
    TransitionSystemBuilder,
};
pub use invariant::{Invariant, InvariantSynthesizer, InvariantTemplate};
pub use loop_analysis::{LoopBoundAnalyzer, LoopInfo};
pub use proof::{generate_kinduction_proof, proof_to_json, KInductionProofBuilder};
pub use result::{Counterexample, KInductionResult, KInductionStats, State};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = KInductionConfigBuilder::new()
            .max_k(100)
            .timeout_per_step_ms(5000)
            .build();

        assert_eq!(config.max_k, 100);
        assert_eq!(config.timeout_per_step.as_millis(), 5000);
    }
}
