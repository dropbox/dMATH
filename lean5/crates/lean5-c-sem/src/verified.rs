//! Helpers for verifying C functions
//!
//! This module provides a reusable wrapper around a C function definition,
//! its specification, and optional separation-logic contract. It offers
//! convenience methods to generate verification conditions and dispatch the
//! automated prover.

use crate::auto::{simplify_spec, ProofStatus, VCProver, VerificationSummary};
use crate::sep::SepFuncSpec;
use crate::spec::FuncSpec;
use crate::stmt::FuncDef;
use crate::vcgen::{VCGen, VC};

/// A function with its specification for verification
#[derive(Debug, Clone)]
pub struct VerifiedFunction {
    pub name: String,
    pub description: String,
    pub func: FuncDef,
    pub spec: FuncSpec,
    pub sep_spec: Option<SepFuncSpec>,
}

impl VerifiedFunction {
    /// Generate verification conditions for this function
    pub fn generate_vcs(&self) -> Vec<VC> {
        let mut vcgen = VCGen::new();
        vcgen.gen_function(&self.func, &self.spec)
    }

    /// Verify this function and return summary
    pub fn verify(&self) -> VerificationSummary {
        let vcs = self.generate_vcs();
        let mut prover = VCProver::new();
        prover.prove_all(&vcs)
    }

    /// Get simplified VCs (with constant folding, etc.)
    pub fn simplified_vcs(&self) -> Vec<VC> {
        self.generate_vcs()
            .into_iter()
            .map(|vc| VC {
                description: vc.description,
                obligation: simplify_spec(&vc.obligation),
                location: vc.location,
                kind: vc.kind,
            })
            .collect()
    }

    /// Convenience helper for printing a human-readable summary
    pub fn print_summary(&self) {
        let summary = self.verify();
        println!("Function: {}", self.name);
        println!(
            "  VCs: {} total, {} proved, {} failed, {} unknown",
            summary.total, summary.proved, summary.failed, summary.unknown
        );
        for (desc, status) in &summary.details {
            let marker = match status {
                ProofStatus::Proved(_) => "✓",
                ProofStatus::Failed(_) => "✗",
                ProofStatus::Unknown => "?",
            };
            println!("    {marker} {desc}");
        }
    }
}
