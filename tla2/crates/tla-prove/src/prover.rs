//! Proof manager for orchestrating proof checking
//!
//! The prover coordinates obligation extraction, backend selection, and result caching.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tla_core::ast::{Module, TheoremDecl};

use crate::backend::{ProofBackend, ProofOutcome, SmtBackend};
use crate::context::ProofContext;
use crate::error::ProofResult;
use crate::obligation::{Obligation, ObligationExtractor, ObligationId};

/// Result of checking a single obligation
#[derive(Debug, Clone)]
pub struct ObligationResult {
    /// The obligation that was checked
    pub obligation_id: ObligationId,
    /// The outcome
    pub outcome: ProofOutcome,
    /// Which backend was used
    pub backend: String,
    /// Time taken to check
    pub duration: Duration,
}

/// Result of checking a theorem
#[derive(Debug)]
pub struct TheoremResult {
    /// Name of the theorem
    pub name: String,
    /// Results for each obligation
    pub obligations: Vec<ObligationResult>,
    /// Total time taken
    pub duration: Duration,
}

impl TheoremResult {
    /// Check if all obligations were proved
    pub fn is_proved(&self) -> bool {
        !self.obligations.is_empty() && self.obligations.iter().all(|r| r.outcome.is_proved())
    }

    /// Get the number of proved obligations
    pub fn proved_count(&self) -> usize {
        self.obligations
            .iter()
            .filter(|r| r.outcome.is_proved())
            .count()
    }

    /// Get the number of failed obligations
    pub fn failed_count(&self) -> usize {
        self.obligations
            .iter()
            .filter(|r| r.outcome.is_failed())
            .count()
    }

    /// Get the number of unknown obligations
    pub fn unknown_count(&self) -> usize {
        self.obligations
            .iter()
            .filter(|r| r.outcome.is_unknown())
            .count()
    }
}

/// Result of checking a module
#[derive(Debug)]
pub struct ModuleResult {
    /// Module name
    pub name: String,
    /// Results for each theorem
    pub theorems: Vec<TheoremResult>,
    /// Total time taken
    pub duration: Duration,
}

impl ModuleResult {
    /// Check if all theorems were proved
    pub fn is_proved(&self) -> bool {
        !self.theorems.is_empty() && self.theorems.iter().all(|t| t.is_proved())
    }

    /// Get total number of obligations
    pub fn total_obligations(&self) -> usize {
        self.theorems.iter().map(|t| t.obligations.len()).sum()
    }

    /// Get number of proved obligations
    pub fn proved_count(&self) -> usize {
        self.theorems.iter().map(|t| t.proved_count()).sum()
    }

    /// Get number of failed obligations
    pub fn failed_count(&self) -> usize {
        self.theorems.iter().map(|t| t.failed_count()).sum()
    }

    /// Get number of unknown obligations
    pub fn unknown_count(&self) -> usize {
        self.theorems.iter().map(|t| t.unknown_count()).sum()
    }
}

/// Cache for proof results
pub struct ProofCache {
    results: HashMap<ObligationId, ProofOutcome>,
}

impl ProofCache {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    pub fn get(&self, id: &ObligationId) -> Option<&ProofOutcome> {
        self.results.get(id)
    }

    pub fn insert(&mut self, id: ObligationId, outcome: ProofOutcome) {
        self.results.insert(id, outcome);
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

impl Default for ProofCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The proof manager orchestrates proof checking
pub struct Prover {
    /// Available backends
    backends: Vec<Box<dyn ProofBackend>>,
    /// Proof result cache
    cache: ProofCache,
    /// Timeout per obligation
    timeout: Duration,
}

impl Prover {
    /// Create a new prover with default settings
    pub fn new() -> Self {
        Self {
            backends: vec![Box::new(SmtBackend::new())],
            cache: ProofCache::new(),
            timeout: Duration::from_secs(60),
        }
    }

    /// Add a proof backend
    pub fn add_backend(&mut self, backend: Box<dyn ProofBackend>) {
        self.backends.push(backend);
    }

    /// Set the timeout per obligation
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Check a single obligation
    pub fn check_obligation(
        &mut self,
        obligation: &Obligation,
        context: &ProofContext,
    ) -> ProofResult<ObligationResult> {
        // Check cache first
        if let Some(cached) = self.cache.get(&obligation.id) {
            return Ok(ObligationResult {
                obligation_id: obligation.id.clone(),
                outcome: cached.clone(),
                backend: "cached".to_string(),
                duration: Duration::ZERO,
            });
        }

        let start = Instant::now();

        // Try each backend until one succeeds or all fail
        for backend in &self.backends {
            if !backend.supports(obligation) {
                continue;
            }

            match backend.prove(obligation, context) {
                Ok(outcome) => {
                    let duration = start.elapsed();

                    // Cache the result
                    self.cache.insert(obligation.id.clone(), outcome.clone());

                    return Ok(ObligationResult {
                        obligation_id: obligation.id.clone(),
                        outcome,
                        backend: backend.name().to_string(),
                        duration,
                    });
                }
                Err(_e) => {
                    // Backend failed, try next one
                    continue;
                }
            }
        }

        // No backend could handle it
        let duration = start.elapsed();
        let outcome = ProofOutcome::Unknown {
            reason: "no backend could handle this obligation".to_string(),
        };

        self.cache.insert(obligation.id.clone(), outcome.clone());

        Ok(ObligationResult {
            obligation_id: obligation.id.clone(),
            outcome,
            backend: "none".to_string(),
            duration,
        })
    }

    /// Check a theorem
    pub fn check_theorem(
        &mut self,
        theorem: &TheoremDecl,
        context: &mut ProofContext,
    ) -> ProofResult<TheoremResult> {
        let start = Instant::now();

        let thm_name = theorem
            .name
            .as_ref()
            .map(|n| n.node.clone())
            .unwrap_or_else(|| "_THEOREM".to_string());

        // Extract obligations
        let extractor = ObligationExtractor::new();
        let obligations = extractor.extract_theorem(theorem)?;

        // Check each obligation
        let mut results = Vec::new();
        for obligation in &obligations {
            let result = self.check_obligation(obligation, context)?;
            results.push(result);
        }

        // If theorem is proved, add it to context
        let all_proved = results.iter().all(|r| r.outcome.is_proved());
        if all_proved {
            context.add_proved_fact(thm_name.clone(), theorem.body.clone());
        }

        Ok(TheoremResult {
            name: thm_name,
            obligations: results,
            duration: start.elapsed(),
        })
    }

    /// Check all theorems in a module
    pub fn check_module(&mut self, module: &Module) -> ProofResult<ModuleResult> {
        let start = Instant::now();
        let mut context = ProofContext::from_module(module);
        let mut theorem_results = Vec::new();

        for unit in &module.units {
            if let tla_core::ast::Unit::Theorem(thm) = &unit.node {
                let result = self.check_theorem(thm, &mut context)?;
                theorem_results.push(result);
            }
        }

        Ok(ModuleResult {
            name: module.name.node.clone(),
            theorems: theorem_results,
            duration: start.elapsed(),
        })
    }

    /// Get the number of cached results
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache = ProofCache::new();
    }
}

impl Default for Prover {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use tla_core::ast::{Expr, Proof};
    use tla_core::span::{FileId, Span, Spanned};

    fn span() -> Span {
        Span::new(FileId(0), 0, 0)
    }

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::new(node, span())
    }

    fn make_theorem(name: &str, body: Expr, proof: Option<Proof>) -> TheoremDecl {
        TheoremDecl {
            name: Some(spanned(name.to_string())),
            body: spanned(body),
            proof: proof.map(spanned),
        }
    }

    #[test]
    fn test_prove_simple_theorem() {
        let mut prover = Prover::new();
        let mut context = ProofContext::new();

        // THEOREM T == x > 5 => x > 3
        let theorem = make_theorem(
            "T",
            Expr::Implies(
                Box::new(spanned(Expr::Gt(
                    Box::new(spanned(Expr::Ident("x".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(5)))),
                ))),
                Box::new(spanned(Expr::Gt(
                    Box::new(spanned(Expr::Ident("x".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(3)))),
                ))),
            ),
            None,
        );

        let result = prover.check_theorem(&theorem, &mut context).unwrap();
        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_with_obvious() {
        let mut prover = Prover::new();
        let mut context = ProofContext::new();

        // THEOREM T == TRUE OBVIOUS
        let theorem = make_theorem("T", Expr::Bool(true), Some(Proof::Obvious));

        let result = prover.check_theorem(&theorem, &mut context).unwrap();
        assert!(result.is_proved());
    }

    #[test]
    fn test_prove_omitted() {
        let mut prover = Prover::new();
        let mut context = ProofContext::new();

        // THEOREM T == FALSE OMITTED
        let theorem = make_theorem("T", Expr::Bool(false), Some(Proof::Omitted));

        let result = prover.check_theorem(&theorem, &mut context).unwrap();
        // OMITTED creates no obligations, so technically "proved"
        assert_eq!(result.obligations.len(), 0);
    }

    #[test]
    fn test_caching() {
        let mut prover = Prover::new();
        let mut context = ProofContext::new();

        let theorem = make_theorem("T", Expr::Bool(true), None);

        // First check
        let result1 = prover.check_theorem(&theorem, &mut context).unwrap();
        assert!(result1.is_proved());
        assert_eq!(prover.cache_size(), 1);

        // Second check should be cached
        let result2 = prover.check_theorem(&theorem, &mut context).unwrap();
        assert!(result2.is_proved());
        assert_eq!(result2.obligations[0].backend, "cached");
    }

    #[test]
    fn test_prove_false_theorem() {
        let mut prover = Prover::new();
        let mut context = ProofContext::new();

        // THEOREM T == x > 3 => x > 5 (this is false!)
        let theorem = make_theorem(
            "T",
            Expr::Implies(
                Box::new(spanned(Expr::Gt(
                    Box::new(spanned(Expr::Ident("x".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(3)))),
                ))),
                Box::new(spanned(Expr::Gt(
                    Box::new(spanned(Expr::Ident("x".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(5)))),
                ))),
            ),
            None,
        );

        let result = prover.check_theorem(&theorem, &mut context).unwrap();
        assert!(!result.is_proved());
        assert_eq!(result.failed_count(), 1);
    }

    #[test]
    fn test_context_propagation() {
        let mut prover = Prover::new();
        let mut context = ProofContext::new();

        // Add assumption: x > 5
        context.add_assumed_fact(
            "A".to_string(),
            spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(5)))),
            )),
        );

        // THEOREM T == x > 3 (follows from assumption)
        let theorem = make_theorem(
            "T",
            Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(3)))),
            ),
            None,
        );

        let result = prover.check_theorem(&theorem, &mut context).unwrap();
        assert!(result.is_proved());
    }
}
