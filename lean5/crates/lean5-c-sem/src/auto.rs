//! VC → lean5-auto Bridge for Automated Proof Discharge
//!
//! This module provides integration between the C verification condition
//! generator and the lean5-auto SMT solver for automated proof discharge.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    VC to SMT Bridge                                  │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  C VCs (Spec) ───────► Lean5 Expr ───────► SMT Solver               │
//! │  (requires/ensures)    translate     prove_or_disprove               │
//! │                                                                      │
//! │  Proof Status ◄─────── ProofResult ◄─────── Sat/Unsat               │
//! │  (Proved/Failed)       (witness/proof)                               │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Supported VC Types
//!
//! - Arithmetic comparisons: `x >= 0`, `a + b < n`
//! - Pointer validity: `valid(p)`, `valid_range(p, 0, n)`
//! - Boolean combinations: `P && Q`, `P || Q`, `!P`
//! - Quantified: `forall i. P(i)`, `exists x. Q(x)`
//!
//! ## Example
//!
//! ```ignore
//! use lean5_c_sem::auto::VCProver;
//! use lean5_c_sem::vcgen::VCGen;
//!
//! let mut vcgen = VCGen::new();
//! // Generate VCs for a function
//! let vcs = vcgen.gen_function(&func, &spec);
//!
//! // Try to prove them
//! let mut prover = VCProver::new();
//! for vc in &vcs {
//!     match prover.prove_vc(vc) {
//!         ProofStatus::Proved(witness) => println!("✓ {}", vc.description),
//!         ProofStatus::Failed(reason) => println!("✗ {}: {}", vc.description, reason),
//!         ProofStatus::Unknown => println!("? {}", vc.description),
//!     }
//! }
//! ```

use crate::expr::BinOp;
use crate::spec::Spec;
use crate::vcgen::{VCKind, VC};
use lean5_kernel::{Environment, Expr};

/// Result of attempting to prove a verification condition
#[derive(Debug, Clone)]
pub enum ProofStatus {
    /// VC was successfully proved
    /// Contains the proof witness (Lean5 expression)
    Proved(Option<Expr>),
    /// VC could not be proved
    /// Contains reason/counterexample if available
    Failed(String),
    /// Prover could not determine provability (timeout, unsupported, etc.)
    Unknown,
}

/// Summary of verification results for multiple VCs
#[derive(Debug, Clone, Default)]
pub struct VerificationSummary {
    /// Total number of VCs
    pub total: usize,
    /// Number of proved VCs
    pub proved: usize,
    /// Number of failed VCs
    pub failed: usize,
    /// Number of unknown VCs
    pub unknown: usize,
    /// Details for each VC
    pub details: Vec<(String, ProofStatus)>,
}

impl VerificationSummary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, description: String, status: ProofStatus) {
        self.total += 1;
        match &status {
            ProofStatus::Proved(_) => self.proved += 1,
            ProofStatus::Failed(_) => self.failed += 1,
            ProofStatus::Unknown => self.unknown += 1,
        }
        self.details.push((description, status));
    }

    /// Check if all VCs were proved
    pub fn all_proved(&self) -> bool {
        self.proved == self.total
    }

    /// Check if any VCs failed
    pub fn has_failures(&self) -> bool {
        self.failed > 0
    }
}

/// Verification condition prover using lean5-auto
pub struct VCProver {
    /// Kernel environment for type checking proofs
    env: Environment,
    /// Timeout for SMT solver (milliseconds)
    timeout_ms: u64,
    /// Whether to use arithmetic theory
    use_arithmetic: bool,
    /// Whether to use array theory
    use_arrays: bool,
}

impl Default for VCProver {
    fn default() -> Self {
        Self::new()
    }
}

impl VCProver {
    pub fn new() -> Self {
        Self {
            env: Environment::new(),
            timeout_ms: 5000, // 5 second default
            use_arithmetic: true,
            use_arrays: true,
        }
    }

    /// Set timeout in milliseconds
    #[must_use]
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Enable/disable arithmetic theory
    #[must_use]
    pub fn with_arithmetic(mut self, enable: bool) -> Self {
        self.use_arithmetic = enable;
        self
    }

    /// Enable/disable array theory
    #[must_use]
    pub fn with_arrays(mut self, enable: bool) -> Self {
        self.use_arrays = enable;
        self
    }

    /// Try to prove a single verification condition
    pub fn prove_vc(&mut self, vc: &VC) -> ProofStatus {
        // Translate Spec to Lean5 Expr
        let Some(lean_expr) = self.spec_to_expr(&vc.obligation) else {
            return ProofStatus::Unknown;
        };

        // Try to prove using SMT bridge
        self.prove_expr(&lean_expr, &vc.kind)
    }

    /// Try to prove a Spec directly
    pub fn prove_spec(&mut self, spec: &Spec) -> ProofStatus {
        let Some(lean_expr) = self.spec_to_expr(spec) else {
            return ProofStatus::Unknown;
        };
        self.prove_expr(&lean_expr, &VCKind::Assertion)
    }

    /// Prove all VCs and return summary
    pub fn prove_all(&mut self, vcs: &[VC]) -> VerificationSummary {
        let mut summary = VerificationSummary::new();
        for vc in vcs {
            let status = self.prove_vc(vc);
            summary.add(vc.description.clone(), status);
        }
        summary
    }

    /// Check if a Spec is trivially true
    pub fn is_trivially_true(&self, spec: &Spec) -> bool {
        match spec {
            Spec::True => true,
            Spec::And(specs) => specs.iter().all(|s| self.is_trivially_true(s)),
            Spec::Implies(p, q) => self.is_trivially_false(p) || self.is_trivially_true(q),
            Spec::BinOp { op, left, right } => {
                // Check for reflexive comparisons
                if left == right {
                    matches!(op, BinOp::Eq | BinOp::Le | BinOp::Ge)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Check if a Spec is trivially false
    pub fn is_trivially_false(&self, spec: &Spec) -> bool {
        match spec {
            Spec::False => true,
            Spec::Or(specs) => specs.iter().all(|s| self.is_trivially_false(s)),
            Spec::BinOp { op, left, right } => {
                // Check for obviously false comparisons
                if left == right {
                    matches!(op, BinOp::Ne | BinOp::Lt | BinOp::Gt)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Translate Spec to Lean5 kernel expression
    fn spec_to_expr(&self, spec: &Spec) -> Option<Expr> {
        let mut ctx = crate::translate::TranslationContext::new();
        Some(ctx.translate_spec(spec))
    }

    /// Core proving logic using lean5-auto
    fn prove_expr(&self, expr: &Expr, kind: &VCKind) -> ProofStatus {
        // Create SMT bridge
        let mut bridge = lean5_auto::bridge::SmtBridge::new(&self.env);

        // Try to prove
        match bridge.prove(expr) {
            Some(result) => {
                // Proof found
                ProofStatus::Proved(result.proof_term)
            }
            None => {
                // Check if this is an easy case we can handle specially
                if let Some(status) = self.try_simple_proof(expr, kind) {
                    return status;
                }
                ProofStatus::Unknown
            }
        }
    }

    /// Try simple proof strategies for common VC patterns
    fn try_simple_proof(&self, expr: &Expr, _kind: &VCKind) -> Option<ProofStatus> {
        // First, try structural analysis of the expression
        if let Some(status) = self.try_structural_proof(expr) {
            return Some(status);
        }

        None
    }

    /// Try proving based on the structure of the expression
    fn try_structural_proof(&self, expr: &Expr) -> Option<ProofStatus> {
        let head = expr.get_app_fn();
        let args = expr.get_app_args();

        match head {
            Expr::Const(name, _) => {
                let name_str = name.to_string();
                match name_str.as_str() {
                    // True is trivially proved
                    "True" => Some(ProofStatus::Proved(None)),
                    // False cannot be proved
                    "False" => Some(ProofStatus::Failed("Cannot prove False".to_string())),

                    // Equality: check reflexivity
                    // Handle both Eq α a b (3 args) and Eq a b (2 args, simplified)
                    "Eq" if args.len() >= 2 => {
                        let len = args.len();
                        let lhs = &args[len - 2];
                        let rhs = &args[len - 1];
                        if self.exprs_structurally_equal(lhs, rhs) {
                            Some(ProofStatus::Proved(None))
                        } else {
                            None
                        }
                    }

                    // Less than or equal: check reflexivity and constants
                    "LE.le" | "GE.ge" | "Int.le" | "Nat.le" if args.len() >= 2 => {
                        let len = args.len();
                        let lhs = &args[len - 2];
                        let rhs = &args[len - 1];
                        // x ≤ x is true
                        if self.exprs_structurally_equal(lhs, rhs) {
                            return Some(ProofStatus::Proved(None));
                        }
                        // Try constant comparison
                        if let (Some(a), Some(b)) =
                            (self.try_extract_int(lhs), self.try_extract_int(rhs))
                        {
                            if a <= b {
                                return Some(ProofStatus::Proved(None));
                            }
                            return Some(ProofStatus::Failed(format!("{a} > {b}")));
                        }
                        None
                    }

                    // Strict less than: check constants
                    "LT.lt" | "GT.gt" | "Int.lt" | "Nat.lt" if args.len() >= 2 => {
                        let len = args.len();
                        let lhs = &args[len - 2];
                        let rhs = &args[len - 1];
                        // x < x is false
                        if self.exprs_structurally_equal(lhs, rhs) {
                            return Some(ProofStatus::Failed("x < x is false".to_string()));
                        }
                        // Try constant comparison
                        if let (Some(a), Some(b)) =
                            (self.try_extract_int(lhs), self.try_extract_int(rhs))
                        {
                            if a < b {
                                return Some(ProofStatus::Proved(None));
                            }
                            return Some(ProofStatus::Failed(format!("{a} >= {b}")));
                        }
                        None
                    }

                    // Conjunction: prove all parts
                    "And" if args.len() == 2 => {
                        let p_status = self.try_structural_proof(args[0]);
                        let q_status = self.try_structural_proof(args[1]);
                        match (p_status, q_status) {
                            (Some(ProofStatus::Proved(_)), Some(ProofStatus::Proved(_))) => {
                                Some(ProofStatus::Proved(None))
                            }
                            (Some(ProofStatus::Failed(reason)), _) => {
                                Some(ProofStatus::Failed(format!("Left conjunct: {reason}")))
                            }
                            (_, Some(ProofStatus::Failed(reason))) => {
                                Some(ProofStatus::Failed(format!("Right conjunct: {reason}")))
                            }
                            _ => None,
                        }
                    }

                    // Disjunction: prove one part
                    "Or" if args.len() == 2 => {
                        // Try to prove either side
                        if let Some(ProofStatus::Proved(_)) = self.try_structural_proof(args[0]) {
                            return Some(ProofStatus::Proved(None));
                        }
                        if let Some(ProofStatus::Proved(_)) = self.try_structural_proof(args[1]) {
                            return Some(ProofStatus::Proved(None));
                        }
                        None
                    }

                    // Negation
                    "Not" if args.len() == 1 => match self.try_structural_proof(args[0]) {
                        Some(ProofStatus::Failed(_)) => Some(ProofStatus::Proved(None)),
                        Some(ProofStatus::Proved(_)) => {
                            Some(ProofStatus::Failed("Cannot prove Not(True)".to_string()))
                        }
                        _ => None,
                    },

                    _ => None,
                }
            }
            // Pi type (implication P → Q or universal quantifier ∀ x : T, P(x))
            Expr::Pi(_, domain, codomain) => {
                // Non-dependent Pi is implication: P → Q
                if !codomain.has_loose_bvars() {
                    // P → True is always true
                    if let Some(ProofStatus::Proved(_)) = self.try_structural_proof(codomain) {
                        return Some(ProofStatus::Proved(None));
                    }
                    // False → Q is always true
                    if let Some(ProofStatus::Failed(_)) = self.try_structural_proof(domain) {
                        return Some(ProofStatus::Proved(None));
                    }
                }
                // Dependent Pi is universal quantifier: ∀ x : T, P(x)
                // Try to prove the body is trivially true for any x
                if let Some(ProofStatus::Proved(_)) = self.try_structural_proof(codomain) {
                    return Some(ProofStatus::Proved(None));
                }
                None
            }
            // Exists quantifier
            Expr::App(func, _) => {
                let head = func.get_app_fn();
                if let Expr::Const(name, _) = head {
                    if name.to_string() == "Exists" {
                        // ∃ x : T, P(x)
                        // For structural proofs, we can prove if the body is provable
                        // for some concrete witness (limited to simple cases)
                        // For now, return None to fall back to SMT
                        return None;
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Check if two expressions are structurally equal
    fn exprs_structurally_equal(&self, a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::BVar(i), Expr::BVar(j)) => i == j,
            (Expr::FVar(i), Expr::FVar(j)) => i == j,
            (Expr::Const(n1, _), Expr::Const(n2, _)) => n1 == n2,
            (Expr::App(f1, a1), Expr::App(f2, a2)) => {
                self.exprs_structurally_equal(f1, f2) && self.exprs_structurally_equal(a1, a2)
            }
            (Expr::Lit(l1), Expr::Lit(l2)) => l1 == l2,
            _ => false,
        }
    }

    /// Try to extract an integer constant from an expression
    fn try_extract_int(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Lit(lean5_kernel::Literal::Nat(n)) => i64::try_from(*n).ok(),
            Expr::App(f, arg) => {
                let head = f.get_app_fn();
                if let Expr::Const(name, _) = head {
                    let name_str = name.to_string();
                    match name_str.as_str() {
                        "Int.ofNat" => self.try_extract_nat(arg),
                        "Int.negOfNat" => {
                            // Handle negation carefully to avoid overflow.
                            // For magnitude m, the result is -m unless m > i64::MAX,
                            // in which case it's only representable if m == i64::MAX + 1 (i.e., i64::MIN's magnitude).
                            let m = self.try_extract_nat(arg)?;
                            if m >= 0 {
                                m.checked_neg()
                            } else {
                                // m is already negative from try_extract_nat, shouldn't happen
                                // but handle defensively
                                None
                            }
                        }
                        "Nat.succ" => self.try_extract_nat(expr),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            Expr::Const(name, _) => {
                if name.to_string() == "Nat.zero" {
                    Some(0)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to extract a natural number from an expression (Nat.zero / Nat.succ chain)
    /// Returns the value as i64; returns None if the value exceeds i64::MAX
    fn try_extract_nat(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Lit(lean5_kernel::Literal::Nat(n)) => i64::try_from(*n).ok(),
            Expr::Const(name, _) if name.to_string() == "Nat.zero" => Some(0),
            Expr::App(f, arg) => {
                if let Expr::Const(name, _) = f.as_ref() {
                    if name.to_string() == "Nat.succ" {
                        return self.try_extract_nat(arg).and_then(|n| n.checked_add(1));
                    }
                }
                None
            }
            _ => None,
        }
    }
}

/// Quick verification of a specification
/// Returns true if provable, false otherwise
pub fn quick_check(spec: &Spec) -> bool {
    let mut prover = VCProver::new().with_timeout(1000);
    matches!(prover.prove_spec(spec), ProofStatus::Proved(_))
}

/// Verify a set of VCs and print results
pub fn verify_and_report(vcs: &[VC]) -> VerificationSummary {
    let mut prover = VCProver::new();
    let summary = prover.prove_all(vcs);

    // Print summary
    eprintln!(
        "Verification: {}/{} proved, {} failed, {} unknown",
        summary.proved, summary.total, summary.failed, summary.unknown
    );

    for (desc, status) in &summary.details {
        match status {
            ProofStatus::Proved(_) => eprintln!("  ✓ {desc}"),
            ProofStatus::Failed(reason) => eprintln!("  ✗ {desc} - {reason}"),
            ProofStatus::Unknown => eprintln!("  ? {desc} - unknown"),
        }
    }

    summary
}

/// Simplify a specification before proving
/// Applies simple algebraic simplifications
pub fn simplify_spec(spec: &Spec) -> Spec {
    match spec {
        // Conjunction simplifications
        Spec::And(specs) => {
            let simplified: Vec<_> = specs
                .iter()
                .map(simplify_spec)
                .filter(|s| !matches!(s, Spec::True))
                .collect();
            if simplified.is_empty() {
                Spec::True
            } else if simplified.iter().any(|s| matches!(s, Spec::False)) {
                Spec::False
            } else if simplified.len() == 1 {
                simplified.into_iter().next().unwrap()
            } else {
                Spec::And(simplified)
            }
        }
        // Disjunction simplifications
        Spec::Or(specs) => {
            let simplified: Vec<_> = specs
                .iter()
                .map(simplify_spec)
                .filter(|s| !matches!(s, Spec::False))
                .collect();
            if simplified.is_empty() {
                Spec::False
            } else if simplified.iter().any(|s| matches!(s, Spec::True)) {
                Spec::True
            } else if simplified.len() == 1 {
                simplified.into_iter().next().unwrap()
            } else {
                Spec::Or(simplified)
            }
        }
        // Negation simplifications
        Spec::Not(inner) => {
            let inner_simp = simplify_spec(inner);
            match inner_simp {
                Spec::True => Spec::False,
                Spec::False => Spec::True,
                Spec::Not(double_neg) => *double_neg,
                other => Spec::Not(Box::new(other)),
            }
        }
        // Implication simplifications
        Spec::Implies(p, q) => {
            let p_simp = simplify_spec(p);
            let q_simp = simplify_spec(q);
            match (&p_simp, &q_simp) {
                (Spec::False, _) | (_, Spec::True) => Spec::True,
                (Spec::True, q) => q.clone(),
                _ => Spec::Implies(Box::new(p_simp), Box::new(q_simp)),
            }
        }
        // Comparison simplifications
        Spec::BinOp { op, left, right } => {
            let left_simp = simplify_spec(left);
            let right_simp = simplify_spec(right);

            // Reflexive comparisons
            if left_simp == right_simp {
                match op {
                    BinOp::Eq | BinOp::Le | BinOp::Ge => return Spec::True,
                    BinOp::Ne | BinOp::Lt | BinOp::Gt => return Spec::False,
                    _ => {}
                }
            }

            // Constant folding for integers
            if let (Spec::Int(a), Spec::Int(b)) = (&left_simp, &right_simp) {
                match op {
                    BinOp::Eq => return if a == b { Spec::True } else { Spec::False },
                    BinOp::Ne => return if a != b { Spec::True } else { Spec::False },
                    BinOp::Lt => return if a < b { Spec::True } else { Spec::False },
                    BinOp::Le => return if a <= b { Spec::True } else { Spec::False },
                    BinOp::Gt => return if a > b { Spec::True } else { Spec::False },
                    BinOp::Ge => return if a >= b { Spec::True } else { Spec::False },
                    BinOp::Add => return Spec::Int(a + b),
                    BinOp::Sub => return Spec::Int(a - b),
                    BinOp::Mul => return Spec::Int(a * b),
                    _ => {}
                }
            }

            Spec::BinOp {
                op: *op,
                left: Box::new(left_simp),
                right: Box::new(right_simp),
            }
        }
        // Pass through other specs
        _ => spec.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::CExpr;
    use crate::spec::FuncSpec;
    use crate::stmt::{CStmt, FuncDef, FuncParam, StorageClass};
    use crate::types::CType;
    use crate::vcgen::VCGen;

    #[test]
    fn test_trivially_true() {
        let prover = VCProver::new();

        assert!(prover.is_trivially_true(&Spec::True));
        assert!(prover.is_trivially_true(&Spec::and(vec![Spec::True, Spec::True])));

        // x == x is trivially true
        let eq_self = Spec::eq(Spec::var("x"), Spec::var("x"));
        assert!(prover.is_trivially_true(&eq_self));

        // x >= x is trivially true
        let ge_self = Spec::ge(Spec::var("x"), Spec::var("x"));
        assert!(prover.is_trivially_true(&ge_self));
    }

    #[test]
    fn test_trivially_false() {
        let prover = VCProver::new();

        assert!(prover.is_trivially_false(&Spec::False));
        assert!(prover.is_trivially_false(&Spec::or(vec![Spec::False, Spec::False])));

        // x != x is trivially false
        let ne_self = Spec::ne(Spec::var("x"), Spec::var("x"));
        assert!(prover.is_trivially_false(&ne_self));

        // x < x is trivially false
        let lt_self = Spec::lt(Spec::var("x"), Spec::var("x"));
        assert!(prover.is_trivially_false(&lt_self));
    }

    #[test]
    fn test_simplify_and() {
        // True && P = P
        let spec = Spec::and(vec![Spec::True, Spec::var("P")]);
        assert_eq!(simplify_spec(&spec), Spec::var("P"));

        // False && P = False
        let spec = Spec::and(vec![Spec::False, Spec::var("P")]);
        assert_eq!(simplify_spec(&spec), Spec::False);

        // True && True = True
        let spec = Spec::and(vec![Spec::True, Spec::True]);
        assert_eq!(simplify_spec(&spec), Spec::True);
    }

    #[test]
    fn test_simplify_or() {
        // False || P = P
        let spec = Spec::or(vec![Spec::False, Spec::var("P")]);
        assert_eq!(simplify_spec(&spec), Spec::var("P"));

        // True || P = True
        let spec = Spec::or(vec![Spec::True, Spec::var("P")]);
        assert_eq!(simplify_spec(&spec), Spec::True);
    }

    #[test]
    fn test_simplify_implies() {
        // False => P = True
        let spec = Spec::implies(Spec::False, Spec::var("P"));
        assert_eq!(simplify_spec(&spec), Spec::True);

        // P => True = True
        let spec = Spec::implies(Spec::var("P"), Spec::True);
        assert_eq!(simplify_spec(&spec), Spec::True);

        // True => P = P
        let spec = Spec::implies(Spec::True, Spec::var("P"));
        assert_eq!(simplify_spec(&spec), Spec::var("P"));
    }

    #[test]
    fn test_simplify_not() {
        // !!P = P
        let spec = Spec::not(Spec::not(Spec::var("P")));
        assert_eq!(simplify_spec(&spec), Spec::var("P"));

        // !True = False
        assert_eq!(simplify_spec(&Spec::not(Spec::True)), Spec::False);

        // !False = True
        assert_eq!(simplify_spec(&Spec::not(Spec::False)), Spec::True);
    }

    #[test]
    fn test_simplify_comparison() {
        // x == x = True
        let spec = Spec::eq(Spec::var("x"), Spec::var("x"));
        assert_eq!(simplify_spec(&spec), Spec::True);

        // x != x = False
        let spec = Spec::ne(Spec::var("x"), Spec::var("x"));
        assert_eq!(simplify_spec(&spec), Spec::False);

        // 1 < 2 = True
        let spec = Spec::lt(Spec::int(1), Spec::int(2));
        assert_eq!(simplify_spec(&spec), Spec::True);

        // 2 < 1 = False
        let spec = Spec::lt(Spec::int(2), Spec::int(1));
        assert_eq!(simplify_spec(&spec), Spec::False);

        // 1 + 2 = 3
        let spec = Spec::binop(BinOp::Add, Spec::int(1), Spec::int(2));
        assert_eq!(simplify_spec(&spec), Spec::Int(3));
    }

    #[test]
    fn test_constant_folding() {
        // 3 >= 0 = True
        let spec = Spec::ge(Spec::int(3), Spec::int(0));
        assert_eq!(simplify_spec(&spec), Spec::True);

        // -1 >= 0 = False
        let spec = Spec::ge(Spec::int(-1), Spec::int(0));
        assert_eq!(simplify_spec(&spec), Spec::False);
    }

    #[test]
    fn test_verification_summary() {
        let mut summary = VerificationSummary::new();

        summary.add("VC1".to_string(), ProofStatus::Proved(None));
        summary.add("VC2".to_string(), ProofStatus::Failed("reason".to_string()));
        summary.add("VC3".to_string(), ProofStatus::Unknown);

        assert_eq!(summary.total, 3);
        assert_eq!(summary.proved, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.unknown, 1);
        assert!(!summary.all_proved());
        assert!(summary.has_failures());
    }

    #[test]
    fn test_trivial_specs() {
        let prover = VCProver::new();

        // x == x should be trivially true (simplification)
        let eq_self = Spec::eq(Spec::var("x"), Spec::var("x"));
        assert!(prover.is_trivially_true(&eq_self));

        // After simplification, x == x becomes True
        let simplified = simplify_spec(&eq_self);
        assert_eq!(simplified, Spec::True);
    }

    #[test]
    fn test_abs_function_vcs() {
        // Generate VCs for abs function
        let mut vcgen = VCGen::new();

        let func = FuncDef {
            name: "abs".into(),
            return_type: CType::int(),
            params: vec![FuncParam {
                name: "n".into(),
                ty: CType::int(),
            }],
            body: Box::new(CStmt::if_else(
                CExpr::binop(BinOp::Lt, CExpr::var("n"), CExpr::int(0)),
                CStmt::return_stmt(Some(CExpr::unary(
                    crate::expr::UnaryOp::Neg,
                    CExpr::var("n"),
                ))),
                CStmt::return_stmt(Some(CExpr::var("n"))),
            )),
            variadic: false,
            storage: StorageClass::Auto,
        };

        let spec = FuncSpec {
            requires: vec![Spec::True],
            ensures: vec![Spec::ge(Spec::result(), Spec::int(0))],
            ..Default::default()
        };

        let vcs = vcgen.gen_function(&func, &spec);
        assert!(!vcs.is_empty());

        // Verify the VCs
        let mut prover = VCProver::new();
        let summary = prover.prove_all(&vcs);

        // At minimum, we should get results for all VCs
        assert_eq!(summary.total, vcs.len());
    }

    #[test]
    fn test_prover_creation() {
        let prover = VCProver::new()
            .with_timeout(2000)
            .with_arithmetic(true)
            .with_arrays(false);

        assert_eq!(prover.timeout_ms, 2000);
        assert!(prover.use_arithmetic);
        assert!(!prover.use_arrays);
    }

    #[test]
    fn test_swap_function_vcs() {
        // Generate VCs for swap function:
        // void swap(int *x, int *y) {
        //     int tmp = *x;
        //     *x = *y;
        //     *y = tmp;
        // }
        //
        // Separation logic spec:
        // PRE:  x ↦ a * y ↦ b
        // POST: x ↦ b * y ↦ a
        use crate::sep::{SepAssertion, SepFuncSpec, Share};

        let mut vcgen = VCGen::new();

        // Build the swap function
        let func = FuncDef {
            name: "swap".into(),
            return_type: CType::Void,
            params: vec![
                FuncParam {
                    name: "x".into(),
                    ty: CType::Pointer(Box::new(CType::int())),
                },
                FuncParam {
                    name: "y".into(),
                    ty: CType::Pointer(Box::new(CType::int())),
                },
            ],
            body: Box::new(CStmt::block(vec![
                // int tmp = *x;
                CStmt::decl_init("tmp", CType::int(), CExpr::deref(CExpr::var("x"))),
                // *x = *y;
                CStmt::Expr(CExpr::assign(
                    CExpr::deref(CExpr::var("x")),
                    CExpr::deref(CExpr::var("y")),
                )),
                // *y = tmp;
                CStmt::Expr(CExpr::assign(
                    CExpr::deref(CExpr::var("y")),
                    CExpr::var("tmp"),
                )),
            ])),
            variadic: false,
            storage: StorageClass::Auto,
        };

        // ACSL-style spec
        let spec = FuncSpec {
            requires: vec![
                Spec::valid(Spec::var("x")),
                Spec::valid(Spec::var("y")),
                Spec::Separated(vec![Spec::var("x"), Spec::var("y")]),
            ],
            ensures: vec![
                Spec::eq(
                    Spec::Expr(CExpr::deref(CExpr::var("x"))),
                    Spec::old(Spec::Expr(CExpr::deref(CExpr::var("y")))),
                ),
                Spec::eq(
                    Spec::Expr(CExpr::deref(CExpr::var("y"))),
                    Spec::old(Spec::Expr(CExpr::deref(CExpr::var("x")))),
                ),
            ],
            ..Default::default()
        };

        // Generate VCs
        let vcs = vcgen.gen_function(&func, &spec);
        assert!(!vcs.is_empty(), "Should generate VCs for swap");

        // Build separation logic spec for additional checking
        let sep_spec = SepFuncSpec::new(
            SepAssertion::sep_conj(
                SepAssertion::data_at(CExpr::var("x"), CType::int(), Spec::var("a"), Share::Full),
                SepAssertion::data_at(CExpr::var("y"), CType::int(), Spec::var("b"), Share::Full),
            ),
            SepAssertion::sep_conj(
                SepAssertion::data_at(CExpr::var("x"), CType::int(), Spec::var("b"), Share::Full),
                SepAssertion::data_at(CExpr::var("y"), CType::int(), Spec::var("a"), Share::Full),
            ),
        );

        // Check that pre and post are different (values swapped)
        assert_ne!(sep_spec.pre, sep_spec.post);

        // Check pointers mentioned
        let pre_ptrs = sep_spec.pre.mentioned_pointers();
        let post_ptrs = sep_spec.post.mentioned_pointers();
        assert_eq!(pre_ptrs.len(), 2);
        assert_eq!(post_ptrs.len(), 2);

        // Verify the VCs
        let mut prover = VCProver::new();
        let summary = prover.prove_all(&vcs);
        assert_eq!(summary.total, vcs.len());
    }

    #[test]
    fn test_increment_function_vcs() {
        // Simple function: increment a pointer value
        // void incr(int *p) { *p = *p + 1; }
        // PRE:  valid(p) && *p == n
        // POST: *p == n + 1

        let mut vcgen = VCGen::new();

        let func = FuncDef {
            name: "incr".into(),
            return_type: CType::Void,
            params: vec![FuncParam {
                name: "p".into(),
                ty: CType::Pointer(Box::new(CType::int())),
            }],
            body: Box::new(CStmt::Expr(CExpr::assign(
                CExpr::deref(CExpr::var("p")),
                CExpr::add(CExpr::deref(CExpr::var("p")), CExpr::int(1)),
            ))),
            variadic: false,
            storage: StorageClass::Auto,
        };

        let spec = FuncSpec {
            requires: vec![Spec::valid(Spec::var("p"))],
            ensures: vec![Spec::eq(
                Spec::Expr(CExpr::deref(CExpr::var("p"))),
                Spec::binop(
                    BinOp::Add,
                    Spec::old(Spec::Expr(CExpr::deref(CExpr::var("p")))),
                    Spec::int(1),
                ),
            )],
            ..Default::default()
        };

        let vcs = vcgen.gen_function(&func, &spec);
        assert!(!vcs.is_empty(), "Should generate VCs for incr function");

        // Verify the VCs (postcondition VCs should be present)
        let has_postcondition = vcs
            .iter()
            .any(|vc| vc.kind == crate::vcgen::VCKind::Postcondition);
        assert!(has_postcondition, "Should have postcondition VC");
    }

    #[test]
    fn test_structural_proof_true() {
        let mut prover = VCProver::new();
        let status = prover.prove_spec(&Spec::True);
        assert!(matches!(status, ProofStatus::Proved(_)));
    }

    #[test]
    fn test_structural_proof_false() {
        let mut prover = VCProver::new();
        let status = prover.prove_spec(&Spec::False);
        assert!(matches!(status, ProofStatus::Failed(_)));
    }

    #[test]
    fn test_structural_proof_reflexive_eq() {
        let mut prover = VCProver::new();
        // x = x should be proved
        let spec = Spec::eq(Spec::var("x"), Spec::var("x"));
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "Reflexive equality should be proved, got {status:?}"
        );
    }

    #[test]
    fn test_structural_proof_reflexive_le() {
        let mut prover = VCProver::new();
        // x ≤ x should be proved
        let spec = Spec::le(Spec::var("x"), Spec::var("x"));
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "Reflexive ≤ should be proved, got {status:?}"
        );
    }

    #[test]
    fn test_structural_proof_constant_comparison() {
        let mut prover = VCProver::new();

        // 1 < 2 should be proved
        let spec = Spec::lt(Spec::int(1), Spec::int(2));
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "1 < 2 should be proved, got {status:?}"
        );

        // 0 ≤ 5 should be proved
        let spec = Spec::le(Spec::int(0), Spec::int(5));
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "0 ≤ 5 should be proved, got {status:?}"
        );

        // 10 ≥ 3 should be proved
        let spec = Spec::ge(Spec::int(10), Spec::int(3));
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "10 ≥ 3 should be proved, got {status:?}"
        );
    }

    #[test]
    fn test_structural_proof_conjunction() {
        let mut prover = VCProver::new();

        // True ∧ True should be proved
        let spec = Spec::and(vec![Spec::True, Spec::True]);
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "True ∧ True should be proved, got {status:?}"
        );

        // 1 < 2 ∧ 3 < 4 should be proved
        let spec = Spec::and(vec![
            Spec::lt(Spec::int(1), Spec::int(2)),
            Spec::lt(Spec::int(3), Spec::int(4)),
        ]);
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "1 < 2 ∧ 3 < 4 should be proved, got {status:?}"
        );
    }

    #[test]
    fn test_structural_proof_disjunction() {
        let mut prover = VCProver::new();

        // True ∨ False should be proved
        let spec = Spec::or(vec![Spec::True, Spec::False]);
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "True ∨ False should be proved, got {status:?}"
        );

        // False ∨ True should be proved
        let spec = Spec::or(vec![Spec::False, Spec::True]);
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "False ∨ True should be proved, got {status:?}"
        );
    }

    #[test]
    fn test_structural_proof_implication() {
        let mut prover = VCProver::new();

        // P → True should be proved
        let spec = Spec::implies(Spec::var("P"), Spec::True);
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "P → True should be proved, got {status:?}"
        );

        // False → P should be proved
        let spec = Spec::implies(Spec::False, Spec::var("P"));
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "False → P should be proved, got {status:?}"
        );
    }

    #[test]
    fn test_structural_proof_negation() {
        let mut prover = VCProver::new();

        // ¬False should be proved
        let spec = Spec::not(Spec::False);
        let status = prover.prove_spec(&spec);
        assert!(
            matches!(status, ProofStatus::Proved(_)),
            "¬False should be proved, got {status:?}"
        );
    }

    #[test]
    fn test_try_extract_nat_overflow() {
        use lean5_kernel::{Expr, Literal};

        let prover = VCProver::new();

        // Values exceeding i64::MAX should return None
        let large_nat = Expr::Lit(Literal::Nat(u64::MAX));
        assert_eq!(prover.try_extract_nat(&large_nat), None);

        // i64::MAX should be extractable
        let max_i64 = Expr::Lit(Literal::Nat(i64::MAX as u64));
        assert_eq!(prover.try_extract_nat(&max_i64), Some(i64::MAX));

        // i64::MAX + 1 should NOT be extractable (exceeds i64)
        let overflow = Expr::Lit(Literal::Nat((i64::MAX as u64) + 1));
        assert_eq!(prover.try_extract_nat(&overflow), None);
    }

    #[test]
    fn test_try_extract_int_from_nat_literal() {
        use lean5_kernel::{Expr, Literal};

        let prover = VCProver::new();

        // Small values should work
        let small = Expr::Lit(Literal::Nat(42));
        assert_eq!(prover.try_extract_int(&small), Some(42));

        // i64::MAX should work
        let max = Expr::Lit(Literal::Nat(i64::MAX as u64));
        assert_eq!(prover.try_extract_int(&max), Some(i64::MAX));

        // Values > i64::MAX should return None
        let overflow = Expr::Lit(Literal::Nat((i64::MAX as u64) + 1));
        assert_eq!(prover.try_extract_int(&overflow), None);
    }

    #[test]
    fn test_try_extract_nat_succ_overflow() {
        use lean5_kernel::{Expr, Literal, Name};
        use std::str::FromStr;
        use std::sync::Arc;

        let prover = VCProver::new();

        // Nat.succ(i64::MAX) should overflow and return None
        let succ_name = Name::from_str("Nat.succ").unwrap();
        let max_nat = Expr::Lit(Literal::Nat(i64::MAX as u64));
        let succ_max = Expr::App(Arc::new(Expr::const_(succ_name, vec![])), Arc::new(max_nat));
        assert_eq!(
            prover.try_extract_nat(&succ_max),
            None,
            "Nat.succ(i64::MAX) should return None due to overflow"
        );
    }
}
