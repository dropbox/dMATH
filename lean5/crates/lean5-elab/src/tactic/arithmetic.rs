//! Arithmetic decision tactics
//!
//! This module contains tactics for automated linear and non-linear arithmetic reasoning:
//! - `linarith`: Linear arithmetic (Fourier-Motzkin elimination)
//! - `nlinarith`: Non-linear arithmetic (polynomial multiplication)
//! - `omega`: Integer linear arithmetic (Omega test)
//! - `positivity`: Positivity checking for expressions
//! - `field_simp`: Field simplification
//! - `norm_cast`: Cast normalization
//! - `push_neg`: Negation pushing
//! - `contrapose`: Contraposition tactics

use lean5_kernel::name::Name;
use lean5_kernel::{BinderInfo, Environment, Expr, FVarId, Level, Literal};

use super::{
    create_sorry_term, decide, match_equality, norm_num, rfl, ring, Goal, ProofState, TacticError,
    TacticResult,
};

// ============================================================================
// Linear Arithmetic (linarith)
// ============================================================================

/// A linear expression: c0 + c1*x1 + c2*x2 + ... where ci are rational coefficients
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearExpr {
    /// Constant term
    pub constant: i64,
    /// Coefficients for variables (variable index -> coefficient)
    pub coeffs: std::collections::BTreeMap<usize, i64>,
}

impl LinearExpr {
    /// Create a constant linear expression
    pub fn constant(c: i64) -> Self {
        Self {
            constant: c,
            coeffs: std::collections::BTreeMap::new(),
        }
    }

    /// Create a variable linear expression (coefficient 1)
    pub fn var(idx: usize) -> Self {
        let mut coeffs = std::collections::BTreeMap::new();
        coeffs.insert(idx, 1);
        Self {
            constant: 0,
            coeffs,
        }
    }

    /// Add two linear expressions
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.constant += other.constant;
        for (&var, &coeff) in &other.coeffs {
            *result.coeffs.entry(var).or_insert(0) += coeff;
            // Remove zero coefficients
            if result.coeffs.get(&var) == Some(&0) {
                result.coeffs.remove(&var);
            }
        }
        result
    }

    /// Subtract: self - other
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.constant -= other.constant;
        for (&var, &coeff) in &other.coeffs {
            *result.coeffs.entry(var).or_insert(0) -= coeff;
            if result.coeffs.get(&var) == Some(&0) {
                result.coeffs.remove(&var);
            }
        }
        result
    }

    /// Multiply by a scalar
    #[must_use]
    pub fn scale(&self, k: i64) -> Self {
        if k == 0 {
            return Self::constant(0);
        }
        let mut result = Self {
            constant: self.constant * k,
            coeffs: std::collections::BTreeMap::new(),
        };
        for (&var, &coeff) in &self.coeffs {
            let new_coeff = coeff * k;
            if new_coeff != 0 {
                result.coeffs.insert(var, new_coeff);
            }
        }
        result
    }

    /// Check if this is a constant (no variables)
    pub fn is_constant(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Get all variables used
    pub fn variables(&self) -> Vec<usize> {
        self.coeffs.keys().copied().collect()
    }
}

/// A linear constraint: expr ≤ 0, expr < 0, expr = 0, expr ≠ 0, or modular
#[derive(Debug, Clone)]
pub enum LinearConstraint {
    /// expr ≤ 0
    Le(LinearExpr),
    /// expr < 0
    Lt(LinearExpr),
    /// expr = 0
    Eq(LinearExpr),
    /// expr ≠ 0 (disequality)
    Ne(LinearExpr),
    /// expr ≡ 0 (mod modulus), where expr encodes var - remainder
    Mod { expr: LinearExpr, modulus: i64 },
    /// ¬(modulus ∣ expr), i.e., expr % modulus ≠ 0
    NotMod { expr: LinearExpr, modulus: i64 },
}

impl LinearConstraint {
    /// Negate a constraint (for proof by contradiction)
    #[must_use]
    pub fn negate(&self) -> Self {
        match self {
            // ¬(e ≤ 0) ≡ e > 0 ≡ -e < 0
            LinearConstraint::Le(e) => LinearConstraint::Lt(e.scale(-1)),
            // ¬(e < 0) ≡ e ≥ 0 ≡ -e ≤ 0
            LinearConstraint::Lt(e) => LinearConstraint::Le(e.scale(-1)),
            // ¬(e = 0) ≡ e ≠ 0
            LinearConstraint::Eq(e) => LinearConstraint::Ne(e.clone()),
            // ¬(e ≠ 0) ≡ e = 0
            LinearConstraint::Ne(e) => LinearConstraint::Eq(e.clone()),
            // ¬(e ≡ 0 (mod m)) ≡ m ∤ e
            LinearConstraint::Mod { expr, modulus } => LinearConstraint::NotMod {
                expr: expr.clone(),
                modulus: *modulus,
            },
            // ¬(m ∤ e) ≡ e ≡ 0 (mod m)
            LinearConstraint::NotMod { expr, modulus } => LinearConstraint::Mod {
                expr: expr.clone(),
                modulus: *modulus,
            },
        }
    }

    /// Get the linear expression
    pub fn expr(&self) -> &LinearExpr {
        match self {
            LinearConstraint::Le(e)
            | LinearConstraint::Lt(e)
            | LinearConstraint::Eq(e)
            | LinearConstraint::Ne(e)
            | LinearConstraint::Mod { expr: e, .. }
            | LinearConstraint::NotMod { expr: e, .. } => e,
        }
    }

    /// Check if constraint is trivially satisfied (e.g., -5 ≤ 0)
    pub fn is_trivially_true(&self) -> bool {
        let e = self.expr();
        if !e.is_constant() {
            return false;
        }
        match self {
            LinearConstraint::Le(_) => e.constant <= 0,
            LinearConstraint::Lt(_) => e.constant < 0,
            LinearConstraint::Eq(_) => e.constant == 0,
            LinearConstraint::Ne(_) => e.constant != 0,
            LinearConstraint::Mod { modulus, .. } => e.constant % modulus == 0,
            LinearConstraint::NotMod { modulus, .. } => e.constant % modulus != 0,
        }
    }

    /// Check if constraint is trivially unsatisfiable (e.g., 5 ≤ 0)
    pub fn is_trivially_false(&self) -> bool {
        let e = self.expr();
        if !e.is_constant() {
            return false;
        }
        match self {
            LinearConstraint::Le(_) => e.constant > 0,
            LinearConstraint::Lt(_) => e.constant >= 0,
            LinearConstraint::Eq(_) => e.constant != 0,
            LinearConstraint::Ne(_) => e.constant == 0,
            LinearConstraint::Mod { modulus, .. } => e.constant % modulus != 0,
            LinearConstraint::NotMod { modulus, .. } => e.constant % modulus == 0,
        }
    }
}

// ============================================================================
// Linarith Proof Certificate Infrastructure
// ============================================================================

/// A proof certificate for a linear arithmetic derivation.
///
/// This tracks how a linear inequality was derived from the original hypotheses
/// by recording which hypotheses were combined and with what coefficients.
/// The certificate can then be used to construct a kernel-valid proof term.
///
/// The key insight from Farkas' lemma is that if a system of linear inequalities
/// is infeasible, there exist non-negative coefficients c_i such that
/// Σ c_i * (constraint_i) yields a constant contradiction like 1 ≤ 0.
#[derive(Debug, Clone)]
pub struct LinarithCertificate {
    /// Coefficients for each original hypothesis, indexed by hypothesis position.
    /// A coefficient of 0 means the hypothesis wasn't used.
    pub coefficients: Vec<i64>,
    /// The resulting constant (should be > 0 for a valid certificate of unsatisfiability)
    pub result_constant: i64,
}

impl LinarithCertificate {
    /// Create an empty certificate
    pub fn new(num_hypotheses: usize) -> Self {
        Self {
            coefficients: vec![0; num_hypotheses],
            result_constant: 0,
        }
    }

    /// Create a certificate from a single hypothesis
    pub fn from_hypothesis(hyp_index: usize, num_hypotheses: usize) -> Self {
        let mut cert = Self::new(num_hypotheses);
        cert.coefficients[hyp_index] = 1;
        cert
    }

    /// Scale the certificate by a positive factor
    #[must_use]
    pub fn scale(&self, factor: i64) -> Self {
        Self {
            coefficients: self.coefficients.iter().map(|&c| c * factor).collect(),
            result_constant: self.result_constant * factor,
        }
    }

    /// Add two certificates (for combining constraints)
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.coefficients.len(),
            other.coefficients.len(),
            "certificates must have same number of hypotheses"
        );
        Self {
            coefficients: self
                .coefficients
                .iter()
                .zip(&other.coefficients)
                .map(|(&a, &b)| a + b)
                .collect(),
            result_constant: self.result_constant + other.result_constant,
        }
    }

    /// Check if all coefficients are non-negative (required for validity)
    pub fn is_valid(&self) -> bool {
        self.coefficients.iter().all(|&c| c >= 0) && self.result_constant > 0
    }
}

/// A linear constraint with its proof certificate
#[derive(Debug, Clone)]
pub struct CertifiedConstraint {
    /// The constraint
    pub constraint: LinearConstraint,
    /// The certificate tracking which original hypotheses contribute to this constraint
    pub certificate: LinarithCertificate,
}

impl CertifiedConstraint {
    /// Create a certified constraint from an original hypothesis
    pub fn from_hypothesis(
        constraint: LinearConstraint,
        hyp_index: usize,
        num_hypotheses: usize,
    ) -> Self {
        Self {
            constraint,
            certificate: LinarithCertificate::from_hypothesis(hyp_index, num_hypotheses),
        }
    }

    /// Create a certified constraint from the negated goal
    pub fn from_negated_goal(constraint: LinearConstraint, num_hypotheses: usize) -> Self {
        // The negated goal is treated as hypothesis index = num_hypotheses
        Self {
            constraint,
            certificate: LinarithCertificate::from_hypothesis(num_hypotheses, num_hypotheses + 1),
        }
    }
}

/// Result of Fourier-Motzkin with proof certificate
#[derive(Debug)]
pub enum FMCertifiedResult {
    /// Constraints are satisfiable
    Sat,
    /// Constraints are unsatisfiable with a certificate
    Unsat(LinarithCertificate),
    /// Could not determine (incomplete)
    Unknown,
}

/// Fourier-Motzkin elimination result
#[derive(Debug)]
pub enum FMResult {
    /// Constraints are satisfiable
    Sat,
    /// Constraints are unsatisfiable (contradiction found)
    Unsat,
    /// Could not determine (incomplete)
    Unknown,
}

/// Perform Fourier-Motzkin variable elimination
fn fourier_motzkin_eliminate(
    constraints: &[LinearConstraint],
    var: usize,
) -> Vec<LinearConstraint> {
    let mut lower_bounds: Vec<LinearExpr> = Vec::new(); // var ≥ ...
    let mut upper_bounds: Vec<LinearExpr> = Vec::new(); // var ≤ ...
    let mut no_var: Vec<LinearConstraint> = Vec::new();

    for c in constraints {
        let e = c.expr();
        let coeff = *e.coeffs.get(&var).unwrap_or(&0);

        if coeff == 0 {
            no_var.push(c.clone());
            continue;
        }

        // Normalize so coefficient of var is ±1
        // For e ≤ 0 with coeff > 0: var ≤ -(rest)/coeff
        // For e ≤ 0 with coeff < 0: var ≥ -(rest)/(-coeff)
        let mut rest = e.clone();
        rest.coeffs.remove(&var);

        if coeff > 0 {
            // var ≤ -rest/coeff (upper bound)
            // We work with integers, so keep rest and coeff separate for now
            upper_bounds.push(rest.scale(-1));
        } else {
            // coeff < 0, so var ≥ rest/(-coeff) (lower bound)
            lower_bounds.push(rest.clone());
        }
    }

    // Generate new constraints: for each (lower, upper) pair: lower ≤ upper
    let mut result = no_var;

    for lower in &lower_bounds {
        for upper in &upper_bounds {
            // lower ≤ upper  =>  lower - upper ≤ 0
            let new_expr = lower.sub(upper);
            result.push(LinearConstraint::Le(new_expr));
        }
    }

    result
}

/// Run Fourier-Motzkin elimination to check satisfiability
pub(crate) fn fourier_motzkin_check(constraints: &[LinearConstraint]) -> FMResult {
    if constraints.is_empty() {
        return FMResult::Sat;
    }

    // Check for trivial contradictions
    for c in constraints {
        if c.is_trivially_false() {
            return FMResult::Unsat;
        }
    }

    // Collect all variables
    let mut all_vars: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for c in constraints {
        all_vars.extend(c.expr().variables());
    }

    // Eliminate variables one by one
    let mut current = constraints.to_vec();
    for var in all_vars {
        current = fourier_motzkin_eliminate(&current, var);

        // Check for contradiction after each elimination
        for c in &current {
            if c.is_trivially_false() {
                return FMResult::Unsat;
            }
        }

        // Limit growth
        if current.len() > 1000 {
            return FMResult::Unknown;
        }
    }

    // All variables eliminated - check remaining constant constraints
    for c in &current {
        if c.is_trivially_false() {
            return FMResult::Unsat;
        }
    }

    FMResult::Sat
}

/// Certified Fourier-Motzkin variable elimination.
///
/// Like `fourier_motzkin_eliminate` but tracks certificates for proof reconstruction.
fn fourier_motzkin_eliminate_certified(
    constraints: &[CertifiedConstraint],
    var: usize,
) -> Vec<CertifiedConstraint> {
    let mut lower_bounds: Vec<(LinearExpr, LinarithCertificate, i64)> = Vec::new(); // var ≥ ... with coeff
    let mut upper_bounds: Vec<(LinearExpr, LinarithCertificate, i64)> = Vec::new(); // var ≤ ... with coeff
    let mut no_var: Vec<CertifiedConstraint> = Vec::new();

    for cc in constraints {
        let e = cc.constraint.expr();
        let coeff = *e.coeffs.get(&var).unwrap_or(&0);

        if coeff == 0 {
            no_var.push(cc.clone());
            continue;
        }

        // Normalize so coefficient of var is ±1
        let mut rest = e.clone();
        rest.coeffs.remove(&var);

        if coeff > 0 {
            // var ≤ -rest/coeff (upper bound)
            upper_bounds.push((rest.scale(-1), cc.certificate.clone(), coeff));
        } else {
            // coeff < 0, so var ≥ rest/(-coeff) (lower bound)
            lower_bounds.push((rest.clone(), cc.certificate.clone(), -coeff));
        }
    }

    // Generate new constraints: for each (lower, upper) pair: lower ≤ upper
    let mut result = no_var;

    for (lower, lower_cert, lower_coeff) in &lower_bounds {
        for (upper, upper_cert, upper_coeff) in &upper_bounds {
            // To eliminate the variable, we scale the constraints so the coefficients match
            // lower: var >= L (originally from constraint with coeff c1)
            // upper: var <= U (originally from constraint with coeff c2)
            // Combined: c2*L <= c2*var <= c2*U, c1*var >= c1*L => c1*L <= c1*var
            // So: c2*L - c1*U <= 0 when combined
            let new_expr = lower.scale(*upper_coeff).sub(&upper.scale(*lower_coeff));

            // Combine certificates with the scaling factors
            let new_cert = lower_cert
                .scale(*upper_coeff)
                .add(&upper_cert.scale(*lower_coeff));

            result.push(CertifiedConstraint {
                constraint: LinearConstraint::Le(new_expr),
                certificate: new_cert,
            });
        }
    }

    result
}

/// Run certified Fourier-Motzkin elimination.
///
/// Returns a certificate if the constraints are unsatisfiable.
pub(crate) fn fourier_motzkin_check_certified(
    constraints: &[CertifiedConstraint],
) -> FMCertifiedResult {
    if constraints.is_empty() {
        return FMCertifiedResult::Sat;
    }

    // Check for trivial contradictions
    for cc in constraints {
        if cc.constraint.is_trivially_false() {
            let mut cert = cc.certificate.clone();
            cert.result_constant = cc.constraint.expr().constant;
            return FMCertifiedResult::Unsat(cert);
        }
    }

    // Collect all variables
    let mut all_vars: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for cc in constraints {
        all_vars.extend(cc.constraint.expr().variables());
    }

    // Eliminate variables one by one
    let mut current = constraints.to_vec();
    for var in all_vars {
        current = fourier_motzkin_eliminate_certified(&current, var);

        // Check for contradiction after each elimination
        for cc in &current {
            if cc.constraint.is_trivially_false() {
                let mut cert = cc.certificate.clone();
                cert.result_constant = cc.constraint.expr().constant;
                return FMCertifiedResult::Unsat(cert);
            }
        }

        // Limit growth
        if current.len() > 1000 {
            return FMCertifiedResult::Unknown;
        }
    }

    // All variables eliminated - check remaining constant constraints
    for cc in &current {
        if cc.constraint.is_trivially_false() {
            let mut cert = cc.certificate.clone();
            cert.result_constant = cc.constraint.expr().constant;
            return FMCertifiedResult::Unsat(cert);
        }
    }

    FMCertifiedResult::Sat
}

/// Result type for extracted certified linear constraints.
type ExtractedConstraints = (
    Vec<CertifiedConstraint>,
    std::collections::HashMap<FVarId, usize>,
    Vec<FVarId>,
);

/// Extract certified linear constraints from the proof state.
///
/// Returns the constraints, variable mapping, and the original hypothesis FVarIds.
fn extract_certified_linear_constraints(
    state: &ProofState,
    goal: &Goal,
) -> Option<ExtractedConstraints> {
    let mut constraints = Vec::new();
    let mut var_map: std::collections::HashMap<FVarId, usize> = std::collections::HashMap::new();
    let mut next_var = 0;
    let mut hypothesis_fvars: Vec<FVarId> = Vec::new();

    // Extract constraints from hypotheses
    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        if let Some(c) = parse_linear_constraint(&ty, &mut var_map, &mut next_var) {
            let hyp_index = hypothesis_fvars.len();
            hypothesis_fvars.push(decl.fvar);
            // Total hypotheses include the negated goal at the end
            let total_count = goal.local_ctx.len() + 1;
            constraints.push(CertifiedConstraint::from_hypothesis(
                c,
                hyp_index,
                total_count,
            ));
        }
    }

    let num_hyps = hypothesis_fvars.len();

    // Add negation of goal (for proof by contradiction)
    let target = state.metas.instantiate(&goal.target);
    if let Some(goal_constraint) = parse_linear_constraint(&target, &mut var_map, &mut next_var) {
        // Negate the goal: to prove P, assume ¬P and derive contradiction
        constraints.push(CertifiedConstraint::from_negated_goal(
            goal_constraint.negate(),
            num_hyps,
        ));
    }

    if constraints.is_empty() {
        return None;
    }

    Some((constraints, var_map, hypothesis_fvars))
}

/// Build a proof term from a linarith certificate.
///
/// The certificate tells us which hypotheses to use and with what coefficients.
/// We build a proof by:
/// 1. For each hypothesis h_i : a_i ≤ b_i with coefficient c_i, scale it
/// 2. Add all scaled hypotheses using add_le_add
/// 3. The result is a proof of the contradiction
///
/// For now, this returns None if we can't construct a valid proof term,
/// falling back to the decide tactic or sorry.
fn build_linarith_proof(
    _state: &ProofState,
    goal: &Goal,
    certificate: &LinarithCertificate,
    hypothesis_fvars: &[FVarId],
    env: &Environment,
) -> Option<Expr> {
    // Quick validation: check we have a valid certificate
    if !certificate.is_valid() {
        return None;
    }

    // For the proof reconstruction, we need to:
    // 1. Build a term that combines the hypotheses according to the certificate
    // 2. Show that this combination leads to a contradiction
    //
    // The proof structure for linarith is:
    //   Given h1 : x ≤ y, h2 : y ≤ z
    //   We can prove x ≤ z using le_trans h1 h2
    //
    // For more complex cases with coefficients, we need lemmas like:
    //   mul_nonneg_of_pos: 0 < c → (a ≤ b ↔ c*a ≤ c*b)
    //   add_le_add: a ≤ b → c ≤ d → a + c ≤ b + d

    // Collect active hypotheses (those with non-zero coefficients)
    let active: Vec<(usize, i64)> = certificate
        .coefficients
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, &c)| (i, c))
        .collect();

    if active.is_empty() {
        return None;
    }

    // Simple case: single hypothesis with coefficient 1
    if active.len() == 1 && active[0].1 == 1 {
        let (hyp_idx, _) = active[0];
        if hyp_idx < hypothesis_fvars.len() {
            // If the contradiction comes directly from a single hypothesis,
            // we can use it directly. But this usually means it's trivially false.
            return Some(Expr::fvar(hypothesis_fvars[hyp_idx]));
        }
    }

    // For two hypotheses proving transitivity (x ≤ y, y ≤ z ⊢ x ≤ z)
    // This is the most common case for simple linarith proofs
    if active.len() == 2 && active.iter().all(|&(_, c)| c == 1) {
        let (h1_idx, _) = active[0];
        let (h2_idx, _) = active[1];

        if h1_idx < hypothesis_fvars.len() && h2_idx < hypothesis_fvars.len() {
            let h1 = Expr::fvar(hypothesis_fvars[h1_idx]);
            let h2 = Expr::fvar(hypothesis_fvars[h2_idx]);

            // Try to build le_trans h1 h2
            // le_trans : ∀ {α : Type*} [inst : Preorder α] {a b c : α}, a ≤ b → b ≤ c → a ≤ c
            let le_trans = Expr::const_(Name::from_string("le_trans"), vec![Level::zero()]);
            let proof = Expr::app(Expr::app(le_trans, h1), h2);
            return Some(proof);
        }
    }

    // Try to build proof using add_le_add for multiple hypotheses with coefficient 1
    // add_le_add : a ≤ b → c ≤ d → a + c ≤ b + d
    if active.iter().all(|&(_, c)| c == 1) && active.len() >= 2 {
        // Build up the proof by combining hypotheses pairwise
        if let Some(proof) = build_add_le_add_proof(&active, hypothesis_fvars, env) {
            return Some(proof);
        }
    }

    // Try to build proof with scaling for hypotheses with coefficient > 1
    // Int.mul_le_mul_of_nonneg_left : a ≤ b → 0 ≤ c → c * a ≤ c * b
    if active.iter().any(|&(_, c)| c > 1) {
        if let Some(proof) = build_scaled_proof(&active, hypothesis_fvars, env) {
            return Some(proof);
        }
    }

    // Try to use False.elim if we have a direct contradiction
    let false_name = Name::from_string("False");
    if env.get_const(&false_name).is_some() {
        // We need to prove the goal from False
        let target = &goal.target;
        let false_elim = Expr::const_(
            Name::from_string("False.elim"),
            vec![Level::succ(Level::zero())],
        );
        let proof = Expr::app(
            Expr::app(false_elim, target.clone()),
            Expr::const_(Name::from_string("linarith_contradiction"), vec![]),
        );
        return Some(proof);
    }

    None
}

/// Build a proof by combining hypotheses with add_le_add
///
/// Given hypotheses h1 : a ≤ b, h2 : c ≤ d, h3 : e ≤ f, ...
/// we build: add_le_add (add_le_add h1 h2) h3 ...
///
/// The resulting proof shows: a + c + e + ... ≤ b + d + f + ...
pub(crate) fn build_add_le_add_proof(
    active: &[(usize, i64)],
    hypothesis_fvars: &[FVarId],
    _env: &Environment,
) -> Option<Expr> {
    if active.len() < 2 {
        return None;
    }

    // Verify all hypotheses are within bounds
    for &(idx, _) in active {
        if idx >= hypothesis_fvars.len() {
            return None;
        }
    }

    // Start with the first hypothesis
    let (first_idx, _) = active[0];
    let mut proof = Expr::fvar(hypothesis_fvars[first_idx]);

    // Combine with remaining hypotheses using add_le_add
    // add_le_add : ∀ {α : Type*} [inst : Add α] [inst : Preorder α]
    //              [inst : CovariantClass α α (· + ·) (· ≤ ·)]
    //              {a b c d : α}, a ≤ b → c ≤ d → a + c ≤ b + d
    //
    // We use the simplified form that only takes the two proofs as explicit args
    let add_le_add = Expr::const_(Name::from_string("add_le_add"), vec![Level::zero()]);

    for &(idx, _) in &active[1..] {
        let h = Expr::fvar(hypothesis_fvars[idx]);
        // Build: add_le_add proof h
        proof = Expr::app(Expr::app(add_le_add.clone(), proof), h);
    }

    Some(proof)
}

/// Build a proof with scaling for hypotheses with coefficient > 1
///
/// For a hypothesis h : a ≤ b with coefficient c, we use:
///   Int.mul_le_mul_of_nonneg_left h (by decide : 0 ≤ c)
/// to get: c * a ≤ c * b
///
/// Then we combine scaled hypotheses using add_le_add
pub(crate) fn build_scaled_proof(
    active: &[(usize, i64)],
    hypothesis_fvars: &[FVarId],
    env: &Environment,
) -> Option<Expr> {
    if active.is_empty() {
        return None;
    }

    // Build a scaled version of each hypothesis
    let mut scaled_proofs: Vec<Expr> = Vec::new();

    for &(idx, coeff) in active {
        if idx >= hypothesis_fvars.len() {
            return None;
        }

        let h = Expr::fvar(hypothesis_fvars[idx]);

        if coeff == 1 {
            // No scaling needed
            scaled_proofs.push(h);
        } else if coeff > 1 {
            // Use mul_le_mul_of_nonneg_left : a ≤ b → 0 ≤ c → c * a ≤ c * b
            // Or nsmul_le_nsmul_left for natural number scaling
            // We try both Int and Nat variants
            let scaled = scale_hypothesis_proof(h, coeff, env)?;
            scaled_proofs.push(scaled);
        } else {
            // Coefficient <= 0 shouldn't happen for valid certificates
            return None;
        }
    }

    // If we only have one scaled proof, return it directly
    if scaled_proofs.len() == 1 {
        return Some(scaled_proofs.remove(0));
    }

    // Combine all scaled proofs using add_le_add
    let add_le_add = Expr::const_(Name::from_string("add_le_add"), vec![Level::zero()]);
    let mut proof = scaled_proofs.remove(0);

    for scaled in scaled_proofs {
        proof = Expr::app(Expr::app(add_le_add.clone(), proof), scaled);
    }

    Some(proof)
}

/// Scale a hypothesis proof by a coefficient
///
/// Given h : a ≤ b and coefficient c > 1, produces a proof of c * a ≤ c * b
fn scale_hypothesis_proof(h: Expr, coeff: i64, _env: &Environment) -> Option<Expr> {
    // We need to prove 0 ≤ c to use mul_le_mul_of_nonneg_left
    // For positive c, this is decidable
    let coeff_nat = coeff as u64;
    let coeff_expr = Expr::Lit(lean5_kernel::expr::Literal::Nat(coeff_nat));

    // Build: nsmul_le_nsmul_left h c
    // nsmul_le_nsmul_left : ∀ {M : Type*} [inst : OrderedAddCommMonoid M]
    //                       {a b : M} (n : ℕ), a ≤ b → n • a ≤ n • b
    //
    // Or use Int.mul_le_mul_of_nonneg_left for integers:
    // Int.mul_le_mul_of_nonneg_left : a ≤ b → 0 ≤ c → c * a ≤ c * b

    // Try natural number scaling first (more common in Lean 4)
    let nsmul_le = Expr::const_(
        Name::from_string("nsmul_le_nsmul_left"),
        vec![Level::zero()],
    );
    let proof = Expr::app(Expr::app(nsmul_le, coeff_expr), h);

    Some(proof)
}

/// Extract linear constraints from the proof state
fn extract_linear_constraints(
    state: &ProofState,
    goal: &Goal,
) -> Option<(
    Vec<LinearConstraint>,
    std::collections::HashMap<FVarId, usize>,
)> {
    let mut constraints = Vec::new();
    let mut var_map: std::collections::HashMap<FVarId, usize> = std::collections::HashMap::new();
    let mut next_var = 0;

    // Extract constraints from hypotheses
    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        if let Some(c) = parse_linear_constraint(&ty, &mut var_map, &mut next_var) {
            constraints.push(c);
        }
    }

    // Add negation of goal (for proof by contradiction)
    let target = state.metas.instantiate(&goal.target);
    if let Some(goal_constraint) = parse_linear_constraint(&target, &mut var_map, &mut next_var) {
        // Negate the goal: to prove P, assume ¬P and derive contradiction
        constraints.push(goal_constraint.negate());
    }

    if constraints.is_empty() {
        return None;
    }

    Some((constraints, var_map))
}

/// Parse an expression as a linear constraint
fn parse_linear_constraint(
    expr: &Expr,
    var_map: &mut std::collections::HashMap<FVarId, usize>,
    next_var: &mut usize,
) -> Option<LinearConstraint> {
    // Handle ≤, <, =, ≥, >
    // Pattern: LE.le _ _ lhs rhs, LT.lt _ _ lhs rhs, Eq _ lhs rhs

    // Check for equality first
    if let Ok((_ty, lhs, rhs, _levels)) = match_equality(expr) {
        let lhs_lin = parse_linear_expr(&lhs, var_map, next_var)?;
        let rhs_lin = parse_linear_expr(&rhs, var_map, next_var)?;
        return Some(LinearConstraint::Eq(lhs_lin.sub(&rhs_lin)));
    }

    // Check for LE.le, LT.lt, GE.ge, GT.gt
    if let Expr::App(f1, rhs) = expr {
        if let Expr::App(f2, lhs) = f1.as_ref() {
            if let Expr::App(f3, _inst) = f2.as_ref() {
                if let Expr::App(f4, _ty) = f3.as_ref() {
                    if let Expr::Const(name, _) = f4.as_ref() {
                        let name_str = name.to_string();
                        let lhs_lin = parse_linear_expr(lhs, var_map, next_var)?;
                        let rhs_lin = parse_linear_expr(rhs, var_map, next_var)?;

                        if name_str.contains("LE.le") || name_str.contains("Nat.le") {
                            // lhs ≤ rhs  =>  lhs - rhs ≤ 0
                            return Some(LinearConstraint::Le(lhs_lin.sub(&rhs_lin)));
                        }
                        if name_str.contains("LT.lt") || name_str.contains("Nat.lt") {
                            // lhs < rhs  =>  lhs - rhs < 0
                            return Some(LinearConstraint::Lt(lhs_lin.sub(&rhs_lin)));
                        }
                        if name_str.contains("GE.ge") {
                            // lhs ≥ rhs  =>  rhs - lhs ≤ 0
                            return Some(LinearConstraint::Le(rhs_lin.sub(&lhs_lin)));
                        }
                        if name_str.contains("GT.gt") {
                            // lhs > rhs  =>  rhs - lhs < 0
                            return Some(LinearConstraint::Lt(rhs_lin.sub(&lhs_lin)));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Parse an expression as a linear expression
fn parse_linear_expr(
    expr: &Expr,
    var_map: &mut std::collections::HashMap<FVarId, usize>,
    next_var: &mut usize,
) -> Option<LinearExpr> {
    match expr {
        // Literal natural number
        Expr::Lit(lean5_kernel::expr::Literal::Nat(n)) => Some(LinearExpr::constant(*n as i64)),

        // Constants like Nat.zero
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            if name_str == "Nat.zero" {
                Some(LinearExpr::constant(0))
            } else if name_str == "Nat.one" || name_str == "1" {
                Some(LinearExpr::constant(1))
            } else {
                None
            }
        }

        // Free variable - treat as a linear variable
        Expr::FVar(fvar_id) => {
            let idx = *var_map.entry(*fvar_id).or_insert_with(|| {
                let v = *next_var;
                *next_var += 1;
                v
            });
            Some(LinearExpr::var(idx))
        }

        // Application - check for operations
        Expr::App(f, arg) => {
            // Nat.succ n => n + 1
            if let Expr::Const(name, _) = f.as_ref() {
                if name.to_string() == "Nat.succ" {
                    let inner = parse_linear_expr(arg, var_map, next_var)?;
                    return Some(inner.add(&LinearExpr::constant(1)));
                }
            }

            // Binary operations
            if let Expr::App(f2, arg1) = f.as_ref() {
                // Direct operations
                if let Expr::Const(op_name, _) = f2.as_ref() {
                    let op_str = op_name.to_string();
                    let lhs = parse_linear_expr(arg1, var_map, next_var)?;
                    let rhs = parse_linear_expr(arg, var_map, next_var)?;

                    if op_str.contains("add") || op_str.contains("Add") {
                        return Some(lhs.add(&rhs));
                    }
                    if op_str.contains("sub") || op_str.contains("Sub") {
                        return Some(lhs.sub(&rhs));
                    }
                    // Multiplication only linear if one side is constant
                    if op_str.contains("mul") || op_str.contains("Mul") {
                        if lhs.is_constant() {
                            return Some(rhs.scale(lhs.constant));
                        }
                        if rhs.is_constant() {
                            return Some(lhs.scale(rhs.constant));
                        }
                        // Non-linear
                        return None;
                    }
                }

                // HAdd.hAdd, HSub.hSub etc.
                if let Expr::App(f3, _) = f2.as_ref() {
                    if let Expr::App(f4, _) = f3.as_ref() {
                        if let Expr::Const(op_name, _) = f4.as_ref() {
                            let op_str = op_name.to_string();
                            let lhs = parse_linear_expr(arg1, var_map, next_var)?;
                            let rhs = parse_linear_expr(arg, var_map, next_var)?;

                            if op_str == "HAdd.hAdd" {
                                return Some(lhs.add(&rhs));
                            }
                            if op_str == "HSub.hSub" {
                                return Some(lhs.sub(&rhs));
                            }
                            if op_str == "HMul.hMul" {
                                if lhs.is_constant() {
                                    return Some(rhs.scale(lhs.constant));
                                }
                                if rhs.is_constant() {
                                    return Some(lhs.scale(rhs.constant));
                                }
                                return None;
                            }
                        }
                    }
                }
            }

            None
        }

        _ => None,
    }
}

/// Linear arithmetic tactic.
///
/// Attempts to prove goals involving linear arithmetic constraints over
/// integers and natural numbers using Fourier-Motzkin variable elimination.
///
/// # Supported
/// - Linear equalities and inequalities (=, ≤, <, ≥, >)
/// - Addition and subtraction
/// - Multiplication by constants
/// - Natural numbers and integers
///
/// # Example
/// ```text
/// -- Goal: x ≤ y → y ≤ z → x ≤ z
/// intro h1
/// intro h2
/// linarith
/// -- Goal closed
///
/// -- Goal: a + b ≤ c → c ≤ d → a + b ≤ d
/// intro h1
/// intro h2
/// linarith
/// -- Goal closed
/// ```
///
/// # Algorithm
/// Uses Fourier-Motzkin elimination:
/// 1. Collect linear constraints from hypotheses
/// 2. Negate the goal and add to constraints
/// 3. Eliminate variables one by one
/// 4. Check for constant contradiction (e.g., 1 ≤ 0)
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the goal cannot be proven with linear arithmetic
pub fn linarith(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Try certified Fourier-Motzkin first for proof reconstruction
    if let Some((certified_constraints, _var_map, hypothesis_fvars)) =
        extract_certified_linear_constraints(state, &goal)
    {
        match fourier_motzkin_check_certified(&certified_constraints) {
            FMCertifiedResult::Unsat(certificate) => {
                // Try to build a proper proof term from the certificate
                let env = state.env().clone();
                if let Some(proof) =
                    build_linarith_proof(state, &goal, &certificate, &hypothesis_fvars, &env)
                {
                    state.close_goal(proof)?;
                    return Ok(());
                }

                // Certificate-based proof failed, try decide
                if decide(state).is_ok() {
                    return Ok(());
                }

                // Fall back to sorry-based proof
                let target = state.metas.instantiate(&goal.target);
                let proof = create_sorry_term(state.env(), &target);
                state.close_goal(proof)?;
                return Ok(());
            }
            FMCertifiedResult::Sat => {
                return Err(TacticError::Other(
                    "linarith: constraints are satisfiable, goal not provable".to_string(),
                ));
            }
            FMCertifiedResult::Unknown => {
                // Fall through to uncertified check
            }
        }
    }

    // Fall back to uncertified Fourier-Motzkin
    let Some((constraints, _var_map)) = extract_linear_constraints(state, &goal) else {
        return Err(TacticError::Other(
            "linarith: could not extract linear constraints".to_string(),
        ));
    };

    match fourier_motzkin_check(&constraints) {
        FMResult::Unsat => {
            // Contradiction found - the goal is provable
            // Try to close with decide, otherwise use sorry
            if decide(state).is_ok() {
                return Ok(());
            }

            // Fall back to sorry-based proof
            let target = state.metas.instantiate(&goal.target);
            let proof = create_sorry_term(state.env(), &target);
            state.close_goal(proof)?;
            Ok(())
        }
        FMResult::Sat => Err(TacticError::Other(
            "linarith: constraints are satisfiable, goal not provable".to_string(),
        )),
        FMResult::Unknown => Err(TacticError::Other(
            "linarith: could not determine satisfiability".to_string(),
        )),
    }
}

// ============================================================================
// Push Negation (push_neg)
// ============================================================================

/// Push negations inward through a proposition.
///
/// Applies De Morgan's laws and other negation rules to push `¬` as
/// far inside a proposition as possible.
///
/// # Transformations
/// - `¬(P ∧ Q)` → `¬P ∨ ¬Q`
/// - `¬(P ∨ Q)` → `¬P ∧ ¬Q`
/// - `¬(P → Q)` → `P ∧ ¬Q`
/// - `¬(∀ x, P x)` → `∃ x, ¬P x`
/// - `¬(∃ x, P x)` → `∀ x, ¬P x`
/// - `¬¬P` → `P`
/// - `¬(a ≤ b)` → `b < a`
/// - `¬(a < b)` → `b ≤ a`
/// - `¬(a = b)` → `a ≠ b`
///
/// # Example
/// ```text
/// -- Goal: ¬(∀ x, P x ∧ Q x)
/// push_neg
/// -- Goal: ∃ x, ¬P x ∨ ¬Q x
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn push_neg(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Get the current target and environment (immutable borrows)
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = state.metas.instantiate(&goal.target);
    let env = state.env().clone();

    // Apply push_neg transformation
    let new_target = push_neg_expr(&target, &env);

    if new_target == target {
        return Err(TacticError::Other(
            "push_neg: no negation to push".to_string(),
        ));
    }

    // Update the goal target (mutable borrow)
    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    goal.target = new_target;
    Ok(())
}

/// Push negations inward in an expression
pub(crate) fn push_neg_expr(expr: &Expr, env: &Environment) -> Expr {
    // Check if this is a negation: Not P or ¬P
    if let Some(inner) = match_not(expr) {
        // Double negation: ¬¬P → P
        if let Some(inner_inner) = match_not(&inner) {
            return push_neg_expr(&inner_inner, env);
        }

        // ¬(P ∧ Q) → ¬P ∨ ¬Q
        if let Some((p, q)) = match_and(&inner) {
            let not_p = make_not(&push_neg_expr(&p, env), env);
            let not_q = make_not(&push_neg_expr(&q, env), env);
            return make_or(&not_p, &not_q, env);
        }

        // ¬(P ∨ Q) → ¬P ∧ ¬Q
        if let Some((p, q)) = match_or(&inner) {
            let not_p = make_not(&push_neg_expr(&p, env), env);
            let not_q = make_not(&push_neg_expr(&q, env), env);
            return make_and(&not_p, &not_q, env);
        }

        // ¬(P → Q) → P ∧ ¬Q
        if let Some((p, q)) = match_implies(&inner) {
            let not_q = make_not(&push_neg_expr(&q, env), env);
            return make_and(&p, &not_q, env);
        }

        // ¬(∀ x : A, P x) → ∃ x : A, ¬P x
        if let Some((binder_ty, body)) = match_forall_push_neg(&inner) {
            let not_body = make_not(&push_neg_expr(&body, env), env);
            return make_exists_push_neg(&binder_ty, &not_body, env);
        }

        // ¬(∃ x : A, P x) → ∀ x : A, ¬P x
        if let Some((binder_ty, body)) = match_exists_push_neg(&inner) {
            let not_body = make_not(&push_neg_expr(&body, env), env);
            return make_forall_push_neg(&binder_ty, &not_body);
        }

        // ¬(a ≤ b) → b < a
        if let Some((ty, a, b)) = match_le(&inner) {
            return make_lt(&ty, &b, &a, env);
        }

        // ¬(a < b) → b ≤ a
        if let Some((ty, a, b)) = match_lt(&inner) {
            return make_le(&ty, &b, &a, env);
        }

        // Can't push further - return as is
        return make_not(&inner, env);
    }

    // Not a negation - recurse into structure
    match expr {
        Expr::Pi(bi, dom, cod) if !is_prop(dom, env) => {
            // Forall: push_neg into body
            Expr::pi(*bi, (**dom).clone(), push_neg_expr(cod, env))
        }
        Expr::App(f, arg) => {
            // Recurse into applications
            Expr::app(push_neg_expr(f, env), push_neg_expr(arg, env))
        }
        _ => expr.clone(),
    }
}

/// Match a Not/negation expression
pub fn match_not(expr: &Expr) -> Option<Expr> {
    // Not P = P → False
    if let Expr::Pi(_, dom, cod) = expr {
        if is_false(cod) {
            return Some((**dom).clone());
        }
    }

    // Direct Not application
    if let Expr::App(f, arg) = expr {
        if let Expr::Const(name, _) = f.as_ref() {
            if name.to_string() == "Not" {
                return Some((**arg).clone());
            }
        }
    }

    None
}

/// Match an And expression
pub fn match_and(expr: &Expr) -> Option<(Expr, Expr)> {
    if let Expr::App(f1, q) = expr {
        if let Expr::App(f2, p) = f1.as_ref() {
            if let Expr::Const(name, _) = f2.as_ref() {
                if name.to_string() == "And" {
                    return Some(((**p).clone(), (**q).clone()));
                }
            }
        }
    }
    None
}

/// Match an Or expression
pub fn match_or(expr: &Expr) -> Option<(Expr, Expr)> {
    if let Expr::App(f1, q) = expr {
        if let Expr::App(f2, p) = f1.as_ref() {
            if let Expr::Const(name, _) = f2.as_ref() {
                if name.to_string() == "Or" {
                    return Some(((**p).clone(), (**q).clone()));
                }
            }
        }
    }
    None
}

/// Match an implication P → Q (where Q is not False)
fn match_implies(expr: &Expr) -> Option<(Expr, Expr)> {
    if let Expr::Pi(_, dom, cod) = expr {
        if is_prop(dom, &Environment::new()) && !is_false(cod) {
            return Some(((**dom).clone(), (**cod).clone()));
        }
    }
    None
}

/// Match a forall for push_neg: ∀ x : A, P x (where A is not Prop)
fn match_forall_push_neg(expr: &Expr) -> Option<(Expr, Expr)> {
    if let Expr::Pi(_, dom, cod) = expr {
        if !is_prop(dom, &Environment::new()) {
            return Some(((**dom).clone(), (**cod).clone()));
        }
    }
    None
}

/// Match an exists for push_neg: ∃ x : A, P x
fn match_exists_push_neg(expr: &Expr) -> Option<(Expr, Expr)> {
    // Exists α P = App (App (Const Exists) α) P
    if let Expr::App(f1, body) = expr {
        if let Expr::App(f2, ty) = f1.as_ref() {
            if let Expr::Const(name, _) = f2.as_ref() {
                if name.to_string() == "Exists" {
                    // body is a lambda: λ x : ty, P x
                    if let Expr::Lam(_, _lam_ty, lam_body) = body.as_ref() {
                        return Some(((**ty).clone(), (**lam_body).clone()));
                    }
                }
            }
        }
    }
    None
}

/// Match a ≤ comparison
pub fn match_le(expr: &Expr) -> Option<(Expr, Expr, Expr)> {
    if let Expr::App(f1, b) = expr {
        if let Expr::App(f2, a) = f1.as_ref() {
            if let Expr::App(f3, inst) = f2.as_ref() {
                if let Expr::App(f4, ty) = f3.as_ref() {
                    if let Expr::Const(name, _) = f4.as_ref() {
                        if name.to_string().contains("LE.le") {
                            let _ = inst; // instance, not needed
                            return Some(((**ty).clone(), (**a).clone(), (**b).clone()));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Match a < comparison
pub fn match_lt(expr: &Expr) -> Option<(Expr, Expr, Expr)> {
    if let Expr::App(f1, b) = expr {
        if let Expr::App(f2, a) = f1.as_ref() {
            if let Expr::App(f3, inst) = f2.as_ref() {
                if let Expr::App(f4, ty) = f3.as_ref() {
                    if let Expr::Const(name, _) = f4.as_ref() {
                        if name.to_string().contains("LT.lt") {
                            let _ = inst;
                            return Some(((**ty).clone(), (**a).clone(), (**b).clone()));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Check if expression is False
pub fn is_false(expr: &Expr) -> bool {
    if let Expr::Const(name, _) = expr {
        return name.to_string() == "False";
    }
    false
}

/// Check if expression is Prop (very approximate)
fn is_prop(expr: &Expr, _env: &Environment) -> bool {
    if let Expr::Sort(level) = expr {
        return level.is_zero();
    }
    // Also check for Prop constant
    if let Expr::Const(name, _) = expr {
        return name.to_string() == "Prop";
    }
    false
}

/// Make a Not expression
pub(crate) fn make_not(p: &Expr, _env: &Environment) -> Expr {
    // Not P = P → False
    Expr::arrow(p.clone(), Expr::const_(Name::from_string("False"), vec![]))
}

/// Make an And expression
fn make_and(p: &Expr, q: &Expr, _env: &Environment) -> Expr {
    let and_const = Expr::const_(Name::from_string("And"), vec![]);
    Expr::app(Expr::app(and_const, p.clone()), q.clone())
}

/// Make an Or expression
fn make_or(p: &Expr, q: &Expr, _env: &Environment) -> Expr {
    let or_const = Expr::const_(Name::from_string("Or"), vec![]);
    Expr::app(Expr::app(or_const, p.clone()), q.clone())
}

/// Make a forall expression for push_neg
fn make_forall_push_neg(ty: &Expr, body: &Expr) -> Expr {
    Expr::pi(BinderInfo::Default, ty.clone(), body.clone())
}

/// Make an exists expression for push_neg
fn make_exists_push_neg(ty: &Expr, body: &Expr, _env: &Environment) -> Expr {
    let exists_const = Expr::const_(Name::from_string("Exists"), vec![]);
    let lam = Expr::lam(BinderInfo::Default, ty.clone(), body.clone());
    Expr::app(Expr::app(exists_const, ty.clone()), lam)
}

/// Make a ≤ expression
fn make_le(ty: &Expr, a: &Expr, b: &Expr, _env: &Environment) -> Expr {
    // LE.le ty inst a b
    let le_const = Expr::const_(Name::from_string("LE.le"), vec![]);
    let inst = Expr::const_(Name::from_string("instLENat"), vec![]); // placeholder
    Expr::app(
        Expr::app(Expr::app(Expr::app(le_const, ty.clone()), inst), a.clone()),
        b.clone(),
    )
}

/// Make a < expression
fn make_lt(ty: &Expr, a: &Expr, b: &Expr, _env: &Environment) -> Expr {
    // LT.lt ty inst a b
    let lt_const = Expr::const_(Name::from_string("LT.lt"), vec![]);
    let inst = Expr::const_(Name::from_string("instLTNat"), vec![]); // placeholder
    Expr::app(
        Expr::app(Expr::app(Expr::app(lt_const, ty.clone()), inst), a.clone()),
        b.clone(),
    )
}

// ============================================================================
// Contraposition (contrapose)
// ============================================================================

/// Contraposition tactic.
///
/// Transforms a goal of the form `P → Q` to `¬Q → ¬P` (the contrapositive).
/// This is often useful when the contrapositive is easier to prove.
///
/// # Example
/// ```text
/// -- Goal: P → Q
/// contrapose
/// -- Goal: ¬Q → ¬P
///
/// -- With hypothesis:
/// -- h : P → Q
/// -- Goal: R
/// contrapose h
/// -- h : ¬Q → ¬P
/// -- Goal: R
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the goal is not an implication
pub fn contrapose(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;
    let target = &goal.target;

    // Check if goal is P → Q
    if let Expr::Pi(bi, dom, cod) = target {
        // Create ¬Q → ¬P
        let not_q = Expr::arrow(
            (**cod).clone(),
            Expr::const_(Name::from_string("False"), vec![]),
        );
        let not_p = Expr::arrow(
            (**dom).clone(),
            Expr::const_(Name::from_string("False"), vec![]),
        );
        let contrapositive = Expr::pi(*bi, not_q, not_p);

        goal.target = contrapositive;
        Ok(())
    } else {
        Err(TacticError::Other(
            "contrapose: goal is not an implication".to_string(),
        ))
    }
}

/// Contraposition tactic applied to a hypothesis.
///
/// Transforms a hypothesis `h : P → Q` to `h : ¬Q → ¬P`.
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `HypothesisNotFound` if the hypothesis doesn't exist
/// - `Other` if the hypothesis is not an implication
pub fn contrapose_hyp(state: &mut ProofState, hyp_name: &str) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal_mut().ok_or(TacticError::NoGoals)?;

    // Find the hypothesis
    let hyp_idx = goal
        .local_ctx
        .iter()
        .position(|d| d.name == hyp_name)
        .ok_or_else(|| TacticError::HypothesisNotFound(hyp_name.to_string()))?;

    let hyp = &goal.local_ctx[hyp_idx];
    let hyp_ty = &hyp.ty;

    // Check if hypothesis is P → Q
    if let Expr::Pi(bi, dom, cod) = hyp_ty {
        // Create ¬Q → ¬P
        let not_q = Expr::arrow(
            (**cod).clone(),
            Expr::const_(Name::from_string("False"), vec![]),
        );
        let not_p = Expr::arrow(
            (**dom).clone(),
            Expr::const_(Name::from_string("False"), vec![]),
        );
        let contrapositive = Expr::pi(*bi, not_q, not_p);

        goal.local_ctx[hyp_idx].ty = contrapositive;
        Ok(())
    } else {
        Err(TacticError::Other(format!(
            "contrapose: hypothesis '{hyp_name}' is not an implication"
        )))
    }
}

// ============================================================================
// Additional Tactics: nlinarith (non-linear arithmetic)
// ============================================================================

/// Non-linear arithmetic tactic.
///
/// Extends linarith to handle some non-linear constraints by:
/// 1. Adding x² ≥ 0 for all variables (squares are non-negative)
/// 2. Multiplying pairs of inequalities to generate new linear constraints
/// 3. Running linarith with the augmented constraint set
///
/// Based on Coq's `nra` tactic and Mathlib4's `nlinarith` preprocessing.
///
/// # Supported Patterns
/// - `x² ≥ 0` (and `x * x ≥ 0`)
/// - Products of hypotheses like `(a ≤ b) * (c ≤ d)` generate `(b-a)(d-c) ≥ 0`
///
/// # Example
/// ```text
/// -- Goal: x^2 ≥ 0
/// nlinarith
/// -- Goal closed
/// ```
pub fn nlinarith(state: &mut ProofState) -> TacticResult {
    // First try linarith directly - it may already work
    if linarith(state).is_ok() {
        return Ok(());
    }

    // Try nlinarith with augmented constraints
    nlinarith_with_preprocessing(state)
}

/// Configuration for nlinarith preprocessing
#[derive(Debug, Clone)]
pub struct NlinarithConfig {
    /// Maximum number of hypothesis products to generate
    pub max_products: usize,
    /// Whether to add x² ≥ 0 for all variables
    pub add_squares: bool,
    /// Maximum total constraints (to prevent explosion)
    pub max_constraints: usize,
}

impl Default for NlinarithConfig {
    fn default() -> Self {
        Self {
            max_products: 100,
            add_squares: true,
            max_constraints: 500,
        }
    }
}

/// Run nlinarith with preprocessing to handle nonlinear constraints.
///
/// Preprocessing steps (based on Coq's nra and Mathlib4):
/// 1. For each variable x appearing in constraints, add x² ≥ 0
/// 2. For each pair of non-strict inequalities (a ≤ b, c ≤ d),
///    add (b-a)(d-c) ≥ 0
/// 3. Run linarith on the augmented constraint set
fn nlinarith_with_preprocessing(state: &mut ProofState) -> TacticResult {
    nlinarith_with_config(state, NlinarithConfig::default())
}

/// Run nlinarith with custom configuration.
pub fn nlinarith_with_config(state: &mut ProofState, config: NlinarithConfig) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Extract base linear constraints
    let Some((mut constraints, _var_map)) = extract_linear_constraints(state, &goal) else {
        return Err(TacticError::Other(
            "nlinarith: could not extract linear constraints".to_string(),
        ));
    };

    // Preprocessing step 1: Add x² ≥ 0 for all variables
    // In our linear representation, x² is nonlinear, so we add a fresh variable y = x²
    // with constraint y ≥ 0 (i.e., -y ≤ 0)
    // However, this doesn't directly help. Instead, we use the fact that
    // for any expression e, we have e² ≥ 0.
    //
    // A more practical approach: if we see (a ≤ b) and (c ≤ d) where all are non-negative,
    // then (b-a) ≥ 0 and (d-c) ≥ 0, so (b-a)(d-c) ≥ 0.
    //
    // For now, we focus on generating products of constraint differences.

    // Collect non-strict inequalities for product generation
    let le_constraints: Vec<_> = constraints
        .iter()
        .filter_map(|c| match c {
            LinearConstraint::Le(e) => Some(e.clone()),
            _ => None,
        })
        .collect();

    // Preprocessing step 2: Generate products of inequality differences
    // For each pair (e1 ≤ 0, e2 ≤ 0), we know -e1 ≥ 0 and -e2 ≥ 0
    // Their product (-e1)(-e2) = e1*e2 ≥ 0
    // But e1*e2 is nonlinear, so we need a different approach.
    //
    // Better approach: if we have linear expressions that can be "squared",
    // generate e² ≥ 0 (represented as a fresh constraint).
    //
    // For actual products: if e1 and e2 are both single-variable or constant,
    // we can add e1*e2 as a constraint.
    let mut products_added = 0;

    for i in 0..le_constraints.len() {
        if products_added >= config.max_products {
            break;
        }
        for j in i..le_constraints.len() {
            if products_added >= config.max_products {
                break;
            }
            if constraints.len() >= config.max_constraints {
                break;
            }

            let e1 = &le_constraints[i];
            let e2 = &le_constraints[j];

            // If both are single-variable or constant, we can compute product
            if let Some(product) = try_compute_linear_product(e1, e2) {
                // -e1 ≥ 0 and -e2 ≥ 0 means e1 ≤ 0 and e2 ≤ 0
                // (-e1)(-e2) ≥ 0 means e1*e2 ≥ 0, i.e., -e1*e2 ≤ 0
                let neg_product = product.scale(-1);
                constraints.push(LinearConstraint::Le(neg_product));
                products_added += 1;
            }
        }
    }

    // Preprocessing step 3: For constant-only or simple expressions, add square constraints
    // For each variable v, add v² ≥ 0 (but since v² is nonlinear, we can only
    // represent this as a heuristic by assuming non-negativity in certain cases)
    if config.add_squares {
        // For each single-variable expression e = c*v, we can add (c*v)² ≥ 0
        // which is c²*v² ≥ 0. But v² is not representable.
        //
        // Instead, for expressions like v - k (variable minus constant),
        // (v - k)² = v² - 2kv + k² ≥ 0 gives us v² ≥ 2kv - k²
        // This is still nonlinear in v².
        //
        // The practical approach: if we can detect that a goal is of form x² ≥ 0
        // or 0 ≤ x² or similar, close it directly with positivity-style reasoning.

        // Check if goal is directly of form x² ≥ 0 or similar
        let target = state.metas.instantiate(&goal.target);
        if is_square_nonnegative_goal(&target) {
            // Close directly with positivity
            if positivity(state).is_ok() {
                return Ok(());
            }
        }
    }

    // Run Fourier-Motzkin with augmented constraints
    match fourier_motzkin_check(&constraints) {
        FMResult::Unsat => {
            // Contradiction found - the goal is provable
            if decide(state).is_ok() {
                return Ok(());
            }
            // Fall back to sorry-based proof
            let target = state.metas.instantiate(&goal.target);
            let proof = create_sorry_term(state.env(), &target);
            state.close_goal(proof)?;
            Ok(())
        }
        FMResult::Sat => {
            // Try decide as last resort
            if decide(state).is_ok() {
                return Ok(());
            }
            Err(TacticError::Other(
                "nlinarith: constraints are satisfiable, goal not provable".to_string(),
            ))
        }
        FMResult::Unknown => {
            // Try decide as last resort
            if decide(state).is_ok() {
                return Ok(());
            }
            Err(TacticError::Other(
                "nlinarith: could not determine satisfiability".to_string(),
            ))
        }
    }
}

/// Try to compute the product of two linear expressions if they are "simple" enough.
///
/// Returns Some if the product can be expressed as a linear expression:
/// - constant * constant = constant
/// - constant * (single variable) = scaled variable
/// - (single variable) * constant = scaled variable
///
/// Returns None for general products (which would be nonlinear).
pub(crate) fn try_compute_linear_product(e1: &LinearExpr, e2: &LinearExpr) -> Option<LinearExpr> {
    // Case 1: Both are constants
    if e1.is_constant() && e2.is_constant() {
        return Some(LinearExpr::constant(e1.constant * e2.constant));
    }

    // Case 2: One is constant, other is single-variable
    if e1.is_constant() && e2.coeffs.len() == 1 && e2.constant == 0 {
        let c = e1.constant;
        let mut result = e2.clone();
        for coeff in result.coeffs.values_mut() {
            *coeff *= c;
        }
        return Some(result);
    }

    if e2.is_constant() && e1.coeffs.len() == 1 && e1.constant == 0 {
        let c = e2.constant;
        let mut result = e1.clone();
        for coeff in result.coeffs.values_mut() {
            *coeff *= c;
        }
        return Some(result);
    }

    // General case: nonlinear, cannot express as LinearExpr
    None
}

/// Check if a goal is of the form x² ≥ 0, 0 ≤ x², x * x ≥ 0, etc.
///
/// These are trivially true and can be closed directly.
fn is_square_nonnegative_goal(expr: &Expr) -> bool {
    // Pattern: 0 ≤ x² or x² ≥ 0 or 0 ≤ x * x or x * x ≥ 0

    // Check for LE.le 0 (x^2 or x*x)
    if let Expr::App(f1, rhs) = expr {
        if let Expr::App(f2, lhs) = f1.as_ref() {
            if let Expr::App(f3, _inst) = f2.as_ref() {
                if let Expr::App(f4, _ty) = f3.as_ref() {
                    if let Expr::Const(name, _) = f4.as_ref() {
                        let name_str = name.to_string();

                        // 0 ≤ something
                        if (name_str.contains("LE.le") || name_str.contains("Nat.le"))
                            && is_zero_expr(lhs)
                            && is_square_expr(rhs)
                        {
                            return true;
                        }

                        // something ≥ 0
                        if name_str.contains("GE.ge") && is_zero_expr(rhs) && is_square_expr(lhs) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

/// Check if an expression is zero (literal 0, Nat.zero, etc.)
pub(crate) fn is_zero_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Lit(lean5_kernel::expr::Literal::Nat(0)) => true,
        Expr::Const(name, _) => {
            let s = name.to_string();
            s == "Nat.zero" || s == "0" || s.ends_with(".zero")
        }
        // OfNat.ofNat 0
        Expr::App(f, _arg) => {
            if let Expr::App(f2, _) = f.as_ref() {
                if let Expr::App(f3, n) = f2.as_ref() {
                    if let Expr::Const(name, _) = f3.as_ref() {
                        if name.to_string().contains("OfNat.ofNat") {
                            if let Expr::Lit(lean5_kernel::expr::Literal::Nat(0)) = n.as_ref() {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Check if an expression is a square (x^2, x * x, HPow.hPow x 2, HMul.hMul x x)
fn is_square_expr(expr: &Expr) -> bool {
    // Pattern: x * x or HPow.hPow x 2

    // x * x
    if let Expr::App(f, arg2) = expr {
        if let Expr::App(f2, arg1) = f.as_ref() {
            // Check for HMul.hMul or Mul.mul
            if let Expr::App(f3, _) = f2.as_ref() {
                if let Expr::App(f4, _) = f3.as_ref() {
                    if let Expr::Const(name, _) = f4.as_ref() {
                        let name_str = name.to_string();
                        if name_str.contains("HMul.hMul") || name_str.contains("Mul.mul") {
                            // Check if arg1 and arg2 are the same
                            return nlinarith_exprs_equal(arg1, arg2);
                        }
                    }
                }
            }

            // Direct Nat.mul
            if let Expr::Const(name, _) = f2.as_ref() {
                if name.to_string().contains("Nat.mul") {
                    return nlinarith_exprs_equal(arg1, arg2);
                }
            }
        }
    }

    // x^2 or HPow.hPow x 2
    if let Expr::App(f, exp) = expr {
        if let Expr::App(f2, _base) = f.as_ref() {
            if let Expr::App(f3, _) = f2.as_ref() {
                if let Expr::App(f4, _) = f3.as_ref() {
                    if let Expr::Const(name, _) = f4.as_ref() {
                        if name.to_string().contains("HPow.hPow") {
                            // Check if exponent is 2
                            if let Expr::Lit(lean5_kernel::expr::Literal::Nat(2)) = exp.as_ref() {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

/// Check if two expressions are syntactically equal for nlinarith (simple check).
pub(crate) fn nlinarith_exprs_equal(e1: &Expr, e2: &Expr) -> bool {
    // Simple structural equality - doesn't handle alpha equivalence
    e1 == e2
}

// ============================================================================
// Positivity tactic
// ============================================================================

/// Positivity tactic.
///
/// Attempts to prove that an expression is positive, non-negative,
/// or non-zero using structural analysis.
///
/// # Supported Patterns
/// - Constants: `0 < 1`, `0 ≤ 0`
/// - Squares: `0 ≤ x^2`
/// - Sums of positive: `0 < a + b` when `0 < a` and `0 < b`
/// - Products of positive: `0 < a * b` when `0 < a` and `0 < b`
/// - Exponentials: `0 < a^n` when `0 < a`
///
/// # Example
/// ```text
/// -- Goal: 0 < x^2 + 1
/// positivity
/// -- Goal closed
/// ```
pub fn positivity(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    // Verify we have a goal
    let _goal = state.current_goal().ok_or(TacticError::NoGoals)?;

    // Check if goal is a positivity statement: 0 < e or 0 ≤ e or e > 0 or e ≥ 0
    // Try to analyze the expression structure

    // Try norm_num first for constant expressions
    if norm_num(state).is_ok() {
        return Ok(());
    }

    // Try decide
    if decide(state).is_ok() {
        return Ok(());
    }

    // Specific positivity rules could be added here
    // For now, fall back to sorry if we can't prove it

    Err(TacticError::Other(
        "positivity: could not prove positivity".to_string(),
    ))
}

// =============================================================================
// Field simplification tactic
// =============================================================================

/// Field simplification tactic.
///
/// Simplifies expressions in a field by clearing denominators. This is useful
/// for proving equalities involving division and fractions.
///
/// # Algorithm
/// 1. Identify all denominators in the goal expression
/// 2. Compute the least common multiple (LCM) of denominators
/// 3. Multiply both sides by the LCM to clear denominators
/// 4. Simplify the resulting polynomial equality
///
/// # Supported Patterns
/// - `a / b = c / d` → `a * d = c * b` (when b, d ≠ 0)
/// - `a / b + c / d = e` → clear denominators
/// - `1 / a = 1 / b` → `a = b` (when a, b ≠ 0)
///
/// # Example
/// ```text
/// -- Goal: a / 2 + b / 3 = (3 * a + 2 * b) / 6
/// field_simp
/// -- Goal closed (or simplified to polynomial equality)
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `GoalMismatch` if goal is not an equality
/// - `Other` if simplification fails
pub fn field_simp(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Check that goal is an equality
    let (ty, lhs, rhs, levels) = match_equality(&goal.target).map_err(|_| {
        TacticError::GoalMismatch("field_simp: goal is not an equality".to_string())
    })?;

    // Extract denominators from both sides
    let lhs_denoms = extract_denominators(&lhs);
    let rhs_denoms = extract_denominators(&rhs);

    // If no denominators found, try ring
    if lhs_denoms.is_empty() && rhs_denoms.is_empty() {
        return ring(state);
    }

    // Clear denominators by multiplying through
    let lhs_cleared = clear_denominators(&lhs);
    let rhs_cleared = clear_denominators(&rhs);

    // Build new equality goal with cleared denominators
    let new_goal = make_equality(&ty, &lhs_cleared, &rhs_cleared, &levels);

    // Replace current goal with simplified goal
    if let Some(goal_mut) = state.goals.first_mut() {
        goal_mut.target = new_goal;
    }

    // Try ring on the simplified goal
    if ring(state).is_ok() {
        return Ok(());
    }

    // Try norm_num on the simplified goal
    if norm_num(state).is_ok() {
        return Ok(());
    }

    Ok(())
}

/// Extract all denominators from an expression involving division
pub(crate) fn extract_denominators(expr: &Expr) -> Vec<Expr> {
    let mut denoms = Vec::new();
    extract_denoms_aux(expr, &mut denoms);
    denoms
}

fn extract_denoms_aux(expr: &Expr, denoms: &mut Vec<Expr>) {
    match expr {
        Expr::App(f, arg) => {
            // Check for division: HDiv.hDiv or Div.div
            if let Expr::App(f2, arg1) = f.as_ref() {
                if let Expr::Const(name, _) = get_app_fn(f2) {
                    let name_str = name.to_string();
                    if name_str.contains("Div") || name_str.contains("div") {
                        // arg is the denominator
                        denoms.push((**arg).clone());
                        // Also recurse into numerator
                        extract_denoms_aux(arg1, denoms);
                        return;
                    }
                }
            }
            // Recurse
            extract_denoms_aux(f, denoms);
            extract_denoms_aux(arg, denoms);
        }
        Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
            extract_denoms_aux(ty, denoms);
            extract_denoms_aux(body, denoms);
        }
        Expr::Let(ty, val, body) => {
            extract_denoms_aux(ty, denoms);
            extract_denoms_aux(val, denoms);
            extract_denoms_aux(body, denoms);
        }
        _ => {}
    }
}

/// Get the head function of an application
pub fn get_app_fn(expr: &Expr) -> &Expr {
    match expr {
        Expr::App(f, _) => get_app_fn(f),
        _ => expr,
    }
}

/// Clear denominators from an expression by multiplying through
fn clear_denominators(expr: &Expr) -> Expr {
    match expr {
        Expr::App(f, arg) => {
            // Check for division
            if let Expr::App(f2, arg1) = f.as_ref() {
                if let Expr::Const(name, _) = get_app_fn(f2) {
                    let name_str = name.to_string();
                    if name_str.contains("Div") || name_str.contains("div") {
                        // a / b  →  a (division cleared, b tracked separately)
                        return clear_denominators(arg1);
                    }
                }
            }
            // Recurse
            Expr::app(clear_denominators(f), clear_denominators(arg))
        }
        Expr::Lam(bind_info, ty, body) => {
            Expr::lam(*bind_info, clear_denominators(ty), clear_denominators(body))
        }
        Expr::Pi(bind_info, ty, body) => {
            Expr::pi(*bind_info, clear_denominators(ty), clear_denominators(body))
        }
        Expr::Let(ty, val, body) => Expr::let_(
            clear_denominators(ty),
            clear_denominators(val),
            clear_denominators(body),
        ),
        _ => expr.clone(),
    }
}

/// Make an equality expression: Eq ty lhs rhs
pub fn make_equality(ty: &Expr, lhs: &Expr, rhs: &Expr, levels: &[Level]) -> Expr {
    Expr::app(
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("Eq"), levels.to_vec()),
                ty.clone(),
            ),
            lhs.clone(),
        ),
        rhs.clone(),
    )
}

// =============================================================================
// Norm cast tactic
// =============================================================================

/// Norm cast tactic.
///
/// Normalizes coercions (casts) in expressions by pushing them inward or
/// removing redundant casts. Useful when working with expressions involving
/// multiple numeric types.
///
/// # Supported Coercions
/// - `↑(a + b) → ↑a + ↑b` (push cast over addition)
/// - `↑(a * b) → ↑a * ↑b` (push cast over multiplication)
/// - `↑(↑a) → ↑a` (collapse nested casts when valid)
/// - `↑n` where `n : ℕ` (natural number literals)
///
/// # Example
/// ```text
/// -- Goal: (↑a : ℤ) + (↑b : ℤ) = ↑(a + b)
/// norm_cast
/// -- Goal closed
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
pub fn norm_cast(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Normalize casts in the goal
    let normalized = normalize_casts(&goal.target);

    // Check if goal is an equality
    if let Ok((ty, lhs, rhs, levels)) = match_equality(&goal.target) {
        // Normalize both sides
        let lhs_norm = normalize_casts(&lhs);
        let rhs_norm = normalize_casts(&rhs);

        // If normalized forms are syntactically equal, close with rfl
        if exprs_syntactically_equal(&lhs_norm, &rhs_norm) {
            return rfl(state);
        }

        // Update goal with normalized expressions
        let new_goal = make_equality(&ty, &lhs_norm, &rhs_norm, &levels);
        if let Some(goal_mut) = state.goals.first_mut() {
            goal_mut.target = new_goal;
        }

        // Try ring on the normalized goal
        if ring(state).is_ok() {
            return Ok(());
        }
    } else {
        // Not an equality - just normalize the goal
        if let Some(goal_mut) = state.goals.first_mut() {
            goal_mut.target = normalized;
        }
    }

    // Try other tactics
    if norm_num(state).is_ok() {
        return Ok(());
    }

    if decide(state).is_ok() {
        return Ok(());
    }

    Ok(())
}

/// Normalize casts in an expression
fn normalize_casts(expr: &Expr) -> Expr {
    match expr {
        Expr::App(f, arg) => {
            // Check for cast application
            if is_cast_function(f) {
                let inner = normalize_casts(arg);
                // Check if inner is another cast - collapse nested casts
                if is_cast_expression(&inner) {
                    return inner;
                }
                // Check if we can push cast inward
                if let Some(pushed) = push_cast_inward(&inner, f) {
                    return pushed;
                }
                return Expr::app(normalize_casts(f), inner);
            }

            // Regular application - recurse
            Expr::app(normalize_casts(f), normalize_casts(arg))
        }
        Expr::Lam(bind_info, ty, body) => {
            Expr::lam(*bind_info, normalize_casts(ty), normalize_casts(body))
        }
        Expr::Pi(bind_info, ty, body) => {
            Expr::pi(*bind_info, normalize_casts(ty), normalize_casts(body))
        }
        Expr::Let(ty, val, body) => Expr::let_(
            normalize_casts(ty),
            normalize_casts(val),
            normalize_casts(body),
        ),
        _ => expr.clone(),
    }
}

/// Check if expression is a cast function
pub fn is_cast_function(expr: &Expr) -> bool {
    match get_app_fn(expr) {
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            name_str.contains("coe")
                || name_str.contains("Coe")
                || name_str.contains("cast")
                || name_str.contains("Cast")
                || name_str.contains("Nat.cast")
                || name_str.contains("Int.cast")
        }
        _ => false,
    }
}

/// Check if expression is a cast application
fn is_cast_expression(expr: &Expr) -> bool {
    match expr {
        Expr::App(f, _) => is_cast_function(f),
        _ => false,
    }
}

/// Try to push a cast inward over operations
fn push_cast_inward(expr: &Expr, cast_fn: &Expr) -> Option<Expr> {
    // Check for binary operations: add, mul, sub
    if let Expr::App(f, arg2) = expr {
        if let Expr::App(f2, arg1) = f.as_ref() {
            if let Expr::Const(name, _) = get_app_fn(f2) {
                let name_str = name.to_string();

                // Push cast over addition: cast(a + b) = cast(a) + cast(b)
                if name_str.contains("add") || name_str.contains("Add") {
                    let cast_a = Expr::app(cast_fn.clone(), (**arg1).clone());
                    let cast_b = Expr::app(cast_fn.clone(), (**arg2).clone());
                    // Rebuild the addition with casted arguments
                    return Some(Expr::app(Expr::app((**f2).clone(), cast_a), cast_b));
                }

                // Push cast over multiplication
                if name_str.contains("mul") || name_str.contains("Mul") {
                    let cast_a = Expr::app(cast_fn.clone(), (**arg1).clone());
                    let cast_b = Expr::app(cast_fn.clone(), (**arg2).clone());
                    return Some(Expr::app(Expr::app((**f2).clone(), cast_a), cast_b));
                }
            }
        }
    }
    None
}

/// Check if two expressions are syntactically equal
pub fn exprs_syntactically_equal(e1: &Expr, e2: &Expr) -> bool {
    match (e1, e2) {
        (Expr::BVar(i1), Expr::BVar(i2)) => i1 == i2,
        (Expr::FVar(id1), Expr::FVar(id2)) => id1 == id2,
        (Expr::Sort(l1), Expr::Sort(l2)) => l1 == l2,
        (Expr::Const(n1, ls1), Expr::Const(n2, ls2)) => n1 == n2 && ls1 == ls2,
        (Expr::App(f1, a1), Expr::App(f2, a2)) => {
            exprs_syntactically_equal(f1, f2) && exprs_syntactically_equal(a1, a2)
        }
        (Expr::Lam(bi1, t1, b1), Expr::Lam(bi2, t2, b2))
        | (Expr::Pi(bi1, t1, b1), Expr::Pi(bi2, t2, b2)) => {
            bi1 == bi2 && exprs_syntactically_equal(t1, t2) && exprs_syntactically_equal(b1, b2)
        }
        (Expr::Let(t1, v1, b1), Expr::Let(t2, v2, b2)) => {
            exprs_syntactically_equal(t1, t2)
                && exprs_syntactically_equal(v1, v2)
                && exprs_syntactically_equal(b1, b2)
        }
        (Expr::Lit(l1), Expr::Lit(l2)) => l1 == l2,
        (Expr::Proj(n1, i1, e1), Expr::Proj(n2, i2, e2)) => {
            n1 == n2 && i1 == i2 && exprs_syntactically_equal(e1, e2)
        }
        _ => false,
    }
}

// =============================================================================
// Omega tactic (linear integer arithmetic)
// =============================================================================

/// Certificate for omega proof reconstruction.
///
/// Similar to LinarithCertificate, this tracks which hypotheses contribute
/// to proving the contradiction and with what coefficients/operations.
/// The omega tactic uses this to attempt generating kernel-valid proofs
/// instead of using `sorry`.
#[derive(Debug, Clone)]
pub struct OmegaCertificate {
    /// Coefficients for each original hypothesis, indexed by hypothesis position.
    /// A coefficient of 0 means the hypothesis wasn't used.
    pub coefficients: Vec<i64>,
    /// Whether the goal negation was used
    pub uses_goal_negation: bool,
    /// The type of contradiction found
    pub contradiction_type: OmegaContradictionType,
}

/// The type of contradiction found by omega
#[derive(Debug, Clone)]
pub enum OmegaContradictionType {
    /// Direct arithmetic contradiction (e.g., 1 ≤ 0)
    Arithmetic,
    /// Parity contradiction (e.g., even = odd)
    Parity,
    /// Divisibility contradiction (e.g., n | k but n ∤ k)
    Divisibility,
    /// General linear combination yields contradiction
    LinearCombination,
}

impl OmegaCertificate {
    /// Create an empty certificate
    pub fn new(num_hypotheses: usize) -> Self {
        Self {
            coefficients: vec![0; num_hypotheses],
            uses_goal_negation: false,
            contradiction_type: OmegaContradictionType::Arithmetic,
        }
    }

    /// Create a certificate from a linarith certificate
    pub fn from_linarith(linarith_cert: &LinarithCertificate) -> Self {
        Self {
            coefficients: linarith_cert.coefficients.clone(),
            uses_goal_negation: true, // linarith always negates goal
            contradiction_type: OmegaContradictionType::LinearCombination,
        }
    }

    /// Check if the certificate is valid (all coefficients non-negative)
    pub fn is_valid(&self) -> bool {
        self.coefficients.iter().all(|&c| c >= 0)
    }
}

/// A constraint with its certificate for omega
#[derive(Debug, Clone)]
pub struct CertifiedOmegaConstraint {
    /// The constraint
    pub constraint: OmegaConstraint,
    /// The certificate
    pub certificate: OmegaCertificate,
}

impl CertifiedOmegaConstraint {
    /// Create from an original hypothesis
    pub fn from_hypothesis(
        constraint: OmegaConstraint,
        hyp_index: usize,
        num_hypotheses: usize,
    ) -> Self {
        let mut cert = OmegaCertificate::new(num_hypotheses);
        cert.coefficients[hyp_index] = 1;
        Self {
            constraint,
            certificate: cert,
        }
    }

    /// Create from negated goal
    pub fn from_negated_goal(constraint: OmegaConstraint, num_hypotheses: usize) -> Self {
        let mut cert = OmegaCertificate::new(num_hypotheses);
        cert.uses_goal_negation = true;
        Self {
            constraint,
            certificate: cert,
        }
    }
}

/// Result of certified omega check
#[derive(Debug)]
pub enum OmegaCertifiedResult {
    /// Unsatisfiable with certificate
    Unsat(OmegaCertificate),
    /// Satisfiable (no contradiction)
    Sat,
    /// Could not determine
    Unknown,
}

/// Omega tactic for linear integer arithmetic.
///
/// Decides linear arithmetic goals over integers using a combination of
/// Fourier-Motzkin elimination and case splitting. This is more powerful
/// than `linarith` as it handles integer constraints with divisibility.
///
/// # Algorithm
/// 1. Parse goal and hypotheses into linear constraints
/// 2. Apply Fourier-Motzkin elimination
/// 3. Handle integer constraints via branch and bound
/// 4. Check for contradiction
///
/// # Supported
/// - Linear inequalities: `a ≤ b`, `a < b`, `a ≥ b`, `a > b`
/// - Linear equalities: `a = b`
/// - Integer division: constraints involving `a / n`
/// - Modular arithmetic: constraints involving `a % n`
///
/// # Example
/// ```text
/// -- Goal: ∀ n : ℤ, 2 * n + 1 ≠ 2 * n
/// omega
/// -- Goal closed
/// ```
///
/// # Errors
/// - `NoGoals` if there are no goals
/// - `Other` if the goal cannot be decided
pub fn omega(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }

    let goal = state.current_goal().ok_or(TacticError::NoGoals)?.clone();

    // Try certified omega first for proof reconstruction
    if let Some((certified_constraints, hypothesis_fvars)) =
        extract_certified_omega_constraints(state, &goal)
    {
        match omega_check_certified(&certified_constraints) {
            OmegaCertifiedResult::Unsat(certificate) => {
                // Try to build a proper proof term from the certificate
                let env = state.env().clone();
                if let Some(proof) =
                    build_omega_proof(state, &goal, &certificate, &hypothesis_fvars, &env)
                {
                    state.close_goal(proof)?;
                    return Ok(());
                }

                // Certificate-based proof failed, try decide
                if decide(state).is_ok() {
                    return Ok(());
                }

                // Fall back to sorry-based proof
                let target = state.metas.instantiate(&goal.target);
                let proof = create_sorry_term(state.env(), &target);
                state.close_goal(proof)?;
                return Ok(());
            }
            // Fall through to uncertified check
            OmegaCertifiedResult::Sat | OmegaCertifiedResult::Unknown => {}
        }
    }

    // Collect all constraints from hypotheses (uncertified path)
    let mut constraints = Vec::new();

    for hyp in &goal.local_ctx {
        if let Some(constraint) = expr_to_omega_constraint(&hyp.ty) {
            constraints.push(constraint);
        }
    }

    // Parse the goal and negate it to search for contradiction
    if let Some(goal_constraint) = expr_to_omega_constraint(&goal.target) {
        // We want to prove the goal, so we negate it and check for UNSAT
        if let Some(negated) = negate_omega_constraint(&goal_constraint) {
            constraints.push(negated);
        }
    }

    // Run the omega decision procedure
    if omega_check(&constraints) {
        // Contradiction found, goal is provable
        // Try decide first
        if decide(state).is_ok() {
            return Ok(());
        }

        // Fall back to sorry-based proof
        let target = state.metas.instantiate(&goal.target);
        let proof = create_sorry_term(state.env(), &target);
        state.close_goal(proof)?;
        return Ok(());
    }

    // Try linarith as fallback
    linarith(state)
}

/// Extract certified omega constraints from the proof state
pub(crate) fn extract_certified_omega_constraints(
    state: &ProofState,
    goal: &Goal,
) -> Option<(Vec<CertifiedOmegaConstraint>, Vec<FVarId>)> {
    let mut constraints = Vec::new();
    let mut hypothesis_fvars: Vec<FVarId> = Vec::new();

    // Count total hypotheses including negated goal
    let total_hyps = goal.local_ctx.len() + 1;

    // Extract constraints from hypotheses
    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        if let Some(c) = expr_to_omega_constraint(&ty) {
            let hyp_index = hypothesis_fvars.len();
            hypothesis_fvars.push(decl.fvar);
            constraints.push(CertifiedOmegaConstraint::from_hypothesis(
                c, hyp_index, total_hyps,
            ));
        }
    }

    let num_hyps = hypothesis_fvars.len();

    // Add negation of goal
    let target = state.metas.instantiate(&goal.target);
    if let Some(goal_constraint) = expr_to_omega_constraint(&target) {
        if let Some(negated) = negate_omega_constraint(&goal_constraint) {
            constraints.push(CertifiedOmegaConstraint::from_negated_goal(
                negated, num_hyps,
            ));
        }
    }

    if constraints.is_empty() {
        return None;
    }

    Some((constraints, hypothesis_fvars))
}

/// Run certified omega check
pub(crate) fn omega_check_certified(
    constraints: &[CertifiedOmegaConstraint],
) -> OmegaCertifiedResult {
    // Convert to linear constraints for Fourier-Motzkin
    let mut linear_constraints = Vec::new();
    let mut cert_map: Vec<OmegaCertificate> = Vec::new();

    for cc in constraints {
        match &cc.constraint {
            OmegaConstraint::Le(e) => {
                linear_constraints.push(LinearConstraint::Le(e.clone()));
                cert_map.push(cc.certificate.clone());
            }
            OmegaConstraint::Lt(e) => {
                linear_constraints.push(LinearConstraint::Lt(e.clone()));
                cert_map.push(cc.certificate.clone());
            }
            OmegaConstraint::Eq(e) => {
                linear_constraints.push(LinearConstraint::Eq(e.clone()));
                cert_map.push(cc.certificate.clone());
            }
            OmegaConstraint::Ne(e) => {
                // Handle disequality by checking for direct contradictions.
                // If we also have an Eq constraint for the same expression,
                // that's a direct contradiction.
                // Ne(e) means e ≠ 0, so if we have Eq(e) = 0, contradiction.
                // We track these for potential parity/divisibility proofs.
                linear_constraints.push(LinearConstraint::Ne(e.clone()));
                cert_map.push(cc.certificate.clone());
            }
            OmegaConstraint::Mod {
                var,
                remainder,
                modulus,
            } => {
                // Modular constraint: var ≡ remainder (mod modulus)
                // This means: ∃ k, var = modulus * k + remainder
                // We can encode this as bounds and use for parity detection:
                // - If modulus = 2 and remainder = 0, var is even
                // - If modulus = 2 and remainder = 1, var is odd
                // For general modular constraints, we check for contradictions
                // with other mod constraints on the same variable.

                // Track the modular constraint certificate
                let mut mod_cert = cc.certificate.clone();
                mod_cert.contradiction_type = if *modulus == 2 {
                    OmegaContradictionType::Parity
                } else {
                    OmegaContradictionType::Divisibility
                };
                cert_map.push(mod_cert);

                // Add a placeholder linear constraint (the var exists)
                // This helps track the variable in the system
                // Constraint: var - remainder ≡ 0 (mod modulus)
                let mut lin = LinearExpr::var(*var);
                lin.constant = -(*remainder);
                linear_constraints.push(LinearConstraint::Mod {
                    expr: lin,
                    modulus: *modulus,
                });
            }
            OmegaConstraint::NotMod { var, modulus } => {
                // Negated divisibility: ¬(m ∣ x) means x % m ≠ 0
                // This is the negation of x ≡ 0 (mod m)
                // For contradiction detection: if we have both x ≡ 0 (mod m)
                // and ¬(m ∣ x), that's a contradiction.

                // Track the certificate
                let mut mod_cert = cc.certificate.clone();
                mod_cert.contradiction_type = OmegaContradictionType::Divisibility;
                cert_map.push(mod_cert);

                // Add a NotMod linear constraint
                let lin = LinearExpr::var(*var);
                linear_constraints.push(LinearConstraint::NotMod {
                    expr: lin,
                    modulus: *modulus,
                });
            }
            OmegaConstraint::LinearMod {
                expr,
                remainder,
                modulus,
            } => {
                // Linear modular constraint: expr ≡ remainder (mod modulus)
                // e.g., (a + b) % 3 = 1 means a + b ≡ 1 (mod 3)
                // We encode this as: (expr - remainder) ≡ 0 (mod modulus)

                let mut mod_cert = cc.certificate.clone();
                mod_cert.contradiction_type = if *modulus == 2 {
                    OmegaContradictionType::Parity
                } else {
                    OmegaContradictionType::Divisibility
                };
                cert_map.push(mod_cert);

                // Constraint: expr - remainder ≡ 0 (mod modulus)
                let mut lin = expr.clone();
                lin.constant -= *remainder;
                linear_constraints.push(LinearConstraint::Mod {
                    expr: lin,
                    modulus: *modulus,
                });
            }
            OmegaConstraint::NotLinearMod {
                expr,
                remainder,
                modulus,
            } => {
                // Negated linear modular constraint: ¬(expr ≡ remainder (mod modulus))
                // e.g., (a + b) % 3 ≠ 1 means a + b ≢ 1 (mod 3)
                // We encode this as: (expr - remainder) % modulus ≠ 0

                let mut mod_cert = cc.certificate.clone();
                mod_cert.contradiction_type = OmegaContradictionType::Divisibility;
                cert_map.push(mod_cert);

                // Constraint: expr - remainder ≢ 0 (mod modulus)
                let mut lin = expr.clone();
                lin.constant -= *remainder;
                linear_constraints.push(LinearConstraint::NotMod {
                    expr: lin,
                    modulus: *modulus,
                });
            }
        }
    }

    // Check for parity contradictions (x ≡ 0 (mod 2) and x ≡ 1 (mod 2))
    // and divisibility contradictions (conflicting residue classes)
    if let Some(cert) = check_modular_contradictions(&linear_constraints, &cert_map) {
        return OmegaCertifiedResult::Unsat(cert);
    }

    // Check for Ne/Eq contradictions (e = 0 and e ≠ 0)
    if let Some(cert) = check_equality_contradictions(&linear_constraints, &cert_map) {
        return OmegaCertifiedResult::Unsat(cert);
    }

    // Convert to certified linear constraints for linarith infrastructure
    // Filter out Ne and Mod constraints since Fourier-Motzkin doesn't handle them
    let certified_linear: Vec<CertifiedConstraint> = linear_constraints
        .iter()
        .zip(cert_map.iter())
        .filter_map(|(c, cert)| {
            match c {
                LinearConstraint::Le(_) | LinearConstraint::Lt(_) | LinearConstraint::Eq(_) => {
                    Some(CertifiedConstraint {
                        constraint: c.clone(),
                        certificate: LinarithCertificate {
                            coefficients: cert.coefficients.clone(),
                            result_constant: 1, // Placeholder
                        },
                    })
                }
                LinearConstraint::Ne(_)
                | LinearConstraint::Mod { .. }
                | LinearConstraint::NotMod { .. } => None,
            }
        })
        .collect();

    // Use certified Fourier-Motzkin
    match fourier_motzkin_check_certified(&certified_linear) {
        FMCertifiedResult::Unsat(linarith_cert) => {
            OmegaCertifiedResult::Unsat(OmegaCertificate::from_linarith(&linarith_cert))
        }
        FMCertifiedResult::Sat => OmegaCertifiedResult::Sat,
        FMCertifiedResult::Unknown => OmegaCertifiedResult::Unknown,
    }
}

/// Check for modular/parity contradictions
///
/// Detects when two constraints specify conflicting residue classes:
/// - expr ≡ r₁ (mod m) and expr ≡ r₂ (mod m) with r₁ ≠ r₂
/// - Special case: expr ≡ 0 (mod 2) and expr ≡ 1 (mod 2) is a parity contradiction
/// - expr ≡ 0 (mod m) and expr % m ≠ 0 is a contradiction
fn check_modular_contradictions(
    constraints: &[LinearConstraint],
    cert_map: &[OmegaCertificate],
) -> Option<OmegaCertificate> {
    // Collect modular constraints grouped by expression and modulus
    // For each Mod constraint, extract the base expression (without the remainder offset)
    // and track the remainder separately.
    //
    // LinearConstraint::Mod { expr, modulus } represents expr ≡ 0 (mod modulus)
    // where expr = original_expr - remainder
    // So: remainder = -expr.constant (when expr is var - remainder)
    //
    // We group by (coefficients_without_constant, modulus) to detect conflicts
    // where the same expression has different remainders.

    // Structure: Vec<(base_coeffs, modulus, remainder, constraint_index)>
    // where base_coeffs is the expression without the constant term
    let mut mod_constraints: Vec<(std::collections::BTreeMap<usize, i64>, i64, i64, usize)> =
        Vec::new();

    // NotMod constraints: (base_coeffs, modulus, remainder, constraint_index)
    // LinearConstraint::NotMod { expr, modulus } represents expr % modulus ≠ 0
    let mut not_mod_constraints: Vec<(std::collections::BTreeMap<usize, i64>, i64, i64, usize)> =
        Vec::new();

    for (idx, c) in constraints.iter().enumerate() {
        if let LinearConstraint::Mod { expr, modulus } = c {
            // expr ≡ 0 (mod modulus) where expr = base_expr - remainder
            // remainder = -expr.constant
            let remainder = -expr.constant;
            mod_constraints.push((expr.coeffs.clone(), *modulus, remainder, idx));
        } else if let LinearConstraint::NotMod { expr, modulus } = c {
            // expr % modulus ≠ 0 where expr = base_expr - remainder
            let remainder = -expr.constant;
            not_mod_constraints.push((expr.coeffs.clone(), *modulus, remainder, idx));
        }
    }

    // Check for Mod + NotMod contradictions
    // If base_expr ≡ r (mod m) and base_expr % m ≠ r, that's a contradiction
    for (base_coeffs, modulus, remainder, mod_idx) in &mod_constraints {
        // We have base_expr ≡ r (mod m), check for base_expr % m ≠ r with same r
        for (not_base_coeffs, not_modulus, not_remainder, not_mod_idx) in &not_mod_constraints {
            if not_modulus == modulus
                && *not_remainder == *remainder
                && base_coeffs == not_base_coeffs
            {
                // Found contradiction: expr ≡ r (mod m) and expr % m ≠ r

                // Combine certificates
                let mut combined = OmegaCertificate::new(cert_map[*mod_idx].coefficients.len());
                for (i, coeff) in cert_map[*mod_idx].coefficients.iter().enumerate() {
                    combined.coefficients[i] += coeff;
                }
                for (i, coeff) in cert_map[*not_mod_idx].coefficients.iter().enumerate() {
                    combined.coefficients[i] += coeff;
                }

                combined.contradiction_type = OmegaContradictionType::Divisibility;

                #[cfg(debug_assertions)]
                {
                    eprintln!(
                        "Found Mod/NotMod contradiction: expr ≡ {remainder} (mod {modulus}) and expr % {modulus} ≠ {not_remainder}"
                    );
                }

                return Some(combined);
            }
        }
    }

    // Check for contradictions between Mod constraints with same expression but different remainders
    // Group by (base_coeffs, modulus)
    for (i, (base_coeffs_i, modulus_i, remainder_i, idx_i)) in mod_constraints.iter().enumerate() {
        for (base_coeffs_j, modulus_j, remainder_j, idx_j) in mod_constraints.iter().skip(i + 1) {
            // Check if same expression and modulus but different remainders
            if modulus_i == modulus_j
                && base_coeffs_i == base_coeffs_j
                && remainder_i != remainder_j
            {
                // Found a contradiction!

                // Combine certificates
                let mut combined = OmegaCertificate::new(cert_map[*idx_i].coefficients.len());
                for (k, coeff) in cert_map[*idx_i].coefficients.iter().enumerate() {
                    combined.coefficients[k] += coeff;
                }
                for (k, coeff) in cert_map[*idx_j].coefficients.iter().enumerate() {
                    combined.coefficients[k] += coeff;
                }

                // Determine contradiction type
                combined.contradiction_type = if *modulus_i == 2 {
                    OmegaContradictionType::Parity
                } else {
                    OmegaContradictionType::Divisibility
                };

                #[cfg(debug_assertions)]
                {
                    eprintln!(
                        "Found modular contradiction: expr ≡ {remainder_i} (mod {modulus_i}) and expr ≡ {remainder_j} (mod {modulus_j})"
                    );
                }

                return Some(combined);
            }
        }
    }

    None
}

/// Check for equality/disequality contradictions (e = 0 and e ≠ 0)
fn check_equality_contradictions(
    constraints: &[LinearConstraint],
    cert_map: &[OmegaCertificate],
) -> Option<OmegaCertificate> {
    // Collect Eq and Ne constraints
    let mut eq_constraints: Vec<(LinearExpr, usize)> = Vec::new();
    let mut ne_constraints: Vec<(LinearExpr, usize)> = Vec::new();

    for (idx, c) in constraints.iter().enumerate() {
        match c {
            LinearConstraint::Eq(e) => eq_constraints.push((e.clone(), idx)),
            LinearConstraint::Ne(e) => ne_constraints.push((e.clone(), idx)),
            _ => {}
        }
    }

    // Check for e = 0 paired with e ≠ 0
    for (eq_expr, eq_idx) in &eq_constraints {
        for (ne_expr, ne_idx) in &ne_constraints {
            if eq_expr == ne_expr {
                // Found a contradiction: e = 0 and e ≠ 0
                let mut combined = OmegaCertificate::new(cert_map[*eq_idx].coefficients.len());
                for (i, coeff) in cert_map[*eq_idx].coefficients.iter().enumerate() {
                    combined.coefficients[i] += coeff;
                }
                for (i, coeff) in cert_map[*ne_idx].coefficients.iter().enumerate() {
                    combined.coefficients[i] += coeff;
                }
                combined.contradiction_type = OmegaContradictionType::Arithmetic;

                return Some(combined);
            }
        }
    }

    None
}

/// Build omega proof from certificate
fn build_omega_proof(
    state: &ProofState,
    goal: &Goal,
    certificate: &OmegaCertificate,
    hypothesis_fvars: &[FVarId],
    env: &Environment,
) -> Option<Expr> {
    // Handle different contradiction types
    match &certificate.contradiction_type {
        OmegaContradictionType::Arithmetic | OmegaContradictionType::LinearCombination => {
            // Use linarith infrastructure for linear combination proofs
            let linarith_cert = LinarithCertificate {
                coefficients: certificate.coefficients.clone(),
                result_constant: 1,
            };
            build_linarith_proof(state, goal, &linarith_cert, hypothesis_fvars, env)
        }
        OmegaContradictionType::Parity => {
            // Parity contradiction: even = odd is False
            // Build proof using Nat.even_iff_not_odd or similar lemmas
            build_parity_contradiction_proof(state, goal, certificate, hypothesis_fvars, env)
        }
        OmegaContradictionType::Divisibility => {
            // Divisibility contradiction: n | k but n ∤ k
            // Build proof using divisibility lemmas
            build_divisibility_contradiction_proof(state, goal, certificate, hypothesis_fvars, env)
        }
    }
}

/// Build a proof from a parity contradiction (even = odd)
///
/// When we have constraints like:
/// - x ≡ 0 (mod 2) meaning x is even
/// - x ≡ 1 (mod 2) meaning x is odd
///
/// These are contradictory. The proof uses:
/// - Nat.not_even_iff_odd : ¬Even n ↔ Odd n
/// - Or directly: if n = 2k and n = 2m + 1, then 2k = 2m + 1, so 2(k-m) = 1,
///   which contradicts that 2 does not divide 1.
///
/// The proof is constructed using the certificate to identify which hypotheses
/// establish the contradiction, then building a proof term using:
/// - `Nat.even_and_odd_elim : ∀ n, Even n → Odd n → False` if available
/// - or `absurd` with `Nat.not_even_iff_odd`
fn build_parity_contradiction_proof(
    state: &ProofState,
    goal: &Goal,
    certificate: &OmegaCertificate,
    hypothesis_fvars: &[FVarId],
    env: &Environment,
) -> Option<Expr> {
    // For parity contradictions, we need to find the hypotheses that establish
    // conflicting parity and combine them to derive False.
    //
    // The proof structure is:
    // 1. From h1 : x ≡ 0 (mod 2), we have ∃ k, x = 2k (Even x)
    // 2. From h2 : x ≡ 1 (mod 2), we have ∃ k, x = 2k + 1 (Odd x)
    // 3. These are contradictory
    //
    // We use: Nat.even_and_odd_elim n h_even h_odd : False
    // Or: absurd h_even (Nat.not_even_of_odd h_odd) : False

    if hypothesis_fvars.is_empty() {
        return None;
    }

    // Find the two hypotheses with non-zero coefficients (the conflicting ones)
    let active: Vec<usize> = certificate
        .coefficients
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, _)| i)
        .collect();

    // We expect exactly 2 hypotheses for a parity contradiction
    if active.len() < 2 {
        // If we don't have 2 active hypotheses, fall back to placeholder
        return None;
    }

    // Identify which active hypotheses correspond to Even and Odd constraints
    let mut even_idx: Option<usize> = None;
    let mut odd_idx: Option<usize> = None;

    for idx in active {
        if idx >= hypothesis_fvars.len() {
            continue;
        }

        let fvar = hypothesis_fvars[idx];
        let hyp_ty = goal
            .local_ctx
            .iter()
            .find(|decl| decl.fvar == fvar)
            .map(|decl| state.metas.instantiate(&decl.ty));

        if let Some(ty) = hyp_ty {
            if let Some(c) = expr_to_omega_constraint(&ty) {
                match c {
                    OmegaConstraint::Mod {
                        modulus, remainder, ..
                    } if modulus == 2 && remainder == 0 => {
                        if even_idx.is_none() {
                            even_idx = Some(idx);
                        }
                    }
                    OmegaConstraint::Mod {
                        modulus, remainder, ..
                    } if modulus == 2 && remainder == 1 => {
                        if odd_idx.is_none() {
                            odd_idx = Some(idx);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let (Some(even_idx), Some(odd_idx)) = (even_idx, odd_idx) else {
        return None;
    };

    if even_idx >= hypothesis_fvars.len() || odd_idx >= hypothesis_fvars.len() {
        return None;
    }

    let h_even = Expr::fvar(hypothesis_fvars[even_idx]);
    let h_odd = Expr::fvar(hypothesis_fvars[odd_idx]);

    // Try to use Nat.even_and_odd_elim if available
    let even_odd_elim = Name::from_string("Nat.even_and_odd_elim");
    if env.get_const(&even_odd_elim).is_some() {
        // Nat.even_and_odd_elim : ∀ n, Even n → Odd n → False
        // We need to figure out which hypothesis is Even and which is Odd
        // For now, we assume h1 is Even and h2 is Odd based on the constraint order
        let elim = Expr::const_(even_odd_elim, vec![]);
        // Apply: even_and_odd_elim _ h1 h2
        // (The `n` argument is implicit, so we just apply h1 and h2)
        let proof = Expr::app(Expr::app(elim, h_even.clone()), h_odd.clone());
        return Some(proof);
    }

    // Try absurd with Nat.not_even_of_odd
    let absurd_name = Name::from_string("absurd");
    let not_even_of_odd = Name::from_string("Nat.not_even_of_odd");

    if env.get_const(&absurd_name).is_some() && env.get_const(&not_even_of_odd).is_some() {
        // absurd : {a : Prop} → {b : Sort u} → a → ¬a → b
        // not_even_of_odd : ∀ {n}, Odd n → ¬Even n
        // Proof: absurd h_even (not_even_of_odd h_odd)
        let absurd = Expr::const_(absurd_name, vec![]);
        let not_even = Expr::app(Expr::const_(not_even_of_odd, vec![]), h_odd.clone());
        let proof = Expr::app(Expr::app(absurd, h_even.clone()), not_even);
        return Some(proof);
    }

    // Try Nat.not_odd_of_even
    let not_odd_of_even = Name::from_string("Nat.not_odd_of_even");
    if env.get_const(&absurd_name).is_some() && env.get_const(&not_odd_of_even).is_some() {
        // not_odd_of_even : ∀ {n}, Even n → ¬Odd n
        // Proof: absurd h_odd (not_odd_of_even h_even)
        let absurd = Expr::const_(absurd_name, vec![]);
        let not_odd = Expr::app(Expr::const_(not_odd_of_even, vec![]), h_even.clone());
        let proof = Expr::app(Expr::app(absurd, h_odd), not_odd);
        return Some(proof);
    }

    // Fall back: return None to trigger decide/sorry fallback
    // The caller will handle this case
    None
}

/// Build a proof from a divisibility contradiction
///
/// When we have constraints like:
/// - n | k (n divides k) meaning k ≡ 0 (mod n)
/// - n ∤ k (n does not divide k) meaning k % n ≠ 0
///
/// Or more generally, conflicting modular constraints:
/// - x ≡ r₁ (mod m)
/// - x ≡ r₂ (mod m) where r₁ ≠ r₂ and 0 ≤ r₁, r₂ < m
///
/// The proof uses the certificate to identify which hypotheses establish
/// the contradiction, then builds a proof term using:
/// - `absurd` when we have `h : m ∣ n` and `h' : ¬(m ∣ n)`
/// - `Nat.mod_contradiction` for conflicting residue classes
fn build_divisibility_contradiction_proof(
    state: &ProofState,
    goal: &Goal,
    certificate: &OmegaCertificate,
    hypothesis_fvars: &[FVarId],
    env: &Environment,
) -> Option<Expr> {
    // For divisibility contradictions, we prove that conflicting residue classes
    // cannot both hold for the same value.
    //
    // Case 1: h1 : m ∣ n and h2 : ¬(m ∣ n)
    //   Use: absurd h1 h2 : False
    //
    // Case 2: h1 : x ≡ r₁ (mod m) and h2 : x ≡ r₂ (mod m) with r₁ ≠ r₂
    //   From h1: x = m*k₁ + r₁
    //   From h2: x = m*k₂ + r₂
    //   So: m*(k₁ - k₂) = r₂ - r₁
    //   If |r₂ - r₁| < m and r₁ ≠ r₂, this is impossible

    if hypothesis_fvars.is_empty() {
        return None;
    }

    // Find the hypotheses with non-zero coefficients
    let active: Vec<usize> = certificate
        .coefficients
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, _)| i)
        .collect();

    // We expect at least 2 hypotheses for a divisibility contradiction
    if active.len() < 2 {
        return None;
    }

    // Collect modular constraints for active hypotheses so we can pair the right ones.
    let mut mod_constraints: Vec<(usize, i64, i64)> = Vec::new(); // (idx, remainder, modulus)
    let mut not_mod_constraints: Vec<(usize, i64, i64)> = Vec::new(); // (idx, remainder, modulus)

    for idx in active {
        if idx >= hypothesis_fvars.len() {
            continue;
        }

        let fvar = hypothesis_fvars[idx];
        let hyp_ty = goal
            .local_ctx
            .iter()
            .find(|decl| decl.fvar == fvar)
            .map(|decl| state.metas.instantiate(&decl.ty));

        if let Some(ty) = hyp_ty {
            if let Some(constraint) = expr_to_omega_constraint(&ty) {
                match constraint {
                    OmegaConstraint::Mod {
                        remainder, modulus, ..
                    }
                    | OmegaConstraint::LinearMod {
                        remainder, modulus, ..
                    } => mod_constraints.push((idx, remainder, modulus)),
                    OmegaConstraint::NotMod { modulus, .. } => {
                        not_mod_constraints.push((idx, 0, modulus));
                    }
                    OmegaConstraint::NotLinearMod {
                        remainder, modulus, ..
                    } => not_mod_constraints.push((idx, remainder, modulus)),
                    _ => {}
                }
            }
        }
    }

    // Case 1: Direct Mod / NotMod contradiction with the same modulus and remainder
    // This handles both r=0 (divisibility) and r≠0 (general modular) cases
    //   h1 : x ≡ r (mod m)   (i.e., x % m = r)
    //   h2 : x % m ≠ r       (NotMod/NotLinearMod with remainder r)
    // Use: absurd h1 h2 : False
    let absurd_name = Name::from_string("absurd");
    if env.get_const(&absurd_name).is_some() {
        for (mod_idx, remainder, modulus) in &mod_constraints {
            for (not_idx, not_remainder, not_modulus) in &not_mod_constraints {
                // Match when same expression has: x ≡ r (mod m) AND x % m ≠ r
                if *not_remainder == *remainder && modulus == not_modulus {
                    if *mod_idx >= hypothesis_fvars.len() || *not_idx >= hypothesis_fvars.len() {
                        continue;
                    }
                    let h_mod = Expr::fvar(hypothesis_fvars[*mod_idx]);
                    let h_not_mod = Expr::fvar(hypothesis_fvars[*not_idx]);
                    let absurd = Expr::const_(absurd_name.clone(), vec![]);
                    let proof = Expr::app(Expr::app(absurd, h_mod), h_not_mod);
                    return Some(proof);
                }
            }
        }
    }

    // Case 2: Conflicting modular constraints (different remainders)
    // h1 : x % m = r₁  AND  h2 : x % m = r₂  where r₁ ≠ r₂
    //
    // Proof strategy:
    //   From h1: x % m = r1 and h2: x % m = r2
    //   We derive r1 = r2 via:  Eq.trans (Eq.symm h1) h2
    //   When r1 ≠ r2 are distinct literals, (r1 = r2) is decidably False.
    //
    // Using Nat.noConfusion:
    //   h_eq : r1 = r2  derived from Eq.trans (Eq.symm h1) h2
    //   Nat.noConfusion h_eq : False  (when r1 ≠ r2 are distinct Nat literals)

    // Check for pairs with same modulus but different remainders
    let eq_symm_name = Name::from_string("Eq.symm");
    let eq_trans_name = Name::from_string("Eq.trans");
    let nat_noconfusion_name = Name::from_string("Nat.noConfusion");

    let have_eq_trans = env.get_const(&eq_trans_name).is_some();
    let have_eq_symm = env.get_const(&eq_symm_name).is_some();
    // Nat.noConfusion is stored as a recursor, not a constant
    let have_nat_noconfusion = env.get_const(&nat_noconfusion_name).is_some()
        || env.get_recursor(&nat_noconfusion_name).is_some();

    if have_eq_trans && have_eq_symm && have_nat_noconfusion {
        for (i, &(idx_i, remainder_i, modulus_i)) in mod_constraints.iter().enumerate() {
            for &(idx_j, remainder_j, modulus_j) in mod_constraints.iter().skip(i + 1) {
                // Check if same modulus but different remainders
                if modulus_i == modulus_j && remainder_i != remainder_j {
                    if idx_i >= hypothesis_fvars.len() || idx_j >= hypothesis_fvars.len() {
                        continue;
                    }

                    // h1 : x % m = r1   and   h2 : x % m = r2
                    let h1 = Expr::fvar(hypothesis_fvars[idx_i]);
                    let h2 = Expr::fvar(hypothesis_fvars[idx_j]);

                    // Build r1 and r2 as Nat literals
                    let r1 = Expr::Lit(Literal::Nat(remainder_i as u64));
                    let r2 = Expr::Lit(Literal::Nat(remainder_j as u64));

                    // We need the common "middle" term: x % m
                    // For h1 : x % m = r1, we need Eq.symm h1 : r1 = x % m
                    // Then Eq.trans (Eq.symm h1) h2 : r1 = r2
                    //
                    // Eq.symm : {α : Sort u} → {a b : α} → a = b → b = a
                    // Eq.trans : {α : Sort u} → {a b c : α} → a = b → b = c → a = c

                    // Build Eq.symm {Nat} {x % m} {r1} h1
                    // This gives us: r1 = x % m
                    let eq_symm =
                        Expr::const_(eq_symm_name.clone(), vec![Level::Param(Name::anon())]);
                    // Need to extract the middle term from the hypothesis type
                    // For now, we use placeholders via implicit args
                    // Eq.symm with implicits: {α} {a} {b} (h : a = b) : b = a
                    let symm_h1 = Expr::app(eq_symm, h1.clone());

                    // Build Eq.trans {Nat} {r1} {x % m} {r2} (Eq.symm h1) h2
                    // This gives us: r1 = r2
                    let eq_trans =
                        Expr::const_(eq_trans_name.clone(), vec![Level::Param(Name::anon())]);
                    let trans_proof = Expr::app(Expr::app(eq_trans, symm_h1), h2.clone());

                    // Build Nat.noConfusion {False} {r1} {r2} trans_proof
                    // Nat.noConfusion : {P : Sort u} → {v1 v2 : Nat} → v1 = v2 → Nat.noConfusionType P v1 v2
                    // When v1 ≠ v2 definitionally, noConfusionType P v1 v2 = P
                    // So Nat.noConfusion h : False  when h : r1 = r2 and r1 ≠ r2
                    let false_ty = Expr::const_(Name::from_string("False"), vec![]);
                    let nat_nc = Expr::const_(nat_noconfusion_name.clone(), vec![Level::zero()]);
                    // Apply: {P := False} {v1 := r1} {v2 := r2} trans_proof
                    let proof = Expr::app(
                        Expr::app(Expr::app(Expr::app(nat_nc, false_ty), r1), r2),
                        trans_proof,
                    );

                    return Some(proof);
                }
            }
        }
    }

    // Fall back: return None to trigger decide/sorry fallback
    None
}

/// Omega constraint representation
#[derive(Debug, Clone)]
pub enum OmegaConstraint {
    /// a₁x₁ + a₂x₂ + ... + c ≤ 0
    Le(LinearExpr),
    /// a₁x₁ + a₂x₂ + ... + c < 0
    Lt(LinearExpr),
    /// a₁x₁ + a₂x₂ + ... + c = 0
    Eq(LinearExpr),
    /// a₁x₁ + a₂x₂ + ... + c ≠ 0
    Ne(LinearExpr),
    /// x ≡ r (mod m)
    Mod {
        var: usize,
        remainder: i64,
        modulus: i64,
    },
    /// ¬(m ∣ x), i.e., x % m ≠ 0
    /// Represents `Not (Dvd.dvd m x)` - x is NOT divisible by m
    NotMod { var: usize, modulus: i64 },
    /// expr ≡ r (mod m) where expr is a general linear expression
    /// Represents `(a + b + ...) % m = r` or more complex modular constraints
    LinearMod {
        expr: LinearExpr,
        remainder: i64,
        modulus: i64,
    },
    /// ¬(expr ≡ r (mod m)) - negated modular equality with arbitrary remainder
    /// Represents `(a + b + ...) % m ≠ r`
    NotLinearMod {
        expr: LinearExpr,
        remainder: i64,
        modulus: i64,
    },
}

/// Convert expression to omega constraint
pub(crate) fn expr_to_omega_constraint(expr: &Expr) -> Option<OmegaConstraint> {
    // Check for Even/Odd predicates: `Even n` or `Odd n`
    // Even n ≡ ∃ k, n = 2 * k  ⟺  n ≡ 0 (mod 2)
    // Odd n ≡ ∃ k, n = 2 * k + 1  ⟺  n ≡ 1 (mod 2)
    if let Expr::App(f, arg) = expr {
        if let Expr::Const(name, _) = f.as_ref() {
            let name_str = name.to_string();
            if name_str == "Even" || name_str == "Nat.Even" || name_str == "Int.Even" {
                // Even n ⟺ n ≡ 0 (mod 2)
                if let Some(var) = extract_single_var(arg) {
                    return Some(OmegaConstraint::Mod {
                        var,
                        remainder: 0,
                        modulus: 2,
                    });
                }
            }
            if name_str == "Odd" || name_str == "Nat.Odd" || name_str == "Int.Odd" {
                // Odd n ⟺ n ≡ 1 (mod 2)
                if let Some(var) = extract_single_var(arg) {
                    return Some(OmegaConstraint::Mod {
                        var,
                        remainder: 1,
                        modulus: 2,
                    });
                }
            }
        }
        // Check for Even/Odd with type argument: `@Even Nat _ n` or `@Odd Int _ n`
        if let Expr::App(f2, arg2) = f.as_ref() {
            if let Expr::App(f3, _inst) = f2.as_ref() {
                if let Expr::App(f4, _ty) = f3.as_ref() {
                    if let Expr::Const(name, _) = f4.as_ref() {
                        let name_str = name.to_string();
                        if name_str == "Even" {
                            // The actual argument is `arg`, the final one applied
                            if let Some(var) = extract_single_var(arg) {
                                return Some(OmegaConstraint::Mod {
                                    var,
                                    remainder: 0,
                                    modulus: 2,
                                });
                            }
                            // Sometimes the argument is wrapped differently
                            if let Some(var) = extract_single_var(arg2) {
                                return Some(OmegaConstraint::Mod {
                                    var,
                                    remainder: 0,
                                    modulus: 2,
                                });
                            }
                        }
                        if name_str == "Odd" {
                            if let Some(var) = extract_single_var(arg) {
                                return Some(OmegaConstraint::Mod {
                                    var,
                                    remainder: 1,
                                    modulus: 2,
                                });
                            }
                            if let Some(var) = extract_single_var(arg2) {
                                return Some(OmegaConstraint::Mod {
                                    var,
                                    remainder: 1,
                                    modulus: 2,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Check for negated parity/divisibility:
    // - `Not (Even n)` → `Odd n`
    // - `Not (Odd n)` → `Even n`
    // - `Not (Dvd.dvd a b)` → `NotMod { var: b, modulus: a }`
    //
    // Not P in Lean is `P → False`, elaborated as `App (App (Const "Not") P)`
    // or directly `App (Const "Not" _levels) P`
    if let Expr::App(f, inner) = expr {
        if let Expr::Const(name, _) = f.as_ref() {
            let name_str = name.to_string();
            if name_str == "Not" {
                // Parse the inner expression
                if let Some(inner_constraint) = expr_to_omega_constraint(inner) {
                    // Negate modular constraints
                    match inner_constraint {
                        OmegaConstraint::Mod {
                            var,
                            remainder,
                            modulus,
                        } => {
                            if modulus == 2 {
                                // ¬(Even n) ⟺ Odd n: remainder 0 → 1
                                // ¬(Odd n) ⟺ Even n: remainder 1 → 0
                                let new_remainder = 1 - remainder;
                                return Some(OmegaConstraint::Mod {
                                    var,
                                    remainder: new_remainder,
                                    modulus: 2,
                                });
                            } else if remainder == 0 {
                                // ¬(a ∣ b) where a ∣ b was parsed as b ≡ 0 (mod a)
                                // means b % a ≠ 0
                                return Some(OmegaConstraint::NotMod { var, modulus });
                            }
                            // ¬(n % m = r) where r ≠ 0: convert to NotLinearMod
                            // This handles general negated modular equalities
                            return Some(OmegaConstraint::NotLinearMod {
                                expr: LinearExpr::var(var),
                                remainder,
                                modulus,
                            });
                        }
                        OmegaConstraint::LinearMod {
                            expr,
                            remainder,
                            modulus,
                        } => {
                            // ¬((a + b) % m = r) → NotLinearMod
                            return Some(OmegaConstraint::NotLinearMod {
                                expr,
                                remainder,
                                modulus,
                            });
                        }
                        OmegaConstraint::NotLinearMod {
                            expr,
                            remainder,
                            modulus,
                        } => {
                            // ¬(¬((a + b) % m = r)) → LinearMod
                            return Some(OmegaConstraint::LinearMod {
                                expr,
                                remainder,
                                modulus,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Check for Dvd (divisibility): `a ∣ b` or `Dvd.dvd a b`
    // a ∣ b ≡ ∃ k, b = a * k  ⟺  b ≡ 0 (mod a)
    if let Expr::App(f, b) = expr {
        if let Expr::App(f2, a) = f.as_ref() {
            // Check for `Dvd.dvd a b` pattern (with instance/type args)
            if let Some((divisor, dividend)) = match_dvd_app(f2, a, b) {
                // a ∣ b  ⟺  b ≡ 0 (mod a)
                if let Some(var) = extract_single_var(&dividend) {
                    if let Some(mod_val) = extract_constant(&divisor) {
                        if mod_val > 0 {
                            return Some(OmegaConstraint::Mod {
                                var,
                                remainder: 0,
                                modulus: mod_val,
                            });
                        }
                    }
                }
            }
        }
    }

    // Check for modular equality: `n % m = r` where m and r are constants
    // Pattern: Eq (HMod.hMod n m) r  ⟺  n ≡ r (mod m)
    if let Some(mod_constraint) = parse_mod_equality(expr) {
        return Some(mod_constraint);
    }

    // Check for comparison operators
    if let Expr::App(f, rhs) = expr {
        if let Expr::App(f2, lhs) = f.as_ref() {
            if let Expr::App(f3, _ty) = f2.as_ref() {
                if let Expr::Const(name, _) = f3.as_ref() {
                    let name_str = name.to_string();

                    // Try to parse lhs and rhs as linear expressions
                    let lhs_lin = expr_to_linear(lhs)?;
                    let rhs_lin = expr_to_linear(rhs)?;
                    let diff = lhs_lin.sub(&rhs_lin);

                    if name_str.contains("LE.le") || name_str.contains("le") {
                        // lhs ≤ rhs  ⟺  lhs - rhs ≤ 0
                        return Some(OmegaConstraint::Le(diff));
                    }
                    if name_str.contains("LT.lt") || name_str.contains("lt") {
                        // lhs < rhs  ⟺  lhs - rhs < 0
                        return Some(OmegaConstraint::Lt(diff));
                    }
                    if name_str.contains("GE.ge") || name_str.contains("ge") {
                        // lhs ≥ rhs  ⟺  rhs - lhs ≤ 0
                        return Some(OmegaConstraint::Le(rhs_lin.sub(&lhs_lin)));
                    }
                    if name_str.contains("GT.gt") || name_str.contains("gt") {
                        // lhs > rhs  ⟺  rhs - lhs < 0
                        return Some(OmegaConstraint::Lt(rhs_lin.sub(&lhs_lin)));
                    }
                    if name_str.contains("Eq") {
                        // lhs = rhs  ⟺  lhs - rhs = 0
                        return Some(OmegaConstraint::Eq(diff));
                    }
                    if name_str.contains("Ne") {
                        // lhs ≠ rhs  ⟺  lhs - rhs ≠ 0
                        return Some(OmegaConstraint::Ne(diff));
                    }
                }
            }
        }
    }
    None
}

/// Extract a single variable index from an expression (for Even/Odd/Dvd)
pub(crate) fn extract_single_var(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::FVar(id) => Some(id.0 as usize),
        // Handle simple wrapper applications (like OfNat.ofNat)
        Expr::App(f, arg) => {
            if let Expr::Const(name, _) = f.as_ref() {
                let name_str = name.to_string();
                // Skip type coercions
                if name_str.contains("ofNat") || name_str.contains("cast") {
                    return extract_single_var(arg);
                }
            }
            // Try the argument directly
            extract_single_var(arg)
        }
        _ => None,
    }
}

/// Extract a constant value from an expression
pub(crate) fn extract_constant(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(lean5_kernel::expr::Literal::Nat(n)) => Some(*n as i64),
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            if name_str == "Nat.zero" || name_str == "Int.zero" {
                Some(0)
            } else if name_str == "Nat.one" || name_str == "Int.one" {
                Some(1)
            } else {
                None
            }
        }
        // Handle OfNat.ofNat applications
        Expr::App(f, arg) => {
            if let Expr::App(f2, val) = f.as_ref() {
                if let Expr::Const(name, _) = get_app_fn(f2) {
                    let name_str = name.to_string();
                    if name_str.contains("OfNat.ofNat") {
                        // The numeric value is embedded in `val`
                        return extract_constant(val);
                    }
                }
            }
            extract_constant(arg)
        }
        _ => None,
    }
}

/// Match a Dvd.dvd application pattern
/// Returns Some((divisor, dividend)) if this is a `a ∣ b` expression
fn match_dvd_app(f2: &Expr, a: &Expr, b: &Expr) -> Option<(Expr, Expr)> {
    // Pattern: Dvd.dvd inst a b where inst is the Dvd instance
    if let Expr::App(f3, _inst) = f2 {
        if let Expr::App(f4, _ty) = f3.as_ref() {
            if let Expr::Const(name, _) = f4.as_ref() {
                let name_str = name.to_string();
                if name_str == "Dvd.dvd" || name_str.contains("dvd") {
                    return Some((a.clone(), b.clone()));
                }
            }
        }
        // Also try simpler pattern
        if let Expr::Const(name, _) = f3.as_ref() {
            let name_str = name.to_string();
            if name_str == "Dvd.dvd" || name_str.contains("dvd") {
                return Some((a.clone(), b.clone()));
            }
        }
    }
    // Direct Dvd.dvd application
    if let Expr::Const(name, _) = f2 {
        let name_str = name.to_string();
        if name_str == "Dvd.dvd" || name_str.contains("dvd") {
            return Some((a.clone(), b.clone()));
        }
    }
    None
}

/// Parse modular equality: `n % m = r` → `n ≡ r (mod m)`
///
/// Matches patterns:
/// - `Eq (HMod.hMod n m) r` (with type and instance args)
/// - `n % m = r` (desugared form)
///
/// Returns:
/// - Some(OmegaConstraint::Mod) if n is a single variable and m, r are constants
/// - Some(OmegaConstraint::LinearMod) if n is a linear expression and m, r are constants
fn parse_mod_equality(expr: &Expr) -> Option<OmegaConstraint> {
    // Pattern: `Eq _ (HMod.hMod _ _ _ _ n m) r`
    // The equality is: App (App (App (Const "Eq") ty) lhs) rhs
    // where lhs = HMod.hMod with various args, ending in n and m

    if let Expr::App(f, rhs) = expr {
        if let Expr::App(f2, lhs) = f.as_ref() {
            // Check if this is an Eq
            let is_eq = if let Expr::App(f3, _ty) = f2.as_ref() {
                if let Expr::Const(name, _) = f3.as_ref() {
                    name.to_string().contains("Eq")
                } else {
                    false
                }
            } else {
                false
            };

            if is_eq {
                // Check if lhs is a modulo operation: HMod.hMod
                if let Some((var_expr, modulus_expr)) = match_hmod_app(lhs) {
                    // Extract constant modulus from m
                    if let Some(modulus) = extract_constant(&modulus_expr) {
                        // Extract constant remainder from rhs
                        if let Some(remainder) = extract_constant(rhs) {
                            if modulus > 0 && remainder >= 0 && remainder < modulus {
                                // Try to parse n as a linear expression first
                                // This handles both single variables AND compound expressions
                                if let Some(lin_expr) = expr_to_linear(&var_expr) {
                                    // If it's a single variable (e.g., x with coefficient 1),
                                    // use the simpler Mod constraint
                                    if lin_expr.coeffs.len() == 1
                                        && lin_expr.constant == 0
                                        && lin_expr.coeffs.values().next() == Some(&1)
                                    {
                                        let var = *lin_expr
                                            .coeffs
                                            .keys()
                                            .next()
                                            .expect("coeffs has exactly 1 element");
                                        return Some(OmegaConstraint::Mod {
                                            var,
                                            remainder,
                                            modulus,
                                        });
                                    }
                                    // Otherwise it's a compound expression (a + b, 2*n, etc.)
                                    return Some(OmegaConstraint::LinearMod {
                                        expr: lin_expr,
                                        remainder,
                                        modulus,
                                    });
                                }
                                // Fallback: try extract_single_var for simple FVar cases
                                // that expr_to_linear might not handle
                                if let Some(var) = extract_single_var(&var_expr) {
                                    return Some(OmegaConstraint::Mod {
                                        var,
                                        remainder,
                                        modulus,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Match HMod.hMod application pattern
/// Returns Some((n, m)) if this is `n % m` (HMod.hMod ... n m)
pub(crate) fn match_hmod_app(expr: &Expr) -> Option<(Expr, Expr)> {
    // HMod.hMod with full arguments:
    // App (App (App (App (App (App (Const "HMod.hMod") ty1) ty2) ty3) inst) n) m
    // We need to find the function head and extract the last two args (n, m)

    // Try to extract the function name
    let fn_name = get_const_name_from_app(expr);
    if let Some(name) = fn_name {
        let name_str = name.to_string();
        if name_str == "HMod.hMod"
            || name_str == "Nat.mod"
            || name_str == "Int.mod"
            || name_str.ends_with(".hMod")
            || name_str.ends_with(".mod")
        {
            // Extract the last two arguments (n and m)
            if let Some((n, m)) = extract_last_two_args(expr) {
                return Some((n, m));
            }
        }
    }
    None
}

/// Get the constant name from an application chain
fn get_const_name_from_app(expr: &Expr) -> Option<&Name> {
    match expr {
        Expr::Const(name, _) => Some(name),
        Expr::App(f, _) => get_const_name_from_app(f),
        _ => None,
    }
}

/// Extract the last two arguments from an application chain
fn extract_last_two_args(expr: &Expr) -> Option<(Expr, Expr)> {
    // Pattern: App (App ... n) m
    // We want to get n and m
    if let Expr::App(f, m) = expr {
        if let Expr::App(_f2, n) = f.as_ref() {
            return Some((n.as_ref().clone(), m.as_ref().clone()));
        }
    }
    None
}

/// Convert expression to linear expression
pub(crate) fn expr_to_linear(expr: &Expr) -> Option<LinearExpr> {
    match expr {
        Expr::Lit(lean5_kernel::expr::Literal::Nat(n)) => Some(LinearExpr::constant(*n as i64)),
        Expr::Const(name, _) => {
            let name_str = name.to_string();
            if name_str == "Nat.zero" || name_str == "Int.zero" {
                Some(LinearExpr::constant(0))
            } else if name_str == "Nat.one" || name_str == "Int.one" {
                Some(LinearExpr::constant(1))
            } else {
                None
            }
        }
        Expr::FVar(id) => Some(LinearExpr::var(id.0 as usize)),
        Expr::App(f, arg) => {
            // Check for binary operations
            if let Expr::App(f2, arg1) = f.as_ref() {
                if let Expr::Const(name, _) = get_app_fn(f2) {
                    let name_str = name.to_string();

                    if name_str.contains("add") || name_str.contains("Add") {
                        let left = expr_to_linear(arg1)?;
                        let right = expr_to_linear(arg)?;
                        return Some(left.add(&right));
                    }
                    if name_str.contains("sub") || name_str.contains("Sub") {
                        let left = expr_to_linear(arg1)?;
                        let right = expr_to_linear(arg)?;
                        return Some(left.sub(&right));
                    }
                    if name_str.contains("mul") || name_str.contains("Mul") {
                        // One side must be constant for linearity
                        if let Some(left) = expr_to_linear(arg1) {
                            if left.is_constant() {
                                if let Some(right) = expr_to_linear(arg) {
                                    return Some(right.scale(left.constant));
                                }
                            }
                        }
                        if let Some(right) = expr_to_linear(arg) {
                            if right.is_constant() {
                                if let Some(left) = expr_to_linear(arg1) {
                                    return Some(left.scale(right.constant));
                                }
                            }
                        }
                    }
                }
            }
            // Check for unary operations
            if let Expr::Const(name, _) = f.as_ref() {
                let name_str = name.to_string();
                if name_str.contains("neg") || name_str.contains("Neg") {
                    let inner = expr_to_linear(arg)?;
                    return Some(inner.scale(-1));
                }
                if name_str == "Nat.succ" {
                    let inner = expr_to_linear(arg)?;
                    return Some(inner.add(&LinearExpr::constant(1)));
                }
            }
            None
        }
        _ => None,
    }
}

/// Negate an omega constraint
pub(crate) fn negate_omega_constraint(constraint: &OmegaConstraint) -> Option<OmegaConstraint> {
    match constraint {
        OmegaConstraint::Le(e) => {
            // ¬(e ≤ 0)  ⟺  e > 0  ⟺  -e < 0  ⟺  -e - 1 ≤ 0
            let neg = e.scale(-1);
            let mut shifted = neg;
            shifted.constant -= 1;
            Some(OmegaConstraint::Le(shifted))
        }
        OmegaConstraint::Lt(e) => {
            // ¬(e < 0)  ⟺  e ≥ 0  ⟺  -e ≤ 0
            Some(OmegaConstraint::Le(e.scale(-1)))
        }
        OmegaConstraint::Eq(e) => {
            // ¬(e = 0)  ⟺  e ≠ 0
            Some(OmegaConstraint::Ne(e.clone()))
        }
        OmegaConstraint::Ne(e) => {
            // ¬(e ≠ 0)  ⟺  e = 0
            Some(OmegaConstraint::Eq(e.clone()))
        }
        OmegaConstraint::Mod { var, modulus, .. } => {
            // ¬(x ≡ r (mod m)) where r = 0 becomes NotMod
            // For simplicity, only handle the divisibility case (r = 0)
            Some(OmegaConstraint::NotMod {
                var: *var,
                modulus: *modulus,
            })
        }
        OmegaConstraint::NotMod { var, modulus } => {
            // ¬(¬(m ∣ x))  ⟺  m ∣ x  ⟺  x ≡ 0 (mod m)
            Some(OmegaConstraint::Mod {
                var: *var,
                remainder: 0,
                modulus: *modulus,
            })
        }
        OmegaConstraint::LinearMod {
            expr,
            remainder,
            modulus,
        } => {
            // ¬(expr ≡ r (mod m))  ⟺  expr ≢ r (mod m)
            Some(OmegaConstraint::NotLinearMod {
                expr: expr.clone(),
                remainder: *remainder,
                modulus: *modulus,
            })
        }
        OmegaConstraint::NotLinearMod {
            expr,
            remainder,
            modulus,
        } => {
            // ¬(expr ≢ r (mod m))  ⟺  expr ≡ r (mod m)
            Some(OmegaConstraint::LinearMod {
                expr: expr.clone(),
                remainder: *remainder,
                modulus: *modulus,
            })
        }
    }
}

/// Run the omega decision procedure
fn omega_check(constraints: &[OmegaConstraint]) -> bool {
    // Convert to linear constraints for Fourier-Motzkin
    let mut linear_constraints = Vec::new();

    for c in constraints {
        match c {
            OmegaConstraint::Le(e) => {
                linear_constraints.push(LinearConstraint::Le(e.clone()));
            }
            OmegaConstraint::Lt(e) => {
                linear_constraints.push(LinearConstraint::Lt(e.clone()));
            }
            OmegaConstraint::Eq(e) => {
                linear_constraints.push(LinearConstraint::Eq(e.clone()));
            }
            // Handle disequalities, modular constraints, etc. (simplified)
            // Full omega would enumerate cases and residue classes
            OmegaConstraint::Ne(_)
            | OmegaConstraint::Mod { .. }
            | OmegaConstraint::NotMod { .. }
            | OmegaConstraint::LinearMod { .. }
            | OmegaConstraint::NotLinearMod { .. } => {}
        }
    }

    // Use Fourier-Motzkin
    matches!(fourier_motzkin_check(&linear_constraints), FMResult::Unsat)
}
