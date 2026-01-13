//! Algebraic normalization tactics for abelian groups and multiplicative groups.
//!
//! This module provides tactics that normalize algebraic expressions:
//! - `abel`: Normalizes expressions in additive abelian groups
//! - `group`: Normalizes expressions in multiplicative groups

use lean5_kernel::Expr;

use super::{expr_to_int, exprs_equal, match_equality, rfl, ProofState, TacticError, TacticResult};

// ============================================================================
// Helper Functions
// ============================================================================

/// Match an equality expression, returning (lhs, rhs) or None
pub(crate) fn match_eq_simple(expr: &Expr) -> Option<(Expr, Expr)> {
    match_equality(expr)
        .ok()
        .map(|(_ty, lhs, rhs, _levels)| (lhs, rhs))
}

/// Check if an expression is a Pi/forall type
pub(crate) fn is_pi_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Pi(_, _, _))
}

// ============================================================================
// Abel Tactic
// ============================================================================

/// Configuration for the abel tactic
#[derive(Debug, Clone)]
pub struct AbelConfig {
    /// Maximum number of normalization steps
    pub max_steps: usize,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for AbelConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            verbose: false,
        }
    }
}

/// Represents an abelian group term in normalized form.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AbelTerm {
    pub(crate) coefficients: std::collections::BTreeMap<usize, i64>,
    pub(crate) variables: Vec<Expr>,
}

impl AbelTerm {
    pub(crate) fn new() -> Self {
        Self {
            coefficients: std::collections::BTreeMap::new(),
            variables: Vec::new(),
        }
    }

    pub(crate) fn zero() -> Self {
        Self::new()
    }

    pub(crate) fn single(var_idx: usize, expr: Expr) -> Self {
        let mut term = Self::new();
        term.coefficients.insert(var_idx, 1);
        term.variables.push(expr);
        term
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (idx, &coeff) in &other.coefficients {
            *result.coefficients.entry(*idx).or_insert(0) += coeff;
        }
        for var in &other.variables {
            if !result.variables.iter().any(|v| exprs_equal(v, var)) {
                result.variables.push(var.clone());
            }
        }
        result.coefficients.retain(|_, &mut c| c != 0);
        result
    }

    pub(crate) fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    pub(crate) fn negate(&self) -> Self {
        let mut result = self.clone();
        for coeff in result.coefficients.values_mut() {
            *coeff = -*coeff;
        }
        result
    }

    #[allow(dead_code)] // Reserved for future use
    pub(crate) fn scale(&self, k: i64) -> Self {
        if k == 0 {
            return Self::zero();
        }
        let mut result = self.clone();
        for coeff in result.coefficients.values_mut() {
            *coeff *= k;
        }
        result
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.values().all(|&c| c == 0)
    }
}

/// Tactic: abel - Normalizes expressions in additive abelian groups.
pub fn abel(state: &mut ProofState) -> TacticResult {
    abel_with_config(state, AbelConfig::default())
}

pub fn abel_with_config(state: &mut ProofState, config: AbelConfig) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = state.metas.instantiate(&goal.target);

    let (lhs, rhs) = match_eq_simple(&target)
        .ok_or_else(|| TacticError::Other("abel: goal must be an equality".to_string()))?;

    let mut var_counter = 0;
    let mut var_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut vars: Vec<Expr> = Vec::new();

    let lhs_term = parse_abel_term(&lhs, &mut var_counter, &mut var_map, &mut vars, &config)?;
    let rhs_term = parse_abel_term(&rhs, &mut var_counter, &mut var_map, &mut vars, &config)?;

    let diff = lhs_term.sub(&rhs_term);
    if !diff.is_zero() {
        return Err(TacticError::Other(format!(
            "abel: terms do not normalize to equal values. Difference: {:?}",
            diff.coefficients
        )));
    }

    rfl(state)
}

fn parse_abel_term(
    expr: &Expr,
    var_counter: &mut usize,
    var_map: &mut std::collections::HashMap<String, usize>,
    vars: &mut Vec<Expr>,
    _config: &AbelConfig,
) -> Result<AbelTerm, TacticError> {
    match expr {
        Expr::App(f, b) => {
            if let Expr::App(f2, a) = f.as_ref() {
                if let Expr::App(f3, _ty) = f2.as_ref() {
                    if let Expr::Const(name, _) = f3.as_ref() {
                        let name_str = name.to_string();
                        if name_str == "HAdd.hAdd" || name_str == "Add.add" {
                            let a_term = parse_abel_term(a, var_counter, var_map, vars, _config)?;
                            let b_term = parse_abel_term(b, var_counter, var_map, vars, _config)?;
                            return Ok(a_term.add(&b_term));
                        }
                        if name_str == "HSub.hSub" || name_str == "Sub.sub" {
                            let a_term = parse_abel_term(a, var_counter, var_map, vars, _config)?;
                            let b_term = parse_abel_term(b, var_counter, var_map, vars, _config)?;
                            return Ok(a_term.sub(&b_term));
                        }
                    }
                }
                if let Expr::App(f3, _ty) = f.as_ref() {
                    if let Expr::Const(name, _) = f3.as_ref() {
                        if name.to_string() == "Neg.neg" {
                            let a_term = parse_abel_term(b, var_counter, var_map, vars, _config)?;
                            return Ok(a_term.negate());
                        }
                    }
                }
            }
            if let Expr::App(f2, _ty) = f.as_ref() {
                if let Expr::Const(name, _) = f2.as_ref() {
                    if name.to_string() == "Neg.neg" {
                        let a_term = parse_abel_term(b, var_counter, var_map, vars, _config)?;
                        return Ok(a_term.negate());
                    }
                }
            }
            let key = format!("{expr:?}");
            let idx = *var_map.entry(key).or_insert_with(|| {
                let idx = *var_counter;
                *var_counter += 1;
                vars.push(expr.clone());
                idx
            });
            Ok(AbelTerm::single(idx, expr.clone()))
        }
        Expr::Const(name, _) if name.to_string().contains("zero") || name.to_string() == "0" => {
            Ok(AbelTerm::zero())
        }
        Expr::Lit(lean5_kernel::Literal::Nat(0)) => Ok(AbelTerm::zero()),
        _ => {
            let key = format!("{expr:?}");
            let idx = *var_map.entry(key).or_insert_with(|| {
                let idx = *var_counter;
                *var_counter += 1;
                vars.push(expr.clone());
                idx
            });
            Ok(AbelTerm::single(idx, expr.clone()))
        }
    }
}

// ============================================================================
// Group Tactic
// ============================================================================

#[derive(Debug, Clone)]
pub struct GroupConfig {
    pub max_steps: usize,
    pub verbose: bool,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroupTerm {
    pub(crate) factors: Vec<(usize, i64)>,
    pub(crate) variables: Vec<Expr>,
}

impl GroupTerm {
    pub(crate) fn identity() -> Self {
        Self {
            factors: Vec::new(),
            variables: Vec::new(),
        }
    }

    pub(crate) fn single(var_idx: usize, expr: Expr) -> Self {
        Self {
            factors: vec![(var_idx, 1)],
            variables: vec![expr],
        }
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for var in &other.variables {
            if !result.variables.iter().any(|v| exprs_equal(v, var)) {
                result.variables.push(var.clone());
            }
        }
        for &(idx, exp) in &other.factors {
            if let Some(pos) = result.factors.iter().position(|&(i, _)| i == idx) {
                result.factors[pos].1 += exp;
                if result.factors[pos].1 == 0 {
                    result.factors.remove(pos);
                }
            } else {
                result.factors.push((idx, exp));
            }
        }
        result.factors.sort_by_key(|&(idx, _)| idx);
        result
    }

    pub(crate) fn inv(&self) -> Self {
        let mut result = self.clone();
        for (_, exp) in &mut result.factors {
            *exp = -*exp;
        }
        result.factors.reverse();
        result
    }

    pub(crate) fn pow(&self, n: i64) -> Self {
        if n == 0 {
            return Self::identity();
        }
        let mut result = self.clone();
        for (_, exp) in &mut result.factors {
            *exp *= n;
        }
        result.factors.retain(|&(_, e)| e != 0);
        result
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.factors.is_empty() || self.factors.iter().all(|&(_, e)| e == 0)
    }

    pub(crate) fn eq_abelian(&self, other: &Self) -> bool {
        if self.factors.len() != other.factors.len() {
            return false;
        }
        let mut self_sorted = self.factors.clone();
        let mut other_sorted = other.factors.clone();
        self_sorted.sort_by_key(|&(idx, _)| idx);
        other_sorted.sort_by_key(|&(idx, _)| idx);
        self_sorted == other_sorted
    }
}

/// Tactic: group - Normalizes expressions in multiplicative groups.
pub fn group(state: &mut ProofState) -> TacticResult {
    group_with_config(state, GroupConfig::default())
}

pub fn group_with_config(state: &mut ProofState, config: GroupConfig) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = state.metas.instantiate(&goal.target);

    let (lhs, rhs) = match_eq_simple(&target)
        .ok_or_else(|| TacticError::Other("group: goal must be an equality".to_string()))?;

    let mut var_counter = 0;
    let mut var_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut vars: Vec<Expr> = Vec::new();

    let lhs_term = parse_group_term(&lhs, &mut var_counter, &mut var_map, &mut vars, &config)?;
    let rhs_term = parse_group_term(&rhs, &mut var_counter, &mut var_map, &mut vars, &config)?;

    let diff = lhs_term.mul(&rhs_term.inv());
    if !diff.is_identity() && !lhs_term.eq_abelian(&rhs_term) {
        return Err(TacticError::Other(format!(
            "group: terms do not normalize to equal values. lhs: {:?}, rhs: {:?}",
            lhs_term.factors, rhs_term.factors
        )));
    }

    rfl(state)
}

fn parse_group_term(
    expr: &Expr,
    var_counter: &mut usize,
    var_map: &mut std::collections::HashMap<String, usize>,
    vars: &mut Vec<Expr>,
    _config: &GroupConfig,
) -> Result<GroupTerm, TacticError> {
    match expr {
        Expr::App(f, b) => {
            if let Expr::App(f2, a) = f.as_ref() {
                if let Expr::App(f3, _ty) = f2.as_ref() {
                    if let Expr::Const(name, _) = f3.as_ref() {
                        let name_str = name.to_string();
                        if name_str == "HMul.hMul" || name_str == "Mul.mul" {
                            let a_term = parse_group_term(a, var_counter, var_map, vars, _config)?;
                            let b_term = parse_group_term(b, var_counter, var_map, vars, _config)?;
                            return Ok(a_term.mul(&b_term));
                        }
                        if name_str == "HDiv.hDiv" || name_str == "Div.div" {
                            let a_term = parse_group_term(a, var_counter, var_map, vars, _config)?;
                            let b_term = parse_group_term(b, var_counter, var_map, vars, _config)?;
                            return Ok(a_term.mul(&b_term.inv()));
                        }
                        if name_str == "HPow.hPow" || name_str == "Pow.pow" {
                            if let Some(n) = expr_to_int(b) {
                                let a_term =
                                    parse_group_term(a, var_counter, var_map, vars, _config)?;
                                return Ok(a_term.pow(n));
                            }
                        }
                    }
                }
                if let Expr::App(f3, _ty) = f.as_ref() {
                    if let Expr::Const(name, _) = f3.as_ref() {
                        if name.to_string() == "Inv.inv" {
                            let a_term = parse_group_term(b, var_counter, var_map, vars, _config)?;
                            return Ok(a_term.inv());
                        }
                    }
                }
            }
            if let Expr::App(f2, _ty) = f.as_ref() {
                if let Expr::Const(name, _) = f2.as_ref() {
                    if name.to_string() == "Inv.inv" {
                        let a_term = parse_group_term(b, var_counter, var_map, vars, _config)?;
                        return Ok(a_term.inv());
                    }
                }
            }
            let key = format!("{expr:?}");
            let idx = *var_map.entry(key).or_insert_with(|| {
                let idx = *var_counter;
                *var_counter += 1;
                vars.push(expr.clone());
                idx
            });
            Ok(GroupTerm::single(idx, expr.clone()))
        }
        Expr::Const(name, _) if name.to_string().contains("one") || name.to_string() == "1" => {
            Ok(GroupTerm::identity())
        }
        Expr::Lit(lean5_kernel::Literal::Nat(1)) => Ok(GroupTerm::identity()),
        _ => {
            let key = format!("{expr:?}");
            let idx = *var_map.entry(key).or_insert_with(|| {
                let idx = *var_counter;
                *var_counter += 1;
                vars.push(expr.clone());
                idx
            });
            Ok(GroupTerm::single(idx, expr.clone()))
        }
    }
}
