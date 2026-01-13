//! Unification algorithm
//!
//! Handles metavariable instantiation and constraint solving.

use hashbrown::HashMap;
use lean5_kernel::{Expr, FVarId};

/// Unique identifier for metavariables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetaId(pub u64);

/// A metavariable
#[derive(Debug, Clone)]
pub struct MetaVar {
    /// The type of the metavariable
    pub ty: Expr,
    /// The assigned value (if any)
    pub assignment: Option<Expr>,
}

/// State for metavariable management
#[derive(Debug, Clone)]
pub struct MetaState {
    /// All metavariables
    metas: HashMap<MetaId, MetaVar>,
    /// Next fresh metavariable id
    next_id: u64,
}

impl MetaState {
    /// High-bit tag to ensure metavariable FVars don't collide with user locals
    const META_FVAR_TAG: u64 = 1 << 63;

    pub fn new() -> Self {
        Self {
            metas: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create a fresh metavariable with the given type
    pub fn fresh(&mut self, ty: Expr) -> MetaId {
        let id = MetaId(self.next_id);
        self.next_id += 1;
        self.metas.insert(
            id,
            MetaVar {
                ty,
                assignment: None,
            },
        );
        id
    }

    /// Convert a metavariable id into the FVarId used in expressions
    pub fn to_fvar(id: MetaId) -> FVarId {
        FVarId(id.0 | Self::META_FVAR_TAG)
    }

    /// Try to decode a metavariable id from a free variable
    pub fn from_fvar(id: FVarId) -> Option<MetaId> {
        if id.0 & Self::META_FVAR_TAG != 0 {
            Some(MetaId(id.0 & !Self::META_FVAR_TAG))
        } else {
            None
        }
    }

    /// Get a metavariable by id
    pub fn get(&self, id: MetaId) -> Option<&MetaVar> {
        self.metas.get(&id)
    }

    /// Assign a value to a metavariable
    pub fn assign(&mut self, id: MetaId, val: Expr) -> bool {
        if let Some(meta) = self.metas.get_mut(&id) {
            if meta.assignment.is_none() {
                meta.assignment = Some(val);
                return true;
            }
        }
        false
    }

    /// Check if a metavariable is assigned
    pub fn is_assigned(&self, id: MetaId) -> bool {
        self.metas.get(&id).is_some_and(|m| m.assignment.is_some())
    }

    /// Get the assignment of a metavariable
    pub fn get_assignment(&self, id: MetaId) -> Option<&Expr> {
        self.metas.get(&id).and_then(|m| m.assignment.as_ref())
    }

    /// Get all unassigned metavariables
    pub fn unassigned(&self) -> Vec<MetaId> {
        self.metas
            .iter()
            .filter(|(_, m)| m.assignment.is_none())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Instantiate metavariables in an expression
    pub fn instantiate(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::FVar(id) => {
                if let Some(meta_id) = Self::from_fvar(*id) {
                    if let Some(meta) = self.get(meta_id) {
                        if let Some(val) = &meta.assignment {
                            return self.instantiate(val);
                        }
                    }
                }
                expr.clone()
            }
            Expr::App(f, a) => Expr::app(self.instantiate(f), self.instantiate(a)),
            Expr::Lam(bi, ty, body) => Expr::lam(*bi, self.instantiate(ty), self.instantiate(body)),
            Expr::Pi(bi, ty, body) => Expr::pi(*bi, self.instantiate(ty), self.instantiate(body)),
            Expr::Let(ty, val, body) => Expr::let_(
                self.instantiate(ty),
                self.instantiate(val),
                self.instantiate(body),
            ),
            Expr::Proj(name, idx, e) => Expr::proj(name.clone(), *idx, self.instantiate(e)),
            _ => expr.clone(),
        }
    }

    /// Check whether a metavariable occurs in an expression (after instantiation)
    pub fn occurs(&self, meta: MetaId, expr: &Expr) -> bool {
        match self.instantiate(expr) {
            Expr::FVar(id) => Self::from_fvar(id) == Some(meta),
            Expr::App(f, a) => self.occurs(meta, &f) || self.occurs(meta, &a),
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                self.occurs(meta, &ty) || self.occurs(meta, &body)
            }
            Expr::Let(ty, val, body) => {
                self.occurs(meta, &ty) || self.occurs(meta, &val) || self.occurs(meta, &body)
            }
            Expr::Proj(_, _, e) => self.occurs(meta, &e),
            _ => false,
        }
    }

    /// Iterate over all metavariables
    pub fn iter(&self) -> impl Iterator<Item = (MetaId, &MetaVar)> {
        self.metas.iter().map(|(id, meta)| (*id, meta))
    }
}

impl Default for MetaState {
    fn default() -> Self {
        Self::new()
    }
}

/// Unification result
#[derive(Debug)]
pub enum UnifyResult {
    /// Unification succeeded
    Success,
    /// Unification failed
    Failure(String),
    /// Unification is stuck (waiting for more information)
    Stuck,
}

/// Unifier for constraint solving
pub struct Unifier<'a> {
    metas: &'a mut MetaState,
}

impl<'a> Unifier<'a> {
    pub fn new(metas: &'a mut MetaState) -> Self {
        Self { metas }
    }

    /// Check if expression is an unsolved metavariable
    fn as_meta(&self, expr: &Expr) -> Option<MetaId> {
        if let Expr::FVar(id) = expr {
            if let Some(meta_id) = MetaState::from_fvar(*id) {
                if self.metas.get(meta_id).is_some() {
                    return Some(meta_id);
                }
            }
        }
        None
    }

    fn unify_meta(&mut self, meta_id: MetaId, other: &Expr) -> UnifyResult {
        let other = self.metas.instantiate(other);
        if self.metas.occurs(meta_id, &other) {
            return UnifyResult::Failure(format!("occurs check failed for {meta_id:?}"));
        }

        if let Some(existing) = self.metas.get_assignment(meta_id).cloned() {
            return self.unify_core(&existing, &other);
        }

        self.metas.assign(meta_id, other);
        UnifyResult::Success
    }

    /// Try to unify two expressions
    pub fn unify(&mut self, left: &Expr, right: &Expr) -> UnifyResult {
        // Instantiate any assigned metavariables
        let left = self.metas.instantiate(left);
        let right = self.metas.instantiate(right);

        self.unify_core(&left, &right)
    }

    fn unify_core(&mut self, left: &Expr, right: &Expr) -> UnifyResult {
        // If they're already equal, we're done
        if left == right {
            return UnifyResult::Success;
        }

        if let Some(meta_id) = self.as_meta(left) {
            return self.unify_meta(meta_id, right);
        }

        if let Some(meta_id) = self.as_meta(right) {
            return self.unify_meta(meta_id, left);
        }

        match (left, right) {
            // Both are sorts - check level equality
            (Expr::Sort(l1), Expr::Sort(l2)) => {
                if l1 == l2 {
                    UnifyResult::Success
                } else {
                    UnifyResult::Failure(format!("level mismatch: {l1:?} vs {l2:?}"))
                }
            }

            // Both are constants
            (Expr::Const(n1, ls1), Expr::Const(n2, ls2)) => {
                if n1 == n2 && ls1.len() == ls2.len() {
                    // Check universe levels match
                    for (l1, l2) in ls1.iter().zip(ls2.iter()) {
                        if l1 != l2 {
                            return UnifyResult::Failure(format!(
                                "level mismatch in const: {l1:?} vs {l2:?}"
                            ));
                        }
                    }
                    UnifyResult::Success
                } else {
                    UnifyResult::Failure(format!("const mismatch: {n1:?} vs {n2:?}"))
                }
            }

            // Both are bound variables
            (Expr::BVar(i1), Expr::BVar(i2)) => {
                if i1 == i2 {
                    UnifyResult::Success
                } else {
                    UnifyResult::Failure(format!("bvar mismatch: {i1} vs {i2}"))
                }
            }

            // Both are free variables (non-metavars at this point)
            (Expr::FVar(id1), Expr::FVar(id2)) => {
                if id1 == id2 {
                    UnifyResult::Success
                } else {
                    UnifyResult::Failure(format!("fvar mismatch: {id1:?} vs {id2:?}"))
                }
            }

            // Application
            (Expr::App(f1, a1), Expr::App(f2, a2)) => {
                match self.unify_core(f1, f2) {
                    UnifyResult::Success => {}
                    other => return other,
                }
                self.unify_core(a1, a2)
            }

            // Lambda and Pi
            (Expr::Lam(bi1, ty1, body1), Expr::Lam(bi2, ty2, body2))
            | (Expr::Pi(bi1, ty1, body1), Expr::Pi(bi2, ty2, body2)) => {
                if bi1 != bi2 {
                    return UnifyResult::Failure("binder info mismatch".to_string());
                }
                match self.unify_core(ty1, ty2) {
                    UnifyResult::Success => {}
                    other => return other,
                }
                self.unify_core(body1, body2)
            }

            // Let
            (Expr::Let(ty1, val1, body1), Expr::Let(ty2, val2, body2)) => {
                match self.unify_core(ty1, ty2) {
                    UnifyResult::Success => {}
                    other => return other,
                }
                match self.unify_core(val1, val2) {
                    UnifyResult::Success => {}
                    other => return other,
                }
                self.unify_core(body1, body2)
            }

            // Literals
            (Expr::Lit(l1), Expr::Lit(l2)) => {
                if l1 == l2 {
                    UnifyResult::Success
                } else {
                    UnifyResult::Failure(format!("literal mismatch: {l1:?} vs {l2:?}"))
                }
            }

            // Projection
            (Expr::Proj(n1, i1, e1), Expr::Proj(n2, i2, e2)) => {
                if n1 != n2 || i1 != i2 {
                    return UnifyResult::Failure(format!(
                        "projection mismatch: {n1}.{i1} vs {n2}.{i2}"
                    ));
                }
                self.unify_core(e1, e2)
            }

            // Default: no match
            _ => UnifyResult::Failure(format!(
                "cannot unify expressions of different shape: {:?} vs {:?}",
                std::mem::discriminant(left),
                std::mem::discriminant(right)
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_state_fresh() {
        let mut state = MetaState::new();
        let id1 = state.fresh(Expr::type_());
        let id2 = state.fresh(Expr::prop());
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_meta_assignment() {
        let mut state = MetaState::new();
        let id = state.fresh(Expr::type_());

        assert!(!state.is_assigned(id));
        assert!(state.assign(id, Expr::prop()));
        assert!(state.is_assigned(id));
        assert_eq!(state.get_assignment(id), Some(&Expr::prop()));
    }

    #[test]
    fn test_unify_same() {
        let mut state = MetaState::new();
        let mut unifier = Unifier::new(&mut state);

        let result = unifier.unify(&Expr::type_(), &Expr::type_());
        assert!(matches!(result, UnifyResult::Success));
    }

    #[test]
    fn test_unify_different() {
        let mut state = MetaState::new();
        let mut unifier = Unifier::new(&mut state);

        let result = unifier.unify(&Expr::type_(), &Expr::prop());
        assert!(matches!(result, UnifyResult::Failure(_)));
    }

    #[test]
    fn test_unify_app() {
        let mut state = MetaState::new();
        let mut unifier = Unifier::new(&mut state);

        let app1 = Expr::app(Expr::BVar(0), Expr::BVar(1));
        let app2 = Expr::app(Expr::BVar(0), Expr::BVar(1));
        let result = unifier.unify(&app1, &app2);
        assert!(matches!(result, UnifyResult::Success));

        let app3 = Expr::app(Expr::BVar(0), Expr::BVar(2));
        let result = unifier.unify(&app1, &app3);
        assert!(matches!(result, UnifyResult::Failure(_)));
    }

    #[test]
    fn test_unify_assigns_meta() {
        let mut state = MetaState::new();
        let meta_id = state.fresh(Expr::type_());
        let meta_expr = Expr::FVar(MetaState::to_fvar(meta_id));

        let mut unifier = Unifier::new(&mut state);
        let result = unifier.unify(&meta_expr, &Expr::prop());
        assert!(matches!(result, UnifyResult::Success));

        // Assignment should be stored and instantiation should replace the metavariable
        assert_eq!(state.get_assignment(meta_id), Some(&Expr::prop()));
        assert_eq!(state.instantiate(&meta_expr), Expr::prop());
    }

    #[test]
    fn test_occurs_check_blocks_self_reference() {
        let mut state = MetaState::new();
        let meta_id = state.fresh(Expr::type_());
        let meta_expr = Expr::FVar(MetaState::to_fvar(meta_id));

        // Try to unify ?m with (?m) applied to Prop, which should fail occurs check
        let mut unifier = Unifier::new(&mut state);
        let bad = Expr::app(meta_expr.clone(), Expr::prop());
        let result = unifier.unify(&meta_expr, &bad);
        assert!(matches!(result, UnifyResult::Failure(_)));
    }
}
