//! Live expression AST for temporal formulas
//!
//! This module defines the internal representation for temporal formulas used
//! during liveness checking. It's distinct from the main AST because:
//! 1. It's already in positive normal form (negation pushed to atoms)
//! 2. It separates state predicates from action predicates
//! 3. It supports efficient evaluation during tableau construction
//!
//! Based on TLC's LiveExprNode hierarchy.

use std::sync::Arc;
use tla_core::ast::Expr;
use tla_core::Spanned;

/// Expression level for liveness formulas
///
/// Corresponds to TLA+ level constants:
/// - Constant: Can be evaluated without any state
/// - State: Depends on current state variables (state predicate)
/// - Action: Depends on current and next state (action predicate)
/// - Temporal: Contains temporal operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExprLevel {
    /// Constant expression (no state dependency)
    Constant = 0,
    /// State-level expression (depends on current state)
    State = 1,
    /// Action-level expression (depends on current and next state)
    Action = 2,
    /// Temporal expression (contains temporal operators)
    Temporal = 3,
}

/// A liveness expression (temporal formula in positive normal form)
///
/// This represents temporal formulas after conversion from the AST.
/// Negation is pushed down to atoms, so only positive temporal operators appear.
#[derive(Debug, Clone)]
pub enum LiveExpr {
    /// Boolean constant (TRUE or FALSE)
    Bool(bool),

    /// State predicate - evaluated on a single state
    /// Contains the original AST expression for evaluation
    StatePred {
        /// The AST expression to evaluate
        expr: Arc<Spanned<Expr>>,
        /// Unique tag for identifying this predicate during tableau construction
        tag: u32,
    },

    /// Action predicate - evaluated on a state transition (s, s')
    /// Used for fairness constraints and primed expressions
    ActionPred {
        /// The AST expression to evaluate
        expr: Arc<Spanned<Expr>>,
        /// Unique tag for identifying this predicate
        tag: u32,
    },

    /// ENABLED predicate - true if action is enabled in state
    Enabled {
        /// The action expression
        action: Arc<Spanned<Expr>>,
        /// If true, only count successors that change state (subscripted-action semantics).
        ///
        /// This is used for fairness constraints like `WF_vars(A)` which are defined in TLC
        /// in terms of `ENABLED(<<A>>_vars)`, where `<<A>>_vars == A /\ (vars' ≠ vars)`.
        require_state_change: bool,
        /// The subscript expression for subscripted action semantics.
        /// For `ENABLED<<A>>_e`, this is `e`. If None, uses global state change check.
        subscript: Option<Arc<Spanned<Expr>>>,
        /// Unique tag
        tag: u32,
    },

    /// Conjunction: P /\ Q /\ ...
    And(Vec<LiveExpr>),

    /// Disjunction: P \/ Q \/ ...
    Or(Vec<LiveExpr>),

    /// Negation: ~P
    /// In positive normal form, this only wraps atoms (Bool, StatePred, ActionPred)
    Not(Box<LiveExpr>),

    /// Always: []P
    Always(Box<LiveExpr>),

    /// Eventually: <>P
    Eventually(Box<LiveExpr>),

    /// Next: ()P (LTL next operator - used internally for tableau)
    /// Not part of TLA+ surface syntax but needed for tableau construction
    Next(Box<LiveExpr>),

    /// State changed predicate - true iff subscript' ≠ subscript (non-stuttering transition)
    /// Used to implement subscripted action semantics `<<A>>_e = A /\ (e' ≠ e)`
    StateChanged {
        /// The subscript expression to check for changes.
        /// For `<<A>>_e`, this is `e`. If None, uses global fingerprint comparison.
        subscript: Option<Arc<Spanned<Expr>>>,
        /// Unique tag for identifying this predicate
        tag: u32,
    },
}

impl LiveExpr {
    /// Create a true constant
    pub fn true_const() -> Self {
        LiveExpr::Bool(true)
    }

    /// Create a false constant
    pub fn false_const() -> Self {
        LiveExpr::Bool(false)
    }

    /// Create a state predicate
    pub fn state_pred(expr: Arc<Spanned<Expr>>, tag: u32) -> Self {
        LiveExpr::StatePred { expr, tag }
    }

    /// Create an action predicate
    pub fn action_pred(expr: Arc<Spanned<Expr>>, tag: u32) -> Self {
        LiveExpr::ActionPred { expr, tag }
    }

    /// Create an ENABLED predicate
    pub fn enabled(action: Arc<Spanned<Expr>>, tag: u32) -> Self {
        LiveExpr::Enabled {
            action,
            require_state_change: false,
            subscript: None,
            tag,
        }
    }

    /// Create an ENABLED predicate that requires a non-stuttering successor.
    /// The subscript expression is used to check for state change (e' ≠ e).
    pub fn enabled_subscripted(
        action: Arc<Spanned<Expr>>,
        subscript: Option<Arc<Spanned<Expr>>>,
        tag: u32,
    ) -> Self {
        LiveExpr::Enabled {
            action,
            require_state_change: true,
            subscript,
            tag,
        }
    }

    /// Create a state changed predicate (e' ≠ e for subscript e)
    /// If subscript is None, uses global fingerprint comparison.
    pub fn state_changed(subscript: Option<Arc<Spanned<Expr>>>, tag: u32) -> Self {
        LiveExpr::StateChanged { subscript, tag }
    }

    /// Create a conjunction
    pub fn and(exprs: Vec<LiveExpr>) -> Self {
        if exprs.is_empty() {
            LiveExpr::Bool(true)
        } else if exprs.len() == 1 {
            exprs.into_iter().next().unwrap()
        } else {
            LiveExpr::And(exprs)
        }
    }

    /// Create a disjunction
    pub fn or(exprs: Vec<LiveExpr>) -> Self {
        if exprs.is_empty() {
            LiveExpr::Bool(false)
        } else if exprs.len() == 1 {
            exprs.into_iter().next().unwrap()
        } else {
            LiveExpr::Or(exprs)
        }
    }

    /// Create a negation
    #[allow(clippy::should_implement_trait)]
    pub fn not(expr: LiveExpr) -> Self {
        match expr {
            LiveExpr::Bool(b) => LiveExpr::Bool(!b),
            LiveExpr::Not(inner) => *inner, // Double negation elimination
            _ => LiveExpr::Not(Box::new(expr)),
        }
    }

    /// Create an always expression
    pub fn always(expr: LiveExpr) -> Self {
        LiveExpr::Always(Box::new(expr))
    }

    /// Create an eventually expression
    pub fn eventually(expr: LiveExpr) -> Self {
        LiveExpr::Eventually(Box::new(expr))
    }

    /// Create a next expression
    pub fn next(expr: LiveExpr) -> Self {
        LiveExpr::Next(Box::new(expr))
    }

    /// Get the level of this expression
    pub fn level(&self) -> ExprLevel {
        match self {
            LiveExpr::Bool(_) => ExprLevel::Constant,
            LiveExpr::StatePred { .. } => ExprLevel::State,
            LiveExpr::ActionPred { .. } => ExprLevel::Action,
            LiveExpr::Enabled { .. } => ExprLevel::State, // ENABLED is state-level
            LiveExpr::StateChanged { .. } => ExprLevel::Action, // StateChanged is action-level (compares vars' to vars)
            LiveExpr::Not(e) => e.level(),
            LiveExpr::And(es) | LiveExpr::Or(es) => es
                .iter()
                .map(|e| e.level())
                .max()
                .unwrap_or(ExprLevel::Constant),
            LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => {
                ExprLevel::Temporal
            }
        }
    }

    /// Check if this expression contains any action-level subexpressions
    pub fn contains_action(&self) -> bool {
        match self {
            LiveExpr::Bool(_) | LiveExpr::StatePred { .. } | LiveExpr::Enabled { .. } => false,
            LiveExpr::ActionPred { .. } | LiveExpr::StateChanged { .. } => true,
            LiveExpr::Not(e) => e.contains_action(),
            LiveExpr::And(es) | LiveExpr::Or(es) => es.iter().any(|e| e.contains_action()),
            LiveExpr::Always(e) | LiveExpr::Eventually(e) | LiveExpr::Next(e) => {
                e.contains_action()
            }
        }
    }

    /// Check if this expression is in positive normal form
    /// (negation only applied to atoms)
    pub fn is_positive_form(&self) -> bool {
        match self {
            LiveExpr::Bool(_)
            | LiveExpr::StatePred { .. }
            | LiveExpr::ActionPred { .. }
            | LiveExpr::Enabled { .. }
            | LiveExpr::StateChanged { .. } => true,

            LiveExpr::Not(inner) => matches!(
                inner.as_ref(),
                LiveExpr::Bool(_)
                    | LiveExpr::StatePred { .. }
                    | LiveExpr::ActionPred { .. }
                    | LiveExpr::Enabled { .. }
                    | LiveExpr::StateChanged { .. }
            ),

            LiveExpr::And(es) | LiveExpr::Or(es) => es.iter().all(|e| e.is_positive_form()),

            LiveExpr::Always(e) | LiveExpr::Eventually(e) | LiveExpr::Next(e) => {
                e.is_positive_form()
            }
        }
    }

    /// Push negation down to atoms (convert to positive normal form)
    ///
    /// Uses the following rewriting rules (Manna & Pnueli, p. 452):
    /// - ~TRUE = FALSE
    /// - ~FALSE = TRUE
    /// - ~~P = P
    /// - ~(P /\ Q) = ~P \/ ~Q
    /// - ~(P \/ Q) = ~P /\ ~Q
    /// - ~[]P = <>~P
    /// - ~<>P = []~P
    /// - ~()P = ()~P  (next distributes over negation)
    pub fn push_negation(self) -> Self {
        self.push_neg_inner(false)
    }

    fn push_neg_inner(self, negate: bool) -> Self {
        if negate {
            match self {
                LiveExpr::Bool(b) => LiveExpr::Bool(!b),

                LiveExpr::StatePred { .. }
                | LiveExpr::ActionPred { .. }
                | LiveExpr::Enabled { .. }
                | LiveExpr::StateChanged { .. } => LiveExpr::Not(Box::new(self)),

                LiveExpr::Not(inner) => inner.push_neg_inner(false),

                LiveExpr::And(es) => {
                    // ~(P /\ Q) = ~P \/ ~Q
                    LiveExpr::Or(es.into_iter().map(|e| e.push_neg_inner(true)).collect())
                }

                LiveExpr::Or(es) => {
                    // ~(P \/ Q) = ~P /\ ~Q
                    LiveExpr::And(es.into_iter().map(|e| e.push_neg_inner(true)).collect())
                }

                LiveExpr::Always(e) => {
                    // ~[]P = <>~P
                    LiveExpr::Eventually(Box::new(e.push_neg_inner(true)))
                }

                LiveExpr::Eventually(e) => {
                    // ~<>P = []~P
                    LiveExpr::Always(Box::new(e.push_neg_inner(true)))
                }

                LiveExpr::Next(e) => {
                    // ~()P = ()~P
                    LiveExpr::Next(Box::new(e.push_neg_inner(true)))
                }
            }
        } else {
            match self {
                LiveExpr::Bool(_)
                | LiveExpr::StatePred { .. }
                | LiveExpr::ActionPred { .. }
                | LiveExpr::Enabled { .. }
                | LiveExpr::StateChanged { .. } => self,

                LiveExpr::Not(inner) => inner.push_neg_inner(true),

                LiveExpr::And(es) => {
                    LiveExpr::And(es.into_iter().map(|e| e.push_neg_inner(false)).collect())
                }

                LiveExpr::Or(es) => {
                    LiveExpr::Or(es.into_iter().map(|e| e.push_neg_inner(false)).collect())
                }

                LiveExpr::Always(e) => LiveExpr::Always(Box::new(e.push_neg_inner(false))),

                LiveExpr::Eventually(e) => LiveExpr::Eventually(Box::new(e.push_neg_inner(false))),

                LiveExpr::Next(e) => LiveExpr::Next(Box::new(e.push_neg_inner(false))),
            }
        }
    }

    /// Get the body if this is of the form []<>A (AE pattern - "always eventually")
    /// Returns None if not in this form
    pub fn get_ae_body(&self) -> Option<&LiveExpr> {
        if let LiveExpr::Always(inner) = self {
            if let LiveExpr::Eventually(body) = inner.as_ref() {
                return Some(body);
            }
        }
        None
    }

    /// Get the body if this is of the form <>[]A (EA pattern - "eventually always")
    /// Returns None if not in this form
    pub fn get_ea_body(&self) -> Option<&LiveExpr> {
        if let LiveExpr::Eventually(inner) = self {
            if let LiveExpr::Always(body) = inner.as_ref() {
                return Some(body);
            }
        }
        None
    }

    /// Check if this is a general temporal formula (not []<> or <>[] at top level)
    pub fn is_general_tf(&self) -> bool {
        self.get_ae_body().is_none() && self.get_ea_body().is_none()
    }

    /// Recursively extract nested []<> (AE) patterns from within this formula.
    ///
    /// For formulas like `<>(P /\ []<>Q)`, this extracts the `[]<>Q` pattern
    /// and returns the formula with that pattern replaced by `true`.
    ///
    /// Returns: `(extracted_ae_bodies, simplified_formula)`
    /// - `extracted_ae_bodies`: Bodies of all []<> patterns found (e.g., `Q`)
    /// - `simplified_formula`: The formula with []<> patterns replaced by true
    ///
    /// This enables proper handling of nested temporal patterns like TLC does:
    /// the []<> obligations are checked via AE constraints (infinitely often),
    /// while the remaining formula goes to the tableau.
    pub fn extract_nested_ae(&self) -> (Vec<LiveExpr>, LiveExpr) {
        let mut ae_bodies = Vec::new();
        let simplified = self.extract_nested_ae_inner(&mut ae_bodies);
        (ae_bodies, simplified)
    }

    fn extract_nested_ae_inner(&self, ae_bodies: &mut Vec<LiveExpr>) -> LiveExpr {
        match self {
            // If this is a []<> pattern, extract the body and return true
            LiveExpr::Always(inner) => {
                if let LiveExpr::Eventually(body) = inner.as_ref() {
                    // Only extract if body is state/constant level (not temporal)
                    if body.level() != ExprLevel::Temporal {
                        if !ae_bodies.iter().any(|e| e.structurally_equal(body)) {
                            ae_bodies.push((**body).clone());
                        }
                        return LiveExpr::Bool(true);
                    }
                }
                // Recurse into Always
                LiveExpr::Always(Box::new(inner.extract_nested_ae_inner(ae_bodies)))
            }

            // Recurse into other temporal operators
            LiveExpr::Eventually(inner) => {
                LiveExpr::Eventually(Box::new(inner.extract_nested_ae_inner(ae_bodies)))
            }
            LiveExpr::Next(inner) => {
                LiveExpr::Next(Box::new(inner.extract_nested_ae_inner(ae_bodies)))
            }
            LiveExpr::Not(inner) => {
                LiveExpr::Not(Box::new(inner.extract_nested_ae_inner(ae_bodies)))
            }

            // Recurse into boolean connectives
            LiveExpr::And(conjuncts) => {
                let simplified: Vec<_> = conjuncts
                    .iter()
                    .map(|c| c.extract_nested_ae_inner(ae_bodies))
                    .collect();
                // Simplify: remove trues from conjunction
                let filtered: Vec<_> = simplified
                    .into_iter()
                    .filter(|e| !matches!(e, LiveExpr::Bool(true)))
                    .collect();
                if filtered.is_empty() {
                    LiveExpr::Bool(true)
                } else if filtered.len() == 1 {
                    filtered.into_iter().next().unwrap()
                } else {
                    LiveExpr::And(filtered)
                }
            }
            LiveExpr::Or(disjuncts) => {
                let simplified: Vec<_> = disjuncts
                    .iter()
                    .map(|d| d.extract_nested_ae_inner(ae_bodies))
                    .collect();
                LiveExpr::Or(simplified)
            }

            // Atoms: return unchanged
            LiveExpr::Bool(_)
            | LiveExpr::StatePred { .. }
            | LiveExpr::ActionPred { .. }
            | LiveExpr::Enabled { .. }
            | LiveExpr::StateChanged { .. } => self.clone(),
        }
    }

    /// Recursively extract []body patterns from within <> contexts.
    ///
    /// For formulas like `<>(P /\ []~Q)` (leads-to violations), this extracts
    /// the `[]~Q` pattern and returns the formula with that pattern replaced by `true`.
    ///
    /// Returns: `(extracted_ea_bodies, simplified_formula)`
    /// - `extracted_ea_bodies`: Bodies of all [] patterns found inside <> (e.g., `~Q`)
    /// - `simplified_formula`: The formula with [] patterns replaced by true
    ///
    /// This enables proper handling of leads-to violations like TLC does:
    /// the [] obligations inside <> are checked via EA constraints (eventually always),
    /// which filter the behavior graph edges to only include transitions where
    /// the body holds on both endpoints.
    pub fn extract_nested_ea(&self) -> (Vec<LiveExpr>, LiveExpr) {
        let mut ea_bodies = Vec::new();
        let simplified = self.extract_nested_ea_outer(&mut ea_bodies);
        (ea_bodies, simplified)
    }

    /// Outer extraction: not inside <> yet, looking for <> to enter
    fn extract_nested_ea_outer(&self, ea_bodies: &mut Vec<LiveExpr>) -> LiveExpr {
        match self {
            // When we hit <>, switch to inner extraction mode
            LiveExpr::Eventually(inner) => {
                // Check for top-level <>[] pattern first
                if let LiveExpr::Always(body) = inner.as_ref() {
                    if body.level() != ExprLevel::Temporal {
                        if !ea_bodies.iter().any(|e| e.structurally_equal(body)) {
                            ea_bodies.push((**body).clone());
                        }
                        return LiveExpr::Bool(true);
                    }
                }
                // Otherwise, extract [] patterns from inside the <>
                LiveExpr::Eventually(Box::new(inner.extract_nested_ea_inner(ea_bodies)))
            }

            // Recurse into other temporal operators
            LiveExpr::Always(inner) => {
                LiveExpr::Always(Box::new(inner.extract_nested_ea_outer(ea_bodies)))
            }
            LiveExpr::Next(inner) => {
                LiveExpr::Next(Box::new(inner.extract_nested_ea_outer(ea_bodies)))
            }
            LiveExpr::Not(inner) => {
                LiveExpr::Not(Box::new(inner.extract_nested_ea_outer(ea_bodies)))
            }

            // Recurse into boolean connectives
            LiveExpr::And(conjuncts) => {
                let simplified: Vec<_> = conjuncts
                    .iter()
                    .map(|c| c.extract_nested_ea_outer(ea_bodies))
                    .collect();
                // Simplify: remove trues from conjunction
                let filtered: Vec<_> = simplified
                    .into_iter()
                    .filter(|e| !matches!(e, LiveExpr::Bool(true)))
                    .collect();
                if filtered.is_empty() {
                    LiveExpr::Bool(true)
                } else if filtered.len() == 1 {
                    filtered.into_iter().next().unwrap()
                } else {
                    LiveExpr::And(filtered)
                }
            }
            LiveExpr::Or(disjuncts) => {
                let simplified: Vec<_> = disjuncts
                    .iter()
                    .map(|d| d.extract_nested_ea_outer(ea_bodies))
                    .collect();
                LiveExpr::Or(simplified)
            }

            // Atoms: return unchanged
            LiveExpr::Bool(_)
            | LiveExpr::StatePred { .. }
            | LiveExpr::ActionPred { .. }
            | LiveExpr::Enabled { .. }
            | LiveExpr::StateChanged { .. } => self.clone(),
        }
    }

    /// Inner extraction: we're inside a <>, so extract any [] patterns
    fn extract_nested_ea_inner(&self, ea_bodies: &mut Vec<LiveExpr>) -> LiveExpr {
        match self {
            // If this is a [] pattern inside <>, extract the body
            LiveExpr::Always(body) => {
                // Only extract if body is state/constant level (not temporal)
                if body.level() != ExprLevel::Temporal {
                    if !ea_bodies.iter().any(|e| e.structurally_equal(body)) {
                        ea_bodies.push((**body).clone());
                    }
                    return LiveExpr::Bool(true);
                }
                // If body is temporal, recurse
                LiveExpr::Always(Box::new(body.extract_nested_ea_inner(ea_bodies)))
            }

            // Nested <> - continue inner extraction
            LiveExpr::Eventually(inner) => {
                LiveExpr::Eventually(Box::new(inner.extract_nested_ea_inner(ea_bodies)))
            }
            LiveExpr::Next(inner) => {
                LiveExpr::Next(Box::new(inner.extract_nested_ea_inner(ea_bodies)))
            }
            LiveExpr::Not(inner) => {
                LiveExpr::Not(Box::new(inner.extract_nested_ea_inner(ea_bodies)))
            }

            // Recurse into boolean connectives
            LiveExpr::And(conjuncts) => {
                let simplified: Vec<_> = conjuncts
                    .iter()
                    .map(|c| c.extract_nested_ea_inner(ea_bodies))
                    .collect();
                // Simplify: remove trues from conjunction
                let filtered: Vec<_> = simplified
                    .into_iter()
                    .filter(|e| !matches!(e, LiveExpr::Bool(true)))
                    .collect();
                if filtered.is_empty() {
                    LiveExpr::Bool(true)
                } else if filtered.len() == 1 {
                    filtered.into_iter().next().unwrap()
                } else {
                    LiveExpr::And(filtered)
                }
            }
            LiveExpr::Or(disjuncts) => {
                let simplified: Vec<_> = disjuncts
                    .iter()
                    .map(|d| d.extract_nested_ea_inner(ea_bodies))
                    .collect();
                LiveExpr::Or(simplified)
            }

            // Atoms: return unchanged
            LiveExpr::Bool(_)
            | LiveExpr::StatePred { .. }
            | LiveExpr::ActionPred { .. }
            | LiveExpr::Enabled { .. }
            | LiveExpr::StateChanged { .. } => self.clone(),
        }
    }

    // NOTE: eval_state and eval_action methods will be implemented in a future
    // iteration when we integrate the tableau with the behavior graph.
    // For now, tableau construction doesn't require evaluation - we just need
    // to build the graph structure.

    /// Structural equality check (for tableau construction)
    pub fn structurally_equal(&self, other: &LiveExpr) -> bool {
        match (self, other) {
            (LiveExpr::Bool(a), LiveExpr::Bool(b)) => a == b,
            (LiveExpr::StatePred { tag: t1, .. }, LiveExpr::StatePred { tag: t2, .. }) => t1 == t2,
            (LiveExpr::ActionPred { tag: t1, .. }, LiveExpr::ActionPred { tag: t2, .. }) => {
                t1 == t2
            }
            (LiveExpr::Enabled { tag: t1, .. }, LiveExpr::Enabled { tag: t2, .. }) => t1 == t2,
            (LiveExpr::StateChanged { tag: t1, .. }, LiveExpr::StateChanged { tag: t2, .. }) => {
                t1 == t2
            }
            (LiveExpr::Not(a), LiveExpr::Not(b)) => a.structurally_equal(b),
            (LiveExpr::And(as_), LiveExpr::And(bs)) if as_.len() == bs.len() => as_
                .iter()
                .zip(bs.iter())
                .all(|(a, b)| a.structurally_equal(b)),
            (LiveExpr::Or(as_), LiveExpr::Or(bs)) if as_.len() == bs.len() => as_
                .iter()
                .zip(bs.iter())
                .all(|(a, b)| a.structurally_equal(b)),
            (LiveExpr::Always(a), LiveExpr::Always(b)) => a.structurally_equal(b),
            (LiveExpr::Eventually(a), LiveExpr::Eventually(b)) => a.structurally_equal(b),
            (LiveExpr::Next(a), LiveExpr::Next(b)) => a.structurally_equal(b),
            _ => false,
        }
    }

    /// Convert this formula to disjunctive normal form (DNF).
    ///
    /// This treats temporal operators ([], <>, ()) as atomic for the purpose of DNF;
    /// it only distributes over explicit boolean connectives (/\ and \/).
    ///
    /// The returned value is a disjunction of conjunction clauses:
    /// - outer `Vec`: disjuncts
    /// - inner `Vec`: conjuncts within a disjunct
    pub fn to_dnf_clauses(&self) -> Vec<Vec<LiveExpr>> {
        match self {
            LiveExpr::Bool(true) => vec![vec![]],
            LiveExpr::Bool(false) => vec![],

            LiveExpr::And(conjuncts) => {
                let mut clauses: Vec<Vec<LiveExpr>> = vec![vec![]];
                for conjunct in conjuncts {
                    let conjunct_clauses = conjunct.to_dnf_clauses();
                    if conjunct_clauses.is_empty() {
                        return vec![];
                    }

                    let mut next_clauses = Vec::new();
                    for base in &clauses {
                        for add in &conjunct_clauses {
                            let mut merged = base.clone();
                            merged.extend(add.iter().cloned());
                            next_clauses.push(merged);
                        }
                    }
                    clauses = next_clauses;
                }
                clauses
            }

            LiveExpr::Or(disjuncts) => {
                let mut clauses = Vec::new();
                for disjunct in disjuncts {
                    clauses.extend(disjunct.to_dnf_clauses());
                }
                clauses
            }

            _ => vec![vec![self.clone()]],
        }
    }

    /// Extract all "promise" subformulas from this formula.
    ///
    /// In TLC terms, promises are all subformulas of the form `<>r`.
    /// The tableau acceptance check requires that each promise is fulfilled
    /// somewhere in a cycle.
    pub fn extract_promises(&self) -> Vec<LiveExpr> {
        fn go(expr: &LiveExpr, out: &mut Vec<LiveExpr>) {
            match expr {
                LiveExpr::Eventually(inner) => {
                    let promise = LiveExpr::Eventually(Box::new((**inner).clone()));
                    if !out.iter().any(|p| p.structurally_equal(&promise)) {
                        out.push(promise);
                    }
                    go(inner, out);
                }
                LiveExpr::Not(inner) | LiveExpr::Always(inner) | LiveExpr::Next(inner) => {
                    go(inner, out);
                }
                LiveExpr::And(es) | LiveExpr::Or(es) => {
                    for e in es {
                        go(e, out);
                    }
                }
                LiveExpr::Bool(_)
                | LiveExpr::StatePred { .. }
                | LiveExpr::ActionPred { .. }
                | LiveExpr::Enabled { .. }
                | LiveExpr::StateChanged { .. } => {}
            }
        }

        let mut promises = Vec::new();
        go(self, &mut promises);
        promises
    }
}

impl std::fmt::Display for LiveExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveExpr::Bool(true) => write!(f, "TRUE"),
            LiveExpr::Bool(false) => write!(f, "FALSE"),
            LiveExpr::StatePred { tag, .. } => write!(f, "S{}", tag),
            LiveExpr::ActionPred { tag, .. } => write!(f, "A{}", tag),
            LiveExpr::Enabled { tag, .. } => write!(f, "ENABLED({})", tag),
            LiveExpr::StateChanged { tag, .. } => write!(f, "CHANGED({})", tag),
            LiveExpr::Not(e) => write!(f, "~{}", e),
            LiveExpr::And(es) => {
                write!(f, "(")?;
                for (i, e) in es.iter().enumerate() {
                    if i > 0 {
                        write!(f, " /\\ ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            LiveExpr::Or(es) => {
                write!(f, "(")?;
                for (i, e) in es.iter().enumerate() {
                    if i > 0 {
                        write!(f, " \\/ ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            LiveExpr::Always(e) => write!(f, "[]{}", e),
            LiveExpr::Eventually(e) => write!(f, "<>{}", e),
            LiveExpr::Next(e) => write!(f, "(){}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_negation_bool() {
        let t = LiveExpr::Bool(true);
        let nt = LiveExpr::not(t.clone()).push_negation();
        assert!(matches!(nt, LiveExpr::Bool(false)));

        let f = LiveExpr::Bool(false);
        let nf = LiveExpr::not(f.clone()).push_negation();
        assert!(matches!(nf, LiveExpr::Bool(true)));
    }

    #[test]
    fn test_push_negation_double() {
        let p = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 1,
        };
        let nnp = LiveExpr::not(LiveExpr::not(p.clone())).push_negation();
        // ~~P = P (double negation elimination)
        assert!(matches!(nnp, LiveExpr::StatePred { tag: 1, .. }));
    }

    #[test]
    fn test_push_negation_de_morgan() {
        let p = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 1,
        };
        let q = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 2,
        };

        // ~(P /\ Q) = ~P \/ ~Q
        let and_expr = LiveExpr::And(vec![p.clone(), q.clone()]);
        let neg_and = LiveExpr::not(and_expr).push_negation();
        assert!(matches!(neg_and, LiveExpr::Or(_)));

        // ~(P \/ Q) = ~P /\ ~Q
        let or_expr = LiveExpr::Or(vec![p.clone(), q.clone()]);
        let neg_or = LiveExpr::not(or_expr).push_negation();
        assert!(matches!(neg_or, LiveExpr::And(_)));
    }

    #[test]
    fn test_push_negation_temporal() {
        let p = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 1,
        };

        // ~[]P = <>~P
        let always_p = LiveExpr::always(p.clone());
        let neg_always = LiveExpr::not(always_p).push_negation();
        assert!(matches!(neg_always, LiveExpr::Eventually(_)));

        // ~<>P = []~P
        let eventually_p = LiveExpr::eventually(p.clone());
        let neg_eventually = LiveExpr::not(eventually_p).push_negation();
        assert!(matches!(neg_eventually, LiveExpr::Always(_)));
    }

    #[test]
    fn test_is_positive_form() {
        let p = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 1,
        };

        // Atoms are in positive form
        assert!(p.is_positive_form());
        assert!(LiveExpr::Bool(true).is_positive_form());

        // Negated atoms are in positive form
        assert!(LiveExpr::not(p.clone()).is_positive_form());

        // Negated temporal operators are NOT in positive form
        let always_p = LiveExpr::always(p.clone());
        let neg_always = LiveExpr::not(always_p);
        assert!(!neg_always.is_positive_form());

        // After push_negation, should be in positive form
        let normalized = neg_always.push_negation();
        assert!(normalized.is_positive_form());
    }

    #[test]
    fn test_ae_ea_patterns() {
        let p = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 1,
        };

        // []<>P (always eventually)
        let ae = LiveExpr::always(LiveExpr::eventually(p.clone()));
        assert!(ae.get_ae_body().is_some());
        assert!(ae.get_ea_body().is_none());
        assert!(!ae.is_general_tf());

        // <>[]P (eventually always)
        let ea = LiveExpr::eventually(LiveExpr::always(p.clone()));
        assert!(ea.get_ea_body().is_some());
        assert!(ea.get_ae_body().is_none());
        assert!(!ea.is_general_tf());

        // []P (just always - general)
        let just_always = LiveExpr::always(p.clone());
        assert!(just_always.get_ae_body().is_none());
        assert!(just_always.get_ea_body().is_none());
        assert!(just_always.is_general_tf());
    }

    #[test]
    fn test_level() {
        let state_pred = LiveExpr::StatePred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 1,
        };
        let action_pred = LiveExpr::ActionPred {
            expr: Arc::new(Spanned::dummy(Expr::Bool(true))),
            tag: 2,
        };

        assert_eq!(LiveExpr::Bool(true).level(), ExprLevel::Constant);
        assert_eq!(state_pred.level(), ExprLevel::State);
        assert_eq!(action_pred.level(), ExprLevel::Action);
        assert_eq!(
            LiveExpr::always(state_pred.clone()).level(),
            ExprLevel::Temporal
        );

        // Conjunction takes max level
        let and_expr = LiveExpr::And(vec![state_pred, action_pred]);
        assert_eq!(and_expr.level(), ExprLevel::Action);
    }
}
