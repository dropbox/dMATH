//! Expression types for CHC formulas

// These constructors build AST nodes, not perform operations.
// Implementing std::ops traits would be semantically incorrect.
#![allow(clippy::should_implement_trait)]

use std::fmt;
use std::sync::Arc;

/// Sort (type) of expressions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChcSort {
    Bool,
    Int,
    Real,
    BitVec(u32),
    /// Array sort: (Array key_sort value_sort)
    Array(Box<ChcSort>, Box<ChcSort>),
}

impl fmt::Display for ChcSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChcSort::Bool => write!(f, "Bool"),
            ChcSort::Int => write!(f, "Int"),
            ChcSort::Real => write!(f, "Real"),
            ChcSort::BitVec(w) => write!(f, "(_ BitVec {w})"),
            ChcSort::Array(k, v) => write!(f, "(Array {} {})", k, v),
        }
    }
}

/// A variable in CHC expressions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChcVar {
    pub name: String,
    pub sort: ChcSort,
}

impl ChcVar {
    pub fn new(name: impl Into<String>, sort: ChcSort) -> Self {
        Self {
            name: name.into(),
            sort,
        }
    }

    /// Check if this is a primed variable (ends with ')
    pub fn is_primed(&self) -> bool {
        self.name.ends_with('\'')
    }

    /// Get the base name (without prime suffix)
    pub fn base_name(&self) -> &str {
        if self.is_primed() {
            &self.name[..self.name.len() - 1]
        } else {
            &self.name
        }
    }

    /// Create a primed version of this variable
    pub fn primed(&self) -> Self {
        if self.is_primed() {
            self.clone()
        } else {
            Self {
                name: format!("{}'", self.name),
                sort: self.sort.clone(),
            }
        }
    }

    /// Create an unprimed version of this variable
    pub fn unprimed(&self) -> Self {
        Self {
            name: self.base_name().to_string(),
            sort: self.sort.clone(),
        }
    }
}

impl fmt::Display for ChcVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Operations in CHC expressions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChcOp {
    // Boolean operations
    Not,
    And,
    Or,
    Implies,
    Iff,

    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    // Comparisons
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Conditional
    Ite,

    // Array operations
    /// select(arr, idx) - read from array
    Select,
    /// store(arr, idx, val) - write to array
    Store,
}

use crate::PredicateId;

/// CHC expression
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChcExpr {
    /// Boolean constant
    Bool(bool),
    /// Integer constant
    Int(i64),
    /// Real constant (as rational)
    Real(i64, i64),
    /// Variable reference
    Var(ChcVar),
    /// Operation application
    Op(ChcOp, Vec<Arc<ChcExpr>>),
    /// Predicate application (uninterpreted relation call)
    /// Contains: predicate name, predicate ID, and argument expressions
    PredicateApp(String, PredicateId, Vec<Arc<ChcExpr>>),
}

impl ChcExpr {
    // Convenience constructors

    pub fn bool_const(b: bool) -> Self {
        ChcExpr::Bool(b)
    }

    pub fn int(n: i64) -> Self {
        ChcExpr::Int(n)
    }

    pub fn var(v: ChcVar) -> Self {
        ChcExpr::Var(v)
    }

    /// Create a predicate application
    pub fn predicate_app(name: impl Into<String>, id: PredicateId, args: Vec<ChcExpr>) -> Self {
        ChcExpr::PredicateApp(name.into(), id, args.into_iter().map(Arc::new).collect())
    }

    pub fn not(e: ChcExpr) -> Self {
        // Double negation elimination: NOT(NOT(x)) = x
        if let ChcExpr::Op(ChcOp::Not, args) = &e {
            if args.len() == 1 {
                return (*args[0]).clone();
            }
        }
        ChcExpr::Op(ChcOp::Not, vec![Arc::new(e)])
    }

    pub fn and(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::And, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn or(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Or, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn implies(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Implies, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn add(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Add, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn sub(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Sub, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn mul(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Mul, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn mod_op(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Mod, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn neg(e: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Neg, vec![Arc::new(e)])
    }

    pub fn eq(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Eq, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn ne(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Ne, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn lt(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Lt, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn le(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Le, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn gt(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Gt, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn ge(a: ChcExpr, b: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Ge, vec![Arc::new(a), Arc::new(b)])
    }

    pub fn ite(cond: ChcExpr, then_: ChcExpr, else_: ChcExpr) -> Self {
        ChcExpr::Op(
            ChcOp::Ite,
            vec![Arc::new(cond), Arc::new(then_), Arc::new(else_)],
        )
    }

    /// Array select: select(arr, idx)
    pub fn select(arr: ChcExpr, idx: ChcExpr) -> Self {
        ChcExpr::Op(ChcOp::Select, vec![Arc::new(arr), Arc::new(idx)])
    }

    /// Array store: store(arr, idx, val)
    pub fn store(arr: ChcExpr, idx: ChcExpr, val: ChcExpr) -> Self {
        ChcExpr::Op(
            ChcOp::Store,
            vec![Arc::new(arr), Arc::new(idx), Arc::new(val)],
        )
    }

    /// Get the sort of this expression
    pub fn sort(&self) -> ChcSort {
        match self {
            ChcExpr::Bool(_) => ChcSort::Bool,
            ChcExpr::Int(_) => ChcSort::Int,
            ChcExpr::Real(_, _) => ChcSort::Real,
            ChcExpr::Var(v) => v.sort.clone(),
            ChcExpr::PredicateApp(_, _, _) => ChcSort::Bool, // Predicates return Bool
            ChcExpr::Op(op, args) => match op {
                ChcOp::Not | ChcOp::And | ChcOp::Or | ChcOp::Implies | ChcOp::Iff => ChcSort::Bool,
                ChcOp::Eq | ChcOp::Ne | ChcOp::Lt | ChcOp::Le | ChcOp::Gt | ChcOp::Ge => {
                    ChcSort::Bool
                }
                ChcOp::Add | ChcOp::Sub | ChcOp::Mul | ChcOp::Div | ChcOp::Mod | ChcOp::Neg => {
                    // Return the sort of the first argument
                    args.first().map(|a| a.sort()).unwrap_or(ChcSort::Int)
                }
                ChcOp::Ite => args.get(1).map(|a| a.sort()).unwrap_or(ChcSort::Bool),
                ChcOp::Select => {
                    // select(arr, idx) returns the value sort of the array
                    if let Some(arr) = args.first() {
                        if let ChcSort::Array(_, v) = arr.sort() {
                            return (*v).clone();
                        }
                    }
                    ChcSort::Int // Fallback
                }
                ChcOp::Store => {
                    // store(arr, idx, val) returns the array sort
                    args.first().map(|a| a.sort()).unwrap_or(ChcSort::Int)
                }
            },
        }
    }

    /// Substitute variables in the expression
    pub fn substitute(&self, subst: &[(ChcVar, ChcExpr)]) -> ChcExpr {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) => self.clone(),
            ChcExpr::Var(v) => {
                for (var, expr) in subst {
                    if var == v {
                        return expr.clone();
                    }
                }
                self.clone()
            }
            ChcExpr::Op(op, args) => {
                let new_args: Vec<_> = args.iter().map(|a| Arc::new(a.substitute(subst))).collect();
                ChcExpr::Op(op.clone(), new_args)
            }
            ChcExpr::PredicateApp(name, id, args) => {
                let new_args: Vec<_> = args.iter().map(|a| Arc::new(a.substitute(subst))).collect();
                ChcExpr::PredicateApp(name.clone(), *id, new_args)
            }
        }
    }

    /// Replace a disequality (not (= lhs rhs)) with a replacement expression.
    ///
    /// This is used for disequality splitting: (not (= a b)) -> (< a b) or (> a b).
    pub fn replace_diseq(&self, target_lhs: &ChcExpr, target_rhs: &ChcExpr, replacement: ChcExpr) -> ChcExpr {
        match self {
            ChcExpr::Op(ChcOp::Not, args) if args.len() == 1 => {
                if let ChcExpr::Op(ChcOp::Eq, eq_args) = args[0].as_ref() {
                    if eq_args.len() == 2 {
                        let lhs = &*eq_args[0];
                        let rhs = &*eq_args[1];
                        // Check if this is the disequality we're looking for
                        if (lhs == target_lhs && rhs == target_rhs)
                            || (lhs == target_rhs && rhs == target_lhs)
                        {
                            return replacement;
                        }
                    }
                }
                // Not the target disequality - recurse
                ChcExpr::Op(
                    ChcOp::Not,
                    vec![Arc::new(args[0].replace_diseq(target_lhs, target_rhs, replacement))],
                )
            }
            ChcExpr::Op(op, args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| Arc::new(a.replace_diseq(target_lhs, target_rhs, replacement.clone())))
                    .collect();
                ChcExpr::Op(op.clone(), new_args)
            }
            _ => self.clone(),
        }
    }

    /// Get all variables in the expression
    pub fn vars(&self) -> Vec<ChcVar> {
        let mut result = Vec::new();
        self.collect_vars(&mut result);
        result
    }

    fn collect_vars(&self, result: &mut Vec<ChcVar>) {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) => {}
            ChcExpr::Var(v) => {
                if !result.contains(v) {
                    result.push(v.clone());
                }
            }
            ChcExpr::Op(_, args) => {
                for arg in args {
                    arg.collect_vars(result);
                }
            }
            ChcExpr::PredicateApp(_, _, args) => {
                for arg in args {
                    arg.collect_vars(result);
                }
            }
        }
    }

    /// Check if the expression contains any array operations (select, store)
    pub fn contains_array_ops(&self) -> bool {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) => false,
            ChcExpr::Var(v) => matches!(v.sort, ChcSort::Array(_, _)),
            ChcExpr::Op(op, args) => {
                matches!(op, ChcOp::Select | ChcOp::Store)
                    || args.iter().any(|a| a.contains_array_ops())
            }
            ChcExpr::PredicateApp(_, _, args) => args.iter().any(|a| a.contains_array_ops()),
        }
    }

    /// Filter out conjuncts that involve array operations, returning only integer/bool constraints
    /// Recursively drills into nested AND structures to extract non-array atomic constraints
    pub fn filter_array_conjuncts(&self) -> ChcExpr {
        // First, flatten all nested ANDs into a list of atomic constraints
        fn flatten_and(expr: &ChcExpr, out: &mut Vec<ChcExpr>) {
            match expr {
                ChcExpr::Op(ChcOp::And, args) => {
                    for a in args {
                        flatten_and(a.as_ref(), out);
                    }
                }
                ChcExpr::Bool(true) => {} // Skip trivial true
                _ => out.push(expr.clone()),
            }
        }

        let mut conjuncts = Vec::new();
        flatten_and(self, &mut conjuncts);

        // Filter out conjuncts that contain array operations
        let filtered: Vec<ChcExpr> = conjuncts
            .into_iter()
            .filter(|c| !c.contains_array_ops())
            .collect();

        if filtered.is_empty() {
            ChcExpr::Bool(true)
        } else if filtered.len() == 1 {
            filtered.into_iter().next().unwrap()
        } else {
            ChcExpr::Op(ChcOp::And, filtered.into_iter().map(Arc::new).collect())
        }
    }

    /// Propagate constants from equalities of the form `var = constant`.
    /// This enables constant folding for expressions like `(mod A 2)` when `A = 0`.
    pub fn propagate_constants(&self) -> ChcExpr {
        // First propagate var = var equalities
        let var_propagated = self.propagate_var_equalities();

        // Extract var = const equalities from conjunction
        let equalities = var_propagated.extract_var_const_equalities();
        if equalities.is_empty() {
            // Always simplify even if no substitutions - this enables algebraic simplification
            // like (+ (+ x 1) (- y 1)) -> (+ x y)
            return var_propagated.simplify_constants();
        }

        // Build substitution list
        let subst: Vec<(ChcVar, ChcExpr)> = equalities
            .iter()
            .map(|(var, val)| (var.clone(), ChcExpr::Int(*val)))
            .collect();

        // Apply substitution and simplify
        var_propagated.substitute(&subst).simplify_constants()
    }

    /// Propagate variable equalities: if x = y, substitute all occurrences of y with x
    pub fn propagate_var_equalities(&self) -> ChcExpr {
        let var_eqs = self.extract_var_var_equalities();
        if var_eqs.is_empty() {
            return self.clone();
        }

        // Build substitution list: substitute second var with first
        let subst: Vec<(ChcVar, ChcExpr)> = var_eqs
            .iter()
            .map(|(v1, v2)| (v2.clone(), ChcExpr::var(v1.clone())))
            .collect();

        // Apply substitution and simplify
        self.substitute(&subst).simplify_constants()
    }

    /// Extract (var1, var2) pairs from var = var equalities in a conjunction
    fn extract_var_var_equalities(&self) -> Vec<(ChcVar, ChcVar)> {
        let mut result = Vec::new();
        self.collect_var_var_equalities(&mut result);
        result
    }

    fn collect_var_var_equalities(&self, result: &mut Vec<(ChcVar, ChcVar)>) {
        match self {
            ChcExpr::Op(ChcOp::And, args) => {
                for arg in args {
                    arg.collect_var_var_equalities(result);
                }
            }
            ChcExpr::Op(ChcOp::Eq, args) if args.len() == 2 => {
                // Check for var = var pattern
                if let (ChcExpr::Var(v1), ChcExpr::Var(v2)) = (args[0].as_ref(), args[1].as_ref()) {
                    if v1.name != v2.name {
                        result.push((v1.clone(), v2.clone()));
                    }
                }
            }
            _ => {}
        }
    }

    /// Extract (variable, constant) pairs from var = const equalities in a conjunction.
    fn extract_var_const_equalities(&self) -> Vec<(ChcVar, i64)> {
        let mut result = Vec::new();
        self.collect_var_const_equalities(&mut result);
        result
    }

    fn collect_var_const_equalities(&self, result: &mut Vec<(ChcVar, i64)>) {
        match self {
            ChcExpr::Op(ChcOp::And, args) => {
                for arg in args {
                    arg.collect_var_const_equalities(result);
                }
            }
            ChcExpr::Op(ChcOp::Eq, args) if args.len() == 2 => {
                // Check for var = const pattern
                if let (ChcExpr::Var(v), ChcExpr::Int(n)) = (args[0].as_ref(), args[1].as_ref()) {
                    result.push((v.clone(), *n));
                } else if let (ChcExpr::Int(n), ChcExpr::Var(v)) =
                    (args[0].as_ref(), args[1].as_ref())
                {
                    result.push((v.clone(), *n));
                }
                // Also handle linear patterns like (= (+ k var) c) => var = c - k
                else if let Some((var, val)) =
                    Self::extract_linear_equality(args[0].as_ref(), args[1].as_ref())
                {
                    result.push((var, val));
                } else if let Some((var, val)) =
                    Self::extract_linear_equality(args[1].as_ref(), args[0].as_ref())
                {
                    result.push((var, val));
                }
            }
            _ => {}
        }
    }

    /// Extract var = value from (+ k var) = c => var = c - k
    fn extract_linear_equality(lhs: &ChcExpr, rhs: &ChcExpr) -> Option<(ChcVar, i64)> {
        let c = match rhs {
            ChcExpr::Int(n) => *n,
            _ => return None,
        };
        match lhs {
            // (+ k var) = c => var = c - k
            ChcExpr::Op(ChcOp::Add, inner) if inner.len() == 2 => {
                match (inner[0].as_ref(), inner[1].as_ref()) {
                    (ChcExpr::Int(k), ChcExpr::Var(v)) | (ChcExpr::Var(v), ChcExpr::Int(k)) => {
                        Some((v.clone(), c - k))
                    }
                    _ => None,
                }
            }
            // (* -1 var) = c => var = -c
            ChcExpr::Op(ChcOp::Mul, inner) if inner.len() == 2 => {
                match (inner[0].as_ref(), inner[1].as_ref()) {
                    (ChcExpr::Int(-1), ChcExpr::Var(v)) | (ChcExpr::Var(v), ChcExpr::Int(-1)) => {
                        Some((v.clone(), -c))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Simplify constant expressions, especially mod with constant arguments.
    pub fn simplify_constants(&self) -> ChcExpr {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Var(_) => {
                self.clone()
            }
            ChcExpr::Op(op, args) => {
                // First simplify all arguments
                let simplified_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.simplify_constants()))
                    .collect();

                // Then try to simplify this operation
                match op {
                    ChcOp::Mod if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            if *b != 0 {
                                // SMT-LIB mod semantics: result has same sign as divisor
                                let result = a % b;
                                let result = if result < 0 {
                                    if *b > 0 {
                                        result + b
                                    } else {
                                        result - b
                                    }
                                } else {
                                    result
                                };
                                return ChcExpr::Int(result);
                            }
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Ite if simplified_args.len() == 3 => {
                        // ITE with constant condition
                        match simplified_args[0].as_ref() {
                            ChcExpr::Bool(true) => simplified_args[1].as_ref().clone(),
                            ChcExpr::Bool(false) => simplified_args[2].as_ref().clone(),
                            _ => ChcExpr::Op(op.clone(), simplified_args),
                        }
                    }
                    ChcOp::Add if simplified_args.len() >= 2 => {
                        // Flatten nested additions and collect terms with coefficients
                        // This allows (+ (+ x 1) (- y 1)) to simplify to (+ x y)
                        let mut constant_sum: i64 = 0;
                        let mut var_terms: Vec<Arc<ChcExpr>> = Vec::new();

                        fn collect_add_terms(
                            expr: &ChcExpr,
                            coeff: i64,
                            constant_sum: &mut i64,
                            var_terms: &mut Vec<Arc<ChcExpr>>,
                        ) {
                            match expr {
                                ChcExpr::Int(n) => {
                                    *constant_sum += coeff * n;
                                }
                                ChcExpr::Op(ChcOp::Add, args) if coeff == 1 => {
                                    // Flatten: (+ a b c) -> collect each
                                    for arg in args {
                                        collect_add_terms(arg.as_ref(), 1, constant_sum, var_terms);
                                    }
                                }
                                ChcExpr::Op(ChcOp::Sub, args) if args.len() == 2 && coeff == 1 => {
                                    // Flatten: (- a b) -> a + (-b)
                                    collect_add_terms(args[0].as_ref(), 1, constant_sum, var_terms);
                                    collect_add_terms(
                                        args[1].as_ref(),
                                        -1,
                                        constant_sum,
                                        var_terms,
                                    );
                                }
                                ChcExpr::Op(ChcOp::Neg, args) if args.len() == 1 && coeff == 1 => {
                                    // Flatten: (- a) -> (-a)
                                    collect_add_terms(
                                        args[0].as_ref(),
                                        -1,
                                        constant_sum,
                                        var_terms,
                                    );
                                }
                                _ => {
                                    // Non-constant term
                                    if coeff == 1 {
                                        var_terms.push(Arc::new(expr.clone()));
                                    } else if coeff == -1 {
                                        var_terms.push(Arc::new(ChcExpr::Op(
                                            ChcOp::Neg,
                                            vec![Arc::new(expr.clone())],
                                        )));
                                    } else {
                                        var_terms.push(Arc::new(ChcExpr::Op(
                                            ChcOp::Mul,
                                            vec![
                                                Arc::new(ChcExpr::Int(coeff)),
                                                Arc::new(expr.clone()),
                                            ],
                                        )));
                                    }
                                }
                            }
                        }

                        for arg in &simplified_args {
                            collect_add_terms(arg.as_ref(), 1, &mut constant_sum, &mut var_terms);
                        }

                        // Build result
                        if var_terms.is_empty() {
                            ChcExpr::Int(constant_sum)
                        } else if constant_sum == 0 {
                            if var_terms.len() == 1 {
                                return var_terms[0].as_ref().clone();
                            }
                            ChcExpr::Op(ChcOp::Add, var_terms)
                        } else {
                            var_terms.push(Arc::new(ChcExpr::Int(constant_sum)));
                            ChcExpr::Op(ChcOp::Add, var_terms)
                        }
                    }
                    ChcOp::Mul if simplified_args.len() >= 2 => {
                        // Try to constant fold: all Int args
                        if simplified_args
                            .iter()
                            .all(|a| matches!(a.as_ref(), ChcExpr::Int(_)))
                        {
                            let prod: i64 = simplified_args
                                .iter()
                                .map(|a| {
                                    if let ChcExpr::Int(n) = a.as_ref() {
                                        *n
                                    } else {
                                        1
                                    }
                                })
                                .product();
                            return ChcExpr::Int(prod);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Sub if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Int(a - b);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Div if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            if *b != 0 {
                                // SMT-LIB div semantics (floor division towards negative infinity)
                                let result = if (*a >= 0) == (*b >= 0) {
                                    a / b
                                } else {
                                    (a - (b.abs() - 1)) / b
                                };
                                return ChcExpr::Int(result);
                            }
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Neg if simplified_args.len() == 1 => {
                        if let ChcExpr::Int(n) = simplified_args[0].as_ref() {
                            return ChcExpr::Int(-n);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Eq if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Bool(a == b);
                        }
                        if let (ChcExpr::Bool(a), ChcExpr::Bool(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Bool(a == b);
                        }
                        // Structural equality: if both sides are identical expressions, return true
                        if simplified_args[0] == simplified_args[1] {
                            return ChcExpr::Bool(true);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Ne if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Bool(a != b);
                        }
                        // Structural inequality: if both sides are identical expressions, return false
                        if simplified_args[0] == simplified_args[1] {
                            return ChcExpr::Bool(false);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Lt if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Bool(a < b);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Le if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Bool(a <= b);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Gt if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Bool(a > b);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Ge if simplified_args.len() == 2 => {
                        if let (ChcExpr::Int(a), ChcExpr::Int(b)) =
                            (simplified_args[0].as_ref(), simplified_args[1].as_ref())
                        {
                            return ChcExpr::Bool(a >= b);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::Not if simplified_args.len() == 1 => {
                        if let ChcExpr::Bool(b) = simplified_args[0].as_ref() {
                            return ChcExpr::Bool(!b);
                        }
                        ChcExpr::Op(op.clone(), simplified_args)
                    }
                    ChcOp::And => {
                        // Flatten nested ANDs and simplify: remove true, return false if any is false
                        fn flatten_and(
                            expr: &ChcExpr,
                            result: &mut Vec<Arc<ChcExpr>>,
                        ) -> bool {
                            match expr {
                                ChcExpr::Bool(true) => true, // skip true
                                ChcExpr::Bool(false) => false, // signal contradiction
                                ChcExpr::Op(ChcOp::And, args) => {
                                    for arg in args {
                                        if !flatten_and(arg.as_ref(), result) {
                                            return false;
                                        }
                                    }
                                    true
                                }
                                _ => {
                                    result.push(Arc::new(expr.clone()));
                                    true
                                }
                            }
                        }

                        let mut new_args = Vec::new();
                        for arg in &simplified_args {
                            if !flatten_and(arg.as_ref(), &mut new_args) {
                                return ChcExpr::Bool(false);
                            }
                        }

                        if new_args.is_empty() {
                            return ChcExpr::Bool(true);
                        }
                        if new_args.len() == 1 {
                            return new_args[0].as_ref().clone();
                        }

                        // Check for P AND NOT P contradictions
                        // Collect all positive conjuncts and check if their negation is also present
                        let mut positive_conjuncts: Vec<&ChcExpr> = Vec::new();
                        let mut negated_conjuncts: Vec<&ChcExpr> = Vec::new();

                        for arg in &new_args {
                            if let ChcExpr::Op(ChcOp::Not, not_args) = arg.as_ref() {
                                if not_args.len() == 1 {
                                    negated_conjuncts.push(not_args[0].as_ref());
                                }
                            } else {
                                positive_conjuncts.push(arg.as_ref());
                            }
                        }

                        // Check if any positive conjunct appears in negated form
                        for pos in &positive_conjuncts {
                            for neg in &negated_conjuncts {
                                if pos == neg {
                                    // Found P AND NOT P - contradiction!
                                    return ChcExpr::Bool(false);
                                }
                            }
                        }

                        ChcExpr::Op(ChcOp::And, new_args)
                    }
                    ChcOp::Or => {
                        // Simplify or: remove false, return true if any is true
                        let mut new_args = Vec::new();
                        for arg in &simplified_args {
                            match arg.as_ref() {
                                ChcExpr::Bool(false) => {} // skip false
                                ChcExpr::Bool(true) => return ChcExpr::Bool(true),
                                _ => new_args.push(arg.clone()),
                            }
                        }
                        if new_args.is_empty() {
                            return ChcExpr::Bool(false);
                        }
                        if new_args.len() == 1 {
                            return new_args[0].as_ref().clone();
                        }
                        ChcExpr::Op(ChcOp::Or, new_args)
                    }
                    _ => ChcExpr::Op(op.clone(), simplified_args),
                }
            }
            ChcExpr::PredicateApp(name, id, args) => {
                let simplified_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.simplify_constants()))
                    .collect();
                ChcExpr::PredicateApp(name.clone(), *id, simplified_args)
            }
        }
    }

    /// Check if expression contains ITE or mod operations (which can be slow to solve)
    pub fn contains_ite_or_mod(&self) -> bool {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Var(_) => false,
            ChcExpr::PredicateApp(_, _, args) => args.iter().any(|a| a.contains_ite_or_mod()),
            ChcExpr::Op(ChcOp::Ite, _) | ChcExpr::Op(ChcOp::Mod, _) => true,
            ChcExpr::Op(_, args) => args.iter().any(|a| a.contains_ite_or_mod()),
        }
    }

    /// Propagate equalities to simplify expressions with known values.
    /// Extracts `var = const` from top-level conjuncts and substitutes throughout.
    /// Iterates until fixpoint.
    pub fn propagate_equalities(&self) -> ChcExpr {
        let debug = std::env::var("Z4_DEBUG_PROP").is_ok();
        let mut current = self.clone();

        for iteration in 0..5 {
            let equalities = current.extract_var_const_equalities();

            if debug {
                eprintln!(
                    "[PROP iter {}] Extracted {} equalities:",
                    iteration,
                    equalities.len()
                );
                for (var, val) in &equalities {
                    eprintln!("  {} = {}", var.name, val);
                }
            }

            if equalities.is_empty() {
                if debug {
                    eprintln!("[PROP iter {}] No equalities, returning current", iteration);
                }
                return current;
            }

            let subst: Vec<(ChcVar, ChcExpr)> = equalities
                .into_iter()
                .map(|(var, val)| (var, ChcExpr::Int(val)))
                .collect();

            let result = current.substitute(&subst).simplify_constants();

            if debug {
                eprintln!("[PROP iter {}] After substitution: {}", iteration, result);
            }

            if result == current {
                if debug {
                    eprintln!("[PROP iter {}] Fixpoint reached", iteration);
                }
                return result;
            }
            if matches!(result, ChcExpr::Bool(_)) {
                if debug {
                    eprintln!("[PROP iter {}] Simplified to Bool", iteration);
                }
                return result;
            }
            current = result;
        }
        current
    }
}

/// State for mod elimination transformation
struct ModEliminationState {
    /// Counter for generating unique variable names
    counter: u32,
    /// Collected definitional constraints for mod expressions
    constraints: Vec<ChcExpr>,
}

impl ModEliminationState {
    fn new() -> Self {
        Self {
            counter: 0,
            constraints: Vec::new(),
        }
    }

    fn fresh_var(&mut self, prefix: &str, sort: ChcSort) -> ChcVar {
        let name = format!("{}_{}", prefix, self.counter);
        self.counter += 1;
        ChcVar::new(name, sort)
    }
}

/// State for ite elimination transformation
struct IteEliminationState {
    /// Counter for generating unique variable names
    counter: u32,
    /// Collected definitional constraints for ite expressions
    constraints: Vec<ChcExpr>,
}

impl IteEliminationState {
    fn new() -> Self {
        Self {
            counter: 0,
            constraints: Vec::new(),
        }
    }

    fn fresh_var(&mut self, prefix: &str, sort: ChcSort) -> ChcVar {
        let name = format!("{}_{}", prefix, self.counter);
        self.counter += 1;
        ChcVar::new(name, sort)
    }
}

impl ChcExpr {
    /// Normalize negations by rewriting `not` applied to comparisons into equivalent operators.
    ///
    /// This keeps formulas in a form that the CHC SMT backend can treat as theory atoms, avoiding
    /// expensive DPLL(T) splitting on Boolean structure like `(not (<= ...))`.
    pub fn normalize_negations(&self) -> ChcExpr {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Var(_) => {
                self.clone()
            }
            ChcExpr::PredicateApp(name, id, args) => ChcExpr::PredicateApp(
                name.clone(),
                *id,
                args.iter()
                    .map(|a| Arc::new(a.normalize_negations()))
                    .collect(),
            ),
            ChcExpr::Op(ChcOp::Not, args) if args.len() == 1 => {
                let inner = args[0].normalize_negations();
                match inner {
                    ChcExpr::Bool(b) => ChcExpr::Bool(!b),
                    ChcExpr::Op(ChcOp::Not, inner_args) if inner_args.len() == 1 => {
                        inner_args[0].normalize_negations()
                    }
                    ChcExpr::Op(op, inner_args) if inner_args.len() == 2 => {
                        let a = (*inner_args[0]).clone();
                        let b = (*inner_args[1]).clone();
                        match op {
                            ChcOp::Eq | ChcOp::Iff => ChcExpr::ne(a, b),
                            ChcOp::Ne => ChcExpr::eq(a, b),
                            ChcOp::Lt => ChcExpr::ge(a, b),
                            ChcOp::Le => ChcExpr::gt(a, b),
                            ChcOp::Gt => ChcExpr::le(a, b),
                            ChcOp::Ge => ChcExpr::lt(a, b),
                            _ => ChcExpr::not(ChcExpr::Op(op, inner_args)),
                        }
                    }
                    other => ChcExpr::not(other),
                }
            }
            ChcExpr::Op(op, args) => ChcExpr::Op(
                op.clone(),
                args.iter()
                    .map(|a| Arc::new(a.normalize_negations()))
                    .collect(),
            ),
        }
    }

    /// Rewrite strict integer comparisons with constant bounds into equivalent non-strict forms.
    ///
    /// For Ints:
    /// - `x < c`  <=>  `x <= c-1`
    /// - `x > c`  <=>  `x >= c+1`
    /// - `c < x`  <=>  `x >= c+1`
    /// - `c > x`  <=>  `x <= c-1`
    pub fn normalize_strict_int_comparisons(&self) -> ChcExpr {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Var(_) => {
                self.clone()
            }
            ChcExpr::PredicateApp(name, id, args) => ChcExpr::PredicateApp(
                name.clone(),
                *id,
                args.iter()
                    .map(|a| Arc::new(a.normalize_strict_int_comparisons()))
                    .collect(),
            ),
            ChcExpr::Op(op, args) if args.len() == 2 => {
                let a = args[0].normalize_strict_int_comparisons();
                let b = args[1].normalize_strict_int_comparisons();

                let is_int = |e: &ChcExpr| matches!(e.sort(), ChcSort::Int);

                match op {
                    ChcOp::Lt if is_int(&a) && is_int(&b) => match (&a, &b) {
                        (_, ChcExpr::Int(c)) => {
                            if let Some(c1) = c.checked_sub(1) {
                                ChcExpr::le(a.clone(), ChcExpr::Int(c1))
                            } else {
                                ChcExpr::lt(a.clone(), b.clone())
                            }
                        }
                        (ChcExpr::Int(c), _) => {
                            if let Some(c1) = c.checked_add(1) {
                                ChcExpr::ge(b.clone(), ChcExpr::Int(c1))
                            } else {
                                ChcExpr::lt(a.clone(), b.clone())
                            }
                        }
                        _ => {
                            // For Ints: a < b  <=>  a + 1 <= b
                            ChcExpr::le(ChcExpr::add(a, ChcExpr::Int(1)), b)
                        }
                    },
                    ChcOp::Gt if is_int(&a) && is_int(&b) => match (&a, &b) {
                        (_, ChcExpr::Int(c)) => {
                            if let Some(c1) = c.checked_add(1) {
                                ChcExpr::ge(a.clone(), ChcExpr::Int(c1))
                            } else {
                                ChcExpr::gt(a.clone(), b.clone())
                            }
                        }
                        (ChcExpr::Int(c), _) => {
                            if let Some(c1) = c.checked_sub(1) {
                                ChcExpr::le(b.clone(), ChcExpr::Int(c1))
                            } else {
                                ChcExpr::gt(a.clone(), b.clone())
                            }
                        }
                        _ => {
                            // For Ints: a > b  <=>  a >= b + 1
                            ChcExpr::ge(a, ChcExpr::add(b, ChcExpr::Int(1)))
                        }
                    },
                    _ => ChcExpr::Op(op.clone(), vec![Arc::new(a), Arc::new(b)]),
                }
            }
            ChcExpr::Op(op, args) => ChcExpr::Op(
                op.clone(),
                args.iter()
                    .map(|a| Arc::new(a.normalize_strict_int_comparisons()))
                    .collect(),
            ),
        }
    }

    /// Eliminate arithmetic ite expressions by introducing auxiliary variables and constraints.
    ///
    /// The CHC SMT backend ultimately relies on the LIA solver, which only supports linear
    /// integer/real arithmetic terms. Arithmetic-valued ite terms (e.g. `(ite c 1 0)`) create
    /// non-linear theory atoms like `(= x (ite ...))`, which can force the backend to return
    /// `unknown`. This pass rewrites arithmetic ite expressions into a fresh variable `v` with:
    /// - (=> c (= v t))
    /// - (=> (not c) (= v e))
    ///
    /// Boolean-valued ite expressions are left intact.
    pub fn eliminate_ite(&self) -> ChcExpr {
        let mut state = IteEliminationState::new();
        let transformed = self.eliminate_ite_recursive(&mut state);

        if state.constraints.is_empty() {
            transformed
        } else {
            let mut all_conjuncts = state.constraints;
            all_conjuncts.push(transformed);
            ChcExpr::and_many(all_conjuncts)
        }
    }

    /// Eliminate mod expressions by introducing auxiliary variables and constraints.
    ///
    /// For each `(mod x k)` where k is a constant:
    /// - Introduces fresh variables q (quotient) and uses the mod expression as r (remainder)
    /// - Adds constraints: x = k*q + r, 0 <= r, r < |k|
    ///
    /// For each `(div x k)` where k is a constant:
    /// - Introduces fresh variables q (quotient) and r (remainder)
    /// - Adds the same constraints as for mod elimination, and replaces the div term with q
    ///
    /// For k = 0, follows SMT-LIB total semantics:
    /// - (mod x 0) = x
    /// - (div x 0) = 0
    ///
    /// Returns the transformed expression with all definitional constraints ANDed.
    pub fn eliminate_mod(&self) -> ChcExpr {
        let mut state = ModEliminationState::new();
        let transformed = self.eliminate_mod_recursive(&mut state);

        if state.constraints.is_empty() {
            transformed
        } else {
            // AND all constraints together with the transformed expression
            let mut all_conjuncts = state.constraints;
            all_conjuncts.push(transformed);
            ChcExpr::and_many(all_conjuncts)
        }
    }

    /// Create conjunction of multiple expressions
    fn and_many(exprs: Vec<ChcExpr>) -> ChcExpr {
        if exprs.is_empty() {
            return ChcExpr::Bool(true);
        }
        if exprs.len() == 1 {
            return exprs.into_iter().next().unwrap();
        }
        ChcExpr::Op(ChcOp::And, exprs.into_iter().map(Arc::new).collect())
    }

    /// Recursive helper for ite elimination
    fn eliminate_ite_recursive(&self, state: &mut IteEliminationState) -> ChcExpr {
        match self {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Var(_) => {
                self.clone()
            }

            ChcExpr::Op(ChcOp::Ite, args) if args.len() == 3 => {
                let cond = args[0].eliminate_ite_recursive(state);
                let then_ = args[1].eliminate_ite_recursive(state);
                let else_ = args[2].eliminate_ite_recursive(state);

                let then_sort = then_.sort();
                let else_sort = else_.sort();
                if then_sort == else_sort && matches!(then_sort, ChcSort::Int | ChcSort::Real) {
                    let v = state.fresh_var("_ite", then_sort);
                    let v_expr = ChcExpr::Var(v);

                    let eq_then = ChcExpr::eq(v_expr.clone(), then_);
                    let eq_else = ChcExpr::eq(v_expr.clone(), else_);

                    state
                        .constraints
                        .push(ChcExpr::implies(cond.clone(), eq_then));
                    state
                        .constraints
                        .push(ChcExpr::implies(ChcExpr::not(cond), eq_else));

                    return v_expr;
                }

                ChcExpr::ite(cond, then_, else_)
            }

            ChcExpr::Op(op, args) => {
                let transformed_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.eliminate_ite_recursive(state)))
                    .collect();
                ChcExpr::Op(op.clone(), transformed_args)
            }

            ChcExpr::PredicateApp(name, id, args) => {
                let transformed_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.eliminate_ite_recursive(state)))
                    .collect();
                ChcExpr::PredicateApp(name.clone(), *id, transformed_args)
            }
        }
    }

    /// Recursive helper for mod elimination
    fn eliminate_mod_recursive(&self, state: &mut ModEliminationState) -> ChcExpr {
        match self {
            // Leaf nodes - no transformation needed
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Var(_) => {
                self.clone()
            }

            ChcExpr::Op(ChcOp::Mod, args) if args.len() == 2 => {
                if let ChcExpr::Int(k) = args[1].as_ref() {
                    // SMT-LIB total semantics: (mod x 0) = x
                    if *k == 0 {
                        return args[0].eliminate_mod_recursive(state);
                    }

                    // Recursively transform the dividend
                    let x = args[0].eliminate_mod_recursive(state);

                    // Create fresh quotient and remainder variables
                    let q = state.fresh_var("_mod_q", ChcSort::Int);
                    let r = state.fresh_var("_mod_r", ChcSort::Int);

                    let k_abs = k.abs();
                    let k_expr = ChcExpr::Int(*k);
                    let k_abs_expr = ChcExpr::Int(k_abs);
                    let q_expr = ChcExpr::Var(q);
                    let r_expr = ChcExpr::Var(r.clone());
                    let zero = ChcExpr::Int(0);

                    // Constraint 1: x = k * q + r
                    let k_times_q = ChcExpr::mul(k_expr, q_expr);
                    let k_times_q_plus_r = ChcExpr::add(k_times_q, r_expr.clone());
                    let eq_constraint = ChcExpr::eq(x, k_times_q_plus_r);

                    // Constraint 2: r >= 0
                    let r_ge_0 = ChcExpr::ge(r_expr.clone(), zero);

                    // Constraint 3: r < |k|
                    let r_lt_k = ChcExpr::lt(r_expr.clone(), k_abs_expr);

                    state.constraints.push(eq_constraint);
                    state.constraints.push(r_ge_0);
                    state.constraints.push(r_lt_k);

                    // Return the remainder variable in place of the mod expression
                    return r_expr;
                }

                // If divisor is not a constant or is zero, just transform arguments
                let transformed_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.eliminate_mod_recursive(state)))
                    .collect();
                ChcExpr::Op(ChcOp::Mod, transformed_args)
            }

            ChcExpr::Op(ChcOp::Div, args) if args.len() == 2 => {
                if let ChcExpr::Int(k) = args[1].as_ref() {
                    // SMT-LIB total semantics: (div x 0) = 0
                    if *k == 0 {
                        return ChcExpr::Int(0);
                    }

                    let x = args[0].eliminate_mod_recursive(state);

                    let q = state.fresh_var("_div_q", ChcSort::Int);
                    let r = state.fresh_var("_div_r", ChcSort::Int);

                    let k_abs = k.abs();
                    let k_expr = ChcExpr::Int(*k);
                    let k_abs_expr = ChcExpr::Int(k_abs);
                    let q_expr = ChcExpr::Var(q.clone());
                    let r_expr = ChcExpr::Var(r.clone());
                    let zero = ChcExpr::Int(0);

                    let k_times_q = ChcExpr::mul(k_expr, q_expr.clone());
                    let k_times_q_plus_r = ChcExpr::add(k_times_q, r_expr.clone());
                    let eq_constraint = ChcExpr::eq(x, k_times_q_plus_r);

                    let r_ge_0 = ChcExpr::ge(r_expr.clone(), zero);
                    let r_lt_k = ChcExpr::lt(r_expr, k_abs_expr);

                    state.constraints.push(eq_constraint);
                    state.constraints.push(r_ge_0);
                    state.constraints.push(r_lt_k);

                    return ChcExpr::Var(q);
                }

                let transformed_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.eliminate_mod_recursive(state)))
                    .collect();
                ChcExpr::Op(ChcOp::Div, transformed_args)
            }

            ChcExpr::Op(op, args) => {
                // Recursively transform all arguments
                let transformed_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.eliminate_mod_recursive(state)))
                    .collect();
                ChcExpr::Op(op.clone(), transformed_args)
            }

            ChcExpr::PredicateApp(name, id, args) => {
                let transformed_args: Vec<Arc<ChcExpr>> = args
                    .iter()
                    .map(|a| Arc::new(a.eliminate_mod_recursive(state)))
                    .collect();
                ChcExpr::PredicateApp(name.clone(), *id, transformed_args)
            }
        }
    }
}

impl fmt::Display for ChcExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChcExpr::Bool(b) => write!(f, "{b}"),
            ChcExpr::Int(n) => write!(f, "{n}"),
            ChcExpr::Real(n, d) => write!(f, "{n}/{d}"),
            ChcExpr::Var(v) => write!(f, "{v}"),
            ChcExpr::PredicateApp(name, _, args) => {
                write!(f, "({name}")?;
                for arg in args {
                    write!(f, " {arg}")?;
                }
                write!(f, ")")
            }
            ChcExpr::Op(op, args) => {
                let op_str = match op {
                    ChcOp::Not => "not",
                    ChcOp::And => "and",
                    ChcOp::Or => "or",
                    ChcOp::Implies => "=>",
                    ChcOp::Iff => "=",
                    ChcOp::Add => "+",
                    ChcOp::Sub => "-",
                    ChcOp::Mul => "*",
                    ChcOp::Div => "div",
                    ChcOp::Mod => "mod",
                    ChcOp::Neg => "-",
                    ChcOp::Eq => "=",
                    ChcOp::Ne => "distinct",
                    ChcOp::Lt => "<",
                    ChcOp::Le => "<=",
                    ChcOp::Gt => ">",
                    ChcOp::Ge => ">=",
                    ChcOp::Ite => "ite",
                    ChcOp::Select => "select",
                    ChcOp::Store => "store",
                };
                write!(f, "({op_str}")?;
                for arg in args {
                    write!(f, " {arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}
