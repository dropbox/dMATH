//! CHC problem definition

use crate::{
    ChcExpr, ChcOp, ChcResult, ChcSort, ChcVar, ClauseBody, ClauseHead, HornClause, Predicate,
    PredicateId,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

/// A Constrained Horn Clause problem
///
/// Contains:
/// - A set of predicate declarations (uninterpreted relations)
/// - A set of Horn clauses (rules)
/// - Query clauses (clauses with false head)
#[derive(Debug, Clone)]
pub struct ChcProblem {
    /// Predicate declarations
    predicates: Vec<Predicate>,
    /// Map from name to predicate ID
    predicate_names: FxHashMap<String, PredicateId>,
    /// All Horn clauses
    clauses: Vec<HornClause>,
}

impl ChcProblem {
    /// Create a new empty CHC problem
    pub fn new() -> Self {
        Self {
            predicates: Vec::new(),
            predicate_names: FxHashMap::default(),
            clauses: Vec::new(),
        }
    }

    /// Declare a new predicate
    pub fn declare_predicate(
        &mut self,
        name: impl Into<String>,
        arg_sorts: Vec<ChcSort>,
    ) -> PredicateId {
        let name = name.into();
        let id = PredicateId::new(self.predicates.len() as u32);
        let pred = Predicate::new(id, name.clone(), arg_sorts);
        self.predicates.push(pred);
        self.predicate_names.insert(name, id);
        id
    }

    /// Get a predicate by ID
    pub fn get_predicate(&self, id: PredicateId) -> Option<&Predicate> {
        self.predicates.get(id.index())
    }

    /// Get a predicate by name
    pub fn get_predicate_by_name(&self, name: &str) -> Option<&Predicate> {
        self.predicate_names
            .get(name)
            .and_then(|id| self.predicates.get(id.index()))
    }

    /// Look up predicate ID by name
    pub fn lookup_predicate(&self, name: &str) -> Option<PredicateId> {
        self.predicate_names.get(name).copied()
    }

    /// Add a Horn clause
    pub fn add_clause(&mut self, clause: HornClause) {
        self.clauses.push(clause);
    }

    /// Get all clauses
    pub fn clauses(&self) -> &[HornClause] {
        &self.clauses
    }

    /// Get all predicates
    pub fn predicates(&self) -> &[Predicate] {
        &self.predicates
    }

    /// Get query clauses (clauses with false head)
    pub fn queries(&self) -> impl Iterator<Item = &HornClause> {
        self.clauses.iter().filter(|c| c.is_query())
    }

    /// Get fact clauses (clauses with no predicates in body)
    pub fn facts(&self) -> impl Iterator<Item = &HornClause> {
        self.clauses.iter().filter(|c| c.is_fact() && !c.is_query())
    }

    /// Get transition clauses (clauses with predicates in both body and head)
    pub fn transitions(&self) -> impl Iterator<Item = &HornClause> {
        self.clauses
            .iter()
            .filter(|c| !c.is_fact() && !c.is_query())
    }

    /// Get clauses that define a predicate (have it in head)
    pub fn clauses_defining(&self, pred: PredicateId) -> impl Iterator<Item = &HornClause> {
        self.clauses
            .iter()
            .filter(move |c| c.head.predicate_id() == Some(pred))
    }

    /// Get clauses that define a predicate with their indices
    pub fn clauses_defining_with_index(
        &self,
        pred: PredicateId,
    ) -> impl Iterator<Item = (usize, &HornClause)> {
        self.clauses
            .iter()
            .enumerate()
            .filter(move |(_, c)| c.head.predicate_id() == Some(pred))
    }

    /// Get clauses that use a predicate (have it in body)
    pub fn clauses_using(&self, pred: PredicateId) -> impl Iterator<Item = &HornClause> {
        self.clauses
            .iter()
            .filter(move |c| c.body.predicates.iter().any(|(id, _)| *id == pred))
    }

    /// Validate the problem
    pub fn validate(&self) -> ChcResult<()> {
        use crate::ChcError;

        // Check that all predicates used in clauses are declared
        for clause in &self.clauses {
            for (pred_id, args) in &clause.body.predicates {
                let pred = self
                    .get_predicate(*pred_id)
                    .ok_or_else(|| ChcError::UndefinedPredicate(format!("P{}", pred_id.0)))?;
                if args.len() != pred.arity() {
                    return Err(ChcError::ArityMismatch {
                        name: pred.name.clone(),
                        expected: pred.arity(),
                        actual: args.len(),
                    });
                }
            }
            if let crate::ClauseHead::Predicate(pred_id, args) = &clause.head {
                let pred = self
                    .get_predicate(*pred_id)
                    .ok_or_else(|| ChcError::UndefinedPredicate(format!("P{}", pred_id.0)))?;
                if args.len() != pred.arity() {
                    return Err(ChcError::ArityMismatch {
                        name: pred.name.clone(),
                        expected: pred.arity(),
                        actual: args.len(),
                    });
                }
            }
        }

        // Check that there's at least one query
        if self.queries().count() == 0 {
            return Err(ChcError::NoQuery);
        }

        Ok(())
    }

    /// Build a dependency graph of predicates
    ///
    /// Returns edges: (from, to) where `from` appears in body and `to` in head
    pub fn dependency_edges(&self) -> Vec<(PredicateId, PredicateId)> {
        let mut edges = Vec::new();
        for clause in &self.clauses {
            if let Some(head_id) = clause.head.predicate_id() {
                for (body_id, _) in &clause.body.predicates {
                    edges.push((*body_id, head_id));
                }
            }
        }
        edges
    }

    /// Topologically sort predicates (returns None if cyclic)
    pub fn topological_order(&self) -> Option<Vec<PredicateId>> {
        let n = self.predicates.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<PredicateId>> = vec![Vec::new(); n];

        for (from, to) in self.dependency_edges() {
            adj[from.index()].push(to);
            in_degree[to.index()] += 1;
        }

        let mut queue: Vec<_> = (0..n)
            .filter(|i| in_degree[*i] == 0)
            .map(|i| PredicateId::new(i as u32))
            .collect();
        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node);
            for &next in &adj[node.index()] {
                in_degree[next.index()] -= 1;
                if in_degree[next.index()] == 0 {
                    queue.push(next);
                }
            }
        }

        if result.len() == n {
            Some(result)
        } else {
            None // Cycle detected
        }
    }

    /// Attempt to scalarize `select` operations on `(Array Int Int)` at constant indices.
    ///
    /// This rewrites array-typed predicate arguments into a fixed set of scalar `Int` arguments
    /// representing the selected values, and eliminates the corresponding `select`/`store`
    /// operations from constraints when possible.
    pub fn try_scalarize_const_array_selects(&mut self) {
        let mut indices = rustc_hash::FxHashSet::default();
        for clause in &self.clauses {
            if let Some(c) = &clause.body.constraint {
                Self::collect_const_int_array_select_indices(c, &mut indices);
            }
            for (_, args) in &clause.body.predicates {
                for arg in args {
                    Self::collect_const_int_array_select_indices(arg, &mut indices);
                }
            }
            if let crate::ClauseHead::Predicate(_, args) = &clause.head {
                for arg in args {
                    Self::collect_const_int_array_select_indices(arg, &mut indices);
                }
            }
        }

        if indices.is_empty() {
            return;
        }

        let mut indices: Vec<i64> = indices.into_iter().collect();
        indices.sort_unstable();

        let old_predicates = self.predicates.clone();

        // Update predicate signatures: replace `(Array Int Int)` args with `Int` args for each index.
        for pred in &mut self.predicates {
            let mut new_sorts = Vec::new();
            for sort in &pred.arg_sorts {
                if Self::is_int_int_array_sort(sort) {
                    for _ in &indices {
                        new_sorts.push(ChcSort::Int);
                    }
                } else {
                    new_sorts.push(sort.clone());
                }
            }
            pred.arg_sorts = new_sorts;
        }

        // Rewrite all clauses to match the new predicate signatures.
        let mut new_clauses = Vec::with_capacity(self.clauses.len());
        for clause in &self.clauses {
            let mut new_body_preds = Vec::with_capacity(clause.body.predicates.len());
            for (pred_id, args) in &clause.body.predicates {
                let old_pred = &old_predicates[pred_id.index()];
                let new_args = Self::scalarize_pred_args(old_pred, args, &indices);
                new_body_preds.push((*pred_id, new_args));
            }

            let new_constraint = clause
                .body
                .constraint
                .as_ref()
                .map(|c| Self::scalarize_constraint(c, &indices));

            let new_body = crate::ClauseBody::new(new_body_preds, new_constraint);

            let new_head = match &clause.head {
                crate::ClauseHead::Predicate(pred_id, args) => {
                    let old_pred = &old_predicates[pred_id.index()];
                    let new_args = Self::scalarize_pred_args(old_pred, args, &indices);
                    crate::ClauseHead::Predicate(*pred_id, new_args)
                }
                crate::ClauseHead::False => crate::ClauseHead::False,
            };

            new_clauses.push(HornClause::new(new_body, new_head));
        }

        self.clauses = new_clauses;
    }

    /// Attempt to split clauses on `(ite ...)` occurrences by case-splitting on the condition.
    ///
    /// This is a semantics-preserving transformation for CHC:
    /// - `body ∧ φ(ite(c,t,e)) -> head` becomes two clauses:
    ///   - `body ∧ c ∧ φ(t) -> head`
    ///   - `body ∧ ¬c ∧ φ(e) -> head`
    ///
    /// The primary goal is to simplify CHC problems that encode program branches using `ite`,
    /// which can otherwise force PDR to learn many point-wise lemmas.
    ///
    /// To avoid exponential blow-ups, this pass limits the number of clauses generated from any
    /// single input clause. When the limit is reached, remaining `ite` nodes are left intact.
    pub fn try_split_ites_in_clauses(
        &mut self,
        max_clauses_per_input_clause: usize,
        verbose: bool,
    ) {
        if max_clauses_per_input_clause <= 1 {
            return;
        }

        let old = std::mem::take(&mut self.clauses);
        let old_len = old.len();
        let mut new_clauses: Vec<HornClause> = Vec::with_capacity(old_len);

        let mut total_splits: usize = 0;
        let mut clauses_with_splits: usize = 0;

        for clause in old {
            let mut pending: Vec<HornClause> = vec![clause];
            let mut out: Vec<HornClause> = Vec::new();
            let mut did_split = false;

            while let Some(current) = pending.pop() {
                if out.len() + pending.len() >= max_clauses_per_input_clause {
                    out.push(current);
                    continue;
                }

                let Some((a, b)) = Self::split_clause_once_on_ite(&current) else {
                    out.push(current);
                    continue;
                };

                did_split = true;
                total_splits += 1;
                pending.push(a);
                pending.push(b);
            }

            if did_split {
                clauses_with_splits += 1;
            }
            new_clauses.extend(
                out.into_iter()
                    .filter(|c| !Self::clause_is_trivially_false(c)),
            );
        }

        if verbose && total_splits > 0 {
            eprintln!(
                "CHC: ite-splitting: {} splits across {} clauses ({} -> {})",
                total_splits,
                clauses_with_splits,
                old_len,
                new_clauses.len()
            );
        }

        self.clauses = new_clauses;
    }

    fn clause_is_trivially_false(clause: &HornClause) -> bool {
        matches!(clause.body.constraint, Some(ChcExpr::Bool(false)))
    }

    fn split_clause_once_on_ite(clause: &HornClause) -> Option<(HornClause, HornClause)> {
        // 1) Split on ite inside background constraint.
        if let Some(constraint) = &clause.body.constraint {
            if let Some((cond, then_c, else_c)) = Self::split_expr_once_on_ite(constraint) {
                let a = HornClause::new(
                    ClauseBody::new(
                        clause.body.predicates.clone(),
                        Self::conjoin_constraint(Some(then_c), cond.clone()),
                    ),
                    clause.head.clone(),
                );
                let b = HornClause::new(
                    ClauseBody::new(
                        clause.body.predicates.clone(),
                        Self::conjoin_constraint(Some(else_c), ChcExpr::not(cond)),
                    ),
                    clause.head.clone(),
                );
                return Some((a, b));
            }
        }

        // 2) Split on ite inside any body predicate argument.
        for (pred_i, (_pred, args)) in clause.body.predicates.iter().enumerate() {
            for (arg_i, arg) in args.iter().enumerate() {
                if let Some((cond, then_arg, else_arg)) = Self::split_expr_once_on_ite(arg) {
                    let mut preds_then = clause.body.predicates.clone();
                    preds_then[pred_i].1[arg_i] = then_arg;

                    let mut preds_else = clause.body.predicates.clone();
                    preds_else[pred_i].1[arg_i] = else_arg;

                    let then_constraint =
                        Self::conjoin_constraint(clause.body.constraint.clone(), cond.clone());
                    let else_constraint = Self::conjoin_constraint(
                        clause.body.constraint.clone(),
                        ChcExpr::not(cond),
                    );

                    let a = HornClause::new(
                        ClauseBody::new(preds_then, then_constraint),
                        clause.head.clone(),
                    );
                    let b = HornClause::new(
                        ClauseBody::new(preds_else, else_constraint),
                        clause.head.clone(),
                    );
                    return Some((a, b));
                }
            }
        }

        // 3) Split on ite inside head predicate arguments.
        let ClauseHead::Predicate(head_pred, head_args) = &clause.head else {
            return None;
        };
        for (arg_i, arg) in head_args.iter().enumerate() {
            if let Some((cond, then_arg, else_arg)) = Self::split_expr_once_on_ite(arg) {
                let mut head_then_args = head_args.clone();
                head_then_args[arg_i] = then_arg;
                let mut head_else_args = head_args.clone();
                head_else_args[arg_i] = else_arg;

                let then_constraint =
                    Self::conjoin_constraint(clause.body.constraint.clone(), cond.clone());
                let else_constraint =
                    Self::conjoin_constraint(clause.body.constraint.clone(), ChcExpr::not(cond));

                let a = HornClause::new(
                    ClauseBody::new(clause.body.predicates.clone(), then_constraint),
                    ClauseHead::Predicate(*head_pred, head_then_args),
                );
                let b = HornClause::new(
                    ClauseBody::new(clause.body.predicates.clone(), else_constraint),
                    ClauseHead::Predicate(*head_pred, head_else_args),
                );
                return Some((a, b));
            }
        }

        None
    }

    fn normalize_constraint_opt(constraint: Option<ChcExpr>) -> Option<ChcExpr> {
        let c = constraint?;
        let simplified = c.simplify_constants();
        match simplified {
            ChcExpr::Bool(true) => None,
            other => Some(other),
        }
    }

    fn conjoin_constraint(existing: Option<ChcExpr>, extra: ChcExpr) -> Option<ChcExpr> {
        let extra = extra.simplify_constants();
        match extra {
            ChcExpr::Bool(true) => Self::normalize_constraint_opt(existing),
            ChcExpr::Bool(false) => Some(ChcExpr::Bool(false)),
            _ => {
                let combined = match existing {
                    None => extra,
                    Some(c) => ChcExpr::and(c, extra),
                };
                Self::normalize_constraint_opt(Some(combined))
            }
        }
    }

    fn split_expr_once_on_ite(expr: &ChcExpr) -> Option<(ChcExpr, ChcExpr, ChcExpr)> {
        let path = Self::find_ite_path(expr)?;
        let ite = Self::get_subexpr(expr, &path)?;
        let ChcExpr::Op(ChcOp::Ite, args) = ite else {
            return None;
        };
        if args.len() != 3 {
            return None;
        }
        let cond = args[0].as_ref().clone();
        let then_ = args[1].as_ref().clone();
        let else_ = args[2].as_ref().clone();

        let then_expr = Self::replace_subexpr(expr, &path, &then_);
        let else_expr = Self::replace_subexpr(expr, &path, &else_);
        Some((cond, then_expr, else_expr))
    }

    fn find_ite_path(expr: &ChcExpr) -> Option<Vec<usize>> {
        match expr {
            ChcExpr::Op(ChcOp::Ite, args)
                if args.len() == 3
                    && args[1].sort() == ChcSort::Bool
                    && args[2].sort() == ChcSort::Bool =>
            {
                Some(Vec::new())
            }
            ChcExpr::Op(_, args) => {
                for (i, a) in args.iter().enumerate() {
                    if let Some(mut sub) = Self::find_ite_path(a.as_ref()) {
                        sub.insert(0, i);
                        return Some(sub);
                    }
                }
                None
            }
            ChcExpr::PredicateApp(_, _, args) => {
                for (i, a) in args.iter().enumerate() {
                    if let Some(mut sub) = Self::find_ite_path(a.as_ref()) {
                        sub.insert(0, i);
                        return Some(sub);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn get_subexpr<'a>(expr: &'a ChcExpr, path: &[usize]) -> Option<&'a ChcExpr> {
        let mut current = expr;
        for &idx in path {
            current = match current {
                ChcExpr::Op(_, args) => args.get(idx)?.as_ref(),
                ChcExpr::PredicateApp(_, _, args) => args.get(idx)?.as_ref(),
                _ => return None,
            };
        }
        Some(current)
    }

    fn replace_subexpr(expr: &ChcExpr, path: &[usize], replacement: &ChcExpr) -> ChcExpr {
        if path.is_empty() {
            return replacement.clone();
        }
        let first = path[0];
        let rest = &path[1..];

        match expr {
            ChcExpr::Op(op, args) => {
                let mut new_args: Vec<Arc<ChcExpr>> = Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    if i == first {
                        new_args.push(Arc::new(Self::replace_subexpr(
                            a.as_ref(),
                            rest,
                            replacement,
                        )));
                    } else {
                        new_args.push(Arc::new(a.as_ref().clone()));
                    }
                }
                ChcExpr::Op(op.clone(), new_args)
            }
            ChcExpr::PredicateApp(name, id, args) => {
                let mut new_args: Vec<Arc<ChcExpr>> = Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    if i == first {
                        new_args.push(Arc::new(Self::replace_subexpr(
                            a.as_ref(),
                            rest,
                            replacement,
                        )));
                    } else {
                        new_args.push(Arc::new(a.as_ref().clone()));
                    }
                }
                ChcExpr::PredicateApp(name.clone(), *id, new_args)
            }
            _ => expr.clone(),
        }
    }

    fn is_int_int_array_sort(sort: &ChcSort) -> bool {
        matches!(
            sort,
            ChcSort::Array(k, v) if **k == ChcSort::Int && **v == ChcSort::Int
        )
    }

    fn collect_const_int_array_select_indices(
        expr: &ChcExpr,
        indices: &mut rustc_hash::FxHashSet<i64>,
    ) {
        match expr {
            ChcExpr::Op(ChcOp::Select, args) if args.len() == 2 => {
                if Self::is_int_int_array_sort(&args[0].sort()) {
                    if let ChcExpr::Int(k) = &*args[1] {
                        indices.insert(*k);
                    }
                }
                for a in args {
                    Self::collect_const_int_array_select_indices(a, indices);
                }
            }
            ChcExpr::Op(_, args) => {
                for a in args {
                    Self::collect_const_int_array_select_indices(a, indices);
                }
            }
            ChcExpr::PredicateApp(_, _, args) => {
                for a in args {
                    Self::collect_const_int_array_select_indices(a, indices);
                }
            }
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) | ChcExpr::Var(_) => {}
        }
    }

    fn scalarize_pred_args(pred: &Predicate, args: &[ChcExpr], indices: &[i64]) -> Vec<ChcExpr> {
        let mut out = Vec::new();
        for (arg, sort) in args.iter().zip(pred.arg_sorts.iter()) {
            if Self::is_int_int_array_sort(sort) {
                for &k in indices {
                    out.push(Self::scalar_value_at_index(arg, k));
                }
            } else {
                out.push(Self::scalarize_expr(arg, indices));
            }
        }
        out
    }

    fn scalar_var_for_array(array_var: &ChcVar, index: i64) -> ChcVar {
        let idx = if index < 0 {
            format!("neg{}", index.abs())
        } else {
            index.to_string()
        };
        ChcVar::new(format!("{}__sel_{}", array_var.name, idx), ChcSort::Int)
    }

    fn scalar_value_at_index(array_expr: &ChcExpr, index: i64) -> ChcExpr {
        match array_expr {
            ChcExpr::Var(v) if Self::is_int_int_array_sort(&v.sort) => {
                ChcExpr::Var(Self::scalar_var_for_array(v, index))
            }
            ChcExpr::Op(ChcOp::Store, args) if args.len() == 3 => {
                // Fallback: ite(idx = k, val, select(base, k))
                let base = &args[0];
                let idx = &args[1];
                let val = &args[2];
                let cond = ChcExpr::eq((**idx).clone(), ChcExpr::Int(index));
                let then_ = (**val).clone();
                let else_ = Self::scalar_value_at_index(base, index);
                ChcExpr::ite(cond, then_, else_)
            }
            _ => ChcExpr::select(array_expr.clone(), ChcExpr::Int(index)),
        }
    }

    fn scalarize_expr(expr: &ChcExpr, indices: &[i64]) -> ChcExpr {
        match expr {
            ChcExpr::Bool(_) | ChcExpr::Int(_) | ChcExpr::Real(_, _) => expr.clone(),
            ChcExpr::Var(_) => expr.clone(),
            ChcExpr::PredicateApp(name, id, args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| Arc::new(Self::scalarize_expr(a, indices)))
                    .collect();
                ChcExpr::PredicateApp(name.clone(), *id, new_args)
            }
            ChcExpr::Op(ChcOp::Select, args) if args.len() == 2 => {
                let arr = &args[0];
                let idx = &args[1];
                if let ChcExpr::Int(k) = &**idx {
                    if indices.contains(k) && Self::is_int_int_array_sort(&arr.sort()) {
                        return Self::scalar_value_at_index(arr, *k);
                    }
                }
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| Arc::new(Self::scalarize_expr(a, indices)))
                    .collect();
                ChcExpr::Op(ChcOp::Select, new_args)
            }
            ChcExpr::Op(op, args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| Arc::new(Self::scalarize_expr(a, indices)))
                    .collect();
                ChcExpr::Op(op.clone(), new_args)
            }
        }
    }

    fn scalarize_constraint(constraint: &ChcExpr, indices: &[i64]) -> ChcExpr {
        let mut conjuncts = Vec::new();
        Self::flatten_and(constraint, &mut conjuncts);

        let mut out = Vec::new();
        for c in conjuncts {
            if let Some(expanded) = Self::scalarize_store_equality(&c, indices) {
                out.extend(expanded);
            } else {
                out.push(Self::scalarize_expr(&c, indices));
            }
        }

        Self::and_all(out)
    }

    fn scalarize_store_equality(conjunct: &ChcExpr, indices: &[i64]) -> Option<Vec<ChcExpr>> {
        let ChcExpr::Op(ChcOp::Eq, args) = conjunct else {
            return None;
        };
        if args.len() != 2 {
            return None;
        }

        fn as_array_var(e: &ChcExpr) -> Option<&ChcVar> {
            match e {
                ChcExpr::Var(v) if ChcProblem::is_int_int_array_sort(&v.sort) => Some(v),
                _ => None,
            }
        }

        fn as_store(e: &ChcExpr) -> Option<(&ChcExpr, &ChcExpr, &ChcExpr)> {
            match e {
                ChcExpr::Op(ChcOp::Store, a) if a.len() == 3 => Some((&a[0], &a[1], &a[2])),
                _ => None,
            }
        }

        let (next_arr, store_base, store_idx, store_val) = if let (Some(lhs), Some((b, i, v))) =
            (as_array_var(&args[0]), as_store(&args[1]))
        {
            (lhs, b, i, v)
        } else if let (Some(rhs), Some((b, i, v))) = (as_array_var(&args[1]), as_store(&args[0])) {
            (rhs, b, i, v)
        } else {
            return None;
        };

        if !ChcProblem::is_int_int_array_sort(&store_base.sort()) {
            return None;
        }

        let idx_expr = ChcProblem::scalarize_expr(store_idx, indices);
        let val_expr = ChcProblem::scalarize_expr(store_val, indices);

        let mut out = Vec::new();
        for &k in indices {
            let next_k = ChcExpr::Var(ChcProblem::scalar_var_for_array(next_arr, k));
            let base_k = ChcProblem::scalar_value_at_index(store_base, k);

            // (idx != k) => next_k = base_k  <==>  (or (not (= idx k)) (= next_k base_k))
            out.push(ChcExpr::or(
                ChcExpr::not(ChcExpr::eq(idx_expr.clone(), ChcExpr::Int(k))),
                ChcExpr::eq(next_k.clone(), base_k),
            ));

            // (idx = k) => next_k = val  <==>  (or (= idx k) (= next_k val))
            out.push(ChcExpr::or(
                ChcExpr::eq(idx_expr.clone(), ChcExpr::Int(k)),
                ChcExpr::eq(next_k, val_expr.clone()),
            ));
        }

        Some(out)
    }

    fn flatten_and(expr: &ChcExpr, out: &mut Vec<ChcExpr>) {
        match expr {
            ChcExpr::Op(ChcOp::And, args) if args.len() == 2 => {
                Self::flatten_and(&args[0], out);
                Self::flatten_and(&args[1], out);
            }
            ChcExpr::Op(ChcOp::And, args) => {
                for a in args {
                    Self::flatten_and(a, out);
                }
            }
            _ => out.push(expr.clone()),
        }
    }

    fn and_all(conjuncts: Vec<ChcExpr>) -> ChcExpr {
        let mut iter = conjuncts.into_iter().filter(|e| *e != ChcExpr::Bool(true));
        let Some(first) = iter.next() else {
            return ChcExpr::Bool(true);
        };
        iter.fold(first, ChcExpr::and)
    }

    /// Expand nullary "fail" predicate queries into direct queries.
    ///
    /// CHC-COMP benchmarks often use a pattern with a nullary `fail` predicate:
    /// 1. `inv(...) AND bad_condition => fail`
    /// 2. `fail => false`
    ///
    /// This transformation replaces `fail => false` queries with expanded queries
    /// that directly reference the original predicates:
    /// - For each clause `body => fail`, create a query `body => false`
    /// - Remove the original `fail => false` query
    /// - Remove clauses that have `fail` in their head
    ///
    /// This enables the PDR solver to work with the actual state predicates
    /// instead of the intermediate nullary fail predicate.
    ///
    /// Returns true if any transformation was performed.
    pub fn expand_nullary_fail_queries(&mut self, verbose: bool) -> bool {
        // Find query clauses with nullary predicates in body
        let mut nullary_fail_preds: Vec<PredicateId> = Vec::new();

        for query in self.queries() {
            if query.body.predicates.len() == 1 {
                let (pred_id, args) = &query.body.predicates[0];
                if args.is_empty() {
                    // This is a query with a nullary predicate (like `fail => false`)
                    nullary_fail_preds.push(*pred_id);
                }
            }
        }

        if nullary_fail_preds.is_empty() {
            return false;
        }

        if verbose {
            let pred_names: Vec<_> = nullary_fail_preds
                .iter()
                .filter_map(|id| self.get_predicate(*id).map(|p| p.name.clone()))
                .collect();
            eprintln!(
                "CHC: expanding {} nullary fail predicates: {:?}",
                nullary_fail_preds.len(),
                pred_names
            );
        }

        // For each nullary fail predicate, find clauses that transition to it
        let mut new_queries: Vec<HornClause> = Vec::new();

        for fail_pred in &nullary_fail_preds {
            // Find all clauses `body => fail_pred`
            for clause in &self.clauses {
                if let crate::ClauseHead::Predicate(head_pred, _) = &clause.head {
                    if head_pred == fail_pred {
                        // Convert `body => fail_pred` to `body => false`
                        let new_query = HornClause::query(clause.body.clone());
                        new_queries.push(new_query);
                    }
                }
            }
        }

        if new_queries.is_empty() {
            return false;
        }

        if verbose {
            eprintln!("CHC: created {} expanded queries", new_queries.len());
        }

        // Remove:
        // 1. Original queries with nullary predicates (`fail => false`)
        // 2. Clauses that have a nullary fail predicate in their head
        // Keep all other clauses (facts, transitions, and any queries not using nullary predicates)
        let nullary_set: FxHashSet<PredicateId> = nullary_fail_preds.iter().copied().collect();

        self.clauses.retain(|clause| {
            // Remove `fail => false` queries
            if clause.is_query() && clause.body.predicates.len() == 1 {
                let (pred_id, args) = &clause.body.predicates[0];
                if args.is_empty() && nullary_set.contains(pred_id) {
                    return false;
                }
            }
            // Remove clauses with nullary fail predicate in head
            if let crate::ClauseHead::Predicate(head_pred, _) = &clause.head {
                if nullary_set.contains(head_pred) {
                    return false;
                }
            }
            true
        });

        // Add the expanded queries
        self.clauses.extend(new_queries);

        true
    }
}

impl Default for ChcProblem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod ite_split_tests {
    use super::*;

    fn contains_ite(expr: &ChcExpr) -> bool {
        match expr {
            ChcExpr::Op(ChcOp::Ite, _) => true,
            ChcExpr::Op(_, args) => args.iter().any(|a| contains_ite(a.as_ref())),
            ChcExpr::PredicateApp(_, _, args) => args.iter().any(|a| contains_ite(a.as_ref())),
            _ => false,
        }
    }

    #[test]
    fn split_boolean_ite_in_transition_constraint() {
        let mut problem = ChcProblem::new();
        let inv = problem.declare_predicate("inv", vec![ChcSort::Int, ChcSort::Int]);

        let a = ChcVar::new("A", ChcSort::Int);
        let b = ChcVar::new("B", ChcSort::Int);
        let c = ChcVar::new("C", ChcSort::Int);
        let d = ChcVar::new("D", ChcSort::Int);

        // inv(A,B) /\ (= C (+ 1 A)) /\ (ite (<= C 50) (= D B) (= D (+ 1 B))) => inv(C,D)
        let constraint = ChcExpr::and(
            ChcExpr::eq(
                ChcExpr::var(c.clone()),
                ChcExpr::add(ChcExpr::int(1), ChcExpr::var(a.clone())),
            ),
            ChcExpr::ite(
                ChcExpr::le(ChcExpr::var(c.clone()), ChcExpr::int(50)),
                ChcExpr::eq(ChcExpr::var(d.clone()), ChcExpr::var(b.clone())),
                ChcExpr::eq(
                    ChcExpr::var(d.clone()),
                    ChcExpr::add(ChcExpr::int(1), ChcExpr::var(b.clone())),
                ),
            ),
        );

        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, vec![ChcExpr::var(a.clone()), ChcExpr::var(b.clone())])],
                Some(constraint),
            ),
            ClauseHead::Predicate(inv, vec![ChcExpr::var(c.clone()), ChcExpr::var(d.clone())]),
        ));

        let before = problem.clauses().len();
        problem.try_split_ites_in_clauses(8, false);
        let after = problem.clauses().len();

        assert!(after > before, "expected ite splitting to add clauses");
        for clause in problem.clauses() {
            if let Some(c) = &clause.body.constraint {
                assert!(!contains_ite(c), "constraint still contains ite: {c}");
            }
        }
    }
}
