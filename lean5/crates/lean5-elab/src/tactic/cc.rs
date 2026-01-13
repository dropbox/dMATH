//! Congruence closure tactic
//!
//! Provides the `cc` tactic for proving equalities using congruence closure.

use std::collections::HashMap;

use lean5_kernel::Expr;

use super::{match_eq_simple, rfl, ProofState, TacticError, TacticResult};

// ============================================================================
// CC (Congruence Closure) Tactic
// ============================================================================

#[derive(Debug, Clone)]
pub struct CCConfig {
    pub max_iterations: usize,
    pub verbose: bool,
}

impl Default for CCConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            verbose: false,
        }
    }
}

pub(crate) struct CCState {
    parent: HashMap<usize, usize>,
    rank: HashMap<usize, usize>,
    expr_to_id: HashMap<String, usize>,
    id_to_expr: Vec<Expr>,
    pending: Vec<(usize, usize)>,
    use_list: HashMap<usize, Vec<usize>>,
}

impl CCState {
    pub(crate) fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
            expr_to_id: HashMap::new(),
            id_to_expr: Vec::new(),
            pending: Vec::new(),
            use_list: HashMap::new(),
        }
    }

    pub(crate) fn add_expr(&mut self, expr: &Expr) -> usize {
        let key = format!("{expr:?}");
        if let Some(&id) = self.expr_to_id.get(&key) {
            return id;
        }

        let id = self.id_to_expr.len();
        self.expr_to_id.insert(key, id);
        self.id_to_expr.push(expr.clone());
        self.parent.insert(id, id);
        self.rank.insert(id, 0);

        if let Expr::App(f, a) = expr {
            let f_id = self.add_expr(f);
            let a_id = self.add_expr(a);
            self.use_list.entry(f_id).or_default().push(id);
            self.use_list.entry(a_id).or_default().push(id);
        }
        id
    }

    pub(crate) fn find(&mut self, mut x: usize) -> usize {
        let mut root = x;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }
        while self.parent[&x] != root {
            let next = self.parent[&x];
            self.parent.insert(x, root);
            x = next;
        }
        root
    }

    pub(crate) fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);
        if x_root == y_root {
            return;
        }

        let x_rank = self.rank[&x_root];
        let y_rank = self.rank[&y_root];

        if x_rank < y_rank {
            self.parent.insert(x_root, y_root);
        } else if x_rank > y_rank {
            self.parent.insert(y_root, x_root);
        } else {
            self.parent.insert(y_root, x_root);
            self.rank.insert(x_root, x_rank + 1);
        }
    }

    fn process_pending(&mut self, max_iterations: usize) {
        for _ in 0..max_iterations {
            if self.pending.is_empty() {
                break;
            }
            let (a, b) = self.pending.pop().expect("pending is not empty");
            self.union(a, b);
        }
    }
}

/// Tactic: cc - Congruence closure tactic.
pub fn cc(state: &mut ProofState) -> TacticResult {
    cc_with_config(state, CCConfig::default())
}

pub fn cc_with_config(state: &mut ProofState, config: CCConfig) -> TacticResult {
    let goal = state.current_goal().ok_or(TacticError::NoGoals)?;
    let target = state.metas.instantiate(&goal.target);

    let (lhs, rhs) = match_eq_simple(&target)
        .ok_or_else(|| TacticError::Other("cc: goal must be an equality".to_string()))?;

    let mut cc_state = CCState::new();
    let lhs_id = cc_state.add_expr(&lhs);
    let rhs_id = cc_state.add_expr(&rhs);

    for decl in &goal.local_ctx {
        let ty = state.metas.instantiate(&decl.ty);
        if let Some((eq_lhs, eq_rhs)) = match_eq_simple(&ty) {
            let l_id = cc_state.add_expr(&eq_lhs);
            let r_id = cc_state.add_expr(&eq_rhs);
            cc_state.union(l_id, r_id);
        }
    }

    cc_state.process_pending(config.max_iterations);

    if cc_state.find(lhs_id) == cc_state.find(rhs_id) {
        rfl(state)
    } else {
        Err(TacticError::Other(
            "cc: could not prove goal by congruence closure".to_string(),
        ))
    }
}
