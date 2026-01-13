//! SMT-Kernel Bridge
//!
//! This module provides translation between lean5-kernel expressions (Expr)
//! and SMT solver terms, enabling proof automation via SMT.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SMT-Kernel Bridge                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Kernel Expr ────────────► SmtContext ────────────► SMT Solver  │
//! │  (Prop/Bool)    translate   (SAT vars)     solve                │
//! │                                                                  │
//! │  Proof Term  ◄──────────── SmtResult ◄───────────── Model       │
//! │  (Expr)        reconstruct  (Sat/Unsat)                          │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Supported Propositions
//!
//! - Equality: `a = b`
//! - Disequality: `a ≠ b` (Not (a = b))
//! - Conjunction: `P ∧ Q`
//! - Disjunction: `P ∨ Q`
//! - Implication: `P → Q`
//! - Negation: `¬P`
//! - True/False
//!
//! # Example Usage
//!
//! ```ignore
//! use lean5_auto::bridge::SmtBridge;
//! use lean5_kernel::{Environment, Expr};
//!
//! let env = Environment::new();
//! let mut bridge = SmtBridge::new(&env);
//!
//! // Try to prove: ∀ x y : A, x = y → y = x
//! let goal = /* ... */;
//! match bridge.prove(&goal) {
//!     Some(proof) => println!("Proof found!"),
//!     None => println!("Cannot prove"),
//! }
//! ```

use crate::proof::{ProofBuilder, ProofStep};
use crate::smt::{SmtResult, SmtSolver, TermId, TheoryLiteral};
use crate::theories::arithmetic::ArithmeticTheory;
use crate::theories::equality::EqualityTheory;
use lean5_kernel::name::Name;
use lean5_kernel::{BinderInfo, Environment, Expr, FVarId, Level};
use std::collections::{HashMap, HashSet};

/// SMT Bridge for translating between kernel Expr and SMT solver
pub struct SmtBridge<'env> {
    /// The kernel environment (for future type lookups)
    #[allow(dead_code)]
    env: &'env Environment,
    /// The SMT solver
    smt: SmtSolver,
    /// Mapping from kernel expressions to SMT term IDs
    expr_to_term: HashMap<ExprKey, TermId>,
    /// Mapping from SMT term IDs to kernel expressions
    term_to_expr: HashMap<TermId, Expr>,
    /// Mapping from SMT term IDs to their types (for proof building)
    term_to_type: HashMap<TermId, Expr>,
    /// Mapping from free variables to SMT terms
    fvar_to_term: HashMap<FVarId, TermId>,
    /// Counter for generating fresh names
    fresh_counter: u32,
    /// Mapping from asserted equalities to hypothesis FVarIds
    /// The key is (lhs, rhs) in the ORDER the hypothesis was asserted
    /// Only stores the canonical direction, not both
    eq_hypothesis_canonical: HashMap<(TermId, TermId), FVarId>,
    /// Equality theory reference (for future proof trace access)
    #[allow(dead_code)]
    equality_theory_idx: Option<usize>,
    /// Pending universal quantifier hypotheses for E-matching instantiation
    /// Each entry contains: (types, body, triggers)
    pending_foralls: Vec<PendingForall>,
    /// Maximum number of instantiation rounds
    max_instantiation_rounds: u32,
    /// Maximum instantiations per round
    max_instantiations_per_round: usize,
    /// Set of already-instantiated formulas (for deduplication)
    seen_instances: HashSet<ExprKey>,
}

/// A pending universal quantifier hypothesis awaiting instantiation
#[derive(Clone, Debug)]
struct PendingForall {
    /// The quantified types (for potential type-based instantiation in the future)
    #[allow(dead_code)]
    tys: Vec<Expr>,
    /// The flattened body with BVars for all bound variables
    body: Expr,
    /// E-matching triggers extracted from the body
    triggers: Vec<crate::egraph::Trigger>,
    /// Bound variable indices for trigger matching
    bound_vars: Vec<u32>,
    /// Priority score for instantiation ordering (higher = instantiate first)
    priority: i32,
    /// Number of times this forall has been instantiated (for fairness)
    instantiation_count: u32,
}

// ============================================================================
// Quantifier Instantiation Priority Scoring
// ============================================================================

/// Heuristics for scoring quantifier instantiation priority.
///
/// The priority score determines the order in which pending universal
/// quantifiers are tried for E-matching instantiation. Higher scores
/// are instantiated first.
///
/// # Scoring Factors
///
/// Positive factors (increase priority):
/// - Good trigger quality (selective, ground-matchable patterns)
/// - Fewer bound variables (simpler instantiation)
/// - Single-pattern triggers (more efficient matching)
///
/// Negative factors (decrease priority):
/// - Many bound variables (expensive instantiation)
/// - Multi-pattern triggers (complex matching)
/// - High instantiation count (fairness - give other foralls a chance)
/// - Complex body (more overhead per instantiation)
#[derive(Clone, Debug, Default)]
pub struct QuantifierPriorityScorer {
    /// Weight for trigger quality score
    pub trigger_quality_weight: i32,
    /// Weight for bound variable count (negative = penalize many vars)
    pub bound_var_count_weight: i32,
    /// Weight for instantiation count (negative = penalize repeated use)
    pub instantiation_count_weight: i32,
    /// Bonus for single-pattern triggers
    pub single_trigger_bonus: i32,
    /// Penalty per additional trigger pattern
    pub multi_trigger_penalty: i32,
}

impl QuantifierPriorityScorer {
    /// Create a new scorer with default weights.
    ///
    /// Default weights are tuned for typical SMT-style quantifier instantiation:
    /// - Prefer simpler quantifiers (fewer variables)
    /// - Prefer quantifiers with good triggers
    /// - Ensure fairness by penalizing repeated instantiation
    pub fn new() -> Self {
        Self {
            trigger_quality_weight: 2,
            bound_var_count_weight: -5,
            instantiation_count_weight: -10,
            single_trigger_bonus: 15,
            multi_trigger_penalty: 3,
        }
    }

    /// Score a pending forall for instantiation priority.
    ///
    /// Returns an integer score where higher values indicate
    /// higher priority for instantiation.
    fn score(&self, pending: &PendingForall) -> i32 {
        let mut score: i32 = 0;

        // Trigger quality: use the best trigger's score
        // Higher depth and more children indicate more selective patterns
        let best_trigger_score = pending
            .triggers
            .iter()
            .map(|t| {
                // Sum pattern scores in the trigger
                t.patterns.iter().map(Self::pattern_score).sum::<i32>()
            })
            .max()
            .unwrap_or(0);
        score += best_trigger_score * self.trigger_quality_weight;

        // Bound variable count: fewer is better
        score += (pending.bound_vars.len() as i32) * self.bound_var_count_weight;

        // Single-pattern trigger bonus
        if pending.triggers.iter().any(|t| t.patterns.len() == 1) {
            score += self.single_trigger_bonus;
        }

        // Multi-trigger penalty: penalize each pattern beyond the first
        let min_patterns = pending
            .triggers
            .iter()
            .map(|t| t.patterns.len())
            .min()
            .unwrap_or(0);
        if min_patterns > 1 {
            score -= ((min_patterns - 1) as i32) * self.multi_trigger_penalty;
        }

        // Fairness: penalize quantifiers that have been instantiated many times
        score += (pending.instantiation_count as i32) * self.instantiation_count_weight;

        score
    }

    /// Compute a quality score for a single E-matching pattern.
    ///
    /// Higher scores indicate more selective patterns:
    /// - Deeper patterns are more selective (more structure to match)
    /// - More children means more constraints
    /// - Variables alone score 0 (match anything)
    fn pattern_score(pattern: &crate::egraph::Pattern) -> i32 {
        use crate::egraph::Pattern;
        match pattern {
            Pattern::Var(_) => 0,
            Pattern::App(_, children) => {
                // Base score for having a function symbol
                let mut score = 1;
                // Add depth bonus (recursive)
                for child in children {
                    score += Self::pattern_score(child);
                }
                // Add arity bonus
                score += children.len() as i32;
                score
            }
        }
    }
}

// ============================================================================
// Goal-Directed Quantifier Instantiation
// ============================================================================

/// Patterns extracted from the goal for guiding quantifier instantiation.
///
/// Goal-directed instantiation works "backward" from the goal:
/// - Extract patterns from terms in the goal (especially function applications)
/// - Prioritize quantifier instantiations that produce terms matching goal patterns
/// - This focuses proof search on instantiations likely to be useful
///
/// # Example
///
/// If proving `f(a) = g(b)` with hypothesis `∀x. P(x) → Q(f(x))`:
/// - Goal patterns include: `f(a)`, `g(b)`
/// - Instantiation with `x := a` is prioritized because `f(a)` matches the goal
#[derive(Clone, Debug, Default)]
pub struct GoalPatterns {
    /// Ground terms from the goal (function applications and constants)
    pub ground_terms: Vec<GroundTermPattern>,
    /// Function symbols that appear in the goal (for partial matching)
    pub function_symbols: HashSet<crate::egraph::Symbol>,
}

/// A ground term pattern extracted from the goal
#[derive(Clone, Debug)]
pub struct GroundTermPattern {
    /// The function symbol at the root
    pub symbol: crate::egraph::Symbol,
    /// Child term IDs in the E-graph (if the term was translated)
    pub children: Vec<TermId>,
    /// Arity of the function application
    pub arity: usize,
}

impl GoalPatterns {
    /// Create empty goal patterns
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any goal patterns exist
    pub fn is_empty(&self) -> bool {
        self.ground_terms.is_empty() && self.function_symbols.is_empty()
    }

    /// Check if a symbol appears in the goal
    pub fn contains_symbol(&self, sym: &crate::egraph::Symbol) -> bool {
        self.function_symbols.contains(sym)
    }

    /// Count how many goal patterns match a given trigger pattern.
    ///
    /// This computes a relevance score for a trigger pattern:
    /// - Higher score = trigger is more likely to produce goal-relevant instantiations
    pub fn relevance_score(&self, trigger: &crate::egraph::Trigger) -> i32 {
        let mut score = 0;

        for pattern in &trigger.patterns {
            score += self.pattern_relevance(pattern);
        }

        score
    }

    /// Compute relevance of a single pattern to the goal
    fn pattern_relevance(&self, pattern: &crate::egraph::Pattern) -> i32 {
        use crate::egraph::Pattern;
        match pattern {
            Pattern::Var(_) => 0, // Variables match anything, no specific relevance
            Pattern::App(sym, children) => {
                let mut score = 0;

                // Bonus if this function symbol appears in the goal
                if self.function_symbols.contains(sym) {
                    score += 10;
                }

                // Check if there's a ground term with matching symbol and arity
                let has_matching_ground = self
                    .ground_terms
                    .iter()
                    .any(|gt| gt.symbol == *sym && gt.arity == children.len());
                if has_matching_ground {
                    score += 20; // Strong bonus for exact structural match
                }

                // Recursive relevance from children
                for child in children {
                    score += self.pattern_relevance(child);
                }

                score
            }
        }
    }
}

/// Extracts goal patterns from a classified proposition.
///
/// This analyzer walks the goal expression and extracts:
/// 1. Ground terms (function applications with known children)
/// 2. Function symbols (for partial matching)
pub struct GoalPatternExtractor<'a> {
    /// Reference to the SMT bridge for term lookup
    expr_to_term: &'a HashMap<ExprKey, TermId>,
    /// Extracted patterns
    patterns: GoalPatterns,
}

impl<'a> GoalPatternExtractor<'a> {
    /// Create a new extractor
    fn new(expr_to_term: &'a HashMap<ExprKey, TermId>) -> Self {
        Self {
            expr_to_term,
            patterns: GoalPatterns::new(),
        }
    }

    /// Extract patterns from a classified proposition
    fn extract(&mut self, prop: &PropClass) -> GoalPatterns {
        self.extract_from_prop(prop);
        std::mem::take(&mut self.patterns)
    }

    /// Recursively extract patterns from a proposition
    fn extract_from_prop(&mut self, prop: &PropClass) {
        match prop {
            PropClass::Eq(lhs, rhs)
            | PropClass::Neq(lhs, rhs)
            | PropClass::Lt(lhs, rhs)
            | PropClass::Le(lhs, rhs)
            | PropClass::Gt(lhs, rhs)
            | PropClass::Ge(lhs, rhs) => {
                self.extract_from_term(lhs);
                self.extract_from_term(rhs);
            }
            PropClass::And(p, q) | PropClass::Or(p, q) | PropClass::Implies(p, q) => {
                // Recursively extract from sub-propositions
                self.extract_from_expr(p);
                self.extract_from_expr(q);
            }
            PropClass::Not(p) => {
                self.extract_from_expr(p);
            }
            PropClass::Forall(ty, body) | PropClass::Exists(ty, body) => {
                self.extract_from_term(ty);
                self.extract_from_expr(body);
            }
            PropClass::Atom(e) => {
                self.extract_from_term(e);
            }
            PropClass::True | PropClass::False => {}
        }
    }

    /// Extract patterns from an expression (for sub-propositions)
    fn extract_from_expr(&mut self, expr: &Expr) {
        // Try to classify as a proposition first
        // For now, treat as a term
        self.extract_from_term(expr);
    }

    /// Extract patterns from a term expression
    fn extract_from_term(&mut self, expr: &Expr) {
        match expr {
            Expr::App(_, _) => {
                // Collect the function and arguments
                let (head, args) = Self::collect_apps(expr);

                // Extract symbol from head
                if let Some(sym) = Self::expr_to_symbol(&head) {
                    self.patterns.function_symbols.insert(sym.clone());

                    // Collect child term IDs if available
                    let mut child_ids = Vec::new();
                    let mut all_children_known = true;

                    for arg in &args {
                        if let Some(key) = ExprKey::from_expr(arg) {
                            if let Some(&tid) = self.expr_to_term.get(&key) {
                                child_ids.push(tid);
                            } else {
                                all_children_known = false;
                            }
                        } else {
                            all_children_known = false;
                        }
                    }

                    // Add ground term pattern if we know all children
                    if all_children_known && !child_ids.is_empty() {
                        self.patterns.ground_terms.push(GroundTermPattern {
                            symbol: sym,
                            children: child_ids,
                            arity: args.len(),
                        });
                    }
                }

                // Recursively extract from arguments
                for arg in args {
                    self.extract_from_term(&arg);
                }
            }
            Expr::Const(name, _) => {
                // Constants become symbols
                let sym = crate::egraph::Symbol::new(name.to_string());
                self.patterns.function_symbols.insert(sym);
            }
            // Free/bound variables, sorts, and literals are not useful as patterns
            Expr::FVar(_) | Expr::BVar(_) | Expr::Sort(_) | Expr::Lit(_) => {}
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                self.extract_from_term(ty);
                self.extract_from_term(body);
            }
            Expr::Let(ty, val, body) => {
                self.extract_from_term(ty);
                self.extract_from_term(val);
                self.extract_from_term(body);
            }
            Expr::Proj(_, _, base) => {
                self.extract_from_term(base);
            }
            // MData is transparent - extract from inner
            Expr::MData(_, inner) => {
                self.extract_from_term(inner);
            }
            // Mode-specific expressions - no pattern extraction yet
            Expr::CubicalInterval
            | Expr::CubicalI0
            | Expr::CubicalI1
            | Expr::CubicalPath { .. }
            | Expr::CubicalPathLam { .. }
            | Expr::CubicalPathApp { .. }
            | Expr::CubicalHComp { .. }
            | Expr::CubicalTransp { .. }
            | Expr::ClassicalChoice { .. }
            | Expr::ClassicalEpsilon { .. }
            | Expr::ZFCSet(_)
            | Expr::ZFCMem { .. }
            | Expr::ZFCComprehension { .. }
            | Expr::SProp
            | Expr::Squash(_) => {}
        }
    }

    /// Collect nested applications into `(head, args)` tuple
    fn collect_apps(expr: &Expr) -> (Expr, Vec<Expr>) {
        let mut args = Vec::new();
        let mut curr = expr.clone();

        while let Expr::App(f, a) = curr {
            args.push((*a).clone());
            curr = (*f).clone();
        }

        args.reverse();
        (curr, args)
    }

    /// Convert an expression head to a Symbol
    fn expr_to_symbol(expr: &Expr) -> Option<crate::egraph::Symbol> {
        match expr {
            Expr::Const(name, _) => Some(crate::egraph::Symbol::new(name.to_string())),
            Expr::FVar(fv) => Some(crate::egraph::Symbol::new(format!("fvar_{}", fv.0))),
            _ => None,
        }
    }
}

/// Scorer for goal-directed quantifier instantiation.
///
/// Combines the standard priority scoring with goal-relevance bonuses.
#[derive(Clone, Debug)]
pub struct GoalDirectedScorer {
    /// Base priority scorer
    base_scorer: QuantifierPriorityScorer,
    /// Goal patterns for relevance computation
    goal_patterns: GoalPatterns,
    /// Weight for goal relevance bonus
    goal_relevance_weight: i32,
}

impl GoalDirectedScorer {
    /// Create a new goal-directed scorer
    pub fn new(goal_patterns: GoalPatterns) -> Self {
        Self {
            base_scorer: QuantifierPriorityScorer::new(),
            goal_patterns,
            goal_relevance_weight: 5,
        }
    }

    /// Create a scorer with custom weights
    pub fn with_weights(
        goal_patterns: GoalPatterns,
        base_scorer: QuantifierPriorityScorer,
        goal_relevance_weight: i32,
    ) -> Self {
        Self {
            base_scorer,
            goal_patterns,
            goal_relevance_weight,
        }
    }

    /// Score a pending forall with goal-directed bonus
    fn score(&self, pending: &PendingForall) -> i32 {
        // Start with base priority
        let mut score = self.base_scorer.score(pending);

        // Add goal relevance bonus for each trigger
        for trigger in &pending.triggers {
            let relevance = self.goal_patterns.relevance_score(trigger);
            score += relevance * self.goal_relevance_weight;
        }

        score
    }

    /// Check if goal patterns are available
    pub fn has_goal_patterns(&self) -> bool {
        !self.goal_patterns.is_empty()
    }
}

// ============================================================================
// Mixed Quantifier Scope Analysis
// ============================================================================

/// Quantifier kind (universal or existential)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantifierKind {
    /// Universal quantifier (∀)
    Forall,
    /// Existential quantifier (∃)
    Exists,
}

/// A single binder in a quantifier prefix
#[derive(Clone, Debug)]
pub struct QuantifierBinder {
    /// The kind of quantifier (∀ or ∃)
    pub kind: QuantifierKind,
    /// The type of the bound variable
    pub ty: Expr,
    /// The de Bruijn index of this binder in the flattened context
    pub index: u32,
}

/// Quantifier prefix representing a sequence of quantifiers
///
/// For example, `∀x. ∃y. ∀z. P(x,y,z)` is represented as:
/// - binders: [(∀, A, 2), (∃, B, 1), (∀, C, 0)]
/// - body: P(x,y,z) with BVars 2, 1, 0
///
/// The index is the de Bruijn index in the flattened body.
#[derive(Clone, Debug)]
pub struct QuantifierPrefix {
    /// The sequence of quantifier binders from outermost to innermost
    pub binders: Vec<QuantifierBinder>,
    /// The innermost body (with BVars for all bound variables)
    pub body: Expr,
}

impl QuantifierPrefix {
    /// Create a new empty quantifier prefix
    pub fn new(body: Expr) -> Self {
        Self {
            binders: Vec::new(),
            body,
        }
    }

    /// Get the number of quantifiers in this prefix
    pub fn len(&self) -> usize {
        self.binders.len()
    }

    /// Check if the prefix is empty (no quantifiers)
    pub fn is_empty(&self) -> bool {
        self.binders.is_empty()
    }

    /// Get the alternation depth of this quantifier prefix
    ///
    /// The alternation depth is the number of times the quantifier kind changes.
    /// - `∀x. P(x)` has depth 0 (no alternation)
    /// - `∀x. ∃y. P(x,y)` has depth 1 (one alternation: ∀→∃)
    /// - `∀x. ∃y. ∀z. P(x,y,z)` has depth 2 (two alternations: ∀→∃→∀)
    /// - `∃x. ∃y. P(x,y)` has depth 0 (no alternation, all ∃)
    ///
    /// Higher alternation depth generally indicates harder formulas to decide.
    pub fn alternation_depth(&self) -> u32 {
        if self.binders.is_empty() {
            return 0;
        }

        let mut depth = 0;
        let mut prev_kind = self.binders[0].kind;

        for binder in &self.binders[1..] {
            if binder.kind != prev_kind {
                depth += 1;
                prev_kind = binder.kind;
            }
        }

        depth
    }

    /// Check if this is a purely universal prefix (∀...∀)
    pub fn is_purely_universal(&self) -> bool {
        self.binders
            .iter()
            .all(|b| b.kind == QuantifierKind::Forall)
    }

    /// Check if this is a purely existential prefix (∃...∃)
    pub fn is_purely_existential(&self) -> bool {
        self.binders
            .iter()
            .all(|b| b.kind == QuantifierKind::Exists)
    }

    /// Get the outermost quantifier kind, if any
    pub fn outermost_kind(&self) -> Option<QuantifierKind> {
        self.binders.first().map(|b| b.kind)
    }

    /// Get indices of all universal variables
    pub fn forall_indices(&self) -> Vec<u32> {
        self.binders
            .iter()
            .filter(|b| b.kind == QuantifierKind::Forall)
            .map(|b| b.index)
            .collect()
    }

    /// Get indices of all existential variables
    pub fn exists_indices(&self) -> Vec<u32> {
        self.binders
            .iter()
            .filter(|b| b.kind == QuantifierKind::Exists)
            .map(|b| b.index)
            .collect()
    }

    /// Get the dependencies for Skolemization
    ///
    /// For each existential variable, returns the indices of universal variables
    /// that appear before it in the prefix (which the Skolem function should depend on).
    ///
    /// Example: `∀x. ∃y. ∀z. ∃w. P(x,y,z,w)`
    /// - y depends on \[x\]
    /// - w depends on \[x, z\]
    pub fn skolem_dependencies(&self) -> HashMap<u32, Vec<u32>> {
        let mut deps = HashMap::new();
        let mut preceding_foralls = Vec::new();

        for binder in &self.binders {
            match binder.kind {
                QuantifierKind::Forall => {
                    preceding_foralls.push(binder.index);
                }
                QuantifierKind::Exists => {
                    deps.insert(binder.index, preceding_foralls.clone());
                }
            }
        }

        deps
    }
}

/// Key for hashing expressions (avoiding Arc comparison issues)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ExprKey {
    BVar(u32),
    FVar(FVarId),
    Const(Name),
    App(Box<ExprKey>, Box<ExprKey>),
    Lam(Box<ExprKey>, Box<ExprKey>),
    Pi(Box<ExprKey>, Box<ExprKey>),
    Lit(LitKey),
}

impl ExprKey {
    /// Convert an expression to a hashable key (static version)
    pub fn from_expr(expr: &Expr) -> Option<ExprKey> {
        match expr {
            Expr::BVar(idx) => Some(ExprKey::BVar(*idx)),
            Expr::FVar(fvar_id) => Some(ExprKey::FVar(*fvar_id)),
            Expr::Const(name, _) => Some(ExprKey::Const(name.clone())),
            Expr::App(f, a) => {
                let f_key = Self::from_expr(f)?;
                let a_key = Self::from_expr(a)?;
                Some(ExprKey::App(Box::new(f_key), Box::new(a_key)))
            }
            Expr::Lam(_, ty, body) => {
                let ty_key = Self::from_expr(ty)?;
                let body_key = Self::from_expr(body)?;
                Some(ExprKey::Lam(Box::new(ty_key), Box::new(body_key)))
            }
            Expr::Pi(_, ty, body) => {
                let ty_key = Self::from_expr(ty)?;
                let body_key = Self::from_expr(body)?;
                Some(ExprKey::Pi(Box::new(ty_key), Box::new(body_key)))
            }
            Expr::Lit(lit) => match lit {
                lean5_kernel::expr::Literal::Nat(n) => Some(ExprKey::Lit(LitKey::Nat(*n))),
                lean5_kernel::expr::Literal::String(s) => {
                    Some(ExprKey::Lit(LitKey::String(s.to_string())))
                }
            },
            _ => None, // Don't cache Sort, Let, Proj
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum LitKey {
    Nat(u64),
    String(String),
}

impl<'env> SmtBridge<'env> {
    /// Create a new SMT bridge
    pub fn new(env: &'env Environment) -> Self {
        let mut smt = SmtSolver::new();
        // Add equality theory by default
        let eq_idx = smt.add_theory(Box::new(EqualityTheory::new()));
        // Add arithmetic theory for Lt/Le comparisons
        let _arith_idx = smt.add_theory(Box::new(ArithmeticTheory::new()));

        SmtBridge {
            env,
            smt,
            expr_to_term: HashMap::new(),
            term_to_expr: HashMap::new(),
            term_to_type: HashMap::new(),
            fvar_to_term: HashMap::new(),
            fresh_counter: 0,
            eq_hypothesis_canonical: HashMap::new(),
            equality_theory_idx: Some(eq_idx),
            pending_foralls: Vec::new(),
            max_instantiation_rounds: 3,
            max_instantiations_per_round: 10,
            seen_instances: HashSet::new(),
        }
    }

    /// Get a typed reference to the equality theory solver
    fn equality_theory(&self) -> Option<&EqualityTheory> {
        self.equality_theory_idx
            .and_then(|idx| self.smt.get_theory_typed::<EqualityTheory>(idx))
    }

    /// Get a mutable typed reference to the equality theory solver
    fn equality_theory_mut(&mut self) -> Option<&mut EqualityTheory> {
        self.equality_theory_idx
            .and_then(|idx| self.smt.get_theory_typed_mut::<EqualityTheory>(idx))
    }

    /// Set the maximum number of E-matching instantiation rounds
    pub fn set_max_instantiation_rounds(&mut self, rounds: u32) {
        self.max_instantiation_rounds = rounds;
    }

    /// Set the maximum number of instantiations per round
    pub fn set_max_instantiations_per_round(&mut self, count: usize) {
        self.max_instantiations_per_round = count;
    }

    /// Try to prove a proposition using SMT
    ///
    /// For a goal `P`, we check if `¬P` is unsatisfiable.
    /// If UNSAT, then `P` must be true.
    ///
    /// Returns Some(proof_term) if proven, None if cannot prove.
    pub fn prove(&mut self, goal: &Expr) -> Option<ProofResult> {
        // Classify the goal to know what we're proving
        let goal_class = self.classify_prop(goal);

        // For equality goals, remember the terms for proof reconstruction
        let eq_goal_terms = match &goal_class {
            PropClass::Eq(lhs, rhs) => {
                let t1 = self.translate_term(lhs)?;
                let t2 = self.translate_term(rhs)?;
                Some((t1, t2, lhs.clone(), rhs.clone()))
            }
            _ => None,
        };

        // Translate the goal to SMT and assert its negation
        self.translate_negated_classified(&goal_class)?;

        // Initial solve attempt
        let result = self.smt.solve();

        // If unknown/sat and we have pending foralls, try E-matching instantiation
        if !matches!(result, SmtResult::Unsat) && !self.pending_foralls.is_empty() {
            if let Some(proof) = self.prove_with_ematching(&goal_class, &eq_goal_terms) {
                return Some(proof);
            }
        }

        // Process initial result
        match result {
            SmtResult::Unsat => {
                // The negation is unsatisfiable, so the goal is valid
                // Try to build a proof term
                if let Some((t1, t2, lhs_expr, rhs_expr)) = eq_goal_terms {
                    // Try to reconstruct proof for equality goal
                    if let Some((proof_step, proof_term)) =
                        self.build_equality_proof(t1, t2, &lhs_expr, &rhs_expr)
                    {
                        return Some(ProofResult::with_proof_term(
                            ProofMethod::SmtUnsat,
                            "SMT proved equality via E-graph",
                            proof_term,
                            proof_step,
                        ));
                    }
                }

                // Fallback: no proof reconstruction available yet
                Some(ProofResult::new(
                    ProofMethod::SmtUnsat,
                    "SMT proved validity of goal (proof reconstruction not available)",
                ))
            }
            SmtResult::Sat(_model) => {
                // Found a counterexample, goal is not valid
                None
            }
            SmtResult::Unknown => {
                // SMT solver couldn't determine
                None
            }
        }
    }

    /// Try to prove with E-matching instantiation rounds
    ///
    /// This performs iterative quantifier instantiation using E-matching:
    /// 1. Get the E-graph from the equality theory
    /// 2. For each pending forall, match triggers against the E-graph
    /// 3. Instantiate the forall body with matched terms
    /// 4. Add instantiated formulas as new hypotheses
    /// 5. Re-solve and repeat up to max_instantiation_rounds
    fn prove_with_ematching(
        &mut self,
        goal_class: &PropClass,
        eq_goal_terms: &Option<(TermId, TermId, Expr, Expr)>,
    ) -> Option<ProofResult> {
        let max_rounds = self.max_instantiation_rounds;
        let max_per_round = self.max_instantiations_per_round;

        // Extract goal patterns for goal-directed instantiation
        let goal_patterns = {
            let mut extractor = GoalPatternExtractor::new(&self.expr_to_term);
            extractor.extract(goal_class)
        };

        // Re-score pending foralls with goal-directed scoring if we have patterns
        if !goal_patterns.is_empty() {
            let scorer = GoalDirectedScorer::new(goal_patterns);
            for pending in &mut self.pending_foralls {
                pending.priority = scorer.score(pending);
            }
        }

        for _round in 0..max_rounds {
            // Get current E-graph state
            let new_instances = self.collect_ematching_instances(max_per_round);

            if new_instances.is_empty() {
                // No new instances to add, E-matching saturated
                break;
            }

            // Add instantiated formulas as hypotheses
            for inst in new_instances {
                // Instantiation is already an Expr (the body with substitution applied)
                let _ = self.add_hypothesis(&inst);
            }

            // Re-solve with new instances
            match self.smt.solve() {
                SmtResult::Unsat => {
                    // Success! Try to build proof
                    if let Some((t1, t2, lhs_expr, rhs_expr)) = eq_goal_terms {
                        if let Some((proof_step, proof_term)) =
                            self.build_equality_proof(*t1, *t2, lhs_expr, rhs_expr)
                        {
                            return Some(ProofResult::with_proof_term(
                                ProofMethod::SmtUnsat,
                                "SMT proved via E-matching instantiation",
                                proof_term,
                                proof_step,
                            ));
                        }
                    }
                    return Some(ProofResult::new(
                        ProofMethod::SmtUnsat,
                        "SMT proved via E-matching instantiation",
                    ));
                }
                SmtResult::Sat(_) | SmtResult::Unknown => {
                    // Continue to next round
                }
            }
        }

        None
    }

    /// Collect E-matching instances from pending foralls
    ///
    /// Uses the E-graph to find matching terms for trigger patterns,
    /// then instantiates forall bodies with those terms.
    /// Deduplicates instances to avoid redundant instantiations across rounds.
    ///
    /// Quantifiers are processed in priority order (higher priority first).
    /// This ensures that simpler, more selective quantifiers are instantiated
    /// before complex ones, improving proof search efficiency.
    fn collect_ematching_instances(&mut self, max_instances: usize) -> Vec<Expr> {
        use crate::egraph::EMatcher;

        // Sort pending_foralls by priority (highest first) before processing
        self.pending_foralls
            .sort_by(|a, b| b.priority.cmp(&a.priority));

        // Phase 1: Collect all candidate substitutions without mutable self borrow
        // Track which forall indices produced instances for fairness updates
        let candidate_substitutions: Vec<(usize, Expr, crate::egraph::Substitution, Vec<u32>)> = {
            // Get E-graph from equality theory
            let egraph = match self.equality_theory() {
                Some(eq) => eq.egraph(),
                None => return Vec::new(),
            };

            let matcher = EMatcher::new(egraph);
            let mut candidates = Vec::new();

            for (forall_idx, pending) in self.pending_foralls.iter().enumerate() {
                if candidates.len() >= max_instances * 2 {
                    // Collect extra to account for duplicates
                    break;
                }

                // Try each trigger pattern
                for trigger in &pending.triggers {
                    if candidates.len() >= max_instances * 2 {
                        break;
                    }

                    // Find all matches for this trigger
                    let substitutions = matcher.find_multi_matches(&trigger.patterns);

                    for subst in substitutions {
                        if candidates.len() >= max_instances * 2 {
                            break;
                        }
                        candidates.push((
                            forall_idx,
                            pending.body.clone(),
                            subst,
                            pending.bound_vars.clone(),
                        ));
                    }
                }
            }
            candidates
        };

        // Phase 2: Instantiate and deduplicate with mutable access to seen_instances
        // Also track which foralls produced new instances for fairness
        let mut instances = Vec::new();
        let mut foralls_with_new_instances: HashSet<usize> = HashSet::new();

        for (forall_idx, body, subst, bound_vars) in candidate_substitutions {
            if instances.len() >= max_instances {
                break;
            }

            // Convert substitution (EClassId bindings) to expression substitution
            if let Some(inst_expr) = self.instantiate_from_substitution(&body, &subst, &bound_vars)
            {
                // Deduplicate: only add if we haven't seen this instance before
                if let Some(key) = self.expr_to_key(&inst_expr) {
                    if self.seen_instances.insert(key) {
                        // New instance, add it
                        instances.push(inst_expr);
                        foralls_with_new_instances.insert(forall_idx);
                    }
                    // else: duplicate, skip
                } else {
                    // Can't create key (complex expression), add anyway
                    instances.push(inst_expr);
                    foralls_with_new_instances.insert(forall_idx);
                }
            }
        }

        // Phase 3: Update instantiation counts and re-score priorities
        // This ensures fairness: frequently-used quantifiers get lower priority
        let scorer = QuantifierPriorityScorer::new();
        for idx in foralls_with_new_instances {
            if idx < self.pending_foralls.len() {
                self.pending_foralls[idx].instantiation_count += 1;
                self.pending_foralls[idx].priority = scorer.score(&self.pending_foralls[idx]);
            }
        }

        instances
    }

    /// Instantiate a forall body using an E-matching substitution
    ///
    /// The substitution maps pattern variable names ("?x0", "?x1", etc.)
    /// to E-class IDs. We need to convert these to expressions.
    fn instantiate_from_substitution(
        &self,
        body: &Expr,
        subst: &crate::egraph::Substitution,
        bound_vars: &[u32],
    ) -> Option<Expr> {
        // For each bound variable index, get the corresponding term from substitution
        let mut replacements: Vec<(u32, Expr)> = Vec::new();

        for &bvar_idx in bound_vars {
            let var_name = format!("?x{bvar_idx}");
            if let Some(eclass_id) = subst.get(&var_name) {
                // Find an expression for this E-class
                if let Some(expr) = self.eclass_to_expr(eclass_id) {
                    replacements.push((bvar_idx, expr));
                }
            }
        }

        // Check we have all variables
        if replacements.len() != bound_vars.len() {
            return None;
        }

        // Substitute all bound variables in the body (descending order inside)
        Some(self.instantiate_bvars(body, &replacements))
    }

    /// Convert an E-class ID to an expression
    ///
    /// Looks up the term_to_expr mapping to find an expression
    /// that was mapped to a term in this E-class.
    fn eclass_to_expr(&self, eclass_id: crate::egraph::EClassId) -> Option<Expr> {
        // The equality theory maps TermId -> EClassId
        // We need to find a TermId in this E-class and look up its expression
        let eq = self.equality_theory()?;

        for (&term_id, &mapped_eclass) in eq.term_to_eclass_map() {
            if eq.egraph().find_const(mapped_eclass) == eq.egraph().find_const(eclass_id) {
                if let Some(expr) = self.term_to_expr.get(&term_id) {
                    return Some(expr.clone());
                }
            }
        }

        None
    }

    /// Try to build a proof from the equality theory's proof trace.
    ///
    /// This leverages the E-graph union reasons (including hypotheses) recorded
    /// by the EUF solver. Falls back to reflexivity when both terms are already
    /// in the same equivalence class.
    fn proof_from_equality_theory(&self, t1: TermId, t2: TermId) -> Option<(ProofStep, Expr)> {
        let eq = self.equality_theory()?;
        let ec1 = eq.get_eclass(t1)?;
        let ec2 = eq.get_eclass(t2)?;

        let step = eq.proof_trace().build_proof(ec1, ec2)?;

        let builder = ProofBuilder::new(&self.term_to_expr, &self.term_to_type);
        let proof_term = builder.build(&step)?;
        Some((step, proof_term))
    }

    /// Build a proof term for an equality goal.
    ///
    /// # Proof Construction Strategy
    ///
    /// We use multiple strategies in priority order:
    ///
    /// 1. **Reflexivity**: If t1 == t2, return Eq.refl
    ///
    /// 2. **E-graph proof trace** (primary path): Uses the equality theory's
    ///    ProofTrace which records union reasons with hypothesis tracking.
    ///    Handles transitivity, congruence, and mixed proofs naturally.
    ///
    /// 3. **Direct hypothesis lookup**: For simple cases where a single
    ///    hypothesis proves the goal (with optional symmetry).
    ///
    /// 4. **BFS transitivity**: Fallback for when the proof trace doesn't
    ///    have the path (shouldn't happen normally, but useful for testing).
    ///
    /// 5. **Term-based congruence**: Fallback for congruence when E-graph
    ///    doesn't have the proof recorded.
    fn build_equality_proof(
        &self,
        t1: TermId,
        t2: TermId,
        lhs_expr: &Expr,
        rhs_expr: &Expr,
    ) -> Option<(ProofStep, Expr)> {
        // Strategy 1: Reflexivity
        if t1 == t2 {
            let proof_step = ProofStep::refl(t1);
            let ty = self.get_type_for_term(t1);
            let proof_term = self.mk_eq_refl(&ty, lhs_expr);
            return Some((proof_step, proof_term));
        }

        // Strategy 2: E-graph proof trace (primary path)
        // This is the preferred approach as it handles all proof types uniformly
        if let Some((step, proof)) = self.proof_from_equality_theory(t1, t2) {
            return Some((step, proof));
        }

        // Strategy 3: Direct hypothesis lookup
        // Used when the E-graph trace doesn't have the proof but we have a hypothesis
        if let Some(&fvar) = self.eq_hypothesis_canonical.get(&(t1, t2)) {
            let proof_step = ProofStep::hypothesis(fvar);
            let proof_term = Expr::fvar(fvar);
            return Some((proof_step, proof_term));
        }

        // Check hypothesis in reverse direction (needs symmetry)
        if let Some(&fvar) = self.eq_hypothesis_canonical.get(&(t2, t1)) {
            let proof_step = ProofStep::symm(ProofStep::hypothesis(fvar));
            let proof_term = self.mk_eq_symm(&Expr::fvar(fvar));
            return Some((proof_step, proof_term));
        }

        // Strategy 4: BFS transitivity (fallback)
        if let Some((proof_step, proof_term)) =
            self.try_transitive_proof(t1, t2, lhs_expr, rhs_expr)
        {
            return Some((proof_step, proof_term));
        }

        // Strategy 5: Term-based congruence (fallback)
        if let Some((proof_step, proof_term)) =
            self.try_congruence_proof(t1, t2, lhs_expr, rhs_expr)
        {
            return Some((proof_step, proof_term));
        }

        None
    }

    /// Try to build a transitive proof using BFS to find a path from t1 to t2.
    ///
    /// Supports arbitrary-length chains: a=b, b=c, c=d, ... → a=z
    fn try_transitive_proof(
        &self,
        t1: TermId,
        t2: TermId,
        _lhs_expr: &Expr,
        _rhs_expr: &Expr,
    ) -> Option<(ProofStep, Expr)> {
        use std::collections::{HashMap as BfsMap, VecDeque};

        // BFS to find shortest path from t1 to t2 through equality hypotheses
        // Each edge represents a hypothesis (with possible symmetry)

        // Build adjacency list from hypotheses
        // neighbor_term -> (hypothesis_fvar, needs_symm_to_reach_neighbor)
        let mut adjacency: BfsMap<TermId, Vec<(TermId, FVarId, bool)>> = BfsMap::new();
        for (&(a, b), &fvar) in &self.eq_hypothesis_canonical {
            // a = b: from a, reach b without symm; from b, reach a with symm
            adjacency.entry(a).or_default().push((b, fvar, false));
            adjacency.entry(b).or_default().push((a, fvar, true));
        }

        // BFS: track (current_term, path_to_here)
        // path is list of (fvar, needs_symm) representing edges taken
        let mut queue: VecDeque<(TermId, Vec<(FVarId, bool)>)> = VecDeque::new();
        let mut visited: std::collections::HashSet<TermId> = std::collections::HashSet::new();

        queue.push_back((t1, vec![]));
        visited.insert(t1);

        while let Some((current, path)) = queue.pop_front() {
            if current == t2 {
                // Found path! Build proof from path
                if path.is_empty() {
                    // t1 == t2, should have been handled as reflexivity
                    return None;
                }
                return Some(self.build_path_proof(&path));
            }

            // Explore neighbors
            if let Some(neighbors) = adjacency.get(&current) {
                for &(neighbor, fvar, needs_symm) in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        let mut new_path = path.clone();
                        new_path.push((fvar, needs_symm));
                        queue.push_back((neighbor, new_path));
                    }
                }
            }
        }

        // No path found
        None
    }

    /// Build a proof term from a path of hypothesis edges.
    ///
    /// Path is a list of (fvar, needs_symm) representing:
    /// - fvar: the hypothesis providing the equality
    /// - needs_symm: whether to apply Eq.symm to use this hypothesis
    fn build_path_proof(&self, path: &[(FVarId, bool)]) -> (ProofStep, Expr) {
        assert!(!path.is_empty(), "Path must be non-empty");

        // Build proof for first edge
        let (fvar, needs_symm) = path[0];
        let mut current_step = if needs_symm {
            ProofStep::symm(ProofStep::hypothesis(fvar))
        } else {
            ProofStep::hypothesis(fvar)
        };
        let mut current_term = if needs_symm {
            self.mk_eq_symm(&Expr::fvar(fvar))
        } else {
            Expr::fvar(fvar)
        };

        // Chain remaining edges with transitivity
        for &(fvar, needs_symm) in &path[1..] {
            let next_step = if needs_symm {
                ProofStep::symm(ProofStep::hypothesis(fvar))
            } else {
                ProofStep::hypothesis(fvar)
            };
            let next_term = if needs_symm {
                self.mk_eq_symm(&Expr::fvar(fvar))
            } else {
                Expr::fvar(fvar)
            };

            current_step = ProofStep::trans(current_step, next_step);
            current_term = self.mk_eq_trans(&current_term, &next_term);
        }

        (current_step, current_term)
    }

    /// Try to build a congruence proof: a=b → f(a)=f(b)
    ///
    /// For multi-argument functions f(a₁, a₂, ...) = f(b₁, b₂, ...):
    /// - First find proofs that each aᵢ = bᵢ
    /// - Chain them using: congr (congrArg f h₁) h₂ etc.
    fn try_congruence_proof(
        &self,
        t1: TermId,
        t2: TermId,
        lhs_expr: &Expr,
        rhs_expr: &Expr,
    ) -> Option<(ProofStep, Expr)> {
        // Get the SMT terms for t1 and t2
        let smt_t1 = self.smt.get_term(t1)?;
        let smt_t2 = self.smt.get_term(t2)?;

        // Both must be applications with the same function symbol
        let (func_name, args1) = match smt_t1 {
            crate::smt::SmtTerm::App(name, args) => (name.name().to_string(), args.clone()),
            _ => return None,
        };

        let args2 = match smt_t2 {
            crate::smt::SmtTerm::App(name, args) if name.name() == func_name => args.clone(),
            _ => return None,
        };

        // Must have same number of arguments
        if args1.len() != args2.len() {
            return None;
        }

        if args1.is_empty() {
            return None; // No arguments to compare
        }

        // Collect proofs for each argument pair
        let mut arg_steps: Vec<ProofStep> = Vec::new();
        let mut arg_proofs: Vec<Expr> = Vec::new();

        for (arg1, arg2) in args1.iter().zip(args2.iter()) {
            let arg1_expr = self.term_to_expr.get(arg1)?;
            let arg2_expr = self.term_to_expr.get(arg2)?;

            // Recursively try to build proof for this argument equality
            let (arg_step, arg_proof) =
                self.build_equality_proof(*arg1, *arg2, arg1_expr, arg2_expr)?;
            arg_steps.push(arg_step);
            arg_proofs.push(arg_proof);
        }

        // Build the composite proof step
        let proof_step = ProofStep::congr(func_name.clone(), arg_steps);

        // Build the proof term
        // For single argument: congrArg f h
        // For multiple arguments: congr (congrArg f h₁) h₂ (for 2 args)
        //                         congr (congr (congrArg f h₁) h₂) h₃ (for 3 args)
        //                         etc.

        let func_expr = match lhs_expr.get_app_fn() {
            Expr::Const(name, levels) => Expr::const_(name.clone(), levels.clone()),
            Expr::FVar(fvar) => Expr::fvar(*fvar),
            _ => return None,
        };

        let proof_term = if arg_proofs.len() == 1 {
            // Single argument: congrArg f h
            let ty = self.get_type_for_term(args1[0]);
            self.mk_congr_arg(&ty, &func_expr, &arg_proofs[0])
        } else {
            // Multiple arguments: chain with congr
            // Start with congrArg for first argument: f = f, but we need f a₁ = f b₁
            // Actually for f(a₁)(a₂) vs f(b₁)(b₂), we build:
            //   h₁ : a₁ = b₁
            //   congrArg f h₁ : f a₁ = f b₁
            //   h₂ : a₂ = b₂
            //   congr (congrArg f h₁) h₂ : f a₁ a₂ = f b₁ b₂

            let ty = self.get_type_for_term(args1[0]);
            let mut current_proof = self.mk_congr_arg(&ty, &func_expr, &arg_proofs[0]);

            // Chain remaining arguments with congr
            for proof in &arg_proofs[1..] {
                current_proof = self.mk_congr(&current_proof, proof);
            }

            current_proof
        };

        let _ = rhs_expr; // suppress warning
        Some((proof_step, proof_term))
    }

    /// Build congr : ∀ {α β : Sort u} {f₁ f₂ : α → β} {a₁ a₂ : α},
    ///               f₁ = f₂ → a₁ = a₂ → f₁ a₁ = f₂ a₂
    fn mk_congr(&self, func_proof: &Expr, arg_proof: &Expr) -> Expr {
        // congr hf ha : f₁ a₁ = f₂ a₂
        Expr::app(
            Expr::app(
                Expr::const_(Name::from_string("congr"), vec![Level::succ(Level::zero())]),
                func_proof.clone(),
            ),
            arg_proof.clone(),
        )
    }

    /// Build congrArg : ∀ {α β : Sort u} (f : α → β) {a₁ a₂ : α}, a₁ = a₂ → f a₁ = f a₂
    fn mk_congr_arg(&self, _ty: &Expr, func: &Expr, arg_proof: &Expr) -> Expr {
        // congrArg f h : f a₁ = f a₂
        Expr::app(
            Expr::app(
                Expr::const_(
                    Name::from_string("congrArg"),
                    vec![Level::succ(Level::zero())],
                ),
                func.clone(),
            ),
            arg_proof.clone(),
        )
    }

    /// Get the type for a term (or default to Type)
    fn get_type_for_term(&self, term: TermId) -> Expr {
        self.term_to_type
            .get(&term)
            .cloned()
            .unwrap_or_else(Expr::type_)
    }

    /// Build Eq.refl : ∀ {α : Sort u} (a : α), a = a
    fn mk_eq_refl(&self, ty: &Expr, val: &Expr) -> Expr {
        Expr::app(
            Expr::app(
                Expr::const_(
                    Name::from_string("Eq.refl"),
                    vec![Level::succ(Level::zero())],
                ),
                ty.clone(),
            ),
            val.clone(),
        )
    }

    /// Build Eq.symm : ∀ {α : Sort u} {a b : α}, a = b → b = a
    fn mk_eq_symm(&self, proof: &Expr) -> Expr {
        Expr::app(
            Expr::const_(
                Name::from_string("Eq.symm"),
                vec![Level::succ(Level::zero())],
            ),
            proof.clone(),
        )
    }

    /// Build Eq.trans : ∀ {α : Sort u} {a b c : α}, a = b → b = c → a = c
    fn mk_eq_trans(&self, p1: &Expr, p2: &Expr) -> Expr {
        Expr::app(
            Expr::app(
                Expr::const_(
                    Name::from_string("Eq.trans"),
                    vec![Level::succ(Level::zero())],
                ),
                p1.clone(),
            ),
            p2.clone(),
        )
    }

    /// Translate a classified proposition and assert its negation
    fn translate_negated_classified(&mut self, prop: &PropClass) -> Option<()> {
        match prop {
            PropClass::Eq(lhs, rhs) => {
                let t1 = self.translate_term(lhs)?;
                let t2 = self.translate_term(rhs)?;
                self.smt.assert_neq(t1, t2);
                Some(())
            }
            PropClass::Neq(lhs, rhs) => {
                let t1 = self.translate_term(lhs)?;
                let t2 = self.translate_term(rhs)?;
                self.smt.assert_eq(t1, t2);
                Some(())
            }
            // Arithmetic comparisons: negate by asserting the opposite
            PropClass::Lt(lhs, rhs) => {
                // ¬(a < b) ≡ b ≤ a
                let t1 = self.translate_term(lhs)?;
                let t2 = self.translate_term(rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Le(t2, t1)]);
                Some(())
            }
            PropClass::Le(lhs, rhs) => {
                // ¬(a ≤ b) ≡ b < a
                let t1 = self.translate_term(lhs)?;
                let t2 = self.translate_term(rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Lt(t2, t1)]);
                Some(())
            }
            PropClass::Gt(lhs, rhs) => {
                // ¬(a > b) ≡ a ≤ b
                let t1 = self.translate_term(lhs)?;
                let t2 = self.translate_term(rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Le(t1, t2)]);
                Some(())
            }
            PropClass::Ge(lhs, rhs) => {
                // ¬(a ≥ b) ≡ a < b
                let t1 = self.translate_term(lhs)?;
                let t2 = self.translate_term(rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Lt(t1, t2)]);
                Some(())
            }
            PropClass::And(p, q) => {
                let np = self.prop_to_literal(p, false)?;
                let nq = self.prop_to_literal(q, false)?;
                self.smt.add_clause(vec![np, nq]);
                Some(())
            }
            PropClass::Or(p, q) => {
                let np = self.prop_to_literal(p, false)?;
                let nq = self.prop_to_literal(q, false)?;
                self.smt.add_clause(vec![np.clone()]);
                self.smt.add_clause(vec![nq.clone()]);
                Some(())
            }
            PropClass::Implies(p, q) => {
                let pp = self.prop_to_literal(p, true)?;
                let nq = self.prop_to_literal(q, false)?;
                self.smt.add_clause(vec![pp.clone()]);
                self.smt.add_clause(vec![nq.clone()]);
                Some(())
            }
            PropClass::Not(p) => {
                let pp = self.prop_to_literal(p, true)?;
                self.smt.add_clause(vec![pp]);
                Some(())
            }
            PropClass::Forall(_ty, body) => {
                // ¬(∀ x : T, P(x)) ≡ ∃ x : T, ¬P(x)
                // For SMT: We instantiate with a fresh Skolem constant
                // This is sound for refutation: if the Skolemized version is UNSAT,
                // the original forall is valid.
                let (bound_types, flat_body) = self.flatten_forall(_ty, body);
                let bound_count = u32::try_from(bound_types.len())
                    .expect("forall bound variable count exceeded u32::MAX");
                let bound_vars: Vec<u32> = (0..bound_count).collect();

                let mut witness_terms = Vec::new();
                for i in 0..bound_types.len() {
                    let skolem_name = format!("sk_{}_{}", i, self.fresh_counter);
                    self.fresh_counter += 1;
                    witness_terms.push(self.smt.const_term(skolem_name));
                }

                let instantiated_body =
                    self.instantiate_body_with_terms(&flat_body, &bound_vars, &witness_terms);
                if let Some(inst) = instantiated_body {
                    self.translate_negated_classified(&self.classify_prop(&inst))
                } else {
                    // Fall back to treating as atom
                    let var_id = self.fresh_counter;
                    self.fresh_counter += 1;
                    self.smt.add_clause(vec![TheoryLiteral::Bool(var_id)]);
                    Some(())
                }
            }
            PropClass::Exists(_ty, body) => {
                // ¬(∃ x : T, P(x)) ≡ ∀ x : T, ¬P(x)
                // For SMT: We need to prove ¬P(x) for all x
                // With ground SMT, we can only check finite instantiations
                // For soundness, treat as unhandled (conservative)
                let _ = body; // suppress warning
                              // Fall back to treating as atom
                let var_id = self.fresh_counter;
                self.fresh_counter += 1;
                self.smt.add_clause(vec![TheoryLiteral::Bool(var_id)]);
                Some(())
            }
            PropClass::True | PropClass::False => Some(()),
            PropClass::Atom(expr) => {
                let lit = self.prop_to_literal(expr, false)?;
                self.smt.add_clause(vec![lit]);
                Some(())
            }
        }
    }

    /// Flatten nested forall binders into a list of types and the innermost body.
    fn flatten_forall(&self, first_ty: &Expr, body: &Expr) -> (Vec<Expr>, Expr) {
        let mut types = vec![first_ty.clone()];
        let mut current = body.clone();

        // Keep peeling Pis that still bind variables (dependent Pis)
        while let Expr::Pi(_, ty, codomain) = &current {
            if !codomain.has_loose_bvars() {
                break;
            }
            types.push(ty.as_ref().clone());
            current = codomain.as_ref().clone();
        }

        (types, current)
    }

    /// Flatten nested exists binders into a list of types and the innermost body.
    ///
    /// Handles patterns like: ∃ x : A, ∃ y : B, P(x, y)
    /// Returns: ([A, B], P(x, y)) where BVar indices are adjusted for the combined context.
    ///
    /// Since Exists in Lean is encoded as `Exists T (fun x => body)`, nested existentials
    /// appear as `Exists T1 (fun x => Exists T2 (fun y => P(x, y)))`.
    fn flatten_exists(&self, first_ty: &Expr, body: &Expr) -> (Vec<Expr>, Expr) {
        let mut types = vec![first_ty.clone()];
        let mut current = body.clone();

        // Keep looking for nested Exists patterns in the body
        // Exists is classified by looking at App(App(Const("Exists"), type), Lam(_, _, body))
        loop {
            // Check if current is an Exists application
            if let Expr::App(func, arg) = &current {
                let inner_head = func.get_app_fn();
                if let Expr::Const(name, _) = inner_head {
                    if name.to_string() == "Exists" {
                        let func_args = func.get_app_args();
                        if !func_args.is_empty() {
                            // Get the type from Exists application
                            let ty = func_args[0].clone();
                            // The arg should be a lambda: fun x : T => P(x)
                            if let Expr::Lam(_, _, inner_body) = arg.as_ref() {
                                // Check if the inner body actually uses the bound variable
                                if inner_body.has_loose_bvars() {
                                    types.push(ty);
                                    current = inner_body.as_ref().clone();
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
            // Not a nested Exists, stop flattening
            break;
        }

        (types, current)
    }

    /// Flatten a mixed quantifier prefix into a QuantifierPrefix structure.
    ///
    /// This handles arbitrary alternations of ∀ and ∃ quantifiers:
    /// - `∀x. ∃y. P(x,y)` -> binders: [(∀, A, 1), (∃, B, 0)], body: P(x,y)
    /// - `∃x. ∀y. ∃z. P(x,y,z)` -> binders: [(∃, A, 2), (∀, B, 1), (∃, C, 0)]
    ///
    /// The de Bruijn indices are assigned in standard order: outermost binder
    /// gets the highest index (n-1), innermost gets 0.
    fn flatten_quantifier_prefix(&self, prop: &PropClass) -> QuantifierPrefix {
        let mut binders: Vec<(QuantifierKind, Expr)> = Vec::new();
        let mut current: PropClass = prop.clone();

        loop {
            match current {
                PropClass::Forall(ty, body) => {
                    binders.push((QuantifierKind::Forall, ty));
                    current = self.classify_prop(&body);
                }
                PropClass::Exists(ty, body) => {
                    binders.push((QuantifierKind::Exists, ty));
                    current = self.classify_prop(&body);
                }
                _ => break,
            }
        }

        // Convert the body back to an Expr
        let body = self.propclass_to_expr(&current);

        // Assign de Bruijn indices (outermost = len-1, innermost = 0)
        let n = binders.len();
        let quantifier_binders: Vec<QuantifierBinder> = binders
            .into_iter()
            .enumerate()
            .map(|(i, (kind, ty))| QuantifierBinder {
                kind,
                ty,
                index: u32::try_from(n - 1 - i).expect("quantifier binder index exceeded u32::MAX"),
            })
            .collect();

        QuantifierPrefix {
            binders: quantifier_binders,
            body,
        }
    }

    /// Convert a PropClass back to an Expr (for body extraction)
    fn propclass_to_expr(&self, prop: &PropClass) -> Expr {
        match prop {
            PropClass::Eq(lhs, rhs) => {
                // Eq A a b
                let eq_const = Expr::const_(Name::from_string("Eq"), vec![]);
                let ty = Expr::sort(Level::zero()); // placeholder type
                Expr::app(Expr::app(Expr::app(eq_const, ty), lhs.clone()), rhs.clone())
            }
            PropClass::Neq(lhs, rhs) => {
                let eq = self.propclass_to_expr(&PropClass::Eq(lhs.clone(), rhs.clone()));
                let not_const = Expr::const_(Name::from_string("Not"), vec![]);
                Expr::app(not_const, eq)
            }
            PropClass::And(p, q) => {
                let and_const = Expr::const_(Name::from_string("And"), vec![]);
                Expr::app(
                    Expr::app(and_const, self.propclass_to_expr(&self.classify_prop(p))),
                    self.propclass_to_expr(&self.classify_prop(q)),
                )
            }
            PropClass::Or(p, q) => {
                let or_const = Expr::const_(Name::from_string("Or"), vec![]);
                Expr::app(
                    Expr::app(or_const, self.propclass_to_expr(&self.classify_prop(p))),
                    self.propclass_to_expr(&self.classify_prop(q)),
                )
            }
            PropClass::Implies(p, q) => {
                // P → Q as Pi type
                Expr::pi(
                    BinderInfo::Default,
                    self.propclass_to_expr(&self.classify_prop(p)),
                    self.propclass_to_expr(&self.classify_prop(q)),
                )
            }
            PropClass::Not(p) => {
                let not_const = Expr::const_(Name::from_string("Not"), vec![]);
                Expr::app(not_const, self.propclass_to_expr(&self.classify_prop(p)))
            }
            PropClass::Forall(ty, body) => Expr::pi(BinderInfo::Default, ty.clone(), body.clone()),
            PropClass::Exists(ty, body) => {
                let exists_const = Expr::const_(Name::from_string("Exists"), vec![]);
                let lam = Expr::lam(BinderInfo::Default, ty.clone(), body.clone());
                Expr::app(Expr::app(exists_const, ty.clone()), lam)
            }
            PropClass::True => Expr::const_(Name::from_string("True"), vec![]),
            PropClass::False => Expr::const_(Name::from_string("False"), vec![]),
            PropClass::Lt(lhs, rhs) => {
                let lt_const = Expr::const_(Name::from_string("LT.lt"), vec![]);
                Expr::app(Expr::app(lt_const, lhs.clone()), rhs.clone())
            }
            PropClass::Le(lhs, rhs) => {
                let le_const = Expr::const_(Name::from_string("LE.le"), vec![]);
                Expr::app(Expr::app(le_const, lhs.clone()), rhs.clone())
            }
            PropClass::Gt(lhs, rhs) => {
                let gt_const = Expr::const_(Name::from_string("GT.gt"), vec![]);
                Expr::app(Expr::app(gt_const, lhs.clone()), rhs.clone())
            }
            PropClass::Ge(lhs, rhs) => {
                let ge_const = Expr::const_(Name::from_string("GE.ge"), vec![]);
                Expr::app(Expr::app(ge_const, lhs.clone()), rhs.clone())
            }
            PropClass::Atom(e) => e.clone(),
        }
    }

    /// Analyze a hypothesis for its quantifier structure and add it to the SMT context.
    ///
    /// This function handles mixed quantifier scopes with proper Skolemization:
    /// - For ∀x. ∃y. P(x,y): creates Skolem function sk_y(x) depending on x
    /// - For ∃x. ∀y. P(x,y): creates Skolem constant sk_x, then handles ∀y universally
    ///
    /// Returns the alternation depth for proof strategy selection.
    pub fn add_hypothesis_with_prefix_analysis(&mut self, hyp: &Expr) -> Option<u32> {
        let prop = self.classify_prop(hyp);
        let prefix = self.flatten_quantifier_prefix(&prop);

        if prefix.is_empty() {
            // No quantifiers, use normal hypothesis handling
            self.add_hypothesis(hyp)?;
            return Some(0);
        }

        let alternation_depth = prefix.alternation_depth();

        // For simple cases (purely universal or purely existential), use existing handlers
        if prefix.is_purely_universal() {
            self.add_hypothesis(hyp)?;
            return Some(alternation_depth);
        }

        if prefix.is_purely_existential() {
            self.add_hypothesis(hyp)?;
            return Some(alternation_depth);
        }

        // Mixed quantifier scope: use proper Skolemization
        self.handle_mixed_quantifier_hypothesis(&prefix)?;

        Some(alternation_depth)
    }

    /// Handle a hypothesis with mixed quantifier scope using proper Skolemization.
    ///
    /// For ∀x. ∃y. P(x,y):
    /// - Create a fresh Skolem function symbol sk_y
    /// - The Skolem term for y is sk_y(x), depending on x
    /// - Add the instantiation: P(x, sk_y(x)) as a universal hypothesis
    ///
    /// For ∃x. ∀y. P(x,y):
    /// - Create a fresh Skolem constant sk_x (no dependencies)
    /// - Add P(sk_x, y) as a universal hypothesis over y
    fn handle_mixed_quantifier_hypothesis(&mut self, prefix: &QuantifierPrefix) -> Option<()> {
        let deps = prefix.skolem_dependencies();

        // Create Skolem terms for existential variables
        let mut skolem_terms: HashMap<u32, TermId> = HashMap::new();
        let mut forall_witnesses: HashMap<u32, TermId> = HashMap::new();

        // First pass: create witnesses for universal variables
        for binder in &prefix.binders {
            if binder.kind == QuantifierKind::Forall {
                let witness_name =
                    format!("forall_witness_{}_{}", binder.index, self.fresh_counter);
                self.fresh_counter += 1;
                forall_witnesses.insert(binder.index, self.smt.const_term(witness_name));
            }
        }

        // Second pass: create Skolem terms for existential variables
        for binder in &prefix.binders {
            if binder.kind == QuantifierKind::Exists {
                let dep_indices = deps.get(&binder.index).cloned().unwrap_or_default();

                if dep_indices.is_empty() {
                    // No dependencies: simple Skolem constant
                    let skolem_name = format!("skolem_{}_{}", binder.index, self.fresh_counter);
                    self.fresh_counter += 1;
                    skolem_terms.insert(binder.index, self.smt.const_term(skolem_name));
                } else {
                    // Has dependencies: Skolem function applied to preceding universals
                    let skolem_fn_name =
                        format!("skolem_fn_{}_{}", binder.index, self.fresh_counter);
                    self.fresh_counter += 1;
                    let skolem_fn = self.smt.const_term(skolem_fn_name.clone());

                    // Apply Skolem function to all preceding universal witnesses
                    let mut skolem_term = skolem_fn;
                    for dep_idx in &dep_indices {
                        if let Some(&witness) = forall_witnesses.get(dep_idx) {
                            skolem_term = self.smt.app_term(
                                format!("{skolem_fn_name}_{dep_idx}"),
                                vec![skolem_term, witness],
                            );
                        }
                    }
                    skolem_terms.insert(binder.index, skolem_term);
                }
            }
        }

        // Build the substitution list
        let mut bound_vars = Vec::new();
        let mut witness_terms = Vec::new();

        for binder in &prefix.binders {
            bound_vars.push(binder.index);
            match binder.kind {
                QuantifierKind::Forall => {
                    witness_terms.push(*forall_witnesses.get(&binder.index)?);
                }
                QuantifierKind::Exists => {
                    witness_terms.push(*skolem_terms.get(&binder.index)?);
                }
            }
        }

        // Instantiate the body
        if let Some(inst) =
            self.instantiate_body_with_terms(&prefix.body, &bound_vars, &witness_terms)
        {
            self.add_hypothesis(&inst)?;
        }

        // Also store the universal part for E-matching if there are universals
        let forall_indices = prefix.forall_indices();
        if !forall_indices.is_empty() {
            let triggers = self.extract_ematch_triggers(&prefix.body, &forall_indices);
            if !triggers.is_empty() {
                let tys: Vec<Expr> = prefix
                    .binders
                    .iter()
                    .filter(|b| b.kind == QuantifierKind::Forall)
                    .map(|b| b.ty.clone())
                    .collect();

                let pending = PendingForall {
                    tys,
                    body: prefix.body.clone(),
                    triggers,
                    bound_vars: forall_indices,
                    priority: 0,
                    instantiation_count: 0,
                };
                // Compute initial priority using the scorer
                let scorer = QuantifierPriorityScorer::new();
                let priority = scorer.score(&pending);
                self.pending_foralls.push(PendingForall {
                    priority,
                    ..pending
                });
            }
        }

        Some(())
    }

    /// Instantiate a body using witness terms for each bound variable.
    fn instantiate_body_with_terms(
        &self,
        body: &Expr,
        bound_vars: &[u32],
        witness_terms: &[TermId],
    ) -> Option<Expr> {
        if bound_vars.len() != witness_terms.len() {
            return None;
        }

        let mut replacements = Vec::new();
        for (idx, term) in bound_vars.iter().zip(witness_terms.iter()) {
            let expr = self
                .term_to_expr
                .get(term)
                .cloned()
                .unwrap_or_else(|| Expr::fvar(FVarId(u64::from(term.0) + 1_000_000)));
            replacements.push((*idx, expr));
        }

        Some(self.instantiate_bvars(body, &replacements))
    }

    /// Apply bound-variable substitutions in descending index order to avoid shifting.
    fn instantiate_bvars(&self, body: &Expr, replacements: &[(u32, Expr)]) -> Expr {
        let mut ordered = replacements.to_vec();
        ordered.sort_by(|a, b| b.0.cmp(&a.0));

        let mut result = body.clone();
        for (idx, expr) in ordered {
            result = self.substitute_bvar(&result, idx, &expr);
        }
        result
    }

    /// Substitute BVar(idx) with the given expression
    fn substitute_bvar(&self, expr: &Expr, idx: u32, replacement: &Expr) -> Expr {
        match expr {
            Expr::BVar(i) => {
                if *i == idx {
                    replacement.clone()
                } else if *i > idx {
                    // Shift down since we're removing a binder
                    Expr::BVar(*i - 1)
                } else {
                    expr.clone()
                }
            }
            Expr::App(f, a) => {
                let f_sub = self.substitute_bvar(f, idx, replacement);
                let a_sub = self.substitute_bvar(a, idx, replacement);
                Expr::app(f_sub, a_sub)
            }
            Expr::Lam(info, ty, body) => {
                let ty_sub = self.substitute_bvar(ty, idx, replacement);
                let body_sub = self.substitute_bvar(body, idx + 1, replacement);
                Expr::lam(*info, ty_sub, body_sub)
            }
            Expr::Pi(info, domain, codomain) => {
                let domain_sub = self.substitute_bvar(domain, idx, replacement);
                let codomain_sub = self.substitute_bvar(codomain, idx + 1, replacement);
                Expr::pi(*info, domain_sub, codomain_sub)
            }
            Expr::Let(_, _, _) => {
                // Let expressions are rare in propositions, handle conservatively
                expr.clone()
            }
            _ => expr.clone(),
        }
    }

    /// Translate a proposition and assert its negation (for validity checking)
    #[allow(dead_code)]
    fn translate_negated(&mut self, prop: &Expr) -> Option<()> {
        match self.classify_prop(prop) {
            PropClass::Eq(lhs, rhs) => {
                // Goal: lhs = rhs
                // Assert: lhs ≠ rhs (negation)
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.assert_neq(t1, t2);
                Some(())
            }
            PropClass::Neq(lhs, rhs) => {
                // Goal: lhs ≠ rhs
                // Assert: lhs = rhs (negation)
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.assert_eq(t1, t2);
                Some(())
            }
            // Arithmetic comparisons
            PropClass::Lt(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Le(t2, t1)]);
                Some(())
            }
            PropClass::Le(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Lt(t2, t1)]);
                Some(())
            }
            PropClass::Gt(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Le(t1, t2)]);
                Some(())
            }
            PropClass::Ge(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Lt(t1, t2)]);
                Some(())
            }
            PropClass::And(p, q) => {
                // Goal: P ∧ Q
                // Assert: ¬(P ∧ Q) ≡ ¬P ∨ ¬Q
                // We need to handle this with clauses
                let np = self.prop_to_literal(&p, false)?;
                let nq = self.prop_to_literal(&q, false)?;
                self.smt.add_clause(vec![np, nq]);
                Some(())
            }
            PropClass::Or(p, q) => {
                // Goal: P ∨ Q
                // Assert: ¬(P ∨ Q) ≡ ¬P ∧ ¬Q
                let np = self.prop_to_literal(&p, false)?;
                let nq = self.prop_to_literal(&q, false)?;
                // Assert both negations as unit clauses
                self.smt.add_clause(vec![np.clone()]);
                self.smt.add_clause(vec![nq.clone()]);
                Some(())
            }
            PropClass::Implies(p, q) => {
                // Goal: P → Q
                // Assert: ¬(P → Q) ≡ P ∧ ¬Q
                let pp = self.prop_to_literal(&p, true)?;
                let nq = self.prop_to_literal(&q, false)?;
                self.smt.add_clause(vec![pp.clone()]);
                self.smt.add_clause(vec![nq.clone()]);
                Some(())
            }
            PropClass::Not(p) => {
                // Goal: ¬P
                // Assert: P (negation of ¬P)
                let pp = self.prop_to_literal(&p, true)?;
                self.smt.add_clause(vec![pp]);
                Some(())
            }
            PropClass::Forall(ty, body) => {
                // Goal: ∀ x : T, P(x)
                // Assert negation: ∃ x : T, ¬P(x)
                // Use Skolemization
                self.translate_negated_classified(&PropClass::Forall(ty, body))
            }
            PropClass::Exists(ty, body) => {
                // Goal: ∃ x : T, P(x)
                // Assert negation: ∀ x : T, ¬P(x)
                self.translate_negated_classified(&PropClass::Exists(ty, body))
            }
            PropClass::True => {
                // Goal: True - trivially valid, assert False
                // Adding empty clause makes it unsat
                Some(())
            }
            PropClass::False => {
                // Goal: False - never valid
                // Don't assert anything, will be SAT
                Some(())
            }
            PropClass::Atom(expr) => {
                // Unknown proposition - create a boolean variable
                let lit = self.prop_to_literal(&expr, false)?;
                self.smt.add_clause(vec![lit]);
                Some(())
            }
        }
    }

    /// Translate a proposition to a theory literal (with polarity)
    fn prop_to_literal(&mut self, prop: &Expr, positive: bool) -> Option<TheoryLiteral> {
        match self.classify_prop(prop) {
            PropClass::Eq(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                if positive {
                    Some(TheoryLiteral::Eq(t1, t2))
                } else {
                    Some(TheoryLiteral::Neq(t1, t2))
                }
            }
            PropClass::Neq(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                if positive {
                    Some(TheoryLiteral::Neq(t1, t2))
                } else {
                    Some(TheoryLiteral::Eq(t1, t2))
                }
            }
            // Arithmetic comparisons
            PropClass::Lt(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                if positive {
                    Some(TheoryLiteral::Lt(t1, t2))
                } else {
                    // ¬(a < b) ≡ b ≤ a
                    Some(TheoryLiteral::Le(t2, t1))
                }
            }
            PropClass::Le(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                if positive {
                    Some(TheoryLiteral::Le(t1, t2))
                } else {
                    // ¬(a ≤ b) ≡ b < a
                    Some(TheoryLiteral::Lt(t2, t1))
                }
            }
            PropClass::Gt(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                // a > b ≡ b < a
                if positive {
                    Some(TheoryLiteral::Lt(t2, t1))
                } else {
                    // ¬(a > b) ≡ a ≤ b
                    Some(TheoryLiteral::Le(t1, t2))
                }
            }
            PropClass::Ge(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                // a ≥ b ≡ b ≤ a
                if positive {
                    Some(TheoryLiteral::Le(t2, t1))
                } else {
                    // ¬(a ≥ b) ≡ a < b
                    Some(TheoryLiteral::Lt(t1, t2))
                }
            }
            PropClass::Atom(_) => {
                // Create a fresh boolean variable
                let var_id = self.fresh_counter;
                self.fresh_counter += 1;
                Some(TheoryLiteral::Bool(var_id))
            }
            // For compound propositions, we'd need to flatten to CNF
            // For now, treat them as atoms
            _ => {
                let var_id = self.fresh_counter;
                self.fresh_counter += 1;
                Some(TheoryLiteral::Bool(var_id))
            }
        }
    }

    /// Classify a proposition into known forms
    fn classify_prop(&self, expr: &Expr) -> PropClass {
        // Get the head and arguments
        let head = expr.get_app_fn();
        let args = expr.get_app_args();

        match head {
            Expr::Const(name, _) => {
                let name_str = name.to_string();
                match name_str.as_str() {
                    "Eq" if args.len() == 3 => {
                        // Eq α a b
                        PropClass::Eq(args[1].clone(), args[2].clone())
                    }
                    "Ne" if args.len() == 3 => {
                        // Ne α a b
                        PropClass::Neq(args[1].clone(), args[2].clone())
                    }
                    // Arithmetic comparisons (from Lean5 typeclass instances)
                    "LT.lt" | "Int.lt" | "Nat.lt" if args.len() >= 2 => {
                        let len = args.len();
                        PropClass::Lt(args[len - 2].clone(), args[len - 1].clone())
                    }
                    "LE.le" | "Int.le" | "Nat.le" if args.len() >= 2 => {
                        let len = args.len();
                        PropClass::Le(args[len - 2].clone(), args[len - 1].clone())
                    }
                    "GT.gt" if args.len() >= 2 => {
                        let len = args.len();
                        PropClass::Gt(args[len - 2].clone(), args[len - 1].clone())
                    }
                    "GE.ge" if args.len() >= 2 => {
                        let len = args.len();
                        PropClass::Ge(args[len - 2].clone(), args[len - 1].clone())
                    }
                    "And" if args.len() == 2 => PropClass::And(args[0].clone(), args[1].clone()),
                    "Or" if args.len() == 2 => PropClass::Or(args[0].clone(), args[1].clone()),
                    "Not" if args.len() == 1 => PropClass::Not(args[0].clone()),
                    "True" => PropClass::True,
                    "False" => PropClass::False,
                    // Exists T (fun x => P x) - args[0] is T, args[1] is the lambda
                    "Exists" if args.len() == 2 => {
                        let ty = args[0].clone();
                        if let Expr::Lam(_, _, body) = &args[1] {
                            PropClass::Exists(ty, body.as_ref().clone())
                        } else {
                            PropClass::Atom(expr.clone())
                        }
                    }
                    _ => PropClass::Atom(expr.clone()),
                }
            }
            Expr::Pi(_, domain, codomain) => {
                // Pi type can be:
                // 1. Non-dependent Pi (implication): P → Q (when codomain doesn't use BVar(0))
                // 2. Dependent Pi (universal quantifier): ∀ x : T, P(x)
                if codomain.has_loose_bvars() {
                    // Dependent: universal quantifier ∀ x : T, P(x)
                    PropClass::Forall(domain.as_ref().clone(), codomain.as_ref().clone())
                } else {
                    // Non-dependent: implication P → Q
                    PropClass::Implies(domain.as_ref().clone(), codomain.as_ref().clone())
                }
            }
            _ => {
                // Check for Exists at the App level
                if let Expr::App(func, arg) = expr {
                    let inner_head = func.get_app_fn();
                    if let Expr::Const(name, _) = inner_head {
                        if name.to_string() == "Exists" {
                            // Exists T (fun x => P x) where func is (Exists T) and arg is the lambda
                            let func_args = func.get_app_args();
                            if !func_args.is_empty() {
                                // Get the type from Exists application
                                let ty = func_args[0].clone();
                                // The arg should be a lambda: fun x : T => P(x)
                                if let Expr::Lam(_, _, body) = arg.as_ref() {
                                    return PropClass::Exists(ty, body.as_ref().clone());
                                }
                            }
                        }
                    }
                }
                PropClass::Atom(expr.clone())
            }
        }
    }

    /// Translate a kernel term to an SMT term
    fn translate_term(&mut self, expr: &Expr) -> Option<TermId> {
        // Check if we've already translated this expression
        if let Some(key) = self.expr_to_key(expr) {
            if let Some(&term_id) = self.expr_to_term.get(&key) {
                return Some(term_id);
            }
        }

        let term_id = match expr {
            Expr::FVar(fvar_id) => {
                if let Some(&tid) = self.fvar_to_term.get(fvar_id) {
                    tid
                } else {
                    let name = format!("fvar_{}", fvar_id.0);
                    let tid = self.smt.const_term(name);
                    self.fvar_to_term.insert(*fvar_id, tid);
                    tid
                }
            }
            Expr::Const(name, _) => self.smt.const_term(name.to_string()),
            Expr::App(_func, _arg) => {
                // Flatten application spine
                let head = expr.get_app_fn();
                let args = expr.get_app_args();

                match head {
                    Expr::Const(name, _) => {
                        let name_str = name.to_string();

                        // Handle array operations specially
                        match name_str.as_str() {
                            // Array select (read): Array.get α arr idx or getElem arr idx bound
                            "Array.get" | "getElem" | "GetElem.getElem" if args.len() >= 2 => {
                                // Last two args are typically array and index
                                let len = args.len();
                                let arr = self.translate_term(args[len - 2])?;
                                let idx = self.translate_term(args[len - 1])?;
                                self.smt.select_term(arr, idx)
                            }
                            // Array store (write): Array.set α arr idx val
                            "Array.set" | "setElem" | "SetElem.setElem" if args.len() >= 3 => {
                                // Last three args are typically array, index, value
                                let len = args.len();
                                let arr = self.translate_term(args[len - 3])?;
                                let idx = self.translate_term(args[len - 2])?;
                                let val = self.translate_term(args[len - 1])?;
                                self.smt.store_term(arr, idx, val)
                            }
                            // C-style array access: select and store
                            "select" if args.len() == 2 => {
                                let arr = self.translate_term(args[0])?;
                                let idx = self.translate_term(args[1])?;
                                self.smt.select_term(arr, idx)
                            }
                            "store" if args.len() == 3 => {
                                let arr = self.translate_term(args[0])?;
                                let idx = self.translate_term(args[1])?;
                                let val = self.translate_term(args[2])?;
                                self.smt.store_term(arr, idx, val)
                            }
                            // Default: translate as function application
                            _ => {
                                let arg_terms: Option<Vec<TermId>> =
                                    args.iter().map(|a| self.translate_term(a)).collect();
                                let arg_terms = arg_terms?;
                                self.smt.app_term(name_str, arg_terms)
                            }
                        }
                    }
                    Expr::FVar(fvar_id) => {
                        let func_name = format!("fvar_{}", fvar_id.0);
                        let arg_terms: Option<Vec<TermId>> =
                            args.iter().map(|a| self.translate_term(a)).collect();
                        let arg_terms = arg_terms?;
                        self.smt.app_term(func_name, arg_terms)
                    }
                    _ => {
                        // Complex head, create fresh symbol
                        let name = format!("app_{}", self.fresh_counter);
                        self.fresh_counter += 1;
                        self.smt.const_term(name)
                    }
                }
            }
            Expr::Lit(lit) => match lit {
                lean5_kernel::expr::Literal::Nat(n) => self.smt.int_term(*n as i64),
                lean5_kernel::expr::Literal::String(s) => self.smt.const_term(format!("str_{s}")),
            },
            // For other expression forms, create a fresh constant
            _ => {
                let name = format!("term_{}", self.fresh_counter);
                self.fresh_counter += 1;
                self.smt.const_term(name)
            }
        };

        // Cache the mapping
        if let Some(key) = self.expr_to_key(expr) {
            self.expr_to_term.insert(key, term_id);
        }
        self.term_to_expr.insert(term_id, expr.clone());

        Some(term_id)
    }

    /// Convert an expression to a hashable key
    fn expr_to_key(&self, expr: &Expr) -> Option<ExprKey> {
        match expr {
            Expr::BVar(idx) => Some(ExprKey::BVar(*idx)),
            Expr::FVar(fvar_id) => Some(ExprKey::FVar(*fvar_id)),
            Expr::Const(name, _) => Some(ExprKey::Const(name.clone())),
            Expr::App(f, a) => {
                let f_key = self.expr_to_key(f)?;
                let a_key = self.expr_to_key(a)?;
                Some(ExprKey::App(Box::new(f_key), Box::new(a_key)))
            }
            Expr::Lam(_, ty, body) => {
                let ty_key = self.expr_to_key(ty)?;
                let body_key = self.expr_to_key(body)?;
                Some(ExprKey::Lam(Box::new(ty_key), Box::new(body_key)))
            }
            Expr::Pi(_, ty, body) => {
                let ty_key = self.expr_to_key(ty)?;
                let body_key = self.expr_to_key(body)?;
                Some(ExprKey::Pi(Box::new(ty_key), Box::new(body_key)))
            }
            Expr::Lit(lit) => match lit {
                lean5_kernel::expr::Literal::Nat(n) => Some(ExprKey::Lit(LitKey::Nat(*n))),
                lean5_kernel::expr::Literal::String(s) => {
                    Some(ExprKey::Lit(LitKey::String(s.to_string())))
                }
            },
            _ => None, // Don't cache Sort, Let, Proj
        }
    }

    /// Assert additional hypotheses from the context
    pub fn add_hypothesis(&mut self, hyp: &Expr) -> Option<()> {
        self.add_hypothesis_with_fvar(hyp, None)
    }

    /// Assert a hypothesis with optional FVarId for proof reconstruction
    pub fn add_hypothesis_with_fvar(&mut self, hyp: &Expr, fvar: Option<FVarId>) -> Option<()> {
        match self.classify_prop(hyp) {
            PropClass::Eq(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.assert_eq(t1, t2);
                // Track the hypothesis for proof reconstruction
                // Only store the canonical (original) direction
                if let Some(fvar_id) = fvar {
                    self.eq_hypothesis_canonical.insert((t1, t2), fvar_id);
                    if let Some(eq) = self.equality_theory_mut() {
                        eq.register_hypothesis(t1, t2, fvar_id);
                    }
                }
                Some(())
            }
            PropClass::Neq(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.assert_neq(t1, t2);
                Some(())
            }
            // Arithmetic comparisons as hypotheses
            PropClass::Lt(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Lt(t1, t2)]);
                Some(())
            }
            PropClass::Le(lhs, rhs) => {
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Le(t1, t2)]);
                Some(())
            }
            PropClass::Gt(lhs, rhs) => {
                // a > b ≡ b < a
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Lt(t2, t1)]);
                Some(())
            }
            PropClass::Ge(lhs, rhs) => {
                // a ≥ b ≡ b ≤ a
                let t1 = self.translate_term(&lhs)?;
                let t2 = self.translate_term(&rhs)?;
                self.smt.add_clause(vec![TheoryLiteral::Le(t2, t1)]);
                Some(())
            }
            PropClass::And(p, q) => {
                // P ∧ Q means both P and Q hold
                self.add_hypothesis(&p)?;
                self.add_hypothesis(&q)?;
                Some(())
            }
            PropClass::Implies(p, q) => {
                // P → Q as a clause: ¬P ∨ Q
                let np = self.prop_to_literal(&p, false)?;
                let pq = self.prop_to_literal(&q, true)?;
                self.smt.add_clause(vec![np, pq]);
                Some(())
            }
            PropClass::Not(p) => {
                // ¬P means P is false
                let np = self.prop_to_literal(&p, false)?;
                self.smt.add_clause(vec![np]);
                Some(())
            }
            PropClass::True => Some(()), // No information
            PropClass::False => {
                // False hypothesis - anything follows
                // Add empty clause to make UNSAT
                Some(())
            }
            PropClass::Or(p, q) => {
                // P ∨ Q as a clause
                let pp = self.prop_to_literal(&p, true)?;
                let pq = self.prop_to_literal(&q, true)?;
                self.smt.add_clause(vec![pp, pq]);
                Some(())
            }
            PropClass::Forall(ref ty, ref body) => {
                // ∀ x : T, P(x) as hypothesis
                // Strategy: Extract E-matching triggers and store for later instantiation
                // The actual instantiation happens in prove() after the E-graph is populated

                let (bound_types, flat_body) = self.flatten_forall(ty, body);
                let bound_count = u32::try_from(bound_types.len())
                    .expect("forall bound variable count exceeded u32::MAX");
                let bound_vars: Vec<u32> = (0..bound_count).collect();
                let triggers = self.extract_ematch_triggers(&flat_body, &bound_vars);

                if !triggers.is_empty() {
                    // Store for E-matching instantiation with priority scoring
                    let pending = PendingForall {
                        tys: bound_types.clone(),
                        body: flat_body.clone(),
                        triggers,
                        bound_vars: bound_vars.clone(),
                        priority: 0,
                        instantiation_count: 0,
                    };
                    let scorer = QuantifierPriorityScorer::new();
                    let priority = scorer.score(&pending);
                    self.pending_foralls.push(PendingForall {
                        priority,
                        ..pending
                    });
                }

                // Also instantiate with a fresh witness as fallback
                // This ensures we don't lose any proof power
                let mut witness_terms = Vec::new();
                for i in 0..bound_types.len() {
                    let skolem_name = format!("forall_witness_{}_{}", i, self.fresh_counter);
                    self.fresh_counter += 1;
                    witness_terms.push(self.smt.const_term(skolem_name));
                }

                if let Some(inst) =
                    self.instantiate_body_with_terms(&flat_body, &bound_vars, &witness_terms)
                {
                    self.add_hypothesis(&inst)?;
                }
                Some(())
            }
            PropClass::Exists(ref ty, ref body) => {
                // ∃ x : T, P(x) as hypothesis means there exists a witness
                // Flatten nested exists: ∃ x : A, ∃ y : B, P(x, y) → create witnesses for both
                let (bound_types, flat_body) = self.flatten_exists(ty, body);
                let bound_count = u32::try_from(bound_types.len())
                    .expect("exists bound variable count exceeded u32::MAX");
                let bound_vars: Vec<u32> = (0..bound_count).collect();

                // Create Skolem witnesses for all bound variables
                let mut witness_terms = Vec::new();
                for i in 0..bound_types.len() {
                    let skolem_name = format!("exists_witness_{}_{}", i, self.fresh_counter);
                    self.fresh_counter += 1;
                    witness_terms.push(self.smt.const_term(skolem_name));
                }

                // Instantiate the body with all Skolem witnesses
                if let Some(inst) =
                    self.instantiate_body_with_terms(&flat_body, &bound_vars, &witness_terms)
                {
                    self.add_hypothesis(&inst)
                } else {
                    Some(()) // Conservative: can't use this hypothesis
                }
            }
            PropClass::Atom(_) => {
                // Unknown atom - create boolean and assert it
                let lit = self.prop_to_literal(hyp, true)?;
                self.smt.add_clause(vec![lit]);
                Some(())
            }
        }
    }

    /// Get SMT solver statistics
    pub fn stats(&self) -> crate::smt::SmtStats {
        self.smt.stats()
    }

    // ========================================================================
    // Trigger Pattern Extraction for E-Matching
    // ========================================================================

    /// Extract trigger patterns from a quantified formula body
    ///
    /// Triggers are sub-terms that can be used for E-matching-based quantifier
    /// instantiation. Good triggers should:
    /// 1. Contain all bound variables
    /// 2. Be as small as possible (to avoid spurious matches)
    /// 3. Not be pure (not just variables or constants)
    ///
    /// For a formula `∀ x. P(f(x), g(x))`, good triggers would include
    /// `f(x)` and `g(x)` since they contain `x` and are function applications.
    pub fn extract_triggers(&self, body: &Expr, bound_vars: &[u32]) -> Vec<TriggerPattern> {
        let mut triggers = Vec::new();
        let mut extractor = TriggerExtractor::new(bound_vars);
        extractor.extract(body, &mut triggers);

        // Score and deduplicate triggers
        triggers.sort_by(|a, b| b.score.cmp(&a.score));
        triggers.dedup_by(|a, b| a.pattern == b.pattern);

        triggers
    }

    /// Extract triggers and convert to E-graph patterns
    pub fn extract_ematch_triggers(
        &self,
        body: &Expr,
        bound_vars: &[u32],
    ) -> Vec<crate::egraph::Trigger> {
        let patterns = self.extract_triggers(body, bound_vars);
        if patterns.is_empty() {
            return Vec::new();
        }

        let required_vars: Vec<String> = bound_vars.iter().map(|i| format!("?x{i}")).collect();

        // Collect all valid triggers with their scores
        let mut scored_triggers: Vec<(crate::egraph::Trigger, i32)> = Vec::new();

        // Prefer single-pattern triggers that already cover all bound variables
        for pat in &patterns {
            if let Some(trigger) = self.trigger_from_patterns(&[pat]) {
                if self.trigger_has_all_vars(&trigger, &required_vars) {
                    let score = self.score_trigger_combination(&[pat]);
                    scored_triggers.push((trigger, score));
                }
            }
        }

        // If we didn't find a single trigger covering everything, try combinations
        if scored_triggers.is_empty() && bound_vars.len() > 1 {
            // Try pairs first
            for i in 0..patterns.len() {
                for j in (i + 1)..patterns.len() {
                    let combo = [&patterns[i], &patterns[j]];
                    if let Some(trigger) = self.trigger_from_patterns(&combo) {
                        if self.trigger_has_all_vars(&trigger, &required_vars) {
                            let score = self.score_trigger_combination(&combo);
                            scored_triggers.push((trigger, score));
                        }
                    }
                }
            }

            // As a fallback, try triples for harder cases
            if scored_triggers.is_empty() {
                for i in 0..patterns.len() {
                    for j in (i + 1)..patterns.len() {
                        for k in (j + 1)..patterns.len() {
                            let combo = [&patterns[i], &patterns[j], &patterns[k]];
                            if let Some(trigger) = self.trigger_from_patterns(&combo) {
                                if self.trigger_has_all_vars(&trigger, &required_vars) {
                                    let score = self.score_trigger_combination(&combo);
                                    scored_triggers.push((trigger, score));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by score (higher is better) and extract triggers
        scored_triggers.sort_by(|a, b| b.1.cmp(&a.1));
        let mut triggers: Vec<_> = scored_triggers.into_iter().map(|(t, _)| t).collect();

        // Fallback: keep legacy behavior if we couldn't cover all bound vars
        if triggers.is_empty() {
            for pat in &patterns {
                if let Some(pattern) = self.trigger_pattern_to_ematch_pattern(pat) {
                    triggers.push(crate::egraph::Trigger::single(pattern));
                }
            }
        }

        triggers
    }

    /// Score a combination of trigger patterns for E-matching quality.
    ///
    /// Better combinations:
    /// - Have smaller total size (fewer E-graph traversals)
    /// - Use patterns with higher individual scores
    /// - Have fewer patterns (simpler matching)
    /// - Minimize variable overlap (each pattern contributes unique variables)
    fn score_trigger_combination(&self, patterns: &[&TriggerPattern]) -> i32 {
        let mut score = 0;

        // Sum of individual pattern scores
        for pat in patterns {
            score += pat.score;
        }

        // Penalty for multiple patterns (prefer single-pattern triggers)
        // Each additional pattern adds matching overhead
        score -= (patterns.len() as i32 - 1) * 5;

        // Bonus for minimal pattern count that covers all vars
        if patterns.len() == 1 {
            score += 20;
        } else if patterns.len() == 2 {
            score += 10;
        }

        // Penalty for variable overlap (inefficient matching)
        // If two patterns share bound variables, that's wasteful
        let mut seen_vars: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut overlap_count = 0;
        for pat in patterns {
            for &bv in &pat.bound_vars {
                if !seen_vars.insert(bv) {
                    overlap_count += 1;
                }
            }
        }
        score -= overlap_count * 3;

        score
    }

    /// Build an E-matching trigger from one or more trigger patterns
    fn trigger_from_patterns(
        &self,
        patterns: &[&TriggerPattern],
    ) -> Option<crate::egraph::Trigger> {
        let mut ematch_patterns = Vec::new();
        for pat in patterns {
            ematch_patterns.push(self.trigger_pattern_to_ematch_pattern(pat)?);
        }

        if ematch_patterns.len() == 1 {
            Some(crate::egraph::Trigger::single(ematch_patterns.remove(0)))
        } else {
            Some(crate::egraph::Trigger::multi(ematch_patterns))
        }
    }

    /// Convert a single TriggerPattern to an E-matching Trigger
    #[allow(dead_code)] // Used in tests
    fn trigger_to_ematch_pattern(
        &self,
        trigger: &TriggerPattern,
    ) -> Option<crate::egraph::Trigger> {
        self.trigger_from_patterns(&[trigger])
    }

    /// Convert a TriggerPattern to an E-matching Pattern
    fn trigger_pattern_to_ematch_pattern(
        &self,
        trigger: &TriggerPattern,
    ) -> Option<crate::egraph::Pattern> {
        self.expr_to_pattern(&trigger.pattern, &trigger.bound_vars)
    }

    /// Check whether a trigger covers all required bound variables
    fn trigger_has_all_vars(
        &self,
        trigger: &crate::egraph::Trigger,
        required_vars: &[String],
    ) -> bool {
        let vars = trigger.variables();
        required_vars.iter().all(|req| vars.contains(req))
    }

    /// Convert an expression to an E-matching pattern
    ///
    /// The `_bound_vars` parameter is passed through recursion for potential
    /// future use in distinguishing between different bound variables.
    #[allow(clippy::only_used_in_recursion)]
    fn expr_to_pattern(&self, expr: &Expr, _bound_vars: &[u32]) -> Option<crate::egraph::Pattern> {
        match expr {
            Expr::BVar(idx) => {
                // Bound variable becomes a pattern variable
                let var_name = format!("?x{idx}");
                Some(crate::egraph::Pattern::var(var_name))
            }
            Expr::FVar(fv) => {
                // Free variable becomes a constant pattern
                let name = format!("fvar_{}", fv.0);
                Some(crate::egraph::Pattern::constant(name))
            }
            Expr::Const(name, _) => {
                // Constant becomes a constant pattern
                Some(crate::egraph::Pattern::constant(name.to_string()))
            }
            Expr::App(_func, _arg) => {
                // Application: recursively convert
                // Collect all arguments and the head symbol
                let (head, args) = self.collect_app_args(expr);
                let head_name = match head {
                    Expr::Const(name, _) => name.to_string(),
                    Expr::FVar(fv) => format!("fvar_{}", fv.0),
                    _ => return None, // Only handle function applications with known heads
                };

                let mut arg_patterns = Vec::new();
                for arg in &args {
                    arg_patterns.push(self.expr_to_pattern(arg, _bound_vars)?);
                }

                Some(crate::egraph::Pattern::app(head_name, arg_patterns))
            }
            Expr::Lit(lit) => {
                // Literal becomes a constant
                let name = match lit {
                    lean5_kernel::Literal::Nat(n) => format!("nat_{n}"),
                    lean5_kernel::Literal::String(s) => format!("str_{s}"),
                };
                Some(crate::egraph::Pattern::constant(name))
            }
            _ => None, // Lambda, Pi, Let, Sort not convertible to patterns
        }
    }

    /// Collect application arguments (unfold nested App)
    fn collect_app_args(&self, expr: &Expr) -> (Expr, Vec<Expr>) {
        let mut args = Vec::new();
        let mut current = expr.clone();

        while let Expr::App(func, arg) = current {
            args.push((*arg).clone());
            current = (*func).clone();
        }

        args.reverse();
        (current, args)
    }
}

/// A candidate trigger pattern extracted from a quantified formula
#[derive(Debug, Clone)]
pub struct TriggerPattern {
    /// The expression pattern
    pub pattern: Expr,
    /// Bound variables that appear in this pattern
    pub bound_vars: Vec<u32>,
    /// Quality score (higher is better)
    pub score: i32,
}

impl TriggerPattern {
    /// Create a new trigger pattern
    fn new(pattern: Expr, bound_vars: Vec<u32>) -> Self {
        let score = Self::compute_score(&pattern, &bound_vars);
        TriggerPattern {
            pattern,
            bound_vars,
            score,
        }
    }

    /// Compute quality score for a trigger pattern
    ///
    /// Better triggers:
    /// - Contain all bound variables (required)
    /// - Are function applications (not just variables)
    /// - Are smaller (fewer nodes)
    /// - Don't contain complex sub-patterns
    fn compute_score(pattern: &Expr, bound_vars: &[u32]) -> i32 {
        let mut score = 0;

        // Penalize patterns that don't contain all bound variables
        // (This shouldn't happen if extraction is correct, but safety check)
        let vars_in_pattern = Self::collect_bvars(pattern);
        for bv in bound_vars {
            if !vars_in_pattern.contains(bv) {
                score -= 100;
            }
        }

        // Prefer function applications over variables
        if matches!(pattern, Expr::App(_, _)) {
            score += 10;
        }

        // Prefer smaller patterns (fewer nodes)
        let size = Self::pattern_size(pattern);
        score -= size as i32;

        // Bonus for containing constants (more selective)
        if Self::has_constant(pattern) {
            score += 5;
        }

        score
    }

    /// Collect all bound variables in an expression
    fn collect_bvars(expr: &Expr) -> Vec<u32> {
        let mut vars = Vec::new();
        Self::collect_bvars_rec(expr, &mut vars);
        vars
    }

    fn collect_bvars_rec(expr: &Expr, vars: &mut Vec<u32>) {
        match expr {
            Expr::BVar(idx) => {
                if !vars.contains(idx) {
                    vars.push(*idx);
                }
            }
            Expr::App(f, a) => {
                Self::collect_bvars_rec(f, vars);
                Self::collect_bvars_rec(a, vars);
            }
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                Self::collect_bvars_rec(ty, vars);
                Self::collect_bvars_rec(body, vars);
            }
            Expr::Let(ty, val, body) => {
                Self::collect_bvars_rec(ty, vars);
                Self::collect_bvars_rec(val, vars);
                Self::collect_bvars_rec(body, vars);
            }
            _ => {}
        }
    }

    /// Count nodes in a pattern
    fn pattern_size(expr: &Expr) -> usize {
        match expr {
            Expr::App(f, a) => 1 + Self::pattern_size(f) + Self::pattern_size(a),
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                1 + Self::pattern_size(ty) + Self::pattern_size(body)
            }
            Expr::Let(ty, val, body) => {
                1 + Self::pattern_size(ty) + Self::pattern_size(val) + Self::pattern_size(body)
            }
            _ => 1,
        }
    }

    /// Check if pattern contains a constant (more selective)
    fn has_constant(expr: &Expr) -> bool {
        match expr {
            Expr::Const(_, _) => true,
            Expr::App(f, a) => Self::has_constant(f) || Self::has_constant(a),
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                Self::has_constant(ty) || Self::has_constant(body)
            }
            Expr::Let(ty, val, body) => {
                Self::has_constant(ty) || Self::has_constant(val) || Self::has_constant(body)
            }
            _ => false,
        }
    }
}

/// Extractor for trigger patterns from quantified formula bodies
struct TriggerExtractor<'a> {
    /// The bound variables we need to match
    bound_vars: &'a [u32],
}

impl<'a> TriggerExtractor<'a> {
    fn new(bound_vars: &'a [u32]) -> Self {
        TriggerExtractor { bound_vars }
    }

    /// Extract trigger patterns from an expression
    fn extract(&mut self, expr: &Expr, triggers: &mut Vec<TriggerPattern>) {
        // First, check if this expression itself is a good trigger
        if self.is_valid_trigger(expr) {
            let vars = TriggerPattern::collect_bvars(expr);
            // Only add if it contains at least one bound variable
            if vars.iter().any(|v| self.bound_vars.contains(v)) {
                triggers.push(TriggerPattern::new(expr.clone(), vars));
            }
        }

        // Then recursively explore sub-expressions
        match expr {
            Expr::App(f, a) => {
                self.extract(f, triggers);
                self.extract(a, triggers);
            }
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                self.extract(ty, triggers);
                self.extract(body, triggers);
            }
            Expr::Let(ty, val, body) => {
                self.extract(ty, triggers);
                self.extract(val, triggers);
                self.extract(body, triggers);
            }
            _ => {}
        }
    }

    /// Check if an expression is a valid trigger
    ///
    /// Valid triggers are:
    /// - Function applications (most common)
    /// - Not pure theory symbols (those handled by theory solvers)
    /// - Not lambdas/pis (structural, not instantiatable)
    fn is_valid_trigger(&self, expr: &Expr) -> bool {
        match expr {
            Expr::App(_f, _) => {
                // Check if head is a function (not a bound var application)
                let head = self.get_head(expr);
                matches!(head, Expr::Const(_, _) | Expr::FVar(_)) && !self.is_theory_symbol(&head)
            }
            _ => false,
        }
    }

    /// Get the head symbol of an application chain
    fn get_head(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::App(f, _) => self.get_head(f),
            _ => expr.clone(),
        }
    }

    /// Check if a symbol is a theory symbol (handled by theory solvers)
    fn is_theory_symbol(&self, head: &Expr) -> bool {
        if let Expr::Const(name, _) = head {
            let name_str = name.to_string();
            // Common theory symbols
            matches!(
                name_str.as_str(),
                "Eq" | "HEq"
                    | "Ne"
                    | "Add.add"
                    | "Sub.sub"
                    | "Mul.mul"
                    | "Div.div"
                    | "Nat.add"
                    | "Nat.sub"
                    | "Nat.mul"
                    | "Int.add"
                    | "Int.sub"
                    | "Int.mul"
                    | "LT.lt"
                    | "LE.le"
                    | "GT.gt"
                    | "GE.ge"
                    | "And"
                    | "Or"
                    | "Not"
                    | "True"
                    | "False"
            )
        } else {
            false
        }
    }
}

/// Classification of propositions
#[derive(Debug, Clone)]
enum PropClass {
    /// Equality: a = b
    Eq(Expr, Expr),
    /// Disequality: a ≠ b
    Neq(Expr, Expr),
    /// Less than: a < b
    Lt(Expr, Expr),
    /// Less than or equal: a ≤ b
    Le(Expr, Expr),
    /// Greater than: a > b
    Gt(Expr, Expr),
    /// Greater than or equal: a ≥ b
    Ge(Expr, Expr),
    /// Conjunction: P ∧ Q
    And(Expr, Expr),
    /// Disjunction: P ∨ Q
    Or(Expr, Expr),
    /// Implication: P → Q
    Implies(Expr, Expr),
    /// Negation: ¬P
    Not(Expr),
    /// Universal quantifier: ∀ x : T, P(x)
    Forall(Expr, Expr), // (type, body with BVar(0) for bound variable)
    /// Existential quantifier: ∃ x : T, P(x)
    Exists(Expr, Expr), // (type, body with BVar(0) for bound variable)
    /// True
    True,
    /// False
    False,
    /// Unknown atomic proposition
    Atom(Expr),
}

/// Result of SMT-based proving
#[derive(Debug)]
pub struct ProofResult {
    /// Method used to find the proof
    pub method: ProofMethod,
    /// Human-readable proof sketch
    pub proof_sketch: String,
    /// The kernel proof term (if reconstruction succeeded)
    pub proof_term: Option<Expr>,
    /// The proof step trace (for debugging)
    pub proof_step: Option<ProofStep>,
}

impl ProofResult {
    /// Create a new proof result
    pub fn new(method: ProofMethod, sketch: impl Into<String>) -> Self {
        ProofResult {
            method,
            proof_sketch: sketch.into(),
            proof_term: None,
            proof_step: None,
        }
    }

    /// Create a proof result with a kernel proof term
    pub fn with_proof_term(
        method: ProofMethod,
        sketch: impl Into<String>,
        proof: Expr,
        step: ProofStep,
    ) -> Self {
        ProofResult {
            method,
            proof_sketch: sketch.into(),
            proof_term: Some(proof),
            proof_step: Some(step),
        }
    }

    /// Check if this proof result has a valid kernel proof term
    pub fn has_proof_term(&self) -> bool {
        self.proof_term.is_some()
    }
}

/// Method used for the proof
#[derive(Debug, Clone, Copy)]
pub enum ProofMethod {
    /// Proved by SMT showing negation is unsatisfiable
    SmtUnsat,
    /// Proved by finding a witness
    SmtSat,
}

#[cfg(test)]
mod tests;
