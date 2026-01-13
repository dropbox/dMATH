//! Superposition calculus (ported from E prover)
//!
//! This module implements the superposition calculus, a complete and refutationally
//! complete calculus for first-order logic with equality. It is based on the work
//! of Bachmair & Ganzinger and implemented in provers like E, Vampire, and SPASS.
//!
//! # Overview
//!
//! Superposition is a saturation-based theorem prover that:
//! 1. Takes a set of clauses (CNF)
//! 2. Applies inference rules until contradiction or saturation
//! 3. Uses term orderings to restrict inferences (completeness-preserving)
//!
//! # Inference Rules
//!
//! ## Generating inferences:
//! - **Superposition Left**: Rewrites into negative literals
//! - **Superposition Right**: Rewrites into positive literals
//! - **Equality Resolution**: Resolves reflexive equalities
//! - **Equality Factoring**: Factors equal positive literals
//!
//! ## Simplification rules:
//! - **Demodulation**: Simplifies terms using oriented equations
//! - **Subsumption**: Removes redundant clauses
//! - **Tautology deletion**: Removes trivially true clauses
//!
//! # Term Orderings
//!
//! The implementation supports:
//! - **KBO** (Knuth-Bendix Ordering): Weight-based, efficient
//! - **LPO** (Lexicographic Path Ordering): Symbol precedence-based
//!
//! # Given Clause Loop
//!
//! Uses the DISCOUNT loop variant:
//! 1. Select a clause from the set of unprocessed clauses
//! 2. Simplify it with processed clauses
//! 3. Generate new clauses by inference with processed clauses
//! 4. Add to processed set

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;

/// A term in first-order logic
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    /// Variable (identified by index)
    Var(u32),
    /// Constant symbol
    Const(Symbol),
    /// Function application: f(t1, ..., tn)
    App(Symbol, Vec<Term>),
}

/// Symbol identifier
pub type Symbol = u32;

impl Term {
    /// Check if this term is a variable
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    /// Get all variables in this term
    pub fn vars(&self) -> HashSet<u32> {
        let mut result = HashSet::new();
        self.collect_vars(&mut result);
        result
    }

    fn collect_vars(&self, vars: &mut HashSet<u32>) {
        match self {
            Term::Var(v) => {
                vars.insert(*v);
            }
            Term::Const(_) => {}
            Term::App(_, args) => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
        }
    }

    /// Apply a substitution to this term
    #[must_use]
    pub fn apply_subst(&self, subst: &Substitution) -> Term {
        match self {
            Term::Var(v) => subst.get(*v).cloned().unwrap_or_else(|| self.clone()),
            Term::Const(_) => self.clone(),
            Term::App(f, args) => {
                Term::App(*f, args.iter().map(|a| a.apply_subst(subst)).collect())
            }
        }
    }

    /// Size of the term (number of symbols)
    pub fn size(&self) -> usize {
        match self {
            Term::Var(_) | Term::Const(_) => 1,
            Term::App(_, args) => 1 + args.iter().map(Term::size).sum::<usize>(),
        }
    }

    /// Get all positions in this term (path from root)
    pub fn positions(&self) -> Vec<Position> {
        let mut result = vec![Position(vec![])]; // Root position
        if let Term::App(_, args) = self {
            for (i, arg) in args.iter().enumerate() {
                for mut pos in arg.positions() {
                    pos.0.insert(0, i);
                    result.push(pos);
                }
            }
        }
        result
    }

    /// Get subterm at position
    pub fn at_position(&self, pos: &Position) -> Option<&Term> {
        let mut current = self;
        for &i in &pos.0 {
            match current {
                Term::App(_, args) if i < args.len() => {
                    current = &args[i];
                }
                _ => return None,
            }
        }
        Some(current)
    }

    /// Replace subterm at position
    pub fn replace_at(&self, pos: &Position, replacement: Term) -> Option<Term> {
        if pos.0.is_empty() {
            return Some(replacement);
        }
        match self {
            Term::App(f, args) => {
                let idx = pos.0[0];
                if idx >= args.len() {
                    return None;
                }
                let subpos = Position(pos.0[1..].to_vec());
                let new_arg = args[idx].replace_at(&subpos, replacement)?;
                let mut new_args = args.clone();
                new_args[idx] = new_arg;
                Some(Term::App(*f, new_args))
            }
            _ => None,
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Var(v) => write!(f, "X{v}"),
            Term::Const(c) => write!(f, "c{c}"),
            Term::App(func, args) if args.is_empty() => write!(f, "f{func}"),
            Term::App(func, args) => {
                write!(f, "f{func}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Position in a term (path from root)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Position(pub Vec<usize>);

impl Position {
    /// Root position
    pub fn root() -> Self {
        Position(vec![])
    }

    /// Check if this is the root position
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }
}

/// A substitution mapping variables to terms
#[derive(Clone, Debug, Default)]
pub struct Substitution {
    map: HashMap<u32, Term>,
}

impl Substitution {
    /// Create an empty substitution
    pub fn new() -> Self {
        Substitution {
            map: HashMap::new(),
        }
    }

    /// Bind a variable to a term
    pub fn bind(&mut self, var: u32, term: Term) {
        self.map.insert(var, term);
    }

    /// Get the binding for a variable
    pub fn get(&self, var: u32) -> Option<&Term> {
        self.map.get(&var)
    }

    /// Check if this substitution is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Compose two substitutions: (self ∘ other)
    #[must_use]
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();

        // Apply other to all terms in self
        for (var, term) in &self.map {
            result.bind(*var, term.apply_subst(other));
        }

        // Add bindings from other that aren't in self
        for (var, term) in &other.map {
            if !result.map.contains_key(var) {
                result.bind(*var, term.clone());
            }
        }

        result
    }
}

/// Attempt to unify two terms, returning a most general unifier (MGU)
pub fn unify(t1: &Term, t2: &Term) -> Option<Substitution> {
    let mut subst = Substitution::new();
    if unify_rec(t1, t2, &mut subst) {
        Some(subst)
    } else {
        None
    }
}

fn unify_rec(t1: &Term, t2: &Term, subst: &mut Substitution) -> bool {
    let t1 = apply_subst_to_term(t1, subst);
    let t2 = apply_subst_to_term(t2, subst);

    match (&t1, &t2) {
        (Term::Var(v1), Term::Var(v2)) if v1 == v2 => true,
        (Term::Var(v), t) | (t, Term::Var(v)) => {
            // Occurs check
            if t.vars().contains(v) {
                return false;
            }
            subst.bind(*v, t.clone());
            true
        }
        (Term::Const(c1), Term::Const(c2)) => c1 == c2,
        (Term::App(f1, args1), Term::App(f2, args2)) => {
            if f1 != f2 || args1.len() != args2.len() {
                return false;
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                if !unify_rec(a1, a2, subst) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn apply_subst_to_term(term: &Term, subst: &Substitution) -> Term {
    match term {
        Term::Var(v) => match subst.get(*v) {
            Some(t) => apply_subst_to_term(t, subst),
            None => term.clone(),
        },
        Term::Const(_) => term.clone(),
        Term::App(f, args) => Term::App(
            *f,
            args.iter().map(|a| apply_subst_to_term(a, subst)).collect(),
        ),
    }
}

/// Attempt to match t1 against t2 (one-way unification)
/// Returns a substitution σ such that t1σ = t2
pub fn match_terms(pattern: &Term, target: &Term) -> Option<Substitution> {
    let mut subst = Substitution::new();
    if match_rec(pattern, target, &mut subst) {
        Some(subst)
    } else {
        None
    }
}

fn match_rec(pattern: &Term, target: &Term, subst: &mut Substitution) -> bool {
    match (pattern, target) {
        (Term::Var(v), _) => {
            if let Some(bound) = subst.get(*v) {
                bound == target
            } else {
                subst.bind(*v, target.clone());
                true
            }
        }
        (Term::Const(c1), Term::Const(c2)) => c1 == c2,
        (Term::App(f1, args1), Term::App(f2, args2)) => {
            if f1 != f2 || args1.len() != args2.len() {
                return false;
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                if !match_rec(a1, a2, subst) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// A literal in a clause
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Literal {
    /// Left-hand side of equation
    pub lhs: Term,
    /// Right-hand side of equation
    pub rhs: Term,
    /// True if positive (=), false if negative (≠)
    pub positive: bool,
}

impl Literal {
    /// Create a positive equation
    pub fn eq(lhs: Term, rhs: Term) -> Self {
        Literal {
            lhs,
            rhs,
            positive: true,
        }
    }

    /// Create a negative equation (disequation)
    pub fn neq(lhs: Term, rhs: Term) -> Self {
        Literal {
            lhs,
            rhs,
            positive: false,
        }
    }

    /// Negate this literal
    #[must_use]
    pub fn negate(&self) -> Self {
        Literal {
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
            positive: !self.positive,
        }
    }

    /// Apply a substitution to this literal
    #[must_use]
    pub fn apply_subst(&self, subst: &Substitution) -> Self {
        Literal {
            lhs: self.lhs.apply_subst(subst),
            rhs: self.rhs.apply_subst(subst),
            positive: self.positive,
        }
    }

    /// Check if this is a trivial literal (s = s or s ≠ s)
    pub fn is_trivial(&self) -> bool {
        self.lhs == self.rhs
    }

    /// Check if this is a reflexive positive equation (s = s)
    pub fn is_reflexive(&self) -> bool {
        self.positive && self.lhs == self.rhs
    }

    /// Get all variables in this literal
    pub fn vars(&self) -> HashSet<u32> {
        let mut vars = self.lhs.vars();
        vars.extend(self.rhs.vars());
        vars
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.positive {
            write!(f, "{} = {}", self.lhs, self.rhs)
        } else {
            write!(f, "{} ≠ {}", self.lhs, self.rhs)
        }
    }
}

/// A clause is a disjunction of literals
#[derive(Clone, Debug)]
pub struct Clause {
    /// The literals in this clause
    pub literals: Vec<Literal>,
    /// Unique identifier
    pub id: u64,
    /// Parent clause IDs (for proof reconstruction)
    pub parents: Vec<u64>,
    /// Inference rule that derived this clause
    pub inference: Inference,
}

/// Inference rule that derived a clause
#[derive(Clone, Debug)]
pub enum Inference {
    /// Input clause from the problem
    Input,
    /// Superposition left or right
    Superposition(u64, u64, Position),
    /// Equality resolution
    EqualityResolution(u64),
    /// Equality factoring
    EqualityFactoring(u64),
    /// Demodulation (simplification)
    Demodulation(u64, u64),
    /// Subsumption deletion
    Subsumption(u64),
}

impl Clause {
    /// Create a new input clause
    pub fn new(literals: Vec<Literal>, id: u64) -> Self {
        Clause {
            literals,
            id,
            parents: vec![],
            inference: Inference::Input,
        }
    }

    /// Check if this is the empty clause (contradiction)
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if this is a unit clause
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Check if this is a tautology (contains s=s or both s=t and s≠t)
    pub fn is_tautology(&self) -> bool {
        // Check for reflexive equalities
        for lit in &self.literals {
            if lit.positive && lit.lhs == lit.rhs {
                return true;
            }
        }

        // Check for complementary literals
        for (i, lit1) in self.literals.iter().enumerate() {
            for lit2 in self.literals.iter().skip(i + 1) {
                if lit1.lhs == lit2.lhs && lit1.rhs == lit2.rhs && lit1.positive != lit2.positive {
                    return true;
                }
            }
        }

        false
    }

    /// Apply a substitution to this clause
    #[must_use]
    pub fn apply_subst(&self, subst: &Substitution) -> Self {
        Clause {
            literals: self.literals.iter().map(|l| l.apply_subst(subst)).collect(),
            id: self.id,
            parents: self.parents.clone(),
            inference: self.inference.clone(),
        }
    }

    /// Get all variables in this clause
    pub fn vars(&self) -> HashSet<u32> {
        let mut vars = HashSet::new();
        for lit in &self.literals {
            vars.extend(lit.vars());
        }
        vars
    }

    /// Rename variables to avoid conflicts with another clause
    #[must_use]
    pub fn rename_vars(&self, offset: u32) -> Self {
        let subst = Substitution {
            map: self
                .vars()
                .into_iter()
                .map(|v| (v, Term::Var(v + offset)))
                .collect(),
        };
        self.apply_subst(&subst)
    }

    /// Get positive literals
    pub fn positive_literals(&self) -> Vec<(usize, &Literal)> {
        self.literals
            .iter()
            .enumerate()
            .filter(|(_, l)| l.positive)
            .collect()
    }

    /// Get negative literals
    pub fn negative_literals(&self) -> Vec<(usize, &Literal)> {
        self.literals
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.positive)
            .collect()
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.literals.is_empty() {
            write!(f, "□")
        } else {
            for (i, lit) in self.literals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "{lit}")?;
            }
            Ok(())
        }
    }
}

/// Term ordering trait
pub trait TermOrdering: Send + Sync {
    /// Compare two terms
    fn compare(&self, t1: &Term, t2: &Term) -> Option<Ordering>;

    /// Check if t1 > t2
    fn greater(&self, t1: &Term, t2: &Term) -> bool {
        matches!(self.compare(t1, t2), Some(Ordering::Greater))
    }

    /// Check if t1 >= t2
    fn greater_equal(&self, t1: &Term, t2: &Term) -> bool {
        matches!(
            self.compare(t1, t2),
            Some(Ordering::Greater | Ordering::Equal)
        )
    }
}

/// Knuth-Bendix Ordering
pub struct KBO {
    /// Weight of each function symbol
    weights: HashMap<Symbol, u32>,
    /// Precedence of function symbols (higher = greater)
    precedence: HashMap<Symbol, u32>,
    /// Default weight for unknown symbols
    default_weight: u32,
    /// Weight of variables
    var_weight: u32,
}

impl Default for KBO {
    fn default() -> Self {
        Self::new()
    }
}

impl KBO {
    /// Create a new KBO with default settings
    pub fn new() -> Self {
        KBO {
            weights: HashMap::new(),
            precedence: HashMap::new(),
            default_weight: 1,
            var_weight: 1,
        }
    }

    /// Set the weight of a symbol
    pub fn set_weight(&mut self, sym: Symbol, weight: u32) {
        self.weights.insert(sym, weight);
    }

    /// Set the precedence of a symbol
    pub fn set_precedence(&mut self, sym: Symbol, prec: u32) {
        self.precedence.insert(sym, prec);
    }

    fn weight(&self, term: &Term) -> u32 {
        match term {
            Term::Var(_) => self.var_weight,
            Term::Const(c) => *self.weights.get(c).unwrap_or(&self.default_weight),
            Term::App(f, args) => {
                let f_weight = *self.weights.get(f).unwrap_or(&self.default_weight);
                f_weight + args.iter().map(|a| self.weight(a)).sum::<u32>()
            }
        }
    }

    fn prec(&self, sym: Symbol) -> u32 {
        *self.precedence.get(&sym).unwrap_or(&sym)
    }

    /// Count variable occurrences in a term
    fn var_count(&self, term: &Term) -> HashMap<u32, i32> {
        let mut counts = HashMap::new();
        self.collect_var_counts(term, &mut counts, 1);
        counts
    }

    fn collect_var_counts(&self, term: &Term, counts: &mut HashMap<u32, i32>, sign: i32) {
        match term {
            Term::Var(v) => {
                *counts.entry(*v).or_insert(0) += sign;
            }
            Term::Const(_) => {}
            Term::App(_, args) => {
                for arg in args {
                    self.collect_var_counts(arg, counts, sign);
                }
            }
        }
    }
}

impl TermOrdering for KBO {
    fn compare(&self, t1: &Term, t2: &Term) -> Option<Ordering> {
        if t1 == t2 {
            return Some(Ordering::Equal);
        }

        let w1 = self.weight(t1);
        let w2 = self.weight(t2);

        // Check variable condition: for each variable, count(t1) >= count(t2)
        let mut counts = self.var_count(t1);
        self.collect_var_counts(t2, &mut counts, -1);

        let all_non_negative = counts.values().all(|&c| c >= 0);
        let all_non_positive = counts.values().all(|&c| c <= 0);

        if w1 > w2 && all_non_negative {
            return Some(Ordering::Greater);
        }
        if w1 < w2 && all_non_positive {
            return Some(Ordering::Less);
        }

        // If weights are equal, compare by precedence
        if w1 == w2 {
            match (t1, t2) {
                (Term::App(f1, args1), Term::App(f2, args2)) => {
                    let p1 = self.prec(*f1);
                    let p2 = self.prec(*f2);
                    if p1 > p2 && all_non_negative {
                        return Some(Ordering::Greater);
                    }
                    if p1 < p2 && all_non_positive {
                        return Some(Ordering::Less);
                    }
                    if p1 == p2 && f1 == f2 {
                        // Lexicographic comparison of arguments
                        for (a1, a2) in args1.iter().zip(args2.iter()) {
                            match self.compare(a1, a2) {
                                Some(Ordering::Equal) => continue,
                                Some(Ordering::Greater) if all_non_negative => {
                                    return Some(Ordering::Greater)
                                }
                                Some(Ordering::Less) if all_non_positive => {
                                    return Some(Ordering::Less)
                                }
                                _ => return None,
                            }
                        }
                    }
                }
                (Term::Const(c1), Term::Const(c2)) => {
                    let p1 = self.prec(*c1);
                    let p2 = self.prec(*c2);
                    if p1 > p2 {
                        return Some(Ordering::Greater);
                    }
                    if p1 < p2 {
                        return Some(Ordering::Less);
                    }
                }
                _ => {}
            }
        }

        None
    }
}

/// Lexicographic Path Ordering
pub struct LPO {
    /// Precedence of function symbols (higher = greater)
    precedence: HashMap<Symbol, u32>,
}

impl Default for LPO {
    fn default() -> Self {
        Self::new()
    }
}

impl LPO {
    /// Create a new LPO with default precedence
    pub fn new() -> Self {
        LPO {
            precedence: HashMap::new(),
        }
    }

    /// Set the precedence of a symbol
    pub fn set_precedence(&mut self, sym: Symbol, prec: u32) {
        self.precedence.insert(sym, prec);
    }

    fn prec(&self, sym: Symbol) -> u32 {
        *self.precedence.get(&sym).unwrap_or(&sym)
    }

    fn lpo_gt(&self, s: &Term, t: &Term) -> bool {
        match s {
            Term::Var(_) => false,
            Term::Const(f) => match t {
                Term::Var(x) => s.vars().contains(x),
                Term::Const(g) => self.prec(*f) > self.prec(*g),
                Term::App(_, _) => false,
            },
            Term::App(f, ss) => {
                // s = f(s1,...,sm)
                // Case 1: some si >= t
                if ss.iter().any(|si| self.lpo_ge(si, t)) {
                    return true;
                }

                match t {
                    Term::Var(x) => s.vars().contains(x),
                    Term::Const(g) => self.prec(*f) > self.prec(*g),
                    Term::App(g, ts) => {
                        // t = g(t1,...,tn)
                        // Case 2: f > g and s > ti for all i
                        if self.prec(*f) > self.prec(*g) {
                            return ts.iter().all(|ti| self.lpo_gt(s, ti));
                        }
                        // Case 3: f = g and lexicographic comparison
                        if f == g {
                            for (si, ti) in ss.iter().zip(ts.iter()) {
                                if self.lpo_gt(si, ti) {
                                    return ts.iter().skip(1).all(|tj| self.lpo_gt(s, tj));
                                }
                                if si != ti {
                                    return false;
                                }
                            }
                        }
                        false
                    }
                }
            }
        }
    }

    fn lpo_ge(&self, s: &Term, t: &Term) -> bool {
        s == t || self.lpo_gt(s, t)
    }
}

impl TermOrdering for LPO {
    fn compare(&self, t1: &Term, t2: &Term) -> Option<Ordering> {
        if t1 == t2 {
            Some(Ordering::Equal)
        } else if self.lpo_gt(t1, t2) {
            Some(Ordering::Greater)
        } else if self.lpo_gt(t2, t1) {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

/// Clause selection strategy
#[derive(Clone, Copy, Debug)]
pub enum SelectionStrategy {
    /// First-in-first-out
    FIFO,
    /// Prefer smaller clauses
    SizeFirst,
    /// Prefer clauses with fewer symbols
    SymbolCount,
}

/// A wrapper for clause priority queue ordering
struct PrioritizedClause {
    clause: Clause,
    priority: i64,
}

impl PartialEq for PrioritizedClause {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedClause {}

impl PartialOrd for PrioritizedClause {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedClause {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.priority.cmp(&self.priority)
    }
}

/// Superposition prover
pub struct SuperpositionProver {
    /// Term ordering
    ordering: Box<dyn TermOrdering>,
    /// Processed clauses (active set)
    processed: Vec<Clause>,
    /// Unprocessed clauses (passive set)
    unprocessed: BinaryHeap<PrioritizedClause>,
    /// Clause ID counter
    next_id: u64,
    /// Selection strategy
    strategy: SelectionStrategy,
    /// Maximum clause size (for fair enumeration)
    max_clause_size: usize,
    /// Statistics
    pub stats: ProverStats,
}

/// Prover statistics
#[derive(Clone, Debug, Default)]
pub struct ProverStats {
    /// Number of inferences performed
    pub inferences: u64,
    /// Number of clauses generated
    pub generated: u64,
    /// Number of clauses kept (after simplification)
    pub kept: u64,
    /// Number of clauses deleted by subsumption
    pub subsumed: u64,
    /// Number of tautologies deleted
    pub tautologies: u64,
}

/// Result of the prover
#[derive(Clone, Debug)]
pub enum ProverResult {
    /// Unsatisfiable - found empty clause
    Unsatisfiable(ProofTrace),
    /// Satisfiable - saturated without finding empty clause
    Saturated,
    /// Resource limit reached
    ResourceLimit,
}

/// Proof trace for reconstruction
#[derive(Clone, Debug)]
pub struct ProofTrace {
    /// The empty clause
    pub empty_clause: Clause,
    /// All clauses used in the proof
    pub clauses: Vec<Clause>,
}

impl Default for SuperpositionProver {
    fn default() -> Self {
        Self::new()
    }
}

impl SuperpositionProver {
    /// Create a new prover with default KBO ordering
    pub fn new() -> Self {
        SuperpositionProver {
            ordering: Box::new(KBO::new()),
            processed: Vec::new(),
            unprocessed: BinaryHeap::new(),
            next_id: 0,
            strategy: SelectionStrategy::SizeFirst,
            max_clause_size: 100,
            stats: ProverStats::default(),
        }
    }

    /// Create a new prover with custom ordering
    pub fn with_ordering(ordering: Box<dyn TermOrdering>) -> Self {
        SuperpositionProver {
            ordering,
            processed: Vec::new(),
            unprocessed: BinaryHeap::new(),
            next_id: 0,
            strategy: SelectionStrategy::SizeFirst,
            max_clause_size: 100,
            stats: ProverStats::default(),
        }
    }

    /// Set the selection strategy
    pub fn set_strategy(&mut self, strategy: SelectionStrategy) {
        self.strategy = strategy;
    }

    /// Add an input clause
    pub fn add_clause(&mut self, literals: Vec<Literal>) {
        let clause = Clause::new(literals, self.next_id);
        self.next_id += 1;

        // Skip tautologies
        if clause.is_tautology() {
            self.stats.tautologies += 1;
            return;
        }

        let priority = self.compute_priority(&clause);
        self.unprocessed
            .push(PrioritizedClause { clause, priority });
        self.stats.generated += 1;
    }

    fn compute_priority(&self, clause: &Clause) -> i64 {
        match self.strategy {
            SelectionStrategy::FIFO => -(clause.id as i64),
            SelectionStrategy::SizeFirst => {
                -(clause
                    .literals
                    .iter()
                    .map(|l| l.lhs.size() + l.rhs.size())
                    .sum::<usize>() as i64)
            }
            SelectionStrategy::SymbolCount => -(clause.literals.len() as i64),
        }
    }

    /// Run the prover with a given iteration limit
    pub fn prove(&mut self, max_iterations: u64) -> ProverResult {
        for _ in 0..max_iterations {
            // Select the next clause to process
            let given = match self.unprocessed.pop() {
                Some(pc) => pc.clause,
                None => return ProverResult::Saturated,
            };

            // Check for empty clause
            if given.is_empty() {
                return ProverResult::Unsatisfiable(self.build_proof_trace(&given));
            }

            // Skip if clause is too large
            if given
                .literals
                .iter()
                .map(|l| l.lhs.size() + l.rhs.size())
                .sum::<usize>()
                > self.max_clause_size
            {
                continue;
            }

            // Forward simplification
            let Some(given) = self.forward_simplify(given) else {
                continue; // Subsumed or trivial
            };

            // Backward simplification
            self.backward_simplify(&given);

            // Generate new clauses
            let new_clauses = self.generate_clauses(&given);

            // Add given to processed
            self.processed.push(given);
            self.stats.kept += 1;

            // Add new clauses to unprocessed
            for clause in new_clauses {
                if clause.is_empty() {
                    return ProverResult::Unsatisfiable(self.build_proof_trace(&clause));
                }

                if clause.is_tautology() {
                    self.stats.tautologies += 1;
                } else {
                    let priority = self.compute_priority(&clause);
                    self.unprocessed
                        .push(PrioritizedClause { clause, priority });
                    self.stats.generated += 1;
                }
            }
        }

        ProverResult::ResourceLimit
    }

    fn build_proof_trace(&self, empty_clause: &Clause) -> ProofTrace {
        // Collect all clauses used in the proof
        let mut used = HashSet::new();
        let mut to_visit = vec![empty_clause.id];
        used.insert(empty_clause.id);

        while let Some(id) = to_visit.pop() {
            if let Some(clause) = self.find_clause(id) {
                for parent_id in &clause.parents {
                    if !used.contains(parent_id) {
                        used.insert(*parent_id);
                        to_visit.push(*parent_id);
                    }
                }
            }
        }

        let clauses: Vec<Clause> = self
            .processed
            .iter()
            .filter(|c| used.contains(&c.id))
            .cloned()
            .collect();

        ProofTrace {
            empty_clause: empty_clause.clone(),
            clauses,
        }
    }

    fn find_clause(&self, id: u64) -> Option<&Clause> {
        self.processed.iter().find(|c| c.id == id)
    }

    fn forward_simplify(&mut self, mut clause: Clause) -> Option<Clause> {
        // Check subsumption by processed clauses
        for processed in &self.processed {
            if self.subsumes(processed, &clause) {
                self.stats.subsumed += 1;
                return None;
            }
        }

        // Demodulation: simplify using unit equations
        for processed in &self.processed {
            if processed.is_unit() && processed.literals[0].positive {
                clause = self.demodulate(&clause, processed);
            }
        }

        // Remove duplicate and trivial literals
        clause.literals.retain(|l| !l.is_reflexive());
        clause
            .literals
            .sort_by(|a, b| format!("{a}").cmp(&format!("{b}")));
        clause.literals.dedup();

        if clause.is_tautology() {
            self.stats.tautologies += 1;
            return None;
        }

        Some(clause)
    }

    fn backward_simplify(&mut self, given: &Clause) {
        // For simplicity, we only do backward subsumption
        if !given.is_unit() {
            return;
        }

        let mut to_remove = vec![];
        for (i, processed) in self.processed.iter().enumerate() {
            if self.subsumes(given, processed) {
                to_remove.push(i);
                self.stats.subsumed += 1;
            }
        }

        // Remove in reverse order to preserve indices
        for i in to_remove.into_iter().rev() {
            self.processed.swap_remove(i);
        }
    }

    /// Check if c1 subsumes c2 (c1 is more general)
    fn subsumes(&self, c1: &Clause, c2: &Clause) -> bool {
        if c1.literals.len() > c2.literals.len() {
            return false;
        }

        // Try to find a substitution that maps c1's literals to a subset of c2's
        self.subsumes_rec(&c1.literals, &c2.literals, &Substitution::new())
    }

    fn subsumes_rec(
        &self,
        remaining: &[Literal],
        target: &[Literal],
        subst: &Substitution,
    ) -> bool {
        if remaining.is_empty() {
            return true;
        }

        let lit = &remaining[0];
        let rest = &remaining[1..];

        for target_lit in target {
            if lit.positive != target_lit.positive {
                continue;
            }

            // Try to match lit against target_lit
            let lit_applied = lit.apply_subst(subst);
            if let Some(ext) = match_terms(&lit_applied.lhs, &target_lit.lhs) {
                let combined = subst.compose(&ext);
                if let Some(ext2) = match_terms(&lit_applied.rhs.apply_subst(&ext), &target_lit.rhs)
                {
                    let final_subst = combined.compose(&ext2);
                    if self.subsumes_rec(rest, target, &final_subst) {
                        return true;
                    }
                }
            }
            // Also try symmetric matching for equations
            if lit.positive {
                if let Some(ext) = match_terms(&lit_applied.lhs, &target_lit.rhs) {
                    let combined = subst.compose(&ext);
                    if let Some(ext2) =
                        match_terms(&lit_applied.rhs.apply_subst(&ext), &target_lit.lhs)
                    {
                        let final_subst = combined.compose(&ext2);
                        if self.subsumes_rec(rest, target, &final_subst) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Demodulate (simplify) a clause using a unit equation
    fn demodulate(&self, clause: &Clause, unit: &Clause) -> Clause {
        debug_assert!(unit.is_unit() && unit.literals[0].positive);

        let eq = &unit.literals[0];
        let (big, small) = if self.ordering.greater(&eq.lhs, &eq.rhs) {
            (&eq.lhs, &eq.rhs)
        } else if self.ordering.greater(&eq.rhs, &eq.lhs) {
            (&eq.rhs, &eq.lhs)
        } else {
            return clause.clone(); // Not oriented
        };

        let mut result = clause.clone();
        let mut changed = false;

        for lit in &mut result.literals {
            for side in [&mut lit.lhs, &mut lit.rhs] {
                for pos in side.positions() {
                    if let Some(subterm) = side.at_position(&pos) {
                        if let Some(subst) = match_terms(big, subterm) {
                            let replacement = small.apply_subst(&subst);
                            if let Some(new_side) = side.replace_at(&pos, replacement) {
                                *side = new_side;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        if changed {
            result.parents = vec![clause.id, unit.id];
            result.inference = Inference::Demodulation(clause.id, unit.id);
        }

        result
    }

    /// Generate new clauses by inference rules
    fn generate_clauses(&mut self, given: &Clause) -> Vec<Clause> {
        let mut result = Vec::new();

        // Collect all pairs for superposition first to avoid borrow issues
        let processed_clauses: Vec<Clause> = self.processed.clone();

        // Superposition with processed clauses
        for processed in &processed_clauses {
            // Superposition: given into processed
            result.extend(self.superposition(given, processed));
            // Superposition: processed into given
            result.extend(self.superposition(processed, given));
        }

        // Self-superposition
        result.extend(self.superposition(given, given));

        // Equality resolution
        result.extend(self.equality_resolution(given));

        // Equality factoring
        result.extend(self.equality_factoring(given));

        self.stats.inferences += result.len() as u64;
        result
    }

    /// Superposition inference: rewrite c2 using equations from c1
    fn superposition(&mut self, c1: &Clause, c2: &Clause) -> Vec<Clause> {
        let mut result = Vec::new();

        // Rename variables in c2 to avoid conflicts
        let max_var = c1.vars().into_iter().max().unwrap_or(0);
        let c2_renamed = c2.rename_vars(max_var + 1);

        // For each positive equation l=r in c1
        for (i, lit1) in c1.positive_literals() {
            // Try l=r and r=l
            for (big, small) in [(&lit1.lhs, &lit1.rhs), (&lit1.rhs, &lit1.lhs)] {
                // Skip if big is a variable (no superposition into variables)
                if big.is_var() {
                    continue;
                }

                // For each literal in c2_renamed
                for (j, lit2) in c2_renamed.literals.iter().enumerate() {
                    // Get positions to superpose into
                    let positions: Vec<(bool, Position)> = lit2
                        .lhs
                        .positions()
                        .into_iter()
                        .map(|p| (true, p))
                        .chain(lit2.rhs.positions().into_iter().map(|p| (false, p)))
                        .collect();

                    for (is_lhs, pos) in positions {
                        // Skip variable positions
                        let target_term = if is_lhs { &lit2.lhs } else { &lit2.rhs };
                        if let Some(subterm) = target_term.at_position(&pos) {
                            if subterm.is_var() {
                                continue;
                            }

                            // Try to unify big with subterm
                            if let Some(mgu) = unify(big, subterm) {
                                // Check ordering constraints
                                let big_sigma = big.apply_subst(&mgu);
                                let small_sigma = small.apply_subst(&mgu);

                                if !self.ordering.greater_equal(&big_sigma, &small_sigma) {
                                    continue;
                                }

                                // Build the new literal
                                let new_lit = if is_lhs {
                                    let new_lhs = lit2
                                        .lhs
                                        .replace_at(&pos, small_sigma)
                                        .unwrap_or_else(|| lit2.lhs.apply_subst(&mgu));
                                    Literal {
                                        lhs: new_lhs,
                                        rhs: lit2.rhs.apply_subst(&mgu),
                                        positive: lit2.positive,
                                    }
                                } else {
                                    let new_rhs = lit2
                                        .rhs
                                        .replace_at(&pos, small_sigma)
                                        .unwrap_or_else(|| lit2.rhs.apply_subst(&mgu));
                                    Literal {
                                        lhs: lit2.lhs.apply_subst(&mgu),
                                        rhs: new_rhs,
                                        positive: lit2.positive,
                                    }
                                };

                                // Build the new clause
                                let mut new_literals = Vec::new();

                                // Add literals from c1 except the equation used
                                for (k, lit) in c1.literals.iter().enumerate() {
                                    if k != i {
                                        new_literals.push(lit.apply_subst(&mgu));
                                    }
                                }

                                // Add literals from c2_renamed except the one superposed into
                                for (k, lit) in c2_renamed.literals.iter().enumerate() {
                                    if k != j {
                                        new_literals.push(lit.apply_subst(&mgu));
                                    }
                                }

                                // Add the new literal
                                new_literals.push(new_lit);

                                let new_clause = Clause {
                                    literals: new_literals,
                                    id: self.next_id,
                                    parents: vec![c1.id, c2.id],
                                    inference: Inference::Superposition(c1.id, c2.id, pos),
                                };
                                self.next_id += 1;

                                result.push(new_clause);
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Equality resolution: from s ≠ t ∨ C, derive Cσ where σ = mgu(s, t)
    fn equality_resolution(&mut self, clause: &Clause) -> Vec<Clause> {
        let mut result = Vec::new();

        for (i, lit) in clause.negative_literals() {
            if let Some(mgu) = unify(&lit.lhs, &lit.rhs) {
                let mut new_literals = Vec::new();
                for (j, l) in clause.literals.iter().enumerate() {
                    if j != i {
                        new_literals.push(l.apply_subst(&mgu));
                    }
                }

                let new_clause = Clause {
                    literals: new_literals,
                    id: self.next_id,
                    parents: vec![clause.id],
                    inference: Inference::EqualityResolution(clause.id),
                };
                self.next_id += 1;

                result.push(new_clause);
            }
        }

        result
    }

    /// Equality factoring: from s = t ∨ s' = t' ∨ C, derive (s = t ∨ t ≠ t' ∨ C)σ
    /// where σ = mgu(s, s') and tσ is not greater than sσ
    fn equality_factoring(&mut self, clause: &Clause) -> Vec<Clause> {
        let mut result = Vec::new();
        let positive: Vec<_> = clause.positive_literals();

        for (idx1, (i, lit1)) in positive.iter().enumerate() {
            for (_, lit2) in positive.iter().skip(idx1 + 1) {
                // Try to unify the left-hand sides
                if let Some(mgu) = unify(&lit1.lhs, &lit2.lhs) {
                    let s_sigma = lit1.lhs.apply_subst(&mgu);
                    let t_sigma = lit1.rhs.apply_subst(&mgu);

                    // Check ordering constraint: t ≤ s
                    if self.ordering.greater(&t_sigma, &s_sigma) {
                        continue;
                    }

                    let mut new_literals = Vec::new();

                    // Keep first equation
                    new_literals.push(lit1.apply_subst(&mgu));

                    // Add disequation t ≠ t'
                    new_literals.push(Literal::neq(t_sigma, lit2.rhs.apply_subst(&mgu)));

                    // Add remaining literals
                    for (j, l) in clause.literals.iter().enumerate() {
                        if j != *i && !positive.iter().skip(idx1 + 1).any(|(k, _)| *k == j) {
                            new_literals.push(l.apply_subst(&mgu));
                        }
                    }

                    let new_clause = Clause {
                        literals: new_literals,
                        id: self.next_id,
                        parents: vec![clause.id],
                        inference: Inference::EqualityFactoring(clause.id),
                    };
                    self.next_id += 1;

                    result.push(new_clause);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: u32) -> Term {
        Term::Var(n)
    }

    fn const_(c: Symbol) -> Term {
        Term::Const(c)
    }

    fn app(f: Symbol, args: Vec<Term>) -> Term {
        Term::App(f, args)
    }

    #[test]
    fn test_unification_simple() {
        // X = a
        let t1 = var(0);
        let t2 = const_(0);
        let mgu = unify(&t1, &t2).unwrap();
        assert_eq!(t1.apply_subst(&mgu), t2);
    }

    #[test]
    fn test_unification_function() {
        // f(X, b) = f(a, Y)
        let t1 = app(0, vec![var(0), const_(1)]);
        let t2 = app(0, vec![const_(0), var(1)]);
        let mgu = unify(&t1, &t2).unwrap();
        assert_eq!(t1.apply_subst(&mgu), t2.apply_subst(&mgu));
    }

    #[test]
    fn test_unification_occurs_check() {
        // X = f(X) - should fail
        let t1 = var(0);
        let t2 = app(0, vec![var(0)]);
        assert!(unify(&t1, &t2).is_none());
    }

    #[test]
    fn test_matching() {
        // Pattern: f(X, X), Target: f(a, a)
        let pattern = app(0, vec![var(0), var(0)]);
        let target = app(0, vec![const_(0), const_(0)]);
        let subst = match_terms(&pattern, &target).unwrap();
        assert_eq!(pattern.apply_subst(&subst), target);
    }

    #[test]
    fn test_matching_fails() {
        // Pattern: f(X, X), Target: f(a, b) - should fail
        let pattern = app(0, vec![var(0), var(0)]);
        let target = app(0, vec![const_(0), const_(1)]);
        assert!(match_terms(&pattern, &target).is_none());
    }

    #[test]
    fn test_kbo_simple() {
        let kbo = KBO::new();

        // f(a) > a (more symbols)
        let t1 = app(0, vec![const_(0)]);
        let t2 = const_(0);
        assert!(kbo.greater(&t1, &t2));
    }

    #[test]
    fn test_kbo_same_weight() {
        let mut kbo = KBO::new();
        kbo.set_precedence(0, 10); // f has higher precedence
        kbo.set_precedence(1, 5); // g has lower precedence

        // f(a) > g(a) by precedence
        let t1 = app(0, vec![const_(0)]);
        let t2 = app(1, vec![const_(0)]);
        assert!(kbo.greater(&t1, &t2));
    }

    #[test]
    fn test_clause_tautology() {
        // a = a is a tautology
        let clause = Clause::new(vec![Literal::eq(const_(0), const_(0))], 0);
        assert!(clause.is_tautology());
    }

    #[test]
    fn test_clause_not_tautology() {
        // a = b is not a tautology
        let clause = Clause::new(vec![Literal::eq(const_(0), const_(1))], 0);
        assert!(!clause.is_tautology());
    }

    #[test]
    fn test_prover_trivial_unsat() {
        // { a = b }, { a ≠ b } is unsatisfiable
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::eq(const_(0), const_(1))]);
        prover.add_clause(vec![Literal::neq(const_(0), const_(1))]);

        match prover.prove(100) {
            ProverResult::Unsatisfiable(_) => {}
            other => panic!("Expected Unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn test_prover_trivial_sat() {
        // { a = a } is satisfiable (tautology removed, saturates immediately)
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::eq(const_(0), const_(0))]);

        match prover.prove(100) {
            ProverResult::Saturated => {}
            other => panic!("Expected Saturated, got {other:?}"),
        }
    }

    #[test]
    fn test_prover_symmetry() {
        // { a = b }, { b ≠ a } is unsatisfiable
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::eq(const_(0), const_(1))]);
        prover.add_clause(vec![Literal::neq(const_(1), const_(0))]);

        match prover.prove(100) {
            ProverResult::Unsatisfiable(_) => {}
            other => panic!("Expected Unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn test_prover_transitivity() {
        // { a = b }, { b = c }, { a ≠ c } is unsatisfiable
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::eq(const_(0), const_(1))]); // a = b
        prover.add_clause(vec![Literal::eq(const_(1), const_(2))]); // b = c
        prover.add_clause(vec![Literal::neq(const_(0), const_(2))]); // a ≠ c

        match prover.prove(100) {
            ProverResult::Unsatisfiable(_) => {}
            other => panic!("Expected Unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn test_prover_congruence() {
        // { a = b }, { f(a) ≠ f(b) } is unsatisfiable
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::eq(const_(0), const_(1))]); // a = b
        let fa = app(0, vec![const_(0)]); // f(a)
        let fb = app(0, vec![const_(1)]); // f(b)
        prover.add_clause(vec![Literal::neq(fa, fb)]); // f(a) ≠ f(b)

        match prover.prove(100) {
            ProverResult::Unsatisfiable(_) => {}
            other => panic!("Expected Unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn test_prover_nested_congruence() {
        // { a = b }, { g(f(a)) ≠ g(f(b)) } is unsatisfiable
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::eq(const_(0), const_(1))]); // a = b
        let fa = app(0, vec![const_(0)]); // f(a)
        let fb = app(0, vec![const_(1)]); // f(b)
        let gfa = app(1, vec![fa]); // g(f(a))
        let gfb = app(1, vec![fb]); // g(f(b))
        prover.add_clause(vec![Literal::neq(gfa, gfb)]); // g(f(a)) ≠ g(f(b))

        match prover.prove(1000) {
            ProverResult::Unsatisfiable(_) => {}
            other => panic!("Expected Unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn test_prover_equality_resolution() {
        // { X ≠ X } is unsatisfiable (equality resolution derives empty clause)
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::neq(var(0), var(0))]); // X ≠ X

        match prover.prove(100) {
            ProverResult::Unsatisfiable(_) => {}
            other => panic!("Expected Unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn test_prover_disjunction() {
        // { a = b ∨ a = c }, { a ≠ b }, { a ≠ c } is unsatisfiable
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![
            Literal::eq(const_(0), const_(1)), // a = b
            Literal::eq(const_(0), const_(2)), // a = c
        ]);
        prover.add_clause(vec![Literal::neq(const_(0), const_(1))]); // a ≠ b
        prover.add_clause(vec![Literal::neq(const_(0), const_(2))]); // a ≠ c

        match prover.prove(100) {
            ProverResult::Unsatisfiable(_) => {}
            other => panic!("Expected Unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn test_term_positions() {
        // f(g(a), b) has positions: [], [0], [0,0], [1]
        let term = app(0, vec![app(1, vec![const_(0)]), const_(1)]);
        let positions = term.positions();
        assert!(positions.contains(&Position(vec![])));
        assert!(positions.contains(&Position(vec![0])));
        assert!(positions.contains(&Position(vec![0, 0])));
        assert!(positions.contains(&Position(vec![1])));
        assert_eq!(positions.len(), 4);
    }

    #[test]
    fn test_term_replacement() {
        // Replace g(a) with c in f(g(a), b) to get f(c, b)
        let term = app(0, vec![app(1, vec![const_(0)]), const_(1)]);
        let result = term.replace_at(&Position(vec![0]), const_(2)).unwrap();
        assert_eq!(result, app(0, vec![const_(2), const_(1)]));
    }

    #[test]
    fn test_lpo_simple() {
        let lpo = LPO::new();

        // f(a, b) > a
        let t1 = app(0, vec![const_(0), const_(1)]);
        let t2 = const_(0);
        assert!(lpo.greater(&t1, &t2));
    }

    #[test]
    fn test_subsumption() {
        let prover = SuperpositionProver::new();

        // a = b subsumes a = b ∨ c = d
        let c1 = Clause::new(vec![Literal::eq(const_(0), const_(1))], 0);
        let c2 = Clause::new(
            vec![
                Literal::eq(const_(0), const_(1)),
                Literal::eq(const_(2), const_(3)),
            ],
            1,
        );
        assert!(prover.subsumes(&c1, &c2));

        // a = b ∨ c = d does not subsume a = b
        assert!(!prover.subsumes(&c2, &c1));
    }

    #[test]
    fn test_subsumption_with_variables() {
        let prover = SuperpositionProver::new();

        // X = Y subsumes a = b
        let c1 = Clause::new(vec![Literal::eq(var(0), var(1))], 0);
        let c2 = Clause::new(vec![Literal::eq(const_(0), const_(1))], 1);
        assert!(prover.subsumes(&c1, &c2));

        // a = b does not subsume X = Y
        assert!(!prover.subsumes(&c2, &c1));
    }

    #[test]
    fn test_prover_stats() {
        let mut prover = SuperpositionProver::new();
        prover.add_clause(vec![Literal::eq(const_(0), const_(1))]);
        prover.add_clause(vec![Literal::neq(const_(0), const_(1))]);

        let _ = prover.prove(100);

        // Should have generated some clauses and performed inferences
        assert!(prover.stats.generated >= 2);
    }
}
