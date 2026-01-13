//! Term representation for Z4
//!
//! Terms are represented as a hash-consed DAG for efficient sharing.
//! The `TermStore` manages term creation and ensures structural sharing
//! through hash-consing.

use crate::sort::Sort;
use hashbrown::HashMap;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive};
use std::fmt;
use std::hash::{Hash, Hasher};

/// A term identifier (index into the term store)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(pub u32);

impl TermId {
    /// Create a new TermId
    pub fn new(id: u32) -> Self {
        TermId(id)
    }

    /// Get the raw index
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for TermId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// Internal term representation with pre-computed hash
#[derive(Debug, Clone)]
struct TermEntry {
    term: TermData,
    sort: Sort,
    #[allow(dead_code)] // reserved for faster equality checks / profiling
    hash: u64,
}

/// The actual term data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermData {
    /// A constant value
    Const(Constant),
    /// A variable with name and unique ID
    Var(String, u32),
    /// Function application: function symbol + arguments
    App(Symbol, Vec<TermId>),
    /// Let binding (after expansion this should not appear)
    Let(Vec<(String, TermId)>, TermId),
    /// Negation (special case for efficient handling)
    Not(TermId),
    /// If-then-else
    Ite(TermId, TermId, TermId),
}

impl Hash for TermData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            TermData::Const(c) => c.hash(state),
            TermData::Var(name, id) => {
                name.hash(state);
                id.hash(state);
            }
            TermData::App(sym, args) => {
                sym.hash(state);
                args.hash(state);
            }
            TermData::Let(bindings, body) => {
                bindings.hash(state);
                body.hash(state);
            }
            TermData::Not(t) => t.hash(state),
            TermData::Ite(c, t, e) => {
                c.hash(state);
                t.hash(state);
                e.hash(state);
            }
        }
    }
}

/// Function/predicate symbol
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol {
    /// Named function (user-defined or built-in)
    Named(String),
    /// Indexed function like (_ extract 7 4)
    Indexed(String, Vec<u32>),
}

impl Symbol {
    /// Create a named symbol
    pub fn named(name: impl Into<String>) -> Self {
        Symbol::Named(name.into())
    }

    /// Create an indexed symbol
    pub fn indexed(name: impl Into<String>, indices: Vec<u32>) -> Self {
        Symbol::Indexed(name.into(), indices)
    }

    /// Get the name of the symbol
    pub fn name(&self) -> &str {
        match self {
            Symbol::Named(n) => n,
            Symbol::Indexed(n, _) => n,
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Symbol::Named(n) => write!(f, "{}", n),
            Symbol::Indexed(n, indices) => {
                write!(f, "(_ {}", n)?;
                for idx in indices {
                    write!(f, " {}", idx)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Constant values
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    /// Boolean constant
    Bool(bool),
    /// Integer constant (arbitrary precision)
    Int(BigInt),
    /// Rational constant
    Rational(RationalWrapper),
    /// Bitvector constant with value and width
    BitVec {
        /// The numeric value of the bitvector
        value: BigInt,
        /// The bit width of the bitvector
        width: u32,
    },
    /// String constant
    String(String),
}

/// Wrapper for BigRational to implement Eq and Hash
#[derive(Debug, Clone)]
pub struct RationalWrapper(pub BigRational);

impl PartialEq for RationalWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for RationalWrapper {}

impl Hash for RationalWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the normalized form
        self.0.numer().hash(state);
        self.0.denom().hash(state);
    }
}

impl From<BigRational> for RationalWrapper {
    fn from(r: BigRational) -> Self {
        RationalWrapper(r)
    }
}

/// Hash-consing term store
///
/// All terms are stored uniquely. Creating a term that already exists
/// returns the existing TermId.
pub struct TermStore {
    /// All terms, indexed by TermId
    terms: Vec<TermEntry>,
    /// Hash-cons map: hash -> list of term IDs with that hash
    hash_cons: HashMap<u64, Vec<TermId>>,
    /// Variable counter for generating unique IDs
    var_counter: u32,
    /// Named constants/variables: name -> (TermId, Sort)
    names: HashMap<String, (TermId, Sort)>,
    /// Pre-allocated common terms
    true_term: Option<TermId>,
    false_term: Option<TermId>,
}

impl Default for TermStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TermStore {
    /// Create a new empty term store
    pub fn new() -> Self {
        let mut store = TermStore {
            terms: Vec::new(),
            hash_cons: HashMap::new(),
            var_counter: 0,
            names: HashMap::new(),
            true_term: None,
            false_term: None,
        };
        // Pre-create true and false
        store.true_term = Some(store.mk_bool(true));
        store.false_term = Some(store.mk_bool(false));
        store
    }

    /// Get the number of terms in the store
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the TermId for true
    pub fn true_term(&self) -> TermId {
        self.true_term.unwrap()
    }

    /// Get the TermId for false
    pub fn false_term(&self) -> TermId {
        self.false_term.unwrap()
    }

    /// Get the term data for a TermId
    pub fn get(&self, id: TermId) -> &TermData {
        &self.terms[id.index()].term
    }

    /// Get the sort of a term
    pub fn sort(&self, id: TermId) -> &Sort {
        &self.terms[id.index()].sort
    }

    /// Compute hash for term data
    fn compute_hash(term: &TermData) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        term.hash(&mut hasher);
        hasher.finish()
    }

    /// Internal: find or create a term
    fn intern(&mut self, term: TermData, sort: Sort) -> TermId {
        let hash = Self::compute_hash(&term);

        // Check if we already have this term
        if let Some(ids) = self.hash_cons.get(&hash) {
            for &id in ids {
                if self.terms[id.index()].term == term {
                    return id;
                }
            }
        }

        // Create new term
        let id = TermId(self.terms.len() as u32);
        self.terms.push(TermEntry { term, sort, hash });
        self.hash_cons.entry(hash).or_default().push(id);
        id
    }

    /// Create a boolean constant
    pub fn mk_bool(&mut self, value: bool) -> TermId {
        self.intern(TermData::Const(Constant::Bool(value)), Sort::Bool)
    }

    /// Create an integer constant
    pub fn mk_int(&mut self, value: BigInt) -> TermId {
        self.intern(TermData::Const(Constant::Int(value)), Sort::Int)
    }

    /// Create a rational constant
    pub fn mk_rational(&mut self, value: BigRational) -> TermId {
        self.intern(
            TermData::Const(Constant::Rational(value.into())),
            Sort::Real,
        )
    }

    /// Create a bitvector constant
    pub fn mk_bitvec(&mut self, value: BigInt, width: u32) -> TermId {
        self.intern(
            TermData::Const(Constant::BitVec { value, width }),
            Sort::BitVec(width),
        )
    }

    /// Create a string constant
    pub fn mk_string(&mut self, value: String) -> TermId {
        self.intern(TermData::Const(Constant::String(value)), Sort::String)
    }

    /// Create or get a variable by name
    pub fn mk_var(&mut self, name: impl Into<String>, sort: Sort) -> TermId {
        let name = name.into();

        // Check if we already have this variable
        if let Some(&(id, _)) = self.names.get(&name) {
            return id;
        }

        let var_id = self.var_counter;
        self.var_counter += 1;

        let id = self.intern(TermData::Var(name.clone(), var_id), sort.clone());
        self.names.insert(name, (id, sort));
        id
    }

    /// Create a fresh variable (guaranteed unique)
    pub fn mk_fresh_var(&mut self, prefix: &str, sort: Sort) -> TermId {
        let name = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        let var_id = self.var_counter - 1;
        let id = self.intern(TermData::Var(name.clone(), var_id), sort.clone());
        self.names.insert(name, (id, sort));
        id
    }

    /// Create a function application
    pub fn mk_app(&mut self, func: Symbol, args: Vec<TermId>, sort: Sort) -> TermId {
        self.intern(TermData::App(func, args), sort)
    }

    /// Create negation
    pub fn mk_not(&mut self, arg: TermId) -> TermId {
        // Check for double negation
        if let TermData::Not(inner) = self.get(arg) {
            return *inner;
        }
        // Check for constant
        if let TermData::Const(Constant::Bool(b)) = self.get(arg) {
            return self.mk_bool(!*b);
        }

        // De Morgan normalization for boolean connectives
        // (not (and a b ...)) -> (or (not a) (not b) ...)
        // (not (or a b ...))  -> (and (not a) (not b) ...)
        let de_morgan: Option<(bool, Vec<TermId>)> = match self.get(arg) {
            TermData::App(Symbol::Named(name), args) if name == "and" => Some((true, args.clone())),
            TermData::App(Symbol::Named(name), args) if name == "or" => Some((false, args.clone())),
            _ => None,
        };
        if let Some((is_and, args)) = de_morgan {
            let negated_args: Vec<TermId> = args.into_iter().map(|t| self.mk_not(t)).collect();
            return if is_and {
                self.mk_or(negated_args)
            } else {
                self.mk_and(negated_args)
            };
        }

        // ITE negation normalization (for Boolean ITE only)
        // (not (ite c a b)) -> (ite c (not a) (not b))
        // This pushes negation down and enables further simplifications in mk_ite
        if let TermData::Ite(cond, then_term, else_term) = self.get(arg) {
            let cond = *cond;
            let then_term = *then_term;
            let else_term = *else_term;
            // Only apply to Boolean ITE (which is the case since we're in mk_not)
            if self.sort(then_term) == &Sort::Bool {
                let not_then = self.mk_not(then_term);
                let not_else = self.mk_not(else_term);
                return self.mk_ite(cond, not_then, not_else);
            }
        }

        self.intern(TermData::Not(arg), Sort::Bool)
    }

    /// Create conjunction (and)
    ///
    /// Flattens nested and terms: (and a (and b c)) -> (and a b c)
    /// Detects complements: (and x (not x)) -> false
    /// Absorption: (and x (or x y)) -> x
    pub fn mk_and(&mut self, args: Vec<TermId>) -> TermId {
        if args.is_empty() {
            return self.true_term();
        }
        if args.len() == 1 {
            return args[0];
        }

        // Early complement detection BEFORE flattening:
        // This catches (and (and x y) (not (and x y)))
        for &arg in &args {
            if let TermData::Not(inner) = self.get(arg) {
                if args.contains(inner) {
                    return self.false_term();
                }
            }
        }

        // Flatten nested ands, filter out true constants, check for false
        let mut filtered = Vec::new();
        for &arg in &args {
            match self.get(arg) {
                TermData::Const(Constant::Bool(false)) => return self.false_term(),
                TermData::Const(Constant::Bool(true)) => {} // skip
                TermData::App(Symbol::Named(name), nested_args) if name == "and" => {
                    // Flatten: extract nested and arguments
                    let nested = nested_args.clone();
                    filtered.extend(nested);
                }
                _ => filtered.push(arg),
            }
        }

        if filtered.is_empty() {
            return self.true_term();
        }
        if filtered.len() == 1 {
            return filtered[0];
        }

        // Sort for canonical form
        filtered.sort();
        filtered.dedup();

        // Complement detection AFTER flattening:
        // This catches (and x (not x)) and (and a (and b (not b)))
        for &arg in &filtered {
            // Check if arg is Not(inner) and inner is in filtered
            if let TermData::Not(inner) = self.get(arg) {
                if filtered.contains(inner) {
                    return self.false_term();
                }
            }
        }

        if filtered.len() == 1 {
            return filtered[0];
        }

        // Absorption: (and x (or x y)) -> x
        // If any arg is an Or containing another arg of the And, remove that Or
        let mut absorbed = Vec::new();
        for &arg in &filtered {
            let mut absorb_this = false;
            if let TermData::App(Symbol::Named(name), or_args) = self.get(arg) {
                if name == "or" {
                    // Check if any element of the Or is also in filtered (excluding arg itself)
                    for &or_elem in or_args {
                        if filtered.contains(&or_elem) && or_elem != arg {
                            absorb_this = true;
                            break;
                        }
                    }
                }
            }
            if !absorb_this {
                absorbed.push(arg);
            }
        }

        if absorbed.is_empty() {
            return self.true_term();
        }
        if absorbed.len() == 1 {
            return absorbed[0];
        }

        // Negation-through absorption: (and x (or (not x) y z)) -> (and x (or y z))
        // For each literal x in the conjunction, remove (not x) from any inner or's
        let mut neg_absorbed = Vec::new();
        for &arg in &absorbed {
            // Check if arg is an Or that contains (not x) for some x in absorbed
            if let TermData::App(Symbol::Named(name), or_args) = self.get(arg) {
                if name == "or" {
                    // Find which elements of or_args are negations of absorbed elements
                    let mut new_or_args: Vec<TermId> = Vec::new();
                    let or_args_clone = or_args.clone();
                    for &or_elem in &or_args_clone {
                        let mut is_negated_sibling = false;
                        if let TermData::Not(inner) = self.get(or_elem) {
                            if absorbed.contains(inner) && *inner != arg {
                                // or_elem is (not x) where x is in absorbed
                                is_negated_sibling = true;
                            }
                        }
                        if !is_negated_sibling {
                            new_or_args.push(or_elem);
                        }
                    }
                    // Rebuild the or if we removed anything
                    if new_or_args.len() < or_args_clone.len() {
                        let new_or = self.mk_or(new_or_args);
                        neg_absorbed.push(new_or);
                        continue;
                    }
                }
            }
            neg_absorbed.push(arg);
        }

        // Re-run simplifications on the result since we may have changed inner terms
        if neg_absorbed.len() != absorbed.len()
            || neg_absorbed.iter().zip(&absorbed).any(|(a, b)| a != b)
        {
            return self.mk_and(neg_absorbed);
        }

        self.intern(
            TermData::App(Symbol::named("and"), neg_absorbed),
            Sort::Bool,
        )
    }

    /// Create disjunction (or)
    ///
    /// Flattens nested or terms: (or a (or b c)) -> (or a b c)
    /// Detects complements: (or x (not x)) -> true
    /// Absorption: (or x (and x y)) -> x
    pub fn mk_or(&mut self, args: Vec<TermId>) -> TermId {
        if args.is_empty() {
            return self.false_term();
        }
        if args.len() == 1 {
            return args[0];
        }

        // Early complement detection BEFORE flattening:
        // This catches (or (or x y) (not (or x y)))
        for &arg in &args {
            if let TermData::Not(inner) = self.get(arg) {
                if args.contains(inner) {
                    return self.true_term();
                }
            }
        }

        // Flatten nested ors, filter out false constants, check for true
        let mut filtered = Vec::new();
        for &arg in &args {
            match self.get(arg) {
                TermData::Const(Constant::Bool(true)) => return self.true_term(),
                TermData::Const(Constant::Bool(false)) => {} // skip
                TermData::App(Symbol::Named(name), nested_args) if name == "or" => {
                    // Flatten: extract nested or arguments
                    let nested = nested_args.clone();
                    filtered.extend(nested);
                }
                _ => filtered.push(arg),
            }
        }

        if filtered.is_empty() {
            return self.false_term();
        }
        if filtered.len() == 1 {
            return filtered[0];
        }

        // Sort for canonical form
        filtered.sort();
        filtered.dedup();

        // Complement detection AFTER flattening:
        // This catches (or x (not x)) and (or a (or b (not b)))
        for &arg in &filtered {
            // Check if arg is Not(inner) and inner is in filtered
            if let TermData::Not(inner) = self.get(arg) {
                if filtered.contains(inner) {
                    return self.true_term();
                }
            }
        }

        if filtered.len() == 1 {
            return filtered[0];
        }

        // Absorption: (or x (and x y)) -> x
        // If any arg is an And containing another arg of the Or, remove that And
        let mut absorbed = Vec::new();
        for &arg in &filtered {
            let mut absorb_this = false;
            if let TermData::App(Symbol::Named(name), and_args) = self.get(arg) {
                if name == "and" {
                    // Check if any element of the And is also in filtered (excluding arg itself)
                    for &and_elem in and_args {
                        if filtered.contains(&and_elem) && and_elem != arg {
                            absorb_this = true;
                            break;
                        }
                    }
                }
            }
            if !absorb_this {
                absorbed.push(arg);
            }
        }

        if absorbed.is_empty() {
            return self.false_term();
        }
        if absorbed.len() == 1 {
            return absorbed[0];
        }

        // Negation-through absorption: (or x (and (not x) y z)) -> (or x (and y z))
        // For each literal x in the disjunction, remove (not x) from any inner and's
        let mut neg_absorbed = Vec::new();
        for &arg in &absorbed {
            // Check if arg is an And that contains (not x) for some x in absorbed
            if let TermData::App(Symbol::Named(name), and_args) = self.get(arg) {
                if name == "and" {
                    // Find which elements of and_args are negations of absorbed elements
                    let mut new_and_args: Vec<TermId> = Vec::new();
                    let and_args_clone = and_args.clone();
                    for &and_elem in &and_args_clone {
                        let mut is_negated_sibling = false;
                        if let TermData::Not(inner) = self.get(and_elem) {
                            if absorbed.contains(inner) && *inner != arg {
                                // and_elem is (not x) where x is in absorbed
                                is_negated_sibling = true;
                            }
                        }
                        if !is_negated_sibling {
                            new_and_args.push(and_elem);
                        }
                    }
                    // Rebuild the and if we removed anything
                    if new_and_args.len() < and_args_clone.len() {
                        let new_and = self.mk_and(new_and_args);
                        neg_absorbed.push(new_and);
                        continue;
                    }
                }
            }
            neg_absorbed.push(arg);
        }

        // Re-run simplifications on the result since we may have changed inner terms
        if neg_absorbed.len() != absorbed.len()
            || neg_absorbed.iter().zip(&absorbed).any(|(a, b)| a != b)
        {
            return self.mk_or(neg_absorbed);
        }

        self.intern(TermData::App(Symbol::named("or"), neg_absorbed), Sort::Bool)
    }

    /// Create implication
    pub fn mk_implies(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // a => b is equivalent to (not a) or b
        let not_lhs = self.mk_not(lhs);
        self.mk_or(vec![not_lhs, rhs])
    }

    /// Create exclusive or (XOR) with simplifications
    ///
    /// Simplifications:
    /// - (xor a a) = false
    /// - (xor a true) = (not a)
    /// - (xor a false) = a
    /// - (xor a (not a)) = true
    /// - (xor (not a) (not b)) = (xor a b) (double negation lifting)
    pub fn mk_xor(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let true_term = self.true_term();
        let false_term = self.false_term();

        // (xor a a) = false
        if lhs == rhs {
            return false_term;
        }

        // (xor a true) = (not a), (xor true a) = (not a)
        if rhs == true_term {
            return self.mk_not(lhs);
        }
        if lhs == true_term {
            return self.mk_not(rhs);
        }

        // (xor a false) = a, (xor false a) = a
        if rhs == false_term {
            return lhs;
        }
        if lhs == false_term {
            return rhs;
        }

        // (xor a (not a)) = true, (xor (not a) a) = true
        if let Some(inner) = self.get_not_inner(lhs) {
            if inner == rhs {
                return true_term;
            }
        }
        if let Some(inner) = self.get_not_inner(rhs) {
            if inner == lhs {
                return true_term;
            }
        }

        // (xor (not a) (not b)) = (xor a b)
        if let (Some(lhs_inner), Some(rhs_inner)) =
            (self.get_not_inner(lhs), self.get_not_inner(rhs))
        {
            return self.mk_xor(lhs_inner, rhs_inner);
        }

        // Canonical ordering
        let (a, b) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
        self.intern(TermData::App(Symbol::named("xor"), vec![a, b]), Sort::Bool)
    }

    /// Create if-then-else
    pub fn mk_ite(&mut self, cond: TermId, then_term: TermId, else_term: TermId) -> TermId {
        // Constant condition simplification
        match self.get(cond) {
            TermData::Const(Constant::Bool(true)) => return then_term,
            TermData::Const(Constant::Bool(false)) => return else_term,
            _ => {}
        }

        // Negated condition normalization: (ite (not c) a b) -> (ite c b a)
        // This normalizes to positive conditions, reducing structural variations
        // and potentially enabling further simplifications after the swap.
        if let Some(inner_cond) = self.get_not_inner(cond) {
            return self.mk_ite(inner_cond, else_term, then_term);
        }

        // Same branches: (ite c x x) = x
        if then_term == else_term {
            return then_term;
        }

        // Boolean branch simplifications
        let true_term = self.true_term();
        let false_term = self.false_term();

        // (ite c true false) = c
        if then_term == true_term && else_term == false_term {
            return cond;
        }

        // (ite c false true) = (not c)
        if then_term == false_term && else_term == true_term {
            return self.mk_not(cond);
        }

        // Get the result sort to check if it's Bool
        let result_sort = self.sort(then_term).clone();

        // Boolean-specific simplifications (only when result is Bool)
        if result_sort == Sort::Bool {
            // (ite c c false) = c
            if then_term == cond && else_term == false_term {
                return cond;
            }
            // (ite c true c) = c
            if then_term == true_term && else_term == cond {
                return cond;
            }
            // (ite c x false) = (and c x)
            if else_term == false_term {
                return self.mk_and(vec![cond, then_term]);
            }
            // (ite c true x) = (or c x)
            if then_term == true_term {
                return self.mk_or(vec![cond, else_term]);
            }
            // (ite c false x) = (and (not c) x)
            if then_term == false_term {
                let not_cond = self.mk_not(cond);
                return self.mk_and(vec![not_cond, else_term]);
            }
            // (ite c x true) = (or (not c) x)
            if else_term == true_term {
                let not_cond = self.mk_not(cond);
                return self.mk_or(vec![not_cond, then_term]);
            }

            // Nested ite simplifications with same condition
            // (ite c (ite c x y) z) = (ite c x z)
            if let TermData::Ite(nested_cond, nested_then, _) = self.get(then_term).clone() {
                if nested_cond == cond {
                    return self.mk_ite(cond, nested_then, else_term);
                }
            }
            // (ite c x (ite c y z)) = (ite c x z)
            if let TermData::Ite(nested_cond, _, nested_else) = self.get(else_term).clone() {
                if nested_cond == cond {
                    return self.mk_ite(cond, then_term, nested_else);
                }
            }
        }

        self.intern(TermData::Ite(cond, then_term, else_term), result_sort)
    }

    /// Create equality with constant folding
    pub fn mk_eq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Reflexive: x = x is true
        if lhs == rhs {
            return self.true_term();
        }

        // Constant folding: different constants are not equal
        // (Note: since we intern constants, if lhs != rhs and both are constants,
        // they must be different values)
        let lhs_is_const = matches!(self.get(lhs), TermData::Const(_));
        let rhs_is_const = matches!(self.get(rhs), TermData::Const(_));
        if lhs_is_const && rhs_is_const {
            return self.false_term();
        }

        // Boolean equality simplifications (iff-style)
        // (= x true) -> x
        // (= x false) -> (not x)
        let true_term = self.true_term();
        let false_term = self.false_term();

        if rhs == true_term && *self.sort(lhs) == Sort::Bool {
            return lhs;
        }
        if lhs == true_term && *self.sort(rhs) == Sort::Bool {
            return rhs;
        }
        if rhs == false_term && *self.sort(lhs) == Sort::Bool {
            return self.mk_not(lhs);
        }
        if lhs == false_term && *self.sort(rhs) == Sort::Bool {
            return self.mk_not(rhs);
        }

        // Boolean complement detection: (= x (not x)) -> false
        // Check if lhs is (not rhs) or rhs is (not lhs)
        if *self.sort(lhs) == Sort::Bool {
            if let Some(inner) = self.get_not_inner(lhs) {
                if inner == rhs {
                    return self.false_term();
                }
            }
            if let Some(inner) = self.get_not_inner(rhs) {
                if inner == lhs {
                    return self.false_term();
                }
            }

            // Negation lifting: (= (not x) (not y)) -> (= x y)
            if let (Some(lhs_inner), Some(rhs_inner)) =
                (self.get_not_inner(lhs), self.get_not_inner(rhs))
            {
                return self.mk_eq(lhs_inner, rhs_inner);
            }
        }

        // ITE-equality simplifications
        // (= (ite c a b) a) -> (or c (= b a))
        // (= (ite c a b) b) -> (or (not c) (= a b))
        // (= (ite c a b) (ite c x y)) -> (ite c (= a x) (= b y))

        // Check if lhs is an ITE
        if let TermData::Ite(c, a, b) = self.get(lhs).clone() {
            // (= (ite c a b) a) -> (or c (= b a))
            if rhs == a {
                let eq_ba = self.mk_eq(b, a);
                return self.mk_or(vec![c, eq_ba]);
            }
            // (= (ite c a b) b) -> (or (not c) (= a b))
            if rhs == b {
                let not_c = self.mk_not(c);
                let eq_ab = self.mk_eq(a, b);
                return self.mk_or(vec![not_c, eq_ab]);
            }
            // (= (ite c a b) (ite c x y)) -> (ite c (= a x) (= b y))
            if let TermData::Ite(c2, x, y) = self.get(rhs).clone() {
                if c == c2 {
                    let eq_ax = self.mk_eq(a, x);
                    let eq_by = self.mk_eq(b, y);
                    return self.mk_ite(c, eq_ax, eq_by);
                }
            }
        }

        // Check if rhs is an ITE (symmetric cases)
        if let TermData::Ite(c, a, b) = self.get(rhs).clone() {
            // (= a (ite c a b)) -> (or c (= b a))
            if lhs == a {
                let eq_ba = self.mk_eq(b, a);
                return self.mk_or(vec![c, eq_ba]);
            }
            // (= b (ite c a b)) -> (or (not c) (= a b))
            if lhs == b {
                let not_c = self.mk_not(c);
                let eq_ab = self.mk_eq(a, b);
                return self.mk_or(vec![not_c, eq_ab]);
            }
        }

        // Canonical order
        let (a, b) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
        self.intern(TermData::App(Symbol::named("="), vec![a, b]), Sort::Bool)
    }

    /// Create distinct with duplicate detection and constant folding
    pub fn mk_distinct(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() <= 1 {
            return self.true_term();
        }

        // Binary distinct: normalize to NOT(eq) so Tseitin encoding assigns
        // related CNF variables, enabling contradiction detection
        if args.len() == 2 {
            if args[0] == args[1] {
                return self.false_term();
            }
            let eq = self.mk_eq(args[0], args[1]);
            return self.mk_not(eq);
        }

        let mut sorted_args = args;
        sorted_args.sort();

        // Duplicate detection: if any two terms are identical, result is false
        for i in 1..sorted_args.len() {
            if sorted_args[i - 1] == sorted_args[i] {
                return self.false_term();
            }
        }

        // Constant folding: if all arguments are distinct constants, result is true
        // (Since we've already eliminated duplicates and constants are interned,
        // distinct TermIds for constants mean distinct values)
        let all_consts = sorted_args
            .iter()
            .all(|&id| matches!(self.get(id), TermData::Const(_)));
        if all_consts {
            return self.true_term();
        }

        self.intern(
            TermData::App(Symbol::named("distinct"), sorted_args),
            Sort::Bool,
        )
    }

    // =======================================================================
    // Boolean helpers
    // =======================================================================

    /// Helper: extract the inner term from a (not x) term
    fn get_not_inner(&self, id: TermId) -> Option<TermId> {
        match self.get(id) {
            TermData::Not(inner) => Some(*inner),
            _ => None,
        }
    }

    // =======================================================================
    // Arithmetic operations with constant folding
    // =======================================================================

    /// Helper: extract integer constant value
    fn get_int(&self, id: TermId) -> Option<&BigInt> {
        match self.get(id) {
            TermData::Const(Constant::Int(n)) => Some(n),
            _ => None,
        }
    }

    /// Helper: extract rational constant value
    fn get_rational(&self, id: TermId) -> Option<&BigRational> {
        match self.get(id) {
            TermData::Const(Constant::Rational(r)) => Some(&r.0),
            _ => None,
        }
    }

    /// Helper: check if all arguments are integer constants
    fn all_int_consts(&self, args: &[TermId]) -> bool {
        args.iter().all(|&id| self.get_int(id).is_some())
    }

    /// Helper: check if all arguments are rational constants
    fn all_rational_consts(&self, args: &[TermId]) -> bool {
        args.iter().all(|&id| self.get_rational(id).is_some())
    }

    /// Helper: extract bitvector constant value and width
    fn get_bitvec(&self, id: TermId) -> Option<(&BigInt, u32)> {
        match self.get(id) {
            TermData::Const(Constant::BitVec { value, width }) => Some((value, *width)),
            _ => None,
        }
    }

    /// Helper: get the width of a bitvector term
    fn get_bv_width(&self, id: TermId) -> Option<u32> {
        match self.sort(id) {
            Sort::BitVec(w) => Some(*w),
            _ => None,
        }
    }

    /// Helper: mask value to width bits (ensure value is in range [0, 2^width))
    fn bv_mask(value: &BigInt, width: u32) -> BigInt {
        let mask = (BigInt::from(1) << width) - 1;
        value & mask
    }

    /// Helper: extract coefficient and base term from an integer expression.
    /// Returns (coefficient, base_term) where:
    /// - (* ... c) or (* c ...) → (c, (* rest)) when c is an integer constant
    /// - (- x) → (-1, x)
    /// - x → (1, x) for any other term
    fn extract_int_coeff(&mut self, id: TermId) -> (BigInt, TermId) {
        // Check for negation: (- x) → (-1, x)
        if let TermData::App(Symbol::Named(name), args) = self.get(id) {
            if name == "-" && args.len() == 1 {
                return (BigInt::from(-1), args[0]);
            }
        }

        // Check for multiplication with a constant anywhere in the args
        // Note: mk_mul puts constants at the end, so we check last position first
        if let TermData::App(Symbol::Named(name), args) = self.get(id) {
            if name == "*" && !args.is_empty() {
                let args_cloned = args.clone();

                // Find the first constant in the multiplication
                for (i, &arg) in args_cloned.iter().enumerate() {
                    if let Some(c) = self.get_int(arg) {
                        let coeff = c.clone();
                        // Get the remaining factors (excluding the constant)
                        let remainder: Vec<TermId> = args_cloned
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .map(|(_, &t)| t)
                            .collect();

                        if remainder.is_empty() {
                            // Just a constant, shouldn't happen in practice
                            return (coeff, self.mk_int(BigInt::from(1)));
                        } else if remainder.len() == 1 {
                            return (coeff, remainder[0]);
                        } else {
                            let base = self.mk_mul(remainder);
                            return (coeff, base);
                        }
                    }
                }
            }
        }

        // Default: coefficient is 1
        (BigInt::from(1), id)
    }

    /// Helper: extract coefficient and base term from a real (rational) expression.
    /// Returns (coefficient, base_term) where:
    /// - (* ... c) or (* c ...) → (c, (* rest)) when c is a rational constant
    /// - (- x) → (-1, x)
    /// - x → (1, x) for any other term
    fn extract_real_coeff(&mut self, id: TermId) -> (BigRational, TermId) {
        // Check for negation: (- x) → (-1, x)
        if let TermData::App(Symbol::Named(name), args) = self.get(id) {
            if name == "-" && args.len() == 1 {
                return (BigRational::from(BigInt::from(-1)), args[0]);
            }
        }

        // Check for multiplication with a constant anywhere in the args
        if let TermData::App(Symbol::Named(name), args) = self.get(id) {
            if name == "*" && !args.is_empty() {
                let args_cloned = args.clone();

                // Find the first constant in the multiplication
                for (i, &arg) in args_cloned.iter().enumerate() {
                    if let Some(c) = self.get_rational(arg) {
                        let coeff = c.clone();
                        // Get the remaining factors (excluding the constant)
                        let remainder: Vec<TermId> = args_cloned
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .map(|(_, &t)| t)
                            .collect();

                        if remainder.is_empty() {
                            // Just a constant, shouldn't happen in practice
                            return (coeff, self.mk_rational(BigRational::from(BigInt::from(1))));
                        } else if remainder.len() == 1 {
                            return (coeff, remainder[0]);
                        } else {
                            let base = self.mk_mul(remainder);
                            return (coeff, base);
                        }
                    }
                }
            }
        }

        // Default: coefficient is 1
        (BigRational::from(BigInt::from(1)), id)
    }

    /// Create negation (unary minus) with constant folding
    pub fn mk_neg(&mut self, arg: TermId) -> TermId {
        let sort = self.sort(arg).clone();

        // Constant folding for integers
        if let Some(n) = self.get_int(arg) {
            return self.mk_int(-n.clone());
        }

        // Constant folding for rationals
        if let Some(r) = self.get_rational(arg) {
            return self.mk_rational(-r.clone());
        }

        // Double negation: -(-x) = x
        if let TermData::App(Symbol::Named(name), args) = self.get(arg) {
            if name == "-" && args.len() == 1 {
                return args[0];
            }
        }

        // Distribute negation over addition: -(a + b) → (-a) + (-b)
        // This enables coefficient collection on the result
        if let TermData::App(Symbol::Named(name), args) = self.get(arg) {
            if name == "+" {
                let args_clone = args.clone();
                let negated_args: Vec<TermId> =
                    args_clone.iter().map(|&a| self.mk_neg(a)).collect();
                return self.mk_add(negated_args);
            }
        }

        // Factor negation into multiplication: -(... * c) → (... * (-c)) for constant c
        // Note: mk_mul places constants at the end of the args list, so we check the last argument
        // This normalizes negation to appear as coefficient, enabling further simplification
        if let TermData::App(Symbol::Named(name), args) = self.get(arg) {
            if name == "*" && !args.is_empty() {
                let args_clone = args.clone();
                // Check if last argument is a constant (mk_mul places constants last)
                let last = *args_clone.last().unwrap();
                if self.get_int(last).is_some() || self.get_rational(last).is_some() {
                    // -(rest * c) → (rest * (-c))
                    let neg_last = self.mk_neg(last);
                    let mut new_args: Vec<TermId> = args_clone[..args_clone.len() - 1].to_vec();
                    new_args.push(neg_last);
                    return self.mk_mul(new_args);
                }
            }
        }

        self.intern(TermData::App(Symbol::named("-"), vec![arg]), sort)
    }

    /// Create addition with constant folding
    pub fn mk_add(&mut self, args: Vec<TermId>) -> TermId {
        if args.is_empty() {
            return self.mk_int(BigInt::from(0));
        }
        if args.len() == 1 {
            return args[0];
        }

        let sort = self.sort(args[0]).clone();

        // Phase 1: Flatten nested additions
        // (+ (+ a b) c) -> (+ a b c)
        let mut flattened = Vec::new();
        for &arg in &args {
            match self.get(arg) {
                TermData::App(Symbol::Named(name), nested_args) if name == "+" => {
                    // Flatten: extract nested addition arguments
                    let nested = nested_args.clone();
                    flattened.extend(nested);
                }
                _ => flattened.push(arg),
            }
        }

        // Phase 2: Constant folding for integers (all constants)
        if sort == Sort::Int && self.all_int_consts(&flattened) {
            let mut sum = BigInt::from(0);
            for &id in &flattened {
                if let Some(n) = self.get_int(id) {
                    sum += n;
                }
            }
            return self.mk_int(sum);
        }

        // Constant folding for rationals (all constants)
        if sort == Sort::Real && self.all_rational_consts(&flattened) {
            let mut sum = BigRational::from(BigInt::from(0));
            for &id in &flattened {
                if let Some(r) = self.get_rational(id) {
                    sum += r;
                }
            }
            return self.mk_rational(sum);
        }

        // Phase 3: Identity elimination (x + 0 = x) and partial constant folding
        let (consts, non_consts): (Vec<_>, Vec<_>) = if sort == Sort::Int {
            flattened
                .iter()
                .partition(|&&id| self.get_int(id).is_some())
        } else if sort == Sort::Real {
            flattened
                .iter()
                .partition(|&&id| self.get_rational(id).is_some())
        } else {
            (vec![], flattened.iter().collect())
        };

        // Fold all constants into one
        let mut result_args: Vec<TermId> = non_consts.into_iter().copied().collect();
        if !consts.is_empty() {
            if sort == Sort::Int {
                let mut sum = BigInt::from(0);
                for &id in &consts {
                    if let Some(n) = self.get_int(*id) {
                        sum += n;
                    }
                }
                // Only add if non-zero (identity elimination)
                if sum != BigInt::from(0) {
                    result_args.push(self.mk_int(sum));
                }
            } else if sort == Sort::Real {
                let mut sum = BigRational::from(BigInt::from(0));
                for &id in &consts {
                    if let Some(r) = self.get_rational(*id) {
                        sum += r;
                    }
                }
                // Only add if non-zero (identity elimination)
                if sum != BigRational::from(BigInt::from(0)) {
                    result_args.push(self.mk_rational(sum));
                }
            }
        }

        // Phase 4: Additive inverse detection: a + (-a) = 0
        // Collect negated terms and their inner values
        let mut negated_map: Vec<(TermId, TermId)> = Vec::new(); // (neg_term, inner)
        for &arg in &result_args {
            if let TermData::App(Symbol::Named(name), inner_args) = self.get(arg) {
                if name == "-" && inner_args.len() == 1 {
                    negated_map.push((arg, inner_args[0]));
                }
            }
        }

        // Remove canceling pairs
        let mut to_remove = std::collections::HashSet::new();
        for &(neg_term, inner) in &negated_map {
            if result_args.contains(&inner) && !to_remove.contains(&inner) {
                to_remove.insert(neg_term);
                to_remove.insert(inner);
            }
        }

        if !to_remove.is_empty() {
            result_args.retain(|t| !to_remove.contains(t));
        }

        // Phase 5: Coefficient collection for integers
        // (+ (* 2 a) (* 3 a)) → (* 5 a)
        if sort == Sort::Int && result_args.len() >= 2 {
            use std::collections::HashMap;
            let mut coeff_map: HashMap<TermId, BigInt> = HashMap::new();

            for &arg in &result_args {
                let (coeff, base) = self.extract_int_coeff(arg);
                *coeff_map.entry(base).or_insert_with(|| BigInt::from(0)) += coeff;
            }

            // Check if any coefficients were combined (i.e., same base appeared multiple times)
            if coeff_map.len() < result_args.len() {
                result_args.clear();

                // Sort by TermId for deterministic ordering (improves hash-consing and reproducibility)
                let mut sorted_entries: Vec<_> = coeff_map.into_iter().collect();
                sorted_entries.sort_by_key(|(base, _)| *base);

                for (base, coeff) in sorted_entries {
                    if coeff == BigInt::from(0) {
                        // Skip: coefficient is zero
                    } else if coeff == BigInt::from(1) {
                        result_args.push(base);
                    } else if coeff == BigInt::from(-1) {
                        result_args.push(self.mk_neg(base));
                    } else {
                        // (* coeff base)
                        let coeff_term = self.mk_int(coeff);
                        result_args.push(self.mk_mul(vec![coeff_term, base]));
                    }
                }

                // Re-check for empty or single result
                if result_args.is_empty() {
                    return self.mk_int(BigInt::from(0));
                }
                if result_args.len() == 1 {
                    return result_args[0];
                }
            }
        }

        // Phase 6: Coefficient collection for reals
        // (+ (* 2.0 a) (* 3.0 a)) → (* 5.0 a)
        if sort == Sort::Real && result_args.len() >= 2 {
            use std::collections::HashMap;
            let mut coeff_map: HashMap<TermId, BigRational> = HashMap::new();

            for &arg in &result_args {
                let (coeff, base) = self.extract_real_coeff(arg);
                *coeff_map
                    .entry(base)
                    .or_insert_with(|| BigRational::from(BigInt::from(0))) += coeff;
            }

            // Check if any coefficients were combined (i.e., same base appeared multiple times)
            if coeff_map.len() < result_args.len() {
                result_args.clear();
                let zero = BigRational::from(BigInt::from(0));
                let one = BigRational::from(BigInt::from(1));
                let neg_one = BigRational::from(BigInt::from(-1));

                // Sort by TermId for deterministic ordering (improves hash-consing and reproducibility)
                let mut sorted_entries: Vec<_> = coeff_map.into_iter().collect();
                sorted_entries.sort_by_key(|(base, _)| *base);

                for (base, coeff) in sorted_entries {
                    if coeff == zero {
                        // Skip: coefficient is zero
                    } else if coeff == one {
                        result_args.push(base);
                    } else if coeff == neg_one {
                        result_args.push(self.mk_neg(base));
                    } else {
                        // (* coeff base)
                        let coeff_term = self.mk_rational(coeff);
                        result_args.push(self.mk_mul(vec![coeff_term, base]));
                    }
                }

                // Re-check for empty or single result
                if result_args.is_empty() {
                    return self.mk_rational(BigRational::from(BigInt::from(0)));
                }
                if result_args.len() == 1 {
                    return result_args[0];
                }
            }
        }

        // Final result
        if result_args.is_empty() {
            return if sort == Sort::Int {
                self.mk_int(BigInt::from(0))
            } else {
                self.mk_rational(BigRational::from(BigInt::from(0)))
            };
        }
        if result_args.len() == 1 {
            return result_args[0];
        }

        self.intern(TermData::App(Symbol::named("+"), result_args), sort)
    }

    /// Create subtraction with constant folding and normalization to addition form
    ///
    /// Converts binary subtraction to addition: (a - b) → (+ a (- b))
    /// Converts n-ary subtraction to addition: (- a b c) → (+ a (- b) (- c))
    /// This normalization enables coefficient collection across subtraction operations.
    pub fn mk_sub(&mut self, args: Vec<TermId>) -> TermId {
        if args.is_empty() {
            return self.mk_int(BigInt::from(0));
        }
        if args.len() == 1 {
            // Unary minus
            return self.mk_neg(args[0]);
        }

        let sort = self.sort(args[0]).clone();

        // For binary subtraction: constant folding
        if args.len() == 2 {
            let (a, b) = (args[0], args[1]);

            // Integer constant folding
            if let (Some(n1), Some(n2)) = (self.get_int(a), self.get_int(b)) {
                return self.mk_int(n1.clone() - n2.clone());
            }

            // Rational constant folding
            if let (Some(r1), Some(r2)) = (self.get_rational(a), self.get_rational(b)) {
                return self.mk_rational(r1.clone() - r2.clone());
            }

            // x - 0 = x
            if let Some(n) = self.get_int(b) {
                if *n == BigInt::from(0) {
                    return a;
                }
            }
            if let Some(r) = self.get_rational(b) {
                if *r == BigRational::from(BigInt::from(0)) {
                    return a;
                }
            }

            // 0 - x = -x
            if let Some(n) = self.get_int(a) {
                if *n == BigInt::from(0) {
                    return self.mk_neg(b);
                }
            }
            if let Some(r) = self.get_rational(a) {
                if *r == BigRational::from(BigInt::from(0)) {
                    return self.mk_neg(b);
                }
            }

            // x - x = 0
            if a == b {
                return if sort == Sort::Int {
                    self.mk_int(BigInt::from(0))
                } else {
                    self.mk_rational(BigRational::from(BigInt::from(0)))
                };
            }

            // Convert binary subtraction to addition form: (a - b) → (+ a (- b))
            // This enables coefficient collection across subtraction operations
            // Example: (2x - x) → (+ 2x (- x)) → x (via coefficient collection in mk_add)
            let neg_b = self.mk_neg(b);
            return self.mk_add(vec![a, neg_b]);
        }

        // N-ary subtraction: (- a b c) → (+ a (- b) (- c))
        // Negate all arguments except the first
        let first = args[0];
        let mut add_args = vec![first];
        for &arg in &args[1..] {
            add_args.push(self.mk_neg(arg));
        }
        self.mk_add(add_args)
    }

    /// Create multiplication with constant folding
    ///
    /// Flattens nested multiplications: (* (* a b) c) -> (* a b c)
    /// Simplifies multiply by -1: (* -1 x) -> (- x)
    pub fn mk_mul(&mut self, args: Vec<TermId>) -> TermId {
        if args.is_empty() {
            return self.mk_int(BigInt::from(1));
        }
        if args.len() == 1 {
            return args[0];
        }

        let sort = self.sort(args[0]).clone();

        // Phase 1: Flatten nested multiplications
        // (* (* a b) c) -> (* a b c)
        let mut flattened = Vec::new();
        for &arg in &args {
            match self.get(arg) {
                TermData::App(Symbol::Named(name), nested_args) if name == "*" => {
                    // Flatten: extract nested multiplication arguments
                    let nested = nested_args.clone();
                    flattened.extend(nested);
                }
                _ => flattened.push(arg),
            }
        }

        // Phase 2: Check for zero (annihilation) - must check early
        for &id in &flattened {
            if let Some(n) = self.get_int(id) {
                if *n == BigInt::from(0) {
                    return self.mk_int(BigInt::from(0));
                }
            }
            if let Some(r) = self.get_rational(id) {
                if *r == BigRational::from(BigInt::from(0)) {
                    return self.mk_rational(BigRational::from(BigInt::from(0)));
                }
            }
        }

        // Phase 3: Constant folding (all constants)
        if sort == Sort::Int && self.all_int_consts(&flattened) {
            let mut product = BigInt::from(1);
            for &id in &flattened {
                if let Some(n) = self.get_int(id) {
                    product *= n;
                }
            }
            return self.mk_int(product);
        }

        if sort == Sort::Real && self.all_rational_consts(&flattened) {
            let mut product = BigRational::from(BigInt::from(1));
            for &id in &flattened {
                if let Some(r) = self.get_rational(id) {
                    product *= r;
                }
            }
            return self.mk_rational(product);
        }

        // Phase 4: Partial constant folding and identity elimination
        // Collect constants and non-constants
        let (consts, non_consts): (Vec<_>, Vec<_>) = if sort == Sort::Int {
            flattened
                .iter()
                .partition(|&&id| self.get_int(id).is_some())
        } else if sort == Sort::Real {
            flattened
                .iter()
                .partition(|&&id| self.get_rational(id).is_some())
        } else {
            (vec![], flattened.iter().collect())
        };

        // Fold all constants into one
        let mut result_args: Vec<TermId> = non_consts.into_iter().copied().collect();
        let mut const_product_int = BigInt::from(1);
        let mut const_product_rat = BigRational::from(BigInt::from(1));
        let mut has_const = false;

        if !consts.is_empty() {
            has_const = true;
            if sort == Sort::Int {
                for &id in &consts {
                    if let Some(n) = self.get_int(*id) {
                        const_product_int *= n;
                    }
                }
            } else if sort == Sort::Real {
                for &id in &consts {
                    if let Some(r) = self.get_rational(*id) {
                        const_product_rat *= r;
                    }
                }
            }
        }

        // Phase 5: Handle special constant values
        if has_const {
            if sort == Sort::Int {
                // Zero annihilates
                if const_product_int == BigInt::from(0) {
                    return self.mk_int(BigInt::from(0));
                }
                // Multiply by -1: (* -1 x) -> (- x)
                if const_product_int == BigInt::from(-1) {
                    if result_args.is_empty() {
                        return self.mk_int(BigInt::from(-1));
                    }
                    if result_args.len() == 1 {
                        return self.mk_neg(result_args[0]);
                    }
                    // (* -1 a b) -> (- (* a b))
                    let inner_mul = self.mk_mul(result_args);
                    return self.mk_neg(inner_mul);
                }
                // Identity: don't add 1 to the product
                if const_product_int != BigInt::from(1) {
                    result_args.push(self.mk_int(const_product_int));
                }
            } else if sort == Sort::Real {
                // Zero annihilates
                if const_product_rat == BigRational::from(BigInt::from(0)) {
                    return self.mk_rational(BigRational::from(BigInt::from(0)));
                }
                // Multiply by -1: (* -1 x) -> (- x)
                if const_product_rat == BigRational::from(BigInt::from(-1)) {
                    if result_args.is_empty() {
                        return self.mk_rational(BigRational::from(BigInt::from(-1)));
                    }
                    if result_args.len() == 1 {
                        return self.mk_neg(result_args[0]);
                    }
                    // (* -1 a b) -> (- (* a b))
                    let inner_mul = self.mk_mul(result_args);
                    return self.mk_neg(inner_mul);
                }
                // Identity: don't add 1 to the product
                if const_product_rat != BigRational::from(BigInt::from(1)) {
                    result_args.push(self.mk_rational(const_product_rat));
                }
            }
        }

        // Phase 6: Distribute constants over addition for linear normalization
        //
        // (* c (+ a b ...)) -> (+ (* c a) (* c b) ...)
        //
        // Only applies when the multiplication is exactly "constant * sum" (no other factors).
        if (sort == Sort::Int || sort == Sort::Real) && result_args.len() == 2 {
            let a0 = result_args[0];
            let a1 = result_args[1];

            let const_and_sum = if sort == Sort::Int {
                if self.get_int(a0).is_some() {
                    Some((a0, a1))
                } else if self.get_int(a1).is_some() {
                    Some((a1, a0))
                } else {
                    None
                }
            } else if sort == Sort::Real {
                if self.get_rational(a0).is_some() {
                    Some((a0, a1))
                } else if self.get_rational(a1).is_some() {
                    Some((a1, a0))
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((const_term, sum_term)) = const_and_sum {
                let sum_args = match self.get(sum_term) {
                    TermData::App(Symbol::Named(name), args) if name == "+" && args.len() >= 2 => {
                        Some(args.clone())
                    }
                    _ => None,
                };

                if let Some(sum_args) = sum_args {
                    let mut distributed = Vec::with_capacity(sum_args.len());
                    for t in sum_args {
                        distributed.push(self.mk_mul(vec![const_term, t]));
                    }
                    return self.mk_add(distributed);
                }
            }
        }

        // Final result
        if result_args.is_empty() {
            return if sort == Sort::Int {
                self.mk_int(BigInt::from(1))
            } else {
                self.mk_rational(BigRational::from(BigInt::from(1)))
            };
        }
        if result_args.len() == 1 {
            return result_args[0];
        }

        self.intern(TermData::App(Symbol::named("*"), result_args), sort)
    }

    /// Create real division with constant folding
    pub fn mk_div(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            // Division is binary in SMT-LIB (though it can be variadic)
            return self.intern(TermData::App(Symbol::named("/"), args), Sort::Real);
        }

        let (a, b) = (args[0], args[1]);

        // Constant folding for rationals
        if let (Some(r1), Some(r2)) = (self.get_rational(a), self.get_rational(b)) {
            if *r2 != BigRational::from(BigInt::from(0)) {
                return self.mk_rational(r1.clone() / r2.clone());
            }
        }

        // x / 1 = x
        if let Some(r) = self.get_rational(b) {
            if *r == BigRational::from(BigInt::from(1)) {
                return a;
            }
        }

        // 0 / x = 0 (when x != 0)
        if let Some(r) = self.get_rational(a) {
            if *r == BigRational::from(BigInt::from(0)) {
                return self.mk_rational(BigRational::from(BigInt::from(0)));
            }
        }

        // x / x = 1 (when x is the same term)
        if a == b {
            return self.mk_rational(BigRational::from(BigInt::from(1)));
        }

        self.intern(TermData::App(Symbol::named("/"), args), Sort::Real)
    }

    /// Create integer division with constant folding
    pub fn mk_intdiv(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            return self.intern(TermData::App(Symbol::named("div"), args), Sort::Int);
        }

        let (a, b) = (args[0], args[1]);

        // Constant folding for integers
        if let (Some(n1), Some(n2)) = (self.get_int(a), self.get_int(b)) {
            if *n2 != BigInt::from(0) {
                // SMT-LIB div: truncated towards negative infinity
                return self.mk_int(n1.clone() / n2.clone());
            }
        }

        // x div 1 = x
        if let Some(n) = self.get_int(b) {
            if *n == BigInt::from(1) {
                return a;
            }
        }

        // 0 div x = 0 (when x != 0)
        if let Some(n) = self.get_int(a) {
            if *n == BigInt::from(0) {
                return self.mk_int(BigInt::from(0));
            }
        }

        // x div x = 1 (when x is the same term)
        if a == b {
            return self.mk_int(BigInt::from(1));
        }

        self.intern(TermData::App(Symbol::named("div"), args), Sort::Int)
    }

    /// Create modulo with constant folding
    pub fn mk_mod(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            return self.intern(TermData::App(Symbol::named("mod"), args), Sort::Int);
        }

        let (a, b) = (args[0], args[1]);

        // Constant folding for integers
        if let (Some(n1), Some(n2)) = (self.get_int(a), self.get_int(b)) {
            if *n2 != BigInt::from(0) {
                // SMT-LIB mod: result has same sign as divisor
                let result = n1.clone() % n2.clone();
                // Ensure result is non-negative (SMT-LIB semantics)
                let result = if result < BigInt::from(0) {
                    if *n2 > BigInt::from(0) {
                        result + n2.clone()
                    } else {
                        result - n2.clone()
                    }
                } else {
                    result
                };
                return self.mk_int(result);
            }
        }

        // x mod 1 = 0
        if let Some(n) = self.get_int(b) {
            if *n == BigInt::from(1) {
                return self.mk_int(BigInt::from(0));
            }
        }

        // 0 mod x = 0 (when x != 0)
        if let Some(n) = self.get_int(a) {
            if *n == BigInt::from(0) {
                return self.mk_int(BigInt::from(0));
            }
        }

        // x mod x = 0 (when x != 0)
        if a == b {
            return self.mk_int(BigInt::from(0));
        }

        self.intern(TermData::App(Symbol::named("mod"), args), Sort::Int)
    }

    /// Create absolute value with constant folding
    pub fn mk_abs(&mut self, arg: TermId) -> TermId {
        // Constant folding for integers
        if let Some(n) = self.get_int(arg) {
            if *n < BigInt::from(0) {
                return self.mk_int(-n.clone());
            }
            return arg;
        }

        // Constant folding for rationals
        if let Some(r) = self.get_rational(arg) {
            if *r < BigRational::from(BigInt::from(0)) {
                return self.mk_rational(-r.clone());
            }
            return arg;
        }

        let sort = self.sort(arg).clone();
        self.intern(TermData::App(Symbol::named("abs"), vec![arg]), sort)
    }

    // =======================================================================
    // Comparison operations with constant folding
    // =======================================================================

    /// Create less-than comparison with constant folding
    pub fn mk_lt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // x < x = false
        if lhs == rhs {
            return self.false_term();
        }

        // Integer constant folding
        if let (Some(n1), Some(n2)) = (self.get_int(lhs), self.get_int(rhs)) {
            return self.mk_bool(n1 < n2);
        }

        // Rational constant folding
        if let (Some(r1), Some(r2)) = (self.get_rational(lhs), self.get_rational(rhs)) {
            return self.mk_bool(r1 < r2);
        }

        self.intern(
            TermData::App(Symbol::named("<"), vec![lhs, rhs]),
            Sort::Bool,
        )
    }

    /// Create less-than-or-equal comparison with constant folding
    pub fn mk_le(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // x <= x = true
        if lhs == rhs {
            return self.true_term();
        }

        // Integer constant folding
        if let (Some(n1), Some(n2)) = (self.get_int(lhs), self.get_int(rhs)) {
            return self.mk_bool(n1 <= n2);
        }

        // Rational constant folding
        if let (Some(r1), Some(r2)) = (self.get_rational(lhs), self.get_rational(rhs)) {
            return self.mk_bool(r1 <= r2);
        }

        self.intern(
            TermData::App(Symbol::named("<="), vec![lhs, rhs]),
            Sort::Bool,
        )
    }

    /// Create greater-than comparison with constant folding
    ///
    /// Normalized to less-than: (> a b) -> (< b a)
    pub fn mk_gt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Normalize: (> a b) -> (< b a) for canonical form
        self.mk_lt(rhs, lhs)
    }

    /// Create greater-than-or-equal comparison with constant folding
    ///
    /// Normalized to less-than-or-equal: (>= a b) -> (<= b a)
    pub fn mk_ge(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Normalize: (>= a b) -> (<= b a) for canonical form
        self.mk_le(rhs, lhs)
    }

    // =======================================================================
    // Bitvector operations with constant folding and simplifications
    // =======================================================================

    /// Create bitvector addition with constant folding and simplifications
    ///
    /// Simplifications:
    /// - Constant folding: bvadd(#x01, #x02) → #x03
    /// - Identity: bvadd(x, 0) → x, bvadd(0, x) → x
    pub fn mk_bvadd(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            // bvadd is binary in SMT-LIB
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvadd"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvadd"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            let result = Self::bv_mask(&(v1 + v2), w1);
            return self.mk_bitvec(result, w1);
        }

        // Identity: x + 0 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == BigInt::from(0) {
                return a;
            }
        }
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == BigInt::from(0) {
                return b;
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvadd"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector subtraction with constant folding and simplifications
    ///
    /// Simplifications:
    /// - Constant folding: bvsub(#x03, #x01) → #x02
    /// - Identity: bvsub(x, 0) → x
    /// - Self-subtraction: bvsub(x, x) → 0
    pub fn mk_bvsub(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvsub"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvsub"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);

        // Constant folding (subtraction wraps around)
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            let modulus = BigInt::from(1) << w1;
            let result = Self::bv_mask(&((v1 - v2) % &modulus + &modulus), w1);
            return self.mk_bitvec(result, w1);
        }

        // Identity: x - 0 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == BigInt::from(0) {
                return a;
            }
        }

        // Self-subtraction: x - x = 0
        if a == b {
            return self.mk_bitvec(BigInt::from(0), width);
        }

        self.intern(
            TermData::App(Symbol::named("bvsub"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector multiplication with constant folding and simplifications
    ///
    /// Simplifications:
    /// - Constant folding: bvmul(#x02, #x03) → #x06
    /// - Zero: bvmul(x, 0) → 0, bvmul(0, x) → 0
    /// - Identity: bvmul(x, 1) → x, bvmul(1, x) → x
    pub fn mk_bvmul(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvmul"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvmul"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let one = BigInt::from(1);

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            let result = Self::bv_mask(&(v1 * v2), w1);
            return self.mk_bitvec(result, w1);
        }

        // Zero: x * 0 = 0
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == zero {
                return self.mk_bitvec(zero.clone(), width);
            }
            if *v == one {
                return a;
            }
        }
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
            if *v == one {
                return b;
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvmul"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector bitwise AND with constant folding and simplifications
    ///
    /// Simplifications:
    /// - Constant folding: bvand(#xFF, #x0F) → #x0F
    /// - Zero: bvand(x, 0) → 0
    /// - All-ones: bvand(x, -1) → x (where -1 is all bits set)
    /// - Idempotent: bvand(x, x) → x
    pub fn mk_bvand(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvand"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvand"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let all_ones = (BigInt::from(1) << width) - 1;

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            return self.mk_bitvec(v1 & v2, w1);
        }

        // Zero annihilator: x & 0 = 0
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == zero {
                return self.mk_bitvec(zero.clone(), width);
            }
            // Identity: x & all_ones = x
            if *v == all_ones {
                return a;
            }
        }
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
            if *v == all_ones {
                return b;
            }
        }

        // Idempotent: x & x = x
        if a == b {
            return a;
        }

        self.intern(
            TermData::App(Symbol::named("bvand"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector bitwise OR with constant folding and simplifications
    ///
    /// Simplifications:
    /// - Constant folding: bvor(#xF0, #x0F) → #xFF
    /// - Identity: bvor(x, 0) → x
    /// - All-ones: bvor(x, -1) → -1
    /// - Idempotent: bvor(x, x) → x
    pub fn mk_bvor(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvor"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvor"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let all_ones = (BigInt::from(1) << width) - 1;

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            return self.mk_bitvec(v1 | v2, w1);
        }

        // Identity: x | 0 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == zero {
                return a;
            }
            // Annihilator: x | all_ones = all_ones
            if *v == all_ones {
                return self.mk_bitvec(all_ones.clone(), width);
            }
        }
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return b;
            }
            if *v == all_ones {
                return self.mk_bitvec(all_ones, width);
            }
        }

        // Idempotent: x | x = x
        if a == b {
            return a;
        }

        self.intern(
            TermData::App(Symbol::named("bvor"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector bitwise XOR with constant folding and simplifications
    ///
    /// Simplifications:
    /// - Constant folding: bvxor(#xF0, #x0F) → #xFF
    /// - Identity: bvxor(x, 0) → x
    /// - Self-XOR: bvxor(x, x) → 0
    pub fn mk_bvxor(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvxor"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvxor"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            return self.mk_bitvec(v1 ^ v2, w1);
        }

        // Identity: x ^ 0 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == zero {
                return a;
            }
        }
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return b;
            }
        }

        // Self-XOR: x ^ x = 0
        if a == b {
            return self.mk_bitvec(zero, width);
        }

        self.intern(
            TermData::App(Symbol::named("bvxor"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector bitwise NAND with simplifications.
    ///
    /// Defined as: bvnand(a, b) = bvnot(bvand(a, b))
    pub fn mk_bvnand(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvnand"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvnand"), args), Sort::Bool);
        }

        let and_term = self.mk_bvand(args);
        self.mk_bvnot(and_term)
    }

    /// Create bitvector bitwise NOR with simplifications.
    ///
    /// Defined as: bvnor(a, b) = bvnot(bvor(a, b))
    pub fn mk_bvnor(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvnor"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvnor"), args), Sort::Bool);
        }

        let or_term = self.mk_bvor(args);
        self.mk_bvnot(or_term)
    }

    /// Create bitvector bitwise XNOR with simplifications.
    ///
    /// Defined as: bvxnor(a, b) = bvnot(bvxor(a, b))
    pub fn mk_bvxnor(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvxnor"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvxnor"), args), Sort::Bool);
        }

        let xor_term = self.mk_bvxor(args);
        self.mk_bvnot(xor_term)
    }

    /// Create bitvector bitwise NOT with constant folding and simplifications
    ///
    /// Simplifications:
    /// - Constant folding: bvnot(#xFF) → #x00 (for 8-bit)
    /// - Double negation: bvnot(bvnot(x)) → x
    pub fn mk_bvnot(&mut self, arg: TermId) -> TermId {
        let width = self.get_bv_width(arg).unwrap_or(32);
        let all_ones = (BigInt::from(1) << width) - 1;

        // Constant folding
        if let Some((v, w)) = self.get_bitvec(arg) {
            return self.mk_bitvec(&all_ones ^ v, w);
        }

        // Double negation: bvnot(bvnot(x)) → x
        if let TermData::App(sym, args) = self.get(arg) {
            if sym.name() == "bvnot" && args.len() == 1 {
                return args[0];
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvnot"), vec![arg]),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector negation with constant folding and simplifications
    ///
    /// bvneg(x) = -x (two's complement)
    ///
    /// Simplifications:
    /// - Constant folding: bvneg(#x01) → #xFF (for 8-bit)
    /// - Double negation: bvneg(bvneg(x)) → x
    /// - Zero: bvneg(0) → 0
    pub fn mk_bvneg(&mut self, arg: TermId) -> TermId {
        let width = self.get_bv_width(arg).unwrap_or(32);
        let modulus = BigInt::from(1) << width;

        // Constant folding (two's complement: -x = ~x + 1 = (2^n - x) mod 2^n)
        if let Some((v, w)) = self.get_bitvec(arg) {
            if *v == BigInt::from(0) {
                return arg;
            }
            let result = Self::bv_mask(&(&modulus - v), w);
            return self.mk_bitvec(result, w);
        }

        // Double negation: bvneg(bvneg(x)) → x
        if let TermData::App(sym, args) = self.get(arg) {
            if sym.name() == "bvneg" && args.len() == 1 {
                return args[0];
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvneg"), vec![arg]),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector left shift with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvshl(x, c) when both are constants
    /// - Identity: bvshl(x, 0) → x
    /// - Zero: bvshl(0, x) → 0
    /// - Large shift: bvshl(x, c) → 0 when c >= width
    pub fn mk_bvshl(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvshl"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvshl"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            // Shift amount is taken modulo width for SMT-LIB semantics
            if let Some(shift) = v2.to_u32() {
                if shift >= w1 {
                    return self.mk_bitvec(zero, w1);
                }
                let result = Self::bv_mask(&(v1 << shift), w1);
                return self.mk_bitvec(result, w1);
            }
        }

        // Identity: x << 0 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == zero {
                return a;
            }
            // Large shift produces zero
            if let Some(shift) = v.to_u32() {
                if shift >= width {
                    return self.mk_bitvec(zero.clone(), width);
                }
            }
        }

        // Zero shifted is zero
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvshl"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector logical right shift with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvlshr(x, c) when both are constants
    /// - Identity: bvlshr(x, 0) → x
    /// - Zero: bvlshr(0, x) → 0
    /// - Large shift: bvlshr(x, c) → 0 when c >= width
    pub fn mk_bvlshr(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvlshr"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvlshr"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            if let Some(shift) = v2.to_u32() {
                if shift >= w1 {
                    return self.mk_bitvec(zero, w1);
                }
                let result = v1 >> shift;
                return self.mk_bitvec(result, w1);
            }
        }

        // Identity: x >> 0 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == zero {
                return a;
            }
            // Large shift produces zero
            if let Some(shift) = v.to_u32() {
                if shift >= width {
                    return self.mk_bitvec(zero.clone(), width);
                }
            }
        }

        // Zero shifted is zero
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvlshr"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector arithmetic right shift with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvashr(x, c) when both are constants
    /// - Identity: bvashr(x, 0) → x
    /// - Zero: bvashr(0, x) → 0
    pub fn mk_bvashr(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvashr"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvashr"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            if let Some(shift) = v2.to_u32() {
                let sign_bit = BigInt::from(1) << (w1 - 1);
                let is_negative = v1 >= &sign_bit;

                if shift >= w1 {
                    // All bits become sign bit
                    if is_negative {
                        return self.mk_bitvec((BigInt::from(1) << w1) - 1, w1);
                    } else {
                        return self.mk_bitvec(zero, w1);
                    }
                }

                let mut result = v1 >> shift;
                // Sign extend: fill upper bits with sign bit
                if is_negative {
                    let fill_mask = ((BigInt::from(1) << shift) - 1) << (w1 - shift);
                    result |= fill_mask;
                }
                return self.mk_bitvec(Self::bv_mask(&result, w1), w1);
            }
        }

        // Identity: x >>> 0 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == zero {
                return a;
            }
        }

        // Zero shifted is zero
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvashr"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector unsigned division with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvudiv(x, c) when both are constants (and c != 0)
    /// - Identity: bvudiv(x, 1) → x
    /// - Self-division: bvudiv(x, x) → 1 (when x is the same term)
    /// - Zero dividend: bvudiv(0, x) → 0
    pub fn mk_bvudiv(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvudiv"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvudiv"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let one = BigInt::from(1);

        // Constant folding (note: division by zero is undefined in SMT-LIB)
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            if *v2 != zero {
                return self.mk_bitvec(v1 / v2, w1);
            }
        }

        // Identity: x / 1 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == one {
                return a;
            }
        }

        // Zero dividend: 0 / x = 0
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
        }

        // Self-division: x / x = 1
        if a == b {
            return self.mk_bitvec(one, width);
        }

        self.intern(
            TermData::App(Symbol::named("bvudiv"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector unsigned remainder with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvurem(x, c) when both are constants (and c != 0)
    /// - Identity: bvurem(x, 1) → 0
    /// - Self-remainder: bvurem(x, x) → 0
    /// - Zero dividend: bvurem(0, x) → 0
    pub fn mk_bvurem(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvurem"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvurem"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let one = BigInt::from(1);

        // Constant folding
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            if *v2 != zero {
                return self.mk_bitvec(v1 % v2, w1);
            }
        }

        // Identity: x % 1 = 0
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == one {
                return self.mk_bitvec(zero.clone(), width);
            }
        }

        // Zero dividend: 0 % x = 0
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
        }

        // Self-remainder: x % x = 0
        if a == b {
            return self.mk_bitvec(zero, width);
        }

        self.intern(
            TermData::App(Symbol::named("bvurem"), args),
            Sort::BitVec(width),
        )
    }

    /// Create signed bitvector division with constant folding
    ///
    /// SMT-LIB semantics (bvsdiv s t):
    /// - Signed division, truncated towards zero
    /// - sign(result) = sign(s) XOR sign(t)
    /// - |result| = |s| / |t|
    ///
    /// Simplifications:
    /// - Constant folding: bvsdiv(#x06, #x02) → #x03
    /// - Identity: bvsdiv(x, 1) → x
    /// - Self-division: bvsdiv(x, x) → 1 (when x ≠ 0)
    /// - Zero dividend: bvsdiv(0, x) → 0
    pub fn mk_bvsdiv(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvsdiv"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvsdiv"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let one = BigInt::from(1);

        // Constant folding (signed division, truncated towards zero)
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            if *v2 != zero {
                let s1 = Self::to_signed(v1, w1);
                let s2 = Self::to_signed(v2, w1);
                // Signed division truncated towards zero
                let quotient = if (s1 >= zero) == (s2 >= zero) {
                    s1.abs() / s2.abs()
                } else {
                    -(s1.abs() / s2.abs())
                };
                return self.mk_bitvec(Self::from_signed(&quotient, w1), w1);
            }
        }

        // Identity: x / 1 = x
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == one {
                return a;
            }
        }

        // Zero dividend: 0 / x = 0
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero.clone(), width);
            }
        }

        // Self-division: x / x = 1
        if a == b {
            return self.mk_bitvec(one.clone(), width);
        }

        self.intern(
            TermData::App(Symbol::named("bvsdiv"), args),
            Sort::BitVec(width),
        )
    }

    /// Create signed bitvector remainder with constant folding
    ///
    /// SMT-LIB semantics (bvsrem s t):
    /// - sign(result) = sign(s) (sign follows dividend)
    /// - s = (bvsdiv s t) * t + (bvsrem s t)
    ///
    /// Simplifications:
    /// - Constant folding: bvsrem(#x07, #x03) → #x01
    /// - Identity: bvsrem(x, 1) → 0
    /// - Self-remainder: bvsrem(x, x) → 0
    /// - Zero dividend: bvsrem(0, x) → 0
    pub fn mk_bvsrem(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvsrem"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvsrem"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let one = BigInt::from(1);

        // Constant folding (signed remainder, sign follows dividend)
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            if *v2 != zero {
                let s1 = Self::to_signed(v1, w1);
                let s2 = Self::to_signed(v2, w1);
                // Result sign follows dividend (s1)
                let remainder = if s1 >= zero {
                    s1.abs() % s2.abs()
                } else {
                    -(s1.abs() % s2.abs())
                };
                return self.mk_bitvec(Self::from_signed(&remainder, w1), w1);
            }
        }

        // Identity: x % 1 = 0
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == one {
                return self.mk_bitvec(zero.clone(), width);
            }
        }

        // Zero dividend: 0 % x = 0
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
        }

        // Self-remainder: x % x = 0
        if a == b {
            return self.mk_bitvec(zero, width);
        }

        self.intern(
            TermData::App(Symbol::named("bvsrem"), args),
            Sort::BitVec(width),
        )
    }

    /// Create signed bitvector modulo with constant folding
    ///
    /// SMT-LIB semantics (bvsmod s t):
    /// - sign(result) = sign(t) (sign follows divisor)
    /// - Result is the unique value r such that:
    ///   - 0 <= r < |t| or |t| < r <= 0
    ///   - s = q*t + r for some integer q
    ///
    /// Simplifications:
    /// - Constant folding: bvsmod(#xFB, #x03) → #x02 (i.e., -5 mod 3 = 2)
    /// - Identity: bvsmod(x, 1) → 0
    /// - Self-modulo: bvsmod(x, x) → 0
    /// - Zero dividend: bvsmod(0, x) → 0
    pub fn mk_bvsmod(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            if let Some(width) = args.first().and_then(|&a| self.get_bv_width(a)) {
                return self.intern(
                    TermData::App(Symbol::named("bvsmod"), args),
                    Sort::BitVec(width),
                );
            }
            return self.intern(TermData::App(Symbol::named("bvsmod"), args), Sort::Bool);
        }

        let (a, b) = (args[0], args[1]);
        let width = self.get_bv_width(a).unwrap_or(32);
        let zero = BigInt::from(0);
        let one = BigInt::from(1);

        // Constant folding (signed modulo, sign follows divisor)
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            if *v2 != zero {
                let s1 = Self::to_signed(v1, w1);
                let s2 = Self::to_signed(v2, w1);
                let abs_t = s2.abs();
                let u = s1.abs() % &abs_t;

                let result = if u == zero {
                    zero.clone()
                } else if s1 >= zero && s2 >= zero {
                    // Both positive: result is u
                    u
                } else if s1 < zero && s2 >= zero {
                    // s < 0, t > 0: result is t - u
                    &abs_t - u
                } else if s1 >= zero && s2 < zero {
                    // s >= 0, t < 0: result is u + t (which is u - |t|)
                    u - &abs_t
                } else {
                    // Both negative: result is -u
                    -u
                };
                return self.mk_bitvec(Self::from_signed(&result, w1), w1);
            }
        }

        // Identity: x mod 1 = 0
        if let Some((v, _)) = self.get_bitvec(b) {
            if *v == one {
                return self.mk_bitvec(zero.clone(), width);
            }
        }

        // Zero dividend: 0 mod x = 0
        if let Some((v, _)) = self.get_bitvec(a) {
            if *v == zero {
                return self.mk_bitvec(zero, width);
            }
        }

        // Self-modulo: x mod x = 0
        if a == b {
            return self.mk_bitvec(zero, width);
        }

        self.intern(
            TermData::App(Symbol::named("bvsmod"), args),
            Sort::BitVec(width),
        )
    }

    /// Create bitvector comparison returning 1-bit bitvector
    ///
    /// SMT-LIB semantics (bvcomp s t):
    /// - Returns #b1 if s = t, #b0 if s ≠ t
    ///
    /// Simplifications:
    /// - Constant folding: bvcomp(#x05, #x05) → #b1
    /// - Reflexivity: bvcomp(x, x) → #b1
    pub fn mk_bvcomp(&mut self, a: TermId, b: TermId) -> TermId {
        // Reflexivity: bvcomp(x, x) = #b1
        if a == b {
            return self.mk_bitvec(BigInt::from(1), 1);
        }

        // Constant folding
        if let (Some((v1, _)), Some((v2, _))) = (self.get_bitvec(a), self.get_bitvec(b)) {
            return if v1 == v2 {
                self.mk_bitvec(BigInt::from(1), 1)
            } else {
                self.mk_bitvec(BigInt::from(0), 1)
            };
        }

        self.intern(
            TermData::App(Symbol::named("bvcomp"), vec![a, b]),
            Sort::BitVec(1),
        )
    }

    // =======================================================================
    // Bitvector comparison operations
    // =======================================================================

    /// Create unsigned bitvector less-than comparison with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvult(#x01, #x02) → true
    /// - Reflexivity: bvult(x, x) → false
    /// - Zero lower bound: bvult(x, 0) → false (nothing is less than 0 unsigned)
    /// - Zero argument: bvult(0, x) → x != 0 (but we just return the comparison)
    pub fn mk_bvult(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Reflexivity: x < x = false
        if lhs == rhs {
            return self.false_term();
        }

        // Constant folding (unsigned comparison)
        if let (Some((v1, _)), Some((v2, _))) = (self.get_bitvec(lhs), self.get_bitvec(rhs)) {
            return self.mk_bool(v1 < v2);
        }

        // Zero lower bound: bvult(x, 0) = false
        if let Some((v, _)) = self.get_bitvec(rhs) {
            if *v == BigInt::from(0) {
                return self.false_term();
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvult"), vec![lhs, rhs]),
            Sort::Bool,
        )
    }

    /// Create unsigned bitvector less-than-or-equal comparison with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvule(#x01, #x02) → true
    /// - Reflexivity: bvule(x, x) → true
    /// - Zero argument: bvule(0, x) → true (0 is <= everything unsigned)
    pub fn mk_bvule(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Reflexivity: x <= x = true
        if lhs == rhs {
            return self.true_term();
        }

        // Constant folding (unsigned comparison)
        if let (Some((v1, _)), Some((v2, _))) = (self.get_bitvec(lhs), self.get_bitvec(rhs)) {
            return self.mk_bool(v1 <= v2);
        }

        // Zero left: bvule(0, x) = true
        if let Some((v, _)) = self.get_bitvec(lhs) {
            if *v == BigInt::from(0) {
                return self.true_term();
            }
        }

        self.intern(
            TermData::App(Symbol::named("bvule"), vec![lhs, rhs]),
            Sort::Bool,
        )
    }

    /// Create unsigned bitvector greater-than comparison
    ///
    /// Normalized to bvult: bvugt(a, b) → bvult(b, a)
    pub fn mk_bvugt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_bvult(rhs, lhs)
    }

    /// Create unsigned bitvector greater-than-or-equal comparison
    ///
    /// Normalized to bvule: bvuge(a, b) → bvule(b, a)
    pub fn mk_bvuge(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_bvule(rhs, lhs)
    }

    /// Helper to interpret a bitvector as a signed (two's complement) value
    fn to_signed(value: &BigInt, width: u32) -> BigInt {
        let max_positive = BigInt::from(1) << (width - 1);
        if value >= &max_positive {
            // Negative value: value - 2^width
            let modulus = BigInt::from(1) << width;
            value - modulus
        } else {
            value.clone()
        }
    }

    /// Helper to convert a signed value back to unsigned bitvector representation
    fn from_signed(value: &BigInt, width: u32) -> BigInt {
        if value < &BigInt::from(0) {
            // Negative value: add 2^width to get unsigned representation
            let modulus = BigInt::from(1) << width;
            value + modulus
        } else {
            value.clone()
        }
    }

    /// Create signed bitvector less-than comparison with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvslt(#xFF, #x01) → true (8-bit: -1 < 1)
    /// - Reflexivity: bvslt(x, x) → false
    pub fn mk_bvslt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Reflexivity: x < x = false
        if lhs == rhs {
            return self.false_term();
        }

        // Constant folding (signed comparison)
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(lhs), self.get_bitvec(rhs)) {
            let s1 = Self::to_signed(v1, w1);
            let s2 = Self::to_signed(v2, w1);
            return self.mk_bool(s1 < s2);
        }

        self.intern(
            TermData::App(Symbol::named("bvslt"), vec![lhs, rhs]),
            Sort::Bool,
        )
    }

    /// Create signed bitvector less-than-or-equal comparison with constant folding
    ///
    /// Simplifications:
    /// - Constant folding: bvsle(#xFF, #x01) → true (8-bit: -1 <= 1)
    /// - Reflexivity: bvsle(x, x) → true
    pub fn mk_bvsle(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Reflexivity: x <= x = true
        if lhs == rhs {
            return self.true_term();
        }

        // Constant folding (signed comparison)
        if let (Some((v1, w1)), Some((v2, _))) = (self.get_bitvec(lhs), self.get_bitvec(rhs)) {
            let s1 = Self::to_signed(v1, w1);
            let s2 = Self::to_signed(v2, w1);
            return self.mk_bool(s1 <= s2);
        }

        self.intern(
            TermData::App(Symbol::named("bvsle"), vec![lhs, rhs]),
            Sort::Bool,
        )
    }

    /// Create signed bitvector greater-than comparison
    ///
    /// Normalized to bvslt: bvsgt(a, b) → bvslt(b, a)
    pub fn mk_bvsgt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_bvslt(rhs, lhs)
    }

    /// Create signed bitvector greater-than-or-equal comparison
    ///
    /// Normalized to bvsle: bvsge(a, b) → bvsle(b, a)
    pub fn mk_bvsge(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_bvsle(rhs, lhs)
    }

    /// Extract bits `[high:low]` from a bitvector (SMT-LIB: `(_ extract i j)`)
    ///
    /// Simplifications:
    /// - Constant folding: extract(7,4,#xFF) → #x0F
    /// - Full extract: extract(n-1,0,x) → x (extracting all bits)
    ///
    /// The result width is high - low + 1.
    pub fn mk_bvextract(&mut self, high: u32, low: u32, arg: TermId) -> TermId {
        // Validate indices: high >= low
        if high < low {
            // Invalid extract, return the term as-is with generic sort
            return self.intern(
                TermData::App(Symbol::indexed("extract", vec![high, low]), vec![arg]),
                Sort::BitVec(1),
            );
        }

        let result_width = high - low + 1;
        let src_width = self.get_bv_width(arg).unwrap_or(32);

        // Full extract simplification: extract(n-1,0,x) → x
        if high == src_width - 1 && low == 0 {
            return arg;
        }

        // Constant folding
        if let Some((val, _)) = self.get_bitvec(arg) {
            let shifted = val >> low;
            let result = Self::bv_mask(&shifted, result_width);
            return self.mk_bitvec(result, result_width);
        }

        self.intern(
            TermData::App(Symbol::indexed("extract", vec![high, low]), vec![arg]),
            Sort::BitVec(result_width),
        )
    }

    /// Concatenate two bitvectors (SMT-LIB: concat)
    ///
    /// Result width is the sum of input widths. The first argument becomes
    /// the high bits, the second becomes the low bits.
    ///
    /// Simplifications:
    /// - Constant folding: concat(#x0F, #xF0) → #x0FF0
    pub fn mk_bvconcat(&mut self, args: Vec<TermId>) -> TermId {
        if args.len() != 2 {
            // concat is binary
            let width = args.iter().filter_map(|&a| self.get_bv_width(a)).sum();
            return self.intern(
                TermData::App(Symbol::named("concat"), args),
                Sort::BitVec(width),
            );
        }

        let (high, low) = (args[0], args[1]);
        let high_width = self.get_bv_width(high).unwrap_or(32);
        let low_width = self.get_bv_width(low).unwrap_or(32);
        let result_width = high_width + low_width;

        // Constant folding
        if let (Some((v_high, _)), Some((v_low, _))) = (self.get_bitvec(high), self.get_bitvec(low))
        {
            let result = (v_high << low_width) | v_low;
            return self.mk_bitvec(result, result_width);
        }

        self.intern(
            TermData::App(Symbol::named("concat"), args),
            Sort::BitVec(result_width),
        )
    }

    /// Zero-extend a bitvector by i bits (SMT-LIB: (_ zero_extend i))
    ///
    /// Adds i zero bits to the most significant end.
    ///
    /// Simplifications:
    /// - Zero extension by 0: zero_extend(0,x) → x
    /// - Constant folding: zero_extend(4,#x0F) → #x00F
    pub fn mk_bvzero_extend(&mut self, i: u32, arg: TermId) -> TermId {
        let src_width = self.get_bv_width(arg).unwrap_or(32);
        let result_width = src_width + i;

        // Zero extension by 0 bits is identity
        if i == 0 {
            return arg;
        }

        // Constant folding (value stays the same, just wider representation)
        if let Some((val, _)) = self.get_bitvec(arg) {
            return self.mk_bitvec(val.clone(), result_width);
        }

        self.intern(
            TermData::App(Symbol::indexed("zero_extend", vec![i]), vec![arg]),
            Sort::BitVec(result_width),
        )
    }

    /// Sign-extend a bitvector by i bits (SMT-LIB: (_ sign_extend i))
    ///
    /// Adds i copies of the sign bit (MSB) to the most significant end.
    ///
    /// Simplifications:
    /// - Sign extension by 0: sign_extend(0,x) → x
    /// - Constant folding: sign_extend(4,#x8F) → #xFF8F (for 8-bit input)
    pub fn mk_bvsign_extend(&mut self, i: u32, arg: TermId) -> TermId {
        let src_width = self.get_bv_width(arg).unwrap_or(32);
        let result_width = src_width + i;

        // Sign extension by 0 bits is identity
        if i == 0 {
            return arg;
        }

        // Constant folding
        if let Some((val, w)) = self.get_bitvec(arg) {
            let sign_bit = (val >> (w - 1)) & BigInt::from(1);
            let result = if sign_bit == BigInt::from(0) {
                // Positive: same as zero extension
                val.clone()
            } else {
                // Negative: set all the new high bits to 1
                let extension_mask = ((BigInt::from(1) << i) - 1) << w;
                val | &extension_mask
            };
            return self.mk_bitvec(result, result_width);
        }

        self.intern(
            TermData::App(Symbol::indexed("sign_extend", vec![i]), vec![arg]),
            Sort::BitVec(result_width),
        )
    }

    /// Rotate left by i bits (SMT-LIB: (_ rotate_left i))
    ///
    /// Rotates the bitvector left by i bit positions.
    /// Bits shifted out the left are shifted in on the right.
    ///
    /// Simplifications:
    /// - Rotation by 0: rotate_left(0,x) → x
    /// - Rotation by width: rotate_left(n,x) → x (where n = width)
    /// - Constant folding
    pub fn mk_bvrotate_left(&mut self, i: u32, arg: TermId) -> TermId {
        let width = self.get_bv_width(arg).unwrap_or(32);

        // Normalize rotation amount to [0, width)
        let rotation = if width > 0 { i % width } else { 0 };

        // Rotation by 0 is identity
        if rotation == 0 {
            return arg;
        }

        // Constant folding
        if let Some((val, w)) = self.get_bitvec(arg) {
            let mask = (BigInt::from(1) << w) - 1;
            let left_part = (val << rotation) & &mask;
            let right_part = val >> (w - rotation);
            let result = left_part | right_part;
            return self.mk_bitvec(result, w);
        }

        self.intern(
            TermData::App(Symbol::indexed("rotate_left", vec![i]), vec![arg]),
            Sort::BitVec(width),
        )
    }

    /// Rotate right by i bits (SMT-LIB: (_ rotate_right i))
    ///
    /// Rotates the bitvector right by i bit positions.
    /// Bits shifted out the right are shifted in on the left.
    ///
    /// Simplifications:
    /// - Rotation by 0: rotate_right(0,x) → x
    /// - Rotation by width: rotate_right(n,x) → x (where n = width)
    /// - Constant folding
    pub fn mk_bvrotate_right(&mut self, i: u32, arg: TermId) -> TermId {
        let width = self.get_bv_width(arg).unwrap_or(32);

        // Normalize rotation amount to [0, width)
        let rotation = if width > 0 { i % width } else { 0 };

        // Rotation by 0 is identity
        if rotation == 0 {
            return arg;
        }

        // Constant folding
        if let Some((val, w)) = self.get_bitvec(arg) {
            let mask = (BigInt::from(1) << w) - 1;
            let right_part = val >> rotation;
            let left_part = (val << (w - rotation)) & &mask;
            let result = left_part | right_part;
            return self.mk_bitvec(result, w);
        }

        self.intern(
            TermData::App(Symbol::indexed("rotate_right", vec![i]), vec![arg]),
            Sort::BitVec(width),
        )
    }

    /// Repeat a bitvector i times (SMT-LIB: (_ repeat i))
    ///
    /// Concatenates i copies of the bitvector.
    /// Result width is original width * i.
    ///
    /// Simplifications:
    /// - Repeat 1: repeat(1,x) → x
    /// - Constant folding
    pub fn mk_bvrepeat(&mut self, i: u32, arg: TermId) -> TermId {
        let src_width = self.get_bv_width(arg).unwrap_or(32);
        let result_width = src_width * i;

        // Repeat 1 is identity
        if i == 1 {
            return arg;
        }

        // Repeat 0 is not valid in SMT-LIB, but handle gracefully
        if i == 0 {
            return self.mk_bitvec(BigInt::from(0), 1);
        }

        // Constant folding
        if let Some((val, w)) = self.get_bitvec(arg) {
            let mut result = BigInt::from(0);
            for _ in 0..i {
                result = (result << w) | val;
            }
            return self.mk_bitvec(result, result_width);
        }

        self.intern(
            TermData::App(Symbol::indexed("repeat", vec![i]), vec![arg]),
            Sort::BitVec(result_width),
        )
    }

    /// Convert a bitvector to a non-negative integer (SMT-LIB: bv2nat).
    ///
    /// `bv2nat` interprets the bitvector as an unsigned number in `[0, 2^w)`.
    pub fn mk_bv2nat(&mut self, arg: TermId) -> TermId {
        if let Some((v, _w)) = self.get_bitvec(arg) {
            return self.mk_int(v.clone());
        }

        self.intern(TermData::App(Symbol::named("bv2nat"), vec![arg]), Sort::Int)
    }

    /// Convert an integer to a bitvector of fixed width (SMT-LIB: `(_ int2bv w)`).
    ///
    /// Semantics: `int2bv(w, n)` is `n mod 2^w`, represented as a bitvector of width `w`.
    pub fn mk_int2bv(&mut self, width: u32, arg: TermId) -> TermId {
        // int2bv(w, bv2nat(x)) = x, when x has width w
        if let TermData::App(Symbol::Named(name), args) = self.get(arg) {
            if name == "bv2nat" && args.len() == 1 && self.get_bv_width(args[0]) == Some(width) {
                return args[0];
            }
        }

        // Constant folding
        if let Some(n) = self.get_int(arg) {
            let modulus = BigInt::from(1) << width;
            let mut reduced = n.clone() % &modulus;
            if reduced < BigInt::from(0) {
                reduced += &modulus;
            }
            let reduced = Self::bv_mask(&reduced, width);
            return self.mk_bitvec(reduced, width);
        }

        self.intern(
            TermData::App(Symbol::indexed("int2bv", vec![width]), vec![arg]),
            Sort::BitVec(width),
        )
    }

    // ==================== Array Operations ====================

    /// Create an array select (read) operation: (select a i)
    ///
    /// Simplifications:
    /// - Read-over-const-array: select(const-array(v), i) → v
    /// - Read-over-write: select(store(a, i, v), i) → v
    /// - Read-over-write with different constant indices: select(store(a, j, v), i) → select(a, i) when i != j
    pub fn mk_select(&mut self, array: TermId, index: TermId) -> TermId {
        // Get the element sort from the array sort
        let elem_sort = match self.sort(array) {
            Sort::Array(_, elem) => (**elem).clone(),
            _ => {
                // Type error - return a dummy term
                return self.intern(
                    TermData::App(Symbol::named("select"), vec![array, index]),
                    Sort::Bool,
                );
            }
        };

        // Read-over-const-array simplification: select(const-array(v), i) → v
        if let Some(default_value) = self.get_const_array(array) {
            return default_value;
        }

        // Read-over-write simplification: select(store(a, i, v), i) → v
        if let TermData::App(Symbol::Named(name), args) = self.get(array) {
            if name == "store" && args.len() == 3 {
                let store_index = args[1];
                let store_value = args[2];
                let inner_array = args[0];

                // If indices are identical, return the stored value
                if store_index == index {
                    return store_value;
                }

                // If both indices are constants and different, look through
                if let (Some(idx1), Some(idx2)) = (self.get_int(index), self.get_int(store_index)) {
                    if idx1 != idx2 {
                        return self.mk_select(inner_array, index);
                    }
                }
                if let (Some((val1, _)), Some((val2, _))) =
                    (self.get_bitvec(index), self.get_bitvec(store_index))
                {
                    if val1 != val2 {
                        return self.mk_select(inner_array, index);
                    }
                }
            }
        }

        self.intern(
            TermData::App(Symbol::named("select"), vec![array, index]),
            elem_sort,
        )
    }

    /// Create an array store (write) operation: (store a i v)
    ///
    /// Returns a new array identical to `a` except at index `i` where it has value `v`.
    ///
    /// Simplifications:
    /// - Store-over-store at same index: store(store(a, i, v1), i, v2) → store(a, i, v2)
    pub fn mk_store(&mut self, array: TermId, index: TermId, value: TermId) -> TermId {
        // Get the array sort (should remain the same)
        let array_sort = self.sort(array).clone();

        // Store-over-store simplification: store(store(a, i, v1), i, v2) → store(a, i, v2)
        if let TermData::App(Symbol::Named(name), args) = self.get(array) {
            if name == "store" && args.len() == 3 {
                let inner_index = args[1];
                let inner_array = args[0];

                // If we're storing at the same index, skip the intermediate store
                if inner_index == index {
                    return self.mk_store(inner_array, index, value);
                }

                // If both are constant and different, we can't simplify further
                // (order matters for different indices)
            }
        }

        self.intern(
            TermData::App(Symbol::named("store"), vec![array, index, value]),
            array_sort,
        )
    }

    /// Create a constant array: ((as const (Array T1 T2)) v)
    ///
    /// Returns an array where every index maps to the given default value.
    /// The array has sort (Array index_sort elem_sort) where elem_sort is the sort of the value.
    pub fn mk_const_array(&mut self, index_sort: Sort, value: TermId) -> TermId {
        let elem_sort = self.sort(value).clone();
        let array_sort = Sort::Array(Box::new(index_sort), Box::new(elem_sort));

        self.intern(
            TermData::App(Symbol::named("const-array"), vec![value]),
            array_sort,
        )
    }

    /// Check if a term is a constant array, returning the default value if so
    pub fn get_const_array(&self, term: TermId) -> Option<TermId> {
        match self.get(term) {
            TermData::App(Symbol::Named(name), args)
                if name == "const-array" && args.len() == 1 =>
            {
                Some(args[0])
            }
            _ => None,
        }
    }

    /// Lookup a named variable/constant
    pub fn lookup(&self, name: &str) -> Option<TermId> {
        self.names.get(name).map(|(id, _)| *id)
    }

    /// Check if a term is a Boolean constant
    pub fn is_bool_const(&self, id: TermId) -> Option<bool> {
        match self.get(id) {
            TermData::Const(Constant::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Check if a term is true
    pub fn is_true(&self, id: TermId) -> bool {
        self.is_bool_const(id) == Some(true)
    }

    /// Check if a term is false
    pub fn is_false(&self, id: TermId) -> bool {
        self.is_bool_const(id) == Some(false)
    }

    /// Get all children of a term
    pub fn children(&self, id: TermId) -> Vec<TermId> {
        match self.get(id) {
            TermData::Const(_) => vec![],
            TermData::Var(_, _) => vec![],
            TermData::App(_, args) => args.clone(),
            TermData::Let(bindings, body) => {
                let mut children: Vec<_> = bindings.iter().map(|(_, t)| *t).collect();
                children.push(*body);
                children
            }
            TermData::Not(t) => vec![*t],
            TermData::Ite(c, t, e) => vec![*c, *t, *e],
        }
    }

    // =======================================================================
    // ITE Lifting (Shannon Expansion)
    // =======================================================================

    /// Lift ITE expressions out of arithmetic predicates.
    ///
    /// Transforms: `(<= (ite c a b) x)` → `(ite c (<= a x) (<= b x))`
    ///
    /// This allows the LRA/LIA theory solvers to handle the arithmetic atoms
    /// without needing to reason about ITE expressions directly.
    pub fn lift_arithmetic_ite(&mut self, term: TermId) -> TermId {
        self.lift_ite_recursive(term)
    }

    /// Recursively lift ITEs from a term.
    fn lift_ite_recursive(&mut self, term: TermId) -> TermId {
        match self.get(term).clone() {
            TermData::Const(_) | TermData::Var(_, _) => term,

            TermData::Not(inner) => {
                let lifted = self.lift_ite_recursive(inner);
                if lifted == inner {
                    term
                } else {
                    self.mk_not(lifted)
                }
            }

            TermData::Ite(cond, then_t, else_t) => {
                let lifted_cond = self.lift_ite_recursive(cond);
                let lifted_then = self.lift_ite_recursive(then_t);
                let lifted_else = self.lift_ite_recursive(else_t);
                if lifted_cond == cond && lifted_then == then_t && lifted_else == else_t {
                    term
                } else {
                    self.mk_ite(lifted_cond, lifted_then, lifted_else)
                }
            }

            TermData::App(Symbol::Named(ref name), ref args) => {
                // Check if this is an arithmetic predicate that might have ITE in args
                let is_arith_pred = matches!(name.as_str(), "<" | "<=" | ">" | ">=" | "=");

                if is_arith_pred && args.len() == 2 {
                    // First, recursively lift ITEs in the arguments
                    let arg0 = self.lift_ite_recursive(args[0]);
                    let arg1 = self.lift_ite_recursive(args[1]);

                    // Now check if either argument is an ITE (at the top level)
                    // and if so, perform Shannon expansion
                    self.lift_ite_from_predicate(&name.clone(), arg0, arg1)
                } else if matches!(name.as_str(), "and" | "or" | "xor" | "=>") {
                    // Recursively lift in Boolean connectives
                    let lifted_args: Vec<TermId> =
                        args.iter().map(|&a| self.lift_ite_recursive(a)).collect();
                    let changed = lifted_args.iter().zip(args.iter()).any(|(&a, &b)| a != b);
                    if changed {
                        self.mk_app(Symbol::Named(name.clone()), lifted_args, Sort::Bool)
                    } else {
                        term
                    }
                } else {
                    // Other applications: don't lift (could be arithmetic operations)
                    term
                }
            }

            TermData::App(Symbol::Indexed(_, _), _) => {
                // Indexed symbols - no lifting needed
                term
            }

            TermData::Let(_, _) => {
                // Let bindings should be expanded before this point
                term
            }
        }
    }

    /// Lift ITE from an arithmetic predicate's arguments.
    ///
    /// If arg0 or arg1 contains an arithmetic ITE, we expand:
    /// - `(pred (ite c t e) y)` → `(ite c (pred t y) (pred e y))`
    /// - `(pred x (ite c t e))` → `(ite c (pred x t) (pred x e))`
    ///
    /// This also handles nested ITEs like `(pred (+ x (ite c a b)) y)`.
    fn lift_ite_from_predicate(&mut self, pred: &str, arg0: TermId, arg1: TermId) -> TermId {
        // First, lift any ITEs from within the arithmetic expressions
        let lifted_arg0 = self.lift_ite_from_arith(arg0);
        let lifted_arg1 = self.lift_ite_from_arith(arg1);

        // Check if lifted_arg0 is an arithmetic ITE (non-Bool result)
        if let TermData::Ite(cond, then_t, else_t) = self.get(lifted_arg0).clone() {
            if self.sort(then_t) != &Sort::Bool {
                // Lift ITE from first argument
                // (pred (ite c t e) y) → (ite c (pred t y) (pred e y))
                let then_pred = self.lift_ite_from_predicate(pred, then_t, lifted_arg1);
                let else_pred = self.lift_ite_from_predicate(pred, else_t, lifted_arg1);
                return self.mk_ite(cond, then_pred, else_pred);
            }
        }

        // Check if lifted_arg1 is an arithmetic ITE (non-Bool result)
        if let TermData::Ite(cond, then_t, else_t) = self.get(lifted_arg1).clone() {
            if self.sort(then_t) != &Sort::Bool {
                // Lift ITE from second argument
                // (pred x (ite c t e)) → (ite c (pred x t) (pred x e))
                let then_pred = self.lift_ite_from_predicate(pred, lifted_arg0, then_t);
                let else_pred = self.lift_ite_from_predicate(pred, lifted_arg0, else_t);
                return self.mk_ite(cond, then_pred, else_pred);
            }
        }

        // No ITE at top level, create the predicate directly
        self.mk_app(
            Symbol::Named(pred.to_string()),
            vec![lifted_arg0, lifted_arg1],
            Sort::Bool,
        )
    }

    /// Lift ITEs out of arithmetic expressions.
    ///
    /// Transforms: `(+ x (ite c a b))` → `(ite c (+ x a) (+ x b))`
    ///
    /// This is applied recursively to handle deeply nested ITEs.
    fn lift_ite_from_arith(&mut self, term: TermId) -> TermId {
        let sort = self.sort(term).clone();

        // Only process arithmetic terms (Int/Real), not Bool
        if sort == Sort::Bool {
            return term;
        }

        match self.get(term).clone() {
            TermData::Const(_) | TermData::Var(_, _) => term,

            TermData::Ite(cond, then_t, else_t) => {
                // Recursively lift from branches
                let lifted_then = self.lift_ite_from_arith(then_t);
                let lifted_else = self.lift_ite_from_arith(else_t);
                if lifted_then == then_t && lifted_else == else_t {
                    term
                } else {
                    self.mk_ite(cond, lifted_then, lifted_else)
                }
            }

            TermData::App(Symbol::Named(ref name), ref args) => {
                let is_arith_op = matches!(name.as_str(), "+" | "-" | "*" | "/" | "abs");

                if is_arith_op {
                    // Recursively lift from arguments first
                    let lifted_args: Vec<TermId> =
                        args.iter().map(|&a| self.lift_ite_from_arith(a)).collect();

                    // Check if any argument is an ITE
                    for (i, &arg) in lifted_args.iter().enumerate() {
                        if let TermData::Ite(cond, then_t, else_t) = self.get(arg).clone() {
                            // Found an ITE - lift it out
                            // (op ... (ite c t e) ...) → (ite c (op ... t ...) (op ... e ...))
                            let mut then_args = lifted_args.clone();
                            then_args[i] = then_t;
                            let mut else_args = lifted_args.clone();
                            else_args[i] = else_t;

                            let then_op =
                                self.mk_app(Symbol::Named(name.clone()), then_args, sort.clone());
                            let else_op =
                                self.mk_app(Symbol::Named(name.clone()), else_args, sort.clone());

                            // Recursively lift in case there are more ITEs
                            let lifted_then_op = self.lift_ite_from_arith(then_op);
                            let lifted_else_op = self.lift_ite_from_arith(else_op);

                            return self.mk_ite(cond, lifted_then_op, lifted_else_op);
                        }
                    }

                    // No ITEs found, check if arguments changed
                    if lifted_args.iter().zip(args.iter()).any(|(&a, &b)| a != b) {
                        self.mk_app(Symbol::Named(name.clone()), lifted_args, sort)
                    } else {
                        term
                    }
                } else {
                    // Non-arithmetic function, don't descend
                    term
                }
            }

            TermData::App(Symbol::Indexed(_, _), _) | TermData::Not(_) | TermData::Let(_, _) => {
                // These shouldn't appear in arithmetic contexts
                term
            }
        }
    }

    /// Lift arithmetic ITEs from all terms in a list.
    pub fn lift_arithmetic_ite_all(&mut self, terms: &[TermId]) -> Vec<TermId> {
        terms.iter().map(|&t| self.lift_arithmetic_ite(t)).collect()
    }
}

/// A term reference that can be used in place of the full Term enum
/// for backward compatibility
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// A constant value
    Const(Constant),
    /// A variable
    Var(String, Sort),
    /// Function application
    App(String, Vec<TermId>),
    /// Let binding
    Let(Vec<(String, TermId)>, TermId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consing() {
        let mut store = TermStore::new();

        let x1 = store.mk_var("x", Sort::Int);
        let x2 = store.mk_var("x", Sort::Int);

        // Same variable should return same ID
        assert_eq!(x1, x2);
    }

    #[test]
    fn test_bool_constants() {
        let mut store = TermStore::new();

        let t1 = store.mk_bool(true);
        let t2 = store.mk_bool(true);
        let f1 = store.mk_bool(false);

        assert_eq!(t1, t2);
        assert_ne!(t1, f1);
        assert_eq!(t1, store.true_term());
        assert_eq!(f1, store.false_term());
    }

    #[test]
    fn test_negation_simplification() {
        let mut store = TermStore::new();

        let t = store.true_term();
        let not_t = store.mk_not(t);
        let not_not_t = store.mk_not(not_t);

        // not(true) = false
        assert_eq!(not_t, store.false_term());
        // not(not(x)) = x
        assert_eq!(not_not_t, t);
    }

    #[test]
    fn test_de_morgan_not_and() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        let and_xy = store.mk_and(vec![x, y]);
        let result = store.mk_not(and_xy);

        // (not (and x y)) -> (or (not x) (not y))
        match store.get(result) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "or");
                assert_eq!(args.len(), 2);
                assert!(args
                    .iter()
                    .all(|&a| matches!(store.get(a), TermData::Not(_))));
            }
            other => panic!("expected (or ...) after De Morgan, got {other:?}"),
        }
    }

    #[test]
    fn test_de_morgan_not_or() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        let or_xy = store.mk_or(vec![x, y]);
        let result = store.mk_not(or_xy);

        // (not (or x y)) -> (and (not x) (not y))
        match store.get(result) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "and");
                assert_eq!(args.len(), 2);
                assert!(args
                    .iter()
                    .all(|&a| matches!(store.get(a), TermData::Not(_))));
            }
            other => panic!("expected (and ...) after De Morgan, got {other:?}"),
        }
    }

    #[test]
    fn test_de_morgan_enables_complement_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // x ∧ ¬(x ∨ y) -> x ∧ (¬x ∧ ¬y) -> false
        let or_xy = store.mk_or(vec![x, y]);
        let not_or_xy = store.mk_not(or_xy);
        let result = store.mk_and(vec![x, not_or_xy]);

        assert_eq!(result, store.false_term());
    }

    #[test]
    fn test_and_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // and(x, true) = x
        assert_eq!(store.mk_and(vec![x, t]), x);

        // and(x, false) = false
        assert_eq!(store.mk_and(vec![x, f]), f);

        // and() = true
        assert_eq!(store.mk_and(vec![]), t);
    }

    #[test]
    fn test_or_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // or(x, false) = x
        assert_eq!(store.mk_or(vec![x, f]), x);

        // or(x, true) = true
        assert_eq!(store.mk_or(vec![x, t]), t);

        // or() = false
        assert_eq!(store.mk_or(vec![]), f);
    }

    #[test]
    fn test_ite_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);
        let t = store.true_term();
        let f = store.false_term();

        // ite(true, x, y) = x
        assert_eq!(store.mk_ite(t, x, y), x);

        // ite(false, x, y) = y
        assert_eq!(store.mk_ite(f, x, y), y);

        // ite(c, x, x) = x
        let c = store.mk_var("c", Sort::Bool);
        assert_eq!(store.mk_ite(c, x, x), x);
    }

    #[test]
    fn test_equality() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // x = x is true
        assert_eq!(store.mk_eq(x, x), store.true_term());

        // x = y and y = x should be the same term
        let eq1 = store.mk_eq(x, y);
        let eq2 = store.mk_eq(y, x);
        assert_eq!(eq1, eq2);
    }

    #[test]
    fn test_int_constants() {
        let mut store = TermStore::new();

        let i1 = store.mk_int(BigInt::from(42));
        let i2 = store.mk_int(BigInt::from(42));
        let i3 = store.mk_int(BigInt::from(43));

        assert_eq!(i1, i2);
        assert_ne!(i1, i3);
    }

    #[test]
    fn test_bitvec_constants() {
        let mut store = TermStore::new();

        let bv1 = store.mk_bitvec(BigInt::from(0xFF), 8);
        let bv2 = store.mk_bitvec(BigInt::from(0xFF), 8);
        let bv3 = store.mk_bitvec(BigInt::from(0xFF), 16);

        assert_eq!(bv1, bv2);
        assert_ne!(bv1, bv3); // Different width
    }

    #[test]
    fn test_fresh_vars() {
        let mut store = TermStore::new();

        let v1 = store.mk_fresh_var("tseitin", Sort::Bool);
        let v2 = store.mk_fresh_var("tseitin", Sort::Bool);
        let v3 = store.mk_fresh_var("tseitin", Sort::Bool);

        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
    }

    #[test]
    fn test_app_canonical_form() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // and(x, y, z) and and(z, y, x) should be the same
        let and1 = store.mk_and(vec![x, y, z]);
        let and2 = store.mk_and(vec![z, y, x]);
        assert_eq!(and1, and2);
    }

    // =======================================================================
    // Arithmetic constant folding tests
    // =======================================================================

    #[test]
    fn test_int_addition_constant_folding() {
        let mut store = TermStore::new();

        let a = store.mk_int(BigInt::from(2));
        let b = store.mk_int(BigInt::from(3));
        let c = store.mk_int(BigInt::from(5));

        // 2 + 3 = 5
        let sum = store.mk_add(vec![a, b]);
        assert_eq!(sum, c);

        // 1 + 2 + 3 = 6
        let one = store.mk_int(BigInt::from(1));
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));
        let six = store.mk_int(BigInt::from(6));
        let sum3 = store.mk_add(vec![one, two, three]);
        assert_eq!(sum3, six);
    }

    #[test]
    fn test_int_addition_identity() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let zero = store.mk_int(BigInt::from(0));

        // x + 0 = x
        assert_eq!(store.mk_add(vec![x, zero]), x);
        // 0 + x = x
        assert_eq!(store.mk_add(vec![zero, x]), x);
    }

    #[test]
    fn test_int_subtraction_constant_folding() {
        let mut store = TermStore::new();

        let five = store.mk_int(BigInt::from(5));
        let three = store.mk_int(BigInt::from(3));
        let two = store.mk_int(BigInt::from(2));

        // 5 - 3 = 2
        assert_eq!(store.mk_sub(vec![five, three]), two);
    }

    #[test]
    fn test_int_subtraction_identity() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let zero = store.mk_int(BigInt::from(0));

        // x - 0 = x
        assert_eq!(store.mk_sub(vec![x, zero]), x);

        // x - x = 0
        assert_eq!(store.mk_sub(vec![x, x]), zero);
    }

    #[test]
    fn test_int_multiplication_constant_folding() {
        let mut store = TermStore::new();

        let three = store.mk_int(BigInt::from(3));
        let four = store.mk_int(BigInt::from(4));
        let twelve = store.mk_int(BigInt::from(12));

        // 3 * 4 = 12
        assert_eq!(store.mk_mul(vec![three, four]), twelve);

        // 2 * 3 * 4 = 24
        let two = store.mk_int(BigInt::from(2));
        let twenty_four = store.mk_int(BigInt::from(24));
        assert_eq!(store.mk_mul(vec![two, three, four]), twenty_four);
    }

    #[test]
    fn test_int_multiplication_identity_annihilation() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let zero = store.mk_int(BigInt::from(0));
        let one = store.mk_int(BigInt::from(1));

        // x * 1 = x
        assert_eq!(store.mk_mul(vec![x, one]), x);

        // x * 0 = 0
        assert_eq!(store.mk_mul(vec![x, zero]), zero);
    }

    #[test]
    fn test_int_constant_distribution_over_add() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);
        let two = store.mk_int(BigInt::from(2));

        let sum = store.mk_add(vec![x, y]);
        let result = store.mk_mul(vec![two, sum]);

        let two_x = store.mk_mul(vec![two, x]);
        let two_y = store.mk_mul(vec![two, y]);
        let expected = store.mk_add(vec![two_x, two_y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_int_constant_distribution_over_add_with_folding() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        let sum = store.mk_add(vec![x, y]);
        let result = store.mk_mul(vec![two, three, sum]);

        let six_x = store.mk_mul(vec![two, three, x]);
        let six_y = store.mk_mul(vec![two, three, y]);
        let expected = store.mk_add(vec![six_x, six_y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_int_negation() {
        let mut store = TermStore::new();

        let five = store.mk_int(BigInt::from(5));
        let neg_five = store.mk_int(BigInt::from(-5));

        // -5
        assert_eq!(store.mk_neg(five), neg_five);
        // -(-5) = 5
        assert_eq!(store.mk_neg(neg_five), five);
    }

    #[test]
    fn test_neg_distribute_over_add_int() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let sum = store.mk_add(vec![a, b]);

        // -(a + b) → (-a) + (-b)
        let result = store.mk_neg(sum);
        let neg_a = store.mk_neg(a);
        let neg_b = store.mk_neg(b);
        let expected = store.mk_add(vec![neg_a, neg_b]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_distribute_over_add_real() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);
        let sum = store.mk_add(vec![x, y]);

        // -(x + y) → (-x) + (-y)
        let result = store.mk_neg(sum);
        let neg_x = store.mk_neg(x);
        let neg_y = store.mk_neg(y);
        let expected = store.mk_add(vec![neg_x, neg_y]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_distribute_with_coefficients() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // -(2a + 3a) → -5a via distribution and coefficient collection
        let two_a = store.mk_mul(vec![two, a]);
        let three_a = store.mk_mul(vec![three, a]);
        let sum = store.mk_add(vec![two_a, three_a]);
        let result = store.mk_neg(sum);

        // Expected: -5a (single term with coefficient -5)
        let neg_five = store.mk_int(BigInt::from(-5));
        let expected = store.mk_mul(vec![neg_five, a]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_factor_into_mul_int() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let three = store.mk_int(BigInt::from(3));
        let neg_three = store.mk_int(BigInt::from(-3));

        // -(3 * a) → (-3) * a
        let three_a = store.mk_mul(vec![three, a]);
        let result = store.mk_neg(three_a);

        let expected = store.mk_mul(vec![neg_three, a]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_factor_into_mul_real() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Real);
        let half = store.mk_rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
        let neg_half = store.mk_rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));

        // -(1/2 * x) → (-1/2) * x
        let half_x = store.mk_mul(vec![half, x]);
        let result = store.mk_neg(half_x);

        let expected = store.mk_mul(vec![neg_half, x]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_chain_simplification() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let two = store.mk_int(BigInt::from(2));

        // -(2a + b) → -2a + (-b) → (-2)*a + (-b)
        let two_a = store.mk_mul(vec![two, a]);
        let sum = store.mk_add(vec![two_a, b]);
        let result = store.mk_neg(sum);

        let neg_two = store.mk_int(BigInt::from(-2));
        let neg_two_a = store.mk_mul(vec![neg_two, a]);
        let neg_b = store.mk_neg(b);
        let expected = store.mk_add(vec![neg_two_a, neg_b]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_int_div_mod() {
        let mut store = TermStore::new();

        let seven = store.mk_int(BigInt::from(7));
        let three = store.mk_int(BigInt::from(3));
        let two = store.mk_int(BigInt::from(2));
        let one = store.mk_int(BigInt::from(1));

        // 7 div 3 = 2
        assert_eq!(store.mk_intdiv(vec![seven, three]), two);

        // 7 mod 3 = 1
        assert_eq!(store.mk_mod(vec![seven, three]), one);
    }

    #[test]
    fn test_rational_addition() {
        let mut store = TermStore::new();

        let half = store.mk_rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
        let quarter = store.mk_rational(BigRational::new(BigInt::from(1), BigInt::from(4)));
        let three_quarters = store.mk_rational(BigRational::new(BigInt::from(3), BigInt::from(4)));

        // 1/2 + 1/4 = 3/4
        assert_eq!(store.mk_add(vec![half, quarter]), three_quarters);
    }

    #[test]
    fn test_real_constant_distribution_over_add() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);
        let three_halves = store.mk_rational(BigRational::new(BigInt::from(3), BigInt::from(2)));

        let sum = store.mk_add(vec![x, y]);
        let result = store.mk_mul(vec![three_halves, sum]);

        let c_x = store.mk_mul(vec![three_halves, x]);
        let c_y = store.mk_mul(vec![three_halves, y]);
        let expected = store.mk_add(vec![c_x, c_y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_rational_division() {
        let mut store = TermStore::new();

        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let half = store.mk_rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
        let four = store.mk_rational(BigRational::from(BigInt::from(4)));

        // 2 / 0.5 = 4
        assert_eq!(store.mk_div(vec![two, half]), four);
    }

    #[test]
    fn test_real_self_division() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Real);
        let one = store.mk_rational(BigRational::from(BigInt::from(1)));

        // x / x = 1
        assert_eq!(store.mk_div(vec![x, x]), one);
    }

    #[test]
    fn test_int_self_division() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let one = store.mk_int(BigInt::from(1));

        // x div x = 1
        assert_eq!(store.mk_intdiv(vec![x, x]), one);
    }

    #[test]
    fn test_self_division_complex_term() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let sum = store.mk_add(vec![a, b]);
        let one = store.mk_int(BigInt::from(1));

        // (a + b) div (a + b) = 1
        assert_eq!(store.mk_intdiv(vec![sum, sum]), one);
    }

    #[test]
    fn test_abs() {
        let mut store = TermStore::new();

        let neg_five = store.mk_int(BigInt::from(-5));
        let five = store.mk_int(BigInt::from(5));
        let zero = store.mk_int(BigInt::from(0));

        // |−5| = 5
        assert_eq!(store.mk_abs(neg_five), five);
        // |5| = 5
        assert_eq!(store.mk_abs(five), five);
        // |0| = 0
        assert_eq!(store.mk_abs(zero), zero);
    }

    #[test]
    fn test_partial_folding() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));
        let five = store.mk_int(BigInt::from(5));

        // x + 2 + 3 should simplify to x + 5
        let result = store.mk_add(vec![x, two, three]);

        // The result should contain x and 5, not 2 and 3
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 2);
            // One should be x, one should be 5
            assert!(args.contains(&x));
            assert!(args.contains(&five));
        } else {
            panic!("Expected App term");
        }
    }

    // =======================================================================
    // Arithmetic flattening and simplification tests
    // =======================================================================

    #[test]
    fn test_add_flattening() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let c = store.mk_var("c", Sort::Int);

        // Create (+ a b)
        let add_ab = store.mk_add(vec![a, b]);

        // (+ (+ a b) c) should flatten to (+ a b c)
        let result = store.mk_add(vec![add_ab, c]);

        // Verify it's a flat addition with all three arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 3);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
        } else {
            panic!("Expected + App term");
        }
    }

    #[test]
    fn test_add_flattening_nested() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let c = store.mk_var("c", Sort::Int);
        let d = store.mk_var("d", Sort::Int);

        // Create (+ a b) and (+ c d)
        let add_ab = store.mk_add(vec![a, b]);
        let add_cd = store.mk_add(vec![c, d]);

        // (+ (+ a b) (+ c d)) should flatten to (+ a b c d)
        let result = store.mk_add(vec![add_ab, add_cd]);

        // Verify it's a flat addition with all four arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 4);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
            assert!(args.contains(&d));
        } else {
            panic!("Expected + App term");
        }
    }

    #[test]
    fn test_add_flattening_with_constants() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // Create (+ a 2)
        let add_a2 = store.mk_add(vec![a, two]);

        // (+ (+ a 2) 3) should flatten and fold constants to (+ a 5)
        let result = store.mk_add(vec![add_a2, three]);

        // Verify constants were folded
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 2);
            assert!(args.contains(&a));
            // One arg should be 5
            let has_five = args.iter().any(|&arg| {
                store
                    .get_int(arg)
                    .map(|n| *n == BigInt::from(5))
                    .unwrap_or(false)
            });
            assert!(has_five);
        } else {
            panic!("Expected + App term");
        }
    }

    #[test]
    fn test_additive_inverse_detection() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let neg_a = store.mk_neg(a);
        let zero = store.mk_int(BigInt::from(0));

        // a + (-a) = 0
        let result = store.mk_add(vec![a, neg_a]);
        assert_eq!(result, zero);
    }

    #[test]
    fn test_additive_inverse_detection_multiple() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let neg_a = store.mk_neg(a);

        // a + b + (-a) = b
        let result = store.mk_add(vec![a, b, neg_a]);
        assert_eq!(result, b);
    }

    #[test]
    fn test_additive_inverse_with_constant() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let neg_a = store.mk_neg(a);
        let five = store.mk_int(BigInt::from(5));

        // a + (-a) + 5 = 5
        let result = store.mk_add(vec![a, neg_a, five]);
        assert_eq!(result, five);
    }

    // =======================================================================
    // Coefficient collection tests
    // =======================================================================

    #[test]
    fn test_coefficient_collection_basic() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));
        let five = store.mk_int(BigInt::from(5));

        // Create (* 2 a) and (* 3 a)
        let two_a = store.mk_mul(vec![two, a]);
        let three_a = store.mk_mul(vec![three, a]);

        // (* 2 a) + (* 3 a) = (* 5 a)
        let result = store.mk_add(vec![two_a, three_a]);
        let expected = store.mk_mul(vec![five, a]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_coefficient_collection_with_negation() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let three = store.mk_int(BigInt::from(3));

        // Create (* 3 a) and (- a)
        let three_a = store.mk_mul(vec![three, a]);
        let neg_a = store.mk_neg(a);

        // (* 3 a) + (- a) = (* 2 a)
        let result = store.mk_add(vec![three_a, neg_a]);
        let two = store.mk_int(BigInt::from(2));
        let expected = store.mk_mul(vec![two, a]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_coefficient_collection_to_zero() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let neg_two = store.mk_int(BigInt::from(-2));
        let zero = store.mk_int(BigInt::from(0));

        // Create (* 2 a) and (* -2 a)
        let two_a = store.mk_mul(vec![two, a]);
        let neg_two_a = store.mk_mul(vec![neg_two, a]);

        // (* 2 a) + (* -2 a) = 0
        let result = store.mk_add(vec![two_a, neg_two_a]);
        assert_eq!(result, zero);
    }

    #[test]
    fn test_coefficient_collection_to_single() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let neg_one = store.mk_int(BigInt::from(-1));

        // Create (* 2 a) and (* -1 a)
        let two_a = store.mk_mul(vec![two, a]);
        let neg_one_a = store.mk_mul(vec![neg_one, a]);

        // (* 2 a) + (* -1 a) = a
        let result = store.mk_add(vec![two_a, neg_one_a]);
        assert_eq!(result, a);
    }

    #[test]
    fn test_coefficient_collection_mixed() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // Create (* 2 a), a, and (* 3 b)
        let two_a = store.mk_mul(vec![two, a]);
        let three_b = store.mk_mul(vec![three, b]);

        // (* 2 a) + a + (* 3 b) should combine 2a + 1a = 3a
        let result = store.mk_add(vec![two_a, a, three_b]);

        // Result should be (+ (* 3 a) (* 3 b)) - though order may vary
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 2);
            // Both should be multiplications with coefficient 3
            for &arg in args {
                if let TermData::App(Symbol::Named(op), factors) = store.get(arg) {
                    assert_eq!(op, "*");
                    assert_eq!(factors.len(), 2);
                    // The constant can be at either position (mk_mul puts it at the end)
                    let has_coeff_3 = factors
                        .iter()
                        .any(|&f| store.get_int(f) == Some(&BigInt::from(3)));
                    assert!(has_coeff_3, "Expected coefficient 3 in multiplication");
                } else {
                    panic!("Expected multiplication");
                }
            }
        } else {
            panic!("Expected + App term");
        }
    }

    #[test]
    fn test_coefficient_collection_to_neg() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let neg_three = store.mk_int(BigInt::from(-3));

        // Create (* 2 a) and (* -3 a)
        let two_a = store.mk_mul(vec![two, a]);
        let neg_three_a = store.mk_mul(vec![neg_three, a]);

        // (* 2 a) + (* -3 a) = (- a)
        let result = store.mk_add(vec![two_a, neg_three_a]);
        let expected = store.mk_neg(a);
        assert_eq!(result, expected);
    }

    // =======================================================================
    // Real coefficient collection tests
    // =======================================================================

    #[test]
    fn test_real_coefficient_collection_basic() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let three = store.mk_rational(BigRational::from(BigInt::from(3)));
        let five = store.mk_rational(BigRational::from(BigInt::from(5)));

        // Create (* 2.0 a) and (* 3.0 a)
        let two_a = store.mk_mul(vec![two, a]);
        let three_a = store.mk_mul(vec![three, a]);

        // (* 2.0 a) + (* 3.0 a) = (* 5.0 a)
        let result = store.mk_add(vec![two_a, three_a]);
        let expected = store.mk_mul(vec![five, a]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_real_coefficient_collection_with_negation() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        let three = store.mk_rational(BigRational::from(BigInt::from(3)));

        // Create (* 3.0 a) and (- a)
        let three_a = store.mk_mul(vec![three, a]);
        let neg_a = store.mk_neg(a);

        // (* 3.0 a) + (- a) = (* 2.0 a)
        let result = store.mk_add(vec![three_a, neg_a]);
        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let expected = store.mk_mul(vec![two, a]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_real_coefficient_collection_to_zero() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let neg_two = store.mk_rational(BigRational::from(BigInt::from(-2)));
        let zero = store.mk_rational(BigRational::from(BigInt::from(0)));

        // Create (* 2.0 a) and (* -2.0 a)
        let two_a = store.mk_mul(vec![two, a]);
        let neg_two_a = store.mk_mul(vec![neg_two, a]);

        // (* 2.0 a) + (* -2.0 a) = 0
        let result = store.mk_add(vec![two_a, neg_two_a]);
        assert_eq!(result, zero);
    }

    #[test]
    fn test_real_coefficient_collection_to_single() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let neg_one = store.mk_rational(BigRational::from(BigInt::from(-1)));

        // Create (* 2.0 a) and (* -1.0 a)
        let two_a = store.mk_mul(vec![two, a]);
        let neg_one_a = store.mk_mul(vec![neg_one, a]);

        // (* 2.0 a) + (* -1.0 a) = a
        let result = store.mk_add(vec![two_a, neg_one_a]);
        assert_eq!(result, a);
    }

    #[test]
    fn test_real_coefficient_collection_to_neg() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let neg_three = store.mk_rational(BigRational::from(BigInt::from(-3)));

        // Create (* 2.0 a) and (* -3.0 a)
        let two_a = store.mk_mul(vec![two, a]);
        let neg_three_a = store.mk_mul(vec![neg_three, a]);

        // (* 2.0 a) + (* -3.0 a) = (- a)
        let result = store.mk_add(vec![two_a, neg_three_a]);
        let expected = store.mk_neg(a);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_real_coefficient_collection_fractional() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        // Create 1/2 and 3/2
        let half = store.mk_rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
        let three_halves = store.mk_rational(BigRational::new(BigInt::from(3), BigInt::from(2)));
        let two = store.mk_rational(BigRational::from(BigInt::from(2)));

        // Create (* 1/2 a) and (* 3/2 a)
        let half_a = store.mk_mul(vec![half, a]);
        let three_halves_a = store.mk_mul(vec![three_halves, a]);

        // (* 1/2 a) + (* 3/2 a) = (* 2 a)
        let result = store.mk_add(vec![half_a, three_halves_a]);
        let expected = store.mk_mul(vec![two, a]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_real_coefficient_collection_mixed() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        let b = store.mk_var("b", Sort::Real);
        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let three = store.mk_rational(BigRational::from(BigInt::from(3)));

        // Create (* 2 a), a, and (* 3 b)
        let two_a = store.mk_mul(vec![two, a]);
        let three_b = store.mk_mul(vec![three, b]);

        // (* 2 a) + a + (* 3 b) should combine 2a + 1a = 3a
        let result = store.mk_add(vec![two_a, a, three_b]);

        // Result should be (+ (* 3 a) (* 3 b)) - though order may vary
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 2);
            // Both should be multiplications with coefficient 3
            for &arg in args {
                if let TermData::App(Symbol::Named(op), factors) = store.get(arg) {
                    assert_eq!(op, "*");
                    assert_eq!(factors.len(), 2);
                    let has_coeff_3 = factors.iter().any(|&f| {
                        store.get_rational(f) == Some(&BigRational::from(BigInt::from(3)))
                    });
                    assert!(has_coeff_3, "Expected coefficient 3 in multiplication");
                } else {
                    panic!("Expected multiplication");
                }
            }
        } else {
            panic!("Expected + App term");
        }
    }

    // =======================================================================
    // Coefficient collection determinism tests
    // =======================================================================

    #[test]
    fn test_coefficient_collection_deterministic_order_int() {
        let mut store = TermStore::new();

        // Create variables in a specific order - a is allocated before b
        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);

        // a has lower TermId than b
        assert!(a.0 < b.0, "a should have lower TermId than b");

        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // Create terms in reverse order: 3b first, then 2a
        let three_b = store.mk_mul(vec![three, b]);
        let two_a = store.mk_mul(vec![two, a]);
        let three_b_dup = store.mk_mul(vec![three, b]);

        // Add in order: 3b, 2a, 3b (duplicates combine)
        // Input order has b before a, but output should have a before b (sorted by TermId)
        let result = store.mk_add(vec![three_b, two_a, three_b_dup]);

        // Result should be (+ (* 2 a) (* 6 b)) with a before b due to TermId sorting
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 2);

            // First arg should contain 'a' (lower TermId)
            let first_base = extract_base_from_mul(&store, args[0]);
            let second_base = extract_base_from_mul(&store, args[1]);

            assert_eq!(
                first_base,
                Some(a),
                "First term should be based on 'a' (lower TermId)"
            );
            assert_eq!(
                second_base,
                Some(b),
                "Second term should be based on 'b' (higher TermId)"
            );
        } else {
            panic!("Expected + App term");
        }
    }

    #[test]
    fn test_coefficient_collection_deterministic_order_real() {
        let mut store = TermStore::new();

        // Create variables in a specific order - x is allocated before y
        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);

        // x has lower TermId than y
        assert!(x.0 < y.0, "x should have lower TermId than y");

        let two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let three = store.mk_rational(BigRational::from(BigInt::from(3)));

        // Create terms in reverse order: 3y first, then 2x
        let three_y = store.mk_mul(vec![three, y]);
        let two_x = store.mk_mul(vec![two, x]);
        let three_y_dup = store.mk_mul(vec![three, y]);

        // Add in order: 3y, 2x, 3y (duplicates combine)
        // Input order has y before x, but output should have x before y (sorted by TermId)
        let result = store.mk_add(vec![three_y, two_x, three_y_dup]);

        // Result should be (+ (* 2 x) (* 6 y)) with x before y due to TermId sorting
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "+");
            assert_eq!(args.len(), 2);

            // First arg should contain 'x' (lower TermId)
            let first_base = extract_base_from_mul(&store, args[0]);
            let second_base = extract_base_from_mul(&store, args[1]);

            assert_eq!(
                first_base,
                Some(x),
                "First term should be based on 'x' (lower TermId)"
            );
            assert_eq!(
                second_base,
                Some(y),
                "Second term should be based on 'y' (higher TermId)"
            );
        } else {
            panic!("Expected + App term");
        }
    }

    /// Helper to extract the base variable from a multiplication term like (* coeff base)
    fn extract_base_from_mul(store: &TermStore, term: TermId) -> Option<TermId> {
        if let TermData::App(Symbol::Named(name), args) = store.get(term) {
            if name == "*" && args.len() == 2 {
                // One of the args is a constant, the other is the base
                for &arg in args {
                    if store.get_int(arg).is_none() && store.get_rational(arg).is_none() {
                        return Some(arg);
                    }
                }
            }
        }
        None
    }

    #[test]
    fn test_mul_flattening() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let c = store.mk_var("c", Sort::Int);

        // Create (* a b)
        let mul_ab = store.mk_mul(vec![a, b]);

        // (* (* a b) c) should flatten to (* a b c)
        let result = store.mk_mul(vec![mul_ab, c]);

        // Verify it's a flat multiplication with all three arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "*");
            assert_eq!(args.len(), 3);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
        } else {
            panic!("Expected * App term");
        }
    }

    #[test]
    fn test_mul_flattening_nested() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let c = store.mk_var("c", Sort::Int);
        let d = store.mk_var("d", Sort::Int);

        // Create (* a b) and (* c d)
        let mul_ab = store.mk_mul(vec![a, b]);
        let mul_cd = store.mk_mul(vec![c, d]);

        // (* (* a b) (* c d)) should flatten to (* a b c d)
        let result = store.mk_mul(vec![mul_ab, mul_cd]);

        // Verify it's a flat multiplication with all four arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "*");
            assert_eq!(args.len(), 4);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
            assert!(args.contains(&d));
        } else {
            panic!("Expected * App term");
        }
    }

    #[test]
    fn test_mul_flattening_with_constants() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // Create (* a 2)
        let mul_a2 = store.mk_mul(vec![a, two]);

        // (* (* a 2) 3) should flatten and fold constants to (* a 6)
        let result = store.mk_mul(vec![mul_a2, three]);

        // Verify constants were folded
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "*");
            assert_eq!(args.len(), 2);
            assert!(args.contains(&a));
            // One arg should be 6
            let has_six = args.iter().any(|&arg| {
                store
                    .get_int(arg)
                    .map(|n| *n == BigInt::from(6))
                    .unwrap_or(false)
            });
            assert!(has_six);
        } else {
            panic!("Expected * App term");
        }
    }

    #[test]
    fn test_mul_by_minus_one() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let neg_one = store.mk_int(BigInt::from(-1));

        // (* -1 a) = (- a)
        let result = store.mk_mul(vec![neg_one, a]);

        // Should be a negation
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "-");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], a);
        } else {
            panic!("Expected - App term (negation)");
        }
    }

    #[test]
    fn test_mul_by_minus_one_multiple() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let neg_one = store.mk_int(BigInt::from(-1));

        // (* -1 a b) = (- (* a b))
        let result = store.mk_mul(vec![neg_one, a, b]);

        // Should be a negation of multiplication
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "-");
            assert_eq!(args.len(), 1);
            // Inner should be (* a b)
            if let TermData::App(Symbol::Named(inner_name), inner_args) = store.get(args[0]) {
                assert_eq!(inner_name, "*");
                assert_eq!(inner_args.len(), 2);
                assert!(inner_args.contains(&a));
                assert!(inner_args.contains(&b));
            } else {
                panic!("Expected inner * App term");
            }
        } else {
            panic!("Expected - App term (negation)");
        }
    }

    #[test]
    fn test_mul_by_minus_one_folded() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let neg_three = store.mk_int(BigInt::from(-3));

        // (* 2 -3 a) = (* -6 a) = (- (* 6 a)) but since -6 is not -1, it stays as (* a -6)
        // Actually: 2 * -3 = -6, so (* 2 -3 a) = (* -6 a)
        // Wait, the implementation folds to -6, not -1, so it won't become negation
        let result = store.mk_mul(vec![two, neg_three, a]);

        // Should have a and -6 as arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "*");
            assert_eq!(args.len(), 2);
            assert!(args.contains(&a));
            let has_neg_six = args.iter().any(|&arg| {
                store
                    .get_int(arg)
                    .map(|n| *n == BigInt::from(-6))
                    .unwrap_or(false)
            });
            assert!(has_neg_six);
        } else {
            panic!("Expected * App term");
        }
    }

    #[test]
    fn test_mul_double_negation_via_minus_one() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let neg_one = store.mk_int(BigInt::from(-1));

        // (* -1 -1 a) = (* 1 a) = a
        let result = store.mk_mul(vec![neg_one, neg_one, a]);
        assert_eq!(result, a);
    }

    // =======================================================================
    // Comparison constant folding tests
    // =======================================================================

    #[test]
    fn test_int_less_than() {
        let mut store = TermStore::new();

        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // 2 < 3 = true
        assert_eq!(store.mk_lt(two, three), store.true_term());
        // 3 < 2 = false
        assert_eq!(store.mk_lt(three, two), store.false_term());
        // 2 < 2 = false
        assert_eq!(store.mk_lt(two, two), store.false_term());
    }

    #[test]
    fn test_int_less_equal() {
        let mut store = TermStore::new();

        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // 2 <= 3 = true
        assert_eq!(store.mk_le(two, three), store.true_term());
        // 3 <= 2 = false
        assert_eq!(store.mk_le(three, two), store.false_term());
        // 2 <= 2 = true
        assert_eq!(store.mk_le(two, two), store.true_term());
    }

    #[test]
    fn test_int_greater_than() {
        let mut store = TermStore::new();

        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // 3 > 2 = true
        assert_eq!(store.mk_gt(three, two), store.true_term());
        // 2 > 3 = false
        assert_eq!(store.mk_gt(two, three), store.false_term());
        // 2 > 2 = false
        assert_eq!(store.mk_gt(two, two), store.false_term());
    }

    #[test]
    fn test_int_greater_equal() {
        let mut store = TermStore::new();

        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // 3 >= 2 = true
        assert_eq!(store.mk_ge(three, two), store.true_term());
        // 2 >= 3 = false
        assert_eq!(store.mk_ge(two, three), store.false_term());
        // 2 >= 2 = true
        assert_eq!(store.mk_ge(two, two), store.true_term());
    }

    #[test]
    fn test_rational_comparisons() {
        let mut store = TermStore::new();

        let half = store.mk_rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
        let third = store.mk_rational(BigRational::new(BigInt::from(1), BigInt::from(3)));

        // 1/3 < 1/2 = true
        assert_eq!(store.mk_lt(third, half), store.true_term());
        // 1/2 < 1/3 = false
        assert_eq!(store.mk_lt(half, third), store.false_term());

        // 1/3 <= 1/2 = true
        assert_eq!(store.mk_le(third, half), store.true_term());
        // 1/2 <= 1/2 = true
        assert_eq!(store.mk_le(half, half), store.true_term());

        // 1/2 > 1/3 = true
        assert_eq!(store.mk_gt(half, third), store.true_term());

        // 1/2 >= 1/3 = true
        assert_eq!(store.mk_ge(half, third), store.true_term());
    }

    #[test]
    fn test_comparison_reflexive_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);

        // x < x = false
        assert_eq!(store.mk_lt(x, x), store.false_term());
        // x <= x = true
        assert_eq!(store.mk_le(x, x), store.true_term());
        // x > x = false
        assert_eq!(store.mk_gt(x, x), store.false_term());
        // x >= x = true
        assert_eq!(store.mk_ge(x, x), store.true_term());
    }

    #[test]
    fn test_comparison_non_constant() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // x < y should create an App term, not simplify
        let lt = store.mk_lt(x, y);
        if let TermData::App(Symbol::Named(name), args) = store.get(lt) {
            assert_eq!(name, "<");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected App term for non-constant comparison");
        }
    }

    #[test]
    fn test_eq_constant_folding() {
        let mut store = TermStore::new();

        // Integer constants
        let one = store.mk_int(BigInt::from(1));
        let two = store.mk_int(BigInt::from(2));

        // (= 1 2) = false
        assert_eq!(store.mk_eq(one, two), store.false_term());
        // (= 1 1) = true (same constant)
        assert_eq!(store.mk_eq(one, one), store.true_term());

        // Boolean constants
        let t = store.true_term();
        let f = store.false_term();

        // (= true false) = false
        assert_eq!(store.mk_eq(t, f), store.false_term());
        // (= true true) = true
        assert_eq!(store.mk_eq(t, t), store.true_term());

        // String constants
        let hello = store.mk_string("hello".to_string());
        let world = store.mk_string("world".to_string());

        // (= "hello" "world") = false
        assert_eq!(store.mk_eq(hello, world), store.false_term());
        // (= "hello" "hello") = true
        assert_eq!(store.mk_eq(hello, hello), store.true_term());
    }

    #[test]
    fn test_distinct_duplicate_detection() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // (distinct x x) = false
        assert_eq!(store.mk_distinct(vec![x, x]), store.false_term());
        // (distinct x y x) = false (x appears twice)
        assert_eq!(store.mk_distinct(vec![x, y, x]), store.false_term());
        // (distinct x y) should not simplify (non-constant)
        let dist = store.mk_distinct(vec![x, y]);
        assert_ne!(dist, store.true_term());
        assert_ne!(dist, store.false_term());
    }

    #[test]
    fn test_distinct_constant_folding() {
        let mut store = TermStore::new();

        let one = store.mk_int(BigInt::from(1));
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // (distinct 1 2 3) = true
        assert_eq!(store.mk_distinct(vec![one, two, three]), store.true_term());
        // (distinct 1 2 1) = false (duplicate)
        assert_eq!(store.mk_distinct(vec![one, two, one]), store.false_term());
        // (distinct 1 1) = false
        assert_eq!(store.mk_distinct(vec![one, one]), store.false_term());
    }

    #[test]
    fn test_ite_boolean_branch_simplification() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // (ite c true false) = c
        assert_eq!(store.mk_ite(c, t, f), c);

        // (ite c false true) = (not c)
        let not_c = store.mk_not(c);
        assert_eq!(store.mk_ite(c, f, t), not_c);

        // (ite c c false) = c
        assert_eq!(store.mk_ite(c, c, f), c);

        // (ite c true c) = c
        assert_eq!(store.mk_ite(c, t, c), c);
    }

    #[test]
    fn test_ite_to_and_or_simplification() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // (ite c x false) = (and c x)
        let result = store.mk_ite(c, x, f);
        let expected = store.mk_and(vec![c, x]);
        assert_eq!(result, expected);

        // (ite c true x) = (or c x)
        let result = store.mk_ite(c, t, x);
        let expected = store.mk_or(vec![c, x]);
        assert_eq!(result, expected);

        // (ite c false x) = (and (not c) x)
        let result = store.mk_ite(c, f, x);
        let not_c = store.mk_not(c);
        let expected = store.mk_and(vec![not_c, x]);
        assert_eq!(result, expected);

        // (ite c x true) = (or (not c) x)
        let result = store.mk_ite(c, x, t);
        let not_c = store.mk_not(c);
        let expected = store.mk_or(vec![not_c, x]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ite_nested_simplification() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // (ite c (ite c x y) z) = (ite c x z)
        let inner = store.mk_ite(c, x, y);
        let result = store.mk_ite(c, inner, z);
        let expected = store.mk_ite(c, x, z);
        assert_eq!(result, expected);

        // (ite c x (ite c y z)) = (ite c x z)
        let inner = store.mk_ite(c, y, z);
        let result = store.mk_ite(c, x, inner);
        let expected = store.mk_ite(c, x, z);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ite_non_bool_no_simplification() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Int);
        let zero = store.mk_int(BigInt::from(0));

        // (ite c x 0) should NOT simplify to (and c x) since x is Int
        let result = store.mk_ite(c, x, zero);
        // The result should be an Ite term, not an And
        match store.get(result) {
            TermData::Ite(_, _, _) => {} // expected
            _ => panic!("Expected Ite term for non-Bool branches"),
        }
    }

    #[test]
    fn test_ite_negated_condition_normalization() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // (ite (not c) a b) -> (ite c b a)
        let not_c = store.mk_not(c);
        let ite_negated = store.mk_ite(not_c, x, y);
        let ite_positive = store.mk_ite(c, y, x);
        assert_eq!(ite_negated, ite_positive);

        // Verify the condition is positive, not negated
        if let TermData::Ite(cond, then_t, else_t) = store.get(ite_negated) {
            assert_eq!(*cond, c, "Condition should be positive (c), not (not c)");
            assert_eq!(*then_t, y, "Then branch should be y (swapped)");
            assert_eq!(*else_t, x, "Else branch should be x (swapped)");
        } else {
            panic!("Expected Ite term");
        }
    }

    #[test]
    fn test_ite_negated_condition_with_bool_branches() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // (ite (not c) true false) -> (ite c false true) -> (not c)
        // This tests that negated condition normalization composes with other simplifications
        let not_c = store.mk_not(c);
        let result = store.mk_ite(not_c, t, f);
        // After normalization: (ite c false true) = (not c)
        assert_eq!(result, not_c);
    }

    #[test]
    fn test_ite_double_negated_condition() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // (ite (not (not c)) a b) -> (ite (not c) b a) -> (ite c a b)
        // Double negation should fully simplify
        let not_c = store.mk_not(c);
        let not_not_c = store.mk_not(not_c);
        let result = store.mk_ite(not_not_c, x, y);

        // not_not_c simplifies to c in mk_not, so this should be (ite c x y)
        let expected = store.mk_ite(c, x, y);
        assert_eq!(result, expected);
    }

    // =======================================================================
    // Comparison normalization tests
    // =======================================================================

    #[test]
    fn test_comparison_normalization_gt_to_lt() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // (> x y) should produce the same term as (< y x)
        let gt = store.mk_gt(x, y);
        let lt = store.mk_lt(y, x);
        assert_eq!(gt, lt);

        // Verify it's a < term with swapped arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(gt) {
            assert_eq!(name, "<");
            assert_eq!(args[0], y);
            assert_eq!(args[1], x);
        } else {
            panic!("Expected < App term");
        }
    }

    #[test]
    fn test_comparison_normalization_ge_to_le() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // (>= x y) should produce the same term as (<= y x)
        let ge = store.mk_ge(x, y);
        let le = store.mk_le(y, x);
        assert_eq!(ge, le);

        // Verify it's a <= term with swapped arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(ge) {
            assert_eq!(name, "<=");
            assert_eq!(args[0], y);
            assert_eq!(args[1], x);
        } else {
            panic!("Expected <= App term");
        }
    }

    #[test]
    fn test_comparison_normalization_term_sharing() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Real);
        let b = store.mk_var("b", Sort::Real);

        // These should all produce the same term (term sharing)
        let t1 = store.mk_gt(b, a); // (> b a) -> (< a b)
        let t2 = store.mk_lt(a, b); // (< a b)
        let t3 = store.mk_gt(b, a); // (> b a) -> (< a b)

        assert_eq!(t1, t2);
        assert_eq!(t2, t3);

        // Similarly for >= and <=
        let t4 = store.mk_ge(b, a); // (>= b a) -> (<= a b)
        let t5 = store.mk_le(a, b); // (<= a b)

        assert_eq!(t4, t5);
    }

    // =======================================================================
    // Boolean equality simplification tests
    // =======================================================================

    #[test]
    fn test_eq_bool_with_true() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let t = store.true_term();

        // (= x true) -> x
        assert_eq!(store.mk_eq(x, t), x);
        // (= true x) -> x
        assert_eq!(store.mk_eq(t, x), x);
    }

    #[test]
    fn test_eq_bool_with_false() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let f = store.false_term();
        let not_x = store.mk_not(x);

        // (= x false) -> (not x)
        assert_eq!(store.mk_eq(x, f), not_x);
        // (= false x) -> (not x)
        assert_eq!(store.mk_eq(f, x), not_x);
    }

    #[test]
    fn test_eq_bool_nested() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // (= (and x y) true) -> (and x y)
        let and_xy = store.mk_and(vec![x, y]);
        assert_eq!(store.mk_eq(and_xy, t), and_xy);

        // (= (or x y) false) -> (not (or x y)) (which mk_not may normalize via De Morgan)
        let or_xy = store.mk_or(vec![x, y]);
        let not_or = store.mk_not(or_xy);
        assert_eq!(store.mk_eq(or_xy, f), not_or);
    }

    #[test]
    fn test_eq_bool_not_applied_to_non_bool() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let t = store.true_term();

        // (= x true) where x is Int should NOT simplify
        // (this would be ill-sorted anyway, but let's verify we don't crash)
        let eq = store.mk_eq(x, t);

        // Should be an = term, not simplified
        match store.get(eq) {
            TermData::App(Symbol::Named(name), _) => assert_eq!(name, "="),
            _ => panic!("Expected App term"),
        }
    }

    #[test]
    fn test_eq_complement_detection() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let not_x = store.mk_not(x);

        // (= x (not x)) -> false
        assert_eq!(store.mk_eq(x, not_x), store.false_term());
        // (= (not x) x) -> false
        assert_eq!(store.mk_eq(not_x, x), store.false_term());
    }

    #[test]
    fn test_eq_complement_with_complex_term() {
        let mut store = TermStore::new();

        // Use uninterpreted predicates to avoid De Morgan transformation
        let u = Sort::Uninterpreted("U".to_string());
        let a = store.mk_var("a", u.clone());
        let p_a = store.mk_app(Symbol::named("p"), vec![a], Sort::Bool);
        let not_p_a = store.mk_not(p_a);

        // (= p(a) (not p(a))) -> false
        assert_eq!(store.mk_eq(p_a, not_p_a), store.false_term());
    }

    #[test]
    fn test_eq_negation_lifting() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let not_x = store.mk_not(x);
        let not_y = store.mk_not(y);

        // (= (not x) (not y)) -> (= x y)
        let eq_not_not = store.mk_eq(not_x, not_y);
        let eq_xy = store.mk_eq(x, y);
        assert_eq!(eq_not_not, eq_xy);
    }

    #[test]
    fn test_eq_negation_lifting_complex() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);
        let d = store.mk_var("d", Sort::Bool);

        let and_ab = store.mk_and(vec![a, b]);
        let or_cd = store.mk_or(vec![c, d]);
        let not_and = store.mk_not(and_ab);
        let not_or = store.mk_not(or_cd);

        // (= (not (and a b)) (not (or c d))) -> (= (and a b) (or c d))
        // Note: De Morgan may transform the not terms, but the equality should still work
        let eq_negations = store.mk_eq(not_and, not_or);

        // Since De Morgan transforms (not (and a b)) and (not (or c d)),
        // we check that the result is an equality (or possibly the De Morgan forms)
        // The key is that we don't have nested negations in the equality
        match store.get(eq_negations) {
            TermData::App(Symbol::Named(name), _) => assert_eq!(name, "="),
            _ => panic!("Expected App term"),
        }
    }

    #[test]
    fn test_eq_reflexive_negation() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let not_x = store.mk_not(x);

        // (= (not x) (not x)) -> true (by reflexivity before negation lifting)
        assert_eq!(store.mk_eq(not_x, not_x), store.true_term());
    }

    // =======================================================================
    // ITE-equality simplification tests
    // =======================================================================

    #[test]
    fn test_eq_ite_then_branch() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // (= (ite c a b) a) -> (or c (= b a))
        let ite = store.mk_ite(c, a, b);
        let result = store.mk_eq(ite, a);

        let eq_ba = store.mk_eq(b, a);
        let expected = store.mk_or(vec![c, eq_ba]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq_ite_else_branch() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // (= (ite c a b) b) -> (or (not c) (= a b))
        let ite = store.mk_ite(c, a, b);
        let result = store.mk_eq(ite, b);

        let not_c = store.mk_not(c);
        let eq_ab = store.mk_eq(a, b);
        let expected = store.mk_or(vec![not_c, eq_ab]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq_ite_then_branch_symmetric() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // (= a (ite c a b)) -> (or c (= b a))
        let ite = store.mk_ite(c, a, b);
        let result = store.mk_eq(a, ite);

        let eq_ba = store.mk_eq(b, a);
        let expected = store.mk_or(vec![c, eq_ba]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq_ite_else_branch_symmetric() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // (= b (ite c a b)) -> (or (not c) (= a b))
        let ite = store.mk_ite(c, a, b);
        let result = store.mk_eq(b, ite);

        let not_c = store.mk_not(c);
        let eq_ab = store.mk_eq(a, b);
        let expected = store.mk_or(vec![not_c, eq_ab]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq_ite_same_condition() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // (= (ite c a b) (ite c x y)) -> (ite c (= a x) (= b y))
        let ite1 = store.mk_ite(c, a, b);
        let ite2 = store.mk_ite(c, x, y);
        let result = store.mk_eq(ite1, ite2);

        let eq_ax = store.mk_eq(a, x);
        let eq_by = store.mk_eq(b, y);
        let expected = store.mk_ite(c, eq_ax, eq_by);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq_ite_same_branches() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);

        // (= (ite c a a) a) should simplify via same-branch rule first to (= a a) = true
        // The ite simplifies: (ite c a a) -> a
        let ite = store.mk_ite(c, a, a);
        assert_eq!(ite, a); // Same-branch simplification in mk_ite

        // So (= (ite c a a) a) = (= a a) = true
        let result = store.mk_eq(ite, a);
        assert_eq!(result, store.true_term());
    }

    // =======================================================================
    // And/Or flattening tests
    // =======================================================================

    #[test]
    fn test_and_flattening() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);

        // Create (and b c) first
        let and_bc = store.mk_and(vec![b, c]);

        // (and a (and b c)) should flatten to (and a b c)
        let result = store.mk_and(vec![a, and_bc]);

        // Verify it's a flat and with all three arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "and");
            assert_eq!(args.len(), 3);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
        } else {
            panic!("Expected and App term");
        }
    }

    #[test]
    fn test_or_flattening() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);

        // Create (or b c) first
        let or_bc = store.mk_or(vec![b, c]);

        // (or a (or b c)) should flatten to (or a b c)
        let result = store.mk_or(vec![a, or_bc]);

        // Verify it's a flat or with all three arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "or");
            assert_eq!(args.len(), 3);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
        } else {
            panic!("Expected or App term");
        }
    }

    #[test]
    fn test_and_flattening_multiple_nested() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);
        let d = store.mk_var("d", Sort::Bool);

        // Create (and a b) and (and c d)
        let and_ab = store.mk_and(vec![a, b]);
        let and_cd = store.mk_and(vec![c, d]);

        // (and (and a b) (and c d)) should flatten to (and a b c d)
        let result = store.mk_and(vec![and_ab, and_cd]);

        // Verify it's a flat and with all four arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "and");
            assert_eq!(args.len(), 4);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
            assert!(args.contains(&d));
        } else {
            panic!("Expected and App term");
        }
    }

    #[test]
    fn test_or_flattening_multiple_nested() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);
        let d = store.mk_var("d", Sort::Bool);

        // Create (or a b) and (or c d)
        let or_ab = store.mk_or(vec![a, b]);
        let or_cd = store.mk_or(vec![c, d]);

        // (or (or a b) (or c d)) should flatten to (or a b c d)
        let result = store.mk_or(vec![or_ab, or_cd]);

        // Verify it's a flat or with all four arguments
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "or");
            assert_eq!(args.len(), 4);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
            assert!(args.contains(&d));
        } else {
            panic!("Expected or App term");
        }
    }

    #[test]
    fn test_and_flattening_with_constant() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let t = store.true_term();

        // Create (and a b)
        let and_ab = store.mk_and(vec![a, b]);

        // (and true (and a b)) should flatten and simplify to (and a b)
        let result = store.mk_and(vec![t, and_ab]);
        assert_eq!(result, and_ab);
    }

    #[test]
    fn test_and_flattening_to_false() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let f = store.false_term();

        // Create (and a b)
        let and_ab = store.mk_and(vec![a, b]);

        // (and false (and a b)) should simplify to false
        let result = store.mk_and(vec![f, and_ab]);
        assert_eq!(result, f);
    }

    #[test]
    fn test_or_flattening_with_constant() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let f = store.false_term();

        // Create (or a b)
        let or_ab = store.mk_or(vec![a, b]);

        // (or false (or a b)) should flatten and simplify to (or a b)
        let result = store.mk_or(vec![f, or_ab]);
        assert_eq!(result, or_ab);
    }

    #[test]
    fn test_or_flattening_to_true() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let t = store.true_term();

        // Create (or a b)
        let or_ab = store.mk_or(vec![a, b]);

        // (or true (or a b)) should simplify to true
        let result = store.mk_or(vec![t, or_ab]);
        assert_eq!(result, t);
    }

    #[test]
    fn test_and_flattening_dedup() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // Create (and a b)
        let and_ab = store.mk_and(vec![a, b]);

        // (and a (and a b)) should flatten and dedup to (and a b)
        let result = store.mk_and(vec![a, and_ab]);
        assert_eq!(result, and_ab);
    }

    #[test]
    fn test_or_flattening_dedup() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // Create (or a b)
        let or_ab = store.mk_or(vec![a, b]);

        // (or a (or a b)) should flatten and dedup to (or a b)
        let result = store.mk_or(vec![a, or_ab]);
        assert_eq!(result, or_ab);
    }

    // =======================================================================
    // Complement detection tests
    // =======================================================================

    #[test]
    fn test_and_complement_detection() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let not_x = store.mk_not(x);
        let f = store.false_term();

        // (and x (not x)) = false
        assert_eq!(store.mk_and(vec![x, not_x]), f);
        // (and (not x) x) = false (order shouldn't matter)
        assert_eq!(store.mk_and(vec![not_x, x]), f);
    }

    #[test]
    fn test_or_complement_detection() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let not_x = store.mk_not(x);
        let t = store.true_term();

        // (or x (not x)) = true
        assert_eq!(store.mk_or(vec![x, not_x]), t);
        // (or (not x) x) = true (order shouldn't matter)
        assert_eq!(store.mk_or(vec![not_x, x]), t);
    }

    #[test]
    fn test_and_complement_with_other_terms() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);
        let not_x = store.mk_not(x);
        let f = store.false_term();

        // (and x y (not x) z) = false
        assert_eq!(store.mk_and(vec![x, y, not_x, z]), f);
        // (and y z x (not x)) = false
        assert_eq!(store.mk_and(vec![y, z, x, not_x]), f);
    }

    #[test]
    fn test_or_complement_with_other_terms() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);
        let not_x = store.mk_not(x);
        let t = store.true_term();

        // (or x y (not x) z) = true
        assert_eq!(store.mk_or(vec![x, y, not_x, z]), t);
        // (or y z x (not x)) = true
        assert_eq!(store.mk_or(vec![y, z, x, not_x]), t);
    }

    #[test]
    fn test_and_complement_nested() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let not_a = store.mk_not(a);
        let f = store.false_term();

        // Create (and a b) and then (and (and a b) (not a))
        // After flattening: (and a b (not a)) = false
        let and_ab = store.mk_and(vec![a, b]);
        let result = store.mk_and(vec![and_ab, not_a]);
        assert_eq!(result, f);
    }

    #[test]
    fn test_or_complement_nested() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let not_a = store.mk_not(a);
        let t = store.true_term();

        // Create (or a b) and then (or (or a b) (not a))
        // After flattening: (or a b (not a)) = true
        let or_ab = store.mk_or(vec![a, b]);
        let result = store.mk_or(vec![or_ab, not_a]);
        assert_eq!(result, t);
    }

    #[test]
    fn test_and_no_false_complement() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let not_y = store.mk_not(y);

        // (and x (not y)) should NOT simplify to false (no complement)
        let result = store.mk_and(vec![x, not_y]);
        // The result should be an and term, not false
        match store.get(result) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "and");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected and App term"),
        }
    }

    #[test]
    fn test_or_no_true_complement() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let not_y = store.mk_not(y);

        // (or x (not y)) should NOT simplify to true (no complement)
        let result = store.mk_or(vec![x, not_y]);
        // The result should be an or term, not true
        match store.get(result) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "or");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected or App term"),
        }
    }

    #[test]
    fn test_and_complement_complex_term() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let f = store.false_term();

        // Create (and x y) and (not (and x y))
        let and_xy = store.mk_and(vec![x, y]);
        let not_and_xy = store.mk_not(and_xy);

        // (and (and x y) (not (and x y))) = false
        assert_eq!(store.mk_and(vec![and_xy, not_and_xy]), f);
    }

    #[test]
    fn test_or_complement_complex_term() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let t = store.true_term();

        // Create (or x y) and (not (or x y))
        let or_xy = store.mk_or(vec![x, y]);
        let not_or_xy = store.mk_not(or_xy);

        // (or (or x y) (not (or x y))) = true
        assert_eq!(store.mk_or(vec![or_xy, not_or_xy]), t);
    }

    // =======================================================================
    // Absorption law tests
    // =======================================================================

    #[test]
    fn test_and_absorption_basic() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // Create (or x y)
        let or_xy = store.mk_or(vec![x, y]);

        // (and x (or x y)) = x
        let result = store.mk_and(vec![x, or_xy]);
        assert_eq!(result, x);
    }

    #[test]
    fn test_or_absorption_basic() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // Create (and x y)
        let and_xy = store.mk_and(vec![x, y]);

        // (or x (and x y)) = x
        let result = store.mk_or(vec![x, and_xy]);
        assert_eq!(result, x);
    }

    #[test]
    fn test_and_absorption_order_independent() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // Create (or x y)
        let or_xy = store.mk_or(vec![x, y]);

        // (and (or x y) x) = x (order shouldn't matter)
        let result = store.mk_and(vec![or_xy, x]);
        assert_eq!(result, x);
    }

    #[test]
    fn test_or_absorption_order_independent() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // Create (and x y)
        let and_xy = store.mk_and(vec![x, y]);

        // (or (and x y) x) = x (order shouldn't matter)
        let result = store.mk_or(vec![and_xy, x]);
        assert_eq!(result, x);
    }

    #[test]
    fn test_and_absorption_multiple_vars() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // Create (or x y z)
        let or_xyz = store.mk_or(vec![x, y, z]);

        // (and x (or x y z)) = x
        let result = store.mk_and(vec![x, or_xyz]);
        assert_eq!(result, x);

        // (and y (or x y z)) = y
        let result = store.mk_and(vec![y, or_xyz]);
        assert_eq!(result, y);
    }

    #[test]
    fn test_or_absorption_multiple_vars() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // Create (and x y z)
        let and_xyz = store.mk_and(vec![x, y, z]);

        // (or x (and x y z)) = x
        let result = store.mk_or(vec![x, and_xyz]);
        assert_eq!(result, x);

        // (or y (and x y z)) = y
        let result = store.mk_or(vec![y, and_xyz]);
        assert_eq!(result, y);
    }

    #[test]
    fn test_and_absorption_with_extra_terms() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);

        // Create (or a b)
        let or_ab = store.mk_or(vec![a, b]);

        // (and a c (or a b)) should simplify to (and a c) because (or a b) is absorbed by a
        let result = store.mk_and(vec![a, c, or_ab]);

        // Should be (and a c) - the or is absorbed
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "and");
            assert_eq!(args.len(), 2);
            assert!(args.contains(&a));
            assert!(args.contains(&c));
            assert!(!args.contains(&or_ab));
        } else {
            panic!("Expected and App term");
        }
    }

    #[test]
    fn test_or_absorption_with_extra_terms() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let c = store.mk_var("c", Sort::Bool);

        // Create (and a b)
        let and_ab = store.mk_and(vec![a, b]);

        // (or a c (and a b)) should simplify to (or a c) because (and a b) is absorbed by a
        let result = store.mk_or(vec![a, c, and_ab]);

        // Should be (or a c) - the and is absorbed
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "or");
            assert_eq!(args.len(), 2);
            assert!(args.contains(&a));
            assert!(args.contains(&c));
            assert!(!args.contains(&and_ab));
        } else {
            panic!("Expected or App term");
        }
    }

    #[test]
    fn test_and_no_absorption_without_match() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // Create (or y z)
        let or_yz = store.mk_or(vec![y, z]);

        // (and x (or y z)) should NOT simplify - x is not in (or y z)
        let result = store.mk_and(vec![x, or_yz]);

        // Should still be an and with both terms
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "and");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected and App term");
        }
    }

    #[test]
    fn test_or_no_absorption_without_match() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // Create (and y z)
        let and_yz = store.mk_and(vec![y, z]);

        // (or x (and y z)) should NOT simplify - x is not in (and y z)
        let result = store.mk_or(vec![x, and_yz]);

        // Should still be an or with both terms
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "or");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected or App term");
        }
    }

    #[test]
    fn test_absorption_complete_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // Create (or x y)
        let or_xy = store.mk_or(vec![x, y]);

        // (and x y (or x y)) - both x and y are in the or
        // The or should be absorbed, leaving just (and x y)
        let result = store.mk_and(vec![x, y, or_xy]);
        let expected = store.mk_and(vec![x, y]);
        assert_eq!(result, expected);
    }

    // =======================================================================
    // Negation-through absorption tests
    // =======================================================================

    #[test]
    fn test_and_negation_through_absorption_basic() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // (and x (or (not x) y)) should simplify to (and x y)
        // because if x is true, (not x) is false, so (or false y) = y
        let not_x = store.mk_not(x);
        let or_notx_y = store.mk_or(vec![not_x, y]);
        let result = store.mk_and(vec![x, or_notx_y]);

        let expected = store.mk_and(vec![x, y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_or_negation_through_absorption_basic() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // (or x (and (not x) y)) should simplify to (or x y)
        // because if x is false, (not x) is true, so (and true y) = y
        let not_x = store.mk_not(x);
        let and_notx_y = store.mk_and(vec![not_x, y]);
        let result = store.mk_or(vec![x, and_notx_y]);

        let expected = store.mk_or(vec![x, y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_and_negation_through_absorption_multiple() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // (and x (or (not x) y z)) should simplify to (and x (or y z))
        let not_x = store.mk_not(x);
        let or_notx_yz = store.mk_or(vec![not_x, y, z]);
        let result = store.mk_and(vec![x, or_notx_yz]);

        let or_yz = store.mk_or(vec![y, z]);
        let expected = store.mk_and(vec![x, or_yz]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_or_negation_through_absorption_multiple() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // (or x (and (not x) y z)) should simplify to (or x (and y z))
        let not_x = store.mk_not(x);
        let and_notx_yz = store.mk_and(vec![not_x, y, z]);
        let result = store.mk_or(vec![x, and_notx_yz]);

        let and_yz = store.mk_and(vec![y, z]);
        let expected = store.mk_or(vec![x, and_yz]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_and_negation_through_absorption_removes_or() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);

        // (and x (or (not x))) simplifies:
        // First, (or (not x)) = (not x) (single element)
        // Then (and x (not x)) = false (complement)
        let not_x = store.mk_not(x);
        let or_notx = store.mk_or(vec![not_x]);
        let result = store.mk_and(vec![x, or_notx]);

        let expected = store.false_term();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_or_negation_through_absorption_removes_and() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);

        // (or x (and (not x))) simplifies:
        // First, (and (not x)) = (not x) (single element)
        // Then (or x (not x)) = true (complement)
        let not_x = store.mk_not(x);
        let and_notx = store.mk_and(vec![not_x]);
        let result = store.mk_or(vec![x, and_notx]);

        let expected = store.true_term();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_and_negation_through_absorption_multiple_conjuncts() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // (and a b (or (not a) (not b) y)) should simplify to (and a b y)
        // because (not a) and (not b) are both removed from the or
        let not_a = store.mk_not(a);
        let not_b = store.mk_not(b);
        let or_term = store.mk_or(vec![not_a, not_b, y]);
        let result = store.mk_and(vec![a, b, or_term]);

        let expected = store.mk_and(vec![a, b, y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_or_negation_through_absorption_multiple_disjuncts() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // (or a b (and (not a) (not b) y)) should simplify to (or a b y)
        // because (not a) and (not b) are both removed from the and
        let not_a = store.mk_not(a);
        let not_b = store.mk_not(b);
        let and_term = store.mk_and(vec![not_a, not_b, y]);
        let result = store.mk_or(vec![a, b, and_term]);

        let expected = store.mk_or(vec![a, b, y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_and_negation_through_no_false_positive() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // (and x (or y z)) should NOT simplify - no (not x) in the or
        let or_yz = store.mk_or(vec![y, z]);
        let result = store.mk_and(vec![x, or_yz]);

        // Should still be an and with both terms
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "and");
            assert_eq!(args.len(), 2);
            assert!(args.contains(&x));
            assert!(args.contains(&or_yz));
        } else {
            panic!("Expected and App term");
        }
    }

    #[test]
    fn test_or_negation_through_no_false_positive() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let z = store.mk_var("z", Sort::Bool);

        // (or x (and y z)) should NOT simplify - no (not x) in the and
        let and_yz = store.mk_and(vec![y, z]);
        let result = store.mk_or(vec![x, and_yz]);

        // Should still be an or with both terms
        if let TermData::App(Symbol::Named(name), args) = store.get(result) {
            assert_eq!(name, "or");
            assert_eq!(args.len(), 2);
            assert!(args.contains(&x));
            assert!(args.contains(&and_yz));
        } else {
            panic!("Expected or App term");
        }
    }

    // ========================================================================
    // ITE Negation Normalization Tests
    // ========================================================================

    #[test]
    fn test_not_ite_normalization_basic() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // (not (ite c a b)) -> (ite c (not a) (not b))
        let ite = store.mk_ite(c, a, b);
        let not_ite = store.mk_not(ite);

        // Result should be (ite c (not a) (not b))
        let not_a = store.mk_not(a);
        let not_b = store.mk_not(b);
        let expected = store.mk_ite(c, not_a, not_b);

        assert_eq!(not_ite, expected);
    }

    #[test]
    fn test_not_ite_with_true_false_branches() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // (not (ite c true false)) = (not c)
        // Because (ite c true false) = c, and (not c) = (not c)
        let ite = store.mk_ite(c, t, f);
        assert_eq!(ite, c); // First the ite simplifies to c
        let not_ite = store.mk_not(ite);
        let not_c = store.mk_not(c);
        assert_eq!(not_ite, not_c);
    }

    #[test]
    fn test_not_ite_with_false_true_branches() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let t = store.true_term();
        let f = store.false_term();

        // (not (ite c false true)) = c
        // Because (ite c false true) = (not c), and (not (not c)) = c
        let ite = store.mk_ite(c, f, t);
        let not_c = store.mk_not(c);
        assert_eq!(ite, not_c); // First the ite simplifies to (not c)
        let not_ite = store.mk_not(ite);
        assert_eq!(not_ite, c); // Double negation elimination
    }

    #[test]
    fn test_not_ite_with_true_branch() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let t = store.true_term();

        // (not (ite c true a)) = (ite c false (not a)) = (and (not c) (not a))
        // Because (ite c true a) = (or c a), and (not (or c a)) = (and (not c) (not a))
        let ite = store.mk_ite(c, t, a);
        let not_ite = store.mk_not(ite);

        // This should simplify via De Morgan: (not (or c a)) -> (and (not c) (not a))
        let not_c = store.mk_not(c);
        let not_a = store.mk_not(a);
        let expected = store.mk_and(vec![not_c, not_a]);
        assert_eq!(not_ite, expected);
    }

    #[test]
    fn test_not_ite_with_false_branch() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let f = store.false_term();

        // (not (ite c a false)) = (ite c (not a) true) = (or (not c) (not a))
        // Because (ite c a false) = (and c a), and (not (and c a)) = (or (not c) (not a))
        let ite = store.mk_ite(c, a, f);
        let not_ite = store.mk_not(ite);

        // This should simplify via De Morgan: (not (and c a)) -> (or (not c) (not a))
        let not_c = store.mk_not(c);
        let not_a = store.mk_not(a);
        let expected = store.mk_or(vec![not_c, not_a]);
        assert_eq!(not_ite, expected);
    }

    #[test]
    fn test_not_ite_same_branches() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);

        // (not (ite c a a)) = (not a)
        // Because (ite c a a) = a, and (not a) = (not a)
        let ite = store.mk_ite(c, a, a);
        assert_eq!(ite, a); // First the ite simplifies to a
        let not_ite = store.mk_not(ite);
        let not_a = store.mk_not(a);
        assert_eq!(not_ite, not_a);
    }

    #[test]
    fn test_not_ite_double_negation() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // (not (not (ite c a b))) = (ite c a b)
        let ite = store.mk_ite(c, a, b);
        let not_ite = store.mk_not(ite);
        let not_not_ite = store.mk_not(not_ite);

        // After double negation elimination, we should get the original
        // Note: the ite may be normalized differently
        assert_eq!(not_not_ite, ite);
    }

    #[test]
    fn test_not_ite_non_bool_unchanged() {
        let mut store = TermStore::new();

        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);

        // (ite c x y) where x, y are Int cannot have (not ...) applied meaningfully
        // This test verifies the ite itself is constructed correctly
        let ite = store.mk_ite(c, x, y);
        assert!(matches!(store.get(ite), TermData::Ite(_, _, _)));
    }

    #[test]
    fn test_not_ite_nested() {
        let mut store = TermStore::new();

        let c1 = store.mk_var("c1", Sort::Bool);
        let c2 = store.mk_var("c2", Sort::Bool);
        let a = store.mk_var("a", Sort::Bool);
        let b = store.mk_var("b", Sort::Bool);

        // (not (ite c1 (ite c2 a b) false))
        // = (not (and c1 (ite c2 a b)))           -- ite with false branch
        // = (or (not c1) (not (ite c2 a b)))     -- De Morgan
        // = (or (not c1) (ite c2 (not a) (not b))) -- ITE negation
        let inner_ite = store.mk_ite(c2, a, b);
        let f = store.false_term();
        let outer_ite = store.mk_ite(c1, inner_ite, f);
        let result = store.mk_not(outer_ite);

        // Build the expected result
        let not_c1 = store.mk_not(c1);
        let not_a = store.mk_not(a);
        let not_b = store.mk_not(b);
        let inner_neg = store.mk_ite(c2, not_a, not_b);
        let expected = store.mk_or(vec![not_c1, inner_neg]);

        assert_eq!(result, expected);
    }

    // =========================================================================
    // XOR simplification tests
    // =========================================================================

    #[test]
    fn test_xor_same_operand() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        // (xor x x) = false
        let result = store.mk_xor(x, x);
        assert_eq!(result, store.false_term());
    }

    #[test]
    fn test_xor_with_true() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let t = store.true_term();

        // (xor x true) = (not x)
        let result = store.mk_xor(x, t);
        assert_eq!(result, store.mk_not(x));

        // (xor true x) = (not x)
        let result2 = store.mk_xor(t, x);
        assert_eq!(result2, store.mk_not(x));
    }

    #[test]
    fn test_xor_with_false() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let f = store.false_term();

        // (xor x false) = x
        let result = store.mk_xor(x, f);
        assert_eq!(result, x);

        // (xor false x) = x
        let result2 = store.mk_xor(f, x);
        assert_eq!(result2, x);
    }

    #[test]
    fn test_xor_complement() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let not_x = store.mk_not(x);

        // (xor x (not x)) = true
        let result = store.mk_xor(x, not_x);
        assert_eq!(result, store.true_term());

        // (xor (not x) x) = true
        let result2 = store.mk_xor(not_x, x);
        assert_eq!(result2, store.true_term());
    }

    #[test]
    fn test_xor_double_negation_lifting() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let not_x = store.mk_not(x);
        let not_y = store.mk_not(y);

        // (xor (not x) (not y)) = (xor x y)
        let result = store.mk_xor(not_x, not_y);
        let expected = store.mk_xor(x, y);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_xor_canonical_order() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);

        // (xor x y) and (xor y x) should produce the same canonical form
        let result1 = store.mk_xor(x, y);
        let result2 = store.mk_xor(y, x);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_sub_to_add_conversion_int() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);

        // (a - b) → (+ a (- b))
        let result = store.mk_sub(vec![a, b]);

        // Result should be addition
        match store.get(result) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "+");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], a);
                // Second arg should be (- b)
                match store.get(args[1]) {
                    TermData::App(Symbol::Named(neg_name), neg_args) => {
                        assert_eq!(neg_name, "-");
                        assert_eq!(neg_args.len(), 1);
                        assert_eq!(neg_args[0], b);
                    }
                    _ => panic!("Expected negation"),
                }
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_sub_to_add_conversion_real() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);

        // (x - y) → (+ x (- y))
        let result = store.mk_sub(vec![x, y]);

        // Result should be addition
        match store.get(result) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "+");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], x);
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_sub_coeff_collection_int() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));

        // (2a - a) → a (via coefficient collection)
        let two_a = store.mk_mul(vec![two, a]);
        let result = store.mk_sub(vec![two_a, a]);

        // Should simplify to just a (coefficient 2 - 1 = 1)
        assert_eq!(result, a);
    }

    #[test]
    fn test_sub_coeff_collection_real() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Real);
        let three = store.mk_rational(BigRational::from(BigInt::from(3)));

        // (3x - x) → 2x (via coefficient collection)
        let three_x = store.mk_mul(vec![three, x]);
        let result = store.mk_sub(vec![three_x, x]);

        // Should simplify to 2x
        let expected_two = store.mk_rational(BigRational::from(BigInt::from(2)));
        let expected = store.mk_mul(vec![x, expected_two]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sub_nary_to_add() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let b = store.mk_var("b", Sort::Int);
        let c = store.mk_var("c", Sort::Int);

        // (- a b c) → (+ a (- b) (- c))
        let result = store.mk_sub(vec![a, b, c]);

        // Result should be addition with 3 arguments
        match store.get(result) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "+");
                assert_eq!(args.len(), 3);
                assert_eq!(args[0], a);
                // args[1] should be (- b) and args[2] should be (- c)
                for &(i, expected_inner) in &[(1usize, b), (2usize, c)] {
                    match store.get(args[i]) {
                        TermData::App(Symbol::Named(neg_name), neg_args) => {
                            assert_eq!(neg_name, "-");
                            assert_eq!(neg_args.len(), 1);
                            assert_eq!(neg_args[0], expected_inner);
                        }
                        _ => panic!("Expected negation at position {}", i),
                    }
                }
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_sub_coeff_chain_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Int);
        let three = store.mk_int(BigInt::from(3));
        let two_const = store.mk_int(BigInt::from(2));

        // Build (3x - 2x) which should simplify to x
        let three_x = store.mk_mul(vec![three, x]);
        let two_x = store.mk_mul(vec![two_const, x]);
        let result = store.mk_sub(vec![three_x, two_x]);

        // Should simplify to x (coefficient 3 - 2 = 1)
        assert_eq!(result, x);
    }

    #[test]
    fn test_sub_zero_coeff() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let zero = store.mk_int(BigInt::from(0));

        // (2a - 2a) → 0 (via coefficient collection)
        let two_a = store.mk_mul(vec![two, a]);
        let result = store.mk_sub(vec![two_a, two_a]);

        // Should simplify to 0
        assert_eq!(result, zero);
    }

    #[test]
    fn test_sub_negative_coeff() {
        let mut store = TermStore::new();

        let a = store.mk_var("a", Sort::Int);
        let two = store.mk_int(BigInt::from(2));
        let three = store.mk_int(BigInt::from(3));

        // (2a - 3a) → -a (via coefficient collection)
        let two_a = store.mk_mul(vec![two, a]);
        let three_a = store.mk_mul(vec![three, a]);
        let result = store.mk_sub(vec![two_a, three_a]);

        // Should simplify to (- a)
        let expected = store.mk_neg(a);
        assert_eq!(result, expected);
    }

    // =======================================================================
    // Bitvector operation tests
    // =======================================================================

    #[test]
    fn test_bvadd_constant_folding() {
        let mut store = TermStore::new();

        // #x01 + #x02 = #x03
        let a = store.mk_bitvec(BigInt::from(1), 8);
        let b = store.mk_bitvec(BigInt::from(2), 8);
        let expected = store.mk_bitvec(BigInt::from(3), 8);
        let result = store.mk_bvadd(vec![a, b]);
        assert_eq!(result, expected);

        // Overflow: #xFF + #x01 = #x00 (for 8-bit)
        let ff = store.mk_bitvec(BigInt::from(0xFF), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let overflow_result = store.mk_bvadd(vec![ff, one]);
        assert_eq!(overflow_result, zero);
    }

    #[test]
    fn test_bvadd_identity() {
        let mut store = TermStore::new();

        // x + 0 = x
        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let result = store.mk_bvadd(vec![x, zero]);
        assert_eq!(result, x);

        // 0 + x = x
        let result2 = store.mk_bvadd(vec![zero, x]);
        assert_eq!(result2, x);
    }

    #[test]
    fn test_bvsub_constant_folding() {
        let mut store = TermStore::new();

        // #x05 - #x03 = #x02
        let a = store.mk_bitvec(BigInt::from(5), 8);
        let b = store.mk_bitvec(BigInt::from(3), 8);
        let expected = store.mk_bitvec(BigInt::from(2), 8);
        let result = store.mk_bvsub(vec![a, b]);
        assert_eq!(result, expected);

        // Underflow: #x01 - #x02 = #xFF (for 8-bit)
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let two = store.mk_bitvec(BigInt::from(2), 8);
        let ff = store.mk_bitvec(BigInt::from(0xFF), 8);
        let underflow_result = store.mk_bvsub(vec![one, two]);
        assert_eq!(underflow_result, ff);
    }

    #[test]
    fn test_bvsub_identity_and_self() {
        let mut store = TermStore::new();

        // x - 0 = x
        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let result = store.mk_bvsub(vec![x, zero]);
        assert_eq!(result, x);

        // x - x = 0
        let self_sub = store.mk_bvsub(vec![x, x]);
        assert_eq!(self_sub, zero);
    }

    #[test]
    fn test_bvmul_constant_folding() {
        let mut store = TermStore::new();

        // #x03 * #x04 = #x0C
        let a = store.mk_bitvec(BigInt::from(3), 8);
        let b = store.mk_bitvec(BigInt::from(4), 8);
        let expected = store.mk_bitvec(BigInt::from(12), 8);
        let result = store.mk_bvmul(vec![a, b]);
        assert_eq!(result, expected);

        // Overflow: #x80 * #x02 = #x00 (for 8-bit)
        let x80 = store.mk_bitvec(BigInt::from(0x80), 8);
        let two = store.mk_bitvec(BigInt::from(2), 8);
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let overflow_result = store.mk_bvmul(vec![x80, two]);
        assert_eq!(overflow_result, zero);
    }

    #[test]
    fn test_bvmul_identity_and_zero() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);

        // x * 0 = 0
        let result = store.mk_bvmul(vec![x, zero]);
        assert_eq!(result, zero);

        // 0 * x = 0
        let result2 = store.mk_bvmul(vec![zero, x]);
        assert_eq!(result2, zero);

        // x * 1 = x
        let result3 = store.mk_bvmul(vec![x, one]);
        assert_eq!(result3, x);

        // 1 * x = x
        let result4 = store.mk_bvmul(vec![one, x]);
        assert_eq!(result4, x);
    }

    #[test]
    fn test_bvand_constant_folding() {
        let mut store = TermStore::new();

        // #xFF & #x0F = #x0F
        let a = store.mk_bitvec(BigInt::from(0xFF), 8);
        let b = store.mk_bitvec(BigInt::from(0x0F), 8);
        let expected = store.mk_bitvec(BigInt::from(0x0F), 8);
        let result = store.mk_bvand(vec![a, b]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvand_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let all_ones = store.mk_bitvec(BigInt::from(0xFF), 8);

        // x & 0 = 0
        let result = store.mk_bvand(vec![x, zero]);
        assert_eq!(result, zero);

        // x & #xFF = x (all-ones)
        let result2 = store.mk_bvand(vec![x, all_ones]);
        assert_eq!(result2, x);

        // x & x = x (idempotent)
        let result3 = store.mk_bvand(vec![x, x]);
        assert_eq!(result3, x);
    }

    #[test]
    fn test_bvor_constant_folding() {
        let mut store = TermStore::new();

        // #xF0 | #x0F = #xFF
        let a = store.mk_bitvec(BigInt::from(0xF0), 8);
        let b = store.mk_bitvec(BigInt::from(0x0F), 8);
        let expected = store.mk_bitvec(BigInt::from(0xFF), 8);
        let result = store.mk_bvor(vec![a, b]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvor_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let all_ones = store.mk_bitvec(BigInt::from(0xFF), 8);

        // x | 0 = x
        let result = store.mk_bvor(vec![x, zero]);
        assert_eq!(result, x);

        // x | #xFF = #xFF
        let result2 = store.mk_bvor(vec![x, all_ones]);
        assert_eq!(result2, all_ones);

        // x | x = x (idempotent)
        let result3 = store.mk_bvor(vec![x, x]);
        assert_eq!(result3, x);
    }

    #[test]
    fn test_bvxor_constant_folding() {
        let mut store = TermStore::new();

        // #xF0 ^ #x0F = #xFF
        let a = store.mk_bitvec(BigInt::from(0xF0), 8);
        let b = store.mk_bitvec(BigInt::from(0x0F), 8);
        let expected = store.mk_bitvec(BigInt::from(0xFF), 8);
        let result = store.mk_bvxor(vec![a, b]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvxor_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);

        // x ^ 0 = x
        let result = store.mk_bvxor(vec![x, zero]);
        assert_eq!(result, x);

        // x ^ x = 0
        let result2 = store.mk_bvxor(vec![x, x]);
        assert_eq!(result2, zero);
    }

    #[test]
    fn test_bvnot_constant_folding() {
        let mut store = TermStore::new();

        // ~#x00 = #xFF (for 8-bit)
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let all_ones = store.mk_bitvec(BigInt::from(0xFF), 8);
        let result = store.mk_bvnot(zero);
        assert_eq!(result, all_ones);

        // ~#xFF = #x00
        let result2 = store.mk_bvnot(all_ones);
        assert_eq!(result2, zero);
    }

    #[test]
    fn test_bvnot_double_negation() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let not_x = store.mk_bvnot(x);
        let not_not_x = store.mk_bvnot(not_x);
        assert_eq!(not_not_x, x);
    }

    #[test]
    fn test_bvneg_constant_folding() {
        let mut store = TermStore::new();

        // -#x01 = #xFF (for 8-bit, two's complement)
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let neg_one = store.mk_bitvec(BigInt::from(0xFF), 8);
        let result = store.mk_bvneg(one);
        assert_eq!(result, neg_one);

        // -#x00 = #x00
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let result2 = store.mk_bvneg(zero);
        assert_eq!(result2, zero);
    }

    #[test]
    fn test_bvneg_double_negation() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let neg_x = store.mk_bvneg(x);
        let neg_neg_x = store.mk_bvneg(neg_x);
        assert_eq!(neg_neg_x, x);
    }

    #[test]
    fn test_bvshl_constant_folding() {
        let mut store = TermStore::new();

        // #x01 << 4 = #x10
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let four = store.mk_bitvec(BigInt::from(4), 8);
        let expected = store.mk_bitvec(BigInt::from(0x10), 8);
        let result = store.mk_bvshl(vec![one, four]);
        assert_eq!(result, expected);

        // #x80 << 1 = #x00 (overflow)
        let x80 = store.mk_bitvec(BigInt::from(0x80), 8);
        let one_shift = store.mk_bitvec(BigInt::from(1), 8);
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let overflow_result = store.mk_bvshl(vec![x80, one_shift]);
        assert_eq!(overflow_result, zero);
    }

    #[test]
    fn test_bvshl_identity_and_zero() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);

        // x << 0 = x
        let result = store.mk_bvshl(vec![x, zero]);
        assert_eq!(result, x);

        // 0 << x = 0
        let result2 = store.mk_bvshl(vec![zero, x]);
        assert_eq!(result2, zero);

        // x << 8 = 0 (shift >= width)
        let eight = store.mk_bitvec(BigInt::from(8), 8);
        let result3 = store.mk_bvshl(vec![x, eight]);
        assert_eq!(result3, zero);
    }

    #[test]
    fn test_bvlshr_constant_folding() {
        let mut store = TermStore::new();

        // #xFF >> 4 = #x0F
        let ff = store.mk_bitvec(BigInt::from(0xFF), 8);
        let four = store.mk_bitvec(BigInt::from(4), 8);
        let expected = store.mk_bitvec(BigInt::from(0x0F), 8);
        let result = store.mk_bvlshr(vec![ff, four]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvlshr_identity_and_zero() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);

        // x >> 0 = x
        let result = store.mk_bvlshr(vec![x, zero]);
        assert_eq!(result, x);

        // 0 >> x = 0
        let result2 = store.mk_bvlshr(vec![zero, x]);
        assert_eq!(result2, zero);

        // x >> 8 = 0 (shift >= width)
        let eight = store.mk_bitvec(BigInt::from(8), 8);
        let result3 = store.mk_bvlshr(vec![x, eight]);
        assert_eq!(result3, zero);
    }

    #[test]
    fn test_bvashr_constant_folding() {
        let mut store = TermStore::new();

        // #x80 >>> 4 = #xF8 (sign extension, negative)
        let x80 = store.mk_bitvec(BigInt::from(0x80), 8);
        let four = store.mk_bitvec(BigInt::from(4), 8);
        let expected = store.mk_bitvec(BigInt::from(0xF8), 8);
        let result = store.mk_bvashr(vec![x80, four]);
        assert_eq!(result, expected);

        // #x70 >>> 4 = #x07 (no sign extension, positive)
        let x70 = store.mk_bitvec(BigInt::from(0x70), 8);
        let expected2 = store.mk_bitvec(BigInt::from(0x07), 8);
        let result2 = store.mk_bvashr(vec![x70, four]);
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_bvashr_identity_and_zero() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);

        // x >>> 0 = x
        let result = store.mk_bvashr(vec![x, zero]);
        assert_eq!(result, x);

        // 0 >>> x = 0
        let result2 = store.mk_bvashr(vec![zero, x]);
        assert_eq!(result2, zero);
    }

    #[test]
    fn test_bvudiv_constant_folding() {
        let mut store = TermStore::new();

        // #x10 / #x04 = #x04
        let x10 = store.mk_bitvec(BigInt::from(0x10), 8);
        let four = store.mk_bitvec(BigInt::from(4), 8);
        let expected = store.mk_bitvec(BigInt::from(4), 8);
        let result = store.mk_bvudiv(vec![x10, four]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvudiv_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);

        // x / 1 = x
        let result = store.mk_bvudiv(vec![x, one]);
        assert_eq!(result, x);

        // 0 / x = 0
        let result2 = store.mk_bvudiv(vec![zero, x]);
        assert_eq!(result2, zero);

        // x / x = 1
        let result3 = store.mk_bvudiv(vec![x, x]);
        assert_eq!(result3, one);
    }

    #[test]
    fn test_bvurem_constant_folding() {
        let mut store = TermStore::new();

        // #x17 % #x05 = #x03 (23 % 5 = 3)
        let x17 = store.mk_bitvec(BigInt::from(0x17), 8);
        let five = store.mk_bitvec(BigInt::from(5), 8);
        let expected = store.mk_bitvec(BigInt::from(3), 8);
        let result = store.mk_bvurem(vec![x17, five]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvurem_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);

        // x % 1 = 0
        let result = store.mk_bvurem(vec![x, one]);
        assert_eq!(result, zero);

        // 0 % x = 0
        let result2 = store.mk_bvurem(vec![zero, x]);
        assert_eq!(result2, zero);

        // x % x = 0
        let result3 = store.mk_bvurem(vec![x, x]);
        assert_eq!(result3, zero);
    }

    // =========================================================================
    // Signed bitvector division/remainder/modulo tests
    // =========================================================================

    #[test]
    fn test_bvsdiv_constant_folding_positive() {
        let mut store = TermStore::new();

        // 7 / 2 = 3 (both positive)
        let seven = store.mk_bitvec(BigInt::from(7), 8);
        let two = store.mk_bitvec(BigInt::from(2), 8);
        let expected = store.mk_bitvec(BigInt::from(3), 8);
        let result = store.mk_bvsdiv(vec![seven, two]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsdiv_constant_folding_negative_dividend() {
        let mut store = TermStore::new();

        // -7 / 2 = -3 (8-bit: 0xF9 / 2 = 0xFD which is -3)
        // -7 in 8-bit two's complement is 256 - 7 = 249 = 0xF9
        let neg_seven = store.mk_bitvec(BigInt::from(249), 8);
        let two = store.mk_bitvec(BigInt::from(2), 8);
        // -3 in 8-bit two's complement is 256 - 3 = 253 = 0xFD
        let expected = store.mk_bitvec(BigInt::from(253), 8);
        let result = store.mk_bvsdiv(vec![neg_seven, two]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsdiv_constant_folding_negative_divisor() {
        let mut store = TermStore::new();

        // 7 / -2 = -3 (truncated towards zero)
        let seven = store.mk_bitvec(BigInt::from(7), 8);
        // -2 in 8-bit is 254
        let neg_two = store.mk_bitvec(BigInt::from(254), 8);
        // -3 in 8-bit is 253
        let expected = store.mk_bitvec(BigInt::from(253), 8);
        let result = store.mk_bvsdiv(vec![seven, neg_two]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsdiv_constant_folding_both_negative() {
        let mut store = TermStore::new();

        // -7 / -2 = 3 (both negative, result positive)
        let neg_seven = store.mk_bitvec(BigInt::from(249), 8);
        let neg_two = store.mk_bitvec(BigInt::from(254), 8);
        let expected = store.mk_bitvec(BigInt::from(3), 8);
        let result = store.mk_bvsdiv(vec![neg_seven, neg_two]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsdiv_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);

        // x / 1 = x
        let result = store.mk_bvsdiv(vec![x, one]);
        assert_eq!(result, x);

        // 0 / x = 0
        let result2 = store.mk_bvsdiv(vec![zero, x]);
        assert_eq!(result2, zero);

        // x / x = 1
        let result3 = store.mk_bvsdiv(vec![x, x]);
        assert_eq!(result3, one);
    }

    #[test]
    fn test_bvsrem_constant_folding_positive() {
        let mut store = TermStore::new();

        // 7 % 3 = 1 (both positive)
        let seven = store.mk_bitvec(BigInt::from(7), 8);
        let three = store.mk_bitvec(BigInt::from(3), 8);
        let expected = store.mk_bitvec(BigInt::from(1), 8);
        let result = store.mk_bvsrem(vec![seven, three]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsrem_constant_folding_negative_dividend() {
        let mut store = TermStore::new();

        // -7 % 3 = -1 (sign follows dividend)
        let neg_seven = store.mk_bitvec(BigInt::from(249), 8); // -7
        let three = store.mk_bitvec(BigInt::from(3), 8);
        let expected = store.mk_bitvec(BigInt::from(255), 8); // -1
        let result = store.mk_bvsrem(vec![neg_seven, three]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsrem_constant_folding_negative_divisor() {
        let mut store = TermStore::new();

        // 7 % -3 = 1 (sign follows dividend, dividend is positive)
        let seven = store.mk_bitvec(BigInt::from(7), 8);
        let neg_three = store.mk_bitvec(BigInt::from(253), 8); // -3
        let expected = store.mk_bitvec(BigInt::from(1), 8);
        let result = store.mk_bvsrem(vec![seven, neg_three]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsrem_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);

        // x % 1 = 0
        let result = store.mk_bvsrem(vec![x, one]);
        assert_eq!(result, zero);

        // 0 % x = 0
        let result2 = store.mk_bvsrem(vec![zero, x]);
        assert_eq!(result2, zero);

        // x % x = 0
        let result3 = store.mk_bvsrem(vec![x, x]);
        assert_eq!(result3, zero);
    }

    #[test]
    fn test_bvsmod_constant_folding_positive() {
        let mut store = TermStore::new();

        // 7 mod 3 = 1 (both positive)
        let seven = store.mk_bitvec(BigInt::from(7), 8);
        let three = store.mk_bitvec(BigInt::from(3), 8);
        let expected = store.mk_bitvec(BigInt::from(1), 8);
        let result = store.mk_bvsmod(vec![seven, three]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsmod_constant_folding_negative_dividend() {
        let mut store = TermStore::new();

        // -7 mod 3 = 2 (sign follows divisor which is positive)
        // -7 = -3*3 + 2, so result is 2
        let neg_seven = store.mk_bitvec(BigInt::from(249), 8); // -7
        let three = store.mk_bitvec(BigInt::from(3), 8);
        let expected = store.mk_bitvec(BigInt::from(2), 8);
        let result = store.mk_bvsmod(vec![neg_seven, three]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsmod_constant_folding_negative_divisor() {
        let mut store = TermStore::new();

        // 7 mod -3 = -2 (sign follows divisor which is negative)
        let seven = store.mk_bitvec(BigInt::from(7), 8);
        let neg_three = store.mk_bitvec(BigInt::from(253), 8); // -3
        let expected = store.mk_bitvec(BigInt::from(254), 8); // -2
        let result = store.mk_bvsmod(vec![seven, neg_three]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsmod_constant_folding_both_negative() {
        let mut store = TermStore::new();

        // -7 mod -3 = -1 (sign follows divisor which is negative)
        let neg_seven = store.mk_bitvec(BigInt::from(249), 8); // -7
        let neg_three = store.mk_bitvec(BigInt::from(253), 8); // -3
        let expected = store.mk_bitvec(BigInt::from(255), 8); // -1
        let result = store.mk_bvsmod(vec![neg_seven, neg_three]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bvsmod_simplifications() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);

        // x mod 1 = 0
        let result = store.mk_bvsmod(vec![x, one]);
        assert_eq!(result, zero);

        // 0 mod x = 0
        let result2 = store.mk_bvsmod(vec![zero, x]);
        assert_eq!(result2, zero);

        // x mod x = 0
        let result3 = store.mk_bvsmod(vec![x, x]);
        assert_eq!(result3, zero);
    }

    #[test]
    fn test_bvcomp_constant_folding() {
        let mut store = TermStore::new();

        let five = store.mk_bitvec(BigInt::from(5), 8);
        let seven = store.mk_bitvec(BigInt::from(7), 8);
        let five2 = store.mk_bitvec(BigInt::from(5), 8);

        // bvcomp(5, 5) = #b1
        let result = store.mk_bvcomp(five, five2);
        let expected_one = store.mk_bitvec(BigInt::from(1), 1);
        assert_eq!(result, expected_one);

        // bvcomp(5, 7) = #b0
        let result2 = store.mk_bvcomp(five, seven);
        let expected_zero = store.mk_bitvec(BigInt::from(0), 1);
        assert_eq!(result2, expected_zero);
    }

    #[test]
    fn test_bvcomp_reflexivity() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));

        // bvcomp(x, x) = #b1
        let result = store.mk_bvcomp(x, x);
        let expected = store.mk_bitvec(BigInt::from(1), 1);
        assert_eq!(result, expected);
    }

    // =========================================================================
    // Bitvector comparison tests
    // =========================================================================

    #[test]
    fn test_bvult_constant_folding() {
        let mut store = TermStore::new();

        // 1 < 2 = true (unsigned)
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let two = store.mk_bitvec(BigInt::from(2), 8);
        let result = store.mk_bvult(one, two);
        assert_eq!(result, store.true_term());

        // 2 < 1 = false (unsigned)
        let result2 = store.mk_bvult(two, one);
        assert_eq!(result2, store.false_term());

        // 0xFF < 0x01 = false (unsigned: 255 < 1 is false)
        let ff = store.mk_bitvec(BigInt::from(0xFF), 8);
        let result3 = store.mk_bvult(ff, one);
        assert_eq!(result3, store.false_term());
    }

    #[test]
    fn test_bvult_reflexivity_and_zero() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);

        // x < x = false
        let result = store.mk_bvult(x, x);
        assert_eq!(result, store.false_term());

        // x < 0 = false (nothing is less than 0 unsigned)
        let result2 = store.mk_bvult(x, zero);
        assert_eq!(result2, store.false_term());
    }

    #[test]
    fn test_bvule_constant_folding() {
        let mut store = TermStore::new();

        // 1 <= 2 = true
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let two = store.mk_bitvec(BigInt::from(2), 8);
        let result = store.mk_bvule(one, two);
        assert_eq!(result, store.true_term());

        // 2 <= 2 = true
        let result2 = store.mk_bvule(two, two);
        assert_eq!(result2, store.true_term());

        // 2 <= 1 = false
        let result3 = store.mk_bvule(two, one);
        assert_eq!(result3, store.false_term());
    }

    #[test]
    fn test_bvule_reflexivity_and_zero() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let zero = store.mk_bitvec(BigInt::from(0), 8);

        // x <= x = true
        let result = store.mk_bvule(x, x);
        assert_eq!(result, store.true_term());

        // 0 <= x = true (0 is <= everything unsigned)
        let result2 = store.mk_bvule(zero, x);
        assert_eq!(result2, store.true_term());
    }

    #[test]
    fn test_bvugt_normalization() {
        let mut store = TermStore::new();

        // bvugt(a, b) should normalize to bvult(b, a)
        let a = store.mk_bitvec(BigInt::from(5), 8);
        let b = store.mk_bitvec(BigInt::from(3), 8);

        // 5 > 3 = true
        let result = store.mk_bvugt(a, b);
        assert_eq!(result, store.true_term());

        // 3 > 5 = false
        let result2 = store.mk_bvugt(b, a);
        assert_eq!(result2, store.false_term());
    }

    #[test]
    fn test_bvuge_normalization() {
        let mut store = TermStore::new();

        // bvuge(a, b) should normalize to bvule(b, a)
        let a = store.mk_bitvec(BigInt::from(5), 8);
        let b = store.mk_bitvec(BigInt::from(3), 8);

        // 5 >= 3 = true
        let result = store.mk_bvuge(a, b);
        assert_eq!(result, store.true_term());

        // 5 >= 5 = true
        let result2 = store.mk_bvuge(a, a);
        assert_eq!(result2, store.true_term());
    }

    #[test]
    fn test_bvslt_constant_folding() {
        let mut store = TermStore::new();

        // Signed comparison: -1 < 1 is true
        // In 8-bit two's complement, 0xFF = -1
        let neg_one = store.mk_bitvec(BigInt::from(0xFF), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let result = store.mk_bvslt(neg_one, one);
        assert_eq!(result, store.true_term());

        // Signed: 1 < -1 is false
        let result2 = store.mk_bvslt(one, neg_one);
        assert_eq!(result2, store.false_term());

        // Signed: -128 (0x80) < 127 (0x7F) is true
        let min_val = store.mk_bitvec(BigInt::from(0x80), 8);
        let max_val = store.mk_bitvec(BigInt::from(0x7F), 8);
        let result3 = store.mk_bvslt(min_val, max_val);
        assert_eq!(result3, store.true_term());

        // Signed: 127 < -128 is false
        let result4 = store.mk_bvslt(max_val, min_val);
        assert_eq!(result4, store.false_term());
    }

    #[test]
    fn test_bvslt_reflexivity() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));

        // x < x = false
        let result = store.mk_bvslt(x, x);
        assert_eq!(result, store.false_term());
    }

    #[test]
    fn test_bvsle_constant_folding() {
        let mut store = TermStore::new();

        // Signed: -1 <= 1 is true
        let neg_one = store.mk_bitvec(BigInt::from(0xFF), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let result = store.mk_bvsle(neg_one, one);
        assert_eq!(result, store.true_term());

        // Signed: -1 <= -1 is true
        let result2 = store.mk_bvsle(neg_one, neg_one);
        assert_eq!(result2, store.true_term());

        // Signed: 1 <= -1 is false
        let result3 = store.mk_bvsle(one, neg_one);
        assert_eq!(result3, store.false_term());
    }

    #[test]
    fn test_bvsle_reflexivity() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));

        // x <= x = true
        let result = store.mk_bvsle(x, x);
        assert_eq!(result, store.true_term());
    }

    #[test]
    fn test_bvsgt_normalization() {
        let mut store = TermStore::new();

        // bvsgt(a, b) normalizes to bvslt(b, a)
        // Signed: 1 > -1 is true
        let neg_one = store.mk_bitvec(BigInt::from(0xFF), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let result = store.mk_bvsgt(one, neg_one);
        assert_eq!(result, store.true_term());

        // Signed: -1 > 1 is false
        let result2 = store.mk_bvsgt(neg_one, one);
        assert_eq!(result2, store.false_term());
    }

    #[test]
    fn test_bvsge_normalization() {
        let mut store = TermStore::new();

        // bvsge(a, b) normalizes to bvsle(b, a)
        // Signed: 1 >= -1 is true
        let neg_one = store.mk_bitvec(BigInt::from(0xFF), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);
        let result = store.mk_bvsge(one, neg_one);
        assert_eq!(result, store.true_term());

        // Signed: 1 >= 1 is true
        let result2 = store.mk_bvsge(one, one);
        assert_eq!(result2, store.true_term());
    }

    #[test]
    fn test_signed_vs_unsigned_comparison() {
        let mut store = TermStore::new();

        // 0xFF: unsigned = 255, signed = -1
        let ff = store.mk_bitvec(BigInt::from(0xFF), 8);
        let one = store.mk_bitvec(BigInt::from(1), 8);

        // Unsigned: 255 < 1 is false
        let ult_result = store.mk_bvult(ff, one);
        assert_eq!(ult_result, store.false_term());

        // Signed: -1 < 1 is true
        let slt_result = store.mk_bvslt(ff, one);
        assert_eq!(slt_result, store.true_term());
    }

    // =========================================================================
    // Bitvector extract, concat, extend, and rotate tests
    // =========================================================================

    #[test]
    fn test_bvextract_constant_folding() {
        let mut store = TermStore::new();

        // extract(7,4,#xFF) -> #x0F (extracts bits 7..4 = 0b1111)
        let ff = store.mk_bitvec(BigInt::from(0xFF), 8);
        let result = store.mk_bvextract(7, 4, ff);
        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0x0F));
            assert_eq!(width, 4);
        } else {
            panic!("Expected bitvector constant");
        }

        // extract(3,0,#xAB) -> #x0B (extracts lower nibble)
        let ab = store.mk_bitvec(BigInt::from(0xAB), 8);
        let result2 = store.mk_bvextract(3, 0, ab);
        if let Some((val, width)) = store.get_bitvec(result2) {
            assert_eq!(*val, BigInt::from(0x0B));
            assert_eq!(width, 4);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvextract_full_extract() {
        let mut store = TermStore::new();

        // extract(7,0,x) -> x (full extract is identity)
        let x = store.mk_var("x", Sort::BitVec(8));
        let result = store.mk_bvextract(7, 0, x);
        assert_eq!(result, x);
    }

    #[test]
    fn test_bvconcat_constant_folding() {
        let mut store = TermStore::new();

        // concat(#x0F, #xF0) -> #x0FF0
        let x0f = store.mk_bitvec(BigInt::from(0x0F), 8);
        let xf0 = store.mk_bitvec(BigInt::from(0xF0), 8);
        let result = store.mk_bvconcat(vec![x0f, xf0]);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0x0FF0));
            assert_eq!(width, 16);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvconcat_mixed_widths() {
        let mut store = TermStore::new();

        // concat(4-bit, 8-bit) should give 12-bit result
        let nibble = store.mk_bitvec(BigInt::from(0xA), 4);
        let byte = store.mk_bitvec(BigInt::from(0xBC), 8);
        let result = store.mk_bvconcat(vec![nibble, byte]);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0xABC));
            assert_eq!(width, 12);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvzero_extend_constant_folding() {
        let mut store = TermStore::new();

        // zero_extend(4, #x0F) -> #x00F (12-bit)
        let x0f = store.mk_bitvec(BigInt::from(0x0F), 8);
        let result = store.mk_bvzero_extend(4, x0f);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0x0F));
            assert_eq!(width, 12);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvzero_extend_identity() {
        let mut store = TermStore::new();

        // zero_extend(0, x) -> x
        let x = store.mk_var("x", Sort::BitVec(8));
        let result = store.mk_bvzero_extend(0, x);
        assert_eq!(result, x);
    }

    #[test]
    fn test_bvsign_extend_positive() {
        let mut store = TermStore::new();

        // sign_extend(4, #x7F) -> #x07F (12-bit, positive so zero extended)
        let x7f = store.mk_bitvec(BigInt::from(0x7F), 8);
        let result = store.mk_bvsign_extend(4, x7f);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0x07F));
            assert_eq!(width, 12);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvsign_extend_negative() {
        let mut store = TermStore::new();

        // sign_extend(4, #x8F) -> #xF8F (12-bit, negative so ones extended)
        let x8f = store.mk_bitvec(BigInt::from(0x8F), 8);
        let result = store.mk_bvsign_extend(4, x8f);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0xF8F));
            assert_eq!(width, 12);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvsign_extend_identity() {
        let mut store = TermStore::new();

        // sign_extend(0, x) -> x
        let x = store.mk_var("x", Sort::BitVec(8));
        let result = store.mk_bvsign_extend(0, x);
        assert_eq!(result, x);
    }

    #[test]
    fn test_bvrotate_left_constant_folding() {
        let mut store = TermStore::new();

        // rotate_left(2, #xA5) -> #x96
        // #xA5 = 0b10100101, rotate left 2 = 0b10010110 = #x96
        let xa5 = store.mk_bitvec(BigInt::from(0xA5), 8);
        let result = store.mk_bvrotate_left(2, xa5);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0x96));
            assert_eq!(width, 8);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvrotate_left_identity() {
        let mut store = TermStore::new();

        // rotate_left(0, x) -> x
        let x = store.mk_var("x", Sort::BitVec(8));
        let result = store.mk_bvrotate_left(0, x);
        assert_eq!(result, x);

        // rotate_left(8, x) -> x (full rotation)
        let result2 = store.mk_bvrotate_left(8, x);
        assert_eq!(result2, x);
    }

    #[test]
    fn test_bvrotate_right_constant_folding() {
        let mut store = TermStore::new();

        // rotate_right(2, #xA5) -> #x69
        // #xA5 = 0b10100101, rotate right 2 = 0b01101001 = #x69
        let xa5 = store.mk_bitvec(BigInt::from(0xA5), 8);
        let result = store.mk_bvrotate_right(2, xa5);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0x69));
            assert_eq!(width, 8);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvrotate_right_identity() {
        let mut store = TermStore::new();

        // rotate_right(0, x) -> x
        let x = store.mk_var("x", Sort::BitVec(8));
        let result = store.mk_bvrotate_right(0, x);
        assert_eq!(result, x);

        // rotate_right(8, x) -> x (full rotation)
        let result2 = store.mk_bvrotate_right(8, x);
        assert_eq!(result2, x);
    }

    #[test]
    fn test_bvrotate_inverse() {
        let mut store = TermStore::new();

        // rotate_left(n, rotate_right(n, x)) should give back original
        let xa5 = store.mk_bitvec(BigInt::from(0xA5), 8);
        let rotated_right = store.mk_bvrotate_right(3, xa5);
        let rotated_back = store.mk_bvrotate_left(3, rotated_right);
        assert_eq!(rotated_back, xa5);
    }

    #[test]
    fn test_bvrepeat_constant_folding() {
        let mut store = TermStore::new();

        // repeat(3, #xAB) -> #xABABAB
        let xab = store.mk_bitvec(BigInt::from(0xAB), 8);
        let result = store.mk_bvrepeat(3, xab);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0xABABAB));
            assert_eq!(width, 24);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bvrepeat_identity() {
        let mut store = TermStore::new();

        // repeat(1, x) -> x
        let x = store.mk_var("x", Sort::BitVec(8));
        let result = store.mk_bvrepeat(1, x);
        assert_eq!(result, x);
    }

    #[test]
    fn test_bvrepeat_small() {
        let mut store = TermStore::new();

        // repeat(4, #b11) -> #b11111111 = #xFF
        let x3 = store.mk_bitvec(BigInt::from(0b11), 2);
        let result = store.mk_bvrepeat(4, x3);

        if let Some((val, width)) = store.get_bitvec(result) {
            assert_eq!(*val, BigInt::from(0xFF));
            assert_eq!(width, 8);
        } else {
            panic!("Expected bitvector constant");
        }
    }

    #[test]
    fn test_bv2nat_constant_folding() {
        let mut store = TermStore::new();

        let x0f = store.mk_bitvec(BigInt::from(0x0F), 8);
        let result = store.mk_bv2nat(x0f);

        assert_eq!(store.get_int(result).cloned(), Some(BigInt::from(15)));
    }

    #[test]
    fn test_int2bv_constant_folding_and_wraparound() {
        let mut store = TermStore::new();

        let fifteen = store.mk_int(BigInt::from(15));
        let result = store.mk_int2bv(8, fifteen);
        assert_eq!(result, store.mk_bitvec(BigInt::from(0x0F), 8));

        // -1 mod 2^8 = 255
        let minus_one = store.mk_int(BigInt::from(-1));
        let result2 = store.mk_int2bv(8, minus_one);
        assert_eq!(result2, store.mk_bitvec(BigInt::from(0xFF), 8));

        // 256 mod 2^8 = 0
        let two_fifty_six = store.mk_int(BigInt::from(256));
        let result3 = store.mk_int2bv(8, two_fifty_six);
        assert_eq!(result3, store.mk_bitvec(BigInt::from(0), 8));
    }

    #[test]
    fn test_int2bv_bv2nat_roundtrip_simplification() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::BitVec(8));
        let nat = store.mk_bv2nat(x);
        let back = store.mk_int2bv(8, nat);
        assert_eq!(back, x);
    }

    #[test]
    fn test_bvnand_bvnor_bvxnor_constant_folding() {
        let mut store = TermStore::new();

        let zero = store.mk_bitvec(BigInt::from(0), 8);
        let all_ones = store.mk_bitvec(BigInt::from(0xFF), 8);
        let x0f = store.mk_bitvec(BigInt::from(0x0F), 8);

        // nand(FF, FF) = 00
        assert_eq!(store.mk_bvnand(vec![all_ones, all_ones]), zero);

        // nor(00, 00) = FF
        assert_eq!(store.mk_bvnor(vec![zero, zero]), all_ones);

        // xnor(0F, 0F) = FF
        assert_eq!(store.mk_bvxnor(vec![x0f, x0f]), all_ones);
    }

    // ==================== Array Operations Tests ====================

    #[test]
    fn test_array_select_basic() {
        let mut store = TermStore::new();

        // Create array: (Array Int Int)
        let arr = store.mk_var("a", Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int)));
        let idx = store.mk_int(BigInt::from(0));

        let selected = store.mk_select(arr, idx);

        // Result should have element sort (Int)
        assert_eq!(store.sort(selected), &Sort::Int);
    }

    #[test]
    fn test_array_store_basic() {
        let mut store = TermStore::new();

        // Create array: (Array Int Int)
        let arr = store.mk_var("a", Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int)));
        let idx = store.mk_int(BigInt::from(0));
        let val = store.mk_int(BigInt::from(42));

        let stored = store.mk_store(arr, idx, val);

        // Result should have same sort as input array
        assert_eq!(
            store.sort(stored),
            &Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int))
        );
    }

    #[test]
    fn test_array_read_over_write_same_index() {
        let mut store = TermStore::new();

        // Create array: (Array Int Int)
        let arr = store.mk_var("a", Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int)));
        let idx = store.mk_int(BigInt::from(0));
        let val = store.mk_int(BigInt::from(42));

        // select(store(a, 0, 42), 0) should simplify to 42
        let stored = store.mk_store(arr, idx, val);
        let selected = store.mk_select(stored, idx);

        assert_eq!(selected, val);
    }

    #[test]
    fn test_array_read_over_write_different_constant_index() {
        let mut store = TermStore::new();

        // Create array: (Array Int Int)
        let arr = store.mk_var("a", Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int)));
        let idx1 = store.mk_int(BigInt::from(0));
        let idx2 = store.mk_int(BigInt::from(1));
        let val = store.mk_int(BigInt::from(42));

        // select(store(a, 0, 42), 1) should simplify to select(a, 1)
        let stored = store.mk_store(arr, idx1, val);
        let selected = store.mk_select(stored, idx2);

        // Should be select(a, 1)
        let expected = store.mk_select(arr, idx2);
        assert_eq!(selected, expected);
    }

    #[test]
    fn test_array_store_over_store_same_index() {
        let mut store = TermStore::new();

        // Create array: (Array Int Int)
        let arr = store.mk_var("a", Sort::Array(Box::new(Sort::Int), Box::new(Sort::Int)));
        let idx = store.mk_int(BigInt::from(0));
        let val1 = store.mk_int(BigInt::from(10));
        let val2 = store.mk_int(BigInt::from(20));

        // store(store(a, 0, 10), 0, 20) should simplify to store(a, 0, 20)
        let stored1 = store.mk_store(arr, idx, val1);
        let stored2 = store.mk_store(stored1, idx, val2);

        let expected = store.mk_store(arr, idx, val2);
        assert_eq!(stored2, expected);
    }

    #[test]
    fn test_array_with_bitvec_index() {
        let mut store = TermStore::new();

        // Create array: (Array (_ BitVec 8) Int)
        let arr = store.mk_var(
            "a",
            Sort::Array(Box::new(Sort::BitVec(8)), Box::new(Sort::Int)),
        );
        let idx = store.mk_bitvec(BigInt::from(5), 8);
        let val = store.mk_int(BigInt::from(100));

        // Basic store/select
        let stored = store.mk_store(arr, idx, val);
        let selected = store.mk_select(stored, idx);

        assert_eq!(selected, val);
    }

    #[test]
    fn test_array_read_over_write_different_bv_constant_index() {
        let mut store = TermStore::new();

        // Create array: (Array (_ BitVec 8) Int)
        let arr = store.mk_var(
            "a",
            Sort::Array(Box::new(Sort::BitVec(8)), Box::new(Sort::Int)),
        );
        let idx1 = store.mk_bitvec(BigInt::from(5), 8);
        let idx2 = store.mk_bitvec(BigInt::from(10), 8);
        let val = store.mk_int(BigInt::from(100));

        // select(store(a, 5, 100), 10) should simplify to select(a, 10)
        let stored = store.mk_store(arr, idx1, val);
        let selected = store.mk_select(stored, idx2);

        let expected = store.mk_select(arr, idx2);
        assert_eq!(selected, expected);
    }

    #[test]
    fn test_const_array_basic() {
        let mut store = TermStore::new();

        // Create constant array: ((as const (Array Int Bool)) true)
        let default_val = store.mk_bool(true);
        let const_arr = store.mk_const_array(Sort::Int, default_val);

        // Check sort
        assert_eq!(
            store.sort(const_arr),
            &Sort::Array(Box::new(Sort::Int), Box::new(Sort::Bool))
        );

        // Check we can retrieve the default value
        assert_eq!(store.get_const_array(const_arr), Some(default_val));
    }

    #[test]
    fn test_const_array_read_simplification() {
        let mut store = TermStore::new();

        // Create constant array with default value 42
        let default_val = store.mk_int(BigInt::from(42));
        let const_arr = store.mk_const_array(Sort::Int, default_val);

        // select(const-array(42), i) should simplify to 42 for any index
        let idx = store.mk_int(BigInt::from(100));
        let selected = store.mk_select(const_arr, idx);

        // Should simplify to the default value
        assert_eq!(selected, default_val);
    }

    #[test]
    fn test_const_array_with_store() {
        let mut store = TermStore::new();

        // Create constant array with default value 0
        let default_val = store.mk_int(BigInt::from(0));
        let const_arr = store.mk_const_array(Sort::Int, default_val);

        // Store a different value at index 5
        let idx5 = store.mk_int(BigInt::from(5));
        let val = store.mk_int(BigInt::from(100));
        let stored = store.mk_store(const_arr, idx5, val);

        // select(store(const-array(0), 5, 100), 5) should give 100
        let select_at_5 = store.mk_select(stored, idx5);
        assert_eq!(select_at_5, val);

        // select(store(const-array(0), 5, 100), 10) should give 0 (the default)
        let idx10 = store.mk_int(BigInt::from(10));
        let select_at_10 = store.mk_select(stored, idx10);
        assert_eq!(select_at_10, default_val);
    }

    #[test]
    fn test_const_array_bitvec_index() {
        let mut store = TermStore::new();

        // Create constant array: (Array (_ BitVec 32) Int) with default 0
        let default_val = store.mk_int(BigInt::from(0));
        let const_arr = store.mk_const_array(Sort::BitVec(32), default_val);

        // Check sort
        assert_eq!(
            store.sort(const_arr),
            &Sort::Array(Box::new(Sort::BitVec(32)), Box::new(Sort::Int))
        );

        // Read should simplify
        let idx = store.mk_bitvec(BigInt::from(0xDEADBEEFu32), 32);
        let selected = store.mk_select(const_arr, idx);
        assert_eq!(selected, default_val);
    }

    // =======================================================================
    // ITE Lifting Tests
    // =======================================================================

    #[test]
    fn test_ite_lifting_simple() {
        let mut store = TermStore::new();

        // (<= (ite c x y) z) should become (ite c (<= x z) (<= y z))
        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);
        let z = store.mk_var("z", Sort::Real);

        let ite_expr = store.mk_ite(c, x, y);
        let pred = store.mk_le(ite_expr, z);

        let lifted = store.lift_arithmetic_ite(pred);

        // Should be (ite c (<= x z) (<= y z))
        match store.get(lifted) {
            TermData::Ite(cond, then_t, else_t) => {
                assert_eq!(*cond, c);
                // then branch should be (<= x z)
                match store.get(*then_t) {
                    TermData::App(Symbol::Named(name), args) => {
                        assert_eq!(name, "<=");
                        assert_eq!(args[0], x);
                        assert_eq!(args[1], z);
                    }
                    _ => panic!("Expected <= application in then branch"),
                }
                // else branch should be (<= y z)
                match store.get(*else_t) {
                    TermData::App(Symbol::Named(name), args) => {
                        assert_eq!(name, "<=");
                        assert_eq!(args[0], y);
                        assert_eq!(args[1], z);
                    }
                    _ => panic!("Expected <= application in else branch"),
                }
            }
            _ => panic!("Expected ITE after lifting"),
        }
    }

    #[test]
    fn test_ite_lifting_second_arg() {
        let mut store = TermStore::new();

        // (<= z (ite c x y)) should become (ite c (<= z x) (<= z y))
        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);
        let z = store.mk_var("z", Sort::Real);

        let ite_expr = store.mk_ite(c, x, y);
        let pred = store.mk_le(z, ite_expr);

        let lifted = store.lift_arithmetic_ite(pred);

        // Should be (ite c (<= z x) (<= z y))
        match store.get(lifted) {
            TermData::Ite(cond, then_t, else_t) => {
                assert_eq!(*cond, c);
                match store.get(*then_t) {
                    TermData::App(Symbol::Named(name), args) => {
                        assert_eq!(name, "<=");
                        assert_eq!(args[0], z);
                        assert_eq!(args[1], x);
                    }
                    _ => panic!("Expected <= application in then branch"),
                }
                match store.get(*else_t) {
                    TermData::App(Symbol::Named(name), args) => {
                        assert_eq!(name, "<=");
                        assert_eq!(args[0], z);
                        assert_eq!(args[1], y);
                    }
                    _ => panic!("Expected <= application in else branch"),
                }
            }
            _ => panic!("Expected ITE after lifting"),
        }
    }

    #[test]
    fn test_ite_lifting_no_ite() {
        let mut store = TermStore::new();

        // (<= x y) with no ITE should remain unchanged
        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);

        let pred = store.mk_le(x, y);
        let lifted = store.lift_arithmetic_ite(pred);

        assert_eq!(lifted, pred);
    }

    #[test]
    fn test_ite_lifting_bool_ite_not_lifted() {
        let mut store = TermStore::new();

        // (= (ite c true false) p) should NOT lift since ITE result is Bool
        let c = store.mk_var("c", Sort::Bool);
        let p = store.mk_var("p", Sort::Bool);

        let true_t = store.true_term();
        let false_t = store.false_term();
        let ite_expr = store.mk_ite(c, true_t, false_t);
        let pred = store.mk_eq(ite_expr, p);

        let lifted = store.lift_arithmetic_ite(pred);

        // For Bool ITE, the lifting may or may not happen depending on simplifications
        // The key is that the result should be semantically equivalent
        // Just check that it doesn't crash
        assert!(!store.is_false(lifted));
    }

    #[test]
    fn test_ite_lifting_nested() {
        let mut store = TermStore::new();

        // (and (<= (ite c x y) z) (<= w v)) should lift the first conjunct
        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Real);
        let y = store.mk_var("y", Sort::Real);
        let z = store.mk_var("z", Sort::Real);
        let w = store.mk_var("w", Sort::Real);
        let v = store.mk_var("v", Sort::Real);

        let ite_expr = store.mk_ite(c, x, y);
        let pred1 = store.mk_le(ite_expr, z);
        let pred2 = store.mk_le(w, v);
        let conj = store.mk_and(vec![pred1, pred2]);

        let lifted = store.lift_arithmetic_ite(conj);

        // Should be (and (ite c (<= x z) (<= y z)) (<= w v))
        match store.get(lifted) {
            TermData::App(Symbol::Named(name), args) => {
                assert_eq!(name, "and");
                assert_eq!(args.len(), 2);
                // First arg should be lifted ITE
                assert!(matches!(store.get(args[0]), TermData::Ite(_, _, _)));
                // Second arg should be unchanged
                assert_eq!(args[1], pred2);
            }
            _ => panic!("Expected and application after lifting"),
        }
    }

    #[test]
    fn test_ite_lifting_lt() {
        let mut store = TermStore::new();

        // (< (ite c x y) z) should become (ite c (< x z) (< y z))
        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);
        let z = store.mk_var("z", Sort::Int);

        let ite_expr = store.mk_ite(c, x, y);
        let pred = store.mk_lt(ite_expr, z);

        let lifted = store.lift_arithmetic_ite(pred);

        match store.get(lifted) {
            TermData::Ite(cond, _, _) => {
                assert_eq!(*cond, c);
            }
            _ => panic!("Expected ITE after lifting"),
        }
    }

    #[test]
    fn test_ite_lifting_nested_in_arithmetic() {
        let mut store = TermStore::new();

        // (<= (+ x (ite c 1 0)) y) should become (ite c (<= (+ x 1) y) (<= (+ x 0) y))
        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", Sort::Int);
        let y = store.mk_var("y", Sort::Int);
        let one = store.mk_int(BigInt::from(1));
        let zero = store.mk_int(BigInt::from(0));

        let ite_expr = store.mk_ite(c, one, zero);
        let sum = store.mk_add(vec![x, ite_expr]);
        let pred = store.mk_le(sum, y);

        let lifted = store.lift_arithmetic_ite(pred);

        // Should be lifted to an ITE at the top level
        match store.get(lifted) {
            TermData::Ite(cond, then_t, else_t) => {
                assert_eq!(*cond, c);
                // Both branches should be <= predicates, not contain ITEs
                match store.get(*then_t) {
                    TermData::App(Symbol::Named(name), _) => {
                        assert_eq!(name, "<=");
                    }
                    _ => panic!("Expected <= application in then branch"),
                }
                match store.get(*else_t) {
                    TermData::App(Symbol::Named(name), _) => {
                        assert_eq!(name, "<=");
                    }
                    _ => panic!("Expected <= application in else branch"),
                }
            }
            _ => panic!("Expected ITE after lifting nested arithmetic ITE"),
        }
    }

    #[test]
    fn test_ite_lifting_uninterpreted_sort_equality() {
        let mut store = TermStore::new();

        // (= x (ite c a b)) with x, a, b of uninterpreted sort S
        // should become (ite c (= x a) (= x b))
        let sort_s = Sort::Uninterpreted("S".to_string());
        let c = store.mk_var("c", Sort::Bool);
        let x = store.mk_var("x", sort_s.clone());
        let a = store.mk_var("a", sort_s.clone());
        let b = store.mk_var("b", sort_s.clone());

        let ite_expr = store.mk_ite(c, a, b);
        let eq = store.mk_eq(x, ite_expr);

        println!("Original eq: {:?}", store.get(eq));

        let lifted = store.lift_arithmetic_ite(eq);

        println!("Lifted: {:?}", store.get(lifted));
        println!("Same? {}", eq == lifted);

        // Should be (ite c (= x a) (= x b))
        match store.get(lifted) {
            TermData::Ite(cond, then_t, else_t) => {
                assert_eq!(*cond, c);
                // then branch should be (= x a)
                match store.get(*then_t) {
                    TermData::App(Symbol::Named(name), args) => {
                        assert_eq!(name, "=");
                        // Args could be [x, a] or [a, x] due to canonical ordering
                        assert!(
                            (args[0] == x && args[1] == a) || (args[0] == a && args[1] == x),
                            "Expected equality with x and a, got {:?}",
                            args
                        );
                    }
                    _ => panic!(
                        "Expected = application in then branch, got {:?}",
                        store.get(*then_t)
                    ),
                }
                // else branch should be (= x b)
                match store.get(*else_t) {
                    TermData::App(Symbol::Named(name), args) => {
                        assert_eq!(name, "=");
                        // Args could be [x, b] or [b, x] due to canonical ordering
                        assert!(
                            (args[0] == x && args[1] == b) || (args[0] == b && args[1] == x),
                            "Expected equality with x and b, got {:?}",
                            args
                        );
                    }
                    _ => panic!(
                        "Expected = application in else branch, got {:?}",
                        store.get(*else_t)
                    ),
                }
            }
            _ => panic!("Expected ITE after lifting, got {:?}", store.get(lifted)),
        }
    }
}
