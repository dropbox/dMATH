//! Elaboration: convert parsed terms to internal representation
//!
//! This module bridges the parser's AST to the core term representation.
//! It handles:
//! - Sort conversion
//! - Term internalization into the hash-consed store
//! - Symbol table management
//! - Let-binding expansion

use crate::command::{self, Command, Constant as ParsedConstant, Term as ParsedTerm};
use hashbrown::HashMap;
use num_bigint::BigInt;
use num_rational::BigRational;
use z4_core::{Sort, Symbol, TermId, TermStore};

/// Error during elaboration
#[derive(Debug, Clone, thiserror::Error)]
pub enum ElaborateError {
    /// Undefined symbol
    #[error("undefined symbol: {0}")]
    UndefinedSymbol(String),
    /// Sort mismatch
    #[error("sort mismatch: expected {expected}, got {actual}")]
    SortMismatch {
        /// The expected sort
        expected: String,
        /// The actual sort found
        actual: String,
    },
    /// Invalid constant
    #[error("invalid constant: {0}")]
    InvalidConstant(String),
    /// Unsupported feature
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Result type for elaboration
pub type Result<T> = std::result::Result<T, ElaborateError>;

/// Symbol information
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// The term ID if it's a constant
    pub term: Option<TermId>,
    /// The sort of the symbol
    pub sort: Sort,
    /// Argument sorts (empty for constants)
    pub arg_sorts: Vec<Sort>,
}

/// Elaboration context
pub struct Context {
    /// The term store
    pub terms: TermStore,
    /// Symbol table: name -> info
    symbols: HashMap<String, SymbolInfo>,
    /// Sort definitions: name -> sort
    sort_defs: HashMap<String, Sort>,
    /// Function definitions: name -> (params, body)
    fun_defs: HashMap<String, (Vec<(String, Sort)>, ParsedTerm)>,
    /// Current logic
    pub logic: Option<String>,
    /// Assertions
    pub assertions: Vec<TermId>,
    /// Scope stack for push/pop
    scopes: Vec<ScopeFrame>,
    /// Solver options (keyword -> value)
    options: HashMap<String, OptionValue>,
    /// Named formulas: name -> term_id (for get-assignment and get-unsat-core)
    named_terms: HashMap<String, TermId>,
}

/// Value for a solver option
#[derive(Debug, Clone, PartialEq)]
pub enum OptionValue {
    /// Boolean option
    Bool(bool),
    /// String option
    String(String),
    /// Numeric option
    Numeral(String),
}

/// A scope frame for push/pop
#[derive(Debug, Clone, Default)]
struct ScopeFrame {
    /// Symbols defined in this scope
    symbols: Vec<String>,
    /// Number of assertions before this scope
    assertion_count: usize,
    /// Named terms defined in this scope
    named_terms: Vec<String>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Create a new elaboration context
    pub fn new() -> Self {
        let mut options = HashMap::new();
        // Default options per SMT-LIB 2.6 standard
        options.insert("print-success".to_string(), OptionValue::Bool(false));
        options.insert("produce-models".to_string(), OptionValue::Bool(true));
        options.insert("produce-unsat-cores".to_string(), OptionValue::Bool(false));
        options.insert("produce-proofs".to_string(), OptionValue::Bool(false));
        options.insert("produce-assignments".to_string(), OptionValue::Bool(false));
        options.insert(
            "random-seed".to_string(),
            OptionValue::Numeral("0".to_string()),
        );

        Context {
            terms: TermStore::new(),
            symbols: HashMap::new(),
            sort_defs: HashMap::new(),
            fun_defs: HashMap::new(),
            logic: None,
            assertions: Vec::new(),
            scopes: Vec::new(),
            options,
            named_terms: HashMap::new(),
        }
    }

    /// Convert a parsed sort to internal sort
    pub fn elaborate_sort(&self, sort: &command::Sort) -> Result<Sort> {
        match sort {
            command::Sort::Simple(name) => match name.as_str() {
                "Bool" => Ok(Sort::Bool),
                "Int" => Ok(Sort::Int),
                "Real" => Ok(Sort::Real),
                "String" => Ok(Sort::String),
                other => {
                    if let Some(s) = self.sort_defs.get(other) {
                        Ok(s.clone())
                    } else {
                        Ok(Sort::Uninterpreted(other.to_string()))
                    }
                }
            },
            command::Sort::Indexed(name, indices) => match name.as_str() {
                "BitVec" => {
                    let width: u32 =
                        indices
                            .first()
                            .and_then(|s| s.parse().ok())
                            .ok_or_else(|| {
                                ElaborateError::InvalidConstant("BitVec width".to_string())
                            })?;
                    Ok(Sort::BitVec(width))
                }
                "FloatingPoint" => {
                    let eb: u32 =
                        indices
                            .first()
                            .and_then(|s| s.parse().ok())
                            .ok_or_else(|| {
                                ElaborateError::InvalidConstant(
                                    "FloatingPoint exponent bits".to_string(),
                                )
                            })?;
                    let sb: u32 = indices.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| {
                        ElaborateError::InvalidConstant(
                            "FloatingPoint significand bits".to_string(),
                        )
                    })?;
                    Ok(Sort::FloatingPoint(eb, sb))
                }
                other => Err(ElaborateError::Unsupported(format!(
                    "indexed sort: {}",
                    other
                ))),
            },
            command::Sort::Parameterized(name, params) => match name.as_str() {
                "Array" => {
                    if params.len() != 2 {
                        return Err(ElaborateError::InvalidConstant(
                            "Array requires 2 type parameters".to_string(),
                        ));
                    }
                    let index = self.elaborate_sort(&params[0])?;
                    let element = self.elaborate_sort(&params[1])?;
                    Ok(Sort::Array(Box::new(index), Box::new(element)))
                }
                other => Err(ElaborateError::Unsupported(format!(
                    "parameterized sort: {}",
                    other
                ))),
            },
        }
    }

    /// Elaborate a parsed term into the term store
    pub fn elaborate_term(
        &mut self,
        term: &ParsedTerm,
        env: &HashMap<String, TermId>,
    ) -> Result<TermId> {
        match term {
            ParsedTerm::Const(c) => self.elaborate_constant(c),
            ParsedTerm::Symbol(name) => {
                // Check local environment first (let bindings, quantifier vars)
                if let Some(&id) = env.get(name) {
                    return Ok(id);
                }
                // Check function definitions FIRST (expand nullary define-fun)
                // This must come before the symbols check to properly expand
                // definitions like (define-fun my_eq () Bool (= a b))
                if let Some((params, body)) = self.fun_defs.get(name).cloned() {
                    if params.is_empty() {
                        // It's a nullary function, just expand the body
                        return self.elaborate_term(&body, env);
                    }
                }
                // Check global symbols
                if let Some(info) = self.symbols.get(name) {
                    if let Some(id) = info.term {
                        return Ok(id);
                    }
                    // It's a declared function with no definition - create a variable
                    return Ok(self.terms.mk_var(name, info.sort.clone()));
                }
                // Handle negative numeric literals: -1, -42, -3.14, etc.
                // In SMT-LIB these should be (- 1), (- 42) but many benchmarks
                // use the shorthand -1, -42 which the lexer parses as symbols
                if let Some(abs_str) = name.strip_prefix('-') {
                    if !abs_str.is_empty() {
                        // Check for negative integer
                        if abs_str.chars().all(|c| c.is_ascii_digit()) {
                            let abs_value: BigInt = abs_str
                                .parse()
                                .map_err(|_| ElaborateError::InvalidConstant(name.clone()))?;
                            let neg_value = -abs_value;
                            return Ok(self.terms.mk_int(neg_value));
                        }
                        // Check for negative decimal (e.g., -3.14)
                        if abs_str.contains('.')
                            && abs_str.chars().all(|c| c.is_ascii_digit() || c == '.')
                            && abs_str.chars().filter(|&c| c == '.').count() == 1
                        {
                            // Parse as rational and negate
                            let parts: Vec<&str> = abs_str.split('.').collect();
                            if parts.len() == 2 {
                                let int_part: BigInt = parts[0]
                                    .parse()
                                    .map_err(|_| ElaborateError::InvalidConstant(name.clone()))?;
                                let frac_str = parts[1];
                                let frac_part: BigInt = frac_str
                                    .parse()
                                    .map_err(|_| ElaborateError::InvalidConstant(name.clone()))?;
                                let denom = BigInt::from(10).pow(frac_str.len() as u32);
                                let numer = int_part * &denom + frac_part;
                                let rational = BigRational::new(-numer, denom);
                                return Ok(self.terms.mk_rational(rational));
                            }
                        }
                    }
                }
                Err(ElaborateError::UndefinedSymbol(name.clone()))
            }
            ParsedTerm::App(name, args) => self.elaborate_app(name, args, env),
            ParsedTerm::Let(bindings, body) => {
                let mut new_env = env.clone();
                for (name, value) in bindings {
                    let value_id = self.elaborate_term(value, &new_env)?;
                    new_env.insert(name.clone(), value_id);
                }
                self.elaborate_term(body, &new_env)
            }
            ParsedTerm::Forall(bindings, body) => {
                // For now, just elaborate the body with fresh variables
                let mut new_env = env.clone();
                for (name, sort) in bindings {
                    let sort = self.elaborate_sort(sort)?;
                    let var = self.terms.mk_fresh_var(name, sort);
                    new_env.insert(name.clone(), var);
                }
                self.elaborate_term(body, &new_env)
            }
            ParsedTerm::Exists(bindings, body) => {
                // For now, just elaborate the body with fresh variables
                let mut new_env = env.clone();
                for (name, sort) in bindings {
                    let sort = self.elaborate_sort(sort)?;
                    let var = self.terms.mk_fresh_var(name, sort);
                    new_env.insert(name.clone(), var);
                }
                self.elaborate_term(body, &new_env)
            }
            ParsedTerm::Annotated(inner, annotations) => {
                // Elaborate the inner term
                let term_id = self.elaborate_term(inner, env)?;

                // Process annotations - track :named for get-assignment
                for (keyword, value) in annotations {
                    if keyword == ":named" {
                        if let crate::sexp::SExpr::Symbol(name) = value {
                            self.named_terms.insert(name.clone(), term_id);
                            // Track in current scope for proper cleanup on pop
                            if let Some(scope) = self.scopes.last_mut() {
                                scope.named_terms.push(name.clone());
                            }
                        }
                    }
                }

                Ok(term_id)
            }
        }
    }

    /// Elaborate a constant
    fn elaborate_constant(&mut self, constant: &ParsedConstant) -> Result<TermId> {
        match constant {
            ParsedConstant::True => Ok(self.terms.true_term()),
            ParsedConstant::False => Ok(self.terms.false_term()),
            ParsedConstant::Numeral(s) => {
                let value: BigInt = s
                    .parse()
                    .map_err(|_| ElaborateError::InvalidConstant(s.clone()))?;
                Ok(self.terms.mk_int(value))
            }
            ParsedConstant::Decimal(s) => {
                // Parse as rational
                let parts: Vec<&str> = s.split('.').collect();
                if parts.len() == 2 {
                    let int_part: BigInt = parts[0]
                        .parse()
                        .map_err(|_| ElaborateError::InvalidConstant(s.clone()))?;
                    let frac_str = parts[1];
                    let frac_part: BigInt = frac_str
                        .parse()
                        .map_err(|_| ElaborateError::InvalidConstant(s.clone()))?;
                    let denom = BigInt::from(10).pow(frac_str.len() as u32);
                    let numer = int_part * &denom + frac_part;
                    let rational = BigRational::new(numer, denom);
                    Ok(self.terms.mk_rational(rational))
                } else {
                    let value: BigInt = s
                        .parse()
                        .map_err(|_| ElaborateError::InvalidConstant(s.clone()))?;
                    Ok(self.terms.mk_rational(BigRational::from(value)))
                }
            }
            ParsedConstant::Hexadecimal(s) => {
                // #xABCD -> bitvector
                let hex = s.trim_start_matches("#x");
                let value = BigInt::parse_bytes(hex.as_bytes(), 16)
                    .ok_or_else(|| ElaborateError::InvalidConstant(s.clone()))?;
                let width = (hex.len() * 4) as u32;
                Ok(self.terms.mk_bitvec(value, width))
            }
            ParsedConstant::Binary(s) => {
                // #b1010 -> bitvector
                let bin = s.trim_start_matches("#b");
                let value = BigInt::parse_bytes(bin.as_bytes(), 2)
                    .ok_or_else(|| ElaborateError::InvalidConstant(s.clone()))?;
                let width = bin.len() as u32;
                Ok(self.terms.mk_bitvec(value, width))
            }
            ParsedConstant::String(s) => Ok(self.terms.mk_string(s.clone())),
        }
    }

    /// Elaborate a function application
    fn elaborate_app(
        &mut self,
        name: &str,
        args: &[ParsedTerm],
        env: &HashMap<String, TermId>,
    ) -> Result<TermId> {
        // Check for function definition expansion
        if let Some((params, body)) = self.fun_defs.get(name).cloned() {
            if params.len() == args.len() {
                let mut new_env = env.clone();
                for ((param_name, _), arg) in params.iter().zip(args) {
                    let arg_id = self.elaborate_term(arg, env)?;
                    new_env.insert(param_name.clone(), arg_id);
                }
                return self.elaborate_term(&body, &new_env);
            }
        }

        // Elaborate arguments
        let arg_ids: Vec<TermId> = args
            .iter()
            .map(|a| self.elaborate_term(a, env))
            .collect::<Result<Vec<_>>>()?;

        // Check if this is a user-declared function (not a built-in).
        // User-declared functions should be treated as uninterpreted, even if
        // their name matches a built-in (e.g., "xor", "and").
        if let Some(info) = self.symbols.get(name) {
            if !info.arg_sorts.is_empty() {
                // This is a declared function with arguments - treat as uninterpreted
                let sym = Symbol::named(name);
                return Ok(self.terms.mk_app(sym, arg_ids, info.sort.clone()));
            }
        }

        // Handle built-in operations
        match name {
            "not" => {
                if arg_ids.len() != 1 {
                    return Err(ElaborateError::InvalidConstant(
                        "not requires 1 argument".to_string(),
                    ));
                }
                Ok(self.terms.mk_not(arg_ids[0]))
            }
            "and" => Ok(self.terms.mk_and(arg_ids)),
            "or" => Ok(self.terms.mk_or(arg_ids)),
            "=>" | "implies" => {
                if arg_ids.len() < 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "=> requires at least 2 arguments".to_string(),
                    ));
                }
                // Right-associative: a => b => c = a => (b => c)
                let mut result = *arg_ids.last().unwrap();
                for &arg in arg_ids.iter().rev().skip(1) {
                    result = self.terms.mk_implies(arg, result);
                }
                Ok(result)
            }
            "xor" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "xor requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_xor(arg_ids[0], arg_ids[1]))
            }
            "ite" => {
                if arg_ids.len() != 3 {
                    return Err(ElaborateError::InvalidConstant(
                        "ite requires 3 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_ite(arg_ids[0], arg_ids[1], arg_ids[2]))
            }
            "=" => {
                if arg_ids.len() < 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "= requires at least 2 arguments".to_string(),
                    ));
                }
                if arg_ids.len() == 2 {
                    Ok(self.terms.mk_eq(arg_ids[0], arg_ids[1]))
                } else {
                    // Pairwise equality
                    let mut eqs = Vec::new();
                    for i in 0..arg_ids.len() - 1 {
                        eqs.push(self.terms.mk_eq(arg_ids[i], arg_ids[i + 1]));
                    }
                    Ok(self.terms.mk_and(eqs))
                }
            }
            "distinct" => Ok(self.terms.mk_distinct(arg_ids)),
            // Arithmetic operations with constant folding
            "+" => Ok(self.terms.mk_add(arg_ids)),
            "-" => Ok(self.terms.mk_sub(arg_ids)),
            "*" => Ok(self.terms.mk_mul(arg_ids)),
            "/" => Ok(self.terms.mk_div(arg_ids)),
            "div" => Ok(self.terms.mk_intdiv(arg_ids)),
            "mod" => Ok(self.terms.mk_mod(arg_ids)),
            "abs" => {
                if arg_ids.len() != 1 {
                    return Err(ElaborateError::InvalidConstant(
                        "abs requires 1 argument".to_string(),
                    ));
                }
                Ok(self.terms.mk_abs(arg_ids[0]))
            }
            // Comparison operations with constant folding
            "<" => {
                if arg_ids.len() < 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "< requires at least 2 arguments".to_string(),
                    ));
                }
                if arg_ids.len() == 2 {
                    Ok(self.terms.mk_lt(arg_ids[0], arg_ids[1]))
                } else {
                    let mut cmps = Vec::new();
                    for i in 0..arg_ids.len() - 1 {
                        cmps.push(self.terms.mk_lt(arg_ids[i], arg_ids[i + 1]));
                    }
                    Ok(self.terms.mk_and(cmps))
                }
            }
            "<=" => {
                if arg_ids.len() < 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "<= requires at least 2 arguments".to_string(),
                    ));
                }
                if arg_ids.len() == 2 {
                    Ok(self.terms.mk_le(arg_ids[0], arg_ids[1]))
                } else {
                    let mut cmps = Vec::new();
                    for i in 0..arg_ids.len() - 1 {
                        cmps.push(self.terms.mk_le(arg_ids[i], arg_ids[i + 1]));
                    }
                    Ok(self.terms.mk_and(cmps))
                }
            }
            ">" => {
                if arg_ids.len() < 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "> requires at least 2 arguments".to_string(),
                    ));
                }
                if arg_ids.len() == 2 {
                    Ok(self.terms.mk_gt(arg_ids[0], arg_ids[1]))
                } else {
                    let mut cmps = Vec::new();
                    for i in 0..arg_ids.len() - 1 {
                        cmps.push(self.terms.mk_gt(arg_ids[i], arg_ids[i + 1]));
                    }
                    Ok(self.terms.mk_and(cmps))
                }
            }
            ">=" => {
                if arg_ids.len() < 2 {
                    return Err(ElaborateError::InvalidConstant(
                        ">= requires at least 2 arguments".to_string(),
                    ));
                }
                if arg_ids.len() == 2 {
                    Ok(self.terms.mk_ge(arg_ids[0], arg_ids[1]))
                } else {
                    let mut cmps = Vec::new();
                    for i in 0..arg_ids.len() - 1 {
                        cmps.push(self.terms.mk_ge(arg_ids[i], arg_ids[i + 1]));
                    }
                    Ok(self.terms.mk_and(cmps))
                }
            }
            // Bitvector operations
            "bvadd" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvadd requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvadd(arg_ids))
            }
            "bvsub" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvsub requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvsub(arg_ids))
            }
            "bvmul" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvmul requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvmul(arg_ids))
            }
            "bvand" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvand requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvand(arg_ids))
            }
            "bvor" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvor requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvor(arg_ids))
            }
            "bvxor" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvxor requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvxor(arg_ids))
            }
            "bvnand" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvnand requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvnand(arg_ids))
            }
            "bvnor" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvnor requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvnor(arg_ids))
            }
            "bvxnor" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvxnor requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvxnor(arg_ids))
            }
            "bvnot" => {
                if arg_ids.len() != 1 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvnot requires 1 argument".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvnot(arg_ids[0]))
            }
            "bvneg" => {
                if arg_ids.len() != 1 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvneg requires 1 argument".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvneg(arg_ids[0]))
            }
            "bv2nat" => {
                if arg_ids.len() != 1 {
                    return Err(ElaborateError::InvalidConstant(
                        "bv2nat requires 1 argument".to_string(),
                    ));
                }
                Ok(self.terms.mk_bv2nat(arg_ids[0]))
            }
            "bvshl" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvshl requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvshl(arg_ids))
            }
            "bvlshr" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvlshr requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvlshr(arg_ids))
            }
            "bvashr" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvashr requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvashr(arg_ids))
            }
            "bvudiv" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvudiv requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvudiv(arg_ids))
            }
            "bvurem" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvurem requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvurem(arg_ids))
            }
            "bvsdiv" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvsdiv requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvsdiv(arg_ids))
            }
            "bvsrem" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvsrem requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvsrem(arg_ids))
            }
            "bvsmod" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvsmod requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvsmod(arg_ids))
            }
            "bvcomp" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvcomp requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvcomp(arg_ids[0], arg_ids[1]))
            }
            // Bitvector comparison operations
            "bvult" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvult requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvult(arg_ids[0], arg_ids[1]))
            }
            "bvule" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvule requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvule(arg_ids[0], arg_ids[1]))
            }
            "bvugt" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvugt requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvugt(arg_ids[0], arg_ids[1]))
            }
            "bvuge" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvuge requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvuge(arg_ids[0], arg_ids[1]))
            }
            "bvslt" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvslt requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvslt(arg_ids[0], arg_ids[1]))
            }
            "bvsle" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvsle requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvsle(arg_ids[0], arg_ids[1]))
            }
            "bvsgt" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvsgt requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvsgt(arg_ids[0], arg_ids[1]))
            }
            "bvsge" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "bvsge requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvsge(arg_ids[0], arg_ids[1]))
            }
            // Array operations
            "select" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "select requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_select(arg_ids[0], arg_ids[1]))
            }
            "store" => {
                if arg_ids.len() != 3 {
                    return Err(ElaborateError::InvalidConstant(
                        "store requires 3 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_store(arg_ids[0], arg_ids[1], arg_ids[2]))
            }
            "concat" => {
                if arg_ids.len() != 2 {
                    return Err(ElaborateError::InvalidConstant(
                        "concat requires 2 arguments".to_string(),
                    ));
                }
                Ok(self.terms.mk_bvconcat(arg_ids))
            }
            _ => {
                // Handle indexed identifiers like (_ extract 7 0), (_ zero_extend 8), etc.
                if name.starts_with("(_ ") {
                    let parts: Vec<&str> = name
                        .trim_start_matches("(_ ")
                        .trim_end_matches(')')
                        .split_whitespace()
                        .collect();
                    if parts.is_empty() {
                        return Err(ElaborateError::InvalidConstant(name.to_string()));
                    }

                    let indices: Vec<u32> =
                        parts[1..].iter().filter_map(|s| s.parse().ok()).collect();

                    match parts[0] {
                        "extract" => {
                            if indices.len() != 2 || arg_ids.len() != 1 {
                                return Err(ElaborateError::InvalidConstant(
                                    "extract requires 2 indices and 1 argument".to_string(),
                                ));
                            }
                            Ok(self.terms.mk_bvextract(indices[0], indices[1], arg_ids[0]))
                        }
                        "int2bv" => {
                            if indices.len() != 1 || arg_ids.len() != 1 {
                                return Err(ElaborateError::InvalidConstant(
                                    "int2bv requires 1 index and 1 argument".to_string(),
                                ));
                            }
                            Ok(self.terms.mk_int2bv(indices[0], arg_ids[0]))
                        }
                        "zero_extend" => {
                            if indices.len() != 1 || arg_ids.len() != 1 {
                                return Err(ElaborateError::InvalidConstant(
                                    "zero_extend requires 1 index and 1 argument".to_string(),
                                ));
                            }
                            Ok(self.terms.mk_bvzero_extend(indices[0], arg_ids[0]))
                        }
                        "sign_extend" => {
                            if indices.len() != 1 || arg_ids.len() != 1 {
                                return Err(ElaborateError::InvalidConstant(
                                    "sign_extend requires 1 index and 1 argument".to_string(),
                                ));
                            }
                            Ok(self.terms.mk_bvsign_extend(indices[0], arg_ids[0]))
                        }
                        "rotate_left" => {
                            if indices.len() != 1 || arg_ids.len() != 1 {
                                return Err(ElaborateError::InvalidConstant(
                                    "rotate_left requires 1 index and 1 argument".to_string(),
                                ));
                            }
                            Ok(self.terms.mk_bvrotate_left(indices[0], arg_ids[0]))
                        }
                        "rotate_right" => {
                            if indices.len() != 1 || arg_ids.len() != 1 {
                                return Err(ElaborateError::InvalidConstant(
                                    "rotate_right requires 1 index and 1 argument".to_string(),
                                ));
                            }
                            Ok(self.terms.mk_bvrotate_right(indices[0], arg_ids[0]))
                        }
                        "repeat" => {
                            if indices.len() != 1 || arg_ids.len() != 1 {
                                return Err(ElaborateError::InvalidConstant(
                                    "repeat requires 1 index and 1 argument".to_string(),
                                ));
                            }
                            Ok(self.terms.mk_bvrepeat(indices[0], arg_ids[0]))
                        }
                        _ => {
                            // Unknown indexed identifier, create generic App
                            let sym = Symbol::indexed(parts[0], indices);
                            let result_sort = if let Some(info) = self.symbols.get(name) {
                                info.sort.clone()
                            } else {
                                Sort::Bool
                            };
                            Ok(self.terms.mk_app(sym, arg_ids, result_sort))
                        }
                    }
                } else if name.starts_with("(as const ") {
                    // Constant array: ((as const (Array T1 T2)) value)
                    // Parse the array sort from the name
                    if arg_ids.len() != 1 {
                        return Err(ElaborateError::InvalidConstant(
                            "constant array requires exactly 1 argument".to_string(),
                        ));
                    }
                    let index_sort = self.parse_const_array_index_sort(name)?;
                    Ok(self.terms.mk_const_array(index_sort, arg_ids[0]))
                } else {
                    // Not an indexed identifier
                    let sym = Symbol::named(name);
                    let result_sort = if let Some(info) = self.symbols.get(name) {
                        info.sort.clone()
                    } else {
                        Sort::Bool
                    };
                    Ok(self.terms.mk_app(sym, arg_ids, result_sort))
                }
            }
        }
    }

    /// Parse the index sort from a constant array expression like "(as const (Array Int Bool))"
    fn parse_const_array_index_sort(&self, name: &str) -> Result<Sort> {
        // Format: "(as const (Array T1 T2))"
        // We need to extract T1 (the index sort)

        // Strip "(as const " prefix and ")" suffix
        let inner = name
            .strip_prefix("(as const ")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| ElaborateError::InvalidConstant(name.to_string()))?;

        // Now inner should be "(Array T1 T2)"
        let inner = inner.trim();
        if !inner.starts_with("(Array ") || !inner.ends_with(')') {
            return Err(ElaborateError::InvalidConstant(format!(
                "expected Array sort in const: {}",
                name
            )));
        }

        // Strip "(Array " and ")"
        let sorts_part = inner
            .strip_prefix("(Array ")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| ElaborateError::InvalidConstant(name.to_string()))?;

        // Parse the index sort (first sort)
        // This is simplified - we handle common cases: Int, Bool, Real, (BitVec N)
        let index_sort_str = self.extract_first_sort(sorts_part)?;
        self.parse_simple_sort(&index_sort_str)
    }

    /// Extract the first sort from a space-separated sort list
    /// Handles nested sorts like "(BitVec 32)" and "(Array Int Int)"
    fn extract_first_sort(&self, s: &str) -> Result<String> {
        let s = s.trim();
        if s.starts_with('(') {
            // Find matching closing paren
            let mut depth = 0;
            for (i, c) in s.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(s[..=i].to_string());
                        }
                    }
                    _ => {}
                }
            }
            Err(ElaborateError::InvalidConstant(format!(
                "unbalanced parentheses in sort: {}",
                s
            )))
        } else {
            // Simple sort - take until space
            Ok(s.split_whitespace().next().unwrap_or(s).to_string())
        }
    }

    /// Parse a simple sort string into a Sort
    fn parse_simple_sort(&self, s: &str) -> Result<Sort> {
        let s = s.trim();
        match s {
            "Int" => Ok(Sort::Int),
            "Bool" => Ok(Sort::Bool),
            "Real" => Ok(Sort::Real),
            "String" => Ok(Sort::String),
            _ if s.starts_with("(_ BitVec ") => {
                // (_ BitVec N)
                let width_str = s
                    .strip_prefix("(_ BitVec ")
                    .and_then(|s| s.strip_suffix(')'))
                    .ok_or_else(|| ElaborateError::InvalidConstant(s.to_string()))?;
                let width: u32 = width_str
                    .trim()
                    .parse()
                    .map_err(|_| ElaborateError::InvalidConstant(s.to_string()))?;
                Ok(Sort::BitVec(width))
            }
            _ if s.starts_with("(BitVec ") => {
                // (BitVec N) - alternative syntax sometimes seen
                let width_str = s
                    .strip_prefix("(BitVec ")
                    .and_then(|s| s.strip_suffix(')'))
                    .ok_or_else(|| ElaborateError::InvalidConstant(s.to_string()))?;
                let width: u32 = width_str
                    .trim()
                    .parse()
                    .map_err(|_| ElaborateError::InvalidConstant(s.to_string()))?;
                Ok(Sort::BitVec(width))
            }
            _ => {
                // Check if it's a defined sort
                if let Some(sort) = self.sort_defs.get(s) {
                    Ok(sort.clone())
                } else {
                    // Treat as uninterpreted sort
                    Ok(Sort::Uninterpreted(s.to_string()))
                }
            }
        }
    }

    /// Declare a constant
    pub fn declare_const(&mut self, name: &str, sort: &command::Sort) -> Result<()> {
        let sort = self.elaborate_sort(sort)?;
        let term = self.terms.mk_var(name, sort.clone());
        self.symbols.insert(
            name.to_string(),
            SymbolInfo {
                term: Some(term),
                sort,
                arg_sorts: vec![],
            },
        );
        if let Some(frame) = self.scopes.last_mut() {
            frame.symbols.push(name.to_string());
        }
        Ok(())
    }

    /// Declare a function
    pub fn declare_fun(
        &mut self,
        name: &str,
        arg_sorts: &[command::Sort],
        ret_sort: &command::Sort,
    ) -> Result<()> {
        let arg_sorts: Vec<Sort> = arg_sorts
            .iter()
            .map(|s| self.elaborate_sort(s))
            .collect::<Result<Vec<_>>>()?;
        let ret_sort = self.elaborate_sort(ret_sort)?;

        // If no arguments, it's a constant
        let term = if arg_sorts.is_empty() {
            Some(self.terms.mk_var(name, ret_sort.clone()))
        } else {
            None
        };

        self.symbols.insert(
            name.to_string(),
            SymbolInfo {
                term,
                sort: ret_sort,
                arg_sorts,
            },
        );
        if let Some(frame) = self.scopes.last_mut() {
            frame.symbols.push(name.to_string());
        }
        Ok(())
    }

    /// Define a function
    pub fn define_fun(
        &mut self,
        name: &str,
        params: &[(String, command::Sort)],
        ret_sort: &command::Sort,
        body: &ParsedTerm,
    ) -> Result<()> {
        let params: Vec<(String, Sort)> = params
            .iter()
            .map(|(n, s)| Ok((n.clone(), self.elaborate_sort(s)?)))
            .collect::<Result<Vec<_>>>()?;
        let ret_sort = self.elaborate_sort(ret_sort)?;

        // Store the definition for expansion
        self.fun_defs
            .insert(name.to_string(), (params.clone(), body.clone()));

        // Also add to symbol table
        let arg_sorts: Vec<Sort> = params.iter().map(|(_, s)| s.clone()).collect();
        self.symbols.insert(
            name.to_string(),
            SymbolInfo {
                term: None,
                sort: ret_sort,
                arg_sorts,
            },
        );
        Ok(())
    }

    /// Define a recursive function
    ///
    /// For recursive functions, the function can reference itself in its body.
    /// We add to the symbol table first to enable self-reference during expansion.
    pub fn define_fun_rec(
        &mut self,
        name: &str,
        params: &[(String, command::Sort)],
        ret_sort: &command::Sort,
        body: &ParsedTerm,
    ) -> Result<()> {
        let params: Vec<(String, Sort)> = params
            .iter()
            .map(|(n, s)| Ok((n.clone(), self.elaborate_sort(s)?)))
            .collect::<Result<Vec<_>>>()?;
        let ret_sort = self.elaborate_sort(ret_sort)?;

        // For recursive functions, add to symbol table first so body can reference the function
        let arg_sorts: Vec<Sort> = params.iter().map(|(_, s)| s.clone()).collect();
        self.symbols.insert(
            name.to_string(),
            SymbolInfo {
                term: None,
                sort: ret_sort,
                arg_sorts,
            },
        );

        // Store the definition for expansion
        self.fun_defs
            .insert(name.to_string(), (params, body.clone()));

        Ok(())
    }

    /// Define mutually recursive functions
    ///
    /// For mutually recursive functions, all function signatures are registered
    /// first so the bodies can reference each other.
    pub fn define_funs_rec(
        &mut self,
        declarations: &[command::FuncDeclaration],
        bodies: &[ParsedTerm],
    ) -> Result<()> {
        // Elaborated declarations with internal Sort type
        type ElaboratedDecl = (String, Vec<(String, Sort)>, Sort);

        // First pass: register all function signatures in the symbol table
        let mut elaborated_decls: Vec<ElaboratedDecl> = Vec::new();

        for (name, params, ret_sort) in declarations {
            let params: Vec<(String, Sort)> = params
                .iter()
                .map(|(n, s)| Ok((n.clone(), self.elaborate_sort(s)?)))
                .collect::<Result<Vec<_>>>()?;
            let ret_sort = self.elaborate_sort(ret_sort)?;

            let arg_sorts: Vec<Sort> = params.iter().map(|(_, s)| s.clone()).collect();
            self.symbols.insert(
                name.to_string(),
                SymbolInfo {
                    term: None,
                    sort: ret_sort.clone(),
                    arg_sorts,
                },
            );

            elaborated_decls.push((name.clone(), params, ret_sort));
        }

        // Second pass: store all function definitions
        for ((name, params, _ret_sort), body) in elaborated_decls.into_iter().zip(bodies.iter()) {
            self.fun_defs.insert(name, (params, body.clone()));
        }

        Ok(())
    }

    /// Declare a single datatype
    ///
    /// A datatype declaration creates:
    /// - A new uninterpreted sort
    /// - A constructor function for each constructor
    /// - A selector function for each selector
    /// - A tester function (is-Constructor) for each constructor
    pub fn declare_datatype(
        &mut self,
        name: &str,
        datatype_dec: &command::DatatypeDec,
    ) -> Result<()> {
        // Register the sort as uninterpreted
        let sort = Sort::Uninterpreted(name.to_string());
        self.sort_defs.insert(name.to_string(), sort.clone());

        // Register constructors, selectors, and testers
        for ctor in &datatype_dec.constructors {
            // Elaborate selector sorts
            let selector_sorts: Vec<Sort> = ctor
                .selectors
                .iter()
                .map(|s| self.elaborate_sort(&s.sort))
                .collect::<Result<Vec<_>>>()?;

            // Constructor: (sel_sort1, ..., sel_sortN) -> DataType
            self.symbols.insert(
                ctor.name.clone(),
                SymbolInfo {
                    term: if selector_sorts.is_empty() {
                        // Nullary constructor is a constant
                        Some(self.terms.mk_var(&ctor.name, sort.clone()))
                    } else {
                        None
                    },
                    sort: sort.clone(),
                    arg_sorts: selector_sorts.clone(),
                },
            );

            // Selectors: DataType -> field_sort
            for (sel, sel_sort) in ctor.selectors.iter().zip(selector_sorts.iter()) {
                self.symbols.insert(
                    sel.name.clone(),
                    SymbolInfo {
                        term: None,
                        sort: sel_sort.clone(),
                        arg_sorts: vec![sort.clone()],
                    },
                );
            }

            // Tester: DataType -> Bool
            let tester_name = format!("is-{}", ctor.name);
            self.symbols.insert(
                tester_name,
                SymbolInfo {
                    term: None,
                    sort: Sort::Bool,
                    arg_sorts: vec![sort.clone()],
                },
            );
        }

        Ok(())
    }

    /// Declare multiple (possibly mutually recursive) datatypes
    ///
    /// For mutually recursive datatypes, all sort names are registered first
    /// so that constructor/selector sorts can reference each other.
    pub fn declare_datatypes(
        &mut self,
        sort_decs: &[command::SortDec],
        datatype_decs: &[command::DatatypeDec],
    ) -> Result<()> {
        // First pass: register all sort names
        for sort_dec in sort_decs {
            if sort_dec.arity != 0 {
                return Err(ElaborateError::Unsupported(
                    "parametric datatypes are not yet supported".to_string(),
                ));
            }
            let sort = Sort::Uninterpreted(sort_dec.name.clone());
            self.sort_defs.insert(sort_dec.name.clone(), sort);
        }

        // Second pass: register constructors, selectors, and testers for each datatype
        for (sort_dec, datatype_dec) in sort_decs.iter().zip(datatype_decs.iter()) {
            let sort = Sort::Uninterpreted(sort_dec.name.clone());

            for ctor in &datatype_dec.constructors {
                // Elaborate selector sorts
                let selector_sorts: Vec<Sort> = ctor
                    .selectors
                    .iter()
                    .map(|s| self.elaborate_sort(&s.sort))
                    .collect::<Result<Vec<_>>>()?;

                // Constructor: (sel_sort1, ..., sel_sortN) -> DataType
                self.symbols.insert(
                    ctor.name.clone(),
                    SymbolInfo {
                        term: if selector_sorts.is_empty() {
                            Some(self.terms.mk_var(&ctor.name, sort.clone()))
                        } else {
                            None
                        },
                        sort: sort.clone(),
                        arg_sorts: selector_sorts.clone(),
                    },
                );

                // Selectors: DataType -> field_sort
                for (sel, sel_sort) in ctor.selectors.iter().zip(selector_sorts.iter()) {
                    self.symbols.insert(
                        sel.name.clone(),
                        SymbolInfo {
                            term: None,
                            sort: sel_sort.clone(),
                            arg_sorts: vec![sort.clone()],
                        },
                    );
                }

                // Tester: DataType -> Bool
                let tester_name = format!("is-{}", ctor.name);
                self.symbols.insert(
                    tester_name,
                    SymbolInfo {
                        term: None,
                        sort: Sort::Bool,
                        arg_sorts: vec![sort.clone()],
                    },
                );
            }
        }

        Ok(())
    }

    /// Add an assertion
    pub fn assert(&mut self, term: &ParsedTerm) -> Result<()> {
        let id = self.elaborate_term(term, &HashMap::new())?;
        self.assertions.push(id);
        Ok(())
    }

    /// Push a scope
    pub fn push(&mut self) {
        self.scopes.push(ScopeFrame {
            symbols: Vec::new(),
            assertion_count: self.assertions.len(),
            named_terms: Vec::new(),
        });
    }

    /// Pop a scope
    pub fn pop(&mut self) {
        if let Some(frame) = self.scopes.pop() {
            // Remove symbols defined in this scope
            for name in frame.symbols {
                self.symbols.remove(&name);
            }
            // Remove assertions from this scope
            self.assertions.truncate(frame.assertion_count);
            // Remove named terms defined in this scope
            for name in frame.named_terms {
                self.named_terms.remove(&name);
            }
        }
    }

    /// Iterate over all declared symbols
    pub fn symbol_iter(&self) -> impl Iterator<Item = (&String, &SymbolInfo)> {
        self.symbols.iter()
    }

    /// Register a symbol directly (for native API use)
    ///
    /// This is used by the native Rust API to register constants created
    /// via `mk_var` so they appear in models.
    pub fn register_symbol(&mut self, name: String, term: TermId, sort: Sort) {
        self.symbols.insert(
            name.clone(),
            SymbolInfo {
                term: Some(term),
                sort,
                arg_sorts: vec![],
            },
        );
        if let Some(frame) = self.scopes.last_mut() {
            frame.symbols.push(name);
        }
    }

    /// Process a command
    pub fn process_command(&mut self, cmd: &Command) -> Result<Option<CommandResult>> {
        match cmd {
            Command::SetLogic(logic) => {
                self.logic = Some(logic.clone());
                Ok(None)
            }
            Command::DeclareConst(name, sort) => {
                self.declare_const(name, sort)?;
                Ok(None)
            }
            Command::DeclareFun(name, arg_sorts, ret_sort) => {
                self.declare_fun(name, arg_sorts, ret_sort)?;
                Ok(None)
            }
            Command::DefineFun(name, params, ret_sort, body) => {
                self.define_fun(name, params, ret_sort, body)?;
                Ok(None)
            }
            Command::DefineFunRec(name, params, ret_sort, body) => {
                // For recursive functions, register the symbol first so the body can reference it
                self.define_fun_rec(name, params, ret_sort, body)?;
                Ok(None)
            }
            Command::DefineFunsRec(declarations, bodies) => {
                // For mutually recursive functions, register all symbols first
                self.define_funs_rec(declarations, bodies)?;
                Ok(None)
            }
            Command::Assert(term) => {
                self.assert(term)?;
                Ok(None)
            }
            Command::Push(n) => {
                for _ in 0..*n {
                    self.push();
                }
                Ok(None)
            }
            Command::Pop(n) => {
                for _ in 0..*n {
                    self.pop();
                }
                Ok(None)
            }
            Command::CheckSat => Ok(Some(CommandResult::CheckSat)),
            Command::CheckSatAssuming(terms) => {
                // Elaborate each assumption term to get its TermId
                let term_ids: Vec<TermId> = terms
                    .iter()
                    .map(|t| self.elaborate_term(t, &HashMap::new()))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Some(CommandResult::CheckSatAssuming(term_ids)))
            }
            Command::GetModel => Ok(Some(CommandResult::GetModel)),
            Command::GetValue(terms) => {
                // Elaborate each term to get its TermId
                let term_ids: Vec<TermId> = terms
                    .iter()
                    .map(|t| self.elaborate_term(t, &HashMap::new()))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Some(CommandResult::GetValue(term_ids)))
            }
            Command::GetInfo(keyword) => Ok(Some(CommandResult::GetInfo(keyword.clone()))),
            Command::GetOption(keyword) => Ok(Some(CommandResult::GetOption(keyword.clone()))),
            Command::GetAssertions => Ok(Some(CommandResult::GetAssertions)),
            Command::SetOption(keyword, value) => {
                self.set_option(keyword, value);
                Ok(None)
            }
            Command::Exit => Ok(Some(CommandResult::Exit)),
            Command::Reset => {
                *self = Context::new();
                Ok(None)
            }
            Command::ResetAssertions => {
                self.assertions.clear();
                self.scopes.clear();
                Ok(None)
            }
            // Declare/define sort are stored but don't produce output
            Command::DeclareSort(name, _arity) => {
                // Store as uninterpreted sort
                self.sort_defs
                    .insert(name.clone(), Sort::Uninterpreted(name.clone()));
                Ok(None)
            }
            Command::DefineSort(name, _params, sort) => {
                let elaborated = self.elaborate_sort(sort)?;
                self.sort_defs.insert(name.clone(), elaborated);
                Ok(None)
            }
            Command::DeclareDatatype(name, datatype_dec) => {
                self.declare_datatype(name, datatype_dec)?;
                Ok(None)
            }
            Command::DeclareDatatypes(sort_decs, datatype_decs) => {
                self.declare_datatypes(sort_decs, datatype_decs)?;
                Ok(None)
            }
            // SetInfo is acknowledged but not required to produce output
            Command::SetInfo(_, _) => Ok(None),
            // Echo returns the message to be printed (handled by executor)
            Command::Echo(msg) => Ok(Some(CommandResult::Echo(msg.clone()))),
            Command::GetAssignment => Ok(Some(CommandResult::GetAssignment)),
            Command::GetUnsatCore => Ok(Some(CommandResult::GetUnsatCore)),
            Command::GetUnsatAssumptions => Ok(Some(CommandResult::GetUnsatAssumptions)),
            Command::GetProof => Ok(Some(CommandResult::GetProof)),
            Command::Simplify(term) => {
                let term_id = self.elaborate_term(term, &HashMap::new())?;
                Ok(Some(CommandResult::Simplify(term_id)))
            }
        }
    }

    /// Set a solver option
    fn set_option(&mut self, keyword: &str, value: &crate::sexp::SExpr) {
        use crate::sexp::SExpr;
        let key = keyword.trim_start_matches(':').to_string();
        let opt_value = match value {
            SExpr::True => OptionValue::Bool(true),
            SExpr::False => OptionValue::Bool(false),
            SExpr::Numeral(n) => OptionValue::Numeral(n.clone()),
            SExpr::String(s) => OptionValue::String(s.clone()),
            SExpr::Symbol(s) => OptionValue::String(s.clone()),
            _ => return, // Ignore unsupported value types
        };
        self.options.insert(key, opt_value);
    }

    /// Get an option value
    pub fn get_option(&self, keyword: &str) -> Option<&OptionValue> {
        let key = keyword.trim_start_matches(':');
        self.options.get(key)
    }

    /// Iterate over named terms (for get-assignment)
    pub fn named_terms_iter(&self) -> impl Iterator<Item = (&str, TermId)> {
        self.named_terms.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

/// Result of processing a command that requires action
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandResult {
    /// Need to run check-sat
    CheckSat,
    /// Need to run check-sat with assumptions
    CheckSatAssuming(Vec<TermId>),
    /// Need to produce a model
    GetModel,
    /// Need to produce values for specific terms
    GetValue(Vec<TermId>),
    /// Need to get solver info
    GetInfo(String),
    /// Need to get an option value
    GetOption(String),
    /// Need to get current assertions
    GetAssertions,
    /// Need to print a string (echo command)
    Echo(String),
    /// Need to get assignment of named formulas
    GetAssignment,
    /// Need to get unsatisfiable core
    GetUnsatCore,
    /// Need to get unsatisfiable assumptions (from check-sat-assuming)
    GetUnsatAssumptions,
    /// Need to get proof
    GetProof,
    /// Exit the solver
    Exit,
    /// Need to simplify a term
    Simplify(TermId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn test_elaborate_simple() {
        let input = r#"
            (declare-const x Int)
            (declare-const y Int)
            (assert (= x y))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bool_ops() {
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (and a b))
            (assert (or a (not b)))
            (assert (=> a b))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 3);
    }

    #[test]
    fn test_elaborate_let() {
        let input = r#"
            (declare-const x Int)
            (assert (let ((y x)) (= y x)))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        // let (y = x) in (= y x) should simplify to (= x x) = true
        assert_eq!(ctx.assertions.len(), 1);
        assert!(ctx.terms.is_true(ctx.assertions[0]));
    }

    #[test]
    fn test_elaborate_define_fun() {
        let input = r#"
            (define-fun double ((x Int)) Int (+ x x))
            (declare-const a Int)
            (assert (= (double a) (+ a a)))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_define_fun_nullary() {
        // Test nullary define-fun (no parameters)
        // This tests the bug fix where nullary define-fun wasn't being expanded
        let input = r#"
            (declare-sort U 0)
            (declare-fun a () U)
            (declare-fun b () U)
            (define-fun my_eq () Bool (= a b))
            (assert my_eq)
            (assert (not (= a b)))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        // Should have 2 assertions:
        // 1. my_eq expanded to (= a b)
        // 2. (not (= a b))
        assert_eq!(ctx.assertions.len(), 2);

        // The first assertion (= a b) and second assertion (not (= a b))
        // should create a contradiction, so a solver would return unsat
    }

    #[test]
    fn test_elaborate_bitvector() {
        let input = r#"
            (declare-const x (_ BitVec 32))
            (assert (= x #x0000FFFF))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_push_pop() {
        let input = r#"
            (declare-const x Int)
            (push 1)
            (declare-const y Int)
            (assert (= x y))
            (pop 1)
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        // After pop, y should be gone and assertion should be removed
        assert_eq!(ctx.assertions.len(), 0);
        assert!(ctx.symbols.contains_key("x"));
        assert!(!ctx.symbols.contains_key("y"));
    }

    #[test]
    fn test_declare_datatype_simple() {
        let mut ctx = Context::new();

        // Declare a simple enumeration
        let script = "(declare-datatype Color ((Red) (Green) (Blue)))";
        for cmd in parse(script).unwrap() {
            ctx.process_command(&cmd).unwrap();
        }

        // Check sort is registered
        assert!(ctx.sort_defs.contains_key("Color"));

        // Check constructors are registered
        let red = ctx.symbols.get("Red").expect("Red constructor not found");
        assert!(red.arg_sorts.is_empty()); // Nullary constructor
        assert!(red.term.is_some()); // Has a term since it's nullary

        let green = ctx
            .symbols
            .get("Green")
            .expect("Green constructor not found");
        assert!(green.arg_sorts.is_empty());

        let blue = ctx.symbols.get("Blue").expect("Blue constructor not found");
        assert!(blue.arg_sorts.is_empty());

        // Check testers are registered
        let is_red = ctx.symbols.get("is-Red").expect("is-Red tester not found");
        assert_eq!(is_red.arg_sorts.len(), 1);
        assert_eq!(is_red.sort, Sort::Bool);

        let is_green = ctx
            .symbols
            .get("is-Green")
            .expect("is-Green tester not found");
        assert_eq!(is_green.arg_sorts.len(), 1);
        assert_eq!(is_green.sort, Sort::Bool);
    }

    #[test]
    fn test_declare_datatype_with_selectors() {
        let mut ctx = Context::new();

        // Declare a record type
        let script = "(declare-datatype Point ((mk-point (x Int) (y Int))))";
        for cmd in parse(script).unwrap() {
            ctx.process_command(&cmd).unwrap();
        }

        // Check sort is registered
        assert!(ctx.sort_defs.contains_key("Point"));

        // Check constructor
        let mk_point = ctx
            .symbols
            .get("mk-point")
            .expect("mk-point constructor not found");
        assert_eq!(mk_point.arg_sorts.len(), 2);
        assert_eq!(mk_point.arg_sorts[0], Sort::Int);
        assert_eq!(mk_point.arg_sorts[1], Sort::Int);
        assert!(mk_point.term.is_none()); // Not nullary, so no term

        // Check selectors
        let sel_x = ctx.symbols.get("x").expect("x selector not found");
        assert_eq!(sel_x.arg_sorts.len(), 1);
        assert_eq!(sel_x.sort, Sort::Int);

        let sel_y = ctx.symbols.get("y").expect("y selector not found");
        assert_eq!(sel_y.arg_sorts.len(), 1);
        assert_eq!(sel_y.sort, Sort::Int);

        // Check tester
        let is_mk_point = ctx
            .symbols
            .get("is-mk-point")
            .expect("is-mk-point tester not found");
        assert_eq!(is_mk_point.arg_sorts.len(), 1);
        assert_eq!(is_mk_point.sort, Sort::Bool);
    }

    #[test]
    fn test_declare_datatypes_mutually_recursive() {
        let mut ctx = Context::new();

        // Declare mutually recursive Tree/Forest types
        let script = r#"(declare-datatypes ((Tree 0) (Forest 0))
                        (((leaf (val Int)) (node (children Forest)))
                         ((nil) (cons (head Tree) (tail Forest)))))"#;
        for cmd in parse(script).unwrap() {
            ctx.process_command(&cmd).unwrap();
        }

        // Check sorts are registered
        assert!(ctx.sort_defs.contains_key("Tree"));
        assert!(ctx.sort_defs.contains_key("Forest"));

        // Check Tree constructors
        let leaf = ctx.symbols.get("leaf").expect("leaf constructor not found");
        assert_eq!(leaf.arg_sorts.len(), 1);
        assert_eq!(leaf.arg_sorts[0], Sort::Int);

        let node = ctx.symbols.get("node").expect("node constructor not found");
        assert_eq!(node.arg_sorts.len(), 1);
        assert_eq!(node.arg_sorts[0], Sort::Uninterpreted("Forest".to_string()));

        // Check Forest constructors
        let nil = ctx.symbols.get("nil").expect("nil constructor not found");
        assert!(nil.arg_sorts.is_empty());
        assert!(nil.term.is_some()); // Nullary

        let cons = ctx.symbols.get("cons").expect("cons constructor not found");
        assert_eq!(cons.arg_sorts.len(), 2);
        assert_eq!(cons.arg_sorts[0], Sort::Uninterpreted("Tree".to_string()));
        assert_eq!(cons.arg_sorts[1], Sort::Uninterpreted("Forest".to_string()));

        // Check selectors work across types
        let head = ctx.symbols.get("head").expect("head selector not found");
        assert_eq!(head.sort, Sort::Uninterpreted("Tree".to_string()));

        let tail = ctx.symbols.get("tail").expect("tail selector not found");
        assert_eq!(tail.sort, Sort::Uninterpreted("Forest".to_string()));
    }

    #[test]
    fn test_declare_datatype_can_use_in_terms() {
        let mut ctx = Context::new();

        // Declare Option type
        let script = r#"
            (declare-datatype Option ((None) (Some (value Int))))
            (declare-const x Option)
            (assert (is-Some x))
        "#;
        for cmd in parse(script).unwrap() {
            ctx.process_command(&cmd).unwrap();
        }

        // Should have one assertion
        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bv_extract() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (assert (= ((_ extract 7 4) x) #x0F))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bv_concat() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (concat x y) #x0FF0))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bv_zero_extend() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (assert (= ((_ zero_extend 8) x) #x00FF))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bv_sign_extend() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (assert (= ((_ sign_extend 8) x) #xFFFF))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bv_rotate() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (assert (= ((_ rotate_left 2) x) #xAA))
            (assert (= ((_ rotate_right 2) x) #x55))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 2);
    }

    #[test]
    fn test_elaborate_bv_repeat() {
        let input = r#"
            (declare-const x (_ BitVec 4))
            (assert (= ((_ repeat 4) x) #xAAAA))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bvsdiv() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (bvsdiv x y) #xFF))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bvsrem() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (bvsrem x y) #x01))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bvsmod() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (bvsmod x y) #x02))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bvcomp() {
        let input = r#"
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (bvcomp x y) #b1))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_bv2nat_and_int2bv() {
        let input = r#"
            (assert (= (bv2nat #x0F) 15))
            (assert (= ((_ int2bv 8) (- 1)) #xFF))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 2);
    }

    #[test]
    fn test_elaborate_bvnand_bvnor_bvxnor() {
        let input = r#"
            (assert (= (bvnand #xFF #xFF) #x00))
            (assert (= (bvnor #x00 #x00) #xFF))
            (assert (= (bvxnor #x0F #x0F) #xFF))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 3);
    }

    #[test]
    fn test_elaborate_array_select() {
        let input = r#"
            (declare-const a (Array Int Int))
            (assert (= (select a 0) 42))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_array_store() {
        let input = r#"
            (declare-const a (Array Int Int))
            (declare-const b (Array Int Int))
            (assert (= (store a 0 42) b))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_array_select_store_composition() {
        let input = r#"
            (declare-const a (Array Int Int))
            (assert (= (select (store a 0 42) 0) 42))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
    }

    #[test]
    fn test_elaborate_const_array() {
        let input = r#"
            (declare-const x Int)
            (assert (= (select ((as const (Array Int Int)) 0) x) 0))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
        // The assertion should simplify to (= 0 0) which is true
    }

    #[test]
    fn test_elaborate_const_array_bool() {
        let input = r#"
            (declare-const i Int)
            (assert (select ((as const (Array Int Bool)) true) i))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
        // The select should simplify to true
        assert!(ctx.terms.is_true(ctx.assertions[0]));
    }

    #[test]
    fn test_elaborate_const_array_with_store() {
        let input = r#"
            (assert (= (select (store ((as const (Array Int Int)) 0) 5 100) 5) 100))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        assert_eq!(ctx.assertions.len(), 1);
        // Should simplify to (= 100 100) which is true
        assert!(ctx.terms.is_true(ctx.assertions[0]));
    }

    #[test]
    fn test_elaborate_negative_integer_literals() {
        // Test that -1, -42, etc. are parsed correctly as negative integers
        // (many benchmarks use this shorthand instead of (- 1))
        let input = r#"
            (declare-const x Int)
            (assert (= x -1))
            (assert (>= x -42))
            (assert (<= (+ x -5) 0))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        // Should have 3 assertions
        assert_eq!(ctx.assertions.len(), 3);
    }

    #[test]
    fn test_elaborate_negative_in_multiplication() {
        // Test negative literals used in multiplication (* -1 x)
        let input = r#"
            (declare-const v0 Int)
            (declare-const v1 Int)
            (assert (= (+ (* 1 v0) (* 2 v1)) 38))
            (assert (>= (+ (* 2 v0) (* -1 v1)) -46))
            (assert (<= (+ (* -1 v0) (* -1 v1)) 39))
            (assert (<= (+ (* -4 v0) (* 5 v1)) -21))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        // Should have 4 assertions (from the QF_LIA benchmark)
        assert_eq!(ctx.assertions.len(), 4);
    }

    #[test]
    fn test_elaborate_negative_decimal_literals() {
        // Test that -3.14, -0.5, etc. are parsed correctly as negative decimals
        let input = r#"
            (declare-const x Real)
            (assert (= x -3.14))
            (assert (>= x -0.5))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        // Should have 2 assertions
        assert_eq!(ctx.assertions.len(), 2);
    }

    #[test]
    fn test_elaborate_negative_zero() {
        // -0 should be the same as 0
        let input = r#"
            (assert (= -0 0))
        "#;
        let commands = parse(input).unwrap();
        let mut ctx = Context::new();

        for cmd in &commands {
            ctx.process_command(cmd).unwrap();
        }

        // (= -0 0) should simplify to true
        assert_eq!(ctx.assertions.len(), 1);
        assert!(ctx.terms.is_true(ctx.assertions[0]));
    }
}
