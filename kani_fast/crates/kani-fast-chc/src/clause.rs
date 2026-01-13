//! CHC (Constrained Horn Clause) representation
//!
//! A Constrained Horn Clause has the form:
//!   P1(x1) ∧ P2(x2) ∧ ... ∧ Pn(xn) ∧ φ(x) => H(y)
//!
//! Where:
//! - P1..Pn are predicates (body atoms)
//! - φ is a constraint over free variables (SMT formula)
//! - H is the head predicate (or false for queries)

use kani_fast_kinduction::{SmtType, StateFormula};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;

/// SMT-LIB2 keywords that should not be sanitized as identifiers
const SMT_KEYWORDS: &[&str] = &[
    "true",
    "false",
    "and",
    "or",
    "not",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "+",
    "-",
    "*",
    "/",
    "mod",
    "div",
    "ite",
    "select",
    "store",
    "bvadd",
    "bvsub",
    "bvmul",
    "bvsdiv",
    "bvudiv",
    "bvsmod",
    "bvurem",
    "bvand",
    "bvor",
    "bvxor",
    "bvnot",
    "bvshl",
    "bvlshr",
    "bvashr",
    "bvult",
    "bvslt",
    "bvugt",
    "bvsgt",
    "bvule",
    "bvsle",
    "bvuge",
    "bvsge",
    "extract",
    "concat",
    "repeat",
    "zero_extend",
    "sign_extend",
    "deref",
    "Int",
    "Bool",
    "BitVec",
];

/// Check if a token is a number (integer or decimal)
fn is_smt_number(token: &str) -> bool {
    token
        .chars()
        .all(|c| c.is_numeric() || c == '-' || c == '.')
}

/// Check if a token is an SMT keyword
fn is_smt_keyword(token: &str) -> bool {
    SMT_KEYWORDS.contains(&token)
}

/// Sanitize a string to be a valid SMT-LIB2 identifier
/// Replaces non-alphanumeric characters (except '_') with '_'
pub fn sanitize_smt_identifier(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    // Collapse multiple consecutive underscores while preserving leading underscore
    let mut result = String::new();
    let mut last_was_underscore = false;

    for c in sanitized.chars() {
        if c == '_' {
            if !last_was_underscore {
                result.push(c);
                last_was_underscore = true;
            }
        } else {
            result.push(c);
            last_was_underscore = false;
        }
    }

    // SMT-LIB2 identifiers cannot start with a digit
    // If the result starts with a digit, prefix with underscore
    if result.bytes().next().is_some_and(|b| b.is_ascii_digit()) {
        result.insert(0, '_');
    }

    result
}

/// Sanitize an SMT expression by sanitizing identifiers within it
/// Handles S-expressions like "(+ x y)" while only touching identifiers
pub fn sanitize_smt_expr(expr: &str) -> String {
    // For simple identifiers (no spaces or parens), just sanitize directly
    if !expr.contains(' ') && !expr.contains('(') {
        return sanitize_smt_identifier(expr);
    }

    // For S-expressions, we need to be more careful
    // Replace identifier characters in the expression
    let mut result = String::with_capacity(expr.len());
    let mut current_token = String::new();

    for c in expr.chars() {
        if c == '(' || c == ')' || c == ' ' {
            // End of token, sanitize it if it's an identifier
            if !current_token.is_empty() {
                if is_smt_number(&current_token) || is_smt_keyword(&current_token) {
                    result.push_str(&current_token);
                } else {
                    result.push_str(&sanitize_smt_identifier(&current_token));
                }
                current_token.clear();
            }
            result.push(c);
        } else {
            current_token.push(c);
        }
    }

    // Handle remaining token
    if !current_token.is_empty() {
        if is_smt_number(&current_token) || is_smt_keyword(&current_token) {
            result.push_str(&current_token);
        } else {
            result.push_str(&sanitize_smt_identifier(&current_token));
        }
    }

    result
}

/// A predicate symbol with its signature
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Predicate {
    /// Predicate name
    pub name: String,
    /// Parameter types
    pub params: Vec<SmtType>,
}

impl Predicate {
    /// Create a new predicate
    pub fn new(name: impl Into<String>, params: Vec<SmtType>) -> Self {
        Self {
            name: name.into(),
            params,
        }
    }

    /// Generate SMT-LIB2 declaration
    pub fn to_smt_declaration(&self) -> String {
        let mut result = format!("(declare-fun {} (", self.name);
        let mut first = true;
        for t in &self.params {
            if first {
                first = false;
            } else {
                result.push(' ');
            }
            let _ = write!(result, "{}", t.to_smt_string());
        }
        result.push_str(") Bool)");
        result
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = &self.name;
        write!(f, "{name}")
    }
}

/// An application of a predicate to arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateApp {
    /// The predicate being applied
    pub predicate: String,
    /// Arguments (variable names or SMT expressions)
    pub args: Vec<String>,
}

impl PredicateApp {
    /// Create a new predicate application
    pub fn new(predicate: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
        }
    }

    /// Generate SMT-LIB2 expression
    pub fn to_smt(&self) -> String {
        if self.args.is_empty() {
            self.predicate.clone()
        } else {
            let mut result = format!("({}", self.predicate);
            for arg in &self.args {
                result.push(' ');
                let _ = write!(result, "{}", sanitize_smt_expr(arg));
            }
            result.push(')');
            result
        }
    }
}

impl fmt::Display for PredicateApp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let smt = self.to_smt();
        write!(f, "{smt}")
    }
}

/// A variable declaration for use in CHC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    /// Variable name
    pub name: String,
    /// Variable type
    pub smt_type: SmtType,
}

impl Variable {
    /// Create a new variable with the given name and SMT type.
    ///
    /// # Arguments
    /// * `name` - The variable name (used in SMT-LIB2 expressions)
    /// * `smt_type` - The SMT type of the variable (Int, Bool, BitVec, etc.)
    pub fn new(name: impl Into<String>, smt_type: SmtType) -> Self {
        Self {
            name: name.into(),
            smt_type,
        }
    }

    /// Generate SMT-LIB2 binding for use in forall
    pub fn to_smt_binding(&self) -> String {
        let sanitized_name = sanitize_smt_identifier(&self.name);
        format!("({} {})", sanitized_name, self.smt_type.to_smt_string())
    }

    /// Get the sanitized name for use in SMT-LIB2 expressions
    pub fn smt_name(&self) -> String {
        sanitize_smt_identifier(&self.name)
    }
}

/// Head of a Horn clause (either a predicate or false)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClauseHead {
    /// Regular predicate head
    Predicate(PredicateApp),
    /// Query (false head) - used to check for reachability of error states
    Query,
}

impl ClauseHead {
    pub fn to_smt(&self) -> String {
        match self {
            ClauseHead::Predicate(app) => app.to_smt(),
            ClauseHead::Query => "false".to_string(),
        }
    }
}

/// A Constrained Horn Clause
///
/// Represents: body_preds ∧ constraint => head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HornClause {
    /// Quantified variables
    pub variables: Vec<Variable>,
    /// Body predicates
    pub body_preds: Vec<PredicateApp>,
    /// Constraint (SMT formula over variables)
    pub constraint: StateFormula,
    /// Head of the clause
    pub head: ClauseHead,
    /// Optional name for debugging
    pub name: Option<String>,
}

impl HornClause {
    /// Create a new Horn clause
    pub fn new(
        variables: Vec<Variable>,
        body_preds: Vec<PredicateApp>,
        constraint: StateFormula,
        head: ClauseHead,
    ) -> Self {
        Self {
            variables,
            body_preds,
            constraint,
            head,
            name: None,
        }
    }

    /// Set a name for this clause
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Generate SMT-LIB2 assertion
    pub fn to_smt_assertion(&self) -> String {
        // Build the body
        let body = self.build_body();

        // Build the full clause
        let clause = format!("(=> {} {})", body, self.head.to_smt());

        // Wrap in forall if we have variables
        if self.variables.is_empty() {
            format!("(assert {})", clause)
        } else {
            let mut result = String::from("(assert (forall (");
            let mut first = true;
            for v in &self.variables {
                if first {
                    first = false;
                } else {
                    result.push(' ');
                }
                let _ = write!(result, "{}", v.to_smt_binding());
            }
            let _ = write!(result, ") {}))", clause);
            result
        }
    }

    fn build_body(&self) -> String {
        let mut parts = Vec::new();

        // Add body predicates
        for pred in &self.body_preds {
            parts.push(pred.to_smt());
        }

        // Add constraint if not trivially true
        if self.constraint.smt_formula != "true" {
            // Sanitize the constraint formula to ensure valid SMT-LIB2 identifiers
            parts.push(sanitize_smt_expr(&self.constraint.smt_formula));
        }

        match parts.len() {
            0 => "true".to_string(),
            // SAFETY: The match arm guarantees exactly one element
            1 => parts.pop().expect("match guarantees one element"),
            _ => format!("(and {})", parts.join(" ")),
        }
    }
}

impl fmt::Display for HornClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let smt = self.to_smt_assertion();
        write!(f, "{smt}")
    }
}

/// An uninterpreted function declaration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UninterpretedFunction {
    /// Function name (sanitized for SMT-LIB)
    pub name: String,
    /// Parameter types
    pub param_types: Vec<SmtType>,
    /// Return type
    pub return_type: SmtType,
}

impl UninterpretedFunction {
    /// Create a new uninterpreted function
    pub fn new(name: impl Into<String>, param_types: Vec<SmtType>, return_type: SmtType) -> Self {
        Self {
            name: name.into(),
            param_types,
            return_type,
        }
    }

    /// Generate SMT-LIB2 declaration
    pub fn to_smt_declaration(&self) -> String {
        let mut result = format!("(declare-fun {} (", self.name);
        let mut first = true;
        for t in &self.param_types {
            if first {
                first = false;
            } else {
                result.push(' ');
            }
            let _ = write!(result, "{}", t.to_smt_string());
        }
        let _ = write!(result, ") {})", self.return_type.to_smt_string());
        result
    }
}

/// A complete CHC system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChcSystem {
    /// Predicate declarations
    pub predicates: HashMap<String, Predicate>,
    /// Uninterpreted function declarations
    pub functions: HashMap<String, UninterpretedFunction>,
    /// Horn clauses
    pub clauses: Vec<HornClause>,
    /// Logic to use (typically HORN)
    pub logic: String,
    /// Axioms for intrinsic functions (SMT-LIB2 assertions)
    #[serde(default)]
    pub axioms: Vec<String>,
}

impl ChcSystem {
    /// Create a new CHC system
    pub fn new() -> Self {
        Self {
            predicates: HashMap::new(),
            functions: HashMap::new(),
            clauses: Vec::new(),
            logic: "HORN".to_string(),
            axioms: Vec::new(),
        }
    }

    /// Add a predicate declaration
    pub fn add_predicate(&mut self, pred: Predicate) {
        self.predicates.insert(pred.name.clone(), pred);
    }

    /// Add an uninterpreted function declaration
    pub fn add_function(&mut self, func: UninterpretedFunction) {
        self.functions.insert(func.name.clone(), func);
    }

    /// Add a Horn clause
    pub fn add_clause(&mut self, clause: HornClause) {
        self.clauses.push(clause);
    }

    /// Add an axiom (SMT-LIB2 assertion) for a function
    pub fn add_axiom(&mut self, axiom: String) {
        self.axioms.push(axiom);
    }

    /// Add multiple axioms
    pub fn add_axioms(&mut self, axioms: impl IntoIterator<Item = String>) {
        self.axioms.extend(axioms);
    }

    /// Check if this system has any query clauses
    pub fn has_query(&self) -> bool {
        self.clauses
            .iter()
            .any(|c| matches!(c.head, ClauseHead::Query))
    }

    /// Generate complete SMT-LIB2 CHC query
    pub fn to_smt2(&self) -> String {
        let mut output = String::new();

        // Set logic
        let _ = writeln!(output, "(set-logic {})\n", self.logic);

        // Declare predicates
        for pred in self.predicates.values() {
            output.push_str(&pred.to_smt_declaration());
            output.push('\n');
        }

        // Declare uninterpreted functions
        for func in self.functions.values() {
            output.push_str(&func.to_smt_declaration());
            output.push('\n');
        }

        // Add axioms for intrinsic functions (if any)
        if !self.axioms.is_empty() {
            output.push('\n');
            output.push_str("; axioms for intrinsic functions\n");
            for axiom in &self.axioms {
                output.push_str(axiom);
                output.push('\n');
            }
        }
        output.push('\n');

        // Add clauses
        for clause in &self.clauses {
            if let Some(name) = &clause.name {
                let _ = writeln!(output, "; {name}");
            }
            output.push_str(&clause.to_smt_assertion());
            output.push('\n');
        }
        output.push('\n');

        // Check satisfiability
        output.push_str("(check-sat)\n");
        output.push_str("(get-model)\n");

        output
    }
}

impl fmt::Display for ChcSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let smt2 = self.to_smt2();
        write!(f, "{smt2}")
    }
}

/// Builder for creating CHC systems
#[derive(Debug, Default)]
pub struct ChcSystemBuilder {
    system: ChcSystem,
}

impl ChcSystemBuilder {
    pub fn new() -> Self {
        Self {
            system: ChcSystem::new(),
        }
    }

    /// Add a predicate with given name and parameter types
    pub fn predicate(mut self, name: impl Into<String>, params: Vec<SmtType>) -> Self {
        self.system.add_predicate(Predicate::new(name, params));
        self
    }

    /// Add an initial clause: constraint => Inv(args)
    pub fn init(
        mut self,
        inv_name: impl Into<String>,
        args: Vec<String>,
        vars: Vec<Variable>,
        constraint: impl Into<String>,
    ) -> Self {
        let clause = HornClause::new(
            vars,
            vec![],
            StateFormula::new(constraint),
            ClauseHead::Predicate(PredicateApp::new(inv_name, args)),
        )
        .with_name("init");
        self.system.add_clause(clause);
        self
    }

    /// Add a transition clause: Inv(x) ∧ trans => Inv(x')
    pub fn transition(
        mut self,
        inv_name: impl Into<String>,
        current_args: Vec<String>,
        next_args: Vec<String>,
        vars: Vec<Variable>,
        constraint: impl Into<String>,
    ) -> Self {
        let inv_name = inv_name.into();
        let clause = HornClause::new(
            vars,
            vec![PredicateApp::new(&inv_name, current_args)],
            StateFormula::new(constraint),
            ClauseHead::Predicate(PredicateApp::new(&inv_name, next_args)),
        )
        .with_name("transition");
        self.system.add_clause(clause);
        self
    }

    /// Add a property clause: Inv(x) => property
    pub fn property(
        mut self,
        inv_name: impl Into<String>,
        args: Vec<String>,
        vars: Vec<Variable>,
        property: impl Into<String>,
    ) -> Self {
        // Property is asserted as: Inv(x) ∧ ¬property => false
        // Which means if we find a state where Inv holds but property doesn't, it's unsat
        let property_str = property.into();
        let negated = format!("(not {})", property_str);
        let clause = HornClause::new(
            vars,
            vec![PredicateApp::new(inv_name, args)],
            StateFormula::new(negated),
            ClauseHead::Query,
        )
        .with_name("property");
        self.system.add_clause(clause);
        self
    }

    /// Alternative property encoding: Inv(x) => property (direct implication)
    ///
    /// Test-only function - production code uses `property()` method instead.
    #[cfg(test)]
    pub fn property_direct(
        mut self,
        inv_name: impl Into<String>,
        args: Vec<String>,
        vars: Vec<Variable>,
        _property: impl Into<String>,
    ) -> Self {
        // Standard CHC encoding: Inv(x) => property
        // In Horn clause form: Inv(x) ∧ true => property
        // We encode as: ∀x. Inv(x) => property(x)
        // In SMT-LIB: (assert (forall ((x T)) (=> (Inv x) property)))
        let clause = HornClause {
            variables: vars,
            body_preds: vec![PredicateApp::new(inv_name, args)],
            constraint: StateFormula::true_formula(),
            head: ClauseHead::Predicate(PredicateApp::new("_property_check", vec![])),
            name: Some("property".to_string()),
        };
        // Actually, for property checking, we use the negated form
        // Inv(x) => P(x) is checked by asking: can Inv(x) ∧ ¬P(x) be reached?
        // But the standard CHC encoding is: (Inv(x) => P(x)) added as Horn clause
        // Z3 will find an Inv such that Init => Inv, Trans preserves Inv, and Inv => P
        self.system.add_clause(clause);
        self
    }

    pub fn build(self) -> ChcSystem {
        self.system
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Predicate Tests ====================

    #[test]
    fn test_predicate_declaration() {
        let pred = Predicate::new("Inv", vec![SmtType::Int, SmtType::Bool]);
        assert_eq!(
            pred.to_smt_declaration(),
            "(declare-fun Inv (Int Bool) Bool)"
        );
    }

    #[test]
    fn test_predicate_new() {
        let pred = Predicate::new("TestPred", vec![SmtType::Int]);
        assert_eq!(pred.name, "TestPred");
        assert_eq!(pred.params.len(), 1);
        assert_eq!(pred.params[0], SmtType::Int);
    }

    #[test]
    fn test_predicate_empty_params() {
        let pred = Predicate::new("NullaryPred", vec![]);
        assert_eq!(
            pred.to_smt_declaration(),
            "(declare-fun NullaryPred () Bool)"
        );
    }

    #[test]
    fn test_predicate_multiple_params() {
        let pred = Predicate::new("Multi", vec![SmtType::Int, SmtType::Bool, SmtType::Real]);
        assert_eq!(
            pred.to_smt_declaration(),
            "(declare-fun Multi (Int Bool Real) Bool)"
        );
    }

    #[test]
    fn test_predicate_bitvec_param() {
        let pred = Predicate::new("BvPred", vec![SmtType::BitVec(32)]);
        assert_eq!(
            pred.to_smt_declaration(),
            "(declare-fun BvPred ((_ BitVec 32)) Bool)"
        );
    }

    #[test]
    fn test_predicate_array_param() {
        let pred = Predicate::new(
            "ArrPred",
            vec![SmtType::Array {
                index: Box::new(SmtType::Int),
                element: Box::new(SmtType::Bool),
            }],
        );
        assert_eq!(
            pred.to_smt_declaration(),
            "(declare-fun ArrPred ((Array Int Bool)) Bool)"
        );
    }

    #[test]
    fn test_predicate_display() {
        let pred = Predicate::new("MyPred", vec![SmtType::Int]);
        assert_eq!(format!("{}", pred), "MyPred");
    }

    #[test]
    fn test_predicate_clone() {
        let pred = Predicate::new("Clone", vec![SmtType::Int]);
        let cloned = pred.clone();
        assert_eq!(pred.name, cloned.name);
        assert_eq!(pred.params, cloned.params);
    }

    #[test]
    fn test_predicate_debug() {
        let pred = Predicate::new("Debug", vec![SmtType::Int]);
        let debug_str = format!("{:?}", pred);
        assert!(debug_str.contains("Debug"));
        assert!(debug_str.contains("Int"));
    }

    #[test]
    fn test_predicate_serialize_deserialize() {
        let pred = Predicate::new("Serialize", vec![SmtType::Int, SmtType::Bool]);
        let json = serde_json::to_string(&pred).unwrap();
        let deserialized: Predicate = serde_json::from_str(&json).unwrap();
        assert_eq!(pred.name, deserialized.name);
        assert_eq!(pred.params, deserialized.params);
    }

    // ==================== PredicateApp Tests ====================

    #[test]
    fn test_predicate_app() {
        let app = PredicateApp::new("Inv", vec!["x".to_string(), "y".to_string()]);
        assert_eq!(app.to_smt(), "(Inv x y)");

        let app_no_args = PredicateApp::new("P", vec![]);
        assert_eq!(app_no_args.to_smt(), "P");
    }

    #[test]
    fn test_predicate_app_single_arg() {
        let app = PredicateApp::new("Single", vec!["arg".to_string()]);
        assert_eq!(app.to_smt(), "(Single arg)");
    }

    #[test]
    fn test_predicate_app_complex_args() {
        let app = PredicateApp::new(
            "Complex",
            vec!["(+ x 1)".to_string(), "(- y 2)".to_string()],
        );
        assert_eq!(app.to_smt(), "(Complex (+ x 1) (- y 2))");
    }

    #[test]
    fn test_predicate_app_display() {
        let app = PredicateApp::new("Display", vec!["a".to_string()]);
        assert_eq!(format!("{}", app), "(Display a)");
    }

    #[test]
    fn test_predicate_app_debug() {
        let app = PredicateApp::new("Debug", vec!["x".to_string()]);
        let debug_str = format!("{:?}", app);
        assert!(debug_str.contains("Debug"));
        assert!(debug_str.contains('x'));
    }

    #[test]
    fn test_predicate_app_clone() {
        let app = PredicateApp::new("Clone", vec!["a".to_string()]);
        let cloned = app.clone();
        assert_eq!(app.predicate, cloned.predicate);
        assert_eq!(app.args, cloned.args);
    }

    // ==================== Variable Tests ====================

    #[test]
    fn test_variable_new() {
        let var = Variable::new("x", SmtType::Int);
        assert_eq!(var.name, "x");
        assert_eq!(var.smt_type, SmtType::Int);
    }

    #[test]
    fn test_variable_to_smt_binding_int() {
        let var = Variable::new("x", SmtType::Int);
        assert_eq!(var.to_smt_binding(), "(x Int)");
    }

    #[test]
    fn test_variable_to_smt_binding_bool() {
        let var = Variable::new("flag", SmtType::Bool);
        assert_eq!(var.to_smt_binding(), "(flag Bool)");
    }

    #[test]
    fn test_variable_to_smt_binding_bitvec() {
        let var = Variable::new("bv", SmtType::BitVec(64));
        assert_eq!(var.to_smt_binding(), "(bv (_ BitVec 64))");
    }

    #[test]
    fn test_variable_clone() {
        let var = Variable::new("cloned", SmtType::Real);
        let cloned = var.clone();
        assert_eq!(var.name, cloned.name);
        assert_eq!(var.smt_type, cloned.smt_type);
    }

    #[test]
    fn test_variable_debug() {
        let var = Variable::new("debug_var", SmtType::Int);
        let debug_str = format!("{:?}", var);
        assert!(debug_str.contains("debug_var"));
    }

    // ==================== ClauseHead Tests ====================

    #[test]
    fn test_clause_head_predicate_to_smt() {
        let head = ClauseHead::Predicate(PredicateApp::new("Inv", vec!["x".to_string()]));
        assert_eq!(head.to_smt(), "(Inv x)");
    }

    #[test]
    fn test_clause_head_query_to_smt() {
        let head = ClauseHead::Query;
        assert_eq!(head.to_smt(), "false");
    }

    #[test]
    fn test_clause_head_clone() {
        let head = ClauseHead::Predicate(PredicateApp::new("Test", vec![]));
        let cloned = head.clone();
        assert_eq!(head.to_smt(), cloned.to_smt());
    }

    #[test]
    fn test_clause_head_debug() {
        let head = ClauseHead::Query;
        let debug_str = format!("{:?}", head);
        assert!(debug_str.contains("Query"));
    }

    // ==================== HornClause Tests ====================

    #[test]
    fn test_horn_clause_smt() {
        let clause = HornClause::new(
            vec![Variable::new("x", SmtType::Int)],
            vec![],
            StateFormula::new("(= x 0)"),
            ClauseHead::Predicate(PredicateApp::new("Inv", vec!["x".to_string()])),
        );

        let smt = clause.to_smt_assertion();
        assert!(smt.contains("forall"));
        assert!(smt.contains("(= x 0)"));
        assert!(smt.contains("(Inv x)"));
    }

    #[test]
    fn test_horn_clause_no_variables() {
        let clause = HornClause::new(
            vec![],
            vec![],
            StateFormula::new("true"),
            ClauseHead::Predicate(PredicateApp::new("P", vec![])),
        );

        let smt = clause.to_smt_assertion();
        assert!(!smt.contains("forall"));
        assert_eq!(smt, "(assert (=> true P))");
    }

    #[test]
    fn test_horn_clause_with_body_preds() {
        let clause = HornClause::new(
            vec![
                Variable::new("x", SmtType::Int),
                Variable::new("y", SmtType::Int),
            ],
            vec![
                PredicateApp::new("P", vec!["x".to_string()]),
                PredicateApp::new("Q", vec!["y".to_string()]),
            ],
            StateFormula::new("(> x y)"),
            ClauseHead::Predicate(PredicateApp::new(
                "R",
                vec!["x".to_string(), "y".to_string()],
            )),
        );

        let smt = clause.to_smt_assertion();
        assert!(smt.contains("(P x)"));
        assert!(smt.contains("(Q y)"));
        assert!(smt.contains("(> x y)"));
        assert!(smt.contains("(R x y)"));
    }

    #[test]
    fn test_horn_clause_true_constraint() {
        let clause = HornClause::new(
            vec![Variable::new("x", SmtType::Int)],
            vec![PredicateApp::new("P", vec!["x".to_string()])],
            StateFormula::true_formula(),
            ClauseHead::Predicate(PredicateApp::new("Q", vec!["x".to_string()])),
        );

        let smt = clause.to_smt_assertion();
        // With true constraint, body is just the predicate
        assert!(smt.contains("(P x)"));
        assert!(smt.contains("(Q x)"));
    }

    #[test]
    fn test_horn_clause_with_name() {
        let clause = HornClause::new(vec![], vec![], StateFormula::new("true"), ClauseHead::Query)
            .with_name("test_clause");

        assert_eq!(clause.name, Some("test_clause".to_string()));
    }

    #[test]
    fn test_horn_clause_display() {
        let clause = HornClause::new(
            vec![Variable::new("x", SmtType::Int)],
            vec![],
            StateFormula::new("(= x 0)"),
            ClauseHead::Predicate(PredicateApp::new("Inv", vec!["x".to_string()])),
        );

        let display = format!("{}", clause);
        assert!(display.contains("forall"));
        assert!(display.contains("(Inv x)"));
    }

    #[test]
    fn test_horn_clause_clone() {
        let clause = HornClause::new(
            vec![Variable::new("x", SmtType::Int)],
            vec![],
            StateFormula::new("test"),
            ClauseHead::Query,
        );
        let cloned = clause.clone();
        assert_eq!(clause.variables.len(), cloned.variables.len());
        assert_eq!(clause.constraint.smt_formula, cloned.constraint.smt_formula);
    }

    #[test]
    fn test_horn_clause_query_head() {
        let clause = HornClause::new(
            vec![Variable::new("x", SmtType::Int)],
            vec![PredicateApp::new("Inv", vec!["x".to_string()])],
            StateFormula::new("(< x 0)"),
            ClauseHead::Query,
        );

        let smt = clause.to_smt_assertion();
        assert!(smt.contains("false")); // Query head is false
    }

    // ==================== ChcSystem Tests ====================

    #[test]
    fn test_chc_system_new() {
        let system = ChcSystem::new();
        assert!(system.predicates.is_empty());
        assert!(system.clauses.is_empty());
        assert_eq!(system.logic, "HORN");
    }

    #[test]
    fn test_chc_system_default() {
        let system = ChcSystem::default();
        assert!(system.predicates.is_empty());
        assert!(system.clauses.is_empty());
        // Default uses String::default() which is empty
        // ChcSystem::new() sets logic to "HORN"
        assert!(system.logic.is_empty());
    }

    #[test]
    fn test_chc_system_add_predicate() {
        let mut system = ChcSystem::new();
        system.add_predicate(Predicate::new("P", vec![SmtType::Int]));

        assert!(system.predicates.contains_key("P"));
        assert_eq!(system.predicates.get("P").unwrap().params.len(), 1);
    }

    #[test]
    fn test_chc_system_add_clause() {
        let mut system = ChcSystem::new();
        let clause = HornClause::new(vec![], vec![], StateFormula::new("true"), ClauseHead::Query);
        system.add_clause(clause);

        assert_eq!(system.clauses.len(), 1);
    }

    #[test]
    fn test_chc_system_builder() {
        let system = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .init(
                "Inv",
                vec!["x".to_string()],
                vec![Variable::new("x", SmtType::Int)],
                "(= x 0)",
            )
            .transition(
                "Inv",
                vec!["x".to_string()],
                vec!["x1".to_string()],
                vec![
                    Variable::new("x", SmtType::Int),
                    Variable::new("x1", SmtType::Int),
                ],
                "(= x1 (+ x 1))",
            )
            .property(
                "Inv",
                vec!["x".to_string()],
                vec![Variable::new("x", SmtType::Int)],
                "(>= x 0)",
            )
            .build();

        let smt2 = system.to_smt2();
        assert!(smt2.contains("(set-logic HORN)"));
        assert!(smt2.contains("(declare-fun Inv (Int) Bool)"));
        assert!(smt2.contains("(check-sat)"));
    }

    #[test]
    fn test_query_detection() {
        let mut system = ChcSystem::new();
        assert!(!system.has_query());

        system.add_clause(HornClause::new(
            vec![],
            vec![],
            StateFormula::true_formula(),
            ClauseHead::Query,
        ));
        assert!(system.has_query());
    }

    #[test]
    fn test_chc_system_to_smt2_structure() {
        let system = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .build();

        let smt2 = system.to_smt2();

        // Check structure
        assert!(smt2.lines().next().unwrap().contains("set-logic"));
        assert!(smt2.contains("check-sat"));
        assert!(smt2.contains("get-model"));
    }

    #[test]
    fn test_chc_system_to_smt2_with_named_clause() {
        let mut system = ChcSystem::new();
        system.add_predicate(Predicate::new("P", vec![]));

        let clause = HornClause::new(
            vec![],
            vec![],
            StateFormula::new("true"),
            ClauseHead::Predicate(PredicateApp::new("P", vec![])),
        )
        .with_name("test_clause");
        system.add_clause(clause);

        let smt2 = system.to_smt2();
        assert!(smt2.contains("; test_clause"));
    }

    #[test]
    fn test_chc_system_display() {
        let system = ChcSystemBuilder::new().predicate("P", vec![]).build();

        let display = format!("{}", system);
        assert!(display.contains("set-logic"));
    }

    #[test]
    fn test_chc_system_clone() {
        let system = ChcSystemBuilder::new()
            .predicate("P", vec![SmtType::Int])
            .build();

        let cloned = system.clone();
        assert_eq!(system.predicates.len(), cloned.predicates.len());
        assert_eq!(system.clauses.len(), cloned.clauses.len());
    }

    #[test]
    fn test_chc_system_serialize() {
        let system = ChcSystemBuilder::new()
            .predicate("P", vec![SmtType::Int])
            .build();

        let json = serde_json::to_string(&system).unwrap();
        assert!(json.contains("predicates"));
        assert!(json.contains("clauses"));
    }

    // ==================== ChcSystemBuilder Tests ====================

    #[test]
    fn test_chc_system_builder_new() {
        let builder = ChcSystemBuilder::new();
        let system = builder.build();
        assert!(system.predicates.is_empty());
        assert!(system.clauses.is_empty());
    }

    #[test]
    fn test_chc_system_builder_default() {
        let builder = ChcSystemBuilder::default();
        let system = builder.build();
        assert!(system.predicates.is_empty());
    }

    #[test]
    fn test_chc_system_builder_multiple_predicates() {
        let system = ChcSystemBuilder::new()
            .predicate("P", vec![SmtType::Int])
            .predicate("Q", vec![SmtType::Bool])
            .predicate("R", vec![SmtType::Int, SmtType::Int])
            .build();

        assert_eq!(system.predicates.len(), 3);
        assert!(system.predicates.contains_key("P"));
        assert!(system.predicates.contains_key("Q"));
        assert!(system.predicates.contains_key("R"));
    }

    #[test]
    fn test_chc_system_builder_init_creates_clause() {
        let system = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .init(
                "Inv",
                vec!["x".to_string()],
                vec![Variable::new("x", SmtType::Int)],
                "(= x 0)",
            )
            .build();

        assert_eq!(system.clauses.len(), 1);
        assert_eq!(system.clauses[0].name, Some("init".to_string()));
    }

    #[test]
    fn test_chc_system_builder_transition_creates_clause() {
        let system = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .transition(
                "Inv",
                vec!["x".to_string()],
                vec!["x_next".to_string()],
                vec![
                    Variable::new("x", SmtType::Int),
                    Variable::new("x_next", SmtType::Int),
                ],
                "(= x_next (+ x 1))",
            )
            .build();

        assert_eq!(system.clauses.len(), 1);
        assert_eq!(system.clauses[0].name, Some("transition".to_string()));
    }

    #[test]
    fn test_chc_system_builder_property_creates_query() {
        let system = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .property(
                "Inv",
                vec!["x".to_string()],
                vec![Variable::new("x", SmtType::Int)],
                "(>= x 0)",
            )
            .build();

        assert_eq!(system.clauses.len(), 1);
        assert!(system.has_query());
        // Property creates negated query
        let smt = system.clauses[0].to_smt_assertion();
        assert!(smt.contains("(not (>= x 0))"));
    }

    #[test]
    fn test_chc_system_builder_complete_system() {
        let system = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .init(
                "Inv",
                vec!["x".to_string()],
                vec![Variable::new("x", SmtType::Int)],
                "(= x 0)",
            )
            .transition(
                "Inv",
                vec!["x".to_string()],
                vec!["x_next".to_string()],
                vec![
                    Variable::new("x", SmtType::Int),
                    Variable::new("x_next", SmtType::Int),
                ],
                "(= x_next (+ x 1))",
            )
            .property(
                "Inv",
                vec!["x".to_string()],
                vec![Variable::new("x", SmtType::Int)],
                "(>= x 0)",
            )
            .build();

        assert_eq!(system.predicates.len(), 1);
        assert_eq!(system.clauses.len(), 3);
        assert!(system.has_query());
    }

    #[test]
    fn test_chc_system_builder_chaining() {
        // Test that builder methods can be chained
        let system = ChcSystemBuilder::new()
            .predicate("A", vec![])
            .predicate("B", vec![SmtType::Int])
            .predicate("C", vec![SmtType::Bool])
            .build();

        assert_eq!(system.predicates.len(), 3);
    }

    #[test]
    fn test_chc_system_builder_debug() {
        let builder = ChcSystemBuilder::new();
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("ChcSystemBuilder"));
    }

    // ============================================================
    // Mutation coverage tests
    // ============================================================

    /// Test sanitize_smt_expr simple identifier path (no space, no paren)
    /// Catches: clause.rs:65:28 replace && with ||
    /// Catches: clause.rs:65:8 delete ! in condition
    /// Catches: clause.rs:65:31 delete ! in condition
    #[test]
    fn test_sanitize_smt_expr_simple_identifier_path() {
        // Simple identifier with no spaces and no parens takes early return
        assert_eq!(sanitize_smt_expr("my_var"), "my_var");
        // With space - should NOT take simple path
        let result = sanitize_smt_expr("a b");
        assert!(result.contains(' '), "Should preserve space structure");
        // With paren - should NOT take simple path
        let result2 = sanitize_smt_expr("(x)");
        assert!(result2.contains('('), "Should preserve paren structure");
    }

    /// Test sanitize_smt_expr correctly identifies numbers in S-expressions
    /// Catches: clause.rs:81:62 replace == with !=
    #[test]
    fn test_sanitize_smt_expr_number_detection() {
        // Numbers inside S-expressions should NOT be sanitized
        assert_eq!(sanitize_smt_expr("(+ 42 x)"), "(+ 42 x)");
        // Numbers should be preserved in expressions
        assert_eq!(sanitize_smt_expr("(* 123 y)"), "(* 123 y)");
        // Negative numbers in expressions
        assert_eq!(sanitize_smt_expr("(+ -456 z)"), "(+ -456 z)");
        // Mixed: number stays, identifier stays
        assert_eq!(sanitize_smt_expr("(+ 1 my_var)"), "(+ 1 my_var)");
    }

    /// Test sanitize_smt_expr trailing token handling
    /// Catches: clause.rs:150:8 delete ! in condition (remaining token)
    #[test]
    fn test_sanitize_smt_expr_trailing_token_handling() {
        // Expression with trailing token (no final delimiter)
        let result = sanitize_smt_expr("(+ x y)");
        assert!(result.ends_with(')'), "Should preserve trailing structure");

        // Expression ending with identifier
        let result2 = sanitize_smt_expr("x");
        assert_eq!(result2, "x");
    }

    /// Test sanitize_smt_expr keyword detection in trailing token
    /// Catches: clause.rs:153:49 replace || with &&
    /// Catches: clause.rs:153:37 replace || with &&
    /// Catches: clause.rs:153:42 replace == with !=
    /// Catches: clause.rs:153:54 replace == with !=
    #[test]
    fn test_sanitize_smt_expr_trailing_keyword_vs_identifier() {
        // Keywords should NOT be sanitized
        assert_eq!(sanitize_smt_expr("true"), "true");
        assert_eq!(sanitize_smt_expr("false"), "false");
        assert_eq!(sanitize_smt_expr("and"), "and");

        // Non-keywords should be sanitized if they contain special chars
        // Test that identifier-like strings pass through correctly
        assert_eq!(sanitize_smt_expr("my_func"), "my_func");
    }

    /// Test sanitize_smt_expr with mixed number and keyword conditions in expressions
    /// Catches: clause.rs:208:22 replace || with &&
    #[test]
    fn test_sanitize_smt_expr_number_or_keyword_disjunction() {
        // Numbers and keywords as trailing tokens in expressions
        // (trailing token = token at end of expression after all parens)
        // A number as trailing token should pass through (is_number || is_keyword)
        assert_eq!(sanitize_smt_expr("(= x 42)"), "(= x 42)");
        // A keyword as trailing token should pass through
        assert_eq!(sanitize_smt_expr("(and true false)"), "(and true false)");
        // Something that is NEITHER should be treated as identifier
        // (identifiers are passed through sanitize_smt_identifier, which preserves valid names)
        assert_eq!(sanitize_smt_expr("(call my_func x)"), "(call my_func x)");
    }

    /// Test Variable::smt_name returns sanitized name
    /// Catches: clause.rs:321:9 replace Variable::smt_name -> String with String::new()
    /// Catches: clause.rs:321:9 replace Variable::smt_name -> String with "xyzzy".into()
    #[test]
    fn test_variable_smt_name_returns_sanitized() {
        let var = Variable::new("my_var", SmtType::Int);
        let smt_name = var.smt_name();
        assert_eq!(smt_name, "my_var");
        assert!(!smt_name.is_empty(), "smt_name should not be empty");
        assert_ne!(smt_name, "xyzzy", "smt_name should match variable name");
    }

    /// Test HornClause::build_body match arm for multiple parts
    /// Catches: clause.rs:417:13 delete match arm 1 in HornClause::build_body
    #[test]
    fn test_horn_clause_build_body_single_vs_multiple_parts() {
        // Clause with NO body predicates and trivial constraint -> should return "true"
        let clause_empty = HornClause {
            variables: vec![],
            body_preds: vec![],
            constraint: StateFormula::true_formula(),
            head: ClauseHead::Query,
            name: None,
        };
        assert_eq!(clause_empty.build_body(), "true");

        // Clause with ONE body predicate -> should return that predicate directly
        let clause_one = HornClause {
            variables: vec![Variable::new("x", SmtType::Int)],
            body_preds: vec![PredicateApp::new("Inv", vec!["x".to_string()])],
            constraint: StateFormula::true_formula(),
            head: ClauseHead::Query,
            name: None,
        };
        let body_one = clause_one.build_body();
        assert_eq!(body_one, "(Inv x)");
        assert!(
            !body_one.starts_with("(and"),
            "Single part should not use 'and'"
        );

        // Clause with MULTIPLE body predicates -> should use (and ...)
        let clause_multi = HornClause {
            variables: vec![Variable::new("x", SmtType::Int)],
            body_preds: vec![
                PredicateApp::new("A", vec!["x".to_string()]),
                PredicateApp::new("B", vec!["x".to_string()]),
            ],
            constraint: StateFormula::true_formula(),
            head: ClauseHead::Query,
            name: None,
        };
        let body_multi = clause_multi.build_body();
        assert!(
            body_multi.starts_with("(and"),
            "Multiple parts should use 'and'"
        );
    }

    /// Test ChcSystem::add_axiom actually adds the axiom
    /// Catches: clause.rs:507:9 replace ChcSystem::add_axiom with ()
    #[test]
    fn test_chc_system_add_axiom_actually_adds() {
        let mut system = ChcSystem::new();
        assert!(system.axioms.is_empty());

        system.add_axiom("(assert (= 1 1))".to_string());
        assert_eq!(system.axioms.len(), 1);
        assert_eq!(system.axioms[0], "(assert (= 1 1))");
    }

    /// Test ChcSystem::add_axioms actually adds multiple axioms
    /// Catches: clause.rs:512:9 replace ChcSystem::add_axioms with ()
    #[test]
    fn test_chc_system_add_axioms_actually_adds() {
        let mut system = ChcSystem::new();
        assert!(system.axioms.is_empty());

        system.add_axioms(vec![
            "(assert (= 1 1))".to_string(),
            "(assert (= 2 2))".to_string(),
        ]);
        assert_eq!(system.axioms.len(), 2);
    }

    /// Test ChcSystem::to_smt2 includes axioms section when axioms exist
    /// Catches: clause.rs:542:12 delete ! in ChcSystem::to_smt2
    #[test]
    fn test_chc_system_to_smt2_includes_axioms_when_present() {
        let mut system = ChcSystem::new();
        system.add_axiom("(assert (= x 0))".to_string());

        let smt2 = system.to_smt2();
        assert!(
            smt2.contains("axioms for intrinsic"),
            "Should include axiom section header when axioms present"
        );
        assert!(
            smt2.contains("(assert (= x 0))"),
            "Should include the actual axiom"
        );

        // Without axioms, should NOT have the header
        let empty_system = ChcSystem::new();
        let empty_smt2 = empty_system.to_smt2();
        assert!(
            !empty_smt2.contains("axioms for intrinsic"),
            "Should not include axiom section when no axioms"
        );
    }

    /// Test ChcSystemBuilder::property_direct returns modified self (not default)
    /// Catches: clause.rs:671:9 replace ChcSystemBuilder::property_direct -> Self with Default::default()
    #[test]
    fn test_chc_system_builder_property_direct_returns_modified() {
        let builder = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .property_direct(
                "Inv",
                vec!["x".to_string()],
                vec![Variable::new("x", SmtType::Int)],
                "(>= x 0)",
            );

        let system = builder.build();
        // If property_direct returned Default::default(), clauses would be empty
        assert!(
            !system.clauses.is_empty(),
            "property_direct should add a clause"
        );
        // Also check predicate was preserved from before property_direct
        assert!(
            system.predicates.contains_key("Inv"),
            "Should preserve predicates added before property_direct"
        );
    }
}
