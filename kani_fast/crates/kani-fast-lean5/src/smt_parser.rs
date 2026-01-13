//! SMT-LIB2 formula parser
//!
//! Parses SMT-LIB2 formulas into an AST suitable for translation to Lean5.

use std::fmt;
use thiserror::Error;

/// SMT-LIB2 Abstract Syntax Tree
#[derive(Debug, Clone, PartialEq)]
pub enum SmtAst {
    /// Symbol (variable or constant name)
    Symbol(String),
    /// Integer literal
    Int(i64),
    /// Boolean literal
    Bool(bool),
    /// Negation (- n) for negative numbers
    Neg(Box<SmtAst>),
    /// S-expression application (operator args...)
    App(String, Vec<SmtAst>),
    /// Let binding: (let ((var val)...) body)
    Let(Vec<(String, SmtAst)>, Box<SmtAst>),
    /// Forall quantifier: (forall ((var type)...) body)
    Forall(Vec<(String, SmtSort)>, Box<SmtAst>),
    /// Exists quantifier: (exists ((var type)...) body)
    Exists(Vec<(String, SmtSort)>, Box<SmtAst>),
}

/// SMT-LIB2 sort (type)
#[derive(Debug, Clone, PartialEq)]
pub enum SmtSort {
    /// Int
    Int,
    /// Bool
    Bool,
    /// Real
    Real,
    /// BitVec with width
    BitVec(u32),
    /// Array sort
    Array(Box<SmtSort>, Box<SmtSort>),
    /// Unknown/unresolved sort
    Unknown(String),
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtSort::Int => write!(f, "Int"),
            SmtSort::Bool => write!(f, "Bool"),
            SmtSort::Real => write!(f, "Real"),
            SmtSort::BitVec(w) => write!(f, "(_ BitVec {w})"),
            SmtSort::Array(k, v) => write!(f, "(Array {k} {v})"),
            SmtSort::Unknown(s) => write!(f, "{s}"),
        }
    }
}

impl fmt::Display for SmtAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtAst::Symbol(s) => write!(f, "{s}"),
            SmtAst::Int(n) => {
                if *n < 0 {
                    write!(f, "(- {})", -n)
                } else {
                    write!(f, "{n}")
                }
            }
            SmtAst::Bool(b) => write!(f, "{b}"),
            SmtAst::Neg(inner) => write!(f, "(- {inner})"),
            SmtAst::App(op, args) => {
                write!(f, "({op}")?;
                for arg in args {
                    write!(f, " {arg}")?;
                }
                write!(f, ")")
            }
            SmtAst::Let(bindings, body) => {
                write!(f, "(let (")?;
                for (name, val) in bindings {
                    write!(f, "({name} {val})")?;
                }
                write!(f, ") {body})")
            }
            SmtAst::Forall(vars, body) => {
                write!(f, "(forall (")?;
                for (name, sort) in vars {
                    write!(f, "({name} {sort})")?;
                }
                write!(f, ") {body})")
            }
            SmtAst::Exists(vars, body) => {
                write!(f, "(exists (")?;
                for (name, sort) in vars {
                    write!(f, "({name} {sort})")?;
                }
                write!(f, ") {body})")
            }
        }
    }
}

/// Parse an SMT-LIB2 formula string into an AST
pub fn parse_smt_formula(input: &str) -> Result<SmtAst, ParseError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(ParseError::EmptyInput);
    }

    let (ast, rest) = parse_sexp(input)?;
    let rest = rest.trim();
    if !rest.is_empty() {
        return Err(ParseError::TrailingInput(rest.to_string()));
    }
    Ok(ast)
}

/// Parse error
#[derive(Debug, Clone, Error)]
pub enum ParseError {
    #[error("empty input")]
    EmptyInput,
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unexpected character: '{0}'")]
    UnexpectedChar(char),
    #[error("unmatched parenthesis")]
    UnmatchedParen,
    #[error("invalid number: {0}")]
    InvalidNumber(String),
    #[error("trailing input: {0}")]
    TrailingInput(String),
    #[error("invalid let expression")]
    InvalidLet,
    #[error("invalid quantifier")]
    InvalidQuantifier,
}

/// Parse a single S-expression, returning the AST and remaining input
fn parse_sexp(input: &str) -> Result<(SmtAst, &str), ParseError> {
    let input = input.trim_start();
    if input.is_empty() {
        return Err(ParseError::UnexpectedEof);
    }

    let first = input.chars().next().unwrap();

    if first == '(' {
        parse_list(&input[1..])
    } else if first == ')' {
        Err(ParseError::UnexpectedChar(')'))
    } else {
        parse_atom(input)
    }
}

/// Parse an atom (symbol, number, or boolean)
fn parse_atom(input: &str) -> Result<(SmtAst, &str), ParseError> {
    let input = input.trim_start();

    // Find the end of the atom
    let end = input
        .find(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .unwrap_or(input.len());

    if end == 0 {
        return Err(ParseError::UnexpectedEof);
    }

    let atom = &input[..end];
    let rest = &input[end..];

    // Try to parse as different types
    if atom == "true" {
        return Ok((SmtAst::Bool(true), rest));
    }
    if atom == "false" {
        return Ok((SmtAst::Bool(false), rest));
    }

    // Try as integer
    if let Ok(n) = atom.parse::<i64>() {
        return Ok((SmtAst::Int(n), rest));
    }

    // It's a symbol
    Ok((SmtAst::Symbol(atom.to_string()), rest))
}

/// Parse a list (S-expression starting with `(`)
fn parse_list(input: &str) -> Result<(SmtAst, &str), ParseError> {
    let input = input.trim_start();
    if input.is_empty() {
        return Err(ParseError::UnexpectedEof);
    }

    // Check for closing paren (empty list - not valid SMT but handle gracefully)
    if input.starts_with(')') {
        return Err(ParseError::UnexpectedChar(')'));
    }

    // Parse the first element to determine what kind of expression this is
    let (first, mut rest) = parse_sexp(input)?;

    // Check for special forms
    if let SmtAst::Symbol(ref op) = first {
        match op.as_str() {
            "let" => return parse_let(rest),
            "forall" => return parse_quantifier(rest, true),
            "exists" => return parse_quantifier(rest, false),
            "-" => {
                // Check if this is unary negation
                let rest_trimmed = rest.trim_start();
                if !rest_trimmed.starts_with(')') {
                    let (arg, new_rest) = parse_sexp(rest)?;
                    let new_rest = new_rest.trim_start();
                    if let Some(after_paren) = new_rest.strip_prefix(')') {
                        // Unary minus (negation)
                        return Ok((SmtAst::Neg(Box::new(arg)), after_paren));
                    }
                    // Binary minus - continue with normal app parsing
                    rest = new_rest;
                    let mut args = vec![arg];
                    while !rest.trim_start().starts_with(')') {
                        let (arg, new_rest) = parse_sexp(rest)?;
                        args.push(arg);
                        rest = new_rest;
                    }
                    let trimmed = rest.trim_start();
                    return Ok((
                        SmtAst::App("-".to_string(), args),
                        trimmed.strip_prefix(')').unwrap_or(trimmed),
                    ));
                }
            }
            _ => {}
        }
    }

    // Regular application
    let op = match first {
        SmtAst::Symbol(s) => s,
        _ => return Err(ParseError::UnexpectedChar('(')),
    };

    let mut args = Vec::new();
    rest = rest.trim_start();

    while !rest.starts_with(')') {
        if rest.is_empty() {
            return Err(ParseError::UnmatchedParen);
        }
        let (arg, new_rest) = parse_sexp(rest)?;
        args.push(arg);
        rest = new_rest.trim_start();
    }

    // Skip the closing paren
    Ok((SmtAst::App(op, args), &rest[1..]))
}

/// Parse a let expression: (let ((var val)...) body)
fn parse_let(input: &str) -> Result<(SmtAst, &str), ParseError> {
    let input = input.trim_start();

    // Expect opening paren for bindings
    if !input.starts_with('(') {
        return Err(ParseError::InvalidLet);
    }

    let mut rest = &input[1..];
    let mut bindings = Vec::new();

    // Parse bindings until closing paren
    rest = rest.trim_start();
    while !rest.starts_with(')') {
        if rest.is_empty() {
            return Err(ParseError::UnmatchedParen);
        }

        // Each binding is (name value)
        if !rest.starts_with('(') {
            return Err(ParseError::InvalidLet);
        }
        rest = rest[1..].trim_start();

        // Parse name
        let (name_ast, new_rest) = parse_sexp(rest)?;
        let name = match name_ast {
            SmtAst::Symbol(s) => s,
            _ => return Err(ParseError::InvalidLet),
        };
        rest = new_rest.trim_start();

        // Parse value
        let (value, new_rest) = parse_sexp(rest)?;
        rest = new_rest.trim_start();

        // Expect closing paren for this binding
        if !rest.starts_with(')') {
            return Err(ParseError::InvalidLet);
        }
        rest = rest[1..].trim_start();

        bindings.push((name, value));
    }

    // Skip closing paren of bindings list
    rest = rest[1..].trim_start();

    // Parse body
    let (body, rest) = parse_sexp(rest)?;
    let rest = rest.trim_start();

    // Expect closing paren of let
    if !rest.starts_with(')') {
        return Err(ParseError::UnmatchedParen);
    }

    Ok((SmtAst::Let(bindings, Box::new(body)), &rest[1..]))
}

/// Parse a quantifier: (forall/exists ((var type)...) body)
fn parse_quantifier(input: &str, is_forall: bool) -> Result<(SmtAst, &str), ParseError> {
    let input = input.trim_start();

    // Expect opening paren for variable list
    if !input.starts_with('(') {
        return Err(ParseError::InvalidQuantifier);
    }

    let mut rest = &input[1..];
    let mut vars = Vec::new();

    // Parse variables until closing paren
    rest = rest.trim_start();
    while !rest.starts_with(')') {
        if rest.is_empty() {
            return Err(ParseError::UnmatchedParen);
        }

        // Each variable is (name type)
        if !rest.starts_with('(') {
            return Err(ParseError::InvalidQuantifier);
        }
        rest = rest[1..].trim_start();

        // Parse name
        let (name_ast, new_rest) = parse_sexp(rest)?;
        let name = match name_ast {
            SmtAst::Symbol(s) => s,
            _ => return Err(ParseError::InvalidQuantifier),
        };
        rest = new_rest.trim_start();

        // Parse type
        let (type_ast, new_rest) = parse_sexp(rest)?;
        let sort = ast_to_sort(&type_ast);
        rest = new_rest.trim_start();

        // Expect closing paren for this variable
        if !rest.starts_with(')') {
            return Err(ParseError::InvalidQuantifier);
        }
        rest = rest[1..].trim_start();

        vars.push((name, sort));
    }

    // Skip closing paren of variable list
    rest = rest[1..].trim_start();

    // Parse body
    let (body, rest) = parse_sexp(rest)?;
    let rest = rest.trim_start();

    // Expect closing paren of quantifier
    if !rest.starts_with(')') {
        return Err(ParseError::UnmatchedParen);
    }

    let ast = if is_forall {
        SmtAst::Forall(vars, Box::new(body))
    } else {
        SmtAst::Exists(vars, Box::new(body))
    };

    Ok((ast, &rest[1..]))
}

/// Convert an AST node to a sort
fn ast_to_sort(ast: &SmtAst) -> SmtSort {
    match ast {
        SmtAst::Symbol(s) => match s.as_str() {
            "Int" => SmtSort::Int,
            "Bool" => SmtSort::Bool,
            "Real" => SmtSort::Real,
            _ => SmtSort::Unknown(s.clone()),
        },
        SmtAst::App(op, args) => {
            if op == "_" && args.len() == 2 {
                // (_ BitVec N)
                if let SmtAst::Symbol(ref bv) = args[0] {
                    if bv == "BitVec" {
                        if let SmtAst::Int(w) = args[1] {
                            return SmtSort::BitVec(w as u32);
                        }
                    }
                }
            } else if op == "Array" && args.len() == 2 {
                let key_sort = ast_to_sort(&args[0]);
                let val_sort = ast_to_sort(&args[1]);
                return SmtSort::Array(Box::new(key_sort), Box::new(val_sort));
            }
            SmtSort::Unknown(format!("{ast}"))
        }
        _ => SmtSort::Unknown(format!("{ast}")),
    }
}

impl SmtAst {
    /// Check if this is a simple formula (no let bindings or quantifiers)
    pub fn is_simple(&self) -> bool {
        match self {
            SmtAst::Symbol(_) | SmtAst::Int(_) | SmtAst::Bool(_) => true,
            SmtAst::Neg(inner) => inner.is_simple(),
            SmtAst::App(_, args) => args.iter().all(|a| a.is_simple()),
            SmtAst::Let(_, _) | SmtAst::Forall(_, _) | SmtAst::Exists(_, _) => false,
        }
    }

    /// Get all free variables in the formula
    pub fn free_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_free_vars(&mut vars, &[]);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_free_vars(&self, vars: &mut Vec<String>, bound: &[&str]) {
        match self {
            SmtAst::Symbol(s) => {
                if !bound.contains(&s.as_str()) && !is_smt_builtin(s) {
                    vars.push(s.clone());
                }
            }
            SmtAst::Int(_) | SmtAst::Bool(_) => {}
            SmtAst::Neg(inner) => inner.collect_free_vars(vars, bound),
            SmtAst::App(_, args) => {
                for arg in args {
                    arg.collect_free_vars(vars, bound);
                }
            }
            SmtAst::Let(bindings, body) => {
                // First collect from binding values (not yet bound)
                for (_, val) in bindings {
                    val.collect_free_vars(vars, bound);
                }
                // Then collect from body with new bindings
                let mut new_bound: Vec<&str> = bound.to_vec();
                for (name, _) in bindings {
                    new_bound.push(name.as_str());
                }
                body.collect_free_vars(vars, &new_bound);
            }
            SmtAst::Forall(qvars, body) | SmtAst::Exists(qvars, body) => {
                let mut new_bound: Vec<&str> = bound.to_vec();
                for (name, _) in qvars {
                    new_bound.push(name.as_str());
                }
                body.collect_free_vars(vars, &new_bound);
            }
        }
    }

    /// Substitute a variable with an expression
    pub fn subst(&self, var: &str, replacement: &SmtAst) -> SmtAst {
        match self {
            SmtAst::Symbol(s) if s == var => replacement.clone(),
            SmtAst::Symbol(_) | SmtAst::Int(_) | SmtAst::Bool(_) => self.clone(),
            SmtAst::Neg(inner) => SmtAst::Neg(Box::new(inner.subst(var, replacement))),
            SmtAst::App(op, args) => SmtAst::App(
                op.clone(),
                args.iter().map(|a| a.subst(var, replacement)).collect(),
            ),
            SmtAst::Let(bindings, body) => {
                // Check if var is shadowed
                let is_shadowed = bindings.iter().any(|(n, _)| n == var);
                let new_bindings: Vec<_> = bindings
                    .iter()
                    .map(|(n, v)| (n.clone(), v.subst(var, replacement)))
                    .collect();
                let new_body = if is_shadowed {
                    body.as_ref().clone()
                } else {
                    body.subst(var, replacement)
                };
                SmtAst::Let(new_bindings, Box::new(new_body))
            }
            SmtAst::Forall(qvars, body) | SmtAst::Exists(qvars, body) => {
                let is_shadowed = qvars.iter().any(|(n, _)| n == var);
                let new_body = if is_shadowed {
                    body.as_ref().clone()
                } else {
                    body.subst(var, replacement)
                };
                if matches!(self, SmtAst::Forall(_, _)) {
                    SmtAst::Forall(qvars.clone(), Box::new(new_body))
                } else {
                    SmtAst::Exists(qvars.clone(), Box::new(new_body))
                }
            }
        }
    }
}

/// Check if a name is an SMT built-in operator
fn is_smt_builtin(name: &str) -> bool {
    matches!(
        name,
        "and"
            | "or"
            | "not"
            | "=>"
            | "="
            | "distinct"
            | "ite"
            | "+"
            | "-"
            | "*"
            | "/"
            | "div"
            | "mod"
            | "abs"
            | "<"
            | "<="
            | ">"
            | ">="
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_symbol() {
        let ast = parse_smt_formula("x").unwrap();
        assert_eq!(ast, SmtAst::Symbol("x".to_string()));
    }

    #[test]
    fn test_parse_int() {
        let ast = parse_smt_formula("42").unwrap();
        assert_eq!(ast, SmtAst::Int(42));
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(parse_smt_formula("true").unwrap(), SmtAst::Bool(true));
        assert_eq!(parse_smt_formula("false").unwrap(), SmtAst::Bool(false));
    }

    #[test]
    fn test_parse_negation() {
        let ast = parse_smt_formula("(- 5)").unwrap();
        assert_eq!(ast, SmtAst::Neg(Box::new(SmtAst::Int(5))));
    }

    #[test]
    fn test_parse_app() {
        let ast = parse_smt_formula("(+ x 1)").unwrap();
        assert_eq!(
            ast,
            SmtAst::App(
                "+".to_string(),
                vec![SmtAst::Symbol("x".to_string()), SmtAst::Int(1)]
            )
        );
    }

    #[test]
    fn test_parse_comparison() {
        let ast = parse_smt_formula("(>= x 0)").unwrap();
        assert_eq!(
            ast,
            SmtAst::App(
                ">=".to_string(),
                vec![SmtAst::Symbol("x".to_string()), SmtAst::Int(0)]
            )
        );
    }

    #[test]
    fn test_parse_nested() {
        let ast = parse_smt_formula("(and (>= x 0) (< x 10))").unwrap();
        match ast {
            SmtAst::App(op, args) => {
                assert_eq!(op, "and");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected App"),
        }
    }

    #[test]
    fn test_parse_let() {
        let ast = parse_smt_formula("(let ((a 1)) (+ a 2))").unwrap();
        match ast {
            SmtAst::Let(bindings, body) => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].0, "a");
                assert_eq!(bindings[0].1, SmtAst::Int(1));
                match *body {
                    SmtAst::App(op, _) => assert_eq!(op, "+"),
                    _ => panic!("expected App in body"),
                }
            }
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_parse_forall() {
        let ast = parse_smt_formula("(forall ((x Int)) (>= x 0))").unwrap();
        match ast {
            SmtAst::Forall(vars, _body) => {
                assert_eq!(vars.len(), 1);
                assert_eq!(vars[0].0, "x");
                assert_eq!(vars[0].1, SmtSort::Int);
            }
            _ => panic!("expected Forall"),
        }
    }

    #[test]
    fn test_parse_exists() {
        let ast = parse_smt_formula("(exists ((y Bool)) y)").unwrap();
        match ast {
            SmtAst::Exists(vars, body) => {
                assert_eq!(vars.len(), 1);
                assert_eq!(vars[0].0, "y");
                assert_eq!(vars[0].1, SmtSort::Bool);
                assert_eq!(*body, SmtAst::Symbol("y".to_string()));
            }
            _ => panic!("expected Exists"),
        }
    }

    #[test]
    fn test_free_vars() {
        let ast = parse_smt_formula("(and (>= x 0) (< y 10))").unwrap();
        let vars = ast.free_vars();
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_free_vars_with_binding() {
        let ast = parse_smt_formula("(let ((a x)) (+ a y))").unwrap();
        let vars = ast.free_vars();
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_free_vars_quantifier_shadowing() {
        let ast = parse_smt_formula("(forall ((x Int)) (and (> x 0) (> y 0)))").unwrap();
        let vars = ast.free_vars();
        assert_eq!(vars, vec!["y"]);
    }

    #[test]
    fn test_substitution() {
        let ast = parse_smt_formula("(+ x 1)").unwrap();
        let result = ast.subst("x", &SmtAst::Int(5));
        assert_eq!(
            result,
            SmtAst::App("+".to_string(), vec![SmtAst::Int(5), SmtAst::Int(1)])
        );
    }

    #[test]
    fn test_substitution_respects_let_shadowing() {
        let ast = parse_smt_formula("(let ((x 1)) (+ x y))").unwrap();
        let result = ast.subst("x", &SmtAst::Int(5));
        // Outer replacement should not change bound x in let
        assert_eq!(format!("{}", result), "(let ((x 1)) (+ x y))");
    }

    #[test]
    fn test_display() {
        let ast = SmtAst::App(
            ">=".to_string(),
            vec![SmtAst::Symbol("x".to_string()), SmtAst::Int(0)],
        );
        assert_eq!(format!("{}", ast), "(>= x 0)");
    }

    #[test]
    fn test_binary_minus_and_negatives() {
        let ast = parse_smt_formula("(- x y)").unwrap();
        assert_eq!(
            ast,
            SmtAst::App(
                "-".to_string(),
                vec![SmtAst::Symbol("x".into()), SmtAst::Symbol("y".into())]
            )
        );

        let literal = parse_smt_formula("-5").unwrap();
        assert_eq!(literal, SmtAst::Int(-5));
    }

    #[test]
    fn test_bitvec_and_array_sorts() {
        let ast =
            parse_smt_formula("(forall ((bv (_ BitVec 8)) (arr (Array Int Int))) true)").unwrap();
        if let SmtAst::Forall(vars, _) = ast {
            assert_eq!(vars[0].1, SmtSort::BitVec(8));
            assert_eq!(
                vars[1].1,
                SmtSort::Array(Box::new(SmtSort::Int), Box::new(SmtSort::Int))
            );
        } else {
            panic!("expected Forall");
        }
    }

    #[test]
    fn test_is_simple_for_compound_nodes() {
        let simple = parse_smt_formula("(+ x 1)").unwrap();
        assert!(simple.is_simple());

        let quantified = parse_smt_formula("(forall ((x Int)) (> x 0))").unwrap();
        assert!(!quantified.is_simple());

        let let_ast = parse_smt_formula("(let ((a 1)) (+ a 2))").unwrap();
        assert!(!let_ast.is_simple());
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(matches!(parse_smt_formula(""), Err(ParseError::EmptyInput)));

        assert!(matches!(
            parse_smt_formula("(+ 1"),
            Err(ParseError::UnexpectedEof | ParseError::UnmatchedParen)
        ));

        assert!(matches!(
            parse_smt_formula("x extra"),
            Err(ParseError::TrailingInput(_))
        ));
    }

    // ========================================================================
    // Mutation coverage tests
    // ========================================================================

    #[test]
    fn test_smt_sort_display_all_variants() {
        // Mutation: delete match arms in SmtSort::fmt
        assert_eq!(format!("{}", SmtSort::Int), "Int");
        assert_eq!(format!("{}", SmtSort::Bool), "Bool");
        assert_eq!(format!("{}", SmtSort::Real), "Real");
        assert_eq!(format!("{}", SmtSort::BitVec(32)), "(_ BitVec 32)");
        assert_eq!(
            format!(
                "{}",
                SmtSort::Array(Box::new(SmtSort::Int), Box::new(SmtSort::Bool))
            ),
            "(Array Int Bool)"
        );
        assert_eq!(
            format!("{}", SmtSort::Unknown("Custom".to_string())),
            "Custom"
        );
    }

    #[test]
    fn test_smt_ast_display_all_variants() {
        // Mutation: delete match arms in SmtAst::fmt

        // Symbol
        assert_eq!(format!("{}", SmtAst::Symbol("x".to_string())), "x");

        // Int - positive
        assert_eq!(format!("{}", SmtAst::Int(42)), "42");

        // Int - negative (uses (- N) format)
        assert_eq!(format!("{}", SmtAst::Int(-5)), "(- 5)");

        // Bool
        assert_eq!(format!("{}", SmtAst::Bool(true)), "true");
        assert_eq!(format!("{}", SmtAst::Bool(false)), "false");

        // Neg
        assert_eq!(
            format!("{}", SmtAst::Neg(Box::new(SmtAst::Symbol("x".to_string())))),
            "(- x)"
        );

        // App
        assert_eq!(
            format!(
                "{}",
                SmtAst::App("+".to_string(), vec![SmtAst::Int(1), SmtAst::Int(2)])
            ),
            "(+ 1 2)"
        );

        // Let
        assert_eq!(
            format!(
                "{}",
                SmtAst::Let(
                    vec![("a".to_string(), SmtAst::Int(1))],
                    Box::new(SmtAst::Symbol("a".to_string()))
                )
            ),
            "(let ((a 1)) a)"
        );

        // Forall
        assert_eq!(
            format!(
                "{}",
                SmtAst::Forall(
                    vec![("x".to_string(), SmtSort::Int)],
                    Box::new(SmtAst::Bool(true))
                )
            ),
            "(forall ((x Int)) true)"
        );

        // Exists
        assert_eq!(
            format!(
                "{}",
                SmtAst::Exists(
                    vec![("y".to_string(), SmtSort::Bool)],
                    Box::new(SmtAst::Symbol("y".to_string()))
                )
            ),
            "(exists ((y Bool)) y)"
        );
    }

    #[test]
    fn test_ast_to_sort_all_branches() {
        // Mutation: delete match arms in ast_to_sort

        // Int
        let int_ast = SmtAst::Symbol("Int".to_string());
        assert_eq!(ast_to_sort(&int_ast), SmtSort::Int);

        // Bool
        let bool_ast = SmtAst::Symbol("Bool".to_string());
        assert_eq!(ast_to_sort(&bool_ast), SmtSort::Bool);

        // Real
        let real_ast = SmtAst::Symbol("Real".to_string());
        assert_eq!(ast_to_sort(&real_ast), SmtSort::Real);

        // Unknown symbol
        let custom_ast = SmtAst::Symbol("CustomType".to_string());
        assert_eq!(
            ast_to_sort(&custom_ast),
            SmtSort::Unknown("CustomType".to_string())
        );

        // BitVec: (_ BitVec 8)
        let bitvec_ast = SmtAst::App(
            "_".to_string(),
            vec![SmtAst::Symbol("BitVec".to_string()), SmtAst::Int(8)],
        );
        assert_eq!(ast_to_sort(&bitvec_ast), SmtSort::BitVec(8));

        // Array: (Array Int Bool)
        let array_ast = SmtAst::App(
            "Array".to_string(),
            vec![
                SmtAst::Symbol("Int".to_string()),
                SmtAst::Symbol("Bool".to_string()),
            ],
        );
        assert_eq!(
            ast_to_sort(&array_ast),
            SmtSort::Array(Box::new(SmtSort::Int), Box::new(SmtSort::Bool))
        );

        // Unknown App (not matching BitVec or Array patterns)
        let unknown_app = SmtAst::App("Unknown".to_string(), vec![SmtAst::Int(1)]);
        match ast_to_sort(&unknown_app) {
            SmtSort::Unknown(s) => assert!(s.contains("Unknown")),
            _ => panic!("Expected Unknown sort"),
        }

        // Non-Symbol/App AST should return Unknown
        let int_lit = SmtAst::Int(42);
        match ast_to_sort(&int_lit) {
            SmtSort::Unknown(_) => {}
            _ => panic!("Expected Unknown sort for Int literal"),
        }
    }

    #[test]
    fn test_subst_respects_forall_shadowing() {
        // Mutation: delete if is_shadowed check in Forall/Exists
        let ast = parse_smt_formula("(forall ((x Int)) (+ x y))").unwrap();
        let result = ast.subst("x", &SmtAst::Int(100));

        // x is shadowed by forall, so substitution should NOT replace x inside
        let result_str = format!("{}", result);
        assert!(result_str.contains("(forall ((x Int)) (+ x y))"));
    }

    #[test]
    fn test_subst_replaces_free_in_forall() {
        // y is NOT shadowed, so should be replaced
        let ast = parse_smt_formula("(forall ((x Int)) (+ x y))").unwrap();
        let result = ast.subst("y", &SmtAst::Int(100));

        let result_str = format!("{}", result);
        assert!(result_str.contains("100"));
    }

    #[test]
    fn test_subst_exists_shadowing() {
        // Mutation: delete if is_shadowed check in Exists branch
        let ast = parse_smt_formula("(exists ((x Int)) (+ x y))").unwrap();
        let result = ast.subst("x", &SmtAst::Int(100));

        // x is shadowed by exists, so should NOT be replaced inside
        let result_str = format!("{}", result);
        assert!(result_str.contains("(exists ((x Int)) (+ x y))"));
    }

    #[test]
    fn test_subst_in_neg() {
        // Mutation: delete match arm for Neg
        let ast = SmtAst::Neg(Box::new(SmtAst::Symbol("x".to_string())));
        let result = ast.subst("x", &SmtAst::Int(5));
        assert_eq!(result, SmtAst::Neg(Box::new(SmtAst::Int(5))));
    }

    #[test]
    fn test_is_simple_neg() {
        // Mutation: delete match arm for Neg in is_simple
        let neg = SmtAst::Neg(Box::new(SmtAst::Int(5)));
        assert!(neg.is_simple());

        let neg_with_let = SmtAst::Neg(Box::new(SmtAst::Let(
            vec![("a".to_string(), SmtAst::Int(1))],
            Box::new(SmtAst::Symbol("a".to_string())),
        )));
        assert!(!neg_with_let.is_simple());
    }

    #[test]
    fn test_collect_free_vars_in_neg() {
        // Mutation: delete match arm for Neg in collect_free_vars
        let neg = SmtAst::Neg(Box::new(SmtAst::Symbol("x".to_string())));
        let vars = neg.free_vars();
        assert_eq!(vars, vec!["x"]);
    }

    #[test]
    fn test_parse_trailing_whitespace() {
        // Mutation: delete rest.trim() call in parse_smt_formula
        let ast = parse_smt_formula("   x   ").unwrap();
        assert_eq!(ast, SmtAst::Symbol("x".to_string()));
    }

    #[test]
    fn test_parse_unexpected_close_paren() {
        // Mutation: delete check for first == ')' in parse_sexp
        let err = parse_smt_formula(")").unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedChar(')')));
    }

    #[test]
    fn test_parse_multiple_args_in_minus() {
        // Mutation: delete while loop for collecting args after first in binary minus
        let ast = parse_smt_formula("(- a b c)").unwrap();
        if let SmtAst::App(op, args) = ast {
            assert_eq!(op, "-");
            assert_eq!(args.len(), 3); // Should have all three args
        } else {
            panic!("Expected App");
        }
    }

    #[test]
    fn test_subst_let_binding_values() {
        // Mutation: check that substitution happens in let binding values
        let ast = parse_smt_formula("(let ((a x)) (+ a 1))").unwrap();
        let result = ast.subst("x", &SmtAst::Int(42));

        // x should be replaced in the binding value
        match result {
            SmtAst::Let(bindings, _) => {
                assert_eq!(bindings[0].1, SmtAst::Int(42));
            }
            _ => panic!("Expected Let"),
        }
    }
}
