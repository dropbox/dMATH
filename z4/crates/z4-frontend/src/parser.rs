//! SMT-LIB parser
//!
//! Parses SMT-LIB 2.6 input into commands.

use crate::command::Command;
use crate::sexp::{ParseError, SExprParser};

/// SMT-LIB parser
pub struct Parser<'a> {
    sexp_parser: SExprParser<'a>,
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given input
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        Parser {
            sexp_parser: SExprParser::new(input),
        }
    }

    /// Parse the next command from the input
    ///
    /// Returns `None` when the input is exhausted.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is malformed SMT-LIB.
    pub fn parse_command(&mut self) -> Result<Option<Command>, ParseError> {
        if self.sexp_parser.is_eof() {
            return Ok(None);
        }

        let sexp = self.sexp_parser.parse_sexp()?;
        let cmd = Command::from_sexp(&sexp)?;
        Ok(Some(cmd))
    }

    /// Parse all commands from the input
    ///
    /// # Errors
    ///
    /// Returns an error if the input is malformed SMT-LIB.
    pub fn parse_all(&mut self) -> Result<Vec<Command>, ParseError> {
        let mut commands = Vec::new();
        while let Some(cmd) = self.parse_command()? {
            commands.push(cmd);
        }
        Ok(commands)
    }

    /// Check if the parser has reached end of input
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.sexp_parser.is_eof()
    }
}

/// Parse SMT-LIB input into a list of commands
///
/// # Errors
///
/// Returns an error if the input is malformed SMT-LIB.
pub fn parse(input: &str) -> Result<Vec<Command>, ParseError> {
    let mut parser = Parser::new(input);
    parser.parse_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{Constant, Sort, Term};

    #[test]
    fn test_parse_simple_problem() {
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Int)
            (assert (> x 0))
            (assert (< y 10))
            (check-sat)
            (exit)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 7);
        assert!(matches!(commands[0], Command::SetLogic(_)));
        assert!(matches!(commands[1], Command::DeclareConst(_, _)));
        assert!(matches!(commands[2], Command::DeclareConst(_, _)));
        assert!(matches!(commands[3], Command::Assert(_)));
        assert!(matches!(commands[4], Command::Assert(_)));
        assert!(matches!(commands[5], Command::CheckSat));
        assert!(matches!(commands[6], Command::Exit));
    }

    #[test]
    fn test_parse_bitvector_problem() {
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 32))
            (declare-const y (_ BitVec 32))
            (assert (= (bvadd x y) #x00000001))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 5);

        // Check bitvector sort
        if let Command::DeclareConst(name, Sort::Indexed(sort, indices)) = &commands[1] {
            assert_eq!(name, "x");
            assert_eq!(sort, "BitVec");
            assert_eq!(indices, &vec!["32".to_string()]);
        } else {
            panic!("Expected DeclareConst with indexed sort");
        }
    }

    #[test]
    fn test_parse_array_problem() {
        let input = r#"
            (set-logic QF_AUFLIA)
            (declare-const arr (Array Int Int))
            (declare-const i Int)
            (assert (= (select arr i) 42))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 5);

        // Check array sort
        if let Command::DeclareConst(name, Sort::Parameterized(sort, params)) = &commands[1] {
            assert_eq!(name, "arr");
            assert_eq!(sort, "Array");
            assert_eq!(params.len(), 2);
        } else {
            panic!("Expected DeclareConst with parameterized sort");
        }
    }

    #[test]
    fn test_parse_with_let() {
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (let ((y (+ x 1))) (> y 0)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 4);

        if let Command::Assert(Term::Let(bindings, body)) = &commands[2] {
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].0, "y");
            assert!(matches!(**body, Term::App(_, _)));
        } else {
            panic!("Expected Assert with Let term");
        }
    }

    #[test]
    fn test_parse_with_quantifier() {
        let input = r#"
            (set-logic AUFLIA)
            (assert (forall ((x Int) (y Int)) (=> (> x y) (> (+ x 1) y))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 3);

        if let Command::Assert(Term::Forall(bindings, _body)) = &commands[1] {
            assert_eq!(bindings.len(), 2);
            assert_eq!(bindings[0].0, "x");
            assert_eq!(bindings[1].0, "y");
        } else {
            panic!("Expected Assert with Forall term");
        }
    }

    #[test]
    fn test_parse_define_fun() {
        let input = r#"
            (set-logic QF_LIA)
            (define-fun abs ((x Int)) Int (ite (< x 0) (- x) x))
            (declare-const a Int)
            (assert (= (abs a) 5))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 5);

        if let Command::DefineFun(name, params, ret_sort, body) = &commands[1] {
            assert_eq!(name, "abs");
            assert_eq!(params.len(), 1);
            assert_eq!(params[0].0, "x");
            assert!(matches!(ret_sort, Sort::Simple(s) if s == "Int"));
            assert!(matches!(body, Term::App(f, _) if f == "ite"));
        } else {
            panic!("Expected DefineFun command");
        }
    }

    #[test]
    fn test_parse_push_pop() {
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (push 1)
            (assert (> x 0))
            (check-sat)
            (pop 1)
            (assert (< x 0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 8);
        assert!(matches!(commands[2], Command::Push(1)));
        assert!(matches!(commands[5], Command::Pop(1)));
    }

    #[test]
    fn test_parse_constants() {
        let input = r#"
            (assert (and true false))
            (assert (= 42 42))
            (assert (= 3.14 3.14))
            (assert (= #xDEAD #xDEAD))
            (assert (= #b1010 #b1010))
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 5);

        // Check boolean constants
        if let Command::Assert(Term::App(_, args)) = &commands[0] {
            assert!(matches!(&args[0], Term::Const(Constant::True)));
            assert!(matches!(&args[1], Term::Const(Constant::False)));
        }
    }

    #[test]
    fn test_parse_comments() {
        let input = r#"
            ; This is a comment
            (set-logic QF_LIA) ; inline comment
            ; Another comment
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        assert_eq!(commands.len(), 2);
    }

    #[test]
    fn test_parse_empty_input() {
        let commands = parse("").unwrap();
        assert!(commands.is_empty());
    }

    #[test]
    fn test_parse_whitespace_only() {
        let commands = parse("   \n\t\n   ").unwrap();
        assert!(commands.is_empty());
    }

    #[test]
    fn test_parse_error_missing_paren() {
        let result = parse("(set-logic QF_LIA");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unknown_command() {
        let result = parse("(unknown-command foo)");
        assert!(result.is_err());
    }
}
