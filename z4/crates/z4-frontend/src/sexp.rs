//! S-expression representation and parsing
//!
//! SMT-LIB syntax is based on S-expressions.

use crate::lexer::Token;
use logos::Logos;
use std::fmt;

/// An S-expression
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    /// A symbol (identifier)
    Symbol(String),
    /// A keyword (:name)
    Keyword(String),
    /// A numeral
    Numeral(String),
    /// A decimal number
    Decimal(String),
    /// A hexadecimal bitvector
    Hexadecimal(String),
    /// A binary bitvector
    Binary(String),
    /// A string literal
    String(String),
    /// Boolean true
    True,
    /// Boolean false
    False,
    /// A list of S-expressions
    List(Vec<SExpr>),
}

impl fmt::Display for SExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SExpr::Symbol(s) => write!(f, "{s}"),
            SExpr::Keyword(k) => write!(f, "{k}"),
            SExpr::Numeral(n) => write!(f, "{n}"),
            SExpr::Decimal(d) => write!(f, "{d}"),
            SExpr::Hexadecimal(h) => write!(f, "{h}"),
            SExpr::Binary(b) => write!(f, "{b}"),
            SExpr::String(s) => write!(f, "{s}"),
            SExpr::True => write!(f, "true"),
            SExpr::False => write!(f, "false"),
            SExpr::List(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl SExpr {
    /// Check if this is a symbol with the given name
    #[must_use]
    pub fn is_symbol(&self, name: &str) -> bool {
        matches!(self, SExpr::Symbol(s) if s == name)
    }

    /// Get the symbol name if this is a symbol
    #[must_use]
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            SExpr::Symbol(s) => Some(s),
            _ => None,
        }
    }

    /// Get the list contents if this is a list
    #[must_use]
    pub fn as_list(&self) -> Option<&[SExpr]> {
        match self {
            SExpr::List(items) => Some(items),
            _ => None,
        }
    }

    /// Get the numeral value if this is a numeral
    #[must_use]
    pub fn as_numeral(&self) -> Option<&str> {
        match self {
            SExpr::Numeral(n) => Some(n),
            _ => None,
        }
    }
}

/// Parse error
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Position in input (if available)
    pub position: Option<usize>,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(pos) = self.position {
            write!(f, "Parse error at position {}: {}", pos, self.message)
        } else {
            write!(f, "Parse error: {}", self.message)
        }
    }
}

impl std::error::Error for ParseError {}

impl ParseError {
    /// Create a new parse error
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        ParseError {
            message: message.into(),
            position: None,
        }
    }

    /// Create a new parse error with position
    #[must_use]
    pub fn with_position(message: impl Into<String>, position: usize) -> Self {
        ParseError {
            message: message.into(),
            position: Some(position),
        }
    }
}

/// S-expression parser
pub struct SExprParser<'a> {
    lexer: logos::Lexer<'a, Token<'a>>,
    current: Option<Result<Token<'a>, ()>>,
}

impl<'a> SExprParser<'a> {
    /// Create a new parser for the given input
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Token::lexer(input);
        let current = lexer.next();
        SExprParser { lexer, current }
    }

    /// Parse a single S-expression
    pub fn parse_sexp(&mut self) -> Result<SExpr, ParseError> {
        match &self.current {
            None => Err(ParseError::new("Unexpected end of input")),
            Some(Err(())) => Err(ParseError::with_position(
                "Invalid token",
                self.lexer.span().start,
            )),
            Some(Ok(token)) => match token {
                Token::LParen => self.parse_list(),
                Token::RParen => Err(ParseError::with_position(
                    "Unexpected ')'",
                    self.lexer.span().start,
                )),
                Token::Symbol(s) => {
                    let sym = SExpr::Symbol((*s).to_string());
                    self.advance();
                    Ok(sym)
                }
                Token::Keyword(k) => {
                    let kw = SExpr::Keyword((*k).to_string());
                    self.advance();
                    Ok(kw)
                }
                Token::Numeral(n) => {
                    let num = SExpr::Numeral((*n).to_string());
                    self.advance();
                    Ok(num)
                }
                Token::Decimal(d) => {
                    let dec = SExpr::Decimal((*d).to_string());
                    self.advance();
                    Ok(dec)
                }
                Token::Hexadecimal(h) => {
                    let hex = SExpr::Hexadecimal((*h).to_string());
                    self.advance();
                    Ok(hex)
                }
                Token::Binary(b) => {
                    let bin = SExpr::Binary((*b).to_string());
                    self.advance();
                    Ok(bin)
                }
                Token::String(s) => {
                    let st = SExpr::String((*s).to_string());
                    self.advance();
                    Ok(st)
                }
                Token::QuotedSymbol(s) => {
                    // Remove the | delimiters
                    let inner = &s[1..s.len() - 1];
                    let sym = SExpr::Symbol(inner.to_string());
                    self.advance();
                    Ok(sym)
                }
                Token::True => {
                    self.advance();
                    Ok(SExpr::True)
                }
                Token::False => {
                    self.advance();
                    Ok(SExpr::False)
                }
            },
        }
    }

    /// Parse a list (assumes current token is LParen)
    fn parse_list(&mut self) -> Result<SExpr, ParseError> {
        self.advance(); // consume '('
        let mut items = Vec::new();

        loop {
            match &self.current {
                None => return Err(ParseError::new("Unexpected end of input in list")),
                Some(Err(())) => {
                    return Err(ParseError::with_position(
                        "Invalid token in list",
                        self.lexer.span().start,
                    ))
                }
                Some(Ok(Token::RParen)) => {
                    self.advance(); // consume ')'
                    return Ok(SExpr::List(items));
                }
                Some(Ok(_)) => {
                    items.push(self.parse_sexp()?);
                }
            }
        }
    }

    /// Advance to the next token
    fn advance(&mut self) {
        self.current = self.lexer.next();
    }

    /// Check if there are more tokens
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.current.is_none()
    }

    /// Parse all S-expressions from the input
    pub fn parse_all(&mut self) -> Result<Vec<SExpr>, ParseError> {
        let mut result = Vec::new();
        while !self.is_eof() {
            result.push(self.parse_sexp()?);
        }
        Ok(result)
    }
}

/// Parse a string into a single S-expression
pub fn parse_sexp(input: &str) -> Result<SExpr, ParseError> {
    let mut parser = SExprParser::new(input);
    parser.parse_sexp()
}

/// Parse a string into multiple S-expressions
pub fn parse_sexps(input: &str) -> Result<Vec<SExpr>, ParseError> {
    let mut parser = SExprParser::new(input);
    parser.parse_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_symbol() {
        let sexp = parse_sexp("foo").unwrap();
        assert_eq!(sexp, SExpr::Symbol("foo".to_string()));
    }

    #[test]
    fn test_parse_numeral() {
        let sexp = parse_sexp("42").unwrap();
        assert_eq!(sexp, SExpr::Numeral("42".to_string()));
    }

    #[test]
    fn test_parse_empty_list() {
        let sexp = parse_sexp("()").unwrap();
        assert_eq!(sexp, SExpr::List(vec![]));
    }

    #[test]
    fn test_parse_simple_list() {
        let sexp = parse_sexp("(a b c)").unwrap();
        assert_eq!(
            sexp,
            SExpr::List(vec![
                SExpr::Symbol("a".to_string()),
                SExpr::Symbol("b".to_string()),
                SExpr::Symbol("c".to_string()),
            ])
        );
    }

    #[test]
    fn test_parse_nested_list() {
        let sexp = parse_sexp("(a (b c) d)").unwrap();
        assert_eq!(
            sexp,
            SExpr::List(vec![
                SExpr::Symbol("a".to_string()),
                SExpr::List(vec![
                    SExpr::Symbol("b".to_string()),
                    SExpr::Symbol("c".to_string()),
                ]),
                SExpr::Symbol("d".to_string()),
            ])
        );
    }

    #[test]
    fn test_parse_check_sat() {
        let sexp = parse_sexp("(check-sat)").unwrap();
        assert_eq!(
            sexp,
            SExpr::List(vec![SExpr::Symbol("check-sat".to_string())])
        );
    }

    #[test]
    fn test_parse_declare_fun() {
        let sexp = parse_sexp("(declare-fun x () Int)").unwrap();
        assert_eq!(
            sexp,
            SExpr::List(vec![
                SExpr::Symbol("declare-fun".to_string()),
                SExpr::Symbol("x".to_string()),
                SExpr::List(vec![]),
                SExpr::Symbol("Int".to_string()),
            ])
        );
    }

    #[test]
    fn test_parse_assert() {
        let sexp = parse_sexp("(assert (> x 0))").unwrap();
        assert_eq!(
            sexp,
            SExpr::List(vec![
                SExpr::Symbol("assert".to_string()),
                SExpr::List(vec![
                    SExpr::Symbol(">".to_string()),
                    SExpr::Symbol("x".to_string()),
                    SExpr::Numeral("0".to_string()),
                ]),
            ])
        );
    }

    #[test]
    fn test_parse_bitvector() {
        let sexp = parse_sexp("#xDEAD").unwrap();
        assert_eq!(sexp, SExpr::Hexadecimal("#xDEAD".to_string()));

        let sexp = parse_sexp("#b1010").unwrap();
        assert_eq!(sexp, SExpr::Binary("#b1010".to_string()));
    }

    #[test]
    fn test_parse_keyword() {
        let sexp = parse_sexp(":named").unwrap();
        assert_eq!(sexp, SExpr::Keyword(":named".to_string()));
    }

    #[test]
    fn test_parse_multiple() {
        let sexps = parse_sexps("(set-logic QF_LIA) (check-sat)").unwrap();
        assert_eq!(sexps.len(), 2);
        assert_eq!(
            sexps[0],
            SExpr::List(vec![
                SExpr::Symbol("set-logic".to_string()),
                SExpr::Symbol("QF_LIA".to_string()),
            ])
        );
        assert_eq!(
            sexps[1],
            SExpr::List(vec![SExpr::Symbol("check-sat".to_string())])
        );
    }

    #[test]
    fn test_parse_booleans() {
        let sexp = parse_sexp("(and true false)").unwrap();
        assert_eq!(
            sexp,
            SExpr::List(vec![
                SExpr::Symbol("and".to_string()),
                SExpr::True,
                SExpr::False,
            ])
        );
    }

    #[test]
    fn test_parse_quoted_symbol() {
        let sexp = parse_sexp("|quoted symbol|").unwrap();
        assert_eq!(sexp, SExpr::Symbol("quoted symbol".to_string()));
    }

    #[test]
    fn test_error_unmatched_paren() {
        let result = parse_sexp("(a b");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unexpected_rparen() {
        let result = parse_sexp(")");
        assert!(result.is_err());
    }
}
