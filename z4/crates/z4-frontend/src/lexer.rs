//! SMT-LIB lexer
//!
//! Tokenizes SMT-LIB 2.6 input using the logos crate for high performance.

use logos::Logos;

/// SMT-LIB tokens
#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\r]+")]
#[logos(skip r";[^\n]*")]
pub enum Token<'a> {
    /// Left parenthesis
    #[token("(")]
    LParen,

    /// Right parenthesis
    #[token(")")]
    RParen,

    /// Numeral (non-negative integer)
    #[regex(r"[0-9]+", |lex| lex.slice())]
    Numeral(&'a str),

    /// Decimal number
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice())]
    Decimal(&'a str),

    /// Hexadecimal bitvector literal (#xABCD)
    #[regex(r"#x[0-9a-fA-F]+", |lex| lex.slice())]
    Hexadecimal(&'a str),

    /// Binary bitvector literal (#b0101)
    #[regex(r"#b[01]+", |lex| lex.slice())]
    Binary(&'a str),

    /// String literal
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice())]
    String(&'a str),

    /// Symbol (identifier)
    #[regex(r"[a-zA-Z~!@$%^&*_+=<>.?/\-][a-zA-Z0-9~!@$%^&*_+=<>.?/\-]*", |lex| lex.slice())]
    Symbol(&'a str),

    /// Quoted symbol |...|
    #[regex(r"\|[^|]*\|", |lex| lex.slice())]
    QuotedSymbol(&'a str),

    /// Keyword (:keyword)
    #[regex(r":[a-zA-Z0-9~!@$%^&*_+=<>.?/\-]+", |lex| lex.slice())]
    Keyword(&'a str),

    /// Reserved words: true/false
    #[token("true")]
    True,

    /// Boolean false
    #[token("false")]
    False,
    // Note: Indexed identifiers (_ symbol numeral+) are handled by the parser, not lexer
}

/// Lexer error type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexerError {
    /// Position in input where error occurred
    pub position: usize,
    /// Error message
    pub message: String,
}

impl std::fmt::Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Lexer error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for LexerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let input = "(check-sat)";
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::LParen)));
        assert_eq!(lexer.next(), Some(Ok(Token::Symbol("check-sat"))));
        assert_eq!(lexer.next(), Some(Ok(Token::RParen)));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_numerals() {
        let input = "42 0 12345";
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::Numeral("42"))));
        assert_eq!(lexer.next(), Some(Ok(Token::Numeral("0"))));
        assert_eq!(lexer.next(), Some(Ok(Token::Numeral("12345"))));
    }

    #[test]
    fn test_bitvectors() {
        let input = "#xDEADBEEF #b10101010";
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::Hexadecimal("#xDEADBEEF"))));
        assert_eq!(lexer.next(), Some(Ok(Token::Binary("#b10101010"))));
    }

    #[test]
    fn test_strings() {
        let input = r#""hello" "world""#;
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::String("\"hello\""))));
        assert_eq!(lexer.next(), Some(Ok(Token::String("\"world\""))));
    }

    #[test]
    fn test_keywords() {
        let input = ":named :status";
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::Keyword(":named"))));
        assert_eq!(lexer.next(), Some(Ok(Token::Keyword(":status"))));
    }

    #[test]
    fn test_booleans() {
        let input = "true false";
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::True)));
        assert_eq!(lexer.next(), Some(Ok(Token::False)));
    }

    #[test]
    fn test_comments() {
        let input = "(check-sat) ; this is a comment\n(exit)";
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::LParen)));
        assert_eq!(lexer.next(), Some(Ok(Token::Symbol("check-sat"))));
        assert_eq!(lexer.next(), Some(Ok(Token::RParen)));
        assert_eq!(lexer.next(), Some(Ok(Token::LParen)));
        assert_eq!(lexer.next(), Some(Ok(Token::Symbol("exit"))));
        assert_eq!(lexer.next(), Some(Ok(Token::RParen)));
    }

    #[test]
    fn test_quoted_symbol() {
        let input = "|quoted symbol with spaces|";
        let mut lexer = Token::lexer(input);

        assert_eq!(
            lexer.next(),
            Some(Ok(Token::QuotedSymbol("|quoted symbol with spaces|")))
        );
    }

    #[test]
    fn test_declare_fun() {
        let input = "(declare-fun x () Int)";
        let mut lexer = Token::lexer(input);

        assert_eq!(lexer.next(), Some(Ok(Token::LParen)));
        assert_eq!(lexer.next(), Some(Ok(Token::Symbol("declare-fun"))));
        assert_eq!(lexer.next(), Some(Ok(Token::Symbol("x"))));
        assert_eq!(lexer.next(), Some(Ok(Token::LParen)));
        assert_eq!(lexer.next(), Some(Ok(Token::RParen)));
        assert_eq!(lexer.next(), Some(Ok(Token::Symbol("Int"))));
        assert_eq!(lexer.next(), Some(Ok(Token::RParen)));
    }
}
