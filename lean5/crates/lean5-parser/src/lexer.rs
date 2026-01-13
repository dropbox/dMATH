//! Lexer for Lean 4 syntax
//!
//! Tokenizes Lean source text into a stream of tokens.

use crate::surface::Span;
use std::iter::Peekable;
use std::str::CharIndices;

/// Token type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Keywords
    Def,
    Theorem,
    Lemma,
    Axiom,
    Example,
    Let,
    In,
    Fun,
    Forall,
    If,
    Then,
    Else,
    Match,
    With,
    Where,
    Do,
    Return,
    Structure,
    Class,
    Instance,
    Inductive,
    Deriving,
    Namespace,
    Section,
    End,
    Open,
    Variable,
    Universe,
    Import,
    Mutual,
    SetOption,
    By,
    Have,
    Show,
    Suffices,
    From,
    Rfl,
    Sorry,
    Extends,
    Private,
    Protected,
    Partial,
    Unsafe,
    Noncomputable,
    Abbrev,
    Attribute,
    Syntax,     // syntax command for custom syntax
    Macro,      // macro command
    MacroRules, // macro_rules command (multi-arm macros)
    Elab,       // elab command for custom elaborators
    Infixl,     // left-associative infix notation
    Infixr,     // right-associative infix notation
    Prefix,     // prefix notation
    Postfix,    // postfix notation
    Notation,   // general notation command
    Scoped,     // scoped modifier
    Rec,        // rec keyword (for let rec)

    // Types
    Type,
    Prop,
    Sort,

    // Identifiers and literals
    Ident(String),
    NatLit(u64),
    StringLit(String),
    /// Raw syntax quotation: `(foo $bar)` captured after a backtick
    SyntaxQuote(String),

    // Delimiters
    LParen,       // (
    RParen,       // )
    LBrace,       // {
    RBrace,       // }
    LBracket,     // [
    RBracket,     // ]
    LAngle,       // ⟨ (Unicode angle bracket for anonymous constructors)
    RAngle,       // ⟩ (Unicode angle bracket for anonymous constructors)
    BackwardPipe, // <| (backward/reverse pipe operator - low-precedence application)

    // Punctuation
    Colon,      // :
    ColonColon, // :: (cons operator)
    ColonEq,    // :=
    Comma,      // ,
    Dot,        // .
    Semicolon,  // ;
    Arrow,      // → or ->
    FatArrow,   // =>
    Lambda,     // λ or fun
    Pi,         // Π or forall
    At,         // @
    Hash,       // #
    Underscore, // _
    Pipe,       // |
    Amp,        // &
    Star,       // *
    Plus,       // +
    Minus,      // -
    Slash,      // /
    Caret,      // ^ (exponentiation)
    Eq,         // =
    DoubleEq,   // == (BEq equality check)
    Ne,         // ≠ or !=
    Lt,         // <
    Le,         // ≤ or <=
    Gt,         // >
    Ge,         // ≥ or >=
    And,        // ∧ or /\
    Or,         // ∨ or \/
    Not,        // ¬ or !
    Tilde,      // ~ (user-defined operators)
    Percent,    // % (modulo operator)

    // Additional Unicode operators
    HEq,             // ≍ (heterogeneous equality)
    Equiv,           // ≃ (equivalence/isomorphism)
    Iff,             // ↔
    Times,           // ×
    LeftArrow,       // ←
    Exists,          // ∃
    Elem,            // ∈
    NotElem,         // ∉
    Subset,          // ⊆
    ProperSubset,    // ⊂
    Inter,           // ∩
    Union,           // ∪
    EmptySet,        // ∅
    Top,             // ⊤
    Bot,             // ⊥
    Compose,         // ∘
    Cdot,            // · (section placeholder / middle dot)
    Dollar,          // $ (low-precedence application)
    DollarArrow,     // $>
    LeftDollar,      // <$
    LeftDollarArrow, // <$>
    Bind,            // >>=
    Seq,             // >>
    AndThen,         // *>
    OrElse,          // <|>

    // Special
    Eof,
    Error(String),
}

/// A token with its span
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    #[must_use]
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    #[must_use]
    pub fn eof(pos: usize) -> Self {
        Self {
            kind: TokenKind::Eof,
            span: Span::new(pos, pos),
        }
    }
}

/// Lexer state
pub struct Lexer<'a> {
    input: &'a str,
    chars: Peekable<CharIndices<'a>>,
    pos: usize,
}

impl<'a> Lexer<'a> {
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            pos: 0,
        }
    }

    /// Tokenize all input
    #[must_use]
    pub fn tokenize(input: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let tok = lexer.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        tokens
    }

    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, c)| *c)
    }

    fn next_char(&mut self) -> Option<(usize, char)> {
        let result = self.chars.next();
        if let Some((i, c)) = result {
            self.pos = i + c.len_utf8();
        }
        result
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.next_char();
            } else if c == '-' {
                // Check for line comment --
                let start = self.pos;
                let saved_chars = self.chars.clone();
                self.next_char();
                if self.peek_char() == Some('-') {
                    // Line comment, skip to end of line
                    while let Some(c) = self.peek_char() {
                        self.next_char();
                        if c == '\n' {
                            break;
                        }
                    }
                } else {
                    // Not a comment, restore
                    self.chars = saved_chars;
                    self.pos = start;
                    break;
                }
            } else if c == '/' {
                // Check for block comment /-
                let start = self.pos;
                let saved_chars = self.chars.clone();
                self.next_char();
                if self.peek_char() == Some('-') {
                    self.next_char();
                    // Block comment, skip to -/
                    let mut depth = 1;
                    while depth > 0 {
                        match self.next_char() {
                            Some((_, '/')) if self.peek_char() == Some('-') => {
                                self.next_char();
                                depth += 1;
                            }
                            Some((_, '-')) if self.peek_char() == Some('/') => {
                                self.next_char();
                                depth -= 1;
                            }
                            None => break,
                            _ => {}
                        }
                    }
                } else {
                    // Not a comment, restore
                    self.chars = saved_chars;
                    self.pos = start;
                    break;
                }
            } else {
                break;
            }
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let start = self.pos;

        let Some((_, c)) = self.next_char() else {
            return Token::eof(start);
        };

        let kind = match c {
            // Single-character tokens
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            '@' => TokenKind::At,
            '#' => TokenKind::Hash,
            '`' => self.lex_backtick(start),
            '_' => {
                // Could be underscore or start of identifier
                if self
                    .peek_char()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
                {
                    self.lex_ident(start, c)
                } else {
                    TokenKind::Underscore
                }
            }
            '|' => {
                if self.peek_char() == Some('>') {
                    self.next_char();
                    TokenKind::RAngle // |> as angle bracket alternative
                } else {
                    TokenKind::Pipe
                }
            }
            '&' => TokenKind::Amp,
            '*' => {
                if self.peek_char() == Some('>') {
                    self.next_char();
                    TokenKind::AndThen
                } else {
                    TokenKind::Star
                }
            }
            '+' => TokenKind::Plus,
            '^' => TokenKind::Caret,
            '$' => {
                if self.peek_char() == Some('>') {
                    self.next_char();
                    TokenKind::DollarArrow
                } else {
                    TokenKind::Dollar
                }
            }

            // Multi-character tokens
            ':' => {
                if self.peek_char() == Some('=') {
                    self.next_char();
                    TokenKind::ColonEq
                } else if self.peek_char() == Some(':') {
                    self.next_char();
                    TokenKind::ColonColon
                } else {
                    TokenKind::Colon
                }
            }
            '.' => TokenKind::Dot,
            '=' => {
                if self.peek_char() == Some('>') {
                    self.next_char();
                    TokenKind::FatArrow
                } else if self.peek_char() == Some('=') {
                    self.next_char();
                    TokenKind::DoubleEq
                } else {
                    TokenKind::Eq
                }
            }
            '-' => {
                if self.peek_char() == Some('>') {
                    self.next_char();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '/' => {
                if self.peek_char() == Some('\\') {
                    self.next_char();
                    TokenKind::And
                } else {
                    TokenKind::Slash
                }
            }
            '\\' => {
                if self.peek_char() == Some('/') {
                    self.next_char();
                    TokenKind::Or
                } else {
                    // Lambda shorthand
                    TokenKind::Lambda
                }
            }
            '<' => {
                match self.peek_char() {
                    Some('=') => {
                        self.next_char();
                        TokenKind::Le
                    }
                    Some('$') => {
                        self.next_char();
                        if self.peek_char() == Some('>') {
                            self.next_char();
                            TokenKind::LeftDollarArrow // <$>
                        } else {
                            TokenKind::LeftDollar // <$
                        }
                    }
                    Some('|') => {
                        self.next_char();
                        if self.peek_char() == Some('>') {
                            self.next_char();
                            TokenKind::OrElse // <|>
                        } else {
                            TokenKind::BackwardPipe // <| (backward pipe operator)
                        }
                    }
                    Some(next) if is_angle_operator_char(next) => {
                        // Custom operator composed of angle-like symbols, e.g., <><
                        let mut op = String::from("<");
                        while let Some(c) = self.peek_char() {
                            if is_angle_operator_char(c) {
                                self.next_char();
                                op.push(c);
                            } else {
                                break;
                            }
                        }
                        TokenKind::Ident(op)
                    }
                    _ => TokenKind::Lt,
                }
            }
            '>' => {
                match self.peek_char() {
                    Some('=') => {
                        self.next_char();
                        TokenKind::Ge
                    }
                    Some('>') => {
                        self.next_char();
                        if self.peek_char() == Some('=') {
                            self.next_char();
                            TokenKind::Bind // >>=
                        } else {
                            TokenKind::Seq // >>
                        }
                    }
                    Some(next) if is_angle_operator_char(next) => {
                        let mut op = String::from(">");
                        while let Some(c) = self.peek_char() {
                            if is_angle_operator_char(c) {
                                self.next_char();
                                op.push(c);
                            } else {
                                break;
                            }
                        }
                        TokenKind::Ident(op)
                    }
                    _ => TokenKind::Gt,
                }
            }
            '!' => {
                if self.peek_char() == Some('=') {
                    self.next_char();
                    TokenKind::Ne
                } else {
                    TokenKind::Not
                }
            }
            '~' => {
                // Check for ~> (custom arrow operator)
                if self.peek_char() == Some('>') {
                    self.next_char();
                    TokenKind::Ident("~>".to_string())
                } else {
                    TokenKind::Tilde
                }
            }
            '%' => TokenKind::Percent,

            // Unicode
            '→' => TokenKind::Arrow,
            'λ' => TokenKind::Lambda,
            '∀' => TokenKind::Forall,
            'Π' => TokenKind::Pi,
            '∧' => TokenKind::And,
            '∨' => TokenKind::Or,
            '¬' => TokenKind::Not,
            '≤' => TokenKind::Le,
            '≥' => TokenKind::Ge,
            '≠' => TokenKind::Ne,
            '≍' => TokenKind::HEq,   // Heterogeneous equality
            '≃' => TokenKind::Equiv, // Equivalence/isomorphism
            '⟨' => TokenKind::LAngle,
            '⟩' => TokenKind::RAngle,
            '↔' => TokenKind::Iff,
            '×' => TokenKind::Times,
            '←' => TokenKind::LeftArrow,
            '∃' => TokenKind::Exists,
            '∈' => TokenKind::Elem,
            '∉' => TokenKind::NotElem,
            '⊆' => TokenKind::Subset,
            '⊂' => TokenKind::ProperSubset,
            '∩' => TokenKind::Inter,
            '∪' => TokenKind::Union,
            '∅' => TokenKind::EmptySet,
            '⊤' => TokenKind::Top,
            '⊥' => TokenKind::Bot,
            '∘' => {
                // Check if followed by prime (') - user-defined operator like ∘'
                if self.peek_char() == Some('\'') {
                    self.next_char();
                    TokenKind::Ident("∘'".to_string())
                } else {
                    TokenKind::Compose
                }
            }
            '·' | '•' => TokenKind::Cdot, // Middle dot or bullet - section placeholder
            // Blackboard bold letters -> identifiers
            'ℕ' => TokenKind::Ident("Nat".to_string()),
            'ℤ' => TokenKind::Ident("Int".to_string()),
            'ℚ' => TokenKind::Ident("Rat".to_string()),
            'ℝ' => TokenKind::Ident("Real".to_string()),
            'ℂ' => TokenKind::Ident("Complex".to_string()),

            // String literals
            '"' => self.lex_string(start),

            // Number literals
            c if c.is_ascii_digit() => self.lex_number(start, c),

            // Identifiers and keywords
            c if is_ident_start(c) => self.lex_ident(start, c),

            _ => TokenKind::Error(format!("unexpected character: {c}")),
        };

        Token::new(kind, Span::new(start, self.pos))
    }

    /// Lex a syntax quote starting with a backtick.
    /// Captures either a balanced delimited block or a dotted identifier.
    fn lex_backtick(&mut self, _start: usize) -> TokenKind {
        let mut content = String::new();

        let Some(next) = self.peek_char() else {
            return TokenKind::SyntaxQuote(content);
        };

        if let Some(close) = matching_delim(next) {
            // Quoted block like `(…)`, `[…]`, `{…}`
            self.next_char(); // consume opening delimiter
            content.push(next);
            let mut stack = vec![close];
            while let Some((_, ch)) = self.next_char() {
                content.push(ch);
                if let Some(expected) = stack.last().copied() {
                    if ch == expected {
                        stack.pop();
                        if stack.is_empty() {
                            break;
                        }
                        continue;
                    }
                }
                if let Some(new_close) = matching_delim(ch) {
                    stack.push(new_close);
                }
            }
        } else {
            // Quoted identifier or dotted name: `foo, `Foo.bar
            while let Some(c) = self.peek_char() {
                if is_ident_continue(c) || c == '.' {
                    let (_, ch) = self.next_char().expect("peek_char guaranteed a character");
                    content.push(ch);
                } else {
                    break;
                }
            }
        }

        TokenKind::SyntaxQuote(content)
    }

    fn lex_string(&mut self, _start: usize) -> TokenKind {
        let mut s = String::new();
        loop {
            match self.next_char() {
                Some((_, '"')) => break,
                Some((_, '\\')) => {
                    // Escape sequence
                    match self.next_char() {
                        Some((_, 'n')) => s.push('\n'),
                        Some((_, 't')) => s.push('\t'),
                        Some((_, 'r')) => s.push('\r'),
                        Some((_, '\\')) => s.push('\\'),
                        Some((_, '"')) => s.push('"'),
                        Some((_, c)) => {
                            return TokenKind::Error(format!("unknown escape sequence: \\{c}"));
                        }
                        None => return TokenKind::Error("unterminated string".to_string()),
                    }
                }
                Some((_, c)) => s.push(c),
                None => return TokenKind::Error("unterminated string".to_string()),
            }
        }
        TokenKind::StringLit(s)
    }

    fn lex_number(&mut self, start: usize, first: char) -> TokenKind {
        let mut n: u64 = u64::from(
            first
                .to_digit(10)
                .expect("lex_number should only be called with digit start"),
        );
        let mut overflowed = false;
        while let Some(c) = self.peek_char() {
            if let Some(d) = c.to_digit(10) {
                self.next_char();
                if !overflowed {
                    match n
                        .checked_mul(10)
                        .and_then(|value| value.checked_add(u64::from(d)))
                    {
                        Some(value) => n = value,
                        None => overflowed = true,
                    }
                }
            } else if c == '_' {
                // Allow underscores in numbers: 1_000_000
                self.next_char();
            } else {
                break;
            }
        }
        if overflowed {
            let literal = &self.input[start..self.pos];
            TokenKind::Error(format!("numeric literal '{literal}' overflows u64"))
        } else {
            TokenKind::NatLit(n)
        }
    }

    fn lex_ident(&mut self, _start: usize, first: char) -> TokenKind {
        let mut s = String::new();
        s.push(first);
        while let Some(c) = self.peek_char() {
            if is_ident_continue(c) {
                s.push(c);
                self.next_char();
            } else {
                break;
            }
        }

        // Check for keywords
        match s.as_str() {
            "def" => TokenKind::Def,
            "theorem" => TokenKind::Theorem,
            "lemma" => TokenKind::Lemma,
            "axiom" => TokenKind::Axiom,
            "example" => TokenKind::Example,
            "let" => TokenKind::Let,
            "in" => TokenKind::In,
            "fun" => TokenKind::Fun,
            "forall" => TokenKind::Forall,
            "if" => TokenKind::If,
            "then" => TokenKind::Then,
            "else" => TokenKind::Else,
            "match" => TokenKind::Match,
            "with" => TokenKind::With,
            "where" => TokenKind::Where,
            "do" => TokenKind::Do,
            "return" => TokenKind::Return,
            "structure" => TokenKind::Structure,
            "class" => TokenKind::Class,
            "instance" => TokenKind::Instance,
            "inductive" => TokenKind::Inductive,
            "deriving" => TokenKind::Deriving,
            "namespace" => TokenKind::Namespace,
            "section" => TokenKind::Section,
            "end" => TokenKind::End,
            "open" => TokenKind::Open,
            "variable" => TokenKind::Variable,
            "universe" => TokenKind::Universe,
            "import" => TokenKind::Import,
            "mutual" => TokenKind::Mutual,
            "set_option" => TokenKind::SetOption,
            "by" => TokenKind::By,
            "have" => TokenKind::Have,
            "show" => TokenKind::Show,
            "suffices" => TokenKind::Suffices,
            "from" => TokenKind::From,
            "rfl" => TokenKind::Rfl,
            "sorry" => TokenKind::Sorry,
            "extends" => TokenKind::Extends,
            "private" => TokenKind::Private,
            "protected" => TokenKind::Protected,
            "partial" => TokenKind::Partial,
            "unsafe" => TokenKind::Unsafe,
            "noncomputable" => TokenKind::Noncomputable,
            "abbrev" => TokenKind::Abbrev,
            "attribute" => TokenKind::Attribute,
            "syntax" => TokenKind::Syntax,
            "macro" => TokenKind::Macro,
            "macro_rules" => TokenKind::MacroRules,
            "elab" => TokenKind::Elab,
            "infixl" => TokenKind::Infixl,
            "infixr" => TokenKind::Infixr,
            "prefix" => TokenKind::Prefix,
            "postfix" => TokenKind::Postfix,
            "notation" => TokenKind::Notation,
            "scoped" => TokenKind::Scoped,
            "rec" => TokenKind::Rec,
            "Type" => TokenKind::Type,
            "Prop" => TokenKind::Prop,
            "Sort" => TokenKind::Sort,
            _ => TokenKind::Ident(s),
        }
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || (c.is_numeric() && !c.is_ascii_digit())
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '\'' || c == '?' || c == '!'
}

fn matching_delim(c: char) -> Option<char> {
    match c {
        '(' => Some(')'),
        '[' => Some(']'),
        '{' => Some('}'),
        _ => None,
    }
}

fn is_angle_operator_char(c: char) -> bool {
    c == '<' || c == '>'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(input: &str) -> Vec<TokenKind> {
        Lexer::tokenize(input)
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| *k != TokenKind::Eof)
            .collect()
    }

    #[test]
    fn test_keywords() {
        assert_eq!(lex("def"), vec![TokenKind::Def]);
        assert_eq!(lex("theorem"), vec![TokenKind::Theorem]);
        assert_eq!(lex("let"), vec![TokenKind::Let]);
        assert_eq!(lex("fun"), vec![TokenKind::Fun]);
        assert_eq!(lex("forall"), vec![TokenKind::Forall]);
        assert_eq!(lex("Type"), vec![TokenKind::Type]);
        assert_eq!(lex("Prop"), vec![TokenKind::Prop]);
    }

    #[test]
    fn test_identifiers() {
        assert_eq!(lex("foo"), vec![TokenKind::Ident("foo".to_string())]);
        assert_eq!(lex("Nat"), vec![TokenKind::Ident("Nat".to_string())]);
        assert_eq!(lex("x'"), vec![TokenKind::Ident("x'".to_string())]);
        assert_eq!(
            lex("is_valid?"),
            vec![TokenKind::Ident("is_valid?".to_string())]
        );
    }

    #[test]
    fn test_numbers() {
        assert_eq!(lex("0"), vec![TokenKind::NatLit(0)]);
        assert_eq!(lex("42"), vec![TokenKind::NatLit(42)]);
        assert_eq!(lex("1_000_000"), vec![TokenKind::NatLit(1_000_000)]);
    }

    #[test]
    fn test_number_overflow_errors() {
        assert_eq!(
            lex("18446744073709551616"),
            vec![TokenKind::Error(
                "numeric literal '18446744073709551616' overflows u64".to_string()
            )]
        );
    }

    #[test]
    fn test_strings() {
        assert_eq!(
            lex("\"hello\""),
            vec![TokenKind::StringLit("hello".to_string())]
        );
        assert_eq!(
            lex("\"hello\\nworld\""),
            vec![TokenKind::StringLit("hello\nworld".to_string())]
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(lex("->"), vec![TokenKind::Arrow]);
        assert_eq!(lex("→"), vec![TokenKind::Arrow]);
        assert_eq!(lex("=>"), vec![TokenKind::FatArrow]);
        assert_eq!(lex(":="), vec![TokenKind::ColonEq]);
        assert_eq!(lex("λ"), vec![TokenKind::Lambda]);
        assert_eq!(lex("∀"), vec![TokenKind::Forall]);
    }

    #[test]
    fn test_delimiters() {
        assert_eq!(
            lex("(){}[]"),
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::LBracket,
                TokenKind::RBracket,
            ]
        );
    }

    #[test]
    fn test_complex() {
        let tokens = lex("def id (x : Type) := x");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Def,
                TokenKind::Ident("id".to_string()),
                TokenKind::LParen,
                TokenKind::Ident("x".to_string()),
                TokenKind::Colon,
                TokenKind::Type,
                TokenKind::RParen,
                TokenKind::ColonEq,
                TokenKind::Ident("x".to_string()),
            ]
        );
    }

    #[test]
    fn test_lambda() {
        let tokens = lex("fun x => x");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Fun,
                TokenKind::Ident("x".to_string()),
                TokenKind::FatArrow,
                TokenKind::Ident("x".to_string()),
            ]
        );
    }

    #[test]
    fn test_comments() {
        assert_eq!(
            lex("x -- comment\ny"),
            vec![
                TokenKind::Ident("x".to_string()),
                TokenKind::Ident("y".to_string()),
            ]
        );

        assert_eq!(
            lex("x /- block -/ y"),
            vec![
                TokenKind::Ident("x".to_string()),
                TokenKind::Ident("y".to_string()),
            ]
        );
    }

    #[test]
    fn test_whitespace() {
        assert_eq!(
            lex("  x  y  "),
            vec![
                TokenKind::Ident("x".to_string()),
                TokenKind::Ident("y".to_string()),
            ]
        );
    }
}
