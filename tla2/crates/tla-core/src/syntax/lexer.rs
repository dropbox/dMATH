//! TLA+ lexer using logos
//!
//! # TLA+ Token Categories
//!
//! 1. Keywords: MODULE, EXTENDS, VARIABLE, CONSTANT, etc.
//! 2. Operators: ==, =>, /\, \/, ~, etc.
//! 3. Delimiters: `( ) [ ] { }`, etc.
//! 4. Literals: numbers, strings
//! 5. Identifiers
//! 6. Comments: \* line comments, (* block comments *)

use logos::Logos;

/// Callback to lex block comments (* ... *)
/// Returns the end position of the comment relative to the start of `(*`
fn lex_block_comment(lexer: &mut logos::Lexer<Token>) -> bool {
    let remainder = lexer.remainder();
    let mut depth = 1; // Already saw opening (*
    let mut i = 0;
    let bytes = remainder.as_bytes();

    while i < bytes.len() && depth > 0 {
        if i + 1 < bytes.len() {
            if bytes[i] == b'*' && bytes[i + 1] == b')' {
                depth -= 1;
                if depth == 0 {
                    lexer.bump(i + 2); // Include the closing *)
                    return true;
                }
                i += 2;
                continue;
            }
            if bytes[i] == b'(' && bytes[i + 1] == b'*' {
                depth += 1;
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    // Unclosed comment - consume rest of input as error
    false
}

/// TLA+ tokens
#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    // === Trivia (whitespace and comments) ===
    /// Whitespace (spaces, tabs, newlines) - needed for rowan's span calculation
    #[regex(r"[ \t\r\n]+")]
    Whitespace,

    // === Module Structure ===
    // Match 4 or more dashes (TLA+ allows variable length module headers)
    #[regex(r"-{4,}")]
    ModuleStart,

    // Match 4 or more equals signs (TLA+ allows variable length module footers)
    #[regex(r"={4,}")]
    ModuleEnd,

    #[token("MODULE")]
    Module,

    #[token("EXTENDS")]
    Extends,

    #[token("INSTANCE")]
    Instance,

    #[token("WITH")]
    With,

    #[token("LOCAL")]
    Local,

    // === Declarations ===
    #[token("VARIABLE")]
    #[token("VARIABLES")]
    Variable,

    #[token("CONSTANT")]
    #[token("CONSTANTS")]
    Constant,

    #[token("ASSUME")]
    #[token("ASSUMPTION")]
    Assume,

    #[token("THEOREM")]
    Theorem,

    #[token("LEMMA")]
    Lemma,

    #[token("PROPOSITION")]
    Proposition,

    #[token("COROLLARY")]
    Corollary,

    #[token("AXIOM")]
    Axiom,

    // === Proof Keywords ===
    #[token("PROOF")]
    Proof,

    #[token("BY")]
    By,

    #[token("OBVIOUS")]
    Obvious,

    #[token("OMITTED")]
    Omitted,

    #[token("QED")]
    Qed,

    #[token("SUFFICES")]
    Suffices,

    #[token("HAVE")]
    Have,

    #[token("TAKE")]
    Take,

    #[token("WITNESS")]
    Witness,

    #[token("PICK")]
    Pick,

    #[token("USE")]
    Use,

    #[token("HIDE")]
    Hide,

    #[token("DEFINE")]
    Define,

    #[token("DEFS")]
    Defs,

    #[token("DEF")]
    Def,

    #[token("ONLY")]
    Only,

    #[token("NEW")]
    New,

    // === Logic ===
    #[token("TRUE")]
    True,

    #[token("FALSE")]
    False,

    #[token("BOOLEAN")]
    Boolean,

    #[token("IF")]
    If,

    #[token("THEN")]
    Then,

    #[token("ELSE")]
    Else,

    #[token("CASE")]
    Case,

    #[token("OTHER")]
    Other,

    #[token("LET")]
    Let,

    #[token("IN")]
    In,

    #[token("LAMBDA")]
    Lambda,

    // === Quantifiers ===
    #[token("\\A")]
    #[token("\\forall")]
    Forall,

    #[token("\\E")]
    #[token("\\exists")]
    Exists,

    #[token("\\EE")]
    TemporalExists,

    #[token("\\AA")]
    TemporalForall,

    #[token("CHOOSE")]
    Choose,

    #[token("RECURSIVE")]
    Recursive,

    // === Set Operators ===
    #[token("\\in")]
    In_,

    #[token("\\notin")]
    NotIn,

    #[token("\\cup")]
    #[token("\\union")]
    Union,

    #[token("\\cap")]
    #[token("\\intersect")]
    Intersect,

    #[token("\\")]
    #[token("\\setminus")]
    SetMinus,

    #[token("\\subseteq")]
    Subseteq,

    #[token("\\subset")]
    Subset,

    #[token("\\supseteq")]
    Supseteq,

    #[token("\\supset")]
    Supset,

    #[token("\\sqsubseteq")]
    Sqsubseteq,

    #[token("\\sqsupseteq")]
    Sqsupseteq,

    #[token("SUBSET")]
    Powerset,

    #[token("UNION")]
    BigUnion,

    #[token("INTER")]
    BigIntersect,

    // === Temporal Operators ===
    #[token("[]")]
    Always,

    #[token("<>")]
    Eventually,

    #[token("~>")]
    LeadsTo,

    #[token("ENABLED")]
    Enabled,

    #[token("UNCHANGED")]
    Unchanged,

    #[token("WF_", priority = 10)]
    WeakFair,

    #[token("SF_", priority = 10)]
    StrongFair,

    // === Logical Operators ===
    #[token("/\\")]
    #[token("\\land")]
    And,

    #[token("\\/")]
    #[token("\\lor")]
    Or,

    #[token("~")]
    #[token("\\lnot")]
    #[token("\\neg")]
    Not,

    #[token("=>")]
    Implies,

    #[token("<=>")]
    #[token("\\equiv")]
    Equiv,

    // === Comparison ===
    #[token("=")]
    Eq,

    #[token("#")]
    #[token("/=")]
    #[token("\\neq")]
    Neq,

    #[token("<")]
    Lt,

    #[token(">")]
    Gt,

    #[token("<=")]
    #[token("=<")]
    #[token("\\leq")]
    Leq,

    #[token(">=")]
    #[token("\\geq")]
    Geq,

    // === Ordering Relations (user-definable) ===
    #[token("\\prec")]
    Prec,

    #[token("\\preceq")]
    Preceq,

    #[token("\\succ")]
    Succ,

    #[token("\\succeq")]
    Succeq,

    #[token("\\ll")]
    Ll,

    #[token("\\gg")]
    Gg,

    #[token("\\sim")]
    Sim,

    #[token("\\simeq")]
    Simeq,

    #[token("\\asymp")]
    Asymp,

    #[token("\\approx")]
    Approx,

    #[token("\\cong")]
    Cong,

    #[token("\\doteq")]
    Doteq,

    #[token("\\propto")]
    Propto,

    // Action composition
    #[token("\\cdot")]
    Cdot,

    // === Arithmetic ===
    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("^")]
    Caret,

    #[token("%")]
    Percent,

    #[token("\\div")]
    Div,

    #[token("..")]
    DotDot,

    // === Multi-character user-definable operators ===
    #[token("++")]
    PlusPlus,

    #[token("--")]
    MinusMinus,

    #[token("**")]
    StarStar,

    #[token("//")]
    SlashSlash,

    #[token("^^")]
    CaretCaret,

    #[token("%%")]
    PercentPercent,

    #[token("&&")]
    AmpAmp,

    // Circled operators (user-definable)
    #[token("\\oplus")]
    Oplus,

    #[token("\\ominus")]
    Ominus,

    #[token("\\otimes")]
    Otimes,

    #[token("\\oslash")]
    Oslash,

    #[token("\\odot")]
    Odot,

    #[token("\\uplus")]
    Uplus,

    #[token("\\sqcap")]
    Sqcap,

    #[token("\\sqcup")]
    Sqcup,

    // === Definition and Assignment ===
    #[token("==")]
    DefEq,

    #[token("'")]
    Prime,

    #[token("\\triangleq")]
    TriangleEq,

    // === Delimiters ===
    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token("<<")]
    LAngle,

    #[token(">>")]
    RAngle,

    #[token(",")]
    Comma,

    #[token("::=")]
    ColonColonEq,

    #[token("::")]
    ColonColon,

    #[token(":")]
    Colon,

    #[token(";")]
    Semi,

    #[token(".")]
    Dot,

    #[token("_", priority = 3)]
    Underscore,

    #[token("@")]
    At,

    #[token("!")]
    Bang,

    #[token("|->")]
    MapsTo,

    #[token("->")]
    Arrow,

    #[token("<-")]
    LArrow,

    #[token("|-")]
    Turnstile,

    #[token("|")]
    Pipe,

    #[token(":>")]
    ColonGt,

    #[token("@@")]
    AtAt,

    #[token("$")]
    Dollar,

    #[token("$$")]
    DollarDollar,

    #[token("?")]
    Question,

    #[token("&")]
    Amp,

    #[token("\\X")]
    #[token("\\times")]
    Times,

    // === Function Operators ===
    #[token("DOMAIN")]
    Domain,

    #[token("EXCEPT")]
    Except,

    // === Sequence Operators ===
    #[token("Append")]
    Append,

    #[token("Head")]
    Head,

    #[token("Tail")]
    Tail,

    #[token("Len")]
    Len,

    #[token("Seq")]
    Seq,

    #[token("SubSeq")]
    SubSeq,

    #[token("SelectSeq")]
    SelectSeq,

    #[token("\\o")]
    #[token("\\circ")]
    Concat,

    // === Literals ===
    #[regex(r"[0-9]+")]
    Number,

    #[regex(r#""([^"\\]|\\.)*""#)]
    String,

    // === Identifiers ===
    // TLA+ identifiers normally start with a letter, followed by letters, digits, or underscores.
    // Note: PlusCal-translated specs may use _foo identifiers, but those are handled
    // by parsing _ as Underscore token. The parser can combine them if needed.
    //
    // Special case: TLA+ allows number-prefixed operator names like 1aMessage, 2avMessage
    // commonly used in consensus algorithm specs (e.g., Paxos). These must be tokenized
    // as identifiers, not as Number followed by Ident. The pattern matches:
    // - One or more digits followed by at least one letter, then letters/digits/underscores
    #[regex(r"[0-9]+[a-zA-Z][a-zA-Z0-9_]*")]
    #[regex(r"[a-zA-Z][a-zA-Z0-9_]*")]
    Ident,

    // === Comments ===
    #[regex(r"\\\*[^\n]*")]
    LineComment,

    // Block comments: (* ... *)
    // Use a callback to handle block comments with arbitrary asterisks
    #[token("(*", lex_block_comment)]
    BlockComment,
}

impl Token {
    /// Returns true if this token is trivia (whitespace/comments)
    pub fn is_trivia(&self) -> bool {
        matches!(
            self,
            Token::Whitespace | Token::LineComment | Token::BlockComment
        )
    }

    /// Returns the static text for keyword/operator tokens, or None for dynamic tokens
    pub fn static_text(&self) -> Option<&'static str> {
        match self {
            Token::Module => Some("MODULE"),
            Token::Extends => Some("EXTENDS"),
            Token::Variable => Some("VARIABLE"),
            Token::Constant => Some("CONSTANT"),
            Token::DefEq => Some("=="),
            Token::And => Some("/\\"),
            Token::Or => Some("\\/"),
            Token::Not => Some("~"),
            Token::Implies => Some("=>"),
            Token::Forall => Some("\\A"),
            Token::Exists => Some("\\E"),
            Token::True => Some("TRUE"),
            Token::False => Some("FALSE"),
            // ... add more as needed
            _ => None,
        }
    }
}

/// Lex TLA+ source code into tokens (including whitespace)
pub fn lex_all(source: &str) -> impl Iterator<Item = (Token, &str)> {
    Token::lexer(source)
        .spanned()
        .filter_map(|(result, span)| result.ok().map(|token| (token, &source[span])))
}

/// Lex TLA+ source code into non-whitespace tokens only (for tests)
/// Note: This filters whitespace but keeps comments for testing
pub fn lex(source: &str) -> impl Iterator<Item = (Token, &str)> {
    lex_all(source).filter(|(token, _)| *token != Token::Whitespace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let source = "MODULE Test";
        let tokens: Vec<_> = lex(source).collect();
        assert_eq!(
            tokens,
            vec![(Token::Module, "MODULE"), (Token::Ident, "Test"),]
        );
    }

    #[test]
    fn test_operators() {
        let source = "x == y /\\ z";
        let tokens: Vec<_> = lex(source).collect();
        assert_eq!(
            tokens,
            vec![
                (Token::Ident, "x"),
                (Token::DefEq, "=="),
                (Token::Ident, "y"),
                (Token::And, "/\\"),
                (Token::Ident, "z"),
            ]
        );
    }

    #[test]
    fn test_quantifiers() {
        let source = "\\A x \\in S : P(x)";
        let tokens: Vec<_> = lex(source).collect();
        assert_eq!(
            tokens,
            vec![
                (Token::Forall, "\\A"),
                (Token::Ident, "x"),
                (Token::In_, "\\in"),
                (Token::Ident, "S"),
                (Token::Colon, ":"),
                (Token::Ident, "P"),
                (Token::LParen, "("),
                (Token::Ident, "x"),
                (Token::RParen, ")"),
            ]
        );
    }

    #[test]
    fn test_numbers_and_strings() {
        let source = r#"x = 42 /\ y = "hello""#;
        let tokens: Vec<_> = lex(source).collect();
        assert_eq!(tokens[2], (Token::Number, "42"));
        assert_eq!(tokens[6], (Token::String, r#""hello""#));
    }

    #[test]
    fn test_comments() {
        let source = "x \\* this is a comment\ny";
        let tokens: Vec<_> = lex(source).collect();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1].0, Token::LineComment);
    }

    #[test]
    fn test_block_comments() {
        // Simple block comment
        let source = "(* hello *)";
        let tokens: Vec<_> = lex(source).collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].0, Token::BlockComment);

        // Decorative comment (many asterisks)
        let source = "(********************)";
        let tokens: Vec<_> = lex(source).collect();
        assert_eq!(tokens.len(), 1, "Decorative comment should be one token");
        assert_eq!(tokens[0].0, Token::BlockComment);

        // Comment followed by code
        let source = "(****)\nx == 1";
        let tokens: Vec<_> = lex(source).collect();
        eprintln!("Tokens for comment+code: {:?}", tokens);
        assert!(
            tokens.len() >= 4,
            "Should have comment + identifier + == + 1"
        );
        assert_eq!(tokens[0].0, Token::BlockComment);
        assert_eq!(tokens[1].0, Token::Ident);
    }

    #[test]
    fn test_decorative_comments() {
        // Test various block comment patterns including decorative lines with many asterisks
        let patterns = [
            "(* x *)",
            "(***)",
            "(****)",
            "(*****)",
            "(******)",
            "(***************************************************************************)",
        ];

        for pattern in patterns {
            let tokens: Vec<_> = lex(pattern).collect();
            assert!(!tokens.is_empty(), "Pattern {:?} should tokenize", pattern);
            assert_eq!(
                tokens[0].0,
                Token::BlockComment,
                "Pattern {:?} should be BlockComment",
                pattern
            );
        }

        // Block comment followed by code
        let source = "(*****)\nTypeOK == stuff";
        let tokens: Vec<_> = lex(source).collect();
        assert!(
            tokens.len() >= 4,
            "Should have comment + identifier + == + identifier"
        );
        assert_eq!(tokens[0].0, Token::BlockComment);
        assert_eq!(tokens[1].0, Token::Ident);
    }

    #[test]
    fn test_action_subscript_tokenization() {
        // Test that _vars is tokenized as Underscore + Ident for action subscripts
        // e.g., [Next]_vars should parse correctly
        let src = "_vars";
        let tokens: Vec<_> = lex(src).collect();
        assert_eq!(tokens.len(), 2, "Expected 2 tokens for '_vars'");
        assert_eq!(tokens[0].0, Token::Underscore);
        assert_eq!(tokens[1].0, Token::Ident);
        assert_eq!(tokens[1].1, "vars");
    }

    #[test]
    fn test_proof_step_tokens() {
        // Test tokenization of proof step labels
        let src = "<1>. QED";
        let tokens: Vec<_> = lex(src).collect();
        eprintln!("Tokens for '<1>. QED': {:?}", tokens);
        assert_eq!(tokens[0].0, Token::Lt);
        assert_eq!(tokens[1].0, Token::Number);
        assert_eq!(tokens[2].0, Token::Gt);
        assert_eq!(tokens[3].0, Token::Dot);
        assert_eq!(tokens[4].0, Token::Qed);
    }

    #[test]
    fn test_proof_step_with_label() {
        // Test tokenization of proof step labels with numeric suffix
        let src = "<1>2. Foo";
        let tokens: Vec<_> = lex(src).collect();
        eprintln!("Tokens for '<1>2. Foo': {:?}", tokens);
        assert_eq!(tokens[0].0, Token::Lt);
        assert_eq!(tokens[1].0, Token::Number);
        assert_eq!(tokens[1].1, "1");
        assert_eq!(tokens[2].0, Token::Gt);
        assert_eq!(tokens[3].0, Token::Number);
        assert_eq!(tokens[3].1, "2");
        assert_eq!(tokens[4].0, Token::Dot);
        assert_eq!(tokens[5].0, Token::Ident);
    }

    #[test]
    fn test_number_prefixed_identifiers() {
        // TLA+ allows number-prefixed identifiers like 1aMessage, 2avMessage
        // commonly used in consensus algorithm specs (Paxos, BFT, etc.)
        let src = "1aMessage 2avMessage 1bMessage 1cMessage";
        let tokens: Vec<_> = lex(src).collect();
        eprintln!("Tokens for number-prefixed identifiers: {:?}", tokens);
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], (Token::Ident, "1aMessage"));
        assert_eq!(tokens[1], (Token::Ident, "2avMessage"));
        assert_eq!(tokens[2], (Token::Ident, "1bMessage"));
        assert_eq!(tokens[3], (Token::Ident, "1cMessage"));
    }

    #[test]
    fn test_number_prefixed_in_definition() {
        // Test in context of operator definition
        let src = "1aMessage == [type : {\"1a\"}, bal : Ballot]";
        let tokens: Vec<_> = lex(src).collect();
        assert_eq!(tokens[0], (Token::Ident, "1aMessage"));
        assert_eq!(tokens[1], (Token::DefEq, "=="));
    }

    #[test]
    fn test_number_prefixed_complex() {
        // Test various number-prefixed identifier patterns
        let src = "1bOr2bMsgs";
        let tokens: Vec<_> = lex(src).collect();
        eprintln!("Tokens for '1bOr2bMsgs': {:?}", tokens);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], (Token::Ident, "1bOr2bMsgs"));
    }

    #[test]
    fn test_wf_sf_tokenization() {
        // WF_xxx and SF_xxx are tokenized as single identifiers due to lexer "maximal munch".
        // This is correct behavior - the lowering phase (lower.rs) handles converting
        // WF_xxx/SF_xxx identifiers to proper WeakFair/StrongFair AST nodes.
        let src = "WF_vars";
        let tokens: Vec<_> = lex(src).collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], (Token::Ident, "WF_vars"));

        let src2 = "SF_cvars";
        let tokens2: Vec<_> = lex(src2).collect();
        assert_eq!(tokens2.len(), 1);
        assert_eq!(tokens2[0], (Token::Ident, "SF_cvars"));
    }
}
