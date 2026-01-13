//! Recursive descent parser for Lean 4 syntax
//!
//! Parses tokens into a surface syntax AST.

use crate::lexer::{Lexer, Token, TokenKind};
use crate::surface::{
    Attribute, LevelExpr, MacroArm, NotationItem, NotationKind, OpenPath, Projection, Span,
    SurfaceArg, SurfaceBinder, SurfaceBinderInfo, SurfaceCtor, SurfaceDecl, SurfaceExpr,
    SurfaceField, SurfaceFieldAssign, SurfaceLit, SurfaceMatchArm, SurfacePattern,
    SyntaxPatternItem, UniverseExpr,
};
use crate::ParseError;

/// Parser state
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    #[must_use]
    pub fn new(input: &str) -> Self {
        Self {
            tokens: Lexer::tokenize(input),
            pos: 0,
        }
    }

    /// Parse an expression
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or parsing fails.
    pub fn parse_expr(input: &str) -> Result<SurfaceExpr, ParseError> {
        let mut parser = Parser::new(input);
        parser.expr()
    }

    /// Parse a declaration
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or parsing fails.
    pub fn parse_decl(input: &str) -> Result<SurfaceDecl, ParseError> {
        let mut parser = Parser::new(input);
        parser.decl()
    }

    /// Parse a file containing multiple declarations
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or parsing fails.
    pub fn parse_file(input: &str) -> Result<Vec<SurfaceDecl>, ParseError> {
        let mut parser = Parser::new(input);
        parser.file()
    }

    /// Parse a file (sequence of declarations)
    fn file(&mut self) -> Result<Vec<SurfaceDecl>, ParseError> {
        let mut decls = Vec::new();

        while !matches!(self.current_kind(), TokenKind::Eof) {
            decls.push(self.decl()?);
        }

        Ok(decls)
    }

    // Token access

    fn current(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .unwrap_or_else(|| self.tokens.last().expect("tokens should have at least EOF"))
    }

    fn current_kind(&self) -> &TokenKind {
        &self.current().kind
    }

    fn current_span(&self) -> Span {
        self.current().span
    }

    fn advance(&mut self) -> &Token {
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        // Return reference to token we just passed
        &self.tokens[self.pos.saturating_sub(1)]
    }

    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(self.current_kind()) == std::mem::discriminant(kind)
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<&Token, ParseError> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected {:?}, got {:?}", kind, self.current_kind()),
            })
        }
    }

    #[allow(dead_code)]
    fn at_end(&self) -> bool {
        matches!(self.current_kind(), TokenKind::Eof)
    }

    /// Peek at token kind at offset from current position
    fn peek_kind(&self, offset: usize) -> Option<&TokenKind> {
        self.tokens.get(self.pos + offset).map(|t| &t.kind)
    }

    // Parsing

    /// Parse an expression
    fn expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        self.pipe_expr()
    }

    /// Backward pipe expressions: f <| x (very low precedence, right-associative)
    /// Precedence 10 in Lean 4 - lower than ↔
    fn pipe_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let left = self.iff_expr()?;

        // <| is right-associative: f <| g <| x = f <| (g <| x)
        if self.eat(&TokenKind::BackwardPipe) {
            let right = self.pipe_expr()?; // Recursive for right-associativity
            let span = left.span().merge(right.span());
            // f <| x is equivalent to f x (low-precedence application)
            return Ok(SurfaceExpr::App(
                span,
                Box::new(left),
                vec![SurfaceArg::positional(right)],
            ));
        }

        Ok(left)
    }

    /// Iff expressions: A ↔ B (precedence 20)
    fn iff_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.or_expr()?;

        while self.eat(&TokenKind::Iff) {
            let right = self.or_expr()?;
            let span = left.span().merge(right.span());
            // Create application: Iff A B
            left = SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "Iff".to_string())),
                vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
            );
        }

        Ok(left)
    }

    /// Or expressions: A ∨ B
    fn or_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.and_expr()?;

        while self.eat(&TokenKind::Or) {
            let right = self.and_expr()?;
            let span = left.span().merge(right.span());
            left = SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "Or".to_string())),
                vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
            );
        }

        Ok(left)
    }

    /// And expressions: A ∧ B
    fn and_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.cmp_expr()?;

        while self.eat(&TokenKind::And) {
            let right = self.cmp_expr()?;
            let span = left.span().merge(right.span());
            left = SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "And".to_string())),
                vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
            );
        }

        Ok(left)
    }

    /// Comparison expressions: A = B, A ≠ B, A < B, A ≤ B, etc.
    fn cmp_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.arrow_expr()?;

        loop {
            let span = left.span();
            if self.eat(&TokenKind::Eq) {
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "Eq".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Ne) {
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "Ne".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Lt) {
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "LT.lt".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Le) {
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "LE.le".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Gt) {
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "GT.gt".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Ge) {
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "GE.ge".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::HEq) {
                // Heterogeneous equality: a ≍ b
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "HEq".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Equiv) {
                // Equivalence/isomorphism: a ≃ b
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "Equiv".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::DoubleEq) {
                // BEq equality check: a == b
                let right = self.arrow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "BEq.beq".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Arrow types: A → B (right associative)
    fn arrow_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        // Attempt to parse binder-style pi types: (x : T) → U, {x : T} → U, [x : T] → U
        // Use lookahead to avoid exponential backtracking on nested brackets/parens
        let saved_pos = self.pos;
        if self.looks_like_binder_start() {
            if let Ok(binders) = self.binders() {
                if self.eat(&TokenKind::Arrow) {
                    let body = self.arrow_expr()?;
                    let start_span = binders
                        .first()
                        .map_or_else(|| self.current_span(), |b| b.span);
                    let span = start_span.merge(body.span());
                    return Ok(SurfaceExpr::Pi(span, binders, Box::new(body)));
                }
            }
            // Backtrack if this wasn't a binder-arrow form
            self.pos = saved_pos;
        }

        let mut left = self.prod_expr()?;

        while self.eat(&TokenKind::Arrow) {
            let right = self.arrow_expr()?; // Right associative
            let span = left.span().merge(right.span());
            left = SurfaceExpr::Arrow(span, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Product type: A × B
    fn prod_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.add_expr()?;

        while self.eat(&TokenKind::Times) {
            let right = self.add_expr()?;
            let span = left.span().merge(right.span());
            left = SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "Prod".to_string())),
                vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
            );
        }

        Ok(left)
    }

    /// Additive expressions: A + B, A - B
    fn add_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.cons_expr()?;

        loop {
            let span = left.span();
            if self.eat(&TokenKind::Plus) {
                let right = self.mul_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "HAdd.hAdd".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Minus) {
                let right = self.cons_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "HSub.hSub".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Cons operator: x :: xs (right associative, precedence 67)
    fn cons_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let left = self.mul_expr()?;

        if self.eat(&TokenKind::ColonColon) {
            let right = self.cons_expr()?; // Right associative
            let span = left.span().merge(right.span());
            Ok(SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "List.cons".to_string())),
                vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
            ))
        } else {
            Ok(left)
        }
    }

    /// Multiplicative expressions: A * B, A / B
    fn mul_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.pow_expr()?;

        loop {
            let span = left.span();
            if self.eat(&TokenKind::Star) {
                let right = self.pow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "HMul.hMul".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Slash) {
                let right = self.pow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "HDiv.hDiv".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else if self.eat(&TokenKind::Percent) {
                // Modulo operator: a % b
                let right = self.pow_expr()?;
                let end_span = right.span();
                left = SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "HMod.hMod".to_string())),
                    vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
                );
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Exponentiation: A ^ B (right associative, higher precedence than * /)
    fn pow_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let left = self.unary_expr()?;

        if self.eat(&TokenKind::Caret) {
            let right = self.pow_expr()?; // Right associative
            let span = left.span().merge(right.span());
            Ok(SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "HPow.hPow".to_string())),
                vec![SurfaceArg::positional(left), SurfaceArg::positional(right)],
            ))
        } else {
            Ok(left)
        }
    }

    /// Prefix operators: unary - and +
    fn unary_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        match self.current_kind() {
            TokenKind::Minus => {
                let start_span = self.current_span();
                self.advance();
                let inner = self.unary_expr()?;
                let span = start_span.merge(inner.span());
                Ok(SurfaceExpr::App(
                    span,
                    Box::new(SurfaceExpr::Ident(span, "Neg.neg".to_string())),
                    vec![SurfaceArg::positional(inner)],
                ))
            }
            TokenKind::Plus => {
                // Unary plus is a no-op; consume and parse inner
                self.advance();
                self.unary_expr()
            }
            _ => self.app_expr(),
        }
    }

    /// Application: f x y z
    fn app_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut expr = self.atom_expr()?;
        let mut pending_args: Vec<SurfaceArg> = Vec::new();

        loop {
            // Check for Dot FIRST - if followed by ident/number, it's a projection
            // If followed by {, it's universe instantiation (Foo.{u v})
            // Otherwise, let is_atom_start() handle it as anonymous constructor
            if self.check(&TokenKind::Dot) {
                // Peek at what follows the dot
                let is_projection = matches!(
                    self.peek_kind(1),
                    Some(TokenKind::Ident(_) | TokenKind::NatLit(_))
                );
                let is_universe_inst = matches!(self.peek_kind(1), Some(TokenKind::LBrace));

                if is_universe_inst {
                    self.advance(); // consume the dot
                    self.advance(); // consume the {

                    // Parse universe levels: Foo.{u v w}
                    let mut levels = Vec::new();
                    while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                        levels.push(self.level_expr()?);
                    }
                    self.expect(&TokenKind::RBrace)?;

                    // For now, just annotate the expression with universe info
                    // The actual effect is that we parsed past the .{...}
                    // The elaborator will handle the universe instantiation
                    let end_span = self.current_span();
                    let span = expr.span().merge(end_span);
                    expr = SurfaceExpr::UniverseInst(span, Box::new(expr), levels);
                    continue;
                }

                if is_projection {
                    self.advance(); // consume the dot

                    // If we have pending arguments, flush them into an application node
                    if !pending_args.is_empty() {
                        let span = expr.span();
                        expr = SurfaceExpr::App(span, Box::new(expr), pending_args);
                        pending_args = Vec::new();
                    }

                    let (projection, end_span) = match self.current_kind().clone() {
                        TokenKind::Ident(field) => {
                            let end_span = self.current_span();
                            self.advance();
                            (Projection::Named(field), end_span)
                        }
                        TokenKind::NatLit(n) => {
                            let end_span = self.current_span();
                            self.advance();
                            let idx =
                                u32::try_from(n).map_err(|_| ParseError::UnexpectedToken {
                                    line: 0,
                                    col: self.current_span().start,
                                    message: format!("projection index too large: {n}"),
                                })?;
                            (Projection::Index(idx), end_span)
                        }
                        _ => unreachable!("peek_kind already checked"),
                    };

                    let proj_span = expr.span().merge(end_span);
                    expr = SurfaceExpr::Proj(proj_span, Box::new(expr), projection);
                    continue;
                }
                // If not a projection (e.g., `.` followed by something else),
                // fall through to is_atom_start() which will handle it as
                // anonymous constructor syntax if applicable
            }

            if self.is_atom_start() {
                let arg = self.atom_expr()?;

                // If the argument is a pattern-matching lambda, stop application parsing
                // Pattern-matching lambdas use layout-sensitive syntax and we can't
                // reliably determine where they end without indentation info
                let is_pattern_lambda = matches!(&arg, SurfaceExpr::PatternMatchLambda(_, _, _));

                pending_args.push(SurfaceArg::positional(arg));

                if is_pattern_lambda {
                    // Stop collecting arguments after pattern-matching lambda
                    break;
                }
                continue;
            }

            break;
        }

        if pending_args.is_empty() {
            Ok(expr)
        } else {
            let span = expr.span();
            Ok(SurfaceExpr::App(span, Box::new(expr), pending_args))
        }
    }

    fn is_atom_start(&self) -> bool {
        // Note: SetOption is NOT here because `set_option` should not be
        // considered an application argument. It appears at expression-start
        // only in the form `set_option ... in expr`, which is handled by
        // atom_expr() directly, but should not be consumed as an argument
        // to a preceding function call.

        // Special case: @ followed by identifier is explicit application (@f),
        // but @ followed by [ is an attribute (@[...])
        if matches!(self.current_kind(), TokenKind::At) {
            return matches!(self.peek_kind(1), Some(TokenKind::Ident(_)));
        }

        matches!(
            self.current_kind(),
            TokenKind::Ident(_)
                | TokenKind::NatLit(_)
                | TokenKind::StringLit(_)
                | TokenKind::LParen
                | TokenKind::Type
                | TokenKind::Prop
                | TokenKind::Sort
                | TokenKind::Underscore
                | TokenKind::LAngle  // ⟨...⟩ anonymous constructor
                | TokenKind::Not     // ¬ prefix operator
                | TokenKind::By      // by tactic
                | TokenKind::Sorry   // sorry
                | TokenKind::Rfl     // rfl
                | TokenKind::Match   // match expression
                | TokenKind::Do      // do notation
                | TokenKind::Exists  // ∃ exists
                | TokenKind::Show    // show ... from ...
                // Note: Open is NOT included - `open` at start of line is a declaration,
                // not an expression argument. Use parentheses: f (open Foo in x)
                | TokenKind::LBrace  // record literals
                | TokenKind::LBracket // [a, b] list literals
                // Note: Hash is NOT included - `#foo` commands should only appear at declaration level,
                // not as expression arguments. Only `#[...]` array literals are valid in expressions,
                // but we handle that specially by checking for `#` followed by `[`
                | TokenKind::Fun     // fun x => ... lambdas
                | TokenKind::Lambda  // λ x => ... lambdas
                | TokenKind::If      // if ... then ... else ...
                // Note: Let is NOT included - let expressions cannot be bare function arguments
                // They must be parenthesized: f (let x := 1 in x)
                | TokenKind::Forall  // ∀ forall
                | TokenKind::Pi      // Π pi type
                | TokenKind::Cdot    // · section placeholder
                | TokenKind::SyntaxQuote(_)
                | TokenKind::Error(_) // Allow error recovery to parse invalid characters
        )
    }

    /// Check if current position looks like the start of a pi-type binder.
    /// Uses lookahead to distinguish binders from list/tuple literals.
    /// This avoids expensive speculative parsing that caused O(2^n) behavior.
    ///
    /// Patterns that look like binders:
    /// - `(x :` or `(x y :` - explicit binder with type annotation
    /// - `(x)` followed by `→` - simple binder (rare, handled by backtrack)
    /// - `{x :` - implicit binder
    /// - `[Name` where Name starts uppercase - instance binder like [Ord A]
    /// - `[x :` - named instance binder
    ///
    /// Patterns that are NOT binders:
    /// - `[[` - nested list literal
    /// - `[1` or `["` - list with literal element
    /// - `((` - nested parentheses (might be tuple)
    fn looks_like_binder_start(&self) -> bool {
        let curr = self.current_kind();
        let next = self.tokens.get(self.pos + 1).map(|t| &t.kind);

        match (curr, next) {
            // `((` - nested parens, likely not a binder
            // `()` - unit, not a binder
            // `(` followed by literal - tuple, not binder
            // `{{` - nested braces, not a binder
            // `{}` - empty, not a binder
            // `[[` - nested list literal, NOT a binder
            // `[]` - empty list, not a binder
            // `[number` or `[string` - list literal
            (
                TokenKind::LParen,
                Some(
                    TokenKind::LParen
                    | TokenKind::RParen
                    | TokenKind::NatLit(_)
                    | TokenKind::StringLit(_),
                ),
            )
            | (TokenKind::LBrace, Some(TokenKind::LBrace | TokenKind::RBrace))
            | (
                TokenKind::LBracket,
                Some(
                    TokenKind::LBracket
                    | TokenKind::RBracket
                    | TokenKind::NatLit(_)
                    | TokenKind::StringLit(_),
                ),
            ) => false,

            // `(x` where x is an identifier - could be binder
            // `(` followed by other - might be binder
            // `{x` where x is an identifier - likely implicit binder
            // `{` followed by other - might be binder
            (TokenKind::LParen | TokenKind::LBrace, Some(TokenKind::Ident(_)) | _) => true,

            // `[Name` where Name starts uppercase - likely instance binder like [Ord A]
            (TokenKind::LBracket, Some(TokenKind::Ident(name))) => {
                name.chars().next().is_some_and(char::is_uppercase)
            }
            // `[` followed by lowercase ident - ambiguous, could be list or binder
            // Be conservative: only try binder if followed by `:` pattern
            (TokenKind::LBracket, _) => {
                // Check if there's a colon within the next few tokens
                // This is a heuristic to avoid exponential behavior
                for i in 1..=4 {
                    match self.tokens.get(self.pos + i).map(|t| &t.kind) {
                        Some(TokenKind::Colon) => return true,
                        Some(TokenKind::RBracket | TokenKind::LBracket | TokenKind::Comma) => {
                            return false
                        }
                        _ => {}
                    }
                }
                false
            }

            // Bare identifiers are NOT binders in arrow_expr context
            // The original code only tried binders for ( { [
            // `A → B` should parse as Arrow, not Pi with untyped binder
            _ => false,
        }
    }

    /// Check if current token can implicitly start a let body
    /// This is for handling layout-sensitive code where the body follows
    /// without an explicit `in` separator
    fn is_implicit_body_start(&self) -> bool {
        // Only simple identifiers can implicitly start a body
        // This is conservative but handles the common case
        matches!(self.current_kind(), TokenKind::Ident(_))
    }

    /// Parse an expression for implicit let body
    /// This parses a full expression - the caller handles determining when
    /// the body ends (typically by layout/indentation in Lean 4, which we
    /// approximate by checking for command-starting tokens).
    fn implicit_let_body_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        // Parse a full expression - operators, applications, etc.
        // The expression ends naturally when we hit something that can't
        // continue the expression (like a new command or EOF).
        self.expr()
    }

    /// Atomic expressions
    fn atom_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let span = self.current_span();

        match self.current_kind().clone() {
            TokenKind::Ident(name) => {
                // Handle special identifiers that act as type-level operators
                if name == "outParam" {
                    self.advance();
                    // outParam requires an argument
                    if self.is_atom_start() {
                        let inner = self.atom_expr()?;
                        let end_span = inner.span();
                        return Ok(SurfaceExpr::OutParam(span.merge(end_span), Box::new(inner)));
                    }
                    // Just `outParam` alone - return as identifier
                    return Ok(SurfaceExpr::Ident(span, name));
                }

                if name == "semiOutParam" {
                    self.advance();
                    // semiOutParam requires an argument
                    if self.is_atom_start() {
                        let inner = self.atom_expr()?;
                        let end_span = inner.span();
                        return Ok(SurfaceExpr::SemiOutParam(
                            span.merge(end_span),
                            Box::new(inner),
                        ));
                    }
                    // Just `semiOutParam` alone - return as identifier
                    return Ok(SurfaceExpr::Ident(span, name));
                }

                self.advance();
                // Support dotted names like `Nat.succ`
                let mut full_name = name;
                let allow_qualified = full_name.chars().next().is_some_and(char::is_uppercase);
                let mut end_span = span;
                while allow_qualified && self.check(&TokenKind::Dot) {
                    let next_token = match self.tokens.get(self.pos + 1) {
                        Some(tok) => tok.clone(),
                        None => break,
                    };

                    match next_token.kind {
                        TokenKind::Ident(next) => {
                            // Consume '.' and identifier
                            self.advance(); // dot
                            self.advance(); // identifier
                            full_name.push('.');
                            full_name.push_str(&next);
                            end_span = next_token.span;
                        }
                        _ => break, // Leave the dot for projection parsing
                    }
                }

                Ok(SurfaceExpr::Ident(
                    Span::new(span.start, end_span.end),
                    full_name,
                ))
            }

            TokenKind::NatLit(n) => {
                self.advance();
                Ok(SurfaceExpr::Lit(span, SurfaceLit::Nat(n)))
            }

            TokenKind::StringLit(s) => {
                self.advance();
                Ok(SurfaceExpr::Lit(span, SurfaceLit::String(s)))
            }

            // Quoted syntax/macros - treat as opaque holes for now
            TokenKind::SyntaxQuote(content) => {
                self.advance();
                Ok(SurfaceExpr::SyntaxQuote(span, content))
            }

            // Explicit application: @f - disables implicit argument insertion
            // @f x y means call f with all implicit args provided explicitly
            TokenKind::At => {
                self.advance(); // consume @
                                // Parse the expression following @
                                // This should be an identifier or parenthesized expression
                let inner = self.atom_expr()?;
                let end_span = inner.span();
                Ok(SurfaceExpr::Explicit(span.merge(end_span), Box::new(inner)))
            }

            TokenKind::Type => {
                self.advance();
                // Check for explicit level: Type u
                if let TokenKind::Ident(_) | TokenKind::NatLit(_) = self.current_kind() {
                    let level = self.level_expr()?;
                    Ok(SurfaceExpr::Universe(
                        span,
                        UniverseExpr::TypeLevel(Box::new(level)),
                    ))
                } else {
                    Ok(SurfaceExpr::Universe(span, UniverseExpr::Type))
                }
            }

            TokenKind::Prop => {
                self.advance();
                Ok(SurfaceExpr::Universe(span, UniverseExpr::Prop))
            }

            TokenKind::Sort => {
                self.advance();
                // Check for explicit level: Sort u, Sort 0, Sort (u + 1), etc.
                if let TokenKind::Ident(_) | TokenKind::NatLit(_) | TokenKind::LParen =
                    self.current_kind()
                {
                    let level = self.level_expr()?;
                    Ok(SurfaceExpr::Universe(
                        span,
                        UniverseExpr::Sort(Box::new(level)),
                    ))
                } else {
                    // Sort without explicit level = Sort u for fresh universe variable
                    Ok(SurfaceExpr::Universe(span, UniverseExpr::SortImplicit))
                }
            }

            TokenKind::Underscore => {
                self.advance();
                Ok(SurfaceExpr::Hole(span))
            }

            TokenKind::LParen => {
                self.advance();
                // Could be: (), (e), (x : T), (e : T), (e1, e2, ...) tuple, or (x := e) named arg

                // Empty tuple/unit
                if self.check(&TokenKind::RParen) {
                    let end_span = self.current_span();
                    self.advance();
                    return Ok(SurfaceExpr::Ident(
                        span.merge(end_span),
                        "Unit.unit".to_string(),
                    ));
                }

                // Check for named argument syntax: (ident := expr)
                // This is used in contexts like `f (α := o)` where α is a parameter name
                if let TokenKind::Ident(name) = self.current_kind().clone() {
                    if matches!(self.peek_kind(1), Some(TokenKind::ColonEq)) {
                        self.advance(); // consume ident
                        self.advance(); // consume :=
                        let value = self.expr()?;
                        self.expect(&TokenKind::RParen)?;
                        let end_span = self.current_span();
                        // Represent named argument as special application to placeholder
                        // The elaborator will handle this as a named argument
                        return Ok(SurfaceExpr::NamedArg(
                            span.merge(end_span),
                            name,
                            Box::new(value),
                        ));
                    }
                }

                let expr = self.expr()?;

                if self.eat(&TokenKind::Colon) {
                    // Type ascription: (e : T)
                    let ty = self.expr()?;
                    self.expect(&TokenKind::RParen)?;
                    let end_span = self.current_span();
                    Ok(SurfaceExpr::Ascription(
                        span.merge(end_span),
                        Box::new(expr),
                        Box::new(ty),
                    ))
                } else if self.eat(&TokenKind::Comma) {
                    // Tuple: (e1, e2, ...)
                    let mut elems = vec![expr];
                    // Allow trailing comma before RParen
                    if !self.check(&TokenKind::RParen) {
                        elems.push(self.expr()?);
                        while self.eat(&TokenKind::Comma) {
                            if self.check(&TokenKind::RParen) {
                                break; // trailing comma
                            }
                            elems.push(self.expr()?);
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    let end_span = self.current_span();

                    // Build nested Prod.mk: (a, b, c) -> Prod.mk a (Prod.mk b c)
                    let result = elems
                        .into_iter()
                        .rev()
                        .reduce(|acc, elem| {
                            let s = span.merge(end_span);
                            SurfaceExpr::App(
                                s,
                                Box::new(SurfaceExpr::Ident(s, "Prod.mk".to_string())),
                                vec![SurfaceArg::positional(elem), SurfaceArg::positional(acc)],
                            )
                        })
                        .expect("tuple must have elements");

                    Ok(SurfaceExpr::Paren(span.merge(end_span), Box::new(result)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    let end_span = self.current_span();
                    Ok(SurfaceExpr::Paren(span.merge(end_span), Box::new(expr)))
                }
            }

            TokenKind::Fun | TokenKind::Lambda => {
                self.advance();
                self.lambda_body(span)
            }

            TokenKind::Forall | TokenKind::Pi => {
                self.advance();
                self.forall_body(span)
            }

            TokenKind::Let => {
                self.advance();
                self.let_body(span)
            }

            TokenKind::If => {
                self.advance();
                self.if_body(span)
            }

            TokenKind::By => {
                self.advance();
                Ok(self.by_body(span))
            }

            TokenKind::Sorry => {
                self.advance();
                Ok(SurfaceExpr::Ident(span, "sorry".to_string()))
            }

            TokenKind::Rfl => {
                self.advance();
                Ok(SurfaceExpr::Ident(span, "rfl".to_string()))
            }

            TokenKind::Not => {
                // ¬ as prefix operator
                self.advance();
                let inner = self.atom_expr()?;
                let end_span = inner.span();
                Ok(SurfaceExpr::App(
                    span.merge(end_span),
                    Box::new(SurfaceExpr::Ident(span, "Not".to_string())),
                    vec![SurfaceArg::positional(inner)],
                ))
            }

            TokenKind::LAngle => {
                // ⟨...⟩ anonymous constructor
                self.advance();
                self.anon_constructor_body(span)
            }

            TokenKind::LBrace => {
                self.advance();
                self.record_literal_body(span)
            }

            TokenKind::Match => {
                self.advance();
                self.match_body(span)
            }

            TokenKind::Open => {
                self.advance();
                self.open_expr_body(span)
            }

            TokenKind::Do => {
                self.advance();
                Ok(self.do_body(span))
            }

            TokenKind::Exists => {
                // ∃ prefix for exists
                self.advance();
                self.exists_body(span)
            }

            TokenKind::Show => {
                self.advance();
                self.show_body(span)
            }

            TokenKind::Have => {
                self.advance();
                self.have_body(span)
            }

            TokenKind::Suffices => {
                self.advance();
                self.suffices_body(span)
            }

            TokenKind::LBracket => {
                self.advance();
                self.list_literal_body(span)
            }

            TokenKind::SetOption => {
                self.advance();
                self.set_option_expr(span)
            }

            TokenKind::Hash => {
                // Array literal syntax `#[...]`
                let next_is_lbracket = matches!(
                    self.tokens.get(self.pos + 1).map(|t| &t.kind),
                    Some(TokenKind::LBracket)
                );
                self.advance();
                if next_is_lbracket {
                    let _ = self.expect(&TokenKind::LBracket)?;
                    self.array_literal_body(span)
                } else {
                    // Treat other uses of `#` in expressions as holes for now
                    Ok(SurfaceExpr::Hole(span))
                }
            }

            TokenKind::Dot => {
                // Anonymous constructor: .foo or .foo args
                // Leading dot indicates constructor whose type is inferred from context
                self.advance();
                match self.current_kind().clone() {
                    TokenKind::Ident(name) => {
                        let end_span = self.current_span();
                        self.advance();
                        // Build a dotted identifier like ".foo"
                        // The elaborator will resolve this to the appropriate constructor
                        let mut full_name = format!(".{name}");
                        let mut final_span = end_span;
                        // Handle nested dotted access like .foo.bar
                        while self.check(&TokenKind::Dot) {
                            if let Some(TokenKind::Ident(next)) = self.peek_kind(1).cloned() {
                                self.advance(); // dot
                                final_span = self.current_span();
                                self.advance(); // ident
                                full_name.push('.');
                                full_name.push_str(&next);
                            } else {
                                break;
                            }
                        }
                        Ok(SurfaceExpr::Ident(span.merge(final_span), full_name))
                    }
                    TokenKind::LParen => {
                        // .(expr) - parenthesized anonymous constructor call
                        self.advance(); // consume lparen
                        let inner = self.expr()?;
                        let end_span = self.current_span();
                        self.expect(&TokenKind::RParen)?;
                        // Represent as application of hole to inner
                        Ok(SurfaceExpr::App(
                            span.merge(end_span),
                            Box::new(SurfaceExpr::Hole(span)),
                            vec![SurfaceArg::positional(inner)],
                        ))
                    }
                    other => Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: self.current_span().start,
                        message: format!("expected identifier after '.', got {other:?}"),
                    }),
                }
            }

            TokenKind::Cdot => {
                // · (middle dot / cdot) - section placeholder
                // Used in section notation: (· + ·) creates fun x y => x + y
                // We represent it as a special hole marker that the elaborator will resolve
                self.advance();
                Ok(SurfaceExpr::Ident(span, "·".to_string()))
            }

            TokenKind::Error(msg) => {
                // Handle lexer errors gracefully - treat invalid characters as holes
                // This allows parsing malformed test files that test error recovery
                let msg = msg.clone();
                self.advance();
                // If the error is for an invalid character like Unicode replacement,
                // treat it as a hole to allow parsing to continue
                if msg.contains("unexpected character") {
                    Ok(SurfaceExpr::Hole(span))
                } else {
                    Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: span.start,
                        message: format!("lexer error: {msg}"),
                    })
                }
            }

            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: span.start,
                message: format!("unexpected token: {:?}", self.current_kind()),
            }),
        }
    }

    /// Parse lambda body: x y z => e or (x : T) (y : U) => e
    /// Returns (`lambda_expr`, `is_pattern_matching`) where `is_pattern_matching`
    /// indicates if the lambda used pattern syntax (important for layout)
    fn lambda_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        // Check for pattern-matching lambda: fun | pat => e | pat2 => e2
        if self.check(&TokenKind::Pipe) {
            // Parse as pattern-matching lambda
            let mut arms = Vec::new();
            while self.eat(&TokenKind::Pipe) {
                let pattern = self.pattern_with_cons()?;
                self.expect(&TokenKind::FatArrow)?;
                // Parse body until next | or end of lambda
                let body = self.lambda_arm_body()?;
                arms.push(SurfaceMatchArm {
                    span: pattern.span(),
                    pattern,
                    body,
                });
            }
            // Convert to match on anonymous variable
            // Mark this as a pattern-matching lambda by wrapping in a special form
            let scrutinee = SurfaceExpr::Ident(start_span, "_x".to_string());
            let binder = SurfaceBinder::new("_x".to_string(), None, SurfaceBinderInfo::Explicit);
            let match_expr = SurfaceExpr::Match(start_span, Box::new(scrutinee), arms);
            // Return as PatternMatchLambda to signal app_expr to stop
            return Ok(SurfaceExpr::PatternMatchLambda(
                start_span,
                vec![binder],
                Box::new(match_expr),
            ));
        }

        // Regular lambda: fun x => e
        let binders = self.binders()?;
        self.expect(&TokenKind::FatArrow)?;
        let body = self.expr()?;
        let span = start_span.merge(body.span());
        Ok(SurfaceExpr::Lambda(span, binders, Box::new(body)))
    }

    /// Parse the body of a lambda arm (until next | or lambda end)
    fn lambda_arm_body(&mut self) -> Result<SurfaceExpr, ParseError> {
        // The tricky part: we need to parse an expression but stop at the next |
        // We use a limited expression parser that doesn't consume |
        self.lambda_arm_expr()
    }

    /// Parse an expression for lambda arm body (stops at |)
    /// This is a simplified expression parser that doesn't allow multi-atom applications
    /// to help with layout-free parsing of pattern-matching lambdas
    fn lambda_arm_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        // Parse a simple expression - just an atom with optional projections
        // This is a conservative approach that works for most lambda arm bodies
        // Full applications would need layout information to disambiguate
        let expr = self.atom_expr()?;

        // Allow projections on the result
        let mut result = expr;
        while self.eat(&TokenKind::Dot) {
            match self.current_kind().clone() {
                TokenKind::Ident(field) => {
                    let field_span = self.current_span();
                    self.advance();
                    let span = result.span().merge(field_span);
                    result = SurfaceExpr::Proj(span, Box::new(result), Projection::Named(field));
                }
                TokenKind::NatLit(n) => {
                    let field_span = self.current_span();
                    let index = u32::try_from(n).map_err(|_| ParseError::NumericOverflow {
                        value: n,
                        max: u64::from(u32::MAX),
                    })?;
                    self.advance();
                    let span = result.span().merge(field_span);
                    result = SurfaceExpr::Proj(span, Box::new(result), Projection::Index(index));
                }
                _ => break,
            }
        }

        Ok(result)
    }

    /// Parse forall body: (x : T) (y : U), B
    fn forall_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let binders = self.binders()?;
        self.expect(&TokenKind::Comma)?;
        let body = self.expr()?;
        let span = start_span.merge(body.span());
        Ok(SurfaceExpr::Pi(span, binders, Box::new(body)))
    }

    /// Parse let body: x : T := v in e
    /// Also supports:
    /// - Chained let bindings without `in`: let x := 1; let y := 2; x + y
    /// - Recursive let: let rec f (n : Nat) : Nat := ...
    /// - Function let: let f x := x; f 0 (desugars to let f := fun x => x)
    fn let_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        // Check for `rec` keyword
        let is_rec = self.eat(&TokenKind::Rec);

        let name = match self.current_kind() {
            TokenKind::Ident(_) => self.ident()?,
            TokenKind::Underscore => {
                self.advance();
                "_".to_string()
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: self.current_span().start,
                    message: format!(
                        "expected identifier in let binding, got {:?}",
                        self.current_kind()
                    ),
                })
            }
        };

        // Parse optional function parameters (for `let f x y := ...` syntax)
        // These come before the optional return type
        let params = self.optional_binders()?;

        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.expr()?)
        } else {
            None
        };

        self.expect(&TokenKind::ColonEq)?;
        let mut val = self.expr()?;

        // If we have parameters, wrap the value in a lambda
        // let f x y := body  =>  let f := fun x y => body
        if !params.is_empty() {
            let val_span = val.span();
            val = SurfaceExpr::Lambda(val_span, params, Box::new(val));
        }

        // In Lean 4, consecutive let bindings can be chained without explicit `in`:
        // let x := 1
        // let y := 2  -- implicit body is the next let
        // x + y       -- final body
        //
        // NOTE: Without layout-sensitive parsing, we cannot reliably detect where
        // the value ends and the body begins for cases like:
        //   let y := 2
        //   x + y
        // Lean 4 uses indentation to know that `x + y` is the body, not `2 x + y`.
        // Our parser will consume `2 x + y` as the value since `x` looks like an argument.
        // For full compatibility, users should use explicit `in` or `;` separators.
        let body = if self.eat(&TokenKind::In) || self.eat(&TokenKind::Semicolon) {
            // Explicit separator
            self.expr()?
        } else if matches!(self.current_kind(), TokenKind::Let) {
            // Next let is implicitly the body
            let let_span = self.current_span();
            self.advance();
            self.let_body(let_span)?
        } else if self.is_implicit_body_start() {
            // Identifier or other expression that implicitly starts the body
            // This handles cases where the value doesn't extend to an identifier,
            // like after a pattern-matching lambda:
            //   let f := fun | 0 => 1 | n => n
            //   f 0  -- body starts here
            self.implicit_let_body_expr()?
        } else {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!(
                    "expected `in` or `;` after let binding, got {:?}",
                    self.current_kind()
                ),
            });
        };

        let span = start_span.merge(body.span());
        let binder = SurfaceBinder::new(name, ty, SurfaceBinderInfo::Explicit);

        if is_rec {
            Ok(SurfaceExpr::LetRec(
                span,
                binder,
                Box::new(val),
                Box::new(body),
            ))
        } else {
            Ok(SurfaceExpr::Let(
                span,
                binder,
                Box::new(val),
                Box::new(body),
            ))
        }
    }

    /// Parse if body: c then t else e
    /// Also handles:
    /// - `if let pat := e then t else f` (if-let pattern matching)
    /// - `if h : p then t else e` (decidable if with proof witness)
    fn if_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        // Check for `if let` pattern
        if self.eat(&TokenKind::Let) {
            let pat = self.pattern()?;
            self.expect(&TokenKind::ColonEq)?;
            let scrutinee = self.expr()?;
            self.expect(&TokenKind::Then)?;
            let then_branch = self.expr()?;
            self.expect(&TokenKind::Else)?;
            let else_branch = self.expr()?;
            let span = start_span.merge(else_branch.span());
            return Ok(SurfaceExpr::IfLet(
                span,
                pat,
                Box::new(scrutinee),
                Box::new(then_branch),
                Box::new(else_branch),
            ));
        }

        // Check for `if h : p` decidable if
        // This is `if ident : expr then ... else ...`
        // We need to look ahead: if we have `ident :` (not `:=`), it's decidable
        if let TokenKind::Ident(name) = self.current_kind() {
            if matches!(self.peek_kind(1), Some(TokenKind::Colon)) {
                // Check it's not ColonEq
                let name = name.clone();
                self.advance(); // consume ident
                self.advance(); // consume :
                let prop = self.expr()?;
                self.expect(&TokenKind::Then)?;
                let then_branch = self.expr()?;
                self.expect(&TokenKind::Else)?;
                let else_branch = self.expr()?;
                let span = start_span.merge(else_branch.span());
                return Ok(SurfaceExpr::IfDecidable(
                    span,
                    name,
                    Box::new(prop),
                    Box::new(then_branch),
                    Box::new(else_branch),
                ));
            }
        }

        // Regular if-then-else
        let cond = self.expr()?;
        self.expect(&TokenKind::Then)?;
        let then_branch = self.expr()?;
        self.expect(&TokenKind::Else)?;
        let else_branch = self.expr()?;
        let span = start_span.merge(else_branch.span());
        Ok(SurfaceExpr::If(
            span,
            Box::new(cond),
            Box::new(then_branch),
            Box::new(else_branch),
        ))
    }

    /// Parse by body: tactic proof
    /// For simplicity, we parse the tactic block as a hole for now
    fn by_body(&mut self, start_span: Span) -> SurfaceExpr {
        // Skip tactic block until we hit something that looks like the end
        // This is a simplification - full tactic parsing would be complex
        let mut depth = 0;
        let mut end_span = start_span;

        while !self.at_tactic_end(depth) {
            match self.current_kind() {
                TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => depth += 1,
                TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                    if depth > 0 {
                        depth -= 1;
                    } else {
                        break;
                    }
                }
                TokenKind::Eof => break,
                _ => {}
            }
            end_span = self.current_span();
            self.advance();
        }

        // Return sorry as placeholder for tactic proof
        SurfaceExpr::Ident(start_span.merge(end_span), "sorry".to_string())
    }

    /// Check if we're at the end of a tactic block
    fn at_tactic_end(&self, depth: usize) -> bool {
        if depth > 0 {
            return false;
        }
        // Treat hash commands (#check, #eval) as terminators only when they are not array literals
        if matches!(self.current_kind(), TokenKind::Hash)
            && !matches!(self.peek_kind(1), Some(TokenKind::LBracket))
        {
            return true;
        }
        matches!(
            self.current_kind(),
            TokenKind::Eof
                | TokenKind::Def
                | TokenKind::Theorem
                | TokenKind::Lemma
                | TokenKind::Example
                | TokenKind::Axiom
                | TokenKind::Inductive
                | TokenKind::Structure
                | TokenKind::Class
                | TokenKind::Instance
                | TokenKind::Namespace
                | TokenKind::Section
                | TokenKind::End
                | TokenKind::Import
                | TokenKind::Open
                | TokenKind::Variable
                | TokenKind::Universe
                | TokenKind::Mutual
                | TokenKind::At
                | TokenKind::Private
                | TokenKind::Protected
                | TokenKind::Partial
                | TokenKind::Unsafe
                | TokenKind::Noncomputable
                | TokenKind::Abbrev
                | TokenKind::Syntax
                | TokenKind::Macro
                | TokenKind::Elab
                | TokenKind::Notation
                | TokenKind::Infixl
                | TokenKind::Infixr
                | TokenKind::Prefix
                | TokenKind::Postfix
                | TokenKind::Scoped
                | TokenKind::SetOption
                | TokenKind::Comma
                | TokenKind::RAngle
                | TokenKind::RParen
                | TokenKind::RBracket
                | TokenKind::RBrace
                | TokenKind::With
                | TokenKind::Pipe
        )
    }

    /// Parse anonymous constructor: ⟨e1, e2, ...⟩
    fn anon_constructor_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let mut args = Vec::new();

        if !self.check(&TokenKind::RAngle) {
            args.push(self.expr()?);
            while self.eat(&TokenKind::Comma) {
                args.push(self.expr()?);
            }
        }

        self.expect(&TokenKind::RAngle)?;
        let end_span = self.current_span();

        // Create application: anonymousCtor args...
        Ok(SurfaceExpr::App(
            start_span.merge(end_span),
            Box::new(SurfaceExpr::Ident(start_span, "anonymousCtor".to_string())),
            args.into_iter().map(SurfaceArg::positional).collect(),
        ))
    }

    /// Parse match expression: match e with | pat => body | ...
    fn match_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let mut scrutinees = vec![self.expr()?];
        while self.eat(&TokenKind::Comma) {
            scrutinees.push(self.expr()?);
        }
        self.expect(&TokenKind::With)?;

        let scrutinee = scrutinees
            .into_iter()
            .reduce(|acc, expr| {
                let span = acc.span().merge(expr.span());
                SurfaceExpr::App(
                    span,
                    Box::new(SurfaceExpr::Ident(span, "Prod.mk".to_string())),
                    vec![SurfaceArg::positional(acc), SurfaceArg::positional(expr)],
                )
            })
            .expect("expected at least one scrutinee");

        let mut arms = Vec::new();
        while self.eat(&TokenKind::Pipe) {
            let mut patterns = vec![self.pattern_with_or()?];
            while self.eat(&TokenKind::Comma) {
                patterns.push(self.pattern_with_or()?);
            }
            self.expect(&TokenKind::FatArrow)?;
            let body = self.expr()?;
            // Combine multiple patterns into a tuple pattern
            let pattern = if patterns.len() == 1 {
                patterns.pop().expect("patterns is non-empty")
            } else {
                patterns
                    .into_iter()
                    .rev()
                    .reduce(|acc, pat| SurfacePattern::Ctor("Prod.mk".to_string(), vec![pat, acc]))
                    .expect("patterns.len() > 1 in else branch")
            };
            arms.push(SurfaceMatchArm {
                span: pattern.span(),
                pattern,
                body,
            });
        }

        let end_span = arms.last().map_or(start_span, |a| a.body.span());
        Ok(SurfaceExpr::Match(
            start_span.merge(end_span),
            Box::new(scrutinee),
            arms,
        ))
    }

    /// Parse a pattern with optional `+ k` suffix for numeral addition patterns
    /// Example: `n + 1` matches `Nat.succ` patterns
    fn pattern_with_addition(&mut self) -> Result<SurfacePattern, ParseError> {
        let mut pat = self.pattern()?;

        // Check for `+ k` suffix (numeral addition pattern)
        while self.check(&TokenKind::Plus) {
            self.advance();
            if let TokenKind::NatLit(k) = self.current_kind().clone() {
                self.advance();
                pat = SurfacePattern::NumeralAdd(Box::new(pat), k);
            } else {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: self.current_span().start,
                    message: format!(
                        "expected numeral after + in pattern, got {:?}",
                        self.current_kind()
                    ),
                });
            }
        }

        Ok(pat)
    }

    /// Parse a pattern with optional `::` cons operator (right associative)
    /// Example: `x :: xs` matches list cons patterns
    fn pattern_with_cons(&mut self) -> Result<SurfacePattern, ParseError> {
        let left = self.pattern_with_addition()?;

        if self.eat(&TokenKind::ColonColon) {
            let right = self.pattern_with_cons()?; // Right associative
            Ok(SurfacePattern::Ctor(
                "List.cons".to_string(),
                vec![left, right],
            ))
        } else {
            Ok(left)
        }
    }

    /// Parse a pattern with optional `|` or-pattern operator
    /// Example: `0 | 1` matches either 0 or 1
    /// Note: This is called at match arm level where `|` can also start a new arm,
    /// so we only consume `|` when followed by pattern-like tokens (not `=>`).
    fn pattern_with_or(&mut self) -> Result<SurfacePattern, ParseError> {
        let left = self.pattern_with_cons()?;

        // Check if `|` is followed by something that looks like a pattern
        // (not `=>` which would start a new arm's body or indicate end of patterns)
        if self.check(&TokenKind::Pipe) {
            // Peek ahead to see if this is an or-pattern or a new arm
            let next_is_fat_arrow = self
                .tokens
                .get(self.pos + 1)
                .is_some_and(|t| matches!(&t.kind, TokenKind::FatArrow));

            if !next_is_fat_arrow {
                self.advance(); // consume `|`
                let right = self.pattern_with_or()?; // Right associative
                return Ok(SurfacePattern::Or(Box::new(left), Box::new(right)));
            }
        }

        Ok(left)
    }

    /// Parse a pattern (simplified)
    fn pattern(&mut self) -> Result<SurfacePattern, ParseError> {
        match self.current_kind().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                // Handle dotted names like T.t, Option.none, etc.
                let mut full_name = name;
                while self.check(&TokenKind::Dot) {
                    // Peek at what follows the dot
                    let next_is_ident = self
                        .tokens
                        .get(self.pos + 1)
                        .is_some_and(|t| matches!(&t.kind, TokenKind::Ident(_)));
                    if next_is_ident {
                        self.advance(); // consume dot
                        if let TokenKind::Ident(next_name) = self.current_kind().clone() {
                            full_name.push('.');
                            full_name.push_str(&next_name);
                            self.advance(); // consume ident
                        }
                    } else {
                        break;
                    }
                }
                // Check for as-pattern: `n@pat`
                if self.eat(&TokenKind::At) {
                    let inner_pat = self.pattern()?;
                    return Ok(SurfacePattern::As(full_name, Box::new(inner_pat)));
                }
                // Check for constructor arguments
                let mut args = Vec::new();
                while self.is_pattern_arg_start() {
                    args.push(self.pattern()?);
                }
                if args.is_empty() {
                    Ok(SurfacePattern::Var(full_name))
                } else {
                    Ok(SurfacePattern::Ctor(full_name, args))
                }
            }
            TokenKind::NatLit(n) => {
                self.advance();
                Ok(SurfacePattern::Lit(SurfaceLit::Nat(n)))
            }
            TokenKind::SyntaxQuote(_) => {
                // Quoted patterns from macros - treat as wildcard
                self.advance();
                Ok(SurfacePattern::Wildcard)
            }
            TokenKind::Underscore => {
                self.advance();
                Ok(SurfacePattern::Wildcard)
            }
            TokenKind::Dot => {
                // Inaccessible pattern `.t` or `.(expr)` - patterns determined by unification
                self.advance();
                // Handle .(expr) - parenthesized inaccessible pattern
                if self.check(&TokenKind::LParen) {
                    self.advance();
                    // Skip the expression inside - we treat it as wildcard for now
                    let mut depth = 1;
                    while depth > 0 {
                        match self.current_kind() {
                            TokenKind::LParen => depth += 1,
                            TokenKind::RParen => depth -= 1,
                            TokenKind::Eof => break,
                            _ => {}
                        }
                        self.advance();
                    }
                    return Ok(SurfacePattern::Wildcard);
                }
                // Try to parse the following pattern, fallback to wildcard
                match self.pattern() {
                    Ok(pat) => Ok(pat),
                    Err(_) => Ok(SurfacePattern::Wildcard),
                }
            }
            TokenKind::LParen => {
                self.advance();
                // Handle empty tuple pattern ()
                if self.check(&TokenKind::RParen) {
                    self.advance();
                    return Ok(SurfacePattern::Ctor("Unit.unit".to_string(), vec![]));
                }
                let first = self.pattern_with_cons()?;
                // Check for tuple pattern (p1, p2, ...)
                if self.eat(&TokenKind::Comma) {
                    let mut pats = vec![first];
                    if !self.check(&TokenKind::RParen) {
                        pats.push(self.pattern_with_cons()?);
                        while self.eat(&TokenKind::Comma) {
                            if self.check(&TokenKind::RParen) {
                                break;
                            }
                            pats.push(self.pattern_with_cons()?);
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    // Build nested Prod.mk pattern
                    let result = pats
                        .into_iter()
                        .rev()
                        .reduce(|acc, pat| {
                            SurfacePattern::Ctor("Prod.mk".to_string(), vec![pat, acc])
                        })
                        .expect("tuple pattern must have elements");
                    Ok(result)
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::LBracket => {
                // List pattern: [] or [p1, p2, ...]
                self.advance();
                if self.check(&TokenKind::RBracket) {
                    self.advance();
                    // Empty list pattern: []
                    return Ok(SurfacePattern::Ctor("List.nil".to_string(), vec![]));
                }
                // Non-empty list pattern: [p1, p2, ...]
                let mut pats = vec![self.pattern_with_cons()?];
                while self.eat(&TokenKind::Comma) {
                    if self.check(&TokenKind::RBracket) {
                        break;
                    }
                    pats.push(self.pattern_with_cons()?);
                }
                self.expect(&TokenKind::RBracket)?;
                // Build List.cons chain: [a, b, c] => List.cons a (List.cons b (List.cons c List.nil))
                let result = pats.into_iter().rev().fold(
                    SurfacePattern::Ctor("List.nil".to_string(), vec![]),
                    |acc, pat| SurfacePattern::Ctor("List.cons".to_string(), vec![pat, acc]),
                );
                Ok(result)
            }
            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected pattern, got {:?}", self.current_kind()),
            }),
        }
    }

    fn is_pattern_arg_start(&self) -> bool {
        matches!(
            self.current_kind(),
            TokenKind::Ident(_)
                | TokenKind::NatLit(_)
                | TokenKind::Underscore
                | TokenKind::LParen
                | TokenKind::LBracket
        )
    }

    /// Parse do notation: do e1; e2; ...
    fn do_body(&mut self, start_span: Span) -> SurfaceExpr {
        // Simplified: skip the do block and return sorry
        // Full do notation parsing requires handling let, return, <-, etc.
        let mut depth = 0;
        let mut end_span = start_span;

        while !self.at_tactic_end(depth) {
            match self.current_kind() {
                TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => depth += 1,
                TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                    if depth > 0 {
                        depth -= 1;
                    } else {
                        break;
                    }
                }
                TokenKind::Eof => break,
                _ => {}
            }
            end_span = self.current_span();
            self.advance();
        }

        SurfaceExpr::Ident(start_span.merge(end_span), "sorry".to_string())
    }

    /// Parse exists: ∃ x, P x
    fn exists_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let binders = self.binders()?;
        self.expect(&TokenKind::Comma)?;
        let body = self.expr()?;

        // Build nested Exists applications
        let mut result = body;
        for binder in binders.into_iter().rev() {
            let span = start_span.merge(result.span());
            let ty = binder.ty.map_or_else(|| SurfaceExpr::Hole(span), |t| *t);
            result = SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "Exists".to_string())),
                vec![
                    SurfaceArg::positional(ty),
                    SurfaceArg::positional(SurfaceExpr::Lambda(
                        span,
                        vec![SurfaceBinder {
                            span: binder.span,
                            name: binder.name,
                            ty: None,
                            default: None,
                            info: binder.info,
                        }],
                        Box::new(result),
                    )),
                ],
            );
        }

        Ok(result)
    }

    /// Parse show ... from ... expression
    fn show_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let ty = self.expr()?;
        self.expect(&TokenKind::From)?;
        let expr = self.expr()?;
        let span = start_span.merge(expr.span());
        Ok(SurfaceExpr::Ascription(span, Box::new(expr), Box::new(ty)))
    }

    /// Parse have expression in term position: `have h : P := proof; body`
    /// Equivalent to a let binding but for proof terms
    fn have_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        // Parse name (optional, could start with `:` for anonymous)
        let name = if matches!(self.current_kind(), TokenKind::Ident(_)) {
            self.ident()?
        } else {
            "_h".to_string()
        };

        // Parse optional type annotation
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.expr()?)
        } else {
            None
        };

        // Parse value
        self.expect(&TokenKind::ColonEq)?;
        let val = self.expr()?;

        // Parse body after semicolon
        self.expect(&TokenKind::Semicolon)?;
        let body = self.expr()?;

        let span = start_span.merge(body.span());
        let binder = SurfaceBinder::new(name, ty, SurfaceBinderInfo::Explicit);
        Ok(SurfaceExpr::Let(
            span,
            binder,
            Box::new(val),
            Box::new(body),
        ))
    }

    /// Parse suffices expression in term position: `suffices h : P by tactic; body`
    /// States that it suffices to prove P, then provides the proof
    fn suffices_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        // Parse name (optional)
        let name = if matches!(self.current_kind(), TokenKind::Ident(_))
            && !matches!(self.current_kind(), TokenKind::By)
        {
            self.ident()?
        } else {
            "_h".to_string()
        };

        // Parse type annotation (required for suffices)
        // Stop at `by` since that starts the tactic block
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.atom_expr()?)
        } else {
            None
        };

        // Parse `by tactic` for showing how the goal follows from suffices
        // The tactic runs until `;` which separates it from the body
        let tactic = if self.eat(&TokenKind::By) {
            let tac_span = self.current_span();
            // Skip tokens until semicolon (the tactic block)
            let mut depth = 0;
            let mut end_span = tac_span;
            while !self.check(&TokenKind::Semicolon) && !self.check(&TokenKind::Eof) {
                match self.current_kind() {
                    TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => depth += 1,
                    TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                        if depth > 0 {
                            depth -= 1;
                        }
                    }
                    _ => {}
                }
                end_span = self.current_span();
                self.advance();
            }
            Some(Box::new(SurfaceExpr::Ident(
                tac_span.merge(end_span),
                "sorry".to_string(),
            )))
        } else {
            None
        };

        // Parse body after semicolon (the proof of P)
        self.expect(&TokenKind::Semicolon)?;
        let body = self.expr()?;

        let span = start_span.merge(body.span());
        let binder = SurfaceBinder::new(name, ty, SurfaceBinderInfo::Explicit);

        // Represent as a let with tactic application
        // suffices h : P by tac; proof ≈ let h := proof; tac
        if let Some(tac) = tactic {
            // Create the proof binding
            Ok(SurfaceExpr::Let(span, binder, Box::new(body), tac))
        } else {
            // Without tactic, just a simple let
            Ok(SurfaceExpr::Let(
                span,
                binder,
                Box::new(body),
                Box::new(SurfaceExpr::Ident(span, "_".to_string())),
            ))
        }
    }

    /// Parse record literal `{ field := value, ... }`
    fn record_literal_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let mut depth = 1;
        let mut end_span = start_span;

        while depth > 0 {
            match self.current_kind() {
                TokenKind::LBrace => {
                    depth += 1;
                    end_span = self.current_span();
                    self.advance();
                }
                TokenKind::RBrace => {
                    depth -= 1;
                    end_span = self.current_span();
                    self.advance();
                }
                TokenKind::Eof => {
                    // Unclosed brace - return an error
                    return Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: self.current_span().start,
                        message: "unclosed brace, expected '}'".to_string(),
                    });
                }
                _ => {
                    end_span = self.current_span();
                    self.advance();
                }
            }
        }

        Ok(SurfaceExpr::Hole(start_span.merge(end_span)))
    }

    /// Parse `open Foo in expr` expression form (scoping only)
    fn open_expr_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        // Consume one or more module paths (with optional selective names)
        loop {
            if !matches!(self.current_kind(), TokenKind::Ident(_)) {
                break;
            }
            let _ = self.module_path()?;

            if self.eat(&TokenKind::LParen) {
                while let TokenKind::Ident(_) = self.current_kind() {
                    self.advance();
                }
                self.expect(&TokenKind::RParen)?;
            }

            if self.check(&TokenKind::In) || !matches!(self.current_kind(), TokenKind::Ident(_)) {
                break;
            }
        }

        self.expect(&TokenKind::In)?;
        let body = self.expr()?;
        let span = start_span.merge(body.span());

        // Represent as a simple wrapper application to keep AST stable
        Ok(SurfaceExpr::App(
            span,
            Box::new(SurfaceExpr::Ident(start_span, "open".to_string())),
            vec![SurfaceArg::positional(body)],
        ))
    }

    /// Parse list literal: [a, b, c]
    fn list_literal_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        // Empty list
        if self.eat(&TokenKind::RBracket) {
            return Ok(SurfaceExpr::Ident(start_span, "List.nil".to_string()));
        }

        let mut elems = Vec::new();
        elems.push(self.expr()?);
        while self.eat(&TokenKind::Comma) {
            if self.check(&TokenKind::RBracket) {
                break;
            }
            elems.push(self.expr()?);
        }
        let end_span = self.expect(&TokenKind::RBracket)?.span;

        // Build List.cons chain ending with List.nil
        let mut result = SurfaceExpr::Ident(start_span.merge(end_span), "List.nil".to_string());
        for elem in elems.into_iter().rev() {
            let span = start_span.merge(elem.span()).merge(result.span());
            result = SurfaceExpr::App(
                span,
                Box::new(SurfaceExpr::Ident(span, "List.cons".to_string())),
                vec![SurfaceArg::positional(elem), SurfaceArg::positional(result)],
            );
        }

        Ok(result)
    }

    /// Parse array literal: #[a, b, c]
    fn array_literal_body(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let mut elems = Vec::new();

        if !self.check(&TokenKind::RBracket) {
            elems.push(self.expr()?);
            while self.eat(&TokenKind::Comma) {
                if self.check(&TokenKind::RBracket) {
                    break;
                }
                elems.push(self.expr()?);
            }
        }

        let end_span = self.expect(&TokenKind::RBracket)?.span;
        let span = start_span.merge(end_span);
        Ok(SurfaceExpr::App(
            span,
            Box::new(SurfaceExpr::Ident(span, "Array.mk".to_string())),
            elems.into_iter().map(SurfaceArg::positional).collect(),
        ))
    }

    /// Parse `set_option` ... in expr expression form
    fn set_option_expr(&mut self, start_span: Span) -> Result<SurfaceExpr, ParseError> {
        let name = self.qualified_ident()?;

        let value = if self.check(&TokenKind::In) {
            None
        } else {
            Some(self.expr()?)
        };

        self.expect(&TokenKind::In)?;
        let body = self.expr()?;

        let name_expr = SurfaceExpr::Ident(start_span, name);
        let value_expr = value.unwrap_or(SurfaceExpr::Hole(start_span));
        let span = start_span.merge(body.span());
        Ok(SurfaceExpr::App(
            span,
            Box::new(SurfaceExpr::Ident(start_span, "set_option".to_string())),
            vec![
                SurfaceArg::positional(name_expr),
                SurfaceArg::positional(value_expr),
                SurfaceArg::positional(body),
            ],
        ))
    }

    /// Parse one or more binders
    fn binders(&mut self) -> Result<Vec<SurfaceBinder>, ParseError> {
        let mut binders = Vec::new();

        loop {
            match self.current_kind() {
                TokenKind::LParen => {
                    binders.extend(self.explicit_binders()?);
                }
                TokenKind::LBrace => {
                    binders.extend(self.implicit_binders()?);
                }
                TokenKind::LBracket => {
                    binders.extend(self.instance_binders()?);
                }
                TokenKind::Ident(name) => {
                    // Simple binder: could be bare name or `name : type`
                    let span = self.current_span();
                    let name = name.clone();
                    self.advance();

                    // Check for type annotation: `a : Nat` without parentheses
                    let ty = if self.check(&TokenKind::Colon) {
                        self.advance();
                        Some(Box::new(self.app_expr()?))
                    } else {
                        None
                    };

                    binders.push(SurfaceBinder {
                        span,
                        name,
                        ty,
                        default: None,
                        info: SurfaceBinderInfo::Explicit,
                    });
                }
                TokenKind::Underscore => {
                    // Anonymous binder: fun _ => ... or _ : T
                    let span = self.current_span();
                    self.advance();

                    // Check for type annotation: `_ : Nat`
                    let ty = if self.check(&TokenKind::Colon) {
                        self.advance();
                        Some(Box::new(self.app_expr()?))
                    } else {
                        None
                    };

                    binders.push(SurfaceBinder {
                        span,
                        name: "_".to_string(),
                        ty,
                        default: None,
                        info: SurfaceBinderInfo::Explicit,
                    });
                }
                _ => break,
            }
        }

        if binders.is_empty() {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: "expected at least one binder".to_string(),
            });
        }

        Ok(binders)
    }

    /// Parse explicit binders: (x y z : T) or (x y z) without type
    /// Also supports underscore binders: (_ : T)
    fn explicit_binders(&mut self) -> Result<Vec<SurfaceBinder>, ParseError> {
        self.expect(&TokenKind::LParen)?;

        let mut names = Vec::new();
        loop {
            match self.current_kind() {
                TokenKind::Ident(name) => {
                    names.push((self.current_span(), name.clone()));
                    self.advance();
                }
                TokenKind::Underscore => {
                    names.push((self.current_span(), "_".to_string()));
                    self.advance();
                }
                _ => break,
            }
        }

        if names.is_empty() {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: "expected identifier in binder".to_string(),
            });
        }

        // Type annotation is optional: `(x)` is valid
        let ty = if self.eat(&TokenKind::Colon) {
            Some(Box::new(self.expr()?))
        } else {
            None
        };

        // Default value is optional: `(x := 5)` or `(x : Nat := 5)`
        let default = if self.eat(&TokenKind::ColonEq) {
            Some(Box::new(self.expr()?))
        } else {
            None
        };

        self.expect(&TokenKind::RParen)?;

        Ok(names
            .into_iter()
            .map(|(s, name)| SurfaceBinder {
                span: s,
                name,
                ty: ty.clone(),
                default: default.clone(),
                info: SurfaceBinderInfo::Explicit,
            })
            .collect())
    }

    /// Parse implicit binders: {x y z : T} or strict implicit: {{x y z : T}}
    /// Also supports underscore binders: {_ : T}
    fn implicit_binders(&mut self) -> Result<Vec<SurfaceBinder>, ParseError> {
        self.expect(&TokenKind::LBrace)?;

        // Check for strict implicit: {{...}}
        let is_strict = self.eat(&TokenKind::LBrace);

        let mut names = Vec::new();
        loop {
            match self.current_kind() {
                TokenKind::Ident(name) => {
                    names.push((self.current_span(), name.clone()));
                    self.advance();
                }
                TokenKind::Underscore => {
                    names.push((self.current_span(), "_".to_string()));
                    self.advance();
                }
                _ => break,
            }
        }

        if names.is_empty() {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: "expected identifier in binder".to_string(),
            });
        }

        // Check for type annotation (colon) or just close brace
        let ty = if self.eat(&TokenKind::Colon) {
            Some(Box::new(self.expr()?))
        } else {
            None // Implicit binders without explicit type: {α β}
        };
        self.expect(&TokenKind::RBrace)?;

        // For strict implicit, expect closing }}
        if is_strict {
            self.expect(&TokenKind::RBrace)?;
        }

        let binder_info = if is_strict {
            SurfaceBinderInfo::StrictImplicit
        } else {
            SurfaceBinderInfo::Implicit
        };

        Ok(names
            .into_iter()
            .map(|(s, name)| SurfaceBinder {
                span: s,
                name,
                ty: ty.clone(),
                default: None,
                info: binder_info,
            })
            .collect())
    }

    /// Parse instance binders: `[x : T]` or anonymous `[T]` or `[Ord A]`
    fn instance_binders(&mut self) -> Result<Vec<SurfaceBinder>, ParseError> {
        self.expect(&TokenKind::LBracket)?;

        let mut names = Vec::new();
        while let TokenKind::Ident(name) = self.current_kind() {
            names.push((self.current_span(), name.clone()));
            self.advance();
        }

        // Instance binders can be anonymous: [Ord A] or [T]
        // If we see a colon, then names are actual binder names
        // Otherwise, the collected names are the type expression
        let (names, ty) = if names.is_empty() {
            // Nothing collected - expression like [_] or [(A)]?
            let ty_expr = self.expr()?;
            self.expect(&TokenKind::RBracket)?;
            return Ok(vec![SurfaceBinder {
                span: Span::dummy(),
                name: "_".to_string(),
                ty: Some(Box::new(ty_expr)),
                default: None,
                info: SurfaceBinderInfo::Instance,
            }]);
        } else if self.check(&TokenKind::Colon) {
            // Named binder: [x : T]
            self.expect(&TokenKind::Colon)?;
            let ty = self.expr()?;
            (names, ty)
        } else {
            // Anonymous instance: names are actually the type expression
            // e.g., [Add α] where "Add" and "α" were collected as names
            // Build application: Add α β ...
            let mut result = SurfaceExpr::Ident(names[0].0, names[0].1.clone());
            for (span, name) in names.iter().skip(1) {
                let arg = SurfaceExpr::Ident(*span, name.clone());
                let app_span = result.span().merge(arg.span());
                result = SurfaceExpr::App(
                    app_span,
                    Box::new(result),
                    vec![SurfaceArg::positional(arg)],
                );
            }

            // Parse remaining arguments until closing bracket (e.g., `[OfNat R 0]`)
            while !self.check(&TokenKind::RBracket) {
                let arg = self.atom_expr()?;
                let app_span = result.span().merge(arg.span());
                result = SurfaceExpr::App(
                    app_span,
                    Box::new(result),
                    vec![SurfaceArg::positional(arg)],
                );
            }

            self.expect(&TokenKind::RBracket)?;
            return Ok(vec![SurfaceBinder {
                span: names.first().map_or_else(Span::dummy, |(s, _)| *s),
                name: "_".to_string(),
                ty: Some(Box::new(result)),
                default: None,
                info: SurfaceBinderInfo::Instance,
            }]);
        };

        self.expect(&TokenKind::RBracket)?;

        Ok(names
            .into_iter()
            .map(|(s, name)| SurfaceBinder {
                span: s,
                name,
                ty: Some(Box::new(ty.clone())),
                default: None,
                info: SurfaceBinderInfo::Instance,
            })
            .collect())
    }

    /// Parse an identifier
    fn ident(&mut self) -> Result<String, ParseError> {
        match self.current_kind() {
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            TokenKind::Error(msg) if msg.contains("unexpected character") => {
                // Allow invalid characters to be treated as placeholder identifiers
                // for error recovery in malformed test files
                self.advance();
                Ok("_invalid_".to_string())
            }
            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected identifier, got {:?}", self.current_kind()),
            }),
        }
    }

    /// Parse a dotted identifier like `Foo.bar.baz`
    fn qualified_ident(&mut self) -> Result<String, ParseError> {
        match self.current_kind() {
            TokenKind::Ident(name) => {
                let mut full_name = name.clone();
                self.advance();
                while self.eat(&TokenKind::Dot) {
                    match self.current_kind() {
                        TokenKind::Ident(part) => {
                            full_name.push('.');
                            full_name.push_str(part);
                            self.advance();
                        }
                        other => {
                            return Err(ParseError::UnexpectedToken {
                                line: 0,
                                col: self.current_span().start,
                                message: format!("expected identifier after '.', got {other:?}"),
                            });
                        }
                    }
                }
                Ok(full_name)
            }
            TokenKind::Error(msg) if msg.contains("unexpected character") => {
                // Allow invalid characters to be treated as placeholder identifiers
                // for error recovery in malformed test files
                self.advance();
                Ok("_invalid_".to_string())
            }
            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected identifier, got {:?}", self.current_kind()),
            }),
        }
    }

    /// Parse a dotted module path into segments
    fn module_path(&mut self) -> Result<Vec<String>, ParseError> {
        let mut path = Vec::new();
        let name = self.qualified_ident()?;
        path.extend(name.split('.').map(ToString::to_string));
        Ok(path)
    }

    /// Parse a level expression
    /// Handles: numeric literals, identifiers (params), max, imax, +N suffix, and parenthesized levels
    fn level_expr(&mut self) -> Result<LevelExpr, ParseError> {
        let base = self.level_atom()?;

        // Check for +N suffix on the result
        if self.eat(&TokenKind::Plus) {
            if let TokenKind::NatLit(n) = self.current_kind() {
                let n = *n;
                self.advance();
                let mut result = base;
                for _ in 0..n {
                    result = LevelExpr::Succ(Box::new(result));
                }
                return Ok(result);
            }
        }
        Ok(base)
    }

    /// Parse a level atom (the base of a level expression without +N suffix)
    fn level_atom(&mut self) -> Result<LevelExpr, ParseError> {
        match self.current_kind() {
            TokenKind::NatLit(n) => {
                let level = u32::try_from(*n).map_err(|_| ParseError::NumericOverflow {
                    value: *n,
                    max: u64::from(u32::MAX),
                })?;
                self.advance();
                Ok(LevelExpr::Lit(level))
            }
            TokenKind::LParen => {
                // Parenthesized level expression: (max u v), (imax 1 u + 1), etc.
                self.advance();
                let inner = self.level_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(inner)
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                // Could be "max" or "imax" special forms
                if name == "max" {
                    let l1 = self.level_expr()?;
                    let l2 = self.level_expr()?;
                    Ok(LevelExpr::Max(Box::new(l1), Box::new(l2)))
                } else if name == "imax" {
                    let l1 = self.level_expr()?;
                    let l2 = self.level_expr()?;
                    Ok(LevelExpr::IMax(Box::new(l1), Box::new(l2)))
                } else {
                    Ok(LevelExpr::Param(name))
                }
            }
            TokenKind::Underscore => {
                // Treat universe hole "_" as a level parameter placeholder
                self.advance();
                Ok(LevelExpr::Param("_".to_string()))
            }
            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected level expression, got {:?}", self.current_kind()),
            }),
        }
    }

    /// Parse attributes: `@[attr1, attr2]` or `@[attr1] @[attr2]`
    ///
    /// Supported attributes:
    /// - `instance N` - set instance priority to N
    /// - `defaultInstance` - set instance to lowest priority (0)
    fn attributes(&mut self) -> Result<Vec<Attribute>, ParseError> {
        let mut attrs = Vec::new();

        while self.eat(&TokenKind::At) {
            self.expect(&TokenKind::LBracket)?;

            // Parse attributes inside brackets, separated by commas
            loop {
                let attr = self.single_attribute()?;
                attrs.push(attr);

                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }

            self.expect(&TokenKind::RBracket)?;
        }

        Ok(attrs)
    }

    /// Parse a single attribute name (and optional argument)
    fn single_attribute(&mut self) -> Result<Attribute, ParseError> {
        match self.current_kind().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                match name.as_str() {
                    "defaultInstance" | "default_instance" => {
                        self.skip_attribute_args();
                        Ok(Attribute::DefaultInstance)
                    }
                    _ => {
                        // Consume optional attribute parameters
                        // Attributes can have multi-token arguments like:
                        // @[local command_elab Lean.Parser.Command.end]
                        // @[instance 50]
                        // @[simp, reducible]
                        self.skip_attribute_args();
                        Ok(Attribute::Unknown(name))
                    }
                }
            }
            TokenKind::Minus => {
                // Attribute removal syntax: [-attr] or [-instance]
                self.advance();
                // Skip the attribute name (can be identifier or keyword like `instance`)
                if matches!(
                    self.current_kind(),
                    TokenKind::Ident(_) | TokenKind::Instance
                ) {
                    self.advance();
                }
                Ok(Attribute::Unknown("-".to_string()))
            }
            // Handle `instance` keyword used as attribute name
            TokenKind::Instance => {
                self.advance();
                // Check for optional priority number
                if let TokenKind::NatLit(n) = self.current_kind().clone() {
                    let priority = u32::try_from(n).map_err(|_| ParseError::NumericOverflow {
                        value: n,
                        max: u64::from(u32::MAX),
                    })?;
                    self.advance();
                    Ok(Attribute::InstancePriority(priority))
                } else {
                    // Just @[instance] without priority means default
                    Ok(Attribute::InstancePriority(100))
                }
            }
            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected attribute name, got {:?}", self.current_kind()),
            }),
        }
    }

    /// Skip attribute arguments until we hit `,` or `]`
    /// Handles multi-token attribute arguments like:
    /// - `@[local command_elab Lean.Parser.Command.end]`
    /// - `@[scoped elab_rules : command]`
    fn skip_attribute_args(&mut self) {
        // Track nested brackets to handle things like `@[foo (expr)]`
        let mut bracket_depth = 0;
        let mut paren_depth = 0;

        while !matches!(self.current_kind(), TokenKind::Eof) {
            match self.current_kind() {
                // Stop at comma or closing bracket (at depth 0)
                TokenKind::Comma | TokenKind::RBracket
                    if bracket_depth == 0 && paren_depth == 0 =>
                {
                    break
                }
                // Track nested structures
                TokenKind::LBracket => {
                    bracket_depth += 1;
                    self.advance();
                }
                TokenKind::RBracket => {
                    bracket_depth -= 1;
                    self.advance();
                }
                TokenKind::LParen => {
                    paren_depth += 1;
                    self.advance();
                }
                TokenKind::RParen => {
                    paren_depth -= 1;
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Parse a declaration
    fn decl(&mut self) -> Result<SurfaceDecl, ParseError> {
        // Parse leading attributes
        let attrs = self.attributes()?;

        let span = self.current_span();

        match self.current_kind() {
            TokenKind::Def => {
                self.advance();
                self.def_decl(span)
            }
            TokenKind::Theorem | TokenKind::Lemma => {
                self.advance();
                self.theorem_decl(span)
            }
            TokenKind::Axiom => {
                self.advance();
                self.axiom_decl(span)
            }
            TokenKind::Inductive => {
                self.advance();
                self.inductive_decl(span)
            }
            TokenKind::Structure => {
                self.advance();
                self.structure_decl(span)
            }
            TokenKind::Class => {
                self.advance();
                self.class_decl(span)
            }
            TokenKind::Instance => {
                self.advance();
                self.instance_decl(span, &attrs)
            }
            TokenKind::Example => {
                self.advance();
                self.example_decl(span)
            }
            TokenKind::Import => {
                self.advance();
                self.import_decl(span)
            }
            TokenKind::Namespace => {
                self.advance();
                self.namespace_decl(span)
            }
            TokenKind::Section => {
                self.advance();
                self.section_decl(span)
            }
            TokenKind::Universe => {
                self.advance();
                self.universe_decl(span)
            }
            TokenKind::Variable => {
                self.advance();
                self.variable_decl(span)
            }
            TokenKind::Open => {
                self.advance();
                self.open_decl(span)
            }
            TokenKind::Mutual => {
                self.advance();
                self.mutual_decl(span)
            }
            TokenKind::Hash => {
                self.advance();
                self.hash_command(span)
            }
            // Handle modifiers (private, protected, partial, etc.)
            TokenKind::Private
            | TokenKind::Protected
            | TokenKind::Partial
            | TokenKind::Unsafe
            | TokenKind::Noncomputable => {
                // Skip modifier and parse the actual declaration
                self.advance();
                self.decl()
            }
            TokenKind::Abbrev => {
                self.advance();
                self.abbrev_decl(span)
            }
            TokenKind::Attribute => {
                self.advance();
                self.attribute_cmd(span)
            }
            TokenKind::SetOption => {
                self.advance();
                self.set_option_cmd(span)
            }
            // Macro system declarations
            TokenKind::Syntax => {
                self.advance();
                self.syntax_decl(span)
            }
            TokenKind::Macro => {
                self.advance();
                self.macro_decl(span)
            }
            TokenKind::MacroRules => {
                self.advance();
                self.macro_rules_decl(span)
            }
            TokenKind::Elab => {
                self.advance();
                Ok(self.elab_decl(span))
            }
            TokenKind::Notation => {
                self.advance();
                self.notation_decl(span, NotationKind::Notation)
            }
            TokenKind::Infixl => {
                self.advance();
                self.notation_decl(span, NotationKind::Infixl)
            }
            TokenKind::Infixr => {
                self.advance();
                self.notation_decl(span, NotationKind::Infixr)
            }
            TokenKind::Prefix => {
                self.advance();
                self.notation_decl(span, NotationKind::Prefix)
            }
            TokenKind::Postfix => {
                self.advance();
                self.notation_decl(span, NotationKind::Postfix)
            }
            // Handle scoped modifier followed by other things
            TokenKind::Scoped => {
                self.advance();
                self.decl() // Parse the scoped declaration
            }
            // Handle declare_syntax_cat as a proper command
            TokenKind::Ident(name) if name == "declare_syntax_cat" => {
                self.advance();
                self.declare_syntax_cat_decl(span)
            }
            // Skip 'local' modifier and continue with declaration
            TokenKind::Ident(name) if name == "local" => {
                self.advance();
                self.decl() // Parse the local declaration
            }
            _ => {
                let raw = format!("{:?}", self.current_kind());
                Ok(self.skip_to_next_decl(&raw, span))
            }
        }
    }

    fn def_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;

        // Optional universe params
        let universe_params = self.universe_params()?;

        // Optional binders
        let binders = self.optional_binders()?;

        // Optional type annotation
        let ty = if self.eat(&TokenKind::Colon) {
            Some(Box::new(self.expr()?))
        } else {
            None
        };

        // Definition value can be provided with:
        // - := expr
        // - | pattern => expr (pattern matching)
        // - where | pattern => expr (where clause syntax)
        let val = if self.eat(&TokenKind::ColonEq) {
            self.expr()?
        } else if self.check(&TokenKind::Pipe) {
            // Pattern matching definition
            self.def_match_body(start_span)
        } else if self.eat(&TokenKind::Where) {
            // Where clause: def foo : Nat → Nat where | 0 => 1 | n => n
            self.def_match_body(start_span)
        } else {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected := or |, got {:?}", self.current_kind()),
            });
        };

        // Skip optional termination_by and decreasing_by clauses
        self.skip_termination_hints();

        Ok(SurfaceDecl::Def {
            span: start_span,
            name,
            universe_params,
            binders,
            ty,
            val: Box::new(val),
        })
    }

    /// Skip `termination_by` and `decreasing_by` clauses that follow function definitions
    fn skip_termination_hints(&mut self) {
        // These clauses appear after recursive function definitions
        // termination_by args => expr
        // decreasing_by tactic
        loop {
            if let TokenKind::Ident(name) = self.current_kind() {
                if name == "termination_by" {
                    self.advance();
                    // Skip until we see a => and then the expression
                    while !matches!(self.current_kind(), TokenKind::FatArrow | TokenKind::Eof) {
                        self.advance();
                    }
                    if self.eat(&TokenKind::FatArrow) {
                        // Skip the termination measure expression
                        // Be careful not to consume the next declaration
                        let mut depth = 0;
                        while !self.at_termination_hint_end(depth) {
                            match self.current_kind() {
                                TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => {
                                    depth += 1;
                                }
                                TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                                    depth = depth.saturating_sub(1);
                                }
                                _ => {}
                            }
                            self.advance();
                        }
                    }
                    continue;
                } else if name == "decreasing_by" {
                    self.advance();
                    // Skip the tactic block - just skip until next declaration or termination_by
                    let mut depth = 0;
                    while !self.at_termination_hint_end(depth) {
                        match self.current_kind() {
                            TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => {
                                depth += 1;
                            }
                            TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                                depth = depth.saturating_sub(1);
                            }
                            _ => {}
                        }
                        self.advance();
                    }
                    continue;
                }
            }
            break;
        }
    }

    /// Check if we're at the end of a termination hint expression
    fn at_termination_hint_end(&self, depth: usize) -> bool {
        if depth > 0 {
            return false;
        }
        // Check for next termination hint or declaration start
        if let TokenKind::Ident(name) = self.current_kind() {
            if name == "termination_by" || name == "decreasing_by" {
                return true;
            }
        }
        self.is_decl_start() || matches!(self.current_kind(), TokenKind::Eof | TokenKind::End)
    }

    /// Parse a definition body using pattern matching syntax
    fn def_match_body(&mut self, start_span: Span) -> SurfaceExpr {
        // Skip the match arms and return sorry as placeholder
        // Full implementation would parse: | pat1, pat2 => body | pat3, pat4 => body2
        while self.eat(&TokenKind::Pipe) {
            // Skip pattern(s)
            while !matches!(self.current_kind(), TokenKind::FatArrow | TokenKind::Eof) {
                self.advance();
            }
            // Skip =>
            if self.eat(&TokenKind::FatArrow) {
                // Skip body until next | or declaration start
                let mut depth = 0;
                while !self.at_def_match_end(depth) {
                    match self.current_kind() {
                        TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => depth += 1,
                        TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                            depth = depth.saturating_sub(1);
                        }
                        _ => {}
                    }
                    self.advance();
                }
            }
        }
        SurfaceExpr::Ident(start_span, "sorry".to_string())
    }

    /// Check if we're at the end of a pattern match arm
    fn at_def_match_end(&self, depth: usize) -> bool {
        if depth > 0 {
            return false;
        }
        matches!(self.current_kind(), TokenKind::Pipe | TokenKind::Eof) || self.is_decl_start()
    }

    fn theorem_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;
        let universe_params = self.universe_params()?;
        let binders = self.optional_binders()?;

        self.expect(&TokenKind::Colon)?;
        let ty = self.expr()?;

        // Proof can be provided with:
        // - := expr
        // - | pattern => expr (pattern matching without :=)
        let proof = if self.eat(&TokenKind::ColonEq) {
            self.expr()?
        } else if self.check(&TokenKind::Pipe) {
            // Pattern matching theorem: theorem foo : T | p1 => e1 | p2 => e2
            self.def_match_body(start_span)
        } else {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected := or |, got {:?}", self.current_kind()),
            });
        };

        Ok(SurfaceDecl::Theorem {
            span: start_span,
            name,
            universe_params,
            binders,
            ty: Box::new(ty),
            proof: Box::new(proof),
        })
    }

    fn axiom_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;
        let universe_params = self.universe_params()?;
        let binders = self.optional_binders()?;

        self.expect(&TokenKind::Colon)?;
        let ty = self.expr()?;

        Ok(SurfaceDecl::Axiom {
            span: start_span,
            name,
            universe_params,
            binders,
            ty: Box::new(ty),
        })
    }

    fn inductive_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;
        let universe_params = self.universe_params()?;
        let binders = self.optional_binders()?;

        // Type annotation is optional in some cases
        let ty = if self.eat(&TokenKind::Colon) {
            self.expr()?
        } else {
            // Default to Type
            SurfaceExpr::Universe(start_span, UniverseExpr::Type)
        };

        // Parse constructors - two syntaxes supported:
        // 1. where-style: `inductive Foo where | ctor1 | ctor2`
        // 2. pipe-style:  `inductive Foo : Type | ctor1 : ... | ctor2 : ...`
        let mut ctors = Vec::new();

        // Check for `where` syntax
        if self.eat(&TokenKind::Where) {
            // Parse constructors after where
            while self.eat(&TokenKind::Pipe) || self.is_ctor_start() {
                let ctor_span = self.current_span();
                let ctor_name = self.ident()?;

                // Constructor type is optional in where-style
                let ctor_ty = if self.eat(&TokenKind::Colon) {
                    self.expr()?
                } else {
                    // Default: just the inductive type name
                    SurfaceExpr::Ident(ctor_span, name.clone())
                };

                ctors.push(SurfaceCtor {
                    span: ctor_span,
                    name: ctor_name,
                    ty: ctor_ty,
                });
            }
        } else {
            // Pipe-style constructors
            while self.eat(&TokenKind::Pipe) {
                let ctor_span = self.current_span();
                let ctor_name = self.ident()?;

                // Constructor type is optional
                let ctor_ty = if self.eat(&TokenKind::Colon) {
                    self.expr()?
                } else {
                    // Default: just the inductive type name
                    SurfaceExpr::Ident(ctor_span, name.clone())
                };

                ctors.push(SurfaceCtor {
                    span: ctor_span,
                    name: ctor_name,
                    ty: ctor_ty,
                });
            }
        }

        // Parse optional deriving clause: `deriving Repr, BEq`
        let deriving = self.parse_deriving_clause()?;

        Ok(SurfaceDecl::Inductive {
            span: start_span,
            name,
            universe_params,
            binders,
            ty: Box::new(ty),
            ctors,
            deriving,
        })
    }

    /// Check if current token starts a constructor definition
    fn is_ctor_start(&self) -> bool {
        matches!(self.current_kind(), TokenKind::Ident(_))
    }

    /// Parse a structure declaration
    ///
    /// ```text
    /// structure Point where
    ///   x : Nat
    ///   y : Nat
    /// ```
    ///
    /// Or with parameters:
    /// ```text
    /// structure Pair (A : Type) (B : Type) where
    ///   fst : A
    ///   snd : B
    /// ```
    fn structure_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;
        let universe_params = self.universe_params()?;
        let binders = self.optional_binders()?;

        // Optional extends clause: `structure Foo extends Bar, Baz`
        if self.eat(&TokenKind::Extends) {
            // Skip the parent types for now (simplified handling)
            while !matches!(
                self.current_kind(),
                TokenKind::Where | TokenKind::Colon | TokenKind::Eof
            ) {
                self.advance();
            }
        }

        // Optional explicit result type: `structure Foo : Type 1 where`
        let ty = if self.eat(&TokenKind::Colon) {
            Some(Box::new(self.expr()?))
        } else {
            None
        };

        // Handle case where structure has no fields (just extends)
        if !self.check(&TokenKind::Where) {
            return Ok(SurfaceDecl::Structure {
                span: start_span,
                name,
                universe_params,
                binders,
                ty,
                fields: Vec::new(),
                deriving: Vec::new(),
            });
        }

        self.expect(&TokenKind::Where)?;

        // Parse fields
        let mut fields = Vec::new();
        while self.is_field_start() {
            let field_span = self.current_span();
            let field_name = self.ident()?;
            self.expect(&TokenKind::Colon)?;
            let field_ty = self.field_type_expr()?;

            // Optional default value: `field : Type := value`
            let default = if self.eat(&TokenKind::ColonEq) {
                Some(self.field_type_expr()?)
            } else {
                None
            };

            fields.push(SurfaceField {
                span: field_span,
                name: field_name,
                ty: field_ty,
                default,
            });
        }

        // Parse optional deriving clause: `deriving Repr, BEq`
        let deriving = self.parse_deriving_clause()?;

        Ok(SurfaceDecl::Structure {
            span: start_span,
            name,
            universe_params,
            binders,
            ty,
            fields,
            deriving,
        })
    }

    /// Parse a deriving clause: `deriving Class1, Class2, ...`
    fn parse_deriving_clause(&mut self) -> Result<Vec<String>, ParseError> {
        if !self.eat(&TokenKind::Deriving) {
            return Ok(Vec::new());
        }

        let mut classes = Vec::new();

        // Parse first class name
        classes.push(self.ident()?);

        // Parse remaining class names separated by commas
        while self.eat(&TokenKind::Comma) {
            classes.push(self.ident()?);
        }

        Ok(classes)
    }

    /// Check if the current position looks like a field declaration start.
    /// A field starts with an identifier followed by a colon.
    fn is_field_start(&self) -> bool {
        if !matches!(self.current_kind(), TokenKind::Ident(_)) {
            return false;
        }
        // Check if next token is a colon
        self.tokens
            .get(self.pos + 1)
            .is_some_and(|t| matches!(t.kind, TokenKind::Colon))
    }

    /// Parse a field type expression.
    /// This is like `expr()` but stops before identifiers that look like field names
    /// (i.e., identifiers followed by colons).
    fn field_type_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        self.field_arrow_expr()
    }

    /// Arrow types for field expressions, stopping at field boundaries
    fn field_arrow_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.field_app_expr()?;

        while self.eat(&TokenKind::Arrow) {
            let right = self.field_arrow_expr()?;
            let span = left.span().merge(right.span());
            left = SurfaceExpr::Arrow(span, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Application expressions for fields, stopping at field boundaries
    fn field_app_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut expr = self.atom_expr()?;
        let mut pending_args: Vec<SurfaceArg> = Vec::new();

        loop {
            // Stop if the next token looks like a field name (ident followed by colon)
            if self.is_field_start() {
                break;
            }

            if self.is_atom_start() {
                let arg = self.atom_expr()?;
                pending_args.push(SurfaceArg::positional(arg));
                continue;
            }

            if self.eat(&TokenKind::Dot) {
                // Flush pending arguments if any
                if !pending_args.is_empty() {
                    let span = expr.span();
                    expr = SurfaceExpr::App(span, Box::new(expr), pending_args);
                    pending_args = Vec::new();
                }

                let (projection, end_span) = match self.current_kind().clone() {
                    TokenKind::Ident(field) => {
                        let end_span = self.current_span();
                        self.advance();
                        (Projection::Named(field), end_span)
                    }
                    TokenKind::NatLit(n) => {
                        let end_span = self.current_span();
                        self.advance();
                        let idx = u32::try_from(n).map_err(|_| ParseError::UnexpectedToken {
                            line: 0,
                            col: self.current_span().start,
                            message: format!("projection index too large: {n}"),
                        })?;
                        (Projection::Index(idx), end_span)
                    }
                    other => {
                        return Err(ParseError::UnexpectedToken {
                            line: 0,
                            col: self.current_span().start,
                            message: format!("expected field name or index, got {other:?}"),
                        })
                    }
                };

                let proj_span = expr.span().merge(end_span);
                expr = SurfaceExpr::Proj(proj_span, Box::new(expr), projection);
                continue;
            }

            break;
        }

        if pending_args.is_empty() {
            Ok(expr)
        } else {
            let span = expr.span();
            Ok(SurfaceExpr::App(span, Box::new(expr), pending_args))
        }
    }

    fn universe_params(&mut self) -> Result<Vec<String>, ParseError> {
        let mut params = Vec::new();

        // Universe params can be:
        // 1. .{u v} - explicit with dot prefix
        // 2. {u v} - identifiers only, no colon (distinguishes from implicit binders {α : Type})

        if self.check(&TokenKind::Dot) {
            // Check for .{u v} syntax
            let next_is_lbrace = self
                .tokens
                .get(self.pos + 1)
                .is_some_and(|t| matches!(t.kind, TokenKind::LBrace));
            if next_is_lbrace {
                self.advance(); // consume dot
                self.advance(); // consume lbrace
                while let TokenKind::Ident(name) = self.current_kind() {
                    params.push(name.clone());
                    self.advance();
                }
                self.expect(&TokenKind::RBrace)?;
            }
        } else if self.check(&TokenKind::LBrace) {
            // Check for {u v} style - but only if it's NOT followed by a colon
            // (which would make it an implicit binder like {α : Type})
            let saved_pos = self.pos;
            self.advance(); // consume lbrace

            // Collect identifiers
            let mut names = Vec::new();
            while let TokenKind::Ident(name) = self.current_kind() {
                names.push(name.clone());
                self.advance();
            }

            // If we see RBrace (not Colon), these are universe params
            if self.check(&TokenKind::RBrace) && !names.is_empty() {
                self.advance(); // consume rbrace
                params = names;
            } else {
                // Backtrack - this is an implicit binder, not universe params
                self.pos = saved_pos;
            }
        }

        Ok(params)
    }

    fn optional_binders(&mut self) -> Result<Vec<SurfaceBinder>, ParseError> {
        let mut binders = Vec::new();

        loop {
            match self.current_kind() {
                TokenKind::LParen => binders.extend(self.explicit_binders()?),
                TokenKind::LBrace => binders.extend(self.implicit_binders()?),
                TokenKind::LBracket => binders.extend(self.instance_binders()?),
                // Bare identifier binders (without parentheses): `def foo x y := ...`
                TokenKind::Ident(name) => {
                    // First check if current ident immediately precedes := : | where
                    // If so, it's the last binder - consume it then break
                    // Otherwise, consume it and continue looking for more
                    let is_last_binder = matches!(
                        self.peek_kind(1),
                        Some(
                            TokenKind::ColonEq
                                | TokenKind::Colon
                                | TokenKind::Pipe
                                | TokenKind::Where
                        )
                    );
                    let span = self.current_span();
                    let name = name.clone();
                    self.advance();
                    binders.push(SurfaceBinder {
                        span,
                        name,
                        ty: None,
                        default: None,
                        info: SurfaceBinderInfo::Explicit,
                    });
                    if is_last_binder {
                        break;
                    }
                }
                _ => break,
            }
        }

        Ok(binders)
    }

    /// Parse a class declaration
    ///
    /// ```text
    /// class Add (α : Type) where
    ///   add : α → α → α
    /// ```
    fn class_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;
        let universe_params = self.universe_params()?;
        let binders = self.optional_binders()?;

        // Optional extends clause: `class Foo extends Bar, Baz`
        if self.eat(&TokenKind::Extends) {
            // Skip the parent types for now (simplified handling)
            while !matches!(
                self.current_kind(),
                TokenKind::Where | TokenKind::Colon | TokenKind::Eof
            ) {
                self.advance();
            }
        }

        // Optional explicit result type: `class Foo : Type 1 where`
        let ty = if self.eat(&TokenKind::Colon) {
            Some(Box::new(self.expr()?))
        } else {
            None
        };

        // Handle case where class has no fields
        if !self.check(&TokenKind::Where) {
            return Ok(SurfaceDecl::Class {
                span: start_span,
                name,
                universe_params,
                binders,
                ty,
                fields: Vec::new(),
            });
        }

        self.expect(&TokenKind::Where)?;

        // Parse fields (same as structure)
        let mut fields = Vec::new();
        while self.is_field_start() {
            let field_span = self.current_span();
            let field_name = self.ident()?;
            self.expect(&TokenKind::Colon)?;
            let field_ty = self.field_type_expr()?;

            // Optional default value
            let default = if self.eat(&TokenKind::ColonEq) {
                Some(self.field_type_expr()?)
            } else {
                None
            };

            fields.push(SurfaceField {
                span: field_span,
                name: field_name,
                ty: field_ty,
                default,
            });
        }

        Ok(SurfaceDecl::Class {
            span: start_span,
            name,
            universe_params,
            binders,
            ty,
            fields,
        })
    }

    /// Parse an instance declaration
    ///
    /// ```text
    /// instance : Add Nat where
    ///   add := Nat.add
    /// ```
    ///
    /// Or with name:
    /// ```text
    /// instance instAddNat : Add Nat where
    ///   add := Nat.add
    /// ```
    ///
    /// Or with parameters:
    /// ```text
    /// instance [Add α] [Add β] : Add (α × β) where
    ///   add := fun (a, b) (c, d) => (add a c, add b d)
    /// ```
    ///
    /// Or with priority attribute:
    /// ```text
    /// @[instance 50] instance : Add Nat where ...
    /// @[defaultInstance] instance : ToString Nat where ...
    /// ```
    fn instance_decl(
        &mut self,
        start_span: Span,
        attrs: &[Attribute],
    ) -> Result<SurfaceDecl, ParseError> {
        let universe_params = self.universe_params()?;

        // Skip optional (priority := expr) declaration option
        // This appears before the instance name/type
        if self.check(&TokenKind::LParen) {
            let saved_pos = self.pos;
            self.advance();
            if let TokenKind::Ident(kw) = self.current_kind() {
                if kw == "priority"
                    && self.tokens.get(self.pos + 1).map(|t| &t.kind) == Some(&TokenKind::ColonEq)
                {
                    // Skip (priority := expr)
                    let mut depth = 1;
                    while depth > 0 && !matches!(self.current_kind(), TokenKind::Eof) {
                        match self.current_kind() {
                            TokenKind::LParen => depth += 1,
                            TokenKind::RParen => depth -= 1,
                            _ => {}
                        }
                        self.advance();
                    }
                } else {
                    // Not a priority option, backtrack
                    self.pos = saved_pos;
                }
            } else {
                self.pos = saved_pos;
            }
        }

        // Check for optional name: `instance instAddNat : ...`
        // vs anonymous: `instance : ...`
        // We need to distinguish between a name and a binder/colon
        let name = if let TokenKind::Ident(_) = self.current_kind() {
            let saved_pos = self.pos;
            if let Ok(candidate) = self.qualified_ident() {
                if self.check(&TokenKind::Colon) {
                    Some(candidate)
                } else {
                    self.pos = saved_pos;
                    None
                }
            } else {
                self.pos = saved_pos;
                None
            }
        } else {
            None
        };

        // Parse optional binders
        let binders = self.optional_binders()?;

        // Expect colon followed by class type
        self.expect(&TokenKind::Colon)?;
        let class_type = self.expr()?;

        let mut fields = Vec::new();
        if self.eat(&TokenKind::Where) {
            // Parse field assignments
            while self.is_field_assign_start() {
                let field_span = self.current_span();
                let field_name = self.ident()?;
                self.expect(&TokenKind::ColonEq)?;
                let field_val = self.instance_field_value_expr()?;

                fields.push(SurfaceFieldAssign {
                    span: field_span,
                    name: field_name,
                    val: field_val,
                });
            }
        } else if self.eat(&TokenKind::ColonEq) {
            // Short instance form: `instance : Class := expr`
            let val = self.expr()?;
            fields.push(SurfaceFieldAssign {
                span: start_span,
                name: "_value".to_string(),
                val,
            });
        } else {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!(
                    "expected `where` or `:=` in instance declaration, got {:?}",
                    self.current_kind()
                ),
            });
        }

        // Extract priority from attributes (first matching priority takes precedence)
        let priority = attrs.iter().find_map(Attribute::instance_priority);

        Ok(SurfaceDecl::Instance {
            span: start_span,
            name,
            universe_params,
            binders,
            class_type: Box::new(class_type),
            fields,
            priority,
        })
    }

    /// Check if current position looks like a field assignment start
    /// A field assignment starts with an identifier followed by `:=`
    fn is_field_assign_start(&self) -> bool {
        if !matches!(self.current_kind(), TokenKind::Ident(_)) {
            return false;
        }
        self.tokens
            .get(self.pos + 1)
            .is_some_and(|t| matches!(t.kind, TokenKind::ColonEq))
    }

    /// Parse an instance field value expression.
    /// This is like `expr()` but stops before identifiers that look like field assignments
    /// (i.e., identifiers followed by `:=`).
    fn instance_field_value_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        self.instance_field_arrow_expr()
    }

    /// Arrow types for instance field values, stopping at field assignment boundaries
    fn instance_field_arrow_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.instance_field_app_expr()?;

        while self.eat(&TokenKind::Arrow) {
            let right = self.instance_field_arrow_expr()?;
            let span = left.span().merge(right.span());
            left = SurfaceExpr::Arrow(span, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Application expressions for instance fields, stopping at field assignment boundaries
    fn instance_field_app_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut expr = self.atom_expr()?;
        let mut pending_args: Vec<SurfaceArg> = Vec::new();

        loop {
            // Stop if the next token looks like a field assignment (ident followed by :=)
            if self.is_field_assign_start() {
                break;
            }

            if self.is_atom_start() {
                let arg = self.atom_expr()?;
                pending_args.push(SurfaceArg::positional(arg));
                continue;
            }

            if self.eat(&TokenKind::Dot) {
                // Flush pending arguments if any
                if !pending_args.is_empty() {
                    let span = expr.span();
                    expr = SurfaceExpr::App(span, Box::new(expr), pending_args);
                    pending_args = Vec::new();
                }

                let (projection, end_span) = match self.current_kind().clone() {
                    TokenKind::Ident(field) => {
                        let end_span = self.current_span();
                        self.advance();
                        (Projection::Named(field), end_span)
                    }
                    TokenKind::NatLit(n) => {
                        let end_span = self.current_span();
                        self.advance();
                        let idx = u32::try_from(n).map_err(|_| ParseError::UnexpectedToken {
                            line: 0,
                            col: self.current_span().start,
                            message: format!("projection index too large: {n}"),
                        })?;
                        (Projection::Index(idx), end_span)
                    }
                    other => {
                        return Err(ParseError::UnexpectedToken {
                            line: 0,
                            col: self.current_span().start,
                            message: format!("expected field name or index, got {other:?}"),
                        })
                    }
                };

                let proj_span = expr.span().merge(end_span);
                expr = SurfaceExpr::Proj(proj_span, Box::new(expr), projection);
                continue;
            }

            break;
        }

        if pending_args.is_empty() {
            Ok(expr)
        } else {
            let span = expr.span();
            Ok(SurfaceExpr::App(span, Box::new(expr), pending_args))
        }
    }

    // =========================================================================
    // New declaration types for Lean 4 compatibility
    // =========================================================================

    /// Parse example declaration: `example : ty := proof`
    fn example_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let binders = self.optional_binders()?;

        let ty = if self.eat(&TokenKind::Colon) {
            Some(Box::new(self.expr()?))
        } else {
            None
        };

        self.expect(&TokenKind::ColonEq)?;
        let val = self.expr()?;

        Ok(SurfaceDecl::Example {
            span: start_span,
            binders,
            ty,
            val: Box::new(val),
        })
    }

    /// Parse import declaration: `import Lean.Data.List`
    fn import_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let mut paths = Vec::new();

        loop {
            let path = self.module_path()?;
            paths.push(path);

            // Support comma or whitespace separated modules on same line
            if self.eat(&TokenKind::Comma) {
                continue;
            }

            // Stop if next token is not an identifier (start of next declaration)
            if !matches!(self.current_kind(), TokenKind::Ident(_)) {
                break;
            }
        }

        Ok(SurfaceDecl::Import {
            span: start_span,
            paths,
        })
    }

    /// Parse namespace declaration: `namespace Foo ... end Foo`
    fn namespace_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;

        // Parse declarations until `end` or end of file
        // In Lean 4, `namespace` without `end` is valid - the namespace ends at file end
        let mut decls = Vec::new();
        while !matches!(self.current_kind(), TokenKind::End | TokenKind::Eof) {
            decls.push(self.decl()?);
        }

        // `end` is optional - namespace can end at EOF
        if self.eat(&TokenKind::End) {
            // Optionally consume the namespace name after end
            if let TokenKind::Ident(n) = self.current_kind() {
                let expected = name.rsplit('.').next().unwrap_or(&name);
                if n == expected {
                    self.advance();
                }
            }
        }

        Ok(SurfaceDecl::Namespace {
            span: start_span,
            name,
            decls,
        })
    }

    /// Parse section declaration: `section [Name] ... end [Name]`
    fn section_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        // Section name is optional
        let name = if let TokenKind::Ident(_) = self.current_kind() {
            Some(self.ident()?)
        } else {
            None
        };

        // Parse declarations until `end` or end of file
        // In Lean 4, `section` without `end` is valid - the section ends at file end
        let mut decls = Vec::new();
        while !matches!(self.current_kind(), TokenKind::End | TokenKind::Eof) {
            decls.push(self.decl()?);
        }

        // `end` is optional - section can end at EOF
        if self.eat(&TokenKind::End) {
            // Optionally consume the section name after end
            if let (Some(n), TokenKind::Ident(tok_name)) = (&name, self.current_kind()) {
                let expected = n.rsplit('.').next().unwrap_or(n);
                if tok_name == expected {
                    self.advance();
                }
            }
        }

        Ok(SurfaceDecl::Section {
            span: start_span,
            name,
            decls,
        })
    }

    /// Parse universe declaration: `universe u v`
    fn universe_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let mut names = Vec::new();

        while let TokenKind::Ident(_) = self.current_kind() {
            names.push(self.ident()?);
        }

        if names.is_empty() {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: "expected at least one universe parameter name".to_string(),
            });
        }

        Ok(SurfaceDecl::UniverseDecl {
            span: start_span,
            names,
        })
    }

    /// Parse variable declaration: `variable (x : Type)`
    fn variable_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let binders = self.binders()?;

        Ok(SurfaceDecl::Variable {
            span: start_span,
            binders,
        })
    }

    /// Parse open command: `open Nat in ...` or `open Nat (add mul)`
    fn open_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let mut paths = Vec::new();

        loop {
            let path = self.module_path()?;

            // Check for specific names: `open Nat (add mul)`
            let mut names = Vec::new();
            if self.eat(&TokenKind::LParen) {
                while let TokenKind::Ident(_) = self.current_kind() {
                    names.push(self.ident()?);
                }
                self.expect(&TokenKind::RParen)?;
            }

            paths.push(OpenPath { path, names });

            if self.check(&TokenKind::In) || !matches!(self.current_kind(), TokenKind::Ident(_)) {
                break;
            }
        }

        // Check for `in` followed by body
        let body = if self.eat(&TokenKind::In) {
            Some(Box::new(self.decl()?))
        } else {
            None
        };

        Ok(SurfaceDecl::Open {
            span: start_span,
            paths,
            body,
        })
    }

    /// Parse mutual block: `mutual ... end`
    fn mutual_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let mut decls = Vec::new();

        while !matches!(self.current_kind(), TokenKind::End | TokenKind::Eof) {
            decls.push(self.decl()?);
        }

        self.expect(&TokenKind::End)?;

        Ok(SurfaceDecl::Mutual {
            span: start_span,
            decls,
        })
    }

    /// Parse hash commands: `#check`, `#eval`, `#print`
    fn hash_command(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        match self.current_kind().clone() {
            TokenKind::Ident(cmd) => {
                self.advance();
                match cmd.as_str() {
                    "check" => {
                        let expr = self.expr()?;
                        Ok(SurfaceDecl::Check {
                            span: start_span,
                            expr: Box::new(expr),
                        })
                    }
                    "eval" => {
                        let expr = self.expr()?;
                        Ok(SurfaceDecl::Eval {
                            span: start_span,
                            expr: Box::new(expr),
                        })
                    }
                    "print" => {
                        let name = self.qualified_ident()?;
                        Ok(SurfaceDecl::Print {
                            span: start_span,
                            name,
                        })
                    }
                    "reduce" | "whnf" | "norm_num" => {
                        // Treat as eval
                        let expr = self.expr()?;
                        Ok(SurfaceDecl::Eval {
                            span: start_span,
                            expr: Box::new(expr),
                        })
                    }
                    _ => {
                        // Unknown hash command - try to skip it gracefully
                        self.skip_to_next_decl_token();
                        Ok(SurfaceDecl::Check {
                            span: start_span,
                            expr: Box::new(SurfaceExpr::Hole(start_span)),
                        })
                    }
                }
            }
            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!("expected command after #, got {:?}", self.current_kind()),
            }),
        }
    }

    /// Parse abbrev declaration (like def but unfolds eagerly)
    /// Also handles `abbrev class` which creates an abbreviation for a class
    fn abbrev_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        // Check if abbrev is a modifier for class
        if self.eat(&TokenKind::Class) {
            return self.class_decl(start_span);
        }
        // Parse like def
        self.def_decl(start_span)
    }

    /// Parse attribute command: `attribute [simp] foo bar`
    fn attribute_cmd(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let attrs = if self.check(&TokenKind::LBracket) {
            // Parse @[...] style attributes without the leading @
            self.expect(&TokenKind::LBracket)?;
            let mut attrs = Vec::new();
            loop {
                let attr = self.single_attribute()?;
                attrs.push(attr);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RBracket)?;
            attrs
        } else {
            Vec::new()
        };

        let mut names = Vec::new();
        while let TokenKind::Ident(_) = self.current_kind() {
            names.push(self.qualified_ident()?);
        }

        Ok(SurfaceDecl::Attribute {
            span: start_span,
            attrs,
            names,
        })
    }

    /// Parse `set_option` command
    fn set_option_cmd(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.qualified_ident()?;

        let value = match self.current_kind().clone() {
            TokenKind::Ident(v) => {
                self.advance();
                Some(v)
            }
            TokenKind::NatLit(n) => {
                self.advance();
                Some(n.to_string())
            }
            TokenKind::StringLit(s) => {
                self.advance();
                Some(s)
            }
            _ => None,
        };

        Ok(SurfaceDecl::SetOption {
            span: start_span,
            name,
            value,
        })
    }

    // ========================================================================
    // Macro system parsing
    // ========================================================================

    /// Parse a syntax declaration: `syntax [name]? [prec]? pattern... : category`
    ///
    /// Examples:
    /// - `syntax term "+" term : term`
    /// - `syntax:50 term "+" term : term`
    /// - `syntax [myAdd] term "+" term : term`
    fn syntax_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        // Check for optional precedence: syntax:50
        let precedence = self.parse_precedence_suffix();

        // Check for optional name: [name]
        let name = if self.eat(&TokenKind::LBracket) {
            let n = self.ident()?;
            self.expect(&TokenKind::RBracket)?;
            Some(n)
        } else {
            None
        };

        // Check for optional priority: (priority := N)
        let priority = self.parse_priority_attr();

        // Parse the syntax pattern until we hit `:` (category delimiter)
        let pattern = self.parse_syntax_pattern()?;

        // Expect `: category`
        self.expect(&TokenKind::Colon)?;
        let category = self.ident()?;

        Ok(SurfaceDecl::Syntax {
            span: start_span,
            name,
            precedence,
            priority,
            pattern,
            category,
        })
    }

    /// Parse `declare_syntax_cat`: `declare_syntax_cat name`
    fn declare_syntax_cat_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        let name = self.ident()?;
        Ok(SurfaceDecl::DeclareSyntaxCat {
            span: start_span,
            name,
        })
    }

    /// Parse a macro declaration: `macro pattern... : category => expansion`
    ///
    /// Examples:
    /// - `macro "unless" cond:term "then" body:term : term => ...`
    fn macro_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        // Parse the syntax pattern until we hit `:` (category delimiter)
        let pattern = self.parse_syntax_pattern()?;

        // Expect `: category`
        self.expect(&TokenKind::Colon)?;
        let category = self.ident()?;

        // Expect `=>`
        self.expect(&TokenKind::FatArrow)?;

        // Parse the expansion (a syntax quotation or expression)
        let expansion = self.expr()?;

        Ok(SurfaceDecl::Macro {
            span: start_span,
            doc: None,
            pattern,
            category,
            expansion: Box::new(expansion),
        })
    }

    /// Parse `macro_rules` declaration with multiple arms
    ///
    /// Examples:
    /// - `macro_rules | `(...) => `(...) | `(...) => `(...)`
    fn macro_rules_decl(&mut self, start_span: Span) -> Result<SurfaceDecl, ParseError> {
        // Optional name
        let name = if let TokenKind::Ident(_) = self.current_kind() {
            if self.check(&TokenKind::Pipe) {
                None
            } else {
                Some(self.ident()?)
            }
        } else {
            None
        };

        // Parse arms: | pattern => expansion
        let mut arms = Vec::new();
        while self.eat(&TokenKind::Pipe) {
            let arm_span = self.current_span();

            // Parse pattern (typically a syntax quotation)
            let pattern = self.expr()?;

            // Expect =>
            self.expect(&TokenKind::FatArrow)?;

            // Parse expansion
            let expansion = self.expr()?;

            arms.push(MacroArm {
                span: arm_span,
                pattern: Box::new(pattern),
                expansion: Box::new(expansion),
            });
        }

        Ok(SurfaceDecl::MacroRules {
            span: start_span,
            name,
            arms,
        })
    }

    /// Parse elab declaration (captures raw content for now)
    fn elab_decl(&mut self, start_span: Span) -> SurfaceDecl {
        // Elab declarations are complex - capture raw content for now
        let mut content = String::new();
        let mut depth = 0;

        while !self.at_decl_boundary(depth) {
            match self.current_kind().clone() {
                TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => depth += 1,
                TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                    depth = depth.saturating_sub(1);
                }
                TokenKind::Ident(s) => {
                    content.push(' ');
                    content.push_str(&s);
                }
                TokenKind::NatLit(n) => {
                    use std::fmt::Write;
                    write!(content, " {n}").unwrap();
                }
                TokenKind::StringLit(s) => {
                    use std::fmt::Write;
                    write!(content, " \"{s}\"").unwrap();
                }
                _ => content.push(' '),
            }
            self.advance();
        }

        SurfaceDecl::Elab {
            span: start_span,
            content,
        }
    }

    /// Parse notation declaration: `infixl:65 " + " => Add.add`
    fn notation_decl(
        &mut self,
        start_span: Span,
        kind: NotationKind,
    ) -> Result<SurfaceDecl, ParseError> {
        // Check for optional precedence: infixl:65
        let precedence = self.parse_precedence_suffix();

        // Parse the notation pattern
        let pattern = self.parse_notation_pattern();

        // Expect `=>`
        self.expect(&TokenKind::FatArrow)?;

        // Parse the expansion
        let expansion = self.expr()?;

        Ok(SurfaceDecl::Notation {
            span: start_span,
            kind,
            precedence,
            pattern,
            expansion: Box::new(expansion),
        })
    }

    /// Parse optional precedence suffix `:N` or `:max`
    fn parse_precedence_suffix(&mut self) -> Option<u32> {
        if self.eat(&TokenKind::Colon) {
            match self.current_kind().clone() {
                TokenKind::NatLit(n) => {
                    self.advance();
                    Some(n as u32)
                }
                TokenKind::Ident(s) if s == "max" => {
                    self.advance();
                    Some(1024) // max precedence
                }
                TokenKind::Ident(s) if s == "min" => {
                    self.advance();
                    Some(0)
                }
                TokenKind::Ident(s) if s == "arg" => {
                    self.advance();
                    Some(1023) // arg precedence
                }
                TokenKind::Ident(s) if s == "lead" => {
                    self.advance();
                    Some(1024) // lead = max
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Parse optional priority attribute: (priority := N)
    fn parse_priority_attr(&mut self) -> Option<u32> {
        if self.check(&TokenKind::LParen) {
            let pos = self.pos;
            self.advance();
            if let TokenKind::Ident(s) = self.current_kind().clone() {
                if s == "priority" {
                    self.advance();
                    if self.eat(&TokenKind::ColonEq) {
                        if let TokenKind::NatLit(n) = self.current_kind().clone() {
                            self.advance();
                            if self.eat(&TokenKind::RParen) {
                                return Some(n as u32);
                            }
                        }
                    }
                }
            }
            // Backtrack if not a priority attr
            self.pos = pos;
        }
        None
    }

    /// Parse a syntax pattern (sequence of items until `: category`)
    fn parse_syntax_pattern(&mut self) -> Result<Vec<SyntaxPatternItem>, ParseError> {
        let mut items = Vec::new();

        while !matches!(self.current_kind(), TokenKind::Eof) {
            // Check if we're at the end: last identifier followed by `: category`
            // We need to look ahead to detect `ident : ident` at the end
            if self.at_syntax_pattern_end() {
                break;
            }

            let item = self.parse_syntax_pattern_item()?;
            items.push(item);
        }

        Ok(items)
    }

    /// Check if we're at the end of a syntax pattern (` : category`)
    fn at_syntax_pattern_end(&self) -> bool {
        // Pattern ends at `: category` where category is a single identifier
        // and nothing follows (or EOF, or `=>` for macros)
        if self.check(&TokenKind::Colon) {
            // `: ident` at end - this is the category delimiter
            if let Some(TokenKind::Ident(_)) = self.peek_kind(1) {
                // Check that nothing meaningful follows the category
                let peek2 = self.peek_kind(2);
                if matches!(peek2, None | Some(TokenKind::Eof)) {
                    return true;
                }
                // Check if what follows is a declaration start (new declaration)
                // or `=>` for macro expansion
                if let Some(tok) = peek2 {
                    if matches!(
                        tok,
                        TokenKind::FatArrow  // macro => expansion
                            | TokenKind::Def
                            | TokenKind::Theorem
                            | TokenKind::Lemma
                            | TokenKind::Axiom
                            | TokenKind::Example
                            | TokenKind::Inductive
                            | TokenKind::Structure
                            | TokenKind::Class
                            | TokenKind::Instance
                            | TokenKind::Syntax
                            | TokenKind::Macro
                            | TokenKind::MacroRules
                            | TokenKind::Elab
                            | TokenKind::Notation
                            | TokenKind::Infixl
                            | TokenKind::Infixr
                            | TokenKind::Prefix
                            | TokenKind::Postfix
                            | TokenKind::Namespace
                            | TokenKind::Section
                            | TokenKind::Open
                            | TokenKind::Import
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Parse a single syntax pattern item
    fn parse_syntax_pattern_item(&mut self) -> Result<SyntaxPatternItem, ParseError> {
        match self.current_kind().clone() {
            // String literal: "if", "then", "+"
            TokenKind::StringLit(s) => {
                self.advance();
                Ok(SyntaxPatternItem::Literal(s))
            }

            // Identifier with optional category: `cond:term` or just `term`
            TokenKind::Ident(name) => {
                self.advance();

                // Check for `:category` suffix (variable binding like `x:term`)
                // But NOT if this is the final `: category` delimiter
                if self.check(&TokenKind::Colon) {
                    if let Some(TokenKind::Ident(cat)) = self.peek_kind(1).cloned() {
                        // Check if this is `var:cat` followed by more pattern items
                        // or if this is the end delimiter (followed by EOF, =>, or decl-start)
                        let peek2 = self.peek_kind(2);
                        let is_end = matches!(
                            peek2,
                            None | Some(
                                TokenKind::Eof
                                    | TokenKind::FatArrow
                                    | TokenKind::Def
                                    | TokenKind::Theorem
                                    | TokenKind::Syntax
                                    | TokenKind::Macro
                                    | TokenKind::MacroRules
                                    | TokenKind::Elab
                                    | TokenKind::Notation
                                    | TokenKind::Infixl
                                    | TokenKind::Infixr
                                    | TokenKind::Prefix
                                    | TokenKind::Postfix
                            )
                        );
                        if !is_end {
                            // More pattern follows - this is a variable binding
                            self.advance(); // eat :
                            self.advance(); // eat category
                            return Ok(SyntaxPatternItem::Variable {
                                name,
                                category: Some(cat),
                            });
                        }
                    }
                }

                // Check for repetition suffix: `,*` or `,+`
                if self.check(&TokenKind::Comma) {
                    let pos = self.pos;
                    self.advance();
                    if self.check(&TokenKind::Star) {
                        self.advance();
                        return Ok(SyntaxPatternItem::Repetition {
                            pattern: vec![SyntaxPatternItem::Variable {
                                name,
                                category: None,
                            }],
                            separator: Some(",".to_string()),
                            at_least_one: false,
                        });
                    } else if self.check(&TokenKind::Plus) {
                        self.advance();
                        return Ok(SyntaxPatternItem::Repetition {
                            pattern: vec![SyntaxPatternItem::Variable {
                                name,
                                category: None,
                            }],
                            separator: Some(",".to_string()),
                            at_least_one: true,
                        });
                    }
                    // Not a repetition, backtrack
                    self.pos = pos;
                }

                // Just a variable or category reference
                Ok(SyntaxPatternItem::Variable {
                    name,
                    category: None,
                })
            }

            // Optional group: (pattern)?
            TokenKind::LParen => {
                self.advance();
                let inner = self.parse_syntax_pattern()?;
                self.expect(&TokenKind::RParen)?;

                // Check for `?` suffix (optional)
                if let TokenKind::Ident(s) = self.current_kind() {
                    if s == "?" {
                        self.advance();
                        return Ok(SyntaxPatternItem::Optional(inner));
                    }
                }

                // Not optional, just grouped
                if inner.len() == 1 {
                    Ok(inner.into_iter().next().expect("inner.len() == 1"))
                } else {
                    // Multiple items in group - treat as repetition with no separator
                    Ok(SyntaxPatternItem::Repetition {
                        pattern: inner,
                        separator: None,
                        at_least_one: true,
                    })
                }
            }

            // Syntax quotation
            TokenKind::SyntaxQuote(content) => {
                self.advance();
                // For now, treat as a literal pattern
                Ok(SyntaxPatternItem::Literal(format!("`({content})")))
            }

            _ => Err(ParseError::UnexpectedToken {
                line: 0,
                col: self.current_span().start,
                message: format!(
                    "expected syntax pattern item, got {:?}",
                    self.current_kind()
                ),
            }),
        }
    }

    /// Parse a notation pattern (alternating literals and variables)
    fn parse_notation_pattern(&mut self) -> Vec<NotationItem> {
        let mut items = Vec::new();

        while !self.check(&TokenKind::FatArrow) && !matches!(self.current_kind(), TokenKind::Eof) {
            match self.current_kind().clone() {
                TokenKind::StringLit(s) => {
                    self.advance();
                    items.push(NotationItem::Literal(s));
                }
                TokenKind::Ident(name) => {
                    self.advance();
                    items.push(NotationItem::Variable(name));
                }
                _ => {
                    // Skip unknown tokens in notation patterns
                    self.advance();
                }
            }
        }

        items
    }

    /// Check if at declaration boundary (for elab parsing)
    fn at_decl_boundary(&self, depth: usize) -> bool {
        if depth > 0 {
            return false;
        }
        self.is_decl_start() || matches!(self.current_kind(), TokenKind::Eof)
    }

    /// Skip to next declaration (for unrecognized syntax)
    fn skip_to_next_decl(&mut self, kind: &str, start_span: Span) -> SurfaceDecl {
        // Collect the raw content until we hit a recognizable declaration token
        let mut content = kind.to_string();
        while !self.is_decl_start() && !matches!(self.current_kind(), TokenKind::Eof) {
            // Just advance through tokens
            match self.current_kind().clone() {
                TokenKind::Ident(s) => {
                    content.push(' ');
                    content.push_str(&s);
                }
                TokenKind::NatLit(n) => {
                    use std::fmt::Write;
                    write!(content, " {n}").unwrap();
                }
                TokenKind::StringLit(s) => {
                    use std::fmt::Write;
                    write!(content, " \"{s}\"").unwrap();
                }
                _ => content.push(' '),
            }
            self.advance();
        }

        // Return as Elab (raw content for unrecognized commands)
        SurfaceDecl::Elab {
            span: start_span,
            content,
        }
    }

    /// Check if current token starts a declaration
    fn is_decl_start(&self) -> bool {
        // Hash commands (#check, #eval) are declaration starters,
        // but #[ is an array literal, not a declaration
        if matches!(self.current_kind(), TokenKind::Hash) {
            // #[ is an array literal, not a declaration
            return !matches!(self.peek_kind(1), Some(TokenKind::LBracket));
        }
        matches!(
            self.current_kind(),
            TokenKind::Def
                | TokenKind::Theorem
                | TokenKind::Lemma
                | TokenKind::Axiom
                | TokenKind::Example
                | TokenKind::Inductive
                | TokenKind::Structure
                | TokenKind::Class
                | TokenKind::Instance
                | TokenKind::Import
                | TokenKind::Namespace
                | TokenKind::Section
                | TokenKind::Universe
                | TokenKind::Variable
                | TokenKind::Open
                | TokenKind::Mutual
                | TokenKind::End
                | TokenKind::At
                | TokenKind::Private
                | TokenKind::Protected
                | TokenKind::Partial
                | TokenKind::Unsafe
                | TokenKind::Noncomputable
                | TokenKind::Abbrev
                | TokenKind::Attribute
                | TokenKind::SetOption
                | TokenKind::Syntax
                | TokenKind::Macro
                | TokenKind::MacroRules
                | TokenKind::Elab
                | TokenKind::Notation
                | TokenKind::Infixl
                | TokenKind::Infixr
                | TokenKind::Prefix
                | TokenKind::Postfix
        )
    }

    /// Skip to next recognizable declaration token
    fn skip_to_next_decl_token(&mut self) {
        while !self.is_decl_start() && !matches!(self.current_kind(), TokenKind::Eof) {
            self.advance();
        }
    }
}

#[cfg(test)]
mod tests;
