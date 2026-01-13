//! TLA+ parser using rowan for lossless syntax tree
//!
//! This parser uses an event-based approach:
//! 1. Lex input into tokens
//! 2. Parse tokens into events (StartNode, FinishNode, AddToken, Error)
//! 3. Build rowan GreenNode from events
//!
//! The parser is a recursive descent parser with Pratt parsing for expressions.

use crate::syntax::kinds::{SyntaxKind, SyntaxNode};
use crate::syntax::lexer::Token;
use rowan::{GreenNode, GreenNodeBuilder};

/// A parsed token with its text, span, and column info for layout-aware parsing
#[derive(Debug, Clone)]
struct ParsedToken {
    kind: Token,
    text: String,
    start: u32,
    /// Column number (0-indexed) for layout-aware bullet list parsing
    column: u32,
}

/// Parser events that are collected during parsing
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Event {
    /// Start a new node
    StartNode { kind: SyntaxKind },
    /// Finish the current node
    FinishNode,
    /// Add a token
    AddToken { kind: SyntaxKind, text: String },
    /// Record an error
    Error { message: String },
    /// Placeholder for forward parent references (used for Pratt parsing)
    Placeholder,
}

/// Type of junction list (bullet-style And or Or)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JunctionType {
    Conjunction, // /\
    Disjunction, // \/
}

/// Information about an active junction list
#[derive(Debug, Clone)]
struct JunctionListInfo {
    /// The type of junction list (And or Or)
    #[allow(dead_code)] // Used for debugging and potential future extensions
    junction_type: JunctionType,
    /// The column where the junction list bullets appear
    column: u32,
}

/// Context for tracking nested bullet-style conjunction/disjunction lists.
/// This enables layout-aware parsing where the column position of `/\` and `\/`
/// determines list membership, similar to SANY's JunctionListContext.
#[derive(Debug, Default)]
struct JunctionListContext {
    /// Stack of active junction lists, innermost at the end
    stack: Vec<JunctionListInfo>,
}

impl JunctionListContext {
    /// Start a new junction list at the given column with the given type
    fn start(&mut self, column: u32, junction_type: JunctionType) {
        self.stack.push(JunctionListInfo {
            junction_type,
            column,
        });
    }

    /// End the current junction list
    fn end(&mut self) {
        self.stack.pop();
    }

    /// Check if a bullet at the given column and type continues the current junction list
    #[allow(dead_code)] // Kept for debugging and potential future extensions
    fn is_same_bullet(&self, column: u32, junction_type: JunctionType) -> bool {
        self.stack
            .last()
            .map(|info| info.column == column && info.junction_type == junction_type)
            .unwrap_or(false)
    }

    /// Check if the given column is strictly to the right of the current junction list's column,
    /// meaning the content is nested inside the current list item
    #[allow(dead_code)] // Kept for debugging and potential future extensions
    fn is_nested(&self, column: u32) -> bool {
        self.stack
            .last()
            .map(|info| column > info.column)
            .unwrap_or(true) // No active list = everything is allowed
    }

    /// Get the column of the current junction list, if any
    fn current_column(&self) -> Option<u32> {
        self.stack.last().map(|info| info.column)
    }
}

/// The parser state
pub struct Parser {
    /// Lexed tokens
    tokens: Vec<ParsedToken>,
    /// Current position in tokens
    pos: usize,
    /// Collected events
    events: Vec<Event>,
    /// Parse errors
    errors: Vec<ParseError>,
    /// Context for layout-aware bullet list parsing
    junction_context: JunctionListContext,
}

/// A parse error with location
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub start: u32,
    pub end: u32,
}

/// Result of parsing
pub struct ParseResult {
    pub green_node: GreenNode,
    pub errors: Vec<ParseError>,
}

impl Parser {
    /// Create a new parser for the given source
    pub fn new(source: &str) -> Self {
        let tokens: Vec<_> = lex_with_positions(source).collect();
        Self {
            tokens,
            pos: 0,
            events: Vec::new(),
            errors: Vec::new(),
            junction_context: JunctionListContext::default(),
        }
    }

    /// Parse and return the green node
    pub fn parse(mut self) -> ParseResult {
        self.parse_root();
        let errors = std::mem::take(&mut self.errors);
        let green_node = self.build_tree();
        ParseResult { green_node, errors }
    }

    /// Build the rowan tree from events
    fn build_tree(self) -> GreenNode {
        let mut builder = GreenNodeBuilder::new();
        let mut forward_parents: Vec<(usize, SyntaxKind)> = Vec::new();

        for (idx, event) in self.events.into_iter().enumerate() {
            // Check for forward parent references
            while let Some(&(fwd_idx, kind)) = forward_parents.last() {
                if fwd_idx == idx {
                    builder.start_node(kind.into());
                    forward_parents.pop();
                } else {
                    break;
                }
            }

            match event {
                Event::StartNode { kind } => {
                    builder.start_node(kind.into());
                }
                Event::FinishNode => {
                    builder.finish_node();
                }
                Event::AddToken { kind, text } => {
                    builder.token(kind.into(), &text);
                }
                Event::Error { message: _ } => {
                    // Errors are collected separately, add error node
                    builder.start_node(SyntaxKind::Error.into());
                    builder.finish_node();
                }
                Event::Placeholder => {
                    // Skip placeholders
                }
            }
        }

        builder.finish()
    }

    // === Token access ===

    /// Get the current token kind
    fn current(&self) -> Option<Token> {
        self.tokens.get(self.pos).map(|t| t.kind)
    }

    /// Get the current token text
    #[allow(dead_code)]
    fn current_text(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|t| t.text.as_str())
    }

    /// Get the current token's column (for layout-aware bullet list parsing)
    fn current_column(&self) -> u32 {
        self.tokens.get(self.pos).map(|t| t.column).unwrap_or(0)
    }

    /// Get the current position
    fn current_pos(&self) -> u32 {
        self.tokens.get(self.pos).map(|t| t.start).unwrap_or(0)
    }

    /// Peek at the next token kind
    fn peek(&self) -> Option<Token> {
        self.peek_nth(1)
    }

    /// Peek at the Nth non-trivia token ahead (1 = next, 2 = next+1, etc.)
    fn peek_nth(&self, n: usize) -> Option<Token> {
        let mut pos = self.pos + 1;
        let mut count = 0;
        while pos < self.tokens.len() {
            let token = &self.tokens[pos];
            if !token.kind.is_trivia() {
                count += 1;
                if count == n {
                    return Some(token.kind);
                }
            }
            pos += 1;
        }
        None
    }

    /// Check if current position looks like the start of a proof step label.
    ///
    /// Proof step labels are ambiguous with BY-clause step references because both use `<n>...`.
    /// Heuristic:
    /// - A label is `<n>` followed by an optional label token, optionally a dot, then a token that
    ///   can start a proof step body (proof keyword or expression).
    /// - Step references tend to be followed by `,`, `DEF/DEFS`, or immediately by the next
    ///   step label (`<...>`).
    fn is_step_label_start(&self) -> bool {
        if self.current() != Some(Token::Lt) {
            return false;
        }
        // Check: < NUMBER >
        let next1 = self.peek_nth(1);
        let next2 = self.peek_nth(2);
        if !matches!((next1, next2), (Some(Token::Number), Some(Token::Gt))) {
            return false;
        }
        // Check for optional label and then either dot or a token that can start a proof step body.
        let next3 = self.peek_nth(3);
        match next3 {
            Some(Token::Dot) => true,
            Some(Token::Ident) | Some(Token::Number) => {
                let next4 = self.peek_nth(4);
                if next4 == Some(Token::Dot) {
                    return true;
                }
                // If next4 is an Ident followed by == (DefEq), this is a step REFERENCE
                // followed by an operator definition at module level, NOT a step label start.
                // e.g., in `BY <1>1\n\n1bOr2bMsgs ==`, the `<1>1` is a reference.
                if next4 == Some(Token::Ident) {
                    let next5 = self.peek_nth(5);
                    if next5 == Some(Token::DefEq) {
                        return false;
                    }
                }
                // If the token after the label can start a proof step body, treat next3 as a label.
                if next4.is_some_and(is_proof_step_body_start) {
                    return true;
                }
                // Otherwise, next3 might actually be the start of the step body (e.g., `<2> A = 0`).
                // In that case, only accept it as a step label start if it looks unlike a step ref.
                if matches!(
                    next4,
                    Some(
                        Token::Comma
                            | Token::Def
                            | Token::Defs
                            | Token::By
                            | Token::Obvious
                            | Token::Omitted
                            | Token::Proof
                            | Token::Lt
                    )
                ) {
                    return false;
                }
                is_proof_step_body_start(next3.unwrap())
            }
            Some(token) => is_proof_step_body_start(token),
            None => false,
        }
    }

    /// Check if we're at a parenthesized infix operator: (op)
    /// Used to distinguish `B1 (+) B2 ==` from `Op(x, y) ==`
    fn is_parenthesized_infix_op(&self) -> bool {
        if self.current() != Some(Token::LParen) {
            return false;
        }
        let next1 = self.peek_nth(1);
        let next2 = self.peek_nth(2);
        // Pattern: ( op )
        if let Some(op) = next1 {
            if is_infix_op_symbol(op) && next2 == Some(Token::RParen) {
                return true;
            }
        }
        false
    }

    /// Check if at end of file
    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Check if current token matches
    fn at(&self, kind: Token) -> bool {
        self.current() == Some(kind)
    }

    /// Check if current is any of the given tokens
    #[allow(dead_code)]
    fn at_any(&self, kinds: &[Token]) -> bool {
        self.current().map(|k| kinds.contains(&k)).unwrap_or(false)
    }

    // === Event building ===

    /// Create a checkpoint at the current position for Pratt parsing
    fn checkpoint(&self) -> usize {
        self.events.len()
    }

    /// Start a node at a previously saved checkpoint (wraps previous content)
    fn start_node_at(&mut self, checkpoint: usize, kind: SyntaxKind) {
        // Insert StartNode at the checkpoint position
        self.events.insert(checkpoint, Event::StartNode { kind });
    }

    /// Start a new node
    fn start_node(&mut self, kind: SyntaxKind) {
        self.events.push(Event::StartNode { kind });
    }

    /// Finish the current node
    fn finish_node(&mut self) {
        self.events.push(Event::FinishNode);
    }

    /// Bump the current token and advance
    fn bump(&mut self) {
        if let Some(token) = self.tokens.get(self.pos) {
            let kind = token_to_syntax_kind(token.kind);
            self.events.push(Event::AddToken {
                kind,
                text: token.text.clone(),
            });
            self.pos += 1;
        }
    }

    /// Bump and skip trivia
    fn bump_skip_trivia(&mut self) {
        self.bump();
        self.skip_trivia();
    }

    /// Advance past current token, adding it to the syntax tree for position tracking.
    /// Used for leading bullets (/\ and \/) which are syntactic sugar but must be
    /// included in the tree to maintain accurate source positions.
    fn advance_skip_trivia(&mut self) {
        // Add the current token to maintain position tracking
        self.bump();
        // Add following trivia to the tree as well
        self.skip_trivia();
    }

    /// Skip whitespace and comments, adding them to the syntax tree
    fn skip_trivia(&mut self) {
        while let Some(token) = self.tokens.get(self.pos) {
            if token.kind.is_trivia() {
                let kind = token_to_syntax_kind(token.kind);
                self.events.push(Event::AddToken {
                    kind,
                    text: token.text.clone(),
                });
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Skip whitespace and comments WITHOUT adding them to the syntax tree.
    /// Use this during lookahead when we might reset the position.
    fn skip_trivia_no_emit(&mut self) {
        while let Some(token) = self.tokens.get(self.pos) {
            if token.kind.is_trivia() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Expect a specific token, emit error if not found
    fn expect(&mut self, expected: Token) -> bool {
        if self.at(expected) {
            self.bump_skip_trivia();
            true
        } else {
            self.error(format!("expected {:?}", expected));
            false
        }
    }

    /// Record an error
    fn error(&mut self, message: String) {
        let start = self.current_pos();
        let end = self
            .tokens
            .get(self.pos)
            .map(|t| t.start + t.text.len() as u32)
            .unwrap_or(start);
        self.errors.push(ParseError {
            message: message.clone(),
            start,
            end,
        });
        self.events.push(Event::Error { message });
    }

    // === Top-level parsing ===

    /// Parse the root (entire file)
    fn parse_root(&mut self) {
        self.start_node(SyntaxKind::Root);
        self.skip_trivia();

        // Parse module(s), skipping any text before/after/between modules.
        // Many real-world .tla files contain:
        // - Documentation or README text before the module header
        // - CONFIG file content (not a TLA+ module)
        // - Trailing notes, shell commands after the module end
        // SANY effectively ignores non-module content, so we do the same.
        while !self.at_eof() {
            if self.at(Token::ModuleStart) {
                // Only treat dashes as a module header if followed by `MODULE`.
                let mut nth = 1;
                while self.peek_nth(nth) == Some(Token::ModuleStart) {
                    nth += 1;
                }
                if self.peek_nth(nth) == Some(Token::Module) {
                    self.parse_module();
                } else {
                    // Dashes not followed by MODULE - skip them
                    self.bump_skip_trivia();
                }
            } else {
                self.bump_skip_trivia();
            }
        }

        self.finish_node();
    }

    /// Parse a module: ---- MODULE Name ---- ... ====
    fn parse_module(&mut self) {
        self.start_node(SyntaxKind::Module);

        // ---- MODULE
        self.expect(Token::ModuleStart);
        while self.at(Token::ModuleStart) {
            self.bump_skip_trivia(); // Allow multiple ----
        }
        self.expect(Token::Module);

        // Module name (can start with number like 2PCwithBTM)
        self.parse_module_name();

        // ----
        while self.at(Token::ModuleStart) {
            self.bump_skip_trivia();
        }

        // EXTENDS clause (optional)
        if self.at(Token::Extends) {
            self.parse_extends();
        }

        // Module body
        while !self.at_eof() && !self.at(Token::ModuleEnd) {
            self.parse_unit();
        }

        // ====
        if self.at(Token::ModuleEnd) {
            self.bump_skip_trivia();
            while self.at(Token::ModuleEnd) {
                self.bump_skip_trivia(); // Allow multiple ====
            }
        } else {
            self.error("expected module end (====)".to_string());
        }

        self.finish_node();
    }

    /// Parse EXTENDS M1, M2, ...
    fn parse_extends(&mut self) {
        self.start_node(SyntaxKind::ExtendsClause);
        self.bump_skip_trivia(); // EXTENDS

        // Parse comma-separated module names
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                if self.at(Token::Ident) {
                    self.bump_skip_trivia();
                } else {
                    self.error("expected module name after comma".to_string());
                    break;
                }
            }
        }

        self.finish_node();
    }

    /// Parse a submodule definition (embedded module within a module)
    /// Syntax: ---- MODULE Name ---- ... ====
    fn parse_submodule(&mut self) {
        self.start_node(SyntaxKind::Module);

        // Parse module start delimiter (----)
        self.bump_skip_trivia(); // First dash set

        // MODULE keyword
        if self.at(Token::Module) {
            self.bump_skip_trivia();
        }

        // Module name (can start with number like 2PCwithBTM)
        if self.at(Token::Ident) || self.at(Token::Number) {
            self.parse_module_name_optional();
        }

        // Consume trailing dashes/equals after module name
        while self.at(Token::ModuleStart) || self.at(Token::ModuleEnd) {
            self.bump_skip_trivia();
        }

        // Parse EXTENDS clause if present
        if self.at(Token::Extends) {
            self.parse_extends();
        }

        // Parse module body
        loop {
            self.skip_trivia();
            if self.at_eof() || self.at(Token::ModuleEnd) {
                break;
            }
            // Check for nested submodule end
            if self.at(Token::ModuleEnd) {
                break;
            }
            self.parse_unit();
        }

        // Module end delimiter (====)
        if self.at(Token::ModuleEnd) {
            self.bump_skip_trivia();
        }

        self.finish_node();
    }

    /// Parse a module unit (declaration, definition, etc.)
    fn parse_unit(&mut self) {
        self.skip_trivia();

        if self.at_eof() || self.at(Token::ModuleEnd) {
            return;
        }

        match self.current() {
            Some(Token::ModuleStart) => {
                // Could be separator line or start of submodule
                // Check if MODULE follows (submodule definition)
                // Note: we only look past ONE ModuleStart token because the lexer
                // skips whitespace, so we can't distinguish "same line" from "different line"
                // A submodule has: ---- MODULE Name ----
                // A separator has: ---- (alone on a line)
                let checkpoint = self.pos;
                self.pos += 1; // Skip ONE ModuleStart token
                self.skip_trivia(); // Skip any comments
                if self.at(Token::Module) {
                    // This is a submodule definition
                    self.pos = checkpoint;
                    self.parse_submodule();
                } else {
                    // Just a separator line - consume only this one ModuleStart
                    self.pos = checkpoint;
                    self.start_node(SyntaxKind::Separator);
                    self.bump(); // Consume the ModuleStart token
                    self.skip_trivia(); // Skip trivia for next parse
                    self.finish_node();
                }
            }
            Some(Token::Variable) => self.parse_variable_decl(),
            Some(Token::Constant) => self.parse_constant_decl(),
            Some(Token::Recursive) => self.parse_recursive_decl(),
            Some(Token::Assume) => self.parse_assume(),
            Some(Token::Theorem)
            | Some(Token::Lemma)
            | Some(Token::Proposition)
            | Some(Token::Corollary) => {
                self.parse_theorem();
            }
            Some(Token::Instance) => self.parse_instance(),
            Some(Token::Local) => {
                // LOCAL Op == ... or LOCAL INSTANCE
                self.bump_skip_trivia();
                match self.current() {
                    Some(Token::Instance) => self.parse_instance(),
                    Some(Token::Ident) => self.parse_operator_def(),
                    _ => {
                        self.error(
                            "expected INSTANCE or operator definition after LOCAL".to_string(),
                        );
                    }
                }
            }
            Some(Token::Use) | Some(Token::Hide) => {
                // Module-level USE/HIDE statement (TLAPS)
                self.parse_module_use();
            }
            Some(Token::Ident) => self.parse_operator_def(),
            // Standard library tokens can be operator names in standard modules (e.g., Seq in Sequences.tla)
            Some(op) if is_stdlib_operator_name(op) => self.parse_stdlib_operator_def(),
            // Prefix operator definitions: -. a == ..., ~ a == ...
            Some(op) if is_prefix_op_symbol(op) => self.parse_prefix_operator_def(),
            _ => {
                // Skip unrecognized tokens
                self.error(format!("unexpected token: {:?}", self.current()));
                self.bump_skip_trivia();
            }
        }
    }

    /// Parse module-level USE/HIDE statement (TLAPS)
    fn parse_module_use(&mut self) {
        self.start_node(SyntaxKind::UseStmt);
        self.bump_skip_trivia(); // USE or HIDE
        self.parse_proof_hints();
        self.finish_node();
    }

    /// Parse VARIABLE x, y, z
    fn parse_variable_decl(&mut self) {
        self.start_node(SyntaxKind::VariableDecl);
        self.bump_skip_trivia(); // VARIABLE(S)

        self.parse_name_list();

        self.finish_node();
    }

    /// Parse CONSTANT c1, c2(_, _)
    fn parse_constant_decl(&mut self) {
        self.start_node(SyntaxKind::ConstantDecl);
        self.bump_skip_trivia(); // CONSTANT(S)

        // Parse comma-separated constant declarations
        self.parse_constant_list();

        self.finish_node();
    }

    /// Parse a module name (required), which can start with a number (e.g., 2PCwithBTM)
    fn parse_module_name(&mut self) {
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
        } else if self.at(Token::Number) {
            // Module names like "2PCwithBTM" tokenize as Number + Ident
            self.bump_skip_trivia(); // Number
            if self.at(Token::Ident) {
                self.bump_skip_trivia(); // Rest of identifier
            }
        } else {
            self.error("expected module name".to_string());
        }
    }

    /// Parse a module name (optional), which can start with a number
    fn parse_module_name_optional(&mut self) {
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
        } else if self.at(Token::Number) {
            // Module names like "2PCwithBTM" tokenize as Number + Ident
            self.bump_skip_trivia(); // Number
            if self.at(Token::Ident) {
                self.bump_skip_trivia(); // Rest of identifier
            }
        }
        // If neither, just return without error
    }

    /// Parse a list of names: x, y, z
    fn parse_name_list(&mut self) {
        self.start_node(SyntaxKind::NameList);

        if self.at(Token::Ident) {
            self.bump_skip_trivia();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                if self.at(Token::Ident) {
                    self.bump_skip_trivia();
                } else {
                    self.error("expected identifier after comma".to_string());
                    break;
                }
            }
        }

        self.finish_node();
    }

    /// Parse constant declarations: c1, c2(_, _)
    fn parse_constant_list(&mut self) {
        if self.at(Token::Ident) {
            self.parse_constant_item();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                self.parse_constant_item();
            }
        }
    }

    /// Parse a single constant declaration (possibly with arity)
    fn parse_constant_item(&mut self) {
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
            // Check for arity: C(_, _)
            if self.at(Token::LParen) {
                self.bump_skip_trivia();
                while self.at(Token::Underscore) || self.at(Token::Comma) {
                    self.bump_skip_trivia();
                }
                self.expect(Token::RParen);
            }
        }
    }

    /// Parse RECURSIVE declaration: RECURSIVE Op(_, _)
    /// This declares that an operator will be recursively defined
    fn parse_recursive_decl(&mut self) {
        self.start_node(SyntaxKind::RecursiveDecl);
        self.bump_skip_trivia(); // RECURSIVE

        // Parse comma-separated operator signatures
        self.parse_recursive_op_signature();
        while self.at(Token::Comma) {
            self.bump_skip_trivia();
            self.parse_recursive_op_signature();
        }

        self.finish_node();
    }

    /// Parse a single recursive operator signature: Op(_, _)
    fn parse_recursive_op_signature(&mut self) {
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
            // Optional arity: (_, _, _)
            if self.at(Token::LParen) {
                self.bump_skip_trivia();
                while self.at(Token::Underscore) || self.at(Token::Comma) {
                    self.bump_skip_trivia();
                }
                self.expect(Token::RParen);
            }
        }
    }

    /// Parse ASSUME expr
    fn parse_assume(&mut self) {
        self.start_node(SyntaxKind::AssumeStmt);
        self.bump_skip_trivia(); // ASSUME

        // Optional name
        if self.at(Token::Ident) && self.peek() == Some(Token::DefEq) {
            self.bump_skip_trivia(); // name
            self.bump_skip_trivia(); // ==
        }

        self.parse_expr();

        self.finish_node();
    }

    /// Parse THEOREM/LEMMA/PROPOSITION/COROLLARY
    fn parse_theorem(&mut self) {
        self.start_node(SyntaxKind::TheoremStmt);
        self.bump_skip_trivia(); // THEOREM etc.

        // Optional name
        if self.at(Token::Ident) && self.peek() == Some(Token::DefEq) {
            self.bump_skip_trivia(); // name
            self.bump_skip_trivia(); // ==
        }

        if self.at(Token::Assume) {
            // TLAPS-style theorems/lemmas can be written as:
            //   LEMMA Foo ==
            //     ASSUME ...
            //     PROVE  ...
            self.parse_assume_prove_stmt();
        } else {
            self.parse_expr();
        }

        // Skip trivia before checking for proof start
        self.skip_trivia();

        // Optional PROOF - can start with PROOF, BY, OBVIOUS, OMITTED, or a step label (<n>)
        if self.at(Token::Proof)
            || self.at(Token::By)
            || self.at(Token::Obvious)
            || self.at(Token::Omitted)
            || self.at(Token::Lt)
        {
            self.parse_proof();
        }

        self.finish_node();
    }

    /// Parse INSTANCE Module WITH ... (as declaration)
    fn parse_instance(&mut self) {
        self.start_node(SyntaxKind::InstanceDecl);
        self.parse_instance_body();
        self.finish_node();
    }

    /// Parse INSTANCE Module WITH ... (as expression)
    fn parse_instance_expr(&mut self) {
        self.start_node(SyntaxKind::InstanceDecl);
        self.parse_instance_body();
        self.finish_node();
    }

    /// Parse the body of an INSTANCE (shared between declaration and expression contexts)
    fn parse_instance_body(&mut self) {
        self.bump_skip_trivia(); // INSTANCE

        // Module name (can start with number like 2PCwithBTM)
        self.parse_module_name();

        // WITH substitutions
        if self.at(Token::With) {
            self.bump_skip_trivia();
            self.parse_substitution_list();
        }
    }

    /// Parse substitution list: x <- e, y <- f
    fn parse_substitution_list(&mut self) {
        self.parse_substitution();
        while self.at(Token::Comma) {
            self.bump_skip_trivia();
            self.parse_substitution();
        }
    }

    /// Parse single substitution: x <- e
    fn parse_substitution(&mut self) {
        self.start_node(SyntaxKind::Substitution);

        if self.at(Token::Ident) {
            self.bump_skip_trivia();
        }

        // <- (single token)
        self.expect(Token::LArrow);

        self.parse_expr();

        self.finish_node();
    }

    /// Parse operator definition: Op(x, y) == body
    /// Also handles:
    /// - Infix: a | b == body
    /// - Infix with parenthesized op: a (+) b == body
    /// - Function def: f[x \in S] == body
    fn parse_operator_def(&mut self) {
        self.start_node(SyntaxKind::OperatorDef);

        // Operator name
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
        }

        // Check what comes next to determine form
        match self.current() {
            // Could be parameter list OR parenthesized operator definition
            // a (+) b == body vs Op(x, y) == body
            Some(Token::LParen) => {
                // Lookahead to check if this is a parenthesized operator: (op)
                if self.is_parenthesized_infix_op() {
                    // Infix operator definition with parenthesized op: a (+) b == body
                    self.bump_skip_trivia(); // (
                    self.bump_skip_trivia(); // operator symbol
                    self.expect(Token::RParen); // )
                                                // Second operand name
                    if self.at(Token::Ident) {
                        self.bump_skip_trivia();
                    }
                } else {
                    // Standard parameter list: Op(x, y) == body
                    self.parse_param_list();
                }
            }
            // Function definition: f[x \in S] == body
            Some(Token::LBracket) => {
                self.parse_func_def_params();
            }
            // Postfix operator definition: L^+ == body, L^* == body
            // These define Kleene plus/star operators (but NOT a ^ b which is infix)
            Some(Token::Caret) => {
                // Peek ahead to see if this is postfix (^+ or ^*) or infix (^ b)
                let checkpoint = self.pos;
                self.pos += 1;
                self.skip_trivia_no_emit();
                if self.at(Token::Plus) || self.at(Token::Star) {
                    // It's a postfix operator definition
                    self.pos = checkpoint;
                    self.bump_skip_trivia(); // ^
                    self.bump_skip_trivia(); // + or *
                } else if self.at(Token::Ident) {
                    // It's an infix operator definition: a ^ b == body
                    self.pos = checkpoint;
                    self.bump_skip_trivia(); // ^
                    self.bump_skip_trivia(); // b (second operand name)
                } else {
                    self.pos = checkpoint;
                }
            }
            // Infix operator: a OP b == body (where OP is an operator symbol)
            Some(op) if is_infix_op_symbol(op) => {
                self.bump_skip_trivia(); // operator symbol
                if self.at(Token::Ident) {
                    self.bump_skip_trivia(); // second operand name
                }
            }
            _ => {
                // No parameters, just name == body
            }
        }

        // ==
        if self.at(Token::DefEq) || self.at(Token::TriangleEq) {
            self.bump_skip_trivia();
        } else {
            self.error("expected == in operator definition".to_string());
        }

        // Body expression
        self.parse_expr();

        self.finish_node();
    }

    /// Parse prefix operator definition: -. a == 0 - a
    /// Pattern: <prefix-op>[.] <param> == <body>
    fn parse_prefix_operator_def(&mut self) {
        self.start_node(SyntaxKind::OperatorDef);

        // Consume the operator symbol (e.g., -, ~, [], <>)
        self.bump_skip_trivia();

        // Check for optional dot (e.g., -. for unary minus)
        if self.at(Token::Dot) {
            self.bump_skip_trivia();
        }

        // Parse the parameter name
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
        } else {
            self.error("expected parameter name in prefix operator definition".to_string());
        }

        // ==
        if self.at(Token::DefEq) || self.at(Token::TriangleEq) {
            self.bump_skip_trivia();
        } else {
            self.error("expected == in prefix operator definition".to_string());
        }

        // Body expression
        self.parse_expr();

        self.finish_node();
    }

    /// Parse operator definition for standard library tokens (Seq, Head, Tail, etc.)
    /// These tokens are keywords but in standard modules they're used as operator names
    fn parse_stdlib_operator_def(&mut self) {
        self.start_node(SyntaxKind::OperatorDef);

        // Consume the stdlib token as the operator name
        self.bump_skip_trivia();

        // Check for parameters
        if self.at(Token::LParen) {
            self.parse_param_list();
        }

        // ==
        if self.at(Token::DefEq) || self.at(Token::TriangleEq) {
            self.bump_skip_trivia();
        } else {
            self.error("expected == in operator definition".to_string());
        }

        // Body expression
        self.parse_expr();

        self.finish_node();
    }

    /// Parse operator definition in DEFINE context, allowing underscore-prefixed names
    /// e.g., DEFINE _m == m+1
    fn parse_define_operator(&mut self) {
        // For underscore-prefixed names, we need special handling
        if self.at(Token::Underscore) && self.peek() == Some(Token::Ident) {
            self.start_node(SyntaxKind::OperatorDef);
            self.bump_skip_trivia(); // _
            self.bump_skip_trivia(); // identifier

            // Optional parameters
            if self.at(Token::LParen) {
                self.parse_param_list();
            }

            // ==
            if self.at(Token::DefEq) || self.at(Token::TriangleEq) {
                self.bump_skip_trivia();
            } else {
                self.error("expected == in operator definition".to_string());
            }

            // Body expression
            self.parse_expr();

            self.finish_node();
        } else {
            // Normal identifier - use standard operator definition parser
            self.parse_operator_def();
        }
    }

    /// Parse function definition parameters: [x \in S, y \in T]
    fn parse_func_def_params(&mut self) {
        self.start_node(SyntaxKind::ArgList);
        self.bump_skip_trivia(); // [

        if !self.at(Token::RBracket) {
            self.parse_bound_var();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                self.parse_bound_var();
            }
        }

        self.expect(Token::RBracket);
        self.finish_node();
    }

    /// Parse parameter list: (x, y, Op(_, _))
    fn parse_param_list(&mut self) {
        self.start_node(SyntaxKind::ArgList);
        self.bump_skip_trivia(); // (

        if !self.at(Token::RParen) {
            self.parse_param();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                self.parse_param();
            }
        }

        self.expect(Token::RParen);
        self.finish_node();
    }

    /// Parse a single parameter (possibly higher-order)
    /// Handles:
    /// - Simple: x
    /// - Underscore-prefixed: _n (for DEFINE P(_n) == ...)
    /// - Higher-order: F(_) or F(_, _)
    /// - Infix operator param: _+_ (for LET IsAbelianGroup(G, Id, _+_) == ...)
    fn parse_param(&mut self) {
        self.start_node(SyntaxKind::OperatorParam);

        if self.at(Token::Underscore) {
            self.bump_skip_trivia(); // first _
            if let Some(next) = self.current() {
                if next == Token::Ident {
                    // Underscore-prefixed identifier: _n
                    self.bump_skip_trivia();
                } else if is_infix_op_symbol(next) {
                    // Could be _op_ style infix operator parameter
                    self.bump_skip_trivia(); // operator
                    if self.at(Token::Underscore) {
                        self.bump_skip_trivia(); // second _
                    }
                }
                // Otherwise just a bare underscore placeholder
            }
        } else if self.at(Token::Ident) {
            self.bump_skip_trivia();
            // Check for arity
            if self.at(Token::LParen) {
                self.bump_skip_trivia();
                while self.at(Token::Underscore) || self.at(Token::Comma) {
                    self.bump_skip_trivia();
                }
                self.expect(Token::RParen);
            }
        }

        self.finish_node();
    }

    /// Parse a proof
    fn parse_proof(&mut self) {
        self.start_node(SyntaxKind::Proof);

        match self.current() {
            Some(Token::Obvious) => {
                self.bump_skip_trivia();
            }
            Some(Token::Omitted) => {
                self.bump_skip_trivia();
            }
            Some(Token::By) => {
                self.parse_by_clause();
            }
            Some(Token::Proof) => {
                self.bump_skip_trivia();
                // After PROOF, we can have:
                // - BY clause (leaf proof): PROOF BY ...
                // - OBVIOUS (leaf proof): PROOF OBVIOUS
                // - OMITTED (leaf proof): PROOF OMITTED
                // - Step labels (structured proof): PROOF <1>...
                match self.current() {
                    Some(Token::By) => self.parse_by_clause(),
                    Some(Token::Obvious) => {
                        self.bump_skip_trivia();
                    }
                    Some(Token::Omitted) => {
                        self.bump_skip_trivia();
                    }
                    _ => self.parse_proof_steps(),
                }
            }
            Some(Token::Lt) => {
                // Proof starts directly with step labels (no PROOF keyword)
                self.parse_proof_steps();
            }
            _ => {}
        }

        self.finish_node();
    }

    /// Parse proof steps and QED
    fn parse_proof_steps(&mut self) {
        // Parse all step labels
        self.skip_trivia();
        while self.at(Token::Lt) {
            self.parse_proof_step();
            self.skip_trivia();
        }
        // QED at the end
        if self.at(Token::Qed) {
            self.bump_skip_trivia();
            if self.at(Token::By)
                || self.at(Token::Obvious)
                || self.at(Token::Omitted)
                || self.at(Token::Lt)
            {
                self.parse_proof();
            }
        }
    }

    fn at_prove_kw(&self) -> bool {
        self.current() == Some(Token::Ident) && self.current_text() == Some("PROVE")
    }

    /// Check if current position is a proof-local definition: P(x) == ...
    /// Looks for pattern: Ident (params) == OR Ident ==
    fn is_proof_local_definition(&self) -> bool {
        if self.current() != Some(Token::Ident) {
            return false;
        }
        let mut pos = self.pos + 1;

        // Skip trivia
        while pos < self.tokens.len() && self.tokens[pos].kind.is_trivia() {
            pos += 1;
        }

        // Check for == directly (simple definition: P == ...)
        if pos < self.tokens.len() && self.tokens[pos].kind == Token::DefEq {
            return true;
        }

        // Check for (params) ==
        if pos < self.tokens.len() && self.tokens[pos].kind == Token::LParen {
            pos += 1;
            let mut depth = 1;
            while pos < self.tokens.len() && depth > 0 {
                match self.tokens[pos].kind {
                    Token::LParen => depth += 1,
                    Token::RParen => depth -= 1,
                    _ => {}
                }
                pos += 1;
            }
            // Skip trivia after )
            while pos < self.tokens.len() && self.tokens[pos].kind.is_trivia() {
                pos += 1;
            }
            // Check for ==
            if pos < self.tokens.len() && self.tokens[pos].kind == Token::DefEq {
                return true;
            }
        }

        false
    }

    /// Parse BY clause
    fn parse_by_clause(&mut self) {
        self.start_node(SyntaxKind::ByClause);
        self.bump_skip_trivia(); // BY

        // Parse hints
        self.parse_proof_hints();

        self.finish_node();
    }

    /// Parse proof hints
    fn parse_proof_hints(&mut self) {
        // Parse proof hints (facts/lemmas, step references, DEF/DEFS, etc.).
        // Notes:
        // - DEF/DEFS can appear without comma separator: `BY PTL DEF Spec`
        // - Hints can span multiple lines until the next proof step label.
        loop {
            self.skip_trivia();

            if self.at(Token::Only) {
                // `BY ONLY ...` modifier (no comma required after ONLY)
                self.bump_skip_trivia();
                continue;
            }

            if self.at(Token::Lt) && !self.is_step_label_start() {
                // Step reference: <n>label (e.g., <1>a, <2>2) - NOT a step label start.
                self.parse_step_ref();
            } else if self.at(Token::Defs) || self.at(Token::Def) {
                self.parse_def_clause();
            } else {
                let Some(token) = self.current() else { break };
                if !can_start_expr(token) {
                    break;
                }
                self.parse_expr();
            }

            self.skip_trivia();
            if self.at(Token::Comma) {
                self.bump_skip_trivia();
                continue;
            }
            if self.at(Token::Defs) || self.at(Token::Def) {
                // DEF/DEFS can follow without comma
                continue;
            }
            break;
        }
    }

    fn parse_def_clause(&mut self) {
        self.bump_skip_trivia(); // DEF/DEFS

        // DEF/DEFS can list identifiers, operator symbols, and module refs (M!Op).
        while self.parse_def_name() {
            if self.at(Token::Comma) {
                self.bump_skip_trivia();
            } else {
                break;
            }
        }
    }

    fn parse_def_name(&mut self) -> bool {
        match self.current() {
            Some(Token::Underscore) if self.peek() == Some(Token::Ident) => {
                self.bump_skip_trivia(); // _
                self.bump_skip_trivia(); // ident
            }
            Some(Token::Ident) => {
                self.bump_skip_trivia();
                // Module reference in DEF list: M!Op
                if self.at(Token::Bang) {
                    self.bump_skip_trivia(); // !
                    if self.at(Token::Ident) {
                        self.bump_skip_trivia();
                    }
                }
            }
            Some(op) if is_infix_op_symbol(op) => {
                self.bump_skip_trivia();
            }
            Some(
                Token::Union
                | Token::Intersect
                | Token::SetMinus
                | Token::Times
                | Token::In_
                | Token::NotIn
                | Token::Subseteq
                | Token::Subset
                | Token::Supseteq
                | Token::Supset
                | Token::And
                | Token::Or
                | Token::Not
                | Token::Implies
                | Token::Equiv,
            ) => {
                self.bump_skip_trivia();
            }
            _ => {
                return false;
            }
        }

        true
    }

    /// Parse a step reference: <n>label (used in BY clauses)
    fn parse_step_ref(&mut self) {
        self.start_node(SyntaxKind::StepLabel);
        self.bump_skip_trivia(); // <
        if self.at(Token::Number) {
            self.bump_skip_trivia();
        }
        if self.at(Token::Gt) {
            self.bump_skip_trivia();
        }
        // Optional label (letter or number after >)
        if self.at(Token::Ident) || self.at(Token::Number) {
            self.bump_skip_trivia();
        }
        self.finish_node();
    }

    /// Parse a proof step: <n>label. assertion
    fn parse_proof_step(&mut self) {
        self.start_node(SyntaxKind::ProofStep);

        // Step label <n>
        if self.at(Token::Lt) {
            self.start_node(SyntaxKind::StepLabel);
            self.bump_skip_trivia(); // <
            if self.at(Token::Number) {
                self.bump_skip_trivia();
            }
            self.expect(Token::Gt);
            // Optional label name or number (e.g., <1>a or <1>1).
            // Handle cases:
            // - <1>a. or <1>1. - labeled step with explicit dot
            // - <1>a TAKE ... - labeled step without dot
            // - <1>1 \A z ... - numbered label followed by expression
            // - <1> P(b) == ... - proof-local definition (NOT a label)
            // - <1> P == ... - simple definition (NOT a label)
            if self.at(Token::Ident) || self.at(Token::Number) {
                let after = self.peek();
                // If current is Ident (not Number), we need to be careful:
                // - <1> P(b) == ... - the P is NOT a label (it's a definition name)
                // - <1>a TAKE ... - the a IS a label
                // We don't consume Ident if followed by ( or ==, which indicates a definition.
                let is_ident = self.at(Token::Ident);
                let is_definition_start =
                    is_ident && (after == Some(Token::LParen) || after == Some(Token::DefEq));

                let is_label = !is_definition_start
                    && (after == Some(Token::Dot) || after.is_some_and(is_proof_step_body_start));
                if is_label {
                    self.bump_skip_trivia();
                }
            }
            // Dot is optional in many real-world proofs (e.g., `<1> USE ...`, `<1>1 ...`).
            if self.at(Token::Dot) {
                self.bump_skip_trivia();
            }
            self.finish_node();
        }

        // Step kind
        match self.current() {
            Some(Token::Suffices) => {
                self.bump_skip_trivia();
                if self.at(Token::Assume) {
                    self.parse_assume_prove_stmt();
                } else {
                    self.parse_expr();
                }
            }
            Some(Token::Have) => {
                self.bump_skip_trivia();
                self.parse_expr();
            }
            Some(Token::Take) => {
                self.bump_skip_trivia();
                self.parse_bound_var_list();
            }
            Some(Token::Witness) => {
                self.bump_skip_trivia();
                self.parse_expr();
                while self.at(Token::Comma) {
                    self.bump_skip_trivia();
                    self.parse_expr();
                }
            }
            Some(Token::Pick) => {
                self.bump_skip_trivia();
                self.parse_bound_var_list();
                self.expect(Token::Colon);
                self.parse_expr();
            }
            Some(Token::Use) | Some(Token::Hide) => {
                self.bump_skip_trivia();
                self.parse_proof_hints();
            }
            Some(Token::Assume) => {
                self.bump_skip_trivia(); // ASSUME
                self.parse_assume_prove_stmt();
            }
            Some(Token::Case) => {
                // Proof CASE step: `CASE expr` (not CASE expression).
                self.bump_skip_trivia(); // CASE
                self.parse_expr();
            }
            Some(Token::Define) => {
                self.bump_skip_trivia();
                // DEFINE can have multiple definitions, including underscore-prefixed names
                while self.at(Token::Ident)
                    || (self.at(Token::Underscore) && self.peek() == Some(Token::Ident))
                {
                    self.parse_define_operator();
                }
            }
            Some(Token::Qed) => {
                self.bump_skip_trivia();
            }
            Some(Token::Ident) if self.is_proof_local_definition() => {
                // Proof-local definition without DEFINE keyword: <1> P(x) == expr
                self.parse_define_operator();
            }
            _ => {
                // Regular assertion
                self.parse_expr();
            }
        }

        // Optional proof for this step
        if self.at(Token::Proof)
            || self.at(Token::By)
            || self.at(Token::Obvious)
            || self.at(Token::Omitted)
        {
            self.parse_proof();
        }

        self.finish_node();
    }

    fn parse_assume_prove_stmt(&mut self) {
        // Optional leading ASSUME keyword (present in TLAPS-style theorem statements and SUFFICES).
        if self.at(Token::Assume) {
            self.bump_skip_trivia();
        }

        // Optional NEW declarations immediately after ASSUME.
        if self.at(Token::New) {
            self.parse_new_decl_list();
            if self.at(Token::Comma) {
                self.bump_skip_trivia();
            }
        }

        // Parse assumptions until PROVE (or until we hit proof start / next step label).
        // Parse assumptions until PROVE (or until we hit proof start / next step label).
        while !self.at_eof()
            && !self.at_prove_kw()
            && !self.at(Token::By)
            && !self.at(Token::Obvious)
            && !self.at(Token::Omitted)
            && !self.at(Token::Proof)
            && !(self.at(Token::Lt) && self.is_step_label_start())
        {
            if self.at(Token::New) {
                self.parse_new_decl_list();
            } else if let Some(token) = self.current() {
                if can_start_expr(token) {
                    self.parse_expr();
                } else {
                    // Consume unknown tokens without emitting errors (proof language is permissive).
                    self.bump_skip_trivia();
                }
            } else {
                break;
            }

            if self.at(Token::Comma) {
                self.bump_skip_trivia();
            } else {
                break;
            }
        }

        if self.at_prove_kw() {
            self.bump_skip_trivia(); // PROVE (lexed as Ident)
            if let Some(token) = self.current() {
                if can_start_expr(token) {
                    self.parse_expr();
                }
            }
        }
    }

    fn parse_new_decl_list(&mut self) {
        self.bump_skip_trivia(); // NEW

        // TLAPS frequently writes `NEW x` or `NEW x \\in S` and repeats NEW per declaration.
        // Avoid being overly greedy here: subsequent comma-separated items are often assumptions,
        // e.g., `ASSUME NEW i \\in S, i' = ...`.
        self.parse_new_decl();
    }

    fn parse_new_decl(&mut self) {
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
            // Check for higher-order operator syntax: op(_,_) or op(_)
            if self.at(Token::LParen) {
                self.parse_operator_arity_spec();
            } else if self.at(Token::In_) {
                self.bump_skip_trivia();
                self.parse_expr();
            }
        }
    }

    /// Parse operator arity specification: (_) or (_,_) or (_,_,_) etc.
    fn parse_operator_arity_spec(&mut self) {
        self.bump_skip_trivia(); // (
                                 // Parse underscore-separated placeholders
        while !self.at(Token::RParen) && !self.at_eof() {
            if self.at(Token::Underscore) {
                self.bump_skip_trivia();
            }
            if self.at(Token::Comma) {
                self.bump_skip_trivia();
            } else {
                break;
            }
        }
        if self.at(Token::RParen) {
            self.bump_skip_trivia();
        }
    }

    // === Expression parsing (Pratt parser) ===

    /// Parse an expression
    fn parse_expr(&mut self) {
        self.parse_expr_bp(0);
    }

    /// Parse expression with binding power (Pratt parsing)
    fn parse_expr_bp(&mut self, min_bp: u8) {
        self.skip_trivia();

        // Create checkpoint before parsing LHS for Pratt parsing
        let checkpoint = self.checkpoint();

        // Parse prefix/atom
        self.parse_prefix_or_atom();

        // Parse infix operators
        loop {
            self.skip_trivia();

            let Some(op) = self.current() else { break };

            // Layout-aware bullet list parsing: if we're inside a junction list
            // and see And/Or at the same or left of the junction column, stop parsing.
            // This ensures that outer-level bullets don't get parsed as infix operators
            // inside IF-THEN-ELSE branches or other nested expressions.
            if matches!(op, Token::And | Token::Or) {
                let op_column = self.current_column();
                if let Some(junction_column) = self.junction_context.current_column() {
                    if op_column <= junction_column {
                        break;
                    }
                }
            }

            // Special case: don't treat < as infix if it looks like a proof step label
            // Pattern: < NUMBER > [label] . (e.g., <1>., <1>1., <2>a.)
            if op == Token::Lt && self.is_step_label_start() {
                break;
            }

            // Special case: don't treat - as infix if followed by . (prefix operator definition)
            // Pattern: -. a == ... defines unary minus operator
            if op == Token::Minus && self.peek() == Some(Token::Dot) {
                break;
            }

            // Check for parenthesized infix operator: B (-) C
            if op == Token::LParen && self.is_parenthesized_infix_op() {
                // Get the inner operator to determine binding power
                let inner_op = self.peek_nth(1);
                let Some((l_bp, r_bp)) = inner_op.and_then(infix_binding_power) else {
                    break;
                };

                if l_bp < min_bp {
                    break;
                }

                // Wrap the previous expression in a BinaryExpr
                self.start_node_at(checkpoint, SyntaxKind::BinaryExpr);
                self.bump_skip_trivia(); // (
                self.bump_skip_trivia(); // operator
                self.expect(Token::RParen); // )
                self.parse_expr_bp(r_bp);
                self.finish_node();
                continue;
            }

            let Some((l_bp, r_bp)) = infix_binding_power(op) else {
                break;
            };

            if l_bp < min_bp {
                break;
            }

            // Wrap the previous expression in a BinaryExpr
            self.start_node_at(checkpoint, SyntaxKind::BinaryExpr);

            self.bump_skip_trivia(); // operator
            self.parse_expr_bp(r_bp);
            self.finish_node();
        }
    }

    /// Check if we're at a step reference pattern: <N>label
    /// This is used in proofs to reference the result of a prior step
    fn is_step_reference(&self) -> bool {
        if self.current() != Some(Token::Lt) {
            return false;
        }
        let next1 = self.peek_nth(1);
        let next2 = self.peek_nth(2);
        // Pattern: < NUMBER > followed by optional label
        if next1 == Some(Token::Number) && next2 == Some(Token::Gt) {
            // Check what comes after >
            let next3 = self.peek_nth(3);
            // Valid step references: <1>1, <2>a, <3>, etc.
            // Must be followed by a label (Number or Ident) or end of step ref
            matches!(next3, Some(Token::Number) | Some(Token::Ident) | None)
                || !matches!(
                    next3,
                    Some(Token::Plus) | Some(Token::Minus) | Some(Token::Star) | Some(Token::Slash)
                )
        } else {
            false
        }
    }

    /// Parse a step reference: <N>label (e.g., <4>4, <2>a)
    fn parse_step_reference(&mut self) {
        let cp = self.checkpoint();
        self.start_node(SyntaxKind::StepLabel);
        self.bump_skip_trivia(); // <
        self.bump_skip_trivia(); // number
        self.bump_skip_trivia(); // >
                                 // Optional label (number or identifier)
        if self.at(Token::Number) || self.at(Token::Ident) {
            self.bump_skip_trivia();
        }
        self.finish_node();
        self.parse_postfix(cp);
    }

    /// Parse prefix operator or atom
    fn parse_prefix_or_atom(&mut self) {
        match self.current() {
            // Step reference in expression context: <4>4, <2>a, etc.
            Some(Token::Lt) if self.is_step_reference() => {
                self.parse_step_reference();
            }
            // Prefix operators
            Some(Token::Not) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::Minus) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::Always) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::Eventually) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::Enabled) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::Unchanged) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::Powerset) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::BigUnion) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            Some(Token::Domain) => {
                self.start_node(SyntaxKind::UnaryExpr);
                self.bump_skip_trivia();
                self.parse_prefix_or_atom();
                self.finish_node();
            }
            // Quantifiers (including temporal)
            Some(Token::Forall)
            | Some(Token::Exists)
            | Some(Token::TemporalForall)
            | Some(Token::TemporalExists) => {
                self.parse_quantifier();
            }
            Some(Token::Choose) => {
                self.parse_choose();
            }
            // Control flow
            Some(Token::If) => {
                self.parse_if();
            }
            Some(Token::Case) => {
                self.parse_case();
            }
            Some(Token::Let) => {
                self.parse_let();
            }
            // Parenthesized expression
            Some(Token::LParen) => {
                let cp = self.checkpoint();
                // Check for parenthesized operator reference: (+), (-), etc.
                // These are operator values, not expressions inside parens
                if self.is_parenthesized_infix_op() {
                    self.start_node(SyntaxKind::OperatorRef);
                    self.bump_skip_trivia(); // (
                    self.bump_skip_trivia(); // operator
                    self.expect(Token::RParen); // )
                    self.finish_node();
                    self.parse_postfix(cp);
                } else {
                    self.start_node(SyntaxKind::ParenExpr);
                    self.bump_skip_trivia();
                    self.parse_expr();
                    self.expect(Token::RParen);
                    self.finish_node();
                    self.parse_postfix(cp);
                }
            }
            // Set literal or comprehension
            Some(Token::LBrace) => {
                let cp = self.checkpoint();
                self.parse_set_expr();
                self.parse_postfix(cp);
            }
            // Tuple
            Some(Token::LAngle) => {
                let cp = self.checkpoint();
                self.parse_tuple();
                self.parse_postfix(cp);
            }
            // Function/record expression
            Some(Token::LBracket) => {
                let cp = self.checkpoint();
                self.parse_bracket_expr();
                self.parse_postfix(cp);
            }
            // Lambda
            Some(Token::Lambda) => {
                self.parse_lambda();
            }
            // Literals and identifiers
            Some(Token::True) | Some(Token::False) => {
                let cp = self.checkpoint();
                self.bump_skip_trivia();
                self.parse_postfix(cp);
            }
            Some(Token::Number) => {
                let cp = self.checkpoint();
                self.bump_skip_trivia();
                self.parse_postfix(cp);
            }
            Some(Token::String) => {
                let cp = self.checkpoint();
                self.bump_skip_trivia();
                self.parse_postfix(cp);
            }
            // Underscore-prefixed identifier: _n
            Some(Token::Underscore) if self.peek() == Some(Token::Ident) => {
                let cp = self.checkpoint();
                self.bump_skip_trivia(); // _
                self.bump_skip_trivia(); // identifier
                                         // Check for function application
                if self.at(Token::LParen) {
                    self.start_node(SyntaxKind::ApplyExpr);
                    self.parse_arg_list();
                    self.finish_node();
                }
                self.parse_postfix(cp);
            }
            Some(Token::Ident) => {
                let cp = self.checkpoint();
                // Check for label annotation BEFORE bumping the identifier.
                // Label syntax: P0:: expr (TLAPS syntax for labeling subexpressions)
                // We peek ahead to see if :: follows the identifier.
                if self.peek() == Some(Token::ColonColon) {
                    // Skip the label identifier and :: WITHOUT adding them to the tree.
                    // advance_skip_trivia() adds tokens to the tree, but for labels we just
                    // want to skip them entirely so the labeled expression becomes the result.
                    self.pos += 1; // skip label identifier (don't emit)
                    self.skip_trivia_no_emit(); // skip any trivia (don't emit)
                    self.pos += 1; // skip :: (don't emit)
                    self.skip_trivia_no_emit(); // skip any trivia (don't emit)
                    // Parse the labeled expression (the label is discarded)
                    self.parse_prefix_or_atom();
                    return;
                }
                self.bump_skip_trivia();
                // Some TLAPS-generated names include a trailing `!` as part of the operator name,
                // and are then applied like a normal operator: `IInv!(i)`.
                // Disambiguate this from module references `M!Op` by looking for `!(`.
                if self.at(Token::Bang) && self.peek() == Some(Token::LParen) {
                    self.bump_skip_trivia(); // !
                    self.start_node(SyntaxKind::ApplyExpr);
                    self.parse_arg_list();
                    self.finish_node();
                    self.parse_postfix(cp);
                    return;
                }
                // Check for function application
                // BUT NOT if it's a parenthesized infix operator like (-)
                // In that case, the infix loop will handle it
                if self.at(Token::LParen) && !self.is_parenthesized_infix_op() {
                    // Use checkpoint to include the identifier in the ApplyExpr
                    self.start_node_at(cp, SyntaxKind::ApplyExpr);
                    self.parse_arg_list();
                    self.finish_node();
                }
                self.parse_postfix(cp);
            }
            Some(Token::Boolean) => {
                let cp = self.checkpoint();
                self.bump_skip_trivia();
                self.parse_postfix(cp);
            }
            // @ - used in EXCEPT expressions to refer to current value
            Some(Token::At) => {
                let cp = self.checkpoint();
                self.bump_skip_trivia();
                self.parse_postfix(cp);
            }
            // Standard library operators that take arguments via parentheses
            // These are from the Sequences and other standard modules
            Some(Token::Len)
            | Some(Token::Seq)
            | Some(Token::SubSeq)
            | Some(Token::SelectSeq)
            | Some(Token::Head)
            | Some(Token::Tail)
            | Some(Token::Append) => {
                let cp = self.checkpoint();
                self.bump_skip_trivia();
                // Check for function application
                // Use checkpoint to include the stdlib keyword in the ApplyExpr
                if self.at(Token::LParen) {
                    self.start_node_at(cp, SyntaxKind::ApplyExpr);
                    self.parse_arg_list();
                    self.finish_node();
                }
                self.parse_postfix(cp);
            }
            // Weak/Strong fairness
            Some(Token::WeakFair) | Some(Token::StrongFair) => {
                self.parse_fairness();
            }
            // And/Or at start of expression (TLA+ bullet list pattern)
            // In TLA+, you can write:
            //   /\ x = 0
            //   /\ y = 1
            // which is equivalent to: x = 0 /\ y = 1
            //
            // Layout-aware parsing: bullets at the same column form a list,
            // bullets at greater columns are nested expressions.
            Some(Token::And) | Some(Token::Or) => {
                self.parse_bullet_list();
            }
            // INSTANCE as expression: CC == INSTANCE ClientCentric WITH ...
            Some(Token::Instance) => {
                self.parse_instance_expr();
            }
            // Operators as values (can be passed to higher-order operators)
            // e.g., ReduceSet(\intersect, S, base)
            // Note: Minus is handled above as prefix, but can also be an infix op value
            Some(Token::Intersect)
            | Some(Token::Union)
            | Some(Token::SetMinus)
            | Some(Token::Plus)
            | Some(Token::Star)
            | Some(Token::Slash)
            | Some(Token::Div)
            | Some(Token::Percent)
            | Some(Token::Caret)
            | Some(Token::Eq)
            | Some(Token::Neq)
            | Some(Token::Lt)
            | Some(Token::Gt)
            | Some(Token::Leq)
            | Some(Token::Geq)
            | Some(Token::Prec)
            | Some(Token::Preceq)
            | Some(Token::Succ)
            | Some(Token::Succeq)
            | Some(Token::Implies)
            | Some(Token::Equiv)
            | Some(Token::Concat)
            | Some(Token::In_)
            | Some(Token::NotIn)
            | Some(Token::Subseteq)
            | Some(Token::Subset)
            | Some(Token::Supseteq)
            | Some(Token::Supset) => {
                // Treat operator as an expression (operator reference)
                let cp = self.checkpoint();
                self.start_node(SyntaxKind::OperatorRef);
                self.bump_skip_trivia();
                self.finish_node();
                self.parse_postfix(cp);
            }
            _ => {
                // Unknown token - error recovery
                if !self.at_eof() {
                    self.error(format!("expected expression, found {:?}", self.current()));
                }
            }
        }
    }

    /// Parse postfix operators: prime, function application, field access
    /// checkpoint should be the position before the atom being modified
    fn parse_postfix(&mut self, checkpoint: usize) {
        let mut current_checkpoint = checkpoint;
        loop {
            match self.current() {
                Some(Token::Prime) => {
                    // Wrap the preceding expression in a UnaryExpr
                    self.start_node_at(current_checkpoint, SyntaxKind::UnaryExpr);
                    self.bump_skip_trivia();
                    self.finish_node();
                    // Update checkpoint for chained postfix ops
                    current_checkpoint = checkpoint;
                }
                Some(Token::LBracket) => {
                    // Function application f[x] or f[x,y] (tuple index) - wrap the function expression
                    self.start_node_at(current_checkpoint, SyntaxKind::FuncApplyExpr);
                    self.bump_skip_trivia();
                    self.parse_expr();
                    // Handle tuple indexing: f[x,y,z]
                    while self.at(Token::Comma) {
                        self.bump_skip_trivia();
                        self.parse_expr();
                    }
                    self.expect(Token::RBracket);
                    self.finish_node();
                    current_checkpoint = checkpoint;
                }
                Some(Token::Dot) => {
                    // Record field access r.field - wrap the record expression
                    self.start_node_at(current_checkpoint, SyntaxKind::RecordAccessExpr);
                    self.bump_skip_trivia();
                    if self.at(Token::Ident) {
                        self.bump_skip_trivia();
                    } else {
                        self.error("expected field name".to_string());
                    }
                    self.finish_node();
                    current_checkpoint = checkpoint;
                }
                Some(Token::Bang) => {
                    // Disambiguate:
                    // - Theorem assertion: TheoremName!:
                    // - Module reference: Module!Op or Module!Op(args)
                    // - Module reference to operator symbol: R!+(a, b), R!\leq(a, b)
                    // - TLAPS-generated operator names: Op! (as part of name) applied like `Op!(x)`
                    if self.peek() == Some(Token::Colon) {
                        // Theorem assertion: Name!:
                        self.start_node_at(current_checkpoint, SyntaxKind::TheoremRefExpr);
                        self.bump_skip_trivia(); // !
                        self.bump_skip_trivia(); // :
                        self.finish_node();
                        current_checkpoint = checkpoint;
                        continue;
                    }
                    if self.peek() == Some(Token::LParen) {
                        self.start_node_at(current_checkpoint, SyntaxKind::ApplyExpr);
                        self.bump_skip_trivia(); // !
                        self.parse_arg_list();
                        self.finish_node();
                        current_checkpoint = checkpoint;
                        continue;
                    }

                    self.start_node_at(current_checkpoint, SyntaxKind::ModuleRefExpr);
                    self.bump_skip_trivia(); // !
                    if self.at(Token::Ident) {
                        self.bump_skip_trivia();
                        // Check for function application
                        if self.at(Token::LParen) {
                            self.parse_arg_list();
                        }
                    } else if self.at(Token::Number) {
                        // TLAPS sometimes generates names like `TLANext!1`.
                        self.bump_skip_trivia();
                    } else if self.current().is_some_and(is_module_ref_operator) {
                        // Module reference to operator symbol: R!+(a, b), R!\leq(a, b)
                        self.bump_skip_trivia(); // operator symbol
                                                 // Check for function application
                        if self.at(Token::LParen) {
                            self.parse_arg_list();
                        }
                    } else {
                        // Leave as-is for error recovery; do not emit a hard error here because `!`
                        // is overloaded in multiple TLA+/TLAPS syntactic contexts.
                    }
                    self.finish_node();
                    current_checkpoint = checkpoint;
                }
                Some(Token::Underscore) => {
                    // Action subscript: [A]_v or <<A>>_v or [A]_<<v1, v2>> or [A]_(expr)
                    // Subscript can be identifier, tuple, or parenthesized expression
                    let next = self.peek();
                    if next == Some(Token::Ident)
                        || next == Some(Token::LAngle)
                        || next == Some(Token::LParen)
                    {
                        self.start_node_at(current_checkpoint, SyntaxKind::SubscriptExpr);
                        self.bump_skip_trivia(); // _
                        if self.at(Token::Ident) {
                            // Allow postfix expressions in subscripts, e.g. `[A]_M!vars`.
                            // We parse `M` as an atom and then allow module references and other
                            // postfix operators to attach to it.
                            let sub_cp = self.checkpoint();
                            self.bump_skip_trivia(); // identifier
                            self.parse_postfix(sub_cp);
                        } else if self.at(Token::LAngle) {
                            self.parse_tuple(); // tuple expression
                        } else if self.at(Token::LParen) {
                            // Parenthesized subscript expression: _(expr)
                            self.start_node(SyntaxKind::ParenExpr);
                            self.bump_skip_trivia(); // (
                            self.parse_expr();
                            self.expect(Token::RParen); // )
                            self.finish_node();
                        }
                        self.finish_node();
                        current_checkpoint = checkpoint;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
    }

    /// Parse quantifier: \A x \in S : P or \A x : P
    fn parse_quantifier(&mut self) {
        self.start_node(SyntaxKind::QuantExpr);
        self.bump_skip_trivia(); // \A or \E

        self.parse_bound_var_list();

        self.expect(Token::Colon);
        self.parse_expr();

        self.finish_node();
    }

    /// Parse CHOOSE x \in S : P
    fn parse_choose(&mut self) {
        self.start_node(SyntaxKind::ChooseExpr);
        self.bump_skip_trivia(); // CHOOSE

        self.parse_bound_var();

        self.expect(Token::Colon);
        self.parse_expr();

        self.finish_node();
    }

    /// Parse bound variable list: x \in S, y \in T
    fn parse_bound_var_list(&mut self) {
        self.parse_bound_var();
        while self.at(Token::Comma) {
            self.bump_skip_trivia();
            self.parse_bound_var();
        }
    }

    /// Parse bound variable: x \in S or just x
    /// Note: multiple identifiers sharing a set (e.g. `x, y, z \in S`) are parsed as
    /// multiple BoundVar nodes where only the last one has the domain; the lowerer
    /// propagates the domain backwards.
    /// Also handles tuple patterns: <<x, y>> \in S for destructuring
    fn parse_bound_var(&mut self) {
        self.start_node(SyntaxKind::BoundVar);

        // Check for tuple pattern: <<x, y>> \in S
        if self.at(Token::LAngle) {
            self.parse_tuple_pattern();
        } else if self.at(Token::Ident) {
            self.bump_skip_trivia();
        }

        if self.at(Token::In_) {
            self.bump_skip_trivia();
            // Domain expression can be complex (e.g., 0 .. bound, S \cup T)
            // But must be careful not to consume :, }, ], ) which end bound var lists
            self.parse_domain_expr();
        }

        self.finish_node();
    }

    /// Parse tuple pattern: <<x, y>> for destructuring in quantifiers
    fn parse_tuple_pattern(&mut self) {
        self.start_node(SyntaxKind::TuplePattern);
        self.bump_skip_trivia(); // <<

        if !self.at(Token::RAngle) {
            // Parse first identifier
            if self.at(Token::Ident) {
                self.bump_skip_trivia();
            }
            // Parse remaining identifiers
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                if self.at(Token::Ident) {
                    self.bump_skip_trivia();
                }
            }
        }

        self.expect(Token::RAngle);
        self.finish_node();
    }

    /// Parse a domain expression in a bound variable context
    /// Stops at :, }, ], ), | which typically end the domain
    fn parse_domain_expr(&mut self) {
        self.parse_expr_bp(0);
    }

    /// Parse IF cond THEN a ELSE b
    ///
    /// The condition can contain And/Or (e.g., `IF a /\ b THEN c ELSE d`).
    ///
    /// Layout-aware parsing handles bullet lists in THEN/ELSE branches correctly:
    /// - Bullets inside the branch (at greater column) are parsed as part of the branch
    /// - Bullets at the outer junction list's column stop the branch parsing
    fn parse_if(&mut self) {
        self.start_node(SyntaxKind::IfExpr);
        self.bump_skip_trivia(); // IF

        self.parse_expr(); // condition - can contain And/Or

        self.expect(Token::Then);
        self.parse_expr(); // THEN branch - layout-aware parsing handles bullets

        self.expect(Token::Else);
        self.parse_expr(); // ELSE branch - layout-aware parsing handles bullets

        self.finish_node();
    }

    /// Parse CASE arm1 [] arm2 [] OTHER -> default
    fn parse_case(&mut self) {
        self.start_node(SyntaxKind::CaseExpr);
        self.bump_skip_trivia(); // CASE

        // First arm (no [])
        self.start_node(SyntaxKind::CaseArm);
        self.parse_expr(); // guard
        self.expect(Token::Arrow);
        self.parse_expr(); // body
        self.finish_node();

        // Remaining arms with []
        while self.at(Token::Always) {
            // [] is lexed as Always
            self.bump_skip_trivia();

            if self.at(Token::Other) {
                // OTHER case
                self.bump_skip_trivia();
                self.expect(Token::Arrow);
                self.parse_expr();
                break;
            }

            self.start_node(SyntaxKind::CaseArm);
            self.parse_expr();
            self.expect(Token::Arrow);
            self.parse_expr();
            self.finish_node();
        }

        self.finish_node();
    }

    /// Parse LET defs IN body
    fn parse_let(&mut self) {
        self.start_node(SyntaxKind::LetExpr);
        self.bump_skip_trivia(); // LET

        // Definitions (can include RECURSIVE declarations)
        while !self.at(Token::In) && !self.at_eof() {
            if self.at(Token::Recursive) {
                self.parse_recursive_decl();
            } else if self.at(Token::Ident) {
                self.parse_operator_def();
            } else if self.at(Token::Underscore) && self.peek() == Some(Token::Ident) {
                // Underscore-prefixed identifier: _name
                self.parse_underscore_prefixed_def();
            } else {
                break;
            }
        }

        self.expect(Token::In);
        self.parse_expr();

        self.finish_node();
    }

    /// Parse underscore-prefixed operator definition: _name == expr
    fn parse_underscore_prefixed_def(&mut self) {
        self.start_node(SyntaxKind::OperatorDef);
        self.bump_skip_trivia(); // _
        self.bump_skip_trivia(); // identifier

        // Optional parameters
        if self.at(Token::LParen) {
            self.parse_param_list();
        }

        // ==
        if self.at(Token::DefEq) || self.at(Token::TriangleEq) {
            self.bump_skip_trivia();
        } else {
            self.error("expected == in operator definition".to_string());
        }

        // Body expression
        self.parse_expr();

        self.finish_node();
    }

    /// Parse {a, b, c} or {x \in S : P} or {expr : x \in S}
    fn parse_set_expr(&mut self) {
        self.bump_skip_trivia(); // {

        if self.at(Token::RBrace) {
            // Empty set
            self.start_node(SyntaxKind::SetEnumExpr);
            self.bump_skip_trivia();
            self.finish_node();
            return;
        }

        // Save position for lookahead
        let checkpoint = self.pos;

        // Try to detect set filter: {x \in S : P} or {<<x, y>> \in S : P}
        // Pattern: identifier or tuple pattern followed by \in
        // NOTE: Use skip_trivia_no_emit during lookahead to avoid duplicating tokens
        if self.at(Token::Ident) {
            self.pos += 1;
            self.skip_trivia_no_emit();
            // Handle multiple identifiers: {x,y \in S : P}
            while self.at(Token::Comma) {
                self.pos += 1;
                self.skip_trivia_no_emit();
                if self.at(Token::Ident) {
                    self.pos += 1;
                    self.skip_trivia_no_emit();
                } else {
                    break;
                }
            }
            if self.at(Token::In_) {
                // This is a set filter: {x \in S : P}
                self.pos = checkpoint;
                self.start_node(SyntaxKind::SetFilterExpr);
                self.parse_bound_var();
                if self.at(Token::Colon) {
                    self.bump_skip_trivia();
                    self.parse_expr();
                    self.expect(Token::RBrace);
                    self.finish_node();
                    return;
                }
                // No colon - this must be something else, recover
                self.expect(Token::RBrace);
                self.finish_node();
                return;
            }
            self.pos = checkpoint;
        } else if self.at(Token::LAngle) {
            // Tuple pattern: {<<x, y>> \in S : P}
            // Skip over the tuple pattern to check for \in
            self.pos += 1; // <<
            self.skip_trivia_no_emit();
            // Skip tuple contents (identifiers and commas)
            let mut depth = 1;
            while depth > 0 && !self.at_eof() {
                if self.at(Token::LAngle) {
                    depth += 1;
                } else if self.at(Token::RAngle) {
                    depth -= 1;
                }
                self.pos += 1;
                self.skip_trivia_no_emit();
            }
            if self.at(Token::In_) {
                // This is a set filter with tuple pattern: {<<x, y>> \in S : P}
                self.pos = checkpoint;
                self.start_node(SyntaxKind::SetFilterExpr);
                self.parse_bound_var();
                if self.at(Token::Colon) {
                    self.bump_skip_trivia();
                    self.parse_expr();
                    self.expect(Token::RBrace);
                    self.finish_node();
                    return;
                }
                // No colon - this must be something else, recover
                self.expect(Token::RBrace);
                self.finish_node();
                return;
            }
            self.pos = checkpoint;
        }

        // Either set enumeration {a, b, c} or set map {expr : x \in S}
        // Use checkpoint to allow wrapping as SetBuilderExpr if we see :
        let expr_checkpoint = self.checkpoint();
        self.parse_expr();

        if self.at(Token::Colon) {
            // Set map/builder: {expr : x \in S}
            self.start_node_at(expr_checkpoint, SyntaxKind::SetBuilderExpr);
            self.bump_skip_trivia(); // :
            self.parse_bound_var_list();
            self.expect(Token::RBrace);
            self.finish_node();
            return;
        }

        // Set enumeration: {a, b, c}
        // Wrap from checkpoint as SetEnumExpr
        self.start_node_at(expr_checkpoint, SyntaxKind::SetEnumExpr);
        while self.at(Token::Comma) {
            self.bump_skip_trivia();
            if !self.at(Token::RBrace) {
                self.parse_expr();
            }
        }

        self.expect(Token::RBrace);
        self.finish_node();
    }

    /// Parse <<a, b, c>>
    fn parse_tuple(&mut self) {
        self.start_node(SyntaxKind::TupleExpr);
        self.bump_skip_trivia(); // <<

        if !self.at(Token::RAngle) {
            self.parse_expr();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                self.parse_expr();
            }
        }

        self.expect(Token::RAngle);
        self.finish_node();
    }

    /// Parse [x \in S |-> e] or [f EXCEPT ![a] = b] or [S -> T] or [a |-> 1, b |-> 2] or [a : S, b : T]
    fn parse_bracket_expr(&mut self) {
        self.bump_skip_trivia(); // [

        if self.at(Token::RBracket) {
            // Empty - error
            self.start_node(SyntaxKind::Error);
            self.bump_skip_trivia();
            self.finish_node();
            return;
        }

        // Try to determine type by looking ahead
        // [x \in S |-> e] - function definition
        // [f EXCEPT ![...] = ...] - except expression
        // [S -> T] - function set
        // [a |-> e, ...] - record
        // [a : S, ...] - record set

        let checkpoint = self.pos;

        // Check for EXCEPT: [expr EXCEPT ...] where expr can be any expression
        // The expression before EXCEPT can be:
        // - Simple identifier: [f EXCEPT ...]
        // - @ token: [@ EXCEPT ...]
        // - Indexed access: [f[x] EXCEPT ...]
        // - Record literal: [[a |-> 1, b |-> 2] EXCEPT ...]
        // - Function literal: [[x \in S |-> e] EXCEPT ...]
        // Scan ahead for EXCEPT at depth 1 (inside outer brackets, not nested)
        {
            let mut depth = 0;
            let mut found_except = false;
            while self.pos < self.tokens.len() {
                match self.current() {
                    Some(Token::LBracket)
                    | Some(Token::LParen)
                    | Some(Token::LBrace)
                    | Some(Token::LAngle) => {
                        depth += 1;
                        self.pos += 1;
                    }
                    Some(Token::RBracket)
                    | Some(Token::RParen)
                    | Some(Token::RBrace)
                    | Some(Token::RAngle) => {
                        if depth == 0 {
                            break; // End of outer bracket
                        }
                        depth -= 1;
                        self.pos += 1;
                    }
                    Some(Token::Except) if depth == 0 => {
                        // Found EXCEPT at the right level
                        found_except = true;
                        break;
                    }
                    None => break,
                    _ => {
                        self.pos += 1;
                    }
                }
                self.skip_trivia_no_emit();
            }
            self.pos = checkpoint;

            if found_except {
                self.parse_except_expr();
                return;
            }
        }

        // Check for underscore-prefixed record field: [_field |-> e]
        // Tokenized as Underscore + Ident + MapsTo
        if self.at(Token::Underscore) {
            self.pos += 1;
            self.skip_trivia_no_emit();
            if self.at(Token::Ident) {
                self.pos += 1;
                self.skip_trivia_no_emit();
                if self.at(Token::MapsTo) {
                    self.pos = checkpoint;
                    self.parse_record();
                    return;
                }
            }
            self.pos = checkpoint;
        }

        // Check for x \in S |-> which is function definition
        // Also handles x,y \in S |-> (multiple bound vars)
        // And <<x, y>> \in S |-> (tuple pattern)
        if self.at(Token::Ident) {
            self.pos += 1;
            self.skip_trivia_no_emit();
            // Skip comma-separated identifiers: x,y,z \in S
            while self.at(Token::Comma) {
                self.pos += 1;
                self.skip_trivia_no_emit();
                if self.at(Token::Ident) {
                    self.pos += 1;
                    self.skip_trivia_no_emit();
                } else {
                    break;
                }
            }
            if self.at(Token::In_) {
                // Function definition
                self.pos = checkpoint;
                self.parse_func_def();
                return;
            }
            self.pos = checkpoint;
            // Check for |-> (record) or : (record set) - start over from checkpoint
            self.pos += 1; // skip ident
            self.skip_trivia_no_emit();
            if self.at(Token::MapsTo) {
                self.pos = checkpoint;
                self.parse_record();
                return;
            }
            if self.at(Token::Colon) {
                self.pos = checkpoint;
                self.parse_record_set();
                return;
            }
            self.pos = checkpoint;
        }

        // Check for tuple pattern: [<<x, y>> \in S |-> e]
        if self.at(Token::LAngle) {
            // Skip over the tuple to find \in
            self.pos += 1;
            self.skip_trivia_no_emit();
            let mut depth = 1;
            while depth > 0 && !self.at_eof() {
                if self.at(Token::LAngle) {
                    depth += 1;
                } else if self.at(Token::RAngle) {
                    depth -= 1;
                }
                self.pos += 1;
                self.skip_trivia_no_emit();
            }
            if self.at(Token::In_) {
                // Function definition with tuple pattern
                self.pos = checkpoint;
                self.parse_func_def();
                return;
            }
            self.pos = checkpoint;
        }

        // Must be function set [S -> T] or something else
        self.start_node(SyntaxKind::FuncSetExpr);
        self.parse_expr();

        if self.at(Token::Arrow) {
            self.bump_skip_trivia();
            self.parse_expr();
        }

        self.expect(Token::RBracket);
        self.finish_node();
    }

    /// Parse function definition [x \in S |-> e]
    fn parse_func_def(&mut self) {
        self.start_node(SyntaxKind::FuncDefExpr);

        self.parse_bound_var_list();

        self.expect(Token::MapsTo);
        self.parse_expr();
        self.expect(Token::RBracket);

        self.finish_node();
    }

    /// Parse [f EXCEPT ![a] = b, ![c] = d] or [@ EXCEPT ![a] = b] or [[rec] EXCEPT ...]
    fn parse_except_expr(&mut self) {
        self.start_node(SyntaxKind::ExceptExpr);

        // Parse the expression before EXCEPT
        // This can be any expression: f, @, cache[q], [a |-> 1], [x \in S |-> e], etc.
        self.parse_except_base_expr();

        self.expect(Token::Except);

        // Except specs
        self.parse_except_spec();
        while self.at(Token::Comma) {
            self.bump_skip_trivia();
            self.parse_except_spec();
        }

        self.expect(Token::RBracket);
        self.finish_node();
    }

    /// Parse the base expression in an EXCEPT expression (everything before EXCEPT)
    /// This can be: @, f, f[x], [a |-> 1], [x \in S |-> e], etc.
    fn parse_except_base_expr(&mut self) {
        // Handle @ special case
        if self.at(Token::At) {
            self.bump_skip_trivia();
            return;
        }

        // Handle identifier with optional subscripts and field accesses:
        // f, f[x], f[x][y], f.a, f[x].a, f.a[x].b, node[self].insts[s], etc.
        // Each subscript/field access must wrap the preceding expression in the appropriate node.
        if self.at(Token::Ident) {
            let mut current_checkpoint = self.checkpoint();
            self.bump_skip_trivia();
            loop {
                if self.at(Token::LBracket) {
                    // Wrap preceding expression in FuncApplyExpr: f[x]
                    self.start_node_at(current_checkpoint, SyntaxKind::FuncApplyExpr);
                    self.bump_skip_trivia(); // [
                    self.parse_expr();
                    // Handle tuple indexing: f[x,y,z]
                    while self.at(Token::Comma) {
                        self.bump_skip_trivia();
                        self.parse_expr();
                    }
                    self.expect(Token::RBracket);
                    self.finish_node();
                    current_checkpoint = self.checkpoint();
                } else if self.at(Token::Dot) {
                    // Wrap preceding expression in RecordAccessExpr: r.field
                    self.start_node_at(current_checkpoint, SyntaxKind::RecordAccessExpr);
                    self.bump_skip_trivia(); // .
                    if self.at(Token::Ident) {
                        self.bump_skip_trivia();
                    }
                    self.finish_node();
                    current_checkpoint = self.checkpoint();
                } else {
                    break;
                }
            }
            return;
        }

        // Handle bracket expressions: [a |-> 1] or [x \in S |-> e]
        if self.at(Token::LBracket) {
            // Parse the nested bracket expression
            // We need to consume balanced brackets
            self.bump_skip_trivia(); // [

            // Determine what kind of bracket expression this is
            let checkpoint = self.pos;

            // Check for underscore-prefixed record: _field |->
            if self.at(Token::Underscore) {
                self.pos += 1;
                self.skip_trivia_no_emit();
                if self.at(Token::Ident) {
                    self.pos += 1;
                    self.skip_trivia_no_emit();
                    if self.at(Token::MapsTo) {
                        self.pos = checkpoint;
                        self.start_node(SyntaxKind::RecordExpr);
                        self.parse_record_field();
                        while self.at(Token::Comma) {
                            self.bump_skip_trivia();
                            self.parse_record_field();
                        }
                        self.expect(Token::RBracket);
                        self.finish_node();
                        return;
                    }
                }
                self.pos = checkpoint;
            }

            // Check for record: ident |->
            if self.at(Token::Ident) {
                self.pos += 1;
                self.skip_trivia_no_emit();
                if self.at(Token::MapsTo) {
                    self.pos = checkpoint;
                    self.start_node(SyntaxKind::RecordExpr);
                    self.parse_record_field();
                    while self.at(Token::Comma) {
                        self.bump_skip_trivia();
                        self.parse_record_field();
                    }
                    self.expect(Token::RBracket);
                    self.finish_node();
                    return;
                }
                // Check for function def: ident \in
                if self.at(Token::In_) {
                    self.pos = checkpoint;
                    self.start_node(SyntaxKind::FuncDefExpr);
                    self.parse_bound_var_list();
                    self.expect(Token::MapsTo);
                    self.parse_expr();
                    self.expect(Token::RBracket);
                    self.finish_node();
                    return;
                }
                // Check for record set: ident :
                if self.at(Token::Colon) {
                    self.pos = checkpoint;
                    self.start_node(SyntaxKind::RecordSetExpr);
                    self.parse_record_set_field();
                    while self.at(Token::Comma) {
                        self.bump_skip_trivia();
                        self.parse_record_set_field();
                    }
                    self.expect(Token::RBracket);
                    self.finish_node();
                    return;
                }
            }
            self.pos = checkpoint;

            // Check for tuple pattern in function def: <<x, y>> \in S |-> e
            if self.at(Token::LAngle) {
                // Skip over tuple to find \in
                self.pos += 1;
                self.skip_trivia_no_emit();
                let mut depth = 1;
                while depth > 0 && !self.at_eof() {
                    if self.at(Token::LAngle) {
                        depth += 1;
                    } else if self.at(Token::RAngle) {
                        depth -= 1;
                    }
                    self.pos += 1;
                    self.skip_trivia_no_emit();
                }
                if self.at(Token::In_) {
                    self.pos = checkpoint;
                    self.start_node(SyntaxKind::FuncDefExpr);
                    self.parse_bound_var_list();
                    self.expect(Token::MapsTo);
                    self.parse_expr();
                    self.expect(Token::RBracket);
                    self.finish_node();
                    return;
                }
                self.pos = checkpoint;
            }

            // Default: function set [S -> T] or error
            self.start_node(SyntaxKind::FuncSetExpr);
            self.parse_expr();
            if self.at(Token::Arrow) {
                self.bump_skip_trivia();
                self.parse_expr();
            }
            self.expect(Token::RBracket);
            self.finish_node();
            return;
        }

        // Fallback: parse any expression (should rarely get here)
        self.parse_expr();
    }

    /// Parse ![a][b].c = v or ![a, b] = v (multi-argument function)
    fn parse_except_spec(&mut self) {
        self.start_node(SyntaxKind::ExceptSpec);

        self.expect(Token::Bang);

        // Path elements
        while self.at(Token::LBracket) || self.at(Token::Dot) {
            if self.at(Token::LBracket) {
                self.bump_skip_trivia();
                // Parse first index
                self.parse_expr();
                // Handle multi-argument: ![a, b, c] for f[a, b, c]
                while self.at(Token::Comma) {
                    self.bump_skip_trivia();
                    self.parse_expr();
                }
                self.expect(Token::RBracket);
            } else {
                self.bump_skip_trivia(); // .
                if self.at(Token::Ident) {
                    self.bump_skip_trivia();
                }
            }
        }

        self.expect(Token::Eq);
        self.parse_expr();

        self.finish_node();
    }

    /// Parse record [a |-> 1, b |-> 2]
    fn parse_record(&mut self) {
        self.start_node(SyntaxKind::RecordExpr);

        self.parse_record_field();
        while self.at(Token::Comma) {
            self.bump_skip_trivia();
            self.parse_record_field();
        }

        self.expect(Token::RBracket);
        self.finish_node();
    }

    /// Parse record field: a |-> e or _a |-> e (underscore-prefixed field)
    fn parse_record_field(&mut self) {
        self.start_node(SyntaxKind::RecordField);

        // Handle underscore-prefixed field names: _animator |-> e
        // Tokenized as Underscore + Ident
        if self.at(Token::Underscore) {
            self.bump_skip_trivia();
        }
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
        }
        self.expect(Token::MapsTo);
        self.parse_expr();

        self.finish_node();
    }

    /// Parse record set [a : S, b : T]
    fn parse_record_set(&mut self) {
        self.start_node(SyntaxKind::RecordSetExpr);

        self.parse_record_set_field();
        while self.at(Token::Comma) {
            self.bump_skip_trivia();
            self.parse_record_set_field();
        }

        self.expect(Token::RBracket);
        self.finish_node();
    }

    /// Parse record set field: a : S
    fn parse_record_set_field(&mut self) {
        self.start_node(SyntaxKind::RecordField);

        if self.at(Token::Ident) {
            self.bump_skip_trivia();
        }
        self.expect(Token::Colon);
        self.parse_expr();

        self.finish_node();
    }

    /// Parse LAMBDA x, y : body
    fn parse_lambda(&mut self) {
        self.start_node(SyntaxKind::LambdaExpr);
        self.bump_skip_trivia(); // LAMBDA

        // Parameters
        if self.at(Token::Ident) {
            self.bump_skip_trivia();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                if self.at(Token::Ident) {
                    self.bump_skip_trivia();
                }
            }
        }

        self.expect(Token::Colon);
        self.parse_expr();

        self.finish_node();
    }

    /// Parse WF_vars(A) or SF_vars(A)
    fn parse_fairness(&mut self) {
        self.start_node(SyntaxKind::BinaryExpr);
        self.bump_skip_trivia(); // WF_ or SF_

        // Subscript expression
        self.parse_prefix_or_atom();

        // Action in parentheses
        if self.at(Token::LParen) {
            self.bump_skip_trivia();
            self.parse_expr();
            self.expect(Token::RParen);
        }

        self.finish_node();
    }

    /// Parse a bullet-style conjunction or disjunction list with layout awareness.
    ///
    /// TLA+ allows "bullet-style" lists like:
    /// ```tla
    ///   /\ x = 0
    ///   /\ y = 1
    /// ```
    /// which is equivalent to `x = 0 /\ y = 1`.
    ///
    /// The key insight is that bullets at the same column form a list,
    /// while bullets at greater columns are nested inside the current expression.
    /// This is how SANY/TLC handle it, using a JunctionListContext.
    fn parse_bullet_list(&mut self) {
        // Get the column and type of the initial bullet
        let bullet_column = self.current_column();
        let bullet_type = match self.current() {
            Some(Token::And) => JunctionType::Conjunction,
            Some(Token::Or) => JunctionType::Disjunction,
            _ => return, // Should not happen
        };

        // Start tracking this junction list
        self.junction_context.start(bullet_column, bullet_type);

        // Skip the first bullet (don't add to syntax tree)
        self.advance_skip_trivia();

        // Create a checkpoint before parsing the first expression
        let first_checkpoint = self.checkpoint();

        // Parse the first expression in the list
        // Use parse_expr_bp(0) but stop at bullets at our column level
        self.parse_bullet_list_item();

        // Now check for continuation bullets at the same column
        loop {
            self.skip_trivia();

            // Check if next token is a bullet at the same column and same type
            let next_token = self.current();
            let next_column = self.current_column();

            let is_continuation = matches!(next_token, Some(Token::And) | Some(Token::Or))
                && next_column == bullet_column
                && ((next_token == Some(Token::And) && bullet_type == JunctionType::Conjunction)
                    || (next_token == Some(Token::Or) && bullet_type == JunctionType::Disjunction));

            if !is_continuation {
                break;
            }

            // It's a continuation bullet - wrap previous expression in BinaryExpr
            self.start_node_at(first_checkpoint, SyntaxKind::BinaryExpr);

            // Add the bullet operator to the tree
            self.bump_skip_trivia();

            // Parse the next expression in the list
            self.parse_bullet_list_item();

            self.finish_node();
        }

        // End tracking this junction list
        self.junction_context.end();
    }

    /// Parse a single item within a bullet list.
    /// This parses an expression but stops at bullets at the current junction list's column.
    fn parse_bullet_list_item(&mut self) {
        self.skip_trivia();

        // Create checkpoint for potential binary expression wrapping
        let checkpoint = self.checkpoint();

        // Parse prefix/atom
        self.parse_prefix_or_atom();

        // Parse infix operators, but stop at bullets at our junction list column
        loop {
            self.skip_trivia();

            let Some(op) = self.current() else { break };

            // Layout rule: if we encounter a token at or to the left of the current
            // junction list's bullet column, we've left the current list item.
            //
            // This matches SANY's behavior for patterns like:
            //   /\ /\ A
            //      /\ B
            //      => C
            // which should parse as (A /\ B) => C, not A /\ (B => C).
            let op_column = self.current_column();
            if let Some(junction_column) = self.junction_context.current_column() {
                if op_column <= junction_column {
                    break;
                }
            }

            // If this is And/Or, check if it's a bullet at our junction list's column
            if matches!(op, Token::And | Token::Or) {
                // If we're inside a junction list and this bullet is at or left of that column,
                // stop parsing - it's either a continuation or an outer bullet
                if let Some(junction_column) = self.junction_context.current_column() {
                    if op_column <= junction_column {
                        break;
                    }
                }
            }

            // Special case: don't treat < as infix if it looks like a proof step label
            if op == Token::Lt && self.is_step_label_start() {
                break;
            }

            // Special case: don't treat - as infix if followed by .
            if op == Token::Minus && self.peek() == Some(Token::Dot) {
                break;
            }

            // Check for parenthesized infix operator: B (-) C
            if op == Token::LParen && self.is_parenthesized_infix_op() {
                let inner_op = self.peek_nth(1);
                let Some((_, r_bp)) = inner_op.and_then(infix_binding_power) else {
                    break;
                };

                self.start_node_at(checkpoint, SyntaxKind::BinaryExpr);
                self.bump_skip_trivia(); // (
                self.bump_skip_trivia(); // operator
                self.expect(Token::RParen); // )
                self.parse_bullet_list_item_bp(r_bp);
                self.finish_node();
                continue;
            }

            let Some((_, r_bp)) = infix_binding_power(op) else {
                break;
            };

            // Wrap the previous expression in a BinaryExpr
            self.start_node_at(checkpoint, SyntaxKind::BinaryExpr);
            self.bump_skip_trivia(); // operator
            self.parse_bullet_list_item_bp(r_bp);
            self.finish_node();
        }
    }

    /// Parse expression with binding power for bullet list items.
    /// Similar to parse_expr_bp but respects junction list column boundaries.
    fn parse_bullet_list_item_bp(&mut self, min_bp: u8) {
        self.skip_trivia();

        let checkpoint = self.checkpoint();
        self.parse_prefix_or_atom();

        loop {
            self.skip_trivia();

            let Some(op) = self.current() else { break };

            let op_column = self.current_column();
            if let Some(junction_column) = self.junction_context.current_column() {
                if op_column <= junction_column {
                    break;
                }
            }

            // If this is And/Or, check if it's a bullet at our junction list's column
            if matches!(op, Token::And | Token::Or) {
                if let Some(junction_column) = self.junction_context.current_column() {
                    if op_column <= junction_column {
                        break;
                    }
                }
            }

            if op == Token::Lt && self.is_step_label_start() {
                break;
            }

            if op == Token::Minus && self.peek() == Some(Token::Dot) {
                break;
            }

            if op == Token::LParen && self.is_parenthesized_infix_op() {
                let inner_op = self.peek_nth(1);
                let Some((inner_l_bp, r_bp)) = inner_op.and_then(infix_binding_power) else {
                    break;
                };

                if inner_l_bp < min_bp {
                    break;
                }

                self.start_node_at(checkpoint, SyntaxKind::BinaryExpr);
                self.bump_skip_trivia();
                self.bump_skip_trivia();
                self.expect(Token::RParen);
                self.parse_bullet_list_item_bp(r_bp);
                self.finish_node();
                continue;
            }

            let Some((l_bp, r_bp)) = infix_binding_power(op) else {
                break;
            };

            if l_bp < min_bp {
                break;
            }

            self.start_node_at(checkpoint, SyntaxKind::BinaryExpr);
            self.bump_skip_trivia();
            self.parse_bullet_list_item_bp(r_bp);
            self.finish_node();
        }
    }

    /// Parse argument list: (a, b, c)
    fn parse_arg_list(&mut self) {
        self.start_node(SyntaxKind::ArgList);
        self.bump_skip_trivia(); // (

        if !self.at(Token::RParen) {
            self.parse_expr();
            while self.at(Token::Comma) {
                self.bump_skip_trivia();
                self.parse_expr();
            }
        }

        self.expect(Token::RParen);
        self.finish_node();
    }
}

/// Compute line start offsets from source text
fn compute_line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0]; // Line 0 starts at offset 0
    for (i, c) in source.char_indices() {
        if c == '\n' {
            starts.push(i + 1); // Next line starts after the newline
        }
    }
    starts
}

/// Get column number (0-indexed) for a byte offset given line starts
fn offset_to_column(offset: usize, line_starts: &[usize]) -> u32 {
    // Binary search to find which line this offset is on
    let line = line_starts
        .binary_search(&offset)
        .unwrap_or_else(|insert_point| insert_point.saturating_sub(1));
    let line_start = line_starts.get(line).copied().unwrap_or(0);
    (offset - line_start) as u32
}

/// Lex with position and column information for layout-aware parsing
fn lex_with_positions(source: &str) -> impl Iterator<Item = ParsedToken> + '_ {
    use logos::Logos;
    let line_starts = compute_line_starts(source);
    let lexer = Token::lexer(source);
    lexer.spanned().filter_map(move |(result, span)| {
        result.ok().map(|kind| {
            let column = offset_to_column(span.start, &line_starts);
            ParsedToken {
                kind,
                text: source[span.clone()].to_string(),
                start: span.start as u32,
                column,
            }
        })
    })
}

fn can_start_expr(token: Token) -> bool {
    matches!(
        token,
        Token::Not
            | Token::Minus
            | Token::Always
            | Token::Eventually
            | Token::Enabled
            | Token::Unchanged
            | Token::Powerset
            | Token::BigUnion
            | Token::Domain
            | Token::Forall
            | Token::Exists
            | Token::TemporalForall
            | Token::TemporalExists
            | Token::Choose
            | Token::If
            | Token::Case
            | Token::Let
            | Token::LParen
            | Token::LBrace
            | Token::LAngle
            | Token::LBracket
            | Token::Lambda
            | Token::True
            | Token::False
            | Token::Number
            | Token::String
            | Token::Ident
            | Token::Boolean
            | Token::At
            | Token::Len
            | Token::Seq
            | Token::SubSeq
            | Token::SelectSeq
            | Token::Head
            | Token::Tail
            | Token::Append
            | Token::WeakFair
            | Token::StrongFair
            | Token::And
            | Token::Or
            | Token::Instance
    )
}

/// Check if token is a proof step keyword (not including general expression starters).
/// Used to determine if an identifier after step level is a label name.
fn is_proof_step_body_keyword(token: Token) -> bool {
    matches!(
        token,
        Token::Suffices
            | Token::Have
            | Token::Take
            | Token::Witness
            | Token::Pick
            | Token::Use
            | Token::Hide
            | Token::Define
            | Token::Qed
            | Token::Assume
            // Proof CASE step (not CASE expression)
            | Token::Case
    )
}

fn is_proof_step_body_start(token: Token) -> bool {
    is_proof_step_body_keyword(token) || can_start_expr(token)
}

/// Check if a token can be a prefix operator symbol in prefix operator definitions
/// e.g., -. a == ... defines a prefix operator -.
fn is_prefix_op_symbol(token: Token) -> bool {
    matches!(
        token,
        Token::Minus   // -.  (unary minus, as in -. a == 0 - a)
            | Token::Not   // ~   (logical not)
            | Token::Always   // []  (temporal always)
            | Token::Eventually // <>  (temporal eventually)
    )
}

/// Check if a token can be used as an operator symbol in infix operator definitions
/// e.g., a | b == ... defines an infix operator |
fn is_infix_op_symbol(token: Token) -> bool {
    matches!(
        token,
        // User-definable operators
        Token::Plus
            | Token::Minus
            | Token::Star
            | Token::Slash
            | Token::Div
            | Token::Percent
            | Token::Caret
            | Token::Bang
            | Token::At
            | Token::AtAt
            | Token::Dollar
            | Token::DollarDollar
            | Token::Question
            | Token::Amp
            | Token::Concat
            | Token::ColonGt
            | Token::Turnstile
            | Token::Pipe
            // Multi-character user-definable operators
            | Token::PlusPlus
            | Token::MinusMinus
            | Token::StarStar
            | Token::SlashSlash
            | Token::CaretCaret
            | Token::PercentPercent
            | Token::AmpAmp
            // Circled operators
            | Token::Oplus
            | Token::Ominus
            | Token::Otimes
            | Token::Oslash
            | Token::Odot
            | Token::Uplus
            | Token::Sqcap
            | Token::Sqcup
            // Action composition
            | Token::Cdot
            // BNF production operator
            | Token::ColonColonEq
            // Relational (can be user-defined)
            | Token::Eq
            | Token::Neq
            | Token::Lt
            | Token::Gt
            | Token::Leq
            | Token::Geq
            // Ordering relations (user-definable)
            | Token::Prec
            | Token::Preceq
            | Token::Succ
            | Token::Succeq
            | Token::Ll
            | Token::Gg
            | Token::Sim
            | Token::Simeq
            | Token::Asymp
            | Token::Approx
            | Token::Cong
            | Token::Doteq
            | Token::Propto
            // Set operations (can be overloaded)
            | Token::Union
            | Token::Intersect
            | Token::SetMinus
            | Token::Times
            // Bag/multiset operations
            | Token::Sqsubseteq
            | Token::Sqsupseteq
            // Range operator (user-definable)
            | Token::DotDot
    )
}

/// Check if a token can appear after ! in a module reference
/// e.g., R!+(a, b) references the + operator from module R
fn is_module_ref_operator(token: Token) -> bool {
    matches!(
        token,
        // Arithmetic operators
        Token::Plus
            | Token::Minus
            | Token::Star
            | Token::Slash
            | Token::Percent
            | Token::Caret
            // TLA+ operators
            | Token::Leq
            | Token::Geq
            | Token::Lt
            | Token::Gt
            | Token::Eq
            | Token::Neq
            // Set operations
            | Token::Union
            | Token::Intersect
            | Token::SetMinus
            | Token::Times
            | Token::In_
            | Token::NotIn
            | Token::Subseteq
            | Token::Subset
            | Token::Supseteq
            | Token::Supset
            // Logical operators
            | Token::And
            | Token::Or
            | Token::Not
            | Token::Implies
            | Token::Equiv
            // Ordering relations
            | Token::Prec
            | Token::Preceq
            | Token::Succ
            | Token::Succeq
            // Other user-definable operators
            | Token::Concat
            | Token::Pipe
            | Token::Dollar
            | Token::DollarDollar
            | Token::Question
            | Token::Amp
    )
}

/// Check if a token is a standard library function name that can be used as an operator name
/// These are tokenized as keywords but in standard modules they need to be defined as operators
fn is_stdlib_operator_name(token: Token) -> bool {
    matches!(
        token,
        Token::Seq
            | Token::Append
            | Token::Head
            | Token::Tail
            | Token::Len
            | Token::SubSeq
            | Token::SelectSeq
            | Token::Concat
    )
}

/// Get binding power for infix operators
fn infix_binding_power(op: Token) -> Option<(u8, u8)> {
    let bp = match op {
        // Lowest precedence: equivalence
        Token::Equiv => (1, 2),

        // Implication (right associative)
        Token::Implies => (3, 2),

        // Disjunction
        Token::Or => (5, 6),

        // Conjunction
        Token::And => (7, 8),

        // Comparison
        Token::Eq
        | Token::Neq
        | Token::Lt
        | Token::Gt
        | Token::Leq
        | Token::Geq
        | Token::ColonColonEq => (9, 10),

        // Ordering relations (user-definable, same precedence as comparison)
        Token::Prec
        | Token::Preceq
        | Token::Succ
        | Token::Succeq
        | Token::Ll
        | Token::Gg
        | Token::Sim
        | Token::Simeq
        | Token::Asymp
        | Token::Approx
        | Token::Cong
        | Token::Doteq
        | Token::Propto => (9, 10),

        // Set membership and subset/superset
        Token::In_
        | Token::NotIn
        | Token::Subseteq
        | Token::Subset
        | Token::Supseteq
        | Token::Supset => (11, 12),

        // Bag subset operators (same precedence as set subset)
        Token::Sqsubseteq | Token::Sqsupseteq => (11, 12),

        // Set operations
        Token::Union | Token::Intersect | Token::SetMinus => (13, 14),

        // Range
        Token::DotDot => (15, 16),

        // Additive (including multi-char variants)
        Token::Plus | Token::Minus | Token::PlusPlus | Token::MinusMinus => (17, 18),

        // Multiplicative (including multi-char variants)
        Token::Star
        | Token::Slash
        | Token::Div
        | Token::Percent
        | Token::StarStar
        | Token::SlashSlash
        | Token::PercentPercent => (19, 20),

        // Exponentiation (right associative, including multi-char variant)
        Token::Caret | Token::CaretCaret => (22, 21),

        // Circled operators (same precedence as multiplicative)
        Token::Oplus
        | Token::Ominus
        | Token::Otimes
        | Token::Oslash
        | Token::Odot
        | Token::Uplus
        | Token::Sqcap
        | Token::Sqcup => (19, 20),

        // Action composition (\cdot)
        Token::Cdot => (19, 20),

        // Temporal leads-to
        Token::LeadsTo => (4, 3),

        // Cartesian product
        Token::Times => (15, 16),

        // Function composition
        Token::Concat => (17, 18),

        // String concat, function combine
        // Note: @@ has lower precedence than :> so that 1:>2 @@ 2:>3 parses as (1:>2) @@ (2:>3)
        Token::AtAt => (13, 14),

        // User-defined infix operator (like a | b for divisibility)
        Token::Pipe => (11, 12),

        // Bitwise AND (user-defined, including double variant)
        Token::Amp | Token::AmpAmp => (19, 20),

        // Function mapping :> (from Sequences standard module)
        // Higher precedence than @@ so that 1:>2 @@ 2:>3 parses as (1:>2) @@ (2:>3)
        Token::ColonGt => (15, 16),

        _ => return None,
    };
    Some(bp)
}

/// Convert lexer Token to SyntaxKind
fn token_to_syntax_kind(token: Token) -> SyntaxKind {
    match token {
        Token::ModuleStart => SyntaxKind::ModuleStart,
        Token::ModuleEnd => SyntaxKind::ModuleEnd,
        Token::Module => SyntaxKind::ModuleKw,
        Token::Extends => SyntaxKind::ExtendsKw,
        Token::Instance => SyntaxKind::InstanceKw,
        Token::With => SyntaxKind::WithKw,
        Token::Local => SyntaxKind::LocalKw,
        Token::Variable => SyntaxKind::VariableKw,
        Token::Constant => SyntaxKind::ConstantKw,
        Token::Assume => SyntaxKind::AssumeKw,
        Token::Theorem => SyntaxKind::TheoremKw,
        Token::Lemma => SyntaxKind::LemmaKw,
        Token::Proposition => SyntaxKind::PropositionKw,
        Token::Corollary => SyntaxKind::CorollaryKw,
        Token::Axiom => SyntaxKind::AxiomKw,
        Token::Proof => SyntaxKind::ProofKw,
        Token::By => SyntaxKind::ByKw,
        Token::Obvious => SyntaxKind::ObviousKw,
        Token::Omitted => SyntaxKind::OmittedKw,
        Token::Qed => SyntaxKind::QedKw,
        Token::Suffices => SyntaxKind::SufficesKw,
        Token::Have => SyntaxKind::HaveKw,
        Token::Take => SyntaxKind::TakeKw,
        Token::Witness => SyntaxKind::WitnessKw,
        Token::Pick => SyntaxKind::PickKw,
        Token::Use => SyntaxKind::UseKw,
        Token::Hide => SyntaxKind::HideKw,
        Token::Define => SyntaxKind::DefineKw,
        Token::Defs => SyntaxKind::DefsKw,
        Token::Def => SyntaxKind::DefKw,
        Token::Only => SyntaxKind::OnlyKw,
        Token::New => SyntaxKind::NewKw,
        Token::True => SyntaxKind::TrueKw,
        Token::False => SyntaxKind::FalseKw,
        Token::Boolean => SyntaxKind::BooleanKw,
        Token::If => SyntaxKind::IfKw,
        Token::Then => SyntaxKind::ThenKw,
        Token::Else => SyntaxKind::ElseKw,
        Token::Case => SyntaxKind::CaseKw,
        Token::Other => SyntaxKind::OtherKw,
        Token::Let => SyntaxKind::LetKw,
        Token::In => SyntaxKind::InKw,
        Token::Lambda => SyntaxKind::LambdaKw,
        Token::Forall => SyntaxKind::ForallKw,
        Token::Exists => SyntaxKind::ExistsKw,
        Token::TemporalExists => SyntaxKind::TemporalExistsKw,
        Token::TemporalForall => SyntaxKind::TemporalForallKw,
        Token::Choose => SyntaxKind::ChooseKw,
        Token::Recursive => SyntaxKind::RecursiveKw,
        Token::In_ => SyntaxKind::InOp,
        Token::NotIn => SyntaxKind::NotInOp,
        Token::Union => SyntaxKind::UnionOp,
        Token::Intersect => SyntaxKind::IntersectOp,
        Token::SetMinus => SyntaxKind::SetMinusOp,
        Token::Subseteq => SyntaxKind::SubseteqOp,
        Token::Subset => SyntaxKind::SubsetOp,
        Token::Supseteq => SyntaxKind::SupseteqOp,
        Token::Supset => SyntaxKind::SupsetOp,
        Token::Sqsubseteq => SyntaxKind::SqsubseteqOp,
        Token::Sqsupseteq => SyntaxKind::SqsupseteqOp,
        Token::Powerset => SyntaxKind::PowersetKw,
        Token::BigUnion => SyntaxKind::BigUnionKw,
        Token::BigIntersect => SyntaxKind::BigInterKw,
        Token::Always => SyntaxKind::AlwaysOp,
        Token::Eventually => SyntaxKind::EventuallyOp,
        Token::LeadsTo => SyntaxKind::LeadsToOp,
        Token::Enabled => SyntaxKind::EnabledKw,
        Token::Unchanged => SyntaxKind::UnchangedKw,
        Token::WeakFair => SyntaxKind::WeakFairKw,
        Token::StrongFair => SyntaxKind::StrongFairKw,
        Token::And => SyntaxKind::AndOp,
        Token::Or => SyntaxKind::OrOp,
        Token::Not => SyntaxKind::NotOp,
        Token::Implies => SyntaxKind::ImpliesOp,
        Token::Equiv => SyntaxKind::EquivOp,
        Token::Eq => SyntaxKind::EqOp,
        Token::Neq => SyntaxKind::NeqOp,
        Token::Lt => SyntaxKind::LtOp,
        Token::Gt => SyntaxKind::GtOp,
        Token::Leq => SyntaxKind::LeqOp,
        Token::Geq => SyntaxKind::GeqOp,
        Token::Prec => SyntaxKind::PrecOp,
        Token::Preceq => SyntaxKind::PreceqOp,
        Token::Succ => SyntaxKind::SuccOp,
        Token::Succeq => SyntaxKind::SucceqOp,
        Token::Ll => SyntaxKind::LlOp,
        Token::Gg => SyntaxKind::GgOp,
        Token::Sim => SyntaxKind::SimOp,
        Token::Simeq => SyntaxKind::SimeqOp,
        Token::Asymp => SyntaxKind::AsympOp,
        Token::Approx => SyntaxKind::ApproxOp,
        Token::Cong => SyntaxKind::CongOp,
        Token::Doteq => SyntaxKind::DoteqOp,
        Token::Propto => SyntaxKind::ProptoOp,
        Token::Cdot => SyntaxKind::CdotOp,
        Token::Plus => SyntaxKind::PlusOp,
        Token::Minus => SyntaxKind::MinusOp,
        Token::Star => SyntaxKind::StarOp,
        Token::Slash => SyntaxKind::SlashOp,
        Token::Caret => SyntaxKind::CaretOp,
        Token::Percent => SyntaxKind::PercentOp,
        Token::Div => SyntaxKind::DivOp,
        Token::DotDot => SyntaxKind::DotDotOp,
        // Multi-character operators
        Token::PlusPlus => SyntaxKind::PlusPlusOp,
        Token::MinusMinus => SyntaxKind::MinusMinusOp,
        Token::StarStar => SyntaxKind::StarStarOp,
        Token::SlashSlash => SyntaxKind::SlashSlashOp,
        Token::CaretCaret => SyntaxKind::CaretCaretOp,
        Token::PercentPercent => SyntaxKind::PercentPercentOp,
        Token::AmpAmp => SyntaxKind::AmpAmpOp,
        Token::Oplus => SyntaxKind::OplusOp,
        Token::Ominus => SyntaxKind::OminusOp,
        Token::Otimes => SyntaxKind::OtimesOp,
        Token::Oslash => SyntaxKind::OslashOp,
        Token::Odot => SyntaxKind::OdotOp,
        Token::Uplus => SyntaxKind::UplusOp,
        Token::Sqcap => SyntaxKind::SqcapOp,
        Token::Sqcup => SyntaxKind::SqcupOp,
        Token::DefEq => SyntaxKind::DefEqOp,
        Token::Prime => SyntaxKind::PrimeOp,
        Token::TriangleEq => SyntaxKind::TriangleEqOp,
        Token::LParen => SyntaxKind::LParen,
        Token::RParen => SyntaxKind::RParen,
        Token::LBracket => SyntaxKind::LBracket,
        Token::RBracket => SyntaxKind::RBracket,
        Token::LBrace => SyntaxKind::LBrace,
        Token::RBrace => SyntaxKind::RBrace,
        Token::LAngle => SyntaxKind::LAngle,
        Token::RAngle => SyntaxKind::RAngle,
        Token::Comma => SyntaxKind::Comma,
        Token::ColonColonEq => SyntaxKind::ColonColonEqOp,
        Token::ColonColon => SyntaxKind::ColonColon,
        Token::Colon => SyntaxKind::Colon,
        Token::Semi => SyntaxKind::Semi,
        Token::Dot => SyntaxKind::Dot,
        Token::Underscore => SyntaxKind::Underscore,
        Token::At => SyntaxKind::At,
        Token::Bang => SyntaxKind::Bang,
        Token::MapsTo => SyntaxKind::MapsTo,
        Token::Arrow => SyntaxKind::Arrow,
        Token::LArrow => SyntaxKind::LArrow,
        Token::Turnstile => SyntaxKind::Turnstile,
        Token::Pipe => SyntaxKind::Pipe,
        Token::ColonGt => SyntaxKind::ColonGt,
        Token::AtAt => SyntaxKind::AtAt,
        Token::Dollar => SyntaxKind::Dollar,
        Token::DollarDollar => SyntaxKind::DollarDollar,
        Token::Question => SyntaxKind::Question,
        Token::Amp => SyntaxKind::Amp,
        Token::Times => SyntaxKind::TimesOp,
        Token::Domain => SyntaxKind::DomainKw,
        Token::Except => SyntaxKind::ExceptKw,
        Token::Append => SyntaxKind::AppendKw,
        Token::Head => SyntaxKind::HeadKw,
        Token::Tail => SyntaxKind::TailKw,
        Token::Len => SyntaxKind::LenKw,
        Token::Seq => SyntaxKind::SeqKw,
        Token::SubSeq => SyntaxKind::SubSeqKw,
        Token::SelectSeq => SyntaxKind::SelectSeqKw,
        Token::Concat => SyntaxKind::ConcatOp,
        Token::Number => SyntaxKind::Number,
        Token::String => SyntaxKind::String,
        Token::Ident => SyntaxKind::Ident,
        Token::LineComment => SyntaxKind::LineComment,
        Token::BlockComment => SyntaxKind::BlockComment,
        Token::Whitespace => SyntaxKind::Whitespace,
    }
}

/// Parse TLA+ source and return a syntax tree
pub fn parse(source: &str) -> ParseResult {
    Parser::new(source).parse()
}

/// Get the syntax tree root node
pub fn parse_to_syntax_tree(source: &str) -> SyntaxNode {
    let result = parse(source);
    SyntaxNode::new_root(result.green_node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_module() {
        let source = r#"---- MODULE Test ----
VARIABLE x
Next == x' = x + 1
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
        let tree = SyntaxNode::new_root(result.green_node);
        assert_eq!(tree.kind(), SyntaxKind::Root);
    }

    #[test]
    fn test_parse_extends() {
        let source = r#"---- MODULE Test ----
EXTENDS Naturals, Sequences
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_operator_def() {
        let source = r#"---- MODULE Test ----
Add(a, b) == a + b
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_quantifier() {
        let source = r#"---- MODULE Test ----
AllPositive(S) == \A x \in S : x > 0
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_set_expressions() {
        let source = r#"---- MODULE Test ----
S == {1, 2, 3}
T == {x \in S : x > 1}
U == {x + 1 : x \in S}
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_function_def() {
        let source = r#"---- MODULE Test ----
f == [x \in Nat |-> x * 2]
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_record() {
        let source = r#"---- MODULE Test ----
r == [a |-> 1, b |-> 2]
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_if_then_else() {
        let source = r#"---- MODULE Test ----
Max(a, b) == IF a > b THEN a ELSE b
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_prefix_operator_def() {
        // Test parsing prefix operator definition like -. a == 0 - a
        let source = r#"---- MODULE Integers ----
EXTENDS Naturals
LOCAL R == INSTANCE ProtoReals
Int  ==  R!Int
-. a == 0 - a
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_parenthesized_op_ref() {
        // Test parsing parenthesized operator as a value/reference
        let source = r#"---- MODULE Test ----
Op == (-)
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors for operator ref: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_parenthesized_infix_op() {
        // Test parsing parenthesized infix operators like B (-) C
        let source = r#"---- MODULE Test ----
Result == B (-) C
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_parenthesized_infix_op_in_func_call() {
        // Test parsing parenthesized infix operators inside function calls
        let source = r#"---- MODULE Test ----
Result == SumBag(B (-) SetToBag({e}))
===="#;
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }
}

#[cfg(test)]
mod temporal_tests {
    use super::*;

    #[test]
    fn test_parse_temporal_spec() {
        // Test parsing temporal operators and fairness conditions
        let src = r#"---- MODULE Test ----
VARIABLES x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
===="#;
        let result = Parser::new(src).parse();
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }
}

#[cfg(test)]
mod proof_tests {
    use super::*;

    #[test]
    fn test_parse_theorem_with_proof() {
        // Test parsing theorem with structured proof
        let src = r#"---- MODULE Test ----
THEOREM TypeCorrect == TRUE
<1>1. Init => TypeOK
  BY DEF Init, TypeOK
<1>. QED  BY <1>1
===="#;
        let result = Parser::new(src).parse();
        for err in &result.errors {
            eprintln!("Error at {}-{}: {}", err.start, err.end, err.message);
        }
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_lemma_with_multiple_steps() {
        // Test parsing lemma with multiple proof steps
        let src = r#"---- MODULE Test ----
LEMMA TypeCorrect == Spec => []TypeOK
<1>1. Init => TypeOK
  BY DEF Init, TypeOK
<1>2. TypeOK /\ [Next]_vars => TypeOK'
  BY DEF TypeOK, Next, vars
<1>. QED  BY <1>1, <1>2, PTL DEF Spec
===="#;
        let result = Parser::new(src).parse();
        for err in &result.errors {
            eprintln!("Error at {}-{}: {}", err.start, err.end, err.message);
        }
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_number_prefixed_operator() {
        // Test parsing operators with number-prefixed names like 1aMessage, 2avMessage
        // commonly used in consensus algorithm specs (Paxos, BFT, etc.)
        let source = r#"---- MODULE Test ----
1aMessage == [type : {"1a"}, bal : Ballot]
1bMessage == [type : {"1b"}, bal : Ballot, mbal : Ballot, mval : Value]
2avMessage == [type : {"2av"}, bal : Ballot, val : Value]
1bOr2bMsgs == {m \in bmsgs : m.type \in {"1b", "2b"}}
===="#;
        let result = parse(source);
        for err in &result.errors {
            eprintln!("Error at {}-{}: {}", err.start, err.end, err.message);
        }
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_proof_followed_by_number_def() {
        // Test that number-prefixed operator definitions work after a proof
        let source = r#"---- MODULE Test ----
THEOREM Foo == TRUE
<1>1. Init => TypeOK
  BY DEF Init
<1>. QED  BY <1>1

1bOr2bMsgs == {m \in bmsgs : m.type = "1b"}
===="#;
        let result = parse(source);
        for err in &result.errors {
            eprintln!("Error at {}-{}: {}", err.start, err.end, err.message);
        }
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_set_filter_with_tuple_pattern() {
        // Test set filter with tuple pattern: {<<x, y>> \in S : P}
        let source = r#"---- MODULE Test ----
free == {<<pc, m>> \in moved : m \cap board = {}}
===="#;
        let result = parse(source);
        for err in &result.errors {
            eprintln!("Error at {}-{}: {}", err.start, err.end, err.message);
        }
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_proof_local_definition() {
        // Test proof-local definitions without DEFINE keyword: <1> P(x) == expr
        let source = r#"---- MODULE Test ----
LEMMA L == TRUE
<1> SUFFICES TRUE
  OBVIOUS
<1> P(b) == b > 0
<1>1. \A b : P(b)
  OBVIOUS
<1>. QED BY <1>1
===="#;
        let result = parse(source);
        for err in &result.errors {
            eprintln!("Error at {}-{}: {}", err.start, err.end, err.message);
        }
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_parse_submodule() {
        // Test submodule (inner module) parsing
        let source = r#"------------------------- MODULE Outer --------------------------
VARIABLE now
-----------------------------------------------------------------------------

   -------------------------- MODULE Inner ----------------------------------
   VARIABLE t
   TNext == t' = 0
  ==========================================================================

Op == INSTANCE Inner
===="#;
        let result = parse(source);
        for err in &result.errors {
            eprintln!("Error at {}-{}: {}", err.start, err.end, err.message);
        }
        assert!(
            result.errors.is_empty(),
            "Parse errors: {:?}",
            result.errors
        );
    }
}
