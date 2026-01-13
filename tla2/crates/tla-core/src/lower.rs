//! CST to AST lowering
//!
//! This module converts the lossless Concrete Syntax Tree (CST) produced by the
//! parser into a typed Abstract Syntax Tree (AST) suitable for semantic analysis.
//!
//! The CST preserves all whitespace and comments, while the AST only contains
//! the semantically meaningful parts of the source code.

use crate::ast::{
    AssumeDecl, BoundPattern, BoundVar, CaseArm, ConstantDecl, ExceptPathElement, ExceptSpec, Expr,
    InstanceDecl, Module, ModuleTarget, OpParam, OperatorDef, Proof, ProofHint, ProofStep,
    ProofStepKind, RecursiveDecl, Substitution, TheoremDecl, Unit,
};
use crate::span::{FileId, Span, Spanned};
use crate::syntax::kinds::{SyntaxKind, SyntaxNode, SyntaxToken};
use num_bigint::BigInt;

/// Errors that can occur during lowering
#[derive(Debug, Clone)]
pub struct LowerError {
    pub message: String,
    pub span: Span,
}

/// Context for lowering operations
pub struct LowerCtx {
    /// The file being lowered
    file_id: FileId,
    /// Collected errors
    errors: Vec<LowerError>,
}

impl LowerCtx {
    /// Create a new lowering context
    pub fn new(file_id: FileId) -> Self {
        Self {
            file_id,
            errors: Vec::new(),
        }
    }

    /// Get the file ID
    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    /// Record an error
    pub fn error(&mut self, message: impl Into<String>, span: Span) {
        self.errors.push(LowerError {
            message: message.into(),
            span,
        });
    }

    /// Take all collected errors
    pub fn take_errors(&mut self) -> Vec<LowerError> {
        std::mem::take(&mut self.errors)
    }

    /// Create a span from a syntax node
    fn span(&self, node: &SyntaxNode) -> Span {
        let range = node.text_range();
        Span::new(self.file_id, range.start().into(), range.end().into())
    }

    /// Create a span from a syntax token
    fn token_span(&self, token: &SyntaxToken) -> Span {
        let range = token.text_range();
        Span::new(self.file_id, range.start().into(), range.end().into())
    }

    /// Create a "tight" span from a syntax node, excluding trailing trivia.
    /// This walks the children recursively to find the last non-trivia token.
    fn tight_span(&self, node: &SyntaxNode) -> Span {
        let range = node.text_range();
        let start: u32 = range.start().into();

        // Find the last non-trivia token by walking children recursively
        if let Some(end) = Self::find_last_non_trivia_end(node) {
            Span::new(self.file_id, start, end)
        } else {
            // Fallback to full range if no tokens found
            Span::new(self.file_id, start, range.end().into())
        }
    }

    /// Recursively find the end offset of the last non-trivia token in a node
    fn find_last_non_trivia_end(node: &SyntaxNode) -> Option<u32> {
        let mut last_end: Option<u32> = None;

        for child in node.children_with_tokens() {
            match child {
                rowan::NodeOrToken::Token(token) => {
                    if !token.kind().is_trivia() {
                        last_end = Some(token.text_range().end().into());
                    }
                }
                rowan::NodeOrToken::Node(child_node) => {
                    // Recursively find in child node
                    if let Some(end) = Self::find_last_non_trivia_end(&child_node) {
                        last_end = Some(end);
                    }
                }
            }
        }

        last_end
    }
}

/// Lower a CST root to AST Module
pub fn lower_module(ctx: &mut LowerCtx, root: &SyntaxNode) -> Option<Module> {
    // Find the Module node in the Root
    let module_node = root.children().find(|n| n.kind() == SyntaxKind::Module)?;

    lower_module_node(ctx, &module_node)
}

/// Lower a Module node
fn lower_module_node(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Module> {
    let span = ctx.span(node);

    // Find the module name (Ident token after MODULE keyword)
    let name = find_module_name(ctx, node)?;

    // Find EXTENDS clause
    let extends = node
        .children()
        .find(|n| n.kind() == SyntaxKind::ExtendsClause)
        .map(|n| lower_extends(ctx, &n))
        .unwrap_or_default();

    // Collect all units
    let mut units = Vec::new();
    for child in node.children() {
        if let Some(unit) = lower_unit(ctx, &child) {
            let unit_span = ctx.span(&child);
            units.push(Spanned::new(unit, unit_span));
        }
    }

    Some(Module {
        name,
        extends,
        units,
        span,
    })
}

/// Find the module name from the Module node
fn find_module_name(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Spanned<String>> {
    let mut saw_module_kw = false;
    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if token.kind() == SyntaxKind::ModuleKw {
                    saw_module_kw = true;
                } else if saw_module_kw && token.kind() == SyntaxKind::Ident {
                    let span = ctx.token_span(&token);
                    return Some(Spanned::new(token.text().to_string(), span));
                }
            }
            rowan::NodeOrToken::Node(_) => {}
        }
    }
    None
}

/// Lower EXTENDS clause
fn lower_extends(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<Spanned<String>> {
    let mut names = Vec::new();
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            if token.kind() == SyntaxKind::Ident {
                let span = ctx.token_span(&token);
                names.push(Spanned::new(token.text().to_string(), span));
            }
        }
    }
    names
}

/// Create an Apply expression, handling WF_xxx/SF_xxx identifiers.
///
/// Due to lexer "maximal munch" behavior, `WF_cvars(CRcvMsg)` is tokenized as
/// `Ident("WF_cvars")` + `ArgList` rather than `WeakFair` + `Ident("cvars")` + `ArgList`.
/// This function detects WF_/SF_ prefixed identifiers and converts them to the
/// proper WeakFair/StrongFair expression.
fn make_apply_expr(op: Spanned<Expr>, mut args: Vec<Spanned<Expr>>) -> Expr {
    if let Expr::Ident(name) = &op.node {
        // WF_vars(Action) => WeakFair(vars, Action)
        if let Some(vars_name) = name.strip_prefix("WF_") {
            if args.len() == 1 {
                let vars_expr = Spanned::new(Expr::Ident(vars_name.to_string()), op.span);
                return Expr::WeakFair(Box::new(vars_expr), Box::new(args.remove(0)));
            }
        }
        // SF_vars(Action) => StrongFair(vars, Action)
        else if let Some(vars_name) = name.strip_prefix("SF_") {
            if args.len() == 1 {
                let vars_expr = Spanned::new(Expr::Ident(vars_name.to_string()), op.span);
                return Expr::StrongFair(Box::new(vars_expr), Box::new(args.remove(0)));
            }
        }
    }
    Expr::Apply(Box::new(op), args)
}

/// Lower a unit (top-level declaration or definition)
fn lower_unit(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Unit> {
    match node.kind() {
        SyntaxKind::VariableDecl => Some(lower_variable_decl(ctx, node)),
        SyntaxKind::ConstantDecl => Some(lower_constant_decl(ctx, node)),
        SyntaxKind::RecursiveDecl => Some(lower_recursive_decl(ctx, node)),
        SyntaxKind::OperatorDef => lower_operator_def(ctx, node).map(Unit::Operator),
        SyntaxKind::InstanceDecl => lower_instance_decl(ctx, node).map(Unit::Instance),
        SyntaxKind::AssumeStmt => lower_assume(ctx, node).map(Unit::Assume),
        SyntaxKind::TheoremStmt => lower_theorem(ctx, node).map(Unit::Theorem),
        SyntaxKind::Separator => Some(Unit::Separator),
        _ => None,
    }
}

/// Lower VARIABLE declaration
fn lower_variable_decl(ctx: &mut LowerCtx, node: &SyntaxNode) -> Unit {
    let names = collect_idents(ctx, node);
    Unit::Variable(names)
}

/// Lower CONSTANT declaration
fn lower_constant_decl(ctx: &mut LowerCtx, node: &SyntaxNode) -> Unit {
    let mut decls = Vec::new();

    let tokens: Vec<SyntaxToken> = node
        .children_with_tokens()
        .filter_map(|child| match child {
            rowan::NodeOrToken::Token(token) => Some(token),
            rowan::NodeOrToken::Node(_) => None,
        })
        .collect();

    let mut i = 0;
    while i < tokens.len() {
        let token = &tokens[i];

        if token.kind() != SyntaxKind::Ident {
            i += 1;
            continue;
        }

        let span = ctx.token_span(token);
        let name = Spanned::new(token.text().to_string(), span);

        // CONSTANTS can have arity: C(_, _)
        let mut arity = None;
        let mut j = i + 1;

        while j < tokens.len() && tokens[j].kind().is_trivia() {
            j += 1;
        }

        if j < tokens.len() && tokens[j].kind() == SyntaxKind::LParen {
            j += 1;
            let mut count = 0usize;
            while j < tokens.len() {
                match tokens[j].kind() {
                    SyntaxKind::Underscore => {
                        count += 1;
                    }
                    SyntaxKind::RParen => {
                        j += 1;
                        break;
                    }
                    _ => {}
                }
                j += 1;
            }
            arity = Some(count);
            i = j;
        } else {
            i += 1;
        }

        decls.push(ConstantDecl { name, arity });
    }

    Unit::Constant(decls)
}

/// Lower RECURSIVE declaration: RECURSIVE Op(_, _)
fn lower_recursive_decl(ctx: &mut LowerCtx, node: &SyntaxNode) -> Unit {
    let mut decls = Vec::new();

    let tokens: Vec<SyntaxToken> = node
        .children_with_tokens()
        .filter_map(|child| match child {
            rowan::NodeOrToken::Token(token) => Some(token),
            rowan::NodeOrToken::Node(_) => None,
        })
        .collect();

    let mut i = 0;
    while i < tokens.len() {
        let token = &tokens[i];

        if token.kind() != SyntaxKind::Ident {
            i += 1;
            continue;
        }

        let span = ctx.token_span(token);
        let name = Spanned::new(token.text().to_string(), span);

        // Parse arity: Op(_, _)
        let mut arity = 0usize;
        let mut j = i + 1;

        while j < tokens.len() && tokens[j].kind().is_trivia() {
            j += 1;
        }

        if j < tokens.len() && tokens[j].kind() == SyntaxKind::LParen {
            j += 1;
            while j < tokens.len() {
                match tokens[j].kind() {
                    SyntaxKind::Underscore => {
                        arity += 1;
                    }
                    SyntaxKind::RParen => {
                        j += 1;
                        break;
                    }
                    _ => {}
                }
                j += 1;
            }
            i = j;
        } else {
            i += 1;
        }

        decls.push(RecursiveDecl { name, arity });
    }

    Unit::Recursive(decls)
}

fn is_preceded_by_local_kw(node: &SyntaxNode) -> bool {
    let mut prev = node.prev_sibling_or_token();

    while let Some(el) = prev {
        match el {
            rowan::NodeOrToken::Token(token) => {
                if token.kind().is_trivia() {
                    prev = token.prev_sibling_or_token();
                    continue;
                }
                return token.kind() == SyntaxKind::LocalKw;
            }
            rowan::NodeOrToken::Node(_) => return false,
        }
    }

    false
}

/// Lower operator definition
fn lower_operator_def(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<OperatorDef> {
    let mut name: Option<Spanned<String>> = None;
    let mut params = Vec::new();
    let mut func_bounds: Vec<BoundVar> = Vec::new(); // Bound vars for function operators
    let local = is_preceded_by_local_kw(node);
    let mut header_tokens: Vec<(SyntaxKind, String, Span)> = Vec::new();

    // Collect header tokens (before ==) and parse any ArgList params.
    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind == SyntaxKind::DefEqOp || kind == SyntaxKind::TriangleEqOp {
                    break;
                }
                if kind.is_trivia() {
                    continue;
                }
                header_tokens.push((kind, token.text().to_string(), ctx.token_span(&token)));
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::ArgList {
                    // Check if ArgList contains BoundVars (function operator syntax: f[x \in S])
                    let (regular_params, bounds) = lower_op_def_arg_list(ctx, &child_node);
                    params = regular_params;
                    func_bounds = bounds;
                }
            }
        }
    }

    // Detect infix operator definitions:
    // - a \prec b == ...
    // - a (+) b == ...
    // We lower these as operator name = infix symbol, params = [a, b].
    if params.is_empty() && func_bounds.is_empty() {
        // Find left operand identifier.
        let mut idx = 0;
        while idx < header_tokens.len() && header_tokens[idx].0 != SyntaxKind::Ident {
            idx += 1;
        }

        if idx + 2 < header_tokens.len() {
            let (_lk, left_text, left_span) = &header_tokens[idx];
            idx += 1;

            let mut parenthesized = false;
            if idx < header_tokens.len() && header_tokens[idx].0 == SyntaxKind::LParen {
                parenthesized = true;
                idx += 1;
            }

            if idx < header_tokens.len() {
                let (op_kind, _op_text, op_span) = &header_tokens[idx];
                let op_name = operator_token_to_name(*op_kind);
                idx += 1;

                if parenthesized {
                    if idx < header_tokens.len() && header_tokens[idx].0 == SyntaxKind::RParen {
                        idx += 1;
                    } else {
                        // Not a valid parenthesized infix operator form.
                        idx = header_tokens.len();
                    }
                }

                if let Some(op_name) = op_name {
                    if idx < header_tokens.len() && header_tokens[idx].0 == SyntaxKind::Ident {
                        let (_rk, right_text, right_span) = &header_tokens[idx];
                        name = Some(Spanned::new(op_name.to_string(), *op_span));
                        params = vec![
                            OpParam {
                                name: Spanned::new(left_text.clone(), *left_span),
                                arity: 0,
                            },
                            OpParam {
                                name: Spanned::new(right_text.clone(), *right_span),
                                arity: 0,
                            },
                        ];
                    }
                }
            }
        }
    }

    // Fallback: standard operator definition name.
    if name.is_none() {
        // Underscore-prefixed operator name: _name
        if header_tokens.len() >= 2
            && header_tokens[0].0 == SyntaxKind::Underscore
            && header_tokens[1].0 == SyntaxKind::Ident
        {
            let span = header_tokens[0].2;
            name = Some(Spanned::new(format!("_{}", header_tokens[1].1), span));
        } else if let Some((kind, text, span)) = header_tokens.first() {
            if *kind == SyntaxKind::Ident {
                name = Some(Spanned::new(text.clone(), *span));
            } else if let Some(std_name) = stdlib_keyword_to_name(*kind) {
                name = Some(Spanned::new(std_name, *span));
            } else if let Some(op_name) = operator_token_to_name(*kind) {
                name = Some(Spanned::new(op_name.to_string(), *span));
            }
        }
    }

    // Second pass: collect the body expression parts after ==
    let mut body = lower_expr_from_children(ctx, node)?;

    // If there are bound variables from function operator syntax,
    // wrap the body in a FuncDef expression: f[x \in S] == body becomes f == [x \in S |-> body]
    if !func_bounds.is_empty() {
        let body_span = body.span;
        body = Spanned::new(Expr::FuncDef(func_bounds, Box::new(body)), body_span);
    }

    Some(OperatorDef {
        name: name?,
        params,
        body,
        local,
    })
}

/// Lower an ArgList in operator definition, returning both regular params and any bound variables (for function operators)
fn lower_op_def_arg_list(ctx: &mut LowerCtx, node: &SyntaxNode) -> (Vec<OpParam>, Vec<BoundVar>) {
    let mut params = Vec::new();
    let mut bounds = Vec::new();

    for child in node.children() {
        if child.kind() == SyntaxKind::OperatorParam {
            if let Some(param) = lower_operator_param(ctx, &child) {
                params.push(param);
            }
        } else if child.kind() == SyntaxKind::BoundVar {
            if let Some(bv) = lower_bound_var(ctx, &child) {
                bounds.push(bv);
            }
        }
    }

    // Also check for direct Ident tokens (simple params like f(a, b))
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            if token.kind() == SyntaxKind::Ident {
                let span = ctx.token_span(&token);
                params.push(OpParam {
                    name: Spanned::new(token.text().to_string(), span),
                    arity: 0,
                });
            }
        }
    }

    // Propagate domains in bound variable lists.
    // In TLA+, `f[a, b \in S]` means both `a` and `b` are in S.
    propagate_bound_domains(&mut bounds);

    (params, bounds)
}

/// Lower expression from a sequence of children (handles fragmented expressions)
/// This is needed because the parser doesn't always wrap expressions in nodes.
/// For example, `a + b` might be stored as: Ident("a"), BinaryExpr("+", Ident("b"))
enum ExprStart {
    AfterDefEq,
    AfterKeyword(SyntaxKind),
}

fn lower_expr_from_children_start(
    ctx: &mut LowerCtx,
    node: &SyntaxNode,
    start: ExprStart,
) -> Option<Spanned<Expr>> {
    let mut result: Option<Spanned<Expr>> = None;
    let mut in_expr = false;
    // Track pending stdlib operator keyword (Len, Head, etc.) that precedes ApplyExpr
    let mut pending_stdlib_op: Option<(String, Span)> = None;
    // Track underscore-prefixed identifiers like `_msgs`
    let mut pending_underscore = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();

                if !in_expr {
                    match start {
                        ExprStart::AfterDefEq => {
                            if kind == SyntaxKind::DefEqOp || kind == SyntaxKind::TriangleEqOp {
                                in_expr = true;
                            }
                        }
                        ExprStart::AfterKeyword(start_kind) => {
                            if kind == start_kind {
                                in_expr = true;
                            }
                        }
                    }
                    continue;
                }

                // Skip trivia and operators (operators are in BinaryExpr nodes)
                if kind.is_trivia() || is_binary_op(kind) {
                    continue;
                }

                if kind == SyntaxKind::Underscore {
                    pending_underscore = true;
                    continue;
                }

                // Handle stdlib keyword tokens (Len, Head, Tail, etc.) which precede ApplyExpr
                if let Some(op_name) = stdlib_keyword_to_name(kind) {
                    let span = ctx.token_span(&token);
                    pending_stdlib_op = Some((op_name, span));
                    pending_underscore = false;
                    continue;
                }

                // Handle leaf tokens as expressions
                let expr = match kind {
                    SyntaxKind::Ident => {
                        if pending_underscore {
                            pending_underscore = false;
                            Some(Expr::Ident(format!("_{}", token.text())))
                        } else {
                            Some(Expr::Ident(token.text().to_string()))
                        }
                    }
                    // @ is used inside EXCEPT specs to refer to the old value.
                    // Lower it as an identifier so evaluators can bind it.
                    SyntaxKind::At => Some(Expr::Ident("@".to_string())),
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    SyntaxKind::BooleanKw => Some(Expr::Ident("BOOLEAN".to_string())),
                    _ => None,
                };

                if let Some(e) = expr {
                    let span = ctx.token_span(&token);
                    result = Some(Spanned::new(e, span));
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if !in_expr {
                    continue;
                }

                if child_node.kind() == SyntaxKind::Proof {
                    break;
                }

                // For BinaryExpr, we need special handling
                if child_node.kind() == SyntaxKind::BinaryExpr {
                    if let Some(left) = result.take() {
                        // Extract operator and right operand from BinaryExpr
                        if let Some(combined) = lower_binary_with_left(ctx, &child_node, left) {
                            result = Some(combined);
                        }
                    } else {
                        // No left operand, try normal lowering
                        if let Some(expr) = lower_expr(ctx, &child_node) {
                            let span = ctx.tight_span(&child_node);
                            result = Some(Spanned::new(expr, span));
                        }
                    }
                } else if child_node.kind() == SyntaxKind::ApplyExpr {
                    // Check if we have a pending stdlib operator keyword (Len, Head, etc.)
                    if let Some((op_name, op_span)) = pending_stdlib_op.take() {
                        // Combine stdlib keyword with ApplyExpr args
                        let args = lower_apply_args(ctx, &child_node);
                        let op_expr = Spanned::new(Expr::Ident(op_name), op_span);
                        let span = ctx.tight_span(&child_node);
                        result = Some(Spanned::new(Expr::Apply(Box::new(op_expr), args), span));
                    } else if let Some(prev) = result.take() {
                        // Check if previous result is an Ident (operator name like Cardinality)
                        // If so, combine with ApplyExpr args (handles WF_/SF_ identifiers)
                        let args = lower_apply_args(ctx, &child_node);
                        let span = ctx.tight_span(&child_node);
                        result = Some(Spanned::new(make_apply_expr(prev, args), span));
                    } else {
                        // Normal ApplyExpr handling (operator should be inside the node)
                        if let Some(expr) = lower_expr(ctx, &child_node) {
                            let span = ctx.tight_span(&child_node);
                            result = Some(Spanned::new(expr, span));
                        }
                    }
                } else {
                    // Normal expression node
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.tight_span(&child_node);
                        result = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    result
}

fn lower_expr_from_children(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Spanned<Expr>> {
    lower_expr_from_children_start(ctx, node, ExprStart::AfterDefEq)
}

fn lower_expr_from_children_after_keyword(
    ctx: &mut LowerCtx,
    node: &SyntaxNode,
    keyword: SyntaxKind,
) -> Option<Spanned<Expr>> {
    lower_expr_from_children_start(ctx, node, ExprStart::AfterKeyword(keyword))
}

/// Convert stdlib keyword token to operator name
fn stdlib_keyword_to_name(kind: SyntaxKind) -> Option<String> {
    match kind {
        SyntaxKind::LenKw => Some("Len".to_string()),
        SyntaxKind::HeadKw => Some("Head".to_string()),
        SyntaxKind::TailKw => Some("Tail".to_string()),
        SyntaxKind::AppendKw => Some("Append".to_string()),
        SyntaxKind::SubSeqKw => Some("SubSeq".to_string()),
        SyntaxKind::SelectSeqKw => Some("SelectSeq".to_string()),
        SyntaxKind::SeqKw => Some("Seq".to_string()),
        // Note: Cardinality is parsed as an identifier, not a keyword
        _ => None,
    }
}

/// Lower args from ApplyExpr node (extracts from ArgList child)
fn lower_apply_args(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<Spanned<Expr>> {
    for child in node.children() {
        if child.kind() == SyntaxKind::ArgList {
            return lower_arg_list(ctx, &child);
        }
    }
    Vec::new()
}

/// Lower a BinaryExpr node given its left operand (which is a sibling in the CST)
fn lower_binary_with_left(
    ctx: &mut LowerCtx,
    node: &SyntaxNode,
    left: Spanned<Expr>,
) -> Option<Spanned<Expr>> {
    let mut op: Option<SyntaxKind> = None;
    let mut right: Option<Spanned<Expr>> = None;
    // Track pending identifier that might be followed by ApplyExpr
    let mut pending_ident: Option<Spanned<Expr>> = None;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if is_binary_op(kind) && op.is_none() {
                    // Don't flush pending_ident to right - it's part of the left operand
                    // which was already passed as a parameter
                    pending_ident = None;
                    op = Some(kind);
                // Handle stdlib keyword tokens (Len, Head, etc.) which precede ApplyExpr
                // Only process these AFTER we've seen the operator
                } else if op.is_some() {
                    if let Some(op_name) = stdlib_keyword_to_name(kind) {
                        // Flush any existing pending ident first
                        if let Some(ident) = pending_ident.take() {
                            right = Some(ident);
                        }
                        let span = ctx.token_span(&token);
                        pending_ident = Some(Spanned::new(Expr::Ident(op_name), span));
                    } else if right.is_none() {
                        // Try to make expression from token
                        let expr = match kind {
                            SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                            SyntaxKind::At => Some(Expr::Ident("@".to_string())),
                            SyntaxKind::Number => {
                                token.text().parse::<BigInt>().ok().map(Expr::Int)
                            }
                            SyntaxKind::String => {
                                let s = token.text();
                                let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                                Some(Expr::String(inner.to_string()))
                            }
                            SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                            SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                            _ => None,
                        };
                        if let Some(e) = expr {
                            let span = ctx.token_span(&token);
                            // For identifiers, keep pending in case ApplyExpr follows
                            if matches!(e, Expr::Ident(_)) {
                                pending_ident = Some(Spanned::new(e, span));
                            } else {
                                right = Some(Spanned::new(e, span));
                            }
                        }
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                // Only process nodes for the RHS after we've seen the operator
                if op.is_none() {
                    continue;
                }

                // Check for ApplyExpr following an identifier
                if child_node.kind() == SyntaxKind::ApplyExpr {
                    if let Some(ident) = pending_ident.take() {
                        // Combine identifier with ApplyExpr args (handles WF_/SF_ identifiers)
                        let args = lower_apply_args(ctx, &child_node);
                        let span = ctx.tight_span(&child_node);
                        right = Some(Spanned::new(make_apply_expr(ident, args), span));
                        continue;
                    }
                }

                // Flush pending_ident if we're seeing a non-ApplyExpr node
                if let Some(ident) = pending_ident.take() {
                    right = Some(ident);
                }

                // Check for nested BinaryExpr
                if child_node.kind() == SyntaxKind::BinaryExpr {
                    if let Some(r) = right.take() {
                        // We have left and op, combine with right, then process nested
                        if let Some(op_kind) = op {
                            let combined = make_binary_expr(op_kind, false, left.clone(), r);
                            let span = ctx.tight_span(node);
                            let new_left = Spanned::new(combined, span);
                            return lower_binary_with_left(ctx, &child_node, new_left);
                        }
                    } else if let Some(r) = lower_expr(ctx, &child_node) {
                        let span = ctx.tight_span(&child_node);
                        right = Some(Spanned::new(r, span));
                    }
                } else if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.tight_span(&child_node);
                    if right.is_none() {
                        right = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    // Flush any pending identifier
    if let Some(ident) = pending_ident.take() {
        right = Some(ident);
    }

    // Combine left, op, right
    let op = op?;
    let right = right?;
    let combined = make_binary_expr(op, false, left, right);
    let span = ctx.span(node);
    Some(Spanned::new(combined, span))
}

/// Lower an operator parameter
fn lower_operator_param(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<OpParam> {
    let mut name: Option<Spanned<String>> = None;
    let mut arity = 0;

    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            match token.kind() {
                SyntaxKind::Ident if name.is_none() => {
                    let span = ctx.token_span(&token);
                    name = Some(Spanned::new(token.text().to_string(), span));
                }
                SyntaxKind::Underscore => {
                    arity += 1;
                }
                _ => {}
            }
        }
    }

    Some(OpParam { name: name?, arity })
}

/// Lower INSTANCE declaration
fn lower_instance_decl(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<InstanceDecl> {
    let mut module: Option<Spanned<String>> = None;
    let mut substitutions = Vec::new();
    let local = is_preceded_by_local_kw(node);

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if token.kind() == SyntaxKind::Ident && module.is_none() {
                    let span = ctx.token_span(&token);
                    module = Some(Spanned::new(token.text().to_string(), span));
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::Substitution {
                    if let Some(sub) = lower_substitution(ctx, &child_node) {
                        substitutions.push(sub);
                    }
                }
            }
        }
    }

    Some(InstanceDecl {
        module: module?,
        substitutions,
        local,
    })
}

/// Lower a substitution (x <- e)
fn lower_substitution(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Substitution> {
    let mut from: Option<Spanned<String>> = None;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if token.kind() == SyntaxKind::Ident && from.is_none() {
                    let span = ctx.token_span(&token);
                    from = Some(Spanned::new(token.text().to_string(), span));
                }
            }
            rowan::NodeOrToken::Node(_) => {}
        }
    }

    let to = lower_expr_from_children_after_keyword(ctx, node, SyntaxKind::LArrow)?;

    Some(Substitution { from: from?, to })
}

/// Lower ASSUME statement
fn lower_assume(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<AssumeDecl> {
    let mut name: Option<Spanned<String>> = None;

    // Check if there's a name (Ident before ==)
    // Similar to lower_theorem name extraction
    let tokens: Vec<_> = node.children_with_tokens().collect();
    let mut i = 0;
    while i < tokens.len() {
        if let rowan::NodeOrToken::Token(token) = &tokens[i] {
            if token.kind() == SyntaxKind::Ident {
                // Check if next non-trivia is ==
                let mut j = i + 1;
                while j < tokens.len() {
                    if let rowan::NodeOrToken::Token(next) = &tokens[j] {
                        if !next.kind().is_trivia() {
                            if next.kind() == SyntaxKind::DefEqOp {
                                let span = ctx.token_span(token);
                                name = Some(Spanned::new(token.text().to_string(), span));
                            }
                            break;
                        }
                    }
                    j += 1;
                }
            }
        }
        i += 1;
    }

    let expr = lower_expr_from_children(ctx, node)
        .or_else(|| lower_expr_from_children_after_keyword(ctx, node, SyntaxKind::AssumeKw))?;

    Some(AssumeDecl { name, expr })
}

/// Lower THEOREM statement
fn lower_theorem(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<TheoremDecl> {
    let mut name: Option<Spanned<String>> = None;
    let mut proof: Option<Spanned<Proof>> = None;

    // Check if there's a name (Ident before ==)
    let tokens: Vec<_> = node.children_with_tokens().collect();
    let mut i = 0;
    while i < tokens.len() {
        if let rowan::NodeOrToken::Token(token) = &tokens[i] {
            if token.kind() == SyntaxKind::Ident {
                // Check if next non-trivia is ==
                let mut j = i + 1;
                while j < tokens.len() {
                    if let rowan::NodeOrToken::Token(next) = &tokens[j] {
                        if !next.kind().is_trivia() {
                            if next.kind() == SyntaxKind::DefEqOp {
                                let span = ctx.token_span(token);
                                name = Some(Spanned::new(token.text().to_string(), span));
                            }
                            break;
                        }
                    }
                    j += 1;
                }
            }
        }
        i += 1;
    }

    for child in node.children() {
        if child.kind() == SyntaxKind::Proof {
            proof = lower_proof(ctx, &child).map(|p| {
                let span = ctx.span(&child);
                Spanned::new(p, span)
            });
        }
    }

    let theorem_kw = node.children_with_tokens().find_map(|child| match child {
        rowan::NodeOrToken::Token(token) => match token.kind() {
            SyntaxKind::TheoremKw
            | SyntaxKind::LemmaKw
            | SyntaxKind::PropositionKw
            | SyntaxKind::CorollaryKw => Some(token.kind()),
            _ => None,
        },
        rowan::NodeOrToken::Node(_) => None,
    });

    let body = lower_expr_from_children(ctx, node).or_else(|| {
        theorem_kw.and_then(|kw| lower_expr_from_children_after_keyword(ctx, node, kw))
    });

    Some(TheoremDecl {
        name,
        body: body?,
        proof,
    })
}

/// Lower a proof
fn lower_proof(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Proof> {
    // Check for simple proofs first
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            match token.kind() {
                SyntaxKind::ObviousKw => return Some(Proof::Obvious),
                SyntaxKind::OmittedKw => return Some(Proof::Omitted),
                _ => {}
            }
        }
    }

    // Check for BY clause
    for child in node.children() {
        if child.kind() == SyntaxKind::ByClause {
            let hints = lower_proof_hints(ctx, &child);
            return Some(Proof::By(hints));
        }
    }

    // Check for structured proof (steps)
    let mut steps = Vec::new();
    for child in node.children() {
        if child.kind() == SyntaxKind::ProofStep {
            if let Some(step) = lower_proof_step(ctx, &child) {
                steps.push(step);
            }
        }
    }
    if !steps.is_empty() {
        return Some(Proof::Steps(steps));
    }

    None
}

/// Lower proof hints
fn lower_proof_hints(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<ProofHint> {
    let mut hints = Vec::new();

    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            if token.kind() == SyntaxKind::Ident {
                let span = ctx.token_span(&token);
                hints.push(ProofHint::Ref(Spanned::new(token.text().to_string(), span)));
            }
        }
    }

    hints
}

/// Lower a proof step
fn lower_proof_step(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<ProofStep> {
    let mut level = 1;
    let mut label: Option<Spanned<String>> = None;

    // Find step label
    for child in node.children() {
        if child.kind() == SyntaxKind::StepLabel {
            // Parse level from <n>
            for tok in child.children_with_tokens() {
                if let rowan::NodeOrToken::Token(token) = tok {
                    match token.kind() {
                        SyntaxKind::Number => {
                            if let Ok(n) = token.text().parse::<usize>() {
                                level = n;
                            }
                        }
                        SyntaxKind::Ident => {
                            let span = ctx.token_span(&token);
                            label = Some(Spanned::new(token.text().to_string(), span));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Determine step kind
    let kind = lower_proof_step_kind(ctx, node)?;

    Some(ProofStep { level, label, kind })
}

/// Lower proof step kind
fn lower_proof_step_kind(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<ProofStepKind> {
    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                match token.kind() {
                    SyntaxKind::SufficesKw => {
                        // Find the expression and optional proof
                        let (expr, proof) = find_step_expr_and_proof(ctx, node);
                        return Some(ProofStepKind::Suffices(expr?, proof));
                    }
                    SyntaxKind::HaveKw => {
                        let (expr, _) = find_step_expr_and_proof(ctx, node);
                        return Some(ProofStepKind::Have(expr?));
                    }
                    SyntaxKind::TakeKw => {
                        let bounds = find_bound_vars(ctx, node);
                        return Some(ProofStepKind::Take(bounds));
                    }
                    SyntaxKind::WitnessKw => {
                        let exprs = find_all_exprs(ctx, node);
                        return Some(ProofStepKind::Witness(exprs));
                    }
                    SyntaxKind::PickKw => {
                        let bounds = find_bound_vars(ctx, node);
                        let (expr, proof) = find_step_expr_and_proof(ctx, node);
                        return Some(ProofStepKind::Pick(bounds, expr?, proof));
                    }
                    SyntaxKind::UseKw => {
                        let hints = collect_proof_hints_from_node(ctx, node);
                        return Some(ProofStepKind::UseOrHide {
                            use_: true,
                            facts: hints,
                        });
                    }
                    SyntaxKind::HideKw => {
                        let hints = collect_proof_hints_from_node(ctx, node);
                        return Some(ProofStepKind::UseOrHide {
                            use_: false,
                            facts: hints,
                        });
                    }
                    SyntaxKind::DefineKw => {
                        let defs = collect_operator_defs(ctx, node);
                        return Some(ProofStepKind::Define(defs));
                    }
                    SyntaxKind::QedKw => {
                        // Look for nested proof
                        let proof = node
                            .children()
                            .find(|n| n.kind() == SyntaxKind::Proof)
                            .and_then(|n| lower_proof(ctx, &n))
                            .map(|p| Spanned::new(p, ctx.span(node)));
                        return Some(ProofStepKind::Qed(proof));
                    }
                    _ => {}
                }
            }
            rowan::NodeOrToken::Node(_) => {}
        }
    }

    // Default: assertion
    let (expr, proof) = find_step_expr_and_proof(ctx, node);
    Some(ProofStepKind::Assert(expr?, proof))
}

/// Find expression and optional proof in a proof step
fn find_step_expr_and_proof(
    ctx: &mut LowerCtx,
    node: &SyntaxNode,
) -> (Option<Spanned<Expr>>, Option<Spanned<Proof>>) {
    let mut expr: Option<Spanned<Expr>> = None;
    let mut proof: Option<Spanned<Proof>> = None;

    for child in node.children() {
        match child.kind() {
            SyntaxKind::Proof => {
                proof = lower_proof(ctx, &child).map(|p| Spanned::new(p, ctx.span(&child)));
            }
            SyntaxKind::StepLabel => {}
            _ => {
                if expr.is_none() {
                    if let Some(e) = lower_expr(ctx, &child) {
                        expr = Some(Spanned::new(e, ctx.span(&child)));
                    }
                }
            }
        }
    }

    (expr, proof)
}

/// Find bound variables in a node
fn find_bound_vars(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<BoundVar> {
    let mut vars = Vec::new();
    for child in node.children() {
        if child.kind() == SyntaxKind::BoundVar {
            if let Some(bv) = lower_bound_var(ctx, &child) {
                vars.push(bv);
            }
        }
    }
    vars
}

/// Find all expressions in a node (for WITNESS)
fn find_all_exprs(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<Spanned<Expr>> {
    let mut exprs = Vec::new();
    for child in node.children() {
        if let Some(e) = lower_expr(ctx, &child) {
            exprs.push(Spanned::new(e, ctx.span(&child)));
        }
    }
    exprs
}

/// Collect proof hints from a node
fn collect_proof_hints_from_node(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<ProofHint> {
    let mut hints = Vec::new();
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            if token.kind() == SyntaxKind::Ident {
                let span = ctx.token_span(&token);
                hints.push(ProofHint::Ref(Spanned::new(token.text().to_string(), span)));
            }
        }
    }
    hints
}

/// Collect operator definitions from a node (for DEFINE)
fn collect_operator_defs(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<OperatorDef> {
    let mut defs = Vec::new();
    for child in node.children() {
        if child.kind() == SyntaxKind::OperatorDef {
            if let Some(def) = lower_operator_def(ctx, &child) {
                defs.push(def);
            }
        }
    }
    defs
}

/// Collect identifier tokens from a node
fn collect_idents(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<Spanned<String>> {
    let mut idents = Vec::new();

    // Check direct tokens
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            if token.kind() == SyntaxKind::Ident {
                let span = ctx.token_span(&token);
                idents.push(Spanned::new(token.text().to_string(), span));
            }
        }
    }

    // Also check NameList children
    for child in node.children() {
        if child.kind() == SyntaxKind::NameList {
            for tok in child.children_with_tokens() {
                if let rowan::NodeOrToken::Token(token) = tok {
                    if token.kind() == SyntaxKind::Ident {
                        let span = ctx.token_span(&token);
                        idents.push(Spanned::new(token.text().to_string(), span));
                    }
                }
            }
        }
    }

    idents
}

// =============================================================================
// Expression Lowering
// =============================================================================

/// Lower an expression node to AST Expr
fn lower_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    match node.kind() {
        // Literals - handled specially
        SyntaxKind::TrueKw => Some(Expr::Bool(true)),
        SyntaxKind::FalseKw => Some(Expr::Bool(false)),

        // Composite expressions
        SyntaxKind::ParenExpr => lower_paren_expr(ctx, node),
        SyntaxKind::BinaryExpr => lower_binary_expr(ctx, node),
        SyntaxKind::UnaryExpr => lower_unary_expr(ctx, node),
        SyntaxKind::ApplyExpr => lower_apply_expr(ctx, node),
        SyntaxKind::LambdaExpr => lower_lambda_expr(ctx, node),
        SyntaxKind::QuantExpr => lower_quant_expr(ctx, node),
        SyntaxKind::ChooseExpr => lower_choose_expr(ctx, node),
        SyntaxKind::SetEnumExpr => lower_set_enum_expr(ctx, node),
        SyntaxKind::SetBuilderExpr => lower_set_builder_expr(ctx, node),
        SyntaxKind::SetFilterExpr => lower_set_filter_expr(ctx, node),
        SyntaxKind::FuncDefExpr => lower_func_def_expr(ctx, node),
        SyntaxKind::FuncApplyExpr => lower_func_apply_expr(ctx, node),
        SyntaxKind::FuncSetExpr => lower_func_set_expr(ctx, node),
        SyntaxKind::ExceptExpr => lower_except_expr(ctx, node),
        SyntaxKind::RecordExpr => lower_record_expr(ctx, node),
        SyntaxKind::RecordAccessExpr => lower_record_access_expr(ctx, node),
        SyntaxKind::RecordSetExpr => lower_record_set_expr(ctx, node),
        SyntaxKind::TupleExpr => lower_tuple_expr(ctx, node),
        SyntaxKind::IfExpr => lower_if_expr(ctx, node),
        SyntaxKind::CaseExpr => lower_case_expr(ctx, node),
        SyntaxKind::LetExpr => lower_let_expr(ctx, node),
        SyntaxKind::ModuleRefExpr => lower_module_ref_expr(ctx, node),
        SyntaxKind::SubscriptExpr => lower_subscript_expr(ctx, node),
        // Named INSTANCE expressions: InChan == INSTANCE Channel WITH ...
        // We convert these to a special InstanceExpr which stores module name + substitutions
        SyntaxKind::InstanceDecl => lower_instance_expr(ctx, node),
        // Operator reference: bare operator as value (+, -, *, \cup, etc.)
        SyntaxKind::OperatorRef => lower_operator_ref(ctx, node),

        // Try to handle token nodes (leaf nodes that are expressions)
        _ => lower_leaf_expr(ctx, node),
    }
}

/// Lower a leaf expression (identifier, number, string)
fn lower_leaf_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut pending_underscore = false;
    // Check if this is a token wrapped in a node
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            match token.kind() {
                SyntaxKind::Ident => {
                    if pending_underscore {
                        return Some(Expr::Ident(format!("_{}", token.text())));
                    }
                    return Some(Expr::Ident(token.text().to_string()));
                }
                SyntaxKind::Underscore => {
                    pending_underscore = true;
                }
                SyntaxKind::At => {
                    return Some(Expr::Ident("@".to_string()));
                }
                SyntaxKind::Number => {
                    if let Ok(n) = token.text().parse::<BigInt>() {
                        return Some(Expr::Int(n));
                    }
                }
                SyntaxKind::String => {
                    // Remove quotes
                    let s = token.text();
                    let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                    return Some(Expr::String(inner.to_string()));
                }
                SyntaxKind::TrueKw => {
                    return Some(Expr::Bool(true));
                }
                SyntaxKind::FalseKw => {
                    return Some(Expr::Bool(false));
                }
                SyntaxKind::BooleanKw => {
                    return Some(Expr::Ident("BOOLEAN".to_string()));
                }
                _ => {}
            }
        }
    }

    // Try children nodes
    for child in node.children() {
        if let Some(expr) = lower_expr(ctx, &child) {
            return Some(expr);
        }
    }

    None
}

/// Lower an operator reference (bare operator as value: +, -, *, \cup, etc.)
fn lower_operator_ref(_ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    // OperatorRef contains the operator token, we need to convert it to a string name
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            let op_name = match token.kind() {
                SyntaxKind::PlusOp => "+",
                SyntaxKind::MinusOp => "-",
                SyntaxKind::StarOp => "*",
                SyntaxKind::SlashOp => "/",
                SyntaxKind::DivOp => "\\div",
                SyntaxKind::PercentOp => "%",
                SyntaxKind::CaretOp => "^",
                // Multi-character user-definable operators
                SyntaxKind::PlusPlusOp => "++",
                SyntaxKind::MinusMinusOp => "--",
                SyntaxKind::StarStarOp => "**",
                SyntaxKind::SlashSlashOp => "//",
                SyntaxKind::CaretCaretOp => "^^",
                SyntaxKind::PercentPercentOp => "%%",
                SyntaxKind::AmpAmpOp => "&&",
                // Circled operators (user-definable)
                SyntaxKind::OplusOp => "\\oplus",
                SyntaxKind::OminusOp => "\\ominus",
                SyntaxKind::OtimesOp => "\\otimes",
                SyntaxKind::OslashOp => "\\oslash",
                SyntaxKind::OdotOp => "\\odot",
                SyntaxKind::UplusOp => "\\uplus",
                SyntaxKind::SqcapOp => "\\sqcap",
                SyntaxKind::SqcupOp => "\\sqcup",
                SyntaxKind::EqOp => "=",
                SyntaxKind::NeqOp => "/=",
                SyntaxKind::LtOp => "<",
                SyntaxKind::GtOp => ">",
                SyntaxKind::LeqOp => "<=",
                SyntaxKind::GeqOp => ">=",
                SyntaxKind::InOp => "\\in",
                SyntaxKind::NotInOp => "\\notin",
                SyntaxKind::SubseteqOp => "\\subseteq",
                SyntaxKind::SubsetOp => "\\subset",
                SyntaxKind::SupseteqOp => "\\supseteq",
                SyntaxKind::SupsetOp => "\\supset",
                SyntaxKind::UnionOp => "\\cup",
                SyntaxKind::IntersectOp => "\\cap",
                SyntaxKind::SetMinusOp => "\\",
                SyntaxKind::ConcatOp => "\\o",
                SyntaxKind::ImpliesOp => "=>",
                SyntaxKind::EquivOp => "<=>",
                // Ordering relations (user-definable)
                SyntaxKind::PrecOp => "\\prec",
                SyntaxKind::PreceqOp => "\\preceq",
                SyntaxKind::SuccOp => "\\succ",
                SyntaxKind::SucceqOp => "\\succeq",
                // Action composition
                SyntaxKind::CdotOp => "\\cdot",
                // User-definable infix operators
                SyntaxKind::Pipe => "|",
                SyntaxKind::Amp => "&",
                // BNF production operator
                SyntaxKind::ColonColonEqOp => "::=",
                _ => continue,
            };
            return Some(Expr::OpRef(op_name.to_string()));
        }
    }
    None
}

fn operator_token_to_name(kind: SyntaxKind) -> Option<&'static str> {
    Some(match kind {
        // Arithmetic
        SyntaxKind::PlusOp => "+",
        SyntaxKind::MinusOp => "-",
        SyntaxKind::StarOp => "*",
        SyntaxKind::SlashOp => "/",
        SyntaxKind::DivOp => "\\div",
        SyntaxKind::PercentOp => "%",
        SyntaxKind::CaretOp => "^",
        // Multi-character user-definable operators
        SyntaxKind::PlusPlusOp => "++",
        SyntaxKind::MinusMinusOp => "--",
        SyntaxKind::StarStarOp => "**",
        SyntaxKind::SlashSlashOp => "//",
        SyntaxKind::CaretCaretOp => "^^",
        SyntaxKind::PercentPercentOp => "%%",
        SyntaxKind::AmpAmpOp => "&&",
        // Circled operators (user-definable)
        SyntaxKind::OplusOp => "\\oplus",
        SyntaxKind::OminusOp => "\\ominus",
        SyntaxKind::OtimesOp => "\\otimes",
        SyntaxKind::OslashOp => "\\oslash",
        SyntaxKind::OdotOp => "\\odot",
        SyntaxKind::UplusOp => "\\uplus",
        SyntaxKind::SqcapOp => "\\sqcap",
        SyntaxKind::SqcupOp => "\\sqcup",
        // Logic
        SyntaxKind::AndOp => "/\\",
        SyntaxKind::OrOp => "\\/",
        SyntaxKind::ImpliesOp => "=>",
        SyntaxKind::EquivOp => "<=>",
        // Comparison
        SyntaxKind::EqOp => "=",
        SyntaxKind::NeqOp => "/=",
        SyntaxKind::LtOp => "<",
        SyntaxKind::GtOp => ">",
        SyntaxKind::LeqOp => "<=",
        SyntaxKind::GeqOp => ">=",
        // Ordering relations (user-definable)
        SyntaxKind::PrecOp => "\\prec",
        SyntaxKind::PreceqOp => "\\preceq",
        SyntaxKind::SuccOp => "\\succ",
        SyntaxKind::SucceqOp => "\\succeq",
        SyntaxKind::LlOp => "\\ll",
        SyntaxKind::GgOp => "\\gg",
        SyntaxKind::SimOp => "\\sim",
        SyntaxKind::SimeqOp => "\\simeq",
        SyntaxKind::AsympOp => "\\asymp",
        SyntaxKind::ApproxOp => "\\approx",
        SyntaxKind::CongOp => "\\cong",
        SyntaxKind::DoteqOp => "\\doteq",
        SyntaxKind::ProptoOp => "\\propto",
        // Sets
        SyntaxKind::InOp => "\\in",
        SyntaxKind::NotInOp => "\\notin",
        SyntaxKind::SubseteqOp => "\\subseteq",
        SyntaxKind::SubsetOp => "\\subset",
        SyntaxKind::SupseteqOp => "\\supseteq",
        SyntaxKind::SupsetOp => "\\supset",
        SyntaxKind::UnionOp => "\\cup",
        SyntaxKind::IntersectOp => "\\cap",
        SyntaxKind::SetMinusOp => "\\",
        SyntaxKind::SqsubseteqOp => "\\sqsubseteq",
        SyntaxKind::SqsupseteqOp => "\\sqsupseteq",
        // Sequences/functions
        SyntaxKind::ConcatOp => "\\o",
        // Action composition
        SyntaxKind::CdotOp => "\\cdot",
        // User-definable infix operators
        SyntaxKind::Pipe => "|",
        SyntaxKind::Amp => "&",
        // TLC module operators that use infix syntax
        SyntaxKind::ColonGt => ":>",
        SyntaxKind::AtAt => "@@",
        // BNF
        SyntaxKind::ColonColonEqOp => "::=",
        _ => return None,
    })
}

fn make_apply_binary(op_name: &str, left: Spanned<Expr>, right: Spanned<Expr>) -> Expr {
    Expr::Apply(
        Box::new(Spanned::dummy(Expr::Ident(op_name.to_string()))),
        vec![left, right],
    )
}

/// Lower parenthesized expression
fn lower_paren_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    for child in node.children() {
        if let Some(expr) = lower_expr(ctx, &child) {
            return Some(expr);
        }
    }
    lower_leaf_expr(ctx, node)
}

/// Lower binary expression
fn lower_binary_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut exprs = Vec::new();
    let mut op: Option<SyntaxKind> = None;
    // Track pending identifier that might be followed by ApplyExpr
    let mut pending_ident: Option<Spanned<Expr>> = None;
    // Track underscore-prefixed identifiers like `_msgs`
    let mut pending_underscore = false;
    // Track parenthesized operators like (+), (-), etc.
    // In TLA+, (op) is a user-definable operator distinct from op.
    // E.g., (+) is \oplus (circled plus), not + (arithmetic plus).
    let mut saw_lparen = false;
    let mut op_is_parenthesized = false;
    // Track whether we've seen an operand (node) yet.
    // Operators BEFORE any operand are "bullets" (leading markers in bullet lists).
    // The actual operator is the FIRST one that appears AFTER an operand.
    let mut saw_operand = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind == SyntaxKind::LParen {
                    // Track LParen to detect parenthesized operators
                    saw_lparen = true;
                    continue;
                }
                if kind == SyntaxKind::RParen {
                    // RParen after operator confirms it was parenthesized
                    continue;
                }
                if kind == SyntaxKind::Underscore {
                    // Flush any existing pending ident first
                    if let Some(ident) = pending_ident.take() {
                        exprs.push(ident);
                        saw_operand = true;
                    }
                    pending_underscore = true;
                    saw_lparen = false;
                    continue;
                }
                if is_binary_op(kind) {
                    // Flush pending identifier before operator
                    if let Some(ident) = pending_ident.take() {
                        exprs.push(ident);
                        saw_operand = true;
                    }
                    pending_underscore = false;
                    // Take the FIRST operator that appears AFTER an operand.
                    // Operators before any operand are "bullets" in bullet lists:
                    //   /\ case1       <- /\ is a bullet (before case1)
                    //   \/ case2       <- \/ is the actual operator (after case1)
                    //   /\ case3       <- /\ is a bullet (after the operator)
                    // The \/ between case1 and case2 is the actual combining operator.
                    //
                    // EXCEPTION: WF_ and SF_ are prefix-style binary operators:
                    //   WF_<<vars>>(Action)
                    // Here WF_ appears before the operands, so we should capture it
                    // even before seeing any operand.
                    let is_prefix_binary = matches!(kind, SyntaxKind::WeakFairKw | SyntaxKind::StrongFairKw);
                    if op.is_none() && (saw_operand || is_prefix_binary) {
                        op_is_parenthesized = saw_lparen;
                        op = Some(kind);
                    }
                    saw_lparen = false;
                // Handle stdlib keyword tokens (Len, Head, etc.) which precede ApplyExpr
                } else if let Some(op_name) = stdlib_keyword_to_name(kind) {
                    // Flush any existing pending ident first
                    if let Some(ident) = pending_ident.take() {
                        exprs.push(ident);
                    }
                    let span = ctx.token_span(&token);
                    pending_ident = Some(Spanned::new(Expr::Ident(op_name), span));
                    pending_underscore = false;
                } else if kind == SyntaxKind::Ident
                    || kind == SyntaxKind::At
                    || kind == SyntaxKind::Number
                    || kind == SyntaxKind::String
                    || kind == SyntaxKind::TrueKw
                    || kind == SyntaxKind::FalseKw
                    || kind == SyntaxKind::BooleanKw
                {
                    // Flush any existing pending ident first
                    if let Some(ident) = pending_ident.take() {
                        exprs.push(ident);
                    }

                    // Leaf token as operand
                    let expr = match kind {
                        SyntaxKind::Ident => {
                            if pending_underscore {
                                pending_underscore = false;
                                Expr::Ident(format!("_{}", token.text()))
                            } else {
                                Expr::Ident(token.text().to_string())
                            }
                        }
                        SyntaxKind::At => Expr::Ident("@".to_string()),
                        SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int)?,
                        SyntaxKind::String => {
                            let s = token.text();
                            let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                            Expr::String(inner.to_string())
                        }
                        SyntaxKind::TrueKw => Expr::Bool(true),
                        SyntaxKind::FalseKw => Expr::Bool(false),
                        SyntaxKind::BooleanKw => Expr::Ident("BOOLEAN".to_string()),
                        _ => return None,
                    };
                    let span = ctx.token_span(&token);

                    // For identifiers, keep pending in case ApplyExpr follows
                    if matches!(expr, Expr::Ident(_)) {
                        pending_ident = Some(Spanned::new(expr, span));
                    } else {
                        exprs.push(Spanned::new(expr, span));
                        saw_operand = true;
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                // Check for ApplyExpr following an identifier
                if child_node.kind() == SyntaxKind::ApplyExpr {
                    if let Some(ident) = pending_ident.take() {
                        // Combine identifier with ApplyExpr args (handles WF_/SF_ identifiers)
                        let args = lower_apply_args(ctx, &child_node);
                        let span = ctx.span(&child_node);
                        exprs.push(Spanned::new(make_apply_expr(ident, args), span));
                        saw_operand = true;
                        continue;
                    }
                }

                // Flush pending ident if not ApplyExpr
                if let Some(ident) = pending_ident.take() {
                    exprs.push(ident);
                    saw_operand = true;
                }

                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    exprs.push(Spanned::new(expr, span));
                    saw_operand = true;
                }
            }
        }
    }

    // Flush any remaining pending identifier
    if let Some(ident) = pending_ident.take() {
        exprs.push(ident);
    }

    if exprs.len() < 2 || op.is_none() {
        return exprs.into_iter().next().map(|s| s.node);
    }

    let op = op?;
    let left = exprs.remove(0);
    let right = exprs.remove(0);

    Some(make_binary_expr(op, op_is_parenthesized, left, right))
}

/// Make a binary expression from operator and operands
/// If `parenthesized` is true, use the circled form for operators like (+) -> \oplus
fn make_binary_expr(
    op: SyntaxKind,
    parenthesized: bool,
    left: Spanned<Expr>,
    right: Spanned<Expr>,
) -> Expr {
    let left = Box::new(left);
    let right = Box::new(right);

    // Handle parenthesized operators first - these map to different operators
    // In TLA+, (op) is a user-definable operator distinct from op:
    // - (+) is \oplus (circled plus), not + (arithmetic)
    // - (-) is \ominus (circled minus), not - (arithmetic)
    // - (*) is \otimes (circled times), not * (arithmetic)
    if parenthesized {
        let op_name = match op {
            SyntaxKind::PlusOp => "\\oplus",
            SyntaxKind::MinusOp => "\\ominus",
            SyntaxKind::StarOp => "\\otimes",
            SyntaxKind::SlashOp => "\\oslash",
            // Other operators keep their regular name (they don't have a distinct circled form)
            _ => return make_apply_from_op_kind(op, *left, *right),
        };
        return make_apply_binary(op_name, *left, *right);
    }

    match op {
        SyntaxKind::AndOp => Expr::And(left, right),
        SyntaxKind::OrOp => Expr::Or(left, right),
        SyntaxKind::ImpliesOp => Expr::Implies(left, right),
        SyntaxKind::EquivOp => Expr::Equiv(left, right),
        SyntaxKind::EqOp => Expr::Eq(left, right),
        SyntaxKind::NeqOp => Expr::Neq(left, right),
        SyntaxKind::LtOp => Expr::Lt(left, right),
        SyntaxKind::GtOp => Expr::Gt(left, right),
        SyntaxKind::LeqOp => Expr::Leq(left, right),
        SyntaxKind::GeqOp => Expr::Geq(left, right),
        SyntaxKind::InOp => Expr::In(left, right),
        SyntaxKind::NotInOp => Expr::NotIn(left, right),
        SyntaxKind::SubseteqOp => Expr::Subseteq(left, right),
        SyntaxKind::UnionOp => Expr::Union(left, right),
        SyntaxKind::IntersectOp => Expr::Intersect(left, right),
        SyntaxKind::SetMinusOp => Expr::SetMinus(left, right),
        SyntaxKind::PlusOp => Expr::Add(left, right),
        SyntaxKind::MinusOp => Expr::Sub(left, right),
        SyntaxKind::StarOp => Expr::Mul(left, right),
        SyntaxKind::SlashOp => Expr::Div(left, right),
        SyntaxKind::DivOp => Expr::IntDiv(left, right),
        SyntaxKind::PercentOp => Expr::Mod(left, right),
        SyntaxKind::CaretOp => Expr::Pow(left, right),
        SyntaxKind::DotDotOp => Expr::Range(left, right),
        SyntaxKind::LeadsToOp => Expr::LeadsTo(left, right),
        SyntaxKind::TimesOp => {
            // Flatten nested Times expressions so A \X B \X C produces Times([A, B, C])
            // rather than Times([Times([A, B]), C]), which ensures tuples are flat.
            match left.node {
                Expr::Times(mut factors) => {
                    factors.push(*right);
                    Expr::Times(factors)
                }
                _ => Expr::Times(vec![*left, *right]),
            }
        }
        SyntaxKind::ConcatOp => {
            // Sequence concatenation - model as Apply
            make_apply_binary("\\o", *left, *right)
        }
        // TLC module function operators
        SyntaxKind::ColonGt => {
            // d :> e creates single-element function [d |-> e]
            make_apply_binary(":>", *left, *right)
        }
        SyntaxKind::AtAt => {
            // f @@ g merges two functions (f takes priority)
            make_apply_binary("@@", *left, *right)
        }
        // Fairness operators
        SyntaxKind::WeakFairKw => Expr::WeakFair(left, right),
        SyntaxKind::StrongFairKw => Expr::StrongFair(left, right),
        _ => {
            // Unknown binary operator - wrap as application using its concrete token text when possible.
            match operator_token_to_name(op) {
                Some(op_name) => make_apply_binary(op_name, *left, *right),
                None => Expr::Apply(
                    Box::new(Spanned::dummy(Expr::Ident(format!("{:?}", op)))),
                    vec![*left, *right],
                ),
            }
        }
    }
}

/// Make an Apply expression from an operator kind
fn make_apply_from_op_kind(op: SyntaxKind, left: Spanned<Expr>, right: Spanned<Expr>) -> Expr {
    match operator_token_to_name(op) {
        Some(op_name) => make_apply_binary(op_name, left, right),
        None => Expr::Apply(
            Box::new(Spanned::dummy(Expr::Ident(format!("{:?}", op)))),
            vec![left, right],
        ),
    }
}

/// Check if a syntax kind is a binary operator
fn is_binary_op(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::AndOp
            | SyntaxKind::OrOp
            | SyntaxKind::ImpliesOp
            | SyntaxKind::EquivOp
            | SyntaxKind::ColonColonEqOp
            | SyntaxKind::EqOp
            | SyntaxKind::NeqOp
            | SyntaxKind::LtOp
            | SyntaxKind::GtOp
            | SyntaxKind::LeqOp
            | SyntaxKind::GeqOp
            // Ordering relations (user-definable)
            | SyntaxKind::PrecOp
            | SyntaxKind::PreceqOp
            | SyntaxKind::SuccOp
            | SyntaxKind::SucceqOp
            | SyntaxKind::LlOp
            | SyntaxKind::GgOp
            | SyntaxKind::SimOp
            | SyntaxKind::SimeqOp
            | SyntaxKind::AsympOp
            | SyntaxKind::ApproxOp
            | SyntaxKind::CongOp
            | SyntaxKind::DoteqOp
            | SyntaxKind::ProptoOp
            | SyntaxKind::InOp
            | SyntaxKind::NotInOp
            | SyntaxKind::SubseteqOp
            | SyntaxKind::SubsetOp
            | SyntaxKind::SupseteqOp
            | SyntaxKind::SupsetOp
            | SyntaxKind::UnionOp
            | SyntaxKind::IntersectOp
            | SyntaxKind::SetMinusOp
            // Bag subset operators (user-definable)
            | SyntaxKind::SqsubseteqOp
            | SyntaxKind::SqsupseteqOp
            | SyntaxKind::PlusOp
            | SyntaxKind::MinusOp
            // Multi-character user-definable operators
            | SyntaxKind::PlusPlusOp
            | SyntaxKind::MinusMinusOp
            | SyntaxKind::StarOp
            | SyntaxKind::SlashOp
            | SyntaxKind::DivOp
            | SyntaxKind::PercentOp
            | SyntaxKind::StarStarOp
            | SyntaxKind::SlashSlashOp
            | SyntaxKind::PercentPercentOp
            | SyntaxKind::CaretOp
            | SyntaxKind::CaretCaretOp
            // Circled operators (user-definable)
            | SyntaxKind::OplusOp
            | SyntaxKind::OminusOp
            | SyntaxKind::OtimesOp
            | SyntaxKind::OslashOp
            | SyntaxKind::OdotOp
            | SyntaxKind::UplusOp
            | SyntaxKind::SqcapOp
            | SyntaxKind::SqcupOp
            // Action composition
            | SyntaxKind::CdotOp
            | SyntaxKind::DotDotOp
            | SyntaxKind::LeadsToOp
            | SyntaxKind::TimesOp
            | SyntaxKind::ConcatOp
            // User-defined infix operator symbols
            | SyntaxKind::Pipe
            | SyntaxKind::Amp
            | SyntaxKind::AmpAmpOp
            | SyntaxKind::WeakFairKw
            | SyntaxKind::StrongFairKw
            | SyntaxKind::ColonGt
            | SyntaxKind::AtAt
    )
}

/// Lower unary expression
fn lower_unary_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut op: Option<SyntaxKind> = None;
    let mut operand: Option<Spanned<Expr>> = None;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if is_unary_op(kind) {
                    op = Some(kind);
                } else {
                    // Handle leaf token as operand (for postfix ops like Prime)
                    let expr = match kind {
                        SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                        SyntaxKind::At => Some(Expr::Ident("@".to_string())),
                        SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                        SyntaxKind::String => {
                            let s = token.text();
                            let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                            Some(Expr::String(inner.to_string()))
                        }
                        SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                        SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                        _ => None,
                    };
                    if let Some(e) = expr {
                        let span = ctx.token_span(&token);
                        operand = Some(Spanned::new(e, span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    operand = Some(Spanned::new(expr, span));
                }
            }
        }
    }

    let op = op?;
    let operand = operand?;

    Some(make_unary_expr(op, operand))
}

/// Make a unary expression from operator and operand
fn make_unary_expr(op: SyntaxKind, operand: Spanned<Expr>) -> Expr {
    let operand = Box::new(operand);

    match op {
        SyntaxKind::NotOp => Expr::Not(operand),
        SyntaxKind::MinusOp => Expr::Neg(operand),
        SyntaxKind::AlwaysOp => Expr::Always(operand),
        SyntaxKind::EventuallyOp => Expr::Eventually(operand),
        SyntaxKind::EnabledKw => Expr::Enabled(operand),
        SyntaxKind::UnchangedKw => Expr::Unchanged(operand),
        SyntaxKind::PowersetKw => Expr::Powerset(operand),
        SyntaxKind::BigUnionKw => Expr::BigUnion(operand),
        SyntaxKind::DomainKw => Expr::Domain(operand),
        SyntaxKind::PrimeOp => Expr::Prime(operand),
        _ => {
            // Unknown unary operator
            Expr::Apply(
                Box::new(Spanned::dummy(Expr::Ident(format!("{:?}", op)))),
                vec![*operand],
            )
        }
    }
}

/// Check if a syntax kind is a unary operator
fn is_unary_op(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::NotOp
            | SyntaxKind::MinusOp
            | SyntaxKind::AlwaysOp
            | SyntaxKind::EventuallyOp
            | SyntaxKind::EnabledKw
            | SyntaxKind::UnchangedKw
            | SyntaxKind::PowersetKw
            | SyntaxKind::BigUnionKw
            | SyntaxKind::DomainKw
            | SyntaxKind::PrimeOp
    )
}

/// Lower operator application
fn lower_apply_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut op: Option<Spanned<Expr>> = None;
    let mut args = Vec::new();

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                // Handle identifier tokens as operator names
                if kind == SyntaxKind::Ident && op.is_none() {
                    let span = ctx.token_span(&token);
                    op = Some(Spanned::new(Expr::Ident(token.text().to_string()), span));
                }
                // Handle stdlib keyword tokens (Len, Head, Tail, SelectSeq, etc.)
                else if op.is_none() {
                    let op_name = match kind {
                        SyntaxKind::LenKw => Some("Len"),
                        SyntaxKind::SeqKw => Some("Seq"),
                        SyntaxKind::SubSeqKw => Some("SubSeq"),
                        SyntaxKind::SelectSeqKw => Some("SelectSeq"),
                        SyntaxKind::HeadKw => Some("Head"),
                        SyntaxKind::TailKw => Some("Tail"),
                        SyntaxKind::AppendKw => Some("Append"),
                        _ => None,
                    };
                    if let Some(name) = op_name {
                        let span = ctx.token_span(&token);
                        op = Some(Spanned::new(Expr::Ident(name.to_string()), span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::ArgList {
                    args = lower_arg_list(ctx, &child_node);
                } else if op.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        op = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some(make_apply_expr(op?, args))
}

/// Lower argument list
fn lower_arg_list(ctx: &mut LowerCtx, node: &SyntaxNode) -> Vec<Spanned<Expr>> {
    let mut args = Vec::new();

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                // Handle leaf tokens as arguments
                let kind = token.kind();
                let expr = match kind {
                    SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                    SyntaxKind::At => Some(Expr::Ident("@".to_string())),
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    _ => None,
                };
                if let Some(e) = expr {
                    let span = ctx.token_span(&token);
                    args.push(Spanned::new(e, span));
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    args.push(Spanned::new(expr, span));
                }
            }
        }
    }

    args
}

/// Lower lambda expression
fn lower_lambda_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut params = Vec::new();
    let mut body: Option<Spanned<Expr>> = None;
    let mut after_colon = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                match token.kind() {
                    SyntaxKind::Colon => {
                        after_colon = true;
                    }
                    SyntaxKind::Ident if !after_colon => {
                        let span = ctx.token_span(&token);
                        params.push(Spanned::new(token.text().to_string(), span));
                    }
                    // Lambda bodies can be a single leaf token (e.g., `LAMBDA x : 0`).
                    // In that case, the CST stores the body as a token rather than a child node.
                    _ if after_colon && body.is_none() => {
                        let expr = match token.kind() {
                            SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                            SyntaxKind::At => Some(Expr::Ident("@".to_string())),
                            SyntaxKind::Number => {
                                token.text().parse::<BigInt>().ok().map(Expr::Int)
                            }
                            SyntaxKind::String => {
                                let s = token.text();
                                let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                                Some(Expr::String(inner.to_string()))
                            }
                            SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                            SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                            SyntaxKind::BooleanKw => Some(Expr::Ident("BOOLEAN".to_string())),
                            _ => None,
                        };
                        if let Some(e) = expr {
                            let span = ctx.token_span(&token);
                            body = Some(Spanned::new(e, span));
                        }
                    }
                    _ => {}
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if body.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        body = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some(Expr::Lambda(params, Box::new(body?)))
}

/// Lower quantified expression
fn lower_quant_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut is_forall = false;
    let mut bounds = Vec::new();
    let mut body: Option<Spanned<Expr>> = None;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                match kind {
                    SyntaxKind::ForallKw => is_forall = true,
                    SyntaxKind::ExistsKw => is_forall = false,
                    _ => {
                        // Handle bare token bodies like TRUE, FALSE, identifiers
                        if body.is_none() && !bounds.is_empty() {
                            let expr = match kind {
                                SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                                SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                                SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                                SyntaxKind::Number => {
                                    token.text().parse::<BigInt>().ok().map(Expr::Int)
                                }
                                SyntaxKind::String => {
                                    let s = token.text();
                                    let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                                    Some(Expr::String(inner.to_string()))
                                }
                                _ => None,
                            };
                            if let Some(e) = expr {
                                let span = ctx.token_span(&token);
                                body = Some(Spanned::new(e, span));
                            }
                        }
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::BoundVar {
                    if let Some(bv) = lower_bound_var(ctx, &child_node) {
                        bounds.push(bv);
                    }
                } else if body.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        body = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    // In TLA+, syntax like `\A a, b \in S : P` means both a and b are in S.
    // The parser may produce bounds where only the last one has a domain.
    // Propagate the domain from the last bound with a domain to all earlier
    // bounds that don't have one.
    propagate_bound_domains(&mut bounds);

    let body = Box::new(body?);

    if is_forall {
        Some(Expr::Forall(bounds, body))
    } else {
        Some(Expr::Exists(bounds, body))
    }
}

/// Propagate domains in bound variable lists.
/// In TLA+, `\A a, b \in S` means both `a` and `b` are in S.
/// This function finds the domain from the last bound that has one
/// and propagates it to all earlier bounds without domains.
fn propagate_bound_domains(bounds: &mut [BoundVar]) {
    // TLA+ shorthand like `a, b \in S, c \in T` means:
    // - a \in S
    // - b \in S
    // - c \in T
    //
    // The parser may only attach the domain to the last identifier in a comma-run.
    // Propagate domains backwards within each run, stopping at explicit domains.
    let mut last_domain: Option<Box<Spanned<Expr>>> = None;
    for bound in bounds.iter_mut().rev() {
        if bound.domain.is_some() {
            last_domain = bound.domain.clone();
            continue;
        }
        if let Some(domain) = &last_domain {
            bound.domain = Some(domain.clone());
        }
    }
}

/// Lower bound variable
fn lower_bound_var(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<BoundVar> {
    let mut name: Option<Spanned<String>> = None;
    let mut domain: Option<Box<Spanned<Expr>>> = None;
    let mut pattern: Option<BoundPattern> = None;
    let mut seen_in = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind == SyntaxKind::InOp {
                    seen_in = true;
                } else if kind == SyntaxKind::Ident {
                    if !seen_in && name.is_none() {
                        // This is the variable name
                        let span = ctx.token_span(&token);
                        name = Some(Spanned::new(token.text().to_string(), span));
                    } else if seen_in && domain.is_none() {
                        // This is a simple domain (identifier)
                        let span = ctx.token_span(&token);
                        domain = Some(Box::new(Spanned::new(
                            Expr::Ident(token.text().to_string()),
                            span,
                        )));
                    }
                } else if seen_in && domain.is_none() {
                    // Handle other token types as domain
                    let expr = match kind {
                        SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                        SyntaxKind::String => {
                            let s = token.text();
                            let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                            Some(Expr::String(inner.to_string()))
                        }
                        SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                        SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                        // BOOLEAN is a built-in set keyword
                        SyntaxKind::BooleanKw => Some(Expr::Ident(token.text().to_string())),
                        _ => None,
                    };
                    if let Some(e) = expr {
                        let span = ctx.token_span(&token);
                        domain = Some(Box::new(Spanned::new(e, span)));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                let node_kind = child_node.kind();
                if !seen_in && node_kind == SyntaxKind::TuplePattern {
                    // Handle tuple pattern: <<x, y>>
                    let (pat, pat_name) = lower_tuple_pattern(ctx, &child_node);
                    pattern = Some(pat);
                    name = Some(pat_name);
                } else if seen_in && domain.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        domain = Some(Box::new(Spanned::new(expr, span)));
                    }
                }
            }
        }
    }

    Some(BoundVar {
        name: name?,
        domain,
        pattern,
    })
}

/// Lower tuple pattern <<x, y, ...>> and return the pattern and a synthetic name
fn lower_tuple_pattern(ctx: &mut LowerCtx, node: &SyntaxNode) -> (BoundPattern, Spanned<String>) {
    let mut vars = Vec::new();
    let span = ctx.span(node);

    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            if token.kind() == SyntaxKind::Ident {
                let token_span = ctx.token_span(&token);
                vars.push(Spanned::new(token.text().to_string(), token_span));
            }
        }
    }

    // Create a synthetic name from the pattern variables
    let synthetic_name = format!(
        "<<{}>>",
        vars.iter()
            .map(|v| v.node.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    (
        BoundPattern::Tuple(vars),
        Spanned::new(synthetic_name, span),
    )
}

/// Lower CHOOSE expression
fn lower_choose_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut bound: Option<BoundVar> = None;
    let mut body: Option<Spanned<Expr>> = None;
    let mut after_colon = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind.is_trivia() {
                    continue;
                }
                if kind == SyntaxKind::Colon {
                    after_colon = true;
                    continue;
                }
                if !after_colon || body.is_some() {
                    continue;
                }

                let expr = match kind {
                    SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    SyntaxKind::BooleanKw => Some(Expr::Ident("BOOLEAN".to_string())),
                    _ => None,
                };

                if let Some(e) = expr {
                    let span = ctx.token_span(&token);
                    body = Some(Spanned::new(e, span));
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::BoundVar {
                    bound = lower_bound_var(ctx, &child_node);
                } else if body.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        body = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some(Expr::Choose(bound?, Box::new(body?)))
}

/// Lower set enumeration
fn lower_set_enum_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut elements = Vec::new();

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let expr = match token.kind() {
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    _ => None,
                };
                if let Some(e) = expr {
                    let span = ctx.token_span(&token);
                    elements.push(Spanned::new(e, span));
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    elements.push(Spanned::new(expr, span));
                }
            }
        }
    }

    Some(Expr::SetEnum(elements))
}

/// Lower set builder expression {expr : x \in S}
/// Handles multi-variable patterns like {<<x, y>> : x, y \in S} where x and y share domain S
fn lower_set_builder_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut body: Option<Spanned<Expr>> = None;
    let mut bounds = Vec::new();

    for child in node.children() {
        if child.kind() == SyntaxKind::BoundVar {
            if let Some(bv) = lower_bound_var(ctx, &child) {
                bounds.push(bv);
            }
        } else if body.is_none() {
            if let Some(expr) = lower_expr(ctx, &child) {
                let span = ctx.span(&child);
                body = Some(Spanned::new(expr, span));
            }
        }
    }

    // Propagate domains from variables that have them to preceding variables that don't.
    // This handles patterns like "x, y \in S" where both x and y share domain S.
    if !bounds.is_empty() {
        let mut last_domain: Option<Box<Spanned<Expr>>> = None;

        // Iterate backwards to find domain and propagate to preceding variables
        for i in (0..bounds.len()).rev() {
            if bounds[i].domain.is_some() {
                last_domain = bounds[i].domain.clone();
            } else if let Some(ref d) = last_domain {
                bounds[i].domain = Some(d.clone());
            }
        }
    }

    Some(Expr::SetBuilder(Box::new(body?), bounds))
}

/// Lower set filter expression {x \in S : P}
fn lower_set_filter_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut bound: Option<BoundVar> = None;
    let mut predicate: Option<Spanned<Expr>> = None;
    let mut past_colon = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                // Track when we pass the colon to know the predicate starts
                if kind == SyntaxKind::Colon {
                    past_colon = true;
                } else if past_colon && predicate.is_none() {
                    // Handle bare token predicates (TRUE, FALSE, identifier, number, string)
                    let expr = match kind {
                        SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                        SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                        SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                        SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                        SyntaxKind::String => {
                            let s = token.text();
                            let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                            Some(Expr::String(inner.to_string()))
                        }
                        _ => None,
                    };
                    if let Some(e) = expr {
                        let span = ctx.token_span(&token);
                        predicate = Some(Spanned::new(e, span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::BoundVar {
                    bound = lower_bound_var(ctx, &child_node);
                } else if predicate.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        predicate = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some(Expr::SetFilter(bound?, Box::new(predicate?)))
}

/// Lower function definition [x \in S |-> e]
fn lower_func_def_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut bounds = Vec::new();
    let mut body: Option<Spanned<Expr>> = None;
    let mut saw_maps_to = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind == SyntaxKind::MapsTo {
                    saw_maps_to = true;
                } else if saw_maps_to && body.is_none() {
                    // Handle simple token bodies (String, Number, Ident, Bool)
                    let expr = match kind {
                        SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                        SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                        SyntaxKind::String => {
                            let s = token.text();
                            let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                            Some(Expr::String(inner.to_string()))
                        }
                        SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                        SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                        _ => None,
                    };
                    if let Some(e) = expr {
                        let span = ctx.token_span(&token);
                        body = Some(Spanned::new(e, span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::BoundVar {
                    if let Some(bv) = lower_bound_var(ctx, &child_node) {
                        bounds.push(bv);
                    }
                } else if saw_maps_to && body.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        body = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    // Propagate domains from variables that have them to preceding variables that don't.
    // This handles patterns like "x, y \in S" where both x and y share domain S.
    if !bounds.is_empty() {
        let mut last_domain: Option<Box<Spanned<Expr>>> = None;
        for i in (0..bounds.len()).rev() {
            if bounds[i].domain.is_some() {
                last_domain = bounds[i].domain.clone();
            } else if let Some(ref d) = last_domain {
                bounds[i].domain = Some(d.clone());
            }
        }
    }

    Some(Expr::FuncDef(bounds, Box::new(body?)))
}

/// Lower function application f[x]
fn lower_func_apply_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut func: Option<Spanned<Expr>> = None;
    let mut args: Vec<Spanned<Expr>> = Vec::new();

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                let span = ctx.token_span(&token);

                // First, try to set func from Ident or @ (for EXCEPT expressions)
                if kind == SyntaxKind::Ident && func.is_none() {
                    func = Some(Spanned::new(Expr::Ident(token.text().to_string()), span));
                } else if kind == SyntaxKind::At && func.is_none() {
                    // @ is used inside EXCEPT specs to refer to the old value
                    func = Some(Spanned::new(Expr::Ident("@".to_string()), span));
                } else if func.is_some() {
                    // After func is set, handle literal tokens for args
                    let expr = match kind {
                        SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                        SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                        SyntaxKind::String => {
                            let s = token.text();
                            let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                            Some(Expr::String(inner.to_string()))
                        }
                        SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                        SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                        _ => None,
                    };
                    if let Some(e) = expr {
                        args.push(Spanned::new(e, span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    if func.is_none() {
                        func = Some(Spanned::new(expr, span));
                    } else {
                        args.push(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    let func = func?;
    let arg = match args.len() {
        0 => return None,
        1 => args.pop().unwrap(),
        _ => Spanned::new(Expr::Tuple(args), ctx.span(node)),
    };
    Some(Expr::FuncApply(Box::new(func), Box::new(arg)))
}

/// Lower function set [S -> T]
fn lower_func_set_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut domain: Option<Spanned<Expr>> = None;
    let mut range: Option<Spanned<Expr>> = None;
    let mut saw_arrow = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind == SyntaxKind::Arrow {
                    saw_arrow = true;
                } else if kind == SyntaxKind::Ident || kind == SyntaxKind::BooleanKw {
                    // Handle Ident and BOOLEAN keyword tokens as domain or range
                    let span = ctx.token_span(&token);
                    let expr = if kind == SyntaxKind::BooleanKw {
                        Expr::Ident("BOOLEAN".to_string())
                    } else {
                        Expr::Ident(token.text().to_string())
                    };
                    if !saw_arrow && domain.is_none() {
                        domain = Some(Spanned::new(expr, span));
                    } else if saw_arrow && range.is_none() {
                        range = Some(Spanned::new(expr, span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    if !saw_arrow && domain.is_none() {
                        domain = Some(Spanned::new(expr, span));
                    } else if saw_arrow && range.is_none() {
                        range = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some(Expr::FuncSet(Box::new(domain?), Box::new(range?)))
}

/// Lower EXCEPT expression
fn lower_except_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut base: Option<Spanned<Expr>> = None;
    let mut specs = Vec::new();
    let mut pending_underscore = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if base.is_some() {
                    continue;
                }
                match token.kind() {
                    SyntaxKind::Underscore => {
                        pending_underscore = true;
                    }
                    SyntaxKind::Ident => {
                        let span = ctx.token_span(&token);
                        let name = if pending_underscore {
                            pending_underscore = false;
                            format!("_{}", token.text())
                        } else {
                            token.text().to_string()
                        };
                        base = Some(Spanned::new(Expr::Ident(name), span));
                    }
                    // @ is used inside EXCEPT specs to refer to the old value
                    SyntaxKind::At => {
                        let span = ctx.token_span(&token);
                        base = Some(Spanned::new(Expr::Ident("@".to_string()), span));
                    }
                    _ => {}
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::ExceptSpec {
                    if let Some(spec) = lower_except_spec(ctx, &child_node) {
                        specs.push(spec);
                    }
                } else if base.is_none() {
                    // First expression node becomes the base
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        base = Some(Spanned::new(expr, span));
                    }
                } else {
                    // Base already set - these are postfix operations to chain
                    // This handles cases like `node[r].insts[1]` in an EXCEPT where
                    // the CST has FuncApplyExpr, RecordAccessExpr, FuncApplyExpr as siblings
                    let span = ctx.span(&child_node);
                    match child_node.kind() {
                        SyntaxKind::RecordAccessExpr => {
                            // Extract just the field name from partial RecordAccessExpr
                            if let Some(field) =
                                extract_field_from_partial_record_access(ctx, &child_node)
                            {
                                let current_base = base.take().unwrap();
                                base = Some(Spanned::new(
                                    Expr::RecordAccess(Box::new(current_base), field),
                                    span,
                                ));
                            }
                        }
                        SyntaxKind::FuncApplyExpr => {
                            // Extract just the args from partial FuncApplyExpr
                            let args = extract_args_from_partial_func_apply(ctx, &child_node);
                            if !args.is_empty() {
                                let current_base = base.take().unwrap();
                                // Build nested FuncApply for each arg
                                let mut result = current_base;
                                for arg in args {
                                    result = Spanned::new(
                                        Expr::FuncApply(Box::new(result), Box::new(arg)),
                                        span,
                                    );
                                }
                                base = Some(result);
                            }
                        }
                        _ => {
                            // Try lowering normally - might be a complete expression
                            if let Some(expr) = lower_expr(ctx, &child_node) {
                                base = Some(Spanned::new(expr, span));
                            }
                        }
                    }
                }
            }
        }
    }

    Some(Expr::Except(Box::new(base?), specs))
}

/// Extract just the field name from a partial RecordAccessExpr (one without a base)
fn extract_field_from_partial_record_access(
    ctx: &mut LowerCtx,
    node: &SyntaxNode,
) -> Option<Spanned<String>> {
    for child in node.children_with_tokens() {
        if let rowan::NodeOrToken::Token(token) = child {
            if token.kind() == SyntaxKind::Ident {
                let span = ctx.token_span(&token);
                return Some(Spanned::new(token.text().to_string(), span));
            }
        }
    }
    None
}

/// Extract just the args from a partial FuncApplyExpr (one without a base)
fn extract_args_from_partial_func_apply(
    ctx: &mut LowerCtx,
    node: &SyntaxNode,
) -> Vec<Spanned<Expr>> {
    let mut args = Vec::new();

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let span = ctx.token_span(&token);
                let expr = match token.kind() {
                    SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    _ => None,
                };
                if let Some(e) = expr {
                    args.push(Spanned::new(e, span));
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    args.push(Spanned::new(expr, span));
                }
            }
        }
    }

    args
}

/// Lower EXCEPT spec
fn lower_except_spec(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<ExceptSpec> {
    let mut path = Vec::new();
    let mut in_path = true;
    let mut after_dot = false; // Track if we just saw a dot (next ident is a field)
    let mut in_bracket = false; // Track if we're inside a bracket group
    let mut bracket_indices: Vec<Spanned<Expr>> = Vec::new(); // Collect indices within a bracket
    let mut bracket_start_span: Option<Span> = None;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                match token.kind() {
                    SyntaxKind::LBracket if in_path => {
                        in_bracket = true;
                        bracket_indices.clear();
                        bracket_start_span = Some(ctx.token_span(&token));
                    }
                    SyntaxKind::RBracket if in_bracket => {
                        // Close the bracket - create index element(s)
                        if bracket_indices.len() == 1 {
                            // Single index
                            path.push(ExceptPathElement::Index(bracket_indices.remove(0)));
                        } else if bracket_indices.len() > 1 {
                            // Multiple indices - create a tuple for multi-arg functions
                            let exprs: Vec<Spanned<Expr>> = std::mem::take(&mut bracket_indices);
                            let start =
                                bracket_start_span.unwrap_or_else(|| ctx.token_span(&token));
                            let end = ctx.token_span(&token);
                            let combined_span = Span::new(start.file, start.start, end.end);
                            let tuple_expr = Expr::Tuple(exprs);
                            path.push(ExceptPathElement::Index(Spanned::new(
                                tuple_expr,
                                combined_span,
                            )));
                        }
                        in_bracket = false;
                        bracket_start_span = None;
                    }
                    SyntaxKind::Comma if in_bracket => {
                        // Just a separator, continue collecting
                    }
                    SyntaxKind::Dot => {
                        // Next ident is a field
                        after_dot = true;
                    }
                    SyntaxKind::Ident if in_path && after_dot => {
                        // After a dot, this is a field access
                        let span = ctx.token_span(&token);
                        path.push(ExceptPathElement::Field(Spanned::new(
                            token.text().to_string(),
                            span,
                        )));
                        after_dot = false;
                    }
                    SyntaxKind::Ident if in_path => {
                        // Not after a dot - this is a variable reference in an index position
                        // e.g., ![self] where self is a variable
                        let span = ctx.token_span(&token);
                        let expr = Expr::Ident(token.text().to_string());
                        if in_bracket {
                            bracket_indices.push(Spanned::new(expr, span));
                        } else {
                            path.push(ExceptPathElement::Index(Spanned::new(expr, span)));
                        }
                    }
                    SyntaxKind::EqOp => {
                        in_path = false;
                    }
                    // Handle literal number tokens (parser doesn't wrap them in nodes)
                    SyntaxKind::Number => {
                        let span = ctx.token_span(&token);
                        let n = token
                            .text()
                            .parse::<num_bigint::BigInt>()
                            .unwrap_or_default();
                        let expr = Expr::Int(n);
                        if in_path {
                            if in_bracket {
                                bracket_indices.push(Spanned::new(expr, span));
                            } else {
                                path.push(ExceptPathElement::Index(Spanned::new(expr, span)));
                            }
                        }
                    }
                    // Handle literal string tokens
                    SyntaxKind::String => {
                        let span = ctx.token_span(&token);
                        let s = token
                            .text()
                            .trim_matches('"')
                            .replace("\\\"", "\"")
                            .replace("\\\\", "\\");
                        let expr = Expr::String(s);
                        if in_path {
                            if in_bracket {
                                bracket_indices.push(Spanned::new(expr, span));
                            } else {
                                path.push(ExceptPathElement::Index(Spanned::new(expr, span)));
                            }
                        }
                    }
                    // Handle boolean keywords
                    SyntaxKind::TrueKw => {
                        let span = ctx.token_span(&token);
                        let expr = Expr::Bool(true);
                        if in_path {
                            if in_bracket {
                                bracket_indices.push(Spanned::new(expr, span));
                            } else {
                                path.push(ExceptPathElement::Index(Spanned::new(expr, span)));
                            }
                        }
                    }
                    SyntaxKind::FalseKw => {
                        let span = ctx.token_span(&token);
                        let expr = Expr::Bool(false);
                        if in_path {
                            if in_bracket {
                                bracket_indices.push(Spanned::new(expr, span));
                            } else {
                                path.push(ExceptPathElement::Index(Spanned::new(expr, span)));
                            }
                        }
                    }
                    _ => {}
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if in_path {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        if in_bracket {
                            bracket_indices.push(Spanned::new(expr, span));
                        } else {
                            path.push(ExceptPathElement::Index(Spanned::new(expr, span)));
                        }
                    }
                }
            }
        }
    }

    // Lower the value expression as a full TLA+ expression after the '='.
    // This is essential for specs that use `@` and infix operators (e.g., `@ \\cup {x}`).
    let value = lower_expr_from_children_after_keyword(ctx, node, SyntaxKind::EqOp)?;

    Some(ExceptSpec { path, value })
}

/// Lower record expression [a |-> 1, b |-> 2]
fn lower_record_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut fields = Vec::new();

    for child in node.children() {
        if child.kind() == SyntaxKind::RecordField {
            if let Some((name, value)) = lower_record_field(ctx, &child) {
                fields.push((name, value));
            }
        }
    }

    Some(Expr::Record(fields))
}

/// Lower record field
fn lower_record_field(
    ctx: &mut LowerCtx,
    node: &SyntaxNode,
) -> Option<(Spanned<String>, Spanned<Expr>)> {
    let mut name: Option<Spanned<String>> = None;
    let mut value: Option<Spanned<Expr>> = None;
    let mut saw_mapsto = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind == SyntaxKind::MapsTo || kind == SyntaxKind::Colon {
                    saw_mapsto = true;
                    continue;
                }
                if kind == SyntaxKind::Ident && name.is_none() && !saw_mapsto {
                    let span = ctx.token_span(&token);
                    name = Some(Spanned::new(token.text().to_string(), span));
                } else if saw_mapsto && value.is_none() {
                    // Value token
                    let expr = match kind {
                        SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                        SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                        SyntaxKind::String => {
                            let s = token.text();
                            let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                            Some(Expr::String(inner.to_string()))
                        }
                        SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                        SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                        // BOOLEAN is a keyword that represents the set {TRUE, FALSE}
                        // In record set context [field: BOOLEAN], it should be treated as identifier
                        SyntaxKind::BooleanKw => Some(Expr::Ident("BOOLEAN".to_string())),
                        _ => None,
                    };
                    if let Some(e) = expr {
                        let span = ctx.token_span(&token);
                        value = Some(Spanned::new(e, span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if value.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        value = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some((name?, value?))
}

/// Lower record access r.field
fn lower_record_access_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut record: Option<Spanned<Expr>> = None;
    let mut field: Option<Spanned<String>> = None;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if token.kind() == SyntaxKind::Ident {
                    let span = ctx.token_span(&token);
                    if record.is_none() {
                        record = Some(Spanned::new(Expr::Ident(token.text().to_string()), span));
                    } else {
                        field = Some(Spanned::new(token.text().to_string(), span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if record.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        record = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some(Expr::RecordAccess(Box::new(record?), field?))
}

/// Lower record set [a : S, b : T]
fn lower_record_set_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut fields = Vec::new();

    for child in node.children() {
        if child.kind() == SyntaxKind::RecordField {
            if let Some((name, value)) = lower_record_field(ctx, &child) {
                fields.push((name, value));
            }
        }
    }

    Some(Expr::RecordSet(fields))
}

/// Lower tuple expression <<a, b, c>>
fn lower_tuple_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut elements = Vec::new();

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let expr = match token.kind() {
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    // BOOLEAN is a built-in set identifier
                    SyntaxKind::BooleanKw => Some(Expr::Ident("BOOLEAN".to_string())),
                    _ => None,
                };
                if let Some(e) = expr {
                    let span = ctx.token_span(&token);
                    elements.push(Spanned::new(e, span));
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    elements.push(Spanned::new(expr, span));
                }
            }
        }
    }

    Some(Expr::Tuple(elements))
}

/// Lower IF-THEN-ELSE expression
fn lower_if_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut cond: Option<Spanned<Expr>> = None;
    let mut then_: Option<Spanned<Expr>> = None;
    let mut else_: Option<Spanned<Expr>> = None;
    let mut state = 0; // 0=cond, 1=then, 2=else

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                match kind {
                    SyntaxKind::IfKw => {
                        state = 0;
                    }
                    SyntaxKind::ThenKw => {
                        state = 1;
                    }
                    SyntaxKind::ElseKw => {
                        state = 2;
                    }
                    _ => {
                        // Try to make expression from token
                        let expr = match kind {
                            SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                            // @ is used inside EXCEPT specs to refer to the old value
                            SyntaxKind::At => Some(Expr::Ident("@".to_string())),
                            SyntaxKind::Number => {
                                token.text().parse::<BigInt>().ok().map(Expr::Int)
                            }
                            SyntaxKind::String => {
                                let s = token.text();
                                let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                                Some(Expr::String(inner.to_string()))
                            }
                            SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                            SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                            _ => None,
                        };
                        if let Some(e) = expr {
                            let span = ctx.token_span(&token);
                            match state {
                                0 => cond = Some(Spanned::new(e, span)),
                                1 => then_ = Some(Spanned::new(e, span)),
                                2 => else_ = Some(Spanned::new(e, span)),
                                _ => {}
                            }
                        }
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                // Handle BinaryExpr specially to combine with existing cond
                if child_node.kind() == SyntaxKind::BinaryExpr && state == 0 {
                    if let Some(left) = cond.take() {
                        if let Some(combined) = lower_binary_with_left(ctx, &child_node, left) {
                            cond = Some(combined);
                        }
                    } else if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        cond = Some(Spanned::new(expr, span));
                    }
                } else if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    match state {
                        0 => cond = Some(Spanned::new(expr, span)),
                        1 => then_ = Some(Spanned::new(expr, span)),
                        2 => else_ = Some(Spanned::new(expr, span)),
                        _ => {}
                    }
                }
            }
        }
    }

    Some(Expr::If(
        Box::new(cond?),
        Box::new(then_?),
        Box::new(else_?),
    ))
}

/// Lower CASE expression
fn lower_case_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut arms = Vec::new();
    let mut default: Option<Spanned<Expr>> = None;

    let mut in_other = false;
    let mut saw_other_arrow = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::CaseArm {
                    if let Some(arm) = lower_case_arm(ctx, &child_node) {
                        arms.push(arm);
                    }
                    continue;
                }

                if in_other && saw_other_arrow && default.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        default = Some(Spanned::new(expr, span));
                    }
                }
            }
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                if kind.is_trivia() {
                    continue;
                }

                if kind == SyntaxKind::OtherKw {
                    in_other = true;
                    saw_other_arrow = false;
                    continue;
                }
                if !in_other || default.is_some() {
                    continue;
                }
                if kind == SyntaxKind::Arrow {
                    saw_other_arrow = true;
                    continue;
                }
                if !saw_other_arrow {
                    continue;
                }

                let expr = match kind {
                    SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    SyntaxKind::BooleanKw => Some(Expr::Ident("BOOLEAN".to_string())),
                    _ => None,
                };

                if let Some(e) = expr {
                    let span = ctx.token_span(&token);
                    default = Some(Spanned::new(e, span));
                }
            }
        }
    }

    Some(Expr::Case(arms, default.map(Box::new)))
}

/// Lower case arm
fn lower_case_arm(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<CaseArm> {
    let mut guard: Option<Spanned<Expr>> = None;
    let mut body: Option<Spanned<Expr>> = None;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                let kind = token.kind();
                // Skip Arrow and trivia
                if kind == SyntaxKind::Arrow || kind.is_trivia() {
                    continue;
                }
                // Handle leaf tokens as expressions
                let expr = match kind {
                    SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                    SyntaxKind::Number => token.text().parse::<BigInt>().ok().map(Expr::Int),
                    SyntaxKind::String => {
                        let s = token.text();
                        let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                        Some(Expr::String(inner.to_string()))
                    }
                    SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                    SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                    _ => None,
                };
                if let Some(e) = expr {
                    let span = ctx.token_span(&token);
                    if guard.is_none() {
                        guard = Some(Spanned::new(e, span));
                    } else if body.is_none() {
                        body = Some(Spanned::new(e, span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if let Some(expr) = lower_expr(ctx, &child_node) {
                    let span = ctx.span(&child_node);
                    if guard.is_none() {
                        guard = Some(Spanned::new(expr, span));
                    } else if body.is_none() {
                        body = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    Some(CaseArm {
        guard: guard?,
        body: body?,
    })
}

/// Lower LET-IN expression
fn lower_let_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut defs = Vec::new();

    for child in node.children() {
        if child.kind() == SyntaxKind::OperatorDef {
            if let Some(def) = lower_operator_def(ctx, &child) {
                defs.push(def);
            }
        }
    }

    // The parser doesn't always wrap the `IN <expr>` body in an expression node, so use the
    // token/node scanning helper to reliably lower the body expression.
    let body = lower_expr_from_children_after_keyword(ctx, node, SyntaxKind::InKw)?;

    Some(Expr::Let(defs, Box::new(body)))
}

/// Lower module reference expression: M!Op or M!Op(args) or IS(x,y)!Op
fn lower_module_ref_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    // ModuleRefExpr is created by wrapping an already-parsed expression node at
    // the checkpoint where `!` appears. That means the module name is usually
    // contained inside a child expression node (e.g., NameExpr) rather than as a
    // direct Ident token child of ModuleRefExpr.
    //
    // Structure (conceptually): <module_expr> ! <op_token> [ArgList]
    //
    // Examples:
    // - InChan!Init
    // - InChan!Send(msg)
    // - R!+(a, b)
    // - IS(chosen, allInput)!ISpec  (parameterized instance)
    let mut module_expr: Option<SyntaxNode> = None;
    let mut module_target: Option<ModuleTarget> = None;
    let mut op_name: Option<String> = None;
    let mut args: Vec<Spanned<Expr>> = Vec::new();

    let mut seen_bang = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if token.kind() == SyntaxKind::Bang {
                    seen_bang = true;
                    continue;
                }
                if !seen_bang {
                    if module_target.is_none() && token.kind() == SyntaxKind::Ident {
                        // Best-effort: some forms may expose the module name as a direct token.
                        module_target = Some(ModuleTarget::Named(token.text().to_string()));
                    }
                    continue;
                }

                if op_name.is_none() {
                    // Operator name after `!` can be:
                    // - Ident: M!Op
                    // - Number: TLAPS sometimes emits `TLANext!1`
                    // - Operator symbol: R!+(a, b), R!\leq(a, b)
                    match token.kind() {
                        SyntaxKind::Ident | SyntaxKind::Number => {
                            op_name = Some(token.text().to_string());
                        }
                        // For operator symbols, accept the token text. Ignore punctuation that
                        // belongs to argument lists.
                        SyntaxKind::LParen | SyntaxKind::RParen | SyntaxKind::Comma => {}
                        _ => {
                            op_name = Some(token.text().to_string());
                        }
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if !seen_bang {
                    if module_expr.is_none() {
                        module_expr = Some(child_node);
                    }
                    continue;
                }

                // Handle argument list (ApplyExpr or ArgList children)
                if child_node.kind() == SyntaxKind::ApplyExpr {
                    args = lower_apply_args(ctx, &child_node);
                } else if child_node.kind() == SyntaxKind::ArgList {
                    // ArgList is directly a child of ModuleRefExpr
                    args = lower_arg_list(ctx, &child_node);
                }
            }
        }
    }

    if module_target.is_none() {
        if let Some(module_node) = module_expr {
            let lowered = lower_expr(ctx, &module_node)?;
            let module_span = ctx.span(&module_node);
            match lowered {
                Expr::Ident(name) => module_target = Some(ModuleTarget::Named(name)),
                // Handle parameterized instance calls: IS(x, y)!Op
                // The module part is an Apply expression like Apply(Ident("IS"), [x, y])
                Expr::Apply(func_expr, param_args) => {
                    if let Expr::Ident(name) = &func_expr.node {
                        module_target = Some(ModuleTarget::Parameterized(name.clone(), param_args));
                    } else {
                        ctx.error(
                            format!(
                                "parameterized module reference requires identifier, got: {:?}",
                                func_expr.node
                            ),
                            module_span,
                        );
                        return None;
                    }
                }
                // Handle chained module references: A!B!C!D
                // The module part is itself a ModuleRef (e.g., A!B!C)
                Expr::ModuleRef(_, _, _) => {
                    module_target = Some(ModuleTarget::Chained(Box::new(Spanned::new(
                        lowered,
                        module_span,
                    ))));
                }
                other => {
                    ctx.error(
                        format!(
                            "module reference requires module name identifier, got: {:?}",
                            other
                        ),
                        module_span,
                    );
                    return None;
                }
            }
        }
    }

    let Some(module_target) = module_target else {
        ctx.error(
            "module reference requires module and operator name".to_string(),
            ctx.span(node),
        );
        return None;
    };
    let Some(op_name) = op_name else {
        ctx.error(
            "module reference requires module and operator name".to_string(),
            ctx.span(node),
        );
        return None;
    };

    Some(Expr::ModuleRef(module_target, op_name, args))
}

/// Lower action subscript expressions:
/// - `[A]_v`  (stuttering step allowed)
/// - `<<A>>_v` (stuttering step excluded)
///
/// TLC treats these as syntactic sugar:
/// - `[A]_v`   == A \/ UNCHANGED v
/// - `<<A>>_v` == A /\ ~UNCHANGED v
fn lower_subscript_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    // SubscriptExpr wraps a previously-parsed base expression at a checkpoint where `_` appears.
    // The base expression is either:
    // - a bracketed action `[A]` (parsed as FuncSetExpr with no `->`), or
    // - an angle action `<<A>>` (parsed as TupleExpr with a single element).

    let mut base: Option<SyntaxNode> = None;
    let mut subscript: Option<Spanned<Expr>> = None;
    let mut saw_underscore = false;

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if token.kind() == SyntaxKind::Underscore {
                    saw_underscore = true;
                    continue;
                }
                if saw_underscore && subscript.is_none() {
                    // Handle simple identifier subscripts, e.g. `[A]_vars`.
                    if token.kind() == SyntaxKind::Ident {
                        let span = ctx.token_span(&token);
                        subscript = Some(Spanned::new(Expr::Ident(token.text().to_string()), span));
                    } else if token.kind() == SyntaxKind::At {
                        let span = ctx.token_span(&token);
                        subscript = Some(Spanned::new(Expr::Ident("@".to_string()), span));
                    }
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if !saw_underscore {
                    if base.is_none() {
                        base = Some(child_node);
                    }
                    continue;
                }
                if subscript.is_none() {
                    if let Some(expr) = lower_expr(ctx, &child_node) {
                        let span = ctx.span(&child_node);
                        subscript = Some(Spanned::new(expr, span));
                    }
                }
            }
        }
    }

    let base = base?;
    let subscript = subscript?;

    let node_span = ctx.span(node);

    let unchanged = Spanned::new(Expr::Unchanged(Box::new(subscript)), node_span);

    match base.kind() {
        // `[A]_v` - base `[A]` is parsed as a FuncSetExpr with no arrow.
        SyntaxKind::FuncSetExpr => {
            // Extract the action expression `A` from inside `[A]`.
            let mut saw_arrow = false;
            for child in base.children_with_tokens() {
                if let rowan::NodeOrToken::Token(t) = &child {
                    if t.kind() == SyntaxKind::Arrow {
                        saw_arrow = true;
                        break;
                    }
                }
            }
            if saw_arrow {
                ctx.error(
                    "action subscript base '[A]' cannot be a function set '[S -> T]'".to_string(),
                    ctx.span(&base),
                );
                return None;
            }

            let mut action: Option<Spanned<Expr>> = None;
            for child in base.children_with_tokens() {
                match child {
                    rowan::NodeOrToken::Node(child_node) => {
                        if let Some(expr) = lower_expr(ctx, &child_node) {
                            let span = ctx.span(&child_node);
                            action = Some(Spanned::new(expr, span));
                            break;
                        }
                    }
                    rowan::NodeOrToken::Token(token) => {
                        // Fallback for action atoms like `[Next]` where Next is a bare token.
                        let expr = match token.kind() {
                            SyntaxKind::Ident => Some(Expr::Ident(token.text().to_string())),
                            SyntaxKind::At => Some(Expr::Ident("@".to_string())),
                            SyntaxKind::Number => {
                                token.text().parse::<BigInt>().ok().map(Expr::Int)
                            }
                            SyntaxKind::String => {
                                let s = token.text();
                                let inner = if s.len() >= 2 { &s[1..s.len() - 1] } else { s };
                                Some(Expr::String(inner.to_string()))
                            }
                            SyntaxKind::TrueKw => Some(Expr::Bool(true)),
                            SyntaxKind::FalseKw => Some(Expr::Bool(false)),
                            _ => None,
                        };
                        if let Some(e) = expr {
                            let span = ctx.token_span(&token);
                            action = Some(Spanned::new(e, span));
                            break;
                        }
                    }
                }
            }

            Some(Expr::Or(Box::new(action?), Box::new(unchanged)))
        }

        // `<<A>>_v` - base is parsed as a TupleExpr with a single element.
        SyntaxKind::TupleExpr => {
            let tuple = lower_tuple_expr(ctx, &base)?;
            let Expr::Tuple(mut elems) = tuple else {
                ctx.error(
                    "angle action '<<A>>_v' did not lower to a tuple".to_string(),
                    ctx.span(&base),
                );
                return None;
            };
            if elems.len() != 1 {
                ctx.error(
                    "angle action '<<A>>_v' must contain exactly one expression".to_string(),
                    ctx.span(&base),
                );
                return None;
            }
            let action = elems.pop()?;

            let not_unchanged = Spanned::new(Expr::Not(Box::new(unchanged)), node_span);
            Some(Expr::And(Box::new(action), Box::new(not_unchanged)))
        }

        other => {
            ctx.error(
                format!(
                    "unexpected base for action subscript expression: {:?}",
                    other
                ),
                ctx.span(&base),
            );
            None
        }
    }
}

/// Lower an INSTANCE declaration as an expression (for named instances)
/// InChan == INSTANCE Channel WITH Data <- Message, chan <- in
/// becomes Expr::InstanceExpr("Channel", [Data <- Message, chan <- in])
fn lower_instance_expr(ctx: &mut LowerCtx, node: &SyntaxNode) -> Option<Expr> {
    let mut module_name: Option<String> = None;
    let mut substitutions = Vec::new();

    for child in node.children_with_tokens() {
        match child {
            rowan::NodeOrToken::Token(token) => {
                if token.kind() == SyntaxKind::Ident && module_name.is_none() {
                    module_name = Some(token.text().to_string());
                }
            }
            rowan::NodeOrToken::Node(child_node) => {
                if child_node.kind() == SyntaxKind::Substitution {
                    if let Some(sub) = lower_substitution(ctx, &child_node) {
                        substitutions.push(sub);
                    }
                }
            }
        }
    }

    Some(Expr::InstanceExpr(module_name?, substitutions))
}

// =============================================================================
// Public API
// =============================================================================

/// Result of lowering
pub struct LowerResult {
    pub module: Option<Module>,
    pub errors: Vec<LowerError>,
}

/// Result of lowering all modules in a syntax tree (including inline submodules)
pub struct LowerAllResult {
    pub modules: Vec<Module>,
    pub errors: Vec<LowerError>,
}

/// Lower a parsed syntax tree to an AST
pub fn lower(file_id: FileId, root: &SyntaxNode) -> LowerResult {
    let mut ctx = LowerCtx::new(file_id);
    let module = lower_module(&mut ctx, root);
    let errors = ctx.take_errors();
    LowerResult { module, errors }
}

/// Lower all `MODULE` nodes in the syntax tree to AST modules.
///
/// TLA+ allows defining inline submodules (a `MODULE` nested within another `MODULE`).
/// For semantic analysis, tools often need to resolve `INSTANCE Inner` against such
/// inline module definitions.
pub fn lower_all_modules(file_id: FileId, root: &SyntaxNode) -> LowerAllResult {
    let mut ctx = LowerCtx::new(file_id);
    let mut modules = Vec::new();

    for node in root
        .descendants()
        .filter(|n| n.kind() == SyntaxKind::Module)
    {
        if let Some(module) = lower_module_node(&mut ctx, &node) {
            modules.push(module);
        }
    }

    let errors = ctx.take_errors();
    LowerAllResult { modules, errors }
}

/// Lower a single expression syntax node to an AST expression
///
/// This is useful for lowering inline expressions extracted from the syntax tree,
/// such as fairness constraint actions that are not simple operator names.
///
/// Returns `None` if the node cannot be lowered to an expression.
pub fn lower_single_expr(file_id: FileId, node: &SyntaxNode) -> Option<Expr> {
    let mut ctx = LowerCtx::new(file_id);
    lower_expr(&mut ctx, node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::parse_to_syntax_tree;

    #[test]
    fn test_lower_simple_module() {
        let source = r#"---- MODULE Test ----
VARIABLE x
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.name.node, "Test");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Variable(vars) => {
                assert_eq!(vars.len(), 1);
                assert_eq!(vars[0].node, "x");
            }
            _ => panic!("Expected Variable unit"),
        }
    }

    #[test]
    fn test_lower_operator_def() {
        let source = r#"---- MODULE Test ----
Add(a, b) == a + b
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => {
                assert_eq!(def.name.node, "Add");
                // Body should be a + b which is an Add expression
                match &def.body.node {
                    Expr::Add(_, _) => {}
                    other => panic!("Expected Add expression, got {:?}", other),
                }
            }
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_bullet_list_then_implies_layout() {
        let source = r#"---- MODULE Test ----
A == TRUE
B == TRUE
C == FALSE
Expr ==
  /\ /\ A
     /\ B
     => C
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        let expr_def = module
            .units
            .iter()
            .find_map(|unit| match &unit.node {
                Unit::Operator(def) if def.name.node == "Expr" => Some(def),
                _ => None,
            })
            .expect("Expected Expr operator");

        match &expr_def.body.node {
            Expr::Implies(lhs, rhs) => {
                assert!(matches!(&rhs.node, Expr::Ident(name) if name == "C"));
                match &lhs.node {
                    Expr::And(a, b) => {
                        assert!(matches!(&a.node, Expr::Ident(name) if name == "A"));
                        assert!(matches!(&b.node, Expr::Ident(name) if name == "B"));
                    }
                    other => panic!("Expected And(A, B) antecedent, got {:?}", other),
                }
            }
            other => panic!("Expected Implies expression, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_multi_char_infix_operator_use() {
        let source = r#"---- MODULE Test ----
Op == 2 ++ 3
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        let body = match &module.units[0].node {
            Unit::Operator(def) => &def.body.node,
            other => panic!("Expected Operator unit, got {:?}", other),
        };

        match body {
            Expr::Apply(op, args) => {
                assert!(matches!(&op.node, Expr::Ident(name) if name == "++"));
                assert_eq!(args.len(), 2);
            }
            other => panic!("Expected Apply for ++ infix, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_ordering_relation_operator_use() {
        let source = r#"---- MODULE Test ----
Op == 1 \prec 2
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        let body = match &module.units[0].node {
            Unit::Operator(def) => &def.body.node,
            other => panic!("Expected Operator unit, got {:?}", other),
        };

        match body {
            Expr::Apply(op, args) => {
                assert!(matches!(&op.node, Expr::Ident(name) if name == "\\prec"));
                assert_eq!(args.len(), 2);
            }
            other => panic!("Expected Apply for \\prec, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_infix_operator_def() {
        let source = r#"---- MODULE Test ----
a \prec b == a < b
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => {
                assert_eq!(def.name.node, "\\prec");
                assert_eq!(def.params.len(), 2);
                assert_eq!(def.params[0].name.node, "a");
                assert_eq!(def.params[1].name.node, "b");
                assert!(matches!(&def.body.node, Expr::Lt(_, _)));
            }
            other => panic!("Expected Operator unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_underscore_prefixed_let_def_and_ref() {
        let source = r#"---- MODULE Test ----
Op == LET _msgs == 1 IN _msgs
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        let body = match &module.units[0].node {
            Unit::Operator(def) => &def.body.node,
            other => panic!("Expected Operator unit, got {:?}", other),
        };

        match body {
            Expr::Let(defs, inner) => {
                assert_eq!(defs.len(), 1);
                assert_eq!(defs[0].name.node, "_msgs");
                assert!(matches!(&inner.node, Expr::Ident(name) if name == "_msgs"));
            }
            other => panic!("Expected LET expression, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_underscore_ident_as_except_base() {
        let source = r#"---- MODULE Test ----
Op == LET _msgs == [msgs EXCEPT ![self] = @ \ {id}]
      IN [_msgs EXCEPT ![succ(self)] = @ \cup {id}]
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        let body = match &module.units[0].node {
            Unit::Operator(def) => &def.body.node,
            other => panic!("Expected Operator unit, got {:?}", other),
        };

        match body {
            Expr::Let(_defs, inner) => match &inner.node {
                Expr::Except(base, _specs) => {
                    assert!(matches!(&base.node, Expr::Ident(name) if name == "_msgs"));
                }
                other => panic!("Expected EXCEPT expression, got {:?}", other),
            },
            other => panic!("Expected LET expression, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_case_with_other_literal() {
        let source = r#"---- MODULE Test ----
Op(x) == CASE x = 0 -> 0 [] OTHER -> 1
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        let body = match &module.units[0].node {
            Unit::Operator(def) => &def.body.node,
            other => panic!("Expected Operator unit, got {:?}", other),
        };

        match body {
            Expr::Case(arms, Some(default)) => {
                assert_eq!(arms.len(), 1);
                match &default.node {
                    Expr::Int(n) => assert_eq!(n, &BigInt::from(1)),
                    other => panic!("Expected Int default, got {:?}", other),
                }
            }
            other => panic!("Expected CASE with OTHER, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_choose_with_literal_body() {
        let source = r#"---- MODULE Test ----
Op == CHOOSE x \in {1, 2} : TRUE
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        let body = match &module.units[0].node {
            Unit::Operator(def) => &def.body.node,
            other => panic!("Expected Operator unit, got {:?}", other),
        };

        match body {
            Expr::Choose(bound, choose_body) => {
                assert_eq!(bound.name.node, "x");
                assert!(matches!(choose_body.node, Expr::Bool(true)));
            }
            other => panic!("Expected CHOOSE, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_constant_decl_with_arity() {
        let source = r#"---- MODULE Test ----
CONSTANTS F(_, _), G
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Constant(consts) => {
                assert_eq!(consts.len(), 2);
                assert_eq!(consts[0].name.node, "F");
                assert_eq!(consts[0].arity, Some(2));
                assert_eq!(consts[1].name.node, "G");
                assert_eq!(consts[1].arity, None);
            }
            other => panic!("Expected Constant unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_local_operator_def() {
        let source = r#"---- MODULE Test ----
LOCAL Foo == TRUE
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Operator(def) => {
                assert!(def.local);
                assert_eq!(def.name.node, "Foo");
            }
            other => panic!("Expected Operator unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_local_instance_decl() {
        let source = r#"---- MODULE Test ----
LOCAL INSTANCE Foo WITH x <- 1
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Instance(inst) => {
                assert!(inst.local);
                assert_eq!(inst.module.node, "Foo");
                assert_eq!(inst.substitutions.len(), 1);
                assert_eq!(inst.substitutions[0].from.node, "x");
            }
            other => panic!("Expected Instance unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_quantifier() {
        let source = r#"---- MODULE Test ----
AllPositive(S) == \A x \in S : x > 0
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::Forall(bounds, _) => {
                    assert_eq!(bounds.len(), 1);
                    assert_eq!(bounds[0].name.node, "x");
                }
                other => panic!("Expected Forall expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_assume_true() {
        let source = r#"---- MODULE Test ----
ASSUME TRUE
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Assume(assume) => match &assume.expr.node {
                Expr::Bool(true) => {}
                other => panic!("Expected TRUE, got {:?}", other),
            },
            other => panic!("Expected Assume unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_theorem_true_body() {
        let source = r#"---- MODULE Test ----
THEOREM T1 == TRUE
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Theorem(theorem) => match &theorem.body.node {
                Expr::Bool(true) => {}
                other => panic!("Expected TRUE, got {:?}", other),
            },
            other => panic!("Expected Theorem unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_set_expressions() {
        let source = r#"---- MODULE Test ----
S == {1, 2, 3}
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::SetEnum(elements) => {
                    assert_eq!(elements.len(), 3);
                }
                other => panic!("Expected SetEnum expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_function_def() {
        let source = r#"---- MODULE Test ----
f == [x \in Nat |-> x * 2]
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::FuncDef(bounds, _) => {
                    assert_eq!(bounds.len(), 1);
                    assert_eq!(bounds[0].name.node, "x");
                }
                other => panic!("Expected FuncDef expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_function_def_multi_var_domain_propagation() {
        let source = r#"---- MODULE Test ----
f == [x, y \in Nat |-> x + y]
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::FuncDef(bounds, _) => {
                    assert_eq!(bounds.len(), 2);
                    assert_eq!(bounds[0].name.node, "x");
                    assert_eq!(bounds[1].name.node, "y");
                    let x_domain = bounds[0].domain.as_ref().expect("x domain");
                    let y_domain = bounds[1].domain.as_ref().expect("y domain");
                    assert!(matches!(&x_domain.node, Expr::Ident(s) if s == "Nat"));
                    assert!(matches!(&y_domain.node, Expr::Ident(s) if s == "Nat"));
                }
                other => panic!("Expected FuncDef expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_record() {
        let source = r#"---- MODULE Test ----
r == [a |-> 1, b |-> 2]
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::Record(fields) => {
                    assert_eq!(fields.len(), 2);
                    assert_eq!(fields[0].0.node, "a");
                    assert_eq!(fields[1].0.node, "b");
                }
                other => panic!("Expected Record expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_record_set() {
        // Test lowering of record set expression [field: BOOLEAN]
        // BOOLEAN is a keyword and needs special handling
        let source = r#"---- MODULE Test ----
r == [smoking: BOOLEAN]
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::RecordSet(fields) => {
                    assert_eq!(fields.len(), 1, "Expected 1 field");
                    assert_eq!(fields[0].0.node, "smoking");
                    // The value should be Ident("BOOLEAN")
                    assert!(matches!(fields[0].1.node, Expr::Ident(ref s) if s == "BOOLEAN"));
                }
                other => panic!("Expected RecordSet expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_simple_func_apply() {
        // Test: G == F[1] (no parameters, literal argument)
        // The argument 1 is a Number token that needs special handling
        let source = r#"---- MODULE Test ----
S == {1, 2}
F == [r \in S |-> r * 10]
G == F[1]
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        // Find all operators
        let ops: Vec<_> = module
            .units
            .iter()
            .filter_map(|u| {
                if let Unit::Operator(def) = &u.node {
                    Some((def.name.node.clone(), def.body.node.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Should have S, F, and G
        assert_eq!(ops.len(), 3, "Expected 3 operators");

        let g = ops.iter().find(|(n, _)| n == "G").expect("G should exist");
        match &g.1 {
            Expr::FuncApply(func, arg) => {
                // func should be Ident("F")
                assert!(matches!(&func.node, Expr::Ident(s) if s == "F"));
                // arg should be Int(1)
                assert!(matches!(&arg.node, Expr::Int(n) if *n == BigInt::from(1)));
            }
            other => panic!("G should be FuncApply, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_multi_arg_func_apply_as_tuple() {
        let source = r#"---- MODULE Test ----
G == F[1, 2]
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::FuncApply(func, arg) => {
                    assert!(matches!(&func.node, Expr::Ident(s) if s == "F"));
                    match &arg.node {
                        Expr::Tuple(elements) => {
                            assert_eq!(elements.len(), 2);
                            assert!(
                                matches!(&elements[0].node, Expr::Int(n) if *n == BigInt::from(1))
                            );
                            assert!(
                                matches!(&elements[1].node, Expr::Int(n) if *n == BigInt::from(2))
                            );
                        }
                        other => {
                            panic!("Expected tuple arg for multi-index apply, got {:?}", other)
                        }
                    }
                }
                other => panic!("Expected FuncApply expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_if_then_else() {
        let source = r#"---- MODULE Test ----
Max(a, b) == IF a > b THEN a ELSE b
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::If(_, _, _) => {}
                other => panic!("Expected If expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_tuple() {
        let source = r#"---- MODULE Test ----
t == <<1, 2, 3>>
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        match &module.units[0].node {
            Unit::Operator(def) => match &def.body.node {
                Expr::Tuple(elements) => {
                    assert_eq!(elements.len(), 3);
                }
                other => panic!("Expected Tuple expression, got {:?}", other),
            },
            _ => panic!("Expected Operator unit"),
        }
    }

    #[test]
    fn test_lower_recursive_simple() {
        let source = r#"---- MODULE Test ----
RECURSIVE Factorial(_)
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Recursive(decls) => {
                assert_eq!(decls.len(), 1);
                assert_eq!(decls[0].name.node, "Factorial");
                assert_eq!(decls[0].arity, 1);
            }
            other => panic!("Expected Recursive unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_recursive_multiple() {
        let source = r#"---- MODULE Test ----
RECURSIVE F(_), G(_, _)
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Recursive(decls) => {
                assert_eq!(decls.len(), 2);
                assert_eq!(decls[0].name.node, "F");
                assert_eq!(decls[0].arity, 1);
                assert_eq!(decls[1].name.node, "G");
                assert_eq!(decls[1].arity, 2);
            }
            other => panic!("Expected Recursive unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_recursive_no_params() {
        let source = r#"---- MODULE Test ----
RECURSIVE Foo
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 1);

        match &module.units[0].node {
            Unit::Recursive(decls) => {
                assert_eq!(decls.len(), 1);
                assert_eq!(decls[0].name.node, "Foo");
                assert_eq!(decls[0].arity, 0);
            }
            other => panic!("Expected Recursive unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_recursive_with_definition() {
        // Test RECURSIVE followed by actual definition
        let source = r#"---- MODULE Test ----
RECURSIVE Factorial(_)
Factorial(n) == IF n = 0 THEN 1 ELSE n * Factorial(n - 1)
===="#;
        let tree = parse_to_syntax_tree(source);
        let result = lower(FileId(0), &tree);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");
        assert_eq!(module.units.len(), 2);

        // First unit: RECURSIVE declaration
        match &module.units[0].node {
            Unit::Recursive(decls) => {
                assert_eq!(decls.len(), 1);
                assert_eq!(decls[0].name.node, "Factorial");
                assert_eq!(decls[0].arity, 1);
            }
            other => panic!("Expected Recursive unit, got {:?}", other),
        }

        // Second unit: actual definition
        match &module.units[1].node {
            Unit::Operator(op) => {
                assert_eq!(op.name.node, "Factorial");
                assert_eq!(op.params.len(), 1);
            }
            other => panic!("Expected Operator unit, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_func_constructor_in_equality() {
        // Test: Init == pc = [p \in S |-> "b0"]
        // This pattern is common in specs like Barrier and CigaretteSmokers
        let source = r#"---- MODULE Test ----
VARIABLE pc
Init == pc = [p \in {1,2} |-> "b0"]
===="#;
        let tree = parse_to_syntax_tree(source);
        eprintln!("CST:\n{:#?}", tree);

        let result = lower(FileId(0), &tree);
        eprintln!("Lower errors: {:?}", result.errors);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        // Find Init operator
        let init_op = module
            .units
            .iter()
            .find_map(|u| {
                if let Unit::Operator(op) = &u.node {
                    if op.name.node == "Init" {
                        return Some(op);
                    }
                }
                None
            })
            .expect("Expected Init operator");

        eprintln!("Init body: {:?}", init_op.body.node);

        // The body should be Eq(Ident("pc"), FuncDef(...)), not just Ident("pc")
        match &init_op.body.node {
            Expr::Eq(lhs, rhs) => {
                assert!(
                    matches!(&lhs.node, Expr::Ident(name) if name == "pc"),
                    "Expected lhs to be Ident(pc), got {:?}",
                    lhs.node
                );
                assert!(
                    matches!(&rhs.node, Expr::FuncDef(_, _)),
                    "Expected rhs to be FuncDef, got {:?}",
                    rhs.node
                );
            }
            other => panic!("Expected Init body to be Eq(...), got {:?}", other),
        }
    }

    #[test]
    fn test_lower_func_set_with_ident_domain() {
        // Test: TypeOK == pc \in [ProcSet -> {"b0", "b1"}]
        // The domain is an identifier reference, not a literal set
        let source = r#"---- MODULE Test ----
ProcSet == {1, 2, 3}
TypeOK == pc \in [ProcSet -> {"b0"}]
===="#;
        let tree = parse_to_syntax_tree(source);
        eprintln!("CST:\n{:#?}", tree);

        let result = lower(FileId(0), &tree);
        eprintln!("Lower errors: {:?}", result.errors);

        assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
        let module = result.module.expect("Expected module");

        // Find TypeOK operator
        let type_ok = module
            .units
            .iter()
            .find_map(|u| {
                if let Unit::Operator(op) = &u.node {
                    if op.name.node == "TypeOK" {
                        return Some(op);
                    }
                }
                None
            })
            .expect("Expected TypeOK operator");

        eprintln!("TypeOK body: {:?}", type_ok.body.node);

        // The body should be In(Ident("pc"), FuncSet(Ident("ProcSet"), {...}))
        match &type_ok.body.node {
            Expr::In(lhs, rhs) => {
                assert!(
                    matches!(&lhs.node, Expr::Ident(name) if name == "pc"),
                    "Expected lhs to be Ident(pc), got {:?}",
                    lhs.node
                );
                match &rhs.node {
                    Expr::FuncSet(domain, _range) => {
                        assert!(
                            matches!(&domain.node, Expr::Ident(name) if name == "ProcSet"),
                            "Expected domain to be Ident(ProcSet), got {:?}",
                            domain.node
                        );
                    }
                    other => panic!("Expected rhs to be FuncSet, got {:?}", other),
                }
            }
            other => panic!("Expected TypeOK body to be In(...), got {:?}", other),
        }
    }

    #[test]
    fn test_lower_exists_with_apply() {
        // Test lowering \E p \in S : f(p)
        let source = r#"---- MODULE Test ----
CONSTANT N
ProcSet == 1..N
b0(self) == TRUE
Next == \E p \in ProcSet : b0(p)
===="#;
        let tree = parse_to_syntax_tree(source);

        // Debug: print the CST structure
        fn print_tree(node: &crate::syntax::SyntaxNode, indent: usize) {
            let kind = node.kind();
            let text = node.text();
            let text_len: usize = text.len().into();
            if text_len < 60 {
                eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
            } else {
                eprintln!(
                    "{:indent$}{:?}: <{} chars>",
                    "",
                    kind,
                    text_len,
                    indent = indent
                );
            }
            for child in node.children() {
                print_tree(&child, indent + 2);
            }
        }
        eprintln!("=== CST ===");
        print_tree(&tree, 0);

        let result = lower(FileId(0), &tree);

        eprintln!("\n=== Lowering ===");
        eprintln!("Errors: {:?}", result.errors);

        let module = result.module.expect("Expected module");

        let mut found = std::collections::HashMap::new();
        for unit in &module.units {
            if let Unit::Operator(def) = &unit.node {
                eprintln!(
                    "Op '{}': {:?}",
                    def.name.node,
                    std::mem::discriminant(&def.body.node)
                );
                found.insert(def.name.node.clone(), def.body.node.clone());
            }
        }

        // Verify Next was lowered
        assert!(
            found.contains_key("Next"),
            "Next operator should exist, found: {:?}",
            found.keys().collect::<Vec<_>>()
        );

        // Verify Next body is Exists
        match found.get("Next").unwrap() {
            Expr::Exists(bounds, _body) => {
                assert_eq!(bounds.len(), 1);
                assert_eq!(bounds[0].name.node, "p");
            }
            other => panic!("Expected Next to be Exists, got {:?}", other),
        }
    }
}

#[test]
fn test_lower_func_operator_with_tuple_pattern() {
    let src = r#"
---- MODULE Test ----
EXTENDS Integers

f[<<a, b>> \in {1, 2} \X {1, 2}] == a + b
====
"#;
    let tree = crate::parse_to_syntax_tree(src);
    eprintln!("=== CST ===");
    fn print_tree(node: &SyntaxNode, indent: usize) {
        let text = node.text().to_string();
        let text_len = text.len();
        let kind = node.kind();
        if text_len < 80 {
            eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
        } else {
            eprintln!(
                "{:indent$}{:?}: <{} chars>",
                "",
                kind,
                text_len,
                indent = indent
            );
        }
        for child in node.children() {
            print_tree(&child, indent + 2);
        }
    }
    print_tree(&tree, 0);

    let lower_result = crate::lower(FileId(0), &tree);
    eprintln!("\n=== Lower Results ===");
    eprintln!("Errors: {:?}", lower_result.errors);

    let module = lower_result.module.unwrap();
    for unit in &module.units {
        if let Unit::Operator(def) = &unit.node {
            eprintln!("\nOperator '{}' body: {:?}", def.name.node, def.body.node);
        }
    }
}

#[test]
fn test_lower_operator_def_with_params() {
    let src = r#"
---- MODULE Test ----
add_one(x) == x + 1
====
"#;
    let tree = crate::parse_to_syntax_tree(src);
    eprintln!("=== CST ===");
    fn print_tree(node: &SyntaxNode, indent: usize) {
        let text = node.text().to_string();
        let text_len = text.len();
        let kind = node.kind();
        if text_len < 80 {
            eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
        } else {
            eprintln!(
                "{:indent$}{:?}: <{} chars>",
                "",
                kind,
                text_len,
                indent = indent
            );
        }
        for child in node.children() {
            print_tree(&child, indent + 2);
        }
    }
    print_tree(&tree, 0);

    let lower_result = crate::lower(FileId(0), &tree);
    eprintln!("\n=== Lower Results ===");
    eprintln!("Errors: {:?}", lower_result.errors);

    let module = lower_result.module.unwrap();
    for unit in &module.units {
        if let Unit::Operator(def) = &unit.node {
            eprintln!("\nOperator '{}' params: {:?}", def.name.node, def.params);
            eprintln!("Operator '{}' body: {:?}", def.name.node, def.body.node);
            assert_eq!(def.params.len(), 1, "Expected 1 parameter");
            assert_eq!(def.params[0].name.node, "x");
        }
    }
}

#[test]
fn test_lower_func_apply_body() {
    // Test that p[1] in operator body lowers correctly
    let src = r#"
---- MODULE Test ----
f(p) == p[1]
====
"#;
    let tree = crate::parse_to_syntax_tree(src);
    eprintln!("=== CST for p[1] ===");
    fn print_tree(node: &SyntaxNode, indent: usize) {
        let text = node.text().to_string();
        let text_len = text.len();
        let kind = node.kind();
        if text_len < 80 {
            eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
        } else {
            eprintln!(
                "{:indent$}{:?}: <{} chars>",
                "",
                kind,
                text_len,
                indent = indent
            );
        }
        for child in node.children() {
            print_tree(&child, indent + 2);
        }
    }
    print_tree(&tree, 0);

    let lower_result = crate::lower(FileId(0), &tree);
    eprintln!("\n=== Lower Results ===");
    eprintln!("Errors: {:?}", lower_result.errors);

    let module = lower_result.module.unwrap();
    for unit in &module.units {
        if let Unit::Operator(def) = &unit.node {
            eprintln!("\nOperator '{}' params: {:?}", def.name.node, def.params);
            eprintln!("Operator '{}' body: {:?}", def.name.node, def.body.node);
        }
    }
}

#[test]
fn test_lower_except_nested_path() {
    let src = r#"
---- MODULE Test ----
VARIABLE smokers, r
Test == smokers' = [smokers EXCEPT ![r].smoking = FALSE]
====
"#;
    let tree = crate::parse_to_syntax_tree(src);
    eprintln!("=== CST for EXCEPT ===");
    fn print_tree_full(node: &SyntaxNode, indent: usize) {
        let text = node.text().to_string();
        let text_len = text.len();
        let kind = node.kind();
        if text_len < 80 {
            eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
        } else {
            eprintln!(
                "{:indent$}{:?}: <{} chars>",
                "",
                kind,
                text_len,
                indent = indent
            );
        }
        // Also print tokens
        for child in node.children_with_tokens() {
            match child {
                rowan::NodeOrToken::Token(t) => {
                    eprintln!(
                        "{:indent$}  TOKEN {:?}: {:?}",
                        "",
                        t.kind(),
                        t.text(),
                        indent = indent
                    );
                }
                rowan::NodeOrToken::Node(n) => {
                    print_tree_full(&n, indent + 2);
                }
            }
        }
    }
    print_tree_full(&tree, 0);

    let lower_result = crate::lower(FileId(0), &tree);
    eprintln!("\n=== Lower Results ===");
    eprintln!("Errors: {:?}", lower_result.errors);

    let module = lower_result.module.unwrap();
    for unit in &module.units {
        if let Unit::Operator(def) = &unit.node {
            eprintln!("\nOperator '{}' body: {:?}", def.name.node, def.body.node);
            // Check if the body is an Eq with an Except
            if let Expr::Eq(_, rhs) = &def.body.node {
                if let Expr::Except(base, specs) = &rhs.node {
                    eprintln!("EXCEPT base: {:?}", base);
                    eprintln!("EXCEPT specs count: {}", specs.len());
                    for (i, spec) in specs.iter().enumerate() {
                        eprintln!(
                            "EXCEPT spec[{}]: path={:?}, value={:?}",
                            i, spec.path, spec.value.node
                        );
                    }
                } else {
                    eprintln!("RHS is not Except: {:?}", rhs.node);
                }
            }
        }
    }
}

#[test]
fn test_lower_except_at_index() {
    // Tests that @[idx] in EXCEPT expressions is correctly lowered to FuncApply(Ident("@"), idx)
    // This is essential for patterns like: [f EXCEPT ![k] = @[1]] where @ refers to the old value
    let src = r#"
---- MODULE Test ----
Test == [[i \in {1} |-> 0] EXCEPT ![1] = @[1]]
====
"#;
    let tree = crate::parse_to_syntax_tree(src);
    let lower_result = crate::lower(FileId(0), &tree);
    assert!(
        lower_result.errors.is_empty(),
        "Expected no lowering errors"
    );

    let module = lower_result.module.expect("Module should be present");
    let op_def = module
        .units
        .iter()
        .find_map(|u| {
            if let Unit::Operator(def) = &u.node {
                Some(def)
            } else {
                None
            }
        })
        .expect("Should have operator def");

    // The body should be Except with a spec that has FuncApply(@, 1) as the value
    if let Expr::Except(_, specs) = &op_def.body.node {
        assert_eq!(specs.len(), 1, "Should have 1 except spec");
        // Check that the value is FuncApply(Ident("@"), Int(1))
        if let Expr::FuncApply(func, arg) = &specs[0].value.node {
            assert!(
                matches!(&func.node, Expr::Ident(s) if s == "@"),
                "Function should be @ identifier"
            );
            assert!(
                matches!(&arg.node, Expr::Int(n) if n == &num_bigint::BigInt::from(1)),
                "Argument should be Int(1)"
            );
        } else {
            panic!("Expected FuncApply for @[1], got {:?}", specs[0].value.node);
        }
    } else {
        panic!("Expected Except expression");
    }
}

#[test]
fn test_lower_except_if_then_else() {
    let src = r#"
---- MODULE Test ----
VARIABLES y
Next == y' = [y EXCEPT ![0] = IF 1 > 0 THEN "b" ELSE @]
====
"#;
    use crate::SyntaxNode;
    let tree = crate::parse_to_syntax_tree(src);
    eprintln!("=== CST for IF in EXCEPT ===");
    fn print_tree_full(node: &SyntaxNode, indent: usize) {
        let text = node.text().to_string();
        let text_len = text.len();
        let kind = node.kind();
        if text_len < 80 {
            eprintln!("{:indent$}{:?}: {:?}", "", kind, text, indent = indent);
        } else {
            eprintln!(
                "{:indent$}{:?}: <{} chars>",
                "",
                kind,
                text_len,
                indent = indent
            );
        }
        for child in node.children_with_tokens() {
            match child {
                rowan::NodeOrToken::Token(t) => {
                    eprintln!(
                        "{:indent$}  TOKEN {:?}: {:?}",
                        "",
                        t.kind(),
                        t.text(),
                        indent = indent
                    );
                }
                rowan::NodeOrToken::Node(n) => {
                    print_tree_full(&n, indent + 2);
                }
            }
        }
    }
    print_tree_full(&tree, 0);

    let lower_result = crate::lower(FileId(0), &tree);
    eprintln!("\n=== Lower Results ===");
    eprintln!("Errors: {:?}", lower_result.errors);

    let module = lower_result.module.unwrap();
    for unit in &module.units {
        if let Unit::Operator(def) = &unit.node {
            eprintln!("\nOperator '{}' body: {:?}", def.name.node, def.body.node);
            // Check if the body is an Eq with an Except
            if let Expr::Eq(_, rhs) = &def.body.node {
                if let Expr::Except(base, specs) = &rhs.node {
                    eprintln!("EXCEPT base: {:?}", base);
                    eprintln!("EXCEPT specs count: {}", specs.len());
                    for (i, spec) in specs.iter().enumerate() {
                        eprintln!(
                            "EXCEPT spec[{}]: path={:?}, value={:?}",
                            i, spec.path, spec.value.node
                        );
                    }
                    // Verify we have exactly one spec
                    assert_eq!(
                        specs.len(),
                        1,
                        "Expected 1 EXCEPT spec, got {}",
                        specs.len()
                    );
                } else {
                    panic!("RHS is not Except: {:?}", rhs.node);
                }
            }
        }
    }
}

/// Regression test for span offset correctness after whitespace fix
#[test]
fn test_span_offsets_correct() {
    let source = "---- MODULE Counter ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x < 5 /\\ x' = x + 1\nInRange == x >= 0 /\\ x <= 5\n====";

    // Expected positions in source
    let init_body_start = source.find("x = 0").expect("x = 0 not found");
    let next_body_start = source.find("x < 5").expect("x < 5 not found");

    // Parse and lower
    let tree = crate::parse_to_syntax_tree(source);
    let result = crate::lower(FileId(0), &tree);
    let module = result.module.expect("Expected module");

    // Verify spans point to correct source text
    for unit in &module.units {
        if let Unit::Operator(def) = &unit.node {
            let body_start = def.body.span.start as usize;
            let body_end = def.body.span.end as usize;

            match def.name.node.as_str() {
                "Init" => {
                    assert_eq!(
                        body_start, init_body_start,
                        "Init body span start should match 'x = 0' position"
                    );
                    let text = &source[body_start..body_end];
                    assert!(
                        text.starts_with("x = 0"),
                        "Init body should start with 'x = 0', got: {:?}",
                        text
                    );
                }
                "Next" => {
                    assert_eq!(
                        body_start, next_body_start,
                        "Next body span start should match 'x < 5' position"
                    );
                    let text = &source[body_start..body_end];
                    assert!(
                        text.starts_with("x < 5"),
                        "Next body should start with 'x < 5', got: {:?}",
                        text
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn test_lower_nested_except_with_deep_path() {
    // Tests that nested EXCEPT expressions preserve the full base expression path
    // Bug: [node EXCEPT ![r].insts[1] = [node[r].insts[1] EXCEPT !.status = "x"]]
    // was incorrectly lowering the inner EXCEPT base as node[r] instead of node[r].insts[1]
    //
    // The CST for this pattern has sibling nodes for the postfix operations:
    //   ExceptExpr
    //     FuncApplyExpr: "node[r]"
    //     RecordAccessExpr: ".insts"
    //     FuncApplyExpr: "[1]"
    //     ExceptSpec: "!.status = \"x\""
    //
    // The fix chains these sibling postfix operations onto the base expression.
    let src = r#"
---- MODULE Test ----
VARIABLES node
Op(r) == [node EXCEPT ![r].insts[1] = [node[r].insts[1] EXCEPT !.status = "x"]]
====
"#;
    let tree = crate::parse_to_syntax_tree(src);
    let lower_result = crate::lower(FileId(0), &tree);

    assert!(
        lower_result.errors.is_empty(),
        "Expected no lowering errors, got: {:?}",
        lower_result.errors
    );

    let module = lower_result.module.expect("Module should be present");
    let op_def = module
        .units
        .iter()
        .find_map(|u| {
            if let Unit::Operator(def) = &u.node {
                if def.name.node == "Op" {
                    Some(def)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .expect("Op definition should exist");

    // The outer EXCEPT should have one spec with a nested EXCEPT as value
    if let Expr::Except(_, outer_specs) = &op_def.body.node {
        assert_eq!(outer_specs.len(), 1, "Outer EXCEPT should have one spec");

        // The value of the spec should be the inner EXCEPT
        if let Expr::Except(inner_base, _) = &outer_specs[0].value.node {
            // The inner EXCEPT's base should be node[r].insts[1], NOT just node[r]
            // The base should be FuncApply(RecordAccess(FuncApply(node, r), "insts"), 1)
            match &inner_base.node {
                Expr::FuncApply(inner_func, inner_arg) => {
                    // The inner_arg should be Int(1) (the [1] index)
                    assert!(
                        matches!(&inner_arg.node, Expr::Int(n) if *n == 1.into()),
                        "Expected inner EXCEPT base to end with [1], got arg {:?}",
                        inner_arg.node
                    );
                    // The inner_func should be RecordAccess(FuncApply(node, r), "insts")
                    match &inner_func.node {
                        Expr::RecordAccess(rec_base, field) => {
                            assert_eq!(
                                field.node, "insts",
                                "Expected .insts field access, got .{}",
                                field.node
                            );
                            // rec_base should be FuncApply(node, r) i.e., node[r]
                            match &rec_base.node {
                                Expr::FuncApply(node_expr, r_expr) => {
                                    assert!(
                                        matches!(&node_expr.node, Expr::Ident(name) if name == "node"),
                                        "Expected 'node', got {:?}",
                                        node_expr.node
                                    );
                                    assert!(
                                        matches!(&r_expr.node, Expr::Ident(name) if name == "r"),
                                        "Expected 'r', got {:?}",
                                        r_expr.node
                                    );
                                }
                                other => {
                                    panic!("Expected node[r] as base of .insts, got {:?}", other);
                                }
                            }
                        }
                        other => {
                            panic!("Expected node[r].insts as func of [1], got {:?}", other);
                        }
                    }
                }
                other => {
                    panic!(
                        "Inner EXCEPT base should be node[r].insts[1], got {:?}",
                        other
                    );
                }
            }
        } else {
            panic!(
                "Outer EXCEPT value should be inner EXCEPT, got {:?}",
                outer_specs[0].value.node
            );
        }
    } else {
        panic!(
            "Op body should be EXCEPT expression, got {:?}",
            op_def.body.node
        );
    }
}
