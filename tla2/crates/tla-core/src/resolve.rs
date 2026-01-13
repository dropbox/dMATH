//! Name resolution and scope analysis for TLA+
//!
//! This module performs semantic analysis on the AST:
//! - Resolves identifier references to their definitions
//! - Tracks scope (module, LET, quantifier, function definition)
//! - Reports undefined references and duplicate definitions
//! - Builds a symbol table for downstream use
//!
//! # TLA+ Scoping Rules
//!
//! TLA+ has several scoping constructs:
//! - **Module scope**: VARIABLE, CONSTANT, operator definitions
//! - **LET scope**: Local definitions in LET...IN
//! - **Quantifier scope**: Bound variables in \A, \E, CHOOSE
//! - **Function scope**: Parameters in [x \in S |-> ...]
//! - **Set scope**: Variables in {e : x \in S} and {x \in S : P}
//! - **Lambda scope**: Parameters in LAMBDA x, y : body
//!
//! Inner scopes shadow outer scopes. EXTENDS brings external definitions
//! into module scope. INSTANCE creates parameterized imports.

use crate::ast::{
    BoundVar, CaseArm, ExceptSpec, Expr, Module, OperatorDef, Proof, ProofHint, ProofStep,
    ProofStepKind, TheoremDecl, Unit,
};
use crate::span::{Span, Spanned};
use std::collections::HashMap;

/// The kind of symbol in TLA+
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// State variable (VARIABLE declaration)
    Variable,
    /// Constant (CONSTANT declaration)
    Constant,
    /// Operator definition
    Operator,
    /// Bound variable (from quantifier, function def, set comprehension)
    BoundVar,
    /// Higher-order operator parameter
    OpParam,
    /// Module name (from EXTENDS or INSTANCE)
    Module,
}

/// A resolved symbol with metadata
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Name of the symbol
    pub name: String,
    /// Kind of symbol
    pub kind: SymbolKind,
    /// Span of the definition site
    pub def_span: Span,
    /// Arity for operators/constants (0 for non-operators)
    pub arity: usize,
    /// Whether the definition is LOCAL
    pub local: bool,
}

/// A scope level containing symbol bindings
#[derive(Debug, Clone)]
pub struct Scope {
    /// Symbols defined in this scope
    symbols: HashMap<String, Symbol>,
    /// Kind of scope (for error messages and diagnostics)
    #[allow(dead_code)] // Will be used for better error messages
    kind: ScopeKind,
}

/// The kind of scope
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Top-level module scope
    Module,
    /// LET...IN expression
    Let,
    /// Quantifier (\A, \E, CHOOSE)
    Quantifier,
    /// Function definition [x \in S |-> ...]
    Function,
    /// Set builder {e : x \in S}
    SetBuilder,
    /// Set filter {x \in S : P}
    SetFilter,
    /// Lambda expression
    Lambda,
    /// Proof step (TAKE, PICK, etc.)
    Proof,
}

/// Error during name resolution
#[derive(Debug, Clone)]
pub struct ResolveError {
    /// Error kind
    pub kind: ResolveErrorKind,
    /// Span where error occurred
    pub span: Span,
}

/// Kinds of resolution errors
#[derive(Debug, Clone)]
pub enum ResolveErrorKind {
    /// Reference to undefined identifier
    Undefined { name: String },
    /// Duplicate definition in same scope
    Duplicate { name: String, first_def: Span },
    /// Wrong arity in operator application
    ArityMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    /// Using variable where operator expected (or vice versa)
    KindMismatch {
        name: String,
        expected: SymbolKind,
        got: SymbolKind,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ResolveErrorKind::Undefined { name } => {
                write!(f, "undefined identifier `{}`", name)
            }
            ResolveErrorKind::Duplicate { name, .. } => {
                write!(f, "duplicate definition of `{}`", name)
            }
            ResolveErrorKind::ArityMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "operator `{}` expects {} arguments, got {}",
                    name, expected, got
                )
            }
            ResolveErrorKind::KindMismatch {
                name,
                expected,
                got,
            } => {
                write!(f, "`{}` is a {:?}, expected {:?}", name, got, expected)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// Context for name resolution
#[derive(Debug)]
pub struct ResolveCtx {
    /// Stack of scopes (innermost last)
    scopes: Vec<Scope>,
    /// Collected errors
    errors: Vec<ResolveError>,
    /// All resolved symbols (for symbol table)
    all_symbols: Vec<Symbol>,
    /// Reference sites: (use_span, def_span)
    references: Vec<(Span, Span)>,
}

impl Default for ResolveCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolveCtx {
    /// Create a new resolution context
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            errors: Vec::new(),
            all_symbols: Vec::new(),
            references: Vec::new(),
        }
    }

    /// Push a new scope
    pub fn push_scope(&mut self, kind: ScopeKind) {
        self.scopes.push(Scope {
            symbols: HashMap::new(),
            kind,
        });
    }

    /// Pop the current scope
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Define a symbol in the current scope
    pub fn define(&mut self, name: &str, kind: SymbolKind, span: Span, arity: usize, local: bool) {
        let symbol = Symbol {
            name: name.to_string(),
            kind,
            def_span: span,
            arity,
            local,
        };

        if let Some(scope) = self.scopes.last_mut() {
            // Check for duplicate in current scope
            if let Some(existing) = scope.symbols.get(name) {
                self.errors.push(ResolveError {
                    kind: ResolveErrorKind::Duplicate {
                        name: name.to_string(),
                        first_def: existing.def_span,
                    },
                    span,
                });
            } else {
                scope.symbols.insert(name.to_string(), symbol.clone());
            }
        }

        self.all_symbols.push(symbol);
    }

    /// Look up a symbol by name, searching from innermost scope outward
    /// Returns the symbol's def_span if found
    fn lookup_def_span(&self, name: &str) -> Option<Span> {
        for scope in self.scopes.iter().rev() {
            if let Some(sym) = scope.symbols.get(name) {
                return Some(sym.def_span);
            }
        }
        None
    }

    /// Look up a symbol by name, returning a clone to avoid borrow issues
    pub fn lookup(&self, name: &str) -> Option<Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(sym) = scope.symbols.get(name) {
                return Some(sym.clone());
            }
        }
        None
    }

    /// Record a reference to a symbol
    pub fn reference(&mut self, name: &str, use_span: Span) -> bool {
        if let Some(def_span) = self.lookup_def_span(name) {
            self.references.push((use_span, def_span));
            true
        } else {
            self.errors.push(ResolveError {
                kind: ResolveErrorKind::Undefined {
                    name: name.to_string(),
                },
                span: use_span,
            });
            false
        }
    }

    /// Get all errors
    pub fn errors(&self) -> &[ResolveError] {
        &self.errors
    }

    /// Check if resolution succeeded (no errors)
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get all defined symbols
    pub fn symbols(&self) -> &[Symbol] {
        &self.all_symbols
    }

    /// Get all references (use_span -> def_span)
    pub fn references(&self) -> &[(Span, Span)] {
        &self.references
    }
}

/// Resolve names in a module
pub fn resolve_module(module: &Module) -> ResolveCtx {
    let mut ctx = ResolveCtx::new();
    ctx.push_scope(ScopeKind::Module);

    // Inject standard library symbols based on EXTENDS declarations
    let extends: Vec<&str> = module.extends.iter().map(|s| s.node.as_str()).collect();
    crate::stdlib::inject_stdlib(&mut ctx, &extends);

    // First pass: collect all top-level definitions
    for unit in &module.units {
        match &unit.node {
            Unit::Variable(vars) => {
                for var in vars {
                    ctx.define(&var.node, SymbolKind::Variable, var.span, 0, false);
                }
            }
            Unit::Constant(consts) => {
                for c in consts {
                    let arity = c.arity.unwrap_or(0);
                    ctx.define(
                        &c.name.node,
                        SymbolKind::Constant,
                        c.name.span,
                        arity,
                        false,
                    );
                }
            }
            Unit::Recursive(decls) => {
                // RECURSIVE forward-declares operators that will be recursively defined
                for r in decls {
                    ctx.define(
                        &r.name.node,
                        SymbolKind::Operator,
                        r.name.span,
                        r.arity,
                        false,
                    );
                }
            }
            Unit::Operator(op) => {
                ctx.define(
                    &op.name.node,
                    SymbolKind::Operator,
                    op.name.span,
                    op.params.len(),
                    op.local,
                );
            }
            Unit::Instance(inst) => {
                // INSTANCE imports a module - the imported names depend on WITH substitutions
                // For now, just note the module reference
                ctx.define(
                    &inst.module.node,
                    SymbolKind::Module,
                    inst.module.span,
                    0,
                    inst.local,
                );
            }
            Unit::Theorem(thm) => {
                if let Some(name) = &thm.name {
                    ctx.define(&name.node, SymbolKind::Operator, name.span, 0, false);
                }
            }
            Unit::Assume(_) | Unit::Separator => {}
        }
    }

    // Second pass: resolve references in bodies
    for unit in &module.units {
        match &unit.node {
            Unit::Operator(op) => {
                resolve_operator_def(&mut ctx, op);
            }
            Unit::Assume(assume) => {
                resolve_expr(&mut ctx, &assume.expr);
            }
            Unit::Theorem(thm) => {
                resolve_theorem(&mut ctx, thm);
            }
            Unit::Instance(inst) => {
                resolve_instance(&mut ctx, inst);
            }
            Unit::Variable(_) | Unit::Constant(_) | Unit::Recursive(_) | Unit::Separator => {}
        }
    }

    ctx.pop_scope();
    ctx
}

/// Resolve an operator definition
fn resolve_operator_def(ctx: &mut ResolveCtx, op: &OperatorDef) {
    // Create scope for parameters
    ctx.push_scope(ScopeKind::Let);

    for param in &op.params {
        let kind = if param.arity > 0 {
            SymbolKind::OpParam
        } else {
            SymbolKind::BoundVar
        };
        ctx.define(&param.name.node, kind, param.name.span, param.arity, false);
    }

    resolve_expr(ctx, &op.body);

    ctx.pop_scope();
}

/// Resolve a theorem declaration
fn resolve_theorem(ctx: &mut ResolveCtx, thm: &TheoremDecl) {
    resolve_expr(ctx, &thm.body);

    if let Some(proof) = &thm.proof {
        resolve_proof(ctx, &proof.node);
    }
}

/// Resolve an instance declaration
fn resolve_instance(ctx: &mut ResolveCtx, inst: &crate::ast::InstanceDecl) {
    for sub in &inst.substitutions {
        // The 'from' name refers to something in the imported module (not checked here)
        // The 'to' expression is resolved in current scope
        resolve_expr(ctx, &sub.to);
    }
}

/// Resolve an expression
fn resolve_expr(ctx: &mut ResolveCtx, expr: &Spanned<Expr>) {
    match &expr.node {
        // Literals - nothing to resolve
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => {}

        // Identifier reference
        Expr::Ident(name) => {
            ctx.reference(name, expr.span);
        }

        // Operator reference (bare operator as value: +, -, *, etc.)
        // No resolution needed - these are built-in operators
        Expr::OpRef(_) => {}

        // Operator application
        Expr::Apply(op_expr, args) => {
            resolve_expr(ctx, op_expr);
            for arg in args {
                resolve_expr(ctx, arg);
            }
        }

        // Lambda
        Expr::Lambda(params, body) => {
            ctx.push_scope(ScopeKind::Lambda);
            for param in params {
                ctx.define(&param.node, SymbolKind::BoundVar, param.span, 0, false);
            }
            resolve_expr(ctx, body);
            ctx.pop_scope();
        }

        // Module reference (M!Op or M!Op(args))
        // The module and operator references are resolved at evaluation time
        // based on INSTANCE declarations. For now, just resolve the arguments.
        Expr::ModuleRef(_module, _op, args) => {
            for arg in args {
                resolve_expr(ctx, arg);
            }
        }

        // Instance expression (INSTANCE Module WITH ...)
        // Resolve substitution expressions
        Expr::InstanceExpr(_module, substitutions) => {
            for sub in substitutions {
                resolve_expr(ctx, &sub.to);
            }
        }

        // Binary logical operators
        Expr::And(l, r)
        | Expr::Or(l, r)
        | Expr::Implies(l, r)
        | Expr::Equiv(l, r)
        | Expr::In(l, r)
        | Expr::NotIn(l, r)
        | Expr::Subseteq(l, r)
        | Expr::Union(l, r)
        | Expr::Intersect(l, r)
        | Expr::SetMinus(l, r)
        | Expr::Eq(l, r)
        | Expr::Neq(l, r)
        | Expr::Lt(l, r)
        | Expr::Leq(l, r)
        | Expr::Gt(l, r)
        | Expr::Geq(l, r)
        | Expr::Add(l, r)
        | Expr::Sub(l, r)
        | Expr::Mul(l, r)
        | Expr::Div(l, r)
        | Expr::IntDiv(l, r)
        | Expr::Mod(l, r)
        | Expr::Pow(l, r)
        | Expr::Range(l, r)
        | Expr::LeadsTo(l, r) => {
            resolve_expr(ctx, l);
            resolve_expr(ctx, r);
        }

        // Unary operators
        Expr::Not(e)
        | Expr::Powerset(e)
        | Expr::BigUnion(e)
        | Expr::Domain(e)
        | Expr::Prime(e)
        | Expr::Always(e)
        | Expr::Eventually(e)
        | Expr::Enabled(e)
        | Expr::Unchanged(e)
        | Expr::Neg(e) => {
            resolve_expr(ctx, e);
        }

        // Quantifiers with bound variables
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            ctx.push_scope(ScopeKind::Quantifier);
            for bv in bounds {
                resolve_bound_var(ctx, bv);
            }
            resolve_expr(ctx, body);
            ctx.pop_scope();
        }

        // CHOOSE
        Expr::Choose(bv, body) => {
            ctx.push_scope(ScopeKind::Quantifier);
            resolve_bound_var(ctx, bv);
            resolve_expr(ctx, body);
            ctx.pop_scope();
        }

        // Set enumeration
        Expr::SetEnum(elems) => {
            for e in elems {
                resolve_expr(ctx, e);
            }
        }

        // Set builder {e : x \in S}
        Expr::SetBuilder(body, bounds) => {
            ctx.push_scope(ScopeKind::SetBuilder);
            for bv in bounds {
                resolve_bound_var(ctx, bv);
            }
            resolve_expr(ctx, body);
            ctx.pop_scope();
        }

        // Set filter {x \in S : P}
        Expr::SetFilter(bv, pred) => {
            ctx.push_scope(ScopeKind::SetFilter);
            resolve_bound_var(ctx, bv);
            resolve_expr(ctx, pred);
            ctx.pop_scope();
        }

        // Function definition [x \in S |-> e]
        Expr::FuncDef(bounds, body) => {
            ctx.push_scope(ScopeKind::Function);
            for bv in bounds {
                resolve_bound_var(ctx, bv);
            }
            resolve_expr(ctx, body);
            ctx.pop_scope();
        }

        // Function application f[x]
        Expr::FuncApply(f, arg) => {
            resolve_expr(ctx, f);
            resolve_expr(ctx, arg);
        }

        // Function set [S -> T]
        Expr::FuncSet(domain, codomain) => {
            resolve_expr(ctx, domain);
            resolve_expr(ctx, codomain);
        }

        // EXCEPT
        Expr::Except(base, specs) => {
            resolve_expr(ctx, base);
            for spec in specs {
                resolve_except_spec(ctx, spec);
            }
        }

        // Record constructor
        Expr::Record(fields) => {
            for (_, value) in fields {
                resolve_expr(ctx, value);
            }
        }

        // Record access
        Expr::RecordAccess(rec, _field) => {
            resolve_expr(ctx, rec);
            // Field names are not resolved as identifiers
        }

        // Record set
        Expr::RecordSet(fields) => {
            for (_, value) in fields {
                resolve_expr(ctx, value);
            }
        }

        // Tuple
        Expr::Tuple(elems) => {
            for e in elems {
                resolve_expr(ctx, e);
            }
        }

        // Cartesian product
        Expr::Times(factors) => {
            for f in factors {
                resolve_expr(ctx, f);
            }
        }

        // Temporal fairness
        Expr::WeakFair(vars, action) | Expr::StrongFair(vars, action) => {
            resolve_expr(ctx, vars);
            resolve_expr(ctx, action);
        }

        // IF-THEN-ELSE
        Expr::If(cond, then_e, else_e) => {
            resolve_expr(ctx, cond);
            resolve_expr(ctx, then_e);
            resolve_expr(ctx, else_e);
        }

        // CASE
        Expr::Case(arms, other) => {
            for arm in arms {
                resolve_case_arm(ctx, arm);
            }
            if let Some(o) = other {
                resolve_expr(ctx, o);
            }
        }

        // LET
        Expr::Let(defs, body) => {
            ctx.push_scope(ScopeKind::Let);

            // First pass: define all names
            for def in defs {
                ctx.define(
                    &def.name.node,
                    SymbolKind::Operator,
                    def.name.span,
                    def.params.len(),
                    def.local,
                );
            }

            // Second pass: resolve bodies (they can reference each other)
            for def in defs {
                resolve_operator_def(ctx, def);
            }

            resolve_expr(ctx, body);
            ctx.pop_scope();
        }
    }
}

/// Resolve a bound variable (define it and resolve its domain)
fn resolve_bound_var(ctx: &mut ResolveCtx, bv: &BoundVar) {
    // First resolve domain (it's in outer scope)
    if let Some(domain) = &bv.domain {
        resolve_expr(ctx, domain);
    }
    // Then define the variable (it's now in scope for body)
    ctx.define(&bv.name.node, SymbolKind::BoundVar, bv.name.span, 0, false);
}

/// Resolve an EXCEPT specification
fn resolve_except_spec(ctx: &mut ResolveCtx, spec: &ExceptSpec) {
    use crate::ast::ExceptPathElement;
    for elem in &spec.path {
        match elem {
            ExceptPathElement::Index(idx) => resolve_expr(ctx, idx),
            ExceptPathElement::Field(_) => {} // Field names not resolved
        }
    }
    resolve_expr(ctx, &spec.value);
}

/// Resolve a CASE arm
fn resolve_case_arm(ctx: &mut ResolveCtx, arm: &CaseArm) {
    resolve_expr(ctx, &arm.guard);
    resolve_expr(ctx, &arm.body);
}

/// Resolve a proof
fn resolve_proof(ctx: &mut ResolveCtx, proof: &Proof) {
    match proof {
        Proof::By(hints) => {
            for hint in hints {
                resolve_proof_hint(ctx, hint);
            }
        }
        Proof::Obvious | Proof::Omitted => {}
        Proof::Steps(steps) => {
            ctx.push_scope(ScopeKind::Proof);
            for step in steps {
                resolve_proof_step(ctx, step);
            }
            ctx.pop_scope();
        }
    }
}

/// Resolve a proof hint
fn resolve_proof_hint(ctx: &mut ResolveCtx, hint: &ProofHint) {
    match hint {
        ProofHint::Ref(name) => {
            ctx.reference(&name.node, name.span);
        }
        ProofHint::Def(names) => {
            for name in names {
                ctx.reference(&name.node, name.span);
            }
        }
        ProofHint::Module(_) => {
            // Module references not resolved here
        }
    }
}

/// Resolve a proof step
fn resolve_proof_step(ctx: &mut ResolveCtx, step: &ProofStep) {
    // Step labels are defined in proof scope
    if let Some(label) = &step.label {
        ctx.define(&label.node, SymbolKind::Operator, label.span, 0, false);
    }

    match &step.kind {
        ProofStepKind::Assert(expr, proof) => {
            resolve_expr(ctx, expr);
            if let Some(p) = proof {
                resolve_proof(ctx, &p.node);
            }
        }
        ProofStepKind::Suffices(expr, proof) => {
            resolve_expr(ctx, expr);
            if let Some(p) = proof {
                resolve_proof(ctx, &p.node);
            }
        }
        ProofStepKind::Have(expr) => {
            resolve_expr(ctx, expr);
        }
        ProofStepKind::Take(bounds) => {
            for bv in bounds {
                resolve_bound_var(ctx, bv);
            }
        }
        ProofStepKind::Witness(exprs) => {
            for e in exprs {
                resolve_expr(ctx, e);
            }
        }
        ProofStepKind::Pick(bounds, expr, proof) => {
            ctx.push_scope(ScopeKind::Quantifier);
            for bv in bounds {
                resolve_bound_var(ctx, bv);
            }
            resolve_expr(ctx, expr);
            if let Some(p) = proof {
                resolve_proof(ctx, &p.node);
            }
            ctx.pop_scope();
        }
        ProofStepKind::UseOrHide { facts, .. } => {
            for hint in facts {
                resolve_proof_hint(ctx, hint);
            }
        }
        ProofStepKind::Define(defs) => {
            for def in defs {
                ctx.define(
                    &def.name.node,
                    SymbolKind::Operator,
                    def.name.span,
                    def.params.len(),
                    def.local,
                );
                resolve_operator_def(ctx, def);
            }
        }
        ProofStepKind::Qed(proof) => {
            if let Some(p) = proof {
                resolve_proof(ctx, &p.node);
            }
        }
    }
}

/// Result of name resolution
#[derive(Debug)]
pub struct ResolveResult {
    /// All defined symbols
    pub symbols: Vec<Symbol>,
    /// All references (use_span -> def_span)
    pub references: Vec<(Span, Span)>,
    /// Resolution errors
    pub errors: Vec<ResolveError>,
}

impl ResolveResult {
    /// Check if resolution succeeded
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Resolve a module and return the result
pub fn resolve(module: &Module) -> ResolveResult {
    let ctx = resolve_module(module);
    ResolveResult {
        symbols: ctx.all_symbols,
        references: ctx.references,
        errors: ctx.errors,
    }
}

/// Inject all public symbols from a module into a resolution context.
///
/// This is used for EXTENDS: all non-LOCAL definitions from the extended
/// module become available in the extending module.
pub fn inject_module_symbols(ctx: &mut ResolveCtx, module: &Module) {
    let span = Span::dummy(); // Use dummy span for imported symbols

    // Inject variables
    for unit in &module.units {
        match &unit.node {
            Unit::Variable(vars) => {
                for var in vars {
                    ctx.define(&var.node, SymbolKind::Variable, span, 0, false);
                }
            }
            Unit::Constant(consts) => {
                for c in consts {
                    let arity = c.arity.unwrap_or(0);
                    ctx.define(&c.name.node, SymbolKind::Constant, span, arity, false);
                }
            }
            Unit::Operator(op) => {
                // Skip LOCAL operators
                if !op.local {
                    ctx.define(
                        &op.name.node,
                        SymbolKind::Operator,
                        span,
                        op.params.len(),
                        false,
                    );
                }
            }
            Unit::Theorem(thm) => {
                if let Some(name) = &thm.name {
                    ctx.define(&name.node, SymbolKind::Operator, span, 0, false);
                }
            }
            Unit::Recursive(decls) => {
                for r in decls {
                    ctx.define(&r.name.node, SymbolKind::Operator, span, r.arity, false);
                }
            }
            Unit::Instance(_) | Unit::Assume(_) | Unit::Separator => {}
        }
    }
}

/// Resolve a module with extended modules pre-loaded.
///
/// The `extended_modules` should be the already-loaded modules that this
/// module extends (non-stdlib). Their symbols will be injected into the
/// resolution context before resolving the main module.
pub fn resolve_with_extends(module: &Module, extended_modules: &[&Module]) -> ResolveResult {
    let mut ctx = ResolveCtx::new();
    ctx.push_scope(ScopeKind::Module);

    // First inject stdlib symbols
    let extends: Vec<&str> = module.extends.iter().map(|s| s.node.as_str()).collect();
    crate::stdlib::inject_stdlib(&mut ctx, &extends);

    // Then inject symbols from extended user modules
    for ext_mod in extended_modules {
        inject_module_symbols(&mut ctx, ext_mod);
    }

    // First pass: collect all top-level definitions
    for unit in &module.units {
        match &unit.node {
            Unit::Variable(vars) => {
                for var in vars {
                    ctx.define(&var.node, SymbolKind::Variable, var.span, 0, false);
                }
            }
            Unit::Constant(consts) => {
                for c in consts {
                    let arity = c.arity.unwrap_or(0);
                    ctx.define(
                        &c.name.node,
                        SymbolKind::Constant,
                        c.name.span,
                        arity,
                        false,
                    );
                }
            }
            Unit::Recursive(decls) => {
                for r in decls {
                    ctx.define(
                        &r.name.node,
                        SymbolKind::Operator,
                        r.name.span,
                        r.arity,
                        false,
                    );
                }
            }
            Unit::Operator(op) => {
                ctx.define(
                    &op.name.node,
                    SymbolKind::Operator,
                    op.name.span,
                    op.params.len(),
                    op.local,
                );
            }
            Unit::Instance(inst) => {
                ctx.define(
                    &inst.module.node,
                    SymbolKind::Module,
                    inst.module.span,
                    0,
                    inst.local,
                );
            }
            Unit::Theorem(thm) => {
                if let Some(name) = &thm.name {
                    ctx.define(&name.node, SymbolKind::Operator, name.span, 0, false);
                }
            }
            Unit::Assume(_) | Unit::Separator => {}
        }
    }

    // Second pass: resolve references in bodies
    for unit in &module.units {
        match &unit.node {
            Unit::Operator(op) => {
                resolve_operator_def(&mut ctx, op);
            }
            Unit::Assume(assume) => {
                resolve_expr(&mut ctx, &assume.expr);
            }
            Unit::Theorem(thm) => {
                resolve_theorem(&mut ctx, thm);
            }
            Unit::Instance(inst) => {
                resolve_instance(&mut ctx, inst);
            }
            Unit::Variable(_) | Unit::Constant(_) | Unit::Recursive(_) | Unit::Separator => {}
        }
    }

    ctx.pop_scope();

    ResolveResult {
        symbols: ctx.all_symbols,
        references: ctx.references,
        errors: ctx.errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::lower;
    use crate::span::FileId;
    use crate::syntax::parse_to_syntax_tree;

    fn resolve_source(src: &str) -> ResolveResult {
        let tree = parse_to_syntax_tree(src);
        let result = lower(FileId(0), &tree);
        assert!(
            result.errors.is_empty(),
            "Lower errors: {:?}",
            result.errors
        );
        let module = result.module.expect("Expected module");
        resolve(&module)
    }

    #[test]
    fn test_basic_operator() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE x
            Init == x = 0
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
        assert!(result
            .symbols
            .iter()
            .any(|s| s.name == "x" && s.kind == SymbolKind::Variable));
        assert!(result
            .symbols
            .iter()
            .any(|s| s.name == "Init" && s.kind == SymbolKind::Operator));
    }

    #[test]
    fn test_undefined_reference() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            Init == y = 0
            ====
            "#,
        );
        assert!(!result.is_ok());
        assert!(matches!(
            &result.errors[0].kind,
            ResolveErrorKind::Undefined { name } if name == "y"
        ));
    }

    #[test]
    fn test_duplicate_definition() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE x
            VARIABLE x
            ====
            "#,
        );
        assert!(!result.is_ok());
        assert!(matches!(
            &result.errors[0].kind,
            ResolveErrorKind::Duplicate { name, .. } if name == "x"
        ));
    }

    #[test]
    fn test_quantifier_scope() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE S
            AllPositive == \A x \in S : x > 0
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_let_scope() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            Double(n) == LET twice == n * 2 IN twice
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_set_builder_scope() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE S
            Squares == {x * x : x \in S}
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_function_def_scope() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE S
            SquareFunc == [x \in S |-> x * x]
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_lambda_scope() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            Twice == LAMBDA x : x + x
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_operator_parameters() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            Add(a, b) == a + b
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
        // Check that Add is an operator with arity 2
        let add_sym = result.symbols.iter().find(|s| s.name == "Add").unwrap();
        assert_eq!(add_sym.arity, 2);
    }

    #[test]
    fn test_constant_declaration() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            CONSTANT N
            Double == N * 2
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_nested_scopes() {
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE S
            Nested == \A x \in S : \E y \in S : x = y
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_shadowing() {
        // Inner x should shadow outer x
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE x
            Test == \A x \in {1,2,3} : x > 0
            ====
            "#,
        );
        assert!(result.is_ok(), "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_out_of_scope_reference() {
        // y should not be visible outside the quantifier
        let result = resolve_source(
            r#"
            ---- MODULE Test ----
            VARIABLE S
            Bad == (\A x \in S : x > 0) /\ y = 1
            ====
            "#,
        );
        assert!(!result.is_ok());
        assert!(matches!(
            &result.errors[0].kind,
            ResolveErrorKind::Undefined { name } if name == "y"
        ));
    }
}
