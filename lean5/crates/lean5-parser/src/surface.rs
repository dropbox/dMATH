//! Surface syntax AST
//!
//! The AST produced by the parser, before elaboration.
//! Named bindings, optional type annotations, no de Bruijn indices.

/// Span in source text for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[must_use]
    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }

    #[must_use]
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A parsed attribute like `@[instance 50]` or `@[defaultInstance]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attribute {
    /// `@[instance N]` - set instance priority to N
    InstancePriority(u32),
    /// `@[defaultInstance]` - set instance to lowest priority (used for fallbacks)
    DefaultInstance,
    /// Unknown attribute (stored for error reporting)
    Unknown(String),
}

impl Attribute {
    /// Get the instance priority from this attribute, if any
    #[must_use]
    pub fn instance_priority(&self) -> Option<u32> {
        match self {
            Attribute::InstancePriority(p) => Some(*p),
            // Default instance has priority 0 (lowest)
            Attribute::DefaultInstance => Some(0),
            Attribute::Unknown(_) => None,
        }
    }
}

/// Binder information for surface syntax
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceBinderInfo {
    /// Explicit: `(x : T)`
    #[default]
    Explicit,
    /// Implicit: `{x : T}`
    Implicit,
    /// Strict implicit: `{{x : T}}` or `⦃x : T⦄`
    StrictImplicit,
    /// Instance: `[x : T]`
    Instance,
}

/// A binder in surface syntax
#[derive(Debug, Clone)]
pub struct SurfaceBinder {
    pub span: Span,
    /// Binder name (can be "_" for anonymous)
    pub name: String,
    /// Optional type annotation
    pub ty: Option<Box<SurfaceExpr>>,
    /// Optional default value (e.g., `(x := 5)` or `(x : Nat := 5)`)
    pub default: Option<Box<SurfaceExpr>>,
    /// Binder kind (explicit, implicit, instance)
    pub info: SurfaceBinderInfo,
}

impl SurfaceBinder {
    pub fn new(name: impl Into<String>, ty: Option<SurfaceExpr>, info: SurfaceBinderInfo) -> Self {
        Self {
            span: Span::dummy(),
            name: name.into(),
            ty: ty.map(Box::new),
            default: None,
            info,
        }
    }

    pub fn explicit(name: impl Into<String>, ty: SurfaceExpr) -> Self {
        Self::new(name, Some(ty), SurfaceBinderInfo::Explicit)
    }

    pub fn implicit(name: impl Into<String>, ty: SurfaceExpr) -> Self {
        Self::new(name, Some(ty), SurfaceBinderInfo::Implicit)
    }

    pub fn instance(name: impl Into<String>, ty: SurfaceExpr) -> Self {
        Self::new(name, Some(ty), SurfaceBinderInfo::Instance)
    }
}

/// Path opened with optional selective names
#[derive(Debug, Clone)]
pub struct OpenPath {
    pub path: Vec<String>,
    pub names: Vec<String>,
}

/// Surface expression (before elaboration)
#[derive(Debug, Clone)]
pub enum SurfaceExpr {
    /// Identifier: `foo`, `Nat.add`
    Ident(Span, String),

    /// Universe: `Type`, `Type u`, `Prop`, `Sort u`
    Universe(Span, UniverseExpr),

    /// Application: `f x y`
    App(Span, Box<SurfaceExpr>, Vec<SurfaceArg>),

    /// Lambda: `fun x => e` or `fun (x : T) => e`
    Lambda(Span, Vec<SurfaceBinder>, Box<SurfaceExpr>),

    /// Pattern-matching lambda: `fun | pat => e | pat2 => e2`
    /// This is separate from Lambda to signal that application parsing should stop after this
    /// (layout-sensitive construct that we can't fully disambiguate without indentation info)
    PatternMatchLambda(Span, Vec<SurfaceBinder>, Box<SurfaceExpr>),

    /// Pi/forall: `∀ (x : A), B` or `(x : A) → B`
    Pi(Span, Vec<SurfaceBinder>, Box<SurfaceExpr>),

    /// Arrow (non-dependent): `A → B`
    Arrow(Span, Box<SurfaceExpr>, Box<SurfaceExpr>),

    /// Let binding: `let x := v in e` or `let x : T := v in e`
    Let(Span, SurfaceBinder, Box<SurfaceExpr>, Box<SurfaceExpr>),

    /// Recursive let binding: `let rec f (n : Nat) : Nat := ... in e`
    LetRec(Span, SurfaceBinder, Box<SurfaceExpr>, Box<SurfaceExpr>),

    /// Literal: `42`, `"hello"`
    Lit(Span, SurfaceLit),

    /// Parenthesized expression
    Paren(Span, Box<SurfaceExpr>),

    /// Hole/placeholder: `_`
    Hole(Span),

    /// Type ascription: `(e : T)`
    Ascription(Span, Box<SurfaceExpr>, Box<SurfaceExpr>),

    /// Out-parameter marker: `outParam T`
    /// Used in type class parameters to indicate output parameters
    OutParam(Span, Box<SurfaceExpr>),

    /// Semi-out-parameter marker: `semiOutParam T`
    /// Like outParam but allows unification in both directions during instance resolution.
    /// Instances promise to fill in this parameter, but it can also be constrained by context.
    SemiOutParam(Span, Box<SurfaceExpr>),

    /// If-then-else: `if c then t else e`
    If(Span, Box<SurfaceExpr>, Box<SurfaceExpr>, Box<SurfaceExpr>),

    /// If-let pattern match: `if let pat := e then t else f`
    /// Pattern, scrutinee, then-branch, else-branch
    IfLet(
        Span,
        SurfacePattern,
        Box<SurfaceExpr>,
        Box<SurfaceExpr>,
        Box<SurfaceExpr>,
    ),

    /// Decidable if: `if h : p then t else e`
    /// Binds proof witness `h` of proposition `p`
    /// witness name, proposition, then-branch, else-branch
    IfDecidable(
        Span,
        String,
        Box<SurfaceExpr>,
        Box<SurfaceExpr>,
        Box<SurfaceExpr>,
    ),

    /// Match expression (simplified for now)
    Match(Span, Box<SurfaceExpr>, Vec<SurfaceMatchArm>),

    /// Projection: `e.field` or `e.0`
    Proj(Span, Box<SurfaceExpr>, Projection),

    /// Universe instantiation: `Foo.{u v}` - explicit universe level arguments
    UniverseInst(Span, Box<SurfaceExpr>, Vec<LevelExpr>),

    /// Named argument: `(name := expr)` - used in function applications
    /// This represents the parenthesized named argument syntax
    NamedArg(Span, String, Box<SurfaceExpr>),

    /// Raw syntax quotation token (`` `(…) ``) preserved from the lexer
    /// Used by macro declarations and `macro_rules` patterns/expansions.
    SyntaxQuote(Span, String),

    /// Explicit application marker: `@f` - disables implicit argument insertion
    /// When elaborated, the following function will have all its implicit
    /// parameters treated as explicit, requiring explicit type arguments.
    Explicit(Span, Box<SurfaceExpr>),
}

/// Argument to a function application
#[derive(Debug, Clone)]
pub struct SurfaceArg {
    pub span: Span,
    /// The argument expression
    pub expr: SurfaceExpr,
    /// Named argument: `(name := e)`
    pub name: Option<String>,
    /// Explicit `@` or implicit
    pub explicit: bool,
}

impl SurfaceArg {
    #[must_use]
    pub fn positional(expr: SurfaceExpr) -> Self {
        let span = expr.span();
        Self {
            span,
            expr,
            name: None,
            explicit: true,
        }
    }

    #[must_use]
    pub fn named(name: String, expr: SurfaceExpr) -> Self {
        let span = expr.span();
        Self {
            span,
            expr,
            name: Some(name),
            explicit: true,
        }
    }
}

/// Universe expression
#[derive(Debug, Clone)]
pub enum UniverseExpr {
    /// Prop = Sort 0
    Prop,
    /// Type = Sort 1
    Type,
    /// Type u (explicit level)
    TypeLevel(Box<LevelExpr>),
    /// Sort u (explicit level)
    Sort(Box<LevelExpr>),
    /// Sort (implicit level, equivalent to Sort u for fresh u)
    SortImplicit,
}

/// Level expression (surface syntax for universe levels)
#[derive(Debug, Clone)]
pub enum LevelExpr {
    /// Numeric literal: 0, 1, 2, ...
    Lit(u32),
    /// Level parameter: u, v
    Param(String),
    /// Successor: u + 1
    Succ(Box<LevelExpr>),
    /// Max: max u v
    Max(Box<LevelExpr>, Box<LevelExpr>),
    /// `IMax`: `imax u v`
    IMax(Box<LevelExpr>, Box<LevelExpr>),
}

/// A match arm
#[derive(Debug, Clone)]
pub struct SurfaceMatchArm {
    pub span: Span,
    /// Pattern (simplified: just an identifier for now)
    pub pattern: SurfacePattern,
    /// Body expression
    pub body: SurfaceExpr,
}

/// Pattern for match (simplified for Phase 2)
#[derive(Debug, Clone)]
pub enum SurfacePattern {
    /// Variable pattern: `x`
    Var(String),
    /// Constructor pattern: `Nat.zero` or `Nat.succ n`
    Ctor(String, Vec<SurfacePattern>),
    /// Wildcard: `_`
    Wildcard,
    /// Literal pattern
    Lit(SurfaceLit),
    /// Numeral addition pattern: `n + 1` (sugar for successor patterns)
    NumeralAdd(Box<SurfacePattern>, u64),
    /// As pattern: `n@pat` - binds `n` to the matched value and checks `pat`
    As(String, Box<SurfacePattern>),
    /// Or pattern: `pat1 | pat2` - matches if either pattern matches
    Or(Box<SurfacePattern>, Box<SurfacePattern>),
}

impl SurfacePattern {
    /// Get a dummy span for the pattern (simplified)
    #[must_use]
    pub fn span(&self) -> Span {
        Span::dummy()
    }
}

/// Projection target
#[derive(Debug, Clone)]
pub enum Projection {
    /// Named field: `.foo`
    Named(String),
    /// Indexed field: `.1`, `.2`
    Index(u32),
}

/// Surface literal
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceLit {
    Nat(u64),
    String(String),
}

impl SurfaceExpr {
    #[allow(clippy::match_same_arms)] // Each variant returns span; explicit arms aids maintenance
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            SurfaceExpr::Ident(s, _) => *s,
            SurfaceExpr::Universe(s, _) => *s,
            SurfaceExpr::App(s, _, _) => *s,
            SurfaceExpr::Lambda(s, _, _) => *s,
            SurfaceExpr::PatternMatchLambda(s, _, _) => *s,
            SurfaceExpr::Pi(s, _, _) => *s,
            SurfaceExpr::Arrow(s, _, _) => *s,
            SurfaceExpr::Let(s, _, _, _) => *s,
            SurfaceExpr::LetRec(s, _, _, _) => *s,
            SurfaceExpr::Lit(s, _) => *s,
            SurfaceExpr::Paren(s, _) => *s,
            SurfaceExpr::Hole(s) => *s,
            SurfaceExpr::Ascription(s, _, _) => *s,
            SurfaceExpr::OutParam(s, _) => *s,
            SurfaceExpr::SemiOutParam(s, _) => *s,
            SurfaceExpr::If(s, _, _, _) => *s,
            SurfaceExpr::IfLet(s, _, _, _, _) => *s,
            SurfaceExpr::IfDecidable(s, _, _, _, _) => *s,
            SurfaceExpr::Match(s, _, _) => *s,
            SurfaceExpr::Proj(s, _, _) => *s,
            SurfaceExpr::UniverseInst(s, _, _) => *s,
            SurfaceExpr::NamedArg(s, _, _) => *s,
            SurfaceExpr::SyntaxQuote(s, _) => *s,
            SurfaceExpr::Explicit(s, _) => *s,
        }
    }

    /// Create a simple identifier
    pub fn ident(name: impl Into<String>) -> Self {
        SurfaceExpr::Ident(Span::dummy(), name.into())
    }

    /// Create an application
    pub fn app(func: SurfaceExpr, args: Vec<SurfaceExpr>) -> Self {
        let span = func.span();
        let args = args.into_iter().map(SurfaceArg::positional).collect();
        SurfaceExpr::App(span, Box::new(func), args)
    }

    /// Create a lambda
    #[must_use]
    pub fn lambda(binders: Vec<SurfaceBinder>, body: SurfaceExpr) -> Self {
        SurfaceExpr::Lambda(Span::dummy(), binders, Box::new(body))
    }

    /// Create an arrow type
    #[must_use]
    pub fn arrow(from: SurfaceExpr, to: SurfaceExpr) -> Self {
        let span = from.span().merge(to.span());
        SurfaceExpr::Arrow(span, Box::new(from), Box::new(to))
    }

    /// Create a pi type
    #[must_use]
    pub fn pi(binders: Vec<SurfaceBinder>, body: SurfaceExpr) -> Self {
        SurfaceExpr::Pi(Span::dummy(), binders, Box::new(body))
    }

    /// Create Type
    #[must_use]
    pub fn type_() -> Self {
        SurfaceExpr::Universe(Span::dummy(), UniverseExpr::Type)
    }

    /// Create Prop
    #[must_use]
    pub fn prop() -> Self {
        SurfaceExpr::Universe(Span::dummy(), UniverseExpr::Prop)
    }

    /// Create a nat literal
    #[must_use]
    pub fn nat(n: u64) -> Self {
        SurfaceExpr::Lit(Span::dummy(), SurfaceLit::Nat(n))
    }

    /// Create a hole
    #[must_use]
    pub fn hole() -> Self {
        SurfaceExpr::Hole(Span::dummy())
    }
}

/// A surface-level declaration
#[derive(Debug, Clone)]
pub enum SurfaceDecl {
    /// Definition: `def name : ty := val`
    Def {
        span: Span,
        name: String,
        universe_params: Vec<String>,
        binders: Vec<SurfaceBinder>,
        ty: Option<Box<SurfaceExpr>>,
        val: Box<SurfaceExpr>,
    },

    /// Theorem: `theorem name : ty := proof`
    Theorem {
        span: Span,
        name: String,
        universe_params: Vec<String>,
        binders: Vec<SurfaceBinder>,
        ty: Box<SurfaceExpr>,
        proof: Box<SurfaceExpr>,
    },

    /// Axiom: `axiom name : ty`
    Axiom {
        span: Span,
        name: String,
        universe_params: Vec<String>,
        binders: Vec<SurfaceBinder>,
        ty: Box<SurfaceExpr>,
    },

    /// Inductive type
    ///
    /// ```text
    /// inductive Option (α : Type) : Type
    /// | none : Option α
    /// | some : α → Option α
    /// deriving Repr, BEq
    /// ```
    Inductive {
        span: Span,
        name: String,
        universe_params: Vec<String>,
        binders: Vec<SurfaceBinder>,
        ty: Box<SurfaceExpr>,
        ctors: Vec<SurfaceCtor>,
        /// Deriving clauses (class names to derive)
        deriving: Vec<String>,
    },

    /// Structure (single-constructor inductive with named fields)
    ///
    /// ```text
    /// structure Point where
    ///   x : Nat
    ///   y : Nat
    /// deriving Repr, BEq
    /// ```
    Structure {
        span: Span,
        name: String,
        universe_params: Vec<String>,
        /// Parameters of the structure (before `where`)
        binders: Vec<SurfaceBinder>,
        /// Optional explicit result type (defaults to Type)
        ty: Option<Box<SurfaceExpr>>,
        /// Fields of the structure
        fields: Vec<SurfaceField>,
        /// Deriving clauses (class names to derive)
        deriving: Vec<String>,
    },

    /// Type class declaration (structure marked as a class)
    ///
    /// ```text
    /// class Add (α : Type) where
    ///   add : α → α → α
    /// ```
    Class {
        span: Span,
        name: String,
        universe_params: Vec<String>,
        /// Parameters of the class (before `where`)
        binders: Vec<SurfaceBinder>,
        /// Optional explicit result type (defaults to Type)
        ty: Option<Box<SurfaceExpr>>,
        /// Fields/methods of the class
        fields: Vec<SurfaceField>,
    },

    /// Type class instance declaration
    ///
    /// ```text
    /// instance : Add Nat where
    ///   add := Nat.add
    /// ```
    ///
    /// Or with explicit name:
    /// ```text
    /// instance instAddNat : Add Nat where
    ///   add := Nat.add
    /// ```
    Instance {
        span: Span,
        /// Optional instance name (can be auto-generated)
        name: Option<String>,
        universe_params: Vec<String>,
        /// Binders for instance parameters (e.g., `[Ord A]`)
        binders: Vec<SurfaceBinder>,
        /// The class type this instance provides (e.g., `Add Nat`)
        class_type: Box<SurfaceExpr>,
        /// Field assignments
        fields: Vec<SurfaceFieldAssign>,
        /// Optional priority attribute
        priority: Option<u32>,
    },

    /// Example: `example : ty := proof` (anonymous theorem, not saved to environment)
    Example {
        span: Span,
        binders: Vec<SurfaceBinder>,
        ty: Option<Box<SurfaceExpr>>,
        val: Box<SurfaceExpr>,
    },

    /// Import: `import Lean.Data.List` (supports multiple module paths)
    Import { span: Span, paths: Vec<Vec<String>> },

    /// Namespace: `namespace Foo ... end Foo`
    Namespace {
        span: Span,
        name: String,
        decls: Vec<SurfaceDecl>,
    },

    /// Section: `section Foo ... end Foo`
    Section {
        span: Span,
        name: Option<String>,
        decls: Vec<SurfaceDecl>,
    },

    /// Universe declaration: `universe u v`
    UniverseDecl { span: Span, names: Vec<String> },

    /// Variable declaration: `variable (x : Type)`
    Variable {
        span: Span,
        binders: Vec<SurfaceBinder>,
    },

    /// Open command: `open Nat in ...` or `open Nat (add mul)` (multiple paths allowed)
    Open {
        span: Span,
        paths: Vec<OpenPath>,
        /// Body expression or declarations if using `in`
        body: Option<Box<SurfaceDecl>>,
    },

    /// #check command: `#check expr`
    Check { span: Span, expr: Box<SurfaceExpr> },

    /// #eval command: `#eval expr`
    Eval { span: Span, expr: Box<SurfaceExpr> },

    /// #print command: `#print name`
    Print { span: Span, name: String },

    /// Mutual block: `mutual ... end`
    Mutual { span: Span, decls: Vec<SurfaceDecl> },

    /// Syntax declaration: defines a new syntax pattern
    ///
    /// ```text
    /// syntax [name] [prec:num]? term "+" term : term
    /// syntax:20 term "+" term : term  -- with precedence
    /// ```
    Syntax {
        span: Span,
        /// Optional name for the syntax (e.g., `[name]` attribute)
        name: Option<String>,
        /// Optional precedence level
        precedence: Option<u32>,
        /// Optional priority for disambiguation
        priority: Option<u32>,
        /// The syntax pattern (sequence of atoms, idents, category refs)
        pattern: Vec<SyntaxPatternItem>,
        /// The syntax category this extends (e.g., "term", "command", "tactic")
        category: String,
    },

    /// Declare a new syntax category
    ///
    /// ```text
    /// declare_syntax_cat mycat
    /// ```
    DeclareSyntaxCat {
        span: Span,
        /// The name of the new category
        name: String,
    },

    /// Macro declaration: short form for simple macros
    ///
    /// ```text
    /// macro "unless" cond:term "then" body:term : term =>
    ///   `(if !$cond then $body else ())
    /// ```
    Macro {
        span: Span,
        /// Optional doc comment
        doc: Option<String>,
        /// The syntax pattern to match
        pattern: Vec<SyntaxPatternItem>,
        /// The syntax category
        category: String,
        /// The expansion template (syntax quotation)
        expansion: Box<SurfaceExpr>,
    },

    /// Macro rules: multi-arm macro with pattern matching
    ///
    /// ```text
    /// macro_rules
    /// | `($x + $y) => `(Nat.add $x $y)
    /// | `($x - $y) => `(Nat.sub $x $y)
    /// ```
    MacroRules {
        span: Span,
        /// Optional name for the macro
        name: Option<String>,
        /// Match arms: (pattern, expansion)
        arms: Vec<MacroArm>,
    },

    /// Notation declaration: defines infix/prefix/postfix notation
    ///
    /// ```text
    /// infixl:65 " + " => Add.add
    /// prefix:max "!" => Not
    /// notation "⟨" a ", " b "⟩" => Prod.mk a b
    /// ```
    Notation {
        span: Span,
        /// The notation kind (infixl, infixr, prefix, postfix, notation)
        kind: NotationKind,
        /// Optional precedence level
        precedence: Option<u32>,
        /// The notation pattern
        pattern: Vec<NotationItem>,
        /// The expansion (function to apply)
        expansion: Box<SurfaceExpr>,
    },

    /// Elaborator declaration (advanced, for custom elaboration)
    Elab {
        span: Span,
        /// Raw content (not fully parsed)
        content: String,
    },

    /// Attribute application: `attribute [simp] foo bar`
    Attribute {
        span: Span,
        attrs: Vec<Attribute>,
        names: Vec<String>,
    },

    /// `set_option` command
    SetOption {
        span: Span,
        name: String,
        value: Option<String>,
    },
}

/// Constructor of an inductive type
#[derive(Debug, Clone)]
pub struct SurfaceCtor {
    pub span: Span,
    pub name: String,
    pub ty: SurfaceExpr,
}

/// A field in a structure declaration
#[derive(Debug, Clone)]
pub struct SurfaceField {
    pub span: Span,
    /// Field name
    pub name: String,
    /// Field type
    pub ty: SurfaceExpr,
    /// Default value (optional)
    pub default: Option<SurfaceExpr>,
}

/// A field assignment in an instance declaration
///
/// ```text
/// instance : Add Nat where
///   add := Nat.add   -- This is a SurfaceFieldAssign
/// ```
#[derive(Debug, Clone)]
pub struct SurfaceFieldAssign {
    pub span: Span,
    /// Field name
    pub name: String,
    /// Assigned value
    pub val: SurfaceExpr,
}

// ============================================================================
// Macro system types
// ============================================================================

/// An item in a syntax pattern
///
/// Syntax patterns consist of:
/// - Literal strings: `"if"`, `"+"`, `"=>"`
/// - Identifiers with category: `cond:term`, `body:tactic`
/// - Optional items: `("," expr)?`
/// - Repetitions: `expr,*` or `expr,+`
/// - Syntax category references: `term`, `tactic`
#[derive(Debug, Clone)]
pub enum SyntaxPatternItem {
    /// Literal string: `"if"`, `"then"`, `"+"`
    Literal(String),
    /// Variable binding with optional category: `x`, `cond:term`, `body:tactic`
    Variable {
        name: String,
        /// The syntax category (e.g., "term", "tactic"), if specified
        category: Option<String>,
    },
    /// Syntax category reference: `term`, `tactic`, `command`
    CategoryRef(String),
    /// Optional group: `(pattern)?`
    Optional(Vec<SyntaxPatternItem>),
    /// Repetition with separator: `pattern,*` or `pattern,+`
    Repetition {
        pattern: Vec<SyntaxPatternItem>,
        separator: Option<String>,
        at_least_one: bool,
    },
    /// Precedence specifier: `:50` or `:max`
    Precedence(PrecedenceLevel),
}

/// Precedence level for syntax declarations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecedenceLevel {
    /// Numeric precedence level (0-1024)
    Num(u32),
    /// Maximum precedence (for function application)
    Max,
    /// Minimum precedence (for low-priority operators)
    Min,
    /// Argument precedence (for function arguments)
    Arg,
    /// Lead precedence (for leading tokens)
    Lead,
}

/// A single arm in a `macro_rules` declaration
#[derive(Debug, Clone)]
pub struct MacroArm {
    pub span: Span,
    /// The pattern to match (typically a syntax quotation)
    pub pattern: Box<SurfaceExpr>,
    /// The expansion template (typically a syntax quotation)
    pub expansion: Box<SurfaceExpr>,
}

/// Kind of notation declaration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotationKind {
    /// Left-associative infix: `infixl`
    Infixl,
    /// Right-associative infix: `infixr`
    Infixr,
    /// Prefix operator: `prefix`
    Prefix,
    /// Postfix operator: `postfix`
    Postfix,
    /// General notation: `notation`
    Notation,
}

/// An item in a notation pattern
#[derive(Debug, Clone)]
pub enum NotationItem {
    /// Literal token: `"+"`, `"⟨"`, `","`
    Literal(String),
    /// Variable to be filled: `a`, `b`
    Variable(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_expr_construction() {
        let id = SurfaceExpr::ident("x");
        assert!(matches!(id, SurfaceExpr::Ident(_, s) if s == "x"));

        let ty = SurfaceExpr::type_();
        assert!(matches!(ty, SurfaceExpr::Universe(_, UniverseExpr::Type)));

        let prop = SurfaceExpr::prop();
        assert!(matches!(prop, SurfaceExpr::Universe(_, UniverseExpr::Prop)));
    }

    #[test]
    fn test_span_merge() {
        let s1 = Span::new(0, 5);
        let s2 = Span::new(10, 20);
        let merged = s1.merge(s2);
        assert_eq!(merged.start, 0);
        assert_eq!(merged.end, 20);
    }

    #[test]
    fn test_binder_construction() {
        let b = SurfaceBinder::explicit("x", SurfaceExpr::type_());
        assert_eq!(b.name, "x");
        assert!(b.ty.is_some());
        assert_eq!(b.info, SurfaceBinderInfo::Explicit);

        let b = SurfaceBinder::implicit("y", SurfaceExpr::prop());
        assert_eq!(b.info, SurfaceBinderInfo::Implicit);
    }
}
