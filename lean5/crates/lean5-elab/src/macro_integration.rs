//! Macro system integration with elaboration
//!
//! This module provides the bridge between the macro system (lean5-macro)
//! and the elaborator (lean5-elab). It handles:
//!
//! - Converting between parser `SurfaceExpr` and macro `Syntax`
//! - Running macro expansion before elaboration
//! - Managing macro registries in the elaboration context
//!
//! # Architecture
//!
//! The macro expansion phase fits between parsing and elaboration:
//!
//! ```text
//! Source Code → Parser → SurfaceExpr → MacroExpand → SurfaceExpr → Elaboration → Expr
//! ```
//!
//! Macros work on `Syntax` (a generic AST), while elaboration works on
//! `SurfaceExpr` (the typed parser AST). This module provides conversion
//! functions between these representations.

use lean5_macro::quotation::parse_quotation;
use lean5_macro::registry::SyntaxCategoryRegistry;
use lean5_macro::{
    builtin_registry, HygienicExpander, MacroDef, MacroExpander, MacroRegistry, MacroResult,
    Syntax, SyntaxKind, SyntaxQuote,
};
use lean5_parser::{
    LevelExpr, MacroArm, NotationItem, NotationKind, Projection, Span, SurfaceArg, SurfaceBinder,
    SurfaceBinderInfo, SurfaceExpr, SurfaceLit, SurfaceMatchArm, SurfacePattern, SyntaxPatternItem,
    UniverseExpr,
};

/// Macro expansion context for elaboration
pub struct MacroCtx {
    /// The macro registry containing all available macros
    registry: MacroRegistry,
    /// Syntax category registry
    categories: SyntaxCategoryRegistry,
    /// Whether to use hygienic expansion
    hygienic: bool,
    /// Statistics from last expansion
    last_stats: Option<lean5_macro::expand::ExpansionStats>,
}

impl Default for MacroCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl MacroCtx {
    /// Create a new macro context with built-in macros
    pub fn new() -> Self {
        Self {
            registry: builtin_registry(),
            categories: SyntaxCategoryRegistry::new(),
            hygienic: true,
            last_stats: None,
        }
    }

    /// Create with a custom registry
    pub fn with_registry(registry: MacroRegistry) -> Self {
        Self {
            registry,
            categories: SyntaxCategoryRegistry::new(),
            hygienic: true,
            last_stats: None,
        }
    }

    /// Get read access to the registry
    pub fn registry(&self) -> &MacroRegistry {
        &self.registry
    }

    /// Get mutable access to the registry
    pub fn registry_mut(&mut self) -> &mut MacroRegistry {
        &mut self.registry
    }

    /// Enable or disable hygienic expansion
    pub fn set_hygienic(&mut self, hygienic: bool) {
        self.hygienic = hygienic;
    }

    /// Get statistics from last expansion
    pub fn last_stats(&self) -> Option<&lean5_macro::expand::ExpansionStats> {
        self.last_stats.as_ref()
    }

    /// Expand macros in syntax
    pub fn expand(&mut self, syntax: Syntax) -> MacroResult<Syntax> {
        if self.hygienic {
            let mut expander = HygienicExpander::new(&self.registry);
            let result = expander.expand(syntax)?;
            self.last_stats = Some(expander.stats().clone());
            Ok(result)
        } else {
            let mut expander = MacroExpander::new(&self.registry);
            let result = expander.expand(syntax)?;
            self.last_stats = Some(expander.stats().clone());
            Ok(result)
        }
    }

    /// Register a `macro_rules` declaration into the registry.
    pub fn register_macro_rules(
        &mut self,
        name: Option<&str>,
        arms: &[MacroArm],
    ) -> Result<(), MacroRegistrationError> {
        for (idx, arm) in arms.iter().enumerate() {
            let pattern_quote = surface_expr_to_syntax_quote(&arm.pattern)?;
            let expansion_quote = surface_expr_to_syntax_quote(&arm.expansion)?;

            let macro_name = name.map_or_else(|| format!("macro_rules_{idx}"), ToString::to_string);
            let target_kind = if matches!(
                (pattern_quote.syntax.kind(), arm.pattern.as_ref()),
                (Some(kind), SurfaceExpr::SyntaxQuote(_, content))
                    if kind == &SyntaxKind::antiquot() && content.starts_with('(')
            ) {
                SyntaxKind::paren()
            } else {
                pattern_quote
                    .syntax
                    .kind()
                    .cloned()
                    .unwrap_or_else(|| pattern_quote.category.clone())
            };

            let def = MacroDef::new(
                macro_name,
                target_kind,
                pattern_quote.syntax.clone(),
                expansion_quote,
            );
            self.registry.register(def);
        }

        Ok(())
    }

    /// Register a new syntax category (`declare_syntax_cat`).
    pub fn register_syntax_category(&mut self, name: &str) {
        use lean5_macro::registry::SyntaxCategory;
        self.categories.register(SyntaxCategory::new(name));
    }

    /// Check if a syntax category exists.
    pub fn has_syntax_category(&self, name: &str) -> bool {
        self.categories.exists(name)
    }

    /// Register a `syntax` declaration.
    ///
    /// This creates a macro that matches the syntax pattern and produces
    /// an AST node in the specified category.
    pub fn register_syntax(
        &mut self,
        name: Option<&str>,
        precedence: Option<u32>,
        pattern: &[SyntaxPatternItem],
        category: &str,
    ) -> Result<(), MacroRegistrationError> {
        // Build a pattern syntax from the pattern items
        let pattern_syntax = syntax_pattern_to_syntax(pattern);

        // The target kind is based on the pattern's leading literal/variable
        let target_kind = pattern_kind_from_items(pattern);

        // For syntax declarations, the expansion just wraps in the category
        let expansion = SyntaxQuote::new(pattern_syntax.clone(), SyntaxKind::app(category));

        let macro_name = name.map_or_else(
            || format!("syntax_{}", pattern_to_name(pattern)),
            ToString::to_string,
        );

        let mut def = MacroDef::new(macro_name, target_kind, pattern_syntax, expansion);
        if let Some(prec) = precedence {
            def = def.with_priority(prec as i32);
        }
        self.registry.register(def);

        Ok(())
    }

    /// Register a `notation` declaration (infixl, infixr, prefix, postfix, or notation).
    pub fn register_notation(
        &mut self,
        kind: NotationKind,
        precedence: Option<u32>,
        pattern: &[NotationItem],
        expansion: &SurfaceExpr,
    ) -> Result<(), MacroRegistrationError> {
        let expansion_quote = surface_expr_to_syntax_quote(expansion)?;

        // Build pattern syntax from notation items
        let (pattern_syntax, target_kind, var_names) = notation_pattern_to_syntax(kind, pattern);

        // Create macro name from pattern literals
        let macro_name = notation_to_name(kind, pattern);

        // Build the actual expansion: apply the expansion expression to variables
        let actual_expansion = if var_names.is_empty() {
            expansion_quote
        } else {
            // The expansion becomes: expansion var1 var2 ...
            let base = expansion_quote.syntax;
            let applied = var_names.iter().fold(base, |acc, var| {
                Syntax::mk_app(acc, vec![Syntax::mk_antiquot(var)])
            });
            SyntaxQuote::new(applied, expansion_quote.category)
        };

        let mut def = MacroDef::new(macro_name, target_kind, pattern_syntax, actual_expansion);
        if let Some(prec) = precedence {
            def = def.with_priority(prec as i32);
        }
        self.registry.register(def);

        Ok(())
    }

    /// Register a `macro` declaration (simple form with single pattern).
    pub fn register_macro(
        &mut self,
        pattern: &[SyntaxPatternItem],
        _category: &str,
        expansion: &SurfaceExpr,
    ) -> Result<(), MacroRegistrationError> {
        let pattern_syntax = syntax_pattern_to_syntax(pattern);
        let expansion_quote = surface_expr_to_syntax_quote(expansion)?;

        let target_kind = pattern_kind_from_items(pattern);
        let macro_name = format!("macro_{}", pattern_to_name(pattern));

        let def = MacroDef::new(macro_name, target_kind, pattern_syntax, expansion_quote);
        self.registry.register(def);

        Ok(())
    }
}

/// Convert a syntax pattern (from `syntax` decl) to Syntax AST.
fn syntax_pattern_to_syntax(items: &[SyntaxPatternItem]) -> Syntax {
    if items.is_empty() {
        return Syntax::missing();
    }

    let children: Vec<Syntax> = items.iter().map(syntax_pattern_item_to_syntax).collect();

    if children.len() == 1 {
        children
            .into_iter()
            .next()
            .expect("children has exactly 1 element")
    } else {
        // Create a sequence node
        Syntax::node(SyntaxKind::app("seq"), children)
    }
}

/// Convert a single SyntaxPatternItem to Syntax.
fn syntax_pattern_item_to_syntax(item: &SyntaxPatternItem) -> Syntax {
    match item {
        SyntaxPatternItem::Literal(s) => Syntax::atom(s),
        SyntaxPatternItem::Variable { name, category } => {
            if category.is_some() {
                // Variable with category becomes an antiquotation
                Syntax::mk_antiquot(name)
            } else {
                Syntax::mk_antiquot(name)
            }
        }
        SyntaxPatternItem::CategoryRef(cat) => Syntax::mk_antiquot(cat),
        SyntaxPatternItem::Optional(inner) => {
            let inner_syn = syntax_pattern_to_syntax(inner);
            Syntax::node(SyntaxKind::app("optional"), vec![inner_syn])
        }
        SyntaxPatternItem::Repetition {
            pattern,
            separator,
            at_least_one,
        } => {
            let inner_syn = syntax_pattern_to_syntax(pattern);
            let sep_syn = separator
                .as_ref()
                .map_or_else(Syntax::missing, |s| Syntax::atom(s));
            let kind = if *at_least_one { "rep1" } else { "rep0" };
            Syntax::node(SyntaxKind::app(kind), vec![inner_syn, sep_syn])
        }
        SyntaxPatternItem::Precedence(_) => Syntax::missing(), // Precedence is metadata, not pattern
    }
}

/// Determine the target kind from pattern items.
fn pattern_kind_from_items(items: &[SyntaxPatternItem]) -> SyntaxKind {
    // Look for the first literal or use app_kind as default
    for item in items {
        if let SyntaxPatternItem::Literal(lit) = item {
            return SyntaxKind::app(lit);
        }
    }
    SyntaxKind::app_kind()
}

/// Generate a name from pattern items.
fn pattern_to_name(items: &[SyntaxPatternItem]) -> String {
    items
        .iter()
        .filter_map(|item| match item {
            SyntaxPatternItem::Literal(s) => Some(s.replace(' ', "_")),
            SyntaxPatternItem::Variable { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("_")
}

/// Convert notation pattern to Syntax, returning pattern, target kind, and variable names.
fn notation_pattern_to_syntax(
    kind: NotationKind,
    items: &[NotationItem],
) -> (Syntax, SyntaxKind, Vec<String>) {
    let mut children = Vec::new();
    let mut var_names = Vec::new();

    for item in items {
        match item {
            NotationItem::Literal(s) => {
                children.push(Syntax::atom(s));
            }
            NotationItem::Variable(name) => {
                children.push(Syntax::mk_antiquot(name));
                var_names.push(name.clone());
            }
        }
    }

    let target_kind = match kind {
        NotationKind::Infixl | NotationKind::Infixr => {
            // Infix: look for operator literal
            items
                .iter()
                .find_map(|i| {
                    if let NotationItem::Literal(s) = i {
                        Some(SyntaxKind::app(s.trim()))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(SyntaxKind::app_kind)
        }
        NotationKind::Prefix | NotationKind::Postfix => {
            // Prefix/postfix: use the literal as kind
            items
                .iter()
                .find_map(|i| {
                    if let NotationItem::Literal(s) = i {
                        Some(SyntaxKind::app(s.trim()))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(SyntaxKind::app_kind)
        }
        NotationKind::Notation => {
            // General notation: use first literal or app_kind
            items
                .iter()
                .find_map(|i| {
                    if let NotationItem::Literal(s) = i {
                        Some(SyntaxKind::app(s.trim()))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(SyntaxKind::app_kind)
        }
    };

    let pattern = if children.len() == 1 {
        children
            .into_iter()
            .next()
            .expect("children has exactly 1 element")
    } else {
        Syntax::node(SyntaxKind::app("notation"), children)
    };

    (pattern, target_kind, var_names)
}

/// Generate name from notation.
fn notation_to_name(kind: NotationKind, items: &[NotationItem]) -> String {
    let prefix = match kind {
        NotationKind::Infixl => "infixl",
        NotationKind::Infixr => "infixr",
        NotationKind::Prefix => "prefix",
        NotationKind::Postfix => "postfix",
        NotationKind::Notation => "notation",
    };

    let parts: Vec<String> = items
        .iter()
        .map(|i| match i {
            NotationItem::Literal(s) => s.trim().replace(' ', "_"),
            NotationItem::Variable(v) => v.clone(),
        })
        .collect();

    format!("{}_{}", prefix, parts.join("_"))
}

/// Errors that can occur while registering macros from surface syntax
#[derive(Debug)]
pub enum MacroRegistrationError {
    /// Failed to parse a syntax quotation
    QuotationParse(String),
}

impl std::fmt::Display for MacroRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacroRegistrationError::QuotationParse(msg) => {
                write!(f, "failed to parse syntax quotation: {msg}")
            }
        }
    }
}

impl std::error::Error for MacroRegistrationError {}

/// Convert a surface expression that may contain a syntax quotation into a `SyntaxQuote`.
fn surface_expr_to_syntax_quote(expr: &SurfaceExpr) -> Result<SyntaxQuote, MacroRegistrationError> {
    match expr {
        SurfaceExpr::SyntaxQuote(_, content) => {
            let raw = format!("`{content}");
            parse_quotation(&raw).map_err(|e| MacroRegistrationError::QuotationParse(e.to_string()))
        }
        other => Ok(SyntaxQuote::new(
            surface_to_syntax(other),
            SyntaxKind::term(),
        )),
    }
}

/// Convert a level expression to syntax for macro transport.
fn level_to_syntax(level: &LevelExpr) -> Syntax {
    match level {
        LevelExpr::Lit(n) => Syntax::node(
            SyntaxKind::app("levelLit"),
            vec![Syntax::mk_num(u64::from(*n))],
        ),
        LevelExpr::Param(name) => {
            Syntax::node(SyntaxKind::app("levelParam"), vec![Syntax::ident(name)])
        }
        LevelExpr::Succ(inner) => {
            Syntax::node(SyntaxKind::app("levelSucc"), vec![level_to_syntax(inner)])
        }
        LevelExpr::Max(a, b) => Syntax::node(
            SyntaxKind::app("levelMax"),
            vec![level_to_syntax(a), level_to_syntax(b)],
        ),
        LevelExpr::IMax(a, b) => Syntax::node(
            SyntaxKind::app("levelIMax"),
            vec![level_to_syntax(a), level_to_syntax(b)],
        ),
    }
}

/// Convert macro syntax back into a level expression.
fn syntax_to_level(syntax: &Syntax) -> Option<LevelExpr> {
    let kind = syntax.kind()?;
    match kind.name_str() {
        "levelLit" => {
            // The child is a "num" node containing an atom with the value
            let num_node = syntax.child(0)?;
            let value = if let Some(atom_val) = num_node.as_atom() {
                // Direct atom (shouldn't happen, but handle it)
                atom_val.parse::<u64>().ok()?
            } else if num_node.kind().map(SyntaxKind::name_str) == Some("num") {
                // It's a "num" node - extract the atom child
                num_node.child(0)?.as_atom()?.parse::<u64>().ok()?
            } else {
                return None;
            };
            Some(LevelExpr::Lit(value as u32))
        }
        "levelParam" => syntax
            .child(0)?
            .as_ident()
            .map(|s| LevelExpr::Param(s.to_string())),
        "levelSucc" => Some(LevelExpr::Succ(Box::new(syntax_to_level(
            syntax.child(0)?,
        )?))),
        "levelMax" => Some(LevelExpr::Max(
            Box::new(syntax_to_level(syntax.child(0)?)?),
            Box::new(syntax_to_level(syntax.child(1)?)?),
        )),
        "levelIMax" => Some(LevelExpr::IMax(
            Box::new(syntax_to_level(syntax.child(0)?)?),
            Box::new(syntax_to_level(syntax.child(1)?)?),
        )),
        _ => None,
    }
}

// ============================================================================
// Conversion: SurfaceExpr → Syntax
// ============================================================================

/// Convert a surface expression to macro syntax
pub fn surface_to_syntax(expr: &SurfaceExpr) -> Syntax {
    match expr {
        SurfaceExpr::Ident(_, name) => Syntax::ident(name),

        SurfaceExpr::Universe(_, univ) => {
            let mut children = Vec::new();
            let tag = match univ {
                UniverseExpr::Prop => "Prop",
                UniverseExpr::Type => "Type",
                UniverseExpr::TypeLevel(level) => {
                    children.push(level_to_syntax(level));
                    "TypeLevel"
                }
                UniverseExpr::Sort(level) => {
                    children.push(level_to_syntax(level));
                    "Sort"
                }
                UniverseExpr::SortImplicit => "SortImplicit",
            };
            let mut all_children = Vec::with_capacity(1 + children.len());
            all_children.push(Syntax::ident(tag));
            all_children.extend(children);
            Syntax::node(SyntaxKind::app("universe"), all_children)
        }

        SurfaceExpr::App(_, func, args) => {
            let func_syn = surface_to_syntax(func);
            let args_syn: Vec<_> = args.iter().map(surface_arg_to_syntax).collect();
            Syntax::mk_app(func_syn, args_syn)
        }

        SurfaceExpr::Lambda(_, binders, body) => {
            let binders_syn: Vec<_> = binders.iter().map(surface_binder_to_syntax).collect();
            let body_syn = surface_to_syntax(body);
            Syntax::mk_lambda(binders_syn, body_syn)
        }

        SurfaceExpr::PatternMatchLambda(_, binders, body) => {
            // Pattern match lambda - similar to regular lambda for now
            let binders_syn: Vec<_> = binders.iter().map(surface_binder_to_syntax).collect();
            let body_syn = surface_to_syntax(body);
            Syntax::mk_lambda(binders_syn, body_syn)
        }

        SurfaceExpr::Pi(_, binders, body) => {
            let binders_syn: Vec<_> = binders.iter().map(surface_binder_to_syntax).collect();
            let body_syn = surface_to_syntax(body);
            Syntax::mk_forall(binders_syn, body_syn)
        }

        SurfaceExpr::Arrow(_, from, to) => {
            let from_syn = surface_to_syntax(from);
            let to_syn = surface_to_syntax(to);
            Syntax::mk_arrow(from_syn, to_syn)
        }

        SurfaceExpr::Let(_, binder, val, body) => {
            let name_syn = Syntax::ident(&binder.name);
            let ty_syn = binder.ty.as_ref().map(|t| surface_to_syntax(t));
            let val_syn = surface_to_syntax(val);
            let body_syn = surface_to_syntax(body);
            Syntax::mk_let(name_syn, ty_syn, val_syn, body_syn)
        }

        SurfaceExpr::Lit(_, lit) => match lit {
            SurfaceLit::Nat(n) => Syntax::mk_num(*n),
            SurfaceLit::String(s) => Syntax::mk_str(s),
        },

        SurfaceExpr::Paren(_, inner) => Syntax::mk_paren(surface_to_syntax(inner)),

        SurfaceExpr::Hole(_) => Syntax::mk_hole(),

        SurfaceExpr::Ascription(_, expr, ty) => {
            let expr_syn = surface_to_syntax(expr);
            let ty_syn = surface_to_syntax(ty);
            Syntax::node(SyntaxKind::app("ascription"), vec![expr_syn, ty_syn])
        }

        SurfaceExpr::OutParam(_, inner) => {
            let inner_syn = surface_to_syntax(inner);
            Syntax::node(SyntaxKind::app("outParam"), vec![inner_syn])
        }

        SurfaceExpr::SemiOutParam(_, inner) => {
            let inner_syn = surface_to_syntax(inner);
            Syntax::node(SyntaxKind::app("semiOutParam"), vec![inner_syn])
        }

        SurfaceExpr::If(_, cond, then_br, else_br) => {
            let cond_syn = surface_to_syntax(cond);
            let then_syn = surface_to_syntax(then_br);
            let else_syn = surface_to_syntax(else_br);
            Syntax::node(
                SyntaxKind::if_then_else(),
                vec![cond_syn, then_syn, else_syn],
            )
        }

        SurfaceExpr::Match(_, scrutinee, arms) => {
            let scrutinee_syn = surface_to_syntax(scrutinee);
            let mut children = vec![scrutinee_syn];
            for arm in arms {
                children.push(Syntax::node(
                    SyntaxKind::match_arm(),
                    vec![
                        surface_pattern_to_syntax(&arm.pattern),
                        surface_to_syntax(&arm.body),
                    ],
                ));
            }
            Syntax::node(SyntaxKind::match_expr(), children)
        }

        SurfaceExpr::Proj(_, expr, proj) => {
            let expr_syn = surface_to_syntax(expr);
            let field_syn = match proj {
                Projection::Named(name) => Syntax::ident(name),
                Projection::Index(idx) => Syntax::atom(&idx.to_string()),
            };
            Syntax::node(SyntaxKind::app("projection"), vec![expr_syn, field_syn])
        }

        SurfaceExpr::UniverseInst(_, expr, _levels) => {
            // Universe instantiation - just convert the expression for now
            surface_to_syntax(expr)
        }

        SurfaceExpr::NamedArg(_, name, expr) => {
            let name_syn = Syntax::ident(name);
            let expr_syn = surface_to_syntax(expr);
            Syntax::node(SyntaxKind::app("namedArg"), vec![name_syn, expr_syn])
        }

        SurfaceExpr::SyntaxQuote(_, content) => {
            parse_quotation(&format!("`{content}")).map_or_else(|_| Syntax::missing(), |q| q.syntax)
        }

        SurfaceExpr::LetRec(_, binder, val, body) => {
            // Similar to Let, but marked as recursive
            let name_syn = Syntax::ident(&binder.name);
            let ty_syn = binder.ty.as_ref().map(|t| surface_to_syntax(t));
            let val_syn = surface_to_syntax(val);
            let body_syn = surface_to_syntax(body);
            // For now, use the same let representation with a "rec" marker
            Syntax::node(
                SyntaxKind::app("letRec"),
                if let Some(ty) = ty_syn {
                    vec![name_syn, ty, val_syn, body_syn]
                } else {
                    vec![name_syn, val_syn, body_syn]
                },
            )
        }

        SurfaceExpr::IfLet(_, pat, scrutinee, then_br, else_br) => {
            let pat_syn = surface_pattern_to_syntax(pat);
            let scrutinee_syn = surface_to_syntax(scrutinee);
            let then_syn = surface_to_syntax(then_br);
            let else_syn = surface_to_syntax(else_br);
            Syntax::node(
                SyntaxKind::app("ifLet"),
                vec![pat_syn, scrutinee_syn, then_syn, else_syn],
            )
        }

        SurfaceExpr::IfDecidable(_, witness_name, prop, then_br, else_br) => {
            let witness_syn = Syntax::ident(witness_name);
            let prop_syn = surface_to_syntax(prop);
            let then_syn = surface_to_syntax(then_br);
            let else_syn = surface_to_syntax(else_br);
            Syntax::node(
                SyntaxKind::app("ifDecidable"),
                vec![witness_syn, prop_syn, then_syn, else_syn],
            )
        }

        SurfaceExpr::Explicit(_, inner) => {
            // Explicit application marker: @f
            // Wrap the inner syntax in an "explicit" node
            let inner_syn = surface_to_syntax(inner);
            Syntax::node(SyntaxKind::app("explicit"), vec![inner_syn])
        }
    }
}

/// Convert a surface pattern to syntax
fn surface_pattern_to_syntax(pattern: &SurfacePattern) -> Syntax {
    match pattern {
        SurfacePattern::Wildcard => Syntax::ident("_"),
        SurfacePattern::Var(name) => Syntax::ident(name),
        SurfacePattern::Ctor(name, args) => {
            let mut children = vec![Syntax::ident(name)];
            children.extend(args.iter().map(surface_pattern_to_syntax));
            Syntax::node(SyntaxKind::app("ctorPattern"), children)
        }
        SurfacePattern::Lit(lit) => match lit {
            SurfaceLit::Nat(n) => Syntax::mk_num(*n),
            SurfaceLit::String(s) => Syntax::mk_str(s),
        },
        SurfacePattern::NumeralAdd(pat, n) => {
            let pat_syn = surface_pattern_to_syntax(pat);
            let n_syn = Syntax::mk_num(*n);
            Syntax::node(SyntaxKind::app("numeralAddPattern"), vec![pat_syn, n_syn])
        }
        SurfacePattern::As(name, pat) => {
            // As-pattern: name@pat
            let name_syn = Syntax::ident(name);
            let pat_syn = surface_pattern_to_syntax(pat);
            Syntax::node(SyntaxKind::app("asPattern"), vec![name_syn, pat_syn])
        }
        SurfacePattern::Or(left, right) => {
            // Or-pattern: pat1 | pat2
            let left_syn = surface_pattern_to_syntax(left);
            let right_syn = surface_pattern_to_syntax(right);
            Syntax::node(SyntaxKind::app("orPattern"), vec![left_syn, right_syn])
        }
    }
}

/// Convert a surface argument to syntax
fn surface_arg_to_syntax(arg: &SurfaceArg) -> Syntax {
    let expr_syn = surface_to_syntax(&arg.expr);

    if let Some(name) = &arg.name {
        // Named argument
        Syntax::node(
            SyntaxKind::app("namedArg"),
            vec![Syntax::ident(name), expr_syn],
        )
    } else if !arg.explicit {
        // Implicit argument
        Syntax::node(SyntaxKind::app("implicitArg"), vec![expr_syn])
    } else {
        // Explicit positional argument
        expr_syn
    }
}

/// Convert a surface binder to syntax
fn surface_binder_to_syntax(binder: &SurfaceBinder) -> Syntax {
    let name = Syntax::ident(&binder.name);
    let ty = binder
        .ty
        .as_ref()
        .map_or_else(Syntax::missing, |t| surface_to_syntax(t));

    let kind_name = match binder.info {
        SurfaceBinderInfo::Explicit => "binderDefault",
        SurfaceBinderInfo::Implicit => "binderImplicit",
        SurfaceBinderInfo::Instance => "binderInstance",
        SurfaceBinderInfo::StrictImplicit => "binderStrictImplicit",
    };

    Syntax::node(SyntaxKind::app(kind_name), vec![name, ty])
}

// ============================================================================
// Conversion: Syntax → SurfaceExpr
// ============================================================================

/// Convert macro syntax back to a surface expression
///
/// This is used after macro expansion to continue with elaboration.
/// Returns None if the syntax cannot be converted.
pub fn syntax_to_surface(syntax: &Syntax) -> Option<SurfaceExpr> {
    match syntax {
        Syntax::Ident(_, name) => match name.as_str() {
            "Type" => Some(SurfaceExpr::Universe(Span::dummy(), UniverseExpr::Type)),
            "Prop" => Some(SurfaceExpr::Universe(Span::dummy(), UniverseExpr::Prop)),
            _ => Some(SurfaceExpr::Ident(Span::dummy(), name.clone())),
        },

        Syntax::Atom(_, value) => {
            // Try to parse as number
            if let Ok(n) = value.parse::<u64>() {
                Some(SurfaceExpr::Lit(Span::dummy(), SurfaceLit::Nat(n)))
            } else {
                Some(SurfaceExpr::Lit(
                    Span::dummy(),
                    SurfaceLit::String(value.clone()),
                ))
            }
        }

        Syntax::Missing(_) => Some(SurfaceExpr::Hole(Span::dummy())),

        Syntax::Node(node) => {
            let kind_name = node.kind.name_str();

            match kind_name {
                "app" => {
                    if node.children.is_empty() {
                        return None;
                    }
                    let func = syntax_to_surface(&node.children[0])?;
                    let args: Option<Vec<_>> = node.children[1..]
                        .iter()
                        .map(|c| syntax_to_surface(c).map(SurfaceArg::positional))
                        .collect();
                    Some(SurfaceExpr::App(Span::dummy(), Box::new(func), args?))
                }

                "fun" | "lambda" => {
                    if node.children.is_empty() {
                        return None;
                    }
                    let body_idx = node.children.len() - 1;
                    let binders: Option<Vec<_>> = node.children[..body_idx]
                        .iter()
                        .map(syntax_to_binder)
                        .collect();
                    let body = syntax_to_surface(&node.children[body_idx])?;
                    Some(SurfaceExpr::Lambda(Span::dummy(), binders?, Box::new(body)))
                }

                "forall" | "Pi" => {
                    if node.children.is_empty() {
                        return None;
                    }
                    let body_idx = node.children.len() - 1;
                    let binders: Option<Vec<_>> = node.children[..body_idx]
                        .iter()
                        .map(syntax_to_binder)
                        .collect();
                    let body = syntax_to_surface(&node.children[body_idx])?;
                    Some(SurfaceExpr::Pi(Span::dummy(), binders?, Box::new(body)))
                }

                "arrow" => {
                    if node.children.len() != 2 {
                        return None;
                    }
                    let from = syntax_to_surface(&node.children[0])?;
                    let to = syntax_to_surface(&node.children[1])?;
                    Some(SurfaceExpr::Arrow(
                        Span::dummy(),
                        Box::new(from),
                        Box::new(to),
                    ))
                }

                "let" => {
                    if node.children.len() < 3 {
                        return None;
                    }
                    let name = node.children[0].as_ident()?.to_string();
                    let (ty, val_idx) = if node.children.len() == 4 {
                        (Some(Box::new(syntax_to_surface(&node.children[1])?)), 2)
                    } else {
                        (None, 1)
                    };
                    let val = syntax_to_surface(&node.children[val_idx])?;
                    let body = syntax_to_surface(&node.children[val_idx + 1])?;
                    Some(SurfaceExpr::Let(
                        Span::dummy(),
                        SurfaceBinder {
                            span: Span::dummy(),
                            name,
                            ty,
                            default: None,
                            info: SurfaceBinderInfo::Explicit,
                        },
                        Box::new(val),
                        Box::new(body),
                    ))
                }

                "num" => {
                    let value = node.children.first()?.as_atom()?.parse().ok()?;
                    Some(SurfaceExpr::Lit(Span::dummy(), SurfaceLit::Nat(value)))
                }

                "str" => {
                    let value = node.children.first()?.as_atom()?.to_string();
                    Some(SurfaceExpr::Lit(Span::dummy(), SurfaceLit::String(value)))
                }

                "paren" => {
                    let inner = syntax_to_surface(node.children.first()?)?;
                    Some(SurfaceExpr::Paren(Span::dummy(), Box::new(inner)))
                }

                "hole" => Some(SurfaceExpr::Hole(Span::dummy())),

                "universe" => {
                    let tag = node.children.first()?.as_ident()?;
                    let univ = match tag {
                        "Prop" => UniverseExpr::Prop,
                        "Type" => UniverseExpr::Type,
                        "TypeLevel" => UniverseExpr::TypeLevel(Box::new(syntax_to_level(
                            node.children.get(1)?,
                        )?)),
                        "Sort" => {
                            UniverseExpr::Sort(Box::new(syntax_to_level(node.children.get(1)?)?))
                        }
                        "SortImplicit" => UniverseExpr::SortImplicit,
                        _ => return None,
                    };
                    Some(SurfaceExpr::Universe(Span::dummy(), univ))
                }

                "ifThenElse" => {
                    if node.children.len() != 3 {
                        return None;
                    }
                    let cond = syntax_to_surface(&node.children[0])?;
                    let then_br = syntax_to_surface(&node.children[1])?;
                    let else_br = syntax_to_surface(&node.children[2])?;
                    Some(SurfaceExpr::If(
                        Span::dummy(),
                        Box::new(cond),
                        Box::new(then_br),
                        Box::new(else_br),
                    ))
                }

                "ascription" => {
                    if node.children.len() != 2 {
                        return None;
                    }
                    let expr = syntax_to_surface(&node.children[0])?;
                    let ty = syntax_to_surface(&node.children[1])?;
                    Some(SurfaceExpr::Ascription(
                        Span::dummy(),
                        Box::new(expr),
                        Box::new(ty),
                    ))
                }

                "outParam" => {
                    let inner = syntax_to_surface(node.children.first()?)?;
                    Some(SurfaceExpr::OutParam(Span::dummy(), Box::new(inner)))
                }

                "semiOutParam" => {
                    let inner = syntax_to_surface(node.children.first()?)?;
                    Some(SurfaceExpr::SemiOutParam(Span::dummy(), Box::new(inner)))
                }

                "projection" => {
                    if node.children.len() != 2 {
                        return None;
                    }
                    let expr = syntax_to_surface(&node.children[0])?;
                    let proj = if let Some(name) = node.children[1].as_ident() {
                        Projection::Named(name.to_string())
                    } else if let Some(idx_str) = node.children[1].as_atom() {
                        Projection::Index(idx_str.parse().ok()?)
                    } else {
                        return None;
                    };
                    Some(SurfaceExpr::Proj(Span::dummy(), Box::new(expr), proj))
                }

                "match" => {
                    if node.children.is_empty() {
                        return None;
                    }
                    let scrutinee = syntax_to_surface(&node.children[0])?;
                    let arms: Option<Vec<_>> =
                        node.children[1..].iter().map(syntax_to_match_arm).collect();
                    Some(SurfaceExpr::Match(
                        Span::dummy(),
                        Box::new(scrutinee),
                        arms?,
                    ))
                }

                "ifLet" => {
                    // if let pat := scrutinee then then_br else else_br
                    if node.children.len() != 4 {
                        return None;
                    }
                    let pat = syntax_to_pattern(&node.children[0])?;
                    let scrutinee = syntax_to_surface(&node.children[1])?;
                    let then_br = syntax_to_surface(&node.children[2])?;
                    let else_br = syntax_to_surface(&node.children[3])?;
                    Some(SurfaceExpr::IfLet(
                        Span::dummy(),
                        pat,
                        Box::new(scrutinee),
                        Box::new(then_br),
                        Box::new(else_br),
                    ))
                }

                "ifDecidable" => {
                    // if h : p then t else e
                    if node.children.len() != 4 {
                        return None;
                    }
                    let witness_name = node.children[0].as_ident()?.to_string();
                    let prop = syntax_to_surface(&node.children[1])?;
                    let then_br = syntax_to_surface(&node.children[2])?;
                    let else_br = syntax_to_surface(&node.children[3])?;
                    Some(SurfaceExpr::IfDecidable(
                        Span::dummy(),
                        witness_name,
                        Box::new(prop),
                        Box::new(then_br),
                        Box::new(else_br),
                    ))
                }

                // Macros that expand to standard forms
                "showMacro" | "haveMacro" | "letMacro" | "do" => {
                    // These should have been expanded by the macro system
                    // If we see them here, return None to signal need for expansion
                    None
                }

                _ => {
                    // Unknown node type - try to convert as application if it has children
                    if node.children.is_empty() {
                        Some(SurfaceExpr::Ident(Span::dummy(), kind_name.to_string()))
                    } else {
                        let func = Syntax::ident(kind_name);
                        let func_expr = syntax_to_surface(&func)?;
                        let args: Option<Vec<_>> = node
                            .children
                            .iter()
                            .map(|c| syntax_to_surface(c).map(SurfaceArg::positional))
                            .collect();
                        Some(SurfaceExpr::App(Span::dummy(), Box::new(func_expr), args?))
                    }
                }
            }
        }
    }
}

/// Convert syntax to a match arm
fn syntax_to_match_arm(syntax: &Syntax) -> Option<SurfaceMatchArm> {
    match syntax {
        Syntax::Node(node) if node.kind.name_str() == "matchArm" => {
            if node.children.len() != 2 {
                return None;
            }
            let pattern = syntax_to_pattern(&node.children[0])?;
            let body = syntax_to_surface(&node.children[1])?;
            Some(SurfaceMatchArm {
                span: Span::dummy(),
                pattern,
                body,
            })
        }
        _ => None,
    }
}

/// Convert syntax to a pattern
fn syntax_to_pattern(syntax: &Syntax) -> Option<SurfacePattern> {
    match syntax {
        Syntax::Ident(_, name) if name == "_" => Some(SurfacePattern::Wildcard),
        Syntax::Ident(_, name) => Some(SurfacePattern::Var(name.clone())),
        Syntax::Node(node) => {
            let kind_name = node.kind.name_str();
            match kind_name {
                "ctorPattern" => {
                    if node.children.is_empty() {
                        return None;
                    }
                    let name = node.children[0].as_ident()?.to_string();
                    let args: Option<Vec<_>> =
                        node.children[1..].iter().map(syntax_to_pattern).collect();
                    Some(SurfacePattern::Ctor(name, args?))
                }
                "num" => {
                    let value = node.children.first()?.as_atom()?.parse().ok()?;
                    Some(SurfacePattern::Lit(SurfaceLit::Nat(value)))
                }
                "numeralAddPattern" => {
                    if node.children.len() != 2 {
                        return None;
                    }
                    let pat = syntax_to_pattern(&node.children[0])?;
                    let n = node.children[1]
                        .children()
                        .first()?
                        .as_atom()?
                        .parse()
                        .ok()?;
                    Some(SurfacePattern::NumeralAdd(Box::new(pat), n))
                }
                _ => {
                    // Try as constructor with arguments
                    if node.children.is_empty() {
                        Some(SurfacePattern::Var(kind_name.to_string()))
                    } else {
                        let args: Option<Vec<_>> =
                            node.children.iter().map(syntax_to_pattern).collect();
                        Some(SurfacePattern::Ctor(kind_name.to_string(), args?))
                    }
                }
            }
        }
        _ => None,
    }
}

/// Convert syntax to a binder
fn syntax_to_binder(syntax: &Syntax) -> Option<SurfaceBinder> {
    // Handle both simple identifiers and typed binders
    match syntax {
        Syntax::Ident(_, name) => Some(SurfaceBinder {
            span: Span::dummy(),
            name: name.clone(),
            ty: None,
            default: None,
            info: SurfaceBinderInfo::Explicit,
        }),

        Syntax::Node(node) => {
            let kind_name = node.kind.name_str();
            let info = match kind_name {
                "binderImplicit" => SurfaceBinderInfo::Implicit,
                "binderInstance" => SurfaceBinderInfo::Instance,
                "binderStrictImplicit" => SurfaceBinderInfo::StrictImplicit,
                _ => SurfaceBinderInfo::Explicit,
            };

            if node.children.is_empty() {
                return None;
            }

            let name = node.children[0].as_ident()?.to_string();
            let ty = if node.children.len() > 1 && !node.children[1].is_missing() {
                Some(Box::new(syntax_to_surface(&node.children[1])?))
            } else {
                None
            };

            Some(SurfaceBinder {
                span: Span::dummy(),
                name,
                ty,
                default: None,
                info,
            })
        }

        _ => None,
    }
}

// ============================================================================
// High-level expansion API
// ============================================================================

/// Expand macros in a surface expression
///
/// This converts the expression to macro syntax, expands macros,
/// and converts back to a surface expression for elaboration.
pub fn expand_surface_macros(
    ctx: &mut MacroCtx,
    expr: &SurfaceExpr,
) -> Result<SurfaceExpr, MacroExpansionError> {
    // Convert to macro syntax
    let syntax = surface_to_syntax(expr);

    // Expand macros
    let expanded = ctx
        .expand(syntax)
        .map_err(MacroExpansionError::MacroError)?;

    // Convert back to surface expression
    syntax_to_surface(&expanded).ok_or(MacroExpansionError::ConversionFailed)
}

/// Error from macro expansion
#[derive(Debug)]
pub enum MacroExpansionError {
    /// Macro expansion itself failed
    MacroError(lean5_macro::expand::MacroError),
    /// Could not convert expanded syntax back to surface expression
    ConversionFailed,
}

impl std::fmt::Display for MacroExpansionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacroExpansionError::MacroError(e) => write!(f, "macro error: {e}"),
            MacroExpansionError::ConversionFailed => {
                write!(f, "could not convert expanded syntax to surface expression")
            }
        }
    }
}

impl std::error::Error for MacroExpansionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use lean5_parser::{parse_decl, parse_expr, SurfaceDecl};

    #[test]
    fn test_macro_ctx_creation() {
        let ctx = MacroCtx::new();
        assert!(!ctx.registry().is_empty());
    }

    #[test]
    fn test_surface_to_syntax_ident() {
        let expr = SurfaceExpr::Ident(Span::dummy(), "foo".to_string());
        let syntax = surface_to_syntax(&expr);
        assert!(syntax.is_ident());
        assert_eq!(syntax.as_ident(), Some("foo"));
    }

    #[test]
    fn test_surface_to_syntax_lit_nat() {
        let expr = SurfaceExpr::Lit(Span::dummy(), SurfaceLit::Nat(42));
        let syntax = surface_to_syntax(&expr);
        assert!(syntax.is_node());
        assert_eq!(syntax.kind(), Some(&SyntaxKind::num()));
    }

    #[test]
    fn test_surface_to_syntax_app() {
        let func = SurfaceExpr::Ident(Span::dummy(), "f".to_string());
        let arg = SurfaceArg::positional(SurfaceExpr::Ident(Span::dummy(), "x".to_string()));
        let expr = SurfaceExpr::App(Span::dummy(), Box::new(func), vec![arg]);

        let syntax = surface_to_syntax(&expr);
        assert!(syntax.is_node());
        assert_eq!(syntax.kind(), Some(&SyntaxKind::app_kind()));
    }

    #[test]
    fn test_surface_to_syntax_if() {
        let cond = SurfaceExpr::Ident(Span::dummy(), "cond".to_string());
        let then_br = SurfaceExpr::Ident(Span::dummy(), "then".to_string());
        let else_br = SurfaceExpr::Ident(Span::dummy(), "else".to_string());
        let expr = SurfaceExpr::If(
            Span::dummy(),
            Box::new(cond),
            Box::new(then_br),
            Box::new(else_br),
        );

        let syntax = surface_to_syntax(&expr);
        assert!(syntax.is_node());
        assert_eq!(syntax.kind(), Some(&SyntaxKind::if_then_else()));
    }

    #[test]
    fn test_syntax_to_surface_ident() {
        let syntax = Syntax::ident("bar");
        let surface = syntax_to_surface(&syntax).unwrap();
        match surface {
            SurfaceExpr::Ident(_, name) => assert_eq!(name, "bar"),
            _ => panic!("expected ident"),
        }
    }

    #[test]
    fn test_syntax_to_surface_num() {
        let syntax = Syntax::mk_num(123);
        let surface = syntax_to_surface(&syntax).unwrap();
        match surface {
            SurfaceExpr::Lit(_, SurfaceLit::Nat(n)) => assert_eq!(n, 123),
            _ => panic!("expected nat literal"),
        }
    }

    #[test]
    fn test_syntax_to_surface_app() {
        let syntax = Syntax::mk_app(
            Syntax::ident("f"),
            vec![Syntax::ident("x"), Syntax::ident("y")],
        );
        let surface = syntax_to_surface(&syntax).unwrap();
        match surface {
            SurfaceExpr::App(_, func, args) => {
                match func.as_ref() {
                    SurfaceExpr::Ident(_, name) => assert_eq!(name, "f"),
                    _ => panic!("expected ident"),
                }
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected app"),
        }
    }

    #[test]
    fn test_roundtrip_simple() {
        let original = SurfaceExpr::Ident(Span::dummy(), "test".to_string());
        let syntax = surface_to_syntax(&original);
        let recovered = syntax_to_surface(&syntax).unwrap();

        match recovered {
            SurfaceExpr::Ident(_, name) => assert_eq!(name, "test"),
            _ => panic!("roundtrip failed"),
        }
    }

    #[test]
    fn test_roundtrip_app() {
        let func = SurfaceExpr::Ident(Span::dummy(), "add".to_string());
        let arg1 = SurfaceArg::positional(SurfaceExpr::Lit(Span::dummy(), SurfaceLit::Nat(1)));
        let arg2 = SurfaceArg::positional(SurfaceExpr::Lit(Span::dummy(), SurfaceLit::Nat(2)));
        let original = SurfaceExpr::App(Span::dummy(), Box::new(func), vec![arg1, arg2]);

        let syntax = surface_to_syntax(&original);
        let recovered = syntax_to_surface(&syntax).unwrap();

        match recovered {
            SurfaceExpr::App(_, func, args) => {
                match func.as_ref() {
                    SurfaceExpr::Ident(_, name) => assert_eq!(name, "add"),
                    _ => panic!("expected ident"),
                }
                assert_eq!(args.len(), 2);
            }
            _ => panic!("roundtrip failed"),
        }
    }

    #[test]
    fn test_if_expansion() {
        let mut ctx = MacroCtx::new();

        // Create if-then-else syntax
        let syntax = Syntax::node(
            SyntaxKind::if_then_else(),
            vec![
                Syntax::ident("condition"),
                Syntax::ident("thenBranch"),
                Syntax::ident("elseBranch"),
            ],
        );

        // Expand
        let expanded = ctx.expand(syntax).unwrap();

        // Should expand to ite application
        assert!(expanded.is_node());
        assert_eq!(expanded.kind(), Some(&SyntaxKind::app_kind()));

        // First child should be "ite"
        assert_eq!(expanded.child(0).unwrap().as_ident(), Some("ite"));
    }

    #[test]
    fn test_macro_stats() {
        let mut ctx = MacroCtx::new();

        let syntax = Syntax::node(
            SyntaxKind::if_then_else(),
            vec![Syntax::ident("c"), Syntax::ident("t"), Syntax::ident("e")],
        );

        let _ = ctx.expand(syntax).unwrap();

        let stats = ctx.last_stats().unwrap();
        assert!(stats.expansions > 0);
    }

    #[test]
    fn test_register_macro_rules_from_decl() {
        let decl = parse_decl("macro_rules | `(myMacro $x) => `(id $x)").unwrap();
        let arms = match decl {
            SurfaceDecl::MacroRules { arms, .. } => arms,
            other => panic!("unexpected decl {other:?}"),
        };

        let mut ctx = MacroCtx::new();
        ctx.register_macro_rules(None, &arms).unwrap();

        let expr = parse_expr("myMacro foo").unwrap();
        let syntax = surface_to_syntax(&expr);
        assert!(!ctx
            .registry()
            .get_by_kind(&SyntaxKind::app_kind())
            .is_empty());

        let direct = ctx
            .registry()
            .try_expand(&syntax)
            .expect("macro should apply");
        assert_eq!(direct.kind(), Some(&SyntaxKind::app_kind()));

        let expanded = ctx.expand(syntax).unwrap();
        let stats = ctx.last_stats().unwrap();
        assert!(stats.expansions > 0);

        assert_eq!(expanded.kind(), Some(&SyntaxKind::app_kind()));
        assert_eq!(expanded.child(0).unwrap().as_ident(), Some("id"));
    }

    #[test]
    fn test_register_syntax_category() {
        let mut ctx = MacroCtx::new();

        // Built-in categories should exist
        assert!(ctx.has_syntax_category("term"));
        assert!(ctx.has_syntax_category("tactic"));

        // Custom category should not exist yet
        assert!(!ctx.has_syntax_category("mycat"));

        // Register custom category
        ctx.register_syntax_category("mycat");
        assert!(ctx.has_syntax_category("mycat"));
    }

    #[test]
    fn test_register_syntax_declaration() {
        let mut ctx = MacroCtx::new();

        // Register a simple syntax declaration: syntax "mykey" x:term : term
        let pattern = vec![
            SyntaxPatternItem::Literal("mykey".to_string()),
            SyntaxPatternItem::Variable {
                name: "x".to_string(),
                category: Some("term".to_string()),
            },
        ];

        ctx.register_syntax(Some("mykey_syntax"), Some(50), &pattern, "term")
            .unwrap();

        // Check that the macro was registered
        assert!(ctx.registry().get_by_name("mykey_syntax").is_some());
    }

    #[test]
    fn test_register_notation_infixl() {
        let mut ctx = MacroCtx::new();

        // Register: infixl:65 " +++ " => myAdd
        let pattern = vec![
            NotationItem::Variable("a".to_string()),
            NotationItem::Literal(" +++ ".to_string()),
            NotationItem::Variable("b".to_string()),
        ];
        let expansion = SurfaceExpr::Ident(Span::dummy(), "myAdd".to_string());

        ctx.register_notation(NotationKind::Infixl, Some(65), &pattern, &expansion)
            .unwrap();

        // Check macro was registered
        let registered_names = ctx.registry().macro_names();
        assert!(registered_names
            .iter()
            .any(|n| n.contains("infixl") && n.contains("+++")));
    }

    #[test]
    fn test_register_notation_prefix() {
        let mut ctx = MacroCtx::new();

        // Register: prefix:max "!!!" => myNot
        let pattern = vec![
            NotationItem::Literal("!!!".to_string()),
            NotationItem::Variable("x".to_string()),
        ];
        let expansion = SurfaceExpr::Ident(Span::dummy(), "myNot".to_string());

        ctx.register_notation(NotationKind::Prefix, Some(1024), &pattern, &expansion)
            .unwrap();

        // Check macro was registered
        let registered_names = ctx.registry().macro_names();
        assert!(registered_names
            .iter()
            .any(|n| n.contains("prefix") && n.contains("!!!")));
    }

    #[test]
    fn test_register_macro_declaration() {
        let mut ctx = MacroCtx::new();

        // Register: macro "unless" cond:term "then" body:term : term => `(if !$cond then $body else ())
        let pattern = vec![
            SyntaxPatternItem::Literal("unless".to_string()),
            SyntaxPatternItem::Variable {
                name: "cond".to_string(),
                category: Some("term".to_string()),
            },
            SyntaxPatternItem::Literal("then".to_string()),
            SyntaxPatternItem::Variable {
                name: "body".to_string(),
                category: Some("term".to_string()),
            },
        ];
        let expansion = SurfaceExpr::Ident(Span::dummy(), "expanded_unless".to_string());

        ctx.register_macro(&pattern, "term", &expansion).unwrap();

        // Check macro was registered
        let registered_names = ctx.registry().macro_names();
        assert!(registered_names.iter().any(|n| n.contains("unless")));
    }

    #[test]
    fn test_syntax_pattern_to_syntax_simple() {
        let pattern = vec![
            SyntaxPatternItem::Literal("if".to_string()),
            SyntaxPatternItem::Variable {
                name: "cond".to_string(),
                category: Some("term".to_string()),
            },
            SyntaxPatternItem::Literal("then".to_string()),
            SyntaxPatternItem::Variable {
                name: "body".to_string(),
                category: None,
            },
        ];

        let syntax = syntax_pattern_to_syntax(&pattern);
        assert!(syntax.is_node());
        assert_eq!(syntax.kind(), Some(&SyntaxKind::app("seq")));
        assert_eq!(syntax.children().len(), 4);
    }

    #[test]
    fn test_notation_pattern_to_syntax() {
        let items = vec![
            NotationItem::Variable("a".to_string()),
            NotationItem::Literal("+".to_string()),
            NotationItem::Variable("b".to_string()),
        ];

        let (syntax, kind, vars) = notation_pattern_to_syntax(NotationKind::Infixl, &items);

        assert!(syntax.is_node());
        assert_eq!(kind, SyntaxKind::app("+"));
        assert_eq!(vars, vec!["a", "b"]);
    }

    #[test]
    fn test_pattern_to_name() {
        let pattern = vec![
            SyntaxPatternItem::Literal("unless".to_string()),
            SyntaxPatternItem::Variable {
                name: "cond".to_string(),
                category: None,
            },
        ];

        let name = pattern_to_name(&pattern);
        assert_eq!(name, "unless_cond");
    }

    #[test]
    fn test_notation_to_name() {
        let items = vec![
            NotationItem::Variable("a".to_string()),
            NotationItem::Literal(" + ".to_string()),
            NotationItem::Variable("b".to_string()),
        ];

        let name = notation_to_name(NotationKind::Infixl, &items);
        assert!(name.starts_with("infixl_"));
        assert!(name.contains('+'));
    }
}
