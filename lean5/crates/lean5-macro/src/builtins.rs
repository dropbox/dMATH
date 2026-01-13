//! Built-in macro definitions
//!
//! This module provides macro definitions for common Lean 4 syntax patterns.
//! These are the built-in macros that are always available.
//!
//! # Macros implemented:
//! - `if-then-else`: Desugars to `ite` function application
//! - `if let`: Pattern matching with optional binding
//! - `do` notation: Monadic sequencing
//! - `unless`: Negated conditional
//! - `when`: Conditional unit action
//! - `dbg_trace`: Debug tracing macro
//! - `assert!`: Assertion macro
//! - `panic!`: Panic macro

use crate::quotation::SyntaxQuote;
use crate::registry::{MacroDef, MacroRegistry};
use crate::syntax::{Syntax, SyntaxKind};

/// Built-in syntax kinds
impl SyntaxKind {
    pub fn if_then_else() -> Self {
        Self::app("ifThenElse")
    }
    pub fn if_let() -> Self {
        Self::app("ifLet")
    }
    pub fn do_notation() -> Self {
        Self::app("do")
    }
    pub fn do_elem() -> Self {
        Self::app("doElem")
    }
    pub fn do_let() -> Self {
        Self::app("doLet")
    }
    pub fn do_bind() -> Self {
        Self::app("doBind")
    }
    pub fn do_return() -> Self {
        Self::app("doReturn")
    }
    pub fn unless() -> Self {
        Self::app("unless")
    }
    pub fn when() -> Self {
        Self::app("when")
    }
    pub fn dbg_trace() -> Self {
        Self::app("dbgTrace")
    }
    pub fn assert_macro() -> Self {
        Self::app("assert")
    }
    pub fn panic_macro() -> Self {
        Self::app("panic")
    }
    pub fn have_macro() -> Self {
        Self::app("haveMacro")
    }
    pub fn let_macro() -> Self {
        Self::app("letMacro")
    }
    pub fn show_macro() -> Self {
        Self::app("showMacro")
    }
    pub fn match_expr() -> Self {
        Self::app("match")
    }
    pub fn match_arm() -> Self {
        Self::app("matchArm")
    }
    pub fn calc_macro() -> Self {
        Self::app("calc")
    }
    pub fn conv_macro() -> Self {
        Self::app("conv")
    }
}

/// Register all built-in macros into a registry
pub fn register_builtins(registry: &mut MacroRegistry) {
    register_conditionals(registry);
    register_do_notation(registry);
    register_assertions(registry);
    register_let_have_show(registry);
    register_match(registry);
    register_calc_conv(registry);
}

/// Create a registry with all built-in macros pre-registered
pub fn builtin_registry() -> MacroRegistry {
    let mut registry = MacroRegistry::new();
    register_builtins(&mut registry);
    registry
}

// ============================================================================
// Conditionals: if-then-else, unless, when
// ============================================================================

fn register_conditionals(registry: &mut MacroRegistry) {
    // if cond then t else f => ite cond t f
    //
    // Pattern: (ifThenElse $cond $then $else)
    // Result: (app ite $cond $then $else)
    let if_def = MacroDef::new(
        "ifThenElse",
        SyntaxKind::if_then_else(),
        Syntax::node(
            SyntaxKind::if_then_else(),
            vec![
                Syntax::mk_antiquot("cond"),
                Syntax::mk_antiquot("thenBranch"),
                Syntax::mk_antiquot("elseBranch"),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::ident("ite"),
            vec![
                Syntax::mk_antiquot("cond"),
                Syntax::mk_antiquot("thenBranch"),
                Syntax::mk_antiquot("elseBranch"),
            ],
        )),
    )
    .with_doc("if-then-else desugars to ite application");
    registry.register(if_def);

    // unless cond body => if cond then () else body
    //
    // Pattern: (unless $cond $body)
    // Result: (ifThenElse (not $cond) $body (unit))
    let unless_def = MacroDef::new(
        "unless",
        SyntaxKind::unless(),
        Syntax::node(
            SyntaxKind::unless(),
            vec![Syntax::mk_antiquot("cond"), Syntax::mk_antiquot("body")],
        ),
        SyntaxQuote::term(Syntax::node(
            SyntaxKind::if_then_else(),
            vec![
                Syntax::mk_app(Syntax::ident("Not"), vec![Syntax::mk_antiquot("cond")]),
                Syntax::mk_antiquot("body"),
                Syntax::ident("Unit.unit"),
            ],
        )),
    )
    .with_doc("unless cond body => if (Not cond) then body else ()");
    registry.register(unless_def);

    // when cond body => if cond then body else ()
    //
    // Pattern: (when $cond $body)
    // Result: (ifThenElse $cond $body (unit))
    let when_def = MacroDef::new(
        "when",
        SyntaxKind::when(),
        Syntax::node(
            SyntaxKind::when(),
            vec![Syntax::mk_antiquot("cond"), Syntax::mk_antiquot("body")],
        ),
        SyntaxQuote::term(Syntax::node(
            SyntaxKind::if_then_else(),
            vec![
                Syntax::mk_antiquot("cond"),
                Syntax::mk_antiquot("body"),
                Syntax::ident("Unit.unit"),
            ],
        )),
    )
    .with_doc("when cond body => if cond then body else ()");
    registry.register(when_def);
}

// ============================================================================
// Do notation
// ============================================================================

fn register_do_notation(registry: &mut MacroRegistry) {
    // Simple do with single expression: do e => e
    //
    // Pattern: (do (doElem $e))
    // Result: $e
    let do_single_def = MacroDef::new(
        "doSingle",
        SyntaxKind::do_notation(),
        Syntax::node(
            SyntaxKind::do_notation(),
            vec![Syntax::node(
                SyntaxKind::do_elem(),
                vec![Syntax::mk_antiquot("e")],
            )],
        ),
        SyntaxQuote::term(Syntax::mk_antiquot("e")),
    )
    .with_priority(0)
    .with_doc("do e => e (single element do block)");
    registry.register(do_single_def);

    // do { x <- e1; rest } => e1 >>= fun x => do { rest }
    //
    // Pattern: (do (doBind $x $e1) $rest...)
    // This is complex - we'd need splice support for multiple elements.
    // For now, handle the two-element case:
    // Pattern: (do (doBind $x $e1) (doElem $e2))
    // Result: (app (app bind $e1) (fun $x $e2))
    let do_bind_def = MacroDef::new(
        "doBind",
        SyntaxKind::do_notation(),
        Syntax::node(
            SyntaxKind::do_notation(),
            vec![
                Syntax::node(
                    SyntaxKind::do_bind(),
                    vec![Syntax::mk_antiquot("x"), Syntax::mk_antiquot("e1")],
                ),
                Syntax::node(SyntaxKind::do_elem(), vec![Syntax::mk_antiquot("e2")]),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_app(Syntax::ident("bind"), vec![Syntax::mk_antiquot("e1")]),
            vec![Syntax::mk_lambda(
                vec![Syntax::mk_antiquot("x")],
                Syntax::mk_antiquot("e2"),
            )],
        )),
    )
    .with_priority(10)
    .with_doc("do { x <- e1; e2 } => bind e1 (fun x => e2)");
    registry.register(do_bind_def);

    // do { let x := e1; rest } => let x := e1 in do { rest }
    //
    // Pattern: (do (doLet $x $e1) (doElem $e2))
    // Result: (let $x $e1 $e2)
    let do_let_def = MacroDef::new(
        "doLet",
        SyntaxKind::do_notation(),
        Syntax::node(
            SyntaxKind::do_notation(),
            vec![
                Syntax::node(
                    SyntaxKind::do_let(),
                    vec![Syntax::mk_antiquot("x"), Syntax::mk_antiquot("e1")],
                ),
                Syntax::node(SyntaxKind::do_elem(), vec![Syntax::mk_antiquot("e2")]),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_let(
            Syntax::mk_antiquot("x"),
            None,
            Syntax::mk_antiquot("e1"),
            Syntax::mk_antiquot("e2"),
        )),
    )
    .with_priority(10)
    .with_doc("do { let x := e1; e2 } => let x := e1 in e2");
    registry.register(do_let_def);

    // do { e1; e2 } => e1 >>= fun _ => e2 (or e1 >> e2)
    //
    // When e1 doesn't bind a result, we use seq
    // Pattern: (do (doElem $e1) (doElem $e2))
    // Result: (app (app seq $e1) (fun _ => $e2))
    let do_seq_def = MacroDef::new(
        "doSeq",
        SyntaxKind::do_notation(),
        Syntax::node(
            SyntaxKind::do_notation(),
            vec![
                Syntax::node(SyntaxKind::do_elem(), vec![Syntax::mk_antiquot("e1")]),
                Syntax::node(SyntaxKind::do_elem(), vec![Syntax::mk_antiquot("e2")]),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_app(Syntax::ident("seq"), vec![Syntax::mk_antiquot("e1")]),
            vec![Syntax::mk_lambda(
                vec![Syntax::ident("_")],
                Syntax::mk_antiquot("e2"),
            )],
        )),
    )
    .with_priority(5)
    .with_doc("do { e1; e2 } => seq e1 (fun _ => e2)");
    registry.register(do_seq_def);

    // do { return e } => pure e
    //
    // Pattern: (do (doReturn $e))
    // Result: (app pure $e)
    let do_return_def = MacroDef::new(
        "doReturn",
        SyntaxKind::do_notation(),
        Syntax::node(
            SyntaxKind::do_notation(),
            vec![Syntax::node(
                SyntaxKind::do_return(),
                vec![Syntax::mk_antiquot("e")],
            )],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::ident("pure"),
            vec![Syntax::mk_antiquot("e")],
        )),
    )
    .with_priority(10)
    .with_doc("do { return e } => pure e");
    registry.register(do_return_def);
}

// ============================================================================
// Assertions and debugging
// ============================================================================

fn register_assertions(registry: &mut MacroRegistry) {
    // assert! cond => if cond then () else panic "assertion failed"
    //
    // Pattern: (assert $cond)
    // Result: (ifThenElse $cond (unit) (panic "assertion failed"))
    let assert_def = MacroDef::new(
        "assert",
        SyntaxKind::assert_macro(),
        Syntax::node(
            SyntaxKind::assert_macro(),
            vec![Syntax::mk_antiquot("cond")],
        ),
        SyntaxQuote::term(Syntax::node(
            SyntaxKind::if_then_else(),
            vec![
                Syntax::mk_antiquot("cond"),
                Syntax::ident("Unit.unit"),
                Syntax::mk_app(
                    Syntax::ident("panic"),
                    vec![Syntax::mk_str("assertion failed")],
                ),
            ],
        )),
    )
    .with_doc("assert! cond => runtime assertion");
    registry.register(assert_def);

    // panic! msg => panic msg
    //
    // Pattern: (panic $msg)
    // Result: (app panic $msg)
    let panic_def = MacroDef::new(
        "panic",
        SyntaxKind::panic_macro(),
        Syntax::node(SyntaxKind::panic_macro(), vec![Syntax::mk_antiquot("msg")]),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::ident("panic"),
            vec![Syntax::mk_antiquot("msg")],
        )),
    )
    .with_doc("panic! msg => panic with message");
    registry.register(panic_def);

    // dbg_trace msg body => dbgTrace msg (fun () => body)
    //
    // Pattern: (dbgTrace $msg $body)
    // Result: (app (app dbgTrace $msg) (fun () => $body))
    let dbg_trace_def = MacroDef::new(
        "dbgTrace",
        SyntaxKind::dbg_trace(),
        Syntax::node(
            SyntaxKind::dbg_trace(),
            vec![Syntax::mk_antiquot("msg"), Syntax::mk_antiquot("body")],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_app(Syntax::ident("dbgTrace"), vec![Syntax::mk_antiquot("msg")]),
            vec![Syntax::mk_lambda(
                vec![Syntax::ident("_")],
                Syntax::mk_antiquot("body"),
            )],
        )),
    )
    .with_doc("dbg_trace msg body => print msg then evaluate body");
    registry.register(dbg_trace_def);
}

// ============================================================================
// let, have, show macros
// ============================================================================

fn register_let_have_show(registry: &mut MacroRegistry) {
    // have h : T := proof; body => (fun (h : T) => body) proof
    //
    // Pattern: (haveMacro $h $ty $proof $body)
    // Result: (app (fun ($h : $ty) => $body) $proof)
    let have_def = MacroDef::new(
        "have",
        SyntaxKind::have_macro(),
        Syntax::node(
            SyntaxKind::have_macro(),
            vec![
                Syntax::mk_antiquot("h"),
                Syntax::mk_antiquot("ty"),
                Syntax::mk_antiquot("proof"),
                Syntax::mk_antiquot("body"),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_lambda(
                vec![Syntax::node(
                    SyntaxKind::app("binder"),
                    vec![Syntax::mk_antiquot("h"), Syntax::mk_antiquot("ty")],
                )],
                Syntax::mk_antiquot("body"),
            ),
            vec![Syntax::mk_antiquot("proof")],
        )),
    )
    .with_doc("have h : T := proof; body => (fun (h : T) => body) proof");
    registry.register(have_def);

    // let x := e; body => (fun x => body) e
    // Note: This is for the macro form, not the kernel let expression
    //
    // Pattern: (letMacro $x $e $body)
    // Result: (app (fun $x => $body) $e)
    let let_macro_def = MacroDef::new(
        "letMacro",
        SyntaxKind::let_macro(),
        Syntax::node(
            SyntaxKind::let_macro(),
            vec![
                Syntax::mk_antiquot("x"),
                Syntax::mk_antiquot("e"),
                Syntax::mk_antiquot("body"),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_lambda(vec![Syntax::mk_antiquot("x")], Syntax::mk_antiquot("body")),
            vec![Syntax::mk_antiquot("e")],
        )),
    )
    .with_doc("let x := e; body => (fun x => body) e");
    registry.register(let_macro_def);

    // show T from e => (e : T)
    // Used to annotate the expected type of an expression
    //
    // Pattern: (showMacro $ty $e)
    // Result: (app (app Eq.subst Eq.refl) (app (id $ty) $e))
    // Actually simpler: Result: (the $ty $e) which is a type annotation
    let show_def = MacroDef::new(
        "show",
        SyntaxKind::show_macro(),
        Syntax::node(
            SyntaxKind::show_macro(),
            vec![Syntax::mk_antiquot("ty"), Syntax::mk_antiquot("e")],
        ),
        // show T from e => @id T e (type annotation)
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_app(Syntax::ident("@id"), vec![Syntax::mk_antiquot("ty")]),
            vec![Syntax::mk_antiquot("e")],
        )),
    )
    .with_doc("show T from e => type annotation");
    registry.register(show_def);
}

// ============================================================================
// Pattern matching
// ============================================================================

// ============================================================================
// Calc and Conv macros
// ============================================================================

/// Additional syntax kinds for calc and conv
impl SyntaxKind {
    pub fn calc_step() -> Self {
        Self::app("calcStep")
    }
    pub fn conv_step() -> Self {
        Self::app("convStep")
    }
    pub fn conv_seq() -> Self {
        Self::app("convSeq")
    }
}

fn register_calc_conv(registry: &mut MacroRegistry) {
    // calc: Calculational proof style
    //
    // calc a op1 b := h1
    //      _ op2 c := h2
    //      _ op3 d := h3
    //
    // Expands to: Trans.trans (Trans.trans h1 h2) h3
    //
    // Two-step calc: calc a = b := h1; _ = c := h2 => Trans.trans h1 h2
    // Pattern: (calc (calcStep $h1) (calcStep $h2))
    // Result: (app (app Trans.trans $h1) $h2)
    let calc_two_step = MacroDef::new(
        "calcTwoStep",
        SyntaxKind::calc_macro(),
        Syntax::node(
            SyntaxKind::calc_macro(),
            vec![
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::mk_antiquot("h1")]),
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::mk_antiquot("h2")]),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_app(
                Syntax::ident("Trans.trans"),
                vec![Syntax::mk_antiquot("h1")],
            ),
            vec![Syntax::mk_antiquot("h2")],
        )),
    )
    .with_priority(10)
    .with_doc("calc a = b := h1; _ = c := h2 => Trans.trans h1 h2");
    registry.register(calc_two_step);

    // Single-step calc: calc a = b := h => h
    // Pattern: (calc (calcStep $h))
    // Result: $h
    let calc_single_step = MacroDef::new(
        "calcSingleStep",
        SyntaxKind::calc_macro(),
        Syntax::node(
            SyntaxKind::calc_macro(),
            vec![Syntax::node(
                SyntaxKind::calc_step(),
                vec![Syntax::mk_antiquot("h")],
            )],
        ),
        SyntaxQuote::term(Syntax::mk_antiquot("h")),
    )
    .with_priority(0)
    .with_doc("calc a = b := h => h (single step)");
    registry.register(calc_single_step);

    // Three-step calc: calc ... := h1; _ ... := h2; _ ... := h3 => Trans.trans (Trans.trans h1 h2) h3
    let calc_three_step = MacroDef::new(
        "calcThreeStep",
        SyntaxKind::calc_macro(),
        Syntax::node(
            SyntaxKind::calc_macro(),
            vec![
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::mk_antiquot("h1")]),
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::mk_antiquot("h2")]),
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::mk_antiquot("h3")]),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_app(
                Syntax::ident("Trans.trans"),
                vec![Syntax::mk_app(
                    Syntax::mk_app(
                        Syntax::ident("Trans.trans"),
                        vec![Syntax::mk_antiquot("h1")],
                    ),
                    vec![Syntax::mk_antiquot("h2")],
                )],
            ),
            vec![Syntax::mk_antiquot("h3")],
        )),
    )
    .with_priority(15)
    .with_doc("calc with three steps");
    registry.register(calc_three_step);

    // conv: Conversion tactic mode
    //
    // conv => tactic
    //
    // Allows targeted rewriting in goal or hypotheses.
    // For now, provide a basic desugaring.
    //
    // conv at h => tactics => Lean.Parser.Tactic.Conv.conv h tactics
    //
    // Simple conv: conv => t => convTactic t
    let conv_simple = MacroDef::new(
        "convSimple",
        SyntaxKind::conv_macro(),
        Syntax::node(
            SyntaxKind::conv_macro(),
            vec![Syntax::mk_antiquot("tactic")],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::ident("convTactic"),
            vec![Syntax::mk_antiquot("tactic")],
        )),
    )
    .with_priority(0)
    .with_doc("conv => tactic => convTactic tactic");
    registry.register(conv_simple);

    // conv sequence: conv => t1; t2; t3 => convSeq [t1, t2, t3]
    let conv_seq = MacroDef::new(
        "convSeq",
        SyntaxKind::conv_macro(),
        Syntax::node(
            SyntaxKind::conv_macro(),
            vec![Syntax::node(
                SyntaxKind::conv_seq(),
                vec![Syntax::mk_antiquot("t1"), Syntax::mk_antiquot("t2")],
            )],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::ident("convSeq"),
            vec![Syntax::mk_antiquot("t1"), Syntax::mk_antiquot("t2")],
        )),
    )
    .with_priority(5)
    .with_doc("conv sequence");
    registry.register(conv_seq);

    // conv at hypothesis: conv at h => t => convAt h t
    // Pattern: (conv "at" $h $tactic)
    // We represent this with a special node structure
    let conv_at = MacroDef::new(
        "convAt",
        SyntaxKind::conv_macro(),
        Syntax::node(
            SyntaxKind::conv_macro(),
            vec![
                Syntax::ident("at"),
                Syntax::mk_antiquot("h"),
                Syntax::mk_antiquot("tactic"),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_app(Syntax::ident("convAt"), vec![Syntax::mk_antiquot("h")]),
            vec![Syntax::mk_antiquot("tactic")],
        )),
    )
    .with_priority(10)
    .with_doc("conv at h => t => targeted conversion");
    registry.register(conv_at);
}

// ============================================================================
// Pattern matching
// ============================================================================

fn register_match(registry: &mut MacroRegistry) {
    // match e with | p1 => b1 | p2 => b2 ...
    //
    // This is handled by the elaborator directly since it involves
    // pattern compilation. The macro system just provides syntax recognition.
    //
    // For now we register a placeholder that passes through to elaboration.
    // A full implementation would lower match expressions to case trees.

    // Single-arm match: match e with | p => b => (fun p => b) e
    //
    // Pattern: (match $scrutinee (matchArm $pat $body))
    // Result: (app (fun $pat => $body) $scrutinee)
    let match_single_def = MacroDef::new(
        "matchSingle",
        SyntaxKind::match_expr(),
        Syntax::node(
            SyntaxKind::match_expr(),
            vec![
                Syntax::mk_antiquot("scrutinee"),
                Syntax::node(
                    SyntaxKind::match_arm(),
                    vec![Syntax::mk_antiquot("pat"), Syntax::mk_antiquot("body")],
                ),
            ],
        ),
        SyntaxQuote::term(Syntax::mk_app(
            Syntax::mk_lambda(
                vec![Syntax::mk_antiquot("pat")],
                Syntax::mk_antiquot("body"),
            ),
            vec![Syntax::mk_antiquot("scrutinee")],
        )),
    )
    .with_priority(0)
    .with_doc("match e with | p => b => (fun p => b) e");
    registry.register(match_single_def);

    // if let: if let p := e then b1 else b2
    // Desugars to match e with | p => b1 | _ => b2
    //
    // Pattern: (ifLet $pat $scrutinee $thenBranch $elseBranch)
    // Result: (match $scrutinee (matchArm $pat $thenBranch) (matchArm _ $elseBranch))
    //
    // But since we only handle single-arm match, we need different approach:
    // if let Some x := e then b1 else b2 => Option.casesOn e (fun () => b2) (fun x => b1)
    // This is type-specific. For now, lower to match.
    let if_let_def = MacroDef::new(
        "ifLet",
        SyntaxKind::if_let(),
        Syntax::node(
            SyntaxKind::if_let(),
            vec![
                Syntax::mk_antiquot("pat"),
                Syntax::mk_antiquot("scrutinee"),
                Syntax::mk_antiquot("thenBranch"),
                Syntax::mk_antiquot("elseBranch"),
            ],
        ),
        // For now, create a match node that elaboration will handle
        SyntaxQuote::term(Syntax::node(
            SyntaxKind::match_expr(),
            vec![
                Syntax::mk_antiquot("scrutinee"),
                Syntax::node(
                    SyntaxKind::match_arm(),
                    vec![
                        Syntax::mk_antiquot("pat"),
                        Syntax::mk_antiquot("thenBranch"),
                    ],
                ),
                Syntax::node(
                    SyntaxKind::match_arm(),
                    vec![Syntax::ident("_"), Syntax::mk_antiquot("elseBranch")],
                ),
            ],
        )),
    )
    .with_doc("if let p := e then b1 else b2 => pattern matching conditional");
    registry.register(if_let_def);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expand::expand;

    #[test]
    fn test_builtin_registry_creation() {
        let registry = builtin_registry();
        assert!(!registry.is_empty());
        assert!(registry.get_by_name("ifThenElse").is_some());
        assert!(registry.get_by_name("unless").is_some());
        assert!(registry.get_by_name("when").is_some());
    }

    #[test]
    fn test_if_then_else_expansion() {
        let registry = builtin_registry();

        // Create: if cond then thenBranch else elseBranch
        let input = Syntax::node(
            SyntaxKind::if_then_else(),
            vec![
                Syntax::ident("cond"),
                Syntax::ident("thenBranch"),
                Syntax::ident("elseBranch"),
            ],
        );

        let result = expand(&registry, input).unwrap();

        // Should expand to (app ite cond thenBranch elseBranch)
        assert!(result.is_node());
        assert_eq!(result.kind(), Some(&SyntaxKind::app_kind()));
        assert_eq!(result.child(0).unwrap().as_ident(), Some("ite"));
    }

    #[test]
    fn test_unless_expansion() {
        let registry = builtin_registry();

        let input = Syntax::node(
            SyntaxKind::unless(),
            vec![Syntax::ident("cond"), Syntax::ident("body")],
        );

        let result = expand(&registry, input).unwrap();

        // unless expands to if (Not cond) then body else ()
        // which then expands to ite (Not cond) body Unit.unit
        let pretty = result.pretty();
        assert!(pretty.contains("ite"));
        assert!(pretty.contains("Not"));
    }

    #[test]
    fn test_when_expansion() {
        let registry = builtin_registry();

        let input = Syntax::node(
            SyntaxKind::when(),
            vec![Syntax::ident("cond"), Syntax::ident("body")],
        );

        let result = expand(&registry, input).unwrap();

        // when expands to if cond then body else ()
        // which expands to ite cond body Unit.unit
        let pretty = result.pretty();
        assert!(pretty.contains("ite"));
        assert!(pretty.contains("cond"));
    }

    #[test]
    fn test_do_single_expansion() {
        let registry = builtin_registry();

        // do { e } => e
        let input = Syntax::node(
            SyntaxKind::do_notation(),
            vec![Syntax::node(
                SyntaxKind::do_elem(),
                vec![Syntax::ident("expr")],
            )],
        );

        let result = expand(&registry, input).unwrap();
        assert_eq!(result.as_ident(), Some("expr"));
    }

    #[test]
    fn test_do_bind_expansion() {
        let registry = builtin_registry();

        // do { x <- e1; e2 } => bind e1 (fun x => e2)
        let input = Syntax::node(
            SyntaxKind::do_notation(),
            vec![
                Syntax::node(
                    SyntaxKind::do_bind(),
                    vec![Syntax::ident("x"), Syntax::ident("getLine")],
                ),
                Syntax::node(
                    SyntaxKind::do_elem(),
                    vec![Syntax::mk_app(
                        Syntax::ident("putStrLn"),
                        vec![Syntax::ident("x")],
                    )],
                ),
            ],
        );

        let result = expand(&registry, input).unwrap();

        // Should contain bind and fun
        let pretty = result.pretty();
        assert!(pretty.contains("bind"));
        assert!(pretty.contains("fun"));
    }

    #[test]
    fn test_do_return_expansion() {
        let registry = builtin_registry();

        // do { return e } => pure e
        let input = Syntax::node(
            SyntaxKind::do_notation(),
            vec![Syntax::node(
                SyntaxKind::do_return(),
                vec![Syntax::mk_num(42)],
            )],
        );

        let result = expand(&registry, input).unwrap();

        let pretty = result.pretty();
        assert!(pretty.contains("pure"));
    }

    #[test]
    fn test_do_seq_expansion() {
        let registry = builtin_registry();

        // do { e1; e2 } => seq e1 (fun _ => e2)
        let input = Syntax::node(
            SyntaxKind::do_notation(),
            vec![
                Syntax::node(SyntaxKind::do_elem(), vec![Syntax::ident("action1")]),
                Syntax::node(SyntaxKind::do_elem(), vec![Syntax::ident("action2")]),
            ],
        );

        let result = expand(&registry, input).unwrap();

        let pretty = result.pretty();
        assert!(pretty.contains("seq"));
        assert!(pretty.contains("fun"));
    }

    #[test]
    fn test_assert_expansion() {
        let registry = builtin_registry();

        let input = Syntax::node(SyntaxKind::assert_macro(), vec![Syntax::ident("condition")]);

        let result = expand(&registry, input).unwrap();

        // Should expand to if condition then () else panic
        let pretty = result.pretty();
        assert!(pretty.contains("ite"));
        assert!(pretty.contains("panic"));
    }

    #[test]
    fn test_dbg_trace_expansion() {
        let registry = builtin_registry();

        let input = Syntax::node(
            SyntaxKind::dbg_trace(),
            vec![Syntax::mk_str("debug message"), Syntax::ident("result")],
        );

        let result = expand(&registry, input).unwrap();

        let pretty = result.pretty();
        assert!(pretty.contains("dbgTrace"));
        assert!(pretty.contains("fun"));
    }

    #[test]
    fn test_have_expansion() {
        let registry = builtin_registry();

        // have h : T := proof; body => (fun (h : T) => body) proof
        let input = Syntax::node(
            SyntaxKind::have_macro(),
            vec![
                Syntax::ident("h"),
                Syntax::ident("Nat"),
                Syntax::mk_num(42),
                Syntax::ident("body"),
            ],
        );

        let result = expand(&registry, input).unwrap();

        // Should expand to application with lambda
        assert!(result.is_node());
        assert_eq!(result.kind(), Some(&SyntaxKind::app_kind()));
    }

    #[test]
    fn test_match_single_expansion() {
        let registry = builtin_registry();

        // match e with | p => b => (fun p => b) e
        let input = Syntax::node(
            SyntaxKind::match_expr(),
            vec![
                Syntax::ident("scrutinee"),
                Syntax::node(
                    SyntaxKind::match_arm(),
                    vec![Syntax::ident("x"), Syntax::ident("body")],
                ),
            ],
        );

        let result = expand(&registry, input).unwrap();

        // Should expand to (app (fun x => body) scrutinee)
        assert!(result.is_node());
        assert_eq!(result.kind(), Some(&SyntaxKind::app_kind()));
    }

    #[test]
    fn test_if_let_expansion() {
        let registry = builtin_registry();

        // if let p := e then b1 else b2 => match e with ...
        let input = Syntax::node(
            SyntaxKind::if_let(),
            vec![
                Syntax::ident("Some x"),
                Syntax::ident("maybeValue"),
                Syntax::ident("thenBranch"),
                Syntax::ident("elseBranch"),
            ],
        );

        let result = expand(&registry, input).unwrap();

        // Should expand to match expression
        assert!(result.is_node());
        assert_eq!(result.kind(), Some(&SyntaxKind::match_expr()));
    }

    #[test]
    fn test_show_expansion() {
        let registry = builtin_registry();

        // show T from e => @id T e
        let input = Syntax::node(
            SyntaxKind::show_macro(),
            vec![Syntax::ident("Nat"), Syntax::mk_num(42)],
        );

        let result = expand(&registry, input).unwrap();

        let pretty = result.pretty();
        assert!(pretty.contains("@id"));
    }

    #[test]
    fn test_all_builtins_registered() {
        let registry = builtin_registry();

        // Verify key macros are present
        let expected_macros = vec![
            "ifThenElse",
            "unless",
            "when",
            "doSingle",
            "doBind",
            "doLet",
            "doSeq",
            "doReturn",
            "assert",
            "panic",
            "dbgTrace",
            "have",
            "letMacro",
            "show",
            "matchSingle",
            "ifLet",
        ];

        for name in expected_macros {
            assert!(
                registry.get_by_name(name).is_some(),
                "Missing macro: {name}"
            );
        }
    }

    #[test]
    fn test_nested_expansion() {
        let registry = builtin_registry();

        // Test that nested macros expand correctly
        // when (x > 0) (assert! (x > 0))
        let inner = Syntax::node(SyntaxKind::assert_macro(), vec![Syntax::ident("positive")]);

        let outer = Syntax::node(SyntaxKind::when(), vec![Syntax::ident("cond"), inner]);

        let result = expand(&registry, outer).unwrap();

        // Both when and assert should have expanded
        let pretty = result.pretty();
        assert!(pretty.contains("ite")); // from when expansion
        assert!(pretty.contains("panic")); // from assert expansion
    }

    #[test]
    fn test_calc_single_step() {
        let registry = builtin_registry();

        // calc a = b := h => h
        let input = Syntax::node(
            SyntaxKind::calc_macro(),
            vec![Syntax::node(
                SyntaxKind::calc_step(),
                vec![Syntax::ident("h")],
            )],
        );

        let result = expand(&registry, input).unwrap();
        assert_eq!(result.as_ident(), Some("h"));
    }

    #[test]
    fn test_calc_two_step() {
        let registry = builtin_registry();

        // calc a = b := h1; _ = c := h2 => Trans.trans h1 h2
        let input = Syntax::node(
            SyntaxKind::calc_macro(),
            vec![
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::ident("h1")]),
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::ident("h2")]),
            ],
        );

        let result = expand(&registry, input).unwrap();
        let pretty = result.pretty();
        assert!(pretty.contains("Trans.trans"));
        assert!(pretty.contains("h1"));
        assert!(pretty.contains("h2"));
    }

    #[test]
    fn test_calc_three_step() {
        let registry = builtin_registry();

        // calc with three steps => Trans.trans (Trans.trans h1 h2) h3
        let input = Syntax::node(
            SyntaxKind::calc_macro(),
            vec![
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::ident("h1")]),
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::ident("h2")]),
                Syntax::node(SyntaxKind::calc_step(), vec![Syntax::ident("h3")]),
            ],
        );

        let result = expand(&registry, input).unwrap();
        let pretty = result.pretty();
        assert!(pretty.contains("Trans.trans"));
    }

    #[test]
    fn test_conv_simple() {
        let registry = builtin_registry();

        // conv => t => convTactic t
        let input = Syntax::node(SyntaxKind::conv_macro(), vec![Syntax::ident("rfl")]);

        let result = expand(&registry, input).unwrap();
        let pretty = result.pretty();
        assert!(pretty.contains("convTactic"));
    }

    #[test]
    fn test_conv_at() {
        let registry = builtin_registry();

        // conv at h => t => convAt h t
        let input = Syntax::node(
            SyntaxKind::conv_macro(),
            vec![
                Syntax::ident("at"),
                Syntax::ident("h"),
                Syntax::ident("rfl"),
            ],
        );

        let result = expand(&registry, input).unwrap();
        let pretty = result.pretty();
        assert!(pretty.contains("convAt"));
    }

    #[test]
    fn test_all_calc_conv_registered() {
        let registry = builtin_registry();

        // Verify calc and conv macros are present
        let expected_macros = vec![
            "calcSingleStep",
            "calcTwoStep",
            "calcThreeStep",
            "convSimple",
            "convSeq",
            "convAt",
        ];

        for name in expected_macros {
            assert!(
                registry.get_by_name(name).is_some(),
                "Missing macro: {name}"
            );
        }
    }
}
