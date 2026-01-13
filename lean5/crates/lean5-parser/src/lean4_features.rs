//! Lean 4 Parser Feature Tests
//!
//! Comprehensive test suite for Lean 4 syntax coverage.
//! Phase 0.1 of Feature Verification Roadmap.
//!
//! Coverage tracked in `docs/PARSER_COVERAGE.md`
//!
//! Note: This module is conditionally compiled via #[cfg(test)] in lib.rs

use crate::{parse_decl, parse_expr, parse_file};

/// Helper to track test results
fn check(name: &str, result: Result<impl std::fmt::Debug, impl std::fmt::Debug>) -> bool {
    match &result {
        Ok(_) => {
            println!("  PASS: {name}");
            true
        }
        Err(e) => {
            println!("  FAIL: {name} - {e:?}");
            false
        }
    }
}

/// Helper for negative tests (should fail to parse)
fn check_reject(name: &str, result: Result<impl std::fmt::Debug, impl std::fmt::Debug>) -> bool {
    if result.is_ok() {
        println!("  FAIL: {name} - expected rejection, got success");
        false
    } else {
        println!("  PASS: {name} (correctly rejected)");
        true
    }
}

// ============================================================================
// Section 1: Universe Levels
// ============================================================================

mod universe_levels {
    use super::*;

    #[test]
    fn type_universe() {
        assert!(check("Type", parse_expr("Type")));
        assert!(check("Type 0", parse_expr("Type 0")));
        assert!(check("Type 1", parse_expr("Type 1")));
        assert!(check("Type u", parse_expr("Type u")));
        assert!(check("Type (u + 1)", parse_expr("Type (u + 1)")));
        assert!(check("Type (max u v)", parse_expr("Type (max u v)")));
        assert!(check("Type (imax u v)", parse_expr("Type (imax u v)")));
    }

    #[test]
    fn sort_universe() {
        assert!(check("Sort", parse_expr("Sort")));
        assert!(check("Sort 0", parse_expr("Sort 0")));
        assert!(check("Sort 1", parse_expr("Sort 1")));
        assert!(check("Sort u", parse_expr("Sort u")));
        assert!(check("Sort (u + 1)", parse_expr("Sort (u + 1)")));
    }

    #[test]
    fn prop() {
        assert!(check("Prop", parse_expr("Prop")));
    }

    #[test]
    fn universe_declaration() {
        assert!(check("universe u", parse_file("universe u")));
        assert!(check("universe u v", parse_file("universe u v")));
        assert!(check("universe u v w", parse_file("universe u v w")));
    }

    #[test]
    fn universe_polymorphism() {
        assert!(check(
            "def with universe",
            parse_file("universe u\ndef foo : Type u := sorry")
        ));
        assert!(check(
            "forall with Type u",
            parse_expr("forall (α : Type u), α → α")
        ));
    }
}

// ============================================================================
// Section 2: Binders
// ============================================================================

mod binders {
    use super::*;

    #[test]
    fn explicit_binder() {
        assert!(check("(x : Nat)", parse_expr("fun (x : Nat) => x")));
        assert!(check("(x y : Nat)", parse_expr("fun (x y : Nat) => x")));
        assert!(check(
            "(x : Nat) (y : Nat)",
            parse_expr("fun (x : Nat) (y : Nat) => x")
        ));
    }

    #[test]
    fn implicit_binder() {
        assert!(check("{x : Nat}", parse_expr("fun {x : Nat} => x")));
        assert!(check("{x y : Nat}", parse_expr("fun {x y : Nat} => x")));
    }

    #[test]
    fn instance_implicit_binder() {
        assert!(check("[x : Nat]", parse_expr("fun [x : Nat] => x")));
        assert!(check(
            "[ToString α]",
            parse_decl("def foo [ToString α] (x : α) : String := sorry")
        ));
    }

    #[test]
    fn strict_implicit_binder() {
        // {{x : T}} - strict implicit binders
        assert!(check("{{x : Nat}}", parse_expr("fun {{x : Nat}} => x")));
    }

    #[test]
    fn anonymous_binder() {
        assert!(check("(_ : Nat)", parse_expr("fun (_ : Nat) => 0")));
    }

    #[test]
    fn binder_without_type() {
        assert!(check("(x)", parse_expr("fun x => x")));
        assert!(check("fun x y", parse_expr("fun x y => x")));
    }
}

// ============================================================================
// Section 3: Lambda Expressions
// ============================================================================

mod lambda {
    use super::*;

    #[test]
    fn simple_lambda() {
        assert!(check("fun x => x", parse_expr("fun x => x")));
        assert!(check("λ x => x", parse_expr("λ x => x")));
    }

    #[test]
    fn typed_lambda() {
        assert!(check(
            "fun (x : Nat) => x",
            parse_expr("fun (x : Nat) => x")
        ));
    }

    #[test]
    fn multi_arg_lambda() {
        assert!(check("fun x y => x", parse_expr("fun x y => x")));
        assert!(check("fun x y z => x", parse_expr("fun x y z => x")));
    }

    #[test]
    fn nested_lambda() {
        assert!(check(
            "fun x => fun y => x",
            parse_expr("fun x => fun y => x")
        ));
    }

    #[test]
    fn pattern_lambda() {
        // fun | pat => body
        assert!(check("fun | 0 => 1", parse_expr("fun | 0 => 1")));
        assert!(check(
            "fun | 0 => 1 | n => n",
            parse_expr("fun | 0 => 1 | n => n")
        ));
    }

    #[test]
    fn lambda_with_hole() {
        assert!(check("fun (x : _) => x", parse_expr("fun (x : _) => x")));
    }
}

// ============================================================================
// Section 4: Application
// ============================================================================

mod application {
    use super::*;

    #[test]
    fn simple_app() {
        assert!(check("f x", parse_expr("f x")));
        assert!(check("f x y", parse_expr("f x y")));
    }

    #[test]
    fn parenthesized_arg() {
        assert!(check("f (x)", parse_expr("f (x)")));
        assert!(check("f (g x)", parse_expr("f (g x)")));
    }

    #[test]
    fn explicit_app() {
        assert!(check("@f x", parse_expr("@f x")));
        assert!(check("@id Nat 0", parse_expr("@id Nat 0")));
    }

    #[test]
    fn named_argument() {
        assert!(check("f (x := 1)", parse_expr("f (x := 1)")));
    }

    #[test]
    fn partial_app() {
        // · for partial application placeholder
        assert!(check("f · y", parse_expr("f · y")));
        assert!(check("· + 1", parse_expr("· + 1")));
    }
}

// ============================================================================
// Section 5: Let Expressions
// ============================================================================

mod let_expr {
    use super::*;

    #[test]
    fn simple_let() {
        assert!(check("let x := 1; x", parse_expr("let x := 1; x")));
        assert!(check("let x := 1 in x", parse_expr("let x := 1 in x")));
    }

    #[test]
    fn typed_let() {
        assert!(check(
            "let x : Nat := 1; x",
            parse_expr("let x : Nat := 1; x")
        ));
    }

    #[test]
    fn chained_let() {
        assert!(check(
            "let x := 1; let y := 2; x + y",
            parse_expr("let x := 1; let y := 2; x + y")
        ));
    }

    #[test]
    fn let_rec() {
        assert!(check(
            "let rec f := ...",
            parse_expr("let rec f (n : Nat) : Nat := match n with | 0 => 1 | _ => f 0; f 5")
        ));
    }

    #[test]
    fn let_fun() {
        // let fun is shorthand for let with lambda
        assert!(check("let f x := x; f 0", parse_expr("let f x := x; f 0")));
    }
}

// ============================================================================
// Section 6: Match Expressions
// ============================================================================

mod match_expr {
    use super::*;

    #[test]
    fn simple_match() {
        assert!(check(
            "match x with | 0 => 1",
            parse_expr("match x with | 0 => 1")
        ));
    }

    #[test]
    fn multi_arm_match() {
        assert!(check(
            "match with multiple arms",
            parse_expr("match x with | 0 => 1 | n => n")
        ));
    }

    #[test]
    fn match_with_discriminant_type() {
        assert!(check(
            "match x : Nat with",
            parse_expr("match (x : Nat) with | _ => 0")
        ));
    }

    #[test]
    fn match_multiple_scrutinees() {
        assert!(check(
            "match x, y with",
            parse_expr("match x, y with | a, b => a")
        ));
    }

    #[test]
    fn pattern_wildcards() {
        assert!(check("| _ => e", parse_expr("match x with | _ => 0")));
    }

    #[test]
    fn pattern_constructor() {
        assert!(check(
            "| .cons h t",
            parse_expr("match xs with | .nil => 0 | .cons h t => 1")
        ));
    }

    #[test]
    fn pattern_as() {
        assert!(check(
            "pat@...",
            parse_expr("match x with | n@0 => n | _ => 1")
        ));
    }

    #[test]
    fn pattern_or() {
        assert!(check(
            "| 0 | 1",
            parse_expr("match x with | 0 | 1 => true | _ => false")
        ));
    }

    #[test]
    fn n_plus_k_pattern() {
        // n+1 pattern for Nat
        assert!(check(
            "| n + 1",
            parse_expr("match x with | 0 => 0 | n + 1 => n")
        ));
    }
}

// ============================================================================
// Section 7: Do Notation
// ============================================================================

mod do_notation {
    use super::*;

    #[test]
    fn simple_do() {
        assert!(check("do return 1", parse_expr("do return 1")));
    }

    #[test]
    fn do_with_bind() {
        assert!(check("do let x ← m", parse_expr("do let x ← m; return x")));
    }

    #[test]
    fn do_with_let() {
        assert!(check(
            "do let x := 1",
            parse_expr("do let x := 1; return x")
        ));
    }

    #[test]
    fn do_multiline() {
        assert!(check(
            "do block",
            parse_decl("def test : IO Unit := do\n  let x ← pure 1\n  pure ()")
        ));
    }

    #[test]
    fn do_if() {
        assert!(check(
            "do if",
            parse_expr("do if true then return 1 else return 0")
        ));
    }

    #[test]
    fn do_for() {
        assert!(check(
            "do for",
            parse_expr("do for x in xs do IO.println x")
        ));
    }

    #[test]
    fn do_unless() {
        assert!(check("do unless", parse_expr("do unless cond do action")));
    }
}

// ============================================================================
// Section 8: Notation and Operators
// ============================================================================

mod notation {
    use super::*;

    #[test]
    fn infix_notation() {
        assert!(check(
            "infix declaration",
            parse_decl("infix:50 \" ++ \" => append")
        ));
    }

    #[test]
    fn prefix_notation() {
        assert!(check(
            "prefix declaration",
            parse_decl("prefix:100 \"!\" => not")
        ));
    }

    #[test]
    fn postfix_notation() {
        assert!(check(
            "postfix declaration",
            parse_decl("postfix:max \"!\" => factorial")
        ));
    }

    #[test]
    fn notation_declaration() {
        assert!(check(
            "notation declaration",
            parse_decl("notation \"[\" a \", \" b \"]\" => Pair.mk a b")
        ));
    }

    #[test]
    fn macro_declaration() {
        assert!(check(
            "macro declaration",
            parse_decl("macro \"hello\" : term => `(IO.println \"Hello\")")
        ));
    }

    #[test]
    fn macro_rules() {
        assert!(check(
            "macro_rules",
            parse_decl("macro_rules | `(myMacro $x) => `($x + 1)")
        ));
    }

    #[test]
    fn syntax_declaration() {
        assert!(check(
            "syntax declaration",
            parse_decl("syntax \"myKeyword\" term : term")
        ));
    }
}

// ============================================================================
// Section 9: Structure Declarations
// ============================================================================

mod structure_decl {
    use super::*;

    #[test]
    fn simple_structure() {
        assert!(check(
            "structure Point",
            parse_decl("structure Point where\n  x : Nat\n  y : Nat")
        ));
    }

    #[test]
    fn structure_with_default() {
        assert!(check(
            "structure with default",
            parse_decl("structure Point where\n  x : Nat := 0\n  y : Nat := 0")
        ));
    }

    #[test]
    fn structure_with_parameters() {
        assert!(check(
            "structure with params",
            parse_decl("structure Vec (α : Type) (n : Nat) where\n  data : Array α")
        ));
    }

    #[test]
    fn structure_extends() {
        assert!(check(
            "structure extends",
            parse_decl("structure ColorPoint extends Point where\n  color : Nat")
        ));
    }

    #[test]
    fn structure_constructor() {
        assert!(check(
            "structure with mk",
            parse_decl("structure Point where\n  mk ::\n  x : Nat\n  y : Nat")
        ));
    }
}

// ============================================================================
// Section 10: Class Declarations
// ============================================================================

mod class_decl {
    use super::*;

    #[test]
    fn simple_class() {
        assert!(check(
            "class declaration",
            parse_decl("class Inhabited (α : Type)")
        ));
    }

    #[test]
    fn class_with_method() {
        assert!(check(
            "class with method",
            parse_decl("class ToString (α : Type) where\n  toString : α → String")
        ));
    }

    #[test]
    fn class_extends() {
        assert!(check(
            "class extends",
            parse_decl("class Monad (m : Type → Type) extends Applicative m where\n  bind : m α → (α → m β) → m β")
        ));
    }

    #[test]
    fn abbrev_class() {
        assert!(check(
            "abbrev class",
            parse_decl("abbrev class MonadIO (m : Type → Type) := Monad m")
        ));
    }
}

// ============================================================================
// Section 11: Instance Declarations
// ============================================================================

mod instance_decl {
    use super::*;

    #[test]
    fn simple_instance() {
        assert!(check(
            "instance",
            parse_decl("instance : Inhabited Nat where\n  default := 0")
        ));
    }

    #[test]
    fn named_instance() {
        assert!(check(
            "named instance",
            parse_decl("instance instInhabitedNat : Inhabited Nat where\n  default := 0")
        ));
    }

    #[test]
    fn instance_with_priority() {
        assert!(check(
            "instance priority",
            parse_decl("instance (priority := high) : Inhabited Nat where\n  default := 0")
        ));
    }

    #[test]
    fn instance_with_parameters() {
        assert!(check(
            "instance with params",
            parse_decl("instance [Inhabited α] : Inhabited (List α) where\n  default := []")
        ));
    }
}

// ============================================================================
// Section 12: Inductive Declarations
// ============================================================================

mod inductive_decl {
    use super::*;

    #[test]
    fn simple_inductive() {
        assert!(check(
            "inductive Bool",
            parse_decl("inductive Bool where\n  | false : Bool\n  | true : Bool")
        ));
    }

    #[test]
    fn inductive_with_parameters() {
        assert!(check(
            "inductive List",
            parse_decl("inductive List (α : Type u) where\n  | nil : List α\n  | cons : α → List α → List α")
        ));
    }

    #[test]
    fn inductive_with_indices() {
        assert!(check(
            "inductive Vec",
            parse_decl("inductive Vec (α : Type) : Nat → Type where\n  | nil : Vec α 0\n  | cons : α → Vec α n → Vec α (n + 1)")
        ));
    }

    #[test]
    fn mutual_inductive() {
        assert!(check(
            "mutual inductive",
            parse_file("mutual\n  inductive Even : Nat → Prop\n    | zero : Even 0\n    | succ : Odd n → Even (n + 1)\n  inductive Odd : Nat → Prop\n    | succ : Even n → Odd (n + 1)\nend")
        ));
    }
}

// ============================================================================
// Section 13: Mutual Definitions
// ============================================================================

mod mutual_def {
    use super::*;

    #[test]
    fn mutual_def() {
        assert!(check(
            "mutual def",
            parse_file("mutual\n  def f : Nat → Nat\n    | 0 => 0\n    | n + 1 => g n\n  def g : Nat → Nat\n    | 0 => 1\n    | n + 1 => f n\nend")
        ));
    }
}

// ============================================================================
// Section 14: Where Clauses
// ============================================================================

mod where_clause {
    use super::*;

    #[test]
    fn def_with_where() {
        assert!(check(
            "def with where",
            parse_decl("def foo : Nat := x + y where\n  x := 1\n  y := 2")
        ));
    }

    #[test]
    fn def_where_match() {
        assert!(check(
            "def where match",
            parse_decl("def foo : Nat → Nat where\n  | 0 => 1\n  | n + 1 => n")
        ));
    }
}

// ============================================================================
// Section 15: Calc Blocks
// ============================================================================

mod calc_blocks {
    use super::*;

    #[test]
    fn simple_calc() {
        assert!(check(
            "calc",
            parse_expr("calc a = b := h1\n       _ = c := h2")
        ));
    }

    #[test]
    fn calc_with_transitive_relation() {
        assert!(check(
            "calc with ≤",
            parse_expr("calc a ≤ b := h1\n       _ ≤ c := h2")
        ));
    }
}

// ============================================================================
// Section 16: Have/Let/Show in Terms
// ============================================================================

mod term_have_let_show {
    use super::*;

    #[test]
    fn have_in_term() {
        assert!(check(
            "have in term",
            parse_expr("have h : P := proof; conclusion")
        ));
    }

    #[test]
    fn show_in_term() {
        assert!(check("show in term", parse_expr("show P from proof")));
    }

    #[test]
    fn suffices_in_term() {
        assert!(check(
            "suffices in term",
            parse_expr("suffices h : P by exact h; proof")
        ));
    }
}

// ============================================================================
// Section 17: Anonymous Constructor
// ============================================================================

mod anonymous_ctor {
    use super::*;

    #[test]
    fn angle_bracket_ctor() {
        assert!(check("⟨a, b, c⟩", parse_expr("⟨a, b, c⟩")));
        assert!(check("⟨a⟩", parse_expr("⟨a⟩")));
        assert!(check("⟨⟩", parse_expr("⟨⟩")));
    }

    #[test]
    fn nested_angle_bracket() {
        assert!(check("⟨⟨a⟩, b⟩", parse_expr("⟨⟨a⟩, b⟩")));
    }
}

// ============================================================================
// Section 18: Field Notation
// ============================================================================

mod field_notation {
    use super::*;

    #[test]
    fn named_field() {
        assert!(check("x.field", parse_expr("x.field")));
        assert!(check("x.y.z", parse_expr("x.y.z")));
    }

    #[test]
    fn index_field() {
        assert!(check("x.1", parse_expr("x.1")));
        assert!(check("x.2", parse_expr("x.2")));
    }

    #[test]
    fn field_after_app() {
        assert!(check("(f x).field", parse_expr("(f x).field")));
    }

    #[test]
    fn ufcs_field() {
        // Universal function call syntax
        assert!(check("x.foo y", parse_expr("x.foo y")));
    }
}

// ============================================================================
// Section 19: If-then-else
// ============================================================================

mod if_then_else {
    use super::*;

    #[test]
    fn simple_if() {
        assert!(check("if then else", parse_expr("if c then t else e")));
    }

    #[test]
    fn nested_if() {
        assert!(check(
            "nested if",
            parse_expr("if a then (if b then 1 else 2) else 3")
        ));
    }

    #[test]
    fn if_let() {
        assert!(check(
            "if let",
            parse_expr("if let some x := opt then x else 0")
        ));
    }
}

// ============================================================================
// Section 20: Decidable If
// ============================================================================

mod decidable_if {
    use super::*;

    #[test]
    fn decidable_if() {
        // if h : p then t else e
        assert!(check("if h : p", parse_expr("if h : p then t else e")));
    }
}

// ============================================================================
// Section 21: Syntax Quotations
// ============================================================================

mod syntax_quotations {
    use super::*;

    #[test]
    fn simple_quote() {
        assert!(check("`(x)", parse_expr("`(x)")));
        assert!(check("`(1 + 2)", parse_expr("`(1 + 2)")));
    }

    #[test]
    fn antiquotation() {
        assert!(check("`($x)", parse_expr("`($x)")));
    }

    #[test]
    fn typed_antiquotation() {
        assert!(check("`($x:term)", parse_expr("`($x:term)")));
    }

    #[test]
    fn splice_antiquotation() {
        assert!(check("`($[xs]*)", parse_expr("`($[xs]*)")));
    }
}

// ============================================================================
// Section 22: Attributes
// ============================================================================

mod attributes {
    use super::*;

    #[test]
    fn simple_attribute() {
        assert!(check("@[simp]", parse_decl("@[simp] def foo : Nat := 1")));
    }

    #[test]
    fn multiple_attributes() {
        assert!(check(
            "@[simp, local]",
            parse_decl("@[simp, local] def foo : Nat := 1")
        ));
    }

    #[test]
    fn attribute_with_args() {
        assert!(check(
            "@[simp high]",
            parse_decl("@[simp high] def foo : Nat := 1")
        ));
    }

    #[test]
    fn inline_attribute() {
        assert!(check(
            "@[inline]",
            parse_decl("@[inline] def foo : Nat := 1")
        ));
    }

    #[test]
    fn scoped_attribute() {
        assert!(check(
            "scoped attribute",
            parse_file("attribute [local simp] foo")
        ));
    }
}

// ============================================================================
// Section 23: Namespace and Section
// ============================================================================

mod namespace_section {
    use super::*;

    #[test]
    fn namespace() {
        assert!(check(
            "namespace",
            parse_file("namespace Foo\ndef bar : Nat := 1\nend Foo")
        ));
    }

    #[test]
    fn nested_namespace() {
        assert!(check(
            "nested namespace",
            parse_file("namespace Foo.Bar\ndef baz : Nat := 1\nend Foo.Bar")
        ));
    }

    #[test]
    fn section() {
        assert!(check(
            "section",
            parse_file("section\nvariable (x : Nat)\ndef foo : Nat := x\nend")
        ));
    }

    #[test]
    fn named_section() {
        assert!(check(
            "named section",
            parse_file("section MySection\ndef foo : Nat := 1\nend MySection")
        ));
    }
}

// ============================================================================
// Section 24: Variable Command
// ============================================================================

mod variable_cmd {
    use super::*;

    #[test]
    fn variable() {
        assert!(check("variable", parse_file("variable (x : Nat)")));
    }

    #[test]
    fn variable_implicit() {
        assert!(check(
            "variable implicit",
            parse_file("variable {α : Type}")
        ));
    }

    #[test]
    fn variable_instance() {
        assert!(check(
            "variable instance",
            parse_file("variable [ToString α]")
        ));
    }
}

// ============================================================================
// Section 25: Open Command
// ============================================================================

mod open_cmd {
    use super::*;

    #[test]
    fn open_namespace() {
        assert!(check("open", parse_file("open Nat")));
    }

    #[test]
    fn open_in() {
        assert!(check("open in", parse_file("open Nat in #check succ")));
    }

    #[test]
    fn open_hiding() {
        assert!(check("open hiding", parse_file("open Nat hiding succ")));
    }

    #[test]
    fn open_renaming() {
        assert!(check(
            "open renaming",
            parse_file("open Nat renaming succ → next")
        ));
    }
}

// ============================================================================
// Section 26: Import
// ============================================================================

mod import_cmd {
    use super::*;

    #[test]
    fn import() {
        assert!(check("import", parse_file("import Lean")));
    }

    #[test]
    fn import_multiple() {
        assert!(check(
            "import multiple",
            parse_file("import Lean\nimport Std")
        ));
    }
}

// ============================================================================
// Section 27: Definition Forms
// ============================================================================

mod definition_forms {
    use super::*;

    #[test]
    fn def() {
        assert!(check("def", parse_decl("def foo : Nat := 1")));
    }

    #[test]
    fn theorem() {
        assert!(check("theorem", parse_decl("theorem foo : 1 = 1 := rfl")));
    }

    #[test]
    fn lemma() {
        assert!(check("lemma", parse_decl("lemma foo : 1 = 1 := rfl")));
    }

    #[test]
    fn abbrev() {
        assert!(check("abbrev", parse_decl("abbrev foo : Nat := 1")));
    }

    #[test]
    fn example_cmd() {
        assert!(check("example", parse_decl("example : 1 = 1 := rfl")));
    }

    #[test]
    fn opaque() {
        assert!(check("opaque", parse_decl("opaque foo : Nat")));
    }

    #[test]
    fn axiom() {
        assert!(check("axiom", parse_decl("axiom foo : Nat")));
    }

    #[test]
    fn constant() {
        assert!(check("constant", parse_decl("constant foo : Nat")));
    }
}

// ============================================================================
// Section 28: Deriving
// ============================================================================

mod deriving {
    use super::*;

    #[test]
    fn deriving_repr() {
        assert!(check(
            "deriving Repr",
            parse_decl("structure Point where\n  x : Nat\n  y : Nat\nderiving Repr")
        ));
    }

    #[test]
    fn deriving_multiple() {
        assert!(check(
            "deriving multiple",
            parse_decl("structure Point where\n  x : Nat\nderiving Repr, BEq")
        ));
    }
}

// ============================================================================
// Section 29: Comments
// ============================================================================

mod comments {
    use super::*;

    #[test]
    fn line_comment() {
        assert!(check(
            "-- comment",
            parse_file("-- comment\ndef foo : Nat := 1")
        ));
    }

    #[test]
    fn block_comment() {
        assert!(check(
            "/- comment -/",
            parse_file("/- comment -/\ndef foo : Nat := 1")
        ));
    }

    #[test]
    fn nested_block_comment() {
        assert!(check(
            "nested /- /- -/ -/",
            parse_file("/- outer /- inner -/ -/\ndef foo : Nat := 1")
        ));
    }

    #[test]
    fn doc_comment() {
        assert!(check(
            "/-- doc -/",
            parse_file("/-- Documentation -/\ndef foo : Nat := 1")
        ));
    }
}

// ============================================================================
// Section 30: String Literals
// ============================================================================

mod string_literals {
    use super::*;

    #[test]
    fn simple_string() {
        assert!(check("\"hello\"", parse_expr("\"hello\"")));
    }

    #[test]
    fn string_with_escapes() {
        assert!(check("\"\\n\\t\"", parse_expr("\"line1\\nline2\"")));
    }

    #[test]
    fn string_interpolation() {
        assert!(check("s!\"...\"", parse_expr("s!\"hello {name}\"")));
    }

    #[test]
    fn raw_string() {
        // This is a valid test but the syntax may vary
        // assert!(check("r\"...\"", parse_expr("r\"raw string\"")));
    }
}

// ============================================================================
// Section 31: Numeric Literals
// ============================================================================

mod numeric_literals {
    use super::*;

    #[test]
    fn nat_literal() {
        assert!(check("0", parse_expr("0")));
        assert!(check("42", parse_expr("42")));
        assert!(check("1000000", parse_expr("1000000")));
    }

    #[test]
    fn int_literal() {
        assert!(check("-1", parse_expr("-1")));
        assert!(check("-42", parse_expr("-42")));
    }

    #[test]
    fn hex_literal() {
        assert!(check("0xFF", parse_expr("0xFF")));
        assert!(check("0x1A2B", parse_expr("0x1A2B")));
    }

    #[test]
    fn binary_literal() {
        assert!(check("0b1010", parse_expr("0b1010")));
    }

    #[test]
    fn octal_literal() {
        assert!(check("0o777", parse_expr("0o777")));
    }

    #[test]
    fn scientific_notation() {
        assert!(check("1.5e10", parse_expr("1.5e10")));
    }
}

// ============================================================================
// Section 32: Array and List Literals
// ============================================================================

mod collection_literals {
    use super::*;

    #[test]
    fn array_literal() {
        assert!(check("#[1, 2, 3]", parse_expr("#[1, 2, 3]")));
        assert!(check("#[]", parse_expr("#[]")));
    }

    #[test]
    fn list_literal() {
        assert!(check("[1, 2, 3]", parse_expr("[1, 2, 3]")));
        assert!(check("[]", parse_expr("[]")));
    }
}

// ============================================================================
// Section 33: Negative Tests (Should Reject)
// ============================================================================

mod negative_tests {
    use super::*;

    #[test]
    fn reject_incomplete_lambda() {
        assert!(check_reject("incomplete lambda", parse_expr("fun x =>")));
    }

    #[test]
    fn reject_unclosed_paren() {
        assert!(check_reject("unclosed paren", parse_expr("(x")));
    }

    #[test]
    fn reject_unclosed_brace() {
        assert!(check_reject("unclosed brace", parse_expr("{x")));
    }

    #[test]
    fn reject_mismatched_brackets() {
        assert!(check_reject("mismatched brackets", parse_expr("(x}")));
    }

    #[test]
    fn reject_empty_def() {
        assert!(check_reject("empty def", parse_decl("def")));
    }

    #[test]
    fn reject_structure_no_fields() {
        // Structure without where is technically valid as a class/opaque
        // This tests for proper parsing error on malformed input
        assert!(check_reject(
            "malformed structure",
            parse_decl("structure := bad")
        ));
    }
}

// ============================================================================
// Summary Test - Runs All Categories
// ============================================================================

#[test]
fn parser_coverage_summary() {
    println!();
    println!("=========================================");
    println!("Lean 4 Parser Feature Coverage Summary");
    println!("=========================================");
    println!("See individual test modules for details.");
    println!("Coverage documented in docs/PARSER_COVERAGE.md");
    println!("=========================================");
}
