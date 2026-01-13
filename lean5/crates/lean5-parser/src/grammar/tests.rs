use super::*;

#[test]
fn test_parse_ident() {
    let expr = Parser::parse_expr("x").unwrap();
    assert!(matches!(expr, SurfaceExpr::Ident(_, s) if s == "x"));
}

#[test]
fn test_parse_nat_lit() {
    let expr = Parser::parse_expr("42").unwrap();
    assert!(matches!(expr, SurfaceExpr::Lit(_, SurfaceLit::Nat(42))));
}

#[test]
fn test_parse_type() {
    let expr = Parser::parse_expr("Type").unwrap();
    assert!(matches!(expr, SurfaceExpr::Universe(_, UniverseExpr::Type)));
}

#[test]
fn test_parse_prop() {
    let expr = Parser::parse_expr("Prop").unwrap();
    assert!(matches!(expr, SurfaceExpr::Universe(_, UniverseExpr::Prop)));
}

#[test]
fn test_parse_app() {
    let expr = Parser::parse_expr("f x y").unwrap();
    match expr {
        SurfaceExpr::App(_, func, args) => {
            assert!(matches!(*func, SurfaceExpr::Ident(_, s) if s == "f"));
            assert_eq!(args.len(), 2);
        }
        _ => panic!("expected App"),
    }
}

#[test]
fn test_parse_arrow() {
    let expr = Parser::parse_expr("A -> B").unwrap();
    match expr {
        SurfaceExpr::Arrow(_, left, right) => {
            assert!(matches!(*left, SurfaceExpr::Ident(_, s) if s == "A"));
            assert!(matches!(*right, SurfaceExpr::Ident(_, s) if s == "B"));
        }
        _ => panic!("expected Arrow"),
    }
}

#[test]
fn test_parse_arrow_unicode() {
    let expr = Parser::parse_expr("A → B → C").unwrap();
    // Should be right associative: A → (B → C)
    match expr {
        SurfaceExpr::Arrow(_, left, right) => {
            assert!(matches!(*left, SurfaceExpr::Ident(_, s) if s == "A"));
            assert!(matches!(*right, SurfaceExpr::Arrow(_, _, _)));
        }
        _ => panic!("expected Arrow"),
    }
}

#[test]
fn test_parse_lambda() {
    let expr = Parser::parse_expr("fun x => x").unwrap();
    match expr {
        SurfaceExpr::Lambda(_, binders, body) => {
            assert_eq!(binders.len(), 1);
            assert_eq!(binders[0].name, "x");
            assert!(matches!(*body, SurfaceExpr::Ident(_, s) if s == "x"));
        }
        _ => panic!("expected Lambda"),
    }
}

#[test]
fn test_parse_lambda_typed() {
    let expr = Parser::parse_expr("fun (x : Nat) => x").unwrap();
    match expr {
        SurfaceExpr::Lambda(_, binders, _) => {
            assert_eq!(binders.len(), 1);
            assert!(binders[0].ty.is_some());
        }
        _ => panic!("expected Lambda"),
    }
}

#[test]
fn test_parse_forall() {
    let expr = Parser::parse_expr("forall (x : Type), x").unwrap();
    match expr {
        SurfaceExpr::Pi(_, binders, body) => {
            assert_eq!(binders.len(), 1);
            assert_eq!(binders[0].name, "x");
            assert!(matches!(*body, SurfaceExpr::Ident(_, s) if s == "x"));
        }
        _ => panic!("expected Pi"),
    }
}

#[test]
fn test_parse_let() {
    let expr = Parser::parse_expr("let x := 1 in x").unwrap();
    match expr {
        SurfaceExpr::Let(_, binder, val, body) => {
            assert_eq!(binder.name, "x");
            assert!(matches!(*val, SurfaceExpr::Lit(_, SurfaceLit::Nat(1))));
            assert!(matches!(*body, SurfaceExpr::Ident(_, s) if s == "x"));
        }
        _ => panic!("expected Let"),
    }
}

#[test]
fn test_parse_let_typed() {
    let expr = Parser::parse_expr("let x : Nat := 1 in x").unwrap();
    match expr {
        SurfaceExpr::Let(_, binder, _, _) => {
            assert!(binder.ty.is_some());
        }
        _ => panic!("expected Let"),
    }
}

#[test]
fn test_parse_if() {
    let expr = Parser::parse_expr("if c then t else e").unwrap();
    match expr {
        SurfaceExpr::If(_, cond, then_br, else_br) => {
            assert!(matches!(*cond, SurfaceExpr::Ident(_, s) if s == "c"));
            assert!(matches!(*then_br, SurfaceExpr::Ident(_, s) if s == "t"));
            assert!(matches!(*else_br, SurfaceExpr::Ident(_, s) if s == "e"));
        }
        _ => panic!("expected If"),
    }
}

#[test]
fn test_parse_paren() {
    let expr = Parser::parse_expr("(x)").unwrap();
    match expr {
        SurfaceExpr::Paren(_, inner) => {
            assert!(matches!(*inner, SurfaceExpr::Ident(_, s) if s == "x"));
        }
        _ => panic!("expected Paren"),
    }
}

#[test]
fn test_parse_ascription() {
    let expr = Parser::parse_expr("(x : Nat)").unwrap();
    match expr {
        SurfaceExpr::Ascription(_, expr, ty) => {
            assert!(matches!(*expr, SurfaceExpr::Ident(_, s) if s == "x"));
            assert!(matches!(*ty, SurfaceExpr::Ident(_, s) if s == "Nat"));
        }
        _ => panic!("expected Ascription"),
    }
}

#[test]
fn test_parse_hole() {
    let expr = Parser::parse_expr("_").unwrap();
    assert!(matches!(expr, SurfaceExpr::Hole(_)));
}

#[test]
fn test_parse_def() {
    let decl = Parser::parse_decl("def id (x : Type) := x").unwrap();
    match decl {
        SurfaceDecl::Def { name, binders, .. } => {
            assert_eq!(name, "id");
            assert_eq!(binders.len(), 1);
        }
        _ => panic!("expected Def"),
    }
}

#[test]
fn test_parse_theorem() {
    let decl = Parser::parse_decl("theorem foo : Prop := Prop").unwrap();
    match decl {
        SurfaceDecl::Theorem { name, .. } => {
            assert_eq!(name, "foo");
        }
        _ => panic!("expected Theorem"),
    }
}

#[test]
fn test_parse_complex() {
    // Parse a more complex expression
    let expr = Parser::parse_expr("fun (A : Type) (x : A) => x").unwrap();
    match expr {
        SurfaceExpr::Lambda(_, binders, _) => {
            assert_eq!(binders.len(), 2);
            assert_eq!(binders[0].name, "A");
            assert_eq!(binders[1].name, "x");
        }
        _ => panic!("expected Lambda"),
    }
}

#[test]
fn test_parse_file() {
    let input = r"
            def id (x : Type) := x
            def const (A : Type) (B : Type) (x : A) := x
            axiom myAxiom : Type
        ";
    let decls = Parser::parse_file(input).unwrap();
    assert_eq!(decls.len(), 3);

    match &decls[0] {
        SurfaceDecl::Def { name, .. } => assert_eq!(name, "id"),
        _ => panic!("expected Def"),
    }

    match &decls[1] {
        SurfaceDecl::Def { name, binders, .. } => {
            assert_eq!(name, "const");
            assert_eq!(binders.len(), 3);
        }
        _ => panic!("expected Def"),
    }

    match &decls[2] {
        SurfaceDecl::Axiom { name, .. } => assert_eq!(name, "myAxiom"),
        _ => panic!("expected Axiom"),
    }
}

#[test]
fn test_parse_empty_file() {
    let decls = Parser::parse_file("").unwrap();
    assert!(decls.is_empty());
}

#[test]
fn test_parse_greek_implicit_binders() {
    // Test implicit binders with Greek letters
    let input = "inductive Imf {α : Type u} {β : Type v} (f : α → β) : β → Type (max u v)\n| mk : (a : α) → Imf f (f a)";
    let result = Parser::parse_file(input);
    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
}

#[test]
fn test_parse_structure_simple() {
    let decl = Parser::parse_decl(
        r"structure Point where
              x : Nat
              y : Nat",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure { name, fields, .. } => {
            assert_eq!(name, "Point");
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "x");
            assert_eq!(fields[1].name, "y");
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_parse_structure_with_params() {
    let decl = Parser::parse_decl(
        r"structure Pair (A : Type) (B : Type) where
              fst : A
              snd : B",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure {
            name,
            binders,
            fields,
            ..
        } => {
            assert_eq!(name, "Pair");
            assert_eq!(binders.len(), 2);
            assert_eq!(binders[0].name, "A");
            assert_eq!(binders[1].name, "B");
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "fst");
            assert_eq!(fields[1].name, "snd");
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_parse_structure_with_type() {
    let decl = Parser::parse_decl(
        r"structure MyType : Type where
              val : Nat",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure { name, ty, .. } => {
            assert_eq!(name, "MyType");
            assert!(ty.is_some());
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_parse_structure_with_universe_params() {
    let decl = Parser::parse_decl(
        r"structure Container {u} (A : Type u) where
              data : A",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure {
            name,
            universe_params,
            binders,
            fields,
            ..
        } => {
            assert_eq!(name, "Container");
            assert_eq!(universe_params.len(), 1);
            assert_eq!(universe_params[0], "u");
            assert_eq!(binders.len(), 1);
            assert_eq!(fields.len(), 1);
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_parse_structure_dependent_field() {
    // Test: field type that references an earlier field name
    // This tests that `B fst` is parsed as an application, not stopping at `fst`
    let decl = Parser::parse_decl(
        r"structure Sigma (A : Type) (B : A -> Type) where
              fst : A
              snd : B fst",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure {
            name,
            binders,
            fields,
            ..
        } => {
            assert_eq!(name, "Sigma");
            assert_eq!(binders.len(), 2);
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "fst");
            assert_eq!(fields[1].name, "snd");
            // Verify the second field type is an application B fst
            match &fields[1].ty {
                SurfaceExpr::App(_, func, args) => {
                    assert!(
                        matches!(func.as_ref(), SurfaceExpr::Ident(_, name) if name == "B"),
                        "Expected function to be B, got {func:?}"
                    );
                    assert_eq!(args.len(), 1, "Expected one argument (fst)");
                    assert!(
                        matches!(&args[0].expr, SurfaceExpr::Ident(_, name) if name == "fst"),
                        "Expected arg to be fst, got {:?}",
                        args[0].expr
                    );
                }
                other => panic!("Expected App, got {other:?}"),
            }
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_parse_class_simple() {
    let decl = Parser::parse_decl(
        r"class Add (α : Type) where
              add : α → α → α",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Class {
            name,
            binders,
            fields,
            ..
        } => {
            assert_eq!(name, "Add");
            assert_eq!(binders.len(), 1);
            assert_eq!(binders[0].name, "α");
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].name, "add");
        }
        _ => panic!("expected Class"),
    }
}

#[test]
fn test_parse_class_multiple_methods() {
    let decl = Parser::parse_decl(
        r"class Ord (α : Type) where
              lt : α → α → Prop
              le : α → α → Prop
              gt : α → α → Prop",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Class { name, fields, .. } => {
            assert_eq!(name, "Ord");
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0].name, "lt");
            assert_eq!(fields[1].name, "le");
            assert_eq!(fields[2].name, "gt");
        }
        _ => panic!("expected Class"),
    }
}

#[test]
fn test_parse_class_with_default() {
    let decl = Parser::parse_decl(
        r"class Inhabited (α : Type) where
              default : α",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Class {
            name,
            binders,
            fields,
            ..
        } => {
            assert_eq!(name, "Inhabited");
            assert_eq!(binders.len(), 1);
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].name, "default");
        }
        _ => panic!("expected Class"),
    }
}

#[test]
fn test_parse_instance_named() {
    let decl = Parser::parse_decl(
        r"instance instAddNat : Add Nat where
              add := Nat.add",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance {
            name,
            binders,
            class_type,
            fields,
            ..
        } => {
            assert_eq!(name, Some("instAddNat".to_string()));
            assert!(binders.is_empty());
            // class_type should be `Add Nat`
            match class_type.as_ref() {
                SurfaceExpr::App(_, func, args) => {
                    assert!(matches!(func.as_ref(), SurfaceExpr::Ident(_, n) if n == "Add"));
                    assert_eq!(args.len(), 1);
                }
                _ => panic!("expected App"),
            }
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].name, "add");
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_parse_instance_anonymous() {
    let decl = Parser::parse_decl(
        r"instance : Add Nat where
              add := Nat.add",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance { name, fields, .. } => {
            assert!(name.is_none());
            assert_eq!(fields.len(), 1);
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_parse_instance_with_binders() {
    let decl = Parser::parse_decl(
        r"instance [Add α] [Add β] : Add (Prod α β) where
              add := fun x y => x",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance {
            name,
            binders,
            fields,
            ..
        } => {
            assert!(name.is_none());
            // Two instance binders
            assert_eq!(binders.len(), 2);
            assert_eq!(binders[0].info, SurfaceBinderInfo::Instance);
            assert_eq!(binders[1].info, SurfaceBinderInfo::Instance);
            assert_eq!(fields.len(), 1);
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_parse_instance_multiple_fields() {
    let decl = Parser::parse_decl(
        r"instance : Ord Nat where
              lt := Nat.lt
              le := Nat.le
              gt := Nat.gt",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance { fields, .. } => {
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0].name, "lt");
            assert_eq!(fields[1].name, "le");
            assert_eq!(fields[2].name, "gt");
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_parse_outparam() {
    // Test parsing outParam Type
    let expr = Parser::parse_expr("outParam Type").unwrap();
    match expr {
        SurfaceExpr::OutParam(_, inner) => {
            assert!(matches!(
                *inner,
                SurfaceExpr::Universe(_, UniverseExpr::Type)
            ));
        }
        _ => panic!("expected OutParam, got {expr:?}"),
    }
}

#[test]
fn test_parse_outparam_in_binder() {
    // Test parsing outParam in a binder context: (F : outParam Type)
    let expr = Parser::parse_expr("fun (F : outParam Type) => F").unwrap();
    match expr {
        SurfaceExpr::Lambda(_, binders, _) => {
            assert_eq!(binders.len(), 1);
            let binder_ty = binders[0].ty.as_ref().expect("binder should have type");
            assert!(
                matches!(**binder_ty, SurfaceExpr::OutParam(_, _)),
                "binder type should be OutParam"
            );
        }
        _ => panic!("expected Lambda"),
    }
}

#[test]
fn test_parse_class_with_outparam() {
    // Test parsing a class with an out-parameter
    let decl = Parser::parse_decl(
        r"class HAdd (α : Type) (β : Type) (γ : outParam Type) where
              hAdd : α → β → γ",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Class { name, binders, .. } => {
            assert_eq!(name, "HAdd");
            assert_eq!(binders.len(), 3);
            // First two parameters are regular Types
            assert!(matches!(
                binders[0].ty.as_ref().unwrap().as_ref(),
                SurfaceExpr::Universe(_, UniverseExpr::Type)
            ));
            assert!(matches!(
                binders[1].ty.as_ref().unwrap().as_ref(),
                SurfaceExpr::Universe(_, UniverseExpr::Type)
            ));
            // Third parameter is outParam Type
            assert!(
                matches!(
                    binders[2].ty.as_ref().unwrap().as_ref(),
                    SurfaceExpr::OutParam(_, _)
                ),
                "expected OutParam for third binder"
            );
        }
        _ => panic!("expected Class"),
    }
}

#[test]
fn test_parse_semioutparam() {
    // Test parsing semiOutParam Type
    let expr = Parser::parse_expr("semiOutParam Type").unwrap();
    match expr {
        SurfaceExpr::SemiOutParam(_, inner) => {
            assert!(matches!(
                *inner,
                SurfaceExpr::Universe(_, UniverseExpr::Type)
            ));
        }
        _ => panic!("expected SemiOutParam, got {expr:?}"),
    }
}

#[test]
fn test_parse_semioutparam_in_binder() {
    // Test parsing semiOutParam in a binder context: (F : semiOutParam Type)
    let expr = Parser::parse_expr("fun (F : semiOutParam Type) => F").unwrap();
    match expr {
        SurfaceExpr::Lambda(_, binders, _) => {
            assert_eq!(binders.len(), 1);
            let binder_ty = binders[0].ty.as_ref().expect("binder should have type");
            assert!(
                matches!(**binder_ty, SurfaceExpr::SemiOutParam(_, _)),
                "binder type should be SemiOutParam"
            );
        }
        _ => panic!("expected Lambda"),
    }
}

#[test]
fn test_parse_class_with_semioutparam() {
    // Test parsing a class with a semi-out-parameter (like Coe)
    let decl = Parser::parse_decl(
        r"class Coe (α : semiOutParam Type) (β : Type) where
              coe : α → β",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Class { name, binders, .. } => {
            assert_eq!(name, "Coe");
            assert_eq!(binders.len(), 2);
            // First parameter is semiOutParam Type
            assert!(
                matches!(
                    binders[0].ty.as_ref().unwrap().as_ref(),
                    SurfaceExpr::SemiOutParam(_, _)
                ),
                "expected SemiOutParam for first binder"
            );
            // Second parameter is regular Type
            assert!(matches!(
                binders[1].ty.as_ref().unwrap().as_ref(),
                SurfaceExpr::Universe(_, UniverseExpr::Type)
            ));
        }
        _ => panic!("expected Class"),
    }
}

#[test]
fn test_parse_class_with_both_param_types() {
    // Test parsing a class with both outParam and semiOutParam
    let decl = Parser::parse_decl(
        r"class HCoe (α : semiOutParam Type) (β : Type) (γ : outParam Type) where
              hCoe : α → β → γ",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Class { name, binders, .. } => {
            assert_eq!(name, "HCoe");
            assert_eq!(binders.len(), 3);
            // First parameter is semiOutParam
            assert!(
                matches!(
                    binders[0].ty.as_ref().unwrap().as_ref(),
                    SurfaceExpr::SemiOutParam(_, _)
                ),
                "expected SemiOutParam for first binder"
            );
            // Second parameter is regular Type
            assert!(matches!(
                binders[1].ty.as_ref().unwrap().as_ref(),
                SurfaceExpr::Universe(_, UniverseExpr::Type)
            ));
            // Third parameter is outParam
            assert!(
                matches!(
                    binders[2].ty.as_ref().unwrap().as_ref(),
                    SurfaceExpr::OutParam(_, _)
                ),
                "expected OutParam for third binder"
            );
        }
        _ => panic!("expected Class"),
    }
}

// =========================================================================
// Attribute parsing tests
// =========================================================================

#[test]
fn test_parse_instance_with_priority_attribute() {
    // @[instance 50] instance : Add Nat where ...
    let decl = Parser::parse_decl(
        r"@[instance 50] instance : Add Nat where
              add := Nat.add",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance { priority, .. } => {
            assert_eq!(priority, Some(50));
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_parse_instance_with_default_instance_attribute() {
    // @[defaultInstance] instance : ToString Nat where ...
    let decl = Parser::parse_decl(
        r"@[defaultInstance] instance : ToString Nat where
              toString := Nat.repr",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance { priority, .. } => {
            // defaultInstance sets priority to 0 (lowest)
            assert_eq!(priority, Some(0));
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_parse_instance_without_attribute() {
    // instance : Add Nat where ... (no attribute)
    let decl = Parser::parse_decl(
        r"instance : Add Nat where
              add := Nat.add",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance { priority, .. } => {
            // No attribute means no explicit priority
            assert_eq!(priority, None);
        }
        _ => panic!("expected Instance"),
    }
}

#[test]
fn test_parse_attribute_only_instance() {
    // @[instance] instance : Add Nat where ... (attribute without explicit number)
    let decl = Parser::parse_decl(
        r"@[instance] instance : Add Nat where
              add := Nat.add",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Instance { priority, .. } => {
            // @[instance] without number means default priority (100)
            assert_eq!(priority, Some(100));
        }
        _ => panic!("expected Instance"),
    }
}

// =========================================================================
// Deriving clause tests
// =========================================================================

#[test]
fn test_parse_structure_with_deriving_single() {
    let decl = Parser::parse_decl(
        r"structure Point where
              x : Nat
              y : Nat
            deriving Repr",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure {
            name,
            fields,
            deriving,
            ..
        } => {
            assert_eq!(name, "Point");
            assert_eq!(fields.len(), 2);
            assert_eq!(deriving.len(), 1);
            assert_eq!(deriving[0], "Repr");
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_parse_structure_with_deriving_multiple() {
    let decl = Parser::parse_decl(
        r"structure Point where
              x : Nat
              y : Nat
            deriving Repr, BEq, Hashable",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure { name, deriving, .. } => {
            assert_eq!(name, "Point");
            assert_eq!(deriving.len(), 3);
            assert_eq!(deriving[0], "Repr");
            assert_eq!(deriving[1], "BEq");
            assert_eq!(deriving[2], "Hashable");
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_parse_structure_without_deriving() {
    let decl = Parser::parse_decl(
        r"structure Point where
              x : Nat
              y : Nat",
    )
    .unwrap();
    match decl {
        SurfaceDecl::Structure { deriving, .. } => {
            assert!(deriving.is_empty());
        }
        _ => panic!("expected Structure"),
    }
}

#[test]
fn test_section_without_end() {
    // Test section without explicit `end` (valid in Lean 4)
    let code = r"section
def foo : Nat := 1
#eval 42
";
    let result = Parser::parse_file(code);
    assert!(
        result.is_ok(),
        "Section without end should parse: {:?}",
        result.err()
    );
}

#[test]
fn test_namespace_without_end() {
    // Test namespace without explicit `end` (valid in Lean 4)
    let code = r"namespace Foo
def bar : Nat := 1
#eval 42
";
    let result = Parser::parse_file(code);
    assert!(
        result.is_ok(),
        "Namespace without end should parse: {:?}",
        result.err()
    );
}

// ========================================================================
// Macro system parsing tests
// ========================================================================

#[test]
fn test_parse_syntax_simple() {
    let decl = Parser::parse_decl(r#"syntax term "+" term : term"#).unwrap();
    match decl {
        SurfaceDecl::Syntax {
            pattern, category, ..
        } => {
            assert_eq!(category, "term");
            assert_eq!(pattern.len(), 3); // term, "+", term
        }
        _ => panic!("expected Syntax"),
    }
}

#[test]
fn test_parse_syntax_with_precedence() {
    let decl = Parser::parse_decl(r#"syntax:50 term "+" term : term"#).unwrap();
    match decl {
        SurfaceDecl::Syntax {
            precedence,
            category,
            ..
        } => {
            assert_eq!(precedence, Some(50));
            assert_eq!(category, "term");
        }
        _ => panic!("expected Syntax"),
    }
}

#[test]
fn test_parse_syntax_with_name() {
    let decl = Parser::parse_decl(r#"syntax [myAdd] term "+" term : term"#).unwrap();
    match decl {
        SurfaceDecl::Syntax { name, category, .. } => {
            assert_eq!(name, Some("myAdd".to_string()));
            assert_eq!(category, "term");
        }
        _ => panic!("expected Syntax"),
    }
}

#[test]
fn test_parse_declare_syntax_cat() {
    let decl = Parser::parse_decl("declare_syntax_cat myCategory").unwrap();
    match decl {
        SurfaceDecl::DeclareSyntaxCat { name, .. } => {
            assert_eq!(name, "myCategory");
        }
        _ => panic!("expected DeclareSyntaxCat"),
    }
}

#[test]
fn test_parse_macro_simple() {
    let decl = Parser::parse_decl(r#"macro "hello" : term => x"#).unwrap();
    match decl {
        SurfaceDecl::Macro {
            pattern,
            category,
            expansion,
            ..
        } => {
            assert_eq!(pattern.len(), 1); // "hello"
            assert_eq!(category, "term");
            assert!(matches!(*expansion, SurfaceExpr::Ident(_, ref n) if n == "x"));
        }
        _ => panic!("expected Macro"),
    }
}

#[test]
fn test_parse_macro_with_variables() {
    let decl = Parser::parse_decl(r#"macro "unless" cond:term "then" body:term : term => x"#);
    // This is a bit tricky to parse, but should at least not error
    assert!(decl.is_ok(), "Should parse macro with variables");
}

#[test]
fn test_parse_macro_rules() {
    let decl = Parser::parse_decl(r"macro_rules | x => y | a => b").unwrap();
    match decl {
        SurfaceDecl::MacroRules { arms, .. } => {
            assert_eq!(arms.len(), 2);
        }
        _ => panic!("expected MacroRules"),
    }
}

#[test]
fn test_parse_notation_infixl() {
    let decl = Parser::parse_decl(r#"infixl:65 " + " => HAdd.hAdd"#).unwrap();
    match decl {
        SurfaceDecl::Notation {
            kind,
            precedence,
            pattern,
            ..
        } => {
            assert_eq!(kind, NotationKind::Infixl);
            assert_eq!(precedence, Some(65));
            assert!(!pattern.is_empty());
        }
        _ => panic!("expected Notation"),
    }
}

#[test]
fn test_parse_notation_prefix() {
    let decl = Parser::parse_decl(r#"prefix:100 "!" => Not"#).unwrap();
    match decl {
        SurfaceDecl::Notation {
            kind, precedence, ..
        } => {
            assert_eq!(kind, NotationKind::Prefix);
            assert_eq!(precedence, Some(100));
        }
        _ => panic!("expected Notation"),
    }
}

#[test]
fn test_parse_notation_general() {
    let decl = Parser::parse_decl(r#"notation a " ++ " b => List.append a b"#).unwrap();
    match decl {
        SurfaceDecl::Notation { kind, pattern, .. } => {
            assert_eq!(kind, NotationKind::Notation);
            // Should have: a, " ++ ", b
            assert!(pattern.len() >= 2);
        }
        _ => panic!("expected Notation"),
    }
}

#[test]
fn test_parse_file_with_macros() {
    // Test that a file with multiple macro declarations parses
    let code = r#"
syntax term "+" term : term
macro "hello" : term => x
infixl:65 " - " => HSub.hSub
def foo := 42
"#;
    let result = Parser::parse_file(code);
    assert!(
        result.is_ok(),
        "File with macros should parse: {:?}",
        result.err()
    );
    let decls = result.unwrap();
    // Should have at least 4 declarations
    assert!(
        decls.len() >= 3,
        "Expected at least 3 decls, got {}",
        decls.len()
    );
}
