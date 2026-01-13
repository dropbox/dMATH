use tla_core::ast::{Expr, Unit};
use tla_core::{lower, parse_to_syntax_tree, FileId};

#[test]
fn lower_action_subscripts_expand_to_unchanged_sugar() {
    let source = r#"
---- MODULE Main ----
Sub == [Next]_vars
Angle == <<Next>>_vars
SubMod == [M!Next]_M!vars
====
"#;

    let tree = parse_to_syntax_tree(source);
    let lower_result = lower(FileId(0), &tree);

    assert!(
        lower_result.errors.is_empty(),
        "lower errors: {:?}",
        lower_result.errors
    );

    let module = lower_result.module.expect("lower produced no module");

    let mut saw_sub = false;
    let mut saw_angle = false;
    let mut saw_sub_mod = false;

    for unit in &module.units {
        let Unit::Operator(def) = &unit.node else {
            continue;
        };

        match def.name.node.as_str() {
            "Sub" => {
                let Expr::Or(action, unchanged) = &def.body.node else {
                    panic!("expected Sub body to be Or, got: {:?}", def.body.node);
                };
                let Expr::Ident(name) = &action.node else {
                    panic!("expected Sub action to be Ident, got: {:?}", action.node);
                };
                assert_eq!(name, "Next");
                let Expr::Unchanged(sub) = &unchanged.node else {
                    panic!(
                        "expected Sub rhs to be Unchanged, got: {:?}",
                        unchanged.node
                    );
                };
                let Expr::Ident(v) = &sub.node else {
                    panic!(
                        "expected Sub UNCHANGED arg to be Ident, got: {:?}",
                        sub.node
                    );
                };
                assert_eq!(v, "vars");
                saw_sub = true;
            }
            "Angle" => {
                let Expr::And(action, not_unchanged) = &def.body.node else {
                    panic!("expected Angle body to be And, got: {:?}", def.body.node);
                };
                let Expr::Ident(name) = &action.node else {
                    panic!("expected Angle action to be Ident, got: {:?}", action.node);
                };
                assert_eq!(name, "Next");
                let Expr::Not(inner) = &not_unchanged.node else {
                    panic!(
                        "expected Angle rhs to be Not, got: {:?}",
                        not_unchanged.node
                    );
                };
                let Expr::Unchanged(sub) = &inner.node else {
                    panic!(
                        "expected Angle rhs inner to be Unchanged, got: {:?}",
                        inner.node
                    );
                };
                let Expr::Ident(v) = &sub.node else {
                    panic!(
                        "expected Angle UNCHANGED arg to be Ident, got: {:?}",
                        sub.node
                    );
                };
                assert_eq!(v, "vars");
                saw_angle = true;
            }
            "SubMod" => {
                let Expr::Or(action, unchanged) = &def.body.node else {
                    panic!("expected SubMod body to be Or, got: {:?}", def.body.node);
                };
                let Expr::ModuleRef(m, op, args) = &action.node else {
                    panic!(
                        "expected SubMod action to be ModuleRef, got: {:?}",
                        action.node
                    );
                };
                assert_eq!(m.name(), "M");
                assert_eq!(op, "Next");
                assert!(args.is_empty());
                let Expr::Unchanged(sub) = &unchanged.node else {
                    panic!(
                        "expected SubMod rhs to be Unchanged, got: {:?}",
                        unchanged.node
                    );
                };
                let Expr::ModuleRef(m2, op2, args2) = &sub.node else {
                    panic!(
                        "expected SubMod UNCHANGED arg to be ModuleRef, got: {:?}",
                        sub.node
                    );
                };
                assert_eq!(m2.name(), "M");
                assert_eq!(op2, "vars");
                assert!(args2.is_empty());
                saw_sub_mod = true;
            }
            _ => {}
        }
    }

    assert!(saw_sub, "did not find operator Sub");
    assert!(saw_angle, "did not find operator Angle");
    assert!(saw_sub_mod, "did not find operator SubMod");
}
