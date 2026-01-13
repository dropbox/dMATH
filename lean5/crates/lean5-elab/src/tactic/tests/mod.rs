use super::*;
use lean5_kernel::env::Declaration;

pub(super) fn setup_env() -> Environment {
    let mut env = Environment::new();

    // Add a simple type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("A"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    // Add a term of that type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("a"),
        level_params: vec![],
        type_: Expr::const_(Name::from_string("A"), vec![]),
    })
    .unwrap();

    // Add another type
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("B"),
        level_params: vec![],
        type_: Expr::type_(),
    })
    .unwrap();

    // Add a function A → B
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("f"),
        level_params: vec![],
        type_: Expr::arrow(
            Expr::const_(Name::from_string("A"), vec![]),
            Expr::const_(Name::from_string("B"), vec![]),
        ),
    })
    .unwrap();

    env
}

pub(super) fn setup_env_with_and_or() -> Environment {
    let mut env = Environment::new();
    env.init_and().unwrap();
    env.init_classical().unwrap();

    let prop = Expr::prop();

    // Propositions
    for name in ["P", "Q"] {
        env.add_decl(Declaration::Axiom {
            name: Name::from_string(name),
            level_params: vec![],
            type_: prop.clone(),
        })
        .unwrap();
    }

    // Proof witnesses
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("p"),
        level_params: vec![],
        type_: Expr::const_(Name::from_string("P"), vec![]),
    })
    .unwrap();
    env.add_decl(Declaration::Axiom {
        name: Name::from_string("q"),
        level_params: vec![],
        type_: Expr::const_(Name::from_string("Q"), vec![]),
    })
    .unwrap();

    env
}

pub(super) fn setup_env_with_nat() -> Environment {
    let mut env = Environment::new();
    env.init_nat().unwrap();
    env
}

mod advanced;
mod arithmetic;
mod core;
mod simp;
