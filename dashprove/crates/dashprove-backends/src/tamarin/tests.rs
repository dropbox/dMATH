//! Tests for Tamarin backend

use super::parsing::parse_output;
use super::theory::{
    compile_tamarin_term, extract_security_patterns, generate_theory, to_tamarin_ident,
};
use super::TamarinBackend;
use crate::traits::{PropertyType, VerificationBackend, VerificationStatus};
use dashprove_usl::ast::Expr;
use dashprove_usl::typecheck::{typecheck, TypedSpec};
use dashprove_usl::{parse, Spec};
use std::collections::HashSet;

fn create_typed_spec(input: &str) -> TypedSpec {
    let spec = parse(input).expect("should parse");
    typecheck(spec).expect("should typecheck")
}

#[test]
fn parse_verified_output() {
    let status = parse_output("lemma verified\nanalysis complete\n", "");
    assert!(matches!(
        status,
        VerificationStatus::Proven | VerificationStatus::Partial { .. }
    ));
}

#[test]
fn parse_attack_output() {
    let status = parse_output("attack found\n", "");
    assert!(matches!(status, VerificationStatus::Disproven));
}

#[test]
fn test_generate_theory_confidentiality() {
    let input = r#"
        security confidentiality {
            forall s: Secret, a: Agent .
                not authorized(a, s) implies not knows(a, s.content)
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    // Check basic structure
    assert!(theory.contains("theory DashProve"));
    assert!(theory.contains("begin"));
    assert!(theory.contains("end"));

    // Check builtins
    assert!(theory.contains("builtins:"));

    // Check rules are generated
    assert!(theory.contains("rule Generate_Key:"));
    assert!(theory.contains("rule Protocol_Send:"));

    // Check for secret rules based on extracted secrets
    assert!(theory.contains("Secret_content") || theory.contains("Secret("));

    // Check lemma is generated
    assert!(theory.contains("lemma confidentiality"));
}

#[test]
fn test_generate_theory_tenant_isolation() {
    let input = r#"
        type Tenant = { id: String }

        security tenant_isolation {
            forall t1: Tenant, t2: Tenant .
                t1.id != t2.id implies not can_observe(t1, actions(t2))
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    // Check type comment is included
    assert!(theory.contains("Type Tenant"));

    // Check functions are declared (can_observe and actions)
    // These are filtered as known predicates, so check the theory compiles
    assert!(theory.contains("theory DashProve"));
    assert!(theory.contains("lemma tenant_isolation"));
}

#[test]
fn test_generate_theory_integrity() {
    let input = r#"
        security data_integrity {
            forall r: Record, a: Agent .
                modified(a, r) implies authorized(a, r)
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    // Check correspondence lemma is generated
    // `modified implies authorized` generates a lemma checking that all modifications
    // are preceded by authorization
    assert!(theory.contains("lemma data_integrity"));
    assert!(theory.contains("Modified(a, r)"));
    assert!(theory.contains("Authorized(a, r)"));
}

#[test]
fn test_generate_theory_integrity_rules() {
    // This test uses `authorized implies modified` pattern which triggers
    // integrity rules generation (authorize before modify workflow)
    let input = r#"
        security can_modify {
            forall r: Record, a: Agent .
                authorized(a, r) implies modified(a, r)
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    // Check integrity rules are generated
    assert!(theory.contains("rule Authorize:"));
    assert!(theory.contains("rule Modify:"));
    assert!(theory.contains("AuthToken"));

    // Check lemma is generated
    assert!(theory.contains("lemma can_modify"));
}

#[test]
fn test_generate_theory_authentication() {
    let input = r#"
        security authentication_required {
            forall a: Action, u: User .
                performed(u, a) implies authenticated(u)
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    // Check authentication lemma
    assert!(theory.contains("lemma authentication_required"));
}

#[test]
fn test_generate_theory_noninterference() {
    let input = r#"
        security noninterference {
            forall h1: HighInput, h2: HighInput, l: LowInput .
                output(h1, l) == output(h2, l)
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    // Check non-interference handling
    assert!(theory.contains("lemma noninterference"));
    assert!(theory.contains("Non-interference"));
}

#[test]
fn test_generate_theory_multiple_properties() {
    let input = r#"
        security prop1 {
            forall x: Int . true
        }

        security prop2 {
            forall y: Bool . true
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    // Check both properties generate lemmas
    assert!(theory.contains("lemma prop1"));
    assert!(theory.contains("lemma prop2"));
}

#[test]
fn test_to_tamarin_ident() {
    // Test identifier cleaning
    assert_eq!(to_tamarin_ident("Agent"), "Agent");
    assert_eq!(to_tamarin_ident("secret_key"), "secret_key");
    assert_eq!(to_tamarin_ident("myVar123"), "myVar123");
    assert_eq!(to_tamarin_ident(""), "x");
}

#[test]
fn test_compile_tamarin_term() {
    // Test variable
    let var = Expr::Var("secret".to_string());
    assert_eq!(compile_tamarin_term(&var), "secret");

    // Test field access
    let field = Expr::FieldAccess(Box::new(Expr::Var("s".to_string())), "content".to_string());
    assert_eq!(compile_tamarin_term(&field), "s_content");

    // Test function application
    let func = Expr::App("hash".to_string(), vec![Expr::Var("msg".to_string())]);
    assert_eq!(compile_tamarin_term(&func), "hash(msg)");
}

#[test]
fn test_extract_secrets_from_knows_pattern() {
    let mut secrets = HashSet::new();
    let mut integrity = Vec::new();

    // `not knows(agent, secretVar)` should extract secretVar
    let expr = Expr::Not(Box::new(Expr::App(
        "knows".to_string(),
        vec![
            Expr::Var("agent".to_string()),
            Expr::Var("secretVar".to_string()),
        ],
    )));

    extract_security_patterns(&expr, &mut secrets, &mut integrity);
    assert!(secrets.contains("secretVar"));
}

#[test]
fn test_extract_secrets_from_field_access() {
    let mut secrets = HashSet::new();
    let mut integrity = Vec::new();

    // `not knows(a, s.content)` should extract "content"
    let expr = Expr::Not(Box::new(Expr::App(
        "knows".to_string(),
        vec![
            Expr::Var("a".to_string()),
            Expr::FieldAccess(Box::new(Expr::Var("s".to_string())), "content".to_string()),
        ],
    )));

    extract_security_patterns(&expr, &mut secrets, &mut integrity);
    assert!(secrets.contains("content"));
}

#[test]
fn test_extract_integrity_predicates() {
    let mut secrets = HashSet::new();
    let mut integrity = Vec::new();

    // `authorized(a, r) implies modified(a, r)` should extract integrity predicate
    let expr = Expr::Implies(
        Box::new(Expr::App(
            "authorized".to_string(),
            vec![Expr::Var("a".to_string()), Expr::Var("r".to_string())],
        )),
        Box::new(Expr::App(
            "modified".to_string(),
            vec![Expr::Var("a".to_string()), Expr::Var("r".to_string())],
        )),
    );

    extract_security_patterns(&expr, &mut secrets, &mut integrity);
    assert_eq!(integrity.len(), 1);
    assert_eq!(integrity[0], ("a".to_string(), "r".to_string()));
}

#[test]
fn test_theory_without_security_properties() {
    // Test that an empty spec still generates a valid theory
    let typed_spec = TypedSpec {
        spec: Spec::default(),
        type_info: Default::default(),
    };
    let theory = generate_theory(&typed_spec);

    // Should still have basic structure
    assert!(theory.contains("theory DashProve"));
    assert!(theory.contains("begin"));
    assert!(theory.contains("end"));
    assert!(theory.contains("builtins:"));
    assert!(theory.contains("rule Generate_Key:"));
    // Should have default secrecy lemma
    assert!(theory.contains("lemma secrecy"));
}

#[test]
fn test_crypto_builtins_selection() {
    // Test that hash functions trigger hashing builtin
    let input = r#"
        security hash_property {
            forall m: Message . hash(m) == hash(m)
        }
    "#;
    let typed_spec = create_typed_spec(input);
    let theory = generate_theory(&typed_spec);

    assert!(theory.contains("hashing"));
}

#[test]
fn tamarin_backend_supports_security_protocol() {
    let backend = TamarinBackend::new();
    let supports = backend.supports();
    assert!(supports.contains(&PropertyType::SecurityProtocol));
}
