//! Tests for runtime monitor generation

use crate::monitor::config::{MonitorConfig, MonitorTarget, RuntimeMonitor};
use dashprove_usl::{parse, typecheck};

#[test]
fn test_monitor_config_default() {
    let config = MonitorConfig::default();
    assert!(!config.generate_assertions);
    assert!(!config.generate_logging);
    assert!(matches!(config.target, MonitorTarget::Rust));
}

#[test]
fn test_monitor_from_spec() {
    let spec = parse("theorem safety { forall x: Bool . x or not x }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        generate_assertions: true,
        ..Default::default()
    };

    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert_eq!(monitor.name, "safety");
    assert_eq!(monitor.property_count(), 1);
    assert!(monitor.code.contains("SafetyMonitor"));
    assert!(monitor.code.contains("check_safety"));
}

#[test]
fn test_monitor_rust_code() {
    let spec = parse("theorem test { true }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        generate_assertions: true,
        target: MonitorTarget::Rust,
        ..Default::default()
    };

    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("pub struct TestMonitor"));
    assert!(monitor.code.contains("pub fn check_test"));
    assert!(monitor.code.contains("pub fn check_all"));
}

#[test]
fn test_monitor_typescript_code() {
    let spec = parse("theorem my_property { true }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::TypeScript,
        ..Default::default()
    };

    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("export class MyPropertyMonitor"));
    assert!(monitor.code.contains("checkMyProperty()"));
    assert!(monitor.code.contains("checkAll()"));
}

#[test]
fn test_monitor_python_code() {
    let spec = parse("theorem example { true }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::Python,
        ..Default::default()
    };

    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("class ExampleMonitor"));
    assert!(monitor.code.contains("def check_example(self)"));
    assert!(monitor.code.contains("def check_all(self)"));
}

#[test]
fn test_monitor_multiple_properties() {
    let spec = parse(
        r#"
        theorem prop1 { true }
        theorem prop2 { true }
        "#,
    )
    .unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig::default();
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert_eq!(monitor.property_count(), 2);
    assert!(monitor.code.contains("check_prop1"));
    assert!(monitor.code.contains("check_prop2"));
}

#[test]
fn test_monitor_contract_generates_rust_checks() {
    let spec = parse(
        r#"
        contract divide(x: Int, y: Int) -> Result<Int> {
            requires { y != 0 }
            ensures { result * y == x }
            ensures_err { y == 0 }
        }
        "#,
    )
    .unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig::default();
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("check_divide_requires"));
    assert!(monitor.code.contains("check_divide_ensures"));
    assert!(monitor.code.contains("check_divide_ensures_err"));
    assert!(monitor.code.contains("(y != 0)"));
    assert!(monitor.code.contains("result * y"));
}

#[test]
fn test_monitor_contract_typescript_handles_self_param() {
    let spec = parse(
        r#"
        contract Stack::push(self: Stack, value: Int) -> Result<Stack> {
            requires { self.capacity > value }
            ensures { result.capacity >= self.capacity }
        }
        "#,
    )
    .unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::TypeScript,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("checkStackPushRequires"));
    assert!(monitor.code.contains("selfValue"));
    assert!(monitor.code.contains("result.capacity"));
}

#[test]
fn test_monitor_contract_python_output() {
    let spec = parse(
        r#"
        contract Account::deposit(self: Account, amount: Int) -> Result<Int> {
            requires { amount > 0 }
            ensures { result >= amount }
        }
        "#,
    )
    .unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::Python,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("def check_account_deposit_requires"));
    assert!(monitor.code.contains("self_value, amount"));
    assert!(monitor.code.contains("result >= amount"));
}

#[test]
fn test_monitor_generates_real_code() {
    // Test that monitor generates real executable check code, not stubs
    let spec = parse("theorem excluded_middle { forall x: Bool . x or not x }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig::default();
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    // Should contain actual logic, not TODO stubs
    assert!(!monitor.code.contains("// TODO:"));
    assert!(monitor.code.contains("[false, true]"));
    assert!(monitor.code.contains(".all(|x|"));
    assert!(monitor.code.contains("(x || !(x))"));
}

#[test]
fn test_monitor_typescript_generates_real_code() {
    let spec = parse("theorem test { forall x: Bool . x or not x }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::TypeScript,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("[false, true].every"));
    assert!(monitor.code.contains("(x || !(x))"));
}

#[test]
fn test_monitor_python_generates_real_code() {
    let spec = parse("theorem test { forall x: Bool . x or not x }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig {
        target: MonitorTarget::Python,
        ..Default::default()
    };
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    assert!(monitor.code.contains("all("));
    assert!(monitor.code.contains("[False, True]"));
    assert!(monitor.code.contains("(x or not (x))"));
}

#[test]
fn test_compile_nested_quantifiers() {
    // forall p: Bool . forall q: Bool . (p and q) implies q
    let spec =
        parse("theorem test { forall p: Bool . forall q: Bool . (p and q) implies q }").unwrap();
    let typed_spec = typecheck(spec).unwrap();

    let config = MonitorConfig::default();
    let monitor = RuntimeMonitor::from_spec(&typed_spec, &config);

    // Should have nested iterators
    assert!(monitor.code.contains(".all(|p|"));
    assert!(monitor.code.contains(".all(|q|"));
}
