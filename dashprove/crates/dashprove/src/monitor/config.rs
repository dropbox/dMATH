//! Configuration types for runtime monitor generation

use serde::{Deserialize, Serialize};

// Kani proofs for configuration types
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify MonitorConfig default has all flags set to false
    #[kani::proof]
    fn verify_monitor_config_default_all_false() {
        let config = MonitorConfig::default();
        kani::assert(
            !config.generate_assertions,
            "default assertions should be false",
        );
        kani::assert(!config.generate_logging, "default logging should be false");
        kani::assert(!config.generate_metrics, "default metrics should be false");
    }

    /// Verify MonitorTarget default is Rust
    #[kani::proof]
    fn verify_monitor_target_default_is_rust() {
        let target = MonitorTarget::default();
        kani::assert(
            matches!(target, MonitorTarget::Rust),
            "default target should be Rust",
        );
    }

    /// Verify MonitorTarget variants are distinct
    #[kani::proof]
    fn verify_monitor_target_variants_distinct() {
        let rust = MonitorTarget::Rust;
        let ts = MonitorTarget::TypeScript;
        let py = MonitorTarget::Python;

        // Each variant is distinct
        kani::assert(
            !matches!(rust, MonitorTarget::TypeScript),
            "Rust != TypeScript",
        );
        kani::assert(!matches!(rust, MonitorTarget::Python), "Rust != Python");
        kani::assert(!matches!(ts, MonitorTarget::Rust), "TypeScript != Rust");
        kani::assert(!matches!(ts, MonitorTarget::Python), "TypeScript != Python");
        kani::assert(!matches!(py, MonitorTarget::Rust), "Python != Rust");
        kani::assert(
            !matches!(py, MonitorTarget::TypeScript),
            "Python != TypeScript",
        );
    }

    /// Verify PropertyKind variants are distinct
    #[kani::proof]
    fn verify_property_kind_variants_distinct() {
        let inv = PropertyKind::Invariant;
        let pre = PropertyKind::Precondition;
        let post = PropertyKind::Postcondition;
        let temp = PropertyKind::Temporal;

        // Invariant is distinct from others
        kani::assert(!matches!(inv, PropertyKind::Precondition), "Inv != Pre");
        kani::assert(!matches!(inv, PropertyKind::Postcondition), "Inv != Post");
        kani::assert(!matches!(inv, PropertyKind::Temporal), "Inv != Temp");

        // Precondition is distinct from others
        kani::assert(!matches!(pre, PropertyKind::Invariant), "Pre != Inv");
        kani::assert(!matches!(pre, PropertyKind::Postcondition), "Pre != Post");
        kani::assert(!matches!(pre, PropertyKind::Temporal), "Pre != Temp");

        // Postcondition is distinct from others
        kani::assert(!matches!(post, PropertyKind::Invariant), "Post != Inv");
        kani::assert(!matches!(post, PropertyKind::Precondition), "Post != Pre");
        kani::assert(!matches!(post, PropertyKind::Temporal), "Post != Temp");

        // Temporal is distinct from others
        kani::assert(!matches!(temp, PropertyKind::Invariant), "Temp != Inv");
        kani::assert(!matches!(temp, PropertyKind::Precondition), "Temp != Pre");
        kani::assert(!matches!(temp, PropertyKind::Postcondition), "Temp != Post");
    }

    /// Verify MonitoredProperty name is preserved in construction
    #[kani::proof]
    fn verify_monitored_property_name_preserved() {
        // Use a small bounded string for verification
        let name = String::from("test_prop");
        let kind = PropertyKind::Invariant;
        let check_code = String::from("check()");

        let prop = MonitoredProperty {
            name: name.clone(),
            kind,
            check_code: check_code.clone(),
        };

        kani::assert(prop.name == name, "name should be preserved");
        kani::assert(
            prop.check_code == check_code,
            "check_code should be preserved",
        );
    }

    /// Verify RuntimeMonitor property_count is consistent with properties vec length
    #[kani::proof]
    fn verify_runtime_monitor_property_count() {
        let monitor = RuntimeMonitor {
            name: String::from("test"),
            code: String::from("// code"),
            properties: vec![],
            target: MonitorTarget::default(),
        };

        kani::assert(
            monitor.property_count() == monitor.properties.len(),
            "property_count should equal properties.len()",
        );
    }
}

/// Configuration for runtime monitor generation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Generate assertions (panics on violation)
    pub generate_assertions: bool,
    /// Generate logging on property checks
    pub generate_logging: bool,
    /// Generate metrics/counters for property checks
    pub generate_metrics: bool,
    /// Target language for generated monitor code
    pub target: MonitorTarget,
}

/// Target language for runtime monitors
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum MonitorTarget {
    /// Rust code with assertions
    #[default]
    Rust,
    /// TypeScript code for Node.js
    TypeScript,
    /// Python code
    Python,
}

/// A compiled runtime monitor
///
/// Runtime monitors are generated from USL specifications and can be
/// embedded in applications to check invariants at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMonitor {
    /// Name of the monitor (from spec)
    pub name: String,
    /// Generated code for the monitor
    pub code: String,
    /// Properties being monitored
    pub properties: Vec<MonitoredProperty>,
    /// Target language
    pub target: MonitorTarget,
}

/// A property being monitored at runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoredProperty {
    /// Property name
    pub name: String,
    /// Property type (invariant, postcondition, etc.)
    pub kind: PropertyKind,
    /// Generated check code
    pub check_code: String,
}

/// Kind of property being monitored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyKind {
    /// Must hold at all times
    Invariant,
    /// Must hold before operation
    Precondition,
    /// Must hold after operation
    Postcondition,
    /// Must hold for state transitions
    Temporal,
}
