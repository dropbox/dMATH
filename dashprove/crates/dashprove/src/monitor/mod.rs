//! Runtime monitor compilation
//!
//! Compiles USL specifications into runtime monitors that can check
//! invariants and properties at runtime.

mod codegen;
mod compile_expr;
mod config;
mod operators;
mod params;
mod property;
#[cfg(test)]
mod tests;
mod types;
mod utils;

pub use config::{MonitorConfig, MonitorTarget, MonitoredProperty, PropertyKind, RuntimeMonitor};

use codegen::{generate_check_code, generate_monitor_code_with_exprs};
use dashprove_usl::ast::Property;
use dashprove_usl::typecheck::TypedSpec;
use property::property_kind;

impl RuntimeMonitor {
    /// Create a new runtime monitor from a typed specification
    ///
    /// This compiles the specification into executable monitor code
    /// that can be embedded in applications.
    pub fn from_spec(spec: &TypedSpec, config: &MonitorConfig) -> Self {
        let name = spec
            .spec
            .properties
            .first()
            .map(Property::name)
            .unwrap_or_else(|| "monitor".to_string());

        let properties: Vec<MonitoredProperty> = spec
            .spec
            .properties
            .iter()
            .map(|p| {
                let prop_name = p.name();
                let kind = property_kind(p);
                let check_code = generate_check_code(&prop_name, config);
                MonitoredProperty {
                    check_code,
                    name: prop_name,
                    kind,
                }
            })
            .collect();

        let code = generate_monitor_code_with_exprs(&name, &spec.spec.properties, config);

        RuntimeMonitor {
            name,
            code,
            properties,
            target: config.target.clone(),
        }
    }

    /// Get the number of properties being monitored
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }
}
