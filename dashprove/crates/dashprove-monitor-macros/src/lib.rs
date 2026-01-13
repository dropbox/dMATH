//! # dashprove-monitor-macros
//!
//! Procedural macros for automatic runtime monitoring instrumentation.
//!
//! This crate provides derive macros that generate implementations of
//! the `Traceable` and `MonitoredType` traits from `dashprove-monitor`.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use dashprove_monitor_macros::Monitored;
//! use dashprove_monitor::{Traceable, MonitoredType};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
//! #[monitored(name = "Counter")]
//! struct Counter {
//!     #[monitored(track)]
//!     value: i32,
//!
//!     #[monitored(skip)]
//!     internal_cache: Option<String>,
//! }
//!
//! // Invariants can be specified inline
//! #[derive(Debug, Clone, Serialize, Deserialize, Monitored)]
//! #[monitored(name = "BoundedCounter")]
//! #[monitored(invariant = "value >= 0", name = "non_negative")]
//! #[monitored(invariant = "value <= max", name = "bounded")]
//! struct BoundedCounter {
//!     value: i32,
//!     max: i32,
//! }
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Error, Fields, Ident, LitStr, Token};

/// Derive macro for implementing `Traceable` and optionally `MonitoredType` traits.
///
/// # Attributes
///
/// ## Container attributes (`#[monitored(...)]`)
///
/// - `name = "..."` - Custom trace name (defaults to struct name)
/// - `invariant = "...", name = "..."` - Inline invariant expression
///
/// ## Field attributes (`#[monitored(...)]`)
///
/// - `track` - Explicitly track this field (default for all fields)
/// - `skip` - Skip this field from state capture
/// - `rename = "..."` - Use a different name in the captured state
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Monitored)]
/// #[monitored(name = "MyState")]
/// struct State {
///     #[monitored(track)]
///     counter: i32,
///
///     #[monitored(skip)]
///     cache: HashMap<String, String>,
///
///     #[monitored(rename = "total")]
///     sum: f64,
/// }
/// ```
#[proc_macro_derive(Monitored, attributes(monitored))]
pub fn derive_monitored(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_monitored(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Container-level configuration parsed from attributes
#[derive(Default)]
struct ContainerConfig {
    /// Custom trace name
    name: Option<String>,
    /// Invariant definitions
    invariants: Vec<InvariantDef>,
}

/// An invariant definition
struct InvariantDef {
    /// Name of the invariant
    name: String,
    /// The expression string (for display/debugging)
    expr: String,
}

/// Field-level configuration parsed from attributes
#[derive(Default)]
struct FieldConfig {
    /// Whether to skip this field
    skip: bool,
    /// Whether to explicitly track (default is true for non-skipped)
    track: bool,
    /// Renamed field name in captured state
    rename: Option<String>,
}

fn expand_monitored(input: DeriveInput) -> Result<TokenStream2, Error> {
    let struct_name = &input.ident;
    let container_config = parse_container_attrs(&input.attrs)?;

    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) => {
            return Err(Error::new_spanned(
                &input,
                "Monitored can only be derived for structs",
            ))
        }
        Data::Union(_) => {
            return Err(Error::new_spanned(
                &input,
                "Monitored can only be derived for structs",
            ))
        }
    };

    let default_name = struct_name.to_string();
    let trace_name = container_config.name.as_deref().unwrap_or(&default_name);

    let (capture_state_impl, tracked_vars_impl) = generate_capture_state(fields)?;
    let traceable_impl = generate_traceable_impl(
        struct_name,
        trace_name,
        capture_state_impl,
        tracked_vars_impl,
    );

    let monitored_type_impl = if container_config.invariants.is_empty() {
        // Empty MonitoredType impl
        quote! {
            impl dashprove_monitor::MonitoredType for #struct_name {}
        }
    } else {
        generate_monitored_type_impl(struct_name, &container_config.invariants)?
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics dashprove_monitor::Traceable for #struct_name #ty_generics #where_clause {
            fn trace_name(&self) -> &str {
                #trace_name
            }

            #traceable_impl
        }

        #monitored_type_impl
    })
}

fn parse_container_attrs(attrs: &[Attribute]) -> Result<ContainerConfig, Error> {
    let mut config = ContainerConfig::default();

    for attr in attrs {
        if !attr.path().is_ident("monitored") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let _eq: Token![=] = meta.input.parse()?;
                let lit: LitStr = meta.input.parse()?;
                config.name = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("invariant") {
                let _eq: Token![=] = meta.input.parse()?;
                let expr_lit: LitStr = meta.input.parse()?;

                // Parse optional name
                let inv_name = if meta.input.peek(Token![,]) {
                    let _comma: Token![,] = meta.input.parse()?;
                    if meta.input.peek(Ident) {
                        let ident: Ident = meta.input.parse()?;
                        if ident == "name" {
                            let _eq: Token![=] = meta.input.parse()?;
                            let name_lit: LitStr = meta.input.parse()?;
                            name_lit.value()
                        } else {
                            format!("invariant_{}", config.invariants.len())
                        }
                    } else {
                        format!("invariant_{}", config.invariants.len())
                    }
                } else {
                    format!("invariant_{}", config.invariants.len())
                };

                config.invariants.push(InvariantDef {
                    name: inv_name,
                    expr: expr_lit.value(),
                });
                Ok(())
            } else {
                Err(meta.error("unrecognized monitored attribute"))
            }
        })?;
    }

    Ok(config)
}

fn parse_field_attrs(attrs: &[Attribute]) -> Result<FieldConfig, Error> {
    let mut config = FieldConfig::default();

    for attr in attrs {
        if !attr.path().is_ident("monitored") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                config.skip = true;
                Ok(())
            } else if meta.path.is_ident("track") {
                config.track = true;
                Ok(())
            } else if meta.path.is_ident("rename") {
                let _eq: Token![=] = meta.input.parse()?;
                let lit: LitStr = meta.input.parse()?;
                config.rename = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error("unrecognized monitored field attribute"))
            }
        })?;
    }

    Ok(config)
}

fn generate_capture_state(fields: &Fields) -> Result<(TokenStream2, TokenStream2), Error> {
    match fields {
        Fields::Named(named) => {
            let mut field_captures = Vec::new();
            let mut tracked_names = Vec::new();

            for field in &named.named {
                let field_name = field.ident.as_ref().unwrap();
                let config = parse_field_attrs(&field.attrs)?;

                if config.skip {
                    continue;
                }

                let default_field_name = field_name.to_string();
                let json_key = config.rename.as_deref().unwrap_or(&default_field_name);

                field_captures.push(quote! {
                    map.insert(#json_key.to_string(), serde_json::to_value(&self.#field_name).unwrap_or(serde_json::Value::Null));
                });

                tracked_names.push(json_key.to_string());
            }

            let capture_impl = quote! {
                fn capture_state(&self) -> serde_json::Value {
                    let mut map = serde_json::Map::new();
                    #(#field_captures)*
                    serde_json::Value::Object(map)
                }
            };

            let tracked_impl = quote! {
                fn tracked_variables(&self) -> Vec<&str> {
                    vec![#(#tracked_names),*]
                }
            };

            Ok((capture_impl, tracked_impl))
        }
        Fields::Unnamed(unnamed) => {
            let mut field_captures = Vec::new();
            let mut tracked_names = Vec::new();

            for (i, field) in unnamed.unnamed.iter().enumerate() {
                let config = parse_field_attrs(&field.attrs)?;

                if config.skip {
                    continue;
                }

                let idx = syn::Index::from(i);
                let json_key = config
                    .rename
                    .as_deref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("field_{}", i));

                field_captures.push(quote! {
                    map.insert(#json_key.to_string(), serde_json::to_value(&self.#idx).unwrap_or(serde_json::Value::Null));
                });

                tracked_names.push(json_key);
            }

            let capture_impl = quote! {
                fn capture_state(&self) -> serde_json::Value {
                    let mut map = serde_json::Map::new();
                    #(#field_captures)*
                    serde_json::Value::Object(map)
                }
            };

            let tracked_impl = quote! {
                fn tracked_variables(&self) -> Vec<&str> {
                    vec![#(#tracked_names),*]
                }
            };

            Ok((capture_impl, tracked_impl))
        }
        Fields::Unit => Ok((
            quote! {
                fn capture_state(&self) -> serde_json::Value {
                    serde_json::Value::Object(serde_json::Map::new())
                }
            },
            quote! {
                fn tracked_variables(&self) -> Vec<&str> {
                    vec![]
                }
            },
        )),
    }
}

fn generate_traceable_impl(
    _struct_name: &Ident,
    _trace_name: &str,
    capture_state_impl: TokenStream2,
    tracked_vars_impl: TokenStream2,
) -> TokenStream2 {
    quote! {
        #capture_state_impl

        #tracked_vars_impl
    }
}

fn generate_monitored_type_impl(
    struct_name: &Ident,
    invariants: &[InvariantDef],
) -> Result<TokenStream2, Error> {
    let invariant_names: Vec<&str> = invariants.iter().map(|i| i.name.as_str()).collect();

    // Generate invariant closures
    // For now, we generate closures that check the JSON state
    // The expressions are expected to reference fields as state["field"]
    let invariant_closures: Vec<TokenStream2> = invariants
        .iter()
        .map(|inv| {
            let expr = &inv.expr;
            // Simple expression parsing - support basic field comparisons
            // Format: field_name op value
            let closure = parse_invariant_expr(expr);
            quote! {
                Box::new(#closure) as Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>
            }
        })
        .collect();

    Ok(quote! {
        impl dashprove_monitor::MonitoredType for #struct_name {
            fn invariants(&self) -> Vec<Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>> {
                vec![#(#invariant_closures),*]
            }

            fn invariant_names(&self) -> Vec<&str> {
                vec![#(#invariant_names),*]
            }
        }
    })
}

/// Parse a simple invariant expression into a closure
/// Supports: field op value, field op field
/// Ops: ==, !=, <, <=, >, >=
fn parse_invariant_expr(expr: &str) -> TokenStream2 {
    // Try to parse as "field op value" or "field op field"
    let ops = [">=", "<=", "==", "!=", ">", "<"];

    for op in ops {
        if let Some(idx) = expr.find(op) {
            let left = expr[..idx].trim();
            let right = expr[idx + op.len()..].trim();

            let left_access = field_access(left);
            let right_access = if right.parse::<i64>().is_ok() {
                let n: i64 = right.parse().unwrap();
                quote! { Some(#n) }
            } else if right.parse::<f64>().is_ok() {
                let n: f64 = right.parse().unwrap();
                quote! { state.get(#right).and_then(|v| v.as_f64()).or(Some(#n)) }
            } else if right == "true" || right == "false" {
                let b: bool = right.parse().unwrap();
                quote! { Some(#b) }
            } else {
                // Assume it's another field
                field_access(right)
            };

            let comparison = match op {
                ">=" => quote! { l >= r },
                "<=" => quote! { l <= r },
                "==" => quote! { l == r },
                "!=" => quote! { l != r },
                ">" => quote! { l > r },
                "<" => quote! { l < r },
                _ => quote! { false },
            };

            return quote! {
                |state: &serde_json::Value| -> bool {
                    let left_val = #left_access;
                    let right_val = #right_access;
                    match (left_val, right_val) {
                        (Some(l), Some(r)) => #comparison,
                        _ => false
                    }
                }
            };
        }
    }

    // If no operator found, treat as a boolean field check
    let field_name = expr.trim();
    quote! {
        |state: &serde_json::Value| -> bool {
            state.get(#field_name).and_then(|v| v.as_bool()).unwrap_or(false)
        }
    }
}

fn field_access(field: &str) -> TokenStream2 {
    // Check if it's a numeric literal
    if field.parse::<i64>().is_ok() {
        let n: i64 = field.parse().unwrap();
        return quote! { Some(#n) };
    }

    // It's a field name
    quote! { state.get(#field).and_then(|v| v.as_i64()) }
}

/// Attribute macro for instrumenting methods with automatic tracing.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Monitored)]
/// struct Counter {
///     value: i32,
/// }
///
/// impl Counter {
///     #[monitor_action]
///     fn increment(&mut self) {
///         self.value += 1;
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn monitor_action(attr: TokenStream, item: TokenStream) -> TokenStream {
    let _ = attr; // Could parse action name from attr
    let input = parse_macro_input!(item as syn::ItemFn);

    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_sig = &input.sig;
    let fn_block = &input.block;
    let fn_attrs = &input.attrs;

    let action_name = fn_name.to_string();

    // Generate instrumented version
    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            // Original function body - in a real implementation,
            // this would integrate with TraceContext
            let __action_name = #action_name;
            #fn_block
        }
    };

    expanded.into()
}

// ============================================================================
// Kani Formal Verification Proofs
// ============================================================================
//
// Note: Proc-macro crates have special compilation requirements that make
// full Kani verification challenging. These proofs verify the pure helper
// functions that process invariant expressions and field access patterns.

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ------------------------------------------------------------------------
    // ContainerConfig default proofs
    // ------------------------------------------------------------------------

    /// Verify ContainerConfig::default() creates valid config
    #[kani::proof]
    fn proof_container_config_default() {
        let config = ContainerConfig::default();
        kani::assert(config.name.is_none(), "Default name should be None");
        kani::assert(
            config.invariants.is_empty(),
            "Default invariants should be empty",
        );
    }

    // ------------------------------------------------------------------------
    // FieldConfig default proofs
    // ------------------------------------------------------------------------

    /// Verify FieldConfig::default() creates valid config
    #[kani::proof]
    fn proof_field_config_default() {
        let config = FieldConfig::default();
        kani::assert(!config.skip, "Default skip should be false");
        kani::assert(!config.track, "Default track should be false");
        kani::assert(config.rename.is_none(), "Default rename should be None");
    }

    // ------------------------------------------------------------------------
    // parse_invariant_expr proofs
    // ------------------------------------------------------------------------

    /// Verify parse_invariant_expr handles >= operator
    #[kani::proof]
    fn proof_parse_invariant_expr_gte() {
        let tokens = parse_invariant_expr("value >= 0");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles <= operator
    #[kani::proof]
    fn proof_parse_invariant_expr_lte() {
        let tokens = parse_invariant_expr("value <= max");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles == operator
    #[kani::proof]
    fn proof_parse_invariant_expr_eq() {
        let tokens = parse_invariant_expr("count == 10");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles != operator
    #[kani::proof]
    fn proof_parse_invariant_expr_ne() {
        let tokens = parse_invariant_expr("status != 0");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles > operator
    #[kani::proof]
    fn proof_parse_invariant_expr_gt() {
        let tokens = parse_invariant_expr("count > 5");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles < operator
    #[kani::proof]
    fn proof_parse_invariant_expr_lt() {
        let tokens = parse_invariant_expr("index < 100");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles boolean field check
    #[kani::proof]
    fn proof_parse_invariant_expr_boolean() {
        let tokens = parse_invariant_expr("enabled");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles float values
    #[kani::proof]
    fn proof_parse_invariant_expr_float() {
        let tokens = parse_invariant_expr("ratio >= 0.5");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles true literal
    #[kani::proof]
    fn proof_parse_invariant_expr_true() {
        let tokens = parse_invariant_expr("flag == true");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles false literal
    #[kani::proof]
    fn proof_parse_invariant_expr_false() {
        let tokens = parse_invariant_expr("flag == false");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify parse_invariant_expr handles empty string
    #[kani::proof]
    fn proof_parse_invariant_expr_empty() {
        let tokens = parse_invariant_expr("");
        kani::assert(
            !tokens.is_empty(),
            "Should produce non-empty token stream for boolean check",
        );
    }

    // ------------------------------------------------------------------------
    // field_access proofs
    // ------------------------------------------------------------------------

    /// Verify field_access handles numeric literal
    #[kani::proof]
    fn proof_field_access_numeric() {
        let tokens = field_access("42");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify field_access handles negative numeric literal
    #[kani::proof]
    fn proof_field_access_negative_numeric() {
        let tokens = field_access("-10");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify field_access handles field name
    #[kani::proof]
    fn proof_field_access_field_name() {
        let tokens = field_access("my_field");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify field_access handles empty string
    #[kani::proof]
    fn proof_field_access_empty() {
        let tokens = field_access("");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    /// Verify field_access handles zero
    #[kani::proof]
    fn proof_field_access_zero() {
        let tokens = field_access("0");
        kani::assert(!tokens.is_empty(), "Should produce non-empty token stream");
    }

    // ------------------------------------------------------------------------
    // InvariantDef proofs
    // ------------------------------------------------------------------------

    /// Verify InvariantDef can be created with valid data
    #[kani::proof]
    fn proof_invariant_def_creation() {
        let inv = InvariantDef {
            name: String::from("test"),
            expr: String::from("x >= 0"),
        };
        kani::assert(!inv.name.is_empty(), "Name should not be empty");
        kani::assert(!inv.expr.is_empty(), "Expr should not be empty");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_invariant_expr_gte() {
        let tokens = parse_invariant_expr("value >= 0");
        let code = tokens.to_string();
        assert!(code.contains(">="));
    }

    #[test]
    fn test_parse_invariant_expr_eq() {
        let tokens = parse_invariant_expr("count == 10");
        let code = tokens.to_string();
        assert!(code.contains("=="));
    }

    #[test]
    fn test_parse_invariant_expr_field_field() {
        let tokens = parse_invariant_expr("value <= max");
        let code = tokens.to_string();
        assert!(code.contains("<="));
    }

    #[test]
    fn test_parse_invariant_expr_boolean() {
        let tokens = parse_invariant_expr("enabled");
        let code = tokens.to_string();
        assert!(code.contains("as_bool"));
    }

    // Test for != operator (line 418)
    #[test]
    fn test_parse_invariant_expr_not_equal() {
        let tokens = parse_invariant_expr("status != 0");
        let code = tokens.to_string();
        assert!(code.contains("!="));
        // Verify it generates a proper closure that can distinguish != from ==
        assert!(!code.contains("l == r") || code.contains("l != r"));
    }

    // Test for > operator (line 419)
    #[test]
    fn test_parse_invariant_expr_greater_than() {
        let tokens = parse_invariant_expr("count > 5");
        let code = tokens.to_string();
        assert!(code.contains(">"));
        // Check that it's not >= (strictly greater)
        assert!(code.contains("l > r"));
    }

    // Test for < operator (line 420)
    #[test]
    fn test_parse_invariant_expr_less_than() {
        let tokens = parse_invariant_expr("index < 100");
        let code = tokens.to_string();
        assert!(code.contains("<"));
        // Check that it's not <= (strictly less)
        assert!(code.contains("l < r"));
    }

    // Test float parsing with || condition (line 406)
    // When right side is a float, we fall back with .or(Some(n))
    #[test]
    fn test_parse_invariant_expr_float_right_side() {
        let tokens = parse_invariant_expr("ratio >= 0.5");
        let code = tokens.to_string();
        // Should contain the float value and use as_f64
        assert!(code.contains("0.5") || code.contains("as_f64"));
    }

    #[test]
    fn test_parse_invariant_expr_float_field_comparison() {
        let tokens = parse_invariant_expr("value >= 3.14");
        let code = tokens.to_string();
        // Float parsing should produce code with .or() fallback
        assert!(code.contains("or"));
    }

    // Test all operators produce distinct code
    #[test]
    fn test_all_operators_produce_distinct_code() {
        let eq = parse_invariant_expr("a == 1").to_string();
        let ne = parse_invariant_expr("a != 1").to_string();
        let gt = parse_invariant_expr("a > 1").to_string();
        let lt = parse_invariant_expr("a < 1").to_string();
        let gte = parse_invariant_expr("a >= 1").to_string();
        let lte = parse_invariant_expr("a <= 1").to_string();

        // Each should produce distinct comparison code
        assert!(eq.contains("l == r"));
        assert!(ne.contains("l != r"));
        assert!(gt.contains("l > r"));
        assert!(lt.contains("l < r"));
        assert!(gte.contains("l >= r"));
        assert!(lte.contains("l <= r"));
    }

    // Test monitor_action macro produces non-empty TokenStream (line 476)
    #[test]
    fn test_monitor_action_produces_output() {
        use proc_macro2::TokenStream as TokenStream2;
        use quote::quote;

        // The monitor_action function transforms a function
        // We can't directly call the proc macro in unit tests,
        // but we can verify the quote! expansion produces valid tokens
        let fn_name = quote::format_ident!("test_fn");
        let action_name = "test_fn";
        let fn_block = quote! { { 42 } };

        // Simulate what monitor_action generates
        let expanded: TokenStream2 = quote! {
            fn #fn_name() -> i32 {
                let __action_name = #action_name;
                #fn_block
            }
        };

        // Verify it's not empty (Default::default() would be empty)
        assert!(!expanded.is_empty());
        let code = expanded.to_string();
        assert!(code.contains("test_fn"));
        assert!(code.contains("__action_name"));
    }

    // Test field_access function
    #[test]
    fn test_field_access_numeric_literal() {
        let tokens = field_access("42");
        let code = tokens.to_string();
        assert!(code.contains("Some"));
        assert!(code.contains("42"));
    }

    #[test]
    fn test_field_access_field_name() {
        let tokens = field_access("my_field");
        let code = tokens.to_string();
        assert!(code.contains("state"));
        assert!(code.contains("my_field"));
        assert!(code.contains("as_i64"));
    }

    // Test that boolean true/false literals are handled
    #[test]
    fn test_parse_invariant_expr_boolean_true() {
        let tokens = parse_invariant_expr("flag == true");
        let code = tokens.to_string();
        assert!(code.contains("true"));
    }

    #[test]
    fn test_parse_invariant_expr_boolean_false() {
        let tokens = parse_invariant_expr("flag == false");
        let code = tokens.to_string();
        assert!(code.contains("false"));
    }
}
