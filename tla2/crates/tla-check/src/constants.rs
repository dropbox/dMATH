//! Constant value parsing and binding for model checking
//!
//! This module handles parsing constant values from TLC config files
//! and binding them to the evaluation context.
//!
//! # Supported formats
//!
//! - Integer literals: `3`, `-42`, `100`
//! - Set literals: `{a, b, c}`, `{1, 2, 3}`
//! - Nested sets: `{{a, b}, {c, d}}`
//! - Model values: identifiers like `a`, `p1`, `server`
//!
//! # Model Values
//!
//! TLC creates special "model values" for identifiers in constant assignments.
//! These are distinct values that compare equal only to themselves.
//! We represent them as Value::ModelValue(String).

use crate::config::{Config, ConstantValue};
use crate::error::EvalError;
use crate::eval::EvalCtx;
use crate::value::{SetBuilder, SortedSet, Value};

/// Parse a constant value string and return the runtime value
///
/// # Arguments
/// * `value_str` - The value string from the config file (e.g., "{a, b, c}" or "3")
///
/// # Returns
/// * `Ok(Value)` - The parsed runtime value
/// * `Err(String)` - Parse error message
///
/// # Supported syntax
///
/// - Integer literals: `3`, `-42`, `100`
/// - String literals: `"hello"`, `"neg"`, `"SAT"`
/// - Boolean literals: `TRUE`, `FALSE`
/// - Set literals: `{a, b, c}`, `{1, 2, 3}`, nested `{{a}, {b}}`
/// - Tuple/sequence literals: `<<1, 2, 3>>`, `<<a, "neg">>`
/// - Record literals: `[field |-> value]`, `[x |-> 1, y |-> 2]`
/// - Model values: identifiers like `a`, `p1`, `server`
pub fn parse_constant_value(value_str: &str) -> Result<Value, String> {
    let trimmed = value_str.trim();

    // Try integer first
    if let Ok(n) = trimmed.parse::<i64>() {
        return Ok(Value::Int(n.into()));
    }

    // Try boolean literals
    if trimmed == "TRUE" {
        return Ok(Value::Bool(true));
    }
    if trimmed == "FALSE" {
        return Ok(Value::Bool(false));
    }

    // Try string literals
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        return Ok(Value::String(inner.to_string().into()));
    }

    // Try parsing as a set literal
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return parse_set_literal(trimmed);
    }

    // Try parsing as a sequence/tuple
    if trimmed.starts_with("<<") && trimmed.ends_with(">>") {
        return parse_tuple_literal(trimmed);
    }

    // Try parsing as a record literal
    if trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.contains("|->") {
        return parse_record_literal(trimmed);
    }

    // Otherwise treat as model value (identifier)
    if is_valid_identifier(trimmed) {
        return Ok(Value::model_value(trimmed));
    }

    Err(format!("Cannot parse constant value: {}", value_str))
}

/// Parse a set literal like `{a, b, c}` or `{1, 2, 3}`
fn parse_set_literal(s: &str) -> Result<Value, String> {
    let inner = &s[1..s.len() - 1];
    let trimmed = inner.trim();

    if trimmed.is_empty() {
        return Ok(Value::Set(SortedSet::new()));
    }

    let elements = split_set_elements(trimmed)?;
    let mut set = SetBuilder::new();

    for elem in elements {
        let value = parse_constant_value(&elem)?;
        set.insert(value);
    }

    Ok(set.build_value())
}

/// Parse a tuple/sequence literal like `<<a, b, c>>`
fn parse_tuple_literal(s: &str) -> Result<Value, String> {
    let inner = &s[2..s.len() - 2];
    let trimmed = inner.trim();

    if trimmed.is_empty() {
        return Ok(Value::Tuple(Vec::new().into()));
    }

    let elements = split_set_elements(trimmed)?;
    let mut tuple = Vec::new();

    for elem in elements {
        let value = parse_constant_value(&elem)?;
        tuple.push(value);
    }

    Ok(Value::Tuple(tuple.into()))
}

/// Parse a record literal like `[field |-> value]` or `[x |-> 1, y |-> 2]`
fn parse_record_literal(s: &str) -> Result<Value, String> {
    let inner = &s[1..s.len() - 1];
    let trimmed = inner.trim();

    if trimmed.is_empty() {
        return Err("Empty record literal".to_string());
    }

    // Split on commas at the top level
    let field_strs = split_set_elements(trimmed)?;
    let mut fields = Vec::new();

    for field_str in field_strs {
        // Parse "field |-> value"
        if let Some(pos) = field_str.find("|->") {
            let name = field_str[..pos].trim();
            let value_str = field_str[pos + 3..].trim();

            if name.is_empty() {
                return Err(format!("Empty field name in record: {}", field_str));
            }

            let value = parse_constant_value(value_str)?;
            fields.push((name.to_string(), value));
        } else {
            return Err(format!("Invalid record field syntax: {}", field_str));
        }
    }

    Ok(Value::record(fields))
}

/// Split set elements, handling nested braces, brackets, tuples, and strings
fn split_set_elements(s: &str) -> Result<Vec<String>, String> {
    let mut elements = Vec::new();
    let mut current = String::new();
    let mut brace_depth = 0; // {}
    let mut bracket_depth = 0; // []
    let mut angle_depth = 0; // <<>>
    let mut in_string = false;
    let mut prev_char = '\0';

    for c in s.chars() {
        // Handle string state transitions
        if c == '"' && !in_string {
            in_string = true;
            current.push(c);
            prev_char = c;
            continue;
        }
        if c == '"' && in_string && prev_char != '\\' {
            in_string = false;
            current.push(c);
            prev_char = c;
            continue;
        }

        // If inside a string, just accumulate characters
        if in_string {
            current.push(c);
            prev_char = c;
            continue;
        }

        match c {
            '{' => {
                brace_depth += 1;
                current.push(c);
            }
            '}' => {
                brace_depth -= 1;
                current.push(c);
            }
            '[' => {
                bracket_depth += 1;
                current.push(c);
            }
            ']' => {
                bracket_depth -= 1;
                current.push(c);
            }
            '<' => {
                // Check for <<
                if prev_char == '<' {
                    angle_depth += 1;
                }
                current.push(c);
            }
            '>' => {
                // Check for >>
                if prev_char == '>' {
                    angle_depth -= 1;
                }
                current.push(c);
            }
            ',' if brace_depth == 0 && bracket_depth == 0 && angle_depth == 0 => {
                let elem = current.trim().to_string();
                if !elem.is_empty() {
                    elements.push(elem);
                }
                current.clear();
            }
            _ => {
                current.push(c);
            }
        }
        prev_char = c;
    }

    let elem = current.trim().to_string();
    if !elem.is_empty() {
        elements.push(elem);
    }

    Ok(elements)
}

/// Check if string is a valid TLA+ identifier
fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// Bind constants from config to the evaluation context
///
/// # Arguments
/// * `ctx` - The evaluation context to bind constants into
/// * `config` - The parsed TLC config
///
/// # Returns
/// * `Ok(())` - All constants bound successfully
/// * `Err(CheckError)` - Error binding a constant
pub fn bind_constants_from_config(ctx: &mut EvalCtx, config: &Config) -> Result<(), EvalError> {
    for (name, const_val) in &config.constants {
        let value = match const_val {
            ConstantValue::Value(value_str) => {
                parse_constant_value(value_str).map_err(|e| EvalError::Internal {
                    message: format!("Error parsing CONSTANT {}: {}", name, e),
                    span: None,
                })?
            }
            ConstantValue::ModelValue => {
                // A standalone model value - use the constant name as the model value name
                Value::model_value(name.as_str())
            }
            ConstantValue::ModelValueSet(values) => {
                // Set of model values - bind each individual model value to the context
                let mut set = SetBuilder::new();
                for v in values {
                    let mv = Value::model_value(v.as_str());
                    // Bind each model value so it can be referenced in the spec
                    ctx.bind_mut(v.clone(), mv.clone());
                    set.insert(mv);
                }
                set.build_value()
            }
            ConstantValue::Replacement(op_name) => {
                // Operator replacement: when evaluating `name`, use `op_name` instead
                // This is used for things like `Seq <- BoundedSeq` in config files
                ctx.add_op_replacement(name.clone(), op_name.clone());
                // Skip the value binding below - this is an operator replacement, not a value
                continue;
            }
        };

        // Also bind individual model values found in parsed set literals
        // This ensures that when a spec references "m2", it resolves to the same
        // model value that's in the set {m1, m2, m3}
        bind_model_values_from_value(ctx, &value);

        // Mark this constant as config-provided so lookups prioritize env over operator defs.
        // This is essential for cases like `Done == CHOOSE v : v \notin Reg` with
        // `Done = Done` in config - we want the model value @Done, not the CHOOSE.
        ctx.add_config_constant(name.clone());

        // Bind the constant value in the evaluation context
        ctx.bind_mut(name.clone(), value);
    }

    Ok(())
}

/// Recursively extract and bind model values from a Value
/// This ensures that model values like `m1`, `m2` from a set `{m1, m2, m3}`
/// can be referenced directly in the spec
fn bind_model_values_from_value(ctx: &mut EvalCtx, value: &Value) {
    match value {
        Value::ModelValue(name) => {
            // Bind this model value so it can be referenced by name
            ctx.bind_mut(name.to_string(), value.clone());
        }
        Value::Set(set) => {
            // Recursively bind model values in set elements
            for elem in set.iter() {
                bind_model_values_from_value(ctx, elem);
            }
        }
        Value::Tuple(elems) => {
            // Recursively bind model values in tuple elements
            for elem in elems.iter() {
                bind_model_values_from_value(ctx, elem);
            }
        }
        Value::Seq(seq) => {
            // Recursively bind model values in seq elements
            for elem in seq.iter() {
                bind_model_values_from_value(ctx, elem);
            }
        }
        _ => {
            // Other value types don't contain model values to bind
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_integer() {
        assert_eq!(parse_constant_value("3").unwrap(), Value::Int(3.into()));
        assert_eq!(
            parse_constant_value("-42").unwrap(),
            Value::Int((-42).into())
        );
        assert_eq!(
            parse_constant_value("  100  ").unwrap(),
            Value::Int(100.into())
        );
    }

    #[test]
    fn test_parse_model_value() {
        assert_eq!(
            parse_constant_value("foo").unwrap(),
            Value::model_value("foo")
        );
        assert_eq!(
            parse_constant_value("server1").unwrap(),
            Value::model_value("server1")
        );
    }

    #[test]
    fn test_parse_empty_set() {
        assert_eq!(
            parse_constant_value("{}").unwrap(),
            Value::Set(SortedSet::new())
        );
    }

    #[test]
    fn test_parse_integer_set() {
        let result = parse_constant_value("{1, 2, 3}").unwrap();
        if let Value::Set(set) = result {
            assert_eq!(set.len(), 3);
            assert!(set.contains(&Value::Int(1.into())));
            assert!(set.contains(&Value::Int(2.into())));
            assert!(set.contains(&Value::Int(3.into())));
        } else {
            panic!("Expected set");
        }
    }

    #[test]
    fn test_parse_model_value_set() {
        let result = parse_constant_value("{matches, paper, tobacco}").unwrap();
        if let Value::Set(set) = result {
            assert_eq!(set.len(), 3);
            assert!(set.contains(&Value::model_value("matches")));
            assert!(set.contains(&Value::model_value("paper")));
            assert!(set.contains(&Value::model_value("tobacco")));
        } else {
            panic!("Expected set");
        }
    }

    #[test]
    fn test_parse_nested_set() {
        let result = parse_constant_value("{{a, b}, {c, d}}").unwrap();
        if let Value::Set(set) = result {
            assert_eq!(set.len(), 2);
        } else {
            panic!("Expected set");
        }
    }

    #[test]
    fn test_parse_tuple() {
        let result = parse_constant_value("<<1, 2, 3>>").unwrap();
        if let Value::Tuple(tuple) = result {
            assert_eq!(tuple.len(), 3);
            assert_eq!(tuple[0], Value::Int(1.into()));
        } else {
            panic!("Expected tuple");
        }
    }

    #[test]
    fn test_bind_constants() {
        let mut ctx = EvalCtx::new();
        let mut config = Config::new();
        config
            .constants
            .insert("N".to_string(), ConstantValue::Value("3".to_string()));
        config.constants.insert(
            "Procs".to_string(),
            ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );

        bind_constants_from_config(&mut ctx, &config).unwrap();

        // Check N is bound
        let n = ctx.lookup("N").unwrap();
        assert_eq!(n, &Value::Int(3.into()));

        // Check Procs is bound as set of model values
        let procs = ctx.lookup("Procs").unwrap();
        if let Value::Set(set) = procs {
            assert_eq!(set.len(), 2);
            assert!(set.contains(&Value::model_value("p1")));
        } else {
            panic!("Expected Procs to be a set");
        }
    }

    // ==================== New tests for Z4 Integration (Phase 9) ====================

    #[test]
    fn test_parse_string_literal() {
        assert_eq!(
            parse_constant_value("\"hello\"").unwrap(),
            Value::String("hello".into())
        );
        assert_eq!(
            parse_constant_value("\"neg\"").unwrap(),
            Value::String("neg".into())
        );
        assert_eq!(
            parse_constant_value("\"SAT\"").unwrap(),
            Value::String("SAT".into())
        );
    }

    #[test]
    fn test_parse_boolean_literal() {
        assert_eq!(parse_constant_value("TRUE").unwrap(), Value::Bool(true));
        assert_eq!(parse_constant_value("FALSE").unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_parse_tuple_with_string() {
        // Z4 use case: <<v1, "neg">>
        let result = parse_constant_value("<<v1, \"neg\">>").unwrap();
        if let Value::Tuple(tuple) = result {
            assert_eq!(tuple.len(), 2);
            assert_eq!(tuple[0], Value::model_value("v1"));
            assert_eq!(tuple[1], Value::String("neg".into()));
        } else {
            panic!("Expected tuple, got {:?}", result);
        }
    }

    #[test]
    fn test_parse_record_simple() {
        let result = parse_constant_value("[x |-> 1]").unwrap();
        let rec = result.as_record().unwrap();
        assert_eq!(rec.len(), 1);
        assert_eq!(
            rec.get(&std::sync::Arc::from("x")),
            Some(&Value::Int(1.into()))
        );
    }

    #[test]
    fn test_parse_record_multiple_fields() {
        let result = parse_constant_value("[x |-> 1, y |-> 2, z |-> 3]").unwrap();
        let rec = result.as_record().unwrap();
        assert_eq!(rec.len(), 3);
        assert_eq!(
            rec.get(&std::sync::Arc::from("x")),
            Some(&Value::Int(1.into()))
        );
        assert_eq!(
            rec.get(&std::sync::Arc::from("y")),
            Some(&Value::Int(2.into()))
        );
        assert_eq!(
            rec.get(&std::sync::Arc::from("z")),
            Some(&Value::Int(3.into()))
        );
    }

    #[test]
    fn test_parse_record_with_string_value() {
        let result = parse_constant_value("[status |-> \"active\"]").unwrap();
        let rec = result.as_record().unwrap();
        assert_eq!(
            rec.get(&std::sync::Arc::from("status")),
            Some(&Value::String("active".into()))
        );
    }

    #[test]
    fn test_parse_set_of_tuples_with_strings() {
        // Z4 use case: {<<v1, "neg">>, <<v2, "pos">>}
        let result = parse_constant_value("{<<v1, \"neg\">>, <<v2, \"pos\">>}").unwrap();
        if let Value::Set(set) = result {
            assert_eq!(set.len(), 2);
        } else {
            panic!("Expected set, got {:?}", result);
        }
    }

    #[test]
    fn test_parse_z4_cdcl_clauses() {
        // Full Z4 use case: sets of sets of tuples with strings
        // Clauses = { {<<v1, "pos">>, <<v2, "pos">>}, {<<v1, "neg">>}, {<<v2, "neg">>} }
        let input = "{ {<<v1, \"pos\">>, <<v2, \"pos\">>}, {<<v1, \"neg\">>}, {<<v2, \"neg\">>} }";
        let result = parse_constant_value(input).unwrap();

        if let Value::Set(clauses) = result {
            assert_eq!(clauses.len(), 3);

            // Check that we have sets of tuples
            for clause in clauses.iter() {
                if let Value::Set(literals) = clause {
                    for lit in literals.iter() {
                        if let Value::Tuple(tuple) = lit {
                            assert_eq!(tuple.len(), 2);
                            // First element is model value (variable)
                            assert!(matches!(tuple[0], Value::ModelValue(_)));
                            // Second element is string (polarity)
                            assert!(matches!(tuple[1], Value::String(_)));
                        } else {
                            panic!("Expected tuple in clause, got {:?}", lit);
                        }
                    }
                } else {
                    panic!("Expected set of literals, got {:?}", clause);
                }
            }
        } else {
            panic!("Expected set of clauses, got {:?}", result);
        }
    }

    #[test]
    fn test_parse_nested_record_in_set() {
        // Edges = { [src |-> n1, dst |-> n2, weight |-> 5], [src |-> n2, dst |-> n3, weight |-> 3] }
        let input =
            "{ [src |-> n1, dst |-> n2, weight |-> 5], [src |-> n2, dst |-> n3, weight |-> 3] }";
        let result = parse_constant_value(input).unwrap();

        if let Value::Set(edges) = result {
            assert_eq!(edges.len(), 2);
            for edge in edges.iter() {
                let rec = edge.as_record().expect("Expected record for edge");
                assert!(rec.contains_key(&std::sync::Arc::from("src")));
                assert!(rec.contains_key(&std::sync::Arc::from("dst")));
                assert!(rec.contains_key(&std::sync::Arc::from("weight")));
            }
        } else {
            panic!("Expected set of records, got {:?}", result);
        }
    }

    #[test]
    fn test_parse_string_with_comma() {
        // Edge case: string containing comma shouldn't split
        let result = parse_constant_value("\"hello, world\"").unwrap();
        assert_eq!(result, Value::String("hello, world".into()));
    }

    #[test]
    fn test_parse_tuple_with_string_containing_comma() {
        let result = parse_constant_value("<<a, \"b, c\">>").unwrap();
        if let Value::Tuple(tuple) = result {
            assert_eq!(tuple.len(), 2);
            assert_eq!(tuple[0], Value::model_value("a"));
            assert_eq!(tuple[1], Value::String("b, c".into()));
        } else {
            panic!("Expected tuple, got {:?}", result);
        }
    }

    /// Regression test for #62: config constant precedence
    ///
    /// When a constant like `Done = Done` is in the config file, it should:
    /// 1. Be marked as a config constant (is_config_constant returns true)
    /// 2. Take precedence over any operator definition like `Done == CHOOSE v : ...`
    ///
    /// This fixes specs like MCInnerSequential, MCLiveInternalMemory, MCLiveWriteThroughCache
    /// where config values should override unbounded CHOOSE operators.
    #[test]
    fn test_config_constant_precedence() {
        let mut ctx = EvalCtx::new();
        let mut config = Config::new();

        // Simulate config file with `Done = Done` (model value assignment)
        // This is the pattern that overrides `Done == CHOOSE v : v \notin Reg`
        config
            .constants
            .insert("Done".to_string(), ConstantValue::Value("Done".to_string()));

        // Before binding, should not be a config constant
        assert!(
            !ctx.is_config_constant("Done"),
            "Done should not be a config constant before binding"
        );

        bind_constants_from_config(&mut ctx, &config).unwrap();

        // After binding, should be marked as a config constant
        assert!(
            ctx.is_config_constant("Done"),
            "Done should be marked as a config constant after binding from config"
        );

        // The value should be the model value @Done, not evaluated from CHOOSE
        let done = ctx.lookup("Done").unwrap();
        assert_eq!(
            done,
            &Value::model_value("Done"),
            "Done should be model value @Done from config"
        );

        // Also test with a regular integer constant
        let mut config2 = Config::new();
        config2
            .constants
            .insert("MaxRetries".to_string(), ConstantValue::Value("5".to_string()));

        let mut ctx2 = EvalCtx::new();
        bind_constants_from_config(&mut ctx2, &config2).unwrap();

        assert!(
            ctx2.is_config_constant("MaxRetries"),
            "MaxRetries should be marked as config constant"
        );
        assert_eq!(ctx2.lookup("MaxRetries").unwrap(), &Value::Int(5.into()));
    }
}
