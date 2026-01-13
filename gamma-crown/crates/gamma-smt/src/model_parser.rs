//! Parse SMT-LIB model output to extract variable values.
//!
//! The Z4 solver returns models in SMT-LIB format:
//! ```text
//! (model
//!   (define-fun x_0 () Real 5.0)
//!   (define-fun x_1 () Real (- 3.0))
//!   (define-fun y_0 () Real (/ 7 2))
//! )
//! ```
//!
//! This module parses these models to extract f64 values for
//! specified variable names.

use std::collections::HashMap;

/// Parse a model string and extract values for specified variables.
///
/// # Arguments
/// * `model_str` - The SMT-LIB model string from (get-model)
/// * `var_names` - Variable names to extract
///
/// # Returns
/// Vector of values in the same order as var_names.
/// Returns None if any variable is missing from the model.
pub fn parse_model(model_str: &str, var_names: &[String]) -> Option<Vec<f64>> {
    let model = parse_model_to_map(model_str)?;

    let mut values = Vec::with_capacity(var_names.len());
    for name in var_names {
        values.push(*model.get(name)?);
    }

    Some(values)
}

/// Parse a model string into a HashMap of variable names to values.
pub fn parse_model_to_map(model_str: &str) -> Option<HashMap<String, f64>> {
    let mut values = HashMap::new();

    // Handle error messages
    if model_str.starts_with("(error") {
        tracing::debug!("Model error: {}", model_str);
        return None;
    }

    // Handle empty model
    if model_str.trim() == "(model\n)" || model_str.trim() == "(model)" {
        return Some(values);
    }

    // Parse define-fun declarations
    // Format: (define-fun name () type value)
    let mut pos = 0;
    let bytes = model_str.as_bytes();

    while let Some(start) = model_str[pos..].find("(define-fun ") {
        let abs_start = pos + start;
        pos = abs_start + 12; // Skip "(define-fun "

        // Parse name
        let name_end = model_str[pos..].find(' ')?;
        let name = model_str[pos..pos + name_end].to_string();
        pos += name_end + 1;

        // Skip "() " and type to find value
        // Find the type (Real, Int, Bool)
        pos = skip_until_char(bytes, pos, b')')?; // Skip ()
        pos += 1;
        pos = skip_whitespace(bytes, pos);
        pos = skip_until_char(bytes, pos, b' ')?; // Skip type
        pos += 1;
        pos = skip_whitespace(bytes, pos);

        // Parse value
        let (value, new_pos) = parse_value(model_str, pos)?;
        pos = new_pos;

        values.insert(name, value);
    }

    Some(values)
}

/// Skip whitespace and return new position.
fn skip_whitespace(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

/// Skip until we find a specific character.
fn skip_until_char(bytes: &[u8], mut pos: usize, target: u8) -> Option<usize> {
    while pos < bytes.len() {
        if bytes[pos] == target {
            return Some(pos);
        }
        pos += 1;
    }
    None
}

/// Parse a value expression (number, negation, or fraction).
fn parse_value(s: &str, pos: usize) -> Option<(f64, usize)> {
    let bytes = s.as_bytes();
    let pos = skip_whitespace(bytes, pos);

    if pos >= bytes.len() {
        return None;
    }

    match bytes[pos] {
        b'(' => {
            // Could be (- ...), (/ ...), or another SMT expression.
            // Robustly parse operator token (handles newlines/extra whitespace).
            let mut op_pos = pos + 1;
            op_pos = skip_whitespace(bytes, op_pos);
            if op_pos >= bytes.len() {
                return None;
            }

            let op_start = op_pos;
            while op_pos < bytes.len()
                && !bytes[op_pos].is_ascii_whitespace()
                && bytes[op_pos] != b')'
            {
                op_pos += 1;
            }
            if op_pos == op_start {
                return None;
            }
            let op = &s[op_start..op_pos];

            match op {
                "-" => {
                    // Unary negation: (- value)
                    // Binary subtraction: (- a b)
                    let (a, a_end) = parse_value(s, op_pos)?;
                    let a_end_ws = skip_whitespace(bytes, a_end);
                    if a_end_ws < bytes.len() && bytes[a_end_ws] == b')' {
                        Some((-a, a_end_ws + 1))
                    } else {
                        let (b, b_end) = parse_value(s, a_end)?;
                        let close = skip_until_char(bytes, b_end, b')')?;
                        Some((a - b, close + 1))
                    }
                }
                "/" => {
                    // Division / rational: (/ num denom)
                    let (num, num_end) = parse_value(s, op_pos)?;
                    let (denom, denom_end) = parse_value(s, num_end)?;
                    let close = skip_until_char(bytes, denom_end, b')')?;
                    Some((num / denom, close + 1))
                }
                _ => None,
            }
        }
        b'-' if pos + 1 < bytes.len()
            && (bytes[pos + 1].is_ascii_digit() || bytes[pos + 1] == b'.') =>
        {
            // Direct negative number like -5.0
            let (val, end) = parse_number(s, pos + 1)?;
            Some((-val, end))
        }
        b'+' if pos + 1 < bytes.len()
            && (bytes[pos + 1].is_ascii_digit() || bytes[pos + 1] == b'.') =>
        {
            // Direct positive number like +5.0
            parse_number(s, pos + 1)
        }
        _ => parse_number(s, pos),
    }
}

/// Parse a decimal number.
fn parse_number(s: &str, pos: usize) -> Option<(f64, usize)> {
    let bytes = s.as_bytes();
    let mut end = pos;

    // Digits before decimal point
    let mut saw_digit = false;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        saw_digit = true;
        end += 1;
    }

    // Decimal point + digits
    if end < bytes.len() && bytes[end] == b'.' {
        end += 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            saw_digit = true;
            end += 1;
        }
    }

    if !saw_digit {
        return None;
    }

    // Optional exponent
    if end < bytes.len() && (bytes[end] == b'e' || bytes[end] == b'E') {
        end += 1;
        if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
            end += 1;
        }

        let exp_digits_start = end;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }

        if exp_digits_start == end {
            // "1e" is not a valid float.
            return None;
        }
    }

    let num_str = &s[pos..end];
    let value = num_str.parse::<f64>().ok()?;
    Some((value, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_model() {
        let model = r#"(model
  (define-fun x_0 () Real 5.0)
  (define-fun x_1 () Real 3.0)
)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - 5.0).abs() < 1e-10);
        assert!((values[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_negative_values() {
        let model = r#"(model
  (define-fun x_0 () Real (- 5.0))
  (define-fun x_1 () Real (- 3))
)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - (-5.0)).abs() < 1e-10);
        assert!((values[1] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_fractions() {
        let model = r#"(model
  (define-fun x_0 () Real (/ 7 2))
  (define-fun x_1 () Real (- (/ 1 4)))
)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - 3.5).abs() < 1e-10);
        assert!((values[1] - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_missing_variable() {
        let model = r#"(model
  (define-fun x_0 () Real 5.0)
)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string()];
        let values = parse_model(model, &var_names);

        assert!(values.is_none());
    }

    #[test]
    fn test_parse_empty_model() {
        let model = "(model\n)";
        let var_names: Vec<String> = vec![];
        let values = parse_model(model, &var_names).unwrap();
        assert!(values.is_empty());
    }

    #[test]
    fn test_parse_error_model() {
        let model = r#"(error "model is not available")"#;
        let var_names = vec!["x_0".to_string()];
        let values = parse_model(model, &var_names);
        assert!(values.is_none());
    }

    #[test]
    fn test_parse_integer_without_decimal() {
        let model = r#"(model
  (define-fun x_0 () Real 5)
)"#;

        // The Z4 format uses "5.0" for integers, but let's handle "5" too
        let map = parse_model_to_map(model).unwrap();
        assert!((map["x_0"] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_with_lin_vars() {
        // Test with variable naming pattern used in encoder
        let model = r#"(model
  (define-fun x_0 () Real 1.5)
  (define-fun lin_0 () Real 2.5)
  (define-fun relu_1 () Real 3.5)
  (define-fun lin_2 () Real 4.5)
)"#;

        let input_vars = vec!["x_0".to_string()];
        let output_vars = vec!["lin_2".to_string()];

        let input_values = parse_model(model, &input_vars).unwrap();
        let output_values = parse_model(model, &output_vars).unwrap();

        assert!((input_values[0] - 1.5).abs() < 1e-10);
        assert!((output_values[0] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_value_with_newlines_and_whitespace() {
        let model = r#"(model
  (define-fun x_0 () Real (- 
      5.0
  ))
  (define-fun x_1 () Real (/ 
      7
      2
  ))
)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - (-5.0)).abs() < 1e-10);
        assert!((values[1] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_scientific_notation() {
        let model = r#"(model
  (define-fun x_0 () Real 1e-3)
  (define-fun x_1 () Real +2.5E+2)
  (define-fun x_2 () Real (- 1.25e2))
)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string(), "x_2".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 3);
        assert!((values[0] - 0.001).abs() < 1e-10);
        assert!((values[1] - 250.0).abs() < 1e-10);
        assert!((values[2] - (-125.0)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_binary_subtraction() {
        let model = r#"(model
  (define-fun x_0 () Real (- 10 3))
  (define-fun x_1 () Real (- 10.5 0.25))
)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - 7.0).abs() < 1e-10);
        assert!((values[1] - 10.25).abs() < 1e-10);
    }

    // ===== Tests for helper functions =====

    #[test]
    fn test_skip_whitespace() {
        let s = "   abc";
        let pos = skip_whitespace(s.as_bytes(), 0);
        assert_eq!(pos, 3);

        // No whitespace
        let s = "abc";
        let pos = skip_whitespace(s.as_bytes(), 0);
        assert_eq!(pos, 0);

        // All whitespace
        let s = "   ";
        let pos = skip_whitespace(s.as_bytes(), 0);
        assert_eq!(pos, 3);

        // Various whitespace chars
        let s = " \t\n\rabc";
        let pos = skip_whitespace(s.as_bytes(), 0);
        assert_eq!(pos, 4);
    }

    #[test]
    fn test_skip_until_char() {
        let s = "abc)def";
        let pos = skip_until_char(s.as_bytes(), 0, b')');
        assert_eq!(pos, Some(3));

        // Target not found
        let s = "abcdef";
        let pos = skip_until_char(s.as_bytes(), 0, b')');
        assert_eq!(pos, None);

        // Target at start
        let s = ")abc";
        let pos = skip_until_char(s.as_bytes(), 0, b')');
        assert_eq!(pos, Some(0));

        // Search from middle
        let s = "abc)def";
        let pos = skip_until_char(s.as_bytes(), 2, b')');
        assert_eq!(pos, Some(3));
    }

    #[test]
    fn test_parse_number_integer() {
        let (val, end) = parse_number("123", 0).unwrap();
        assert!((val - 123.0).abs() < 1e-10);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_parse_number_decimal() {
        let (val, end) = parse_number("123.456", 0).unwrap();
        assert!((val - 123.456).abs() < 1e-10);
        assert_eq!(end, 7);
    }

    #[test]
    fn test_parse_number_decimal_only() {
        // .5 is just decimal without leading digit
        let (val, end) = parse_number(".5", 0).unwrap();
        assert!((val - 0.5).abs() < 1e-10);
        assert_eq!(end, 2);
    }

    #[test]
    fn test_parse_number_with_exponent() {
        let (val, end) = parse_number("1.5e3", 0).unwrap();
        assert!((val - 1500.0).abs() < 1e-10);
        assert_eq!(end, 5);

        let (val, end) = parse_number("2E-2", 0).unwrap();
        assert!((val - 0.02).abs() < 1e-10);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_parse_number_invalid_exponent() {
        // "1e" without exponent digits should return None
        let result = parse_number("1e", 0);
        assert!(result.is_none());

        // "1e+" without digits
        let result = parse_number("1e+", 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_number_no_digit() {
        // No digit at all
        let result = parse_number("abc", 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_number_stops_at_non_digit() {
        let (val, end) = parse_number("123abc", 0).unwrap();
        assert!((val - 123.0).abs() < 1e-10);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_parse_value_simple_number() {
        let (val, _) = parse_value("42.0", 0).unwrap();
        assert!((val - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_value_direct_negative() {
        let (val, _) = parse_value("-5.5", 0).unwrap();
        assert!((val - (-5.5)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_value_direct_positive() {
        let (val, _) = parse_value("+3.25", 0).unwrap();
        assert!((val - 3.25).abs() < 1e-10);
    }

    #[test]
    fn test_parse_value_unknown_operator() {
        // Unknown operator like (* 2 3) should return None
        let result = parse_value("(* 2 3)", 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_value_empty_parens() {
        // Empty expression
        let result = parse_value("()", 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_value_past_end() {
        // Position past end
        let result = parse_value("123", 10);
        assert!(result.is_none());
    }

    // ===== Tests for parse_model_to_map edge cases =====

    #[test]
    fn test_parse_model_to_map_direct() {
        let model = r#"(model
  (define-fun a () Real 1.0)
  (define-fun b () Real 2.0)
)"#;

        let map = parse_model_to_map(model).unwrap();
        assert_eq!(map.len(), 2);
        assert!((map["a"] - 1.0).abs() < 1e-10);
        assert!((map["b"] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_model_to_map_no_define_fun() {
        // Model with no define-fun statements
        let model = "(model)";
        let map = parse_model_to_map(model).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_parse_model_empty_var_names() {
        let model = r#"(model
  (define-fun x_0 () Real 5.0)
)"#;
        let var_names: Vec<String> = vec![];
        let values = parse_model(model, &var_names).unwrap();
        assert!(values.is_empty());
    }

    #[test]
    fn test_parse_model_single_variable() {
        let model = r#"(model
  (define-fun only_one () Real 42.0)
)"#;

        let var_names = vec!["only_one".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 1);
        assert!((values[0] - 42.0).abs() < 1e-10);
    }

    // ===== Tests for special number formats =====

    #[test]
    fn test_parse_zero() {
        let model = r#"(model
  (define-fun x () Real 0)
  (define-fun y () Real 0.0)
  (define-fun z () Real (- 0))
)"#;

        let values =
            parse_model(model, &["x".to_string(), "y".to_string(), "z".to_string()]).unwrap();
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 0.0).abs() < 1e-10);
        assert!((values[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_very_large_number() {
        let model = r#"(model
  (define-fun x () Real 1e308)
)"#;

        let map = parse_model_to_map(model).unwrap();
        assert!(map["x"] > 1e307);
    }

    #[test]
    fn test_parse_very_small_number() {
        let model = r#"(model
  (define-fun x () Real 1e-308)
)"#;

        let map = parse_model_to_map(model).unwrap();
        assert!(map["x"] < 1e-307);
        assert!(map["x"] > 0.0);
    }

    #[test]
    fn test_parse_nested_negation() {
        let model = r#"(model
  (define-fun x () Real (- (- 5)))
)"#;

        let map = parse_model_to_map(model).unwrap();
        assert!((map["x"] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_nested_division() {
        let model = r#"(model
  (define-fun x () Real (/ (/ 12 3) 2))
)"#;

        let map = parse_model_to_map(model).unwrap();
        // 12 / 3 = 4, then 4 / 2 = 2
        assert!((map["x"] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_negative_fraction() {
        let model = r#"(model
  (define-fun x () Real (/ (- 10) 4))
)"#;

        let map = parse_model_to_map(model).unwrap();
        assert!((map["x"] - (-2.5)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_fraction_with_negative_denominator() {
        let model = r#"(model
  (define-fun x () Real (/ 10 (- 4)))
)"#;

        let map = parse_model_to_map(model).unwrap();
        assert!((map["x"] - (-2.5)).abs() < 1e-10);
    }

    // ===== Tests for unusual variable names =====

    #[test]
    fn test_parse_long_variable_names() {
        let model = r#"(model
  (define-fun very_long_variable_name_with_numbers_123 () Real 99.9)
)"#;

        let var_names = vec!["very_long_variable_name_with_numbers_123".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 1);
        assert!((values[0] - 99.9).abs() < 1e-10);
    }

    #[test]
    fn test_parse_many_variables() {
        let mut model = "(model\n".to_string();
        for i in 0..20 {
            model.push_str(&format!("  (define-fun v_{} () Real {})\n", i, i as f64));
        }
        model.push(')');

        let var_names: Vec<String> = (0..20).map(|i| format!("v_{}", i)).collect();
        let values = parse_model(&model, &var_names).unwrap();

        assert_eq!(values.len(), 20);
        for (i, &val) in values.iter().enumerate() {
            assert!((val - i as f64).abs() < 1e-10);
        }
    }

    // ===== Tests for robustness =====

    #[test]
    fn test_parse_extra_whitespace_between_definitions() {
        // Parser handles extra blank lines between definitions
        let model = r#"(model

  (define-fun x_0 () Real 5.0)

  (define-fun x_1 () Real 10.0)

)"#;

        let var_names = vec!["x_0".to_string(), "x_1".to_string()];
        let values = parse_model(model, &var_names).unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - 5.0).abs() < 1e-10);
        assert!((values[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_compact_model() {
        // Minimal whitespace
        let model = "(model(define-fun x () Real 1))";

        let map = parse_model_to_map(model).unwrap();
        assert!((map["x"] - 1.0).abs() < 1e-10);
    }
}
