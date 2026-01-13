//! TLA+ value parsing
//!
//! Parses TLA+ values into structured CounterexampleValue types.

use crate::traits::CounterexampleValue;

/// Parse a TLA+ value string into CounterexampleValue
pub fn parse_tla_value(value: &str) -> CounterexampleValue {
    let trimmed = value.trim();

    // Boolean
    if trimmed == "TRUE" {
        return CounterexampleValue::Bool(true);
    }
    if trimmed == "FALSE" {
        return CounterexampleValue::Bool(false);
    }

    // Integer (including negative)
    if let Ok(n) = trimmed.parse::<i128>() {
        return CounterexampleValue::Int {
            value: n,
            type_hint: None,
        };
    }

    // String (TLA+ strings are "quoted")
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        return CounterexampleValue::String(trimmed[1..trimmed.len() - 1].to_string());
    }

    // Set: {elem1, elem2, ...}
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return parse_tla_set(trimmed);
    }

    // Sequence/tuple: <<elem1, elem2, ...>>
    if trimmed.starts_with("<<") && trimmed.ends_with(">>") {
        return parse_tla_sequence(trimmed);
    }

    // Record: [field |-> value, ...]
    if trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.contains("|->") {
        return parse_tla_record(trimmed);
    }

    // Function: (key :> value @@ key :> value ...)
    if trimmed.starts_with('(') && trimmed.ends_with(')') && trimmed.contains(":>") {
        return parse_tla_function(trimmed);
    }

    // Interval: a..b (represents set {a, a+1, ..., b})
    if let Some(interval) = parse_tla_interval(trimmed) {
        return interval;
    }

    // Model value (unquoted identifier, commonly used in TLA+ for enumerated types)
    // These are constants like "m1", "m2", "a", "b" used as model values
    if trimmed.chars().all(|c| c.is_alphanumeric() || c == '_') && !trimmed.is_empty() {
        // Could be a model value - store as Unknown since we don't have a ModelValue variant
        return CounterexampleValue::Unknown(trimmed.to_string());
    }

    // Unknown - store as string
    CounterexampleValue::Unknown(trimmed.to_string())
}

/// Parse a TLA+ set: {elem1, elem2, ...}
fn parse_tla_set(value: &str) -> CounterexampleValue {
    let inner = &value[1..value.len() - 1].trim();
    if inner.is_empty() {
        return CounterexampleValue::Set(Vec::new());
    }

    let elements = split_tla_elements(inner);
    let parsed: Vec<CounterexampleValue> =
        elements.into_iter().map(|e| parse_tla_value(&e)).collect();

    CounterexampleValue::Set(parsed)
}

/// Parse a TLA+ sequence: <<elem1, elem2, ...>>
fn parse_tla_sequence(value: &str) -> CounterexampleValue {
    let inner = &value[2..value.len() - 2].trim();
    if inner.is_empty() {
        return CounterexampleValue::Sequence(Vec::new());
    }

    let elements = split_tla_elements(inner);
    let parsed: Vec<CounterexampleValue> =
        elements.into_iter().map(|e| parse_tla_value(&e)).collect();

    CounterexampleValue::Sequence(parsed)
}

/// Parse a TLA+ record: [field |-> value, ...]
fn parse_tla_record(value: &str) -> CounterexampleValue {
    let inner = &value[1..value.len() - 1].trim();
    if inner.is_empty() {
        return CounterexampleValue::Record(std::collections::HashMap::new());
    }

    let mut fields = std::collections::HashMap::new();

    // Split on commas that are not nested inside brackets/braces/parens
    let parts = split_tla_elements(inner);

    for part in parts {
        // Each part should be "field |-> value"
        if let Some(arrow_pos) = part.find("|->") {
            let field_name = part[..arrow_pos].trim().to_string();
            let field_value = part[arrow_pos + 3..].trim();
            fields.insert(field_name, parse_tla_value(field_value));
        }
    }

    CounterexampleValue::Record(fields)
}

/// Parse a TLA+ function: (key :> value @@ key :> value ...)
fn parse_tla_function(value: &str) -> CounterexampleValue {
    let inner = &value[1..value.len() - 1].trim();
    if inner.is_empty() {
        return CounterexampleValue::Function(Vec::new());
    }

    let mut mappings = Vec::new();

    // Split on @@ which separates function mappings
    let parts = split_on_function_sep(inner);

    for part in parts {
        // Each part should be "key :> value"
        if let Some(arrow_pos) = part.find(":>") {
            let key = part[..arrow_pos].trim();
            let val = part[arrow_pos + 2..].trim();
            mappings.push((parse_tla_value(key), parse_tla_value(val)));
        }
    }

    CounterexampleValue::Function(mappings)
}

/// Parse a TLA+ interval: a..b (represents the set {a, a+1, ..., b})
///
/// Returns Some(CounterexampleValue::Set) if this is a valid interval,
/// None if it's not in interval format.
fn parse_tla_interval(value: &str) -> Option<CounterexampleValue> {
    // Look for ".." in the value
    let dot_dot_pos = value.find("..")?;

    // Make sure it's not "..." (three dots, which is different syntax)
    if value[dot_dot_pos..].starts_with("...") {
        return None;
    }

    let left = value[..dot_dot_pos].trim();
    let right = value[dot_dot_pos + 2..].trim();

    // Both sides must be integers
    let start: i128 = left.parse().ok()?;
    let end: i128 = right.parse().ok()?;

    // Handle empty intervals (start > end)
    if start > end {
        return Some(CounterexampleValue::Set(Vec::new()));
    }

    // Limit expansion to prevent memory issues with huge intervals
    // For very large intervals, store as Unknown with the original syntax
    const MAX_INTERVAL_SIZE: i128 = 1000;
    if end - start > MAX_INTERVAL_SIZE {
        return Some(CounterexampleValue::Unknown(value.to_string()));
    }

    // Expand the interval into a set
    let elements: Vec<CounterexampleValue> = (start..=end)
        .map(|n| CounterexampleValue::Int {
            value: n,
            type_hint: None,
        })
        .collect();

    Some(CounterexampleValue::Set(elements))
}

/// Split TLA+ elements by comma, respecting nested brackets
fn split_tla_elements(input: &str) -> Vec<String> {
    let mut elements = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut in_string = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' if !in_string => {
                in_string = true;
                current.push(c);
            }
            '"' if in_string => {
                in_string = false;
                current.push(c);
            }
            '{' | '[' | '(' if !in_string => {
                depth += 1;
                current.push(c);
            }
            '<' if !in_string && chars.peek() == Some(&'<') => {
                chars.next(); // consume second '<'
                depth += 1;
                current.push_str("<<");
            }
            '}' | ']' | ')' if !in_string => {
                depth -= 1;
                current.push(c);
            }
            '>' if !in_string && chars.peek() == Some(&'>') => {
                chars.next(); // consume second '>'
                depth -= 1;
                current.push_str(">>");
            }
            ',' if depth == 0 && !in_string => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    elements.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(c);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        elements.push(trimmed);
    }

    elements
}

/// Split on @@ for function mappings, respecting nested structures
fn split_on_function_sep(input: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut in_string = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' if !in_string => {
                in_string = true;
                current.push(c);
            }
            '"' if in_string => {
                in_string = false;
                current.push(c);
            }
            '{' | '[' | '(' if !in_string => {
                depth += 1;
                current.push(c);
            }
            '<' if !in_string && chars.peek() == Some(&'<') => {
                chars.next();
                depth += 1;
                current.push_str("<<");
            }
            '}' | ']' | ')' if !in_string => {
                depth -= 1;
                current.push(c);
            }
            '>' if !in_string && chars.peek() == Some(&'>') => {
                chars.next();
                depth -= 1;
                current.push_str(">>");
            }
            '@' if depth == 0 && !in_string && chars.peek() == Some(&'@') => {
                chars.next(); // consume second '@'
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(c);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }

    parts
}
