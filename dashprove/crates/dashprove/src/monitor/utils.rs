//! String utility functions for monitor code generation

/// Convert PascalCase to snake_case
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(c.to_lowercase().next().unwrap());
    }
    result
}

/// Convert snake_case to camelCase
pub fn to_camel_case(s: &str) -> String {
    let parts: Vec<&str> = s.split('_').collect();
    if parts.is_empty() {
        return s.to_string();
    }
    let first = parts[0].to_lowercase();
    let rest: String = parts[1..]
        .iter()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect();
    format!("{first}{rest}")
}

/// Convert snake_case to PascalCase
pub fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

/// Sanitize a string for use as an identifier
pub fn sanitize_identifier(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        match ch {
            ch if ch.is_ascii_alphanumeric() || ch == '_' => out.push(ch),
            '\'' => out.push_str("_prime"),
            _ => out.push('_'),
        }
    }
    if out.is_empty() {
        "value".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("hello"), "Hello");
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("foo_bar_baz"), "FooBarBaz");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("HelloWorld"), "hello_world");
        assert_eq!(to_snake_case("ABC"), "a_b_c");
        assert_eq!(to_snake_case("simple"), "simple");
    }

    #[test]
    fn test_to_camel_case() {
        assert_eq!(to_camel_case("hello_world"), "helloWorld");
        assert_eq!(to_camel_case("foo_bar_baz"), "fooBarBaz");
        assert_eq!(to_camel_case("simple"), "simple");
    }
}
