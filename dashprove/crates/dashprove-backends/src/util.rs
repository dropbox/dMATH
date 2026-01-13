//! Utility functions shared across backends

use std::path::PathBuf;

// ============================================================================
// Kani Formal Verification Proofs
// ============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ------------------------------------------------------------------------
    // mermaid_escape proofs
    // ------------------------------------------------------------------------

    /// Verify that mermaid_escape handles empty strings
    #[kani::proof]
    fn proof_mermaid_escape_empty_string() {
        let result = mermaid_escape("");
        kani::assert(
            result.is_empty(),
            "Empty string should produce empty result",
        );
    }

    /// Verify that mermaid_escape replaces double quotes
    #[kani::proof]
    fn proof_mermaid_escape_replaces_double_quotes() {
        let input = "hello\"world";
        let result = mermaid_escape(input);
        kani::assert(!result.contains('"'), "Double quotes should be replaced");
        kani::assert(
            result.contains('\''),
            "Double quotes should become single quotes",
        );
    }

    /// Verify that mermaid_escape replaces square brackets
    #[kani::proof]
    fn proof_mermaid_escape_replaces_square_brackets() {
        let input = "[test]";
        let result = mermaid_escape(input);
        kani::assert(
            !result.contains('['),
            "Open square bracket should be replaced",
        );
        kani::assert(
            !result.contains(']'),
            "Close square bracket should be replaced",
        );
        kani::assert(
            result.contains('('),
            "Open square bracket should become parenthesis",
        );
        kani::assert(
            result.contains(')'),
            "Close square bracket should become parenthesis",
        );
    }

    /// Verify that mermaid_escape replaces curly braces
    #[kani::proof]
    fn proof_mermaid_escape_replaces_curly_braces() {
        let input = "{test}";
        let result = mermaid_escape(input);
        kani::assert(!result.contains('{'), "Open curly brace should be replaced");
        kani::assert(
            !result.contains('}'),
            "Close curly brace should be replaced",
        );
        kani::assert(
            result.contains('('),
            "Open curly brace should become parenthesis",
        );
        kani::assert(
            result.contains(')'),
            "Close curly brace should become parenthesis",
        );
    }

    /// Verify that mermaid_escape preserves safe characters
    #[kani::proof]
    fn proof_mermaid_escape_preserves_safe_chars() {
        let input = "hello world 123";
        let result = mermaid_escape(input);
        kani::assert(result == input, "Safe characters should be preserved");
    }

    /// Verify that mermaid_escape handles all special characters together
    #[kani::proof]
    fn proof_mermaid_escape_all_special_chars() {
        let input = r#"node["label{test}"]"#;
        let result = mermaid_escape(input);
        kani::assert(!result.contains('"'), "No double quotes in result");
        kani::assert(!result.contains('['), "No open square brackets in result");
        kani::assert(!result.contains(']'), "No close square brackets in result");
        kani::assert(!result.contains('{'), "No open curly braces in result");
        kani::assert(!result.contains('}'), "No close curly braces in result");
    }

    // ------------------------------------------------------------------------
    // expand_home_dir proofs
    // ------------------------------------------------------------------------

    /// Verify that expand_home_dir returns absolute paths as-is
    #[kani::proof]
    fn proof_expand_home_dir_absolute_path() {
        let input = "/usr/bin/lake";
        let result = expand_home_dir(input);
        kani::assert(result.is_some(), "Absolute paths should return Some");
        if let Some(path) = result {
            kani::assert(
                path.to_string_lossy() == input,
                "Absolute paths should be returned unchanged",
            );
        }
    }

    /// Verify that expand_home_dir handles relative paths
    #[kani::proof]
    fn proof_expand_home_dir_relative_path() {
        let input = "relative/path";
        let result = expand_home_dir(input);
        kani::assert(result.is_some(), "Relative paths should return Some");
        if let Some(path) = result {
            kani::assert(
                path.to_string_lossy() == input,
                "Relative paths (not starting with ~/) should be returned unchanged",
            );
        }
    }

    /// Verify that expand_home_dir handles empty strings
    #[kani::proof]
    fn proof_expand_home_dir_empty_string() {
        let input = "";
        let result = expand_home_dir(input);
        kani::assert(result.is_some(), "Empty string should return Some");
    }

    /// Verify that expand_home_dir handles tilde alone
    #[kani::proof]
    fn proof_expand_home_dir_tilde_only() {
        let input = "~";
        let result = expand_home_dir(input);
        // "~" alone (without "/") should return as-is since it doesn't start with "~/"
        kani::assert(result.is_some(), "Tilde alone should return Some");
        if let Some(path) = result {
            kani::assert(
                path.to_string_lossy() == "~",
                "Tilde alone should be returned as-is",
            );
        }
    }

    /// Verify that expand_home_dir handles paths starting with ~/
    #[kani::proof]
    fn proof_expand_home_dir_tilde_slash() {
        let input = "~/.config/test";
        let result = expand_home_dir(input);
        // Result depends on whether home_dir() returns Some or None
        // We can only verify the function doesn't panic
        // If home_dir returns Some, result should be Some
        // If home_dir returns None, result should be None
        let _ = result;
    }

    /// Verify that expand_home_dir handles ~/ with empty rest
    #[kani::proof]
    fn proof_expand_home_dir_tilde_slash_only() {
        let input = "~/";
        let result = expand_home_dir(input);
        // Similar to above, depends on home_dir()
        let _ = result;
    }
}

/// Escape special characters for Mermaid diagram syntax.
///
/// Mermaid diagrams have special handling for certain characters.
/// This function converts them to safe alternatives.
#[must_use]
pub fn mermaid_escape(s: &str) -> String {
    s.replace('"', "'")
        .replace('[', "(")
        .replace(']', ")")
        .replace('{', "(")
        .replace('}', ")")
}

/// Expands `~` to the user's home directory in paths.
///
/// # Examples
///
/// ```
/// use dashprove_backends::util::expand_home_dir;
/// use std::path::PathBuf;
///
/// // Paths starting with ~/ are expanded
/// if let Some(path) = expand_home_dir("~/.elan/toolchains/leanprover-lean4-stable/bin/lake") {
///     assert!(path.to_string_lossy().contains("bin/lake"));
/// }
///
/// // Absolute paths are returned as-is
/// let path = expand_home_dir("/usr/bin/lake");
/// assert_eq!(path, Some(PathBuf::from("/usr/bin/lake")));
/// ```
#[must_use]
pub fn expand_home_dir(path: &str) -> Option<PathBuf> {
    if path.starts_with("~/") {
        dirs::home_dir().map(|home| home.join(&path[2..]))
    } else {
        Some(PathBuf::from(path))
    }
}
