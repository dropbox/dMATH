//! Lean 4 Parser Compatibility Tests
//!
//! This module tests Lean5 parser against actual Lean 4 test files
//! to measure and track compatibility percentage.

#[cfg(test)]
mod tests {
    use crate::Parser;
    use std::borrow::Cow;
    use std::fs;
    use std::path::Path;
    use walkdir::WalkDir;

    /// Test parsing of Lean 4 test suite files
    /// Reports compatibility percentage for tracking
    #[test]
    fn lean4_parser_compatibility_suite() {
        // Path relative to crate root (crates/lean5-parser/)
        let test_dir = Path::new("../../tests/lean4_compat/lean4_tests");

        if !test_dir.exists() {
            println!("Lean 4 test files not found at {test_dir:?}");
            println!("Run tests/lean4_compat/download_tests.sh to download test files");
            return;
        }

        let mut passed = 0;
        let mut failed = 0;
        let mut failures: Vec<(String, String)> = Vec::new();

        for entry in WalkDir::new(test_dir)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "lean"))
        {
            let path = entry.path();
            let (content, lossy_utf8) = match fs::read(path) {
                Ok(bytes) => match String::from_utf8_lossy(&bytes) {
                    Cow::Owned(s) => (s, true),
                    Cow::Borrowed(s) => (s.to_string(), false),
                },
                Err(e) => {
                    failed += 1;
                    failures.push((path.display().to_string(), format!("IO error: {e}")));
                    continue;
                }
            };

            // Parse in a separate thread with limited stack to detect stack overflow
            let content_clone = content.clone();
            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            eprint!("Parsing: {filename}... ");
            let result = std::thread::Builder::new()
                .stack_size(4 * 1024 * 1024) // 4MB stack
                .spawn(move || Parser::parse_file(&content_clone))
                .ok()
                .and_then(|h| h.join().ok())
                .unwrap_or_else(|| {
                    Err(crate::ParseError::UnexpectedToken {
                        line: 0,
                        col: 0,
                        message: "Stack overflow or thread panic".to_string(),
                    })
                });
            eprintln!("done");

            match result {
                Ok(_) => passed += 1,
                Err(e) => {
                    failed += 1;
                    let mut msg = format!("{e}");
                    if lossy_utf8 {
                        msg.push_str(" [lossy utf-8 decode]");
                    }
                    failures.push((path.display().to_string(), msg));
                }
            }
        }

        let total = passed + failed;
        let percentage = if total > 0 {
            100.0 * passed as f64 / total as f64
        } else {
            0.0
        };

        println!();
        println!("========================================");
        println!("Lean 4 Parser Compatibility Report");
        println!("========================================");
        println!("Passed: {passed}");
        println!("Failed: {failed}");
        println!("Total:  {total}");
        println!("Compatibility: {percentage:.1}% ({passed}/{total})");
        println!("========================================");

        // Report first 20 failures for debugging
        if !failures.is_empty() {
            println!();
            println!("First {} failures:", failures.len().min(20));
            for (path, err) in failures.iter().take(20) {
                let filename = Path::new(path)
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default();
                // Truncate error to first line
                let short_err = err.lines().next().unwrap_or(err);
                println!("  {filename} - {short_err}");
            }
        }
    }

    /// Test specific Lean 4 syntax constructs
    /// These tests track progress - they pass and print status, not panic
    mod specific_constructs {
        use crate::{parse_decl, parse_expr, parse_file};

        fn track(name: &str, result: Result<impl std::fmt::Debug, impl std::fmt::Debug>) -> bool {
            match &result {
                Ok(_) => {
                    println!("✓ {name}");
                    true
                }
                Err(e) => {
                    println!("✗ {name} - {e:?}");
                    false
                }
            }
        }

        #[test]
        fn lean4_syntax_compatibility_summary() {
            let mut passed = 0;
            let mut total = 0;

            // Class definition
            total += 1;
            if track("class definition", parse_decl("class Vec (X : Type u)")) {
                passed += 1;
            }

            // Instance with priority
            total += 1;
            if track(
                "instance with priority",
                parse_decl("instance (priority := default+1) instFoo : Vec ℝ := sorry"),
            ) {
                passed += 1;
            }

            // Structure
            total += 1;
            if track(
                "structure",
                parse_decl("structure Point where\n  x : Nat\n  y : Nat"),
            ) {
                passed += 1;
            }

            // Inductive
            total += 1;
            if track(
                "inductive",
                parse_decl(
                    "inductive List (α : Type u) where\n  | nil : List α\n  | cons : α → List α → List α",
                ),
            ) {
                passed += 1;
            }

            // Theorem
            total += 1;
            if track("theorem", parse_decl("theorem foo : 1 + 1 = 2 := rfl")) {
                passed += 1;
            }

            // do notation
            total += 1;
            if track(
                "do notation",
                parse_decl("def test : IO Unit := do\n  let x ← pure 1\n  pure ()"),
            ) {
                passed += 1;
            }

            // match expression
            total += 1;
            if track(
                "match expression",
                parse_decl(
                    "def foo (n : Nat) : Nat :=\n  match n with\n  | 0 => 1\n  | n + 1 => n",
                ),
            ) {
                passed += 1;
            }

            // Lambda with type
            total += 1;
            if track("lambda with type", parse_expr("fun (x : Nat) => x + 1")) {
                passed += 1;
            }

            // Forall type
            total += 1;
            if track("forall type", parse_expr("∀ (x : Nat), x = x")) {
                passed += 1;
            }

            // Implicit binder
            total += 1;
            if track(
                "implicit binder",
                parse_decl("def id {α : Type} (x : α) : α := x"),
            ) {
                passed += 1;
            }

            // Instance implicit
            total += 1;
            if track(
                "instance implicit",
                parse_decl("def toString [ToString α] (x : α) : String := ToString.toString x"),
            ) {
                passed += 1;
            }

            // Namespace
            total += 1;
            if track(
                "namespace",
                parse_file("namespace Foo\ndef bar : Nat := 1\nend Foo"),
            ) {
                passed += 1;
            }

            // Attributes
            total += 1;
            if track("attributes", parse_decl("@[simp] def foo : Nat := 1")) {
                passed += 1;
            }

            // Where clause
            total += 1;
            if track(
                "where clause",
                parse_decl("def foo : Nat → Nat where\n  | 0 => 1\n  | n + 1 => foo n"),
            ) {
                passed += 1;
            }

            // If-then-else
            total += 1;
            if track("if-then-else", parse_expr("if true then 1 else 0")) {
                passed += 1;
            }

            // Let-in
            total += 1;
            if track("let-in", parse_expr("let x := 1; x + 1")) {
                passed += 1;
            }

            // Universe
            total += 1;
            if track(
                "universe command",
                parse_file("universe u v\ndef foo : Type u → Type v := sorry"),
            ) {
                passed += 1;
            }

            // Open/import
            total += 1;
            if track("open command", parse_file("open Nat in #check succ")) {
                passed += 1;
            }

            println!();
            println!("========================================");
            println!("Lean 4 Syntax Construct Compatibility");
            println!("========================================");
            println!(
                "Passed: {}/{} ({:.1}%)",
                passed,
                total,
                100.0 * passed as f64 / total as f64
            );
            println!("========================================");

            // This test always passes - it's for tracking progress
        }

        #[test]
        fn anonymous_constructor_syntax() {
            // Test .foo anonymous constructor syntax
            let result = parse_expr(".done");
            assert!(result.is_ok(), "Failed to parse .done: {result:?}");

            // Test .foo with arguments
            let result = parse_expr(".left c");
            assert!(result.is_ok(), "Failed to parse .left c: {result:?}");

            // Test nested (.foo expr)
            let result = parse_expr("(.left c)");
            assert!(result.is_ok(), "Failed to parse (.left c): {result:?}");

            println!("Anonymous constructor tests passed");
        }

        // Note: test_file_1616_anonymous_ctor is disabled because it requires
        // support for multi-name binders with shared type like (x y z : List α)
        // which is tracked as a separate parser compatibility issue.

        #[test]
        fn test_let_with_explicit_in() {
            // Test let bindings with explicit `in` separator (required without layout sensitivity)
            let code = r"def test : Nat :=
  let x := 1 in
  let y := 2 in
  x + y";
            let result = parse_file(code);
            assert!(
                result.is_ok(),
                "Failed to parse let with explicit in: {result:?}"
            );
            println!("Let with explicit in parsed successfully");
        }

        #[test]
        fn test_chained_let_bindings() {
            // Chained let bindings work when the next statement is `let`
            // (no ambiguity about where value ends)
            let code = r"def test : Nat :=
  let x := 1
  let y := 2 in
  x + y";
            let result = parse_file(code);
            assert!(result.is_ok(), "Failed to parse chained let: {result:?}");
            println!("Chained let bindings parsed successfully");
        }

        #[test]
        fn test_multi_name_binder_in_type() {
            // Multi-name binders with shared type annotation
            // Used in dependent type signatures like: (x y z : List α) -> Type u

            // First test simple multi-name binder as expr
            let expr = parse_expr("(x y z : Nat) → Type");
            println!("Multi-name binder expr result: {expr:?}");

            // Test inductive with multi-name binder (file 1616)
            let code = r"inductive Cover : (x y z : List α) -> Type u
  | done  : Cover [] [] []";
            let result = parse_file(code);
            println!("Inductive with multi-name binder result: {result:?}");

            // These should pass once implemented
            if expr.is_err() {
                println!("ISSUE: Multi-name binder (x y z : T) -> U not yet supported");
            }
        }
    }
}
