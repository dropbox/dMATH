//! MIRI output parser

use crate::error::{MiriError, MiriResult};
use crate::execution::MiriOutput;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Kinds of undefined behavior MIRI can detect
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UndefinedBehaviorKind {
    /// Use after free
    UseAfterFree,
    /// Double free
    DoubleFree,
    /// Memory leak
    MemoryLeak,
    /// Out of bounds access
    OutOfBounds,
    /// Invalid pointer offset
    InvalidPointerOffset,
    /// Null pointer dereference
    NullPointerDereference,
    /// Uninitialized memory read
    UninitializedRead,
    /// Invalid alignment
    InvalidAlignment,
    /// Stacked borrows violation
    StackedBorrowsViolation,
    /// Data race
    DataRace,
    /// Deadlock
    Deadlock,
    /// Invalid enum discriminant
    InvalidEnumDiscriminant,
    /// Invalid function pointer
    InvalidFunctionPointer,
    /// Type validation error
    TypeValidation,
    /// Other undefined behavior
    Other(String),
}

impl UndefinedBehaviorKind {
    /// Parse UB kind from MIRI error message
    pub fn from_message(message: &str) -> Self {
        let lower = message.to_lowercase();

        if lower.contains("use after free") || lower.contains("dangling pointer") {
            UndefinedBehaviorKind::UseAfterFree
        } else if lower.contains("double free") {
            UndefinedBehaviorKind::DoubleFree
        } else if lower.contains("memory leak") || lower.contains("leaked memory") {
            UndefinedBehaviorKind::MemoryLeak
        } else if lower.contains("out of bounds") || lower.contains("alloc range") {
            UndefinedBehaviorKind::OutOfBounds
        } else if lower.contains("null pointer") {
            UndefinedBehaviorKind::NullPointerDereference
        } else if lower.contains("uninitialized") || lower.contains("uninit") {
            UndefinedBehaviorKind::UninitializedRead
        } else if lower.contains("alignment") || lower.contains("misaligned") {
            UndefinedBehaviorKind::InvalidAlignment
        } else if lower.contains("stacked borrows") || lower.contains("borrow stack") {
            UndefinedBehaviorKind::StackedBorrowsViolation
        } else if lower.contains("data race") {
            UndefinedBehaviorKind::DataRace
        } else if lower.contains("deadlock") {
            UndefinedBehaviorKind::Deadlock
        } else if lower.contains("enum") && lower.contains("discriminant") {
            UndefinedBehaviorKind::InvalidEnumDiscriminant
        } else if lower.contains("function pointer") {
            UndefinedBehaviorKind::InvalidFunctionPointer
        } else if lower.contains("type validation") {
            UndefinedBehaviorKind::TypeValidation
        } else if lower.contains("invalid pointer") || lower.contains("pointer offset") {
            UndefinedBehaviorKind::InvalidPointerOffset
        } else {
            UndefinedBehaviorKind::Other(message.to_string())
        }
    }
}

/// A detected undefined behavior instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndefinedBehavior {
    /// Kind of UB
    pub kind: UndefinedBehaviorKind,
    /// Full error message
    pub message: String,
    /// Source location (file:line:column)
    pub location: Option<String>,
    /// Backtrace frames
    pub backtrace: Vec<String>,
    /// Additional notes
    pub notes: Vec<String>,
}

/// Diagnostic level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum MiriDiagnosticLevel {
    Error,
    Warning,
    Note,
    Help,
}

/// A diagnostic message from MIRI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiriDiagnostic {
    /// Diagnostic level
    pub level: MiriDiagnosticLevel,
    /// Message text
    pub message: String,
    /// Source location
    pub location: Option<String>,
    /// Diagnostic code (e.g., E0001)
    pub code: Option<String>,
}

/// Status of a test run under MIRI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum MiriTestStatus {
    /// Test passed
    Passed,
    /// Test failed (assertion failure, panic)
    Failed,
    /// Test was ignored
    Ignored,
    /// Test detected UB
    UndefinedBehavior,
    /// Test timed out
    TimedOut,
}

/// Result of a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiriTestResult {
    /// Test name
    pub name: String,
    /// Test status
    pub status: MiriTestStatus,
    /// Duration (if available)
    pub duration_ms: Option<u64>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Parsed MIRI output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedMiriOutput {
    /// Detected undefined behaviors
    pub undefined_behaviors: Vec<UndefinedBehavior>,
    /// All diagnostics
    pub diagnostics: Vec<MiriDiagnostic>,
    /// Test results
    pub test_results: Vec<MiriTestResult>,
    /// Summary statistics
    pub summary: MiriSummary,
    /// Raw stderr (for debugging)
    pub raw_stderr: String,
}

/// Summary statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MiriSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests failed
    pub failed: usize,
    /// Tests ignored
    pub ignored: usize,
    /// UB instances detected
    pub ub_count: usize,
    /// Count by UB kind
    pub ub_by_kind: HashMap<String, usize>,
}

impl ParsedMiriOutput {
    /// Check if any undefined behavior was detected
    pub fn has_undefined_behavior(&self) -> bool {
        !self.undefined_behaviors.is_empty()
    }

    /// Check if all tests passed
    pub fn all_tests_passed(&self) -> bool {
        self.test_results
            .iter()
            .all(|t| t.status == MiriTestStatus::Passed || t.status == MiriTestStatus::Ignored)
    }
}

/// Parse MIRI output into structured format
pub fn parse_miri_output(output: &MiriOutput) -> MiriResult<ParsedMiriOutput> {
    let mut undefined_behaviors = Vec::new();
    let mut diagnostics = Vec::new();
    let mut test_results = Vec::new();

    // Parse stderr for UB and diagnostics
    parse_stderr(&output.stderr, &mut undefined_behaviors, &mut diagnostics)?;

    // Parse stdout for test results
    parse_test_results(&output.stdout, &mut test_results);

    // Build UB by kind map
    let mut ub_by_kind = std::collections::HashMap::new();
    for ub in &undefined_behaviors {
        let kind_name = format!("{:?}", ub.kind);
        *ub_by_kind.entry(kind_name).or_insert(0) += 1;
    }

    // Build summary
    let summary = MiriSummary {
        total_tests: test_results.len(),
        passed: test_results
            .iter()
            .filter(|t| t.status == MiriTestStatus::Passed)
            .count(),
        failed: test_results
            .iter()
            .filter(|t| t.status == MiriTestStatus::Failed)
            .count(),
        ignored: test_results
            .iter()
            .filter(|t| t.status == MiriTestStatus::Ignored)
            .count(),
        ub_count: undefined_behaviors.len(),
        ub_by_kind,
    };

    Ok(ParsedMiriOutput {
        undefined_behaviors,
        diagnostics,
        test_results,
        summary,
        raw_stderr: output.stderr.clone(),
    })
}

/// Parse stderr for UB and diagnostics
fn parse_stderr(
    stderr: &str,
    ub_list: &mut Vec<UndefinedBehavior>,
    diagnostics: &mut Vec<MiriDiagnostic>,
) -> MiriResult<()> {
    // Regex for error messages
    let error_re = Regex::new(r"error(?:\[E\d+\])?: (.+)")
        .map_err(|e| MiriError::ParseError(e.to_string()))?;

    // Regex for location
    let location_re =
        Regex::new(r"-->\s*([^:]+:\d+:\d+)").map_err(|e| MiriError::ParseError(e.to_string()))?;

    // Regex for UB message
    let ub_re = Regex::new(r"Undefined Behavior: (.+)")
        .map_err(|e| MiriError::ParseError(e.to_string()))?;

    // Regex for note/help
    let note_re =
        Regex::new(r"(note|help): (.+)").map_err(|e| MiriError::ParseError(e.to_string()))?;

    let mut current_ub: Option<UndefinedBehavior> = None;
    let mut current_notes: Vec<String> = Vec::new();

    for line in stderr.lines() {
        // Check for UB
        if let Some(caps) = ub_re.captures(line) {
            // Save previous UB if any
            if let Some(mut ub) = current_ub.take() {
                ub.notes = std::mem::take(&mut current_notes);
                ub_list.push(ub);
            }

            let message = caps.get(1).map_or("", |m| m.as_str());
            current_ub = Some(UndefinedBehavior {
                kind: UndefinedBehaviorKind::from_message(message),
                message: message.to_string(),
                location: None,
                backtrace: Vec::new(),
                notes: Vec::new(),
            });
        }
        // Check for location
        else if let Some(caps) = location_re.captures(line) {
            if let Some(ref mut ub) = current_ub {
                if ub.location.is_none() {
                    ub.location = Some(caps.get(1).map_or("", |m| m.as_str()).to_string());
                }
            }
        }
        // Check for note/help
        else if let Some(caps) = note_re.captures(line) {
            let content = caps.get(2).map_or("", |m| m.as_str());
            if current_ub.is_some() {
                current_notes.push(content.to_string());
            }
        }
        // Check for error (non-UB)
        else if let Some(caps) = error_re.captures(line) {
            let message = caps.get(1).map_or("", |m| m.as_str());

            // Skip if this is a UB error (already handled)
            if !message.contains("Undefined Behavior") {
                diagnostics.push(MiriDiagnostic {
                    level: MiriDiagnosticLevel::Error,
                    message: message.to_string(),
                    location: None,
                    code: None,
                });
            }
        }
        // Check for backtrace frame
        else if line.trim().starts_with("at ") || line.contains("in <") {
            if let Some(ref mut ub) = current_ub {
                ub.backtrace.push(line.trim().to_string());
            }
        }
    }

    // Save last UB if any
    if let Some(mut ub) = current_ub {
        ub.notes = current_notes;
        ub_list.push(ub);
    }

    Ok(())
}

/// Parse stdout for test results
fn parse_test_results(stdout: &str, results: &mut Vec<MiriTestResult>) {
    // Regex for test result line
    let test_re = Regex::new(r"test (\S+) \.\.\. (\w+)(?:\s+\((\d+)ms\))?").unwrap();

    for line in stdout.lines() {
        if let Some(caps) = test_re.captures(line) {
            let name = caps.get(1).map_or("", |m| m.as_str()).to_string();
            let status_str = caps.get(2).map_or("", |m| m.as_str());
            let duration_ms = caps.get(3).and_then(|m| m.as_str().parse().ok());

            let status = match status_str.to_lowercase().as_str() {
                "ok" | "passed" => MiriTestStatus::Passed,
                "failed" => MiriTestStatus::Failed,
                "ignored" => MiriTestStatus::Ignored,
                _ => MiriTestStatus::Failed,
            };

            results.push(MiriTestResult {
                name,
                status,
                duration_ms,
                error: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ub_kind_from_message() {
        assert_eq!(
            UndefinedBehaviorKind::from_message("memory access to dangling pointer"),
            UndefinedBehaviorKind::UseAfterFree
        );
        assert_eq!(
            UndefinedBehaviorKind::from_message("out of bounds memory access"),
            UndefinedBehaviorKind::OutOfBounds
        );
        assert_eq!(
            UndefinedBehaviorKind::from_message("stacked borrows violation"),
            UndefinedBehaviorKind::StackedBorrowsViolation
        );
        assert_eq!(
            UndefinedBehaviorKind::from_message("data race detected"),
            UndefinedBehaviorKind::DataRace
        );
    }

    #[test]
    fn test_parse_test_results() {
        let stdout = r#"
running 3 tests
test foo::test_one ... ok
test foo::test_two ... FAILED
test foo::test_three ... ignored
"#;
        let mut results = Vec::new();
        parse_test_results(stdout, &mut results);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].name, "foo::test_one");
        assert_eq!(results[0].status, MiriTestStatus::Passed);
        assert_eq!(results[1].name, "foo::test_two");
        assert_eq!(results[1].status, MiriTestStatus::Failed);
        assert_eq!(results[2].name, "foo::test_three");
        assert_eq!(results[2].status, MiriTestStatus::Ignored);
    }

    #[test]
    fn test_parse_miri_output_with_ub() {
        let output = MiriOutput {
            stdout: "running 1 test\ntest foo ... ok\n".to_string(),
            stderr: r#"
error: Undefined Behavior: memory access to dangling pointer
  --> src/lib.rs:10:5
   |
10 |     *ptr
   |     ^^^^ memory access to dangling pointer
   |
   = note: pointer is dangling
"#
            .to_string(),
            exit_code: Some(1),
            duration: std::time::Duration::from_secs(1),
            has_errors: true,
        };

        let parsed = parse_miri_output(&output).unwrap();
        assert!(parsed.has_undefined_behavior());
        assert_eq!(parsed.undefined_behaviors.len(), 1);
        assert_eq!(
            parsed.undefined_behaviors[0].kind,
            UndefinedBehaviorKind::UseAfterFree
        );
    }

    #[test]
    fn test_parse_clean_output() {
        let output = MiriOutput {
            stdout: "running 2 tests\ntest foo ... ok\ntest bar ... ok\n".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: std::time::Duration::from_secs(1),
            has_errors: false,
        };

        let parsed = parse_miri_output(&output).unwrap();
        assert!(!parsed.has_undefined_behavior());
        assert!(parsed.all_tests_passed());
        assert_eq!(parsed.summary.total_tests, 2);
        assert_eq!(parsed.summary.passed, 2);
    }

    #[test]
    fn test_summary_ub_by_kind() {
        let output = MiriOutput {
            stdout: String::new(),
            stderr: r#"
error: Undefined Behavior: out of bounds memory access
error: Undefined Behavior: out of bounds memory access
error: Undefined Behavior: stacked borrows violation
"#
            .to_string(),
            exit_code: Some(1),
            duration: std::time::Duration::from_secs(1),
            has_errors: true,
        };

        let parsed = parse_miri_output(&output).unwrap();
        assert_eq!(parsed.summary.ub_count, 3);
        assert_eq!(parsed.summary.ub_by_kind.get("OutOfBounds"), Some(&2));
        assert_eq!(
            parsed.summary.ub_by_kind.get("StackedBorrowsViolation"),
            Some(&1)
        );
    }

    #[test]
    fn test_ub_kind_memory_leak() {
        // Test the || condition in line 53: memory leak OR leaked memory
        assert_eq!(
            UndefinedBehaviorKind::from_message("memory leak detected"),
            UndefinedBehaviorKind::MemoryLeak
        );
        assert_eq!(
            UndefinedBehaviorKind::from_message("leaked memory found"),
            UndefinedBehaviorKind::MemoryLeak
        );
    }

    #[test]
    fn test_ub_kind_out_of_bounds() {
        // Test the || condition in line 55: out of bounds OR alloc range
        assert_eq!(
            UndefinedBehaviorKind::from_message("out of bounds access"),
            UndefinedBehaviorKind::OutOfBounds
        );
        assert_eq!(
            UndefinedBehaviorKind::from_message("outside alloc range"),
            UndefinedBehaviorKind::OutOfBounds
        );
    }

    #[test]
    fn test_ub_kind_uninitialized() {
        // Test the || condition in line 59: uninitialized OR uninit
        assert_eq!(
            UndefinedBehaviorKind::from_message("uninitialized memory"),
            UndefinedBehaviorKind::UninitializedRead
        );
        assert_eq!(
            UndefinedBehaviorKind::from_message("read uninit bytes"),
            UndefinedBehaviorKind::UninitializedRead
        );
    }

    #[test]
    fn test_ub_kind_invalid_enum_discriminant() {
        // Test the && condition in line 69: enum AND discriminant
        assert_eq!(
            UndefinedBehaviorKind::from_message("invalid enum discriminant value"),
            UndefinedBehaviorKind::InvalidEnumDiscriminant
        );
        // Should NOT match if only one condition is met
        let result = UndefinedBehaviorKind::from_message("invalid enum value");
        assert!(
            !matches!(result, UndefinedBehaviorKind::InvalidEnumDiscriminant),
            "Should not match without 'discriminant'"
        );
        let result2 = UndefinedBehaviorKind::from_message("invalid discriminant");
        assert!(
            !matches!(result2, UndefinedBehaviorKind::InvalidEnumDiscriminant),
            "Should not match without 'enum'"
        );
    }

    #[test]
    fn test_ub_kind_invalid_pointer_offset() {
        // Test the || condition in line 75: invalid pointer OR pointer offset
        assert_eq!(
            UndefinedBehaviorKind::from_message("invalid pointer to integer cast"),
            UndefinedBehaviorKind::InvalidPointerOffset
        );
        assert_eq!(
            UndefinedBehaviorKind::from_message("pointer offset overflow"),
            UndefinedBehaviorKind::InvalidPointerOffset
        );
    }

    #[test]
    fn test_parse_test_results_failed_status() {
        // Test the "failed" match arm in parse_test_results (line 343)
        let stdout = "test foo::bar ... failed\n";
        let mut results = Vec::new();
        parse_test_results(stdout, &mut results);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, MiriTestStatus::Failed);
    }

    #[test]
    fn test_parse_stderr_skips_ub_in_error() {
        // Test the !message.contains("Undefined Behavior") check (line 304)
        let stderr = "error: some regular error\nerror: Undefined Behavior: use after free\n";
        let mut ub_list = Vec::new();
        let mut diagnostics = Vec::new();
        parse_stderr(stderr, &mut ub_list, &mut diagnostics).unwrap();

        // Should have 1 UB and 1 non-UB diagnostic
        assert_eq!(ub_list.len(), 1);
        assert_eq!(diagnostics.len(), 1);
        assert!(!diagnostics[0].message.contains("Undefined Behavior"));
    }

    #[test]
    fn test_parse_stderr_backtrace_line() {
        // Test the backtrace parsing: line.trim().starts_with("at ") || line.contains("in <")
        let stderr = r#"
error: Undefined Behavior: out of bounds
  at src/lib.rs:10:5
  in <module::func as Trait>::method
"#;
        let mut ub_list = Vec::new();
        let mut diagnostics = Vec::new();
        parse_stderr(stderr, &mut ub_list, &mut diagnostics).unwrap();

        assert_eq!(ub_list.len(), 1);
        assert!(
            ub_list[0].backtrace.len() >= 2,
            "Should have backtrace entries"
        );
    }

    #[test]
    fn test_summary_passed_count() {
        // Test the == comparison in lines 218, 222, 226
        let output = MiriOutput {
            stdout: "test a ... ok\ntest b ... FAILED\ntest c ... ignored\n".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            duration: std::time::Duration::from_secs(1),
            has_errors: false,
        };

        let parsed = parse_miri_output(&output).unwrap();
        assert_eq!(parsed.summary.total_tests, 3);
        assert_eq!(parsed.summary.passed, 1);
        assert_eq!(parsed.summary.failed, 1);
        assert_eq!(parsed.summary.ignored, 1);
    }
}
