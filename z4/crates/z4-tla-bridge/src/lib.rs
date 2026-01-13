//! z4-tla-bridge: TLA+/TLC integration helpers
//!
//! This crate provides a small wrapper around the TLC model checker so Z4 can:
//! - Run TLA+ specs from tests or tooling
//! - Parse TLC's textual output into a structured outcome
//!
//! The runner supports two backends:
//! - `tlc` executable on `PATH` (as used in `docs/PHASE1_EXECUTION_ROADMAP.md`)
//! - `java -cp tla2tools.jar tlc2.TLC ...` when `TLA2TOOLS_JAR` is provided

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TlcError {
    #[error("failed to discover a TLC runner; set `TLC_BIN` or `TLA2TOOLS_JAR`")]
    NotFound,

    #[error("TLC execution failed: {0}")]
    Io(#[from] std::io::Error),

    #[error("TLC output was not valid UTF-8")]
    NonUtf8Output,
}

#[derive(Clone, Debug)]
pub enum TlcBackend {
    /// Run an installed `tlc` executable (or wrapper script).
    Cli { tlc_bin: PathBuf },
    /// Run TLC via `java -cp <jar> tlc2.TLC`.
    JavaJar {
        java_bin: PathBuf,
        tla2tools_jar: PathBuf,
    },
}

impl TlcBackend {
    /// Discover a TLC backend.
    ///
    /// Priority:
    /// 1) `TLC_BIN` env var
    /// 2) `tlc` found on `PATH`
    /// 3) `TLA2TOOLS_JAR` env var (uses `java` on `PATH`)
    pub fn discover() -> Result<Self, TlcError> {
        if let Some(p) = env_path("TLC_BIN") {
            return Ok(TlcBackend::Cli { tlc_bin: p });
        }

        if let Ok(p) = which::which("tlc") {
            return Ok(TlcBackend::Cli { tlc_bin: p });
        }

        if let Some(jar) = env_path("TLA2TOOLS_JAR") {
            return Ok(TlcBackend::JavaJar {
                java_bin: PathBuf::from("java"),
                tla2tools_jar: jar,
            });
        }

        Err(TlcError::NotFound)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TlcArgs {
    pub config: Option<PathBuf>,
    pub cwd: Option<PathBuf>,
    pub workers: Option<u32>,
    pub max_heap_mb: Option<u32>,
    pub extra_args: Vec<OsString>,
}

/// Error code categories matching TLC's internal taxonomy.
///
/// These codes provide machine-readable classification of TLC errors
/// for programmatic handling by verification toolchains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TlcErrorCode {
    /// Config file parsing error (CFG_PARSE_*)
    ConfigParse,
    /// TLA+ spec parsing error (TLA_PARSE_*)
    SpecParse,
    /// Type mismatch error (TLC_TYPE_*)
    TypeError,
    /// Deadlock detected (TLC_DEADLOCK)
    Deadlock,
    /// Invariant violation (TLC_INVARIANT_*)
    InvariantViolation,
    /// Liveness/temporal property violation (TLC_LIVENESS_*)
    LivenessViolation,
    /// Assertion failure (TLC_ASSERTION_*)
    AssertionFailure,
    /// Resource exhaustion (state space, memory)
    ResourceExhausted,
    /// Internal TLC error
    InternalError,
    /// Unknown error
    Unknown,
}

/// A structured representation of a TLC violation.
///
/// This provides machine-readable error information suitable for
/// AI agents and automated tooling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TlcViolation {
    /// The error code category
    pub code: TlcErrorCode,
    /// Human-readable error message
    pub message: String,
    /// The name of the violated property (if applicable)
    pub property_name: Option<String>,
    /// The counterexample trace (state sequence leading to violation)
    pub trace: Option<Vec<String>>,
    /// Suggested fix or action
    pub suggestion: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TlcOutcome {
    /// Model checking completed successfully with no errors
    NoError,
    /// Deadlock detected (no enabled transitions from some state)
    Deadlock,
    /// Safety property (invariant) violation
    InvariantViolation { name: Option<String> },
    /// Liveness/temporal property violation
    LivenessViolation,
    /// Assertion failure in spec
    AssertionFailure { message: Option<String> },
    /// Type error in spec or config
    TypeError,
    /// Parse error in spec
    ParseError,
    /// Config file error
    ConfigError,
    /// State space exhausted (out of memory, too many states)
    StateSpaceExhausted,
    /// TLC execution failed with non-zero exit
    ExecutionFailed { exit_code: Option<i32> },
    /// Unknown outcome
    Unknown { exit_code: Option<i32> },
}

impl TlcOutcome {
    /// Returns the error code category for this outcome.
    pub fn error_code(&self) -> Option<TlcErrorCode> {
        match self {
            TlcOutcome::NoError => None,
            TlcOutcome::Deadlock => Some(TlcErrorCode::Deadlock),
            TlcOutcome::InvariantViolation { .. } => Some(TlcErrorCode::InvariantViolation),
            TlcOutcome::LivenessViolation => Some(TlcErrorCode::LivenessViolation),
            TlcOutcome::AssertionFailure { .. } => Some(TlcErrorCode::AssertionFailure),
            TlcOutcome::TypeError => Some(TlcErrorCode::TypeError),
            TlcOutcome::ParseError => Some(TlcErrorCode::SpecParse),
            TlcOutcome::ConfigError => Some(TlcErrorCode::ConfigParse),
            TlcOutcome::StateSpaceExhausted => Some(TlcErrorCode::ResourceExhausted),
            TlcOutcome::ExecutionFailed { .. } => Some(TlcErrorCode::InternalError),
            TlcOutcome::Unknown { .. } => Some(TlcErrorCode::Unknown),
        }
    }

    /// Returns true if this outcome indicates a successful run.
    pub fn is_success(&self) -> bool {
        matches!(self, TlcOutcome::NoError)
    }

    /// Returns true if this outcome indicates a property violation.
    pub fn is_violation(&self) -> bool {
        matches!(
            self,
            TlcOutcome::Deadlock
                | TlcOutcome::InvariantViolation { .. }
                | TlcOutcome::LivenessViolation
                | TlcOutcome::AssertionFailure { .. }
        )
    }

    /// Returns true if this outcome indicates a spec/config error.
    pub fn is_spec_error(&self) -> bool {
        matches!(
            self,
            TlcOutcome::TypeError | TlcOutcome::ParseError | TlcOutcome::ConfigError
        )
    }
}

#[derive(Clone, Debug)]
pub struct TlcRun {
    pub backend: TlcBackend,
    pub args: TlcArgs,
    pub spec: PathBuf,
    pub exit_status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
    pub outcome: TlcOutcome,
}

impl TlcRun {
    pub fn combined_output(&self) -> String {
        let mut s = String::new();
        if !self.stdout.is_empty() {
            s.push_str(&self.stdout);
        }
        if !self.stderr.is_empty() {
            if !s.is_empty() && !s.ends_with('\n') {
                s.push('\n');
            }
            s.push_str(&self.stderr);
        }
        s
    }

    /// Returns true if model checking completed successfully.
    pub fn is_success(&self) -> bool {
        self.outcome.is_success()
    }

    /// Returns true if a property violation was detected.
    pub fn is_violation(&self) -> bool {
        self.outcome.is_violation()
    }

    /// Extract a structured violation from this run.
    ///
    /// Returns `None` if the run was successful.
    pub fn violation(&self) -> Option<TlcViolation> {
        extract_violation(&self.stdout, &self.stderr, self.exit_status.code())
    }
}

#[derive(Clone, Debug)]
pub struct TlcRunner {
    backend: TlcBackend,
}

impl TlcRunner {
    pub fn new(backend: TlcBackend) -> Self {
        Self { backend }
    }

    pub fn discover() -> Result<Self, TlcError> {
        Ok(Self::new(TlcBackend::discover()?))
    }

    pub fn backend(&self) -> &TlcBackend {
        &self.backend
    }

    pub fn run(&self, spec: impl AsRef<Path>, args: TlcArgs) -> Result<TlcRun, TlcError> {
        let spec = spec.as_ref().to_path_buf();

        let mut cmd = match &self.backend {
            TlcBackend::Cli { tlc_bin } => {
                let mut cmd = Command::new(tlc_bin);
                // Prefer standard ordering: options first, then the spec module/file.
                if let Some(cfg) = &args.config {
                    cmd.arg("-config").arg(cfg);
                }
                if let Some(workers) = args.workers {
                    cmd.arg("-workers").arg(workers.to_string());
                }
                cmd.arg(&spec);
                cmd
            }
            TlcBackend::JavaJar {
                java_bin,
                tla2tools_jar,
            } => {
                let mut cmd = Command::new(java_bin);
                if let Some(max_heap_mb) = args.max_heap_mb {
                    cmd.arg(format!("-Xmx{max_heap_mb}m"));
                }
                cmd.arg("-cp").arg(tla2tools_jar);
                cmd.arg("tlc2.TLC");
                if let Some(cfg) = &args.config {
                    cmd.arg("-config").arg(cfg);
                }
                if let Some(workers) = args.workers {
                    cmd.arg("-workers").arg(workers.to_string());
                }
                cmd.arg(&spec);
                cmd
            }
        };

        cmd.args(&args.extra_args);
        if let Some(cwd) = &args.cwd {
            cmd.current_dir(cwd);
        }

        let Output {
            status,
            stdout,
            stderr,
        } = cmd.output()?;

        let stdout = String::from_utf8(stdout).map_err(|_| TlcError::NonUtf8Output)?;
        let stderr = String::from_utf8(stderr).map_err(|_| TlcError::NonUtf8Output)?;
        let outcome = parse_tlc_outcome(&stdout, &stderr, status.code());

        Ok(TlcRun {
            backend: self.backend.clone(),
            args,
            spec,
            exit_status: status,
            stdout,
            stderr,
            outcome,
        })
    }
}

pub fn parse_tlc_outcome(stdout: &str, stderr: &str, exit_code: Option<i32>) -> TlcOutcome {
    let combined = if stderr.is_empty() {
        stdout
    } else if stdout.is_empty() {
        stderr
    } else {
        // Avoid allocating for the fast path by only using stderr/stdout checks below.
        ""
    };

    let text = if combined.is_empty() {
        // Fall back to allocation only when both are present.
        let mut s = String::with_capacity(stdout.len() + 1 + stderr.len());
        s.push_str(stdout);
        if !s.ends_with('\n') {
            s.push('\n');
        }
        s.push_str(stderr);
        s
    } else {
        combined.to_string()
    };

    // Success marker.
    if text.contains("Model checking completed. No error has been found.") {
        return TlcOutcome::NoError;
    }

    // Common failure markers (ordered by specificity).

    // Deadlock
    if text.contains("Error: Deadlock reached.") || text.contains("TLC_DEADLOCK") {
        return TlcOutcome::Deadlock;
    }

    // Invariant violation
    if text.contains("is violated.") && text.contains("Error: Invariant") {
        return TlcOutcome::InvariantViolation {
            name: extract_invariant_name(&text),
        };
    }

    // Assertion failure
    if text.contains("Error: The following assertion failed")
        || text.contains("Assertion failed")
        || text.contains("ASSERT")
    {
        return TlcOutcome::AssertionFailure {
            message: extract_assertion_message(&text),
        };
    }

    // Liveness/temporal properties
    if text.contains("Temporal properties were violated")
        || text.contains("liveness properties were violated")
        || text.contains("stuttering")
    {
        return TlcOutcome::LivenessViolation;
    }

    // State space exhaustion
    if text.contains("Out of memory")
        || text.contains("too many states")
        || text.contains("state space too large")
        || text.contains("SANY_STATE_SPACE")
    {
        return TlcOutcome::StateSpaceExhausted;
    }

    // Config file errors
    if text.contains("CFG_PARSE")
        || text.contains("Error reading configuration file")
        || text.contains("Configuration file error")
        || text.contains(".cfg")
            && (text.contains("error") || text.contains("Error") || text.contains("cannot"))
    {
        return TlcOutcome::ConfigError;
    }

    // Type errors
    if text.contains("TLC_TYPE")
        || text.contains("was not in the domain")
        || text.contains("is not a")
        || text.contains("type mismatch")
        || text.contains("Type mismatch")
    {
        return TlcOutcome::TypeError;
    }

    // Parse errors
    if text.contains("TLA_PARSE")
        || text.contains("Parse error")
        || text.contains("TLC parsing")
        || text.contains("Syntax error")
        || text.contains("SANY error")
    {
        return TlcOutcome::ParseError;
    }

    if exit_code != Some(0) {
        return TlcOutcome::ExecutionFailed { exit_code };
    }

    TlcOutcome::Unknown { exit_code }
}

/// Extract a structured violation from TLC output.
///
/// This provides more detailed error information including the
/// counterexample trace when available.
pub fn extract_violation(
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
) -> Option<TlcViolation> {
    let outcome = parse_tlc_outcome(stdout, stderr, exit_code);
    if outcome.is_success() {
        return None;
    }

    let code = outcome.error_code()?;
    let text = format!("{stdout}\n{stderr}");

    let (property_name, message) = match &outcome {
        TlcOutcome::Deadlock => (
            None,
            "Deadlock reached: no enabled transitions from current state".to_string(),
        ),
        TlcOutcome::InvariantViolation { name } => {
            let msg = name
                .as_ref()
                .map(|n| format!("Invariant {n} is violated"))
                .unwrap_or_else(|| "Invariant violation".to_string());
            (name.clone(), msg)
        }
        TlcOutcome::LivenessViolation => (None, "Temporal/liveness property violated".to_string()),
        TlcOutcome::AssertionFailure { message } => {
            let msg = message
                .clone()
                .unwrap_or_else(|| "Assertion failed".to_string());
            (None, msg)
        }
        TlcOutcome::TypeError => (None, "Type error in specification".to_string()),
        TlcOutcome::ParseError => (None, "Parse error in specification".to_string()),
        TlcOutcome::ConfigError => (None, "Configuration file error".to_string()),
        TlcOutcome::StateSpaceExhausted => (None, "State space exhausted".to_string()),
        TlcOutcome::ExecutionFailed { exit_code } => {
            let msg = exit_code
                .map(|c| format!("TLC execution failed with exit code {c}"))
                .unwrap_or_else(|| "TLC execution failed".to_string());
            (None, msg)
        }
        TlcOutcome::Unknown { .. } => (None, "Unknown TLC error".to_string()),
        TlcOutcome::NoError => return None,
    };

    let trace = extract_trace(&text);
    let suggestion = suggest_fix(&outcome, &text);

    Some(TlcViolation {
        code,
        message,
        property_name,
        trace,
        suggestion,
    })
}

fn extract_assertion_message(text: &str) -> Option<String> {
    // Look for assertion message patterns
    for line in text.lines() {
        if line.contains("Assertion failed") || line.contains("ASSERT") {
            // Extract the relevant part
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Extract the counterexample trace from TLC output.
fn extract_trace(text: &str) -> Option<Vec<String>> {
    let mut trace = Vec::new();
    let mut in_trace = false;
    let mut current_state = String::new();

    for line in text.lines() {
        // TLC trace typically starts with "State 1:" or similar
        if line.starts_with("State ") && line.contains(':') {
            if !current_state.is_empty() {
                trace.push(current_state.trim().to_string());
            }
            in_trace = true;
            current_state = line.to_string();
        } else if in_trace {
            // Continue accumulating state info
            if line.starts_with('/') && line.contains('\\') {
                // This looks like a variable assignment
                current_state.push('\n');
                current_state.push_str(line);
            } else if line.trim().is_empty() && !current_state.is_empty() {
                // End of state
                trace.push(current_state.trim().to_string());
                current_state = String::new();
            } else if !line.trim().is_empty()
                && !line.contains("Error")
                && !line.contains("Finished")
            {
                current_state.push('\n');
                current_state.push_str(line);
            }
        }
    }

    // Don't forget the last state
    if !current_state.is_empty() {
        trace.push(current_state.trim().to_string());
    }

    if trace.is_empty() {
        None
    } else {
        Some(trace)
    }
}

/// Suggest a fix based on the error type.
fn suggest_fix(outcome: &TlcOutcome, text: &str) -> Option<String> {
    match outcome {
        TlcOutcome::Deadlock => {
            if text.contains("state =") && (text.contains("SAT") || text.contains("UNSAT")) {
                Some(
                    "This may be an expected terminal state. Consider adding a TERMINAL \
                     declaration in your config or adding a self-loop in terminal states."
                        .to_string(),
                )
            } else {
                Some(
                    "Check that all states have at least one enabled transition, \
                     or mark terminal states explicitly."
                        .to_string(),
                )
            }
        }
        TlcOutcome::InvariantViolation { name } => {
            let inv_name = name.as_deref().unwrap_or("the invariant");
            Some(format!(
                "Review the counterexample trace to understand how {inv_name} was violated. \
                 Check preconditions and action guards."
            ))
        }
        TlcOutcome::TypeError => Some(
            "Check that all operators receive arguments of the expected type. \
             Verify CONSTANT declarations match their usage."
                .to_string(),
        ),
        TlcOutcome::ParseError => Some(
            "Check TLA+ syntax. Common issues: missing EXTENDS, unbalanced delimiters, \
             invalid operator names."
                .to_string(),
        ),
        TlcOutcome::ConfigError => Some(
            "Check config file syntax. Ensure CONSTANTS, SPECIFICATION, and INVARIANT \
             declarations are properly formatted."
                .to_string(),
        ),
        TlcOutcome::StateSpaceExhausted => Some(
            "Reduce state space by using symmetry sets, constraining CONSTANTS, \
             or adding state constraints with CONSTRAINT."
                .to_string(),
        ),
        _ => None,
    }
}

fn extract_invariant_name(text: &str) -> Option<String> {
    // Typical TLC line:
    // "Error: Invariant TypeInvariant is violated."
    let needle_start = "Error: Invariant ";
    let needle_end = " is violated.";
    let start = text.find(needle_start)? + needle_start.len();
    let rest = &text[start..];
    let end = rest.find(needle_end)?;
    let name = rest[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    let v = std::env::var_os(name)?;
    if v.is_empty() {
        return None;
    }
    Some(PathBuf::from(v))
}

// Minimal `which` implementation avoids pulling in extra dependencies.
mod which {
    use std::path::{Path, PathBuf};

    pub fn which(bin: &str) -> Result<PathBuf, ()> {
        let path_var = std::env::var_os("PATH").ok_or(())?;
        for dir in std::env::split_paths(&path_var) {
            let p = dir.join(bin);
            if is_executable(&p) {
                return Ok(p);
            }
        }
        Err(())
    }

    fn is_executable(p: &Path) -> bool {
        if !p.is_file() {
            return false;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = p.metadata() {
                return (meta.permissions().mode() & 0o111) != 0;
            }
        }
        #[cfg(not(unix))]
        {
            // On Windows we rely on PATH search semantics; existence is good enough.
            return p.is_file();
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_no_error() {
        let out = "Model checking completed. No error has been found.\n";
        assert_eq!(parse_tlc_outcome(out, "", Some(0)), TlcOutcome::NoError);
    }

    #[test]
    fn parse_deadlock() {
        let out = "Error: Deadlock reached.\n";
        assert_eq!(parse_tlc_outcome(out, "", Some(1)), TlcOutcome::Deadlock);
    }

    #[test]
    fn parse_invariant_violation_extracts_name() {
        let out = "Error: Invariant TypeInvariant is violated.\n";
        assert_eq!(
            parse_tlc_outcome(out, "", Some(1)),
            TlcOutcome::InvariantViolation {
                name: Some("TypeInvariant".to_string())
            }
        );
    }

    #[test]
    fn parse_execution_failed_falls_back_to_exit_code() {
        let out = "Some unknown output\n";
        assert_eq!(
            parse_tlc_outcome(out, "", Some(2)),
            TlcOutcome::ExecutionFailed { exit_code: Some(2) }
        );
    }

    #[test]
    fn can_skip_real_tlc_run_when_missing() {
        // This test ensures discovery failure is well-typed; it does not try to run TLC.
        let _ = TlcBackend::discover().err();
    }

    // Tests for new outcome variants

    #[test]
    fn parse_assertion_failure() {
        let out = "Error: The following assertion failed\n";
        assert!(matches!(
            parse_tlc_outcome(out, "", Some(1)),
            TlcOutcome::AssertionFailure { .. }
        ));
    }

    #[test]
    fn parse_liveness_violation() {
        let out = "Temporal properties were violated.\n";
        assert_eq!(
            parse_tlc_outcome(out, "", Some(1)),
            TlcOutcome::LivenessViolation
        );
    }

    #[test]
    fn parse_config_error() {
        let out = "Error reading configuration file test.cfg\n";
        assert_eq!(parse_tlc_outcome(out, "", Some(1)), TlcOutcome::ConfigError);
    }

    #[test]
    fn parse_state_space_exhausted() {
        let out = "Out of memory while exploring state space\n";
        assert_eq!(
            parse_tlc_outcome(out, "", Some(1)),
            TlcOutcome::StateSpaceExhausted
        );
    }

    #[test]
    fn parse_type_error() {
        let out = "TLC_TYPE error: value was not in the domain\n";
        assert_eq!(parse_tlc_outcome(out, "", Some(1)), TlcOutcome::TypeError);
    }

    #[test]
    fn parse_parse_error() {
        let out = "Parse error in module Test at line 10\n";
        assert_eq!(parse_tlc_outcome(out, "", Some(1)), TlcOutcome::ParseError);
    }

    // Tests for TlcOutcome helper methods

    #[test]
    fn outcome_is_success() {
        assert!(TlcOutcome::NoError.is_success());
        assert!(!TlcOutcome::Deadlock.is_success());
        assert!(!TlcOutcome::InvariantViolation { name: None }.is_success());
    }

    #[test]
    fn outcome_is_violation() {
        assert!(!TlcOutcome::NoError.is_violation());
        assert!(TlcOutcome::Deadlock.is_violation());
        assert!(TlcOutcome::InvariantViolation { name: None }.is_violation());
        assert!(TlcOutcome::LivenessViolation.is_violation());
        assert!(TlcOutcome::AssertionFailure { message: None }.is_violation());
        assert!(!TlcOutcome::TypeError.is_violation());
        assert!(!TlcOutcome::ParseError.is_violation());
    }

    #[test]
    fn outcome_is_spec_error() {
        assert!(!TlcOutcome::NoError.is_spec_error());
        assert!(!TlcOutcome::Deadlock.is_spec_error());
        assert!(TlcOutcome::TypeError.is_spec_error());
        assert!(TlcOutcome::ParseError.is_spec_error());
        assert!(TlcOutcome::ConfigError.is_spec_error());
    }

    #[test]
    fn outcome_error_code() {
        assert!(TlcOutcome::NoError.error_code().is_none());
        assert_eq!(
            TlcOutcome::Deadlock.error_code(),
            Some(TlcErrorCode::Deadlock)
        );
        assert_eq!(
            TlcOutcome::InvariantViolation { name: None }.error_code(),
            Some(TlcErrorCode::InvariantViolation)
        );
    }

    // Tests for extract_violation

    #[test]
    fn extract_violation_returns_none_for_success() {
        let out = "Model checking completed. No error has been found.\n";
        assert!(extract_violation(out, "", Some(0)).is_none());
    }

    #[test]
    fn extract_violation_deadlock() {
        let out = "Error: Deadlock reached.\nState 1:\n/\\ x = 1\n";
        let violation = extract_violation(out, "", Some(1)).unwrap();
        assert_eq!(violation.code, TlcErrorCode::Deadlock);
        assert!(violation.message.contains("Deadlock"));
        assert!(violation.suggestion.is_some());
    }

    #[test]
    fn extract_violation_invariant() {
        let out = "Error: Invariant SatCorrect is violated.\nState 1:\n/\\ state = \"SAT\"\n";
        let violation = extract_violation(out, "", Some(1)).unwrap();
        assert_eq!(violation.code, TlcErrorCode::InvariantViolation);
        assert_eq!(violation.property_name, Some("SatCorrect".to_string()));
        assert!(violation.suggestion.is_some());
    }

    #[test]
    fn extract_trace_from_output() {
        let out = r#"
State 1: <Initial predicate>
/\ assignment = [v1 |-> "UNDEF", v2 |-> "UNDEF"]
/\ state = "PROPAGATING"

State 2: <Propagate>
/\ assignment = [v1 |-> "FALSE", v2 |-> "UNDEF"]
/\ state = "PROPAGATING"

Error: Invariant violated.
"#;
        let trace = extract_trace(out);
        assert!(trace.is_some());
        let trace = trace.unwrap();
        assert!(trace.len() >= 2);
        assert!(trace[0].contains("State 1"));
        assert!(trace[1].contains("State 2"));
    }

    #[test]
    fn suggest_fix_for_deadlock_with_terminal_state() {
        let outcome = TlcOutcome::Deadlock;
        let text = "state = \"UNSAT\"";
        let suggestion = suggest_fix(&outcome, text);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("terminal state"));
    }

    #[test]
    fn suggest_fix_for_invariant_violation() {
        let outcome = TlcOutcome::InvariantViolation {
            name: Some("TypeInvariant".to_string()),
        };
        let suggestion = suggest_fix(&outcome, "");
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("TypeInvariant"));
    }
}
