//! z4-lean-bridge: Lean 5 integration for Z4 SMT solver
//!
//! This crate provides a bridge between Z4 and Lean 5, enabling:
//! - Export of Z4 formulas to Lean syntax
//! - Verification of Z4 SAT/UNSAT results in Lean
//! - Generation of Lean tactics that call Z4
//! - Integration with Mathlib and other Lean libraries
//!
//! # Architecture
//!
//! The bridge works via command-line integration:
//! 1. Z4 formulas are exported to Lean syntax
//! 2. Lean is invoked to check/prove the formulas
//! 3. Results are parsed back into Rust
//!
//! # Example
//!
//! ```ignore
//! use z4_lean_bridge::{LeanBridge, LeanExporter};
//! use z4_core::{TermStore, Sort};
//!
//! let mut store = TermStore::new();
//! let x = store.declare("x", Sort::Bool);
//! let y = store.declare("y", Sort::Bool);
//! let formula = store.mk_and(x, y);
//!
//! // Export to Lean
//! let exporter = LeanExporter::new(&store);
//! let lean_code = exporter.export_term(formula);
//! assert_eq!(lean_code, "x && y");
//!
//! // Or verify a SAT result
//! let bridge = LeanBridge::discover()?;
//! let result = bridge.verify_sat(&formula, &model)?;
//! ```

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};

use thiserror::Error;
use z4_core::{Constant, Sort, Symbol, TermData, TermId, TermStore};

/// Errors that can occur when interacting with Lean.
#[derive(Debug, Error)]
pub enum LeanError {
    /// Lean executable not found
    #[error("failed to discover Lean; set `LEAN_BIN` or ensure `lake` is on PATH")]
    NotFound,

    /// IO error when running Lean
    #[error("Lean execution failed: {0}")]
    Io(#[from] std::io::Error),

    /// Lean output was not valid UTF-8
    #[error("Lean output was not valid UTF-8")]
    NonUtf8Output,

    /// Lean compilation/type check failed
    #[error("Lean compilation failed: {message}")]
    CompilationFailed { message: String },

    /// Lean proof failed
    #[error("Lean proof failed: {message}")]
    ProofFailed { message: String },

    /// Unsupported term for export
    #[error("unsupported term for Lean export: {0}")]
    UnsupportedTerm(String),

    /// Temporary file error
    #[error("temporary file error: {0}")]
    TempFile(String),
}

/// Backend for running Lean commands.
#[derive(Clone, Debug)]
pub enum LeanBackend {
    /// Run Lean directly via the `lean` executable
    LeanExe { lean_bin: PathBuf },
    /// Run Lean via Lake (Lean's package manager)
    Lake {
        lake_bin: PathBuf,
        project_dir: Option<PathBuf>,
    },
}

impl LeanBackend {
    /// Discover a Lean backend.
    ///
    /// Priority:
    /// 1. `LEAN_BIN` env var
    /// 2. `lean` on PATH
    /// 3. `lake` on PATH (for project-based usage)
    pub fn discover() -> Result<Self, LeanError> {
        if let Some(p) = env_path("LEAN_BIN") {
            return Ok(LeanBackend::LeanExe { lean_bin: p });
        }

        if let Ok(p) = which("lean") {
            return Ok(LeanBackend::LeanExe { lean_bin: p });
        }

        if let Ok(p) = which("lake") {
            return Ok(LeanBackend::Lake {
                lake_bin: p,
                project_dir: None,
            });
        }

        Err(LeanError::NotFound)
    }
}

/// Arguments for running Lean.
#[derive(Clone, Debug, Default)]
pub struct LeanArgs {
    /// Working directory for Lean execution
    pub cwd: Option<PathBuf>,
    /// Additional Lean options
    pub extra_args: Vec<OsString>,
    /// Timeout in seconds (0 = no timeout)
    pub timeout_secs: u32,
}

/// Result of checking a statement in Lean.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeanResult {
    /// Statement was proved/type-checked successfully
    Proved,
    /// Statement was disproved (counterexample found)
    Disproved { counterexample: Option<String> },
    /// Lean couldn't determine the result
    Unknown,
    /// Execution timed out
    Timeout,
    /// Type check or compilation error
    Error { message: String },
}

impl LeanResult {
    /// Returns true if the statement was proved.
    pub fn is_proved(&self) -> bool {
        matches!(self, LeanResult::Proved)
    }

    /// Returns true if this is an error result.
    pub fn is_error(&self) -> bool {
        matches!(self, LeanResult::Error { .. })
    }
}

/// Complete output from a Lean run.
#[derive(Clone, Debug)]
pub struct LeanRun {
    /// The backend used
    pub backend: LeanBackend,
    /// Exit status of the process
    pub exit_status: ExitStatus,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Parsed result
    pub result: LeanResult,
}

impl LeanRun {
    /// Combined stdout and stderr.
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
}

/// Bridge to Lean 5 proof assistant.
///
/// This struct manages the Lean installation and provides methods for
/// verifying Z4 results in Lean.
#[derive(Clone, Debug)]
pub struct LeanBridge {
    backend: LeanBackend,
}

impl LeanBridge {
    /// Create a new LeanBridge with the specified backend.
    pub fn new(backend: LeanBackend) -> Self {
        Self { backend }
    }

    /// Try to discover a Lean installation and create a bridge.
    pub fn discover() -> Result<Self, LeanError> {
        Ok(Self::new(LeanBackend::discover()?))
    }

    /// Get the backend being used.
    pub fn backend(&self) -> &LeanBackend {
        &self.backend
    }

    /// Check if a Lean file type-checks.
    pub fn check_file(&self, path: impl AsRef<Path>, args: LeanArgs) -> Result<LeanRun, LeanError> {
        let path = path.as_ref();

        let mut cmd = match &self.backend {
            LeanBackend::LeanExe { lean_bin } => {
                let mut cmd = Command::new(lean_bin);
                cmd.arg(path);
                cmd
            }
            LeanBackend::Lake {
                lake_bin,
                project_dir,
            } => {
                let mut cmd = Command::new(lake_bin);
                cmd.arg("env");
                cmd.arg("lean");
                cmd.arg(path);
                if let Some(dir) = project_dir {
                    cmd.current_dir(dir);
                }
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

        let stdout = String::from_utf8(stdout).map_err(|_| LeanError::NonUtf8Output)?;
        let stderr = String::from_utf8(stderr).map_err(|_| LeanError::NonUtf8Output)?;
        let result = parse_lean_result(&stdout, &stderr, status.code());

        Ok(LeanRun {
            backend: self.backend.clone(),
            exit_status: status,
            stdout,
            stderr,
            result,
        })
    }

    /// Check if Lean code (as a string) type-checks by writing to a temp file.
    pub fn check_code(&self, code: &str, args: LeanArgs) -> Result<LeanRun, LeanError> {
        // Create a temporary file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("z4_lean_check_{}.lean", std::process::id()));

        std::fs::write(&temp_file, code).map_err(|e| LeanError::TempFile(e.to_string()))?;

        let result = self.check_file(&temp_file, args);

        // Clean up temp file (ignore errors)
        let _ = std::fs::remove_file(&temp_file);

        result
    }

    /// Verify that a Z4 SAT result is correct by checking the model in Lean.
    ///
    /// This generates Lean code that asserts the model satisfies the formula,
    /// and checks if Lean can verify this.
    pub fn verify_sat(
        &self,
        store: &TermStore,
        formula: TermId,
        model: &[(String, bool)],
    ) -> Result<LeanRun, LeanError> {
        let exporter = LeanExporter::new(store);
        let lean_code = exporter.generate_sat_verification(formula, model)?;
        self.check_code(&lean_code, LeanArgs::default())
    }

    /// Verify that a Z4 UNSAT result is plausible by checking in Lean.
    ///
    /// Note: Full UNSAT verification requires checking the DRAT proof,
    /// which is beyond simple Lean type-checking.
    pub fn verify_unsat(&self, store: &TermStore, formula: TermId) -> Result<LeanRun, LeanError> {
        let exporter = LeanExporter::new(store);
        let lean_code = exporter.generate_unsat_check(formula)?;
        self.check_code(&lean_code, LeanArgs::default())
    }
}

/// Exporter for converting Z4 terms to Lean syntax.
pub struct LeanExporter<'a> {
    store: &'a TermStore,
}

impl<'a> LeanExporter<'a> {
    /// Create a new exporter with a reference to a term store.
    pub fn new(store: &'a TermStore) -> Self {
        Self { store }
    }

    /// Export a term to Lean syntax.
    pub fn export_term(&self, term: TermId) -> Result<String, LeanError> {
        self.export_term_inner(term, 0)
    }

    /// Export a term with precedence tracking for parenthesization.
    fn export_term_inner(&self, term: TermId, parent_prec: u8) -> Result<String, LeanError> {
        let data = self.store.get(term).clone();
        let sort = self.store.sort(term);

        let (result, prec) = match &data {
            TermData::Const(c) => (self.export_constant(c)?, 100),
            TermData::Var(name, _) => (sanitize_lean_name(name), 100),
            TermData::Not(inner) => {
                let inner_str = self.export_term_inner(*inner, 90)?;
                (format!("!{}", inner_str), 90)
            }
            TermData::Ite(cond, then_br, else_br) => {
                let cond_str = self.export_term_inner(*cond, 0)?;
                let then_str = self.export_term_inner(*then_br, 0)?;
                let else_str = self.export_term_inner(*else_br, 0)?;
                (
                    format!("if {} then {} else {}", cond_str, then_str, else_str),
                    10,
                )
            }
            TermData::App(sym, args) => self.export_app(sym, args, sort)?,
            TermData::Let(bindings, body) => {
                let mut result = String::new();
                for (name, value) in bindings {
                    let value_str = self.export_term_inner(*value, 0)?;
                    result.push_str(&format!(
                        "let {} := {} in ",
                        sanitize_lean_name(name),
                        value_str
                    ));
                }
                let body_str = self.export_term_inner(*body, 0)?;
                result.push_str(&body_str);
                (result, 5)
            }
        };

        // Add parentheses if needed
        if prec < parent_prec {
            Ok(format!("({})", result))
        } else {
            Ok(result)
        }
    }

    /// Export a constant to Lean syntax.
    fn export_constant(&self, c: &Constant) -> Result<String, LeanError> {
        match c {
            Constant::Bool(true) => Ok("true".to_string()),
            Constant::Bool(false) => Ok("false".to_string()),
            Constant::Int(i) => {
                if i.sign() == num_bigint::Sign::Minus {
                    Ok(format!("({})", i))
                } else {
                    Ok(i.to_string())
                }
            }
            Constant::Rational(r) => {
                let numer = r.0.numer();
                let denom = r.0.denom();
                if denom == &num_bigint::BigInt::from(1) {
                    if numer.sign() == num_bigint::Sign::Minus {
                        Ok(format!("({})", numer))
                    } else {
                        Ok(numer.to_string())
                    }
                } else {
                    Ok(format!("({} / {})", numer, denom))
                }
            }
            Constant::BitVec { value, width } => Ok(format!("(BitVec.ofNat {} {})", width, value)),
            Constant::String(s) => Ok(format!("\"{}\"", escape_lean_string(s))),
        }
    }

    /// Export a function application to Lean syntax.
    fn export_app(
        &self,
        sym: &Symbol,
        args: &[TermId],
        _sort: &Sort,
    ) -> Result<(String, u8), LeanError> {
        let name = sym.name();

        // Handle indexed symbols (like extract, repeat)
        if let Symbol::Indexed(base, indices) = sym {
            return self.export_indexed_app(base, indices, args);
        }

        // Handle common SMT-LIB operations
        match (name, args.len()) {
            // Boolean operations
            ("and", _) => {
                let parts: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.export_term_inner(*a, 35))
                    .collect();
                Ok((parts?.join(" && "), 35))
            }
            ("or", _) => {
                let parts: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.export_term_inner(*a, 30))
                    .collect();
                Ok((parts?.join(" || "), 30))
            }
            ("=>", 2) | ("implies", 2) => {
                let lhs = self.export_term_inner(args[0], 25)?;
                let rhs = self.export_term_inner(args[1], 24)?;
                Ok((format!("{} -> {}", lhs, rhs), 25))
            }
            ("xor", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("{} ^^ {}", lhs, rhs), 50))
            }

            // Equality
            ("=", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("{} = {}", lhs, rhs), 50))
            }
            ("distinct", _) => {
                // Generate pairwise inequalities
                let mut parts = Vec::new();
                for i in 0..args.len() {
                    for j in (i + 1)..args.len() {
                        let lhs = self.export_term_inner(args[i], 50)?;
                        let rhs = self.export_term_inner(args[j], 50)?;
                        parts.push(format!("{} != {}", lhs, rhs));
                    }
                }
                Ok((parts.join(" && "), 35))
            }

            // Arithmetic operations
            ("+", _) => {
                let parts: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.export_term_inner(*a, 65))
                    .collect();
                Ok((parts?.join(" + "), 65))
            }
            ("-", 1) => {
                let inner = self.export_term_inner(args[0], 70)?;
                Ok((format!("-{}", inner), 70))
            }
            ("-", 2) => {
                let lhs = self.export_term_inner(args[0], 65)?;
                let rhs = self.export_term_inner(args[1], 66)?;
                Ok((format!("{} - {}", lhs, rhs), 65))
            }
            ("*", _) => {
                let parts: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.export_term_inner(*a, 70))
                    .collect();
                Ok((parts?.join(" * "), 70))
            }
            ("div", 2) | ("/", 2) => {
                let lhs = self.export_term_inner(args[0], 70)?;
                let rhs = self.export_term_inner(args[1], 71)?;
                Ok((format!("{} / {}", lhs, rhs), 70))
            }
            ("mod", 2) => {
                let lhs = self.export_term_inner(args[0], 70)?;
                let rhs = self.export_term_inner(args[1], 71)?;
                Ok((format!("{} % {}", lhs, rhs), 70))
            }

            // Comparisons
            ("<", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("{} < {}", lhs, rhs), 50))
            }
            ("<=", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("{} <= {}", lhs, rhs), 50))
            }
            (">", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("{} > {}", lhs, rhs), 50))
            }
            (">=", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("{} >= {}", lhs, rhs), 50))
            }

            // Bitvector operations
            ("bvadd", 2) => {
                let lhs = self.export_term_inner(args[0], 65)?;
                let rhs = self.export_term_inner(args[1], 65)?;
                Ok((format!("{} + {}", lhs, rhs), 65))
            }
            ("bvsub", 2) => {
                let lhs = self.export_term_inner(args[0], 65)?;
                let rhs = self.export_term_inner(args[1], 66)?;
                Ok((format!("{} - {}", lhs, rhs), 65))
            }
            ("bvmul", 2) => {
                let lhs = self.export_term_inner(args[0], 70)?;
                let rhs = self.export_term_inner(args[1], 70)?;
                Ok((format!("{} * {}", lhs, rhs), 70))
            }
            ("bvand", 2) => {
                let lhs = self.export_term_inner(args[0], 56)?;
                let rhs = self.export_term_inner(args[1], 56)?;
                Ok((format!("{} &&& {}", lhs, rhs), 56))
            }
            ("bvor", 2) => {
                let lhs = self.export_term_inner(args[0], 55)?;
                let rhs = self.export_term_inner(args[1], 55)?;
                Ok((format!("{} ||| {}", lhs, rhs), 55))
            }
            ("bvxor", 2) => {
                let lhs = self.export_term_inner(args[0], 58)?;
                let rhs = self.export_term_inner(args[1], 58)?;
                Ok((format!("{} ^^^ {}", lhs, rhs), 58))
            }
            ("bvnot", 1) => {
                let inner = self.export_term_inner(args[0], 90)?;
                Ok((format!("~~~{}", inner), 90))
            }
            ("bvneg", 1) => {
                let inner = self.export_term_inner(args[0], 70)?;
                Ok((format!("-{}", inner), 70))
            }
            ("bvult", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("BitVec.ult {} {}", lhs, rhs), 50))
            }
            ("bvule", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("BitVec.ule {} {}", lhs, rhs), 50))
            }
            ("bvslt", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("BitVec.slt {} {}", lhs, rhs), 50))
            }
            ("bvsle", 2) => {
                let lhs = self.export_term_inner(args[0], 50)?;
                let rhs = self.export_term_inner(args[1], 50)?;
                Ok((format!("BitVec.sle {} {}", lhs, rhs), 50))
            }

            // Array operations
            ("select", 2) => {
                let arr = self.export_term_inner(args[0], 100)?;
                let idx = self.export_term_inner(args[1], 0)?;
                Ok((format!("{}[{}]", arr, idx), 100))
            }
            ("store", 3) => {
                let arr = self.export_term_inner(args[0], 100)?;
                let idx = self.export_term_inner(args[1], 0)?;
                let val = self.export_term_inner(args[2], 0)?;
                Ok((format!("Array.set {} {} {}", arr, idx, val), 80))
            }

            // Default: function application
            _ => {
                let lean_name = sanitize_lean_name(name);
                if args.is_empty() {
                    Ok((lean_name, 100))
                } else {
                    let arg_strs: Result<Vec<_>, _> = args
                        .iter()
                        .map(|a| self.export_term_inner(*a, 100))
                        .collect();
                    Ok((format!("{} {}", lean_name, arg_strs?.join(" ")), 80))
                }
            }
        }
    }

    /// Export an indexed function application (like extract, repeat).
    fn export_indexed_app(
        &self,
        base: &str,
        indices: &[u32],
        args: &[TermId],
    ) -> Result<(String, u8), LeanError> {
        match (base, indices.len(), args.len()) {
            ("extract", 2, 1) => {
                let hi = indices[0];
                let lo = indices[1];
                let inner = self.export_term_inner(args[0], 100)?;
                Ok((format!("BitVec.extractLsb {} {} {}", hi, lo, inner), 80))
            }
            ("repeat", 1, 1) => {
                let n = indices[0];
                let inner = self.export_term_inner(args[0], 100)?;
                Ok((format!("BitVec.replicate {} {}", n, inner), 80))
            }
            ("zero_extend", 1, 1) => {
                let n = indices[0];
                let inner = self.export_term_inner(args[0], 100)?;
                Ok((format!("BitVec.zeroExtend {} {}", n, inner), 80))
            }
            ("sign_extend", 1, 1) => {
                let n = indices[0];
                let inner = self.export_term_inner(args[0], 100)?;
                Ok((format!("BitVec.signExtend {} {}", n, inner), 80))
            }
            _ => {
                // Generic indexed function
                let arg_strs: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.export_term_inner(*a, 100))
                    .collect();
                let idx_str = indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                Ok((format!("{}_{} {}", base, idx_str, arg_strs?.join(" ")), 80))
            }
        }
    }

    /// Export a sort to Lean type syntax.
    #[allow(clippy::only_used_in_recursion)]
    pub fn export_sort(&self, sort: &Sort) -> String {
        match sort {
            Sort::Bool => "Bool".to_string(),
            Sort::Int => "Int".to_string(),
            Sort::Real => "Real".to_string(),
            Sort::BitVec(w) => format!("BitVec {}", w),
            Sort::Array(idx, elem) => {
                format!("Array {} {}", self.export_sort(idx), self.export_sort(elem))
            }
            Sort::String => "String".to_string(),
            Sort::FloatingPoint(eb, sb) => format!("FloatingPoint {} {}", eb, sb),
            Sort::Uninterpreted(name) => sanitize_lean_name(name),
            Sort::Datatype(name) => sanitize_lean_name(name),
        }
    }

    /// Generate Lean code to verify a SAT model.
    pub fn generate_sat_verification(
        &self,
        formula: TermId,
        model: &[(String, bool)],
    ) -> Result<String, LeanError> {
        let mut code = String::new();

        // Header
        code.push_str("-- Z4 SAT Verification\n");
        code.push_str("-- Auto-generated by z4-lean-bridge\n\n");

        // Define the variables with their values from the model
        for (name, value) in model {
            let lean_name = sanitize_lean_name(name);
            code.push_str(&format!("def {} : Bool := {}\n", lean_name, value));
        }
        code.push('\n');

        // Export the formula
        let formula_str = self.export_term(formula)?;

        // Create a theorem that the formula evaluates to true
        code.push_str(&format!(
            "theorem sat_verification : {} = true := by native_decide\n",
            formula_str
        ));

        Ok(code)
    }

    /// Generate Lean code to check an UNSAT claim.
    ///
    /// Note: This only checks that the formula is well-typed and
    /// semantically valid Lean code. Full UNSAT verification requires
    /// DRAT proof checking.
    pub fn generate_unsat_check(&self, formula: TermId) -> Result<String, LeanError> {
        let mut code = String::new();

        // Header
        code.push_str("-- Z4 UNSAT Check\n");
        code.push_str("-- Auto-generated by z4-lean-bridge\n\n");

        // Collect free variables
        let vars = self.collect_variables(formula);

        // Declare variables
        for (name, sort) in &vars {
            let lean_name = sanitize_lean_name(name);
            let lean_sort = self.export_sort(sort);
            code.push_str(&format!("variable ({} : {})\n", lean_name, lean_sort));
        }
        if !vars.is_empty() {
            code.push('\n');
        }

        // Export the formula
        let formula_str = self.export_term(formula)?;

        // Create a #check to verify the formula is well-typed
        code.push_str(&format!("#check ({} : Bool)\n", formula_str));

        Ok(code)
    }

    /// Collect all free variables in a term.
    fn collect_variables(&self, term: TermId) -> Vec<(String, Sort)> {
        let mut vars = Vec::new();
        let mut seen = std::collections::HashSet::new();
        self.collect_variables_inner(term, &mut vars, &mut seen);
        vars
    }

    fn collect_variables_inner(
        &self,
        term: TermId,
        vars: &mut Vec<(String, Sort)>,
        seen: &mut std::collections::HashSet<String>,
    ) {
        let data = self.store.get(term).clone();
        let sort = self.store.sort(term).clone();

        match &data {
            TermData::Var(name, _) => {
                if !seen.contains(name) {
                    seen.insert(name.clone());
                    vars.push((name.clone(), sort));
                }
            }
            TermData::Const(_) => {}
            TermData::Not(inner) => {
                self.collect_variables_inner(*inner, vars, seen);
            }
            TermData::Ite(c, t, e) => {
                self.collect_variables_inner(*c, vars, seen);
                self.collect_variables_inner(*t, vars, seen);
                self.collect_variables_inner(*e, vars, seen);
            }
            TermData::App(_, args) => {
                for arg in args {
                    self.collect_variables_inner(*arg, vars, seen);
                }
            }
            TermData::Let(bindings, body) => {
                for (_, value) in bindings {
                    self.collect_variables_inner(*value, vars, seen);
                }
                self.collect_variables_inner(*body, vars, seen);
            }
        }
    }
}

/// Generate the Z4 tactic for Lean.
///
/// This generates Lean code that provides tactics for calling Z4
/// from within Lean proofs.
pub fn generate_z4_tactic() -> String {
    r#"/-
  Z4 Decision Tactics for Lean 5
  Auto-generated by z4-lean-bridge

  These tactics call the Z4 SMT solver to decide propositional
  and bitvector goals.
-/

import Lean

namespace Z4

/-- Call Z4 SAT solver to decide propositional goals. -/
syntax "z4_decide" : tactic

/-- Call Z4 SMT solver for theory goals. -/
syntax "z4_smt" : tactic

/-- Call Z4 with bitvector decision procedure. -/
syntax "z4_bv" : tactic

-- Placeholder implementations using native_decide
-- In production, these would call Z4 via FFI or subprocess

macro_rules
| `(tactic| z4_decide) => `(tactic| native_decide)

macro_rules
| `(tactic| z4_smt) => `(tactic| native_decide)

macro_rules
| `(tactic| z4_bv) => `(tactic| native_decide)

end Z4
"#
    .to_string()
}

/// Generate a Lean project structure for Z4 integration.
pub fn generate_project_files() -> Vec<(&'static str, String)> {
    vec![
        ("lakefile.lean", generate_lakefile()),
        ("Z4.lean", generate_z4_tactic()),
        ("Z4/Basic.lean", generate_basic_module()),
    ]
}

fn generate_lakefile() -> String {
    r#"import Lake
open Lake DSL

package z4 where
  version := v!"0.1.0"

lean_lib Z4 where
  -- add library configuration options here

@[default_target]
lean_exe z4_test where
  root := `Main
"#
    .to_string()
}

fn generate_basic_module() -> String {
    r#"/-
  Z4 Basic Module
  Provides basic types and utilities for Z4 integration.
-/

namespace Z4

/-- Result of a Z4 solve call. -/
inductive SolveResult
  | sat (model : List (String × Bool))
  | unsat
  | unknown

end Z4
"#
    .to_string()
}

/// Parse Lean output to determine the result.
fn parse_lean_result(stdout: &str, stderr: &str, exit_code: Option<i32>) -> LeanResult {
    let combined = format!("{}\n{}", stdout, stderr);

    // Check for success (exit code 0 with no errors)
    if exit_code == Some(0) && !combined.contains("error:") {
        return LeanResult::Proved;
    }

    // Check for type errors or other compilation failures
    if combined.contains("error:") {
        let message = combined
            .lines()
            .find(|l| l.contains("error:"))
            .unwrap_or("unknown error")
            .to_string();
        return LeanResult::Error { message };
    }

    // Unknown result
    LeanResult::Unknown
}

/// Sanitize a name for use in Lean.
fn sanitize_lean_name(name: &str) -> String {
    // Lean identifiers can contain letters, digits, underscores, and apostrophes
    // They must start with a letter or underscore
    let mut result = String::with_capacity(name.len());
    let mut first = true;

    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '\'' {
            if first && c.is_ascii_digit() {
                result.push('_');
            }
            result.push(c);
            first = false;
        } else if c == '!' || c == '?' {
            // Keep these for Lean naming conventions
            result.push(c);
            first = false;
        } else {
            // Replace other characters with underscore
            result.push('_');
            first = false;
        }
    }

    if result.is_empty() {
        return "_unnamed".to_string();
    }

    // Check for Lean reserved words
    if is_lean_reserved(&result) {
        format!("{}_", result)
    } else {
        result
    }
}

/// Check if a name is a Lean reserved word.
fn is_lean_reserved(name: &str) -> bool {
    matches!(
        name,
        "def"
            | "theorem"
            | "lemma"
            | "axiom"
            | "example"
            | "structure"
            | "class"
            | "instance"
            | "inductive"
            | "where"
            | "with"
            | "do"
            | "if"
            | "then"
            | "else"
            | "match"
            | "let"
            | "in"
            | "have"
            | "show"
            | "by"
            | "fun"
            | "forall"
            | "exists"
            | "true"
            | "false"
            | "Type"
            | "Prop"
            | "Sort"
            | "Bool"
            | "Nat"
            | "Int"
    )
}

/// Escape a string for use in Lean string literals.
fn escape_lean_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            _ => result.push(c),
        }
    }
    result
}

/// Get a path from an environment variable.
fn env_path(name: &str) -> Option<PathBuf> {
    let v = std::env::var_os(name)?;
    if v.is_empty() {
        return None;
    }
    Some(PathBuf::from(v))
}

/// Find an executable on PATH.
fn which(bin: &str) -> Result<PathBuf, ()> {
    let path_var = std::env::var_os("PATH").ok_or(())?;
    for dir in std::env::split_paths(&path_var) {
        let p = dir.join(bin);
        if is_executable(&p) {
            return Ok(p);
        }
    }
    Err(())
}

/// Check if a path is executable.
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
        return p.is_file();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_lean_name() {
        assert_eq!(sanitize_lean_name("x"), "x");
        assert_eq!(sanitize_lean_name("my_var"), "my_var");
        assert_eq!(sanitize_lean_name("x'"), "x'");
        assert_eq!(sanitize_lean_name("123"), "_123");
        assert_eq!(sanitize_lean_name("a-b"), "a_b");
        assert_eq!(sanitize_lean_name("def"), "def_");
        assert_eq!(sanitize_lean_name(""), "_unnamed");
    }

    #[test]
    fn test_escape_lean_string() {
        assert_eq!(escape_lean_string("hello"), "hello");
        assert_eq!(escape_lean_string("hello\"world"), "hello\\\"world");
        assert_eq!(escape_lean_string("a\nb"), "a\\nb");
    }

    #[test]
    fn test_export_constant() {
        let store = TermStore::new();
        let exporter = LeanExporter::new(&store);

        assert_eq!(
            exporter.export_constant(&Constant::Bool(true)).unwrap(),
            "true"
        );
        assert_eq!(
            exporter.export_constant(&Constant::Bool(false)).unwrap(),
            "false"
        );
        assert_eq!(
            exporter
                .export_constant(&Constant::Int(num_bigint::BigInt::from(42)))
                .unwrap(),
            "42"
        );
        assert_eq!(
            exporter
                .export_constant(&Constant::Int(num_bigint::BigInt::from(-42)))
                .unwrap(),
            "(-42)"
        );
    }

    #[test]
    fn test_export_sort() {
        let store = TermStore::new();
        let exporter = LeanExporter::new(&store);

        assert_eq!(exporter.export_sort(&Sort::Bool), "Bool");
        assert_eq!(exporter.export_sort(&Sort::Int), "Int");
        assert_eq!(exporter.export_sort(&Sort::Real), "Real");
        assert_eq!(exporter.export_sort(&Sort::BitVec(32)), "BitVec 32");
        assert_eq!(
            exporter.export_sort(&Sort::Array(Box::new(Sort::Int), Box::new(Sort::Bool))),
            "Array Int Bool"
        );
    }

    #[test]
    fn test_lean_tactic_generation() {
        let tactic = generate_z4_tactic();
        assert!(tactic.contains("z4_decide"));
        assert!(tactic.contains("z4_smt"));
        assert!(tactic.contains("native_decide"));
    }

    #[test]
    fn test_lean_result_parsing() {
        // Success case
        assert_eq!(parse_lean_result("", "", Some(0)), LeanResult::Proved);

        // Error case
        assert!(matches!(
            parse_lean_result("error: type mismatch", "", Some(1)),
            LeanResult::Error { .. }
        ));
    }

    #[test]
    fn test_export_simple_formula() {
        let mut store = TermStore::new();

        // Create: x && y
        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let formula = store.mk_and(vec![x, y]);

        let exporter = LeanExporter::new(&store);
        let result = exporter.export_term(formula).unwrap();
        assert!(result.contains("&&"));
    }

    #[test]
    fn test_export_arithmetic() {
        use num_bigint::BigInt;
        let mut store = TermStore::new();

        // Create: x + 1 < 10
        let x = store.mk_var("x", Sort::Int);
        let one = store.mk_int(BigInt::from(1));
        let ten = store.mk_int(BigInt::from(10));
        let sum = store.mk_add(vec![x, one]);
        let lt = store.mk_lt(sum, ten);

        let exporter = LeanExporter::new(&store);
        let result = exporter.export_term(lt).unwrap();
        assert!(result.contains("+"));
        assert!(result.contains("<"));
    }

    #[test]
    fn test_sat_verification_generation() {
        let mut store = TermStore::new();

        let x = store.mk_var("x", Sort::Bool);
        let y = store.mk_var("y", Sort::Bool);
        let formula = store.mk_and(vec![x, y]);

        let exporter = LeanExporter::new(&store);
        let code = exporter
            .generate_sat_verification(formula, &[("x".to_string(), true), ("y".to_string(), true)])
            .unwrap();

        assert!(code.contains("def x : Bool := true"));
        assert!(code.contains("def y : Bool := true"));
        assert!(code.contains("theorem sat_verification"));
    }
}
