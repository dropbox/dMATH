//! Z4 Proof - Proof production and export
//!
//! Generate and export proofs in various formats (LFSC, Alethe).
//!
//! ## Alethe Format
//!
//! The Alethe format is the standard proof format for SMT solvers,
//! supported by carcara and SMTCoq. It uses SMT-LIB syntax with
//! additional proof commands.
//!
//! ## Example
//!
//! ```text
//! ; Declarations from problem
//! (declare-const a Int)
//! (declare-const b Int)
//!
//! ; Proof commands
//! (assume h1 (= a b))
//! (assume h2 (not (= a a)))
//! (step t1 (cl (= a a)) :rule refl)
//! (step t2 (cl) :rule resolution :premises (h2 t1))
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

use num_bigint::Sign;
pub use z4_core::proof::{AletheRule, Proof, ProofId, ProofStep};
use z4_core::{Constant, Symbol, TermData, TermId, TermStore};

/// Export a proof to LFSC format
#[must_use]
pub fn export_lfsc(_proof: &Proof) -> String {
    // TODO: Implement LFSC export
    String::new()
}

/// Export a proof to Alethe format
///
/// Converts a Z4 proof to the Alethe format, which can be verified
/// by carcara or other Alethe-compatible checkers.
///
/// # Arguments
///
/// * `proof` - The proof to export
/// * `terms` - The term store containing all terms referenced in the proof
///
/// # Returns
///
/// A string containing the Alethe proof commands
#[must_use]
pub fn export_alethe(proof: &Proof, terms: &TermStore) -> String {
    let mut output = String::new();
    let printer = AlethePrinter::new(terms);

    for (idx, step) in proof.steps.iter().enumerate() {
        let step_id = ProofId(idx as u32);
        output.push_str(&printer.format_step(step, step_id));
        output.push('\n');
    }

    output
}

/// Alethe proof printer
struct AlethePrinter<'a> {
    terms: &'a TermStore,
}

impl<'a> AlethePrinter<'a> {
    fn new(terms: &'a TermStore) -> Self {
        Self { terms }
    }

    /// Format a proof step as an Alethe command
    fn format_step(&self, step: &ProofStep, id: ProofId) -> String {
        match step {
            ProofStep::Assume(term_id) => {
                let term_str = self.format_term(*term_id);
                format!("(assume {} {})", id, term_str)
            }

            ProofStep::Resolution {
                pivot,
                clause1,
                clause2,
            } => {
                // Resolution is a special case - we format it as a step
                format!(
                    "(step {} (cl) :rule resolution :premises ({} {}) :args ({}))",
                    id,
                    clause1,
                    clause2,
                    self.format_term(*pivot)
                )
            }

            ProofStep::TheoryLemma { theory, clause } => {
                let clause_str = self.format_clause(clause);
                // Map theory name to appropriate Alethe rule
                let rule = match theory.as_str() {
                    "EUF" => "eq_congruent",
                    "LRA" => "la_generic",
                    "LIA" => "lia_generic",
                    "BV" => "bv_bitblast",
                    _ => "trust",
                };
                format!("(step {} {} :rule {})", id, clause_str, rule)
            }

            ProofStep::Step {
                rule,
                clause,
                premises,
                args,
            } => {
                let clause_str = self.format_clause(clause);
                let mut result = format!("(step {} {} :rule {}", id, clause_str, rule);

                if !premises.is_empty() {
                    let premises_str: Vec<String> =
                        premises.iter().map(|p| p.to_string()).collect();
                    result.push_str(&format!(" :premises ({})", premises_str.join(" ")));
                }

                if !args.is_empty() {
                    let args_str: Vec<String> = args.iter().map(|a| self.format_term(*a)).collect();
                    result.push_str(&format!(" :args ({})", args_str.join(" ")));
                }

                result.push(')');
                result
            }

            ProofStep::Anchor {
                end_step,
                variables,
            } => {
                let mut result = format!("(anchor :step {}", end_step);
                if !variables.is_empty() {
                    let vars_str: Vec<String> = variables
                        .iter()
                        .map(|(name, sort)| format!("({} {})", name, sort))
                        .collect();
                    result.push_str(&format!(" :args ({})", vars_str.join(" ")));
                }
                result.push(')');
                result
            }
        }
    }

    /// Format a clause (list of literals) as "(cl lit1 lit2 ...)"
    fn format_clause(&self, clause: &[TermId]) -> String {
        if clause.is_empty() {
            "(cl)".to_string()
        } else {
            let lits: Vec<String> = clause.iter().map(|t| self.format_term(*t)).collect();
            format!("(cl {})", lits.join(" "))
        }
    }

    /// Format a term as an SMT-LIB expression
    fn format_term(&self, term_id: TermId) -> String {
        let term = self.terms.get(term_id);
        self.format_term_data(term)
    }

    /// Format term data recursively
    fn format_term_data(&self, term: &TermData) -> String {
        match term {
            TermData::Var(name, _) => quote_symbol(name),

            TermData::Const(c) => self.format_constant(c),

            TermData::Not(inner) => {
                format!("(not {})", self.format_term(*inner))
            }

            TermData::Ite(cond, then_br, else_br) => {
                format!(
                    "(ite {} {} {})",
                    self.format_term(*cond),
                    self.format_term(*then_br),
                    self.format_term(*else_br)
                )
            }

            TermData::App(sym, args) => {
                let name = self.format_symbol(sym);
                if args.is_empty() {
                    name
                } else {
                    let args_str: Vec<String> = args.iter().map(|&a| self.format_term(a)).collect();
                    format!("({} {})", name, args_str.join(" "))
                }
            }

            TermData::Let(bindings, body) => {
                let bindings_str: Vec<String> = bindings
                    .iter()
                    .map(|(name, term)| {
                        format!("({} {})", quote_symbol(name), self.format_term(*term))
                    })
                    .collect();
                format!(
                    "(let ({}) {})",
                    bindings_str.join(" "),
                    self.format_term(*body)
                )
            }
        }
    }

    /// Format a constant value
    fn format_constant(&self, c: &Constant) -> String {
        match c {
            Constant::Bool(true) => "true".to_string(),
            Constant::Bool(false) => "false".to_string(),
            Constant::Int(i) => {
                if i.sign() == Sign::Minus {
                    format!("(- {})", i.magnitude())
                } else {
                    i.to_string()
                }
            }
            Constant::Rational(r) => {
                let rat = &r.0;
                if rat.is_integer() {
                    if rat.numer().sign() == Sign::Minus {
                        format!("(- {}.0)", rat.numer().magnitude())
                    } else {
                        format!("{}.0", rat.numer())
                    }
                } else if rat.numer().sign() == Sign::Minus {
                    format!("(- (/ {}.0 {}.0))", rat.numer().magnitude(), rat.denom())
                } else {
                    format!("(/ {}.0 {}.0)", rat.numer(), rat.denom())
                }
            }
            Constant::BitVec { value, width } => {
                // Use binary format for bitvectors
                format!("#b{:0>width$b}", value, width = *width as usize)
            }
            Constant::String(s) => {
                // Escape quotes in strings
                format!("\"{}\"", s.replace('\"', "\"\""))
            }
        }
    }

    /// Format a function symbol
    fn format_symbol(&self, sym: &Symbol) -> String {
        match sym {
            Symbol::Named(name) => quote_symbol(name),
            Symbol::Indexed(name, indices) => {
                let indices_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                format!("(_ {} {})", quote_symbol(name), indices_str.join(" "))
            }
        }
    }
}

/// Quote a symbol if needed (contains special characters or is a reserved word)
fn quote_symbol(name: &str) -> String {
    // Reserved words in SMT-LIB that need quoting
    const RESERVED: &[&str] = &[
        "let",
        "forall",
        "exists",
        "match",
        "par",
        "_",
        "!",
        "as",
        "BINARY",
        "DECIMAL",
        "HEXADECIMAL",
        "NUMERAL",
        "STRING",
        "assert",
        "check-sat",
        "declare-const",
        "declare-datatype",
        "declare-datatypes",
        "declare-fun",
        "declare-sort",
        "define-fun",
        "define-fun-rec",
        "define-funs-rec",
        "define-sort",
        "echo",
        "exit",
        "get-assertions",
        "get-assignment",
        "get-info",
        "get-model",
        "get-option",
        "get-proof",
        "get-unsat-assumptions",
        "get-unsat-core",
        "get-value",
        "pop",
        "push",
        "reset",
        "reset-assertions",
        "set-info",
        "set-logic",
        "set-option",
    ];

    let needs_quoting = name.is_empty()
        || name.starts_with(|c: char| c.is_ascii_digit())
        || RESERVED.contains(&name)
        || name.contains(|c: char| !is_symbol_char(c));

    if needs_quoting {
        format!("|{}|", name)
    } else {
        name.to_string()
    }
}

/// Check if a character is valid in an unquoted SMT-LIB symbol
fn is_symbol_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            '+' | '-'
                | '/'
                | '*'
                | '='
                | '%'
                | '?'
                | '!'
                | '.'
                | '$'
                | '_'
                | '~'
                | '&'
                | '^'
                | '<'
                | '>'
                | '@'
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use z4_core::proof::AletheRule;

    #[test]
    fn test_quote_symbol() {
        assert_eq!(quote_symbol("x"), "x");
        assert_eq!(quote_symbol("myVar"), "myVar");
        assert_eq!(quote_symbol("let"), "|let|");
        assert_eq!(quote_symbol("123abc"), "|123abc|");
        assert_eq!(quote_symbol("with space"), "|with space|");
        assert_eq!(quote_symbol(""), "||");
    }

    #[test]
    fn test_is_symbol_char() {
        assert!(is_symbol_char('a'));
        assert!(is_symbol_char('Z'));
        assert!(is_symbol_char('0'));
        assert!(is_symbol_char('_'));
        assert!(is_symbol_char('+'));
        assert!(!is_symbol_char(' '));
        assert!(!is_symbol_char('('));
        assert!(!is_symbol_char(')'));
    }

    #[test]
    fn test_empty_proof() {
        let proof = Proof::new();
        let terms = TermStore::new();
        let output = export_alethe(&proof, &terms);
        assert_eq!(output, "");
    }

    #[test]
    fn test_assume_step() {
        let mut terms = TermStore::new();
        let x = terms.mk_var("x", z4_core::Sort::Bool);

        let mut proof = Proof::new();
        proof.add_assume(x, None);

        let output = export_alethe(&proof, &terms);
        assert_eq!(output, "(assume t0 x)\n");
    }

    #[test]
    fn test_step_with_premises() {
        let mut terms = TermStore::new();
        let x = terms.mk_var("x", z4_core::Sort::Bool);
        let not_x = terms.mk_not(x);

        let mut proof = Proof::new();
        let h1 = proof.add_assume(x, None);
        let h2 = proof.add_assume(not_x, None);
        proof.add_rule_step(
            AletheRule::Resolution,
            vec![], // empty clause = contradiction
            vec![h1, h2],
            vec![x], // pivot
        );

        let output = export_alethe(&proof, &terms);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "(assume t0 x)");
        assert_eq!(lines[1], "(assume t1 (not x))");
        assert!(lines[2].contains(":rule resolution"));
        assert!(lines[2].contains(":premises (t0 t1)"));
    }

    #[test]
    fn test_theory_lemma() {
        let mut terms = TermStore::new();
        let a = terms.mk_var("a", z4_core::Sort::Int);
        let b = terms.mk_var("b", z4_core::Sort::Int);
        let eq = terms.mk_eq(a, b);

        let mut proof = Proof::new();
        proof.add_theory_lemma("EUF", vec![eq]);

        let output = export_alethe(&proof, &terms);
        assert!(output.contains("eq_congruent"));
        assert!(output.contains("(= a b)"));
    }

    #[test]
    fn test_format_rational() {
        let mut terms = TermStore::new();
        let rat = terms.mk_rational(num_rational::BigRational::new(1.into(), 2.into()));

        let printer = AlethePrinter::new(&terms);
        let output = printer.format_term(rat);
        assert_eq!(output, "(/ 1.0 2.0)");
    }

    #[test]
    fn test_format_negative_rational() {
        let mut terms = TermStore::new();
        let rat = terms.mk_rational(num_rational::BigRational::new((-3).into(), 4.into()));

        let printer = AlethePrinter::new(&terms);
        let output = printer.format_term(rat);
        assert_eq!(output, "(- (/ 3.0 4.0))");
    }

    #[test]
    fn test_format_bitvector() {
        let mut terms = TermStore::new();
        let bv = terms.mk_bitvec(5.into(), 4);

        let printer = AlethePrinter::new(&terms);
        let output = printer.format_term(bv);
        assert_eq!(output, "#b0101");
    }

    /// Integration test that verifies proofs with carcara (if available)
    #[test]
    fn test_carcara_verification() {
        use std::process::Command;

        // Check if carcara is available
        let carcara_path = std::env::var("CARCARA_PATH")
            .unwrap_or_else(|_| "reference/carcara/target/release/carcara".to_string());

        if !std::path::Path::new(&carcara_path).exists() {
            eprintln!("Skipping carcara test: {} not found", carcara_path);
            return;
        }

        // Create a simple proof: a AND (NOT a) is unsatisfiable
        let mut terms = TermStore::new();
        let a = terms.mk_var("a", z4_core::Sort::Bool);
        let not_a = terms.mk_not(a);

        let mut proof = Proof::new();
        let h0 = proof.add_assume(a, Some("h0".to_string()));
        let h1 = proof.add_assume(not_a, Some("h1".to_string()));
        // Use hole rule since we're just testing structure
        proof.add_step(ProofStep::Step {
            rule: AletheRule::Hole,
            clause: vec![],
            premises: vec![h0, h1],
            args: vec![],
        });

        let proof_str = export_alethe(&proof, &terms);

        // Write problem and proof to temp files
        let temp_dir = std::env::temp_dir();
        let problem_path = temp_dir.join("z4_test_problem.smt2");
        let proof_path = temp_dir.join("z4_test_proof.alethe");

        let problem = "(set-logic QF_UF)\n(declare-const a Bool)\n(assert a)\n(assert (not a))\n(check-sat)\n";
        std::fs::write(&problem_path, problem).expect("Failed to write problem file");
        std::fs::write(&proof_path, &proof_str).expect("Failed to write proof file");

        // Run carcara
        let output = Command::new(&carcara_path)
            .arg("check")
            .arg(&proof_path)
            .arg(&problem_path)
            .output()
            .expect("Failed to execute carcara");

        // Clean up temp files
        let _ = std::fs::remove_file(&problem_path);
        let _ = std::fs::remove_file(&proof_path);

        // Check result - "holey" is acceptable (means proof structure is valid but contains holes)
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        assert!(
            output.status.success(),
            "Carcara verification failed!\nstdout: {}\nstderr: {}\nproof:\n{}",
            stdout,
            stderr,
            proof_str
        );
    }
}
