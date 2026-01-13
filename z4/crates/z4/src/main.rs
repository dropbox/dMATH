//! Z4 - A high-performance SMT solver in Rust
//!
//! Usage: z4 [OPTIONS] [FILE]

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

use z4_chc::{KindConfig, KindResult, KindSolver, PdrConfig, PdrResult, PdrSolver};
use z4_dpll::Executor;
use z4_frontend::{parse, Command};
use z4_sat::{parse_dimacs, SolveResult as SatSolveResult};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        // Interactive mode
        run_interactive();
    } else if args[1] == "--help" || args[1] == "-h" {
        print_help();
    } else if args[1] == "--version" || args[1] == "-v" {
        print_version();
    } else if args[1] == "--chc" {
        // CHC mode: --chc [--verbose] FILE or --chc FILE [--verbose]
        let verbose = args.iter().any(|a| a == "--verbose");
        let file = args.iter().skip(2).find(|a| !a.starts_with('-')).cloned();
        match file {
            Some(f) => run_chc(&f, verbose),
            None => {
                eprintln!("Error: --chc requires a file argument");
                std::process::exit(1);
            }
        }
    } else if args[1].starts_with('-') {
        eprintln!("Unknown option: {}", args[1]);
        print_help();
        std::process::exit(1);
    } else {
        // File mode
        run_file(&args[1]);
    }
}

fn print_help() {
    println!("Z4 - A high-performance SMT solver in Rust");
    println!();
    println!("Usage: z4 [OPTIONS] [FILE]");
    println!();
    println!("Options:");
    println!("  -h, --help          Print this help message");
    println!("  -v, --version       Print version information");
    println!("  --chc FILE          Run CHC (Horn clause) solver on FILE");
    println!("  --verbose           Enable verbose output (for --chc mode)");
    println!();
    println!("File format auto-detection:");
    println!("  .cnf / 'p cnf'      DIMACS CNF (SAT competition format)");
    println!("  (set-logic HORN)    CHC (Horn clauses, PDR/IC3 solver)");
    println!("  Otherwise           SMT-LIB 2.6");
    println!();
    println!("If no file is given, runs in interactive mode.");
}

fn print_version() {
    println!("Z4 version {}", env!("CARGO_PKG_VERSION"));
}

fn run_interactive() {
    use std::io::IsTerminal;

    let stdin = io::stdin();

    // If stdin is piped (not a TTY), read all at once and process like a file
    if !stdin.is_terminal() {
        let mut content = String::new();
        if let Err(e) = stdin.lock().read_line(&mut content) {
            eprintln!("Error reading input: {}", e);
            return;
        }
        // Read rest of stdin
        use std::io::Read;
        if let Err(e) = stdin.lock().read_to_string(&mut content) {
            eprintln!("Error reading input: {}", e);
            return;
        }

        // Check for DIMACS CNF format (content-based detection for stdin)
        if is_dimacs_format(&content) {
            run_dimacs_from_content(&content);
            return;
        }

        // Check for HORN logic
        if is_horn_logic(&content) {
            run_chc_from_content(&content, false);
            return;
        }

        // Standard DPLL(T) path
        match parse(&content) {
            Ok(commands) => {
                let mut executor = Executor::new();
                for cmd in &commands {
                    match cmd {
                        Command::Exit => return,
                        _ => match executor.execute(cmd) {
                            Ok(Some(output)) => println!("{}", output),
                            Ok(None) => {}
                            Err(e) => eprintln!("(error \"{}\")", e),
                        },
                    }
                }
            }
            Err(e) => eprintln!("(error \"parse error: {}\")", e),
        }
        return;
    }

    // True interactive mode (TTY)
    println!(
        "Z4 version {} - Interactive mode",
        env!("CARGO_PKG_VERSION")
    );
    println!("Type SMT-LIB commands, or (exit) to quit.");

    let mut stdout = io::stdout();
    let mut input_buffer = String::new();
    let mut executor = Executor::new();

    loop {
        print!("> ");
        let _ = stdout.flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }

        input_buffer.push_str(&line);

        match parse(&input_buffer) {
            Ok(commands) => {
                input_buffer.clear();
                for cmd in &commands {
                    if matches!(cmd, Command::Exit) {
                        return;
                    }
                    match executor.execute(cmd) {
                        Ok(Some(output)) => println!("{}", output),
                        Ok(None) => {}
                        Err(e) => eprintln!("(error \"{}\")", e),
                    }
                }
            }
            Err(_) => {
                let opens = input_buffer.matches('(').count();
                let closes = input_buffer.matches(')').count();
                if opens <= closes {
                    if let Ok(commands) = parse(&input_buffer) {
                        input_buffer.clear();
                        for cmd in &commands {
                            if matches!(cmd, Command::Exit) {
                                return;
                            }
                            match executor.execute(cmd) {
                                Ok(Some(output)) => println!("{}", output),
                                Ok(None) => {}
                                Err(e) => eprintln!("(error \"{}\")", e),
                            }
                        }
                    } else {
                        eprintln!("(error \"parse error\")");
                        input_buffer.clear();
                    }
                }
            }
        }
    }
}

fn run_file(path: &str) {
    match fs::read_to_string(path) {
        Ok(content) => {
            // Check for DIMACS CNF format first (by extension or content)
            if has_cnf_extension(path) || is_dimacs_format(&content) {
                run_dimacs_from_content(&content);
                return;
            }

            // Check for HORN logic and route to CHC solver if found
            if is_horn_logic(&content) {
                run_chc_from_content(&content, false);
                return;
            }

            // Parse and execute all commands (standard DPLL(T) path)
            match parse(&content) {
                Ok(commands) => {
                    let mut executor = Executor::new();
                    for cmd in &commands {
                        match cmd {
                            Command::Exit => return,
                            _ => match executor.execute(cmd) {
                                Ok(Some(output)) => println!("{}", output),
                                Ok(None) => {}
                                Err(e) => {
                                    eprintln!("(error \"{}\")", e);
                                }
                            },
                        }
                    }
                }
                Err(e) => {
                    eprintln!("(error \"parse error: {}\")", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("(error \"Error reading file '{}': {}\")", path, e);
            std::process::exit(1);
        }
    }
}

/// Check if content uses HORN logic
fn is_horn_logic(content: &str) -> bool {
    // Fast check: look for "(set-logic HORN)" in the content
    // This is more efficient than parsing the entire file
    if let Ok(commands) = parse(content) {
        commands
            .iter()
            .any(|cmd| matches!(cmd, Command::SetLogic(logic) if logic == "HORN"))
    } else {
        // Fallback: simple string check for unparseable content
        content.contains("(set-logic HORN)")
    }
}

/// Check if content is DIMACS CNF format
///
/// Detection criteria:
/// - First non-comment, non-empty line starts with "p cnf"
/// - Or file starts with comment lines ("c ") followed by "p cnf"
fn is_dimacs_format(content: &str) -> bool {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Comment lines in DIMACS start with 'c'
        if trimmed.starts_with('c') {
            continue;
        }
        // Problem line starts with 'p cnf'
        if trimmed.starts_with("p cnf") {
            return true;
        }
        // If we hit a non-comment, non-empty, non-"p cnf" line, it's not DIMACS
        return false;
    }
    false
}

/// Check if file has .cnf extension
fn has_cnf_extension(path: &str) -> bool {
    path.to_lowercase().ends_with(".cnf")
}

/// Run DIMACS CNF solver on content string
///
/// Output follows SAT competition format:
/// - s SATISFIABLE / s UNSATISFIABLE / s UNKNOWN
/// - v 1 -2 3 ... 0 (satisfying assignment if SAT)
fn run_dimacs_from_content(content: &str) {
    match parse_dimacs(content) {
        Ok(formula) => {
            let mut solver = formula.into_solver();
            match solver.solve() {
                SatSolveResult::Sat(model) => {
                    println!("s SATISFIABLE");
                    // Print model in DIMACS format: v lit1 lit2 ... 0
                    // Variables are 1-indexed in DIMACS
                    print!("v ");
                    for (i, &val) in model.iter().enumerate() {
                        let var = (i + 1) as i32;
                        let lit = if val { var } else { -var };
                        print!("{} ", lit);
                    }
                    println!("0");
                }
                SatSolveResult::Unsat => {
                    println!("s UNSATISFIABLE");
                }
                SatSolveResult::Unknown => {
                    println!("s UNKNOWN");
                }
            }
        }
        Err(e) => {
            eprintln!("c Parse error: {}", e);
            println!("s UNKNOWN");
            std::process::exit(1);
        }
    }
}

/// Run CHC solver on content string
fn run_chc_from_content(content: &str, verbose: bool) {
    use z4_chc::ChcParser;

    let problem = match ChcParser::parse(content) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("(error \"Parse error: {}\")", e);
            std::process::exit(1);
        }
    };

    // Try K-Induction first for linear CHC problems (single predicate)
    // K-Induction is complete for bounded loops and often faster than PDR
    // Use conservative settings to avoid stealing too much time from PDR:
    // - max_k=20: most k-inductive invariants are found early
    // - query_timeout=200ms: fail fast on hard SMT queries
    // - total_timeout=3s: give PDR at least 80% of the time budget
    let kind_config = KindConfig {
        max_k: 20,
        verbose,
        query_timeout: std::time::Duration::from_millis(200),
        total_timeout: std::time::Duration::from_secs(3),
    };
    let kind_solver = KindSolver::new(&problem, kind_config);
    match kind_solver.solve() {
        KindResult::Safe { k, .. } => {
            if verbose {
                eprintln!("KIND: Safe with k-induction at k = {}", k);
            }
            if let Some(model) = kind_solver.to_model(&KindResult::Safe {
                k,
                invariant: z4_chc::ChcExpr::Bool(true),
            }) {
                println!("sat");
                if verbose {
                    print!("{}", model.to_spacer_format(&problem));
                }
            } else {
                println!("sat");
            }
            return;
        }
        KindResult::Unsafe { k } => {
            if verbose {
                eprintln!("KIND: Unsafe at k = {}", k);
            }
            println!("unsat");
            return;
        }
        KindResult::NotApplicable => {
            if verbose {
                eprintln!("KIND: Not applicable (multi-predicate), falling back to PDR");
            }
        }
        KindResult::Unknown => {
            if verbose {
                eprintln!("KIND: Unknown, falling back to PDR");
            }
        }
    }

    // Fall back to PDR for multi-predicate problems or when K-Induction doesn't find a solution
    // The solver should be batteries-included with no user-facing knobs, but some
    // techniques are still experimental and can cause regressions on CHC-COMP.
    let config = PdrConfig {
        max_frames: 100,
        max_iterations: 10000,
        max_obligations: 100_000,
        verbose,
        use_range_weakening: false,
        use_farkas_combination: false,
        use_relational_equality: false,
        use_interpolation: false,
        ..Default::default()
    };

    let mut solver = PdrSolver::new(problem.clone(), config);
    match solver.solve() {
        PdrResult::Safe(model) => {
            println!("sat");
            if verbose {
                print!("{}", model.to_spacer_format(&problem));
            }
        }
        PdrResult::Unsafe(cex) => {
            println!("unsat");
            if verbose {
                eprintln!("Counterexample trace ({} steps):", cex.steps.len());
                for (i, step) in cex.steps.iter().enumerate() {
                    eprintln!("  Step {}: {:?}", i, step.predicate);
                    for (var, val) in &step.assignments {
                        eprintln!("    {} = {}", var, val);
                    }
                }
            }
        }
        PdrResult::Unknown => {
            println!("unknown");
        }
    }
}

/// Run CHC solver on a file
fn run_chc(path: &str, verbose: bool) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("(error \"Error reading file '{}': {}\")", path, e);
            std::process::exit(1);
        }
    };
    run_chc_from_content(&content, verbose);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_horn_logic() {
        // HORN logic
        let horn_content = "(set-logic HORN)
(declare-fun Inv (Int) Bool)
(check-sat)";
        assert!(is_horn_logic(horn_content));

        // Non-HORN logic
        let non_horn_content = "(set-logic QF_LIA)
(declare-const x Int)
(check-sat)";
        assert!(!is_horn_logic(non_horn_content));

        // No logic specified
        let no_logic = "(declare-const x Int)
(check-sat)";
        assert!(!is_horn_logic(no_logic));
    }

    #[test]
    fn test_is_dimacs_format() {
        // Valid DIMACS
        let dimacs = "p cnf 3 2
1 -2 0
-1 2 0";
        assert!(is_dimacs_format(dimacs));

        // DIMACS with comments
        let dimacs_comments = "c A comment
c Another comment
p cnf 3 2
1 -2 0
-1 2 0";
        assert!(is_dimacs_format(dimacs_comments));

        // SMT-LIB is not DIMACS
        let smtlib = "(set-logic QF_LIA)
(declare-const x Int)
(check-sat)";
        assert!(!is_dimacs_format(smtlib));

        // Empty is not DIMACS
        assert!(!is_dimacs_format(""));

        // Comments only is not DIMACS
        assert!(!is_dimacs_format("c comment"));
    }

    #[test]
    fn test_has_cnf_extension() {
        assert!(has_cnf_extension("test.cnf"));
        assert!(has_cnf_extension("test.CNF"));
        assert!(has_cnf_extension("/path/to/file.cnf"));
        assert!(!has_cnf_extension("test.smt2"));
        assert!(!has_cnf_extension("test.cnf.bak"));
    }
}
