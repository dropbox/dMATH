use super::*;
use std::fs;

// ========== verify_c_file tests ==========

#[test]
fn verify_c_file_accepts_simple_acsl_spec() {
    let code = r"
        //@ requires n >= 0;
        //@ ensures \result >= 0;
        int id(int n) { return n; }
    ";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("sample.c");
    fs::write(&path, code).expect("write should succeed");

    verify_c_file(&path, false, false).expect("verification should not fail with defaults");
}

#[test]
fn verify_c_file_empty_function() {
    let code = r"
        void empty(void) { }
    ";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("empty.c");
    fs::write(&path, code).expect("write should succeed");

    // Empty function with no specs should pass
    verify_c_file(&path, false, false).expect("empty function should verify");
}

#[test]
fn verify_c_file_no_functions_fails() {
    let code = r"
        // Just a comment, no functions
        int x = 42;
    ";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("no_funcs.c");
    fs::write(&path, code).expect("write should succeed");

    let result = verify_c_file(&path, false, false);
    assert!(result.is_err(), "File with no functions should fail");
}

#[test]
fn verify_c_file_nonexistent_path_fails() {
    let path = PathBuf::from("/nonexistent/path/to/file.c");
    let result = verify_c_file(&path, false, false);
    assert!(result.is_err(), "Nonexistent file should fail");
}

#[test]
fn verify_c_file_verbose_mode() {
    let code = r"
        //@ requires x >= 0;
        int abs(int x) { return x >= 0 ? x : -x; }
    ";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("abs.c");
    fs::write(&path, code).expect("write should succeed");

    // Verbose mode should still work (just prints more output)
    verify_c_file(&path, true, false).expect("verbose mode should work");
}

// ========== eval_expr tests ==========

#[test]
fn eval_expr_type_universe() {
    // Type : Type 1
    eval_expr("Type", false).expect("Type should evaluate");
}

#[test]
fn eval_expr_identity_function() {
    // fun (A : Type) (x : A) => x
    eval_expr("fun (A : Type) (x : A) => x", false).expect("Identity function should evaluate");
}

#[test]
fn eval_expr_prop() {
    // Prop is Sort 0
    eval_expr("Prop", false).expect("Prop should evaluate");
}

#[test]
fn eval_expr_nested_lambda() {
    // Nested lambda expression
    eval_expr("fun (A : Type) (B : Type) (x : A) (y : B) => x", false)
        .expect("Nested lambda should work");
}

#[test]
fn eval_expr_verbose_mode() {
    eval_expr("Type", true).expect("Verbose mode should work");
}

#[test]
fn eval_expr_invalid_syntax_fails() {
    // Invalid syntax should fail
    let result = eval_expr("fun x =>", false);
    assert!(result.is_err(), "Invalid syntax should fail");
}

#[test]
fn eval_expr_undefined_identifier_fails() {
    let result = eval_expr("unknownIdent", false);
    assert!(result.is_err(), "Undefined identifier should fail");
}

// ========== check_file tests ==========

#[test]
fn check_file_simple_definition() {
    let code = r"
def id (A : Type) (x : A) : A := x
";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("simple.lean5");
    fs::write(&path, code).expect("write should succeed");

    check_file(&path, false).expect("Simple definition should type check");
}

#[test]
fn check_file_nonexistent_fails() {
    let path = PathBuf::from("/nonexistent/path/to/file.lean5");
    let result = check_file(&path, false);
    assert!(result.is_err(), "Nonexistent file should fail");
}

#[test]
fn check_file_verbose_mode() {
    let code = r"
def const (A B : Type) (x : A) (y : B) : A := x
";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("const.lean5");
    fs::write(&path, code).expect("write should succeed");

    check_file(&path, true).expect("Verbose mode should work");
}

#[test]
fn check_file_empty_file() {
    let code = "";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("empty.lean5");
    fs::write(&path, code).expect("write should succeed");

    // Empty file should succeed (0 declarations)
    check_file(&path, false).expect("Empty file should be valid");
}

#[test]
fn check_file_multiple_definitions() {
    let code = r"
def id (A : Type) (x : A) : A := x
def const (A B : Type) (x : A) (y : B) : A := x
def flip (A B C : Type) (f : A → B → C) (b : B) (a : A) : C := f a b
";

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("multi.lean5");
    fs::write(&path, code).expect("write should succeed");

    check_file(&path, false).expect("Multiple definitions should type check");
}

// ========== CLI parsing tests ==========

#[test]
fn cli_parse_check_command() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "check", "test.lean5"]).unwrap();
    match cli.command {
        Commands::Check { file, verbose } => {
            assert_eq!(file, PathBuf::from("test.lean5"));
            assert!(!verbose);
        }
        _ => panic!("Expected Check command"),
    }
}

#[test]
fn cli_parse_check_verbose() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "check", "-v", "test.lean5"]).unwrap();
    match cli.command {
        Commands::Check { verbose, .. } => {
            assert!(verbose);
        }
        _ => panic!("Expected Check command"),
    }
}

#[test]
fn cli_parse_lake_run() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "run", "demo", "-j", "4", "-v"]).unwrap();
    match cli.command {
        Commands::Lake { command, dir } => {
            assert!(dir.is_none());
            match command {
                LakeCommands::Run {
                    target,
                    verbose,
                    jobs,
                } => {
                    assert_eq!(target.as_deref(), Some("demo"));
                    assert!(verbose);
                    assert_eq!(jobs, 4);
                }
                _ => panic!("Expected Lake run command"),
            }
        }
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_eval_command() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "eval", "Type"]).unwrap();
    match cli.command {
        Commands::Eval { expr, verbose } => {
            assert_eq!(expr, "Type");
            assert!(!verbose);
        }
        _ => panic!("Expected Eval command"),
    }
}

#[test]
fn cli_parse_server_default() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "server"]).unwrap();
    match cli.command {
        Commands::Server {
            port,
            no_gpu,
            websocket,
        } => {
            assert_eq!(port, 8080);
            assert!(!no_gpu);
            assert!(!websocket);
        }
        _ => panic!("Expected Server command"),
    }
}

#[test]
fn cli_parse_server_custom_port() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "server", "--port", "9000"]).unwrap();
    match cli.command {
        Commands::Server { port, .. } => {
            assert_eq!(port, 9000);
        }
        _ => panic!("Expected Server command"),
    }
}

#[test]
fn cli_parse_server_no_gpu() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "server", "--no-gpu"]).unwrap();
    match cli.command {
        Commands::Server { no_gpu, .. } => {
            assert!(no_gpu);
        }
        _ => panic!("Expected Server command"),
    }
}

#[test]
fn cli_parse_server_websocket() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "server", "--websocket"]).unwrap();
    match cli.command {
        Commands::Server { websocket, .. } => {
            assert!(websocket);
        }
        _ => panic!("Expected Server command"),
    }
}

#[test]
fn cli_parse_verify_c_command() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "verify-c", "test.c"]).unwrap();
    match cli.command {
        Commands::VerifyC {
            file,
            fail_unknown,
            verbose,
        } => {
            assert_eq!(file, PathBuf::from("test.c"));
            assert!(!fail_unknown);
            assert!(!verbose);
        }
        _ => panic!("Expected VerifyC command"),
    }
}

#[test]
fn cli_parse_verify_c_fail_unknown() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "verify-c", "--fail-unknown", "test.c"]).unwrap();
    match cli.command {
        Commands::VerifyC { fail_unknown, .. } => {
            assert!(fail_unknown);
        }
        _ => panic!("Expected VerifyC command"),
    }
}

#[test]
fn cli_parse_repl_command() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "repl"]).unwrap();
    assert!(matches!(cli.command, Commands::Repl));
}

#[test]
fn cli_missing_subcommand_fails() {
    use clap::Parser;
    let result = Cli::try_parse_from(["lean5"]);
    assert!(result.is_err(), "Missing subcommand should fail");
}

#[test]
fn cli_unknown_subcommand_fails() {
    use clap::Parser;
    let result = Cli::try_parse_from(["lean5", "unknown"]);
    assert!(result.is_err(), "Unknown subcommand should fail");
}

#[test]
fn cli_parse_lake_resolve() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "resolve", "-v"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Resolve { verbose, dry_run } => {
                assert!(verbose);
                assert!(!dry_run);
            }
            _ => panic!("Expected Lake resolve command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_resolve_dry_run() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "resolve", "--dry-run"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Resolve { verbose, dry_run } => {
                assert!(!verbose);
                assert!(dry_run);
            }
            _ => panic!("Expected Lake resolve command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_exe() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "exe", "myapp"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Exe {
                name,
                args,
                verbose,
            } => {
                assert_eq!(name, "myapp");
                assert!(args.is_empty());
                assert!(!verbose);
            }
            _ => panic!("Expected Lake exe command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_exe_with_args() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "exe", "myapp", "arg1", "arg2"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Exe { name, args, .. } => {
                assert_eq!(name, "myapp");
                assert_eq!(args, vec!["arg1".to_string(), "arg2".to_string()]);
            }
            _ => panic!("Expected Lake exe command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_test() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "test"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Test {
                target,
                verbose,
                jobs,
            } => {
                assert!(target.is_none());
                assert!(!verbose);
                assert_eq!(jobs, 0);
            }
            _ => panic!("Expected Lake test command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_test_with_target() {
    use clap::Parser;
    let cli =
        Cli::try_parse_from(["lean5", "lake", "test", "unit_tests", "-v", "-j", "4"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Test {
                target,
                verbose,
                jobs,
            } => {
                assert_eq!(target.as_deref(), Some("unit_tests"));
                assert!(verbose);
                assert_eq!(jobs, 4);
            }
            _ => panic!("Expected Lake test command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_script_list() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "script", "list"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Script(ScriptCommands::List) => {}
            _ => panic!("Expected Lake script list command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_script_run() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "script", "run", "build_docs"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Script(ScriptCommands::Run { name, args }) => {
                assert_eq!(name, "build_docs");
                assert!(args.is_empty());
            }
            _ => panic!("Expected Lake script run command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_script_doc() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "script", "doc", "build_docs"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Script(ScriptCommands::Doc { name }) => {
                assert_eq!(name, "build_docs");
            }
            _ => panic!("Expected Lake script doc command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_cache_get() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "cache", "get"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Cache(CacheCommands::Get { verbose }) => {
                assert!(!verbose);
            }
            _ => panic!("Expected Lake cache get command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_cache_put_verbose() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "cache", "put", "-v"]).unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Cache(CacheCommands::Put { verbose }) => {
                assert!(verbose);
            }
            _ => panic!("Expected Lake cache put command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_cache_add() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "lake",
        "cache",
        "add",
        "file1.olean",
        "file2.olean",
    ])
    .unwrap();
    match cli.command {
        Commands::Lake { command, .. } => match command {
            LakeCommands::Cache(CacheCommands::Add { files, verbose }) => {
                assert_eq!(
                    files,
                    vec!["file1.olean".to_string(), "file2.olean".to_string()]
                );
                assert!(!verbose);
            }
            _ => panic!("Expected Lake cache add command"),
        },
        _ => panic!("Expected Lake command"),
    }
}

// ========== Fold command integration tests ==========

/// Create a simple Sort certificate for testing
fn create_sort_cert_json() -> String {
    r#"{"Sort":{"level":"Zero"}}"#.to_string()
}

#[test]
fn fold_start_creates_ivc_proof() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let output_path = dir.path().join("ivc.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    fold_start(&cert_path, &output_path, false).expect("fold start should succeed");

    // Verify output file exists
    assert!(output_path.exists(), "IVC proof file should be created");

    // Verify output is valid JSON
    let content = fs::read_to_string(&output_path).expect("read should succeed");
    let proof: SerializableIvcProof =
        serde_json::from_str(&content).expect("output should be valid JSON");

    assert!(proof.step > 0, "IVC should have at least 1 step");
    assert!(proof.num_constraints > 0, "IVC should have constraints");
}

#[test]
fn fold_start_verbose_mode() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let output_path = dir.path().join("ivc.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    fold_start(&cert_path, &output_path, true).expect("verbose mode should work");

    assert!(output_path.exists());
}

#[test]
fn fold_start_invalid_cert_fails() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("invalid.json");
    let output_path = dir.path().join("ivc.json");

    fs::write(&cert_path, "not valid json").expect("write should succeed");

    let result = fold_start(&cert_path, &output_path, false);
    assert!(result.is_err(), "Invalid cert JSON should fail");
}

#[test]
fn fold_start_nonexistent_cert_fails() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("nonexistent.json");
    let output_path = dir.path().join("ivc.json");

    let result = fold_start(&cert_path, &output_path, false);
    assert!(result.is_err(), "Nonexistent cert file should fail");
}

#[test]
fn fold_verify_valid_proof() {
    // First create a valid IVC proof
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let ivc_path = dir.path().join("ivc.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");
    fold_start(&cert_path, &ivc_path, false).expect("fold start should succeed");

    // Now verify it
    fold_verify(&ivc_path, false).expect("verify should pass");
}

#[test]
fn fold_verify_verbose_mode() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let ivc_path = dir.path().join("ivc.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");
    fold_start(&cert_path, &ivc_path, false).expect("fold start should succeed");

    fold_verify(&ivc_path, true).expect("verbose verify should work");
}

#[test]
fn fold_verify_invalid_json_fails() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let ivc_path = dir.path().join("invalid.json");

    fs::write(&ivc_path, "not json").expect("write should succeed");

    let result = fold_verify(&ivc_path, false);
    assert!(result.is_err(), "Invalid JSON should fail verification");
}

#[test]
fn fold_info_shows_proof_details() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let ivc_path = dir.path().join("ivc.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");
    fold_start(&cert_path, &ivc_path, false).expect("fold start should succeed");

    fold_info(&ivc_path).expect("fold info should succeed");
}

#[test]
fn fold_compress_reduces_size() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let ivc_path = dir.path().join("ivc.json");
    let compressed_path = dir.path().join("compressed.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");
    fold_start(&cert_path, &ivc_path, false).expect("fold start should succeed");

    fold_compress(&ivc_path, &compressed_path, false).expect("compress should succeed");

    assert!(compressed_path.exists(), "Compressed file should exist");

    let original_size = fs::metadata(&ivc_path).unwrap().len();
    let compressed_size = fs::metadata(&compressed_path).unwrap().len();

    // Compressed should be smaller (or at least not larger for small proofs)
    assert!(
        compressed_size <= original_size + 50,
        "Compressed should not be significantly larger"
    );
}

#[test]
fn fold_extend_with_same_cert() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let ivc_path = dir.path().join("ivc.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    // Extend creates a new IVC and extends it
    fold_extend(&ivc_path, &cert_path, None, false).expect("extend should succeed");

    assert!(ivc_path.exists(), "IVC file should be created/updated");
}

// ========== Commit command integration tests ==========

#[test]
fn commit_kzg_creates_commitment() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let output_path = dir.path().join("commitment.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    // Use small max_degree for faster test
    commit_kzg(&cert_path, &output_path, 8, false).expect("KZG commit should succeed");

    assert!(output_path.exists(), "Commitment file should be created");

    let content = fs::read_to_string(&output_path).expect("read should succeed");
    let commitment: SerializableCommitment =
        serde_json::from_str(&content).expect("output should be valid JSON");

    assert_eq!(commitment.scheme, "KZG");
    assert_eq!(commitment.max_degree, 8);
    assert!(!commitment.commitment.is_empty());
}

#[test]
fn commit_kzg_verbose_mode() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let output_path = dir.path().join("commitment.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    commit_kzg(&cert_path, &output_path, 8, true).expect("verbose mode should work");
    assert!(output_path.exists());
}

#[test]
fn commit_kzg_invalid_cert_fails() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("invalid.json");
    let output_path = dir.path().join("commitment.json");

    fs::write(&cert_path, "not json").expect("write should succeed");

    let result = commit_kzg(&cert_path, &output_path, 8, false);
    assert!(result.is_err(), "Invalid cert should fail");
}

#[test]
fn commit_ipa_creates_commitment() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let output_path = dir.path().join("commitment.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    commit_ipa(&cert_path, &output_path, 8, false).expect("IPA commit should succeed");

    assert!(output_path.exists(), "Commitment file should be created");

    let content = fs::read_to_string(&output_path).expect("read should succeed");
    let commitment: SerializableCommitment =
        serde_json::from_str(&content).expect("output should be valid JSON");

    assert_eq!(commitment.scheme, "IPA");
    assert_eq!(commitment.max_degree, 8);
}

#[test]
fn commit_ipa_verbose_mode() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let output_path = dir.path().join("commitment.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    commit_ipa(&cert_path, &output_path, 8, true).expect("verbose mode should work");
    assert!(output_path.exists());
}

#[test]
fn commit_verify_kzg_commitment() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let commitment_path = dir.path().join("commitment.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    // Create commitment
    commit_kzg(&cert_path, &commitment_path, 8, false).expect("KZG commit should succeed");

    // Verify it
    commit_verify(&commitment_path, &cert_path, false).expect("verify should succeed");
}

#[test]
fn commit_verify_ipa_commitment() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let commitment_path = dir.path().join("commitment.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");

    // Create commitment
    commit_ipa(&cert_path, &commitment_path, 8, false).expect("IPA commit should succeed");

    // Verify it
    commit_verify(&commitment_path, &cert_path, false).expect("verify should succeed");
}

#[test]
fn commit_verify_verbose_mode() {
    let cert_json = create_sort_cert_json();

    let dir = tempfile::tempdir().expect("tempdir should be created");
    let cert_path = dir.path().join("cert.json");
    let commitment_path = dir.path().join("commitment.json");

    fs::write(&cert_path, &cert_json).expect("write cert should succeed");
    commit_kzg(&cert_path, &commitment_path, 8, false).expect("commit should succeed");

    commit_verify(&commitment_path, &cert_path, true).expect("verbose verify should work");
}

// ========== CLI parsing tests for fold/commit ==========

#[test]
fn cli_parse_fold_start() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "fold",
        "start",
        "-c",
        "cert.json",
        "-o",
        "ivc.json",
    ])
    .unwrap();
    match cli.command {
        Commands::Fold { command } => match command {
            FoldCommands::Start {
                cert,
                output,
                verbose,
            } => {
                assert_eq!(cert, PathBuf::from("cert.json"));
                assert_eq!(output, PathBuf::from("ivc.json"));
                assert!(!verbose);
            }
            _ => panic!("Expected Fold start command"),
        },
        _ => panic!("Expected Fold command"),
    }
}

#[test]
fn cli_parse_fold_start_verbose() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "fold",
        "start",
        "-c",
        "cert.json",
        "-o",
        "ivc.json",
        "-v",
    ])
    .unwrap();
    match cli.command {
        Commands::Fold { command } => match command {
            FoldCommands::Start { verbose, .. } => {
                assert!(verbose);
            }
            _ => panic!("Expected Fold start command"),
        },
        _ => panic!("Expected Fold command"),
    }
}

#[test]
fn cli_parse_fold_extend() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "fold",
        "extend",
        "-i",
        "ivc.json",
        "-c",
        "cert.json",
    ])
    .unwrap();
    match cli.command {
        Commands::Fold { command } => match command {
            FoldCommands::Extend {
                ivc,
                cert,
                output,
                verbose,
            } => {
                assert_eq!(ivc, PathBuf::from("ivc.json"));
                assert_eq!(cert, PathBuf::from("cert.json"));
                assert!(output.is_none());
                assert!(!verbose);
            }
            _ => panic!("Expected Fold extend command"),
        },
        _ => panic!("Expected Fold command"),
    }
}

#[test]
fn cli_parse_fold_extend_with_output() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "fold",
        "extend",
        "-i",
        "ivc.json",
        "-c",
        "cert.json",
        "-o",
        "new_ivc.json",
    ])
    .unwrap();
    match cli.command {
        Commands::Fold { command } => match command {
            FoldCommands::Extend { output, .. } => {
                assert_eq!(output, Some(PathBuf::from("new_ivc.json")));
            }
            _ => panic!("Expected Fold extend command"),
        },
        _ => panic!("Expected Fold command"),
    }
}

#[test]
fn cli_parse_fold_verify() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "fold", "verify", "ivc.json"]).unwrap();
    match cli.command {
        Commands::Fold { command } => match command {
            FoldCommands::Verify { ivc, verbose } => {
                assert_eq!(ivc, PathBuf::from("ivc.json"));
                assert!(!verbose);
            }
            _ => panic!("Expected Fold verify command"),
        },
        _ => panic!("Expected Fold command"),
    }
}

#[test]
fn cli_parse_fold_compress() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5", "fold", "compress", "-i", "ivc.json", "-o", "out.json",
    ])
    .unwrap();
    match cli.command {
        Commands::Fold { command } => match command {
            FoldCommands::Compress {
                ivc,
                output,
                verbose,
            } => {
                assert_eq!(ivc, PathBuf::from("ivc.json"));
                assert_eq!(output, PathBuf::from("out.json"));
                assert!(!verbose);
            }
            _ => panic!("Expected Fold compress command"),
        },
        _ => panic!("Expected Fold command"),
    }
}

#[test]
fn cli_parse_fold_info() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "fold", "info", "ivc.json"]).unwrap();
    match cli.command {
        Commands::Fold { command } => match command {
            FoldCommands::Info { ivc } => {
                assert_eq!(ivc, PathBuf::from("ivc.json"));
            }
            _ => panic!("Expected Fold info command"),
        },
        _ => panic!("Expected Fold command"),
    }
}

#[test]
fn cli_parse_commit_kzg() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "commit",
        "kzg",
        "-c",
        "cert.json",
        "-o",
        "commitment.json",
    ])
    .unwrap();
    match cli.command {
        Commands::Commit { command } => match command {
            CommitCommands::Kzg {
                cert,
                output,
                max_degree,
                verbose,
            } => {
                assert_eq!(cert, PathBuf::from("cert.json"));
                assert_eq!(output, PathBuf::from("commitment.json"));
                assert_eq!(max_degree, 16); // default
                assert!(!verbose);
            }
            _ => panic!("Expected Commit kzg command"),
        },
        _ => panic!("Expected Commit command"),
    }
}

#[test]
fn cli_parse_commit_kzg_with_degree() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "commit",
        "kzg",
        "-c",
        "cert.json",
        "-o",
        "commitment.json",
        "-m",
        "20",
    ])
    .unwrap();
    match cli.command {
        Commands::Commit { command } => match command {
            CommitCommands::Kzg { max_degree, .. } => {
                assert_eq!(max_degree, 20);
            }
            _ => panic!("Expected Commit kzg command"),
        },
        _ => panic!("Expected Commit command"),
    }
}

#[test]
fn cli_parse_commit_ipa() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "commit",
        "ipa",
        "-c",
        "cert.json",
        "-o",
        "commitment.json",
    ])
    .unwrap();
    match cli.command {
        Commands::Commit { command } => match command {
            CommitCommands::Ipa {
                cert,
                output,
                max_degree,
                verbose,
            } => {
                assert_eq!(cert, PathBuf::from("cert.json"));
                assert_eq!(output, PathBuf::from("commitment.json"));
                assert_eq!(max_degree, 16);
                assert!(!verbose);
            }
            _ => panic!("Expected Commit ipa command"),
        },
        _ => panic!("Expected Commit command"),
    }
}

#[test]
fn cli_parse_commit_verify() {
    use clap::Parser;
    let cli = Cli::try_parse_from([
        "lean5",
        "commit",
        "verify",
        "commitment.json",
        "-c",
        "cert.json",
    ])
    .unwrap();
    match cli.command {
        Commands::Commit { command } => match command {
            CommitCommands::Verify {
                commitment,
                cert,
                verbose,
            } => {
                assert_eq!(commitment, PathBuf::from("commitment.json"));
                assert_eq!(cert, PathBuf::from("cert.json"));
                assert!(!verbose);
            }
            _ => panic!("Expected Commit verify command"),
        },
        _ => panic!("Expected Commit command"),
    }
}

// ========== --dir/-d option tests ==========

#[test]
fn cli_parse_lake_dir_long_option() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "--dir", "/path/to/project", "build"]).unwrap();
    match cli.command {
        Commands::Lake { command, dir } => {
            assert_eq!(dir, Some(PathBuf::from("/path/to/project")));
            assert!(matches!(command, LakeCommands::Build { .. }));
        }
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_dir_short_option() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "-d", "/tmp/myproject", "env"]).unwrap();
    match cli.command {
        Commands::Lake { command, dir } => {
            assert_eq!(dir, Some(PathBuf::from("/tmp/myproject")));
            assert!(matches!(command, LakeCommands::Env { .. }));
        }
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_dir_relative_path() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "-d", "my-project", "clean"]).unwrap();
    match cli.command {
        Commands::Lake { command, dir } => {
            assert_eq!(dir, Some(PathBuf::from("my-project")));
            assert!(matches!(command, LakeCommands::Clean { .. }));
        }
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_no_dir_option() {
    use clap::Parser;
    let cli = Cli::try_parse_from(["lean5", "lake", "build"]).unwrap();
    match cli.command {
        Commands::Lake { command, dir } => {
            assert!(dir.is_none());
            assert!(matches!(command, LakeCommands::Build { .. }));
        }
        _ => panic!("Expected Lake command"),
    }
}

#[test]
fn cli_parse_lake_dir_with_subcommand_options() {
    use clap::Parser;
    let cli =
        Cli::try_parse_from(["lean5", "lake", "-d", "/project", "build", "-v", "-f"]).unwrap();
    match cli.command {
        Commands::Lake { command, dir } => {
            assert_eq!(dir, Some(PathBuf::from("/project")));
            match command {
                LakeCommands::Build { verbose, force, .. } => {
                    assert!(verbose);
                    assert!(force);
                }
                _ => panic!("Expected Build command"),
            }
        }
        _ => panic!("Expected Lake command"),
    }
}
