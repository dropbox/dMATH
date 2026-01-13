//! Exhaustive formal verification research collection
//!
//! Collects comprehensive data on the entire formal verification landscape:
//! - All theorem provers and proof assistants
//! - All model checkers
//! - All SMT/SAT solvers
//! - AI for theorem proving
//! - Program verification tools
//! - Neural network verification
//! - Security protocol verification
//! - Hardware verification
//! - Benchmarks and datasets

use dashprove_knowledge::{ArxivConfig, ArxivFetcher, GithubConfig, GithubSearcher};
use std::path::PathBuf;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let base_dir = PathBuf::from("data/knowledge");
    std::fs::create_dir_all(&base_dir)?;

    info!("=== EXHAUSTIVE FORMAL VERIFICATION RESEARCH ===");
    info!("Goal: Map the entire formal verification landscape");

    // ============================================================
    // PHASE 1: ArXiv - Comprehensive paper collection
    // ============================================================
    info!("\n=== PHASE 1: ArXiv Research Papers ===");

    let arxiv_queries = vec![
        // Core theorem proving
        (
            "Theorem Provers",
            vec![
                "ti:theorem proving",
                "ti:proof assistant",
                "ti:interactive theorem",
                "ti:automated theorem",
                "ti:formal proof",
            ],
        ),
        // Specific provers
        (
            "Specific Provers",
            vec![
                "ti:lean prover OR ti:lean4",
                "ti:coq proof OR abs:coq",
                "ti:isabelle OR ti:isabelle/hol",
                "ti:agda OR ti:dependent types",
                "ti:HOL4 OR ti:HOL Light",
                "ti:metamath OR ti:mizar",
                "ti:ACL2",
            ],
        ),
        // AI + Theorem Proving (CRITICAL for superhuman coding)
        (
            "AI for Theorem Proving",
            vec![
                "ti:neural theorem proving",
                "ti:language model proof",
                "ti:LLM theorem",
                "ti:autoformalization",
                "ti:machine learning proof",
                "ti:deep learning theorem",
                "ti:transformer proof",
                "ti:reinforcement learning proof",
                "abs:AlphaProof OR abs:AlphaGeometry",
                "abs:LeanDojo OR abs:ReProver",
                "ti:premise selection",
                "ti:tactic prediction",
                "ti:proof synthesis",
            ],
        ),
        // SMT/SAT Solvers
        (
            "SMT/SAT Solvers",
            vec![
                "ti:SMT solver",
                "ti:SAT solver",
                "ti:satisfiability modulo",
                "ti:Z3 solver OR abs:Z3 theorem",
                "ti:CVC5 OR ti:CVC4",
                "ti:DPLL OR ti:CDCL",
                "ti:MaxSAT",
            ],
        ),
        // Model Checking
        (
            "Model Checking",
            vec![
                "ti:model checking",
                "ti:temporal logic verification",
                "ti:CTL verification OR ti:LTL verification",
                "ti:bounded model checking",
                "ti:symbolic model checking",
                "ti:TLA+ OR abs:TLA+",
                "ti:SPIN model checker",
                "ti:NuSMV",
            ],
        ),
        // Program Verification
        (
            "Program Verification",
            vec![
                "ti:program verification",
                "ti:software verification",
                "ti:Hoare logic",
                "ti:separation logic",
                "ti:deductive verification",
                "ti:Dafny OR abs:Dafny",
                "ti:Why3",
                "ti:KeY verifier",
                "ti:F* language OR ti:fstar",
            ],
        ),
        // Rust Verification
        (
            "Rust Verification",
            vec![
                "ti:rust verification",
                "ti:rust safety",
                "ti:rust formal",
                "ti:Kani verifier OR abs:Kani rust",
                "ti:Verus OR abs:verus rust",
                "ti:Prusti OR abs:prusti",
                "ti:Creusot",
                "ti:MIRAI rust",
                "ti:RustBelt",
                "ti:ownership types",
            ],
        ),
        // Neural Network Verification
        (
            "Neural Network Verification",
            vec![
                "ti:neural network verification",
                "ti:deep learning verification",
                "ti:robustness verification neural",
                "ti:certified defense",
                "ti:abstract interpretation neural",
                "ti:Marabou verifier",
                "ti:alpha-beta-CROWN OR abs:CROWN",
                "ti:ERAN verifier",
                "ti:interval bound propagation",
                "ti:lipschitz neural",
            ],
        ),
        // Probabilistic Verification
        (
            "Probabilistic Verification",
            vec![
                "ti:probabilistic model checking",
                "ti:Markov decision process verification",
                "ti:stochastic verification",
                "ti:PRISM model checker",
                "ti:Storm checker",
                "ti:probabilistic program",
            ],
        ),
        // Security Protocol Verification
        (
            "Security Verification",
            vec![
                "ti:security protocol verification",
                "ti:cryptographic protocol verification",
                "ti:Tamarin prover",
                "ti:ProVerif",
                "ti:symbolic security",
                "ti:Dolev-Yao",
                "ti:computational soundness",
            ],
        ),
        // Concurrency Verification
        (
            "Concurrency Verification",
            vec![
                "ti:concurrent program verification",
                "ti:parallel program verification",
                "ti:race condition detection",
                "ti:deadlock verification",
                "ti:linearizability verification",
                "ti:memory model verification",
            ],
        ),
        // Hardware Verification
        (
            "Hardware Verification",
            vec![
                "ti:hardware verification",
                "ti:RTL verification",
                "ti:processor verification",
                "ti:equivalence checking hardware",
                "ti:property checking hardware",
            ],
        ),
        // Abstract Interpretation
        (
            "Abstract Interpretation",
            vec![
                "ti:abstract interpretation",
                "ti:static analysis sound",
                "ti:numerical abstract domain",
                "ti:shape analysis",
                "ti:pointer analysis",
            ],
        ),
        // Symbolic Execution
        (
            "Symbolic Execution",
            vec![
                "ti:symbolic execution",
                "ti:concolic testing",
                "ti:path exploration",
                "ti:KLEE symbolic",
                "ti:angr binary",
            ],
        ),
        // Certified/Verified Compilers
        (
            "Certified Systems",
            vec![
                "ti:CompCert",
                "ti:verified compiler",
                "ti:seL4 OR abs:seL4",
                "ti:CakeML",
                "ti:certified kernel",
                "ti:verified operating system",
            ],
        ),
        // Program Synthesis
        (
            "Program Synthesis",
            vec![
                "ti:program synthesis",
                "ti:inductive synthesis",
                "ti:syntax-guided synthesis",
                "ti:SyGuS",
                "ti:neural program synthesis",
                "ti:LLM code generation verification",
            ],
        ),
        // Verification Benchmarks
        (
            "Benchmarks",
            vec![
                "ti:verification benchmark",
                "ti:SV-COMP",
                "ti:SMT-LIB",
                "ti:TPTP",
                "ti:miniF2F",
                "ti:ProofNet",
            ],
        ),
        // Cutting Edge 2024-2025
        (
            "Cutting Edge",
            vec![
                "ti:large language model verification",
                "ti:code verification LLM",
                "ti:formal specification LLM",
                "ti:proof generation neural",
                "ti:verified code generation",
            ],
        ),
    ];

    let arxiv_dir = base_dir.join("arxiv_exhaustive");
    std::fs::create_dir_all(&arxiv_dir)?;

    let arxiv_config = ArxivConfig {
        categories: vec![],
        start_date: "2023-01-01".to_string(), // Extended to 2023 for more coverage
        max_results_per_category: 100,
    };
    let arxiv = ArxivFetcher::new(arxiv_config, arxiv_dir.clone());

    let mut all_papers = Vec::new();
    let mut total_queries = 0;

    for (category_name, queries) in &arxiv_queries {
        info!("\n--- {} ---", category_name);
        for query in queries {
            total_queries += 1;
            info!("  Query {}: {}", total_queries, query);
            match arxiv.fetch_papers(query, 50).await {
                Ok(papers) => {
                    info!("    -> {} papers", papers.len());
                    all_papers.extend(papers);
                }
                Err(e) => {
                    warn!("    -> Failed: {}", e);
                }
            }
            // ArXiv rate limit
            tokio::time::sleep(std::time::Duration::from_secs(4)).await;
        }
    }

    // Deduplicate
    all_papers.sort_by(|a, b| a.arxiv_id.cmp(&b.arxiv_id));
    all_papers.dedup_by(|a, b| a.arxiv_id == b.arxiv_id);
    info!("\nTotal unique ArXiv papers: {}", all_papers.len());

    if let Err(e) = arxiv.save_papers(&all_papers).await {
        warn!("Failed to save papers: {}", e);
    }

    // ============================================================
    // PHASE 2: GitHub - Comprehensive tool collection
    // ============================================================
    info!("\n=== PHASE 2: GitHub Repositories ===");

    let github_queries = vec![
        // Theorem Provers
        "theorem prover",
        "proof assistant",
        "lean4",
        "lean prover",
        "coq proof",
        "isabelle hol",
        "agda language",
        "idris language",
        // SMT/SAT
        "smt solver",
        "sat solver",
        "z3 prover",
        "cvc5",
        "satisfiability",
        // Model Checking
        "model checker",
        "tla+ spec",
        "alloy analyzer",
        "spin model",
        "nusmv",
        "uppaal",
        // Program Verification
        "program verification",
        "formal verification",
        "dafny verifier",
        "why3 platform",
        "fstar verification",
        "viper verification",
        // Rust Verification
        "rust verification",
        "kani rust",
        "verus rust",
        "prusti rust",
        "creusot rust",
        "miri rust",
        // Neural Network Verification
        "neural network verification",
        "nn verification",
        "robustness verification",
        "marabou verifier",
        "auto_LiRPA",
        "crown verifier",
        // Security
        "protocol verification",
        "tamarin prover",
        "proverif",
        "cryptographic verification",
        // AI + Theorem Proving
        "neural theorem prover",
        "llm proof",
        "autoformalization",
        "leandojo",
        "reprover",
        "gpt proof",
        "mathlib4",
        // Symbolic Execution
        "symbolic execution",
        "klee symbolic",
        "angr binary",
        "manticore symbolic",
        // Static Analysis
        "abstract interpretation",
        "static analyzer sound",
        "infer facebook",
        // Certified Systems
        "compcert compiler",
        "sel4 kernel",
        "cakeml",
        "verified compiler",
        // Benchmarks
        "verification benchmark",
        "sv-comp",
        "smt-lib",
        "tptp benchmark",
        "minif2f",
        // Program Synthesis
        "program synthesis",
        "sygus",
        "neural synthesis",
    ];

    let github_dir = base_dir.join("github_exhaustive");
    std::fs::create_dir_all(&github_dir)?;

    let github_config = GithubConfig {
        api_token: std::env::var("GITHUB_TOKEN").ok(),
        min_stars: 5, // Lower threshold to catch more tools
        queries: vec![],
    };
    let github = GithubSearcher::new(github_config, github_dir.clone());

    let mut all_repos = Vec::new();

    for query in &github_queries {
        info!("GitHub: {}", query);
        match github.search(query).await {
            Ok(repos) => {
                info!("  -> {} repos", repos.len());
                all_repos.extend(repos);
            }
            Err(e) => {
                warn!("  -> Failed: {}", e);
                // If rate limited, wait longer
                if format!("{}", e).contains("rate") {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Deduplicate and sort by stars
    all_repos.sort_by(|a, b| a.full_name.cmp(&b.full_name));
    all_repos.dedup_by(|a, b| a.full_name == b.full_name);
    all_repos.sort_by(|a, b| b.stars.cmp(&a.stars));

    info!("\nTotal unique GitHub repos: {}", all_repos.len());

    if let Err(e) = github.save_repos(&all_repos).await {
        warn!("Failed to save repos: {}", e);
    }

    // ============================================================
    // Summary
    // ============================================================
    info!("\n=== EXHAUSTIVE COLLECTION COMPLETE ===");
    info!("ArXiv papers: {}", all_papers.len());
    info!("GitHub repos: {}", all_repos.len());
    info!("Data location: {:?}", base_dir);

    // Print top repos by stars
    info!("\nTop 20 repositories by stars:");
    for repo in all_repos.iter().take(20) {
        info!(
            "  ⭐ {} - {} ({})",
            repo.stars,
            repo.full_name,
            repo.description
                .as_deref()
                .unwrap_or("")
                .chars()
                .take(50)
                .collect::<String>()
        );
    }

    Ok(())
}
