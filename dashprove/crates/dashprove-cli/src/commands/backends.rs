//! Backends command implementation

use dashprove::backends::{
    AlloyBackend, CoqBackend, DafnyBackend, HealthStatus, IsabelleBackend, KaniBackend,
    Lean4Backend, TlaPlusBackend, VerificationBackend,
};

/// Check and display available backends
pub async fn run_backends() {
    println!("Checking available verification backends...\n");

    let backends: Vec<(&str, Box<dyn VerificationBackend>)> = vec![
        ("LEAN 4", Box::new(Lean4Backend::new())),
        ("TLA+ (TLC)", Box::new(TlaPlusBackend::new())),
        ("Kani", Box::new(KaniBackend::new())),
        ("Alloy", Box::new(AlloyBackend::new())),
        ("Isabelle", Box::new(IsabelleBackend::new())),
        ("Coq", Box::new(CoqBackend::new())),
        ("Dafny", Box::new(DafnyBackend::new())),
    ];

    let total = backends.len();
    let mut available = 0;
    for (name, backend) in backends {
        let health = backend.health_check().await;
        let (status, detail) = match health {
            HealthStatus::Healthy => {
                available += 1;
                ("Available", String::new())
            }
            HealthStatus::Degraded { reason } => {
                available += 1;
                ("Degraded", format!(" - {}", reason))
            }
            HealthStatus::Unavailable { reason } => ("Unavailable", format!(" - {}", reason)),
        };
        println!("{:<12} {}{}", name, status, detail);
    }

    println!("\n{} of {} backends available", available, total);
}
