# dMATH - Formal Verification Tools

![Status](https://img.shields.io/badge/status-preview-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

Formal verification and theorem proving tools from Dropbox.

## Thesis

**Proof is the only scalable code review.** AI generates code faster than humans can review it. But we don't need to understand the code or remember the proof—we just need to verify that a proof exists. These tools make verification practical: SMT solvers check constraints, model checkers explore state spaces, theorem provers verify logic. The result is durable progress without growing context windows or trusting code we haven't read.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| **z4** | Z3 ported to Rust. SAT/SMT queries, constraint solving, proving code correct. | Planned |
| **tla2** | TLA+ 2.0 in Rust. Distributed system specs, temporal logic, model checking. | Planned |
| **gamma-crown** | α,β-CROWN NN verifier. Prove neural network properties (robustness, safety). | Planned |
| **kani_fast** | Kani fork. Bounded model checking of Rust code. | Planned |
| **lean5** | Lean4 in Rust. Theorem proving, type theory, mathematical proofs. | Planned |
| **dashprove** | Orchestrates multiple provers. | Planned |
| **zksolve** | R1CS/ZK constraint solver. Witness generation, proving. | Planned |
| **proverif-rs** | Dolev-Yao protocol verification. Security protocol proofs. | Planned |
| **galg** | Gröbner basis (F4/F5). Polynomial system solving. | Planned |

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Release History

See [RELEASES.md](RELEASES.md) for version history.
