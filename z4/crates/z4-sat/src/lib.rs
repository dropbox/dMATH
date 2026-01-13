//! Z4 SAT - CDCL SAT solver core
//!
//! A high-performance Conflict-Driven Clause Learning SAT solver with
//! competitive techniques from CaDiCaL and Kissat.
//!
//! ## Core CDCL Features
//! - 2-watched literal scheme for efficient unit propagation
//! - VSIDS/EVSIDS variable selection heuristics with decay
//! - 1UIP conflict analysis with recursive clause minimization
//! - Luby and glucose-style EMA restarts
//! - LBD-based tier clause management (core/mid/local)
//! - Chronological backtracking with lazy reimplication
//! - Phase saving
//!
//! ## Inprocessing Techniques
//! - Vivification (clause strengthening via propagation)
//! - Bounded variable elimination (BVE)
//! - Blocked clause elimination (BCE)
//! - Subsumption and self-subsumption
//! - Failed literal probing
//! - Hyper-ternary resolution (HTR)
//!
//! ## Advanced SAT Techniques
//! - Gate extraction (AND/XOR/ITE/EQUIV recognition)
//! - SAT sweeping (equivalent literal detection)
//! - Congruence closure (gate-based equivalence detection)
//! - Model reconstruction for equisatisfiable transformations
//!
//! ## Parallel Portfolio Solving
//! - Multiple solver configurations running in parallel
//! - Different restart policies (Luby, Glucose EMA)
//! - Configurable inprocessing strategies
//! - First-result wins, others terminate
//!
//! ## Proof Generation
//! - DRAT proof output (text and binary formats)
//! - LRAT proof output with resolution hints (text and binary formats)
//! - Variable-length binary encoding for compact proofs

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod bce;
pub mod bve;
pub mod clause;
pub mod clause_db;
pub mod conflict;
pub mod congruence;
pub mod dimacs;
pub mod gates;
pub mod htr;
pub mod literal;
pub mod portfolio;
pub mod probe;
pub mod proof;
pub mod reconstruct;
pub mod solver;
pub mod subsume;
pub mod sweep;
pub mod vivify;
pub mod vsids;
pub mod walk;
pub mod warmup;
pub mod watched;

pub use bce::{BCEStats, BCE};
pub use bve::{BVEStats, EliminationResult, BVE};
pub use clause::Clause;
pub use congruence::{CongruenceClosure, CongruenceResult, CongruenceStats};
pub use dimacs::{parse_str as parse_dimacs, DimacsError, DimacsFormula};
pub use gates::{Gate, GateExtractor, GateStats, GateType};
pub use htr::{HTRResult, HTRStats, HTR};
pub use literal::{Literal, Variable};
pub use portfolio::{PortfolioSolver, SolverConfig, Strategy};
pub use probe::{ProbeResult, ProbeStats, Prober};
pub use proof::{DratWriter, LratWriter, ProofOutput};
pub use reconstruct::{ReconstructionStack, ReconstructionStep};
pub use solver::{AssumeResult, MemoryStats, SolveResult, Solver};
pub use subsume::{SubsumeResult, SubsumeStats, Subsumer};
pub use sweep::{SweepOutcome, SweepStats, Sweeper};
pub use vivify::{Vivifier, VivifyResult, VivifyStats};
pub use watched::ClauseRef;
