//! tla-check - TLA+ Model Checker
//!
//! This crate provides:
//! - **Value types**: Runtime representation of TLA+ values (Bool, Int, Set, Function, etc.)
//! - **Expression evaluator**: Evaluate TLA+ expressions to runtime values
//! - **State exploration**: BFS state space exploration
//! - **Invariant checking**: Verify safety properties
//! - **Configuration parsing**: Parse TLC .cfg files
//!
//! # Quick Start
//!
//! ```rust
//! use tla_check::{eval, EvalCtx, Value};
//! use tla_core::{lower, parse_to_syntax_tree, FileId};
//!
//! // Parse and evaluate a TLA+ expression
//! let src = "---- MODULE Test ----\nOp == 1 + 2 * 3\n====";
//! let tree = parse_to_syntax_tree(src);
//! let result = lower(FileId(0), &tree);
//! let module = result.module.unwrap();
//!
//! // Find the operator and evaluate it
//! let mut ctx = EvalCtx::new();
//! ctx.load_module(&module);
//!
//! // The result of 1 + 2 * 3 is 7
//! ```
//!
//! # Model Checking
//!
//! ```ignore
//! use tla_check::{check_module, Config, State};
//!
//! // Parse config file
//! let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety")?;
//!
//! // Run model checker
//! let result = check_module(&module, &config);
//! match result {
//!     CheckResult::Success(stats) => println!("OK: {} states", stats.states_found),
//!     CheckResult::InvariantViolation { invariant, trace, .. } => {
//!         println!("Violated: {}\n{}", invariant, trace);
//!     }
//!     _ => {}
//! }
//! ```

pub mod adaptive;
pub mod arena;
pub mod check;
pub mod checkpoint;
pub mod compiled_guard;
pub mod config;
pub mod constants;
pub mod coverage;
pub mod enumerate;
pub mod error;
pub mod eval;
pub mod fingerprint;
pub mod intern;
pub mod json_output;
pub mod liveness;
pub mod parallel;
pub mod spec_formula;
pub mod state;
pub mod storage;
pub mod trace_file;
pub mod value;
pub mod var_index;

// Kani formal verification harnesses (only compiled with cargo kani)
#[cfg(kani)]
mod kani_harnesses;

// Regular tests that mirror Kani proofs
#[cfg(test)]
mod kani_harnesses;

// Re-exports
pub use adaptive::{check_module_adaptive, AdaptiveChecker, PilotAnalysis, Strategy};
pub use check::{
    check_module, resolve_spec_from_config, resolve_spec_from_config_with_extends, simulate_module,
    CheckError, CheckResult, CheckStats, LimitType, ModelChecker, Progress, ProgressCallback,
    ResolvedSpec, SimulationConfig, SimulationResult, SimulationStats, SuccessorWitnessMap, Trace,
};
pub use config::{Config, ConfigError, ConstantValue};
pub use constants::{bind_constants_from_config, parse_constant_value};
pub use error::{EvalError, EvalResult};
pub use eval::{eval, Env, EvalCtx, OpEnv};
pub use parallel::{check_module_parallel, ParallelChecker};
pub use spec_formula::{extract_spec_formula, FairnessConstraint, SpecFormula};
pub use state::{
    value_fingerprint, value_fingerprint_ahash, value_fingerprint_xxh3, ArrayState, Fingerprint,
    State, UndoEntry,
};
pub use value::{FuncSetValue, FuncValue, IntervalValue, SubsetValue, Value};

// Liveness checking
pub use liveness::{
    eval_live_expr, is_state_consistent, is_transition_consistent, AstToLive, BehaviorGraph,
    BehaviorGraphNode, ConvertError, ExprLevel, LiveExpr, LivenessChecker, LivenessConstraints,
    LivenessResult, LivenessStats, NodeInfo, Particle, Tableau, TableauNode,
};

// Coverage statistics
pub use coverage::{detect_actions, ActionStats, CoverageStats, DetectedAction};

// Scalable storage
pub use storage::{
    CapacityStatus, DiskFingerprintSet, FingerprintSet, FingerprintStorage,
    InMemoryTraceLocations, MmapError, MmapFingerprintSet, MmapTraceLocations,
    ShardedFingerprintSet, TraceLocationStorage, TraceLocationsStorage,
};

// Disk-based trace storage
pub use trace_file::{TraceFile, TraceLocations};

// Variable indexing
pub use var_index::{VarIndex, VarRegistry};

// Value interning for parallel performance
pub use intern::{
    clear_global_interner, get_interner, DiffHandleSuccessor, HandleState, ValueHandle,
    ValueInterner,
};

// Compiled guards
pub use compiled_guard::{compile_guard, CmpOp, CompiledExpr, CompiledGuard};

// Checkpoint/resume support
pub use checkpoint::{
    Checkpoint, CheckpointMetadata, CheckpointStats, SerializableState, SerializableValue,
};

// JSON output format for AI agents
pub use json_output::{
    liveness_trace_to_dot, trace_to_dot, value_to_json, ActionInfo, ActionRef, CounterexampleInfo,
    DiagnosticMessage, DiagnosticsInfo, InputInfo, JsonOutput, JsonValue, JsonlEvent, PrintOutput,
    ResultInfo, SourceLocation, SpecInfo, StateDiff, StateInfo, StatisticsInfo, ValueChange,
    ViolatedProperty, OUTPUT_VERSION,
};

// Memory arena for efficient state storage
pub use arena::{BulkStateHandle, BulkStateStorage, StateArena, ThreadLocalArena};
