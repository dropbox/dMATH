# Backend Guide

DashProve executes verification by parsing USL, type-checking, and dispatching to backends that implement `VerificationBackend` (`crates/dashprove-backends/src/traits.rs`). This guide documents the current backends and how to add new ones.

## VerificationBackend Trait
- `fn id(&self) -> BackendId` unique identifier.
- `fn supports(&self) -> Vec<PropertyType>` declares compatible property kinds.
- `async fn verify(&self, spec: &TypedSpec) -> Result<BackendResult, BackendError>` performs compilation + execution.
- `async fn health_check(&self) -> HealthStatus` lightweight availability probe.
- `BackendResult` contains `status`, optional `proof`/`counterexample`, `diagnostics`, and `time_taken`.

### Adding a Backend
1) Implement `VerificationBackend` with compilation from `TypedSpec`.
2) Add a config struct for paths/timeouts.
3) Register the backend where appropriate:
   - `dashprove` crate (client) for programmatic use,
   - CLI (`crates/dashprove-cli`) for `--backends` filtering,
   - Dispatcher defaults (`DispatcherConfig`) if it should participate automatically.
4) Provide a `health_check` that avoids heavy work (version check or binary presence).
5) Add tests under `tests/backends/` or integration fixtures.

## Built-In Backends

### LEAN 4
- File: `crates/dashprove-backends/src/lean4.rs`
- Supports: theorems, invariants, refinements.
- Detection: searches for `lake` and `lean` in PATH or common elan locations; optional `Lean4Config` allows custom paths, mathlib toggle, and timeout.
- Execution: writes a temp lake project, runs `lake build`; parses proof/counterexample from output; mathlib disabled by default to avoid network fetch.
- Health: `lake --version`.

### TLA+
- File: `crates/dashprove-backends/src/tlaplus.rs`
- Supports: temporal properties and invariants.
- Detection: uses `tlc` if available or `tla2tools.jar` with `java`; configurable workers, depth limit, timeout via `TlaPlusConfig`.
- Execution: writes `<module>.tla` plus `<module>.cfg`, runs TLC, parses `Model checking completed` vs. error/counterexample lines.
- Health: TLC version check (standalone or jar).

### Kani
- File: `crates/dashprove-backends/src/kani.rs`
- Supports: contracts (pre/post conditions).
- Detection: requires `cargo kani`; config allows custom cargo path, project directory, timeout, and concrete playback toggle.
- Execution: generates a temp harness crate that depends on the target Rust crate, injects `use <crate>::*;`, runs `cargo kani`; captures SAT/UNSAT and counterexample hints from stdout/stderr.
- Health: `cargo kani --version` with timeout.
- Note: workspace manifests are unsupported; `project_dir` should point to a single-package crate.

### Alloy
- File: `crates/dashprove-backends/src/alloy.rs`
- Supports: invariants/theorems as bounded checks.
- Detection: looks for `alloy` CLI; `AlloyConfig` controls path, scope, solver, timeout.
- Execution: compiles to `spec.als`, runs `alloy exec spec.als`; inspects SAT/UNSAT in output because exit code alone is insufficient.
- Health: `alloy --help` output inspection.

### Dispatcher Integration
- The dispatcher selects backends by `PropertyType` and merge strategy; see `crates/dashprove-dispatcher`.
- Default client registers LEAN, TLA+, Alloy; Kani must be registered explicitly because it needs a project path.
- Selection strategies: single, redundant (multi backend), all, or specific; merge strategies include first success, unanimous, majority.
