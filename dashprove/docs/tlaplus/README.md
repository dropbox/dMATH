# TLA+ Specifications for DashFlow

**Status:** ⏸️ DEFERRED - Example specs documented for future implementation.

This directory will contain TLA+ specifications for DashFlow's distributed protocols.

## Why TLA+?

DashFlow's complexity is in **distributed/concurrent behavior**, not memory safety:
- Graph execution with parallel branches
- Checkpoint/restore correctness
- Time-travel state reconstruction
- Distributed worker scheduling

TLA+ catches these bugs at **design time**, before implementation.

## Installation

```bash
# TLA+ Toolbox (GUI)
brew install --cask tla-plus-toolbox

# TLC command-line model checker
brew install tlaplus
```

## Specifications

| Spec | Verifies | Status |
|------|----------|--------|
| `GraphExecution.tla` | Node ordering, no deadlock | Draft |
| `CheckpointRestore.tla` | No lost state, idempotent restore | Draft |
| `TimeTravel.tla` | Cursor consistency, monotonic seq | Draft |
| `ParallelExecution.tla` | No race conditions in parallel nodes | TODO |
| `DistributedScheduler.tla` | Worker assignment, fault tolerance | TODO |
| `StateDiff.tla` | Diff/patch invertibility | TODO |
| `EventOrdering.tla` | Out-of-order handling correctness | TODO |

## Draft Specs

The three draft specs (`GraphExecution.tla`, `CheckpointRestore.tla`, `TimeTravel.tla`) are example implementations that demonstrate the TLA+ modeling approach for DashFlow. They verify:

- **GraphExecution**: Nodes execute only after predecessors complete; no orphan execution; eventual termination
- **CheckpointRestore**: Checkpoint state matches reconstructed state; no lost updates; idempotent restore
- **TimeTravel**: Cursor never exceeds high water mark; state reconstruction is deterministic; sequence numbers immutable
