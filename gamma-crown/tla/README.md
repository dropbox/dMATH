# TLA Specs (γ-CROWN v2 Protocol Invariants)

This folder contains TLA+ specs intended to model-check **protocol-level correctness** of γ-CROWN v2’s DPLL(T) loop (scoping, caching, concurrency), not neural network math.

## Files

- `DPLLT_GammaCrown.tla`: Abstract DPLL(T) state machine + invariants for:
  - decision-level vs theory `push/pop` alignment
  - guarded lemma scoping (no leaks across backtracking)
  - “certified mode” gates (no decisive inference without justification)
  - no stale async propagation results (e.g., GPU batches)

## How to use (today)

This spec is a **starting point** for `tla2` (next-gen model checker) and for shaping invariants and event schemas in `tCore`.

## Running with TLC

A configuration file is provided for small bounded model checking:

```bash
# Install TLC (requires Java)
# Option 1: Download from https://github.com/tlaplus/tlaplus/releases
# Option 2: Use TLA+ Toolbox IDE

# Run TLC from command line (example):
java -jar tla2tools.jar -config DPLLT_GammaCrown.cfg DPLLT_GammaCrown.tla
```

The config file (`DPLLT_GammaCrown.cfg`) defines:
- 4 literals with involutive negation (l1↔l2, l3↔l4)
- MaxLevel=3 (small search depth)
- Certified modes: {"heuristic", "replay", "certified"}
- Numeric modes: {"f32", "real+envelope"}

## Current invariants

1. **Inv_ContextAligned**: `theoryDepth = level` (push/pop discipline)
2. **Inv_TrailConsistent**: No literal and its negation both assigned
3. **Inv_CertifiedRequiresJustifications**: In certified mode, all theory/bounds implications must carry justification tokens
4. **Inv_LemmaScoping**: Active lemmas must not exceed current decision level

## Next steps (recommended)

1. ~~Add `DPLLT_GammaCrown.cfg`~~ ✅ Done
2. Refine the "bounds conflict" path to require a subsequent theory validation before learning in certified mode (currently modeled conservatively: backtrack without learning).
3. Add explicit variables for:
   - active lemma set in the theory context,
   - cache dependency signature details,
   - worker queue / in-flight job results.

