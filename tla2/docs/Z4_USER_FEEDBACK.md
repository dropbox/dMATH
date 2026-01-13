# TLA+ User Feedback: Z4 SMT Solver Project

**Date:** 2025-12-30
**Project:** Z4 - High-Performance SMT Solver in Rust
**Repo:** https://github.com/dropbox/z4

---

## Context: How We Use TLA+

Z4 is a production SMT (Satisfiability Modulo Theories) solver targeting feature parity with Z3. We use TLA+ to formally verify our core algorithms before and during implementation.

### Current Use Case: CDCL SAT Solver Verification

We wrote a TLA+ specification for the CDCL (Conflict-Driven Clause Learning) algorithm - the core of modern SAT solvers. The spec models:

- **State variables:** `assignment`, `trail`, `state`, `decisionLevel`
- **Actions:** `Propagate`, `DetectConflict`, `Decide`, `Backtrack`, `DeclareSat`, `DeclareUnsat`
- **Invariants:** `TypeInvariant`, `Soundness`, `SatCorrect`

We run TLC to verify the algorithm correctly handles SAT and UNSAT formulas before implementing in Rust.

### Example Spec Structure

```tla
---------------------------- MODULE cdcl_test ----------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS v1, v2
Variables == {v1, v2}

\* A literal is <<var, sign>> where sign is "pos" or "neg"
PosLit(v) == <<v, "pos">>
NegLit(v) == <<v, "neg">>

\* The UNSAT formula: (v1 OR v2) AND (NOT v1) AND (NOT v2)
Clauses == {
    {PosLit(v1), PosLit(v2)},   \* v1 OR v2
    {NegLit(v1)},               \* NOT v1
    {NegLit(v2)}                \* NOT v2
}

VARIABLES assignment, trail, state, decisionLevel

\* ... actions and invariants ...
=============================================================================
```

---

## Feature Requests

### 1. Nested Tuple/Record Syntax in Config Files (HIGH PRIORITY)

**Problem:** TLC's config parser cannot handle nested tuple syntax in CONSTANTS.

**What I wanted to write:**

```tla
\* cdcl.tla - Parameterized spec
CONSTANTS Variables, Clauses
```

```
\* cdcl.cfg
CONSTANTS
    Variables = {v1, v2}
    Clauses = { {v1, v2}, {<<v1, "neg">>}, {<<v2, "neg">>} }
```

**What happens:** Parser error - "expecting }, but did not find it"

**Workaround:** Hardcode clauses directly in the .tla file, losing parameterization:

```tla
Clauses == {
    {PosLit(v1), PosLit(v2)},
    {NegLit(v1)},
    {NegLit(v2)}
}
```

**Impact:** Cannot easily test different formulas without editing the spec. For SAT/SMT solver verification, we want to test many different formulas against the same algorithm spec.

**Request:** Support full TLA+ expression syntax in .cfg CONSTANTS, including:
- Tuples: `<<a, b, c>>`
- Records: `[field1 |-> val1, field2 |-> val2]`
- Nested structures: `{<<v1, "neg">>}`, `[edge |-> <<n1, n2>>, weight |-> 5]`

**Use cases this enables:**
- SAT formulas (sets of clauses, where clauses contain signed literals)
- Graphs with labeled/weighted edges
- State machines with structured transitions
- Protocol messages with fields
- Any domain using compound values

---

### 2. Terminal States Declaration (MEDIUM PRIORITY)

**Problem:** TLC reports "Error: Deadlock reached" for legitimate terminal states.

In our CDCL spec, `SAT` and `UNSAT` are correct terminal states - the algorithm should stop there. But TLC treats any state with no enabled actions as a deadlock error.

**Current output:**
```
Error: Deadlock reached.
State 5: <DeclareUnsat>
/\ state = "UNSAT"
...
```

**Current workarounds:**

1. `ACTION_CONSTRAINT state # "SAT" /\ state # "UNSAT"` - But this prevents exploring terminal states
2. Add self-loops: `Stutter == state \in {"SAT", "UNSAT"} /\ UNCHANGED vars` - Clutters the spec

**Request:** A `TERMINAL_STATES` config option:

```
TERMINAL_STATES
    state = "SAT"
    state = "UNSAT"
```

Or a TLA+ operator:
```tla
Terminal == state \in {"SAT", "UNSAT"}
```

With config:
```
TERMINAL Terminal
```

TLC would report these as successful termination, not deadlock.

---

### 3. Structured Error Messages for AI/Tooling (HIGH PRIORITY)

**Problem:** TLC error messages are prose, requiring regex parsing. AI agents and CI/CD pipelines need structured, actionable errors.

**Current error:**
```
Error in configuration file at line 6.
expecting }, but did not find it.
```

**Ideal error (JSON):**
```json
{
  "error_code": "CFG_PARSE_UNSUPPORTED_SYNTAX",
  "severity": "error",
  "location": {
    "file": "cdcl.cfg",
    "line": 6,
    "column": 23,
    "context": "Clauses = { {<<v1, \"neg\">>} }"
  },
  "message": "Nested tuple syntax not supported in config files",
  "expected": "flat value or model value reference",
  "found": "<<v1, \"neg\">>",
  "suggestion": {
    "action": "define_in_spec",
    "example": "In .tla file: MyClause == <<v1, \"neg\">>\nIn .cfg file: Clauses = { {MyClause} }"
  },
  "docs": "https://tla.dev/docs/cfg-syntax#constants"
}
```

**Type mismatch example:**
```json
{
  "error_code": "TLC_TYPE_MISMATCH",
  "severity": "error",
  "location": {"file": "cdcl.tla", "line": 47, "column": 8},
  "message": "Cannot compare values of different types",
  "lhs": {"value": "TRUE", "type": "boolean"},
  "rhs": {"value": "\"UNDEF\"", "type": "string"},
  "operator": "=",
  "suggestion": {
    "action": "use_consistent_types",
    "options": [
      "Change TRUE to \"TRUE\" (string)",
      "Change \"UNDEF\" to a boolean sentinel"
    ]
  }
}
```

**Deadlock example:**
```json
{
  "error_code": "TLC_DEADLOCK",
  "severity": "warning",
  "location": {"state_num": 5, "action": "DeclareUnsat", "line": 117},
  "state": {
    "state": "UNSAT",
    "assignment": {"v1": "FALSE", "v2": "FALSE"},
    "decisionLevel": 0
  },
  "message": "No enabled actions from this state",
  "suggestion": {
    "action": "if_intentional",
    "options": [
      "Add TERMINAL_STATES state = \"UNSAT\" to config",
      "Add self-loop: Stutter == state = \"UNSAT\" /\\ UNCHANGED vars"
    ]
  }
}
```

**Key properties for AI/tooling consumption:**

| Property | Why |
|----------|-----|
| `error_code` | Categorical - can match against known fixes |
| `location.line/column` | Exact position for automated edits |
| `found` / `expected` | Diff-style - shows exactly what's wrong |
| `suggestion.action` | Imperative - tells what to do |
| `suggestion.example` | Copy-paste ready fix |
| `docs` | Link to relevant documentation |

**Request:** Add output format flag:
```bash
java -jar tla2tools.jar --output-format=json spec.tla
```

This enables:
- AI agents to parse and fix errors programmatically
- CI/CD integration with structured test results
- IDE plugins to show inline diagnostics
- Automated spec repair/suggestion tools

---

## Interface Wishlist

### 1. Programmatic TLC API

For our verification pipeline, we want to:
- Run TLC from Rust code
- Parse results programmatically
- Integrate with CI/CD

Currently we shell out to `java -jar tla2tools.jar` and parse stdout. A proper API (even JSON output) would help.

### 2. Incremental Model Checking

When iterating on a spec, re-checking the entire state space is slow. Would be useful to:
- Cache explored states between runs
- Only re-explore states affected by spec changes

### 3. Counter-Example Export Formats

TLC's trace output is text. Would be useful to export as:
- JSON (for programmatic analysis)
- GraphViz DOT (for visualization)
- SMT-LIB (for our domain - feed back into solver testing)

---

## What's Working Well

- TLC is fast for small state spaces
- The trace output clearly shows the path to errors
- Standard modules (Integers, Sequences, FiniteSets) cover common needs
- Model values work well for creating distinct constants

---

## Our Verification Stack

For context, here's how TLA+ fits into our overall verification approach for Z4:

| Layer | Tool | What It Verifies |
|-------|------|------------------|
| Algorithm Design | **TLA+/TLC** | CDCL state machine correctness |
| Implementation | Kani (CBMC) | Rust code bounded model checking |
| Runtime | DRAT proofs | UNSAT results externally verified |
| Testing | Differential | Agreement with MiniSat reference |

TLA+ is our first line of defense - verify the algorithm is correct before writing any Rust code.

---

## Contact

This feedback comes from the Z4 SMT solver project. Happy to provide more details or test proposed features.
