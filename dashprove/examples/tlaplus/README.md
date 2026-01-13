# TLA+ / TLC Examples

This directory contains minimal TLA+ specifications for testing TLC model checker behavior.

## Prerequisites

- **Java**: JDK 17+ (OpenJDK recommended)
- **TLA+ Tools**: `tla2tools.jar` from https://github.com/tlaplus/tlaplus/releases

### macOS Installation

```bash
# Install Java via Homebrew
brew install openjdk

# Set JAVA_HOME
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

# Download TLA+ tools
curl -L -o tla2tools.jar "https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
```

## Files

| File | Description |
|------|-------------|
| `MinimalPass.tla` | Counter 0..3, invariant holds |
| `MinimalPass.cfg` | Config for MinimalPass |
| `MinimalFail.tla` | Counter 0..5, invariant counter <= 3 fails at state 5 |
| `MinimalFail.cfg` | Config for MinimalFail |
| `MinimalError.tla` | Intentional syntax error for error output capture |
| `MinimalError.cfg` | Config for MinimalError |
| `OUTPUT_pass.txt` | Real TLC output - successful verification |
| `OUTPUT_fail.txt` | Real TLC output - invariant violation with counterexample |
| `OUTPUT_error.txt` | Real TLC output - parse error |

## Running TLC

Basic command:
```bash
java -jar tla2tools.jar -config <spec>.cfg <spec>.tla
```

Common options:
```bash
# Use parallel GC for better performance
java -XX:+UseParallelGC -jar tla2tools.jar -config MinimalPass.cfg MinimalPass.tla

# Multiple workers
java -jar tla2tools.jar -workers auto -config MinimalPass.cfg MinimalPass.tla

# No deadlock checking
java -jar tla2tools.jar -deadlock -config MinimalPass.cfg MinimalPass.tla

# Depth limit
java -jar tla2tools.jar -depth 100 -config MinimalPass.cfg MinimalPass.tla
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (no errors, model checking complete) |
| 12 | Safety violation (invariant violated) |
| 13 | Liveness violation (temporal property violated) |
| 150 | Parse/semantic error |

## Output Patterns

### Success
```
Model checking completed. No error has been found.
```

### Invariant Violation
```
Error: Invariant <name> is violated.
Error: The behavior up to this point is:
State 1: <Initial predicate>
<variable> = <value>

State 2: <Next line X, col Y to line X, col Z of module Name>
<variable> = <value>
...
```

### Parse Error
```
***Parse Error***
Was expecting "<expected>"
Encountered "<found>" at line N, column M and token "<token>"
```

### Semantic Error
```
Semantic errors:
*** Errors: N
line X, col Y to line X, col Z of module ModuleName
<error description>
```

## TLA+ Spec Structure

```tla
----------------------------- MODULE Name -----------------------------
(* Comments *)

EXTENDS Naturals, Sequences, FiniteSets  \* Standard library modules

CONSTANT Foo  \* Model parameters
VARIABLE bar  \* State variables

Init == bar = 0  \* Initial state predicate

Next ==          \* Next-state relation
    \/ condition1 /\ bar' = newValue1
    \/ condition2 /\ bar' = newValue2

Spec == Init /\ [][Next]_bar  \* Full specification

TypeOK == bar \in SomeSet     \* Type invariant
Safety == <predicate>          \* Safety invariant

=============================================================================
```

## CFG File Structure

```
SPECIFICATION Spec

\* Constants (model values)
CONSTANT Foo = {a, b, c}

\* Invariants to check
INVARIANT TypeOK
INVARIANT Safety

\* Temporal properties
PROPERTY Liveness
```

## Key Discoveries

1. **EXTENDS required**: Operators like `<`, `+`, `..` require `EXTENDS Naturals`
2. **CFG required**: TLC needs a .cfg file specifying the specification and properties
3. **Exit code 12**: Invariant violations return exit code 12, not 1
4. **Counterexample format**: States numbered, include action location in source
5. **Trace file**: TLC generates `*_TTrace_*.tla` file for trace exploration
