# LEAN 4 Examples

This directory contains minimal LEAN 4 files demonstrating theorem proving behavior.

## Prerequisites

- **elan**: LEAN 4 toolchain manager
- **lake**: LEAN 4 build system (comes with elan)

### macOS/Linux Installation

```bash
# Install elan (includes lean and lake)
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.13.0

# Activate environment
source ~/.elan/env

# Verify installation
lean --version
lake --version
```

## Files

| File | Description |
|------|-------------|
| `lakefile.toml` | Lake build configuration |
| `lean-toolchain` | Specifies LEAN 4 version (v4.13.0) |
| `MinimalLean.lean` | Theorems that successfully prove |
| `MinimalFail.lean` | Theorems that fail to prove (type errors) |
| `MinimalError.lean` | Syntax/parsing errors |
| `MinimalSorry.lean` | Incomplete proofs with `sorry` placeholders |
| `OUTPUT_pass.txt` | Real output - successful build |
| `OUTPUT_fail.txt` | Real output - proof failures |
| `OUTPUT_error.txt` | Real output - parse errors |
| `OUTPUT_sorry.txt` | Real output - sorry warnings |

## Building

### Build entire project
```bash
cd examples/lean4
lake build
```

### Compile single file
```bash
lake env lean MinimalLean.lean
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (all theorems proven, or only warnings) |
| 1 | Error (proof failures, syntax errors, type errors) |

## Output Patterns

### Successful Build
```
Build completed successfully.
```

### Warning (sorry placeholders)
```
MinimalSorry.lean:7:8: warning: declaration uses 'sorry'
```
**Key pattern**: `warning: declaration uses 'sorry'`

### Type Mismatch Error
```
MinimalFail.lean:14:2: error: tactic 'rfl' failed, the left-hand side
  2 + 2
is not definitionally equal to the right-hand side
  5
⊢ 2 + 2 = 5
```
**Key patterns**:
- `error: tactic 'rfl' failed`
- `is not definitionally equal to`

### Unsolved Goals
```
⊢ <goal>
```
The `⊢` symbol indicates proof obligations that remain unsolved.

### Syntax Error
```
MinimalError.lean:7:5: error: unexpected token 'theorem'; expected ':=', 'where' or '|'
```
**Key pattern**: `error: unexpected token`

### Unknown Identifier
```
error: unknown identifier 'foo'
```

## LEAN 4 Theorem Syntax

```lean
-- Basic theorem
theorem name : type := proof

-- Theorem with tactic proof
theorem name : type := by
  tactic1
  tactic2

-- Theorem with placeholder
theorem name : type := sorry

-- Universal quantification
theorem name : ∀ (x : Type), P x := by
  intro x
  ...

-- Implication
theorem name : A → B := by
  intro ha
  ...
```

## Common Tactics

| Tactic | Description |
|--------|-------------|
| `intro` | Introduce hypothesis into context |
| `exact` | Provide exact proof term |
| `rfl` | Reflexivity (definitional equality) |
| `simp` | Simplify using lemmas |
| `constructor` | Split conjunction into two goals |
| `cases` | Case analysis |
| `induction` | Induction on natural numbers etc |
| `sorry` | Admit proof (incomplete) |
| `decide` | Decidable propositions |

## Project Structure

```
examples/lean4/
├── lakefile.toml      # Build configuration
├── lean-toolchain     # Toolchain version
├── MinimalLean.lean   # Main library source
└── .lake/             # Build artifacts (generated)
```

## Key Discoveries

1. **Exit code 0 with warnings**: `sorry` produces warnings but exit code 0
2. **Exit code 1 for errors**: Type mismatches and syntax errors return exit code 1
3. **Error format**: `file:line:col: error: <message>`
4. **Warning format**: `file:line:col: warning: <message>`
5. **Goal display**: Unsolved goals shown with `⊢` symbol
6. **Lake vs lean**: `lake build` builds project; `lake env lean file.lean` compiles single file
