# USL Example Specifications

This directory contains example Unified Specification Language (USL) files demonstrating the various property types supported by DashProve.

## Files

| File | Description | Target Backends |
|------|-------------|-----------------|
| `basic.usl` | Simple theorems and invariants with boolean logic | LEAN 4 |
| `graph.usl` | Graph theory properties with custom types | LEAN 4, Alloy |
| `temporal.usl` | Temporal logic (safety, liveness, fairness) | TLA+ |
| `contracts.usl` | Pre/post conditions for functions | Kani |
| `refinement.usl` | Implementation refinement proofs | LEAN 4 |
| `probabilistic.usl` | Probability bounds | Storm/PRISM |
| `security.usl` | Information flow properties | Tamarin/ProVerif |

## Usage

### Verify a specification
```bash
dashprove verify examples/usl/basic.usl
```

### Export to a specific backend format
```bash
# Export to LEAN 4
dashprove export examples/usl/basic.usl --target lean

# Export to TLA+
dashprove export examples/usl/temporal.usl --target tla+

# Export to Kani
dashprove export examples/usl/contracts.usl --target kani

# Export to Alloy
dashprove export examples/usl/graph.usl --target alloy
```

### Verify with specific backends
```bash
dashprove verify examples/usl/basic.usl --backends lean
dashprove verify examples/usl/temporal.usl --backends tla+
```

## USL Syntax Overview

### Property Types

- **theorem**: Mathematical properties to prove
- **invariant**: Always-true properties
- **temporal**: TLA+-style temporal logic (always, eventually, leads-to)
- **contract**: Pre/post conditions for functions
- **refinement**: Proves implementation refines specification
- **probabilistic**: Probability bounds
- **security**: Information flow properties

### Basic Syntax

```
// Type definitions
type Node = {
    id: String,
    data: Int
}

// Theorem
theorem example {
    forall x: Bool . x or not x
}

// Invariant
invariant positive {
    forall n: Int . n >= 0 implies n * n >= 0
}

// Temporal property
temporal eventually_done {
    always(eventually(done))
}

// Contract
contract divide(x: Int, y: Int) -> Result<Int> {
    requires { y != 0 }
    ensures { result * y == x }
}
```

### Quantifiers

- `forall x: Type . body` - Universal quantification
- `exists x: Type . body` - Existential quantification
- `forall x in collection . body` - Bounded quantification

### Operators

- Boolean: `and`, `or`, `not`, `implies`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Temporal: `always(...)`, `eventually(...)`, `~>` (leads-to)
