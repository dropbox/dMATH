# Coq Examples

Example Coq proof files for testing the DashProve Coq backend.

## Files

- `MinimalPass.v` - Vernacular file with provable theorems
- `MinimalFail.v` - Vernacular file with unprovable theorems

## Running Manually

```bash
# Compile a file
coqc MinimalPass.v

# Or run interactively
coqtop -load-vernac-source MinimalPass.v
```

## Requirements

Install Coq via opam:
```bash
opam install coq
```

Or from: https://coq.inria.fr/download
