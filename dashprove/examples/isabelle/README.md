# Isabelle Examples

Example Isabelle/HOL theory files for testing the DashProve Isabelle backend.

## Files

- `MinimalPass.thy` - Theory with provable lemmas
- `MinimalFail.thy` - Theory with unprovable lemmas

## Running Manually

```bash
# Build a theory file
isabelle build -D .

# Or run interactively
isabelle jedit MinimalPass.thy
```

## Requirements

Install Isabelle from: https://isabelle.in.tum.de/
