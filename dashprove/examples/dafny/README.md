# Dafny Examples

Example Dafny files for testing the DashProve Dafny backend.

## Files

- `MinimalPass.dfy` - Dafny file with verifiable code
- `MinimalFail.dfy` - Dafny file with verification failures

## Running Manually

```bash
# Verify a file
dafny verify MinimalPass.dfy

# Run with detailed output
dafny verify --verbose MinimalFail.dfy
```

## Requirements

Install Dafny from: https://github.com/dafny-lang/dafny/releases

Or via dotnet:
```bash
dotnet tool install --global dafny
```
