# .olean Test Fixtures (Lean 4.13.0)

These fixtures are compiled .olean files for testing lean5-olean parsing and import functionality without requiring a system Lean installation.

## Directory Structure

```
v4.13.0/
├── custom/         # Custom test modules
│   ├── Minimal.lean/.olean     - Basic defs and theorems
│   ├── Inductive.lean/.olean   - Inductive type definitions
│   └── Structure.lean/.olean   - Structure definitions
└── stdlib/         # Subset of Lean 4.13.0 standard library
    ├── Init.olean              - Root Init module
    └── Init/
        ├── Char.olean          - Character handling
        └── Option.olean        - Option type
```

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| custom/Minimal.olean | 19K | Test basic parsing |
| custom/Inductive.olean | 41K | Test inductive type parsing |
| custom/Structure.olean | 50K | Test structure parsing |
| stdlib/Init.olean | 18K | Test import resolution |
| stdlib/Init/Char.olean | 15K | Test submodule imports |
| stdlib/Init/Option.olean | 15K | Test submodule imports |

Total: ~158K

## Regenerating Fixtures

Custom fixtures can be regenerated with:

```bash
cd tests/fixtures/olean/v4.13.0/custom
lean Minimal.lean -o Minimal.olean
lean Inductive.lean -o Inductive.olean
lean Structure.lean -o Structure.olean
```

Stdlib fixtures come from the Lean 4.13.0 installation:
```bash
cp ~/.elan/toolchains/leanprover--lean4---v4.13.0/lib/lean/Init.olean stdlib/
```

## Version Compatibility

These fixtures were compiled with Lean 4.13.0 (commit 6d22e0e5cc5a). The .olean format version is 1.

To check compatibility:
```bash
xxd -l 56 Minimal.olean  # Shows header including version
```
