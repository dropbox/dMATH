# rustfmt - The Rust Code Formatter

rustfmt is a tool for formatting Rust code according to style guidelines. It ensures consistent code formatting across your project and team.

## Installation

rustfmt is included with Rust via rustup:

```bash
rustup component add rustfmt
```

## Basic Usage

```bash
# Format a file
rustfmt src/main.rs

# Format entire project
cargo fmt

# Check formatting without changes
cargo fmt --check

# Format with verbose output
cargo fmt -- --verbose
```

## Configuration

Create `rustfmt.toml` or `.rustfmt.toml` in your project root:

```toml
# Edition (2015, 2018, 2021, 2024)
edition = "2021"

# Maximum line width
max_width = 100

# Tabs vs spaces
hard_tabs = false
tab_spaces = 4

# Import grouping
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true

# Struct/enum formatting
struct_lit_single_line = true
enum_discrim_align_threshold = 20

# Function formatting
fn_params_layout = "Tall"
fn_single_line = false

# Comment formatting
wrap_comments = false
normalize_comments = false

# Control flow
control_brace_style = "AlwaysSameLine"

# Match arms
match_arm_leading_pipes = "Never"
match_block_trailing_comma = true

# Miscellaneous
use_small_heuristics = "Default"
newline_style = "Auto"
```

## Common Options

### Line Width

```toml
max_width = 100          # Maximum line width
use_small_heuristics = "Default"  # Or "Off", "Max"
```

### Imports

```toml
# How to group imports
imports_granularity = "Crate"  # "Preserve", "Crate", "Module", "Item", "One"

# Order of import groups
group_imports = "StdExternalCrate"  # "Preserve", "StdExternalCrate"

# Sort imports alphabetically
reorder_imports = true
```

### Indentation

```toml
hard_tabs = false    # Use spaces
tab_spaces = 4       # Spaces per indent
```

### Function Signatures

```toml
fn_params_layout = "Tall"  # "Tall", "Compressed", "Vertical"
fn_call_width = 60         # Max width for function calls
fn_single_line = false     # Never put function on single line
```

### Struct Literals

```toml
struct_lit_width = 18                # Max width for struct literals
struct_lit_single_line = true        # Allow single-line struct literals
struct_variant_width = 35            # Max width for struct variants
```

## Nightly Features

Some options require nightly rustfmt:

```bash
rustup run nightly cargo fmt
```

```toml
# Requires nightly
unstable_features = true
format_code_in_doc_comments = true
format_macro_matchers = true
format_strings = true
```

## Ignoring Code

### Skip Formatting

```rust
#[rustfmt::skip]
fn messy_function() {
    // This function won't be formatted
}

#[rustfmt::skip::macros(my_macro)]
fn uses_macro() {
    my_macro!(/* preserved formatting */);
}
```

### Skip Files

Create `.rustfmt.toml`:
```toml
ignore = [
    "src/generated/",
    "src/legacy.rs",
]
```

## CI Integration

### GitHub Actions

```yaml
- name: Check formatting
  run: cargo fmt --all -- --check
```

### Pre-commit Hook

```bash
#!/bin/bash
cargo fmt --all -- --check || {
    echo "Please run 'cargo fmt' before committing"
    exit 1
}
```

## Editor Integration

### VS Code

Install the rust-analyzer extension, then in settings:
```json
{
    "editor.formatOnSave": true,
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
}
```

### Neovim

```lua
-- With null-ls or conform.nvim
require("conform").setup({
    formatters_by_ft = {
        rust = { "rustfmt" },
    },
})
```

## Command Line Options

```bash
# Format and show diff
rustfmt --check src/main.rs

# Format in place
rustfmt src/main.rs

# Format from stdin
echo "fn main(){}" | rustfmt

# Specify edition
rustfmt --edition 2021 src/main.rs

# Use specific config
rustfmt --config-path /path/to/rustfmt.toml src/main.rs

# Print config
rustfmt --print-config default .

# Verbose output
rustfmt --verbose src/main.rs
```

## Cargo Integration

```bash
# Format all code in workspace
cargo fmt --all

# Format specific packages
cargo fmt -p my_crate

# Check without modifying
cargo fmt -- --check

# Format with unstable features
cargo +nightly fmt
```

## Best Practices

1. **Use Default Config**: Start with defaults, customize sparingly
2. **Check in CI**: Run `cargo fmt --check` in CI
3. **Format on Save**: Configure editor to format automatically
4. **Team Consistency**: Commit `.rustfmt.toml` to version control
5. **Edition Match**: Set edition in rustfmt.toml to match Cargo.toml

## Documentation

- Official: https://rust-lang.github.io/rustfmt/
- GitHub: https://github.com/rust-lang/rustfmt
- Configuration Options: https://rust-lang.github.io/rustfmt/?version=master&search=
