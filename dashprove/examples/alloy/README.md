# Alloy Examples

This directory contains minimal Alloy models demonstrating relational modeling and SAT-based analysis.

## Prerequisites

- **Java**: JDK 17+ (OpenJDK recommended)
- **Alloy**: Relational modeling tool

### macOS Installation

```bash
# Install via Homebrew
brew install alloy-analyzer

# Set Java path (if needed)
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
```

## Files

| File | Description |
|------|-------------|
| `MinimalPass.als` | State machine model with passing assertion |
| `MinimalFail.als` | Graph model with failing assertion (finds cycle) |
| `MinimalError.als` | Model with intentional syntax error |
| `OUTPUT_pass.txt` | Real Alloy output - assertion holds |
| `OUTPUT_fail.txt` | Real Alloy output - counterexample found |
| `OUTPUT_error.txt` | Real Alloy output - parse error |

## Running Alloy

### Execute all commands in model
```bash
alloy exec model.als
```

### Execute specific command
```bash
alloy exec -c 'check AssertionName' model.als
alloy exec -c 1 model.als  # By index
```

### Output options
```bash
# Output to console
alloy exec -o - model.als

# JSON format
alloy exec -t json -o - model.als

# Text format
alloy exec -t text -o - model.als
```

### Key Options
```bash
-c, --command <name>  # Run specific command
-o, --output <dir>    # Output directory (- for stdout)
-t, --type <format>   # Output type: none, text, table, json, xml
-f, --force           # Overwrite output directory
-r, --repeat <n>      # Find multiple solutions (0 = all)
-s, --solver <name>   # SAT solver (default: SAT4J)
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (commands executed, may include counterexamples) |
| 1 | Error (syntax error, type error, etc.) |

## Output Patterns

### Check Command Results

| Result | Meaning |
|--------|---------|
| `UNSAT` | No counterexample found = Assertion HOLDS |
| `SAT` | Counterexample found = Assertion VIOLATED |

### Run Command Results

| Result | Meaning |
|--------|---------|
| `SAT` | Instance found |
| `UNSAT` | No instance exists within scope |

### Console Output Format
```
00. check AssertionName      0       UNSAT
01. run   run$1              0    1/1     SAT
```

Format: `index. type name  states solutions result`

### Error Format
```
[main] ERROR alloy - ... Syntax error in file.als at line X column Y:
There are N possible tokens that can appear here:
<list of expected tokens>
```

## Alloy Model Structure

```alloy
/*
 * Module documentation
 */

-- Signatures (types)
sig Node {
    edges: set Node,      -- Set relation
    value: one Int        -- Single value
}

one sig Root extends Node {}  -- Singleton

-- Facts (constraints always true)
fact Constraints {
    all n: Node | n.value >= 0
}

-- Predicates (reusable constraints)
pred Connected[n1, n2: Node] {
    n2 in n1.^edges
}

-- Functions
fun ancestors[n: Node]: set Node {
    n.^~edges  -- Transitive closure of reverse edges
}

-- Assertions (properties to check)
assert NoSelfLoop {
    no n: Node | n in n.edges
}

-- Commands
check NoSelfLoop for 5        -- Check with scope 5
run Connected for 3 but 2 Int -- Run with specific scopes
```

## Operators Reference

| Operator | Meaning |
|----------|---------|
| `.` | Join (composition) |
| `+` | Union |
| `&` | Intersection |
| `-` | Difference |
| `^` | Transitive closure |
| `*` | Reflexive transitive closure |
| `~` | Transpose |
| `->` | Product (arrow) |
| `in` | Subset |
| `=` | Equality |
| `#` | Cardinality |

## Key Discoveries

1. **SAT/UNSAT reversed for check**: UNSAT = assertion holds, SAT = counterexample
2. **Scope matters**: Results only valid within specified scope
3. **JSON output**: Use `-t json -o -` for structured output
4. **Receipt file**: `receipt.json` contains all command results
5. **Skolems**: Counterexamples include skolem variables (witnesses)
6. **Exit code 0 even with counterexamples**: Exit code only indicates execution errors
