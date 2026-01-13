# TLA2 AI-Optimized Output Format

**Purpose:** Machine-readable output designed for AI agents and automated tooling.

**Flag:** `--output json` or `--output ai` (default remains human-readable)

---

## Design Principles

1. **Structured JSON** - No free-form text to parse
2. **Self-describing** - Field names are explicit and unambiguous
3. **Actionable** - Includes exact source locations (file:line:column)
4. **Diff-oriented** - State changes show what changed, not full state
5. **Categorized** - Errors have semantic types for automated handling
6. **Complete** - All information in one object, no need to correlate multiple outputs

---

## Output Schema

```json
{
  "version": "1.0",
  "tool": "tla2",
  "timestamp": "2025-12-29T18:04:37Z",

  "input": {
    "spec_file": "/path/to/spec.tla",
    "config_file": "/path/to/spec.cfg",
    "module": "SpecName",
    "workers": 4
  },

  "specification": {
    "init": "Init",
    "next": "Next",
    "invariants": ["TypeOK", "Safety"],
    "properties": ["Liveness"],
    "constants": {
      "N": 3,
      "MaxVal": 10
    },
    "variables": ["x", "y", "pc"]
  },

  "result": {
    "status": "error",  // "ok" | "error" | "timeout" | "interrupted"
    "error_type": "invariant_violation",  // see Error Types below
    "error_message": "Invariant 'Safety' violated",
    "violated_property": {
      "name": "Safety",
      "type": "invariant",
      "location": {
        "file": "/path/to/spec.tla",
        "line": 42,
        "column": 5,
        "end_line": 42,
        "end_column": 25
      },
      "expression": "x < MaxVal"
    }
  },

  "counterexample": {
    "length": 4,
    "states": [
      {
        "index": 1,
        "fingerprint": "21b0dcad88491ccc",
        "action": {
          "name": "Init",
          "type": "initial",
          "location": null
        },
        "variables": {
          "x": {"type": "int", "value": 0},
          "y": {"type": "set", "value": [1, 2, 3]},
          "pc": {"type": "string", "value": "ready"}
        },
        "diff_from_previous": null
      },
      {
        "index": 2,
        "fingerprint": "61040979918333cb",
        "action": {
          "name": "Increment",
          "type": "next",
          "location": {
            "file": "/path/to/spec.tla",
            "line": 15,
            "column": 1,
            "end_line": 17,
            "end_column": 20
          }
        },
        "variables": {
          "x": {"type": "int", "value": 1},
          "y": {"type": "set", "value": [1, 2, 3]},
          "pc": {"type": "string", "value": "running"}
        },
        "diff_from_previous": {
          "changed": {
            "x": {"from": 0, "to": 1},
            "pc": {"from": "ready", "to": "running"}
          },
          "unchanged": ["y"]
        }
      }
    ],
    "loop_start": null  // For liveness: index where loop begins
  },

  "statistics": {
    "states_found": 501552,
    "states_initial": 16,
    "states_distinct": 501552,
    "transitions": 9718320,
    "max_depth": 47,
    "max_queue_depth": 232295,
    "time_seconds": 4.123,
    "states_per_second": 121650,
    "memory_mb": 1024
  },

  "diagnostics": {
    "warnings": [
      {
        "code": "W001",
        "message": "Invariant 'TypeOK' is a constant expression",
        "location": {
          "file": "/path/to/spec.tla",
          "line": 30,
          "column": 1
        },
        "suggestion": "Use ASSUME instead of INVARIANT for constant expressions"
      }
    ],
    "info": [
      {
        "code": "I001",
        "message": "Using parallel mode with 4 workers"
      }
    ],
    "print_outputs": [
      {"value": "Test 1 OK", "location": {"file": "spec.tla", "line": 50}}
    ]
  },

  "actions_detected": [
    {
      "name": "Increment",
      "occurrences": 450000,
      "percentage": 46.3
    },
    {
      "name": "Decrement",
      "occurrences": 51552,
      "percentage": 5.3
    }
  ]
}
```

---

## Error Types

Semantic categorization for automated handling:

| Type | Description | Actionable? |
|------|-------------|-------------|
| `invariant_violation` | Safety property violated | Yes - shows counterexample |
| `liveness_violation` | Liveness property violated | Yes - shows loop |
| `deadlock` | No enabled actions from reachable state | Yes - shows trace to deadlock |
| `terminal_state` | Intentional termination (not error) | No - expected behavior |
| `assertion_failure` | Assert(...) evaluated to FALSE | Yes - shows failing state |
| `type_error` | Runtime type mismatch | Yes - shows expression |
| `undefined_value` | Evaluated to undefined | Yes - shows expression |
| `parse_error` | Syntax error in spec | Yes - shows location |
| `semantic_error` | Semantic analysis failed | Yes - shows location |
| `config_error` | Invalid configuration | Yes - describes issue |
| `resource_exhausted` | Out of memory/disk | No - increase resources |
| `timeout` | Exceeded time limit | No - incomplete result |

### Error Codes

Hierarchical error codes for precise categorization:

| Code Prefix | Category | Examples |
|-------------|----------|----------|
| `CFG_PARSE_*` | Config file parse errors | `CFG_PARSE_UNSUPPORTED_SYNTAX`, `CFG_PARSE_MISSING_VALUE` |
| `TLA_PARSE_*` | Spec parse errors | `TLA_PARSE_UNEXPECTED_TOKEN`, `TLA_PARSE_UNCLOSED_BRACKET` |
| `TLC_TYPE_*` | Type mismatches | `TLC_TYPE_MISMATCH`, `TLC_TYPE_NOT_A_SET` |
| `TLC_EVAL_*` | Evaluation errors | `TLC_EVAL_UNDEFINED`, `TLC_EVAL_OVERFLOW` |
| `TLC_DEADLOCK` | Deadlock detection | `TLC_DEADLOCK` |
| `TLC_INVARIANT_*` | Invariant violations | `TLC_INVARIANT_VIOLATED` |
| `TLC_LIVENESS_*` | Liveness violations | `TLC_LIVENESS_VIOLATED` |

### Detailed Error Schema

Errors include rich context for AI agents to understand and fix issues:

```json
{
  "error_code": "CFG_PARSE_UNSUPPORTED_SYNTAX",
  "severity": "error",
  "location": {
    "file": "cdcl.cfg",
    "line": 6,
    "column": 23,
    "end_line": 6,
    "end_column": 45,
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

**Type mismatch error**:
```json
{
  "error_code": "TLC_TYPE_MISMATCH",
  "severity": "error",
  "location": {"file": "spec.tla", "line": 47, "column": 8},
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

**Deadlock error with terminal state suggestion**:
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
      "Add TERMINAL state = \"UNSAT\" to config",
      "Add self-loop: Stutter == state = \"UNSAT\" /\\ UNCHANGED vars"
    ]
  }
}
```

**Key properties for AI agent consumption**:

| Property | Why It Helps |
|----------|--------------|
| `error_code` | Categorical - can match against known fixes |
| `location.line/column` | Exact position for automated edits |
| `found` / `expected` | Diff-style - shows exactly what's wrong |
| `suggestion.action` | Imperative - tells what to do |
| `suggestion.example` | Copy-paste ready fix |
| `suggestion.options` | Multiple fix strategies when applicable |
| `docs` | Link to relevant documentation |

---

## Compact Mode

For streaming/large outputs, use `--output json-lines` (JSONL):

```jsonl
{"type":"start","spec":"spec.tla","timestamp":"..."}
{"type":"progress","states":10000,"depth":15,"time":1.2}
{"type":"progress","states":50000,"depth":23,"time":5.1}
{"type":"error","error_type":"invariant_violation","state_index":4}
{"type":"state","index":1,"action":"Init","variables":{"x":0}}
{"type":"state","index":2,"action":"Step","variables":{"x":1},"diff":{"x":[0,1]}}
{"type":"state","index":3,"action":"Step","variables":{"x":2},"diff":{"x":[1,2]}}
{"type":"state","index":4,"action":"Step","variables":{"x":3},"diff":{"x":[2,3]}}
{"type":"done","status":"error","time":0.001}
```

---

## Value Encoding

TLA+ values map to JSON as follows:

| TLA+ Type | JSON Encoding |
|-----------|---------------|
| `Int` | `{"type": "int", "value": 42}` |
| `Bool` | `{"type": "bool", "value": true}` |
| `String` | `{"type": "string", "value": "hello"}` |
| `Set` | `{"type": "set", "value": [1, 2, 3]}` |
| `Sequence` | `{"type": "seq", "value": [1, 2, 3]}` |
| `Record` | `{"type": "record", "value": {"a": ..., "b": ...}}` |
| `Function` | `{"type": "function", "domain": [...], "mapping": [[key, val], ...]}` |
| `Tuple` | `{"type": "tuple", "value": [1, "a", true]}` |
| `Model Value` | `{"type": "model_value", "value": "m1"}` |

---

## AI Agent Integration Examples

### Python: Parse and Extract Counterexample

```python
import json
import subprocess

result = subprocess.run(
    ["tla2", "check", "spec.tla", "--output", "json"],
    capture_output=True, text=True
)
output = json.loads(result.stdout)

if output["result"]["status"] == "error":
    error = output["result"]
    print(f"Error: {error['error_type']}")
    print(f"Property: {error['violated_property']['name']}")
    print(f"Location: {error['violated_property']['location']['file']}:"
          f"{error['violated_property']['location']['line']}")

    # Show state diffs
    for state in output["counterexample"]["states"]:
        if state["diff_from_previous"]:
            for var, change in state["diff_from_previous"]["changed"].items():
                print(f"  {var}: {change['from']} -> {change['to']}")
```

### Bash: Quick Status Check

```bash
status=$(tla2 check spec.tla --output json | jq -r '.result.status')
if [ "$status" = "ok" ]; then
    echo "Model checking passed"
else
    tla2 check spec.tla --output json | jq '.result.error_message'
fi
```

### AI Agent: Automated Fix Suggestion

```python
output = json.loads(tla2_output)

if output["result"]["error_type"] == "invariant_violation":
    # Get the invariant that failed
    inv = output["result"]["violated_property"]

    # Get the state where it failed
    bad_state = output["counterexample"]["states"][-1]

    # Construct context for LLM
    context = f"""
    The invariant `{inv['expression']}` at {inv['location']['file']}:{inv['location']['line']}
    was violated in state:
    {json.dumps(bad_state['variables'], indent=2)}

    The transition that led to this state was `{bad_state['action']['name']}`
    at {bad_state['action']['location']['file']}:{bad_state['action']['location']['line']}.

    Changes from previous state:
    {json.dumps(bad_state['diff_from_previous']['changed'], indent=2)}

    Please suggest a fix.
    """
```

---

## Comparison: TLC vs TLA2 AI Format

### TLC Output (Hard to Parse)
```
Error: Invariant Inv is violated.
Error: The behavior up to this point is:
State 1: <Initial predicate>
x = 0

State 2: <Next line 6, col 9 to line 6, col 18 of module failing_spec>
x = 1
```

**Problems:**
- Must regex parse "line 6, col 9 to line 6, col 18"
- State delimiter is blank line (fragile)
- No structured variable types
- Action name mixed with location
- No diff between states

### TLA2 AI Format (Easy to Parse)
```json
{
  "counterexample": {
    "states": [
      {
        "index": 1,
        "action": {"name": "Init", "type": "initial"},
        "variables": {"x": {"type": "int", "value": 0}}
      },
      {
        "index": 2,
        "action": {
          "name": "Next",
          "location": {"line": 6, "column": 9, "end_column": 18}
        },
        "variables": {"x": {"type": "int", "value": 1}},
        "diff_from_previous": {"changed": {"x": {"from": 0, "to": 1}}}
      }
    ]
  }
}
```

**Benefits:**
- Direct JSON access: `output["counterexample"]["states"][1]["action"]["location"]["line"]`
- Typed values
- Explicit diffs
- No regex needed

---

## Implementation Plan

1. Add `--output` flag to CLI with options: `human` (default), `json`, `json-lines`
2. Create `OutputFormat` enum and formatter trait
3. Implement JSON serialization for all result types
4. Add state diff computation to trace output
5. Preserve human-readable format as default for backward compatibility

---

## Future Extensions

- `--output yaml` - YAML format for human-readable structured output
- `--output sarif` - SARIF format for IDE integration
- `--output prometheus` - Metrics format for monitoring
- WebSocket streaming for real-time progress updates
