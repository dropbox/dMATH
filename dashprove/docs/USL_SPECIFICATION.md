# USL Specification

Unified Specification Language (USL) is the front-end language for DashProve. It is parsed by `crates/dashprove-usl/src/usl.pest`, type-checked in `crates/dashprove-usl/src/typecheck.rs`, and compiled to each backend by `crates/dashprove-usl/src/compile.rs`.

## Syntax Overview
- A file contains zero or more `type` definitions followed by properties.
- Supported property kinds:
  - `theorem <name> { <expr> }`
  - `temporal <name> { <temporal_expr> }`
  - `contract <Type::method(params) -> Return> { requires {...} ensures {...} ensures_err {...} }`
  - `invariant <name> { <expr> }`
  - `refinement <name> refines <base> { abstraction {..} simulation {..} }`
  - `probabilistic <name> { probability(<expr>) <cmp> <float> }`
  - `security <name> { <expr> }`
- Types: `Bool`, `Int`, `Float`, `String`, user `Named`, `Set<T>`, `List<T>`, `Map<K,V>`, `Relation<A,B>`, `Result<T>`, functions `A -> B`, and unit `()`.
- Identifiers: ASCII with `_`/`'` suffixes allowed; comments use `//` or `/* */`.

## Expressions
- Quantifiers: `forall x: T . body`, `exists x: T . body`, `forall x in (collection) . body`, `exists x in (collection) . body`.
- Calls and access: `f(a, b)`, `obj.method(args)`, `obj.field`.
- Literals: integers, floats, strings in double quotes, booleans `true`/`false`.
- Comparisons: `== != < <= > >=` (operand types must match).
- Arithmetic: `+ - * / %`, unary `-` or `!` via `neg_op`.
- Boolean ops:
  - Implication is right-associative: `a implies b implies c`.
  - `or` / `and` chain left-to-right.
  - `not` binds tighter than comparisons.
- Precedence (high → low): unary `-`/`!`, multiplicative `* / %`, additive `+ -`, `not`, comparisons, `and`, `or`, `implies`.

### Temporal Logic
- Forms: `always(<temporal_expr>)`, `eventually(<temporal_expr>)`, leadsto `<a> ~> <b>`.
- Atomic temporal predicates embed regular USL expressions.

### Contracts
- Signature: `contract Type::method(arg: Type, ...) -> Return { ... }`.
- Clauses:
  - `requires { bool }` preconditions.
  - `ensures { bool }` postconditions on primed state/value (use `self'` for post-state).
  - `ensures_err { bool }` conditions when the function fails.
- Parameters and return types are optional for contracts that reference existing implementations.

### Probabilistic and Security
- Probabilistic: `probability(<expr>) <cmp> <float>` with bound in `[0.0, 1.0]`.
- Security: arbitrary boolean formulas; built-in helper `actions()` returns a collection.

## Type Checking Rules
- Theorem, invariant, and security bodies must be boolean.
- Contract `requires`/`ensures`/`ensures_err` clauses must be boolean.
- Arithmetic operators require numeric operands; logical operators require booleans.
- Comparisons require operands of the same type.
- Iteration bindings (`forall x in set`) require set/list types; otherwise `NotIterable` is raised.
- Built-in predicates registered by default: `acyclic`, `is_terminal`, `reachable`, `enabled`, `at_checkpoint`, `contains`, `can_observe`, `executes`, `actions`, and `probability`. Unknown function names are rejected unless added as user-defined functions.

## Standard Library
- Reference templates live in `usl/`:
  - `prelude.usl` (logic/arith lemmas),
  - `graph.usl` (graph types + invariants),
  - `temporal.usl`,
  - `contracts.usl`.
- USL currently inlines specs; imports are conceptual only.

## Backend Mapping
- Theorems/Invariants/Refinements → LEAN 4, Alloy (bounded), Coq (planned).
- Temporal → TLA+.
- Contracts → Kani harnesses.
- Probabilistic/Security → placeholders for Storm/PRISM and Tamarin/ProVerif (not yet executed).

## Example
```usl
type Counter = { value: Int }

theorem non_negative { forall c: Counter . c.value >= 0 }

contract Counter::inc(self: Counter) -> Counter {
    requires { self.value >= 0 }
    ensures  { self'.value == self.value + 1 }
    ensures_err { self' == self }
}

temporal progress { always(eventually(reachable(state, goal, graph))) }
```
