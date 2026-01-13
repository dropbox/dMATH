# Lean5DB Text Format: Human & LLM Readable Export

**Status:** Draft v1.0
**Date:** 2026-01-06
**Purpose:** Define canonical text representation for debugging, diffing, and LLM analysis

---

## 1. Motivation

### 1.1 Use Cases

| Use Case | Requirement |
|----------|-------------|
| **Debugging** | See exactly what's in a .lean5db file |
| **Diffing** | Compare two versions with standard diff tools |
| **LLM Training** | Structured text for language model fine-tuning |
| **Documentation** | Generate human-readable library docs |
| **Verification** | Cross-check against Lean 4 source |
| **Auditing** | Review what theorems say without running Lean |
| **Search** | grep for constants, types, patterns |

### 1.2 Design Goals

1. **Canonical:** Same content → identical text output (byte-for-byte)
2. **Readable:** Humans can understand types and definitions
3. **Parseable:** Can round-trip back to structured data
4. **Diff-friendly:** Minimal spurious differences between versions
5. **LLM-friendly:** Good for tokenization, clear structure

---

## 2. Text Format Specification

### 2.1 File Structure

```
# Lean5DB Text Export
# Format: lean5db-text v1.0
# Source: mathlib.lean5db
# Hash: 3a7f9c2e...
# Exported: 2026-01-06T14:30:00Z
# Constants: 120847
# Modules: 1842

================================================================================
MODULE Init.Prelude
================================================================================

-- Inductive Types --

inductive Nat : Type
  | zero : Nat
  | succ : Nat → Nat

inductive Bool : Type
  | false : Bool
  | true : Bool

inductive List (α : Type u) : Type u
  | nil : List α
  | cons : α → List α → List α

-- Definitions --

def Nat.add : Nat → Nat → Nat :=
  fun n m =>
    Nat.rec m (fun _ ih => Nat.succ ih) n

def Nat.mul : Nat → Nat → Nat :=
  fun n m =>
    Nat.rec Nat.zero (fun _ ih => Nat.add m ih) n

-- Theorems --

theorem Nat.zero_add : ∀ (n : Nat), Nat.add Nat.zero n = n :=
  fun n => rfl

theorem Nat.add_zero : ∀ (n : Nat), Nat.add n Nat.zero = n :=
  fun n =>
    Nat.rec rfl (fun n ih => congrArg Nat.succ ih) n

-- [Proof omitted: 847 bytes, hash: 9f3a...]

theorem Nat.add_comm : ∀ (n m : Nat), Nat.add n m = Nat.add m n :=
  [Proof omitted: 12847 bytes, hash: 2c4f...]

================================================================================
MODULE Init.Core
================================================================================

-- [continues for all modules...]
```

### 2.2 Syntax Reference

#### Module Header
```
================================================================================
MODULE <fully.qualified.name>
================================================================================
```

#### Inductive Types
```
inductive <Name> <params> : <type>
  | <ctor1> : <ctor_type>
  | <ctor2> : <ctor_type>
  ...
```

#### Definitions
```
def <Name> <levels> : <type> :=
  <value>

-- or for large values:
def <Name> <levels> : <type> :=
  [Value omitted: <size> bytes, hash: <hash>]
```

#### Theorems
```
theorem <Name> <levels> : <type> :=
  <proof>

-- or with proof elision:
theorem <Name> <levels> : <type> :=
  [Proof omitted: <size> bytes, hash: <hash>]
```

#### Axioms
```
axiom <Name> <levels> : <type>
```

#### Opaque Definitions
```
opaque <Name> <levels> : <type> :=
  [Opaque value: <size> bytes, hash: <hash>]
```

### 2.3 Expression Syntax

Use Lean 4-like syntax for readability:

```
-- Function application (left associative)
f x y z        →  App(App(App(f, x), y), z)

-- Lambda
fun x => e     →  Lam(_, x_type, e)
fun (x : T) => e

-- Pi / forall
∀ (x : T), U   →  Pi(_, T, U)
T → U          →  Pi(_, T, U) when x not free in U

-- Let
let x : T := v in e  →  Let(T, v, e)

-- Sort
Type           →  Sort(Type 0)
Type u         →  Sort(Type u)
Prop           →  Sort(Prop)

-- Constant
Nat            →  Const("Nat", [])
List.cons.{u}  →  Const("List.cons", [u])

-- Bound variable (de Bruijn)
#0, #1, #2     →  BVar(0), BVar(1), BVar(2)

-- Projection
e.1            →  Proj(_, 0, e)
e.fst          →  Proj("Prod", 0, e)

-- Literals
42             →  Lit(Nat(42))
"hello"        →  Lit(String("hello"))
```

### 2.4 Level Syntax

```
0              →  Zero
1              →  Succ(Zero)
u              →  Param("u")
u + 1          →  Succ(Param("u"))
max u v        →  Max(u, v)
imax u v       →  IMax(u, v)
```

### 2.5 Pretty-Printing Rules

**Line width:** 100 characters max
**Indentation:** 2 spaces
**Large expressions:** Break at application boundaries

```
-- Short (inline)
def id : ∀ (α : Type u), α → α := fun α x => x

-- Medium (multi-line)
def compose : ∀ (α β γ : Type u), (β → γ) → (α → β) → α → γ :=
  fun α β γ g f x => g (f x)

-- Large (with breaks)
theorem long_theorem_name
    : ∀ (α : Type u) (β : Type v) (γ : Type w)
        (f : α → β) (g : β → γ) (h : α → γ),
        (∀ x, h x = g (f x)) →
        ∀ x, h x = g (f x) :=
  fun α β γ f g h H x => H x
```

---

## 3. Export Modes

### 3.1 Full Export (Default)

Everything included, proofs expanded:
```bash
lean5-fastload export mathlib.lean5db -o mathlib.txt
```

Size estimate for Mathlib: ~2-3 GB text

### 3.2 Types Only

Skip definition values and proof terms:
```bash
lean5-fastload export mathlib.lean5db -o mathlib-types.txt --types-only
```

Size estimate: ~200 MB text

### 3.3 Signatures Only

Just names and types, no values at all:
```bash
lean5-fastload export mathlib.lean5db -o mathlib-sigs.txt --signatures-only
```

Output:
```
def Nat.add : Nat → Nat → Nat
def Nat.mul : Nat → Nat → Nat
theorem Nat.add_comm : ∀ (n m : Nat), Nat.add n m = Nat.add m n
theorem Nat.add_assoc : ∀ (a b c : Nat), Nat.add (Nat.add a b) c = Nat.add a (Nat.add b c)
```

Size estimate: ~20 MB text

### 3.4 Module Filter

Export specific modules only:
```bash
lean5-fastload export mathlib.lean5db -o nat.txt --modules "Mathlib.Data.Nat.*"
```

### 3.5 Markdown Export (Recommended for Humans)

Human-readable with embedded Lean code blocks and optional LaTeX math:
```bash
lean5-fastload export mathlib.lean5db -o mathlib.md --format markdown
```

```markdown
# Module: Init.Prelude

## Inductive Types

### `Nat` : Type

The natural numbers, built from zero and successor.

```lean
inductive Nat : Type
  | zero : Nat
  | succ : Nat → Nat
```

**Constructors:**
- `Nat.zero : Nat`
- `Nat.succ : Nat → Nat`

---

## Definitions

### `Nat.add` : Nat → Nat → Nat

Addition of natural numbers.

```lean
def Nat.add : Nat → Nat → Nat :=
  fun n m => Nat.rec m (fun _ ih => Nat.succ ih) n
```

---

## Theorems

### `Nat.add_comm` : ∀ (n m : Nat), n + m = m + n

Commutativity of addition.

In LaTeX: $\forall (n\, m : \mathbb{N}),\; n + m = m + n$

```lean
theorem Nat.add_comm : ∀ (n m : Nat), n + m = m + n :=
  [Proof: 2847 bytes, hash: 9f3a2c4e...]
```
```

### 3.6 LaTeX Export (Academic/Papers)

Full LaTeX document with proper mathematical typesetting:
```bash
lean5-fastload export mathlib.lean5db -o mathlib.tex --format latex
```

```latex
\documentclass{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{listings}
\usepackage{hyperref}

\title{Mathlib Library Reference}
\author{Generated from mathlib.lean5db}

\begin{document}
\maketitle

\section{Module: Init.Prelude}

\subsection{Inductive Types}

\begin{definition}[Nat]
The natural numbers.
\begin{align*}
\mathbf{inductive}\ \mathtt{Nat} &: \mathsf{Type} \\
| \ \mathtt{zero} &: \mathtt{Nat} \\
| \ \mathtt{succ} &: \mathtt{Nat} \to \mathtt{Nat}
\end{align*}
\end{definition}

\subsection{Theorems}

\begin{theorem}[Nat.add\_comm]
\label{thm:nat-add-comm}
Addition of natural numbers is commutative:
\[
\forall (n\, m : \mathbb{N}),\quad n + m = m + n
\]
\end{theorem}

\begin{proof}
By induction on $n$. [Proof term: 2847 bytes]
\end{proof}

\end{document}
```

### 3.7 JSON/JSONL Export (Machine Processing)

For ML pipelines - JSON structure with Lean/LaTeX content strings:
```bash
lean5-fastload export mathlib.lean5db -o mathlib.jsonl --format jsonl
```

```jsonl
{"name":"Nat","kind":"inductive","type_lean":"Type","type_latex":"\\mathsf{Type}","module":"Init.Prelude"}
{"name":"Nat.add_comm","kind":"theorem","type_lean":"∀ (n m : Nat), n + m = m + n","type_latex":"\\forall (n\\, m : \\mathbb{N}),\\; n + m = m + n","module":"Init.Prelude"}
```

**Key insight:** JSON is the container, but content is Lean syntax AND LaTeX.

### 3.8 Format Comparison

| Format | Best For | Math Rendering | Machine Readable |
|--------|----------|----------------|------------------|
| **Lean text** | Authoritative reference, round-trip | Native Unicode (∀, →, ∈) | Parseable |
| **Markdown** | Documentation, GitHub, humans | Lean blocks + inline LaTeX | Semi-structured |
| **LaTeX** | Papers, PDFs, formal docs | Full typesetting | No |
| **JSON/JSONL** | ML training, APIs | Embedded Lean/LaTeX strings | Yes |

---

## 4. Import (Parse Text Back)

### 4.1 Text → Lean5DB

```bash
# Parse text format back to .lean5db
lean5-fastload import mathlib.txt -o mathlib-rebuilt.lean5db

# Verify round-trip
lean5-fastload verify-equivalent mathlib.lean5db mathlib-rebuilt.lean5db
```

### 4.2 Import API

```rust
pub fn parse_text_format(text: &str) -> Result<Vec<ParsedConstant>, ParseError>;

pub fn text_to_lean5db(
    text_path: &Path,
    output_path: &Path,
    options: BuildOptions,
) -> Result<(), Error>;
```

### 4.3 Partial Import

Import only specific constants (for patching):
```bash
# Export single constant
lean5-fastload export mathlib.lean5db --constant Nat.add -o nat_add.txt

# Edit the text file...

# Import back (creates overlay)
lean5-fastload import-patch nat_add.txt --base mathlib.lean5db -o mathlib-patched.lean5db
```

---

## 5. Verification Tools

### 5.1 Format Conversion Matrix

```
            ┌──────────┐
            │  .olean  │
            └────┬─────┘
                 │ parse
                 ▼
            ┌──────────┐
            │ In-Memory│
            │ (kernel) │
            └────┬─────┘
                 │
       ┌─────────┼─────────┐
       │         │         │
       ▼         ▼         ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │.lean5db │ │  .txt   │ │  .json  │
  └────┬────┘ └────┬────┘ └────┬────┘
       │           │           │
       └─────────┬─┴───────────┘
                 │
                 ▼
           ┌───────────┐
           │   Hash    │ ← Same hash = equivalent
           └───────────┘
```

### 5.2 Cross-Format Verification

```bash
# Verify .olean matches .lean5db
lean5-fastload check-equivalent \
    --olean ~/.elan/.../lib/lean \
    --lean5db mathlib.lean5db

# Verify .txt matches .lean5db
lean5-fastload check-equivalent \
    --text mathlib.txt \
    --lean5db mathlib.lean5db

# Verify all three match
lean5-fastload check-equivalent \
    --olean ~/.elan/.../lib/lean \
    --lean5db mathlib.lean5db \
    --text mathlib.txt
```

Output:
```
Checking equivalence...
  .olean hash:   3a7f9c2e8b4d1f6a...
  .lean5db hash: 3a7f9c2e8b4d1f6a...
  .txt hash:     3a7f9c2e8b4d1f6a...

✓ All formats are equivalent (hash: 3a7f9c2e8b4d1f6a...)
```

### 5.3 Diff Between Versions

```bash
# Human-readable diff
lean5-fastload diff mathlib-v4.4.lean5db mathlib-v4.5.lean5db

# Output:
# === Added Constants (47) ===
# + Mathlib.Data.Nat.Factors.new_theorem : ...
# + Mathlib.Algebra.Group.new_def : ...
#
# === Removed Constants (3) ===
# - Mathlib.Deprecated.old_lemma : ...
#
# === Modified Constants (12) ===
# ~ Mathlib.Data.Nat.Basic.some_theorem
#   Type: unchanged
#   Value: changed (hash: abc123 → def456)
#
# === Statistics ===
# v4.4: 120,000 constants
# v4.5: 120,044 constants
# Net change: +44
```

### 5.4 Semantic Diff (Type-Level)

```bash
# Show only API-breaking changes
lean5-fastload diff --semantic mathlib-v4.4.lean5db mathlib-v4.5.lean5db

# Output:
# === Breaking Changes ===
# ! Mathlib.Data.Nat.pow : Type changed
#   Old: Nat → Nat → Nat
#   New: Nat → Nat → Nat → Nat  ← BREAKING
#
# ! Mathlib.Algebra.Group.inv : Removed  ← BREAKING
#
# === Non-Breaking Changes ===
# + 47 new constants
# ~ 10 proof changes (types unchanged)
```

---

## 6. LLM-Friendly Features

### 6.1 Structured Prompts Export

Generate training data for theorem provers:

```bash
lean5-fastload export-prompts mathlib.lean5db -o prompts.jsonl
```

Output (one per line):
```json
{"type":"prove","statement":"∀ (n m : Nat), n + m = m + n","context":["def Nat.add","theorem Nat.zero_add","theorem Nat.succ_add"],"proof":"..."}
{"type":"define","name":"Nat.factorial","type":"Nat → Nat","examples":["factorial 0 = 1","factorial 5 = 120"]}
{"type":"complete","partial":"theorem foo : ∀ n, n + 0 = ","expected":"n := fun n => rfl"}
```

### 6.2 Context Windows

Export with dependency context for better LLM understanding:

```bash
lean5-fastload export mathlib.lean5db \
    --constant Nat.add_comm \
    --include-deps \
    --depth 3 \
    -o nat_add_comm_context.txt
```

Output:
```
-- Dependencies (depth 3) --

inductive Nat : Type
  | zero : Nat
  | succ : Nat → Nat

def Nat.add : Nat → Nat → Nat := ...

theorem Nat.zero_add : ∀ n, 0 + n = n := ...
theorem Nat.add_zero : ∀ n, n + 0 = n := ...
theorem Nat.succ_add : ∀ n m, (succ n) + m = succ (n + m) := ...
theorem Nat.add_succ : ∀ n m, n + (succ m) = succ (n + m) := ...

-- Target --

theorem Nat.add_comm : ∀ (n m : Nat), n + m = m + n :=
  fun n m =>
    Nat.rec
      (Nat.zero_add m)
      (fun n ih => calc
        succ n + m = succ (n + m) := Nat.succ_add n m
        _ = succ (m + n) := congrArg succ ih
        _ = m + succ n := (Nat.add_succ m n).symm)
      n
```

### 6.3 Tokenization Hints

Export with explicit token boundaries for better subword tokenization:

```bash
lean5-fastload export mathlib.lean5db --tokenize-hints -o mathlib-tokens.txt
```

Output:
```
▸theorem▸ ▸Nat▸.▸add▸_▸comm▸ ▸:▸ ▸∀▸ ▸(▸n▸ ▸m▸ ▸:▸ ▸Nat▸)▸,▸ ▸n▸ ▸+▸ ▸m▸ ▸=▸ ▸m▸ ▸+▸ ▸n▸
```

---

## 7. CLI Summary

```bash
# === Export Commands ===

# Full export to text
lean5-fastload export <input.lean5db> -o <output.txt>
lean5-fastload export <input.lean5db> -o <output.txt> --types-only
lean5-fastload export <input.lean5db> -o <output.txt> --signatures-only

# JSON export
lean5-fastload export <input.lean5db> -o <output.json> --format json
lean5-fastload export <input.lean5db> -o <output.jsonl> --format jsonl

# Filtered export
lean5-fastload export <input.lean5db> -o <output.txt> --modules "Init.*"
lean5-fastload export <input.lean5db> -o <output.txt> --constant Nat.add
lean5-fastload export <input.lean5db> -o <output.txt> --constant Nat.add --include-deps

# === Import Commands ===

# Text to lean5db
lean5-fastload import <input.txt> -o <output.lean5db>

# === Verification Commands ===

# Check format equivalence
lean5-fastload check-equivalent --olean <dir> --lean5db <file>
lean5-fastload check-equivalent --text <file.txt> --lean5db <file>

# Diff
lean5-fastload diff <old.lean5db> <new.lean5db>
lean5-fastload diff <old.lean5db> <new.lean5db> --semantic
lean5-fastload diff <old.lean5db> <new.lean5db> --json

# Verify internal consistency
lean5-fastload verify <file.lean5db>
lean5-fastload verify <file.lean5db> --full  # recompute all hashes

# === LLM Export Commands ===

lean5-fastload export-prompts <input.lean5db> -o <prompts.jsonl>
lean5-fastload export <input.lean5db> -o <output.txt> --tokenize-hints
```

---

## 8. Implementation Notes

### 8.1 Pretty Printer

```rust
pub struct PrettyPrinter {
    /// Maximum line width
    max_width: usize,

    /// Current indentation
    indent: usize,

    /// Whether to include values
    include_values: bool,

    /// Whether to include proofs
    include_proofs: bool,

    /// Hash placeholder for omitted content
    show_omission_hashes: bool,
}

impl PrettyPrinter {
    pub fn print_constant(&mut self, c: &ConstantInfo) -> String;
    pub fn print_expr(&mut self, e: &Expr) -> String;
    pub fn print_level(&mut self, l: &Level) -> String;
    pub fn print_module(&mut self, m: &ModuleInfo, constants: &[ConstantInfo]) -> String;
}
```

### 8.2 Parser

```rust
pub struct TextParser {
    /// Current position
    pos: usize,

    /// Input text
    input: &str,

    /// Errors accumulated
    errors: Vec<ParseError>,
}

impl TextParser {
    pub fn parse_file(&mut self) -> Result<Vec<ParsedModule>, ParseError>;
    pub fn parse_constant(&mut self) -> Result<ParsedConstant, ParseError>;
    pub fn parse_expr(&mut self) -> Result<ParsedExpr, ParseError>;
}
```

### 8.3 Round-Trip Property

**Invariant:** For any valid .lean5db file:
```
parse_text(print_text(load(file))) ≡ load(file)
```

This is verified by:
1. Canonical hash before export
2. Canonical hash after import
3. Hashes must match

---

## 9. Example Output

### 9.1 Sample Module (Init.Prelude excerpt)

```
================================================================================
MODULE Init.Prelude
================================================================================

-- Source: lean5db v1.1
-- Module hash: 7a3f9c2e8b4d1f6a...
-- Constants: 847

--------------------------------------------------------------------------------
-- Universes
--------------------------------------------------------------------------------

universe u v w

--------------------------------------------------------------------------------
-- Basic Inductive Types
--------------------------------------------------------------------------------

/-- The natural numbers, built from zero and successor. -/
inductive Nat : Type
  | zero : Nat
  | succ : Nat → Nat

/-- Boolean values. -/
inductive Bool : Type
  | false : Bool
  | true : Bool

/-- Propositional equality. -/
inductive Eq.{u} {α : Sort u} (a : α) : α → Prop
  | refl : Eq a a

/-- The unit type with a single inhabitant. -/
inductive Unit : Type
  | unit : Unit

/-- The empty type with no inhabitants. -/
inductive Empty : Type

/-- Dependent pairs (Sigma types). -/
inductive Sigma.{u, v} {α : Type u} (β : α → Type v) : Type (max u v)
  | mk : (fst : α) → β fst → Sigma β

--------------------------------------------------------------------------------
-- Basic Definitions
--------------------------------------------------------------------------------

/-- Addition of natural numbers. -/
def Nat.add : Nat → Nat → Nat :=
  fun n m => Nat.rec m (fun _ ih => Nat.succ ih) n

/-- Multiplication of natural numbers. -/
def Nat.mul : Nat → Nat → Nat :=
  fun n m => Nat.rec Nat.zero (fun _ ih => Nat.add m ih) n

/-- Boolean negation. -/
def Bool.not : Bool → Bool :=
  fun b => Bool.casesOn b Bool.true Bool.false

/-- Boolean conjunction. -/
def Bool.and : Bool → Bool → Bool :=
  fun a b => Bool.casesOn a Bool.false b

--------------------------------------------------------------------------------
-- Basic Theorems
--------------------------------------------------------------------------------

/-- Zero is a left identity for addition. -/
theorem Nat.zero_add : ∀ (n : Nat), Nat.add Nat.zero n = n :=
  fun n => Eq.refl n

/-- Zero is a right identity for addition. -/
theorem Nat.add_zero : ∀ (n : Nat), Nat.add n Nat.zero = n :=
  fun n => Nat.rec (Eq.refl Nat.zero) (fun n ih => congrArg Nat.succ ih) n

/-- Addition is commutative. -/
theorem Nat.add_comm : ∀ (n m : Nat), Nat.add n m = Nat.add m n :=
  [Proof: 2847 bytes, hash: 9f3a2c4e...]

/-- Addition is associative. -/
theorem Nat.add_assoc : ∀ (a b c : Nat), Nat.add (Nat.add a b) c = Nat.add a (Nat.add b c) :=
  [Proof: 1923 bytes, hash: 4b7d8e1f...]

--------------------------------------------------------------------------------
-- Recursors
--------------------------------------------------------------------------------

/-- Recursor for Nat. -/
recursor Nat.rec.{u} : {motive : Nat → Sort u} →
    motive Nat.zero →
    (∀ (n : Nat), motive n → motive (Nat.succ n)) →
    ∀ (t : Nat), motive t

/-- Recursor for Bool. -/
recursor Bool.rec.{u} : {motive : Bool → Sort u} →
    motive Bool.false →
    motive Bool.true →
    ∀ (t : Bool), motive t

-- End of module Init.Prelude
-- Total: 847 constants, 23 inductives, 156 definitions, 412 theorems
```

---

*End of Text Format Specification*
