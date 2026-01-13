# Known Soundness Limitations - Detailed Reference

This file contains detailed explanations and examples for each soundness limitation. For a summary, see [CLAUDE.md](../CLAUDE.md).

---

## For Loops / Iterators

Range-based for loops (`for i in 0..n`) are now fully supported with semantic modeling. The CHC encoder properly expands `Range::into_iter()` and `Range::next()` with correct Option handling.

**Note:** Complex loops requiring loop invariant synthesis (e.g., summation loops computing `sum = 0 + 1 + 2 + ... + n`) may timeout as Z3 Spacer cannot automatically infer the required invariants.

---

## Trait Method Dispatch

Trait method calls (`x.trait_method()`) are not inlined and become uninterpreted functions.

**Workaround:** Use direct function calls or struct impl methods instead.

---

## Static Variables

Static variable reads are modeled as pointer dereferences with uninterpreted function semantics. The actual value stored in a static is not constrained in the CHC encoding.

**Workaround:** Use `const` for values that need precise verification.

---

## Non-linear Arithmetic

Multiplication of two variables (`n * m`, `n * n`) causes CHC solver timeout. This is a fundamental limitation of CHC-based verification with non-linear arithmetic (undecidable in general). Linear arithmetic works fine (`n * 2`, `n + m`).

**Workaround:** Use constants when possible or delegate to Kani for non-linear cases.

---

## Heap Allocation (Box, Vec, etc.)

**Box::new is now supported** (2026-01-05): `Box::new(v)` generates the constraint `*result == v` in the weakest precondition calculus. The value stored in the Box is correctly tracked.

**Vec and other collections** are still uninterpreted. `Vec::push`, `Vec::pop`, etc. do not have semantic models.

**Workaround for Vec:** Use arrays or fixed-size structures when possible.

---

## Array Repeat Syntax ([val; n])

Array repeat initialization `[5; 3]` generates an `Rvalue::Repeat` which is modeled as an uninterpreted function. The elements are unconstrained.

**Workaround:** Use explicit array literals `[5, 5, 5]` for precise verification.

---

## Ref and Ref Mut Patterns in Match

`ref` and `ref mut` patterns in match expressions involve pointer creation and dereferencing, which are modeled as uninterpreted functions. The dereferenced value is unconstrained.

**Workaround:** Use direct pattern matching (`(a, b)`) instead of ref patterns (`(ref a, ref b)`).

---

## Generic Functions

Generic functions like `Pair::<T>::new()` are not inlined during CHC encoding because their MIR is polymorphic. The return values are unconstrained.

**Workaround:** Use non-generic (monomorphic) functions: `fn new(a: i32, b: i32) -> IntPair` instead of `fn new<T>(a: T, b: T) -> Pair<T>`.

---

## Match Guards

Match guards like `Some(n) if n > 0` or `n if n > 0` extract bound values but don't constrain them correctly in the CHC encoding. The guard condition is checked but the extracted value becomes unconstrained.

**Workaround:** Use explicit match without guards and handle conditions in the arm body:
```rust
// Instead of: Some(n) if n > 0 => ...
Some(n) => { if n > 0 { ... } }

// Instead of: n if n > 0 => n
n => { if n > 0 { /* use n */ } }
```

---

## Inter-dependent Field Assignments

When assigning struct fields where one assignment reads another field that gets modified (e.g., `p.a = p.b; p.b = 10;`), the CHC encoding may not correctly preserve the temporal ordering. The first assignment may see the new value instead of the old value.

**Workaround:** Use explicit temporaries:
```rust
let old_b = p.b;
p.a = old_b;
p.b = 10;
```

---

## Deeply Nested Structs (3+ levels)

Struct fields nested 3 or more levels deep (e.g., `o.middle.inner.value`) have unconstrained values. The CHC encoding does not properly track deeply nested field initialization.

**Workaround:** Use at most 2 levels of struct nesting or flatten the structure.

---

## Const Generic Methods

Methods that return const generic parameters (e.g., `fn len(&self) -> usize { N }` where N is `const N: usize`) produce unconstrained values. The const generic parameter value is not properly propagated through method returns.

**Workaround:** Use explicit values or pass the const as a regular parameter.

---

## Array Destructuring Patterns

**Status:** Fixed in #214

Array destructuring patterns like `let [first, second, third] = arr;` are now fully supported. The `ConstantIndex` projection element is properly handled, generating correct SMT `select` operations to access array elements.

```rust
let arr = [1, 2, 3];
let [first, second, third] = arr;  // Works correctly: first=1, second=2, third=3
assert!(first == 1);  // Verified
```

---

## Closure Return Values

Closure calls return unconstrained values because the closure body is not inlined during CHC encoding. This includes both regular closures and `move` closures.

**Workaround:** Store results in variables before calling closures and verify the original values, or avoid using closure return values in assertions.

---

## Reference-Returning Functions

Functions that return references (`&T` or `&mut T`) have unconstrained dereference values. The CHC encoding does not track pointer aliasing - dereferencing a returned reference produces an unconstrained value.

**Workaround:** Verify the original data directly rather than through returned references.

---

## Reference Parameters to Enums

Functions that take enum reference parameters (`fn foo(e: &MyEnum)`) cause Z3 to return "unknown" with "model is not available" error. This is because the reference requires dereferencing to access the enum discriminant and variant data, which creates complex pointer aliasing queries that Z3 cannot solve.

**Workaround:** Use by-value parameters: `fn foo(e: MyEnum)` with `#[derive(Clone, Copy)]` on the enum.

---

## Enum Extraction via Reference

When extracting values from enum variants through a reference (`match &enum { Variant(v) => *v }`), the extracted value is unconstrained. The dereference operation creates a pointer query that the CHC solver cannot resolve.

```rust
enum Data { Value(i32) }
let d = Data::Value(42);
let extracted = match &d {
    Data::Value(v) => *v,  // extracted is unconstrained
};
assert!(extracted == 42);  // May incorrectly fail
```

**Workaround:** Pass enums by value or store fields in separate variables before passing to functions that take enum references.

---

## Char Type

**Status:** Fixed in #223

The `char` type is now properly encoded as u32 in the CHC system. Char literals and char field access work correctly.

```rust
let c = 'A';
assert!(c == 'A');  // Verified
```

---

## Array Index Assignment

Array index assignments (`arr[i] = value`) are not tracked. The write is dropped, so later reads see unconstrained values and assertions may be (incorrectly) proven.

```rust
let mut arr = [0, 0];
arr[0] = 5;
assert!(arr[0] == 5);  // May pass even though the write was ignored
```

**Workaround:** Avoid array mutation in proofs. Use struct fields or separate locals to model updates, or rebuild a new array literal after each change.

---

## Array Reference Parameters

When arrays are passed by reference to functions (`fn foo(arr: &[T; N])`), array index access (`arr[i]`) returns unconstrained values. The reference dereference combined with index access creates pointer aliasing that the CHC solver cannot track.

**Affected patterns:**
- Binary search on array references
- Array reversal algorithms
- Element counting functions
- Weighted sum calculations

```rust
fn sum_first_two(arr: &[i32; 3]) -> i32 {
    arr[0] + arr[1]  // Both accesses return unconstrained values
}

let data = [1, 2, 3];
let sum = sum_first_two(&data);
assert!(sum == 3);  // May incorrectly fail
```

**Workaround:** Pass array elements directly as separate parameters, or use struct fields instead of arrays.

---

## Nested Enum Types

Enums containing other enums as data (`enum Outer { A(Inner) }` where `Inner` is another enum) produce CHC unsatisfiable during match operations. The nested discriminant tracking is incomplete.

**Workaround:** Flatten the enum structure or use structs with discriminant fields.

---

## Recursion

Recursive function calls are modeled as uninterpreted functions. The recursive call returns an unconstrained value, so properties about recursive computations cannot be verified.

**Workaround:** Use iterative implementations (while loops) instead of recursion.

---

## assert_ne! and assert_eq! Macros

These macros expand to match expressions with reference patterns:
```rust
match (&x, &5) { (left_val, right_val) => if *left_val == *right_val { panic!() } }
```

The dereferences `*left_val` and `*right_val` become unconstrained values in the CHC encoding because reference patterns create intermediate pointers.

**Workaround:** Use direct assertions: `assert!(x != 5)` or `assert!(x == 5)`.

---

## Checked Arithmetic Intrinsics

**Status:** Fixed in #438

Most checked arithmetic intrinsics now correctly constrain Option values:
- `checked_add` - Fully supported
- `checked_sub` - Fully supported
- `checked_div` - Fully supported
- `checked_rem` - Fully supported
- `checked_shl` - Fully supported
- `checked_shr` - Fully supported
- `checked_mul` - Works but may timeout due to non-linear arithmetic

```rust
let x: u8 = 200;
let result = x.checked_add(50);  // Returns None (overflow)
assert!(result.is_none());       // Verified correctly

let y: u8 = 100;
let result = y.checked_add(50);  // Returns Some(150)
if let Some(v) = result {
    assert!(v == 150);           // Verified correctly
}
```

---

## Overflowing Arithmetic Intrinsics

`overflowing_add`, `overflowing_sub`, `overflowing_mul` return `(T, bool)` tuples where tuple extraction produces unconstrained values. Assertions about extracted values fail with CHC unsatisfiable.

```rust
let (result, overflowed) = 200u8.overflowing_add(100);
assert!(overflowed);  // May incorrectly fail - overflowed is unconstrained
```

**Workaround:** Use `wrapping_*` or `saturating_*` operations instead, which are fully supported.

---

## Bit Manipulation Intrinsics

**Status:** All supported in BitVec mode (#194, #195)

**Fully supported (with SMT-LIB2 encodings):**
- `rotate_left`, `rotate_right` - Variable rotation amounts supported
- `count_ones` (ctpop) - Manual bit-counting encoding
- `leading_zeros` (ctlz) - Cascade conditional encoding
- `trailing_zeros` (cttz) - Cascade conditional encoding
- `swap_bytes` (bswap) - Extract+concat encoding
- `reverse_bits` (bitreverse) - Extract+concat encoding

These intrinsics are encoded using SMT-LIB2 bitvector operations in `smt_intrinsics.rs`. The encodings generate O(n) terms for n-bit values but SMT solvers handle them efficiently.

**Note:** In non-BitVec (integer) mode, these intrinsics may still return unconstrained values. Enable BitVec mode (`--bitvec` or `KANI_FAST_BITVEC=1`) for full support.

---

## BitVec Boolean AND Bug (#340)

**Status:** Fixed in #343

When BitVec mode is enabled, the `bitand` operation applied to boolean values (e.g., overflow check results) was incorrectly converted to `bvand` instead of boolean `and`. This caused Z3 to report a type error and return SAT incorrectly, leading to false positives.

The fix in `convert_int_to_bitvec()` now detects boolean operands (comparison results, overflow flags) and uses `and`/`or`/`xor` instead of `bvand`/`bvor`/`bvxor`.

---

## Slice Operations and len()

Slice operations (`&arr[..]`, `&arr[n..]`, `&arr[..m]`) and the `len()` method return unconstrained values. The CHC encoding does not track slice metadata (pointer + length).

```rust
let arr = [1, 2, 3, 4, 5];
let slice = &arr[..];
assert!(slice.len() == 5);  // May incorrectly pass with wrong length
```

**Workaround:** Use array length directly via const generics or explicit constants.

---

## Empty Arrays

Empty arrays (`[T; 0]`) have unconstrained `len()` return values.

```rust
let arr: [i32; 0] = [];
assert!(arr.len() == 0);  // May incorrectly pass with wrong assertion
```

**Workaround:** Use explicit constants for array lengths.

---

## matches! Macro and Match-to-Bool Pattern

The `matches!` macro and equivalent match-to-bool patterns produce unconstrained boolean values when used with enum references:

```rust
let c = Color::Green;
let is_red = match &c {
    Color::Red => true,
    _ => false,
};
assert!(is_red);  // Should fail but may pass
```

**Workaround:** Use direct enum matching without boolean intermediaries, or match on owned values instead of references.

---

## Nested Tuple Destructuring

Nested tuple destructuring can produce unconstrained inner field values:

```rust
let nested = ((1i32, 2i32), (3i32, 4i32));
let ((a, b), (c, d)) = nested;
// Inner fields may be unconstrained
assert!(a + b + c + d == 10);  // May incorrectly pass or fail
```

**Workaround:** Destructure one level at a time, or assign intermediate tuples to variables first:

```rust
let nested = ((1i32, 2i32), (3i32, 4i32));
let (inner1, inner2) = nested;
let (a, b) = inner1;
let (c, d) = inner2;
// Now fields are properly constrained
```
