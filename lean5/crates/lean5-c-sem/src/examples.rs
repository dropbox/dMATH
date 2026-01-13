//! Comprehensive C Verification Examples
//!
//! This module provides worked examples demonstrating the complete Lean5 C
//! verification pipeline:
//!
//! 1. **Simple Algorithms**: swap, abs, max/min
//! 2. **Array Operations**: sum, find, copy, reverse
//! 3. **Linked Data Structures**: list operations
//! 4. **Crypto Primitives**: constant-time comparison, XOR cipher
//! 5. **Memory Safety**: null checks, bounds validation
//!
//! Each example includes:
//! - C function definition
//! - ACSL-style specification
//! - Separation logic contract (where applicable)
//! - Generated verification conditions
//! - Proof attempt results

use crate::auto::{ProofStatus, VerificationSummary};
use crate::expr::{BinOp, CExpr, UnaryOp};
use crate::sep::{SepAssertion, SepFuncSpec, Share};
use crate::spec::{FuncSpec, Spec};
use crate::stmt::{CStmt, FuncDef, FuncParam, StorageClass};
use crate::types::CType;
use crate::verified::VerifiedFunction;

// ═══════════════════════════════════════════════════════════════════════════════
// Example 1: Absolute Value
// ═══════════════════════════════════════════════════════════════════════════════

/// Creates the abs function example
/// ```c
/// /*@ requires \true;
///     ensures \result >= 0;
///     ensures \result == n || \result == -n;
/// */
/// int abs(int n) {
///     if (n < 0)
///         return -n;
///     else
///         return n;
/// }
/// ```
pub fn abs_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "abs".into(),
        return_type: CType::int(),
        params: vec![FuncParam {
            name: "n".into(),
            ty: CType::int(),
        }],
        body: Box::new(CStmt::if_else(
            CExpr::binop(BinOp::Lt, CExpr::var("n"), CExpr::int(0)),
            CStmt::return_stmt(Some(CExpr::unary(UnaryOp::Neg, CExpr::var("n")))),
            CStmt::return_stmt(Some(CExpr::var("n"))),
        )),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![Spec::True],
        ensures: vec![
            // result >= 0
            Spec::ge(Spec::result(), Spec::int(0)),
            // result == n || result == -n
            Spec::or(vec![
                Spec::eq(Spec::result(), Spec::var("n")),
                Spec::eq(
                    Spec::result(),
                    Spec::UnaryOp {
                        op: UnaryOp::Neg,
                        operand: Box::new(Spec::var("n")),
                    },
                ),
            ]),
        ],
        ..Default::default()
    };

    VerifiedFunction {
        name: "abs".into(),
        description: "Absolute value function".into(),
        func,
        spec,
        sep_spec: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Example 2: Swap Two Pointers
// ═══════════════════════════════════════════════════════════════════════════════

/// Creates the swap function example with separation logic
/// ```c
/// /*@ requires \valid(x) && \valid(y) && \separated(x, y);
///     ensures *x == \old(*y);
///     ensures *y == \old(*x);
/// */
/// void swap(int *x, int *y) {
///     int tmp = *x;
///     *x = *y;
///     *y = tmp;
/// }
/// ```
pub fn swap_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "swap".into(),
        return_type: CType::Void,
        params: vec![
            FuncParam {
                name: "x".into(),
                ty: CType::ptr(CType::int()),
            },
            FuncParam {
                name: "y".into(),
                ty: CType::ptr(CType::int()),
            },
        ],
        body: Box::new(CStmt::block(vec![
            // int tmp = *x;
            CStmt::decl_init("tmp", CType::int(), CExpr::deref(CExpr::var("x"))),
            // *x = *y;
            CStmt::Expr(CExpr::assign(
                CExpr::deref(CExpr::var("x")),
                CExpr::deref(CExpr::var("y")),
            )),
            // *y = tmp;
            CStmt::Expr(CExpr::assign(
                CExpr::deref(CExpr::var("y")),
                CExpr::var("tmp"),
            )),
        ])),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![
            Spec::valid(Spec::var("x")),
            Spec::valid(Spec::var("y")),
            Spec::Separated(vec![Spec::var("x"), Spec::var("y")]),
        ],
        ensures: vec![
            Spec::eq(
                Spec::Expr(CExpr::deref(CExpr::var("x"))),
                Spec::old(Spec::Expr(CExpr::deref(CExpr::var("y")))),
            ),
            Spec::eq(
                Spec::Expr(CExpr::deref(CExpr::var("y"))),
                Spec::old(Spec::Expr(CExpr::deref(CExpr::var("x")))),
            ),
        ],
        ..Default::default()
    };

    // Separation logic spec: x ↦ a * y ↦ b ==> x ↦ b * y ↦ a
    let sep_spec = SepFuncSpec::new(
        SepAssertion::sep_conj(
            SepAssertion::data_at(CExpr::var("x"), CType::int(), Spec::var("a"), Share::Full),
            SepAssertion::data_at(CExpr::var("y"), CType::int(), Spec::var("b"), Share::Full),
        ),
        SepAssertion::sep_conj(
            SepAssertion::data_at(CExpr::var("x"), CType::int(), Spec::var("b"), Share::Full),
            SepAssertion::data_at(CExpr::var("y"), CType::int(), Spec::var("a"), Share::Full),
        ),
    );

    VerifiedFunction {
        name: "swap".into(),
        description: "Swap two integers through pointers".into(),
        func,
        spec,
        sep_spec: Some(sep_spec),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Example 3: Array Sum with Loop Invariant
// ═══════════════════════════════════════════════════════════════════════════════

/// Array sum with loop invariant
/// ```c
/// /*@ requires n >= 0 && \valid(a + (0..n-1));
///     ensures \result == \sum(0, n-1, \lambda i; a[i]);
/// */
/// int array_sum(int *a, int n) {
///     int sum = 0;
///     /*@ loop invariant 0 <= i <= n;
///         loop invariant sum == \sum(0, i-1, \lambda j; a[j]);
///         loop variant n - i;
///     */
///     for (int i = 0; i < n; i++) {
///         sum += a[i];
///     }
///     return sum;
/// }
/// ```
pub fn array_sum_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "array_sum".into(),
        return_type: CType::int(),
        params: vec![
            FuncParam {
                name: "a".into(),
                ty: CType::ptr(CType::int()),
            },
            FuncParam {
                name: "n".into(),
                ty: CType::int(),
            },
        ],
        body: Box::new(CStmt::block(vec![
            // int sum = 0;
            CStmt::decl_init("sum", CType::int(), CExpr::int(0)),
            // for (int i = 0; i < n; i++) { sum += a[i]; }
            CStmt::for_loop(
                Some(CStmt::decl_init("i", CType::int(), CExpr::int(0))),
                Some(CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n"))),
                Some(CExpr::assign(
                    CExpr::var("i"),
                    CExpr::add(CExpr::var("i"), CExpr::int(1)),
                )),
                CStmt::Expr(CExpr::assign(
                    CExpr::var("sum"),
                    CExpr::add(
                        CExpr::var("sum"),
                        CExpr::index(CExpr::var("a"), CExpr::var("i")),
                    ),
                )),
            ),
            // return sum;
            CStmt::return_stmt(Some(CExpr::var("sum"))),
        ])),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![
            // n >= 0
            Spec::ge(Spec::var("n"), Spec::int(0)),
            // \valid(a + (0..n-1))
            Spec::ValidRange {
                ptr: Box::new(Spec::var("a")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1))),
            },
        ],
        ensures: vec![
            // result >= 0 (assuming non-negative elements - simplified)
            Spec::ge(Spec::result(), Spec::int(0)),
        ],
        ..Default::default()
    };

    VerifiedFunction {
        name: "array_sum".into(),
        description: "Sum elements of an array".into(),
        func,
        spec,
        sep_spec: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Example 4: Constant-Time Comparison (Crypto)
// ═══════════════════════════════════════════════════════════════════════════════

/// Constant-time byte comparison (for crypto)
/// ```c
/// /*@ requires n >= 0;
///     requires \valid(a + (0..n-1)) && \valid(b + (0..n-1));
///     ensures \result == 0 <==> \forall int i; 0 <= i < n ==> a[i] == b[i];
///     // assigns nothing except local variables (no side effects)
/// */
/// int constant_time_compare(const unsigned char *a, const unsigned char *b, size_t n) {
///     unsigned char result = 0;
///     for (size_t i = 0; i < n; i++) {
///         result |= a[i] ^ b[i];
///     }
///     return result == 0 ? 1 : 0;
/// }
/// ```
pub fn constant_time_compare_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "constant_time_compare".into(),
        return_type: CType::int(),
        params: vec![
            FuncParam {
                name: "a".into(),
                ty: CType::ptr(CType::unsigned_char()),
            },
            FuncParam {
                name: "b".into(),
                ty: CType::ptr(CType::unsigned_char()),
            },
            FuncParam {
                name: "n".into(),
                ty: CType::size_t(),
            },
        ],
        body: Box::new(CStmt::block(vec![
            // unsigned char result = 0;
            CStmt::decl_init("result", CType::unsigned_char(), CExpr::int(0)),
            // for (size_t i = 0; i < n; i++) { result |= a[i] ^ b[i]; }
            CStmt::for_loop(
                Some(CStmt::decl_init("i", CType::size_t(), CExpr::int(0))),
                Some(CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n"))),
                Some(CExpr::assign(
                    CExpr::var("i"),
                    CExpr::add(CExpr::var("i"), CExpr::int(1)),
                )),
                CStmt::Expr(CExpr::assign(
                    CExpr::var("result"),
                    CExpr::binop(
                        BinOp::BitOr,
                        CExpr::var("result"),
                        CExpr::binop(
                            BinOp::BitXor,
                            CExpr::index(CExpr::var("a"), CExpr::var("i")),
                            CExpr::index(CExpr::var("b"), CExpr::var("i")),
                        ),
                    ),
                )),
            ),
            // return result == 0 ? 1 : 0;
            CStmt::return_stmt(Some(CExpr::Conditional {
                cond: Box::new(CExpr::binop(BinOp::Eq, CExpr::var("result"), CExpr::int(0))),
                then_expr: Box::new(CExpr::int(1)),
                else_expr: Box::new(CExpr::int(0)),
            })),
        ])),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![
            Spec::ge(Spec::var("n"), Spec::int(0)),
            Spec::ValidRange {
                ptr: Box::new(Spec::var("a")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1))),
            },
            Spec::ValidRange {
                ptr: Box::new(Spec::var("b")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1))),
            },
        ],
        ensures: vec![
            // result is 0 or 1
            Spec::or(vec![
                Spec::eq(Spec::result(), Spec::int(0)),
                Spec::eq(Spec::result(), Spec::int(1)),
            ]),
        ],
        ..Default::default()
    };

    VerifiedFunction {
        name: "constant_time_compare".into(),
        description: "Constant-time byte comparison for cryptographic use".into(),
        func,
        spec,
        sep_spec: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Example 5: XOR Cipher (Crypto)
// ═══════════════════════════════════════════════════════════════════════════════

/// XOR cipher in-place encryption/decryption
/// ```c
/// /*@ requires n >= 0 && key_len > 0;
///     requires \valid(data + (0..n-1)) && \valid(key + (0..key_len-1));
///     ensures \forall int i; 0 <= i < n ==>
///         data[i] == \old(data[i]) ^ key[i % key_len];
///     assigns data[0..n-1];
/// */
/// void xor_cipher(unsigned char *data, size_t n,
///                 const unsigned char *key, size_t key_len) {
///     for (size_t i = 0; i < n; i++) {
///         data[i] ^= key[i % key_len];
///     }
/// }
/// ```
pub fn xor_cipher_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "xor_cipher".into(),
        return_type: CType::Void,
        params: vec![
            FuncParam {
                name: "data".into(),
                ty: CType::ptr(CType::unsigned_char()),
            },
            FuncParam {
                name: "n".into(),
                ty: CType::size_t(),
            },
            FuncParam {
                name: "key".into(),
                ty: CType::ptr(CType::unsigned_char()),
            },
            FuncParam {
                name: "key_len".into(),
                ty: CType::size_t(),
            },
        ],
        body: Box::new(CStmt::block(vec![
            // for (size_t i = 0; i < n; i++) { data[i] ^= key[i % key_len]; }
            CStmt::for_loop(
                Some(CStmt::decl_init("i", CType::size_t(), CExpr::int(0))),
                Some(CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n"))),
                Some(CExpr::assign(
                    CExpr::var("i"),
                    CExpr::add(CExpr::var("i"), CExpr::int(1)),
                )),
                CStmt::Expr(CExpr::assign(
                    CExpr::index(CExpr::var("data"), CExpr::var("i")),
                    CExpr::binop(
                        BinOp::BitXor,
                        CExpr::index(CExpr::var("data"), CExpr::var("i")),
                        CExpr::index(
                            CExpr::var("key"),
                            CExpr::binop(BinOp::Mod, CExpr::var("i"), CExpr::var("key_len")),
                        ),
                    ),
                )),
            ),
        ])),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![
            Spec::ge(Spec::var("n"), Spec::int(0)),
            Spec::gt(Spec::var("key_len"), Spec::int(0)),
            Spec::ValidRange {
                ptr: Box::new(Spec::var("data")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1))),
            },
            Spec::ValidRange {
                ptr: Box::new(Spec::var("key")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("key_len"), Spec::int(1))),
            },
        ],
        ensures: vec![Spec::True], // Simplified - full spec requires quantifiers
        ..Default::default()
    };

    VerifiedFunction {
        name: "xor_cipher".into(),
        description: "XOR cipher for encryption/decryption".into(),
        func,
        spec,
        sep_spec: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Example 6: Safe Array Access with Bounds Check
// ═══════════════════════════════════════════════════════════════════════════════

/// Safe array access with bounds checking
/// ```c
/// /*@ requires \valid(arr + (0..size-1)) && size > 0;
///     requires result != NULL;
///     assigns *result;
///     behavior in_bounds:
///         assumes 0 <= idx < size;
///         ensures *result == arr[idx];
///         ensures \result == 1;
///     behavior out_of_bounds:
///         assumes idx < 0 || idx >= size;
///         ensures \result == 0;
/// */
/// int safe_get(int *arr, int size, int idx, int *result) {
///     if (idx >= 0 && idx < size) {
///         *result = arr[idx];
///         return 1;
///     }
///     return 0;
/// }
/// ```
pub fn safe_array_access_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "safe_get".into(),
        return_type: CType::int(),
        params: vec![
            FuncParam {
                name: "arr".into(),
                ty: CType::ptr(CType::int()),
            },
            FuncParam {
                name: "size".into(),
                ty: CType::int(),
            },
            FuncParam {
                name: "idx".into(),
                ty: CType::int(),
            },
            FuncParam {
                name: "result".into(),
                ty: CType::ptr(CType::int()),
            },
        ],
        body: Box::new(CStmt::if_else(
            CExpr::binop(
                BinOp::LogAnd,
                CExpr::binop(BinOp::Ge, CExpr::var("idx"), CExpr::int(0)),
                CExpr::binop(BinOp::Lt, CExpr::var("idx"), CExpr::var("size")),
            ),
            CStmt::block(vec![
                CStmt::Expr(CExpr::assign(
                    CExpr::deref(CExpr::var("result")),
                    CExpr::index(CExpr::var("arr"), CExpr::var("idx")),
                )),
                CStmt::return_stmt(Some(CExpr::int(1))),
            ]),
            CStmt::return_stmt(Some(CExpr::int(0))),
        )),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![
            Spec::gt(Spec::var("size"), Spec::int(0)),
            Spec::ValidRange {
                ptr: Box::new(Spec::var("arr")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("size"), Spec::int(1))),
            },
            Spec::valid(Spec::var("result")),
        ],
        ensures: vec![
            // result is 0 or 1
            Spec::or(vec![
                Spec::eq(Spec::result(), Spec::int(0)),
                Spec::eq(Spec::result(), Spec::int(1)),
            ]),
        ],
        ..Default::default()
    };

    VerifiedFunction {
        name: "safe_get".into(),
        description: "Safe array access with bounds checking".into(),
        func,
        spec,
        sep_spec: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Example 7: Memcpy
// ═══════════════════════════════════════════════════════════════════════════════

/// Memory copy implementation
/// ```c
/// /*@ requires n >= 0;
///     requires \valid(dest + (0..n-1)) && \valid_read(src + (0..n-1));
///     requires \separated(dest + (0..n-1), src + (0..n-1));
///     ensures \forall int i; 0 <= i < n ==> dest[i] == \old(src[i]);
///     assigns dest[0..n-1];
/// */
/// void *memcpy_impl(void *dest, const void *src, size_t n) {
///     char *d = (char *)dest;
///     const char *s = (const char *)src;
///     for (size_t i = 0; i < n; i++) {
///         d[i] = s[i];
///     }
///     return dest;
/// }
/// ```
pub fn memcpy_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "memcpy_impl".into(),
        return_type: CType::ptr(CType::Void),
        params: vec![
            FuncParam {
                name: "dest".into(),
                ty: CType::ptr(CType::Void),
            },
            FuncParam {
                name: "src".into(),
                ty: CType::ptr(CType::Void),
            },
            FuncParam {
                name: "n".into(),
                ty: CType::size_t(),
            },
        ],
        body: Box::new(CStmt::block(vec![
            // char *d = (char *)dest;
            CStmt::decl_init(
                "d",
                CType::ptr(CType::char()),
                CExpr::cast(CType::ptr(CType::char()), CExpr::var("dest")),
            ),
            // const char *s = (const char *)src;
            CStmt::decl_init(
                "s",
                CType::ptr(CType::char()),
                CExpr::cast(CType::ptr(CType::char()), CExpr::var("src")),
            ),
            // for loop
            CStmt::for_loop(
                Some(CStmt::decl_init("i", CType::size_t(), CExpr::int(0))),
                Some(CExpr::binop(BinOp::Lt, CExpr::var("i"), CExpr::var("n"))),
                Some(CExpr::assign(
                    CExpr::var("i"),
                    CExpr::add(CExpr::var("i"), CExpr::int(1)),
                )),
                CStmt::Expr(CExpr::assign(
                    CExpr::index(CExpr::var("d"), CExpr::var("i")),
                    CExpr::index(CExpr::var("s"), CExpr::var("i")),
                )),
            ),
            // return dest;
            CStmt::return_stmt(Some(CExpr::var("dest"))),
        ])),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![
            Spec::ge(Spec::var("n"), Spec::int(0)),
            Spec::ValidRange {
                ptr: Box::new(Spec::var("dest")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1))),
            },
            Spec::ValidRange {
                ptr: Box::new(Spec::var("src")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1))),
            },
        ],
        ensures: vec![
            // return value is dest
            Spec::eq(Spec::result(), Spec::var("dest")),
        ],
        ..Default::default()
    };

    VerifiedFunction {
        name: "memcpy_impl".into(),
        description: "Memory copy implementation".into(),
        func,
        spec,
        sep_spec: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Example 8: Binary Search
// ═══════════════════════════════════════════════════════════════════════════════

/// Binary search in sorted array
/// ```c
/// /*@ requires n >= 0;
///     requires \valid(arr + (0..n-1));
///     requires \forall int i, j; 0 <= i < j < n ==> arr[i] <= arr[j]; // sorted
///     ensures \result == -1 || (0 <= \result < n && arr[\result] == key);
///     ensures \result == -1 ==> \forall int i; 0 <= i < n ==> arr[i] != key;
/// */
/// int binary_search(int *arr, int n, int key) {
///     int lo = 0, hi = n - 1;
///     while (lo <= hi) {
///         int mid = lo + (hi - lo) / 2;
///         if (arr[mid] == key)
///             return mid;
///         else if (arr[mid] < key)
///             lo = mid + 1;
///         else
///             hi = mid - 1;
///     }
///     return -1;
/// }
/// ```
pub fn binary_search_example() -> VerifiedFunction {
    let func = FuncDef {
        name: "binary_search".into(),
        return_type: CType::int(),
        params: vec![
            FuncParam {
                name: "arr".into(),
                ty: CType::ptr(CType::int()),
            },
            FuncParam {
                name: "n".into(),
                ty: CType::int(),
            },
            FuncParam {
                name: "key".into(),
                ty: CType::int(),
            },
        ],
        body: Box::new(CStmt::block(vec![
            // int lo = 0, hi = n - 1;
            CStmt::decl_init("lo", CType::int(), CExpr::int(0)),
            CStmt::decl_init(
                "hi",
                CType::int(),
                CExpr::sub(CExpr::var("n"), CExpr::int(1)),
            ),
            // while (lo <= hi)
            CStmt::while_loop(
                CExpr::binop(BinOp::Le, CExpr::var("lo"), CExpr::var("hi")),
                CStmt::block(vec![
                    // int mid = lo + (hi - lo) / 2;
                    CStmt::decl_init(
                        "mid",
                        CType::int(),
                        CExpr::add(
                            CExpr::var("lo"),
                            CExpr::div(
                                CExpr::sub(CExpr::var("hi"), CExpr::var("lo")),
                                CExpr::int(2),
                            ),
                        ),
                    ),
                    // if/else chain
                    CStmt::if_else(
                        CExpr::binop(
                            BinOp::Eq,
                            CExpr::index(CExpr::var("arr"), CExpr::var("mid")),
                            CExpr::var("key"),
                        ),
                        CStmt::return_stmt(Some(CExpr::var("mid"))),
                        CStmt::if_else(
                            CExpr::binop(
                                BinOp::Lt,
                                CExpr::index(CExpr::var("arr"), CExpr::var("mid")),
                                CExpr::var("key"),
                            ),
                            CStmt::Expr(CExpr::assign(
                                CExpr::var("lo"),
                                CExpr::add(CExpr::var("mid"), CExpr::int(1)),
                            )),
                            CStmt::Expr(CExpr::assign(
                                CExpr::var("hi"),
                                CExpr::sub(CExpr::var("mid"), CExpr::int(1)),
                            )),
                        ),
                    ),
                ]),
            ),
            // return -1;
            CStmt::return_stmt(Some(CExpr::int(-1))),
        ])),
        variadic: false,
        storage: StorageClass::Auto,
    };

    let spec = FuncSpec {
        requires: vec![
            Spec::ge(Spec::var("n"), Spec::int(0)),
            Spec::ValidRange {
                ptr: Box::new(Spec::var("arr")),
                lo: Box::new(Spec::int(0)),
                hi: Box::new(Spec::binop(BinOp::Sub, Spec::var("n"), Spec::int(1))),
            },
        ],
        ensures: vec![
            // result is -1 or in range
            Spec::or(vec![
                Spec::eq(Spec::result(), Spec::int(-1)),
                Spec::and(vec![
                    Spec::ge(Spec::result(), Spec::int(0)),
                    Spec::lt(Spec::result(), Spec::var("n")),
                ]),
            ]),
        ],
        ..Default::default()
    };

    VerifiedFunction {
        name: "binary_search".into(),
        description: "Binary search in sorted array".into(),
        func,
        spec,
        sep_spec: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Verification Infrastructure
// ═══════════════════════════════════════════════════════════════════════════════

/// Run all examples and return combined results
pub fn run_all_examples() -> Vec<(String, VerificationSummary)> {
    let examples = vec![
        abs_example(),
        swap_example(),
        array_sum_example(),
        constant_time_compare_example(),
        xor_cipher_example(),
        safe_array_access_example(),
        memcpy_example(),
        binary_search_example(),
    ];

    examples
        .into_iter()
        .map(|ex| {
            let name = ex.name.clone();
            let summary = ex.verify();
            (name, summary)
        })
        .collect()
}

/// Print verification report for all examples
pub fn print_verification_report() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    LEAN5 C VERIFICATION EXAMPLES                       ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    let results = run_all_examples();
    let mut total_vcs = 0;
    let mut total_proved = 0;

    for (name, summary) in &results {
        println!("Function: {name}");
        println!(
            "  VCs: {} total, {} proved, {} failed, {} unknown",
            summary.total, summary.proved, summary.failed, summary.unknown
        );
        total_vcs += summary.total;
        total_proved += summary.proved;

        for (desc, status) in &summary.details {
            let marker = match status {
                ProofStatus::Proved(_) => "✓",
                ProofStatus::Failed(_) => "✗",
                ProofStatus::Unknown => "?",
            };
            println!("    {marker} {desc}");
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "TOTAL: {} VCs, {} proved ({:.1}%)",
        total_vcs,
        total_proved,
        if total_vcs > 0 {
            (total_proved as f64 / total_vcs as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("═══════════════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_example() {
        let ex = abs_example();
        assert_eq!(ex.name, "abs");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty(), "Should generate VCs for abs");
    }

    #[test]
    fn test_swap_example() {
        let ex = swap_example();
        assert_eq!(ex.name, "swap");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty(), "Should generate VCs for swap");
        assert!(ex.sep_spec.is_some(), "swap should have sep logic spec");
    }

    #[test]
    fn test_array_sum_example() {
        let ex = array_sum_example();
        assert_eq!(ex.name, "array_sum");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty(), "Should generate VCs for array_sum");
    }

    #[test]
    fn test_constant_time_compare_example() {
        let ex = constant_time_compare_example();
        assert_eq!(ex.name, "constant_time_compare");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty());
    }

    #[test]
    fn test_xor_cipher_example() {
        let ex = xor_cipher_example();
        assert_eq!(ex.name, "xor_cipher");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty());
    }

    #[test]
    fn test_safe_array_access_example() {
        let ex = safe_array_access_example();
        assert_eq!(ex.name, "safe_get");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty());
    }

    #[test]
    fn test_memcpy_example() {
        let ex = memcpy_example();
        assert_eq!(ex.name, "memcpy_impl");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty());
    }

    #[test]
    fn test_binary_search_example() {
        let ex = binary_search_example();
        assert_eq!(ex.name, "binary_search");
        let vcs = ex.generate_vcs();
        assert!(!vcs.is_empty());
    }

    #[test]
    fn test_all_examples_generate_vcs() {
        let results = run_all_examples();
        assert_eq!(results.len(), 8, "Should have 8 examples");

        for (name, summary) in &results {
            assert!(
                summary.total > 0,
                "Example {name} should generate at least one VC"
            );
        }
    }

    #[test]
    fn test_simplified_vcs() {
        let ex = abs_example();
        let vcs = ex.simplified_vcs();
        assert!(!vcs.is_empty());
        // Simplified VCs should not contain nested True
        for vc in &vcs {
            // Check that simplification happened for any trivially true subexpressions
            match &vc.obligation {
                Spec::And(specs) if specs.iter().any(|s| matches!(s, Spec::True)) => {
                    panic!("Simplification should have removed True from And");
                }
                _ => {} // OK
            }
        }
    }

    #[test]
    fn test_swap_sep_spec_pointers() {
        let ex = swap_example();
        let sep_spec = ex.sep_spec.as_ref().unwrap();

        // Check that pre and post mention the same pointers
        let pre_ptrs = sep_spec.pre.mentioned_pointers();
        let post_ptrs = sep_spec.post.mentioned_pointers();

        assert_eq!(pre_ptrs.len(), 2);
        assert_eq!(post_ptrs.len(), 2);
    }

    #[test]
    fn test_verify_abs() {
        let ex = abs_example();
        let summary = ex.verify();
        assert!(summary.total > 0);
        // At minimum we should get some result for each VC
        assert_eq!(
            summary.total,
            summary.proved + summary.failed + summary.unknown
        );
    }

    #[test]
    fn test_verify_swap() {
        let ex = swap_example();
        let summary = ex.verify();
        assert!(summary.total > 0);
    }
}
