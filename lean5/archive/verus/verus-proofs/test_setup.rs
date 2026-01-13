//! Test Verus setup - simple verification example

use vstd::prelude::*;

verus! {

// A simple spec function that returns true
spec fn is_positive(x: int) -> bool {
    x > 0
}

// A proof that 1 is positive
proof fn one_is_positive()
    ensures is_positive(1)
{
    // trivial by definition
}

// A simple exec function with verification
fn checked_add(a: u64, b: u64) -> (result: u64)
    requires a as int + b as int <= u64::MAX as int
    ensures result == a + b
{
    a + b
}

// Test simple math property
proof fn add_commutative(a: int, b: int)
    ensures a + b == b + a
{
    // trivial by math
}

} // verus!

fn main() {
    println!("Verus setup test passed!");
}
