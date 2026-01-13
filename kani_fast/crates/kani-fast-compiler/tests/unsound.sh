# shellcheck shell=bash
# Test definitions for category: unsound
# Sourced by test_driver.sh; relies on shared helpers like run_test.
#
# ============================================================================
# REMAINING SOUNDNESS BUGS (7 total)
# ============================================================================
# These tests expose soundness bugs that are NOT YET FIXED.
# "unsound" means the assertion PASSES when it should FAIL.
#
# Categories:
# - UNSOUND_PASS: Still unsound (3 tests)
# - UNSOUND_TIMEOUT: Possibly fixed but times out (4 tests)
# ============================================================================


# ============================================================================
# FIXED TESTS - moved to features.sh (23 tests)
# ============================================================================
# The following tests were fixed and moved to features.sh as of #576:
# - array_mut_soundness → array_mut_fixed
# - box_allocation → box_allocation_fixed
# - range_slices_soundness → range_slices_fixed
# - matches_macro_soundness → matches_macro_fixed
# - enum_extract_soundness → enum_extract_fixed
# - rotate_right_soundness → rotate_right_fixed
# - leading_zeros_soundness → leading_zeros_fixed
# - trailing_zeros_soundness → trailing_zeros_fixed
# - swap_bytes_soundness → swap_bytes_fixed
# - reverse_bits_soundness → reverse_bits_fixed
# - rotate_left_intrinsic_soundness → rotate_left_fixed
# - deep_struct_access_soundness → deep_struct_fixed
# - const_generics_soundness → const_generics_fixed
# - move_closure_soundness → move_closure_fixed
# - closure_basic_soundness → closure_basic_fixed
# - ref_mut_pattern_soundness → ref_mut_fixed
# - match_negative_soundness → match_negative_fixed
# Added in #577:
# - count_ones_soundness → count_ones_fixed
# - lifetime_annotation_soundness → lifetime_annotation_fixed
# - array_find_soundness → array_find_fixed
# - ref_pattern_soundness → ref_pattern_fixed
# - full_slice_soundness → full_slice_fixed
# - assert_ne_soundness → assert_ne_fixed
# ============================================================================


# ============================================================================
# UNSOUND_TIMEOUT TESTS (4 tests) - may be fixed but timeout
# ============================================================================

# Test 265b: Binary search soundness (TIMEOUT)
# Purpose: Exposes soundness bug in binary search position.
# Category: unsound
# Status: May be fixed but times out due to complex loop + array reference params.
cat > "$TMPDIR/binary_search_pos_soundness.rs" << 'ENDOFFILE'
struct SearchResult {
    found: bool,
    pos: usize,
}

fn binary_search(arr: &[i32; 8], target: i32) -> SearchResult {
    let mut left = 0usize;
    let mut right = 7usize;

    while left <= right {
        let mid = left + (right - left) / 2;
        if arr[mid] == target {
            return SearchResult { found: true, pos: mid };
        }
        if arr[mid] < target {
            left = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            right = mid - 1;
        }
    }
    SearchResult { found: false, pos: left }
}

fn binary_search_pos_soundness_proof() {
    let arr = [1, 3, 5, 7, 9, 11, 13, 15];
    let result = binary_search(&arr, 7);
    assert!(result.pos == 0);  // MUST fail: pos is 3
}
ENDOFFILE
run_test "binary_search_pos_soundness" "$TMPDIR/binary_search_pos_soundness.rs" "unsound" "unsound"


# Test 267b: Array reverse soundness (TIMEOUT)
# Purpose: Exposes soundness bug in array reverse operation.
# Category: unsound
# Status: May be fixed but times out due to loop + array reference params.
cat > "$TMPDIR/array_reverse_soundness.rs" << 'ENDOFFILE'
fn reverse_array(arr: &mut [i32; 4]) {
    let mut i = 0usize;
    let mut j = 3usize;
    while i < j {
        let tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
        i += 1;
        j -= 1;
    }
}

fn array_reverse_soundness_proof() {
    let mut arr = [1, 2, 3, 4];
    reverse_array(&mut arr);
    assert!(arr[0] == 1);  // MUST fail: arr[0] is 4
}
ENDOFFILE
run_test "array_reverse_soundness" "$TMPDIR/array_reverse_soundness.rs" "unsound" "unsound"


# Test 279b: Count occur soundness (TIMEOUT)
# Purpose: Exposes soundness bug - array/slice ops unconstrained.
# Category: unsound
# Status: May be fixed but times out due to loop + array reference params.
cat > "$TMPDIR/count_occur_soundness.rs" << 'ENDOFFILE'
fn count_occurrences(arr: &[i32; 6], target: i32) -> i32 {
    let mut count = 0i32;
    let mut i = 0usize;
    while i < 6 {
        if arr[i] == target {
            count += 1;
        }
        i += 1;
    }
    count
}

fn count_occur_soundness_proof() {
    let arr = [1, 2, 3, 2, 4, 2];
    assert!(count_occurrences(&arr, 2) == 1);  // MUST fail: count is 3
}
ENDOFFILE
run_test "count_occur_soundness" "$TMPDIR/count_occur_soundness.rs" "unsound" "unsound"


# Test 106b: While let with match guard soundness (TIMEOUT)
# Purpose: Exposes soundness bug - match guard loop values unconstrained.
# Category: unsound
# Status: May be fixed but times out due to complex loop structure.
cat > "$TMPDIR/while_let_soundness_2.rs" << 'ENDOFFILE'
fn while_let_soundness_proof() {
    let mut x: Option<i32> = Some(3);
    let mut sum = 0i32;

    loop {
        match x {
            Some(n) if n > 0 => {
                sum += n;
                x = Some(n - 1);
            }
            _ => break,
        }
    }
    assert!(sum == 5);  // UNSOUND: passes when should fail - sum unconstrained
}
ENDOFFILE
run_test "while_let_soundness_2" "$TMPDIR/while_let_soundness_2.rs" "unsound" "unsound"


# ============================================================================
# UNSOUND_PASS TESTS (3 tests) - still truly unsound
# ============================================================================

# Test 214b: Empty array soundness
# Purpose: Exposes Z4 PDR soundness bug where it incorrectly finds invariant.
# Category: unsound
# Note: The encoding is correct (len=0), but Z4 PDR returns "Safe" for
#       CHC problems where the error state IS reachable. Same CHC returns
#       unsat (correct) in Z3. This is a documented Z4 soundness bug.
cat > "$TMPDIR/empty_array_soundness.rs" << 'ENDOFFILE'
fn empty_array_soundness_proof() {
    let arr: [i32; 0] = [];
    assert!(arr.len() == 5);  // MUST fail: len is 0
}
ENDOFFILE
run_test "empty_array_soundness" "$TMPDIR/empty_array_soundness.rs" "unsound" "unsound"


# Test 266b: Array palindrome soundness
# Purpose: Exposes soundness bug in array reference parameter access.
# Category: unsound
# Note: Array index access through reference param returns unconstrained.
cat > "$TMPDIR/array_palindrome_soundness.rs" << 'ENDOFFILE'
fn is_array_palindrome(arr: &[i32; 5]) -> bool {
    arr[0] == arr[4] && arr[1] == arr[3]
}

fn array_palindrome_soundness_proof() {
    let arr = [1, 2, 3, 4, 5];
    assert!(is_array_palindrome(&arr));  // MUST fail: not a palindrome
}
ENDOFFILE
run_test "array_palindrome_soundness" "$TMPDIR/array_palindrome_soundness.rs" "unsound" "unsound"


# Test 270b: Weighted avg soundness
# Purpose: Exposes soundness bug - array reference parameter access unconstrained.
# Category: unsound
# Note: Array index access through reference param returns unconstrained.
cat > "$TMPDIR/weighted_avg_soundness.rs" << 'ENDOFFILE'
fn weighted_sum(values: &[i32; 4], weights: &[i32; 4]) -> i32 {
    values[0] * weights[0] + values[1] * weights[1] + values[2] * weights[2] + values[3] * weights[3]
}

fn weighted_avg_soundness_proof() {
    let vals = [10, 10, 10, 10];
    let weights = [1, 2, 3, 4];
    assert!(weighted_sum(&vals, &weights) == 50);  // MUST fail: sum is 100
}
ENDOFFILE
run_test "weighted_avg_soundness" "$TMPDIR/weighted_avg_soundness.rs" "unsound" "unsound"


