// Benchmark: Bounds validation using simple integer logic
// Kani Fast: CHC proves instantly
// Kani: SAT encoding overhead
//
// Note: This benchmark uses simple integer logic for bounds checking.

fn is_valid_index(len: i32, idx: i32) -> bool {
    idx >= 0 && idx < len
}

fn is_in_range(value: i32, min: i32, max: i32) -> bool {
    value >= min && value <= max
}

fn clamp_to_bounds(value: i32, min: i32, max: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn simple_bounds_check_proof() {
    // Valid index
    assert!(is_valid_index(5, 2));
    assert!(is_valid_index(5, 0));
    assert!(is_valid_index(5, 4));

    // Invalid index
    assert!(!is_valid_index(5, 5));
    assert!(!is_valid_index(5, -1));

    // Range check
    assert!(is_in_range(50, 0, 100));
    assert!(!is_in_range(150, 0, 100));
    assert!(!is_in_range(-10, 0, 100));

    // Clamp
    assert!(clamp_to_bounds(50, 0, 100) == 50);
    assert!(clamp_to_bounds(-10, 0, 100) == 0);
    assert!(clamp_to_bounds(200, 0, 100) == 100);
}
