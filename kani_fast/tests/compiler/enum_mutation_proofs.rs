// Test CHC encoding for enum mutation (SetDiscriminant)
// These tests verify that changing enum variants works correctly

/// Simple enum with unit variants
enum Color {
    Red,
    Green,
    Blue,
}

/// Enum with data in one variant
enum Container {
    Empty,
    Value(i32),
}

/// Test mutating a unit enum - should verify
fn enum_mutate_unit_proof() {
    let mut c = Color::Red;
    // Mutate to a different variant
    c = Color::Blue;
    let is_blue = matches!(c, Color::Blue);
    assert!(is_blue);
}

/// Test mutating enum with data to empty variant - should verify
fn enum_to_empty_proof() {
    let mut container = Container::Value(42);
    // Mutate to Empty variant
    container = Container::Empty;
    let is_empty = matches!(container, Container::Empty);
    assert!(is_empty);
}

/// Test mutating enum from empty to data variant - should verify
fn enum_to_value_proof() {
    let mut container = Container::Empty;
    // Mutate to Value variant
    container = Container::Value(100);
    if let Container::Value(x) = container {
        assert!(x == 100);
    } else {
        assert!(false);
    }
}

/// Test mutating enum multiple times - should verify
fn enum_multiple_mutations_proof() {
    let mut c = Color::Red;
    c = Color::Green;
    c = Color::Blue;
    c = Color::Red;
    let is_red = matches!(c, Color::Red);
    assert!(is_red);
}

/// Test wrong assertion after mutation - should fail
fn enum_mutation_wrong_assertion_proof() {
    let mut c = Color::Red;
    c = Color::Blue;
    // Wrong: c is now Blue, not Red
    let is_red = matches!(c, Color::Red);
    assert!(is_red); // Should fail
}

/// Test data preserved across no-op assignment - should verify
fn enum_data_preserved_proof() {
    let mut container = Container::Value(42);
    // Re-assign with same variant but different value
    container = Container::Value(99);
    if let Container::Value(x) = container {
        assert!(x == 99);
    } else {
        assert!(false);
    }
}

fn main() {}
