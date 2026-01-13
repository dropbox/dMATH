// Test CHC encoding for match expressions
// These tests verify that the CHC encoding correctly handles match statements
// Note: Match compiles to SwitchInt on discriminant, similar to if-let

/// Simple enum for testing match
enum Direction {
    North,
    South,
    East,
    West,
}

/// Match on enum discriminant - should verify
fn match_direction_proof() {
    let dir = Direction::North;
    let dx = match dir {
        Direction::North => 0,
        Direction::South => 0,
        Direction::East => 1,
        Direction::West => -1,
    };
    assert!(dx == 0);
}

/// Match returns different values for each arm - should verify
fn match_south_proof() {
    let dir = Direction::South;
    let dy = match dir {
        Direction::North => 1,
        Direction::South => -1,
        Direction::East => 0,
        Direction::West => 0,
    };
    assert!(dy == -1);
}

/// Match on boolean - should verify
fn match_bool_proof() {
    let flag = true;
    let value = match flag {
        true => 1,
        false => 0,
    };
    assert!(value == 1);
}

/// Match result combined with addition - should verify
fn match_addition_proof() {
    let dir = Direction::East;
    let offset = match dir {
        Direction::North => 0,
        Direction::South => 0,
        Direction::East => 10,
        Direction::West => -10,
    };
    let base = 5;
    // Simple addition (no wrapping/checked arithmetic)
    let result = base + offset;
    assert!(result == 15);
}

/// Match with wrong expected value - should fail
fn match_wrong_assertion_proof() {
    let dir = Direction::South;
    let dy = match dir {
        Direction::North => 1,
        Direction::South => -1,
        Direction::East => 0,
        Direction::West => 0,
    };
    // Wrong: South gives -1, not 1
    assert!(dy == 1);
}

fn main() {}
