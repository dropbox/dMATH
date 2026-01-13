//! Goal management tactics
//!
//! Provides tactics for manipulating the goal list: swapping, rotating,
//! and selecting specific goals to focus on.

use super::{ProofState, TacticError, TacticResult};

/// Swap the first two goals in the goal list.
///
/// This is useful when you want to work on the second goal before the first.
///
/// # Example
/// ```text
/// -- Goals: [⊢ A, ⊢ B, ⊢ C]
/// swap
/// -- Goals: [⊢ B, ⊢ A, ⊢ C]
/// ```
pub fn swap(state: &mut ProofState) -> TacticResult {
    if state.goals.len() < 2 {
        return Err(TacticError::Other(
            "swap: need at least 2 goals".to_string(),
        ));
    }
    state.goals.swap(0, 1);
    Ok(())
}

/// Rotate goals by moving the first goal to the end.
///
/// # Example
/// ```text
/// -- Goals: [⊢ A, ⊢ B, ⊢ C]
/// rotate
/// -- Goals: [⊢ B, ⊢ C, ⊢ A]
/// ```
pub fn rotate(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }
    if state.goals.len() > 1 {
        let first = state.goals.remove(0);
        state.goals.push(first);
    }
    Ok(())
}

/// Rotate goals backward by moving the last goal to the front.
///
/// # Example
/// ```text
/// -- Goals: [⊢ A, ⊢ B, ⊢ C]
/// rotate_back
/// -- Goals: [⊢ C, ⊢ A, ⊢ B]
/// ```
pub fn rotate_back(state: &mut ProofState) -> TacticResult {
    if state.goals.is_empty() {
        return Err(TacticError::NoGoals);
    }
    if state.goals.len() > 1 {
        let last = state.goals.pop().expect("goals has more than 1 element");
        state.goals.insert(0, last);
    }
    Ok(())
}

/// Pick and focus on a specific goal by index (0-based).
///
/// Moves the specified goal to the front of the goal list.
///
/// # Example
/// ```text
/// -- Goals: [⊢ A, ⊢ B, ⊢ C]
/// pick_goal 2
/// -- Goals: [⊢ C, ⊢ A, ⊢ B]
/// ```
pub fn pick_goal(state: &mut ProofState, index: usize) -> TacticResult {
    if index >= state.goals.len() {
        return Err(TacticError::Other(format!(
            "pick_goal: index {} out of bounds (have {} goals)",
            index,
            state.goals.len()
        )));
    }
    if index == 0 {
        return Ok(()); // Already at front
    }
    let goal = state.goals.remove(index);
    state.goals.insert(0, goal);
    Ok(())
}

/// Get the number of remaining goals.
pub fn goal_count(state: &ProofState) -> usize {
    state.goals.len()
}
