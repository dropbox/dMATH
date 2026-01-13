//! VSIDS variable selection heuristic with heap-based O(log n) selection
//!
//! This module provides VSIDS (Variable State Independent Decaying Sum) with
//! a binary max-heap for efficient variable selection. The heap maintains
//! unassigned variables ordered by activity, enabling O(log n) decisions
//! instead of O(n) linear scans.
//!
//! Design based on CaDiCaL's heap.hpp - maintains position mapping for
//! efficient update operations (decrease-key/increase-key).

use crate::literal::Variable;

/// Invalid heap position marker (variable not in heap)
const INVALID_POS: u32 = u32::MAX;
/// Invalid variable marker (used for VMTF linked-list pointers)
const INVALID_VAR: u32 = u32::MAX;

/// VSIDS activity-based variable selection with heap
#[derive(Debug)]
pub struct VSIDS {
    /// Activity scores for each variable
    activities: Vec<f64>,
    /// Activity increment (grows on each decay)
    increment: f64,
    /// Decay factor
    decay: f64,
    /// Random seed for tie-breaking (affects initial perturbation)
    random_seed: u64,
    /// Bump order for each variable - higher means more recently bumped (VMTF)
    bump_order: Vec<u64>,
    /// Counter for bump ordering
    bump_counter: u64,

    // Heap data structures
    /// Binary max-heap of variable indices (ordered by activity)
    heap: Vec<u32>,
    /// Position of each variable in heap (INVALID_POS if not in heap)
    heap_pos: Vec<u32>,

    // VMTF decision queue (focused mode)
    /// Previous variable in bump-order list (towards older variables)
    vmtf_prev: Vec<u32>,
    /// Next variable in bump-order list (towards more recent variables)
    vmtf_next: Vec<u32>,
    /// Oldest variable in the queue
    vmtf_first: u32,
    /// Most recently bumped variable in the queue (front)
    vmtf_last: u32,
    /// Most recently bumped *unassigned* variable (used as starting point)
    vmtf_unassigned: u32,
    /// Bump timestamp of `vmtf_unassigned` at last update
    vmtf_unassigned_bumped: u64,
}

impl VSIDS {
    /// Create a new VSIDS with n variables
    pub fn new(num_vars: usize) -> Self {
        // Initialize bump_order so variables with lower indices are tried first initially
        // (same as CaDiCaL's init_queue)
        let mut bump_order = Vec::with_capacity(num_vars);
        for i in 0..num_vars {
            // Lower index = higher initial bump order (will be picked first)
            bump_order.push((num_vars - i) as u64);
        }

        // Initialize heap with all variables (all unassigned initially)
        // Variables are ordered by index initially (lower index = higher priority)
        let mut heap = Vec::with_capacity(num_vars);
        let mut heap_pos = vec![INVALID_POS; num_vars];
        for (i, pos) in heap_pos.iter_mut().enumerate().take(num_vars) {
            heap.push(i as u32);
            *pos = i as u32;
        }

        // Note: With zero initial activities, heap order doesn't matter much
        // The heap will reorganize as variables get bumped during solving

        // Initialize VMTF queue in index order, preferring smaller indices first.
        // The queue is a doubly linked list ordered by bump-recency where
        // `vmtf_last` is the most recently bumped (front).
        //
        // To pick smaller indices first initially, we build the initial order as:
        // (num_vars - 1) (oldest) -> ... -> 1 -> 0 (newest).
        let (vmtf_prev, vmtf_next, vmtf_first, vmtf_last, vmtf_unassigned, vmtf_unassigned_bumped) =
            if num_vars == 0 {
                (
                    Vec::new(),
                    Vec::new(),
                    INVALID_VAR,
                    INVALID_VAR,
                    INVALID_VAR,
                    0u64,
                )
            } else {
                let mut vmtf_prev = vec![INVALID_VAR; num_vars];
                let mut vmtf_next = vec![INVALID_VAR; num_vars];
                for i in 0..num_vars {
                    vmtf_prev[i] = if i + 1 < num_vars {
                        (i + 1) as u32
                    } else {
                        INVALID_VAR
                    };
                    vmtf_next[i] = if i > 0 { (i - 1) as u32 } else { INVALID_VAR };
                }
                let vmtf_first = (num_vars as u32) - 1;
                let vmtf_last = 0;
                let vmtf_unassigned = vmtf_last;
                let vmtf_unassigned_bumped = bump_order[vmtf_unassigned as usize];
                (
                    vmtf_prev,
                    vmtf_next,
                    vmtf_first,
                    vmtf_last,
                    vmtf_unassigned,
                    vmtf_unassigned_bumped,
                )
            };

        VSIDS {
            activities: vec![0.0; num_vars],
            increment: 1.0,
            decay: 0.95,
            random_seed: 0,
            bump_order,
            bump_counter: num_vars as u64 + 1,
            heap,
            heap_pos,
            vmtf_prev,
            vmtf_next,
            vmtf_first,
            vmtf_last,
            vmtf_unassigned,
            vmtf_unassigned_bumped,
        }
    }

    /// Ensure VSIDS has storage for `num_vars` variables.
    pub fn ensure_num_vars(&mut self, num_vars: usize) {
        let old_len = self.activities.len();
        if old_len < num_vars {
            self.activities.resize(num_vars, 0.0);
            self.heap_pos.resize(num_vars, INVALID_POS);
            self.vmtf_prev.resize(num_vars, INVALID_VAR);
            self.vmtf_next.resize(num_vars, INVALID_VAR);

            // New variables get increasing bump order
            for _ in old_len..num_vars {
                self.bump_order.push(self.bump_counter);
                self.bump_counter += 1;
            }

            // Add new variables to heap (they are unassigned)
            for i in old_len..num_vars {
                self.push_heap(i as u32);
            }

            // Add new variables to the VMTF queue (most recent/front).
            for i in old_len..num_vars {
                self.vmtf_enqueue(i as u32);
                // New variables are unassigned initially, so treat them as best candidate.
                self.vmtf_unassigned = i as u32;
                self.vmtf_unassigned_bumped = self.bump_order[i];
            }
        }
    }

    /// Bump the activity of a variable
    ///
    /// This updates both the VSIDS activity (for stable mode) and the
    /// VMTF bump order (for focused mode).
    #[inline]
    pub fn bump(&mut self, var: Variable) {
        let idx = var.index();
        // Update VSIDS activity
        self.activities[idx] += self.increment;
        // Rescale if needed
        if self.activities[idx] > 1e100 {
            self.rescale();
        }
        // Update VMTF bump order
        self.bump_order[idx] = self.bump_counter;
        self.bump_counter += 1;

        // Move variable to the front of the VMTF queue (most recently bumped).
        self.vmtf_bump_to_front(var);

        // Update heap position (activity increased, bubble up)
        if self.heap_pos[idx] != INVALID_POS {
            self.sift_up(self.heap_pos[idx] as usize);
        }
    }

    /// Decay all activities
    #[inline]
    pub fn decay(&mut self) {
        self.increment /= self.decay;
    }

    /// Get activity of a variable
    #[inline]
    pub fn activity(&self, var: Variable) -> f64 {
        self.activities[var.index()]
    }

    /// Rescale all activities to prevent overflow
    fn rescale(&mut self) {
        for act in &mut self.activities {
            *act *= 1e-100;
        }
        self.increment *= 1e-100;
        // Note: No need to restructure heap - relative ordering unchanged
    }

    /// Remove a variable from the heap (when assigned)
    #[inline]
    pub fn remove_from_heap(&mut self, var: Variable) {
        let idx = var.index();
        let pos = self.heap_pos[idx];
        if pos == INVALID_POS {
            return; // Already not in heap
        }

        let last_idx = self.heap.len() - 1;
        if pos as usize == last_idx {
            // Removing last element, just pop
            self.heap.pop();
            self.heap_pos[idx] = INVALID_POS;
        } else {
            // Swap with last element and remove
            let last_var = self.heap[last_idx] as usize;
            self.heap[pos as usize] = last_var as u32;
            self.heap_pos[last_var] = pos;
            self.heap.pop();
            self.heap_pos[idx] = INVALID_POS;

            // Restore heap property for the moved element
            self.sift_up(pos as usize);
            self.sift_down(self.heap_pos[last_var] as usize);
        }
    }

    /// Insert a variable into the heap (when unassigned during backtrack)
    #[inline]
    pub fn insert_into_heap(&mut self, var: Variable) {
        let idx = var.index();
        if self.heap_pos[idx] != INVALID_POS {
            return; // Already in heap
        }
        self.push_heap(idx as u32);
    }

    /// Push a variable onto the heap and restore heap property
    #[inline]
    fn push_heap(&mut self, var_idx: u32) {
        let pos = self.heap.len();
        self.heap.push(var_idx);
        self.heap_pos[var_idx as usize] = pos as u32;
        self.sift_up(pos);
    }

    /// Compare two variables for heap ordering
    /// Returns true if var_a should be higher in heap than var_b
    /// Higher activity wins; on ties, lower variable index wins (same as old linear scan)
    #[inline]
    fn var_less(&self, var_a: usize, var_b: usize) -> bool {
        let act_a = self.activities[var_a];
        let act_b = self.activities[var_b];
        act_a > act_b || (act_a == act_b && var_a < var_b)
    }

    /// Sift up an element to restore heap property (after activity increase)
    #[inline]
    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            let var = self.heap[pos] as usize;
            let parent_var = self.heap[parent] as usize;

            // Max-heap: parent should be >= child
            if !self.var_less(var, parent_var) {
                break;
            }

            // Swap with parent
            self.heap[pos] = parent_var as u32;
            self.heap[parent] = var as u32;
            self.heap_pos[var] = parent as u32;
            self.heap_pos[parent_var] = pos as u32;
            pos = parent;
        }
    }

    /// Sift down an element to restore heap property
    #[inline]
    fn sift_down(&mut self, mut pos: usize) {
        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;
            let mut largest = pos;

            let var = self.heap[pos] as usize;

            if left < self.heap.len() {
                let left_var = self.heap[left] as usize;
                if self.var_less(left_var, var) {
                    largest = left;
                }
            }

            if right < self.heap.len() {
                let right_var = self.heap[right] as usize;
                let largest_var = self.heap[largest] as usize;
                if self.var_less(right_var, largest_var) {
                    largest = right;
                }
            }

            if largest == pos {
                break;
            }

            // Swap with largest child
            let largest_var = self.heap[largest] as usize;
            self.heap[pos] = largest_var as u32;
            self.heap[largest] = var as u32;
            self.heap_pos[var] = largest as u32;
            self.heap_pos[largest_var] = pos as u32;
            pos = largest;
        }
    }

    /// Select next variable to branch on using VSIDS (highest activity)
    /// Returns the unassigned variable with highest activity, or None if all assigned
    /// O(1) operation - just returns the top of the heap
    #[inline]
    pub fn pick_branching_variable(&self, _assignment: &[Option<bool>]) -> Option<Variable> {
        // The heap only contains unassigned variables, so top is always valid
        if self.heap.is_empty() {
            None
        } else {
            Some(Variable(self.heap[0]))
        }
    }

    /// Select next variable to branch on using VMTF (most recently bumped)
    /// Returns the unassigned variable with highest bump order, or None if all assigned
    /// Use this in focused mode for more aggressive exploration.
    /// Uses a doubly-linked VMTF queue with an "unassigned" cursor (CaDiCaL style).
    #[inline]
    pub fn pick_branching_variable_vmtf(
        &mut self,
        assignment: &[Option<bool>],
    ) -> Option<Variable> {
        if self.vmtf_unassigned == INVALID_VAR {
            return None;
        }

        let mut idx = self.vmtf_unassigned;
        let mut searched = 0u64;

        while idx != INVALID_VAR && assignment[idx as usize].is_some() {
            idx = self.vmtf_prev[idx as usize];
            searched += 1;
        }

        if idx == INVALID_VAR {
            return None;
        }

        if searched > 0 {
            self.vmtf_unassigned = idx;
            self.vmtf_unassigned_bumped = self.bump_order[idx as usize];
        }

        Some(Variable(idx))
    }

    /// Get the bump order for a variable (for trail reuse comparison)
    #[inline]
    pub fn bump_order(&self, var: Variable) -> u64 {
        self.bump_order[var.index()]
    }

    /// Notify the VMTF queue that `var` became unassigned (during backtracking).
    ///
    /// This updates the "unassigned cursor" if this variable is more recently bumped
    /// than the current cursor (CaDiCaL's `update_queue_unassigned` logic).
    #[inline]
    pub fn vmtf_on_unassign(&mut self, var: Variable) {
        let idx = var.index();
        let order = self.bump_order[idx];
        if self.vmtf_unassigned == INVALID_VAR || order > self.vmtf_unassigned_bumped {
            self.vmtf_unassigned = idx as u32;
            self.vmtf_unassigned_bumped = order;
        }
    }

    /// Reset VMTF cursor assuming all variables are unassigned.
    #[inline]
    pub fn reset_vmtf_unassigned(&mut self) {
        if self.vmtf_last == INVALID_VAR {
            self.vmtf_unassigned = INVALID_VAR;
            self.vmtf_unassigned_bumped = 0;
        } else {
            self.vmtf_unassigned = self.vmtf_last;
            self.vmtf_unassigned_bumped = self.bump_order[self.vmtf_unassigned as usize];
        }
    }

    /// Get the capacity of the activities vector (for memory statistics)
    pub fn capacity(&self) -> usize {
        self.activities.capacity()
    }

    /// Set the random seed for tie-breaking
    ///
    /// This adds a small perturbation to activity scores to break ties
    /// differently for different solver configurations.
    pub fn set_random_seed(&mut self, seed: u64) {
        self.random_seed = seed;
        // Add small perturbations based on seed to break ties
        // The perturbations are tiny (1e-10 scale) so they only affect ties
        if seed != 0 {
            let mut state = seed;
            for i in 0..self.activities.len() {
                // Simple xorshift for pseudo-random perturbation
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Add tiny perturbation (won't affect normal activity comparisons)
                self.activities[i] += (state as f64) * 1e-15;
            }
            // Rebuild heap after perturbation
            self.rebuild_heap();
        }
    }

    /// Rebuild the heap from scratch (used after bulk activity changes)
    fn rebuild_heap(&mut self) {
        // Heapify from bottom up
        if self.heap.len() <= 1 {
            return;
        }
        for i in (0..self.heap.len() / 2).rev() {
            self.sift_down(i);
        }
    }

    /// Get the random seed
    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }

    /// Reset the heap to contain all variables (called on solver reset)
    ///
    /// This ensures all variables are in the heap when the solver state is reset.
    /// Variables retain their activity scores for search continuity.
    pub fn reset_heap(&mut self) {
        let num_vars = self.activities.len();
        self.heap.clear();
        self.heap_pos.fill(INVALID_POS);

        // Add all variables to heap
        for i in 0..num_vars {
            self.heap.push(i as u32);
            self.heap_pos[i] = i as u32;
        }

        // Rebuild heap to establish correct ordering based on activities
        self.rebuild_heap();
    }

    #[inline]
    fn vmtf_bump_to_front(&mut self, var: Variable) {
        let idx = var.0;
        if idx == INVALID_VAR || idx == self.vmtf_last {
            return;
        }
        self.vmtf_dequeue(idx);
        self.vmtf_enqueue(idx);
    }

    #[inline]
    fn vmtf_dequeue(&mut self, idx: u32) {
        let prev = self.vmtf_prev[idx as usize];
        let next = self.vmtf_next[idx as usize];
        if prev != INVALID_VAR {
            self.vmtf_next[prev as usize] = next;
        } else {
            self.vmtf_first = next;
        }
        if next != INVALID_VAR {
            self.vmtf_prev[next as usize] = prev;
        } else {
            self.vmtf_last = prev;
        }
        self.vmtf_prev[idx as usize] = INVALID_VAR;
        self.vmtf_next[idx as usize] = INVALID_VAR;
    }

    #[inline]
    fn vmtf_enqueue(&mut self, idx: u32) {
        if self.vmtf_last == INVALID_VAR {
            self.vmtf_first = idx;
            self.vmtf_last = idx;
            self.vmtf_prev[idx as usize] = INVALID_VAR;
            self.vmtf_next[idx as usize] = INVALID_VAR;
            return;
        }
        let last = self.vmtf_last;
        self.vmtf_prev[idx as usize] = last;
        self.vmtf_next[idx as usize] = INVALID_VAR;
        self.vmtf_next[last as usize] = idx;
        self.vmtf_last = idx;
    }

    /// Check heap invariant (for debugging)
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    fn check_heap(&self) -> bool {
        for i in 0..self.heap.len() {
            let var = self.heap[i] as usize;
            if self.heap_pos[var] != i as u32 {
                return false;
            }
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < self.heap.len() {
                let left_var = self.heap[left] as usize;
                // Child should not be "better" than parent
                if self.var_less(left_var, var) {
                    return false;
                }
            }
            if right < self.heap.len() {
                let right_var = self.heap[right] as usize;
                if self.var_less(right_var, var) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_operations() {
        let mut vsids = VSIDS::new(5);
        let assignment = vec![None; 5];

        // Initially all variables in heap, some var should be picked (all equal activity)
        assert!(vsids.pick_branching_variable(&assignment).is_some());

        // Bump variable 3 twice - it should become the top with activity 2.0
        vsids.bump(Variable(3));
        vsids.bump(Variable(3));
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(3))
        );

        // Remove variable 3 from heap (assigned)
        vsids.remove_from_heap(Variable(3));
        // Now should pick something else
        let picked = vsids.pick_branching_variable(&assignment);
        assert!(picked.is_some());
        assert_ne!(picked, Some(Variable(3)));

        // Bump variable 2 once - activity 1.0
        vsids.bump(Variable(2));
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(2))
        );

        // Insert variable 3 back (unassigned)
        vsids.insert_into_heap(Variable(3));
        // Variable 3 has activity 2.0, var 2 has 1.0 - var 3 should be top
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(3))
        );
    }

    #[test]
    fn test_heap_empty() {
        let mut vsids = VSIDS::new(3);
        let assignment = vec![None; 3];

        // Remove all variables
        vsids.remove_from_heap(Variable(0));
        vsids.remove_from_heap(Variable(1));
        vsids.remove_from_heap(Variable(2));

        assert_eq!(vsids.pick_branching_variable(&assignment), None);

        // Insert one back
        vsids.insert_into_heap(Variable(1));
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(1))
        );
    }

    #[test]
    fn test_heap_activity_ordering() {
        let mut vsids = VSIDS::new(5);
        let assignment = vec![None; 5];

        // Bump each variable different number of times to create distinct activities
        // var 0: 5 bumps, var 1: 4 bumps, var 2: 3 bumps, var 3: 2 bumps, var 4: 1 bump
        for i in 0..5u32 {
            for _ in 0..(5 - i) {
                vsids.bump(Variable(i));
            }
        }

        // Variable 0 has highest activity (5.0)
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(0))
        );
        vsids.remove_from_heap(Variable(0));

        // Then var 1 (4.0)
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(1))
        );
        vsids.remove_from_heap(Variable(1));

        // Then var 2 (3.0)
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(2))
        );
    }

    #[test]
    fn test_ensure_num_vars() {
        let mut vsids = VSIDS::new(3);
        vsids.ensure_num_vars(5);

        let assignment = vec![None; 5];
        // New variables should be in heap
        // Bump var 4 to make it top
        vsids.bump(Variable(4));
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(4))
        );
    }

    #[test]
    fn test_heap_bump_while_assigned() {
        let mut vsids = VSIDS::new(3);
        let assignment = vec![None; 3];

        // Bump var 0 and remove it (assigned)
        vsids.bump(Variable(0));
        vsids.remove_from_heap(Variable(0));

        // Bump var 0 again while it's assigned (this can happen during conflict analysis)
        vsids.bump(Variable(0));
        vsids.bump(Variable(0));
        // var 0 now has activity 3.0

        // var 1 has activity 0, so it should be the top of remaining heap
        // (or var 2, depending on initial order)
        let picked = vsids.pick_branching_variable(&assignment);
        assert!(picked.is_some());
        assert_ne!(picked, Some(Variable(0))); // var 0 is not in heap

        // Insert var 0 back
        vsids.insert_into_heap(Variable(0));
        // Now var 0 with activity 3.0 should be top
        assert_eq!(
            vsids.pick_branching_variable(&assignment),
            Some(Variable(0))
        );
    }

    #[test]
    fn test_vmtf_cursor_skips_assigned() {
        let mut vsids = VSIDS::new(3);
        let assignment = vec![Some(true), Some(false), None];
        assert_eq!(
            vsids.pick_branching_variable_vmtf(&assignment),
            Some(Variable(2))
        );
        // Cursor should now be updated to 2 (the found unassigned variable).
        assert_eq!(
            vsids.pick_branching_variable_vmtf(&assignment),
            Some(Variable(2))
        );
    }

    #[test]
    fn test_vmtf_updates_on_unassign_after_bump() {
        let mut vsids = VSIDS::new(4);

        // Simulate a conflict bumping var 2 while it is assigned.
        vsids.bump(Variable(2));

        // Now simulate backtracking that unassigns var 2.
        vsids.vmtf_on_unassign(Variable(2));

        let assignment = vec![None, None, None, None];
        assert_eq!(
            vsids.pick_branching_variable_vmtf(&assignment),
            Some(Variable(2))
        );
    }
}
