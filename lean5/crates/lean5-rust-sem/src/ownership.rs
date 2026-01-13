//! Rust Ownership and Borrowing Model
//!
//! This module formalizes Rust's ownership system, which is central to
//! memory safety without garbage collection.
//!
//! ## Core Rules
//!
//! 1. **Ownership**: Each value has exactly one owner
//! 2. **Move Semantics**: When ownership is transferred, the source is invalidated
//! 3. **Borrowing**: References can borrow values temporarily
//! 4. **Borrow Rules**:
//!    - Any number of shared references (&T), OR
//!    - Exactly one mutable reference (&mut T)
//!    - References cannot outlive the referent
//!
//! ## Places and Projections
//!
//! A "place" is a location in memory that can hold a value:
//! - Local variables
//! - Static variables
//! - Heap allocations
//! - Fields of structs
//! - Elements of arrays/vectors

use crate::types::{Lifetime, Mutability, RustType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// A place expression (lvalue) representing a memory location
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Place {
    /// Local variable by index
    Local(u32),
    /// Static variable by name
    Static(String),
    /// Field projection: base.field
    Field { base: Box<Place>, field: String },
    /// Array/slice index: `base[index]`
    Index { base: Box<Place>, index: Box<Place> },
    /// Dereference: *base
    Deref(Box<Place>),
    /// Downcast enum variant: base as Variant
    Downcast { base: Box<Place>, variant: String },
}

impl Place {
    /// Create a local variable place
    pub fn local(index: u32) -> Self {
        Place::Local(index)
    }

    /// Create a field projection
    #[must_use]
    pub fn field(self, name: &str) -> Self {
        Place::Field {
            base: Box::new(self),
            field: name.to_string(),
        }
    }

    /// Create a dereference
    #[must_use]
    pub fn deref(self) -> Self {
        Place::Deref(Box::new(self))
    }

    /// Get the base place (without projections)
    pub fn base(&self) -> &Place {
        match self {
            Place::Local(_) | Place::Static(_) => self,
            Place::Field { base, .. }
            | Place::Index { base, .. }
            | Place::Deref(base)
            | Place::Downcast { base, .. } => base.base(),
        }
    }

    /// Check if this place is a prefix of another
    pub fn is_prefix_of(&self, other: &Place) -> bool {
        if self == other {
            return true;
        }
        match other {
            Place::Field { base, .. }
            | Place::Index { base, .. }
            | Place::Deref(base)
            | Place::Downcast { base, .. } => self.is_prefix_of(base),
            _ => false,
        }
    }

    /// Check if two places conflict (overlap in memory)
    pub fn conflicts_with(&self, other: &Place) -> bool {
        self.is_prefix_of(other) || other.is_prefix_of(self)
    }
}

/// Borrow information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Borrow {
    /// The place being borrowed
    pub place: Place,
    /// Whether the borrow is mutable
    pub mutability: Mutability,
    /// Lifetime of the borrow
    pub lifetime: Lifetime,
    /// Point in the program where borrow was created
    pub origin: u32,
}

/// Ownership state for a single place
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaceState {
    /// Place is owned and valid
    Owned,
    /// Place has been moved out
    Moved,
    /// Place is partially moved (some fields moved)
    PartiallyMoved,
    /// Place is borrowed (immutably)
    SharedBorrowed,
    /// Place is borrowed (mutably)
    MutBorrowed,
    /// Place is uninitialized
    Uninitialized,
}

/// State of all places and borrows at a program point
#[derive(Debug, Clone, Default)]
pub struct OwnershipState {
    /// State of each place
    place_states: HashMap<Place, PlaceState>,
    /// Active borrows
    active_borrows: Vec<Borrow>,
    /// Counter for borrow origins
    borrow_counter: u32,
}

impl OwnershipState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a place as owned
    pub fn mark_owned(&mut self, place: Place) {
        self.place_states.insert(place, PlaceState::Owned);
    }

    /// Mark a place as moved
    pub fn mark_moved(&mut self, place: Place) {
        self.place_states.insert(place, PlaceState::Moved);
    }

    /// Mark a place as uninitialized
    pub fn mark_uninitialized(&mut self, place: Place) {
        self.place_states.insert(place, PlaceState::Uninitialized);
    }

    /// Check if a place is owned
    pub fn is_owned(&self, place: &Place) -> bool {
        matches!(self.place_states.get(place), Some(PlaceState::Owned))
    }

    /// Check if a place is moved
    pub fn is_moved(&self, place: &Place) -> bool {
        matches!(self.place_states.get(place), Some(PlaceState::Moved))
    }

    /// Check if a place is borrowed (either shared or mutable)
    pub fn is_borrowed(&self, place: &Place) -> bool {
        matches!(
            self.place_states.get(place),
            Some(PlaceState::SharedBorrowed | PlaceState::MutBorrowed)
        )
    }

    /// Check if a place is initialized and accessible
    pub fn is_accessible(&self, place: &Place) -> bool {
        matches!(
            self.place_states.get(place),
            Some(PlaceState::Owned | PlaceState::SharedBorrowed)
        )
    }

    /// Add a borrow
    pub fn add_borrow(&mut self, place: Place, mutability: Mutability, lifetime: Lifetime) {
        let borrow = Borrow {
            place: place.clone(),
            mutability,
            lifetime,
            origin: self.borrow_counter,
        };
        self.borrow_counter += 1;

        let new_state = match mutability {
            Mutability::Shared => PlaceState::SharedBorrowed,
            Mutability::Mutable => PlaceState::MutBorrowed,
        };
        self.place_states.insert(place, new_state);
        self.active_borrows.push(borrow);
    }

    /// End borrows with the given lifetime
    pub fn end_borrows(&mut self, lifetime: &Lifetime) {
        let ended: Vec<_> = self
            .active_borrows
            .iter()
            .filter(|b| &b.lifetime == lifetime)
            .map(|b| b.place.clone())
            .collect();

        self.active_borrows.retain(|b| &b.lifetime != lifetime);

        // Restore owned state for places no longer borrowed
        for place in ended {
            if !self.active_borrows.iter().any(|b| b.place == place) {
                self.place_states.insert(place, PlaceState::Owned);
            }
        }
    }

    /// Get all active borrows of a place
    pub fn borrows_of(&self, place: &Place) -> Vec<&Borrow> {
        self.active_borrows
            .iter()
            .filter(|b| b.place.conflicts_with(place))
            .collect()
    }

    /// Check if a mutable borrow is active on a place
    pub fn has_mutable_borrow(&self, place: &Place) -> bool {
        self.active_borrows
            .iter()
            .any(|b| b.mutability == Mutability::Mutable && b.place.conflicts_with(place))
    }

    /// Check if any borrow is active on a place
    pub fn has_any_borrow(&self, place: &Place) -> bool {
        self.active_borrows
            .iter()
            .any(|b| b.place.conflicts_with(place))
    }
}

/// Borrow check errors
#[derive(Debug, Clone, Error)]
pub enum BorrowError {
    #[error("cannot move out of `{place:?}`: value is borrowed")]
    MoveWhileBorrowed { place: Place },

    #[error("cannot borrow `{place:?}` as mutable: already borrowed as immutable")]
    MutBorrowWhileSharedBorrow { place: Place },

    #[error("cannot borrow `{place:?}` as mutable: already borrowed as mutable")]
    MutBorrowWhileMutBorrow { place: Place },

    #[error("cannot borrow `{place:?}` as immutable: already borrowed as mutable")]
    SharedBorrowWhileMutBorrow { place: Place },

    #[error("use of moved value: `{place:?}`")]
    UseAfterMove { place: Place },

    #[error("use of uninitialized value: `{place:?}`")]
    UseOfUninitialized { place: Place },

    #[error("cannot assign to `{place:?}`: not mutable")]
    AssignToImmutable { place: Place },

    #[error("cannot assign to `{place:?}`: borrowed")]
    AssignWhileBorrowed { place: Place },

    #[error("lifetime `{lifetime:?}` does not outlive `{required:?}`")]
    LifetimeTooShort {
        lifetime: Lifetime,
        required: Lifetime,
    },

    #[error("cannot return reference to local variable")]
    ReturnLocalRef { place: Place },
}

/// Borrow checker implementation
#[derive(Debug, Clone)]
pub struct BorrowChecker {
    /// Whether to enforce strict borrowing rules
    pub strict_mode: bool,
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self { strict_mode: true }
    }

    /// Check if a move is valid
    pub fn check_move(&self, state: &OwnershipState, place: &Place) -> Result<(), BorrowError> {
        // Cannot move if place is borrowed
        if state.has_any_borrow(place) {
            return Err(BorrowError::MoveWhileBorrowed {
                place: place.clone(),
            });
        }

        // Cannot move if already moved
        if state.is_moved(place) {
            return Err(BorrowError::UseAfterMove {
                place: place.clone(),
            });
        }

        // Cannot move uninitialized
        if matches!(
            state.place_states.get(place),
            Some(PlaceState::Uninitialized)
        ) {
            return Err(BorrowError::UseOfUninitialized {
                place: place.clone(),
            });
        }

        Ok(())
    }

    /// Check if a borrow is valid
    pub fn check_borrow(
        &self,
        state: &OwnershipState,
        place: &Place,
        mutability: Mutability,
        _lifetime: &Lifetime,
    ) -> Result<(), BorrowError> {
        // Cannot borrow moved value
        if state.is_moved(place) {
            return Err(BorrowError::UseAfterMove {
                place: place.clone(),
            });
        }

        match mutability {
            Mutability::Shared => {
                // Cannot create shared borrow while mutably borrowed
                if state.has_mutable_borrow(place) {
                    return Err(BorrowError::SharedBorrowWhileMutBorrow {
                        place: place.clone(),
                    });
                }
            }
            Mutability::Mutable => {
                // Cannot create mutable borrow while any borrow exists
                if state.has_any_borrow(place) {
                    if state.has_mutable_borrow(place) {
                        return Err(BorrowError::MutBorrowWhileMutBorrow {
                            place: place.clone(),
                        });
                    }
                    return Err(BorrowError::MutBorrowWhileSharedBorrow {
                        place: place.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check if a use (read) is valid
    pub fn check_use(&self, state: &OwnershipState, place: &Place) -> Result<(), BorrowError> {
        if state.is_moved(place) {
            return Err(BorrowError::UseAfterMove {
                place: place.clone(),
            });
        }

        if matches!(
            state.place_states.get(place),
            Some(PlaceState::Uninitialized)
        ) {
            return Err(BorrowError::UseOfUninitialized {
                place: place.clone(),
            });
        }

        Ok(())
    }

    /// Check if an assignment is valid
    pub fn check_assign(
        &self,
        state: &OwnershipState,
        place: &Place,
        is_mutable: bool,
    ) -> Result<(), BorrowError> {
        // Place must be mutable
        if !is_mutable {
            return Err(BorrowError::AssignToImmutable {
                place: place.clone(),
            });
        }

        // Cannot assign while borrowed
        if state.has_any_borrow(place) {
            return Err(BorrowError::AssignWhileBorrowed {
                place: place.clone(),
            });
        }

        Ok(())
    }
}

impl Default for BorrowChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Drop elaboration - determines when destructors run
#[derive(Debug, Clone)]
pub struct DropElaborator {
    /// Places that need to be dropped
    pending_drops: Vec<(Place, RustType)>,
}

impl DropElaborator {
    pub fn new() -> Self {
        Self {
            pending_drops: Vec::new(),
        }
    }

    /// Schedule a drop for a place
    pub fn schedule_drop(&mut self, place: Place, ty: RustType) {
        if !ty.is_copy() {
            self.pending_drops.push((place, ty));
        }
    }

    /// Get drops in order (reverse of creation)
    pub fn drain_drops(&mut self) -> Vec<(Place, RustType)> {
        let mut drops = std::mem::take(&mut self.pending_drops);
        drops.reverse();
        drops
    }
}

impl Default for DropElaborator {
    fn default() -> Self {
        Self::new()
    }
}

/// Move path analysis for tracking partial moves
#[derive(Debug, Clone, Default)]
pub struct MoveAnalysis {
    /// Places that have been fully moved
    moved: HashSet<Place>,
    /// Places that have been partially moved (with moved children)
    partial_moves: HashMap<Place, HashSet<String>>,
}

impl MoveAnalysis {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a move
    pub fn record_move(&mut self, place: &Place) {
        self.moved.insert(place.clone());

        // Mark parent as partially moved
        if let Place::Field { base, field } = place {
            self.partial_moves
                .entry((**base).clone())
                .or_default()
                .insert(field.clone());
        }
    }

    /// Check if a place is fully moved
    pub fn is_moved(&self, place: &Place) -> bool {
        self.moved.contains(place)
    }

    /// Check if a place is partially moved
    pub fn is_partially_moved(&self, place: &Place) -> bool {
        self.partial_moves.contains_key(place)
    }

    /// Get moved fields of a place
    pub fn moved_fields(&self, place: &Place) -> Option<&HashSet<String>> {
        self.partial_moves.get(place)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_construction() {
        let local = Place::local(0);
        let field = local.clone().field("x");
        let deref = local.clone().deref();

        assert!(matches!(local, Place::Local(0)));
        assert!(matches!(field, Place::Field { .. }));
        assert!(matches!(deref, Place::Deref(_)));
    }

    #[test]
    fn test_place_prefix() {
        let base = Place::local(0);
        let field = base.clone().field("x");
        let nested = field.clone().field("y");

        assert!(base.is_prefix_of(&base));
        assert!(base.is_prefix_of(&field));
        assert!(base.is_prefix_of(&nested));
        assert!(field.is_prefix_of(&nested));

        assert!(!field.is_prefix_of(&base));
        assert!(!nested.is_prefix_of(&base));
    }

    #[test]
    fn test_ownership_state() {
        let place = Place::local(0);
        let mut state = OwnershipState::new();

        state.mark_owned(place.clone());
        assert!(state.is_owned(&place));
        assert!(!state.is_moved(&place));

        state.mark_moved(place.clone());
        assert!(!state.is_owned(&place));
        assert!(state.is_moved(&place));
    }

    #[test]
    fn test_borrow_checking() {
        let checker = BorrowChecker::new();
        let place = Place::local(0);
        let lifetime = Lifetime::Named("a".to_string());

        let mut state = OwnershipState::new();
        state.mark_owned(place.clone());

        // Can create shared borrow of owned value
        assert!(checker
            .check_borrow(&state, &place, Mutability::Shared, &lifetime)
            .is_ok());

        // Can create mutable borrow of owned value
        assert!(checker
            .check_borrow(&state, &place, Mutability::Mutable, &lifetime)
            .is_ok());

        // Add a mutable borrow
        state.add_borrow(place.clone(), Mutability::Mutable, lifetime.clone());

        // Cannot create another mutable borrow
        assert!(checker
            .check_borrow(&state, &place, Mutability::Mutable, &lifetime)
            .is_err());

        // Cannot create shared borrow while mutably borrowed
        assert!(checker
            .check_borrow(&state, &place, Mutability::Shared, &lifetime)
            .is_err());
    }

    #[test]
    fn test_multiple_shared_borrows() {
        let checker = BorrowChecker::new();
        let place = Place::local(0);
        let lifetime = Lifetime::Named("a".to_string());

        let mut state = OwnershipState::new();
        state.mark_owned(place.clone());

        // Add first shared borrow
        state.add_borrow(place.clone(), Mutability::Shared, lifetime.clone());

        // Can add another shared borrow
        assert!(checker
            .check_borrow(&state, &place, Mutability::Shared, &lifetime)
            .is_ok());
    }

    #[test]
    fn test_move_while_borrowed() {
        let checker = BorrowChecker::new();
        let place = Place::local(0);
        let lifetime = Lifetime::Named("a".to_string());

        let mut state = OwnershipState::new();
        state.mark_owned(place.clone());
        state.add_borrow(place.clone(), Mutability::Shared, lifetime);

        // Cannot move while borrowed
        let result = checker.check_move(&state, &place);
        assert!(matches!(result, Err(BorrowError::MoveWhileBorrowed { .. })));
    }

    #[test]
    fn test_use_after_move() {
        let checker = BorrowChecker::new();
        let place = Place::local(0);

        let mut state = OwnershipState::new();
        state.mark_moved(place.clone());

        // Cannot use moved value
        let result = checker.check_use(&state, &place);
        assert!(matches!(result, Err(BorrowError::UseAfterMove { .. })));
    }

    #[test]
    fn test_end_borrow() {
        let place = Place::local(0);
        let lifetime = Lifetime::Named("a".to_string());

        let mut state = OwnershipState::new();
        state.mark_owned(place.clone());
        state.add_borrow(place.clone(), Mutability::Mutable, lifetime.clone());

        assert!(state.is_borrowed(&place));

        state.end_borrows(&lifetime);

        assert!(state.is_owned(&place));
        assert!(!state.is_borrowed(&place));
    }

    #[test]
    fn test_move_analysis() {
        let mut analysis = MoveAnalysis::new();

        let base = Place::local(0);
        let field_x = base.clone().field("x");
        let field_y = base.clone().field("y");

        analysis.record_move(&field_x);

        assert!(analysis.is_moved(&field_x));
        assert!(!analysis.is_moved(&field_y));
        assert!(analysis.is_partially_moved(&base));

        let moved = analysis.moved_fields(&base).unwrap();
        assert!(moved.contains("x"));
        assert!(!moved.contains("y"));
    }
}
