//! Rust Memory Model Formalization
//!
//! This module formalizes the Rust memory model, including:
//!
//! - Memory allocation and deallocation
//! - Pointer validity and provenance
//! - Alignment requirements
//! - Read/write operations
//!
//! The model is inspired by CompCert and Stacked Borrows (Ralf Jung et al.)
//!
//! ## Memory Regions
//!
//! Memory is organized into regions:
//! - Stack frames (per function call)
//! - Heap allocations (Box, Vec, etc.)
//! - Static/global memory
//!
//! Each allocation has:
//! - A unique allocation ID (provenance)
//! - Size and alignment
//! - Read/write permissions

use crate::types::RustType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Allocation ID (provenance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllocId(pub u64);

/// Memory address (abstract, not real addresses)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address {
    /// Allocation this pointer came from
    pub alloc_id: AllocId,
    /// Offset within the allocation
    pub offset: u64,
}

impl Address {
    pub fn new(alloc_id: AllocId, offset: u64) -> Self {
        Self { alloc_id, offset }
    }

    /// Add offset to address
    pub fn offset(self, delta: i64) -> Option<Self> {
        let new_offset = if delta >= 0 {
            self.offset.checked_add(delta as u64)?
        } else {
            self.offset.checked_sub((-delta) as u64)?
        };
        Some(Self {
            alloc_id: self.alloc_id,
            offset: new_offset,
        })
    }
}

/// Memory allocation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    /// Unique ID
    pub id: AllocId,
    /// Size in bytes
    pub size: usize,
    /// Alignment requirement
    pub align: usize,
    /// Whether allocation is still valid
    pub valid: bool,
    /// The data stored (as bytes)
    pub data: Vec<u8>,
    /// Optional type information
    pub ty: Option<RustType>,
}

impl Allocation {
    /// Create a new allocation
    pub fn new(id: AllocId, size: usize, align: usize) -> Self {
        Self {
            id,
            size,
            align,
            valid: true,
            data: vec![0; size],
            ty: None,
        }
    }

    /// Check if an offset is in bounds
    pub fn in_bounds(&self, offset: u64, size: usize) -> bool {
        offset as usize + size <= self.size
    }

    /// Check if offset is aligned for the given size
    pub fn is_aligned(&self, offset: u64, align: usize) -> bool {
        offset.is_multiple_of(align as u64)
    }
}

/// Memory operation errors
#[derive(Debug, Clone, Error)]
pub enum MemoryError {
    #[error("allocation failed: size {size}, alignment {align}")]
    AllocationFailed { size: usize, align: usize },

    #[error("invalid pointer: allocation {0:?} does not exist")]
    InvalidPointer(AllocId),

    #[error("use after free: allocation {0:?} has been deallocated")]
    UseAfterFree(AllocId),

    #[error("double free: allocation {0:?}")]
    DoubleFree(AllocId),

    #[error("out of bounds access: offset {offset} + size {size} > allocation size {alloc_size}")]
    OutOfBounds {
        offset: u64,
        size: usize,
        alloc_size: usize,
    },

    #[error("misaligned access: offset {offset} not aligned to {align}")]
    Misaligned { offset: u64, align: usize },

    #[error("null pointer dereference")]
    NullPointer,

    #[error("integer overflow in pointer arithmetic")]
    PointerOverflow,
}

/// The memory model
#[derive(Debug, Clone)]
pub struct Memory {
    /// All allocations indexed by ID
    allocations: HashMap<AllocId, Allocation>,
    /// Counter for allocation IDs
    next_alloc_id: u64,
    /// Null allocation (never valid to access)
    null_alloc: AllocId,
}

impl Memory {
    /// Create a new memory model
    pub fn new() -> Self {
        let null_alloc = AllocId(0);
        Self {
            allocations: HashMap::new(),
            next_alloc_id: 1, // Start from 1, 0 is null
            null_alloc,
        }
    }

    /// Allocate memory of given size and alignment
    pub fn allocate(&mut self, size: usize) -> Result<Address, MemoryError> {
        self.allocate_aligned(size, 1)
    }

    /// Allocate memory with specific alignment
    pub fn allocate_aligned(&mut self, size: usize, align: usize) -> Result<Address, MemoryError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(MemoryError::AllocationFailed { size, align });
        }

        let id = AllocId(self.next_alloc_id);
        self.next_alloc_id += 1;

        let alloc = Allocation::new(id, size, align);
        self.allocations.insert(id, alloc);

        Ok(Address::new(id, 0))
    }

    /// Allocate memory for a typed value
    pub fn allocate_typed(&mut self, ty: &RustType) -> Result<Address, MemoryError> {
        let size = ty
            .size()
            .ok_or(MemoryError::AllocationFailed { size: 0, align: 1 })?;
        let align = self.type_alignment(ty);
        let addr = self.allocate_aligned(size, align)?;

        // Store type info
        if let Some(alloc) = self.allocations.get_mut(&addr.alloc_id) {
            alloc.ty = Some(ty.clone());
        }

        Ok(addr)
    }

    /// Get alignment for a type
    fn type_alignment(&self, ty: &RustType) -> usize {
        match ty {
            RustType::Unit | RustType::Bool => 1,
            RustType::Char => 4,
            RustType::Uint(u) => u.size(),
            RustType::Int(i) => i.size(),
            RustType::Float(f) => f.size(),
            RustType::Array { element, .. } => self.type_alignment(element),
            RustType::Tuple(elems) => elems
                .iter()
                .map(|e| self.type_alignment(e))
                .max()
                .unwrap_or(1),
            // Default to pointer alignment for references, raw pointers, and other types
            _ => 8,
        }
    }

    /// Deallocate memory
    pub fn deallocate(&mut self, addr: Address) -> Result<(), MemoryError> {
        let alloc = self
            .allocations
            .get_mut(&addr.alloc_id)
            .ok_or(MemoryError::InvalidPointer(addr.alloc_id))?;

        if !alloc.valid {
            return Err(MemoryError::DoubleFree(addr.alloc_id));
        }

        alloc.valid = false;
        Ok(())
    }

    /// Check if a pointer is valid
    pub fn is_valid(&self, addr: Address) -> bool {
        self.allocations
            .get(&addr.alloc_id)
            .is_some_and(|a| a.valid)
    }

    /// Check if pointer is null
    pub fn is_null(&self, addr: Address) -> bool {
        addr.alloc_id == self.null_alloc
    }

    /// Get a null pointer
    pub fn null_ptr(&self) -> Address {
        Address::new(self.null_alloc, 0)
    }

    /// Read bytes from memory
    pub fn read_bytes(&self, addr: Address, size: usize) -> Result<&[u8], MemoryError> {
        let alloc = self
            .allocations
            .get(&addr.alloc_id)
            .ok_or(MemoryError::InvalidPointer(addr.alloc_id))?;

        if !alloc.valid {
            return Err(MemoryError::UseAfterFree(addr.alloc_id));
        }

        if !alloc.in_bounds(addr.offset, size) {
            return Err(MemoryError::OutOfBounds {
                offset: addr.offset,
                size,
                alloc_size: alloc.size,
            });
        }

        let start = addr.offset as usize;
        Ok(&alloc.data[start..start + size])
    }

    /// Write bytes to memory
    pub fn write_bytes(&mut self, addr: Address, data: &[u8]) -> Result<(), MemoryError> {
        let alloc = self
            .allocations
            .get_mut(&addr.alloc_id)
            .ok_or(MemoryError::InvalidPointer(addr.alloc_id))?;

        if !alloc.valid {
            return Err(MemoryError::UseAfterFree(addr.alloc_id));
        }

        if !alloc.in_bounds(addr.offset, data.len()) {
            return Err(MemoryError::OutOfBounds {
                offset: addr.offset,
                size: data.len(),
                alloc_size: alloc.size,
            });
        }

        let start = addr.offset as usize;
        alloc.data[start..start + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read a u8
    pub fn read_u8(&self, addr: Address) -> Result<u8, MemoryError> {
        let bytes = self.read_bytes(addr, 1)?;
        Ok(bytes[0])
    }

    /// Read a u16
    pub fn read_u16(&self, addr: Address) -> Result<u16, MemoryError> {
        let bytes = self.read_bytes(addr, 2)?;
        Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Read a u32
    pub fn read_u32(&self, addr: Address) -> Result<u32, MemoryError> {
        let bytes = self.read_bytes(addr, 4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Read a u64
    pub fn read_u64(&self, addr: Address) -> Result<u64, MemoryError> {
        let bytes = self.read_bytes(addr, 8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Write a u8
    pub fn write_u8(&mut self, addr: Address, val: u8) -> Result<(), MemoryError> {
        self.write_bytes(addr, &[val])
    }

    /// Write a u16
    pub fn write_u16(&mut self, addr: Address, val: u16) -> Result<(), MemoryError> {
        self.write_bytes(addr, &val.to_le_bytes())
    }

    /// Write a u32
    pub fn write_u32(&mut self, addr: Address, val: u32) -> Result<(), MemoryError> {
        self.write_bytes(addr, &val.to_le_bytes())
    }

    /// Write a u64
    pub fn write_u64(&mut self, addr: Address, val: u64) -> Result<(), MemoryError> {
        self.write_bytes(addr, &val.to_le_bytes())
    }

    /// Get allocation info
    pub fn get_allocation(&self, id: AllocId) -> Option<&Allocation> {
        self.allocations.get(&id)
    }

    /// Get mutable allocation info
    pub fn get_allocation_mut(&mut self, id: AllocId) -> Option<&mut Allocation> {
        self.allocations.get_mut(&id)
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

/// Stack frame for local variables
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Frame identifier
    pub id: u64,
    /// Local variable allocations
    pub locals: Vec<Address>,
    /// Return address placeholder
    pub return_dest: Option<Address>,
}

impl StackFrame {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            locals: Vec::new(),
            return_dest: None,
        }
    }

    /// Add a local variable
    pub fn add_local(&mut self, addr: Address) -> u32 {
        // SAFETY: Local variable count is bounded by practical stack depth limits,
        // which are far below u32::MAX. Use saturating conversion for defense.
        let idx = u32::try_from(self.locals.len()).unwrap_or(u32::MAX);
        self.locals.push(addr);
        idx
    }

    /// Get local variable address
    pub fn get_local(&self, idx: u32) -> Option<Address> {
        self.locals.get(idx as usize).copied()
    }
}

/// Execution stack (call stack)
#[derive(Debug, Clone)]
pub struct Stack {
    frames: Vec<StackFrame>,
    next_frame_id: u64,
}

impl Stack {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            next_frame_id: 0,
        }
    }

    /// Push a new frame
    pub fn push_frame(&mut self) -> &mut StackFrame {
        let id = self.next_frame_id;
        self.next_frame_id += 1;
        self.frames.push(StackFrame::new(id));
        self.frames.last_mut().unwrap()
    }

    /// Pop the current frame
    pub fn pop_frame(&mut self) -> Option<StackFrame> {
        self.frames.pop()
    }

    /// Get current frame
    pub fn current_frame(&self) -> Option<&StackFrame> {
        self.frames.last()
    }

    /// Get current frame mutably
    pub fn current_frame_mut(&mut self) -> Option<&mut StackFrame> {
        self.frames.last_mut()
    }

    /// Get frame at depth (0 = current)
    pub fn frame_at(&self, depth: usize) -> Option<&StackFrame> {
        let idx = self.frames.len().checked_sub(depth + 1)?;
        self.frames.get(idx)
    }

    /// Stack depth
    pub fn depth(&self) -> usize {
        self.frames.len()
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::UintType;

    #[test]
    fn test_allocation() {
        let mut mem = Memory::new();

        let addr = mem.allocate(16).expect("allocation failed");
        assert!(mem.is_valid(addr));
        assert!(!mem.is_null(addr));
    }

    #[test]
    fn test_read_write() {
        let mut mem = Memory::new();

        let addr = mem.allocate(8).unwrap();

        mem.write_u32(addr, 42).unwrap();
        assert_eq!(mem.read_u32(addr).unwrap(), 42);

        mem.write_u64(addr, 0xDEAD_BEEF).unwrap();
        assert_eq!(mem.read_u64(addr).unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn test_use_after_free() {
        let mut mem = Memory::new();

        let addr = mem.allocate(4).unwrap();
        mem.deallocate(addr).unwrap();

        let result = mem.read_u32(addr);
        assert!(matches!(result, Err(MemoryError::UseAfterFree(_))));
    }

    #[test]
    fn test_double_free() {
        let mut mem = Memory::new();

        let addr = mem.allocate(4).unwrap();
        mem.deallocate(addr).unwrap();

        let result = mem.deallocate(addr);
        assert!(matches!(result, Err(MemoryError::DoubleFree(_))));
    }

    #[test]
    fn test_out_of_bounds() {
        let mut mem = Memory::new();

        let addr = mem.allocate(4).unwrap();

        // Try to read 8 bytes from 4-byte allocation
        let result = mem.read_u64(addr);
        assert!(matches!(result, Err(MemoryError::OutOfBounds { .. })));
    }

    #[test]
    fn test_pointer_offset() {
        let addr = Address::new(AllocId(1), 10);

        let forward = addr.offset(5).unwrap();
        assert_eq!(forward.offset, 15);

        let backward = addr.offset(-3).unwrap();
        assert_eq!(backward.offset, 7);

        // Overflow protection
        let overflow = addr.offset(-20);
        assert!(overflow.is_none());
    }

    #[test]
    fn test_typed_allocation() {
        let mut mem = Memory::new();

        let u32_ty = RustType::Uint(UintType::U32);
        let addr = mem.allocate_typed(&u32_ty).unwrap();

        let alloc = mem.get_allocation(addr.alloc_id).unwrap();
        assert_eq!(alloc.size, 4);
        assert_eq!(alloc.ty, Some(u32_ty));
    }

    #[test]
    fn test_stack_frame() {
        let mut mem = Memory::new();
        let mut stack = Stack::new();

        // Push a frame
        let frame = stack.push_frame();
        let addr = mem.allocate(4).unwrap();
        frame.add_local(addr);

        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.current_frame().unwrap().get_local(0), Some(addr));

        // Pop frame
        let popped = stack.pop_frame().unwrap();
        assert_eq!(popped.get_local(0), Some(addr));
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_null_pointer() {
        let mem = Memory::new();

        let null = mem.null_ptr();
        assert!(mem.is_null(null));

        // Reading from null should fail
        let result = mem.read_u8(null);
        assert!(result.is_err());
    }
}
