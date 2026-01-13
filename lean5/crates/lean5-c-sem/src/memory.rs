//! C Memory Model Formalization
//!
//! This module implements a CompCert-style block-based memory model for C.
//! Key design principles:
//!
//! ## Block-Based Memory
//!
//! Memory is organized as a collection of blocks, each identified by a unique
//! block ID. This models:
//! - Stack allocations (function locals)
//! - Heap allocations (malloc/free)
//! - Global variables
//!
//! ## Pointers and Provenance
//!
//! Pointers are (block_id, offset) pairs. The block_id provides "provenance"
//! tracking to detect:
//! - Use-after-free
//! - Out-of-bounds access
//! - Pointer comparison between different allocations
//!
//! ## Undefined Behavior Detection
//!
//! Memory operations can trigger undefined behavior:
//! - Null pointer dereference
//! - Use after free
//! - Out of bounds access
//! - Unaligned access
//! - Type punning violations (strict aliasing)
//!
//! ## References
//!
//! - CompCert memory model: <https://github.com/AbsInt/CompCert>
//! - Cerberus C semantics: <https://www.cl.cam.ac.uk/~pes20/cerberus/>

use crate::types::CType;
use crate::ub::{UBKind, UBResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Block identifier (provenance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u64);

impl BlockId {
    /// Null block (invalid)
    pub const NULL: BlockId = BlockId(0);

    /// Check if this is the null block
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

/// A pointer in the C memory model
///
/// Pointers are (block_id, offset) pairs. This allows tracking provenance
/// to detect use-after-free and out-of-bounds access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pointer {
    /// Block this pointer belongs to
    pub block: BlockId,
    /// Offset within the block
    pub offset: i64,
}

impl Pointer {
    /// Create a null pointer
    pub const fn null() -> Self {
        Self {
            block: BlockId::NULL,
            offset: 0,
        }
    }

    /// Create a pointer to the start of a block
    pub fn new(block: BlockId) -> Self {
        Self { block, offset: 0 }
    }

    /// Create a pointer with offset
    pub fn with_offset(block: BlockId, offset: i64) -> Self {
        Self { block, offset }
    }

    /// Check if this is a null pointer
    pub fn is_null(&self) -> bool {
        self.block.is_null()
    }

    /// Add offset to pointer (pointer arithmetic)
    pub fn offset(self, delta: i64) -> Option<Self> {
        let new_offset = self.offset.checked_add(delta)?;
        Some(Self {
            block: self.block,
            offset: new_offset,
        })
    }

    /// Subtract pointers (result is difference in bytes)
    ///
    /// Returns None if pointers are to different blocks (UB in C)
    pub fn diff(self, other: Self) -> Option<i64> {
        if self.block != other.block {
            None // UB: comparing pointers to different objects
        } else {
            self.offset.checked_sub(other.offset)
        }
    }
}

/// Allocation permission flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permissions {
    /// Can read from this allocation
    pub readable: bool,
    /// Can write to this allocation
    pub writable: bool,
    /// Can free this allocation
    pub freeable: bool,
}

impl Permissions {
    /// Full permissions (heap allocation)
    pub fn heap() -> Self {
        Self {
            readable: true,
            writable: true,
            freeable: true,
        }
    }

    /// Stack permissions (not freeable via free())
    pub fn stack() -> Self {
        Self {
            readable: true,
            writable: true,
            freeable: false,
        }
    }

    /// Read-only (const global)
    pub fn readonly() -> Self {
        Self {
            readable: true,
            writable: false,
            freeable: false,
        }
    }
}

/// Memory block metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Unique identifier
    pub id: BlockId,
    /// Size in bytes
    pub size: usize,
    /// Alignment requirement
    pub align: usize,
    /// Whether block is still valid (not freed)
    pub valid: bool,
    /// Permissions
    pub perms: Permissions,
    /// The actual data
    pub data: Vec<u8>,
    /// Type information (for strict aliasing)
    pub ty: Option<CType>,
    /// Debug name (for error messages)
    pub name: Option<String>,
}

impl Block {
    /// Create a new block
    pub fn new(id: BlockId, size: usize, align: usize, perms: Permissions) -> Self {
        Self {
            id,
            size,
            align,
            valid: true,
            perms,
            data: vec![0; size],
            ty: None,
            name: None,
        }
    }

    /// Check if offset is in bounds for access of given size
    pub fn in_bounds(&self, offset: i64, access_size: usize) -> bool {
        if offset < 0 {
            return false;
        }
        let offset = offset as usize;
        offset
            .checked_add(access_size)
            .is_some_and(|end| end <= self.size)
    }

    /// Check if offset is properly aligned for given alignment requirement
    pub fn is_aligned(&self, offset: i64, required_align: usize) -> bool {
        if offset < 0 {
            return false;
        }
        (offset as usize).is_multiple_of(required_align)
    }
}

/// The C memory model
///
/// Manages allocation, deallocation, and access to memory blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// All allocated blocks (including freed ones for provenance tracking)
    blocks: HashMap<BlockId, Block>,
    /// Next block ID to allocate
    next_block_id: u64,
    /// Stack frames (for automatic storage duration)
    stack_frames: Vec<Vec<BlockId>>,
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory {
    /// Create a new empty memory
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            next_block_id: 1,               // 0 is reserved for NULL
            stack_frames: vec![Vec::new()], // Global frame
        }
    }

    /// Allocate a new block (malloc)
    pub fn alloc(&mut self, size: usize, align: usize) -> UBResult<Pointer> {
        // Zero-size allocation returns a unique non-null pointer
        let actual_size = if size == 0 { 1 } else { size };

        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;

        let block = Block::new(id, actual_size, align, Permissions::heap());
        self.blocks.insert(id, block);

        Ok(Pointer::new(id))
    }

    /// Allocate with type information
    pub fn alloc_typed(&mut self, ty: &CType) -> UBResult<Pointer> {
        let size = ty.size();
        let align = ty.align();
        let ptr = self.alloc(size, align)?;

        if let Some(block) = self.blocks.get_mut(&ptr.block) {
            block.ty = Some(ty.clone());
        }

        Ok(ptr)
    }

    /// Allocate on stack (automatic storage duration)
    pub fn alloc_stack(
        &mut self,
        size: usize,
        align: usize,
        name: Option<String>,
    ) -> UBResult<Pointer> {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;

        let mut block = Block::new(id, size, align, Permissions::stack());
        block.name = name;
        self.blocks.insert(id, block);

        // Track in current stack frame
        if let Some(frame) = self.stack_frames.last_mut() {
            frame.push(id);
        }

        Ok(Pointer::new(id))
    }

    /// Free a block (free)
    pub fn free(&mut self, ptr: Pointer) -> UBResult<()> {
        if ptr.is_null() {
            // free(NULL) is a no-op
            return Ok(());
        }

        if ptr.offset != 0 {
            // Must free at block start
            return Err(UBKind::InvalidFree);
        }

        let block = self
            .blocks
            .get_mut(&ptr.block)
            .ok_or(UBKind::UseAfterFree)?;

        if !block.valid {
            return Err(UBKind::DoubleFree);
        }

        if !block.perms.freeable {
            return Err(UBKind::InvalidFree);
        }

        block.valid = false;
        Ok(())
    }

    /// Push a new stack frame (function entry)
    pub fn push_frame(&mut self) {
        self.stack_frames.push(Vec::new());
    }

    /// Pop a stack frame (function exit)
    pub fn pop_frame(&mut self) {
        if let Some(frame) = self.stack_frames.pop() {
            for block_id in frame {
                if let Some(block) = self.blocks.get_mut(&block_id) {
                    block.valid = false;
                }
            }
        }
    }

    /// Check if a pointer is valid for the given access
    pub fn is_valid(&self, ptr: Pointer) -> bool {
        if ptr.is_null() {
            return false;
        }

        self.blocks
            .get(&ptr.block)
            .is_some_and(|b| b.valid && b.in_bounds(ptr.offset, 0))
    }

    /// Check if pointer is valid for read of given size
    pub fn can_read(&self, ptr: Pointer, size: usize) -> UBResult<()> {
        if ptr.is_null() {
            return Err(UBKind::NullDeref);
        }

        let block = self.blocks.get(&ptr.block).ok_or(UBKind::InvalidPointer)?;

        if !block.valid {
            return Err(UBKind::UseAfterFree);
        }

        if !block.perms.readable {
            return Err(UBKind::ReadViolation);
        }

        if !block.in_bounds(ptr.offset, size) {
            return Err(UBKind::OutOfBounds);
        }

        Ok(())
    }

    /// Check if pointer is valid for write of given size
    pub fn can_write(&self, ptr: Pointer, size: usize) -> UBResult<()> {
        if ptr.is_null() {
            return Err(UBKind::NullDeref);
        }

        let block = self.blocks.get(&ptr.block).ok_or(UBKind::InvalidPointer)?;

        if !block.valid {
            return Err(UBKind::UseAfterFree);
        }

        if !block.perms.writable {
            return Err(UBKind::WriteViolation);
        }

        if !block.in_bounds(ptr.offset, size) {
            return Err(UBKind::OutOfBounds);
        }

        Ok(())
    }

    /// Load bytes from memory
    pub fn load_bytes(&self, ptr: Pointer, size: usize) -> UBResult<Vec<u8>> {
        self.can_read(ptr, size)?;

        let block = self.blocks.get(&ptr.block).unwrap();
        let start = ptr.offset as usize;
        let end = start + size;

        Ok(block.data[start..end].to_vec())
    }

    /// Store bytes to memory
    pub fn store_bytes(&mut self, ptr: Pointer, bytes: &[u8]) -> UBResult<()> {
        self.can_write(ptr, bytes.len())?;

        let block = self.blocks.get_mut(&ptr.block).unwrap();
        let start = ptr.offset as usize;

        block.data[start..start + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Load a u8 value
    pub fn load_u8(&self, ptr: Pointer) -> UBResult<u8> {
        let bytes = self.load_bytes(ptr, 1)?;
        Ok(bytes[0])
    }

    /// Store a u8 value
    pub fn store_u8(&mut self, ptr: Pointer, val: u8) -> UBResult<()> {
        self.store_bytes(ptr, &[val])
    }

    /// Load a u16 value (little-endian)
    pub fn load_u16(&self, ptr: Pointer) -> UBResult<u16> {
        let bytes = self.load_bytes(ptr, 2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Store a u16 value (little-endian)
    pub fn store_u16(&mut self, ptr: Pointer, val: u16) -> UBResult<()> {
        self.store_bytes(ptr, &val.to_le_bytes())
    }

    /// Load a u32 value (little-endian)
    pub fn load_u32(&self, ptr: Pointer) -> UBResult<u32> {
        let bytes = self.load_bytes(ptr, 4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Store a u32 value (little-endian)
    pub fn store_u32(&mut self, ptr: Pointer, val: u32) -> UBResult<()> {
        self.store_bytes(ptr, &val.to_le_bytes())
    }

    /// Load a u64 value (little-endian)
    pub fn load_u64(&self, ptr: Pointer) -> UBResult<u64> {
        let bytes = self.load_bytes(ptr, 8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Store a u64 value (little-endian)
    pub fn store_u64(&mut self, ptr: Pointer, val: u64) -> UBResult<()> {
        self.store_bytes(ptr, &val.to_le_bytes())
    }

    /// Load a signed i32 value
    pub fn load_i32(&self, ptr: Pointer) -> UBResult<i32> {
        let bytes = self.load_bytes(ptr, 4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Store a signed i32 value
    pub fn store_i32(&mut self, ptr: Pointer, val: i32) -> UBResult<()> {
        self.store_bytes(ptr, &val.to_le_bytes())
    }

    /// Load a signed i64 value
    pub fn load_i64(&self, ptr: Pointer) -> UBResult<i64> {
        let bytes = self.load_bytes(ptr, 8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Store a signed i64 value
    pub fn store_i64(&mut self, ptr: Pointer, val: i64) -> UBResult<()> {
        self.store_bytes(ptr, &val.to_le_bytes())
    }

    /// Load a pointer value
    ///
    /// Pointers are stored as 8 bytes in LP64 format: 4 bytes for block_id (u32) and
    /// 4 bytes for offset (i32). This matches C pointer size while preserving provenance.
    ///
    /// # Note
    /// The internal Pointer type uses u64/i64 for flexibility, but C-level storage
    /// uses the LP64 8-byte format. Values are zero/sign-extended on load.
    pub fn load_ptr(&self, ptr: Pointer) -> UBResult<Pointer> {
        let bytes = self.load_bytes(ptr, 8)?;
        // Block ID stored as u32 in first 4 bytes, zero-extended to u64
        let block_id = u64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        // Offset stored as i32 in last 4 bytes, sign-extended to i64
        let offset = i64::from(i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]));
        Ok(Pointer {
            block: BlockId(block_id),
            offset,
        })
    }

    /// Store a pointer value
    ///
    /// Pointers are stored as 8 bytes in LP64 format: 4 bytes for block_id (u32) and
    /// 4 bytes for offset (i32). This matches C pointer size while preserving provenance.
    ///
    /// # Errors
    /// Returns an error if the block_id exceeds u32::MAX or offset exceeds i32 range,
    /// as these cannot be represented in 8-byte pointer storage.
    pub fn store_ptr(&mut self, ptr: Pointer, val: Pointer) -> UBResult<()> {
        // Check that values fit in 8-byte pointer format
        // If block_id or offset exceed representable range, this indicates an invalid pointer
        let block_id = u32::try_from(val.block.0).map_err(|_| UBKind::InvalidPointer)?;
        let offset = i32::try_from(val.offset).map_err(|_| UBKind::PointerOverflow)?;

        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&block_id.to_le_bytes());
        bytes[4..8].copy_from_slice(&offset.to_le_bytes());
        self.store_bytes(ptr, &bytes)
    }

    /// Load a float (f32)
    pub fn load_f32(&self, ptr: Pointer) -> UBResult<f32> {
        let bytes = self.load_bytes(ptr, 4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Store a float (f32)
    pub fn store_f32(&mut self, ptr: Pointer, val: f32) -> UBResult<()> {
        self.store_bytes(ptr, &val.to_le_bytes())
    }

    /// Load a double (f64)
    pub fn load_f64(&self, ptr: Pointer) -> UBResult<f64> {
        let bytes = self.load_bytes(ptr, 8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Store a double (f64)
    pub fn store_f64(&mut self, ptr: Pointer, val: f64) -> UBResult<()> {
        self.store_bytes(ptr, &val.to_le_bytes())
    }

    /// Copy memory (memcpy semantics - no overlap allowed)
    pub fn memcpy(&mut self, dst: Pointer, src: Pointer, size: usize) -> UBResult<()> {
        // Check for overlap (UB)
        if dst.block == src.block {
            let dst_start = dst.offset;
            let dst_end = dst.offset + size as i64;
            let src_start = src.offset;
            let src_end = src.offset + size as i64;

            if (dst_start < src_end) && (src_start < dst_end) {
                return Err(UBKind::OverlappingMemcpy);
            }
        }

        let bytes = self.load_bytes(src, size)?;
        self.store_bytes(dst, &bytes)
    }

    /// Move memory (memmove semantics - overlap allowed)
    pub fn memmove(&mut self, dst: Pointer, src: Pointer, size: usize) -> UBResult<()> {
        let bytes = self.load_bytes(src, size)?;
        self.store_bytes(dst, &bytes)
    }

    /// Set memory (memset semantics)
    pub fn memset(&mut self, dst: Pointer, val: u8, size: usize) -> UBResult<()> {
        self.can_write(dst, size)?;

        let block = self.blocks.get_mut(&dst.block).unwrap();
        let start = dst.offset as usize;

        for i in 0..size {
            block.data[start + i] = val;
        }

        Ok(())
    }

    /// Compare memory (memcmp semantics)
    pub fn memcmp(&self, ptr1: Pointer, ptr2: Pointer, size: usize) -> UBResult<i32> {
        let bytes1 = self.load_bytes(ptr1, size)?;
        let bytes2 = self.load_bytes(ptr2, size)?;

        for i in 0..size {
            if bytes1[i] != bytes2[i] {
                return Ok((bytes1[i] as i32) - (bytes2[i] as i32));
            }
        }

        Ok(0)
    }

    /// Get block info (for debugging)
    pub fn block_info(&self, ptr: Pointer) -> Option<&Block> {
        self.blocks.get(&ptr.block)
    }

    /// Get total allocated size (for stats)
    pub fn total_allocated(&self) -> usize {
        self.blocks
            .values()
            .filter(|b| b.valid)
            .map(|b| b.size)
            .sum()
    }

    /// Get number of live allocations
    pub fn num_allocations(&self) -> usize {
        self.blocks.values().filter(|b| b.valid).count()
    }
}

/// Memory comparison result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtrCmpResult {
    /// Definitely less than
    Less,
    /// Definitely equal
    Equal,
    /// Definitely greater than
    Greater,
    /// Comparison is undefined (different blocks)
    Undefined,
}

impl Memory {
    /// Compare two pointers
    ///
    /// Pointer comparison is only defined for pointers to the same object
    /// or one-past-the-end.
    pub fn ptr_cmp(&self, p1: Pointer, p2: Pointer) -> PtrCmpResult {
        // NULL comparisons are always defined
        if p1.is_null() && p2.is_null() {
            return PtrCmpResult::Equal;
        }
        if p1.is_null() || p2.is_null() {
            // One null, one not - technically implementation-defined
            // but we'll say undefined for safety
            return PtrCmpResult::Undefined;
        }

        // Same block: comparison is defined
        if p1.block == p2.block {
            use std::cmp::Ordering;
            match p1.offset.cmp(&p2.offset) {
                Ordering::Less => PtrCmpResult::Less,
                Ordering::Equal => PtrCmpResult::Equal,
                Ordering::Greater => PtrCmpResult::Greater,
            }
        } else {
            // Different blocks: undefined behavior
            PtrCmpResult::Undefined
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointer_arithmetic() {
        let ptr = Pointer::new(BlockId(1));
        let ptr_plus_10 = ptr.offset(10).unwrap();
        let ptr_minus_5 = ptr_plus_10.offset(-5).unwrap();

        assert_eq!(ptr_plus_10.offset, 10);
        assert_eq!(ptr_minus_5.offset, 5);
    }

    #[test]
    fn test_pointer_diff() {
        let ptr1 = Pointer::with_offset(BlockId(1), 100);
        let ptr2 = Pointer::with_offset(BlockId(1), 60);
        let ptr3 = Pointer::with_offset(BlockId(2), 100);

        // Same block: diff is defined
        assert_eq!(ptr1.diff(ptr2), Some(40));
        assert_eq!(ptr2.diff(ptr1), Some(-40));

        // Different blocks: diff is undefined
        assert_eq!(ptr1.diff(ptr3), None);
    }

    #[test]
    fn test_allocation_and_free() {
        let mut mem = Memory::new();

        // Allocate
        let ptr = mem.alloc(100, 8).unwrap();
        assert!(mem.is_valid(ptr));
        assert_eq!(mem.num_allocations(), 1);

        // Free
        mem.free(ptr).unwrap();
        assert!(!mem.is_valid(ptr));
        assert_eq!(mem.num_allocations(), 0);
    }

    #[test]
    fn test_double_free() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(100, 8).unwrap();
        mem.free(ptr).unwrap();

        let result = mem.free(ptr);
        assert!(matches!(result, Err(UBKind::DoubleFree)));
    }

    #[test]
    fn test_use_after_free() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(100, 8).unwrap();
        mem.free(ptr).unwrap();

        let result = mem.load_bytes(ptr, 10);
        assert!(matches!(result, Err(UBKind::UseAfterFree)));
    }

    #[test]
    fn test_null_deref() {
        let mem = Memory::new();
        let null = Pointer::null();

        let result = mem.load_bytes(null, 1);
        assert!(matches!(result, Err(UBKind::NullDeref)));
    }

    #[test]
    fn test_out_of_bounds() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(10, 1).unwrap();

        // In bounds
        assert!(mem.load_bytes(ptr, 10).is_ok());

        // Out of bounds
        let result = mem.load_bytes(ptr, 11);
        assert!(matches!(result, Err(UBKind::OutOfBounds)));

        // Offset out of bounds
        let ptr_plus_5 = ptr.offset(5).unwrap();
        let result = mem.load_bytes(ptr_plus_5, 10);
        assert!(matches!(result, Err(UBKind::OutOfBounds)));
    }

    #[test]
    fn test_stack_frames() {
        let mut mem = Memory::new();

        // Push frame and allocate
        mem.push_frame();
        let ptr = mem.alloc_stack(10, 4, Some("local".to_string())).unwrap();
        assert!(mem.is_valid(ptr));

        // Pop frame - allocation should be invalidated
        mem.pop_frame();
        assert!(!mem.is_valid(ptr));
    }

    #[test]
    fn test_load_store_primitives() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(16, 8).unwrap();

        // u32
        mem.store_u32(ptr, 0x1234_5678).unwrap();
        assert_eq!(mem.load_u32(ptr).unwrap(), 0x1234_5678);

        // i32 negative
        mem.store_i32(ptr, -42).unwrap();
        assert_eq!(mem.load_i32(ptr).unwrap(), -42);

        // u64
        let ptr8 = ptr.offset(8).unwrap();
        mem.store_u64(ptr8, 0xDEAD_BEEF_CAFE_BABE).unwrap();
        assert_eq!(mem.load_u64(ptr8).unwrap(), 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn test_load_store_pointer() {
        let mut mem = Memory::new();
        // Pointer storage is 8 bytes (4 for block_id + 4 for offset)
        let ptr = mem.alloc(8, 8).unwrap();
        let target = Pointer::with_offset(BlockId(42), 100);

        mem.store_ptr(ptr, target).unwrap();
        let loaded = mem.load_ptr(ptr).unwrap();

        assert_eq!(loaded.block.0, 42);
        assert_eq!(loaded.offset, 100);
    }

    #[test]
    fn test_load_store_pointer_max_valid_values() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(8, 8).unwrap();
        // Test with maximum values that fit in u32/i32
        let max_block_id = u64::from(u32::MAX);
        let max_offset = i32::MAX as i64;
        let target = Pointer::with_offset(BlockId(max_block_id), max_offset);

        mem.store_ptr(ptr, target).unwrap();
        let loaded = mem.load_ptr(ptr).unwrap();

        assert_eq!(loaded.block.0, max_block_id);
        assert_eq!(loaded.offset, max_offset);
    }

    #[test]
    fn test_load_store_pointer_negative_offset() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(8, 8).unwrap();
        // Test with negative offset (should be sign-extended on load)
        let target = Pointer::with_offset(BlockId(1), -100);

        mem.store_ptr(ptr, target).unwrap();
        let loaded = mem.load_ptr(ptr).unwrap();

        assert_eq!(loaded.block.0, 1);
        assert_eq!(loaded.offset, -100);
    }

    #[test]
    fn test_store_pointer_overflow_block_id() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(8, 8).unwrap();
        // Block ID that exceeds u32::MAX should fail
        let large_block_id = u64::MAX / 2;
        let target = Pointer::with_offset(BlockId(large_block_id), 0);

        let result = mem.store_ptr(ptr, target);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_pointer_overflow_offset() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(8, 8).unwrap();
        // Offset that exceeds i32 range should fail
        let large_offset = i64::MAX / 2;
        let target = Pointer::with_offset(BlockId(1), large_offset);

        let result = mem.store_ptr(ptr, target);
        assert!(result.is_err());
    }

    #[test]
    fn test_memcpy_no_overlap() {
        let mut mem = Memory::new();
        let src = mem.alloc(10, 1).unwrap();
        let dst = mem.alloc(10, 1).unwrap();

        mem.store_bytes(src, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .unwrap();
        mem.memcpy(dst, src, 10).unwrap();

        assert_eq!(
            mem.load_bytes(dst, 10).unwrap(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
    }

    #[test]
    fn test_memcpy_overlap_error() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(20, 1).unwrap();

        // Overlapping regions in same block
        let src = ptr;
        let dst = ptr.offset(5).unwrap();

        mem.store_bytes(src, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .unwrap();
        let result = mem.memcpy(dst, src, 10);

        assert!(matches!(result, Err(UBKind::OverlappingMemcpy)));
    }

    #[test]
    fn test_memmove_overlap_ok() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(20, 1).unwrap();

        // Overlapping regions in same block - memmove handles this
        let src = ptr;
        let dst = ptr.offset(5).unwrap();

        mem.store_bytes(src, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .unwrap();
        mem.memmove(dst, src, 10).unwrap();

        // memmove should work (copies via temp buffer)
        assert_eq!(
            mem.load_bytes(dst, 10).unwrap(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
    }

    #[test]
    fn test_memset() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(10, 1).unwrap();

        mem.memset(ptr, 0xAB, 10).unwrap();
        assert_eq!(mem.load_bytes(ptr, 10).unwrap(), vec![0xAB; 10]);
    }

    #[test]
    fn test_memcmp() {
        let mut mem = Memory::new();
        let ptr1 = mem.alloc(5, 1).unwrap();
        let ptr2 = mem.alloc(5, 1).unwrap();

        mem.store_bytes(ptr1, &[1, 2, 3, 4, 5]).unwrap();
        mem.store_bytes(ptr2, &[1, 2, 3, 4, 5]).unwrap();

        // Equal
        assert_eq!(mem.memcmp(ptr1, ptr2, 5).unwrap(), 0);

        // Different
        mem.store_bytes(ptr2, &[1, 2, 4, 4, 5]).unwrap();
        assert!(mem.memcmp(ptr1, ptr2, 5).unwrap() < 0); // 3 < 4
    }

    #[test]
    fn test_ptr_cmp() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(100, 1).unwrap();
        let ptr_50 = ptr.offset(50).unwrap();
        let ptr_100 = ptr.offset(100).unwrap();

        assert_eq!(mem.ptr_cmp(ptr, ptr), PtrCmpResult::Equal);
        assert_eq!(mem.ptr_cmp(ptr, ptr_50), PtrCmpResult::Less);
        assert_eq!(mem.ptr_cmp(ptr_100, ptr_50), PtrCmpResult::Greater);

        // Different blocks
        let ptr2 = mem.alloc(100, 1).unwrap();
        assert_eq!(mem.ptr_cmp(ptr, ptr2), PtrCmpResult::Undefined);

        // NULL
        assert_eq!(
            mem.ptr_cmp(Pointer::null(), Pointer::null()),
            PtrCmpResult::Equal
        );
    }

    #[test]
    fn test_free_null() {
        let mut mem = Memory::new();
        // free(NULL) is a no-op
        assert!(mem.free(Pointer::null()).is_ok());
    }

    #[test]
    fn test_free_at_offset() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(100, 8).unwrap();
        let ptr_plus_10 = ptr.offset(10).unwrap();

        // Can't free at offset
        let result = mem.free(ptr_plus_10);
        assert!(matches!(result, Err(UBKind::InvalidFree)));
    }

    #[test]
    fn test_floats() {
        let mut mem = Memory::new();
        let ptr = mem.alloc(16, 8).unwrap();

        let test_f32 = std::f32::consts::PI;
        mem.store_f32(ptr, test_f32).unwrap();
        let f = mem.load_f32(ptr).unwrap();
        assert!((f - test_f32).abs() < 1e-5);

        let ptr8 = ptr.offset(8).unwrap();
        let test_f64 = std::f64::consts::E;
        mem.store_f64(ptr8, test_f64).unwrap();
        let d = mem.load_f64(ptr8).unwrap();
        assert!((d - test_f64).abs() < 1e-10);
    }
}
