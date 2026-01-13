//! Compacted region parsing
//!
//! Lean 4 stores objects in a "compacted region" - a contiguous memory region
//! with pointer-sharing and relocation support.
//!
//! # File Layout
//!
//! The .olean file has this structure:
//! - Offset 0-55: Header (56 bytes)
//! - Offset 56-63: Root object pointer (8 bytes)
//! - Offset 64+: Compacted region objects
//!
//! The `base_addr` in the header corresponds to file offset 0 when memory-mapped.
//! So a pointer P corresponds to file offset (P - base_addr).
//!
//! # Lean 4 Object Layout
//!
//! Each object has an 8-byte header:
//! ```text
//! struct lean_object {
//!     int      m_rc;       // 4 bytes: reference count (0 for compacted)
//!     unsigned m_cs_sz:16; // 2 bytes: compact size or region size
//!     unsigned m_other:8;  // 1 byte: num fields or element size
//!     unsigned m_tag:8;    // 1 byte: object type tag
//! }
//! ```
//!
//! # Object Tags
//!
//! Tags 0-243: Constructor objects (tag = constructor index)
//! Tag 244: Promise
//! Tag 245: Closure
//! Tag 246: Array
//! Tag 247: Struct Array (array of objects)
//! Tag 248: Scalar Array (array of primitives)
//! Tag 249: String
//! Tag 250: MPZ (big integer)
//! Tag 251: Thunk
//! Tag 252: Task
//! Tag 253: Ref
//! Tag 254: External
//! Tag 255: Reserved
//!
//! # Tagged Pointers
//!
//! Lean uses tagged pointers for small scalars:
//! - If LSB is 1: It's a scalar value (unbox by shifting right 1)
//! - If LSB is 0 and non-null: It's an actual pointer
//! - Common special value: 1 = boxed 0 (e.g., Name.anonymous)
//!
//! # String Object Layout
//!
//! ```text
//! struct lean_string_object {
//!     lean_object m_header;   // 8 bytes
//!     size_t      m_size;     // 8 bytes: byte length including null
//!     size_t      m_capacity; // 8 bytes: buffer capacity
//!     size_t      m_length;   // 8 bytes: UTF-8 character count
//!     char        m_data[];   // Variable: actual string data
//! }
//! ```

use crate::error::{OleanError, OleanResult};

/// Object tags from Lean 4 runtime
pub mod tags {
    /// Maximum constructor tag (tags 0-243 are constructors)
    pub const MAX_CTOR_TAG: u8 = 243;
    pub const PROMISE: u8 = 244;
    pub const CLOSURE: u8 = 245;
    pub const ARRAY: u8 = 246;
    pub const STRUCT_ARRAY: u8 = 247;
    pub const SCALAR_ARRAY: u8 = 248;
    pub const STRING: u8 = 249;
    pub const MPZ: u8 = 250;
    pub const THUNK: u8 = 251;
    pub const TASK: u8 = 252;
    pub const REF: u8 = 253;
    pub const EXTERNAL: u8 = 254;
    pub const RESERVED: u8 = 255;
}

/// Size of the lean_object header
pub const OBJECT_HEADER_SIZE: usize = 8;

/// Object header from a compacted region
#[derive(Debug, Clone, Copy)]
pub struct ObjectHeader {
    /// Reference count (always 0 in compacted regions)
    pub rc: i32,
    /// Compact size or region size
    pub cs_sz: u16,
    /// Number of fields or element size
    pub other: u8,
    /// Object type tag
    pub tag: u8,
}

impl ObjectHeader {
    /// Parse an object header from bytes
    pub fn parse(bytes: &[u8]) -> OleanResult<Self> {
        if bytes.len() < OBJECT_HEADER_SIZE {
            return Err(OleanError::OutOfBounds {
                offset: 0,
                size: bytes.len(),
            });
        }

        let rc = i32::from_le_bytes(bytes[0..4].try_into().expect("slice length verified above"));
        let cs_sz =
            u16::from_le_bytes(bytes[4..6].try_into().expect("slice length verified above"));
        let other = bytes[6];
        let tag = bytes[7];

        Ok(Self {
            rc,
            cs_sz,
            other,
            tag,
        })
    }

    /// Check if this is a constructor object
    pub fn is_constructor(&self) -> bool {
        self.tag <= tags::MAX_CTOR_TAG
    }

    /// Check if this is a scalar (non-pointer) object
    pub fn is_scalar(&self) -> bool {
        matches!(self.tag, tags::SCALAR_ARRAY | tags::STRING | tags::MPZ)
    }

    /// Get the number of pointer fields for constructor objects
    pub fn num_fields(&self) -> usize {
        if self.is_constructor() {
            self.other as usize
        } else {
            0
        }
    }
}

/// A reference to an object within a compacted region
#[derive(Debug, Clone, Copy)]
pub struct ObjectRef {
    /// Offset within the region
    pub offset: usize,
    /// The object header
    pub header: ObjectHeader,
}

/// Check if a value is a tagged scalar (LSB = 1)
#[inline]
pub fn is_scalar(ptr: u64) -> bool {
    (ptr & 1) == 1
}

/// Unbox a tagged scalar value
#[inline]
pub fn unbox_scalar(ptr: u64) -> u64 {
    ptr >> 1
}

/// Check if a value is an actual pointer (not null, not scalar)
#[inline]
pub fn is_ptr(ptr: u64) -> bool {
    ptr != 0 && (ptr & 1) == 0
}

/// Parser for compacted regions (entire .olean file)
///
/// This parser operates on the entire .olean file, not just the region portion.
/// Offsets are file offsets, and pointers are converted using base_addr.
pub struct CompactedRegion<'a> {
    /// The raw bytes of the entire .olean file
    pub(crate) data: &'a [u8],
    /// Base address the region was compiled with (corresponds to file offset 0)
    base_addr: u64,
}

impl<'a> CompactedRegion<'a> {
    /// Create a new compacted region parser from the full .olean file
    ///
    /// Note: This takes the entire file bytes, not just the region portion.
    /// The base_addr corresponds to file offset 0.
    pub fn new(data: &'a [u8], base_addr: u64) -> Self {
        Self { data, base_addr }
    }

    /// Get the size of the data in bytes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the region is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read an object header at a file offset
    pub fn read_header_at(&self, offset: usize) -> OleanResult<ObjectHeader> {
        if offset + OBJECT_HEADER_SIZE > self.data.len() {
            return Err(OleanError::OutOfBounds {
                offset,
                size: self.data.len(),
            });
        }

        ObjectHeader::parse(&self.data[offset..])
    }

    /// Read a u64 at a file offset
    pub fn read_u64_at(&self, offset: usize) -> OleanResult<u64> {
        if offset + 8 > self.data.len() {
            return Err(OleanError::OutOfBounds {
                offset,
                size: self.data.len(),
            });
        }

        Ok(u64::from_le_bytes(
            self.data[offset..offset + 8]
                .try_into()
                .expect("slice length verified above"),
        ))
    }

    /// Read an i32 at a file offset
    pub fn read_i32_at(&self, offset: usize) -> OleanResult<i32> {
        if offset + 4 > self.data.len() {
            return Err(OleanError::OutOfBounds {
                offset,
                size: self.data.len(),
            });
        }

        Ok(i32::from_le_bytes(
            self.data[offset..offset + 4]
                .try_into()
                .expect("slice length verified above"),
        ))
    }

    /// Read a pointer (u64) at a file offset (alias for read_u64_at)
    pub fn read_ptr_at(&self, offset: usize) -> OleanResult<u64> {
        self.read_u64_at(offset)
    }

    /// Convert a raw pointer to a file offset
    ///
    /// Pointers in the region are stored as absolute addresses based on base_addr.
    /// Since base_addr corresponds to file offset 0, ptr - base_addr = file offset.
    pub fn ptr_to_offset(&self, ptr: u64) -> OleanResult<usize> {
        if ptr == 0 {
            // Null pointer
            return Ok(0);
        }

        if is_scalar(ptr) {
            return Err(OleanError::InvalidPointer { ptr, offset: 0 });
        }

        if ptr < self.base_addr {
            return Err(OleanError::InvalidPointer { ptr, offset: 0 });
        }

        let offset = (ptr - self.base_addr) as usize;
        if offset >= self.data.len() {
            return Err(OleanError::InvalidPointer { ptr, offset });
        }

        Ok(offset)
    }

    /// Convert a file offset back to a pointer value
    pub fn offset_to_ptr(&self, offset: usize) -> u64 {
        self.base_addr + offset as u64
    }

    /// Get the raw bytes at a file offset
    pub fn bytes_at(&self, offset: usize, len: usize) -> OleanResult<&'a [u8]> {
        if offset + len > self.data.len() {
            return Err(OleanError::OutOfBounds {
                offset,
                size: self.data.len(),
            });
        }
        Ok(&self.data[offset..offset + len])
    }

    /// Read a Nat value from a pointer
    ///
    /// If the pointer is a tagged scalar, returns the unboxed value.
    /// If it's a pointer to an MPZ object, reads the big integer (currently limited to u64).
    pub fn read_nat_value(&self, ptr: u64) -> OleanResult<u64> {
        if is_scalar(ptr) {
            return Ok(unbox_scalar(ptr));
        }

        if !is_ptr(ptr) {
            return Ok(0);
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        // MPZ tag = 250
        if header.tag == tags::MPZ {
            // MPZ layout: header(8) + capacity(i32 at +8) + size(i32 at +12) + digits[...]
            // The size can be negative for negative numbers
            // For simplicity, we only handle small positive MPZ values
            let size_raw = self.read_i32_at(offset + 12)?;
            let size = size_raw.unsigned_abs() as usize;

            if size == 0 {
                return Ok(0);
            }

            // Read the first limb (lowest 64 bits)
            let limb = self.read_u64_at(offset + 16)?;
            if size == 1 {
                return Ok(limb);
            }

            // For multi-limb MPZ, we can't fit in u64 - return the low bits
            // This is a limitation, but BVar indices should never be this large
            Ok(limb)
        } else {
            // Not an MPZ - might be a different Nat representation
            // Return the raw value (likely an error case)
            Err(OleanError::Region(format!(
                "unexpected object tag {} for Nat at offset {}",
                header.tag, offset
            )))
        }
    }

    /// Read a Lean String object at a file offset
    ///
    /// Returns the string content (without null terminator).
    pub fn read_lean_string_at(&self, offset: usize) -> OleanResult<&'a str> {
        let header = self.read_header_at(offset)?;
        if header.tag != tags::STRING {
            return Err(OleanError::InvalidObjectTag {
                tag: header.tag,
                offset,
            });
        }

        // String layout: header(8) + size(8) + capacity(8) + length(8) + data
        if offset + 32 > self.data.len() {
            return Err(OleanError::OutOfBounds {
                offset,
                size: self.data.len(),
            });
        }

        let m_size = self.read_u64_at(offset + 8)? as usize;
        // m_capacity at offset + 16
        // m_length at offset + 24

        let data_start = offset + 32;
        if data_start + m_size > self.data.len() {
            return Err(OleanError::OutOfBounds {
                offset: data_start,
                size: self.data.len(),
            });
        }

        // String data (exclude null terminator)
        let str_len = if m_size > 0 { m_size - 1 } else { 0 };
        let bytes = &self.data[data_start..data_start + str_len];
        std::str::from_utf8(bytes).map_err(|_| OleanError::Region("invalid UTF-8 in string".into()))
    }

    /// Read a Lean Name object at a file offset
    ///
    /// Returns the fully qualified name as a string (e.g., "Nat.add").
    pub fn read_name_at(&self, offset: usize) -> OleanResult<String> {
        self.read_name_at_depth(offset, 0)
    }

    fn read_name_at_depth(&self, offset: usize, depth: usize) -> OleanResult<String> {
        if depth > 100 {
            return Err(OleanError::Region("Name depth limit exceeded".into()));
        }

        let header = self.read_header_at(offset)?;

        match (header.tag, header.other) {
            // Name.anonymous (constructor 0, 0 fields)
            (0, 0) => Ok(String::new()),

            // Name.str (constructor 1, 2 fields: parent, string)
            (1, 2) => {
                let parent_ptr = self.read_u64_at(offset + 8)?;
                let string_ptr = self.read_u64_at(offset + 16)?;

                // Read parent name
                let parent = if is_scalar(parent_ptr) {
                    // Scalar 0 = Name.anonymous
                    String::new()
                } else if is_ptr(parent_ptr) {
                    let parent_off = self.ptr_to_offset(parent_ptr)?;
                    self.read_name_at_depth(parent_off, depth + 1)?
                } else {
                    String::new()
                };

                // Read string component
                let component = if is_ptr(string_ptr) {
                    let str_off = self.ptr_to_offset(string_ptr)?;
                    self.read_lean_string_at(str_off)?
                } else {
                    "<invalid>"
                };

                if parent.is_empty() {
                    Ok(component.to_string())
                } else {
                    Ok(format!("{parent}.{component}"))
                }
            }

            // Name.num (constructor 2, 2 fields: parent, number)
            (2, 2) => {
                let parent_ptr = self.read_u64_at(offset + 8)?;
                let num = self.read_u64_at(offset + 16)?;

                // Read parent name
                let parent = if is_scalar(parent_ptr) {
                    String::new()
                } else if is_ptr(parent_ptr) {
                    let parent_off = self.ptr_to_offset(parent_ptr)?;
                    self.read_name_at_depth(parent_off, depth + 1)?
                } else {
                    String::new()
                };

                if parent.is_empty() {
                    Ok(num.to_string())
                } else {
                    Ok(format!("{parent}.{num}"))
                }
            }

            _ => Err(OleanError::InvalidObjectTag {
                tag: header.tag,
                offset,
            }),
        }
    }

    /// Find all Name.str objects in the file and return their names
    ///
    /// This is useful for exploring the .olean contents.
    pub fn find_all_names(&self) -> Vec<(usize, String)> {
        let mut names = Vec::new();

        // Start at offset 64 (after header + root pointer)
        let mut offset = 64;
        while offset + 24 < self.data.len() {
            if let Ok(header) = self.read_header_at(offset) {
                if header.tag == 1 && header.other == 2 {
                    // This is a Name.str
                    if let Ok(name) = self.read_name_at(offset) {
                        if !name.is_empty() {
                            names.push((offset, name));
                        }
                    }
                }
            }
            offset += 8;
        }

        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_header_parse() {
        // Construct a constructor object header
        // rc=0, cs_sz=0x20, other=2 (2 fields), tag=0 (first constructor)
        let bytes = [
            0, 0, 0, 0, // rc = 0
            0x20, 0, // cs_sz = 0x20
            2, // other = 2
            0, // tag = 0
        ];

        let header = ObjectHeader::parse(&bytes).unwrap();
        assert_eq!(header.rc, 0);
        assert_eq!(header.cs_sz, 0x20);
        assert_eq!(header.other, 2);
        assert_eq!(header.tag, 0);
        assert!(header.is_constructor());
        assert_eq!(header.num_fields(), 2);
    }

    #[test]
    fn test_object_header_string() {
        // String object header
        let bytes = [
            0,
            0,
            0,
            0, // rc = 0
            0,
            0,            // cs_sz = 0
            5,            // other = 5 (length?)
            tags::STRING, // tag = STRING
        ];

        let header = ObjectHeader::parse(&bytes).unwrap();
        assert_eq!(header.tag, tags::STRING);
        assert!(!header.is_constructor());
        assert!(header.is_scalar());
    }

    #[test]
    fn test_scalar_detection() {
        // Scalar values have LSB = 1
        assert!(is_scalar(1)); // boxed 0 (Name.anonymous)
        assert!(is_scalar(3)); // boxed 1
        assert!(is_scalar(0xFF)); // odd number

        // Pointers have LSB = 0 (and non-zero)
        assert!(is_ptr(0x0010_0000));
        assert!(is_ptr(2));

        // Zero is neither scalar nor valid pointer
        assert!(!is_scalar(0));
        assert!(!is_ptr(0));
    }

    #[test]
    fn test_unbox_scalar() {
        assert_eq!(unbox_scalar(1), 0);
        assert_eq!(unbox_scalar(3), 1);
        assert_eq!(unbox_scalar(5), 2);
        assert_eq!(unbox_scalar(201), 100);
    }

    #[test]
    fn test_compacted_region_string() {
        // Create a mock .olean file structure with a string
        let base_addr = 0x1000u64;

        // We need at least 64 bytes header + object
        let mut data = vec![0u8; 128];

        // String object at offset 64 (first object after header)
        let str_offset = 64;
        // Header
        data[str_offset..str_offset + 4].copy_from_slice(&0i32.to_le_bytes()); // rc
        data[str_offset + 4..str_offset + 6].copy_from_slice(&0u16.to_le_bytes()); // cs_sz
        data[str_offset + 6] = 0; // other
        data[str_offset + 7] = tags::STRING;
        // m_size = 6 (5 chars + null)
        data[str_offset + 8..str_offset + 16].copy_from_slice(&6u64.to_le_bytes());
        // m_capacity = 6
        data[str_offset + 16..str_offset + 24].copy_from_slice(&6u64.to_le_bytes());
        // m_length = 5
        data[str_offset + 24..str_offset + 32].copy_from_slice(&5u64.to_le_bytes());
        // String data "hello\0"
        data[str_offset + 32..str_offset + 38].copy_from_slice(b"hello\0");

        let region = CompactedRegion::new(&data, base_addr);
        let s = region.read_lean_string_at(str_offset).unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn test_ptr_to_offset() {
        let base_addr = 0x0010_0000_u64;
        let data = vec![0u8; 100];
        let region = CompactedRegion::new(&data, base_addr);

        // Null pointer
        assert_eq!(region.ptr_to_offset(0).unwrap(), 0);

        // Valid pointer
        assert_eq!(region.ptr_to_offset(base_addr + 50).unwrap(), 50);

        // Pointer before base is invalid
        assert!(region.ptr_to_offset(base_addr - 1).is_err());

        // Pointer beyond region is invalid
        assert!(region.ptr_to_offset(base_addr + 200).is_err());

        // Scalar values are not valid pointers
        assert!(region.ptr_to_offset(1).is_err()); // boxed 0
        assert!(region.ptr_to_offset(3).is_err()); // boxed 1
    }
}
