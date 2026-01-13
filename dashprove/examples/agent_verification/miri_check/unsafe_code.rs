//! Example unsafe code for MIRI verification
//!
//! This module contains intentionally unsafe patterns that MIRI can detect.
//! Run with: dashprove miri --file examples/agent_verification/miri_check/unsafe_code.rs

use std::ptr;

/// A simple buffer that demonstrates memory safety issues MIRI can detect.
pub struct StreamingBuffer {
    data: Vec<u8>,
    read_pos: usize,
    write_pos: usize,
}

impl StreamingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0; capacity],
            read_pos: 0,
            write_pos: 0,
        }
    }

    /// Write data to the buffer.
    pub fn write(&mut self, bytes: &[u8]) -> usize {
        let available = self.data.len() - self.write_pos;
        let to_write = bytes.len().min(available);

        // Safe write using slice copy
        self.data[self.write_pos..self.write_pos + to_write]
            .copy_from_slice(&bytes[..to_write]);
        self.write_pos += to_write;
        to_write
    }

    /// Read data from the buffer.
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        let available = self.write_pos - self.read_pos;
        let to_read = buf.len().min(available);

        buf[..to_read].copy_from_slice(&self.data[self.read_pos..self.read_pos + to_read]);
        self.read_pos += to_read;
        to_read
    }

    /// UNSAFE: Raw pointer manipulation that MIRI will check.
    /// This is safe but uses unsafe for demonstration.
    pub unsafe fn write_unchecked(&mut self, bytes: &[u8]) {
        let dst = self.data.as_mut_ptr().add(self.write_pos);
        ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        self.write_pos += bytes.len();
    }

    /// UNSAFE: Potential out-of-bounds access if not careful.
    /// MIRI will detect if this goes out of bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> u8 {
        *self.data.as_ptr().add(index)
    }
}

/// Demonstrates use-after-free that MIRI will detect.
/// DO NOT USE IN PRODUCTION - this is intentionally buggy for testing.
#[cfg(test)]
mod dangerous_examples {
    /// This function has a potential use-after-free bug.
    /// MIRI will detect this.
    pub unsafe fn use_after_free_example() -> u8 {
        let data = Box::new([1u8, 2, 3, 4]);
        let ptr = data.as_ptr();
        drop(data); // data is freed here

        // BUG: Reading from freed memory
        // MIRI will catch this: "memory access to allocation that was freed"
        *ptr
    }

    /// Demonstrates uninitialized memory access.
    /// MIRI will detect reading uninitialized memory.
    pub unsafe fn uninitialized_read_example() -> u8 {
        let mut data: [u8; 4] = std::mem::MaybeUninit::uninit().assume_init();
        // BUG: Reading uninitialized memory
        data[0]
    }

    /// Demonstrates out-of-bounds access.
    /// MIRI will detect buffer overflow.
    pub unsafe fn out_of_bounds_example() -> u8 {
        let data = [1u8, 2, 3, 4];
        let ptr = data.as_ptr();
        // BUG: Reading past end of array
        *ptr.add(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test is safe and will pass under MIRI.
    #[test]
    fn test_safe_operations() {
        let mut buffer = StreamingBuffer::new(1024);

        // Safe write
        let written = buffer.write(b"Hello, World!");
        assert_eq!(written, 13);

        // Safe read
        let mut buf = [0u8; 13];
        let read = buffer.read(&mut buf);
        assert_eq!(read, 13);
        assert_eq!(&buf, b"Hello, World!");
    }

    /// This test uses unsafe code correctly and will pass under MIRI.
    #[test]
    fn test_safe_unsafe_operations() {
        let mut buffer = StreamingBuffer::new(1024);

        unsafe {
            buffer.write_unchecked(b"Test");
            assert_eq!(buffer.get_unchecked(0), b'T');
            assert_eq!(buffer.get_unchecked(1), b'e');
            assert_eq!(buffer.get_unchecked(2), b's');
            assert_eq!(buffer.get_unchecked(3), b't');
        }
    }

    /// This test would trigger MIRI's use-after-free detection.
    /// Uncomment to see MIRI catch the bug.
    // #[test]
    // fn test_use_after_free() {
    //     unsafe {
    //         let _ = dangerous_examples::use_after_free_example();
    //     }
    // }

    /// This test would trigger MIRI's uninitialized read detection.
    /// Uncomment to see MIRI catch the bug.
    // #[test]
    // fn test_uninitialized_read() {
    //     unsafe {
    //         let _ = dangerous_examples::uninitialized_read_example();
    //     }
    // }

    /// This test would trigger MIRI's out-of-bounds detection.
    /// Uncomment to see MIRI catch the bug.
    // #[test]
    // fn test_out_of_bounds() {
    //     unsafe {
    //         let _ = dangerous_examples::out_of_bounds_example();
    //     }
    // }
}
