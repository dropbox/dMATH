//! Error types for .olean parsing

use thiserror::Error;

/// Result type for .olean operations
pub type OleanResult<T> = Result<T, OleanError>;

/// Errors that can occur while parsing .olean files
#[derive(Debug, Error)]
pub enum OleanError {
    /// File is too small to contain a valid header
    #[error("file too small: expected at least {expected} bytes, got {actual}")]
    FileTooSmall { expected: usize, actual: usize },

    /// Invalid magic bytes (not an .olean file)
    #[error("invalid magic bytes: expected 'olean', got {0:?}")]
    InvalidMagic([u8; 5]),

    /// Unsupported .olean format version
    #[error("unsupported version: expected {expected}, got {actual}")]
    UnsupportedVersion { expected: u8, actual: u8 },

    /// Git hash contains invalid characters
    #[error("invalid git hash: {0}")]
    InvalidGitHash(String),

    /// I/O error reading the file
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing compacted region
    #[error("region error: {0}")]
    Region(String),

    /// Error serializing or deserializing auxiliary data
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Invalid object tag in compacted region
    #[error("invalid object tag: {tag} at offset {offset}")]
    InvalidObjectTag { tag: u8, offset: usize },

    /// Invalid pointer in compacted region
    #[error("invalid pointer: 0x{ptr:x} at offset {offset}")]
    InvalidPointer { ptr: u64, offset: usize },

    /// Object extends beyond region bounds
    #[error("object at offset {offset} extends beyond region (size {size})")]
    OutOfBounds { offset: usize, size: usize },

    /// Unsupported Lean5 payload version
    #[error("unsupported Lean5 payload version: expected {expected}, got {actual}")]
    UnsupportedPayloadVersion { expected: u32, actual: u32 },

    /// Invalid Lean5 payload footer or length
    #[error("invalid Lean5 payload: {0}")]
    InvalidPayload(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_file_too_small_error_display() {
        let err = OleanError::FileTooSmall {
            expected: 100,
            actual: 50,
        };
        assert_eq!(
            err.to_string(),
            "file too small: expected at least 100 bytes, got 50"
        );
    }

    #[test]
    fn test_invalid_magic_error_display() {
        let err = OleanError::InvalidMagic([0x00, 0x01, 0x02, 0x03, 0x04]);
        assert!(err.to_string().contains("invalid magic bytes"));
    }

    #[test]
    fn test_unsupported_version_error_display() {
        let err = OleanError::UnsupportedVersion {
            expected: 1,
            actual: 2,
        };
        assert_eq!(err.to_string(), "unsupported version: expected 1, got 2");
    }

    #[test]
    fn test_invalid_git_hash_error_display() {
        let err = OleanError::InvalidGitHash("not-a-valid-hash".to_string());
        assert_eq!(err.to_string(), "invalid git hash: not-a-valid-hash");
    }

    #[test]
    fn test_region_error_display() {
        let err = OleanError::Region("failed to parse object".to_string());
        assert_eq!(err.to_string(), "region error: failed to parse object");
    }

    #[test]
    fn test_invalid_object_tag_error_display() {
        let err = OleanError::InvalidObjectTag {
            tag: 255,
            offset: 0x100,
        };
        assert_eq!(err.to_string(), "invalid object tag: 255 at offset 256");
    }

    #[test]
    fn test_invalid_pointer_error_display() {
        let err = OleanError::InvalidPointer {
            ptr: 0xDEAD_BEEF,
            offset: 64,
        };
        assert_eq!(err.to_string(), "invalid pointer: 0xdeadbeef at offset 64");
    }

    #[test]
    fn test_out_of_bounds_error_display() {
        let err = OleanError::OutOfBounds {
            offset: 1000,
            size: 512,
        };
        assert_eq!(
            err.to_string(),
            "object at offset 1000 extends beyond region (size 512)"
        );
    }

    #[test]
    fn test_io_error_from_trait() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let olean_err: OleanError = io_err.into();
        assert!(olean_err.to_string().contains("I/O error"));
        assert!(olean_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_io_error_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let olean_err: OleanError = io_err.into();
        // Verify the error has a source (the underlying io::Error)
        assert!(olean_err.source().is_some());
    }

    #[test]
    fn test_errors_implement_debug() {
        let err = OleanError::FileTooSmall {
            expected: 100,
            actual: 50,
        };
        // Verify Debug is implemented by formatting
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("FileTooSmall"));
    }

    #[test]
    fn test_olean_result_type() {
        #[allow(clippy::unnecessary_wraps)]
        fn returns_ok() -> OleanResult<i32> {
            Ok(42)
        }

        fn returns_err() -> OleanResult<i32> {
            Err(OleanError::Region("test error".to_string()))
        }

        assert_eq!(returns_ok().unwrap(), 42);
        assert!(returns_err().is_err());
    }
}
