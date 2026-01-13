//! Content-addressable hashing for proofs
//!
//! Uses BLAKE3 for fast, cryptographically secure hashing.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A content-addressable hash using BLAKE3
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Create a hash from bytes
    pub fn from_bytes(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self(*hash.as_bytes())
    }

    /// Create a hash from a string (hashes the string's bytes)
    pub fn hash_str(s: &str) -> Self {
        Self::from_bytes(s.as_bytes())
    }

    /// Create a hash from multiple components
    pub fn from_components(components: &[&[u8]]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for component in components {
            hasher.update(component);
        }
        Self(*hasher.finalize().as_bytes())
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse from hex string
    pub fn from_hex(s: &str) -> Result<Self, hex::FromHexError> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 32 {
            return Err(hex::FromHexError::InvalidStringLength);
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }

    /// Create a zero hash (for testing/placeholder)
    pub fn zero() -> Self {
        Self([0u8; 32])
    }

    /// Check if this is the zero hash
    pub fn is_zero(&self) -> bool {
        self.0 == [0u8; 32]
    }

    /// Get truncated hex for display (first 8 chars)
    pub fn short_hex(&self) -> String {
        self.to_hex()[..8].to_string()
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({})", self.short_hex())
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = self.to_hex();
        write!(f, "{hex}")
    }
}

/// Trait for types that can be content-addressed
pub trait ContentAddressable {
    /// Compute the content hash for this value
    fn content_hash(&self) -> ContentHash;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_from_bytes() {
        let hash = ContentHash::from_bytes(b"hello world");
        assert!(!hash.is_zero());
        assert_eq!(hash.to_hex().len(), 64);
    }

    #[test]
    fn test_hash_from_str() {
        let hash = ContentHash::hash_str("hello world");
        let hash2 = ContentHash::from_bytes(b"hello world");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_hash_from_components() {
        let hash1 = ContentHash::from_bytes(b"helloworld");
        let hash2 = ContentHash::from_components(&[b"hello", b"world"]);
        // Streaming components is equivalent to hashing the concatenated bytes
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_deterministic() {
        let hash1 = ContentHash::hash_str("test");
        let hash2 = ContentHash::hash_str("test");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let hash1 = ContentHash::hash_str("test1");
        let hash2 = ContentHash::hash_str("test2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_hex_roundtrip() {
        let hash = ContentHash::hash_str("test data");
        let hex = hash.to_hex();
        let parsed = ContentHash::from_hex(&hex).unwrap();
        assert_eq!(hash, parsed);
    }

    #[test]
    fn test_hash_short_hex() {
        let hash = ContentHash::hash_str("test");
        let short = hash.short_hex();
        assert_eq!(short.len(), 8);
        assert!(hash.to_hex().starts_with(&short));
    }

    #[test]
    fn test_zero_hash() {
        let zero = ContentHash::zero();
        assert!(zero.is_zero());
        assert_eq!(zero, ContentHash::default());
    }

    #[test]
    fn test_non_zero_hash() {
        let hash = ContentHash::hash_str("anything");
        assert!(!hash.is_zero());
    }

    #[test]
    fn test_hash_debug() {
        let hash = ContentHash::hash_str("test");
        let debug = format!("{:?}", hash);
        assert!(debug.starts_with("ContentHash("));
        assert!(debug.len() < 30); // Short form
    }

    #[test]
    fn test_hash_display() {
        let hash = ContentHash::hash_str("test");
        let display = format!("{}", hash);
        assert_eq!(display.len(), 64); // Full hex
    }

    #[test]
    fn test_hash_clone() {
        let hash = ContentHash::hash_str("clone test");
        let cloned = hash;
        assert_eq!(hash, cloned);
    }

    #[test]
    fn test_hash_serialize() {
        let hash = ContentHash::hash_str("serialize test");
        let json = serde_json::to_string(&hash).unwrap();
        let parsed: ContentHash = serde_json::from_str(&json).unwrap();
        assert_eq!(hash, parsed);
    }

    #[test]
    fn test_from_hex_invalid_length() {
        let result = ContentHash::from_hex("abcd");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_hex_invalid_chars() {
        let result = ContentHash::from_hex("zzzz");
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_as_bytes() {
        let hash = ContentHash::hash_str("bytes test");
        let bytes = hash.as_bytes();
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_hash_eq() {
        use std::collections::HashSet;
        let hash1 = ContentHash::hash_str("a");
        let hash2 = ContentHash::hash_str("a");
        let hash3 = ContentHash::hash_str("b");

        let mut set = HashSet::new();
        set.insert(hash1);
        assert!(set.contains(&hash2));
        assert!(!set.contains(&hash3));
    }
}
