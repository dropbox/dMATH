//! .olean file header parsing
//!
//! The header is exactly 56 bytes:
//! - 5 bytes: "olean" magic
//! - 1 byte: version (currently 1)
//! - 42 bytes: git hash (40 chars + 2 null padding)
//! - 8 bytes: base address (little-endian u64)

use crate::error::{OleanError, OleanResult};

/// Magic bytes at the start of every .olean file
pub const MAGIC: &[u8; 5] = b"olean";

/// Current supported version
pub const VERSION: u8 = 1;

/// Size of the header in bytes
pub const HEADER_SIZE: usize = 56;

/// Size of the git hash field (includes null padding)
const GIT_HASH_FIELD_SIZE: usize = 42;

/// Size of actual git hash (40 hex characters)
const GIT_HASH_LEN: usize = 40;

/// Parsed .olean file header
#[derive(Debug, Clone)]
pub struct OleanHeader {
    /// Magic bytes (should be "olean")
    pub magic: [u8; 5],

    /// Format version
    pub version: u8,

    /// Git commit hash of the Lean build (40 chars, null-padded to 42)
    pub git_hash: [u8; GIT_HASH_FIELD_SIZE],

    /// Base address for memory mapping
    /// The compacted region was serialized with pointers relative to this address
    pub base_addr: u64,
}

impl OleanHeader {
    /// Parse an .olean header from bytes
    ///
    /// Returns error if:
    /// - File is too small
    /// - Magic bytes don't match
    /// - Version is unsupported
    pub fn parse(bytes: &[u8]) -> OleanResult<Self> {
        // Check minimum size
        if bytes.len() < HEADER_SIZE {
            return Err(OleanError::FileTooSmall {
                expected: HEADER_SIZE,
                actual: bytes.len(),
            });
        }

        // Parse magic
        let mut magic = [0u8; 5];
        magic.copy_from_slice(&bytes[0..5]);

        if &magic != MAGIC {
            return Err(OleanError::InvalidMagic(magic));
        }

        // Parse version
        let version = bytes[5];
        if version != VERSION {
            return Err(OleanError::UnsupportedVersion {
                expected: VERSION,
                actual: version,
            });
        }

        // Parse git hash
        let mut git_hash = [0u8; GIT_HASH_FIELD_SIZE];
        git_hash.copy_from_slice(&bytes[6..48]);

        // Validate git hash (should be hex characters)
        for &b in &git_hash[..GIT_HASH_LEN] {
            if !b.is_ascii_hexdigit() {
                return Err(OleanError::InvalidGitHash(format!(
                    "non-hex character 0x{b:02x} in git hash"
                )));
            }
        }

        // Parse base address (little-endian)
        let base_addr = u64::from_le_bytes(
            bytes[48..56]
                .try_into()
                .expect("header size verified above"),
        );

        Ok(Self {
            magic,
            version,
            git_hash,
            base_addr,
        })
    }

    /// Get the git hash as a string (40 characters)
    pub fn git_hash_str(&self) -> &str {
        // The first 40 bytes are the hash, rest is null padding
        std::str::from_utf8(&self.git_hash[..GIT_HASH_LEN]).unwrap_or("<invalid utf8>")
    }

    /// Get the short git hash (first 12 characters, like Lean displays)
    pub fn git_hash_short(&self) -> &str {
        &self.git_hash_str()[..12]
    }

    /// Check if this header is compatible with a given git hash
    ///
    /// Returns true if the first `len` characters match
    pub fn matches_git_hash(&self, hash: &str) -> bool {
        let our_hash = self.git_hash_str();
        let len = hash.len().min(GIT_HASH_LEN);
        our_hash[..len] == hash[..len]
    }

    /// Create a new header with the given git hash and base address
    pub fn new(git_hash: &str, base_addr: u64) -> OleanResult<Self> {
        if git_hash.len() != GIT_HASH_LEN {
            return Err(OleanError::InvalidGitHash(format!(
                "expected {} characters, got {}",
                GIT_HASH_LEN,
                git_hash.len()
            )));
        }

        for c in git_hash.chars() {
            if !c.is_ascii_hexdigit() {
                return Err(OleanError::InvalidGitHash(format!(
                    "non-hex character '{c}' in git hash"
                )));
            }
        }

        let mut git_hash_bytes = [0u8; GIT_HASH_FIELD_SIZE];
        git_hash_bytes[..GIT_HASH_LEN].copy_from_slice(git_hash.as_bytes());

        Ok(Self {
            magic: *MAGIC,
            version: VERSION,
            git_hash: git_hash_bytes,
            base_addr,
        })
    }

    /// Serialize the header to bytes
    pub fn serialize(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];

        // Magic
        bytes[0..5].copy_from_slice(&self.magic);

        // Version
        bytes[5] = self.version;

        // Git hash
        bytes[6..48].copy_from_slice(&self.git_hash);

        // Base address (little-endian)
        bytes[48..56].copy_from_slice(&self.base_addr.to_le_bytes());

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        // Verify our size constant matches the struct layout
        assert_eq!(HEADER_SIZE, 56);
        assert_eq!(GIT_HASH_FIELD_SIZE, 42);
    }

    #[test]
    fn test_parse_valid_header() {
        // Construct a valid header
        let mut bytes = vec![0u8; HEADER_SIZE];

        // Magic
        bytes[0..5].copy_from_slice(b"olean");

        // Version
        bytes[5] = 1;

        // Git hash (40 hex chars + 2 null)
        bytes[6..46].copy_from_slice(b"0123456789abcdef0123456789abcdef01234567");
        bytes[46] = 0;
        bytes[47] = 0;

        // Base address (0x1234567890abcdef little-endian)
        bytes[48..56].copy_from_slice(&0x1234_5678_90ab_cdef_u64.to_le_bytes());

        let header = OleanHeader::parse(&bytes).expect("Should parse valid header");

        assert_eq!(header.magic, *b"olean");
        assert_eq!(header.version, 1);
        assert_eq!(
            header.git_hash_str(),
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert_eq!(header.git_hash_short(), "0123456789ab");
        assert_eq!(header.base_addr, 0x1234_5678_90ab_cdef);

        assert!(header.matches_git_hash("0123456789ab"));
        assert!(header.matches_git_hash("0123456789abcdef0123456789abcdef01234567"));
        assert!(!header.matches_git_hash("aaaaaaaaaa"));
    }

    #[test]
    fn test_parse_too_small() {
        let bytes = vec![0u8; 10];
        let err = OleanHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, OleanError::FileTooSmall { .. }));
    }

    #[test]
    fn test_parse_invalid_magic() {
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..5].copy_from_slice(b"notok");

        let err = OleanHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, OleanError::InvalidMagic(_)));
    }

    #[test]
    fn test_parse_invalid_version() {
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..5].copy_from_slice(b"olean");
        bytes[5] = 99; // Invalid version

        let err = OleanHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, OleanError::UnsupportedVersion { .. }));
    }

    #[test]
    fn test_parse_invalid_git_hash() {
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..5].copy_from_slice(b"olean");
        bytes[5] = 1;
        bytes[6..46].copy_from_slice(b"not_a_valid_hex_hash!!!!!!!!!!!!!!!!!!!!"); // Invalid

        let err = OleanHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, OleanError::InvalidGitHash(_)));
    }

    fn make_header_with_hash(hash: &str) -> OleanHeader {
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..5].copy_from_slice(b"olean");
        bytes[5] = 1;
        bytes[6..46].copy_from_slice(hash.as_bytes());
        bytes[46] = 0;
        bytes[47] = 0;
        bytes[48..56].copy_from_slice(&0u64.to_le_bytes());
        OleanHeader::parse(&bytes).unwrap()
    }

    #[test]
    fn test_git_hash_short_returns_12_chars() {
        let header = make_header_with_hash("abcdef1234567890abcdef1234567890abcdef12");
        let short = header.git_hash_short();
        assert_eq!(short.len(), 12);
        assert_eq!(short, "abcdef123456");
    }

    #[test]
    fn test_git_hash_str_returns_40_chars() {
        let hash = "0123456789abcdef0123456789abcdef01234567";
        let header = make_header_with_hash(hash);
        assert_eq!(header.git_hash_str().len(), 40);
        assert_eq!(header.git_hash_str(), hash);
    }

    #[test]
    fn test_matches_git_hash_empty_string() {
        let header = make_header_with_hash("0123456789abcdef0123456789abcdef01234567");
        // Empty string should match anything (vacuous truth)
        assert!(header.matches_git_hash(""));
    }

    #[test]
    fn test_matches_git_hash_partial_prefix() {
        let header = make_header_with_hash("0123456789abcdef0123456789abcdef01234567");
        assert!(header.matches_git_hash("0"));
        assert!(header.matches_git_hash("01"));
        assert!(header.matches_git_hash("0123"));
        assert!(header.matches_git_hash("0123456789ab"));
        assert!(!header.matches_git_hash("1"));
        assert!(!header.matches_git_hash("abcdef"));
    }

    #[test]
    fn test_matches_git_hash_full_match() {
        let hash = "0123456789abcdef0123456789abcdef01234567";
        let header = make_header_with_hash(hash);
        assert!(header.matches_git_hash(hash));
    }

    #[test]
    fn test_matches_git_hash_longer_than_stored() {
        let header = make_header_with_hash("0123456789abcdef0123456789abcdef01234567");
        // If input is longer than 40 chars, only compare first 40
        let long_hash = "0123456789abcdef0123456789abcdef01234567extra";
        assert!(header.matches_git_hash(long_hash));
    }

    #[test]
    fn test_matches_git_hash_case_sensitive() {
        let header = make_header_with_hash("abcdef1234567890abcdef1234567890abcdef12");
        // Hash matching is case sensitive
        assert!(header.matches_git_hash("abcdef"));
        assert!(!header.matches_git_hash("ABCDEF"));
    }

    #[test]
    fn test_git_hash_all_zeros() {
        let header = make_header_with_hash("0000000000000000000000000000000000000000");
        assert_eq!(
            header.git_hash_str(),
            "0000000000000000000000000000000000000000"
        );
        assert_eq!(header.git_hash_short(), "000000000000");
        assert!(header.matches_git_hash("0000"));
    }

    #[test]
    fn test_git_hash_all_fs() {
        let header = make_header_with_hash("ffffffffffffffffffffffffffffffffffffffff");
        assert_eq!(
            header.git_hash_str(),
            "ffffffffffffffffffffffffffffffffffffffff"
        );
        assert_eq!(header.git_hash_short(), "ffffffffffff");
        assert!(header.matches_git_hash("ffff"));
    }

    #[test]
    fn test_header_new() {
        let hash = "0123456789abcdef0123456789abcdef01234567";
        let header = OleanHeader::new(hash, 0x1000).unwrap();

        assert_eq!(header.magic, *b"olean");
        assert_eq!(header.version, 1);
        assert_eq!(header.git_hash_str(), hash);
        assert_eq!(header.base_addr, 0x1000);
    }

    #[test]
    fn test_header_new_invalid_length() {
        let err = OleanHeader::new("tooshort", 0x1000).unwrap_err();
        assert!(matches!(err, OleanError::InvalidGitHash(_)));
    }

    #[test]
    fn test_header_new_invalid_char() {
        let err = OleanHeader::new("0123456789abcdef0123456789abcdef0123456g", 0x1000).unwrap_err();
        assert!(matches!(err, OleanError::InvalidGitHash(_)));
    }

    #[test]
    fn test_header_serialize_roundtrip() {
        let hash = "0123456789abcdef0123456789abcdef01234567";
        let original = OleanHeader::new(hash, 0x1234_5678_90ab_cdef).unwrap();
        let bytes = original.serialize();

        assert_eq!(bytes.len(), HEADER_SIZE);

        let parsed = OleanHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.magic, original.magic);
        assert_eq!(parsed.version, original.version);
        assert_eq!(parsed.git_hash, original.git_hash);
        assert_eq!(parsed.base_addr, original.base_addr);
    }

    #[test]
    fn test_header_serialize_format() {
        let hash = "abcdef1234567890abcdef1234567890abcdef12";
        let header = OleanHeader::new(hash, 0x0100).unwrap();
        let bytes = header.serialize();

        // Check magic
        assert_eq!(&bytes[0..5], b"olean");
        // Check version
        assert_eq!(bytes[5], 1);
        // Check git hash
        assert_eq!(&bytes[6..46], hash.as_bytes());
        // Check null padding
        assert_eq!(bytes[46], 0);
        assert_eq!(bytes[47], 0);
        // Check base address (0x0100 in little-endian)
        assert_eq!(
            &bytes[48..56],
            &[0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }
}
