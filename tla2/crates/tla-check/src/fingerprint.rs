//! TLC-compatible FP64 polynomial rolling hash (Rabin fingerprint)
//!
//! This module implements the same fingerprinting algorithm as TLC's FP64.java,
//! enabling incremental fingerprint computation. Instead of hashing entire values
//! from scratch, fingerprints can be extended one component at a time.
//!
//! This is critical for performance in specs like `bosco` where operations like
//! `[rcvd EXCEPT ![self] = newRcvd]` repeatedly modify large functions. With
//! incremental fingerprinting, we can cache the base fingerprint and only extend
//! with the modified entry, achieving O(1) instead of O(n) per modification.
//!
//! # Algorithm
//!
//! FP64 is a polynomial rolling hash over GF(2^64). It uses an irreducible
//! polynomial as the modulus and precomputes a 256-entry table for byte-at-a-time
//! extension. The specific polynomial is from TLC's FP64.java Polys[0].
//!
//! # Reference
//!
//! - TLC source: `tlatools/org.lamport.tlatools/src/tlc2/util/FP64.java`
//! - Original authors: Allan Heydon and Marc Najork (Compaq, 1999)

use num_bigint::BigInt;
use num_traits::ToPrimitive;

/// Irreducible polynomial used as the initial fingerprint value.
/// This is Polys[0] from TLC's FP64.java.
pub const FP64_INIT: u64 = 0x911498AE0E66BAD6;

/// Internal constants for table computation
const ONE: u64 = 0x8000000000000000;
const X63: u64 = 0x1;

/// Precomputed byte mod table for polynomial rolling hash.
/// This table is computed once at startup from the irreducible polynomial.
/// Each entry ByteModTable_7[b] represents the contribution of byte b to the fingerprint.
static BYTE_MOD_TABLE: std::sync::OnceLock<[u64; 256]> = std::sync::OnceLock::new();

/// Initialize and get the byte mod table.
/// Computed lazily on first access, then cached.
#[inline]
fn get_byte_mod_table() -> &'static [u64; 256] {
    BYTE_MOD_TABLE.get_or_init(|| compute_byte_mod_table(FP64_INIT))
}

/// Compute the ByteModTable_7 from TLC's FP64.java Init() method.
/// This precomputes the polynomial contributions for all 256 byte values.
fn compute_byte_mod_table(irred_poly: u64) -> [u64; 256] {
    // Maximum power needed: 127 - 7*8 = 71, so we need 72 entries
    const PLENGTH: usize = 72;
    let mut power_table = [0u64; PLENGTH];

    // Build power table: power_table[i] = x^i mod IrredPoly
    let mut t = ONE;
    for entry in power_table.iter_mut() {
        *entry = t;
        // t = t * x (multiplication in GF(2^64))
        let mask = if (t & X63) != 0 { irred_poly } else { 0 };
        t = (t >> 1) ^ mask;
    }

    // Compute ByteModTable_7: the 7th iteration of ByteModTable initialization
    // This is what TLC uses for byte-at-a-time extension
    let mut table = [0u64; 256];
    for (j, entry) in table.iter_mut().enumerate() {
        let mut v = 0u64;
        for k in 0..=7 {
            if (j & (1usize << k)) != 0 {
                v ^= power_table[127 - 7 * 8 - k];
            }
        }
        *entry = v;
    }
    table
}

/// Extend a fingerprint by one byte.
///
/// This is the core operation of FP64. It corresponds to TLC's:
/// ```java
/// fp = ((fp >>> 8) ^ (mod[(b ^ ((int)fp)) & 0xFF]));
/// ```
#[inline]
pub fn fp64_extend_byte(fp: u64, b: u8) -> u64 {
    let table = get_byte_mod_table();
    let idx = ((b as u64) ^ fp) as usize & 0xFF;
    (fp >> 8) ^ table[idx]
}

/// Extend a fingerprint by an i32 (4 bytes, little-endian).
/// Corresponds to TLC's `FP64.Extend(fp, int)`.
#[inline]
pub fn fp64_extend_i32(mut fp: u64, x: i32) -> u64 {
    let bytes = x.to_le_bytes();
    for &b in &bytes {
        fp = fp64_extend_byte(fp, b);
    }
    fp
}

/// Extend a fingerprint by an i64/long (8 bytes, little-endian).
/// Corresponds to TLC's `FP64.Extend(fp, long)`.
#[inline]
pub fn fp64_extend_i64(mut fp: u64, x: i64) -> u64 {
    let bytes = x.to_le_bytes();
    for &b in &bytes {
        fp = fp64_extend_byte(fp, b);
    }
    fp
}

/// Extend a fingerprint by a string.
/// Corresponds to TLC's `FP64.Extend(fp, String)`.
///
/// Note: TLC uses Java's charAt which returns UTF-16 code units.
/// For ASCII strings (common in TLA+), this is equivalent to bytes.
/// For non-ASCII, we use the same UTF-16 encoding as TLC.
#[inline]
pub fn fp64_extend_str(mut fp: u64, s: &str) -> u64 {
    // Iterate over UTF-16 code units to match Java's behavior
    for c in s.encode_utf16() {
        // Java's char is 16-bit, but FP64.Extend(fp, char) only uses lower 8 bits
        // Looking at TLC source: fp = ((fp >>> 8) ^ (mod[(((int)c) ^ ((int)fp)) & 0xFF]))
        // It masks with 0xFF, so only the low byte of each char is used
        fp = fp64_extend_byte(fp, (c & 0xFF) as u8);
    }
    fp
}

/// Extend a fingerprint by a BigInt.
///
/// TLC fingerprints integers by extending with their byte representation.
/// For consistency with IntValue.fingerPrint in TLC, we use the value as i32 when it fits,
/// otherwise as i64.
#[inline]
pub fn fp64_extend_bigint(fp: u64, n: &BigInt) -> u64 {
    // Try to fit in i32 first (matches TLC's IntValue which uses int)
    if let Some(i) = n.to_i32() {
        return fp64_extend_i32(fp, i);
    }
    // Fall back to i64 for larger values
    if let Some(i) = n.to_i64() {
        return fp64_extend_i64(fp, i);
    }
    // For very large BigInts, use the signed bytes representation
    let bytes = n.to_signed_bytes_le();
    let mut fp = fp;
    for &b in &bytes {
        fp = fp64_extend_byte(fp, b);
    }
    fp
}

/// TLC Value type constants for fingerprinting.
/// These match TLC's ValueConstants.java exactly.
///
/// When fingerprinting a value, we first extend with its type tag,
/// then with its contents. This ensures different value types with
/// the same content produce different fingerprints.
pub mod value_tags {
    pub const BOOLVALUE: i64 = 0;
    pub const INTVALUE: i64 = 1;
    pub const REALVALUE: i64 = 2;
    pub const STRINGVALUE: i64 = 3;
    pub const RECORDVALUE: i64 = 4;
    pub const SETENUMVALUE: i64 = 5;
    pub const SETPREDVALUE: i64 = 6;
    pub const TUPLEVALUE: i64 = 7;
    pub const FCNLAMBDAVALUE: i64 = 8;
    pub const FCNRCDVALUE: i64 = 9;
    pub const OPLAMBDAVALUE: i64 = 10;
    pub const OPRCDVALUE: i64 = 11;
    pub const METHODVALUE: i64 = 12;
    pub const SETOFFCNSVALUE: i64 = 13;
    pub const SETOFRCDSVALUE: i64 = 14;
    pub const SETOFTUPLESVALUE: i64 = 15;
    pub const SUBSETVALUE: i64 = 16;
    pub const SETDIFFVALUE: i64 = 17;
    pub const SETCAPVALUE: i64 = 18;
    pub const SETCUPVALUE: i64 = 19;
    pub const UNIONVALUE: i64 = 20;
    pub const MODELVALUE: i64 = 21;
    pub const USERVALUE: i64 = 22;
    pub const INTERVALVALUE: i64 = 23;
    pub const UNDEFVALUE: i64 = 24;
    pub const LAZYVALUE: i64 = 25;
    pub const DUMMYVALUE: i64 = 26;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_mod_table_initialization() {
        // Ensure the table initializes without panicking
        let table = get_byte_mod_table();
        // Table should have 256 entries
        assert_eq!(table.len(), 256);
        // Entry 0 should be 0 (no bits set)
        assert_eq!(table[0], 0);
    }

    #[test]
    fn test_fp64_extend_byte() {
        // Start with the initial fingerprint
        let fp = FP64_INIT;
        // Extending with 0 should change the fingerprint
        let fp2 = fp64_extend_byte(fp, 0);
        assert_ne!(fp, fp2);
        // Same input should give same output
        let fp3 = fp64_extend_byte(fp, 0);
        assert_eq!(fp2, fp3);
    }

    #[test]
    fn test_fp64_extend_deterministic() {
        // Fingerprinting should be deterministic
        let fp1 = fp64_extend_str(FP64_INIT, "hello");
        let fp2 = fp64_extend_str(FP64_INIT, "hello");
        assert_eq!(fp1, fp2);

        // Different strings should (usually) produce different fingerprints
        let fp3 = fp64_extend_str(FP64_INIT, "world");
        assert_ne!(fp1, fp3);
    }

    #[test]
    fn test_fp64_extend_i64() {
        let fp = FP64_INIT;
        // Extending with the same value twice should give the same result
        let fp1 = fp64_extend_i64(fp, 42);
        let fp2 = fp64_extend_i64(fp, 42);
        assert_eq!(fp1, fp2);

        // Different values should give different results
        let fp3 = fp64_extend_i64(fp, 43);
        assert_ne!(fp1, fp3);
    }

    #[test]
    fn test_type_tags_match_tlc() {
        // Verify our type tags match TLC's ValueConstants
        use value_tags::*;
        assert_eq!(BOOLVALUE, 0);
        assert_eq!(INTVALUE, 1);
        assert_eq!(STRINGVALUE, 3);
        assert_eq!(SETENUMVALUE, 5);
        assert_eq!(FCNRCDVALUE, 9);
        assert_eq!(MODELVALUE, 21);
    }
}
