//! Lean5-specific payload attached to .olean files.
//!
//! We append a small trailer to .olean files we generate to carry the actual
//! kernel objects serialized with `bincode`. This avoids having to fully
//! replicate Lean 4's ConstantInfo layout while still enabling dependent
//! modules to load and reuse definitions.

use crate::error::{OleanError, OleanResult};
use lean5_kernel::env::ConstantInfo;
use lean5_kernel::inductive::{ConstructorVal, InductiveVal, RecursorVal};
use lean5_kernel::name::Name;
use serde::{Deserialize, Serialize};

/// Magic footer identifying Lean5 payloads.
pub const LEAN5_PAYLOAD_MAGIC: &[u8; 8] = b"LEAN5ENV";
/// Version of the Lean5 payload format.
pub const LEAN5_PAYLOAD_VERSION: u32 = 1;

/// Serialized kernel data embedded in a Lean5-generated `.olean`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Lean5Payload {
    pub constants: Vec<ConstantInfo>,
    pub inductives: Vec<InductiveVal>,
    pub constructors: Vec<ConstructorVal>,
    pub recursors: Vec<RecursorVal>,
    pub structure_fields: Vec<(Name, Vec<Name>)>,
}

impl Lean5Payload {
    /// Total number of constants represented (counts inductives/ctors/recs too).
    pub fn total_constants(&self) -> usize {
        self.constants.len()
            + self.inductives.len()
            + self.constructors.len()
            + self.recursors.len()
    }
}

/// Encode a payload and append a footer for easy detection.
pub fn encode_lean5_payload(payload: &Lean5Payload) -> OleanResult<Vec<u8>> {
    let data = bincode::serialize(payload).map_err(|e| {
        OleanError::Serialization(format!("failed to serialize Lean5 payload: {e}"))
    })?;

    let mut out =
        Vec::with_capacity(data.len() + LEAN5_PAYLOAD_MAGIC.len() + std::mem::size_of::<u32>() + 8);
    out.extend_from_slice(&data);
    out.extend_from_slice(LEAN5_PAYLOAD_MAGIC);
    out.extend_from_slice(&LEAN5_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    Ok(out)
}

/// Attempt to decode a Lean5 payload from the end of the provided bytes.
///
/// Returns `Ok(None)` if no payload footer is present.
pub fn decode_lean5_payload(bytes: &[u8]) -> OleanResult<Option<Lean5Payload>> {
    let trailer_len = LEAN5_PAYLOAD_MAGIC.len() + std::mem::size_of::<u32>() + 8;
    if bytes.len() < trailer_len {
        return Ok(None);
    }

    let len_offset = bytes.len() - 8;
    let version_offset = len_offset - std::mem::size_of::<u32>();
    let magic_offset = version_offset - LEAN5_PAYLOAD_MAGIC.len();

    if &bytes[magic_offset..version_offset] != LEAN5_PAYLOAD_MAGIC {
        return Ok(None);
    }

    let version = u32::from_le_bytes(
        bytes[version_offset..len_offset]
            .try_into()
            .map_err(|_| OleanError::InvalidPayload("truncated payload footer".into()))?,
    );
    if version != LEAN5_PAYLOAD_VERSION {
        return Err(OleanError::UnsupportedPayloadVersion {
            expected: LEAN5_PAYLOAD_VERSION,
            actual: version,
        });
    }

    let payload_len = u64::from_le_bytes(
        bytes[len_offset..]
            .try_into()
            .map_err(|_| OleanError::InvalidPayload("truncated payload length".into()))?,
    ) as usize;
    if payload_len > magic_offset {
        return Err(OleanError::InvalidPayload(format!(
            "payload length {} exceeds available bytes {}",
            payload_len,
            bytes.len()
        )));
    }
    let start = magic_offset - payload_len;
    let data = &bytes[start..magic_offset];

    let payload: Lean5Payload = bincode::deserialize(data).map_err(|e| {
        OleanError::Serialization(format!("failed to deserialize Lean5 payload: {e}"))
    })?;

    Ok(Some(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let payload = Lean5Payload {
            constants: vec![],
            inductives: vec![],
            constructors: vec![],
            recursors: vec![],
            structure_fields: vec![],
        };

        let encoded = encode_lean5_payload(&payload).expect("encode");
        let decoded = decode_lean5_payload(&encoded)
            .expect("decode result")
            .expect("payload missing");

        assert_eq!(decoded.total_constants(), 0);
    }

    #[test]
    fn decode_absent_payload_returns_none() {
        let bytes = vec![0u8; 16];
        let decoded = decode_lean5_payload(&bytes).unwrap();
        assert!(decoded.is_none());
    }
}
