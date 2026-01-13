//! GGUF format support for loading llama.cpp model weights.
//!
//! GGUF (GPT-Generated Unified Format) is the format used by llama.cpp
//! for efficient storage and loading of LLM weights. This module provides
//! support for loading weight metadata and data from GGUF files.
//!
//! # Supported Data Types
//!
//! **Unquantized:**
//! - F32: Full precision (directly loaded)
//! - F16: Half precision (converted to f32)
//!
//! **Simple Quants (32 elements per block):**
//! - Q8_0: 8-bit quantized (dequantized to f32)
//! - Q4_0: 4-bit quantized (dequantized to f32)
//! - Q4_1: 4-bit quantized with min (dequantized to f32)
//! - Q5_0: 5-bit quantized (dequantized to f32)
//! - Q5_1: 5-bit quantized with min (dequantized to f32)
//! - Q8_1: 8-bit quantized with sum (dequantized to f32)
//!
//! **K-Quants (256 elements per super-block):**
//! - Q2_K: 2-bit quantized with per-group scales
//! - Q3_K: 3-bit quantized with per-group scales
//! - Q4_K: 4-bit quantized with per-group scales
//! - Q5_K: 5-bit quantized with per-group scales
//! - Q6_K: 6-bit quantized with per-group scales
//!
//! # Usage
//!
//! ```ignore
//! use gamma_onnx::gguf::{load_gguf, gguf_info};
//!
//! // Get info about a GGUF file
//! let info = gguf_info("model.gguf")?;
//! println!("Model has {} tensors", info.tensor_count);
//!
//! // Load weights (including dequantized quantized tensors)
//! let weights = load_gguf("model.gguf")?;
//! ```

use crate::safetensors::half_to_f32;
use crate::WeightStore;
use gamma_core::{GammaError, Result};
use gguf::{GGMLType, GGUFFile, GGUFMetadataValue, GGUFTensorInfo};
use memmap2::Mmap;
use ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::path::Path;
use tracing::{debug, info, warn};

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> std::result::Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("Unexpected EOF while reading u32".to_string());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64_le(data: &[u8], pos: &mut usize) -> std::result::Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("Unexpected EOF while reading u64".to_string());
    }
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    Ok(v)
}

fn read_gguf_string(data: &[u8], pos: &mut usize) -> std::result::Result<Vec<u8>, String> {
    let len = read_u64_le(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err("Unexpected EOF while reading GGUF string".to_string());
    }
    let bytes = data[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(bytes)
}

fn skip_bytes(data: &[u8], pos: &mut usize, n: usize) -> std::result::Result<(), String> {
    if *pos + n > data.len() {
        return Err("Unexpected EOF while skipping bytes".to_string());
    }
    *pos += n;
    Ok(())
}

fn skip_metadata_value(
    data: &[u8],
    pos: &mut usize,
    value_type: u32,
    depth: usize,
) -> std::result::Result<(), String> {
    if depth > 4 {
        return Err("GGUF metadata nesting too deep".to_string());
    }

    match value_type {
        0 | 1 | 7 => skip_bytes(data, pos, 1), // u8, i8, bool
        2 | 3 => skip_bytes(data, pos, 2),     // u16, i16
        4..=6 => skip_bytes(data, pos, 4),     // u32, i32, f32
        10..=12 => skip_bytes(data, pos, 8),   // u64, i64, f64
        8 => {
            let len = read_u64_le(data, pos)? as usize;
            skip_bytes(data, pos, len)
        }
        9 => {
            let inner_type = read_u32_le(data, pos)?;
            let len = read_u64_le(data, pos)? as usize;
            for _ in 0..len {
                skip_metadata_value(data, pos, inner_type, depth + 1)?;
            }
            Ok(())
        }
        other => Err(format!("Unknown GGUF metadata value type: {}", other)),
    }
}

fn compute_data_section_offset(file_data: &[u8]) -> std::result::Result<usize, String> {
    // GGUF layout:
    // - Header + metadata + tensor infos
    // - Tensor data section begins after tensor infos, aligned to `general.alignment` (default 32).
    //
    // TensorInfo offsets are *relative* to the start of the data section, not absolute file
    // offsets. Many files have the first tensor offset = 0, so treating it as absolute would
    // read the GGUF header bytes as tensor data.
    let mut pos = 0usize;

    if file_data.len() < 4 || &file_data[0..4] != b"GGUF" {
        return Err("Invalid GGUF magic".to_string());
    }
    pos += 4;
    let _version = read_u32_le(file_data, &mut pos)?;
    let tensor_count = read_u64_le(file_data, &mut pos)? as usize;
    let metadata_count = read_u64_le(file_data, &mut pos)? as usize;

    let mut alignment: usize = 32;

    for _ in 0..metadata_count {
        let key_bytes = read_gguf_string(file_data, &mut pos)?;
        let value_type = read_u32_le(file_data, &mut pos)?;

        if key_bytes == b"general.alignment" {
            match value_type {
                4 => alignment = read_u32_le(file_data, &mut pos)? as usize,
                10 => alignment = read_u64_le(file_data, &mut pos)? as usize,
                _ => skip_metadata_value(file_data, &mut pos, value_type, 0)?,
            }
        } else {
            skip_metadata_value(file_data, &mut pos, value_type, 0)?;
        }
    }

    for _ in 0..tensor_count {
        let _name = read_gguf_string(file_data, &mut pos)?;
        let n_dimensions = read_u32_le(file_data, &mut pos)? as usize;
        skip_bytes(file_data, &mut pos, 8 * n_dimensions)?; // dims u64
        skip_bytes(file_data, &mut pos, 4)?; // tensor_type u32
        skip_bytes(file_data, &mut pos, 8)?; // offset u64
    }

    Ok(align_up(pos, alignment.max(1)))
}

// =============================================================================
// GGML Dequantization
// =============================================================================
// Implements dequantization for common GGML quantization formats.
// Reference: llama.cpp ggml-quants.c
//
// Block sizes:
// - Q8_0: 32 elements, 34 bytes (2 byte half scale + 32 int8 quants)
// - Q4_0: 32 elements, 18 bytes (2 byte half scale + 16 bytes nibbles)
// - Q4_1: 32 elements, 20 bytes (2 byte half scale + 2 byte half min + 16 bytes nibbles)
// - Q5_0: 32 elements, 22 bytes (2 byte half scale + 4 bytes high bits + 16 bytes low nibbles)
// - Q5_1: 32 elements, 24 bytes (2 byte half d + 2 byte half m + 4 bytes high + 16 bytes low)
// - Q8_1: 32 elements, 36 bytes (2 byte half d + 2 byte half s + 32 int8 quants)

const QK8_0: usize = 32; // Elements per Q8_0 block
const QK4_0: usize = 32; // Elements per Q4_0 block
const QK4_1: usize = 32; // Elements per Q4_1 block
const QK5_0: usize = 32; // Elements per Q5_0 block
const QK5_1: usize = 32; // Elements per Q5_1 block
const QK8_1: usize = 32; // Elements per Q8_1 block

// =============================================================================
// K-Quant Constants
// =============================================================================
// K-quants use super-blocks of 256 elements with per-group scales.
// Reference: llama.cpp ggml-quants.c, ggml-common.h

const QK_K: usize = 256; // Elements per K-quant super-block
#[allow(dead_code)]
const K_SCALE_SIZE: usize = 12; // Bytes for scales in Q4_K and Q5_K (documented for reference)

// K-quant block sizes (bytes per super-block of 256 elements):
// Q2_K: scales[16] + qs[64] + d(2) + dmin(2) = 84 bytes
// Q3_K: hmask[32] + qs[64] + scales[12] + d(2) = 110 bytes
// Q4_K: d(2) + dmin(2) + scales[12] + qs[128] = 144 bytes
// Q5_K: d(2) + dmin(2) + scales[12] + qh[32] + qs[128] = 176 bytes
// Q6_K: ql[128] + qh[64] + scales[16] + d(2) = 210 bytes

/// Dequantize Q8_0 format: 32 elements per block, each stored as int8.
/// Block layout: [f16 delta (2 bytes)][32 x int8 quants]
/// Formula: y[i] = qs[i] * d
fn dequantize_q8_0(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + QK8_0; // 34 bytes per block

    if elements % QK8_0 != 0 {
        return Err(format!(
            "Q8_0 requires element count divisible by {}, got {}",
            QK8_0, elements
        ));
    }

    let num_blocks = elements / QK8_0;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q8_0 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale)
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = half_to_f32(d_bits);

        // Dequantize each int8 value
        for j in 0..QK8_0 {
            let qs = data[block_start + 2 + j] as i8;
            result.push(qs as f32 * d);
        }
    }

    Ok(result)
}

/// Dequantize Q4_0 format: 32 elements per block, packed as nibbles.
/// Block layout: [f16 delta (2 bytes)][16 x uint8 nibble pairs]
/// Formula: y[i] = (nibble - 8) * d
fn dequantize_q4_0(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + QK4_0 / 2; // 18 bytes per block

    if elements % QK4_0 != 0 {
        return Err(format!(
            "Q4_0 requires element count divisible by {}, got {}",
            QK4_0, elements
        ));
    }

    let num_blocks = elements / QK4_0;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q4_0 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale)
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = half_to_f32(d_bits);

        // Unpack nibbles: lower nibble goes to first half, upper to second half
        // Note: llama.cpp interleaves: y[j] and y[j + qk/2]
        let half_qk = QK4_0 / 2;
        // First pass: lower nibbles (first half of block output)
        for j in 0..half_qk {
            let byte = data[block_start + 2 + j];
            let x0 = (byte & 0x0F) as i32 - 8;
            result.push(x0 as f32 * d);
        }
        // Second pass: upper nibbles (second half of block output)
        for j in 0..half_qk {
            let byte = data[block_start + 2 + j];
            let x1 = (byte >> 4) as i32 - 8;
            result.push(x1 as f32 * d);
        }
    }

    Ok(result)
}

/// Dequantize Q4_1 format: 32 elements per block, packed as nibbles with min.
/// Block layout: [f16 d (2 bytes)][f16 m (2 bytes)][16 x uint8 nibble pairs]
/// Formula: y[i] = nibble * d + m
fn dequantize_q4_1(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 2 + QK4_1 / 2; // 20 bytes per block

    if elements % QK4_1 != 0 {
        return Err(format!(
            "Q4_1 requires element count divisible by {}, got {}",
            QK4_1, elements
        ));
    }

    let num_blocks = elements / QK4_1;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q4_1 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale) and min
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let m_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let d = half_to_f32(d_bits);
        let m = half_to_f32(m_bits);

        // Unpack nibbles
        let half_qk = QK4_1 / 2;
        for j in 0..half_qk {
            let byte = data[block_start + 4 + j];
            let x0 = (byte & 0x0F) as f32;
            result.push(x0 * d + m);
        }
        for j in 0..half_qk {
            let byte = data[block_start + 4 + j];
            let x1 = (byte >> 4) as f32;
            result.push(x1 * d + m);
        }
    }

    Ok(result)
}

/// Dequantize Q5_0 format: 32 elements per block, 5 bits per element.
/// Block layout: [f16 d (2 bytes)][4 bytes high bits][16 bytes low nibbles]
/// Formula: y[i] = ((nibble | (high_bit << 4)) - 16) * d
fn dequantize_q5_0(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 4 + QK5_0 / 2; // 22 bytes per block

    if elements % QK5_0 != 0 {
        return Err(format!(
            "Q5_0 requires element count divisible by {}, got {}",
            QK5_0, elements
        ));
    }

    let num_blocks = elements / QK5_0;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q5_0 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 delta (scale)
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = half_to_f32(d_bits);

        // Read high bits (4 bytes = 32 bits for 32 elements)
        let qh = u32::from_le_bytes([
            data[block_start + 2],
            data[block_start + 3],
            data[block_start + 4],
            data[block_start + 5],
        ]);

        // Unpack: combine low nibble with high bit
        let half_qk = QK5_0 / 2;
        // First half (low nibbles)
        for j in 0..half_qk {
            let byte = data[block_start + 6 + j];
            let x0_low = (byte & 0x0F) as i32;
            let x0_high = ((qh >> j) & 1) as i32;
            let x0 = (x0_low | (x0_high << 4)) - 16;
            result.push(x0 as f32 * d);
        }
        // Second half (upper nibbles)
        for j in 0..half_qk {
            let byte = data[block_start + 6 + j];
            let x1_low = (byte >> 4) as i32;
            let x1_high = ((qh >> (j + 16)) & 1) as i32;
            let x1 = (x1_low | (x1_high << 4)) - 16;
            result.push(x1 as f32 * d);
        }
    }

    Ok(result)
}

/// Dequantize Q5_1 format: 32 elements per block, 5 bits per element with min.
/// Block layout: [f16 d (2 bytes)][f16 m (2 bytes)][4 bytes high bits][16 bytes low nibbles]
/// Formula: y[i] = (nibble | (high_bit << 4)) * d + m
fn dequantize_q5_1(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 2 + 4 + QK5_1 / 2; // 24 bytes per block

    if elements % QK5_1 != 0 {
        return Err(format!(
            "Q5_1 requires element count divisible by {}, got {}",
            QK5_1, elements
        ));
    }

    let num_blocks = elements / QK5_1;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q5_1 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 d and m
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let m_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let d = half_to_f32(d_bits);
        let m = half_to_f32(m_bits);

        // Read high bits
        let qh = u32::from_le_bytes([
            data[block_start + 4],
            data[block_start + 5],
            data[block_start + 6],
            data[block_start + 7],
        ]);

        let half_qk = QK5_1 / 2;
        for j in 0..half_qk {
            let byte = data[block_start + 8 + j];
            let x0_low = (byte & 0x0F) as i32;
            let x0_high = ((qh >> j) & 1) as i32;
            let x0 = x0_low | (x0_high << 4);
            result.push(x0 as f32 * d + m);
        }
        for j in 0..half_qk {
            let byte = data[block_start + 8 + j];
            let x1_low = (byte >> 4) as i32;
            let x1_high = ((qh >> (j + 16)) & 1) as i32;
            let x1 = x1_low | (x1_high << 4);
            result.push(x1 as f32 * d + m);
        }
    }

    Ok(result)
}

/// Dequantize Q8_1 format: 32 elements per block, 8 bits with sum.
/// Block layout: [f16 d (2 bytes)][f16 s (2 bytes)][32 x int8 quants]
/// Formula: y[i] = qs[i] * d (s is used for dot product optimization, not dequant)
fn dequantize_q8_1(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 2 + 2 + QK8_1; // 36 bytes per block

    if elements % QK8_1 != 0 {
        return Err(format!(
            "Q8_1 requires element count divisible by {}, got {}",
            QK8_1, elements
        ));
    }

    let num_blocks = elements / QK8_1;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q8_1 data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;

        // Read f16 d (s is not used for dequantization)
        let d_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let d = half_to_f32(d_bits);

        // Dequantize
        for j in 0..QK8_1 {
            let qs = data[block_start + 4 + j] as i8;
            result.push(qs as f32 * d);
        }
    }

    Ok(result)
}

// =============================================================================
// K-Quant Dequantization
// =============================================================================
// K-quants use super-blocks of 256 elements with more sophisticated quantization.
// Reference: llama.cpp ggml-quants.c dequantize_row_* functions

/// Helper function to extract scale and min for Q4_K and Q5_K formats.
/// The scales are packed in a 12-byte array using 6-bit values.
/// For j < 4: scale in lower 6 bits of q[j], min in lower 6 bits of q[j+4]
/// For j >= 4: scale and min are split across multiple bytes
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q2_K format: 256 elements per super-block, 2-bit quantization.
/// Block layout: [scales[16]][qs[64]][f16 d][f16 dmin] = 84 bytes
/// Reference: llama.cpp dequantize_row_q2_K
fn dequantize_q2_k(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 84;

    if elements % QK_K != 0 {
        return Err(format!(
            "Q2_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q2_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_SIZE..];

        // Block layout: scales[16], qs[64], d(2), dmin(2)
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d_bits = u16::from_le_bytes([block[80], block[81]]);
        let dmin_bits = u16::from_le_bytes([block[82], block[83]]);
        let d = half_to_f32(d_bits);
        let dmin = half_to_f32(dmin_bits);

        let mut is: usize = 0;
        let mut q_idx: usize = 0;

        // Process in chunks of 128 elements
        for _ in 0..2 {
            let mut shift = 0;
            // Each chunk has 4 sub-groups of 32 elements
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0F) as f32;
                let ml = dmin * (sc >> 4) as f32;

                // First 16 values
                for l in 0..16 {
                    let q_val = ((qs[q_idx + l] >> shift) & 3) as i8;
                    result.push(dl * q_val as f32 - ml);
                }

                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0F) as f32;
                let ml = dmin * (sc >> 4) as f32;

                // Second 16 values
                for l in 0..16 {
                    let q_val = ((qs[q_idx + 16 + l] >> shift) & 3) as i8;
                    result.push(dl * q_val as f32 - ml);
                }

                shift += 2;
            }
            q_idx += 32;
        }
    }

    Ok(result)
}

/// Dequantize Q3_K format: 256 elements per super-block, 3-bit quantization.
/// Block layout: [hmask[32]][qs[64]][scales[12]][f16 d] = 110 bytes
/// Reference: llama.cpp dequantize_row_q3_K
fn dequantize_q3_k(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 110;

    if elements % QK_K != 0 {
        return Err(format!(
            "Q3_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q3_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_SIZE..];

        // Block layout: hmask[32], qs[64], scales[12], d(2)
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let raw_scales = &block[96..108];
        let d_bits = u16::from_le_bytes([block[108], block[109]]);
        let d_all = half_to_f32(d_bits);

        // Unpack the 6-bit scales from 12 bytes into 16 values
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([raw_scales[0], raw_scales[1], raw_scales[2], raw_scales[3]]);
        aux[1] = u32::from_le_bytes([raw_scales[4], raw_scales[5], raw_scales[6], raw_scales[7]]);
        aux[2] = u32::from_le_bytes([raw_scales[8], raw_scales[9], raw_scales[10], raw_scales[11]]);

        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        // Interpret aux bytes as signed scales (little-endian, platform-independent)
        let mut scales_bytes = [0u8; 16];
        for (i, word) in aux.iter().enumerate() {
            scales_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        let scales: [i8; 16] = scales_bytes.map(|b| b as i8);

        let mut is: usize = 0;
        let mut q_idx: usize = 0;
        let mut m: u8 = 1;

        // Process in chunks of 128 elements
        for _ in 0..2 {
            let mut shift = 0;
            // Each chunk has 4 sub-groups of 32 elements
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;

                // First 16 values
                for l in 0..16 {
                    let q_low = ((qs[q_idx + l] >> shift) & 3) as i8;
                    let h = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    result.push(dl * (q_low - h) as f32);
                }

                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;

                // Second 16 values
                for l in 0..16 {
                    let q_low = ((qs[q_idx + 16 + l] >> shift) & 3) as i8;
                    let h = if (hmask[16 + l] & m) != 0 { 0 } else { 4 };
                    result.push(dl * (q_low - h) as f32);
                }

                shift += 2;
                m <<= 1;
            }
            q_idx += 32;
        }
    }

    Ok(result)
}

/// Dequantize Q4_K format: 256 elements per super-block, 4-bit quantization.
/// Block layout: [f16 d][f16 dmin][scales[12]][qs[128]] = 144 bytes
/// Reference: llama.cpp dequantize_row_q4_K
fn dequantize_q4_k(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 144;

    if elements % QK_K != 0 {
        return Err(format!(
            "Q4_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q4_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_SIZE..];

        // Block layout: d(2), dmin(2), scales[12], qs[128]
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let d = half_to_f32(d_bits);
        let dmin = half_to_f32(dmin_bits);
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut is: usize = 0;
        let mut q_idx: usize = 0;

        // Process in chunks of 64 elements (4 chunks total)
        for _ in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            // Low nibbles (32 values)
            for l in 0..32 {
                result.push(d1 * (qs[q_idx + l] & 0x0F) as f32 - m1);
            }
            // High nibbles (32 values)
            for l in 0..32 {
                result.push(d2 * (qs[q_idx + l] >> 4) as f32 - m2);
            }

            q_idx += 32;
            is += 2;
        }
    }

    Ok(result)
}

/// Dequantize Q5_K format: 256 elements per super-block, 5-bit quantization.
/// Block layout: [f16 d][f16 dmin][scales[12]][qh[32]][qs[128]] = 176 bytes
/// Reference: llama.cpp dequantize_row_q5_K
fn dequantize_q5_k(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 176;

    if elements % QK_K != 0 {
        return Err(format!(
            "Q5_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q5_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_SIZE..];

        // Block layout: d(2), dmin(2), scales[12], qh[32], qs[128]
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let d = half_to_f32(d_bits);
        let dmin = half_to_f32(dmin_bits);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut is: usize = 0;
        let mut q_idx: usize = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        // Process in chunks of 64 elements (4 chunks total)
        for _ in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            // Low nibbles + high bit (32 values)
            for l in 0..32 {
                let low = qs[q_idx + l] & 0x0F;
                let high = if (qh[l] & u1) != 0 { 16 } else { 0 };
                result.push(d1 * (low + high) as f32 - m1);
            }
            // High nibbles + high bit (32 values)
            for l in 0..32 {
                let low = qs[q_idx + l] >> 4;
                let high = if (qh[l] & u2) != 0 { 16 } else { 0 };
                result.push(d2 * (low + high) as f32 - m2);
            }

            q_idx += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(result)
}

/// Dequantize Q6_K format: 256 elements per super-block, 6-bit quantization.
/// Block layout: [ql[128]][qh[64]][scales[16]][f16 d] = 210 bytes
/// Reference: llama.cpp dequantize_row_q6_K
fn dequantize_q6_k(data: &[u8], elements: usize) -> std::result::Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 210;

    if elements % QK_K != 0 {
        return Err(format!(
            "Q6_K requires element count divisible by {}, got {}",
            QK_K, elements
        ));
    }

    let num_blocks = elements / QK_K;
    let expected_bytes = num_blocks * BLOCK_SIZE;

    if data.len() < expected_bytes {
        return Err(format!(
            "Q6_K data too short: expected {} bytes, got {}",
            expected_bytes,
            data.len()
        ));
    }

    let mut result = Vec::with_capacity(elements);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_SIZE..];

        // Block layout: ql[128], qh[64], scales[16], d(2)
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d = half_to_f32(d_bits);

        // Scales are signed 8-bit values
        let mut scales_i8 = [0i8; 16];
        for (dst, src) in scales_i8.iter_mut().zip(scales.iter()) {
            *dst = *src as i8;
        }

        let mut ql_idx: usize = 0;
        let mut qh_idx: usize = 0;
        let mut sc_idx: usize = 0;

        // Process in chunks of 128 elements (2 chunks total)
        for _ in 0..2 {
            // 32 elements at a time, 4 groups per chunk
            for l in 0..32 {
                let is = l / 16;

                // Reconstruct 6-bit values from low 4 bits and high 2 bits
                let q1 = ((ql[ql_idx + l] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[ql_idx + 32 + l] & 0x0F) | (((qh[qh_idx + l] >> 2) & 0x03) << 4))
                    as i8
                    - 32;
                let q3 = ((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_idx + 32 + l] >> 4) | (((qh[qh_idx + l] >> 6) & 0x03) << 4)) as i8 - 32;

                result.push(d * scales_i8[sc_idx + is] as f32 * q1 as f32);
                result.push(d * scales_i8[sc_idx + is + 2] as f32 * q2 as f32);
                result.push(d * scales_i8[sc_idx + is + 4] as f32 * q3 as f32);
                result.push(d * scales_i8[sc_idx + is + 6] as f32 * q4 as f32);
            }
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }

    Ok(result)
}

/// Get the byte size of a quantized block.
fn get_block_size(dtype: &GGMLType) -> Option<usize> {
    match dtype {
        GGMLType::Q8_0 => Some(2 + QK8_0),             // 34
        GGMLType::Q4_0 => Some(2 + QK4_0 / 2),         // 18
        GGMLType::Q4_1 => Some(2 + 2 + QK4_1 / 2),     // 20
        GGMLType::Q5_0 => Some(2 + 4 + QK5_0 / 2),     // 22
        GGMLType::Q5_1 => Some(2 + 2 + 4 + QK5_1 / 2), // 24
        GGMLType::Q8_1 => Some(2 + 2 + QK8_1),         // 36
        // K-quants: 256 elements per super-block
        GGMLType::Q2K => Some(84),  // scales[16] + qs[64] + d(2) + dmin(2)
        GGMLType::Q3K => Some(110), // hmask[32] + qs[64] + scales[12] + d(2)
        GGMLType::Q4K => Some(144), // d(2) + dmin(2) + scales[12] + qs[128]
        GGMLType::Q5K => Some(176), // d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
        GGMLType::Q6K => Some(210), // ql[128] + qh[64] + scales[16] + d(2)
        _ => None,
    }
}

/// Get elements per block for a quantized type.
fn get_block_elements(dtype: &GGMLType) -> Option<usize> {
    match dtype {
        GGMLType::Q8_0 => Some(QK8_0),
        GGMLType::Q4_0 => Some(QK4_0),
        GGMLType::Q4_1 => Some(QK4_1),
        GGMLType::Q5_0 => Some(QK5_0),
        GGMLType::Q5_1 => Some(QK5_1),
        GGMLType::Q8_1 => Some(QK8_1),
        // K-quants all use QK_K = 256 elements per super-block
        GGMLType::Q2K | GGMLType::Q3K | GGMLType::Q4K | GGMLType::Q5K | GGMLType::Q6K => Some(QK_K),
        _ => None,
    }
}

/// Metadata about a GGUF file.
#[derive(Debug, Clone)]
pub struct GGUFInfo {
    /// GGUF format version.
    pub version: u32,
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total parameter count (sum of all tensor elements).
    pub param_count: usize,
    /// Model architecture (from metadata, if present).
    pub architecture: Option<String>,
    /// Model name (from metadata, if present).
    pub model_name: Option<String>,
    /// Tensor information (name, shape, dtype, quantized).
    pub tensors: Vec<(String, Vec<u64>, String, bool)>,
    /// Key metadata entries.
    pub metadata: Vec<(String, String)>,
}

/// Get information about a GGUF file without fully loading tensor data.
pub fn gguf_info<P: AsRef<Path>>(path: P) -> Result<GGUFInfo> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(GammaError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    // Use memory-mapped I/O for efficient large file handling.
    let file = File::open(path)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to open GGUF file: {}", e)))?;

    // SAFETY: File is read-only and Mmap lifetime is contained within this function.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| GammaError::ModelLoad(format!("Failed to mmap GGUF file: {}", e)))?;

    let data: &[u8] = &mmap;

    let gguf_file = GGUFFile::read(data)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to parse GGUF: {}", e)))?
        .ok_or_else(|| GammaError::ModelLoad("Incomplete GGUF file".to_string()))?;

    // Extract architecture and model name from metadata
    let mut architecture = None;
    let mut model_name = None;
    let mut metadata = Vec::new();

    for meta in &gguf_file.header.metadata {
        let value_str = format_metadata_value(&meta.value);

        // Look for key metadata
        if meta.key == "general.architecture" {
            if let GGUFMetadataValue::String(s) = &meta.value {
                architecture = Some(s.clone());
            }
        }
        if meta.key == "general.name" {
            if let GGUFMetadataValue::String(s) = &meta.value {
                model_name = Some(s.clone());
            }
        }

        // Store interesting metadata
        if meta.key.starts_with("general.")
            || meta.key.contains(".context_length")
            || meta.key.contains(".embedding_length")
            || meta.key.contains(".block_count")
            || meta.key.contains(".attention.head_count")
        {
            metadata.push((meta.key.clone(), value_str));
        }
    }

    // Process tensor info
    let mut tensors = Vec::new();
    let mut param_count = 0;

    for tensor in &gguf_file.tensors {
        let elements: u64 = tensor.dimensions.iter().product();
        param_count += elements as usize;

        let is_quantized = is_quantized_type(&tensor.tensor_type);
        tensors.push((
            tensor.name.clone(),
            tensor.dimensions.clone(),
            format!("{:?}", tensor.tensor_type),
            is_quantized,
        ));
    }

    Ok(GGUFInfo {
        version: gguf_file.header.version,
        tensor_count: gguf_file.tensors.len(),
        param_count,
        architecture,
        model_name,
        tensors,
        metadata,
    })
}

/// Load weights from a GGUF file.
///
/// Loads F32/F16 tensors directly and dequantizes supported quantized tensors into f32.
///
/// # Arguments
///
/// * `path` - Path to the .gguf file
///
/// # Returns
///
/// A `WeightStore` with all loadable (non-quantized) tensors.
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    info!("Loading GGUF from: {}", path.display());

    if !path.exists() {
        return Err(GammaError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    // Use memory-mapped I/O for efficient large file handling.
    // This allows the OS to page in tensor data on-demand rather than
    // loading the entire file (which can be 30GB+) into memory upfront.
    let file = File::open(path)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to open GGUF file: {}", e)))?;

    // SAFETY: The file is opened read-only and we maintain the Mmap for the
    // lifetime of this function. The file is not modified during loading.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| GammaError::ModelLoad(format!("Failed to mmap GGUF file: {}", e)))?;

    let data: &[u8] = &mmap;

    let gguf_file = GGUFFile::read(data)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to parse GGUF: {}", e)))?
        .ok_or_else(|| GammaError::ModelLoad("Incomplete GGUF file".to_string()))?;

    let data_section_offset = compute_data_section_offset(data).map_err(|e| {
        GammaError::ModelLoad(format!("Failed to compute GGUF data section: {}", e))
    })?;

    let mut weights = WeightStore::new();
    let mut loaded_count = 0;
    let mut dequant_count = 0;
    let mut skipped_count = 0;

    for tensor in &gguf_file.tensors {
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let elements: usize = shape.iter().product();

        match load_tensor_data(data, data_section_offset, tensor, elements) {
            Ok(arr) => {
                if is_quantized_type(&tensor.tensor_type) {
                    debug!(
                        "Dequantized tensor: {} ({:?}) shape {:?}",
                        tensor.name, tensor.tensor_type, shape
                    );
                    dequant_count += 1;
                } else {
                    debug!("Loaded tensor: {} shape {:?}", tensor.name, shape);
                }
                weights.insert(tensor.name.clone(), arr);
                loaded_count += 1;
            }
            Err(e) => {
                if is_quantized_type(&tensor.tensor_type) && !is_dequantizable(&tensor.tensor_type)
                {
                    debug!(
                        "Skipping unsupported quantized tensor '{}' ({:?}): {}",
                        tensor.name, tensor.tensor_type, e
                    );
                } else {
                    warn!(
                        "Failed to load tensor '{}' ({:?}): {}",
                        tensor.name, tensor.tensor_type, e
                    );
                }
                skipped_count += 1;
            }
        }
    }

    if dequant_count > 0 {
        info!(
            "Loaded {} tensors from GGUF ({} dequantized, {} skipped)",
            loaded_count, dequant_count, skipped_count
        );
    } else {
        info!(
            "Loaded {} tensors from GGUF ({} skipped)",
            loaded_count, skipped_count
        );
    }

    Ok(weights)
}

/// Load tensor data from the GGUF file.
fn load_tensor_data(
    file_data: &[u8],
    data_section_offset: usize,
    tensor: &GGUFTensorInfo,
    elements: usize,
) -> std::result::Result<ArrayD<f32>, String> {
    let offset = data_section_offset + tensor.offset as usize;
    let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();

    match tensor.tensor_type {
        GGMLType::F32 => {
            let byte_size = elements * 4;
            if offset + byte_size > file_data.len() {
                return Err(format!(
                    "Tensor data out of bounds (offset={}, size={}, file_len={})",
                    offset,
                    byte_size,
                    file_data.len()
                ));
            }

            let data = &file_data[offset..offset + byte_size];
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            ArrayD::from_shape_vec(IxDyn(&shape), floats)
                .map_err(|e| format!("Shape mismatch: {}", e))
        }
        GGMLType::F16 => {
            let byte_size = elements * 2;
            if offset + byte_size > file_data.len() {
                return Err(format!(
                    "Tensor data out of bounds (offset={}, size={}, file_len={})",
                    offset,
                    byte_size,
                    file_data.len()
                ));
            }

            let data = &file_data[offset..offset + byte_size];
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half_to_f32(bits)
                })
                .collect();

            ArrayD::from_shape_vec(IxDyn(&shape), floats)
                .map_err(|e| format!("Shape mismatch: {}", e))
        }
        // Quantized types - dequantize to f32
        GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q5_0
        | GGMLType::Q5_1
        | GGMLType::Q8_1
        // K-quants (256 elements per super-block)
        | GGMLType::Q2K
        | GGMLType::Q3K
        | GGMLType::Q4K
        | GGMLType::Q5K
        | GGMLType::Q6K => {
            let block_size = get_block_size(&tensor.tensor_type)
                .ok_or_else(|| format!("Unknown block size for {:?}", tensor.tensor_type))?;
            let block_elements = get_block_elements(&tensor.tensor_type)
                .ok_or_else(|| format!("Unknown block elements for {:?}", tensor.tensor_type))?;

            if elements % block_elements != 0 {
                return Err(format!(
                    "Element count {} not divisible by block size {} for {:?}",
                    elements, block_elements, tensor.tensor_type
                ));
            }

            let num_blocks = elements / block_elements;
            let byte_size = num_blocks * block_size;

            if offset + byte_size > file_data.len() {
                return Err(format!(
                    "Tensor data out of bounds (offset={}, size={}, file_len={})",
                    offset,
                    byte_size,
                    file_data.len()
                ));
            }

            let data = &file_data[offset..offset + byte_size];
            let floats = match tensor.tensor_type {
                GGMLType::Q8_0 => dequantize_q8_0(data, elements)?,
                GGMLType::Q4_0 => dequantize_q4_0(data, elements)?,
                GGMLType::Q4_1 => dequantize_q4_1(data, elements)?,
                GGMLType::Q5_0 => dequantize_q5_0(data, elements)?,
                GGMLType::Q5_1 => dequantize_q5_1(data, elements)?,
                GGMLType::Q8_1 => dequantize_q8_1(data, elements)?,
                // K-quants
                GGMLType::Q2K => dequantize_q2_k(data, elements)?,
                GGMLType::Q3K => dequantize_q3_k(data, elements)?,
                GGMLType::Q4K => dequantize_q4_k(data, elements)?,
                GGMLType::Q5K => dequantize_q5_k(data, elements)?,
                GGMLType::Q6K => dequantize_q6_k(data, elements)?,
                _ => unreachable!(),
            };

            ArrayD::from_shape_vec(IxDyn(&shape), floats)
                .map_err(|e| format!("Shape mismatch: {}", e))
        }
        // Other quantization types not yet supported
        _ => Err(format!(
            "Quantized type {:?} not yet supported",
            tensor.tensor_type
        )),
    }
}

/// Check if a GGML type is quantized.
fn is_quantized_type(dtype: &GGMLType) -> bool {
    !matches!(dtype, GGMLType::F32 | GGMLType::F16)
}

/// Check if a GGML type can be dequantized by this library.
fn is_dequantizable(dtype: &GGMLType) -> bool {
    matches!(
        dtype,
        GGMLType::F32
            | GGMLType::F16
            | GGMLType::Q8_0
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q5_0
            | GGMLType::Q5_1
            | GGMLType::Q8_1
            // K-quants
            | GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
    )
}

/// Format a metadata value for display.
fn format_metadata_value(value: &GGUFMetadataValue) -> String {
    match value {
        GGUFMetadataValue::Uint8(v) => v.to_string(),
        GGUFMetadataValue::Int8(v) => v.to_string(),
        GGUFMetadataValue::Uint16(v) => v.to_string(),
        GGUFMetadataValue::Int16(v) => v.to_string(),
        GGUFMetadataValue::Uint32(v) => v.to_string(),
        GGUFMetadataValue::Int32(v) => v.to_string(),
        GGUFMetadataValue::Float32(v) => v.to_string(),
        GGUFMetadataValue::Uint64(v) => v.to_string(),
        GGUFMetadataValue::Int64(v) => v.to_string(),
        GGUFMetadataValue::Float64(v) => v.to_string(),
        GGUFMetadataValue::Bool(v) => v.to_string(),
        GGUFMetadataValue::String(v) => v.clone(),
        GGUFMetadataValue::Array(arr) => format!("[{} elements]", arr.value.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn push_u64(v: &mut Vec<u8>, x: u64) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn push_string(v: &mut Vec<u8>, s: &str) {
        push_u64(v, s.len() as u64);
        v.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn test_gguf_tensor_offsets_are_relative_to_data_section() {
        // Minimal GGUF v3 file with one F32 tensor at offset 0 (relative to the data section).
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"GGUF");
        push_u32(&mut buf, 3); // version
        push_u64(&mut buf, 1); // tensor_count
        push_u64(&mut buf, 1); // metadata_count

        // metadata: general.alignment = 32 (Uint32)
        push_string(&mut buf, "general.alignment");
        push_u32(&mut buf, 4);
        push_u32(&mut buf, 32);

        // tensor info: name, ndims, dims, type, offset
        push_string(&mut buf, "test.weight");
        push_u32(&mut buf, 1);
        push_u64(&mut buf, 4);
        push_u32(&mut buf, GGMLType::F32 as u32);
        push_u64(&mut buf, 0);

        let expected_data_start = align_up(buf.len(), 32);
        buf.resize(expected_data_start, 0);

        for f in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&f.to_le_bytes());
        }

        let gguf_file = GGUFFile::read(&buf).unwrap().unwrap();
        let tensor = &gguf_file.tensors[0];
        assert_eq!(tensor.offset, 0);

        let data_start = compute_data_section_offset(&buf).unwrap();
        assert_eq!(data_start, expected_data_start);

        let arr = load_tensor_data(&buf, data_start, tensor, 4).unwrap();
        let (raw, _) = arr.into_raw_vec_and_offset();
        assert_eq!(raw, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // Helper: convert f32 to f16 bits (for test data creation)
    fn f32_to_half(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;

        if exp == 0xFF {
            // Inf/NaN
            return sign | 0x7C00 | ((mant != 0) as u16);
        }
        if exp == 0 {
            // Zero/Denormal
            return sign;
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return sign | 0x7C00; // Overflow to inf
        }
        if new_exp <= 0 {
            return sign; // Underflow to zero
        }

        sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
    }

    #[test]
    fn test_is_quantized_type() {
        assert!(!is_quantized_type(&GGMLType::F32));
        assert!(!is_quantized_type(&GGMLType::F16));
        assert!(is_quantized_type(&GGMLType::Q4_0));
        assert!(is_quantized_type(&GGMLType::Q8_0));
    }

    #[test]
    fn test_is_dequantizable() {
        assert!(is_dequantizable(&GGMLType::F32));
        assert!(is_dequantizable(&GGMLType::F16));
        assert!(is_dequantizable(&GGMLType::Q8_0));
        assert!(is_dequantizable(&GGMLType::Q4_0));
        assert!(is_dequantizable(&GGMLType::Q4_1));
        assert!(is_dequantizable(&GGMLType::Q5_0));
        assert!(is_dequantizable(&GGMLType::Q5_1));
        assert!(is_dequantizable(&GGMLType::Q8_1));
        // K-quants are now supported
        assert!(is_dequantizable(&GGMLType::Q2K));
        assert!(is_dequantizable(&GGMLType::Q3K));
        assert!(is_dequantizable(&GGMLType::Q4K));
        assert!(is_dequantizable(&GGMLType::Q5K));
        assert!(is_dequantizable(&GGMLType::Q6K));
    }

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Create a Q8_0 block: 2 bytes (f16 scale) + 32 bytes (int8 quants)
        let scale: f32 = 0.5;
        let scale_bits = f32_to_half(scale);

        let mut data = Vec::with_capacity(34);
        data.extend_from_slice(&scale_bits.to_le_bytes());

        // Create int8 values: -127 to +127 range, we'll use simple values
        // Let's use values 0, 1, 2, ..., 31 as signed int8
        for i in 0..32i8 {
            data.push(i as u8);
        }

        let result = dequantize_q8_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);

        // Verify: y[i] = qs[i] * scale
        for (i, &val) in result.iter().enumerate() {
            let expected = i as f32 * scale;
            assert!(
                (val - expected).abs() < 0.01,
                "Mismatch at {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }
    }

    #[test]
    fn test_dequantize_q8_0_negative() {
        // Test with negative scale and negative values
        let scale: f32 = -0.25;
        let scale_bits = f32_to_half(scale);

        let mut data = Vec::with_capacity(34);
        data.extend_from_slice(&scale_bits.to_le_bytes());

        // Use negative values
        for i in 0..32i8 {
            data.push((i - 16) as u8); // -16 to +15
        }

        let result = dequantize_q8_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);

        for (i, &val) in result.iter().enumerate() {
            let qs = (i as i8 - 16) as f32;
            let expected = qs * scale;
            assert!(
                (val - expected).abs() < 0.01,
                "Mismatch at {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0_basic() {
        // Create a Q4_0 block: 2 bytes (f16 scale) + 16 bytes (nibble pairs)
        let scale: f32 = 1.0;
        let scale_bits = f32_to_half(scale);

        let mut data = Vec::with_capacity(18);
        data.extend_from_slice(&scale_bits.to_le_bytes());

        // Each byte stores two nibbles. Lower nibble = first half, upper = second half.
        // Values are 0-15, offset by -8 to get -8 to +7.
        // Let's use byte values where lower nibble = 8, upper nibble = 8 (both = 0 after offset)
        data.extend(std::iter::repeat(0x88).take(16)); // Both nibbles = 8, so (8-8)=0 and (8-8)=0

        let result = dequantize_q4_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);

        // All values should be 0.0
        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 0.01, "Expected 0.0 at {}, got {}", i, val);
        }
    }

    #[test]
    fn test_dequantize_q4_0_varied() {
        // Test with varied nibble values
        let scale: f32 = 2.0;
        let scale_bits = f32_to_half(scale);

        let mut data = Vec::with_capacity(18);
        data.extend_from_slice(&scale_bits.to_le_bytes());

        // Create 16 bytes with pattern: low=0, high=15 (after offset: -8 and +7)
        data.extend(std::iter::repeat(0xF0).take(16)); // low nibble = 0, high nibble = 15

        let result = dequantize_q4_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);

        // First 16 values should be (0-8)*2 = -16
        for (i, &val) in result[..16].iter().enumerate() {
            assert!(
                (val - (-16.0)).abs() < 0.01,
                "Expected -16.0 at {}, got {}",
                i,
                val
            );
        }
        // Next 16 values should be (15-8)*2 = 14
        for (i, &val) in result[16..32].iter().enumerate() {
            assert!(
                (val - 14.0).abs() < 0.01,
                "Expected 14.0 at {}, got {}",
                i + 16,
                val
            );
        }
    }

    #[test]
    fn test_dequantize_q4_1_basic() {
        // Q4_1: scale + min, no offset subtraction
        let scale: f32 = 1.0;
        let min: f32 = -5.0;
        let scale_bits = f32_to_half(scale);
        let min_bits = f32_to_half(min);

        let mut data = Vec::with_capacity(20);
        data.extend_from_slice(&scale_bits.to_le_bytes());
        data.extend_from_slice(&min_bits.to_le_bytes());

        // All nibbles = 0
        data.extend(std::iter::repeat(0x00).take(16));

        let result = dequantize_q4_1(&data, 32).unwrap();
        assert_eq!(result.len(), 32);

        // y = 0 * 1.0 + (-5.0) = -5.0
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - (-5.0)).abs() < 0.01,
                "Expected -5.0 at {}, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        // Test with 2 blocks (64 elements)
        let scale1: f32 = 1.0;
        let scale2: f32 = 2.0;

        let mut data = Vec::with_capacity(68);

        // Block 1
        data.extend_from_slice(&f32_to_half(scale1).to_le_bytes());
        for i in 0..32u8 {
            data.push(i);
        }

        // Block 2
        data.extend_from_slice(&f32_to_half(scale2).to_le_bytes());
        for i in 0..32u8 {
            data.push(i);
        }

        let result = dequantize_q8_0(&data, 64).unwrap();
        assert_eq!(result.len(), 64);

        // First block: y = i * 1.0
        for (i, &val) in result[..32].iter().enumerate() {
            assert!(
                (val - i as f32).abs() < 0.01,
                "Block1 mismatch at {}: got {}, expected {}",
                i,
                val,
                i as f32
            );
        }
        // Second block: y = i * 2.0
        for (i, &val) in result[32..64].iter().enumerate() {
            assert!(
                (val - (i as f32 * 2.0)).abs() < 0.01,
                "Block2 mismatch at {}: got {}, expected {}",
                i,
                val,
                i as f32 * 2.0
            );
        }
    }

    #[test]
    fn test_dequantize_invalid_element_count() {
        // Q8_0 requires elements divisible by 32
        let data = vec![0u8; 34];
        let result = dequantize_q8_0(&data, 31);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("divisible by 32"));

        // Q4_0 requires elements divisible by 32
        let data = vec![0u8; 18];
        let result = dequantize_q4_0(&data, 16);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("divisible by 32"));
    }

    #[test]
    fn test_dequantize_data_too_short() {
        // Q8_0: needs 34 bytes per block
        let data = vec![0u8; 33];
        let result = dequantize_q8_0(&data, 32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));

        // Q4_0: needs 18 bytes per block
        let data = vec![0u8; 17];
        let result = dequantize_q4_0(&data, 32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_block_size_functions() {
        // Simple quants
        assert_eq!(get_block_size(&GGMLType::Q8_0), Some(34));
        assert_eq!(get_block_size(&GGMLType::Q4_0), Some(18));
        assert_eq!(get_block_size(&GGMLType::Q4_1), Some(20));
        assert_eq!(get_block_size(&GGMLType::Q5_0), Some(22));
        assert_eq!(get_block_size(&GGMLType::Q5_1), Some(24));
        assert_eq!(get_block_size(&GGMLType::Q8_1), Some(36));
        // K-quants
        assert_eq!(get_block_size(&GGMLType::Q2K), Some(84));
        assert_eq!(get_block_size(&GGMLType::Q3K), Some(110));
        assert_eq!(get_block_size(&GGMLType::Q4K), Some(144));
        assert_eq!(get_block_size(&GGMLType::Q5K), Some(176));
        assert_eq!(get_block_size(&GGMLType::Q6K), Some(210));

        // Elements per block
        assert_eq!(get_block_elements(&GGMLType::Q8_0), Some(32));
        assert_eq!(get_block_elements(&GGMLType::Q4_0), Some(32));
        // K-quants all use 256 elements per super-block
        assert_eq!(get_block_elements(&GGMLType::Q2K), Some(256));
        assert_eq!(get_block_elements(&GGMLType::Q3K), Some(256));
        assert_eq!(get_block_elements(&GGMLType::Q4K), Some(256));
        assert_eq!(get_block_elements(&GGMLType::Q5K), Some(256));
        assert_eq!(get_block_elements(&GGMLType::Q6K), Some(256));
    }

    #[test]
    fn test_format_metadata_value() {
        assert_eq!(format_metadata_value(&GGUFMetadataValue::Uint32(42)), "42");
        assert_eq!(
            format_metadata_value(&GGUFMetadataValue::String("test".to_string())),
            "test"
        );
        assert_eq!(
            format_metadata_value(&GGUFMetadataValue::Bool(true)),
            "true"
        );
    }

    #[test]
    fn test_file_not_found() {
        let result = load_gguf("/nonexistent/path/model.gguf");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    #[test]
    fn test_info_file_not_found() {
        let result = gguf_info("/nonexistent/path/model.gguf");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    // ==========================================================================
    // K-Quant Dequantization Tests
    // ==========================================================================

    #[test]
    fn test_dequantize_q2_k_basic() {
        // Q2_K: 84 bytes per 256 elements
        // Layout: scales[16], qs[64], d(2), dmin(2)
        let mut data = vec![0u8; 84];

        // Set d = 1.0, dmin = 0.0
        let d_bits = f32_to_half(1.0);
        let dmin_bits = f32_to_half(0.0);
        data[80..82].copy_from_slice(&d_bits.to_le_bytes());
        data[82..84].copy_from_slice(&dmin_bits.to_le_bytes());

        // Set scales: first scale nibble = 1, second (min) nibble = 0
        data[0..16].fill(0x01); // scale = 1, min = 0

        // Set all quants to 0 (2-bit values = 0)
        data[16..80].fill(0);

        let result = dequantize_q2_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        // All values should be 0 (d * scale * quant - dmin * min = 1*1*0 - 0*0*0 = 0)
        for v in &result {
            assert!(v.abs() < 1e-5, "Expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q2_k_all_ones() {
        // Q2_K: choose a pattern that produces quant=1 for all 256 values:
        // 0x55 = 0b01010101, so each 2-bit lane (shift 0/2/4/6) yields 1.
        let mut data = vec![0u8; 84];

        // d = 1.0, dmin = 0.0
        let d_bits = f32_to_half(1.0);
        let dmin_bits = f32_to_half(0.0);
        data[80..82].copy_from_slice(&d_bits.to_le_bytes());
        data[82..84].copy_from_slice(&dmin_bits.to_le_bytes());

        // dl = d * 1, ml = dmin * 0
        data[0..16].fill(0x01);
        data[16..80].fill(0x55);

        let result = dequantize_q2_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        for v in &result {
            assert!((v - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q3_k_basic() {
        // Q3_K: 110 bytes per 256 elements
        // Layout: hmask[32], qs[64], scales[12], d(2)
        let mut data = vec![0u8; 110];

        // Set d = 1.0
        let d_bits = f32_to_half(1.0);
        data[108..110].copy_from_slice(&d_bits.to_le_bytes());

        // Set scales to 32 (so scale - 32 = 0)
        // The scales are packed in a complex way, but setting all to 32 gives zero scales
        data[96..108].fill(0x20); // This approximates scales of 32

        // Set hmask to all 1s (high bit = 0)
        data[0..32].fill(0xFF);

        let result = dequantize_q3_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        // Values depend on the complex scale unpacking, but should be finite
        for v in &result {
            assert!(v.is_finite(), "Got non-finite value: {}", v);
        }
    }

    #[test]
    fn test_dequantize_q4_k_basic() {
        // Q4_K: 144 bytes per 256 elements
        // Layout: d(2), dmin(2), scales[12], qs[128]
        let mut data = vec![0u8; 144];

        // Set d = 1.0, dmin = 0.0
        let d_bits = f32_to_half(1.0);
        let dmin_bits = f32_to_half(0.0);
        data[0..2].copy_from_slice(&d_bits.to_le_bytes());
        data[2..4].copy_from_slice(&dmin_bits.to_le_bytes());

        // Set scales: first 4 have scale in lower 6 bits
        // Set scale = 1, min = 0 for simplicity
        data[4..12].fill(0x01); // scale = 1

        // Set qs: 4-bit values packed in nibbles
        // Set all nibbles to 2 (so value = 2 * scale = 2)
        data[16..144].fill(0x22); // low nibble = 2, high nibble = 2

        let result = dequantize_q4_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        // Most values should be around 2 (with scale = 1)
        let sum: f32 = result.iter().sum();
        assert!(sum.is_finite(), "Sum should be finite");
    }

    #[test]
    fn test_dequantize_q5_k_basic() {
        // Q5_K: 176 bytes per 256 elements
        // Layout: d(2), dmin(2), scales[12], qh[32], qs[128]
        let mut data = vec![0u8; 176];

        // Set d = 1.0, dmin = 0.0
        let d_bits = f32_to_half(1.0);
        let dmin_bits = f32_to_half(0.0);
        data[0..2].copy_from_slice(&d_bits.to_le_bytes());
        data[2..4].copy_from_slice(&dmin_bits.to_le_bytes());

        // Set scales
        data[4..12].fill(0x01);

        // qh = 0 (no high bits)
        // qs: all nibbles = 3
        data[48..176].fill(0x33);

        let result = dequantize_q5_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        for v in &result {
            assert!(v.is_finite(), "Got non-finite value: {}", v);
        }
    }

    #[test]
    fn test_dequantize_q6_k_basic() {
        // Q6_K: 210 bytes per 256 elements
        // Layout: ql[128], qh[64], scales[16], d(2)
        let mut data = vec![0u8; 210];

        // Set d = 0.1
        let d_bits = f32_to_half(0.1);
        data[208..210].copy_from_slice(&d_bits.to_le_bytes());

        // Set scales to 1 (signed i8)
        data[192..208].fill(1);

        // Set ql: 4-bit low values in nibbles
        // Set all to 0x88 (nibbles = 8)
        data[0..128].fill(0x88);

        // qh = 0 (high 2 bits = 0)
        // So 6-bit value = 8, minus 32 offset = -24

        let result = dequantize_q6_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);
        // All values should be d * scale * (q - 32) = 0.1 * 1 * -24 = -2.4
        for v in &result {
            assert!((v - (-2.4)).abs() < 0.05, "Expected ~-2.4, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q6_k_with_scales() {
        // Test Q6_K with varying scales
        let mut data = vec![0u8; 210];

        // Set d = 1.0
        let d_bits = f32_to_half(1.0);
        data[208..210].copy_from_slice(&d_bits.to_le_bytes());

        // Set scales to 2 (signed i8)
        data[192..208].fill(2);

        // Set ql to all zeros, qh to all zeros
        // So 6-bit value = 0, minus 32 = -32
        // Result = d * scale * (0 - 32) = 1 * 2 * -32 = -64

        let result = dequantize_q6_k(&data, 256).unwrap();
        assert_eq!(result.len(), 256);

        // Check that values are close to expected
        for v in &result {
            assert!((v - (-64.0)).abs() < 0.5, "Expected ~-64, got {}", v);
        }
    }

    #[test]
    fn test_k_quant_data_too_short() {
        // Q2_K needs 84 bytes per 256 elements
        let data = vec![0u8; 83]; // 1 byte short
        let result = dequantize_q2_k(&data, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));

        // Q3_K needs 110 bytes per 256 elements
        let data = vec![0u8; 109];
        let result = dequantize_q3_k(&data, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));

        // Q4_K needs 144 bytes per 256 elements
        let data = vec![0u8; 143];
        let result = dequantize_q4_k(&data, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));

        // Q5_K needs 176 bytes per 256 elements
        let data = vec![0u8; 175];
        let result = dequantize_q5_k(&data, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));

        // Q6_K needs 210 bytes per 256 elements
        let data = vec![0u8; 209];
        let result = dequantize_q6_k(&data, 256);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_k_quant_invalid_element_count() {
        // K-quants require elements divisible by 256
        let data = vec![0u8; 84];
        let result = dequantize_q2_k(&data, 255); // Not divisible by 256
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("divisible by 256"));

        let result = dequantize_q2_k(&data, 128); // Not divisible by 256
        assert!(result.is_err());
    }

    #[test]
    fn test_k_quant_multiple_blocks() {
        // Test with 2 super-blocks (512 elements)
        let data = vec![0u8; 84 * 2]; // 2 Q2_K blocks
        let result = dequantize_q2_k(&data, 512).unwrap();
        assert_eq!(result.len(), 512);

        let data = vec![0u8; 144 * 2]; // 2 Q4_K blocks
        let result = dequantize_q4_k(&data, 512).unwrap();
        assert_eq!(result.len(), 512);

        let data = vec![0u8; 210 * 2]; // 2 Q6_K blocks
        let result = dequantize_q6_k(&data, 512).unwrap();
        assert_eq!(result.len(), 512);
    }
}
