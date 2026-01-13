//! SafeTensors format support for loading model weights.
//!
//! SafeTensors is a simple, safe, and fast format for storing tensors,
//! commonly used by Hugging Face for model weights.
//!
//! # Usage
//!
//! ```ignore
//! use gamma_onnx::safetensors::load_safetensors;
//!
//! let weights = load_safetensors("model.safetensors")?;
//! for (name, tensor) in weights.iter() {
//!     println!("{}: {:?}", name, tensor.shape());
//! }
//! ```

use crate::WeightStore;
use gamma_core::{GammaError, Result};
use ndarray::{ArrayD, IxDyn};
use safetensors::SafeTensors;
use std::path::Path;
use tracing::{debug, info, warn};

/// Load weights from a SafeTensors file.
///
/// Returns a `WeightStore` containing all tensors from the file.
/// Only f32 and f16 (converted to f32) tensors are supported.
///
/// # Arguments
///
/// * `path` - Path to the .safetensors file
///
/// # Returns
///
/// A `WeightStore` with all loaded tensors.
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    info!("Loading SafeTensors from: {}", path.display());

    if !path.exists() {
        return Err(GammaError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    // Read the file
    let data = std::fs::read(path)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to read file: {}", e)))?;

    // Parse SafeTensors
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to parse SafeTensors: {}", e)))?;

    let mut weights = WeightStore::new();
    let mut loaded_count = 0;
    let mut skipped_count = 0;

    for (name, tensor_view) in tensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = tensor_view.dtype();

        // Convert to f32 array
        let array: Option<ArrayD<f32>> = match dtype {
            safetensors::Dtype::F32 => {
                let data = tensor_view.data();
                // Safety: data is aligned and sized for f32
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                ArrayD::from_shape_vec(IxDyn(&shape), floats).ok()
            }
            safetensors::Dtype::F16 => {
                // Convert f16 to f32
                let data = tensor_view.data();
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half_to_f32(bits)
                    })
                    .collect();
                ArrayD::from_shape_vec(IxDyn(&shape), floats).ok()
            }
            safetensors::Dtype::BF16 => {
                // Convert bf16 to f32
                let data = tensor_view.data();
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16_to_f32(bits)
                    })
                    .collect();
                ArrayD::from_shape_vec(IxDyn(&shape), floats).ok()
            }
            safetensors::Dtype::F64 => {
                // Convert f64 to f32 with precision loss warning
                let data = tensor_view.data();
                let floats: Vec<f32> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        val as f32
                    })
                    .collect();
                debug!(
                    "Warning: converting f64 tensor '{}' to f32 with potential precision loss",
                    name
                );
                ArrayD::from_shape_vec(IxDyn(&shape), floats).ok()
            }
            safetensors::Dtype::I64 => {
                // Convert i64 to f32 (for shape tensors, etc.)
                let data = tensor_view.data();
                let floats: Vec<f32> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        let val = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        val as f32
                    })
                    .collect();
                ArrayD::from_shape_vec(IxDyn(&shape), floats).ok()
            }
            safetensors::Dtype::I32 => {
                // Convert i32 to f32
                let data = tensor_view.data();
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        val as f32
                    })
                    .collect();
                ArrayD::from_shape_vec(IxDyn(&shape), floats).ok()
            }
            _ => {
                warn!(
                    "Skipping tensor '{}' with unsupported dtype {:?}",
                    name, dtype
                );
                skipped_count += 1;
                None
            }
        };

        if let Some(arr) = array {
            debug!("Loaded tensor: {} shape {:?}", name, arr.shape());
            weights.insert(name.to_string(), arr);
            loaded_count += 1;
        }
    }

    info!(
        "Loaded {} tensors from SafeTensors ({} skipped)",
        loaded_count, skipped_count
    );

    Ok(weights)
}

/// Convert IEEE 754 half-precision (f16) to f32.
#[inline]
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = -14i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            let new_exp = ((127 + e) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (new_exp << 23) | (f << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC00000 | (frac << 13))
        }
    } else {
        // Normal
        let new_exp = (exp + 127 - 15) & 0xFF;
        f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
    }
}

/// Convert bfloat16 to f32.
/// BF16 has the same exponent range as f32, just truncated mantissa.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    // BF16 is just the top 16 bits of f32
    f32::from_bits((bits as u32) << 16)
}

/// Metadata about a SafeTensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorsInfo {
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total parameter count (sum of all tensor elements).
    pub param_count: usize,
    /// Tensor names and their shapes.
    pub tensors: Vec<(String, Vec<usize>, String)>, // (name, shape, dtype)
}

/// Get information about a SafeTensors file without fully loading it.
pub fn safetensors_info<P: AsRef<Path>>(path: P) -> Result<SafeTensorsInfo> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(GammaError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let data = std::fs::read(path)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to read file: {}", e)))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to parse SafeTensors: {}", e)))?;

    let mut tensor_info = Vec::new();
    let mut param_count = 0;

    for (name, tensor_view) in tensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = format!("{:?}", tensor_view.dtype());
        let elements: usize = shape.iter().product();
        param_count += elements;
        tensor_info.push((name.to_string(), shape, dtype));
    }

    Ok(SafeTensorsInfo {
        tensor_count: tensor_info.len(),
        param_count,
        tensors: tensor_info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Create a test safetensors file with known content.
    fn create_test_safetensors() -> NamedTempFile {
        use safetensors::tensor::{serialize, TensorView};

        // Create test data
        let data1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2: Vec<f32> = vec![0.1, 0.2, 0.3];

        // Convert to bytes
        let bytes1: Vec<u8> = data1.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bytes2: Vec<u8> = data2.iter().flat_map(|f| f.to_le_bytes()).collect();

        let view1 = TensorView::new(safetensors::Dtype::F32, vec![2, 3], &bytes1).unwrap();
        let view2 = TensorView::new(safetensors::Dtype::F32, vec![3], &bytes2).unwrap();

        let tensors = vec![("layer1.weight", &view1), ("layer1.bias", &view2)];

        let serialized = serialize(tensors, None).unwrap();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_safetensors() {
        let file = create_test_safetensors();
        let weights = load_safetensors(file.path()).unwrap();

        assert_eq!(weights.len(), 2);

        let weight = weights.get("layer1.weight").unwrap();
        assert_eq!(weight.shape(), &[2, 3]);
        assert!((weight[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((weight[[0, 1]] - 2.0).abs() < 1e-6);
        assert!((weight[[1, 2]] - 6.0).abs() < 1e-6);

        let bias = weights.get("layer1.bias").unwrap();
        assert_eq!(bias.shape(), &[3]);
        assert!((bias[[0]] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_safetensors_info() {
        let file = create_test_safetensors();
        let info = safetensors_info(file.path()).unwrap();

        assert_eq!(info.tensor_count, 2);
        assert_eq!(info.param_count, 9); // 6 + 3
    }

    #[test]
    fn test_half_to_f32() {
        // Test zero
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x8000), -0.0);

        // Test one
        assert!((half_to_f32(0x3C00) - 1.0).abs() < 1e-6);

        // Test negative one
        assert!((half_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);

        // Test infinity
        assert!(half_to_f32(0x7C00).is_infinite());
        assert!(half_to_f32(0x7C00) > 0.0);

        // Test NaN
        assert!(half_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn test_bf16_to_f32() {
        // BF16 1.0 = 0x3F80
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);

        // BF16 -1.0 = 0xBF80
        assert!((bf16_to_f32(0xBF80) - (-1.0)).abs() < 1e-6);

        // BF16 0.0 = 0x0000
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_file_not_found() {
        let result = load_safetensors("/nonexistent/path/model.safetensors");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    #[test]
    fn test_safetensors_info_file_not_found() {
        let result = safetensors_info("/nonexistent/path/model.safetensors");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    #[test]
    fn test_half_to_f32_subnormal() {
        // Smallest positive subnormal: 0x0001 = 2^-24 ≈ 5.96e-8
        let smallest_subnormal = half_to_f32(0x0001);
        assert!(smallest_subnormal > 0.0);
        assert!(smallest_subnormal < 1e-6);
        assert!((smallest_subnormal - 5.960_464_5e-8).abs() < 1e-10);

        // Largest subnormal: 0x03FF = (2^-14) * (1 - 2^-10)
        let largest_subnormal = half_to_f32(0x03FF);
        assert!(largest_subnormal > 0.0);
        assert!(largest_subnormal < 6.1e-5);

        // Negative subnormal
        let neg_subnormal = half_to_f32(0x8001);
        assert!(neg_subnormal < 0.0);
        assert!((neg_subnormal - (-5.960_464_5e-8)).abs() < 1e-10);
    }

    #[test]
    fn test_half_to_f32_negative_infinity() {
        // Negative infinity: 0xFC00
        let neg_inf = half_to_f32(0xFC00);
        assert!(neg_inf.is_infinite());
        assert!(neg_inf < 0.0);
    }

    #[test]
    fn test_half_to_f32_various_normals() {
        // Test 2.0: 0x4000
        assert!((half_to_f32(0x4000) - 2.0).abs() < 1e-6);

        // Test 0.5: 0x3800
        assert!((half_to_f32(0x3800) - 0.5).abs() < 1e-6);

        // Test -2.0: 0xC000
        assert!((half_to_f32(0xC000) - (-2.0)).abs() < 1e-6);

        // Test 0.25: 0x3400
        assert!((half_to_f32(0x3400) - 0.25).abs() < 1e-6);

        // Test max finite (65504): 0x7BFF
        let max_finite = half_to_f32(0x7BFF);
        assert!((max_finite - 65504.0).abs() < 100.0);
        assert!(max_finite.is_finite());

        // Test min normal: 0x0400 = 2^-14 ≈ 6.1e-5
        let min_normal = half_to_f32(0x0400);
        assert!((min_normal - 6.103_515_6e-5).abs() < 1e-8);
    }

    #[test]
    fn test_half_to_f32_nan_propagates() {
        // Multiple NaN representations should all give NaN
        assert!(half_to_f32(0x7E00).is_nan()); // Quiet NaN
        assert!(half_to_f32(0x7C01).is_nan()); // Signaling NaN
        assert!(half_to_f32(0x7FFF).is_nan()); // Max NaN
        assert!(half_to_f32(0xFE00).is_nan()); // Negative quiet NaN
    }

    #[test]
    fn test_bf16_to_f32_infinity() {
        // BF16 +infinity: 0x7F80
        let pos_inf = bf16_to_f32(0x7F80);
        assert!(pos_inf.is_infinite());
        assert!(pos_inf > 0.0);

        // BF16 -infinity: 0xFF80
        let neg_inf = bf16_to_f32(0xFF80);
        assert!(neg_inf.is_infinite());
        assert!(neg_inf < 0.0);
    }

    #[test]
    fn test_bf16_to_f32_nan() {
        // BF16 quiet NaN: 0x7FC0
        assert!(bf16_to_f32(0x7FC0).is_nan());

        // BF16 signaling NaN: 0x7F81
        assert!(bf16_to_f32(0x7F81).is_nan());

        // Negative NaN
        assert!(bf16_to_f32(0xFFC0).is_nan());
    }

    #[test]
    fn test_bf16_to_f32_various_values() {
        // BF16 2.0 = 0x4000
        assert!((bf16_to_f32(0x4000) - 2.0).abs() < 1e-5);

        // BF16 0.5 = 0x3F00
        assert!((bf16_to_f32(0x3F00) - 0.5).abs() < 1e-5);

        // BF16 -2.0 = 0xC000
        assert!((bf16_to_f32(0xC000) - (-2.0)).abs() < 1e-5);

        // Negative zero: 0x8000
        let neg_zero = bf16_to_f32(0x8000);
        assert_eq!(neg_zero, 0.0);
        assert!(neg_zero.is_sign_negative());
    }

    #[test]
    fn test_load_safetensors_invalid_file() {
        // Create a file with invalid safetensors content
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"not a valid safetensors file").unwrap();
        file.flush().unwrap();

        let result = load_safetensors(file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse"));
    }

    #[test]
    fn test_safetensors_info_invalid_file() {
        // Create a file with invalid safetensors content
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"not a valid safetensors file").unwrap();
        file.flush().unwrap();

        let result = safetensors_info(file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse"));
    }

    #[test]
    fn test_safetensors_info_empty_file() {
        // Create an empty file
        let file = NamedTempFile::new().unwrap();

        let result = safetensors_info(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_info_reports_dtype() {
        let file = create_test_safetensors();
        let info = safetensors_info(file.path()).unwrap();

        // Verify dtype is reported in tensor info
        for (name, _shape, dtype) in &info.tensors {
            assert!(
                dtype.contains("F32") || dtype.contains("F16"),
                "Unexpected dtype {} for tensor {}",
                dtype,
                name
            );
        }
    }

    #[test]
    fn test_load_safetensors_preserves_tensor_names() {
        let file = create_test_safetensors();
        let weights = load_safetensors(file.path()).unwrap();

        // Check that expected tensor names exist
        assert!(weights.get("layer1.weight").is_some());
        assert!(weights.get("layer1.bias").is_some());

        // Check non-existent tensor returns None
        assert!(weights.get("nonexistent").is_none());
    }

    #[test]
    fn test_load_safetensors_all_values() {
        let file = create_test_safetensors();
        let weights = load_safetensors(file.path()).unwrap();

        // Verify all values of weight tensor
        let weight = weights.get("layer1.weight").unwrap();
        let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            let row = i / 3;
            let col = i % 3;
            assert!(
                (weight[[row, col]] - expected_val).abs() < 1e-6,
                "Mismatch at [{}, {}]: got {}, expected {}",
                row,
                col,
                weight[[row, col]],
                expected_val
            );
        }

        // Verify all values of bias tensor
        let bias = weights.get("layer1.bias").unwrap();
        let expected_bias = [0.1f32, 0.2, 0.3];
        for (i, &expected_val) in expected_bias.iter().enumerate() {
            assert!(
                (bias[[i]] - expected_val).abs() < 1e-6,
                "Bias mismatch at [{}]: got {}, expected {}",
                i,
                bias[[i]],
                expected_val
            );
        }
    }

    #[test]
    fn test_safetensors_info_tensor_shapes() {
        let file = create_test_safetensors();
        let info = safetensors_info(file.path()).unwrap();

        // Find tensors and verify shapes
        let mut found_weight = false;
        let mut found_bias = false;

        for (name, shape, _dtype) in &info.tensors {
            if name == "layer1.weight" {
                assert_eq!(shape, &[2, 3], "Weight shape mismatch");
                found_weight = true;
            } else if name == "layer1.bias" {
                assert_eq!(shape, &[3], "Bias shape mismatch");
                found_bias = true;
            }
        }

        assert!(found_weight, "layer1.weight not found");
        assert!(found_bias, "layer1.bias not found");
    }

    /// Create a safetensors file with f16 data.
    fn create_test_safetensors_f16() -> NamedTempFile {
        use safetensors::tensor::{serialize, TensorView};

        // Create f16 test data: 1.0, 2.0, 3.0 in half precision
        let data: Vec<u8> = [
            0x3C00u16, // 1.0
            0x4000u16, // 2.0
            0x4200u16, // 3.0
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

        let view = TensorView::new(safetensors::Dtype::F16, vec![3], &data).unwrap();
        let tensors = vec![("f16_tensor", &view)];
        let serialized = serialize(tensors, None).unwrap();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_safetensors_f16_conversion() {
        let file = create_test_safetensors_f16();
        let weights = load_safetensors(file.path()).unwrap();

        let tensor = weights.get("f16_tensor").unwrap();
        assert_eq!(tensor.shape(), &[3]);

        // Verify conversion accuracy
        assert!((tensor[[0]] - 1.0).abs() < 1e-3);
        assert!((tensor[[1]] - 2.0).abs() < 1e-3);
        assert!((tensor[[2]] - 3.0).abs() < 1e-3);
    }

    /// Create a safetensors file with bf16 data.
    fn create_test_safetensors_bf16() -> NamedTempFile {
        use safetensors::tensor::{serialize, TensorView};

        // Create bf16 test data: 1.0, -1.0, 0.0 in bfloat16
        let data: Vec<u8> = [
            0x3F80u16, // 1.0
            0xBF80u16, // -1.0
            0x0000u16, // 0.0
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

        let view = TensorView::new(safetensors::Dtype::BF16, vec![3], &data).unwrap();
        let tensors = vec![("bf16_tensor", &view)];
        let serialized = serialize(tensors, None).unwrap();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_safetensors_bf16_conversion() {
        let file = create_test_safetensors_bf16();
        let weights = load_safetensors(file.path()).unwrap();

        let tensor = weights.get("bf16_tensor").unwrap();
        assert_eq!(tensor.shape(), &[3]);

        // Verify conversion accuracy
        assert!((tensor[[0]] - 1.0).abs() < 1e-5);
        assert!((tensor[[1]] - (-1.0)).abs() < 1e-5);
        assert!((tensor[[2]] - 0.0).abs() < 1e-5);
    }

    /// Create a safetensors file with i32 data.
    fn create_test_safetensors_i32() -> NamedTempFile {
        use safetensors::tensor::{serialize, TensorView};

        let data: Vec<u8> = [1i32, -2i32, 1000i32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let view = TensorView::new(safetensors::Dtype::I32, vec![3], &data).unwrap();
        let tensors = vec![("i32_tensor", &view)];
        let serialized = serialize(tensors, None).unwrap();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_safetensors_i32_conversion() {
        let file = create_test_safetensors_i32();
        let weights = load_safetensors(file.path()).unwrap();

        let tensor = weights.get("i32_tensor").unwrap();
        assert_eq!(tensor.shape(), &[3]);

        // Verify conversion
        assert!((tensor[[0]] - 1.0).abs() < 1e-5);
        assert!((tensor[[1]] - (-2.0)).abs() < 1e-5);
        assert!((tensor[[2]] - 1000.0).abs() < 1e-5);
    }

    /// Create a safetensors file with i64 data.
    fn create_test_safetensors_i64() -> NamedTempFile {
        use safetensors::tensor::{serialize, TensorView};

        let data: Vec<u8> = [100i64, -50i64, 0i64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let view = TensorView::new(safetensors::Dtype::I64, vec![3], &data).unwrap();
        let tensors = vec![("i64_tensor", &view)];
        let serialized = serialize(tensors, None).unwrap();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_safetensors_i64_conversion() {
        let file = create_test_safetensors_i64();
        let weights = load_safetensors(file.path()).unwrap();

        let tensor = weights.get("i64_tensor").unwrap();
        assert_eq!(tensor.shape(), &[3]);

        // Verify conversion
        assert!((tensor[[0]] - 100.0).abs() < 1e-5);
        assert!((tensor[[1]] - (-50.0)).abs() < 1e-5);
        assert!((tensor[[2]] - 0.0).abs() < 1e-5);
    }

    /// Create a safetensors file with f64 data.
    fn create_test_safetensors_f64() -> NamedTempFile {
        use safetensors::tensor::{serialize, TensorView};

        let data: Vec<u8> = [1.5f64, -2.5f64, 3.125f64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let view = TensorView::new(safetensors::Dtype::F64, vec![3], &data).unwrap();
        let tensors = vec![("f64_tensor", &view)];
        let serialized = serialize(tensors, None).unwrap();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_safetensors_f64_conversion() {
        let file = create_test_safetensors_f64();
        let weights = load_safetensors(file.path()).unwrap();

        let tensor = weights.get("f64_tensor").unwrap();
        assert_eq!(tensor.shape(), &[3]);

        // Verify conversion (with f64->f32 precision loss)
        assert!((tensor[[0]] - 1.5).abs() < 1e-5);
        assert!((tensor[[1]] - (-2.5)).abs() < 1e-5);
        assert!((tensor[[2]] - 3.125).abs() < 1e-5);
    }

    #[test]
    fn test_safetensors_info_param_count_accuracy() {
        // Create a safetensors with known element count
        use safetensors::tensor::{serialize, TensorView};

        // Tensor 1: 2x3 = 6 elements
        // Tensor 2: 4 elements
        // Total: 10 elements
        let data1: Vec<u8> = vec![0u8; 6 * 4]; // 6 f32 values
        let data2: Vec<u8> = vec![0u8; 4 * 4]; // 4 f32 values

        let view1 = TensorView::new(safetensors::Dtype::F32, vec![2, 3], &data1).unwrap();
        let view2 = TensorView::new(safetensors::Dtype::F32, vec![4], &data2).unwrap();
        let tensors = vec![("tensor1", &view1), ("tensor2", &view2)];
        let serialized = serialize(tensors, None).unwrap();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&serialized).unwrap();
        file.flush().unwrap();

        let info = safetensors_info(file.path()).unwrap();
        assert_eq!(info.tensor_count, 2);
        assert_eq!(info.param_count, 10); // 6 + 4 = 10
    }
}
