use flate2::read::GzDecoder;
use gamma_core::{GammaError, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub(crate) fn read_bytes_maybe_gzip(path: &Path) -> Result<Vec<u8>> {
    if !path.exists() {
        return Err(GammaError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let is_gzip = path.extension().and_then(|e| e.to_str()) == Some("gz");
    if !is_gzip {
        return std::fs::read(path)
            .map_err(|e| GammaError::ModelLoad(format!("Failed to read file: {}", e)));
    }

    let file = File::open(path)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to open file: {}", e)))?;
    let mut decoder = GzDecoder::new(file);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to decode gzip: {}", e)))?;
    Ok(out)
}

pub(crate) fn read_string_maybe_gzip(path: &Path) -> Result<String> {
    let bytes = read_bytes_maybe_gzip(path)?;
    String::from_utf8(bytes)
        .map_err(|e| GammaError::ModelLoad(format!("Failed to decode UTF-8: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp_file(bytes: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        file
    }

    fn write_temp_gz_file(bytes: &[u8]) -> NamedTempFile {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(bytes).unwrap();
        let gz_bytes = encoder.finish().unwrap();

        let mut file = tempfile::Builder::new().suffix(".gz").tempfile().unwrap();
        file.write_all(&gz_bytes).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_read_bytes_plain() {
        let file = write_temp_file(b"hello");
        let bytes = read_bytes_maybe_gzip(file.path()).unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn test_read_bytes_gzip() {
        let file = write_temp_gz_file(b"hello gzip");
        let bytes = read_bytes_maybe_gzip(file.path()).unwrap();
        assert_eq!(bytes, b"hello gzip");
    }

    #[test]
    fn test_read_string_plain() {
        let file = write_temp_file("hello".as_bytes());
        let s = read_string_maybe_gzip(file.path()).unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn test_read_string_gzip() {
        let file = write_temp_gz_file("hello gzip".as_bytes());
        let s = read_string_maybe_gzip(file.path()).unwrap();
        assert_eq!(s, "hello gzip");
    }

    #[test]
    fn test_missing_file_is_error() {
        let err = read_bytes_maybe_gzip(Path::new("/nonexistent/file.bin"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("File not found"), "{err}");
    }

    #[test]
    fn test_invalid_gzip_is_error() {
        let mut file = tempfile::Builder::new().suffix(".gz").tempfile().unwrap();
        file.write_all(b"not a gzip stream").unwrap();
        file.flush().unwrap();

        let err = read_bytes_maybe_gzip(file.path()).unwrap_err().to_string();
        assert!(err.contains("Failed to decode gzip"), "{err}");
    }

    #[test]
    fn test_invalid_utf8_is_error() {
        let file = write_temp_file(&[0xff, 0xfe, 0xfd]);
        let err = read_string_maybe_gzip(file.path()).unwrap_err().to_string();
        assert!(err.contains("Failed to decode UTF-8"), "{err}");
    }
}
