//! Content hashing for content-addressable storage
//!
//! Uses BLAKE3 for fast, secure hashing of source code and clauses.

use std::fmt;
use std::path::Path;

/// A content hash representing a unique identifier for content
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentHash {
    /// The raw hash bytes
    bytes: [u8; 32],
    /// The kind of content this hash represents
    kind: HashKind,
}

/// The kind of content being hashed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashKind {
    /// Hash of a source file's content
    SourceFile,
    /// Hash of a function's normalized AST
    Function,
    /// Hash of a clause database entry
    Clause,
    /// Hash of verification context (config + dependencies)
    Context,
    /// Hash of a project (all files combined)
    Project,
}

impl ContentHash {
    /// Create a hash from source code content
    pub fn from_source(content: &str) -> Self {
        Self {
            bytes: *blake3::hash(content.as_bytes()).as_bytes(),
            kind: HashKind::SourceFile,
        }
    }

    /// Create a hash from source file path (reads and hashes the file)
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_source(&content))
    }

    /// Create a hash from function name and body
    pub fn from_function(name: &str, body: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(name.as_bytes());
        hasher.update(b"\x00"); // Separator
        hasher.update(body.as_bytes());
        Self {
            bytes: *hasher.finalize().as_bytes(),
            kind: HashKind::Function,
        }
    }

    /// Create a hash from clause data
    pub fn from_clause(clause: &[i32]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for lit in clause {
            hasher.update(&lit.to_le_bytes());
        }
        Self {
            bytes: *hasher.finalize().as_bytes(),
            kind: HashKind::Clause,
        }
    }

    /// Create a hash from verification context
    pub fn from_context(config: &str, deps: &[&str]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(config.as_bytes());
        hasher.update(b"\x00");
        for dep in deps {
            hasher.update(dep.as_bytes());
            hasher.update(b"\x00");
        }
        Self {
            bytes: *hasher.finalize().as_bytes(),
            kind: HashKind::Context,
        }
    }

    /// Create a hash from multiple file hashes (project hash)
    pub fn from_files(file_hashes: &[&ContentHash]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for hash in file_hashes {
            hasher.update(&hash.bytes);
        }
        Self {
            bytes: *hasher.finalize().as_bytes(),
            kind: HashKind::Project,
        }
    }

    /// Create from raw bytes and kind
    pub fn from_bytes(bytes: [u8; 32], kind: HashKind) -> Self {
        Self { bytes, kind }
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Get the kind of content
    pub fn kind(&self) -> HashKind {
        self.kind
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.bytes)
    }

    /// Parse from hex string
    pub fn from_hex(s: &str, kind: HashKind) -> Result<Self, hex::FromHexError> {
        let mut bytes = [0u8; 32];
        hex::decode_to_slice(s, &mut bytes)?;
        Ok(Self { bytes, kind })
    }

    /// Get a short prefix for display (8 hex chars = 4 bytes)
    pub fn short(&self) -> String {
        hex::encode(&self.bytes[..4])
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.short())
    }
}

impl fmt::Display for HashKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashKind::SourceFile => write!(f, "src"),
            HashKind::Function => write!(f, "fn"),
            HashKind::Clause => write!(f, "cls"),
            HashKind::Context => write!(f, "ctx"),
            HashKind::Project => write!(f, "prj"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ==================== ContentHash Tests ====================

    #[test]
    fn test_source_hash() {
        let hash1 = ContentHash::from_source("fn foo() {}");
        let hash2 = ContentHash::from_source("fn foo() {}");
        let hash3 = ContentHash::from_source("fn bar() {}");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.kind(), HashKind::SourceFile);
    }

    #[test]
    fn test_source_hash_empty_string() {
        let hash = ContentHash::from_source("");
        assert_eq!(hash.kind(), HashKind::SourceFile);
        // Empty string should still produce a valid hash
        assert_eq!(hash.as_bytes().len(), 32);
    }

    #[test]
    fn test_source_hash_unicode() {
        let hash1 = ContentHash::from_source("fn 日本語() {}");
        let hash2 = ContentHash::from_source("fn 日本語() {}");
        let hash3 = ContentHash::from_source("fn 中文() {}");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_source_hash_whitespace_matters() {
        let hash1 = ContentHash::from_source("fn foo() {}");
        let hash2 = ContentHash::from_source("fn foo()  {}"); // extra space
        let hash3 = ContentHash::from_source("fn foo()\n{}"); // newline

        assert_ne!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_ne!(hash2, hash3);
    }

    #[test]
    fn test_function_hash() {
        let hash1 = ContentHash::from_function("foo", "{}");
        let hash2 = ContentHash::from_function("foo", "{}");
        let hash3 = ContentHash::from_function("bar", "{}");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.kind(), HashKind::Function);
    }

    #[test]
    fn test_function_hash_name_vs_body() {
        let hash1 = ContentHash::from_function("foo", "bar");
        let hash2 = ContentHash::from_function("bar", "foo");
        // Different name+body combinations should produce different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_function_hash_empty_name() {
        let hash = ContentHash::from_function("", "body");
        assert_eq!(hash.kind(), HashKind::Function);
    }

    #[test]
    fn test_function_hash_empty_body() {
        let hash = ContentHash::from_function("name", "");
        assert_eq!(hash.kind(), HashKind::Function);
    }

    #[test]
    fn test_function_hash_both_empty() {
        let hash = ContentHash::from_function("", "");
        assert_eq!(hash.kind(), HashKind::Function);
    }

    #[test]
    fn test_clause_hash() {
        let hash1 = ContentHash::from_clause(&[1, -2, 3]);
        let hash2 = ContentHash::from_clause(&[1, -2, 3]);
        let hash3 = ContentHash::from_clause(&[1, 2, 3]);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.kind(), HashKind::Clause);
    }

    #[test]
    fn test_clause_hash_empty() {
        let hash = ContentHash::from_clause(&[]);
        assert_eq!(hash.kind(), HashKind::Clause);
    }

    #[test]
    fn test_clause_hash_single_literal() {
        let hash1 = ContentHash::from_clause(&[1]);
        let hash2 = ContentHash::from_clause(&[-1]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_clause_hash_order_matters() {
        let hash1 = ContentHash::from_clause(&[1, 2, 3]);
        let hash2 = ContentHash::from_clause(&[3, 2, 1]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_clause_hash_large_values() {
        let hash1 = ContentHash::from_clause(&[i32::MAX, i32::MIN, 0]);
        let hash2 = ContentHash::from_clause(&[i32::MAX, i32::MIN, 0]);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_context_hash() {
        let hash1 = ContentHash::from_context("config1", &["dep1", "dep2"]);
        let hash2 = ContentHash::from_context("config1", &["dep1", "dep2"]);
        let hash3 = ContentHash::from_context("config2", &["dep1", "dep2"]);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.kind(), HashKind::Context);
    }

    #[test]
    fn test_context_hash_empty_deps() {
        let hash = ContentHash::from_context("config", &[]);
        assert_eq!(hash.kind(), HashKind::Context);
    }

    #[test]
    fn test_context_hash_dep_order_matters() {
        let hash1 = ContentHash::from_context("config", &["a", "b"]);
        let hash2 = ContentHash::from_context("config", &["b", "a"]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_context_hash_empty_config() {
        let hash = ContentHash::from_context("", &["dep1"]);
        assert_eq!(hash.kind(), HashKind::Context);
    }

    #[test]
    fn test_project_hash() {
        let file1 = ContentHash::from_source("file1");
        let file2 = ContentHash::from_source("file2");

        let project1 = ContentHash::from_files(&[&file1, &file2]);
        let project2 = ContentHash::from_files(&[&file1, &file2]);
        let project3 = ContentHash::from_files(&[&file2, &file1]); // Different order

        assert_eq!(project1, project2);
        assert_ne!(project1, project3); // Order matters
        assert_eq!(project1.kind(), HashKind::Project);
    }

    #[test]
    fn test_project_hash_empty() {
        let project = ContentHash::from_files(&[]);
        assert_eq!(project.kind(), HashKind::Project);
    }

    #[test]
    fn test_project_hash_single_file() {
        let file = ContentHash::from_source("content");
        let project = ContentHash::from_files(&[&file]);
        assert_eq!(project.kind(), HashKind::Project);
        // Single file project hash should be different from the file hash itself
        assert_ne!(project.as_bytes(), file.as_bytes());
    }

    #[test]
    fn test_from_bytes() {
        let bytes: [u8; 32] = [0u8; 32];
        let hash = ContentHash::from_bytes(bytes, HashKind::Clause);
        assert_eq!(hash.as_bytes(), &bytes);
        assert_eq!(hash.kind(), HashKind::Clause);
    }

    #[test]
    fn test_from_bytes_all_kinds() {
        let bytes: [u8; 32] = [42u8; 32];
        for kind in [
            HashKind::SourceFile,
            HashKind::Function,
            HashKind::Clause,
            HashKind::Context,
            HashKind::Project,
        ] {
            let hash = ContentHash::from_bytes(bytes, kind);
            assert_eq!(hash.kind(), kind);
        }
    }

    #[test]
    fn test_from_file() {
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(temp, "fn test() {{}}").unwrap();

        let hash = ContentHash::from_file(temp.path()).unwrap();
        assert_eq!(hash.kind(), HashKind::SourceFile);
    }

    #[test]
    fn test_from_file_not_found() {
        let result = ContentHash::from_file(Path::new("/nonexistent/path/file.rs"));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_content_matches_source() {
        let content = "fn test() {}";
        let mut temp = NamedTempFile::new().unwrap();
        write!(temp, "{content}").unwrap();

        let file_hash = ContentHash::from_file(temp.path()).unwrap();
        let source_hash = ContentHash::from_source(content);
        assert_eq!(file_hash, source_hash);
    }

    // ==================== Hex Conversion Tests ====================

    #[test]
    fn test_hex_roundtrip() {
        let original = ContentHash::from_source("test content");
        let hex = original.to_hex();
        let parsed = ContentHash::from_hex(&hex, HashKind::SourceFile).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_to_hex_length() {
        let hash = ContentHash::from_source("test");
        let hex = hash.to_hex();
        assert_eq!(hex.len(), 64); // 32 bytes * 2 hex chars
    }

    #[test]
    fn test_from_hex_invalid_length() {
        let result = ContentHash::from_hex("abc", HashKind::SourceFile);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_hex_invalid_chars() {
        let result = ContentHash::from_hex(
            "gg00000000000000000000000000000000000000000000000000000000000000",
            HashKind::SourceFile,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_from_hex_all_zeros() {
        let hex = "0".repeat(64);
        let hash = ContentHash::from_hex(&hex, HashKind::Clause).unwrap();
        assert_eq!(hash.as_bytes(), &[0u8; 32]);
    }

    #[test]
    fn test_from_hex_all_ff() {
        let hex = "f".repeat(64);
        let hash = ContentHash::from_hex(&hex, HashKind::Clause).unwrap();
        assert_eq!(hash.as_bytes(), &[0xffu8; 32]);
    }

    #[test]
    fn test_short() {
        let hash = ContentHash::from_source("test");
        let short = hash.short();
        assert_eq!(short.len(), 8); // 4 bytes * 2 hex chars

        // short() should be prefix of to_hex()
        assert!(hash.to_hex().starts_with(&short));
    }

    // ==================== Display Tests ====================

    #[test]
    fn test_display() {
        let hash = ContentHash::from_source("test");
        let display = format!("{}", hash);
        assert!(display.starts_with("src:"));
        assert_eq!(display.len(), 3 + 1 + 8); // "src" + ":" + 8 hex chars
    }

    #[test]
    fn test_display_function() {
        let hash = ContentHash::from_function("name", "body");
        let display = format!("{}", hash);
        assert!(display.starts_with("fn:"));
    }

    #[test]
    fn test_display_clause() {
        let hash = ContentHash::from_clause(&[1, 2, 3]);
        let display = format!("{}", hash);
        assert!(display.starts_with("cls:"));
    }

    #[test]
    fn test_display_context() {
        let hash = ContentHash::from_context("config", &["dep"]);
        let display = format!("{}", hash);
        assert!(display.starts_with("ctx:"));
    }

    #[test]
    fn test_display_project() {
        let file = ContentHash::from_source("file");
        let hash = ContentHash::from_files(&[&file]);
        let display = format!("{}", hash);
        assert!(display.starts_with("prj:"));
    }

    // ==================== HashKind Tests ====================

    #[test]
    fn test_hash_kind_display() {
        assert_eq!(format!("{}", HashKind::SourceFile), "src");
        assert_eq!(format!("{}", HashKind::Function), "fn");
        assert_eq!(format!("{}", HashKind::Clause), "cls");
        assert_eq!(format!("{}", HashKind::Context), "ctx");
        assert_eq!(format!("{}", HashKind::Project), "prj");
    }

    #[test]
    fn test_hash_kind_debug() {
        assert!(format!("{:?}", HashKind::SourceFile).contains("SourceFile"));
        assert!(format!("{:?}", HashKind::Function).contains("Function"));
        assert!(format!("{:?}", HashKind::Clause).contains("Clause"));
        assert!(format!("{:?}", HashKind::Context).contains("Context"));
        assert!(format!("{:?}", HashKind::Project).contains("Project"));
    }

    #[test]
    fn test_hash_kind_clone() {
        let kind = HashKind::SourceFile;
        let cloned = kind;
        assert_eq!(kind, cloned);
    }

    #[test]
    fn test_hash_kind_copy() {
        let kind = HashKind::Function;
        let copied: HashKind = kind;
        assert_eq!(kind, copied);
    }

    #[test]
    fn test_hash_kind_eq() {
        assert_eq!(HashKind::SourceFile, HashKind::SourceFile);
        assert_ne!(HashKind::SourceFile, HashKind::Function);
        assert_ne!(HashKind::Clause, HashKind::Context);
    }

    // ==================== ContentHash Trait Tests ====================

    #[test]
    fn test_content_hash_debug() {
        let hash = ContentHash::from_source("test");
        let debug = format!("{:?}", hash);
        assert!(debug.contains("ContentHash"));
        assert!(debug.contains("bytes"));
        assert!(debug.contains("kind"));
    }

    #[test]
    fn test_content_hash_clone() {
        let hash = ContentHash::from_source("test");
        let cloned = hash.clone();
        assert_eq!(hash, cloned);
    }

    #[test]
    fn test_content_hash_hash_trait() {
        use std::collections::HashSet;

        let hash1 = ContentHash::from_source("test1");
        let hash2 = ContentHash::from_source("test2");
        let hash3 = ContentHash::from_source("test1"); // same as hash1

        let mut set = HashSet::new();
        set.insert(hash1.clone());
        set.insert(hash2);
        set.insert(hash3);

        // hash1 and hash3 are equal, so only 2 unique hashes
        assert_eq!(set.len(), 2);
    }

    // ==================== Determinism Tests ====================

    #[test]
    fn test_hash_determinism_source() {
        let content = "fn foo() { let x = 42; }";
        let hash1 = ContentHash::from_source(content);
        let hash2 = ContentHash::from_source(content);
        let hash3 = ContentHash::from_source(content);
        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_hash_determinism_function() {
        let hash1 = ContentHash::from_function("test", "body");
        let hash2 = ContentHash::from_function("test", "body");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_determinism_clause() {
        let clause = vec![1, -2, 3, -4, 5];
        let hash1 = ContentHash::from_clause(&clause);
        let hash2 = ContentHash::from_clause(&clause);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_determinism_context() {
        let hash1 = ContentHash::from_context("cfg", &["a", "b", "c"]);
        let hash2 = ContentHash::from_context("cfg", &["a", "b", "c"]);
        assert_eq!(hash1, hash2);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_large_content() {
        let large_content = "x".repeat(1_000_000); // 1MB
        let hash = ContentHash::from_source(&large_content);
        assert_eq!(hash.kind(), HashKind::SourceFile);
        assert_eq!(hash.as_bytes().len(), 32);
    }

    #[test]
    fn test_binary_content_in_source() {
        // Source with binary-like content (null bytes)
        let content = "fn foo() {}\0\0\0";
        let hash = ContentHash::from_source(content);
        assert_eq!(hash.kind(), HashKind::SourceFile);
    }

    #[test]
    fn test_long_clause() {
        let clause: Vec<i32> = (1..10000).collect();
        let hash = ContentHash::from_clause(&clause);
        assert_eq!(hash.kind(), HashKind::Clause);
    }

    #[test]
    fn test_many_dependencies() {
        let deps: Vec<&str> = (0..100).map(|_| "dep").collect();
        let hash = ContentHash::from_context("config", &deps);
        assert_eq!(hash.kind(), HashKind::Context);
    }

    #[test]
    fn test_different_kinds_different_hashes() {
        // Even with same underlying data processed differently, different kinds should
        // result in different hashes (due to different construction methods)
        let source = ContentHash::from_source("test");
        let func = ContentHash::from_function("test", "");

        // These won't be equal because from_function uses a separator
        assert_ne!(source.as_bytes(), func.as_bytes());
    }
}
