//! Lean5 Kernel - Trusted Type Checker
//!
//! This crate implements the core type checking algorithm for Lean5.
//! It is the trusted computing base - all proofs ultimately reduce to
//! kernel type checking.
//!
//! # Architecture
//!
//! The kernel consists of:
//! - Expression representation (`expr.rs`)
//! - Universe levels (`level.rs`)
//! - Environment with declarations (`env.rs`)
//! - Type checker (`tc.rs`)
//! - Definitional equality / conversion (`conv.rs`)
//! - Inductive types (`inductive.rs`)
//!
//! # Performance
//!
//! The kernel is designed for maximum performance:
//! - Small expression nodes (16 bytes target)
//! - Arena allocation
//! - Hash consing for structural sharing
//! - Aggressive caching of type inference results

pub mod cert;
pub mod conv;
pub mod env;
pub mod expr;
pub mod inductive;
mod lean4_compat;
pub mod level;
pub mod micro;
pub mod mode;
pub mod name;
pub mod quot;
pub mod tc; // Lean 4 compatibility tests

pub use env::{ConstantInfo, Declaration, EnvError, Environment};
pub use expr::{BinderInfo, Expr, FVarId, LevelVec, Literal, MDataMap, MDataValue};
pub use inductive::{
    Constructor, ConstructorVal, InductiveDecl, InductiveError, InductiveType, InductiveVal,
    RecursorArgOrder, RecursorRule, RecursorVal,
};
pub use level::Level;
pub use name::Name;
pub use quot::{QuotKind, QuotVal};
pub use tc::{LocalContext, LocalDecl, TypeChecker, TypeError};

pub use mode::{AxiomId, Lean5Mode, ModeError, SourceSystem};

pub use cert::{
    // Byte-level compression (LZ4)
    archive_cert,
    // Compression algorithm choice
    archive_cert_with_algorithm,
    archive_cert_with_algorithm_stats,
    archive_cert_with_stats,
    // Batch verification (parallel)
    batch_verify,
    batch_verify_sequential,
    batch_verify_sequential_with_stats,
    batch_verify_with_stats,
    batch_verify_with_stats_progress,
    batch_verify_with_stats_threads,
    batch_verify_with_threads,
    // Structure-sharing compression
    compress_cert,
    compress_cert_with_stats,
    decompress_cert,
    lz4_compress,
    lz4_decompress,
    replay_cert,
    // Streaming compression
    stream_certs_from_file,
    stream_certs_to_file,
    unarchive_cert,
    unarchive_cert_envelope,
    // Byte-level compression (Zstd)
    zstd_archive_cert,
    zstd_archive_cert_level,
    // Dictionary-based compression
    zstd_archive_cert_with_dict,
    zstd_archive_cert_with_dict_level,
    zstd_archive_cert_with_dict_stats,
    zstd_archive_cert_with_dict_stats_level,
    zstd_archive_cert_with_stats,
    zstd_archive_cert_with_stats_level,
    zstd_compress,
    zstd_compress_level,
    zstd_compress_with_dict,
    zstd_compress_with_dict_level,
    zstd_decompress,
    zstd_decompress_with_dict,
    zstd_unarchive_cert,
    zstd_unarchive_cert_with_dict,
    ArchiveStats,
    ArchiveVariantStats,
    BatchVerifyInput,
    BatchVerifyResult,
    BatchVerifyStats,
    ByteCompressError,
    CertArchive,
    CertArchiveEnvelope,
    CertArchiveError,
    CertDictionary,
    CertError,
    CertIdx,
    CertVerifier,
    CompressedCert,
    CompressedCertNode,
    CompressedExpr,
    CompressedLevel,
    CompressionAlgorithm,
    CompressionStats,
    DecompressError,
    DefEqStep,
    DictArchiveStats,
    DictCertArchive,
    DictCompressError,
    DictTrainError,
    ExprIdx,
    LevelIdx,
    ProofCert,
    StreamingArchiveHeader,
    StreamingCertReader,
    StreamingCertWriter,
    StreamingError,
    StreamingProgressCallback,
    StreamingStats,
    ZstdArchiveStats,
    ZstdCertArchive,
    ZstdCompressError,
};
pub use micro::{
    cross_validate_with_micro, MicroCert, MicroChecker, MicroError, MicroExpr, MicroLevel,
    TranslateError,
};
