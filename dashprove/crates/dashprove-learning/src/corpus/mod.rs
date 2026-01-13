//! Proof corpus storage and retrieval
//!
//! Stores successful proofs indexed by property features for similarity search.
//!
//! # Category-Filtered Search
//!
//! The corpus maintains a secondary index by PropertyCategory (hierarchical
//! property type) that enables fast category-filtered searches:
//!
//! ```ignore
//! // Find similar proofs only in the "Memory Safety" category (category 4)
//! let results = corpus.find_similar_in_category(&property, 4, 10);
//!
//! // Find similar proofs with specific subtype (e.g., DataRace = category 4, subtype 2)
//! let results = corpus.find_similar_in_subtype(&property, 4, 2, 10);
//! ```
//!
//! # LSH-Accelerated Search
//!
//! For large corpora (>1000 proofs), use the LSH index for approximate but faster
//! embedding-based similarity search:
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusLsh};
//!
//! // Build LSH index (do this once after loading or populating corpus)
//! let lsh = ProofCorpusLsh::build_auto_config(&corpus).unwrap();
//!
//! // Query using LSH (approximate, but ~10-30x faster for large corpora)
//! let results = lsh.find_similar_approximate(&corpus, &query_embedding, 10);
//! ```
//!
//! # PQ-LSH Hybrid Index
//!
//! For very large corpora (>10000 proofs), use the PQ-LSH hybrid index for both
//! memory efficiency and fast search:
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusPqLsh};
//!
//! // Build PQ-LSH index (requires at least 64-256 embeddings for training)
//! let pq_lsh = ProofCorpusPqLsh::build_auto_config(&corpus)?;
//!
//! // Query using PQ-LSH (memory-efficient and fast)
//! let results = pq_lsh.find_similar_approximate(&corpus, &query_embedding, 10);
//!
//! // Check memory savings
//! let stats = pq_lsh.memory_stats();
//! println!("Compression ratio: {:.1}x", stats.compression_ratio);
//! ```
//!
//! # OPQ-LSH Hybrid Index
//!
//! For even better recall with similar memory efficiency, use OPQ-LSH which adds
//! a learned rotation matrix to decorrelate dimensions before quantization:
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusOpqLsh};
//!
//! // Build OPQ-LSH index (10-30% lower quantization error than PQ-LSH)
//! let opq_lsh = ProofCorpusOpqLsh::build_auto_config(&corpus)?;
//!
//! // Query using OPQ-LSH
//! let results = opq_lsh.find_similar_approximate(&corpus, &query_embedding, 10);
//!
//! // Check quantization error (should be lower than PQ)
//! let error = opq_lsh.quantization_error(&corpus);
//! ```
//!
//! # IVFPQ Index (Billion-Scale)
//!
//! For very large corpora (>100K entries) where sub-linear search time is needed,
//! use IVFPQ which combines coarse quantization (IVF) with fine quantization (PQ):
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusIvfPq, IvfPqConfig};
//!
//! // Build IVFPQ index with auto-configured settings
//! let index = ProofCorpusIvfPq::build(&corpus)?;
//!
//! // Or with custom config for very large corpus
//! let config = IvfPqConfig::for_large_corpus();
//! let index = ProofCorpusIvfPq::build_with_config(&corpus, config)?;
//!
//! // Search with nprobe=16 (number of cells to probe)
//! let results = index.find_similar_approximate(&corpus, &query_embedding, 10, 16);
//!
//! // Check memory usage and list distribution
//! let mem_stats = index.memory_stats();
//! let list_stats = index.list_stats();
//! ```
//!
//! # IVFOPQ Index (Better Accuracy)
//!
//! For improved recall over IVFPQ, use IVFOPQ which adds OPQ's learned rotation matrix:
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusIvfOpq, IvfOpqConfig};
//!
//! // Build IVFOPQ index (10-30% lower quantization error than IVFPQ)
//! let config = IvfOpqConfig::for_large_corpus();
//! let index = ProofCorpusIvfOpq::build_with_config(&corpus, config)?;
//!
//! // Search with nprobe=16
//! let results = index.find_similar_approximate(&corpus, &query_embedding, 10, 16);
//!
//! // Compare quantization error
//! let error = index.measure_quantization_error(&corpus, 100);
//! ```
//!
//! # HNSW Index (Graph-Based)
//!
//! For high-recall approximate search with O(log N) time complexity, use HNSW
//! (Hierarchical Navigable Small World) which builds a multi-layer graph:
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusHnsw, HnswConfig};
//!
//! // Build HNSW index
//! let config = HnswConfig::default();
//! let index = ProofCorpusHnsw::build_with_config(&corpus, config)?;
//!
//! // Search (ef controls recall vs speed tradeoff)
//! let results = index.find_similar(&corpus, &query_embedding, 10);
//!
//! // Higher ef = better recall but slower
//! let high_recall = index.find_similar_with_ef(&corpus, &query_embedding, 10, 200);
//!
//! // Check graph structure
//! let graph_stats = index.graph_stats();
//! ```
//!
//! # HNSW-PQ Index (Memory-Efficient Graph-Based)
//!
//! For large corpora where HNSW's full embedding storage is too expensive, use
//! HNSW-PQ which combines HNSW's graph navigation with PQ's ~48x memory compression:
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusHnswPq, HnswPqConfig};
//!
//! // Build HNSW-PQ index (requires full embeddings at build time)
//! let config = HnswPqConfig::default();
//! let index = ProofCorpusHnswPq::build_with_config(&corpus, config)?;
//!
//! // Search (graph navigation + asymmetric PQ distance)
//! let results = index.find_similar(&corpus, &query_embedding, 10);
//!
//! // Check memory savings vs full HNSW
//! let stats = index.memory_stats();
//! println!("Compression: {:.1}x", stats.compression_ratio);
//! ```
//!
//! # HNSW-OPQ Index (Better Accuracy Graph-Based)
//!
//! For improved recall over HNSW-PQ with similar memory efficiency, use HNSW-OPQ
//! which adds OPQ's learned rotation matrix for 10-30% lower quantization error:
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusHnswOpq, HnswOpqConfig};
//!
//! // Build HNSW-OPQ index (requires full embeddings at build time, slower training)
//! let config = HnswOpqConfig::default();
//! let index = ProofCorpusHnswOpq::build_with_config(&corpus, config)?;
//!
//! // Search (graph navigation + asymmetric OPQ distance)
//! let results = index.find_similar(&corpus, &query_embedding, 10);
//!
//! // Compare quantization error (should be lower than HNSW-PQ)
//! let error = index.measure_quantization_error(&corpus, 100);
//!
//! // Check memory stats (includes rotation matrix overhead)
//! let stats = index.memory_stats();
//! println!("Rotation matrix: {} bytes", stats.rotation_bytes);
//! ```
//!
//! # HNSW-Binary Index (Extreme Compression)
//!
//! For the fastest builds and extreme memory compression, use HNSW-Binary which
//! combines HNSW navigation with binary quantization (sign-based encoding):
//!
//! ```ignore
//! use dashprove_learning::corpus::{ProofCorpus, ProofCorpusHnswBinary, HnswBinaryConfig};
//!
//! // Build HNSW-Binary index (no training required - instant quantization!)
//! let config = HnswBinaryConfig::default();
//! let index = ProofCorpusHnswBinary::build_with_config(&corpus, config)?;
//!
//! // Search (graph navigation + Hamming distance)
//! let results = index.find_similar(&corpus, &query_embedding, 10);
//!
//! // Check memory savings (32x compression)
//! let stats = index.memory_stats();
//! println!("Compression: {:.1}x", stats.compression_ratio);
//!
//! // Check recall (lower than PQ/OPQ but faster to build)
//! let recall = index.measure_recall(&corpus, &queries, 10);
//! ```
//!
//! # Index Selection Advisor
//!
//! Not sure which index to use? The [`IndexAdvisor`] can help you choose:
//!
//! ```ignore
//! use dashprove_learning::corpus::{IndexAdvisor, IndexRequirements};
//!
//! let advisor = IndexAdvisor::new();
//!
//! // Define your requirements
//! let requirements = IndexRequirements {
//!     corpus_size: 100_000,
//!     embedding_dim: 96,
//!     max_memory_mb: Some(50.0),
//!     min_recall: 0.85,
//!     ..Default::default()
//! };
//!
//! // Get a recommendation
//! let recommendation = advisor.recommend(&requirements);
//! println!("Use: {:?}", recommendation.index_type);
//! println!("Expected recall: {:.0}%", recommendation.estimates.expected_recall * 100.0);
//! println!("Memory: {:.1} MB", recommendation.estimates.memory_mb);
//!
//! // Or get all viable options sorted by score
//! let all = advisor.recommend_all(&requirements);
//! for rec in all {
//!     println!("{}: score={:.2}", rec.index_type, rec.score);
//! }
//!
//! // Quick recommendation by corpus size alone
//! let quick = IndexAdvisor::quick_recommend(100_000);
//! ```
//!
//! # Streaming and Batched Index Builds
//!
//! For very large corpora that don't fit in memory, use the streaming builders:
//!
//! ```ignore
//! use dashprove_learning::corpus::{StreamingIvfBuilder, StreamingIvfConfig};
//!
//! // Create builder
//! let config = StreamingIvfConfig::for_large_corpus();
//! let mut builder = StreamingIvfBuilder::new(config, 96)?;
//!
//! // Phase 1: Train on a sample
//! for (id, embedding) in sample_iter.take(10_000) {
//!     builder.add_training_sample(id, embedding);
//! }
//! builder.train()?;
//!
//! // Phase 2: Stream through all data
//! for (id, embedding) in full_data_iter {
//!     builder.add_entry(id, embedding)?;
//! }
//!
//! // Finalize
//! let index = builder.build()?;
//! ```
//!
//! For HNSW with progress tracking:
//!
//! ```ignore
//! use dashprove_learning::corpus::{BatchedHnswBuilder, BatchedHnswConfig};
//!
//! let mut builder = BatchedHnswBuilder::new(BatchedHnswConfig::default())?;
//! builder.set_progress_callback(|p| println!("Processed: {}", p.items_processed));
//!
//! for (id, embedding) in data_iter {
//!     builder.add_entry(id, embedding)?;
//! }
//!
//! let index = builder.build()?;
//! ```

mod advisor;
mod category;
mod history;
mod hnsw;
mod hnsw_binary;
mod hnsw_opq;
mod hnsw_pq;
mod ivfopq;
mod ivfpq;
mod lsh_index;
mod opq_lsh;
mod pq_lsh;
mod search;
mod storage;
mod streaming;
mod types;

#[cfg(test)]
mod tests;

pub use advisor::{
    CalibrationData, IndexAdvisor, IndexComparison, IndexEstimates, IndexRecommendation,
    IndexRequirements, IndexType,
};
pub use category::CategoryIndex;
pub use hnsw::{HnswConfig, HnswError, HnswGraphStats, HnswMemoryStats, ProofCorpusHnsw};
pub use hnsw_binary::{
    HnswBinaryConfig, HnswBinaryError, HnswBinaryGraphStats, HnswBinaryMemoryStats,
    ProofCorpusHnswBinary,
};
pub use hnsw_opq::{
    HnswOpqConfig, HnswOpqError, HnswOpqGraphStats, HnswOpqMemoryStats, ProofCorpusHnswOpq,
};
pub use hnsw_pq::{
    HnswPqConfig, HnswPqError, HnswPqGraphStats, HnswPqMemoryStats, ProofCorpusHnswPq,
};
pub use ivfopq::{IvfOpqConfig, IvfOpqMemoryStats, ProofCorpusIvfOpq};
pub use ivfpq::{IvfListStats, IvfPqConfig, IvfPqMemoryStats, ProofCorpusIvfPq};
pub use lsh_index::{
    ProofCorpusLsh, RecallReason, RecallStatus, RecallTuningDecision, RecallTuningResult,
};
pub use opq_lsh::{OpqLshConfig, OpqLshMemoryStats, ProofCorpusOpqLsh};
pub use pq_lsh::{PqLshConfig, PqLshMemoryStats, ProofCorpusPqLsh};
pub use storage::ProofCorpus;
pub use streaming::{
    BatchedHnswBuilder, BatchedHnswConfig, BatchedHnswIndex, BatchedHnswMemoryStats, BuildPhase,
    BuildProgress, StreamingBuildError, StreamingIvfBuilder, StreamingIvfConfig, StreamingIvfIndex,
    StreamingIvfListStats, StreamingIvfMemoryStats,
};
pub use types::{ProofEntry, ProofHistory, ProofId};
