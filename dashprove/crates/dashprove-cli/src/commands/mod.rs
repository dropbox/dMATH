//! CLI command implementations
//!
//! Each command is in its own module for better organization.

pub mod analyze;
pub mod backends;
pub mod bisim;
pub mod cache;
pub mod check_tools;
pub mod common;
pub mod corpus;
pub mod expert;
pub mod explain;
pub mod export;
pub mod mbt;
pub mod miri;
pub mod monitor;
pub mod proof_search;
pub mod prove;
pub mod reputation;
pub mod research;
pub mod selfimp;
pub mod topics;
pub mod train;
pub mod verify;
pub mod verify_code;
pub mod verify_trace;
pub mod visualize;

// Re-export commonly used items
pub use analyze::run_analyze;
pub use backends::run_backends;
pub use bisim::{run_bisim, BisimConfig};
pub use cache::{
    run_cache_autosave, run_cache_clear, run_cache_stats, run_cache_time_series,
    CacheAutosaveConfig, CacheClearConfig, CacheStatsConfig, CacheTimeSeriesConfig,
};
pub use check_tools::{run_check_tools, CheckToolsConfig};
pub use corpus::{
    run_corpus_compare, run_corpus_history, run_corpus_search, run_corpus_stats,
    run_corpus_suggest_compare, run_cx_add, run_cx_classify, run_cx_record_clusters, run_cx_search,
    run_search,
};
pub use expert::{
    run_expert_backend, run_expert_compile, run_expert_error, run_expert_tactic,
    ExpertBackendConfig, ExpertCompileConfig, ExpertErrorConfig, ExpertTacticConfig,
};
pub use explain::run_explain;
pub use export::run_export;
pub use mbt::{run_mbt, MbtConfig};
pub use miri::{run_miri_cmd, MiriCmdConfig};
pub use monitor::{run_monitor, MonitorCmdConfig};
pub use proof_search::{run_proof_search, ProofSearchCmdConfig};
pub use prove::run_prove;
pub use reputation::{
    run_reputation_export, run_reputation_import, run_reputation_reset, run_reputation_stats,
    ReputationCmdConfig,
};
pub use research::{
    run_arxiv_fetch, run_github_fetch, run_paper_search, run_research_stats, ArxivFetchConfig,
    GithubFetchConfig, PaperSearchConfig,
};
pub use selfimp::{
    run_selfimp_gate, run_selfimp_history, run_selfimp_rollback, run_selfimp_status,
    run_selfimp_verify, SelfImpGateConfig, SelfImpHistoryConfig, SelfImpRollbackConfig,
    SelfImpStatusConfig, SelfImpVerifyConfig,
};
pub use topics::run_topics;
pub use train::{run_ensemble, run_train, run_tune, EnsembleConfig, TrainConfig, TuneConfig};
pub use verify::{run_verify, VerifyConfig};
pub use verify_code::{run_verify_code, VerifyCodeConfig};
pub use verify_trace::{run_verify_trace, VerifyTraceConfig};
pub use visualize::{run_cluster, run_visualize};
