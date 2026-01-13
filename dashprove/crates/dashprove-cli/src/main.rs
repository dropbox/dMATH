//! DashProve CLI binary

mod commands;

use clap::Parser;
use commands::{
    run_analyze, run_arxiv_fetch, run_backends, run_bisim, run_cache_autosave, run_cache_clear,
    run_cache_stats, run_cache_time_series, run_check_tools, run_cluster, run_corpus_compare,
    run_corpus_history, run_corpus_search, run_corpus_stats, run_corpus_suggest_compare,
    run_cx_add, run_cx_classify, run_cx_record_clusters, run_cx_search, run_ensemble,
    run_expert_backend, run_expert_compile, run_expert_error, run_expert_tactic, run_explain,
    run_export, run_github_fetch, run_mbt, run_miri_cmd, run_monitor, run_paper_search,
    run_proof_search, run_prove, run_reputation_export, run_reputation_import,
    run_reputation_reset, run_reputation_stats, run_research_stats, run_search, run_selfimp_gate,
    run_selfimp_history, run_selfimp_rollback, run_selfimp_status, run_selfimp_verify, run_topics,
    run_train, run_tune, run_verify, run_verify_code, run_verify_trace, run_visualize,
    train::SchedulerType, ArxivFetchConfig, BisimConfig, CacheAutosaveConfig, CacheClearConfig,
    CacheStatsConfig, CacheTimeSeriesConfig, CheckToolsConfig, EnsembleConfig, ExpertBackendConfig,
    ExpertCompileConfig, ExpertErrorConfig, ExpertTacticConfig, GithubFetchConfig, MbtConfig,
    MiriCmdConfig, MonitorCmdConfig, PaperSearchConfig, ProofSearchCmdConfig, ReputationCmdConfig,
    SelfImpGateConfig, SelfImpHistoryConfig, SelfImpRollbackConfig, SelfImpStatusConfig,
    SelfImpVerifyConfig, TrainConfig, TuneConfig, VerifyCodeConfig, VerifyConfig,
    VerifyTraceConfig,
};
use dashprove_cli::cli::{
    ArxivAction, CacheAction, Cli, Commands, CorpusAction, ExpertAction, GithubAction,
    ReputationAction, ResearchAction, SelfImpAction,
};
use tracing::error;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Verify {
            path,
            backends,
            timeout,
            skip_health_check,
            learn,
            data_dir,
            suggest,
            incremental,
            since,
            ml,
            ml_model,
            ml_confidence,
            verbose,
        } => {
            run_verify(VerifyConfig {
                path: &path,
                backends_filter: backends.as_deref(),
                timeout_secs: timeout,
                skip_health_check,
                learn,
                data_dir: data_dir.as_deref(),
                suggest,
                incremental,
                since: since.as_deref(),
                ml_enabled: ml || ml_model.is_some(),
                ml_model_path: ml_model.as_deref(),
                ml_min_confidence: ml_confidence,
                verbose,
            })
            .await
        }
        Commands::Export {
            path,
            target,
            output,
        } => run_export(&path, &target, output.as_deref()).await,
        Commands::Backends => {
            run_backends().await;
            Ok(())
        }
        Commands::CheckTools {
            verbose,
            category,
            missing,
            format,
        } => run_check_tools(CheckToolsConfig {
            verbose,
            category: category.as_deref(),
            missing_only: missing,
            format: &format,
        }),
        Commands::Explain { path, backend } => run_explain(&path, backend.as_deref()),
        Commands::Corpus { action } => match action {
            CorpusAction::Stats { data_dir } => run_corpus_stats(data_dir.as_deref()),
            CorpusAction::Search {
                query,
                data_dir,
                limit,
            } => run_corpus_search(&query, data_dir.as_deref(), limit),
            CorpusAction::CxSearch {
                path,
                data_dir,
                limit,
            } => run_cx_search(&path, data_dir.as_deref(), limit),
            CorpusAction::CxAdd {
                path,
                property,
                backend,
                cluster,
                data_dir,
            } => run_cx_add(
                &path,
                &property,
                &backend,
                cluster.as_deref(),
                data_dir.as_deref(),
            ),
            CorpusAction::CxClassify { path, data_dir } => {
                run_cx_classify(&path, data_dir.as_deref())
            }
            CorpusAction::CxRecordClusters {
                paths,
                threshold,
                data_dir,
            } => run_cx_record_clusters(&paths, threshold, data_dir.as_deref()),
            CorpusAction::History {
                corpus,
                period,
                format,
                output,
                from,
                to,
                data_dir,
            } => run_corpus_history(
                &corpus,
                &period,
                &format,
                output.as_deref(),
                from.as_deref(),
                to.as_deref(),
                data_dir.as_deref(),
            ),
            CorpusAction::Compare {
                corpus,
                baseline_from,
                baseline_to,
                compare_from,
                compare_to,
                period,
                format,
                output,
                data_dir,
            } => run_corpus_compare(
                &corpus,
                &baseline_from,
                &baseline_to,
                &compare_from,
                &compare_to,
                &period,
                &format,
                output.as_deref(),
                data_dir.as_deref(),
            ),
            CorpusAction::SuggestCompare {
                corpus,
                format,
                output,
                data_dir,
            } => {
                run_corpus_suggest_compare(&corpus, &format, output.as_deref(), data_dir.as_deref())
            }
        },
        Commands::Search {
            query,
            data_dir,
            limit,
        } => run_search(&query, data_dir.as_deref(), limit),
        Commands::Prove { path, hints } => run_prove(&path, hints),
        Commands::Monitor {
            path,
            target,
            output,
            assertions,
            logging,
            metrics,
        } => run_monitor(MonitorCmdConfig {
            path: &path,
            target: &target,
            output: output.as_deref(),
            assertions,
            logging,
            metrics,
        }),
        Commands::Visualize {
            path,
            format,
            output,
            title,
        } => run_visualize(&path, &format, output.as_deref(), title.as_deref()),
        Commands::Analyze { path, action } => run_analyze(&path, action),
        Commands::Cluster {
            paths,
            threshold,
            format,
            output,
            title,
        } => run_cluster(
            &paths,
            threshold,
            &format,
            output.as_deref(),
            title.as_deref(),
        ),
        Commands::Topics { topic } => run_topics(topic.as_deref()),
        Commands::VerifyCode {
            code,
            spec,
            timeout,
            verbose,
        } => {
            run_verify_code(VerifyCodeConfig {
                code_path: code.as_deref(),
                spec_path: &spec,
                timeout_secs: timeout,
                verbose,
            })
            .await
        }
        Commands::Train {
            data_dir,
            output,
            learning_rate,
            epochs,
            verbose,
            early_stopping,
            patience,
            min_delta,
            validation_split,
            lr_scheduler,
            lr_step_size,
            lr_gamma,
            lr_min,
            lr_warmup_epochs,
            checkpoint,
            checkpoint_dir,
            checkpoint_interval,
            keep_best,
            resume,
        } => {
            let scheduler_type = lr_scheduler.parse().unwrap_or_else(|e| {
                eprintln!("Warning: {}", e);
                SchedulerType::Constant
            });
            run_train(TrainConfig {
                data_dir: data_dir.as_deref(),
                output: output.as_deref(),
                learning_rate,
                epochs,
                verbose,
                early_stopping,
                patience,
                min_delta,
                validation_split,
                lr_scheduler: scheduler_type,
                lr_step_size,
                lr_gamma,
                lr_min,
                lr_warmup_epochs,
                checkpoint,
                checkpoint_dir: checkpoint_dir.as_deref(),
                checkpoint_interval,
                keep_best,
                resume: resume.as_deref(),
            })
        }
        Commands::Tune {
            method,
            data_dir,
            output,
            iterations,
            seed,
            lr_values,
            epoch_values,
            lr_min,
            lr_max,
            epochs_min,
            epochs_max,
            initial_samples,
            kappa,
            cv_folds,
            verbose,
        } => run_tune(TuneConfig {
            method: &method,
            data_dir: data_dir.as_deref(),
            output: output.as_deref(),
            iterations,
            seed,
            lr_values: &lr_values,
            epoch_values: &epoch_values,
            lr_min,
            lr_max,
            epochs_min,
            epochs_max,
            initial_samples,
            kappa,
            cv_folds,
            verbose,
        }),
        Commands::Ensemble {
            models,
            weights,
            method,
            output,
            data_dir,
            verbose,
        } => run_ensemble(EnsembleConfig {
            models: models.iter().map(String::as_str).collect(),
            weights: weights.as_deref(),
            method: &method,
            output: output.as_deref(),
            data_dir: data_dir.as_deref(),
            verbose,
        }),
        Commands::Bisim {
            oracle,
            subject,
            inputs,
            recorded_traces,
            timeout,
            threshold,
            ignore_timing,
            nondeterminism,
            verbose,
        } => run_bisim(BisimConfig {
            oracle: &oracle,
            subject: &subject,
            inputs: inputs.as_deref(),
            recorded_traces,
            timeout_secs: timeout,
            threshold,
            ignore_timing,
            nondeterminism: &nondeterminism,
            verbose,
        })
        .await
        .map_err(|e| e.into()),
        Commands::VerifyTrace {
            trace,
            spec,
            invariants,
            liveness,
            timeout,
            format,
            verbose,
        } => run_verify_trace(VerifyTraceConfig {
            trace_path: &trace,
            spec_path: spec.as_deref(),
            invariants,
            liveness,
            timeout_secs: timeout,
            verbose,
            format: &format,
        })
        .map_err(|e| e.into()),
        Commands::Mbt {
            spec,
            coverage,
            format,
            output,
            max_states,
            max_depth,
            max_test_length,
            max_tests,
            seed,
            timeout,
            verbose,
        } => run_mbt(MbtConfig {
            spec_path: &spec,
            coverage: &coverage,
            format: &format,
            output: output.as_deref(),
            max_states,
            max_depth,
            max_test_length,
            max_tests,
            seed,
            timeout_secs: timeout,
            verbose,
        })
        .map_err(|e| e.into()),
        Commands::Miri {
            path,
            test_filter,
            timeout,
            disable_isolation,
            skip_stacked_borrows,
            skip_data_races,
            track_raw_pointers,
            seed,
            format,
            output,
            generate_harness,
            setup_only,
            verbose,
        } => run_miri_cmd(MiriCmdConfig {
            path: &path,
            test_filter: test_filter.as_deref(),
            timeout_secs: timeout,
            disable_isolation,
            skip_stacked_borrows,
            skip_data_races,
            track_raw_pointers,
            seed,
            format: &format,
            output: output.as_deref(),
            generate_harness: generate_harness.as_deref(),
            setup_only,
            verbose,
        })
        .await
        .map_err(|e| e.into()),
        Commands::Expert { action } => match action {
            ExpertAction::Backend {
                spec,
                property_types,
                code_lang,
                tags,
                data_dir,
                format,
            } => run_expert_backend(ExpertBackendConfig {
                spec: spec.as_deref(),
                property_types: property_types.as_deref(),
                code_lang: code_lang.as_deref(),
                tags: tags.as_deref(),
                data_dir: data_dir.as_deref(),
                format: &format,
            })
            .await
            .map_err(|e| e.into()),
            ExpertAction::Error {
                message,
                file,
                backend,
                data_dir,
                format,
            } => run_expert_error(ExpertErrorConfig {
                message: message.as_deref(),
                file: file.as_deref(),
                backend: backend.as_deref(),
                data_dir: data_dir.as_deref(),
                format: &format,
            })
            .await
            .map_err(|e| e.into()),
            ExpertAction::Tactic {
                goal,
                backend,
                context,
                data_dir,
                format,
            } => run_expert_tactic(ExpertTacticConfig {
                goal: &goal,
                backend: &backend,
                context: context.as_deref(),
                data_dir: data_dir.as_deref(),
                format: &format,
            })
            .await
            .map_err(|e| e.into()),
            ExpertAction::Compile {
                spec,
                backend,
                data_dir,
                format,
            } => run_expert_compile(ExpertCompileConfig {
                spec: &spec,
                backend: &backend,
                data_dir: data_dir.as_deref(),
                format: &format,
            })
            .await
            .map_err(|e| e.into()),
        },
        Commands::Reputation { action } => match action {
            ReputationAction::Stats {
                data_dir,
                format,
                domains,
                backend,
            } => run_reputation_stats(ReputationCmdConfig {
                data_dir: data_dir.as_deref(),
                format: &format,
                show_domains: domains,
                backend: backend.as_deref(),
            }),
            ReputationAction::Reset { data_dir } => run_reputation_reset(data_dir.as_deref()),
            ReputationAction::Export { output, data_dir } => {
                run_reputation_export(data_dir.as_deref(), &output)
            }
            ReputationAction::Import {
                input,
                data_dir,
                merge,
            } => run_reputation_import(data_dir.as_deref(), &input, merge),
        },
        Commands::Research { action } => match action {
            ResearchAction::Arxiv(arxiv_action) => match arxiv_action {
                ArxivAction::Fetch {
                    output,
                    categories,
                    since,
                    max_per_category,
                    download_pdfs,
                    extract_text,
                    verbose,
                } => run_arxiv_fetch(ArxivFetchConfig {
                    output_dir: output.as_deref(),
                    categories: categories.as_deref(),
                    since: since.as_deref(),
                    max_per_category,
                    download_pdfs,
                    extract_text,
                    verbose,
                })
                .await
                .map_err(|e| e.into()),
                ArxivAction::Search {
                    query,
                    data_dir,
                    limit,
                    category,
                    format,
                } => run_paper_search(PaperSearchConfig {
                    query: &query,
                    data_dir: data_dir.as_deref(),
                    limit,
                    category: category.as_deref(),
                    format: &format,
                })
                .await
                .map_err(|e| e.into()),
            },
            ResearchAction::Github(github_action) => match github_action {
                GithubAction::Fetch {
                    output,
                    queries,
                    min_stars,
                    verbose,
                } => run_github_fetch(GithubFetchConfig {
                    output_dir: output.as_deref(),
                    queries: queries.as_deref(),
                    min_stars,
                    verbose,
                })
                .await
                .map_err(|e| e.into()),
            },
            ResearchAction::Stats { data_dir } => {
                run_research_stats(data_dir.as_deref()).map_err(|e| e.into())
            }
        },
        Commands::ProofSearch {
            path,
            backend,
            property,
            max_iterations,
            threshold,
            hints,
            tactics,
            propagate_to,
            data_dir,
            verbose,
            format,
            hierarchical,
            max_depth,
            complexity_threshold,
            induction_mode,
        } => {
            run_proof_search(ProofSearchCmdConfig {
                path: &path,
                backend: &backend,
                property: property.as_deref(),
                max_iterations,
                validation_threshold: threshold,
                hints,
                tactics,
                propagate_to,
                data_dir: data_dir.as_deref(),
                verbose,
                format: &format,
                hierarchical,
                max_decomposition_depth: max_depth,
                decomposition_complexity_threshold: complexity_threshold,
                induction_mode: &induction_mode,
            })
            .await
        }
        Commands::SelfImp { action } => match action {
            SelfImpAction::Status { format, verbose } => run_selfimp_status(SelfImpStatusConfig {
                format: &format,
                verbose,
            }),
            SelfImpAction::History {
                limit,
                format,
                verbose,
            } => run_selfimp_history(SelfImpHistoryConfig {
                limit,
                format: &format,
                verbose,
            }),
            SelfImpAction::Verify {
                proposal,
                current_version,
                format,
                verbose,
                dry_run,
            } => run_selfimp_verify(SelfImpVerifyConfig {
                proposal: &proposal,
                current_version: &current_version,
                format: &format,
                verbose,
                dry_run,
            }),
            SelfImpAction::Rollback {
                target,
                format,
                verbose,
                dry_run,
            } => run_selfimp_rollback(SelfImpRollbackConfig {
                target_version: target.as_deref(),
                format: &format,
                verbose,
                dry_run,
            }),
            SelfImpAction::Gate { format, all_checks } => run_selfimp_gate(SelfImpGateConfig {
                format: &format,
                all_checks,
            }),
        },
        Commands::Cache { action } => match action {
            CacheAction::Stats {
                snapshot,
                format,
                verbose,
                window,
            } => run_cache_stats(CacheStatsConfig {
                snapshot_path: snapshot.as_deref(),
                format: &format,
                verbose,
                window_secs: window,
            }),
            CacheAction::TimeSeries {
                snapshot,
                format,
                verbose,
                window,
                limit,
            } => run_cache_time_series(CacheTimeSeriesConfig {
                snapshot_path: snapshot.as_deref(),
                format: &format,
                verbose,
                window_secs: window,
                limit,
            }),
            CacheAction::Clear {
                cache_dir,
                format,
                dry_run,
            } => run_cache_clear(CacheClearConfig {
                cache_dir: cache_dir.as_deref(),
                format: &format,
                dry_run,
            }),
            CacheAction::Autosave {
                snapshot,
                format,
                verbose,
            } => run_cache_autosave(CacheAutosaveConfig {
                snapshot_path: snapshot.as_deref(),
                format: &format,
                verbose,
            }),
        },
    };

    if let Err(e) = result {
        error!("{}", e);
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
