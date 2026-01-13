//! Cache management command implementation
//!
//! Provides CLI access to verification cache statistics, compaction time-series,
//! and cache management operations.

use dashprove_selfimp::{CacheSnapshot, VerificationCache};
use std::path::Path;
use std::time::{Duration, UNIX_EPOCH};

/// Configuration for cache stats command
pub struct CacheStatsConfig<'a> {
    /// Path to cache snapshot file (optional)
    pub snapshot_path: Option<&'a str>,
    /// Output format (text, json)
    pub format: &'a str,
    /// Show verbose output
    pub verbose: bool,
    /// Time window for compaction rates (in seconds)
    pub window_secs: u64,
}

/// Configuration for cache time-series command
pub struct CacheTimeSeriesConfig<'a> {
    /// Path to cache snapshot file (optional)
    pub snapshot_path: Option<&'a str>,
    /// Output format (text, json)
    pub format: &'a str,
    /// Show verbose output
    pub verbose: bool,
    /// Time window for analysis (in seconds)
    pub window_secs: u64,
    /// Maximum entries to show
    pub limit: usize,
}

/// Configuration for cache clear command
pub struct CacheClearConfig<'a> {
    /// Path to cache directory
    pub cache_dir: Option<&'a str>,
    /// Output format (text, json)
    pub format: &'a str,
    /// Dry run (don't actually clear)
    pub dry_run: bool,
}

/// Run cache stats command
pub fn run_cache_stats(config: CacheStatsConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let window = Duration::from_secs(config.window_secs);

    // Create a cache and optionally restore from snapshot
    let cache = if let Some(path) = config.snapshot_path {
        let snapshot_path = Path::new(path);
        if !snapshot_path.exists() {
            return Err(format!("Snapshot file not found: {}", path).into());
        }
        // Use from_bytes for auto-detection of gzip vs JSON format
        let data = std::fs::read(snapshot_path)?;
        let snapshot = CacheSnapshot::from_bytes(&data)?;
        let mut cache = VerificationCache::new();
        cache.restore_from_snapshot(&snapshot)?;
        cache
    } else {
        VerificationCache::new()
    };

    let stats = cache.stats();
    let compaction_counts = cache.historical_compaction_counts();
    let time_series_summary = cache.compaction_summary(window);

    if config.format == "json" {
        #[derive(serde::Serialize)]
        struct StatsJson {
            cache_entries: usize,
            hit_rate: f64,
            hits: u64,
            misses: u64,
            compaction_counts: CompactionCountsJson,
            compaction_rates: CompactionRatesJson,
            time_series_summary: TimeSeriesSummaryJson,
        }

        #[derive(serde::Serialize)]
        struct CompactionCountsJson {
            size_based: usize,
            time_based: usize,
            hit_rate_based: usize,
            partition_imbalance: usize,
            insert_based: usize,
            memory_based: usize,
            total: usize,
        }

        #[derive(serde::Serialize)]
        struct CompactionRatesJson {
            per_minute: f64,
            per_hour: f64,
            window_seconds: u64,
        }

        #[derive(serde::Serialize)]
        struct TimeSeriesSummaryJson {
            event_count: usize,
            total_entries_removed: usize,
            rate_per_minute: f64,
            rate_per_hour: f64,
        }

        let stats_json = StatsJson {
            cache_entries: stats.entry_count,
            hit_rate: stats.hit_rate(),
            hits: stats.hits,
            misses: stats.misses,
            compaction_counts: CompactionCountsJson {
                size_based: compaction_counts.size_based,
                time_based: compaction_counts.time_based,
                hit_rate_based: compaction_counts.hit_rate_based,
                partition_imbalance: compaction_counts.partition_imbalance,
                insert_based: compaction_counts.insert_based,
                memory_based: compaction_counts.memory_based,
                total: compaction_counts.total(),
            },
            compaction_rates: CompactionRatesJson {
                per_minute: cache.compaction_rate_per_minute(window),
                per_hour: cache.compaction_rate_per_hour(window),
                window_seconds: config.window_secs,
            },
            time_series_summary: TimeSeriesSummaryJson {
                event_count: time_series_summary.event_count,
                total_entries_removed: time_series_summary.total_entries_removed,
                rate_per_minute: time_series_summary.rate_per_minute,
                rate_per_hour: time_series_summary.rate_per_hour,
            },
        };

        println!("{}", serde_json::to_string_pretty(&stats_json)?);
    } else {
        println!("=== Verification Cache Statistics ===\n");

        println!("Cache Entries: {}", stats.entry_count);
        println!(
            "Hit Rate: {:.1}% ({} hits, {} misses)",
            stats.hit_rate() * 100.0,
            stats.hits,
            stats.misses
        );
        println!();

        println!("Compaction History (cumulative):");
        println!("  Total compactions: {}", compaction_counts.total());
        println!("  Size-based: {}", compaction_counts.size_based);
        println!("  Time-based: {}", compaction_counts.time_based);
        println!("  Hit-rate-based: {}", compaction_counts.hit_rate_based);
        println!(
            "  Partition imbalance: {}",
            compaction_counts.partition_imbalance
        );
        println!("  Insert-based: {}", compaction_counts.insert_based);
        println!("  Memory-based: {}", compaction_counts.memory_based);
        println!();

        println!("Compaction Rates ({}s window):", config.window_secs);
        println!(
            "  Per minute: {:.2}",
            cache.compaction_rate_per_minute(window)
        );
        println!("  Per hour: {:.2}", cache.compaction_rate_per_hour(window));
        println!();

        println!("Time-Series Summary:");
        println!("  Events in window: {}", time_series_summary.event_count);
        println!(
            "  Total entries removed: {}",
            time_series_summary.total_entries_removed
        );
        println!(
            "  Rate: {:.1}/min, {:.1}/hour",
            time_series_summary.rate_per_minute, time_series_summary.rate_per_hour
        );

        if config.verbose {
            println!();
            println!("Trigger Breakdown in Window:");
            let counts = &time_series_summary.counts_by_type;
            if counts.size_based > 0 {
                println!("  SizeBased: {}", counts.size_based);
            }
            if counts.time_based > 0 {
                println!("  TimeBased: {}", counts.time_based);
            }
            if counts.hit_rate_based > 0 {
                println!("  HitRateBased: {}", counts.hit_rate_based);
            }
            if counts.partition_imbalance > 0 {
                println!("  PartitionImbalance: {}", counts.partition_imbalance);
            }
            if counts.insert_based > 0 {
                println!("  InsertBased: {}", counts.insert_based);
            }
            if counts.memory_based > 0 {
                println!("  MemoryBased: {}", counts.memory_based);
            }
        }
    }

    Ok(())
}

/// Run cache time-series command
pub fn run_cache_time_series(
    config: CacheTimeSeriesConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let window = Duration::from_secs(config.window_secs);

    // Create a cache and optionally restore from snapshot
    let cache = if let Some(path) = config.snapshot_path {
        let snapshot_path = Path::new(path);
        if !snapshot_path.exists() {
            return Err(format!("Snapshot file not found: {}", path).into());
        }
        // Use from_bytes for auto-detection of gzip vs JSON format
        let data = std::fs::read(snapshot_path)?;
        let snapshot = CacheSnapshot::from_bytes(&data)?;
        let mut cache = VerificationCache::new();
        cache.restore_from_snapshot(&snapshot)?;
        cache
    } else {
        VerificationCache::new()
    };

    let time_series = cache.compaction_time_series();
    let entries_in_window = time_series.entries_in_window(window);
    let all_entries = time_series.entries();

    if config.format == "json" {
        #[derive(serde::Serialize)]
        struct TimeSeriesJson {
            total_entries: usize,
            entries_in_window: usize,
            window_seconds: u64,
            oldest_timestamp: Option<u64>,
            newest_timestamp: Option<u64>,
            rate_per_minute: f64,
            rate_per_hour: f64,
            entries: Vec<EntryJson>,
        }

        #[derive(serde::Serialize)]
        struct EntryJson {
            timestamp_secs: u64,
            trigger_type: String,
            entries_removed: usize,
        }

        let entries: Vec<EntryJson> = all_entries
            .into_iter()
            .take(config.limit)
            .map(|e| EntryJson {
                timestamp_secs: e.timestamp_secs,
                trigger_type: format!("{:?}", e.trigger_type),
                entries_removed: e.entries_removed,
            })
            .collect();

        let ts_json = TimeSeriesJson {
            total_entries: time_series.len(),
            entries_in_window: entries_in_window.len(),
            window_seconds: config.window_secs,
            oldest_timestamp: time_series
                .oldest_timestamp()
                .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()),
            newest_timestamp: time_series
                .newest_timestamp()
                .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()),
            rate_per_minute: time_series.rate_per_minute(window),
            rate_per_hour: time_series.rate_per_hour(window),
            entries,
        };

        println!("{}", serde_json::to_string_pretty(&ts_json)?);
    } else {
        println!("=== Compaction Time-Series ===\n");

        println!("Total events: {}", time_series.len());
        println!(
            "Events in window ({}s): {}",
            config.window_secs,
            entries_in_window.len()
        );

        if let Some(oldest) = time_series.oldest_timestamp() {
            let secs = oldest
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            println!("Oldest timestamp: {} (unix)", secs);
        }
        if let Some(newest) = time_series.newest_timestamp() {
            let secs = newest
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            println!("Newest timestamp: {} (unix)", secs);
        }

        println!();
        println!("Rates:");
        println!("  Per minute: {:.2}", time_series.rate_per_minute(window));
        println!("  Per hour: {:.2}", time_series.rate_per_hour(window));
        println!();

        if time_series.is_empty() {
            println!("No compaction events recorded yet.");
        } else {
            println!(
                "Recent Events (showing up to {}):",
                config.limit.min(time_series.len())
            );
            println!();

            let all_entries = time_series.entries();
            for (i, entry) in all_entries.iter().take(config.limit).enumerate() {
                println!(
                    "  {}. [{}] {:?} - {} entries removed",
                    i + 1,
                    entry.timestamp_secs,
                    entry.trigger_type,
                    entry.entries_removed
                );
            }

            if config.verbose && all_entries.len() > config.limit {
                println!();
                println!("  ... and {} more events", all_entries.len() - config.limit);
            }
        }

        if config.verbose {
            println!();
            println!("Trigger Distribution (in window):");
            let counts = time_series.counts_by_type(window);
            if counts.size_based > 0 {
                println!("  SizeBased: {}", counts.size_based);
            }
            if counts.time_based > 0 {
                println!("  TimeBased: {}", counts.time_based);
            }
            if counts.hit_rate_based > 0 {
                println!("  HitRateBased: {}", counts.hit_rate_based);
            }
            if counts.partition_imbalance > 0 {
                println!("  PartitionImbalance: {}", counts.partition_imbalance);
            }
            if counts.insert_based > 0 {
                println!("  InsertBased: {}", counts.insert_based);
            }
            if counts.memory_based > 0 {
                println!("  MemoryBased: {}", counts.memory_based);
            }
        }
    }

    Ok(())
}

/// Run cache clear command
pub fn run_cache_clear(config: CacheClearConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = config.cache_dir.unwrap_or("~/.dashprove/cache");
    // Expand ~ manually
    let expanded = if let Some(stripped) = cache_dir.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            home.join(stripped).to_string_lossy().to_string()
        } else {
            cache_dir.to_string()
        }
    } else {
        cache_dir.to_string()
    };
    let path = Path::new(&expanded);

    if config.format == "json" {
        #[derive(serde::Serialize)]
        struct ClearJson {
            path: String,
            exists: bool,
            dry_run: bool,
            cleared: bool,
        }

        let exists = path.exists();
        let cleared = if !config.dry_run && exists {
            std::fs::remove_dir_all(path).is_ok()
        } else {
            false
        };

        let clear_json = ClearJson {
            path: cache_dir.to_string(),
            exists,
            dry_run: config.dry_run,
            cleared,
        };

        println!("{}", serde_json::to_string_pretty(&clear_json)?);
    } else {
        println!("=== Cache Clear ===\n");
        println!("Cache directory: {}", cache_dir);

        if !path.exists() {
            println!("Cache directory does not exist.");
            return Ok(());
        }

        if config.dry_run {
            println!("DRY RUN: Would clear cache at {}", cache_dir);
            println!("(Run without --dry-run to actually clear)");
        } else {
            std::fs::remove_dir_all(path)?;
            println!("Cache cleared successfully.");
        }
    }

    Ok(())
}

/// Configuration for cache autosave command
pub struct CacheAutosaveConfig<'a> {
    /// Path to cache snapshot file (optional)
    pub snapshot_path: Option<&'a str>,
    /// Output format (text, json)
    pub format: &'a str,
    /// Show verbose output
    pub verbose: bool,
}

/// Run cache autosave command
pub fn run_cache_autosave(
    config: CacheAutosaveConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load snapshot if provided
    let snapshot = if let Some(path) = config.snapshot_path {
        let snapshot_path = Path::new(path);
        if !snapshot_path.exists() {
            return Err(format!("Snapshot file not found: {}", path).into());
        }
        // Use from_bytes for auto-detection of gzip vs JSON format
        let data = std::fs::read(snapshot_path)?;
        let snapshot = CacheSnapshot::from_bytes(&data)?;
        Some(snapshot)
    } else {
        None
    };

    // Get autosave metrics from snapshot
    let autosave_metrics = snapshot.as_ref().and_then(|s| s.autosave_metrics.clone());

    if config.format == "json" {
        #[derive(serde::Serialize)]
        struct AutosaveJson {
            has_metrics: bool,
            metrics: Option<AutosaveMetricsJson>,
        }

        #[derive(serde::Serialize)]
        struct AutosaveMetricsJson {
            save_count: usize,
            error_count: usize,
            skip_count: usize,
            forced_save_count: usize,
            total_bytes_written: u64,
            last_save_bytes: u64,
            avg_bytes_per_save: u64,
            interval_ms: u64,
            compressed: bool,
            session_duration_secs: u64,
            success_rate: f64,
            save_reasons: SaveReasonsJson,
            last_save_reason: Option<String>,
        }

        #[derive(serde::Serialize)]
        struct SaveReasonsJson {
            initial: usize,
            interval: usize,
            stale_data: usize,
            coalesced: usize,
        }

        let json = AutosaveJson {
            has_metrics: autosave_metrics.is_some(),
            metrics: autosave_metrics.map(|m| AutosaveMetricsJson {
                save_count: m.save_count,
                error_count: m.error_count,
                skip_count: m.skip_count,
                forced_save_count: m.forced_save_count,
                total_bytes_written: m.total_bytes_written,
                last_save_bytes: m.last_save_bytes,
                avg_bytes_per_save: m.avg_bytes_per_save(),
                interval_ms: m.interval_ms,
                compressed: m.compressed,
                session_duration_secs: m.session_duration_secs(),
                success_rate: m.success_rate(),
                save_reasons: SaveReasonsJson {
                    initial: m.save_reason_counts.initial,
                    interval: m.save_reason_counts.interval,
                    stale_data: m.save_reason_counts.stale_data,
                    coalesced: m.save_reason_counts.coalesced,
                },
                last_save_reason: m.last_save_reason.map(|r| format!("{}", r)),
            }),
        };

        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("=== Autosave Session Metrics ===\n");

        match autosave_metrics {
            Some(m) => {
                println!("Save Operations:");
                println!("  Successful saves: {}", m.save_count);
                println!("  Skipped saves: {}", m.skip_count);
                println!("  Forced saves: {}", m.forced_save_count);
                println!("  Errors: {}", m.error_count);
                println!("  Success rate: {:.1}%", m.success_rate() * 100.0);
                println!();

                println!("Data Written:");
                println!("  Total bytes: {}", format_bytes(m.total_bytes_written));
                println!("  Last save: {}", format_bytes(m.last_save_bytes));
                println!("  Avg per save: {}", format_bytes(m.avg_bytes_per_save()));
                println!(
                    "  Compression: {}",
                    if m.compressed { "enabled" } else { "disabled" }
                );
                println!();

                println!("Session Info:");
                println!("  Duration: {} seconds", m.session_duration_secs());
                println!("  Interval: {} ms", m.interval_ms);
                if let Some(reason) = m.last_save_reason {
                    println!("  Last save reason: {}", reason);
                }

                if config.verbose {
                    println!();
                    println!("Save Reason Breakdown:");
                    let reasons = &m.save_reason_counts;
                    println!("  Initial: {}", reasons.initial);
                    println!("  Interval: {}", reasons.interval);
                    println!("  Stale data: {}", reasons.stale_data);
                    println!("  Coalesced: {}", reasons.coalesced);
                }
            }
            None => {
                println!("No autosave metrics found in snapshot.");
                println!();
                println!("Autosave metrics are recorded when:");
                println!("  - A cache autosave session is active");
                println!("  - The snapshot includes format version 5 or later");
                println!();
                if config.snapshot_path.is_none() {
                    println!("Tip: Provide a snapshot file with --snapshot <path>");
                } else if let Some(s) = snapshot {
                    println!("Snapshot format version: {}", s.format_version);
                    println!("(Metrics require format version 5+)");
                }
            }
        }
    }

    Ok(())
}

/// Format bytes in human-readable form
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_stats_no_snapshot() {
        let config = CacheStatsConfig {
            snapshot_path: None,
            format: "text",
            verbose: false,
            window_secs: 3600,
        };
        let result = run_cache_stats(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_stats_json_format() {
        let config = CacheStatsConfig {
            snapshot_path: None,
            format: "json",
            verbose: false,
            window_secs: 3600,
        };
        let result = run_cache_stats(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_time_series_no_snapshot() {
        let config = CacheTimeSeriesConfig {
            snapshot_path: None,
            format: "text",
            verbose: false,
            window_secs: 3600,
            limit: 20,
        };
        let result = run_cache_time_series(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_time_series_json_format() {
        let config = CacheTimeSeriesConfig {
            snapshot_path: None,
            format: "json",
            verbose: false,
            window_secs: 3600,
            limit: 20,
        };
        let result = run_cache_time_series(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_clear_dry_run() {
        let config = CacheClearConfig {
            cache_dir: Some("/nonexistent/path"),
            format: "text",
            dry_run: true,
        };
        let result = run_cache_clear(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_stats_missing_snapshot() {
        let config = CacheStatsConfig {
            snapshot_path: Some("/nonexistent/snapshot.json"),
            format: "text",
            verbose: false,
            window_secs: 3600,
        };
        let result = run_cache_stats(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_autosave_no_snapshot() {
        let config = CacheAutosaveConfig {
            snapshot_path: None,
            format: "text",
            verbose: false,
        };
        let result = run_cache_autosave(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_autosave_json_format() {
        let config = CacheAutosaveConfig {
            snapshot_path: None,
            format: "json",
            verbose: false,
        };
        let result = run_cache_autosave(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_autosave_missing_snapshot() {
        let config = CacheAutosaveConfig {
            snapshot_path: Some("/nonexistent/snapshot.json"),
            format: "text",
            verbose: false,
        };
        let result = run_cache_autosave(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 bytes");
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }
}
