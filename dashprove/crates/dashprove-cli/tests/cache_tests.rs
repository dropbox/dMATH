//! Integration tests for cache CLI commands
//!
//! Tests the `dashprove cache stats`, `cache time-series`, and `cache clear` commands
//! with actual snapshot files containing compaction history.

use serial_test::serial;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dashprove_cache_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dashprove_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dashprove"))
}

/// Create a test snapshot JSON with cache entries and compaction history
fn create_test_snapshot(dir: &std::path::Path, with_time_series: bool) -> std::path::PathBuf {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build time series JSON if needed
    let time_series_json = if with_time_series {
        format!(
            r#",
  "compaction_time_series": {{
    "entries": [
      {{ "timestamp_secs": {}, "trigger_type": "SizeBased", "entries_removed": 50 }},
      {{ "timestamp_secs": {}, "trigger_type": "TimeBased", "entries_removed": 30 }},
      {{ "timestamp_secs": {}, "trigger_type": "HitRateBased", "entries_removed": 20 }},
      {{ "timestamp_secs": {}, "trigger_type": "InsertBased", "entries_removed": 15 }},
      {{ "timestamp_secs": {}, "trigger_type": "SizeBased", "entries_removed": 25 }}
    ],
    "max_entries": 1000,
    "head": 0,
    "len": 5
  }}"#,
            now_secs - 3600,
            now_secs - 1800,
            now_secs - 900,
            now_secs - 300,
            now_secs - 60
        )
    } else {
        String::new()
    };

    // Use recent timestamps for cached_at to avoid TTL expiration during restore
    // (default TTL is 1 hour, so entries cached within the last hour won't be expired)
    let snapshot_json = format!(
        r#"{{
  "format_version": 4,
  "created_at": {{ "secs_since_epoch": {}, "nanos_since_epoch": 0 }},
  "entries": [
    {{
      "version_hash": "abc123",
      "property_name": "test_property",
      "result": {{
        "property": {{ "name": "test_property", "passed": true, "status": "verified" }},
        "backends": ["mock"],
        "cached_at": {{ "secs_since_epoch": {}, "nanos_since_epoch": 0 }},
        "dependency_hash": "dep123",
        "confidence": 0.95
      }}
    }},
    {{
      "version_hash": "def456",
      "property_name": "another_property",
      "result": {{
        "property": {{ "name": "another_property", "passed": true, "status": "verified" }},
        "backends": ["mock", "z3"],
        "cached_at": {{ "secs_since_epoch": {}, "nanos_since_epoch": 0 }},
        "dependency_hash": "dep456",
        "confidence": 0.85
      }}
    }}
  ],
  "stats": {{
    "hits": 150,
    "misses": 50,
    "evictions": 10,
    "entry_count": 2
  }},
  "compaction_trigger_counts": {{
    "size_based": 5,
    "time_based": 3,
    "hit_rate_based": 2,
    "partition_imbalance": 1,
    "insert_based": 4,
    "memory_based": 0
  }}{}
}}"#,
        now_secs,
        now_secs - 60,  // 1 minute ago (well within TTL)
        now_secs - 300, // 5 minutes ago (still within 1 hour TTL)
        time_series_json
    );

    let path = dir.join("test_snapshot.json");
    std::fs::write(&path, snapshot_json).unwrap();
    path
}

/// Create an empty snapshot for edge case testing
fn create_empty_snapshot(dir: &std::path::Path) -> std::path::PathBuf {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let snapshot_json = format!(
        r#"{{
  "format_version": 4,
  "created_at": {{ "secs_since_epoch": {}, "nanos_since_epoch": 0 }},
  "entries": [],
  "stats": {{
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "entry_count": 0
  }}
}}"#,
        now_secs
    );

    let path = dir.join("empty_snapshot.json");
    std::fs::write(&path, snapshot_json).unwrap();
    path
}

// ============================================================================
// cache stats tests
// ============================================================================

#[test]
#[serial]
fn test_cache_stats_with_snapshot() {
    let dir = temp_dir("cache_stats_snapshot");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args(["cache", "stats", "--snapshot", snapshot.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Verify expected output content
    assert!(stdout.contains("Cache Entries:"), "Missing cache entries");
    assert!(stdout.contains("Hit Rate:"), "Missing hit rate");
    assert!(
        stdout.contains("Compaction History"),
        "Missing compaction history"
    );
    assert!(stdout.contains("Size-based:"), "Missing size-based count");
    assert!(stdout.contains("Time-based:"), "Missing time-based count");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_stats_json_format_with_snapshot() {
    let dir = temp_dir("cache_stats_json");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args([
            "cache",
            "stats",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Parse JSON to verify structure
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    assert!(json.get("cache_entries").is_some(), "Missing cache_entries");
    assert!(json.get("hit_rate").is_some(), "Missing hit_rate");
    assert!(
        json.get("compaction_counts").is_some(),
        "Missing compaction_counts"
    );
    assert!(
        json.get("compaction_rates").is_some(),
        "Missing compaction_rates"
    );
    assert!(
        json.get("time_series_summary").is_some(),
        "Missing time_series_summary"
    );

    // Verify values
    // Note: cache_entries reflects entries restored (may be 2 if both are non-expired)
    // hits/misses are runtime stats from the fresh cache (0 since no lookups performed)
    // compaction counts are restored from snapshot
    assert_eq!(
        json["cache_entries"], 2,
        "Expected 2 cache entries after restore"
    );
    assert_eq!(json["hits"], 0, "Expected 0 hits for fresh restored cache");
    assert_eq!(
        json["misses"], 0,
        "Expected 0 misses for fresh restored cache"
    );
    assert_eq!(json["compaction_counts"]["size_based"], 5);
    assert_eq!(json["compaction_counts"]["time_based"], 3);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_stats_verbose_mode() {
    let dir = temp_dir("cache_stats_verbose");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args([
            "cache",
            "stats",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--verbose",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Verbose mode should show trigger breakdown
    assert!(
        stdout.contains("Trigger Breakdown"),
        "Missing trigger breakdown in verbose mode"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_stats_missing_snapshot() {
    let output = dashprove_cmd()
        .args(["cache", "stats", "--snapshot", "/nonexistent/snapshot.json"])
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "Should fail with missing snapshot"
    );
}

#[test]
#[serial]
fn test_cache_stats_empty_snapshot() {
    let dir = temp_dir("cache_stats_empty");
    let snapshot = create_empty_snapshot(&dir);

    let output = dashprove_cmd()
        .args(["cache", "stats", "--snapshot", snapshot.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);
    assert!(stdout.contains("Cache Entries: 0"), "Missing zero entries");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// cache time-series tests
// ============================================================================

#[test]
#[serial]
fn test_cache_time_series_with_events() {
    let dir = temp_dir("cache_ts_events");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args([
            "cache",
            "time-series",
            "--snapshot",
            snapshot.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Verify time series output
    assert!(
        stdout.contains("Compaction Time-Series"),
        "Missing time-series header"
    );
    assert!(stdout.contains("Total events:"), "Missing total events");
    assert!(stdout.contains("Events in window"), "Missing window info");
    assert!(stdout.contains("Per minute:"), "Missing rate info");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_time_series_json_format() {
    let dir = temp_dir("cache_ts_json");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args([
            "cache",
            "time-series",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Parse JSON to verify structure
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    assert!(json.get("total_entries").is_some(), "Missing total_entries");
    assert!(
        json.get("entries_in_window").is_some(),
        "Missing entries_in_window"
    );
    assert!(
        json.get("rate_per_minute").is_some(),
        "Missing rate_per_minute"
    );
    assert!(json.get("entries").is_some(), "Missing entries array");

    // Should have 5 events from our test data
    assert_eq!(json["total_entries"], 5, "Expected 5 events in time series");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_time_series_with_limit() {
    let dir = temp_dir("cache_ts_limit");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args([
            "cache",
            "time-series",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--limit",
            "2",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    // entries array should be limited to 2
    let entries = json["entries"].as_array().unwrap();
    assert!(
        entries.len() <= 2,
        "Limit not applied: got {} entries",
        entries.len()
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_time_series_verbose_mode() {
    let dir = temp_dir("cache_ts_verbose");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args([
            "cache",
            "time-series",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--verbose",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Verbose should show trigger distribution
    assert!(
        stdout.contains("Trigger Distribution"),
        "Missing trigger distribution in verbose mode"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_time_series_empty() {
    let dir = temp_dir("cache_ts_empty");
    let snapshot = create_empty_snapshot(&dir);

    let output = dashprove_cmd()
        .args([
            "cache",
            "time-series",
            "--snapshot",
            snapshot.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);
    assert!(
        stdout.contains("No compaction events") || stdout.contains("Total events: 0"),
        "Should indicate empty time series"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_time_series_custom_window() {
    let dir = temp_dir("cache_ts_window");
    let snapshot = create_test_snapshot(&dir, true);

    let output = dashprove_cmd()
        .args([
            "cache",
            "time-series",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--window",
            "600", // 10 minute window
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");
    assert_eq!(json["window_seconds"], 600, "Window should be 600 seconds");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// cache clear tests
// ============================================================================

#[test]
#[serial]
fn test_cache_clear_dry_run() {
    let dir = temp_dir("cache_clear_dry");

    // Create a fake cache directory
    let cache_dir = dir.join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    std::fs::write(cache_dir.join("test.json"), "{}").unwrap();

    let output = dashprove_cmd()
        .args([
            "cache",
            "clear",
            "--cache-dir",
            cache_dir.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);
    assert!(stdout.contains("DRY RUN"), "Should indicate dry run");

    // Directory should still exist
    assert!(
        cache_dir.exists(),
        "Directory should not be deleted in dry run"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_clear_actual() {
    let dir = temp_dir("cache_clear_actual");

    // Create a fake cache directory
    let cache_dir = dir.join("cache_to_clear");
    std::fs::create_dir_all(&cache_dir).unwrap();
    std::fs::write(cache_dir.join("test.json"), "{}").unwrap();

    let output = dashprove_cmd()
        .args(["cache", "clear", "--cache-dir", cache_dir.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);
    assert!(
        stdout.contains("cleared successfully"),
        "Should confirm clearing"
    );

    // Directory should be deleted
    assert!(
        !cache_dir.exists(),
        "Directory should be deleted after clear"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_clear_nonexistent() {
    let output = dashprove_cmd()
        .args(["cache", "clear", "--cache-dir", "/nonexistent/cache/path"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "Should succeed for nonexistent dir"
    );
    assert!(
        stdout.contains("does not exist"),
        "Should indicate directory doesn't exist"
    );
}

#[test]
#[serial]
fn test_cache_clear_json_format_dry_run() {
    let dir = temp_dir("cache_clear_json");
    let cache_dir = dir.join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let output = dashprove_cmd()
        .args([
            "cache",
            "clear",
            "--cache-dir",
            cache_dir.to_str().unwrap(),
            "--format",
            "json",
            "--dry-run",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");
    assert_eq!(json["dry_run"], true, "Should indicate dry_run");
    assert_eq!(json["exists"], true, "Should indicate exists");
    assert_eq!(json["cleared"], false, "Should not be cleared in dry run");

    std::fs::remove_dir_all(&dir).ok();
}

// ============================================================================
// Help and basic command tests
// ============================================================================

#[test]
fn test_cache_help() {
    let output = dashprove_cmd()
        .args(["cache", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Help should succeed");
    assert!(stdout.contains("stats"), "Should mention stats subcommand");
    assert!(
        stdout.contains("time-series"),
        "Should mention time-series subcommand"
    );
    assert!(stdout.contains("clear"), "Should mention clear subcommand");
}

#[test]
fn test_cache_stats_help() {
    let output = dashprove_cmd()
        .args(["cache", "stats", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Help should succeed");
    assert!(stdout.contains("--snapshot"), "Should document --snapshot");
    assert!(stdout.contains("--format"), "Should document --format");
}

#[test]
fn test_cache_time_series_help() {
    let output = dashprove_cmd()
        .args(["cache", "time-series", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Help should succeed");
    assert!(stdout.contains("--limit"), "Should document --limit");
    assert!(stdout.contains("--window"), "Should document --window");
}

#[test]
fn test_cache_clear_help() {
    let output = dashprove_cmd()
        .args(["cache", "clear", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Help should succeed");
    assert!(stdout.contains("--dry-run"), "Should document --dry-run");
    assert!(
        stdout.contains("--cache-dir"),
        "Should document --cache-dir"
    );
}

// ============================================================================
// cache autosave tests
// ============================================================================

/// Create a test snapshot JSON with autosave metrics (format v5)
fn create_test_snapshot_with_autosave_metrics(dir: &std::path::Path) -> std::path::PathBuf {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let snapshot_json = format!(
        r#"{{
  "format_version": 5,
  "created_at": {{ "secs_since_epoch": {}, "nanos_since_epoch": 0 }},
  "entries": [
    {{
      "version_hash": "abc123",
      "property_name": "test_property",
      "result": {{
        "property": {{ "name": "test_property", "passed": true, "status": "verified" }},
        "backends": ["mock"],
        "cached_at": {{ "secs_since_epoch": {}, "nanos_since_epoch": 0 }},
        "dependency_hash": "dep123",
        "confidence": 0.95
      }}
    }}
  ],
  "stats": {{
    "hits": 100,
    "misses": 20,
    "evictions": 5,
    "entry_count": 1
  }},
  "autosave_metrics": {{
    "save_count": 15,
    "error_count": 1,
    "skip_count": 5,
    "forced_save_count": 2,
    "save_reason_counts": {{
      "initial": 1,
      "interval": 10,
      "stale_data": 2,
      "coalesced": 2
    }},
    "last_save_reason": "Interval",
    "total_bytes_written": 1048576,
    "last_save_bytes": 65536,
    "interval_ms": 5000,
    "compressed": true,
    "session_start_secs": {},
    "session_end_secs": {}
  }}
}}"#,
        now_secs,
        now_secs - 60,
        now_secs - 3600, // session started 1 hour ago
        now_secs         // session ended now
    );

    let path = dir.join("snapshot_with_autosave.json");
    std::fs::write(&path, snapshot_json).unwrap();
    path
}

/// Create a test snapshot JSON without autosave metrics (format v4)
fn create_test_snapshot_without_autosave_metrics(dir: &std::path::Path) -> std::path::PathBuf {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let snapshot_json = format!(
        r#"{{
  "format_version": 4,
  "created_at": {{ "secs_since_epoch": {}, "nanos_since_epoch": 0 }},
  "entries": [],
  "stats": {{
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "entry_count": 0
  }}
}}"#,
        now_secs
    );

    let path = dir.join("snapshot_no_autosave.json");
    std::fs::write(&path, snapshot_json).unwrap();
    path
}

#[test]
#[serial]
fn test_cache_autosave_with_metrics() {
    let dir = temp_dir("cache_autosave_metrics");
    let snapshot = create_test_snapshot_with_autosave_metrics(&dir);

    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Verify expected output content
    assert!(
        stdout.contains("Autosave Session Metrics"),
        "Missing header"
    );
    assert!(
        stdout.contains("Save Operations:"),
        "Missing save operations"
    );
    assert!(
        stdout.contains("Successful saves: 15"),
        "Missing save count"
    );
    assert!(stdout.contains("Errors: 1"), "Missing error count");
    assert!(stdout.contains("Skipped saves: 5"), "Missing skip count");
    assert!(stdout.contains("Forced saves: 2"), "Missing forced count");
    assert!(
        stdout.contains("Data Written:"),
        "Missing data written section"
    );
    assert!(stdout.contains("Total bytes:"), "Missing total bytes");
    assert!(stdout.contains("1.00 MB"), "Missing formatted total bytes");
    assert!(stdout.contains("Session Info:"), "Missing session info");
    assert!(
        stdout.contains("Compression: enabled"),
        "Missing compression"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_autosave_json_format() {
    let dir = temp_dir("cache_autosave_json");
    let snapshot = create_test_snapshot_with_autosave_metrics(&dir);

    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Parse JSON to verify structure
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");

    assert_eq!(json["has_metrics"], true, "Should have metrics");
    assert!(
        json.get("metrics").is_some(),
        "Missing metrics object in JSON"
    );

    let metrics = &json["metrics"];
    assert_eq!(metrics["save_count"], 15, "Wrong save_count");
    assert_eq!(metrics["error_count"], 1, "Wrong error_count");
    assert_eq!(metrics["skip_count"], 5, "Wrong skip_count");
    assert_eq!(metrics["forced_save_count"], 2, "Wrong forced_save_count");
    assert_eq!(metrics["total_bytes_written"], 1048576, "Wrong total_bytes");
    assert_eq!(metrics["last_save_bytes"], 65536, "Wrong last_save_bytes");
    assert_eq!(metrics["interval_ms"], 5000, "Wrong interval");
    assert_eq!(metrics["compressed"], true, "Wrong compression flag");

    // Verify computed fields
    assert!(
        metrics.get("avg_bytes_per_save").is_some(),
        "Missing avg_bytes_per_save"
    );
    assert!(
        metrics.get("session_duration_secs").is_some(),
        "Missing session_duration_secs"
    );
    assert!(
        metrics.get("success_rate").is_some(),
        "Missing success_rate"
    );

    // Verify save reasons
    let reasons = &metrics["save_reasons"];
    assert_eq!(reasons["initial"], 1, "Wrong initial count");
    assert_eq!(reasons["interval"], 10, "Wrong interval count");
    assert_eq!(reasons["stale_data"], 2, "Wrong stale_data count");
    assert_eq!(reasons["coalesced"], 2, "Wrong coalesced count");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_autosave_verbose_mode() {
    let dir = temp_dir("cache_autosave_verbose");
    let snapshot = create_test_snapshot_with_autosave_metrics(&dir);

    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--verbose",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Verbose mode should show save reason breakdown
    assert!(
        stdout.contains("Save Reason Breakdown"),
        "Missing save reason breakdown in verbose mode"
    );
    assert!(stdout.contains("Initial:"), "Missing initial reason");
    assert!(stdout.contains("Interval:"), "Missing interval reason");
    assert!(stdout.contains("Stale data:"), "Missing stale_data reason");
    assert!(stdout.contains("Coalesced:"), "Missing coalesced reason");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_autosave_no_metrics_in_snapshot() {
    let dir = temp_dir("cache_autosave_no_metrics");
    let snapshot = create_test_snapshot_without_autosave_metrics(&dir);

    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    // Should indicate no metrics found
    assert!(
        stdout.contains("No autosave metrics found"),
        "Should indicate missing metrics"
    );
    assert!(
        stdout.contains("format version 5"),
        "Should mention format requirement"
    );
    assert!(
        stdout.contains("Snapshot format version: 4"),
        "Should show actual version"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_autosave_no_metrics_json() {
    let dir = temp_dir("cache_autosave_no_metrics_json");
    let snapshot = create_test_snapshot_without_autosave_metrics(&dir);

    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command failed: {}", stdout);

    let json: serde_json::Value = serde_json::from_str(&stdout).expect("Invalid JSON output");
    assert_eq!(json["has_metrics"], false, "Should not have metrics");
    assert!(
        json["metrics"].is_null(),
        "metrics should be null when missing"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[serial]
fn test_cache_autosave_missing_snapshot() {
    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            "/nonexistent/snapshot.json",
        ])
        .output()
        .expect("Failed to execute command");

    assert!(
        !output.status.success(),
        "Should fail with missing snapshot"
    );
}

#[test]
#[serial]
fn test_cache_autosave_no_snapshot_provided() {
    let output = dashprove_cmd()
        .args(["cache", "autosave"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Command should succeed");

    // Should indicate no metrics and suggest providing snapshot
    assert!(
        stdout.contains("No autosave metrics found"),
        "Should indicate missing metrics"
    );
    assert!(
        stdout.contains("--snapshot"),
        "Should suggest providing snapshot"
    );

    std::fs::remove_dir_all(std::env::temp_dir().join("dashprove_cache_test_autosave_no_snapshot"))
        .ok();
}

#[test]
fn test_cache_autosave_help() {
    let output = dashprove_cmd()
        .args(["cache", "autosave", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Help should succeed");
    assert!(stdout.contains("--snapshot"), "Should document --snapshot");
    assert!(stdout.contains("--format"), "Should document --format");
    assert!(stdout.contains("--verbose"), "Should document --verbose");
}

#[test]
fn test_cache_help_includes_autosave() {
    let output = dashprove_cmd()
        .args(["cache", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "Help should succeed");
    assert!(
        stdout.contains("autosave"),
        "Should mention autosave subcommand"
    );
}

// ============================================================================
// End-to-end autosave integration tests
// ============================================================================

/// End-to-end test that runs the actual autosave loop and verifies
/// that autosave metrics are persisted in the snapshot file.
///
/// This test:
/// 1. Creates an AsyncImprovementVerifier with cache enabled
/// 2. Starts the autosave loop with a short interval
/// 3. Waits for multiple saves to complete
/// 4. Stops the autosave and gets the summary
/// 5. Verifies the snapshot file exists and contains autosave metrics
/// 6. Uses the CLI to validate the metrics from the snapshot
///
/// Note: The metrics in the snapshot reflect the state AT THE TIME of each save,
/// meaning save_count in the snapshot shows (N-1) after the Nth save callback fires,
/// because the counter is incremented AFTER the file is written.
#[test]
#[serial]
fn test_e2e_autosave_persists_metrics_to_snapshot() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let dir = temp_dir("e2e_autosave_metrics");
    let snapshot_path = dir.join("autosave_test.json");

    // Track saves via callback
    let save_count = Arc::new(AtomicUsize::new(0));
    let save_count_cb = Arc::clone(&save_count);

    // We need to use tokio runtime for async operations
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let final_summary_save_count = rt.block_on(async {
        // Create verifier with cache enabled
        let verifier = dashprove_selfimp::AsyncImprovementVerifier::with_cache();

        // Start autosave with short interval and callbacks to track progress
        let callbacks = dashprove_selfimp::CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_cb.fetch_add(1, Ordering::Relaxed);
        });

        let handle = verifier
            .start_cache_autosave_with_callbacks(
                snapshot_path.clone(),
                std::time::Duration::from_millis(50), // 50ms interval for fast test
                callbacks,
            )
            .expect("Failed to start autosave")
            .expect("Cache should be enabled");

        // Wait for at least 2 saves to complete (so we have at least 1 with save_count > 0)
        let start = std::time::Instant::now();
        while save_count.load(Ordering::Relaxed) < 2 && start.elapsed().as_secs() < 5 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        // Verify at least 2 saves occurred
        let saves = save_count.load(Ordering::Relaxed);
        assert!(saves >= 2, "Expected at least 2 saves, got {}", saves);

        // Stop autosave and get summary
        let summary = handle.stop().await;

        // Verify summary has metrics
        assert!(
            summary.save_count >= 2,
            "Summary should show at least 2 saves"
        );
        summary.save_count
    });

    // Verify snapshot file was created
    assert!(snapshot_path.exists(), "Snapshot file should exist");

    // Read and parse the snapshot to verify autosave metrics
    let snapshot_contents =
        std::fs::read_to_string(&snapshot_path).expect("Failed to read snapshot");
    let snapshot: serde_json::Value =
        serde_json::from_str(&snapshot_contents).expect("Failed to parse snapshot JSON");

    // Verify format version is 5 (which includes autosave_metrics)
    assert_eq!(
        snapshot["format_version"], 5,
        "Snapshot should be format version 5"
    );

    // Verify autosave_metrics exists and has expected fields
    assert!(
        snapshot.get("autosave_metrics").is_some(),
        "Snapshot should contain autosave_metrics"
    );
    let metrics = &snapshot["autosave_metrics"];
    assert!(
        metrics.get("save_count").is_some(),
        "Metrics should have save_count"
    );
    assert!(
        metrics.get("session_start_secs").is_some(),
        "Metrics should have session_start_secs"
    );
    assert!(
        metrics.get("session_end_secs").is_some(),
        "Metrics should have session_end_secs"
    );
    assert!(
        metrics.get("save_reason_counts").is_some(),
        "Metrics should have save_reason_counts"
    );

    // The snapshot metrics show state BEFORE the current save completes.
    // So if N saves occurred, the last snapshot shows save_count = N-1.
    // We need at least 1 in the snapshot (which means 2+ saves occurred).
    let save_count_val = metrics["save_count"].as_u64().unwrap_or(0);
    assert!(
        save_count_val >= 1,
        "Metrics save_count should be >= 1 (got {}). Summary showed {} saves.",
        save_count_val,
        final_summary_save_count
    );

    // Now verify the CLI can read and display the metrics
    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "CLI command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse CLI JSON output
    let cli_output: serde_json::Value =
        serde_json::from_str(&stdout).expect("CLI output should be valid JSON");

    // Verify CLI shows metrics
    assert_eq!(
        cli_output["has_metrics"], true,
        "CLI should report has_metrics=true"
    );
    assert!(
        cli_output.get("metrics").is_some(),
        "CLI output should have metrics"
    );

    // Verify CLI metrics match snapshot metrics
    let cli_metrics = &cli_output["metrics"];
    assert_eq!(
        cli_metrics["save_count"], metrics["save_count"],
        "CLI save_count should match snapshot"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// End-to-end test that verifies autosave metrics survive across multiple
/// save cycles and accurately reflect cumulative statistics.
///
/// Note: Snapshot metrics reflect state BEFORE each save completes (N-1),
/// so we wait for 4+ saves to ensure snapshot has >= 3.
#[test]
#[serial]
fn test_e2e_autosave_cumulative_metrics() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let dir = temp_dir("e2e_autosave_cumulative");
    let snapshot_path = dir.join("cumulative_test.json");

    let save_count = Arc::new(AtomicUsize::new(0));
    let save_count_cb = Arc::clone(&save_count);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let verifier = dashprove_selfimp::AsyncImprovementVerifier::with_cache();

        let callbacks = dashprove_selfimp::CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_cb.fetch_add(1, Ordering::Relaxed);
        });

        let handle = verifier
            .start_cache_autosave_with_callbacks(
                snapshot_path.clone(),
                std::time::Duration::from_millis(30), // 30ms for multiple saves
                callbacks,
            )
            .expect("Failed to start autosave")
            .expect("Cache should be enabled");

        // Wait for at least 4 saves (so snapshot shows >= 3)
        let start = std::time::Instant::now();
        while save_count.load(Ordering::Relaxed) < 4 && start.elapsed().as_secs() < 10 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let final_count = save_count.load(Ordering::Relaxed);
        assert!(
            final_count >= 4,
            "Expected at least 4 saves, got {}",
            final_count
        );

        // Check status before stopping
        let status = handle.status();
        assert!(
            status.save_count >= 4,
            "Status should show >= 4 saves, got {}",
            status.save_count
        );

        let summary = handle.stop().await;
        assert!(
            summary.save_count >= 4,
            "Summary should show >= 4 saves, got {}",
            summary.save_count
        );
    });

    // Verify snapshot metrics reflect multiple saves
    // Note: snapshot shows N-1 where N is number of completed saves
    let snapshot_contents =
        std::fs::read_to_string(&snapshot_path).expect("Failed to read snapshot");
    let snapshot: serde_json::Value =
        serde_json::from_str(&snapshot_contents).expect("Failed to parse snapshot JSON");

    let metrics = &snapshot["autosave_metrics"];
    let save_count_val = metrics["save_count"].as_u64().unwrap_or(0);
    assert!(
        save_count_val >= 3,
        "Snapshot metrics save_count should be >= 3, got {}",
        save_count_val
    );

    // Verify total_bytes_written is non-zero (indicates data was actually written)
    let total_bytes = metrics["total_bytes_written"].as_u64().unwrap_or(0);
    assert!(
        total_bytes > 0,
        "total_bytes_written should be > 0, got {}",
        total_bytes
    );

    // Verify save_reason_counts has at least initial + intervals
    // Note: these counts also reflect N-1 state
    let reasons = &metrics["save_reason_counts"];
    let initial = reasons["initial"].as_u64().unwrap_or(0);
    let interval = reasons["interval"].as_u64().unwrap_or(0);
    assert_eq!(initial, 1, "Should have exactly 1 initial save");
    assert!(
        interval >= 2,
        "Should have >= 2 interval saves, got {}",
        interval
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Test that verifies session duration is calculated correctly
#[test]
#[serial]
fn test_e2e_autosave_session_duration() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let dir = temp_dir("e2e_autosave_duration");
    let snapshot_path = dir.join("duration_test.json");

    let save_count = Arc::new(AtomicUsize::new(0));
    let save_count_cb = Arc::clone(&save_count);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let verifier = dashprove_selfimp::AsyncImprovementVerifier::with_cache();

        let callbacks = dashprove_selfimp::CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_cb.fetch_add(1, Ordering::Relaxed);
        });

        let handle = verifier
            .start_cache_autosave_with_callbacks(
                snapshot_path.clone(),
                std::time::Duration::from_millis(100),
                callbacks,
            )
            .expect("Failed to start autosave")
            .expect("Cache should be enabled");

        // Wait for at least 2 saves with some time between them
        let start = std::time::Instant::now();
        while save_count.load(Ordering::Relaxed) < 2 && start.elapsed().as_secs() < 5 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Let it run a bit more to accumulate session time
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        handle.stop().await;
    });

    // Verify session duration in snapshot
    let snapshot_contents =
        std::fs::read_to_string(&snapshot_path).expect("Failed to read snapshot");
    let snapshot: serde_json::Value =
        serde_json::from_str(&snapshot_contents).expect("Failed to parse snapshot JSON");

    let metrics = &snapshot["autosave_metrics"];

    let session_start = metrics["session_start_secs"].as_u64().unwrap_or(0);
    let session_end = metrics["session_end_secs"].as_u64().unwrap_or(0);

    assert!(session_start > 0, "session_start_secs should be set");
    assert!(session_end > 0, "session_end_secs should be set");
    assert!(
        session_end >= session_start,
        "session_end should be >= session_start"
    );

    // Use CLI to check computed session duration
    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "CLI command failed");

    let cli_output: serde_json::Value =
        serde_json::from_str(&stdout).expect("CLI output should be valid JSON");

    // Verify CLI provides session_duration_secs field
    assert!(
        cli_output["metrics"].get("session_duration_secs").is_some(),
        "CLI should provide session_duration_secs"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// End-to-end test that verifies autosave works correctly with gzip compression.
///
/// This test:
/// 1. Creates an AsyncImprovementVerifier with cache enabled
/// 2. Starts compressed autosave with Fast compression level
/// 3. Waits for multiple saves to complete
/// 4. Verifies the snapshot file is actually gzip compressed (magic bytes)
/// 5. Verifies the compressed snapshot can be loaded via CLI and contains autosave metrics
/// 6. Verifies the summary reports correct compression level
#[test]
#[serial]
fn test_e2e_autosave_compressed_persists_gzip_snapshot() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let dir = temp_dir("e2e_autosave_compressed");
    let snapshot_path = dir.join("compressed_autosave.json.gz");

    let save_count = Arc::new(AtomicUsize::new(0));
    let save_count_cb = Arc::clone(&save_count);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let summary = rt.block_on(async {
        // Create verifier with cache enabled
        let verifier = dashprove_selfimp::AsyncImprovementVerifier::with_cache();

        // Start compressed autosave with callbacks to track progress
        let callbacks = dashprove_selfimp::CacheAutosaveCallbacks::new().on_save(move |_event| {
            save_count_cb.fetch_add(1, Ordering::Relaxed);
        });

        let handle = verifier
            .start_cache_autosave_compressed_with_callbacks(
                snapshot_path.clone(),
                std::time::Duration::from_millis(50),
                dashprove_selfimp::SnapshotCompressionLevel::Fast,
                callbacks,
            )
            .expect("Failed to start compressed autosave")
            .expect("Cache should be enabled");

        // Wait for at least 2 saves
        let start = std::time::Instant::now();
        while save_count.load(Ordering::Relaxed) < 2 && start.elapsed().as_secs() < 5 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let saves = save_count.load(Ordering::Relaxed);
        assert!(saves >= 2, "Expected at least 2 saves, got {}", saves);

        handle.stop().await
    });

    // Verify summary reports compression was used
    assert_eq!(
        summary.compression,
        Some(dashprove_selfimp::SnapshotCompressionLevel::Fast),
        "Summary should report Fast compression"
    );

    // Verify snapshot file exists
    assert!(snapshot_path.exists(), "Compressed snapshot should exist");

    // Verify file is actually gzip compressed (check magic bytes: 0x1f 0x8b)
    let raw_bytes = std::fs::read(&snapshot_path).expect("Failed to read snapshot file");
    assert!(
        raw_bytes.len() >= 2,
        "Snapshot should have at least 2 bytes"
    );
    assert_eq!(raw_bytes[0], 0x1f, "First byte should be gzip magic (0x1f)");
    assert_eq!(
        raw_bytes[1], 0x8b,
        "Second byte should be gzip magic (0x8b)"
    );

    // Verify CacheSnapshot::is_compressed recognizes the format
    assert!(
        dashprove_selfimp::CacheSnapshot::is_compressed(&raw_bytes),
        "CacheSnapshot::is_compressed should recognize gzip format"
    );

    // Decompress and verify content
    let snapshot = dashprove_selfimp::CacheSnapshot::from_compressed(&raw_bytes)
        .expect("Should decompress snapshot successfully");

    // Verify format version 5 (with autosave_metrics)
    assert_eq!(
        snapshot.format_version, 5,
        "Compressed snapshot should be format version 5"
    );

    // Verify autosave_metrics is present
    assert!(
        snapshot.autosave_metrics.is_some(),
        "Compressed snapshot should contain autosave_metrics"
    );

    let metrics = snapshot.autosave_metrics.as_ref().unwrap();
    assert!(
        metrics.save_count >= 1,
        "Metrics should show at least 1 save (got {})",
        metrics.save_count
    );

    // Verify CLI can also read the compressed snapshot via cache autosave command
    // The CLI's auto-detection should handle gzip transparently
    let output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            snapshot_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute CLI command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "CLI should succeed reading compressed snapshot: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let cli_output: serde_json::Value =
        serde_json::from_str(&stdout).expect("CLI output should be valid JSON");

    assert_eq!(
        cli_output["has_metrics"], true,
        "CLI should report has_metrics=true for compressed snapshot"
    );
    assert!(
        cli_output.get("metrics").is_some(),
        "CLI output should contain metrics from compressed snapshot"
    );

    // Verify metrics values match what we read directly
    let cli_save_count = cli_output["metrics"]["save_count"].as_u64().unwrap_or(0);
    assert_eq!(
        cli_save_count, metrics.save_count as u64,
        "CLI metrics should match directly read metrics"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Test that compressed autosave files can be loaded via the CLI's auto-detection.
///
/// This verifies that CLI commands transparently handle both compressed and
/// uncompressed snapshot formats.
#[test]
#[serial]
fn test_e2e_autosave_compressed_cli_auto_detection() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let dir = temp_dir("e2e_autosave_cli_detection");
    let compressed_path = dir.join("snapshot.json.gz");

    let save_count = Arc::new(AtomicUsize::new(0));
    let save_count_cb = Arc::clone(&save_count);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let summary = rt.block_on(async {
        let verifier = dashprove_selfimp::AsyncImprovementVerifier::with_cache();

        let callbacks = dashprove_selfimp::CacheAutosaveCallbacks::new().on_save(move |_| {
            save_count_cb.fetch_add(1, Ordering::Relaxed);
        });

        let handle = verifier
            .start_cache_autosave_compressed_with_callbacks(
                compressed_path.clone(),
                std::time::Duration::from_millis(50),
                dashprove_selfimp::SnapshotCompressionLevel::Best,
                callbacks,
            )
            .expect("Failed to start compressed autosave")
            .expect("Cache should be enabled");

        // Wait for at least 1 save
        let start = std::time::Instant::now();
        while save_count.load(Ordering::Relaxed) < 1 && start.elapsed().as_secs() < 5 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        handle.stop().await
    });

    // Verify the summary shows correct compression level
    assert_eq!(
        summary.compression,
        Some(dashprove_selfimp::SnapshotCompressionLevel::Best),
        "Summary should report Best compression"
    );

    // Verify snapshot file exists and is gzip compressed
    assert!(compressed_path.exists(), "Compressed snapshot should exist");
    let raw_bytes = std::fs::read(&compressed_path).expect("Read snapshot");
    assert!(
        dashprove_selfimp::CacheSnapshot::is_compressed(&raw_bytes),
        "File should be recognized as gzip compressed"
    );

    // Test that `cache stats` command handles compressed files via auto-detection
    let stats_output = dashprove_cmd()
        .args([
            "cache",
            "stats",
            "--snapshot",
            compressed_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute cache stats command");

    let stats_stdout = String::from_utf8_lossy(&stats_output.stdout);
    assert!(
        stats_output.status.success(),
        "cache stats should succeed on compressed file: {}",
        String::from_utf8_lossy(&stats_output.stderr)
    );

    let stats_json: serde_json::Value =
        serde_json::from_str(&stats_stdout).expect("stats output should be valid JSON");
    assert!(
        stats_json.get("cache_entries").is_some(),
        "stats should include cache_entries"
    );

    // Test that `cache autosave` command also works with compressed files
    let autosave_output = dashprove_cmd()
        .args([
            "cache",
            "autosave",
            "--snapshot",
            compressed_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute cache autosave command");

    let autosave_stdout = String::from_utf8_lossy(&autosave_output.stdout);
    assert!(
        autosave_output.status.success(),
        "cache autosave should succeed on compressed file: {}",
        String::from_utf8_lossy(&autosave_output.stderr)
    );

    let autosave_json: serde_json::Value =
        serde_json::from_str(&autosave_stdout).expect("autosave output should be valid JSON");
    assert_eq!(
        autosave_json["has_metrics"], true,
        "autosave should report has_metrics=true"
    );

    std::fs::remove_dir_all(&dir).ok();
}
