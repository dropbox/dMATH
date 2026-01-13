//! Disk-based trace storage for large state space exploration.
//!
//! This module provides `TraceFile`, a disk-backed structure for storing
//! state trace information that enables counterexample reconstruction
//! for state spaces that exceed available RAM.
//!
//! # Design
//!
//! Inspired by TLC's TLCTrace, we store only (predecessor_offset, fingerprint)
//! pairs on disk rather than full states. To reconstruct a trace:
//!
//! 1. Walk backward from the error state's offset, collecting fingerprints
//! 2. Replay forward from the initial state, generating successors and
//!    matching by fingerprint until we reach the error state
//!
//! This trades CPU time (regenerating states) for memory efficiency.
//!
//! # File Format
//!
//! ```text
//! Record 0:
//!   predecessor_offset: u64 (8 bytes) - 0 for initial states
//!   fingerprint: u64 (8 bytes)
//! Record 1:
//!   predecessor_offset: u64 - file offset of predecessor record
//!   fingerprint: u64
//! ...
//! ```
//!
//! Each record is exactly 16 bytes. The file offset of a record
//! serves as its unique identifier.

use crate::state::Fingerprint;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique trace file names.
/// Combined with process ID to ensure uniqueness across concurrent tests.
static TRACE_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Sentinel value indicating no predecessor (initial state).
/// Using u64::MAX because 0 is a valid file offset (the first record).
const NO_PREDECESSOR: u64 = u64::MAX;

/// Size of each trace record in bytes.
const RECORD_SIZE: u64 = 16; // 8 bytes predecessor + 8 bytes fingerprint

/// Disk-based trace file for storing (predecessor, fingerprint) pairs.
///
/// This enables trace reconstruction for large state spaces without
/// keeping full states in memory.
///
/// # Thread Safety
///
/// This type is **not** thread-safe. For parallel model checking,
/// each worker should have its own `TraceFile` or use external
/// synchronization.
///
/// # Example
///
/// ```ignore
/// use tla_check::trace_file::TraceFile;
/// use tla_check::state::Fingerprint;
///
/// let mut trace = TraceFile::create("/tmp/trace.st")?;
///
/// // Record initial state
/// let init_loc = trace.write_initial(Fingerprint(12345))?;
///
/// // Record successor state with its predecessor
/// let succ_loc = trace.write_state(init_loc, Fingerprint(67890))?;
///
/// // Reconstruct fingerprint path from successor back to initial
/// let fps = trace.get_fingerprint_path(succ_loc)?;
/// assert_eq!(fps, vec![Fingerprint(12345), Fingerprint(67890)]);
/// ```
pub struct TraceFile {
    /// File path for the trace
    path: PathBuf,
    /// Writer for appending records (buffered)
    writer: BufWriter<File>,
    /// Current write position (next record offset)
    write_pos: u64,
}

impl TraceFile {
    /// Create a new trace file, overwriting if it exists.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the trace file (typically with `.st` extension)
    ///
    /// # Returns
    ///
    /// Returns the new `TraceFile`, or an I/O error if file creation fails.
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        Ok(TraceFile {
            path,
            writer: BufWriter::new(file),
            write_pos: 0,
        })
    }

    /// Create a trace file in a temporary location.
    ///
    /// The file is created in the system's temp directory with a unique name.
    /// Each call generates a distinct filename using a global counter to ensure
    /// concurrent tests/checkers don't share trace files.
    /// Use `path()` to get the actual file path.
    pub fn create_temp() -> io::Result<Self> {
        let temp_dir = std::env::temp_dir();
        let counter = TRACE_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let filename = format!("tla2_trace_{}_{}.st", std::process::id(), counter);
        Self::create(temp_dir.join(filename))
    }

    /// Get the path to this trace file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write an initial state record (no predecessor).
    ///
    /// # Arguments
    ///
    /// * `fingerprint` - Fingerprint of the initial state
    ///
    /// # Returns
    ///
    /// Returns the file offset (location) of this record, which can be
    /// used as the `predecessor_loc` when writing successor states.
    pub fn write_initial(&mut self, fingerprint: Fingerprint) -> io::Result<u64> {
        self.write_record(NO_PREDECESSOR, fingerprint)
    }

    /// Write a state record with its predecessor.
    ///
    /// # Arguments
    ///
    /// * `predecessor_loc` - File offset of the predecessor state's record
    /// * `fingerprint` - Fingerprint of this state
    ///
    /// # Returns
    ///
    /// Returns the file offset (location) of this record.
    pub fn write_state(
        &mut self,
        predecessor_loc: u64,
        fingerprint: Fingerprint,
    ) -> io::Result<u64> {
        self.write_record(predecessor_loc, fingerprint)
    }

    /// Internal: write a single record and return its location.
    fn write_record(&mut self, predecessor_loc: u64, fingerprint: Fingerprint) -> io::Result<u64> {
        let loc = self.write_pos;

        // Write predecessor offset (8 bytes, little-endian)
        self.writer.write_all(&predecessor_loc.to_le_bytes())?;

        // Write fingerprint (8 bytes, little-endian)
        self.writer.write_all(&fingerprint.0.to_le_bytes())?;

        self.write_pos += RECORD_SIZE;

        Ok(loc)
    }

    /// Flush any buffered writes to disk.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Get the number of records written.
    pub fn record_count(&self) -> u64 {
        self.write_pos / RECORD_SIZE
    }

    /// Get the fingerprint path from an initial state to the given state.
    ///
    /// Walks backward from `end_loc` to an initial state, then reverses
    /// to return the path in forward order (initial -> end).
    ///
    /// # Arguments
    ///
    /// * `end_loc` - File offset of the end state's record
    ///
    /// # Returns
    ///
    /// Returns a vector of fingerprints from initial state to end state,
    /// or an I/O error if reading fails.
    pub fn get_fingerprint_path(&mut self, end_loc: u64) -> io::Result<Vec<Fingerprint>> {
        // Flush writes first to ensure we can read them
        self.flush()?;

        // Open a separate reader for backward traversal
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        let mut fps = Vec::new();
        let mut current_loc = end_loc;

        loop {
            // Read record at current_loc
            reader.seek(SeekFrom::Start(current_loc))?;

            let mut buf = [0u8; 16];
            reader.read_exact(&mut buf)?;

            let predecessor_loc = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let fp = Fingerprint(u64::from_le_bytes(buf[8..16].try_into().unwrap()));

            fps.push(fp);

            if predecessor_loc == NO_PREDECESSOR {
                // Reached initial state
                break;
            }

            current_loc = predecessor_loc;
        }

        // Reverse to get initial -> end order
        fps.reverse();

        Ok(fps)
    }

    /// Read a single record at the given location.
    ///
    /// # Arguments
    ///
    /// * `loc` - File offset of the record
    ///
    /// # Returns
    ///
    /// Returns (predecessor_loc, fingerprint), where predecessor_loc is 0
    /// for initial states.
    pub fn read_record(&mut self, loc: u64) -> io::Result<(u64, Fingerprint)> {
        self.flush()?;

        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        reader.seek(SeekFrom::Start(loc))?;

        let mut buf = [0u8; 16];
        reader.read_exact(&mut buf)?;

        let predecessor_loc = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let fp = Fingerprint(u64::from_le_bytes(buf[8..16].try_into().unwrap()));

        Ok((predecessor_loc, fp))
    }
}

impl Drop for TraceFile {
    fn drop(&mut self) {
        // Best-effort flush on drop
        let _ = self.flush();
    }
}

/// Mapping from fingerprint to trace file location.
///
/// This is used during model checking to track where each state's
/// trace record is stored, enabling trace reconstruction.
#[derive(Debug, Default)]
pub struct TraceLocations {
    /// Map from fingerprint to file offset in the trace file
    locations: std::collections::HashMap<Fingerprint, u64>,
}

impl TraceLocations {
    /// Create an empty location map.
    pub fn new() -> Self {
        TraceLocations {
            locations: std::collections::HashMap::new(),
        }
    }

    /// Record the location of a state in the trace file.
    pub fn insert(&mut self, fp: Fingerprint, loc: u64) {
        self.locations.insert(fp, loc);
    }

    /// Get the location of a state in the trace file.
    pub fn get(&self, fp: &Fingerprint) -> Option<u64> {
        self.locations.get(fp).copied()
    }

    /// Check if a fingerprint has a recorded location.
    pub fn contains(&self, fp: &Fingerprint) -> bool {
        self.locations.contains_key(fp)
    }

    /// Number of recorded locations.
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fp(v: u64) -> Fingerprint {
        Fingerprint(v)
    }

    #[test]
    fn test_trace_file_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let trace = TraceFile::create(&path).unwrap();

        assert!(path.exists());
        assert_eq!(trace.record_count(), 0);
    }

    #[test]
    fn test_write_initial_state() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let mut trace = TraceFile::create(&path).unwrap();

        let loc = trace.write_initial(fp(12345)).unwrap();
        assert_eq!(loc, 0); // First record at offset 0
        assert_eq!(trace.record_count(), 1);

        // Write another initial state
        let loc2 = trace.write_initial(fp(67890)).unwrap();
        assert_eq!(loc2, 16); // Second record at offset 16
        assert_eq!(trace.record_count(), 2);
    }

    #[test]
    fn test_write_successor_state() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let mut trace = TraceFile::create(&path).unwrap();

        let init_loc = trace.write_initial(fp(100)).unwrap();
        let succ_loc = trace.write_state(init_loc, fp(200)).unwrap();

        assert_eq!(init_loc, 0);
        assert_eq!(succ_loc, 16);
        assert_eq!(trace.record_count(), 2);
    }

    #[test]
    fn test_read_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let mut trace = TraceFile::create(&path).unwrap();

        let init_loc = trace.write_initial(fp(100)).unwrap();
        let succ_loc = trace.write_state(init_loc, fp(200)).unwrap();

        // Read initial state
        let (pred, fingerprint) = trace.read_record(init_loc).unwrap();
        assert_eq!(pred, NO_PREDECESSOR);
        assert_eq!(fingerprint, fp(100));

        // Read successor state
        let (pred, fingerprint) = trace.read_record(succ_loc).unwrap();
        assert_eq!(pred, init_loc);
        assert_eq!(fingerprint, fp(200));
    }

    #[test]
    fn test_get_fingerprint_path_single() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let mut trace = TraceFile::create(&path).unwrap();

        let init_loc = trace.write_initial(fp(100)).unwrap();

        let fps = trace.get_fingerprint_path(init_loc).unwrap();
        assert_eq!(fps, vec![fp(100)]);
    }

    #[test]
    fn test_get_fingerprint_path_chain() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let mut trace = TraceFile::create(&path).unwrap();

        // Build a chain: 100 -> 200 -> 300 -> 400
        let loc0 = trace.write_initial(fp(100)).unwrap();
        let loc1 = trace.write_state(loc0, fp(200)).unwrap();
        let loc2 = trace.write_state(loc1, fp(300)).unwrap();
        let loc3 = trace.write_state(loc2, fp(400)).unwrap();

        // Get path to end
        let fps = trace.get_fingerprint_path(loc3).unwrap();
        assert_eq!(fps, vec![fp(100), fp(200), fp(300), fp(400)]);

        // Get path to middle
        let fps = trace.get_fingerprint_path(loc2).unwrap();
        assert_eq!(fps, vec![fp(100), fp(200), fp(300)]);
    }

    #[test]
    fn test_get_fingerprint_path_branching() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let mut trace = TraceFile::create(&path).unwrap();

        // Build a tree:
        //       100
        //      /   \
        //    200   300
        //    /
        //  400
        let loc0 = trace.write_initial(fp(100)).unwrap();
        let loc1 = trace.write_state(loc0, fp(200)).unwrap();
        let loc2 = trace.write_state(loc0, fp(300)).unwrap();
        let loc3 = trace.write_state(loc1, fp(400)).unwrap();

        // Path to 400 (through 200)
        let fps = trace.get_fingerprint_path(loc3).unwrap();
        assert_eq!(fps, vec![fp(100), fp(200), fp(400)]);

        // Path to 300 (direct from 100)
        let fps = trace.get_fingerprint_path(loc2).unwrap();
        assert_eq!(fps, vec![fp(100), fp(300)]);
    }

    #[test]
    fn test_trace_locations() {
        let mut locs = TraceLocations::new();

        assert!(locs.is_empty());

        locs.insert(fp(100), 0);
        locs.insert(fp(200), 16);

        assert_eq!(locs.len(), 2);
        assert!(locs.contains(&fp(100)));
        assert!(locs.contains(&fp(200)));
        assert!(!locs.contains(&fp(300)));

        assert_eq!(locs.get(&fp(100)), Some(0));
        assert_eq!(locs.get(&fp(200)), Some(16));
        assert_eq!(locs.get(&fp(300)), None);
    }

    #[test]
    fn test_large_trace() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.st");
        let mut trace = TraceFile::create(&path).unwrap();

        // Build a long chain
        let mut prev_loc = trace.write_initial(fp(0)).unwrap();
        for i in 1..1000u64 {
            prev_loc = trace.write_state(prev_loc, fp(i)).unwrap();
        }

        assert_eq!(trace.record_count(), 1000);

        // Reconstruct full path
        let fps = trace.get_fingerprint_path(prev_loc).unwrap();
        assert_eq!(fps.len(), 1000);
        for (i, fp) in fps.iter().enumerate() {
            assert_eq!(fp.0, i as u64);
        }
    }

    #[test]
    fn test_temp_file() {
        let mut trace = TraceFile::create_temp().unwrap();

        let loc = trace.write_initial(fp(12345)).unwrap();
        assert_eq!(trace.record_count(), 1);

        let fps = trace.get_fingerprint_path(loc).unwrap();
        assert_eq!(fps, vec![fp(12345)]);

        // File should exist
        assert!(trace.path().exists());
    }
}
