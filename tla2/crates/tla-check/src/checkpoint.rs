//! Checkpoint/resume support for model checking.
//!
//! This module provides facilities to save and restore model checking state,
//! enabling long-running checks to be interrupted and resumed later.
//!
//! # Design
//!
//! A checkpoint contains:
//! - **Fingerprints**: Set of seen state fingerprints
//! - **Frontier**: Queue of states still to be explored
//! - **Parent map**: Parent fingerprint for each state (for trace reconstruction)
//! - **Depths**: Depth of each state in the BFS tree
//! - **Statistics**: State counts, transitions, etc.
//!
//! The checkpoint is stored as a directory containing:
//! - `checkpoint.json`: Metadata and statistics
//! - `fingerprints.bin`: Binary fingerprint set
//! - `frontier.json`: States to explore (JSON for debuggability)
//! - `parents.bin`: Parent pointer map
//! - `depths.bin`: Depth map
//!
//! # Usage
//!
//! ```ignore
//! // Save a checkpoint
//! let checkpoint = Checkpoint::from_checker(&checker);
//! checkpoint.save("/path/to/checkpoint")?;
//!
//! // Resume from checkpoint
//! let checkpoint = Checkpoint::load("/path/to/checkpoint")?;
//! checker.restore_from_checkpoint(checkpoint)?;
//! ```

use crate::check::CheckStats;
use crate::state::{Fingerprint, State};
use crate::value::{intern_string, RecordValue, SortedSet, Value};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::SystemTime;

/// Magic bytes identifying a TLA2 checkpoint fingerprint file
const FINGERPRINT_MAGIC: &[u8; 8] = b"TLA2FP01";

/// Magic bytes identifying a TLA2 checkpoint parent map file
const PARENTS_MAGIC: &[u8; 8] = b"TLA2PM01";

/// Magic bytes identifying a TLA2 checkpoint depths file
const DEPTHS_MAGIC: &[u8; 8] = b"TLA2DP01";

/// Checkpoint version
const CHECKPOINT_VERSION: u32 = 1;

/// Serializable value representation for JSON export
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum SerializableValue {
    Bool(bool),
    Int(String), // BigInt as string
    String(String),
    Set(Vec<SerializableValue>),
    Seq(Vec<SerializableValue>),
    Record(Vec<(String, SerializableValue)>),
    Tuple(Vec<SerializableValue>),
    ModelValue(String),
    // Lazy values are enumerated for checkpoint
    Interval { lo: String, hi: String },
}

impl SerializableValue {
    /// Convert a Value to its serializable form
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Bool(b) => SerializableValue::Bool(*b),
            Value::SmallInt(n) => SerializableValue::Int(n.to_string()),
            Value::Int(n) => SerializableValue::Int(n.to_string()),
            Value::String(s) => SerializableValue::String(s.to_string()),
            Value::Set(s) => {
                SerializableValue::Set(s.iter().map(SerializableValue::from_value).collect())
            }
            Value::Interval(iv) => SerializableValue::Interval {
                lo: iv.low.to_string(),
                hi: iv.high.to_string(),
            },
            Value::Subset(_)
            | Value::FuncSet(_)
            | Value::RecordSet(_)
            | Value::TupleSet(_)
            | Value::SetCup(_)
            | Value::SetCap(_)
            | Value::SetDiff(_)
            | Value::SetPred(_)
            | Value::KSubset(_)
            | Value::BigUnion(_) => {
                // Lazy set types - enumerate for checkpoint using iter_set()
                // Note: SetPred cannot be enumerated without context, skip those
                if let Some(iter) = value.iter_set() {
                    let elements: Vec<_> =
                        iter.map(|v| SerializableValue::from_value(&v)).collect();
                    SerializableValue::Set(elements)
                } else {
                    // Can't enumerate, serialize as empty set
                    SerializableValue::Set(vec![])
                }
            }
            Value::Func(f) => {
                // Serialize function as set of tuples
                let pairs: Vec<_> = f
                    .mapping_iter()
                    .map(|(k, v)| {
                        SerializableValue::Tuple(vec![
                            SerializableValue::from_value(k),
                            SerializableValue::from_value(v),
                        ])
                    })
                    .collect();
                SerializableValue::Set(pairs)
            }
            Value::IntFunc(f) => {
                // Serialize IntFunc as set of tuples (like Func)
                let pairs: Vec<_> = (0..f.values.len())
                    .map(|i| {
                        let k = Value::SmallInt(f.min + i as i64);
                        SerializableValue::Tuple(vec![
                            SerializableValue::from_value(&k),
                            SerializableValue::from_value(&f.values[i]),
                        ])
                    })
                    .collect();
                SerializableValue::Set(pairs)
            }
            Value::LazyFunc(_) => {
                // Lazy functions cannot be fully serialized
                SerializableValue::Set(vec![])
            }
            Value::Seq(s) => {
                SerializableValue::Seq(s.iter().map(SerializableValue::from_value).collect())
            }
            Value::Record(r) => SerializableValue::Record(
                r.iter()
                    .map(|(k, v)| (k.to_string(), SerializableValue::from_value(v)))
                    .collect(),
            ),
            Value::Tuple(t) => {
                SerializableValue::Tuple(t.iter().map(SerializableValue::from_value).collect())
            }
            Value::ModelValue(m) => SerializableValue::ModelValue(m.to_string()),
            Value::Closure(_) => {
                // Closures cannot be serialized
                SerializableValue::Set(vec![])
            }
            Value::StringSet | Value::AnySet | Value::SeqSet(_) => {
                // Infinite sets - serialize as empty (cannot be serialized)
                SerializableValue::Set(vec![])
            }
        }
    }

    /// Convert back to a Value
    pub fn to_value(&self) -> Value {
        match self {
            SerializableValue::Bool(b) => Value::Bool(*b),
            SerializableValue::Int(s) => {
                // Use big_int to normalize to SmallInt when value fits
                Value::big_int(s.parse::<BigInt>().unwrap_or_else(|_| BigInt::from(0)))
            }
            SerializableValue::String(s) => Value::String(intern_string(s.as_str())),
            SerializableValue::Set(elems) => {
                let set: SortedSet = elems.iter().map(|e| e.to_value()).collect();
                Value::Set(set)
            }
            SerializableValue::Seq(elems) => {
                Value::Seq(elems.iter().map(|e| e.to_value()).collect())
            }
            SerializableValue::Record(fields) => {
                let record: RecordValue = fields
                    .iter()
                    .map(|(k, v)| (intern_string(k.as_str()), v.to_value()))
                    .collect();
                Value::Record(record)
            }
            SerializableValue::Tuple(elems) => {
                Value::Tuple(elems.iter().map(|e| e.to_value()).collect())
            }
            SerializableValue::ModelValue(m) => Value::model_value(m.as_str()),
            SerializableValue::Interval { lo, hi } => {
                let lo = lo.parse::<BigInt>().unwrap_or_else(|_| BigInt::from(0));
                let hi = hi.parse::<BigInt>().unwrap_or_else(|_| BigInt::from(0));
                Value::Interval(crate::value::IntervalValue::new(lo, hi))
            }
        }
    }
}

/// Serializable state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableState {
    pub vars: Vec<(String, SerializableValue)>,
}

impl SerializableState {
    /// Convert a State to serializable form
    pub fn from_state(state: &State) -> Self {
        SerializableState {
            vars: state
                .vars()
                .map(|(k, v)| (k.to_string(), SerializableValue::from_value(v)))
                .collect(),
        }
    }

    /// Convert back to a State
    pub fn to_state(&self) -> State {
        let mut state = State::new();
        for (name, value) in &self.vars {
            state = state.with_var(name.as_str(), value.to_value());
        }
        state
    }
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint format version
    pub version: u32,
    /// When the checkpoint was created
    pub created_at: String,
    /// Spec file path (for verification)
    pub spec_path: Option<String>,
    /// Config file path (for verification)
    pub config_path: Option<String>,
    /// Model checking statistics at checkpoint time
    pub stats: CheckpointStats,
}

/// Statistics stored in checkpoint
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointStats {
    pub states_found: usize,
    pub initial_states: usize,
    pub transitions: usize,
    pub max_depth: usize,
    pub frontier_size: usize,
}

impl From<&CheckStats> for CheckpointStats {
    fn from(stats: &CheckStats) -> Self {
        CheckpointStats {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
            max_depth: stats.max_depth,
            frontier_size: 0, // Set separately
        }
    }
}

/// A complete model checking checkpoint
#[derive(Debug)]
pub struct Checkpoint {
    /// Metadata about the checkpoint
    pub metadata: CheckpointMetadata,
    /// Seen state fingerprints
    pub fingerprints: Vec<Fingerprint>,
    /// States in the frontier (to be explored)
    pub frontier: Vec<State>,
    /// Parent pointers: child fingerprint -> parent fingerprint
    pub parents: HashMap<Fingerprint, Fingerprint>,
    /// Depth of each state
    pub depths: HashMap<Fingerprint, usize>,
}

impl Checkpoint {
    /// Create a new empty checkpoint
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Checkpoint {
            metadata: CheckpointMetadata {
                version: CHECKPOINT_VERSION,
                created_at: format!("{}", now),
                spec_path: None,
                config_path: None,
                stats: CheckpointStats::default(),
            },
            fingerprints: Vec::new(),
            frontier: Vec::new(),
            parents: HashMap::new(),
            depths: HashMap::new(),
        }
    }

    /// Set spec and config paths for verification on resume
    pub fn with_paths(mut self, spec_path: Option<&str>, config_path: Option<&str>) -> Self {
        self.metadata.spec_path = spec_path.map(|s| s.to_string());
        self.metadata.config_path = config_path.map(|s| s.to_string());
        self
    }

    /// Save checkpoint to a directory
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        // Save metadata
        let meta_path = dir.join("checkpoint.json");
        let mut meta = self.metadata.clone();
        meta.stats.frontier_size = self.frontier.len();
        let meta_file = File::create(&meta_path)?;
        serde_json::to_writer_pretty(BufWriter::new(meta_file), &meta).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("JSON error: {}", e))
        })?;

        // Save fingerprints (binary)
        self.save_fingerprints(dir.join("fingerprints.bin"))?;

        // Save frontier (JSON for debuggability)
        self.save_frontier(dir.join("frontier.json"))?;

        // Save parent pointers (binary)
        self.save_parents(dir.join("parents.bin"))?;

        // Save depths (binary)
        self.save_depths(dir.join("depths.bin"))?;

        Ok(())
    }

    /// Load checkpoint from a directory
    pub fn load<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let dir = dir.as_ref();

        // Load metadata
        let meta_path = dir.join("checkpoint.json");
        let meta_file = File::open(&meta_path)?;
        let metadata: CheckpointMetadata = serde_json::from_reader(BufReader::new(meta_file))
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("JSON error: {}", e))
            })?;

        if metadata.version != CHECKPOINT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Checkpoint version mismatch: expected {}, got {}",
                    CHECKPOINT_VERSION, metadata.version
                ),
            ));
        }

        // Load components
        let fingerprints = Self::load_fingerprints(dir.join("fingerprints.bin"))?;
        let frontier = Self::load_frontier(dir.join("frontier.json"))?;
        let parents = Self::load_parents(dir.join("parents.bin"))?;
        let depths = Self::load_depths(dir.join("depths.bin"))?;

        Ok(Checkpoint {
            metadata,
            fingerprints,
            frontier,
            parents,
            depths,
        })
    }

    /// Save fingerprints to binary file
    fn save_fingerprints<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(FINGERPRINT_MAGIC)?;
        file.write_all(&(self.fingerprints.len() as u64).to_le_bytes())?;
        for fp in &self.fingerprints {
            file.write_all(&fp.0.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }

    /// Load fingerprints from binary file
    fn load_fingerprints<P: AsRef<Path>>(path: P) -> io::Result<Vec<Fingerprint>> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != FINGERPRINT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid fingerprint file magic",
            ));
        }

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut fingerprints = Vec::with_capacity(len);
        for _ in 0..len {
            let mut fp_bytes = [0u8; 8];
            file.read_exact(&mut fp_bytes)?;
            fingerprints.push(Fingerprint(u64::from_le_bytes(fp_bytes)));
        }

        Ok(fingerprints)
    }

    /// Save frontier states to JSON file
    fn save_frontier<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let serializable: Vec<_> = self
            .frontier
            .iter()
            .map(SerializableState::from_state)
            .collect();
        let file = File::create(path)?;
        serde_json::to_writer(BufWriter::new(file), &serializable).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("JSON error: {}", e))
        })?;
        Ok(())
    }

    /// Load frontier states from JSON file
    fn load_frontier<P: AsRef<Path>>(path: P) -> io::Result<Vec<State>> {
        let file = File::open(path)?;
        let serializable: Vec<SerializableState> = serde_json::from_reader(BufReader::new(file))
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("JSON error: {}", e))
            })?;
        Ok(serializable.iter().map(|s| s.to_state()).collect())
    }

    /// Save parent pointers to binary file
    fn save_parents<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(PARENTS_MAGIC)?;
        file.write_all(&(self.parents.len() as u64).to_le_bytes())?;
        for (child, parent) in &self.parents {
            file.write_all(&child.0.to_le_bytes())?;
            file.write_all(&parent.0.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }

    /// Load parent pointers from binary file
    fn load_parents<P: AsRef<Path>>(path: P) -> io::Result<HashMap<Fingerprint, Fingerprint>> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != PARENTS_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid parents file magic",
            ));
        }

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut parents = HashMap::with_capacity(len);
        for _ in 0..len {
            let mut child_bytes = [0u8; 8];
            let mut parent_bytes = [0u8; 8];
            file.read_exact(&mut child_bytes)?;
            file.read_exact(&mut parent_bytes)?;
            parents.insert(
                Fingerprint(u64::from_le_bytes(child_bytes)),
                Fingerprint(u64::from_le_bytes(parent_bytes)),
            );
        }

        Ok(parents)
    }

    /// Save depths to binary file
    fn save_depths<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(DEPTHS_MAGIC)?;
        file.write_all(&(self.depths.len() as u64).to_le_bytes())?;
        for (fp, depth) in &self.depths {
            file.write_all(&fp.0.to_le_bytes())?;
            file.write_all(&(*depth as u64).to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }

    /// Load depths from binary file
    fn load_depths<P: AsRef<Path>>(path: P) -> io::Result<HashMap<Fingerprint, usize>> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != DEPTHS_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid depths file magic",
            ));
        }

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut depths = HashMap::with_capacity(len);
        for _ in 0..len {
            let mut fp_bytes = [0u8; 8];
            let mut depth_bytes = [0u8; 8];
            file.read_exact(&mut fp_bytes)?;
            file.read_exact(&mut depth_bytes)?;
            depths.insert(
                Fingerprint(u64::from_le_bytes(fp_bytes)),
                u64::from_le_bytes(depth_bytes) as usize,
            );
        }

        Ok(depths)
    }
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn test_serializable_value_roundtrip() {
        // Test basic types
        let bool_val = Value::Bool(true);
        let sv = SerializableValue::from_value(&bool_val);
        assert_eq!(sv.to_value(), bool_val);

        let int_val = Value::Int(BigInt::from(42));
        let sv = SerializableValue::from_value(&int_val);
        assert_eq!(sv.to_value(), int_val);

        let str_val = Value::String(Arc::from("hello"));
        let sv = SerializableValue::from_value(&str_val);
        assert_eq!(sv.to_value(), str_val);

        // Test collections
        let set_val = Value::Set(
            vec![
                Value::Int(BigInt::from(1)),
                Value::Int(BigInt::from(2)),
                Value::Int(BigInt::from(3)),
            ]
            .into_iter()
            .collect(),
        );
        let sv = SerializableValue::from_value(&set_val);
        assert_eq!(sv.to_value(), set_val);

        let seq_val =
            Value::Seq(vec![Value::Int(BigInt::from(1)), Value::Int(BigInt::from(2))].into());
        let sv = SerializableValue::from_value(&seq_val);
        assert_eq!(sv.to_value(), seq_val);
    }

    #[test]
    fn test_serializable_state_roundtrip() {
        let mut state = State::new();
        state = state.with_var("x", Value::Int(BigInt::from(42)));
        state = state.with_var("y", Value::Bool(true));
        state = state.with_var(
            "s",
            Value::Set(
                vec![
                    Value::Int(BigInt::from(1)),
                    Value::Int(BigInt::from(2)),
                    Value::Int(BigInt::from(3)),
                ]
                .into_iter()
                .collect(),
            ),
        );

        let ss = SerializableState::from_state(&state);
        let restored = ss.to_state();

        assert_eq!(restored.get("x").unwrap(), &Value::Int(BigInt::from(42)));
        assert_eq!(restored.get("y").unwrap(), &Value::Bool(true));
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path().join("checkpoint");

        // Create a checkpoint
        let mut checkpoint = Checkpoint::new();
        checkpoint.fingerprints = vec![Fingerprint(100), Fingerprint(200), Fingerprint(300)];

        let mut state = State::new();
        state = state.with_var("x", Value::Int(BigInt::from(42)));
        checkpoint.frontier = vec![state];

        checkpoint
            .parents
            .insert(Fingerprint(200), Fingerprint(100));
        checkpoint
            .parents
            .insert(Fingerprint(300), Fingerprint(200));

        checkpoint.depths.insert(Fingerprint(100), 0);
        checkpoint.depths.insert(Fingerprint(200), 1);
        checkpoint.depths.insert(Fingerprint(300), 2);

        // Save
        checkpoint.save(&checkpoint_dir).unwrap();

        // Load
        let loaded = Checkpoint::load(&checkpoint_dir).unwrap();

        assert_eq!(loaded.fingerprints.len(), 3);
        assert_eq!(loaded.frontier.len(), 1);
        assert_eq!(loaded.parents.len(), 2);
        assert_eq!(loaded.depths.len(), 3);

        assert_eq!(
            loaded.parents.get(&Fingerprint(200)),
            Some(&Fingerprint(100))
        );
        assert_eq!(loaded.depths.get(&Fingerprint(300)), Some(&2));
    }

    #[test]
    fn test_fingerprints_binary_format() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fingerprints.bin");

        let checkpoint = Checkpoint {
            metadata: CheckpointMetadata {
                version: CHECKPOINT_VERSION,
                created_at: "0".to_string(),
                spec_path: None,
                config_path: None,
                stats: CheckpointStats::default(),
            },
            fingerprints: vec![
                Fingerprint(0x1234567890ABCDEF),
                Fingerprint(0xFEDCBA0987654321),
            ],
            frontier: vec![],
            parents: HashMap::new(),
            depths: HashMap::new(),
        };

        checkpoint.save_fingerprints(&path).unwrap();
        let loaded = Checkpoint::load_fingerprints(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, 0x1234567890ABCDEF);
        assert_eq!(loaded[1].0, 0xFEDCBA0987654321);
    }
}
