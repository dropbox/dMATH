//! Persistence helpers for the learning system

use crate::LearningError;
use serde::{de::DeserializeOwned, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Write JSON to a file atomically by using a temporary file + rename
pub(crate) fn write_json_atomic<P: AsRef<Path>, T: Serialize>(
    path: P,
    value: &T,
) -> Result<(), LearningError> {
    let path = path.as_ref();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("tmp");
    {
        let mut file = File::create(&tmp_path)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.flush()?;
        file.sync_all()?;
    }

    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Read JSON from a file into the requested type
pub(crate) fn read_json<P: AsRef<Path>, T: DeserializeOwned>(path: P) -> Result<T, LearningError> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}
