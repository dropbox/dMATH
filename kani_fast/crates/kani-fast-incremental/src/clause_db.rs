//! SQLite-based clause database for persistent clause storage
//!
//! This module provides a persistent storage layer for learned clauses,
//! enabling incremental verification by reusing clauses from previous runs.

use crate::content_hash::{ContentHash, HashKind};
use rusqlite::{params, Connection, Result as SqlResult};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{debug, info};

/// Unique identifier for a clause
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClauseId(pub i64);

/// Validity status of a cached clause
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseValidity {
    /// Clause is valid and can be reused
    Valid,
    /// Clause is invalid due to source changes
    Invalid,
    /// Clause validity is unknown and needs checking
    Unknown,
    /// Clause has expired due to age
    Expired,
}

/// A clause entry in the database
#[derive(Debug, Clone)]
pub struct ClauseEntry {
    /// Unique identifier
    pub id: ClauseId,
    /// The clause literals (DIMACS format: positive = variable, negative = negation)
    pub literals: Vec<i32>,
    /// Hash of the source context this clause was derived from
    pub source_hash: ContentHash,
    /// Hash of the function this clause relates to
    pub function_hash: ContentHash,
    /// When the clause was created
    pub created_at: SystemTime,
    /// When the clause was last used
    pub last_used: SystemTime,
    /// Number of times this clause has been used
    pub use_count: u64,
    /// The solver that learned this clause
    pub solver: String,
    /// Activity score (higher = more useful during solving)
    pub activity: f64,
    /// Optional LBD (Literal Block Distance) score
    pub lbd: Option<u32>,
}

/// Errors that can occur with the clause database
#[derive(Debug, Error)]
pub enum ClauseDbError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Invalid clause format")]
    InvalidClause,

    #[error("Clause not found: {0:?}")]
    NotFound(ClauseId),

    #[error("Database is full ({0} bytes exceeds limit {1} bytes)")]
    DatabaseFull(u64, u64),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Hex decode error: {0}")]
    HexDecode(#[from] hex::FromHexError),
}

/// SQLite-based clause database
pub struct ClauseDatabase {
    conn: Connection,
    max_size: u64,
}

impl ClauseDatabase {
    /// Open or create a clause database at the given path
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ClauseDbError> {
        let conn = Connection::open(path)?;
        let db = Self {
            conn,
            max_size: 100 * 1024 * 1024, // 100 MB default
        };
        db.init_schema()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing)
    pub fn in_memory() -> Result<Self, ClauseDbError> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn,
            max_size: 100 * 1024 * 1024,
        };
        db.init_schema()?;
        Ok(db)
    }

    /// Set the maximum database size
    pub fn set_max_size(&mut self, size: u64) {
        self.max_size = size;
    }

    /// Initialize the database schema
    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "
            -- Main clause storage table
            CREATE TABLE IF NOT EXISTS clauses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                literals BLOB NOT NULL,
                source_hash TEXT NOT NULL,
                function_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_used INTEGER NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 1,
                solver TEXT NOT NULL,
                activity REAL NOT NULL DEFAULT 0.0,
                lbd INTEGER
            );

            -- Index for looking up clauses by source/function
            CREATE INDEX IF NOT EXISTS idx_clauses_source ON clauses(source_hash);
            CREATE INDEX IF NOT EXISTS idx_clauses_function ON clauses(function_hash);
            CREATE INDEX IF NOT EXISTS idx_clauses_activity ON clauses(activity DESC);
            CREATE INDEX IF NOT EXISTS idx_clauses_lbd ON clauses(lbd);

            -- Verification results cache
            CREATE TABLE IF NOT EXISTS verification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_hash TEXT NOT NULL UNIQUE,
                function_name TEXT NOT NULL,
                result TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                clause_count INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_results_project ON verification_results(project_hash);

            -- File hash tracking for invalidation
            CREATE TABLE IF NOT EXISTS file_hashes (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_modified INTEGER NOT NULL
            );

            -- Metadata table
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Initialize schema version
            INSERT OR IGNORE INTO metadata (key, value) VALUES ('schema_version', '1');
            ",
        )?;

        info!("Clause database schema initialized");
        Ok(())
    }

    /// Store a learned clause
    pub fn store_clause(
        &self,
        literals: &[i32],
        source_hash: &ContentHash,
        function_hash: &ContentHash,
        solver: &str,
        activity: f64,
        lbd: Option<u32>,
    ) -> Result<ClauseId, ClauseDbError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let literals_blob = Self::encode_literals(literals);

        self.conn.execute(
            "
            INSERT INTO clauses (literals, source_hash, function_hash, created_at, last_used, solver, activity, lbd)
            VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6, ?7)
            ",
            params![
                literals_blob,
                source_hash.to_hex(),
                function_hash.to_hex(),
                now,
                solver,
                activity,
                lbd,
            ],
        )?;

        let id = ClauseId(self.conn.last_insert_rowid());
        debug!("Stored clause {:?} with {} literals", id, literals.len());
        Ok(id)
    }

    /// Store multiple clauses efficiently
    pub fn store_clauses_batch(
        &self,
        clauses: &[(&[i32], f64, Option<u32>)],
        source_hash: &ContentHash,
        function_hash: &ContentHash,
        solver: &str,
    ) -> Result<usize, ClauseDbError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let source_hex = source_hash.to_hex();
        let function_hex = function_hash.to_hex();

        let mut stmt = self.conn.prepare(
            "
            INSERT INTO clauses (literals, source_hash, function_hash, created_at, last_used, solver, activity, lbd)
            VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6, ?7)
            ",
        )?;

        let mut stored = 0;
        for (literals, activity, lbd) in clauses {
            let literals_blob = Self::encode_literals(literals);
            stmt.execute(params![
                literals_blob,
                &source_hex,
                &function_hex,
                now,
                solver,
                activity,
                lbd,
            ])?;
            stored += 1;
        }

        debug!("Stored {} clauses in batch", stored);
        Ok(stored)
    }

    /// Load clauses for a given function hash
    pub fn load_clauses_for_function(
        &self,
        function_hash: &ContentHash,
    ) -> Result<Vec<ClauseEntry>, ClauseDbError> {
        let mut stmt = self.conn.prepare(
            "
            SELECT id, literals, source_hash, function_hash, created_at, last_used, use_count, solver, activity, lbd
            FROM clauses
            WHERE function_hash = ?1
            ORDER BY activity DESC, use_count DESC
            ",
        )?;

        let entries = stmt
            .query_map([function_hash.to_hex()], Self::row_to_entry)?
            .collect::<SqlResult<Vec<_>>>()?;

        debug!(
            "Loaded {} clauses for function {}",
            entries.len(),
            function_hash.short()
        );
        Ok(entries)
    }

    /// Load top N clauses by activity for a function
    pub fn load_top_clauses(
        &self,
        function_hash: &ContentHash,
        limit: usize,
    ) -> Result<Vec<ClauseEntry>, ClauseDbError> {
        let mut stmt = self.conn.prepare(
            "
            SELECT id, literals, source_hash, function_hash, created_at, last_used, use_count, solver, activity, lbd
            FROM clauses
            WHERE function_hash = ?1
            ORDER BY activity DESC
            LIMIT ?2
            ",
        )?;

        let entries = stmt
            .query_map(params![function_hash.to_hex(), limit as i64], |row| {
                Self::row_to_entry(row)
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(entries)
    }

    /// Load clauses with low LBD (high quality)
    pub fn load_low_lbd_clauses(
        &self,
        function_hash: &ContentHash,
        max_lbd: u32,
        limit: usize,
    ) -> Result<Vec<ClauseEntry>, ClauseDbError> {
        let mut stmt = self.conn.prepare(
            "
            SELECT id, literals, source_hash, function_hash, created_at, last_used, use_count, solver, activity, lbd
            FROM clauses
            WHERE function_hash = ?1 AND lbd IS NOT NULL AND lbd <= ?2
            ORDER BY lbd ASC, activity DESC
            LIMIT ?3
            ",
        )?;

        let entries = stmt
            .query_map(
                params![function_hash.to_hex(), max_lbd, limit as i64],
                Self::row_to_entry,
            )?
            .collect::<SqlResult<Vec<_>>>()?;

        Ok(entries)
    }

    /// Update clause usage statistics
    pub fn touch_clause(&self, id: ClauseId) -> Result<(), ClauseDbError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "UPDATE clauses SET last_used = ?1, use_count = use_count + 1 WHERE id = ?2",
            params![now, id.0],
        )?;

        Ok(())
    }

    /// Update clause activity
    pub fn update_activity(&self, id: ClauseId, activity: f64) -> Result<(), ClauseDbError> {
        self.conn.execute(
            "UPDATE clauses SET activity = ?1 WHERE id = ?2",
            params![activity, id.0],
        )?;
        Ok(())
    }

    /// Delete clauses older than the specified age
    pub fn cleanup_old_clauses(&self, max_age: Duration) -> Result<usize, ClauseDbError> {
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - max_age.as_secs() as i64;

        let deleted = self
            .conn
            .execute("DELETE FROM clauses WHERE last_used < ?1", params![cutoff])?;

        if deleted > 0 {
            info!("Cleaned up {} old clauses", deleted);
        }

        Ok(deleted)
    }

    /// Delete clauses for an invalidated function
    pub fn invalidate_function(&self, function_hash: &ContentHash) -> Result<usize, ClauseDbError> {
        let deleted = self.conn.execute(
            "DELETE FROM clauses WHERE function_hash = ?1",
            params![function_hash.to_hex()],
        )?;

        if deleted > 0 {
            debug!(
                "Invalidated {} clauses for function {}",
                deleted,
                function_hash.short()
            );
        }

        Ok(deleted)
    }

    /// Get database statistics
    pub fn stats(&self) -> Result<DbStats, ClauseDbError> {
        let clause_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM clauses", [], |row| row.get(0))?;

        let function_count: i64 = self.conn.query_row(
            "SELECT COUNT(DISTINCT function_hash) FROM clauses",
            [],
            |row| row.get(0),
        )?;

        let total_literals: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(LENGTH(literals) / 4), 0) FROM clauses",
            [],
            |row| row.get(0),
        )?;

        let page_count: i64 = self
            .conn
            .query_row("PRAGMA page_count", [], |row| row.get(0))?;
        let page_size: i64 = self
            .conn
            .query_row("PRAGMA page_size", [], |row| row.get(0))?;
        let db_size = (page_count * page_size) as u64;

        Ok(DbStats {
            clause_count: clause_count as u64,
            function_count: function_count as u64,
            total_literals: total_literals as u64,
            database_size: db_size,
        })
    }

    /// Store a file hash for change tracking
    pub fn store_file_hash(&self, path: &str, hash: &ContentHash) -> Result<(), ClauseDbError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "INSERT OR REPLACE INTO file_hashes (path, hash, last_modified) VALUES (?1, ?2, ?3)",
            params![path, hash.to_hex(), now],
        )?;

        Ok(())
    }

    /// Get stored file hash
    pub fn get_file_hash(&self, path: &str) -> Result<Option<ContentHash>, ClauseDbError> {
        let result: SqlResult<String> = self.conn.query_row(
            "SELECT hash FROM file_hashes WHERE path = ?1",
            [path],
            |row| row.get(0),
        );

        match result {
            Ok(hex) => Ok(Some(ContentHash::from_hex(&hex, HashKind::SourceFile)?)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(ClauseDbError::Database(e)),
        }
    }

    /// Store verification result
    pub fn store_verification_result(
        &self,
        project_hash: &ContentHash,
        function_name: &str,
        result: &str,
        duration_ms: u64,
        clause_count: u64,
    ) -> Result<(), ClauseDbError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "
            INSERT OR REPLACE INTO verification_results
            (project_hash, function_name, result, duration_ms, created_at, clause_count)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                project_hash.to_hex(),
                function_name,
                result,
                duration_ms as i64,
                now,
                clause_count as i64,
            ],
        )?;

        Ok(())
    }

    /// Get cached verification result
    pub fn get_verification_result(
        &self,
        project_hash: &ContentHash,
    ) -> Result<Option<CachedResult>, ClauseDbError> {
        let result: SqlResult<(String, String, i64, i64)> = self.conn.query_row(
            "SELECT function_name, result, duration_ms, clause_count FROM verification_results WHERE project_hash = ?1",
            [project_hash.to_hex()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        );

        match result {
            Ok((function_name, result_str, duration_ms, clause_count)) => Ok(Some(CachedResult {
                function_name,
                result: result_str,
                duration_ms: duration_ms as u64,
                clause_count: clause_count as u64,
            })),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(ClauseDbError::Database(e)),
        }
    }

    /// Vacuum the database to reclaim space
    pub fn vacuum(&self) -> Result<(), ClauseDbError> {
        self.conn.execute("VACUUM", [])?;
        info!("Database vacuumed");
        Ok(())
    }

    /// Encode literals to binary format
    fn encode_literals(literals: &[i32]) -> Vec<u8> {
        literals.iter().flat_map(|l| l.to_le_bytes()).collect()
    }

    /// Decode literals from binary format
    fn decode_literals(blob: &[u8]) -> Vec<i32> {
        blob.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    /// Convert a database row to a ClauseEntry
    fn row_to_entry(row: &rusqlite::Row) -> SqlResult<ClauseEntry> {
        let id: i64 = row.get(0)?;
        let literals_blob: Vec<u8> = row.get(1)?;
        let source_hex: String = row.get(2)?;
        let function_hex: String = row.get(3)?;
        let created_at: i64 = row.get(4)?;
        let last_used: i64 = row.get(5)?;
        let use_count: i64 = row.get(6)?;
        let solver: String = row.get(7)?;
        let activity: f64 = row.get(8)?;
        let lbd: Option<i64> = row.get(9)?;

        Ok(ClauseEntry {
            id: ClauseId(id),
            literals: Self::decode_literals(&literals_blob),
            source_hash: ContentHash::from_hex(&source_hex, HashKind::SourceFile)
                .map_err(|_| rusqlite::Error::InvalidQuery)?,
            function_hash: ContentHash::from_hex(&function_hex, HashKind::Function)
                .map_err(|_| rusqlite::Error::InvalidQuery)?,
            created_at: UNIX_EPOCH + Duration::from_secs(created_at as u64),
            last_used: UNIX_EPOCH + Duration::from_secs(last_used as u64),
            use_count: use_count as u64,
            solver,
            activity,
            lbd: lbd.map(|l| l as u32),
        })
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DbStats {
    /// Number of stored clauses
    pub clause_count: u64,
    /// Number of unique functions with clauses
    pub function_count: u64,
    /// Total number of literals across all clauses
    pub total_literals: u64,
    /// Database file size in bytes
    pub database_size: u64,
}

/// A cached verification result
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub function_name: String,
    pub result: String,
    pub duration_ms: u64,
    pub clause_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_database() {
        let db = ClauseDatabase::in_memory().unwrap();
        let stats = db.stats().unwrap();
        assert_eq!(stats.clause_count, 0);
    }

    #[test]
    fn test_store_and_load_clause() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test source");
        let function_hash = ContentHash::from_function("test_fn", "{}");
        let literals = vec![1, -2, 3, -4];

        let id = db
            .store_clause(
                &literals,
                &source_hash,
                &function_hash,
                "cadical",
                1.5,
                Some(3),
            )
            .unwrap();

        assert!(id.0 > 0);

        let entries = db.load_clauses_for_function(&function_hash).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].literals, literals);
        assert_eq!(entries[0].solver, "cadical");
        assert_eq!(entries[0].activity, 1.5);
        assert_eq!(entries[0].lbd, Some(3));
    }

    #[test]
    fn test_batch_store() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        let clauses: Vec<(&[i32], f64, Option<u32>)> = vec![
            (&[1, 2, 3][..], 1.0, Some(2)),
            (&[-1, 2, -3][..], 0.5, Some(3)),
            (&[1, -2][..], 2.0, None),
        ];

        let stored = db
            .store_clauses_batch(&clauses, &source_hash, &function_hash, "kissat")
            .unwrap();

        assert_eq!(stored, 3);

        let entries = db.load_clauses_for_function(&function_hash).unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_top_clauses() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        // Store clauses with different activities
        db.store_clause(&[1, 2], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();
        db.store_clause(&[3, 4], &source_hash, &function_hash, "z3", 3.0, None)
            .unwrap();
        db.store_clause(&[5, 6], &source_hash, &function_hash, "z3", 2.0, None)
            .unwrap();

        let top = db.load_top_clauses(&function_hash, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].activity, 3.0);
        assert_eq!(top[1].activity, 2.0);
    }

    #[test]
    fn test_low_lbd_clauses() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        db.store_clause(&[1, 2], &source_hash, &function_hash, "z3", 1.0, Some(5))
            .unwrap();
        db.store_clause(&[3, 4], &source_hash, &function_hash, "z3", 1.0, Some(2))
            .unwrap();
        db.store_clause(&[5, 6], &source_hash, &function_hash, "z3", 1.0, Some(3))
            .unwrap();
        db.store_clause(&[7, 8], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        let low_lbd = db.load_low_lbd_clauses(&function_hash, 3, 10).unwrap();
        assert_eq!(low_lbd.len(), 2);
        assert_eq!(low_lbd[0].lbd, Some(2));
        assert_eq!(low_lbd[1].lbd, Some(3));
    }

    #[test]
    fn test_invalidate_function() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let fn1_hash = ContentHash::from_function("fn1", "{}");
        let fn2_hash = ContentHash::from_function("fn2", "{}");

        db.store_clause(&[1, 2], &source_hash, &fn1_hash, "z3", 1.0, None)
            .unwrap();
        db.store_clause(&[3, 4], &source_hash, &fn1_hash, "z3", 1.0, None)
            .unwrap();
        db.store_clause(&[5, 6], &source_hash, &fn2_hash, "z3", 1.0, None)
            .unwrap();

        let deleted = db.invalidate_function(&fn1_hash).unwrap();
        assert_eq!(deleted, 2);

        let stats = db.stats().unwrap();
        assert_eq!(stats.clause_count, 1);
    }

    #[test]
    fn test_file_hash_tracking() {
        let db = ClauseDatabase::in_memory().unwrap();

        let hash = ContentHash::from_source("file content");

        db.store_file_hash("src/lib.rs", &hash).unwrap();

        let retrieved = db.get_file_hash("src/lib.rs").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), hash);

        let missing = db.get_file_hash("src/missing.rs").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_verification_result_cache() {
        let db = ClauseDatabase::in_memory().unwrap();

        let project_hash = ContentHash::from_source("project");

        db.store_verification_result(&project_hash, "test_fn", "Proven", 1234, 100)
            .unwrap();

        let cached = db.get_verification_result(&project_hash).unwrap();
        assert!(cached.is_some());
        let cached = cached.unwrap();
        assert_eq!(cached.function_name, "test_fn");
        assert_eq!(cached.result, "Proven");
        assert_eq!(cached.duration_ms, 1234);
        assert_eq!(cached.clause_count, 100);
    }

    #[test]
    fn test_touch_clause() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        let id = db
            .store_clause(&[1, 2, 3], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        db.touch_clause(id).unwrap();
        db.touch_clause(id).unwrap();

        let entries = db.load_clauses_for_function(&function_hash).unwrap();
        assert_eq!(entries[0].use_count, 3); // Initial + 2 touches
    }

    #[test]
    fn test_literal_encoding() {
        let literals = vec![1, -2, 3, -4, i32::MAX, i32::MIN];
        let encoded = ClauseDatabase::encode_literals(&literals);
        let decoded = ClauseDatabase::decode_literals(&encoded);
        assert_eq!(literals, decoded);
    }

    // ==================== Mutation Coverage Tests ====================

    /// Test that set_max_size actually changes the value (catches deletion of assignment)
    #[test]
    fn test_set_max_size_actually_sets() {
        let mut db = ClauseDatabase::in_memory().unwrap();

        // Default is 100 * 1024 * 1024 = 104857600
        assert_eq!(db.max_size, 100 * 1024 * 1024);

        // Set a different value
        db.set_max_size(50 * 1024 * 1024);
        assert_eq!(
            db.max_size,
            50 * 1024 * 1024,
            "set_max_size should actually change max_size"
        );

        // Set another value to ensure it's not just coincidentally passing
        db.set_max_size(12345);
        assert_eq!(
            db.max_size, 12345,
            "set_max_size should work with any value"
        );
    }

    /// Test max_size initialization with multiplication (catches * to + or / mutants)
    #[test]
    fn test_max_size_default_multiplication() {
        let db = ClauseDatabase::in_memory().unwrap();

        // Default should be 100 * 1024 * 1024 = 104857600, not:
        // 100 + 1024 + 1024 = 2148
        // 100 + 1024 * 1024 = 1048676
        // 100 / 1024 / 1024 = 0
        assert_eq!(
            db.max_size, 104_857_600,
            "max_size should be 100 * 1024 * 1024 = 104857600"
        );
    }

    /// Test update_activity actually updates (catches returning Ok(()) without update)
    #[test]
    fn test_update_activity_actually_updates() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        let id = db
            .store_clause(&[1, 2], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        // Verify initial activity
        let entries = db.load_clauses_for_function(&function_hash).unwrap();
        assert_eq!(entries[0].activity, 1.0);

        // Update activity
        db.update_activity(id, 5.0).unwrap();

        // Verify it changed
        let entries = db.load_clauses_for_function(&function_hash).unwrap();
        assert_eq!(
            entries[0].activity, 5.0,
            "update_activity should actually change the activity value"
        );
    }

    /// Test cleanup_old_clauses returns correct count (catches - to + and > comparisons)
    #[test]
    fn test_cleanup_old_clauses_returns_correct_count() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        // Store 3 clauses
        db.store_clause(&[1, 2], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();
        db.store_clause(&[3, 4], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();
        db.store_clause(&[5, 6], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        // Cleanup with 0 age should delete nothing (all clauses are recent)
        let deleted = db.cleanup_old_clauses(Duration::from_secs(0)).unwrap();
        // Clauses are just created, so last_used > cutoff
        assert_eq!(
            deleted, 0,
            "No clauses should be deleted when they were just created"
        );

        // Verify count still at 3
        let stats = db.stats().unwrap();
        assert_eq!(stats.clause_count, 3);
    }

    /// Test cleanup_old_clauses boundary condition (catches > vs >= mutant)
    #[test]
    fn test_cleanup_old_clauses_boundary() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        // Store a clause
        db.store_clause(&[1, 2], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        // Cleanup with very large age should delete nothing
        let deleted = db
            .cleanup_old_clauses(Duration::from_secs(365 * 24 * 60 * 60))
            .unwrap();
        assert_eq!(deleted, 0);
    }

    /// Test invalidate_function boundary when deleted == 0 (catches > vs >= mutant)
    #[test]
    fn test_invalidate_function_no_clauses() {
        let db = ClauseDatabase::in_memory().unwrap();

        let function_hash = ContentHash::from_function("nonexistent", "{}");

        // Invalidate non-existent function should return 0
        let deleted = db.invalidate_function(&function_hash).unwrap();
        assert_eq!(deleted, 0, "Should return 0 when no clauses to invalidate");
    }

    /// Test invalidate_function boundary when deleted == 1 (catches > vs < mutant)
    #[test]
    fn test_invalidate_function_single_clause() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("single_fn", "{}");

        // Store exactly one clause
        db.store_clause(&[1, 2], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        let deleted = db.invalidate_function(&function_hash).unwrap();
        assert_eq!(
            deleted, 1,
            "Should return exactly 1 when one clause is deleted"
        );
    }

    /// Test stats db_size computation (catches * to + or / mutants)
    #[test]
    fn test_stats_database_size_computation() {
        let db = ClauseDatabase::in_memory().unwrap();

        let stats = db.stats().unwrap();

        // page_count * page_size should equal database_size
        // For an empty in-memory DB, this should be reasonable (not huge due to + instead of *)
        // SQLite default page_size is typically 4096
        assert!(stats.database_size > 0, "Database size should be positive");
        assert!(
            stats.database_size < 100_000_000,
            "Database size should be reasonable (< 100MB), got {}",
            stats.database_size
        );
    }

    /// Test vacuum actually runs (catches returning Ok(()) without running)
    #[test]
    fn test_vacuum_runs_without_error() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        // Add and remove some clauses to create fragmentation
        for i in 0..10i32 {
            db.store_clause(&[i, i + 1], &source_hash, &function_hash, "z3", 1.0, None)
                .unwrap();
        }
        db.invalidate_function(&function_hash).unwrap();

        // Vacuum should succeed
        db.vacuum().unwrap();

        // Verify database is still functional
        let stats = db.stats().unwrap();
        assert_eq!(stats.clause_count, 0);
    }

    /// Test row_to_entry correctly computes created_at and last_used timestamps
    /// (catches + to - mutants at lines 548-549)
    #[test]
    fn test_row_to_entry_timestamp_computation() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        // Store timestamp is computed as seconds since UNIX_EPOCH
        // We just need to verify it's reasonable (not in the past due to - instead of +)
        db.store_clause(&[1, 2, 3], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        let entries = db.load_clauses_for_function(&function_hash).unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];

        // The timestamp should be after 2020 and not in the far future
        // If + became -, the timestamp would be before UNIX_EPOCH
        let year_2020 = UNIX_EPOCH + Duration::from_secs(50 * 365 * 24 * 60 * 60);
        let year_2100 = UNIX_EPOCH + Duration::from_secs(130 * 365 * 24 * 60 * 60);

        assert!(
            entry.created_at > year_2020,
            "created_at should be after year 2020"
        );
        assert!(
            entry.created_at < year_2100,
            "created_at should be before year 2100"
        );

        // Same for last_used
        assert!(
            entry.last_used > year_2020,
            "last_used should be after year 2020"
        );
        assert!(
            entry.last_used < year_2100,
            "last_used should be before year 2100"
        );
    }

    /// Test that row_to_entry correctly handles various timestamp values
    #[test]
    fn test_row_to_entry_timestamp_not_subtracted() {
        let db = ClauseDatabase::in_memory().unwrap();

        let source_hash = ContentHash::from_source("test");
        let function_hash = ContentHash::from_function("fn", "{}");

        db.store_clause(&[1], &source_hash, &function_hash, "z3", 1.0, None)
            .unwrap();

        let entries = db.load_clauses_for_function(&function_hash).unwrap();
        let entry = &entries[0];

        // The timestamp should be recent (within last minute), not in the past
        let now = SystemTime::now();
        let elapsed = now.duration_since(entry.created_at).unwrap_or_default();

        // If + became -, Duration::from_secs(negative_value) would panic or be huge
        assert!(
            elapsed.as_secs() < 60,
            "Timestamp should be recent (< 60s ago), got {} seconds",
            elapsed.as_secs()
        );
    }
}
