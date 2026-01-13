//! Invariant Corpus - Storage and retrieval of successful invariants
//!
//! The corpus stores successfully verified invariants along with metadata
//! about the systems they came from. This enables:
//! - Fast lookup of similar problems
//! - Learning from past successes
//! - Building a knowledge base for invariant synthesis

use kani_fast_kinduction::{Property, SmtType, StateFormula, TransitionSystem};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::AiError;

/// Database row type for structural match query
/// Fields: id, invariant_formula, invariant_description, system_hash, variable_types, property_hash, success_count, last_used
type StructuralMatchRow = (
    i64,
    String,
    Option<String>,
    String,
    String,
    String,
    u32,
    i64,
);

/// A seed pattern for pre-populating the corpus
#[derive(Debug, Clone)]
struct SeedPattern {
    formula: String,
    description: Option<String>,
    var_types: String,
    system_hash: String,
    property_hash: String,
    base_success_count: u32,
}

/// Configuration for the invariant corpus
#[derive(Debug, Clone)]
pub struct CorpusConfig {
    /// Path to the corpus database
    pub db_path: PathBuf,
    /// Maximum number of entries to keep
    pub max_entries: usize,
    /// Similarity threshold for lookup (0.0 to 1.0)
    pub similarity_threshold: f64,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        let db_path = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("kani-fast")
            .join("invariant_corpus.db");

        Self {
            db_path,
            max_entries: 10000,
            similarity_threshold: 0.7,
        }
    }
}

/// An entry in the invariant corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantEntry {
    /// Unique identifier
    pub id: i64,
    /// The invariant formula
    pub invariant: StateFormula,
    /// Hash of the system structure (for similarity lookup)
    pub system_hash: String,
    /// Variable types in the system
    pub variable_types: Vec<SmtType>,
    /// Property hash
    pub property_hash: String,
    /// Number of times this invariant was used successfully
    pub success_count: u32,
    /// Timestamp of last use
    pub last_used: i64,
}

/// Invariant corpus database
pub struct InvariantCorpus {
    conn: Connection,
    config: CorpusConfig,
}

impl InvariantCorpus {
    /// Open the corpus database with default configuration
    pub fn open_default() -> Result<Self, AiError> {
        Self::open(CorpusConfig::default())
    }

    /// Open the corpus database with custom configuration
    pub fn open(config: CorpusConfig) -> Result<Self, AiError> {
        // Ensure parent directory exists
        if let Some(parent) = config.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&config.db_path)
            .map_err(|e| AiError::Database(format!("Failed to open corpus: {e}")))?;

        let corpus = Self { conn, config };
        corpus.init_schema()?;

        Ok(corpus)
    }

    /// Open an in-memory corpus for testing
    pub fn open_in_memory() -> Result<Self, AiError> {
        let conn = Connection::open_in_memory()
            .map_err(|e| AiError::Database(format!("Failed to open in-memory corpus: {e}")))?;

        let config = CorpusConfig {
            db_path: PathBuf::from(":memory:"),
            ..Default::default()
        };

        let corpus = Self { conn, config };
        corpus.init_schema()?;

        Ok(corpus)
    }

    /// Initialize the database schema
    fn init_schema(&self) -> Result<(), AiError> {
        self.conn
            .execute_batch(
                r"
            CREATE TABLE IF NOT EXISTS invariants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invariant_formula TEXT NOT NULL,
                invariant_description TEXT,
                system_hash TEXT NOT NULL,
                variable_types TEXT NOT NULL,
                property_hash TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                last_used INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_system_hash ON invariants(system_hash);
            CREATE INDEX IF NOT EXISTS idx_property_hash ON invariants(property_hash);
            CREATE INDEX IF NOT EXISTS idx_success_count ON invariants(success_count DESC);
            ",
            )
            .map_err(|e| AiError::Database(format!("Failed to initialize schema: {e}")))?;

        Ok(())
    }

    /// Store a successful invariant
    pub fn store(
        &mut self,
        system: &TransitionSystem,
        property: &Property,
        invariant: &StateFormula,
    ) -> Result<i64, AiError> {
        let system_hash = self.hash_system(system);
        let property_hash = self.hash_property(property);
        let var_types = self.encode_variable_types(system);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Check if we already have this invariant for this system
        let existing: Option<i64> = self.conn.query_row(
            "SELECT id FROM invariants WHERE system_hash = ? AND property_hash = ? AND invariant_formula = ?",
            params![system_hash, property_hash, invariant.smt_formula],
            |row| row.get(0),
        ).optional().map_err(|e| AiError::Database(e.to_string()))?;

        if let Some(id) = existing {
            // Update success count
            self.conn
                .execute(
                    "UPDATE invariants SET success_count = success_count + 1, last_used = ? WHERE id = ?",
                    params![now, id],
                )
                .map_err(|e| AiError::Database(e.to_string()))?;
            return Ok(id);
        }

        // Insert new entry
        self.conn
            .execute(
                r"
            INSERT INTO invariants (invariant_formula, invariant_description, system_hash,
                                   variable_types, property_hash, last_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ",
                params![
                    invariant.smt_formula,
                    invariant.description,
                    system_hash,
                    var_types,
                    property_hash,
                    now,
                    now
                ],
            )
            .map_err(|e| AiError::Database(e.to_string()))?;

        let id = self.conn.last_insert_rowid();

        // Enforce max entries
        self.prune_old_entries()?;

        Ok(id)
    }

    /// Find a similar invariant in the corpus
    pub fn find_similar(
        &self,
        system: &TransitionSystem,
        property: &Property,
    ) -> Result<Option<InvariantEntry>, AiError> {
        let system_hash = self.hash_system(system);
        let property_hash = self.hash_property(property);

        // First try exact match
        if let Some(entry) = self.find_exact(system, property)? {
            return Ok(Some(entry));
        }

        // Then try structural match (same variable types)
        let var_types = self.encode_variable_types(system);

        let result: Option<StructuralMatchRow> = self
            .conn
            .query_row(
                r"
                SELECT id, invariant_formula, invariant_description, system_hash,
                       variable_types, property_hash, success_count, last_used
                FROM invariants
                WHERE variable_types = ?
                ORDER BY success_count DESC, last_used DESC
                LIMIT 1
                ",
                params![var_types],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| AiError::Database(e.to_string()))?;

        if let Some((
            id,
            formula,
            description,
            _sys_hash,
            var_types_str,
            _prop_hash,
            count,
            last_used,
        )) = result
        {
            let entry = InvariantEntry {
                id,
                invariant: StateFormula {
                    smt_formula: formula,
                    description,
                },
                system_hash,
                variable_types: self.decode_variable_types(&var_types_str),
                property_hash,
                success_count: count,
                last_used,
            };
            return Ok(Some(entry));
        }

        Ok(None)
    }

    /// Find an exact match in the corpus
    pub fn find_exact(
        &self,
        system: &TransitionSystem,
        property: &Property,
    ) -> Result<Option<InvariantEntry>, AiError> {
        let system_hash = self.hash_system(system);
        let property_hash = self.hash_property(property);

        let result: Option<(i64, String, Option<String>, String, u32, i64)> = self
            .conn
            .query_row(
                r"
                SELECT id, invariant_formula, invariant_description, variable_types,
                       success_count, last_used
                FROM invariants
                WHERE system_hash = ? AND property_hash = ?
                ORDER BY success_count DESC
                LIMIT 1
                ",
                params![system_hash, property_hash],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| AiError::Database(e.to_string()))?;

        if let Some((id, formula, description, var_types_str, count, last_used)) = result {
            let entry = InvariantEntry {
                id,
                invariant: StateFormula {
                    smt_formula: formula,
                    description,
                },
                system_hash,
                variable_types: self.decode_variable_types(&var_types_str),
                property_hash,
                success_count: count,
                last_used,
            };
            return Ok(Some(entry));
        }

        Ok(None)
    }

    /// Get statistics about the corpus
    pub fn stats(&self) -> Result<CorpusStats, AiError> {
        let total_entries: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM invariants", [], |row| row.get(0))
            .map_err(|e| AiError::Database(e.to_string()))?;

        let total_uses: i64 = self
            .conn
            .query_row(
                "SELECT COALESCE(SUM(success_count), 0) FROM invariants",
                [],
                |row| row.get(0),
            )
            .map_err(|e| AiError::Database(e.to_string()))?;

        let unique_systems: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(DISTINCT system_hash) FROM invariants",
                [],
                |row| row.get(0),
            )
            .map_err(|e| AiError::Database(e.to_string()))?;

        Ok(CorpusStats {
            total_entries: total_entries as usize,
            total_uses: total_uses as usize,
            unique_systems: unique_systems as usize,
        })
    }

    /// Get top invariants by success count
    pub fn top_invariants(&self, limit: usize) -> Result<Vec<InvariantEntry>, AiError> {
        let mut stmt = self
            .conn
            .prepare(
                r"
            SELECT id, invariant_formula, invariant_description, system_hash,
                   variable_types, property_hash, success_count, last_used
            FROM invariants
            ORDER BY success_count DESC
            LIMIT ?
            ",
            )
            .map_err(|e| AiError::Database(e.to_string()))?;

        let rows = stmt
            .query_map([limit as i64], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, u32>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            })
            .map_err(|e| AiError::Database(e.to_string()))?;

        let mut entries = Vec::new();
        for row in rows {
            let (
                id,
                formula,
                description,
                system_hash,
                var_types_str,
                property_hash,
                count,
                last_used,
            ) = row.map_err(|e| AiError::Database(e.to_string()))?;

            entries.push(InvariantEntry {
                id,
                invariant: StateFormula {
                    smt_formula: formula,
                    description,
                },
                system_hash,
                variable_types: self.decode_variable_types(&var_types_str),
                property_hash,
                success_count: count,
                last_used,
            });
        }

        Ok(entries)
    }

    /// Hash a transition system for similarity lookup
    fn hash_system(&self, system: &TransitionSystem) -> String {
        let mut hasher = blake3::Hasher::new();

        // Hash variable structure
        for var in &system.variables {
            hasher.update(var.name.as_bytes());
            hasher.update(var.smt_type.to_smt_string().as_bytes());
        }

        // Hash init and transition patterns (normalized)
        hasher.update(self.normalize_formula(&system.init.smt_formula).as_bytes());
        hasher.update(
            self.normalize_formula(&system.transition.smt_formula)
                .as_bytes(),
        );

        hasher.finalize().to_hex().to_string()
    }

    /// Hash a property for lookup
    fn hash_property(&self, property: &Property) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(
            self.normalize_formula(&property.formula.smt_formula)
                .as_bytes(),
        );
        hasher.finalize().to_hex().to_string()
    }

    /// Normalize a formula for hashing (remove whitespace variations)
    fn normalize_formula(&self, formula: &str) -> String {
        formula.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Encode variable types as a string
    fn encode_variable_types(&self, system: &TransitionSystem) -> String {
        system
            .variables
            .iter()
            .map(|v| v.smt_type.to_smt_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Decode variable types from string
    fn decode_variable_types(&self, encoded: &str) -> Vec<SmtType> {
        encoded
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| match s {
                "Int" => SmtType::Int,
                "Bool" => SmtType::Bool,
                "Real" => SmtType::Real,
                other if other.starts_with("(_ BitVec ") => {
                    let width_str = other
                        .strip_prefix("(_ BitVec ")
                        .and_then(|s| s.strip_suffix(')'))
                        .unwrap_or("32");
                    SmtType::BitVec(width_str.parse().unwrap_or(32))
                }
                _ => SmtType::Int, // Default fallback
            })
            .collect()
    }

    /// Seed the corpus with common invariant patterns
    ///
    /// This pre-populates the corpus with frequently-occurring invariant patterns
    /// that are useful for many verification problems:
    /// - Non-negativity (x >= 0)
    /// - Upper bounds (x <= N)
    /// - Range constraints (0 <= x <= N)
    /// - Monotonicity (x' >= x)
    /// - Counter invariants
    /// - Index bounds
    /// - Boolean relationships
    ///
    /// Returns the number of patterns seeded.
    pub fn seed_common_patterns(&mut self) -> Result<usize, AiError> {
        let patterns = Self::common_invariant_patterns();
        let mut count = 0;

        for pattern in &patterns {
            // Only seed if not already present
            let existing: Option<i64> = self
                .conn
                .query_row(
                    "SELECT id FROM invariants WHERE invariant_formula = ?",
                    params![pattern.formula],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| AiError::Database(e.to_string()))?;

            if existing.is_none() {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;

                self.conn
                    .execute(
                        r"
                        INSERT INTO invariants (invariant_formula, invariant_description, system_hash,
                                               variable_types, property_hash, success_count, last_used, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ",
                        params![
                            pattern.formula,
                            pattern.description,
                            pattern.system_hash,
                            pattern.var_types,
                            pattern.property_hash,
                            pattern.base_success_count,
                            now,
                            now
                        ],
                    )
                    .map_err(|e| AiError::Database(e.to_string()))?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Get the list of common invariant patterns for seeding
    fn common_invariant_patterns() -> Vec<SeedPattern> {
        vec![
            // Non-negativity patterns (most common)
            SeedPattern {
                formula: "(>= x 0)".to_string(),
                description: Some("Non-negativity: variable x is always non-negative".to_string()),
                var_types: "Int".to_string(),
                system_hash: "seed_nonneg_int".to_string(),
                property_hash: "prop_nonneg".to_string(),
                base_success_count: 100,
            },
            SeedPattern {
                formula: "(>= i 0)".to_string(),
                description: Some("Non-negativity: index i is always non-negative".to_string()),
                var_types: "Int".to_string(),
                system_hash: "seed_nonneg_idx".to_string(),
                property_hash: "prop_nonneg_idx".to_string(),
                base_success_count: 100,
            },
            SeedPattern {
                formula: "(>= n 0)".to_string(),
                description: Some("Non-negativity: counter n is always non-negative".to_string()),
                var_types: "Int".to_string(),
                system_hash: "seed_nonneg_counter".to_string(),
                property_hash: "prop_nonneg_n".to_string(),
                base_success_count: 100,
            },
            // Upper bound patterns
            SeedPattern {
                formula: "(<= x n)".to_string(),
                description: Some("Upper bound: x is bounded by n".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_upperbound".to_string(),
                property_hash: "prop_upper".to_string(),
                base_success_count: 80,
            },
            SeedPattern {
                formula: "(< i len)".to_string(),
                description: Some("Index bound: index i is less than length".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_idx_bound".to_string(),
                property_hash: "prop_idx_bound".to_string(),
                base_success_count: 90,
            },
            // Range patterns
            SeedPattern {
                formula: "(and (>= x 0) (<= x n))".to_string(),
                description: Some("Range: x is in range [0, n]".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_range".to_string(),
                property_hash: "prop_range".to_string(),
                base_success_count: 75,
            },
            SeedPattern {
                formula: "(and (>= i 0) (< i len))".to_string(),
                description: Some("Valid index: i is a valid array index".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_valid_idx".to_string(),
                property_hash: "prop_valid_idx".to_string(),
                base_success_count: 95,
            },
            // Counter patterns
            SeedPattern {
                formula: "(= x (* k step))".to_string(),
                description: Some("Counter multiple: x is a multiple of step".to_string()),
                var_types: "Int,Int,Int".to_string(),
                system_hash: "seed_counter_mult".to_string(),
                property_hash: "prop_mult".to_string(),
                base_success_count: 60,
            },
            SeedPattern {
                formula: "(>= x init)".to_string(),
                description: Some("Monotonic increase: x is at least init".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_mono_inc".to_string(),
                property_hash: "prop_mono".to_string(),
                base_success_count: 70,
            },
            SeedPattern {
                formula: "(<= x init)".to_string(),
                description: Some("Monotonic decrease: x is at most init".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_mono_dec".to_string(),
                property_hash: "prop_mono_dec".to_string(),
                base_success_count: 70,
            },
            // Sum/accumulator patterns
            SeedPattern {
                formula: "(= sum (+ sum_prev x))".to_string(),
                description: Some("Sum accumulation: sum is sum of previous values".to_string()),
                var_types: "Int,Int,Int".to_string(),
                system_hash: "seed_sum".to_string(),
                property_hash: "prop_sum".to_string(),
                base_success_count: 50,
            },
            SeedPattern {
                formula: "(>= sum 0)".to_string(),
                description: Some("Sum non-negative: accumulated sum is non-negative".to_string()),
                var_types: "Int".to_string(),
                system_hash: "seed_sum_nonneg".to_string(),
                property_hash: "prop_sum_nonneg".to_string(),
                base_success_count: 65,
            },
            // Boolean patterns
            SeedPattern {
                formula: "(=> done (= result expected))".to_string(),
                description: Some("Postcondition: when done, result equals expected".to_string()),
                var_types: "Bool,Int,Int".to_string(),
                system_hash: "seed_postcond".to_string(),
                property_hash: "prop_postcond".to_string(),
                base_success_count: 55,
            },
            SeedPattern {
                formula: "(or (not started) (>= progress 0))".to_string(),
                description: Some("Progress: once started, progress is non-negative".to_string()),
                var_types: "Bool,Int".to_string(),
                system_hash: "seed_progress".to_string(),
                property_hash: "prop_progress".to_string(),
                base_success_count: 50,
            },
            // Loop invariant patterns
            SeedPattern {
                formula: "(and (>= i 0) (<= i n) (= acc (f i)))".to_string(),
                description: Some(
                    "Loop invariant: index bounded, accumulator is function of i".to_string(),
                ),
                var_types: "Int,Int,Int".to_string(),
                system_hash: "seed_loop_inv".to_string(),
                property_hash: "prop_loop".to_string(),
                base_success_count: 45,
            },
            // Bitvector patterns
            SeedPattern {
                formula: "(bvuge x #x00000000)".to_string(),
                description: Some("Bitvector non-negative: unsigned x >= 0".to_string()),
                var_types: "(_ BitVec 32)".to_string(),
                system_hash: "seed_bv_nonneg".to_string(),
                property_hash: "prop_bv_nonneg".to_string(),
                base_success_count: 40,
            },
            SeedPattern {
                formula: "(bvult i len)".to_string(),
                description: Some("Bitvector index bound: unsigned i < len".to_string()),
                var_types: "(_ BitVec 32),(_ BitVec 32)".to_string(),
                system_hash: "seed_bv_idx".to_string(),
                property_hash: "prop_bv_idx".to_string(),
                base_success_count: 45,
            },
            // Equality preservation
            SeedPattern {
                formula: "(= x old_x)".to_string(),
                description: Some("Unchanged: x equals its initial value".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_unchanged".to_string(),
                property_hash: "prop_unchanged".to_string(),
                base_success_count: 35,
            },
            // Difference bound
            SeedPattern {
                formula: "(<= (- y x) k)".to_string(),
                description: Some("Difference bound: y - x is bounded by k".to_string()),
                var_types: "Int,Int,Int".to_string(),
                system_hash: "seed_diff_bound".to_string(),
                property_hash: "prop_diff".to_string(),
                base_success_count: 40,
            },
            // Parity
            SeedPattern {
                formula: "(= (mod x 2) (mod init 2))".to_string(),
                description: Some("Parity preservation: x has same parity as init".to_string()),
                var_types: "Int,Int".to_string(),
                system_hash: "seed_parity".to_string(),
                property_hash: "prop_parity".to_string(),
                base_success_count: 30,
            },
        ]
    }

    /// Check if the corpus has been seeded with common patterns
    pub fn is_seeded(&self) -> Result<bool, AiError> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM invariants WHERE system_hash LIKE 'seed_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| AiError::Database(e.to_string()))?;

        Ok(count > 0)
    }

    /// Get all seeded patterns (for diagnostics)
    pub fn seeded_patterns(&self) -> Result<Vec<InvariantEntry>, AiError> {
        let mut stmt = self
            .conn
            .prepare(
                r"
                SELECT id, invariant_formula, invariant_description, system_hash,
                       variable_types, property_hash, success_count, last_used
                FROM invariants
                WHERE system_hash LIKE 'seed_%'
                ORDER BY success_count DESC
                ",
            )
            .map_err(|e| AiError::Database(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, u32>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            })
            .map_err(|e| AiError::Database(e.to_string()))?;

        let mut entries = Vec::new();
        for row in rows {
            let (
                id,
                formula,
                description,
                system_hash,
                var_types_str,
                property_hash,
                count,
                last_used,
            ) = row.map_err(|e| AiError::Database(e.to_string()))?;

            entries.push(InvariantEntry {
                id,
                invariant: StateFormula {
                    smt_formula: formula,
                    description,
                },
                system_hash,
                variable_types: self.decode_variable_types(&var_types_str),
                property_hash,
                success_count: count,
                last_used,
            });
        }

        Ok(entries)
    }

    /// Prune old entries to stay under max_entries
    fn prune_old_entries(&self) -> Result<(), AiError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM invariants", [], |row| row.get(0))
            .map_err(|e| AiError::Database(e.to_string()))?;

        if count as usize > self.config.max_entries {
            let to_delete = count as usize - self.config.max_entries;
            self.conn
                .execute(
                    r"
                DELETE FROM invariants WHERE id IN (
                    SELECT id FROM invariants
                    ORDER BY success_count ASC, last_used ASC
                    LIMIT ?
                )
                ",
                    params![to_delete as i64],
                )
                .map_err(|e| AiError::Database(e.to_string()))?;
        }

        Ok(())
    }
}

/// Statistics about the corpus
#[derive(Debug, Clone)]
pub struct CorpusStats {
    /// Total number of invariant entries
    pub total_entries: usize,
    /// Total number of times invariants were used
    pub total_uses: usize,
    /// Number of unique systems
    pub unique_systems: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kani_fast_kinduction::TransitionSystemBuilder;

    fn test_system() -> TransitionSystem {
        TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build()
    }

    fn test_property() -> Property {
        Property::safety("p1", "non_negative", StateFormula::new("(>= x 0)"))
    }

    #[test]
    fn test_corpus_creation() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();
        let stats = corpus.stats().unwrap();
        assert_eq!(stats.total_entries, 0);
    }

    #[test]
    fn test_store_and_find() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");

        // Store the invariant
        let id = corpus.store(&system, &property, &invariant).unwrap();
        assert!(id > 0);

        // Find it again
        let found = corpus.find_exact(&system, &property).unwrap();
        assert!(found.is_some());
        let entry = found.unwrap();
        assert_eq!(entry.invariant.smt_formula, "(>= x 0)");
        assert_eq!(entry.success_count, 1);
    }

    #[test]
    fn test_success_count_increment() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");

        // Store the same invariant twice
        corpus.store(&system, &property, &invariant).unwrap();
        corpus.store(&system, &property, &invariant).unwrap();

        // Check success count
        let found = corpus.find_exact(&system, &property).unwrap().unwrap();
        assert_eq!(found.success_count, 2);

        // Stats should show 1 entry
        let stats = corpus.stats().unwrap();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_uses, 2);
    }

    #[test]
    fn test_find_similar() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system1 = test_system();
        let property1 = test_property();
        let invariant = StateFormula::new("(>= x 0)");

        corpus.store(&system1, &property1, &invariant).unwrap();

        // Create a different system with same variable types
        let system2 = TransitionSystemBuilder::new()
            .variable("y", SmtType::Int)
            .init("(= y 5)")
            .transition("(= y' (- y 1))")
            .build();

        let property2 = Property::safety("p2", "positive", StateFormula::new("(>= y 0)"));

        // Should find a similar invariant (same variable types)
        let found = corpus.find_similar(&system2, &property2).unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn test_top_invariants() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();

        // Store multiple invariants with different success counts
        let inv1 = StateFormula::new("(>= x 0)");
        let inv2 = StateFormula::new("(and (>= x 0) (<= x 100))");

        corpus.store(&system, &property, &inv1).unwrap();
        corpus.store(&system, &property, &inv1).unwrap();
        corpus.store(&system, &property, &inv1).unwrap();

        // Store inv2 for a slightly different property
        let property2 = Property::safety("p2", "bounded", StateFormula::new("(<= x 100)"));
        corpus.store(&system, &property2, &inv2).unwrap();

        let top = corpus.top_invariants(10).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].success_count, 3); // inv1 should be first
    }

    #[test]
    fn test_variable_type_encoding() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("b", SmtType::Bool)
            .variable("bv", SmtType::BitVec(32))
            .init("true")
            .transition("true")
            .build();

        let encoded = corpus.encode_variable_types(&system);
        assert_eq!(encoded, "Int,Bool,(_ BitVec 32)");

        let decoded = corpus.decode_variable_types(&encoded);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], SmtType::Int);
        assert_eq!(decoded[1], SmtType::Bool);
        assert_eq!(decoded[2], SmtType::BitVec(32));
    }

    #[test]
    fn test_system_hashing() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let system1 = test_system();
        let system2 = test_system();

        // Same systems should have same hash
        let hash1 = corpus.hash_system(&system1);
        let hash2 = corpus.hash_system(&system2);
        assert_eq!(hash1, hash2);

        // Different system should have different hash
        let system3 = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 5)") // Different
            .transition("(= x' (+ x 1))")
            .build();

        let hash3 = corpus.hash_system(&system3);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_corpus_stats() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // Store some invariants
        let system = test_system();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");

        corpus.store(&system, &property, &invariant).unwrap();

        let stats = corpus.stats().unwrap();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_uses, 1);
        assert_eq!(stats.unique_systems, 1);
    }

    // =============================================================================
    // Additional Comprehensive Tests
    // =============================================================================

    #[test]
    fn test_corpus_config_default() {
        let config = CorpusConfig::default();
        assert_eq!(config.max_entries, 10000);
        assert!((config.similarity_threshold - 0.7).abs() < f64::EPSILON);
        assert!(config.db_path.to_string_lossy().contains("kani-fast"));
    }

    #[test]
    fn test_corpus_config_custom() {
        let config = CorpusConfig {
            db_path: PathBuf::from("/tmp/test_corpus.db"),
            max_entries: 500,
            similarity_threshold: 0.9,
        };
        assert_eq!(config.max_entries, 500);
        assert!((config.similarity_threshold - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.db_path, PathBuf::from("/tmp/test_corpus.db"));
    }

    #[test]
    fn test_invariant_entry_clone() {
        let entry = InvariantEntry {
            id: 1,
            invariant: StateFormula::new("(>= x 0)"),
            system_hash: "abc123".to_string(),
            variable_types: vec![SmtType::Int],
            property_hash: "def456".to_string(),
            success_count: 5,
            last_used: 1704067200,
        };

        let cloned = entry.clone();
        assert_eq!(cloned.id, entry.id);
        assert_eq!(cloned.invariant.smt_formula, entry.invariant.smt_formula);
        assert_eq!(cloned.system_hash, entry.system_hash);
        assert_eq!(cloned.success_count, entry.success_count);
    }

    #[test]
    fn test_invariant_entry_debug() {
        let entry = InvariantEntry {
            id: 42,
            invariant: StateFormula::new("(>= x 0)"),
            system_hash: "hash".to_string(),
            variable_types: vec![SmtType::Int, SmtType::Bool],
            property_hash: "prop".to_string(),
            success_count: 10,
            last_used: 1704067200,
        };

        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("hash"));
    }

    #[test]
    fn test_invariant_entry_serialization() {
        let entry = InvariantEntry {
            id: 1,
            invariant: StateFormula::new("(>= x 0)"),
            system_hash: "abc".to_string(),
            variable_types: vec![SmtType::Int],
            property_hash: "def".to_string(),
            success_count: 3,
            last_used: 1704067200,
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("(>= x 0)"));
        assert!(json.contains("abc"));

        let parsed: InvariantEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 1);
        assert_eq!(parsed.success_count, 3);
    }

    #[test]
    fn test_corpus_stats_clone() {
        let stats = CorpusStats {
            total_entries: 100,
            total_uses: 500,
            unique_systems: 25,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.total_entries, 100);
        assert_eq!(cloned.total_uses, 500);
        assert_eq!(cloned.unique_systems, 25);
    }

    #[test]
    fn test_corpus_stats_debug() {
        let stats = CorpusStats {
            total_entries: 10,
            total_uses: 50,
            unique_systems: 5,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("50"));
        assert!(debug_str.contains("5"));
    }

    #[test]
    fn test_find_exact_not_found() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();

        // Empty corpus should return None
        let found = corpus.find_exact(&system, &property).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_find_similar_not_found() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();

        // Empty corpus should return None
        let found = corpus.find_similar(&system, &property).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_find_similar_different_types() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // Store an invariant for Int system
        let int_system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");
        corpus.store(&int_system, &property, &invariant).unwrap();

        // Try to find similar for Bool system (different types)
        let bool_system = TransitionSystemBuilder::new()
            .variable("b", SmtType::Bool)
            .init("(= b true)")
            .transition("(= b' (not b))")
            .build();
        let bool_property = Property::safety("p1", "always_true", StateFormula::new("b"));

        // Should not find similar (different variable types)
        let found = corpus.find_similar(&bool_system, &bool_property).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_property_hashing() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let property1 = Property::safety("p1", "test", StateFormula::new("(>= x 0)"));
        let property2 = Property::safety("p2", "test", StateFormula::new("(>= x 0)"));
        let property3 = Property::safety("p1", "test", StateFormula::new("(>= y 0)"));

        let hash1 = corpus.hash_property(&property1);
        let hash2 = corpus.hash_property(&property2);
        let hash3 = corpus.hash_property(&property3);

        // Same formula should have same hash (name doesn't matter for property hash)
        assert_eq!(hash1, hash2);
        // Different formula should have different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_normalize_formula_whitespace() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let f1 = "(>=  x   0)";
        let f2 = "(>= x 0)";
        let f3 = "(>=\n\tx\n0)";

        let n1 = corpus.normalize_formula(f1);
        let n2 = corpus.normalize_formula(f2);
        let n3 = corpus.normalize_formula(f3);

        assert_eq!(n1, n2);
        assert_eq!(n2, n3);
        assert_eq!(n1, "(>= x 0)");
    }

    #[test]
    fn test_decode_variable_types_empty() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let decoded = corpus.decode_variable_types("");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_variable_types_real() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let decoded = corpus.decode_variable_types("Real");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], SmtType::Real);
    }

    #[test]
    fn test_decode_variable_types_bitvec_various_widths() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let decoded8 = corpus.decode_variable_types("(_ BitVec 8)");
        assert_eq!(decoded8.len(), 1);
        assert_eq!(decoded8[0], SmtType::BitVec(8));

        let decoded64 = corpus.decode_variable_types("(_ BitVec 64)");
        assert_eq!(decoded64.len(), 1);
        assert_eq!(decoded64[0], SmtType::BitVec(64));
    }

    #[test]
    fn test_decode_variable_types_unknown() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        // Unknown types fall back to Int
        let decoded = corpus.decode_variable_types("UnknownType");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], SmtType::Int);
    }

    #[test]
    fn test_top_invariants_empty_corpus() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let top = corpus.top_invariants(10).unwrap();
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_invariants_limit() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();

        // Store 5 different invariants
        for i in 0..5 {
            let property =
                Property::safety(format!("p{}", i), "test", StateFormula::new("(>= x 0)"));
            let invariant = StateFormula::new(format!("(>= x {})", i));
            corpus.store(&system, &property, &invariant).unwrap();
        }

        // Request only 3
        let top = corpus.top_invariants(3).unwrap();
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_multiple_systems_stats() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // Store invariants for different systems
        let system1 = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let system2 = TransitionSystemBuilder::new()
            .variable("y", SmtType::Int)
            .init("(= y 10)")
            .transition("(= y' (- y 1))")
            .build();

        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");

        corpus.store(&system1, &property, &invariant).unwrap();
        corpus.store(&system2, &property, &invariant).unwrap();

        let stats = corpus.stats().unwrap();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.unique_systems, 2);
    }

    #[test]
    fn test_store_with_description() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();
        let mut invariant = StateFormula::new("(>= x 0)");
        invariant.description = Some("Non-negativity invariant".to_string());

        corpus.store(&system, &property, &invariant).unwrap();

        let found = corpus.find_exact(&system, &property).unwrap().unwrap();
        assert_eq!(
            found.invariant.description,
            Some("Non-negativity invariant".to_string())
        );
    }

    #[test]
    fn test_system_hash_transition_matters() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let system1 = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let system2 = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 2))") // Different increment
            .build();

        let hash1 = corpus.hash_system(&system1);
        let hash2 = corpus.hash_system(&system2);

        assert_ne!(
            hash1, hash2,
            "Different transitions should produce different hashes"
        );
    }

    #[test]
    fn test_system_hash_variable_order_matters() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let system1 = TransitionSystemBuilder::new()
            .variable("a", SmtType::Int)
            .variable("b", SmtType::Bool)
            .init("true")
            .transition("true")
            .build();

        let system2 = TransitionSystemBuilder::new()
            .variable("b", SmtType::Bool)
            .variable("a", SmtType::Int)
            .init("true")
            .transition("true")
            .build();

        let hash1 = corpus.hash_system(&system1);
        let hash2 = corpus.hash_system(&system2);

        assert_ne!(
            hash1, hash2,
            "Different variable order should produce different hashes"
        );
    }

    #[test]
    fn test_encode_variable_types_multiple() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = TransitionSystemBuilder::new()
            .variable("i", SmtType::Int)
            .variable("r", SmtType::Real)
            .variable("b", SmtType::Bool)
            .variable("bv16", SmtType::BitVec(16))
            .init("true")
            .transition("true")
            .build();

        let encoded = corpus.encode_variable_types(&system);
        assert_eq!(encoded, "Int,Real,Bool,(_ BitVec 16)");
    }

    #[test]
    fn test_find_similar_prefers_higher_success_count() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property1 = Property::safety("p1", "test1", StateFormula::new("(> x 0)"));
        let property2 = Property::safety("p2", "test2", StateFormula::new("(< x 100)"));

        let inv1 = StateFormula::new("(>= x 0)");
        let inv2 = StateFormula::new("(and (>= x 0) (< x 100))");

        // Store inv1 multiple times to increase success count
        corpus.store(&system, &property1, &inv1).unwrap();
        corpus.store(&system, &property1, &inv1).unwrap();
        corpus.store(&system, &property1, &inv1).unwrap();

        // Store inv2 once
        corpus.store(&system, &property2, &inv2).unwrap();

        // Create a new system with same variable types
        let new_system = TransitionSystemBuilder::new()
            .variable("y", SmtType::Int)
            .init("(= y 50)")
            .transition("(= y' (+ y 1))")
            .build();
        let new_property = Property::safety("p3", "test3", StateFormula::new("(>= y 0)"));

        // find_similar should prefer the one with higher success count
        let found = corpus.find_similar(&new_system, &new_property).unwrap();
        assert!(found.is_some());
        let entry = found.unwrap();
        assert_eq!(entry.success_count, 3);
    }

    #[test]
    fn test_corpus_config_clone() {
        let config = CorpusConfig {
            db_path: PathBuf::from("/test/path"),
            max_entries: 1000,
            similarity_threshold: 0.8,
        };

        let cloned = config.clone();
        assert_eq!(cloned.db_path, config.db_path);
        assert_eq!(cloned.max_entries, config.max_entries);
    }

    #[test]
    fn test_corpus_config_debug() {
        let config = CorpusConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CorpusConfig"));
        assert!(debug_str.contains("max_entries"));
    }

    #[test]
    fn test_store_returns_same_id_for_duplicate() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");

        let id1 = corpus.store(&system, &property, &invariant).unwrap();
        let id2 = corpus.store(&system, &property, &invariant).unwrap();

        // Same ID should be returned for duplicate
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_store_different_invariants_different_ids() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();

        let inv1 = StateFormula::new("(>= x 0)");
        let inv2 = StateFormula::new("(> x -1)");

        let id1 = corpus.store(&system, &property, &inv1).unwrap();
        let id2 = corpus.store(&system, &property, &inv2).unwrap();

        // Different invariants should have different IDs
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_last_used_updates_on_duplicate_store() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");

        corpus.store(&system, &property, &invariant).unwrap();

        // Small delay to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(10));

        let entry_before = corpus.find_exact(&system, &property).unwrap().unwrap();

        corpus.store(&system, &property, &invariant).unwrap();

        let entry_after = corpus.find_exact(&system, &property).unwrap().unwrap();

        // last_used should be updated (or same if within same second)
        assert!(entry_after.last_used >= entry_before.last_used);
    }

    #[test]
    fn test_invariant_entry_variable_types() {
        let entry = InvariantEntry {
            id: 1,
            invariant: StateFormula::new("true"),
            system_hash: "hash".to_string(),
            variable_types: vec![SmtType::Int, SmtType::Bool, SmtType::BitVec(32)],
            property_hash: "prop".to_string(),
            success_count: 1,
            last_used: 0,
        };

        assert_eq!(entry.variable_types.len(), 3);
        assert_eq!(entry.variable_types[0], SmtType::Int);
        assert_eq!(entry.variable_types[1], SmtType::Bool);
        assert_eq!(entry.variable_types[2], SmtType::BitVec(32));
    }

    #[test]
    fn test_empty_corpus_stats() {
        let corpus = InvariantCorpus::open_in_memory().unwrap();
        let stats = corpus.stats().unwrap();

        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_uses, 0);
        assert_eq!(stats.unique_systems, 0);
    }

    #[test]
    fn test_find_exact_returns_highest_success_count() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        let system = test_system();
        let property = test_property();

        // Store same system/property with different invariants
        let inv1 = StateFormula::new("(>= x 0)");
        let inv2 = StateFormula::new("(> x -1)");

        // Store inv1 5 times
        for _ in 0..5 {
            corpus.store(&system, &property, &inv1).unwrap();
        }

        // Store inv2 2 times
        for _ in 0..2 {
            corpus.store(&system, &property, &inv2).unwrap();
        }

        let found = corpus.find_exact(&system, &property).unwrap().unwrap();
        assert_eq!(found.success_count, 5);
        assert_eq!(found.invariant.smt_formula, "(>= x 0)");
    }

    // =============================================================================
    // Corpus Seeding Tests
    // =============================================================================

    #[test]
    fn test_seed_common_patterns() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // Initially not seeded
        assert!(!corpus.is_seeded().unwrap());

        // Seed the corpus
        let count = corpus.seed_common_patterns().unwrap();
        assert!(count > 0, "Should seed some patterns");

        // Now it should be seeded
        assert!(corpus.is_seeded().unwrap());

        // Stats should reflect seeded entries
        let stats = corpus.stats().unwrap();
        assert_eq!(stats.total_entries, count);
    }

    #[test]
    fn test_seed_idempotent() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // First seed
        let count1 = corpus.seed_common_patterns().unwrap();
        assert!(count1 > 0);

        // Second seed should add nothing (idempotent)
        let count2 = corpus.seed_common_patterns().unwrap();
        assert_eq!(count2, 0, "Seeding again should not add duplicates");

        // Total entries should still be count1
        let stats = corpus.stats().unwrap();
        assert_eq!(stats.total_entries, count1);
    }

    #[test]
    fn test_seeded_patterns_retrieval() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        corpus.seed_common_patterns().unwrap();

        let patterns = corpus.seeded_patterns().unwrap();
        assert!(!patterns.is_empty());

        // All seeded patterns should have seed_ prefix in system_hash
        for pattern in &patterns {
            assert!(
                pattern.system_hash.starts_with("seed_"),
                "Expected seed_ prefix, got: {}",
                pattern.system_hash
            );
        }

        // Should be ordered by success count (descending)
        for window in patterns.windows(2) {
            assert!(
                window[0].success_count >= window[1].success_count,
                "Patterns should be ordered by success_count desc"
            );
        }
    }

    #[test]
    fn test_seeded_patterns_have_descriptions() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        corpus.seed_common_patterns().unwrap();

        let patterns = corpus.seeded_patterns().unwrap();

        // All seeded patterns should have descriptions
        for pattern in &patterns {
            assert!(
                pattern.invariant.description.is_some(),
                "Seeded pattern should have description: {}",
                pattern.invariant.smt_formula
            );
        }
    }

    #[test]
    fn test_seeded_pattern_categories() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        corpus.seed_common_patterns().unwrap();

        let patterns = corpus.seeded_patterns().unwrap();

        // Check for key pattern categories
        let formulas: Vec<&str> = patterns
            .iter()
            .map(|p| p.invariant.smt_formula.as_str())
            .collect();

        // Non-negativity patterns
        assert!(
            formulas.iter().any(|f| f.contains("(>= x 0)")),
            "Should have non-negativity pattern"
        );
        assert!(
            formulas.iter().any(|f| f.contains("(>= i 0)")),
            "Should have index non-negativity"
        );

        // Bound patterns
        assert!(
            formulas.iter().any(|f| f.contains("(<= x n)")),
            "Should have upper bound pattern"
        );
        assert!(
            formulas.iter().any(|f| f.contains("(< i len)")),
            "Should have index bound pattern"
        );

        // Range patterns
        assert!(
            formulas
                .iter()
                .any(|f| f.contains("(and (>= x 0) (<= x n))")),
            "Should have range pattern"
        );

        // Bitvector patterns
        assert!(
            formulas
                .iter()
                .any(|f| f.contains("bvuge") || f.contains("bvult")),
            "Should have bitvector patterns"
        );
    }

    #[test]
    fn test_seed_pattern_success_counts() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        corpus.seed_common_patterns().unwrap();

        let patterns = corpus.seeded_patterns().unwrap();

        // Non-negativity should have highest success count (100)
        let nonneg_pattern = patterns
            .iter()
            .find(|p| p.invariant.smt_formula == "(>= x 0)")
            .expect("Should have (>= x 0) pattern");
        assert_eq!(nonneg_pattern.success_count, 100);

        // Valid index should have high success count (95)
        let valid_idx = patterns
            .iter()
            .find(|p| p.invariant.smt_formula == "(and (>= i 0) (< i len))")
            .expect("Should have valid index pattern");
        assert_eq!(valid_idx.success_count, 95);
    }

    #[test]
    fn test_common_invariant_patterns_count() {
        let patterns = InvariantCorpus::common_invariant_patterns();

        // We defined 20 patterns
        assert_eq!(patterns.len(), 20, "Should have 20 seed patterns");
    }

    #[test]
    fn test_seeded_patterns_dont_conflict_with_user_entries() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // Add a user entry first
        let system = test_system();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");
        corpus.store(&system, &property, &invariant).unwrap();

        // Now seed
        corpus.seed_common_patterns().unwrap();

        // User entry should still exist with its own system_hash
        let found = corpus.find_exact(&system, &property).unwrap();
        assert!(found.is_some());
        let entry = found.unwrap();
        assert!(
            !entry.system_hash.starts_with("seed_"),
            "User entry should not have seed_ prefix"
        );
    }

    #[test]
    fn test_is_seeded_false_with_only_user_entries() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // Add only user entries
        let system = test_system();
        let property = test_property();
        let invariant = StateFormula::new("(>= x 0)");
        corpus.store(&system, &property, &invariant).unwrap();

        // Should not be considered seeded
        assert!(!corpus.is_seeded().unwrap());
    }

    #[test]
    fn test_seed_pattern_variable_types() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        corpus.seed_common_patterns().unwrap();

        let patterns = corpus.seeded_patterns().unwrap();

        // Check various variable type combinations exist
        let has_single_int = patterns
            .iter()
            .any(|p| p.variable_types == vec![SmtType::Int]);
        let has_two_ints = patterns
            .iter()
            .any(|p| p.variable_types == vec![SmtType::Int, SmtType::Int]);
        let has_three_ints = patterns
            .iter()
            .any(|p| p.variable_types == vec![SmtType::Int, SmtType::Int, SmtType::Int]);
        let has_bitvec = patterns
            .iter()
            .any(|p| p.variable_types.contains(&SmtType::BitVec(32)));
        let has_bool = patterns
            .iter()
            .any(|p| p.variable_types.contains(&SmtType::Bool));

        assert!(has_single_int, "Should have single Int patterns");
        assert!(has_two_ints, "Should have two Int patterns");
        assert!(has_three_ints, "Should have three Int patterns");
        assert!(has_bitvec, "Should have BitVec patterns");
        assert!(has_bool, "Should have Bool patterns");
    }

    #[test]
    fn test_find_similar_can_match_seeded_patterns() {
        let mut corpus = InvariantCorpus::open_in_memory().unwrap();

        // Seed first
        corpus.seed_common_patterns().unwrap();

        // Create a system with Int variable type
        let system = TransitionSystemBuilder::new()
            .variable("counter", SmtType::Int)
            .init("(= counter 0)")
            .transition("(= counter' (+ counter 1))")
            .build();

        let property = Property::safety("p1", "nonneg", StateFormula::new("(>= counter 0)"));

        // find_similar should find a seeded pattern with matching variable types
        let found = corpus.find_similar(&system, &property).unwrap();
        assert!(found.is_some(), "Should find a similar seeded pattern");

        let entry = found.unwrap();
        assert_eq!(entry.variable_types, vec![SmtType::Int]);
    }

    // =============================================================================
    // Mutation Coverage Tests (targeting specific mutants)
    // =============================================================================

    #[test]
    fn test_prune_old_entries_when_over_limit() {
        // Create corpus with very low max_entries limit
        let conn = Connection::open_in_memory().unwrap();
        let config = CorpusConfig {
            db_path: PathBuf::from(":memory:"),
            max_entries: 3,
            similarity_threshold: 0.7,
        };
        let corpus = InvariantCorpus { conn, config };
        corpus.init_schema().unwrap();

        // Insert 5 entries directly via SQL to control success_count and last_used
        for i in 1..=5 {
            corpus.conn.execute(
                "INSERT INTO invariants (invariant_formula, system_hash, variable_types, property_hash, success_count, last_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![
                    format!("(>= x {})", i),
                    format!("hash_{}", i),
                    "Int",
                    "prop",
                    i, // success_count: 1, 2, 3, 4, 5
                    i as i64, // last_used: 1, 2, 3, 4, 5
                    0i64
                ],
            ).unwrap();
        }

        // Verify 5 entries exist
        let count_before: i64 = corpus
            .conn
            .query_row("SELECT COUNT(*) FROM invariants", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count_before, 5);

        // Prune should remove 2 entries (5 - 3 = 2)
        corpus.prune_old_entries().unwrap();

        // Should have 3 entries remaining
        let count_after: i64 = corpus
            .conn
            .query_row("SELECT COUNT(*) FROM invariants", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count_after, 3, "Should have pruned down to max_entries");

        // The remaining entries should be the ones with highest success_count (3, 4, 5)
        let remaining: Vec<i32> = corpus
            .conn
            .prepare("SELECT success_count FROM invariants ORDER BY success_count")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(
            remaining,
            vec![3, 4, 5],
            "Should keep entries with highest success_count"
        );
    }

    #[test]
    fn test_prune_old_entries_under_limit_does_nothing() {
        let conn = Connection::open_in_memory().unwrap();
        let config = CorpusConfig {
            db_path: PathBuf::from(":memory:"),
            max_entries: 10,
            similarity_threshold: 0.7,
        };
        let corpus = InvariantCorpus { conn, config };
        corpus.init_schema().unwrap();

        // Insert only 3 entries (under limit of 10)
        for i in 1..=3 {
            corpus.conn.execute(
                "INSERT INTO invariants (invariant_formula, system_hash, variable_types, property_hash, success_count, last_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![
                    format!("(>= x {})", i),
                    format!("hash_{}", i),
                    "Int",
                    "prop",
                    1i32,
                    0i64,
                    0i64
                ],
            ).unwrap();
        }

        // Prune should do nothing
        corpus.prune_old_entries().unwrap();

        // Should still have 3 entries
        let count: i64 = corpus
            .conn
            .query_row("SELECT COUNT(*) FROM invariants", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 3, "Should not prune when under limit");
    }

    #[test]
    fn test_prune_boundary_condition_exactly_at_limit() {
        let conn = Connection::open_in_memory().unwrap();
        let config = CorpusConfig {
            db_path: PathBuf::from(":memory:"),
            max_entries: 3,
            similarity_threshold: 0.7,
        };
        let corpus = InvariantCorpus { conn, config };
        corpus.init_schema().unwrap();

        // Insert exactly 3 entries (at limit)
        for i in 1..=3 {
            corpus.conn.execute(
                "INSERT INTO invariants (invariant_formula, system_hash, variable_types, property_hash, success_count, last_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![
                    format!("(>= x {})", i),
                    format!("hash_{}", i),
                    "Int",
                    "prop",
                    1i32,
                    0i64,
                    0i64
                ],
            ).unwrap();
        }

        // Prune should do nothing when exactly at limit
        corpus.prune_old_entries().unwrap();

        let count: i64 = corpus
            .conn
            .query_row("SELECT COUNT(*) FROM invariants", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 3, "Should not prune when exactly at limit");
    }

    #[test]
    fn test_decode_variable_types_int_explicit() {
        // This test ensures the Int match arm is covered
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let decoded = corpus.decode_variable_types("Int");
        assert_eq!(decoded.len(), 1);
        assert!(
            matches!(decoded[0], SmtType::Int),
            "Must decode 'Int' to SmtType::Int"
        );

        // Also test in combination
        let decoded_multi = corpus.decode_variable_types("Int,Int,Int");
        assert_eq!(decoded_multi.len(), 3);
        for t in &decoded_multi {
            assert!(matches!(t, SmtType::Int), "All should be Int");
        }
    }

    #[test]
    fn test_decode_variable_types_bool_explicit() {
        // This test ensures the Bool match arm is covered
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        let decoded = corpus.decode_variable_types("Bool");
        assert_eq!(decoded.len(), 1);
        assert!(
            matches!(decoded[0], SmtType::Bool),
            "Must decode 'Bool' to SmtType::Bool"
        );
    }

    #[test]
    fn test_open_in_memory_sets_db_path() {
        // This test ensures the db_path field is set in open_in_memory
        let corpus = InvariantCorpus::open_in_memory().unwrap();

        // The config should have a db_path set (even if it's :memory:)
        assert_eq!(corpus.config.db_path, PathBuf::from(":memory:"));
    }
}
