//! Load parsed .olean modules into the Lean5 kernel environment.
//!
//! This bridges the low-level parsed structures into the kernel's
//! `Environment` by converting names, levels, and expressions.
//!
//! # Caching
//!
//! Module loading involves two expensive operations:
//! 1. **Parsing**: Reading and interpreting the .olean binary format (~40% of time)
//! 2. **Loading**: Converting and registering constants in the environment (~60% of time)
//!
//! The `ModuleCache` can be used to cache parsed modules across multiple
//! `load_module_with_deps` calls, avoiding re-parsing when the same module
//! is needed by multiple dependents.

use crate::error::OleanResult;
use crate::expr::{ParsedBinderInfo, ParsedExpr, ParsedLiteral};
use crate::level::ParsedLevel;
use crate::module::{ConstantKind, ParsedConstant, ParsedModule};
use crate::payload::Lean5Payload;
use crate::region::CompactedRegion;
use crate::{parse_header, OleanError};
use ahash::RandomState;
use hashbrown::{HashMap, HashSet};
use lean5_kernel::env::{ConstantInfo, Declaration, EnvError, Environment};
use lean5_kernel::expr::{BinderInfo, Expr, FVarId, LevelVec, Literal};
use lean5_kernel::inductive::{
    ConstructorVal, InductiveVal, RecursorArgOrder, RecursorRule, RecursorVal,
};
use lean5_kernel::level::Level;
use lean5_kernel::name::Name;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::env;
use std::hash::{BuildHasher, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Errors that can arise while importing an .olean module into the kernel environment.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("parse error: {0}")]
    Parse(#[from] OleanError),
    #[error("environment error: {0}")]
    Env(#[from] EnvError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("module {module} not found; searched paths: {searched:?}")]
    ModuleNotFound {
        module: String,
        searched: Vec<std::path::PathBuf>,
    },
    #[error("missing type for constant {0}")]
    MissingType(String),
    #[error("missing value for constant {0}")]
    MissingValue(String),
    #[error("unsupported metavariable in constant {0}")]
    UnsupportedMVar(String),
    #[error("expression conversion failed for {name}: {message}")]
    ExprConversion { name: String, message: String },
    #[error("level conversion failed for {name}: {message}")]
    LevelConversion { name: String, message: String },
}

/// Reason a constant was skipped during import.
#[derive(Debug, Clone)]
pub struct SkippedConstant {
    pub name: String,
    pub reason: String,
}

/// Summary of an import attempt.
#[derive(Debug, Clone)]
pub struct LoadSummary {
    /// Optional module name (deduced from path)
    pub module_name: Option<String>,
    /// Names of imported modules (as strings)
    pub imports: Vec<String>,
    /// Number of successfully added constants
    pub added_constants: usize,
    /// Constants skipped due to conversion issues
    pub skipped_constants: Vec<SkippedConstant>,
    /// Constants ignored because they already exist in the environment
    pub duplicate_constants: usize,
}

impl LoadSummary {
    fn empty() -> Self {
        Self {
            module_name: None,
            imports: Vec::new(),
            added_constants: 0,
            skipped_constants: Vec::new(),
            duplicate_constants: 0,
        }
    }
}

/// Cache entry for a parsed module
#[derive(Clone)]
struct CacheEntry {
    /// The parsed module data (Arc to avoid expensive clones)
    module: Arc<ParsedModule>,
    /// File modification time when cached
    mtime: Option<std::time::SystemTime>,
}

/// Cache for parsed .olean modules.
///
/// This cache stores parsed modules to avoid re-parsing when the same
/// module is needed multiple times. It's particularly useful when loading
/// multiple modules that share dependencies.
///
/// # Example
///
/// ```ignore
/// use lean5_olean::{ModuleCache, load_module_with_deps_cached, default_search_paths};
/// use lean5_kernel::env::Environment;
///
/// let cache = ModuleCache::new();
/// let paths = default_search_paths();
///
/// // First load - parses all modules
/// let mut env1 = Environment::default();
/// load_module_with_deps_cached(&mut env1, "Init.Core", &paths, &cache)?;
///
/// // Second load - reuses cached modules
/// let mut env2 = Environment::default();
/// load_module_with_deps_cached(&mut env2, "Init.Data.List.Basic", &paths, &cache)?;
/// ```
#[derive(Default)]
pub struct ModuleCache {
    /// Map from module name to cached entry
    entries: RwLock<HashMap<String, CacheEntry>>,
}

impl ModuleCache {
    /// Create a new empty module cache.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Get the number of cached modules.
    pub fn len(&self) -> usize {
        self.entries
            .read()
            .expect("ModuleCache lock poisoned")
            .len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries
            .read()
            .expect("ModuleCache lock poisoned")
            .is_empty()
    }

    /// Clear all cached modules.
    pub fn clear(&self) {
        self.entries
            .write()
            .expect("ModuleCache lock poisoned")
            .clear();
    }

    /// Get cached module if available and not stale.
    /// Returns Arc to avoid expensive clones of ParsedModule.
    fn get(&self, module: &str, path: &Path) -> Option<Arc<ParsedModule>> {
        let mtime = std::fs::metadata(path)
            .ok()
            .and_then(|metadata| metadata.modified().ok());
        let mut entries = self.entries.write().expect("ModuleCache lock poisoned");

        match entries.get(module) {
            Some(entry) if entry.mtime.is_some() && entry.mtime == mtime => {
                Some(Arc::clone(&entry.module))
            }
            Some(_) => {
                // Drop stale entries so follow-up loads don't keep retrying outdated data.
                entries.remove(module);
                None
            }
            None => None,
        }
    }

    /// Insert a module into the cache.
    /// Returns Arc to the inserted module to avoid re-cloning.
    fn insert(&self, module: &str, path: &Path, parsed: ParsedModule) -> Arc<ParsedModule> {
        let mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
        let arc_module = Arc::new(parsed);

        let entry = CacheEntry {
            module: Arc::clone(&arc_module),
            mtime,
        };

        self.entries
            .write()
            .expect("ModuleCache lock poisoned")
            .insert(module.to_string(), entry);

        arc_module
    }
}

fn module_name_to_rel_path(module: &str) -> Option<PathBuf> {
    let trimmed = module.trim_matches('.');
    if trimmed.is_empty() {
        return None;
    }

    let mut path = PathBuf::new();
    for part in trimmed.split('.') {
        if part.is_empty() {
            return None;
        }
        path.push(part);
    }
    path.set_extension("olean");
    Some(path)
}

fn collect_default_search_paths<F, R>(mut var_lookup: F, mut read_dir: R) -> Vec<PathBuf>
where
    F: for<'a> FnMut(&'a str) -> Option<std::ffi::OsString>,
    R: for<'a> FnMut(&'a Path) -> std::io::Result<std::fs::ReadDir>,
{
    let mut paths = Vec::new();
    let mut seen = HashSet::new();

    if let Some(val) = var_lookup("LEAN_PATH") {
        for path in env::split_paths(&val) {
            if path.exists() && seen.insert(path.clone()) {
                paths.push(path);
            }
        }
    }

    for var in ["HOME", "USERPROFILE"] {
        let Some(home) = var_lookup(var) else {
            continue;
        };

        let elan_path = PathBuf::from(home).join(".elan/toolchains");
        let Ok(entries) = read_dir(&elan_path) else {
            continue;
        };

        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().contains("lean4") {
                let lib_path = entry.path().join("lib/lean");
                if lib_path.exists() && seen.insert(lib_path.clone()) {
                    paths.push(lib_path);
                }
            }
        }
    }

    paths
}

/// Discover likely search paths for Lean .olean files.
///
/// Priority order:
/// 1. `LEAN_PATH` environment variable entries (first match wins)
/// 2. Lean4 toolchains under `.elan/toolchains/*/lib/lean` using `HOME` or
///    `USERPROFILE` as the base directory
pub fn default_search_paths() -> Vec<PathBuf> {
    collect_default_search_paths(
        |key: &str| env::var_os(key),
        |path: &Path| std::fs::read_dir(path),
    )
}

fn resolve_module_path(module: &str, search_paths: &[PathBuf]) -> Result<PathBuf, ImportError> {
    let rel = module_name_to_rel_path(module).ok_or_else(|| ImportError::ModuleNotFound {
        module: module.to_string(),
        searched: Vec::new(),
    })?;

    let mut tried = Vec::new();
    for base in search_paths {
        let candidate = base.join(&rel);
        tried.push(candidate.clone());
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(ImportError::ModuleNotFound {
        module: module.to_string(),
        searched: tried,
    })
}

/// Parse a complete .olean file into a `ParsedModule`.
pub fn parse_module(bytes: &[u8]) -> OleanResult<ParsedModule> {
    let header = parse_header(bytes)?;
    let region = CompactedRegion::new(bytes, header.base_addr);
    region.read_module_data()
}

/// Parse a module from disk.
pub fn parse_module_file(path: impl AsRef<Path>) -> OleanResult<ParsedModule> {
    let bytes = std::fs::read(path)?;
    parse_module(&bytes)
}

/// Load an .olean file from disk into the given environment.
pub fn load_olean_file(
    env: &mut Environment,
    path: impl AsRef<Path>,
) -> Result<LoadSummary, ImportError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let module_name = module_name_from_path(path);
    let module = parse_module(&bytes)?;
    load_parsed_module(env, module, module_name)
}

/// Load a module by name along with all of its imports (depth-first).
///
/// `module` should be a dot-separated Lean module name like `Init.Core`.
/// `search_paths` should contain directories that hold `.olean` files, typically
/// something like `~/.elan/toolchains/<toolchain>/lib/lean`.
pub fn load_module_with_deps(
    env: &mut Environment,
    module: &str,
    search_paths: &[PathBuf],
) -> Result<Vec<LoadSummary>, ImportError> {
    let mut visited = HashSet::new();
    let mut summaries = Vec::new();
    load_module_recursive(env, module, search_paths, &mut visited, &mut summaries)?;
    Ok(summaries)
}

fn load_module_recursive(
    env: &mut Environment,
    module: &str,
    search_paths: &[PathBuf],
    visited: &mut HashSet<String>,
    summaries: &mut Vec<LoadSummary>,
) -> Result<(), ImportError> {
    if !visited.insert(module.to_string()) {
        return Ok(());
    }

    let path = resolve_module_path(module, search_paths)?;
    let bytes = std::fs::read(&path)?;
    let parsed = parse_module(&bytes)?;

    for import in &parsed.imports {
        if !import.module_name.is_empty() {
            load_module_recursive(env, &import.module_name, search_paths, visited, summaries)?;
        }
    }

    let summary = load_parsed_module(env, parsed, Some(module.to_string()))?;
    summaries.push(summary);

    Ok(())
}

/// Load a module by name along with all of its imports, using a cache.
///
/// Like `load_module_with_deps` but uses the provided cache to avoid
/// re-parsing modules. The cache persists across calls and automatically
/// invalidates entries when file modification times change.
///
/// # Example
///
/// ```ignore
/// let cache = ModuleCache::new();
///
/// // First load - parses all modules
/// let mut env1 = Environment::default();
/// load_module_with_deps_cached(&mut env1, "Init.Core", &paths, &cache)?;
///
/// // Second load - reuses cached modules for shared dependencies
/// let mut env2 = Environment::default();
/// load_module_with_deps_cached(&mut env2, "Init.Data.List.Basic", &paths, &cache)?;
/// ```
pub fn load_module_with_deps_cached(
    env: &mut Environment,
    module: &str,
    search_paths: &[PathBuf],
    cache: &ModuleCache,
) -> Result<Vec<LoadSummary>, ImportError> {
    let mut visited = HashSet::new();
    let mut summaries = Vec::new();
    load_module_recursive_cached(
        env,
        module,
        search_paths,
        &mut visited,
        &mut summaries,
        cache,
    )?;
    Ok(summaries)
}

fn load_module_recursive_cached(
    env: &mut Environment,
    module: &str,
    search_paths: &[PathBuf],
    visited: &mut HashSet<String>,
    summaries: &mut Vec<LoadSummary>,
    cache: &ModuleCache,
) -> Result<(), ImportError> {
    if !visited.insert(module.to_string()) {
        return Ok(());
    }

    let path = resolve_module_path(module, search_paths)?;

    // Try cache first - now returns Arc<ParsedModule> to avoid expensive clones
    let parsed: Arc<ParsedModule> = if let Some(cached) = cache.get(module, &path) {
        cached
    } else {
        // Parse and cache - insert returns Arc to avoid clone
        let bytes = std::fs::read(&path)?;
        let parsed = parse_module(&bytes)?;
        cache.insert(module, &path, parsed)
    };

    for import in &parsed.imports {
        if !import.module_name.is_empty() {
            load_module_recursive_cached(
                env,
                &import.module_name,
                search_paths,
                visited,
                summaries,
                cache,
            )?;
        }
    }

    // Unwrap Arc if we're the only reference, otherwise clone
    // (Cache still holds a reference, so this will clone, but we saved on cache operations)
    let owned_module = Arc::try_unwrap(parsed).unwrap_or_else(|arc| (*arc).clone());
    let summary = load_parsed_module(env, owned_module, Some(module.to_string()))?;
    summaries.push(summary);

    Ok(())
}

/// Load a module and all its dependencies with parallel file I/O and parsing.
///
/// This is an optimized version that:
/// 1. Reads all .olean files in parallel using rayon
/// 2. Parses all files in parallel
/// 3. Loads modules into environment in topological order
///
/// This is faster than sequential loading for large module graphs by parallelizing
/// the expensive parsing phase.
pub fn load_module_with_deps_parallel(
    env: &mut Environment,
    module: &str,
    search_paths: &[PathBuf],
    cache: &ModuleCache,
) -> Result<Vec<LoadSummary>, ImportError> {
    // Phase 1: Discover all modules and collect their paths
    // We read files only once during discovery and keep the bytes for parsing
    let mut to_discover: Vec<String> = vec![module.to_string()];
    let mut discovered: HashSet<String> = HashSet::new();
    let mut module_bytes: Vec<(String, PathBuf, Vec<u8>)> = Vec::new();

    while let Some(mod_name) = to_discover.pop() {
        if !discovered.insert(mod_name.clone()) {
            continue;
        }

        let path = resolve_module_path(&mod_name, search_paths)?;

        // Check if already in cache
        if let Some(cached) = cache.get(&mod_name, &path) {
            // Get dependencies from cached module
            for import in &cached.imports {
                if !import.module_name.is_empty() && !discovered.contains(&import.module_name) {
                    to_discover.push(import.module_name.clone());
                }
            }
            continue;
        }

        // Read file and extract imports for discovery (keep bytes for later parsing)
        let bytes = std::fs::read(&path)?;
        let imports = crate::parse_imports_only(&bytes)?;

        // Store bytes for later parallel parsing
        module_bytes.push((mod_name.clone(), path, bytes));

        for import in imports {
            if !import.module_name.is_empty() && !discovered.contains(&import.module_name) {
                to_discover.push(import.module_name.clone());
            }
        }
    }

    // Phase 2: Parse all modules in parallel (reusing already-read bytes)
    let parsed_modules: Vec<Result<(String, PathBuf, ParsedModule), ImportError>> = module_bytes
        .into_par_iter()
        .map(|(name, path, bytes)| {
            let parsed = parse_module(&bytes)?;
            Ok((name, path, parsed))
        })
        .collect();

    // Insert parsed modules into cache
    for result in parsed_modules {
        let (name, path, parsed) = result?;
        cache.insert(&name, &path, parsed);
    }

    // Phase 3: Load modules in topological order (post-order DFS)
    let mut visited = HashSet::new();
    let mut summaries = Vec::new();

    fn load_in_order(
        env: &mut Environment,
        module: &str,
        search_paths: &[PathBuf],
        visited: &mut HashSet<String>,
        summaries: &mut Vec<LoadSummary>,
        cache: &ModuleCache,
    ) -> Result<(), ImportError> {
        if !visited.insert(module.to_string()) {
            return Ok(());
        }

        let path = resolve_module_path(module, search_paths)?;
        let parsed = cache
            .get(module, &path)
            .expect("module should be in cache after parallel parse");

        // Load dependencies first
        for import in &parsed.imports {
            if !import.module_name.is_empty() {
                load_in_order(
                    env,
                    &import.module_name,
                    search_paths,
                    visited,
                    summaries,
                    cache,
                )?;
            }
        }

        // Load this module
        let owned_module = Arc::try_unwrap(parsed).unwrap_or_else(|arc| (*arc).clone());
        let summary = load_parsed_module(env, owned_module, Some(module.to_string()))?;
        summaries.push(summary);

        Ok(())
    }

    load_in_order(
        env,
        module,
        search_paths,
        &mut visited,
        &mut summaries,
        cache,
    )?;

    Ok(summaries)
}

/// Convert Declaration to ConstantInfo without redundant allocation
#[inline]
fn decl_to_constant_info(decl: Declaration) -> lean5_kernel::env::ConstantInfo {
    match decl {
        Declaration::Definition {
            name,
            level_params,
            type_,
            value,
            is_reducible,
        } => lean5_kernel::env::ConstantInfo {
            name,
            level_params,
            type_,
            value: Some(value),
            is_reducible,
        },
        Declaration::Axiom {
            name,
            level_params,
            type_,
        } => lean5_kernel::env::ConstantInfo {
            name,
            level_params,
            type_,
            value: None,
            is_reducible: false,
        },
        Declaration::Theorem {
            name,
            level_params,
            type_,
            value,
        }
        | Declaration::Opaque {
            name,
            level_params,
            type_,
            value,
        } => lean5_kernel::env::ConstantInfo {
            name,
            level_params,
            type_,
            value: Some(value),
            is_reducible: false,
        },
    }
}

/// Converted constant with its original name for error reporting.
enum ConvertedConstant {
    Inductive(String, Result<InductiveVal, ImportError>),
    Constructor(String, Result<ConstructorVal, ImportError>),
    Recursor(String, Result<(RecursorVal, Vec<Name>, u32), ImportError>),
    Other(String, Result<Declaration, ImportError>),
}

/// Merge a Lean5 payload (serialized kernel objects) into the environment.
fn load_lean5_payload(env: &mut Environment, payload: &Lean5Payload) -> (usize, usize) {
    let mut added = 0usize;
    let mut duplicates = 0usize;

    let mut inductives: Vec<InductiveVal> = payload
        .inductives
        .iter()
        .filter(|ind| {
            if env.get_inductive(&ind.name).is_some() {
                duplicates += 1;
                false
            } else {
                true
            }
        })
        .cloned()
        .collect();
    added += inductives.len();
    env.extend_inductives_unchecked(inductives.drain(..));

    let mut constructors: Vec<ConstructorVal> = payload
        .constructors
        .iter()
        .filter(|ctor| {
            if env.get_constructor(&ctor.name).is_some() {
                duplicates += 1;
                false
            } else {
                true
            }
        })
        .cloned()
        .collect();
    added += constructors.len();
    env.extend_constructors_unchecked(constructors.drain(..));

    let mut recursors: Vec<RecursorVal> = payload
        .recursors
        .iter()
        .filter(|rec| {
            if env.get_recursor(&rec.name).is_some() {
                duplicates += 1;
                false
            } else {
                true
            }
        })
        .cloned()
        .collect();
    added += recursors.len();
    env.extend_recursors_unchecked(recursors.drain(..));

    let mut constants: Vec<ConstantInfo> = payload
        .constants
        .iter()
        .filter(|c| {
            if env.get_const(&c.name).is_some() {
                duplicates += 1;
                false
            } else {
                true
            }
        })
        .cloned()
        .collect();
    added += constants.len();
    env.extend_constants_unchecked(constants.drain(..));

    for (struct_name, fields) in &payload.structure_fields {
        if env.get_structure_field_names(struct_name).is_some() {
            duplicates += 1;
            continue;
        }
        // register_structure_fields expects inductive + constructors already present
        if env
            .register_structure_fields(struct_name.clone(), fields.clone())
            .is_err()
        {
            duplicates += 1;
        }
    }

    (added, duplicates)
}

/// Load a parsed module into the environment.
///
/// Uses two-pass loading to ensure inductives are registered before recursors.
/// This is necessary because recursors need to look up inductive info to correctly
/// determine which constructor fields are recursive.
///
/// Uses parallel conversion within the module for optimal single-module performance.
pub fn load_parsed_module(
    env: &mut Environment,
    module: ParsedModule,
    module_name: Option<String>,
) -> Result<LoadSummary, ImportError> {
    let mut summary = LoadSummary {
        module_name,
        imports: module
            .imports
            .iter()
            .map(|i| i.module_name.clone())
            .collect(),
        ..LoadSummary::empty()
    };

    let payload_const_count = module
        .lean5_payload
        .as_ref()
        .map_or(0, Lean5Payload::total_constants);

    let constant_count = module.constants.len();
    // Pre-allocate environment capacity for this module's constants and payload
    env.reserve_capacity(constant_count + payload_const_count);

    // Load Lean5 payload definitions first if present.
    if let Some(payload) = module.lean5_payload.as_ref() {
        let (added, duplicates) = load_lean5_payload(env, payload);
        summary.added_constants += added;
        summary.duplicate_constants += duplicates;
    }

    let mut duplicate_filtered = 0usize;
    // Phase 1: Separate constants into categories and convert in parallel
    let constants: Vec<_> = module
        .constants
        .into_iter()
        .filter(|c| !c.name.is_empty() || c.type_.is_some())
        .filter(|c| {
            let name = Name::interned(&c.name);
            let exists = env.get_const(&name).is_some()
                || env.get_inductive(&name).is_some()
                || env.get_constructor(&name).is_some()
                || env.get_recursor(&name).is_some();
            if exists {
                duplicate_filtered += 1;
            }
            !exists
        })
        .collect();

    summary.duplicate_constants += duplicate_filtered;

    // Convert constants in parallel using rayon
    let converted: Vec<ConvertedConstant> =
        constants.par_iter().map(convert_parsed_constant).collect();

    // Phase 2: Register using bulk operations where possible

    // Separate by category, extracting Ok values for bulk registration
    let mut ok_inductives = Vec::new();
    let mut ok_constructors = Vec::new();
    let mut recursors = Vec::new(); // Recursors still need individual processing
    let mut ok_others = Vec::new();

    for converted_const in converted {
        match converted_const {
            ConvertedConstant::Inductive(name, result) => match result {
                Ok(ind_val) => ok_inductives.push(ind_val),
                Err(e) => summary.skipped_constants.push(SkippedConstant {
                    name,
                    reason: e.to_string(),
                }),
            },
            ConvertedConstant::Constructor(name, result) => match result {
                Ok(ctor_val) => ok_constructors.push(ctor_val),
                Err(e) => summary.skipped_constants.push(SkippedConstant {
                    name,
                    reason: e.to_string(),
                }),
            },
            ConvertedConstant::Recursor(name, result) => recursors.push((name, result)),
            ConvertedConstant::Other(name, result) => match result {
                Ok(decl) => ok_others.push(decl),
                Err(e) => summary.skipped_constants.push(SkippedConstant {
                    name,
                    reason: e.to_string(),
                }),
            },
        }
    }

    // Pass 1: Bulk register inductives (so recursors can look them up)
    summary.added_constants += ok_inductives.len();
    env.extend_inductives_unchecked(ok_inductives.into_iter());

    // Pass 2: Bulk register constructors
    summary.added_constants += ok_constructors.len();
    env.extend_constructors_unchecked(ok_constructors.into_iter());

    // Pass 3: Register recursors (needs env lookups for recursive fields)
    for (name, result) in recursors {
        match result {
            Ok((mut rec_val, mutual_inductives, param_count)) => {
                // Now we can look up constructor info for recursive fields
                rec_val.rules = rec_val
                    .rules
                    .into_iter()
                    .map(|mut rule| {
                        rule.recursive_fields = compute_recursive_fields_from_env(
                            env,
                            &rule.constructor_name,
                            &mutual_inductives,
                            param_count,
                            rule.num_fields,
                        );
                        rule
                    })
                    .collect();
                env.register_recursor_unchecked(rec_val);
                summary.added_constants += 1;
            }
            Err(e) => summary.skipped_constants.push(SkippedConstant {
                name,
                reason: e.to_string(),
            }),
        }
    }

    // Pass 4: Bulk register other constants
    summary.added_constants += ok_others.len();
    env.extend_constants_unchecked(ok_others.into_iter().map(decl_to_constant_info));

    Ok(summary)
}

/// Convert a parsed constant to a ConvertedConstant.
#[inline]
fn convert_parsed_constant(constant: &ParsedConstant) -> ConvertedConstant {
    let name = constant.name.clone();
    match constant.kind {
        ConstantKind::Inductive => {
            ConvertedConstant::Inductive(name, convert_inductive_val(constant))
        }
        ConstantKind::Constructor => {
            ConvertedConstant::Constructor(name, convert_constructor_val(constant))
        }
        ConstantKind::Recursor => {
            ConvertedConstant::Recursor(name, convert_recursor_val_partial(constant))
        }
        _ => ConvertedConstant::Other(name, convert_constant(constant)),
    }
}

/// Convert an inductive constant to InductiveVal
fn convert_inductive_val(constant: &ParsedConstant) -> Result<InductiveVal, ImportError> {
    let type_ = constant
        .type_
        .as_ref()
        .ok_or_else(|| ImportError::MissingType(constant.name.clone()))
        .and_then(|t| convert_expr(&constant.name, t))?;

    let level_params: Vec<Name> = constant
        .level_params
        .iter()
        .map(|s| Name::interned(s))
        .collect();
    let name = Name::interned(&constant.name);

    let ind_data = constant.inductive_val.as_ref();

    Ok(InductiveVal {
        name: name.clone(),
        level_params,
        type_,
        num_params: ind_data.map_or(0, |d| d.num_params),
        num_indices: ind_data.map_or(0, |d| d.num_indices),
        all_names: ind_data.map_or_else(
            || vec![name.clone()],
            |d| d.all.iter().map(|s| Name::interned(s)).collect(),
        ),
        constructor_names: ind_data
            .map(|d| d.ctors.iter().map(|s| Name::interned(s)).collect())
            .unwrap_or_default(),
        is_recursive: ind_data.is_some_and(|d| d.is_rec),
        is_reflexive: ind_data.is_some_and(|d| d.is_reflexive),
        is_large_elim: true,
    })
}

/// Convert a constructor constant to ConstructorVal
fn convert_constructor_val(constant: &ParsedConstant) -> Result<ConstructorVal, ImportError> {
    let type_ = constant
        .type_
        .as_ref()
        .ok_or_else(|| ImportError::MissingType(constant.name.clone()))
        .and_then(|t| convert_expr(&constant.name, t))?;

    let level_params: Vec<Name> = constant
        .level_params
        .iter()
        .map(|s| Name::interned(s))
        .collect();
    let name = Name::interned(&constant.name);

    let ctor_data = constant.constructor_val.as_ref();

    Ok(ConstructorVal {
        name: name.clone(),
        inductive_name: ctor_data.map_or_else(
            || {
                Name::interned(
                    constant
                        .name
                        .rsplit_once('.')
                        .map_or(constant.name.as_str(), |(p, _)| p),
                )
            },
            |d| Name::interned(&d.induct),
        ),
        level_params,
        type_,
        num_params: ctor_data.map_or(0, |d| d.num_params),
        num_fields: ctor_data.map_or(0, |d| d.num_fields),
        constructor_idx: ctor_data.map_or(0, |d| d.cidx),
    })
}

/// Convert a recursor constant partially - recursive fields computed later
fn convert_recursor_val_partial(
    constant: &ParsedConstant,
) -> Result<(RecursorVal, Vec<Name>, u32), ImportError> {
    let type_ = constant
        .type_
        .as_ref()
        .ok_or_else(|| ImportError::MissingType(constant.name.clone()))
        .and_then(|t| convert_expr(&constant.name, t))?;

    let level_params: Vec<Name> = constant
        .level_params
        .iter()
        .map(|s| Name::interned(s))
        .collect();
    let name = Name::interned(&constant.name);

    let inductive_name = Name::interned(
        constant
            .name
            .strip_suffix(".rec")
            .or_else(|| constant.name.strip_suffix(".recOn"))
            .or_else(|| constant.name.strip_suffix(".casesOn"))
            .or_else(|| constant.name.strip_suffix(".brecOn"))
            .unwrap_or(&constant.name),
    );

    let rec_data = constant.recursor_val.as_ref();

    let mutual_inductives: Vec<Name> = rec_data.map_or_else(
        || vec![inductive_name.clone()],
        |d| d.all.iter().map(|s| Name::interned(s)).collect(),
    );

    let param_count = rec_data.map_or(0, |d| d.num_params);

    // Convert rules with placeholder recursive_fields (will be filled in later)
    let rules: Vec<RecursorRule> = rec_data
        .map(|d| {
            d.rules
                .iter()
                .map(|r| {
                    let rhs = match r.rhs.as_ref() {
                        Some(e) => convert_expr(&constant.name, e)?,
                        None => {
                            return Err(ImportError::ExprConversion {
                                name: constant.name.clone(),
                                message: format!(
                                    "recursor rule for {} has no RHS expression",
                                    r.ctor
                                ),
                            });
                        }
                    };

                    Ok(RecursorRule {
                        constructor_name: Name::interned(&r.ctor),
                        num_fields: r.num_fields,
                        recursive_fields: vec![], // Placeholder, filled in later
                        rhs,
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let arg_order = if constant.name.ends_with(".recOn") || constant.name.ends_with(".casesOn") {
        RecursorArgOrder::MajorAfterMotive
    } else {
        RecursorArgOrder::MajorAfterMinors
    };

    let rec_val = RecursorVal {
        name: name.clone(),
        arg_order,
        level_params,
        type_,
        inductive_name,
        num_params: param_count,
        num_indices: rec_data.map_or(0, |d| d.num_indices),
        num_motives: rec_data.map_or(1, |d| d.num_motives),
        num_minors: rec_data.map_or(0, |d| d.num_minors),
        rules,
        is_k: rec_data.is_some_and(|d| d.k),
    };

    Ok((rec_val, mutual_inductives, param_count))
}

fn module_name_from_path(path: &Path) -> Option<String> {
    let mut components: Vec<String> = path
        .with_extension("")
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => {
                let s = s.to_string_lossy();
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            }
            _ => None,
        })
        .collect();

    if let Some(pos) = components
        .iter()
        .rposition(|c| c == "lean" || c == "library")
    {
        components = components.split_off(pos + 1);
    }

    if components.is_empty() {
        return None;
    }

    Some(components.join("."))
}

fn convert_binder_info(info: ParsedBinderInfo) -> BinderInfo {
    match info {
        ParsedBinderInfo::Default => BinderInfo::Default,
        ParsedBinderInfo::Implicit => BinderInfo::Implicit,
        ParsedBinderInfo::StrictImplicit => BinderInfo::StrictImplicit,
        ParsedBinderInfo::InstImplicit => BinderInfo::InstImplicit,
    }
}

fn convert_level(name: &str, level: &ParsedLevel) -> Result<Level, ImportError> {
    match level {
        ParsedLevel::Zero => Ok(Level::zero()),
        ParsedLevel::Succ(l) => Ok(Level::succ(convert_level(name, l)?)),
        ParsedLevel::Max(l, r) => Ok(Level::max(convert_level(name, l)?, convert_level(name, r)?)),
        ParsedLevel::IMax(l, r) => Ok(Level::imax(
            convert_level(name, l)?,
            convert_level(name, r)?,
        )),
        // Level parameters often reuse names like "u", "v" across many constants
        ParsedLevel::Param(n) => Ok(Level::param(Name::interned(n))),
        ParsedLevel::MVar(_) => Err(ImportError::UnsupportedMVar(name.to_string())),
    }
}

fn convert_level_params(params: &[String]) -> Vec<Name> {
    params
        .iter()
        .map(|p| {
            if p.is_empty() {
                Name::anon()
            } else {
                Name::interned(p)
            }
        })
        .collect()
}

/// Work item for iterative expression conversion (avoids stack overflow on deep trees).
enum ConvertWork<'a> {
    /// Process this expression and push children
    Process(&'a ParsedExpr),
    /// Build App from top 2 results
    BuildApp,
    /// Build Lam from top 2 results
    BuildLam(BinderInfo),
    /// Build Pi from top 2 results
    BuildPi(BinderInfo),
    /// Build Let from top 3 results
    BuildLet,
    /// Build MData from top result
    BuildMData,
    /// Build Proj from top result
    BuildProj(Name, u32),
}

/// Iterative expression conversion to avoid stack overflow on deeply nested expressions.
fn convert_expr(name: &str, expr: &ParsedExpr) -> Result<Expr, ImportError> {
    // Use stack-allocated small vectors to avoid per-conversion heap allocations
    let mut work: SmallVec<[ConvertWork; 64]> = SmallVec::new();
    let mut results: SmallVec<[Expr; 32]> = SmallVec::new();
    work.push(ConvertWork::Process(expr));

    while let Some(item) = work.pop() {
        match item {
            ConvertWork::Process(e) => match e {
                ParsedExpr::BVar(i) => {
                    if *i > u64::from(u32::MAX) {
                        return Err(ImportError::ExprConversion {
                            name: name.to_string(),
                            message: format!("bvar index too large: {i}"),
                        });
                    }
                    results.push(Expr::BVar(*i as u32));
                }
                ParsedExpr::FVar(id) => {
                    results.push(Expr::FVar(FVarId(hash_str(id))));
                }
                ParsedExpr::MVar(_) => {
                    return Err(ImportError::UnsupportedMVar(name.to_string()));
                }
                ParsedExpr::Sort(lvl) => {
                    results.push(Expr::Sort(convert_level(name, lvl)?));
                }
                ParsedExpr::Const(n, lvls) => {
                    let levels: LevelVec = lvls
                        .iter()
                        .map(|l| convert_level(name, l))
                        .collect::<Result<_, _>>()?;
                    // Use interning - names like "Nat", "Bool", "List" appear thousands
                    // of times across different constants' expression trees
                    results.push(Expr::Const(Name::interned(n), levels));
                }
                ParsedExpr::Lit(lit) => {
                    let expr = match lit {
                        ParsedLiteral::Nat(n) => Expr::Lit(Literal::Nat(*n)),
                        ParsedLiteral::String(s) => Expr::Lit(Literal::String(s.clone().into())),
                    };
                    results.push(expr);
                }
                ParsedExpr::App(f, a) => {
                    // Push build instruction, then children (in reverse order)
                    work.push(ConvertWork::BuildApp);
                    work.push(ConvertWork::Process(a)); // arg processed second (on top)
                    work.push(ConvertWork::Process(f)); // func processed first
                }
                ParsedExpr::Lam(_, ty, body, info) => {
                    let binder_info = convert_binder_info(*info);
                    work.push(ConvertWork::BuildLam(binder_info));
                    work.push(ConvertWork::Process(body)); // body on top
                    work.push(ConvertWork::Process(ty)); // type first
                }
                ParsedExpr::ForallE(_, ty, body, info) => {
                    let binder_info = convert_binder_info(*info);
                    work.push(ConvertWork::BuildPi(binder_info));
                    work.push(ConvertWork::Process(body)); // body on top
                    work.push(ConvertWork::Process(ty)); // type first
                }
                ParsedExpr::LetE(_, ty, val, body, _) => {
                    work.push(ConvertWork::BuildLet);
                    work.push(ConvertWork::Process(body)); // body on top
                    work.push(ConvertWork::Process(val)); // val second
                    work.push(ConvertWork::Process(ty)); // type first
                }
                ParsedExpr::MData(inner) => {
                    work.push(ConvertWork::BuildMData);
                    work.push(ConvertWork::Process(inner));
                }
                ParsedExpr::Proj(struct_name, idx, inner) => {
                    if *idx > u64::from(u32::MAX) {
                        return Err(ImportError::ExprConversion {
                            name: name.to_string(),
                            message: format!("projection index too large: {idx}"),
                        });
                    }
                    // Use interning - struct names like "Prod", "And" appear many times
                    work.push(ConvertWork::BuildProj(
                        Name::interned(struct_name),
                        *idx as u32,
                    ));
                    work.push(ConvertWork::Process(inner));
                }
            },
            ConvertWork::BuildApp => {
                let arg = results.pop().expect("stack balance invariant");
                let func = results.pop().expect("stack balance invariant");
                results.push(Expr::App(Arc::new(func), Arc::new(arg)));
            }
            ConvertWork::BuildLam(info) => {
                let body = results.pop().expect("stack balance invariant");
                let ty = results.pop().expect("stack balance invariant");
                results.push(Expr::Lam(info, Arc::new(ty), Arc::new(body)));
            }
            ConvertWork::BuildPi(info) => {
                let body = results.pop().expect("stack balance invariant");
                let ty = results.pop().expect("stack balance invariant");
                results.push(Expr::Pi(info, Arc::new(ty), Arc::new(body)));
            }
            ConvertWork::BuildLet => {
                let body = results.pop().expect("stack balance invariant");
                let val = results.pop().expect("stack balance invariant");
                let ty = results.pop().expect("stack balance invariant");
                results.push(Expr::Let(Arc::new(ty), Arc::new(val), Arc::new(body)));
            }
            ConvertWork::BuildMData => {
                let inner = results.pop().expect("stack balance invariant");
                results.push(Expr::MData(Vec::new(), Arc::new(inner)));
            }
            ConvertWork::BuildProj(struct_name, idx) => {
                let inner = results.pop().expect("stack balance invariant");
                results.push(Expr::Proj(struct_name, idx, Arc::new(inner)));
            }
        }
    }

    debug_assert_eq!(results.len(), 1);
    Ok(results.pop().expect("stack balance invariant"))
}

/// Check if a type mentions any inductive in the given list.
fn type_mentions_any_inductive(expr: &Expr, inductive_names: &[Name]) -> bool {
    match expr {
        Expr::Const(name, _) => inductive_names.contains(name),
        Expr::App(f, a) => {
            type_mentions_any_inductive(f, inductive_names)
                || type_mentions_any_inductive(a, inductive_names)
        }
        Expr::Pi(_, domain, codomain) | Expr::Lam(_, domain, codomain) => {
            type_mentions_any_inductive(domain, inductive_names)
                || type_mentions_any_inductive(codomain, inductive_names)
        }
        Expr::Let(ty, val, body) => {
            type_mentions_any_inductive(ty, inductive_names)
                || type_mentions_any_inductive(val, inductive_names)
                || type_mentions_any_inductive(body, inductive_names)
        }
        Expr::Proj(_, _, e) | Expr::MData(_, e) => type_mentions_any_inductive(e, inductive_names),
        _ => false,
    }
}

/// Compute recursive field flags for a constructor type.
///
/// A field is recursive if its domain mentions any inductive from the mutual group.
fn recursive_field_flags_from_ctor(
    ctor_ty: &Expr,
    inductive_names: &[Name],
    num_params: u32,
) -> Vec<bool> {
    let mut flags = Vec::new();
    let mut current = ctor_ty.clone();
    let mut arg_idx = 0u32;

    while let Expr::Pi(_, domain, codomain) = current {
        if arg_idx >= num_params {
            flags.push(type_mentions_any_inductive(&domain, inductive_names));
        }
        current = (*codomain).clone();
        arg_idx += 1;
    }

    flags
}

/// Derive recursive field flags for a constructor using environment data.
fn compute_recursive_fields_from_env(
    env: &Environment,
    ctor_name: &Name,
    inductive_names: &[Name],
    num_params: u32,
    num_fields_hint: u32,
) -> Vec<bool> {
    if let Some(ctor_val) = env.get_constructor(ctor_name) {
        let mut flags =
            recursive_field_flags_from_ctor(&ctor_val.type_, inductive_names, num_params);
        if flags.len() < num_fields_hint as usize {
            flags.resize(num_fields_hint as usize, false);
        } else if flags.len() > num_fields_hint as usize {
            flags.truncate(num_fields_hint as usize);
        }
        flags
    } else {
        vec![false; num_fields_hint as usize]
    }
}

fn convert_constant(constant: &ParsedConstant) -> Result<Declaration, ImportError> {
    let type_ = constant
        .type_
        .as_ref()
        .ok_or_else(|| ImportError::MissingType(constant.name.clone()))
        .and_then(|t| convert_expr(&constant.name, t))?;

    let value = match &constant.value {
        Some(v) => Some(convert_expr(&constant.name, v)?),
        None => None,
    };

    let level_params = convert_level_params(&constant.level_params);
    let name = Name::interned(&constant.name);

    match constant.kind {
        ConstantKind::Axiom | ConstantKind::Quot => Ok(Declaration::Axiom {
            name,
            level_params,
            type_,
        }),
        ConstantKind::Definition => {
            let value = value.ok_or_else(|| ImportError::MissingValue(constant.name.clone()))?;
            Ok(Declaration::Definition {
                name,
                level_params,
                type_,
                value,
                is_reducible: true,
            })
        }
        ConstantKind::Theorem => {
            let value = value.ok_or_else(|| ImportError::MissingValue(constant.name.clone()))?;
            Ok(Declaration::Theorem {
                name,
                level_params,
                type_,
                value,
            })
        }
        ConstantKind::Opaque => {
            let value = value.ok_or_else(|| ImportError::MissingValue(constant.name.clone()))?;
            Ok(Declaration::Opaque {
                name,
                level_params,
                type_,
                value,
            })
        }
        // Inductive-related constants are now handled by try_register_* functions
        // This branch should not be reached from normal code path, but kept for safety
        ConstantKind::Inductive | ConstantKind::Constructor | ConstantKind::Recursor => {
            Ok(Declaration::Axiom {
                name,
                level_params,
                type_,
            })
        }
    }
}

fn hash_str(s: &str) -> u64 {
    let mut hasher = RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(s.as_bytes());
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::{collect_default_search_paths, module_name_from_path, ModuleCache, ParsedModule};
    use std::collections::HashMap;
    use std::env;
    use std::fs;
    use std::path::Path;
    use std::time::Duration;
    use tempfile::{NamedTempFile, TempDir};

    fn dummy_module() -> ParsedModule {
        ParsedModule {
            const_names: Vec::new(),
            constants: Vec::new(),
            extra_const_names: Vec::new(),
            imports: Vec::new(),
            mod_idx: 0,
            lean5_payload: None,
        }
    }

    #[test]
    fn test_module_name_from_sample_path() {
        let path =
            Path::new("/tmp/.elan/toolchains/leanprover--lean4---v4.3.0/lib/lean/Init/Core.olean");
        let name = module_name_from_path(path);
        assert_eq!(name.as_deref(), Some("Init.Core"));
    }

    #[test]
    fn test_default_search_paths_prefers_lean_path() {
        let temp_home = TempDir::new().expect("tempdir");
        let first = temp_home.path().join("lean_path_first");
        let second = temp_home.path().join("lean_path_second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();

        let lean_path = env::join_paths([&first, &second]).unwrap();
        let mut env_map = HashMap::new();
        env_map.insert("LEAN_PATH", lean_path);
        env_map.insert("HOME", temp_home.path().as_os_str().to_os_string());

        let paths = collect_default_search_paths(
            |key| env_map.get(key).cloned(),
            |path| std::fs::read_dir(path),
        );

        assert!(
            paths.starts_with(&[first.clone(), second.clone()]),
            "LEAN_PATH entries should be first: {paths:?}"
        );
    }

    #[test]
    fn test_default_search_paths_uses_userprofile_when_home_missing() {
        let temp_home = TempDir::new().expect("tempdir");
        let toolchain_lib = temp_home
            .path()
            .join(".elan/toolchains/leanprover--lean4---v4.3.0/lib/lean");
        fs::create_dir_all(&toolchain_lib).unwrap();

        let mut env_map = HashMap::new();
        env_map.insert("USERPROFILE", temp_home.path().as_os_str().to_os_string());

        let paths = collect_default_search_paths(
            |key| env_map.get(key).cloned(),
            |path| std::fs::read_dir(path),
        );

        assert!(
            !paths.is_empty(),
            "expected toolchain path from USERPROFILE to be discovered"
        );
        assert_eq!(paths[0], toolchain_lib);
    }

    #[test]
    fn module_cache_returns_entry_when_mtime_matches() {
        let cache = ModuleCache::new();
        let file = NamedTempFile::new().expect("temp file");
        std::fs::write(file.path(), b"original").unwrap();

        cache.insert("Init.Core", file.path(), dummy_module());

        let cached = cache.get("Init.Core", file.path());
        assert!(cached.is_some(), "expected cache hit for unchanged file");
        assert_eq!(cache.len(), 1, "entry should remain cached");
    }

    #[test]
    fn module_cache_evicts_when_timestamp_changes() {
        let cache = ModuleCache::new();
        let file = NamedTempFile::new().expect("temp file");
        std::fs::write(file.path(), b"v1").unwrap();

        cache.insert("Init.Changed", file.path(), dummy_module());

        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(file.path(), b"v2").unwrap();

        assert!(
            cache.get("Init.Changed", file.path()).is_none(),
            "stale cache entry should be dropped when mtime changes"
        );
        assert_eq!(cache.len(), 0, "stale entry should be removed");
    }

    #[test]
    fn module_cache_evicts_when_file_is_missing() {
        let cache = ModuleCache::new();
        let file = NamedTempFile::new().expect("temp file");
        let path = file.path().to_path_buf();
        std::fs::write(&path, b"v1").unwrap();

        cache.insert("Init.Missing", &path, dummy_module());
        std::fs::remove_file(&path).unwrap();

        assert!(
            cache.get("Init.Missing", &path).is_none(),
            "cache should not return entries for missing files"
        );
        assert!(cache.is_empty(), "missing file should clear cache entry");
    }
}
