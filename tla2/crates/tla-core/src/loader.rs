//! Module loader for TLA+ EXTENDS and INSTANCE
//!
//! This module provides functionality to load TLA+ modules from disk,
//! enabling support for user-defined modules referenced via EXTENDS.
//!
//! # Search Order
//!
//! When searching for a module `Foo`, the loader looks in:
//! 1. The same directory as the main .tla file
//! 2. Parent directories (up to a limit)
//! 3. Directories in TLA_PATH environment variable
//!
//! # Example
//!
//! ```ignore
//! use tla_core::loader::ModuleLoader;
//! use std::path::Path;
//!
//! let loader = ModuleLoader::new(Path::new("/path/to/Main.tla"));
//! let voting = loader.load("Voting")?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ast::{Expr, Module};
use crate::lower::lower_all_modules;
use crate::span::FileId;
use crate::stdlib::is_stdlib_module;
use crate::syntax::{parse_to_syntax_tree, SyntaxNode};

/// Error during module loading
#[derive(Debug, Clone)]
pub enum LoadError {
    /// Module file not found
    NotFound {
        module: String,
        search_paths: Vec<PathBuf>,
    },
    /// IO error reading file
    IoError { path: PathBuf, message: String },
    /// Parse error in module
    ParseError { path: PathBuf, errors: Vec<String> },
    /// Lower error in module
    LowerError { path: PathBuf, errors: Vec<String> },
    /// Circular dependency detected
    CircularDependency { chain: Vec<String> },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::NotFound {
                module,
                search_paths,
            } => {
                write!(f, "Module '{}' not found. Searched in:", module)?;
                for path in search_paths {
                    write!(f, "\n  - {}", path.display())?;
                }
                Ok(())
            }
            LoadError::IoError { path, message } => {
                write!(f, "Error reading {}: {}", path.display(), message)
            }
            LoadError::ParseError { path, errors } => {
                write!(f, "Parse errors in {}:", path.display())?;
                for err in errors {
                    write!(f, "\n  {}", err)?;
                }
                Ok(())
            }
            LoadError::LowerError { path, errors } => {
                write!(f, "Lower errors in {}:", path.display())?;
                for err in errors {
                    write!(f, "\n  {}", err)?;
                }
                Ok(())
            }
            LoadError::CircularDependency { chain } => {
                write!(
                    f,
                    "Circular module dependency detected: {}",
                    chain.join(" -> ")
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Loaded module with metadata
#[derive(Debug, Clone)]
pub struct LoadedModule {
    /// The parsed and lowered module
    pub module: Module,
    /// The syntax tree (for SPECIFICATION resolution)
    pub syntax_tree: SyntaxNode,
    /// Path to the source file
    pub path: PathBuf,
    /// File ID for span tracking
    pub file_id: FileId,
}

/// Module loader that caches loaded modules
#[derive(Debug)]
pub struct ModuleLoader {
    /// Base directory for module search (directory containing main .tla file)
    base_dir: PathBuf,
    /// Additional search paths
    search_paths: Vec<PathBuf>,
    /// Cache of loaded modules (module name -> loaded module)
    cache: HashMap<String, LoadedModule>,
    /// Counter for file IDs
    next_file_id: u32,
    /// Stack for detecting circular dependencies
    loading_stack: Vec<String>,
}

impl ModuleLoader {
    /// Create a new module loader with base directory derived from the main file
    pub fn new(main_file: &Path) -> Self {
        let base_dir = main_file
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        // Parse TLA_PATH environment variable
        let tla_path = std::env::var("TLA_PATH").unwrap_or_default();
        let search_paths: Vec<PathBuf> = tla_path
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();

        Self {
            base_dir,
            search_paths,
            cache: HashMap::new(),
            next_file_id: 1, // 0 is reserved for main file
            loading_stack: Vec::new(),
        }
    }

    /// Create a loader with explicit base directory
    pub fn with_base_dir(base_dir: PathBuf) -> Self {
        Self {
            base_dir,
            search_paths: Vec::new(),
            cache: HashMap::new(),
            next_file_id: 1,
            loading_stack: Vec::new(),
        }
    }

    /// Add a search path
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    /// Get the base directory
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Check if a module is a standard library module
    pub fn is_stdlib(&self, name: &str) -> bool {
        is_stdlib_module(name)
    }

    /// Load a module by name
    ///
    /// Returns the cached module if already loaded, otherwise searches for
    /// and parses the .tla file.
    ///
    /// Note: TLA+ "standard library" modules are usually provided by tools
    /// (e.g. SANY/TLC) rather than loaded from disk. However, some specs
    /// vendor community/stdlib modules as `.tla` files alongside the spec
    /// (or via `TLA_PATH`). For correctness, we allow explicit on-disk loads
    /// even when a module name appears in our stdlib registry.
    pub fn load(&mut self, name: &str) -> Result<&LoadedModule, LoadError> {
        // Check cache first
        if self.cache.contains_key(name) {
            return Ok(self.cache.get(name).unwrap());
        }

        // Check for circular dependency
        if self.loading_stack.contains(&name.to_string()) {
            let mut chain = self.loading_stack.clone();
            chain.push(name.to_string());
            return Err(LoadError::CircularDependency { chain });
        }

        // Find the module file
        let path = self.find_module(name)?;

        // Load and parse the module
        self.loading_stack.push(name.to_string());
        let result = self.load_from_path(name, &path);
        self.loading_stack.pop();

        result?;

        Ok(self.cache.get(name).unwrap())
    }

    /// Find the .tla file for a module
    fn find_module(&self, name: &str) -> Result<PathBuf, LoadError> {
        let filename = format!("{}.tla", name);
        let mut searched = Vec::new();

        // Search in base directory first
        let base_path = self.base_dir.join(&filename);
        searched.push(self.base_dir.clone());
        if base_path.exists() {
            return Ok(base_path);
        }

        // Search in additional search paths
        for search_path in &self.search_paths {
            searched.push(search_path.clone());
            let path = search_path.join(&filename);
            if path.exists() {
                return Ok(path);
            }
        }

        // Search parent directories (up to 3 levels)
        let mut current = self.base_dir.clone();
        for _ in 0..3 {
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
                searched.push(current.clone());
                let path = current.join(&filename);
                if path.exists() {
                    return Ok(path);
                }
            } else {
                break;
            }
        }

        Err(LoadError::NotFound {
            module: name.to_string(),
            search_paths: searched,
        })
    }

    /// Load a module from a specific path
    fn load_from_path(&mut self, name: &str, path: &Path) -> Result<(), LoadError> {
        // Read the file
        let source = std::fs::read_to_string(path).map_err(|e| LoadError::IoError {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

        // Parse
        let tree = parse_to_syntax_tree(&source);
        // Note: parse_to_syntax_tree doesn't return errors directly,
        // errors are in the green node. For now, we proceed.

        // Lower
        let file_id = FileId(self.next_file_id);
        self.next_file_id += 1;

        let result = lower_all_modules(file_id, &tree);
        if !result.errors.is_empty() {
            return Err(LoadError::LowerError {
                path: path.to_path_buf(),
                errors: result.errors.iter().map(|e| e.message.clone()).collect(),
            });
        }

        if result.modules.is_empty() {
            return Err(LoadError::LowerError {
                path: path.to_path_buf(),
                errors: vec!["No MODULE found in file".to_string()],
            });
        }

        // Cache all modules found in the file, including inline submodules.
        // This is required for specs that use `INSTANCE Inner` where `Inner` is
        // defined inside another module (e.g. diskpaxos/Synod.tla).
        let mut saw_requested = false;
        for module in result.modules {
            let module_name = module.name.node.clone();
            if module_name == name {
                saw_requested = true;
            }
            self.cache.insert(
                module_name,
                LoadedModule {
                    module,
                    syntax_tree: tree.clone(),
                    path: path.to_path_buf(),
                    file_id,
                },
            );
        }

        if !saw_requested {
            return Err(LoadError::LowerError {
                path: path.to_path_buf(),
                errors: vec![format!(
                    "File defines no module named '{}' (found different MODULE header)",
                    name
                )],
            });
        }

        Ok(())
    }

    /// Load all modules that a module extends (non-stdlib only)
    ///
    /// This recursively loads all extended modules and their instanced modules.
    /// Extended modules may define named instances (e.g., C == INSTANCE Consensus)
    /// which need to be loaded for proper operator resolution.
    pub fn load_extends(&mut self, module: &Module) -> Result<Vec<String>, LoadError> {
        let mut loaded = Vec::new();

        for ext in &module.extends {
            let name = &ext.node;
            let loaded_this = match self.load(name) {
                Ok(_) => true,
                Err(LoadError::NotFound { .. }) if is_stdlib_module(name) => false,
                Err(e) => return Err(e),
            };
            if !loaded_this {
                continue;
            }

            loaded.push(name.clone());

            // Clone to avoid borrow issues
            let sub_module = self.cache.get(name).map(|m| m.module.clone());
            if let Some(sub_mod) = sub_module {
                // Recursively load extended modules
                let sub_extends = self.load_extends(&sub_mod)?;
                loaded.extend(sub_extends);
                // Also load instanced modules from extended modules
                // This is needed for cases like: Main EXTENDS Voting, Voting has C == INSTANCE Consensus
                let sub_instances = self.load_instances(&sub_mod)?;
                for name in sub_instances {
                    if !loaded.contains(&name) {
                        loaded.push(name);
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Load all modules that a module instances (non-stdlib only)
    ///
    /// This recursively loads all instanced modules and their dependencies.
    /// Unlike EXTENDS, INSTANCE modules may have substitutions (WITH clauses),
    /// which are handled at evaluation time.
    pub fn load_instances(&mut self, module: &Module) -> Result<Vec<String>, LoadError> {
        use crate::ast::Unit;

        let mut loaded = Vec::new();

        for unit in &module.units {
            // Handle standalone INSTANCE declarations
            if let Unit::Instance(inst) = &unit.node {
                let name = &inst.module.node;
                let loaded_this = match self.load(name) {
                    Ok(_) => true,
                    Err(LoadError::NotFound { .. }) if is_stdlib_module(name) => false,
                    Err(e) => return Err(e),
                };
                if !loaded_this {
                    continue;
                }

                loaded.push(name.clone());

                // Clone the module to avoid borrow issues
                let sub_module = self.cache.get(name).map(|m| m.module.clone());
                if let Some(sub_mod) = sub_module {
                    // Load its extended modules
                    let sub_extends = self.load_extends(&sub_mod)?;
                    loaded.extend(sub_extends);
                    // Load its instanced modules
                    let sub_instances = self.load_instances(&sub_mod)?;
                    loaded.extend(sub_instances);
                }
            }

            // Handle named instances: InChan == INSTANCE Channel WITH ...
            // These are operators whose body is InstanceExpr
            if let Unit::Operator(def) = &unit.node {
                if let Expr::InstanceExpr(module_name, _subs) = &def.body.node {
                    let loaded_this = match self.load(module_name) {
                        Ok(_) => true,
                        Err(LoadError::NotFound { .. }) if is_stdlib_module(module_name) => false,
                        Err(e) => return Err(e),
                    };
                    if !loaded_this {
                        continue;
                    }
                    if !loaded.contains(module_name) {
                        loaded.push(module_name.clone());
                    }

                    // Clone the module to avoid borrow issues
                    let sub_module = self.cache.get(module_name).map(|m| m.module.clone());
                    if let Some(sub_mod) = sub_module {
                        // Load its extended modules
                        let sub_extends = self.load_extends(&sub_mod)?;
                        loaded.extend(sub_extends);
                        // Load its instanced modules
                        let sub_instances = self.load_instances(&sub_mod)?;
                        loaded.extend(sub_instances);
                    }
                }
            }

            // Handle nested named instances inside operator bodies (e.g. LET G == INSTANCE Graphs IN ...).
            //
            // TLC/SANY allow INSTANCE expressions anywhere an operator definition can appear, not just
            // as a top-level `Op == INSTANCE M ...` declaration. We need to load these instanced
            // modules so that module references like `G!Transpose` can be evaluated.
            if let Unit::Operator(def) = &unit.node {
                fn collect_instance_expr_modules(
                    expr: &Expr,
                    out: &mut std::collections::HashSet<String>,
                ) {
                    match expr {
                        Expr::InstanceExpr(module_name, subs) => {
                            out.insert(module_name.clone());
                            for sub in subs {
                                collect_instance_expr_modules(&sub.to.node, out);
                            }
                        }

                        Expr::Bool(_)
                        | Expr::Int(_)
                        | Expr::String(_)
                        | Expr::Ident(_)
                        | Expr::OpRef(_) => {}

                        Expr::Apply(op, args) => {
                            collect_instance_expr_modules(&op.node, out);
                            for arg in args {
                                collect_instance_expr_modules(&arg.node, out);
                            }
                        }
                        Expr::ModuleRef(_, _, args) => {
                            for arg in args {
                                collect_instance_expr_modules(&arg.node, out);
                            }
                        }
                        Expr::Lambda(_params, body) => {
                            collect_instance_expr_modules(&body.node, out)
                        }

                        Expr::And(a, b)
                        | Expr::Or(a, b)
                        | Expr::Implies(a, b)
                        | Expr::Equiv(a, b)
                        | Expr::In(a, b)
                        | Expr::NotIn(a, b)
                        | Expr::Subseteq(a, b)
                        | Expr::Union(a, b)
                        | Expr::Intersect(a, b)
                        | Expr::SetMinus(a, b)
                        | Expr::FuncSet(a, b)
                        | Expr::LeadsTo(a, b)
                        | Expr::WeakFair(a, b)
                        | Expr::StrongFair(a, b)
                        | Expr::Eq(a, b)
                        | Expr::Neq(a, b)
                        | Expr::Lt(a, b)
                        | Expr::Leq(a, b)
                        | Expr::Gt(a, b)
                        | Expr::Geq(a, b)
                        | Expr::Add(a, b)
                        | Expr::Sub(a, b)
                        | Expr::Mul(a, b)
                        | Expr::Div(a, b)
                        | Expr::IntDiv(a, b)
                        | Expr::Mod(a, b)
                        | Expr::Pow(a, b)
                        | Expr::Range(a, b) => {
                            collect_instance_expr_modules(&a.node, out);
                            collect_instance_expr_modules(&b.node, out);
                        }

                        Expr::Not(a)
                        | Expr::Powerset(a)
                        | Expr::BigUnion(a)
                        | Expr::Domain(a)
                        | Expr::Prime(a)
                        | Expr::Always(a)
                        | Expr::Eventually(a)
                        | Expr::Enabled(a)
                        | Expr::Unchanged(a)
                        | Expr::Neg(a)
                        | Expr::RecordAccess(a, _) => collect_instance_expr_modules(&a.node, out),

                        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                            for b in bounds {
                                if let Some(domain) = &b.domain {
                                    collect_instance_expr_modules(&domain.node, out);
                                }
                            }
                            collect_instance_expr_modules(&body.node, out);
                        }
                        Expr::Choose(bound, body) => {
                            if let Some(domain) = &bound.domain {
                                collect_instance_expr_modules(&domain.node, out);
                            }
                            collect_instance_expr_modules(&body.node, out);
                        }

                        Expr::SetEnum(elems) | Expr::Tuple(elems) | Expr::Times(elems) => {
                            for e in elems {
                                collect_instance_expr_modules(&e.node, out);
                            }
                        }
                        Expr::SetBuilder(expr, bounds) => {
                            collect_instance_expr_modules(&expr.node, out);
                            for b in bounds {
                                if let Some(domain) = &b.domain {
                                    collect_instance_expr_modules(&domain.node, out);
                                }
                            }
                        }
                        Expr::SetFilter(bound, pred) => {
                            if let Some(domain) = &bound.domain {
                                collect_instance_expr_modules(&domain.node, out);
                            }
                            collect_instance_expr_modules(&pred.node, out);
                        }

                        Expr::FuncDef(bounds, body) => {
                            for b in bounds {
                                if let Some(domain) = &b.domain {
                                    collect_instance_expr_modules(&domain.node, out);
                                }
                            }
                            collect_instance_expr_modules(&body.node, out);
                        }
                        Expr::FuncApply(f, arg) => {
                            collect_instance_expr_modules(&f.node, out);
                            collect_instance_expr_modules(&arg.node, out);
                        }
                        Expr::Except(f, specs) => {
                            collect_instance_expr_modules(&f.node, out);
                            for spec in specs {
                                for p in &spec.path {
                                    if let crate::ast::ExceptPathElement::Index(idx) = p {
                                        collect_instance_expr_modules(&idx.node, out);
                                    }
                                }
                                collect_instance_expr_modules(&spec.value.node, out);
                            }
                        }

                        Expr::Record(fields) | Expr::RecordSet(fields) => {
                            for (_name, expr) in fields {
                                collect_instance_expr_modules(&expr.node, out);
                            }
                        }

                        Expr::If(c, t, e) => {
                            collect_instance_expr_modules(&c.node, out);
                            collect_instance_expr_modules(&t.node, out);
                            collect_instance_expr_modules(&e.node, out);
                        }
                        Expr::Case(arms, other) => {
                            for arm in arms {
                                collect_instance_expr_modules(&arm.guard.node, out);
                                collect_instance_expr_modules(&arm.body.node, out);
                            }
                            if let Some(other) = other {
                                collect_instance_expr_modules(&other.node, out);
                            }
                        }
                        Expr::Let(defs, body) => {
                            for d in defs {
                                collect_instance_expr_modules(&d.body.node, out);
                            }
                            collect_instance_expr_modules(&body.node, out);
                        }
                    }
                }

                let mut nested = std::collections::HashSet::new();
                collect_instance_expr_modules(&def.body.node, &mut nested);

                for module_name in nested {
                    // If the instanced module cannot be found, surface the error: the spec will not
                    // be able to evaluate `I!Op` references without those definitions.
                    let loaded_this = match self.load(&module_name) {
                        Ok(_) => true,
                        Err(LoadError::NotFound { .. }) if is_stdlib_module(&module_name) => false,
                        Err(e) => return Err(e),
                    };
                    if !loaded_this {
                        continue;
                    }
                    if !loaded.contains(&module_name) {
                        loaded.push(module_name.clone());
                    }

                    let sub_module = self.cache.get(&module_name).map(|m| m.module.clone());
                    if let Some(sub_mod) = sub_module {
                        loaded.extend(self.load_extends(&sub_mod)?);
                        loaded.extend(self.load_instances(&sub_mod)?);
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Seed the cache with all modules found in a syntax tree.
    ///
    /// This is used to pre-populate the cache with inline modules from the main file.
    /// When a TLA+ file contains multiple MODULE definitions (inline submodules),
    /// those modules should be available for EXTENDS/INSTANCE resolution.
    ///
    /// Example: BufferedRandomAccessFile.tla contains inline modules Common and
    /// RandomAccessFile. When the main module does `EXTENDS Common`, we need
    /// Common to be in the cache before `load_extends` is called.
    pub fn seed_from_syntax_tree(&mut self, tree: &SyntaxNode, path: &Path) {
        let file_id = FileId(0); // Main file always has ID 0

        let result = lower_all_modules(file_id, tree);
        // Ignore errors here - main.rs already reported them during initial lowering

        for module in result.modules {
            let module_name = module.name.node.clone();
            // Don't overwrite existing entries (main module was already processed)
            self.cache
                .entry(module_name)
                .or_insert_with(|| LoadedModule {
                    module,
                    syntax_tree: tree.clone(),
                    path: path.to_path_buf(),
                    file_id,
                });
        }
    }

    /// Get a loaded module from the cache
    pub fn get(&self, name: &str) -> Option<&LoadedModule> {
        self.cache.get(name)
    }

    /// Get all loaded modules
    pub fn loaded_modules(&self) -> impl Iterator<Item = (&str, &LoadedModule)> {
        self.cache.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_module(dir: &Path, name: &str, content: &str) {
        let path = dir.join(format!("{}.tla", name));
        fs::write(path, content).unwrap();
    }

    #[test]
    fn test_find_module_in_base_dir() {
        let temp = TempDir::new().unwrap();
        create_test_module(
            temp.path(),
            "Voting",
            r#"
            ---- MODULE Voting ----
            VARIABLE votes
            ====
        "#,
        );

        let main_file = temp.path().join("Main.tla");
        let loader = ModuleLoader::new(&main_file);

        let path = loader.find_module("Voting").unwrap();
        assert!(path.ends_with("Voting.tla"));
    }

    #[test]
    fn test_module_not_found() {
        let temp = TempDir::new().unwrap();
        let main_file = temp.path().join("Main.tla");
        let loader = ModuleLoader::new(&main_file);

        let result = loader.find_module("NonExistent");
        assert!(matches!(result, Err(LoadError::NotFound { .. })));
    }

    #[test]
    fn test_load_module() {
        let temp = TempDir::new().unwrap();
        create_test_module(
            temp.path(),
            "Voting",
            r#"
            ---- MODULE Voting ----
            VARIABLE votes
            Init == votes = 0
            ====
        "#,
        );

        let main_file = temp.path().join("Main.tla");
        let mut loader = ModuleLoader::new(&main_file);

        let loaded = loader.load("Voting").unwrap();
        assert_eq!(loaded.module.name.node, "Voting");
    }

    #[test]
    fn test_module_caching() {
        let temp = TempDir::new().unwrap();
        create_test_module(
            temp.path(),
            "Voting",
            r#"
            ---- MODULE Voting ----
            VARIABLE votes
            ====
        "#,
        );

        let main_file = temp.path().join("Main.tla");
        let mut loader = ModuleLoader::new(&main_file);

        // Load twice - should use cache
        let _ = loader.load("Voting").unwrap();
        let _ = loader.load("Voting").unwrap();

        // Only one entry in cache
        assert_eq!(loader.cache.len(), 1);
    }

    #[test]
    fn test_load_inline_submodule() {
        let temp = TempDir::new().unwrap();
        create_test_module(
            temp.path(),
            "Synod",
            r#"
            ---- MODULE Synod ----
            EXTENDS Naturals

            CONSTANTS N, Inputs
            Proc == 1..N

            ---- MODULE Inner ----
            VARIABLES chosen
            IInit == chosen = CHOOSE x : x \in Inputs
            ====

            IS(chosen) == INSTANCE Inner
            Spec == \EE chosen : IS(chosen)!IInit
            ====
        "#,
        );

        let main_file = temp.path().join("Main.tla");
        let mut loader = ModuleLoader::new(&main_file);

        let _ = loader.load("Synod").unwrap();

        let inner = loader
            .get("Inner")
            .expect("Inner submodule should be cached");
        assert_eq!(inner.module.name.node, "Inner");
        assert_eq!(loader.cache.len(), 2);
    }

    #[test]
    fn test_seed_from_syntax_tree_for_extends() {
        // Test that seed_from_syntax_tree enables EXTENDS to resolve inline modules.
        // This is the pattern used by BufferedRandomAccessFile.tla which defines
        // Common and RandomAccessFile as inline modules, then does EXTENDS Common.
        let temp = TempDir::new().unwrap();
        create_test_module(
            temp.path(),
            "Main",
            r#"
            ---- MODULE Main ----
            EXTENDS Common
            VARIABLES x
            Init == Common!Helper(x)
            ====

            ---- MODULE Common ----
            Helper(v) == v \in {1,2,3}
            ====
        "#,
        );

        let main_file = temp.path().join("Main.tla");
        let source = fs::read_to_string(&main_file).unwrap();
        let tree = crate::syntax::parse_to_syntax_tree(&source);

        let mut loader = ModuleLoader::new(&main_file);

        // Before seeding, Common is not in cache
        assert!(loader.get("Common").is_none());

        // Seed from syntax tree
        loader.seed_from_syntax_tree(&tree, &main_file);

        // Now Common should be in cache
        let common = loader
            .get("Common")
            .expect("Common should be cached after seeding");
        assert_eq!(common.module.name.node, "Common");

        // Main is also cached
        let main = loader.get("Main").expect("Main should also be cached");
        assert_eq!(main.module.name.node, "Main");
    }

    #[test]
    fn test_is_stdlib() {
        let loader = ModuleLoader::with_base_dir(PathBuf::from("."));

        assert!(loader.is_stdlib("Naturals"));
        assert!(loader.is_stdlib("Integers"));
        assert!(loader.is_stdlib("Sequences"));
        assert!(!loader.is_stdlib("Voting"));
        assert!(!loader.is_stdlib("MyModule"));
    }
}
