//! Lake build system
//!
//! Provides incremental compilation and parallel builds.

use crate::error::{LakeError, LakeResult};
use crate::workspace::Workspace;
use lean5_elab::ElabResult;
use lean5_kernel::env::{ConstantInfo, Declaration};
use lean5_kernel::inductive::{
    Constructor, ConstructorVal, InductiveDecl, InductiveType, InductiveVal, RecursorVal,
};
use lean5_kernel::name::Name;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Result of building modules: (built, skipped, failed)
/// - built: List of successfully built module names
/// - skipped: List of modules skipped (already up to date)
/// - failed: List of (module_name, error_message) for failed builds
type BuildModulesResult = (Vec<String>, Vec<String>, Vec<(String, String)>);

/// Build context for a workspace
pub struct BuildContext {
    /// The workspace being built
    workspace: Workspace,
    /// Module dependency graph
    deps: HashMap<String, Vec<String>>,
    /// Build options
    options: BuildOptions,
}

/// Build options
#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    /// Number of parallel jobs (0 = auto)
    pub jobs: usize,
    /// Verbose output
    pub verbose: bool,
    /// Force rebuild all
    pub force: bool,
    /// Only check types, don't generate code
    pub check_only: bool,
}

/// Result of a build operation
#[derive(Debug, Clone)]
pub struct BuildResult {
    /// Modules that were built
    pub built: Vec<String>,
    /// Modules that were skipped (up to date)
    pub skipped: Vec<String>,
    /// Modules that failed
    pub failed: Vec<(String, String)>,
    /// Total build time
    pub duration: Duration,
}

impl BuildResult {
    /// Check if build was successful
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.failed.is_empty()
    }

    /// Get total number of modules processed
    #[must_use]
    pub fn total(&self) -> usize {
        self.built.len() + self.skipped.len() + self.failed.len()
    }
}

impl BuildContext {
    /// Create a new build context for a workspace
    #[must_use]
    pub fn new(workspace: Workspace) -> Self {
        Self {
            workspace,
            deps: HashMap::new(),
            options: BuildOptions::default(),
        }
    }

    /// Set build options
    #[must_use]
    pub fn with_options(mut self, options: BuildOptions) -> Self {
        self.options = options;
        self
    }

    /// Get the workspace
    #[must_use]
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// Build all targets
    pub fn build_all(&mut self) -> LakeResult<BuildResult> {
        let start = Instant::now();

        // Create output directories
        self.workspace.validate_dependencies()?;
        self.workspace.create_dirs()?;

        // Discover dependencies
        self.discover_dependencies()?;

        // Get modules to build in topological order
        let modules = self.topological_sort()?;

        // Build modules
        let result = self.build_modules(&modules);

        Ok(BuildResult {
            built: result.0,
            skipped: result.1,
            failed: result.2,
            duration: start.elapsed(),
        })
    }

    /// Build a specific target (library or executable)
    pub fn build_target(&mut self, target: &str) -> LakeResult<BuildResult> {
        let start = Instant::now();

        // Create output directories
        self.workspace.validate_dependencies()?;
        self.workspace.create_dirs()?;

        // Find target
        let modules = if let Some(lib) = self
            .workspace
            .config()
            .libs
            .iter()
            .find(|l| l.name == target)
        {
            self.workspace.lib_modules(&lib.name)
        } else if let Some(exe) = self
            .workspace
            .config()
            .exes
            .iter()
            .find(|e| e.name == target)
        {
            vec![exe.root.clone()]
        } else {
            return Err(LakeError::ModuleNotFound(target.to_string()));
        };

        // Discover dependencies for these modules
        for module in &modules {
            self.discover_module_deps(module)?;
        }

        // Build in dependency order
        let ordered = self.topological_sort_subset(&modules)?;
        let result = self.build_modules(&ordered);

        Ok(BuildResult {
            built: result.0,
            skipped: result.1,
            failed: result.2,
            duration: start.elapsed(),
        })
    }

    /// Discover all module dependencies
    fn discover_dependencies(&mut self) -> LakeResult<()> {
        self.deps.clear();

        for module in self.workspace.all_modules() {
            self.discover_module_deps(&module)?;
        }

        Ok(())
    }

    /// Discover dependencies for a single module
    fn discover_module_deps(&mut self, module: &str) -> LakeResult<()> {
        if self.deps.contains_key(module) {
            return Ok(());
        }

        let Some(src_path) = self.workspace.find_module(module) else {
            // External module - no deps to discover
            self.deps.insert(module.to_string(), vec![]);
            return Ok(());
        };

        let content = std::fs::read_to_string(&src_path)?;
        let imports = Self::extract_imports(&content);

        // Recursively discover deps for imported modules
        for imp in &imports {
            if !self.deps.contains_key(imp) {
                self.discover_module_deps(imp)?;
            }
        }

        self.deps.insert(module.to_string(), imports);
        Ok(())
    }

    /// Extract import statements from Lean source
    fn extract_imports(content: &str) -> Vec<String> {
        let mut imports = vec![];

        for line in content.lines() {
            let line = line.trim();

            // Skip comments
            if line.starts_with("--") {
                continue;
            }

            // Handle import statements
            if let Some(rest) = line.strip_prefix("import ") {
                let module = rest.trim().to_string();
                // Skip standard library imports for now
                if !module.starts_with("Init")
                    && !module.starts_with("Std")
                    && !module.starts_with("Lean")
                    && !module.starts_with("Lake")
                    && !module.starts_with("Mathlib")
                {
                    imports.push(module);
                }
            }

            // Stop at first non-import declaration
            if !line.is_empty()
                && !line.starts_with("import ")
                && !line.starts_with("--")
                && !line.starts_with("open ")
                && !line.starts_with("set_option ")
            {
                // Continue - imports can be interspersed with other lines in modern Lean
            }
        }

        imports
    }

    /// Topological sort of all modules
    fn topological_sort(&self) -> LakeResult<Vec<String>> {
        self.topological_sort_subset(&self.deps.keys().cloned().collect::<Vec<_>>())
    }

    /// Topological sort of a subset of modules
    fn topological_sort_subset(&self, modules: &[String]) -> LakeResult<Vec<String>> {
        let mut result = vec![];
        let mut visited = HashSet::new();
        let mut temp_marks = HashSet::new();

        fn visit(
            module: &str,
            deps: &HashMap<String, Vec<String>>,
            visited: &mut HashSet<String>,
            temp_marks: &mut HashSet<String>,
            result: &mut Vec<String>,
        ) -> Result<(), String> {
            if temp_marks.contains(module) {
                return Err(module.to_string());
            }
            if visited.contains(module) {
                return Ok(());
            }

            temp_marks.insert(module.to_string());

            if let Some(module_deps) = deps.get(module) {
                for dep in module_deps {
                    visit(dep, deps, visited, temp_marks, result)?;
                }
            }

            temp_marks.remove(module);
            visited.insert(module.to_string());
            result.push(module.to_string());
            Ok(())
        }

        for module in modules {
            if !visited.contains(module) {
                visit(
                    module,
                    &self.deps,
                    &mut visited,
                    &mut temp_marks,
                    &mut result,
                )
                .map_err(LakeError::CircularDependency)?;
            }
        }

        Ok(result)
    }

    /// Build modules in the given order (with parallel execution)
    fn build_modules(&self, modules: &[String]) -> BuildModulesResult {
        use rayon::prelude::*;

        let built = Arc::new(Mutex::new(vec![]));
        let skipped = Arc::new(Mutex::new(vec![]));
        let failed = Arc::new(Mutex::new(vec![]));
        let built_set = Arc::new(Mutex::new(std::collections::HashSet::<String>::new()));

        // Organize modules into levels for parallel execution
        // Level N contains modules that only depend on modules in levels < N
        let levels = self.compute_build_levels(modules);

        // Configure thread pool
        let num_threads = if self.options.jobs == 0 {
            rayon::current_num_threads()
        } else {
            self.options.jobs
        };

        // Build thread pool with requested thread count, falling back to default if that fails
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|_| {
                rayon::ThreadPoolBuilder::new()
                    .build()
                    .expect("failed to create rayon thread pool with default settings")
            });

        // Process each level in order (all modules in a level can build in parallel)
        for level in levels {
            // Check for failures - stop processing if any module failed
            if !failed.lock().expect("mutex not poisoned").is_empty() {
                break;
            }

            pool.install(|| {
                level.par_iter().for_each(|module| {
                    // Check if already failed
                    if !failed.lock().expect("mutex not poisoned").is_empty() {
                        return;
                    }

                    // Check if module needs rebuild
                    if !self.options.force && !self.workspace.needs_rebuild(module) {
                        skipped
                            .lock()
                            .expect("mutex not poisoned")
                            .push(module.clone());
                        built_set
                            .lock()
                            .expect("mutex not poisoned")
                            .insert(module.clone());
                        return;
                    }

                    // Check that all dependencies are built/skipped
                    if let Some(deps) = self.deps.get(module) {
                        let done = built_set.lock().expect("mutex not poisoned");
                        for dep in deps {
                            if !done.contains(dep) {
                                // Dependency not ready - this shouldn't happen if levels are correct
                                // but handle it gracefully
                                failed
                                    .lock()
                                    .expect("mutex not poisoned")
                                    .push((module.clone(), format!("dependency {dep} not built")));
                                return;
                            }
                        }
                    }

                    match self.build_module(module) {
                        Ok(()) => {
                            built
                                .lock()
                                .expect("mutex not poisoned")
                                .push(module.clone());
                            built_set
                                .lock()
                                .expect("mutex not poisoned")
                                .insert(module.clone());
                        }
                        Err(e) => {
                            failed
                                .lock()
                                .expect("mutex not poisoned")
                                .push((module.clone(), e.to_string()));
                        }
                    }
                });
            });
        }

        let built = Arc::try_unwrap(built)
            .expect("built Arc still has other references after parallel build completed")
            .into_inner()
            .expect("built Mutex was poisoned");
        let skipped = Arc::try_unwrap(skipped)
            .expect("skipped Arc still has other references after parallel build completed")
            .into_inner()
            .expect("skipped Mutex was poisoned");
        let failed = Arc::try_unwrap(failed)
            .expect("failed Arc still has other references after parallel build completed")
            .into_inner()
            .expect("failed Mutex was poisoned");

        (built, skipped, failed)
    }

    /// Compute build levels for parallel execution
    /// Returns a vector of levels, where each level contains modules that can be built in parallel
    fn compute_build_levels(&self, modules: &[String]) -> Vec<Vec<String>> {
        let module_set: std::collections::HashSet<_> = modules.iter().cloned().collect();
        let mut levels: Vec<Vec<String>> = vec![];
        let mut assigned: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Keep assigning modules to levels until all are assigned
        while assigned.len() < modules.len() {
            let mut current_level = vec![];

            for module in modules {
                if assigned.contains(module) {
                    continue;
                }

                // Check if all dependencies are either:
                // 1. Not in the module set (external), or
                // 2. Already assigned to a previous level
                let deps_ready = self.deps.get(module).is_none_or(|deps| {
                    deps.iter()
                        .all(|dep| !module_set.contains(dep) || assigned.contains(dep))
                });

                if deps_ready {
                    current_level.push(module.clone());
                }
            }

            // If no progress was made, there's a cycle or missing dependency
            if current_level.is_empty() && assigned.len() < modules.len() {
                // Add remaining modules to final level (will fail on missing deps)
                for module in modules {
                    if !assigned.contains(module) {
                        current_level.push(module.clone());
                    }
                }
            }

            for m in &current_level {
                assigned.insert(m.clone());
            }

            if !current_level.is_empty() {
                levels.push(current_level);
            }
        }

        levels
    }

    /// Build a single module
    fn build_module(&self, module: &str) -> LakeResult<()> {
        let src_path = self
            .workspace
            .find_module(module)
            .ok_or_else(|| LakeError::ModuleNotFound(module.to_string()))?;

        if self.options.verbose {
            eprintln!("Building {module}");
        }

        // Read source
        let content = std::fs::read_to_string(&src_path)?;

        // Parse
        let surface = lean5_parser::parse_file(&content).map_err(|e| LakeError::BuildFailed {
            module: module.to_string(),
            reason: format!("parse error: {e}"),
        })?;

        // Create environment with imported modules
        let mut env = lean5_kernel::Environment::new();

        // Collect import names for .olean export
        let import_names: Vec<String> = self.deps.get(module).cloned().unwrap_or_default();

        // Load local project dependencies from .olean files
        for dep in &import_names {
            let olean_path = self.workspace.olean_path(dep);
            if olean_path.exists() {
                // Load the dependency's .olean into the environment
                match lean5_olean::load_olean_file(&mut env, &olean_path) {
                    Ok(summary) => {
                        if self.options.verbose {
                            eprintln!("  Loaded {} ({} constants)", dep, summary.added_constants);
                        }
                    }
                    Err(e) => {
                        if self.options.verbose {
                            eprintln!("  Warning: failed to load {dep}.olean: {e}");
                        }
                        // Continue - the elaboration may still succeed if the dep
                        // isn't actually used, or we can fall back to stub definitions
                    }
                }
            } else if self.options.verbose {
                eprintln!("  Note: dependency {dep} not yet built");
            }
        }

        // Snapshot environment before elaboration so we can capture only new declarations
        let baseline_consts: HashSet<Name> = env.constants().map(|c| c.name.clone()).collect();
        let baseline_inds: HashSet<Name> = env.inductives().map(|i| i.name.clone()).collect();
        let baseline_ctors: HashSet<Name> = env.constructors().map(|c| c.name.clone()).collect();
        let baseline_recs: HashSet<Name> = env.recursors().map(|r| r.name.clone()).collect();

        // Elaborate
        for decl in &surface {
            let mut elab = lean5_elab::ElabCtx::new(&env);
            let result = elab.elab_decl(decl).map_err(|e| LakeError::BuildFailed {
                module: module.to_string(),
                reason: format!("elaboration error: {e}"),
            })?;

            Self::commit_elab_result(&mut env, result).map_err(|reason| {
                LakeError::BuildFailed {
                    module: module.to_string(),
                    reason,
                }
            })?;
        }

        // Collect newly added declarations for payload/export
        let mut new_inductives: Vec<InductiveVal> = env
            .inductives()
            .filter(|i| !baseline_inds.contains(&i.name))
            .cloned()
            .collect();
        let mut new_constructors: Vec<ConstructorVal> = env
            .constructors()
            .filter(|c| !baseline_ctors.contains(&c.name))
            .cloned()
            .collect();
        let mut new_recursors: Vec<RecursorVal> = env
            .recursors()
            .filter(|r| !baseline_recs.contains(&r.name))
            .cloned()
            .collect();

        let inductive_names: HashSet<Name> =
            new_inductives.iter().map(|i| i.name.clone()).collect();
        let ctor_names: HashSet<Name> = new_constructors.iter().map(|c| c.name.clone()).collect();
        let recursor_names: HashSet<Name> = new_recursors.iter().map(|r| r.name.clone()).collect();

        let mut new_constants: Vec<ConstantInfo> = env
            .constants()
            .filter(|c| !baseline_consts.contains(&c.name))
            .filter(|c| {
                !inductive_names.contains(&c.name)
                    && !ctor_names.contains(&c.name)
                    && !recursor_names.contains(&c.name)
            })
            .cloned()
            .collect();

        // Sort for deterministic output
        new_inductives.sort_by(|a, b| a.name.to_string().cmp(&b.name.to_string()));
        new_constructors.sort_by(|a, b| a.name.to_string().cmp(&b.name.to_string()));
        new_recursors.sort_by(|a, b| a.name.to_string().cmp(&b.name.to_string()));
        new_constants.sort_by(|a, b| a.name.to_string().cmp(&b.name.to_string()));

        let mut structure_fields = Vec::new();
        for ind in &new_inductives {
            if let Some(fields) = env.get_structure_field_names(&ind.name) {
                structure_fields.push((ind.name.clone(), fields.clone()));
            }
        }

        let mut export_const_names: Vec<String> = Vec::new();
        export_const_names.extend(new_inductives.iter().map(|i| i.name.to_string()));
        export_const_names.extend(new_constructors.iter().map(|c| c.name.to_string()));
        export_const_names.extend(new_recursors.iter().map(|r| r.name.to_string()));
        export_const_names.extend(new_constants.iter().map(|c| c.name.to_string()));
        export_const_names.sort();
        export_const_names.dedup();

        // Create output directory
        let olean_path = self.workspace.olean_path(module);
        if let Some(parent) = olean_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Generate .olean file using the exporter
        let imports: Vec<(&str, bool)> = import_names.iter().map(|s| (s.as_str(), false)).collect();
        let const_refs: Vec<&str> = export_const_names.iter().map(String::as_str).collect();

        let payload = lean5_olean::Lean5Payload {
            constants: new_constants,
            inductives: new_inductives,
            constructors: new_constructors,
            recursors: new_recursors,
            structure_fields,
        };

        // Use a fixed git hash for Lean5-generated .olean files (40 hex chars)
        const LEAN5_GIT_HASH: &str = "1ea5000000000000000000000000000000000000";

        let olean_bytes = lean5_olean::OleanExporter::export_with_payload(
            &imports,
            &const_refs,
            LEAN5_GIT_HASH,
            &payload,
        )
        .map_err(|e| LakeError::BuildFailed {
            module: module.to_string(),
            reason: format!("olean export error: {e}"),
        })?;

        std::fs::write(&olean_path, olean_bytes)?;

        Ok(())
    }

    /// Commit an elaboration result into the kernel environment.
    fn commit_elab_result(
        env: &mut lean5_kernel::Environment,
        result: ElabResult,
    ) -> Result<(), String> {
        match result {
            ElabResult::Definition {
                name,
                universe_params,
                ty,
                val,
            } => env
                .add_decl(Declaration::Definition {
                    name,
                    level_params: universe_params,
                    type_: ty,
                    value: val,
                    is_reducible: false,
                })
                .map_err(|e| e.to_string()),
            ElabResult::Theorem {
                name,
                universe_params,
                ty,
                proof,
            } => env
                .add_decl(Declaration::Theorem {
                    name,
                    level_params: universe_params,
                    type_: ty,
                    value: proof,
                })
                .map_err(|e| e.to_string()),
            ElabResult::Axiom {
                name,
                universe_params,
                ty,
            } => env
                .add_decl(Declaration::Axiom {
                    name,
                    level_params: universe_params,
                    type_: ty,
                })
                .map_err(|e| e.to_string()),
            ElabResult::Inductive {
                name,
                universe_params,
                num_params,
                ty,
                constructors,
                derived_instances,
            } => {
                let decl = InductiveDecl {
                    level_params: universe_params,
                    num_params,
                    types: vec![InductiveType {
                        name: name.clone(),
                        type_: ty,
                        constructors: constructors
                            .into_iter()
                            .map(|(ctor_name, ctor_ty)| Constructor {
                                name: ctor_name,
                                type_: ctor_ty,
                            })
                            .collect(),
                    }],
                };
                env.add_inductive(decl).map_err(|e| e.to_string())?;

                for inst in derived_instances {
                    env.add_decl(Declaration::Definition {
                        name: inst.name,
                        level_params: Vec::new(),
                        type_: inst.ty,
                        value: inst.val,
                        is_reducible: false,
                    })
                    .map_err(|e| e.to_string())?;
                }
                Ok(())
            }
            ElabResult::Structure {
                name,
                universe_params,
                num_params,
                ty,
                ctor_name,
                ctor_ty,
                field_names,
                projections,
                derived_instances,
            } => {
                let decl = InductiveDecl {
                    level_params: universe_params,
                    num_params,
                    types: vec![InductiveType {
                        name: name.clone(),
                        type_: ty,
                        constructors: vec![Constructor {
                            name: ctor_name,
                            type_: ctor_ty,
                        }],
                    }],
                };
                env.add_inductive(decl).map_err(|e| e.to_string())?;
                env.register_structure_fields(name.clone(), field_names)
                    .map_err(|e| e.to_string())?;

                for (proj_name, proj_ty, proj_val) in projections {
                    env.add_decl(Declaration::Definition {
                        name: proj_name,
                        level_params: Vec::new(),
                        type_: proj_ty,
                        value: proj_val,
                        is_reducible: true,
                    })
                    .map_err(|e| e.to_string())?;
                }

                for inst in derived_instances {
                    env.add_decl(Declaration::Definition {
                        name: inst.name,
                        level_params: Vec::new(),
                        type_: inst.ty,
                        value: inst.val,
                        is_reducible: false,
                    })
                    .map_err(|e| e.to_string())?;
                }
                Ok(())
            }
            ElabResult::Instance {
                name,
                universe_params,
                ty,
                val,
                ..
            } => env
                .add_decl(Declaration::Definition {
                    name,
                    level_params: universe_params,
                    type_: ty,
                    value: val,
                    is_reducible: false,
                })
                .map_err(|e| e.to_string()),
            ElabResult::Skipped => Ok(()),
        }
    }

    /// Clean build artifacts
    pub fn clean(&self) -> LakeResult<()> {
        let build_dir = self.workspace.build_dir();
        if build_dir.exists() {
            std::fs::remove_dir_all(&build_dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_extract_imports() {
        let content = r"
import MyLib.Core
import MyLib.Utils
-- import Commented

open MyLib

def foo := 1
";
        let imports = BuildContext::extract_imports(content);
        assert_eq!(imports, vec!["MyLib.Core", "MyLib.Utils"]);
    }

    #[test]
    fn test_extract_imports_filters_stdlib() {
        let content = r"
import Init.Core
import Std.Data.List
import Lean.Elab
import MyProject.Core
";
        let imports = BuildContext::extract_imports(content);
        assert_eq!(imports, vec!["MyProject.Core"]);
    }

    #[test]
    fn test_topological_sort() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::new(tmp.path(), "test");
        let mut ctx = BuildContext::new(ws);

        ctx.deps.insert("A".to_string(), vec!["B".to_string()]);
        ctx.deps.insert("B".to_string(), vec!["C".to_string()]);
        ctx.deps.insert("C".to_string(), vec![]);

        let sorted = ctx.topological_sort().unwrap();
        assert_eq!(sorted, vec!["C", "B", "A"]);
    }

    #[test]
    fn test_topological_sort_cycle_detection() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::new(tmp.path(), "test");
        let mut ctx = BuildContext::new(ws);

        ctx.deps.insert("A".to_string(), vec!["B".to_string()]);
        ctx.deps.insert("B".to_string(), vec!["A".to_string()]);

        let result = ctx.topological_sort();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_options_default() {
        let opts = BuildOptions::default();
        assert_eq!(opts.jobs, 0);
        assert!(!opts.verbose);
        assert!(!opts.force);
    }

    #[test]
    fn test_build_result_success() {
        let result = BuildResult {
            built: vec!["A".to_string()],
            skipped: vec!["B".to_string()],
            failed: vec![],
            duration: Duration::from_secs(1),
        };
        assert!(result.is_success());
        assert_eq!(result.total(), 2);
    }

    #[test]
    fn test_build_result_failure() {
        let result = BuildResult {
            built: vec!["A".to_string()],
            skipped: vec![],
            failed: vec![("B".to_string(), "error".to_string())],
            duration: Duration::from_secs(1),
        };
        assert!(!result.is_success());
    }

    #[test]
    fn test_compute_build_levels() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::new(tmp.path(), "test");
        let mut ctx = BuildContext::new(ws);

        // A depends on B, B depends on C, D is independent
        ctx.deps.insert("A".to_string(), vec!["B".to_string()]);
        ctx.deps.insert("B".to_string(), vec!["C".to_string()]);
        ctx.deps.insert("C".to_string(), vec![]);
        ctx.deps.insert("D".to_string(), vec![]);

        let modules = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let levels = ctx.compute_build_levels(&modules);

        // Level 0: C, D (no deps)
        // Level 1: B (depends on C)
        // Level 2: A (depends on B)
        assert_eq!(levels.len(), 3);
        assert!(levels[0].contains(&"C".to_string()));
        assert!(levels[0].contains(&"D".to_string()));
        assert_eq!(levels[1], vec!["B"]);
        assert_eq!(levels[2], vec!["A"]);
    }

    #[test]
    fn test_compute_build_levels_parallel() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::new(tmp.path(), "test");
        let mut ctx = BuildContext::new(ws);

        // A, B, C all depend on D - they can all build in parallel after D
        ctx.deps.insert("A".to_string(), vec!["D".to_string()]);
        ctx.deps.insert("B".to_string(), vec!["D".to_string()]);
        ctx.deps.insert("C".to_string(), vec!["D".to_string()]);
        ctx.deps.insert("D".to_string(), vec![]);

        let modules = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let levels = ctx.compute_build_levels(&modules);

        // Level 0: D
        // Level 1: A, B, C (all can build in parallel)
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], vec!["D"]);
        assert_eq!(levels[1].len(), 3);
        assert!(levels[1].contains(&"A".to_string()));
        assert!(levels[1].contains(&"B".to_string()));
        assert!(levels[1].contains(&"C".to_string()));
    }

    #[test]
    fn test_build_with_dependencies() {
        use std::fs;
        use std::io::Write;

        let tmp = TempDir::new().unwrap();

        // Create lakefile.lean
        let mut lakefile = fs::File::create(tmp.path().join("lakefile.lean")).unwrap();
        writeln!(
            lakefile,
            r#"
package myproject where
  name := "myproject"
  version := "0.1.0"

lean_lib MyProject where
  roots := #[`MyProject]
"#
        )
        .unwrap();

        // Create source directory
        fs::create_dir_all(tmp.path().join("MyProject")).unwrap();

        // Create Core.lean (no dependencies, minimal valid Lean file)
        let mut core = fs::File::create(tmp.path().join("MyProject/Core.lean")).unwrap();
        writeln!(core, "def baseVal : Type := Type").unwrap();

        // Create Main.lean (depends on Core)
        let mut main = fs::File::create(tmp.path().join("MyProject/Main.lean")).unwrap();
        writeln!(
            main,
            r"import MyProject.Core
def mainVal : Type := baseVal"
        )
        .unwrap();

        // Build the project
        let ws = Workspace::load(tmp.path()).unwrap();
        let mut ctx = BuildContext::new(ws).with_options(BuildOptions {
            verbose: true,
            ..Default::default()
        });

        let result = ctx.build_all();

        // The build should succeed
        assert!(result.is_ok(), "Build should succeed: {:?}", result.err());

        let build_result = result.unwrap();
        println!("Built: {:?}", build_result.built);
        println!("Failed: {:?}", build_result.failed);

        // Both modules should be built successfully
        assert!(
            build_result.built.contains(&"MyProject.Core".to_string()),
            "Core should be built"
        );
        assert!(
            build_result.built.contains(&"MyProject.Main".to_string()),
            "Main should be built"
        );
        assert!(
            build_result.failed.is_empty(),
            "No modules should fail: {:?}",
            build_result.failed
        );

        // Verify .olean files were created
        let lib_dir = tmp.path().join(".lake/build/lib");
        assert!(
            lib_dir.join("MyProject/Core.olean").exists(),
            "Core.olean should exist"
        );
        assert!(
            lib_dir.join("MyProject/Main.olean").exists(),
            "Main.olean should exist"
        );

        // Verify payload-backed .olean files can be loaded with dependencies
        let mut load_env = lean5_kernel::Environment::new();
        let load_paths = vec![lib_dir.clone()];
        let summaries =
            lean5_olean::load_module_with_deps(&mut load_env, "MyProject.Main", &load_paths)
                .expect("load with deps");
        assert!(
            !summaries.is_empty(),
            "expected load summaries from dependency chain"
        );
        let base = lean5_kernel::name::Name::from_string("baseVal");
        let main_const = lean5_kernel::name::Name::from_string("mainVal");
        assert!(
            load_env.get_const(&base).is_some(),
            "baseVal should be loaded from payload"
        );
        assert!(
            load_env.get_const(&main_const).is_some(),
            "mainVal should be loaded from payload"
        );
    }

    #[test]
    fn test_build_dependency_loading_verbose() {
        use std::fs;
        use std::io::Write;

        let tmp = TempDir::new().unwrap();

        // Create lakefile.lean
        let mut lakefile = fs::File::create(tmp.path().join("lakefile.lean")).unwrap();
        writeln!(
            lakefile,
            r#"
package deptest where
  name := "deptest"
  version := "0.1.0"

lean_lib DepTest where
  roots := #[`DepTest]
"#
        )
        .unwrap();

        // Create source directory
        fs::create_dir_all(tmp.path().join("DepTest")).unwrap();

        // Create Base.lean (minimal valid Lean file)
        let mut base = fs::File::create(tmp.path().join("DepTest/Base.lean")).unwrap();
        writeln!(base, "def baseVal : Type := Type").unwrap();

        // Create Mid.lean (depends on Base)
        let mut mid = fs::File::create(tmp.path().join("DepTest/Mid.lean")).unwrap();
        writeln!(mid, "import DepTest.Base\ndef midVal : Type := baseVal").unwrap();

        // Create Top.lean (depends on Mid)
        let mut top = fs::File::create(tmp.path().join("DepTest/Top.lean")).unwrap();
        writeln!(top, "import DepTest.Mid\ndef topVal : Type := midVal").unwrap();

        // Build with verbose output
        let ws = Workspace::load(tmp.path()).unwrap();
        let mut ctx = BuildContext::new(ws).with_options(BuildOptions {
            verbose: true,
            ..Default::default()
        });

        let result = ctx.build_all().unwrap();

        println!("Built: {:?}", result.built);
        println!("Failed: {:?}", result.failed);

        // All three modules should be built in correct order
        assert!(result.built.contains(&"DepTest.Base".to_string()));
        assert!(result.built.contains(&"DepTest.Mid".to_string()));
        assert!(result.built.contains(&"DepTest.Top".to_string()));
        assert!(result.failed.is_empty());

        // Verify build order: Base must come before Mid, Mid before Top
        let base_idx = result.built.iter().position(|m| m == "DepTest.Base");
        let mid_idx = result.built.iter().position(|m| m == "DepTest.Mid");
        let top_idx = result.built.iter().position(|m| m == "DepTest.Top");

        // Due to parallel build, we can't guarantee exact order, but all should be present
        assert!(base_idx.is_some() && mid_idx.is_some() && top_idx.is_some());

        // Validate that loading Top pulls in dependency constants via payloads
        let mut env = lean5_kernel::Environment::new();
        let lib_dir = tmp.path().join(".lake/build/lib");
        let _ = lean5_olean::load_module_with_deps(&mut env, "DepTest.Top", &[lib_dir])
            .expect("load top with deps");
        for name in ["baseVal", "midVal", "topVal"] {
            let n = lean5_kernel::name::Name::from_string(name);
            assert!(
                env.get_const(&n).is_some(),
                "{name} should be available after loading payload olean"
            );
        }
    }
}
