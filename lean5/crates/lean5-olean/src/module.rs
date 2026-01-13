//! Module data parsing from .olean files
//!
//! The .olean file structure at the high level:
//!
//! ```text
//! Offset 0-55:   Header (magic, version, git hash, base_addr)
//! Offset 56:     Root pointer to ModuleData object
//! Offset 64+:    Compacted region (serialized objects)
//! ```
//!
//! ModuleData (structure that stores a module's compiled data):
//!
//! ```text
//! structure ModuleData where
//!   constNames   : Array Name           -- exported constant names
//!   constants    : Array ConstantInfo   -- constant definitions
//!   extraConstNames : Array Name        -- extra constants (from codegen)
//!   entries      : Array (Name × Array (Name × DataValue)) -- extension entries
//!   modIdx       : Nat                  -- module index
//!   imports      : Array Import         -- import declarations
//! ```

use crate::error::{OleanError, OleanResult};
use crate::expr::ParsedExpr;
use crate::payload::{decode_lean5_payload, Lean5Payload};
use crate::region::{is_ptr, is_scalar, tags, unbox_scalar, CompactedRegion};

/// Import data
#[derive(Debug, Clone)]
pub struct ParsedImport {
    pub module_name: String,
    pub runtime_only: bool,
}

/// A parsed constant from the module
#[derive(Debug, Clone)]
pub struct ParsedConstant {
    /// Full name of the constant
    pub name: String,
    /// Kind of constant
    pub kind: ConstantKind,
    /// Universe parameter names
    pub level_params: Vec<String>,
    /// Type of the constant
    pub type_: Option<ParsedExpr>,
    /// Value (for definitions, theorems)
    pub value: Option<ParsedExpr>,
    /// Extra data for inductive types
    pub inductive_val: Option<InductiveValData>,
    /// Extra data for constructors
    pub constructor_val: Option<ConstructorValData>,
    /// Extra data for recursors
    pub recursor_val: Option<RecursorValData>,
}

/// Extra data from InductiveVal
#[derive(Debug, Clone)]
pub struct InductiveValData {
    pub num_params: u32,
    pub num_indices: u32,
    /// Names of all inductives in mutual group
    pub all: Vec<String>,
    /// Constructor names
    pub ctors: Vec<String>,
    pub is_rec: bool,
    pub is_unsafe: bool,
    pub is_reflexive: bool,
    pub is_nested: bool,
}

/// Extra data from ConstructorVal
#[derive(Debug, Clone)]
pub struct ConstructorValData {
    /// Name of the inductive type
    pub induct: String,
    /// Constructor index
    pub cidx: u32,
    pub num_params: u32,
    pub num_fields: u32,
    pub is_unsafe: bool,
}

/// Extra data from RecursorVal
#[derive(Debug, Clone)]
pub struct RecursorValData {
    /// Names of all inductives in mutual group
    pub all: Vec<String>,
    pub num_params: u32,
    pub num_indices: u32,
    pub num_motives: u32,
    pub num_minors: u32,
    /// Recursor rules for each constructor
    pub rules: Vec<RecursorRuleData>,
    pub k: bool,
    pub is_unsafe: bool,
}

/// A recursor rule for a constructor
#[derive(Debug, Clone)]
pub struct RecursorRuleData {
    pub ctor: String,
    pub num_fields: u32,
    pub rhs: Option<ParsedExpr>,
}

/// Kind of constant
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantKind {
    Axiom,
    Definition,
    Theorem,
    Opaque,
    Quot,
    Inductive,
    Constructor,
    Recursor,
}

/// Parsed module data from .olean file
#[derive(Debug, Clone)]
pub struct ParsedModule {
    /// Constant names exported by this module
    pub const_names: Vec<String>,
    /// Constant definitions
    pub constants: Vec<ParsedConstant>,
    /// Extra constant names (codegen)
    pub extra_const_names: Vec<String>,
    /// Import declarations
    pub imports: Vec<ParsedImport>,
    /// Module index
    pub mod_idx: u64,
    /// Optional Lean5 payload attached to the file
    pub lean5_payload: Option<Lean5Payload>,
}

impl<'a> CompactedRegion<'a> {
    /// Get the root pointer (first pointer after the header, at offset 56)
    pub fn root_ptr(&self) -> OleanResult<u64> {
        self.read_u64_at(56)
    }

    /// Read the ModuleData structure from the root object
    ///
    /// Lean 4 ModuleData layout (from Environment.lean):
    /// - Field 0: imports (Array Import)
    /// - Field 1: constNames (Array Name)
    /// - Field 2: constants (Array ConstantInfo)
    /// - Field 3: extraConstNames (Array Name)
    /// - Field 4: entries (Array (Name × Array EnvExtensionEntry))
    pub fn read_module_data(&self) -> OleanResult<ParsedModule> {
        let root_ptr = self.root_ptr()?;

        if !is_ptr(root_ptr) {
            return Err(OleanError::Region("Invalid root pointer".into()));
        }

        let root_offset = self.ptr_to_offset(root_ptr)?;
        let header = self.read_header_at(root_offset)?;

        let num_fields = header.other as usize;
        let field_offset = root_offset + 8; // Skip header

        let mut imports = Vec::new();
        let mut const_names = Vec::new();
        let mut constants = Vec::new();
        let mut extra_const_names = Vec::new();
        let mod_idx = 0u64;

        // Field 0: imports (Array Import)
        if num_fields >= 1 {
            let imports_ptr = self.read_u64_at(field_offset)?;
            imports = self.read_import_array(imports_ptr)?;
        }

        // Field 1: constNames (Array Name)
        if num_fields >= 2 {
            let const_names_ptr = self.read_u64_at(field_offset + 8)?;
            const_names = self.read_name_array_from_names(const_names_ptr)?;
        }

        // Field 2: constants (Array ConstantInfo)
        if num_fields >= 3 {
            let constants_ptr = self.read_u64_at(field_offset + 16)?;
            constants = self.read_constant_array_v2(constants_ptr)?;
        }

        // Field 3: extraConstNames (Array Name)
        if num_fields >= 4 {
            let extra_ptr = self.read_u64_at(field_offset + 24)?;
            extra_const_names = self.read_name_array_from_names(extra_ptr)?;
        }

        // Field 4: entries (skip for now - environment extensions)

        Ok(ParsedModule {
            const_names,
            constants,
            extra_const_names,
            imports,
            mod_idx,
            lean5_payload: decode_lean5_payload(self.data)?,
        })
    }

    /// Read only the imports from a module, skipping all constant parsing.
    ///
    /// This is much faster than `read_module_data` when you only need imports
    /// (e.g., for dependency graph discovery).
    pub fn read_imports_only(&self) -> OleanResult<Vec<ParsedImport>> {
        let root_ptr = self.root_ptr()?;

        if !is_ptr(root_ptr) {
            return Err(OleanError::Region("Invalid root pointer".into()));
        }

        let root_offset = self.ptr_to_offset(root_ptr)?;
        let header = self.read_header_at(root_offset)?;

        let num_fields = header.other as usize;
        let field_offset = root_offset + 8; // Skip header

        // Field 0: imports (Array Import)
        if num_fields >= 1 {
            let imports_ptr = self.read_u64_at(field_offset)?;
            self.read_import_array(imports_ptr)
        } else {
            Ok(Vec::new())
        }
    }

    /// Read an array of Import structures
    fn read_import_array(&self, ptr: u64) -> OleanResult<Vec<ParsedImport>> {
        if !is_ptr(ptr) {
            return Ok(Vec::new());
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        if header.tag != tags::ARRAY && header.tag != tags::STRUCT_ARRAY {
            return Ok(Vec::new());
        }

        let size = self.read_u64_at(offset + 8)? as usize;
        let mut imports = Vec::with_capacity(size);

        for i in 0..size {
            let elem_ptr = self.read_u64_at(offset + 24 + i * 8)?;
            if is_ptr(elem_ptr) {
                if let Ok(import) = self.read_import(elem_ptr) {
                    imports.push(import);
                }
            }
        }

        Ok(imports)
    }

    /// Read a single Import structure
    ///
    /// Import layout: { module: Name, runtimeOnly: Bool }
    /// - tag=0, fields=1 (the Name pointer)
    /// - Bool is stored as scalar data after the pointer fields
    fn read_import(&self, ptr: u64) -> OleanResult<ParsedImport> {
        if !is_ptr(ptr) {
            return Err(OleanError::Region("Invalid import pointer".into()));
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        // Read the module name (first field)
        let module_name = if header.other >= 1 {
            let name_ptr = self.read_u64_at(offset + 8)?;
            self.resolve_name_ptr_safe(name_ptr)
        } else {
            String::new()
        };

        // Read runtimeOnly (scalar byte after pointer fields)
        // cs_sz includes the 8-byte header, so scalar data starts at offset + 8 + num_fields * 8
        let scalar_offset = offset + 8 + (header.other as usize * 8);
        let runtime_only = if scalar_offset < self.data.len() {
            self.data[scalar_offset] != 0
        } else {
            false
        };

        Ok(ParsedImport {
            module_name,
            runtime_only,
        })
    }

    /// Read an array of Name objects (where elements are actual Name.str objects)
    fn read_name_array_from_names(&self, ptr: u64) -> OleanResult<Vec<String>> {
        if !is_ptr(ptr) {
            return Ok(Vec::new());
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        if header.tag != tags::ARRAY && header.tag != tags::STRUCT_ARRAY {
            return Ok(Vec::new());
        }

        let size = self.read_u64_at(offset + 8)? as usize;
        let mut names = Vec::with_capacity(size);

        for i in 0..size {
            let elem_ptr = self.read_u64_at(offset + 24 + i * 8)?;
            if is_ptr(elem_ptr) {
                let elem_off = self.ptr_to_offset(elem_ptr)?;
                if let Ok(name) = self.read_name_at(elem_off) {
                    names.push(name);
                }
            } else if is_scalar(elem_ptr) {
                // Name.anonymous
                names.push(String::new());
            }
        }

        Ok(names)
    }

    /// Read an array of ConstantInfo (v2 - based on actual structure)
    fn read_constant_array_v2(&self, ptr: u64) -> OleanResult<Vec<ParsedConstant>> {
        if !is_ptr(ptr) {
            return Ok(Vec::new());
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        if header.tag != tags::ARRAY && header.tag != tags::STRUCT_ARRAY {
            return Ok(Vec::new());
        }

        let size = self.read_u64_at(offset + 8)? as usize;
        let mut constants = Vec::with_capacity(size);

        for i in 0..size {
            let const_ptr = self.read_u64_at(offset + 24 + i * 8)?;
            if let Ok(constant) = self.read_constant_info_v2(const_ptr) {
                constants.push(constant);
            }
        }

        Ok(constants)
    }

    /// Read a single ConstantInfo (v2)
    ///
    /// Structure observed from Init/Prelude.olean:
    /// - Outer wrapper: tag=1, 1 field (pointing to inner)
    /// - Inner: tag=0, 4 fields containing:
    ///   - Field 0: XxxVal (the actual constant data) - tag=0, 3 fields
    ///   - Field 1: some metadata
    ///   - Field 2: scalar
    ///   - Field 3: name reference
    /// - XxxVal: tag=0, 3 fields:
    ///   - Field 0: Name (constant name)
    ///   - Field 1: List Name (level params)
    ///   - Field 2: Expr (type)
    ///   - (for defn/thm) Field 3: Expr (value)
    fn read_constant_info_v2(&self, ptr: u64) -> OleanResult<ParsedConstant> {
        if !is_ptr(ptr) {
            return Err(OleanError::Region("Invalid constant pointer".into()));
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        // ConstantInfo is an inductive with tags 0-7 for the different variants.
        // Each variant has 1 field pointing to the XxxVal.
        //
        // Lean 4 ConstantInfo constructor order:
        // 0 = axiomInfo, 1 = defnInfo, 2 = thmInfo, 3 = opaqueInfo,
        // 4 = quotInfo, 5 = inductInfo, 6 = ctorInfo, 7 = recInfo
        let kind = match header.tag {
            1 => ConstantKind::Definition,
            2 => ConstantKind::Theorem,
            3 => ConstantKind::Opaque,
            4 => ConstantKind::Quot,
            5 => ConstantKind::Inductive,
            6 => ConstantKind::Constructor,
            7 => ConstantKind::Recursor,
            _ => ConstantKind::Axiom, // 0 or unknown defaults to Axiom
        };

        // The XxxVal is the first (and only) field
        // ConstantInfo variants have 1 field pointing to the XxxVal
        let val_ptr = self.read_u64_at(offset + 8)?;
        if !is_ptr(val_ptr) {
            return Err(OleanError::Region("Invalid XxxVal pointer".into()));
        }

        let val_offset = self.ptr_to_offset(val_ptr)?;
        let val_header = self.read_header_at(val_offset)?;

        // XxxVal layout depends on the type:
        // - AxiomVal, QuotVal: { name, levelParams, type }
        // - DefinitionVal, TheoremVal, OpaqueVal: { name, levelParams, type, value, ... }
        // - InductiveVal, ConstructorVal, RecursorVal: { name, levelParams, type, ... }
        //
        // All XxxVal types start with: header(8) + name(8) + levelParams(8) + type(8)
        // For structures, fields are contiguous after header.
        //
        // However, val_header.other tells us how many pointer fields there are.
        // We need to check if this structure is correct.

        // XxxVal structures use inheritance from ConstantVal.
        // In Lean 4's object layout, this means:
        // - Field 0 is a pointer to the parent ConstantVal (or embedded inline)
        // - ConstantVal has fields: name, levelParams, type
        //
        // So we need to follow the ConstantVal pointer first.
        let const_val_ptr = self.read_u64_at(val_offset + 8)?;

        // Determine if this is a pointer to ConstantVal or if fields are inline
        let (name_ptr, level_params_ptr, type_ptr) = if is_ptr(const_val_ptr) {
            // ConstantVal is pointed to
            let cv_offset = self.ptr_to_offset(const_val_ptr)?;
            let name = self.read_u64_at(cv_offset + 8)?;
            let level_params = self.read_u64_at(cv_offset + 16)?;
            let type_ = self.read_u64_at(cv_offset + 24)?;
            (name, level_params, type_)
        } else {
            // Fields are inline (shouldn't happen, but fallback)
            let name = const_val_ptr;
            let level_params = self.read_u64_at(val_offset + 16)?;
            let type_ = self.read_u64_at(val_offset + 24)?;
            (name, level_params, type_)
        };

        let name = self.resolve_name_ptr_safe(name_ptr);
        let level_params = self.read_level_param_names(level_params_ptr)?;

        let type_ = if is_ptr(type_ptr) {
            if let Ok(type_off) = self.ptr_to_offset(type_ptr) {
                self.read_expr_at(type_off).ok()
            } else {
                None
            }
        } else {
            None
        };

        // For definitions and theorems, the value is at field 1 (after ConstantVal pointer)
        // XxxVal structure for defn/thm/opaque:
        //   Field 0: ConstantVal pointer (+8)
        //   Field 1: value Expr (+16)
        //   Field 2: all list (+24) (or hints for defn)
        // Only read value for types that have it: defn (1), thm (2), opaque (3)
        let needs_value = matches!(
            kind,
            ConstantKind::Definition | ConstantKind::Theorem | ConstantKind::Opaque
        );
        let value = if needs_value && val_header.other >= 2 {
            let value_ptr = self.read_u64_at(val_offset + 16)?;
            if is_ptr(value_ptr) {
                if let Ok(val_off) = self.ptr_to_offset(value_ptr) {
                    self.read_expr_at(val_off).ok()
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Parse extra fields based on kind
        let (inductive_val, constructor_val, recursor_val) = match kind {
            ConstantKind::Inductive => {
                let ind_val = self.read_inductive_val_data(val_offset)?;
                (Some(ind_val), None, None)
            }
            ConstantKind::Constructor => {
                let ctor_val = self.read_constructor_val_data(val_offset)?;
                (None, Some(ctor_val), None)
            }
            ConstantKind::Recursor => {
                let rec_val = self.read_recursor_val_data(val_offset)?;
                (None, None, Some(rec_val))
            }
            _ => (None, None, None),
        };

        Ok(ParsedConstant {
            name,
            kind,
            level_params,
            type_,
            value,
            inductive_val,
            constructor_val,
            recursor_val,
        })
    }

    /// Read InductiveVal extra data
    /// InductiveVal extends ConstantVal with these additional fields:
    ///   numParams, numIndices, all, ctors, numNested, isRec, isUnsafe, isReflexive
    ///
    /// Actual observed layout (with inheritance and scalar inline):
    ///   +8:  toConstantVal ptr
    ///   +16: numParams (inline scalar Nat)
    ///   +24: numIndices (inline scalar Nat)
    ///   +32: all (List Name ptr)
    ///   +40: ctors (List Name ptr)
    ///   +48: numNested (inline scalar Nat)
    ///   +56: padding or bool flags
    ///   +64: more data...
    fn read_inductive_val_data(&self, val_offset: usize) -> OleanResult<InductiveValData> {
        // Observed layout from debug output:
        // +16 and +24 are scalar Nats (numParams, numIndices)
        // +32 and +40 are pointer fields (all, ctors)
        // +48 is scalar Nat (numNested)
        // Bools follow after

        let num_params = self.read_u32_at(val_offset + 16, "numParams")?;
        let num_indices = self.read_u32_at(val_offset + 24, "numIndices")?;

        let all_ptr = self.read_u64_at(val_offset + 32)?;
        let all = if is_ptr(all_ptr) {
            self.read_name_list(all_ptr)?
        } else {
            Vec::new()
        };

        let ctors_ptr = self.read_u64_at(val_offset + 40)?;
        let ctors = if is_ptr(ctors_ptr) {
            self.read_name_list(ctors_ptr)?
        } else {
            Vec::new()
        };

        let _num_nested = self.read_nat_at(val_offset + 48)?;

        // Bool flags - they're packed as individual bytes, not as Lean scalars
        // In Lean 4 runtime, Bool is a UInt8 (0 or 1) stored directly
        // Look at raw bytes at +56, +57, +58
        let is_rec = self.data.get(val_offset + 56).copied().unwrap_or(0) != 0;
        let is_unsafe = self.data.get(val_offset + 57).copied().unwrap_or(0) != 0;
        let is_reflexive = self.data.get(val_offset + 58).copied().unwrap_or(0) != 0;

        Ok(InductiveValData {
            num_params,
            num_indices,
            all,
            ctors,
            is_rec,
            is_unsafe,
            is_reflexive,
            is_nested: false,
        })
    }

    /// Read ConstructorVal extra data
    /// Layout (after ConstantVal pointer at +8):
    ///   +16: induct (Name)
    ///   +24: cidx (Nat, scalar)
    ///   +32: numParams (Nat, scalar)
    ///   +40: numFields (Nat, scalar)
    ///   +48: isUnsafe (Bool, scalar)
    fn read_constructor_val_data(&self, val_offset: usize) -> OleanResult<ConstructorValData> {
        let induct_ptr = self.read_u64_at(val_offset + 16)?;
        let induct = self.resolve_name_ptr_safe(induct_ptr);

        let cidx = self.read_u32_at(val_offset + 24, "cidx")?;
        let num_params = self.read_u32_at(val_offset + 32, "numParams")?;
        let num_fields = self.read_u32_at(val_offset + 40, "numFields")?;
        let is_unsafe = self.read_bool_at(val_offset + 48)?;

        Ok(ConstructorValData {
            induct,
            cidx,
            num_params,
            num_fields,
            is_unsafe,
        })
    }

    /// Read RecursorVal extra data
    /// Layout (after ConstantVal pointer at +8):
    ///   +16: all (List Name)
    ///   +24: numParams (Nat, scalar)
    ///   +32: numIndices (Nat, scalar)
    ///   +40: numMotives (Nat, scalar)
    ///   +48: numMinors (Nat, scalar)
    ///   +56: rules (List RecursorRule)
    ///   +64: k (Bool, scalar)
    ///   +72: isUnsafe (Bool, scalar)
    fn read_recursor_val_data(&self, val_offset: usize) -> OleanResult<RecursorValData> {
        let all_ptr = self.read_u64_at(val_offset + 16)?;
        let all = self.read_name_list(all_ptr)?;

        let num_params = self.read_u32_at(val_offset + 24, "numParams")?;
        let num_indices = self.read_u32_at(val_offset + 32, "numIndices")?;
        let num_motives = self.read_u32_at(val_offset + 40, "numMotives")?;
        let num_minors = self.read_u32_at(val_offset + 48, "numMinors")?;

        let rules_ptr = self.read_u64_at(val_offset + 56)?;
        let rules = self.read_recursor_rules(rules_ptr)?;

        let k = self.read_bool_at(val_offset + 64)?;
        let is_unsafe = self.read_bool_at(val_offset + 72)?;

        Ok(RecursorValData {
            all,
            num_params,
            num_indices,
            num_motives,
            num_minors,
            rules,
            k,
            is_unsafe,
        })
    }

    /// Read a list of RecursorRule
    fn read_recursor_rules(&self, ptr: u64) -> OleanResult<Vec<RecursorRuleData>> {
        let mut rules = Vec::new();
        let mut current_ptr = ptr;

        for _ in 0..1000 {
            // limit iterations
            if is_scalar(current_ptr) || !is_ptr(current_ptr) {
                break;
            }

            let offset = self.ptr_to_offset(current_ptr)?;
            let header = self.read_header_at(offset)?;

            match (header.tag, header.other) {
                (1, 2) => {
                    // cons
                    let head_ptr = self.read_u64_at(offset + 8)?;
                    let tail_ptr = self.read_u64_at(offset + 16)?;

                    if is_ptr(head_ptr) {
                        let rule = self.read_recursor_rule(head_ptr)?;
                        rules.push(rule);
                    }
                    current_ptr = tail_ptr;
                }
                _ => break, // nil or unknown
            }
        }

        Ok(rules)
    }

    /// Read a single RecursorRule
    /// Layout:
    ///   +8: ctor (Name)
    ///   +16: nfields (Nat, scalar)
    ///   +24: rhs (Expr)
    fn read_recursor_rule(&self, ptr: u64) -> OleanResult<RecursorRuleData> {
        let offset = self.ptr_to_offset(ptr)?;

        let ctor_ptr = self.read_u64_at(offset + 8)?;
        let ctor = self.resolve_name_ptr_safe(ctor_ptr);

        let num_fields = self.read_u32_at(offset + 16, "nfields")?;

        let rhs_ptr = self.read_u64_at(offset + 24)?;
        let rhs = if is_ptr(rhs_ptr) {
            if let Ok(rhs_off) = self.ptr_to_offset(rhs_ptr) {
                self.read_expr_at(rhs_off).ok()
            } else {
                None
            }
        } else {
            None
        };

        Ok(RecursorRuleData {
            ctor,
            num_fields,
            rhs,
        })
    }

    /// Read a Nat (scalar) at offset
    fn read_nat_at(&self, offset: usize) -> OleanResult<u64> {
        let val = self.read_u64_at(offset)?;
        if is_scalar(val) {
            Ok(unbox_scalar(val))
        } else {
            // Big nat - read the actual value if needed
            // For now, just return 0 as a fallback
            Ok(0)
        }
    }

    /// Read a u32 from a Nat, with overflow checking
    fn read_u32_at(&self, offset: usize, field: &str) -> OleanResult<u32> {
        let val = self.read_nat_at(offset)?;
        u32::try_from(val)
            .map_err(|_| OleanError::Region(format!("{field} value too large: {val}")))
    }

    /// Read a Bool (scalar) at offset
    fn read_bool_at(&self, offset: usize) -> OleanResult<bool> {
        let val = self.read_u64_at(offset)?;
        if is_scalar(val) {
            Ok(unbox_scalar(val) != 0)
        } else {
            Ok(false)
        }
    }

    /// Read a list of names
    fn read_name_list(&self, ptr: u64) -> OleanResult<Vec<String>> {
        let mut names = Vec::new();
        let mut current_ptr = ptr;

        for _ in 0..10000 {
            // limit iterations
            if is_scalar(current_ptr) || !is_ptr(current_ptr) {
                break;
            }

            let offset = self.ptr_to_offset(current_ptr)?;
            let header = self.read_header_at(offset)?;

            match (header.tag, header.other) {
                (1, 2) => {
                    // cons
                    let head_ptr = self.read_u64_at(offset + 8)?;
                    let tail_ptr = self.read_u64_at(offset + 16)?;

                    let name = self.resolve_name_ptr_safe(head_ptr);
                    names.push(name);
                    current_ptr = tail_ptr;
                }
                _ => break, // nil or unknown
            }
        }

        Ok(names)
    }

    /// Resolve a name pointer safely (returns empty string on error)
    fn resolve_name_ptr_safe(&self, ptr: u64) -> String {
        if is_scalar(ptr) {
            return String::new();
        }
        if !is_ptr(ptr) {
            return String::new();
        }
        match self.ptr_to_offset(ptr) {
            Ok(offset) => self.read_name_at(offset).unwrap_or_default(),
            Err(_) => String::new(),
        }
    }

    /// Read level parameter names (list of names)
    fn read_level_param_names(&self, ptr: u64) -> OleanResult<Vec<String>> {
        self.read_name_list(ptr)
    }

    /// Get basic statistics about the root object
    pub fn analyze_root(&self) -> OleanResult<RootAnalysis> {
        let root_ptr = self.root_ptr()?;

        if is_scalar(root_ptr) {
            return Err(OleanError::Region(format!(
                "Root is scalar: {}",
                unbox_scalar(root_ptr)
            )));
        }

        if !is_ptr(root_ptr) {
            return Err(OleanError::Region("Root is null".into()));
        }

        let root_offset = self.ptr_to_offset(root_ptr)?;
        let header = self.read_header_at(root_offset)?;

        // Read the first several pointers after the root header
        let mut field_info = Vec::new();
        for i in 0..8 {
            let field_offset = root_offset + 8 + i * 8;
            if field_offset + 8 > self.data.len() {
                break;
            }
            let ptr = self.read_u64_at(field_offset)?;
            let kind = if is_scalar(ptr) {
                format!("scalar({})", unbox_scalar(ptr))
            } else if is_ptr(ptr) {
                if let Ok(off) = self.ptr_to_offset(ptr) {
                    if let Ok(h) = self.read_header_at(off) {
                        format!("ptr->tag{}/{}fields", h.tag, h.other)
                    } else {
                        "ptr->invalid".to_string()
                    }
                } else {
                    "ptr->out_of_bounds".to_string()
                }
            } else {
                "null".to_string()
            };
            field_info.push((i, ptr, kind));
        }

        Ok(RootAnalysis {
            root_ptr,
            root_offset,
            tag: header.tag,
            num_fields: header.other,
            cs_sz: header.cs_sz,
            field_info,
        })
    }
}

/// Analysis of the root object
#[derive(Debug)]
pub struct RootAnalysis {
    pub root_ptr: u64,
    pub root_offset: usize,
    pub tag: u8,
    pub num_fields: u8,
    pub cs_sz: u16,
    pub field_info: Vec<(usize, u64, String)>,
}

/// Analysis of an array in the module
#[derive(Debug)]
pub struct ArrayAnalysis {
    pub size: usize,
    pub sample_elements: Vec<ElementInfo>,
}

/// Information about an array element
#[derive(Debug)]
pub struct ElementInfo {
    pub index: usize,
    pub tag: u8,
    pub num_fields: u8,
    pub description: String,
}

impl<'a> CompactedRegion<'a> {
    /// Analyze an array at a given pointer
    pub fn analyze_array(&self, ptr: u64, max_samples: usize) -> OleanResult<ArrayAnalysis> {
        if !is_ptr(ptr) {
            return Err(OleanError::Region("Not a pointer".into()));
        }

        let offset = self.ptr_to_offset(ptr)?;
        let header = self.read_header_at(offset)?;

        if header.tag != tags::ARRAY && header.tag != tags::STRUCT_ARRAY {
            return Err(OleanError::Region(format!(
                "Not an array (tag={})",
                header.tag
            )));
        }

        let size = self.read_u64_at(offset + 8)? as usize;
        let mut sample_elements = Vec::new();

        for i in 0..size.min(max_samples) {
            let elem_ptr = self.read_u64_at(offset + 24 + i * 8)?;

            let description = if is_scalar(elem_ptr) {
                format!("scalar({})", unbox_scalar(elem_ptr))
            } else if is_ptr(elem_ptr) {
                if let Ok(elem_off) = self.ptr_to_offset(elem_ptr) {
                    if let Ok(elem_header) = self.read_header_at(elem_off) {
                        // Try to identify what kind of object this is
                        match elem_header.tag {
                            tags::STRING => {
                                if let Ok(s) = self.read_lean_string_at(elem_off) {
                                    format!(
                                        "String(\"{}\")",
                                        s.chars().take(30).collect::<String>()
                                    )
                                } else {
                                    "String(?)".to_string()
                                }
                            }
                            0..=7 => {
                                // Could be a constant info or other constructor
                                // Try reading as Name first
                                if elem_header.other <= 2 {
                                    if let Ok(name) = self.read_name_at(elem_off) {
                                        format!(
                                            "ctor{}/{}(name={})",
                                            elem_header.tag, elem_header.other, name
                                        )
                                    } else {
                                        format!("ctor{}/{}", elem_header.tag, elem_header.other)
                                    }
                                } else {
                                    format!("ctor{}/{}", elem_header.tag, elem_header.other)
                                }
                            }
                            tags::ARRAY | tags::STRUCT_ARRAY => {
                                let arr_size = self.read_u64_at(elem_off + 8).unwrap_or(0);
                                format!("Array(size={arr_size})")
                            }
                            _ => format!("tag{}/{}", elem_header.tag, elem_header.other),
                        }
                    } else {
                        "invalid".to_string()
                    }
                } else {
                    "out_of_bounds".to_string()
                }
            } else {
                "null".to_string()
            };

            let (tag, num_fields) = if is_ptr(elem_ptr) {
                if let Ok(elem_off) = self.ptr_to_offset(elem_ptr) {
                    if let Ok(h) = self.read_header_at(elem_off) {
                        (h.tag, h.other)
                    } else {
                        (255, 0)
                    }
                } else {
                    (255, 0)
                }
            } else {
                (255, 0)
            };

            sample_elements.push(ElementInfo {
                index: i,
                tag,
                num_fields,
                description,
            });
        }

        Ok(ArrayAnalysis {
            size,
            sample_elements,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_lean_lib_path() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let elan_path = std::path::PathBuf::from(home).join(".elan/toolchains");

        if elan_path.exists() {
            for entry in std::fs::read_dir(&elan_path).ok()? {
                let entry = entry.ok()?;
                let name = entry.file_name();
                if name.to_string_lossy().contains("lean4") {
                    return Some(entry.path().join("lib/lean"));
                }
            }
        }
        None
    }

    #[test]
    fn test_analyze_root_prelude() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        let analysis = region.analyze_root().expect("Failed to analyze root");

        println!("Root analysis for Init/Prelude.olean:");
        println!("  Root pointer: 0x{:x}", analysis.root_ptr);
        println!("  Root offset: {}", analysis.root_offset);
        println!("  Tag: {}", analysis.tag);
        println!("  Num fields: {}", analysis.num_fields);
        println!("  cs_sz: {}", analysis.cs_sz);
        println!("  Fields:");
        for (i, ptr, kind) in &analysis.field_info {
            println!("    Field {i}: 0x{ptr:x} -> {kind}");
        }
    }

    #[test]
    fn test_analyze_arrays_prelude() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        let root_ptr = region.root_ptr().expect("Failed to read root pointer");
        let root_offset = region
            .ptr_to_offset(root_ptr)
            .expect("Invalid root pointer");

        println!("\nAnalyzing arrays in Init/Prelude.olean root object:");

        // Read each field of the root object
        for i in 0..5 {
            let field_ptr = region
                .read_u64_at(root_offset + 8 + i * 8)
                .expect("Failed to read field");
            println!("\nField {i}:");

            if let Ok(analysis) = region.analyze_array(field_ptr, 5) {
                println!("  Array size: {}", analysis.size);
                println!("  Sample elements:");
                for elem in &analysis.sample_elements {
                    println!(
                        "    [{}] tag={}, fields={}: {}",
                        elem.index, elem.tag, elem.num_fields, elem.description
                    );
                }
            } else {
                println!("  Not an array or failed to analyze");
            }
        }
    }

    #[test]
    fn test_read_module_data_prelude() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        match region.read_module_data() {
            Ok(module) => {
                println!("Module data from Init/Prelude.olean:");
                println!("  Const names: {}", module.const_names.len());
                println!("  Constants: {}", module.constants.len());
                println!("  Extra const names: {}", module.extra_const_names.len());
                println!("  Imports: {}", module.imports.len());
                println!("  Module index: {}", module.mod_idx);

                if !module.const_names.is_empty() {
                    println!("\n  First 10 const names:");
                    for name in module.const_names.iter().take(10) {
                        println!("    - {name}");
                    }

                    // Check for Nat in const_names
                    let nat_names: Vec<_> = module
                        .const_names
                        .iter()
                        .filter(|n| *n == "Nat" || n.starts_with("Nat."))
                        .take(10)
                        .collect();
                    println!("\n  Nat-related in const_names: {nat_names:?}");

                    // Check for exact "Nat"
                    let has_nat = module.const_names.iter().any(|n| n == "Nat");
                    println!("\n  Has exact 'Nat' in const_names: {has_nat}");
                }

                // Check extra_const_names for Nat
                if !module.extra_const_names.is_empty() {
                    let nat_extra: Vec<_> = module
                        .extra_const_names
                        .iter()
                        .filter(|n| *n == "Nat" || n.starts_with("Nat."))
                        .take(10)
                        .collect();
                    println!("\n  Nat-related in extra_const_names: {nat_extra:?}");
                }

                if !module.constants.is_empty() {
                    println!("\n  First 10 constants:");
                    for c in module.constants.iter().take(10) {
                        println!("    - {} ({:?})", c.name, c.kind);
                    }
                }

                // Count by kind
                let mut by_kind: std::collections::HashMap<ConstantKind, usize> =
                    std::collections::HashMap::new();
                for c in &module.constants {
                    *by_kind.entry(c.kind.clone()).or_insert(0) += 1;
                }
                println!("\n  Constants by kind:");
                for (kind, count) in &by_kind {
                    println!("    {kind:?}: {count}");
                }
            }
            Err(e) => {
                println!("Failed to read module data: {e:?}");
            }
        }
    }

    #[test]
    fn test_read_core_olean_with_imports() {
        // Init/Core.olean has imports (imports Prelude, SizeOf)
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let core_path = lib_path.join("Init/Core.olean");
        if !core_path.exists() {
            eprintln!("Skipping test: Init/Core.olean not found");
            return;
        }

        let bytes = std::fs::read(&core_path).expect("Failed to read file");
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        let module = region
            .read_module_data()
            .expect("Failed to read module data");

        // Verify imports are parsed correctly
        assert_eq!(module.imports.len(), 2, "Init/Core should have 2 imports");

        let import_names: Vec<_> = module
            .imports
            .iter()
            .map(|i| i.module_name.as_str())
            .collect();
        assert!(
            import_names.contains(&"Init.Prelude"),
            "Should import Init.Prelude"
        );
        assert!(
            import_names.contains(&"Init.SizeOf"),
            "Should import Init.SizeOf"
        );

        // All imports should have runtime_only=false
        for imp in &module.imports {
            assert!(
                !imp.runtime_only,
                "Expected runtime_only=false for {}",
                imp.module_name
            );
        }

        // Verify constants are parsed
        assert_eq!(module.const_names.len(), 1416, "Expected 1416 const names");
        assert_eq!(module.constants.len(), 1416, "Expected 1416 constants");
    }

    #[test]
    fn test_read_init_olean_many_imports() {
        // Init.olean has many imports (all Init submodules)
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        // Init.olean is one level up from Init/
        let init_path = lib_path.join("Init.olean");
        if !init_path.exists() {
            eprintln!("Skipping test: Init.olean not found at {init_path:?}");
            return;
        }

        let bytes = std::fs::read(&init_path).expect("Failed to read file");
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        let module = region
            .read_module_data()
            .expect("Failed to read module data");

        // Init.olean should have many imports (31 in v4.13.0)
        assert!(
            module.imports.len() >= 30,
            "Expected ~31 imports in Init.olean, got {}",
            module.imports.len()
        );

        // Verify some expected imports
        let import_names: Vec<_> = module
            .imports
            .iter()
            .map(|i| i.module_name.as_str())
            .collect();
        assert!(
            import_names.contains(&"Init.Prelude"),
            "Should import Init.Prelude"
        );
        assert!(
            import_names.contains(&"Init.Core"),
            "Should import Init.Core"
        );
        assert!(
            import_names.contains(&"Init.Data"),
            "Should import Init.Data"
        );

        // Init.olean itself has no constants (it's just re-exports)
        assert_eq!(
            module.constants.len(),
            0,
            "Init.olean should have no direct constants"
        );
    }

    #[test]
    fn test_analyze_first_constant() {
        let Some(lib_path) = get_lean_lib_path() else {
            eprintln!("Skipping test: Lean 4 not found");
            return;
        };

        let prelude_path = lib_path.join("Init/Prelude.olean");
        if !prelude_path.exists() {
            return;
        }

        let bytes = std::fs::read(&prelude_path).expect("Failed to read file");
        let header = crate::parse_header(&bytes).expect("Failed to parse header");
        let region = CompactedRegion::new(&bytes, header.base_addr);

        let root_ptr = region.root_ptr().expect("Failed to read root pointer");
        let root_offset = region
            .ptr_to_offset(root_ptr)
            .expect("Invalid root pointer");

        // Field 2 contains constants array
        let constants_ptr = region.read_u64_at(root_offset + 8 + 16).unwrap();
        let constants_offset = region.ptr_to_offset(constants_ptr).unwrap();

        // Get first constant
        let first_const_ptr = region.read_u64_at(constants_offset + 24).unwrap();
        let first_const_offset = region.ptr_to_offset(first_const_ptr).unwrap();
        let first_header = region.read_header_at(first_const_offset).unwrap();

        println!("First constant wrapper:");
        println!(
            "  tag={}, other={}, cs_sz={}",
            first_header.tag, first_header.other, first_header.cs_sz
        );

        // The first field is the actual ConstantInfo
        let inner_ptr = region.read_u64_at(first_const_offset + 8).unwrap();
        println!("  Inner ptr: 0x{inner_ptr:x}");

        if is_ptr(inner_ptr) {
            let inner_offset = region.ptr_to_offset(inner_ptr).unwrap();
            let inner_header = region.read_header_at(inner_offset).unwrap();
            println!("  Inner object:");
            println!(
                "    tag={}, other={}, cs_sz={}",
                inner_header.tag, inner_header.other, inner_header.cs_sz
            );

            // Print the first few fields
            for i in 0..8 {
                let field_ptr = region.read_u64_at(inner_offset + 8 + i * 8).unwrap();
                let desc = if is_scalar(field_ptr) {
                    format!("scalar({})", unbox_scalar(field_ptr))
                } else if is_ptr(field_ptr) {
                    if let Ok(off) = region.ptr_to_offset(field_ptr) {
                        if let Ok(h) = region.read_header_at(off) {
                            if h.tag == crate::region::tags::STRING {
                                if let Ok(s) = region.read_lean_string_at(off) {
                                    format!("String(\"{s}\")")
                                } else {
                                    format!("tag{}/{}", h.tag, h.other)
                                }
                            } else if h.tag <= 2 && h.other <= 2 {
                                if let Ok(name) = region.read_name_at(off) {
                                    format!("Name({name})")
                                } else {
                                    format!("tag{}/{}", h.tag, h.other)
                                }
                            } else {
                                format!("tag{}/{}", h.tag, h.other)
                            }
                        } else {
                            "invalid".to_string()
                        }
                    } else {
                        "oob".to_string()
                    }
                } else {
                    "null".to_string()
                };
                println!("    Field {i}: 0x{field_ptr:x} -> {desc}");
            }

            // Follow Field 0 to see what's inside
            let field0_ptr = region.read_u64_at(inner_offset + 8).unwrap();
            if is_ptr(field0_ptr) {
                let field0_offset = region.ptr_to_offset(field0_ptr).unwrap();
                let field0_header = region.read_header_at(field0_offset).unwrap();
                println!(
                    "\n  Field 0 details (tag={}, other={}):",
                    field0_header.tag, field0_header.other
                );

                for i in 0..6 {
                    let ptr = region.read_u64_at(field0_offset + 8 + i * 8).unwrap();
                    let desc = if is_scalar(ptr) {
                        format!("scalar({})", unbox_scalar(ptr))
                    } else if is_ptr(ptr) {
                        if let Ok(off) = region.ptr_to_offset(ptr) {
                            if let Ok(h) = region.read_header_at(off) {
                                if h.tag == crate::region::tags::STRING {
                                    if let Ok(s) = region.read_lean_string_at(off) {
                                        format!("String(\"{s}\")")
                                    } else {
                                        format!("tag{}/{}", h.tag, h.other)
                                    }
                                } else if h.tag <= 2 && h.other <= 2 {
                                    if let Ok(name) = region.read_name_at(off) {
                                        format!("Name({name})")
                                    } else {
                                        format!("tag{}/{}", h.tag, h.other)
                                    }
                                } else {
                                    format!("tag{}/{}", h.tag, h.other)
                                }
                            } else {
                                "invalid".to_string()
                            }
                        } else {
                            "oob".to_string()
                        }
                    } else {
                        "null".to_string()
                    };
                    println!("      Field {i}: 0x{ptr:x} -> {desc}");
                }

                let type_ptr = region.read_u64_at(field0_offset + 24).unwrap();
                if is_ptr(type_ptr) {
                    let type_offset = region.ptr_to_offset(type_ptr).unwrap();
                    let type_header = region.read_header_at(type_offset).unwrap();
                    println!(
                        "  Type header tag={}, other={}, cs_sz={}",
                        type_header.tag, type_header.other, type_header.cs_sz
                    );
                    let field_base = type_offset + 8;
                    let mut type_fields = Vec::new();
                    for i in 0..type_header.other as usize {
                        let ptr = region.read_u64_at(field_base + i * 8).unwrap_or(0);
                        println!("    type field {i}: 0x{ptr:x}");
                        type_fields.push(ptr);
                        if let Ok(off) = region.ptr_to_offset(ptr) {
                            if let Ok(h) = region.read_header_at(off) {
                                println!(
                                    "      field {} header: tag={}, other={}, cs_sz={}",
                                    i, h.tag, h.other, h.cs_sz
                                );
                            }
                        }
                    }
                    let scalar_base = field_base + type_header.other as usize * 8;
                    if let Ok(bytes) = region.bytes_at(scalar_base, 1) {
                        println!("    binder info byte: {}", bytes[0]);
                    }
                    if let Some(ptr) = type_fields.get(1) {
                        if let Ok(off) = region.ptr_to_offset(*ptr) {
                            let sort_field_base = off + 8;
                            let sort_scalar_base = sort_field_base
                                + region.read_header_at(off).unwrap().other as usize * 8;
                            let raw_level_ptr = region.read_u64_at(sort_field_base).unwrap_or(0);
                            println!("    sort level raw ptr: 0x{raw_level_ptr:x}");
                            if let Ok(level_off) = region.ptr_to_offset(raw_level_ptr) {
                                if let Ok(h) = region.read_header_at(level_off) {
                                    println!(
                                        "    sort level header: tag={}, other={}, cs_sz={}",
                                        h.tag, h.other, h.cs_sz
                                    );
                                    let level_field_base = level_off + 8;
                                    if let Ok(pred_ptr) = region.read_u64_at(level_field_base) {
                                        println!("    sort level field0: 0x{pred_ptr:x}");
                                        if let Ok(pred_off) = region.ptr_to_offset(pred_ptr) {
                                            if let Ok(pred_header) = region.read_header_at(pred_off)
                                            {
                                                println!(
                                                    "    pred header: tag={}, other={}, cs_sz={}",
                                                    pred_header.tag,
                                                    pred_header.other,
                                                    pred_header.cs_sz
                                                );
                                                if let Ok(bytes) = region
                                                    .bytes_at(pred_off, pred_header.cs_sz as usize)
                                                {
                                                    println!("    pred raw bytes: {bytes:x?}");
                                                }
                                            }
                                        }
                                    }
                                    match region.read_level_at(level_off) {
                                        Ok(level) => println!("    parsed level: {level:?}"),
                                        Err(err) => println!("    level parse error: {err:?}"),
                                    }
                                }
                            }
                            if let Ok(bytes) = region.bytes_at(sort_scalar_base, 8) {
                                println!("    sort scalar bytes: {bytes:x?}");
                            }
                            match region.read_expr_at(off) {
                                Ok(expr) => println!("  Parsed binder type: {expr:?}"),
                                Err(err) => println!("  Binder type parse error: {err:?}"),
                            }
                        }
                    }
                    if let Some(ptr) = type_fields.get(2) {
                        if let Ok(off) = region.ptr_to_offset(*ptr) {
                            match region.read_expr_at(off) {
                                Ok(expr) => println!("  Parsed body expr: {expr:?}"),
                                Err(err) => println!("  Body parse error: {err:?}"),
                            }
                        }
                    }
                    match region.read_expr_at(type_offset) {
                        Ok(expr) => println!("\n  Parsed type expr for first constant: {expr:?}"),
                        Err(err) => {
                            println!("\n  Failed to parse type expr for first constant: {err:?}");
                        }
                    }
                }
            }
        }
    }
}
