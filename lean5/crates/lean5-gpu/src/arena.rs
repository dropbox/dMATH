//! GPU memory arena for expression storage
//!
//! Expressions are serialized to a flat GPU-friendly format. Each node is 16 bytes:
//! - 4 bytes: tag (expression variant)
//! - 4 bytes: data1 (payload depends on variant)
//! - 4 bytes: data2 (payload depends on variant)
//! - 4 bytes: data3 (payload depends on variant)
//!
//! This allows efficient GPU processing with coalesced memory access.

use bytemuck::{Pod, Zeroable};
use lean5_kernel::{BinderInfo, Expr, FVarId, Level, Literal, Name};
use std::collections::HashMap;
use std::sync::Arc;

/// GPU-friendly expression node (16 bytes, aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuExpr {
    /// Expression tag (see TAG_* constants)
    pub tag: u32,
    /// Payload data 1 - meaning depends on tag
    pub data1: u32,
    /// Payload data 2 - meaning depends on tag
    pub data2: u32,
    /// Payload data 3 - meaning depends on tag
    pub data3: u32,
}

// Expression tags (matches Expr variants)
pub const TAG_BVAR: u32 = 0;
pub const TAG_FVAR: u32 = 1;
pub const TAG_SORT: u32 = 2;
pub const TAG_CONST: u32 = 3;
pub const TAG_APP: u32 = 4;
pub const TAG_LAM: u32 = 5;
pub const TAG_PI: u32 = 6;
pub const TAG_LET: u32 = 7;
pub const TAG_LIT_NAT: u32 = 8;
pub const TAG_LIT_STR: u32 = 9;
pub const TAG_PROJ: u32 = 10;

// Special marker for "no expression" (null pointer equivalent)
pub const TAG_NONE: u32 = 0xFFFF_FFFF;

// Level tags
pub const LEVEL_TAG_ZERO: u32 = 0;
pub const LEVEL_TAG_SUCC: u32 = 1;
pub const LEVEL_TAG_MAX: u32 = 2;
pub const LEVEL_TAG_IMAX: u32 = 3;
pub const LEVEL_TAG_PARAM: u32 = 4;

/// GPU-friendly level node (16 bytes, aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuLevel {
    /// Level tag
    pub tag: u32,
    /// Payload data 1
    pub data1: u32,
    /// Payload data 2
    pub data2: u32,
    /// Reserved for alignment
    pub _pad: u32,
}

/// Binder info encoding (2 bits)
pub const BINDER_DEFAULT: u32 = 0;
pub const BINDER_IMPLICIT: u32 = 1;
pub const BINDER_STRICT_IMPLICIT: u32 = 2;
pub const BINDER_INST_IMPLICIT: u32 = 3;

impl GpuExpr {
    /// Create a new GPU expression node
    pub fn new(tag: u32, data1: u32, data2: u32, data3: u32) -> Self {
        Self {
            tag,
            data1,
            data2,
            data3,
        }
    }

    /// Create a bound variable node
    pub fn bvar(index: u32) -> Self {
        Self::new(TAG_BVAR, index, 0, 0)
    }

    /// Create a free variable node
    pub fn fvar(id: u32) -> Self {
        Self::new(TAG_FVAR, id, 0, 0)
    }

    /// Create a sort node (level_idx points to GpuLevel)
    pub fn sort(level_idx: u32) -> Self {
        Self::new(TAG_SORT, level_idx, 0, 0)
    }

    /// Create a constant node
    /// - name_idx: index into name table
    /// - levels_start: start index in level array
    /// - levels_count: number of universe levels
    pub fn const_(name_idx: u32, levels_start: u32, levels_count: u32) -> Self {
        Self::new(TAG_CONST, name_idx, levels_start, levels_count)
    }

    /// Create an application node
    /// - func_idx: index of function expression
    /// - arg_idx: index of argument expression
    pub fn app(func_idx: u32, arg_idx: u32) -> Self {
        Self::new(TAG_APP, func_idx, arg_idx, 0)
    }

    /// Create a lambda node
    /// - binder_info: encoded binder info
    /// - ty_idx: index of type expression
    /// - body_idx: index of body expression
    pub fn lam(binder_info: u32, ty_idx: u32, body_idx: u32) -> Self {
        Self::new(TAG_LAM, binder_info, ty_idx, body_idx)
    }

    /// Create a pi/forall node
    /// - binder_info: encoded binder info
    /// - ty_idx: index of type expression
    /// - body_idx: index of body expression
    pub fn pi(binder_info: u32, ty_idx: u32, body_idx: u32) -> Self {
        Self::new(TAG_PI, binder_info, ty_idx, body_idx)
    }

    /// Create a let binding node
    /// - ty_idx: index of type expression
    /// - val_idx: index of value expression
    /// - body_idx: index of body expression
    pub fn let_(ty_idx: u32, val_idx: u32, body_idx: u32) -> Self {
        Self::new(TAG_LET, ty_idx, val_idx, body_idx)
    }

    /// Create a nat literal node (stores lower 32 bits; for full 64-bit, use two nodes)
    pub fn lit_nat(value_low: u32, value_high: u32) -> Self {
        Self::new(TAG_LIT_NAT, value_low, value_high, 0)
    }

    /// Create a string literal node (references string table)
    pub fn lit_str(string_idx: u32) -> Self {
        Self::new(TAG_LIT_STR, string_idx, 0, 0)
    }

    /// Create a projection node
    /// - name_idx: index of structure name
    /// - field_idx: field index
    /// - expr_idx: index of expression
    pub fn proj(name_idx: u32, field_idx: u32, expr_idx: u32) -> Self {
        Self::new(TAG_PROJ, name_idx, field_idx, expr_idx)
    }

    /// Check if this is a none/null expression
    pub fn is_none(&self) -> bool {
        self.tag == TAG_NONE
    }
}

impl GpuLevel {
    /// Create a new GPU level node
    pub fn new(tag: u32, data1: u32, data2: u32) -> Self {
        Self {
            tag,
            data1,
            data2,
            _pad: 0,
        }
    }

    /// Create zero level
    pub fn zero() -> Self {
        Self::new(LEVEL_TAG_ZERO, 0, 0)
    }

    /// Create successor level (succ_idx points to inner level)
    pub fn succ(inner_idx: u32) -> Self {
        Self::new(LEVEL_TAG_SUCC, inner_idx, 0)
    }

    /// Create max level
    pub fn max(l1_idx: u32, l2_idx: u32) -> Self {
        Self::new(LEVEL_TAG_MAX, l1_idx, l2_idx)
    }

    /// Create imax level
    pub fn imax(l1_idx: u32, l2_idx: u32) -> Self {
        Self::new(LEVEL_TAG_IMAX, l1_idx, l2_idx)
    }

    /// Create param level (name_idx references name table)
    pub fn param(name_idx: u32) -> Self {
        Self::new(LEVEL_TAG_PARAM, name_idx, 0)
    }
}

/// Arena for building GPU expression buffers
///
/// This arena collects expressions, levels, and names during serialization.
/// After building, the buffers can be uploaded to the GPU.
pub struct GpuExprArena {
    /// Expression nodes
    exprs: Vec<GpuExpr>,
    /// Level nodes
    levels: Vec<GpuLevel>,
    /// Name table (interned names for deduplication)
    names: Vec<String>,
    /// String table (for string literals)
    strings: Vec<String>,
    /// Name deduplication map
    name_map: HashMap<String, u32>,
    /// Expression deduplication map (for common subexpressions)
    expr_cache: HashMap<ExprKey, u32>,
    /// Level deduplication map
    level_cache: HashMap<LevelKey, u32>,
}

/// Key for expression deduplication (simplified hash key)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum ExprKey {
    BVar(u32),
    FVar(u64),
    Sort(u32),
    Const(u32, Vec<u32>),
    App(u32, u32),
    Lam(u32, u32, u32),
    Pi(u32, u32, u32),
    Let(u32, u32, u32),
    LitNat(u64),
    LitStr(u32),
    Proj(u32, u32, u32),
}

/// Key for level deduplication
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum LevelKey {
    Zero,
    Succ(u32),
    Max(u32, u32),
    IMax(u32, u32),
    Param(u32),
}

impl Default for GpuExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuExprArena {
    /// Create a new empty arena
    pub fn new() -> Self {
        Self {
            exprs: Vec::new(),
            levels: Vec::new(),
            names: Vec::new(),
            strings: Vec::new(),
            name_map: HashMap::new(),
            expr_cache: HashMap::new(),
            level_cache: HashMap::new(),
        }
    }

    /// Convert usize to u32 with overflow checking.
    /// Returns u32::MAX if the value overflows (safe fallback for GPU indices).
    #[inline]
    fn usize_to_u32(value: usize) -> u32 {
        u32::try_from(value).unwrap_or(u32::MAX)
    }

    /// Convert u64 to u32 with overflow checking.
    /// Truncates to lower 32 bits if overflow occurs - intentional for FVarIds
    /// as most IDs are small and GPU only has 32-bit addressing.
    #[inline]
    fn u64_to_u32(value: u64) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        u32::try_from(value).unwrap_or(value as u32)
    }

    /// Get the expression buffer
    pub fn exprs(&self) -> &[GpuExpr] {
        &self.exprs
    }

    /// Get the level buffer
    pub fn levels(&self) -> &[GpuLevel] {
        &self.levels
    }

    /// Get the name table
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Get the string table
    pub fn strings(&self) -> &[String] {
        &self.strings
    }

    /// Intern a name and return its index
    fn intern_name(&mut self, name: &Name) -> u32 {
        let s = name.to_string();
        if let Some(&idx) = self.name_map.get(&s) {
            return idx;
        }
        let idx = Self::usize_to_u32(self.names.len());
        self.name_map.insert(s.clone(), idx);
        self.names.push(s);
        idx
    }

    /// Intern a string literal and return its index
    fn intern_string(&mut self, s: &str) -> u32 {
        // For simplicity, we don't deduplicate strings (they're less common)
        let idx = Self::usize_to_u32(self.strings.len());
        self.strings.push(s.to_string());
        idx
    }

    /// Encode binder info as u32
    fn encode_binder_info(bi: BinderInfo) -> u32 {
        match bi {
            BinderInfo::Default => BINDER_DEFAULT,
            BinderInfo::Implicit => BINDER_IMPLICIT,
            BinderInfo::StrictImplicit => BINDER_STRICT_IMPLICIT,
            BinderInfo::InstImplicit => BINDER_INST_IMPLICIT,
        }
    }

    /// Add a level to the arena and return its index
    pub fn add_level(&mut self, level: &Level) -> u32 {
        // First serialize to compute key
        let (gpu_level, key) = self.serialize_level_inner(level);

        // Check cache
        if let Some(&idx) = self.level_cache.get(&key) {
            return idx;
        }

        // Add new level
        let idx = Self::usize_to_u32(self.levels.len());
        self.level_cache.insert(key, idx);
        self.levels.push(gpu_level);
        idx
    }

    fn serialize_level_inner(&mut self, level: &Level) -> (GpuLevel, LevelKey) {
        match level {
            Level::Zero => (GpuLevel::zero(), LevelKey::Zero),
            Level::Succ(inner) => {
                let inner_idx = self.add_level(inner);
                (GpuLevel::succ(inner_idx), LevelKey::Succ(inner_idx))
            }
            Level::Max(l1, l2) => {
                let l1_idx = self.add_level(l1);
                let l2_idx = self.add_level(l2);
                (GpuLevel::max(l1_idx, l2_idx), LevelKey::Max(l1_idx, l2_idx))
            }
            Level::IMax(l1, l2) => {
                let l1_idx = self.add_level(l1);
                let l2_idx = self.add_level(l2);
                (
                    GpuLevel::imax(l1_idx, l2_idx),
                    LevelKey::IMax(l1_idx, l2_idx),
                )
            }
            Level::Param(name) => {
                let name_idx = self.intern_name(name);
                (GpuLevel::param(name_idx), LevelKey::Param(name_idx))
            }
        }
    }

    /// Add an expression to the arena and return its index
    pub fn add_expr(&mut self, expr: &Expr) -> u32 {
        // Serialize and check cache
        let (gpu_expr, key) = self.serialize_expr_inner(expr);

        if let Some(&idx) = self.expr_cache.get(&key) {
            return idx;
        }

        let idx = Self::usize_to_u32(self.exprs.len());
        self.expr_cache.insert(key, idx);
        self.exprs.push(gpu_expr);
        idx
    }

    fn serialize_expr_inner(&mut self, expr: &Expr) -> (GpuExpr, ExprKey) {
        match expr {
            Expr::BVar(idx) => (GpuExpr::bvar(*idx), ExprKey::BVar(*idx)),
            Expr::FVar(FVarId(id)) => {
                // Truncate to 32 bits for GPU (most IDs are small)
                (GpuExpr::fvar(Self::u64_to_u32(*id)), ExprKey::FVar(*id))
            }
            Expr::Sort(level) => {
                let level_idx = self.add_level(level);
                (GpuExpr::sort(level_idx), ExprKey::Sort(level_idx))
            }
            Expr::Const(name, levels) => {
                let name_idx = self.intern_name(name);
                let levels_start = Self::usize_to_u32(self.levels.len());
                let level_indices: Vec<u32> = levels.iter().map(|l| self.add_level(l)).collect();
                let levels_count = Self::usize_to_u32(levels.len());
                (
                    GpuExpr::const_(name_idx, levels_start, levels_count),
                    ExprKey::Const(name_idx, level_indices),
                )
            }
            Expr::App(func, arg) => {
                let func_idx = self.add_expr(func);
                let arg_idx = self.add_expr(arg);
                (
                    GpuExpr::app(func_idx, arg_idx),
                    ExprKey::App(func_idx, arg_idx),
                )
            }
            Expr::Lam(bi, ty, body) => {
                let bi_encoded = Self::encode_binder_info(*bi);
                let ty_idx = self.add_expr(ty);
                let body_idx = self.add_expr(body);
                (
                    GpuExpr::lam(bi_encoded, ty_idx, body_idx),
                    ExprKey::Lam(bi_encoded, ty_idx, body_idx),
                )
            }
            Expr::Pi(bi, ty, body) => {
                let bi_encoded = Self::encode_binder_info(*bi);
                let ty_idx = self.add_expr(ty);
                let body_idx = self.add_expr(body);
                (
                    GpuExpr::pi(bi_encoded, ty_idx, body_idx),
                    ExprKey::Pi(bi_encoded, ty_idx, body_idx),
                )
            }
            Expr::Let(ty, val, body) => {
                let ty_idx = self.add_expr(ty);
                let val_idx = self.add_expr(val);
                let body_idx = self.add_expr(body);
                (
                    GpuExpr::let_(ty_idx, val_idx, body_idx),
                    ExprKey::Let(ty_idx, val_idx, body_idx),
                )
            }
            Expr::Lit(Literal::Nat(n)) => {
                // SAFETY: Intentional truncation - split u64 into low/high u32 parts for GPU transfer
                #[allow(clippy::cast_possible_truncation)]
                let low = *n as u32;
                #[allow(clippy::cast_possible_truncation)]
                let high = (*n >> 32) as u32;
                (GpuExpr::lit_nat(low, high), ExprKey::LitNat(*n))
            }
            Expr::Lit(Literal::String(s)) => {
                let str_idx = self.intern_string(s);
                (GpuExpr::lit_str(str_idx), ExprKey::LitStr(str_idx))
            }
            Expr::Proj(name, field_idx, expr) => {
                let name_idx = self.intern_name(name);
                let expr_idx = self.add_expr(expr);
                (
                    GpuExpr::proj(name_idx, *field_idx, expr_idx),
                    ExprKey::Proj(name_idx, *field_idx, expr_idx),
                )
            }
            // MData is transparent - serialize the inner expression
            Expr::MData(_, inner) => self.serialize_expr_inner(inner),
            // Mode-specific expressions are not supported on GPU
            Expr::CubicalInterval
            | Expr::CubicalI0
            | Expr::CubicalI1
            | Expr::CubicalPath { .. }
            | Expr::CubicalPathLam { .. }
            | Expr::CubicalPathApp { .. }
            | Expr::CubicalHComp { .. }
            | Expr::CubicalTransp { .. }
            | Expr::ClassicalChoice { .. }
            | Expr::ClassicalEpsilon { .. }
            | Expr::ZFCSet(_)
            | Expr::ZFCMem { .. }
            | Expr::ZFCComprehension { .. }
            | Expr::SProp
            | Expr::Squash(_) => {
                panic!("Mode-specific expressions are not supported on GPU")
            }
        }
    }

    /// Add multiple expressions and return their indices
    pub fn add_exprs(&mut self, exprs: &[Expr]) -> Vec<u32> {
        exprs.iter().map(|e| self.add_expr(e)).collect()
    }

    /// Deserialize a GPU expression back to kernel Expr
    ///
    /// This is used for debugging and result extraction.
    pub fn to_expr(&self, idx: u32) -> Option<Expr> {
        let gpu_expr = self.exprs.get(idx as usize)?;
        self.deserialize_expr(gpu_expr)
    }

    fn deserialize_expr(&self, gpu_expr: &GpuExpr) -> Option<Expr> {
        match gpu_expr.tag {
            TAG_BVAR => Some(Expr::BVar(gpu_expr.data1)),
            TAG_FVAR => Some(Expr::FVar(FVarId(u64::from(gpu_expr.data1)))),
            TAG_SORT => {
                let level = self.to_level(gpu_expr.data1)?;
                Some(Expr::Sort(level))
            }
            TAG_CONST => {
                let name = self.to_name(gpu_expr.data1)?;
                let levels_start = gpu_expr.data2 as usize;
                let levels_count = gpu_expr.data3 as usize;
                let mut levels = Vec::with_capacity(levels_count);
                for i in 0..levels_count {
                    // SAFETY: levels_start and i are both from GPU data which uses u32 indices;
                    // overflow here would indicate corrupted GPU data
                    let idx = u32::try_from(levels_start + i).unwrap_or(u32::MAX);
                    levels.push(self.to_level(idx)?);
                }
                Some(Expr::const_(name, levels))
            }
            TAG_APP => {
                let func = self.to_expr(gpu_expr.data1)?;
                let arg = self.to_expr(gpu_expr.data2)?;
                Some(Expr::App(Arc::new(func), Arc::new(arg)))
            }
            TAG_LAM => {
                let bi = Self::decode_binder_info(gpu_expr.data1);
                let ty = self.to_expr(gpu_expr.data2)?;
                let body = self.to_expr(gpu_expr.data3)?;
                Some(Expr::Lam(bi, Arc::new(ty), Arc::new(body)))
            }
            TAG_PI => {
                let bi = Self::decode_binder_info(gpu_expr.data1);
                let ty = self.to_expr(gpu_expr.data2)?;
                let body = self.to_expr(gpu_expr.data3)?;
                Some(Expr::Pi(bi, Arc::new(ty), Arc::new(body)))
            }
            TAG_LET => {
                let ty = self.to_expr(gpu_expr.data1)?;
                let val = self.to_expr(gpu_expr.data2)?;
                let body = self.to_expr(gpu_expr.data3)?;
                Some(Expr::Let(Arc::new(ty), Arc::new(val), Arc::new(body)))
            }
            TAG_LIT_NAT => {
                let value = u64::from(gpu_expr.data1) | (u64::from(gpu_expr.data2) << 32);
                Some(Expr::Lit(Literal::Nat(value)))
            }
            TAG_LIT_STR => {
                let s = self.strings.get(gpu_expr.data1 as usize)?;
                Some(Expr::Lit(Literal::String(Arc::from(s.as_str()))))
            }
            TAG_PROJ => {
                let name = self.to_name(gpu_expr.data1)?;
                let expr = self.to_expr(gpu_expr.data3)?;
                Some(Expr::Proj(name, gpu_expr.data2, Arc::new(expr)))
            }
            _ => None,
        }
    }

    fn decode_binder_info(encoded: u32) -> BinderInfo {
        match encoded {
            BINDER_IMPLICIT => BinderInfo::Implicit,
            BINDER_STRICT_IMPLICIT => BinderInfo::StrictImplicit,
            BINDER_INST_IMPLICIT => BinderInfo::InstImplicit,
            _ => BinderInfo::Default,
        }
    }

    fn to_level(&self, idx: u32) -> Option<Level> {
        let gpu_level = self.levels.get(idx as usize)?;
        self.deserialize_level(gpu_level)
    }

    fn deserialize_level(&self, gpu_level: &GpuLevel) -> Option<Level> {
        match gpu_level.tag {
            LEVEL_TAG_ZERO => Some(Level::Zero),
            LEVEL_TAG_SUCC => {
                let inner = self.to_level(gpu_level.data1)?;
                Some(Level::Succ(Arc::new(inner)))
            }
            LEVEL_TAG_MAX => {
                let l1 = self.to_level(gpu_level.data1)?;
                let l2 = self.to_level(gpu_level.data2)?;
                Some(Level::Max(Arc::new(l1), Arc::new(l2)))
            }
            LEVEL_TAG_IMAX => {
                let l1 = self.to_level(gpu_level.data1)?;
                let l2 = self.to_level(gpu_level.data2)?;
                Some(Level::IMax(Arc::new(l1), Arc::new(l2)))
            }
            LEVEL_TAG_PARAM => {
                let name = self.to_name(gpu_level.data1)?;
                Some(Level::Param(name))
            }
            _ => None,
        }
    }

    fn to_name(&self, idx: u32) -> Option<Name> {
        let s = self.names.get(idx as usize)?;
        Some(Name::from_string(s))
    }

    /// Clear the arena for reuse
    pub fn clear(&mut self) {
        self.exprs.clear();
        self.levels.clear();
        self.names.clear();
        self.strings.clear();
        self.name_map.clear();
        self.expr_cache.clear();
        self.level_cache.clear();
    }

    /// Get the total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<GpuExpr>() * self.exprs.capacity()
            + std::mem::size_of::<GpuLevel>() * self.levels.capacity()
            + self.names.iter().map(String::len).sum::<usize>()
            + self.strings.iter().map(String::len).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_expr_size() {
        // Verify GpuExpr is exactly 16 bytes
        assert_eq!(std::mem::size_of::<GpuExpr>(), 16);
        assert_eq!(std::mem::align_of::<GpuExpr>(), 4);

        // Verify GpuLevel is exactly 16 bytes
        assert_eq!(std::mem::size_of::<GpuLevel>(), 16);
    }

    #[test]
    fn test_bvar_roundtrip() {
        let mut arena = GpuExprArena::new();
        let expr = Expr::BVar(42);
        let idx = arena.add_expr(&expr);
        let recovered = arena.to_expr(idx).unwrap();
        assert_eq!(expr, recovered);
    }

    #[test]
    fn test_sort_roundtrip() {
        let mut arena = GpuExprArena::new();
        let expr = Expr::Sort(Level::succ(Level::succ(Level::Zero)));
        let idx = arena.add_expr(&expr);
        let recovered = arena.to_expr(idx).unwrap();
        assert_eq!(expr, recovered);
    }

    #[test]
    fn test_app_roundtrip() {
        let mut arena = GpuExprArena::new();
        let func = Expr::const_(Name::from_string("f"), vec![]);
        let arg = Expr::BVar(0);
        let expr = Expr::App(Arc::new(func), Arc::new(arg));
        let idx = arena.add_expr(&expr);
        let recovered = arena.to_expr(idx).unwrap();
        assert_eq!(expr, recovered);
    }

    #[test]
    fn test_lambda_roundtrip() {
        let mut arena = GpuExprArena::new();
        let expr = Expr::Lam(
            BinderInfo::Implicit,
            Arc::new(Expr::Sort(Level::Zero)),
            Arc::new(Expr::BVar(0)),
        );
        let idx = arena.add_expr(&expr);
        let recovered = arena.to_expr(idx).unwrap();
        assert_eq!(expr, recovered);
    }

    #[test]
    fn test_pi_roundtrip() {
        let mut arena = GpuExprArena::new();
        let expr = Expr::Pi(
            BinderInfo::Default,
            Arc::new(Expr::Sort(Level::Zero)),
            Arc::new(Expr::Sort(Level::succ(Level::Zero))),
        );
        let idx = arena.add_expr(&expr);
        let recovered = arena.to_expr(idx).unwrap();
        assert_eq!(expr, recovered);
    }

    #[test]
    fn test_nat_literal_roundtrip() {
        let mut arena = GpuExprArena::new();

        // Small number
        let expr = Expr::Lit(Literal::Nat(42));
        let idx = arena.add_expr(&expr);
        assert_eq!(arena.to_expr(idx).unwrap(), expr);

        // Large number (> 32 bits)
        let expr = Expr::Lit(Literal::Nat(0x1_0000_0001));
        let idx = arena.add_expr(&expr);
        assert_eq!(arena.to_expr(idx).unwrap(), expr);
    }

    #[test]
    fn test_string_literal_roundtrip() {
        let mut arena = GpuExprArena::new();
        let expr = Expr::Lit(Literal::String(Arc::from("hello world")));
        let idx = arena.add_expr(&expr);
        assert_eq!(arena.to_expr(idx).unwrap(), expr);
    }

    #[test]
    fn test_const_with_levels_roundtrip() {
        let mut arena = GpuExprArena::new();
        let expr = Expr::const_(
            Name::from_string("List.map"),
            vec![Level::param(Name::from_string("u")), Level::Zero],
        );
        let idx = arena.add_expr(&expr);
        let recovered = arena.to_expr(idx).unwrap();
        assert_eq!(expr, recovered);
    }

    #[test]
    fn test_complex_expr_roundtrip() {
        let mut arena = GpuExprArena::new();

        // (λ (x : Type) => λ (y : x) => y) : (Type → Type → Type)
        let inner_body = Expr::BVar(0); // y
        let inner_lam = Expr::Lam(
            BinderInfo::Default,
            Arc::new(Expr::BVar(0)), // x (shifted into inner scope)
            Arc::new(inner_body),
        );
        let outer_lam = Expr::Lam(
            BinderInfo::Default,
            Arc::new(Expr::Sort(Level::succ(Level::Zero))), // Type
            Arc::new(inner_lam),
        );

        let idx = arena.add_expr(&outer_lam);
        let recovered = arena.to_expr(idx).unwrap();
        assert_eq!(outer_lam, recovered);
    }

    #[test]
    fn test_expression_deduplication() {
        let mut arena = GpuExprArena::new();

        // Add the same expression twice
        let expr = Expr::BVar(0);
        let idx1 = arena.add_expr(&expr);
        let idx2 = arena.add_expr(&expr);

        // Should return same index (deduplicated)
        assert_eq!(idx1, idx2);
        assert_eq!(arena.exprs().len(), 1);
    }

    #[test]
    fn test_level_deduplication() {
        let mut arena = GpuExprArena::new();

        let level = Level::succ(Level::succ(Level::Zero));
        let idx1 = arena.add_level(&level);
        let idx2 = arena.add_level(&level);

        assert_eq!(idx1, idx2);
    }
}
