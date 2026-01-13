//! Rust to Lean5 Translation
//!
//! This module provides translation from Rust semantics to Lean5 kernel terms,
//! enabling verification of Rust programs in Lean5.
//!
//! ## Translation Strategy
//!
//! Rust programs are translated to Lean5 using:
//!
//! 1. **Types**: Rust types become Lean5 inductive types with ownership predicates
//! 2. **Values**: Rust values become Lean5 terms
//! 3. **Expressions**: Rust expressions become function applications
//! 4. **Ownership**: Borrow rules become proof obligations
//!
//! ## Ownership as Proofs
//!
//! The key insight is that ownership rules translate to proof requirements:
//!
//! - `&T` requires proof that the value is valid
//! - `&mut T` requires proof of exclusive access
//! - Moving requires proof that no borrows exist
//!
//! These proofs are generated automatically when possible, or left as
//! goals for the user/automation to fill.

use crate::ownership::{BorrowChecker, BorrowError, OwnershipState, Place};
use crate::types::{Lifetime, Mutability, RustType};
use crate::values::Value as RustValue;

use lean5_kernel::env::Environment;
use lean5_kernel::expr::Expr as LeanExpr;
use lean5_kernel::level::Level as LeanLevel;
use lean5_kernel::name::Name as LeanName;

use std::collections::HashMap;

/// Helper to create a Name from a string like "Foo.bar"
fn make_name(s: &str) -> LeanName {
    LeanName::from_string(s)
}

/// Helper to create a constant expression
fn const_expr(name: &str, levels: Vec<LeanLevel>) -> LeanExpr {
    LeanExpr::const_(make_name(name), levels)
}

/// Helper to create a nat literal expression
fn nat_lit(n: u64) -> LeanExpr {
    LeanExpr::nat_lit(n)
}

/// Translation context
#[derive(Debug)]
pub struct TranslationContext {
    /// Lean5 environment for definitions
    pub env: Environment,
    /// Mapping from Rust type names to Lean5 names
    pub type_map: HashMap<String, LeanName>,
    /// Current local variable context (Rust name → de Bruijn level)
    pub locals: Vec<(String, RustType)>,
    /// Ownership state for proof generation
    pub ownership: OwnershipState,
    /// Generated proof obligations
    pub proof_obligations: Vec<ProofObligation>,
}

/// A proof obligation generated during translation
#[derive(Debug, Clone)]
pub struct ProofObligation {
    /// Description of what needs to be proven
    pub description: String,
    /// The Lean5 type (proposition) to prove
    pub goal: LeanExpr,
    /// Location in original Rust code
    pub location: Option<String>,
}

impl TranslationContext {
    /// Create a new translation context
    pub fn new() -> Self {
        let mut type_map = HashMap::new();

        // Standard type mappings
        type_map.insert("bool".to_string(), make_name("Bool"));
        type_map.insert("u8".to_string(), make_name("UInt8"));
        type_map.insert("u16".to_string(), make_name("UInt16"));
        type_map.insert("u32".to_string(), make_name("UInt32"));
        type_map.insert("u64".to_string(), make_name("UInt64"));
        type_map.insert("i8".to_string(), make_name("Int8"));
        type_map.insert("i16".to_string(), make_name("Int16"));
        type_map.insert("i32".to_string(), make_name("Int32"));
        type_map.insert("i64".to_string(), make_name("Int64"));
        type_map.insert("f32".to_string(), make_name("Float32"));
        type_map.insert("f64".to_string(), make_name("Float"));
        type_map.insert("String".to_string(), make_name("String"));
        type_map.insert("char".to_string(), make_name("Char"));

        Self {
            env: Environment::new(),
            type_map,
            locals: Vec::new(),
            ownership: OwnershipState::new(),
            proof_obligations: Vec::new(),
        }
    }

    /// Push a local variable
    pub fn push_local(&mut self, name: String, ty: RustType) {
        self.locals.push((name, ty));
    }

    /// Pop a local variable
    pub fn pop_local(&mut self) -> Option<(String, RustType)> {
        self.locals.pop()
    }

    /// Look up a local variable (returns de Bruijn index)
    pub fn lookup_local(&self, name: &str) -> Option<(u32, &RustType)> {
        for (idx, (n, ty)) in self.locals.iter().rev().enumerate() {
            if n == name {
                // SAFETY: Local variable count is bounded by practical stack depth limits,
                // which are far below u32::MAX. Use saturating conversion for defense.
                let idx_u32 = u32::try_from(idx).unwrap_or(u32::MAX);
                return Some((idx_u32, ty));
            }
        }
        None
    }

    /// Add a proof obligation
    pub fn add_obligation(&mut self, description: &str, goal: LeanExpr) {
        self.proof_obligations.push(ProofObligation {
            description: description.to_string(),
            goal,
            location: None,
        });
    }

    /// Get all proof obligations
    pub fn obligations(&self) -> &[ProofObligation] {
        &self.proof_obligations
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Translate a Rust type to a Lean5 expression
pub fn translate_type(ty: &RustType, ctx: &TranslationContext) -> LeanExpr {
    match ty {
        RustType::Unit => const_expr("Unit", vec![]),

        RustType::Bool => const_expr("Bool", vec![]),

        RustType::Char => const_expr("Char", vec![]),

        RustType::Uint(uint_ty) => {
            let name = match uint_ty {
                crate::types::UintType::U8 => "UInt8",
                crate::types::UintType::U16 => "UInt16",
                crate::types::UintType::U32 => "UInt32",
                crate::types::UintType::U64 => "UInt64",
                crate::types::UintType::U128 => "UInt128",
                crate::types::UintType::Usize => "USize",
            };
            const_expr(name, vec![])
        }

        RustType::Int(int_ty) => {
            let name = match int_ty {
                crate::types::IntType::I8 => "Int8",
                crate::types::IntType::I16 => "Int16",
                crate::types::IntType::I32 => "Int32",
                crate::types::IntType::I64 => "Int64",
                crate::types::IntType::I128 => "Int128",
                crate::types::IntType::Isize => "ISize",
            };
            const_expr(name, vec![])
        }

        RustType::Float(float_ty) => {
            let name = match float_ty {
                crate::types::FloatType::F32 => "Float32",
                crate::types::FloatType::F64 => "Float",
            };
            const_expr(name, vec![])
        }

        RustType::Reference {
            mutability, inner, ..
        } => {
            // References become dependent pairs: { r : Ref T // Valid r }
            // For now, simplify to just the inner type
            let inner_ty = translate_type(inner, ctx);
            let ref_name = match mutability {
                Mutability::Shared => "Ref",
                Mutability::Mutable => "RefMut",
            };
            LeanExpr::app(const_expr(ref_name, vec![LeanLevel::zero()]), inner_ty)
        }

        RustType::Tuple(elems) => {
            if elems.is_empty() {
                const_expr("Unit", vec![])
            } else if elems.len() == 1 {
                translate_type(&elems[0], ctx)
            } else {
                // Build nested Prod type
                let mut result = translate_type(elems.last().unwrap(), ctx);
                for elem in elems.iter().rev().skip(1) {
                    let elem_ty = translate_type(elem, ctx);
                    result = LeanExpr::app(
                        LeanExpr::app(
                            const_expr("Prod", vec![LeanLevel::zero(), LeanLevel::zero()]),
                            elem_ty,
                        ),
                        result,
                    );
                }
                result
            }
        }

        RustType::Array { element, len } => {
            let elem_ty = translate_type(element, ctx);
            // Array T n becomes Array T n in Lean
            LeanExpr::app(
                LeanExpr::app(const_expr("Array", vec![LeanLevel::zero()]), elem_ty),
                nat_lit(*len as u64),
            )
        }

        RustType::Vec { element } => {
            let elem_ty = translate_type(element, ctx);
            LeanExpr::app(const_expr("Array", vec![LeanLevel::zero()]), elem_ty)
        }

        RustType::Option { inner } => {
            let inner_ty = translate_type(inner, ctx);
            LeanExpr::app(const_expr("Option", vec![LeanLevel::zero()]), inner_ty)
        }

        RustType::Result { ok, err } => {
            let ok_ty = translate_type(ok, ctx);
            let err_ty = translate_type(err, ctx);
            LeanExpr::app(
                LeanExpr::app(
                    const_expr("Except", vec![LeanLevel::zero(), LeanLevel::zero()]),
                    err_ty,
                ),
                ok_ty,
            )
        }

        RustType::Named {
            name, type_args, ..
        } => {
            let lean_name = ctx
                .type_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| make_name(name));
            let mut result = LeanExpr::const_(lean_name, vec![LeanLevel::zero()]);
            for arg in type_args {
                let arg_ty = translate_type(arg, ctx);
                result = LeanExpr::app(result, arg_ty);
            }
            result
        }

        RustType::Never => {
            // Empty/False type
            const_expr("Empty", vec![])
        }

        RustType::Function { params, ret } => {
            // Function type: A → B → C
            let mut result = translate_type(ret, ctx);
            for param in params.iter().rev() {
                let param_ty = translate_type(param, ctx);
                result = LeanExpr::pi(lean5_kernel::expr::BinderInfo::Default, param_ty, result);
            }
            result
        }

        // Other types get mapped to a generic representation
        _ => const_expr("Any", vec![LeanLevel::zero()]),
    }
}

/// Translate a Rust value to a Lean5 expression
#[allow(clippy::only_used_in_recursion)]
pub fn translate_value(val: &RustValue, ctx: &TranslationContext) -> LeanExpr {
    match val {
        RustValue::Unit => const_expr("Unit.unit", vec![]),

        RustValue::Bool(b) => {
            let name = if *b { "Bool.true" } else { "Bool.false" };
            const_expr(name, vec![])
        }

        RustValue::Char(c) => {
            // Characters as nat codes
            nat_lit(*c as u64)
        }

        RustValue::Uint { value, .. } => nat_lit(*value as u64),

        RustValue::Int { value, .. } => {
            // Encode signed integers
            if *value >= 0 {
                nat_lit(*value as u64)
            } else {
                // Negative: Int.negSucc (n - 1) for -n
                let abs = (-value - 1) as u64;
                LeanExpr::app(const_expr("Int.negSucc", vec![]), nat_lit(abs))
            }
        }

        RustValue::Tuple(elems) => {
            if elems.is_empty() {
                const_expr("Unit.unit", vec![])
            } else if elems.len() == 1 {
                translate_value(&elems[0], ctx)
            } else {
                // Build nested Prod.mk
                let mut result = translate_value(elems.last().unwrap(), ctx);
                for elem in elems.iter().rev().skip(1) {
                    let elem_val = translate_value(elem, ctx);
                    result = LeanExpr::app(
                        LeanExpr::app(
                            const_expr("Prod.mk", vec![LeanLevel::zero(), LeanLevel::zero()]),
                            elem_val,
                        ),
                        result,
                    );
                }
                result
            }
        }

        RustValue::Struct { name, fields } => {
            // Struct constructor: Name.mk field1 field2 ...
            let ctor_name = format!("{name}.mk");
            let mut result = const_expr(&ctor_name, vec![LeanLevel::zero()]);
            // Fields are stored in a BTreeMap, so iteration is deterministic by name
            for val in fields.values() {
                result = LeanExpr::app(result, translate_value(val, ctx));
            }
            result
        }

        RustValue::Enum {
            name,
            variant,
            payload,
        } => {
            let ctor_name = format!("{name}.{variant}");
            let mut result = const_expr(&ctor_name, vec![LeanLevel::zero()]);

            match payload.as_ref() {
                crate::values::EnumPayload::Unit => {}
                crate::values::EnumPayload::Tuple(vals) => {
                    for val in vals {
                        result = LeanExpr::app(result, translate_value(val, ctx));
                    }
                }
                crate::values::EnumPayload::Struct(fields) => {
                    // Struct variant fields iterate in deterministic name order (BTreeMap)
                    for val in fields.values() {
                        result = LeanExpr::app(result, translate_value(val, ctx));
                    }
                }
            }
            result
        }

        // Other values get a placeholder
        _ => const_expr("sorry", vec![]),
    }
}

/// Check ownership and generate proof obligations
pub fn check_ownership(
    ctx: &mut TranslationContext,
    place: &Place,
    operation: OwnershipOp,
) -> Result<(), BorrowError> {
    let checker = BorrowChecker::new();

    match operation {
        OwnershipOp::Move => {
            checker.check_move(&ctx.ownership, place)?;
            ctx.ownership.mark_moved(place.clone());
        }
        OwnershipOp::SharedBorrow(lt) => {
            checker.check_borrow(&ctx.ownership, place, Mutability::Shared, &lt)?;
            ctx.ownership
                .add_borrow(place.clone(), Mutability::Shared, lt);
        }
        OwnershipOp::MutBorrow(lt) => {
            checker.check_borrow(&ctx.ownership, place, Mutability::Mutable, &lt)?;
            ctx.ownership
                .add_borrow(place.clone(), Mutability::Mutable, lt);
        }
        OwnershipOp::Use => {
            checker.check_use(&ctx.ownership, place)?;
        }
    }

    Ok(())
}

/// Ownership operation type
#[derive(Debug, Clone)]
pub enum OwnershipOp {
    Move,
    SharedBorrow(Lifetime),
    MutBorrow(Lifetime),
    Use,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::UintType;
    use crate::values::EnumPayload;
    use std::collections::BTreeMap;

    #[test]
    fn test_translate_primitive_types() {
        let ctx = TranslationContext::new();

        let bool_ty = RustType::Bool;
        let result = translate_type(&bool_ty, &ctx);
        assert!(matches!(result, LeanExpr::Const(_, _)));

        let u32_ty = RustType::Uint(UintType::U32);
        let result = translate_type(&u32_ty, &ctx);
        assert!(matches!(result, LeanExpr::Const(_, _)));
    }

    #[test]
    fn test_translate_tuple_type() {
        let ctx = TranslationContext::new();

        let tuple_ty = RustType::Tuple(vec![RustType::Bool, RustType::Uint(UintType::U32)]);
        let result = translate_type(&tuple_ty, &ctx);
        // Should be Prod Bool UInt32
        assert!(matches!(result, LeanExpr::App(_, _)));
    }

    #[test]
    fn test_translate_values() {
        let ctx = TranslationContext::new();

        let bool_val = RustValue::Bool(true);
        let result = translate_value(&bool_val, &ctx);
        assert!(matches!(result, LeanExpr::Const(_, _)));

        let int_val = RustValue::u32(42);
        let result = translate_value(&int_val, &ctx);
        assert!(matches!(result, LeanExpr::Lit(_)));
    }

    #[test]
    fn test_translate_option_type() {
        let ctx = TranslationContext::new();

        let option_ty = RustType::Option {
            inner: Box::new(RustType::Uint(UintType::U32)),
        };
        let result = translate_type(&option_ty, &ctx);
        // Should be Option UInt32
        assert!(matches!(result, LeanExpr::App(_, _)));
    }

    #[test]
    fn test_translate_struct_value_orders_fields_deterministically() {
        let ctx = TranslationContext::new();

        let mut fields = BTreeMap::new();
        fields.insert("y".to_string(), RustValue::u32(2));
        fields.insert("x".to_string(), RustValue::u32(1));

        let value = RustValue::Struct {
            name: "Pair".to_string(),
            fields,
        };

        let expr = translate_value(&value, &ctx);
        let expected = LeanExpr::app(
            LeanExpr::app(const_expr("Pair.mk", vec![LeanLevel::zero()]), nat_lit(1)),
            nat_lit(2),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_translate_enum_struct_variant_orders_fields_deterministically() {
        let ctx = TranslationContext::new();

        let mut fields = BTreeMap::new();
        fields.insert("b".to_string(), RustValue::Bool(true));
        fields.insert("a".to_string(), RustValue::u32(7));

        let value = RustValue::Enum {
            name: "Wrapper".to_string(),
            variant: "Data".to_string(),
            payload: Box::new(EnumPayload::Struct(fields)),
        };

        let expr = translate_value(&value, &ctx);
        let expected = LeanExpr::app(
            LeanExpr::app(
                const_expr("Wrapper.Data", vec![LeanLevel::zero()]),
                nat_lit(7),
            ),
            const_expr("Bool.true", vec![]),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_ownership_tracking() {
        let mut ctx = TranslationContext::new();
        let place = Place::local(0);

        ctx.ownership.mark_owned(place.clone());

        // Should allow use
        let result = check_ownership(&mut ctx, &place, OwnershipOp::Use);
        assert!(result.is_ok());

        // Should allow move
        let result = check_ownership(&mut ctx, &place, OwnershipOp::Move);
        assert!(result.is_ok());

        // After move, use should fail
        let result = check_ownership(&mut ctx, &place, OwnershipOp::Use);
        assert!(result.is_err());
    }

    #[test]
    fn test_borrow_tracking() {
        let mut ctx = TranslationContext::new();
        let place = Place::local(0);
        let lifetime = Lifetime::Named("a".to_string());

        ctx.ownership.mark_owned(place.clone());

        // Create shared borrow
        let result = check_ownership(
            &mut ctx,
            &place,
            OwnershipOp::SharedBorrow(lifetime.clone()),
        );
        assert!(result.is_ok());

        // Move should now fail (borrowed)
        let result = check_ownership(&mut ctx, &place, OwnershipOp::Move);
        assert!(result.is_err());

        // End the borrow
        ctx.ownership.end_borrows(&lifetime);

        // Now move should succeed
        let result = check_ownership(&mut ctx, &place, OwnershipOp::Move);
        assert!(result.is_ok());
    }
}
