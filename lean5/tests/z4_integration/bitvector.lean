/-
  Z4 Integration Tests: Bitvector Arithmetic (QF_BV)

  Tests the z4_bv tactic for fixed-width bitvector goals.
-/

import Lean5.Tactic.Z4

-- =============================================================================
-- Basic Bitwise Operations
-- =============================================================================

-- AND with all-ones is identity
theorem and_ff (x : UInt8) : x &&& 0xFF = x := by
  z4_bv

-- OR with zero is identity
theorem or_zero (x : UInt8) : x ||| 0x00 = x := by
  z4_bv

-- XOR with self is zero
theorem xor_self (x : UInt8) : x ^^^ x = 0 := by
  z4_bv

-- NOT NOT is identity
theorem not_not (x : UInt8) : ~~~(~~~x) = x := by
  z4_bv

-- De Morgan's law
theorem de_morgan_and (x y : UInt8) : ~~~(x &&& y) = (~~~x) ||| (~~~y) := by
  z4_bv

theorem de_morgan_or (x y : UInt8) : ~~~(x ||| y) = (~~~x) &&& (~~~y) := by
  z4_bv

-- =============================================================================
-- Arithmetic Operations
-- =============================================================================

-- Addition commutativity
theorem add_comm (x y : UInt8) : x + y = y + x := by
  z4_bv

-- Addition associativity
theorem add_assoc (x y z : UInt8) : (x + y) + z = x + (y + z) := by
  z4_bv

-- Subtraction as addition of negation (two's complement)
theorem sub_as_add_neg (x y : UInt8) : x - y = x + (~~~y + 1) := by
  z4_bv

-- Zero multiplication
theorem mul_zero (x : UInt8) : x * 0 = 0 := by
  z4_bv

-- =============================================================================
-- Shifts and Rotations
-- =============================================================================

-- Left shift by 0 is identity
theorem shl_zero (x : UInt8) : x <<< 0 = x := by
  z4_bv

-- Right shift by 0 is identity
theorem shr_zero (x : UInt8) : x >>> 0 = x := by
  z4_bv

-- Left shift then right shift (for small shifts)
theorem shl_shr (x : UInt8) : (x <<< 2) >>> 2 = x &&& 0x3F := by
  z4_bv

-- Double shift
theorem double_shl (x : UInt8) : (x <<< 2) <<< 2 = x <<< 4 := by
  z4_bv

-- =============================================================================
-- Comparison Operations
-- =============================================================================

-- Unsigned comparison reflexivity
theorem ule_refl (x : UInt8) : x <= x := by
  z4_bv

-- Unsigned comparison transitivity
theorem ule_trans (x y z : UInt8) (h1 : x <= y) (h2 : y <= z) : x <= z := by
  z4_bv

-- Equality is decidable
theorem eq_dec (x y : UInt8) : x = y ∨ x ≠ y := by
  z4_bv

-- =============================================================================
-- Overflow Detection
-- =============================================================================

-- Addition overflow condition
theorem add_overflow_cond (x y : UInt8) :
    (x + y < x) ↔ (x.toNat + y.toNat >= 256) := by
  z4_bv

-- Multiplication by 2 overflow (when high bit set)
theorem mul2_overflow (x : UInt8) :
    (x * 2 < x) ↔ (x &&& 0x80 ≠ 0) := by
  z4_bv

-- =============================================================================
-- Bit Extraction and Manipulation
-- =============================================================================

-- Extract low nibble
theorem low_nibble (x : UInt8) : x &&& 0x0F < 16 := by
  z4_bv

-- Extract high nibble
theorem high_nibble (x : UInt8) : (x >>> 4) < 16 := by
  z4_bv

-- Set a bit
theorem set_bit_3 (x : UInt8) : (x ||| 0x08) &&& 0x08 = 0x08 := by
  z4_bv

-- Clear a bit
theorem clear_bit_3 (x : UInt8) : (x &&& ~~~0x08) &&& 0x08 = 0 := by
  z4_bv

-- =============================================================================
-- Multi-Word Operations (16-bit)
-- =============================================================================

-- 16-bit AND identity
theorem and_16_ff (x : UInt16) : x &&& 0xFFFF = x := by
  z4_bv

-- 16-bit addition commutativity
theorem add_16_comm (x y : UInt16) : x + y = y + x := by
  z4_bv

-- Sign extension (8 to 16 bit)
theorem sign_ext_positive (x : UInt8) (h : x < 128) :
    (x.toUInt16 &&& 0x00FF) = x.toUInt16 := by
  z4_bv

-- =============================================================================
-- UNSAT Tests
-- =============================================================================

-- Impossible equality
theorem unsat_bv_eq : ¬(0x00 : UInt8) = (0xFF : UInt8) := by
  z4_bv

-- Overflow contradiction
theorem unsat_overflow (x : UInt8) : ¬(x > 0 ∧ x + 255 < x) := by
  z4_bv

-- =============================================================================
-- Performance Tests
-- =============================================================================

-- Small (< 1ms)
theorem perf_small_bv (x : UInt8) : x ||| 0 = x := by
  z4_bv

-- Medium - 32-bit operation (< 10ms)
theorem perf_medium_bv (x : UInt32) : x ^^^ x = 0 := by
  z4_bv

-- Larger - chained operations (< 100ms)
theorem perf_chain_bv (x y z : UInt16) :
    ((x &&& y) ||| z) ^^^ ((x ||| y) &&& z) =
    ((x ^^^ y) &&& (y ^^^ z)) ||| ((x ^^^ z) &&& (y ^^^ x)) := by
  z4_bv  -- This may take longer; marks performance boundary
