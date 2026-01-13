/-
  LEAN 4 file with intentional syntax errors.
-/

-- Missing type annotation
theorem bad_syntax : by
  rfl

-- Unknown identifier
theorem unknown_ident : unknownType := sorry
