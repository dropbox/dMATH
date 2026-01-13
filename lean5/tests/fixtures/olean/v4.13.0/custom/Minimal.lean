-- Minimal Lean module for .olean fixture testing
def identity (α : Type) (x : α) : α := x

theorem id_id : ∀ (α : Type) (x : α), identity α x = x := fun _ _ => rfl
