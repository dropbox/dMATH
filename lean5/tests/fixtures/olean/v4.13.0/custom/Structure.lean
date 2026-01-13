-- Test structure definitions
structure MyPair (α β : Type) where
  fst : α
  snd : β

def swap (p : MyPair α β) : MyPair β α := ⟨p.snd, p.fst⟩
