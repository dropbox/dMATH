-- Simple benchmark file to test Lean 4 profiling
-- Run: lean --profile simple_bench.lean

-- Type inference for Sort
def test_prop : Type := Prop

-- Simple identity function
def id' (α : Type) (x : α) : α := x

-- Nested lambda
def nested2 := fun (x : Type) => fun (y : Type) => x

-- Beta reduction test (normalized at elaboration)
def beta_test := (fun (x : Prop) => x) True

-- These match the expressions benchmarked in Lean5
#check @id' Prop True
#check nested2 Prop Nat
