-- Test inductive type definitions
inductive MyBool : Type where
  | myTrue : MyBool
  | myFalse : MyBool

def myNot : MyBool -> MyBool
  | .myTrue => .myFalse
  | .myFalse => .myTrue
