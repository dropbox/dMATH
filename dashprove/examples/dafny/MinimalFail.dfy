// Minimal Dafny example with failing assertion

// This method has a bug - missing division by zero check
method UnsafeDivide(x: int, y: int) returns (result: int)
  ensures result == x / y  // postcondition might not hold if y == 0
{
  result := x / y;  // Error: possible division by zero
}

// Lemma with false assertion
lemma FalseAssertion()
  ensures false  // This is unprovable
{
  // Cannot prove false
}
