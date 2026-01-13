// Minimal Dafny example with provable assertions

// Simple method with verified contract
method SafeDivide(x: int, y: int) returns (result: int)
  requires y != 0
  ensures result == x / y
{
  result := x / y;
}

// Verified lemma
lemma AddCommutative(a: int, b: int)
  ensures a + b == b + a
{
  // Dafny proves this automatically
}

// Method with loop invariant
method Sum(n: nat) returns (s: nat)
  ensures s == n * (n + 1) / 2
{
  s := 0;
  var i := 0;
  while i < n
    invariant 0 <= i <= n
    invariant s == i * (i + 1) / 2
  {
    i := i + 1;
    s := s + i;
  }
}
