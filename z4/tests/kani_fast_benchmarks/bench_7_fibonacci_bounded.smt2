; BENCHMARK 7: Fibonacci Sequence (Bounded)
; Tests: Recursive-like pattern, multiple state variables
; Z3: 0.04s | Z4 Target: <0.4s
(set-logic HORN)
(declare-fun Inv (Int Int Int Int) Bool)
; n: iteration, fib_n: current fib, fib_prev: previous fib, sum: running sum
(assert (forall ((n Int) (fib_n Int) (fib_prev Int) (sum Int))
  (=> (and (= n 0) (= fib_n 1) (= fib_prev 0) (= sum 1))
      (Inv n fib_n fib_prev sum))))
(assert (forall ((n Int) (fib_n Int) (fib_prev Int) (sum Int)
                 (n_next Int) (fib_n_next Int) (fib_prev_next Int) (sum_next Int))
  (=> (and (Inv n fib_n fib_prev sum)
           (< n 20)
           (= n_next (+ n 1))
           (= fib_n_next (+ fib_n fib_prev))
           (= fib_prev_next fib_n)
           (= sum_next (+ sum fib_n_next)))
      (Inv n_next fib_n_next fib_prev_next sum_next))))
; Property: fib values are always positive
(assert (forall ((n Int) (fib_n Int) (fib_prev Int) (sum Int))
  (=> (and (Inv n fib_n fib_prev sum) (<= fib_n 0)) false)))
(check-sat)
(get-model)
