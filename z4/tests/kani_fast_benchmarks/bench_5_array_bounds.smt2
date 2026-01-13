; BENCHMARK 5: Array Bounds Check (using arrays theory)
; Tests: Array theory integration, bounds checking
; Z3: 0.03s | Z4 Target: <0.3s
(set-logic HORN)
(declare-fun Inv (Int (Array Int Int)) Bool)
; Initialize: arr[0..9] = 0, i = 0
(assert (forall ((i Int) (arr (Array Int Int)))
  (=> (and (= i 0) (= (select arr 0) 0))
      (Inv i arr))))
; Loop: arr[i] := i, i++, while i < 10
(assert (forall ((i Int) (arr (Array Int Int)) (i_next Int) (arr_next (Array Int Int)))
  (=> (and (Inv i arr) (< i 10) 
           (= arr_next (store arr i i))
           (= i_next (+ i 1)))
      (Inv i_next arr_next))))
; Property: after loop, arr[5] >= 0
(assert (forall ((i Int) (arr (Array Int Int)))
  (=> (and (Inv i arr) (>= i 10) (< (select arr 5) 0)) false)))
(check-sat)
(get-model)
