; BENCHMARK 4: Conditional Branching
; Tests: Path-sensitive analysis, branch conditions
; Z3: 0.01s | Z4 Target: <0.1s
(set-logic HORN)
(declare-fun Inv (Int Int Int) Bool)
; pc=0: entry, pc=1: then branch, pc=2: else branch, pc=3: merge
(assert (forall ((pc Int) (x Int) (y Int)) 
  (=> (and (= pc 0) (= x 0) (= y 0)) (Inv pc x y))))
; Branch on x > 5
(assert (forall ((pc Int) (x Int) (y Int) (pc_next Int) (x_next Int) (y_next Int))
  (=> (and (Inv pc x y) (= pc 0) (> x 5) (= pc_next 1) (= x_next x) (= y_next (+ y 1)))
      (Inv pc_next x_next y_next))))
(assert (forall ((pc Int) (x Int) (y Int) (pc_next Int) (x_next Int) (y_next Int))
  (=> (and (Inv pc x y) (= pc 0) (<= x 5) (= pc_next 2) (= x_next x) (= y_next y))
      (Inv pc_next x_next y_next))))
; Merge and loop
(assert (forall ((pc Int) (x Int) (y Int) (pc_next Int) (x_next Int) (y_next Int))
  (=> (and (Inv pc x y) (or (= pc 1) (= pc 2)) (= pc_next 0) (= x_next (+ x 1)) (= y_next y))
      (Inv pc_next x_next y_next))))
; Property: y >= 0
(assert (forall ((pc Int) (x Int) (y Int)) (=> (and (Inv pc x y) (< y 0)) false)))
(check-sat)
(get-model)
