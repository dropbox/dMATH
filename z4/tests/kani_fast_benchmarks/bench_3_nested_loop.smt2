; BENCHMARK 3: Nested Loop Pattern
; Tests: Multiple loop variables, nested iteration
; Z3: 0.05s | Z4 Target: <0.5s
(set-logic HORN)
(declare-fun Inv (Int Int Int) Bool)
; pc=0: outer loop entry, pc=1: inner loop, pc=2: exit
(assert (forall ((pc Int) (i Int) (j Int)) 
  (=> (and (= pc 0) (= i 0) (= j 0)) (Inv pc i j))))
; Outer loop: i < 10
(assert (forall ((pc Int) (i Int) (j Int) (pc_next Int) (i_next Int) (j_next Int))
  (=> (and (Inv pc i j) (= pc 0) (< i 10) (= pc_next 1) (= i_next i) (= j_next 0))
      (Inv pc_next i_next j_next))))
; Inner loop: j < 10
(assert (forall ((pc Int) (i Int) (j Int) (pc_next Int) (i_next Int) (j_next Int))
  (=> (and (Inv pc i j) (= pc 1) (< j 10) (= j_next (+ j 1)) (= pc_next 1) (= i_next i))
      (Inv pc_next i_next j_next))))
; Inner loop exit -> outer loop continue
(assert (forall ((pc Int) (i Int) (j Int) (pc_next Int) (i_next Int) (j_next Int))
  (=> (and (Inv pc i j) (= pc 1) (>= j 10) (= i_next (+ i 1)) (= pc_next 0) (= j_next j))
      (Inv pc_next i_next j_next))))
; Property: bounds always hold
(assert (forall ((pc Int) (i Int) (j Int)) 
  (=> (and (Inv pc i j) (not (and (>= i 0) (<= i 10) (>= j 0) (<= j 10)))) false)))
(check-sat)
(get-model)
