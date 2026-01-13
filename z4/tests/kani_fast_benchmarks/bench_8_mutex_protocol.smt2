; BENCHMARK 8: Mutex Protocol (Safety)
; Tests: Concurrent state, mutual exclusion property
; Z3: 0.02s | Z4 Target: <0.2s
(set-logic HORN)
(declare-fun Inv (Int Int Int) Bool)
; pc1: process 1 state, pc2: process 2 state, lock: mutex state
; pc=0: idle, pc=1: waiting, pc=2: critical
(assert (forall ((pc1 Int) (pc2 Int) (lock Int))
  (=> (and (= pc1 0) (= pc2 0) (= lock 0))
      (Inv pc1 pc2 lock))))
; Process 1: idle -> waiting
(assert (forall ((pc1 Int) (pc2 Int) (lock Int) (pc1_next Int) (pc2_next Int) (lock_next Int))
  (=> (and (Inv pc1 pc2 lock) (= pc1 0) (= pc1_next 1) (= pc2_next pc2) (= lock_next lock))
      (Inv pc1_next pc2_next lock_next))))
; Process 1: waiting -> critical (if lock free)
(assert (forall ((pc1 Int) (pc2 Int) (lock Int) (pc1_next Int) (pc2_next Int) (lock_next Int))
  (=> (and (Inv pc1 pc2 lock) (= pc1 1) (= lock 0) (= pc1_next 2) (= pc2_next pc2) (= lock_next 1))
      (Inv pc1_next pc2_next lock_next))))
; Process 1: critical -> idle (release lock)
(assert (forall ((pc1 Int) (pc2 Int) (lock Int) (pc1_next Int) (pc2_next Int) (lock_next Int))
  (=> (and (Inv pc1 pc2 lock) (= pc1 2) (= pc1_next 0) (= pc2_next pc2) (= lock_next 0))
      (Inv pc1_next pc2_next lock_next))))
; Process 2: symmetric transitions
(assert (forall ((pc1 Int) (pc2 Int) (lock Int) (pc1_next Int) (pc2_next Int) (lock_next Int))
  (=> (and (Inv pc1 pc2 lock) (= pc2 0) (= pc2_next 1) (= pc1_next pc1) (= lock_next lock))
      (Inv pc1_next pc2_next lock_next))))
(assert (forall ((pc1 Int) (pc2 Int) (lock Int) (pc1_next Int) (pc2_next Int) (lock_next Int))
  (=> (and (Inv pc1 pc2 lock) (= pc2 1) (= lock 0) (= pc2_next 2) (= pc1_next pc1) (= lock_next 1))
      (Inv pc1_next pc2_next lock_next))))
(assert (forall ((pc1 Int) (pc2 Int) (lock Int) (pc1_next Int) (pc2_next Int) (lock_next Int))
  (=> (and (Inv pc1 pc2 lock) (= pc2 2) (= pc2_next 0) (= pc1_next pc1) (= lock_next 0))
      (Inv pc1_next pc2_next lock_next))))
; SAFETY: mutual exclusion - never both in critical
(assert (forall ((pc1 Int) (pc2 Int) (lock Int))
  (=> (and (Inv pc1 pc2 lock) (= pc1 2) (= pc2 2)) false)))
(check-sat)
(get-model)
