; BENCHMARK 6: Overflow Detection Pattern
; Tests: Arithmetic overflow, bounded integers simulation
; Z3: 0.02s | Z4 Target: <0.2s
(set-logic HORN)
(declare-fun Inv (Int Int Bool) Bool)
; x: value, steps: iteration count, overflow: flag
; Simulate i8 overflow at 127
(assert (forall ((x Int) (steps Int) (overflow Bool))
  (=> (and (= x 0) (= steps 0) (= overflow false))
      (Inv x steps overflow))))
; Increment with overflow check
(assert (forall ((x Int) (steps Int) (overflow Bool) 
                 (x_next Int) (steps_next Int) (overflow_next Bool))
  (=> (and (Inv x steps overflow) 
           (not overflow)
           (< steps 200)
           (= steps_next (+ steps 1))
           (ite (>= x 127)
                (and (= overflow_next true) (= x_next x))
                (and (= overflow_next false) (= x_next (+ x 1)))))
      (Inv x_next steps_next overflow_next))))
; Property: x always in [-128, 127] range
(assert (forall ((x Int) (steps Int) (overflow Bool))
  (=> (and (Inv x steps overflow) (or (< x (- 128)) (> x 127))) false)))
(check-sat)
(get-model)
