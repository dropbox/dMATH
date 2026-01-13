theory MinimalPass
  imports Main
begin

(* A trivially true lemma *)
lemma trivial_true: "True"
  by simp

(* Excluded middle *)
lemma excluded_middle: "P \<or> \<not>P"
  by simp

(* Double negation *)
lemma double_neg: "\<not>\<not>P \<longrightarrow> P"
  by simp

end
