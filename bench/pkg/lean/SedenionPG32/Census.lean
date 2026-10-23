import SedenionPG32.Core
import SedenionPG32.Geometry

set_option maxRecDepth 100000
set_option maxHeartbeats 0

namespace SedenionPG32
/-! ## 6. Internal annihilating pairs in every zero-divisor plane (Thm 3.1 Case 2)

  For each ω ∈ {9..15} an explicit pair u, v ∈ A_ω with u·v = 0, all four basis indices
  inside H_ω.  Enumerated by the Python test-suite; checked here by the kernel. -/

def inPlane (ω : Nat) (idx : List Nat) : Bool := idx.all (inH ω)

theorem internal_pair_9  : isZero (mul (add (basis 2) (basis 9))  (add (basis 4) (basis 15)) ) ∧ inPlane 9  [2, 9, 4, 15] := by decide +kernel
theorem internal_pair_10 : isZero (mul (add (basis 1) (basis 10)) (sub (basis 4) (basis 15)) ) ∧ inPlane 10 [1, 10, 4, 15] := by decide +kernel
theorem internal_pair_11 : isZero (mul (add (basis 3) (basis 9))  (sub (basis 4) (basis 14)) ) ∧ inPlane 11 [3, 9, 4, 14] := by decide +kernel
theorem internal_pair_12 : isZero (mul (add (basis 1) (basis 12)) (add (basis 2) (basis 15)) ) ∧ inPlane 12 [1, 12, 2, 15] := by decide +kernel
theorem internal_pair_13 : isZero (mul (add (basis 2) (basis 9))  (sub (basis 5) (basis 14)) ) ∧ inPlane 13 [2, 9, 5, 14] := by decide +kernel
theorem internal_pair_14 : isZero (mul (add (basis 1) (basis 10)) (sub (basis 6) (basis 13)) ) ∧ inPlane 14 [1, 10, 6, 13] := by decide +kernel
theorem internal_pair_15 : isZero (mul (add (basis 3) (basis 9))  (sub (basis 5) (basis 15)) ) ∧ inPlane 15 [3, 9, 5, 15] := by decide +kernel

/-! ## 7. Associator census on basis triads (Sec. 1, item 2, corrected) -/

def assocZero (i j k : Nat) : Bool :=
  isZero (sub (mul (mul (basis i) (basis j)) (basis k)) (mul (basis i) (mul (basis j) (basis k))))

/-- collinear basis triads associate (Thm 3.2): for fixed i, every j ≠ i gives [e_i, e_j, e_i⊕j] = 0 -/
def collinearRow (i : Nat) : Bool :=
  (List.range 16).all fun j => j == 0 || i == j || assocZero i j (i ^^^ j)

theorem collinear_row_1 : collinearRow 1 = true := by decide +kernel
theorem collinear_row_2 : collinearRow 2 = true := by decide +kernel
theorem collinear_row_3 : collinearRow 3 = true := by decide +kernel
theorem collinear_row_4 : collinearRow 4 = true := by decide +kernel
theorem collinear_row_5 : collinearRow 5 = true := by decide +kernel
theorem collinear_row_6 : collinearRow 6 = true := by decide +kernel
theorem collinear_row_7 : collinearRow 7 = true := by decide +kernel
theorem collinear_row_8 : collinearRow 8 = true := by decide +kernel
theorem collinear_row_9 : collinearRow 9 = true := by decide +kernel
theorem collinear_row_10 : collinearRow 10 = true := by decide +kernel
theorem collinear_row_11 : collinearRow 11 = true := by decide +kernel
theorem collinear_row_12 : collinearRow 12 = true := by decide +kernel
theorem collinear_row_13 : collinearRow 13 = true := by decide +kernel
theorem collinear_row_14 : collinearRow 14 = true := by decide +kernel
theorem collinear_row_15 : collinearRow 15 = true := by decide +kernel

/-- ordered non-collinear basis triads with leading index `i`, and how many of them associate -/
def triadsFrom (i : Nat) : List (Nat × Nat) :=
  ((List.range 16).flatMap fun j => (List.range 16).map fun k => (j, k)).filter
    fun ⟨j, k⟩ => 0 < j && 0 < k && i ≠ j && j ≠ k && i ≠ k && (i ^^^ j ^^^ k) ≠ 0

def censusRow (i : Nat) : Nat × Nat :=
  ((triadsFrom i).length, ((triadsFrom i).filter fun ⟨j, k⟩ => assocZero i j k).length)

/-- per leading index: 168 non-collinear triads, of which 48 associate (0 when the leading unit is
    the apex e₈, which lies in no mixed line) -/
theorem census_row_1 : censusRow 1 = (168, 48) := by decide +kernel
theorem census_row_2 : censusRow 2 = (168, 48) := by decide +kernel
theorem census_row_3 : censusRow 3 = (168, 48) := by decide +kernel
theorem census_row_4 : censusRow 4 = (168, 48) := by decide +kernel
theorem census_row_5 : censusRow 5 = (168, 48) := by decide +kernel
theorem census_row_6 : censusRow 6 = (168, 48) := by decide +kernel
theorem census_row_7 : censusRow 7 = (168, 48) := by decide +kernel
theorem census_row_8 : censusRow 8 = (168, 0) := by decide +kernel
theorem census_row_9 : censusRow 9 = (168, 48) := by decide +kernel
theorem census_row_10 : censusRow 10 = (168, 48) := by decide +kernel
theorem census_row_11 : censusRow 11 = (168, 48) := by decide +kernel
theorem census_row_12 : censusRow 12 = (168, 48) := by decide +kernel
theorem census_row_13 : censusRow 13 = (168, 48) := by decide +kernel
theorem census_row_14 : censusRow 14 = (168, 48) := by decide +kernel
theorem census_row_15 : censusRow 15 = (168, 48) := by decide +kernel

/-- 672 of the 2520 ordered non-collinear basis triads associate (Sec. 1, corrected claim) -/
theorem associator_census :
    (censusRow 1).1 + (censusRow 2).1 + (censusRow 3).1 + (censusRow 4).1 + (censusRow 5).1 + (censusRow 6).1 + (censusRow 7).1 + (censusRow 8).1 + (censusRow 9).1 + (censusRow 10).1 + (censusRow 11).1 + (censusRow 12).1 + (censusRow 13).1 + (censusRow 14).1 + (censusRow 15).1 = 2520 ∧
    (censusRow 1).2 + (censusRow 2).2 + (censusRow 3).2 + (censusRow 4).2 + (censusRow 5).2 + (censusRow 6).2 + (censusRow 7).2 + (censusRow 8).2 + (censusRow 9).2 + (censusRow 10).2 + (censusRow 11).2 + (censusRow 12).2 + (censusRow 13).2 + (censusRow 14).2 + (censusRow 15).2 = 672 := by
  rw [census_row_1, census_row_2, census_row_3, census_row_4, census_row_5, census_row_6, census_row_7, census_row_8, census_row_9, census_row_10, census_row_11, census_row_12, census_row_13, census_row_14, census_row_15]
  decide

end SedenionPG32
