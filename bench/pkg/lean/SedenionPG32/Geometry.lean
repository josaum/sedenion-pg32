import SedenionPG32.Core

namespace SedenionPG32
/-! ## 5. PG(3,2): Fano hyperplanes and the 8+7 bifurcation (Sec. 3) -/

/-- popcount parity of a 4-bit index (structural, so the kernel can evaluate it) -/
def parity (x : Nat) : Nat :=
  (((x >>> 0) &&& 1) + ((x >>> 1) &&& 1) + ((x >>> 2) &&& 1) + ((x >>> 3) &&& 1)) % 2

/-- i ∈ H_ω  ⟺  ⟨v_ω, v_i⟩_{F₂} = 0   (eq. fano_hyperplane) -/
def inH (ω i : Nat) : Bool := 0 < i && i < 16 && parity (ω &&& i) = 0

def H (ω : Nat) : List Nat := (List.range 16).filter (inH ω)

/-- every hyperplane has exactly 7 points -/
theorem H_card : ∀ ω, ω < 16 → 0 < ω → (H ω).length = 7 := by decide +kernel

/-- every hyperplane is closed under ⊕ (hence its span is a subalgebra) -/
theorem H_closed : ∀ ω, ω < 16 → ∀ i, i < 16 → ∀ j, j < 16 →
    (0 < ω ∧ inH ω i ∧ inH ω j ∧ i ≠ j) → inH ω (i ^^^ j) := by decide +kernel

/-- Case 1: for ω ∈ 1..7 the apex e₈ lies in H_ω; H₈ = {1..7} is the octonion core itself -/
theorem apex_in_alt_planes : ∀ ω, ω < 8 → 0 < ω → inH ω 8 := by decide +kernel
theorem H8_is_octonion_core : H 8 = [1, 2, 3, 4, 5, 6, 7] := by decide +kernel

/-- Case 2: for ω ≥ 9 the apex is excluded -/
theorem apex_not_in_zd_planes : ∀ ω, ω < 16 → 9 ≤ ω → ¬ inH ω 8 := by decide +kernel

/-- Case 2 (corrected statement): the octonionic part of H_{8+p} is the polar line
    p^⊥ = { i ∈ {1..7} | ⟨p,i⟩ = 0 }, which contains p itself iff popcount(p) is even. -/
theorem zd_plane_octonion_part_is_polar :
    ∀ p, p < 8 → 0 < p → ∀ i, i < 8 → 0 < i → (inH (8 + p) i ↔ parity (p &&& i) = 0) := by
  decide +kernel

theorem polar_line_contains_p_iff_even :
    ∀ p, p < 8 → 0 < p → (inH (8 + p) p ↔ parity p = 0) := by decide +kernel

/-- the four points of H_{8+p} in the twist stratum are exactly {q+8 | ⟨p,q⟩ = 1} -/
theorem zd_plane_twist_part :
    ∀ p, p < 8 → 0 < p → ∀ q, q < 8 → 0 < q → (inH (8 + p) (q + 8) ↔ parity (p &&& q) = 1) := by
  decide +kernel

/-! ### Line classification (Sec. 3.1): 7 pure, 7 apex, 21 mixed zero-divisor lines -/

/-- unordered lines {i, j, i⊕j} with i < j < i⊕j -/
def lines : List (Nat × Nat × Nat) :=
  ((List.range 16).flatMap fun i => (List.range 16).map fun j => (i, j, i ^^^ j)).filter
    fun ⟨i, j, k⟩ => 0 < i && i < j && j < k && k < 16

theorem lines_card : lines.length = 35 := by decide +kernel

def isPure  : Nat × Nat × Nat → Bool := fun ⟨i, j, k⟩ => i < 8 && j < 8 && k < 8
def isApex  : Nat × Nat × Nat → Bool := fun ⟨i, j, k⟩ => i = 8 || j = 8 || k = 8
def isMixed : Nat × Nat × Nat → Bool := fun ⟨i, j, k⟩ =>
  (decide (i < 8)).toNat + (decide (j < 8)).toNat + (decide (k < 8)).toNat = 1 && !(isApex (i, j, k))

theorem line_counts :
    (lines.filter isPure).length = 7 ∧
    (lines.filter isApex).length = 7 ∧
    (lines.filter isMixed).length = 21 := by decide +kernel

/-- lines inside hyperplane ω -/
def linesIn (ω : Nat) : List (Nat × Nat × Nat) :=
  lines.filter fun ⟨i, j, k⟩ => inH ω i && inH ω j && inH ω k

theorem seven_lines_per_plane : ∀ ω, ω < 16 → 0 < ω → (linesIn ω).length = 7 := by decide +kernel

/-- Corrected count from Thm 3.1: alternative planes ω ∈ 1..7 carry 3 mixed lines, the
    canonical octonion plane ω = 8 carries 0, every zero-divisor plane carries 6. -/
theorem mixed_lines_per_plane :
    (∀ ω, ω < 8 → 0 < ω → ((linesIn ω).filter isMixed).length = 3) ∧
    ((linesIn 8).filter isMixed).length = 0 ∧
    (∀ ω, ω < 16 → 9 ≤ ω → ((linesIn ω).filter isMixed).length = 6) := by decide +kernel

/-- Theorem 3.2 (hinge): any two distinct hyperplanes meet in exactly 3 points -/
theorem hinge_card : ∀ ω₁, ω₁ < 16 → 0 < ω₁ → ∀ ω₂, ω₂ < 16 → 0 < ω₂ → ω₁ ≠ ω₂ →
    ((List.range 16).filter fun i => inH ω₁ i && inH ω₂ i).length = 3 := by decide +kernel

/-! ### 8+7 Cyclic Associator & BCJ Stratum Theorems (Sec. 5.2, Thm 5.3) -/

/-- Exact scalar coefficient of e_{i⊕j⊕k} in the basis associator [e_i, e_j, e_k] -/
def assocSign (i j k : Nat) : Int :=
  epsT i j * epsT (i ^^^ j) k - epsT j k * epsT i (j ^^^ k)

/-- In all 8 alternative Fano planes F_alt (ω ∈ 1..8), the associator is strictly invariant
    under cyclic permutations (i,j,k) ↦ (j,k,i), so cyclic BCJ differences vanish identically. -/
theorem falt_cyclic_assoc_invariant :
    ∀ ω, ω < 9 → 0 < ω →
    all16 (fun i => all16 (fun j => all16 (fun k =>
      !(inH ω i && inH ω j && inH ω k) ||
      (assocSign i j k == assocSign j k i && assocSign j k i == assocSign k i j)))) = true := by
  decide +kernel

/-- In all 7 zero-divisor Fano planes F_ZD (ω ∈ 9..15, H_ω = L₀ ⊔ Q₁), whenever (i, j) lie in
    the same stratum (both in L₀ or both in Q₁) and k lies in the complementary stratum:
    (1) both adjacent associators [e_i, e_j, e_k] and [e_k, e_i, e_j] vanish identically, and
    (2) for every non-collinear triad (i ≠ j ∧ (i ⊕ j) ≠ k), the separated associator
        [e_j, e_k, e_i] is strictly non-zero (±2). -/
theorem fzd_stratum_bcj_pole_cancellation :
    ∀ ω, ω < 16 → 9 ≤ ω →
    all16 (fun i => all16 (fun j => all16 (fun k =>
      !(inH ω i && inH ω j && inH ω k && (decide (i < 8) == decide (j < 8)) && (decide (i < 8) != decide (k < 8))) ||
      (assocSign i j k == 0 && assocSign k i j == 0 && (!(i != j && (i ^^^ j) != k) || assocSign j k i != 0))))) = true := by
  decide +kernel

end SedenionPG32
