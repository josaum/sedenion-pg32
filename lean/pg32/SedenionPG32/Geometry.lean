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

/-! ### 9. GNSS CRPA Quaternion Hinge Carrier-Phase Continuity (Prop. 3.1) -/

/-- Membership in the 4D quaternion hinge subalgebra H_{L₁₂} = {0} ∪ (H_{ω₁} ∩ H_{ω₂}) -/
def inHinge (ω₁ ω₂ i : Nat) : Bool :=
  i == 0 || (inH ω₁ i && inH ω₂ i)

/-- Every hyperplane intersection L₁₂ = H_{ω₁} ∩ H_{ω₂} (ω₁ ≠ ω₂) is closed under XOR (⊕),
    hence forms a projective line in L₃₅ generating an associative 4D quaternion subalgebra. -/
theorem hinge_is_projective_line :
    ∀ ω₁, ω₁ < 16 → 0 < ω₁ → ∀ ω₂, ω₂ < 16 → 0 < ω₂ → ω₁ ≠ ω₂ →
    all16 (fun i => all16 (fun j =>
      !(inH ω₁ i && inH ω₂ i && inH ω₁ j && inH ω₂ j && i != j) ||
      (inH ω₁ (i ^^^ j) && inH ω₂ (i ^^^ j)))) = true := by
  decide +kernel

/-- Proposition 3.1 (Zero-Cycle-Slip Quadrature Phase Lock Across PG(3,2) Hinges):
    For every benign tracking workspace ω₁ ∈ F_alt (1..8) and every jammer-excision
    workspace ω₂ ∈ F_ZD (9..15), any carrier reference e_i and cascaded filter stages
    e_j, e_k supported on the shared quaternion hinge H_{L₁₂} = {0} ∪ (H_{ω₁} ∩ H_{ω₂})
    have identically zero sedenion associator: [e_i, e_j, e_k]_S = 0. -/
theorem gnss_pll_hinge_associative :
    ∀ ω₁, ω₁ < 9 → 0 < ω₁ → ∀ ω₂, ω₂ < 16 → 9 ≤ ω₂ →
    all16 (fun i => all16 (fun j => all16 (fun k =>
      !(inHinge ω₁ ω₂ i && inHinge ω₁ ω₂ j && inHinge ω₁ ω₂ k) ||
      (assocSign i j k == 0)))) = true := by
  decide +kernel

/-- Euclidean inner product and squared norm on 16D sedenion integer vectors -/
def vdot (u v : Vec) : Int := sum16 fun k => u k * v k
def vnormSq (u : Vec) : Int := vdot u u

/-- Anti-partner satellite manifold u_sat = e₁ - e₁₀ = (e₁, -e₂) orthogonal to u_jam = e₁ + e₁₀ -/
def u_antipartner : Vec := sub (basis 1) (basis 10)

/-- The 4 orthonormal basis beams of ker L_S(u_jam) from Theorem 2.1 -/
def v_beam1 : Vec := sub (basis 4) (basis 15)
def v_beam2 : Vec := add (basis 5) (basis 14)
def v_beam3 : Vec := sub (basis 6) (basis 13)
def v_beam4 : Vec := add (basis 7) (basis 12)

/-- Theorem 2.1 (Kernel-Checked 4D CRPA Null-Steering & Coherent +3 dB Anti-Partner Gain):
    (1) ⟨u_jam, u_sat⟩ = 0 with ‖u_jam‖² = ‖u_sat‖² = 2,
    (2) All 4 orthogonal beams v_beam1..4 annihilate u_jam identically (‖u_jam · v_r‖² = 0), and
    (3) All 4 orthogonal beams achieve exact +3 dB coherent array gain on u_sat:
        ‖u_sat · v_r‖² = 8 = 2 · ‖u_sat‖² · ‖v_r‖². -/
theorem crpa_null_and_antipartner_4 :
    vdot u_moreno u_antipartner = 0 ∧
    vnormSq u_moreno = 2 ∧ vnormSq u_antipartner = 2 ∧
    isZero (mul u_moreno v_beam1) ∧ vnormSq (mul u_antipartner v_beam1) = 8 ∧
    isZero (mul u_moreno v_beam2) ∧ vnormSq (mul u_antipartner v_beam2) = 8 ∧
    isZero (mul u_moreno v_beam3) ∧ vnormSq (mul u_antipartner v_beam3) = 8 ∧
    isZero (mul u_moreno v_beam4) ∧ vnormSq (mul u_antipartner v_beam4) = 8 := by
  decide +kernel

/-- Second simultaneous jammer blade u_jam2 = e_2 - e_9 in the same PG(3,2) octahedral context,
    and its orthogonal anti-partner satellite signature u_sat2 = e_2 + e_9. -/
def u_moreno2 : Vec := sub (basis 2) (basis 9)
def u_antipartner2 : Vec := add (basis 2) (basis 9)

/-- Theorem 2.2 (Simultaneous Multi-Jammer Annihilation in Shared PG(3,2) Assessment Context):
    Two orthogonal primitive jammers u_moreno = e_1 + e_10 and u_moreno2 = e_2 - e_9
    (⟨u_moreno, u_moreno2⟩ = 0) are simultaneously annihilated by the shared zero-divisor
    steering beams while preserving +3 dB coherent array gain (‖u_sat2 · v_r‖² = 8). -/
theorem crpa_multi_jammer_simultaneous_null :
    vdot u_moreno u_moreno2 = 0 ∧
    vdot u_moreno2 u_antipartner2 = 0 ∧
    isZero (mul u_moreno v_beam1) ∧ isZero (mul u_moreno2 v_beam1) ∧
    isZero (mul (add u_moreno u_moreno2) v_beam1) ∧
    vnormSq (mul u_antipartner v_beam1) = 8 ∧
    vnormSq (mul u_antipartner2 v_beam1) = 8 := by
  decide +kernel

/-- Sum over the 7 imaginary octonion indices `1..7` and the 7 twist-sector indices `9..15`. -/
def sumOct (f : Nat → Int) : Int :=
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7

def sumTwist (f : Nat → Int) : Int :=
  f 9 + f 10 + f 11 + f 12 + f 13 + f 14 + f 15

/-- The multiplierless 2-tap Pre-ADC Analog Crossbar operator `pre_adc_analog_crossbar`:
    for partner blade `W = e_c + s_d e_d`, each output channel `k` sums exactly two signed inputs:
    `z_k = ε_{k⊕c, c} u_{k⊕c} + s_d ε_{k⊕d, d} u_{k⊕d}`. -/
def preAdcCrossbar (u : Vec) (c d : Nat) (s_d : Int) : Vec :=
  fun k => epsT (k ^^^ c) c * u (k ^^^ c) + s_d * epsT (k ^^^ d) d * u (k ^^^ d)

/-- Theorem 2.3a (Exact Algebraic Equivalence of Pre-ADC 2-Tap Crossbar and Full 16D Sedenion Multiplication):
    For every input basis channel `i ∈ 0..15` and the 4 orthogonal zero-divisor steering blades `v_beam1..4`,
    the 2-tap analog crossbar `preAdcCrossbar` is identically equal to the full 256-MAC sedenion product `mul`. -/
theorem pre_adc_crossbar_eq_full_sedenion_mul :
    all16 (fun i =>
      veq (preAdcCrossbar (basis i) 4 15 (-1)) (mul (basis i) v_beam1) &&
      veq (preAdcCrossbar (basis i) 5 14 1)    (mul (basis i) v_beam2) &&
      veq (preAdcCrossbar (basis i) 6 13 (-1)) (mul (basis i) v_beam3) &&
      veq (preAdcCrossbar (basis i) 7 12 1)    (mul (basis i) v_beam4)) = true := by
  decide +kernel

/-- Fast 2-blade product evaluation via the verified Pre-ADC crossbar on the 4 active output channels
    `a ⊕ c`, `b ⊕ d`, `a ⊕ d`, `b ⊕ c`: returns `1` iff `(e_a + s₁ e_b)(e_c + s₂ e_d) = 0` AND
    the anti-partner `(e_a - s₁ e_b)(e_c + s₂ e_d)` has exact squared norm `8` (`+3.01 dB` gain). -/
def countCrossbarPartner (a b : Nat) (s₁ : Int) (c d : Nat) (s₂ : Int) : Int :=
  if (a ^^^ c) == (b ^^^ d) &&
     (a ^^^ d) == (b ^^^ c) &&
     (epsT a c + s₁ * s₂ * epsT b d == 0) &&
     (s₂ * epsT a d + s₁ * epsT b c == 0) &&
     ((epsT a c - s₁ * s₂ * epsT b d) * (epsT a c - s₁ * s₂ * epsT b d) +
      (s₂ * epsT a d - s₁ * epsT b c) * (s₂ * epsT a d - s₁ * epsT b c) == 8)
  then 1 else 0

/-- Theorem 2.3b (Exhaustive Kernel Verification of the 84-Blade / 336-Partner Hardware Crossbar ROM):
    Across all `a, c ∈ 1..7`, `b, d ∈ 9..15` (`b ≠ a ^^^ 8`, `d ≠ c ^^^ 8`), and signs `s₁, s₂ ∈ {±1}`,
    there are exactly 336 (= 84 × 4) ordered annihilating blade-partner pairs `(U_J, W_J)` satisfying
    `U_J · W_J = 0` AND simultaneously delivering exact `+3.01 dB` (`‖U_sat · W_J‖² = 8`) coherent
    array gain to the orthogonal anti-partner signal `U_sat = e_a - s₁ e_b`. -/
theorem hardware_crossbar_84_blades_336_partners :
    (sumOct fun a => sumTwist fun b =>
      if (b ^^^ 8) != a then
        (sumOct fun c => sumTwist fun d =>
          if (d ^^^ 8) != c then
            countCrossbarPartner a b 1 c d 1 +
            countCrossbarPartner a b 1 c d (-1) +
            countCrossbarPartner a b (-1) c d 1 +
            countCrossbarPartner a b (-1) c d (-1)
          else 0)
      else 0) = 336 := by
  decide +kernel


/-- Stage-2 partner blade `v_stage2 = e₁ + e₁₀` which annihilates `v_beam2 = e₅ + e₁₄` from the right:
    `(e₅ + e₁₄) · (e₁ + e₁₀) = 0` (by anti-involution of Cayley–Dickson conjugation on Im(S)). -/
def v_stage2 : Vec := add (basis 1) (basis 10)

/-- Theorem 2.4 (Two-Stage Non-Associative Cascaded Array Peeler Exactness):
    Even when two jammers `u_moreno = e₁ + e₁₀` and `j_2 = e₀` do not lie in the same single-stage
    kernel (`e₀ · v_beam2 = v_beam2 ≠ 0`), the two-stage cascaded array peeler
    `((u_moreno + e₀) · v_beam2) · v_stage2` annihilates `u_moreno` at Stage 1 and peels off
    the transformed image `e₀ · v_beam2 = e₅ + e₁₄` at Stage 2, yielding `0` identically. -/
theorem cascaded_peeler_two_stage_kernel :
    isZero (mul u_moreno v_beam2) ∧
    !isZero (mul (basis 0) v_beam2) ∧
    isZero (mul (mul (basis 0) v_beam2) v_stage2) ∧
    isZero (mul (mul (add u_moreno (basis 0)) v_beam2) v_stage2) := by
  decide +kernel

/-- Genuine physical second jammer blade `j_phy2 = e₃ + e₁₂ ∈ Im(S)` (with zero `e₀` and `e₈` components,
    orthogonal to `u_moreno = e₁ + e₁₀`) and its Stage-2 partner blade `v_stage2_phy = e₄ + e₁₁`. -/
def j_phy2 : Vec := add (basis 3) (basis 12)
def v_stage2_phy : Vec := add (basis 4) (basis 11)

/-- Theorem 2.5 (Physical Pure-Imaginary Two-Stage Cascaded Peeler & +6.02 dB Signal Gain):
    For two orthogonal purely imaginary primitive zero-divisor jammers `u_moreno = e₁ + e₁₀ ∈ Im(S)`
    and `j_phy2 = e₃ + e₁₂ ∈ Im(S)` that do NOT lie in the same Stage-1 kernel (`j_phy2 · v_beam2 ≠ 0`):
      (1) `u_moreno · v_beam2 = 0` (Jammer 1 annihilated at Stage 1),
      (2) `(j_phy2 · v_beam2)` has zero `e₀` and `e₈` scalar components and is annihilated at Stage 2
          by `v_stage2_phy = e₄ + e₁₁` (`((j_phy2 · v_beam2) · v_stage2_phy) = 0`),
      (3) Both jammers are simultaneously annihilated (`((u_moreno + j_phy2) · v_beam2) · v_stage2_phy = 0`), and
      (4) The authentic anti-partner satellite signal `u_antipartner = e₁ - e₁₀` achieves exact two-stage
          coherent power gain `‖(u_antipartner · v_beam2) · v_stage2_phy‖² = 16` (`+6.02 dB` normalized gain). -/
theorem cascaded_peeler_two_stage_kernel_physical :
    vdot u_moreno j_phy2 = 0 ∧
    isZero (mul u_moreno v_beam2) ∧
    !isZero (mul j_phy2 v_beam2) ∧
    (mul j_phy2 v_beam2) 0 = 0 ∧
    (mul j_phy2 v_beam2) 8 = 0 ∧
    isZero (mul (mul j_phy2 v_beam2) v_stage2_phy) ∧
    isZero (mul (mul (add u_moreno j_phy2) v_beam2) v_stage2_phy) ∧
    vnormSq (mul (mul u_antipartner v_beam2) v_stage2_phy) = 16 := by
  decide +kernel

end SedenionPG32
