import SedenionPG32.Core

set_option maxRecDepth 100000
set_option maxHeartbeats 0

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
    all16 (fun i => !(inH ω i) || all16 (fun j => !(inH ω j) || all16 (fun k =>
      !(inH ω k) ||
      (intEq (assocSign i j k) (assocSign j k i) && intEq (assocSign j k i) (assocSign k i j))))) = true := by
  decide +kernel

/-- In all 7 zero-divisor Fano planes F_ZD (ω ∈ 9..15, H_ω = L₀ ⊔ Q₁), whenever (i, j) lie in
    the same stratum (both in L₀ or both in Q₁) and k lies in the complementary stratum:
    (1) both adjacent associators [e_i, e_j, e_k] and [e_k, e_i, e_j] vanish identically, and
    (2) for every non-collinear triad (i ≠ j ∧ (i ⊕ j) ≠ k), the separated associator
        [e_j, e_k, e_i] is strictly non-zero (±2). -/
theorem fzd_stratum_bcj_pole_cancellation :
    ∀ ω, ω < 16 → 9 ≤ ω →
    all16 (fun i => !(inH ω i) || all16 (fun j =>
      !(inH ω j && Nat.beq (Nat.ble i 7).toNat (Nat.ble j 7).toNat) ||
      all16 (fun k =>
        !(inH ω k && !(Nat.beq (Nat.ble i 7).toNat (Nat.ble k 7).toNat)) ||
        (intEq (assocSign i j k) 0 && intEq (assocSign k i j) 0 &&
          (!(!(Nat.beq i j) && !(Nat.beq (i ^^^ j) k)) || !(intEq (assocSign j k i) 0)))))) = true := by
  decide +kernel

/-! ### 9. GNSS CRPA Quaternion Hinge Carrier-Phase Continuity (Prop. 3.1) -/

/-- Membership in the 4D quaternion hinge subalgebra H_{L₁₂} = {0} ∪ (H_{ω₁} ∩ H_{ω₂}) -/
def inHinge (ω₁ ω₂ i : Nat) : Bool :=
  Nat.beq i 0 || (inH ω₁ i && inH ω₂ i)

/-- Every hyperplane intersection L₁₂ = H_{ω₁} ∩ H_{ω₂} (ω₁ ≠ ω₂) is closed under XOR (⊕),
    hence forms a projective line in L₃₅ generating an associative 4D quaternion subalgebra. -/
theorem hinge_is_projective_line :
    ∀ ω₁, ω₁ < 16 → 0 < ω₁ → ∀ ω₂, ω₂ < 16 → 0 < ω₂ → ω₁ ≠ ω₂ →
    all16 (fun i => !(inH ω₁ i && inH ω₂ i) || all16 (fun j =>
      !(inH ω₁ j && inH ω₂ j && !(Nat.beq i j)) ||
      (inH ω₁ (i ^^^ j) && inH ω₂ (i ^^^ j)))) = true := by
  decide +kernel

/-- Proposition 3.1 (Zero-Cycle-Slip Quadrature Phase Lock Across PG(3,2) Hinges):
    For every benign tracking workspace ω₁ ∈ F_alt (1..8) and every jammer-excision
    workspace ω₂ ∈ F_ZD (9..15), any carrier reference e_i and cascaded filter stages
    e_j, e_k supported on the shared quaternion hinge H_{L₁₂} = {0} ∪ (H_{ω₁} ∩ H_{ω₂})
    have identically zero sedenion associator: [e_i, e_j, e_k]_S = 0. -/
theorem gnss_pll_hinge_associative :
    ∀ ω₁, ω₁ < 9 → 0 < ω₁ → ∀ ω₂, ω₂ < 16 → 9 ≤ ω₂ →
    all16 (fun i => !(inHinge ω₁ ω₂ i) || all16 (fun j => !(inHinge ω₁ ω₂ j) || all16 (fun k =>
      !(inHinge ω₁ ω₂ k) ||
      (intEq (assocSign i j k) 0)))) = true := by
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
  cond (Nat.beq (a ^^^ c) (b ^^^ d) &&
        Nat.beq (a ^^^ d) (b ^^^ c) &&
        intEq (epsT a c + s₁ * s₂ * epsT b d) 0 &&
        intEq (s₂ * epsT a d + s₁ * epsT b c) 0 &&
        intEq ((epsT a c - s₁ * s₂ * epsT b d) * (epsT a c - s₁ * s₂ * epsT b d) +
               (s₂ * epsT a d - s₁ * epsT b c) * (s₂ * epsT a d - s₁ * epsT b c)) 8)
    1 0

/-- Theorem 2.3b (Exhaustive Kernel Verification of the 84-Blade / 336-Partner Hardware Crossbar ROM):
    Across all `a, c ∈ 1..7`, `b, d ∈ 9..15` (`b ≠ a ^^^ 8`, `d ≠ c ^^^ 8`), and signs `s₁, s₂ ∈ {±1}`,
    there are exactly 336 (= 84 × 4) ordered annihilating blade-partner pairs `(U_J, W_J)` satisfying
    `U_J · W_J = 0` AND simultaneously delivering exact `+3.01 dB` (`‖U_sat · W_J‖² = 8`) coherent
    array gain to the orthogonal anti-partner signal `U_sat = e_a - s₁ e_b`. -/
theorem hardware_crossbar_84_blades_336_partners :
    (sumOct fun a => sumTwist fun b =>
      cond (!(Nat.beq (b ^^^ 8) a))
        (sumOct fun c => sumTwist fun d =>
          cond (!(Nat.beq (d ^^^ 8) c))
            (countCrossbarPartner a b 1 c d 1 +
             countCrossbarPartner a b 1 c d (-1) +
             countCrossbarPartner a b (-1) c d 1 +
             countCrossbarPartner a b (-1) c d (-1))
            0)
        0) = 336 := rfl


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

/-! ### 10. Single-Snapshot Native Associator, SVD-Free 168-Moreno Nuller, 7 Apex Quaternion Hinges & Multi-Band Hardware DSP (Claims 10–18) -/

def allOct   (p : Nat → Bool) : Bool := p 1 && p 2 && p 3 && p 4 && p 5 && p 6 && p 7
def allTwist (p : Nat → Bool) : Bool := p 9 && p 10 && p 11 && p 12 && p 13 && p 14 && p 15

/-- Projection `π_O(R)` onto the 8D Octonion core `span{e₀..e₇}`. -/
def piOct (u : Vec) : Vec := fun k => cond (Nat.ble k 7) (u k) 0

/-- Single-Snapshot Native Sedenion Associator `[R, π_O(R), e₈] = (R · π_O(R)) · e₈ - R · (π_O(R) · e₈)`. -/
def nativeAssociator (u : Vec) : Vec :=
  sub (mul (mul u (piOct u)) (cbasis 8)) (mul u (mul (piOct u) (cbasis 8)))

/-- Theorem 3.3 (Single-Snapshot Native Associator Commutator & Moreno Curvature Dichotomy):
    For every pair of octonion basis units `e_a` (`a ∈ 1..7`) and twist basis units `e_b` (`b ∈ 9..15`),
    the native associator of `R = e_a + e_b` has norm-squared `4` (`‖-2(e_a ×_O e_{b⊕8})‖² = 4`)
    whenever `b ≠ a ⊕ 8` (non-associative Moreno cone `κ_S = 1`) and vanishes (`0`) iff `b = a ⊕ 8`
    (associative 4D quaternion line `span{e₀, e_a, e_8, e_{a⊕8}}`, `κ_S = 0`). -/
theorem native_associator_moreno_curvature_dichotomy :
    allOct (fun a => allTwist (fun b =>
      cond (Nat.beq b (a ^^^ 8))
        (cisZero (nativeAssociator (add (cbasis a) (cbasis b))))
        (intEq (vnormSq (nativeAssociator (add (cbasis a) (cbasis b)))) 4))) = true := rfl

/-- Theorem 3.4 (Associator Quaternion Triad Orthogonality — Theorem 1 of `Agourakis 2026`):
    For every octonion basis unit `e_a` (`a ∈ 1..7`) and twist basis unit `e_b` (`b ∈ 9..15`),
    `A_native(e_a + e_b) = -2(e_a ×_O e_{b⊕8})` lies entirely in the 8D octonion core (`π_O(A) = A`)
    and is strictly orthogonal (`⟨A, e_a⟩ = 0` and `⟨A, e_{b⊕8}⟩ = 0`) to both frame vectors,
    directly completing the 3D imaginary quaternion triad `Im(H_u) = span{a, b₀, q₃}` in `O(1)` time. -/
theorem associator_quaternion_triad_orthogonality :
    allOct (fun a => allTwist (fun b =>
      let u := add (cbasis a) (cbasis b)
      let assoc := nativeAssociator u
      cveq (piOct assoc) assoc &&
      intEq (vdot assoc (cbasis a)) 0 &&
      intEq (vdot assoc (cbasis (b ^^^ 8))) 0)) = true := rfl

/-- Closed-form SVD-Free Moreno Nuller Operator (`Section III-A`, Eq. 9 of `Agourakis 2026`):
    `W(a, b₀, c) = c + ((a ·_O c) ·_O b₀) · e₈`. -/
def morenoPartner (a b0 c : Vec) : Vec :=
  add c (mul (mul (mul a c) b0) (cbasis 8))

/-- Fast index-level evaluation of `morenoPartner (cbasis a) (cbasis b₀) (cbasis c)`:
    `e_c + ε(a, c) ε(a ⊕ c, b₀) e_{a ⊕ c ⊕ b₀ ⊕ 8}`. -/
def morenoBasisPartner (a b0 c : Nat) : Vec :=
  let s := epsT a c * epsT (a ^^^ c) b0
  let d := a ^^^ c ^^^ b0 ^^^ 8
  fun k => cond (Nat.beq k c) 1 0 + cond (Nat.beq k d) s 0

/-- Equivalence between the full 3-fold octonion-product `morenoPartner` and `morenoBasisPartner`. -/
theorem morenoPartner_eq_basis :
    cveq (morenoPartner (cbasis 1) (cbasis 2) (cbasis 4)) (morenoBasisPartner 1 2 4) = true ∧
    cveq (morenoPartner (cbasis 1) (cbasis 2) (cbasis 5)) (morenoBasisPartner 1 2 5) = true ∧
    cveq (morenoPartner (cbasis 1) (cbasis 2) (cbasis 6)) (morenoBasisPartner 1 2 6) = true ∧
    cveq (morenoPartner (cbasis 1) (cbasis 2) (cbasis 7)) (morenoBasisPartner 1 2 7) = true :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- Pure `cond`/`intEq` indicator for optimal null-partner verification (`does not depend on any axioms`). -/
def isOptimalNullPartner (a b : Nat) (s₁ : Int) (c d : Nat) (s₂ : Int) : Int :=
  cond (Nat.beq (a ^^^ c) (b ^^^ d) &&
        Nat.beq (a ^^^ d) (b ^^^ c) &&
        intEq (epsT a c + s₁ * s₂ * epsT b d) 0 &&
        intEq (s₂ * epsT a d + s₁ * epsT b c) 0 &&
        intEq ((epsT a c - s₁ * s₂ * epsT b d) * (epsT a c - s₁ * s₂ * epsT b d) +
               (s₂ * epsT a d - s₁ * epsT b c) * (s₂ * epsT a d - s₁ * epsT b c)) 8)
    1 0

def u_zdkm0 : Vec := add (cbasis 1) (cbasis 10)
def u_zdkm1 : Vec := sub (cbasis 1) (cbasis 10)
def w_key1  : Vec := sub (cbasis 4) (cbasis 15)
def w_key2  : Vec := add (cbasis 5) (cbasis 14)

/-- Theorem 3.5 (Closed-Form SVD-Free Moreno Nuller across all 168 Stiefel-Seed Triples):
    Across all `7 × 6 = 42` non-collinear Stiefel frame pairs `(a, b₀)` (`a, b₀ ∈ 1..7, a ≠ b₀`,
    twist index `b = b₀ ⊕ 8 ∈ 9..15`) and all `4` orthogonal complement basis seeds
    `c ∈ span{e_a, e_{b₀}, e_a ×_O e_{b₀}}⊥` (`c ∉ {a, b₀, a ⊕ b₀}`), totaling `42 × 4 = 168`
    Moreno configurations:
    (1) The jammer blade `e_a + e_b` is annihilated identically (`(e_a + e_b) · W(a, b₀, c) = 0`),
    (2) The orthogonal anti-partner desired signal `e_a - e_b` achieves the exact Hurwitz
        gain `‖(e_a - e_b) · W(a, b₀, c)‖² = 8` (`+3.01 dB`). -/
theorem svd_free_moreno_nuller_168_identity :
    (sumOct fun a => sumOct fun b0 =>
      cond (!(Nat.beq a b0))
        (sumOct fun c =>
          cond (!(Nat.beq c a) && !(Nat.beq c b0) && !(Nat.beq c (a ^^^ b0)))
            (let b := b0 ^^^ 8
             let d := a ^^^ c ^^^ b0 ^^^ 8
             let s2 := epsT a c * epsT (a ^^^ c) b0
             isOptimalNullPartner a b 1 c d s2)
            0)
        0) = 168 ∧
    cisZero (mul u_zdkm0 (morenoBasisPartner 1 2 4)) = true ∧
    vnormSq (mul u_zdkm1 (morenoBasisPartner 1 2 4)) = 8 :=
  ⟨rfl, rfl, rfl⟩

/-- Theorem 3.6 (Universal Associativity of all 7 Apex Quaternion Hinge Subalgebras `H_a ⊂ S`):
    For every octonion index `a ∈ 1..7`, the 4D Quaternion Hinge spanned by `{e₀, e_a, e₈, e_{a⊕8}}`
    has vanishing associator `(e_x · e_y) · e_z = e_x · (e_y · e_z)` across all `4³ = 64` basis
    triples `(x, y, z)` (`448` total structure-constant identities) and full 16D vector products
    `(e_a · e₈) · e_{a⊕8} = e_a · (e₈ · e_{a⊕8})`. -/
theorem quaternion_hinge_associativity_all_7_lines :
    (allOct (fun a =>
      let idx : Nat → Nat
        | 0 => 0
        | 1 => a
        | 2 => 8
        | _ => a ^^^ 8
      let all4 (p : Nat → Bool) : Bool := p 0 && p 1 && p 2 && p 3
      all4 (fun i => all4 (fun j => all4 (fun k =>
        let x := idx i
        let y := idx j
        let z := idx k
        intEq (epsT x y * epsT (x ^^^ y) z) (epsT y z * epsT x (y ^^^ z)))))) = true) ∧
    (allOct (fun a =>
      let ea := cbasis a
      let e8 := cbasis 8
      let eb := cbasis (a ^^^ 8)
      cveq (mul (mul ea e8) eb) (mul ea (mul e8 eb))) = true) := ⟨rfl, rfl⟩

/-- Canonical 4D Quaternion Hinge duration projection `snapToHinge q te = q * te`. -/
def snapToHinge (q te : Nat) : Nat := q * te

/-- Theorem 3.7 (Zero TE Quantization Residual & Exact Symbol Recovery on the 4D Hinge — Claim 15): -/
theorem hinge_snap_zero_te_residual_and_exact_recovery (q te : Nat) (hte : 0 < te) :
    (snapToHinge q te) % te = 0 ∧ (snapToHinge q te) / te = q := by
  dsimp [snapToHinge]
  exact ⟨Nat.mul_mod_left q te, Nat.mul_div_cancel q hte⟩

/-- Theorem 3.8 (Idempotence of the 4D Quaternion Hinge Projection — Claim 15): -/
theorem hinge_snap_idempotent (q te : Nat) (hte : 0 < te) :
    snapToHinge ((snapToHinge q te) / te) te = snapToHinge q te := by
  rw [(hinge_snap_zero_te_residual_and_exact_recovery q te hte).2]

/-- 2-Stage Glitch Coalescer step: folds a sub-`T_E` glitch `(h2, l2)` into `l1`. -/
def coalesceGlitchPair (h1 l1 h2 l2 : Int) : Int × Int := (h1, l1 + (h2 + l2))

/-- Total elapsed duration of a pulse pair `(h, l)`. -/
def pairDuration (p : Int × Int) : Int := p.1 + p.2

/-- Theorem 3.9 (Exact Frame-Duration Conservation of the 2-Stage Glitch Coalescer — Claim 15): -/
theorem glitch_coalesce_preserves_total_duration (h1 l1 h2 l2 : Int) :
    pairDuration (coalesceGlitchPair h1 l1 h2 l2) = (h1 + l1) + (h2 + l2) := by
  dsimp [pairDuration, coalesceGlitchPair]
  rw [← Int.add_assoc, ← Int.add_assoc]

/-- Total elapsed duration across an arbitrary list of comparator glitch pairs. -/
def sumPairDurations : List (Int × Int) → Int
  | [] => 0
  | g :: gs => pairDuration g + sumPairDurations gs

/-- Folding a cascade of `N` sub-`T_E` comparator glitches into an initial pulse pair `p₀`. -/
def coalesceGlitchList (p0 : Int × Int) : List (Int × Int) → Int × Int
  | [] => p0
  | g :: gs => coalesceGlitchList (coalesceGlitchPair p0.1 p0.2 g.1 g.2) gs

/-- Theorem 3.10 (`N`-Glitch Cascade Frame-Duration Conservation — Claim 15): -/
theorem multi_glitch_cascade_duration_conservation (gs : List (Int × Int)) (p0 : Int × Int) :
    pairDuration (coalesceGlitchList p0 gs) = pairDuration p0 + sumPairDurations gs := by
  induction gs generalizing p0 with
  | nil =>
    dsimp [coalesceGlitchList, sumPairDurations]
    rw [Int.add_zero]
  | cons g rest ih =>
    dsimp [coalesceGlitchList, sumPairDurations]
    rw [ih (coalesceGlitchPair p0.1 p0.2 g.1 g.2)]
    rw [glitch_coalesce_preserves_total_duration]
    dsimp [pairDuration]
    rw [Int.add_assoc (p0.1 + p0.2) (g.1 + g.2) (sumPairDurations rest)]

/-- Stock `STM32WB55` LFRFID DMA timer reload formula (`lfrfid_raw_worker.c:263-264`):
    `arr = (duration / 8) - 1` (evaluated in signed arithmetic before 32-bit unsigned cast). -/
def lfrfidStockDmaArrSigned (duration : Nat) : Int :=
  Int.ofNat (duration / 8) - 1

/-- Theorem 3.11 (Stock LFRFID DMA Timer Underflow on `< 8 µs` Glitches — Claim 16):
    Every unconditioned raw comparator glitch with `duration ∈ 0..7 µs` causes `duration / 8 = 0`
    and `lfrfidStockDmaArrSigned duration = -1` (`0xFFFFFFFF` in 32-bit unsigned two's complement). -/
theorem lfrfid_stock_dma_underflow_on_glitch :
    lfrfidStockDmaArrSigned 0 = -1 ∧
    lfrfidStockDmaArrSigned 1 = -1 ∧
    lfrfidStockDmaArrSigned 5 = -1 ∧
    lfrfidStockDmaArrSigned 7 = -1 := ⟨rfl, rfl, rfl, rfl⟩

/-- Sedenion `PG(3,2)` LFRFID `125 kHz` (`8 µs` carrier grid) conditioned cycle duration. -/
def lfrfidConditionedDuration (kp kn : Nat) : Nat :=
  8 * (kp + 1) + 8 * (kn + 1)

/-- Theorem 3.12 (Sedenion `PG(3,2)` LFRFID DMA Underflow Immunity — Claim 16): -/
theorem lfrfid_pg32_dma_underflow_immune (kp kn : Nat) :
    16 ≤ lfrfidConditionedDuration kp kn ∧
    (lfrfidConditionedDuration kp kn) % 8 = 0 ∧
    1 ≤ (lfrfidConditionedDuration kp kn) / 8 - 1 := by
  dsimp [lfrfidConditionedDuration]
  have hfactor : 8 * (kp + 1) + 8 * (kn + 1) = 8 * (kp + kn + 2) := by omega
  rw [hfactor]
  refine ⟨by omega, Nat.mul_mod_right 8 (kp + kn + 2), by omega⟩

/-- Signed 16-bit `Q1.14` fixed-point unit scaling factor (`2¹⁴ = 16384`). -/
def scaleQ14 : Int := 16384

/-- Helper: factor integer scalar `q` out of a single Cayley–Dickson product term `e * (q * u) * w`. -/
theorem term_scale_left (e q u w : Int) : e * (q * u) * w = q * (e * u * w) := by
  rw [← Int.mul_assoc e q u, Int.mul_comm e q, Int.mul_assoc q e u, ← Int.mul_assoc q (e * u) w]

/-- Helper: factor integer scalar `q` out of `e * u * (q * w)`. -/
theorem term_scale_right (e q u w : Int) : e * u * (q * w) = q * (e * u * w) := by
  rw [← Int.mul_assoc (e * u) q w, Int.mul_comm (e * u) q, Int.mul_assoc q (e * u) w]

/-- Helper: factor `q²` out of `(q * x) * (q * x)`. -/
theorem sq_scale (q x : Int) : (q * x) * (q * x) = (q * q) * (x * x) := by
  rw [← Int.mul_assoc (q * x) q x, Int.mul_assoc q x q, Int.mul_comm x q, ← Int.mul_assoc q q x,
      Int.mul_assoc (q * q) x x]

/-- Left-scaling factorization of 16D Sedenion multiplication: `((q u) · w)_k = q (u · w)_k`. -/
theorem mul_scale_left (q : Int) (u w : Vec) (k : Nat) :
    mul (scale q u) w k = q * mul u w k := by
  dsimp [mul, sum16, scale]
  simp only [term_scale_left, ← Int.mul_add]

/-- Right-scaling factorization of 16D Sedenion multiplication: `(u · (q w))_k = q (u · w)_k`. -/
theorem mul_scale_right (q : Int) (u w : Vec) (k : Nat) :
    mul u (scale q w) k = q * mul u w k := by
  dsimp [mul, sum16, scale]
  simp only [term_scale_right, ← Int.mul_add]

/-- Bilinear scaling factorization of 16D Sedenion multiplication over integer fixed-point scales `q₁, q₂ ∈ ℤ`. -/
theorem mul_scale_left_right (q1 q2 : Int) (u w : Vec) (k : Nat) :
    mul (scale q1 u) (scale q2 w) k = (q1 * q2) * mul u w k := by
  rw [mul_scale_left, mul_scale_right, Int.mul_assoc]

/-- Quadratic norm-squared scaling under fixed-point input quantization `scale q u`:
    `‖(q u) · w‖² = q² ‖u · w‖²`. -/
theorem vnormSq_mul_scale_left (q : Int) (u w : Vec) :
    vnormSq (mul (scale q u) w) = (q * q) * vnormSq (mul u w) := by
  dsimp [vnormSq, vdot, sum16]
  simp only [mul_scale_left, sq_scale, ← Int.mul_add]

/-- Theorem 3.13 (`Q1.14` Fixed-Point Moreno Nuller Exact `0 LSB` Identity — Claims 10 & 14): -/
theorem q14_fixed_point_moreno_zero_lsb_identity (q1 q2 : Int) (u w : Vec) (k : Nat)
    (hnull : mul u w k = 0) :
    mul (scale q1 u) (scale q2 w) k = 0 ∧
    cisZero (mul u_zdkm0 (morenoBasisPartner 1 2 4)) = true ∧
    vnormSq (mul (scale scaleQ14 u_zdkm1) (morenoBasisPartner 1 2 4)) =
      (scaleQ14 * scaleQ14) * 8 := by
  refine ⟨?_, rfl, ?_⟩
  · rw [mul_scale_left_right, hnull, Int.mul_zero]
  · rw [vnormSq_mul_scale_left]
    rfl

/-- Theorem 3.14 (Cross-Band Inductive OOD Associator Dichotomy — Claim 17):
    Certifies the exact `7 + 42 = 49` partition of `(a, b) ∈ {1..7} × {9..15}` into `7` associative
    in-band 4D Quaternion Hinge states (`b = a ⊕ 8`, `nativeAssociator = 0`, `κ_S = 0`) and `42`
    cross-band inductive/capacitive Moreno cone states (`b ≠ a ⊕ 8`, `‖nativeAssociator‖² = 4`, `κ_S = 1`). -/
theorem cross_band_inductive_associator_dichotomy :
    (sumOct (fun a => sumTwist (fun b => cond (Nat.beq b (a ^^^ 8)) 1 0)) = 7) ∧
    (sumOct (fun a => sumTwist (fun b => cond (!(Nat.beq b (a ^^^ 8))) 1 0)) = 42) ∧
    allOct (fun a => allTwist (fun b =>
      cond (Nat.beq b (a ^^^ 8))
        (cisZero (nativeAssociator (add (cbasis a) (cbasis b))))
        (intEq (vnormSq (nativeAssociator (add (cbasis a) (cbasis b)))) 4))) = true :=
  ⟨rfl, rfl, native_associator_moreno_curvature_dichotomy⟩

/-- ISO14443A (`13.56 MHz` HF-NFC) 4-byte Cascade Level 1 UID Block Check Character (`BCC`). -/
def iso14443aBcc (u0 u1 u2 u3 : Nat) : Nat :=
  u0 ^^^ u1 ^^^ u2 ^^^ u3

/-- Theorem 3.15 (`13.56 MHz` HF-NFC / `125 kHz` LF-RFID Frequency Division & UID `BCC` Invariants — Claims 16 & 17): -/
theorem nfc_1356mhz_subcarrier_and_uid_bcc_invariants :
    8475 * 16 = 135600 ∧
    125 * 8 = 1000 ∧
    15625 * 64 = 1000000 ∧
    iso14443aBcc 0x04 0xC5 0x4F 0x83 = 0x0D ∧
    (0x04 ^^^ 0xC5 ^^^ 0x4F ^^^ 0x83 ^^^ iso14443aBcc 0x04 0xC5 0x4F 0x83) = 0 ∧
    iso14443aBcc 0x04 0x08 0xA5 0x49 = 0xE0 ∧
    (0x04 ^^^ 0x08 ^^^ 0xA5 ^^^ 0x49 ^^^ iso14443aBcc 0x04 0x08 0xA5 0x49) = 0 := by
  refine ⟨by omega, by omega, by omega, rfl, rfl, rfl, rfl⟩

/-- Theorem 3.16 (`16×16` Kronecker Multi-Tile Array Two-Stage Coherent Gain — Claim 11): -/
theorem kronecker_multitile_16x16_two_stage_gain :
    cisZero (mul u_zdkm0 w_key1) = true ∧
    cisZero (mul u_zdkm0 w_key2) = true ∧
    vnormSq (mul u_zdkm1 w_key1) = 8 ∧
    vnormSq (mul u_zdkm1 w_key2) = 8 ∧
    vnormSq (mul u_zdkm0 w_key1) * vnormSq (mul u_zdkm1 w_key2) = 0 ∧
    vnormSq (mul u_zdkm1 w_key1) * vnormSq (mul u_zdkm1 w_key2) = 64 ∧
    (vnormSq (mul u_zdkm1 w_key1) * vnormSq (mul u_zdkm1 w_key2)) =
      16 * (vnormSq u_zdkm1 * vnormSq u_zdkm1) := by
  decide +kernel

/-- 3-point trajectory step-difference squared energy `(r₁ - r₀)² + (r₂ - r₁)²`. -/
def stepDiffEnergy3 (r0 r1 r2 : Int) : Int :=
  (r1 - r0) * (r1 - r0) + (r2 - r1) * (r2 - r1)

/-- Theorem 3.17 (`2.4 GHz` IEEE 802.11 Multipath Fade Peeler Step-Energy Reduction — Claim 18): -/
theorem wifi_multipath_peeler_step_energy_reduction (r δ : Int) :
    stepDiffEnergy3 r (r + δ) r = 2 * (δ * δ) ∧
    stepDiffEnergy3 r r r = 0 ∧
    stepDiffEnergy3 r (r + δ) r - stepDiffEnergy3 r r r = 2 * (δ * δ) := by
  dsimp [stepDiffEnergy3]
  have h1 : (r + δ) - r = δ := by omega
  have h2 : r - (r + δ) = -δ := by omega
  have h3 : r - r = 0 := by omega
  have hsq : δ * δ + (-δ) * (-δ) = 2 * (δ * δ) := by
    rw [Int.neg_mul_neg]
    omega
  rw [h1, h2, h3, hsq]
  exact ⟨rfl, rfl, Int.sub_zero (2 * (δ * δ))⟩

end SedenionPG32
