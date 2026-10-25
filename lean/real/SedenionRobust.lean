/-
  SedenionRobust.lean — Residual-leakage bound for sedenion zero-divisor nulling.

  Setting (same algebra as SedenionReal.lean): sedenions are pairs (a, b) of octonions with
  the Cayley–Dickson doubling product (a,b)(c,d) = (ac − d* b, da + b c*).

  Main result (`leakage_bound`):  if v ∈ ker L(u), i.e. u·v = 0, then for any perturbation δ
        ‖(u + δ)·v‖² ≤ 2 ‖δ‖² ‖v‖².
  So the null depth achievable against a true jammer signature u+δ, when the nuller was built
  from the estimate u, is bounded below by −10·log₁₀(2‖δ‖²‖v‖²): it is governed entirely by the
  signature-estimation error, exactly as for any signature-driven nuller.  No algebraic
  structure can beat this; the "312 dB" of a noiseless simulation is the δ = 0 case.

  Ingredients:
    • `smul_add_left`   — bilinearity in the first slot (so the residual is δ·v exactly);
    • `snorm_smul_le`   — ‖x·y‖² ≤ 2‖x‖²‖y‖² (from the octonion composition law);
    • `leakage_bound`   — the two combined.

  No `sorry`, no `native_decide`; all identities close by `ring` / `linarith` / `positivity`.
-/
import Mathlib.Tactic
import SedenionReal

/-! The octonion/sedenion carrier, product, inner product and the constructive nullspace theorem
    live in `SedenionReal` (same directory); this file only adds the robustness
    and dual-polarisation results on top of them. -/

namespace SedenionRobust
open SedenionReal

def sadd (u w : O × O) : O × O := (oadd u.1 w.1, oadd u.2 w.2)

/-- squared Euclidean norm on O and on S = O × O -/
def onorm2 (x : O) : ℝ := dot x x
def snorm2 (u : O × O) : ℝ := onorm2 u.1 + onorm2 u.2

/-! ### Elementary facts -/

theorem dot_nonneg (x : O) : 0 ≤ dot x x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [dot]
  nlinarith [sq_nonneg x0, sq_nonneg x1, sq_nonneg x2, sq_nonneg x3,
             sq_nonneg x4, sq_nonneg x5, sq_nonneg x6, sq_nonneg x7]

theorem conj_norm (x : O) : dot (conj x) (conj x) = dot x x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [dot, conj]; ring

/-- parallelogram-type inequality ‖p − q‖² ≤ 2‖p‖² + 2‖q‖² -/
theorem osub_norm_le (p q : O) : dot (osub p q) (osub p q) ≤ 2 * dot p p + 2 * dot q q := by
  have h : dot (osub p q) (osub p q) + dot (oadd p q) (oadd p q) = 2 * dot p p + 2 * dot q q := by
    obtain ⟨p0, p1, p2, p3, p4, p5, p6, p7⟩ := p
    obtain ⟨q0, q1, q2, q3, q4, q5, q6, q7⟩ := q
    simp only [dot, osub, oadd]; ring
  have := dot_nonneg (oadd p q)
  linarith

theorem oadd_norm_le (p q : O) : dot (oadd p q) (oadd p q) ≤ 2 * dot p p + 2 * dot q q := by
  have h : dot (osub p q) (osub p q) + dot (oadd p q) (oadd p q) = 2 * dot p p + 2 * dot q q := by
    obtain ⟨p0, p1, p2, p3, p4, p5, p6, p7⟩ := p
    obtain ⟨q0, q1, q2, q3, q4, q5, q6, q7⟩ := q
    simp only [dot, osub, oadd]; ring
  have := dot_nonneg (osub p q)
  linarith

/-! ### Bilinearity in the first argument -/

theorem smul_add_left (u w v : O × O) : smul (sadd u w) v = sadd (smul u v) (smul w v) := by
  obtain ⟨⟨u0, u1, u2, u3, u4, u5, u6, u7⟩, ⟨u8, u9, u10, u11, u12, u13, u14, u15⟩⟩ := u
  obtain ⟨⟨w0, w1, w2, w3, w4, w5, w6, w7⟩, ⟨w8, w9, w10, w11, w12, w13, w14, w15⟩⟩ := w
  obtain ⟨⟨v0, v1, v2, v3, v4, v5, v6, v7⟩, ⟨v8, v9, v10, v11, v12, v13, v14, v15⟩⟩ := v
  simp only [smul, sadd, omul, conj, osub, oadd, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩ <;> ring

theorem sadd_zero_left (w : O × O) : sadd (ozero, ozero) w = w := by
  obtain ⟨⟨w0, w1, w2, w3, w4, w5, w6, w7⟩, ⟨w8, w9, w10, w11, w12, w13, w14, w15⟩⟩ := w
  simp only [sadd, oadd, ozero, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩ <;> ring

/-- If u·v = 0 then (u + δ)·v = δ·v exactly: the leakage *is* the perturbation's product. -/
theorem residual_eq (u δ v : O × O) (h : smul u v = (ozero, ozero)) :
    smul (sadd u δ) v = smul δ v := by
  rw [smul_add_left, h, sadd_zero_left]

/-! ### Sub-multiplicativity ‖x·y‖² ≤ 2‖x‖²‖y‖² -/

theorem snorm_smul_le (x y : O × O) : snorm2 (smul x y) ≤ 2 * snorm2 x * snorm2 y := by
  obtain ⟨a, b⟩ := x
  obtain ⟨c, d⟩ := y
  simp only [snorm2, onorm2, smul]
  have h1 := osub_norm_le (omul a c) (omul (conj d) b)
  have h2 := oadd_norm_le (omul d a) (omul b (conj c))
  rw [composition, composition, conj_norm] at h1
  rw [composition, composition, conj_norm] at h2
  nlinarith [h1, h2, dot_nonneg a, dot_nonneg b, dot_nonneg c, dot_nonneg d]

/-! ### Main theorem -/

/-- **Residual-leakage bound.**  If the nulling weight v annihilates the estimated signature u
    (u·v = 0) but the true signature is u + δ, then
        ‖(u + δ)·v‖² ≤ 2 ‖δ‖² ‖v‖².
    Null depth is therefore bounded by the signature-estimation error, independent of the
    cone geometry used to construct v. -/
theorem leakage_bound (u δ v : O × O) (h : smul u v = (ozero, ozero)) :
    snorm2 (smul (sadd u δ) v) ≤ 2 * snorm2 δ * snorm2 v := by
  rw [residual_eq u δ v h]
  exact snorm_smul_le δ v

/-- Corollary in "dB" form, as the bound the experiment plots: for unit ‖v‖² = 1,
    ‖(u+δ)·v‖² ≤ 2‖δ‖². -/
theorem leakage_bound_unit (u δ v : O × O) (h : smul u v = (ozero, ozero))
    (hv : snorm2 v = 1) :
    snorm2 (smul (sadd u δ) v) ≤ 2 * snorm2 δ := by
  have := leakage_bound u δ v h
  rw [hv] at this
  linarith

theorem snorm2_nonneg (u : O × O) : 0 ≤ snorm2 u := by
  simp only [snorm2, onorm2]
  have := dot_nonneg u.1; have := dot_nonneg u.2; linarith

/-- **Cascade leakage bound.**  Two-stage nuller: if the estimated signature u is annihilated by the
    cascade, ((u·v₁)·v₂) = 0, then for the true signature u + δ
        ‖((u+δ)·v₁)·v₂‖² ≤ 4 ‖δ‖² ‖v₁‖² ‖v₂‖².
    (The hypothesis covers both a stage-1 null and a stage-2 null of the stage-1 residual.) -/
theorem cascade_leakage_bound (u δ v₁ v₂ : O × O)
    (h : smul (smul u v₁) v₂ = (ozero, ozero)) :
    snorm2 (smul (smul (sadd u δ) v₁) v₂) ≤ 4 * snorm2 δ * snorm2 v₁ * snorm2 v₂ := by
  rw [smul_add_left, smul_add_left, h, sadd_zero_left]
  have h1 := snorm_smul_le (smul δ v₁) v₂
  have h2 := snorm_smul_le δ v₁
  have hv2 := snorm2_nonneg v₂
  nlinarith [mul_le_mul_of_nonneg_right h2 hv2]

/-! ### Frobenius Trace Identity, 4D Kernel Subspace Leakage, and Shannon Dimension Bounds -/

def e0  : O × O := (⟨1,0,0,0,0,0,0,0⟩, ozero)
def e1  : O × O := (⟨0,1,0,0,0,0,0,0⟩, ozero)
def e2  : O × O := (⟨0,0,1,0,0,0,0,0⟩, ozero)
def e3  : O × O := (⟨0,0,0,1,0,0,0,0⟩, ozero)
def e4  : O × O := (⟨0,0,0,0,1,0,0,0⟩, ozero)
def e5  : O × O := (⟨0,0,0,0,0,1,0,0⟩, ozero)
def e6  : O × O := (⟨0,0,0,0,0,0,1,0⟩, ozero)
def e7  : O × O := (⟨0,0,0,0,0,0,0,1⟩, ozero)
def e8  : O × O := (ozero, ⟨1,0,0,0,0,0,0,0⟩)
def e9  : O × O := (ozero, ⟨0,1,0,0,0,0,0,0⟩)
def e10 : O × O := (ozero, ⟨0,0,1,0,0,0,0,0⟩)
def e11 : O × O := (ozero, ⟨0,0,0,1,0,0,0,0⟩)
def e12 : O × O := (ozero, ⟨0,0,0,0,1,0,0,0⟩)
def e13 : O × O := (ozero, ⟨0,0,0,0,0,1,0,0⟩)
def e14 : O × O := (ozero, ⟨0,0,0,0,0,0,1,0⟩)
def e15 : O × O := (ozero, ⟨0,0,0,0,0,0,0,1⟩)

/-- **Right-multiplication Frobenius trace identity (`tr(R(v)ᵀ R(v)) = 16 ‖v‖²`).**
    Across the 16 canonical orthonormal basis directions `e0..e15` of `S ≅ ℝ¹⁶`,
    the sum of squared output norms equals `16 ‖v‖²`. Together with the eigenvalue
    partition `{0, 1, 2}` of multiplicities `{4, 8, 4}` on the unit Moreno cone
    (`4·0 + 8·1 + 4·2 = 16`), this locks the total spectral energy across all 16 dimensions. -/
theorem frobenius_norm_right (v : O × O) :
    snorm2 (smul e0 v)  + snorm2 (smul e1 v)  + snorm2 (smul e2 v)  + snorm2 (smul e3 v) +
    snorm2 (smul e4 v)  + snorm2 (smul e5 v)  + snorm2 (smul e6 v)  + snorm2 (smul e7 v) +
    snorm2 (smul e8 v)  + snorm2 (smul e9 v)  + snorm2 (smul e10 v) + snorm2 (smul e11 v) +
    snorm2 (smul e12 v) + snorm2 (smul e13 v) + snorm2 (smul e14 v) + snorm2 (smul e15 v)
    = 16 * snorm2 v := by
  obtain ⟨⟨v0, v1, v2, v3, v4, v5, v6, v7⟩, ⟨v8, v9, v10, v11, v12, v13, v14, v15⟩⟩ := v
  simp only [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
             ozero, smul, omul, conj, osub, oadd, snorm2, onorm2, dot]
  ring

/-- **Left-multiplication Frobenius trace identity (`tr(L(u)ᵀ L(u)) = 16 ‖u‖²`).** -/
theorem frobenius_norm_left (u : O × O) :
    snorm2 (smul u e0)  + snorm2 (smul u e1)  + snorm2 (smul u e2)  + snorm2 (smul u e3) +
    snorm2 (smul u e4)  + snorm2 (smul u e5)  + snorm2 (smul u e6)  + snorm2 (smul u e7) +
    snorm2 (smul u e8)  + snorm2 (smul u e9)  + snorm2 (smul u e10) + snorm2 (smul u e11) +
    snorm2 (smul u e12) + snorm2 (smul u e13) + snorm2 (smul u e14) + snorm2 (smul u e15)
    = 16 * snorm2 u := by
  obtain ⟨⟨u0, u1, u2, u3, u4, u5, u6, u7⟩, ⟨u8, u9, u10, u11, u12, u13, u14, u15⟩⟩ := u
  simp only [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
             ozero, smul, omul, conj, osub, oadd, snorm2, onorm2, dot]
  ring

/-- **4D kernel subspace perturbation bound.**
    If `u₁, u₂, u₃, u₄` span the 4-dimensional annihilated subspace (`uₖ · v = 0`) and each
    mode suffers a perturbation `δₖ`, the total residual energy leaked across all four
    nulled dimensions is bounded by `2 (∑ₖ ‖δₖ‖²) ‖v‖²`. -/
theorem kernel_subspace_leakage_bound (u₁ u₂ u₃ u₄ δ₁ δ₂ δ₃ δ₄ v : O × O)
    (h1 : smul u₁ v = (ozero, ozero))
    (h2 : smul u₂ v = (ozero, ozero))
    (h3 : smul u₃ v = (ozero, ozero))
    (h4 : smul u₄ v = (ozero, ozero)) :
    snorm2 (smul (sadd u₁ δ₁) v) + snorm2 (smul (sadd u₂ δ₂) v) +
    snorm2 (smul (sadd u₃ δ₃) v) + snorm2 (smul (sadd u₄ δ₄) v) ≤
    2 * (snorm2 δ₁ + snorm2 δ₂ + snorm2 δ₃ + snorm2 δ₄) * snorm2 v := by
  have b1 := leakage_bound u₁ δ₁ v h1
  have b2 := leakage_bound u₂ δ₂ v h2
  have b3 := leakage_bound u₃ δ₃ v h3
  have b4 := leakage_bound u₄ δ₄ v h4
  linarith

/-- **Calibration floor threshold (`‖(u + δ)·v‖² ≤ ε²`).**
    When unit-weight calibration satisfies `2 ‖δ‖² ≤ ε²` (i.e., `‖δ‖ ≤ ε / √2` for hardware
    quantisation or thermal noise radius `ε`), the leaked interferer norm squared is bounded
    by `ε²`, collapsing the nulled semi-axis below the `ε`-ball covering resolution. -/
theorem calibration_floor_bound (u δ v : O × O) (ε_sq : ℝ)
    (h : smul u v = (ozero, ozero)) (hv : snorm2 v = 1)
    (hcal : 2 * snorm2 δ ≤ ε_sq) :
    snorm2 (smul (sadd u δ) v) ≤ ε_sq := by
  have hleak := leakage_bound_unit u δ v h hv
  linarith

/-- **Ergodic PG(3,2) Shannon dimension-rate bounds.**
    For duty fractions `p₀, p₁, p₂ ≥ 0` with `p₀ + p₁ + p₂ = 1` across rank-16 octonion
    tracking (`F_alt`), rank-12 single-stage excision (`F_ZD`), and rank-8 two-stage
    cascaded excision, the effective dimension per snapshot `D = 16 - 4 p₁ - 8 p₂`
    equals the convex combination `16 p₀ + 12 p₁ + 8 p₂` and satisfies `8 ≤ D ≤ 16`. -/
theorem ergodic_dimension_bounds (p₀ p₁ p₂ : ℝ)
    (hp0 : 0 ≤ p₀) (hp1 : 0 ≤ p₁) (hp2 : 0 ≤ p₂)
    (hsum : p₀ + p₁ + p₂ = 1) :
    16 * p₀ + 12 * p₁ + 8 * p₂ = 16 - 4 * p₁ - 8 * p₂ ∧
    8 ≤ 16 - 4 * p₁ - 8 * p₂ ∧
    16 - 4 * p₁ - 8 * p₂ ≤ 16 := by
   refine ⟨by linarith, by linarith, by linarith⟩

/-! ### Co-Designed PG(3,2) Sedenion Transceiver: 8 ⊕ 4 ⊕ 4 Decomposition & Idempotent Recovery -/

/-- Inner product on S = O × O -/
def sdot (u v : O × O) : ℝ := dot u.1 v.1 + dot u.2 v.2

/-- Scalar multiplication on O and S = O × O -/
def oscal (c : ℝ) (x : O) : O :=
  ⟨c * x.c0, c * x.c1, c * x.c2, c * x.c3, c * x.c4, c * x.c5, c * x.c6, c * x.c7⟩
def sscal (c : ℝ) (u : O × O) : O × O := (oscal c u.1, oscal c u.2)

/-- Canonical ternary Moreno blade pair on mixed line (m=3, n=4): `u_pos = e_3 + e_12`, `u_neg = e_3 - e_12`. -/
def u_pos : O × O := (⟨0,0,0,1,0,0,0,0⟩, ⟨0,0,0,0,1,0,0,0⟩)
def u_neg : O × O := (⟨0,0,0,1,0,0,0,0⟩, ⟨0,0,0,0,-1,0,0,0⟩)

/-- 4D Moreno Fibre `W_pos(a1,a2,a3,a4) = ker L(u_pos)` spanned by ternary 2-sparse blades
    `e_1 + e_14`, `e_2 - e_13`, `e_5 + e_10`, `e_6 - e_9`. -/
def W_pos (a1 a2 a3 a4 : ℝ) : O × O :=
  (⟨0, a1, a2, 0, 0, a3, a4, 0⟩, ⟨0, -a4, a3, 0, 0, -a2, a1, 0⟩)

/-- 4D Dual Moreno Fibre `W_neg(b1,b2,b3,b4) = ker L(u_neg)` spanned by ternary 2-sparse blades
    `e_1 - e_14`, `e_2 + e_13`, `e_5 - e_10`, `e_6 + e_9`. -/
def W_neg (b1 b2 b3 b4 : ℝ) : O × O :=
  (⟨0, b1, b2, 0, 0, b3, b4, 0⟩, ⟨0, b4, -b3, 0, 0, b2, -b1, 0⟩)

/-- 8D Alternative Octonion Core `A_core` spanned by `{e_0, e_3, e_4, e_7, e_8, e_11, e_12, e_15}`. -/
def A_core (g0 g3 g4 g7 g8 g11 g12 g15 : ℝ) : O × O :=
  (⟨g0, 0, 0, g3, g4, 0, 0, g7⟩, ⟨g8, 0, 0, g11, g12, 0, 0, g15⟩)

/-- **1. Orthogonal 8 ⊕ 4 ⊕ 4 Transceiver Decomposition:**
    `W_pos`, `W_neg`, and `A_core` are mutually orthogonal in `S ≅ ℝ¹⁶`. -/
theorem transceiver_orthogonality (a1 a2 a3 a4 b1 b2 b3 b4 g0 g3 g4 g7 g8 g11 g12 g15 : ℝ) :
    sdot (W_pos a1 a2 a3 a4) (W_neg b1 b2 b3 b4) = 0 ∧
    sdot (A_core g0 g3 g4 g7 g8 g11 g12 g15) (W_pos a1 a2 a3 a4) = 0 ∧
    sdot (A_core g0 g3 g4 g7 g8 g11 g12 g15) (W_neg b1 b2 b3 b4) = 0 := by
  simp only [W_pos, W_neg, A_core, sdot, dot]
  refine ⟨by ring, by ring, by ring⟩

/-- **2. Exact Zero-Divisor Multi-Access Annihilation:**
    `u_pos · W_pos(a) = 0` and `u_neg · W_neg(b) = 0` identically for all `a, b ∈ ℝ⁴`. -/
theorem transceiver_zd_annihilation (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) :
    smul u_pos (W_pos a1 a2 a3 a4) = (ozero, ozero) ∧
    smul u_neg (W_neg b1 b2 b3 b4) = (ozero, ozero) := by
  simp only [u_pos, u_neg, W_pos, W_neg, ozero, smul, omul, conj, osub, oadd, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨⟨?_,?_,?_,?_,?_,?_,?_,?_⟩, ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩⟩,
          ⟨⟨?_,?_,?_,?_,?_,?_,?_,?_⟩, ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩⟩⟩ <;> ring

/-- **3. Fibre Invariance & Built-in Complex Structure (`L(u_neg) W_pos = W_pos`):**
    Left-multiplying `W_pos(a1,a2,a3,a4)` by the anti-partner blade `u_neg` maps `W_pos`
    onto itself via the 90° complex structure `(a1, a2, a3, a4) ↦ 2 (-a2, a1, a4, -a3)`. -/
theorem transceiver_antipartner_invariance (a1 a2 a3 a4 : ℝ) :
    smul u_neg (W_pos a1 a2 a3 a4) = sscal 2 (W_pos (-a2) a1 a4 (-a3)) := by
  simp only [u_neg, W_pos, sscal, oscal, smul, omul, conj, osub, oadd, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_,?_,?_,?_,?_,?_,?_,?_⟩, ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩⟩ <;> ring

/-- **4. Exact Complex-Structure Square (`J² = -4 I_4` for unnormalized `u_neg`, or `-2 I_4` for unit `u_neg/√2`):** -/
theorem transceiver_complex_structure_sq (a1 a2 a3 a4 : ℝ) :
    smul u_neg (smul u_neg (W_pos a1 a2 a3 a4)) = sscal (-4) (W_pos a1 a2 a3 a4) := by
  simp only [u_neg, W_pos, sscal, oscal, smul, omul, conj, osub, oadd, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_,?_,?_,?_,?_,?_,?_,?_⟩, ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩⟩ <;> ring

/-- **5. Joint Multiplier-Free Decoding & Co-Channel Interference Annihilation:**
    Given any superposition `X = W_pos(a) + W_neg(b)` of an authentic 4-symbol stream `W_pos(a)`
    and an interfering/co-channel 4-symbol stream `W_neg(b)`, double left-multiplication by `u_neg`
    recovers `-4 W_pos(a)` identically (a 2-bit shift + negation) while annihilating `W_neg(b)` to zero. -/
theorem transceiver_joint_decode_and_null (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) :
    smul u_neg (smul u_neg (sadd (W_pos a1 a2 a3 a4) (W_neg b1 b2 b3 b4))) =
    sscal (-4) (W_pos a1 a2 a3 a4) := by
  simp only [u_neg, W_pos, W_neg, sadd, sscal, oscal, smul, omul, conj, osub, oadd, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_,?_,?_,?_,?_,?_,?_,?_⟩, ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩⟩ <;> ring

/-! ### Pivot 1 & Pivot 2: Dual-Pol RHCP/LHCP Co-Linear Preservation & Passive RF S-Matrix Bounds -/

/-- **1. RHCP Hurwitz Norm Preservation (`‖(a, 0) · v‖² = ‖a‖² ‖v‖²`):**
    Any purely RHCP satellite signal `X_sat = (a, 0) ∈ O ⊕ {0}` multiplied by ANY sedenion
    weight `v = (c, d) ∈ S` (including any Moreno zero divisor!) experiences exact Hurwitz
    norm preservation (`0.0000 dB` insertion loss), regardless of angle of arrival or phase. -/
theorem rhcp_satellite_norm_preservation (a : O) (v : O × O) :
    snorm2 (smul (a, ozero) v) = onorm2 a * snorm2 v := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨⟨c0, c1, c2, c3, c4, c5, c6, c7⟩, ⟨d0, d1, d2, d3, d4, d5, d6, d7⟩⟩ := v
  simp only [ozero, smul, omul, conj, osub, oadd, snorm2, onorm2, dot]
  ring

/-- **2. LHCP Hurwitz Norm Preservation (`‖(0, b) · v‖² = ‖b‖² ‖v‖²`):**
    Symmetrically, any purely LHCP state `(0, b) ∈ {0} ⊕ O e_8` obeys exact norm multiplicativity
    under right-multiplication by any sedenion `v ∈ S`. Only mixed dual-polarised states `(a, b)`
    on the Moreno cone `‖a‖ = ‖b‖, ⟨a, b⟩ = 0` can be annihilated. -/
theorem lhcp_satellite_norm_preservation (b : O) (v : O × O) :
    snorm2 (smul (ozero, b) v) = onorm2 b * snorm2 v := by
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨⟨c0, c1, c2, c3, c4, c5, c6, c7⟩, ⟨d0, d1, d2, d3, d4, d5, d6, d7⟩⟩ := v
  simp only [ozero, smul, omul, conj, osub, oadd, snorm2, onorm2, dot]
  ring

/-- **3. Co-Linear (`Δθ = 0°`) Dual-Pol Jammer Annihilation with Lossless RHCP Preservation:**
    Even when a linearly polarised jammer `u_jam = (a_J, b_J)` arrives from the EXACT same
    spatial direction as an RHCP satellite `X_sat = (a_sat, 0)` (including `a_sat = a_J`),
    a unit Moreno annihilator `v` (`‖v‖² = 1`) annihilates `u_jam` identically while preserving
    `100%` of the RHCP satellite energy (`‖(a_sat, 0) · v‖² = ‖a_sat‖²`). -/
theorem colinear_dualpol_jammer_null_and_sat_preserve (a_sat : O) (u_jam v : O × O)
    (hnull : smul u_jam v = (ozero, ozero)) (hv : snorm2 v = 1) :
    smul u_jam v = (ozero, ozero) ∧ snorm2 (smul (a_sat, ozero) v) = onorm2 a_sat := by
  refine ⟨hnull, ?_⟩
  rw [rhcp_satellite_norm_preservation, hv, mul_one]

/-- **4. Passive RF Metamaterial / Stripline S-Parameter Bound (`‖X · v_rf‖² ≤ ‖X‖²`):**
    Scaling a unit sedenion weight by `1/√2` (`‖v_rf‖² = 1/2`) yields a strictly passive
    16-port transmission scattering operator `S_21(X) = X · v_rf` (`I - S_21ᵀ S_21 ≥ 0`),
    realisable in passive reciprocal stripline without amplifiers or ferrites. -/
theorem rf_smatrix_passivity_bound (x v_rf : O × O) (hv : snorm2 v_rf = 1 / 2) :
    snorm2 (smul x v_rf) ≤ snorm2 x := by
  have h := snorm_smul_le x v_rf
  rw [hv] at h
  linarith

/-! ### Pivot 2 (constructive): Dual-Pol Co-Linear Preservation on top of `SedenionReal.nullspace` -/

/-- Sedenion conjugate `(a, b)* = (a*, -b)`. -/
def sconj (u : O × O) : O × O := (conj u.1, oscal (-1) u.2)

/-- **Right-multiplication adjoint identity (`R(v)ᵀ = R(v*)`).**
    `⟨x·v, y⟩ = ⟨x, y·v*⟩` for all `x, y, v ∈ S`. Hence the 32-port transmission matrix
    `S₃₂ = [[0, S₂₁ᵀ], [S₂₁, 0]]` with `S₂₁ = R(v)/√2` has `S₂₁ᵀ = R(v*)/√2`: the reverse path
    is the same passive network driven by the conjugate weight (reciprocity), no ferrites. -/
theorem right_mul_adjoint (x y v : O × O) :
    sdot (smul x v) y = sdot x (smul y (sconj v)) := by
  obtain ⟨⟨x0, x1, x2, x3, x4, x5, x6, x7⟩, ⟨x8, x9, x10, x11, x12, x13, x14, x15⟩⟩ := x
  obtain ⟨⟨y0, y1, y2, y3, y4, y5, y6, y7⟩, ⟨y8, y9, y10, y11, y12, y13, y14, y15⟩⟩ := y
  obtain ⟨⟨v0, v1, v2, v3, v4, v5, v6, v7⟩, ⟨v8, v9, v10, v11, v12, v13, v14, v15⟩⟩ := v
  simp only [sconj, oscal, sdot, smul, omul, conj, osub, oadd, dot]
  ring

/-- **Dual-Pol Co-Linear Preservation Theorem (Pivot 1 + Pivot 2, constructive).**
    An RHCP satellite `X_sat = (a_sat, 0)` and a linearly polarised jammer `(a_J, b_J)` on the
    Moreno cone arrive from the *same* direction (any `a_sat`, including `a_sat = a_J`). For the
    annihilator `v(c) = (c, partner a_J b_J c)` of `SedenionReal.nullspace` with `‖c‖² = 1/2` (unit weight):
      • `(a_J, b_J) · v(c) = 0`                         — jammer annihilated,
      • `‖(a_sat, 0) · v(c)‖² = ‖a_sat‖²`               — satellite at 0.0000 dB loss,
      • `‖v(c)‖² = 1`.
    No Householder gauge `T` appears: the embedding is the identity on `S`. -/
theorem dualpol_colinear_preservation (a_sat a_J b_J c : O)
    (ha0 : a_J.c0 = 0) (hb0 : b_J.c0 = 0) (hc0 : c.c0 = 0)
    (haa : dot a_J a_J = 1) (hbb : dot b_J b_J = 1) (hab : dot a_J b_J = 0)
    (hca : dot a_J c = 0) (hcb : dot b_J c = 0) (hcab : dot (omul a_J b_J) c = 0)
    (hc : onorm2 c = 1 / 2) :
    smul (a_J, b_J) ((c, partner a_J b_J c)) = (ozero, ozero) ∧
    snorm2 (smul (a_sat, ozero) ((c, partner a_J b_J c))) = onorm2 a_sat ∧
    snorm2 ((c, partner a_J b_J c)) = 1 := by
  have hv : snorm2 (c, partner a_J b_J c) = 1 := by
    simp only [snorm2, onorm2] at hc ⊢; rw [partner_norm a_J b_J c haa hbb]; linarith
  refine ⟨nullspace a_J b_J c ha0 hb0 hc0 hab hca hcb hcab haa hbb, ?_, hv⟩
  rw [rhcp_satellite_norm_preservation, hv, mul_one]

/-- **Passive RF realisation of the dual-pol excision stage.**
    Scaling the unit annihilator to `v_rf = v(c)/√2` (i.e. `‖c‖² = 1/4`) gives a transmission
    operator `S₂₁(X) = X · v_rf` that (i) is strictly passive on every input, `‖X·v_rf‖² ≤ ‖X‖²`,
    (ii) still annihilates the cone jammer, and (iii) passes the RHCP satellite at exactly
    `‖a_sat‖²/2` (the −3 dB hybrid split, recovered losslessly by the 1/√2 pair). -/
theorem dualpol_passive_rf_stage (a_sat a_J b_J c : O)
    (ha0 : a_J.c0 = 0) (hb0 : b_J.c0 = 0) (hc0 : c.c0 = 0)
    (haa : dot a_J a_J = 1) (hbb : dot b_J b_J = 1) (hab : dot a_J b_J = 0)
    (hca : dot a_J c = 0) (hcb : dot b_J c = 0) (hcab : dot (omul a_J b_J) c = 0)
    (hc : onorm2 c = 1 / 4) :
    (∀ x : O × O, snorm2 (smul x ((c, partner a_J b_J c))) ≤ snorm2 x) ∧
    smul (a_J, b_J) ((c, partner a_J b_J c)) = (ozero, ozero) ∧
    snorm2 (smul (a_sat, ozero) ((c, partner a_J b_J c))) = onorm2 a_sat / 2 := by
  have hv : snorm2 (c, partner a_J b_J c) = 1 / 2 := by
    simp only [snorm2, onorm2] at hc ⊢; rw [partner_norm a_J b_J c haa hbb]; linarith
  refine ⟨fun x => rf_smatrix_passivity_bound x _ hv,
          nullspace a_J b_J c ha0 hb0 hc0 hab hca hcb hcab haa hbb, ?_⟩
  rw [rhcp_satellite_norm_preservation, hv]; ring

/-! ### Pivot 2 (closed form): linearly polarised jammer family under the fixed RHCP/LHCP wiring -/

/-- Scaling commutes with the product: `x · (k v) = k (x · v)`. -/
theorem smul_sscal_right (x v : O × O) (k : ℝ) : smul x (sscal k v) = sscal k (smul x v) := by
  obtain ⟨⟨x0, x1, x2, x3, x4, x5, x6, x7⟩, ⟨x8, x9, x10, x11, x12, x13, x14, x15⟩⟩ := x
  obtain ⟨⟨v0, v1, v2, v3, v4, v5, v6, v7⟩, ⟨v8, v9, v10, v11, v12, v13, v14, v15⟩⟩ := v
  simp only [sscal, oscal, smul, omul, conj, osub, oadd, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩ <;> ring

theorem sscal_zero (k : ℝ) : sscal k (ozero, ozero) = (ozero, ozero) := by
  simp only [sscal, oscal, ozero, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩ <;> ring

/-- **Dual-pol wiring (hardware, data-independent).** The RHCP port's I/Q pair feeds lanes
    `(e1, e2)` and the LHCP port's I/Q pair feeds lanes `(e5, e6)`. A linearly polarised wave with
    tilt `ψ` has LHCP = RHCP rotated by `2ψ`: writing `(cψ, sψ) = (cos 2ψ, sin 2ψ)` and the complex
    sample `s = sI + j sQ`,
        `X_J(ψ, s) = (sI e1 + sQ e2 ,  (sI cψ − sQ sψ) e5 + (sI sψ + sQ cψ) e6)`. -/
def jam_linear (cψ sψ sI sQ : ℝ) : O × O :=
  (⟨0, sI, sQ, 0, 0, 0, 0, 0⟩, ⟨0, 0, 0, 0, 0, sI * cψ - sQ * sψ, sI * sψ + sQ * cψ, 0⟩)

/-- The tilt-dependent zero divisor `v(ψ) = e3 + sin 2ψ · e12 − cos 2ψ · e15`. -/
def v_tilt (cψ sψ : ℝ) : O × O := (⟨0, 0, 0, 1, 0, 0, 0, 0⟩, ⟨0, 0, 0, 0, sψ, 0, 0, -cψ⟩)

/-- Every linearly polarised sample lies on the Moreno cone: `‖a_J‖ = ‖b_J‖`, `⟨a_J, b_J⟩ = 0`. -/
theorem jam_linear_on_cone (cψ sψ sI sQ : ℝ) (h : cψ ^ 2 + sψ ^ 2 = 1) :
    onorm2 (jam_linear cψ sψ sI sQ).1 = onorm2 (jam_linear cψ sψ sI sQ).2 ∧
    dot (jam_linear cψ sψ sI sQ).1 (jam_linear cψ sψ sI sQ).2 = 0 := by
  simp only [jam_linear, onorm2, dot]
  constructor
  · linear_combination (-(sI ^ 2 + sQ ^ 2)) * h
  · ring

/-- **Whole-waveform annihilation.** One fixed `v(ψ)` kills *every* complex sample of the
    tilt-`ψ` linearly polarised jammer: `X_J(ψ, s) · v(ψ) = 0` for all `(sI, sQ)`. (The generic
    point annihilator `v(c)` of a single sample does not do this; the wiring `(e1,e2)/(e5,e6)` is
    what makes the 2-D I/Q signal subspace sit inside one 4-D Moreno fibre.) -/
theorem dualpol_tilt_annihilation (cψ sψ sI sQ : ℝ) (h : cψ ^ 2 + sψ ^ 2 = 1) :
    smul (jam_linear cψ sψ sI sQ) (v_tilt cψ sψ) = (ozero, ozero) := by
  simp only [jam_linear, v_tilt, ozero, smul, omul, conj, osub, oadd, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩ <;>
    first | ring1 | linear_combination (-sQ) * h | linear_combination sI * h

theorem v_tilt_norm (cψ sψ : ℝ) (h : cψ ^ 2 + sψ ^ 2 = 1) : snorm2 (v_tilt cψ sψ) = 2 := by
  simp only [v_tilt, snorm2, onorm2, dot]
  linear_combination h

/-- **Dual-Pol Co-Linear Preservation Theorem, closed form (Δθ = 0°, T = I).**
    For every tilt `ψ`, every jammer sample `s`, and every RHCP satellite sample `a_sat`:
      • `X_J(ψ, s) · v(ψ) = 0`,
      • `‖(a_sat, 0) · v(ψ)‖² = ‖v(ψ)‖² ‖a_sat‖² = 2 ‖a_sat‖²`   (unit weight `v(ψ)/√2` ⇒ exactly `‖a_sat‖²`, 0.0000 dB). -/
theorem dualpol_tilt_colinear_preservation (cψ sψ sI sQ : ℝ) (a_sat : O) (h : cψ ^ 2 + sψ ^ 2 = 1) :
    smul (jam_linear cψ sψ sI sQ) (v_tilt cψ sψ) = (ozero, ozero) ∧
    snorm2 (smul (a_sat, ozero) (v_tilt cψ sψ)) = 2 * onorm2 a_sat := by
  refine ⟨dualpol_tilt_annihilation cψ sψ sI sQ h, ?_⟩
  rw [rhcp_satellite_norm_preservation, v_tilt_norm cψ sψ h]; ring

/-- **Passive RF stage, closed form.** `v_rf(ψ) = v(ψ)/2` has `‖v_rf‖² = 1/2`, so `S₂₁(X) = X · v_rf`
    is passive on every input, still annihilates the whole jammer waveform, and passes the RHCP
    satellite at exactly `‖a_sat‖²/2` (the hybrid's −3 dB split, lossless across the 1/√2 pair).
    The nonzero entries of `v_rf` are `{1/2, ±sin2ψ/2, ∓cos2ψ/2}`: two couplers per lane. -/
theorem dualpol_tilt_passive_rf_stage (cψ sψ sI sQ : ℝ) (a_sat : O) (h : cψ ^ 2 + sψ ^ 2 = 1) :
    (∀ x : O × O, snorm2 (smul x (sscal (1 / 2) (v_tilt cψ sψ))) ≤ snorm2 x) ∧
    smul (jam_linear cψ sψ sI sQ) (sscal (1 / 2) (v_tilt cψ sψ)) = (ozero, ozero) ∧
    snorm2 (smul (a_sat, ozero) (sscal (1 / 2) (v_tilt cψ sψ))) = onorm2 a_sat / 2 := by
  have hv : snorm2 (sscal (1 / 2) (v_tilt cψ sψ)) = 1 / 2 := by
    simp only [v_tilt, sscal, oscal, snorm2, onorm2, dot]
    linear_combination (1 / 4) * h
  refine ⟨fun x => rf_smatrix_passivity_bound x _ hv, ?_, ?_⟩
  · rw [smul_sscal_right, dualpol_tilt_annihilation cψ sψ sI sQ h, sscal_zero]
  · rw [rhcp_satellite_norm_preservation, hv]; ring

end SedenionRobust
