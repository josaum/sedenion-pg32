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

namespace SedenionRobust

@[ext] structure O where
  (c0 c1 c2 c3 c4 c5 c6 c7 : ℝ)

def conj (x : O) : O := ⟨x.c0, -x.c1, -x.c2, -x.c3, -x.c4, -x.c5, -x.c6, -x.c7⟩

def omul (x y : O) : O := ⟨
  x.c0 * y.c0 - x.c1 * y.c1 - x.c2 * y.c2 - x.c3 * y.c3 - x.c4 * y.c4 - x.c5 * y.c5 - x.c6 * y.c6 - x.c7 * y.c7,
  x.c0 * y.c1 + x.c1 * y.c0 + x.c2 * y.c3 - x.c3 * y.c2 + x.c4 * y.c5 - x.c5 * y.c4 - x.c6 * y.c7 + x.c7 * y.c6,
  x.c0 * y.c2 - x.c1 * y.c3 + x.c2 * y.c0 + x.c3 * y.c1 + x.c4 * y.c6 + x.c5 * y.c7 - x.c6 * y.c4 - x.c7 * y.c5,
  x.c0 * y.c3 + x.c1 * y.c2 - x.c2 * y.c1 + x.c3 * y.c0 + x.c4 * y.c7 - x.c5 * y.c6 + x.c6 * y.c5 - x.c7 * y.c4,
  x.c0 * y.c4 - x.c1 * y.c5 - x.c2 * y.c6 - x.c3 * y.c7 + x.c4 * y.c0 + x.c5 * y.c1 + x.c6 * y.c2 + x.c7 * y.c3,
  x.c0 * y.c5 + x.c1 * y.c4 - x.c2 * y.c7 + x.c3 * y.c6 - x.c4 * y.c1 + x.c5 * y.c0 - x.c6 * y.c3 + x.c7 * y.c2,
  x.c0 * y.c6 + x.c1 * y.c7 + x.c2 * y.c4 - x.c3 * y.c5 - x.c4 * y.c2 + x.c5 * y.c3 + x.c6 * y.c0 - x.c7 * y.c1,
  x.c0 * y.c7 - x.c1 * y.c6 + x.c2 * y.c5 + x.c3 * y.c4 - x.c4 * y.c3 - x.c5 * y.c2 + x.c6 * y.c1 + x.c7 * y.c0⟩

def dot (x y : O) : ℝ :=
  x.c0 * y.c0 + x.c1 * y.c1 + x.c2 * y.c2 + x.c3 * y.c3 + x.c4 * y.c4 + x.c5 * y.c5 + x.c6 * y.c6 + x.c7 * y.c7

def osub (x y : O) : O := ⟨x.c0 - y.c0, x.c1 - y.c1, x.c2 - y.c2, x.c3 - y.c3, x.c4 - y.c4, x.c5 - y.c5, x.c6 - y.c6, x.c7 - y.c7⟩
def oadd (x y : O) : O := ⟨x.c0 + y.c0, x.c1 + y.c1, x.c2 + y.c2, x.c3 + y.c3, x.c4 + y.c4, x.c5 + y.c5, x.c6 + y.c6, x.c7 + y.c7⟩
def ozero : O := ⟨0, 0, 0, 0, 0, 0, 0, 0⟩

/-- sedenion product (Cayley–Dickson doubling) -/
def smul (u v : O × O) : O × O :=
  (osub (omul u.1 v.1) (omul (conj v.2) u.2), oadd (omul v.2 u.1) (omul u.2 (conj v.1)))

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

theorem composition (x y : O) : dot (omul x y) (omul x y) = dot x x * dot y y := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [dot, omul]; ring

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

/-! ## Phase invariance of the annihilator fibre

For a cone point u = (a, b), multiplication by i in the C^8 ≅ R^16 embedding is
J(a, b) = (−b, a).  We show ker L(u) ⊆ ker L(J u): every fibre vector v(c) = (c, (ac)b) that
annihilates u also annihilates J u, hence annihilates u·e^{iφ} for every carrier phase φ.
The key octonion identity is (ac)b = −(bc)a for mutually orthogonal imaginary a, b, c. -/

def oneg (x : O) : O := ⟨-x.c0, -x.c1, -x.c2, -x.c3, -x.c4, -x.c5, -x.c6, -x.c7⟩

/-- the constructive partner d = (a c) b of Theorem 5.1 -/
def partner (a b c : O) : O := omul (omul a c) b

/-- J (a, b) = (−b, a): multiplication by i on C^8 ≅ R^16 -/
def Jc (u : O × O) : O × O := (oneg u.2, u.1)

/-- (ac)b + (bc)a = 2⟨a,b⟩c − 2⟨a,c⟩b − 2⟨b,c⟩a for imaginary octonions; hence
    (ac)b = −(bc)a when a, b, c are mutually orthogonal. -/
theorem partner_swap (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0)
    (hab : dot a b = 0) (hac : dot a c = 0) (hbc : dot b c = 0) :
    omul (omul a c) b = oneg (omul (omul b c) a) := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := c
  simp only at ha0 hb0 hc0
  subst ha0 hb0 hc0
  simp only [dot] at hab hac hbc
  simp only [omul, oneg, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · ring
  · linear_combination (2*c1) * hab + (-2*b1) * hac + (-2*a1) * hbc
  · linear_combination (2*c2) * hab + (-2*b2) * hac + (-2*a2) * hbc
  · linear_combination (2*c3) * hab + (-2*b3) * hac + (-2*a3) * hbc
  · linear_combination (2*c4) * hab + (-2*b4) * hac + (-2*a4) * hbc
  · linear_combination (2*c5) * hab + (-2*b5) * hac + (-2*a5) * hbc
  · linear_combination (2*c6) * hab + (-2*b6) * hac + (-2*a6) * hbc
  · linear_combination (2*c7) * hab + (-2*b7) * hac + (-2*a7) * hbc

theorem omul_oneg_left (x y : O) : omul (oneg x) y = oneg (omul x y) := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [omul, oneg, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

theorem oneg_oneg (x : O) : oneg (oneg x) = x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [oneg, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- partner (−b) a c = partner a b c under orthogonality -/
theorem partner_J (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0)
    (hab : dot a b = 0) (hac : dot a c = 0) (hbc : dot b c = 0) :
    partner (oneg b) a c = partner a b c := by
  unfold partner
  rw [omul_oneg_left, omul_oneg_left, partner_swap a b c ha0 hb0 hc0 hab hac hbc]

theorem dot_omul_J (a b c : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0) :
    dot (omul (oneg b) a) c = dot (omul a b) c := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := c
  simp only at ha0 hb0 hc0; subst ha0 hb0 hc0
  simp only [dot, omul, oneg]; ring

theorem dot_oneg_left (x y : O) : dot (oneg x) y = - dot x y := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [dot, oneg]; ring

theorem dot_oneg_oneg (x : O) : dot (oneg x) (oneg x) = dot x x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [dot, oneg]; ring

theorem dot_comm (x y : O) : dot x y = dot y x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [dot]; ring

/-- **Theorem 5.1 (Constructive 4D Zero-Divisor Nullspace Operator).**
    Let a, b, c be purely imaginary octonions with ‖a‖ = ‖b‖ = 1, ⟨a,b⟩ = 0 and
    c ⊥ span{a, b, ab}.  Then the sedenion u = (a, b) annihilates v = (c, (ac)b):  u·v = 0. -/
theorem nullspace
    (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0)
    (hab : dot a b = 0) (hac : dot a c = 0) (hbc : dot b c = 0) (habc : dot (omul a b) c = 0)
    (haa : dot a a = 1) (hbb : dot b b = 1) :
    smul (a, b) (c, partner a b c) = (ozero, ozero) := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := c
  simp only at ha0 hb0 hc0
  subst ha0 hb0 hc0
  simp only [dot, omul] at hab hac hbc habc haa hbb
  simp only [smul, partner, omul, conj, osub, oadd, ozero, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩
  · linear_combination (b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 + b7^2 - 1) * hac
  · linear_combination (-2*b1) * habc + (-a2*c3 + a3*c2 - a4*c5 + a5*c4 + a6*c7 - a7*c6) * hbb
  · linear_combination (-2*b2) * habc + (a1*c3 - a3*c1 - a4*c6 - a5*c7 + a6*c4 + a7*c5) * hbb
  · linear_combination (-2*b3) * habc + (-a1*c2 + a2*c1 - a4*c7 + a5*c6 - a6*c5 + a7*c4) * hbb
  · linear_combination (-2*b4) * habc + (a1*c5 + a2*c6 + a3*c7 - a5*c1 - a6*c2 - a7*c3) * hbb
  · linear_combination (-2*b5) * habc + (-a1*c4 + a2*c7 - a3*c6 + a4*c1 + a6*c3 - a7*c2) * hbb
  · linear_combination (-2*b6) * habc + (-a1*c7 - a2*c4 + a3*c5 + a4*c2 - a5*c3 + a7*c1) * hbb
  · linear_combination (-2*b7) * habc + (a1*c6 - a2*c5 - a3*c4 + a4*c3 + a5*c2 - a6*c1) * hbb
  · linear_combination (a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 + a7^2 + 1) * hbc
  · linear_combination (-2*a2*c3 + 2*a3*c2 - 2*a4*c5 + 2*a5*c4 + 2*a6*c7 - 2*a7*c6) * hab + (2*a2*b3 - 2*a3*b2 + 2*a4*b5 - 2*a5*b4 - 2*a6*b7 + 2*a7*b6) * hac + (b2*c3 - b3*c2 + b4*c5 - b5*c4 - b6*c7 + b7*c6) * haa
  · linear_combination (2*a1*c3 - 2*a3*c1 - 2*a4*c6 - 2*a5*c7 + 2*a6*c4 + 2*a7*c5) * hab + (-2*a1*b3 + 2*a3*b1 + 2*a4*b6 + 2*a5*b7 - 2*a6*b4 - 2*a7*b5) * hac + (-b1*c3 + b3*c1 + b4*c6 + b5*c7 - b6*c4 - b7*c5) * haa
  · linear_combination (-2*a1*c2 + 2*a2*c1 - 2*a4*c7 + 2*a5*c6 - 2*a6*c5 + 2*a7*c4) * hab + (2*a1*b2 - 2*a2*b1 + 2*a4*b7 - 2*a5*b6 + 2*a6*b5 - 2*a7*b4) * hac + (b1*c2 - b2*c1 + b4*c7 - b5*c6 + b6*c5 - b7*c4) * haa
  · linear_combination (2*a1*c5 + 2*a2*c6 + 2*a3*c7 - 2*a5*c1 - 2*a6*c2 - 2*a7*c3) * hab + (-2*a1*b5 - 2*a2*b6 - 2*a3*b7 + 2*a5*b1 + 2*a6*b2 + 2*a7*b3) * hac + (-b1*c5 - b2*c6 - b3*c7 + b5*c1 + b6*c2 + b7*c3) * haa
  · linear_combination (-2*a1*c4 + 2*a2*c7 - 2*a3*c6 + 2*a4*c1 + 2*a6*c3 - 2*a7*c2) * hab + (2*a1*b4 - 2*a2*b7 + 2*a3*b6 - 2*a4*b1 - 2*a6*b3 + 2*a7*b2) * hac + (b1*c4 - b2*c7 + b3*c6 - b4*c1 - b6*c3 + b7*c2) * haa
  · linear_combination (-2*a1*c7 - 2*a2*c4 + 2*a3*c5 + 2*a4*c2 - 2*a5*c3 + 2*a7*c1) * hab + (2*a1*b7 + 2*a2*b4 - 2*a3*b5 - 2*a4*b2 + 2*a5*b3 - 2*a7*b1) * hac + (b1*c7 + b2*c4 - b3*c5 - b4*c2 + b5*c3 - b7*c1) * haa
  · linear_combination (2*a1*c6 - 2*a2*c5 - 2*a3*c4 + 2*a4*c3 + 2*a5*c2 - 2*a6*c1) * hab + (-2*a1*b6 + 2*a2*b5 + 2*a3*b4 - 2*a4*b3 - 2*a5*b2 + 2*a6*b1) * hac + (-b1*c6 + b2*c5 + b3*c4 - b4*c3 - b5*c2 + b6*c1) * haa

/-- **Phase invariance.**  Under the hypotheses of Theorem 5.1, the same fibre vector
    v = (c, (ac)b) also annihilates J u = (−b, a).  Since J u is the quadrature rotation of u,
    u·e^{iφ} = cos φ·u + sin φ·J u is annihilated for every carrier phase φ. -/
theorem nullspace_J
    (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0)
    (hab : dot a b = 0) (hac : dot a c = 0) (hbc : dot b c = 0) (habc : dot (omul a b) c = 0)
    (haa : dot a a = 1) (hbb : dot b b = 1) :
    smul (Jc (a, b)) (c, partner a b c) = (ozero, ozero) := by
  show smul (oneg b, a) (c, partner a b c) = (ozero, ozero)
  rw [← partner_J a b c ha0 hb0 hc0 hab hac hbc]
  apply nullspace (oneg b) a c
  · show -b.c0 = 0; rw [hb0]; ring
  · exact ha0
  · exact hc0
  · rw [dot_oneg_left, dot_comm, hab]; ring
  · rw [dot_oneg_left, hbc]; ring
  · exact hac
  · rw [dot_omul_J a b c ha0 hb0 hc0]; exact habc
  · rw [dot_oneg_oneg]; exact hbb
  · exact haa

end SedenionRobust
