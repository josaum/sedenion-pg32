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

end SedenionRobust
