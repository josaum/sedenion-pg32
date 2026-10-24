/-
  SedenionReal.lean — Theorem 5.1 (Constructive 4D Moreno Nullspace) over ℝ.

  Octonions are 8 real coordinates with the product given by the same Cayley–Dickson
  structure constants as `SedenionPG32.Core.eps` (top-left 8×8 block); sedenions are pairs
  (a, b) ∈ O × O with the doubling product  (a,b)(c,d) = (ac − d* b, da + b c*).

  The theorem is an ideal-membership statement: every coordinate of u·v is a polynomial
  combination of the six constraint polynomials.  The multipliers were found by linear
  algebra (certs.py) and are verified here by `linear_combination`, i.e. by `ring`.
  No `sorry`, no `native_decide`.
-/
import Mathlib.Tactic

namespace SedenionReal

@[ext] structure O where
  (c0 c1 c2 c3 c4 c5 c6 c7 : ℝ)

/-- conjugation: negate the imaginary part -/
def conj (x : O) : O := ⟨x.c0, -x.c1, -x.c2, -x.c3, -x.c4, -x.c5, -x.c6, -x.c7⟩

/-- octonion product, e_i e_j = ε_ij e_(i xor j) with the Cayley–Dickson signs -/
def omul (x y : O) : O := ⟨
  x.c0 * y.c0 - x.c1 * y.c1 - x.c2 * y.c2 - x.c3 * y.c3 - x.c4 * y.c4 - x.c5 * y.c5 - x.c6 * y.c6 - x.c7 * y.c7,
  x.c0 * y.c1 + x.c1 * y.c0 + x.c2 * y.c3 - x.c3 * y.c2 + x.c4 * y.c5 - x.c5 * y.c4 - x.c6 * y.c7 + x.c7 * y.c6,
  x.c0 * y.c2 - x.c1 * y.c3 + x.c2 * y.c0 + x.c3 * y.c1 + x.c4 * y.c6 + x.c5 * y.c7 - x.c6 * y.c4 - x.c7 * y.c5,
  x.c0 * y.c3 + x.c1 * y.c2 - x.c2 * y.c1 + x.c3 * y.c0 + x.c4 * y.c7 - x.c5 * y.c6 + x.c6 * y.c5 - x.c7 * y.c4,
  x.c0 * y.c4 - x.c1 * y.c5 - x.c2 * y.c6 - x.c3 * y.c7 + x.c4 * y.c0 + x.c5 * y.c1 + x.c6 * y.c2 + x.c7 * y.c3,
  x.c0 * y.c5 + x.c1 * y.c4 - x.c2 * y.c7 + x.c3 * y.c6 - x.c4 * y.c1 + x.c5 * y.c0 - x.c6 * y.c3 + x.c7 * y.c2,
  x.c0 * y.c6 + x.c1 * y.c7 + x.c2 * y.c4 - x.c3 * y.c5 - x.c4 * y.c2 + x.c5 * y.c3 + x.c6 * y.c0 - x.c7 * y.c1,
  x.c0 * y.c7 - x.c1 * y.c6 + x.c2 * y.c5 + x.c3 * y.c4 - x.c4 * y.c3 - x.c5 * y.c2 + x.c6 * y.c1 + x.c7 * y.c0⟩

/-- Euclidean inner product on O -/
def dot (x y : O) : ℝ :=
  x.c0 * y.c0 + x.c1 * y.c1 + x.c2 * y.c2 + x.c3 * y.c3 + x.c4 * y.c4 + x.c5 * y.c5 + x.c6 * y.c6 + x.c7 * y.c7

def osub (x y : O) : O := ⟨x.c0 - y.c0, x.c1 - y.c1, x.c2 - y.c2, x.c3 - y.c3, x.c4 - y.c4, x.c5 - y.c5, x.c6 - y.c6, x.c7 - y.c7⟩
def oadd (x y : O) : O := ⟨x.c0 + y.c0, x.c1 + y.c1, x.c2 + y.c2, x.c3 + y.c3, x.c4 + y.c4, x.c5 + y.c5, x.c6 + y.c6, x.c7 + y.c7⟩
def ozero : O := ⟨0, 0, 0, 0, 0, 0, 0, 0⟩

/-- sedenion = pair of octonions; Cayley–Dickson doubling product (eq. cd_doubling) -/
def smul (u v : O × O) : O × O :=
  (osub (omul u.1 v.1) (omul (conj v.2) u.2), oadd (omul v.2 u.1) (omul u.2 (conj v.1)))

/-- the constructive partner d = (a c) b of Theorem 5.1 -/
def partner (a b c : O) : O := omul (omul a c) b

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



/-- Octonion composition law ‖xy‖² = ‖x‖²‖y‖² (Hurwitz), by `ring` on the explicit formulas. -/
theorem composition (x y : O) : dot (omul x y) (omul x y) = dot x x * dot y y := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [dot, omul]; ring

/-- ‖(a c) b‖² = ‖c‖² when ‖a‖ = ‖b‖ = 1: the partner has the same norm as its seed, so
    ‖v‖² = ‖c‖² + ‖d‖² = 2‖c‖² and v ≠ 0 whenever c ≠ 0. -/
theorem partner_norm (a b c : O) (haa : dot a a = 1) (hbb : dot b b = 1) :
    dot (partner a b c) (partner a b c) = dot c c := by
  unfold partner; rw [composition, composition, haa, hbb]; ring

/-- the partner map c ↦ (a c) b is ℝ-linear, so the kernel contains the image of the 4-dim
    space W_ab^⊥ under an injective (norm-preserving) linear map: dim ker L(u) ≥ 4. -/
theorem partner_linear (a b c c' : O) (s t : ℝ) :
    partner a b ⟨s * c.c0 + t * c'.c0, s * c.c1 + t * c'.c1, s * c.c2 + t * c'.c2, s * c.c3 + t * c'.c3,
                 s * c.c4 + t * c'.c4, s * c.c5 + t * c'.c5, s * c.c6 + t * c'.c6, s * c.c7 + t * c'.c7⟩
      = ⟨s * (partner a b c).c0 + t * (partner a b c').c0, s * (partner a b c).c1 + t * (partner a b c').c1,
         s * (partner a b c).c2 + t * (partner a b c').c2, s * (partner a b c).c3 + t * (partner a b c').c3,
         s * (partner a b c).c4 + t * (partner a b c').c4, s * (partner a b c).c5 + t * (partner a b c').c5,
         s * (partner a b c).c6 + t * (partner a b c').c6, s * (partner a b c).c7 + t * (partner a b c').c7⟩ := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := c
  obtain ⟨d0, d1, d2, d3, d4, d5, d6, d7⟩ := c'
  simp only [partner, omul, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring


/-! ## Consistency with the integer development

`sign` below is a verbatim copy of `SedenionPG32.Core.sign` (the Cayley–Dickson recursion).
`basis_mul_sign` shows the explicit real formulas of `omul` are that recursion on basis units,
so the real theorems above are about the same algebra as the kernel-checked integer theorems. -/

def conjS (j : Nat) : Int := if j = 0 then 1 else -1

def sign : Nat → Nat → Nat → Int
  | 0,     _, _ => 1
  | k + 1, i, j =>
    let n := 2 ^ k
    if i < n then
      (if j < n then sign k i j else sign k (j - n) i)
    else
      (if j < n then conjS j * sign k (i - n) j
                else -(conjS (j - n)) * sign k (j - n) (i - n))

/-- literal copy of the 8×8 block of ε, certified against the recursion by `decide` -/
def signTab (i j : Nat) : Int :=
  match i with
  | 0 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => 1
    | 3 => 1
    | 4 => 1
    | 5 => 1
    | 6 => 1
    | 7 => 1
    | _ => 0
  | 1 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => 1
    | 3 => -1
    | 4 => 1
    | 5 => -1
    | 6 => -1
    | 7 => 1
    | _ => 0
  | 2 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => -1
    | 3 => 1
    | 4 => 1
    | 5 => 1
    | 6 => -1
    | 7 => -1
    | _ => 0
  | 3 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => -1
    | 3 => -1
    | 4 => 1
    | 5 => -1
    | 6 => 1
    | 7 => -1
    | _ => 0
  | 4 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => -1
    | 3 => -1
    | 4 => -1
    | 5 => 1
    | 6 => 1
    | 7 => 1
    | _ => 0
  | 5 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => -1
    | 3 => 1
    | 4 => -1
    | 5 => -1
    | 6 => -1
    | 7 => 1
    | _ => 0
  | 6 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => 1
    | 3 => -1
    | 4 => -1
    | 5 => 1
    | 6 => -1
    | 7 => -1
    | _ => 0
  | 7 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => 1
    | 3 => 1
    | 4 => -1
    | 5 => -1
    | 6 => 1
    | 7 => -1
    | _ => 0
  | _ => 0

theorem signTab_eq : ∀ i, i < 8 → ∀ j, j < 8 → signTab i j = sign 3 i j := by decide

def e (i : Nat) : O :=
  ⟨if i = 0 then 1 else 0, if i = 1 then 1 else 0, if i = 2 then 1 else 0, if i = 3 then 1 else 0,
   if i = 4 then 1 else 0, if i = 5 then 1 else 0, if i = 6 then 1 else 0, if i = 7 then 1 else 0⟩

def scale (r : ℝ) (x : O) : O := ⟨r * x.c0, r * x.c1, r * x.c2, r * x.c3, r * x.c4, r * x.c5, r * x.c6, r * x.c7⟩

/-- Euclidean norm squared of a sedenion pair (x, y) ∈ O × O -/
def snormSq (u : O × O) : ℝ := dot u.1 u.1 + dot u.2 u.2

/-- Lemma: scaling an octonion scales its dot product by r² -/
theorem dot_scale (r : ℝ) (x : O) : dot (scale r x) (scale r x) = r ^ 2 * dot x x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [dot, scale]; ring

@[simp] theorem omul_scale_left (r : ℝ) (x y : O) : omul (scale r x) y = scale r (omul x y) := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [omul, scale, O.mk.injEq]
  refine ⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩

@[simp] theorem omul_scale_right (r : ℝ) (x y : O) : omul x (scale r y) = scale r (omul x y) := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [omul, scale, O.mk.injEq]
  refine ⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩

@[simp] theorem conj_scale (r : ℝ) (x : O) : conj (scale r x) = scale r (conj x) := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [conj, scale, O.mk.injEq]
  refine ⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩

@[simp] theorem omul_add_left (x y z : O) : omul (oadd x y) z = oadd (omul x z) (omul y z) := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  obtain ⟨z0, z1, z2, z3, z4, z5, z6, z7⟩ := z
  simp only [omul, oadd, O.mk.injEq]
  refine ⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩

@[simp] theorem omul_add_right (x y z : O) : omul x (oadd y z) = oadd (omul x y) (omul x z) := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  obtain ⟨z0, z1, z2, z3, z4, z5, z6, z7⟩ := z
  simp only [omul, oadd, O.mk.injEq]
  refine ⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩

/-- **Theorem 2.1 Part 2 (GNSS CRPA Anti-Partner Coherent Array Gain — Product Formula).**
    While the jammer u_jam = (a, b) is annihilated identically by v_beam = (c, (ac)b),
    the orthogonal anti-partner satellite manifold u_sat = (a, -b) experiences constructive
    reinforcement: u_sat · v_beam = (2 a c, 2 ((ac)b) a). -/
theorem crpa_antipartner_product
    (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0)
    (hab : dot a b = 0) (hac : dot a c = 0) (hbc : dot b c = 0) (habc : dot (omul a b) c = 0)
    (haa : dot a a = 1) (hbb : dot b b = 1) :
    smul (a, scale (-1) b) (c, partner a b c) =
      (scale 2 (omul a c), scale 2 (omul (partner a b c) a)) := by
  have h := nullspace a b c ha0 hb0 hc0 hab hac hbc habc haa hbb
  simp only [smul, omul_scale_left, omul_scale_right, conj_scale]
  simp only [smul, osub, oadd, ozero, scale, Prod.mk.injEq, O.mk.injEq] at h ⊢
  obtain ⟨⟨h10, h11, h12, h13, h14, h15, h16, h17⟩, ⟨h20, h21, h22, h23, h24, h25, h26, h27⟩⟩ := h
  refine ⟨⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩,
          ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩⟩

/-- **Theorem 2.1 Part 2 (GNSS CRPA Coherent +3 dB Array Gain Norm Identity).**
    ‖(a, -b) · (c, (ac)b)‖² = 8 ‖c‖² = 2 ‖(a, -b)‖² ‖(c, (ac)b)‖². -/
theorem crpa_antipartner_gain_norm
    (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0)
    (hab : dot a b = 0) (hac : dot a c = 0) (hbc : dot b c = 0) (habc : dot (omul a b) c = 0)
    (haa : dot a a = 1) (hbb : dot b b = 1) :
    snormSq (smul (a, scale (-1) b) (c, partner a b c)) = 8 * dot c c := by
  rw [crpa_antipartner_product a b c ha0 hb0 hc0 hab hac hbc habc haa hbb]
  unfold snormSq
  simp only [dot_scale, composition, partner_norm a b c haa hbb, haa]
  ring

/-- **Section 4 (Half-Sphere Key Modulation Passband Identity at φ = 0).**
    At φ = 0, k(0) = (c, -(ac)b) achieves full constructive passband gain ‖u · k(0)‖² = 8 ‖c‖²,
    whereas at φ = π, k(π) = (c, (ac)b) achieves exact zero-divisor spoofer excision u · k(π) = 0. -/
theorem spoofer_half_sphere_passband
    (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hc0 : c.c0 = 0)
    (hab : dot a b = 0) (hac : dot a c = 0) (hbc : dot b c = 0) (habc : dot (omul a b) c = 0)
    (haa : dot a a = 1) (hbb : dot b b = 1) :
    snormSq (smul (a, b) (c, scale (-1) (partner a b c))) = 8 * dot c c := by
  have h := nullspace a b c ha0 hb0 hc0 hab hac hbc habc haa hbb
  have hprod : smul (a, b) (c, scale (-1) (partner a b c)) =
      (scale 2 (omul a c), scale (-2) (omul (partner a b c) a)) := by
    simp only [smul, omul_scale_left, omul_scale_right, conj_scale]
    simp only [smul, osub, oadd, ozero, scale, Prod.mk.injEq, O.mk.injEq] at h ⊢
    obtain ⟨⟨h10, h11, h12, h13, h14, h15, h16, h17⟩, ⟨h20, h21, h22, h23, h24, h25, h26, h27⟩⟩ := h
    refine ⟨⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩,
            ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩⟩
  rw [hprod]
  unfold snormSq
  simp only [dot_scale, composition, partner_norm a b c haa hbb, haa]
  ring

set_option maxHeartbeats 1000000 in
theorem basis_mul_sign (i j : Nat) (hi : i < 8) (hj : j < 8) :
    omul (e i) (e j) = scale ((sign 3 i j : Int) : ℝ) (e (i ^^^ j)) := by
  rw [← signTab_eq i hi j hj]
  interval_cases i <;> interval_cases j <;> simp [omul, e, scale, signTab]

/-- Sedenion addition and real scalar multiplication on O × O -/
def sadd (u v : O × O) : O × O := (oadd u.1 v.1, oadd u.2 v.2)
def sscale (r : ℝ) (u : O × O) : O × O := (scale r u.1, scale r u.2)

/-- Left-linearity of sedenion multiplication:
    (α u₁ + β u₂) · w = α (u₁ · w) + β (u₂ · w). -/
theorem smul_linear_left (u1 u2 w : O × O) (α β : ℝ) :
    smul (sadd (sscale α u1) (sscale β u2)) w =
      sadd (sscale α (smul u1 w)) (sscale β (smul u2 w)) := by
  obtain ⟨a1, b1⟩ := u1
  obtain ⟨a2, b2⟩ := u2
  obtain ⟨c, d⟩ := w
  simp only [smul, sadd, sscale, omul_add_left, omul_add_right, omul_scale_left, omul_scale_right]
  simp only [oadd, osub, scale, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩,
          ⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩⟩

/-- Right-distributivity and scalar bilinearity of sedenion multiplication:
    if u₁ · w = 0 and u₂ · w = 0, then for any real amplitudes α, β ∈ ℝ,
    the simultaneous multi-jammer superposition (α u₁ + β u₂) · w = 0 identically. -/
theorem crpa_multi_jammer_superposition_null
    (u1 u2 w : O × O) (α β : ℝ)
    (h1 : smul u1 w = (ozero, ozero))
    (h2 : smul u2 w = (ozero, ozero)) :
    smul (sadd (sscale α u1) (sscale β u2)) w = (ozero, ozero) := by
  rw [smul_linear_left, h1, h2]
  simp only [sadd, sscale, oadd, scale, ozero, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩,
          ⟨by ring, by ring, by ring, by ring, by ring, by ring, by ring, by ring⟩⟩

end SedenionReal
