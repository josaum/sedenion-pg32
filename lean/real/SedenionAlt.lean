/-
  SedenionAlt.lean — structural octonion identities used for the kernel characterisation.

  R x  := (a x) b            Rt y := a (y b)   (its adjoint)
  Π c  := projector onto span(e₀, a, b, ab)^⊥

  Every lemma is a polynomial identity closed by `ring` (unconditional ones) or by
  `linear_combination` with tiny multipliers (ones that need the cone).  No certificate
  search is involved; this file replaces a single 5.9 GB `ring` call by ~20 small ones.
-/
import SedenionReal

namespace SedenionReal

def oe0 : O := ⟨1, 0, 0, 0, 0, 0, 0, 0⟩
def R  (a b x : O) : O := omul (omul a x) b
def Rt (a b y : O) : O := omul a (omul y b)

def Pi (a b c : O) : O :=
  osub (osub (osub (osub c (scale c.c0 oe0)) (scale (dot a c) a)) (scale (dot b c) b)) (scale (dot (omul a b) c) (omul a b))

def T2 (a b c : O) : O :=
  oadd (oadd (oadd (osub (omul (omul a c) b) (scale c.c0 (omul a b))) (scale (dot a c) b)) (scale (dot b c) a)) (scale (-(dot (omul a b) c)) oe0)

def T1 (a b d : O) : O :=
  oadd (oadd (oadd (osub (omul a (omul d b)) (scale (dot (omul a b) d) oe0)) (scale (dot b d) a)) (scale (dot a d) b)) (scale (-d.c0) (omul a b))

/-! ### small unconditional algebra (depth ≤ 2 products, all by `ring`) -/

theorem omul_e0_left (x : O) : omul oe0 x = x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [omul, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

theorem omul_e0_right (x : O) : omul x oe0 = x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [omul, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

theorem scale_scale (r s : ℝ) (x : O) : scale r (scale s x) = scale (r * s) x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [scale, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

theorem scale_one (x : O) : scale 1 x = x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [scale, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

theorem dot_scale_left (r : ℝ) (x y : O) : dot (scale r x) y = r * dot x y := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [dot, scale]; ring

/-- b² = −‖b‖² for imaginary b -/
theorem sq_imag (b : O) (hb0 : b.c0 = 0) : omul b b = scale (-(dot b b)) oe0 := by
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  simp only at hb0; subst hb0
  simp only [omul, scale, dot, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

theorem conj_imag (a : O) (ha0 : a.c0 = 0) : conj a = scale (-1) a := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  simp only at ha0; subst ha0
  simp only [conj, scale, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- right alternativity  (y b) b = y (b b) -/
theorem right_alt (y b : O) : omul (omul y b) b = omul y (omul b b) := by
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  simp only [omul, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- left alternativity  a (a y) = (a a) y -/
theorem left_alt (a y : O) : omul a (omul a y) = omul (omul a a) y := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [omul, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- ⟨y, a z⟩ = ⟨a* y, z⟩  (adjoint of left multiplication, Thm 2.2 for octonions) -/
theorem dot_omul_left (a y z : O) : dot y (omul a z) = dot (omul (conj a) y) z := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  obtain ⟨z0, z1, z2, z3, z4, z5, z6, z7⟩ := z
  simp only [omul, conj, dot]; ring

/-- ⟨w, z b⟩ = ⟨w b*, z⟩ -/
theorem dot_omul_right (b w z : O) : dot w (omul z b) = dot (omul w (conj b)) z := by
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨w0, w1, w2, w3, w4, w5, w6, w7⟩ := w
  obtain ⟨z0, z1, z2, z3, z4, z5, z6, z7⟩ := z
  simp only [omul, conj, dot]; ring

/-! ### R = (a ·) b and its adjoint, by rewriting only -/

/-- RᵀR = ‖a‖²‖b‖² on imaginary a, b -/
theorem Rt_R (a b x : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) :
    Rt a b (R a b x) = scale (dot a a * dot b b) x := by
  unfold Rt R
  rw [right_alt, sq_imag b hb0, omul_scale_right, omul_e0_right, omul_scale_right, left_alt,
      sq_imag a ha0, omul_scale_left, omul_e0_left, scale_scale]
  congr 1; ring

/-- ⟨y, Rᵀ d⟩ = ⟨R y, d⟩ on imaginary a, b -/
theorem dot_Rt (a b y d : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) :
    dot y (Rt a b d) = dot (R a b y) d := by
  unfold Rt R
  rw [dot_omul_left, dot_omul_right, conj_imag a ha0, conj_imag b hb0, omul_scale_left,
      omul_scale_right, omul_scale_left, scale_scale, dot_scale_left]
  ring

/-- R is linear -/
theorem R_sub (a b x y : O) : R a b (osub x y) = osub (R a b x) (R a b y) := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [R, omul, osub, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

theorem R_scale (a b : O) (r : ℝ) (x : O) : R a b (scale r x) = scale r (R a b x) := by
  unfold R; rw [omul_scale_right, omul_scale_left]

/-- R on the frame e₀, a, b, ab -/
theorem R_e0 (a b : O) : R a b oe0 = omul a b := by
  unfold R; rw [omul_e0_right]
theorem R_a (a b : O) (ha0 : a.c0 = 0) : R a b a = scale (-(dot a a)) b := by
  unfold R; rw [sq_imag a ha0, omul_scale_left, omul_e0_left]
theorem R_b (a b : O) (hb0 : b.c0 = 0) : R a b b = scale (-(dot b b)) a := by
  unfold R; rw [right_alt, sq_imag b hb0, omul_scale_right, omul_e0_right]
theorem R_ab (a b : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) :
    R a b (omul a b) = scale (dot a a * dot b b) oe0 := by
  unfold R
  rw [left_alt, sq_imag a ha0, omul_scale_left, omul_e0_left, omul_scale_left, sq_imag b hb0,
      scale_scale]
  congr 1; ring

/-- T₂ = R ∘ Π on the cone -/
theorem T2_eq_R_Pi (a b c : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    T2 a b c = R a b (Pi a b c) := by
  unfold Pi
  rw [R_sub, R_sub, R_sub, R_sub, R_scale, R_scale, R_scale, R_scale,
      R_e0, R_a a b ha0, R_b a b hb0, R_ab a b ha0 hb0, haa, hbb]
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := c
  simp only [T2, R, omul, osub, oadd, scale, dot, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- T₁ = Π ∘ Rᵀ on the cone -/
theorem T1_eq_Pi_Rt (a b d : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    T1 a b d = Pi a b (Rt a b d) := by
  have h0 : (Rt a b d).c0 = dot (omul a b) d := by
    have := dot_Rt a b oe0 d ha0 hb0
    rw [R_e0] at this
    rw [← this]
    obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7⟩ := Rt a b d
    simp only [dot, oe0]; ring
  have ha : dot a (Rt a b d) = -(dot b d) := by
    rw [dot_Rt a b a d ha0 hb0, R_a a b ha0, haa, dot_scale_left]; ring
  have hb : dot b (Rt a b d) = -(dot a d) := by
    rw [dot_Rt a b b d ha0 hb0, R_b a b hb0, hbb, dot_scale_left]; ring
  have hab : dot (omul a b) (Rt a b d) = d.c0 := by
    rw [dot_Rt a b _ d ha0 hb0, R_ab a b ha0 hb0, haa, hbb, dot_scale_left]
    obtain ⟨d0, d1, d2, d3, d4, d5, d6, d7⟩ := d
    simp only [dot, oe0]; ring
  unfold Pi
  rw [h0, ha, hb, hab]
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨d0, d1, d2, d3, d4, d5, d6, d7⟩ := d
  simp only [T1, Rt, omul, osub, oadd, scale, dot, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-! ### dot-product bookkeeping (depth ≤ 1) -/
theorem dot_comm (x y : O) : dot x y = dot y x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [dot]; ring
theorem dot_osub_right (x y z : O) : dot x (osub y z) = dot x y - dot x z := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  obtain ⟨z0, z1, z2, z3, z4, z5, z6, z7⟩ := z
  simp only [dot, osub]; ring
theorem dot_scale_right (r : ℝ) (x y : O) : dot x (scale r y) = r * dot x y := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [dot, scale]; ring
theorem dot_e0_right (x : O) : dot x oe0 = x.c0 := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [dot, oe0]; ring
theorem c0_osub (y z : O) : (osub y z).c0 = y.c0 - z.c0 := rfl
theorem c0_scale (r : ℝ) (y : O) : (scale r y).c0 = r * y.c0 := rfl
theorem c0_e0 : oe0.c0 = 1 := rfl
/-- (ab)₀ = −⟨a,b⟩ for imaginary a, b -/
theorem c0_omul (a b : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) : (omul a b).c0 = -(dot a b) := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  simp only at ha0 hb0; subst ha0 hb0
  simp only [omul, dot]; ring
/-- ⟨a, ab⟩ = 0 for imaginary b -/
theorem dot_a_ab (a b : O) (hb0 : b.c0 = 0) : dot a (omul a b) = 0 := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  simp only at hb0; subst hb0
  simp only [omul, dot]; ring
/-- ⟨b, ab⟩ = 0 for imaginary a -/
theorem dot_b_ab (a b : O) (ha0 : a.c0 = 0) : dot b (omul a b) = 0 := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  simp only at ha0; subst ha0
  simp only [omul, dot]; ring

/-- the four coordinates of Π c along e₀, a, b, ab vanish on the cone -/
theorem Pi_orth (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    (Pi a b c).c0 = 0 ∧ dot a (Pi a b c) = 0 ∧ dot b (Pi a b c) = 0 ∧ dot (omul a b) (Pi a b c) = 0 := by
  have hba : dot b a = 0 := by rw [dot_comm]; exact hab
  have haab : dot a (omul a b) = 0 := dot_a_ab a b hb0
  have hbab : dot b (omul a b) = 0 := dot_b_ab a b ha0
  have hab0 : (omul a b).c0 = -(dot a b) := c0_omul a b ha0 hb0
  have habab : dot (omul a b) (omul a b) = 1 := by rw [composition, haa, hbb]; ring
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [Pi, c0_osub, c0_scale, c0_e0, ha0, hb0, hab0, hab]; ring
  · simp only [Pi, dot_osub_right, dot_scale_right, dot_e0_right, ha0, haa, hab, haab]; ring
  · simp only [Pi, dot_osub_right, dot_scale_right, dot_e0_right, hb0, hbb, hba, hbab]; ring
  · simp only [Pi, dot_osub_right, dot_scale_right, dot_e0_right, hab0, hab, habab,
               dot_comm (omul a b) a, dot_comm (omul a b) b, haab, hbab]; ring

/-- Π is linear (keeping ⟨a,x⟩, (ab)ₖ as atoms so that `ring` stays small) -/
theorem Pi_sub (a b x y : O) : Pi a b (osub x y) = osub (Pi a b x) (Pi a b y) := by
  simp only [Pi, dot_osub_right, c0_osub]
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := y
  simp only [osub, scale, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring
theorem Pi_scale (a b : O) (r : ℝ) (x : O) : Pi a b (scale r x) = scale r (Pi a b x) := by
  simp only [Pi, dot_scale_right, c0_scale]
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  simp only [osub, scale, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- Π kills the frame on the cone -/
theorem Pi_frame (a b : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    Pi a b oe0 = ozero ∧ Pi a b a = ozero ∧ Pi a b b = ozero ∧ Pi a b (omul a b) = ozero := by
  have hba : dot b a = 0 := by rw [dot_comm]; exact hab
  have haab : dot a (omul a b) = 0 := dot_a_ab a b hb0
  have hbab : dot b (omul a b) = 0 := dot_b_ab a b ha0
  have hab0 : (omul a b).c0 = -(dot a b) := c0_omul a b ha0 hb0
  have habab : dot (omul a b) (omul a b) = 1 := by rw [composition, haa, hbb]; ring
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [Pi, c0_e0, dot_e0_right, ha0, hb0, hab0, hab]
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    simp only [omul, osub, scale, oe0, ozero, O.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring
  · simp only [Pi, ha0, haa, hba, dot_comm (omul a b) a, haab]
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    simp only [omul, osub, scale, oe0, ozero, O.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring
  · simp only [Pi, hb0, hbb, hab, dot_comm (omul a b) b, hbab]
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    simp only [omul, osub, scale, oe0, ozero, O.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring
  · simp only [Pi, hab0, hab, haab, hbab, habab]
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    simp only [omul, osub, scale, oe0, ozero, O.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- Π is idempotent on the cone -/
theorem Pi_Pi (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    Pi a b (Pi a b c) = Pi a b c := by
  obtain ⟨f0, fa, fb, fab⟩ := Pi_frame a b ha0 hb0 hab haa hbb
  have e : Pi a b (Pi a b c) = Pi a b (osub (osub (osub (osub c (scale c.c0 oe0)) (scale (dot a c) a)) (scale (dot b c) b)) (scale (dot (omul a b) c) (omul a b))) := rfl
  rw [e, Pi_sub, Pi_sub, Pi_sub, Pi_sub, Pi_scale, Pi_scale, Pi_scale, Pi_scale, f0, fa, fb, fab]
  obtain ⟨p0, p1, p2, p3, p4, p5, p6, p7⟩ := Pi a b c
  simp only [osub, scale, ozero, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> ring

/-- (I2)  T₁(T₂ c) = Π c -/
theorem adjoint_identity (a b c : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    T1 a b (T2 a b c) = Pi a b c := by
  rw [T1_eq_Pi_Rt a b _ ha0 hb0 haa hbb, T2_eq_R_Pi a b c ha0 hb0 haa hbb,
      Rt_R a b _ ha0 hb0, haa, hbb, one_mul, scale_one, Pi_Pi a b c ha0 hb0 hab haa hbb]

end SedenionReal
