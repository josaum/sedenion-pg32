/-
  SedenionRank.lean — the kernel of L(u) on the Moreno cone is EXACTLY the constructive
  4-dimensional family of Theorem 5.1, hence rank L(u) = 12.

  Structure of the argument (all identities are polynomial, modulo the cone constraints,
  and are discharged by `linear_combination` with multipliers found by certs2.py):

    (I1)  u·(u·(c,d)) = −2(c,d) + 2(T₁ d, T₂ c)
    (I2)  T₁(T₂ c)   = Π c

  where Π is the orthogonal projector onto W_ab^⊥, T₂ c = (a (Π c)) b and T₁ = T₂ᵀ, written
  out in closed degree-2 form.  (I1) is proved by certificates (certs2.py); (I2) is proved
  structurally in SedenionAlt.lean from alternativity and the adjoint identities.  If u·x = 0 then u·(u·x) = 0, so by (I1) c = T₁ d and d = T₂ c;
  by (I2) c = Π c, i.e. c ∈ W_ab^⊥, and then d = T₂ c = (a c) b.  Together with
  `SedenionReal.nullspace` this is ker L(u) = { (c, (ac)b) : c ∈ W_ab^⊥ }.
-/
import SedenionAlt

namespace SedenionReal

/-- (I1) -/
theorem square_identity (a b c d : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    smul (a, b) (smul (a, b) (c, d)) =
      (oadd (scale (-2) c) (scale 2 (T1 a b d)), oadd (scale (-2) d) (scale 2 (T2 a b c))) := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := c
  obtain ⟨d0, d1, d2, d3, d4, d5, d6, d7⟩ := d
  simp only at ha0 hb0; subst ha0 hb0
  simp only [dot, omul] at hab haa hbb
  simp only [smul, T1, T2, omul, conj, osub, oadd, scale, dot, oe0, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩
  · linear_combination (-2*d0) * hab + (-c0) * haa + (-c0) * hbb
  · linear_combination (-2*d1) * hab + (-c1) * haa + (-c1) * hbb
  · linear_combination (-2*d2) * hab + (-c2) * haa + (-c2) * hbb
  · linear_combination (-2*d3) * hab + (-c3) * haa + (-c3) * hbb
  · linear_combination (-2*d4) * hab + (-c4) * haa + (-c4) * hbb
  · linear_combination (-2*d5) * hab + (-c5) * haa + (-c5) * hbb
  · linear_combination (-2*d6) * hab + (-c6) * haa + (-c6) * hbb
  · linear_combination (-2*d7) * hab + (-c7) * haa + (-c7) * hbb
  · linear_combination (-2*c0) * hab + (-d0) * haa + (-d0) * hbb
  · linear_combination (-2*c1) * hab + (-d1) * haa + (-d1) * hbb
  · linear_combination (-2*c2) * hab + (-d2) * haa + (-d2) * hbb
  · linear_combination (-2*c3) * hab + (-d3) * haa + (-d3) * hbb
  · linear_combination (-2*c4) * hab + (-d4) * haa + (-d4) * hbb
  · linear_combination (-2*c5) * hab + (-d5) * haa + (-d5) * hbb
  · linear_combination (-2*c6) * hab + (-d6) * haa + (-d6) * hbb
  · linear_combination (-2*c7) * hab + (-d7) * haa + (-d7) * hbb

/-- smul (a,b) 0 = 0 -/
theorem smul_zero_right (a b : O) : smul (a, b) (ozero, ozero) = (ozero, ozero) := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  simp only [smul, omul, conj, osub, oadd, ozero, Prod.mk.injEq, O.mk.injEq]
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩ <;> ring

/-- −2x + 2t = 0 ⟹ x = t -/
theorem eq_of_comb (x t : O) (h : oadd (scale (-2) x) (scale 2 t) = ozero) : x = t := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7⟩ := t
  simp only [oadd, scale, ozero, O.mk.injEq] at h ⊢
  obtain ⟨e0, e1, e2, e3, e4, e5, e6, e7⟩ := h
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> linarith

/-- on W_ab^⊥ the correction terms of T₂ vanish: T₂ c = (a c) b -/
theorem T2_eq_partner (a b c : O) (ha0 : a.c0 = 0) (hb0 : b.c0 = 0)
    (hc0 : c.c0 = 0) (h1 : dot a c = 0) (h2 : dot b c = 0) (h3 : dot (omul a b) c = 0) :
    T2 a b c = partner a b c := by
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := c
  simp only at ha0 hb0 hc0; subst ha0 hb0 hc0
  simp only [dot, omul] at h1 h2 h3
  simp only [T2, partner, omul, osub, oadd, scale, dot, oe0, O.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · linear_combination (-1) * h3
  · linear_combination (b1) * h1 + (a1) * h2
  · linear_combination (b2) * h1 + (a2) * h2
  · linear_combination (b3) * h1 + (a3) * h2
  · linear_combination (b4) * h1 + (a4) * h2
  · linear_combination (b5) * h1 + (a5) * h2
  · linear_combination (b6) * h1 + (a6) * h2
  · linear_combination (b7) * h1 + (a7) * h2

/-- **Kernel characterisation.**  On the cone, u·(c,d) = 0 implies c ∈ W_ab^⊥ (Π c = c,
    equivalently c₀ = ⟨a,c⟩ = ⟨b,c⟩ = ⟨ab,c⟩ = 0) and d = (a c) b.  With `nullspace` (the
    converse) this is ker L(u) = partner(W_ab^⊥). -/
theorem kernel_char (a b c d : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1)
    (h : smul (a, b) (c, d) = (ozero, ozero)) :
    (c.c0 = 0 ∧ dot a c = 0 ∧ dot b c = 0 ∧ dot (omul a b) c = 0) ∧ d = partner a b c := by
  have h2 : smul (a, b) (smul (a, b) (c, d)) = (ozero, ozero) := by rw [h, smul_zero_right]
  rw [square_identity a b c d ha0 hb0 hab haa hbb] at h2
  have hc : c = T1 a b d := eq_of_comb _ _ (Prod.mk.inj h2).1
  have hd : d = T2 a b c := eq_of_comb _ _ (Prod.mk.inj h2).2
  have hPi : Pi a b c = c := by
    rw [← adjoint_identity a b c ha0 hb0 hab haa hbb, ← hd, ← hc]
  have orth := Pi_orth a b c ha0 hb0 hab haa hbb
  rw [hPi] at orth
  obtain ⟨o0, o1, o2, o3⟩ := orth
  exact ⟨⟨o0, o1, o2, o3⟩, by rw [hd, T2_eq_partner a b c ha0 hb0 o0 o1 o2 o3]⟩

end SedenionReal
