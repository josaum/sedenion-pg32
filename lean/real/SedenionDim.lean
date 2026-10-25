/-
  SedenionDim.lean — rank L(u) = 12 on the Moreno cone, as a statement about finrank.

  V = Fin 16 → ℝ carries the sedenion coordinates.  `Lu a b : V →ₗ[ℝ] V` is left
  multiplication by u = (a, b).  `phi a b : (Fin 8 → ℝ) →ₗ[ℝ] V` is c ↦ (c, (ac)b) and
  `Wperp a b` is the kernel of c ↦ (c₀, ⟨a,c⟩, ⟨b,c⟩, ⟨ab,c⟩).  Then

      ker (Lu a b) = (Wperp a b).map (phi a b)          (nullspace + kernel_char)
      finrank (Wperp a b) = 4                            (rank–nullity, G surjective)
      phi injective                                      (first block is the identity)

  give  finrank (ker (Lu a b)) = 4  and  finrank (range (Lu a b)) = 12.
-/
import SedenionRank

namespace SedenionReal

open Module

abbrev V := Fin 16 → ℝ
abbrev V8 := Fin 8 → ℝ

def toO8 (w : V8) : O := ⟨w 0, w 1, w 2, w 3, w 4, w 5, w 6, w 7⟩
def ofO8 (x : O) : V8 := fun i =>
  match i.val with
  | 0 => x.c0 | 1 => x.c1 | 2 => x.c2 | 3 => x.c3 | 4 => x.c4 | 5 => x.c5 | 6 => x.c6 | _ => x.c7
def toO (v : V) : O × O :=
  (⟨v 0, v 1, v 2, v 3, v 4, v 5, v 6, v 7⟩, ⟨v 8, v 9, v 10, v 11, v 12, v 13, v 14, v 15⟩)
def ofO (p : O × O) : V := fun i =>
  match i.val with
  | 0 => p.1.c0 | 1 => p.1.c1 | 2 => p.1.c2 | 3 => p.1.c3
  | 4 => p.1.c4 | 5 => p.1.c5 | 6 => p.1.c6 | 7 => p.1.c7
  | 8 => p.2.c0 | 9 => p.2.c1 | 10 => p.2.c2 | 11 => p.2.c3
  | 12 => p.2.c4 | 13 => p.2.c5 | 14 => p.2.c6 | _ => p.2.c7

theorem toO8_ofO8 (x : O) : toO8 (ofO8 x) = x := by
  obtain ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ := x; rfl
theorem ofO8_toO8 (w : V8) : ofO8 (toO8 w) = w := by
  funext i; fin_cases i <;> rfl
theorem toO_ofO (p : O × O) : toO (ofO p) = p := by
  obtain ⟨⟨x0, x1, x2, x3, x4, x5, x6, x7⟩, ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩⟩ := p; rfl
theorem ofO_toO (v : V) : ofO (toO v) = v := by
  funext i; fin_cases i <;> rfl
theorem ofO_zero : ofO (ozero, ozero) = 0 := by
  funext i; fin_cases i <;> rfl
theorem ofO_eq_zero {p : O × O} (h : ofO p = 0) : p = (ozero, ozero) := by
  rw [← toO_ofO p, h]; rfl

set_option maxHeartbeats 0 in
/-- left multiplication by u = (a, b), as a linear map on coordinates -/
def Lu (a b : O) : V →ₗ[ℝ] V where
  toFun v := ofO (smul (a, b) (toO v))
  map_add' v w := by
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    funext i; fin_cases i <;> simp only [ofO, toO, smul, omul, conj, osub, oadd, Pi.add_apply, Pi.smul_apply, smul_eq_mul, RingHom.id_apply] <;> ring
  map_smul' r v := by
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    funext i; fin_cases i <;> simp only [ofO, toO, smul, omul, conj, osub, oadd, Pi.add_apply, Pi.smul_apply, smul_eq_mul, RingHom.id_apply] <;> ring

set_option maxHeartbeats 0 in
/-- c ↦ (c, (a c) b) -/
def phi (a b : O) : V8 →ₗ[ℝ] V where
  toFun w := ofO (toO8 w, partner a b (toO8 w))
  map_add' v w := by
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    funext i; fin_cases i <;> simp only [ofO, toO8, partner, omul, Pi.add_apply, Pi.smul_apply, smul_eq_mul, RingHom.id_apply] <;> ring
  map_smul' r v := by
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    funext i; fin_cases i <;> simp only [ofO, toO8, partner, omul, Pi.add_apply, Pi.smul_apply, smul_eq_mul, RingHom.id_apply] <;> ring

set_option maxHeartbeats 0 in
/-- c ↦ (c₀, ⟨a,c⟩, ⟨b,c⟩, ⟨ab,c⟩) -/
def G (a b : O) : V8 →ₗ[ℝ] (Fin 4 → ℝ) where
  toFun w := fun i =>
    match i.val with
    | 0 => (toO8 w).c0 | 1 => dot a (toO8 w) | 2 => dot b (toO8 w) | _ => dot (omul a b) (toO8 w)
  map_add' v w := by
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    funext i; fin_cases i <;> simp only [toO8, dot, omul, Pi.add_apply, Pi.smul_apply, smul_eq_mul, RingHom.id_apply] <;> ring
  map_smul' r v := by
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
    funext i; fin_cases i <;> simp only [toO8, dot, omul, Pi.add_apply, Pi.smul_apply, smul_eq_mul, RingHom.id_apply] <;> ring

def Wperp (a b : O) : Submodule ℝ V8 := LinearMap.ker (G a b)

theorem mem_Wperp (a b : O) (w : V8) :
    w ∈ Wperp a b ↔ (toO8 w).c0 = 0 ∧ dot a (toO8 w) = 0 ∧ dot b (toO8 w) = 0 ∧ dot (omul a b) (toO8 w) = 0 := by
  simp only [Wperp, LinearMap.mem_ker, G, LinearMap.coe_mk, AddHom.coe_mk]
  constructor
  · intro h
    exact ⟨congrFun h 0, congrFun h 1, congrFun h 2, congrFun h 3⟩
  · rintro ⟨h0, h1, h2, h3⟩
    funext i; fin_cases i <;> simp only [h0, h1, h2, h3, Pi.zero_apply]

/-- ker L(u) = phi(W_ab^⊥) -/
theorem ker_Lu_eq (a b : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    LinearMap.ker (Lu a b) = (Wperp a b).map (phi a b) := by
  ext v
  simp only [LinearMap.mem_ker, Submodule.mem_map]
  constructor
  · intro h
    have h' : smul (a, b) (toO v) = (ozero, ozero) := ofO_eq_zero h
    obtain ⟨⟨c0, c1, c2, c3⟩, hd⟩ :=
      kernel_char a b (toO v).1 (toO v).2 ha0 hb0 hab haa hbb h'
    refine ⟨ofO8 (toO v).1, ?_, ?_⟩
    · rw [mem_Wperp, toO8_ofO8]; exact ⟨c0, c1, c2, c3⟩
    · show ofO (toO8 (ofO8 (toO v).1), partner a b (toO8 (ofO8 (toO v).1))) = v
      rw [toO8_ofO8, ← hd]
      exact ofO_toO v
  · rintro ⟨w, hw, rfl⟩
    rw [mem_Wperp] at hw
    obtain ⟨c0, c1, c2, c3⟩ := hw
    show ofO (smul (a, b) (toO (ofO (toO8 w, partner a b (toO8 w))))) = 0
    rw [toO_ofO, nullspace a b (toO8 w) ha0 hb0 c0 hab c1 c2 c3 haa hbb, ofO_zero]

/-- phi is injective: its first eight coordinates are the identity -/
theorem phi_injective (a b : O) : Function.Injective (phi a b) := by
  intro v w h
  have hc : toO8 v = toO8 w := by
    have h1 := congrArg toO h
    simp only [phi, LinearMap.coe_mk, AddHom.coe_mk, toO_ofO, Prod.mk.injEq] at h1
    exact h1.1
  rw [← ofO8_toO8 v, ← ofO8_toO8 w, hc]

/-- G is surjective on the cone: t ↦ t₀ e₀ + t₁ a + t₂ b + t₃ ab is a right inverse -/
theorem G_surjective (a b : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    Function.Surjective (G a b) := by
  intro t
  refine ⟨ofO8 (oadd (oadd (oadd (scale (t 0) oe0) (scale (t 1) a)) (scale (t 2) b)) (scale (t 3) (omul a b))), ?_⟩
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := a
  obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ := b
  simp only at ha0 hb0; subst ha0 hb0
  simp only [dot, omul] at hab haa hbb
  simp only [G, LinearMap.coe_mk, AddHom.coe_mk, toO8_ofO8]
  funext i; fin_cases i <;> simp only [Fin.isValue, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk]
  · simp only [dot, omul, oadd, scale, oe0]; linear_combination (-(t 3)) * hab
  · simp only [dot, omul, oadd, scale, oe0]; linear_combination (t 1) * haa + (t 2) * hab
  · simp only [dot, omul, oadd, scale, oe0]; linear_combination (t 1) * hab + (t 2) * hbb
  · simp only [dot, omul, oadd, scale, oe0]
    linear_combination (-(t 0)) * hab + (t 3 * (b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 + b7^2)) * haa + (t 3) * hbb

theorem finrank_Wperp (a b : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    finrank ℝ (Wperp a b) = 4 := by
  have h := LinearMap.finrank_range_add_finrank_ker (G a b)
  rw [LinearMap.range_eq_top.mpr (G_surjective a b ha0 hb0 hab haa hbb), finrank_top,
      Module.finrank_fin_fun, Module.finrank_fin_fun] at h
  unfold Wperp; omega

/-- **Theorem (rank of L(u)).**  On the Moreno cone, dim ker L(u) = 4 and rank L(u) = 12. -/
theorem finrank_ker_Lu (a b : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    finrank ℝ (LinearMap.ker (Lu a b)) = 4 := by
  rw [ker_Lu_eq a b ha0 hb0 hab haa hbb, ← LinearMap.range_domRestrict,
      LinearMap.finrank_range_of_inj, finrank_Wperp a b ha0 hb0 hab haa hbb]
  intro x y hxy
  exact Subtype.ext (phi_injective a b hxy)

theorem finrank_range_Lu (a b : O)
    (ha0 : a.c0 = 0) (hb0 : b.c0 = 0) (hab : dot a b = 0) (haa : dot a a = 1) (hbb : dot b b = 1) :
    finrank ℝ (LinearMap.range (Lu a b)) = 12 := by
  have h := LinearMap.finrank_range_add_finrank_ker (Lu a b)
  rw [finrank_ker_Lu a b ha0 hb0 hab haa hbb, Module.finrank_fin_fun] at h
  omega

end SedenionReal
