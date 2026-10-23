/-
  SedenionPG32.lean — machine-checked finite statements from
  "Sedenion PG(3,2) Neural Topologies" (Agourakis, 2026).

  No Mathlib. Every theorem is closed by `decide` (kernel evaluation) or `rfl`;
  there is no `sorry`, no `native_decide`, and no axiom beyond Lean's core.

  Numbering refers to the manuscript.  Everything below is about the
  *integer* structure constants ε_{ij} and ℤ-coefficient sedenions, i.e. the
  combinatorial content of Sections 2.1, 2.2, 3 and the explicit zero-divisor
  witnesses of 3.2 / 5.1.  Statements that genuinely need ℝ (rank of L(u) on the
  whole Moreno cone, Hurwitz) are not claimed here.
-/

namespace SedenionPG32


/-! ## 1. Cayley–Dickson structure constants (Sec. 2.1, eq. cd_doubling) -/

/-- conjugation sign of basis unit `j` of the lower algebra -/
def conj (j : Nat) : Int := if j = 0 then 1 else -1

/-- `sign k i j` = ε_{ij} in the 2^k-dimensional Cayley–Dickson algebra, defined by the
    same recursion the reference implementation uses:
      (a,0)(c,0) = (ac,0)      (a,0)(0,d) = (0,da)
      (0,b)(c,0) = (0,b c*)    (0,b)(0,d) = (−d* b, 0)                      -/
def sign : Nat → Nat → Nat → Int
  | 0,     _, _ => 1
  | k + 1, i, j =>
    let n := 2 ^ k
    if i < n then
      (if j < n then sign k i j else sign k (j - n) i)
    else
      (if j < n then conj j * sign k (i - n) j
                else -(conj (j - n)) * sign k (j - n) (i - n))

/-- ε_{ij} for the sedenions (k = 4, dimension 16). -/
def eps (i j : Nat) : Int := sign 4 i j

/-- basis product: e_i · e_j = ε_{ij} e_{i ⊕ j}  (eq. basis_xor) -/
theorem eps_unit_left  : ∀ j, j < 16 → eps 0 j = 1 := by decide +kernel
theorem eps_unit_right : ∀ i, i < 16 → eps i 0 = 1 := by decide +kernel
theorem eps_square     : ∀ i, i < 16 → 0 < i → eps i i = -1 := by decide +kernel
theorem eps_values     : ∀ i, i < 16 → ∀ j, j < 16 → eps i j = 1 ∨ eps i j = -1 := by decide +kernel

/-! ## 2. The two sign identities behind the Adjoint Theorem (Thm 2.3, eqs. after (adjoint_inner))

  e_i* (e_i e_j) = e_j  ⟹  ε_{i, i⊕j} = −ε_{ij}      (i ≥ 1)
  (e_i e_j) e_j* = e_i  ⟹  ε_{i⊕j, j} = −ε_{ij}      (j ≥ 1)
-/
theorem adjoint_left_identity :
    ∀ i, i < 16 → 0 < i → ∀ j, j < 16 → eps i (i ^^^ j) = - eps i j := by decide +kernel

theorem adjoint_right_identity :
    ∀ i, i < 16 → ∀ j, j < 16 → 0 < j → eps (i ^^^ j) j = - eps i j := by decide +kernel

/-! ## 3. ℤ-coefficient sedenions and the bilinear product

  A sedenion with integer coefficients is a function `Nat → Int` on indices 0..15.
  `epsT` is a literal copy of ε; `epsT_eq_eps` certifies it against the recursion, and
  it is what the product uses so that kernel evaluation stays cheap. -/

def epsT (i j : Nat) : Int :=
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
    | 8 => 1
    | 9 => 1
    | 10 => 1
    | 11 => 1
    | 12 => 1
    | 13 => 1
    | 14 => 1
    | 15 => 1
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
    | 8 => 1
    | 9 => -1
    | 10 => -1
    | 11 => 1
    | 12 => -1
    | 13 => 1
    | 14 => 1
    | 15 => -1
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
    | 8 => 1
    | 9 => 1
    | 10 => -1
    | 11 => -1
    | 12 => -1
    | 13 => -1
    | 14 => 1
    | 15 => 1
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
    | 8 => 1
    | 9 => -1
    | 10 => 1
    | 11 => -1
    | 12 => -1
    | 13 => 1
    | 14 => -1
    | 15 => 1
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
    | 8 => 1
    | 9 => 1
    | 10 => 1
    | 11 => 1
    | 12 => -1
    | 13 => -1
    | 14 => -1
    | 15 => -1
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
    | 8 => 1
    | 9 => -1
    | 10 => 1
    | 11 => -1
    | 12 => 1
    | 13 => -1
    | 14 => 1
    | 15 => -1
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
    | 8 => 1
    | 9 => -1
    | 10 => -1
    | 11 => 1
    | 12 => 1
    | 13 => -1
    | 14 => -1
    | 15 => 1
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
    | 8 => 1
    | 9 => 1
    | 10 => -1
    | 11 => -1
    | 12 => 1
    | 13 => 1
    | 14 => -1
    | 15 => -1
    | _ => 0
  | 8 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => -1
    | 3 => -1
    | 4 => -1
    | 5 => -1
    | 6 => -1
    | 7 => -1
    | 8 => -1
    | 9 => 1
    | 10 => 1
    | 11 => 1
    | 12 => 1
    | 13 => 1
    | 14 => 1
    | 15 => 1
    | _ => 0
  | 9 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => -1
    | 3 => 1
    | 4 => -1
    | 5 => 1
    | 6 => 1
    | 7 => -1
    | 8 => -1
    | 9 => -1
    | 10 => -1
    | 11 => 1
    | 12 => -1
    | 13 => 1
    | 14 => 1
    | 15 => -1
    | _ => 0
  | 10 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => 1
    | 3 => -1
    | 4 => -1
    | 5 => -1
    | 6 => 1
    | 7 => 1
    | 8 => -1
    | 9 => 1
    | 10 => -1
    | 11 => -1
    | 12 => -1
    | 13 => -1
    | 14 => 1
    | 15 => 1
    | _ => 0
  | 11 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => 1
    | 3 => 1
    | 4 => -1
    | 5 => 1
    | 6 => -1
    | 7 => 1
    | 8 => -1
    | 9 => -1
    | 10 => 1
    | 11 => -1
    | 12 => -1
    | 13 => 1
    | 14 => -1
    | 15 => 1
    | _ => 0
  | 12 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => 1
    | 3 => 1
    | 4 => 1
    | 5 => -1
    | 6 => -1
    | 7 => -1
    | 8 => -1
    | 9 => 1
    | 10 => 1
    | 11 => 1
    | 12 => -1
    | 13 => -1
    | 14 => -1
    | 15 => -1
    | _ => 0
  | 13 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => 1
    | 3 => -1
    | 4 => 1
    | 5 => 1
    | 6 => 1
    | 7 => -1
    | 8 => -1
    | 9 => -1
    | 10 => 1
    | 11 => -1
    | 12 => 1
    | 13 => -1
    | 14 => 1
    | 15 => -1
    | _ => 0
  | 14 => match j with
    | 0 => 1
    | 1 => -1
    | 2 => -1
    | 3 => 1
    | 4 => 1
    | 5 => -1
    | 6 => 1
    | 7 => 1
    | 8 => -1
    | 9 => -1
    | 10 => -1
    | 11 => 1
    | 12 => 1
    | 13 => -1
    | 14 => -1
    | 15 => 1
    | _ => 0
  | 15 => match j with
    | 0 => 1
    | 1 => 1
    | 2 => -1
    | 3 => -1
    | 4 => 1
    | 5 => 1
    | 6 => -1
    | 7 => 1
    | 8 => -1
    | 9 => 1
    | 10 => -1
    | 11 => -1
    | 12 => 1
    | 13 => 1
    | 14 => -1
    | 15 => -1
    | _ => 0
  | _ => 0

theorem epsT_eq_eps : ∀ i, i < 16 → ∀ j, j < 16 → epsT i j = eps i j := by decide +kernel

abbrev Vec := Nat → Int

def sum16 (f : Nat → Int) : Int :=
  f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 + f 11 + f 12 + f 13 + f 14 + f 15

def all16 (p : Nat → Bool) : Bool :=
  p 0 && p 1 && p 2 && p 3 && p 4 && p 5 && p 6 && p 7 && p 8 && p 9 && p 10 && p 11 && p 12 && p 13 && p 14 && p 15

def basis (i : Nat) : Vec := fun k => if k == i then 1 else 0
/-- c · e_m -/
def sbasis (c : Int) (m : Nat) : Vec := fun k => if k == m then c else 0
def add (u v : Vec) : Vec := fun k => u k + v k
def sub (u v : Vec) : Vec := fun k => u k - v k

/-- y_k = Σ_j ε_{(k⊕j), j} · u_{k⊕j} · v_j   (the L(w) / R(x) contraction of Sec. 2.1) -/
def mul (u v : Vec) : Vec := fun k => sum16 fun j => epsT (k ^^^ j) j * u (k ^^^ j) * v j

/-- extensional equality on indices 0..15 -/
def veq (u v : Vec) : Bool := all16 fun k => u k == v k
def isZero (u : Vec) : Bool := all16 fun k => u k == 0

/-- the product of basis units is a signed basis unit (eq. basis_xor) -/
theorem basis_mul : ∀ i, i < 16 → ∀ j, j < 16 → veq (mul (basis i) (basis j)) (sbasis (eps i j) (i ^^^ j)) := by
  decide +kernel

/-! ## 4. Explicit zero divisors (Sec. 2.3 / Thm 3.1 Case 2 / Thm 5.1)

  The sign of the second term is fixed by the Cayley–Dickson convention above. -/

/-- (e₁ + e₁₀)(e₅ + e₁₄) = 0 -/
theorem zero_divisor_1_10 : isZero (mul (add (basis 1) (basis 10)) (add (basis 5) (basis 14))) := by
  decide +kernel

/-- the example quoted in Theorem 3.1:  (e₂ + e₉)(e₄ + e₁₅) = 0, all four indices in H₉ -/
theorem zero_divisor_in_A9 : isZero (mul (add (basis 2) (basis 9)) (add (basis 4) (basis 15))) := by
  decide +kernel

/-- Four annihilators of u = e₁ + e₁₀ with distinct leading basis index (hence linearly
    independent): the constructive partners v = (c, (a c) b) of Thm 5.1 with a = e₁, b = e₂ and
    c ∈ {e₄,e₅,e₆,e₇}, a basis of W_{ab}^⊥.  So dim ker L(u) ≥ 4, as Thm 5.1 states. -/
def u_moreno : Vec := add (basis 1) (basis 10)

theorem kernel_witness_4 :
    isZero (mul u_moreno (sub (basis 4) (basis 15))) ∧
    isZero (mul u_moreno (add (basis 5) (basis 14))) ∧
    isZero (mul u_moreno (sub (basis 6) (basis 13))) ∧
    isZero (mul u_moreno (add (basis 7) (basis 12))) := by
  decide +kernel

end SedenionPG32
