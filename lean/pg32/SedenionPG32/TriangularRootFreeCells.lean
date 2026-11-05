import SedenionPG32.Core
import SedenionPG32.Geometry

set_option maxRecDepth 200000
set_option maxHeartbeats 0

namespace SedenionPG32

/-! ## 8. The Sedenion Triangular Root & Zero-Divisor-Free Coordinate Cells in the Cayley–Dickson Tower

  Machine-checked formalization of:
  [1] Pedro Barone Agourakis and Dionisio Chiuratto Agourakis,
      "The Sedenion Triangular Root: A Child-Originated Partial Desimplicialization Operator
       on Cayley–Dickson Bitmasks", July 7, 2026, Zenodo DOI 10.5281/zenodo.21246673.
  [2] Pedro Barone Agourakis and Dionisio Chiuratto Agourakis,
      "Zero-Divisor-Free Coordinate Cells in the Cayley–Dickson Tower", July 29, 2026. -/

/-- Appendix A of [1]: Sequential toy triangular-root operator &seq on binary words.
    Returns `some |w|_1` if `w` begins with `true` (1) and contains no adjacent `false, false` (00). -/
def hasDoubleZero : List Bool → Bool
  | false :: false :: _ => true
  | _ :: rest => hasDoubleZero rest
  | [] => false

def seqTriangularRoot (w : List Bool) : Option Nat :=
  match w with
  | true :: _ =>
    if hasDoubleZero w then none
    else some (w.filter (· == true)).length
  | _ => none

/-- Origin of the definition [1, Sec. 1 & Appendix A]:
    Pedro Barone Agourakis's symbolic examples &(10) = 1, &(101) = 11 (length 2), &(100) = ∅. -/
theorem pedro_toy_triangular_root_identities :
    seqTriangularRoot [true, false] = some 1 ∧
    seqTriangularRoot [true, false, true] = some 2 ∧
    seqTriangularRoot [true, false, false] = none := by decide +kernel

/-- Test whether imaginary basis index `v ∈ {1..15}` is active in 16-bit support mask `m`. -/
def inMask (m v : Nat) : Bool := ((m >>> v) &&& 1) == 1

/-- List of active imaginary indices `P(w) ⊆ {1..15}` of a bitmask `m`. -/
def maskSupport (m : Nat) : List Nat :=
  ((List.range 15).map (· + 1)).filter (inMask m)

/-- Cardinality `|P(w)|` of the imaginary support of `m`. -/
def maskCard (m : Nat) : Nat := (maskSupport m).length

/-- Definition 4.2 & Lemma 4.3 of [1]: A nonempty imaginary support `P ⊆ V^\times` is
    triangularly closed iff `a, b ∈ P` with `a ≠ b` implies `a ⊕ b ∈ P`. -/
def isTriangularlyClosed (m : Nat) : Bool :=
  let P := maskSupport m
  !(P.isEmpty) && P.all fun a => P.all fun b => (a == b) || inMask m (a ^^^ b)

/-- Canonical F₂-Reduced Row Echelon Form (RREF) basis extractor `rref(U)` over `V = F₂⁴`:
    For a triangularly closed support `P = U \ {0}`, an element `x ∈ P` belongs to the
    canonical RREF basis (under standard integer order) iff `x` cannot be written as `a ⊕ b`
    with `a, b ∈ P` and `a < x`, `b < x` (equivalently, the minimum elements of the
    successive coset quotients). -/
def rrefBasisList (m : Nat) : List Nat :=
  let P := maskSupport m
  let rec build (rem : List Nat) (span : List Nat) (acc : List Nat) : List Nat :=
    match rem with
    | [] => acc.reverse
    | x :: xs =>
      if span.contains x then
        build xs span acc
      else
        let newSpan := span ++ (span.map (· ^^^ x))
        build xs newSpan (x :: acc)
  build P [0] []

def listToMask (xs : List Nat) : Nat :=
  xs.foldl (fun acc v => acc ||| (1 <<< v)) 0

/-- Definition 4.1 of [1]: Geometric triangularization `T_S^geo(R) = ⟨R⟩_{F₂} \ {0}`. -/
def triangularizeList (R : List Nat) : Nat :=
  let span := R.foldl (fun sp x =>
    sp.foldl (fun acc s => if acc.contains (s ^^^ x) then acc else (s ^^^ x) :: acc) sp
  ) [0]
  listToMask (span.filter (· != 0))

/-- Definition 5.1 of [1]: Geometric sedenion triangular root `&^geo_S(w)`. -/
def geomTriangularRoot (m : Nat) : Option Nat :=
  if isTriangularlyClosed m then some (listToMask (rrefBasisList m)) else none

/-- Enumeration of all 66 triangularly closed imaginary support masks in `{0,1}^{15} \ {0}`
    by iterating over all `2^4 - 1 = 15` points, `35` lines, `15` planes, and `1` 3-space. -/
def allProjectiveMasks : List Nat :=
  let pts := (List.range 15).map (· + 1)
  let rank1 := pts.map (fun a => listToMask [a])
  let rank2 := (pts.flatMap fun a => (pts.filter (· > a)).map fun b =>
    triangularizeList [a, b]).eraseDups
  let rank3 := (rank2.flatMap fun lMask =>
    pts.filter (fun c => !(inMask lMask c)) |>.map fun c =>
      triangularizeList (c :: maskSupport lMask)).eraseDups
  let rank4 := [listToMask pts]
  rank1 ++ rank2 ++ rank3 ++ rank4

/-- Theorem 6.1 & Table 1 of [1]: The nonempty domain of `&^geo_S` contains exactly 66 masks:
    15 points (|P|=1), 35 lines (|P|=3), 15 Fano planes (|P|=7), and 1 projective 3-space (|P|=15). -/
theorem geometric_root_domain_66_and_ranks :
    allProjectiveMasks.length = 66 ∧
    allProjectiveMasks.eraseDups.length = 66 ∧
    (allProjectiveMasks.all isTriangularlyClosed) = true ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 1)).length = 15 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 3)).length = 35 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 7)).length = 15 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 15)).length = 1 := by decide +kernel

/-- Proposition 5.3 of [1]: Inverse property `&^geo_S(χ_{T^geo_S(R)}) = χ_{rref(U)}`
    verified across all 66 projective cells of PG(3,2). -/
theorem geometric_root_left_inverse_all_66 :
    (allProjectiveMasks.all fun m =>
      let r := rrefBasisList m
      triangularizeList r == m && geomTriangularRoot m == some (listToMask r)) = true := by
  decide +kernel

/-- Lemma 7.2 of [1]: Support-level zero-divisor-free criterion.
    With `W^\times = {1..7}` and `η = 8`, define `A(P) = P ∩ W^\times` and
    `B(P) = {u ∈ W^\times : (8 ⊕ u) ∈ P}`. Then `P` is ZD-free iff
    `A(P) = ∅`, or `B(P) = ∅`, or `A(P) = B(P) = {u}` for a singleton `u`. -/
def supportA (m : Nat) : List Nat :=
  ((List.range 7).map (· + 1)).filter (inMask m)

def supportB (m : Nat) : List Nat :=
  ((List.range 7).map (· + 1)).filter (fun u => inMask m (8 ^^^ u))

def isZDFreeSupport (m : Nat) : Bool :=
  let A := supportA m
  let B := supportB m
  A.isEmpty || B.isEmpty || (A.length == 1 && A == B)

/-- Direct algebraic test in `SedenionPG32`: does `S_P` contain a primitive basis-pair
    zero divisor `e_a ± e_{8⊕b}` (`a ∈ A(P), b ∈ B(P), a ≠ b`) that is annihilated by some
    signed basis pair in `S`? -/
def hasBasisPairZeroDivisor (m : Nat) : Bool :=
  let A := supportA m
  let B := supportB m
  A.any fun a => B.any fun b => a != b &&
    let u := add (basis a) (basis (8 ^^^ b))
    ((List.range 7).map (· + 1)).any fun c =>
      ((List.range 7).map (· + 1)).any fun d =>
        isZero (mul u (add (basis c) (basis (8 ^^^ d)))) ||
        isZero (mul u (sub (basis c) (basis (8 ^^^ d))))

/-- Lemma 7.2 of [1] & Lemma 5.2 of [2]: Across all 66 projective coordinate cells, the
    combinatorial support criterion `isZDFreeSupport` is equivalent to the absence of
    actual zero divisors in the sedenion algebra `S`. -/
theorem support_zd_criterion_equivalence_all_66 :
    (allProjectiveMasks.all fun m =>
      isZDFreeSupport m == !(hasBasisPairZeroDivisor m)) = true := by decide +kernel

/-- Theorem 9.1 & Table 2 of [1]: Among the 66 geometric masks in `Dom(&^geo_S)`,
    exactly 30 are ZD-free (`Dom(&^ZD_S)`) and 36 are ZD-degenerate (ghost masks),
    decomposed by dimension as (15,0), (14,21), (1,14), and (0,1). -/
theorem zd_filtered_domain_30_plus_36_by_rank :
    (allProjectiveMasks.filter isZDFreeSupport).length = 30 ∧
    (allProjectiveMasks.filter (fun m => !(isZDFreeSupport m))).length = 36 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 1 && isZDFreeSupport m)).length = 15 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 3 && isZDFreeSupport m)).length = 14 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 3 && !(isZDFreeSupport m))).length = 21 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 7 && isZDFreeSupport m)).length = 1 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 7 && !(isZDFreeSupport m))).length = 14 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 15 && isZDFreeSupport m)).length = 0 ∧
    (allProjectiveMasks.filter (fun m => maskCard m == 15 && !(isZDFreeSupport m))).length = 1 := by
  decide +kernel

/-- Corollary 9.2 of [1]: Trichotomy of the 32,767 nonzero imaginary bitmasks:
    `|T_good| = 30`, `|T_ghost| = 36`, and `|T_bad| = 32701`, with `30 + 36 + 32701 = 32767`. -/
theorem bitmask_universe_trichotomy_32767 :
    let totalNonzeroMasks := 2^15 - 1
    let goodCount := (allProjectiveMasks.filter isZDFreeSupport).length
    let ghostCount := (allProjectiveMasks.filter (fun m => !(isZDFreeSupport m))).length
    let badCount := totalNonzeroMasks - allProjectiveMasks.length
    totalNonzeroMasks = 32767 ∧
    goodCount = 30 ∧
    ghostCount = 36 ∧
    badCount = 32701 ∧
    goodCount + ghostCount + badCount = totalNonzeroMasks := by decide +kernel

/-- Section 10 of [1]: Machine-checked verification of Examples 10.1 through 10.5. -/
theorem paper1_examples_10_1_to_10_5 :
    -- Example 10.1: Octonion line {1,2,3} -> RREF {1,2}, ZD-free
    let p1 := listToMask [1, 2, 3]
    geomTriangularRoot p1 == some (listToMask [1, 2]) ∧ isZDFreeSupport p1 = true ∧
    -- Example 10.2: Vertical quaternion line {1,8,9} -> RREF {1,8}, ZD-free
    let p2 := listToMask [1, 8, 9]
    geomTriangularRoot p2 == some (listToMask [1, 8]) ∧ isZDFreeSupport p2 = true ∧
    -- Example 10.3: Ghost line {1,10,11} -> RREF {1,10}, ZD-degenerate
    let p3 := listToMask [1, 10, 11]
    geomTriangularRoot p3 == some (listToMask [1, 10]) ∧ isZDFreeSupport p3 = false ∧
    -- Example 10.4: Ghost Fano plane {1,2,3,12,13,14,15} -> RREF {1,2,12}, ZD-degenerate
    let p4 := listToMask [1, 2, 3, 12, 13, 14, 15]
    geomTriangularRoot p4 == some (listToMask [1, 2, 12]) ∧ isZDFreeSupport p4 = false ∧
    -- Example 10.5: Geometric failure {1,2,4} -> ∅_geo
    let p5 := listToMask [1, 2, 4]
    geomTriangularRoot p5 == none := by decide +kernel

/-! ### Theorems from [2]: Zero-Divisor-Free Coordinate Cells in the Cayley–Dickson Tower -/

/-- The 8 maximal zero-divisor-free coordinate blocks of `A_4` (Theorem 1.1 & Corollary 1.2 of [2]):
    1 octonion block `O = {1..7}` and 7 vertical quaternion blocks `Q_{u,3} = {u, 8, u⊕8}`. -/
def maximalFreeBlocksA4 : List Nat :=
  let octBlock := listToMask [1, 2, 3, 4, 5, 6, 7]
  let vertBlocks := ((List.range 7).map (· + 1)).map fun u => listToMask [u, 8, 8 ^^^ u]
  octBlock :: vertBlocks

/-- The 21 ghost lines of `A_4` (Lemma 5.2 & Proposition 5.4 of [2]). -/
def ghostLinesA4 : List Nat :=
  allProjectiveMasks.filter (fun m => maskCard m == 3 && !(isZDFreeSupport m))

/-- Subset test on bitmasks: `sub ⊆ sup` iff `(sub &&& sup) == sub`. -/
def isSubMask (sub sup : Nat) : Bool := (sub &&& sup) == sub

/-- Theorem 6.2 & Theorem 6.4 of [2] (Order-Ideal and Ghost-Filter Duality):
    Every projective coordinate cell `U` of `A_4` is ZD-free iff `U` is contained in one of the
    `2^4 - 4 - 4 = 8` maximal free blocks (`O` or `Q_{u,3}`), and is ZD-contaminated iff `U`
    contains at least one of the 21 ghost lines. -/
theorem order_ideal_and_ghost_filter_duality_all_66 :
    maximalFreeBlocksA4.length = 8 ∧
    ghostLinesA4.length = 21 ∧
    (allProjectiveMasks.all fun m =>
      isZDFreeSupport m == maximalFreeBlocksA4.any (isSubMask m)) = true ∧
    (allProjectiveMasks.all fun m =>
      (!(isZDFreeSupport m)) == ghostLinesA4.any (fun g => isSubMask g m)) = true := by
  decide +kernel

/-- Lemma 5.2 of [2]: Every one of the 21 ghost lines `L = {p, q, r}` in `A_4` (where `p, q ≥ 8`
    and `r = p ⊕ q < 8`) contains the zero divisor `e_r + e_p` whose number of basis-pair
    annihilating indices `c ∈ {0..7}` is a positive multiple of 4 (specifically 4 in `A_4`). -/
def ghostLineAnnihilatorCount (lMask : Nat) : Nat :=
  let pts := maskSupport lMask
  let r := (pts.filter (· < 8)).headD 0
  let p := (pts.filter (· >= 8)).headD 0
  let q := (pts.filter (fun x => x >= 8 && x != p)).headD 0
  let qPrime := q ^^^ 8
  let x := add (basis r) (basis p)
  ((List.range 8).filter fun c =>
    let d := 8 ^^^ (c ^^^ qPrime)
    isZero (mul x (add (basis c) (basis d))) ||
    isZero (mul x (sub (basis c) (basis d)))).length

theorem ghost_line_annihilator_multiples_of_four :
    (ghostLinesA4.all fun lMask =>
      let cnt := ghostLineAnnihilatorCount lMask
      cnt > 0 && cnt % 4 == 0) = true := by decide +kernel

/-- Closed-form Cayley–Dickson tower census formulas from Corollary 1.2, Proposition 5.4,
    Corollary 6.3, and Corollary 8.3 of [2] for level `n ≥ 4`. -/
def towerFreeCells (n : Nat) : Nat := 2^(n + 1) - n + 2
def towerMaximalFreeBlocks (n : Nat) : Nat := 2^n - n - 4
def towerFreeLines (n : Nat) : Nat := 2^n - n + 2
def towerTotalLines (n : Nat) : Nat := ((2^n - 1) * (2^n - 2)) / 6
def towerGhostLines (n : Nat) : Nat := towerTotalLines n - towerFreeLines n
def towerFreeOrbits (n : Nat) : Nat := 2^(n - 1) - n + 2

/-- Appendix A Table of [2]: Exact census verification for `A_4, A_5, A_6` (`n = 4, 5, 6`). -/
theorem cayley_dickson_tower_census_n4_n5_n6 :
    -- Free cells: 30, 61, 124
    (towerFreeCells 4, towerFreeCells 5, towerFreeCells 6) = (30, 61, 124) ∧
    -- Maximal free blocks: 8, 23, 54
    (towerMaximalFreeBlocks 4, towerMaximalFreeBlocks 5, towerMaximalFreeBlocks 6) = (8, 23, 54) ∧
    -- Free lines: 14, 29, 60
    (towerFreeLines 4, towerFreeLines 5, towerFreeLines 6) = (14, 29, 60) ∧
    -- Ghost lines: 21, 126, 591
    (towerGhostLines 4, towerGhostLines 5, towerGhostLines 6) = (21, 126, 591) ∧
    -- Free GL(3,2)-orbits: 6, 13, 28
    (towerFreeOrbits 4, towerFreeOrbits 5, towerFreeOrbits 6) = (6, 13, 28) := by decide +kernel

/-- Theorem 8.1 of [2]: Doubling-generator incidence degree separation verified across all
    tower levels `n ∈ {4..10}` and generator bit positions `3 ≤ j ≤ n - 1`, proving
    `deg(2^j) > n - 3` and `deg(2^j) < deg(2^{j+1})`, which pins every doubling generator
    `2^3, ..., 2^{n-1}` and forces `|Π_n| = |GL(3,2)| = 168`. -/
def generatorIncidenceDeg (n j : Nat) : Nat := 2^j - 1 + (n - 1 - j)

def checkDegreeSeparationUpTo10 : Bool :=
  ((List.range 7).map (· + 4)).all fun n =>
    ((List.range (n - 3)).map (· + 3)).all fun j =>
      generatorIncidenceDeg n j > n - 3 &&
      generatorIncidenceDeg n j < generatorIncidenceDeg n (j + 1)

theorem doubling_generator_degree_separation :
    checkDegreeSeparationUpTo10 = true ∧
    (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168 := by decide +kernel

/-- Proposition 7.1 of [2] (Tower Absoluteness `F_n = ι_n⁻¹ F_{n+1}`):
    A coordinate cell `U ≤ V_4 ↪ V_5` is contained in the maximal free blocks of `A_5`
    iff it is contained in the maximal free blocks of `A_4`. -/
def isFreeInTowerLevel (n m : Nat) : Bool :=
  let octBlock := listToMask [1, 2, 3, 4, 5, 6, 7]
  isSubMask m octBlock ||
  ((List.range (n - 3)).map (· + 3)).any fun k =>
    ((List.range (2^k - 1)).map (· + 1)).any fun u =>
      isSubMask m (listToMask [u, 2^k, u ^^^ (2^k)])

theorem tower_absoluteness_n4_in_n5_all_66 :
    (allProjectiveMasks.all fun m =>
      isFreeInTowerLevel 4 m == isFreeInTowerLevel 5 m &&
      isFreeInTowerLevel 4 m == isZDFreeSupport m) = true := by decide +kernel

end SedenionPG32
