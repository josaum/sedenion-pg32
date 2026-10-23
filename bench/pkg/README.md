# sedenion_pg32

Reference implementation, Lean 4 formalization, and Track A / Track B benchmark suite for *Sedenion PG(3,2) Neural Topologies* (`Agourakis_2026_Sedenion_PG32.pdf`).

```bash
pip install torch pytest onnx onnxruntime
PYTHONPATH=. python -m pytest -q tests                        # 25 algebraic & geometric unit tests
python benchmarks/track_a/train.py                             # Track A: Yang-Mills 4-gluon BCJ & collinear poles (Table 2)
python benchmarks/track_b/train.py softmax,bilinear,octonion,clifford,sedenion_plain,sedenion,sedenion_noffn 0,1 12 2 0.0 _d2_aux0
```

## Benchmarks (`benchmarks/`)

- **`benchmarks/track_a/` (Track A — Particle Scattering Amplitudes & $G_2$ Symmetries):**
  Evaluates parameter-matched neural surrogates (`Dense-R16`, `Quat-H4`, `Oct-O2`, `Clifford-Cl04`, `Sed-Free`, `Sed-PG32-Moreno`, and `Sed-PG32-Moreno-NoFFN`) on 4-gluon Yang–Mills tree scattering amplitudes ($s + t + u = 0$), measuring the BCJ Jacobi Violation Residual $\mathcal{R}_{\mathrm{BCJ}}$ and out-of-distribution Collinear Pole Relative Error $\mathcal{E}_{\mathrm{pole}}$ ($|s_{ij}| < 10^{-4}$).
- **`benchmarks/track_b/` (Track B — Deontic Non-Monotonic Defeasible Logic & No-FFN Paradigm):**
  Evaluates multi-hop recursive exception-override chains (`d <= 2` in-distribution, `d = 3, 4` out-of-distribution depth generalization, `aux = 0.0` pure binary cross-entropy) across standard and No-FFN (`arXiv:2601.06793`) Transformer backbones.
- **`benchmarks/algebraic_baselines.py`:** Analytic algebraic baseline checks.

## Lean 4 formalisation (`lean/`)

Core Lean only (v4.15, no Mathlib). Every theorem is closed by `decide +kernel`; `#print axioms` shows empty axioms (`does not depend on any axioms`).

```bash
cd lean && lake build SedenionPG32          # builds Core, Geometry, Census
LEAN_PATH=.lake/build/lib lean Axioms.lean  # axiom audit
```

Modules:
- `Core` — structure constants, adjoint sign identities, basis product, explicit zero divisors, 4 kernel witnesses (`moreno_4d_nullspace_witnesses`).
- `Geometry` — $\mathrm{PG}(3,2)$ hyperplanes, polar line, $8 + 7$ Fano bifurcation (`fano_bifurcation_8_plus_7`), cyclic associator invariance on $\mathcal{F}_{\mathrm{alt}}$ (`falt_cyclic_assoc_invariant`), and adjacent-stratum $L_0 \sqcup Q_1$ BCJ pole cancellation on $\mathcal{F}_{\mathrm{ZD}}$ (`fzd_stratum_bcj_pole_cancellation`, Theorem 5.3).
- `Census` — internal annihilating pairs in every zero-divisor plane, collinear associativity, $672 / 2520$ associator census (`associator_census_672_2520`).

## Theorem 5.1 and $\operatorname{rank} L(u) = 12$ over $\mathbb{R}$ (`lean-real/`, Mathlib)

```bash
cd lean-real && lake exe cache get && lake build SedenionDim
lake env lean AxiomsReal.lean                                  # propext, Classical.choice, Quot.sound only
```

- `SedenionReal.lean` — `nullspace` (Theorem 5.1 verbatim), `composition` (Hurwitz), `partner_norm`, `partner_linear`, `basis_mul_sign`.
- `SedenionAlt.lean` — alternativity, adjoint identities, $R = (a \cdot) b$ and $\Pi$; `adjoint_identity` ($\Pi R^\top R \Pi = \Pi$).
- `SedenionRank.lean` — `square_identity` ($u(u x) = -2x + 2(\Pi R^\top d, R\Pi c)$) and `kernel_char`: $u \cdot x = 0 \implies x = (c,(ac)b), c \in W^\perp$.
- `SedenionDim.lean` — `ker_Lu_eq`, `finrank_ker_Lu = 4`, `finrank_range_Lu = 12` via Mathlib rank–nullity.
