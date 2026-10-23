# Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23588274.svg)](https://doi.org/10.5281/zenodo.23588274)

Official repository, Lean 4 formalizations (`121` machine-checked theorems, zero `sorry`), and PyTorch benchmark suite for:

> **Dionisio Chiuratto Agourakis** (2026). *Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning: Algebraic Foundations, Machine-Checked Theorems, and Dual-Track Validation*. Zenodo. [`https://doi.org/10.5281/zenodo.23588274`](https://doi.org/10.5281/zenodo.23588274)

- **Manuscript PDF:** [`Agourakis_2026_Sedenion_PG32.pdf`](./Agourakis_2026_Sedenion_PG32.pdf)
- **LaTeX Source:** [`Agourakis_2026_Sedenion_PG32.tex`](./Agourakis_2026_Sedenion_PG32.tex)
- **Self-Contained Release Archive:** [`bench/sedenion_pg32.zip`](./bench/sedenion_pg32.zip)

---

## Quick Start & Reproducibility

```bash
PYTHONPATH=bench/pkg python -m pytest -q bench/pkg/tests     # 25 algebraic & geometric unit tests (incl. 0.0 ULP IEEE-754 sign-bit XOR)
python bench/pkg/benchmarks/algebraic_baselines.py           # Table 2: Algebraic baselines (CLR, PSI, rank L(u) = 12)
python bench/track_a/train.py                                # Table 2: Track A 4-gluon Yang-Mills BCJ & collinear pole benchmark
python bench/gate3/train.py                                  # Table 3: Track B multi-hop contradiction pruning & No-FFN ablation
python bench/ext/proofwriter.py                              # Table 4: AllenAI ProofWriter (ACL 2021) OOD depth & Birds-Electricity
python bench/ext/listops.py                                  # Table 4: Long-Range Arena ListOps (ICLR 2021) OOD tree reduction
```

### Lean 4 Formal Verification (`bench/pkg/lean/` & `bench/pkg/lean-real/`)

```bash
# 1. Finite PG(3,2), 8+7 Fano bifurcation, 672/2520 associator census, and BCJ stratum theorems (decide +kernel, zero axioms)
cd bench/pkg/lean && lake build SedenionPG32 && LEAN_PATH=.lake/build/lib lean Axioms.lean

# 2. Real-analytic Theorem 5.1 (nullspace) and rank L(u) = 12 over ℝ (Mathlib)
cd bench/pkg/lean-real && lake exe cache get && lake build SedenionDim && lake env lean AxiomsReal.lean
```

---

## Citation

```bibtex
@misc{agourakis2026sedenion,
  author       = {Agourakis, Dionisio Chiuratto},
  title        = {{Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning: Algebraic Foundations, Machine-Checked Theorems, and Dual-Track Validation}},
  year         = {2026},
  month        = oct,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23588274},
  url          = {https://doi.org/10.5281/zenodo.23588274}
}
```
