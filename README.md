# Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning

[![DOI: Main Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.23588274.svg)](https://doi.org/10.5281/zenodo.23588274)
[![DOI: GNSS Anti-Jamming](https://zenodo.org/badge/DOI/10.5281/zenodo.23603006.svg)](https://doi.org/10.5281/zenodo.23603006)

Reference implementation, Lean 4 formalisations (zero `sorry`), and benchmark suite for:

1. **Dionisio Chiuratto Agourakis** (2026). *Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning: Algebraic Foundations, Machine-Checked Theorems, and Dual-Track Validation*. Zenodo. [`10.5281/zenodo.23588274`](https://doi.org/10.5281/zenodo.23588274)
2. **Dionisio Chiuratto Agourakis** (2026). *Sedenion PG(3,2) Zero-Divisor Beamforming for GNSS Anti-Jamming and Anti-Spoofing: Matrix-Inversion-Free Null Steering, Associative Quaternion Phase Hinges, and Neural CRPA Validation*. Zenodo. [`10.5281/zenodo.23603006`](https://doi.org/10.5281/zenodo.23603006)

## Layout

```
papers/            manuscripts and proposals (LaTeX + PDF)
sedenion_pg32/     Python package: Cayley–Dickson tables, adjoint autograd, PG(3,2) router,
                   Moreno projection, constructive nullspace partner, Triton kernel + PyTorch mirror
tests/             pytest suite for the package
benchmarks/        algebraic_baselines.py (Table 2), track_a/ (Yang–Mills BCJ), track_b/ (deontic
                   multi-hop pruning, No-FFN ablation), ext/ (ProofWriter, ListOps), gps_jamming/
                   (CRPA null steering, phase-2 defences, dual-pol excision, Q1.14 RTL)
lean/pg32/         finite development: Lean 4.15 core only, every theorem by decide +kernel, no axioms
lean/real/         real-analytic development: Lean 4.15 + Mathlib v4.15.0
                   SedenionReal.lean   — Theorem 5.1 nullspace, Hurwitz composition, CRPA theorems
                   SedenionAlt/Rank/Dim — alternativity, kernel characterisation, rank L(u) = 12 (rank–nullity)
                   SedenionRobust.lean — leakage bounds, 8⊕4⊕4 transceiver, dual-pol RHCP/LHCP excision
reports/           analysis notes that back numbers in the papers
zenodo/            deposit metadata, package and upload script for the dual-pol supplement
```

| Document | Source | PDF |
|---|---|---|
| Main manuscript | [`papers/Agourakis_2026_Sedenion_PG32.tex`](papers/Agourakis_2026_Sedenion_PG32.tex) | [PDF](papers/Agourakis_2026_Sedenion_PG32.pdf) |
| GNSS anti-jamming companion paper | [`papers/Agourakis_2026_Sedenion_GNSS_AntiJamming.tex`](papers/Agourakis_2026_Sedenion_GNSS_AntiJamming.tex) | [PDF](papers/Agourakis_2026_Sedenion_GNSS_AntiJamming.pdf) |
| DARPA / ONR Global BAA white paper | [`papers/DARPA_ONR_WhitePaper_Sedenion_GNSS.tex`](papers/DARPA_ONR_WhitePaper_Sedenion_GNSS.tex) | [PDF](papers/DARPA_ONR_WhitePaper_Sedenion_GNSS.pdf) |
| DARPA-PS-26-09 STO abstract | [`papers/DARPA_PS_26_09_Abstract_Sedenion_GNSS.tex`](papers/DARPA_PS_26_09_Abstract_Sedenion_GNSS.tex) | [PDF](papers/DARPA_PS_26_09_Abstract_Sedenion_GNSS.pdf) |
| Dual-pol excision report (Lean + simulation) | [`reports/DUALPOL_PIVOT12_REPORT.md`](reports/DUALPOL_PIVOT12_REPORT.md) | |

## Quick start

```bash
pip install torch pytest onnx onnxruntime
PYTHONPATH=. python -m pytest -q tests                 # algebraic & geometric unit tests (incl. 0.0 ULP sign-bit XOR)
python benchmarks/algebraic_baselines.py               # Table 2: CLR, PSI, rank L(u) = 12 (CPU, < 1 min)
python benchmarks/track_a/train.py                     # Table 2: Track A 4-gluon Yang–Mills BCJ & collinear pole
python benchmarks/track_b/train.py                     # Table 3: Track B multi-hop contradiction pruning & No-FFN ablation
python benchmarks/ext/proofwriter.py                   # Table 4: ProofWriter OOD depth & Birds-Electricity
python benchmarks/ext/listops.py                       # Table 4: Long-Range Arena ListOps OOD tree reduction
python benchmarks/gps_jamming/run_gps_bench.py         # companion paper: +80 dB JSR CRPA null steering, RTK hinges, anti-spoofing
python benchmarks/gps_jamming/run_phase2_defense.py    # mutual-coupling / LSMI defences
python benchmarks/gps_jamming/run_dualpol_pivot12.py out.json   # dual-pol RHCP/LHCP excision, S-matrix, CRLB (numpy only)
```

Benchmark scripts write their `results_*.json` next to themselves; the committed JSON files are the runs reported in the papers.

## Lean 4 verification

```bash
# finite PG(3,2) development: 8+7 Fano bifurcation, 672/2520 associator census, BCJ stratum theorems
cd lean/pg32 && lake build SedenionPG32 && LEAN_PATH=.lake/build/lib lean Axioms.lean     # propext at most; several "does not depend on any axioms"

# real-analytic development (Mathlib): nullspace, CRPA gains, leakage bounds, dual-pol excision
cd lean/real && lake exe cache get && lake build && lake env lean Axioms.lean            # propext, Classical.choice, Quot.sound only
```

`lean/real/Axioms.lean` audits every theorem the papers cite from the real development, including the
§5.5 rank–nullity chain (`SedenionAlt` → `SedenionRank` → `SedenionDim`: `square_identity`,
`adjoint_identity`, `kernel_char`, `ker_Lu_eq`, `finrank_ker_Lu = 4`, `finrank_range_Lu = 12`).

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

@misc{agourakis2026gnss,
  author       = {Agourakis, Dionisio Chiuratto},
  title        = {{Sedenion PG(3,2) Zero-Divisor Beamforming for GNSS Anti-Jamming and Anti-Spoofing: Matrix-Inversion-Free Null Steering, Associative Quaternion Phase Hinges, and Neural CRPA Validation}},
  year         = {2026},
  month        = oct,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23603006},
  url          = {https://doi.org/10.5281/zenodo.23603006}
}
```
